# Streaming Sortformer diarization (ExecuTorch C++)

This example exports `nvidia/diar_streaming_sortformer_4spk-v2.1` to an ExecuTorch `.pte` (portable ops only),
then runs offline **streaming-style** diarization in C++ (chunk-by-chunk) using `model_step`.

## Export (portable ops only)

From `executorch/`:

```bash
python examples/models/diar_streaming_sortformer/export_diar_streaming_sortformer.py \
  --output-dir ./sortformer_diar_exports
```

Artifacts:
- `./sortformer_diar_exports/model.pte`

## Build + run the C++ runner

This uses the same pattern as `examples/models/parakeet`.

```bash
# Build ExecuTorch + this runner
make diar-streaming-sortformer-cpu

./cmake-out/examples/models/diar_streaming_sortformer/diar_streaming_sortformer_runner \
  --model_path ./sortformer_diar_exports/model.pte \
  --audio_path /path/to/mono_16khz.wav \
  --threshold 0.5 \
  --audio_chunk_ms 100
```

Notes:
- The WAV loader expects **mono** audio and does **not** resample.
- This runner implements a simplified cache update (keeps the most recent cache frames) and does not implement
  NeMo's speaker-cache compression logic.
- `--audio_chunk_ms` controls how much audio is fed into the ExecuTorch preprocessor per step to simulate
  end-to-end streaming. Smaller values increase overhead; values below 100ms are rejected.


## Notes about difference between simplified cache and NeMo:
What “cache update is simplified” means (in our C++ runner)

- We only keep embedding buffers fifo and spkcache and update them as plain “recent history” arrays. See examples/models/
  diar_streaming_sortformer/main.cpp:277 and the update block at examples/models/diar_streaming_sortformer/main.cpp:384.
- When a new chunk arrives, we append chunk_embs into fifo. If fifo would overflow, we “pop” the oldest frames and append them
  into spkcache (and cap spkcache by just dropping the oldest frames). See examples/models/diar_streaming_sortformer/main.cpp:389.
- We do not track the extra streaming state NeMo uses to manage / compress caches:
    - no spkcache_preds / fifo_preds (posteriors aligned to cached frames),
    - no mean_sil_emb / n_sil_frames (silence profile),
    - no speaker-cache compression based on per-speaker importance scores.
- Practical impact: spkcache becomes “whatever frames happened most recently”, not “a balanced, speaker-representative memory”.
  This usually hurts long-form diarization and speaker re-entry stability.

What NeMo does (the missing pieces)
NeMo’s real streaming update is in /Users/matt/Workspace/NeMo/nemo/collections/asr/modules/sortformer_modules.py:395:

- It maintains a richer streaming state: spkcache, spkcache_lengths, spkcache_preds, fifo, fifo_lengths, fifo_preds, plus
  mean_sil_emb/n_sil_frames (init_streaming_state is at /Users/matt/Workspace/NeMo/nemo/collections/asr/modules/
  sortformer_modules.py:360).
- Every step it refreshes fifo_preds from the newly computed preds for the [spkcache + fifo + chunk] sequence. (streaming_update
  shows the slice logic clearly at /Users/matt/Workspace/NeMo/nemo/collections/asr/modules/sortformer_modules.py:562.)
- When FIFO overflows, it:
    - pops frames from FIFO → appends them into the speaker cache,
    - updates the silence profile based on popped-frame posteriors (_get_silence_profile at /Users/matt/Workspace/NeMo/nemo/
      collections/asr/modules/sortformer_modules.py:636),
    - and if speaker cache is too large, it runs _compress_spkcache to select a fixed-size set of “important” frames per speaker
      (uses log-score, thresholds, boosts, forced silence frames, topk/sort/gather) at /Users/matt/Workspace/NeMo/nemo/
      collections/asr/modules/sortformer_modules.py:838.

Next steps to match NeMo (portable-only ExecuTorch inference)

1. Export more info from model_step so C++ can implement NeMo’s cache logic:

- Add at least fifo_preds (posteriors for the existing FIFO frames, padded to fifo_len) as an additional output. Today we only
  export chunk_preds/chunk_embs/chunk_pred_len (examples/models/diar_streaming_sortformer/
  export_diar_streaming_sortformer.py:169).
- Potentially also export a few more constants (as constant_methods) used by _compress_spkcache: sil_threshold,
  pred_score_threshold, spkcache_sil_frames_per_spk, scores_boost_latest, strong_boost_rate, weak_boost_rate, min_pos_scores_rate,
  etc.

2. Track the full NeMo streaming state in C++

- Add buffers for fifo_preds and spkcache_preds (and lengths), plus mean_sil_emb and n_sil_frames.

3. Implement NeMo’s cache update + compression in C++

- Port streaming_update_async (or the simpler streaming_update) logic for batch=1, including:
    - the exact pop-out length rules,
    - silence profile update (_get_silence_profile),
    - speaker-cache compression (_compress_spkcache and helpers).
- This is the main reason we left cache update in C++: _compress_spkcache depends on ops like topk/sort that are often painful to
  guarantee in “portable kernels only” graphs.

4. Validation

- Write a small “step-by-step parity” harness: run NeMo’s forward_streaming_step() and your C++ runner on the same audio/chunking
  params and compare per-step chunk_preds and final accumulated posteriors (before any VAD postprocessing).

If you want, I can outline exactly what extra tensors to output from model_step (names + fixed shapes) to support a faithful C++
port of streaming_update_async without re-running the whole model twice per step.
