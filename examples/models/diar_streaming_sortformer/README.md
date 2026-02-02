# Streaming Sortformer diarization (ExecuTorch C++)

This example exports `nvidia/diar_streaming_sortformer_4spk-v2.1` to an ExecuTorch `.pte` (portable ops only),
then runs end-to-end **streaming-style** diarization in C++:
- audio is fed chunk-by-chunk into the exported `preprocessor`
- features are buffered incrementally
- `model_step` is run whenever enough right-context audio has arrived
- diarization segments are printed as they are committed

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
- This runner implements a simplified cache update (keeps the most recent cache frames) and does not implement NeMo's speaker-cache compression logic.
- `--audio_chunk_ms` controls how much audio is fed into the ExecuTorch preprocessor per step to simulate end-to-end streaming. Smaller values increase overhead; values below 100ms are rejected.
- `--streaming_output` prints segments as they are committed (default: true). `--final_summary` prints a final sorted summary (default: false).
