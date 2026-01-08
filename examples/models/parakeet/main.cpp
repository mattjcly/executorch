/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/wav_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>

DEFINE_string(model_path, "parakeet.pte", "Path to Parakeet model (.pte).");
DEFINE_string(audio_path, "", "Path to input audio file (.wav).");
DEFINE_string(
    tokenizer_path,
    "tokenizer.json",
    "Path to SentencePiece tokenizer model file.");
DEFINE_string(
    data_path,
    "",
    "Path to data file (.ptd) for delegate data (optional, required for CUDA).");
DEFINE_bool(
    timestamps,
    false,
    "Output subword-, word-, and segment-level timestamps (requires model metadata).");

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

namespace {

// TDT duration values
const std::vector<int> DURATIONS = {0, 1, 2, 3, 4};
// NeMo: TDT maps a duration-class argmax -> "skip" (advance) in encoder frames.
// - Viable duration choices come from loss/config into decoding_cfg.durations:
//   https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/models/rnnt_models.py#L230-L238
// - Greedy TDT decoding uses: skip = self.durations[d_k]
//   https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py#L2665-L2669
// Divergence: we hardcode {0,1,2,3,4} here (matches current Parakeet TDT
// export). If the exported model's duration set changes, this must be updated
// to match or timestamps/decoding will drift.

struct TokenTimestamp {
  int64_t id;
  int64_t start_offset; // encoder frame index
  int64_t end_offset; // encoder frame index
};
// NeMo: TDT timing is represented on the Hypothesis as two parallel lists:
// - `Hypothesis.timestamp` holds the encoder frame index where each non-blank
//   token was emitted.
// - `Hypothesis.token_duration` holds the predicted duration/skip for that token.
// These are later converted to `{start_offset, end_offset}` via
// RNNTDecoding._compute_offsets_tdt().
// - Where NeMo records `timestamp` + `token_duration` during greedy TDT decode:
//   https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py#L2604-L2693
// - Where NeMo converts them into offsets:
//   https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/submodules/rnnt_decoding.py#L1128-L1156
// Divergence: this ExecuTorch example stores `{start_offset,end_offset}` directly
// as it decodes. We don't preserve NeMo's intermediate `timestep` list.

struct SubwordTimestamp {
  std::string text;
  int64_t start_offset;
  int64_t end_offset;
  double start_sec;
  double end_sec;
};

struct WordTimestamp {
  std::string text;
  int64_t start_offset;
  int64_t end_offset;
  double start_sec;
  double end_sec;
};
// NeMo: word-level timestamps are built from per-token offsets with
// `get_words_offsets()` and then converted from offsets->seconds by
// `process_timestamp_outputs()`.
// - Word grouping: https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/utils/timestamp_utils.py#L54-L224
// - Offsets->seconds: https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/utils/timestamp_utils.py#L428-L479
// Note: NeMo's `timestamp['char']` for Parakeet is per-subword token offsets
// (not true per-character).

struct SegmentTimestamp {
  std::string text;
  int64_t start_offset;
  int64_t end_offset;
  double start_sec;
  double end_sec;
};
// NeMo: segment-level timestamps are built from word offsets with
// `get_segment_offsets()`.
//   https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/utils/timestamp_utils.py#L227-L327
// Divergence: we only segment on terminal punctuation (.,!,?) and do not
// implement NeMo's optional `segment_gap_threshold` behavior.

bool is_ascii_punctuation_only(const std::string& s) {
  // NeMo: TDT punctuation timestamp refinement is applied when a punctuation
  // token appears long after the previous token; NeMo "pins" punctuation to the
  // previous token's end offset.
  //   https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/submodules/rnnt_decoding.py#L1169-L1189
  // Divergence: NeMo checks membership in a model-specific `supported_punctuation`
  // set (can include non-ASCII). Here we approximate by checking ASCII
  // `std::ispunct()` on bytes.
  if (s.empty()) {
    return false;
  }
  for (unsigned char ch : s) {
    if (!std::ispunct(ch)) {
      return false;
    }
  }
  return true;
}

size_t ltrim_ascii_whitespace(const std::string& s) {
  // NeMo: word boundaries for BPE/WPE are detected via tokenizer-type-specific
  // logic in `get_words_offsets()` (word delimiter char, special markers, etc).
  //   https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/utils/timestamp_utils.py#L79-L99
  // Divergence: we treat leading *ASCII whitespace* in the decoded piece as the
  // only word boundary signal.
  size_t i = 0;
  while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) {
    i++;
  }
  return i;
}

std::vector<SubwordTimestamp> tokens_to_subword_timestamps(
    const std::vector<TokenTimestamp>& tokens,
    tokenizers::Tokenizer* tokenizer,
    double seconds_per_encoder_frame) {
  // NeMo reference: TDT per-token "char" timestamps are computed in
  // `compute_rnnt_timestamps()` via `_compute_offsets_tdt()` and
  // `_refine_timestamps_tdt()`:
  //   https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/submodules/rnnt_decoding.py#L991
  // NeMo: "char" timestamps for Parakeet-TDT correspond to per-subword token
  // offsets, with a TDT punctuation refinement step.
  std::vector<SubwordTimestamp> subwords;
  if (!tokenizer) {
    return subwords;
  }

  const uint64_t bos_token = tokenizer->bos_tok();
  int64_t prev_end_offset = 0;
  bool has_prev_end_offset = false;

  for (const auto& token_ts : tokens) {
    uint64_t token = static_cast<uint64_t>(token_ts.id);
    auto decode_result = tokenizer->decode(bos_token, token);

    std::string piece = decode_result.ok() ? decode_result.get() : std::string();

    TokenTimestamp adjusted = token_ts;
    size_t non_ws = ltrim_ascii_whitespace(piece);
    std::string trimmed_piece = piece.substr(non_ws);

    const bool is_punct = is_ascii_punctuation_only(trimmed_piece);
    if (is_punct && has_prev_end_offset) {
      adjusted.start_offset = prev_end_offset;
      adjusted.end_offset = prev_end_offset;
    }

    double start_sec = -1.0;
    double end_sec = -1.0;
    if (seconds_per_encoder_frame > 0.0) {
      start_sec = seconds_per_encoder_frame * adjusted.start_offset;
      end_sec = seconds_per_encoder_frame * adjusted.end_offset;
    }

    subwords.push_back(SubwordTimestamp{
        piece, adjusted.start_offset, adjusted.end_offset, start_sec, end_sec});

    prev_end_offset = adjusted.end_offset;
    has_prev_end_offset = true;
  }

  return subwords;
}

std::vector<WordTimestamp> tokens_to_word_timestamps(
    const std::vector<TokenTimestamp>& tokens,
    tokenizers::Tokenizer* tokenizer,
    double seconds_per_encoder_frame) {
  // NeMo reference for word grouping (subword/char offsets -> word offsets):
  //   https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/utils/timestamp_utils.py#L54-L224
  //
  // Divergences from NeMo:
  // - NeMo builds words from decoded token offsets (and handles tokenizer types);
  //   here we build words by incrementally decoding each token and using leading
  //   ASCII whitespace as the boundary.
  // - NeMo returns `Hypothesis.timestamp['char']` in addition to word/segment;
  //   this example also emits per-subword timestamps.
  std::vector<WordTimestamp> words;
  if (!tokenizer || tokens.empty()) {
    return words;
  }

  uint64_t prev_token = 0;

  std::string current_word;
  int64_t current_start_offset = 0;
  int64_t current_end_offset = 0;
  int64_t prev_end_offset = 0;
  bool has_prev_end_offset = false;

  auto emit_word = [&]() {
    if (current_word.empty()) {
      return;
    }
    double start_sec = -1.0;
    double end_sec = -1.0;
    if (seconds_per_encoder_frame > 0.0) {
      start_sec = seconds_per_encoder_frame * current_start_offset;
      end_sec = seconds_per_encoder_frame * current_end_offset;
    }
    words.push_back(WordTimestamp{
        current_word, current_start_offset, current_end_offset, start_sec, end_sec});
    current_word.clear();
  };

  for (const auto& token_ts : tokens) {
    uint64_t token = static_cast<uint64_t>(token_ts.id);
    auto decode_result = tokenizer->decode(prev_token, token);
    prev_token = token;

    std::string piece = decode_result.ok() ? decode_result.get() : std::string();
    size_t non_ws = ltrim_ascii_whitespace(piece);
    bool had_leading_ws = non_ws > 0;
    std::string trimmed_piece = piece.substr(non_ws);

    if (trimmed_piece.empty()) {
      continue;
    }

    TokenTimestamp adjusted = token_ts;
    const bool is_punct = is_ascii_punctuation_only(trimmed_piece);
    if (is_punct && has_prev_end_offset) {
      // TDT can sometimes emit punctuation long after the preceding word. Pin
      // punctuation timing to the previous token end.
      // NeMo: RNNTDecoding._refine_timestamps_tdt() applies the same correction:
      //   https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/submodules/rnnt_decoding.py#L1169-L1189
      // Divergence: NeMo consults `supported_punctuation` from the model; here we
      // approximate punctuation detection (ASCII-only) via `is_ascii_punctuation_only()`.
      adjusted.start_offset = prev_end_offset;
      adjusted.end_offset = prev_end_offset;
    }

    if (current_word.empty()) {
      current_word = trimmed_piece;
      current_start_offset = adjusted.start_offset;
      current_end_offset = adjusted.end_offset;
    } else if (had_leading_ws && !is_punct) {
      // NeMo: `get_words_offsets()` decides when a new word starts using
      // tokenizer-aware rules (delimiter markers, WPE "##" prefixes, etc):
      //   https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/utils/timestamp_utils.py#L79-L99
      // Divergence: our boundary rule is strictly "decoded piece had leading
      // ASCII whitespace and is not punctuation".
      emit_word();
      current_word = trimmed_piece;
      current_start_offset = adjusted.start_offset;
      current_end_offset = adjusted.end_offset;
    } else {
      current_word += trimmed_piece;
      current_end_offset = adjusted.end_offset;
    }

    prev_end_offset = adjusted.end_offset;
    has_prev_end_offset = true;
  }

  emit_word();
  return words;
}

std::vector<SegmentTimestamp> words_to_segment_timestamps(
    const std::vector<WordTimestamp>& words,
    double seconds_per_encoder_frame) {
  // NeMo reference for segment grouping (word offsets -> segment offsets):
  //   https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/utils/timestamp_utils.py#L227-L327
  // Divergence: we only segment on terminal punctuation (.,!,?) and do not
  // implement NeMo's optional `segment_gap_threshold` splitting.
  std::vector<SegmentTimestamp> segments;
  if (words.empty()) {
    return segments;
  }

  std::string current_segment;
  int64_t segment_start_offset = 0;
  int64_t segment_end_offset = 0;
  bool has_segment = false;

  auto emit_segment = [&]() {
    if (!has_segment || current_segment.empty()) {
      return;
    }
    double start_sec = -1.0;
    double end_sec = -1.0;
    if (seconds_per_encoder_frame > 0.0) {
      start_sec = seconds_per_encoder_frame * segment_start_offset;
      end_sec = seconds_per_encoder_frame * segment_end_offset;
    }
    segments.push_back(SegmentTimestamp{
        current_segment, segment_start_offset, segment_end_offset, start_sec, end_sec});
    current_segment.clear();
    has_segment = false;
  };

  for (const auto& word : words) {
    if (!has_segment) {
      has_segment = true;
      current_segment = word.text;
      segment_start_offset = word.start_offset;
      segment_end_offset = word.end_offset;
    } else {
      current_segment += " ";
      current_segment += word.text;
      segment_end_offset = word.end_offset;
    }

    if (!word.text.empty()) {
      char last = word.text.back();
      if (last == '.' || last == '!' || last == '?') {
        // NeMo: segment delimiters are configurable (default includes '.', '?', '!'):
        //   https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/utils/timestamp_utils.py#L287-L296
        emit_segment();
      }
    }
  }

  emit_segment();
  return segments;
}

std::vector<TokenTimestamp> greedy_decode_executorch(
    Module& model,
    const ::executorch::aten::Tensor& encoder_output,
    int64_t encoder_len,
    int64_t blank_id,
    int64_t vocab_size,
    int64_t num_rnn_layers = 2,
    int64_t pred_hidden = 640,
    int64_t max_symbols_per_step = 10) {
  // NeMo reference for greedy TDT decoding (where the *token timing* originates):
  // - Core greedy TDT loop (token argmax + duration argmax + skip advance):
  //   https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py#L2627-L2717
  // - NeMo records per-token `timestamp` + `token_duration` here:
  //   https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py#L2684-L2693
  //
  // Divergences from NeMo:
  // - We implement a single-item (B=1) decoder loop directly over ExecuTorch
  //   exported methods (`joint_project_*`, `decoder_predict`, `joint`).
  // - We take argmax directly on raw logits (no log_softmax); this matches
  //   NeMo's argmax choice but we do not compute scores/confidence.
  // - NeMo's loop structure uses an explicit inner loop for `skip==0` label
  //   looping; here we emulate it with `dur==0` and `symbols_on_frame`.
  std::vector<TokenTimestamp> hypothesis;
  int64_t num_token_classes = vocab_size + 1;

  // Transpose encoder output from [1, enc_dim, time] to [1, time, enc_dim]
  auto enc_sizes = encoder_output.sizes();
  int64_t batch = enc_sizes[0];
  int64_t enc_dim = enc_sizes[1];
  int64_t time_steps = enc_sizes[2];

  // Create transposed tensor
  std::vector<float> transposed_data(batch * time_steps * enc_dim);
  const float* src = encoder_output.const_data_ptr<float>();
  for (int64_t t = 0; t < time_steps; t++) {
    for (int64_t d = 0; d < enc_dim; d++) {
      transposed_data[t * enc_dim + d] = src[d * time_steps + t];
    }
  }

  auto transposed_tensor = from_blob(
      transposed_data.data(),
      {static_cast<::executorch::aten::SizesType>(batch),
       static_cast<::executorch::aten::SizesType>(time_steps),
       static_cast<::executorch::aten::SizesType>(enc_dim)},
      ::executorch::aten::ScalarType::Float);

  // Project encoder output
  auto proj_enc_result = model.execute(
      "joint_project_encoder",
      std::vector<::executorch::runtime::EValue>{transposed_tensor});
  if (!proj_enc_result.ok()) {
    ET_LOG(Error, "joint_project_encoder failed");
    return hypothesis;
  }
  auto f_proj = proj_enc_result.get()[0].toTensor();

  // Initialize LSTM state
  std::vector<float> h_data(num_rnn_layers * 1 * pred_hidden, 0.0f);
  std::vector<float> c_data(num_rnn_layers * 1 * pred_hidden, 0.0f);

  auto h = from_blob(
      h_data.data(),
      {static_cast<::executorch::aten::SizesType>(num_rnn_layers),
       1,
       static_cast<::executorch::aten::SizesType>(pred_hidden)},
      ::executorch::aten::ScalarType::Float);
  auto c = from_blob(
      c_data.data(),
      {static_cast<::executorch::aten::SizesType>(num_rnn_layers),
       1,
       static_cast<::executorch::aten::SizesType>(pred_hidden)},
      ::executorch::aten::ScalarType::Float);

  // Prime the prediction network state with SOS (= blank_id) to match NeMo TDT
  // greedy label-looping decoding behavior:
  // - SOS is defined as blank:
  // https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/submodules/transducer_decoding/tdt_label_looping.py#L250
  // - Predictor priming with SOS:
  // https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/submodules/transducer_decoding/tdt_label_looping.py#L363-L368
  std::vector<int64_t> sos_token_data = {blank_id};
  auto sos_token = from_blob(
      sos_token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);
  auto decoder_init_result = model.execute(
      "decoder_predict",
      std::vector<::executorch::runtime::EValue>{sos_token, h, c});
  if (!decoder_init_result.ok()) {
    ET_LOG(Error, "decoder_predict (SOS) failed");
    return hypothesis;
  }
  auto& init_outputs = decoder_init_result.get();
  auto g_init = init_outputs[0].toTensor();
  auto new_h_init = init_outputs[1].toTensor();
  auto new_c_init = init_outputs[2].toTensor();
  std::memcpy(
      h_data.data(),
      new_h_init.const_data_ptr<float>(),
      h_data.size() * sizeof(float));
  std::memcpy(
      c_data.data(),
      new_c_init.const_data_ptr<float>(),
      c_data.size() * sizeof(float));

  auto g_proj_result = model.execute(
      "joint_project_decoder",
      std::vector<::executorch::runtime::EValue>{g_init});
  if (!g_proj_result.ok()) {
    ET_LOG(Error, "joint_project_decoder failed");
    return hypothesis;
  }
  auto g_proj_tensor = g_proj_result.get()[0].toTensor();

  // Copy g_proj data for reuse
  std::vector<float> g_proj_data(
      g_proj_tensor.const_data_ptr<float>(),
      g_proj_tensor.const_data_ptr<float>() + g_proj_tensor.numel());

  int64_t t = 0;
  int64_t symbols_on_frame = 0;

  // Scan over encoder output
  while (t < encoder_len) {
    // Get encoder frame at time t: f_proj[:, t:t+1, :]
    const float* f_proj_data = f_proj.const_data_ptr<float>();
    int64_t proj_dim = f_proj.sizes()[2];

    std::vector<float> f_t_data(1 * 1 * proj_dim);
    for (int64_t d = 0; d < proj_dim; d++) {
      f_t_data[d] = f_proj_data[t * proj_dim + d];
    }
    auto f_t = from_blob(
        f_t_data.data(),
        {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
        ::executorch::aten::ScalarType::Float);

    auto g_proj = from_blob(
        g_proj_data.data(),
        {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
        ::executorch::aten::ScalarType::Float);

    // Joint network
    auto joint_result = model.execute(
        "joint", std::vector<::executorch::runtime::EValue>{f_t, g_proj});
    if (!joint_result.ok()) {
      ET_LOG(Error, "joint failed at t=%lld", static_cast<long long>(t));
      return hypothesis;
    }
    auto full_logits = joint_result.get()[0].toTensor();

    // Split logits into token and duration
    const float* logits_data = full_logits.const_data_ptr<float>();

    // Find argmax for token logits
    int64_t k = 0;
    float max_token_logit = logits_data[0];
    for (int64_t i = 1; i < num_token_classes; i++) {
      if (logits_data[i] > max_token_logit) {
        max_token_logit = logits_data[i];
        k = i;
      }
    }

    // Find argmax for duration logits
    int64_t dur_idx = 0;
    float max_dur_logit = logits_data[num_token_classes];
    for (size_t i = 1; i < DURATIONS.size(); i++) {
      if (logits_data[num_token_classes + i] > max_dur_logit) {
        max_dur_logit = logits_data[num_token_classes + i];
        dur_idx = i;
      }
    }
    int64_t dur = DURATIONS[dur_idx];

    if (k == blank_id) {
      // NeMo: if blank is emitted with duration=0, it forces progress to avoid
      // infinite loops (skip==0 -> skip=1):
      //   https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py#L2700-L2704
      // Divergence: NeMo advances `time_idx += skip` first and patches `skip`
      // after the inner loop; here we apply `max(dur,1)` immediately in the
      // blank branch.
      t += std::max(dur, (int64_t)1);
      symbols_on_frame = 0;
    } else {
      // NeMo: emits token at `time_idx` and stores duration separately:
      //   https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py#L2684-L2693
      // NeMo later converts (timestamp, token_duration) -> (start_offset, end_offset):
      //   https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/submodules/rnnt_decoding.py#L1128-L1156
      // Divergence: we store (start_offset=t, end_offset=t+dur) directly.
      hypothesis.push_back(TokenTimestamp{k, t, t + dur});

      // Update decoder state
      std::vector<int64_t> token_data = {k};
      auto token = from_blob(
          token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);

      auto decoder_result = model.execute(
          "decoder_predict",
          std::vector<::executorch::runtime::EValue>{token, h, c});
      if (!decoder_result.ok()) {
        ET_LOG(Error, "decoder_predict failed");
        return hypothesis;
      }
      auto& outputs = decoder_result.get();
      auto g = outputs[0].toTensor();
      auto new_h = outputs[1].toTensor();
      auto new_c = outputs[2].toTensor();

      // Update h and c
      std::memcpy(
          h_data.data(),
          new_h.const_data_ptr<float>(),
          h_data.size() * sizeof(float));
      std::memcpy(
          c_data.data(),
          new_c.const_data_ptr<float>(),
          c_data.size() * sizeof(float));

      // Project decoder output
      auto proj_dec_result = model.execute(
          "joint_project_decoder",
          std::vector<::executorch::runtime::EValue>{g});
      if (!proj_dec_result.ok()) {
        ET_LOG(Error, "joint_project_decoder failed");
        return hypothesis;
      }
      auto new_g_proj = proj_dec_result.get()[0].toTensor();
      std::memcpy(
          g_proj_data.data(),
          new_g_proj.const_data_ptr<float>(),
          g_proj_data.size() * sizeof(float));

      t += dur;

      if (dur == 0) {
        // NeMo: label looping occurs when `skip == 0` (stay on same encoder frame)
        // until a non-zero skip is predicted, capped by `max_symbols_per_step`:
        // - need_loop = (skip == 0):
        //   https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py#L2695-L2699
        // - force progress after max symbols:
        //   https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py#L2715-L2716
        symbols_on_frame++;
        if (symbols_on_frame >= max_symbols_per_step) {
          t++;
          symbols_on_frame = 0;
        }
      } else {
        symbols_on_frame = 0;
      }
    }
  }

  return hypothesis;
}

std::string tokens_to_text(
    const std::vector<int64_t>& tokens,
    tokenizers::Tokenizer* tokenizer) {
  // Decode tokens to text one by one
  std::string result;
  uint64_t prev_token = 0;
  for (size_t i = 0; i < tokens.size(); i++) {
    uint64_t token = static_cast<uint64_t>(tokens[i]);
    auto decode_result = tokenizer->decode(prev_token, token);
    if (decode_result.ok()) {
      result += decode_result.get();
    }
    prev_token = token;
  }

  return result;
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_audio_path.empty()) {
    ET_LOG(Error, "audio_path flag must be provided.");
    return 1;
  }

  // Load model (which includes the bundled preprocessor)
  ET_LOG(Info, "Loading model from: %s", FLAGS_model_path.c_str());
  std::unique_ptr<Module> model;
  if (!FLAGS_data_path.empty()) {
    ET_LOG(Info, "Loading data from: %s", FLAGS_data_path.c_str());
    model = std::make_unique<Module>(
        FLAGS_model_path, FLAGS_data_path, Module::LoadMode::Mmap);
  } else {
    model = std::make_unique<Module>(FLAGS_model_path, Module::LoadMode::Mmap);
  }
  auto model_load_error = model->load();
  if (model_load_error != Error::Ok) {
    ET_LOG(Error, "Failed to load model.");
    return 1;
  }

  // Load audio
  ET_LOG(Info, "Loading audio from: %s", FLAGS_audio_path.c_str());
  std::vector<float> audio_data =
      ::executorch::extension::llm::load_wav_audio_data(FLAGS_audio_path);
  ET_LOG(Info, "Loaded %zu audio samples", audio_data.size());

  auto audio_tensor = from_blob(
      audio_data.data(),
      {static_cast<::executorch::aten::SizesType>(audio_data.size())},
      ::executorch::aten::ScalarType::Float);
  std::vector<int64_t> audio_len_data = {
      static_cast<int64_t>(audio_data.size())};
  auto audio_len_tensor = from_blob(
      audio_len_data.data(), {1}, ::executorch::aten::ScalarType::Long);

  ET_LOG(Info, "Running preprocessor...");
  auto proc_result = model->execute(
      "preprocessor",
      std::vector<::executorch::runtime::EValue>{
          audio_tensor, audio_len_tensor});
  if (!proc_result.ok()) {
    ET_LOG(Error, "Preprocessor forward failed.");
    return 1;
  }
  auto& proc_outputs = proc_result.get();
  auto mel = proc_outputs[0].toTensor();
  auto mel_len_tensor_out = proc_outputs[1].toTensor();
  int64_t mel_len_value = mel_len_tensor_out.const_data_ptr<int64_t>()[0];

  // Create mel_len tensor for encoder
  std::vector<int64_t> mel_len_data = {mel_len_value};
  auto mel_len =
      from_blob(mel_len_data.data(), {1}, ::executorch::aten::ScalarType::Long);

  ET_LOG(
      Info,
      "Mel spectrogram shape: [%ld, %ld, %ld], mel_len: %lld",
      static_cast<long>(mel.sizes()[0]),
      static_cast<long>(mel.sizes()[1]),
      static_cast<long>(mel.sizes()[2]),
      static_cast<long long>(mel_len_value));

  // Run encoder
  ET_LOG(Info, "Running encoder...");
  auto enc_result = model->execute(
      "encoder", std::vector<::executorch::runtime::EValue>{mel, mel_len});
  if (!enc_result.ok()) {
    ET_LOG(Error, "Encoder forward failed.");
    return 1;
  }
  auto& enc_outputs = enc_result.get();
  auto encoded = enc_outputs[0].toTensor();
  int64_t encoded_len = enc_outputs[1].toTensor().const_data_ptr<int64_t>()[0];

  ET_LOG(
      Info,
      "Encoder output shape: [%ld, %ld, %ld], len=%ld",
      static_cast<long>(encoded.sizes()[0]),
      static_cast<long>(encoded.sizes()[1]),
      static_cast<long>(encoded.sizes()[2]),
      static_cast<long>(encoded_len));

  // Query model metadata from constant_methods
  std::vector<::executorch::runtime::EValue> empty_inputs;
  auto num_rnn_layers_result = model->execute("num_rnn_layers", empty_inputs);
  auto pred_hidden_result = model->execute("pred_hidden", empty_inputs);
  auto vocab_size_result = model->execute("vocab_size", empty_inputs);
  auto blank_id_result = model->execute("blank_id", empty_inputs);
  auto sample_rate_result = model->execute("sample_rate", empty_inputs);

  if (!num_rnn_layers_result.ok() || !pred_hidden_result.ok() ||
      !vocab_size_result.ok() || !blank_id_result.ok() ||
      !sample_rate_result.ok()) {
    ET_LOG(
        Error,
        "Failed to query model metadata. Make sure the model was exported with constant_methods.");
    return 1;
  }

  int64_t vocab_size = vocab_size_result.get()[0].toInt();
  int64_t blank_id = blank_id_result.get()[0].toInt();
  int64_t num_rnn_layers = num_rnn_layers_result.get()[0].toInt();
  int64_t pred_hidden = pred_hidden_result.get()[0].toInt();
  int64_t sample_rate = sample_rate_result.get()[0].toInt();

  ET_LOG(
      Info,
      "Model metadata: vocab_size=%lld, blank_id=%lld, num_rnn_layers=%lld, pred_hidden=%lld, sample_rate=%lld",
      static_cast<long long>(vocab_size),
      static_cast<long long>(blank_id),
      static_cast<long long>(num_rnn_layers),
      static_cast<long long>(pred_hidden),
      static_cast<long long>(sample_rate));

  ET_LOG(Info, "Running TDT greedy decode...");
  auto tokens = greedy_decode_executorch(
      *model,
      encoded,
      encoded_len,
      blank_id,
      vocab_size,
      num_rnn_layers,
      pred_hidden);

  ET_LOG(Info, "Decoded %zu tokens", tokens.size());

  // Load tokenizer
  ET_LOG(Info, "Loading tokenizer from: %s", FLAGS_tokenizer_path.c_str());
  auto tokenizer =
      ::executorch::extension::llm::load_tokenizer(FLAGS_tokenizer_path);
  if (!tokenizer || !tokenizer->is_loaded()) {
    ET_LOG(
        Error,
        "Failed to load tokenizer from: %s",
        FLAGS_tokenizer_path.c_str());
    return 1;
  }

  // Convert tokens to text
  std::vector<int64_t> token_ids;
  token_ids.reserve(tokens.size());
  for (const auto& t : tokens) {
    token_ids.push_back(t.id);
  }
  std::string text = tokens_to_text(token_ids, tokenizer.get());
  std::cout << "Transcription tokens: " << text << std::endl;

  if (FLAGS_timestamps) {
    // NeMo: offset->seconds conversion uses
    //   start = start_offset * window_stride * subsampling_factor
    //   end   = end_offset   * window_stride * subsampling_factor
    //   https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/utils/timestamp_utils.py#L428-L479
    //
    // Divergence: NeMo reads `window_stride` from the preprocessor config and
    // `subsampling_factor` from the encoder module. In ExecuTorch we require
    // these values to be exported as `constant_methods` (`window_stride` and
    // `encoder_subsampling_factor`). If unavailable, we print raw offsets.
    std::vector<::executorch::runtime::EValue> empty_inputs;
    auto window_stride_result = model->execute("window_stride", empty_inputs);
    auto subsampling_factor_result =
        model->execute("encoder_subsampling_factor", empty_inputs);

    double seconds_per_encoder_frame = -1.0;
    if (window_stride_result.ok() && subsampling_factor_result.ok()) {
      double window_stride = window_stride_result.get()[0].toDouble();
      int64_t encoder_subsampling_factor =
          subsampling_factor_result.get()[0].toInt();
      seconds_per_encoder_frame = window_stride * encoder_subsampling_factor;
      ET_LOG(
          Info,
          "Timestamp metadata: window_stride=%f, encoder_subsampling_factor=%lld, seconds_per_encoder_frame=%f",
          window_stride,
          static_cast<long long>(encoder_subsampling_factor),
          seconds_per_encoder_frame);
    } else {
      ET_LOG(
          Error,
          "Timestamps requested but model metadata is missing. Re-export the model with constant_methods for window_stride and encoder_subsampling_factor.");
    }

    auto words = tokens_to_word_timestamps(
        tokens, tokenizer.get(), seconds_per_encoder_frame);
    auto segments =
        words_to_segment_timestamps(words, seconds_per_encoder_frame);

    std::cout << std::fixed << std::setprecision(2);

    auto subwords = tokens_to_subword_timestamps(
        tokens, tokenizer.get(), seconds_per_encoder_frame);

    std::cout << "\nSubword timestamps:\n";
    for (const auto& sw : subwords) {
      if (seconds_per_encoder_frame > 0.0) {
        std::cout << "[" << sw.start_sec << ", " << sw.end_sec << "] ";
      } else {
        std::cout << "[" << sw.start_offset << ", " << sw.end_offset << "] ";
      }
      std::cout << sw.text << "\n";
    }

    std::cout << "\nWord timestamps:\n";
    for (const auto& w : words) {
      if (seconds_per_encoder_frame > 0.0) {
        std::cout << "[" << w.start_sec << ", " << w.end_sec << "] ";
      } else {
        std::cout << "[" << w.start_offset << ", " << w.end_offset << "] ";
      }
      std::cout << w.text << "\n";
    }

    std::cout << "\nSegment timestamps:\n";
    for (const auto& s : segments) {
      if (seconds_per_encoder_frame > 0.0) {
        std::cout << "[" << s.start_sec << ", " << s.end_sec << "] ";
      } else {
        std::cout << "[" << s.start_offset << ", " << s.end_offset << "] ";
      }
      std::cout << s.text << "\n";
    }
  }

  ET_LOG(Info, "Done!");
  return 0;
}
