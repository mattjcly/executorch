/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "sentencepiece_processor.h"

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
    "tokenizer.model",
    "Path to SentencePiece tokenizer model file.");
DEFINE_string(
    data_path,
    "",
    "Path to data file (.ptd) for delegate data (optional, required for CUDA).");
DEFINE_string(
    timestamps,
    "none",
    "Timestamp output mode: none|subword|word|segment|all");

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

namespace {

// TDT duration values (Comes from model config in NeMo implementation)
// https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/models/rnnt_models.py#L230-L238
const std::vector<int> DURATIONS = {0, 1, 2, 3, 4};

struct Token {
  int64_t id;
  int64_t start_offset; // encoder frame index
  int64_t end_offset; // encoder frame index
};

struct TimestampedTextSpan {
  std::string text;
  int64_t start_offset;
  int64_t end_offset;
  double start_sec;
  double end_sec;
};

std::vector<std::string> split_lines(const std::string_view s) {
  std::vector<std::string> lines;
  size_t start = 0;
  while (start <= s.size()) {
    const size_t end = s.find('\n', start);
    const size_t line_end =
        (end == std::string_view::npos) ? s.size() : end;
    const std::string_view line_view = s.substr(start, line_end - start);
    if (!line_view.empty()) {
      lines.emplace_back(line_view);
    }
    if (end == std::string_view::npos) {
      break;
    }
    start = end + 1;
  }
  return lines;
}

std::unordered_set<std::string> split_lines_to_set(const std::string_view s) {
  std::unordered_set<std::string> out;
  for (const auto& line : split_lines(s)) {
    out.insert(line);
  }
  return out;
}

std::string sp_decode_ids(const sentencepiece::SentencePieceProcessor& sp, int id) {
  std::string out;
  const auto status = sp.Decode(std::vector<int>{id}, &out);
  return status.ok() ? out : std::string();
}

std::string sp_decode_pieces(
    const sentencepiece::SentencePieceProcessor& sp,
    const std::vector<std::string>& pieces) {
  std::string out;
  const auto status = sp.Decode(pieces, &out);
  return status.ok() ? out : std::string();
}

// NeMo TDT punctuation pinning (`_refine_timestamps_tdt`).
std::vector<Token> apply_tdt_punctuation_timestamp_correction(
    const std::vector<Token>& tokens,
    const sentencepiece::SentencePieceProcessor& sp,
    const std::unordered_set<std::string>& supported_punctuation) {
  std::vector<Token> corrected;
  corrected.reserve(tokens.size());

  int64_t prev_end_offset = 0;
  bool has_prev_end_offset = false;

  for (const auto& token : tokens) {
    const std::string token_text = sp_decode_ids(sp, static_cast<int>(token.id));
    const bool is_punct =
        supported_punctuation.find(token_text) != supported_punctuation.end();
    if (is_punct && has_prev_end_offset) {
      corrected.push_back(Token{token.id, prev_end_offset, prev_end_offset});
    } else {
      corrected.push_back(token);
    }
    prev_end_offset = corrected.back().end_offset;
    has_prev_end_offset = true;
  }

  return corrected;
}

struct TimestampOutputMode {
  bool subword = false;
  bool word = false;
  bool segment = false;

  bool enabled() const {
    return subword || word || segment;
  }
};

std::string to_lower_ascii(std::string s) {
  for (char& ch : s) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return s;
}

TimestampOutputMode parse_timestamp_output_mode(const std::string& raw_arg) {
  if (raw_arg.empty()) {
    throw std::invalid_argument(
        "Invalid --timestamps value (empty). Expected: subword, word, segment, all.");
  }
  const std::string mode = to_lower_ascii(raw_arg);
  if (mode == "none") {
    return {false, false, false};
  }
  if (mode == "subword") {
    return {true, false, false};
  }
  if (mode == "word") {
    return {false, true, false};
  }
  if (mode == "segment") {
    return {false, false, true};
  }
  if (mode == "all") {
    return {true, true, true};
  }
  throw std::invalid_argument(
      "Invalid --timestamps value '" + raw_arg +
      "'. Expected: subword, word, segment, all.");
}

std::vector<TimestampedTextSpan> tokens_to_timestamped_subwords(
    const std::vector<Token>& tokens,
    const sentencepiece::SentencePieceProcessor& sp,
    const double seconds_per_encoder_frame) {
  // NeMo reference (TDT offsets): RNNTDecoding._compute_offsets_tdt().
  std::vector<TimestampedTextSpan> subwords;
  subwords.reserve(tokens.size());

  for (const auto& token : tokens) {
    const std::string token_text = sp_decode_ids(sp, static_cast<int>(token.id));
    subwords.push_back(
        {token_text,
         token.start_offset,
         token.end_offset,
         seconds_per_encoder_frame * token.start_offset,
         seconds_per_encoder_frame * token.end_offset});
  }

  return subwords;
}

std::vector<TimestampedTextSpan> tokens_to_timestamped_words(
    const std::vector<Token>& tokens,
    const sentencepiece::SentencePieceProcessor& sp,
    const std::string& tokenizer_type,
    const std::string& word_delimiter_char,
    const std::unordered_set<std::string>& supported_punctuation,
    const double seconds_per_encoder_frame) {
  // NeMo reference: get_words_offsets().
  std::vector<TimestampedTextSpan> words;
  if (tokens.empty()) {
    return words;
  }

  // Cache decoded per-token text and per-token piece (id->piece).
  std::vector<std::string> char_texts;
  std::vector<std::string> char_tokens;
  char_texts.reserve(tokens.size());
  char_tokens.reserve(tokens.size());
  for (const auto& token : tokens) {
    char_texts.push_back(sp_decode_ids(sp, static_cast<int>(token.id)));
    char_tokens.push_back(sp.IdToPiece(static_cast<int>(token.id)));
  }

  auto starts_with = [](const std::string& s, const std::string_view prefix) {
    return s.size() >= prefix.size() &&
        std::string_view(s.data(), prefix.size()) == prefix;
  };

  auto is_word_start = [&](const std::string& token_piece,
                           const std::string& token_text,
                           const std::optional<std::string>& next_non_delim) {
    const bool next_is_punct =
        next_non_delim.has_value() &&
        supported_punctuation.find(*next_non_delim) != supported_punctuation.end();
    if ((tokenizer_type == "bpe" || tokenizer_type == "wpe") &&
        word_delimiter_char == " ") {
      if (tokenizer_type == "wpe") {
        return (token_text.size() > 0 && !starts_with(token_text, "##")) ||
            (token_text == word_delimiter_char && !next_is_punct);
      }
      return (token_piece != token_text) ||
          (token_text == word_delimiter_char && !next_is_punct);
    }
    if (word_delimiter_char == " ") {
      return token_text == word_delimiter_char && !next_is_punct;
    }
    return token_text == word_delimiter_char;
  };

  std::vector<std::string> built_tokens;
  built_tokens.reserve(8);
  size_t previous_token_index = 0;

  auto finalize_word = [&](size_t start_index, size_t end_index_exclusive) {
    if (built_tokens.empty() || end_index_exclusive == 0) {
      return;
    }
    const std::string word_text = sp_decode_pieces(sp, built_tokens);
    const int64_t start_offset = tokens[start_index].start_offset;
    const int64_t end_offset = tokens[end_index_exclusive - 1].end_offset;
    words.push_back(
        {word_text,
         start_offset,
         end_offset,
         seconds_per_encoder_frame * start_offset,
         seconds_per_encoder_frame * end_offset});
  };

  for (size_t i = 0; i < tokens.size(); i++) {
    const std::string& char_text = char_texts[i];
    const std::string& char_token = char_tokens[i];

    const bool curr_punctuation =
        !char_text.empty() &&
        char_text != word_delimiter_char &&
        supported_punctuation.find(char_text) != supported_punctuation.end();

    std::optional<std::string> next_non_delimiter_token = std::nullopt;
    for (size_t j = i + 1; j < tokens.size(); j++) {
      if (char_texts[j] != word_delimiter_char) {
        next_non_delimiter_token = char_texts[j];
        break;
      }
    }

    if (is_word_start(char_token, char_text, next_non_delimiter_token) &&
        !curr_punctuation) {
      if (!built_tokens.empty()) {
        finalize_word(previous_token_index, i);
      }
      built_tokens.clear();
      if (char_text != word_delimiter_char) {
        built_tokens.push_back(char_token);
        previous_token_index = i;
      }
      continue;
    }

    if (curr_punctuation && built_tokens.empty() && !words.empty()) {
      auto& last = words.back();
      last.end_offset = tokens[i].end_offset;
      last.end_sec = seconds_per_encoder_frame * last.end_offset;
      if (!last.text.empty() && last.text.back() == ' ') {
        last.text.pop_back();
      }
      last.text += char_text;
      continue;
    }

    if (curr_punctuation && !built_tokens.empty()) {
      if (built_tokens.back() == " " || built_tokens.back() == "_" ||
          built_tokens.back() == "▁") {
        built_tokens.pop_back();
      }
      built_tokens.push_back(char_token);
      continue;
    }

    if (built_tokens.empty()) {
      previous_token_index = i;
    }
    built_tokens.push_back(char_token);
  }

  if (words.empty()) {
    if (!built_tokens.empty()) {
      const std::string word_text = sp_decode_pieces(sp, built_tokens);
      const int64_t start_offset = tokens.front().start_offset;
      const int64_t end_offset = tokens.back().end_offset;
      words.push_back(
          {word_text,
           start_offset,
           end_offset,
           seconds_per_encoder_frame * start_offset,
           seconds_per_encoder_frame * end_offset});
    }
  } else {
    words[0].start_offset = tokens.front().start_offset;
    words[0].start_sec = seconds_per_encoder_frame * words[0].start_offset;
    if (!built_tokens.empty()) {
      const std::string word_text = sp_decode_pieces(sp, built_tokens);
      const int64_t start_offset = tokens[previous_token_index].start_offset;
      const int64_t end_offset = tokens.back().end_offset;
      words.push_back(
          {word_text,
           start_offset,
           end_offset,
           seconds_per_encoder_frame * start_offset,
           seconds_per_encoder_frame * end_offset});
    }
  }

  return words;
}

std::vector<TimestampedTextSpan> timestamped_words_to_timestamped_segments(
    const std::vector<TimestampedTextSpan>& words,
    const std::vector<std::string>& segment_delimiter_tokens,
    const std::optional<int64_t>& segment_gap_threshold,
    double seconds_per_encoder_frame) {
  // NeMo reference: get_segment_offsets().
  std::vector<TimestampedTextSpan> segments;
  if (words.empty()) {
    return segments;
  }

  std::unordered_set<std::string> delimiter_set;
  delimiter_set.reserve(segment_delimiter_tokens.size());
  for (const auto& d : segment_delimiter_tokens) {
    delimiter_set.insert(d);
  }

  std::vector<std::string> segment_words;
  segment_words.reserve(16);
  size_t previous_word_index = 0;

  auto join_words = [](const std::vector<std::string>& ws) {
    std::string out;
    for (size_t i = 0; i < ws.size(); i++) {
      if (i > 0) {
        out += " ";
      }
      out += ws[i];
    }
    return out;
  };

  auto emit_segment = [&](size_t start_word_index, size_t end_word_index) {
    if (segment_words.empty() || start_word_index >= words.size() ||
        end_word_index >= words.size() || end_word_index < start_word_index) {
      return;
    }
    const int64_t start_offset = words[start_word_index].start_offset;
    const int64_t end_offset = words[end_word_index].end_offset;
    segments.push_back(
        {join_words(segment_words),
         start_offset,
         end_offset,
         seconds_per_encoder_frame * start_offset,
         seconds_per_encoder_frame * end_offset});
    segment_words.clear();
  };

  auto is_delimiter_word = [&](const std::string& w) {
    if (w.empty()) {
      return false;
    }
    const std::string last_char(1, w.back());
    return delimiter_set.find(last_char) != delimiter_set.end() ||
        delimiter_set.find(w) != delimiter_set.end();
  };

  for (size_t i = 0; i < words.size(); i++) {
    const auto& w = words[i].text;
    if (segment_gap_threshold.has_value() && !segment_words.empty() && i > 0) {
      const int64_t gap = words[i].start_offset - words[i - 1].end_offset;
      if (gap >= *segment_gap_threshold) {
        emit_segment(previous_word_index, i - 1);
        segment_words.push_back(w);
        previous_word_index = i;
        continue;
      }
    }

    if (!w.empty() && is_delimiter_word(w)) {
      segment_words.push_back(w);
      emit_segment(previous_word_index, i);
      previous_word_index = i + 1;
      continue;
    }

    segment_words.push_back(w);
  }

  if (!segment_words.empty() && previous_word_index < words.size()) {
    emit_segment(previous_word_index, words.size() - 1);
  }

  return segments;
}

std::vector<Token> greedy_decode_executorch(
    Module& model,
    const ::executorch::aten::Tensor& encoder_output,
    int64_t encoder_len,
    int64_t blank_id,
    int64_t vocab_size,
    int64_t num_rnn_layers = 2,
    int64_t pred_hidden = 640,
    int64_t max_symbols_per_step = 10) {
  std::vector<Token> hypothesis;
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
      t += std::max(dur, (int64_t)1);
      symbols_on_frame = 0;
    } else {
      hypothesis.push_back(Token{k, t, t + dur});

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
    const std::vector<Token>& tokens,
    const tokenizers::Tokenizer& tokenizer) {
  // Decode tokens to text one by one
  std::string result;
  uint64_t prev_token = 0;
  for (const auto& token : tokens) {
    auto decode_result = tokenizer.decode(prev_token, token.id);
    if (decode_result.ok()) {
      result += decode_result.get();
    }
    prev_token = token.id;
  }

  return result;
}

void print_timestamped_spans(
    const char* label,
    const std::vector<TimestampedTextSpan>& spans) {
  std::cout << "\n" << label << " timestamps:\n";
  const std::ios_base::fmtflags old_flags = std::cout.flags();
  const std::streamsize old_precision = std::cout.precision();
  std::cout << std::fixed << std::setprecision(2);
  for (const auto& span : spans) {
    std::cout << "[" << span.start_sec << ", " << span.end_sec << "] ";
    std::cout << span.text << "\n";
  }
  std::cout.flags(old_flags);
  std::cout.precision(old_precision);
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  TimestampOutputMode timestamp_mode;
  try {
    timestamp_mode = parse_timestamp_output_mode(FLAGS_timestamps);
  } catch (const std::invalid_argument& e) {
    ET_LOG(Error, "%s", e.what());
    return 1;
  }

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
  const std::vector<::executorch::runtime::EValue> empty_inputs;
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
  std::string text = tokens_to_text(tokens, *tokenizer);
  std::cout << "Transcribed text: " << text << std::endl;

  if (!timestamp_mode.enabled()) {
    return 0;
  }

  // Query timestamp metadata
  auto window_stride_result = model->execute("window_stride", empty_inputs);
  auto subsampling_factor_result =
      model->execute("encoder_subsampling_factor", empty_inputs);
  auto supported_punctuation_result =
      model->execute("supported_punctuation", empty_inputs);
  auto tokenizer_type_result = model->execute("tokenizer_type", empty_inputs);
  auto word_separator_result = model->execute("word_separator", empty_inputs);
  auto segment_separators_result =
      model->execute("segment_separators", empty_inputs);
  auto segment_gap_threshold_result =
      model->execute("segment_gap_threshold", empty_inputs);

  if (!window_stride_result.ok() || !subsampling_factor_result.ok() ||
      !supported_punctuation_result.ok() || !tokenizer_type_result.ok() ||
      !word_separator_result.ok() || !segment_separators_result.ok() ||
      !segment_gap_threshold_result.ok()) {
    ET_LOG(
        Error,
        "Timestamps requested (--timestamps=%s) but model metadata is missing. Re-export the model with constant_methods for timestamp metadata.",
        FLAGS_timestamps.c_str());
    return 1;
  }

  double window_stride = window_stride_result.get()[0].toDouble();
  int64_t encoder_subsampling_factor =
      subsampling_factor_result.get()[0].toInt();
  const double seconds_per_encoder_frame =
      window_stride * encoder_subsampling_factor;

  const std::string supported_punctuation_str =
      std::string(supported_punctuation_result.get()[0].toString());
  const std::unordered_set<std::string> supported_punctuation =
      split_lines_to_set(supported_punctuation_str);
  const std::string tokenizer_type =
      to_lower_ascii(std::string(tokenizer_type_result.get()[0].toString()));
  const std::string word_separator =
      std::string(word_separator_result.get()[0].toString());
  const std::vector<std::string> segment_separators =
      split_lines(segment_separators_result.get()[0].toString());
  const int64_t segment_gap_threshold_value =
      segment_gap_threshold_result.get()[0].toInt();
  const std::optional<int64_t> segment_gap_threshold =
      (segment_gap_threshold_value < 0)
      ? std::nullopt
      : std::optional<int64_t>(segment_gap_threshold_value);

  sentencepiece::SentencePieceProcessor sp;
  const auto sp_load_status = sp.Load(FLAGS_tokenizer_path);
  if (!sp_load_status.ok()) {
    ET_LOG(
        Error,
        "Failed to load SentencePiece model from: %s",
        FLAGS_tokenizer_path.c_str());
    return 1;
  }

  const std::vector<Token> corrected_tokens =
      apply_tdt_punctuation_timestamp_correction(tokens, sp, supported_punctuation);

  ET_LOG(
      Info,
      "Timestamp metadata: window_stride=%f, encoder_subsampling_factor=%lld, seconds_per_encoder_frame=%f",
      window_stride,
      static_cast<long long>(encoder_subsampling_factor),
      seconds_per_encoder_frame);

  if (timestamp_mode.subword) {
    print_timestamped_spans(
        "Subword",
        tokens_to_timestamped_subwords(
            corrected_tokens, sp, seconds_per_encoder_frame));
  }

  std::vector<TimestampedTextSpan> words;
  if (timestamp_mode.word || timestamp_mode.segment) {
    words = tokens_to_timestamped_words(
        corrected_tokens,
        sp,
        tokenizer_type,
        word_separator,
        supported_punctuation,
        seconds_per_encoder_frame);
  }
  if (timestamp_mode.word) {
    print_timestamped_spans("Word", words);
  }
  if (timestamp_mode.segment) {
    print_timestamped_spans(
        "Segment",
        timestamped_words_to_timestamped_segments(
            words,
            segment_separators,
            segment_gap_threshold,
            seconds_per_encoder_frame));
  }

  return 0;
}
