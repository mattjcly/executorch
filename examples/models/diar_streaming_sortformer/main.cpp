/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gflags/gflags.h>

#include <executorch/extension/llm/runner/wav_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>

DEFINE_string(model_path, "model.pte", "Path to diarization model (.pte).");
DEFINE_string(audio_path, "", "Path to input audio file (.wav).");
DEFINE_string(
    data_path,
    "",
    "Path to data file (.ptd) for delegate data (optional).");
DEFINE_double(
    threshold,
    0.5,
    "Speaker activity threshold in [0,1] used to form segments.");
DEFINE_int32(
    audio_chunk_ms,
    100,
    "Audio chunk size (ms) used to simulate real-time streaming. Must be >= 100ms.");
DEFINE_bool(
    streaming_output,
    true,
    "Print diarization segments as they are committed during streaming.");
DEFINE_bool(
    final_summary,
    false,
    "Print a final sorted segment summary after processing completes (in addition to streaming output).");

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::EValue;

namespace {

constexpr int64_t kMinPreprocessorSamples = 1600; // matches exporter dynamic min

struct Segment {
  int speaker = -1;
  double start_sec = 0.0;
  double end_sec = 0.0;
};

void print_segment_line(const Segment& seg) {
  std::cout << "speaker_" << seg.speaker << "  " << seg.start_sec << "  "
            << seg.end_sec << "\n";
}

int64_t ceil_div_int64(int64_t a, int64_t b) {
  if (b <= 0) {
    throw std::invalid_argument("ceil_div_int64: b must be > 0");
  }
  if (a <= 0) {
    return 0;
  }
  return (a + b - 1) / b;
}

int64_t get_int_constant(Module& model, const char* name) {
  std::vector<EValue> empty_inputs;
  auto r = model.execute(name, empty_inputs);
  if (!r.ok()) {
    throw std::runtime_error(std::string("Failed to query constant method: ") + name);
  }
  return r.get()[0].toInt();
}

double get_double_constant(Module& model, const char* name) {
  std::vector<EValue> empty_inputs;
  auto r = model.execute(name, empty_inputs);
  if (!r.ok()) {
    throw std::runtime_error(std::string("Failed to query constant method: ") + name);
  }
  return r.get()[0].toDouble();
}

class FeatureBuffer {
 public:
  explicit FeatureBuffer(int64_t feat_dim) : feat_dim_(feat_dim) {
    if (feat_dim_ <= 0) {
      throw std::invalid_argument("FeatureBuffer: feat_dim must be > 0");
    }
  }

  int64_t feat_dim() const {
    return feat_dim_;
  }

  int64_t start_frame() const {
    return start_frame_;
  }

  int64_t num_frames() const {
    return static_cast<int64_t>(data_.size()) / feat_dim_;
  }

  int64_t end_frame() const {
    return start_frame_ + num_frames();
  }

  const float* frame_ptr(int64_t global_frame) const {
    if (global_frame < start_frame_ || global_frame >= end_frame()) {
      throw std::out_of_range("FeatureBuffer::frame_ptr: global_frame out of range");
    }
    const int64_t local = global_frame - start_frame_;
    return data_.data() + local * feat_dim_;
  }

  void append_frames(const float* frames, int64_t n_frames) {
    if (n_frames <= 0) {
      return;
    }
    const size_t old_sz = data_.size();
    const size_t add_sz = static_cast<size_t>(n_frames * feat_dim_);
    data_.resize(old_sz + add_sz);
    std::memcpy(data_.data() + old_sz, frames, add_sz * sizeof(float));
  }

  void replace_frames(int64_t global_start_frame, const float* frames, int64_t n_frames) {
    if (n_frames <= 0) {
      return;
    }
    if (global_start_frame < start_frame_ ||
        global_start_frame + n_frames > end_frame()) {
      throw std::out_of_range(
          "FeatureBuffer::replace_frames: replacement range out of buffer");
    }
    const int64_t local = global_start_frame - start_frame_;
    std::memcpy(
        data_.data() + local * feat_dim_,
        frames,
        static_cast<size_t>(n_frames * feat_dim_) * sizeof(float));
  }

  void drop_before(int64_t global_frame) {
    if (global_frame <= start_frame_) {
      return;
    }
    const int64_t drop_frames = std::min(global_frame - start_frame_, num_frames());
    if (drop_frames <= 0) {
      return;
    }
    if (drop_frames >= num_frames()) {
      data_.clear();
      start_frame_ = global_frame;
      return;
    }
    const size_t keep_frames = static_cast<size_t>(num_frames() - drop_frames);
    std::memmove(
        data_.data(),
        data_.data() + static_cast<size_t>(drop_frames * feat_dim_),
        keep_frames * static_cast<size_t>(feat_dim_) * sizeof(float));
    data_.resize(keep_frames * static_cast<size_t>(feat_dim_));
    start_frame_ += drop_frames;
  }

 private:
  int64_t feat_dim_ = 0;
  int64_t start_frame_ = 0;
  std::vector<float> data_;
};

class StreamingFrontend {
 public:
  StreamingFrontend(
      Module& model,
      int64_t hop_samples,
      int64_t feat_dim,
      int64_t tail_samples_target,
      int64_t holdback_frames)
      : model_(model),
        hop_samples_(hop_samples),
        feat_dim_(feat_dim),
        tail_samples_target_(tail_samples_target),
        holdback_frames_(holdback_frames),
        feats_(feat_dim) {
    if (hop_samples_ <= 0) {
      throw std::invalid_argument("StreamingFrontend: hop_samples must be > 0");
    }
    if (tail_samples_target_ < 0) {
      throw std::invalid_argument(
          "StreamingFrontend: tail_samples_target must be >= 0");
    }
    if (tail_samples_target_ % hop_samples_ != 0) {
      throw std::invalid_argument(
          "StreamingFrontend: tail_samples_target must be a multiple of hop_samples");
    }
    if (holdback_frames_ < 0) {
      throw std::invalid_argument("StreamingFrontend: holdback_frames must be >= 0");
    }
  }

  // Ingest a chunk of audio samples. `valid_n_samples` can be <= `n_samples`
  // if the chunk includes zero padding (e.g. end-of-stream).
  void ingest_audio(const float* samples, int64_t n_samples, int64_t valid_n_samples) {
    if (n_samples <= 0) {
      return;
    }
    if (valid_n_samples < 0 || valid_n_samples > n_samples) {
      throw std::invalid_argument(
          "StreamingFrontend::ingest_audio: invalid valid_n_samples");
    }

    const int64_t tail_samples = static_cast<int64_t>(audio_tail_.size());
    const int64_t valid_total_samples = tail_samples + valid_n_samples;
    const int64_t chunk_total_samples = tail_samples + n_samples;
    const int64_t tensor_total_samples =
        std::max<int64_t>(chunk_total_samples, kMinPreprocessorSamples);

    std::vector<float> segment(static_cast<size_t>(tensor_total_samples), 0.0f);
    if (tail_samples > 0) {
      std::memcpy(
          segment.data(),
          audio_tail_.data(),
          static_cast<size_t>(tail_samples) * sizeof(float));
    }
    std::memcpy(
        segment.data() + tail_samples,
        samples,
        static_cast<size_t>(n_samples) * sizeof(float));

    std::vector<int64_t> len_vec = {
        std::min<int64_t>(valid_total_samples, tensor_total_samples)};

    auto audio_tensor = from_blob(
        segment.data(),
        {static_cast<::executorch::aten::SizesType>(tensor_total_samples)},
        ::executorch::aten::ScalarType::Float);
    auto audio_len_tensor =
        from_blob(len_vec.data(), {1}, ::executorch::aten::ScalarType::Long);

    auto prep_result = model_.execute(
        "preprocessor", std::vector<EValue>{audio_tensor, audio_len_tensor});
    if (!prep_result.ok()) {
      throw std::runtime_error("preprocessor failed");
    }

    auto& prep_out = prep_result.get();
    auto features = prep_out[0].toTensor(); // [1, T_feat, feat_dim]
    if (features.scalar_type() != ::executorch::aten::ScalarType::Float) {
      throw std::runtime_error("Expected float features from preprocessor");
    }
    if (static_cast<int64_t>(features.sizes()[2]) != feat_dim_) {
      throw std::runtime_error("Unexpected feat_dim from preprocessor output");
    }

    const int64_t feat_len = prep_out[1].toTensor().const_data_ptr<int64_t>()[0];
    if (feat_len < 0) {
      throw std::runtime_error("Invalid feat_len from preprocessor");
    }

    const int64_t tail_frames = tail_samples / hop_samples_;
    const int64_t prev_end_frame = feats_.end_frame();
    const int64_t overlap_frames = std::min<int64_t>(
        std::min<int64_t>(tail_frames, feat_len), prev_end_frame);
    const int64_t replace_start = prev_end_frame - overlap_frames;

    const float* feat_ptr = features.const_data_ptr<float>();
    if (overlap_frames > 0) {
      feats_.replace_frames(replace_start, feat_ptr, overlap_frames);
    }
    const int64_t new_frames = feat_len - overlap_frames;
    if (new_frames > 0) {
      feats_.append_frames(feat_ptr + overlap_frames * feat_dim_, new_frames);
    }

    // Update audio tail with the provided samples (including any caller padding).
    audio_tail_.insert(audio_tail_.end(), samples, samples + n_samples);
    if (static_cast<int64_t>(audio_tail_.size()) > tail_samples_target_) {
      const auto drop = audio_tail_.size() - static_cast<size_t>(tail_samples_target_);
      audio_tail_.erase(audio_tail_.begin(), audio_tail_.begin() + drop);
    }
  }

  void finalize() {
    holdback_frames_ = 0;
  }

  int64_t stable_end_frame() const {
    const int64_t end = feats_.end_frame();
    return std::max<int64_t>(0, end - holdback_frames_);
  }

  const FeatureBuffer& features() const {
    return feats_;
  }

  FeatureBuffer& mutable_features() {
    return feats_;
  }

 private:
  Module& model_;
  int64_t hop_samples_ = 0;
  int64_t feat_dim_ = 0;
  int64_t tail_samples_target_ = 0;
  int64_t holdback_frames_ = 0;
  std::vector<float> audio_tail_;
  FeatureBuffer feats_;
};

class OnlineSegmentTracker {
 public:
  OnlineSegmentTracker(int64_t n_spk, double frame_sec, double threshold)
      : n_spk_(n_spk), frame_sec_(frame_sec), threshold_(threshold) {
    if (n_spk_ <= 0) {
      throw std::invalid_argument("OnlineSegmentTracker: n_spk must be > 0");
    }
    if (frame_sec_ <= 0) {
      throw std::invalid_argument("OnlineSegmentTracker: frame_sec must be > 0");
    }
    in_seg_.assign(static_cast<size_t>(n_spk_), false);
    start_frame_.assign(static_cast<size_t>(n_spk_), 0);
  }

  void process_frames(
      const float* posteriors,
      int64_t num_frames,
      int64_t global_start_frame,
      bool print_output) {
    if (num_frames <= 0) {
      return;
    }
    if (!posteriors) {
      throw std::invalid_argument("OnlineSegmentTracker::process_frames: null posteriors");
    }

    for (int64_t t = 0; t < num_frames; ++t) {
      const float* row = posteriors + t * n_spk_;
      const int64_t frame_idx = global_start_frame + t;
      for (int64_t spk = 0; spk < n_spk_; ++spk) {
        const bool active = static_cast<double>(row[spk]) >= threshold_;
        const size_t spk_u = static_cast<size_t>(spk);
        if (active && !in_seg_[spk_u]) {
          in_seg_[spk_u] = true;
          start_frame_[spk_u] = frame_idx;
        } else if (!active && in_seg_[spk_u]) {
          in_seg_[spk_u] = false;
          Segment seg;
          seg.speaker = static_cast<int>(spk);
          seg.start_sec = static_cast<double>(start_frame_[spk_u]) * frame_sec_;
          seg.end_sec = static_cast<double>(frame_idx) * frame_sec_;
          if (print_output) {
            print_segment_line(seg);
          }
        }
      }
    }
  }

  void flush(int64_t global_end_frame, bool print_output) {
    for (int64_t spk = 0; spk < n_spk_; ++spk) {
      const size_t spk_u = static_cast<size_t>(spk);
      if (!in_seg_[spk_u]) {
        continue;
      }
      in_seg_[spk_u] = false;
      Segment seg;
      seg.speaker = static_cast<int>(spk);
      seg.start_sec = static_cast<double>(start_frame_[spk_u]) * frame_sec_;
      seg.end_sec = static_cast<double>(global_end_frame) * frame_sec_;
      if (print_output) {
        print_segment_line(seg);
      }
    }
  }

 private:
  int64_t n_spk_ = 0;
  double frame_sec_ = 0.0;
  double threshold_ = 0.0;
  std::vector<bool> in_seg_;
  std::vector<int64_t> start_frame_;
};

void append_to_fixed_cache(
    std::vector<float>& cache,
    int64_t& cache_len,
    int64_t cache_max_len,
    int64_t emb_dim,
    const float* frames,
    int64_t n_frames) {
  if (n_frames <= 0) {
    return;
  }

  if (n_frames >= cache_max_len) {
    // Keep only the most recent cache_max_len frames.
    const float* src = frames + (n_frames - cache_max_len) * emb_dim;
    std::memcpy(cache.data(), src, cache_max_len * emb_dim * sizeof(float));
    cache_len = cache_max_len;
    return;
  }

  const int64_t total = cache_len + n_frames;
  if (total <= cache_max_len) {
    std::memcpy(
        cache.data() + cache_len * emb_dim,
        frames,
        n_frames * emb_dim * sizeof(float));
    cache_len = total;
    return;
  }

  // Drop the oldest `overflow` frames, shift left, append new.
  const int64_t overflow = total - cache_max_len;
  const int64_t remain = cache_len - overflow;
  if (remain > 0) {
    std::memmove(
        cache.data(),
        cache.data() + overflow * emb_dim,
        remain * emb_dim * sizeof(float));
  }
  std::memcpy(
      cache.data() + remain * emb_dim,
      frames,
      n_frames * emb_dim * sizeof(float));
  cache_len = cache_max_len;
}

std::vector<Segment> segments_from_posteriors(
    const std::vector<float>& posteriors,
    int64_t num_frames,
    int64_t n_spk,
    double frame_sec,
    double threshold) {
  std::vector<Segment> out;
  if (num_frames <= 0 || n_spk <= 0) {
    return out;
  }

  for (int64_t spk = 0; spk < n_spk; ++spk) {
    bool in_seg = false;
    int64_t start = 0;
    for (int64_t t = 0; t < num_frames; ++t) {
      const float p = posteriors[static_cast<size_t>(t * n_spk + spk)];
      const bool active = static_cast<double>(p) >= threshold;
      if (active && !in_seg) {
        in_seg = true;
        start = t;
      } else if (!active && in_seg) {
        in_seg = false;
        out.push_back(
            Segment{static_cast<int>(spk), start * frame_sec, t * frame_sec});
      }
    }
    if (in_seg) {
      out.push_back(Segment{
          static_cast<int>(spk), start * frame_sec, num_frames * frame_sec});
    }
  }

  std::sort(out.begin(), out.end(), [](const Segment& a, const Segment& b) {
    if (a.start_sec != b.start_sec) {
      return a.start_sec < b.start_sec;
    }
    if (a.end_sec != b.end_sec) {
      return a.end_sec < b.end_sec;
    }
    return a.speaker < b.speaker;
  });
  return out;
}

struct StreamingCacheState {
  std::vector<float> spkcache;
  std::vector<float> fifo;
  int64_t spkcache_len = 0;
  int64_t fifo_len = 0;
};

void update_embedding_caches(
    StreamingCacheState& state,
    int64_t spkcache_max_len,
    int64_t fifo_max_len,
    int64_t spkcache_update_period,
    int64_t emb_dim,
    const float* chunk_embs,
    int64_t chunk_pred_len) {
  if (chunk_pred_len <= 0) {
    return;
  }

  const int64_t fifo_len_before = state.fifo_len;
  const int64_t fifo_len_after = fifo_len_before + chunk_pred_len;

  if (fifo_len_after <= fifo_max_len) {
    std::memcpy(
        state.fifo.data() + fifo_len_before * emb_dim,
        chunk_embs,
        static_cast<size_t>(chunk_pred_len * emb_dim) * sizeof(float));
    state.fifo_len = fifo_len_after;
    return;
  }

  // Build a temporary FIFO buffer: [old_fifo_valid, new_chunk]
  std::vector<float> fifo_tmp(static_cast<size_t>(fifo_len_after * emb_dim), 0.0f);
  if (fifo_len_before > 0) {
    std::memcpy(
        fifo_tmp.data(),
        state.fifo.data(),
        static_cast<size_t>(fifo_len_before * emb_dim) * sizeof(float));
  }
  std::memcpy(
      fifo_tmp.data() + fifo_len_before * emb_dim,
      chunk_embs,
      static_cast<size_t>(chunk_pred_len * emb_dim) * sizeof(float));

  int64_t pop_out_len = spkcache_update_period;
  pop_out_len = std::max(pop_out_len, chunk_pred_len - fifo_max_len + fifo_len_before);
  pop_out_len = std::min(pop_out_len, fifo_len_after);

  // Move the oldest `pop_out_len` frames from FIFO into speaker cache.
  append_to_fixed_cache(
      state.spkcache,
      state.spkcache_len,
      spkcache_max_len,
      emb_dim,
      fifo_tmp.data(),
      pop_out_len);

  // Keep the remaining FIFO frames.
  const int64_t new_fifo_len = fifo_len_after - pop_out_len;
  if (new_fifo_len > 0) {
    std::memcpy(
        state.fifo.data(),
        fifo_tmp.data() + pop_out_len * emb_dim,
        static_cast<size_t>(new_fifo_len * emb_dim) * sizeof(float));
  }
  if (new_fifo_len < fifo_max_len) {
    std::fill(
        state.fifo.begin() + static_cast<size_t>(new_fifo_len * emb_dim),
        state.fifo.end(),
        0.0f);
  }
  state.fifo_len = new_fifo_len;
}

} // namespace

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_audio_path.empty()) {
    ET_LOG(Error, "--audio_path is required.");
    return 1;
  }

  try {
    ET_LOG(Info, "Loading model from: %s", FLAGS_model_path.c_str());
    std::unique_ptr<Module> model;
    if (!FLAGS_data_path.empty()) {
      ET_LOG(Info, "Loading data from: %s", FLAGS_data_path.c_str());
      model = std::make_unique<Module>(
          FLAGS_model_path, FLAGS_data_path, Module::LoadMode::Mmap);
    } else {
      model = std::make_unique<Module>(FLAGS_model_path, Module::LoadMode::Mmap);
    }
    auto err = model->load();
    if (err != ::executorch::runtime::Error::Ok) {
      ET_LOG(Error, "Failed to load model program.");
      return 1;
    }

    const int64_t model_sample_rate = get_int_constant(*model, "sample_rate");
    const double window_stride = get_double_constant(*model, "window_stride");
    const int64_t hop_length_samples = get_int_constant(*model, "hop_length_samples");
    const int64_t n_fft = get_int_constant(*model, "n_fft");
    const int64_t subsampling_factor = get_int_constant(*model, "subsampling_factor");
    const int64_t feat_dim = get_int_constant(*model, "feat_dim");
    const int64_t chunk_len = get_int_constant(*model, "chunk_len");
    const int64_t chunk_left_context = get_int_constant(*model, "chunk_left_context");
    const int64_t chunk_right_context = get_int_constant(*model, "chunk_right_context");
    const int64_t spkcache_max_len = get_int_constant(*model, "spkcache_len");
    const int64_t fifo_max_len = get_int_constant(*model, "fifo_len");
    const int64_t spkcache_update_period =
        get_int_constant(*model, "spkcache_update_period");
    const int64_t emb_dim = get_int_constant(*model, "emb_dim");
    const int64_t n_spk = get_int_constant(*model, "n_spk");
    const double negative_init_val = get_double_constant(*model, "negative_init_val");
    const int64_t max_chunk_feat_frames =
        get_int_constant(*model, "max_chunk_feat_frames");

    const int64_t chunk_feat_frames = chunk_len * subsampling_factor;
    const int64_t left_feat_frames = chunk_left_context * subsampling_factor;
    const int64_t right_feat_frames = chunk_right_context * subsampling_factor;

    const double frame_sec = window_stride * static_cast<double>(subsampling_factor);

    if (hop_length_samples <= 0) {
      ET_LOG(Error, "Invalid hop_length_samples=%lld", static_cast<long long>(hop_length_samples));
      return 1;
    }

    // Hold back a small number of feature frames at the stream boundary so we don't
    // commit STFT frames that would change once more audio arrives.
    const int64_t frontend_holdback_frames =
        (n_fft > 0) ? ceil_div_int64(n_fft / 2, hop_length_samples) : 2;
    const int64_t frontend_tail_frames =
        (n_fft > 0) ? ceil_div_int64(n_fft, hop_length_samples) : 8;
    const int64_t frontend_tail_samples =
        frontend_tail_frames * hop_length_samples;

    ET_LOG(
        Info,
        "Frontend: hop=%lld samples, n_fft=%lld, tail=%lld samples (%lld frames), holdback=%lld frames",
        static_cast<long long>(hop_length_samples),
        static_cast<long long>(n_fft),
        static_cast<long long>(frontend_tail_samples),
        static_cast<long long>(frontend_tail_frames),
        static_cast<long long>(frontend_holdback_frames));

    ET_LOG(
        Info,
        "Model metadata: sample_rate=%lld, window_stride=%.6f, subsampling_factor=%lld, frame_sec=%.4f, "
        "feat_dim=%lld, n_spk=%lld, emb_dim=%lld, chunk_len=%lld, lc=%lld, rc=%lld, max_chunk_feat_frames=%lld, "
        "spkcache_len=%lld, fifo_len=%lld, spkcache_update_period=%lld",
        static_cast<long long>(model_sample_rate),
        window_stride,
        static_cast<long long>(subsampling_factor),
        frame_sec,
        static_cast<long long>(feat_dim),
        static_cast<long long>(n_spk),
        static_cast<long long>(emb_dim),
        static_cast<long long>(chunk_len),
        static_cast<long long>(chunk_left_context),
        static_cast<long long>(chunk_right_context),
        static_cast<long long>(max_chunk_feat_frames),
        static_cast<long long>(spkcache_max_len),
        static_cast<long long>(fifo_max_len),
        static_cast<long long>(spkcache_update_period));

    // Load WAV and validate format.
    auto header = executorch::extension::llm::load_wav_header(FLAGS_audio_path);
    if (header.get() == nullptr) {
      ET_LOG(Error, "Failed to load WAV header: %s", FLAGS_audio_path.c_str());
      return 1;
    }
    if (header->NumOfChan != 1) {
      ET_LOG(
          Error,
          "Only mono WAV is supported. Got NumOfChan=%d",
          static_cast<int>(header->NumOfChan));
      return 1;
    }
    if (static_cast<int64_t>(header->SamplesPerSec) != model_sample_rate) {
      ET_LOG(
          Error,
          "WAV sample rate (%d) != model sample rate (%lld). Resample the WAV first.",
          static_cast<int>(header->SamplesPerSec),
          static_cast<long long>(model_sample_rate));
      return 1;
    }

    ET_LOG(Info, "Loading WAV audio samples...");
    std::vector<float> audio =
        executorch::extension::llm::load_wav_audio_data(FLAGS_audio_path);

    // True streaming frontend: ingest audio chunk-by-chunk and emit feature frames incrementally.
    if (FLAGS_audio_chunk_ms < 100) {
      ET_LOG(Error, "--audio_chunk_ms must be >= 100");
      return 1;
    }
    int64_t audio_chunk_samples = static_cast<int64_t>(std::llround(
        (static_cast<double>(model_sample_rate) * static_cast<double>(FLAGS_audio_chunk_ms)) /
        1000.0));
    audio_chunk_samples = (audio_chunk_samples / hop_length_samples) * hop_length_samples;
    if (audio_chunk_samples < kMinPreprocessorSamples) {
      ET_LOG(
          Error,
          "--audio_chunk_ms=%d yields %lld samples (<%lld). Increase chunk size.",
          FLAGS_audio_chunk_ms,
          static_cast<long long>(audio_chunk_samples),
          static_cast<long long>(kMinPreprocessorSamples));
      return 1;
    }

    StreamingFrontend frontend(
        *model,
        hop_length_samples,
        feat_dim,
        frontend_tail_samples,
        frontend_holdback_frames);

    // Streaming caches (embeddings, not posteriors).
    StreamingCacheState cache_state;
    cache_state.spkcache.resize(
        static_cast<size_t>(spkcache_max_len * emb_dim), 0.0f);
    cache_state.fifo.resize(
        static_cast<size_t>(fifo_max_len * emb_dim), 0.0f);

    OnlineSegmentTracker segment_tracker(n_spk, frame_sec, FLAGS_threshold);
    if (FLAGS_streaming_output) {
      std::cout << "Streaming segments (threshold=" << FLAGS_threshold << ")\n";
      std::cout << std::fixed << std::setprecision(3);
    }

    // Optional: accumulate frame posteriors for a final sorted summary.
    std::vector<float> posteriors; // row-major: [T_diar, n_spk]

    std::vector<float> chunk_feat_buf(
        static_cast<size_t>(max_chunk_feat_frames * feat_dim), 0.0f);

    int64_t stt_feat = 0; // feature-frame cursor for chunk starts (no left context)
    int step_idx = 0;
    int64_t diar_frame_cursor = 0; // global diar-frame index for streaming output
    const int64_t total_samples = static_cast<int64_t>(audio.size());
    int64_t audio_pos = 0;

    auto run_step = [&](int64_t total_feat_frames, bool require_full_right_context) -> bool {
      const int64_t stable_end = std::min<int64_t>(frontend.stable_end_frame(), total_feat_frames);
      if (require_full_right_context) {
        const int64_t need = stt_feat + chunk_feat_frames + right_feat_frames;
        if (stable_end < need) {
          return false;
        }
      } else {
        // Tail mode: allow a final partial chunk when less than chunk_feat_frames remain.
        if (stable_end <= stt_feat) {
          return false;
        }
      }

      const int64_t end_feat = std::min<int64_t>(stt_feat + chunk_feat_frames, stable_end);
      const int64_t left_offset = std::min<int64_t>(left_feat_frames, stt_feat);
      const int64_t right_offset = std::min<int64_t>(right_feat_frames, stable_end - end_feat);
      const int64_t chunk_start = stt_feat - left_offset;
      const int64_t chunk_valid_frames = (end_feat + right_offset) - chunk_start;

      // Prepare fixed-size chunk input with padding.
      std::fill(
          chunk_feat_buf.begin(),
          chunk_feat_buf.end(),
          static_cast<float>(negative_init_val));
      const auto& feats = frontend.features();
      for (int64_t i = 0; i < chunk_valid_frames; ++i) {
        const float* src = feats.frame_ptr(chunk_start + i);
        float* dst = chunk_feat_buf.data() + i * feat_dim;
        std::memcpy(dst, src, static_cast<size_t>(feat_dim) * sizeof(float));
      }

      std::vector<int64_t> chunk_len_vec = {chunk_valid_frames};
      auto chunk_tensor = from_blob(
          chunk_feat_buf.data(),
          {static_cast<::executorch::aten::SizesType>(1),
           static_cast<::executorch::aten::SizesType>(max_chunk_feat_frames),
           static_cast<::executorch::aten::SizesType>(feat_dim)},
          ::executorch::aten::ScalarType::Float);
      auto chunk_len_tensor = from_blob(
          chunk_len_vec.data(),
          {static_cast<::executorch::aten::SizesType>(1)},
          ::executorch::aten::ScalarType::Long);

      auto spkcache_tensor = from_blob(
          cache_state.spkcache.data(),
          {static_cast<::executorch::aten::SizesType>(1),
           static_cast<::executorch::aten::SizesType>(spkcache_max_len),
           static_cast<::executorch::aten::SizesType>(emb_dim)},
          ::executorch::aten::ScalarType::Float);
      std::vector<int64_t> spkcache_len_vec = {cache_state.spkcache_len};
      auto spkcache_len_tensor = from_blob(
          spkcache_len_vec.data(),
          {static_cast<::executorch::aten::SizesType>(1)},
          ::executorch::aten::ScalarType::Long);

      auto fifo_tensor = from_blob(
          cache_state.fifo.data(),
          {static_cast<::executorch::aten::SizesType>(1),
           static_cast<::executorch::aten::SizesType>(fifo_max_len),
           static_cast<::executorch::aten::SizesType>(emb_dim)},
          ::executorch::aten::ScalarType::Float);
      std::vector<int64_t> fifo_len_vec = {cache_state.fifo_len};
      auto fifo_len_tensor = from_blob(
          fifo_len_vec.data(),
          {static_cast<::executorch::aten::SizesType>(1)},
          ::executorch::aten::ScalarType::Long);

      // lc/rc in diar frames (post-subsampling), derived from feature-frame offsets.
      const int64_t lc = static_cast<int64_t>(std::llround(
          static_cast<double>(left_offset) / static_cast<double>(subsampling_factor)));
      const int64_t rc =
          (right_offset + subsampling_factor - 1) / subsampling_factor;
      std::vector<int64_t> lc_vec = {lc};
      std::vector<int64_t> rc_vec = {rc};
      auto lc_tensor =
          from_blob(lc_vec.data(), {1}, ::executorch::aten::ScalarType::Long);
      auto rc_tensor =
          from_blob(rc_vec.data(), {1}, ::executorch::aten::ScalarType::Long);

      ET_LOG(
          Info,
          "Streaming step %d: feat[%lld:%lld] lc=%lld rc=%lld stable_end=%lld",
          step_idx,
          static_cast<long long>(chunk_start),
          static_cast<long long>(end_feat),
          static_cast<long long>(lc),
          static_cast<long long>(rc),
          static_cast<long long>(stable_end));

      auto step_result = model->execute(
          "model_step",
          std::vector<EValue>{
              chunk_tensor,
              chunk_len_tensor,
              spkcache_tensor,
              spkcache_len_tensor,
              fifo_tensor,
              fifo_len_tensor,
              lc_tensor,
              rc_tensor,
          });
      if (!step_result.ok()) {
        ET_LOG(Error, "model_step failed at step %d.", step_idx);
        return false;
      }

      auto& step_out = step_result.get();
      auto chunk_preds = step_out[0].toTensor(); // [1, chunk_len_max, n_spk]
      auto chunk_embs = step_out[1].toTensor(); // [1, chunk_len_max, emb_dim]
      const int64_t chunk_pred_len =
          step_out[2].toTensor().const_data_ptr<int64_t>()[0];

      if (chunk_pred_len > 0) {
        const float* preds_ptr = chunk_preds.const_data_ptr<float>();
        const float* embs_ptr = chunk_embs.const_data_ptr<float>();
        if (FLAGS_final_summary) {
          posteriors.resize(
              posteriors.size() + static_cast<size_t>(chunk_pred_len * n_spk));
          float* out_ptr = posteriors.data() +
              (posteriors.size() - static_cast<size_t>(chunk_pred_len * n_spk));
          std::memcpy(
              out_ptr,
              preds_ptr,
              static_cast<size_t>(chunk_pred_len * n_spk) * sizeof(float));
        }

        segment_tracker.process_frames(
            preds_ptr,
            chunk_pred_len,
            diar_frame_cursor,
            /*print_output=*/FLAGS_streaming_output);
        diar_frame_cursor += chunk_pred_len;

        update_embedding_caches(
            cache_state,
            spkcache_max_len,
            fifo_max_len,
            spkcache_update_period,
            emb_dim,
            embs_ptr,
            chunk_pred_len);
      }

      stt_feat = end_feat;
      step_idx += 1;

      // Keep only what is needed for future left context and for frontend overlap updates.
      const int64_t keep_from = std::max<int64_t>(0, stt_feat - left_feat_frames);
      frontend.mutable_features().drop_before(keep_from);
      return true;
    };

    // Simulate streaming by feeding audio chunks and running diar steps whenever enough
    // right-context audio has arrived.
    while (audio_pos < total_samples) {
      const int64_t remaining = total_samples - audio_pos;
      const int64_t take = std::min<int64_t>(audio_chunk_samples, remaining);
      const int64_t padded =
          (take % hop_length_samples == 0)
          ? take
          : ((take / hop_length_samples) + 1) * hop_length_samples;

      std::vector<float> chunk(static_cast<size_t>(padded), 0.0f);
      std::memcpy(
          chunk.data(),
          audio.data() + audio_pos,
          static_cast<size_t>(take) * sizeof(float));

      frontend.ingest_audio(chunk.data(), padded, take);
      audio_pos += take;

      const int64_t total_feat_frames =
          static_cast<int64_t>(audio_pos) / hop_length_samples;
      while (run_step(total_feat_frames, /*require_full_right_context=*/true)) {
        // Drain all runnable steps for current available audio.
      }
    }

    // Finalize the frontend (no more audio will arrive) and flush remaining chunks,
    // allowing partial right context on the tail.
    frontend.finalize();
    const int64_t total_feat_frames = total_samples / hop_length_samples;
    while (run_step(total_feat_frames, /*require_full_right_context=*/false)) {
      // Drain the tail.
    }

    // Flush any open segments at end-of-stream.
    segment_tracker.flush(
        diar_frame_cursor, /*print_output=*/FLAGS_streaming_output);

    const int64_t num_diar_frames = diar_frame_cursor;
    ET_LOG(
        Info,
        "Produced %lld diar frames (%.2f sec)",
        static_cast<long long>(num_diar_frames),
        num_diar_frames * frame_sec);

    if (FLAGS_final_summary) {
      auto segments = segments_from_posteriors(
          posteriors, num_diar_frames, n_spk, frame_sec, FLAGS_threshold);

      std::cout << "\nFinal segments (sorted, threshold=" << FLAGS_threshold << ")\n";
      for (const auto& seg : segments) {
        print_segment_line(seg);
      }
    }

    return 0;
  } catch (const std::exception& e) {
    ET_LOG(Error, "Exception: %s", e.what());
    return 1;
  }
}
