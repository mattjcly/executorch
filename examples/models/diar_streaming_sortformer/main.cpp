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
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gflags/gflags.h>

#include <executorch/extension/llm/runner/wav_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
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

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::EValue;

namespace {

struct Segment {
  int speaker = -1;
  double start_sec = 0.0;
  double end_sec = 0.0;
};

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

} // namespace

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_audio_path.empty()) {
    ET_LOG(Error, "--audio_path is required.");
    return 1;
  }

  try {
    ET_LOG(Info, "Loading model from: %s", FLAGS_model_path.c_str());
    auto model = std::make_unique<Module>(FLAGS_model_path, FLAGS_data_path);

    const int64_t model_sample_rate = get_int_constant(*model, "sample_rate");
    const double window_stride = get_double_constant(*model, "window_stride");
    const int64_t subsampling_factor = get_int_constant(*model, "subsampling_factor");
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

    const int64_t chunk_feat_frames = chunk_len * subsampling_factor;
    const int64_t left_feat_frames = chunk_left_context * subsampling_factor;
    const int64_t right_feat_frames = chunk_right_context * subsampling_factor;
    const int64_t max_chunk_feat_frames =
        (chunk_left_context + chunk_len + chunk_right_context) *
        subsampling_factor;

    const double frame_sec = window_stride * static_cast<double>(subsampling_factor);

    ET_LOG(
        Info,
        "Model metadata: sample_rate=%lld, window_stride=%.6f, subsampling_factor=%lld, frame_sec=%.4f, "
        "n_spk=%lld, emb_dim=%lld, chunk_len=%lld, lc=%lld, rc=%lld, max_chunk_feat_frames=%lld, "
        "spkcache_len=%lld, fifo_len=%lld, spkcache_update_period=%lld",
        static_cast<long long>(model_sample_rate),
        window_stride,
        static_cast<long long>(subsampling_factor),
        frame_sec,
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

    // Run preprocessor once for the full audio file.
    ET_LOG(Info, "Running preprocessor...");
    std::vector<int64_t> audio_len_vec = {static_cast<int64_t>(audio.size())};
    auto audio_tensor = from_blob(
        audio.data(),
        {static_cast<::executorch::aten::SizesType>(audio.size())},
        ::executorch::aten::ScalarType::Float);
    auto audio_len_tensor = from_blob(
        audio_len_vec.data(), {1}, ::executorch::aten::ScalarType::Long);

    auto prep_result = model->execute(
        "preprocessor",
        std::vector<EValue>{audio_tensor, audio_len_tensor});
    if (!prep_result.ok()) {
      ET_LOG(Error, "preprocessor failed.");
      return 1;
    }
    auto& prep_out = prep_result.get();
    auto features = prep_out[0].toTensor(); // [1, T_feat, feat_dim]
    int64_t feat_len = prep_out[1].toTensor().const_data_ptr<int64_t>()[0];

    const int64_t feat_dim = static_cast<int64_t>(features.sizes()[2]);
    if (features.scalar_type() != ::executorch::aten::ScalarType::Float) {
      ET_LOG(Error, "Expected float features from preprocessor.");
      return 1;
    }

    ET_LOG(
        Info,
        "Features shape: [1, %lld, %lld], feat_len=%lld",
        static_cast<long long>(static_cast<int64_t>(features.sizes()[1])),
        static_cast<long long>(feat_dim),
        static_cast<long long>(feat_len));

    const float* feat_ptr = features.const_data_ptr<float>();

    // Streaming caches (embeddings, not posteriors).
    std::vector<float> spkcache(
        static_cast<size_t>(spkcache_max_len * emb_dim), 0.0f);
    std::vector<float> fifo(static_cast<size_t>(fifo_max_len * emb_dim), 0.0f);
    int64_t spkcache_len = 0;
    int64_t fifo_len = 0;

    // Accumulate diar posteriors per diar frame.
    std::vector<float> posteriors; // row-major: [T_diar, n_spk]

    std::vector<float> chunk_feat_buf(
        static_cast<size_t>(max_chunk_feat_frames * feat_dim), 0.0f);

    int64_t stt_feat = 0;
    int step_idx = 0;
    while (stt_feat < feat_len) {
      const int64_t left_offset =
          std::min<int64_t>(left_feat_frames, stt_feat);
      const int64_t end_feat =
          std::min<int64_t>(stt_feat + chunk_feat_frames, feat_len);
      const int64_t right_offset =
          std::min<int64_t>(right_feat_frames, feat_len - end_feat);

      const int64_t chunk_start = stt_feat - left_offset;
      const int64_t chunk_valid_frames =
          (end_feat + right_offset) - chunk_start;

      // Prepare fixed-size chunk input with padding.
      std::fill(
          chunk_feat_buf.begin(),
          chunk_feat_buf.end(),
          static_cast<float>(negative_init_val));
      for (int64_t i = 0; i < chunk_valid_frames; ++i) {
        const float* src = feat_ptr + (chunk_start + i) * feat_dim;
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
          spkcache.data(),
          {static_cast<::executorch::aten::SizesType>(1),
           static_cast<::executorch::aten::SizesType>(spkcache_max_len),
           static_cast<::executorch::aten::SizesType>(emb_dim)},
          ::executorch::aten::ScalarType::Float);
      std::vector<int64_t> spkcache_len_vec = {spkcache_len};
      auto spkcache_len_tensor = from_blob(
          spkcache_len_vec.data(),
          {static_cast<::executorch::aten::SizesType>(1)},
          ::executorch::aten::ScalarType::Long);

      auto fifo_tensor = from_blob(
          fifo.data(),
          {static_cast<::executorch::aten::SizesType>(1),
           static_cast<::executorch::aten::SizesType>(fifo_max_len),
           static_cast<::executorch::aten::SizesType>(emb_dim)},
          ::executorch::aten::ScalarType::Float);
      std::vector<int64_t> fifo_len_vec = {fifo_len};
      auto fifo_len_tensor = from_blob(
          fifo_len_vec.data(),
          {static_cast<::executorch::aten::SizesType>(1)},
          ::executorch::aten::ScalarType::Long);

      // lc/rc in diar frames (post-subsampling), derived from feature-frame offsets.
      const int64_t lc = static_cast<int64_t>(
          std::llround(static_cast<double>(left_offset) /
                       static_cast<double>(subsampling_factor)));
      const int64_t rc = (right_offset + subsampling_factor - 1) / subsampling_factor;
      std::vector<int64_t> lc_vec = {lc};
      std::vector<int64_t> rc_vec = {rc};
      auto lc_tensor =
          from_blob(
              lc_vec.data(),
              {static_cast<::executorch::aten::SizesType>(1)},
              ::executorch::aten::ScalarType::Long);
      auto rc_tensor =
          from_blob(
              rc_vec.data(),
              {static_cast<::executorch::aten::SizesType>(1)},
              ::executorch::aten::ScalarType::Long);

      ET_LOG(Info, "Processing step %d: feat[%lld:%lld] lc=%lld rc=%lld", step_idx,
             static_cast<long long>(chunk_start),
             static_cast<long long>(end_feat),
             static_cast<long long>(lc),
             static_cast<long long>(rc));
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
        return 1;
      }

      auto& step_out = step_result.get();
      auto chunk_preds = step_out[0].toTensor(); // [1, chunk_len_max, n_spk]
      auto chunk_embs = step_out[1].toTensor(); // [1, chunk_len_max, emb_dim]
      const int64_t chunk_pred_len =
          step_out[2].toTensor().const_data_ptr<int64_t>()[0];

      if (chunk_pred_len > 0) {
        const float* preds_ptr = chunk_preds.const_data_ptr<float>();
        const float* embs_ptr = chunk_embs.const_data_ptr<float>();
        posteriors.resize(
            posteriors.size() + static_cast<size_t>(chunk_pred_len * n_spk));
        float* out_ptr =
            posteriors.data() + (posteriors.size() - static_cast<size_t>(chunk_pred_len * n_spk));
        std::memcpy(
            out_ptr,
            preds_ptr,
            static_cast<size_t>(chunk_pred_len * n_spk) * sizeof(float));

        // Update FIFO + speaker cache (embedding-only). This approximates NeMo streaming
        // cache management but does not implement Sortformer speaker-cache compression.
        const int64_t fifo_len_before = fifo_len;
        const int64_t fifo_len_after = fifo_len_before + chunk_pred_len;

        if (fifo_len_after <= fifo_max_len) {
          std::memcpy(
              fifo.data() + fifo_len_before * emb_dim,
              embs_ptr,
              static_cast<size_t>(chunk_pred_len * emb_dim) * sizeof(float));
          fifo_len = fifo_len_after;
        } else {
          // Build a temporary FIFO buffer: [old_fifo_valid, new_chunk]
          std::vector<float> fifo_tmp(
              static_cast<size_t>(fifo_len_after * emb_dim), 0.0f);
          if (fifo_len_before > 0) {
            std::memcpy(
                fifo_tmp.data(),
                fifo.data(),
                static_cast<size_t>(fifo_len_before * emb_dim) * sizeof(float));
          }
          std::memcpy(
              fifo_tmp.data() + fifo_len_before * emb_dim,
              embs_ptr,
              static_cast<size_t>(chunk_pred_len * emb_dim) * sizeof(float));

          int64_t pop_out_len = spkcache_update_period;
          pop_out_len = std::max(
              pop_out_len, chunk_pred_len - fifo_max_len + fifo_len_before);
          pop_out_len = std::min(pop_out_len, fifo_len_after);

          // Move the oldest `pop_out_len` frames from FIFO into speaker cache.
          append_to_fixed_cache(
              spkcache,
              spkcache_len,
              spkcache_max_len,
              emb_dim,
              fifo_tmp.data(),
              pop_out_len);

          // Keep the remaining FIFO frames.
          const int64_t new_fifo_len = fifo_len_after - pop_out_len;
          if (new_fifo_len > 0) {
            std::memcpy(
                fifo.data(),
                fifo_tmp.data() + pop_out_len * emb_dim,
                static_cast<size_t>(new_fifo_len * emb_dim) * sizeof(float));
          }
          // Zero out tail for cleanliness.
          if (new_fifo_len < fifo_max_len) {
            std::fill(
                fifo.begin() + static_cast<size_t>(new_fifo_len * emb_dim),
                fifo.end(),
                0.0f);
          }
          fifo_len = new_fifo_len;
        }
      }

      stt_feat = end_feat;
      step_idx += 1;
    }

    const int64_t num_diar_frames = static_cast<int64_t>(posteriors.size() / static_cast<size_t>(n_spk));
    ET_LOG(
        Info,
        "Produced %lld diar frames (%.2f sec)",
        static_cast<long long>(num_diar_frames),
        num_diar_frames * frame_sec);

    auto segments = segments_from_posteriors(
        posteriors, num_diar_frames, n_spk, frame_sec, FLAGS_threshold);

    std::cout << "Segments (threshold=" << FLAGS_threshold << ")\n";
    for (const auto& seg : segments) {
      std::cout << "speaker_" << seg.speaker << "  "
                << seg.start_sec << "  " << seg.end_sec << "\n";
    }

    return 0;
  } catch (const std::exception& e) {
    ET_LOG(Error, "Exception: %s", e.what());
    return 1;
  }
}
