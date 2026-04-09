"""
Export nvidia/diar_streaming_sortformer_4spk-v2.1 to ExecuTorch (portable ops only).

This exports two runtime methods into a single `model.pte`:
  - `preprocessor(audio_1d, audio_len) -> (features, features_len)`
      where `features` is time-major: [1, T_feat, feat_dim]
  - `model_step(chunk, chunk_len, spkcache, spkcache_len, fifo, fifo_len, lc, rc)
        -> (chunk_preds, chunk_embs, chunk_pred_len)`
      where:
        - `chunk` is time-major log-mel features for one step: [1, T_chunk, feat_dim]
        - `spkcache` / `fifo` are embedding caches: [1, L, emb_dim]
        - `lc` / `rc` are left/right context in diar frames (post-subsampling)
        - `chunk_preds` is fixed-size [1, chunk_len_max, n_spk] (padded with zeros)
        - `chunk_embs`  is fixed-size [1, chunk_len_max, emb_dim] (padded with zeros)
        - `chunk_pred_len` is the number of valid diar frames in this step.

The exported program also includes constant metadata methods (via `constant_methods`) so C++
doesn't need to hardcode model parameters.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from torch.export import Dim, export


@dataclass(frozen=True)
class StreamingConfig:
    spkcache_len: int
    fifo_len: int
    spkcache_update_period: int
    chunk_len: int
    chunk_left_context: int
    chunk_right_context: int


def _load_model(model_name_or_path: str):
    try:
        from nemo.collections.asr.models.sortformer_diar_models import (
            SortformerEncLabelModel,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import NeMo. Install NeMo (or set PYTHONPATH) to use this exporter."
        ) from e

    if model_name_or_path.endswith(".nemo"):
        model = SortformerEncLabelModel.restore_from(
            restore_path=model_name_or_path, map_location="cpu", strict=False
        )
    else:
        model = SortformerEncLabelModel.from_pretrained(
            model_name_or_path, map_location="cpu"
        )
    model.eval()
    model.freeze()

    # Streaming-safe preprocessor defaults (match NeMo streaming examples).
    if hasattr(model, "preprocessor") and hasattr(model.preprocessor, "featurizer"):
        model.preprocessor.featurizer.dither = 0.0
        model.preprocessor.featurizer.pad_to = 0

    return model


class PreprocessorWrapper(torch.nn.Module):
    def __init__(self, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor

    def forward(
        self, audio: torch.Tensor, length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # NeMo preprocessors expect (B, T). Export contract uses 1D audio.
        audio_signal = audio.unsqueeze(0)
        feats, feat_len = self.preprocessor(input_signal=audio_signal, length=length)
        # Convert to time-major (B, T, F) for easier slicing in C++.
        feats = feats.transpose(1, 2)
        return feats, feat_len


class SortformerStreamingStep(torch.nn.Module):
    """One streaming inference step of Sortformer, with fixed-shape outputs.

    This wrapper keeps the neural network in ExecuTorch and leaves cache update logic to C++.
    """

    def __init__(
        self,
        diar_model,
        cfg: StreamingConfig,
    ):
        super().__init__()
        self.diar_model = diar_model
        self.cfg = cfg

        self.spkcache_max_len = int(cfg.spkcache_len)
        self.fifo_max_len = int(cfg.fifo_len)
        self.chunk_len_max = int(cfg.chunk_len)
        self.chunk_total_diar = int(
            cfg.chunk_left_context + cfg.chunk_len + cfg.chunk_right_context
        )
        self.total_max_len = int(
            self.spkcache_max_len + self.fifo_max_len + self.chunk_total_diar
        )

    def _pack_spkcache_fifo_chunk(
        self,
        spkcache: torch.Tensor,
        spkcache_len: torch.Tensor,
        fifo: torch.Tensor,
        fifo_len: torch.Tensor,
        chunk_pre_encode: torch.Tensor,
        chunk_pre_encode_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # All tensors are batch=1 with fixed max shapes:
        #   spkcache: [1, spkcache_max_len, emb_dim], spkcache_len: [1]
        #   fifo:     [1, fifo_max_len, emb_dim],     fifo_len: [1]
        #   chunk:    [1, chunk_total_diar, emb_dim], chunk_len: [1]
        #
        # We need a packed tensor where *valid* frames are contiguous:
        #   [spkcache[:spk_len], fifo[:fifo_len], chunk[:chunk_len], padding...]

        combined = torch.cat([spkcache, fifo, chunk_pre_encode], dim=1)
        emb_dim = combined.size(-1)

        # Length scalars (0-dim tensors)
        spk_len0 = spkcache_len[0]
        fifo_len0 = fifo_len[0]
        chunk_len0 = chunk_pre_encode_len[0]
        total_len0 = spk_len0 + fifo_len0 + chunk_len0

        p = torch.arange(self.total_max_len, device=combined.device, dtype=torch.long)
        idx = torch.zeros_like(p)

        # spkcache part
        idx = torch.where(p < spk_len0, p, idx)

        # fifo part (packed immediately after spk_len0)
        fifo_mask = (p >= spk_len0) & (p < (spk_len0 + fifo_len0))
        fifo_src = (p - spk_len0) + self.spkcache_max_len
        idx = torch.where(fifo_mask, fifo_src, idx)

        # chunk part (packed immediately after spk_len0 + fifo_len0)
        chunk_mask = (p >= (spk_len0 + fifo_len0)) & (p < total_len0)
        chunk_src = (p - spk_len0 - fifo_len0) + self.spkcache_max_len + self.fifo_max_len
        idx = torch.where(chunk_mask, chunk_src, idx)

        idx_exp = idx.view(1, -1, 1).expand(1, -1, emb_dim)
        packed = torch.gather(combined, dim=1, index=idx_exp)

        valid = (p < total_len0).view(1, -1, 1)
        packed = packed * valid

        total_len = (spkcache_len + fifo_len + chunk_pre_encode_len).to(torch.int64)
        return packed, total_len

    def forward(
        self,
        chunk: torch.Tensor,
        chunk_len: torch.Tensor,
        spkcache: torch.Tensor,
        spkcache_len: torch.Tensor,
        fifo: torch.Tensor,
        fifo_len: torch.Tensor,
        lc: torch.Tensor,
        rc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Pre-encode chunk (subsample to diar frames).
        chunk_pre_encode, chunk_pre_encode_len = self.diar_model.encoder.pre_encode(
            x=chunk, lengths=chunk_len
        )
        chunk_pre_encode_len = chunk_pre_encode_len.to(torch.int64)

        # Pack [spkcache, fifo, chunk] into a contiguous sequence for the neural net.
        packed, packed_len = self._pack_spkcache_fifo_chunk(
            spkcache=spkcache,
            spkcache_len=spkcache_len,
            fifo=fifo,
            fifo_len=fifo_len,
            chunk_pre_encode=chunk_pre_encode,
            chunk_pre_encode_len=chunk_pre_encode_len,
        )

        # Run FastConformer encoder (bypass pre_encode because `packed` is already in embedding space).
        emb_seq, emb_seq_len = self.diar_model.frontend_encoder(
            processed_signal=packed,
            processed_signal_length=packed_len,
            bypass_pre_encode=True,
        )
        preds = self.diar_model.forward_infer(emb_seq=emb_seq, emb_seq_length=emb_seq_len)

        # Produce fixed-size per-step outputs (padded with zeros).
        chunk_pred_len = torch.clamp(
            (chunk_pre_encode_len - lc - rc), min=0, max=self.chunk_len_max
        ).to(torch.int64)

        # Indices 0..chunk_len_max-1
        i = torch.arange(self.chunk_len_max, device=chunk.device, dtype=torch.long)
        # Masks to avoid relying on dynamic slicing.
        valid_i = (i < chunk_pred_len[0]).view(1, -1, 1)

        # Chunk embeddings used to update caches in C++: chunk_pre_encode[:, lc:lc+chunk_pred_len]
        pos_chunk = (i + lc[0]).to(torch.long)
        pos_chunk = torch.clamp(pos_chunk, min=0, max=self.chunk_total_diar - 1)
        pos_chunk_exp = pos_chunk.view(1, -1, 1).expand(1, -1, chunk_pre_encode.size(-1))
        chunk_embs = torch.gather(chunk_pre_encode, dim=1, index=pos_chunk_exp) * valid_i

        # Chunk speaker posteriors: preds[:, spk_len+fifo_len+lc : + chunk_pred_len]
        base = (spkcache_len + fifo_len + lc).to(torch.int64)
        pos_pred = (i + base[0]).to(torch.long)
        pos_pred = torch.clamp(pos_pred, min=0, max=self.total_max_len - 1)
        pos_pred_exp = pos_pred.view(1, -1, 1).expand(1, -1, preds.size(-1))
        chunk_preds = torch.gather(preds, dim=1, index=pos_pred_exp) * valid_i

        return chunk_preds, chunk_embs, chunk_pred_len


def _export_programs(model, cfg: StreamingConfig, max_audio_sec: int) -> Tuple[Dict, Dict]:
    programs: Dict[str, torch.export.ExportedProgram] = {}

    sample_rate = int(model._cfg.preprocessor.sample_rate)
    window_stride = float(model._cfg.preprocessor.window_stride)
    subsampling_factor = int(model.encoder.subsampling_factor)
    feat_dim = int(model._cfg.preprocessor.features)
    emb_dim = int(model._cfg.sortformer_modules.fc_d_model)
    n_spk = int(model.sortformer_modules.n_spk)
    negative_init_val = float(getattr(model, "negative_init_val", -99.0))

    max_audio_samples = int(sample_rate * int(max_audio_sec))

    # Export preprocessor
    preprocessor_wrapper = PreprocessorWrapper(model.preprocessor)
    preprocessor_wrapper.eval()

    sample_audio = torch.randn(max_audio_samples, dtype=torch.float32)
    sample_length = torch.tensor([sample_audio.shape[0]], dtype=torch.int64)

    # NeMo feature extractors sometimes branch on CUDA availability (data-dependent paths).
    old_cuda_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    programs["preprocessor"] = export(
        preprocessor_wrapper,
        (sample_audio, sample_length),
        dynamic_shapes={
            "audio": {0: Dim("audio_len", min=1600, max=max_audio_samples)},
            "length": {},
        },
        strict=False,
    )
    torch.cuda.is_available = old_cuda_is_available

    # Export model_step (fixed shapes)
    step = SortformerStreamingStep(model, cfg)
    step.eval()

    max_chunk_feat_frames = (
        (cfg.chunk_left_context + cfg.chunk_len + cfg.chunk_right_context)
        * subsampling_factor
    )

    sample_chunk = torch.randn(1, max_chunk_feat_frames, feat_dim, dtype=torch.float32)
    sample_chunk_len = torch.tensor([max_chunk_feat_frames], dtype=torch.int64)

    sample_spkcache = torch.zeros(1, cfg.spkcache_len, emb_dim, dtype=torch.float32)
    sample_spkcache_len = torch.tensor([0], dtype=torch.int64)

    sample_fifo = torch.zeros(1, cfg.fifo_len, emb_dim, dtype=torch.float32)
    sample_fifo_len = torch.tensor([0], dtype=torch.int64)

    sample_lc = torch.tensor([cfg.chunk_left_context], dtype=torch.int64)
    sample_rc = torch.tensor([cfg.chunk_right_context], dtype=torch.int64)

    programs["model_step"] = export(
        step,
        (
            sample_chunk,
            sample_chunk_len,
            sample_spkcache,
            sample_spkcache_len,
            sample_fifo,
            sample_fifo_len,
            sample_lc,
            sample_rc,
        ),
        strict=False,
    )

    metadata = {
        "sample_rate": sample_rate,
        "window_stride": window_stride,
        "subsampling_factor": subsampling_factor,
        "feat_dim": feat_dim,
        "emb_dim": emb_dim,
        "n_spk": n_spk,
        "negative_init_val": negative_init_val,
        "spkcache_len": int(cfg.spkcache_len),
        "fifo_len": int(cfg.fifo_len),
        "spkcache_update_period": int(cfg.spkcache_update_period),
        "chunk_len": int(cfg.chunk_len),
        "chunk_left_context": int(cfg.chunk_left_context),
        "chunk_right_context": int(cfg.chunk_right_context),
        "max_chunk_feat_frames": int(max_chunk_feat_frames),
    }

    return programs, metadata


def _lower_to_executorch(programs: Dict, metadata: Dict):
    constant_methods = dict(metadata)
    et_prog = to_edge_transform_and_lower(
        programs,
        partitioner=[],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=constant_methods,
    )
    return et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=False,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/diar_streaming_sortformer_4spk-v2.1",
        help="NeMo model name (from_pretrained) or path to a .nemo file.",
    )
    parser.add_argument("--output-dir", type=str, default="./sortformer_diar_exports")
    parser.add_argument(
        "--max-audio-sec",
        type=int,
        default=60,
        help="Max audio duration (sec) used to bound preprocessor dynamic shapes during export.",
    )

    # Streaming parameters (match NeMo examples by default)
    parser.add_argument("--spkcache-len", type=int, default=188)
    parser.add_argument("--fifo-len", type=int, default=188)
    parser.add_argument("--spkcache-update-period", type=int, default=144)
    parser.add_argument("--chunk-len", type=int, default=6)
    parser.add_argument("--chunk-left-context", type=int, default=1)
    parser.add_argument("--chunk-right-context", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading diarization model: {args.model}")
    model = _load_model(args.model)

    cfg = StreamingConfig(
        spkcache_len=int(args.spkcache_len),
        fifo_len=int(args.fifo_len),
        spkcache_update_period=int(args.spkcache_update_period),
        chunk_len=int(args.chunk_len),
        chunk_left_context=int(args.chunk_left_context),
        chunk_right_context=int(args.chunk_right_context),
    )

    # Apply streaming parameter overrides on the NeMo model (for metadata + consistency).
    model.sortformer_modules.spkcache_len = cfg.spkcache_len
    model.sortformer_modules.fifo_len = cfg.fifo_len
    model.sortformer_modules.spkcache_update_period = cfg.spkcache_update_period
    model.sortformer_modules.chunk_len = cfg.chunk_len
    model.sortformer_modules.chunk_left_context = cfg.chunk_left_context
    model.sortformer_modules.chunk_right_context = cfg.chunk_right_context
    model.sortformer_modules._check_streaming_parameters()

    print("Exporting methods...")
    programs, metadata = _export_programs(model, cfg, max_audio_sec=int(args.max_audio_sec))

    print("Lowering to ExecuTorch (portable ops only)...")
    et = _lower_to_executorch(programs, metadata)

    pte_path = os.path.join(args.output_dir, "model.pte")
    with open(pte_path, "wb") as f:
        et.write_to_file(f)
    print(f"Saved: {pte_path}")
    print(f"Size: {os.path.getsize(pte_path) / (1024 * 1024):.1f} MB")

    print("Done.")


if __name__ == "__main__":
    main()

