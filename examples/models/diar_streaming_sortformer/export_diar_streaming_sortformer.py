"""
Export nvidia/diar_streaming_sortformer_4spk-v2.1 to ExecuTorch.

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

Notes:
  - The preprocessor is always lowered with the portable backend.
  - When `--backend metal` is selected, `model_step` is delegated to the Metal backend.
"""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager, nullcontext
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


@contextmanager
def _patch_inductor_for_metal_empty_strided_workaround():
    """Work around Inductor producing unsupported padded/odd strides.

    ExecuTorch's non-ATen Tensor wrapper currently enforces a dense/permuted
    stride invariant. AOTInductor sometimes emits `empty_strided` with padded
    or otherwise non-dense strides for layout/padding optimizations.

    Disable those optimizations during AOTInductor compilation so the generated
    wrapper requests dense strides only.
    """

    try:
        import torch._inductor.config as inductor_config
    except Exception:
        yield
        return

    changes = {
        # Primary knobs that lead to padded/odd `empty_strided` requests.
        "layout_optimization": False,
        "shape_padding": False,
        "comprehensive_padding": False,
        "inplace_padding": False,
        # Avoid preserving non-dense layouts across ops.
        "keep_output_stride": False,
        # Make padding a no-op even if any code path still consults it.
        "padding_alignment_bytes": 1,
        "padding_stride_threshold": 1 << 60,
    }

    filtered = {k: v for k, v in changes.items() if hasattr(inductor_config, k)}
    if not filtered:
        yield
        return

    print(f"  Applying torch._inductor.config patch for Metal: {filtered}")
    with inductor_config.patch(filtered):
        yield


def _install_nemo_no_bool_patches() -> None:
    """Monkey patches NeMo to avoid dtype=bool tensors during export.

    ExecuTorch Metal AOTI runtime currently doesn't support allocating bool tensors
    (and even if it did, some tensor construction/layout invariants can still abort).
    This patch rewrites:
      - SortformerModules.length_to_mask: returns a float 0/1 mask without lt/bool
      - form_attention_mask: produces the same NEG_INF attention mask without bool
      - ConformerEncoder._create_masks: produces float 0/1 masks (no bool/logical ops)
      - MaskedConvSequential._create_mask: produces float mask without lt/bool
      - ConformerConvolution / MultiHeadAttention: consume float masks without masked_fill(bool)

    This is intended to be semantics-equivalent for this diarization model.
    """

    try:
        from nemo.collections.asr.modules.sortformer_modules import SortformerModules
        from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
        from nemo.collections.asr.parts.submodules.conformer_modules import ConformerConvolution
        from nemo.collections.asr.parts.submodules.multi_head_attention import (
            INF_VAL as NEMO_INF_VAL,
            MultiHeadAttention,
        )
        from nemo.collections.asr.parts.submodules.subsampling import MaskedConvSequential
        from nemo.collections.common.parts import transformer_utils as nemo_transformer_utils
        import nemo.collections.common.parts as nemo_common_parts
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import NeMo modules required for monkey patching. "
            "Ensure you're running the exporter in the same environment used to install NeMo."
        ) from e

    def _prefix_mask_01(length_0d: torch.Tensor, positions_1d: torch.Tensor) -> torch.Tensor:
        # Returns an int64 vector with 1 where positions < length, else 0.
        length_0d = length_0d.to(dtype=torch.int64)
        positions_1d = positions_1d.to(dtype=torch.int64)
        return torch.clamp(length_0d - positions_1d, min=0, max=1)

    def _suffix_mask_01(offset_0d: torch.Tensor, positions_1d: torch.Tensor) -> torch.Tensor:
        # Returns an int64 vector with 1 where positions >= offset, else 0.
        offset_0d = offset_0d.to(dtype=torch.int64)
        positions_1d = positions_1d.to(dtype=torch.int64)
        # positions >= offset  <=>  not(positions < offset)
        return 1 - torch.clamp(offset_0d - positions_1d, min=0, max=1)

    @staticmethod
    def length_to_mask_no_bool(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
        # Original NeMo implementation uses `<` and returns bool.
        # Here we return a float 0/1 mask without emitting any bool ops.
        lengths64 = lengths.to(dtype=torch.int64)
        arange = torch.arange(max_length, device=lengths.device, dtype=torch.int64)
        mask_01 = _prefix_mask_01(lengths64.unsqueeze(1), arange)  # broadcast to (B, L)
        return mask_01.to(dtype=torch.float32)

    def form_attention_mask_no_bool(
        input_mask: torch.Tensor | None, diagonal: int | None = None
    ) -> torch.Tensor | None:
        # Mirrors nemo.collections.common.parts.transformer_utils.form_attention_mask
        # but avoids dtype=bool (no to(bool), no &, no tril(bool)).
        if input_mask is None:
            return None

        input_mask_f = input_mask.to(dtype=torch.float32)
        attn_mask = input_mask_f.unsqueeze(1)  # (B, 1, L)
        if diagonal is not None:
            L = input_mask_f.shape[1]
            attn_shape = (1, L, L)
            future_mask = torch.tril(
                torch.ones(attn_shape, dtype=torch.float32, device=input_mask.device),
                diagonal,
            )
            attn_mask = attn_mask * future_mask  # (B, L, L) via broadcast

        attention_mask = (1.0 - attn_mask) * nemo_transformer_utils.NEG_INF
        return attention_mask.unsqueeze(1)

    def masked_conv_create_mask_no_bool(self, tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Original NeMo implementation uses `<` to create a bool mask, then casts to float.
        # Build a 0/1 float mask without emitting bool ops.
        batch_size, _, time, features = tensor.shape
        p = torch.arange(time, device=tensor.device, dtype=torch.int64)
        valid_01 = _prefix_mask_01(lengths.to(dtype=torch.int64).unsqueeze(1), p)  # (B, T)
        return valid_01.unsqueeze(-1).expand(batch_size, time, features).to(dtype=tensor.dtype)

    def conformer_encoder_create_masks_no_bool(
        self,
        att_context_size,
        padding_length: torch.Tensor,
        max_audio_length: int,
        offset: torch.Tensor | None,
        device,
    ):
        # Return:
        #   pad_mask: float32 0/1 where 1 indicates padding (masked for conv)
        #   att_mask: float32 0/1 where 1 indicates masked attention positions (B, T, T) or None
        L = int(max_audio_length)
        p = torch.arange(0, L, device=device, dtype=torch.int64)

        valid_len_01 = _prefix_mask_01(padding_length.to(dtype=torch.int64).unsqueeze(1), p).to(
            dtype=torch.float32
        )  # (B, T)
        if offset is not None:
            valid_off_01 = _suffix_mask_01(offset.to(dtype=torch.int64).unsqueeze(1), p).to(dtype=torch.float32)
            valid_01 = valid_len_01 * valid_off_01
        else:
            valid_01 = valid_len_01

        pad_mask = (1.0 - valid_01).to(dtype=torch.float32)

        if self.self_attention_model == "rel_pos_local_attn":
            return pad_mask, None

        # Context-visibility mask as 0/1 float (1 = visible/allowed).
        att_allowed = torch.ones((1, L, L), device=device, dtype=torch.float32)

        if self.att_context_style == "regular":
            if att_context_size[0] >= 0:
                att_allowed = torch.triu(att_allowed, diagonal=-att_context_size[0])
            if att_context_size[1] >= 0:
                att_allowed = torch.tril(att_allowed, diagonal=att_context_size[1])
        elif self.att_context_style == "chunked_limited":
            # Keep logic equivalent to NeMo but avoid bool comparisons/logical ops.
            if att_context_size[1] == -1:
                if att_context_size[0] >= 0:
                    att_allowed = torch.triu(att_allowed, diagonal=-att_context_size[0])
            else:
                chunk_size = int(att_context_size[1]) + 1
                if chunk_size <= 0:
                    raise ValueError("chunk_size must be > 0 for chunked_limited attention")
                if att_context_size[0] >= 0:
                    left_chunks_num = int(att_context_size[0]) // chunk_size
                else:
                    left_chunks_num = 10000

                chunk_idx = torch.arange(0, L, device=device, dtype=torch.int64)
                chunk_idx = torch.div(chunk_idx, chunk_size, rounding_mode="trunc")
                diff_chunks = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)  # (T, T)

                # 1 if 0 <= diff_chunks <= left_chunks_num else 0
                ge0_01 = 1 - torch.clamp(-diff_chunks, min=0, max=1)
                le_left_01 = 1 - torch.clamp(diff_chunks - left_chunks_num, min=0, max=1)
                chunk_allowed = (ge0_01 * le_left_01).to(dtype=torch.float32)
                att_allowed = att_allowed * chunk_allowed.unsqueeze(0)

        # Apply padding mask to attention visibility.
        pad_allowed = valid_01.unsqueeze(1) * valid_01.unsqueeze(2)  # (B, T, T)
        att_allowed = att_allowed * pad_allowed  # broadcast (1, T, T) -> (B, T, T)
        att_mask = (1.0 - att_allowed).to(dtype=torch.float32)
        return pad_mask, att_mask

    def multihead_forward_attention_no_bool(
        self, value: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        # mask is expected to be float 0/1 with 1 indicating a masked position.
        n_batch = value.size(0)
        if mask is not None:
            mask_f = mask.to(dtype=scores.dtype).unsqueeze(1)  # (B, 1, T1, T2)
            scores = scores * (1.0 - mask_f) + mask_f * (-float(NEMO_INF_VAL))
            attn = torch.softmax(scores, dim=-1) * (1.0 - mask_f)
        else:
            attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)

    def conformer_convolution_forward_no_bool(self, x, pad_mask=None, cache=None):
        # Equivalent to original forward, but consumes a float 0/1 pad_mask without masked_fill(bool).
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)

        if self.pointwise_activation == "glu_":
            x = torch.nn.functional.glu(x, dim=1)
        else:
            x = self.pointwise_activation(x)

        if pad_mask is not None:
            pad_f = pad_mask.to(dtype=x.dtype).unsqueeze(1)  # (B, 1, T)
            x = x * (1.0 - pad_f)

        x = self.depthwise_conv(x, cache=cache)
        if cache is not None:
            x, cache = x

        if self.norm_type == "layer_norm":
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.batch_norm(x)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        if cache is None:
            return x
        else:
            return x, cache

    # Patch the functions used by the model.
    SortformerModules.length_to_mask = length_to_mask_no_bool  # type: ignore[assignment]
    nemo_transformer_utils.form_attention_mask = form_attention_mask_no_bool  # type: ignore[assignment]
    if hasattr(nemo_common_parts, "form_attention_mask"):
        nemo_common_parts.form_attention_mask = form_attention_mask_no_bool  # type: ignore[assignment]

    MaskedConvSequential._create_mask = masked_conv_create_mask_no_bool  # type: ignore[assignment]
    ConformerEncoder._create_masks = conformer_encoder_create_masks_no_bool  # type: ignore[assignment]
    MultiHeadAttention.forward_attention = multihead_forward_attention_no_bool  # type: ignore[assignment]
    ConformerConvolution.forward = conformer_convolution_forward_no_bool  # type: ignore[assignment]

    # Some NeMo modules import `form_attention_mask` into their module scope.
    # Patch those module-level references too (best-effort).
    for mod_name in (
        "nemo.collections.asr.modules.transformer.transformer_encoders",
        "nemo.collections.asr.modules.transformer.transformer_encoders_nlp",
        "nemo.collections.asr.modules.transformer.transformer_decoders",
        "nemo.collections.asr.modules.transformer.transformer_modules",
    ):
        try:
            mod = __import__(mod_name, fromlist=["form_attention_mask"])
            if hasattr(mod, "form_attention_mask"):
                setattr(mod, "form_attention_mask", form_attention_mask_no_bool)
        except Exception:
            pass


def _load_model(model_name_or_path: str):
    try:
        from nemo.collections.asr.models.sortformer_diar_models import (
            SortformerEncLabelModel,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import NeMo. Install NeMo (or set PYTHONPATH) to use this exporter."
        ) from e

    # Monkey patch NeMo to avoid bool tensor materialization during export.
    _install_nemo_no_bool_patches()

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


def _linear_bias_decomposition(input, weight, bias=None):
    """Decompose linear with bias into matmul + add.

    This avoids decompositions that can introduce reinterpret views with
    unsupported strides in ExecuTorch, and also avoids requiring addmm.
    """
    weight_t = torch.ops.aten.t.default(weight)
    out = torch.ops.aten.matmul.default(input, weight_t)
    if bias is not None:
        return torch.ops.aten.add.Tensor(out, bias)
    return out


def _create_metal_partitioners(programs: Dict) -> Tuple[Dict, Dict]:
    """Create Metal partitioners for all programs except preprocessor."""
    from executorch.backends.apple.metal.metal_backend import MetalBackend
    from executorch.backends.apple.metal.metal_partitioner import MetalPartitioner

    updated_programs = {}
    for name, ep in programs.items():
        if name == "preprocessor":
            updated_programs[name] = ep
            continue
        updated_programs[name] = ep.run_decompositions(
            {torch.ops.aten.linear.default: _linear_bias_decomposition}
        )

    partitioner = {}
    for name in updated_programs.keys():
        if name == "preprocessor":
            partitioner[name] = []
        else:
            compile_specs = [MetalBackend.generate_method_name_compile_spec(name)]
            partitioner[name] = [MetalPartitioner(compile_specs)]

    return partitioner, updated_programs


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

        # Avoid creating bool tensors (no comparisons, no where, no &).
        p = torch.arange(self.total_max_len, device=combined.device, dtype=torch.int64)

        # Build 0/1 prefix masks using arithmetic: mask(p < k) = clamp(k - p, 0, 1)
        spk_end = spk_len0.to(dtype=torch.int64)
        fifo_end = (spk_len0 + fifo_len0).to(dtype=torch.int64)
        total_end = total_len0.to(dtype=torch.int64)

        m_spk = torch.clamp(spk_end - p, min=0, max=1)
        m_spk_fifo = torch.clamp(fifo_end - p, min=0, max=1)
        m_all = torch.clamp(total_end - p, min=0, max=1)

        m_fifo = m_spk_fifo - m_spk
        m_chunk = m_all - m_spk_fifo

        fifo_src = (p - spk_end) + self.spkcache_max_len
        chunk_src = (p - fifo_end) + self.spkcache_max_len + self.fifo_max_len
        idx = (m_spk * p) + (m_fifo * fifo_src) + (m_chunk * chunk_src)

        idx_exp = idx.view(1, -1, 1).expand(1, -1, emb_dim)
        packed = torch.gather(combined, dim=1, index=idx_exp)

        valid = m_all.to(dtype=packed.dtype).view(1, -1, 1)
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
        # Masks to avoid relying on dynamic slicing, without emitting bool.
        chunk_pred_len0 = chunk_pred_len[0].to(dtype=torch.int64)
        valid_i = torch.clamp(chunk_pred_len0 - i.to(dtype=torch.int64), min=0, max=1).view(1, -1, 1)
        valid_i = valid_i.to(dtype=preds.dtype)

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
    # Preprocessor implementation details that are useful for true streaming.
    # Expose them via constant_methods so C++ can do correct audio/frame accounting.
    featurizer = getattr(model.preprocessor, "featurizer", None)
    hop_length_samples = int(getattr(featurizer, "hop_length", 0)) or int(
        round(window_stride * sample_rate)
    )
    win_length_samples = int(getattr(featurizer, "win_length", 0))
    if not win_length_samples:
        # Try common NeMo config fields. Fall back to 20ms if missing.
        if getattr(model._cfg.preprocessor, "n_window_size", None):
            win_length_samples = int(model._cfg.preprocessor.n_window_size)
        elif getattr(model._cfg.preprocessor, "window_size", None):
            win_length_samples = int(round(float(model._cfg.preprocessor.window_size) * sample_rate))
        else:
            win_length_samples = int(round(0.02 * sample_rate))
    n_fft = int(getattr(featurizer, "n_fft", 0))
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
        "hop_length_samples": hop_length_samples,
        "win_length_samples": win_length_samples,
        "n_fft": n_fft,
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


def _lower_to_executorch(programs: Dict, metadata: Dict, backend: str):
    constant_methods = dict(metadata)

    if backend == "metal":
        print("  Using Metal backend for model_step (preprocessor stays portable).")
        partitioner, programs = _create_metal_partitioners(programs)
        extract_delegate_segments = True
    elif backend == "portable":
        partitioner = []
        extract_delegate_segments = False
    else:
        raise ValueError(f"Unsupported backend: {backend!r}")

    with (
        _patch_inductor_for_metal_empty_strided_workaround()
        if backend == "metal"
        else nullcontext()
    ):
        et_prog = to_edge_transform_and_lower(
            programs,
            partitioner=partitioner,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
            ),
            constant_methods=constant_methods,
        )
        return et_prog.to_executorch(
            config=ExecutorchBackendConfig(
                extract_delegate_segments=extract_delegate_segments,
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
        "--backend",
        type=str,
        default="portable",
        choices=["portable", "metal"],
        help="Backend for acceleration (default: portable).",
    )
    parser.add_argument(
        "--max-audio-sec",
        type=int,
        default=60,
        help="Max audio duration (sec) used to bound preprocessor dynamic shapes during export.",
    )

    # TODO(matt): should these be exported as constants at all, or just runtime parameters?
    #  There's a chance we'll need to export methods at a lower level..?

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

    if args.backend == "metal":
        print("Lowering to ExecuTorch with Metal...")
    else:
        print("Lowering to ExecuTorch (portable ops only)...")
    et = _lower_to_executorch(programs, metadata, backend=args.backend)

    pte_path = os.path.join(args.output_dir, "model.pte")
    with open(pte_path, "wb") as f:
        et.write_to_file(f)
    print(f"Saved: {pte_path}")
    print(f"Size: {os.path.getsize(pte_path) / (1024 * 1024):.1f} MB")

    print("Done.")


if __name__ == "__main__":
    main()
