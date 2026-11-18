r"""Looped 48-Layer Dual-mHC + `+1.0B` Hybrid Multimodal MoE-Engram (`En1B-A256M`) for Spark-X2.6-VL.

Ports and unifies `Spark-X2.6-Recursion` (`v0.3`) with `SparkNative` multimodal conditioning:
1. **Lesser-Layer Recurrent Loop (`[12..24)` x 2 passes = `48` effective layers)**:
   - Sublayer Dual-mHC (`SublayerDualMHC`: closed-form 2x2 Sinkhorn Birkhoff routing + pass-indexed
     SwiGLU deliberation adapters, exact zero-gated Pass-2+ identity at Step 0).
   - `_VisitCache` redirecting per-visit KV cache writes to sequential BF16 virtual cache slots (`0..47`),
     while re-executing `Layer 23`'s `GatedVisualCrossAttentionBlock` on Pass 2 scaled by `fgain`.
2. **`+1.002B` Hybrid Multimodal MoE-Engram (`En1B-A256M` across 8 Macro-Block Anchors `3,7,11,15,19,23,27,31`)**:
   - Tier 1A (`GPU HBM` Hot N-Gram Cache): `8,192` buckets x `256` dim x 3 scales (`50.3M`, zero-init).
   - Tier 1B (`Pinned Host DRAM` Long-Tail N-Gram Bank): `32,768` buckets x `256` dim x 3 scales (`201.3M`,
     `0 GB` GPU VRAM, microbatch-accumulated Coalesced Sparse Row Adam `<0.5 ms/step`, persisted in `state_dict`).
   - Tier 2 (`GPU HBM` Always-Active Shared SwiGLU Expert): `2560 -> 1536 -> 2560` (`94.4M` active).
   - Tier 3 (`GPU HBM` 2D APOLLO-Compatible Orthogonal SwiGLU Routed Experts): `E=16, Top-K=4, rank=1368`
     (`537.9M` total / `134.5M` active) stored as 2D `ParameterList` matrices so APOLLO projects moments to
     rank 64 in BF16 merged grad buffers (saving `>7.5 GB` VRAM vs 3D AdamW), routed by Multimodal Hypersphere
     Cosine + Sigmoid Affinity (`W_route h_t + W_vis v_bar`), DeepSeek-V3 bias load balancing, and ST-MoE Z-Loss.
   - Strict Multimodal N-Gram Boundary Masking: non-text visual/audio positions never pollute causal N-gram hashes.
3. **Fused Triton Kernels & Runtime $O(n \log n)$ NoPE Log-Linear KV Cache**:
   - Fused Triton SwiGLU + Triton/Liger RMSNorm + BF16 GQA SDPA + runtime `patch_spark_attention_for_loglinear_kv`.
"""
from __future__ import annotations

import importlib
import inspect
import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    HAS_TRITON = False

try:
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction

    HAS_LIGER_RMSNORM = True
except Exception:
    LigerRMSNormFunction = None
    HAS_LIGER_RMSNORM = False


if HAS_TRITON:

    @triton.jit
    def _swiglu_fwd_kernel(gate_ptr, up_ptr, out_ptr, n_elements: int, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        g_raw = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
        g = g_raw.to(tl.float32)
        u = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-g))
        out = (g * sig) * u
        tl.store(out_ptr + offsets, out.to(g_raw.dtype), mask=mask)

    @triton.jit
    def _swiglu_bwd_kernel(dout_ptr, gate_ptr, up_ptr, dgate_ptr, dup_ptr, n_elements: int, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        dout = tl.load(dout_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g_raw = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
        u_raw = tl.load(up_ptr + offsets, mask=mask, other=0.0)
        g = g_raw.to(tl.float32)
        u = u_raw.to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-g))
        silu = g * sig
        dsilu = sig * (1.0 + g * (1.0 - sig))
        tl.store(dgate_ptr + offsets, (dout * u * dsilu).to(g_raw.dtype), mask=mask)
        tl.store(dup_ptr + offsets, (dout * silu).to(u_raw.dtype), mask=mask)


class _TritonSwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        gate_c = gate.contiguous()
        up_c = up.contiguous()
        out = torch.empty_like(gate_c)
        n_el = gate_c.numel()
        _swiglu_fwd_kernel[(triton.cdiv(n_el, 1024),)](gate_c, up_c, out, n_el, BLOCK_SIZE=1024)
        ctx.save_for_backward(gate_c, up_c)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        gate_c, up_c = ctx.saved_tensors
        go_c = grad_out.contiguous()
        dg = torch.empty_like(gate_c)
        du = torch.empty_like(up_c)
        n_el = gate_c.numel()
        _swiglu_bwd_kernel[(triton.cdiv(n_el, 1024),)](go_c, gate_c, up_c, dg, du, n_el, BLOCK_SIZE=1024)
        return dg, du


def triton_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if HAS_TRITON and gate.is_cuda and up.is_cuda and gate.numel() > 0:
        return _TritonSwiGLUFn.apply(gate, up)
    return F.silu(gate) * up


class _ImmediateMatmulFn(torch.autograd.Function):
    """2D/ND matmul (`x @ w`) that flushes `grad_w` straight into `MergedGradientAccumulator` across recurrent passes."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, w)
        ctx.w_ref = w if w.requires_grad else None
        return torch.matmul(x, w)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, w = ctx.saved_tensors
        grad_x = torch.matmul(grad_out, w.t()) if ctx.needs_input_grad[0] else None
        grad_w = None
        if ctx.needs_input_grad[1] and ctx.w_ref is not None:
            x2 = x.reshape(-1, x.shape[-1])
            go2 = grad_out.reshape(-1, grad_out.shape[-1])
            grad_w = x2.t().matmul(go2).to(w.dtype)
            consume = getattr(ctx.w_ref, "_immediate_accum_fn", None)
            if consume is not None and consume(grad_w):
                grad_w = None
        return grad_x, grad_w


class _ImmediateLinearFn(torch.autograd.Function):
    """Linear (`F.linear(x, w, b)`) that flushes `grad_w`/`grad_b` straight into `MergedGradientAccumulator`."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]) -> torch.Tensor:
        ctx.save_for_backward(x, w)
        ctx.w_ref = w if w.requires_grad else None
        ctx.b_ref = b if (b is not None and b.requires_grad) else None
        ctx.b_dtype = b.dtype if b is not None else None
        return F.linear(x, w, b)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, w = ctx.saved_tensors
        grad_x = torch.matmul(grad_out, w) if ctx.needs_input_grad[0] else None
        grad_w = None
        grad_b = None
        if ctx.needs_input_grad[1] and ctx.w_ref is not None:
            x2 = x.reshape(-1, x.shape[-1])
            go2 = grad_out.reshape(-1, grad_out.shape[-1])
            grad_w = go2.t().matmul(x2).to(w.dtype)
            consume_w = getattr(ctx.w_ref, "_immediate_accum_fn", None)
            if consume_w is not None and consume_w(grad_w):
                grad_w = None
        if ctx.needs_input_grad[2] and ctx.b_ref is not None:
            dims = tuple(range(grad_out.ndim - 1))
            grad_b = grad_out.float().sum(dim=dims).to(ctx.b_dtype)
            consume_b = getattr(ctx.b_ref, "_immediate_accum_fn", None)
            if consume_b is not None and consume_b(grad_b):
                grad_b = None
        return grad_x, grad_w, grad_b


def _imm_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled() and getattr(w, "_immediate_accum_fn", None) is not None:
        return _ImmediateMatmulFn.apply(x, w)
    return torch.matmul(x, w)


def _imm_linear(mod: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled() and getattr(mod.weight, "_immediate_accum_fn", None) is not None:
        return _ImmediateLinearFn.apply(x, mod.weight, mod.bias)
    return mod(x)


def fast_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if x.is_cuda and HAS_LIGER_RMSNORM and torch.is_grad_enabled():
        return LigerRMSNormFunction.apply(x, weight.to(x.dtype), eps)
    orig_dt = x.dtype
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (normed * weight.float()).to(orig_dt)


def _attn_accepts_position_embeddings(attn_mod: nn.Module) -> bool:
    cached = getattr(attn_mod, "_spark_vl_accepts_pos_emb", None)
    if cached is not None:
        return bool(cached)
    try:
        accepts = "position_embeddings" in inspect.signature(attn_mod.forward).parameters
    except (TypeError, ValueError):
        accepts = True
    attn_mod._spark_vl_accepts_pos_emb = accepts
    return accepts


class _CacheLayersView:
    def __init__(self, cache: Any, physical: int, virtual: int):
        self.cache = cache
        self.physical = physical
        self.virtual = virtual

    def __getitem__(self, index: int):
        if index != self.physical:
            raise IndexError("A physical layer may only access its own virtual cache slot")
        return self.cache.layers[self.virtual]


class _VisitCache:
    """Redirect physical `layer_idx` KV cache updates to sequential per-visit `virtual_idx` BF16 slots."""

    def __init__(self, cache: Any, physical_idx: int, virtual_idx: int, kv_dtype: Optional[torch.dtype] = None):
        self.cache = cache
        self.physical_idx = physical_idx
        self.virtual_idx = virtual_idx
        self.kv_dtype = kv_dtype
        if hasattr(cache, "layers"):
            self.layers = _CacheLayersView(cache, physical_idx, virtual_idx)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None):
        target = self.virtual_idx if layer_idx == self.physical_idx else layer_idx
        if self.kv_dtype is not None:
            if key_states.dtype != self.kv_dtype:
                key_states = key_states.to(self.kv_dtype)
            if value_states.dtype != self.kv_dtype:
                value_states = value_states.to(self.kv_dtype)
        elif key_states.dtype == torch.float32 and value_states.dtype in (torch.bfloat16, torch.float16):
            key_states = key_states.to(value_states.dtype)
        return self.cache.update(key_states, value_states, target, cache_kwargs)

    def __getattr__(self, name: str):
        return getattr(self.cache, name)


class _SimpleTupleVisitCache:
    """Lightweight sequential virtual-slot KV cache for non-HF synthetic test backbones."""

    def __init__(self) -> None:
        self.slots: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = []

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if 0 <= layer_idx < len(self.slots) and self.slots[layer_idx] is not None:
            return int(self.slots[layer_idx][0].shape[-2])
        return 0

    def get_slot(self, virtual_idx: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        return self.slots[virtual_idx] if 0 <= virtual_idx < len(self.slots) else None

    def set_slot(self, virtual_idx: int, kv: Optional[tuple[torch.Tensor, torch.Tensor]]) -> None:
        while len(self.slots) <= virtual_idx:
            self.slots.append(None)
        self.slots[virtual_idx] = kv


def _compress_nope_kv_loglinear(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sink_tokens: int = 32,
    local_window: int = 1024,
    base_block: int = 32,
    hip_top_blocks: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Log-Mass-Conserved Dyadic Fenwick Tree + HiP Top-M Exact Recall for Spark NoPE Full-Attention layers."""
    bsz, n_kv_heads, seq_k, head_dim = key.shape
    device = key.device
    dtype = query.dtype

    mid_start = min(sink_tokens, seq_k)
    mid_end = max(mid_start, seq_k - local_window)
    mid_len = mid_end - mid_start
    n_blocks = mid_len // base_block
    if n_blocks <= hip_top_blocks + 2:
        return key, value, torch.zeros((bsz, 1, 1, seq_k), dtype=dtype, device=device)

    usable_mid_end = mid_start + n_blocks * base_block
    k_sink, v_sink = key[:, :, :mid_start, :], value[:, :, :mid_start, :]
    k_local, v_local = key[:, :, usable_mid_end:, :], value[:, :, usable_mid_end:, :]

    k_mid = key[:, :, mid_start:usable_mid_end, :].view(bsz, n_kv_heads, n_blocks, base_block, head_dim)
    v_mid = value[:, :, mid_start:usable_mid_end, :].view(bsz, n_kv_heads, n_blocks, base_block, head_dim)
    centroid_k = k_mid.float().mean(dim=3)
    centroid_v = v_mid.float().mean(dim=3)

    q_ref = query[:, :, -min(4, query.shape[2]) :, :].float().mean(dim=2)
    if q_ref.shape[1] != n_kv_heads:
        groups = q_ref.shape[1] // n_kv_heads
        q_ref_kv = q_ref.view(bsz, n_kv_heads, groups, head_dim).mean(dim=2)
    else:
        q_ref_kv = q_ref
    block_scores = torch.einsum("bhd,bhnd->bn", q_ref_kv, centroid_k) / math.sqrt(head_dim)
    top_m = min(hip_top_blocks, n_blocks)
    _, top_idx = torch.topk(block_scores, k=top_m, dim=-1, sorted=True)

    top_set = set(top_idx[0].tolist())
    exact_k_list = [k_sink] + [k_mid[:, :, b_i, :, :] for b_i in sorted(top_set)] + [k_local]
    exact_v_list = [v_sink] + [v_mid[:, :, b_i, :, :] for b_i in sorted(top_set)] + [v_local]

    sum_k_list, sum_v_list, sum_log_mass = [], [], []
    i = 0
    while i < n_blocks:
        if i in top_set:
            i += 1
            continue
        span = 1
        while (i + span * 2 <= n_blocks) and all((i + j) not in top_set for j in range(span * 2)) and span < 16:
            span *= 2
        if span == 1:
            sum_k_list.append(centroid_k[:, :, i : i + 1, :].to(key.dtype))
            sum_v_list.append(centroid_v[:, :, i : i + 1, :].to(value.dtype))
        else:
            sum_k_list.append(centroid_k[:, :, i : i + span, :].mean(dim=2, keepdim=True).to(key.dtype))
            sum_v_list.append(centroid_v[:, :, i : i + span, :].mean(dim=2, keepdim=True).to(value.dtype))
        sum_log_mass.append(math.log(float(span * base_block)))
        i += span

    k_exact_all = torch.cat(exact_k_list, dim=2)
    v_exact_all = torch.cat(exact_v_list, dim=2)
    n_exact = k_exact_all.shape[2]
    if sum_k_list:
        k_out = torch.cat([k_exact_all, torch.cat(sum_k_list, dim=2)], dim=2)
        v_out = torch.cat([v_exact_all, torch.cat(sum_v_list, dim=2)], dim=2)
        bias = torch.zeros((bsz, 1, 1, n_exact + len(sum_log_mass)), dtype=dtype, device=device)
        bias[:, :, :, n_exact:] = torch.tensor(sum_log_mass, dtype=dtype, device=device).view(1, 1, 1, -1)
        return k_out, v_out, bias
    return k_exact_all, v_exact_all, torch.zeros((bsz, 1, 1, n_exact), dtype=dtype, device=device)


def patch_spark_attention_for_loglinear_kv(base_model: nn.Module, exact_threshold: int = 8192) -> int:
    """Install runtime-only O(n log n) NoPE KV compression + sliding-window cache slicing on Spark."""
    if not hasattr(base_model, "model") or not hasattr(base_model.model, "layers"):
        return 0
    layers = base_model.model.layers
    if not layers or not hasattr(layers[0], "self_attn"):
        return 0
    if not _attn_accepts_position_embeddings(layers[0].self_attn):
        return 0
    default_sw = int(getattr(getattr(base_model, "config", None), "sliding_window", 512) or 512)
    for lyr in layers:
        is_sl = (
            getattr(lyr, "layer_type", "") == "sliding_attention"
            or bool(getattr(lyr.self_attn, "sliding_window", None))
            or bool(getattr(lyr.self_attn, "is_sliding", False))
        )
        lyr.self_attn._sparkx_sliding_window = int(getattr(lyr.self_attn, "sliding_window", None) or default_sw) if is_sl else None

    attn_mod = importlib.import_module(type(layers[0]).__module__)
    if getattr(attn_mod, "_spark_vl_loglinear_patched", False):
        return len(layers)
    orig_repeat_kv = getattr(attn_mod, "repeat_kv", None)

    def sdpa_loglinear_forward(attention, query, key, value, attention_mask=None, scaling=None, dropout=0.0, **kwargs):
        qkv_proj = getattr(attention, "q_k_v_proj", None)
        target_dtype = getattr(qkv_proj, "weight", value).dtype if qkv_proj is not None else value.dtype
        if query.dtype != target_dtype:
            query = query.to(target_dtype)
        if key.dtype != target_dtype:
            key = key.to(target_dtype)
        if value.dtype != target_dtype:
            value = value.to(target_dtype)

        sw = getattr(attention, "_sparkx_sliding_window", None)
        bias_mask = None
        if sw is not None and attention_mask is None and query.shape[-2] == 1 and key.shape[-2] > int(sw):
            key = key[:, :, -int(sw) :, :]
            value = value[:, :, -int(sw) :, :]
        elif sw is None and not attention.training and query.shape[-2] <= 4 and key.shape[-2] > exact_threshold:
            key, value, bias_mask = _compress_nope_kv_loglinear(query, key, value)

        if orig_repeat_kv is not None and getattr(attention, "num_key_value_groups", 1) > 1:
            key = orig_repeat_kv(key, attention.num_key_value_groups)
            value = orig_repeat_kv(value, attention.num_key_value_groups)

        mask = attention_mask[:, :, :, : key.shape[-2]] if attention_mask is not None else None
        if mask is not None and mask.dtype != torch.bool and mask.dtype != query.dtype:
            mask = mask.to(query.dtype)
        if bias_mask is not None:
            mask = bias_mask if mask is None else (mask + bias_mask)
        use_causal = (mask is None) and (query.shape[-2] > 1) and (query.shape[-2] == key.shape[-2])
        result = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=dropout if attention.training else 0.0,
            is_causal=use_causal,
            scale=scaling,
        )
        return result, None

    prev_fwd = getattr(attn_mod, "eager_attention_forward", None)
    sdpa_loglinear_forward._spark_reference = getattr(prev_fwd, "_spark_reference", prev_fwd)
    sdpa_loglinear_forward._spark_native_sdpa = True
    attn_mod.eager_attention_forward = sdpa_loglinear_forward
    attn_mod._spark_vl_loglinear_patched = True
    return len(layers)


class MultimodalTieredMoEEngram(nn.Module):
    """Hybrid `+1.002B` Multimodal MoE-Engram (`En1B-A256M`: `125.3M` total / `~32.5M` active per anchor x 8 anchors).

    - `tier1a_hot`: GPU HBM Hot N-Gram Cache (`hot_buckets * 3 x eng_dim`, zero-init).
    - `tier1b_cold_weight`: Pinned Host DRAM Long-Tail N-Gram Bank (`cold_buckets * 3 x eng_dim`, zero-init,
      kept strictly on CPU via `_apply` override so it consumes `0` bytes of GPU VRAM while remaining in
      `model.state_dict()` for seamless `.safetensors` checkpointing).
    - `shared_*`: 100%-active SwiGLU Shared Expert (`width -> shared_rank -> width`).
    - `expert_gates` / `expert_ups` / `expert_downs`: `E = 16` orthogonal 2D SwiGLU Routed Memory Experts
      (`Top-K = 4`, `rank = 1368`) stored as 2D `ParameterList` matrices for rank-64 APOLLO + BF16 merged grad
      compatibility, routed by Multimodal Hypersphere Cosine + Sigmoid Affinity (`W_route h_t + W_vis v_bar`).
    - `project`: Zero-initialized residual projection (`max_abs_diff == 0.0` at Step 0).
    """

    def __init__(
        self,
        *,
        width: int = 2560,
        hot_buckets: int = 8192,
        cold_buckets: int = 32768,
        eng_dim: int = 256,
        shared_rank: int = 1536,
        num_experts: int = 16,
        top_k: int = 4,
        expert_rank: int = 1368,
        route_dim: int = 64,
        max_passes: int = 2,
        offload_tier1b_cpu: bool = True,
        param_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.width = int(width)
        self.hot_buckets = int(hot_buckets)
        self.cold_buckets = int(cold_buckets)
        self.total_buckets = self.hot_buckets + self.cold_buckets
        self.eng_dim = int(eng_dim)
        self.shared_rank = int(shared_rank)
        self.num_experts = int(num_experts)
        self.top_k = min(int(top_k), self.num_experts)
        self.expert_rank = int(expert_rank)
        self.route_dim = min(int(route_dim), self.width)
        self.max_passes = max(1, int(max_passes))
        self.offload_tier1b_cpu = bool(offload_tier1b_cpu)

        # Tier 1A: GPU HBM Hot Lexical Cache (zero-init)
        self.tier1a_hot = nn.Embedding(self.hot_buckets * 3, self.eng_dim, dtype=param_dtype)
        nn.init.zeros_(self.tier1a_hot.weight)
        self.scale_weights = nn.Parameter(torch.zeros(3, dtype=torch.float32))

        # Tier 1B: Pinned Host DRAM Long-Tail N-Gram Bank (persisted in state_dict, kept strictly on CPU!)
        if self.cold_buckets > 0 and self.offload_tier1b_cpu:
            pin = torch.cuda.is_available()
            cold_t = torch.zeros((self.cold_buckets * 3, self.eng_dim), dtype=param_dtype, device="cpu")
            if pin:
                cold_t = cold_t.pin_memory()
            self.register_buffer("tier1b_cold_weight", cold_t, persistent=True)
            self.tier1b_exp_avg = torch.zeros_like(cold_t)
            self.tier1b_exp_avg_sq = torch.zeros_like(cold_t)
            self.tier1b_step = 0
            self._microbatch_tier1b_grads: list[tuple[torch.Tensor, torch.Tensor]] = []
        elif self.cold_buckets > 0:
            self.tier1b_cold = nn.Embedding(self.cold_buckets * 3, self.eng_dim, dtype=param_dtype)
            nn.init.zeros_(self.tier1b_cold.weight)
            self.register_buffer("tier1b_cold_weight", None, persistent=False)
            self._microbatch_tier1b_grads = []
        else:
            self.register_buffer("tier1b_cold_weight", None, persistent=False)
            self._microbatch_tier1b_grads = []

        self.norm_eps = 1e-6
        self.norm_weight = nn.Parameter(torch.ones(self.width, dtype=torch.float32))
        self.lex_dropout = nn.Dropout(0.10)

        self.lexical_up = nn.Linear(self.eng_dim, self.width, bias=False, dtype=param_dtype)
        nn.init.normal_(self.lexical_up.weight, mean=0.0, std=0.02)

        # Tier 2: Always-Active Shared SwiGLU Expert
        self.shared_gate = nn.Linear(self.width, self.shared_rank, bias=False, dtype=param_dtype)
        self.shared_up = nn.Linear(self.width, self.shared_rank, bias=False, dtype=param_dtype)
        self.shared_down = nn.Linear(self.shared_rank, self.width, bias=False, dtype=param_dtype)
        nn.init.kaiming_uniform_(self.shared_gate.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.normal_(self.shared_down.weight, mean=0.0, std=0.01)

        # Tier 3: Multimodal Hypersphere Cosine + Sigmoid Router
        self.router_proj = nn.Linear(self.width, self.route_dim, bias=False, dtype=torch.float32)
        self.vis_router_proj = nn.Linear(self.width, self.route_dim, bias=False, dtype=torch.float32)
        self.expert_centroids = nn.Parameter(torch.empty(self.num_experts, self.route_dim, dtype=torch.float32))
        nn.init.orthogonal_(self.router_proj.weight)
        nn.init.zeros_(self.vis_router_proj.weight)  # Exact 0 shift at Step 0 and on pure-text inputs
        nn.init.orthogonal_(self.expert_centroids)
        self.router_log_temp = nn.Parameter(torch.tensor([math.log(0.15)], dtype=torch.float32))
        self.pass_router_bias = nn.Parameter(torch.zeros(self.max_passes, self.num_experts, dtype=torch.float32))
        self.register_buffer("expert_bias", torch.zeros(self.num_experts, dtype=torch.float32), persistent=True)
        self.register_buffer(
            "expert_load_ema",
            torch.full((self.num_experts,), 1.0 / self.num_experts, dtype=torch.float32),
            persistent=True,
        )

        # Tier 3: 2D APOLLO-Compatible Orthogonal SwiGLU Routed Memory Experts
        self.query_proj = nn.Linear(self.width, self.eng_dim, bias=False, dtype=param_dtype)
        nn.init.kaiming_uniform_(self.query_proj.weight, a=math.sqrt(5))
        self.expert_gates = nn.ParameterList(
            [nn.Parameter(torch.empty(self.eng_dim, self.expert_rank, dtype=param_dtype)) for _ in range(self.num_experts)]
        )
        self.expert_ups = nn.ParameterList(
            [nn.Parameter(torch.empty(self.eng_dim, self.expert_rank, dtype=param_dtype)) for _ in range(self.num_experts)]
        )
        self.expert_downs = nn.ParameterList(
            [nn.Parameter(torch.empty(self.expert_rank, self.width, dtype=param_dtype)) for _ in range(self.num_experts)]
        )
        self._init_orthogonal_experts()

        # Context-aware sigmoid gate & zero-initialized residual projection
        self.ctx_gate = nn.Linear(self.width, self.width, bias=True, dtype=param_dtype)
        nn.init.zeros_(self.ctx_gate.weight)
        nn.init.zeros_(self.ctx_gate.bias)
        self.project = nn.Linear(self.width, self.width, bias=False, dtype=param_dtype)
        nn.init.zeros_(self.project.weight)
        self.gain = nn.Parameter(torch.tensor([-1.8], dtype=torch.float32))

        self.last_router_z_loss: torch.Tensor = torch.tensor(0.0)
        self._pending_expert_loads: list[torch.Tensor] = []

    def _apply(self, fn):
        """Keep `tier1b_cold_weight` strictly in pinned Host DRAM even when `model.to('cuda')` is called."""
        cold_saved = self._buffers.get("tier1b_cold_weight", None)
        if cold_saved is not None and self.offload_tier1b_cpu:
            self._buffers["tier1b_cold_weight"] = None
        super()._apply(fn)
        if cold_saved is not None and self.offload_tier1b_cpu:
            cpu_t = cold_saved.detach().to(device="cpu", dtype=self.tier1a_hot.weight.dtype)
            if torch.cuda.is_available() and not cpu_t.is_pinned():
                cpu_t = cpu_t.pin_memory()
            self._buffers["tier1b_cold_weight"] = cpu_t
        return self

    @torch.no_grad()
    def _init_orthogonal_experts(self) -> None:
        for e in range(self.num_experts):
            g_f = torch.empty_like(self.expert_gates[e], dtype=torch.float32)
            u_f = torch.empty_like(self.expert_ups[e], dtype=torch.float32)
            d_f = torch.empty_like(self.expert_downs[e], dtype=torch.float32)
            nn.init.orthogonal_(g_f)
            nn.init.orthogonal_(u_f)
            nn.init.orthogonal_(d_f)
            self.expert_gates[e].copy_((g_f * 0.05).to(self.expert_gates[e].dtype))
            self.expert_ups[e].copy_((u_f * 0.05).to(self.expert_ups[e].dtype))
            self.expert_downs[e].copy_((d_f * 0.02).to(self.expert_downs[e].dtype))

    def total_capacity_elements(self) -> dict[str, int]:
        gpu_params = sum(p.numel() for p in self.parameters())
        cpu_tier1b = int(self.tier1b_cold_weight.numel()) if self.tier1b_cold_weight is not None else 0
        shared_active = (
            self.lexical_up.weight.numel()
            + self.shared_gate.weight.numel()
            + self.shared_up.weight.numel()
            + self.shared_down.weight.numel()
            + self.query_proj.weight.numel()
            + self.ctx_gate.weight.numel()
            + self.ctx_gate.bias.numel()
            + self.project.weight.numel()
        )
        per_expert = self.eng_dim * self.expert_rank * 2 + self.expert_rank * self.width
        routed_active = self.top_k * per_expert
        return {
            "gpu_parameters": gpu_params,
            "cpu_tier1b_elements": cpu_tier1b,
            "total_elements": gpu_params + cpu_tier1b,
            "active_compute_elements": shared_active + routed_active,
        }

    @torch.no_grad()
    def engram_telemetry(self) -> dict[str, Any]:
        ema = self.expert_load_ema.detach().float()
        mean_l = float(ema.mean().item()) or 1e-6
        cv = float(ema.std(unbiased=False).item()) / mean_l
        t1b_nz = 0
        if self.tier1b_cold_weight is not None:
            t1b_nz = int((self.tier1b_cold_weight.abs().amax(dim=-1) > 0).sum().item())
        return {
            "gain_sigmoid": round(float(torch.sigmoid(self.gain.float()).item()), 5),
            "project_weight_norm": round(float(self.project.weight.detach().float().norm().item()), 5),
            "expert_load_cv": round(cv, 4),
            "min_load_ratio": round(float(ema.min().item()) / mean_l, 4),
            "max_load_ratio": round(float(ema.max().item()) / mean_l, 4),
            "tier1b_step": int(getattr(self, "tier1b_step", 0)),
            "tier1b_nonzero_rows": t1b_nz,
        }

    def tier1b_optim_state_dict(self) -> dict[str, Any]:
        if not self.offload_tier1b_cpu or self.tier1b_cold_weight is None:
            return {}
        return {
            "step": int(self.tier1b_step),
            "exp_avg": self.tier1b_exp_avg.detach().cpu().clone(),
            "exp_avg_sq": self.tier1b_exp_avg_sq.detach().cpu().clone(),
        }

    def load_tier1b_optim_state_dict(self, state: dict[str, Any]) -> None:
        if not state or not self.offload_tier1b_cpu or self.tier1b_cold_weight is None:
            return
        self.tier1b_step = int(state.get("step", 0))
        if "exp_avg" in state and state["exp_avg"].shape == self.tier1b_cold_weight.shape:
            self.tier1b_exp_avg = state["exp_avg"].detach().to(device="cpu", dtype=self.tier1b_cold_weight.dtype).clone()
        if "exp_avg_sq" in state and state["exp_avg_sq"].shape == self.tier1b_cold_weight.shape:
            self.tier1b_exp_avg_sq = state["exp_avg_sq"].detach().to(device="cpu", dtype=self.tier1b_cold_weight.dtype).clone()

    @staticmethod
    def compute_ngram_indices(
        input_ids: torch.Tensor,
        total_buckets: int,
        prev_ids: Optional[torch.Tensor] = None,
        text_token_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deterministic causal hashing over `(t-2, t-1, t)` with multimodal image-span isolation."""
        ids = input_ids.long()
        if text_token_mask is not None:
            ids = torch.where(text_token_mask, ids, torch.zeros_like(ids))
        seq_len = ids.shape[1]
        if prev_ids is None or prev_ids.numel() == 0:
            p1 = F.pad(ids[:, :-1], (1, 0), value=0)
            p2 = F.pad(ids[:, :-2], (2, 0), value=0) if seq_len >= 2 else torch.zeros_like(ids)
            if text_token_mask is not None:
                m1 = F.pad(text_token_mask[:, :-1], (1, 0), value=False)
                m2 = F.pad(text_token_mask[:, :-2], (2, 0), value=False) if seq_len >= 2 else torch.zeros_like(text_token_mask)
                p1 = torch.where(m1, p1, torch.zeros_like(p1))
                p2 = torch.where(m2, p2, torch.zeros_like(p2))
        else:
            prev = prev_ids.long()
            last1 = prev[:, -1:]
            p1 = torch.cat([last1, ids[:, :-1]], dim=1) if seq_len > 1 else last1
            last2 = prev[:, -2:-1] if prev.shape[1] >= 2 else torch.zeros_like(last1)
            p2 = last2 if seq_len == 1 else torch.cat([last2, last1, ids[:, :-2]], dim=1)
        bi = (p1 * 1315423911 + ids * 2654435761) % total_buckets
        tri = (p2 * 97531 + p1 * 1315423911 + ids * 2654435761) % total_buckets
        skip = (p2 * 65537 + ids * 2654435761) % total_buckets
        return bi, tri, skip

    def _lookup_tier1_lexical(
        self,
        ngram_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        device: torch.device,
        param_dtype: torch.dtype,
        text_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bi, tri, skip = ngram_indices
        sw = F.softmax(self.scale_weights, dim=0).to(param_dtype)
        hot_b = self.hot_buckets

        e_bi = self.tier1a_hot(bi.clamp(max=hot_b - 1))
        e_tri = self.tier1a_hot(tri.clamp(max=hot_b - 1) + hot_b)
        e_skip = self.tier1a_hot(skip.clamp(max=hot_b - 1) + 2 * hot_b)

        if self.cold_buckets > 0:
            is_cold_bi = bi >= hot_b
            is_cold_tri = tri >= hot_b
            is_cold_skip = skip >= hot_b
            cold_b = self.cold_buckets
            c_bi = (bi - hot_b).clamp(min=0, max=cold_b - 1)
            c_tri = (tri - hot_b).clamp(min=0, max=cold_b - 1) + cold_b
            c_skip = (skip - hot_b).clamp(min=0, max=cold_b - 1) + 2 * cold_b

            if self.offload_tier1b_cpu and self.tier1b_cold_weight is not None:
                stacked = torch.stack([c_bi, c_tri, c_skip], dim=0)
                uniq_gpu, inv = torch.unique(stacked, return_inverse=True)
                uniq_cpu = uniq_gpu.cpu()
                fetched_cpu = self.tier1b_cold_weight[uniq_cpu]
                fetched_gpu = fetched_cpu.to(device=device, dtype=param_dtype, non_blocking=True)
                if self.training and torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters()):
                    fetched_gpu = fetched_gpu.detach().requires_grad_(True)
                    rows_ref = uniq_cpu

                    def _capture_grad(grad: torch.Tensor, _r=rows_ref) -> None:
                        self._microbatch_tier1b_grads.append((_r, grad.detach().to(device="cpu", dtype=torch.float32)))

                    fetched_gpu.register_hook(_capture_grad)
                gathered = fetched_gpu[inv]
                c_e_bi, c_e_tri, c_e_skip = gathered[0], gathered[1], gathered[2]
            else:
                c_e_bi = self.tier1b_cold(c_bi)
                c_e_tri = self.tier1b_cold(c_tri)
                c_e_skip = self.tier1b_cold(c_skip)

            e_bi = torch.where(is_cold_bi.unsqueeze(-1), c_e_bi, e_bi)
            e_tri = torch.where(is_cold_tri.unsqueeze(-1), c_e_tri, e_tri)
            e_skip = torch.where(is_cold_skip.unsqueeze(-1), c_e_skip, e_skip)

        out = sw[0] * e_bi + sw[1] * e_tri + sw[2] * e_skip
        if text_token_mask is not None:
            out = out * text_token_mask.unsqueeze(-1).to(out.dtype)
        return out

    @torch.no_grad()
    def zero_tier1b_buffers(self) -> None:
        self._microbatch_tier1b_grads.clear()
        self._pending_expert_loads.clear()

    @torch.no_grad()
    def step_tier1b_sparse_adam(
        self,
        *,
        lr: float = 1.2e-4,
        denom: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
    ) -> None:
        """Coalesced sparse Adam update on pinned Host DRAM across accumulated microbatches + Router EMA."""
        if self._pending_expert_loads:
            load = torch.stack(self._pending_expert_loads, dim=0).mean(dim=0).to(
                device=self.expert_load_ema.device, dtype=torch.float32
            )
            self._pending_expert_loads.clear()
            if load.numel() == self.num_experts and torch.isfinite(load).all():
                self.expert_load_ema.lerp_(load, 0.05)
                target_load = 1.0 / float(self.num_experts)
                imbalance = (target_load - self.expert_load_ema) / max(1e-6, target_load)
                step_b = 0.0015 * (0.5 * torch.sign(imbalance) + 0.5 * imbalance.clamp(-2.0, 2.0))
                self.expert_bias.add_(step_b).clamp_(-0.5, 0.5)

        if not self.offload_tier1b_cpu or self.tier1b_cold_weight is None or not self._microbatch_tier1b_grads:
            self._microbatch_tier1b_grads.clear()
            return

        all_rows = torch.cat([r for r, _ in self._microbatch_tier1b_grads], dim=0)
        all_grads = torch.cat([g for _, g in self._microbatch_tier1b_grads], dim=0)
        self._microbatch_tier1b_grads.clear()

        uniq_rows, inv = torch.unique(all_rows, return_inverse=True)
        coal_grad = torch.zeros((uniq_rows.numel(), self.eng_dim), dtype=torch.float32, device="cpu")
        coal_grad.index_add_(0, inv, all_grads)
        coal_grad.div_(max(1.0, float(denom)))
        if not torch.isfinite(coal_grad).all():
            return

        if self.tier1b_exp_avg.shape != self.tier1b_cold_weight.shape:
            self.tier1b_exp_avg = torch.zeros_like(self.tier1b_cold_weight)
            self.tier1b_exp_avg_sq = torch.zeros_like(self.tier1b_cold_weight)

        self.tier1b_step += 1
        m = self.tier1b_exp_avg[uniq_rows].float()
        v = self.tier1b_exp_avg_sq[uniq_rows].float()
        m.mul_(beta1).add_(coal_grad, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(coal_grad, coal_grad, value=1.0 - beta2)
        bc1 = 1.0 - (beta1 ** self.tier1b_step)
        bc2 = 1.0 - (beta2 ** self.tier1b_step)
        step_size = float(lr) / bc1
        denom_v = (v.sqrt() / math.sqrt(bc2)).add_(eps)
        w = self.tier1b_cold_weight[uniq_rows].float()
        w.addcdiv_(m, denom_v, value=-step_size)
        self.tier1b_exp_avg[uniq_rows] = m.to(self.tier1b_exp_avg.dtype)
        self.tier1b_exp_avg_sq[uniq_rows] = v.to(self.tier1b_exp_avg_sq.dtype)
        self.tier1b_cold_weight[uniq_rows] = w.to(self.tier1b_cold_weight.dtype)

    @torch.no_grad()
    def _route_dispatch_plan(
        self,
        hidden_states: torch.Tensor,
        vis_summary: Optional[torch.Tensor],
        pass_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
        """Compute discrete Top-K routing plan outside `checkpoint` so FP8 activation hooks never alter expert shapes."""
        param_dtype = self.project.weight.dtype
        h_norm_f = fast_rmsnorm(hidden_states.detach().to(param_dtype), self.norm_weight.detach(), self.norm_eps).float()
        p_idx = min(max(int(pass_idx), 0), self.max_passes - 1)
        route_in = self.router_proj(h_norm_f)
        if vis_summary is not None:
            route_in = route_in + self.vis_router_proj(vis_summary.detach().float())
        u = F.normalize(route_in, p=2, dim=-1, eps=1e-6)
        c = F.normalize(self.expert_centroids.detach(), p=2, dim=-1, eps=1e-6)
        temp = self.router_log_temp.detach().exp().clamp(min=0.03, max=2.0)
        router_logits = (torch.matmul(u, c.t()) / temp) + self.pass_router_bias.detach()[p_idx]
        scores = torch.sigmoid(router_logits)
        _, topk_idx = torch.topk(scores + self.expert_bias, k=self.top_k, dim=-1)

        if self.training and torch.is_grad_enabled():
            # Note: `no_grad()` disables grad inside this helper; check caller via `self.training` in `forward_with_z_loss`.
            pass

        n_tokens = int(hidden_states.shape[0] * hidden_states.shape[1])
        flat_e = topk_idx.reshape(n_tokens * self.top_k)
        sorted_e, sort_perm = torch.sort(flat_e)
        token_rows = torch.div(sort_perm, self.top_k, rounding_mode="floor")
        counts = tuple(int(x) for x in torch.bincount(sorted_e, minlength=self.num_experts).tolist())
        return topk_idx, token_rows, sort_perm, counts

    def _forward_core(
        self,
        hidden_states: torch.Tensor,
        e_lexical: torch.Tensor,
        vis_summary: Optional[torch.Tensor],
        topk_idx: torch.Tensor,
        token_rows: torch.Tensor,
        sort_perm: torch.Tensor,
        counts: tuple[int, ...],
        pass_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_dtype = hidden_states.dtype
        param_dtype = self.project.weight.dtype
        lex_dropped = self.lex_dropout(e_lexical)
        lex_hidden = _imm_linear(self.lexical_up, lex_dropped)

        h_norm = fast_rmsnorm(hidden_states.to(param_dtype), self.norm_weight, self.norm_eps)
        h_norm_f = h_norm.float()

        # Tier 2: 100%-active Shared SwiGLU Expert
        h_fused = h_norm + lex_hidden
        shared_out = _imm_linear(
            self.shared_down,
            triton_swiglu(_imm_linear(self.shared_gate, h_fused), _imm_linear(self.shared_up, h_fused)),
        )

        # Tier 3: Multimodal Hypersphere Cosine Router (`h_t + W_vis v_bar`) using frozen forward dispatch plan
        p_idx = min(max(int(pass_idx), 0), self.max_passes - 1)
        route_in = _imm_linear(self.router_proj, h_norm_f)
        if vis_summary is not None:
            route_in = route_in + _imm_linear(self.vis_router_proj, vis_summary.float())
        u = F.normalize(route_in, p=2, dim=-1, eps=1e-6)
        c = F.normalize(self.expert_centroids, p=2, dim=-1, eps=1e-6)
        temp = self.router_log_temp.exp().clamp(min=0.03, max=2.0)
        router_logits = (torch.matmul(u, c.t()) / temp) + self.pass_router_bias[p_idx]

        z_curr = torch.logsumexp(router_logits, dim=-1).pow(2).mean()

        scores = torch.sigmoid(router_logits)
        topk_raw = torch.gather(scores, dim=-1, index=topk_idx)
        topk_gates = (topk_raw / (topk_raw.sum(dim=-1, keepdim=True) + 1e-6)).to(param_dtype)

        q = _imm_linear(self.query_proj, h_norm) + lex_dropped
        flat_q = q.reshape(-1, self.eng_dim)
        n_tokens = flat_q.shape[0]
        flat_gates = topk_gates.reshape(n_tokens, self.top_k)

        sorted_q = flat_q.index_select(0, token_rows)
        sorted_g = flat_gates.reshape(-1).index_select(0, sort_perm).unsqueeze(-1)
        out_chunks = []
        offset = 0
        for e_id, cnt in enumerate(counts):
            if cnt == 0:
                continue
            q_e = sorted_q[offset : offset + cnt]
            g_e = sorted_g[offset : offset + cnt]
            offset += cnt
            act_e = triton_swiglu(_imm_matmul(q_e, self.expert_gates[e_id]), _imm_matmul(q_e, self.expert_ups[e_id]))
            out_chunks.append(_imm_matmul(act_e, self.expert_downs[e_id]) * g_e)
        routed_flat = torch.zeros((n_tokens, self.width), dtype=param_dtype, device=flat_q.device)
        if out_chunks:
            routed_flat.index_add_(0, token_rows, torch.cat(out_chunks, dim=0))

        routed_out = routed_flat.view_as(shared_out)
        combined = 0.5 * lex_hidden + shared_out + routed_out
        ctx = (2.0 * torch.sigmoid(_imm_linear(self.ctx_gate, h_norm).float())).to(param_dtype)
        projected = _imm_linear(self.project, combined * ctx)
        gate = torch.sigmoid(self.gain).to(out_dtype)
        return projected.to(out_dtype) * gate, z_curr

    def forward_with_z_loss(
        self,
        hidden_states: torch.Tensor,
        *,
        ngram_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        pass_idx: int = 0,
        vis_summary: Optional[torch.Tensor] = None,
        text_token_mask: Optional[torch.Tensor] = None,
        use_grad_checkpoint: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Tier-1 lookup + discrete Top-K dispatch plan stay outside `checkpoint` so Host DRAM sparse row hooks
        # fire strictly once and COAT FP8 activation compression never perturbs per-expert token counts!
        param_dtype = self.project.weight.dtype
        e_lexical = self._lookup_tier1_lexical(
            ngram_indices, hidden_states.device, param_dtype, text_token_mask=text_token_mask
        )
        topk_idx, token_rows, sort_perm, counts = self._route_dispatch_plan(hidden_states, vis_summary, pass_idx)
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                one_hot = F.one_hot(topk_idx, num_classes=self.num_experts).float()
                load = one_hot.sum(dim=-2).mean(dim=(0, 1)) / float(self.top_k)
                self._pending_expert_loads.append(load.detach())

        can_ckpt = (
            use_grad_checkpoint
            and self.training
            and torch.is_grad_enabled()
            and (hidden_states.requires_grad or any(p.requires_grad for p in self.parameters()))
        )
        if can_ckpt:
            delta, z_curr = checkpoint(
                self._forward_core,
                hidden_states,
                e_lexical,
                vis_summary,
                topk_idx,
                token_rows,
                sort_perm,
                counts,
                pass_idx,
                use_reentrant=False,
            )
        else:
            delta, z_curr = self._forward_core(
                hidden_states,
                e_lexical,
                vis_summary,
                topk_idx,
                token_rows,
                sort_perm,
                counts,
                pass_idx,
            )
        self.last_router_z_loss = z_curr.detach()
        return delta, z_curr

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        ngram_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        pass_idx: int = 0,
        vis_summary: Optional[torch.Tensor] = None,
        text_token_mask: Optional[torch.Tensor] = None,
        use_grad_checkpoint: bool = False,
    ) -> torch.Tensor:
        delta, _ = self.forward_with_z_loss(
            hidden_states,
            ngram_indices=ngram_indices,
            pass_idx=pass_idx,
            vis_summary=vis_summary,
            text_token_mask=text_token_mask,
            use_grad_checkpoint=use_grad_checkpoint,
        )
        return delta


class SublayerDualMHC(nn.Module):
    """Closed-form 2x2 Sinkhorn Birkhoff Sublayer Dual-mHC with Pass-2 Visual Cross-Attention Re-Attendance."""

    def __init__(
        self,
        *,
        width: int = 2560,
        pass_adapter_rank: int = 16,
        max_passes: int = 2,
        rmsnorm_eps: float = 1e-6,
    ):
        super().__init__()
        self.width = int(width)
        self.pass_adapter_rank = int(pass_adapter_rank)
        self.max_passes = max(1, int(max_passes))
        self.norm_eps = float(rmsnorm_eps)

        self.mix_logits = nn.Parameter(torch.tensor([[5.5, -5.5], [-5.5, 5.5]], dtype=torch.float32))
        self.read_logits = nn.Parameter(torch.tensor([6.0, -6.0], dtype=torch.float32))
        self.write_logits = nn.Parameter(torch.tensor([6.0, -6.0], dtype=torch.float32))
        self.extra_gain = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        if self.max_passes >= 3:
            self.p3_extra_gain = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        else:
            self.register_parameter("p3_extra_gain", None)

        self.ffn_mix_logits = nn.Parameter(torch.tensor([[5.5, -5.5], [-5.5, 5.5]], dtype=torch.float32))
        self.ffn_read_logits = nn.Parameter(torch.tensor([6.0, -6.0], dtype=torch.float32))
        self.ffn_write_logits = nn.Parameter(torch.tensor([6.0, -6.0], dtype=torch.float32))
        self.ffn_extra_gain = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        if self.max_passes >= 3:
            self.p3_ffn_extra_gain = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        else:
            self.register_parameter("p3_ffn_extra_gain", None)

        self.pass_routing_bias = nn.Parameter(torch.zeros(self.max_passes, 2, 8, dtype=torch.float32))

        flat_dim = 2 * self.width
        self.norm_weight = nn.Parameter(torch.ones(flat_dim, dtype=torch.float32))
        self.dynamic_proj = nn.Linear(flat_dim, 8, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.dynamic_proj.weight)

        self.ffn_norm_weight = nn.Parameter(torch.ones(flat_dim, dtype=torch.float32))
        self.ffn_dynamic_proj = nn.Linear(flat_dim, 8, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.ffn_dynamic_proj.weight)

        if self.pass_adapter_rank > 0:
            self.pass_down = nn.ModuleList(
                [nn.Linear(self.width, self.pass_adapter_rank, bias=False, dtype=torch.float32) for _ in range(self.max_passes)]
            )
            self.pass_up = nn.ModuleList(
                [nn.Linear(self.pass_adapter_rank, self.width, bias=False, dtype=torch.float32) for _ in range(self.max_passes)]
            )
            for p_idx in range(self.max_passes):
                nn.init.kaiming_uniform_(self.pass_down[p_idx].weight, a=math.sqrt(5))
                nn.init.zeros_(self.pass_up[p_idx].weight)
        else:
            self.pass_down = None
            self.pass_up = None

    @torch.no_grad()
    def controller_telemetry(self) -> dict[str, Any]:
        g_p2_a = float(torch.tanh(self.extra_gain.float()).item())
        g_p2_f = float(torch.tanh(self.ffn_extra_gain.float()).item())
        r_a = self.read_logits.float().softmax(dim=0)
        w_a = self.write_logits.float().softmax(dim=0)
        return {
            "gain_p2_attn": round(g_p2_a, 5),
            "gain_p2_ffn": round(g_p2_f, 5),
            "read_s1_prob_attn": round(float(r_a[1].item()), 5),
            "write_s0_prob_attn": round(float(w_a[0].item()), 5),
        }

    def _compute_routing_scalars(self, s0: torch.Tensor, s1: torch.Tensor, pass_idx: int = 0, is_ffn: bool = False):
        p_idx = min(max(0, int(pass_idx)), self.max_passes - 1)
        sub_idx = 1 if is_ffn else 0
        base_r = self.ffn_read_logits if is_ffn else self.read_logits
        base_w = self.ffn_write_logits if is_ffn else self.write_logits
        base_m = self.ffn_mix_logits if is_ffn else self.mix_logits
        proj = self.ffn_dynamic_proj if is_ffn else self.dynamic_proj
        nw = self.ffn_norm_weight if is_ffn else self.norm_weight
        p_bias = self.pass_routing_bias[p_idx, sub_idx].float()

        with torch.autocast(device_type=s0.device.type, enabled=False):
            flat = torch.cat((s0, s1), dim=-1)
            rms = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + self.norm_eps)
            dyn = proj(flat * rms * nw) + p_bias.view(1, 1, -1)
            r_log = base_r.float() + dyn[..., 0:2]
            w_log = base_w.float() + dyn[..., 2:4]
            m_log = (base_m.float().reshape(1, 1, 4) + dyn[..., 4:8]).clamp(min=-30.0, max=30.0)
            r1 = torch.sigmoid(r_log[..., 1:2] - r_log[..., 0:1])
            w1 = torch.sigmoid(w_log[..., 1:2] - w_log[..., 0:1])
            diag_adv = 0.5 * ((m_log[..., 0:1] + m_log[..., 3:4]) - (m_log[..., 1:2] + m_log[..., 2:3]))
            p_swap = 1.0 - torch.sigmoid(diag_adv)
            return r1, 1.0 - w1, w1, p_swap

    def _apply_pass_adapter(self, x: torch.Tensor, pass_idx: int) -> torch.Tensor:
        if self.pass_down is None or self.pass_up is None:
            return torch.zeros_like(x, dtype=torch.float32)
        p_idx = min(max(0, int(pass_idx)), self.max_passes - 1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.norm_eps)
            return self.pass_up[p_idx](F.silu(self.pass_down[p_idx](normed)))

    def forward(
        self,
        streams: torch.Tensor,
        layer: nn.Module,
        position_embeddings: Any,
        attention_mask: Any,
        *,
        raw_attention_mask_2d: Optional[torch.Tensor] = None,
        extra: bool = False,
        pass_idx: int = 0,
        past_key_values: Any = None,
        virtual_idx: Optional[int] = None,
        cache_position: Any = None,
        xattn_block: Optional[nn.Module] = None,
        vis_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        orig_dtype = streams.dtype
        s0, s1 = streams[0].float(), streams[1].float()

        if hasattr(layer, "input_layernorm") and hasattr(layer, "post_attention_layernorm"):
            target_dtype = layer.mlp.gate_proj.weight.dtype
            r1_a, w0_a, w1_a, p_swap_a = self._compute_routing_scalars(s0, s1, pass_idx=pass_idx, is_ffn=False)
            diff_a = s1 - s0
            mixed_a = s0 + r1_a * diff_a
            normed_a = layer.input_layernorm(mixed_a.to(orig_dtype)).to(target_dtype)
            if _attn_accepts_position_embeddings(layer.self_attn):
                attn_out, _ = layer.self_attn(
                    hidden_states=normed_a,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                )
            else:
                # Synthetic CPU test backbone compatibility
                use_c = isinstance(past_key_values, _SimpleTupleVisitCache)
                slot_kv = past_key_values.get_slot(int(virtual_idx or 0)) if use_c else None
                attn_out, next_kv = layer.self_attn(
                    normed_a,
                    attention_mask=raw_attention_mask_2d,
                    past_kv=slot_kv,
                    use_cache=use_c,
                )
                if use_c and virtual_idx is not None:
                    past_key_values.set_slot(int(virtual_idx), next_kv)

            gain_a_p = self.p3_extra_gain if (pass_idx >= 2 and self.p3_extra_gain is not None) else self.extra_gain
            gain_a = torch.tanh(gain_a_p.float()) if extra else 1.0
            alpha_a = gain_a * p_swap_a
            attn_delta = gain_a * attn_out.float()
            m0 = s0 + alpha_a * diff_a + w0_a * attn_delta
            m1 = s1 - alpha_a * diff_a + w1_a * attn_delta

            r1_f, w0_f, w1_f, p_swap_f = self._compute_routing_scalars(m0, m1, pass_idx=pass_idx, is_ffn=True)
            diff_f = m1 - m0
            mixed_f = m0 + r1_f * diff_f
            normed_f = layer.post_attention_layernorm(mixed_f.to(orig_dtype)).to(target_dtype)
            mlp_out = layer.mlp(normed_f).float() + self._apply_pass_adapter(mixed_f, pass_idx)
            gain_f_p = self.p3_ffn_extra_gain if (pass_idx >= 2 and self.p3_ffn_extra_gain is not None) else self.ffn_extra_gain
            gain_f = torch.tanh(gain_f_p.float()) if extra else 1.0
            alpha_f = gain_f * p_swap_f
            mlp_delta = gain_f * mlp_out
            out0 = m0 + alpha_f * diff_f + w0_f * mlp_delta
            out1 = m1 - alpha_f * diff_f + w1_f * mlp_delta
        else:
            r1, w0, w1, p_swap = self._compute_routing_scalars(s0, s1, pass_idx=pass_idx, is_ffn=False)
            diff = s1 - s0
            mixed_in = (s0 + r1 * diff).to(orig_dtype)
            updated = layer(
                mixed_in,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )
            if isinstance(updated, tuple):
                updated = updated[0]
            delta = updated.float() - mixed_in.float() + self._apply_pass_adapter(mixed_in, pass_idx)
            gain_f_p = self.p3_extra_gain if (pass_idx >= 2 and self.p3_extra_gain is not None) else self.extra_gain
            gain_f = torch.tanh(gain_f_p.float()) if extra else 1.0
            alpha = gain_f * p_swap
            out0 = s0 + alpha * diff + w0 * (gain_f * delta)
            out1 = s1 - alpha * diff + w1 * (gain_f * delta)

        # Explicit Visual Cross-Attention execution inside recurrent core:
        # - Pass 0 (`extra=False`): applies `xattn_block(out0, vis_ctx)` identically to unlooped SparkNative.
        # - Pass 1+ (`extra=True`): scales `xattn` delta by `gain_f` so Step-0 diff == 0.0 and Pass-2 smoothly re-attends!
        if xattn_block is not None and vis_ctx is not None:
            adapted0 = xattn_block(out0.to(orig_dtype), vis_ctx).float()
            if extra:
                out0 = out0 + gain_f * (adapted0 - out0)
            else:
                out0 = adapted0

        return torch.stack((out0.to(orig_dtype), out1.to(orig_dtype)), dim=0)


def build_spark_loop_context(
    spark_model: nn.Module,
    embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Any = None,
    cache_position: Optional[torch.Tensor] = None,
    cfg: Any = None,
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor]:
    """Construct exact `(causal_masks_by_type, rope_cos_sin_by_type, cache_position)` for Spark2_5Model."""
    cfg = cfg if cfg is not None else getattr(spark_model, "config", None)
    bsz, q_len, _ = embeds.shape
    device = embeds.device
    custom_cache_pos = cache_position is not None
    if cache_position is None:
        if past_key_values is not None:
            try:
                past_seen = int(past_key_values.get_seq_length(0))
            except Exception:
                try:
                    past_seen = int(past_key_values.get_seq_length())
                except Exception:
                    past_seen = 0
        else:
            past_seen = 0
        cache_position = torch.arange(past_seen, past_seen + q_len, device=device)

    raw_layer_types = getattr(cfg, "layer_types", None)
    if not raw_layer_types and hasattr(spark_model, "layers"):
        raw_layer_types = [getattr(getattr(lyr, "self_attn", lyr), "layer_type", "full_attention") for lyr in spark_model.layers]
    layer_types = set(raw_layer_types or ("full_attention",))

    if not _attn_accepts_position_embeddings(spark_model.layers[0].self_attn):
        return {lt: None for lt in layer_types}, {lt: None for lt in layer_types}, cache_position

    if past_key_values is not None and q_len == 1 and attention_mask is None:
        masks = {lt: None for lt in layer_types}
    elif not custom_cache_pos and past_key_values is None and hasattr(spark_model, "make_loop_context"):
        m_ctx, r_ctx = spark_model.make_loop_context(embeds, attention_mask)
        return m_ctx, r_ctx, cache_position
    else:
        q_idx = cache_position.view(1, 1, q_len, 1)
        if past_key_values is None:
            k_len = q_len
            k_idx = cache_position.view(1, 1, 1, q_len)
        else:
            k_len = int(cache_position[-1].item()) + 1 if cache_position.numel() > 0 else q_len
            k_idx = torch.arange(k_len, device=device).view(1, 1, 1, k_len)
        causal_ok = k_idx <= q_idx
        if attention_mask is not None and isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
            pad_ok = attention_mask[:, :k_len].bool().view(bsz, 1, 1, -1)
            if pad_ok.shape[-1] == k_len:
                causal_ok = causal_ok & pad_ok
        min_val = torch.finfo(embeds.dtype).min
        zero_val = torch.tensor(0.0, dtype=embeds.dtype, device=device)
        masks = {"full_attention": torch.where(causal_ok, zero_val, min_val)}
        if "sliding_attention" in layer_types or getattr(spark_model, "has_sliding_layers", False):
            sw = int(getattr(cfg, "sliding_window", 512) or 512)
            sw_ok = causal_ok & ((q_idx - k_idx) < sw)
            masks["sliding_attention"] = torch.where(sw_ok, zero_val, min_val)

    rope: dict[str, Any] = {}
    attn_mod = importlib.import_module(type(spark_model.layers[0].self_attn).__module__)
    if hasattr(attn_mod, "compute_rope_cos_sin") and hasattr(cfg, "get_rope_theta"):
        for lt in layer_types:
            rope[lt] = attn_mod.compute_rope_cos_sin(
                cache_position,
                cfg.head_dim,
                cfg.get_rope_theta(lt),
                partial_rotary_factor=cfg.get_partial_rotary_factor(lt),
                device=device,
            )
    elif hasattr(spark_model, "rotary_emb"):
        pos_ids = cache_position.unsqueeze(0).expand(bsz, -1)
        for lt in layer_types:
            try:
                rope[lt] = spark_model.rotary_emb(embeds, pos_ids, layer_type=lt)
            except TypeError:
                rope[lt] = spark_model.rotary_emb(embeds, pos_ids)
    else:
        for lt in layer_types:
            rope[lt] = None
    return masks, rope, cache_position
