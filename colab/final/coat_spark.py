"""Spark-specific COAT FP8 activation compression and weight-cache controller.

Preserves Spark's exact architecture:
- Gated GELU (`erf` and `tanh` approximations — never COAT's Llama SiLU).
- Headwise attention output gates, hybrid sliding/full causal masks, and RoPE
  left in BF16/FP32 precision.
- Untouched BF16/FP32 inference route when `eval()` or `controller.enabled=False`.
- Autograd through frozen Spark operations to upstream trainable visual inputs
  without saving unnecessary linear input activations when `weight` is frozen.
- Explicit FP8 weight-cache invalidation on optimizer step and rebuild on restore.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from typing import Any, Iterator

import torch
from torch import nn
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    HAS_TRITON = False

FP8_ACT_DTYPE = torch.float8_e4m3fn
FP8_ACT_MAX = 448.0
FP8_ACT_MIN_NORMAL = 2.0**-6


if HAS_TRITON:

    @triton.jit
    def _coat_quant_fp8_kernel(
        x_ptr,
        q_ptr,
        s_ptr,
        numel,
        num_groups,
        GROUP_SIZE: tl.constexpr,
        GROUPS_PER_BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        g_idx = pid * GROUPS_PER_BLOCK + tl.arange(0, GROUPS_PER_BLOCK)[:, None]
        c_idx = tl.arange(0, GROUP_SIZE)[None, :]
        elem_idx = g_idx * GROUP_SIZE + c_idx
        g_mask = g_idx < num_groups
        e_mask = g_mask & (elem_idx < numel)
        vals = tl.load(x_ptr + elem_idx, mask=e_mask, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(vals), axis=1, keep_dims=True)
        scale = tl.maximum(amax * (1.0 / 448.0), 1e-12)
        scaled = tl.clamp(vals / scale, -448.0, 448.0)
        tl.store(q_ptr + elem_idx, scaled.to(q_ptr.dtype.element_ty), mask=g_mask)
        s_row = pid * GROUPS_PER_BLOCK + tl.arange(0, GROUPS_PER_BLOCK)
        tl.store(s_ptr + s_row, tl.reshape(scale, (GROUPS_PER_BLOCK,)).to(tl.bfloat16), mask=s_row < num_groups)

    @triton.jit
    def _coat_dequant_fp8_kernel(
        q_ptr,
        s_ptr,
        out_ptr,
        numel,
        num_groups,
        GROUP_SIZE: tl.constexpr,
        GROUPS_PER_BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        g_idx = pid * GROUPS_PER_BLOCK + tl.arange(0, GROUPS_PER_BLOCK)[:, None]
        c_idx = tl.arange(0, GROUP_SIZE)[None, :]
        elem_idx = g_idx * GROUP_SIZE + c_idx
        e_mask = (g_idx < num_groups) & (elem_idx < numel)
        q_val = tl.load(q_ptr + elem_idx, mask=e_mask, other=0.0).to(tl.float32)
        s_val = tl.load(s_ptr + g_idx, mask=g_idx < num_groups, other=0.0).to(tl.float32)
        restored = q_val * s_val
        tl.store(out_ptr + elem_idx, restored.to(out_ptr.dtype.element_ty), mask=e_mask)


@dataclass
class CoatActivationTelemetry:
    compressed_tensors: int = 0
    uncompressed_bytes: int = 0
    compressed_bytes: int = 0
    saturated_elements: int = 0
    underflow_elements: int = 0
    total_elements: int = 0
    skipped_frozen_linear_inputs: int = 0
    cache_invalidations: int = 0
    cache_rebuilds: int = 0
    collect_detailed_stats: bool = False

    def reset_step(self) -> None:
        self.compressed_tensors = 0
        self.uncompressed_bytes = 0
        self.compressed_bytes = 0
        self.saturated_elements = 0
        self.underflow_elements = 0
        self.total_elements = 0
        self.skipped_frozen_linear_inputs = 0

    def summary(self) -> dict[str, float | int]:
        total = max(1, self.total_elements)
        comp = max(1, self.compressed_bytes)
        return {
            "compressed_tensors": self.compressed_tensors,
            "uncompressed_bytes": self.uncompressed_bytes,
            "compressed_bytes": self.compressed_bytes,
            "compression_ratio": (
                self.uncompressed_bytes / comp if self.uncompressed_bytes > 0 else 1.0
            ),
            "saved_bytes_delta": max(0, self.uncompressed_bytes - self.compressed_bytes),
            "saturation_fraction": self.saturated_elements / total,
            "underflow_fraction": self.underflow_elements / total,
            "skipped_frozen_linear_inputs": self.skipped_frozen_linear_inputs,
            "cache_invalidations": self.cache_invalidations,
            "cache_rebuilds": self.cache_rebuilds,
        }


def quantize_activation_per_group_fp8(
    x: torch.Tensor,
    group_size: int = 128,
    telemetry: CoatActivationTelemetry | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], torch.dtype, int]:
    """Per-group FP8 (`float8_e4m3fn`) quantization via 1-pass Triton kernel (zero host-device syncs)."""
    orig_shape = tuple(x.shape)
    orig_dtype = x.dtype
    flat = x.detach().contiguous().reshape(-1)
    numel = flat.numel()
    pad = (-numel) % group_size
    num_groups = (numel + pad) // group_size
    want_stats = telemetry is not None and bool(getattr(telemetry, "collect_detailed_stats", False))

    if (
        HAS_TRITON
        and flat.is_cuda
        and num_groups > 0
        and (group_size & (group_size - 1) == 0)
        and not want_stats
    ):
        q = torch.empty((num_groups, group_size), dtype=FP8_ACT_DTYPE, device=x.device)
        scale_bf16 = torch.empty((num_groups,), dtype=torch.bfloat16, device=x.device)
        gpb = max(1, min(32, 256 // max(1, group_size)))
        grid = (triton.cdiv(num_groups, gpb),)
        _coat_quant_fp8_kernel[grid](
            flat,
            q,
            scale_bf16,
            numel,
            num_groups,
            GROUP_SIZE=group_size,
            GROUPS_PER_BLOCK=gpb,
        )
        if telemetry is not None:
            telemetry.compressed_tensors += 1
            telemetry.uncompressed_bytes += numel * x.element_size()
            telemetry.compressed_bytes += num_groups * group_size + num_groups * 2
            telemetry.total_elements += numel
        return q, scale_bf16, orig_shape, orig_dtype, numel

    q = torch.empty((num_groups, group_size), dtype=FP8_ACT_DTYPE, device=x.device)
    scale_bf16 = torch.empty((num_groups,), dtype=torch.bfloat16, device=x.device)

    sat_count = 0
    uflow_count = 0
    chunk_groups = max(1, 2_097_152 // max(1, group_size))

    for g_start in range(0, num_groups, chunk_groups):
        g_end = min(num_groups, g_start + chunk_groups)
        e_start = g_start * group_size
        e_end = min(numel, g_end * group_size)
        raw_slice = flat[e_start:e_end].float()
        valid_len = raw_slice.numel()
        need_pad = (g_end - g_start) * group_size - valid_len
        if need_pad > 0:
            groups = F.pad(raw_slice, (0, need_pad)).reshape(-1, group_size)
        else:
            groups = raw_slice.reshape(-1, group_size)
        amax = groups.abs().amax(dim=-1, keepdim=True)
        scale = (amax / FP8_ACT_MAX).clamp_min(1e-12)
        scaled = groups / scale
        q[g_start:g_end] = scaled.clamp(-FP8_ACT_MAX, FP8_ACT_MAX).to(FP8_ACT_DTYPE)
        scale_bf16[g_start:g_end] = scale.squeeze(-1).to(torch.bfloat16)

        if want_stats and valid_len > 0:
            valid_scaled = scaled.reshape(-1)[:valid_len].abs()
            sat_count += int((valid_scaled > FP8_ACT_MAX * 0.999).sum().item())
            uflow_count += int(
                ((raw_slice != 0) & (valid_scaled < FP8_ACT_MIN_NORMAL * 0.5)).sum().item()
            )

    if telemetry is not None:
        telemetry.compressed_tensors += 1
        telemetry.uncompressed_bytes += numel * x.element_size()
        telemetry.compressed_bytes += (
            q.numel() * q.element_size() + scale_bf16.numel() * scale_bf16.element_size()
        )
        telemetry.saturated_elements += sat_count
        telemetry.underflow_elements += uflow_count
        telemetry.total_elements += numel

    return q, scale_bf16, orig_shape, orig_dtype, numel


def dequantize_activation_per_group_fp8(
    q: torch.Tensor,
    scale_bf16: torch.Tensor,
    orig_shape: tuple[int, ...],
    orig_dtype: torch.dtype,
    numel: int,
    *,
    as_float32: bool = False,
) -> torch.Tensor:
    target_dtype = torch.float32 if as_float32 else orig_dtype
    num_groups, group_size = q.shape
    if (
        HAS_TRITON
        and q.is_cuda
        and num_groups > 0
        and (group_size & (group_size - 1) == 0)
    ):
        out_flat = torch.empty((numel,), dtype=target_dtype, device=q.device)
        gpb = max(1, min(32, 256 // max(1, group_size)))
        grid = (triton.cdiv(num_groups, gpb),)
        _coat_dequant_fp8_kernel[grid](
            q,
            scale_bf16,
            out_flat,
            numel,
            num_groups,
            GROUP_SIZE=group_size,
            GROUPS_PER_BLOCK=gpb,
        )
        return out_flat.reshape(orig_shape)
    if as_float32 or q.numel() <= 2_097_152:
        restored = q.float() * scale_bf16.float().unsqueeze(-1)
        flat = restored.reshape(-1)[:numel].reshape(orig_shape)
        return flat if as_float32 else flat.to(orig_dtype)
    out = torch.empty((num_groups, group_size), dtype=target_dtype, device=q.device)
    chunk_groups = max(1, 2_097_152 // max(1, group_size))
    for g_start in range(0, num_groups, chunk_groups):
        g_end = min(num_groups, g_start + chunk_groups)
        out[g_start:g_end] = (
            q[g_start:g_end].float() * scale_bf16[g_start:g_end].float().unsqueeze(-1)
        ).to(target_dtype)
    return out.reshape(-1)[:numel].reshape(orig_shape)


def _gelu_forward_and_grad(
    g: torch.Tensor, approximate: str = "none"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact analytical GELU(g) and d/dg GELU(g) in FP32 (never SiLU)."""
    gf = g.float()
    if approximate == "tanh":
        k0 = math.sqrt(2.0 / math.pi)
        k1 = 0.044715
        inner = k0 * (gf + k1 * gf * gf * gf)
        th = torch.tanh(inner)
        sech2 = 1.0 - th * th
        dinner = k0 * (1.0 + 3.0 * k1 * gf * gf)
        val = 0.5 * gf * (1.0 + th)
        grad = 0.5 * (1.0 + th) + 0.5 * gf * sech2 * dinner
        return val, grad
    if approximate == "none":
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        inv_sqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)
        cdf = 0.5 * (1.0 + torch.erf(gf * inv_sqrt2))
        pdf = inv_sqrt2pi * torch.exp(-0.5 * gf * gf)
        val = gf * cdf
        grad = cdf + gf * pdf
        return val, grad
    raise ValueError(f"Unsupported Spark GELU approximation mode: {approximate!r}")


class _CoatSparkGatedGELUFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        gate: torch.Tensor,
        up: torch.Tensor,
        approximate: str,
        group_size: int,
        telemetry: CoatActivationTelemetry | None,
    ) -> torch.Tensor:
        out = F.gelu(gate, approximate=approximate) * up
        gq, gs, gshape, gdtype, gnumel = quantize_activation_per_group_fp8(
            gate, group_size=group_size, telemetry=telemetry
        )
        uq, us, ushape, udtype, unumel = quantize_activation_per_group_fp8(
            up, group_size=group_size, telemetry=telemetry
        )
        ctx.save_for_backward(gq, gs, uq, us)
        ctx.meta = (gshape, gdtype, gnumel, ushape, udtype, unumel, approximate)
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[Any, ...]:
        gq, gs, uq, us = ctx.saved_tensors
        gshape, gdtype, gnumel, ushape, udtype, unumel, approximate = ctx.meta
        gate_f = dequantize_activation_per_group_fp8(
            gq, gs, gshape, gdtype, gnumel, as_float32=True
        )
        up_f = dequantize_activation_per_group_fp8(
            uq, us, ushape, udtype, unumel, as_float32=True
        )
        gelu_g, dgelu_g = _gelu_forward_and_grad(gate_f, approximate=approximate)
        go = grad_out.float()
        grad_gate = (go * up_f * dgelu_g).to(gdtype) if ctx.needs_input_grad[0] else None
        grad_up = (go * gelu_g).to(udtype) if ctx.needs_input_grad[1] else None
        return grad_gate, grad_up, None, None, None


def coat_spark_gated_gelu(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    approximate: str = "none",
    enabled: bool = True,
    training: bool = True,
    group_size: int = 128,
    telemetry: CoatActivationTelemetry | None = None,
) -> torch.Tensor:
    """Spark gated-GELU with COAT FP8 saved activations during training only."""
    if not (enabled and training and (gate.requires_grad or up.requires_grad)):
        return F.gelu(gate, approximate=approximate) * up
    return _CoatSparkGatedGELUFn.apply(gate, up, approximate, group_size, telemetry)


class _CoatLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        group_size: int,
        telemetry: CoatActivationTelemetry | None,
        fp8_weight_cache: dict[str, Any] | None,
        use_fp8_matmul: bool,
    ) -> torch.Tensor:
        compute_dtype = x.dtype
        w_cast = weight if weight.dtype == compute_dtype else weight.to(compute_dtype)
        b_cast = None if bias is None else (bias if bias.dtype == compute_dtype else bias.to(compute_dtype))

        if (
            use_fp8_matmul
            and x.is_cuda
            and hasattr(torch, "_scaled_mm")
            and x.shape[-1] % 16 == 0
            and weight.shape[0] % 16 == 0
        ):
            out = _fp8_scaled_linear_forward(x, weight, b_cast, fp8_weight_cache)
        else:
            out = F.linear(x, w_cast, b_cast)

        need_w_grad = weight.requires_grad
        need_b_grad = bias is not None and bias.requires_grad
        ctx.need_w_grad = need_w_grad
        ctx.need_b_grad = need_b_grad
        ctx.weight_ref = weight if need_w_grad else None
        ctx.bias_ref = bias if need_b_grad else None
        ctx.x_dtype = x.dtype
        ctx.w_dtype = weight.dtype
        ctx.b_dtype = bias.dtype if bias is not None else None

        if need_w_grad:
            xq, xs, xshape, xdtype, xnumel = quantize_activation_per_group_fp8(
                x, group_size=group_size, telemetry=telemetry
            )
            ctx.save_for_backward(weight, xq, xs)
            ctx.x_meta = (xshape, xdtype, xnumel)
        else:
            if telemetry is not None and x.requires_grad:
                telemetry.skipped_frozen_linear_inputs += 1
            ctx.save_for_backward(weight)
            ctx.x_meta = None
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[Any, ...]:
        saved = ctx.saved_tensors
        weight = saved[0]
        grad_x = None
        grad_w = None
        grad_b = None

        if ctx.needs_input_grad[0]:
            w_for_dx = weight if weight.dtype == grad_out.dtype else weight.to(grad_out.dtype)
            grad_x = torch.matmul(grad_out, w_for_dx)
            if grad_x.dtype != ctx.x_dtype:
                grad_x = grad_x.to(ctx.x_dtype)

        if ctx.need_w_grad and ctx.needs_input_grad[1]:
            _, xq, xs = saved
            xshape, xdtype, xnumel = ctx.x_meta
            x_restored = dequantize_activation_per_group_fp8(
                xq, xs, xshape, xdtype, xnumel, as_float32=False
            )
            go_2d = grad_out.reshape(-1, grad_out.shape[-1])
            xr_2d = x_restored.reshape(-1, x_restored.shape[-1])
            if go_2d.dtype != xr_2d.dtype:
                xr_2d = xr_2d.to(go_2d.dtype)
            grad_w = go_2d.t().matmul(xr_2d)
            consume_w = getattr(ctx.weight_ref, "_immediate_accum_fn", None)
            if consume_w is not None and consume_w(grad_w):
                grad_w = None
            elif grad_w.dtype != ctx.w_dtype:
                grad_w = grad_w.to(ctx.w_dtype)

        if ctx.need_b_grad and ctx.needs_input_grad[2]:
            reduce_dims = tuple(range(grad_out.ndim - 1))
            grad_b = grad_out.float().sum(dim=reduce_dims).to(ctx.b_dtype)
            consume_b = getattr(ctx.bias_ref, "_immediate_accum_fn", None)
            if consume_b is not None and consume_b(grad_b):
                grad_b = None

        return grad_x, grad_w, grad_b, None, None, None, None


def _fp8_scaled_linear_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    cache: dict[str, Any] | None,
) -> torch.Tensor:
    """SM89+/SM120 Tensor Core `torch._scaled_mm` forward with cached FP8 weight."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    if x_2d.shape[0] % 16 != 0:
        # _scaled_mm requires M divisible by 16 on several CUDA backends; pad cleanly.
        pad_m = (-x_2d.shape[0]) % 16
        x_2d_padded = F.pad(x_2d, (0, 0, 0, pad_m))
    else:
        pad_m = 0
        x_2d_padded = x_2d

    x_amax = x_2d_padded.abs().amax().float().clamp_min(1e-12)
    scale_a = (x_amax / FP8_ACT_MAX).reshape(1)
    x_fp8 = (x_2d_padded.float() / scale_a).clamp(-FP8_ACT_MAX, FP8_ACT_MAX).to(FP8_ACT_DTYPE)

    if cache is not None and "w_fp8_col_major" in cache and cache.get("device") == weight.device:
        w_t_fp8 = cache["w_fp8_col_major"]
        scale_b = cache["scale_b"]
    else:
        w_amax = weight.abs().amax().float().clamp_min(1e-12)
        scale_b = (w_amax / FP8_ACT_MAX).reshape(1)
        w_fp8 = (weight.float() / scale_b).clamp(-FP8_ACT_MAX, FP8_ACT_MAX).to(FP8_ACT_DTYPE)
        # _scaled_mm requires second operand column-major: [K, N] = w_fp8.t()
        w_t_fp8 = w_fp8.t()
        if cache is not None:
            cache["w_fp8_col_major"] = w_t_fp8
            cache["scale_b"] = scale_b
            cache["device"] = weight.device

    out_dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float16, torch.float32) else torch.bfloat16
    res = torch._scaled_mm(x_fp8, w_t_fp8, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype)
    out_2d = res[0] if isinstance(res, tuple) else res
    if pad_m:
        out_2d = out_2d[:-pad_m]
    out = out_2d.reshape(*orig_shape[:-1], weight.shape[0])
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


class CoatLinear(nn.Module):
    """Drop-in `nn.Linear` wrapper delegating untouched `eval()` to base Linear."""

    def __init__(self, base: nn.Linear, controller: "CoatSparkController", name: str = ""):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = base.weight
        self.bias = base.bias
        self._controller = controller
        self._layer_name = name
        self._fp8_weight_cache: dict[str, Any] = {}

    def invalidate_fp8_cache(self) -> None:
        self._fp8_weight_cache.clear()

    def rebuild_fp8_cache(self, *, force: bool = False) -> None:
        self._fp8_weight_cache.clear()
        if force or (self._controller.use_fp8_matmul and self.weight.is_cuda):
            with torch.no_grad():
                w_amax = self.weight.abs().amax().float().clamp_min(1e-12)
                scale_b = (w_amax / FP8_ACT_MAX).reshape(1)
                w_fp8 = (self.weight.float() / scale_b).clamp(-FP8_ACT_MAX, FP8_ACT_MAX).to(FP8_ACT_DTYPE)
                self._fp8_weight_cache["w_fp8_col_major"] = w_fp8.t()
                self._fp8_weight_cache["scale_b"] = scale_b
                self._fp8_weight_cache["device"] = self.weight.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (self.training and self._controller.enabled):
            w = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
            b = None if self.bias is None else (self.bias if self.bias.dtype == x.dtype else self.bias.to(x.dtype))
            return F.linear(x, w, b)
        return _CoatLinearFn.apply(
            x,
            self.weight,
            self.bias,
            self._controller.group_size,
            self._controller.telemetry,
            self._fp8_weight_cache,
            self._controller.use_fp8_matmul,
        )


@dataclass
class CoatSparkController:
    """Owns COAT activation compression state, hooks, and FP8 weight caches."""

    enabled: bool = True
    group_size: int = 128
    min_features: int = 128
    min_saved_hook_numel: int = 4096
    use_fp8_matmul: bool = False
    use_saved_tensors_hook: bool = True
    telemetry: CoatActivationTelemetry = field(default_factory=CoatActivationTelemetry)
    wrapped_linears: list[CoatLinear] = field(default_factory=list)
    verified_mlp_activations: list[str] = field(default_factory=list)
    _protected_ptrs: set[int] = field(default_factory=set)

    def register_protected_module(self, module: nn.Module) -> None:
        for p in module.parameters():
            self._protected_ptrs.add(p.data_ptr())
        for b in module.buffers():
            self._protected_ptrs.add(b.data_ptr())

    def invalidate_weight_caches(self) -> None:
        for mod in self.wrapped_linears:
            mod.invalidate_fp8_cache()
        self.telemetry.cache_invalidations += 1

    def rebuild_weight_caches(self, *, force: bool = False) -> None:
        for mod in self.wrapped_linears:
            mod.rebuild_fp8_cache(force=force)
        self.telemetry.cache_rebuilds += 1

    @contextmanager
    def activation_context(self, *, training: bool) -> Iterator[None]:
        """Optional `saved_tensors_hooks` compressing remaining large MLP activations.

        Skips parameters, buffers, masks, integer indices, small gate/norm tensors,
        and already-quantized FP8 tensors. Never active when `training=False`.
        """
        if not (self.enabled and training and self.use_saved_tensors_hook):
            yield
            return

        group_size = self.group_size
        min_numel = self.min_saved_hook_numel
        protected = self._protected_ptrs
        telemetry = self.telemetry

        def pack(t: torch.Tensor) -> Any:
            if (
                not isinstance(t, torch.Tensor)
                or isinstance(t, nn.Parameter)
                or t.dtype not in (torch.bfloat16, torch.float16, torch.float32)
                or t.ndim < 2
                or t.numel() < min_numel
                or t.data_ptr() in protected
                # Skip square/4D attention score/mask/RoPE tensors: compress hidden-feature activations only.
                or t.ndim == 4
            ):
                return t
            q, s, shape, dtype, numel = quantize_activation_per_group_fp8(
                t, group_size=group_size, telemetry=telemetry
            )
            return ("__coat_saved_fp8__", q, s, shape, dtype, numel)

        def unpack(packed: Any) -> torch.Tensor:
            if isinstance(packed, tuple) and len(packed) == 6 and packed[0] == "__coat_saved_fp8__":
                _, q, s, shape, dtype, numel = packed
                return dequantize_activation_per_group_fp8(q, s, shape, dtype, numel, as_float32=False)
            return packed

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            yield


def verify_spark_mlp_not_silu(spark_module: nn.Module) -> list[str]:
    """Verify Spark decoder MLPs preserve GELU semantics and reject Llama SiLU assumptions."""
    found: list[str] = []
    for name, submodule in spark_module.named_modules():
        act = getattr(submodule, "act_fn", None) or getattr(submodule, "activation_fn", None)
        if act is None:
            continue
        cls_name = type(act).__name__.lower()
        fn_name = getattr(act, "__name__", "").lower()
        tag = cls_name or fn_name
        if "silu" in tag or "swish" in tag:
            raise ValueError(
                f"Module {name} uses SiLU ({tag}); Spark COAT converter forbids silent Llama-SiLU substitution"
            )
        found.append(f"{name}:{tag}")
    return found


def install_spark_coat(
    spark_module: nn.Module,
    *,
    enabled: bool = True,
    group_size: int = 128,
    min_features: int = 128,
    min_saved_hook_numel: int = 4096,
    use_fp8_matmul: bool = False,
    use_saved_tensors_hook: bool = True,
) -> CoatSparkController:
    """Wrap eligible Spark decoder Linear layers with `CoatLinear` while protecting norms/gates/heads."""
    existing = getattr(spark_module, "_coat_controller", None)
    if isinstance(existing, CoatSparkController):
        existing.enabled = enabled
        existing.use_fp8_matmul = use_fp8_matmul
        return existing

    verified = verify_spark_mlp_not_silu(spark_module)
    controller = CoatSparkController(
        enabled=enabled,
        group_size=group_size,
        min_features=min_features,
        min_saved_hook_numel=min_saved_hook_numel,
        use_fp8_matmul=use_fp8_matmul,
        use_saved_tensors_hook=use_saved_tensors_hook,
        verified_mlp_activations=verified,
    )

    excluded_tokens = (
        "lm_head",
        "embed_tokens",
        "norm",
        "output_gate",
        "attn_gate",
        "gate_proj_scalar",
    )

    def _wrap_children(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            lower = full_name.lower()
            if isinstance(child, nn.Linear):
                if (
                    min(child.in_features, child.out_features) >= min_features
                    and not any(tok in lower for tok in excluded_tokens)
                ):
                    wrapped = CoatLinear(child, controller, name=full_name)
                    setattr(parent, child_name, wrapped)
                    controller.wrapped_linears.append(wrapped)
                continue
            if not isinstance(child, CoatLinear):
                _wrap_children(child, full_name)

    # Wrap inside `spark.model` (decoder stack) if present so causal LM `lm_head` stays untouched.
    target = getattr(spark_module, "model", spark_module)
    _wrap_children(target, "model" if target is not spark_module else "")
    controller.register_protected_module(spark_module)
    setattr(spark_module, "_coat_controller", controller)
    return controller


def qualify_coat_kernels(device: str | torch.device = "cpu") -> dict[str, Any]:
    """Qualify COAT FP8 activation codecs and (on CUDA SM89+/SM120) `_scaled_mm` kernels."""
    dev = torch.device(device)
    torch.manual_seed(17)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32

    x_ref = torch.randn(4, 64, 256, device=dev, dtype=dtype, requires_grad=True)
    g_ref = torch.randn(4, 64, 256, device=dev, dtype=dtype, requires_grad=True)
    lin_ref = nn.Linear(256, 256, device=dev, dtype=dtype)
    y_ref = coat_spark_gated_gelu(lin_ref(g_ref), lin_ref(x_ref), enabled=False, training=True)
    y_ref.float().square().mean().backward()
    dx_ref = x_ref.grad.detach().float().clone()
    dw_ref = lin_ref.weight.grad.detach().float().clone()

    x_coat = x_ref.detach().clone().requires_grad_(True)
    g_coat = g_ref.detach().clone().requires_grad_(True)
    holder = nn.Module()
    holder.proj = nn.Linear(256, 256, device=dev, dtype=dtype)
    with torch.no_grad():
        holder.proj.weight.copy_(lin_ref.weight)
        holder.proj.bias.copy_(lin_ref.bias)

    use_scaled_mm = dev.type == "cuda" and torch.cuda.get_device_capability(dev)[0] >= 9
    ctrl = install_spark_coat(
        holder,
        enabled=True,
        group_size=128,
        min_features=64,
        use_fp8_matmul=use_scaled_mm,
        use_saved_tensors_hook=False,
    )
    ctrl.telemetry.collect_detailed_stats = True
    holder.train()
    y_coat = coat_spark_gated_gelu(
        holder.proj(g_coat),
        holder.proj(x_coat),
        enabled=True,
        training=True,
        telemetry=ctrl.telemetry,
    )
    y_coat.float().square().mean().backward()

    dx_err = float((x_coat.grad.float() - dx_ref).norm() / dx_ref.norm().clamp_min(1e-12))
    dw_err = float((holder.proj.weight.grad.float() - dw_ref).norm() / dw_ref.norm().clamp_min(1e-12))

    holder.eval()
    with torch.no_grad():
        eval_coat = holder.proj(x_ref.detach())
        eval_orig = F.linear(x_ref.detach(), lin_ref.weight, lin_ref.bias)
        eval_exact_match = bool(torch.equal(eval_coat, eval_orig))

    if dx_err > 0.08 or dw_err > 0.08 or not eval_exact_match:
        raise RuntimeError(
            f"COAT kernel parity failed: dx_rel_err={dx_err:.5f}, dw_rel_err={dw_err:.5f}, eval_exact={eval_exact_match}"
        )

    return {
        "passed": True,
        "device": str(dev),
        "fp8_scaled_mm_tested": use_scaled_mm,
        "dx_relative_l2_error": dx_err,
        "dw_relative_l2_error": dw_err,
        "eval_route_exact_match": eval_exact_match,
        "telemetry": ctrl.telemetry.summary(),
    }
