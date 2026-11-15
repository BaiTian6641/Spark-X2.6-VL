"""APOLLO optimizer, COAT FP8 moment codec, and single-buffer merged gradients.

Implements the training backend contract for Spark-X2.6-VL:
1. Channelwise APOLLO (`scale_type='channel'`, e.g. rank 64 vs 32) and rank-1
   APOLLO-Mini (`scale_type='tensor'`) using projected Adam moments while
   applying the channel/tensor-scaled update + decoupled weight decay to the
   full accumulated gradient.
2. Separate high-precision FP32-master AdamW group for biases, normalizations,
   attention/adapter gates, and small matrices. Stock `CoatAdamW` never touches
   APOLLO-owned parameters.
3. Optional COAT-style dynamic-range-expanded FP8 (`float8_e4m3fn`) storage for
   APOLLO projected moments while preserving the exact APOLLO update rule.
4. One owned accumulation buffer per unique trainable parameter via
   `register_post_accumulate_grad_hook`, exact per-example / per-token
   normalization across uneven final batches, global finiteness/clipping,
   reproducible projection refresh & stochastic-rounding RNG persistence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable

import torch
from torch import nn

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0
FP8_MIN_NORMAL = 2.0**-6  # 0.015625 for E4M3
FP8_DYNAMIC_RANGE = FP8_MAX / FP8_MIN_NORMAL  # 28672.0


@dataclass
class StepDiagnostics:
    applied: bool
    finite: bool
    grad_norm: float
    clip_coef: float
    supervised_units: int
    microbatches: int
    step: int


class CoatDynamicRangeFP8Codec:
    """COAT-style per-group dynamic-range-expanded FP8 codec for APOLLO moments.

    Compresses FP32 projected moments `(exp_avg, exp_avg_sq)` into `float8_e4m3fn`
    with per-group `amax` and power-law range expansion exponent `k_g` so small
    moment magnitudes utilize the full E4M3 dynamic range without altering the
    APOLLO update equations.
    """

    def __init__(self, group_size: int = 128, min_exp: float = 0.25, max_exp: float = 4.0):
        if group_size < 8:
            raise ValueError("group_size must be at least 8")
        self.group_size = int(group_size)
        self.min_exp = float(min_exp)
        self.max_exp = float(max_exp)
        self.last_stats: dict[str, float] = {
            "saturated_fraction": 0.0,
            "underflow_fraction": 0.0,
        }

    def encode(self, tensor: torch.Tensor, *, signed: bool = True) -> dict[str, Any]:
        x = tensor.detach().float()
        orig_shape = tuple(x.shape)
        flat = x.reshape(-1)
        numel = flat.numel()
        pad = (-numel) % self.group_size
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        groups = flat.reshape(-1, self.group_size)
        abs_g = groups.abs()
        amax = abs_g.amax(dim=-1, keepdim=True)  # [G, 1]

        nonzero = torch.where(abs_g > 0, abs_g, torch.full_like(abs_g, float("inf")))
        amin = nonzero.amin(dim=-1, keepdim=True)
        has_span = (amax > 0) & torch.isfinite(amin) & (amax > amin * 1.01)

        ratio = torch.where(has_span, (amax / amin.clamp_min(1e-30)).clamp_min(1.01), torch.full_like(amax, FP8_DYNAMIC_RANGE))
        exp = (math.log(FP8_DYNAMIC_RANGE) / torch.log(ratio)).clamp(self.min_exp, self.max_exp)
        exp = torch.where(has_span, exp, torch.ones_like(exp))

        safe_amax = torch.where(amax > 0, amax, torch.ones_like(amax))
        unit = (abs_g / safe_amax).clamp(0.0, 1.0)
        expanded = torch.pow(unit + 1e-38, exp) * FP8_MAX
        expanded = torch.where(abs_g > 0, expanded, torch.zeros_like(expanded))
        if signed:
            expanded = expanded * torch.sign(groups)

        valid = abs_g.reshape(-1)[:numel]
        exp_valid = expanded.reshape(-1)[:numel].abs()
        nonzero_count = max(1, int((valid > 0).sum().item()))
        saturated = float((exp_valid > FP8_MAX * 0.999).sum().item()) / max(1, numel)
        underflow = float(((valid > 0) & (exp_valid < FP8_MIN_NORMAL * 0.5)).sum().item()) / nonzero_count
        self.last_stats = {
            "saturated_fraction": saturated,
            "underflow_fraction": underflow,
        }

        q = expanded.clamp(-FP8_MAX if signed else 0.0, FP8_MAX).to(FP8_DTYPE)
        return {
            "__coat_fp8__": True,
            "q": q,
            "amax": amax.squeeze(-1),
            "exp": exp.squeeze(-1).to(torch.bfloat16),
            "shape": orig_shape,
            "numel": numel,
            "signed": signed,
        }

    @staticmethod
    def decode(state_obj: dict[str, Any], *, device: torch.device | None = None) -> torch.Tensor:
        q = state_obj["q"]
        if device is not None and q.device != device:
            q = q.to(device)
        qf = q.float()
        amax = state_obj["amax"].to(device=qf.device, dtype=torch.float32).unsqueeze(-1)
        exp = state_obj["exp"].to(device=qf.device, dtype=torch.float32).unsqueeze(-1)
        inv_exp = 1.0 / exp.clamp_min(1e-6)
        mag = (qf.abs() / FP8_MAX).clamp(0.0, 1.0)
        restored = amax * torch.where(mag > 0, torch.pow(mag, inv_exp), torch.zeros_like(mag))
        if state_obj["signed"]:
            restored = restored * torch.sign(qf)
        flat = restored.reshape(-1)[: int(state_obj["numel"])]
        return flat.reshape(state_obj["shape"])

    @staticmethod
    def storage_bytes(state_obj: dict[str, Any]) -> int:
        return (
            state_obj["q"].numel() * state_obj["q"].element_size()
            + state_obj["amax"].numel() * state_obj["amax"].element_size()
            + state_obj["exp"].numel() * state_obj["exp"].element_size()
        )


def stochastic_round_fp32_to_bf16(
    fp32_tensor: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Unbiased IEEE-754 float32 -> bfloat16 stochastic rounding using persisted RNG."""
    if fp32_tensor.dtype != torch.float32:
        fp32_tensor = fp32_tensor.float()
    raw = fp32_tensor.contiguous().view(torch.int32)
    sign = raw & -0x80000000  # 0x80000000 in signed int32
    abs_bits = raw & 0x7FFFFFFF
    # Do not perturb NaN/Inf exponents (0x7F800000).
    finite_mask = abs_bits < 0x7F800000
    noise = torch.randint(
        0,
        1 << 16,
        size=raw.shape,
        generator=generator,
        device=raw.device,
        dtype=torch.int32,
    )
    rounded_abs = torch.where(finite_mask, (abs_bits + noise) & ~0xFFFF, abs_bits)
    rounded_bits = sign | rounded_abs
    return rounded_bits.view(torch.float32).to(torch.bfloat16)


def _make_projection_matrix(
    rank: int,
    proj_dim: int,
    *,
    seed: int,
    refresh_epoch: int,
    param_slot: int,
    device: torch.device,
    proj_type: str = "normal",
) -> torch.Tensor:
    """Deterministic basis P in R^{rank x proj_dim} scaled by 1/sqrt(rank)."""
    # Generate on CPU first for cross-device / save-restore determinism, then move.
    # Projected dims are bounded (e.g. 64 x 2560 = 655 KB) and refreshed every ~200 steps.
    gen = torch.Generator(device="cpu")
    mixed_seed = (
        (int(seed) * 0x9E3779B97F4A7C15)
        ^ (int(param_slot + 1) * 0xBF58476D1CE4E5B9)
        ^ (int(refresh_epoch) * 0x94D049BB133111EB)
    ) & ((1 << 63) - 1)
    gen.manual_seed(mixed_seed)
    scale = 1.0 / math.sqrt(rank)
    if proj_type == "rademacher":
        bits = torch.randint(0, 2, (rank, proj_dim), generator=gen, dtype=torch.int8)
        mat = (bits.float() * 2.0 - 1.0) * scale
    else:
        mat = torch.randn((rank, proj_dim), generator=gen, dtype=torch.float32) * scale
    return mat.to(device=device, dtype=torch.float32)


def is_high_precision_small_or_sensitive_param(
    name: str,
    param: nn.Parameter,
    *,
    min_apollo_dim: int = 128,
    allow_embeddings_in_apollo: bool = True,
) -> bool:
    """Keep biases, norms, residual/attention gates, and small tensors in FP32 AdamW."""
    if param.ndim != 2 or min(param.shape) < min_apollo_dim:
        return True
    lower = name.lower()
    sensitive_tokens = (
        "norm",
        "ln_",
        "layernorm",
        "rmsnorm",
        "bias",
        "gate_attn",
        "gate_ffn",
        "attn_output_gate",
        "output_gate",
        "residual_gate",
        "spatial_clock",
        "audio_clock",
        "alpha_projector",
        "view_embedding",
        "modality",
        "default_voice",
    )
    if any(tok in lower for tok in sensitive_tokens):
        return True
    if not allow_embeddings_in_apollo and ("embed_tokens" in lower or "lm_head" in lower or "wte" in lower):
        return True
    return False


def build_apollo_param_groups(
    model: nn.Module,
    config: dict[str, Any],
    *,
    rank: int | None = None,
    scale_type: str | None = None,
    moment_codec: str = "fp32",
) -> list[dict[str, Any]]:
    """Partition unique trainable parameters into APOLLO 2D vs FP32 AdamW groups."""
    rank_val = int(rank if rank is not None else config.get("apollo_rank", 64))
    scale_kind = str(scale_type if scale_type is not None else config.get("apollo_scale_type", "channel"))
    if scale_kind not in ("channel", "tensor"):
        raise ValueError(f"Unsupported apollo_scale_type: {scale_kind}")
    min_dim = int(config.get("apollo_min_dim", 128))
    allow_embed = bool(config.get("apollo_include_embeddings", False))
    base_backbone_lr = float(
        config.get("apollo_learning_rate", config.get("backbone_lr", config.get("learning_rate", 1e-4)))
    )
    base_connector_lr = float(
        config.get("apollo_connector_learning_rate", config.get("adapter_lr", config.get("connector_learning_rate", 2e-4)))
    )
    base_vision_lr = float(config.get("vision_lr", base_backbone_lr * 0.5))
    base_output_lr = float(config.get("output_lr", config.get("audio_output_learning_rate", 5e-6)))
    apollo_scale = float(
        config.get("apollo_scale", 128.0 if scale_kind == "tensor" and rank_val == 1 else 1.0)
    )
    update_proj_gap = int(config.get("apollo_update_proj_gap", 200))
    norm_growth = config.get("apollo_norm_growth_bound", 1.01)
    weight_decay = float(config.get("weight_decay", 0.01))

    buckets: dict[tuple[Any, ...], list[nn.Parameter]] = {}
    names_by_bucket: dict[tuple[Any, ...], list[str]] = {}
    seen: set[int] = set()

    iterator = (
        model.named_groups()
        if hasattr(model, "named_groups")
        else (("default", n, p) for n, p in model.named_parameters())
    )
    for group_tag, name, param in iterator:
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        is_connector = (
            group_tag in ("connectors", "visual_adapters")
            or ".bridge." in name
            or "vision_projector" in name
            or "cross_attn" in name
        )
        is_vision = group_tag in ("vision", "motion", "speech_in", "sound_in")
        is_audio_out = group_tag in ("speech_output", "sound_output")
        no_decay = param.ndim < 2 or name.endswith(".bias") or "norm" in name.lower() or "gate" in name.lower()
        wd = 0.0 if no_decay else weight_decay

        target_lr = (
            base_connector_lr
            if is_connector
            else (base_output_lr if is_audio_out else (base_vision_lr if is_vision else base_backbone_lr))
        )

        if is_high_precision_small_or_sensitive_param(
            name, param, min_apollo_dim=min_dim, allow_embeddings_in_apollo=allow_embed
        ):
            key = ("adamw", target_lr, wd)
        else:
            key = ("apollo", target_lr, wd)
        buckets.setdefault(key, []).append(param)
        names_by_bucket.setdefault(key, []).append(name)

    param_groups: list[dict[str, Any]] = []
    for (kind, lr, wd), params in buckets.items():
        if kind == "apollo":
            param_groups.append(
                {
                    "params": params,
                    "param_names": names_by_bucket[(kind, lr, wd)],
                    "kind": "apollo",
                    "lr": lr,
                    "initial_lr": lr,
                    "weight_decay": wd,
                    "betas": tuple(config.get("apollo_betas", (0.9, 0.999))),
                    "eps": float(config.get("apollo_eps", 1e-8)),
                    "rank": rank_val,
                    "scale_type": scale_kind,
                    "apollo_scale": apollo_scale,
                    "update_proj_gap": update_proj_gap,
                    "norm_growth_bound": norm_growth,
                    "moment_codec": moment_codec,
                    "proj_type": str(config.get("apollo_proj_type", "normal")),
                    "persist_basis": bool(config.get("apollo_persist_basis", False)),
                    "stochastic_rounding": bool(config.get("stochastic_rounding", True)),
                }
            )
        else:
            param_groups.append(
                {
                    "params": params,
                    "param_names": names_by_bucket[(kind, lr, wd)],
                    "kind": "adamw",
                    "lr": lr,
                    "initial_lr": lr,
                    "weight_decay": wd,
                    "betas": tuple(config.get("adamw_betas", (0.9, 0.95))),
                    "eps": float(config.get("adamw_eps", 1e-8)),
                }
            )
    return param_groups


class ApolloMergedOptimizer(torch.optim.Optimizer):
    """Hybrid APOLLO + high-precision AdamW optimizer for merged gradient steps.

    Invariants:
    - Each trainable parameter appears at most once across all parameter groups.
    - APOLLO parameters retain projected moments (`exp_avg`, `exp_avg_sq`) in FP32
      or optional COAT dynamic-range-expanded FP8, while applying channelwise or
      tensorwise adaptive scaling to the full accumulated gradient.
    - Small/sensitive parameters run standard FP32 AdamW (with an automatic FP32
      master weight shadow when the model parameter is stored in BF16).
    - Full state (basis seed & refresh epoch, moments, norm-growth history, step
      counters, FP8 codec metadata, and stochastic-rounding RNG) survives
      `state_dict` / `load_state_dict`.
    """

    def __init__(
        self,
        params: Iterable[dict[str, Any]] | Iterable[nn.Parameter],
        *,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        seed: int = 39526,
        coat_group_size: int = 128,
        chunk_rows: int = 4096,
    ):
        defaults = dict(
            kind="adamw",
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=64,
            scale_type="channel",
            apollo_scale=1.0,
            update_proj_gap=200,
            norm_growth_bound=1.01,
            moment_codec="fp32",
            proj_type="normal",
            persist_basis=False,
            stochastic_rounding=True,
        )
        super().__init__(params, defaults)
        seen_ids: set[int] = set()
        self._param_slots: dict[int, int] = {}
        slot = 0
        for group in self.param_groups:
            group.setdefault("initial_lr", group["lr"])
            for p in group["params"]:
                pid = id(p)
                if pid in seen_ids:
                    raise ValueError("Shared/duplicate parameter registered twice in optimizer")
                seen_ids.add(pid)
                self._param_slots[pid] = slot
                slot += 1
        self.base_seed = int(seed)
        self.chunk_rows = int(chunk_rows)
        self.fp8_codec = CoatDynamicRangeFP8Codec(group_size=coat_group_size)
        self._rounding_cpu_gen = torch.Generator(device="cpu")
        self._rounding_cpu_gen.manual_seed((self.base_seed ^ 0xA5A55A5A) & ((1 << 63) - 1))
        self._cached_bases: dict[int, tuple[int, torch.Tensor]] = {}
        self.last_codec_stats: dict[str, float] = {
            "saturated_fraction": 0.0,
            "underflow_fraction": 0.0,
        }

    def _next_rounding_generator(self, device: torch.device) -> torch.Generator:
        draw_seed = int(
            torch.randint(0, (1 << 62) - 1, (1,), generator=self._rounding_cpu_gen).item()
        )
        if device.type == "cpu":
            gen = torch.Generator(device="cpu")
            gen.manual_seed(draw_seed)
            return gen
        gen = torch.Generator(device=device)
        gen.manual_seed(draw_seed)
        return gen

    def _get_or_refresh_basis(
        self,
        p: nn.Parameter,
        state: dict[str, Any],
        group: dict[str, Any],
        *,
        step: int,
        rank: int,
        proj_dim: int,
    ) -> tuple[torch.Tensor, bool]:
        gap = max(1, int(group["update_proj_gap"]))
        target_epoch = (step - 1) // gap
        prev_epoch = state.get("proj_refresh_epoch")
        refreshed = prev_epoch is not None and prev_epoch != target_epoch
        pid = id(p)

        if group.get("persist_basis") and "proj_basis" in state and not refreshed:
            basis = state["proj_basis"].to(device=p.device, dtype=torch.float32)
            state["proj_refresh_epoch"] = target_epoch
            return basis, False

        cached = self._cached_bases.get(pid)
        if cached is not None and cached[0] == target_epoch and cached[1].device == p.device:
            state["proj_refresh_epoch"] = target_epoch
            return cached[1], False

        new_basis = _make_projection_matrix(
            rank,
            proj_dim,
            seed=state["proj_seed"],
            refresh_epoch=target_epoch,
            param_slot=state["param_slot"],
            device=p.device,
            proj_type=group.get("proj_type", "normal"),
        )
        old_basis: torch.Tensor | None = None
        if refreshed:
            if cached is not None and cached[0] == prev_epoch:
                old_basis = cached[1].to(device=p.device, dtype=torch.float32)
            elif "proj_basis" in state:
                old_basis = state["proj_basis"].to(device=p.device, dtype=torch.float32)
            else:
                old_basis = _make_projection_matrix(
                    rank,
                    proj_dim,
                    seed=state["proj_seed"],
                    refresh_epoch=int(prev_epoch),
                    param_slot=state["param_slot"],
                    device=p.device,
                    proj_type=group.get("proj_type", "normal"),
                )
        state["proj_refresh_epoch"] = target_epoch
        if group.get("persist_basis"):
            state["proj_basis"] = new_basis.clone()
        self._cached_bases[pid] = (target_epoch, new_basis)
        if refreshed and old_basis is not None:
            state["_transient_old_basis"] = old_basis
        return new_basis, refreshed

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
        *,
        grad_scale: float = 1.0,
        grad_overrides: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        scale_factor = float(grad_scale)
        max_sat = 0.0
        max_uflow = 0.0

        for group in self.param_groups:
            kind = group.get("kind", "adamw")
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            wd = float(group["weight_decay"])

            for p in group["params"]:
                grad_raw = grad_overrides.get(id(p)) if grad_overrides is not None else p.grad
                if grad_raw is None:
                    continue
                if grad_raw.is_sparse:
                    raise RuntimeError("ApolloMergedOptimizer does not support sparse gradients")

                state = self.state[p]
                if kind == "adamw":
                    self._step_adamw_param(
                        p, grad_raw, state, lr=lr, beta1=beta1, beta2=beta2, eps=eps, wd=wd, grad_scale=scale_factor
                    )
                elif kind == "apollo":
                    sat, uflow = self._step_apollo_param(
                        p,
                        grad_raw,
                        state,
                        group,
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        eps=eps,
                        wd=wd,
                        grad_scale=scale_factor,
                    )
                    max_sat = max(max_sat, sat)
                    max_uflow = max(max_uflow, uflow)
                else:
                    raise ValueError(f"Unknown optimizer group kind: {kind}")

        self.last_codec_stats = {
            "saturated_fraction": max_sat,
            "underflow_fraction": max_uflow,
        }
        return loss

    def _step_adamw_param(
        self,
        p: nn.Parameter,
        grad_raw: torch.Tensor,
        state: dict[str, Any],
        *,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        wd: float,
        grad_scale: float,
    ) -> None:
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            if p.dtype != torch.float32:
                state["master_weight"] = p.detach().float().clone()

        state["step"] += 1
        t = int(state["step"])
        g = grad_raw.float()
        if grad_scale != 1.0:
            g = g * grad_scale

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

        bc1 = 1.0 - beta1**t
        bc2 = 1.0 - beta2**t
        step_size = lr / bc1
        denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)

        if "master_weight" in state:
            master = state["master_weight"]
            if wd != 0.0:
                master.mul_(1.0 - lr * wd)
            master.addcdiv_(exp_avg, denom, value=-step_size)
            p.copy_(master.to(p.dtype))
        else:
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            p.addcdiv_(exp_avg, denom, value=-step_size)

    def _step_apollo_param(
        self,
        p: nn.Parameter,
        grad_raw: torch.Tensor,
        state: dict[str, Any],
        group: dict[str, Any],
        *,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        wd: float,
        grad_scale: float,
    ) -> tuple[float, float]:
        if p.ndim != 2:
            raise ValueError(f"APOLLO group requires 2D matrix parameters, got shape {tuple(p.shape)}")
        rows, cols = p.shape
        proj_side = "left" if rows >= cols else "right"
        proj_dim = rows if proj_side == "left" else cols
        chan_dim = cols if proj_side == "left" else rows
        rank = min(int(group["rank"]), rows, cols)
        moment_shape = (rank, cols) if proj_side == "left" else (rows, rank)
        use_fp8_moments = group.get("moment_codec", "fp32") == "coat_fp8"

        if len(state) == 0:
            slot = self._param_slots[id(p)]
            state["step"] = 0
            state["param_slot"] = slot
            state["proj_seed"] = self.base_seed
            state["proj_side"] = proj_side
            state["rank"] = rank
            state["prev_scaled_norm"] = None
            zero_m = torch.zeros(moment_shape, device=p.device, dtype=torch.float32)
            zero_v = torch.zeros(moment_shape, device=p.device, dtype=torch.float32)
            if use_fp8_moments:
                state["exp_avg"] = self.fp8_codec.encode(zero_m, signed=True)
                state["exp_avg_sq"] = self.fp8_codec.encode(zero_v, signed=False)
            else:
                state["exp_avg"] = zero_m
                state["exp_avg_sq"] = zero_v

        state["step"] += 1
        t = int(state["step"])

        basis, refreshed = self._get_or_refresh_basis(
            p, state, group, step=t, rank=rank, proj_dim=proj_dim
        )

        # Decode moments to FP32 for exact APOLLO low-rank update.
        if isinstance(state["exp_avg"], dict) and state["exp_avg"].get("__coat_fp8__"):
            exp_avg = self.fp8_codec.decode(state["exp_avg"], device=p.device)
            exp_avg_sq = self.fp8_codec.decode(state["exp_avg_sq"], device=p.device)
        else:
            exp_avg = state["exp_avg"].to(device=p.device, dtype=torch.float32)
            exp_avg_sq = state["exp_avg_sq"].to(device=p.device, dtype=torch.float32)

        if refreshed and "_transient_old_basis" in state:
            old_basis = state.pop("_transient_old_basis")
            # Subspace transfer T = P_new @ P_old^T in R^{r x r}; channel variance mean is basis-invariant.
            transfer = basis @ old_basis.t()
            if proj_side == "left":
                exp_avg = transfer @ exp_avg
                exp_avg_sq = exp_avg_sq.mean(dim=0, keepdim=True).expand_as(exp_avg_sq).clone()
            else:
                exp_avg = exp_avg @ transfer.t()
                exp_avg_sq = exp_avg_sq.mean(dim=1, keepdim=True).expand_as(exp_avg_sq).clone()

        # Compute projected gradient R_t in FP32 by chunking along rows if large.
        if proj_side == "left":
            r_grad = torch.zeros((rank, cols), device=p.device, dtype=torch.float32)
            for start in range(0, rows, self.chunk_rows):
                end = min(rows, start + self.chunk_rows)
                g_chunk = grad_raw[start:end].float()
                if grad_scale != 1.0:
                    g_chunk = g_chunk * grad_scale
                r_grad.addmm_(basis[:, start:end], g_chunk)
        else:
            r_grad = torch.zeros((rows, rank), device=p.device, dtype=torch.float32)
            basis_t = basis.t()
            for start in range(0, rows, self.chunk_rows):
                end = min(rows, start + self.chunk_rows)
                g_chunk = grad_raw[start:end].float()
                if grad_scale != 1.0:
                    g_chunk = g_chunk * grad_scale
                r_grad[start:end] = g_chunk @ basis_t

        exp_avg.mul_(beta1).add_(r_grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(r_grad, r_grad, value=1.0 - beta2)

        bc1 = 1.0 - beta1**t
        bc2 = 1.0 - beta2**t
        m_hat = exp_avg / bc1
        v_hat_sqrt = exp_avg_sq.sqrt() / math.sqrt(bc2)
        norm_grad = m_hat / (v_hat_sqrt + eps)

        scale_type = group.get("scale_type", "channel")
        if scale_type == "channel":
            reduce_dim = 0 if proj_side == "left" else 1
            num = torch.linalg.vector_norm(norm_grad, dim=reduce_dim)
            den = torch.linalg.vector_norm(r_grad, dim=reduce_dim).add_(eps)
            channel_scale = num / den  # shape [chan_dim]
        elif scale_type == "tensor":
            num = torch.linalg.vector_norm(norm_grad)
            den = torch.linalg.vector_norm(r_grad).add_(eps)
            channel_scale = (num / den).expand(chan_dim)
        else:
            raise ValueError(f"Unsupported scale_type: {scale_type}")

        # Compute full-space scaled gradient norm in chunks to apply Fira norm-growth limiter without a full FP32 copy.
        sq_norm = torch.zeros((), device=p.device, dtype=torch.float32)
        for start in range(0, rows, self.chunk_rows):
            end = min(rows, start + self.chunk_rows)
            g_chunk = grad_raw[start:end].float()
            if grad_scale != 1.0:
                g_chunk = g_chunk * grad_scale
            if proj_side == "left":
                s_chunk = g_chunk * channel_scale.unsqueeze(0)
            else:
                s_chunk = g_chunk * channel_scale[start:end].unsqueeze(1)
            sq_norm = sq_norm + s_chunk.square().sum()

        current_norm = float(sq_norm.sqrt().item())
        prev_norm = state.get("prev_scaled_norm")
        limiter = 1.0
        growth_bound = group.get("norm_growth_bound")
        if growth_bound is not None and prev_norm is not None and prev_norm > 0.0:
            ratio = current_norm / (prev_norm + eps)
            if ratio > float(growth_bound):
                limiter = float(growth_bound) / ratio
                current_norm = current_norm * limiter
        state["prev_scaled_norm"] = current_norm

        effective_channel_scale = channel_scale * (float(group.get("apollo_scale", 1.0)) * limiter)

        # Persist updated projected moments (FP32 or COAT dynamic-range FP8).
        sat, uflow = 0.0, 0.0
        if use_fp8_moments:
            state["exp_avg"] = self.fp8_codec.encode(exp_avg, signed=True)
            sat = max(sat, self.fp8_codec.last_stats["saturated_fraction"])
            uflow = max(uflow, self.fp8_codec.last_stats["underflow_fraction"])
            state["exp_avg_sq"] = self.fp8_codec.encode(exp_avg_sq, signed=False)
            sat = max(sat, self.fp8_codec.last_stats["saturated_fraction"])
            uflow = max(uflow, self.fp8_codec.last_stats["underflow_fraction"])
        else:
            state["exp_avg"] = exp_avg
            state["exp_avg_sq"] = exp_avg_sq

        # Apply chunked full-space update + decoupled weight decay directly to p.
        use_sr = bool(group.get("stochastic_rounding", True)) and p.dtype == torch.bfloat16
        sr_gen = self._next_rounding_generator(p.device) if use_sr else None

        for start in range(0, rows, self.chunk_rows):
            end = min(rows, start + self.chunk_rows)
            w_chunk = p.data[start:end].float()
            g_chunk = grad_raw[start:end].float()
            if grad_scale != 1.0:
                g_chunk = g_chunk * grad_scale
            if proj_side == "left":
                upd = g_chunk * effective_channel_scale.unsqueeze(0)
            else:
                upd = g_chunk * effective_channel_scale[start:end].unsqueeze(1)
            if wd != 0.0:
                w_chunk.mul_(1.0 - lr * wd)
            w_chunk.add_(upd, alpha=-lr)
            if use_sr and sr_gen is not None:
                p.data[start:end].copy_(stochastic_round_fp32_to_bf16(w_chunk, sr_gen))
            else:
                p.data[start:end].copy_(w_chunk.to(p.dtype))

        return sat, uflow

    @staticmethod
    def _clone_state_tree(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.detach().clone()
        if isinstance(obj, dict):
            return {k: ApolloMergedOptimizer._clone_state_tree(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [ApolloMergedOptimizer._clone_state_tree(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(ApolloMergedOptimizer._clone_state_tree(v) for v in obj)
        return obj

    def state_dict(self) -> dict[str, Any]:
        base = self._clone_state_tree(super().state_dict())
        base["apollo_meta"] = {
            "base_seed": self.base_seed,
            "chunk_rows": self.chunk_rows,
            "rounding_cpu_rng": self._rounding_cpu_gen.get_state().clone(),
        }
        return base

    @staticmethod
    def _restore_exact_dtypes_and_device(raw_val: Any, device: torch.device) -> Any:
        if isinstance(raw_val, torch.Tensor):
            return raw_val.detach().to(device=device, dtype=raw_val.dtype).clone()
        if isinstance(raw_val, dict):
            return {
                k: ApolloMergedOptimizer._restore_exact_dtypes_and_device(v, device)
                for k, v in raw_val.items()
            }
        if isinstance(raw_val, list):
            return [ApolloMergedOptimizer._restore_exact_dtypes_and_device(v, device) for v in raw_val]
        if isinstance(raw_val, tuple):
            return tuple(ApolloMergedOptimizer._restore_exact_dtypes_and_device(v, device) for v in raw_val)
        return raw_val

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        copied = dict(state_dict)
        meta = copied.pop("apollo_meta", None)
        raw_states = copied.get("state", {})
        saved_groups = copied.get("param_groups", [])
        super().load_state_dict(copied)
        # PyTorch's Optimizer.load_state_dict casts floating state tensors to param.dtype (e.g. BF16),
        # which would truncate FP32 master weights, FP32 low-rank moments, and FP8 codec scales!
        # Restore exact saved dtypes on each parameter's target device.
        for cur_group, saved_group in zip(self.param_groups, saved_groups):
            for p, saved_pid in zip(cur_group["params"], saved_group["params"]):
                if saved_pid in raw_states:
                    self.state[p] = self._restore_exact_dtypes_and_device(raw_states[saved_pid], p.device)
                elif str(saved_pid) in raw_states:
                    self.state[p] = self._restore_exact_dtypes_and_device(raw_states[str(saved_pid)], p.device)
        self._cached_bases.clear()
        if meta is not None:
            self.base_seed = int(meta.get("base_seed", self.base_seed))
            self.chunk_rows = int(meta.get("chunk_rows", self.chunk_rows))
            if "rounding_cpu_rng" in meta:
                rng_tensor = meta["rounding_cpu_rng"]
                if isinstance(rng_tensor, torch.Tensor):
                    self._rounding_cpu_gen.set_state(rng_tensor.cpu())

    def memory_breakdown_bytes(self) -> dict[str, int]:
        apollo_moment_bytes = 0
        adamw_moment_bytes = 0
        master_weight_bytes = 0
        basis_cache_bytes = sum(t.numel() * t.element_size() for _, t in self._cached_bases.values())
        for group in self.param_groups:
            kind = group.get("kind", "adamw")
            for p in group["params"]:
                st = self.state.get(p, {})
                if not st:
                    continue
                for key in ("exp_avg", "exp_avg_sq"):
                    val = st.get(key)
                    if isinstance(val, torch.Tensor):
                        nbytes = val.numel() * val.element_size()
                    elif isinstance(val, dict) and val.get("__coat_fp8__"):
                        nbytes = self.fp8_codec.storage_bytes(val)
                    else:
                        nbytes = 0
                    if kind == "apollo":
                        apollo_moment_bytes += nbytes
                    else:
                        adamw_moment_bytes += nbytes
                if "master_weight" in st and isinstance(st["master_weight"], torch.Tensor):
                    mw = st["master_weight"]
                    master_weight_bytes += mw.numel() * mw.element_size()
                if "proj_basis" in st and isinstance(st["proj_basis"], torch.Tensor):
                    pb = st["proj_basis"]
                    basis_cache_bytes += pb.numel() * pb.element_size()
        return {
            "apollo_moment_bytes": apollo_moment_bytes,
            "adamw_moment_bytes": adamw_moment_bytes,
            "master_weight_bytes": master_weight_bytes,
            "basis_cache_bytes": basis_cache_bytes,
            "total_optimizer_state_bytes": (
                apollo_moment_bytes + adamw_moment_bytes + master_weight_bytes + basis_cache_bytes
            ),
        }


class MergedGradientAccumulator:
    """Single owned accumulation buffer per trainable parameter.

    Uses `register_post_accumulate_grad_hook` so leaf `.grad` tensors are merged
    into the owned buffer and cleared (`p.grad = None`) immediately as each
    layer finishes backward, preventing duplicate `.grad` + accumulator residency.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        accum_mode: str = "fp32",
        min_bf16_accum_dim: int = 128,
        allow_embeddings_in_apollo: bool = False,
        attach_hooks: bool = True,
    ):
        if accum_mode not in ("fp32", "bf16_apollo_fp32_small", "param_dtype"):
            raise ValueError(f"Unsupported accum_mode: {accum_mode}")
        self.model = model
        self.accum_mode = accum_mode
        self.min_bf16_accum_dim = int(min_bf16_accum_dim)
        self.allow_embeddings_in_apollo = bool(allow_embeddings_in_apollo)
        self.params: list[nn.Parameter] = []
        self.param_names: dict[int, str] = {}
        self.buffers: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []
        self._active_weight: float = 1.0
        self._hook_enabled: bool = attach_hooks
        self.completed_steps: int = 0
        self.skipped_nonfinite_steps: int = 0
        self.microbatches_in_step: int = 0
        self.units_in_step: int = 0

        seen: set[int] = set()
        for name, p in model.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            self.params.append(p)
            self.param_names[id(p)] = name
            buf_dtype = self._buffer_dtype_for(name, p)
            self.buffers[id(p)] = torch.zeros_like(p, dtype=buf_dtype)
            if attach_hooks and hasattr(p, "register_post_accumulate_grad_hook"):
                self._handles.append(p.register_post_accumulate_grad_hook(self._make_hook(id(p))))

    def _buffer_dtype_for(self, name: str, p: nn.Parameter) -> torch.dtype:
        if self.accum_mode == "fp32":
            return torch.float32
        if self.accum_mode == "param_dtype":
            return p.dtype
        # bf16_apollo_fp32_small: keep norms/gates/biases in FP32, large 2D matrices in BF16.
        if is_high_precision_small_or_sensitive_param(
            name,
            p,
            min_apollo_dim=self.min_bf16_accum_dim,
            allow_embeddings_in_apollo=self.allow_embeddings_in_apollo,
        ):
            return torch.float32
        return torch.bfloat16

    def _make_hook(self, pid: int) -> Callable[[torch.Tensor], None]:
        def _hook(param: torch.Tensor) -> None:
            if not self._hook_enabled or param.grad is None:
                return
            buf = self.buffers[pid]
            g = param.grad.detach()
            if buf.dtype == g.dtype:
                buf.add_(g, alpha=self._active_weight)
            else:
                buf.add_(g.to(buf.dtype), alpha=self._active_weight)
            param.grad = None

        return _hook

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def buffer_bytes(self) -> int:
        return sum(buf.numel() * buf.element_size() for buf in self.buffers.values())

    def zero_buffers(self) -> None:
        for p in self.params:
            p.grad = None
        for buf in self.buffers.values():
            buf.zero_()
        self.microbatches_in_step = 0
        self.units_in_step = 0

    zero_grad = zero_buffers

    class _MicrobatchScope:
        def __init__(self, owner: "MergedGradientAccumulator", weight: float, units: int):
            self.owner = owner
            self.weight = float(weight)
            self.units = int(units)
            self.prev_weight = 1.0

        def __enter__(self) -> "MergedGradientAccumulator._MicrobatchScope":
            self.prev_weight = self.owner._active_weight
            self.owner._active_weight = self.weight
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            self.owner._active_weight = self.prev_weight
            if exc_type is None:
                if not self.owner._handles:
                    self.owner.merge_existing_grads(weight=self.weight)
                self.owner.microbatches_in_step += 1
                self.owner.units_in_step += self.units

    def microbatch(self, *, weight: float = 1.0, supervised_units: int = 1) -> _MicrobatchScope:
        return self._MicrobatchScope(self, weight=weight, units=supervised_units)

    @torch.no_grad()
    def merge_existing_grads(self, *, weight: float = 1.0) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            buf = self.buffers[id(p)]
            g = p.grad.detach()
            if buf.dtype == g.dtype:
                buf.add_(g, alpha=weight)
            else:
                buf.add_(g.to(buf.dtype), alpha=weight)
            p.grad = None

    @torch.no_grad()
    def rescale_buffers(self, factor: float) -> None:
        if factor == 1.0:
            return
        for buf in self.buffers.values():
            buf.mul_(factor)

    @torch.no_grad()
    def step(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
        *,
        max_grad_norm: float = 1.0,
        post_scale: float = 1.0,
        invalidate_caches_fn: Callable[[], None] | None = None,
    ) -> StepDiagnostics:
        """Reduce, check finiteness, clip globally once, step once, clear & invalidate."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world = torch.distributed.get_world_size()
            if world > 1:
                for buf in self.buffers.values():
                    torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.AVG)

        total_sq = torch.zeros((), device=self.params[0].device if self.params else "cpu", dtype=torch.float32)
        all_finite = True
        for buf in self.buffers.values():
            if not torch.isfinite(buf).all():
                all_finite = False
                break
            bf = buf.float()
            if post_scale != 1.0:
                bf = bf * post_scale
            total_sq = total_sq + bf.square().sum()

        microbatches = self.microbatches_in_step
        units = self.units_in_step

        if not all_finite:
            self.zero_buffers()
            self.skipped_nonfinite_steps += 1
            return StepDiagnostics(
                applied=False,
                finite=False,
                grad_norm=float("nan"),
                clip_coef=0.0,
                supervised_units=units,
                microbatches=microbatches,
                step=self.completed_steps,
            )

        grad_norm = float(total_sq.sqrt().item())
        if not math.isfinite(grad_norm):
            self.zero_buffers()
            self.skipped_nonfinite_steps += 1
            return StepDiagnostics(
                applied=False,
                finite=False,
                grad_norm=grad_norm,
                clip_coef=0.0,
                supervised_units=units,
                microbatches=microbatches,
                step=self.completed_steps,
            )

        clip_coef = 1.0
        if max_grad_norm is not None and max_grad_norm > 0.0 and grad_norm > max_grad_norm:
            clip_coef = float(max_grad_norm) / (grad_norm + 1e-6)
        combined_scale = float(post_scale) * clip_coef

        prev_hook = self._hook_enabled
        self._hook_enabled = False
        try:
            if isinstance(optimizer, ApolloMergedOptimizer):
                optimizer.step(grad_scale=combined_scale, grad_overrides=self.buffers)
            else:
                for p in self.params:
                    buf = self.buffers[id(p)]
                    if combined_scale != 1.0:
                        buf.mul_(combined_scale)
                    p.grad = buf if buf.dtype == p.dtype else buf.to(p.dtype)
                optimizer.step()
        finally:
            for p in self.params:
                p.grad = None
            self._hook_enabled = prev_hook

        self.zero_buffers()
        if invalidate_caches_fn is not None:
            invalidate_caches_fn()
        if scheduler is not None:
            scheduler.step()
        self.completed_steps += 1
        return StepDiagnostics(
            applied=True,
            finite=True,
            grad_norm=grad_norm,
            clip_coef=clip_coef,
            supervised_units=units,
            microbatches=microbatches,
            step=self.completed_steps,
        )

    def run_effective_batch(
        self,
        microbatches: list[Any],
        compute_fn: Callable[[Any], tuple[torch.Tensor, int]],
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
        *,
        normalization: str = "per_example",
        microbatch_size_fn: Callable[[Any], int] = len,
        nominal_unit_scale: float = 256.0,
        max_grad_norm: float = 1.0,
        invalidate_caches_fn: Callable[[], None] | None = None,
    ) -> tuple[float, StepDiagnostics]:
        """Accumulate an effective batch (including uneven final batches) and step once.

        - `normalization='per_example'`: compute_fn returns `(mean_microbatch_loss, units)`.
          Each microbatch is weighted by `len(mb) / sum(len(m) for m in microbatches)`,
          matching a monolithic batch mean regardless of uneven tail sizes.
        - `normalization='per_token'`: compute_fn returns `(mean_per_token_loss, token_count)`.
          Backward scales each microbatch by `token_count / nominal_unit_scale` and rescales
          once by `nominal_unit_scale / total_tokens` before clipping and stepping.
        """
        if not microbatches:
            raise ValueError("microbatches must not be empty")
        if normalization not in ("per_example", "per_token"):
            raise ValueError(f"Unsupported loss normalization: {normalization}")

        self.zero_buffers()
        if normalization == "per_example":
            counts = [int(microbatch_size_fn(mb)) for mb in microbatches]
            total_examples = sum(counts)
            if total_examples <= 0:
                raise ValueError("Effective batch has zero examples")
            weighted_loss_sum = 0.0
            for mb, count in zip(microbatches, counts):
                weight = count / total_examples
                loss, units = compute_fn(mb)
                with self.microbatch(weight=weight, supervised_units=units):
                    loss.backward()
                weighted_loss_sum += float(loss.detach().item()) * weight
            diag = self.step(
                optimizer,
                scheduler,
                max_grad_norm=max_grad_norm,
                post_scale=1.0,
                invalidate_caches_fn=invalidate_caches_fn,
            )
            return weighted_loss_sum, diag

        total_tokens = 0
        token_loss_sum = 0.0
        scale_ref = max(1.0, float(nominal_unit_scale))
        for mb in microbatches:
            loss, tokens = compute_fn(mb)
            tokens = int(tokens)
            if tokens <= 0:
                raise ValueError("per_token normalization requires positive token count")
            weight = tokens / scale_ref
            with self.microbatch(weight=weight, supervised_units=tokens):
                loss.backward()
            total_tokens += tokens
            token_loss_sum += float(loss.detach().item()) * tokens

        post_scale = scale_ref / total_tokens
        diag = self.step(
            optimizer,
            scheduler,
            max_grad_norm=max_grad_norm,
            post_scale=post_scale,
            invalidate_caches_fn=invalidate_caches_fn,
        )
        return token_loss_sum / total_tokens, diag
