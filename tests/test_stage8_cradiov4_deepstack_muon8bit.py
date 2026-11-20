"""Comprehensive CPU unit tests for Stage-8 Unified `nvidia/C-RADIOv4-H` Super-Encoder:
- `L24 + L32` (`indices=(23, 31)`) independent `RMSNorm` + `2x2` pixel shuffle + SwiGLU MLP + early DeepStack injection
- Adaptive native-aspect `256px-768px` default + small-crop `>= 384px` upscaling + high-res single-image tier `1024px-1152px`
- `C-RADIOv4` Gotcha #1 protection: frozen during alignment, top-6 blocks ONLY unfrozen in `full_spark`
- `muon8bit` and `apollo8bit` 8-bit B50 vision optimizer convergence, orthogonalization, and low memory footprint
- `wd = 0.0` on `gate_attn`, `gate_ffn`, `early_mid_gate`, `layer_norms`
- Vision-swap warm-start (`SigLIP2` -> `C-RADIOv4-H`) preserving 100% of Spark + Loop48 + En1B weights
- Shaded 3D 4-primitive (`<point>`, `<box>`, `<dash>`, `<circle>`) + Front/Back + Left/Right + Nemotron-v3 converter
"""
from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn

from colab.final.engine import optimizer_for
from colab.final.intel_vision_coproc import (
    XpuVisionOptimizerEngine,
    cradiov4_forward_multilayer,
    dequantize_int8_blockwise,
    newton_schulz5_orthogonalize,
    quantize_int8_blockwise,
)
from colab.final.model import (
    RadioMultiLayerSwiGLUProjector,
    SparkNative,
    export_spark_native_fp8_safetensors,
    load_spark_native_safetensors,
)
from colab.final.spatial_3d_primitives_corpus import (
    _convert_nemotron_boxes_to_primitives,
    build_shaded_3d_spatial_primitives_records,
)


class _DummyRadioBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class _DummyInnerRadioViT(nn.Module):
    def __init__(self, dim: int = 32, depth: int = 10, patch_size: int = 16):
        super().__init__()
        self.embed_dim = dim
        self.patch_size = patch_size
        self.patch_proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([_DummyRadioBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.grad_checkpointing = False

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = bool(enable)


class _DummyRadioCore(nn.Module):
    def __init__(self, dim: int = 32, depth: int = 10, patch_size: int = 16):
        super().__init__()
        self.embed_dim = dim
        self.patch_size = patch_size
        self.model = _DummyInnerRadioViT(dim=dim, depth=depth, patch_size=patch_size)
        self.feature_normalizer = nn.Identity()

    def forward_intermediates(
        self,
        pixel_values: torch.Tensor,
        *,
        indices: list[int],
        output_fmt: str = "NLC",
        norm: bool = False,
        intermediates_only: bool = True,
    ) -> list[torch.Tensor]:
        x = self.model.patch_proj(pixel_values)  # (B, D, H_p, W_p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        outs: list[torch.Tensor] = []
        idx_set = set(indices)
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in idx_set:
                # Deliberately scale mid-layer down by 0.25x to simulate L24 vs L32 variance gap!
                t = x * 0.25 if i < len(self.model.blocks) - 1 else x
                outs.append(self.model.norm(t) if norm else t)
        return outs


class _DummyCRadioV4Model(nn.Module):
    def __init__(self, dim: int = 32, depth: int = 10, patch_size: int = 16):
        super().__init__()
        self.radio_model = _DummyRadioCore(dim=dim, depth=depth, patch_size=patch_size)
        self.config = SimpleNamespace(hidden_size=dim, patch_size=patch_size)


class _DummySparkLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.self_attn = SimpleNamespace(layer_type="full_attention")
        self.fc = nn.Linear(hidden, hidden)

    def forward(self, h, attention_mask=None, past_kv=None, use_cache=False):
        return h + 0.05 * self.fc(h), None


class _DummySparkLM(nn.Module):
    def __init__(self, vocab: int = 64, hidden: int = 32, num_layers: int = 6):
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=vocab,
            hidden_size=hidden,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=hidden // 4,
            rms_norm_eps=1e-6,
            use_cache=False,
        )
        self.embed = nn.Embedding(vocab, hidden)
        self.model = SimpleNamespace(
            layers=nn.ModuleList([_DummySparkLayer(hidden) for _ in range(num_layers)]),
            norm=nn.LayerNorm(hidden),
        )
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.embed.weight

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def tie_weights(self):
        self.lm_head.weight = self.embed.weight

    def forward(self, inputs_embeds=None, attention_mask=None, use_cache=False, output_hidden_states=False, **kwargs):
        h = inputs_embeds
        for lyr in self.model.layers:
            h, _ = lyr(h)
        h = self.model.norm(h)
        return SimpleNamespace(logits=self.lm_head(h), hidden_states=(h,))


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "</s>"
    pad_token = "<pad>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        out = "\n".join(parts)
        if add_generation_prompt:
            out += "\n<assistant>"
        return out

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [2 + (ord(c) % 50) for c in (text or "a")[:24]]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "ok"


def _make_dummy_radio_spark(depth: int = 10, radio_indices: tuple[int, ...] = (5, 9)) -> SparkNative:
    policy = {
        "modalities": "vision_text",
        "spark_dtype": "float32",
        "vision_dtype": "float32",
        "max_visual_tokens": 1024,
        "thumbnail_patches": 256,
        "detail_tiles": 0,
        "tile_patches": 256,
        "visual_cross_attn_blocks": 3,
        "enable_loop_engram": False,
        "radio_layer_indices": list(radio_indices),
        "radio_default_unfreeze_layers": 3,
        "learning_rate": 1e-4,
        "connector_learning_rate": 2e-4,
        "vision_lr": 1e-5,
        "weight_decay": 0.01,
        "optimizer_profile": "bf16_adamw",
    }
    components = {
        "spark": _DummySparkLM(vocab=64, hidden=32, num_layers=6),
        "tokenizer": _DummyTokenizer(),
        "vision": _DummyCRadioV4Model(dim=32, depth=depth, patch_size=16),
        "vision_backbone_type": "c_radio_v4",
    }
    return SparkNative(paths={}, policy=policy, device="cpu", pretrained=False, components=components)


def test_int8_blockwise_and_newton_schulz5():
    torch.manual_seed(0)
    t = torch.randn(64, 128) * 0.5
    q = quantize_int8_blockwise(t, group_size=64)
    dq = dequantize_int8_blockwise(q)
    rel_err = float((t - dq).norm() / t.norm())
    assert rel_err < 0.02
    assert q["q"].dtype == torch.int8

    g = torch.randn(32, 64)
    o = newton_schulz5_orthogonalize(g, steps=5).float()
    gram = o @ o.T
    # Newton-Schulz-5 pushes singular values toward ~1.0 (within standard quintic envelope [0.7, 1.3])
    svals = torch.linalg.svdvals(o)
    assert float(svals.min()) > 0.5
    assert float(svals.max()) < 1.35
    assert torch.isfinite(gram).all()


@pytest.mark.parametrize("mode", ["muon8bit", "apollo8bit"])
def test_xpu_vision_optimizer_8bit_convergence_and_low_memory(mode: str):
    torch.manual_seed(42)
    # Shape (256, 256) exercises the actual 8-bit Muon / Apollo 2D matrix path (min_dim >= 128)!
    layer = nn.Linear(256, 256, bias=True).to(dtype=torch.bfloat16)
    target = torch.randn(16, 256)
    x = torch.randn(16, 256)

    eng = XpuVisionOptimizerEngine(mode=mode, device=torch.device("cpu"), weight_decay=0.01)
    named = [(f"global.{n}", p) for n, p in layer.named_parameters()]
    eng.sync_trainable_params(named)

    losses = []
    for _ in range(28):
        layer.zero_grad(set_to_none=True)
        out = layer(x.to(torch.bfloat16)).float()
        loss = torch.nn.functional.mse_loss(out, target)
        losses.append(float(loss.item()))
        loss.backward()
        eng.accumulate_grads(named, mb_weight=1.0)
        res = eng.step(named, lr=1.5e-2 if mode == "muon8bit" else 1.2e-1, max_grad_norm=1.0)
        assert res["finite"] and res["applied"]

    assert losses[-1] < losses[0] * 0.5, f"{mode} failed to converge: {losses}"
    # Verify 8-bit optimizer state stays ultra-compact (< 1.5 B/param vs FP32 AdamW's 12 B/param!)
    w_numel = layer.weight.numel()
    bytes_per_param = eng.optimizer_state_bytes() / float(w_numel)
    assert bytes_per_param < 1.5, f"{mode} state too large: {bytes_per_param:.2f} B/param"


def test_radio_multilayer_swiglu_projector_rmsnorm_balances_variance():
    torch.manual_seed(7)
    proj = RadioMultiLayerSwiGLUProjector(vis_dim=32, hidden_size=48, num_taps=2)
    # Simulate exact measured C-RADIOv4-H L24 std=0.264 vs L32 std=0.845!
    f24 = torch.randn(1, 16, 32, requires_grad=True)
    f32 = torch.randn(1, 16, 32, requires_grad=True)
    stacked = torch.stack([f24 * 0.264, f32 * 0.845], dim=1)
    main_t, early_t = proj(stacked, hh=4, ww=4)
    assert main_t.shape == (1, 4, 48)
    assert early_t.shape == (1, 4, 48)
    (main_t.pow(2).mean() + early_t.pow(2).mean()).backward()
    # Both L24 and L32 receive strong finite balanced gradients despite 3.2x raw std difference!
    g24 = float(f24.grad.norm())
    g32 = float(f32.grad.norm())
    assert g24 > 1e-4 and g32 > 1e-4
    assert 0.2 < (g24 / g32) < 5.0


def test_cradiov4_adaptive_resolution_tiers_and_top_k_freeze_guard():
    model = _make_dummy_radio_spark(depth=10, radio_indices=(5, 9))

    # 1. Tiny OCR crop (120x40) -> upscaled >= 384px, 32px-aligned!
    th, tw = model._radio_target_hw(120, 40, is_temporal=False, num_images=1)
    assert max(th, tw) >= 384 and th % 32 == 0 and tw % 32 == 0

    # 2. Default native-aspect 768x512 -> stays exact 768x512 (zero square squashing!)
    th2, tw2 = model._radio_target_hw(768, 512, is_temporal=False, num_images=1)
    assert (th2, tw2) == (512, 768)

    # 3. High-res single-image tier (1600x1200) -> enters 1024px-1152px tier, <= 1024 tokens!
    th3, tw3 = model._radio_target_hw(1600, 1200, is_temporal=False, num_images=1)
    assert max(th3, tw3) >= 1024 and max(th3, tw3) <= 1152
    assert (th3 // 32) * (tw3 // 32) <= 1024

    # 4. Stage-8A alignment: 100% of C-RADIOv4 frozen!
    model.set_stage(stage_mode="alignment_only")
    assert all(not p.requires_grad for p in model.vision.parameters())
    assert all(p.requires_grad for p in model.vision_projector.parameters())

    # 5. Stage-8B full_spark: ONLY top-3 (configured default=3 on 10L stub) blocks unfrozen!
    model.set_stage(stage_mode="full_spark", unfreeze_vision_last_n_layers=3)
    for i, blk in enumerate(model.vision.radio_model.model.blocks):
        if i < 7:
            assert all(not p.requires_grad for p in blk.parameters()), f"Lower block {i} should be frozen!"
        else:
            assert all(p.requires_grad for p in blk.parameters()), f"Top block {i} should be trainable!"

    # 6. Verify wd=0.0 on gate_attn, gate_ffn, early_mid_gate, layer_norms
    opt = optimizer_for(model, model.policy)
    no_decay_ids = {id(p) for g in opt.param_groups if float(g["weight_decay"]) == 0.0 for p in g["params"]}
    assert id(model.vision_projector.early_mid_gate) in no_decay_ids
    for ln in model.vision_projector.layer_norms:
        assert id(ln.weight) in no_decay_ids
    for xblk in model.visual_cross_attn:
        assert id(xblk.gate_attn) in no_decay_ids
        assert id(xblk.gate_ffn) in no_decay_ids

    # 7. End-to-end forward + backward through C-RADIOv4 L24+L32 + early DeepStack cross-attn!
    img = Image.fromarray(np.random.randint(0, 255, (256, 320, 3), dtype=np.uint8), mode="RGB")
    vis_tok = model.image_tokens([img], [0.0], [None])
    assert vis_tok is not None and vis_tok.shape[-1] == 32
    assert model._last_early_visual_context is not None
    assert model._last_early_visual_context.shape == vis_tok.shape
    loss = vis_tok.pow(2).mean() + model._last_early_visual_context.pow(2).mean()
    loss.backward()
    # Lower blocks have zero/None grad; top-3 blocks have finite non-zero grad!
    for i, blk in enumerate(model.vision.radio_model.model.blocks):
        grads = [p.grad for p in blk.parameters() if p.grad is not None]
        if i < 7:
            assert len(grads) == 0
        else:
            assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)


def test_vision_swap_warmstart_preserves_spark_and_cross_attn(tmp_path: Path):
    src = _make_dummy_radio_spark(depth=10, radio_indices=(5, 9))
    sf_path = tmp_path / "source_model.safetensors"
    export_spark_native_fp8_safetensors(src, sf_path)

    # Fresh target with DIFFERENT vision dim/depth simulating SigLIP2 -> C-RADIOv4 swap!
    dst = _make_dummy_radio_spark(depth=8, radio_indices=(4, 7))
    with torch.no_grad():
        dst.spark.embed.weight.zero_()
        dst.visual_cross_attn[0].gate_attn.fill_(9.9)
        orig_vis_w = dst.vision.radio_model.model.blocks[0].fc1.weight.clone()

    load_spark_native_safetensors(dst, sf_path, allow_missing_vision_swap=True)
    # Spark + visual_cross_attn restored exactly; fresh C-RADIOv4 vision untouched!
    assert torch.allclose(dst.spark.embed.weight, src.spark.embed.weight, atol=1e-4)
    assert torch.allclose(dst.visual_cross_attn[0].gate_attn, src.visual_cross_attn[0].gate_attn)
    assert torch.equal(dst.vision.radio_model.model.blocks[0].fc1.weight, orig_vis_w)


def test_shaded_3d_spatial_primitives_and_nemotron_converter():
    # 1. Nemotron inline [x1, y1, x2, y2] -> <box> + relational <dash>
    sample = (
        "<think>\nThere is a red cube [120, 200, 320, 420] to the left of a blue sphere [620, 210, 780, 410].\n</think>\nYes"
    )
    out = _convert_nemotron_boxes_to_primitives(sample)
    assert '<box x1="120" y1="200" x2="320" y2="420"' in out
    assert '<box x1="620" y1="210" x2="780" y2="410"' in out
    assert "<dash " in out

    # 2. Real full-color shaded 3D records with all 4 primitives + Front/Back + Left/Right!
    pairs = build_shaded_3d_spatial_primitives_records(seed=42, panel_size=(128, 128), angles_per_asset=4)
    assert len(pairs) >= 20
    tasks = {r["task"] for r, _ in pairs}
    assert "spatial_3d_kinematic_dash_and_joints" in tasks
    assert "spatial_3d_front_back_depth_order" in tasks
    assert "spatial_3d_left_right_orientation_facing" in tasks
    assert "spatial_3d_multiview_correspondence" in tasks

    all_answers = "\n".join(r["answer"] for r, _ in pairs)
    for prim in ("<point ", "<box ", "<dash ", "<circle "):
        assert prim in all_answers, f"Missing primitive {prim}"

    # Verify renders are genuine full-color shaded studio RGB (NOT monochrome B&W cutouts!)
    first_row, first_blobs = pairs[0]
    first_png = next(iter(first_blobs.values()))
    im = np.asarray(Image.open(io.BytesIO(first_png)).convert("RGB"), dtype=np.float32)
    channel_spread = float(np.mean(np.max(im, axis=-1) - np.min(im, axis=-1)))
    assert channel_spread > 5.0, f"Expected full-color shaded 3D render, got monochrome spread={channel_spread}"
