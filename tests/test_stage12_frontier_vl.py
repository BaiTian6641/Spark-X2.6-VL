"""Stage-12 Frontier VL unit tests: RadioMultiLayerGELUProjector, 2D row-break newlines, FP32-AdamW routing,
realistic scene-word OCR crops, and SyncCraft unseen v28 syntax-slip repair."""
from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from colab.final.engine import _save_model_safetensors
from colab.final.frontier_stage9_corpus import (
    build_realistic_scene_word_ocr_pairs,
    build_v28_grammar_repair_records,
)
from colab.final.model import (
    RadioMultiLayerGELUProjector,
    SparkNative,
    export_spark_native_fp8_safetensors,
    load_spark_native_safetensors,
)
from colab.final.optimizer import build_apollo_param_groups


class _MiniTokenizer:
    eos_token = "</s>"
    eos_token_id = 1
    pad_token = "</s>"
    pad_token_id = 1
    vocab_size = 64

    def __call__(self, text: str, add_special_tokens: bool = False):
        if text == "\n":
            ids = [7]
        else:
            ids = [2 + (ord(c) % 60) for c in text[:32]] or [2]
        return SimpleNamespace(input_ids=ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        out = []
        for m in messages:
            out.append(f"<{m['role']}>{m['content']}</{m['role']}>")
        if add_generation_prompt:
            out.append("<assistant>" + ("<think>\n" if enable_thinking else ""))
        return "".join(out)


class _MiniSparkLayer(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.self_attn = nn.Linear(hidden, hidden)
        self.mlp = nn.Linear(hidden, hidden)

    def forward(self, x, attention_mask=None, past_kv=None, use_cache=False):
        return x + 0.01 * self.mlp(torch.tanh(self.self_attn(x))), None


class _MiniSparkLM(nn.Module):
    def __init__(self, hidden: int = 128, vocab: int = 64):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden, vocab_size=vocab, use_cache=False)
        self.generation_config = SimpleNamespace()
        emb = nn.Embedding(vocab, hidden)
        nn.init.normal_(emb.weight, mean=0.0, std=0.028)
        self.model = SimpleNamespace(
            embed_tokens=emb,
            layers=nn.ModuleList([_MiniSparkLayer(hidden) for _ in range(2)]),
            norm=nn.LayerNorm(hidden),
            gradient_checkpointing=False,
        )
        # Register as proper child module so parameters appear in state_dict
        self._inner = nn.Module()
        self._inner.embed_tokens = emb
        self._inner.layers = self.model.layers
        self._inner.norm = self.model.norm
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = emb.weight

    def get_input_embeddings(self):
        return self._inner.embed_tokens

    def tie_weights(self):
        self.lm_head.weight = self._inner.embed_tokens.weight


class _MiniRadioBackbone(nn.Module):
    def __init__(self, embed_dim: int = 64, patch_size: int = 16):
        super().__init__()
        self.radio_model = SimpleNamespace(embed_dim=embed_dim, patch_size=patch_size)
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor):
        x = self.proj(pixel_values).flatten(2).transpose(1, 2)
        return x, x


def test_radio_multilayer_gelu_projector_shape_rms_and_grad():
    torch.manual_seed(0)
    proj = RadioMultiLayerGELUProjector(vis_dim=64, hidden_size=128, num_taps=2, intermediate_size=256, target_rms=0.025)
    assert proj.output_rms_match is False
    assert proj.normalize_meta_add is True
    assert not hasattr(proj, "out_norm")

    feats = torch.randn(2, 2, 64, 64, requires_grad=True)  # 8x8 patches -> 4x4 merged = 16 tokens
    main_tok, early_tok = proj(feats, hh=8, ww=8)
    assert main_tok.shape == (2, 16, 128)
    assert early_tok.shape == (2, 16, 128)

    main_rms = float(main_tok.detach().float().pow(2).mean().sqrt().item())
    assert 0.010 <= main_rms <= 0.050, f"Unexpected initial output RMS {main_rms}"

    loss = (main_tok.pow(2).mean() + early_tok.pow(2).mean())
    loss.backward()
    for name, p in proj.named_parameters():
        if name == "meta_scale":
            continue
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"Bad grad for {name}"


def test_spark_native_gelu_row_break_newline_and_roundtrip(tmp_path: Path, monkeypatch):
    torch.manual_seed(7)
    policy = {
        "modalities": "vision_text",
        "max_visual_tokens": 512,
        "thumbnail_patches": 256,
        "detail_tiles": 0,
        "visual_token_dropout": 0.0,
        "visual_cross_attn_blocks": 0,
        "enable_loop_engram": False,
        "radio_projector_type": "gelu",
        "radio_projector_intermediate_size": 256,
        "radio_projector_target_rms": 0.025,
        "radio_meta_scale_init": 0.004,
        "radio_row_break_newline": True,
        "radio_output_rms_match": False,
        "radio_layer_indices": [0, 1],
        "radio_small_crop_min_side": 128,
        "radio_min_side": 128,
        "radio_default_max_side": 256,
        "radio_default_max_patches": 256,
        "spark_dtype": "float32",
        "vision_dtype": "float32",
    }

    def _fake_multilayer(vision, pv, indices=(0, 1)):
        x = vision.proj(pv).flatten(2).transpose(1, 2)  # [B, N, D]
        return torch.stack([x, x * 1.1 + 0.05], dim=1)

    import colab.final.intel_vision_coproc as ivc

    monkeypatch.setattr(ivc, "cradiov4_forward_multilayer", _fake_multilayer)

    comps = {
        "spark": _MiniSparkLM(hidden=128, vocab=64),
        "vision": _MiniRadioBackbone(embed_dim=64, patch_size=16),
        "tokenizer": _MiniTokenizer(),
        "vision_backbone_type": "c_radio_v4",
    }
    model = SparkNative(paths=None, policy=policy, device="cpu", pretrained=False, components=comps)
    assert isinstance(model.vision_projector, RadioMultiLayerGELUProjector)
    assert model.radio_row_break_newline is True
    assert model._row_break_nl_id == 7

    # 128x192 image -> hh=8, ww=12 -> gh=4, gw=6 -> with row-break newlines: 4 * (6 + 1) = 28 tokens!
    img = Image.new("RGB", (192, 128), (180, 90, 40))
    vis_tok = model.image_tokens([img], [0.0], None)
    assert vis_tok is not None
    assert vis_tok.shape == (1, 28, 128), f"Expected (1, 28, 128) with row-breaks, got {tuple(vis_tok.shape)}"
    assert model._last_early_visual_context is not None
    assert model._last_early_visual_context.shape == vis_tok.shape

    # Verify exact newline embedding placed at end of each of the 4 rows (indices 6, 13, 20, 27)!
    expected_nl = model.get_input_embeddings()(torch.tensor([7])).detach().squeeze(0)
    for r in range(4):
        row_end = vis_tok[0, r * 7 + 6]
        assert torch.allclose(row_end, expected_nl, atol=1e-6), f"Row {r} end token did not match newline embedding!"

    # Verify optimizer routes 100% of RadioMultiLayerGELUProjector params to full-rank FP32 AdamW (zero APOLLO)
    model.set_stage(alignment_only=False, stage_mode="full_spark")
    groups = build_apollo_param_groups(model, {"learning_rate": 1e-5, "connector_learning_rate": 1e-4, "apollo_min_dim": 64})
    proj_param_ids = {id(p) for p in model.vision_projector.parameters()}
    found = 0
    for g in groups:
        for p in g["params"]:
            if id(p) in proj_param_ids:
                assert g.get("is_apollo", False) is False, "vision_projector must never enter low-rank APOLLO"
                found += 1
    assert found == len(proj_param_ids)

    # Verify strict BF16/FP32 and FP8 safetensors round-trip!
    sf_path = tmp_path / "model.safetensors"
    _save_model_safetensors(model, sf_path)
    comps2 = {
        "spark": _MiniSparkLM(hidden=128, vocab=64),
        "vision": _MiniRadioBackbone(embed_dim=64, patch_size=16),
        "tokenizer": _MiniTokenizer(),
        "vision_backbone_type": "c_radio_v4",
    }
    model2 = SparkNative(paths=None, policy=policy, device="cpu", pretrained=False, components=comps2)
    load_spark_native_safetensors(model2, sf_path)
    assert torch.allclose(model.vision_projector.fc2.weight, model2.vision_projector.fc2.weight)

    fp8_path = tmp_path / "model_fp8.safetensors"
    export_spark_native_fp8_safetensors(model, fp8_path)
    model3 = SparkNative(paths=None, policy=policy, device="cpu", pretrained=False, components=comps2)
    load_spark_native_safetensors(model3, fp8_path)
    assert torch.allclose(model.vision_projector.fc1.weight, model3.vision_projector.fc1.weight)


def test_realistic_scene_word_ocr_pairs_quality_and_chroma():
    pairs = build_realistic_scene_word_ocr_pairs(max_items=24, seed=42)
    assert len(pairs) == 24
    submodes = set()
    for rec, blobs in pairs:
        assert rec["enable_thinking"] is False
        assert rec["task"] == "stage12_realistic_word_ocr"
        assert str(rec["answer"]).strip()
        submodes.add(rec["metadata"]["submode"])
        mem = rec["frames"][0]
        with Image.open(io.BytesIO(blobs[mem])) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32)
            chroma = float(np.abs(arr[:, :, 0] - arr[:, :, 1]).mean() + np.abs(arr[:, :, 1] - arr[:, :, 2]).mean())
            assert chroma > 10.0, f"Word OCR image too monochrome: chroma={chroma}"
    assert submodes == {"alnum_code", "scene_word", "sign_phrase"}


def test_v28_grammar_repair_covers_stage115_unseen_syntax_slips():
    from colab.final.aesthetic_v28_biomech_corpus import build_aesthetic_v28_biomech_records

    v28_recs = build_aesthetic_v28_biomech_records(seed=42, max_records=2)
    assert len(v28_recs) >= 1
    recs = build_v28_grammar_repair_records(v28_recs, seed=12, max_items=32)
    muts = {r["metadata"]["mutator"] for r in recs}
    assert "missing_bone_attach" in muts, f"Missing missing_bone_attach in {muts}"
    assert "meta_chain_field" in muts, f"Missing meta_chain_field in {muts}"
    assert "illegal_bone_statement" in muts, f"Missing illegal_bone_statement in {muts}"
