"""CPU unit tests for Stage-10 Visual Projector RMS-Match Alignment Surgery,
6-Mode Synthetic OCR/Chart/Table Generation, and `v28` Block-DSL v4.1 Grammar Repair Corpus.
"""
from __future__ import annotations

import io
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch

from colab.final.frontier_stage9_corpus import (
    build_synthetic_scene_ocr_pairs,
    build_v28_grammar_repair_records,
    convert_catalogue_row_to_direct_mode,
)
from colab.final.model import (
    RadioMultiLayerSwiGLUProjector,
    _write_safetensors_dict,
    load_spark_native_safetensors,
)


def test_radio_projector_output_rms_match_clamps_50x_explosion_to_spark_text_scale():
    torch.manual_seed(42)
    # 1. Legacy Stage-8/9 behavior (`output_rms_match=False`) remains bit-for-bit intact
    legacy = RadioMultiLayerSwiGLUProjector(vis_dim=32, hidden_size=64, num_taps=2, output_rms_match=False)
    assert not hasattr(legacy, "out_norm")
    assert not hasattr(legacy, "out_scale")
    assert not hasattr(legacy, "meta_scale")

    # 2. Stage-10 `output_rms_match=True` clamps even 100x huge intermediate SwiGLU outputs down to `out_scale_init=0.035`
    rms_proj = RadioMultiLayerSwiGLUProjector(
        vis_dim=32,
        hidden_size=64,
        num_taps=2,
        output_rms_match=True,
        out_scale_init=0.035,
        meta_scale_init=0.008,
    )
    with torch.no_grad():
        # Deliberately inflate SwiGLU linear weights 25x to simulate Stage-8/9 `53.06` RMS explosion
        rms_proj.w_gate.weight.mul_(25.0)
        rms_proj.w_up.weight.mul_(25.0)
        rms_proj.w_down.weight.mul_(10.0)

    x = torch.randn(1, 2, 16, 32) * 5.0
    main_toks, early_toks = rms_proj(x, 4, 4)
    assert main_toks.shape == (1, 4, 64)
    assert early_toks.shape == (1, 4, 64)
    main_rms = float(main_toks.float().pow(2).mean().sqrt().item())
    early_rms = float(early_toks.float().pow(2).mean().sqrt().item())
    assert 0.025 <= main_rms <= 0.045, f"Expected main_rms near 0.035, got {main_rms}"
    assert 0.025 <= early_rms <= 0.045, f"Expected early_rms near 0.035, got {early_rms}"


def test_load_spark_native_safetensors_warmstarts_stage9_to_stage10_rms_match():
    torch.manual_seed(7)

    class _MiniModel(torch.nn.Module):
        def __init__(self, rms_match: bool):
            super().__init__()
            self.vision_projector = RadioMultiLayerSwiGLUProjector(
                vis_dim=16, hidden_size=32, num_taps=2, output_rms_match=rms_match
            )
            self.other = torch.nn.Linear(32, 32)

    m_s9 = _MiniModel(rms_match=False)
    m_s10 = _MiniModel(rms_match=True)

    with tempfile.TemporaryDirectory() as td:
        sf = Path(td) / "stage9.safetensors"
        _write_safetensors_dict(dict(m_s9.state_dict()), sf)

        # Strict load without allow_missing_projector_out_norm must raise
        raised = False
        try:
            load_spark_native_safetensors(m_s10, sf, allow_missing_projector_out_norm=False)
        except RuntimeError:
            raised = True
        assert raised

        # Warm-start load with allow_missing_projector_out_norm=True succeeds and preserves weights + init scales!
        load_spark_native_safetensors(m_s10, sf, allow_missing_projector_out_norm=True)
        assert torch.allclose(m_s10.vision_projector.w_gate.weight, m_s9.vision_projector.w_gate.weight)
        assert torch.allclose(m_s10.other.weight, m_s9.other.weight)
        assert abs(float(m_s10.vision_projector.out_scale.item()) - 0.035) < 1e-6
        assert abs(float(m_s10.vision_projector.meta_scale.item()) - 0.008) < 1e-6

        # Stage-10.5 clean projector re-init: preserves `other` bit-exact while keeping fresh zero-bias `vision_projector`!
        m_s105 = _MiniModel(rms_match=True)
        fresh_w_gate = m_s105.vision_projector.w_gate.weight.detach().clone()
        load_spark_native_safetensors(m_s105, sf, allow_missing_projector_reinit=True)
        assert torch.allclose(m_s105.other.weight, m_s9.other.weight)
        assert torch.allclose(m_s105.vision_projector.w_gate.weight, fresh_w_gate)
        assert not torch.allclose(m_s105.vision_projector.w_gate.weight, m_s9.vision_projector.w_gate.weight)
        assert float(m_s105.vision_projector.w_down.bias.abs().max().item()) == 0.0
        assert float(m_s105.vision_projector.early_mid_proj.bias.abs().max().item()) == 0.0


def test_stage10_six_mode_synthetic_ocr_and_v28_grammar_repair_corpus():
    # 1. 6-mode synthetic OCR including Bar-Chart VQA & Dense Table VQA (all full-color RGB chroma > 10!)
    pairs = build_synthetic_scene_ocr_pairs(seed=20261155, max_items=24, include_charts_and_tables=True)
    assert len(pairs) == 24
    subtasks = set()
    for rec, blobs in pairs:
        assert rec["enable_thinking"] is False
        assert rec["answer"] and "<think>" not in rec["answer"]
        subtasks.add(rec["metadata"]["subtask"])
        im = Image.open(io.BytesIO(blobs[rec["frames"][0]])).convert("RGB")
        arr = np.asarray(im, dtype=np.float32)
        chroma = float(np.mean(np.abs(arr[..., 0] - arr[..., 1]) + np.abs(arr[..., 1] - arr[..., 2])))
        assert chroma > 10.0, f"Expected full-color RGB card, got chroma={chroma:.2f}"
    assert subtasks == {
        "scene_text_recognition",
        "kie_receipt",
        "doc_vqa_notice",
        "formula_ocr",
        "bar_chart_vqa",
        "dense_table_vqa",
    }

    # 2. v28 Block-DSL v4.1 Grammar Repair Corpus produces real SyncCraft parser errors + compilable targets!
    from colab.final.synccraft_bridge import _extract_dsl_candidate, ensure_synccraft_importable

    ensure_synccraft_importable()
    from synccraft.dsl.parser import parse_program

    clean_dsl = """@meta { archetype:entity, target_profile:bedrock_entity, texel_density:16, units_per_block:16, dsl_version:"4.1", size_step:0.5, audit_profile:sculpted }
@palette {
  hide: ramp(base_lch=[0.46,0.12,42], steps=5, hue_shift=[-16,18])
}
@rig {
  root bone torso {
    pivot_local:[0,9.5,0]
    box chest { size:[5,4.5,7], material:hide, prism_group:"ch" }
    bone leg_l attach(parent_box=chest, anchor=[-1,-0.65,0], align=[1,1,0]) mirror_x(leg_r) {
      joint_limits:[[-60,60],[-25,25],[-25,25]]
      box femur { size:[1.5,4,2], material:hide }
    }
  }
}
"""
    mock_v28 = [{"id": "sample_v28_rig", "answer": f"<think>\nBuild clean rig.\n</think>\n```blockdsl\n{clean_dsl}\n```"}]
    repairs = build_v28_grammar_repair_records(mock_v28, seed=20261150, max_items=24)
    assert len(repairs) >= 9
    mutations = {r["metadata"].get("mutation") for r in repairs if r["metadata"].get("mutation")}
    assert {"texel_size", "root_keyword", "box_axis", "bone_and", "palette_rgb", "meta_author", "size_step_025"}.issubset(mutations)
    for r in repairs:
        if r["metadata"].get("mutation"):
            assert r["metadata"]["parser_error"]
            repaired_dsl = _extract_dsl_candidate(r["answer"])
            assert repaired_dsl and "@rig" in repaired_dsl
            parse_program(repaired_dsl)

    # 3. Direct-mode conversion preserves multi-line `document_ocr` transcriptions instead of cutting to line 1
    doc_row = {
        "task": "document_ocr",
        "prompt": "Read all visible text in reading order.",
        "answer": "Line One Header\n" + ("Line Two Body Content " * 14),
    }
    doc_direct = convert_catalogue_row_to_direct_mode(doc_row)
    assert doc_direct["enable_thinking"] is False
    assert "Line One Header\nLine Two Body Content" in doc_direct["answer"]
