"""Unit tests for Stage-11 Visual Awakening, L_decorr, Split Early-Layer LR, AIME 8192 Token Budget,
v38.8 64-Archetype SyncCraft Corpus, Mode-7 Poster OCR, 3D MCQ, and Extra User Dataset Hook."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn

from colab.final.engine import _lr_for_group_and_name
from colab.final.frontier_stage9_corpus import (
    build_3d_spatial_mcq_pairs,
    build_synthetic_scene_ocr_pairs,
    load_extra_user_dataset_records,
)
from colab.final.model import SparkNative
from colab.final.optimizer import build_apollo_param_groups
from colab.final.v38_sculpted_corpus import build_v38_64_archetype_corpus
from scripts.eval_stage9_frontier_and_3d_suite import UNSEEN_V28_PROMPTS


def test_visual_decorrelation_loss_penalizes_dc_collapse_and_relaxes_on_diverse_tokens() -> None:
    dummy = SparkNative.__new__(SparkNative)
    nn.Module.__init__(dummy)
    dummy.policy = {
        "visual_decorr_loss_weight": 0.2,
        "visual_decorr_target_dc_ratio": 0.65,
        "visual_decorr_target_cos": 0.35,
    }
    dummy.train(True)

    # Collapsed constant DC stub + tiny noise -> high loss with finite gradient
    torch.manual_seed(42)
    base_dc = torch.ones((1, 64, 128), dtype=torch.float32)
    collapsed = (base_dc + 0.05 * torch.randn_like(base_dc)).requires_grad_(True)
    loss_bad = dummy._visual_decorrelation_loss(collapsed)
    assert loss_bad is not None
    assert float(loss_bad.item()) > 0.2
    loss_bad.backward()
    assert collapsed.grad is not None and torch.isfinite(collapsed.grad).all()

    # Zero-mean spatially diverse tokens -> near-zero penalty
    diverse = torch.randn((1, 64, 128), dtype=torch.float32)
    diverse = (diverse - diverse.mean(dim=1, keepdim=True)).requires_grad_(True)
    loss_good = dummy._visual_decorrelation_loss(diverse)
    assert loss_good is not None
    assert float(loss_good.item()) < 1e-4

    # Disabled during eval()
    dummy.eval()
    assert dummy._visual_decorrelation_loss(collapsed.detach()) is None


def test_split_early_spark_layerwise_lr_in_apollo_and_adamw() -> None:
    cfg = {
        "early_spark_layers": 12,
        "early_spark_lr": 2.2e-5,
        "backbone_lr": 7.5e-6,
        "apollo_learning_rate": 7.5e-6,
        "connector_learning_rate": 8.0e-5,
        "apollo_connector_learning_rate": 8.0e-5,
        "engram_lr": 2.8e-5,
        "vision_lr": 3.0e-6,
        "weight_decay": 0.01,
    }
    assert _lr_for_group_and_name("spark", "spark.model.layers.0.self_attn.q_proj.weight", cfg) == 2.2e-5
    assert _lr_for_group_and_name("spark", "spark.model.layers.11.mlp.down_proj.weight", cfg) == 2.2e-5
    assert _lr_for_group_and_name("spark", "spark.model.layers.12.self_attn.q_proj.weight", cfg) == 7.5e-6
    assert _lr_for_group_and_name("spark", "spark.model.embed_tokens.weight", cfg) == 7.5e-6

    class _MiniModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p_early = nn.Parameter(torch.randn(128, 128))
            self.p_deep = nn.Parameter(torch.randn(128, 128))
            self.p_conn = nn.Parameter(torch.randn(128, 128))

        def named_groups(self):
            yield ("spark", "spark.model.layers.3.self_attn.q_proj.weight", self.p_early)
            yield ("spark", "spark.model.layers.20.self_attn.q_proj.weight", self.p_deep)
            yield ("connectors", "vision_projector.w_up.weight", self.p_conn)

    groups = build_apollo_param_groups(_MiniModel(), cfg, rank=16)
    lr_by_name = {
        nm: g["lr"]
        for g in groups
        for nm in g["param_names"]
    }
    assert lr_by_name["spark.model.layers.3.self_attn.q_proj.weight"] == 2.2e-5
    assert lr_by_name["spark.model.layers.20.self_attn.q_proj.weight"] == 7.5e-6
    assert lr_by_name["vision_projector.w_up.weight"] == 8.0e-5


def test_v38_64_archetypes_compile_audit_and_zero_unseen_overlap() -> None:
    text_recs, vis_pairs, rep_recs = build_v38_64_archetype_corpus(
        seed=20261170,
        max_visual_renders=4,
        max_repairs=32,
    )
    assert len(text_recs) == 64
    assert len(vis_pairs) >= 2
    assert len(rep_recs) == 32
    unseen_ids = {u["id"].lower() for u in UNSEEN_V28_PROMPTS}
    for r in text_recs:
        assert r["id"].lower() not in unseen_ids
        assert "```blockdsl" in r["answer"]
        assert "size_step: 0.5" in r["answer"]
    for vrow, blobs in vis_pairs:
        assert vrow["frames"] and vrow["frames"][0] in blobs
        assert vrow["metadata"]["chroma"] > 10.0


def test_mode7_poster_ocr_3d_mcq_and_extra_jsonl_loader(tmp_path: Path) -> None:
    ocr_pairs = build_synthetic_scene_ocr_pairs(
        seed=99, max_items=14, include_charts_and_tables=True, include_posters=True
    )
    subtasks = {r["metadata"]["subtask"] for r, _ in ocr_pairs}
    assert "poster_banner_ocr" in subtasks
    assert "bar_chart_vqa" in subtasks
    assert "dense_table_vqa" in subtasks

    # Synthetic 3D spatial pair -> balanced (A)/(B) MCQ conversion
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 64), (40, 120, 210)).save(img_path)
    raw_png = img_path.read_bytes()
    fake_s3d = [
        (
            {
                "id": f"s3d_{i}",
                "language": "en",
                "frames": ["f.png"],
                "prompt": "Which side is the creature facing?",
                "answer": '<think>\n<point x="400" y="500" z="near" label="head"/> <dash x1="600" y1="500" x2="300" y2="500" label="facing_left"/>\n</think>\nIt is facing left.',
                "metadata": {"pixel_indispensable": True},
            },
            {"f.png": raw_png},
        )
        for i in range(6)
    ]
    mcqs = build_3d_spatial_mcq_pairs(fake_s3d, max_items=6, seed=7)
    assert len(mcqs) == 6
    lets = [r["metadata"]["mcq_answer"] for r, _ in mcqs]
    assert lets.count("A") == 3 and lets.count("B") == 3

    # Extra user JSONL loader (both text and image records)
    jpath = tmp_path / "extra.jsonl"
    rows = [
        {"id": "u1", "prompt": "Solve x+2=5.", "answer": "<think>\nx = 5 - 2 = 3.\n</think>\nFinal Answer: 3"},
        {"id": "u2", "prompt": "What color is the square?", "answer": "Blue", "image": "sample.png"},
    ]
    jpath.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
    t_recs, v_pairs = load_extra_user_dataset_records(jsonl_paths=[jpath])
    assert len(t_recs) == 1 and len(v_pairs) == 1
