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


def test_install_spark_sdpa_strict_causal_no_future_leakage() -> None:
    import sys
    import types
    from colab.final.model import install_spark_sdpa

    fake_mod_name = "_fake_spark_sdpa_test_mod"
    fake_mod = types.ModuleType(fake_mod_name)
    fake_mod.repeat_kv = lambda x, n: x.repeat_interleave(n, dim=1) if n > 1 else x
    fake_mod.eager_attention_forward = lambda *a, **kw: None
    sys.modules[fake_mod_name] = fake_mod
    try:
        class _FakeSpark(nn.Module):
            pass

        _FakeSpark.__module__ = fake_mod_name
        install_spark_sdpa(_FakeSpark())
        sdpa = fake_mod.eager_attention_forward
        assert getattr(sdpa, "_spark_native_sdpa", False) is True

        class _Attn(nn.Module):
            num_key_value_groups = 1

        attn = _Attn()
        attn.eval()
        torch.manual_seed(123)
        q = torch.randn(1, 2, 16, 32)
        k1 = torch.randn(1, 2, 16, 32)
        v1 = torch.randn(1, 2, 16, 32)
        k2, v2 = k1.clone(), v1.clone()
        k2[:, :, 15, :] += 50.0
        v2[:, :, 15, :] -= 50.0

        # Unpadded training/prefill (`mask=None`, `q_len == k_len > 1`): strictly causal, zero future leakage!
        o1, _ = sdpa(attn, q, k1, v1, attention_mask=None)
        o2, _ = sdpa(attn, q, k2, v2, attention_mask=None)
        assert float((o1[:, :, :15, :] - o2[:, :, :15, :]).abs().max().item()) == 0.0
        assert float((o1[:, :, 15:, :] - o2[:, :, 15:, :]).abs().max().item()) > 1e-2

        # Single-token decode (`mask=None`, `q_len == 1 < k_len`): attends to ALL keys including key 15!
        od1, _ = sdpa(attn, q[:, :, -1:, :], k1, v1, attention_mask=None)
        od2, _ = sdpa(attn, q[:, :, -1:, :], k2, v2, attention_mask=None)
        assert float((od1 - od2).abs().max().item()) > 1e-2
    finally:
        sys.modules.pop(fake_mod_name, None)


def test_sliding_window_decode_mask_enforced_past_512() -> None:
    import sys
    import types
    from types import SimpleNamespace
    from colab.final.loop_engram import build_spark_loop_context

    fake_mod_name = "_fake_sw_loop_ctx_mod"
    fake_mod = types.ModuleType(fake_mod_name)
    fake_mod.eager_attention_forward = SimpleNamespace(_spark_native_sdpa=True)
    sys.modules[fake_mod_name] = fake_mod
    try:
        class _FakeAttn(nn.Module):
            def forward(self, hidden_states, position_embeddings=None):
                return hidden_states, None

        class _FakeLayer(nn.Module):
            def __init__(self, lt: str) -> None:
                super().__init__()
                self.layer_type = lt
                self.self_attn = _FakeAttn()

        _FakeLayer.__module__ = fake_mod_name

        class _FakeSparkModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.has_sliding_layers = True
                self.config = SimpleNamespace(
                    sliding_window=512,
                    layer_types=["sliding_attention", "full_attention"],
                )
                self.layers = nn.ModuleList([_FakeLayer("sliding_attention"), _FakeLayer("full_attention")])

        class _FakeCache:
            def __init__(self, seen: int) -> None:
                self._seen = seen

            def get_seq_length(self, layer_idx: int = 0) -> int:
                return self._seen

        sm = _FakeSparkModel()
        emb = torch.zeros(1, 1, 16)

        # Within sliding window (`past_seen=100 -> k_len=101 <= 512`): both masks None
        m_short, _, _ = build_spark_loop_context(sm, emb, attention_mask=None, past_key_values=_FakeCache(100))
        assert m_short["full_attention"] is None
        assert m_short["sliding_attention"] is None

        # Past sliding window (`past_seen=600 -> k_len=601 > 512`): full stays None, sliding keeps exact 512 keys [89..600]
        m_long, _, _ = build_spark_loop_context(sm, emb, attention_mask=None, past_key_values=_FakeCache(600))
        assert m_long["full_attention"] is None
        sw_mask = m_long["sliding_attention"]
        assert sw_mask is not None and tuple(sw_mask.shape) == (1, 1, 1, 601)
        assert int(sw_mask.sum().item()) == 512
        assert bool(sw_mask[0, 0, 0, 88].item()) is False
        assert bool(sw_mask[0, 0, 0, 89].item()) is True
        assert bool(sw_mask[0, 0, 0, 600].item()) is True
    finally:
        sys.modules.pop(fake_mod_name, None)


def test_vision_projector_uses_fp32_adamw_and_scalar_gate_does_not_hijack_clip() -> None:
    from colab.final.optimizer import MergedGradientAccumulator

    class _MiniModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spark_w = nn.Parameter(torch.randn(128, 128))
            self.proj_w = nn.Parameter(torch.randn(128, 128))
            self.gate_attn = nn.Parameter(torch.zeros(1))

        def named_groups(self):
            yield ("spark", "spark.model.layers.3.self_attn.q_proj.weight", self.spark_w)
            yield ("connectors", "vision_projector.w_up.weight", self.proj_w)
            yield ("connectors", "visual_cross_attn.0.gate_attn", self.gate_attn)

        def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
            yield ("spark.model.layers.3.self_attn.q_proj.weight", self.spark_w)
            yield ("vision_projector.w_up.weight", self.proj_w)
            yield ("visual_cross_attn.0.gate_attn", self.gate_attn)

    cfg = {
        "backbone_lr": 7.5e-6,
        "apollo_learning_rate": 7.5e-6,
        "connector_learning_rate": 1.2e-4,
        "apollo_connector_learning_rate": 1.2e-4,
        "weight_decay": 0.01,
    }
    mm = _MiniModel()
    groups = build_apollo_param_groups(mm, cfg, rank=16)
    kind_by_name = {nm: g["kind"] for g in groups for nm in g["param_names"]}
    assert kind_by_name["spark.model.layers.3.self_attn.q_proj.weight"] == "apollo"
    assert kind_by_name["vision_projector.w_up.weight"] == "adamw"
    assert kind_by_name["visual_cross_attn.0.gate_attn"] == "adamw"

    # Huge scalar gate grad (`50.0`) must be capped cleanly before global norm reduction so `clip_coef` isn't crushed!
    acc = MergedGradientAccumulator(mm, accum_mode="fp32", attach_hooks=False)
    try:
        acc.buffers[id(mm.spark_w)].fill_(0.001)
        acc.buffers[id(mm.proj_w)].fill_(0.001)
        acc.buffers[id(mm.gate_attn)].fill_(50.0)
        opt = torch.optim.SGD(mm.parameters(), lr=1e-3)
        diag = acc.step(opt, max_grad_norm=1.2, post_scale=1.0)
        assert diag.finite and diag.applied
        assert diag.grad_norm < 2.0  # Was > 50.0 before scalar gate cap!
        assert diag.clip_coef > 0.7  # Was ~0.012 before scalar gate cap!
    finally:
        acc.close()


def test_extract_answer_body_preserves_unclosed_think_answers() -> None:
    from scripts.eval_stage9_frontier_and_3d_suite import _extract_answer_body

    th1, bd1 = _extract_answer_body("<think>\nStep 1: 252 + 25 = 277.\n\nFinal Answer: 277")
    assert th1 is not None and "Final Answer: 277" in bd1

    th2, bd2 = _extract_answer_body("<think>\nDraft guess: 100\n</think>\nFinal Answer: 277")
    assert th2 == "Draft guess: 100"
    assert bd2 == "Final Answer: 277"


def test_open_vocab_ocr_pairs_unique_vivid_and_zero_leakage() -> None:
    import io
    import numpy as np
    from colab.final.frontier_stage9_corpus import BANNED_BENCHMARK_SUBSTRINGS, build_open_vocab_ocr_pairs

    pairs = build_open_vocab_ocr_pairs(seed=20261181, max_items=24)
    assert len(pairs) == 24
    subtasks = {r["metadata"]["subtask"] for r, _ in pairs}
    assert len(subtasks) == 6
    answers = [r["answer"] for r, _ in pairs]
    assert len(set(answers)) >= 20  # High uniqueness, zero tiny-template memorization
    for r, blobs in pairs:
        low = (r["prompt"] + "\n" + r["answer"]).lower()
        assert not any(bad in low for bad in BANNED_BENCHMARK_SUBSTRINGS)
        im = Image.open(io.BytesIO(blobs[r["frames"][0]])).convert("RGB")
        arr = np.asarray(im, dtype=np.float32)
        chroma = float(np.mean(arr.max(axis=-1) - arr.min(axis=-1)))
        assert chroma > 10.0, f"Low chroma {chroma} on {r['id']}"
