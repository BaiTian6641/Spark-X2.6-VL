"""CPU unit tests for Stage-13 PRISM-Omni corpus normalization, counterfactual Color x Shape scenes,
deterministic RLVR verifiers, Visual-Causality-Gated GRPO (`VC-GRPO`) rewards, and user-feedback JSONL."""
from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image

from colab.final.multimodal_rlvr import (
    compute_vc_grpo_rewards,
    load_user_feedback_jsonl,
    verify_rlvr_prediction,
)
from colab.final.stage13_prism_rlvr_corpus import (
    build_counterfactual_color_shape_spatial_pairs,
    normalize_prism_primitives,
)
from scripts.collect_user_vl_feedback import append_feedback_record


def test_normalize_prism_primitives_unifies_spatial_tags():
    raw = (
        "Locate <|ref|>red mug<|/ref|><|point|>[[240, 512], [700, 120]]<|/point|> "
        "inside <|ref|>table<|/ref|><|box|>[[10, 20, 800, 900]]<|/box|> "
        "and <|ref|>chair<|/ref|><|box3d|>[[0.45, -0.20, 1.85, 0.5, 0.5, 0.9]]<|/box3d|> "
        "along <|ref|>route<|/ref|><|dashed_line|>[[100, 100], [500, 600]]<|/dashed_line|>."
    )
    out = normalize_prism_primitives(raw)
    assert '<point x="240" y="512" label="red mug"/>' in out
    assert '<point x="700" y="120" label="red mug"/>' in out
    assert '<box x1="10" y1="20" x2="800" y2="900" label="table"/>' in out
    assert '3D center=[0.45, -0.20, z=1.85m], z="near"' in out
    assert '<dash x1="100" y1="100" x2="500" y2="600" label="route"/>' in out
    assert "<|ref|>" not in out and "<|point|>" not in out


def test_counterfactual_color_shape_spatial_pairs_all_modes_and_balance():
    pairs = build_counterfactual_color_shape_spatial_pairs(max_items=64, seed=13)
    assert len(pairs) == 64
    tasks = set()
    yn_counts = {"Yes": 0, "No": 0}
    for rec, blobs in pairs:
        tasks.add(rec["task"])
        assert rec["enable_thinking"] is False
        assert rec["metadata"]["pixel_indispensable"] is True
        assert rec["metadata"]["verifier"]["gold"]
        mem = rec["frames"][0]
        with Image.open(io.BytesIO(blobs[mem])) as im:
            assert im.size == (512, 384)
            assert im.mode == "RGB"
        if rec["task"] == "stage13_cf_colorshape_m1":
            yn_counts[rec["answer"]] += 1
    assert tasks == {f"stage13_cf_colorshape_m{i}" for i in range(4)}
    assert yn_counts["Yes"] == yn_counts["No"] == 8


def test_verify_rlvr_prediction_covers_all_deterministic_verifiers():
    assert verify_rlvr_prediction("Yes.", {"kind": "yesno_bool", "gold": "yes"})["reward"] == 1.0
    assert verify_rlvr_prediction("False", {"kind": "yesno_bool", "gold": "no"})["reward"] == 1.0
    assert verify_rlvr_prediction("Yes", {"kind": "yesno_bool", "gold": "no"})["reward"] == 0.0

    assert verify_rlvr_prediction("Answer: C", {"kind": "mcq_letter", "gold": "C"})["reward"] == 1.0
    assert verify_rlvr_prediction("Option (B)", {"kind": "mcq_letter", "gold": "B"})["reward"] == 1.0
    assert verify_rlvr_prediction("Answer: A", {"kind": "mcq_letter", "gold": "D"})["reward"] == 0.0

    assert verify_rlvr_prediction("The total is 42.0.", {"kind": "numeric_exact", "gold": "42"})["reward"] == 1.0
    assert verify_rlvr_prediction("15", {"kind": "dapo_math", "gold": "15"})["reward"] == 1.0
    assert verify_rlvr_prediction("19", {"kind": "numeric_exact", "gold": "15"})["reward"] == 0.0

    ocr_exact = verify_rlvr_prediction("STATION 42", {"kind": "ocr_cer", "gold": "STATION 42"})
    ocr_partial = verify_rlvr_prediction("STATIOM 42", {"kind": "ocr_cer", "gold": "STATION 42"})
    assert ocr_exact["reward"] == 1.0 and 0.6 < ocr_partial["reward"] < 0.9

    cs_ok = verify_rlvr_prediction("blue square", {"kind": "color_shape", "gold": "blue square"})
    cs_half = verify_rlvr_prediction("red square", {"kind": "color_shape", "gold": "blue square"})
    assert cs_ok["reward"] == 1.0 and 0.15 <= cs_half["reward"] <= 0.30

    # Unclosed `<think>` must receive collapse penalty!
    bad = verify_rlvr_prediction("<think>\nstill thinking yes", {"kind": "yesno_bool", "gold": "yes"}, expect_thinking=True)
    assert bad["healthy"] is False and bad["penalty"] < 0.0


def test_vc_grpo_rewards_strictly_prefer_visually_grounded_correct_rollout():
    ver_res = [
        {"reward": 1.0, "exact": True, "penalty": 0.0, "healthy": True},  # Visually grounded correct!
        {"reward": 1.0, "exact": True, "penalty": 0.0, "healthy": True},  # Blind guess correct!
        {"reward": 0.0, "exact": False, "penalty": 0.0, "healthy": True},  # Wrong
        {"reward": 0.0, "exact": False, "penalty": -0.35, "healthy": False},  # Collapsed wrong
    ]
    delta_logps = [0.85, -0.05, 0.10, -0.20]
    raw_r, advs = compute_vc_grpo_rewards(ver_res, delta_logps)
    assert raw_r[0] > raw_r[1] > raw_r[2] > raw_r[3], f"Unexpected VC-GRPO reward ordering: {raw_r}"
    assert advs[0] > advs[1] and advs[0] > 0.0 and advs[3] < 0.0


def test_user_feedback_jsonl_roundtrip_and_cli_help(tmp_path: Path):
    img_p = tmp_path / "sample.png"
    Image.new("RGB", (64, 64), (20, 120, 220)).save(img_p)
    fb_p = tmp_path / "feedback.jsonl"
    append_feedback_record(
        fb_p,
        prompt="What color is this square?",
        chosen="blue",
        rejected="red",
        image_path=str(img_p),
        rating=2.0,
        note="fixed color prior",
    )
    loaded = load_user_feedback_jsonl(fb_p)
    assert len(loaded) == 1
    assert loaded[0]["chosen"] == "blue"
    assert loaded[0]["rejected"] == "red"
    assert loaded[0]["image_bytes"] is not None and len(loaded[0]["image_bytes"]) > 50

    for script in ("scripts/run_stage13_24h_multimodal_rlvr.py", "scripts/collect_user_vl_feedback.py"):
        res = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True, check=False)
        assert res.returncode == 0, f"{script} --help failed: {res.stderr}"
