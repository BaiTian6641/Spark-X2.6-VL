"""CPU unit tests for Stage-9 Frontier (`TB 4.1`, `HLE-Diamond`, `GPQA-Diamond`, `3DSRBench`, `MMMU-Pro`)
+ `Blockbench_sculpted_style-v28` / SyncCraft `ba969f3` Corpus + Dual-Track Verified RSI-3 & Promotion Gate.
"""
from __future__ import annotations

import json
import numpy as np
from PIL import Image

from colab.final.frontier_stage9_corpus import (
    HLE_CANARY,
    _format_bounded_agent_trajectory,
    _is_contaminated,
    _split_reasoning_and_body,
)
from scripts.eval_stage9_frontier_and_3d_suite import (
    _extract_choice_letter,
    _is_stage5_hle_train_id,
    _normalize_exact_answer,
    _validate_terminal_blocks,
)
from scripts.run_stage9_frontier_v28_and_rsi3 import (
    audit_stage9_records_strict,
    evaluate_rsi3_promotion_gate,
    score_rsi3_terminal_or_stem_candidate,
)


def test_stage9_agent_trajectory_masks_tool_responses_and_rejects_contamination():
    conv = [
        {"role": "user", "content": "Fix the failing unit test in /Users/alice/repo/tests/test_math.py"},
        {"role": "assistant", "content": "<think>\nInspect the test file first.\n</think>\n```bash\ncat tests/test_math.py\n```"},
        {"role": "user", "content": "AssertionError: add(2, 3) returned 6"},
        {
            "role": "assistant",
            "content": "```bash\nsed -i 's/a * b/a + b/' math_ops.py && pytest -q\n```",
        },
        {"role": "user", "content": "1 passed in 0.02s"},
    ]
    res = _format_bounded_agent_trajectory(conv)
    assert res is not None
    prompt, ans = res
    assert "/Users/alice" not in prompt
    assert "<tool_response>" in ans and "</tool_response>" in ans
    assert not ans.rstrip().endswith("</tool_response>")
    assert ans.count("<think>") == 1 and ans.count("</think>") == 1

    # Zero-leakage contamination guard
    assert _is_contaminated(f"Some question containing {HLE_CANARY}")
    assert _is_contaminated("normal task", excluded_task_ids={"atrx-vep-crispr"}, task_id="atrx-vep-crispr")
    assert _is_contaminated("Evaluate on gpqa_diamond split")
    assert not _is_contaminated("Write a Python script to parse CSV logs cleanly.", excluded_task_ids={"atrx-vep-crispr"}, task_id="other-task")

    th, bd = _split_reasoning_and_body("<|begin_of_thought|>Step 1: integrate.<|end_of_thought|><|begin_of_solution|>Final Answer: 42<|end_of_solution|>")
    assert th == "Step 1: integrate."
    assert bd == "Final Answer: 42"


def test_stage9_eval_helpers_tb41_hle_mmmupro():
    assert _extract_choice_letter("Therefore, Answer: (H)", "ABCDEFGHIJ") == "H"
    assert _extract_choice_letter("Selected Option: D", "ABCD") == "D"
    assert _extract_choice_letter(r"Final Answer: \boxed{J}", "ABCDEFGHIJ") == "J"
    assert _normalize_exact_answer(r"Final Answer: \boxed{42.5}") == "42.5"

    ok_py, chars_py, _ = _validate_terminal_blocks("```python\nx = [i * 2 for i in range(10)]\nprint(sum(x))\n```", "t1")
    assert ok_py and chars_py >= 30
    bad_py, _, reason_py = _validate_terminal_blocks("```python\ndef broken(:\n  pass\n```", "t2")
    assert not bad_py and reason_py == "python_syntax_error"
    ok_sh, _, _ = _validate_terminal_blocks("```bash\nset -euo pipefail\necho hello\n```", "t3")
    assert ok_sh

    # Verify Stage-5 HLE holdout determinism (~20% untouched)
    untouched = sum(1 for i in range(200) if not _is_stage5_hle_train_id(f"hle_item_{i}"))
    assert 25 <= untouched <= 55


def test_stage9_strict_audit_rsi3_oracle_and_promotion_gate():
    color_im = Image.fromarray(np.full((128, 128, 3), [60, 140, 210], dtype=np.uint8), mode="RGB")
    bw_im = Image.fromarray(np.full((128, 128, 3), [128, 128, 128], dtype=np.uint8), mode="RGB")

    clean_front = [
        {
            "id": "f1",
            "prompt": "Solve for x when 2x + 4 = 10.",
            "answer": "<think>\nSubtract 4 from both sides to get 2x = 6, then divide by 2.\n</think>\nFinal Answer: \\boxed{3}",
            "enable_thinking": True,
            "task": "stage9_openthoughts3_stem_reasoning",
            "target": {"kind": "text", "text": "Final Answer: \\boxed{3}"},
            "metadata": {"task_slug": "clean_math"},
        }
    ]
    clean_v28 = [
        {
            "id": "v1",
            "prompt": "Inspect this shaded 3D render.",
            "answer": "<think>\nChecking <box x1=\"100\" y1=\"100\" x2=\"500\" y2=\"500\" label=\"head\"/>.\n</think>\nPole B locked.",
            "enable_thinking": True,
            "images": [color_im],
            "metadata": {},
        }
    ]
    rep_ok = audit_stage9_records_strict(clean_front, clean_v28, [], excluded_tb41_ids={"tb41_secret_task"})
    assert rep_ok["passed"], rep_ok

    # Catch TB 4.1 leak & B&W cutout on v28
    leaked_front = [{**clean_front[0], "id": "f_leak", "metadata": {"task_slug": "tb41_secret_task"}}]
    bw_v28 = [{**clean_v28[0], "id": "v_bw", "images": [bw_im]}]
    rep_bad = audit_stage9_records_strict(leaked_front, bw_v28, [], excluded_tb41_ids={"tb41_secret_task"})
    assert not rep_bad["passed"]
    checks = {f["check"].split(":")[0] for f in rep_bad["failures"]}
    assert "tb41_leak" in checks and "bw_cutout_detected" in checks

    # RSI-3 Track 2 Terminal-Pivot & STEM oracles
    piv_rec = {
        "task": "stage9_nemotron_terminal_pivot",
        "metadata": {"expected_json": json.dumps({"analysis": "check logs", "plan": "run pytest", "commands": ["pytest -q"]})},
    }
    good_piv = "<think>\nInspect test output first.\n</think>\n```json\n" + json.dumps(
        {"analysis": "Inspect failing service logs", "plan": "Run pytest suite", "commands": ["pytest -q"]}
    ) + "\n```"
    assert score_rsi3_terminal_or_stem_candidate(piv_rec, good_piv)["verified"]
    assert score_rsi3_terminal_or_stem_candidate(clean_front[0], clean_front[0]["answer"])["verified"]
    assert not score_rsi3_terminal_or_stem_candidate(
        clean_front[0], "<think>\nGuessing.\n</think>\nFinal Answer: \\boxed{99}"
    )["verified"]

    # 5-Axis Anti-Collapse Promotion Gate (accepts gain, rejects regression)
    parent = {"mean_real_loss": 0.55, "mean_media_advantage": 0.08, "v25_locate_probe": {"predicted": "Clean concise answer."}}
    good_cand = {"mean_real_loss": 0.54, "mean_media_advantage": 0.09, "v25_locate_probe": {"predicted": "Clean concise answer."}}
    bad_cand = {"mean_real_loss": 0.62, "mean_media_advantage": 0.02, "v25_locate_probe": {"predicted": "Clean concise answer."}}
    assert evaluate_rsi3_promotion_gate(parent, good_cand)["accepted"]
    assert not evaluate_rsi3_promotion_gate(parent, bad_cand)["accepted"]


def test_aesthetic_v28_biomech_corpus_compiles_and_audits_clean():
    from colab.final.aesthetic_v28_biomech_corpus import (
        AESTHETIC_V28_TASKS,
        build_aesthetic_v28_biomech_records,
        build_all_v28_verified_masterworks,
    )

    mws = build_all_v28_verified_masterworks()
    assert len(mws) >= 4
    domains = {m.get("domain") for m in mws}
    assert "D4" in domains and "D5" in domains

    recs = build_aesthetic_v28_biomech_records(seed=20261128, max_records=48)
    assert len(recs) >= 24
    tasks_seen = {r["task"] for r in recs}
    assert len(tasks_seen & set(AESTHETIC_V28_TASKS)) >= 4
    assert any(r.get("language") == "zh" for r in recs)

    rep = audit_stage9_records_strict([], recs, [])
    assert rep["passed"], rep["failures"]
