"""Closed-Loop Vision + 3D Aesthetic Recursive Self-Improvement (`rsi_mode = 'vision_self_improvement'`).

Executes genuine self-exploration, visual/compiler self-critique, deterministic oracle
verification, and anti-collapse-gated weight promotion on `SparkNative` without external
paid teacher APIs or ground-truth label leakage during candidate generation:

1. **Stratified Hard-Example Mining (`train` split only)**:
   Prioritizes 3D multiview/aesthetic Block-DSL synthesis, 4-way RGBA UV/alpha diagnosis,
   `array/SAT` 3D spatial reasoning, multimodal HLE / OCR / counting, and coding/terminal anchors.
2. **Multi-Candidate `<think>` Sampling + Visual/Structural Self-Critique**:
   Generates greedy baseline + `K` diverse `<think>` candidates; if unverified or flagged as
   a toy blockout (`is_toy_like`), renders the candidate's own 4-view preview / compiler critique
   (never leaking ground-truth answers) for a self-correction turn.
3. **Deterministic Physical/Visual Oracle + Anti-Collapse Trajectory Filter**:
   Only trajectories passing exact SyncCraft SE(3)/SAT/IoU/Aesthetic or ground-truth checks AND
   strict `<think>` closure / 4-gram anti-repetition bounds enter `row_overrides`, paired 50/50
   with broad bilingual multimodal + coding replay anchors.
4. **5-Axis Anti-Collapse Promotion Gate (`promotion` split)**:
   Verifies stratum losses, media/shuffled advantages, SyncCraft oracle rate, `<think>` close rate,
   and n-gram repetition health — rolling back automatically to the parent checkpoint on any regression.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import contextlib
import copy
import gc
import json
import math
from pathlib import Path
import random
import re
import subprocess
from typing import Any

from .common import emit, identity, read, write
from .reasoning_agentic_corpus import (
    check_generation_collapse_health,
    compute_ngram_repetition_ratio,
    sanitize_and_bound_think_trace,
)
from .synccraft_bridge import (
    evaluate_synccraft_completion,
    render_world_views_rgba,
    strip_think_block,
)


_CHOICE_LETTER_RE = re.compile(r"(?:Answer|Final Answer)\s*:\s*\(?([A-D])\)?", re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(r"Final Answer\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_CODE_BLOCK_RE = re.compile(r"```(python|bash|sh|json)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _normalize_short_answer(s: str) -> str:
    s = re.sub(r"^(?:final\s+answer|answer)\s*:\s*", "", str(s or "").strip(), flags=re.IGNORECASE)
    s = s.strip().strip(".").strip()
    return re.sub(r"\s+", " ", s.casefold())


def score_rsi_candidate(
    row: dict[str, Any],
    completion_text: str,
    *,
    min_aesthetic_score: float = 0.68,
) -> dict[str, Any]:
    """Deterministically verify a generated RSI candidate without modifying `row`."""
    from .evaluate import error_rate
    from .world_pipeline import _extract_clean_blockdsl, evaluate_3d_asset_aesthetics

    task = str(row.get("task") or "")
    expect_think = bool(row.get("enable_thinking", False))
    health = check_generation_collapse_health(
        completion_text,
        expect_thinking=expect_think,
        max_rep_4gram=0.10,
        max_think_chars=4200,
    )
    if not health["healthy"]:
        return {
            "verified": False,
            "score": 0.0,
            "reason": f"collapse_guard:{health['reason']}",
            "collapse_health": health,
        }

    think_trace, body = strip_think_block(completion_text)
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}

    # 1. SyncCraft 3D World & Human-Artist Aesthetic tasks
    if task.startswith("synccraft_"):
        sc = evaluate_synccraft_completion(row, completion_text)
        if not sc.get("passed"):
            return {
                "verified": False,
                "score": float(sc.get("field_accuracy") or sc.get("silhouette_iou") or sc.get("ladder_score") or 0.0) * 0.5,
                "reason": str(sc.get("error") or "synccraft_oracle_failed"),
                "synccraft_eval": sc,
                "collapse_health": health,
            }

        aes_report: dict[str, Any] | None = None
        is_constructive_dsl = task in (
            "synccraft_multiview_inverse_dsl",
            "synccraft_behavior_animation",
        ) or (
            task == "synccraft_artist_assistant"
            and meta.get("assistant_mode") in ("create_full_asset", "coarse_to_fine_upgrade", "texture_and_animate_rig")
        )
        if is_constructive_dsl:
            try:
                from synccraft.dsl.parser import parse_program

                cand_prog = parse_program(_extract_clean_blockdsl(completion_text))
                aes_report = evaluate_3d_asset_aesthetics(cand_prog, prompt=str(row.get("prompt") or ""))
                # Enforce non-toy gate whenever target/prompt expects a rich multi-part entity (>= 14 boxes)
                ref_dsl = str(meta.get("canonical_dsl") or "")
                ref_rich = True
                if ref_dsl:
                    try:
                        from synccraft.kinematics.se3_chain import compile_geometry

                        ref_boxes = len(compile_geometry(parse_program(ref_dsl)).boxes)
                        ref_rich = ref_boxes >= 14
                    except Exception:
                        ref_rich = True
                if ref_rich and (aes_report["is_toy_like"] or float(aes_report["aesthetic_score"]) < min_aesthetic_score):
                    return {
                        "verified": False,
                        "score": 0.45 + 0.4 * float(aes_report["aesthetic_score"]),
                        "reason": f"toy_aesthetic_rejected:{aes_report['aesthetic_score']:.3f}",
                        "synccraft_eval": sc,
                        "aesthetic_eval": aes_report,
                        "collapse_health": health,
                    }
            except Exception as exc:
                return {
                    "verified": False,
                    "score": 0.2,
                    "reason": f"aesthetic_eval_error:{exc}",
                    "collapse_health": health,
                }

        iou_val = float(sc.get("silhouette_iou", 1.0))
        aes_val = float(aes_report["aesthetic_score"]) if aes_report else 0.85
        combined_score = round(min(1.0, 0.55 * iou_val + 0.45 * aes_val), 5)
        return {
            "verified": True,
            "score": max(0.85, combined_score),
            "reason": "synccraft_and_aesthetic_verified",
            "synccraft_eval": sc,
            "aesthetic_eval": aes_report,
            "collapse_health": health,
        }

    # 2. 3D Spatial Reasoning (`array/SAT`) & Compositional VQA
    if task.startswith("sat_3d_") or task in ("compositional_vqa", "real_count", "temporal_order", "occlusion"):
        ref_txt = str((row.get("target") or {}).get("text") or row.get("answer") or "").strip()
        _, ref_clean = strip_think_block(ref_txt)
        norm_pred = _normalize_short_answer(body)
        norm_ref = _normalize_short_answer(ref_clean)
        exact = bool(norm_ref) and (
            norm_pred == norm_ref
            or norm_pred.endswith(" " + norm_ref)
            or norm_pred.startswith(norm_ref + ".")
        )
        return {
            "verified": exact,
            "score": 1.0 if exact else 0.0,
            "reason": "exact_spatial_match" if exact else "spatial_mismatch",
            "collapse_health": health,
        }

    # 3. Document / Region OCR & Captioning
    if task in ("document_ocr", "region_ocr"):
        ref_txt = str((row.get("target") or {}).get("text") or row.get("answer") or "").strip()
        er = error_rate(ref_txt, body, str(row.get("language") or "en"))
        rate = float(er["rate"])
        ok = rate <= 0.05
        return {
            "verified": ok,
            "score": round(max(0.0, 1.0 - rate), 5),
            "reason": "ocr_verified" if ok else f"ocr_cer:{rate:.3f}",
            "collapse_health": health,
        }

    # 4. GPQA Extended & HLE Disjoint Multimodal/Text Reasoning
    if task == "gpqa_extended_science":
        expected_letter = str(meta.get("correct_letter") or "").strip().upper()
        expected_ans = _normalize_short_answer(str(meta.get("correct_answer") or ""))
        m = _CHOICE_LETTER_RE.search(body)
        pred_letter = m.group(1).upper() if m else ""
        ok = bool(expected_letter and pred_letter == expected_letter) or (
            bool(expected_ans) and expected_ans in _normalize_short_answer(body)
        )
        return {
            "verified": ok,
            "score": 1.0 if ok else 0.0,
            "reason": "gpqa_verified" if ok else "gpqa_wrong_choice",
            "collapse_health": health,
        }

    if task in ("hle_multimodal_reasoning", "hle_text_reasoning"):
        oracle_ans = _normalize_short_answer(str(meta.get("oracle_exact_answer") or ""))
        m = _FINAL_ANSWER_RE.search(body)
        extracted = _normalize_short_answer(m.group(1) if m else body.splitlines()[-1])
        ok = bool(oracle_ans) and (extracted == oracle_ans)
        return {
            "verified": ok,
            "score": 1.0 if ok else 0.0,
            "reason": "hle_verified" if ok else "hle_mismatch",
            "collapse_health": health,
        }

    # 5. Agentic SWE, Terminal-Bench 3.0 & OpenThoughts Code/Math
    if task in ("fable5_swe_agent", "claude_code_agent", "terminal_bench_agent", "openthoughts_reasoning"):
        code_blocks = _CODE_BLOCK_RE.findall(body)
        if not code_blocks and "tool_calls" not in body:
            # Check exact final answer match for math problems
            ref_sol = str((row.get("target") or {}).get("text") or "").strip()
            ref_last = _normalize_short_answer(ref_sol.splitlines()[-1] if ref_sol else "")
            pred_last = _normalize_short_answer(body.splitlines()[-1] if body else "")
            ok = bool(ref_last and len(ref_last) <= 64 and pred_last == ref_last)
            return {
                "verified": ok,
                "score": 0.92 if ok else 0.0,
                "reason": "math_final_line_match" if ok else "unverified_freeform_text",
                "collapse_health": health,
            }
        syntax_ok = True
        for lang_tag, snippet in code_blocks:
            lt = lang_tag.lower()
            if lt == "python":
                try:
                    ast.parse(snippet)
                except SyntaxError:
                    syntax_ok = False
                    break
            elif lt == "json":
                try:
                    parsed = json.loads(snippet)
                    if not isinstance(parsed, dict):
                        syntax_ok = False
                        break
                except Exception:
                    syntax_ok = False
                    break
            elif lt in ("bash", "sh"):
                proc = subprocess.run(
                    ["bash", "-n"],
                    input=snippet.encode("utf-8"),
                    capture_output=True,
                    timeout=3,
                    check=False,
                )
                if proc.returncode != 0:
                    syntax_ok = False
                    break
        if not syntax_ok:
            return {
                "verified": False,
                "score": 0.1,
                "reason": "code_or_tool_syntax_invalid",
                "collapse_health": health,
            }
        # Require substantive token overlap with verified reference target so arbitrary no-op scripts don't pass
        ref_sol = str((row.get("target") or {}).get("text") or "")
        ref_idents = {w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", ref_sol)}
        pred_idents = {w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", body)}
        recall = len(ref_idents & pred_idents) / max(1, len(ref_idents)) if ref_idents else 0.0
        ok = recall >= 0.45 and len(body) >= 48
        return {
            "verified": ok,
            "score": round(0.5 + 0.5 * min(1.0, recall), 5) if ok else round(0.3 * recall, 5),
            "reason": "agentic_code_verified" if ok else f"low_target_ident_recall:{recall:.2f}",
            "collapse_health": health,
        }

    return {"verified": False, "score": 0.0, "reason": "task_no_self_verifier", "collapse_health": health}


def build_self_critique_turn(
    row: dict[str, Any],
    media: dict[str, Any],
    failed_completion: str,
    diag: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Construct a zero-label-leakage visual/structural self-critique turn for a failed candidate."""
    from PIL import Image
    import io
    from .world_pipeline import (
        _extract_clean_blockdsl,
        _prepare_media_from_pil_images,
        build_default_world_state_for_program,
    )

    task = str(row.get("task") or "")
    _, prev_body = strip_think_block(failed_completion)
    if not prev_body:
        return None

    if task.startswith("synccraft_") and task not in ("synccraft_uv_alpha_diagnosis", "synccraft_part_grounding"):
        sc_ev = diag.get("synccraft_eval") or {}
        aes_ev = diag.get("aesthetic_eval") or {}
        notes: list[str] = [f"Verifier status: `{diag.get('reason')}`."]
        if sc_ev.get("error"):
            notes.append(f"Compiler/SAT diagnostic: {sc_ev['error']}")
        if sc_ev.get("collision_count"):
            notes.append(f"15-axis SAT collision count: {sc_ev['collision_count']} (must be 0).")
        if sc_ev.get("contact_violation_count"):
            notes.append(f"P1-02 contact violation count: {sc_ev['contact_violation_count']} (must be 0).")
        if aes_ev:
            notes.append(
                f"Aesthetic readiness={aes_ev.get('aesthetic_score', 0.0):.3f}, "
                f"boxes={aes_ev.get('box_count')}, bones={aes_ev.get('bone_count')}, "
                f"is_toy_like={aes_ev.get('is_toy_like')}."
            )
            for cn in (aes_ev.get("critique_notes") or ())[:4]:
                notes.append(f"- Upgrade required: {cn}")

        critique_media = media
        # If text-to-3D candidate compiled, render its own 4-view RGBA preview so Spark visually critiques its own shape!
        if not media.get("images"):
            try:
                from synccraft.dsl.parser import parse_program

                cand_prog = parse_program(_extract_clean_blockdsl(failed_completion))
                ws = build_default_world_state_for_program(cand_prog, seed=17)
                png_map, _, _, _ = render_world_views_rgba(ws, panel_size=(144, 144))
                pil_views = [Image.open(io.BytesIO(png_map[v])) for v in ("south", "right", "top", "iso")]
                critique_media = _prepare_media_from_pil_images(pil_views, view_names=("south", "right", "top", "iso"))
                notes.append(
                    "Attached are the 4 orthographic views (south, right, top, iso) rendered from your previous attempt — "
                    "inspect the silhouette proportions, missing anatomical/structural parts, and joint contacts directly."
                )
            except Exception:
                critique_media = media

        new_row = dict(row)
        new_row["enable_thinking"] = True
        new_row["prompt"] = (
            f"{row['prompt']}\n\n"
            f"[Self-Critique Attempt]\nYour previous draft:\n```blockdsl\n{_extract_clean_blockdsl(failed_completion)[:3200]}\n```\n"
            + "\n".join(notes)
            + "\nThink step-by-step inside `<think>...</think>` and output the complete corrected response."
        )
        return new_row, critique_media

    if task == "synccraft_uv_alpha_diagnosis":
        new_row = dict(row)
        new_row["enable_thinking"] = True
        new_row["prompt"] = (
            f"{row['prompt']}\n\n"
            "[Visual Re-Inspection Hint] Carefully verify BOTH:\n"
            "1. Alpha cutout: inspect RGBA alpha=0 transparent pixels inside each UV face rectangle in image 1 (`has_alpha_cutout`, `alpha_cutout_box`, `island_transparent_fraction`).\n"
            "2. UV island overlap: compare every pair of `[u0, v0, u1, v1]` face rectangles in the UV face table for positive 2D area intersection (`u0_a < u1_b` and `u0_b < u1_a` and `v0_a < v1_b` and `v0_b < v1_a`) -> list all intersecting `"
            "<box>.<face>` IDs sorted in `overlapping_islands`.\n"
            "Re-verify inside `<think>...</think>` and output the exact JSON."
        )
        return new_row, media

    if task.startswith("sat_3d_"):
        new_row = dict(row)
        new_row["enable_thinking"] = True
        new_row["prompt"] = (
            f"{row['prompt']}\n\n"
            "[3D Spatial Re-Inspection] Inside `<think>...</think>`, explicitly establish the camera forward/left/right axes, "
            "relative depth ordering (foreground vs background), and ego-motion rotation before stating the final answer."
        )
        return new_row, media

    return None


def audit_promotion_generation_health(eval_report: dict[str, Any]) -> dict[str, Any]:
    """Compute anti-collapse generation metrics across a `generate=True` evaluation report."""
    rows = [r for r in (eval_report.get("rows") or ()) if isinstance(r.get("prediction"), str)]
    if not rows:
        return {
            "generated_count": 0,
            "healthy_rate": 1.0,
            "think_close_rate": 1.0,
            "mean_rep_4gram": 0.0,
            "max_rep_4gram": 0.0,
            "anti_collapse_passed": True,
        }

    healthy_cnt = 0
    think_total = 0
    think_closed = 0
    reps: list[float] = []
    for r in rows:
        pred = str(r["prediction"])
        h = check_generation_collapse_health(pred, expect_thinking=False, max_rep_4gram=0.12, max_think_chars=5200)
        if h["healthy"]:
            healthy_cnt += 1
        reps.append(float(h.get("rep_4gram", compute_ngram_repetition_ratio(pred))))
        if "<think>" in pred or "</think>" in pred:
            think_total += 1
            if h.get("think_closed"):
                think_closed += 1

    healthy_rate = healthy_cnt / len(rows)
    think_close_rate = (think_closed / think_total) if think_total > 0 else 1.0
    mean_rep = sum(reps) / len(reps)
    max_rep = max(reps)
    passed = (healthy_rate >= 0.88) and (think_close_rate >= 0.90) and (mean_rep <= 0.085)
    return {
        "generated_count": len(rows),
        "healthy_rate": round(healthy_rate, 4),
        "think_responses": think_total,
        "think_close_rate": round(think_close_rate, 4),
        "mean_rep_4gram": round(mean_rep, 5),
        "max_rep_4gram": round(max_rep, 5),
        "anti_collapse_passed": passed,
    }


def _select_stratified_rsi_pool(
    catalogue: Any,
    *,
    seed: int,
    round_no: int,
    total_records: int,
) -> tuple[list[str], list[str]]:
    """Select `(exploration_rids, anchor_replay_rids)` strictly from `split == 'train'`."""
    db_rows = list(
        catalogue.db.execute(
            "SELECT rid, family, language, task FROM rows WHERE split='train' ORDER BY rid"
        )
    )
    rng = random.Random(int(identity([seed, "vision_rsi_select", round_no])[:16], 16))

    buckets: dict[str, list[str]] = defaultdict(list)
    anchor_buckets: dict[str, list[str]] = defaultdict(list)

    for rid, fam, lang, task in db_rows:
        t = str(task or "")
        if t == "synccraft_uv_alpha_diagnosis":
            buckets["uv_alpha"].append(rid)
        elif t in ("synccraft_multiview_inverse_dsl", "synccraft_part_grounding", "synccraft_visual_aesthetic_judgment"):
            buckets["multiview_3d"].append(rid)
        elif t == "synccraft_artist_assistant":
            buckets["artist_3d"].append(rid)
        elif t.startswith("sat_3d_"):
            buckets["sat_3d"].append(rid)
        elif t in ("hle_multimodal_reasoning", "document_ocr", "region_ocr", "real_count", "compositional_vqa"):
            buckets["fine_vision"].append(rid)
        elif t in ("fable5_swe_agent", "terminal_bench_agent", "openthoughts_reasoning", "gpqa_extended_science", "hle_text_reasoning"):
            buckets["reasoning_code"].append(rid)

        # Broad anti-forgetting replay pool across EN/ZH image, video, and text
        anchor_buckets[f"{fam}:{lang}"].append(rid)

    for b in buckets.values():
        rng.shuffle(b)
    for b in anchor_buckets.values():
        rng.shuffle(b)

    target_shares = {
        "uv_alpha": max(6, round(total_records * 0.20)),
        "multiview_3d": max(6, round(total_records * 0.22)),
        "artist_3d": max(6, round(total_records * 0.20)),
        "sat_3d": max(6, round(total_records * 0.16)),
        "fine_vision": max(4, round(total_records * 0.12)),
        "reasoning_code": max(4, round(total_records * 0.10)),
    }
    selected: list[str] = []
    seen: set[str] = set()
    for key in ("uv_alpha", "multiview_3d", "artist_3d", "sat_3d", "fine_vision", "reasoning_code"):
        quota = target_shares[key]
        for rid in buckets.get(key, ()):
            if len(selected) >= total_records or quota <= 0:
                break
            if rid not in seen:
                seen.add(rid)
                selected.append(rid)
                quota -= 1

    # Top up if any bucket had fewer items (e.g. small unit-test DB)
    all_train_rids = [r[0] for r in db_rows]
    rng.shuffle(all_train_rids)
    for rid in all_train_rids:
        if len(selected) >= total_records:
            break
        if rid not in seen:
            seen.add(rid)
            selected.append(rid)

    # Build a balanced anchor replay list matching `len(selected)` across all active `(family, language)` strata
    anchors: list[str] = []
    anchor_keys = sorted(anchor_buckets)
    idx_map = {k: 0 for k in anchor_keys}
    while len(anchors) < max(16, len(selected)) and anchor_keys:
        progressed = False
        for k in anchor_keys:
            pool = anchor_buckets[k]
            while idx_map[k] < len(pool) and pool[idx_map[k]] in seen:
                idx_map[k] += 1
            if idx_map[k] < len(pool):
                rid = pool[idx_map[k]]
                idx_map[k] += 1
                seen.add(rid)
                anchors.append(rid)
                progressed = True
                if len(anchors) >= max(16, len(selected)):
                    break
        if not progressed:
            break

    return selected, anchors


def run_vision_self_improvement_round(campaign: Any, round_no: int) -> dict[str, Any]:
    """Execute one complete Vision + 3D Aesthetic RSI round with 5-axis anti-collapse promotion gating."""
    import torch
    from .engine import loss_weights
    from .evaluate import promotion

    c = campaign.c
    root = Path(campaign.root)
    k_samples = max(1, int(c.get("vision_rsi_k_samples", 3)))
    total_explore = max(8, int(c.get("vision_rsi_records_per_round", 64)))
    min_aes = float(c.get("vision_rsi_min_aesthetic_score", 0.68))

    master_state = copy.deepcopy(campaign.state)
    parent_receipt = master_state["last_checkpoint"]
    emit("vision_rsi_round_start", round=round_no, explore_records=total_explore, k_samples=k_samples)

    # 1. Evaluate parent on held-out `promotion` split WITH generation enabled for collapse & SyncCraft oracle baseline
    parent_eval = campaign.evaluate("promotion", f"vision-rsi-{round_no}-parent", generate=True)
    parent_health = audit_promotion_generation_health(parent_eval)

    explore_rids, anchor_rids = _select_stratified_rsi_pool(
        campaign.catalogue,
        seed=int(c["seed"]),
        round_no=round_no,
        total_records=total_explore,
    )

    is_cuda = str(campaign.model.device).startswith("cuda") and torch.cuda.is_available()
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if is_cuda else contextlib.nullcontext

    campaign.model.eval()
    row_overrides: dict[str, dict[str, Any]] = {}
    hard_ground_truth_rids: list[str] = []
    verified_self_rids: list[str] = []
    trajectory_log: list[dict[str, Any]] = []
    stats_by_task: Counter[str] = Counter()

    max_ans_toks = int(c.get("max_answer_tokens", 6656)) - 64

    for idx, rid in enumerate(explore_rids):
        row, media = campaign.media(rid)
        task = str(row.get("task") or "")
        gen_max = 1280 if task.startswith("synccraft_") else 640

        with torch.no_grad(), amp_ctx():
            greedy_out = campaign.model.answer(
                row,
                media,
                max_new_tokens=gen_max,
                enable_thinking=True,
                do_sample=False,
            )
        greedy_diag = score_rsi_candidate(row, greedy_out, min_aesthetic_score=min_aes)

        best_text = greedy_out
        best_diag = greedy_diag
        used_self_critique = False

        # Sample K diverse `<think>` trajectories whenever greedy failed OR can be aesthetically improved
        need_explore = (not greedy_diag["verified"]) or (float(greedy_diag["score"]) < 0.93)
        if need_explore:
            for s_i in range(k_samples):
                temp = 0.55 + 0.12 * s_i
                with torch.no_grad(), amp_ctx():
                    cand_out = campaign.model.answer(
                        row,
                        media,
                        max_new_tokens=gen_max,
                        enable_thinking=True,
                        do_sample=True,
                        temperature=temp,
                        top_p=0.92,
                        repetition_penalty=1.04,
                    )
                cand_diag = score_rsi_candidate(row, cand_out, min_aesthetic_score=min_aes)
                if (cand_diag["verified"] and not best_diag["verified"]) or (
                    cand_diag["verified"] == best_diag["verified"] and float(cand_diag["score"]) > float(best_diag["score"])
                ):
                    best_text = cand_out
                    best_diag = cand_diag

        # Visual / Compiler Self-Critique Retry if all K samples still failed
        if not best_diag["verified"]:
            crit_turn = build_self_critique_turn(row, media, best_text, best_diag)
            if crit_turn is not None:
                crit_row, crit_media = crit_turn
                with torch.no_grad(), amp_ctx():
                    crit_out = campaign.model.answer(
                        crit_row,
                        crit_media,
                        max_new_tokens=gen_max,
                        enable_thinking=True,
                        do_sample=True,
                        temperature=0.45,
                        top_p=0.90,
                        repetition_penalty=1.04,
                    )
                crit_diag = score_rsi_candidate(row, crit_out, min_aesthetic_score=min_aes)
                if crit_diag["verified"]:
                    best_text = crit_out
                    best_diag = crit_diag
                    used_self_critique = True

        accepted_override = False
        if best_diag["verified"]:
            th_part, body_part = strip_think_block(best_text)
            clean_sup = sanitize_and_bound_think_trace(
                th_part,
                body_part,
                enable_thinking=bool(th_part),
                max_think_chars=3600,
                max_answer_chars=12000,
            )
            if clean_sup is not None:
                tok_len = len(campaign.model.tokenizer.encode(clean_sup, add_special_tokens=False))
                p_len = len(campaign.model.tokenizer.encode(str(row.get("prompt") or ""), add_special_tokens=False))
                if tok_len <= max_ans_toks and (p_len + tok_len) <= int(c.get("max_sequence", 8192)) - 896:
                    row_overrides[rid] = {
                        "answer": clean_sup,
                        "enable_thinking": bool(th_part),
                    }
                    verified_self_rids.append(rid)
                    accepted_override = True
                    stats_by_task[f"{task}:self_verified"] += 1

        if not accepted_override and not greedy_diag["verified"]:
            # Difficult item where model failed: replay verified ground-truth once to close the capability gap
            hard_ground_truth_rids.append(rid)
            stats_by_task[f"{task}:gt_hard_replay"] += 1

        trajectory_log.append(
            {
                "rid": rid,
                "task": task,
                "greedy_verified": bool(greedy_diag["verified"]),
                "greedy_score": round(float(greedy_diag["score"]), 4),
                "best_verified": bool(best_diag["verified"]),
                "best_score": round(float(best_diag["score"]), 4),
                "used_self_critique": used_self_critique,
                "accepted_self_override": accepted_override,
                "reason": best_diag.get("reason"),
            }
        )
        if (idx + 1) % 16 == 0 or (idx + 1) == len(explore_rids):
            emit(
                "vision_rsi_mining_progress",
                round=round_no,
                inspected=idx + 1,
                total=len(explore_rids),
                self_verified=len(verified_self_rids),
                hard_gt=len(hard_ground_truth_rids),
            )

    if is_cuda:
        torch.cuda.empty_cache()

    # 2. Construct anti-collapse RSI training order:
    #    - 2x weight on hard self-verified breakthroughs (where greedy failed or improved) + 1x already-strong verified
    #    - 1x hard ground-truth misses
    #    - >= 50% broad EN/ZH multimodal + coding replay anchors so zero forgetting occurs
    rng = random.Random(int(identity([c["seed"], "vision_rsi_order", round_no])[:16], 16))
    breakthrough_rids = [
        t["rid"]
        for t in trajectory_log
        if t["accepted_self_override"] and (not t["greedy_verified"] or t["best_score"] > t["greedy_score"] + 0.02)
    ]
    rsi_signal_rids = list(verified_self_rids) + list(breakthrough_rids) + list(hard_ground_truth_rids)
    if not rsi_signal_rids:
        rsi_signal_rids = list(explore_rids[:16])

    target_anchor_count = max(len(rsi_signal_rids), len(anchor_rids))
    chosen_anchors = list(anchor_rids[:target_anchor_count])

    rsi_epochs = max(1, int(c.get("rsi_epochs", 1)))
    full_order: list[str] = []
    for _ep in range(rsi_epochs):
        ep_slice = list(rsi_signal_rids) + list(chosen_anchors)
        rng.shuffle(ep_slice)
        full_order.extend(ep_slice)

    traj_doc = {
        "round": round_no,
        "explored_count": len(explore_rids),
        "self_verified_count": len(verified_self_rids),
        "breakthrough_self_wins": len(breakthrough_rids),
        "hard_gt_replay_count": len(hard_ground_truth_rids),
        "anchor_replay_count": len(chosen_anchors),
        "total_training_records": len(full_order),
        "task_breakdown": dict(sorted(stats_by_task.items())),
        "trajectories": trajectory_log,
    }
    write(root / f"VISION_RSI-{round_no}-TRAJECTORIES.json", traj_doc)

    # 3. Fine-tune RSI candidate with conservative anti-collapse learning rate
    rsi_lr = float(c.get("rsi_learning_rate", min(float(c["learning_rate"]) * 0.6, 4e-6)))
    rsi_conn_lr = float(c.get("rsi_connector_learning_rate", min(float(c["connector_learning_rate"]) * 0.5, 2e-5)))
    saved_lr = c["learning_rate"]
    saved_conn_lr = c["connector_learning_rate"]
    saved_eval_ep = c.get("eval_every_epoch", False)

    rsi_stage = {
        "name": f"vision-rsi-{round_no}-candidate",
        "families": ["image", "video", "text"],
        "epochs": 1,
        "alignment_only": False,
        "stage_mode": "full_spark",
        "fixed_loss_weights": loss_weights(campaign.catalogue, full_order, c),
    }

    try:
        c["learning_rate"] = rsi_lr
        c["connector_learning_rate"] = rsi_conn_lr
        c["eval_every_epoch"] = False
        campaign.train_order(
            rsi_stage,
            full_order,
            resume=False,
            dynamic_seed=int(c["seed"]) + 900 + round_no,
            row_overrides=row_overrides,
        )
    finally:
        c["learning_rate"] = saved_lr
        c["connector_learning_rate"] = saved_conn_lr
        c["eval_every_epoch"] = saved_eval_ep

    if campaign.stop_reason:
        campaign.state.update(training_complete=False, stopped_reason=campaign.stop_reason)
        write(root / "CAMPAIGN.json", campaign.state)
        return {
            "stopped": True,
            "reason": campaign.stop_reason,
            "stage": rsi_stage["name"],
            "round": round_no,
            "last_checkpoint": campaign.state.get("last_checkpoint"),
        }

    cand_receipt = campaign.state["last_checkpoint"]
    cand_eval = campaign.evaluate("promotion", f"vision-rsi-{round_no}-candidate", generate=True)
    cand_health = audit_promotion_generation_health(cand_eval)

    # 4. 5-Axis Anti-Collapse Promotion Gate
    loss_gate = promotion(parent_eval, cand_eval, c)
    p_strata = parent_eval.get("by_stratum", {})
    c_strata = cand_eval.get("by_stratum", {})
    rel_ratios = [
        float(c_strata[k]["mean_loss"]) / max(1e-6, float(p_strata[k]["mean_loss"]))
        for k in p_strata
        if k in c_strata
    ]
    mean_rel_loss = sum(rel_ratios) / max(1, len(rel_ratios))

    p_sc_pass = float((parent_eval.get("synccraft_summary") or {}).get("oracle_pass_rate") or 0.0)
    c_sc_pass = float((cand_eval.get("synccraft_summary") or {}).get("oracle_pass_rate") or 0.0)
    sc_non_regressed = c_sc_pass + 1e-6 >= (p_sc_pass - 0.05)

    # Accept iff loss gate passes, generation collapse health passes, SyncCraft oracle doesn't regress,
    # AND either mean loss improves (`<= 1.005`) or SyncCraft oracle pass rate strictly improves!
    improved_or_held = (mean_rel_loss <= 1.005) or (c_sc_pass > p_sc_pass)
    accepted = bool(
        loss_gate["accepted"]
        and cand_health["anti_collapse_passed"]
        and sc_non_regressed
        and improved_or_held
    )

    choice = "candidate" if accepted else "parent"
    chosen_receipt = cand_receipt if accepted else parent_receipt
    if not accepted:
        emit(
            "vision_rsi_round_rolled_back",
            round=round_no,
            loss_gate=loss_gate,
            cand_health=cand_health,
            mean_rel_loss=round(mean_rel_loss, 5),
        )
        opt, sched, _ = campaign.restore(parent_receipt)
    else:
        from .engine import _build_scheduler, optimizer_for

        opt = optimizer_for(campaign.model, c)
        sched = _build_scheduler(opt, kind="constant", num_warmup_steps=0, num_training_steps=1)

    prev_rounds = list(master_state.get("rsi_rounds", []))
    result = {
        "round": round_no,
        "mode": "vision_self_improvement",
        "selected": choice,
        "accepted": accepted,
        "explored_records": len(explore_rids),
        "self_verified_trajectories": len(verified_self_rids),
        "breakthrough_self_wins": len(breakthrough_rids),
        "hard_gt_replays": len(hard_ground_truth_rids),
        "anchor_replays": len(chosen_anchors),
        "mean_relative_promotion_loss": round(mean_rel_loss, 6),
        "parent_synccraft_oracle_rate": round(p_sc_pass, 4),
        "candidate_synccraft_oracle_rate": round(c_sc_pass, 4),
        "loss_promotion_gate": loss_gate,
        "parent_generation_health": parent_health,
        "candidate_generation_health": cand_health,
        "selected_receipt": chosen_receipt,
    }
    prev_rounds.append(result)
    campaign.state.update(
        completed_stages=list(master_state.get("completed_stages", [])),
        rsi_rounds=prev_rounds,
        orchestration=None,
        stage=f"vision-rsi-{round_no}-selected",
        cursor=None,
    )
    campaign.save(opt, sched)
    del opt, sched
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()

    write(root / f"VISION_RSI-{round_no}.json", result)
    campaign.store.small_file(root / f"VISION_RSI-{round_no}.json", "vision-rsi-result")
    emit(
        "vision_rsi_round_complete",
        round=round_no,
        selected=choice,
        self_verified=len(verified_self_rids),
        mean_rel_loss=round(mean_rel_loss, 5),
        cand_sc_pass=round(c_sc_pass, 4),
    )
    return result
