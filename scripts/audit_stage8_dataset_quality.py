#!/usr/bin/env python3
"""Stage-8 Dataset Quality Audit & Visual Verification Sheet Generator.

Validates 100% of Stage-8 full-color shaded 3D Visual Primitives records (and optionally
`nvidia/Nemotron-Image-Training-v3` records) BEFORE Stage-8 training launches:
1. Schema, uniqueness, and frame-blob completeness.
2. RGB image integrity (`min(w, h) >= 128`, `std > 8.0`) & Anti-Cutout full-color shading check
   (`mean(|R-G| + |G-B| + |R-B|) > 5.0` across foreground pixels on all `spatial_3d_*` renders).
3. Strict 4-primitive XML syntax & geometry validation (`<point>`, `<box>`, `<dash>`, `<circle>`).
4. Balanced `<think>...</think>`, `check_generation_collapse_health` (`rep_4gram < 0.15`), and length caps (`<= 7000`).
5. Distribution balance & anti-bias checks: facing L/R `[0.35, 0.65]`, Yes/No `[0.45, 0.55]`,
   bilingual `zh` share `>= 0.10`, and all 6 required 3D tasks present.
6. Zero eval-benchmark leakage against `BANNED_BENCHMARK_SUBSTRINGS`.
7. Visual overlay contact sheets (`primitive_overlay_samples.png` 3x3 + optional `nemotron_overlay_samples.png` 2x3)
   with exact parsed `<box>`, `<dash>` (with arrowheads), `<circle>`, and depth-coded `<point>` overlays.
8. Structured `STAGE8_DATASET_AUDIT.json` summary + deterministic exit code (`0` iff passed, else `1`).
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.final.reasoning_agentic_corpus import check_generation_collapse_health  # noqa: E402
from colab.final.spatial_3d_primitives_corpus import (  # noqa: E402
    SPATIAL_3D_PRIMITIVE_TASKS,
    build_nemotron_v3_spatial_records,
    build_shaded_3d_spatial_primitives_records,
)
from colab.final.visual_primitives_corpus import BANNED_BENCHMARK_SUBSTRINGS  # noqa: E402

REQUIRED_RECORD_KEYS: tuple[str, ...] = ("id", "task", "language", "prompt", "answer", "frames")
REQUIRED_3D_TASKS: tuple[str, ...] = tuple(SPATIAL_3D_PRIMITIVE_TASKS)

_PRIM_TAG_RE = re.compile(r'<(point|box|dash|circle)\b((?:"[^"]*"|[^">])*)/>')
_ANY_PRIM_OPEN_RE = re.compile(r'<(point|box|dash|circle)\b(?:"[^"]*"|[^">])*>?')
_ATTR_RE = re.compile(r'([a-zA-Z0-9_]+)\s*=\s*"([^"]*)"')
_THINK_SPAN_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)


def parse_visual_primitives(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse and strictly validate every `<point>`, `<box>`, `<dash>`, and `<circle>` tag in `text`.

    Returns `(primitives, errors)`.
    """
    raw = str(text or "")
    primitives: list[dict[str, Any]] = []
    errors: list[str] = []

    matched_spans: set[tuple[int, int]] = set()
    for m in _PRIM_TAG_RE.finditer(raw):
        matched_spans.add(m.span())
        kind = m.group(1)
        attr_str = m.group(2)
        attrs = dict(_ATTR_RE.findall(attr_str))

        def _req_int(k: str, lo: int = 0, hi: int = 1000) -> int | None:
            if k not in attrs:
                errors.append(f"{kind}_missing_{k}")
                return None
            v_str = attrs[k].strip()
            if not re.fullmatch(r"-?\d+", v_str):
                errors.append(f"{kind}_non_int_{k}:{v_str}")
                return None
            val = int(v_str)
            if val < lo or val > hi:
                errors.append(f"{kind}_out_of_range_{k}:{val}")
                return None
            return val

        lbl = attrs.get("label", "")
        if kind == "point":
            x = _req_int("x", 0, 1000)
            y = _req_int("y", 0, 1000)
            z = attrs.get("z")
            if z is not None and z not in ("near", "mid", "far"):
                errors.append(f"point_invalid_z:{z}")
                continue
            if x is None or y is None:
                continue
            primitives.append({"kind": "point", "x": x, "y": y, "z": z, "label": lbl})
        elif kind == "box":
            x1 = _req_int("x1", 0, 1000)
            y1 = _req_int("y1", 0, 1000)
            x2 = _req_int("x2", 0, 1000)
            y2 = _req_int("y2", 0, 1000)
            if None in (x1, y1, x2, y2):
                continue
            assert x1 is not None and y1 is not None and x2 is not None and y2 is not None
            if x1 > x2 or y1 > y2:
                errors.append(f"box_inverted:{x1},{y1},{x2},{y2}")
                continue
            if (x2 - x1) < 2 and (y2 - y1) < 2:
                errors.append(f"box_degenerate:{x1},{y1},{x2},{y2}")
                continue
            primitives.append({"kind": "box", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": lbl})
        elif kind == "dash":
            x1 = _req_int("x1", 0, 1000)
            y1 = _req_int("y1", 0, 1000)
            x2 = _req_int("x2", 0, 1000)
            y2 = _req_int("y2", 0, 1000)
            if None in (x1, y1, x2, y2):
                continue
            assert x1 is not None and y1 is not None and x2 is not None and y2 is not None
            dist = math.hypot(x2 - x1, y2 - y1)
            if dist < 4.0:
                errors.append(f"dash_degenerate_len:{dist:.2f}")
                continue
            primitives.append({"kind": "dash", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": lbl})
        elif kind == "circle":
            cx = _req_int("cx", 0, 1000)
            cy = _req_int("cy", 0, 1000)
            r = _req_int("r", 8, 400)
            if None in (cx, cy, r):
                continue
            primitives.append({"kind": "circle", "cx": cx, "cy": cy, "r": r, "label": lbl})

    # Detect malformed or unclosed primitive tags (skip explanatory prose mentions like `<point>` or `<box>`)
    for m in _ANY_PRIM_OPEN_RE.finditer(raw):
        if m.span() in matched_spans:
            continue
        snippet = m.group(0)
        # Prose references like `<point>`, `<box>`, `<dash>`, `<circle>` have no attributes; only flag broken tags with `=`!
        if "=" in snippet:
            errors.append(f"malformed_primitive_tag:{snippet[:60]}")

    return primitives, errors


def _inspect_image_metrics(raw_bytes: bytes) -> dict[str, Any]:
    """Decode image bytes and compute size, pixel std, and foreground full-color chroma spread."""
    try:
        with Image.open(io.BytesIO(raw_bytes)) as im:
            fmt = str(im.format or "").upper()
            rgb = im.convert("RGB")
            w, h = rgb.size
            arr = np.asarray(rgb, dtype=np.float32)
    except Exception as exc:
        return {"ok": False, "error": f"decode_error:{type(exc).__name__}:{exc}"}

    std_all = float(np.std(arr))
    # Foreground chroma check: isolate pixels differing from the 4 border strips' median RGB
    bh = max(2, min(6, h // 16))
    bw = max(2, min(6, w // 16))
    border_px = np.concatenate(
        [
            arr[:bh, :, :].reshape(-1, 3),
            arr[-bh:, :, :].reshape(-1, 3),
            arr[:, :bw, :].reshape(-1, 3),
            arr[:, -bw:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_med = np.median(border_px, axis=0)
    fg_dist = np.sum(np.abs(arr - bg_med[None, None, :]), axis=-1)
    fg_mask = fg_dist > 18.0
    fg_pixels = arr[fg_mask] if int(np.sum(fg_mask)) >= 32 else arr.reshape(-1, 3)
    chroma_l1 = float(
        np.mean(
            np.abs(fg_pixels[:, 0] - fg_pixels[:, 1])
            + np.abs(fg_pixels[:, 1] - fg_pixels[:, 2])
            + np.abs(fg_pixels[:, 0] - fg_pixels[:, 2])
        )
    )
    return {
        "ok": True,
        "format": fmt,
        "width": int(w),
        "height": int(h),
        "std": round(std_all, 3),
        "fg_chroma_l1": round(chroma_l1, 3),
        "fg_ratio": round(float(np.mean(fg_mask)), 4),
    }


def _classify_yesno_final_answer(final_answer: str) -> str | None:
    s = str(final_answer or "").strip()
    low = s.lower()
    if re.match(r"^(yes)\b", low) or s.startswith("是"):
        return "yes"
    if re.match(r"^(no)\b", low) or s.startswith("否") or s.startswith("不是"):
        return "no"
    return None


def _length_bucket(n_chars: int) -> str:
    if n_chars <= 500:
        return "0-500"
    if n_chars <= 1000:
        return "501-1000"
    if n_chars <= 2000:
        return "1001-2000"
    if n_chars <= 3500:
        return "2001-3500"
    if n_chars <= 5000:
        return "3501-5000"
    if n_chars <= 7000:
        return "5001-7000"
    return ">7000"


def audit_records(
    pairs: list[tuple[dict[str, Any], dict[str, bytes]]],
    *,
    check_distribution: bool = True,
    check_all_tasks: bool = True,
    max_rep_4gram: float = 0.15,
    max_total_chars: int = 7000,
) -> dict[str, Any]:
    """Run all 6 programmatic quality gates across 100% of `(row, blobs)` records and return a full structured report."""
    failures: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    img_cache: dict[str, dict[str, Any]] = {}

    task_counts: Counter[str] = Counter()
    lang_counts: Counter[str] = Counter()
    lang_3d_counts: Counter[str] = Counter()
    prim_counts: Counter[str] = Counter()
    prim_3d_counts: Counter[str] = Counter()
    facing_lr_counts: Counter[str] = Counter()
    yesno_counts: Counter[str] = Counter()
    yesno_en_counts: Counter[str] = Counter()
    len_hist: Counter[str] = Counter()
    rep_values: list[float] = []
    total_char_values: list[int] = []
    chroma_3d_values: list[float] = []

    def _fail(rid: str, check: str, detail: str) -> None:
        failures.append({"id": rid, "check": check, "detail": detail})

    if not pairs:
        _fail("<corpus>", "non_empty", "No records provided for audit")

    for idx, pair in enumerate(pairs):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            _fail(f"idx_{idx}", "schema", "Record entry must be a (row, blobs) 2-tuple")
            continue
        row, blobs = pair
        if not isinstance(row, dict) or not isinstance(blobs, dict):
            _fail(f"idx_{idx}", "schema", "Row and blobs must be dicts")
            continue

        rid = str(row.get("id") or f"idx_{idx}")
        # 1. Schema & Uniqueness
        missing = [k for k in REQUIRED_RECORD_KEYS if k not in row]
        if missing:
            _fail(rid, "schema_missing_keys", f"Missing required keys: {missing}")
            continue
        if not str(row["id"]).strip():
            _fail(rid, "schema_empty_id", "Empty record id")
        if rid in seen_ids:
            _fail(rid, "duplicate_id", f"Duplicate record id: {rid}")
        seen_ids.add(rid)

        task = str(row["task"]).strip()
        lang = str(row["language"]).strip()
        prompt = str(row["prompt"])
        answer = str(row["answer"])
        frames = row["frames"]
        is_3d = task.startswith("spatial_3d_")

        task_counts[task] += 1
        lang_counts[lang] += 1
        if is_3d:
            lang_3d_counts[lang] += 1

        if not prompt.strip() or not answer.strip():
            _fail(rid, "empty_text", "Empty prompt or answer")

        if not isinstance(frames, list) or not frames:
            _fail(rid, "missing_frames", "Record frames must be a non-empty list")
            frames = []

        # 2. Image Integrity & Anti-Cutout Shading Check
        for fpath in frames:
            raw_b = blobs.get(fpath)
            if not isinstance(raw_b, (bytes, bytearray)) or len(raw_b) == 0:
                _fail(rid, "missing_blob", f"Frame '{fpath}' missing or empty in blobs")
                continue
            b_key = hashlib.sha256(raw_b).hexdigest()
            metrics = img_cache.get(b_key)
            if metrics is None:
                metrics = _inspect_image_metrics(bytes(raw_b))
                img_cache[b_key] = metrics
            if not metrics["ok"]:
                _fail(rid, "invalid_image", f"Frame '{fpath}': {metrics['error']}")
                continue
            if min(metrics["width"], metrics["height"]) < 128:
                _fail(rid, "image_too_small", f"Frame '{fpath}' size={metrics['width']}x{metrics['height']} < 128")
            if metrics["std"] <= 8.0:
                _fail(rid, "image_low_std", f"Frame '{fpath}' std={metrics['std']} <= 8.0")
            if is_3d:
                chroma_3d_values.append(metrics["fg_chroma_l1"])
                if metrics["fg_chroma_l1"] <= 5.0:
                    _fail(
                        rid,
                        "bw_cutout_detected",
                        f"Frame '{fpath}' fg_chroma_l1={metrics['fg_chroma_l1']} <= 5.0 (possible B&W cutout!)",
                    )

        # 3. 4-Primitive XML Syntax & Geometry Validity
        prims, p_errs = parse_visual_primitives(answer)
        for pe in p_errs:
            _fail(rid, "primitive_invalid", pe)
        for pr in prims:
            prim_counts[pr["kind"]] += 1
            if is_3d:
                prim_3d_counts[pr["kind"]] += 1

        # 4. CoT <think> & Anti-Collapse Check
        has_open = "<think>" in answer
        has_close = "</think>" in answer
        think_span = _THINK_SPAN_RE.search(answer)
        final_ans = answer
        if not (has_open and has_close and think_span is not None and answer.count("<think>") == 1 and answer.count("</think>") == 1):
            _fail(rid, "think_unbalanced", "Missing or unbalanced <think>...</think>")
        else:
            think_body = think_span.group(1).strip()
            final_ans = (answer[: think_span.start()] + answer[think_span.end() :]).strip()
            if not think_body:
                _fail(rid, "think_empty", "Empty <think> reasoning block")
            if not final_ans:
                _fail(rid, "final_answer_empty", "Empty final answer outside </think>")

        health = check_generation_collapse_health(
            answer,
            expect_thinking=bool(row.get("enable_thinking", True)),
            max_rep_4gram=max_rep_4gram - 1e-6,
            max_think_chars=4800,
        )
        rep_val = float(health.get("rep_4gram", 0.0))
        rep_values.append(rep_val)
        if not health.get("healthy") or rep_val >= max_rep_4gram:
            _fail(rid, "collapse_health", f"reason={health.get('reason')} rep_4gram={rep_val:.4f}")

        total_chars = len(prompt) + len(answer)
        total_char_values.append(total_chars)
        len_hist[_length_bucket(total_chars)] += 1
        if total_chars > max_total_chars:
            _fail(rid, "excessive_length", f"prompt+answer={total_chars} > {max_total_chars}")

        # 5. Track Distribution Metrics
        if task == "spatial_3d_left_right_orientation_facing":
            flr = str((row.get("metadata") or {}).get("facing_lr") or "").lower()
            if flr in ("left", "right"):
                facing_lr_counts[flr] += 1
        if task == "spatial_3d_yesno_depth_and_lr":
            yn_cls = _classify_yesno_final_answer(final_ans)
            if yn_cls is None:
                _fail(rid, "yesno_unclear_answer", f"Final answer does not begin cleanly with Yes/No: {final_ans[:50]!r}")
            else:
                yesno_counts[yn_cls] += 1
                if lang == "en":
                    yesno_en_counts[yn_cls] += 1

        # 6. Zero Eval-Benchmark Leakage
        leak_surface = " ".join(
            [
                rid,
                str(row.get("source_group") or ""),
                json.dumps(row.get("metadata") or {}, ensure_ascii=False),
                " ".join(str(f) for f in frames),
            ]
        ).lower()
        for banned in BANNED_BENCHMARK_SUBSTRINGS:
            if banned in leak_surface:
                _fail(rid, "benchmark_leakage", f"Matched banned benchmark substring '{banned}'")

    # Corpus-level distribution & coverage checks
    n_3d = sum(lang_3d_counts.values())
    zh_3d_share = (lang_3d_counts.get("zh", 0) / float(n_3d)) if n_3d > 0 else 0.0
    lr_total = facing_lr_counts.get("left", 0) + facing_lr_counts.get("right", 0)
    facing_left_ratio = (facing_lr_counts.get("left", 0) / float(lr_total)) if lr_total > 0 else None
    yn_total = yesno_counts.get("yes", 0) + yesno_counts.get("no", 0)
    yes_ratio = (yesno_counts.get("yes", 0) / float(yn_total)) if yn_total > 0 else None

    if check_distribution and n_3d > 0:
        if check_all_tasks:
            missing_tasks = [t for t in REQUIRED_3D_TASKS if task_counts.get(t, 0) <= 0]
            if missing_tasks:
                _fail("<corpus>", "missing_3d_tasks", f"Missing required 3D tasks: {missing_tasks}")

        missing_prims = [p for p in ("point", "box", "dash", "circle") if prim_3d_counts.get(p, 0) <= 0]
        if missing_prims:
            _fail("<corpus>", "missing_3d_primitives", f"Missing primitive types in 3D corpus: {missing_prims}")

        if facing_left_ratio is not None and not (0.35 <= facing_left_ratio <= 0.65):
            _fail(
                "<corpus>",
                "facing_lr_imbalance",
                f"facing left ratio={facing_left_ratio:.3f} outside [0.35, 0.65] (counts={dict(facing_lr_counts)})",
            )
        if yn_total > 0 and (yes_ratio is None or not (0.45 <= yes_ratio <= 0.55)):
            _fail(
                "<corpus>",
                "yesno_imbalance",
                f"Yes ratio={yes_ratio} outside [0.45, 0.55] (counts={dict(yesno_counts)})",
            )
        if zh_3d_share < 0.10:
            _fail(
                "<corpus>",
                "low_zh_share",
                f"Bilingual zh share on 3D corpus={zh_3d_share:.3f} < 0.10 (counts={dict(lang_3d_counts)})",
            )

    rep_sorted = sorted(rep_values)
    len_sorted = sorted(total_char_values)
    chroma_sorted = sorted(chroma_3d_values)

    def _pct(arr_s: list[float] | list[int], q: float) -> float:
        if not arr_s:
            return 0.0
        pos = min(len(arr_s) - 1, max(0, int(math.floor(q * (len(arr_s) - 1)))))
        return float(arr_s[pos])

    return {
        "passed": len(failures) == 0,
        "total_records": len(pairs),
        "unique_images_checked": len(img_cache),
        "task_counts": dict(sorted(task_counts.items())),
        "language_counts": dict(sorted(lang_counts.items())),
        "language_3d_counts": dict(sorted(lang_3d_counts.items())),
        "zh_3d_share": round(zh_3d_share, 4),
        "primitive_counts": dict(sorted(prim_counts.items())),
        "primitive_3d_counts": dict(sorted(prim_3d_counts.items())),
        "facing_lr_balance": {
            "left": int(facing_lr_counts.get("left", 0)),
            "right": int(facing_lr_counts.get("right", 0)),
            "left_ratio": round(facing_left_ratio, 4) if facing_left_ratio is not None else None,
        },
        "yesno_balance": {
            "yes": int(yesno_counts.get("yes", 0)),
            "no": int(yesno_counts.get("no", 0)),
            "yes_ratio": round(yes_ratio, 4) if yes_ratio is not None else None,
            "en_yes": int(yesno_en_counts.get("yes", 0)),
            "en_no": int(yesno_en_counts.get("no", 0)),
        },
        "repetition_stats": {
            "max_rep_4gram": round(max(rep_values) if rep_values else 0.0, 5),
            "p95_rep_4gram": round(_pct(rep_sorted, 0.95), 5),
            "median_rep_4gram": round(_pct(rep_sorted, 0.50), 5),
        },
        "length_stats": {
            "min_chars": int(min(total_char_values)) if total_char_values else 0,
            "median_chars": int(_pct(len_sorted, 0.50)),
            "p95_chars": int(_pct(len_sorted, 0.95)),
            "max_chars": int(max(total_char_values)) if total_char_values else 0,
            "histogram": dict(len_hist),
        },
        "shading_chroma_3d_stats": {
            "min_fg_chroma_l1": round(min(chroma_3d_values), 3) if chroma_3d_values else None,
            "median_fg_chroma_l1": round(_pct(chroma_sorted, 0.50), 3) if chroma_3d_values else None,
            "max_fg_chroma_l1": round(max(chroma_3d_values), 3) if chroma_3d_values else None,
        },
        "failures_count": len(failures),
        "failures": failures[:200],
    }


def _load_font(size: int = 12, *, cjk: bool = False) -> tuple[Any, bool]:
    cjk_candidates = (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    )
    latin_candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    )
    if cjk:
        for fp in cjk_candidates:
            if Path(fp).is_file():
                try:
                    return ImageFont.truetype(fp, size=size), True
                except Exception:
                    continue
    for fp in latin_candidates:
        if Path(fp).is_file():
            try:
                return ImageFont.truetype(fp, size=size), False
            except Exception:
                continue
    return ImageFont.load_default(), False


def _format_caption(text: str, max_len: int = 64, *, allow_unicode: bool = False) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not allow_unicode:
        clean = clean.encode("ascii", errors="replace").decode("ascii")
    return clean[:max_len] + ("..." if len(clean) > max_len else "")


def _draw_arrow_head(draw: ImageDraw.ImageDraw, x1: float, y1: float, x2: float, y2: float, color: tuple[int, int, int], size: float = 9.0) -> None:
    ang = math.atan2(y2 - y1, x2 - x1)
    wing = math.radians(26.0)
    p1 = (x2, y2)
    p2 = (x2 - size * math.cos(ang - wing), y2 - size * math.sin(ang - wing))
    p3 = (x2 - size * math.cos(ang + wing), y2 - size * math.sin(ang + wing))
    draw.polygon([p1, p2, p3], fill=color)


def _draw_primitives_onto_canvas(
    draw: ImageDraw.ImageDraw,
    prims: list[dict[str, Any]],
    *,
    ox: int,
    oy: int,
    draw_w: int,
    draw_h: int,
    font_sm: Any,
    label_filter: Any | None = None,
) -> Counter[str]:
    """Render parsed `<box>`, `<circle>`, `<dash>` (with arrowheads), and depth-coded `<point>` onto a tile."""
    counts: Counter[str] = Counter()

    def _sx(v: float) -> float:
        return ox + (float(v) / 1000.0) * (draw_w - 1)

    def _sy(v: float) -> float:
        return oy + (float(v) / 1000.0) * (draw_h - 1)

    # Order: boxes -> circles -> dashes -> points so points & arrowheads stay crisp on top!
    boxes = [p for p in prims if p["kind"] == "box"]
    circles = [p for p in prims if p["kind"] == "circle"]
    dashes = [p for p in prims if p["kind"] == "dash"]
    points = [p for p in prims if p["kind"] == "point"]

    for p in boxes:
        if label_filter and not label_filter(p):
            continue
        counts["box"] += 1
        x1, y1, x2, y2 = _sx(p["x1"]), _sy(p["y1"]), _sx(p["x2"]), _sy(p["y2"])
        draw.rectangle([x1, y1, x2, y2], outline=(34, 211, 238), width=2)

    for p in circles:
        if label_filter and not label_filter(p):
            continue
        counts["circle"] += 1
        cx, cy = _sx(p["cx"]), _sy(p["cy"])
        rx = (float(p["r"]) / 1000.0) * draw_w
        ry = (float(p["r"]) / 1000.0) * draw_h
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=(132, 204, 22), width=2)
        draw.line([cx - 5, cy, cx + 5, cy], fill=(132, 204, 22), width=1)
        draw.line([cx, cy - 5, cx, cy + 5], fill=(132, 204, 22), width=1)

    for p in dashes:
        if label_filter and not label_filter(p):
            continue
        counts["dash"] += 1
        x1, y1, x2, y2 = _sx(p["x1"]), _sy(p["y1"]), _sx(p["x2"]), _sy(p["y2"])
        amber = (245, 158, 11)
        draw.line([x1, y1, x2, y2], fill=amber, width=3)
        _draw_arrow_head(draw, x1, y1, x2, y2, amber, size=10.0)

    z_colors = {
        "near": (239, 68, 68),    # Red
        "mid": (250, 204, 21),    # Yellow
        "far": (59, 130, 246),    # Blue
    }
    drawn_labels = 0
    for p in points:
        if label_filter and not label_filter(p):
            continue
        counts["point"] += 1
        cx, cy = _sx(p["x"]), _sy(p["y"])
        col = z_colors.get(p.get("z") or "", (232, 121, 249))
        r_dot = 5
        draw.ellipse([cx - r_dot - 1, cy - r_dot - 1, cx + r_dot + 1, cy + r_dot + 1], fill=(255, 255, 255))
        draw.ellipse([cx - r_dot, cy - r_dot, cx + r_dot, cy + r_dot], fill=col)
        if drawn_labels < 6 and p.get("label"):
            short_l = _format_caption(str(p["label"]).split(":")[-1], 14, allow_unicode=False)
            tx = min(ox + draw_w - 64, max(ox + 2, int(cx + 6)))
            ty = min(oy + draw_h - 14, max(oy + 2, int(cy - 12)))
            draw.rectangle([tx - 1, ty - 1, tx + len(short_l) * 6 + 3, ty + 12], fill=(15, 23, 42))
            draw.text((tx + 1, ty), short_l, fill=(248, 250, 252), font=font_sm)
            drawn_labels += 1

    return counts


def _render_contact_sheet(
    tile_specs: list[dict[str, Any]],
    *,
    cols: int,
    rows: int,
    out_path: Path,
    sheet_title: str,
    tile_img_size: int = 384,
) -> Path:
    hdr_h = 28
    ftr_h = 40
    pad = 10
    legend_h = 34
    title_h = 32
    tile_w = tile_img_size
    tile_h = hdr_h + tile_img_size + ftr_h

    canvas_w = pad + cols * (tile_w + pad)
    canvas_h = title_h + pad + rows * (tile_h + pad) + legend_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (12, 16, 26))
    draw = ImageDraw.Draw(canvas)
    font_title, _ = _load_font(15, cjk=False)
    font_hdr, _ = _load_font(12, cjk=False)
    font_sm, has_cjk = _load_font(10, cjk=True)

    draw.text((pad, 7), _format_caption(sheet_title, 120, allow_unicode=False), fill=(241, 245, 249), font=font_title)

    for idx, spec in enumerate(tile_specs[: cols * rows]):
        r_i, c_i = divmod(idx, cols)
        tx = pad + c_i * (tile_w + pad)
        ty = title_h + pad + r_i * (tile_h + pad)

        # Tile frame background
        draw.rectangle([tx - 1, ty - 1, tx + tile_w + 1, ty + tile_h + 1], fill=(22, 29, 44), outline=(51, 65, 85), width=1)
        draw.rectangle([tx, ty, tx + tile_w, ty + hdr_h], fill=(30, 41, 59))
        draw.text((tx + 6, ty + 7), _format_caption(spec["header"], 52, allow_unicode=False), fill=(226, 232, 240), font=font_hdr)

        # Aspect-preserving letterbox fit into tile_img_size x tile_img_size!
        with Image.open(io.BytesIO(spec["img_bytes"])) as raw_im:
            rgb = raw_im.convert("RGB")
            iw, ih = rgb.size
            scale = min(tile_img_size / max(1.0, float(iw)), tile_img_size / max(1.0, float(ih)))
            nw = max(1, int(round(iw * scale)))
            nh = max(1, int(round(ih * scale)))
            resized = rgb.resize((nw, nh), resample=Image.Resampling.BICUBIC)

        img_area_y = ty + hdr_h
        ox = tx + (tile_img_size - nw) // 2
        oy = img_area_y + (tile_img_size - nh) // 2
        draw.rectangle([tx, img_area_y, tx + tile_w, img_area_y + tile_img_size], fill=(10, 14, 22))
        canvas.paste(resized, (ox, oy))

        prims, _ = parse_visual_primitives(spec["answer"])
        cnts = _draw_primitives_onto_canvas(
            draw,
            prims,
            ox=ox,
            oy=oy,
            draw_w=nw,
            draw_h=nh,
            font_sm=font_sm,
            label_filter=spec.get("label_filter"),
        )

        # Footer strip
        fy = img_area_y + tile_img_size
        draw.rectangle([tx, fy, tx + tile_w, ty + tile_h], fill=(18, 24, 38))
        p_summary = f"P:{cnts['point']} B:{cnts['box']} D:{cnts['dash']} C:{cnts['circle']}"
        line1 = f"{p_summary} | {_format_caption(spec.get('subline', ''), 40, allow_unicode=has_cjk)}"
        line2 = _format_caption(spec.get("caption", ""), 54, allow_unicode=has_cjk)
        draw.text((tx + 6, fy + 4), line1, fill=(148, 163, 184), font=font_sm)
        draw.text((tx + 6, fy + 20), line2, fill=(226, 232, 240), font=font_sm)

    # Bottom Legend Strip
    ly = canvas_h - legend_h + 6
    legend_items = [
        ((34, 211, 238), "<box> cyan"),
        ((245, 158, 11), "<dash> amber arrow"),
        ((132, 204, 22), "<circle> lime pivot"),
        ((239, 68, 68), "<point z=near> red"),
        ((250, 204, 21), "<point z=mid> yellow"),
        ((59, 130, 246), "<point z=far> blue"),
    ]
    lx = pad
    for col, txt in legend_items:
        draw.rectangle([lx, ly + 3, lx + 12, ly + 15], fill=col)
        draw.text((lx + 17, ly + 2), txt, fill=(226, 232, 240), font=font_hdr)
        lx += 165

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")
    return out_path


def render_primitive_overlay_sheet(
    pairs_3d: list[tuple[dict[str, Any], dict[str, bytes]]],
    out_path: Path,
) -> Path | None:
    """Build the 3x3 visual verification sheet (`primitive_overlay_samples.png`) across 8+ distinct sculpted 3D assets."""
    if not pairs_3d:
        return None

    used_ids: set[str] = set()
    used_families: set[str] = set()

    def _fam_key(r: dict[str, Any]) -> str:
        sg = str(r.get("source_group") or "")
        return re.sub(r"_(kin|fb|lr|mv|anim|yn)$", "", sg)

    def _pick(pred: Any, *, prefer_hero: bool = False) -> tuple[dict[str, Any], dict[str, bytes]] | None:
        # Pass 1: unseen family (preferring dragon/creature hero models when requested)
        for require_fresh_fam in (True, False):
            cands = []
            for pair in pairs_3d:
                r = pair[0]
                if r["id"] in used_ids:
                    continue
                if not pred(r):
                    continue
                fk = _fam_key(r)
                if require_fresh_fam and fk in used_families:
                    continue
                is_hero = any(k in r["id"] for k in ("dragon", "drake", "wyrm", "wyvern", "automaton", "crawler"))
                cands.append((0 if (prefer_hero and is_hero) else 1, pair))
            if cands:
                cands.sort(key=lambda x: x[0])
                chosen = cands[0][1]
                used_ids.add(chosen[0]["id"])
                used_families.add(_fam_key(chosen[0]))
                return chosen
        return None

    tile_specs: list[dict[str, Any]] = []

    def _add_single(pair: tuple[dict[str, Any], dict[str, bytes]] | None, tag_note: str) -> None:
        if pair is None:
            return
        r, b = pair
        f0 = r["frames"][0]
        final_ans = r["answer"].split("</think>")[-1].strip()
        tile_specs.append(
            {
                "header": f"{r['task'].replace('spatial_3d_', '')} [{r['language']}]",
                "subline": f"{tag_note} | {r['id'][:24]}",
                "caption": final_ans,
                "img_bytes": b[f0],
                "answer": r["answer"],
            }
        )

    # 1. Kinematic chains (EN, hero dragon/creature)
    _add_single(_pick(lambda r: r["task"] == "spatial_3d_kinematic_dash_and_joints" and r["language"] == "en", prefer_hero=True), "kin-EN")
    # 2. Kinematic chains (ZH, distinct asset)
    _add_single(_pick(lambda r: r["task"] == "spatial_3d_kinematic_dash_and_joints" and r["language"] == "zh", prefer_hero=True), "kin-ZH")
    # 3. Front/Back multi-hop depth (EN, distinct hero creature)
    _add_single(_pick(lambda r: r["task"] == "spatial_3d_front_back_depth_order" and r["language"] == "en", prefer_hero=True), "depth-EN")
    # 4. Front/Back multi-hop depth (ZH, distinct vehicle/creature)
    _add_single(_pick(lambda r: r["task"] == "spatial_3d_front_back_depth_order" and r["language"] == "zh", prefer_hero=True), "depth-ZH")
    # 5. Left/Right facing orientation (EN, distinct hero profile/3-4)
    _add_single(_pick(lambda r: r["task"] == "spatial_3d_left_right_orientation_facing" and r["language"] == "en", prefer_hero=True), "facing-EN")

    # 6 & 7. Animation Arc BEFORE (Frame 1 Rest) + AFTER (Frame 2 Articulated Pose with anim_arc <dash>!)
    anim_pair = _pick(lambda r: r["task"] == "spatial_3d_rig_animation_arc" and len(r["frames"]) >= 2, prefer_hero=True)
    if anim_pair is not None:
        r, b = anim_pair
        final_ans = r["answer"].split("</think>")[-1].strip()
        act = str((r.get("metadata") or {}).get("action_id") or "pose")
        tile_specs.append(
            {
                "header": f"rig_anim Frame1 REST [{r['language']}]",
                "subline": f"{act} | pivot+before",
                "caption": f"REST -> {final_ans}",
                "img_bytes": b[r["frames"][0]],
                "answer": r["answer"],
                "label_filter": lambda p: (p["kind"] == "circle") or str(p.get("label", "")).startswith("before:"),
            }
        )
        tile_specs.append(
            {
                "header": f"rig_anim Frame2 POSED [{r['language']}]",
                "subline": f"{act} | after+anim_arc",
                "caption": final_ans,
                "img_bytes": b[r["frames"][1]],
                "answer": r["answer"],
                "label_filter": lambda p: (p["kind"] in ("circle", "dash")) or str(p.get("label", "")).startswith("after:"),
            }
        )

    # 8. Yes/No — Positive ("Yes", distinct asset)
    _add_single(
        _pick(lambda r: r["task"] == "spatial_3d_yesno_depth_and_lr" and (r.get("metadata") or {}).get("ground_truth_yesno") == "yes", prefer_hero=True),
        "YesNo=YES",
    )
    # 9. Yes/No — Negative ("No", distinct asset)
    _add_single(
        _pick(lambda r: r["task"] == "spatial_3d_yesno_depth_and_lr" and (r.get("metadata") or {}).get("ground_truth_yesno") == "no"),
        "YesNo=NO",
    )

    # Fill any remaining slots up to 9 cleanly
    while len(tile_specs) < 9:
        extra = _pick(lambda _r: True)
        if extra is None:
            break
        _add_single(extra, "extra")

    return _render_contact_sheet(
        tile_specs[:9],
        cols=3,
        rows=3,
        out_path=out_path,
        sheet_title="Stage-8 Full-Color Shaded 3D Visual Primitives Verification (3x3 Overlay Sheet)",
        tile_img_size=384,
    )


def render_nemotron_overlay_sheet(
    pairs_nem: list[tuple[dict[str, Any], dict[str, bytes]]],
    out_path: Path,
) -> Path | None:
    """Build a 2x3 Nemotron-v3 overlay verification sheet (`nemotron_overlay_samples.png`)."""
    if not pairs_nem:
        return None
    # Prioritize records that contain parsed visual primitives across distinct Nemotron tasks!
    by_task: dict[str, list[tuple[dict[str, Any], dict[str, bytes]]]] = {}
    for pair in pairs_nem:
        by_task.setdefault(pair[0]["task"], []).append(pair)

    chosen: list[tuple[dict[str, Any], dict[str, bytes]]] = []
    used: set[str] = set()
    # Pass 1: pick 1 record WITH primitives per distinct task
    for task in sorted(by_task.keys()):
        for pair in by_task[task]:
            prims, _ = parse_visual_primitives(pair[0]["answer"])
            if prims and pair[0]["id"] not in used:
                used.add(pair[0]["id"])
                chosen.append(pair)
                break
        if len(chosen) >= 6:
            break
    # Pass 2: fill remaining slots with any record having primitives, then any diverse task record
    for pair in pairs_nem:
        if len(chosen) >= 6:
            break
        if pair[0]["id"] in used:
            continue
        prims, _ = parse_visual_primitives(pair[0]["answer"])
        if prims:
            used.add(pair[0]["id"])
            chosen.append(pair)
    for pair in pairs_nem:
        if len(chosen) >= 6:
            break
        if pair[0]["id"] not in used:
            used.add(pair[0]["id"])
            chosen.append(pair)

    tile_specs: list[dict[str, Any]] = []
    for r, b in chosen[:6]:
        f0 = r["frames"][0]
        final_ans = r["answer"].split("</think>")[-1].strip()
        tile_specs.append(
            {
                "header": f"{r['task'].replace('nemotron_v3_', '')} [{r['language']}]",
                "subline": f"Q: {r['prompt']}",
                "caption": f"A: {final_ans}",
                "img_bytes": b[f0],
                "answer": r["answer"],
            }
        )

    return _render_contact_sheet(
        tile_specs,
        cols=3,
        rows=2,
        out_path=out_path,
        sheet_title="Stage-8 Nemotron-Image-Training-v3 Visual Primitives Verification (2x3 Overlay Sheet)",
        tile_img_size=384,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=Path("work/stage8_dataset_audit"))
    ap.add_argument("--include-nemotron", action="store_true", help="Also load and audit Nemotron-Image-Training-v3 records")
    ap.add_argument("--nemotron-cache-dir", type=Path, default=None)
    ap.add_argument("--nemotron-max-gqa", type=int, default=1200)
    ap.add_argument("--nemotron-max-geo", type=int, default=900)
    ap.add_argument("--nemotron-max-aokvqa", type=int, default=900)
    ap.add_argument("--nemotron-max-clevr", type=int, default=800)
    ap.add_argument("--nemotron-max-zhwiki", type=int, default=400)
    ap.add_argument("--nemotron-max-plotqa", type=int, default=300)
    ap.add_argument("--angles-per-asset", type=int, default=6)
    ap.add_argument("--panel-size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=20261121)
    args = ap.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")

    pairs_3d = build_shaded_3d_spatial_primitives_records(
        seed=args.seed,
        panel_size=(args.panel_size, args.panel_size),
        angles_per_asset=args.angles_per_asset,
    )
    pairs_nem: list[tuple[dict[str, Any], dict[str, bytes]]] = []
    if args.include_nemotron:
        nem_cache = args.nemotron_cache_dir or (out_dir / "nemotron_v3_cache")
        pairs_nem = build_nemotron_v3_spatial_records(
            cache_dir=nem_cache,
            hf_token=hf_token,
            max_gqa=args.nemotron_max_gqa,
            max_geometry3k=args.nemotron_max_geo,
            max_aokvqa=args.nemotron_max_aokvqa,
            max_clevr=args.nemotron_max_clevr,
            max_zhwiki=args.nemotron_max_zhwiki,
            max_plotqa=args.nemotron_max_plotqa,
            seed=args.seed + 202,
        )

    all_pairs = list(pairs_3d) + list(pairs_nem)
    report = audit_records(all_pairs, check_distribution=True, check_all_tasks=True)

    artifacts: dict[str, str] = {}
    sheet_3d = render_primitive_overlay_sheet(pairs_3d, out_dir / "primitive_overlay_samples.png")
    if sheet_3d is not None:
        artifacts["primitive_overlay_samples"] = str(sheet_3d.resolve())
    if args.include_nemotron and pairs_nem:
        sheet_nem = render_nemotron_overlay_sheet(pairs_nem, out_dir / "nemotron_overlay_samples.png")
        if sheet_nem is not None:
            artifacts["nemotron_overlay_samples"] = str(sheet_nem.resolve())

    report["config"] = {
        "include_nemotron": bool(args.include_nemotron),
        "angles_per_asset": int(args.angles_per_asset),
        "panel_size": int(args.panel_size),
        "seed": int(args.seed),
        "spatial_3d_records": len(pairs_3d),
        "nemotron_records": len(pairs_nem),
    }
    report["artifacts"] = artifacts

    summary_path = out_dir / "STAGE8_DATASET_AUDIT.json"
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
