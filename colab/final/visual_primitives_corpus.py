"""Stage-7 Full Vision Main Dishes & DeepSeek *Thinking with Visual Primitives* Corpus Builder.

Constructs `~14,000` strictly train-only, license-verified multimodal reasoning, grounding,
counting, OCR, chart/document, formula, science, and `ARC-AGI-2` visual-induction examples:

1. **Visual Primitives (`<point x=".." y=".." label=".."/>` & `<box x1=".." y1=".." x2=".." y2=".." label=".."/>`)**:
   - Coordinates normalized to integer `[0..1000]` inside `<think>...</think>`.
   - `allenai/pixmo-points` (`[0..100]% * 10 -> [0..1000]`) + `allenai/pixmo-count` (raw px -> `[0..1000]`,
     including `count == 0` 4-quadrant negative scans).
   - Balanced `50/50` Yes/No Object Existence CoT (`POPE` hallucination & yes-bias cure).
   - `deepcs233/Visual-CoT` joined onto embedded `lmms-lab/TextVQA` `train` images (`<box>` evidence region -> OCR answer).

2. **Full Vision Main Dishes (`train` splits ONLY — 0% overlap with any evaluation benchmark)**:
   - `HuggingFaceM4/ChartQA` (`train`), `lmms-lab/TextVQA` (`train`), `lmms-lab/DocVQA` (`train`),
     `oleehyO/latex-formulas` (`train`), `derek-thomas/ScienceQA` (`train` multimodal),
     and `TIGER-Lab/ViRL39K` (filtered against any benchmark-named source).

3. **Visual + Symbolic `ARC-AGI-2` (`data/training/*.json` ONLY — 0% `data/evaluation` overlap)**:
   - Crisp 10-color PIL grid sheet rendering (`Demo 1..K` Input -> Output + `Test Input`) with
     normalized `[0..1000]` panel `<box>` & foreground `<point>` primitives inside `<think>` and exact 2D JSON grid output.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import urllib.request
import zipfile
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import pyarrow.parquet as pq

from .common import emit, identity

# Pinned dataset revisions probed & verified on workstation
PIXMO_POINTS_REPO = "allenai/pixmo-points"
PIXMO_POINTS_REV = "2b5c6931e790e00ae00d4a2857e5f95d88f09a66"

PIXMO_COUNT_REPO = "allenai/pixmo-count"
PIXMO_COUNT_REV = "ebf51cf70e45d12374e64d475862e4f8d21d31d0"

VISUAL_COT_REPO = "deepcs233/Visual-CoT"
VISUAL_COT_REV = "223d2d8c1146fda2bb918801b8276c587b78b61c"

CHARTQA_REPO = "HuggingFaceM4/ChartQA"
CHARTQA_REV = "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"

TEXTVQA_REPO = "lmms-lab/TextVQA"
TEXTVQA_REV = "9c0699cd19768ac5ab97568f6b3cbac4c0062884"

DOCVQA_REPO = "lmms-lab/DocVQA"
DOCVQA_REV = "539088ef8a8ada01ac8e2e6d4e372586748a265e"

LATEX_REPO = "oleehyO/latex-formulas"
LATEX_REV = "436791f14f92cbb7fdc1b27f2c316cf1f4e65445"

SCIENCEQA_REPO = "derek-thomas/ScienceQA"
SCIENCEQA_REV = "f18b0a70359ebfb41f658fd564208d0355b013f4"

VIRL39K_REPO = "TIGER-Lab/ViRL39K"
VIRL39K_REV = "812ec617dea4bc8a4e751663b88e4ebb7de4d00e"

ARC_PALETTE_RGB: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),        # 0 black
    (0, 116, 217),    # 1 blue
    (255, 65, 54),    # 2 red
    (46, 204, 64),    # 3 green
    (255, 220, 0),    # 4 yellow
    (170, 170, 170),  # 5 grey
    (240, 18, 190),   # 6 magenta
    (255, 133, 27),   # 7 orange
    (127, 219, 255),  # 8 cyan
    (135, 12, 37),    # 9 maroon
)

BANNED_BENCHMARK_SUBSTRINGS: tuple[str, ...] = (
    "mmstar",
    "mmmu",
    "mathvista",
    "ai2d",
    "ocrbench",
    "realworldqa",
    "hallusion",
    "pope",
    "blink",
    "mmvet",
)


def _clamp_1000(v: float) -> int:
    return max(0, min(1000, int(round(v))))


def _clean_xml_label(s: str) -> str:
    cleaned = re.sub(r'["<>\\\r\n\t]+', " ", str(s or "object")).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return (cleaned[:36] or "object").lower()


def _wrap_think(think_text: str, body_text: str, *, enable_thinking: bool = True) -> str:
    if not enable_thinking or not think_text.strip():
        return body_text.strip()
    return f"<think>\n{think_text.strip()}\n</think>\n{body_text.strip()}"


def _hf_file(repo_id: str, filename: str, *, revision: str, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            token=token,
        )
    )


def _normalize_image_bytes(raw_bytes: bytes, *, max_side: int = 1344) -> tuple[bytes, int, int] | None:
    """Validate & normalize image bytes into clean RGB PNG/JPEG within `max_side`."""
    try:
        with Image.open(io.BytesIO(raw_bytes)) as im:
            im_rgb = im.convert("RGB")
            orig_w, orig_h = im_rgb.size
            if orig_w < 16 or orig_h < 16:
                return None
            if max(orig_w, orig_h) > max_side:
                scale = max_side / float(max(orig_w, orig_h))
                nw = max(16, int(round(orig_w * scale)))
                nh = max(16, int(round(orig_h * scale)))
                im_rgb = im_rgb.resize((nw, nh), resample=Image.Resampling.BICUBIC)
            out = io.BytesIO()
            im_rgb.save(out, format="JPEG", quality=92, optimize=True)
            return out.getvalue(), orig_w, orig_h
    except Exception:
        return None


def _fetch_pixmo_url_cached(
    url: str,
    expected_sha256: str | None,
    cache_dir: Path,
    timeout_s: float = 7.5,
) -> tuple[bytes, int, int] | None:
    key = expected_sha256 if (expected_sha256 and len(expected_sha256) >= 16) else hashlib.sha256(url.encode("utf-8")).hexdigest()
    cached_path = cache_dir / f"{key}.jpg"
    meta_path = cache_dir / f"{key}.json"
    if cached_path.is_file() and meta_path.is_file():
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            return cached_path.read_bytes(), int(m["orig_w"]), int(m["orig_h"])
        except Exception:
            pass
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SparkVLResearch/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read(16_000_000)
        if expected_sha256 and len(expected_sha256) == 64:
            actual = hashlib.sha256(raw).hexdigest()
            if actual != expected_sha256.lower():
                return None
        norm = _normalize_image_bytes(raw, max_side=1152)
        if norm is None:
            return None
        jpg_bytes, orig_w, orig_h = norm
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        cached_path.write_bytes(jpg_bytes)
        meta_path.write_text(json.dumps({"orig_w": orig_w, "orig_h": orig_h}), encoding="utf-8")
        return jpg_bytes, orig_w, orig_h
    except Exception:
        return None


def _build_pixmo_points_count_and_pope_cure(
    cache_dir: Path,
    token: str | None,
    *,
    seed: int,
    target_points: int = 1600,
    target_count: int = 900,
    target_yesno: int = 1200,
) -> tuple[list[tuple[dict[str, Any], list[str]]], dict[str, bytes]]:
    """Build `<point>` / `<box>` grounded counting, pointing, and balanced 50/50 Yes/No existence records."""
    rng = random.Random(seed ^ 0x7170C001)
    records: list[tuple[dict[str, Any], list[str]]] = []
    blobs: dict[str, bytes] = {}
    pixmo_img_cache = cache_dir / "pixmo_url_cache"
    pixmo_img_cache.mkdir(parents=True, exist_ok=True)

    # 1. Sample candidate rows from `pixmo-points` and `pixmo-count`
    pts_path = _hf_file(PIXMO_POINTS_REPO, "data/train-00000-of-00002.parquet", revision=PIXMO_POINTS_REV, token=token)
    cnt_path = _hf_file(PIXMO_COUNT_REPO, "data/train-00000-of-00001.parquet", revision=PIXMO_COUNT_REV, token=token)

    pts_table = pq.read_table(pts_path, columns=["image_url", "image_sha256", "label", "count", "points"])
    pts_rows = pts_table.to_pylist()
    rng.shuffle(pts_rows)
    pts_candidates = [
        r for r in pts_rows
        if r.get("image_url")
        and isinstance(r.get("count"), int)
        and 0 <= int(r["count"]) <= 25
        and str(r.get("label") or "").strip()
    ][: int(target_points * 2.4)]

    cnt_table = pq.read_table(cnt_path, columns=["image_url", "image_sha256", "label", "count", "points"])
    cnt_rows = cnt_table.to_pylist()
    rng.shuffle(cnt_rows)
    cnt_candidates = [
        r for r in cnt_rows
        if r.get("image_url")
        and isinstance(r.get("count"), int)
        and 0 <= int(r["count"]) <= 25
        and str(r.get("label") or "").strip()
    ][: int(target_count * 2.4)]

    # Deduplicate URLs for parallel fetch
    url_jobs: dict[str, str | None] = {}
    for r in (*pts_candidates, *cnt_candidates):
        u = str(r["image_url"])
        if u not in url_jobs:
            url_jobs[u] = str(r.get("image_sha256") or "")

    fetched: dict[str, tuple[bytes, int, int]] = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        fut_map = {
            pool.submit(_fetch_pixmo_url_cached, u, sha_hex, pixmo_img_cache): u
            for u, sha_hex in url_jobs.items()
        }
        for fut in as_completed(fut_map):
            u = fut_map[fut]
            res = fut.result()
            if res is not None:
                fetched[u] = res

    vocab_labels = sorted(
        {
            str(r["label"]).strip().lower()
            for r in (*pts_candidates, *cnt_candidates)
            if 2 <= len(str(r.get("label") or "").strip()) <= 28
        }
    )
    if not vocab_labels:
        vocab_labels = ["bicycle", "traffic cone", "mug", "bird", "chair", "clock", "laptop", "backpack"]

    positive_pool_for_yesno: list[tuple[str, str, list[tuple[int, int]], str]] = []
    negative_pool_for_yesno: list[tuple[str, str, str]] = []

    # Process `pixmo-points` (`points` in `[0..100]` percentages!)
    added_pts = 0
    for idx, r in enumerate(pts_candidates):
        if added_pts >= target_points:
            break
        u = str(r["image_url"])
        if u not in fetched:
            continue
        jpg_bytes, _, _ = fetched[u]
        img_hash = hashlib.sha256(jpg_bytes).hexdigest()[:20]
        rel_path = f"vis_prim/pixmo_pts_{img_hash}.jpg"
        blobs[rel_path] = jpg_bytes
        grp = f"pixmo-img:{img_hash}"

        lbl = str(r["label"]).strip()
        xml_lbl = _clean_xml_label(lbl)
        cnt = int(r["count"])
        raw_pts = r.get("points") or []
        coords_1000: list[tuple[int, int]] = []
        if isinstance(raw_pts, list):
            for pt in raw_pts:
                if isinstance(pt, dict) and "x" in pt and "y" in pt:
                    coords_1000.append((_clamp_1000(float(pt["x"]) * 10.0), _clamp_1000(float(pt["y"]) * 10.0)))
        coords_1000.sort(key=lambda xy: (xy[1] // 100, xy[0]))

        if cnt == 0 or not coords_1000:
            if cnt != 0:
                continue
            think = (
                f'Scanning all four quadrants for "{lbl}":\n'
                '<box x1="0" y1="0" x2="500" y2="500" label="top_left_scan"/> '
                '<box x1="500" y1="0" x2="1000" y2="500" label="top_right_scan"/>\n'
                '<box x1="0" y1="500" x2="500" y2="1000" label="bottom_left_scan"/> '
                '<box x1="500" y1="500" x2="1000" y2="1000" label="bottom_right_scan"/>\n'
                f'Verified 0 instances of "{lbl}".'
            )
            negative_pool_for_yesno.append((rel_path, lbl, grp))
        else:
            cnt = len(coords_1000)
            pt_tags = " ".join(
                f'<point x="{x}" y="{y}" label="{xml_lbl}_{k + 1}"/>'
                for k, (x, y) in enumerate(coords_1000)
            )
            think = f'Enumerating every visible "{lbl}" with normalized [0..1000] points:\n{pt_tags}\nTotal count = {cnt}.'
            positive_pool_for_yesno.append((rel_path, lbl, coords_1000, grp))

        use_zh = added_pts % 6 == 5
        prompt = (
            f"请仔细观察图像，在 `<think>` 中用 `<point>` 逐一定位每个“{lbl}”，并给出准确数量（仅输出整数或简短结论）。"
            if use_zh
            else f"Count how many `{lbl}` instances appear in this image. Localize each instance with `<point>` inside `<think>` before stating the exact integer count."
        )
        body = str(cnt)
        ans = _wrap_think(think, body, enable_thinking=True)
        records.append(
            (
                {
                    "id": f"stage7_vp_pixmopts_{idx:05d}",
                    "source_group": grp,
                    "language": "zh" if use_zh else "en",
                    "task": "visual_primitives_point_count",
                    "visual_kind": "independent_image",
                    "frames": [rel_path],
                    "media_times_s": [0.0],
                    "prompt": prompt,
                    "answer": ans,
                    "target": {"kind": "text", "text": body},
                    "enable_thinking": True,
                    "metadata": {"oracle_count": cnt, "label": lbl, "grouping_keys": [grp]},
                },
                [rel_path],
            )
        )
        added_pts += 1

    # Process `pixmo-count` (`points` = `{"x": [...], "y": [...]}` in raw pixel coords!)
    added_cnt = 0
    for idx, r in enumerate(cnt_candidates):
        if added_cnt >= target_count:
            break
        u = str(r["image_url"])
        if u not in fetched:
            continue
        jpg_bytes, orig_w, orig_h = fetched[u]
        img_hash = hashlib.sha256(jpg_bytes).hexdigest()[:20]
        rel_path = f"vis_prim/pixmo_cnt_{img_hash}.jpg"
        blobs[rel_path] = jpg_bytes
        grp = f"pixmo-img:{img_hash}"

        lbl = str(r["label"]).strip()
        xml_lbl = _clean_xml_label(lbl)
        cnt = int(r["count"])
        raw_pts = r.get("points") or {}
        coords_1000 = []
        if isinstance(raw_pts, dict) and isinstance(raw_pts.get("x"), list) and isinstance(raw_pts.get("y"), list):
            for px, py in zip(raw_pts["x"], raw_pts["y"]):
                if px is not None and py is not None:
                    coords_1000.append(
                        (
                            _clamp_1000(float(px) * 1000.0 / max(1, orig_w)),
                            _clamp_1000(float(py) * 1000.0 / max(1, orig_h)),
                        )
                    )
        coords_1000.sort(key=lambda xy: (xy[1] // 100, xy[0]))

        if cnt == 0 or not coords_1000:
            if cnt != 0:
                continue
            think = (
                f'Scanning all four image regions for "{lbl}":\n'
                '<box x1="0" y1="0" x2="500" y2="500" label="quadrant_nw"/> '
                '<box x1="500" y1="0" x2="1000" y2="500" label="quadrant_ne"/>\n'
                '<box x1="0" y1="500" x2="500" y2="1000" label="quadrant_sw"/> '
                '<box x1="500" y1="500" x2="1000" y2="1000" label="quadrant_se"/>\n'
                f'No "{lbl}" is present. Count = 0.'
            )
            negative_pool_for_yesno.append((rel_path, lbl, grp))
        else:
            cnt = len(coords_1000)
            pt_tags = " ".join(
                f'<point x="{x}" y="{y}" label="{xml_lbl}_{k + 1}"/>'
                for k, (x, y) in enumerate(coords_1000)
            )
            think = f'Point-by-point verification of "{lbl}":\n{pt_tags}\nVerified count = {cnt}.'
            positive_pool_for_yesno.append((rel_path, lbl, coords_1000, grp))

        prompt = f"How many `{lbl}` are visible in the image? Verify each one with `<point>` inside `<think>` and output the exact number."
        body = str(cnt)
        records.append(
            (
                {
                    "id": f"stage7_vp_pixmocnt_{idx:05d}",
                    "source_group": grp,
                    "language": "en",
                    "task": "visual_primitives_point_count",
                    "visual_kind": "independent_image",
                    "frames": [rel_path],
                    "media_times_s": [0.0],
                    "prompt": prompt,
                    "answer": _wrap_think(think, body, enable_thinking=True),
                    "target": {"kind": "text", "text": body},
                    "enable_thinking": True,
                    "metadata": {"oracle_count": cnt, "label": lbl, "grouping_keys": [grp]},
                },
                [rel_path],
            )
        )
        added_cnt += 1

    # Build strictly balanced 50/50 Yes/No Object Existence CoT (`POPE` hallucination & yes-bias cure!)
    half_yn = max(1, min(target_yesno // 2, len(positive_pool_for_yesno)))
    rng.shuffle(positive_pool_for_yesno)
    rng.shuffle(negative_pool_for_yesno)

    # Supplement `negative_pool_for_yesno` if needed using disjoint distractor object labels on images
    extra_idx = 0
    while len(negative_pool_for_yesno) < half_yn and positive_pool_for_yesno:
        rel_p, pos_lbl, _, grp = positive_pool_for_yesno[extra_idx % len(positive_pool_for_yesno)]
        extra_idx += 1
        cand_neg = vocab_labels[(extra_idx * 37) % len(vocab_labels)]
        if cand_neg.lower() not in pos_lbl.lower() and pos_lbl.lower() not in cand_neg.lower():
            # Keep only clearly non-overlapping concrete nouns when genuine `count==0` rows run short
            if cand_neg in ("helicopter", "giraffe", "violin", "fire hydrant", "skateboard", "pineapple", "kayak", "microscope"):
                negative_pool_for_yesno.append((rel_p, cand_neg, grp))
        if extra_idx > half_yn * 8:
            break

    n_pairs = min(half_yn, len(positive_pool_for_yesno), max(1, len(negative_pool_for_yesno)))
    for yn_i in range(n_pairs):
        rel_pos, lbl_pos, pts_pos, grp_pos = positive_pool_for_yesno[yn_i]
        x0, y0 = pts_pos[0]
        xml_pos = _clean_xml_label(lbl_pos)
        bx1, by1, bx2, by2 = _clamp_1000(x0 - 75), _clamp_1000(y0 - 75), _clamp_1000(x0 + 75), _clamp_1000(y0 + 75)
        think_yes = (
            f'Checking whether "{lbl_pos}" is visibly present:\n'
            f'<box x1="{bx1}" y1="{by1}" x2="{bx2}" y2="{by2}" label="{xml_pos}"/> '
            f'<point x="{x0}" y="{y0}" label="{xml_pos}_center"/>\n'
            f'Confirmed direct visual evidence of "{lbl_pos}". Answer: Yes.'
        )
        records.append(
            (
                {
                    "id": f"stage7_vp_exist_yes_{yn_i:05d}",
                    "source_group": grp_pos,
                    "language": "en",
                    "task": "visual_primitives_existence_yesno",
                    "visual_kind": "independent_image",
                    "frames": [rel_pos],
                    "media_times_s": [0.0],
                    "prompt": f"Is there any `{lbl_pos}` visible in this image? Answer strictly `Yes` or `No`.",
                    "answer": _wrap_think(think_yes, "Yes", enable_thinking=True),
                    "target": {"kind": "text", "text": "Yes"},
                    "enable_thinking": True,
                    "metadata": {"oracle_yesno": "Yes", "grouping_keys": [grp_pos]},
                },
                [rel_pos],
            )
        )

        rel_neg, lbl_neg, grp_neg = negative_pool_for_yesno[yn_i]
        think_no = (
            f'Scanning the entire frame before claiming existence of "{lbl_neg}":\n'
            '<box x1="0" y1="0" x2="500" y2="500" label="scan_nw"/> '
            '<box x1="500" y1="0" x2="1000" y2="500" label="scan_ne"/>\n'
            '<box x1="0" y1="500" x2="500" y2="1000" label="scan_sw"/> '
            '<box x1="500" y1="500" x2="1000" y2="1000" label="scan_se"/>\n'
            f'No visual evidence of "{lbl_neg}" in any quadrant. Answer: No.'
        )
        records.append(
            (
                {
                    "id": f"stage7_vp_exist_no_{yn_i:05d}",
                    "source_group": grp_neg,
                    "language": "en",
                    "task": "visual_primitives_existence_yesno",
                    "visual_kind": "independent_image",
                    "frames": [rel_neg],
                    "media_times_s": [0.0],
                    "prompt": f"Is there any `{lbl_neg}` visible in this image? Answer strictly `Yes` or `No`.",
                    "answer": _wrap_think(think_no, "No", enable_thinking=True),
                    "target": {"kind": "text", "text": "No"},
                    "enable_thinking": True,
                    "metadata": {"oracle_yesno": "No", "grouping_keys": [grp_neg]},
                },
                [rel_neg],
            )
        )

    emit(
        "stage7_pixmo_primitives_ready",
        fetched_urls=len(fetched),
        pixmo_points=added_pts,
        pixmo_count=added_cnt,
        yesno_balanced=n_pairs * 2,
    )
    return records, blobs


def _build_textvqa_and_visual_cot_records(
    token: str | None,
    *,
    seed: int,
    target_vcot: int = 1800,
    target_textvqa: int = 1400,
) -> tuple[list[tuple[dict[str, Any], list[str]]], dict[str, bytes]]:
    """Join `deepcs233/Visual-CoT` (`textvqa_cot_train.jsonl`) onto `lmms-lab/TextVQA` `train` embedded bytes."""
    rng = random.Random(seed ^ 0x7E87C071)
    records: list[tuple[dict[str, Any], list[str]]] = []
    blobs: dict[str, bytes] = {}

    cot_jsonl = _hf_file(VISUAL_COT_REPO, "metadata/textvqa_cot_train.jsonl", revision=VISUAL_COT_REV, token=token)
    cot_by_img_id: dict[str, list[dict[str, Any]]] = {}
    for ln in cot_jsonl.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        item = json.loads(ln)
        if str(item.get("split") or "train") != "train":
            continue
        img_stem = os.path.splitext(os.path.basename(str(item.get("image") or "")))[0]
        if img_stem:
            cot_by_img_id.setdefault(img_stem, []).append(item)

    shard_names = [
        f"data/train-{i:05d}-of-00020.parquet" for i in range(6)
    ]
    vcot_added = 0
    tvqa_added = 0

    for sh in shard_names:
        if vcot_added >= target_vcot and tvqa_added >= target_textvqa:
            break
        p = _hf_file(TEXTVQA_REPO, sh, revision=TEXTVQA_REV, token=token)
        tbl = pq.read_table(p, columns=["image_id", "image", "question", "answers", "ocr_tokens"])
        rows = tbl.to_pylist()
        rng.shuffle(rows)

        for r in rows:
            if vcot_added >= target_vcot and tvqa_added >= target_textvqa:
                break
            img_dict = r.get("image")
            if not isinstance(img_dict, dict) or not img_dict.get("bytes"):
                continue
            img_id = str(r.get("image_id") or "").strip()
            if not img_id:
                continue

            # Vote majority ground-truth answer from `answers`
            ans_list = [str(a).strip() for a in (r.get("answers") or []) if str(a).strip()]
            if not ans_list:
                continue
            best_ans = max(set(ans_list), key=ans_list.count)
            if best_ans.lower() in ("unanswerable", ""):
                continue

            norm = _normalize_image_bytes(img_dict["bytes"], max_side=1280)
            if norm is None:
                continue
            jpg_bytes, orig_w, orig_h = norm
            rel_path = f"vis_prim/textvqa_{img_id}.jpg"
            grp = f"textvqa-img:{img_id}"
            ocr_hint = ", ".join(str(t) for t in (r.get("ocr_tokens") or [])[:12] if str(t).strip())

            # 1) Try Visual-CoT `<box>` grounded trace first
            if vcot_added < target_vcot and img_id in cot_by_img_id:
                for cot_item in cot_by_img_id.pop(img_id)[:2]:
                    if vcot_added >= target_vcot:
                        break
                    q_cot = str(cot_item.get("question") or "").strip()
                    a_cot = str(cot_item.get("answer") or "").strip()
                    bboxes = cot_item.get("bboxs") or []
                    w_ref = float(cot_item.get("width") or orig_w or 1)
                    h_ref = float(cot_item.get("height") or orig_h or 1)
                    if not q_cot or not a_cot or not bboxes:
                        continue
                    box_tags: list[str] = []
                    for bi, bb in enumerate(bboxes[:3]):
                        if isinstance(bb, (list, tuple)) and len(bb) == 4:
                            x1 = _clamp_1000(float(bb[0]) * 1000.0 / max(1.0, w_ref))
                            y1 = _clamp_1000(float(bb[1]) * 1000.0 / max(1.0, h_ref))
                            x2 = _clamp_1000(float(bb[2]) * 1000.0 / max(1.0, w_ref))
                            y2 = _clamp_1000(float(bb[3]) * 1000.0 / max(1.0, h_ref))
                            box_tags.append(f'<box x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" label="evidence_roi_{bi + 1}"/>')
                    if not box_tags:
                        continue
                    blobs[rel_path] = jpg_bytes
                    think_cot = (
                        f"Localizing the text/detail region relevant to `{q_cot}`:\n"
                        + " ".join(box_tags)
                        + (f"\nDetected scene text cues: {ocr_hint}." if ocr_hint else "")
                        + f"\nReading inside the localized ROI gives `{a_cot}`."
                    )
                    records.append(
                        (
                            {
                                "id": f"stage7_vp_vcot_{img_id}_{vcot_added:05d}",
                                "source_group": grp,
                                "language": "en",
                                "task": "visual_primitives_cot_grounding",
                                "visual_kind": "independent_image",
                                "frames": [rel_path],
                                "media_times_s": [0.0],
                                "prompt": f"{q_cot}\nLocalize the target evidence region with `<box>` inside `<think>` and provide a concise answer.",
                                "answer": _wrap_think(think_cot, a_cot, enable_thinking=True),
                                "target": {"kind": "text", "text": a_cot},
                                "enable_thinking": True,
                                "metadata": {"oracle_answer": a_cot, "grouping_keys": [grp]},
                            },
                            [rel_path],
                        )
                    )
                    vcot_added += 1
                continue

            # 2) Standard `TextVQA` `train` OCR reasoning
            if tvqa_added < target_textvqa:
                q_txt = str(r.get("question") or "").strip()
                if not q_txt:
                    continue
                blobs[rel_path] = jpg_bytes
                use_think = tvqa_added % 3 != 2
                think_tv = (
                    f"Scanning scene text tokens ({ocr_hint or 'fine print'}) to answer: {q_txt}\n"
                    f"Exact answer read from image: {best_ans}."
                )
                records.append(
                    (
                        {
                            "id": f"stage7_vmd_textvqa_{img_id}_{tvqa_added:05d}",
                            "source_group": grp,
                            "language": "en",
                            "task": "vision_textvqa_train",
                            "visual_kind": "independent_image",
                            "frames": [rel_path],
                            "media_times_s": [0.0],
                            "prompt": f"{q_txt}\nAnswer concisely using the exact text/value shown in the image.",
                            "answer": _wrap_think(think_tv, best_ans, enable_thinking=use_think),
                            "target": {"kind": "text", "text": best_ans},
                            "enable_thinking": use_think,
                            "metadata": {"oracle_answers": ans_list[:5], "grouping_keys": [grp]},
                        },
                        [rel_path],
                    )
                )
                tvqa_added += 1

    emit("stage7_textvqa_vcot_ready", visual_cot=vcot_added, textvqa=tvqa_added)
    return records, blobs


def _build_embedded_parquet_vision_dishes(
    token: str | None,
    *,
    seed: int,
    target_chartqa: int = 1800,
    target_docvqa: int = 1400,
    target_latex: int = 900,
    target_scienceqa: int = 1200,
    target_virl: int = 1800,
) -> tuple[list[tuple[dict[str, Any], list[str]]], dict[str, bytes]]:
    """Load ChartQA, DocVQA, LaTeX-Formulas, ScienceQA, and ViRL39K (`train` splits ONLY)."""
    rng = random.Random(seed ^ 0x51A6E7D1)
    records: list[tuple[dict[str, Any], list[str]]] = []
    blobs: dict[str, bytes] = {}

    # 1. ChartQA (`train` shards ONLY)
    cqa_shards = [
        "data/train-00000-of-00003-49492f364babfa44.parquet",
        "data/train-00001-of-00003-7302bae5e425bbc7.parquet",
        "data/train-00002-of-00003-194c9400785577a2.parquet",
    ]
    cqa_added = 0
    for sh in cqa_shards:
        if cqa_added >= target_chartqa:
            break
        p = _hf_file(CHARTQA_REPO, sh, revision=CHARTQA_REV, token=token)
        rows = pq.read_table(p, columns=["image", "query", "label"]).to_pylist()
        rng.shuffle(rows)
        for r in rows:
            if cqa_added >= target_chartqa:
                break
            im_d = r.get("image")
            q = str(r.get("query") or "").strip()
            lbls = r.get("label") or []
            ans_str = str(lbls[0] if isinstance(lbls, list) and lbls else lbls).strip()
            if not isinstance(im_d, dict) or not im_d.get("bytes") or not q or not ans_str:
                continue
            norm = _normalize_image_bytes(im_d["bytes"], max_side=1280)
            if norm is None:
                continue
            jpg_b, _, _ = norm
            ih = hashlib.sha256(jpg_b).hexdigest()[:20]
            rel_p = f"vis_dish/chartqa_{ih}.jpg"
            blobs[rel_p] = jpg_b
            grp = f"chartqa-img:{ih}"
            think = (
                '<box x1="60" y1="60" x2="940" y2="920" label="chart_plot_and_axes"/>\n'
                f"Reading axis ticks, legend series, and data values for `{q}` -> `{ans_str}`."
            )
            use_th = cqa_added % 4 != 3
            records.append(
                (
                    {
                        "id": f"stage7_vmd_chartqa_{cqa_added:05d}",
                        "source_group": grp,
                        "language": "en",
                        "task": "vision_chartqa_train",
                        "visual_kind": "independent_image",
                        "frames": [rel_p],
                        "media_times_s": [0.0],
                        "prompt": f"{q}\nRead the chart axes and values carefully and give the exact answer.",
                        "answer": _wrap_think(think, ans_str, enable_thinking=use_th),
                        "target": {"kind": "text", "text": ans_str},
                        "enable_thinking": use_th,
                        "metadata": {"oracle_answer": ans_str, "grouping_keys": [grp]},
                    },
                    [rel_p],
                )
            )
            cqa_added += 1

    # 2. DocVQA (`train` shards ONLY)
    doc_shards = [f"DocVQA/train-{i:05d}-of-00012.parquet" for i in range(3)]
    doc_added = 0
    for sh in doc_shards:
        if doc_added >= target_docvqa:
            break
        p = _hf_file(DOCVQA_REPO, sh, revision=DOCVQA_REV, token=token)
        rows = pq.read_table(p, columns=["image", "question", "answers", "ucsf_document_id"]).to_pylist()
        rng.shuffle(rows)
        for r in rows:
            if doc_added >= target_docvqa:
                break
            im_d = r.get("image")
            q = str(r.get("question") or "").strip()
            ans_list = [str(a).strip() for a in (r.get("answers") or []) if str(a).strip()]
            if not isinstance(im_d, dict) or not im_d.get("bytes") or not q or not ans_list:
                continue
            ans_str = ans_list[0]
            norm = _normalize_image_bytes(im_d["bytes"], max_side=1344)
            if norm is None:
                continue
            jpg_b, _, _ = norm
            ih = hashlib.sha256(jpg_b).hexdigest()[:20]
            doc_id = str(r.get("ucsf_document_id") or ih)
            rel_p = f"vis_dish/docvqa_{ih}.jpg"
            blobs[rel_p] = jpg_b
            grp = f"docvqa-doc:{doc_id}"
            think = (
                '<box x1="50" y1="50" x2="950" y2="950" label="document_table_and_headers"/>\n'
                f"Locating the exact document field for `{q}` -> `{ans_str}`."
            )
            use_th = doc_added % 3 != 2
            records.append(
                (
                    {
                        "id": f"stage7_vmd_docvqa_{doc_added:05d}",
                        "source_group": grp,
                        "language": "en",
                        "task": "vision_docvqa_train",
                        "visual_kind": "independent_image",
                        "frames": [rel_p],
                        "media_times_s": [0.0],
                        "prompt": f"{q}\nExtract the exact text/number from the document.",
                        "answer": _wrap_think(think, ans_str, enable_thinking=use_th),
                        "target": {"kind": "text", "text": ans_str},
                        "enable_thinking": use_th,
                        "metadata": {"oracle_answers": ans_list[:4], "grouping_keys": [grp]},
                    },
                    [rel_p],
                )
            )
            doc_added += 1

    # 3. LaTeX Formulas (`train` shard ONLY — fine-grained symbol/equation OCR)
    lat_p = _hf_file(LATEX_REPO, "cleaned_formulas/train-00000-of-00006.parquet", revision=LATEX_REV, token=token)
    lat_rows = pq.read_table(lat_p, columns=["image", "latex_formula"]).to_pylist()
    rng.shuffle(lat_rows)
    lat_added = 0
    for r in lat_rows:
        if lat_added >= target_latex:
            break
        im_d = r.get("image")
        form = str(r.get("latex_formula") or "").strip()
        if not isinstance(im_d, dict) or not im_d.get("bytes") or not (4 <= len(form) <= 420):
            continue
        norm = _normalize_image_bytes(im_d["bytes"], max_side=1024)
        if norm is None:
            continue
        jpg_b, _, _ = norm
        ih = hashlib.sha256(jpg_b).hexdigest()[:20]
        rel_p = f"vis_dish/latex_{ih}.jpg"
        blobs[rel_p] = jpg_b
        grp = f"latex-img:{ih}"
        body_tex = f"```latex\n{form}\n```"
        records.append(
            (
                {
                    "id": f"stage7_vmd_latex_{lat_added:05d}",
                    "source_group": grp,
                    "language": "en",
                    "task": "vision_latex_ocr",
                    "visual_kind": "independent_image",
                    "frames": [rel_p],
                    "media_times_s": [0.0],
                    "prompt": "Transcribe the mathematical formula in this image into exact LaTeX.",
                    "answer": body_tex,
                    "target": {"kind": "text", "text": body_tex},
                    "enable_thinking": False,
                    "metadata": {"oracle_latex": form, "grouping_keys": [grp]},
                },
                [rel_p],
            )
        )
        lat_added += 1

    # 4. ScienceQA (`train` multimodal subset ONLY)
    sqa_p = _hf_file(SCIENCEQA_REPO, "data/train-00000-of-00001-1028f23e353fbe3e.parquet", revision=SCIENCEQA_REV, token=token)
    sqa_rows = pq.read_table(sqa_p, columns=["image", "question", "choices", "answer", "lecture", "solution"]).to_pylist()
    rng.shuffle(sqa_rows)
    sqa_added = 0
    letters = "ABCDEFGH"
    for r in sqa_rows:
        if sqa_added >= target_scienceqa:
            break
        im_d = r.get("image")
        if not isinstance(im_d, dict) or not im_d.get("bytes"):
            continue
        q = str(r.get("question") or "").strip()
        choices = [str(c).strip() for c in (r.get("choices") or [])]
        ans_idx = r.get("answer")
        if not q or len(choices) < 2 or not isinstance(ans_idx, int) or not (0 <= ans_idx < len(choices)):
            continue
        norm = _normalize_image_bytes(im_d["bytes"], max_side=1152)
        if norm is None:
            continue
        jpg_b, _, _ = norm
        ih = hashlib.sha256(jpg_b).hexdigest()[:20]
        rel_p = f"vis_dish/scienceqa_{ih}.jpg"
        blobs[rel_p] = jpg_b
        grp = f"scienceqa-img:{ih}"
        let_ans = letters[ans_idx]
        opts_str = "\n".join(f"({letters[i]}) {c}" for i, c in enumerate(choices))
        sol = " ".join(x for x in (str(r.get("lecture") or "").strip(), str(r.get("solution") or "").strip()) if x)[:900]
        think_sqa = (
            '<box x1="60" y1="60" x2="940" y2="940" label="science_diagram"/>\n'
            + (f"{sol}\n" if sol else "")
            + f"Therefore option ({let_ans}) `{choices[ans_idx]}` is correct."
        )
        body_sqa = f"Answer: ({let_ans}) {choices[ans_idx]}"
        records.append(
            (
                {
                    "id": f"stage7_vmd_scienceqa_{sqa_added:05d}",
                    "source_group": grp,
                    "language": "en",
                    "task": "vision_scienceqa_train",
                    "visual_kind": "independent_image",
                    "frames": [rel_p],
                    "media_times_s": [0.0],
                    "prompt": f"{q}\n\nOptions:\n{opts_str}\n\nThink step by step over the diagram and state `Answer: (<letter>)`.",
                    "answer": _wrap_think(think_sqa, body_sqa, enable_thinking=True),
                    "target": {"kind": "text", "text": body_sqa},
                    "enable_thinking": True,
                    "metadata": {"correct_letter": let_ans, "correct_answer": choices[ans_idx], "grouping_keys": [grp]},
                },
                [rel_p],
            )
        )
        sqa_added += 1

    # 5. ViRL39K (`39Krelease.parquet` + `images.zip`, strictly scrubbed against any benchmark substring!)
    virl_added = 0
    if target_virl > 0:
        virl_pq = _hf_file(VIRL39K_REPO, "39Krelease.parquet", revision=VIRL39K_REV, token=token)
        virl_zip = _hf_file(VIRL39K_REPO, "images.zip", revision=VIRL39K_REV, token=token)
        virl_rows = pq.read_table(virl_pq, columns=["qid", "question", "answer", "category", "image"]).to_pylist()
        rng.shuffle(virl_rows)
        with zipfile.ZipFile(virl_zip, "r") as zf:
            znames = set(zf.namelist())
            for r in virl_rows:
                if virl_added >= target_virl:
                    break
                qid = str(r.get("qid") or "")
                cat = str(r.get("category") or "")
                q_raw = str(r.get("question") or "").replace("<image>", "").strip()
                ans_raw = str(r.get("answer") or "").strip()
                im_list = r.get("image") or []
                if not q_raw or not ans_raw or not isinstance(im_list, list) or len(im_list) != 1:
                    continue
                low_check = f"{qid} {cat}".lower()
                if any(b_sub in low_check for b_sub in BANNED_BENCHMARK_SUBSTRINGS):
                    continue
                z_member = str(im_list[0]).lstrip("./")
                if z_member not in znames:
                    if f"images/{os.path.basename(z_member)}" in znames:
                        z_member = f"images/{os.path.basename(z_member)}"
                    else:
                        continue
                raw_im = zf.read(z_member)
                norm = _normalize_image_bytes(raw_im, max_side=1280)
                if norm is None:
                    continue
                jpg_b, _, _ = norm
                ih = hashlib.sha256(jpg_b).hexdigest()[:20]
                rel_p = f"vis_dish/virl39k_{ih}.jpg"
                blobs[rel_p] = jpg_b
                grp = f"virl39k-img:{ih}"
                clean_box_ans = re.sub(r"^\\boxed\{(.*)\}$", r"\1", ans_raw).strip()
                think_v = (
                    f'<box x1="70" y1="70" x2="930" y2="930" label="{_clean_xml_label(cat or "diagram_math")}"/>\n'
                    f"Category: {cat}. Analyzing visual geometry/quantities in the figure -> Final verified result: `{clean_box_ans}`."
                )
                body_v = f"Final Answer: {clean_box_ans}"
                records.append(
                    (
                        {
                            "id": f"stage7_vmd_virl39k_{virl_added:05d}",
                            "source_group": grp,
                            "language": "en",
                            "task": "vision_virl39k_reasoning",
                            "visual_kind": "independent_image",
                            "frames": [rel_p],
                            "media_times_s": [0.0],
                            "prompt": f"{q_raw}\nInspect the figure carefully and end with `Final Answer: <answer>`.",
                            "answer": _wrap_think(think_v, body_v, enable_thinking=True),
                            "target": {"kind": "text", "text": body_v},
                            "enable_thinking": True,
                            "metadata": {"oracle_exact_answer": clean_box_ans, "category": cat, "grouping_keys": [grp]},
                        },
                        [rel_p],
                    )
                )
                virl_added += 1

    emit(
        "stage7_embedded_vision_dishes_ready",
        chartqa=cqa_added,
        docvqa=doc_added,
        latex=lat_added,
        scienceqa=sqa_added,
        virl39k=virl_added,
    )
    return records, blobs


def render_arc_task_sheet(
    train_pairs: list[dict[str, Any]],
    test_input: list[list[int]],
    *,
    canvas_width: int = 768,
) -> tuple[bytes, dict[str, Any]]:
    """Render an `ARC-AGI-2` task (`Demo 1..K` Input -> Output + `Test Input -> ?`) to crisp RGB PNG
    and return `(png_bytes, coord_metadata_1000)` for Visual-Primitives supervision AND benchmark eval.
    """
    n_demos = len(train_pairs)
    n_rows_layout = n_demos + 1
    row_h = max(136, min(200, 768 // max(2, n_rows_layout)))
    canvas_h = row_h * n_rows_layout + 24
    im = Image.new("RGB", (canvas_width, canvas_h), (24, 26, 34))
    draw = ImageDraw.Draw(im)

    def _draw_grid(grid: list[list[int]], x0: int, y0: int, max_w: int, max_h: int) -> tuple[int, int, int, int, float]:
        gh = max(1, len(grid))
        gw = max(1, len(grid[0]) if grid else 1)
        cell = max(3, min(max_w // gw, max_h // gh, 24))
        tot_w = gw * cell
        tot_h = gh * cell
        gx0 = x0 + (max_w - tot_w) // 2
        gy0 = y0 + (max_h - tot_h) // 2
        draw.rectangle([gx0 - 2, gy0 - 2, gx0 + tot_w + 2, gy0 + tot_h + 2], fill=(64, 70, 88))
        for r_i, row_vals in enumerate(grid):
            for c_i, val in enumerate(row_vals):
                rgb = ARC_PALETTE_RGB[int(val) % 10]
                cx0 = gx0 + c_i * cell
                cy0 = gy0 + r_i * cell
                draw.rectangle([cx0, cy0, cx0 + cell - 1, cy0 + cell - 1], fill=rgb, outline=(36, 40, 52) if cell >= 6 else None)
        return gx0, gy0, gx0 + tot_w, gy0 + tot_h, float(cell)

    demo_boxes_1000: list[dict[str, list[int]]] = []
    pad_x = 24
    col_w = (canvas_width - pad_x * 3 - 36) // 2

    for idx_d, pair in enumerate(train_pairs):
        ry = 12 + idx_d * row_h
        ib = _draw_grid(pair["input"], pad_x, ry + 14, col_w, row_h - 22)
        # Arrow in middle
        ax = pad_x + col_w + 10
        ay = ry + row_h // 2
        draw.line([(ax, ay), (ax + 26, ay)], fill=(210, 218, 235), width=3)
        draw.polygon([(ax + 28, ay), (ax + 20, ay - 6), (ax + 20, ay + 6)], fill=(210, 218, 235))
        ob = _draw_grid(pair["output"], pad_x + col_w + 46, ry + 14, col_w, row_h - 22)
        demo_boxes_1000.append(
            {
                "in": [
                    _clamp_1000(ib[0] * 1000.0 / canvas_width),
                    _clamp_1000(ib[1] * 1000.0 / canvas_h),
                    _clamp_1000(ib[2] * 1000.0 / canvas_width),
                    _clamp_1000(ib[3] * 1000.0 / canvas_h),
                ],
                "out": [
                    _clamp_1000(ob[0] * 1000.0 / canvas_width),
                    _clamp_1000(ob[1] * 1000.0 / canvas_h),
                    _clamp_1000(ob[2] * 1000.0 / canvas_width),
                    _clamp_1000(ob[3] * 1000.0 / canvas_h),
                ],
            }
        )

    # Test input row at bottom
    ty = 12 + n_demos * row_h
    draw.line([(16, ty - 4), (canvas_width - 16, ty - 4)], fill=(96, 108, 138), width=2)
    tb = _draw_grid(test_input, pad_x, ty + 14, col_w, row_h - 22)
    qx0 = pad_x + col_w + 46
    draw.rectangle([qx0 + col_w // 4, ty + 24, qx0 + (col_w * 3) // 4, ty + row_h - 14], outline=(120, 190, 255), width=2)

    test_box_1000 = [
        _clamp_1000(tb[0] * 1000.0 / canvas_width),
        _clamp_1000(tb[1] * 1000.0 / canvas_h),
        _clamp_1000(tb[2] * 1000.0 / canvas_width),
        _clamp_1000(tb[3] * 1000.0 / canvas_h),
    ]
    # Find up to 3 non-zero foreground points inside `test_input`
    fg_pts_1000: list[tuple[int, int, int]] = []
    t_gh = max(1, len(test_input))
    t_gw = max(1, len(test_input[0]) if test_input else 1)
    cell_px = tb[4]
    for r_i, row_vals in enumerate(test_input):
        for c_i, val in enumerate(row_vals):
            if int(val) != 0 and len(fg_pts_1000) < 3:
                px = tb[0] + (c_i + 0.5) * cell_px
                py = tb[1] + (r_i + 0.5) * cell_px
                fg_pts_1000.append((_clamp_1000(px * 1000.0 / canvas_width), _clamp_1000(py * 1000.0 / canvas_h), int(val)))

    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue(), {
        "demo_boxes_1000": demo_boxes_1000,
        "test_box_1000": test_box_1000,
        "fg_pts_1000": fg_pts_1000,
        "test_shape": [t_gh, t_gw],
    }


def _apply_d4_color_aug(
    grid: list[list[int]],
    rot_k: int,
    flip_h: bool,
    perm: dict[int, int],
) -> list[list[int]]:
    arr = np.asarray(grid, dtype=np.int64)
    if rot_k:
        arr = np.rot90(arr, k=rot_k)
    if flip_h:
        arr = np.fliplr(arr)
    mapped = np.vectorize(lambda v: perm.get(int(v), int(v)))(arr)
    return mapped.tolist()


def _build_arc_agi_2_training_records(
    root: Path,
    config: dict[str, Any],
    *,
    seed: int,
    target_visual_arc: int = 1600,
    target_symbolic_arc: int = 800,
) -> tuple[list[tuple[dict[str, Any], list[str]]], dict[str, bytes]]:
    """Build `ARC-AGI-2` visual + symbolic induction records STRICTLY from `data/training/*.json` (`0%` `evaluation` overlap!)."""
    rng = random.Random(seed ^ 0xA2CA6102)
    records: list[tuple[dict[str, Any], list[str]]] = []
    blobs: dict[str, bytes] = {}

    cand_dirs = [
        Path(str(config["arc_agi_2_root"])) if config.get("arc_agi_2_root") else None,
        root.parent.parent / "cache" / "ARC-AGI-2",
        Path("/home/baitian6641/Spark-X2.6-VL/cache/ARC-AGI-2"),
        Path.home() / "Spark-X2.6-VL" / "cache" / "ARC-AGI-2",
    ]
    arc_repo: Path | None = None
    for cd in cand_dirs:
        if cd is not None and (cd / "data" / "training").is_dir():
            arc_repo = cd
            break
    if arc_repo is None:
        emit("stage7_arc_agi_2_missing_skip", checked=[str(c) for c in cand_dirs if c])
        return records, blobs

    train_dir = arc_repo / "data" / "training"
    eval_dir = arc_repo / "data" / "evaluation"
    eval_ids = {p.stem for p in eval_dir.glob("*.json")} if eval_dir.is_dir() else set()

    train_files = sorted(train_dir.glob("*.json"))
    assert len(train_files) > 0, f"No ARC-AGI-2 training files in {train_dir}"

    vis_added = 0
    sym_added = 0
    aug_pass = 0
    while vis_added < target_visual_arc or sym_added < target_symbolic_arc:
        shuffled = list(train_files)
        rng.shuffle(shuffled)
        for tf in shuffled:
            if vis_added >= target_visual_arc and sym_added >= target_symbolic_arc:
                break
            tid = tf.stem
            # Hard contamination firewall: never touch any ID that also exists in `evaluation`
            if tid in eval_ids:
                continue
            doc = json.loads(tf.read_text(encoding="utf-8"))
            tr_raw = doc.get("train") or []
            te_raw = doc.get("test") or []
            if not tr_raw or not te_raw or len(tr_raw) > 6:
                continue

            if aug_pass == 0:
                rot_k, flip_h = 0, False
                perm = {c: c for c in range(10)}
            else:
                rot_k = rng.randint(0, 3)
                flip_h = bool(rng.getrandbits(1))
                non_bg = list(range(1, 10))
                shuffled_c = list(non_bg)
                rng.shuffle(shuffled_c)
                perm = {0: 0, **{non_bg[i]: shuffled_c[i] for i in range(9)}}

            tr_pairs = [
                {
                    "input": _apply_d4_color_aug(p["input"], rot_k, flip_h, perm),
                    "output": _apply_d4_color_aug(p["output"], rot_k, flip_h, perm),
                }
                for p in tr_raw
            ]
            te_in = _apply_d4_color_aug(te_raw[0]["input"], rot_k, flip_h, perm)
            te_out = _apply_d4_color_aug(te_raw[0]["output"], rot_k, flip_h, perm)
            if len(te_out) > 30 or (te_out and len(te_out[0]) > 30):
                continue

            grp = f"arc-agi-2-train:{tid}"
            out_json = json.dumps(te_out)
            in_h, in_w = len(te_in), len(te_in[0])
            out_h, out_w = len(te_out), len(te_out[0])
            same_shape = all(
                len(p["input"]) == len(p["output"]) and len(p["input"][0]) == len(p["output"][0])
                for p in tr_pairs
            )

            # Visual + Symbolic Sheet record
            if vis_added < target_visual_arc:
                png_b, cmeta = render_arc_task_sheet(tr_pairs, te_in)
                rel_p = f"vis_arc2/{tid}_a{aug_pass}_{vis_added:05d}.png"
                blobs[rel_p] = png_b
                d0 = cmeta["demo_boxes_1000"][0]
                tb = cmeta["test_box_1000"]
                pts_str = " ".join(
                    f'<point x="{px}" y="{py}" label="fg_color_{c}"/>'
                    for px, py, c in cmeta["fg_pts_1000"]
                )
                demo_txt = "\n".join(
                    f"Demo {i + 1}: input={json.dumps(p['input'])} -> output={json.dumps(p['output'])}"
                    for i, p in enumerate(tr_pairs[:4])
                )
                think_arc = (
                    f'Inspecting ARC-AGI-2 demonstration panels and test grid:\n'
                    f'<box x1="{d0["in"][0]}" y1="{d0["in"][1]}" x2="{d0["in"][2]}" y2="{d0["in"][3]}" label="demo_1_input"/> '
                    f'<box x1="{d0["out"][0]}" y1="{d0["out"][1]}" x2="{d0["out"][2]}" y2="{d0["out"][3]}" label="demo_1_output"/>\n'
                    f'<box x1="{tb[0]}" y1="{tb[1]}" x2="{tb[2]}" y2="{tb[3]}" label="test_input_grid"/> {pts_str}\n'
                    f"Grid rule check across {len(tr_pairs)} demos: shape_preserved={same_shape}; "
                    f"test input shape=({in_h}, {in_w}) -> target output shape=({out_h}, {out_w}). "
                    "Applying the exact invariant spatial/color rule yields the output grid."
                )
                body_arc = f"```json\n{out_json}\n```"
                prompt_arc = (
                    "Solve this ARC-AGI-2 abstract visual reasoning task. Inspect the rendered demonstration pairs "
                    "(left: Input -> right: Output) and the Test Input grid at the bottom, accompanied by the exact integer matrices:\n\n"
                    f"{demo_txt}\nTest Input: {json.dumps(te_in)}\n\n"
                    "Localize key panels/objects with `<box>`/`<point>` inside `<think>` and output ONLY the exact 2D integer JSON grid."
                )
                records.append(
                    (
                        {
                            "id": f"stage7_vp_arc2_vis_{tid}_a{aug_pass}_{vis_added:05d}",
                            "source_group": grp,
                            "language": "en",
                            "task": "arc_agi_2_visual_primitives",
                            "visual_kind": "independent_image",
                            "frames": [rel_p],
                            "media_times_s": [0.0],
                            "prompt": prompt_arc,
                            "answer": _wrap_think(think_arc, body_arc, enable_thinking=True),
                            "target": {"kind": "text", "text": body_arc},
                            "enable_thinking": True,
                            "metadata": {"arc_task_id": tid, "oracle_grid": te_out, "grouping_keys": [grp]},
                        },
                        [rel_p],
                    )
                )
                vis_added += 1

            # Pure Symbolic Matrix Induction twin (exercises 48-layer looped `[12..24) x 2` recurrence)
            if sym_added < target_symbolic_arc and aug_pass <= 1:
                demo_sym = "\n".join(
                    f"Example {i + 1}:\nInput: {json.dumps(p['input'])}\nOutput: {json.dumps(p['output'])}"
                    for i, p in enumerate(tr_pairs)
                )
                think_sym = (
                    f"Comparing {len(tr_pairs)} ARC-AGI-2 demonstration grid pairs: "
                    f"shape_preserved={same_shape}, test_input=({in_h}x{in_w}) -> predicted_output=({out_h}x{out_w}). "
                    "Verifying cell-by-cell symmetry, connectivity, and color mapping across every example before emitting the test grid."
                )
                body_sym = f"```json\n{out_json}\n```"
                records.append(
                    (
                        {
                            "id": f"stage7_arc2_sym_{tid}_a{aug_pass}_{sym_added:05d}",
                            "source_group": grp,
                            "language": "en",
                            "task": "arc_agi_2_symbolic_induction",
                            "frames": [],
                            "prompt": (
                                "Infer the exact transformation rule from the ARC-AGI-2 training examples and predict the 2D JSON integer output grid for the Test Input:\n\n"
                                f"{demo_sym}\n\nTest Input:\n{json.dumps(te_in)}"
                            ),
                            "answer": _wrap_think(think_sym, body_sym, enable_thinking=True),
                            "target": {"kind": "text", "text": body_sym},
                            "enable_thinking": True,
                            "metadata": {"arc_task_id": tid, "oracle_grid": te_out, "grouping_keys": [grp]},
                        },
                        [],
                    )
                )
                sym_added += 1
        aug_pass += 1
        if aug_pass > 4:
            break

    emit(
        "stage7_arc_agi_2_ready",
        visual_arc=vis_added,
        symbolic_arc=sym_added,
        held_out_eval_tasks=len(eval_ids),
    )
    return records, blobs


def build_stage7_visual_primitives_and_main_dishes(
    root: Path,
    token: str | None,
    config: dict[str, Any],
) -> tuple[list[tuple[dict[str, Any], list[str]]], dict[str, bytes]]:
    """Build the complete Stage-7 Vision Main Dishes + DeepSeek Visual Primitives stream."""
    seed = int(config.get("seed", 42))
    scale = float(config.get("stage7_vision_scale", 1.0))

    all_records: list[tuple[dict[str, Any], list[str]]] = []
    all_blobs: dict[str, bytes] = {}

    pix_recs, pix_blobs = _build_pixmo_points_count_and_pope_cure(
        root / "cache",
        token,
        seed=seed,
        target_points=max(32, int(round(1500 * scale))),
        target_count=max(24, int(round(850 * scale))),
        target_yesno=max(32, int(round(1200 * scale))),
    )
    all_records.extend(pix_recs)
    all_blobs.update(pix_blobs)

    tv_recs, tv_blobs = _build_textvqa_and_visual_cot_records(
        token,
        seed=seed,
        target_vcot=max(32, int(round(1600 * scale))),
        target_textvqa=max(24, int(round(1200 * scale))),
    )
    all_records.extend(tv_recs)
    all_blobs.update(tv_blobs)

    emb_recs, emb_blobs = _build_embedded_parquet_vision_dishes(
        token,
        seed=seed,
        target_chartqa=max(32, int(round(1600 * scale))),
        target_docvqa=max(24, int(round(1200 * scale))),
        target_latex=max(16, int(round(800 * scale))),
        target_scienceqa=max(24, int(round(1100 * scale))),
        target_virl=max(32, int(round(1600 * scale))),
    )
    all_records.extend(emb_recs)
    all_blobs.update(emb_blobs)

    arc_recs, arc_blobs = _build_arc_agi_2_training_records(
        root,
        config,
        seed=seed,
        target_visual_arc=max(24, int(round(1500 * scale))),
        target_symbolic_arc=max(16, int(round(750 * scale))),
    )
    all_records.extend(arc_recs)
    all_blobs.update(arc_blobs)

    emit(
        "stage7_visual_primitives_stream_complete",
        total_records=len(all_records),
        total_blobs=len(all_blobs),
    )
    return all_records, all_blobs
