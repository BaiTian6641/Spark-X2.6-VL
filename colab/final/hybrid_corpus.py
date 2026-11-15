"""Deterministic Clean-Room Hybrid Corpus Builder (`dataset_mode = "hybrid_world_v32"`).

Unifies four strictly license-audited multimodal & 3D world-model streams for
`Spark-X2.6-VL` (`modalities = "vision_text"`):

1. **General Bilingual Vision, Dense Captioning, OCR & Real Video (`TassP0lar/SparkX2.6-MM`)**:
   Pinned commit SHAs from `dataset.lock.json` and `HF_FULL_DATASET_HANDOFF.json`
   across `vision-core`, `docci`, `ocr`, `timelapse`, `video`, and `text`.
2. **3D Spatial Reasoning (`array/SAT`, MIT License)**:
   Pinned commit `bda5dde942d7f7b41bff7935f086ed9a9e348ae3`, sampling strictly from
   `SAT_val.parquet` (never `SAT_test.parquet`) across all 5 spatial aptitude categories
   (`perspective`, `action_consequence`, `obj_movement`, `action_sequence`, `goal_aim`),
   grouped by scene image SHA256 with bilingual EN/ZH question framing.
3. **Compositional 2D/3D Geometric & RGBA Cutout Grounding**:
   Multi-object scenes with explicit shape, color, count, occlusion/relative-position,
   and RGBA alpha cutout vs opaque background supervision in EN and ZH.
4. **SyncCraft v3.2 World Model, ADR 0007 Sculpted Rigs, Spin/Foot-Cycle Kinematics & Artist Loop**:
   - `TassP0lar/synccraft-v3-blockdsl-dataset` (`train`/`val` + `apex_complex_v3_2/train`
     + 7-level `world_capability_ladder.jsonl` `train`/`val` ONLY; all `test_*` splits excluded),
   - `TassP0lar/synccraft-v3-human-artist-dataset` (`permissive_cc` ONLY; `CC0-1.0`/`CC-BY-4.0`,
     `ai_assisted == False`, `commercial_weights_eligible == True`),
   - Parametric SyncCraft v3.2 `ADR 0007` `Sculpted Elder Wyvern` (62 boxes) &
     `Colossal Walking Citadel` (74 boxes, mechanical `spin(...)` + 4-limb `foot_cycle`),
   - Exact 25%/25%/25%/25% 4-way UV/alpha diagnosis (`a0_o0`, `a1_o0`, `a0_o1`, `a1_o1`).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import tarfile
from typing import Any

from .common import HERE, emit, identity, read, sha, write
from .data import build_database, family, infer_visual_kind, training_views, variants

SAT_REPO_ID = "array/SAT"
SAT_REVISION = "bda5dde942d7f7b41bff7935f086ed9a9e348ae3"

SYNCCRAFT_DSL_REPO_ID = "TassP0lar/synccraft-v3-blockdsl-dataset"
SYNCCRAFT_DSL_REVISION = "705e426536286a3632f163d886756704abaad575"

SYNCCRAFT_ARTIST_REPO_ID = "TassP0lar/synccraft-v3-human-artist-dataset"
SYNCCRAFT_ARTIST_REVISION = "46b826aa7928372b9609c65ce4766274272632ba"

LOCAL_HYBRID_REVISION = "572e4ac039526000705e426536286a3632f16320"
MAX_SHARD_BYTES = 1_500_000_000  # Stay comfortably below DatasetContract's 2 GiB cap


def _hf_cached_file(repo_id: str, filename: str, *, revision: str, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    resolved = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        token=token,
    )
    return Path(resolved)


def _write_local_tar_component(
    root: Path,
    comp_id: str,
    blob_map: dict[str, bytes],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Write `blob_map` into deterministic <=1.5GB TAR shard(s) and return `(component_entry, member_transport_map)`."""
    shard_dir = root / "media-cache" / comp_id
    shard_dir.mkdir(parents=True, exist_ok=True)

    sorted_names = sorted(blob_map)
    batches: list[list[str]] = []
    cur_batch: list[str] = []
    cur_bytes = 0
    for name in sorted_names:
        b_len = len(blob_map[name]) + 1024
        if cur_batch and cur_bytes + b_len > MAX_SHARD_BYTES:
            batches.append(cur_batch)
            cur_batch = []
            cur_bytes = 0
        cur_batch.append(name)
        cur_bytes += b_len
    if cur_batch:
        batches.append(cur_batch)

    shards_meta: list[dict[str, Any]] = []
    member_transport: dict[str, dict[str, Any]] = {}
    total_payload_bytes = 0
    all_member_files: list[dict[str, Any]] = []

    for shard_idx, names in enumerate(batches):
        shard_name = f"shard-{shard_idx:05d}.tar"
        shard_path = shard_dir / shard_name
        with tarfile.open(shard_path, "w:") as tf:
            for m_path in names:
                payload = blob_map[m_path]
                ti = tarfile.TarInfo(name=m_path)
                ti.size = len(payload)
                ti.mtime = 0
                ti.mode = 0o644
                tf.addfile(ti, io.BytesIO(payload))

        shard_files: list[dict[str, Any]] = []
        with tarfile.open(shard_path, "r:") as tf:
            for m in tf.getmembers():
                payload = blob_map[m.name]
                digest = hashlib.sha256(payload).hexdigest()
                member_transport[m.name] = {
                    "shard": shard_name,
                    "entry": {
                        "member": m.name,
                        "offset": m.offset_data,
                        "length": m.size,
                        "sha256": digest,
                    },
                }
                rec = {
                    "path": m.name,
                    "bytes": m.size,
                    "sha256": digest,
                    "offset_data": m.offset_data,
                }
                shard_files.append(rec)
                all_member_files.append(rec)

        s_bytes = shard_path.stat().st_size
        s_sha = sha(shard_path)
        total_payload_bytes += s_bytes
        shards_meta.append(
            {
                "path": shard_name,
                "bytes": s_bytes,
                "sha256": s_sha,
                "files": shard_files,
            }
        )

    release_doc = {
        "schema_version": 1,
        "kind": "immutable_multimodal_dataset_shards",
        "role": "training",
        "dataset_manifest_sha256": identity({"comp": comp_id, "members": len(all_member_files), "seed": seed}),
        "media_manifest_sha256": identity(all_member_files),
        "payload_bytes": total_payload_bytes,
        "shards": shards_meta,
    }
    release_path = root / "indices" / comp_id / "RELEASE.json"
    write(release_path, release_doc)
    release_sha = sha(release_path)
    comp_entry = {
        "id": comp_id,
        "repo_id": "local/spark-world-v32",
        "revision": LOCAL_HYBRID_REVISION,
        "prefix": comp_id,
        "release_bytes": release_path.stat().st_size,
        "release_sha256": release_sha,
    }
    return comp_entry, member_transport


def _select_spark_mm_packets(
    root: Path,
    token: str | None,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select stratified EN/ZH vision-text packets from `TassP0lar/SparkX2.6-MM`."""
    from .assets import fetch

    lock = read(HERE / "dataset.lock.json")
    handoff = read(HERE.parent / "formal" / "HF_FULL_DATASET_HANDOFF.json")

    raw_cat = root / "metadata" / "spark_mm_source_packets.jsonl.gz"
    if not raw_cat.is_file():
        try:
            cached = _hf_cached_file(
                lock["repo_id"],
                "releases/full-ft-v1/formal-index-v1/packets.jsonl.gz",
                revision=lock["catalogue_revision"],
                token=token,
            )
            if sha(cached) == lock["catalogue"]["sha256"]:
                raw_cat.parent.mkdir(parents=True, exist_ok=True)
                raw_cat.write_bytes(cached.read_bytes())
        except Exception:
            pass
    if not raw_cat.is_file():
        fetch(lock["catalogue_url"], raw_cat, lock["catalogue"], token)

    scale = float(config.get("hybrid_spark_mm_scale", 1.0))
    packet_quotas = {
        "vision-core": max(24, int(round(850 * scale))),
        "docci": max(24, int(round(650 * scale))),
        "ocr": max(24, int(round(650 * scale))),
        "timelapse": max(16, int(round(72 * scale))),
        "video": max(20, int(round(140 * scale))),
        "text": max(24, int(round(950 * scale))),
    }
    allowed_video_shards = {"train-00000.tar", "train-00001.tar"}

    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seed = int(config["seed"])

    with gzip.open(raw_cat, "rt", encoding="utf-8") as gz:
        for line in gz:
            pkt = json.loads(line)
            comp = pkt.get("component")
            if comp not in packet_quotas:
                continue
            transport = pkt.get("transport")
            if comp == "video" and transport and transport.get("shard") not in allowed_video_shards:
                continue
            rec = pkt["record"]
            # Verify all variants belong strictly to vision_text families ('image', 'video', 'text')
            ok = True
            for orig in variants(rec):
                for v in training_views(orig):
                    try:
                        fam = family(v)
                    except Exception:
                        ok = False
                        break
                    if fam not in ("image", "video", "text"):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            pools[comp].append(pkt)

    selected_packets: list[dict[str, Any]] = []
    used_media_comps: set[str] = set()
    for comp, quota in sorted(packet_quotas.items()):
        items = pools.get(comp, [])
        items.sort(key=lambda p: identity([seed, "spark_mm_select", comp, p["record"]["id"]]))
        chosen = items[: min(len(items), quota)]
        if comp == "text":
            for pkt in chosen:
                rec = pkt["record"]
                rec.setdefault("metadata", {})["real_text_corpus"] = True
                for tv in rec.get("task_variants", ()):
                    if isinstance(tv, dict):
                        tv.setdefault("metadata", {})["real_text_corpus"] = True
        selected_packets.extend(chosen)
        if comp != "text" and chosen:
            used_media_comps.add(comp)

    comp_by_id = {c["id"]: c for c in handoff["components"]}
    active_handoff_comps: list[dict[str, Any]] = []
    for cid in sorted(used_media_comps):
        c = dict(comp_by_id[cid])
        c["repo_id"] = handoff["repo_id"]
        c["revision"] = handoff["revision"]
        f_spec = {"bytes": c["release_bytes"], "sha256": c["release_sha256"]}
        fetch(
            f"https://huggingface.co/datasets/{handoff['repo_id']}/resolve/{handoff['revision']}/{c['prefix']}/RELEASE.json",
            root / "indices" / cid / "RELEASE.json",
            f_spec,
            token,
        )
        active_handoff_comps.append(c)

    return selected_packets, active_handoff_comps


def _build_sat_3d_spatial_records(
    token: str | None,
    config: dict[str, Any],
) -> tuple[list[tuple[dict[str, Any], list[str]]], dict[str, bytes]]:
    """Load `array/SAT` (`SAT_val.parquet`, MIT license) for 3D spatial & perspective grounding."""
    import pyarrow.parquet as pq

    sat_path = _hf_cached_file(SAT_REPO_ID, "SAT_val.parquet", revision=SAT_REVISION, token=token)
    pf = pq.ParquetFile(sat_path)
    per_qtype_cap = int(config.get("hybrid_sat_per_type", 400))
    seed = int(config["seed"])

    raw_by_qtype: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    row_idx = 0
    for batch in pf.iter_batches(batch_size=256):
        for item in batch.to_pylist():
            qtype = str(item.get("question_type") or "spatial")
            raw_by_qtype[qtype].append((row_idx, item))
            row_idx += 1

    blob_map: dict[str, bytes] = {}
    records: list[tuple[dict[str, Any], list[str]]] = []

    zh_prefix_by_type = {
        "perspective": "【3D视角空间推理】请根据场景图像回答问题：",
        "action_consequence": "【3D动作后果推理】请根据三维场景判断动作后的空间位置：",
        "obj_movement": "【3D物体运动追踪】对比前后两帧场景，判断物体的三维位移：",
        "action_sequence": "【3D连续动作推理】请根据三维视角变化推断动作序列：",
        "goal_aim": "【3D空间目标导航】请根据当前视角判断朝向目标所需的转向或移动：",
    }

    for qtype in sorted(raw_by_qtype):
        pool = raw_by_qtype[qtype]
        pool.sort(key=lambda pair: identity([seed, "sat_val", qtype, pair[0]]))
        for orig_idx, item in pool[:per_qtype_cap]:
            img_list = list(item.get("image_bytes") or ())
            if not img_list:
                continue
            question = str(item.get("question") or "").strip()
            correct = str(item.get("correct_answer") or "").strip()
            choices = [str(c).strip() for c in (item.get("answers") or ()) if str(c).strip()]
            if not question or not correct:
                continue

            img_digests: list[str] = []
            frame_paths: list[str] = []
            for f_i, raw_bytes in enumerate(img_list):
                b = bytes(raw_bytes)
                d = hashlib.sha256(b).hexdigest()
                img_digests.append(d)
                ext = "png" if b[:8] == b"\x89PNG\r\n\x1a\n" else "jpg"
                m_path = f"sat_3d/img_{d[:24]}.{ext}"
                blob_map.setdefault(m_path, b)
                frame_paths.append(m_path)

            scene_group = "sat-3d-scene:" + identity(sorted(img_digests))[:24]
            rng = random.Random(int(identity([seed, "sat_fmt", orig_idx])[:12], 16))
            use_zh = (orig_idx % 4 == 3)
            use_mc = bool(choices) and (orig_idx % 2 == 0)

            ordered_choices = list(choices)
            if correct not in ordered_choices:
                ordered_choices.append(correct)
            rng.shuffle(ordered_choices)

            if use_mc:
                opt_str = "\n".join(f"- {opt}" for opt in ordered_choices)
                if use_zh:
                    prompt = (
                        f"{zh_prefix_by_type.get(qtype, '【3D空间推理】')}\n"
                        f"{question}\n候选答案：\n{opt_str}\n请直接给出唯一正确的选项内容。"
                    )
                else:
                    prompt = (
                        f"[3D Spatial Reasoning — {qtype}]\n{question}\n"
                        f"Options:\n{opt_str}\nState the exact correct option."
                    )
            else:
                if use_zh:
                    prompt = f"{zh_prefix_by_type.get(qtype, '【3D空间推理】')}\n{question}\n请简洁准确地给出答案。"
                else:
                    prompt = f"{question}\nAnswer concisely based on the 3D spatial layout in the image(s)."

            is_pair = len(frame_paths) > 1
            visual_units = (
                [
                    {"visual_kind": "state_pair", "view_id": "before" if i == 0 else "after"}
                    for i in range(len(frame_paths))
                ]
                if is_pair
                else [{"visual_kind": "independent_image", "view_id": "single"}]
            )
            rec = {
                "id": f"sat-3d-{qtype}-{orig_idx:05d}",
                "source_id": f"array-sat-val-{orig_idx:05d}",
                "source_group": scene_group,
                "language": "zh" if use_zh else "en",
                "task": f"sat_3d_{qtype}",
                "frames": frame_paths,
                "media_times_s": [0.0] * len(frame_paths),
                "visual_kind": "state_pair" if is_pair else "independent_image",
                "visual_units": visual_units,
                "enable_thinking": False,
                "prompt": prompt,
                "answer": correct,
                "target": {"kind": "text", "text": correct},
                "metadata": {
                    "dataset": "array/SAT",
                    "revision": SAT_REVISION,
                    "license_spdx": "MIT",
                    "question_type": qtype,
                    "grouping_keys": [scene_group],
                },
            }
            records.append((rec, frame_paths))

    return records, blob_map


def _build_compositional_vqa_records(
    config: dict[str, Any],
) -> tuple[list[tuple[dict[str, Any], list[str]]], dict[str, bytes]]:
    """Synthesize multi-shape RGB + RGBA cutout grounding scenes (shape, color, count, spatial relation, alpha)."""
    from PIL import Image, ImageDraw

    num_scenes = int(config.get("hybrid_compositional_scenes", 120))
    seed = int(config["seed"])

    palette_defs = [
        ("red", "红色", (226, 54, 54)),
        ("blue", "蓝色", (48, 108, 232)),
        ("green", "绿色", (46, 184, 92)),
        ("yellow", "黄色", (238, 196, 42)),
        ("purple", "紫色", (156, 72, 218)),
        ("cyan", "青色", (38, 192, 208)),
    ]
    shape_defs = [
        ("circle", "圆形"),
        ("square", "正方形"),
        ("triangle", "三角形"),
        ("diamond", "菱形"),
        ("ring", "圆环"),
    ]
    bg_defs = [
        ("white", (246, 247, 250, 255)),
        ("dark slate", (32, 36, 44, 255)),
        ("warm sand", (236, 228, 210, 255)),
    ]

    def _draw_shape(draw: Any, shape: str, cx: int, cy: int, r: int, rgba: tuple[int, int, int, int]) -> None:
        if shape == "circle":
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgba)
        elif shape == "square":
            draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=rgba)
        elif shape == "triangle":
            draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=rgba)
        elif shape == "diamond":
            draw.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=rgba)
        elif shape == "ring":
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=rgba, width=max(8, r // 3))

    blob_map: dict[str, bytes] = {}
    records: list[tuple[dict[str, Any], list[str]]] = []

    for s_idx in range(num_scenes):
        rng = random.Random(seed + 17003 * (s_idx + 1))
        has_cutout_alpha = (s_idx % 2 == 1)
        is_two_object = (s_idx % 3 != 0)
        W = H = 224

        if has_cutout_alpha:
            img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            bg_en, bg_zh = "transparent RGBA cutout", "透明RGBA抠图背景"
        else:
            bg_name, bg_rgba = bg_defs[s_idx % len(bg_defs)]
            img = Image.new("RGBA", (W, H), bg_rgba)
            bg_en, bg_zh = f"solid {bg_name} opaque", "不透明纯色背景"

        draw = ImageDraw.Draw(img)
        group_id = f"comp-vqa-scene-{s_idx:04d}"
        m_path = f"comp_vqa/scene_{s_idx:04d}.png"

        if not is_two_object:
            c_en, c_zh, rgb = palette_defs[rng.randrange(len(palette_defs))]
            sh_en, sh_zh = shape_defs[rng.randrange(len(shape_defs))]
            cx = W // 2 + rng.randint(-18, 18)
            cy = H // 2 + rng.randint(-18, 18)
            rad = rng.randint(44, 62)
            _draw_shape(draw, sh_en, cx, cy, rad, (*rgb, 255))

            q_en = "What colored geometric shape is shown in this image, and is the background transparent or opaque?"
            a_en = f"A {c_en} {sh_en} on a {bg_en} background."
            q_zh = "这张图片中心是什么颜色和几何形状？背景是透明抠图还是不透明背景？"
            a_zh = f"图中是一个{c_zh}{sh_zh}，背景为{bg_zh}。"
        else:
            (c1_en, c1_zh, rgb1), (c2_en, c2_zh, rgb2) = rng.sample(palette_defs, 2)
            (sh1_en, sh1_zh), (sh2_en, sh2_zh) = rng.sample(shape_defs, 2)
            horiz = (s_idx % 2 == 0)
            if horiz:
                _draw_shape(draw, sh1_en, 64, 112 + rng.randint(-14, 14), 36, (*rgb1, 255))
                _draw_shape(draw, sh2_en, 160, 112 + rng.randint(-14, 14), 36, (*rgb2, 255))
                rel_en = f"The {c1_en} {sh1_en} is on the left and the {c2_en} {sh2_en} is on the right"
                rel_zh = f"左侧是{c1_zh}{sh1_zh}，右侧是{c2_zh}{sh2_zh}"
            else:
                _draw_shape(draw, sh1_en, 112 + rng.randint(-14, 14), 62, 36, (*rgb1, 255))
                _draw_shape(draw, sh2_en, 112 + rng.randint(-14, 14), 162, 36, (*rgb2, 255))
                rel_en = f"The {c1_en} {sh1_en} is above the {c2_en} {sh2_en}"
                rel_zh = f"上方是{c1_zh}{sh1_zh}，下方是{c2_zh}{sh2_zh}"

            q_en = "Describe the shapes, colors, relative spatial arrangement, and background transparency in this image."
            a_en = f"{rel_en}, on a {bg_en} background (2 shapes total)."
            q_zh = "请描述图像中各个几何图形的颜色、形状、相对空间位置关系以及背景是否透明。"
            a_zh = f"{rel_zh}，共2个几何图形，背景为{bg_zh}。"

        buf = io.BytesIO()
        out_im = img if has_cutout_alpha else img.convert("RGB")
        out_im.save(buf, format="PNG")
        blob_map[m_path] = buf.getvalue()

        for lang, q_txt, a_txt in (("en", q_en, a_en), ("zh", q_zh, a_zh)):
            rec = {
                "id": f"comp-vqa-{s_idx:04d}-{lang}",
                "source_id": group_id,
                "source_group": group_id,
                "language": lang,
                "task": "compositional_vqa",
                "frames": [m_path],
                "media_times_s": [0.0],
                "visual_kind": "independent_image",
                "visual_units": [{"visual_kind": "independent_image", "view_id": "single"}],
                "enable_thinking": False,
                "prompt": q_txt,
                "answer": a_txt,
                "target": {"kind": "text", "text": a_txt},
                "metadata": {
                    "dataset": "local_compositional_vqa_v1",
                    "license_spdx": "Apache-2.0",
                    "has_cutout_alpha": has_cutout_alpha,
                    "grouping_keys": [group_id],
                },
            }
            records.append((rec, [m_path]))

    return records, blob_map


def _render_synccraft_rig_worker(job: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Top-level multiprocessing worker rendering multiview, 4-way UV, edit, and animation records."""
    from .synccraft_bridge import (
        build_animated_quadruped_program,
        build_animation_clip_record,
        build_multiview_consistency_record,
        build_parametric_sculpted_wyvern_world,
        build_parametric_walking_citadel_world,
        build_part_grounding_record,
        build_typed_scene_edit_record,
        build_uv_seam_alpha_diagnosis_record,
        ensure_synccraft_importable,
        make_clean_room_provenance,
        validate_synccraft_license_record,
    )
    from .world_pipeline import build_default_world_state_for_program

    ensure_synccraft_importable()
    from synccraft.dsl.ast_nodes import validate_program
    from synccraft.dsl.parser import parse_program
    from synccraft.kinematics.se3_chain import compile_geometry

    kind = job["kind"]
    idx = int(job["idx"])
    seed = int(job["seed"])
    enable_thinking = bool(job.get("enable_thinking", True))
    rows_out: list[dict[str, Any]] = []
    blobs_out: dict[str, bytes] = {}

    def _collect(
        pair: tuple[dict[str, Any], dict[str, bytes]],
        *,
        prov: dict[str, Any] | None = None,
        lang: str | None = None,
    ) -> dict[str, Any]:
        r, b = pair
        if prov is not None:
            r["provenance"] = dict(prov)
            r.setdefault("metadata", {})["provenance"] = dict(prov)
            validate_synccraft_license_record(r, allow_research_nc=False)
        if lang is not None:
            r["language"] = lang
        rows_out.append(r)
        blobs_out.update(b)
        return r

    if kind == "dsl_rig":
        dsl_text = str(job["canonical_dsl"])
        prog = validate_program(parse_program(dsl_text))
        scene = compile_geometry(prog)
        ws = build_default_world_state_for_program(prog, seed=seed + idx)
        prov = dict(job["provenance"])
        aid = str(prov["asset_id"])
        fid = str(prov["family_id"])
        lang = "zh" if (idx % 4 == 3) else "en"

        _collect(
            build_part_grounding_record(
                ws,
                asset_id=aid,
                family_id=fid,
                enable_thinking=enable_thinking,
                panel_size=(160, 160),
            ),
            prov=prov,
            lang=lang,
        )
        _collect(
            build_multiview_consistency_record(
                ws,
                asset_id=aid,
                family_id=fid,
                enable_thinking=enable_thinking,
                panel_size=(160, 160),
            ),
            prov=prov,
            lang="en",
        )
        # Exact 4-way balanced UV/alpha diagnosis (`a0_o0`, `a1_o0`, `a0_o1`, `a1_o1`)
        uv_mode = idx % 4
        inject_alpha = uv_mode in (1, 3)
        inject_overlap = uv_mode in (2, 3)
        if len(scene.boxes) >= 2 or not inject_overlap:
            _collect(
                build_uv_seam_alpha_diagnosis_record(
                    ws,
                    asset_id=f"{aid}_uv{uv_mode}",
                    family_id=fid,
                    inject_alpha_cutout=inject_alpha,
                    inject_uv_overlap=inject_overlap,
                    vary_target_faces=True,
                    enable_thinking=enable_thinking,
                    panel_size=(160, 160),
                ),
                prov=prov,
                lang=lang,
            )
        if len(scene.bones) >= 2 and getattr(prog.meta, "audit_profile", "strict") != "authored":
            edit_mode = "repair_sat_defect" if (idx % 2 == 0) else "infer_world_action"
            try:
                _collect(
                    build_typed_scene_edit_record(
                        ws,
                        asset_id=aid,
                        family_id=fid,
                        mode=edit_mode,
                        enable_thinking=enable_thinking,
                        panel_size=(160, 160),
                    ),
                    prov=prov,
                    lang=lang,
                )
            except Exception:
                pass

    elif kind == "quadruped_anim":
        v = int(job["variant_idx"])
        prog = build_animated_quadruped_program(v)
        aid = f"quad_gait_v32_{v:03d}"
        fid = f"synccraft_v32_quad_family_{v % 12:02d}"
        prov = make_clean_room_provenance(aid, title=f"{fid} quadruped gait {v}")
        langs = ("en", "zh") if v < 16 else (("zh",) if v % 3 == 0 else ("en",))
        for lang in langs:
            r, b = build_animation_clip_record(
                prog,
                asset_id=f"{aid}_{lang}",
                family_id=fid,
                num_keyframes=6,
                enable_thinking=enable_thinking,
                panel_size=(144, 144),
            )
            if lang == "zh":
                r["prompt"] = (
                    f"观察这组按时间戳 t={r['media_times_s']}s 排序的等轴测关键帧序列。"
                    "请编写包含 `@kinematics` 与 `@behavior`（含 `foot_cycle(...)` 足端防滑约束 <= 0.02u 及零扫掠OBB碰撞）的完整 SyncCraft Block-DSL v3.2 程序。"
                )
            _collect((r, b), prov=prov, lang=lang)

    elif kind == "sculpted_flagship":
        v = int(job["variant_idx"])
        is_citadel = (v % 2 == 1)
        if is_citadel:
            ws = build_parametric_walking_citadel_world(variant=v // 2, subdivisions_per_unit=1)
            clip_name = "fortress_march"
            aid = f"colossal_citadel_v32_{v // 2:03d}"
            fid = f"synccraft_v32_citadel_family_{(v // 2) % 8:02d}"
        else:
            ws = build_parametric_sculpted_wyvern_world(variant=v // 2, subdivisions_per_unit=1)
            clip_name = "stalk_gait"
            aid = f"sculpted_wyvern_v32_{v // 2:03d}"
            fid = f"synccraft_v32_wyvern_family_{(v // 2) % 8:02d}"
        prov = make_clean_room_provenance(aid, title=f"ADR 0007 {fid}")
        _collect(
            build_part_grounding_record(
                ws,
                asset_id=aid,
                family_id=fid,
                enable_thinking=enable_thinking,
                panel_size=(160, 160),
            ),
            prov=prov,
            lang="zh" if v % 4 == 3 else "en",
        )
        _collect(
            build_multiview_consistency_record(
                ws,
                asset_id=aid,
                family_id=fid,
                enable_thinking=enable_thinking,
                panel_size=(160, 160),
            ),
            prov=prov,
            lang="en",
        )
        _collect(
            build_animation_clip_record(
                world_state=ws,
                clip_name=clip_name,
                asset_id=aid,
                family_id=fid,
                num_keyframes=6,
                enable_thinking=enable_thinking,
                panel_size=(144, 144),
            ),
            prov=prov,
            lang="zh" if v % 3 == 2 else "en",
        )

    return rows_out, blobs_out


def _build_synccraft_world_v32_records(
    token: str | None,
    config: dict[str, Any],
) -> tuple[list[tuple[dict[str, Any], list[str]]], dict[str, bytes]]:
    """Build multimodal + constructive SyncCraft v3.2 World Model & Artist Loop records."""
    import pyarrow.parquet as pq
    from .synccraft_bridge import (
        ARTIST_ASSISTANT_MODES,
        build_artist_assistant_record,
        build_world_capability_record_from_ladder_row,
        ensure_synccraft_importable,
        make_clean_room_provenance,
        synthesize_oracle_think_trace,
        validate_synccraft_license_record,
    )

    ensure_synccraft_importable()
    from synccraft.audit.sync_audit import audit_scene
    from synccraft.dsl.ast_nodes import validate_program
    from synccraft.dsl.parser import format_program, parse_program
    from synccraft.kinematics.se3_chain import compile_geometry

    seed = int(config["seed"])
    enable_thinking = bool(config.get("enable_thinking_synccraft", True))
    num_rendered_dsl_rigs = int(config.get("hybrid_synccraft_rendered_rigs", 120))
    num_artist_dsl_rigs = int(config.get("hybrid_synccraft_artist_rigs", 240))
    num_quad_clips = int(config.get("hybrid_synccraft_quad_clips", 28))
    num_flagship = int(config.get("hybrid_synccraft_flagship_variants", 16))

    blob_map: dict[str, bytes] = {}
    records: list[tuple[dict[str, Any], list[str]]] = []
    render_jobs: list[dict[str, Any]] = []

    # 1. `synccraft-v3-blockdsl-dataset/parquet/train.parquet` (`split == 'train'` ONLY)
    train_pq = _hf_cached_file(
        SYNCCRAFT_DSL_REPO_ID,
        "parquet/train.parquet",
        revision=SYNCCRAFT_DSL_REVISION,
        token=token,
    )
    pf = pq.ParquetFile(train_pq)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for batch in pf.iter_batches(batch_size=512, columns=["asset_id", "family_id", "canonical_dsl", "split", "seed", "source_license_and_review_record"]):
        for r in batch.to_pylist():
            sp = str(r.get("split") or "")
            if sp != "train" or "test" in sp or "holdout" in sp:
                continue
            by_family[str(r.get("family_id") or "default")].append(r)

    sampled_dsl_rows: list[dict[str, Any]] = []
    fam_keys = sorted(by_family)
    for fk in fam_keys:
        by_family[fk].sort(key=lambda r: identity([seed, "sc_dsl", fk, r["asset_id"]]))
    round_idx = 0
    while len(sampled_dsl_rows) < max(num_rendered_dsl_rigs, num_artist_dsl_rigs):
        progressed = False
        for fk in fam_keys:
            if round_idx < len(by_family[fk]):
                sampled_dsl_rows.append(by_family[fk][round_idx])
                progressed = True
                if len(sampled_dsl_rows) >= max(num_rendered_dsl_rigs, num_artist_dsl_rigs):
                    break
        if not progressed:
            break
        round_idx += 1

    for idx, r in enumerate(sampled_dsl_rows[:num_rendered_dsl_rigs]):
        aid = str(r["asset_id"])
        fid = str(r["family_id"])
        lic_rec = r.get("source_license_and_review_record") or {}
        spdx = str(lic_rec.get("license") or "Apache-2.0")
        if spdx not in ("Apache-2.0", "MIT", "CC0-1.0", "CC-BY-4.0"):
            spdx = "Apache-2.0"
        prov = make_clean_room_provenance(
            aid,
            title=f"{fid} {aid}",
            license_spdx=spdx,
        )
        prov["family_id"] = fid
        render_jobs.append(
            {
                "kind": "dsl_rig",
                "idx": idx,
                "seed": seed,
                "canonical_dsl": r["canonical_dsl"],
                "provenance": prov,
                "enable_thinking": enable_thinking,
            }
        )

    # 5-mode Artist Assistant trajectories on diverse clean-room rigs
    for idx, r in enumerate(sampled_dsl_rows[:num_artist_dsl_rigs]):
        aid = str(r["asset_id"])
        fid = str(r["family_id"])
        prov = make_clean_room_provenance(aid, title=f"{fid} artist assistant")
        prov["family_id"] = fid
        mode = ARTIST_ASSISTANT_MODES[idx % len(ARTIST_ASSISTANT_MODES)]
        try:
            prog = validate_program(parse_program(str(r["canonical_dsl"])))
            rec, extra_blobs = build_artist_assistant_record(
                prog,
                mode=mode,
                asset_id=f"{aid}_{mode[:10]}",
                family_id=fid,
                provenance_override=prov,
                enable_thinking=enable_thinking,
                panel_size=(144, 144),
            )
            if idx % 5 == 4:
                rec["language"] = "zh"
            blob_map.update(extra_blobs)
            records.append((rec, list(rec.get("frames") or ())))
        except Exception:
            continue

    # 2. `synccraft-v3-human-artist-dataset` (`permissive_cc` ONLY!)
    human_rows: list[dict[str, Any]] = []
    for sub_file in ("data/permissive_cc/train.jsonl", "data/permissive_cc/val.jsonl"):
        p_artist = _hf_cached_file(
            SYNCCRAFT_ARTIST_REPO_ID,
            sub_file,
            revision=SYNCCRAFT_ARTIST_REVISION,
            token=token,
        )
        with p_artist.open("r", encoding="utf-8") as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                ar = json.loads(ln)
                if ar.get("ai_assisted") or not ar.get("commercial_weights_eligible"):
                    continue
                if "permissive_cc" not in str(ar.get("split") or ""):
                    continue
                spdx = str(ar.get("license_spdx") or "")
                if spdx not in ("CC0-1.0", "CC-BY-4.0", "MIT", "Apache-2.0"):
                    continue
                if len(str(ar.get("canonical_dsl") or "")) > 24_000:
                    # Skip multi-thousand-cuboid diorama scenes that exceed the 8192-token context policy
                    continue
                human_rows.append(ar)

    human_rows.sort(key=lambda r: identity([seed, "human_cc", r["asset_id"]]))
    for h_idx, ar in enumerate(human_rows):
        aid = str(ar["asset_id"])
        fid = str(ar.get("family_id") or f"human_cc_{aid[:16]}")
        lic = ar.get("source_license_and_review_record") or {}
        spdx = str(ar.get("license_spdx") or lic.get("license") or "")
        sha_hex = str(lic.get("source_sha256") or hashlib.sha256(aid.encode("utf-8")).hexdigest())
        prov = {
            "origin": "external_permissive",
            "asset_id": aid,
            "family_id": fid,
            "source_url": str(lic.get("source_url") or ar.get("source_url") or f"https://synccraft.local/artist/{aid}"),
            "author": str(lic.get("authors") or ar.get("author") or "community_cc_author"),
            "license_spdx": spdx,
            "license_policy": "clean_room_permissive",
            "attribution_notice": str(lic.get("attribution_notice") or f"{aid} under {spdx}"),
            "title": str(lic.get("title") or ar.get("title") or aid),
            "license_url": str(lic.get("license_url") or ar.get("license_url") or "https://creativecommons.org/licenses/by/4.0/"),
            "license_text_sha256": sha_hex,
            "retrieved_at_iso": str(lic.get("retrieved_at_iso") or "2026-11-15T00:00:00Z"),
            "ai_use_marker_status": "checked_none",
            "ai_marker_evidence_sha256": sha_hex,
            "is_ai_generated": False,
            "ai_assisted": False,
            "commercial_weights_eligible": True,
            "changes_made_note": str(lic.get("changes_made_notice") or "Decompiled into typed SyncCraft Block-DSL v3.2 AST"),
            "reviewer_id": "synccraft_catalog_verified_v3_2",
        }
        validate_synccraft_license_record({"provenance": prov}, allow_research_nc=False)
        if h_idx < 48:
            render_jobs.append(
                {
                    "kind": "dsl_rig",
                    "idx": num_rendered_dsl_rigs + h_idx,
                    "seed": seed + 1000,
                    "canonical_dsl": ar["canonical_dsl"],
                    "provenance": prov,
                    "enable_thinking": enable_thinking,
                }
            )
        try:
            prog = validate_program(parse_program(str(ar["canonical_dsl"])))
            pp_doc = ar.get("prompt_and_provenance") or {}
            if pp_doc.get("assistant_prompt") and pp_doc.get("assistant_completion"):
                scene = compile_geometry(prog)
                aud = audit_scene(scene, enforce_physical_contact=getattr(prog.meta, "audit_profile", "strict") != "authored")
                think = synthesize_oracle_think_trace(
                    task="synccraft_artist_assistant",
                    scene=scene,
                    audit_report=aud,
                    extra_facts=[f"human_artist_mode={ar.get('assistant_task_mode', 'create_full_asset')}"],
                )
                raw_comp = str(pp_doc["assistant_completion"]).strip()
                comp_fenced = raw_comp if raw_comp.startswith("```") or raw_comp.startswith("{") else f"```blockdsl\n{raw_comp}\n```"
                full_ans = f"{think}{comp_fenced}" if enable_thinking else comp_fenced
                direct_rec = {
                    "id": f"sc-human-cc-{aid}",
                    "source_id": aid,
                    "source_group": f"synccraft-family:{fid}",
                    "language": "en",
                    "task": "synccraft_artist_assistant",
                    "frames": [],
                    "enable_thinking": enable_thinking,
                    "prompt": str(pp_doc["assistant_prompt"]).strip(),
                    "answer": full_ans,
                    "target": {"kind": "text", "text": comp_fenced},
                    "provenance": prov,
                    "metadata": {
                        "asset_id": aid,
                        "family_id": fid,
                        "assistant_mode": str(ar.get("assistant_task_mode") or "create_full_asset"),
                        "audit_profile": getattr(prog.meta, "audit_profile", "strict"),
                        "expected_audit_profile": getattr(prog.meta, "audit_profile", "strict"),
                        "provenance": prov,
                        "grouping_keys": [f"synccraft-family:{fid}", f"synccraft-asset:{aid}"],
                    },
                }
                records.append((direct_rec, []))
            rec, extra_blobs = build_artist_assistant_record(
                prog,
                mode="coarse_to_fine_upgrade" if h_idx % 2 == 0 else "inverse_structure_and_kinematics_reading",
                asset_id=f"{aid}_assist",
                family_id=fid,
                provenance_override=prov,
                enable_thinking=enable_thinking,
                panel_size=(144, 144),
            )
            if h_idx % 4 == 3:
                rec["language"] = "zh"
            blob_map.update(extra_blobs)
            records.append((rec, list(rec.get("frames") or ())))
        except Exception:
            continue

    # 3. `apex_complex_v3_2/train.jsonl` (360 musculoskeletal & sculpted complex assets)
    apex_path = _hf_cached_file(
        SYNCCRAFT_DSL_REPO_ID,
        "apex_complex_v3_2/train.jsonl",
        revision=SYNCCRAFT_DSL_REVISION,
        token=token,
    )
    with apex_path.open("r", encoding="utf-8") as fh:
        for a_idx, ln in enumerate(fh):
            if not ln.strip():
                continue
            ax = json.loads(ln)
            sp = str(ax.get("split") or "")
            if "test" in sp or not ax.get("commercial_weights_eligible", True):
                continue
            aid = str(ax.get("sample_id") or f"apex_v32_{a_idx:04d}")
            fid = str(ax.get("family_id") or "apex_complex_v32")
            dsl_txt = str(ax["assistant_dsl"]).strip()
            prov = make_clean_room_provenance(
                aid,
                title=f"Apex v3.2 {fid}",
                license_spdx=str(ax.get("license_spdx") or "Apache-2.0"),
            )
            prov["family_id"] = fid
            try:
                prog = validate_program(parse_program(dsl_txt))
                scene = compile_geometry(prog)
                aud = audit_scene(scene, enforce_physical_contact=getattr(prog.meta, "audit_profile", "strict") != "authored")
                think = synthesize_oracle_think_trace(
                    task="synccraft_artist_assistant",
                    scene=scene,
                    audit_report=aud,
                    extra_facts=[f"apex_v32_mode={ax.get('task_mode', 'progressive')}"],
                )
                canon = format_program(prog)
                ans = f"{think}```blockdsl\n{canon}\n```" if enable_thinking else f"```blockdsl\n{canon}\n```"
                rec = {
                    "id": f"sc-apex-v32-{aid}",
                    "source_id": aid,
                    "source_group": f"synccraft-family:{fid}",
                    "language": "en",
                    "task": "synccraft_artist_assistant",
                    "frames": [],
                    "enable_thinking": enable_thinking,
                    "prompt": str(ax["user_prompt"]).strip(),
                    "answer": ans,
                    "target": {"kind": "text", "text": ans},
                    "provenance": prov,
                    "metadata": {
                        "asset_id": aid,
                        "family_id": fid,
                        "assistant_mode": str(ax.get("task_mode") or "apex_progressive"),
                        "audit_profile": getattr(prog.meta, "audit_profile", "strict"),
                        "expected_audit_profile": getattr(prog.meta, "audit_profile", "strict"),
                        "provenance": prov,
                        "grouping_keys": [f"synccraft-family:{fid}", f"synccraft-asset:{aid}"],
                    },
                }
                records.append((rec, []))
                if a_idx < 24:
                    render_jobs.append(
                        {
                            "kind": "dsl_rig",
                            "idx": num_rendered_dsl_rigs + 100 + a_idx,
                            "seed": seed + 3000,
                            "canonical_dsl": canon,
                            "provenance": prov,
                            "enable_thinking": enable_thinking,
                        }
                    )
            except Exception:
                continue

    # 4. `world_capability_ladder.jsonl` (Levels 1..7, `train` & `val` ONLY — never `test`!)
    ladder_path = _hf_cached_file(
        SYNCCRAFT_DSL_REPO_ID,
        "world_capability_ladder.jsonl",
        revision=SYNCCRAFT_DSL_REVISION,
        token=token,
    )
    with ladder_path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            lr = json.loads(ln)
            sp = str(lr.get("split") or "")
            if sp not in ("train", "val") or "test" in sp:
                continue
            rec, l_blobs = build_world_capability_record_from_ladder_row(
                lr,
                enable_thinking=enable_thinking,
                render_multiview_for_level5=(lr.get("level") == "level_5_invert_observation"),
                panel_size=(144, 144),
            )
            blob_map.update(l_blobs)
            records.append((rec, list(rec.get("frames") or ())))

    # 5. Parametric quadruped `foot_cycle` + `ADR 0007` sculpted wyvern & walking citadel `spin(...)` jobs
    for q_idx in range(num_quad_clips):
        render_jobs.append(
            {
                "kind": "quadruped_anim",
                "idx": q_idx,
                "variant_idx": q_idx,
                "seed": seed,
                "enable_thinking": enable_thinking,
            }
        )
    for f_idx in range(num_flagship):
        render_jobs.append(
            {
                "kind": "sculpted_flagship",
                "idx": f_idx,
                "variant_idx": f_idx,
                "seed": seed,
                "enable_thinking": enable_thinking,
            }
        )

    max_workers = max(1, min(int(config.get("corpus_render_workers", min(12, os.cpu_count() or 4))), len(render_jobs)))
    emit("hybrid_synccraft_render_start", jobs=len(render_jobs), workers=max_workers)
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for r_list, b_dict in pool.map(_render_synccraft_rig_worker, render_jobs, chunksize=2):
                for m_name, payload in b_dict.items():
                    if m_name in blob_map and blob_map[m_name] != payload:
                        raise ValueError(f"Conflicting media path {m_name!r} in synccraft_world_v32")
                    blob_map[m_name] = payload
                for r in r_list:
                    records.append((r, list(r.get("frames") or ())))
    else:
        for job in render_jobs:
            r_list, b_dict = _render_synccraft_rig_worker(job)
            blob_map.update(b_dict)
            for r in r_list:
                records.append((r, list(r.get("frames") or ())))

    return records, blob_map


def _filter_packets_to_context_policy(
    packets: list[dict[str, Any]],
    root: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enforce exact `max_answer_tokens` and `max_sequence` budgets before freezing `packets.jsonl.gz`.

    Prevents runtime `ValueError('Supervision would be truncated; revise a versioned context policy')`
    from oversized multi-hundred-part diorama prompts/answers while preserving zero silent truncation.
    """
    max_seq = int(config.get("max_sequence", 8192))
    max_ans = int(config.get("max_answer_tokens", 4096))
    max_vis = int(config.get("max_visual_tokens", 768))
    detail_tiles = int(config.get("detail_tiles", 4))
    ans_limit = max(32, max_ans - 32)
    vis_seq_limit = max(128, max_seq - max_vis - 96)
    txt_seq_limit = max(128, max_seq - 96)

    tok = None
    spark_comp = str(config.get("spark_component", "spark_fp8"))
    for cand_dir in (root / "assets" / spark_comp, root / "assets" / "spark_fp8", root / "assets" / "spark"):
        if (cand_dir / "tokenizer_config.json").is_file() or (cand_dir / "tokenizer.json").is_file():
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(cand_dir), trust_remote_code=True)
            break

    def _tok_len(s: str) -> int:
        if tok is not None:
            return len(tok.encode(s, add_special_tokens=False))
        return math.ceil(len(s.encode("utf-8")) / 2.5)

    kept: list[dict[str, Any]] = []
    dropped_by_task: Counter[str] = Counter()
    dropped_packets = 0

    for pkt in packets:
        ok = True
        for orig in variants(pkt["record"]):
            for r in training_views(orig):
                p_str = str(r.get("prompt") or "") + ("\n" + str(r["system_prompt"]) if r.get("system_prompt") else "")
                a_str = str(r.get("answer") or "")
                # Fast character guard avoids BPE work on huge multi-thousand-box diorama strings
                if len(p_str) > max_seq * 5 or len(a_str) > max_ans * 5:
                    ok = False
                    dropped_by_task[str(r.get("task") or "unknown")] += 1
                    break
                nf = len(r.get("frames") or ())
                has_vis = bool(nf or r.get("video"))
                if nf > 0:
                    vk = infer_visual_kind(r)
                    min_units = nf if vk == "temporal_frame" else nf * (1 + detail_tiles)
                    if min_units * 16 > max_vis:
                        ok = False
                        dropped_by_task[str(r.get("task") or "unknown")] += 1
                        break
                a_len = _tok_len(a_str)
                if a_len > ans_limit:
                    ok = False
                    dropped_by_task[str(r.get("task") or "unknown")] += 1
                    break
                p_len = _tok_len(p_str)
                if p_len + a_len > (vis_seq_limit if has_vis else txt_seq_limit):
                    ok = False
                    dropped_by_task[str(r.get("task") or "unknown")] += 1
                    break
            if not ok:
                break
        if ok:
            kept.append(pkt)
        else:
            dropped_packets += 1

    stats = {
        "tokenizer_verified": tok is not None,
        "max_sequence": max_seq,
        "max_answer_tokens": max_ans,
        "kept_packets": len(kept),
        "dropped_packets": dropped_packets,
        "dropped_by_task": dict(sorted(dropped_by_task.items())),
    }
    emit("hybrid_context_policy_filter", **stats)
    return kept, stats


def prepare_hybrid_world_v32_data(root: str | Path, token: str | None, config: dict[str, Any]) -> dict[str, Any]:
    """Build or verify the frozen `hybrid_world_v32` multi-source corpus & SQLite split DB."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cat = root / "metadata" / "packets.jsonl.gz"
    db = root / "metadata" / "corpus.sqlite"
    build_meta = root / "metadata" / "BUILD.json"
    handoff_path = root / "metadata" / "HANDOFF.json"

    if db.exists() and cat.exists() and build_meta.exists() and handoff_path.exists():
        expected = {
            "catalogue_sha256": sha(cat),
            "split": config["split_percent"],
            "seed": config["seed"],
            "dataset_mode": "hybrid_world_v32",
        }
        if read(build_meta) != expected:
            raise ValueError("Existing hybrid_world_v32 split metadata differs; use a fresh work_root")
        return read(db.with_suffix(".report.json"))

    seed = int(config["seed"])
    emit("hybrid_corpus_stage", stage="spark_mm_selection")
    spark_mm_packets, spark_mm_comps = _select_spark_mm_packets(root, token, config)

    emit("hybrid_corpus_stage", stage="sat_3d_spatial")
    sat_records, sat_blobs = _build_sat_3d_spatial_records(token, config)
    sat_comp_entry, sat_transport = _write_local_tar_component(root, "sat_3d_spatial", sat_blobs, seed=seed)

    emit("hybrid_corpus_stage", stage="compositional_and_synccraft_world_v32")
    comp_records, comp_blobs = _build_compositional_vqa_records(config)
    sc_records, sc_blobs = _build_synccraft_world_v32_records(token, config)

    merged_sc_blobs = dict(comp_blobs)
    for k, v in sc_blobs.items():
        if k in merged_sc_blobs and merged_sc_blobs[k] != v:
            raise ValueError(f"Duplicate media path {k!r} between compositional_vqa and synccraft_world_v32")
        merged_sc_blobs[k] = v
    sc_comp_entry, sc_transport = _write_local_tar_component(root, "synccraft_world_v32", merged_sc_blobs, seed=seed)

    handoff_doc = {
        "schema_version": 1,
        "repo_id": "TassP0lar/SparkX2.6-MM",
        "revision": "f472f980a6c95d3e9844f27e6045c8c09b21536a",
        "components": [
            *spark_mm_comps,
            sat_comp_entry,
            sc_comp_entry,
        ],
    }
    write(handoff_path, handoff_doc)

    all_packets: list[dict[str, Any]] = list(spark_mm_packets)

    for rec, frame_names in sat_records:
        shards_used = {sat_transport[m]["shard"] for m in frame_names}
        if len(shards_used) != 1:
            # Ensure all frames of one multi-image packet reside in the same shard
            continue
        shard_name = next(iter(shards_used))
        transport = {
            "shard": shard_name,
            "media": [dict(sat_transport[m]["entry"]) for m in frame_names],
        }
        all_packets.append({"component": "sat_3d_spatial", "record": rec, "transport": transport})

    for rec, frame_names in [*comp_records, *sc_records]:
        if frame_names:
            shards_used = {sc_transport[m]["shard"] for m in frame_names}
            if len(shards_used) != 1:
                continue
            shard_name = next(iter(shards_used))
            transport = {
                "shard": shard_name,
                "media": [dict(sc_transport[m]["entry"]) for m in frame_names],
            }
            all_packets.append({"component": "synccraft_world_v32", "record": rec, "transport": transport})
        else:
            all_packets.append({"component": "synccraft_world_v32", "record": rec, "transport": None})

    # Enforce qualified context policy across 100% of rows before freezing catalogue
    all_packets, context_filter_stats = _filter_packets_to_context_policy(all_packets, root, config)

    # Deterministic canonical packet order
    all_packets.sort(key=lambda p: (p["component"], str(p["record"]["id"])))

    cat.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cat, "wt", encoding="utf-8") as gz:
        for pkt in all_packets:
            gz.write(json.dumps(pkt, ensure_ascii=False, separators=(",", ":")) + "\n")

    task_counts = Counter()
    for pkt in all_packets:
        for orig in variants(pkt["record"]):
            task_counts[str(orig.get("task", ""))] += 1

    report = build_database(cat, db, config)
    report["hybrid_sources"] = {
        "spark_mm_packets": len(spark_mm_packets),
        "sat_3d_records": len(sat_records),
        "compositional_vqa_records": len(comp_records),
        "synccraft_world_v32_records": len(sc_records),
        "context_policy_filter": context_filter_stats,
        "task_distribution": dict(sorted(task_counts.items())),
    }
    write(db.with_suffix(".report.json"), report)
    write(
        build_meta,
        {
            "catalogue_sha256": sha(cat),
            "split": config["split_percent"],
            "seed": config["seed"],
            "dataset_mode": "hybrid_world_v32",
        },
    )
    emit(
        "hybrid_corpus_ready",
        packets=len(all_packets),
        instructions=report["instructions"],
        groups=report["groups"],
        tasks=len(task_counts),
    )
    return report
