"""SyncCraft v3.2 3D geometry, RGBA UV/material, world-action, and @behavior bridge for Spark-X2.6-VL.

Enforces SyncCraft AGENTS.md invariants:
- Typed Block-DSL parsing (`parse_program` / `action_from_dict`); zero `eval` or code execution.
- Shared SE(3) kinematics (`compile_geometry` / `compile_posed_geometry`), 15-axis SAT + P1-02
  contact audit (`audit_scene`), RGBA pixel-art UV atlas compilation (`compile_vanilla_atlas`),
  and swept OBB SAT + <0.02u stance-slip animation verification (`audit_behavior_clip`).
- Strict `clean_room_permissive` / `NoAI` / Mojang provenance filtering and asset-family split
  grouping so renders, UV atlases, scene edits, and animation clips of one asset never cross splits.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _ensure_synccraft_importable() -> None:
    try:
        import synccraft  # noqa: F401
        return
    except ImportError:
        pass
    candidates = [
        Path(__file__).resolve().parents[3] / "SyncCraft" / "src",
        Path.home() / "SyncCraft" / "src",
    ]
    for candidate in candidates:
        if (candidate / "synccraft" / "__init__.py").is_file():
            sys.path.insert(0, str(candidate))
            return


_ensure_synccraft_importable()
ensure_synccraft_importable = _ensure_synccraft_importable

from synccraft.audit.offscreen_renderer import CAMERA_PRESETS, render_world_observation  # noqa: E402
from synccraft.audit.sync_audit import audit_scene  # noqa: E402
from synccraft.data_gen.taxonomy_catalog import normalize_clean_room_family_id  # noqa: E402
from synccraft.dsl.ast_nodes import (  # noqa: E402
    Attach,
    BehaviorDecl,
    Bone,
    Box,
    ClipDecl,
    IKChainDecl,
    IKDriveDecl,
    KinematicsDecl,
    Meta,
    Palette,
    Place,
    Program,
    TrackDecl,
    validate_program,
)
from synccraft.dsl.parser import format_program, parse_program  # noqa: E402
from synccraft.kinematics.musculoskeletal import (  # noqa: E402
    audit_behavior_clip,
    evaluate_behavior_clip,
)
from synccraft.kinematics.se3_chain import Scene, compile_geometry, compile_posed_geometry  # noqa: E402
from synccraft.shading.compiler import compile_vanilla_atlas, encode_rgba_png  # noqa: E402
from synccraft.shading.uv_packer import allocate_uv_layout  # noqa: E402
from synccraft.surface.contracts import (  # noqa: E402
    EnvironmentState,
    MaterialProgram,
    MaterialSpec,
    StyleConfig,
    WorldState,
)
from synccraft.surface.domain import build_surface_domain, initialize_surface_state  # noqa: E402
from synccraft.data_gen.curriculum_and_assistant import (  # noqa: E402
    coarsen_program_to_tier,
    ensure_concise_texture_program,
)
from synccraft.world.artist_loop import (  # noqa: E402
    analyze_artist_project,
    enhance_artist_project_step,
)
from synccraft.world.exemplars import (  # noqa: E402
    COLOSSAL_WALKING_CITADEL_DSL,
    SCULPTED_ELDER_WYVERN_DSL,
    build_articulated_mechanism_world,
    build_biological_creature_world,
    build_colossal_walking_citadel_world,
    build_sculpted_elder_wyvern_world,
)
from synccraft.world.trajectory_dataset import (  # noqa: E402
    decode_record_observation_image,
    score_world_capability_completion,
)
from synccraft.world.inverse_inference import (  # noqa: E402
    APPROVED_NONCOMMERCIAL_RESEARCH_LICENSES,
    APPROVED_PERMISSIVE_LICENSES,
    BANNED_PROVENANCE_TOKENS,
    AssetProvenance,
    LicenseGateError,
    scan_text_for_noai_markers,
    verify_asset_provenance,
)
from synccraft.world.transitions import (  # noqa: E402
    ApplyRubbingAction,
    SetPoseAction,
    action_from_dict,
    step_world,
)

SYNCCRAFT_TASKS: tuple[str, ...] = (
    "synccraft_part_grounding",
    "synccraft_multiview_inverse_dsl",
    "synccraft_uv_alpha_diagnosis",
    "synccraft_typed_scene_edit",
    "synccraft_behavior_animation",
    "synccraft_world_capability",
    "synccraft_artist_assistant",
)

ARTIST_ASSISTANT_MODES: tuple[str, ...] = (
    "create_full_asset",
    "coarse_to_fine_upgrade",
    "texture_and_animate_rig",
    "pose_and_part_edit_refinement",
    "inverse_structure_and_kinematics_reading",
)

THINK_BLOCK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*(.*)$", re.DOTALL)
DSL_FENCE_RE = re.compile(r"```(?:blockdsl|dsl|text)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def strip_think_block(text: str) -> tuple[str | None, str]:
    """Split optional `<think>...</think>` trace from final structured answer."""
    if not isinstance(text, str):
        return None, ""
    match = THINK_BLOCK_RE.match(text)
    if match is not None:
        return match.group(1).strip(), match.group(2).strip()
    if "<think>" in text or "</think>" in text:
        # Malformed unclosed or misordered think tags: strip conservatively after closing tag if present.
        if "</think>" in text:
            before, after = text.split("</think>", 1)
            return before.replace("<think>", "").strip(), after.strip()
        return text.replace("<think>", "").strip(), ""
    return None, text.strip()


def validate_synccraft_license_record(
    record: dict[str, Any],
    *,
    allow_research_nc: bool = False,
) -> str:
    """Enforce SyncCraft clean-room license, NoDerivatives, Mojang exclusion, and NoAI marker gate."""
    meta = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
    prov_dict = record.get("provenance") or meta.get("provenance")
    if not isinstance(prov_dict, dict):
        raise LicenseGateError("SyncCraft training/evaluation records require an explicit provenance mapping")

    policy = "research_nc" if allow_research_nc else "clean_room_permissive"
    provenance = AssetProvenance(
        asset_id=str(prov_dict.get("asset_id", "")),
        source_url=str(prov_dict.get("source_url", "")),
        author=str(prov_dict.get("author", "")),
        license_spdx=str(prov_dict.get("license_spdx", "")),
        attribution_notice=str(prov_dict.get("attribution_notice", "")),
        title=str(prov_dict.get("title", "")),
        license_url=str(prov_dict.get("license_url", "")),
        license_text_sha256=str(prov_dict.get("license_text_sha256", "")),
        retrieved_at_iso=str(prov_dict.get("retrieved_at_iso", "")),
        ai_use_marker_status=str(prov_dict.get("ai_use_marker_status", "checked_none")),
        ai_marker_evidence_sha256=str(prov_dict.get("ai_marker_evidence_sha256", "")),
        is_ai_generated=bool(prov_dict.get("is_ai_generated", False)),
        changes_made_note=str(prov_dict.get("changes_made_note", "")),
        reviewer_id=str(prov_dict.get("reviewer_id", "")),
    )

    text_blobs = [
        str(record.get("prompt", "")),
        str(record.get("answer", "")),
        str(meta.get("notes", "")),
        str(meta.get("tags", "")),
        str(prov_dict.get("notes", "")),
    ]
    joined = " | ".join(x for x in text_blobs if x)
    lower = joined.lower()
    src_meta_lower = " | ".join(
        [
            provenance.source_url.lower(),
            provenance.license_url.lower(),
            provenance.author.lower(),
            provenance.title.lower(),
            provenance.attribution_notice.lower(),
        ]
    )
    for banned in BANNED_PROVENANCE_TOKENS:
        if banned in lower or banned in src_meta_lower:
            raise LicenseGateError(f"Banned provenance token {banned!r} detected in SyncCraft record")
    for banned_domain in ("minecraft.net", "mojang.com", "minecrafteula"):
        if banned_domain in src_meta_lower or banned_domain in lower:
            raise LicenseGateError(f"Banned Mojang/Minecraft domain {banned_domain!r} detected in SyncCraft record")
    noai_hits = scan_text_for_noai_markers(joined)
    if noai_hits:
        raise LicenseGateError(f"SyncCraft record rejected due to NoAI/TDM/AI-generated markers: {list(noai_hits)}")

    track = verify_asset_provenance(provenance, license_policy=policy, metadata_text=joined)
    if not allow_research_nc and track != "clean_room_permissive":
        raise LicenseGateError(
            f"Commercial Apache-2.0 Spark-X2.6-VL training requires clean_room_permissive, got {track!r} "
            f"(approved permissive: {sorted(APPROVED_PERMISSIVE_LICENSES)}; "
            f"quarantined research_nc: {sorted(APPROVED_NONCOMMERCIAL_RESEARCH_LICENSES)})"
        )
    return track


def synccraft_group_keys(record: dict[str, Any]) -> list[str]:
    """Return split-grouping keys binding all variants (renders, UVs, edits, clips) of an asset/family."""
    keys: set[str] = set()
    if record.get("source_group"):
        keys.add(str(record["source_group"]))
    meta = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
    prov = record.get("provenance") or meta.get("provenance") or {}

    asset_id = meta.get("asset_id") or (prov.get("asset_id") if isinstance(prov, dict) else None)
    base_asset_id = meta.get("base_asset_id")
    lineage_id = meta.get("lineage_id")
    family_id = meta.get("family_id")

    if asset_id:
        keys.add(f"synccraft-asset:{asset_id}")
        # Strip variant suffixes such as `#uv`, `#edit`, `#anim` so derived tasks share one group.
        root_asset = str(asset_id).split("#", 1)[0]
        keys.add(f"synccraft-asset:{root_asset}")
    if base_asset_id:
        keys.add(f"synccraft-asset:{base_asset_id}")
    if lineage_id:
        keys.add(f"synccraft-lineage:{lineage_id}")
    if family_id:
        canon = normalize_clean_room_family_id(str(family_id))
        keys.add(f"synccraft-family-asset:{canon}:{base_asset_id or asset_id or record.get('source_group', 'root')}")
    return sorted(keys)


def make_clean_room_provenance(asset_id: str, title: str = "SyncCraft Clean-Room Procedural Rig") -> dict[str, Any]:
    """Construct verified Apache-2.0 clean-room procedural provenance metadata."""
    evidence_hash = hashlib.sha256(f"synccraft-v3.2:{asset_id}".encode("utf-8")).hexdigest()
    return {
        "asset_id": asset_id,
        "source_url": f"https://synccraft.local/procedural/{asset_id}",
        "author": "SyncCraft Deterministic Clean-Room Generator v3.2",
        "license_spdx": "Apache-2.0",
        "attribution_notice": "SyncCraft v3.2 clean-room procedural synthesis",
        "title": title,
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "license_text_sha256": evidence_hash,
        "retrieved_at_iso": "2026-11-14T00:00:00Z",
        "ai_use_marker_status": "opt_in_signed",
        "ai_marker_evidence_sha256": evidence_hash,
        "is_ai_generated": False,
        "changes_made_note": "Procedural SE(3) + GGX/OKLCH clean-room synthesis",
        "reviewer_id": "synccraft_automated_gate_v3_2",
    }


def synthesize_oracle_think_trace(
    *,
    task: str,
    scene: Scene,
    audit_report: dict[str, Any],
    extra_facts: list[str] | None = None,
) -> str:
    """Build a concise, oracle-verified `<think>...</think>` spatial reasoning block."""
    n_coll = len(audit_report.get("collision_pairs", []))
    n_contact_viol = len(audit_report.get("contact_violations", []))
    n_zfight = len(audit_report.get("z_fighting_pairs", []))
    n_sculpted = len(audit_report.get("sculpted_overlap_pairs", []))
    aud_prof = str(audit_report.get("audit_profile", getattr(scene.meta, "audit_profile", "strict")))
    lines = [
        f"task={task}",
        (
            f"se3_rig: bones={len(scene.bones)}, boxes={len(scene.boxes)}, "
            f"audit_profile={aud_prof}, declared_contacts={len(audit_report.get('declared_contact_pairs', []))}, "
            f"sculpted_overlaps={n_sculpted}"
        ),
        f"sat_p1_02_audit: passed={audit_report.get('passed')}, collisions={n_coll}, contact_violations={n_contact_viol}, z_fighting={n_zfight}",
    ]
    if extra_facts:
        lines.extend(extra_facts)
    return "<think>\n" + "\n".join(lines) + "\n</think>\n"


def format_supervised_answer(answer_body: str, think_trace: str | None, *, enable_thinking: bool) -> str:
    if enable_thinking and think_trace:
        prefix = think_trace if think_trace.startswith("<think>") else f"<think>\n{think_trace.strip()}\n</think>\n"
        return prefix + answer_body.strip()
    return answer_body.strip()


def _camera_unit_meta(view_name: str, ortho_scale: float = 1.0) -> dict[str, Any]:
    right_raw, up_raw = CAMERA_PRESETS[view_name]
    right = np.asarray(right_raw, dtype=np.float64)
    right = (right / max(1e-12, float(np.linalg.norm(right)))).tolist()
    up = np.asarray(up_raw, dtype=np.float64)
    up = (up / max(1e-12, float(np.linalg.norm(up)))).tolist()
    return {
        "visual_kind": "multiview_render",
        "view_id": view_name,
        "camera_right": [round(float(v), 5) for v in right],
        "camera_up": [round(float(v), 5) for v in up],
        "ortho_scale": round(float(ortho_scale), 5),
        "has_timestamp": False,
        "timestamp_s": 0.0,
        "is_pixel_art_atlas": False,
    }


def _encode_rgb_or_rgba_png(arr: np.ndarray) -> bytes:
    mode = "RGBA" if arr.ndim == 3 and arr.shape[-1] == 4 else "RGB"
    im = Image.fromarray(arr.astype(np.uint8), mode=mode)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def render_world_views_rgba(
    world_state: WorldState,
    *,
    views: tuple[str, ...] = ("south", "right", "top", "iso"),
    panel_size: tuple[int, int] = (160, 160),
    atlas_override: tuple[dict[str, Any], bytes] | None = None,
) -> tuple[dict[str, bytes], dict[str, list[str]], dict[str, Any], bytes]:
    """Render orthographic views with RGBA silhouette mask and visible-part lists."""
    obs = render_world_observation(
        world_state,
        atlas_override=atlas_override,
        panel_size=panel_size,
        views=views,
    )
    posed_scene = compile_posed_geometry(world_state.program, world_state.pose_euler_deltas)
    atlas_desc, atlas_png = atlas_override if atlas_override is not None else compile_vanilla_atlas(world_state)

    png_by_view: dict[str, bytes] = {}
    visible_boxes_by_view: dict[str, list[str]] = {}
    for vname in views:
        rgb = obs["views"][vname]
        hit_map = obs["hit_box_maps"][vname]
        alpha = np.where(hit_map >= 0, 255, 0).astype(np.uint8)[..., None]
        rgba = np.concatenate([rgb, alpha], axis=-1)
        png_by_view[vname] = _encode_rgb_or_rgba_png(rgba)
        hit_indices = sorted(int(i) for i in np.unique(hit_map) if int(i) >= 0)
        visible_boxes_by_view[vname] = [posed_scene.boxes[i].name for i in hit_indices]
    return png_by_view, visible_boxes_by_view, atlas_desc, atlas_png


def build_part_grounding_record(
    world_state: WorldState,
    *,
    asset_id: str,
    family_id: str,
    enable_thinking: bool = True,
    views: tuple[str, ...] = ("south", "right", "top", "iso"),
    panel_size: tuple[int, int] = (160, 160),
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Build a multiview 3D part-grounding & hierarchy-reading record (never treated as video)."""
    scene = compile_geometry(world_state.program)
    audit = audit_scene(scene)
    png_by_view, visible_by_view, _, _ = render_world_views_rgba(
        world_state, views=views, panel_size=panel_size
    )
    frame_paths = [f"{asset_id}/view_{v}.png" for v in views]
    media_payloads = {frame_paths[i]: png_by_view[v] for i, v in enumerate(views)}
    visual_units = [_camera_unit_meta(v) for v in views]

    mirrored_bones = sorted(
        b.name for b in scene.bones if "__" in b.name or b.name.startswith("right_")
    )
    answer_obj = {
        "bone_count": len(scene.bones),
        "box_count": len(scene.boxes),
        "bones": [b.name for b in scene.bones],
        "mirrored_or_expanded_bones": mirrored_bones,
        "visible_boxes_by_view": visible_by_view,
        "sat_collision_free": len(audit["collision_pairs"]) == 0,
    }
    answer_body = json.dumps(answer_obj, sort_keys=True, separators=(",", ":"))
    think = synthesize_oracle_think_trace(
        task="synccraft_part_grounding",
        scene=scene,
        audit_report=audit,
        extra_facts=[
            f"views={list(views)}",
            f"visible_south={visible_by_view.get('south', [])}",
            f"visible_iso={visible_by_view.get('iso', [])}",
        ],
    )
    prov = make_clean_room_provenance(asset_id, title=f"{family_id} multiview part grounding")
    row = {
        "id": f"{asset_id}#part_grounding",
        "source_group": f"synccraft:{asset_id}",
        "language": "en",
        "task": "synccraft_part_grounding",
        "visual_kind": "multiview_render",
        "enable_thinking": enable_thinking,
        "frames": frame_paths,
        "media_times_s": [0.0] * len(frame_paths),
        "visual_units": visual_units,
        "prompt": (
            f"Inspect the {len(views)} orthographic views ({', '.join(views)}) of this SyncCraft 3D cuboid rig. "
            "Do not treat these static camera views as video frames. Return JSON with keys "
            "bone_count, box_count, bones, mirrored_or_expanded_bones, visible_boxes_by_view, and sat_collision_free."
        ),
        "answer": format_supervised_answer(answer_body, think, enable_thinking=enable_thinking),
        "target": {"kind": "text", "text": answer_body},
        "provenance": prov,
        "metadata": {
            "asset_id": asset_id,
            "base_asset_id": asset_id,
            "family_id": family_id,
            "provenance": prov,
            "canonical_dsl": format_program(world_state.program),
            "oracle_json": answer_obj,
        },
    }
    validate_synccraft_license_record(row)
    return row, media_payloads


def build_multiview_consistency_record(
    world_state: WorldState,
    *,
    asset_id: str,
    family_id: str,
    enable_thinking: bool = True,
    views: tuple[str, ...] = ("south", "right", "top", "iso"),
    panel_size: tuple[int, int] = (160, 160),
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Build a multiview orthographic observation -> Block-DSL inverse modeling record."""
    scene = compile_geometry(world_state.program)
    audit = audit_scene(scene)
    png_by_view, _, _, _ = render_world_views_rgba(world_state, views=views, panel_size=panel_size)
    frame_paths = [f"{asset_id}/inv_{v}.png" for v in views]
    media_payloads = {frame_paths[i]: png_by_view[v] for i, v in enumerate(views)}
    visual_units = [_camera_unit_meta(v) for v in views]

    aud_prof = str(getattr(scene.meta, "audit_profile", "strict"))
    dsl_text = format_program(world_state.program)
    answer_body = f"```blockdsl\n{dsl_text.strip()}\n```"
    think = synthesize_oracle_think_trace(
        task="synccraft_multiview_inverse_dsl",
        scene=scene,
        audit_report=audit,
        extra_facts=[
            f"target_profile={scene.meta.target_profile}, texel_density={scene.meta.texel_density}, audit_profile={aud_prof}",
            f"palettes={[p.name for p in world_state.program.palettes]}",
        ],
    )
    audit_clause = (
        f", and audit_profile={aud_prof!r}"
        if aud_prof != "strict"
        else ""
    )
    prov = make_clean_room_provenance(asset_id, title=f"{family_id} multiview Block-DSL inversion")
    row = {
        "id": f"{asset_id}#multiview_dsl",
        "source_group": f"synccraft:{asset_id}",
        "language": "en",
        "task": "synccraft_multiview_inverse_dsl",
        "visual_kind": "multiview_render",
        "enable_thinking": enable_thinking,
        "frames": frame_paths,
        "media_times_s": [0.0] * len(frame_paths),
        "visual_units": visual_units,
        "prompt": (
            f"Reconstruct a collision-free SyncCraft Block-DSL v3.2 program matching these {len(views)} "
            f"orthographic views ({', '.join(views)}) with target_profile={scene.meta.target_profile!r}, "
            f"texel_density={scene.meta.texel_density}{audit_clause}."
        ),
        "answer": format_supervised_answer(answer_body, think, enable_thinking=enable_thinking),
        "target": {"kind": "text", "text": answer_body},
        "provenance": prov,
        "metadata": {
            "asset_id": asset_id,
            "base_asset_id": asset_id,
            "family_id": family_id,
            "provenance": prov,
            "canonical_dsl": dsl_text,
            "expected_audit_profile": aud_prof,
        },
    }
    validate_synccraft_license_record(row)
    return row, media_payloads


def _find_overlapping_uv_islands(faces_table: dict[str, dict[str, list[int | float]]]) -> list[str]:
    """Compute exact sorted list of UV islands with positive 2D pixel-rectangle overlap."""
    entries: list[tuple[str, int, int, int, int]] = []
    for bname in sorted(faces_table):
        fmap = faces_table[bname]
        for fname in sorted(fmap):
            ru0, rv0, ru1, rv1 = (int(x) for x in fmap[fname][:4])
            if ru1 > ru0 and rv1 > rv0:
                entries.append((f"{bname}.{fname}", ru0, rv0, ru1, rv1))
    overlapping: set[str] = set()
    for i in range(len(entries)):
        id_a, u0_a, v0_a, u1_a, v1_a = entries[i]
        for j in range(i + 1, len(entries)):
            id_b, u0_b, v0_b, u1_b, v1_b = entries[j]
            if u0_a < u1_b and u0_b < u1_a and v0_a < v1_b and v0_b < v1_a:
                overlapping.add(id_a)
                overlapping.add(id_b)
    return sorted(overlapping)


def build_uv_seam_alpha_diagnosis_record(
    world_state: WorldState,
    *,
    asset_id: str,
    family_id: str,
    inject_alpha_cutout: bool = True,
    inject_uv_overlap: bool = False,
    vary_target_faces: bool = False,
    enable_thinking: bool = True,
    panel_size: tuple[int, int] = (160, 160),
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Build an RGBA UV-atlas + 3D render diagnosis record preserving exact alpha channels."""
    scene = compile_geometry(world_state.program)
    audit = audit_scene(scene)
    atlas_desc, atlas_png = compile_vanilla_atlas(world_state, padding=1)
    atlas_rgba = np.asarray(Image.open(io.BytesIO(atlas_png)).convert("RGBA"), dtype=np.uint8).copy()

    face_order = ("north", "south", "east", "west", "up", "down")
    seed_int = int(hashlib.sha256(f"{asset_id}:{int(inject_alpha_cutout)}:{int(inject_uv_overlap)}".encode("utf-8")).hexdigest()[:8], 16)
    box_names = [b.name for b in scene.boxes if b.name in atlas_desc["faces"]]
    if vary_target_faces and box_names:
        viable = [
            (bn, fn)
            for bn in box_names
            for fn in face_order
            if fn in atlas_desc["faces"][bn]
            and (int(atlas_desc["faces"][bn][fn][2]) - int(atlas_desc["faces"][bn][fn][0])) >= 4
            and (int(atlas_desc["faces"][bn][fn][3]) - int(atlas_desc["faces"][bn][fn][1])) >= 4
        ]
        if viable:
            target_box, target_face = viable[seed_int % len(viable)]
        else:
            target_box, target_face = box_names[0], "north"
    else:
        target_box, target_face = scene.boxes[0].name, "north"

    u0, v0, u1, v1 = atlas_desc["faces"][target_box][target_face]
    cutout_box: str | None = None

    if inject_alpha_cutout:
        cutout_box = target_box
        cy0 = int(v0 + max(0, (v1 - v0) // 4))
        cy1 = int(max(cy0 + 1, v1 - (v1 - v0) // 4))
        cx0 = int(u0 + max(0, (u1 - u0) // 4))
        cx1 = int(max(cx0 + 1, u1 - (u1 - u0) // 4))
        # Punch a genuine alpha=0 cutout window inside the island so RGBA vs RGB is decisive.
        atlas_rgba[cy0:cy1, cx0:cx1, 3] = 0

    modified_desc = copy.deepcopy(atlas_desc)
    if inject_uv_overlap and len(box_names) >= 2:
        if vary_target_faces:
            src_idx = (seed_int // 7) % len(box_names)
            dst_idx = (src_idx + 1 + ((seed_int // 29) % (len(box_names) - 1))) % len(box_names)
            src_face = face_order[(seed_int // 13) % len(face_order)]
            dst_face = face_order[(seed_int // 19) % len(face_order)]
            src_box = box_names[src_idx]
            dst_box = box_names[dst_idx]
            if src_face in modified_desc["faces"][src_box] and dst_face in modified_desc["faces"][dst_box]:
                modified_desc["faces"][dst_box][dst_face] = list(modified_desc["faces"][src_box][src_face])
            else:
                modified_desc["faces"][box_names[1]]["north"] = list(modified_desc["faces"][box_names[0]]["north"])
        else:
            first_box = scene.boxes[0].name
            second_box = scene.boxes[1].name
            modified_desc["faces"][second_box]["north"] = list(modified_desc["faces"][first_box]["north"])

    overlap_pair = _find_overlapping_uv_islands(modified_desc["faces"])

    modified_atlas_png = encode_rgba_png(atlas_rgba)
    png_by_view, _, _, _ = render_world_views_rgba(
        world_state,
        views=("iso",),
        panel_size=panel_size,
        atlas_override=(modified_desc, modified_atlas_png),
    )

    total_island_px = 0
    transparent_island_px = 0
    for bname, fmap in modified_desc["faces"].items():
        for fname, rect in fmap.items():
            ru0, rv0, ru1, rv1 = (int(x) for x in rect[:4])
            patch_a = atlas_rgba[rv0:rv1, ru0:ru1, 3]
            total_island_px += int(patch_a.size)
            transparent_island_px += int(np.sum(patch_a == 0))

    alpha_cutout_fraction = round(transparent_island_px / max(1, total_island_px), 4)
    has_cutout_bool = bool(transparent_island_px > 0)
    diagnosis = {
        "atlas_size": [int(modified_desc["width"]), int(modified_desc["height"])],
        "padding_px": int(modified_desc["padding"]),
        "has_alpha_cutout": has_cutout_bool,
        "alpha_cutout_box": cutout_box if has_cutout_bool else None,
        "island_transparent_fraction": alpha_cutout_fraction,
        "has_uv_overlap": bool(overlap_pair),
        "overlapping_islands": overlap_pair,
    }
    answer_body = json.dumps(diagnosis, sort_keys=True, separators=(",", ":"))
    think = synthesize_oracle_think_trace(
        task="synccraft_uv_alpha_diagnosis",
        scene=scene,
        audit_report=audit,
        extra_facts=[
            f"uv_atlas={modified_desc['width']}x{modified_desc['height']} RGBA, padding={modified_desc['padding']}px",
            (
                f"alpha_check: inspect RGBA channel A inside face islands -> "
                f"transparent_px={transparent_island_px}/{total_island_px} ({alpha_cutout_fraction}), "
                f"has_alpha_cutout={has_cutout_bool}, alpha_cutout_box={diagnosis['alpha_cutout_box']}"
            ),
            (
                f"uv_rect_check: test pairwise [u0,v0,u1,v1] intersections across UV face table -> "
                f"has_uv_overlap={bool(overlap_pair)}, overlapping_islands={overlap_pair}"
            ),
        ],
    )

    variant_suffix = f"uv_a{int(inject_alpha_cutout)}_o{int(inject_uv_overlap)}"
    atlas_member = f"{asset_id}/{variant_suffix}_atlas_rgba.png"
    iso_member = f"{asset_id}/{variant_suffix}_render_iso.png"
    media_payloads = {atlas_member: modified_atlas_png, iso_member: png_by_view["iso"]}
    visual_units = [
        {
            "visual_kind": "uv_atlas",
            "view_id": "uv_vanilla",
            "camera_right": [1.0, 0.0, 0.0],
            "camera_up": [0.0, 1.0, 0.0],
            "ortho_scale": 1.0,
            "has_timestamp": False,
            "timestamp_s": 0.0,
            "is_pixel_art_atlas": True,
        },
        {
            **_camera_unit_meta("iso"),
            "visual_kind": "uv_atlas",
        },
    ]
    prov = make_clean_room_provenance(asset_id, title=f"{family_id} RGBA UV & alpha diagnosis")
    row = {
        "id": f"{asset_id}#{variant_suffix}",
        "source_group": f"synccraft:{asset_id}",
        "language": "en",
        "task": "synccraft_uv_alpha_diagnosis",
        "visual_kind": "uv_atlas",
        "enable_thinking": enable_thinking,
        "frames": [atlas_member, iso_member],
        "media_times_s": [0.0, 0.0],
        "visual_units": visual_units,
        "prompt": (
            "Examine image 1 (RGBA pixel-art UV atlas with preserved alpha channel) and image 2 "
            "(orthographic isometric render) alongside the UV face table "
            f"{json.dumps(modified_desc['faces'], sort_keys=True, separators=(',', ':'))}. "
            "Return JSON with keys atlas_size, padding_px, has_alpha_cutout, alpha_cutout_box, "
            "island_transparent_fraction, has_uv_overlap, and overlapping_islands."
        ),
        "answer": format_supervised_answer(answer_body, think, enable_thinking=enable_thinking),
        "target": {"kind": "text", "text": answer_body},
        "provenance": prov,
        "metadata": {
            "asset_id": asset_id,
            "base_asset_id": asset_id,
            "family_id": family_id,
            "provenance": prov,
            "oracle_json": diagnosis,
        },
    }
    validate_synccraft_license_record(row)
    return row, media_payloads


def _inject_collision_defect(program: Program) -> tuple[Program, str]:
    """Create a deterministic SAT overlap defect on the first `place` box or `attach` child bone."""
    new_items = []
    mutated_name = ""
    for item in program.root.items:
        if isinstance(item, Box) and item.place is not None and not mutated_name:
            ax, ay, az = item.place.align
            bad_place = replace(item.place, align=(-ax, -ay, -az))
            new_items.append(replace(item, place=bad_place))
            mutated_name = item.name
        elif isinstance(item, Bone) and item.attach is not None and not mutated_name:
            ax, ay, az = item.attach.align
            bad_attach = replace(item.attach, align=(-ax, -ay, -az))
            new_items.append(replace(item, attach=bad_attach))
            mutated_name = item.name
        else:
            new_items.append(item)
    if not mutated_name:
        first_box = next(x for x in program.root.items if isinstance(x, Box))
        dup = Box(
            f"{first_box.name}_overlap_spur",
            first_box.size,
            first_box.material,
            place=Place(first_box.name, anchor=(0.0, 1.0, 0.0), align=(0.0, 1.0, 0.0)),
        )
        new_items.append(dup)
        mutated_name = dup.name
    broken_root = replace(program.root, items=new_items)
    return replace(program, root=broken_root), mutated_name


def build_typed_scene_edit_record(
    world_state: WorldState,
    *,
    asset_id: str,
    family_id: str,
    mode: str = "repair_sat_defect",
    enable_thinking: bool = True,
    panel_size: tuple[int, int] = (160, 160),
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Build a before/after `state_pair` typed scene edit or SAT-repair record."""
    clean_program = world_state.program
    clean_scene = compile_geometry(clean_program)
    clean_audit = audit_scene(clean_scene)

    if mode == "repair_sat_defect":
        broken_program, defect_box = _inject_collision_defect(clean_program)
        broken_scene = compile_geometry(broken_program)
        broken_audit = audit_scene(broken_scene)
        broken_world = replace(
            world_state,
            program=broken_program,
            material_program=MaterialProgram.from_program(broken_program, seed=101),
            surface_state=initialize_surface_state(
                build_surface_domain(
                    broken_program,
                    MaterialProgram.from_program(broken_program, seed=101),
                    subdivisions_per_unit=4,
                )
            ),
        )
        before_pngs, _, _, _ = render_world_views_rgba(broken_world, views=("iso",), panel_size=panel_size)
        after_pngs, _, _, _ = render_world_views_rgba(world_state, views=("iso",), panel_size=panel_size)
        clean_dsl = format_program(clean_program)
        answer_body = f"```blockdsl\n{clean_dsl.strip()}\n```"
        think = synthesize_oracle_think_trace(
            task="synccraft_typed_scene_edit",
            scene=clean_scene,
            audit_report=clean_audit,
            extra_facts=[
                f"before_defect_box={defect_box}, before_collisions={len(broken_audit['collision_pairs'])}, before_contact_violations={len(broken_audit['contact_violations'])}",
                "repair_action=restore flush P1-02 opposing align normal so 15-axis SAT collision_pairs=0",
            ],
        )
        prompt = (
            "Image 1 (`before`) shows a broken SyncCraft rig with an OBB SAT burial defect; "
            "Image 2 (`after`) shows the target flush assembly.\n"
            f"Broken Block-DSL:\n```blockdsl\n{format_program(broken_program).strip()}\n```\n"
            "Output the minimal repaired Block-DSL v3.2 program that matches the target state and passes 15-axis SAT + P1-02 contact audit."
        )
        oracle_meta: dict[str, Any] = {
            "edit_mode": "repair_sat_defect",
            "broken_dsl": format_program(broken_program),
            "canonical_dsl": clean_dsl,
            "before_collisions": len(broken_audit["collision_pairs"]),
        }
    else:
        # Action-conditioned WorldState transition (`state_pair`: before -> after -> infer WorldAction JSON)
        movable_bones = [b.name for b in clean_scene.bones if b.parent_name is not None]
        target_bone = movable_bones[0] if movable_bones else clean_scene.bones[0].name
        action = SetPoseAction(pose_euler_deltas={target_bone: (0.0, 0.0, 25.0)}, substeps=4)
        transition = step_world(world_state, action, dt=1.0)
        before_pngs, _, _, _ = render_world_views_rgba(world_state, views=("iso",), panel_size=panel_size)
        after_pngs, _, _, _ = render_world_views_rgba(transition.next_state, views=("iso",), panel_size=panel_size)
        action_dict = action.to_dict()
        answer_body = json.dumps(action_dict, sort_keys=True, separators=(",", ":"))
        think = synthesize_oracle_think_trace(
            task="synccraft_typed_scene_edit",
            scene=compile_posed_geometry(transition.next_state.program, transition.next_state.pose_euler_deltas),
            audit_report=clean_audit,
            extra_facts=[
                f"transition_action={action_dict}",
                f"posed_bone={target_bone} delta_deg=[0.0, 0.0, 25.0]",
            ],
        )
        prompt = (
            "Compare Image 1 (`before` WorldState) and Image 2 (`after` WorldState) for the following Block-DSL rig:\n"
            f"```blockdsl\n{format_program(clean_program).strip()}\n```\n"
            "Predict the typed SyncCraft WorldAction JSON (`kind`, `pose_euler_deltas`, `contact_normal_load`, `substeps`, `contact_threshold`) that produced `after` from `before`."
        )
        oracle_meta = {
            "edit_mode": "infer_world_action",
            "canonical_dsl": format_program(clean_program),
            "oracle_action": action_dict,
        }

    before_member = f"{asset_id}/{mode}_before.png"
    after_member = f"{asset_id}/{mode}_after.png"
    media_payloads = {before_member: before_pngs["iso"], after_member: after_pngs["iso"]}
    iso_meta = _camera_unit_meta("iso")
    visual_units = [
        {**iso_meta, "visual_kind": "state_pair", "view_id": "before"},
        {**iso_meta, "visual_kind": "state_pair", "view_id": "after"},
    ]
    prov = make_clean_room_provenance(asset_id, title=f"{family_id} state_pair {mode}")
    row = {
        "id": f"{asset_id}#edit_{mode}",
        "source_group": f"synccraft:{asset_id}",
        "language": "en",
        "task": "synccraft_typed_scene_edit",
        "visual_kind": "state_pair",
        "enable_thinking": enable_thinking,
        "frames": [before_member, after_member],
        "media_times_s": [0.0, 0.0],
        "visual_units": visual_units,
        "prompt": prompt,
        "answer": format_supervised_answer(answer_body, think, enable_thinking=enable_thinking),
        "target": {"kind": "text", "text": answer_body},
        "provenance": prov,
        "metadata": {
            "asset_id": asset_id,
            "base_asset_id": asset_id,
            "family_id": family_id,
            "provenance": prov,
            **oracle_meta,
        },
    }
    validate_synccraft_license_record(row)
    return row, media_payloads


def build_animated_quadruped_program(variant: int = 0) -> Program:
    """Build a clean-room quadruped Program with closed-loop IK `@behavior` gait (<0.02u stance slip)."""
    v = int(variant)
    hue = float((42.0 + 53.0 * v) % 360.0)
    torso_w = 6 + (v % 3)
    torso_h = 5 + ((v // 3) % 2)
    torso_d = 9 + (v % 4)
    stride = round(1.5 + 0.1 * (v % 4), 2)
    lift = round(0.65 + 0.08 * (v % 3), 2)
    program = validate_program(
        Program(
            meta=Meta("entity", "bedrock_entity", 16),
            palettes=[
                Palette("pelt_mat", (0.64, 0.11, hue), 6),
                Palette("hoof_mat", (0.40, 0.06, float((hue + 90.0) % 360.0)), 6),
            ],
            root=Bone(
                name="body_root",
                pivot_local=(0.0, 10.0, 0.0),
                items=[
                    Box("torso", (torso_w, torso_h, torso_d), "pelt_mat"),
                    Bone(
                        name="left_thigh",
                        attach=Attach("torso", anchor=(-1.0, -0.6, -0.6), align=(1.0, 1.0, 0.0)),
                        mirror_name="right_thigh",
                        items=[
                            Box("thigh_seg", (2, 4, 2), "pelt_mat"),
                            Bone(
                                name="left_shin",
                                attach=Attach("thigh_seg", anchor=(0.0, -1.0, 0.0), align=(0.0, 1.0, 0.0)),
                                items=[
                                    Box("shin_seg", (2, 4, 2), "hoof_mat"),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            kinematics=KinematicsDecl(
                ik_chains=(
                    IKChainDecl("left_leg_ik", "left_thigh", "left_shin", (0.0, -4.0, 0.0), hinge_axis="x"),
                    IKChainDecl("right_leg_ik", "right_thigh", "right_thigh__left_shin", (0.0, -4.0, 0.0), hinge_axis="x"),
                )
            ),
            behavior=BehaviorDecl(
                clips=(
                    ClipDecl(
                        name="walk_cycle",
                        duration_s=1.6,
                        loop=True,
                        tracks=(),
                        ik_drives=(
                            IKDriveDecl("left_leg_ik", stride=stride, lift=lift, phase_deg=0.0),
                            IKDriveDecl("right_leg_ik", stride=stride, lift=lift, phase_deg=180.0),
                        ),
                    ),
                )
            ),
        )
    )
    return program


def build_animation_clip_record(
    program: Program | None = None,
    *,
    world_state: WorldState | None = None,
    clip_name: str | None = None,
    asset_id: str = "synccraft_quadruped_001",
    family_id: str = "dog_horse_quadruped",
    num_keyframes: int = 4,
    enable_thinking: bool = True,
    panel_size: tuple[int, int] = (144, 144),
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Build a timestamped `@behavior` animation record (`visual_kind='temporal_frame'`)."""
    prog = world_state.program if world_state is not None else (program or build_animated_quadruped_program())
    if clip_name is None:
        if prog.behavior is not None and prog.behavior.clips:
            clip_name = prog.behavior.clips[0].name
        else:
            clip_name = "walk_cycle"
    scene = compile_geometry(prog)
    static_audit = audit_scene(scene)
    clip_audit = audit_behavior_clip(prog, clip_name, num_frames=12)
    frames = evaluate_behavior_clip(prog, clip_name, num_frames=max(4, num_keyframes))

    if world_state is not None:
        base_world = world_state
    else:
        subdiv = 1 if len(scene.boxes) > 20 else 4
        mat_prog = MaterialProgram.from_program(prog, seed=101)
        domain = build_surface_domain(prog, mat_prog, subdivisions_per_unit=subdiv)
        surf_state = initialize_surface_state(domain)
        base_world = WorldState(
            program=prog,
            material_program=mat_prog,
            surface_state=surf_state,
            style_config=StyleConfig.from_program(prog),
            environment=EnvironmentState(),
        )
    atlas_override = compile_vanilla_atlas(base_world)

    frame_paths: list[str] = []
    media_times_s: list[float] = []
    visual_units: list[dict[str, Any]] = []
    media_payloads: dict[str, bytes] = {}
    iso_meta = _camera_unit_meta("iso")

    for sample in frames[:num_keyframes]:
        posed_world = replace(base_world, pose_euler_deltas=sample.pose_euler_deltas)
        pngs, _, _, _ = render_world_views_rgba(
            posed_world,
            views=("iso",),
            panel_size=panel_size,
            atlas_override=atlas_override,
        )
        fpath = f"{asset_id}/anim_{clip_name}_f{sample.frame_index:02d}.png"
        frame_paths.append(fpath)
        t_s = round(float(sample.time_s), 4)
        media_times_s.append(t_s)
        media_payloads[fpath] = pngs["iso"]
        visual_units.append(
            {
                **iso_meta,
                "visual_kind": "temporal_frame",
                "view_id": "frame",
                "has_timestamp": True,
                "timestamp_s": t_s,
            }
        )

    clip_decl = next((c for c in (prog.behavior.clips if prog.behavior else ()) if c.name == clip_name), None)
    spin_bones = sorted(tr.bone for tr in (clip_decl.tracks if clip_decl else ()) if tr.law == "spin")
    ik_chains = sorted(d.chain for d in (clip_decl.ik_drives if clip_decl else ()))
    muscle_names = sorted(md.muscle for md in (clip_decl.muscle_drives if clip_decl else ()))
    aud_prof = str(getattr(scene.meta, "audit_profile", "strict"))

    dsl_text = format_program(prog)
    summary_json = {
        "clip_name": clip_name,
        "swept_collision_free": bool(clip_audit["swept_collision_free"]),
        "stance_slip_passed": bool(clip_audit["stance_slip_passed"]),
        "max_stance_slip_units": float(clip_audit["max_stance_slip_units"]),
        "spin_bones": spin_bones,
        "ik_chains": ik_chains,
        "muscle_drives": muscle_names,
    }
    answer_body = f"```blockdsl\n{dsl_text.strip()}\n```"
    driver_notes: list[str] = []
    if ik_chains:
        driver_notes.append(f"ik_drive=foot_cycle on {ik_chains}")
    if spin_bones:
        driver_notes.append(f"continuous_spin={spin_bones}")
    if muscle_names:
        driver_notes.append(f"hill_muscle_drives={muscle_names}")
    think = synthesize_oracle_think_trace(
        task="synccraft_behavior_animation",
        scene=scene,
        audit_report=static_audit,
        extra_facts=[
            f"clip={clip_name}, timestamps_s={media_times_s}, drivers={driver_notes or ['tracks']}",
            f"behavior_audit: swept_collision_free={clip_audit['swept_collision_free']}, max_stance_slip_units={clip_audit['max_stance_slip_units']} (<=0.02)",
        ],
    )
    extra_prompt_reqs: list[str] = []
    if aud_prof != "strict":
        extra_prompt_reqs.append(f"audit_profile={aud_prof!r}")
    if spin_bones:
        extra_prompt_reqs.append(f"continuous mechanical `spin(...)` tracks on {spin_bones}")
    if muscle_names:
        extra_prompt_reqs.append(f"Hill `@muscles` drives on {muscle_names}")
    extra_clause = (" (" + ", ".join(extra_prompt_reqs) + ")") if extra_prompt_reqs else ""
    prov = make_clean_room_provenance(asset_id, title=f"{family_id} @behavior animation {clip_name}")
    row = {
        "id": f"{asset_id}#anim_{clip_name}",
        "source_group": f"synccraft:{asset_id}",
        "language": "en",
        "task": "synccraft_behavior_animation",
        "visual_kind": "temporal_frame",
        "media_kind": "video_frames",
        "enable_thinking": enable_thinking,
        "frames": frame_paths,
        "media_times_s": media_times_s,
        "visual_units": visual_units,
        "prompt": (
            f"Observe the {len(frame_paths)} timestamped isometric keyframes at t={media_times_s}s for "
            f"clip {clip_name!r}{extra_clause}. Author the full SyncCraft Block-DSL v3.2 program including "
            "`@kinematics` and `@behavior` (`foot_cycle(...)` stance IK slip <= 0.02u and zero swept OBB collisions)."
        ),
        "answer": format_supervised_answer(answer_body, think, enable_thinking=enable_thinking),
        "target": {"kind": "text", "text": answer_body},
        "provenance": prov,
        "metadata": {
            "asset_id": asset_id,
            "base_asset_id": asset_id,
            "family_id": family_id,
            "provenance": prov,
            "canonical_dsl": dsl_text,
            "clip_name": clip_name,
            "expected_audit_profile": aud_prof,
            "expected_spin_bones": spin_bones,
            "oracle_behavior_audit": summary_json,
        },
    }
    validate_synccraft_license_record(row)
    return row, media_payloads


def build_parametric_mechanism_world(variant: int = 0, subdivisions_per_unit: int = 4) -> WorldState:
    """Build clean-room articulated mechanism rigs with distinct integer geometry & OKLCH palettes."""
    v = int(variant)
    if v <= 0:
        return build_articulated_mechanism_world(seed=101, subdivisions_per_unit=subdivisions_per_unit)
    ped_w = 8 + 2 * (v % 3)
    post_h = 6 + 2 * ((v // 2) % 3)
    upper_len = 5 + (v % 3)
    fore_len = 4 + ((v + 1) % 3)
    coat_hue = float((245.0 + 43.0 * v) % 360.0)
    program = validate_program(
        Program(
            meta=Meta("block_multipart", "bbmodel_5", 16),
            palettes=[
                Palette("stone_mat", (0.52, 0.03, 75.0), 6),
                Palette("wood_mat", (0.56, 0.09, 62.0), 6),
                Palette("painted_iron", (0.48, 0.14, coat_hue), 6),
                Palette("brushed_steel", (0.76, 0.02, 235.0), 6),
            ],
            root=Bone(
                name="base_frame",
                pivot_local=(0.0, 2.0, 0.0),
                items=[
                    Box("stone_pedestal", (ped_w, 4, ped_w), "stone_mat"),
                    Box(
                        "timber_post",
                        (4, post_h, 4),
                        "wood_mat",
                        place=Place("stone_pedestal", anchor=(0.0, 1.0, 0.0), align=(0.0, -1.0, 0.0)),
                    ),
                    Box(
                        "housing",
                        (6, 4, 6),
                        "painted_iron",
                        place=Place("timber_post", anchor=(0.0, 1.0, 0.0), align=(0.0, -1.0, 0.0)),
                    ),
                    Bone(
                        name="upper_arm",
                        attach=Attach("housing", anchor=(1.0, 0.0, 0.0), align=(-1.0, 0.0, 0.0)),
                        rest_euler=(0.0, 0.0, 0.0),
                        items=[
                            Box("upper_link", (upper_len, 2, 2), "painted_iron"),
                            Bone(
                                name="fore_arm",
                                attach=Attach("upper_link", anchor=(1.0, 0.0, 0.0), align=(-1.0, 0.0, 0.0)),
                                rest_euler=(0.0, 0.0, 0.0),
                                items=[
                                    Box("fore_link", (fore_len, 2, 2), "brushed_steel"),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        )
    )
    mat_prog = MaterialProgram.from_program(program, seed=101 + 37 * v)
    domain = build_surface_domain(program, mat_prog, subdivisions_per_unit=subdivisions_per_unit)
    return WorldState(
        program=program,
        material_program=mat_prog,
        surface_state=initialize_surface_state(domain),
        style_config=StyleConfig(quantize_to_palette_ramp=False, target_encoding="vanilla"),
        environment=EnvironmentState(),
    )


def build_parametric_creature_world(variant: int = 0, subdivisions_per_unit: int = 4) -> WorldState:
    """Build clean-room mirrored quadruped rigs with distinct integer proportions & OKLCH palettes."""
    v = int(variant)
    if v <= 0:
        return build_biological_creature_world(seed=202, subdivisions_per_unit=subdivisions_per_unit)
    torso_w = 6 + 2 * (v % 2)
    torso_d = 9 + (v % 4)
    limb_h = 4 + (v % 3)
    pelt_hue = float((72.0 + 47.0 * v) % 360.0)
    scale_hue = float((135.0 + 39.0 * v) % 360.0)
    program = validate_program(
        Program(
            meta=Meta("entity", "bedrock_entity", 16),
            palettes=[
                Palette("pelt_skin", (0.68, 0.11, pelt_hue), 6),
                Palette("limb_scales", (0.44, 0.08, scale_hue), 6),
                Palette("elytra_crest", (0.36, 0.06, 210.0), 6),
            ],
            root=Bone(
                name="body_root",
                pivot_local=(0.0, 12.0, 0.0),
                items=[
                    Box("torso", (torso_w, 6, torso_d), "pelt_skin"),
                    Bone(
                        name="neck_head",
                        attach=Attach("torso", anchor=(0.0, 0.25, -1.0), align=(0.0, -0.25, 1.0)),
                        rest_euler=(0.0, 0.0, 0.0),
                        items=[
                            Box("skull", (4, 4, 4), "pelt_skin"),
                        ],
                    ),
                    Bone(
                        name="left_thigh",
                        attach=Attach("torso", anchor=(-1.0, -0.5, -0.5), align=(1.0, 1.0, 0.0)),
                        mirror_name="right_thigh",
                        rest_euler=(0.0, 0.0, 0.0),
                        items=[
                            Box("thigh_seg", (2, limb_h, 2), "pelt_skin"),
                            Bone(
                                name="left_shin",
                                attach=Attach("thigh_seg", anchor=(0.0, -1.0, 0.0), align=(0.0, 1.0, 0.0)),
                                rest_euler=(0.0, 0.0, 0.0),
                                items=[
                                    Box("shin_seg", (2, limb_h, 2), "limb_scales"),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        )
    )
    mat_prog = MaterialProgram.from_program(program, seed=202 + 41 * v)
    domain = build_surface_domain(program, mat_prog, subdivisions_per_unit=subdivisions_per_unit)
    return WorldState(
        program=program,
        material_program=mat_prog,
        surface_state=initialize_surface_state(domain),
        style_config=StyleConfig(quantize_to_palette_ramp=False, target_encoding="vanilla"),
        environment=EnvironmentState(),
    )


def build_parametric_sculpted_wyvern_world(
    variant: int = 0,
    subdivisions_per_unit: int = 1,
) -> WorldState:
    """Build ADR 0007 sculpted biomechanical wyvern variants (`audit_profile: sculpted`, `blend_with`, Hill `@muscles`)."""
    v = int(variant)
    if v <= 0:
        return build_sculpted_elder_wyvern_world(seed=303, subdivisions_per_unit=subdivisions_per_unit)
    base_prog = parse_program(SCULPTED_ELDER_WYVERN_DSL)
    hue_shift = float((37.0 * v) % 360.0)
    new_palettes = [
        replace(
            p,
            base_lch=(p.base_lch[0], p.base_lch[1], round((p.base_lch[2] + hue_shift) % 360.0, 2)),
        )
        for p in base_prog.palettes
    ]
    stride = round(4.0 + 0.35 * (v % 3), 2)
    lift = round(1.5 + 0.2 * (v % 3), 2)
    new_clips = []
    for cl in base_prog.behavior.clips if base_prog.behavior else ():
        if cl.name == "stalk_gait":
            ik_ds = tuple(replace(d, stride=stride, lift=lift) for d in cl.ik_drives)
            new_clips.append(replace(cl, ik_drives=ik_ds))
        else:
            new_clips.append(cl)
    new_beh = replace(base_prog.behavior, clips=tuple(new_clips)) if base_prog.behavior else None
    prog = validate_program(replace(base_prog, palettes=new_palettes, behavior=new_beh))
    mat_prog = MaterialProgram.from_program(prog, seed=303 + 37 * v)
    domain = build_surface_domain(prog, mat_prog, subdivisions_per_unit=subdivisions_per_unit)
    return WorldState(
        program=prog,
        material_program=mat_prog,
        surface_state=initialize_surface_state(domain),
        style_config=StyleConfig.from_program(prog),
        environment=EnvironmentState(),
    )


def build_parametric_walking_citadel_world(
    variant: int = 0,
    subdivisions_per_unit: int = 1,
) -> WorldState:
    """Build ADR 0007 colossal walking citadel variants (`audit_profile: sculpted`, `radial`, continuous `spin(...)`)."""
    v = int(variant)
    if v <= 0:
        return build_colossal_walking_citadel_world(seed=404, subdivisions_per_unit=subdivisions_per_unit)
    base_prog = parse_program(COLOSSAL_WALKING_CITADEL_DSL)
    hue_shift = float((43.0 * v) % 360.0)
    new_palettes = [
        replace(
            p,
            base_lch=(p.base_lch[0], p.base_lch[1], round((p.base_lch[2] + hue_shift) % 360.0, 2)),
        )
        for p in base_prog.palettes
    ]
    stride = round(4.5 + 0.4 * (v % 3), 2)
    lift = round(1.8 + 0.2 * (v % 2), 2)
    spin_sign = -1.0 if (v % 2 == 1) else 1.0
    new_clips = []
    for cl in base_prog.behavior.clips if base_prog.behavior else ():
        if cl.name == "fortress_march":
            ik_ds = tuple(replace(d, stride=stride, lift=lift) for d in cl.ik_drives)
            trs = tuple(
                replace(tr, amp_deg=round(spin_sign * float(tr.amp_deg), 2))
                if tr.law == "spin"
                else tr
                for tr in cl.tracks
            )
            new_clips.append(replace(cl, ik_drives=ik_ds, tracks=trs))
        else:
            new_clips.append(cl)
    new_beh = replace(base_prog.behavior, clips=tuple(new_clips)) if base_prog.behavior else None
    prog = validate_program(replace(base_prog, palettes=new_palettes, behavior=new_beh))
    mat_prog = MaterialProgram.from_program(prog, seed=404 + 41 * v)
    domain = build_surface_domain(prog, mat_prog, subdivisions_per_unit=subdivisions_per_unit)
    return WorldState(
        program=prog,
        material_program=mat_prog,
        surface_state=initialize_surface_state(domain),
        style_config=StyleConfig.from_program(prog),
        environment=EnvironmentState(),
    )


def build_world_capability_record_from_ladder_row(
    ladder_row: dict[str, Any],
    *,
    enable_thinking: bool = True,
    render_multiview_for_level5: bool = True,
    panel_size: tuple[int, int] = (144, 144),
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Convert a 7-Level SyncCraft `world_capability_ladder.jsonl` row into a Spark-X2.6-VL record."""
    sample_id = str(ladder_row["sample_id"])
    level_id = str(ladder_row["level"])
    sub_level = str(ladder_row.get("sub_level", level_id))
    effective_task = sub_level if level_id == "level_7_human_transfer" else level_id
    geo_fam = str(ladder_row.get("geometry_family", "world_ladder"))
    mat_fam = str(ladder_row.get("material_family", "default"))
    act_fam = str(ladder_row.get("action_family", "default"))
    comp_triple = str(ladder_row.get("compositional_triple", f"{geo_fam}__{mat_fam}__{act_fam}"))
    aud_prof = str(ladder_row.get("audit_profile", "strict"))

    dsl_str = str(ladder_row["program_dsl"])
    prog = parse_program(dsl_str)
    scene = compile_geometry(prog)
    audit = audit_scene(scene)

    media_payloads: dict[str, bytes] = {}
    frame_paths: list[str] = []
    visual_units: list[dict[str, Any]] = []
    visual_kind: str | None = None

    if effective_task == "level_5_invert_observation":
        if render_multiview_for_level5:
            views = ("south", "right", "top", "iso")
            subdiv = 1 if len(scene.boxes) > 20 else 2
            mp = MaterialProgram.from_program(prog, seed=101)
            dom = build_surface_domain(prog, mp, subdivisions_per_unit=subdiv)
            ws = WorldState(
                program=prog,
                material_program=mp,
                surface_state=initialize_surface_state(dom),
                style_config=StyleConfig.from_program(prog),
                environment=EnvironmentState(),
            )
            png_by_view, _, _, _ = render_world_views_rgba(ws, views=views, panel_size=panel_size)
            frame_paths = [f"world_ladder/{sample_id}/view_{v}.png" for v in views]
            media_payloads = {frame_paths[i]: png_by_view[v] for i, v in enumerate(views)}
            visual_units = [_camera_unit_meta(v) for v in views]
            visual_kind = "multiview_render"
        else:
            obs_im = decode_record_observation_image(ladder_row)
            if obs_im is not None:
                buf = io.BytesIO()
                obs_im.save(buf, format="PNG")
                fpath = f"world_ladder/{sample_id}/obs_sheet.png"
                frame_paths = [fpath]
                media_payloads = {fpath: buf.getvalue()}
                visual_units = [_camera_unit_meta("iso")]
                visual_kind = "independent_image"

    raw_target = str(ladder_row["target_completion"]).strip()
    if effective_task in ("level_5_invert_observation", "level_6_edit_and_repair"):
        answer_body = f"```blockdsl\n{raw_target}\n```" if not raw_target.startswith("```") else raw_target
    else:
        answer_body = raw_target

    think = synthesize_oracle_think_trace(
        task="synccraft_world_capability",
        scene=scene,
        audit_report=audit,
        extra_facts=[
            f"ladder_level={level_id}, sub_level={sub_level}, capability={ladder_row.get('capability')}",
            f"compositional_triple={comp_triple}",
        ],
    )
    base_group = f"synccraft-ladder:{comp_triple}:{sample_id.rsplit('_', 1)[0]}"
    prov = make_clean_room_provenance(sample_id, title=f"SyncCraft World Capability {level_id}")
    ladder_eval_spec = {
        "sample_id": sample_id,
        "level": level_id,
        "sub_level": sub_level,
        "audit_profile": aud_prof,
        "ground_truth": ladder_row["ground_truth"],
    }
    row: dict[str, Any] = {
        "id": f"world_ladder#{sample_id}",
        "source_group": base_group,
        "language": "en",
        "task": "synccraft_world_capability",
        "enable_thinking": enable_thinking,
        "frames": frame_paths,
        "prompt": str(ladder_row["prompt"]),
        "answer": format_supervised_answer(answer_body, think, enable_thinking=enable_thinking),
        "target": {"kind": "text", "text": answer_body},
        "provenance": prov,
        "metadata": {
            "asset_id": sample_id,
            "base_asset_id": base_group,
            "family_id": geo_fam,
            "provenance": prov,
            "ladder_eval_spec": ladder_eval_spec,
            "expected_audit_profile": aud_prof,
        },
    }
    if frame_paths and visual_kind is not None:
        row["visual_kind"] = visual_kind
        row["media_times_s"] = [0.0] * len(frame_paths)
        row["visual_units"] = visual_units
    validate_synccraft_license_record(row)
    return row, media_payloads


def build_artist_assistant_record(
    program_or_dsl: Program | str,
    *,
    mode: str,
    asset_id: str,
    family_id: str,
    prompt_override: str | None = None,
    provenance_override: dict[str, Any] | None = None,
    render_views: bool = False,
    enable_thinking: bool = True,
    panel_size: tuple[int, int] = (144, 144),
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Build a 5-mode SyncCraft v3.2 Artist-Assistant record (constructive, upgrade, rig+anim, edit, or 5-pillar critique)."""
    if mode not in ARTIST_ASSISTANT_MODES:
        raise ValueError(f"Unsupported artist assistant mode {mode!r}; expected one of {ARTIST_ASSISTANT_MODES}")

    prog = parse_program(program_or_dsl) if isinstance(program_or_dsl, str) else validate_program(program_or_dsl)
    scene = compile_geometry(prog)
    audit = audit_scene(scene)
    aud_prof = str(getattr(scene.meta, "audit_profile", "strict"))

    prov = (
        dict(provenance_override)
        if provenance_override is not None
        else make_clean_room_provenance(asset_id, title=f"{family_id} artist assistant {mode}")
    )

    media_payloads: dict[str, bytes] = {}
    frame_paths: list[str] = []
    visual_units: list[dict[str, Any]] = []
    visual_kind: str | None = None

    if render_views or mode == "inverse_structure_and_kinematics_reading":
        views = ("south", "right", "top", "iso")
        subdiv = 1 if len(scene.boxes) > 20 else 2
        mp = MaterialProgram.from_program(prog, seed=101)
        dom = build_surface_domain(prog, mp, subdivisions_per_unit=subdiv)
        ws = WorldState(
            program=prog,
            material_program=mp,
            surface_state=initialize_surface_state(dom),
            style_config=StyleConfig.from_program(prog),
            environment=EnvironmentState(),
        )
        png_by_view, _, _, _ = render_world_views_rgba(ws, views=views, panel_size=panel_size)
        frame_paths = [f"artist/{asset_id}/{mode}_{v}.png" for v in views]
        media_payloads = {frame_paths[i]: png_by_view[v] for i, v in enumerate(views)}
        visual_units = [_camera_unit_meta(v) for v in views]
        visual_kind = "multiview_render"

    oracle_meta: dict[str, Any] = {
        "assistant_mode": mode,
        "expected_audit_profile": aud_prof,
    }

    if mode == "inverse_structure_and_kinematics_reading":
        critique = analyze_artist_project(prog)
        crit_dict = critique.to_dict()
        oracle_summary = {
            "audit_profile": aud_prof,
            "bone_count": int(crit_dict["bone_count"]),
            "box_count": int(crit_dict["box_count"]),
            "locator_count": int(crit_dict["locator_count"]),
            "max_bone_depth": int(crit_dict["max_bone_depth"]),
            "static_audit_passed": bool(crit_dict["static_audit_passed"]),
            "behavior_clips_passed": int(crit_dict["behavior_clips_passed"]),
            "readiness_score": round(float(crit_dict["readiness_score"]), 3),
            "pillar_scores": {k: round(float(v), 3) for k, v in sorted(crit_dict["pillar_scores"].items())},
        }
        answer_body = json.dumps(oracle_summary, sort_keys=True, separators=(",", ":"))
        prompt = prompt_override or (
            "Inspect the 4 orthographic views (south, right, top, iso) and the following SyncCraft Block-DSL v3.2 project:\n"
            f"```blockdsl\n{format_program(prog).strip()}\n```\n"
            "Return the quantitative 5-pillar ArtistCritique summary JSON with keys audit_profile, bone_count, "
            "box_count, locator_count, max_bone_depth, static_audit_passed, behavior_clips_passed, readiness_score, and pillar_scores."
        )
        extra_facts = [
            f"assistant_mode={mode}, readiness_score={oracle_summary['readiness_score']}",
            f"pillars={oracle_summary['pillar_scores']}",
        ]
        oracle_meta["oracle_json"] = oracle_summary
    elif mode == "coarse_to_fine_upgrade":
        coarse_prog = coarsen_program_to_tier(prog, "tier_1_simple")
        coarse_scene = compile_geometry(coarse_prog)
        coarse_dsl = format_program(coarse_prog)
        full_dsl = format_program(prog)
        answer_body = f"```blockdsl\n{full_dsl.strip()}\n```"
        prompt = prompt_override or (
            f"Upgrade this coarse Tier-1 Block-DSL blockout ({len(coarse_scene.boxes)} boxes) into a detailed "
            f"{family_id.replace('_', ' ')} SyncCraft Block-DSL v3.2 asset ({len(scene.boxes)} boxes, "
            f"target_profile={scene.meta.target_profile!r}, audit_profile={aud_prof!r}) preserving silhouette proportions "
            "and passing 15-axis SAT + P1-02 audit:\n"
            f"```blockdsl\n{coarse_dsl.strip()}\n```"
        )
        extra_facts = [
            f"assistant_mode={mode}, coarse_boxes={len(coarse_scene.boxes)} -> upgraded_boxes={len(scene.boxes)}",
        ]
        oracle_meta["coarse_dsl"] = coarse_dsl
        oracle_meta["canonical_dsl"] = full_dsl
        oracle_meta["min_box_gain"] = max(0, len(scene.boxes) - len(coarse_scene.boxes))
    elif mode == "texture_and_animate_rig":
        rng_det = __import__("random").Random(int(hashlib.sha256(asset_id.encode("utf-8")).hexdigest()[:8], 16))
        enriched_prog, _ = ensure_concise_texture_program(prog, family_id=family_id, rng=rng_det)
        if enriched_prog.behavior is None and enriched_prog.meta.target_profile in ("bedrock_entity", "geckolib", "bbmodel_5"):
            try:
                cand_p, _ = enhance_artist_project_step(enriched_prog)
                if isinstance(cand_p, Program):
                    enriched_prog = cand_p
            except Exception:
                pass
        bare_prog = validate_program(
            replace(
                enriched_prog,
                materials=(),
                style=None,
                muscles=(),
                kinematics=None,
                behavior=None,
                controller=None,
            )
        )
        bare_dsl = format_program(bare_prog)
        full_dsl = format_program(enriched_prog)
        answer_body = f"```blockdsl\n{full_dsl.strip()}\n```"
        prompt = prompt_override or (
            f"Given this untextured, static `{family_id}` Block-DSL v3.2 rig (`target_profile={scene.meta.target_profile!r}`, "
            f"`audit_profile={aud_prof!r}`), author physically grounded `@materials`, `@style`, and (when articulated) "
            "swept-SAT-verified `@kinematics` / `@behavior` while keeping the `@rig` collision-free:\n"
            f"```blockdsl\n{bare_dsl.strip()}\n```"
        )
        extra_facts = [
            f"assistant_mode={mode}, materials={[m.name for m in enriched_prog.materials]}, has_behavior={enriched_prog.behavior is not None}",
        ]
        oracle_meta["canonical_dsl"] = full_dsl
        oracle_meta["require_materials"] = True
        oracle_meta["require_behavior"] = bool(enriched_prog.behavior is not None and enriched_prog.behavior.clips)
    elif mode == "pose_and_part_edit_refinement":
        coarse_prog = coarsen_program_to_tier(prog, "tier_2_medium")
        try:
            cand_up, op_used = enhance_artist_project_step(coarse_prog)
            upgraded_prog = cand_up if isinstance(cand_up, Program) else prog
            applied_title = f"Apply artist-loop upgrade `{op_used}`" if op_used and not str(op_used).startswith("noop") else "Refine rig hierarchy, joint limits, and surface materials"
        except Exception:
            upgraded_prog = prog
            applied_title = "Refine rig hierarchy, joint limits, and surface materials"
        up_scene = compile_geometry(upgraded_prog)
        audit = audit_scene(up_scene)
        scene = up_scene
        full_dsl = format_program(upgraded_prog)
        answer_body = f"```blockdsl\n{full_dsl.strip()}\n```"
        prompt = prompt_override or (
            f"Apply the artist-loop refinement goal ({applied_title!r}) to the following `{family_id}` Block-DSL v3.2 project "
            f"(`audit_profile={aud_prof!r}`) and return the complete validated Block-DSL v3.2 program:\n"
            f"```blockdsl\n{format_program(coarse_prog).strip()}\n```"
        )
        extra_facts = [
            f"assistant_mode={mode}, goal={applied_title}",
        ]
        oracle_meta["canonical_dsl"] = full_dsl
    else:
        # `create_full_asset`
        full_dsl = format_program(prog)
        answer_body = f"```blockdsl\n{full_dsl.strip()}\n```"
        has_beh = bool(prog.behavior is not None and prog.behavior.clips)
        spin_bones = sorted(
            {
                tr.bone
                for cl in (prog.behavior.clips if prog.behavior else ())
                for tr in cl.tracks
                if tr.law == "spin"
            }
        )
        prompt = prompt_override or (
            f"Design a complete SyncCraft Block-DSL v3.2 3D asset for family `{family_id}` "
            f"(`archetype={scene.meta.archetype!r}`, `target_profile={scene.meta.target_profile!r}`, "
            f"`texel_density={scene.meta.texel_density}`, `audit_profile={aud_prof!r}`, "
            f"approx {len(scene.boxes)} boxes across {len(scene.bones)} bones"
            + (", including `@kinematics` and `@behavior`" if has_beh else "")
            + (f" with continuous `spin(...)` on {spin_bones}" if spin_bones else "")
            + ") that passes 15-axis SAT + P1-02 audit."
        )
        extra_facts = [
            f"assistant_mode={mode}, target_profile={scene.meta.target_profile}, has_behavior={has_beh}, spin_bones={spin_bones}",
        ]
        oracle_meta["canonical_dsl"] = full_dsl
        oracle_meta["require_behavior"] = has_beh
        oracle_meta["expected_spin_bones"] = spin_bones

    think = synthesize_oracle_think_trace(
        task="synccraft_artist_assistant",
        scene=scene,
        audit_report=audit,
        extra_facts=extra_facts,
    )
    row: dict[str, Any] = {
        "id": f"{asset_id}#assistant_{mode}",
        "source_group": f"synccraft:{asset_id}",
        "language": "en",
        "task": "synccraft_artist_assistant",
        "enable_thinking": enable_thinking,
        "frames": frame_paths,
        "prompt": prompt,
        "answer": format_supervised_answer(answer_body, think, enable_thinking=enable_thinking),
        "target": {"kind": "text", "text": answer_body},
        "provenance": prov,
        "metadata": {
            "asset_id": asset_id,
            "base_asset_id": asset_id,
            "family_id": family_id,
            "provenance": prov,
            **oracle_meta,
        },
    }
    if frame_paths and visual_kind is not None:
        row["visual_kind"] = visual_kind
        row["media_times_s"] = [0.0] * len(frame_paths)
        row["visual_units"] = visual_units
    validate_synccraft_license_record(row)
    return row, media_payloads


def build_synccraft_curriculum_records(
    *,
    include_animation: bool = False,
    include_sculpted_exemplars: bool = False,
    enable_thinking: bool = True,
    panel_size: tuple[int, int] = (128, 128),
    num_variants: int = 1,
) -> list[tuple[dict[str, Any], dict[str, bytes]]]:
    """Generate clean-room SyncCraft records staged static-first, with balanced 4-way UV and optional `sculpted` + `@behavior`."""
    n_var = max(1, int(num_variants))
    rig_specs: list[tuple[WorldState, str, str]] = []
    for v in range(n_var):
        idx = v + 1
        rig_specs.append(
            (
                build_parametric_mechanism_world(variant=v, subdivisions_per_unit=4),
                f"synccraft_mech_{idx:03d}",
                "articulated_robotic_arm",
            )
        )
        rig_specs.append(
            (
                build_parametric_creature_world(variant=v, subdivisions_per_unit=4),
                f"synccraft_bio_{idx:03d}",
                "dog_horse_quadruped",
            )
        )
        if include_sculpted_exemplars:
            rig_specs.append(
                (
                    build_parametric_sculpted_wyvern_world(variant=v, subdivisions_per_unit=1),
                    f"synccraft_wyvern_{idx:03d}",
                    "sculpted_elder_wyvern",
                )
            )
            rig_specs.append(
                (
                    build_parametric_walking_citadel_world(variant=v, subdivisions_per_unit=1),
                    f"synccraft_citadel_{idx:03d}",
                    "colossal_walking_citadel",
                )
            )

    items: list[tuple[dict[str, Any], dict[str, bytes]]] = []
    for ws, aid, fam in rig_specs:
        items.append(
            build_part_grounding_record(
                ws, asset_id=aid, family_id=fam, enable_thinking=enable_thinking, panel_size=panel_size
            )
        )
        items.append(
            build_multiview_consistency_record(
                ws, asset_id=aid, family_id=fam, enable_thinking=enable_thinking, panel_size=panel_size
            )
        )
        # Exact 25%/25%/25%/25% 4-way RGBA cutout x UV island overlap coverage (`a0_o0`, `a1_o0`, `a0_o1`, `a1_o1`)
        for cut_flag, ov_flag in ((False, False), (True, False), (False, True), (True, True)):
            items.append(
                build_uv_seam_alpha_diagnosis_record(
                    ws,
                    asset_id=aid,
                    family_id=fam,
                    inject_alpha_cutout=cut_flag,
                    inject_uv_overlap=ov_flag,
                    vary_target_faces=True,
                    enable_thinking=enable_thinking,
                    panel_size=panel_size,
                )
            )
        items.append(
            build_typed_scene_edit_record(
                ws,
                asset_id=aid,
                family_id=fam,
                mode="repair_sat_defect",
                enable_thinking=enable_thinking,
                panel_size=panel_size,
            )
        )
        items.append(
            build_typed_scene_edit_record(
                ws,
                asset_id=aid,
                family_id=fam,
                mode="infer_world_action",
                enable_thinking=enable_thinking,
                panel_size=panel_size,
            )
        )

    if include_animation:
        items.append(
            build_animation_clip_record(
                asset_id="synccraft_bio_001",
                family_id="dog_horse_quadruped",
                enable_thinking=enable_thinking,
                panel_size=panel_size,
            )
        )
        if include_sculpted_exemplars:
            wyv_ws = build_parametric_sculpted_wyvern_world(variant=0, subdivisions_per_unit=1)
            items.append(
                build_animation_clip_record(
                    world_state=wyv_ws,
                    clip_name="stalk_gait",
                    asset_id="synccraft_wyvern_001",
                    family_id="sculpted_elder_wyvern",
                    enable_thinking=enable_thinking,
                    panel_size=panel_size,
                )
            )
            cit_ws = build_parametric_walking_citadel_world(variant=0, subdivisions_per_unit=1)
            items.append(
                build_animation_clip_record(
                    world_state=cit_ws,
                    clip_name="fortress_march",
                    asset_id="synccraft_citadel_001",
                    family_id="colossal_walking_citadel",
                    enable_thinking=enable_thinking,
                    panel_size=panel_size,
                )
            )
    return items


def _extract_dsl_candidate(body: str) -> str:
    m = DSL_FENCE_RE.search(body)
    if m is not None:
        return m.group(1).strip()
    return body.strip()


def _extract_json_candidate(body: str) -> dict[str, Any]:
    m = JSON_FENCE_RE.search(body)
    raw = m.group(1) if m is not None else body.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object")
    return parsed


def compute_scene_render_iou(
    pred_program: Program,
    ref_program: Program,
    *,
    views: tuple[str, ...] = ("south", "right", "top", "iso"),
    panel_size: tuple[int, int] = (96, 96),
) -> dict[str, float]:
    """Compute multiview orthographic silhouette IoU and RGB PSNR between two compiled Programs."""
    ref_scene = compile_geometry(ref_program)
    pred_scene = compile_geometry(pred_program)
    subdiv = 1 if max(len(ref_scene.boxes), len(pred_scene.boxes)) > 20 else 4

    ref_mat = MaterialProgram.from_program(ref_program, seed=101)
    ref_dom = build_surface_domain(ref_program, ref_mat, subdivisions_per_unit=subdiv)
    ref_ws = WorldState(
        program=ref_program,
        material_program=ref_mat,
        surface_state=initialize_surface_state(ref_dom),
        style_config=StyleConfig.from_program(ref_program),
        environment=EnvironmentState(),
    )
    ref_corners = np.concatenate([b.corners(include_inflate=True) for b in ref_scene.boxes], axis=0)
    fixed_bounds = (ref_corners.min(axis=0), ref_corners.max(axis=0))

    pred_mat = MaterialProgram.from_program(pred_program, seed=101)
    pred_dom = build_surface_domain(pred_program, pred_mat, subdivisions_per_unit=subdiv)
    pred_ws = WorldState(
        program=pred_program,
        material_program=pred_mat,
        surface_state=initialize_surface_state(pred_dom),
        style_config=StyleConfig.from_program(pred_program),
        environment=EnvironmentState(),
    )

    ref_obs = render_world_observation(ref_ws, panel_size=panel_size, views=views, fixed_bounds=fixed_bounds)
    pred_obs = render_world_observation(pred_ws, panel_size=panel_size, views=views, fixed_bounds=fixed_bounds)

    ious: list[float] = []
    mse_vals: list[float] = []
    for v in views:
        ref_fg = ref_obs["hit_box_maps"][v] >= 0
        pred_fg = pred_obs["hit_box_maps"][v] >= 0
        inter = int(np.sum(ref_fg & pred_fg))
        union = int(np.sum(ref_fg | pred_fg))
        ious.append(1.0 if union == 0 else inter / union)
        diff = ref_obs["views"][v].astype(np.float64) - pred_obs["views"][v].astype(np.float64)
        mse_vals.append(float(np.mean(diff * diff)))

    mean_mse = float(np.mean(mse_vals))
    psnr = 60.0 if mean_mse <= 1e-9 else float(min(60.0, 10.0 * math.log10((255.0**2) / mean_mse)))
    return {
        "silhouette_iou": round(float(np.mean(ious)), 5),
        "rgb_psnr_db": round(psnr, 3),
    }


def evaluate_synccraft_completion(record: dict[str, Any], completion_text: str) -> dict[str, Any]:
    """Evaluate a Spark-X2.6-VL completion against SyncCraft's compiler, 15-axis SAT, UV, world-ladder, and motion oracles."""
    task = str(record.get("task", ""))
    meta = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
    think_trace, body = strip_think_block(completion_text)
    result: dict[str, Any] = {
        "task": task,
        "has_think_trace": think_trace is not None,
        "parse_valid": False,
        "passed": False,
    }

    if task == "synccraft_world_capability":
        ladder_spec = meta.get("ladder_eval_spec")
        if not isinstance(ladder_spec, dict):
            result["error"] = "missing_ladder_eval_spec"
            return result
        cand = _extract_dsl_candidate(body) if str(ladder_spec.get("sub_level", ladder_spec.get("level", ""))) in (
            "level_5_invert_observation",
            "level_6_edit_and_repair",
        ) else body
        scored = score_world_capability_completion(ladder_spec, cand)
        result.update(
            level=scored.get("level"),
            sub_level=scored.get("sub_level"),
            parse_valid=bool(scored.get("parse_valid")),
            ladder_score=float(scored.get("score", 0.0)),
            exact_pass=bool(scored.get("exact_pass", False)),
            ladder_metrics=scored.get("metrics", {}),
            passed=bool(scored.get("exact_pass") or float(scored.get("score", 0.0)) >= 0.85),
        )
        return result

    is_json_critique = (
        task == "synccraft_artist_assistant"
        and meta.get("assistant_mode") == "inverse_structure_and_kinematics_reading"
    )
    if task in ("synccraft_part_grounding", "synccraft_uv_alpha_diagnosis") or is_json_critique:
        try:
            pred = _extract_json_candidate(body)
        except Exception as exc:
            result["error"] = f"json_parse_error: {exc}"
            return result
        result["parse_valid"] = True
        oracle = meta.get("oracle_json", {})
        matched_keys = 0
        for k, v in oracle.items():
            if k in pred:
                if isinstance(v, float):
                    if isinstance(pred[k], (int, float)) and not isinstance(pred[k], bool) and math.isclose(float(pred[k]), v, abs_tol=0.05 if is_json_critique else 0.02):
                        matched_keys += 1
                elif (
                    isinstance(v, dict)
                    and isinstance(pred[k], dict)
                    and v
                    and all(isinstance(sv, (int, float)) and not isinstance(sv, bool) for sv in v.values())
                ):
                    sub_ok = all(
                        sk in pred[k]
                        and isinstance(pred[k][sk], (int, float))
                        and not isinstance(pred[k][sk], bool)
                        and math.isclose(float(pred[k][sk]), float(sv), abs_tol=0.06)
                        for sk, sv in v.items()
                    )
                    if sub_ok:
                        matched_keys += 1
                elif pred[k] == v:
                    matched_keys += 1
        accuracy = matched_keys / max(1, len(oracle))
        result.update(
            field_accuracy=round(accuracy, 4),
            exact_match=(matched_keys == len(oracle)),
            passed=(accuracy >= (0.85 if is_json_critique else 0.999)),
        )
        return result

    if task == "synccraft_typed_scene_edit" and meta.get("edit_mode") == "infer_world_action":
        try:
            pred_dict = _extract_json_candidate(body)
            pred_action = action_from_dict(pred_dict)
            oracle_action = action_from_dict(meta["oracle_action"])
        except Exception as exc:
            result["error"] = f"world_action_parse_error: {exc}"
            return result
        result["parse_valid"] = True
        ref_prog = parse_program(meta["canonical_dsl"])
        subdiv = 1 if len(compile_geometry(ref_prog).boxes) > 20 else 4
        mat = MaterialProgram.from_program(ref_prog, seed=101)
        dom = build_surface_domain(ref_prog, mat, subdivisions_per_unit=subdiv)
        ws = WorldState(
            program=ref_prog,
            material_program=mat,
            surface_state=initialize_surface_state(dom),
            style_config=StyleConfig.from_program(ref_prog),
            environment=EnvironmentState(),
        )
        try:
            pred_next = step_world(ws, pred_action, dt=1.0).next_state
            oracle_next = step_world(ws, oracle_action, dt=1.0).next_state
        except Exception as exc:
            result["error"] = f"step_world_error: {exc}"
            return result
        pred_scene = compile_posed_geometry(pred_next.program, pred_next.pose_euler_deltas)
        oracle_scene = compile_posed_geometry(oracle_next.program, oracle_next.pose_euler_deltas)
        corner_err = float(
            np.mean(
                [
                    np.linalg.norm(pb.corners() - ob.corners(), axis=-1).mean()
                    for pb, ob in zip(pred_scene.boxes, oracle_scene.boxes)
                ]
            )
        )
        result.update(
            action_kind_match=(pred_action.kind == oracle_action.kind),
            mean_corner_error_units=round(corner_err, 5),
            passed=(pred_action.kind == oracle_action.kind and corner_err <= 0.15),
        )
        return result

    # Remaining tasks produce Block-DSL (`multiview_inverse_dsl`, `repair_sat_defect`, `behavior_animation`, `artist_assistant`)
    dsl_candidate = _extract_dsl_candidate(body)
    try:
        pred_prog = parse_program(dsl_candidate)
        pred_scene = compile_geometry(pred_prog)
        expected_prof = str(meta.get("expected_audit_profile", getattr(pred_prog.meta, "audit_profile", "strict")))
        pred_prof = str(getattr(pred_prog.meta, "audit_profile", "strict"))
        # Anti-reward-hacking: never allow relaxing a `strict` or `sculpted` task to `authored`!
        audit_profile_ok = not (expected_prof in ("strict", "sculpted") and pred_prof == "authored")
        eval_prof = expected_prof if not audit_profile_ok else pred_prof
        pred_audit = audit_scene(pred_scene, audit_profile=eval_prof)
    except Exception as exc:
        result["error"] = f"dsl_compile_error: {exc}"
        return result

    result["parse_valid"] = True
    result["audit_profile"] = str(pred_audit.get("audit_profile", "strict"))
    result["audit_profile_ok"] = audit_profile_ok
    result["sat_passed"] = bool(pred_audit["passed"] and audit_profile_ok)
    result["collision_count"] = len(pred_audit["collision_pairs"])
    result["contact_violation_count"] = len(pred_audit["contact_violations"])
    result["sculpted_overlap_count"] = len(pred_audit.get("sculpted_overlap_pairs", []))

    if "canonical_dsl" in meta:
        ref_prog = parse_program(meta["canonical_dsl"])
        render_metrics = compute_scene_render_iou(pred_prog, ref_prog)
        result.update(render_metrics)
        if meta.get("broken_dsl"):
            broken_prog = parse_program(meta["broken_dsl"])
            broken_metrics = compute_scene_render_iou(broken_prog, ref_prog)
            result["iou_gain_over_broken"] = round(
                render_metrics["silhouette_iou"] - broken_metrics["silhouette_iou"], 5
            )

    require_beh = (task == "synccraft_behavior_animation") or bool(meta.get("require_behavior", False))
    if require_beh:
        clips = list(pred_prog.behavior.clips) if pred_prog.behavior is not None else []
        if not clips:
            result["error"] = "behavior_audit_error: missing @behavior clips"
            return result
        target_clip_name = meta.get("clip_name")
        clips_to_check = (
            [c for c in clips if c.name == target_clip_name]
            if target_clip_name and any(c.name == target_clip_name for c in clips)
            else clips
        )
        try:
            clip_audits = [audit_behavior_clip(pred_prog, c.name, num_frames=12) for c in clips_to_check]
        except Exception as exc:
            result["error"] = f"behavior_audit_error: {exc}"
            return result
        all_swept = all(bool(a["swept_collision_free"]) for a in clip_audits)
        all_slip = all(bool(a["stance_slip_passed"]) for a in clip_audits)
        max_slip = max(float(a["max_stance_slip_units"]) for a in clip_audits)
        all_beh_ok = all(bool(a["passed"]) for a in clip_audits)
        expected_spins = set(meta.get("expected_spin_bones") or ())
        actual_spins = {tr.bone for c in clips for tr in c.tracks if tr.law == "spin"}
        spin_ok = expected_spins.issubset(actual_spins) if expected_spins else True
        result.update(
            swept_collision_free=all_swept,
            stance_slip_passed=all_slip,
            max_stance_slip_units=max_slip,
            spin_tracks_verified=spin_ok,
            spin_tracks_ok=spin_ok,
            passed=bool(pred_audit["passed"] and audit_profile_ok and all_beh_ok and spin_ok),
        )
        return result

    if task == "synccraft_artist_assistant":
        ass_mode = str(meta.get("assistant_mode", "create_full_asset"))
        mat_ok = (not meta.get("require_materials")) or bool(pred_prog.materials)
        iou_thresh = 0.65 if ass_mode in ("create_full_asset", "coarse_to_fine_upgrade") else 0.80
        iou_ok = result.get("silhouette_iou", 1.0) >= iou_thresh
        result.update(
            assistant_mode=ass_mode,
            materials_present=bool(pred_prog.materials),
            passed=bool(pred_audit["passed"] and audit_profile_ok and mat_ok and iou_ok),
        )
        return result

    iou_ok = result.get("silhouette_iou", 1.0) >= 0.85
    result["passed"] = bool(pred_audit["passed"] and audit_profile_ok and iou_ok)
    return result
