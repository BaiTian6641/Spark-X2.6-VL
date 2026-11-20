"""Stage-8 Expanded 4-Primitive Visual Primitives (`<point>`, `<box>`, `<dash>`, `<circle>`),
Full-Color Shaded 3D Spatial Reasoning (Front/Back + Left/Right + Multi-View Kinematics +
2-Frame SE(3) Rig Animation Arcs + Balanced 50/50 Yes/No + Bilingual EN/ZH),
and `nvidia/Nemotron-Image-Training-v3` (`CC-BY-4.0`) Grounded Spatial Corpus Builder.

Core Capabilities Taught:
1. **Expanded 4-Primitive Visual Primitives Grammar (`[0..1000]` normalized image coordinates)**:
   - `<point x=".." y=".." z="near|mid|far" label=".."/>` — projected anatomical/object centers with depth tier.
   - `<box x1=".." y1=".." x2=".." y2=".." label=".."/>` — 2D extent of limbs, sub-assemblies, or scene objects.
   - `<dash x1=".." y1=".." x2=".." y2=".." label=".."/>` — directed skeletal kinematic bone axes (`parent->child`),
     multi-segment spine/neck/tail/wing chains, facing-direction arrows (`facing_left` / `facing_right`),
     `front_to_back` depth links, and 2-frame SE(3) articulation motion arcs (`anim_arc:{bone}`).
   - `<circle cx=".." cy=".." r=".." label=".."/>` — rotational joint pivots & articulation sweep radii for 3D rigging
     and animation readiness.
2. **Real Full-Color Shaded 3D Model Understanding (NEVER black-and-white silhouettes/cutouts)**:
   - Renders `CC0-1.0` procedural sculpted 3D creatures, vehicles, automatons, and dioramas with diverse dark-studio
     key+rim lighting presets and exact Z-buffering across orbital angles.
   - Directly cures Stage-7.5's `POPE yes_ratio = 1.0` bias via strictly balanced 50/50 Yes/No depth, left/right,
     and facing verification (`spatial_3d_yesno_depth_and_lr`), and cures `C-RADIOv4-H`'s horizontal-flip left/right
     orientation artifact via explicit facing-direction `<dash>` and left/right limb QA.
   - Includes ~18-20% bilingual Chinese (`zh`) spatial 3D supervision covering front/back depth ordering,
     left/right orientation, kinematic chains, animation arcs, and balanced Yes/No verification.
3. **`nvidia/Nemotron-Image-Training-v3` (`CC-BY-4.0`) Grounded Spatial & Geometric Reasoning**:
   - Joins verified `Qwen-3.5` + `Gemini-3-Flash` `<think>` traces from `clevr_1` (embedded media tars), `zhwiki_1`
     (embedded media tars, `zh`), `plotqa_1` (embedded media tars), `gqa_1`/`gqa_2` (`lmms-lab/GQA` train),
     `geometry3k` (`hiyouga/geometry3k` train), and `aokvqa_2` (`HuggingFaceM4/A-OKVQA` train) with `0%` eval overlap,
     upgrading inline `[x1, y1, x2, y2]` coordinates into `<box>`, `<point>`, `<circle>`, and relational `<dash>`.
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path
import random
import re
import tarfile
from typing import Any

import numpy as np
from PIL import Image

from .common import emit
from .reasoning_agentic_corpus import check_generation_collapse_health
from .sculpted_style_corpus import (
    FACES,
    _face_vertices_local,
    _uv_corners_px,
    bake_sculpted_rgba_atlas,
    build_sculpted_alchemist_studio_diorama_dsl,
    build_sculpted_crimson_western_dragon_dsl,
    build_sculpted_expedition_crawler_dsl,
    build_sculpted_glacial_crystal_drake_dsl,
    build_sculpted_harvest_automaton_dsl,
    build_sculpted_ochre_club_wyvern_dsl,
    build_sculpted_pastel_chibi_dragonet_dsl,
    build_sculpted_storm_wolf_dragon_v4_dsl,
)
from .synccraft_bridge import ensure_synccraft_importable, make_clean_room_provenance
from .visual_primitives_corpus import BANNED_BENCHMARK_SUBSTRINGS, _clamp_1000, _clean_xml_label, _normalize_image_bytes

NEMOTRON_V3_REPO = "nvidia/Nemotron-Image-Training-v3"
NEMOTRON_V3_REV = "7656391d4d4c55b024b120e3748a5f2f950f4b8c"

SPATIAL_3D_PRIMITIVE_TASKS: tuple[str, ...] = (
    "spatial_3d_kinematic_dash_and_joints",
    "spatial_3d_front_back_depth_order",
    "spatial_3d_left_right_orientation_facing",
    "spatial_3d_multiview_correspondence",
    "spatial_3d_rig_animation_arc",
    "spatial_3d_yesno_depth_and_lr",
)

# Diverse dark-studio key+rim lighting presets (ALWAYS full-color shaded, NEVER B&W cutout!)
STUDIO_LIGHTING_PRESETS: tuple[dict[str, Any], ...] = (
    {
        "name": "cool_slate",
        "bg_base": (20.0, 25.0, 36.0),
        "bg_gain": (18.0, 22.0, 28.0),
        "floor_drop": (6.0, 5.0, 4.0),
        "key_dir": (0.46, 0.78, -0.42),
        "rim_dir": (-0.55, 0.48, 0.68),
        "key_tint": (1.03, 1.01, 0.98),
        "rim_tint": (0.86, 0.98, 1.16),
    },
    {
        "name": "warm_charcoal",
        "bg_base": (29.0, 25.0, 23.0),
        "bg_gain": (21.0, 18.0, 16.0),
        "floor_drop": (5.0, 5.0, 4.0),
        "key_dir": (-0.42, 0.80, -0.42),
        "rim_dir": (0.58, 0.46, 0.67),
        "key_tint": (1.07, 1.01, 0.93),
        "rim_tint": (0.94, 0.96, 1.12),
    },
    {
        "name": "twilight_indigo",
        "bg_base": (21.0, 21.0, 39.0),
        "bg_gain": (18.0, 19.0, 29.0),
        "floor_drop": (5.0, 4.0, 6.0),
        "key_dir": (0.52, 0.72, -0.46),
        "rim_dir": (-0.58, 0.42, 0.70),
        "key_tint": (1.00, 1.00, 1.04),
        "rim_tint": (1.14, 1.02, 0.86),
    },
    {
        "name": "forest_dusk",
        "bg_base": (19.0, 29.0, 29.0),
        "bg_gain": (17.0, 21.0, 20.0),
        "floor_drop": (4.0, 6.0, 5.0),
        "key_dir": (0.36, 0.84, -0.40),
        "rim_dir": (-0.48, 0.50, 0.72),
        "key_tint": (1.05, 1.02, 0.94),
        "rim_tint": (0.86, 1.08, 1.08),
    },
)

_INLINE_BBOX_RE = re.compile(r"\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]")
_ROUND_OBJECT_KEYWORDS: tuple[str, ...] = (
    "sphere",
    "ball",
    "wheel",
    "clock",
    "circle",
    "circular",
    "round",
    "disc",
    "disk",
    "ring",
    "globe",
    "cylinder",
    "joint",
    "pivot",
    "hub",
    "tire",
    "tyre",
    "coin",
    "bowl",
    "sun",
    "moon",
    "orb",
    "球",
    "圆",
    "轮",
)
_LOCATION_KEYWORDS: tuple[str, ...] = (
    "point",
    "center",
    "centre",
    "locate",
    "location",
    "position",
    "where",
    "spot",
    "target",
    "focus",
    "left",
    "right",
    "top",
    "bottom",
    "front",
    "behind",
    "near",
    "far",
    "middle",
    "corner",
    "位置",
    "中心",
    "哪里",
)
_SPATIAL_REL_KEYWORDS: tuple[str, ...] = (
    "left",
    "right",
    "behind",
    "in front",
    "closer",
    "farther",
    "further",
    "above",
    "below",
    "between",
    "next to",
    "toward",
    "away",
    "beside",
    "左",
    "右",
    "前",
    "后",
    "上",
    "下",
)


def _wrap_think(think_text: str, body_text: str) -> str:
    return f"<think>\n{think_text.strip()}\n</think>\n{body_text.strip()}"


def fmt_point(x: float, y: float, label: str, *, z: str | None = None) -> str:
    lbl = _clean_xml_label(label)
    z_attr = f' z="{z}"' if z in ("near", "mid", "far") else ""
    return f'<point x="{_clamp_1000(x)}" y="{_clamp_1000(y)}"{z_attr} label="{lbl}"/>'


def fmt_box(x1: float, y1: float, x2: float, y2: float, label: str) -> str:
    lbl = _clean_xml_label(label)
    a, b = _clamp_1000(min(x1, x2)), _clamp_1000(min(y1, y2))
    c, d = _clamp_1000(max(x1, x2)), _clamp_1000(max(y1, y2))
    # Guarantee non-degenerate 2D extent >= 2px in at least one dimension
    if (c - a) < 2 and (d - b) < 2:
        if c + 2 <= 1000:
            c += 2
        elif a - 2 >= 0:
            a -= 2
        if d + 2 <= 1000:
            d += 2
        elif b - 2 >= 0:
            b -= 2
    return f'<box x1="{a}" y1="{b}" x2="{c}" y2="{d}" label="{lbl}"/>'


def fmt_dash(x1: float, y1: float, x2: float, y2: float, label: str) -> str:
    lbl = _clean_xml_label(str(label).replace("->", "_to_"))
    ix1, iy1 = _clamp_1000(x1), _clamp_1000(y1)
    ix2, iy2 = _clamp_1000(x2), _clamp_1000(y2)
    return f'<dash x1="{ix1}" y1="{iy1}" x2="{ix2}" y2="{iy2}" label="{lbl}"/>'


def _clamped_dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(_clamp_1000(x2) - _clamp_1000(x1), _clamp_1000(y2) - _clamp_1000(y1))


def fmt_circle(cx: float, cy: float, r: float, label: str) -> str:
    lbl = _clean_xml_label(label)
    rad = max(8, min(400, int(round(r))))
    return f'<circle cx="{_clamp_1000(cx)}" cy="{_clamp_1000(cy)}" r="{rad}" label="{lbl}"/>'


def _extract_world_quads_and_parts(
    baked: dict[str, Any],
) -> tuple[list[tuple[np.ndarray, np.ndarray, bool]], dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    quads: list[tuple[np.ndarray, np.ndarray, bool]] = []
    bone_verts: dict[str, list[np.ndarray]] = {}
    box_verts: dict[str, list[np.ndarray]] = {}
    for rec in baked["elements"]:
        ctr = rec["world_center"]
        rot = rec["world_rotation"]
        half_ext = 0.5 * rec["eff_size"]
        is_pl = bool(rec["is_plane"])
        p_norm = rec["plane_normal"]
        local_faces = _face_vertices_local(half_ext)
        bname = str(rec.get("bone", rec.get("bone_name", "root")))
        ename = str(rec.get("name", bname))

        el_verts = []
        if is_pl:
            pf = {"x": "east", "y": "up", "z": "south"}[p_norm]
            fv = local_faces[pf] @ rot.T + ctr[None, :]
            quads.append((fv, _uv_corners_px(pf, rec["uv"][pf]), True))
            el_verts.append(fv)
        else:
            for f in FACES:
                fv = local_faces[f] @ rot.T + ctr[None, :]
                quads.append((fv, _uv_corners_px(f, rec["uv"][f]), False))
                el_verts.append(fv)
        vcat = np.concatenate(el_verts, axis=0)
        bone_verts.setdefault(bname, []).append(vcat)
        box_verts.setdefault(ename, []).append(vcat)
    return quads, bone_verts, box_verts


def _build_perspective_camera(
    all_pts: np.ndarray,
    *,
    az_deg: float,
    el_deg: float,
    fill_frac: float,
    pan_uv: tuple[float, float],
    panel_size: tuple[int, int],
) -> dict[str, Any]:
    width, height = int(panel_size[0]), int(panel_size[1])
    az, el = math.radians(az_deg), math.radians(el_deg)
    cam_dir = np.array([math.cos(el) * math.sin(az), math.sin(el), -math.cos(el) * math.cos(az)], dtype=np.float64)
    fwd = -cam_dir / np.linalg.norm(cam_dir)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    right = np.array([1.0, 0.0, 0.0], dtype=np.float64) if np.linalg.norm(right) < 1e-6 else (right / np.linalg.norm(right))
    up = np.cross(right, fwd)

    fov_deg = 32.0
    aspect = width / max(1.0, float(height))
    half_tan_y = math.tan(math.radians(fov_deg * 0.5))
    half_tan_x = half_tan_y * aspect

    px_c = all_pts @ right
    py_c = all_pts @ up
    pz_c = all_pts @ (-fwd)
    target = (
        right * (0.5 * (px_c.min() + px_c.max()))
        + up * (0.5 * (py_c.min() + py_c.max()))
        + (-fwd) * (0.5 * (pz_c.min() + pz_c.max()))
    )
    rel_x = np.abs(all_pts @ right - float(target @ right))
    rel_y = np.abs(all_pts @ up - float(target @ up))
    rel_d = (all_pts - target[None, :]) @ cam_dir
    cam_dist = max(
        10.0,
        float(max(np.max(rel_x / (half_tan_x * fill_frac) + rel_d), np.max(rel_y / (half_tan_y * fill_frac) + rel_d))),
    )
    pan_scale = cam_dist * half_tan_y * 0.35
    target = target - right * (float(pan_uv[0]) * pan_scale) - up * (float(pan_uv[1]) * pan_scale)
    cam_pos = target + cam_dir * cam_dist
    bbox_diag = float(np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0)))
    return {
        "az_deg": float(az_deg),
        "el_deg": float(el_deg),
        "target": target,
        "cam_pos": cam_pos,
        "right": right,
        "up": up,
        "fwd": fwd,
        "half_tan_x": half_tan_x,
        "half_tan_y": half_tan_y,
        "bbox_diag": max(4.0, bbox_diag),
    }


def _repose_baked_for_scene(baked: dict[str, Any], posed_scene: Any) -> dict[str, Any]:
    """Build a non-mutating copy of `baked` with updated `world_center`/`world_rotation` from `posed_scene`."""
    boxes_by_idx = posed_scene.boxes
    boxes_by_name = {cb.name: cb for cb in boxes_by_idx}
    new_elems: list[dict[str, Any]] = []
    for rec in baked["elements"]:
        idx = int(rec["idx"])
        cb = boxes_by_idx[idx] if (0 <= idx < len(boxes_by_idx) and boxes_by_idx[idx].name == rec["name"]) else boxes_by_name.get(rec["name"])
        if cb is None:
            new_elems.append(dict(rec))
        else:
            nr = dict(rec)
            nr["world_center"] = np.asarray(cb.center, dtype=np.float64)
            nr["world_rotation"] = np.asarray(cb.rotation, dtype=np.float64)
            new_elems.append(nr)
    return {
        "width": baked["width"],
        "height": baked["height"],
        "atlas_rgba": baked["atlas_rgba"],
        "emissive_map": baked["emissive_map"],
        "atlas_png": baked.get("atlas_png"),
        "elements": new_elems,
    }


def render_shaded_3d_with_spatial_annotations(
    program: Any,
    baked: dict[str, Any],
    *,
    az_deg: float = 36.0,
    el_deg: float = 18.0,
    fill_frac: float = 0.78,
    pan_uv: tuple[float, float] = (0.0, 0.0),
    panel_size: tuple[int, int] = (384, 384),
    studio_preset: str | int = "cool_slate",
    scene: Any | None = None,
    camera_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render a full-color shaded, key+rim-lit 3D perspective studio frame AND extract exact bone-level
    `<point>`, `<box>`, `<dash>` (parent->child & facing & front-to-back), and `<circle>` joint pivot metadata.
    """
    ensure_synccraft_importable()
    from synccraft.kinematics.se3_chain import compile_geometry

    width, height = int(panel_size[0]), int(panel_size[1])
    atlas_rgba: np.ndarray = baked["atlas_rgba"]
    emissive_map: np.ndarray = baked["emissive_map"]
    tex_h, tex_w = atlas_rgba.shape[:2]

    sc = scene if scene is not None else compile_geometry(program)
    bone_parent_map: dict[str, str | None] = {}
    bone_pivot_world: dict[str, np.ndarray] = {}
    for wb in (getattr(sc, "bones", None) or ()):
        bname = str(wb.name)
        bone_parent_map[bname] = str(wb.parent_name) if getattr(wb, "parent_name", None) else None
        if getattr(wb, "pivot_world", None) is not None:
            bone_pivot_world[bname] = np.asarray(wb.pivot_world, dtype=np.float64)

    quads, bone_verts, box_verts = _extract_world_quads_and_parts(baked)
    all_pts = np.concatenate([q[0] for q in quads], axis=0)

    if camera_override is not None:
        cam = camera_override
    else:
        cam = _build_perspective_camera(
            all_pts,
            az_deg=az_deg,
            el_deg=el_deg,
            fill_frac=fill_frac,
            pan_uv=pan_uv,
            panel_size=panel_size,
        )
    target: np.ndarray = cam["target"]
    cam_pos: np.ndarray = cam["cam_pos"]
    right: np.ndarray = cam["right"]
    up: np.ndarray = cam["up"]
    fwd: np.ndarray = cam["fwd"]
    half_tan_x: float = float(cam["half_tan_x"])
    half_tan_y: float = float(cam["half_tan_y"])
    bbox_diag: float = float(cam["bbox_diag"])

    def _project(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rel = pts - cam_pos[None, :]
        cx = rel @ right
        cy = rel @ up
        cz = rel @ fwd
        cz_safe = np.maximum(cz, 0.1)
        sx = (0.5 + 0.5 * (cx / (cz_safe * half_tan_x))) * (width - 1)
        sy = (0.5 - 0.5 * (cy / (cz_safe * half_tan_y))) * (height - 1)
        return sx, sy, cz

    # Select rich dark-studio lighting & backdrop preset (NEVER flat B&W cutout!)
    if isinstance(studio_preset, int):
        preset = STUDIO_LIGHTING_PRESETS[studio_preset % len(STUDIO_LIGHTING_PRESETS)]
    else:
        preset = next((p for p in STUDIO_LIGHTING_PRESETS if p["name"] == studio_preset), STUDIO_LIGHTING_PRESETS[0])

    yy_bg, xx_bg = np.mgrid[0:height, 0:width]
    ry_n = (yy_bg / max(1.0, height - 1.0)) - 0.45
    rx_n = (xx_bg / max(1.0, width - 1.0)) - 0.50
    rad = np.sqrt(rx_n * rx_n + ry_n * ry_n)
    bg = np.clip(1.0 - 0.50 * rad, 0.35, 1.0)
    floor_grad = np.clip((yy_bg / max(1.0, height - 1.0) - 0.62) / 0.38, 0.0, 1.0)
    bb, bgain, fdrop = preset["bg_base"], preset["bg_gain"], preset["floor_drop"]
    color_srgb = np.stack(
        [
            bb[0] + bgain[0] * bg - fdrop[0] * floor_grad,
            bb[1] + bgain[1] * bg - fdrop[1] * floor_grad,
            bb[2] + bgain[2] * bg - fdrop[2] * floor_grad,
        ],
        axis=-1,
    )
    zbuf = np.full((height, width), 1e9, dtype=np.float64)
    key_light = np.array(preset["key_dir"], dtype=np.float64)
    key_light /= np.linalg.norm(key_light)
    rim_light = np.array(preset["rim_dir"], dtype=np.float64)
    rim_light /= np.linalg.norm(rim_light)
    key_tint = np.array(preset["key_tint"], dtype=np.float64)
    rim_tint = np.array(preset["rim_tint"], dtype=np.float64)

    for verts, uv_c, double_sided in quads:
        sx, sy, cz = _project(verts)
        if np.any(cz <= 0.1):
            continue
        fn = np.cross(verts[1] - verts[0], verts[2] - verts[0])
        fn_norm = float(np.linalg.norm(fn))
        if fn_norm < 1e-8:
            continue
        fn /= fn_norm
        to_cam = cam_pos - verts.mean(axis=0)
        facing = float(np.dot(fn, to_cam))
        if not double_sided and facing <= 0.0:
            continue
        if double_sided and facing < 0.0:
            fn = -fn
        diff = max(0.0, float(np.dot(fn, key_light)))
        rim = max(0.0, float(np.dot(fn, rim_light)))
        shade_rgb = 0.54 + 0.42 * diff * key_tint + 0.16 * rim * rim_tint  # (3,)

        for i0, i1, i2 in ((0, 1, 2), (0, 2, 3)):
            x0, y0, z0 = sx[i0], sy[i0], cz[i0]
            x1, y1, z1 = sx[i1], sy[i1], cz[i1]
            x2, y2, z2 = sx[i2], sy[i2], cz[i2]
            min_x = max(0, int(math.floor(min(x0, x1, x2))))
            max_x = min(width - 1, int(math.ceil(max(x0, x1, x2))))
            min_y = max(0, int(math.floor(min(y0, y1, y2))))
            max_y = min(height - 1, int(math.ceil(max(y0, y1, y2))))
            if min_x > max_x or min_y > max_y:
                continue
            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) < 1e-8:
                continue
            gx, gy = np.meshgrid(np.arange(min_x, max_x + 1, dtype=np.float64), np.arange(min_y, max_y + 1, dtype=np.float64))
            w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
            w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
            if not np.any(inside):
                continue
            z_interp = w0 * z0 + w1 * z1 + w2 * z2
            sub_z = zbuf[min_y : max_y + 1, min_x : max_x + 1]
            vis = inside & (z_interp < sub_z)
            if not np.any(vis):
                continue
            u0, v0 = uv_c[i0]
            u1, v1 = uv_c[i1]
            u2, v2 = uv_c[i2]
            iu = np.clip((w0 * u0 + w1 * u1 + w2 * u2).astype(np.int32), 0, tex_w - 1)
            iv = np.clip((w0 * v0 + w1 * v1 + w2 * v2).astype(np.int32), 0, tex_h - 1)
            rgba = atlas_rgba[iv, iu]
            alpha_ok = rgba[..., 3] > 127
            vis = vis & alpha_ok
            if not np.any(vis):
                continue
            sub_z[vis] = z_interp[vis]
            em = emissive_map[iv, iu]
            lit = rgba[..., :3].astype(np.float64) * (
                shade_rgb[None, None, :] * (1.0 - em[..., None]) + 1.08 * em[..., None]
            )
            sub_c = color_srgb[min_y : max_y + 1, min_x : max_x + 1]
            sub_c[vis] = np.clip(lit[vis], 0.0, 255.0)

    img = Image.fromarray(np.clip(color_srgb, 0, 255).astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    def _project_part_dict(part_verts: dict[str, list[np.ndarray]], use_pivots: bool) -> dict[str, dict[str, Any]]:
        out_info: dict[str, dict[str, Any]] = {}
        cz_vals: list[float] = []
        for pname, vlist in part_verts.items():
            pts = np.concatenate(vlist, axis=0)
            sx, sy, cz = _project(pts)
            x_norm = sx * (1000.0 / max(1.0, float(width - 1)))
            y_norm = sy * (1000.0 / max(1.0, float(height - 1)))
            cx_1000 = float(np.mean(x_norm))
            cy_1000 = float(np.mean(y_norm))
            bx1, by1 = float(np.min(x_norm)), float(np.min(y_norm))
            bx2, by2 = float(np.max(x_norm)), float(np.max(y_norm))
            cz_mean = float(np.mean(cz))
            cz_min = float(np.min(cz))
            if use_pivots and pname in bone_pivot_world:
                psx, psy, pcz = _project(bone_pivot_world[pname][None, :])
                px_1000 = float(psx[0] * (1000.0 / max(1.0, float(width - 1))))
                py_1000 = float(psy[0] * (1000.0 / max(1.0, float(height - 1))))
                pcz_val = float(pcz[0])
            else:
                px_1000, py_1000, pcz_val = cx_1000, cy_1000, cz_mean
            radius = max(14.0, min(140.0, 0.35 * math.hypot(bx2 - bx1, by2 - by1)))
            out_info[pname] = {
                "name": pname,
                "parent": bone_parent_map.get(pname) if use_pivots else None,
                "cx": cx_1000,
                "cy": cy_1000,
                "box": [bx1, by1, bx2, by2],
                "pivot_xy": [px_1000, py_1000],
                "pivot_cz": pcz_val,
                "cz_mean": cz_mean,
                "cz_min": cz_min,
                "radius": radius,
            }
            cz_vals.append(cz_mean)
        if cz_vals:
            z_lo, z_hi = min(cz_vals), max(cz_vals)
            z_span = max(1e-4, z_hi - z_lo)
            for info in out_info.values():
                rel_z = (info["cz_mean"] - z_lo) / z_span
                info["depth_tier"] = "near" if rel_z <= 0.33 else ("far" if rel_z >= 0.67 else "mid")
                info["depth_rel"] = float(rel_z)
        return out_info

    bones_info = _project_part_dict(bone_verts, use_pivots=True)
    # For static single-bone dioramas/props, expose named boxes as spatial parts while keeping kinematic_edges empty
    spatial_parts = bones_info if len(bones_info) >= 3 else _project_part_dict(box_verts, use_pivots=False)

    # Compute exact canonical facing vector (model local -Z is forward in SyncCraft!)
    ray_len = max(4.0, 0.22 * bbox_diag)
    f_pts = np.stack([target, target + np.array([0.0, 0.0, -ray_len], dtype=np.float64)], axis=0)
    fsx, fsy, fcz = _project(f_pts)
    fx0, fy0 = float(fsx[0] * 1000.0 / max(1.0, width - 1)), float(fsy[0] * 1000.0 / max(1.0, height - 1))
    fx1, fy1 = float(fsx[1] * 1000.0 / max(1.0, width - 1)), float(fsy[1] * 1000.0 / max(1.0, height - 1))
    dx_face = fx1 - fx0
    dz_face = float(fcz[1] - fcz[0])
    if abs(dx_face) >= 22.0:
        lr_facing = "right" if dx_face > 0.0 else "left"
    else:
        lr_facing = "center"
    fb_facing = "toward_camera" if dz_face < -0.8 else ("away_from_camera" if dz_face > 0.8 else "profile")

    # Build kinematic parent -> child `<dash>` edges (post-clamp distance >= 12.0!)
    kinematic_edges: list[dict[str, Any]] = []
    for bname, info in bones_info.items():
        par = info.get("parent")
        if par and par in bones_info:
            pinfo = bones_info[par]
            dist = _clamped_dist(pinfo["cx"], pinfo["cy"], info["cx"], info["cy"])
            if dist >= 12.0:
                kinematic_edges.append(
                    {
                        "parent": par,
                        "child": bname,
                        "x1": pinfo["cx"],
                        "y1": pinfo["cy"],
                        "x2": info["cx"],
                        "y2": info["cy"],
                        "dist": dist,
                        "dz": info["cz_mean"] - pinfo["cz_mean"],
                    }
                )

    return {
        "png_bytes": png_bytes,
        "rgb_array": np.clip(color_srgb, 0, 255).astype(np.uint8),
        "az_deg": float(cam["az_deg"]),
        "el_deg": float(cam["el_deg"]),
        "studio_preset": preset["name"],
        "camera": cam,
        "camera_right": [float(v) for v in right.tolist()],
        "camera_up": [float(v) for v in up.tolist()],
        "bones": bones_info,
        "spatial_parts": spatial_parts,
        "kinematic_edges": kinematic_edges,
        "facing": {
            "lr": lr_facing,
            "fb": fb_facing,
            "x1": fx0,
            "y1": fy0,
            "x2": fx1,
            "y2": fy1,
            "dx": dx_face,
            "dz": dz_face,
            "dist": _clamped_dist(fx0, fy0, fx1, fy1),
        },
    }


def _collect_clean_3d_assets(seed: int = 808) -> list[dict[str, Any]]:
    """Compile a diverse library of CC0-1.0 original sculpted 3D models (`v25` masterworks + sculpted creatures/props)."""
    ensure_synccraft_importable()
    from synccraft.dsl.parser import parse_program
    from synccraft.kinematics.se3_chain import compile_geometry
    from .aesthetic_v25_vigorous_corpus import build_all_v25_verified_masterworks

    assets: list[dict[str, Any]] = []
    # 1. All 8 v25 verified masterworks (4 archetypes x 2 palettes)
    for item in build_all_v25_verified_masterworks():
        try:
            aid = str(item["asset_id"]).strip()
            prog = parse_program(item["dsl"])
            sc = compile_geometry(prog)
            baked = bake_sculpted_rgba_atlas(prog, scene=sc, texels_per_unit=2)
            fam = str(item.get("family_id") or f"v25_{item.get('archetype') or aid}")
            assets.append(
                {
                    "asset_id": aid,
                    "family_id": fam,
                    "title": aid.replace("_", " "),
                    "program": prog,
                    "scene": sc,
                    "baked": baked,
                    "is_creature": item.get("domain") in ("D2", "D3", "D4"),
                }
            )
        except Exception as exc:
            emit("spatial3d_v25_asset_skipped", asset_id=item.get("asset_id"), error=f"{type(exc).__name__}: {exc}")

    # 2. All 8 sculpted CC0-1.0 creatures, robots, vehicles & dioramas x 2 variants
    #    Note: `fn(variant=var)` returns `(sculpted_dsl, blockout_dsl, meta_dict)` — ALWAYS use `sculpted_dsl`!
    builders = (
        ("crimson_western_dragon", build_sculpted_crimson_western_dragon_dsl, True),
        ("glacial_crystal_drake", build_sculpted_glacial_crystal_drake_dsl, True),
        ("ochre_club_wyvern", build_sculpted_ochre_club_wyvern_dsl, True),
        ("pastel_chibi_dragonet", build_sculpted_pastel_chibi_dragonet_dsl, True),
        ("storm_wolf_dragon_v4", build_sculpted_storm_wolf_dragon_v4_dsl, True),
        ("harvest_automaton", build_sculpted_harvest_automaton_dsl, True),
        ("expedition_crawler", build_sculpted_expedition_crawler_dsl, False),
        ("alchemist_studio_diorama", build_sculpted_alchemist_studio_diorama_dsl, False),
    )
    for name, fn, is_creature in builders:
        for var in (0, 1):
            try:
                sculpted_dsl, _blockout_dsl, meta = fn(variant=var)
                prog = parse_program(sculpted_dsl)
                sc = compile_geometry(prog)
                baked = bake_sculpted_rgba_atlas(prog, scene=sc, texels_per_unit=2)
                fam = f"spatial3d_{meta.get('family', name)}"
                title = str(meta.get("title") or name.replace("_", " "))
                assets.append(
                    {
                        "asset_id": f"{name}_v{var}",
                        "family_id": fam,
                        "title": title,
                        "program": prog,
                        "scene": sc,
                        "baked": baked,
                        "is_creature": is_creature,
                    }
                )
            except Exception as exc:
                emit("spatial3d_sculpted_asset_skipped", name=name, variant=var, error=f"{type(exc).__name__}: {exc}")
    return assets


def _extract_structural_chains(bones: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Trace multi-segment structural chains (spine/neck/head, tail, wings, limbs) from valid kinematic edges."""
    by_parent: dict[str, list[dict[str, Any]]] = {}
    has_incoming: set[str] = set()
    for e in edges:
        by_parent.setdefault(e["parent"], []).append(e)
        has_incoming.add(e["child"])
    for ch_list in by_parent.values():
        ch_list.sort(key=lambda e: e["dist"], reverse=True)

    # Walk all directed paths of length >= 2 (at least 2 connected <dash> hops)
    chains: list[list[dict[str, Any]]] = []

    def _dfs(node: str, cur: list[dict[str, Any]], visited: set[str]) -> None:
        outs = [e for e in by_parent.get(node, ()) if e["child"] not in visited]
        if not outs:
            if len(cur) >= 2:
                chains.append(list(cur))
            return
        for e in outs[:2]:
            visited.add(e["child"])
            cur.append(e)
            _dfs(e["child"], cur, visited)
            cur.pop()
            visited.remove(e["child"])

    roots = sorted(set(by_parent.keys()) - has_incoming) or sorted(by_parent.keys())
    for r in roots:
        _dfs(r, [], {r})
    # Score chains prioritizing spine/neck/tail/wing chains and total projected length
    def _chain_score(ch: list[dict[str, Any]]) -> float:
        names = " ".join(e["child"].lower() for e in ch)
        bonus = 180.0 if any(k in names for k in ("neck", "head", "tail", "wing", "spine", "chest")) else 0.0
        return bonus + sum(e["dist"] for e in ch)

    chains.sort(key=_chain_score, reverse=True)
    return chains[:3]


def _select_pose_recipes_for_asset(scene: Any) -> list[dict[str, Any]]:
    """Return deterministic, anatomically meaningful SE(3) articulation pose recipes for an articulated asset."""
    bone_names = {str(wb.name) for wb in (getattr(scene, "bones", None) or ())}
    if len(bone_names) < 2:
        return []

    recipes: list[dict[str, Any]] = []

    # 1. Wing downstroke / upstroke
    wing_specs = (
        ("wing_l_upper", "wing_l_fore", "wing_l_wrist", "wing_r_upper", "wing_r_upper__wing_l_fore"),
        ("wing_sh_l", "wing_fore_l", "finger_a_l", "wing_sh_r", "wing_sh_r__wing_fore_l"),
        ("wing_arm_l", "wing_fore_l", "folded_finger_l", "wing_arm_r", "wing_arm_r__wing_fore_l"),
        ("tiny_wing_l", None, "tiny_wing_l", "tiny_wing_r", None),
    )
    for wl, wf, wt, wr, wrf in wing_specs:
        if wl in bone_names and wt in bone_names:
            deltas: dict[str, list[float]] = {wl: [-16.0, 14.0, -38.0]}
            chain = [wl]
            if wf and wf in bone_names:
                deltas[wf] = [-10.0, 18.0, -14.0]
                chain.append(wf)
            if wt != wl:
                chain.append(wt)
            if wr and wr in bone_names:
                deltas[wr] = [-16.0, -14.0, 38.0]
            if wrf and wrf in bone_names:
                deltas[wrf] = [-10.0, -18.0, 14.0]
            recipes.append(
                {
                    "action_id": "wing_downstroke",
                    "action_en": "wing downstroke sweep",
                    "action_zh": "展翼下扑动作",
                    "pivot_bone": wl,
                    "effector_bone": wt,
                    "chain_bones": chain,
                    "deltas": deltas,
                    "primary_delta": deltas[wl],
                }
            )
            break

    # 2. Tail lateral sweep + lift
    tail_chains = (
        ["tail_1", "tail_2", "tail_3", "tail_4", "tail_5"],
        ["tail_1", "tail_2", "tail_3", "tail_4"],
        ["tail_1", "tail_2", "tail_mace"],
        ["curl_tail_1", "curl_tail_2", "curl_tail_3"],
        ["t1", "t2"],
        ["tail", "tail_tip"],
    )
    for tchain in tail_chains:
        present = [b for b in tchain if b in bone_names]
        if len(present) >= 2:
            deltas_t: dict[str, list[float]] = {}
            for i, bn in enumerate(present[:3]):
                scale = 1.0 - 0.22 * i
                deltas_t[bn] = [round(16.0 * scale, 1), round(30.0 * scale, 1), 0.0]
            recipes.append(
                {
                    "action_id": "tail_sweep",
                    "action_en": "tail lateral sweep and lift",
                    "action_zh": "尾部侧扫上扬动作",
                    "pivot_bone": present[0],
                    "effector_bone": present[-1],
                    "chain_bones": present,
                    "deltas": deltas_t,
                    "primary_delta": deltas_t[present[0]],
                }
            )
            break

    # 3. Neck turn + jaw roar / head strike
    neck_specs = (
        (["neck_1", "neck_2", "head"], "jaw"),
        (["neck_base", "neck_upper", "head"], "jaw"),
        (["neck_seg", "head"], "heavy_jaw"),
        (["neck_seg", "head"], "jaw"),
        (["neck", "head"], None),
        (["big_head"], None),
        (["head"], "jaw"),
        (["sensor_head"], None),
    )
    for nchain, jaw_b in neck_specs:
        present = [b for b in nchain if b in bone_names]
        if present:
            deltas_n: dict[str, list[float]] = {}
            if len(present) >= 2:
                deltas_n[present[0]] = [-18.0, 32.0, 0.0]
                deltas_n[present[-1]] = [-10.0, 18.0, 0.0]
            else:
                deltas_n[present[0]] = [-20.0, 36.0, 0.0]
            has_jaw = bool(jaw_b and jaw_b in bone_names)
            if has_jaw and jaw_b is not None:
                deltas_n[jaw_b] = [40.0, 0.0, 0.0]
            recipes.append(
                {
                    "action_id": "neck_turn_jaw_open" if has_jaw else "head_turn",
                    "action_en": "neck turn with jaw opening roar" if has_jaw else "head and neck turn",
                    "action_zh": "颈部转向并张开下颌咆哮" if has_jaw else "头部转向动作",
                    "pivot_bone": present[0],
                    "effector_bone": present[-1],
                    "chain_bones": present + ([jaw_b] if has_jaw and jaw_b else []),
                    "deltas": deltas_n,
                    "primary_delta": deltas_n[present[0]],
                }
            )
            break

    # 4. Limb stride / mechanical arm reach / bogie articulation / prop tilt
    limb_specs = (
        (["arm_l_upper", "arm_l_fore", "paw_l"], "forelimb stride forward", "前肢前跨步态"),
        (["leg_l_thigh", "leg_l_shin", "hind_foot_l"], "hind leg stride swing", "后腿跨步摆动"),
        (["fl_upper", "fl_lower", "fl_foot"], "foreleg stride lift", "前腿抬升跨步"),
        (["fl_arm", "fl_fore"], "forearm reach forward", "前臂前伸动作"),
        (["hind_l", "shin_l", "foot_l"], "hind leg stance stride", "后肢跨步动作"),
        (["picker_arm_l", "forearm_l"], "mechanical picker arm reach", "机械臂前伸作业"),
        (["fl", "fl__shin"], "foreleg stride swing", "前肢迈步摆动"),
        (["arm_l"], "arm wave raise", "前臂上举挥动"),
        (["stub_arm_l"], "stub arm raise", "短臂上举动作"),
        (["bogie_fl"], "front bogie suspension pitch", "前转向架悬架俯仰"),
        (["hearth_lantern"], "hanging lantern swing", "提灯摆动姿态"),
        (["produce_bundle"], "produce bundle tilt shift", "蔬果束倾斜位移"),
    )
    for lchain, act_en, act_zh in limb_specs:
        present = [b for b in lchain if b in bone_names]
        if present:
            deltas_l: dict[str, list[float]] = {present[0]: [-36.0, 14.0, -14.0]}
            if len(present) >= 2:
                deltas_l[present[1]] = [26.0, 0.0, 0.0]
            recipes.append(
                {
                    "action_id": "limb_stride",
                    "action_en": act_en,
                    "action_zh": act_zh,
                    "pivot_bone": present[0],
                    "effector_bone": present[-1],
                    "chain_bones": present,
                    "deltas": deltas_l,
                    "primary_delta": deltas_l[present[0]],
                }
            )
            break

    return recipes


def _select_distinct_depth_trio(parts: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Pick `(front, mid, back)` parts with clear depth separation and non-degenerate 2D dash >= 10px."""
    sorted_d = sorted(parts.values(), key=lambda b: b["cz_mean"])
    if len(sorted_d) < 3:
        return None
    front = sorted_d[0]
    back = None
    for cand in reversed(sorted_d[2:]):
        if (cand["cz_mean"] - front["cz_mean"]) >= 1.8 and _clamped_dist(front["cx"], front["cy"], cand["cx"], cand["cy"]) >= 10.0:
            back = cand
            break
    if back is None:
        return None
    # Pick mid part distinct from both front & back
    mid_cands = [b for b in sorted_d[1:-1] if b["name"] not in (front["name"], back["name"])]
    if not mid_cands:
        return None
    mid = mid_cands[len(mid_cands) // 2]
    return front, mid, back


def _build_yesno_record(
    *,
    aid: str,
    fam: str,
    title: str,
    view_tag: str,
    member: str,
    base_u: dict[str, Any],
    ren: dict[str, Any],
    subtype: str,
    want_yes: bool,
    lang: str,
    seq_idx: int,
) -> tuple[dict[str, Any], dict[str, bytes]] | None:
    """Build one strictly verified Yes/No record (`depth`, `left_right`, or `facing`) with visual primitive proof."""
    parts = ren["spatial_parts"]
    facing = ren["facing"]
    sorted_d = sorted(parts.values(), key=lambda b: b["cz_mean"])
    sorted_x = sorted(parts.values(), key=lambda b: b["cx"])

    if subtype == "depth":
        trio = _select_distinct_depth_trio(parts)
        if trio is None:
            return None
        front_b, _mid_b, back_b = trio
        a_part, b_part = (front_b, back_b) if want_yes else (back_b, front_b)
        dash_tag = fmt_dash(front_b["cx"], front_b["cy"], back_b["cx"], back_b["cy"], f"front_to_back:{front_b['name']}->{back_b['name']}")
        if lang == "zh":
            prompt = (
                f"观察这张全彩着色3D渲染图，判断 `{a_part['name']}` 是否比 `{b_part['name']}` 更靠近镜头（位于前方）？"
                "请先用视觉原语标注位置与深度，再明确回答是或否。"
            )
            think = "\n".join(
                [
                    f"核对着色3D模型 {title} 中 `{a_part['name']}` 与 `{b_part['name']}` 的相机深度关系：",
                    f"- 近景部件：{fmt_point(front_b['cx'], front_b['cy'], front_b['name'], z='near')} {fmt_box(*front_b['box'], front_b['name'])}",
                    f"- 远景部件：{fmt_point(back_b['cx'], back_b['cy'], back_b['name'], z='far')} {fmt_box(*back_b['box'], back_b['name'])}",
                    f"- 由前向后深度射线：{dash_tag}",
                    f"结论：`{front_b['name']}` 明显更靠近镜头，而 `{back_b['name']}` 位于后方。",
                ]
            )
            ans_body = (
                f"是，`{a_part['name']}` 更靠近镜头，位于 `{b_part['name']}` 的前方。"
                if want_yes
                else f"否，`{a_part['name']}` 位于后方，离镜头更近的是 `{b_part['name']}`。"
            )
        else:
            prompt = (
                f"In this full-color shaded 3D render, is `{a_part['name']}` closer to the camera (in front of) `{b_part['name']}`? "
                "Verify with <point>, <box>, and <dash>, then answer Yes or No."
            )
            think = "\n".join(
                [
                    f"Checking camera-space depth between `{a_part['name']}` and `{b_part['name']}` on shaded {title}:",
                    f"- Closer foreground assembly: {fmt_point(front_b['cx'], front_b['cy'], front_b['name'], z='near')} {fmt_box(*front_b['box'], front_b['name'])}",
                    f"- Farther background assembly: {fmt_point(back_b['cx'], back_b['cy'], back_b['name'], z='far')} {fmt_box(*back_b['box'], back_b['name'])}",
                    f"- Front-to-back ray: {dash_tag}",
                    f"Since `{front_b['name']}` has smaller camera depth than `{back_b['name']}`, the statement is {'true (Yes)' if want_yes else 'false (No)'}.",
                ]
            )
            ans_body = (
                f"Yes, `{a_part['name']}` is closer to the camera and sits in front of `{b_part['name']}`."
                if want_yes
                else f"No, `{a_part['name']}` is farther back; `{b_part['name']}` is closer to the camera."
            )

    elif subtype == "left_right":
        if len(sorted_x) < 2:
            return None
        left_b, right_b = sorted_x[0], sorted_x[-1]
        if (right_b["cx"] - left_b["cx"]) < 55.0 or _clamped_dist(left_b["cx"], left_b["cy"], right_b["cx"], right_b["cy"]) < 10.0:
            return None
        a_part, b_part = (left_b, right_b) if want_yes else (right_b, left_b)
        dash_tag = fmt_dash(left_b["cx"], left_b["cy"], right_b["cx"], right_b["cy"], f"left_to_right:{left_b['name']}->{right_b['name']}")
        if lang == "zh":
            prompt = (
                f"在这张着色3D渲染图中，`{a_part['name']}` 是否位于 `{b_part['name']}` 的左侧？"
                "请结合图像坐标原语给出推理，并直接回答是或否。"
            )
            think = "\n".join(
                [
                    f"比较着色3D模型 {title} 中 `{a_part['name']}` 与 `{b_part['name']}` 的画面水平位置：",
                    f"- 靠左部件：{fmt_point(left_b['cx'], left_b['cy'], left_b['name'], z=left_b['depth_tier'])} {fmt_box(*left_b['box'], left_b['name'])}",
                    f"- 靠右部件：{fmt_point(right_b['cx'], right_b['cy'], right_b['name'], z=right_b['depth_tier'])} {fmt_box(*right_b['box'], right_b['name'])}",
                    f"- 从左至右连线：{dash_tag}",
                    f"水平坐标 x({left_b['name']})={_clamp_1000(left_b['cx'])} < x({right_b['name']})={_clamp_1000(right_b['cx'])}。",
                ]
            )
            ans_body = (
                f"是，`{a_part['name']}` 在画面中位于 `{b_part['name']}` 的左侧。"
                if want_yes
                else f"否，`{a_part['name']}` 位于画面右侧，在左侧的是 `{b_part['name']}`。"
            )
        else:
            prompt = (
                f"Is `{a_part['name']}` to the left of `{b_part['name']}` in this shaded 3D render? "
                "Ground both parts with <point>, <box>, and <dash>, then answer Yes or No."
            )
            think = "\n".join(
                [
                    f"Comparing horizontal screen placement on shaded {title}:",
                    f"- Leftward part: {fmt_point(left_b['cx'], left_b['cy'], left_b['name'], z=left_b['depth_tier'])} {fmt_box(*left_b['box'], left_b['name'])}",
                    f"- Rightward part: {fmt_point(right_b['cx'], right_b['cy'], right_b['name'], z=right_b['depth_tier'])} {fmt_box(*right_b['box'], right_b['name'])}",
                    f"- Left-to-right vector: {dash_tag}",
                    f"Because x({left_b['name']})={_clamp_1000(left_b['cx'])} < x({right_b['name']})={_clamp_1000(right_b['cx'])}, the claim is {'true (Yes)' if want_yes else 'false (No)'}.",
                ]
            )
            ans_body = (
                f"Yes, `{a_part['name']}` sits to the left of `{b_part['name']}` in this render."
                if want_yes
                else f"No, `{a_part['name']}` is to the right of `{b_part['name']}` in this render."
            )

    else:  # subtype == "facing"
        lr_dir = facing["lr"]
        if lr_dir not in ("left", "right") or facing["dist"] < 10.0 or len(sorted_x) < 2:
            return None
        opp_dir = "left" if lr_dir == "right" else "right"
        asked_dir = lr_dir if want_yes else opp_dir
        left_b, right_b = sorted_x[0], sorted_x[-1]
        face_dash = fmt_dash(facing["x1"], facing["y1"], facing["x2"], facing["y2"], f"facing_{lr_dir}")
        zh_dir = {"left": "左侧", "right": "右侧"}
        if lang == "zh":
            prompt = (
                f"在这张着色3D渲染图中，该3D模型是否正朝向画面{zh_dir[asked_dir]}？"
                "请使用 <dash> 标出朝向矢量并回答是或否。"
            )
            think = "\n".join(
                [
                    f"判定着色3D模型 {title} 的正面朝向矢量：",
                    f"- 左端可见区域：{fmt_point(left_b['cx'], left_b['cy'], left_b['name'], z=left_b['depth_tier'])} {fmt_box(*left_b['box'], left_b['name'])}",
                    f"- 右端可见区域：{fmt_point(right_b['cx'], right_b['cy'], right_b['name'], z=right_b['depth_tier'])} {fmt_box(*right_b['box'], right_b['name'])}",
                    f"- 3D前向朝向箭头：{face_dash}",
                    f"前向投影水平位移 dx={facing['dx']:.1f}，表明模型实际朝向画面{zh_dir[lr_dir]}。",
                ]
            )
            ans_body = (
                f"是，该3D模型正朝向画面{zh_dir[lr_dir]}。"
                if want_yes
                else f"否，该3D模型并非朝向{zh_dir[asked_dir]}，而是朝向画面{zh_dir[lr_dir]}。"
            )
        else:
            prompt = (
                f"Is this 3D model facing {asked_dir} in this shaded render? "
                "Verify the forward orientation using <point>, <box>, and <dash>, then answer Yes or No."
            )
            think = "\n".join(
                [
                    f"Checking horizontal facing orientation of the shaded 3D {title}:",
                    f"- Left boundary part: {fmt_point(left_b['cx'], left_b['cy'], left_b['name'], z=left_b['depth_tier'])} {fmt_box(*left_b['box'], left_b['name'])}",
                    f"- Right boundary part: {fmt_point(right_b['cx'], right_b['cy'], right_b['name'], z=right_b['depth_tier'])} {fmt_box(*right_b['box'], right_b['name'])}",
                    f"- Projected 3D forward arrow: {face_dash}",
                    f"Forward horizontal delta dx={facing['dx']:.1f} proves the model faces {lr_dir}, so asking '{asked_dir}' is {'Yes' if want_yes else 'No'}.",
                ]
            )
            ans_body = (
                f"Yes, the 3D model is facing {lr_dir} in this render."
                if want_yes
                else f"No, the 3D model is facing {lr_dir}, not {asked_dir}."
            )

    row = {
        "id": f"spatial3d_yn_{aid}_{view_tag}_{subtype}_{seq_idx}",
        "source_group": f"{fam}_yn",
        "language": lang,
        "task": "spatial_3d_yesno_depth_and_lr",
        "visual_kind": "independent_image",
        "enable_thinking": True,
        "frames": [member],
        "media_times_s": [0.0],
        "visual_units": [base_u],
        "prompt": prompt,
        "answer": _wrap_think(think, ans_body),
        "target": {"kind": "text", "text": ans_body},
        "metadata": {
            "provenance": make_clean_room_provenance(aid, title=f"spatial3d_yesno_{subtype}:{aid}", license_spdx="CC0-1.0"),
            "yesno_subtype": subtype,
            "ground_truth_yesno": "yes" if want_yes else "no",
            "studio_preset": ren["studio_preset"],
        },
    }
    return row, {member: ren["png_bytes"]}


def build_shaded_3d_spatial_primitives_records(
    *,
    seed: int = 20268,
    panel_size: tuple[int, int] = (384, 384),
    angles_per_asset: int = 6,
) -> list[tuple[dict[str, Any], dict[str, bytes]]]:
    """Generate full-color shaded 3D Visual Primitives (`<point>`, `<box>`, `<dash>`, `<circle>`) covering:
    1. `spatial_3d_kinematic_dash_and_joints` (with multi-segment spine/neck/tail/wing chains + EN/ZH)
    2. `spatial_3d_front_back_depth_order` (with multi-hop front/back + left/right relational chains + EN/ZH)
    3. `spatial_3d_left_right_orientation_facing` (curing C-RADIOv4 horizontal-flip artifact + EN/ZH)
    4. `spatial_3d_multiview_correspondence` (4-angle 3D depth/side flip tracking)
    5. `spatial_3d_rig_animation_arc` (genuine 2-frame Rest -> SE(3) Articulated Action motion arcs + EN/ZH)
    6. `spatial_3d_yesno_depth_and_lr` (strictly balanced 50/50 Yes/No curing POPE yes-bias + EN/ZH)
    """
    ensure_synccraft_importable()
    from synccraft.kinematics.se3_chain import compile_posed_geometry

    rng = random.Random(seed)
    assets = _collect_clean_3d_assets(seed=seed)
    if not assets:
        return []

    # Balanced orbital viewpoints covering left-facing, right-facing, front-3/4, rear-3/4, high/low!
    canonical_angles = [
        ("front_right_34", 36.0, 18.0),
        ("front_left_34", -36.0, 18.0),
        ("right_profile", 84.0, 14.0),
        ("left_profile", -84.0, 14.0),
        ("rear_right_34", 142.0, 22.0),
        ("rear_left_34", -142.0, 22.0),
        ("high_hero_right", 48.0, 38.0),
        ("low_hero_left", -52.0, 8.0),
    ]

    pairs: list[tuple[dict[str, Any], dict[str, bytes]]] = []
    # Track exact 50/50 alternation per (language, subtype) for Yes/No verification!
    yn_toggle: dict[tuple[str, str], int] = {}
    yn_seq = 0
    rec_counter = 0

    for a_idx, asset in enumerate(assets):
        aid = asset["asset_id"]
        fam = asset["family_id"]
        title = asset["title"]
        prog = asset["program"]
        sc_rest = asset["scene"]
        baked = asset["baked"]

        chosen_angles = canonical_angles[: max(4, min(len(canonical_angles), int(angles_per_asset)))]
        rendered_views: list[tuple[str, dict[str, Any]]] = []
        for v_i, (view_tag, base_az, base_el) in enumerate(chosen_angles):
            az = base_az + rng.uniform(-6.0, 6.0)
            el = max(5.0, min(68.0, base_el + rng.uniform(-4.0, 4.0)))
            pan = (rng.uniform(-0.07, 0.07), rng.uniform(-0.05, 0.05))
            preset_idx = (a_idx + v_i) % len(STUDIO_LIGHTING_PRESETS)
            ren = render_shaded_3d_with_spatial_annotations(
                prog,
                baked,
                az_deg=az,
                el_deg=el,
                pan_uv=pan,
                panel_size=panel_size,
                studio_preset=preset_idx,
                scene=sc_rest,
            )
            rendered_views.append((view_tag, ren))

        # Single-view Tasks A, B, C, F across shaded 3D angles
        for v_idx, (view_tag, ren) in enumerate(rendered_views):
            bones = ren["bones"]
            parts = ren["spatial_parts"]
            edges = ren["kinematic_edges"]
            facing = ren["facing"]
            if len(parts) < 3:
                continue

            member = f"spatial3d/{aid}/{view_tag}.png"
            base_u = {
                "visual_kind": "independent_image",
                "view_id": "iso",
                "camera_right": ren["camera_right"],
                "camera_up": ren["camera_up"],
                "ortho_scale": 1.0,
                "is_pixel_art_atlas": False,
            }
            sorted_by_x = sorted(parts.values(), key=lambda b: b["cx"])

            # --- Task A: Kinematic <dash> + Multi-Segment Structural Chains + <circle> Joint Pivots ---
            if edges and len(bones) >= 3:
                lang_a = "zh" if ((a_idx + v_idx) % 5 == 0) else "en"
                chains = _extract_structural_chains(bones, edges)
                top_edges = sorted(edges, key=lambda e: e["dist"], reverse=True)[:7]
                # Prefer chain edges first so multi-segment structural flow is always explicit!
                ordered_edges: list[dict[str, Any]] = []
                seen_ek: set[tuple[str, str]] = set()
                for ch in chains[:2]:
                    for e in ch[:4]:
                        ek = (e["parent"], e["child"])
                        if ek not in seen_ek:
                            seen_ek.add(ek)
                            ordered_edges.append(e)
                for e in top_edges:
                    ek = (e["parent"], e["child"])
                    if ek not in seen_ek and len(ordered_edges) < 6:
                        seen_ek.add(ek)
                        ordered_edges.append(e)
                pivot_bones: list[str] = []
                for e in ordered_edges[:5]:
                    for bn in (e["parent"], e["child"]):
                        if bn not in pivot_bones and len(pivot_bones) < 5:
                            pivot_bones.append(bn)

                if lang_a == "zh":
                    think_lines = [
                        f"分析着色3D模型 {title} 在 {view_tag} 视角（方位角 {ren['az_deg']:.0f}°，俯仰角 {ren['el_deg']:.0f}°）下的骨骼层次：",
                        "1. 用 <circle> 标出旋转关节轴点，并用 <box> 与 <point> 标注各子组件边界与深度层：",
                    ]
                    for bn in pivot_bones:
                        bi = bones[bn]
                        think_lines.append(
                            f"   - `{bn}`：{fmt_circle(bi['pivot_xy'][0], bi['pivot_xy'][1], bi['radius'], f'pivot:{bn}')} "
                            f"{fmt_box(*bi['box'], bn)} {fmt_point(bi['cx'], bi['cy'], bn, z=bi['depth_tier'])}"
                        )
                    think_lines.append("2. 沿多段结构链使用有向 <dash> 追踪父骨到子骨的运动连杆：")
                    chain_summary_zh = []
                    for e in ordered_edges[:6]:
                        z_zh = "向后延伸" if e["dz"] > 0.6 else ("向前延伸" if e["dz"] < -0.6 else "近似共面")
                        edge_lbl = f"{e['parent']}->{e['child']}"
                        think_lines.append(
                            f"   - {fmt_dash(e['x1'], e['y1'], e['x2'], e['y2'], edge_lbl)}（{z_zh}）"
                        )
                        chain_summary_zh.append(f"`{e['parent']}` -> `{e['child']}`（{z_zh}）")
                    prompt_a = (
                        "请直接根据这张全彩着色3D渲染图分析模型的骨骼拓扑结构。"
                        "使用 <circle> 标注关节旋转轴点、<box> 框选子部件、<point> 标明远近深度层（near/mid/far），"
                        "并用 <dash> 串联父骨到子骨的多段结构链。"
                    )
                    ans_body_a = (
                        f"着色3D模型 {title} 的主要骨骼运动链如下：\n"
                        + "\n".join(f"- {c}" for c in chain_summary_zh)
                        + f"\n关键驱动关节轴点：{', '.join(pivot_bones[:5])}。"
                    )
                else:
                    think_lines = [
                        f"Inspecting the shaded 3D {title} from {view_tag} (az={ren['az_deg']:.0f}deg, el={ren['el_deg']:.0f}deg).",
                        "1. Mark rotational joint pivots with <circle>, sub-assemblies with <box>, and centers with <point>:",
                    ]
                    for bn in pivot_bones:
                        bi = bones[bn]
                        think_lines.append(
                            f"   - {bn}: {fmt_circle(bi['pivot_xy'][0], bi['pivot_xy'][1], bi['radius'], f'pivot:{bn}')} "
                            f"{fmt_box(*bi['box'], bn)} {fmt_point(bi['cx'], bi['cy'], bn, z=bi['depth_tier'])}"
                        )
                    if chains:
                        chain_str = " -> ".join([chains[0][0]["parent"]] + [e["child"] for e in chains[0][:4]])
                        think_lines.append(f"2. Trace multi-segment structural flow ({chain_str}) and child <dash> bone axes:")
                    else:
                        think_lines.append("2. Trace directed parent-to-child skeletal <dash> bone axes:")
                    chain_summary = []
                    for e in ordered_edges[:6]:
                        z_rel = "toward_back" if e["dz"] > 0.6 else ("toward_front" if e["dz"] < -0.6 else "coplanar")
                        edge_lbl = f"{e['parent']}->{e['child']}"
                        think_lines.append(
                            f"   - {fmt_dash(e['x1'], e['y1'], e['x2'], e['y2'], edge_lbl)} ({z_rel})"
                        )
                        chain_summary.append(f"{e['parent']} -> {e['child']} ({z_rel})")
                    prompt_a = (
                        "Analyze the 3D skeletal structure directly from this shaded studio render. "
                        "Use <circle> for joint pivots, <box> for sub-assemblies, <point> with depth tier (near/mid/far), "
                        "and <dash> to trace multi-segment parent-to-child bone chains."
                    )
                    ans_body_a = (
                        f"Primary 3D kinematic chains on the shaded {title}:\n"
                        + "\n".join(f"- {c}" for c in chain_summary)
                        + f"\nKey articulation pivots: {', '.join(pivot_bones[:5])}."
                    )

                row_kin = {
                    "id": f"spatial3d_kin_{aid}_{view_tag}",
                    "source_group": f"{fam}_kin",
                    "language": lang_a,
                    "task": "spatial_3d_kinematic_dash_and_joints",
                    "visual_kind": "independent_image",
                    "enable_thinking": True,
                    "frames": [member],
                    "media_times_s": [0.0],
                    "visual_units": [base_u],
                    "prompt": prompt_a,
                    "answer": _wrap_think("\n".join(think_lines), ans_body_a),
                    "target": {"kind": "text", "text": ans_body_a},
                    "metadata": {
                        "provenance": make_clean_room_provenance(aid, title=f"spatial3d_kinematic_dash:{aid}", license_spdx="CC0-1.0"),
                        "primitives": ["point", "box", "dash", "circle"],
                        "studio_preset": ren["studio_preset"],
                    },
                }
                pairs.append((row_kin, {member: ren["png_bytes"]}))
                rec_counter += 1

            # --- Task B: Multi-Hop Front/Back + Left/Right Spatial Depth & Occlusion Ordering ---
            trio = _select_distinct_depth_trio(parts)
            if trio is not None:
                front_b, mid_b, back_b = trio
                lang_b = "zh" if ((a_idx + v_idx) % 5 == 1) else "en"
                swap_order = (v_idx % 2 == 1)
                opt_a, opt_b = (back_b, front_b) if swap_order else (front_b, back_b)
                closer, mid_name, farther = front_b["name"], mid_b["name"], back_b["name"]

                # Compute secondary hop (front->mid and mid->back) + horizontal side relation!
                lr_fm = "left" if front_b["cx"] < mid_b["cx"] - 18 else ("right" if front_b["cx"] > mid_b["cx"] + 18 else "aligned")
                lr_mb = "left" if mid_b["cx"] < back_b["cx"] - 18 else ("right" if mid_b["cx"] > back_b["cx"] + 18 else "aligned")
                hop1_ok = _clamped_dist(front_b["cx"], front_b["cy"], mid_b["cx"], mid_b["cy"]) >= 8.0
                hop2_ok = _clamped_dist(mid_b["cx"], mid_b["cy"], back_b["cx"], back_b["cy"]) >= 8.0

                if lang_b == "zh":
                    lr_map_zh = {"left": "左侧", "right": "右侧", "aligned": "近乎同列"}
                    t_lines = [
                        f"在着色3D渲染图中对 {title} 的部件进行多跳前后深度与左右位置推理：",
                        f"- 最近前景：{fmt_point(front_b['cx'], front_b['cy'], closer, z='near')} {fmt_box(*front_b['box'], closer)}",
                        f"- 中层核心：{fmt_point(mid_b['cx'], mid_b['cy'], mid_name, z='mid')} {fmt_circle(mid_b['pivot_xy'][0], mid_b['pivot_xy'][1], mid_b['radius'], f'pivot:{mid_name}')}",
                        f"- 最远后景：{fmt_point(back_b['cx'], back_b['cy'], farther, z='far')} {fmt_box(*back_b['box'], farther)}",
                        f"- 前后深度主射线：{fmt_dash(front_b['cx'], front_b['cy'], back_b['cx'], back_b['cy'], f'front_to_back:{closer}->{farther}')}",
                    ]
                    if hop1_ok and hop2_ok:
                        t_lines.append(
                            f"- 两跳深度链：{fmt_dash(front_b['cx'], front_b['cy'], mid_b['cx'], mid_b['cy'], f'hop1:{closer}->{mid_name}')} "
                            f"与 {fmt_dash(mid_b['cx'], mid_b['cy'], back_b['cx'], back_b['cy'], f'hop2:{mid_name}->{farther}')}"
                        )
                    t_lines.append(
                        f"综合空间关系：`{closer}` 位于 `{mid_name}` 前方且偏{lr_map_zh[lr_fm]}，`{mid_name}` 位于 `{farther}` 前方且偏{lr_map_zh[lr_mb]}。"
                    )
                    prompt_b = (
                        f"直接观察这张着色3D模型渲染图，分析 `{opt_a['name']}`、`{mid_name}` 与 `{opt_b['name']}` "
                        "的前后空间顺序（谁更靠近镜头、谁在最远后方），并说明相邻两跳之间的左右相对位置。"
                    )
                    ans_fb = (
                        f"由近及远的前后深度顺序为：`{closer}`（前景） -> `{mid_name}`（中景） -> `{farther}`（后景）。"
                        f"同时在画面水平方向上，`{closer}` 位于 `{mid_name}` 的{lr_map_zh[lr_fm]}，`{mid_name}` 位于 `{farther}` 的{lr_map_zh[lr_mb]}。"
                    )
                else:
                    t_lines = [
                        f"Multi-hop front-to-back depth and left/right reasoning on shaded 3D {title}:",
                        f"- Nearest foreground part: {fmt_point(front_b['cx'], front_b['cy'], closer, z='near')} {fmt_box(*front_b['box'], closer)}",
                        f"- Mid-depth bridge: {fmt_point(mid_b['cx'], mid_b['cy'], mid_name, z='mid')} {fmt_circle(mid_b['pivot_xy'][0], mid_b['pivot_xy'][1], mid_b['radius'], f'pivot:{mid_name}')}",
                        f"- Farthest background part: {fmt_point(back_b['cx'], back_b['cy'], farther, z='far')} {fmt_box(*back_b['box'], farther)}",
                        f"- Overall front-to-back ray: {fmt_dash(front_b['cx'], front_b['cy'], back_b['cx'], back_b['cy'], f'front_to_back:{closer}->{farther}')}",
                    ]
                    if hop1_ok and hop2_ok:
                        t_lines.append(
                            f"- Two-hop depth chain: {fmt_dash(front_b['cx'], front_b['cy'], mid_b['cx'], mid_b['cy'], f'hop1:{closer}->{mid_name}')} "
                            f"then {fmt_dash(mid_b['cx'], mid_b['cy'], back_b['cx'], back_b['cy'], f'hop2:{mid_name}->{farther}')}"
                        )
                    t_lines.append(
                        f"Horizontally, `{closer}` sits {lr_fm} of `{mid_name}` while `{mid_name}` sits {lr_mb} of `{farther}`, "
                        f"confirming `{closer}` is strictly closest to the camera and `{farther}` is farthest back."
                    )
                    prompt_b = (
                        f"Looking directly at this shaded 3D render, determine the multi-hop front-to-back depth order "
                        f"and left/right relation across `{opt_a['name']}`, `{mid_name}`, and `{opt_b['name']}`. "
                        "Which part is closest to the camera and which is farthest back?"
                    )
                    ans_fb = (
                        f"`{closer}` is closest to the camera (in front), `{mid_name}` lies at mid-depth, "
                        f"and `{farther}` is farthest back (depth chain: `{closer}` -> `{mid_name}` -> `{farther}`). "
                        f"Horizontally, `{closer}` is {lr_fm} of `{mid_name}`, and `{mid_name}` is {lr_mb} of `{farther}`."
                    )

                row_fb = {
                    "id": f"spatial3d_fb_{aid}_{view_tag}",
                    "source_group": f"{fam}_fb",
                    "language": lang_b,
                    "task": "spatial_3d_front_back_depth_order",
                    "visual_kind": "independent_image",
                    "enable_thinking": True,
                    "frames": [member],
                    "media_times_s": [0.0],
                    "visual_units": [base_u],
                    "prompt": prompt_b,
                    "answer": _wrap_think("\n".join(t_lines), ans_fb),
                    "target": {"kind": "text", "text": ans_fb},
                    "metadata": {
                        "provenance": make_clean_room_provenance(aid, title=f"spatial3d_front_back:{aid}", license_spdx="CC0-1.0"),
                        "front_bone": closer,
                        "back_bone": farther,
                        "studio_preset": ren["studio_preset"],
                    },
                }
                pairs.append((row_fb, {member: ren["png_bytes"]}))
                rec_counter += 1

            # --- Task C: Explicit Left / Right Facing & Orientation QA (cures C-RADIOv4 horizontal-flip artifact!) ---
            if v_idx < 4 and len(sorted_by_x) >= 3 and facing["dist"] >= 10.0:
                left_b = sorted_by_x[0]
                right_b = sorted_by_x[-1]
                if _clamped_dist(left_b["cx"], left_b["cy"], right_b["cx"], right_b["cy"]) >= 12.0:
                    lang_c = "zh" if ((a_idx + v_idx) % 5 == 2) else "en"
                    lr_dir = facing["lr"]
                    fb_dir = facing["fb"]
                    lr_lbl = f"left_to_right:{left_b['name']}->{right_b['name']}"
                    face_dash = fmt_dash(facing["x1"], facing["y1"], facing["x2"], facing["y2"], f"facing_{lr_dir}_{fb_dir}")
                    if lang_c == "zh":
                        lr_zh = {"left": "画面左侧", "right": "画面右侧", "center": "画面正中"}[lr_dir]
                        fb_zh = {"toward_camera": "朝向镜头", "away_from_camera": "背离镜头", "profile": "侧面轮廓"}[fb_dir]
                        think_lr = "\n".join(
                            [
                                f"在全彩着色3D渲染图上判定 {title} 的左右朝向与边缘组件位置：",
                                f"- 最左侧可见组件：{fmt_point(left_b['cx'], left_b['cy'], left_b['name'], z=left_b['depth_tier'])} {fmt_box(*left_b['box'], left_b['name'])}",
                                f"- 最右侧可见组件：{fmt_point(right_b['cx'], right_b['cy'], right_b['name'], z=right_b['depth_tier'])} {fmt_box(*right_b['box'], right_b['name'])}",
                                f"- 左右跨度轴线：{fmt_dash(left_b['cx'], left_b['cy'], right_b['cx'], right_b['cy'], lr_lbl)}",
                                f"- 3D前向朝向矢量：{face_dash}",
                                f"前向水平分量 dx={facing['dx']:.1f}，确认模型朝向{lr_zh}（{fb_zh}）。",
                            ]
                        )
                        prompt_c = (
                            "请根据这张着色3D渲染图判断该3D模型的正面朝向（向左、向右、朝向镜头或背离镜头），"
                            "指出画面最左侧与最右侧的组件，并用 <dash> 绘制3D朝向矢量。"
                        )
                        ans_lr = (
                            f"该3D模型正朝向{lr_zh}（{fb_zh}）。画面左侧为 `{left_b['name']}`，右侧为 `{right_b['name']}`。"
                        )
                    else:
                        if lr_dir in ("left", "right"):
                            facing_phrase = (
                                f"facing {lr_dir} ({fb_dir.replace('_', ' ')})"
                                if fb_dir != "profile"
                                else f"facing {lr_dir} in side profile"
                            )
                        else:
                            facing_phrase = f"facing {fb_dir.replace('_', ' ')} (centered horizontally)"
                        think_lr = "\n".join(
                            [
                                f"Determining exact left/right orientation and facing vector on the shaded 3D {title}:",
                                f"- Leftmost visible assembly: {fmt_point(left_b['cx'], left_b['cy'], left_b['name'], z=left_b['depth_tier'])} {fmt_box(*left_b['box'], left_b['name'])}",
                                f"- Rightmost visible assembly: {fmt_point(right_b['cx'], right_b['cy'], right_b['name'], z=right_b['depth_tier'])} {fmt_box(*right_b['box'], right_b['name'])}",
                                f"- Left-to-right span axis: {fmt_dash(left_b['cx'], left_b['cy'], right_b['cx'], right_b['cy'], lr_lbl)}",
                                f"- 3D forward facing arrow: {face_dash}",
                                f"Screen delta dx={facing['dx']:.1f} confirms the model is {facing_phrase}, with `{left_b['name']}` on the left and `{right_b['name']}` on the right.",
                            ]
                        )
                        prompt_c = (
                            "From this shaded 3D render, determine which direction the 3D model is facing "
                            "(left, right, toward camera, or away from camera) and identify which sub-assembly sits on the "
                            "left side versus the right side of the frame. Trace the facing vector with <dash>."
                        )
                        ans_lr = (
                            f"The 3D model is {facing_phrase}. On screen, `{left_b['name']}` sits on the left side "
                            f"and `{right_b['name']}` sits on the right side."
                        )
                    row_lr = {
                        "id": f"spatial3d_lr_{aid}_{view_tag}",
                        "source_group": f"{fam}_lr",
                        "language": lang_c,
                        "task": "spatial_3d_left_right_orientation_facing",
                        "visual_kind": "independent_image",
                        "enable_thinking": True,
                        "frames": [member],
                        "media_times_s": [0.0],
                        "visual_units": [base_u],
                        "prompt": prompt_c,
                        "answer": _wrap_think(think_lr, ans_lr),
                        "target": {"kind": "text", "text": ans_lr},
                        "metadata": {
                            "provenance": make_clean_room_provenance(aid, title=f"spatial3d_left_right_orientation:{aid}", license_spdx="CC0-1.0"),
                            "facing_lr": lr_dir,
                            "facing_fb": fb_dir,
                            "studio_preset": ren["studio_preset"],
                        },
                    }
                    pairs.append((row_lr, {member: ren["png_bytes"]}))
                    rec_counter += 1

            # --- Task F: Strictly Balanced 50/50 Spatial Yes/No Verification (POPE Yes-Bias & L/R Cure!) ---
            # Cycle across depth, left_right, facing subtypes with strict per-(lang, subtype) 50/50 toggle!
            subtypes = ("depth", "left_right", "facing")
            sub = subtypes[(a_idx + v_idx) % len(subtypes)]
            lang_yn = "zh" if ((a_idx + v_idx) % 5 == 3) else "en"
            tkey = (lang_yn, sub)
            cur_t = yn_toggle.get(tkey, 0)
            want_yes = (cur_t % 2 == 0)
            yn_pair = _build_yesno_record(
                aid=aid,
                fam=fam,
                title=title,
                view_tag=view_tag,
                member=member,
                base_u=base_u,
                ren=ren,
                subtype=sub,
                want_yes=want_yes,
                lang=lang_yn,
                seq_idx=yn_seq,
            )
            if yn_pair is not None:
                yn_toggle[tkey] = cur_t + 1
                yn_seq += 1
                pairs.append(yn_pair)
                rec_counter += 1

        # --- Task D: Multi-View 3D Cross-Angle Correspondence & Front/Back Flip ---
        if len(rendered_views) >= 4:
            v_indices = [0, 1, 2, 4] if len(rendered_views) >= 5 else [0, 1, 2, 3]
            mv_frames: list[str] = []
            mv_units: list[dict[str, Any]] = []
            mv_blobs: dict[str, bytes] = {}
            mv_rens: list[tuple[str, dict[str, Any]]] = []
            for vi in v_indices:
                vtag, vren = rendered_views[vi]
                mpath = f"spatial3d/{aid}/{vtag}.png"
                mv_frames.append(mpath)
                mv_blobs[mpath] = vren["png_bytes"]
                mv_units.append(
                    {
                        "visual_kind": "multiview_render",
                        "view_id": "iso" if "34" in vtag else "right",
                        "camera_right": vren["camera_right"],
                        "camera_up": vren["camera_up"],
                        "ortho_scale": 1.0,
                        "is_pixel_art_atlas": False,
                    }
                )
                mv_rens.append((vtag, vren))

            common_parts = set(mv_rens[0][1]["spatial_parts"].keys())
            for _, vr in mv_rens[1:]:
                common_parts &= set(vr["spatial_parts"].keys())
            best_b = None
            best_swing = -1.0
            for bn in sorted(common_parts):
                rels = [vr["spatial_parts"][bn]["depth_rel"] for _, vr in mv_rens]
                swing = max(rels) - min(rels)
                if swing > best_swing:
                    best_swing = swing
                    best_b = bn
            if best_b is not None:
                think_mv = [f"Tracking `{best_b}` across 4 orbital views of shaded 3D {title}:"]
                summary_mv = []
                for idx_v, (vtag, vr) in enumerate(mv_rens, start=1):
                    bi = vr["spatial_parts"][best_b]
                    side_lr = "left" if bi["cx"] < 440 else ("right" if bi["cx"] > 560 else "center")
                    f_lr = vr["facing"]["lr"]
                    dash_str = (
                        f" {fmt_dash(vr['facing']['x1'], vr['facing']['y1'], vr['facing']['x2'], vr['facing']['y2'], f'v{idx_v}_facing_{f_lr}')}"
                        if vr["facing"]["dist"] >= 10.0
                        else ""
                    )
                    think_mv.append(
                        f"- View {idx_v} [{vtag}, faces {f_lr}]: "
                        f"{fmt_point(bi['cx'], bi['cy'], best_b, z=bi['depth_tier'])} "
                        f"{fmt_circle(bi['pivot_xy'][0], bi['pivot_xy'][1], bi['radius'], f'pivot:{best_b}')}"
                        f"{dash_str} -> {side_lr}/{bi['depth_tier']}"
                    )
                    summary_mv.append(f"V{idx_v} ({vtag}): {side_lr} frame sector, {bi['depth_tier']} depth")
                ans_mv = (
                    f"Cross-view 3D correspondence for `{best_b}`:\n"
                    + "\n".join(f"- {s}" for s in summary_mv)
                    + "\nIts front/back depth tier and left/right screen sector transform consistently with the 3D orbit."
                )
                row_mv = {
                    "id": f"spatial3d_mv_{aid}",
                    "source_group": f"{fam}_mv",
                    "language": "en",
                    "task": "spatial_3d_multiview_correspondence",
                    "visual_kind": "multiview_render",
                    "enable_thinking": True,
                    "frames": mv_frames,
                    "media_times_s": [0.0] * len(mv_frames),
                    "visual_units": mv_units,
                    "prompt": (
                        f"These 4 shaded renders show the same 3D {title} from different angles. "
                        f"Track `{best_b}` across all 4 views using <point>, <circle> pivot, and <dash> orientation vectors, "
                        "and describe how its left/right screen position and front/back depth change as the camera orbits."
                    ),
                    "answer": _wrap_think("\n".join(think_mv), ans_mv),
                    "target": {"kind": "text", "text": ans_mv},
                    "metadata": {
                        "provenance": make_clean_room_provenance(aid, title=f"spatial3d_multiview_correspondence:{aid}", license_spdx="CC0-1.0"),
                        "tracked_bone": best_b,
                    },
                }
                pairs.append((row_mv, mv_blobs))
                rec_counter += 1

        # --- Task E: Genuine 2-Frame SE(3) Rig Animation Arc (`spatial_3d_rig_animation_arc`) ---
        recipes = _select_pose_recipes_for_asset(sc_rest)
        if recipes:
            # Generate up to 2 diverse articulated actions per asset across clean studio viewpoints!
            n_anim_actions = min(2, len(recipes))
            selected_recipes = [recipes[(a_idx + k) % len(recipes)] for k in range(n_anim_actions)]
            for r_k, recipe in enumerate(selected_recipes):
                try:
                    sc_posed = compile_posed_geometry(
                        prog,
                        pose_euler_deltas=recipe["deltas"],
                        clamp_joint_limits=False,
                    )
                except Exception:
                    continue
                baked_posed = _repose_baked_for_scene(baked, sc_posed)
                q_rest, _, _ = _extract_world_quads_and_parts(baked)
                q_posed, _, _ = _extract_world_quads_and_parts(baked_posed)
                union_pts = np.concatenate([q[0] for q in q_rest] + [q[0] for q in q_posed], axis=0)

                # Choose an orbital viewpoint that shows the articulation clearly (try 2 angles if needed)
                anim_angles = [
                    (42.0 if (a_idx + r_k) % 2 == 0 else -42.0, 18.0),
                    (78.0 if (a_idx + r_k) % 2 == 0 else -78.0, 16.0),
                ]
                preset_anim = (a_idx + r_k + 1) % len(STUDIO_LIGHTING_PRESETS)
                piv_name = recipe["pivot_bone"]
                eff_name = recipe["effector_bone"]

                built_anim = False
                for anim_az, anim_el in anim_angles:
                    shared_cam = _build_perspective_camera(
                        union_pts,
                        az_deg=anim_az + rng.uniform(-5.0, 5.0),
                        el_deg=anim_el + rng.uniform(-3.0, 3.0),
                        fill_frac=0.76,
                        pan_uv=(0.0, 0.0),
                        panel_size=panel_size,
                    )
                    ren_f1 = render_shaded_3d_with_spatial_annotations(
                        prog,
                        baked,
                        panel_size=panel_size,
                        studio_preset=preset_anim,
                        scene=sc_rest,
                        camera_override=shared_cam,
                    )
                    ren_f2 = render_shaded_3d_with_spatial_annotations(
                        prog,
                        baked_posed,
                        panel_size=panel_size,
                        studio_preset=preset_anim,
                        scene=sc_posed,
                        camera_override=shared_cam,
                    )
                    if eff_name not in ren_f1["bones"] or eff_name not in ren_f2["bones"] or piv_name not in ren_f1["bones"]:
                        continue
                    b_before = ren_f1["bones"][eff_name]
                    b_after = ren_f2["bones"][eff_name]
                    b_piv = ren_f1["bones"][piv_name]

                    disp_2d = _clamped_dist(b_before["cx"], b_before["cy"], b_after["cx"], b_after["cy"])
                    pix_diff = float(np.mean(np.abs(ren_f2["rgb_array"].astype(np.float32) - ren_f1["rgb_array"].astype(np.float32))))
                    if disp_2d < 10.0 or pix_diff < 0.6:
                        continue

                    dx = b_after["cx"] - b_before["cx"]
                    dy = b_after["cy"] - b_before["cy"]
                    delta_cz = b_after["cz_mean"] - b_before["cz_mean"]
                    depth_move = "moves_closer" if delta_cz < -0.40 else ("moves_back" if delta_cz > 0.40 else "coplanar")
                    h_dir = "right" if dx > 8.0 else ("left" if dx < -8.0 else "horizontally steady")
                    v_dir = "upward" if dy < -8.0 else ("downward" if dy > 8.0 else "vertically steady")
                    pd = recipe["primary_delta"]

                    # Collect posed structural chain <dash> segments if >= 2 bones in recipe chain
                    chain_dashes: list[str] = []
                    c_bones = recipe["chain_bones"]
                    for ci in range(len(c_bones) - 1):
                        u_b, v_b = c_bones[ci], c_bones[ci + 1]
                        if u_b in ren_f2["bones"] and v_b in ren_f2["bones"]:
                            bu, bv = ren_f2["bones"][u_b], ren_f2["bones"][v_b]
                            if _clamped_dist(bu["cx"], bu["cy"], bv["cx"], bv["cy"]) >= 8.0:
                                chain_dashes.append(fmt_dash(bu["cx"], bu["cy"], bv["cx"], bv["cy"], f"chain:{u_b}->{v_b}"))

                    # Radius of articulation sweep around pivot covering the swing arc!
                    arc_r = max(
                        18.0,
                        min(
                            220.0,
                            max(
                                _clamped_dist(b_piv["pivot_xy"][0], b_piv["pivot_xy"][1], b_before["cx"], b_before["cy"]),
                                _clamped_dist(b_piv["pivot_xy"][0], b_piv["pivot_xy"][1], b_after["cx"], b_after["cy"]),
                                b_piv["radius"],
                            ),
                        ),
                    )
                    piv_circle = fmt_circle(b_piv["pivot_xy"][0], b_piv["pivot_xy"][1], arc_r, f"pivot:{piv_name}")
                    pt_before = fmt_point(b_before["cx"], b_before["cy"], f"before:{eff_name}", z=b_before["depth_tier"])
                    box_before = fmt_box(*b_before["box"], f"before:{eff_name}")
                    pt_after = fmt_point(b_after["cx"], b_after["cy"], f"after:{eff_name}", z=b_after["depth_tier"])
                    box_after = fmt_box(*b_after["box"], f"after:{eff_name}")
                    arc_dash = fmt_dash(b_before["cx"], b_before["cy"], b_after["cx"], b_after["cy"], f"anim_arc:{eff_name}")

                    lang_e = "zh" if ((a_idx + r_k) % 4 == 0) else "en"
                    if lang_e == "zh":
                        d_zh = {"moves_closer": "向镜头拉近（变近）", "moves_back": "远离镜头（后退）", "coplanar": "保持近乎同深度"}[depth_move]
                        h_zh = "向右" if dx > 8.0 else ("向左" if dx < -8.0 else "水平微移")
                        v_zh = "向上" if dy < -8.0 else ("向下" if dy > 8.0 else "垂直微移")
                        t_anim = [
                            f"对比全彩着色3D模型 {title} 在静息帧（帧1）与动作帧（帧2，{recipe['action_zh']}）间的局部SE(3)骨骼旋转：",
                            f"- 旋转关节轴点与扫掠半径：{piv_circle}（局部欧拉增量 {pd[0]:+.0f}°/{pd[1]:+.0f}°/{pd[2]:+.0f}°）",
                            f"- 帧1静息位置：{pt_before} {box_before}",
                            f"- 帧2动作位置：{pt_after} {box_after}",
                            f"- 2D屏幕位移弧向量：{arc_dash}（屏幕方向：{h_zh}、{v_zh}；3D前后深度：{d_zh}）",
                        ]
                        if chain_dashes:
                            t_anim.append(f"- 动作帧骨骼结构链：{' '.join(chain_dashes[:3])}")
                        prompt_e = (
                            f"这两张同机位全彩着色3D渲染图展示了 {title} 从静息姿态（帧1）到动作姿态（帧2）的骨骼运动。"
                            f"请用 <circle> 标出 `{piv_name}` 关节轴点，用 <point>/<box> 标注 `{eff_name}` 的运动前后位置，"
                            "用 <dash> 连接运动位移弧，并说明其在2D屏幕方向与3D前后深度上的变化。"
                        )
                        ans_anim = (
                            f"绕 `{piv_name}` 关节执行{recipe['action_zh']}后，`{eff_name}` 在画面中向{h_zh}、{v_zh}摆动"
                            f"（位移约 {disp_2d:.0f} 归一化像素），并在三维深度上{d_zh}（depth={depth_move}）。"
                        )
                    else:
                        t_anim = [
                            f"Comparing fixed-camera Frame 1 (Rest) -> Frame 2 ({recipe['action_en']}) on shaded 3D {title}:",
                            f"- Articulation pivot & sweep radius: {piv_circle} (SE(3) local delta [{pd[0]:+.0f}, {pd[1]:+.0f}, {pd[2]:+.0f}] deg)",
                            f"- Frame 1 rest effector: {pt_before} {box_before}",
                            f"- Frame 2 posed effector: {pt_after} {box_after}",
                            f"- Screen motion displacement arc: {arc_dash} ({h_dir}, {v_dir}; 3D depth: {depth_move}, dZ={delta_cz:+.2f}u)",
                        ]
                        if chain_dashes:
                            t_anim.append(f"- Articulated limb chain: {' '.join(chain_dashes[:3])}")
                        prompt_e = (
                            f"These 2 fixed-camera shaded 3D renders show `{title}` transitioning from Rest (Frame 1) to an articulated pose (Frame 2). "
                            f"Mark the `{piv_name}` rotational pivot with <circle>, localize `{eff_name}` before and after with <point> and <box>, "
                            "trace the motion vector with <dash>, and state both its 2D screen shift and 3D depth change (moves_closer / moves_back / coplanar)."
                        )
                        ans_anim = (
                            f"Rotating around `{piv_name}` ({recipe['action_en']}) swings `{eff_name}` {h_dir} and {v_dir} across the frame "
                            f"(2D arc ~{disp_2d:.0f} norm-px) while its camera-space depth `{depth_move}` "
                            f"(tier: {b_before['depth_tier']} -> {b_after['depth_tier']})."
                        )

                    m_f1 = f"spatial3d/{aid}/anim_{recipe['action_id']}_rest.png"
                    m_f2 = f"spatial3d/{aid}/anim_{recipe['action_id']}_posed.png"
                    u_f1 = {
                        "visual_kind": "multiview_render",
                        "view_id": "iso",
                        "camera_right": ren_f1["camera_right"],
                        "camera_up": ren_f1["camera_up"],
                        "ortho_scale": 1.0,
                        "is_pixel_art_atlas": False,
                    }
                    u_f2 = dict(u_f1)
                    row_anim = {
                        "id": f"spatial3d_anim_{aid}_{recipe['action_id']}_{r_k}",
                        "source_group": f"{fam}_anim",
                        "language": lang_e,
                        "task": "spatial_3d_rig_animation_arc",
                        "visual_kind": "multiview_render",
                        "enable_thinking": True,
                        "frames": [m_f1, m_f2],
                        "media_times_s": [0.0, 0.5],
                        "visual_units": [u_f1, u_f2],
                        "prompt": prompt_e,
                        "answer": _wrap_think("\n".join(t_anim), ans_anim),
                        "target": {"kind": "text", "text": ans_anim},
                        "metadata": {
                            "provenance": make_clean_room_provenance(aid, title=f"spatial3d_anim_arc:{aid}:{recipe['action_id']}", license_spdx="CC0-1.0"),
                            "action_id": recipe["action_id"],
                            "pivot_bone": piv_name,
                            "effector_bone": eff_name,
                            "depth_shift": depth_move,
                            "disp_2d_norm": round(disp_2d, 2),
                            "mean_pixel_diff": round(pix_diff, 3),
                            "studio_preset": ren_f1["studio_preset"],
                        },
                    }
                    pairs.append((row_anim, {m_f1: ren_f1["png_bytes"], m_f2: ren_f2["png_bytes"]}))
                    rec_counter += 1
                    built_anim = True
                    break
                if not built_anim:
                    continue

    emit("spatial_3d_primitives_built", records=len(pairs), assets=len(assets))
    return pairs


def _convert_nemotron_boxes_to_primitives(text: str) -> str:
    """Upgrade Nemotron inline `[x1, y1, x2, y2]` coordinates inside `<think>` into `<box>`, plus `<point>`,
    `<circle>` (for round/spherical objects), or relational `<dash>` (strictly distance >= 6px).
    """
    if not text:
        return text
    found_boxes: list[tuple[float, float, float, float, str, str]] = []

    def _repl(m: re.Match[str]) -> str:
        x1, y1, x2, y2 = (int(m.group(i)) for i in (1, 2, 3, 4))
        if max(x1, y1, x2, y2) > 1000 or min(x1, y1, x2, y2) < 0:
            return m.group(0)
        lo_x, hi_x = min(x1, x2), max(x1, x2)
        lo_y, hi_y = min(y1, y2), max(y1, y2)
        cx, cy = 0.5 * (lo_x + hi_x), 0.5 * (lo_y + hi_y)
        w, h = float(hi_x - lo_x), float(hi_y - lo_y)
        tag = f"obj_{len(found_boxes) + 1}"
        ctx = text[max(0, m.start() - 48) : min(len(text), m.end() + 24)].lower()
        found_boxes.append((cx, cy, w, h, tag, ctx))
        return fmt_box(lo_x, lo_y, hi_x, hi_y, tag)

    converted = _INLINE_BBOX_RE.sub(_repl, text)
    low = converted.lower()

    extra_lines: list[str] = []
    if len(found_boxes) == 1:
        cx, cy, w, h, tag, ctx = found_boxes[0]
        if any(kw in ctx or kw in low for kw in _ROUND_OBJECT_KEYWORDS):
            rad = max(8.0, min(400.0, 0.25 * (w + h)))
            extra_lines.append(f"Grounded round focus: {fmt_circle(cx, cy, rad, tag)}")
        elif any(kw in low for kw in _LOCATION_KEYWORDS):
            extra_lines.append(f"Grounded focus point: {fmt_point(cx, cy, tag)}")
    elif len(found_boxes) >= 2 and any(kw in low for kw in _SPATIAL_REL_KEYWORDS):
        # Pick the first pair of grounded boxes with non-degenerate distance >= 6px!
        chosen_pair = None
        for i in range(len(found_boxes)):
            for j in range(i + 1, len(found_boxes)):
                cx1, cy1, _, _, l1, _ = found_boxes[i]
                cx2, cy2, _, _, l2, _ = found_boxes[j]
                if _clamped_dist(cx1, cy1, cx2, cy2) >= 6.0:
                    chosen_pair = (cx1, cy1, l1, cx2, cy2, l2)
                    break
            if chosen_pair is not None:
                break
        if chosen_pair is not None:
            cx1, cy1, l1, cx2, cy2, l2 = chosen_pair
            rel_dash = fmt_dash(cx1, cy1, cx2, cy2, f"rel:{l1}->{l2}")
            extra_lines.append(f"Spatial relation trace: {rel_dash}")

    if extra_lines:
        insertion = "\n" + "\n".join(extra_lines) + "\n"
        if "</think>" in converted:
            converted = converted.replace("</think>", f"{insertion}</think>", 1)
        else:
            converted = converted + insertion
    return converted


def _has_banned_benchmark_name(text: str) -> bool:
    low = str(text or "").lower()
    return any(b in low for b in BANNED_BENCHMARK_SUBSTRINGS)


def _validate_normalized_image_quality(jpg_bytes: bytes, *, min_side: int = 128, min_std: float = 8.0) -> bool:
    try:
        with Image.open(io.BytesIO(jpg_bytes)) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            if min(w, h) < min_side:
                return False
            arr = np.asarray(rgb, dtype=np.float32)
            return float(np.std(arr)) > min_std
    except Exception:
        return False


def _sanitize_nemotron_answer(answer: str) -> str | None:
    """Ensure every Nemotron answer has balanced `<think>...</think>`, non-empty final answer, and healthy repetition."""
    raw = str(answer or "").strip()
    if not raw:
        return None
    has_open = "<think>" in raw
    has_close = "</think>" in raw
    if not has_open and not has_close:
        # Clean direct answer -> wrap concisely so Stage-8 CoT format is uniform
        if len(raw) < 2:
            return None
        raw = _wrap_think("Inspecting the image carefully to verify the visual evidence before answering.", raw)
    elif not (has_open and has_close and raw.find("<think>") < raw.find("</think>")):
        return None
    health = check_generation_collapse_health(raw, expect_thinking=True, max_rep_4gram=0.145, max_think_chars=4400)
    return raw if health["healthy"] else None


def _extract_qa_from_nemotron_row(obj: dict[str, Any]) -> tuple[str, str, str] | None:
    msgs = obj.get("messages") or []
    img_ref = ""
    user_txt_parts: list[str] = []
    asst_txt_parts: list[str] = []
    for m in msgs:
        role = m.get("role", "")
        content = m.get("content")
        if isinstance(content, str):
            if role == "user":
                user_txt_parts.append(content)
            elif role == "assistant":
                asst_txt_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    if role == "user":
                        user_txt_parts.append(part)
                    elif role == "assistant":
                        asst_txt_parts.append(part)
                elif isinstance(part, dict):
                    if part.get("image") and not img_ref:
                        img_ref = str(part["image"])
                    if part.get("text"):
                        if role == "user":
                            user_txt_parts.append(str(part["text"]))
                        elif role == "assistant":
                            asst_txt_parts.append(str(part["text"]))
    prompt = "\n".join(p.strip() for p in user_txt_parts if p.strip()).strip()
    answer = "\n".join(p.strip() for p in asst_txt_parts if p.strip()).strip()
    if not img_ref or not prompt or not answer:
        return None
    if _has_banned_benchmark_name(prompt) or _has_banned_benchmark_name(img_ref):
        return None
    converted = _convert_nemotron_boxes_to_primitives(answer)
    clean_ans = _sanitize_nemotron_answer(converted)
    if clean_ans is None:
        return None
    if len(prompt) > 2000 or len(clean_ans) > 4600 or (len(prompt) + len(clean_ans)) > 6800:
        return None
    return img_ref, prompt, clean_ans


def _load_nemotron_embedded_tar_subset(
    *,
    subset: str,
    task_name: str,
    language: str,
    max_items: int,
    hf_token: str | None,
    seen_ids: set[str],
    prioritize_spatial: bool = False,
) -> list[tuple[dict[str, Any], dict[str, bytes]]]:
    """Stream `<subset>/<subset>.jsonl` + embedded `<subset>/media/shard_*.tar` directly from `Nemotron-Image-Training-v3`."""
    if max_items <= 0:
        return []
    from huggingface_hub import HfApi, hf_hub_download

    jsonl_path = Path(
        hf_hub_download(
            repo_id=NEMOTRON_V3_REPO,
            filename=f"{subset}/{subset}.jsonl",
            repo_type="dataset",
            token=hf_token,
        )
    )
    by_exact: dict[str, tuple[str, str, bool]] = {}
    by_name: dict[str, tuple[str, str, bool]] = {}
    by_stem: dict[str, tuple[str, str, bool]] = {}

    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = _extract_qa_from_nemotron_row(json.loads(line))
            if rec is None:
                continue
            img_ref, pr, ans = rec
            p_low = pr.lower()
            is_sp = prioritize_spatial and any(
                kw in p_low
                for kw in ("left", "right", "behind", "front", "closer", "farther", "between", "sphere", "cube", "cylinder", "angle")
            )
            name_k = Path(img_ref).name
            stem_k = Path(img_ref).stem
            for tbl, key in ((by_exact, img_ref), (by_name, name_k), (by_stem, stem_k)):
                prev = tbl.get(key)
                if prev is None or (is_sp and not prev[2]):
                    tbl[key] = (pr, ans, is_sp)

    api = HfApi(token=hf_token)
    tar_files = sorted(
        f
        for f in api.list_repo_files(NEMOTRON_V3_REPO, repo_type="dataset")
        if f.startswith(f"{subset}/media/") and f.endswith(".tar")
    )
    results: list[tuple[dict[str, Any], dict[str, bytes]]] = []
    used_keys: set[str] = set()

    for tar_rel in tar_files:
        if len(results) >= max_items:
            break
        local_tar = Path(
            hf_hub_download(
                repo_id=NEMOTRON_V3_REPO,
                filename=tar_rel,
                repo_type="dataset",
                token=hf_token,
            )
        )
        with tarfile.open(local_tar, "r:*") as tf:
            for member in tf:
                if len(results) >= max_items:
                    break
                if not member.isfile():
                    continue
                m_exact = member.name.lstrip("./")
                m_name = Path(m_exact).name
                m_stem = Path(m_exact).stem
                hit = by_exact.get(m_exact) or by_name.get(m_name) or by_stem.get(m_stem)
                if hit is None:
                    continue
                dedup_k = m_name or m_stem
                if dedup_k in used_keys:
                    continue
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                raw_bytes = fobj.read()
                norm = _normalize_image_bytes(raw_bytes, max_side=1024)
                if norm is None:
                    continue
                jpg_b, _, _ = norm
                if not _validate_normalized_image_quality(jpg_b):
                    continue
                used_keys.add(dedup_k)
                pr, ans, _ = hit
                slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", m_stem)[:48] or f"{len(results):05d}"
                rid = f"nemotron_v3_{subset}_{slug}"
                if rid in seen_ids:
                    rid = f"{rid}_{len(results):05d}"
                seen_ids.add(rid)
                mem = f"nemotron_v3/{subset}/{slug}.jpg"
                row = {
                    "id": rid,
                    "source_group": f"nemotron_v3_{subset}_{len(results) % 64:02d}",
                    "language": language,
                    "task": task_name,
                    "visual_kind": "independent_image",
                    "enable_thinking": True,
                    "frames": [mem],
                    "media_times_s": [0.0],
                    "prompt": pr,
                    "answer": ans,
                    "target": {"kind": "text", "text": ans.split("</think>")[-1].strip()},
                    "metadata": {
                        "source": f"{NEMOTRON_V3_REPO}:{subset}",
                        "license": "CC-BY-4.0",
                    },
                }
                results.append((row, {mem: jpg_b}))
    return results


def build_nemotron_v3_spatial_records(
    *,
    cache_dir: str | Path,
    hf_token: str | None = None,
    max_gqa: int = 1200,
    max_geometry3k: int = 900,
    max_aokvqa: int = 900,
    max_clevr: int = 800,
    max_zhwiki: int = 400,
    max_plotqa: int = 300,
    seed: int = 20268,
) -> list[tuple[dict[str, Any], dict[str, bytes]]]:
    """Build train-only grounded spatial & geometric reasoning records from `nvidia/Nemotron-Image-Training-v3` (`CC-BY-4.0`):
    - Embedded media tars: `clevr_1` (3D spatial scenes), `zhwiki_1` (Chinese multimodal), `plotqa_1` (charts)
    - Joined public HF train splits: `gqa_1` (+ `gqa_2`), `geometry3k`, `aokvqa_2` (`0%` eval benchmark overlap).
    """
    from huggingface_hub import hf_hub_download

    rng = random.Random(seed)
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[dict[str, Any], dict[str, bytes]]] = []
    seen_ids: set[str] = set()

    # 1. `clevr_1` embedded 3D spatial scenes (`clevr_1/clevr_1.jsonl` + `clevr_1/media/shard_*.tar`)
    if max_clevr > 0:
        try:
            clevr_pairs = _load_nemotron_embedded_tar_subset(
                subset="clevr_1",
                task_name="nemotron_v3_clevr_3d_spatial",
                language="en",
                max_items=max_clevr,
                hf_token=hf_token,
                seen_ids=seen_ids,
                prioritize_spatial=True,
            )
            pairs.extend(clevr_pairs)
            emit("nemotron_v3_clevr_loaded", count=len(clevr_pairs))
        except Exception as exc:
            emit("nemotron_v3_clevr_skipped", error=f"{type(exc).__name__}: {exc}")

    # 2. `zhwiki_1` embedded Chinese multimodal reasoning (`zhwiki_1/zhwiki_1.jsonl` + `zhwiki_1/media/shard_*.tar`)
    if max_zhwiki > 0:
        try:
            zh_pairs = _load_nemotron_embedded_tar_subset(
                subset="zhwiki_1",
                task_name="nemotron_v3_zhwiki_multimodal",
                language="zh",
                max_items=max_zhwiki,
                hf_token=hf_token,
                seen_ids=seen_ids,
                prioritize_spatial=False,
            )
            pairs.extend(zh_pairs)
            emit("nemotron_v3_zhwiki_loaded", count=len(zh_pairs))
        except Exception as exc:
            emit("nemotron_v3_zhwiki_skipped", error=f"{type(exc).__name__}: {exc}")

    # 3. `plotqa_1` embedded chart & visual reasoning (`plotqa_1/plotqa_1.jsonl` + `plotqa_1/media/shard_*.tar`)
    if max_plotqa > 0:
        try:
            plot_pairs = _load_nemotron_embedded_tar_subset(
                subset="plotqa_1",
                task_name="nemotron_v3_plotqa_chart_reasoning",
                language="en",
                max_items=max_plotqa,
                hf_token=hf_token,
                seen_ids=seen_ids,
                prioritize_spatial=False,
            )
            pairs.extend(plot_pairs)
            emit("nemotron_v3_plotqa_loaded", count=len(plot_pairs))
        except Exception as exc:
            emit("nemotron_v3_plotqa_skipped", error=f"{type(exc).__name__}: {exc}")

    # 4. `geometry3k` from Nemotron-Image-Training-v3 paired with `hiyouga/geometry3k` (`train` split ONLY!)
    if max_geometry3k > 0:
        try:
            import pyarrow.parquet as pq
            from huggingface_hub import HfApi

            geo_jsonl = Path(
                hf_hub_download(
                    repo_id=NEMOTRON_V3_REPO,
                    filename="geometry3k/geometry3k.jsonl",
                    repo_type="dataset",
                    token=hf_token,
                )
            )
            geo_by_id: dict[str, tuple[str, str]] = {}
            with geo_jsonl.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    rec = _extract_qa_from_nemotron_row(json.loads(line))
                    if rec is None:
                        continue
                    img_ref, pr, ans = rec
                    stem = Path(img_ref).parent.name or Path(img_ref).stem
                    geo_by_id.setdefault(stem, (pr, ans))

            api = HfApi(token=hf_token)
            files = [f for f in api.list_repo_files("hiyouga/geometry3k", repo_type="dataset") if "train" in f and f.endswith(".parquet")]
            added_geo = 0
            for pf in sorted(files):
                if added_geo >= max_geometry3k:
                    break
                lp = Path(hf_hub_download("hiyouga/geometry3k", pf, repo_type="dataset", token=hf_token))
                tbl = pq.read_table(lp)
                cols = set(tbl.column_names)
                img_col = "images" if "images" in cols else ("image" if "image" in cols else None)
                if img_col is None:
                    continue
                idxs = list(range(tbl.num_rows))
                rng.shuffle(idxs)
                for r_i in idxs:
                    if added_geo >= max_geometry3k:
                        break
                    raw_img_field = tbl[img_col][r_i].as_py()
                    if isinstance(raw_img_field, list) and raw_img_field:
                        raw_img_field = raw_img_field[0]
                    raw_bytes = raw_img_field.get("bytes") if isinstance(raw_img_field, dict) else None
                    if not raw_bytes:
                        continue
                    norm = _normalize_image_bytes(raw_bytes, max_side=1024)
                    if norm is None:
                        continue
                    jpg_b, _, _ = norm
                    if not _validate_normalized_image_quality(jpg_b):
                        continue
                    pid = str(tbl["problem_id"][r_i].as_py()) if "problem_id" in cols else ""
                    if pid in geo_by_id:
                        pr, ans = geo_by_id[pid]
                        src_tag = "nemotron_v3_geometry3k"
                    else:
                        pr = str(tbl["problem"][r_i].as_py() if "problem" in cols else "").replace("<image>", "").strip()
                        gt = str(tbl["answer"][r_i].as_py() if "answer" in cols else "").strip()
                        if not pr or not gt:
                            continue
                        ans = _sanitize_nemotron_answer(
                            _wrap_think(
                                f"Analyzing the geometric diagram labels and angle/length constraints carefully to solve for the target value: {gt}.",
                                gt,
                            )
                        )
                        if ans is None:
                            continue
                        src_tag = "hiyouga_geometry3k_train"
                    rid = f"nemotron_v3_geo_{added_geo:05d}"
                    seen_ids.add(rid)
                    mem = f"nemotron_v3/geometry3k/geo_{added_geo:05d}.jpg"
                    row = {
                        "id": rid,
                        "source_group": f"nemotron_v3_geo_{added_geo % 64:02d}",
                        "language": "en",
                        "task": "nemotron_v3_geometry_spatial",
                        "visual_kind": "independent_image",
                        "enable_thinking": True,
                        "frames": [mem],
                        "media_times_s": [0.0],
                        "prompt": pr,
                        "answer": ans,
                        "target": {"kind": "text", "text": ans.split("</think>")[-1].strip()},
                        "metadata": {"source": src_tag, "license": "CC-BY-4.0/MIT"},
                    }
                    pairs.append((row, {mem: jpg_b}))
                    added_geo += 1
            emit("nemotron_v3_geometry3k_loaded", count=added_geo)
        except Exception as exc:
            emit("nemotron_v3_geometry3k_skipped", error=f"{type(exc).__name__}: {exc}")

    # 5. `aokvqa_2` from Nemotron-Image-Training-v3 (`CC-BY-4.0`) joined onto `HuggingFaceM4/A-OKVQA` (`train` ONLY!)
    if max_aokvqa > 0:
        try:
            import pyarrow.parquet as pq
            from huggingface_hub import HfApi

            aok_jsonl = Path(
                hf_hub_download(
                    repo_id=NEMOTRON_V3_REPO,
                    filename="aokvqa_2/aokvqa_2.jsonl",
                    repo_type="dataset",
                    token=hf_token,
                )
            )
            aok_by_q: dict[str, tuple[str, str]] = {}
            with aok_jsonl.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    rec = _extract_qa_from_nemotron_row(json.loads(line))
                    if rec is None:
                        continue
                    _img_ref, pr, ans = rec
                    first_line = next((ln.strip() for ln in pr.splitlines() if ln.strip()), "")
                    q_key = re.sub(r"\s+", " ", first_line).lower()
                    if q_key:
                        aok_by_q.setdefault(q_key, (pr, ans))

            api = HfApi(token=hf_token)
            files = [f for f in api.list_repo_files("HuggingFaceM4/A-OKVQA", repo_type="dataset") if "train" in f and f.endswith(".parquet")]
            added_aok = 0
            for pf in sorted(files):
                if added_aok >= max_aokvqa:
                    break
                lp = Path(hf_hub_download("HuggingFaceM4/A-OKVQA", pf, repo_type="dataset", token=hf_token))
                tbl = pq.read_table(lp, columns=["image", "question_id", "question"])
                for r_i in range(tbl.num_rows):
                    if added_aok >= max_aokvqa:
                        break
                    q_txt = re.sub(r"\s+", " ", str(tbl["question"][r_i].as_py() or "").strip()).lower()
                    hit = aok_by_q.pop(q_txt, None)
                    if hit is None:
                        continue
                    qid = str(tbl["question_id"][r_i].as_py())
                    rid = f"nemotron_v3_aok_{qid}"
                    if rid in seen_ids:
                        continue
                    raw_img = tbl["image"][r_i].as_py()
                    raw_bytes = raw_img.get("bytes") if isinstance(raw_img, dict) else None
                    if not raw_bytes:
                        continue
                    norm = _normalize_image_bytes(raw_bytes, max_side=1024)
                    if norm is None:
                        continue
                    jpg_b, _, _ = norm
                    if not _validate_normalized_image_quality(jpg_b):
                        continue
                    seen_ids.add(rid)
                    pr, ans = hit
                    mem = f"nemotron_v3/aokvqa/aok_{qid}.jpg"
                    row = {
                        "id": rid,
                        "source_group": f"nemotron_v3_aok_{abs(hash(qid)) % 64:02d}",
                        "language": "en",
                        "task": "nemotron_v3_aokvqa_spatial_reasoning",
                        "visual_kind": "independent_image",
                        "enable_thinking": True,
                        "frames": [mem],
                        "media_times_s": [0.0],
                        "prompt": pr,
                        "answer": ans,
                        "target": {"kind": "text", "text": ans.split("</think>")[-1].strip()},
                        "metadata": {"source": "nvidia/Nemotron-Image-Training-v3:aokvqa_2", "license": "CC-BY-4.0"},
                    }
                    pairs.append((row, {mem: jpg_b}))
                    added_aok += 1
            emit("nemotron_v3_aokvqa_loaded", count=added_aok)
        except Exception as exc:
            emit("nemotron_v3_aokvqa_skipped", error=f"{type(exc).__name__}: {exc}")

    # 6. `gqa_1` (+ optional `gqa_2`) from Nemotron-Image-Training-v3 (`CC-BY-4.0`) joined onto up to 16 `lmms-lab/GQA` shards!
    if max_gqa > 0:
        try:
            import pyarrow.parquet as pq
            from huggingface_hub import HfApi

            gqa_by_img: dict[str, tuple[str, str, bool]] = {}
            for gqa_sub in ("gqa_1", "gqa_2"):
                try:
                    gqa_jsonl = Path(
                        hf_hub_download(
                            repo_id=NEMOTRON_V3_REPO,
                            filename=f"{gqa_sub}/{gqa_sub}.jsonl",
                            repo_type="dataset",
                            token=hf_token,
                        )
                    )
                except Exception:
                    continue
                with gqa_jsonl.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        rec = _extract_qa_from_nemotron_row(json.loads(line))
                        if rec is None:
                            continue
                        img_ref, pr, ans = rec
                        stem = Path(img_ref).stem
                        is_spatial = any(
                            kw in pr.lower()
                            for kw in ("left", "right", "behind", "front", "closer", "farther", "above", "below", "next to", "between", "facing", "side")
                        )
                        prev = gqa_by_img.get(stem)
                        if prev is None or (is_spatial and not prev[2]):
                            gqa_by_img[stem] = (pr, ans, is_spatial)

            api = HfApi(token=hf_token)
            gqa_files = [
                f
                for f in api.list_repo_files("lmms-lab/GQA", repo_type="dataset")
                if f.startswith("train_all_images/") and f.endswith(".parquet")
            ]
            added_gqa = 0
            for pf in sorted(gqa_files)[:16]:
                if added_gqa >= max_gqa:
                    break
                lp = Path(hf_hub_download("lmms-lab/GQA", pf, repo_type="dataset", token=hf_token))
                tbl = pq.read_table(lp)
                cols = set(tbl.column_names)
                id_col = "id" if "id" in cols else ("imageId" if "imageId" in cols else None)
                if id_col is None or "image" not in cols:
                    continue
                for r_i in range(tbl.num_rows):
                    if added_gqa >= max_gqa:
                        break
                    img_id = str(tbl[id_col][r_i].as_py())
                    hit = gqa_by_img.pop(img_id, None)
                    if hit is None:
                        continue
                    rid = f"nemotron_v3_gqa_{img_id}"
                    if rid in seen_ids:
                        continue
                    raw_img = tbl["image"][r_i].as_py()
                    raw_bytes = raw_img.get("bytes") if isinstance(raw_img, dict) else None
                    if not raw_bytes:
                        continue
                    norm = _normalize_image_bytes(raw_bytes, max_side=1024)
                    if norm is None:
                        continue
                    jpg_b, _, _ = norm
                    if not _validate_normalized_image_quality(jpg_b):
                        continue
                    seen_ids.add(rid)
                    pr, ans, _ = hit
                    mem = f"nemotron_v3/gqa/gqa_{img_id}.jpg"
                    row = {
                        "id": rid,
                        "source_group": f"nemotron_v3_gqa_{abs(hash(img_id)) % 64:02d}",
                        "language": "en",
                        "task": "nemotron_v3_gqa_spatial_relations",
                        "visual_kind": "independent_image",
                        "enable_thinking": True,
                        "frames": [mem],
                        "media_times_s": [0.0],
                        "prompt": pr,
                        "answer": ans,
                        "target": {"kind": "text", "text": ans.split("</think>")[-1].strip()},
                        "metadata": {"source": "nvidia/Nemotron-Image-Training-v3:gqa_1_2", "license": "CC-BY-4.0/MIT"},
                    }
                    pairs.append((row, {mem: jpg_b}))
                    added_gqa += 1
            emit("nemotron_v3_gqa_loaded", count=added_gqa)
        except Exception as exc:
            emit("nemotron_v3_gqa_skipped", error=f"{type(exc).__name__}: {exc}")

    return pairs
