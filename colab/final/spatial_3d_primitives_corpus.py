"""Stage-8 Expanded 4-Primitive Visual Primitives (`<point>`, `<box>`, `<dash>`, `<circle>`),
Full-Color Shaded 3D Spatial Reasoning (Front/Back + Left/Right + Multi-View Kinematics),
and `nvidia/Nemotron-Image-Training-v3` (`CC-BY-4.0`) Grounded Spatial Corpus Builder.

Core Capabilities Taught:
1. **Expanded 4-Primitive Visual Primitives Grammar (`[0..1000]` normalized image coordinates)**:
   - `<point x=".." y=".." z="near|mid|far" label=".."/>` — projected anatomical/object centers with depth tier.
   - `<box x1=".." y1=".." x2=".." y2=".." label=".."/>` — 2D extent of limbs, sub-assemblies, or scene objects.
   - `<dash x1=".." y1=".." x2=".." y2=".." label=".."/>` — directed skeletal kinematic bone axes (`parent->child`),
     spine/tail flow curves, facing-direction arrows (`facing_left` / `facing_right`), and `front_to_back` depth links.
   - `<circle cx=".." cy=".." r=".." label=".."/>` — rotational joint pivots & articulation sweep radii for 3D rigging
     and animation readiness.
2. **Real Full-Color Shaded 3D Model Understanding (NEVER black-and-white silhouettes/cutouts)**:
   - Renders `CC0-1.0` procedural 3D creatures, vehicles, and props with key+rim studio lighting and exact Z-buffering
     across diverse orbital angles (front-3/4, left/right profile, rear-3/4, elevated/low views).
   - Directly cures `C-RADIOv4-H`'s horizontal-flip left/right orientation artifact via explicit facing-direction `<dash>`
     and left/right limb QA, while harnessing its `DINOv3-7B` + `SAM3` 3D priors for front/back depth ordering,
     occlusion reasoning, and multi-view 3D part correspondence.
3. **`nvidia/Nemotron-Image-Training-v3` (`CC-BY-4.0`) Grounded Spatial & Geometric Reasoning**:
   - Joins verified `Qwen-3.5` + `Gemini-3-Flash` `<think>` traces from `gqa_1`, `gqa_2`, `geometry3k`, `aokvqa_2`
     (and optional `clevr_1`) onto strictly train-only public images (`0%` eval benchmark overlap), normalizing inline
     `[x1, y1, x2, y2]` coordinates into `<box>` + relational `<dash>` primitives.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
from typing import Any

import numpy as np
from PIL import Image

from .common import emit, identity
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
)

_INLINE_BBOX_RE = re.compile(r"\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]")


def _wrap_think(think_text: str, body_text: str) -> str:
    return f"<think>\n{think_text.strip()}\n</think>\n{body_text.strip()}"


def fmt_point(x: float, y: float, label: str, *, z: str | None = None) -> str:
    lbl = _clean_xml_label(label)
    z_attr = f' z="{z}"' if z else ""
    return f'<point x="{_clamp_1000(x)}" y="{_clamp_1000(y)}"{z_attr} label="{lbl}"/>'


def fmt_box(x1: float, y1: float, x2: float, y2: float, label: str) -> str:
    lbl = _clean_xml_label(label)
    a, b = _clamp_1000(min(x1, x2)), _clamp_1000(min(y1, y2))
    c, d = _clamp_1000(max(x1, x2)), _clamp_1000(max(y1, y2))
    return f'<box x1="{a}" y1="{b}" x2="{c}" y2="{d}" label="{lbl}"/>'


def fmt_dash(x1: float, y1: float, x2: float, y2: float, label: str) -> str:
    lbl = _clean_xml_label(label)
    return f'<dash x1="{_clamp_1000(x1)}" y1="{_clamp_1000(y1)}" x2="{_clamp_1000(x2)}" y2="{_clamp_1000(y2)}" label="{lbl}"/>'


def fmt_circle(cx: float, cy: float, r: float, label: str) -> str:
    lbl = _clean_xml_label(label)
    rad = max(8, min(400, int(round(r))))
    return f'<circle cx="{_clamp_1000(cx)}" cy="{_clamp_1000(cy)}" r="{rad}" label="{lbl}"/>'


def render_shaded_3d_with_spatial_annotations(
    program: Any,
    baked: dict[str, Any],
    *,
    az_deg: float = 36.0,
    el_deg: float = 18.0,
    fill_frac: float = 0.78,
    pan_uv: tuple[float, float] = (0.0, 0.0),
    panel_size: tuple[int, int] = (384, 384),
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

    sc = compile_geometry(program)
    bone_parent_map: dict[str, str | None] = {}
    bone_pivot_world: dict[str, np.ndarray] = {}
    for wb in (getattr(sc, "bones", None) or ()):
        bname = str(wb.name)
        bone_parent_map[bname] = str(wb.parent_name) if getattr(wb, "parent_name", None) else None
        if getattr(wb, "pivot_world", None) is not None:
            bone_pivot_world[bname] = np.asarray(wb.pivot_world, dtype=np.float64)

    quads: list[tuple[np.ndarray, np.ndarray, bool]] = []
    bone_verts: dict[str, list[np.ndarray]] = {}
    for rec in baked["elements"]:
        ctr = rec["world_center"]
        rot = rec["world_rotation"]
        half_ext = 0.5 * rec["eff_size"]
        is_pl = bool(rec["is_plane"])
        p_norm = rec["plane_normal"]
        local_faces = _face_vertices_local(half_ext)
        bname = str(rec.get("bone", rec.get("bone_name", "root")))

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
        bone_verts.setdefault(bname, []).append(np.concatenate(el_verts, axis=0))

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

    all_pts = np.concatenate([q[0] for q in quads], axis=0)
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

    def _project(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rel = pts - cam_pos[None, :]
        cx = rel @ right
        cy = rel @ up
        cz = rel @ fwd
        cz_safe = np.maximum(cz, 0.1)
        sx = (0.5 + 0.5 * (cx / (cz_safe * half_tan_x))) * (width - 1)
        sy = (0.5 - 0.5 * (cy / (cz_safe * half_tan_y))) * (height - 1)
        return sx, sy, cz

    # Rich full-color gradient studio backdrop (subtle floor contact shadow feel, NEVER flat B&W cutout!)
    yy_bg, xx_bg = np.mgrid[0:height, 0:width]
    ry_n = (yy_bg / max(1.0, height - 1.0)) - 0.45
    rx_n = (xx_bg / max(1.0, width - 1.0)) - 0.50
    rad = np.sqrt(rx_n * rx_n + ry_n * ry_n)
    bg = np.clip(1.0 - 0.50 * rad, 0.35, 1.0)
    floor_grad = np.clip((yy_bg / max(1.0, height - 1.0) - 0.62) / 0.38, 0.0, 1.0)
    color_srgb = np.stack(
        [
            20.0 + 18.0 * bg - 6.0 * floor_grad,
            25.0 + 22.0 * bg - 5.0 * floor_grad,
            36.0 + 28.0 * bg - 4.0 * floor_grad,
        ],
        axis=-1,
    )
    zbuf = np.full((height, width), 1e9, dtype=np.float64)
    key_light = np.array([0.46, 0.78, -0.42], dtype=np.float64)
    key_light /= np.linalg.norm(key_light)
    rim_light = np.array([-0.55, 0.48, 0.68], dtype=np.float64)
    rim_light /= np.linalg.norm(rim_light)

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
        shade = 0.55 + 0.42 * diff + 0.15 * rim

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
            lit = rgba[..., :3].astype(np.float64) * (shade * (1.0 - em[..., None]) + 1.08 * em[..., None])
            sub_c = color_srgb[min_y : max_y + 1, min_x : max_x + 1]
            sub_c[vis] = np.clip(lit[vis], 0.0, 255.0)

    img = Image.fromarray(np.clip(color_srgb, 0, 255).astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    # Project every bone into exact [0..1000] screen coordinates + camera-space depth `cz`!
    bones_info: dict[str, dict[str, Any]] = {}
    all_cz_means: list[float] = []
    for bname, vlist in bone_verts.items():
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
        # Joint pivot projection
        if bname in bone_pivot_world:
            psx, psy, pcz = _project(bone_pivot_world[bname][None, :])
            px_1000 = float(psx[0] * (1000.0 / max(1.0, float(width - 1))))
            py_1000 = float(psy[0] * (1000.0 / max(1.0, float(height - 1))))
            pcz_val = float(pcz[0])
        else:
            px_1000, py_1000, pcz_val = cx_1000, cy_1000, cz_mean
        radius = max(14.0, min(140.0, 0.35 * math.hypot(bx2 - bx1, by2 - by1)))
        bones_info[bname] = {
            "name": bname,
            "parent": bone_parent_map.get(bname),
            "cx": cx_1000,
            "cy": cy_1000,
            "box": [bx1, by1, bx2, by2],
            "pivot_xy": [px_1000, py_1000],
            "pivot_cz": pcz_val,
            "cz_mean": cz_mean,
            "cz_min": cz_min,
            "radius": radius,
        }
        all_cz_means.append(cz_mean)

    # Assign exact near/mid/far depth tiers from camera-space `cz`!
    if all_cz_means:
        z_lo, z_hi = min(all_cz_means), max(all_cz_means)
        z_span = max(1e-4, z_hi - z_lo)
        for info in bones_info.values():
            rel_z = (info["cz_mean"] - z_lo) / z_span
            info["depth_tier"] = "near" if rel_z <= 0.33 else ("far" if rel_z >= 0.67 else "mid")
            info["depth_rel"] = float(rel_z)

    # Compute exact canonical facing vector (model local -Z is forward in SyncCraft!)
    # Project target and target + (0, 0, -10) to measure exact 2D screen facing direction!
    f_pts = np.stack([target, target + np.array([0.0, 0.0, -8.0], dtype=np.float64)], axis=0)
    fsx, fsy, fcz = _project(f_pts)
    fx0, fy0 = float(fsx[0] * 1000.0 / max(1.0, width - 1)), float(fsy[0] * 1000.0 / max(1.0, height - 1))
    fx1, fy1 = float(fsx[1] * 1000.0 / max(1.0, width - 1)), float(fsy[1] * 1000.0 / max(1.0, height - 1))
    dx_face = fx1 - fx0
    dz_face = float(fcz[1] - fcz[0])  # negative = toward camera (facing camera), positive = away!
    if abs(dx_face) >= 28.0:
        lr_facing = "right" if dx_face > 0.0 else "left"
    else:
        lr_facing = "center"
    fb_facing = "toward_camera" if dz_face < -1.5 else ("away_from_camera" if dz_face > 1.5 else "profile")

    # Build kinematic parent -> child `<dash>` edges
    kinematic_edges: list[dict[str, Any]] = []
    for bname, info in bones_info.items():
        par = info.get("parent")
        if par and par in bones_info:
            pinfo = bones_info[par]
            dist = math.hypot(info["cx"] - pinfo["cx"], info["cy"] - pinfo["cy"])
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
        "az_deg": float(az_deg),
        "el_deg": float(el_deg),
        "camera_right": [float(v) for v in right.tolist()],
        "camera_up": [float(v) for v in up.tolist()],
        "bones": bones_info,
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
        },
    }


def _collect_clean_3d_assets(seed: int = 808) -> list[dict[str, Any]]:
    """Compile a diverse library of CC0-1.0 original 3D models (`v25` masterworks + sculpted dragons/creatures/props)."""
    ensure_synccraft_importable()
    from synccraft.dsl.parser import parse_program
    from .aesthetic_v25_vigorous_corpus import build_all_v25_verified_masterworks

    assets: list[dict[str, Any]] = []
    # 1. All 4 v25 verified masterworks (with variants)
    for item in build_all_v25_verified_masterworks():
        try:
            prog = parse_program(item["dsl"])
            baked = bake_sculpted_rgba_atlas(prog, texels_per_unit=2)
            assets.append(
                {
                    "asset_id": item["asset_id"],
                    "family_id": item["family_id"],
                    "title": item["asset_id"].replace("_", " "),
                    "program": prog,
                    "baked": baked,
                    "is_creature": item.get("domain") in ("D2", "D3", "D4"),
                }
            )
        except Exception:
            continue

    # 2. Additional sculpted CC0-1.0 creatures, robots, vehicles & props for broad 3D structural diversity
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
                asset_id, dsl, meta = fn(variant=var)
                prog = parse_program(dsl)
                baked = bake_sculpted_rgba_atlas(prog, texels_per_unit=2)
                assets.append(
                    {
                        "asset_id": f"{asset_id}_v{var}",
                        "family_id": f"spatial3d_{name}",
                        "title": name.replace("_", " "),
                        "program": prog,
                        "baked": baked,
                        "is_creature": is_creature,
                    }
                )
            except Exception:
                continue
    return assets


def build_shaded_3d_spatial_primitives_records(
    *,
    seed: int = 20268,
    panel_size: tuple[int, int] = (384, 384),
    angles_per_asset: int = 6,
) -> list[tuple[dict[str, Any], dict[str, bytes]]]:
    """Generate full-color shaded 3D Visual Primitives (`<point>`, `<box>`, `<dash>`, `<circle>`) +
    Front/Back depth ordering + Left/Right facing orientation + Multi-View 3D correspondence records.
    """
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

    for asset in assets:
        aid = asset["asset_id"]
        fam = asset["family_id"]
        title = asset["title"]
        prog = asset["program"]
        baked = asset["baked"]

        chosen_angles = canonical_angles[: max(4, min(len(canonical_angles), int(angles_per_asset)))]
        rendered_views: list[tuple[str, dict[str, Any]]] = []
        for view_tag, base_az, base_el in chosen_angles:
            az = base_az + rng.uniform(-7.0, 7.0)
            el = max(4.0, min(68.0, base_el + rng.uniform(-4.0, 4.0)))
            pan = (rng.uniform(-0.08, 0.08), rng.uniform(-0.06, 0.06))
            ren = render_shaded_3d_with_spatial_annotations(
                prog, baked, az_deg=az, el_deg=el, pan_uv=pan, panel_size=panel_size
            )
            rendered_views.append((view_tag, ren))

        # Task 1, 2, 3, 5 on individual shaded 3D views
        for v_idx, (view_tag, ren) in enumerate(rendered_views):
            bones = ren["bones"]
            edges = ren["kinematic_edges"]
            facing = ren["facing"]
            if len(bones) < 3:
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
            sorted_by_depth = sorted(bones.values(), key=lambda b: b["cz_mean"])
            sorted_by_x = sorted(bones.values(), key=lambda b: b["cx"])

            # --- Task A: Kinematic <dash> + <circle> Joint Pivots + <box> + <point> ---
            if edges:
                top_edges = sorted(edges, key=lambda e: e["dist"], reverse=True)[:8]
                pivot_bones = list({e["parent"] for e in top_edges[:5]} | {e["child"] for e in top_edges[:5]})[:6]
                think_lines = [
                    f"Inspecting the shaded 3D {title} from {view_tag} (az={ren['az_deg']:.0f}deg, el={ren['el_deg']:.0f}deg).",
                    "1. Mark rotational joint pivots with <circle> and bounding extents with <box>:",
                ]
                for bn in pivot_bones:
                    bi = bones[bn]
                    think_lines.append(
                        f"   - {bn}: {fmt_circle(bi['pivot_xy'][0], bi['pivot_xy'][1], bi['radius'], f'pivot:{bn}')} "
                        f"{fmt_box(*bi['box'], bn)} {fmt_point(bi['cx'], bi['cy'], bn, z=bi['depth_tier'])}"
                    )
                think_lines.append("2. Trace directed parent-to-child skeletal <dash> bone axes:")
                chain_summary = []
                for e in top_edges[:6]:
                    z_rel = "toward_back" if e["dz"] > 0.6 else ("toward_front" if e["dz"] < -0.6 else "coplanar")
                    edge_lbl = f"{e['parent']}->{e['child']}"
                    think_lines.append(
                        f"   - {fmt_dash(e['x1'], e['y1'], e['x2'], e['y2'], edge_lbl)} ({z_rel})"
                    )
                    chain_summary.append(f"{e['parent']} -> {e['child']} ({z_rel})")
                ans_body = (
                    f"Primary 3D kinematic chains on the shaded {title}:\n"
                    + "\n".join(f"- {c}" for c in chain_summary)
                    + f"\nKey articulation pivots: {', '.join(pivot_bones[:5])}."
                )
                row_kin = {
                    "id": f"spatial3d_kin_{aid}_{view_tag}",
                    "source_group": f"{fam}_kin",
                    "language": "en",
                    "task": "spatial_3d_kinematic_dash_and_joints",
                    "visual_kind": "independent_image",
                    "enable_thinking": True,
                    "frames": [member],
                    "media_times_s": [0.0],
                    "visual_units": [base_u],
                    "prompt": (
                        "Analyze the 3D skeletal structure directly from this shaded studio render. "
                        "Use <circle> for joint pivots, <box> for sub-assemblies, <point> with depth tier (near/mid/far), "
                        "and <dash> to trace parent-to-child bone relationships."
                    ),
                    "answer": _wrap_think("\n".join(think_lines), ans_body),
                    "target": {"kind": "text", "text": ans_body},
                    "metadata": {
                        "provenance": make_clean_room_provenance(aid, title=f"spatial3d_kinematic_dash:{aid}", license_spdx="CC0-1.0"),
                        "primitives": ["point", "box", "dash", "circle"],
                    },
                }
                pairs.append((row_kin, {member: ren["png_bytes"]}))

            # --- Task B: Explicit Front / Back Spatial Depth & Occlusion Ordering ---
            if len(sorted_by_depth) >= 4:
                front_b = sorted_by_depth[0]
                mid_b = sorted_by_depth[len(sorted_by_depth) // 2]
                back_b = sorted_by_depth[-1]
                depth_gap = back_b["cz_mean"] - front_b["cz_mean"]
                if depth_gap >= 2.0:
                    swap_order = (v_idx % 2 == 1)
                    opt_a, opt_b = (back_b, front_b) if swap_order else (front_b, back_b)
                    closer = front_b["name"]
                    farther = back_b["name"]
                    mid_name = mid_b["name"]
                    think_fb = "\n".join(
                        [
                            f"Comparing camera-space front-to-back depth on the shaded 3D {title}:",
                            f"- Nearest foreground part: {fmt_point(front_b['cx'], front_b['cy'], front_b['name'], z='near')} {fmt_box(*front_b['box'], front_b['name'])}",
                            f"- Mid-depth core: {fmt_point(mid_b['cx'], mid_b['cy'], mid_name, z='mid')} {fmt_circle(mid_b['pivot_xy'][0], mid_b['pivot_xy'][1], mid_b['radius'], f'pivot:{mid_name}')}",
                            f"- Farthest background part: {fmt_point(back_b['cx'], back_b['cy'], back_b['name'], z='far')} {fmt_box(*back_b['box'], back_b['name'])}",
                            f"- Front-to-back depth ray: {fmt_dash(front_b['cx'], front_b['cy'], back_b['cx'], back_b['cy'], f'front_to_back:{closer}->{farther}')}",
                            f"Therefore, `{closer}` is in front (closer to camera) and `{farther}` is farther back.",
                        ]
                    )
                    ans_fb = (
                        f"`{closer}` is in front (closer to the camera), `{mid_b['name']}` lies at mid-depth, "
                        f"and `{farther}` is farther in the back (front-to-back order: `{closer}` -> `{mid_b['name']}` -> `{farther}`)."
                    )
                    row_fb = {
                        "id": f"spatial3d_fb_{aid}_{view_tag}",
                        "source_group": f"{fam}_fb",
                        "language": "en",
                        "task": "spatial_3d_front_back_depth_order",
                        "visual_kind": "independent_image",
                        "enable_thinking": True,
                        "frames": [member],
                        "media_times_s": [0.0],
                        "visual_units": [base_u],
                        "prompt": (
                            f"Looking directly at this shaded 3D render, determine the front-to-back spatial relationship "
                            f"between `{opt_a['name']}` and `{opt_b['name']}` (and the core `{mid_b['name']}`). "
                            f"Which part is closer to the camera (in front) and which is farther back?"
                        ),
                        "answer": _wrap_think(think_fb, ans_fb),
                        "target": {"kind": "text", "text": ans_fb},
                        "metadata": {
                            "provenance": make_clean_room_provenance(aid, title=f"spatial3d_front_back:{aid}", license_spdx="CC0-1.0"),
                            "front_bone": closer,
                            "back_bone": farther,
                        },
                    }
                    pairs.append((row_fb, {member: ren["png_bytes"]}))

            # --- Task C: Explicit Left / Right Facing & Orientation QA (cures C-RADIOv4 horizontal-flip artifact!) ---
            if v_idx < 4 and len(sorted_by_x) >= 3:
                left_b = sorted_by_x[0]
                right_b = sorted_by_x[-1]
                lr_dir = facing["lr"]
                fb_dir = facing["fb"]
                if lr_dir in ("left", "right"):
                    facing_phrase = (
                        f"facing {lr_dir} ({fb_dir.replace('_', ' ')})"
                        if fb_dir != "profile"
                        else f"facing {lr_dir} in side profile"
                    )
                else:
                    facing_phrase = f"facing {fb_dir.replace('_', ' ')} (centered horizontally)"
                lr_lbl = f"left_to_right:{left_b['name']}->{right_b['name']}"
                think_lr = "\n".join(
                    [
                        f"Determining exact left/right orientation and facing vector on the shaded 3D {title}:",
                        f"- Leftmost visible assembly: {fmt_point(left_b['cx'], left_b['cy'], left_b['name'], z=left_b['depth_tier'])} {fmt_box(*left_b['box'], left_b['name'])}",
                        f"- Rightmost visible assembly: {fmt_point(right_b['cx'], right_b['cy'], right_b['name'], z=right_b['depth_tier'])} {fmt_box(*right_b['box'], right_b['name'])}",
                        f"- Left-to-right span axis: {fmt_dash(left_b['cx'], left_b['cy'], right_b['cx'], right_b['cy'], lr_lbl)}",
                        f"- 3D forward facing arrow: {fmt_dash(facing['x1'], facing['y1'], facing['x2'], facing['y2'], f'facing_{lr_dir}_{fb_dir}')}",
                        f"Screen delta dx={facing['dx']:.1f} confirms the model is {facing_phrase}, with `{left_b['name']}` on the left and `{right_b['name']}` on the right.",
                    ]
                )
                ans_lr = (
                    f"The 3D model is {facing_phrase}. On screen, `{left_b['name']}` sits on the left side "
                    f"and `{right_b['name']}` sits on the right side."
                )
                row_lr = {
                    "id": f"spatial3d_lr_{aid}_{view_tag}",
                    "source_group": f"{fam}_lr",
                    "language": "en",
                    "task": "spatial_3d_left_right_orientation_facing",
                    "visual_kind": "independent_image",
                    "enable_thinking": True,
                    "frames": [member],
                    "media_times_s": [0.0],
                    "visual_units": [base_u],
                    "prompt": (
                        "From this shaded 3D render, determine which direction the 3D model is facing "
                        "(left, right, toward camera, or away from camera) and identify which sub-assembly sits on the "
                        "left side versus the right side of the frame. Trace the facing vector with <dash>."
                    ),
                    "answer": _wrap_think(think_lr, ans_lr),
                    "target": {"kind": "text", "text": ans_lr},
                    "metadata": {
                        "provenance": make_clean_room_provenance(aid, title=f"spatial3d_left_right_orientation:{aid}", license_spdx="CC0-1.0"),
                        "facing_lr": lr_dir,
                        "facing_fb": fb_dir,
                    },
                }
                pairs.append((row_lr, {member: ren["png_bytes"]}))

        # --- Task D: Multi-View 3D Cross-Angle Correspondence & Front/Back Flip ---
        if len(rendered_views) >= 4:
            v_indices = [0, 1, 2, 4] if len(rendered_views) >= 5 else [0, 1, 2, 3]  # front-right, front-left, right profile, rear/left
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

            # Pick a bone that shifts clearly between near/front and far/back across the 4 angles!
            common_bones = set(mv_rens[0][1]["bones"].keys())
            for _, vr in mv_rens[1:]:
                common_bones &= set(vr["bones"].keys())
            best_b = None
            best_swing = -1.0
            for bn in sorted(common_bones):
                rels = [vr["bones"][bn]["depth_rel"] for _, vr in mv_rens]
                swing = max(rels) - min(rels)
                if swing > best_swing:
                    best_swing = swing
                    best_b = bn
            if best_b is not None:
                think_mv = [f"Tracking `{best_b}` across all 4 shaded 3D views of {title}:"]
                summary_mv = []
                for idx_v, (vtag, vr) in enumerate(mv_rens, start=1):
                    bi = vr["bones"][best_b]
                    side_lr = "left" if bi["cx"] < 440 else ("right" if bi["cx"] > 560 else "center")
                    f_lr = vr["facing"]["lr"]
                    think_mv.append(
                        f"- View {idx_v} ({vtag}, facing {f_lr}): "
                        f"{fmt_point(bi['cx'], bi['cy'], best_b, z=bi['depth_tier'])} "
                        f"{fmt_circle(bi['pivot_xy'][0], bi['pivot_xy'][1], bi['radius'], f'pivot:{best_b}')} "
                        f"{fmt_dash(vr['facing']['x1'], vr['facing']['y1'], vr['facing']['x2'], vr['facing']['y2'], f'v{idx_v}_facing_{f_lr}')} "
                        f"(screen {side_lr}, depth={bi['depth_tier']})"
                    )
                    summary_mv.append(f"View {idx_v} ({vtag}): {side_lr} side, depth={bi['depth_tier']}")
                ans_mv = (
                    f"Cross-view 3D correspondence for `{best_b}`:\n"
                    + "\n".join(f"- {s}" for s in summary_mv)
                    + "\nIts front/back depth and left/right screen placement flip consistently with the 3D camera orbit."
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
                        f"and describe how its left/right screen position and front/back depth change as the camera orbits."
                    ),
                    "answer": _wrap_think("\n".join(think_mv), ans_mv),
                    "target": {"kind": "text", "text": ans_mv},
                    "metadata": {
                        "provenance": make_clean_room_provenance(aid, title=f"spatial3d_multiview_correspondence:{aid}", license_spdx="CC0-1.0"),
                        "tracked_bone": best_b,
                    },
                }
                pairs.append((row_mv, mv_blobs))

    emit("spatial_3d_primitives_built", records=len(pairs), assets=len(assets))
    return pairs


def _convert_nemotron_boxes_to_primitives(text: str) -> str:
    """Upgrade Nemotron inline `[x1, y1, x2, y2]` coordinates inside `<think>` into `<box>` + relational `<dash>` tags."""
    if not text:
        return text
    found_centers: list[tuple[float, float, str]] = []

    def _repl(m: re.Match[str]) -> str:
        x1, y1, x2, y2 = (int(m.group(i)) for i in (1, 2, 3, 4))
        if max(x1, y1, x2, y2) > 1000:
            return m.group(0)
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        tag = f"obj_{len(found_centers) + 1}"
        found_centers.append((cx, cy, tag))
        return fmt_box(x1, y1, x2, y2, tag)

    converted = _INLINE_BBOX_RE.sub(_repl, text)
    # When >= 2 objects are grounded and a spatial relationship is discussed, append a relational <dash>!
    low = converted.lower()
    if len(found_centers) >= 2 and any(
        kw in low for kw in ("left", "right", "behind", "in front", "closer", "above", "below", "between", "next to")
    ):
        (cx1, cy1, l1), (cx2, cy2, l2) = found_centers[0], found_centers[1]
        rel_dash = fmt_dash(cx1, cy1, cx2, cy2, f"rel:{l1}->{l2}")
        if "</think>" in converted:
            converted = converted.replace("</think>", f"\nSpatial relation trace: {rel_dash}\n</think>", 1)
    return converted


def _has_banned_benchmark_name(text: str) -> bool:
    low = str(text or "").lower()
    return any(b in low for b in BANNED_BENCHMARK_SUBSTRINGS)


def build_nemotron_v3_spatial_records(
    *,
    cache_dir: str | Path,
    hf_token: str | None = None,
    max_gqa: int = 1200,
    max_geometry3k: int = 900,
    max_aokvqa: int = 900,
    seed: int = 20268,
) -> list[tuple[dict[str, Any], dict[str, bytes]]]:
    """Build train-only grounded spatial & geometric reasoning records from `nvidia/Nemotron-Image-Training-v3` (`CC-BY-4.0`)
    joined onto public HF train images (`lmms-lab/GQA` MIT, `hiyouga/geometry3k` MIT, `HuggingFaceM4/A-OKVQA`).
    """
    from huggingface_hub import hf_hub_download

    rng = random.Random(seed)
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[dict[str, Any], dict[str, bytes]]] = []

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
        if len(prompt) > 2000 or len(answer) > 3600:
            return None
        return img_ref, prompt, _convert_nemotron_boxes_to_primitives(answer)

    # 1. `geometry3k` from Nemotron-Image-Training-v3 paired with `hiyouga/geometry3k` (`train` split ONLY!)
    if max_geometry3k > 0:
        try:
            import pyarrow.parquet as pq

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

            # Load hiyouga/geometry3k train parquet (zero test/val overlap!)
            from huggingface_hub import HfApi

            api = HfApi(token=hf_token)
            files = [f for f in api.list_repo_files("hiyouga/geometry3k", repo_type="dataset") if "train" in f and f.endswith(".parquet")]
            added_geo = 0
            for pf in sorted(files):
                lp = Path(hf_hub_download("hiyouga/geometry3k", pf, repo_type="dataset", token=hf_token))
                tbl = pq.read_table(lp)
                cols = set(tbl.column_names)
                img_col = "images" if "images" in cols else ("image" if "image" in cols else None)
                if img_col is None:
                    continue
                n_rows = tbl.num_rows
                idxs = list(range(n_rows))
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
                    # Prefer Nemotron Qwen3-VL-235B CoT if matched by problem id, else clean train geometry QA
                    pid = str(tbl["problem_id"][r_i].as_py()) if "problem_id" in cols else ""
                    if pid in geo_by_id:
                        pr, ans = geo_by_id[pid]
                        src_tag = "nemotron_v3_geometry3k"
                    else:
                        pr = str(tbl["problem"][r_i].as_py() if "problem" in cols else "").replace("<image>", "").strip()
                        gt = str(tbl["answer"][r_i].as_py() if "answer" in cols else "").strip()
                        if not pr or not gt:
                            continue
                        ans = _wrap_think(
                            f"Analyzing the geometric figure carefully to solve for the requested quantity: {gt}.",
                            gt,
                        )
                        src_tag = "hiyouga_geometry3k_train"
                    mem = f"nemotron_v3/geometry3k/geo_{added_geo:05d}.jpg"
                    row = {
                        "id": f"nemotron_v3_geo_{added_geo:05d}",
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

    # 2. `aokvqa_2` from Nemotron-Image-Training-v3 (`CC-BY-4.0`) joined onto `HuggingFaceM4/A-OKVQA` (`train` ONLY!)
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
                    raw_img = tbl["image"][r_i].as_py()
                    raw_bytes = raw_img.get("bytes") if isinstance(raw_img, dict) else None
                    if not raw_bytes:
                        continue
                    norm = _normalize_image_bytes(raw_bytes, max_side=1024)
                    if norm is None:
                        continue
                    jpg_b, _, _ = norm
                    pr, ans = hit
                    mem = f"nemotron_v3/aokvqa/aok_{qid}.jpg"
                    row = {
                        "id": f"nemotron_v3_aok_{qid}",
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

    # 3. `gqa_1` from Nemotron-Image-Training-v3 (`CC-BY-4.0` Qwen3.5 + Gemini-3-Flash spatial relations!)
    #    joined onto `lmms-lab/GQA` (`MIT`) image parquet shards!
    if max_gqa > 0:
        try:
            import pyarrow.parquet as pq
            from huggingface_hub import HfApi

            gqa_jsonl = Path(
                hf_hub_download(
                    repo_id=NEMOTRON_V3_REPO,
                    filename="gqa_1/gqa_1.jsonl",
                    repo_type="dataset",
                    token=hf_token,
                )
            )
            gqa_by_img: dict[str, tuple[str, str]] = {}
            with gqa_jsonl.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    rec = _extract_qa_from_nemotron_row(json.loads(line))
                    if rec is None:
                        continue
                    img_ref, pr, ans = rec
                    stem = Path(img_ref).stem
                    # Prioritize explicit left/right/front/behind spatial relation queries!
                    is_spatial = any(
                        kw in pr.lower()
                        for kw in ("left", "right", "behind", "front", "above", "below", "next to", "between", "facing", "side")
                    )
                    if stem not in gqa_by_img or is_spatial:
                        gqa_by_img[stem] = (pr, ans)

            api = HfApi(token=hf_token)
            gqa_files = [
                f
                for f in api.list_repo_files("lmms-lab/GQA", repo_type="dataset")
                if f.startswith("train_all_images/") and f.endswith(".parquet")
            ]
            added_gqa = 0
            for pf in sorted(gqa_files)[:6]:
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
                    raw_img = tbl["image"][r_i].as_py()
                    raw_bytes = raw_img.get("bytes") if isinstance(raw_img, dict) else None
                    if not raw_bytes:
                        continue
                    norm = _normalize_image_bytes(raw_bytes, max_side=1024)
                    if norm is None:
                        continue
                    jpg_b, _, _ = norm
                    pr, ans = hit
                    mem = f"nemotron_v3/gqa/gqa_{img_id}.jpg"
                    row = {
                        "id": f"nemotron_v3_gqa_{img_id}",
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
                        "metadata": {"source": "nvidia/Nemotron-Image-Training-v3:gqa_1", "license": "CC-BY-4.0/MIT"},
                    }
                    pairs.append((row, {mem: jpg_b}))
                    added_gqa += 1
            emit("nemotron_v3_gqa_loaded", count=added_gqa)
        except Exception as exc:
            emit("nemotron_v3_gqa_skipped", error=f"{type(exc).__name__}: {exc}")

    return pairs
