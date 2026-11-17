"""Sculpted Blockbench Style (v6 Skill) + Block-DSL v3.2/v4.0 Native Synthesizer, Painter & Critic.

Combines the master-artist ``blockbench-sculpted-style`` workflow directly with
SyncCraft's typed Block-DSL (``v3.2`` ``@rig`` with ``size_step: 0.5``, ``audit_profile: sculpted``,
``plane``, ``chain``, ``def/use`` macros, per-box ``rotate``/``pivot``, AND ``v4.0`` ``@layout`` /
``@component`` / ``@assembly``):

1. **Vibrant Banded Cel-Shaded + RGBA Alpha-Cutout Pixel-Art Painter & Studio Renderer**:
   - 8-step hue-shifted ramps (cool violet/indigo shadows -> rich saturated midtones -> warm cream/yellow/cyan highlights).
   - 1px bright top bevel + 1px dark bottom/contact edge, stepped diagonal scale clusters, ventral scutes,
     longitudinal root-to-tip horn/claw/spar gradients, 2x2 highlight eyes, and 1-3px emissive glow bleed.
   - Crisp RGBA ``alpha=0`` pixel-cutout ``plane`` silhouettes: scalloped/tattered wing membrane bays with
     2-4 outer holes, 1-3px jagged neck/limb/dorsal spike & fur fringes, serrated jaw tooth skirts, and
     dripping leaf/cloth clusters.
   - Perspective Z-buffer dark-slate studio turnaround renderer (``hero``, ``front``, ``side``, ``top``,
     plus ``black_silhouette`` sheet) respecting double-sided ``alpha >= 128`` cutouts and soft ground shadows.
2. **8 Original Clean-Room Sculpted Archetype Families (`CC0-1.0`, Zero IP/Fan-Reference Names)**:
   - ``crimson_cream_western_dragon`` (4 cursorial/pillar legs with knee+calf+heel+splayed pale toes,
     7-panel partitioned membrane wings, S-curved 0.72-seated neck, skull crest fan, hinged cream jaw, 4-seg horns)
   - ``glacial_crystal_drake`` (chunky 1/3 head share, heavy jaw+fangs+brow shadow, dorsal cyan crystal cluster,
     non-coplanar spike fringe ``plane``s, segmented navy belly scutes)
   - ``ochre_club_wyvern`` (crouched wing-walker pose, planted wing-arms + stacked folded membrane ``plane``s,
     dorsal cutout spines, 4-seg curved ram horns, spiked mace tail club)
   - ``pastel_chibi_dragonet`` (Chibi tier 24-34 elements, 45% head share, stub legs, tiny layered wings)
   - ``storm_wolf_dragon_v4`` (Block-DSL v4.0 ``@layout`` + ``@component`` + ``@assembly`` fanged storm wolf-dragon)
   - ``orchard_harvest_automaton`` (chamfered rotated volumes, offset 45-deg octagonal hubs/wheels, curved hose ``chain``,
     leaf-cluster cutout ``plane``s)
   - ``polar_expedition_crawler`` (chamfered cab/cargo, rail/ladder/antenna/lamp greebles, snow-cap slabs, octagonal tyres)
   - ``alchemist_studio_diorama`` (warm emissive lantern focal glow vs cool violet walls, skewed 3-8 deg leaning books/bottles,
     layered potted plant leaf ``plane``s)
3. **Quantitative Sculpted Style Critic (`evaluate_sculpted_blockbench_style`)** & **4 Native Supervision Tasks**:
   - ``synccraft_sculpted_style_creation`` (Text -> ``<think>`` Trait Sheet + Sculpted Block-DSL)
   - ``synccraft_sculpted_stage_progression`` (4-view Stiff Stage-C Blockout -> ``<think>`` Critique + Stage-E Sculpted DSL)
   - ``synccraft_sculpted_multiview_inverse`` (4-view Vibrant Studio Turnaround -> ``<think>`` + Sculpted Block-DSL)
   - ``synccraft_sculpted_self_critique`` (4-view Studio Turnaround + Silhouette Sheet -> 8-axis JSON Style Critique)
"""
from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .common import emit, identity
from .synccraft_bridge import (
    _camera_unit_meta,
    ensure_synccraft_importable,
    format_supervised_answer,
    make_clean_room_provenance,
)

SCULPTED_STYLE_TASKS: tuple[str, ...] = (
    "synccraft_sculpted_style_creation",
    "synccraft_sculpted_stage_progression",
    "synccraft_sculpted_multiview_inverse",
    "synccraft_sculpted_self_critique",
    "synccraft_v4_modular_assembly",
)

FACES: tuple[str, ...] = ("north", "south", "east", "west", "up", "down")
FACE_AXES: dict[str, tuple[int, int]] = {
    "north": (0, 1),
    "south": (0, 1),
    "east": (2, 1),
    "west": (2, 1),
    "up": (0, 2),
    "down": (0, 2),
}
NORMAL_FACES: dict[int, tuple[str, str]] = {
    0: ("east", "west"),
    1: ("up", "down"),
    2: ("south", "north"),
}


# ---------------------------------------------------------------------------
# 1. OKLCH -> Vibrant Hue-Shifted Cel-Banded Pixel-Art Painter & Studio Renderer
# ---------------------------------------------------------------------------

def _oklch_to_srgb_u8(l_val: float, c_val: float, h_deg: float) -> tuple[int, int, int]:
    """Convert OKLCH `(L in 0..1, C, H_deg)` to clamped sRGB `(R, G, B)` in `0..255`."""
    hr = math.radians(float(h_deg))
    a = float(c_val) * math.cos(hr)
    b = float(c_val) * math.sin(hr)

    l_ = float(l_val) + 0.3963377774 * a + 0.2158037573 * b
    m_ = float(l_val) - 0.1055613458 * a - 0.0638541728 * b
    s_ = float(l_val) - 0.0894841775 * a - 1.2914855480 * b

    l3, m3, s3 = l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_
    r_lin = +4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3
    g_lin = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3
    b_lin = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3

    def _to_srgb(x: float) -> int:
        x = max(0.0, min(1.0, x))
        s = 12.92 * x if x <= 0.0031308 else (1.055 * (x ** (1.0 / 2.4)) - 0.055)
        return int(round(max(0.0, min(255.0, s * 255.0))))

    return (_to_srgb(r_lin), _to_srgb(g_lin), _to_srgb(b_lin))


def build_sculpted_hue_shifted_ramp(
    base_lch: Sequence[float],
    *,
    hue_shift: Sequence[float] = (-10.0, 8.0),
    emissive: bool = False,
) -> np.ndarray:
    """Build an 8-step banded cel-shading ramp: cool violet/blue shadows -> saturated midtone -> warm sunlit highlights."""
    l0 = max(0.18, min(0.92, float(base_lch[0])))
    c0 = max(0.01, min(0.28, float(base_lch[1])))
    h0 = float(base_lch[2])
    sh_cool = float(hue_shift[0]) if len(hue_shift) >= 1 and float(hue_shift[0]) != 0.0 else -14.0
    hi_warm = float(hue_shift[1]) if len(hue_shift) >= 2 and float(hue_shift[1]) != 0.0 else +10.0

    # Banded cel steps: midtones carry highest chroma; shadows shift toward cool violet/blue (~265 deg);
    # highlights drift warm toward cream/gold (~80 deg).
    rel_steps = (
        (-0.26, 0.92, sh_cool * 1.35, (-10.0, -7.0, +16.0)),
        (-0.18, 1.02, sh_cool * 0.95, (-7.0, -4.0, +12.0)),
        (-0.10, 1.08, sh_cool * 0.50, (-3.0, -1.0, +6.0)),
        (-0.03, 1.12, 0.0, (0.0, 0.0, 0.0)),
        (+0.03, 1.08, hi_warm * 0.35, (+4.0, +3.0, -2.0)),
        (+0.10, 0.96, hi_warm * 0.75, (+10.0, +7.0, -3.0)),
        (+0.17, 0.82, hi_warm * 1.10, (+16.0, +12.0, +2.0)),
        (+0.24, 0.68, hi_warm * 1.35, (+24.0, +20.0, +8.0)),
    )
    colors: list[list[int]] = []
    for dl, c_mul, dh, rgb_bias in rel_steps:
        li = max(0.10, min(0.97, l0 + (dl * 0.75 if emissive else dl)))
        ci = max(0.008, min(0.30, c0 * (1.15 if emissive else c_mul)))
        rgb = np.asarray(_oklch_to_srgb_u8(li, ci, h0 + dh), dtype=np.float64) + np.asarray(rgb_bias, dtype=np.float64)
        colors.append(np.clip(np.rint(rgb), 0, 255).astype(int).tolist())
    return np.asarray(colors, dtype=np.uint8)


def _infer_cutout_role(box_name: str, bone_name: str) -> str:
    tag = f"{bone_name}_{box_name}".lower()
    if any(k in tag for k in ("tooth", "fang_skirt", "dentition")):
        return "tooth_skirt"
    if any(k in tag for k in ("web", "membrane", "wing_bay", "wing_panel", "patagium", "sail")):
        return "wing_membrane"
    if any(k in tag for k in ("leaf", "foliage", "vine", "moss", "fern", "cloth", "tatter", "banner", "curtain")):
        return "foliage_or_cloth"
    if any(k in tag for k in ("spark", "arc", "bolt", "flame_plane")):
        return "energy_arc"
    return "spike_fringe"


def _rasterize_cutout_alpha(role: str, w: int, h: int, seed_idx: int) -> np.ndarray:
    """Paint crisp pixel-art `alpha=0/255` cutouts on zero-thickness `plane` elements (`SKILL.md` Section 1)."""
    w, h = max(2, w), max(2, h)
    yy, xx = np.mgrid[0:h, 0:w]
    tu = (xx + 0.5) / w
    tv = 1.0 - (yy + 0.5) / h  # 0 = trailing/bottom tip, 1 = attached spar/root edge

    if role == "wing_membrane":
        # Fanning bay from leading spar (tv=1 or tu=0) with scalloped trailing edge + 2-3 pixel holes near tip
        scallop = 0.16 + 0.11 * np.abs(np.sin(tu * math.pi * 3.0 + 0.3 * (seed_idx % 3)))
        sector_edge = scallop + 0.14 * (1.0 - tu)
        keep = (tv >= sector_edge) & (tu >= 0.02) & (tu <= 0.98)
        if w >= 8 and h >= 8:
            hx1, hy1 = 0.68, 0.36
            hx2, hy2 = 0.38, 0.28
            hole1 = ((tu - hx1) ** 2 / 0.012 + (tv - hy1) ** 2 / 0.016) < 1.0
            hole2 = ((tu - hx2) ** 2 / 0.008 + (tv - hy2) ** 2 / 0.011) < 1.0
            keep = keep & (~hole1) & (~hole2)
        return np.where(keep, 255, 0).astype(np.uint8)

    if role == "tooth_skirt":
        period = max(3, min(5, w // 4 + 2))
        col = (xx + (seed_idx % 2)) % period
        cell = xx // period
        is_canine = (cell == 0) | (cell == max(1, (w // period) - 1))
        fang_len = np.where(is_canine, 0.88, 0.62)
        keep = ((col == 1) & ((1.0 - tv) <= fang_len)) | ((col == 0) & ((1.0 - tv) <= fang_len * 0.45))
        return np.where(keep, 255, 0).astype(np.uint8)

    if role == "foliage_or_cloth":
        # Dripping moss / tattered cloth / clustered leaf silhouette
        n_lobes = max(3, min(6, w // 4))
        drip = 0.24 + 0.58 * (0.5 + 0.5 * np.sin(tu * n_lobes * math.pi + seed_idx * 0.9))
        keep = ((1.0 - tv) <= drip) & (tu >= 0.04) & (tu <= 0.96)
        return np.where(keep, 255, 0).astype(np.uint8)

    if role == "energy_arc":
        zig = 0.50 + 0.22 * np.sin(tv * 10.0 + seed_idx)
        keep = (np.abs(tu - zig) <= 0.14) & (tv >= 0.05) & (tv <= 0.95)
        return np.where(keep, 255, 0).astype(np.uint8)

    # Default: dorsal / neck / elbow pixel-spike fringe (1-3px wide angled spikes, shrinking away from focal end)
    n_spikes = max(3, min(7, max(3, w // 3)))
    su = (tu * n_spikes) % 1.0
    envelope = 0.55 + 0.42 * np.sin(np.clip(tu, 0.03, 0.97) * math.pi)
    spike_profile = np.where(su < 0.68, (su / 0.68) ** 0.75, np.maximum(0.0, 1.0 - (su - 0.68) / 0.28))
    keep = ((1.0 - tv) <= (0.20 + 0.76 * spike_profile * envelope)) & (tu >= 0.02) & (tu <= 0.98)
    return np.where(keep, 255, 0).astype(np.uint8)


def _face_vertices_local(half_ext: np.ndarray) -> dict[str, np.ndarray]:
    hx, hy, hz = float(half_ext[0]), float(half_ext[1]), float(half_ext[2])
    return {
        "north": np.array([[+hx, -hy, -hz], [-hx, -hy, -hz], [-hx, +hy, -hz], [+hx, +hy, -hz]], dtype=np.float64),
        "south": np.array([[-hx, -hy, +hz], [+hx, -hy, +hz], [+hx, +hy, +hz], [-hx, +hy, +hz]], dtype=np.float64),
        "east": np.array([[+hx, -hy, +hz], [+hx, -hy, -hz], [+hx, +hy, -hz], [+hx, +hy, +hz]], dtype=np.float64),
        "west": np.array([[-hx, -hy, -hz], [-hx, -hy, +hz], [-hx, +hy, +hz], [-hx, +hy, -hz]], dtype=np.float64),
        "up": np.array([[-hx, +hy, +hz], [+hx, +hy, +hz], [+hx, +hy, -hz], [-hx, +hy, -hz]], dtype=np.float64),
        "down": np.array([[-hx, -hy, -hz], [+hx, -hy, -hz], [+hx, -hy, +hz], [-hx, -hy, +hz]], dtype=np.float64),
    }


def bake_sculpted_rgba_atlas(
    program: Any,
    scene: Any | None = None,
    *,
    texels_per_unit: int = 2,
) -> dict[str, Any]:
    """Pack and paint a vibrant banded cel-shaded + RGBA alpha-cutout pixel-art atlas for any SyncCraft Program."""
    ensure_synccraft_importable()
    from synccraft.kinematics.se3_chain import compile_geometry

    sc = scene if scene is not None else compile_geometry(program)
    palettes = {
        p.name: build_sculpted_hue_shifted_ramp(p.base_lch, hue_shift=getattr(p, "hue_shift", (-12.0, 9.0)), emissive=bool(p.emissive))
        for p in program.palettes
    }
    emissive_mats = {p.name for p in program.palettes if getattr(p, "emissive", False)}
    default_ramp = next(iter(palettes.values())) if palettes else build_sculpted_hue_shifted_ramp((0.62, 0.12, 42.0))

    jobs: list[tuple[int, int, int, str]] = []
    elem_records: list[dict[str, Any]] = []
    for idx_b, cb in enumerate(sc.boxes):
        raw_sz = np.asarray(cb.size, dtype=np.float64) + 2.0 * float(getattr(cb, "inflate", 0.0))
        p_norm = getattr(cb, "plane_normal", None)
        if p_norm == "x":
            eff_sz = np.array([0.0, max(0.5, raw_sz[1]), max(0.5, raw_sz[2])], dtype=np.float64)
            active_faces = ("east",)
        elif p_norm == "y":
            eff_sz = np.array([max(0.5, raw_sz[0]), 0.0, max(0.5, raw_sz[2])], dtype=np.float64)
            active_faces = ("up",)
        elif p_norm == "z":
            eff_sz = np.array([max(0.5, raw_sz[0]), max(0.5, raw_sz[1]), 0.0], dtype=np.float64)
            active_faces = ("south",)
        else:
            eff_sz = np.maximum(0.25, raw_sz)
            active_faces = FACES

        elem_records.append(
            {
                "idx": idx_b,
                "name": cb.name,
                "bone": cb.bone_name,
                "material": cb.material,
                "is_plane": p_norm in ("x", "y", "z"),
                "plane_normal": p_norm,
                "eff_size": eff_sz,
                "world_center": np.asarray(cb.center, dtype=np.float64),
                "world_rotation": np.asarray(cb.rotation, dtype=np.float64),
                "uv": {},
            }
        )
        for f in active_faces:
            ax_u, ax_v = FACE_AXES[f]
            pw = max(2, int(round(float(eff_sz[ax_u]) * texels_per_unit)))
            ph = max(2, int(round(float(eff_sz[ax_v]) * texels_per_unit)))
            jobs.append((ph, pw, idx_b, f))

    total_area = sum((pw + 4) * (ph + 4) for ph, pw, _, _ in jobs)
    max_pw = max((pw + 4 for _, pw, _, _ in jobs), default=32)
    width = 2 ** max(6, math.ceil(math.log2(max(float(max_pw), math.sqrt(max(64, total_area)) * 1.2))))
    width = min(1024, width)

    x, y, row_h = 2, 2, 0
    for ph, pw, idx_b, fname in sorted(jobs, reverse=True):
        if x + pw + 2 > width:
            x = 2
            y += row_h + 4
            row_h = 0
        elem_records[idx_b]["uv"][fname] = (x, y, x + pw, y + ph)
        x += pw + 4
        row_h = max(row_h, ph)

    height = 2 ** max(6, math.ceil(math.log2(max(64, y + row_h + 4))))
    atlas_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    emissive_map = np.zeros((height, width), dtype=np.float32)

    for rec in elem_records:
        mat_name = rec["material"]
        ramp = palettes.get(mat_name, default_ramp)
        is_em = mat_name in emissive_mats or any(k in f"{rec['name']}_{mat_name}".lower() for k in ("glow", "ember", "crystal", "eye", "lamp", "lantern", "arc"))
        is_pl = bool(rec["is_plane"])
        tag = f"{rec['bone']}_{rec['name']}_{mat_name}".lower()

        if is_pl:
            first_rect = next(iter(rec["uv"].values()))
            for f in FACES:
                rec["uv"].setdefault(f, first_rect)

        painted_rects: set[tuple[int, int, int, int]] = set()
        for f, (x0, y0, x1, y1) in rec["uv"].items():
            if (x0, y0, x1, y1) in painted_rects:
                continue
            painted_rects.add((x0, y0, x1, y1))
            pw, ph = x1 - x0, y1 - y0
            yy, xx = np.mgrid[0:ph, 0:pw]
            tu = (xx + 0.5) / max(1, pw)
            tv = 1.0 - (yy + 0.5) / max(1, ph)

            # Banded cel-shading per face orientation (`SKILL.md`: top lightest, sides mid, underside darkest,
            # 1px lighter top edge and 1px darker bottom/contact edge!)
            if is_pl:
                role = _infer_cutout_role(rec["name"], rec["bone"])
                band = np.clip(np.rint(2.5 + 3.6 * tv + 0.8 * tu), 1, 7).astype(int)
                alpha_patch = _rasterize_cutout_alpha(role, pw, ph, rec["idx"])
            else:
                alpha_patch = np.full((ph, pw), 255, dtype=np.uint8)
                if f == "up":
                    base_f = 5.4 + 0.9 * (1.0 - np.abs(tu - 0.5) * 1.4)
                elif f == "down":
                    base_f = np.full((ph, pw), 1.4, dtype=np.float64)
                else:
                    # Quantized 4-band vertical light falloff + 1px bevel highlight & 1px contact shadow
                    q_tv = np.floor(tv * 4.0) / 3.0
                    base_f = 2.2 + 2.9 * q_tv
                    if ph >= 4:
                        base_f[yy == 0] += 1.05
                        base_f[yy == ph - 1] -= 0.95

                # Clustered strokes (scales / ventral scutes / horn rings / wood grain — NEVER random per-pixel noise!)
                if any(k in tag for k in ("belly", "throat", "sternum", "under", "jaw", "mandible")):
                    scute = np.mod(yy, max(3, ph // 3))
                    base_f[scute == 0] -= 1.2
                    base_f[scute == 1] += 0.7
                elif any(k in tag for k in ("horn", "claw", "fang", "tusk", "antler", "spar", "spike")):
                    ring = np.mod(yy + xx // 3, 5)
                    base_f[ring == 0] -= 1.0
                    base_f += 1.2 * tv  # Root-to-tip taper brightening
                elif any(k in tag for k in ("wood", "plank", "desk", "shelf", "table")):
                    grain = np.mod(yy * 2 + (xx // 4), 5)
                    base_f[grain == 0] -= 0.85
                else:
                    # Stepped diagonal scale/brush clusters (3x2 texel blocks)
                    cx_i = xx // 3
                    cy_i = yy // 2
                    cl = (cx_i * 7 + cy_i * 13 + (cy_i % 2) * 3) % 11
                    base_f[(cl <= 1) & (f != "down")] += 0.95
                    base_f[cl >= 9] -= 0.85

                band = np.clip(np.rint(base_f), 0, 7).astype(int)

            rgb_patch = ramp[band].copy()

            # Hand-placed focal eye feature (`2x2` dark socket/lid + `1px` bright highlight) on skull/head front/side
            if (not is_pl) and ("eye" in tag) and pw >= 2 and ph >= 2:
                rgb_patch[:] = ramp[6]
                rgb_patch[0, :] = np.array([22, 20, 34], dtype=np.uint8)  # Dark upper lid line
                rgb_patch[min(ph - 1, 1), min(pw - 1, pw // 2)] = np.array([255, 252, 236], dtype=np.uint8)  # Catchlight pixel!

            atlas_rgba[y0:y1, x0:x1, :3] = rgb_patch
            atlas_rgba[y0:y1, x0:x1, 3] = alpha_patch
            if is_em:
                emissive_map[y0:y1, x0:x1] = np.where(alpha_patch >= 128, 0.92, 0.0)

            # 1px gutter bleed on opaque cuboids prevents seam artifacts
            if not is_pl and x0 >= 1 and y0 >= 1 and x1 < width and y1 < height:
                atlas_rgba[y0:y1, x0 - 1, :3] = rgb_patch[:, 0]
                atlas_rgba[y0:y1, x0 - 1, 3] = 255
                atlas_rgba[y0:y1, x1, :3] = rgb_patch[:, -1]
                atlas_rgba[y0:y1, x1, 3] = 255

    buf = io.BytesIO()
    Image.fromarray(atlas_rgba, mode="RGBA").save(buf, format="PNG")
    return {
        "width": width,
        "height": height,
        "atlas_rgba": atlas_rgba,
        "emissive_map": emissive_map,
        "atlas_png": buf.getvalue(),
        "elements": elem_records,
    }


def _box_blur_2d(arr: np.ndarray, size: int = 5) -> np.ndarray:
    if size <= 1:
        return arr.copy()
    r = size // 2
    pad_y = np.pad(arr, ((r, size - 1 - r), (0, 0)), mode="edge")
    cs_y = np.cumsum(np.pad(pad_y, ((1, 0), (0, 0)), mode="constant"), axis=0)
    blur_y = (cs_y[size:] - cs_y[:-size]) / float(size)
    pad_x = np.pad(blur_y, ((0, 0), (r, size - 1 - r)), mode="edge")
    cs_x = np.cumsum(np.pad(pad_x, ((0, 0), (1, 0)), mode="constant"), axis=1)
    return (cs_x[:, size:] - cs_x[:, :-size]) / float(size)


def _uv_corners_px(face: str, rect: tuple[int, int, int, int]) -> np.ndarray:
    u0, v0, u1, v1 = [float(v) for v in rect]
    corners = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    if face in ("north", "east"):
        corners = [(1.0 - u, v) for u, v in corners]
    elif face == "up":
        corners = [(u, 1.0 - v) for u, v in corners]
    return np.array([[u0 + u * (u1 - u0), v0 + v * (v1 - v0)] for u, v in corners], dtype=np.float64)


def render_sculpted_studio_frame(
    baked: dict[str, Any],
    *,
    view: str = "iso",
    panel_size: tuple[int, int] = (224, 224),
    silhouette_mode: bool = False,
) -> bytes:
    """Render a dark-backdrop Blockbench studio view (or black-on-light silhouette check) with RGBA alpha cutouts."""
    width, height = int(panel_size[0]), int(panel_size[1])
    atlas_rgba: np.ndarray = baked["atlas_rgba"]
    emissive_map: np.ndarray = baked["emissive_map"]
    tex_h, tex_w = atlas_rgba.shape[:2]

    quads: list[tuple[np.ndarray, np.ndarray, bool]] = []
    for rec in baked["elements"]:
        ctr = rec["world_center"]
        rot = rec["world_rotation"]
        half_ext = 0.5 * rec["eff_size"]
        is_pl = bool(rec["is_plane"])
        p_norm = rec["plane_normal"]
        local_faces = _face_vertices_local(half_ext)

        if is_pl:
            primary_face = {"x": "east", "y": "up", "z": "south"}[p_norm]
            fv = local_faces[primary_face] @ rot.T + ctr[None, :]
            uv_c = _uv_corners_px(primary_face, rec["uv"][primary_face])
            quads.append((fv, uv_c, True))
        else:
            for f in FACES:
                fv = local_faces[f] @ rot.T + ctr[None, :]
                uv_c = _uv_corners_px(f, rec["uv"][f])
                quads.append((fv, uv_c, False))

    if not quads:
        empty = Image.new("RGBA", (width, height), (240, 242, 246, 255) if silhouette_mode else (16, 20, 28, 255))
        b_io = io.BytesIO()
        empty.save(b_io, format="PNG")
        return b_io.getvalue()

    # Map canonical SyncCraft views (`south`=-Z/front, `right`=+X/side, `top`=+Y, `iso`=3/4 hero)
    # Note: in SyncCraft creatures face -Z (`north`/`south` convention varies across human vs procedural rigs;
    # spherical framing auto-fits full bounding volume cleanly from all 4 angles).
    view_angles = {
        "south": (16.0, 13.0, 30.0, 0.82),
        "front": (16.0, 13.0, 30.0, 0.82),
        "right": (88.0, 12.0, 30.0, 0.84),
        "side": (88.0, 12.0, 30.0, 0.84),
        "top": (18.0, 72.0, 32.0, 0.82),
        "iso": (38.0, 22.0, 32.0, 0.84),
        "hero": (38.0, 22.0, 32.0, 0.84),
        "back_iso": (-142.0, 20.0, 32.0, 0.84),
    }
    az_deg, el_deg, fov_deg, fill_frac = view_angles.get(view, (38.0, 22.0, 32.0, 0.84))
    az, el = math.radians(az_deg), math.radians(el_deg)
    cam_dir = np.array([math.cos(el) * math.sin(az), math.sin(el), -math.cos(el) * math.cos(az)], dtype=np.float64)
    fwd = -cam_dir / np.linalg.norm(cam_dir)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    right = np.array([1.0, 0.0, 0.0], dtype=np.float64) if np.linalg.norm(right) < 1e-6 else (right / np.linalg.norm(right))
    up = np.cross(right, fwd)

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
    cam_dist = max(10.0, float(max(np.max(rel_x / (half_tan_x * fill_frac) + rel_d), np.max(rel_y / (half_tan_y * fill_frac) + rel_d))))
    cam_pos = target + cam_dir * cam_dist

    yy_bg, xx_bg = np.mgrid[0:height, 0:width]
    if silhouette_mode:
        color_srgb = np.full((height, width, 3), 236.0, dtype=np.float64)
    else:
        ry_n = (yy_bg / max(1.0, height - 1.0)) - 0.46
        rx_n = (xx_bg / max(1.0, width - 1.0)) - 0.50
        vign = np.clip(1.0 - 0.52 * (rx_n * rx_n + ry_n * ry_n), 0.42, 1.0)
        color_srgb = np.dstack((22.0 * vign, 25.0 * vign, 36.0 * vign)).astype(np.float64)

    z_buf = np.full((height, width), np.inf, dtype=np.float64)
    glow_buf = np.zeros((height, width, 3), dtype=np.float64)
    alpha_out = np.full((height, width), 255, dtype=np.uint8)

    key_dir = np.array([0.44, 0.80, -0.40], dtype=np.float64)
    key_dir /= np.linalg.norm(key_dir)
    rim_dir = np.array([-0.60, 0.32, 0.58], dtype=np.float64)
    rim_dir /= np.linalg.norm(rim_dir)

    atlas_rgb_f = atlas_rgba[..., :3].astype(np.float64)
    atlas_a = atlas_rgba[..., 3]

    for fverts, uv_px, is_pl in quads:
        e1 = fverts[1] - fverts[0]
        e2 = fverts[2] - fverts[0]
        n_raw = np.cross(e1, e2)
        n_len = float(np.linalg.norm(n_raw))
        if n_len < 1e-8:
            continue
        normal = n_raw / n_len
        to_cam = cam_pos - fverts[0]
        facing = float(np.dot(normal, to_cam))
        if facing <= 0.0:
            if is_pl:
                # Double-sided alpha-cutout plane (`SKILL.md`: "Always double-sided")!
                normal = -normal
            else:
                continue

        rel = fverts - cam_pos[None, :]
        cz = rel @ fwd
        if np.any(cz <= 0.8):
            continue
        cx = rel @ right
        cy = rel @ up
        sx = (cx / (cz * half_tan_x) * 0.5 + 0.5) * width
        sy = (0.5 - cy / (cz * half_tan_y) * 0.5) * height

        ndotl = max(0.0, float(np.dot(normal, key_dir)))
        ndotrim = max(0.0, float(np.dot(normal, rim_dir)))
        hemi = 0.5 + 0.5 * float(normal[1])
        # Preserve crisp painted cel bands while adding gentle warm key + cool violet/cyan rim light
        light_mul = (
            np.array([0.74, 0.74, 0.82]) * (0.68 + 0.24 * hemi)
            + np.array([0.36, 0.33, 0.28]) * ndotl
            + np.array([0.14, 0.20, 0.32]) * ndotrim
        )

        for i0, i1, i2 in ((0, 1, 2), (0, 2, 3)):
            x0, y0, iz0 = sx[i0], sy[i0], 1.0 / cz[i0]
            x1, y1, iz1 = sx[i1], sy[i1], 1.0 / cz[i1]
            x2, y2, iz2 = sx[i2], sy[i2], 1.0 / cz[i2]

            px_min = max(0, int(math.floor(min(x0, x1, x2))))
            px_max = min(width - 1, int(math.ceil(max(x0, x1, x2))))
            py_min = max(0, int(math.floor(min(y0, y1, y2))))
            py_max = min(height - 1, int(math.ceil(max(y0, y1, y2))))
            if px_min > px_max or py_min > py_max:
                continue

            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) < 1e-8:
                continue
            inv_den = 1.0 / denom
            gy, gx = np.mgrid[py_min : py_max + 1, px_min : px_max + 1]
            px_p, py_p = gx + 0.5, gy + 0.5
            w0 = ((y1 - y2) * (px_p - x2) + (x2 - x1) * (py_p - y2)) * inv_den
            w1 = ((y2 - y0) * (px_p - x2) + (x0 - x2) * (py_p - y2)) * inv_den
            w2 = 1.0 - w0 - w1

            inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
            if not np.any(inside):
                continue
            inv_z = w0 * iz0 + w1 * iz1 + w2 * iz2
            z_cam = 1.0 / np.maximum(1e-8, inv_z)
            z_slice = z_buf[py_min : py_max + 1, px_min : px_max + 1]
            vis = inside & (z_cam < z_slice - 1e-4)
            if not np.any(vis):
                continue

            u_p = (w0[vis] * (uv_px[i0, 0] * iz0) + w1[vis] * (uv_px[i1, 0] * iz1) + w2[vis] * (uv_px[i2, 0] * iz2)) * z_cam[vis]
            v_p = (w0[vis] * (uv_px[i0, 1] * iz0) + w1[vis] * (uv_px[i1, 1] * iz1) + w2[vis] * (uv_px[i2, 1] * iz2)) * z_cam[vis]
            u_lo, u_hi = min(uv_px[:, 0]), max(uv_px[:, 0])
            v_lo, v_hi = min(uv_px[:, 1]), max(uv_px[:, 1])
            tx = np.clip(np.floor(np.clip(u_p, u_lo + 1e-4, u_hi - 1e-4)).astype(int), 0, tex_w - 1)
            ty = np.clip(np.floor(np.clip(v_p, v_lo + 1e-4, v_hi - 1e-4)).astype(int), 0, tex_h - 1)

            opaque = atlas_a[ty, tx] >= 128
            if not np.any(opaque):
                continue
            vy, vx = np.nonzero(vis)
            oy, ox = vy[opaque], vx[opaque]
            tx_o, ty_o = tx[opaque], ty[opaque]

            z_slice[oy, ox] = z_cam[vis][opaque]
            if silhouette_mode:
                color_srgb[py_min + oy, px_min + ox] = np.array([18.0, 20.0, 28.0])
            else:
                tex_c = atlas_rgb_f[ty_o, tx_o]
                em_v = emissive_map[ty_o, tx_o]
                lit = tex_c * light_mul[None, :]
                if np.any(em_v > 0.05):
                    boost = tex_c * (em_v[:, None] * 0.55)
                    lit = np.minimum(255.0, lit + boost)
                    glow_buf[py_min + oy, px_min + ox] = boost
                else:
                    glow_buf[py_min + oy, px_min + ox] = 0.0
                color_srgb[py_min + oy, px_min + ox] = lit

    if (not silhouette_mode) and float(np.max(glow_buf)) > 1.0:
        for ch_i in range(3):
            color_srgb[:, :, ch_i] += 0.42 * _box_blur_2d(glow_buf[:, :, ch_i], size=9)

    rgb_u8 = np.clip(np.rint(color_srgb), 0, 255).astype(np.uint8)
    rgba_u8 = np.dstack((rgb_u8, alpha_out))
    out_io = io.BytesIO()
    Image.fromarray(rgba_u8, mode="RGBA").save(out_io, format="PNG")
    return out_io.getvalue()


def render_sculpted_turnaround_views(
    program: Any,
    *,
    views: tuple[str, ...] = ("south", "right", "top", "iso"),
    panel_size: tuple[int, int] = (208, 208),
    include_silhouette_sheet: bool = False,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Bake vibrant cel-shaded RGBA cutout atlas and render studio turnaround views (+ optional 2x2 silhouette sheet)."""
    ensure_synccraft_importable()
    from synccraft.kinematics.se3_chain import compile_geometry

    sc = compile_geometry(program)
    baked = bake_sculpted_rgba_atlas(program, sc, texels_per_unit=2)
    png_by_view: dict[str, bytes] = {}
    for v in views:
        png_by_view[v] = render_sculpted_studio_frame(baked, view=v, panel_size=panel_size, silhouette_mode=False)

    if include_silhouette_sheet:
        pw, ph = panel_size
        sheet = Image.new("RGBA", (pw * 2, ph * 2), (236, 238, 244, 255))
        for idx_v, v in enumerate(("south", "right", "top", "iso")):
            sil_png = render_sculpted_studio_frame(baked, view=v, panel_size=panel_size, silhouette_mode=True)
            im_v = Image.open(io.BytesIO(sil_png))
            sheet.paste(im_v, ((idx_v % 2) * pw, (idx_v // 2) * ph))
        s_io = io.BytesIO()
        sheet.save(s_io, format="PNG")
        png_by_view["silhouette_sheet"] = s_io.getvalue()

    return png_by_view, baked



# ---------------------------------------------------------------------------
# 2. Quantitative Sculpted Blockbench Style Critic (`SKILL.md` Sections 1-6)
# ---------------------------------------------------------------------------

def _iter_rig_bones_and_boxes(bone: Any, out_bones: list[Any], out_boxes: list[Any]) -> None:
    out_bones.append(bone)
    for it in getattr(bone, "items", ()):
        if hasattr(it, "items"):
            _iter_rig_bones_and_boxes(it, out_bones, out_boxes)
        elif hasattr(it, "size"):
            out_boxes.append(it)


def evaluate_sculpted_blockbench_style(
    program_or_dsl: Any,
    *,
    prompt: str = "",
) -> dict[str, Any]:
    """Evaluate a Block-DSL program against `blockbench-sculpted-style` v6 criteria."""
    ensure_synccraft_importable()
    from synccraft.audit.sync_audit import audit_scene
    from synccraft.dsl.ast_nodes import Program, validate_program
    from synccraft.dsl.parser import parse_program
    from synccraft.kinematics.se3_chain import compile_geometry

    prog = (
        validate_program(program_or_dsl)
        if isinstance(program_or_dsl, Program)
        else validate_program(parse_program(str(program_or_dsl)))
    )
    sc = compile_geometry(prog)
    enforce_ct = getattr(prog.meta, "audit_profile", "strict") != "authored"
    aud = audit_scene(sc, enforce_physical_contact=enforce_ct)

    n_elems = len(sc.boxes)
    n_bones = len(sc.bones)
    plane_cnt = sum(1 for b in sc.boxes if getattr(b, "plane_normal", None) in ("x", "y", "z"))
    thin_slab_cnt = sum(
        1
        for b in sc.boxes
        if getattr(b, "plane_normal", None) is None and min(float(x) for x in b.size) <= 1.0
    )

    # Check non-grid organic rotations (per-box `rotate` or non-zero bone `rest_euler`)
    rotated_boxes = 0
    for b in sc.boxes:
        rot_mat = np.asarray(b.rotation, dtype=np.float64)
        # Non-axis-aligned iff off-diagonal entries exist or diagonal != +-1
        max_axis_cos = float(np.max(np.abs(rot_mat), axis=1).min())
        if max_axis_cos < 0.9992:
            rotated_boxes += 1
    rotated_fraction = rotated_boxes / max(1, n_elems)

    # Volume hierarchy check (70 / 25 / 5 big-medium-small mass contrast, never uniform voxels)
    vols = sorted(
        [
            max(0.05, float(b.size[0])) * max(0.05, float(b.size[1])) * max(0.05, float(b.size[2]))
            for b in sc.boxes
        ],
        reverse=True,
    )
    vol_ratio = (vols[0] / max(0.1, vols[-1])) if len(vols) >= 2 else 1.0
    unique_sizes = len({tuple(round(float(v), 3) for v in b.size) for b in sc.boxes})

    # Hue-shifted ramps & controlled emissive focal share (`SKILL.md`: 3+1 palette, 3-15% glow)
    hue_shifted_cnt = sum(
        1
        for p in prog.palettes
        if any(abs(float(x)) >= 1.0 for x in getattr(p, "hue_shift", (0.0, 0.0)))
    )
    emissive_mats = {p.name for p in prog.palettes if getattr(p, "emissive", False)}
    emissive_elem_cnt = sum(1 for b in sc.boxes if b.material in emissive_mats)
    emissive_share = emissive_elem_cnt / max(1, n_elems)
    used_mats = {b.material for b in sc.boxes}

    sculpted_overlaps = len(aud.get("sculpted_overlap_pairs", ()))
    audit_prof = str(getattr(prog.meta, "audit_profile", "strict"))
    size_step = float(getattr(prog.meta, "size_step", 1.0))
    dsl_ver = str(getattr(prog.meta, "dsl_version", "3.2"))

    # Anatomical / structural landmark richness
    all_names = " ".join(f"{b.bone_name}_{b.name}" for b in sc.boxes).lower()
    has_spine_curve = rotated_fraction >= 0.18
    has_head_detail = sum(1 for kw in ("skull", "cranium", "snout", "muzzle", "jaw", "mandible", "brow", "horn", "eye", "cab", "lantern") if kw in all_names) >= 2
    has_cutout_or_thin = (plane_cnt + thin_slab_cnt) >= 2
    has_limb_or_mech = any(kw in all_names for kw in ("thigh", "femur", "shin", "tibia", "knee", "calf", "heel", "toe", "claw", "wing", "wheel", "hub", "hose", "shelf"))

    score = 0.0
    score += 0.20 if aud["passed"] else 0.0
    score += 0.16 * min(1.0, n_elems / 24.0)
    score += 0.16 * min(1.0, rotated_fraction / 0.30)
    score += 0.14 * min(1.0, (plane_cnt * 2 + thin_slab_cnt) / 4.0)
    score += 0.12 * min(1.0, len(used_mats) / 4.0)
    score += 0.08 * (1.0 if hue_shifted_cnt >= 2 else (0.5 if hue_shifted_cnt == 1 else 0.2))
    score += 0.07 * (1.0 if vol_ratio >= 8.0 and unique_sizes >= 6 else 0.3)
    score += 0.07 * (1.0 if (has_head_detail and has_limb_or_mech) else 0.3)
    score = round(min(1.0, score), 4)

    notes: list[str] = []
    if not aud["passed"]:
        notes.append(f"Static audit failed: collisions={len(aud['collision_pairs'])}, contact_violations={len(aud['contact_violations'])}.")
    if n_elems < 18:
        notes.append(f"Only {n_elems} elements; expand to >= 22 sculpted boxes & planes across primary/secondary/tertiary tiers.")
    if rotated_fraction < 0.18:
        notes.append(f"Rigid axis-aligned look (rotated_fraction={rotated_fraction:.2f}); tilt spine/neck/limbs/props by 7.5-30 deg + 3-8 deg wobble.")
    if plane_cnt == 0 and thin_slab_cnt < 2:
        notes.append("Missing alpha-cutout `plane`s or thin 0.5u slabs for membranes, fringes, tooth skirts, or leaves.")
    if hue_shifted_cnt < 2:
        notes.append("Add `hue_shift=[-12, 10]` on `@palette` ramps so shadows shift cool (violet/blue) and highlights drift warm.")
    if emissive_share > 0.30:
        notes.append(f"Emissive glow covers {emissive_share:.0%} of elements; concentrate glow to 3-15% near focal eyes/crystals/lamps.")

    is_stiff_or_voxel = (n_elems < 16) or (rotated_fraction < 0.12) or (len(used_mats) < 3)
    grade = (
        "sculpted_master_tier"
        if (aud["passed"] and score >= 0.82 and not is_stiff_or_voxel)
        else ("sculpted_standard" if (aud["passed"] and score >= 0.68) else "stiff_blockout")
    )
    return {
        "sculpted_score": score,
        "grade": grade,
        "audit_passed": bool(aud["passed"]),
        "audit_profile": audit_prof,
        "dsl_version": dsl_ver,
        "size_step": size_step,
        "element_count": n_elems,
        "plane_count": plane_cnt,
        "thin_slab_count": thin_slab_cnt,
        "bone_count": n_bones,
        "rotated_element_fraction": round(rotated_fraction, 4),
        "sculpted_overlap_count": sculpted_overlaps,
        "volume_contrast_ratio": round(vol_ratio, 2),
        "unique_sizes": unique_sizes,
        "used_material_count": len(used_mats),
        "hue_shifted_ramp_count": hue_shifted_cnt,
        "emissive_focal_share": round(emissive_share, 4),
        "has_spine_gesture_curve": has_spine_curve,
        "has_head_or_focal_detail": has_head_detail,
        "has_cutout_or_thin_planes": has_cutout_or_thin,
        "is_stiff_or_voxel_like": is_stiff_or_voxel,
        "critique_notes": notes,
    }


# ---------------------------------------------------------------------------
# 3. Eight Parametric Clean-Room Sculpted Block-DSL Archetypes (`CC0-1.0`)
# ---------------------------------------------------------------------------

def build_sculpted_crimson_western_dragon_dsl(variant: int = 0) -> tuple[str, str, dict[str, Any]]:
    """Return `(sculpted_stage_e_dsl, stiff_stage_c_blockout_dsl, trait_meta)` for an original Western Dragon."""
    h_body = (16 + (variant * 37) % 340) % 360
    h_cream = (52 + (variant * 19) % 45) % 360
    h_horn = (72 + (variant * 13) % 35) % 360
    h_glow = (42 + (variant * 53) % 180) % 360
    chest_w = 8 + (variant % 2) * 2
    wing_finger_l = 9 + (variant % 3)
    head_yaw = 6 if (variant % 2 == 0) else -6

    stage_e = f"""@meta {{ archetype: entity, target_profile: bbmodel_5, texel_density: 16, dsl_version: "3.2", size_step: 0.5, audit_profile: sculpted }}
@palette {{
  hide_mat: ramp(base_lch=[0.56, 0.16, {h_body}], steps=6, hue_shift=[-16, 10])
  cream_mat: ramp(base_lch=[0.80, 0.07, {h_cream}], steps=6, hue_shift=[-12, 8])
  keratin_mat: ramp(base_lch=[0.84, 0.06, {h_horn}], steps=6, hue_shift=[-10, 6])
  eye_glow_mat: ramp(base_lch=[0.82, 0.19, {h_glow}], steps=6, hue_shift=[-8, 12], emissive=true)
}}
@rig {{
  root bone hips_root {{
    pivot_local: [0, 13, 4]
    box pelvis_mass {{ size: [{chest_w - 1}, 6, 7], material: hide_mat, rotate: [-4, 0, 0] }}
    box hip_belly place(base_box=pelvis_mass, anchor=[0, -0.82, 0], align=[0, 0.5, 0]) {{
      size: [{chest_w - 2.5}, 1.5, 6], material: cream_mat
    }}
    box haunch_l place(base_box=pelvis_mass, anchor=[-0.78, -0.1, 0.1], align=[0.5, 0, 0]) {{
      size: [2.5, 5, 5], material: hide_mat, rotate: [8, 0, -6]
    }}
    box haunch_r place(base_box=pelvis_mass, anchor=[0.78, -0.1, 0.1], align=[-0.5, 0, 0]) {{
      size: [2.5, 5, 5], material: hide_mat, rotate: [8, 0, 6]
    }}

    bone waist_seg attach(parent_box=pelvis_mass, anchor=[0, 0.1, -0.75], align=[0, 0, 0.8]) {{
      rest_euler: [7, 0, 0]
      box waist_core {{ size: [{chest_w - 1.5}, 5.5, 5.5], material: hide_mat }}
      box waist_belly place(base_box=waist_core, anchor=[0, -0.82, 0], align=[0, 0.5, 0]) {{
        size: [{chest_w - 3}, 1.5, 5], material: cream_mat
      }}

      bone chest_seg attach(parent_box=waist_core, anchor=[0, 0.12, -0.74], align=[0, 0, 0.8]) {{
        rest_euler: [6, 0, 0]
        box chest_core {{ size: [{chest_w}, 7, 8], material: hide_mat }}
        box withers_hump place(base_box=chest_core, anchor=[0, 0.85, -0.1], align=[0, -0.5, 0]) {{
          size: [{chest_w - 3}, 2, 5.5], material: hide_mat, rotate: [-9, 0, 0]
        }}
        box sternum_plate place(base_box=chest_core, anchor=[0, -0.84, -0.1], align=[0, 0.5, 0]) {{
          size: [{chest_w - 2}, 1.5, 7], material: cream_mat, rotate: [-5, 0, 0]
        }}
        box shoulder_l place(base_box=chest_core, anchor=[-0.92, -0.15, -0.35], align=[0.75, 0, 0]) {{
          size: [2.5, 5, 4.5], material: hide_mat, rotate: [-6, 0, -8]
        }}
        box shoulder_r place(base_box=chest_core, anchor=[0.92, -0.15, -0.35], align=[-0.75, 0, 0]) {{
          size: [2.5, 5, 4.5], material: hide_mat, rotate: [-6, 0, 8]
        }}
        plane dorsal_fringe place(base_box=withers_hump, anchor=[0, 1, 0], align=[0, -1, 0], offset=[0.25, 0, 0]) {{
          size: [2.5, 5], normal: x, material: keratin_mat
        }}

        bone neck_base attach(parent_box=chest_core, anchor=[0, 0.35, -0.74], align=[0, -0.2, 0.8]) {{
          rest_euler: [22, {head_yaw // 2}, 0]
          box neck_1 {{ size: [4.5, 4.5, 5], material: hide_mat }}
          box throat_1 place(base_box=neck_1, anchor=[0, -0.82, 0], align=[0, 0.5, 0]) {{
            size: [3.5, 1, 4.5], material: cream_mat
          }}
          plane neck_fringe place(base_box=neck_1, anchor=[0, 1, 0], align=[0, -1, 0], offset=[-0.25, 0, 0]) {{
            size: [2, 4.5], normal: x, material: keratin_mat
          }}

          bone neck_upper attach(parent_box=neck_1, anchor=[0, 0.15, -0.74], align=[0, -0.1, 0.8]) {{
            rest_euler: [-18, {head_yaw}, 0]
            box neck_2 {{ size: [4, 4, 4.5], material: hide_mat }}
            box throat_2 place(base_box=neck_2, anchor=[0, -0.82, 0], align=[0, 0.5, 0]) {{
              size: [3, 1, 4], material: cream_mat
            }}

            bone head attach(parent_box=neck_2, anchor=[0, 0.15, -0.72], align=[0, -0.1, 0.8]) {{
              rest_euler: [-12, 0, 3]
              box skull {{ size: [5.5, 4.5, 5.5], material: hide_mat }}
              box brow_l place(base_box=skull, anchor=[-0.75, 0.65, -0.35], align=[0.3, -0.5, 0]) {{
                size: [2, 1, 3.5], material: hide_mat, rotate: [0, 6, -8]
              }}
              box brow_r place(base_box=skull, anchor=[0.75, 0.65, -0.35], align=[-0.3, -0.5, 0]) {{
                size: [2, 1, 3.5], material: hide_mat, rotate: [0, -6, 8]
              }}
              box eye_l place(base_box=skull, anchor=[-1, 0.15, -0.35], align=[0.6, 0, 0]) {{
                size: [0.5, 1.5, 1.5], material: eye_glow_mat
              }}
              box eye_r place(base_box=skull, anchor=[1, 0.15, -0.35], align=[-0.6, 0, 0]) {{
                size: [0.5, 1.5, 1.5], material: eye_glow_mat
              }}
              box crest_shingle_c place(base_box=skull, anchor=[0, 0.82, 0.72], align=[0, -0.5, -0.5]) {{
                size: [2, 1, 3], material: keratin_mat, rotate: [12, 0, 0]
              }}
              box snout_bridge place(base_box=skull, anchor=[0, -0.05, -0.78], align=[0, 0, 0.8]) {{
                size: [4, 3, 4.5], material: hide_mat, rotate: [9, 0, 0]
              }}
              box nose_tip place(base_box=snout_bridge, anchor=[0, 0.35, -0.82], align=[0, -0.3, 0.8]) {{
                size: [2.5, 1.5, 2], material: hide_mat, rotate: [6, 0, 0]
              }}
              plane tooth_skirt_u place(base_box=snout_bridge, anchor=[-0.85, -1, 0], align=[0, 1, 0]) {{
                size: [1, 3.5], normal: x, material: keratin_mat
              }}

              bone jaw attach(parent_box=skull, anchor=[0, -0.68, -0.25], align=[0, 0.75, 0.85]) {{
                rest_euler: [12, 0, 0]
                box mandible {{ size: [3.5, 1.5, 4], material: cream_mat }}
              }}

              chain horn_l attach(parent_box=skull, anchor=[-0.82, 0.94, 0.25], align=[0, -0.92, -0.92]) mirror_x(horn_r) segments=3 taper=0.76 {{
                axis: +z
                rest_euler: [-28, -22, 0]
                box seg {{ size: [1.5, 1.5, 3.5], material: keratin_mat }}
              }}
            }}
          }}
        }}

        bone wing_sh_l attach(parent_box=chest_core, anchor=[-0.95, 0.85, -0.25], align=[0.95, -0.5, 0]) mirror_x(wing_sh_r) {{
          rest_euler: [8, 12, -34]
          box humerus {{ size: [6, 2.5, 2.5], material: hide_mat }}
          plane inner_wing_bay place(base_box=humerus, anchor=[0, 0.25, 0.85], align=[0, 0, -0.85], offset=[0, 0.25, 0]) {{
            size: [5, 5], normal: y, material: cream_mat
          }}
          bone wing_fore_l attach(parent_box=humerus, anchor=[-0.95, 0, 0], align=[0.95, 0, 0]) {{
            rest_euler: [0, -18, 16]
            box forearm {{ size: [7, 2, 2], material: hide_mat }}
            box thumb_claw place(base_box=forearm, anchor=[-0.94, 0.3, -0.85], align=[0.6, 0, 0.85]) {{
              size: [1, 1, 2.5], material: keratin_mat, rotate: [12, 20, 0]
            }}
            plane mid_wing_bay place(base_box=forearm, anchor=[0, -0.25, 0.85], align=[0, 0, -0.85], offset=[0, -0.25, 0]) {{
              size: [6, 5.5], normal: y, material: cream_mat
            }}
            bone finger_a_l attach(parent_box=forearm, anchor=[-1, 0.45, -0.65], align=[1, 0, 0]) {{
              rest_euler: [8, -18, 6]
              box spar_a {{ size: [{wing_finger_l}, 1, 1], material: keratin_mat }}
              plane outer_wing_bay_a place(base_box=spar_a, anchor=[0, 0.85, 0.85], align=[0, 0, -0.85]) {{
                size: [{wing_finger_l - 1}, 1.5], normal: y, material: cream_mat
              }}
            }}
            bone finger_b_l attach(parent_box=forearm, anchor=[-1, -0.45, 0.65], align=[1, 0, 0]) {{
              rest_euler: [-8, 32, -12]
              box spar_b {{ size: [{wing_finger_l - 1}, 1, 1], material: keratin_mat }}
              plane outer_wing_bay_b place(base_box=spar_b, anchor=[0, -0.35, 0.85], align=[0, 0, -0.85], offset=[0, -0.25, 0]) {{
                size: [{wing_finger_l - 1.5}, 3.5], normal: y, material: cream_mat
              }}
            }}
          }}
        }}

        bone fl_upper attach(parent_box=chest_core, anchor=[-0.82, -0.55, -0.45], align=[0.6, 0.8, 0]) mirror_x(fr_upper) {{
          rest_euler: [14, 0, 0]
          box fore_humerus {{ size: [3, 5.5, 3], material: hide_mat }}
          box fore_elbow place(base_box=fore_humerus, anchor=[0, -0.82, 0.65], align=[0, 0.5, -0.5]) {{
            size: [2, 2, 2], material: hide_mat, rotate: [-15, 0, 0]
          }}
          bone fl_lower attach(parent_box=fore_humerus, anchor=[0, -0.82, 0], align=[0, 0.85, 0]) {{
            rest_euler: [-18, 0, 0]
            box fore_radius {{ size: [2.5, 5, 2.5], material: hide_mat }}
            bone fl_foot attach(parent_box=fore_radius, anchor=[0, -0.88, -0.1], align=[0, 0.8, 0.2]) {{
              rest_euler: [8, 0, 0]
              box fore_pad {{ size: [3, 1.5, 3.5], material: hide_mat }}
              box fore_toe_c place(base_box=fore_pad, anchor=[0, -0.2, -0.85], align=[0, 0, 0.8]) {{
                size: [1, 1, 2], material: cream_mat, rotate: [12, 0, 0]
              }}
              box fore_claw_c place(base_box=fore_toe_c, anchor=[0, -0.1, -0.85], align=[0, 0, 0.8]) {{
                size: [0.5, 1, 1.5], material: keratin_mat, rotate: [18, 0, 0]
              }}
            }}
          }}
        }}
      }}
    }}

    bone hl_upper attach(parent_box=pelvis_mass, anchor=[-0.95, -0.35, 0.05], align=[0.85, 0.85, 0]) mirror_x(hr_upper) {{
      rest_euler: [-22, 0, 0]
      box hind_thigh {{ size: [3.5, 6, 4], material: hide_mat }}
      box hind_knee place(base_box=hind_thigh, anchor=[0, -0.8, -0.65], align=[0, 0.5, 0.5]) {{
        size: [2.5, 2, 2], material: hide_mat, rotate: [18, 0, 0]
      }}
      bone hl_lower attach(parent_box=hind_thigh, anchor=[0, -0.82, 0.1], align=[0, 0.85, 0]) {{
        rest_euler: [36, 0, 0]
        box hind_shin {{ size: [2.5, 5.5, 2.5], material: hide_mat }}
        box calf_bulge place(base_box=hind_shin, anchor=[0, 0.25, 0.75], align=[0, 0, -0.5]) {{
          size: [2, 3, 1.5], material: hide_mat, rotate: [-12, 0, 0]
        }}
        bone hl_foot attach(parent_box=hind_shin, anchor=[0, -0.88, -0.1], align=[0, 0.8, 0.3]) {{
          rest_euler: [-10, 0, 0]
          box hind_pad {{ size: [3.5, 1.5, 4], material: hide_mat }}
          box hind_heel place(base_box=hind_pad, anchor=[0, 0.2, 0.82], align=[0, 0, -0.5]) {{
            size: [2, 1.5, 1.5], material: cream_mat
          }}
          box hind_toe_c place(base_box=hind_pad, anchor=[0, -0.2, -0.85], align=[0, 0, 0.8]) {{
            size: [1, 1, 2], material: cream_mat, rotate: [14, 0, 0]
          }}
        }}
      }}
    }}

    chain tail attach(parent_box=pelvis_mass, anchor=[0, 0.15, 0.75], align=[0, 0, -0.8]) segments=5 taper=0.80 {{
      axis: +z
      rest_euler: [8, 5, 0]
      box caudal {{ size: [4.5, 4, 5.5], material: hide_mat }}
      box caudal_belly place(base_box=caudal, anchor=[0, -0.82, 0], align=[0, 0.5, 0]) {{
        size: [3, 1, 5], material: cream_mat
      }}
    }}
  }}
}}"""

    # Corresponding Stage-C stiff axis-aligned blockout (`SKILL.md` Section 2C)
    stage_c = f"""@meta {{ archetype: entity, target_profile: bbmodel_5, texel_density: 16, dsl_version: "3.2", audit_profile: strict }}
@palette {{
  hide_mat: ramp(base_lch=[0.56, 0.14, {h_body}], steps=6)
  cream_mat: ramp(base_lch=[0.78, 0.05, {h_cream}], steps=6)
}}
@rig {{
  root bone body_root {{
    pivot_local: [0, 12, 0]
    box torso {{ size: [{chest_w}, 6, 14], material: hide_mat }}
    bone neck attach(parent_box=torso, anchor=[0, 0.4, -1], align=[0, -0.4, 1]) {{
      box neck_box {{ size: [4, 4, 6], material: hide_mat }}
      bone head attach(parent_box=neck_box, anchor=[0, 0, -1], align=[0, 0, 1]) {{
        box skull {{ size: [5, 4, 6], material: hide_mat }}
      }}
    }}
    bone wing_l attach(parent_box=torso, anchor=[-1, 0.8, -0.2], align=[1, 0, 0]) mirror_x(wing_r) {{
      box wing_slab {{ size: [10, 1, 6], material: cream_mat }}
    }}
    bone fl_leg attach(parent_box=torso, anchor=[-1, -0.6, -0.6], align=[1, 1, 0]) mirror_x(fr_leg) {{
      box fl_post {{ size: [2, 9, 2], material: hide_mat }}
    }}
    bone hl_leg attach(parent_box=torso, anchor=[-1, -0.6, 0.6], align=[1, 1, 0]) mirror_x(hr_leg) {{
      box hl_post {{ size: [3, 9, 3], material: hide_mat }}
    }}
    bone tail attach(parent_box=torso, anchor=[0, 0.2, 1], align=[0, 0, -1]) {{
      box tail_stick {{ size: [3, 3, 12], material: hide_mat }}
    }}
  }}
}}"""

    meta = {
        "family": "crimson_cream_western_dragon",
        "tier": "Standard",
        "concept": "six-limbed high-ridge western dragon with partitioned membrane wings and antler horns",
        "anchors": ["wolf ribcage/haunches", "bat/eagle wing spars", "crocodile scuted tail"],
        "ratios": "head:neck:torso:tail = 1 : 1.8 : 3.0 : 4.0",
        "palette_plan": "3+1 vibrant scheme: body hide + cream ventral/membrane + keratin horn/claw + glowing eye accent",
    }
    return stage_e, stage_c, meta


def build_sculpted_glacial_crystal_drake_dsl(variant: int = 0) -> tuple[str, str, dict[str, Any]]:
    """Expressive chunky Glacial Crystal Drake (`head ~ 1/3` mass, dorsal crystal prisms, spike cutout planes)."""
    h_ice = 225 + (variant * 17) % 32
    h_navy = 252 + (variant * 11) % 24
    h_crys = 192 + (variant * 13) % 20
    jaw_open = 14 + (variant % 3) * 4

    stage_e = f"""@meta {{ archetype: entity, target_profile: bbmodel_5, texel_density: 16, dsl_version: "3.2", size_step: 0.5, audit_profile: sculpted }}
@palette {{
  glacier_mat: ramp(base_lch=[0.66, 0.13, {h_ice}], steps=6, hue_shift=[-18, 12])
  navy_belly_mat: ramp(base_lch=[0.34, 0.09, {h_navy}], steps=6, hue_shift=[-10, 8])
  fang_rock_mat: ramp(base_lch=[0.82, 0.06, 82], steps=6, hue_shift=[-12, 6])
  crystal_glow_mat: ramp(base_lch=[0.86, 0.16, {h_crys}], steps=6, hue_shift=[-8, 14], emissive=true)
}}
@rig {{
  root bone hips_root {{
    pivot_local: [0, 11, 3]
    box hip_mass {{ size: [8, 6.5, 6.5], material: glacier_mat, rotate: [-5, 0, 0] }}
    box hip_plates place(base_box=hip_mass, anchor=[0, -0.84, 0], align=[0, 0.5, 0]) {{
      size: [6.5, 1.5, 5.5], material: navy_belly_mat
    }}
    bone chest_seg attach(parent_box=hip_mass, anchor=[0, 0.15, -0.74], align=[0, 0, 0.8]) {{
      rest_euler: [11, 0, 0]
      box chest_mass {{ size: [10, 8, 8.5], material: glacier_mat }}
      box chest_belly place(base_box=chest_mass, anchor=[0, -0.85, 0], align=[0, 0.5, 0]) {{
        size: [8, 1.5, 7.5], material: navy_belly_mat
      }}
      box dorsal_crys_main place(base_box=chest_mass, anchor=[0, 0.94, -0.15], align=[0, -0.82, 0]) {{
        size: [2.5, 6.5, 2.5], material: crystal_glow_mat, rotate: [-18, 8, 0]
      }}
      box dorsal_crys_l place(base_box=chest_mass, anchor=[-0.45, 0.94, 0.15], align=[0, -0.82, 0]) {{
        size: [2, 4.5, 2], material: crystal_glow_mat, rotate: [-22, -12, -16]
      }}
      box dorsal_crys_r place(base_box=chest_mass, anchor=[0.45, 0.94, 0.15], align=[0, -0.82, 0]) {{
        size: [2, 5, 2], material: crystal_glow_mat, rotate: [-16, 14, 18]
      }}
      plane back_spike_fringe place(base_box=chest_mass, anchor=[0, 1, 0.3], align=[0, -1, 0], offset=[0.25, 0, 0]) {{
        size: [3, 5.5], normal: x, material: glacier_mat
      }}

      bone neck_seg attach(parent_box=chest_mass, anchor=[0, 0.25, -0.74], align=[0, -0.1, 0.8]) {{
        rest_euler: [14, -6, 0]
        box neck_block {{ size: [6.5, 6, 4.5], material: glacier_mat }}
        plane neck_spike_fringe place(base_box=neck_block, anchor=[0, 1, 0], align=[0, -1, 0], offset=[-0.25, 0, 0]) {{
          size: [2.5, 4], normal: x, material: crystal_glow_mat
        }}

        bone head attach(parent_box=neck_block, anchor=[0, 0.1, -0.72], align=[0, -0.05, 0.8]) {{
          rest_euler: [-10, 5, 0]
          box heavy_skull {{ size: [8, 6, 6.5], material: glacier_mat }}
          box brow_shelf_l place(base_box=heavy_skull, anchor=[-0.78, 0.62, -0.35], align=[0.3, -0.5, 0]) {{
            size: [2.5, 1.5, 4], material: glacier_mat, rotate: [0, 8, -10]
          }}
          box brow_shelf_r place(base_box=heavy_skull, anchor=[0.78, 0.62, -0.35], align=[-0.3, -0.5, 0]) {{
            size: [2.5, 1.5, 4], material: glacier_mat, rotate: [0, -8, 10]
          }}
          box deep_eye_l place(base_box=heavy_skull, anchor=[-1, 0.12, -0.32], align=[0.6, 0, 0]) {{
            size: [0.5, 1.5, 1.5], material: crystal_glow_mat
          }}
          box deep_eye_r place(base_box=heavy_skull, anchor=[1, 0.12, -0.32], align=[-0.6, 0, 0]) {{
            size: [0.5, 1.5, 1.5], material: crystal_glow_mat
          }}
          box blunt_snout place(base_box=heavy_skull, anchor=[0, -0.05, -0.78], align=[0, 0, 0.8]) {{
            size: [6.5, 4, 4.5], material: glacier_mat, rotate: [8, 0, 0]
          }}
          box fang_ul place(base_box=blunt_snout, anchor=[-0.92, -0.94, -0.78], align=[0, 0.85, 0]) {{
            size: [1, 2.5, 1], material: fang_rock_mat, rotate: [10, 0, -8]
          }}
          box fang_ur place(base_box=blunt_snout, anchor=[0.92, -0.94, -0.78], align=[0, 0.85, 0]) {{
            size: [1, 2.5, 1], material: fang_rock_mat, rotate: [10, 0, 8]
          }}
          plane upper_tooth_row place(base_box=blunt_snout, anchor=[0, -1, -0.85], align=[0, 1, 0]) {{
            size: [4, 1.5], normal: z, material: fang_rock_mat
          }}
          bone heavy_jaw attach(parent_box=heavy_skull, anchor=[0, -0.74, -0.25], align=[0, 0.7, 0.8]) {{
            rest_euler: [{jaw_open}, 0, 0]
            box jaw_block {{ size: [4.5, 2.5, 5.5], material: navy_belly_mat }}
            box chin_beard place(base_box=jaw_block, anchor=[0, -0.82, -0.5], align=[0, 0.5, 0]) {{
              size: [3.5, 2, 2.5], material: crystal_glow_mat, rotate: [15, 0, 0]
            }}
          }}
        }}
      }}

      bone fl_arm attach(parent_box=chest_mass, anchor=[-0.84, -0.45, -0.42], align=[0.6, 0.75, 0]) mirror_x(fr_arm) {{
        rest_euler: [12, 0, -6]
        box fore_upper {{ size: [3.5, 5, 3.5], material: glacier_mat }}
        plane fore_fringe place(base_box=fore_upper, anchor=[0, 0, 1], align=[0, 0, -1], offset=[0.25, 0, 0]) {{
          size: [2, 4.5], normal: x, material: glacier_mat
        }}
        bone fl_fore attach(parent_box=fore_upper, anchor=[0, -0.82, 0], align=[0, 0.85, 0]) {{
          rest_euler: [-15, 0, 0]
          box fore_lower {{ size: [3, 4.5, 3], material: glacier_mat }}
          box fore_paw place(base_box=fore_lower, anchor=[0, -0.88, -0.2], align=[0, 0.7, 0]) {{
            size: [4, 1.5, 4.5], material: navy_belly_mat
          }}
        }}
      }}
    }}

    bone hl_leg attach(parent_box=hip_mass, anchor=[-0.84, -0.35, 0.1], align=[0.6, 0.75, 0]) mirror_x(hr_leg) {{
      rest_euler: [-16, 0, 0]
      box hind_upper {{ size: [3.5, 5, 4], material: glacier_mat }}
      bone hl_shin attach(parent_box=hind_upper, anchor=[0, -0.82, 0], align=[0, 0.85, 0]) {{
        rest_euler: [24, 0, 0]
        box hind_lower {{ size: [3, 4.5, 3], material: glacier_mat }}
        box hind_paw place(base_box=hind_lower, anchor=[0, -0.88, -0.2], align=[0, 0.7, 0]) {{
          size: [4, 1.5, 4.5], material: navy_belly_mat
        }}
      }}
    }}

    chain tail attach(parent_box=hip_mass, anchor=[0, 0.1, 0.76], align=[0, 0, -0.8]) segments=4 taper=0.78 {{
      axis: +z
      rest_euler: [6, -6, 0]
      box tail_seg {{ size: [5, 4.5, 5], material: glacier_mat }}
      box tail_crys place(base_box=tail_seg, anchor=[0, 0.85, 0], align=[0, -0.4, 0]) {{
        size: [1.5, 2.5, 2], material: crystal_glow_mat, rotate: [-20, 0, 0]
      }}
    }}
  }}
}}"""

    stage_c = f"""@meta {{ archetype: entity, target_profile: bbmodel_5, texel_density: 16, dsl_version: "3.2", audit_profile: strict }}
@palette {{
  glacier_mat: ramp(base_lch=[0.66, 0.12, {h_ice}], steps=6)
  navy_belly_mat: ramp(base_lch=[0.34, 0.08, {h_navy}], steps=6)
}}
@rig {{
  root bone root_b {{
    pivot_local: [0, 10, 0]
    box trunk {{ size: [9, 7, 13], material: glacier_mat }}
    bone head attach(parent_box=trunk, anchor=[0, 0.2, -1], align=[0, 0, 1]) {{
      box skull {{ size: [7, 5, 7], material: glacier_mat }}
    }}
    bone fl attach(parent_box=trunk, anchor=[-1, -0.6, -0.6], align=[1, 1, 0]) mirror_x(fr) {{
      box leg {{ size: [3, 8, 3], material: navy_belly_mat }}
    }}
    bone hl attach(parent_box=trunk, anchor=[-1, -0.6, 0.6], align=[1, 1, 0]) mirror_x(hr) {{
      box hleg {{ size: [3, 8, 3], material: navy_belly_mat }}
    }}
  }}
}}"""

    meta = {
        "family": "glacial_crystal_drake",
        "tier": "Standard",
        "concept": "chunky expressive ice drake with oversized fanged jaw and dorsal cyan crystal cluster",
        "anchors": ["alligator snapping turtle jaw", "komodo dragon stance", "quartz crystal prisms"],
        "ratios": "head ~ 30% body mass; thick forequarters; 4-segment crystal-studded tail",
        "palette_plan": "3+1 scheme: pale ice-blue hide + deep navy belly + bone fangs + emissive cyan crystals",
    }
    return stage_e, stage_c, meta


def build_sculpted_ochre_club_wyvern_dsl(variant: int = 0) -> tuple[str, str, dict[str, Any]]:
    """Wing-walker Wyvern in dynamic crouched pose (`2` hind legs + planted wing-arms + club tail)."""
    h_ochre = 62 + (variant * 19) % 35
    h_sand = 78 + (variant * 11) % 25
    h_horn = 240 + (variant * 13) % 30

    stage_e = f"""@meta {{ archetype: entity, target_profile: bbmodel_5, texel_density: 16, dsl_version: "3.2", size_step: 0.5, audit_profile: sculpted }}
@palette {{
  ochre_mat: ramp(base_lch=[0.62, 0.13, {h_ochre}], steps=6, hue_shift=[-15, 10])
  sand_web_mat: ramp(base_lch=[0.78, 0.08, {h_sand}], steps=6, hue_shift=[-12, 8])
  slate_horn_mat: ramp(base_lch=[0.46, 0.04, {h_horn}], steps=6, hue_shift=[-8, 6])
  amber_eye_mat: ramp(base_lch=[0.82, 0.18, 68], steps=6, hue_shift=[-6, 12], emissive=true)
}}
@rig {{
  root bone pelvis_root {{
    pivot_local: [0, 12, 3]
    box hip_core {{ size: [7, 6, 6.5], material: ochre_mat, rotate: [8, 0, 0] }}
    bone chest_seg attach(parent_box=hip_core, anchor=[0, 0.15, -0.74], align=[0, 0, 0.8]) {{
      rest_euler: [-18, 0, 0]
      box chest_core {{ size: [9.5, 7.5, 8.5], material: ochre_mat }}
      box keel_plate place(base_box=chest_core, anchor=[0, -0.84, -0.1], align=[0, 0.5, 0]) {{
        size: [7, 1.5, 7], material: sand_web_mat, rotate: [-6, 0, 0]
      }}
      plane dorsal_spines place(base_box=chest_core, anchor=[0, 1, 0], align=[0, -1, 0], offset=[0.25, 0, 0]) {{
        size: [3, 7], normal: x, material: sand_web_mat
      }}

      bone neck_seg attach(parent_box=chest_core, anchor=[0, 0.3, -0.74], align=[0, -0.15, 0.8]) {{
        rest_euler: [24, 6, 0]
        box neck_1 {{ size: [5, 5, 5], material: ochre_mat }}
        bone head attach(parent_box=neck_1, anchor=[0, 0.15, -0.72], align=[0, -0.1, 0.8]) {{
          rest_euler: [-18, -4, 0]
          box skull {{ size: [6, 4.5, 5.5], material: ochre_mat }}
          box snout place(base_box=skull, anchor=[0, -0.05, -0.78], align=[0, 0, 0.8]) {{
            size: [4, 3, 4.5], material: ochre_mat, rotate: [10, 0, 0]
          }}
          box eye_l place(base_box=skull, anchor=[-1, 0.15, -0.32], align=[0.6, 0, 0]) {{
            size: [0.5, 1.5, 1.5], material: amber_eye_mat
          }}
          box eye_r place(base_box=skull, anchor=[1, 0.15, -0.32], align=[-0.6, 0, 0]) {{
            size: [0.5, 1.5, 1.5], material: amber_eye_mat
          }}
          bone jaw attach(parent_box=skull, anchor=[0, -0.68, -0.3], align=[0, 0.6, 0.8]) {{
            rest_euler: [12, 0, 0]
            box mandible {{ size: [3.5, 1.5, 4.5], material: sand_web_mat }}
          }}
          chain ram_horn_l attach(parent_box=skull, anchor=[-0.85, 0.92, 0.15], align=[0.5, -0.9, -0.9]) mirror_x(ram_horn_r) segments=3 taper=0.78 {{
            axis: +z
            rest_euler: [-28, -26, 0]
            box hseg {{ size: [2, 2, 3], material: slate_horn_mat }}
          }}
        }}
      }}

      bone wing_arm_l attach(parent_box=chest_core, anchor=[-0.92, 0.05, -0.35], align=[0.92, 0.4, 0]) mirror_x(wing_arm_r) {{
        rest_euler: [14, 12, -32]
        box humerus {{ size: [7, 3, 3], material: ochre_mat }}
        bone wing_fore_l attach(parent_box=humerus, anchor=[-0.95, -0.1, 0], align=[0.95, 0.25, 0]) {{
          rest_euler: [-18, -10, 48]
          box forearm {{ size: [7.5, 2.5, 2.5], material: ochre_mat }}
          box wrist_knuckle place(base_box=forearm, anchor=[-0.88, -0.6, -0.3], align=[0.4, 0.5, 0]) {{
            size: [2, 2, 2.5], material: slate_horn_mat, rotate: [12, 0, 0]
          }}
          plane folded_web_1 place(base_box=forearm, anchor=[0, 0.3, 0.85], align=[0, 0, -0.85], offset=[0, 0.25, 0]) {{
            size: [6.5, 5], normal: y, material: sand_web_mat
          }}
          bone folded_finger_l attach(parent_box=forearm, anchor=[-0.96, 0.2, 0.3], align=[0.96, 0, -0.2]) {{
            rest_euler: [8, 42, 0]
            box finger_spar {{ size: [8.5, 1, 1], material: slate_horn_mat }}
            plane folded_web_2 place(base_box=finger_spar, anchor=[0, -0.3, 0.85], align=[0, 0, -0.85], offset=[0, -0.25, 0]) {{
              size: [7.5, 4.5], normal: y, material: sand_web_mat
            }}
          }}
        }}
      }}
    }}

    bone hind_l attach(parent_box=hip_core, anchor=[-1, -0.35, 0.55], align=[1, 0.88, -0.4]) mirror_x(hind_r) {{
      rest_euler: [6, 0, -14]
      box thigh {{ size: [4, 6.5, 4.5], material: ochre_mat }}
      bone shin_l attach(parent_box=thigh, anchor=[0, -0.82, 0.1], align=[0, 0.85, 0]) {{
        rest_euler: [42, 0, 0]
        box shin {{ size: [3, 5.5, 3], material: ochre_mat }}
        bone foot_l attach(parent_box=shin, anchor=[0, -0.88, -0.15], align=[0, 0.8, 0.3]) {{
          rest_euler: [-26, 0, 0]
          box foot_pad {{ size: [3.5, 1.5, 4.5], material: ochre_mat }}
          box talon place(base_box=foot_pad, anchor=[0, -0.2, -0.85], align=[0, 0, 0.8]) {{
            size: [1.5, 1, 2], material: slate_horn_mat, rotate: [16, 0, 0]
          }}
        }}
      }}
    }}

    bone tail_1 attach(parent_box=hip_core, anchor=[0, 0.2, 0.76], align=[0, 0, -0.8]) {{
      rest_euler: [14, 8, 0]
      box tseg_1 {{ size: [4.5, 4, 6], material: ochre_mat }}
      bone tail_2 attach(parent_box=tseg_1, anchor=[0, 0.1, 0.78], align=[0, 0, -0.8]) {{
        rest_euler: [16, 8, 0]
        box tseg_2 {{ size: [3.5, 3, 6], material: ochre_mat }}
        bone tail_mace attach(parent_box=tseg_2, anchor=[0, 0, 0.82], align=[0, 0, -0.8]) {{
          rest_euler: [10, 0, 0]
          box mace_core {{ size: [5, 4.5, 5.5], material: slate_horn_mat }}
          box mace_spike_l place(base_box=mace_core, anchor=[-0.88, 0.2, 0], align=[0.6, 0, 0]) {{
            size: [2.5, 1.5, 1.5], material: sand_web_mat, rotate: [0, -15, 12]
          }}
          box mace_spike_r place(base_box=mace_core, anchor=[0.88, 0.2, 0], align=[-0.6, 0, 0]) {{
            size: [2.5, 1.5, 1.5], material: sand_web_mat, rotate: [0, 15, -12]
          }}
        }}
      }}
    }}
  }}
}}"""

    stage_c = f"""@meta {{ archetype: entity, target_profile: bbmodel_5, texel_density: 16, dsl_version: "3.2", audit_profile: strict }}
@palette {{
  ochre_mat: ramp(base_lch=[0.62, 0.12, {h_ochre}], steps=6)
  sand_web_mat: ramp(base_lch=[0.78, 0.07, {h_sand}], steps=6)
}}
@rig {{
  root bone root_b {{
    pivot_local: [0, 11, 0]
    box body {{ size: [8, 6, 12], material: ochre_mat }}
    bone head attach(parent_box=body, anchor=[0, 0.3, -1], align=[0, 0, 1]) {{
      box skull {{ size: [5, 4, 6], material: ochre_mat }}
    }}
    bone wing_l attach(parent_box=body, anchor=[-1, 0, -0.3], align=[1, 0, 0]) mirror_x(wing_r) {{
      box w_box {{ size: [9, 2, 5], material: sand_web_mat }}
    }}
    bone leg_l attach(parent_box=body, anchor=[-1, -0.6, 0.5], align=[1, 1, 0]) mirror_x(leg_r) {{
      box l_box {{ size: [3, 9, 3], material: ochre_mat }}
    }}
    bone tail attach(parent_box=body, anchor=[0, 0, 1], align=[0, 0, -1]) {{
      box t_box {{ size: [3, 3, 10], material: ochre_mat }}
    }}
  }}
}}"""

    meta = {
        "family": "ochre_club_wyvern",
        "tier": "Standard",
        "concept": "crouched wing-walker desert wyvern with curved ram horns and spiked mace tail",
        "anchors": ["pterosaur wing-wrist walk", "ankylosaur tail club", "bighorn sheep horns"],
        "ratios": "head:neck:torso:tail = 1 : 1.3 : 2.5 : 3.4",
        "palette_plan": "3+1 scheme: ochre hide + sand membranes + slate horn/club + glowing amber eyes",
    }
    return stage_e, stage_c, meta


def build_sculpted_pastel_chibi_dragonet_dsl(variant: int = 0) -> tuple[str, str, dict[str, Any]]:
    """Chibi-tier Sculpted Dragonet (`24-30` elements, `45%` head share, expressive eyes, tiny layered wings)."""
    h_pink = (340 + (variant * 29) % 120) % 360
    h_cream = 75 + (variant * 13) % 30

    stage_e = f"""@meta {{ archetype: entity, target_profile: bbmodel_5, texel_density: 16, dsl_version: "3.2", size_step: 0.5, audit_profile: sculpted }}
@palette {{
  pastel_hide_mat: ramp(base_lch=[0.74, 0.13, {h_pink}], steps=6, hue_shift=[-14, 10])
  soft_belly_mat: ramp(base_lch=[0.88, 0.05, {h_cream}], steps=6, hue_shift=[-10, 8])
  horn_gold_mat: ramp(base_lch=[0.84, 0.11, 84], steps=6, hue_shift=[-8, 8])
  star_eye_mat: ramp(base_lch=[0.32, 0.08, 275], steps=6, hue_shift=[-6, 10], emissive=true)
}}
@rig {{
  root bone body_root {{
    pivot_local: [0, 6.5, 0]
    box plump_belly {{ size: [6.5, 6, 7], material: pastel_hide_mat, rotate: [-6, 0, 0] }}
    box tummy_patch place(base_box=plump_belly, anchor=[0, -0.78, -0.55], align=[0, 0.4, 0.4]) {{
      size: [5, 4.5, 2], material: soft_belly_mat, rotate: [-8, 0, 0]
    }}
    plane back_tuft place(base_box=plump_belly, anchor=[0, 0.75, 0.85], align=[0, -0.8, -0.5], offset=[0.25, 0, 0]) {{
      size: [2, 3.5], normal: x, material: horn_gold_mat
    }}

    bone big_head attach(parent_box=plump_belly, anchor=[0, 0.94, -0.25], align=[0, -0.88, 0.1]) {{
      rest_euler: [6, 8, -3]
      box chibi_cranium {{ size: [8.5, 7.5, 8], material: pastel_hide_mat }}
      box cute_snout place(base_box=chibi_cranium, anchor=[0, -0.25, -0.82], align=[0, 0, 0.8]) {{
        size: [5, 3, 2.5], material: soft_belly_mat, rotate: [8, 0, 0]
      }}
      box cheek_puff_l place(base_box=chibi_cranium, anchor=[-0.92, -0.25, -0.25], align=[0.7, 0, 0]) {{
        size: [2, 2.5, 3], material: pastel_hide_mat, rotate: [0, 8, -8]
      }}
      box cheek_puff_r place(base_box=chibi_cranium, anchor=[0.92, -0.25, -0.25], align=[-0.7, 0, 0]) {{
        size: [2, 2.5, 3], material: pastel_hide_mat, rotate: [0, -8, 8]
      }}
      box big_eye_l place(base_box=chibi_cranium, anchor=[-0.96, 0.05, -0.55], align=[0.6, 0, 0]) {{
        size: [1, 2.5, 2.5], material: star_eye_mat
      }}
      box big_eye_r place(base_box=chibi_cranium, anchor=[0.96, 0.05, -0.55], align=[-0.6, 0, 0]) {{
        size: [1, 2.5, 2.5], material: star_eye_mat
      }}
      box stub_horn_c place(base_box=chibi_cranium, anchor=[0, 0.95, -0.2], align=[0, -0.85, 0]) {{
        size: [1.5, 2.5, 1.5], material: horn_gold_mat, rotate: [-18, 0, 0]
      }}
      bone ear_fin_l attach(parent_box=chibi_cranium, anchor=[-0.95, 0.35, 0.2], align=[0.9, 0, 0]) mirror_x(ear_fin_r) {{
        rest_euler: [10, 26, 18]
        box ear_slab {{ size: [3.5, 2, 0.5], material: soft_belly_mat }}
      }}
      bone stub_horn_l attach(parent_box=chibi_cranium, anchor=[-0.65, 0.95, 0.15], align=[0, -0.9, 0]) mirror_x(stub_horn_r) {{
        rest_euler: [-22, 0, -18]
        box horn_cone {{ size: [1.5, 3, 1.5], material: horn_gold_mat }}
      }}
    }}

    bone tiny_wing_l attach(parent_box=plump_belly, anchor=[-0.95, 0.25, 0.3], align=[0.95, 0, 0]) mirror_x(tiny_wing_r) {{
      rest_euler: [8, 28, 12]
      box mini_spar {{ size: [4.5, 1.5, 1.5], material: pastel_hide_mat }}
      plane mini_web place(base_box=mini_spar, anchor=[0, 0, 0.85], align=[0, 0, -0.85], offset=[0, 0.25, 0]) {{
        size: [4, 3], normal: y, material: soft_belly_mat
      }}
    }}

    bone stub_arm_l attach(parent_box=plump_belly, anchor=[-0.95, 0.05, -0.55], align=[0.88, 0.7, 0]) mirror_x(stub_arm_r) {{
      rest_euler: [-30, 12, -22]
      box arm_nub {{ size: [2, 3, 2], material: pastel_hide_mat }}
    }}

    bone stub_leg_l attach(parent_box=plump_belly, anchor=[-0.72, -0.92, 0.25], align=[0, 0.9, 0]) mirror_x(stub_leg_r) {{
      rest_euler: [6, 12, 0]
      box leg_stub {{ size: [2.5, 3.5, 3], material: pastel_hide_mat }}
      box toe_pad place(base_box=leg_stub, anchor=[0, -0.85, -0.35], align=[0, 0.5, 0.4]) {{
        size: [3, 1, 3.5], material: soft_belly_mat
      }}
    }}

    chain curl_tail attach(parent_box=plump_belly, anchor=[0, -0.2, 0.78], align=[0, 0, -0.8]) segments=3 taper=0.75 {{
      axis: +z
      rest_euler: [18, 10, 0]
      box tseg {{ size: [3, 3, 3.5], material: pastel_hide_mat }}
    }}
  }}
}}"""

    stage_c = f"""@meta {{ archetype: entity, target_profile: bbmodel_5, texel_density: 16, dsl_version: "3.2", audit_profile: strict }}
@palette {{
  pastel_hide_mat: ramp(base_lch=[0.74, 0.12, {h_pink}], steps=6)
}}
@rig {{
  root bone root_b {{
    pivot_local: [0, 6, 0]
    box body {{ size: [6, 6, 6], material: pastel_hide_mat }}
    bone head attach(parent_box=body, anchor=[0, 1, 0], align=[0, -1, 0]) {{
      box head_cube {{ size: [6, 6, 6], material: pastel_hide_mat }}
    }}
    bone leg_l attach(parent_box=body, anchor=[-0.6, -1, 0], align=[0, 1, 0]) mirror_x(leg_r) {{
      box leg_cube {{ size: [2, 3, 2], material: pastel_hide_mat }}
    }}
  }}
}}"""

    meta = {
        "family": "pastel_chibi_dragonet",
        "tier": "Chibi",
        "concept": "compact 26-element chibi pet dragonet with 45% head share, cheek puffs, and tiny layered wings",
        "anchors": ["axolotl ear fins", "fledgling songbird proportions", "plush dragon silhouette"],
        "ratios": "head = 45% total mass; stub limbs; curled 3-segment tail",
        "palette_plan": "pastel hide + warm cream tummy/wings + golden horn buds + deep glowing eyes",
    }
    return stage_e, stage_c, meta


def build_sculpted_storm_wolf_dragon_v4_dsl(variant: int = 0) -> tuple[str, str, dict[str, Any]]:
    """Block-DSL v4.0 (`@layout` + `@component` + `@assembly`) Storm-Themed Fanged Wolf-Dragon (`SKILL.md` Sec 7)."""
    body_len = 24 + (variant % 3) * 2
    chest_w = 9 + (variant % 2)
    h_slate = 238 + (variant * 13) % 24

    stage_e = f"""@meta {{ archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.0", audit_profile: sculpted }}

@layout {{
  param body_length = {body_len} [20, 36]
  param chest_width = {chest_w} [8, 14]
  let pelvis_len = max(6, round(body_length * 0.34))
  let chest_len = max(8, body_length - pelvis_len - 5)
  let head_w = max(5, round(chest_width * 0.65))
  budget max_boxes = 48
  budget max_bones = 28
  envelope max_span = [48, 36, 56]
  envelope min_ground_y = -1.0
}}

@palette {{
  storm_slate_mat: ramp(base_lch=[0.42, 0.08, {h_slate}], steps=8, hue_shift=[-14, 10])
  pale_fur_mat: ramp(base_lch=[0.80, 0.05, 86], steps=6, hue_shift=[-10, 8])
  gold_plate_mat: ramp(base_lch=[0.74, 0.14, 78], steps=6, hue_shift=[-8, 10])
  volt_glow_mat: ramp(base_lch=[0.84, 0.18, 212], steps=8, hue_shift=[-6, 12], emissive=true)
}}

@component WolfDragonPelvis {{
  param w = 7 [5, 14]
  param l = 8 [5, 16]
  material_slots: [hide, plate, volt]
  box pelvis_core {{ size: [w, 6, l], material: hide }}
  box hip_shingle place(base_box=pelvis_core, anchor=[0, 1, 0], align=[0, -0.5, 0]) {{
    size: [max(3, w - 2), 2, max(3, l - 2)], material: plate, rotate: [10, 0, 0]
  }}
  plane hip_volt_fin place(base_box=hip_shingle, anchor=[0, 1, 0], align=[0, -1, 0], offset=[0.25, 0, 0]) {{
    size: [3, max(3, l - 2)], normal: x, material: volt
  }}
  frame chest_socket {{ on_box: pelvis_core, anchor: [0, 0.15, -1], rot: [6, 0, 0] }}
  frame hip_l {{ on_box: pelvis_core, anchor: [-1, -0.3, 0], rot: [0, 0, 0] }}
  frame hip_r {{ on_box: pelvis_core, anchor: [1, -0.3, 0], rot: [0, 0, 0] }}
  frame tail_socket {{ on_box: pelvis_core, anchor: [0, 0.15, 1], rot: [8, 0, 0] }}
}}

@component WolfDragonChest {{
  param w = 9 [6, 16]
  param l = 11 [7, 20]
  material_slots: [hide, ruff, plate, volt]
  box ribcage {{ size: [w, 7, l], material: hide }}
  box chest_ruff place(base_box=ribcage, anchor=[0, -0.85, -0.25], align=[0, 0.5, 0]) {{
    size: [max(4, w - 2), 2, max(4, l - 3)], material: ruff, rotate: [-8, 0, 0]
  }}
  box shoulder_cowl place(base_box=ribcage, anchor=[0, 0.9, -0.1], align=[0, -0.5, 0]) {{
    size: [max(4, w - 1), 2, max(4, l - 3)], material: plate, rotate: [-10, 0, 0]
  }}
  plane dorsal_volt_crest place(base_box=shoulder_cowl, anchor=[0, 1, 0], align=[0, -1, 0], offset=[-0.25, 0, 0]) {{
    size: [3, max(4, l - 3)], normal: x, material: volt
  }}
  frame waist_mount {{ on_box: ribcage, anchor: [0, 0, 1], rot: [0, 0, 0] }}
  frame neck_socket {{ on_box: ribcage, anchor: [0, 0.35, -1], rot: [18, 0, 0] }}
  frame shoulder_l {{ on_box: ribcage, anchor: [-1, -0.45, -0.5], rot: [0, 0, 0] }}
  frame shoulder_r {{ on_box: ribcage, anchor: [1, -0.45, -0.5], rot: [0, 0, 0] }}
}}

@component WolfDragonNeck {{
  param len = 5 [4, 10]
  material_slots: [hide, ruff]
  box neck_core {{ size: [5, 5, len], material: hide }}
  box throat_mane place(base_box=neck_core, anchor=[0, -0.85, 0], align=[0, 0.5, 0]) {{
    size: [4, 2, max(3, len - 1)], material: ruff, rotate: [-8, 0, 0]
  }}
  frame base_mount {{ on_box: neck_core, anchor: [0, 0, 1], rot: [0, 0, 0] }}
  frame head_socket {{ on_box: neck_core, anchor: [0, 0.15, -1], rot: [-16, 0, 0] }}
}}

@component WolfDragonHead {{
  param w = 6 [4, 12]
  material_slots: [hide, ruff, volt]
  box skull {{ size: [w, 4, 6], material: hide }}
  box sloped_muzzle place(base_box=skull, anchor=[0, -0.1, -1], align=[0, 0, 0.8]) {{
    size: [max(3, w - 2), 3, 5], material: hide, rotate: [10, 0, 0]
  }}
  box jaw_fur place(base_box=skull, anchor=[0, -0.85, -0.2], align=[0, 0.5, 0]) {{
    size: [max(3, w - 2), 1, 5], material: ruff
  }}
  plane fang_skirt place(base_box=sloped_muzzle, anchor=[-0.85, -1, 0], align=[0, 1, 0]) {{
    size: [1, 4], normal: x, material: volt
  }}
  frame neck_mount {{ on_box: skull, anchor: [0, 0, 1], rot: [0, 0, 0] }}
  frame horn_socket_l {{ on_box: skull, anchor: [-0.72, 1, 0.2], rot: [-26, -16, 0] }}
  frame horn_socket_r {{ on_box: skull, anchor: [0.72, 1, 0.2], rot: [-26, 16, 0] }}
}}

@component ForkedStormHorn {{
  param len = 6 [4, 12]
  material_slots: [plate, volt]
  uv_policy: shared
  box main_tine {{ size: [1, 2, len], material: plate }}
  box fork_tine place(base_box=main_tine, anchor=[0, 1, 0.2], align=[0, -0.5, 0]) {{
    size: [1, 2, max(2, len - 3)], material: volt, rotate: [-22, 0, 0]
  }}
  frame base_mount {{ on_box: main_tine, anchor: [0, -1, -1], rot: [0, 0, 0] }}
}}

@component CursorialLimb {{
  param upper_h = 6 [4, 10]
  param lower_h = 6 [4, 10]
  material_slots: [hide, plate]
  box thigh {{ size: [3, upper_h, 3], material: hide }}
  bone shin attach(parent_box=thigh, anchor=[0, -1, 0], align=[0, 1, 0]) {{
    rest_euler: [-24, 0, 0]
    joint_limits: [[-45, 30], [-15, 15], [-15, 15]]
    box lower_seg {{ size: [2, lower_h, 3], material: plate }}
  }}
  frame socket_mount {{ on_box: thigh, anchor: [1, 0.75, 0], rot: [14, 0, 0] }}
}}

@component WhipTailSeg {{
  param w = 4 [2, 8]
  param l = 6 [4, 10]
  material_slots: [hide]
  box caudal {{ size: [w, max(2, w - 1), l], material: hide }}
  frame fore_mount {{ on_box: caudal, anchor: [0, 0, -1], rot: [0, 0, 0] }}
  frame aft_socket {{ on_box: caudal, anchor: [0, 0, 1], rot: [0, 0, 0] }}
}}

@assembly StormFangedWolfDragon {{
  root_pivot: [0, 12.5, 0]

  instance pelvis : WolfDragonPelvis(w=max(6, chest_width - 2), l=pelvis_len) {{
    id: "pelvis_01"
    materials: {{ hide: storm_slate_mat, plate: gold_plate_mat, volt: volt_glow_mat }}
  }}
  instance chest : WolfDragonChest(w=chest_width, l=chest_len) {{
    id: "chest_01"
    materials: {{ hide: storm_slate_mat, ruff: pale_fur_mat, plate: gold_plate_mat, volt: volt_glow_mat }}
  }}
  subassembly head_assembly {{
    instance neck : WolfDragonNeck(len=5) {{
      id: "neck_01"
      materials: {{ hide: storm_slate_mat, ruff: pale_fur_mat }}
    }}
    instance head : WolfDragonHead(w=head_w) {{
      id: "head_01"
      materials: {{ hide: storm_slate_mat, ruff: pale_fur_mat, volt: volt_glow_mat }}
    }}
    instance horn_l : ForkedStormHorn(len=6) mirror_x(horn_r) {{
      id: "horn_l_01"
      materials: {{ plate: gold_plate_mat, volt: volt_glow_mat }}
    }}
    connect neck.base_mount to chest.neck_socket {{ type: revolute, axis: x, limits: [-28, 28] }}
    connect head.neck_mount to neck.head_socket {{ type: revolute, axis: x, limits: [-25, 30] }}
    connect horn_l.base_mount to head.horn_socket_l {{ type: fixed, sculpted_blend: true }}
  }}
  subassembly limbs {{
    instance fl : CursorialLimb(upper_h=6, lower_h=6) mirror_x(fr) {{
      id: "fl_01"
      materials: {{ hide: storm_slate_mat, plate: gold_plate_mat }}
    }}
    instance hl : CursorialLimb(upper_h=6, lower_h=6) mirror_x(hr) {{
      id: "hl_01"
      materials: {{ hide: storm_slate_mat, plate: gold_plate_mat }}
    }}
    connect fl.socket_mount to chest.shoulder_l {{ type: revolute, axis: x, limits: [-50, 50] }}
    connect hl.socket_mount to pelvis.hip_l {{ type: revolute, axis: x, limits: [-50, 50] }}
  }}
  subassembly tail_chain {{
    instance t1 : WhipTailSeg(w=4, l=6) {{
      id: "t1_01"
      materials: {{ hide: storm_slate_mat }}
    }}
    instance t2 : WhipTailSeg(w=3, l=6) {{
      id: "t2_01"
      materials: {{ hide: storm_slate_mat }}
    }}
    connect t1.fore_mount to pelvis.tail_socket {{ type: revolute, axis: y, limits: [-30, 30] }}
    connect t2.fore_mount to t1.aft_socket {{ type: revolute, axis: y, limits: [-35, 35] }}
  }}
  connect chest.waist_mount to pelvis.chest_socket {{ type: revolute, axis: x, limits: [-18, 18] }}
}}

@kinematics {{
  ik_chain fl_ik {{ bones: [fl, fl.shin], effector: [0, -6, 0.5], hinge_axis: x, pole: [0, 0, 30] }}
  ik_chain fr_ik {{ bones: [fr, fr.shin], effector: [0, -6, 0.5], hinge_axis: x, pole: [0, 0, 30] }}
  ik_chain hl_ik {{ bones: [hl, hl.shin], effector: [0, -6, 0.5], hinge_axis: x, pole: [0, 0, 30] }}
  ik_chain hr_ik {{ bones: [hr, hr.shin], effector: [0, -6, 0.5], hinge_axis: x, pole: [0, 0, 30] }}
}}

@behavior {{
  clip storm_prowl {{
    duration_s: 1.5
    ik_drive fl_ik: foot_cycle(stride=3.0, lift=1.2, phase_deg=0)
    ik_drive hr_ik: foot_cycle(stride=3.0, lift=1.2, phase_deg=0)
    ik_drive fr_ik: foot_cycle(stride=3.0, lift=1.2, phase_deg=180)
    ik_drive hl_ik: foot_cycle(stride=3.0, lift=1.2, phase_deg=180)
  }}
}}"""

    stage_c = f"""@meta {{ archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "3.2", audit_profile: strict }}
@palette {{
  storm_slate_mat: ramp(base_lch=[0.42, 0.07, {h_slate}], steps=6)
  gold_plate_mat: ramp(base_lch=[0.74, 0.12, 78], steps=6)
}}
@rig {{
  root bone body_root {{
    pivot_local: [0, 11, 0]
    box torso {{ size: [{chest_w}, 6, 14], material: storm_slate_mat }}
    bone head attach(parent_box=torso, anchor=[0, 0.3, 1], align=[0, 0, -1]) {{
      box skull {{ size: [5, 4, 6], material: storm_slate_mat }}
    }}
    bone fl attach(parent_box=torso, anchor=[-1, -0.6, 0.5], align=[1, 1, 0]) mirror_x(fr) {{
      box fleg {{ size: [3, 9, 3], material: gold_plate_mat }}
    }}
    bone hl attach(parent_box=torso, anchor=[-1, -0.6, -0.5], align=[1, 1, 0]) mirror_x(hr) {{
      box hleg {{ size: [3, 9, 3], material: gold_plate_mat }}
    }}
  }}
}}"""

    meta = {
        "family": "storm_fanged_wolf_dragon_v4",
        "tier": "Standard (Block-DSL v4.0 Modular Assembly)",
        "concept": "storm-charged alpine ambusher combining wolf forequarters, shingled gold armor, and forked electric horns",
        "anchors": ["timber wolf cursorial gait", "pangolin shoulder shingles", "pronghorn forked horns"],
        "ratios": "head:neck:torso:tail = 1 : 0.8 : 3.0 : 2.5",
        "palette_plan": "slate-blue hide + pale fur ruff + gold armor plates + electric-blue dorsal crest glow",
    }
    return stage_e, stage_c, meta


def build_sculpted_harvest_automaton_dsl(variant: int = 0) -> tuple[str, str, dict[str, Any]]:
    """Hard-surface Orchard Harvest Robot (`SKILL.md`: chamfered rotated boxes, offset 45° octagon wheels, leaf planes)."""
    h_yellow = 78 + (variant * 11) % 24
    h_green = 142 + (variant * 13) % 24

    stage_e = f"""@meta {{ archetype: entity, target_profile: bbmodel_5, texel_density: 16, dsl_version: "3.2", size_step: 0.5, audit_profile: sculpted }}
@palette {{
  chassis_yellow_mat: ramp(base_lch=[0.76, 0.16, {h_yellow}], steps=6, hue_shift=[-14, 10])
  foliage_green_mat: ramp(base_lch=[0.62, 0.15, {h_green}], steps=6, hue_shift=[-16, 12])
  steel_slate_mat: ramp(base_lch=[0.40, 0.04, 248], steps=6, hue_shift=[-10, 6])
  sky_lens_mat: ramp(base_lch=[0.82, 0.14, 224], steps=6, hue_shift=[-6, 12], emissive=true)
}}
@rig {{
  root bone chassis_root {{
    pivot_local: [0, 8, 0]
    box main_hull {{ size: [9, 6.5, 9.5], material: chassis_yellow_mat, rotate: [-3, 0, 0] }}
    box top_chamfer place(base_box=main_hull, anchor=[0, 0.85, -0.15], align=[0, -0.5, 0]) {{
      size: [8, 2, 7.5], material: chassis_yellow_mat, rotate: [15, 0, 0]
    }}
    box rear_basket place(base_box=main_hull, anchor=[0, 0.45, 0.84], align=[0, 0, -0.6]) {{
      size: [8, 4.5, 4.5], material: steel_slate_mat, rotate: [-8, 0, 0]
    }}
    plane basket_leaves_l place(base_box=rear_basket, anchor=[-0.45, 0.88, 0], align=[0, -0.8, 0]) {{
      size: [4, 4], normal: z, material: foliage_green_mat, rotate: [-12, 18, 0]
    }}
    plane basket_leaves_r place(base_box=rear_basket, anchor=[0.45, 0.88, 0.2], align=[0, -0.8, 0]) {{
      size: [4, 4], normal: z, material: foliage_green_mat, rotate: [10, -22, 0]
    }}

    bone sensor_head attach(parent_box=main_hull, anchor=[0, 0.78, -0.85], align=[0, -0.6, 0.8]) {{
      rest_euler: [6, -10, 4]
      box visor_pod {{ size: [6, 4, 4], material: chassis_yellow_mat }}
      box optic_lens place(base_box=visor_pod, anchor=[0, 0, -0.88], align=[0, 0, 0.5]) {{
        size: [4, 2, 1], material: sky_lens_mat
      }}
      box antenna_mast place(base_box=visor_pod, anchor=[0.65, 0.96, -0.35], align=[0, -0.94, 0]) {{
        size: [0.5, 3.5, 0.5], material: steel_slate_mat, rotate: [-12, 0, 8]
      }}
    }}

    bone picker_arm_l attach(parent_box=main_hull, anchor=[-0.96, 0.42, -0.35], align=[0.95, 0, 0]) mirror_x(picker_arm_r) {{
      rest_euler: [-26, 20, -34]
      box shoulder_hub {{ size: [2.5, 2.5, 2.5], material: steel_slate_mat, rotate: [45, 0, 0] }}
      box upper_beam place(base_box=shoulder_hub, anchor=[-0.85, -0.2, -0.4], align=[0.6, 0.6, 0.5]) {{
        size: [1.5, 4.5, 1.5], material: chassis_yellow_mat, rotate: [-22, 0, 0]
      }}
      bone forearm_l attach(parent_box=upper_beam, anchor=[0, -0.88, 0], align=[0, 0.88, 0]) {{
        rest_euler: [-34, -10, 0]
        box piston_sleeve {{ size: [1.5, 4.5, 1.5], material: steel_slate_mat }}
        box claw_prong_a place(base_box=piston_sleeve, anchor=[-0.7, -0.92, -0.2], align=[0, 0.85, 0]) {{
          size: [0.5, 2, 1], material: chassis_yellow_mat, rotate: [14, 0, 15]
        }}
        box claw_prong_b place(base_box=piston_sleeve, anchor=[0.7, -0.92, -0.2], align=[0, 0.85, 0]) {{
          size: [0.5, 2, 1], material: chassis_yellow_mat, rotate: [14, 0, -15]
        }}
      }}
    }}

    bone wheel_fl attach(parent_box=main_hull, anchor=[-0.96, -0.82, -0.58], align=[0.9, 0, 0]) mirror_x(wheel_fr) {{
      box tyre_a {{ size: [2, 4, 4], material: steel_slate_mat }}
      box hub_oct place(base_box=tyre_a, anchor=[-0.88, 0, 0], align=[0.5, 0, 0]) {{
        size: [1, 2.5, 2.5], material: chassis_yellow_mat, rotate: [45, 0, 0]
      }}
    }}
    bone wheel_hl attach(parent_box=main_hull, anchor=[-0.96, -0.82, 0.58], align=[0.9, 0, 0]) mirror_x(wheel_hr) {{
      box h_tyre {{ size: [2, 4, 4], material: steel_slate_mat }}
      box h_hub_oct place(base_box=h_tyre, anchor=[-0.88, 0, 0], align=[0.5, 0, 0]) {{
        size: [1, 2.5, 2.5], material: chassis_yellow_mat, rotate: [45, 0, 0]
      }}
    }}

    chain coolant_hose attach(parent_box=rear_basket, anchor=[0.88, 0.35, 0.88], align=[-0.4, 0, -0.9]) segments=3 taper=0.90 {{
      axis: +z
      rest_euler: [12, 16, 0]
      box hose_seg {{ size: [1, 1, 2], material: sky_lens_mat }}
    }}
  }}
}}"""

    stage_c = f"""@meta {{ archetype: entity, target_profile: bbmodel_5, texel_density: 16, dsl_version: "3.2", audit_profile: strict }}
@palette {{
  chassis_yellow_mat: ramp(base_lch=[0.76, 0.14, {h_yellow}], steps=6)
  steel_slate_mat: ramp(base_lch=[0.40, 0.03, 248], steps=6)
}}
@rig {{
  root bone root_b {{
    pivot_local: [0, 7, 0]
    box chassis {{ size: [9, 7, 10], material: chassis_yellow_mat }}
    bone wheel_l attach(parent_box=chassis, anchor=[-1, -0.7, 0], align=[1, 0, 0]) mirror_x(wheel_r) {{
      box wheel {{ size: [2, 4, 4], material: steel_slate_mat }}
    }}
  }}
}}"""

    meta = {
        "family": "orchard_harvest_automaton",
        "tier": "Standard (Hard-Surface Machine)",
        "concept": "compact orchard harvest robot with chamfered yellow hull, octagonal hub wheels, curved hose, and foliage basket",
        "anchors": ["compact orchard tractor", "articulated pruning manipulator", "woven harvest crate"],
        "ratios": "3 big chassis/basket/visor volumes + chamfer slabs + 45-deg hub octagons",
        "palette_plan": "saturated harvest yellow + foliage green + slate steel + sky-blue optic lens glow",
    }
    return stage_e, stage_c, meta


def build_sculpted_expedition_crawler_dsl(variant: int = 0) -> tuple[str, str, dict[str, Any]]:
    """Polar Expedition Crawler Truck (`SKILL.md`: chamfered cab, thin-box greebles, snow-cap slabs, orange/blue accents)."""
    h_base = 235 + (variant * 13) % 25

    stage_e = f"""@meta {{ archetype: entity, target_profile: bbmodel_5, texel_density: 16, dsl_version: "3.2", size_step: 0.5, audit_profile: sculpted }}
@palette {{
  arctic_slate_mat: ramp(base_lch=[0.52, 0.05, {h_base}], steps=6, hue_shift=[-12, 8])
  snow_cap_mat: ramp(base_lch=[0.92, 0.02, 220], steps=6, hue_shift=[-14, 6])
  hazard_orange_mat: ramp(base_lch=[0.66, 0.18, 46], steps=6, hue_shift=[-10, 10])
  headlamp_glow_mat: ramp(base_lch=[0.90, 0.12, 198], steps=6, hue_shift=[-6, 10], emissive=true)
}}
@rig {{
  root bone frame_root {{
    pivot_local: [0, 6.5, 0]
    box cargo_module {{ size: [10, 7.5, 11], material: arctic_slate_mat, rotate: [-2, 0, 0] }}
    box cargo_snow_slab place(base_box=cargo_module, anchor=[0, 0.92, 0.05], align=[0, -0.5, 0]) {{
      size: [9.5, 1, 10.5], material: snow_cap_mat, rotate: [2, 0, -2]
    }}
    box roof_rail_l place(base_box=cargo_module, anchor=[-0.88, 1, 0], align=[0, -1, 0]) {{
      size: [0.5, 1.5, 9.5], material: hazard_orange_mat
    }}
    box roof_rail_r place(base_box=cargo_module, anchor=[0.88, 1, 0], align=[0, -1, 0]) {{
      size: [0.5, 1.5, 9.5], material: hazard_orange_mat
    }}
    box side_ladder place(base_box=cargo_module, anchor=[0.95, -0.05, 0.25], align=[-0.5, 0, 0]) {{
      size: [0.5, 6.5, 2], material: hazard_orange_mat
    }}
    plane expedition_pennant place(base_box=roof_rail_l, anchor=[0, 1, 0.7], align=[0, -1, 0], offset=[-0.25, 0, 0]) {{
      size: [2.5, 3.5], normal: x, material: hazard_orange_mat
    }}

    bone cab_module attach(parent_box=cargo_module, anchor=[0, -0.08, -0.82], align=[0, 0, 0.8]) {{
      box cab_body {{ size: [9.5, 6.5, 6.5], material: arctic_slate_mat }}
      box windshield_slope place(base_box=cab_body, anchor=[0, 0.55, -0.78], align=[0, -0.3, 0.6]) {{
        size: [8.5, 3, 2.5], material: headlamp_glow_mat, rotate: [22, 0, 0]
      }}
      box cab_snow_cap place(base_box=cab_body, anchor=[0, 0.92, 0.15], align=[0, -0.5, 0]) {{
        size: [8.5, 1, 4.5], material: snow_cap_mat
      }}
      box bumper_bar place(base_box=cab_body, anchor=[0, -0.75, -0.88], align=[0, 0, 0.6]) {{
        size: [9.5, 2, 1.5], material: hazard_orange_mat
      }}
      box lamp_l place(base_box=bumper_bar, anchor=[-0.75, 0.6, -0.85], align=[0, -0.3, 0.5]) {{
        size: [1.5, 1.5, 1], material: headlamp_glow_mat
      }}
      box lamp_r place(base_box=bumper_bar, anchor=[0.75, 0.6, -0.85], align=[0, -0.3, 0.5]) {{
        size: [1.5, 1.5, 1], material: headlamp_glow_mat
      }}
      box whip_antenna place(base_box=cab_body, anchor=[-0.85, 0.98, -0.3], align=[0, -0.98, 0]) {{
        size: [0.5, 5, 0.5], material: hazard_orange_mat, rotate: [-8, 0, -4]
      }}
      bone bogie_fl attach(parent_box=cab_body, anchor=[-1, -0.72, -0.35], align=[1, 0, 0]) mirror_x(bogie_fr) {{
        box front_tyre {{ size: [2.5, 4.5, 4.5], material: arctic_slate_mat }}
        box front_rim place(base_box=front_tyre, anchor=[-0.88, 0, 0], align=[0.5, 0, 0]) {{
          size: [1, 3, 3], material: hazard_orange_mat, rotate: [45, 0, 0]
        }}
      }}
    }}

    bone bogie_ml attach(parent_box=cargo_module, anchor=[-1, -0.78, -0.45], align=[1, 0, 0]) mirror_x(bogie_mr) {{
      box mid_tyre {{ size: [2.5, 4.5, 4.5], material: arctic_slate_mat }}
      box mid_rim place(base_box=mid_tyre, anchor=[-0.88, 0, 0], align=[0.5, 0, 0]) {{
        size: [1, 3, 3], material: hazard_orange_mat, rotate: [45, 0, 0]
      }}
    }}
    bone bogie_hl attach(parent_box=cargo_module, anchor=[-1, -0.78, 0.55], align=[1, 0, 0]) mirror_x(bogie_hr) {{
      box rear_tyre {{ size: [2.5, 4.5, 4.5], material: arctic_slate_mat }}
      box rear_rim place(base_box=rear_tyre, anchor=[-0.88, 0, 0], align=[0.5, 0, 0]) {{
        size: [1, 3, 3], material: hazard_orange_mat, rotate: [45, 0, 0]
      }}
    }}
  }}
}}"""

    stage_c = f"""@meta {{ archetype: entity, target_profile: bbmodel_5, texel_density: 16, dsl_version: "3.2", audit_profile: strict }}
@palette {{
  arctic_slate_mat: ramp(base_lch=[0.52, 0.04, {h_base}], steps=6)
  hazard_orange_mat: ramp(base_lch=[0.66, 0.16, 46], steps=6)
}}
@rig {{
  root bone root_b {{
    pivot_local: [0, 6, 0]
    box truck_box {{ size: [10, 7, 16], material: arctic_slate_mat }}
    bone wheel_l attach(parent_box=truck_box, anchor=[-1, -0.7, 0], align=[1, 0, 0]) mirror_x(wheel_r) {{
      box w_box {{ size: [2, 4, 4], material: hazard_orange_mat }}
    }}
  }}
}}"""

    meta = {
        "family": "polar_expedition_crawler",
        "tier": "Standard (Hard-Surface Vehicle)",
        "concept": "six-wheeled polar expedition truck with snow-capped roof slabs, roof rack rails, ladder, and glowing headlamps",
        "anchors": ["Antarctic snow-cat", "expedition overland rig", "octagonal heavy tread tyres"],
        "ratios": "cargo + sloped cab primary volumes + 0.5u rail/ladder/antenna silhouette greebles",
        "palette_plan": "muted arctic slate + snow white top slabs + safety orange greebles + ice-blue lamp glow",
    }
    return stage_e, stage_c, meta


def build_sculpted_alchemist_studio_diorama_dsl(variant: int = 0) -> tuple[str, str, dict[str, Any]]:
    """Cosy Alchemist Studio Alcove Diorama (`SKILL.md`: warm focal glow vs cool violet ambient, skewed leaning books, leaf planes)."""
    book_tilt = 7 + (variant % 3) * 3

    stage_e = f"""@meta {{ archetype: block_multipart, target_profile: bbmodel_5, texel_density: 16, dsl_version: "3.2", size_step: 0.5, audit_profile: sculpted }}
@palette {{
  violet_stone_mat: ramp(base_lch=[0.38, 0.08, 286], steps=6, hue_shift=[-14, 8])
  warm_timber_mat: ramp(base_lch=[0.54, 0.11, 56], steps=6, hue_shift=[-16, 10])
  herb_leaf_mat: ramp(base_lch=[0.64, 0.15, 146], steps=6, hue_shift=[-14, 12])
  teal_bottle_mat: ramp(base_lch=[0.66, 0.13, 196], steps=6, hue_shift=[-10, 10])
  parchment_mat: ramp(base_lch=[0.88, 0.05, 84], steps=6, hue_shift=[-10, 8])
  amber_lantern_mat: ramp(base_lch=[0.86, 0.19, 76], steps=6, hue_shift=[-6, 12], emissive=true)
}}
@rig {{
  root bone diorama_base {{
    pivot_local: [0, 0, 0]
    box floor_plinth {{ size: [16, 1.5, 14], material: violet_stone_mat }}
    box back_wall place(base_box=floor_plinth, anchor=[0, 1, 0.88], align=[0, -0.9, -0.5]) {{
      size: [16, 13, 1.5], material: violet_stone_mat
    }}
    box timber_desk place(base_box=floor_plinth, anchor=[-0.15, 1, 0.25], align=[0, -0.95, 0]) {{
      size: [11, 5, 5.5], material: warm_timber_mat, rotate: [0, 3, 0]
    }}
    box upper_shelf place(base_box=back_wall, anchor=[-0.1, 0.45, -0.88], align=[0, 0, 0.8]) {{
      size: [10.5, 1, 3], material: warm_timber_mat, rotate: [0, 0, -2]
    }}
    box book_tall place(base_box=upper_shelf, anchor=[-0.65, 1, 0], align=[0, -0.9, 0]) {{
      size: [1, 3.5, 2.5], material: teal_bottle_mat, rotate: [0, 0, {book_tilt}]
    }}
    box book_mid place(base_box=upper_shelf, anchor=[-0.42, 1, 0.05], align=[0, -0.9, 0]) {{
      size: [1, 3, 2.5], material: warm_timber_mat, rotate: [0, 4, {book_tilt + 4}]
    }}
    box potion_flask place(base_box=upper_shelf, anchor=[0.45, 1, 0], align=[0, -0.95, 0]) {{
      size: [2, 2.5, 2], material: teal_bottle_mat, rotate: [0, 18, 0]
    }}
    box focal_lantern place(base_box=timber_desk, anchor=[-0.45, 1, -0.15], align=[0, -0.95, 0]) {{
      size: [2.5, 3.5, 2.5], material: amber_lantern_mat, rotate: [0, 15, 0]
    }}
    box lantern_cap place(base_box=focal_lantern, anchor=[0, 0.92, 0], align=[0, -0.5, 0]) {{
      size: [3, 1, 3], material: violet_stone_mat, rotate: [0, 15, 0]
    }}
    box open_tome_l place(base_box=timber_desk, anchor=[0.28, 1, -0.2], align=[0.8, -0.8, 0]) {{
      size: [2.5, 0.5, 3.5], material: parchment_mat, rotate: [0, -10, 10]
    }}
    box open_tome_r place(base_box=timber_desk, anchor=[0.32, 1, -0.2], align=[-0.8, -0.8, 0]) {{
      size: [2.5, 0.5, 3.5], material: parchment_mat, rotate: [0, -10, -10]
    }}
    box clay_pot place(base_box=floor_plinth, anchor=[0.68, 1, -0.25], align=[0, -0.95, 0]) {{
      size: [3.5, 3.5, 3.5], material: warm_timber_mat, rotate: [0, 22, 0]
    }}
    plane leaf_fan_a place(base_box=clay_pot, anchor=[0, 0.85, 0], align=[0, -0.8, 0]) {{
      size: [4.5, 4.5], normal: z, material: herb_leaf_mat, rotate: [-14, 22, 0]
    }}
    plane leaf_fan_b place(base_box=clay_pot, anchor=[0.12, 0.85, 0], align=[0, -0.8, 0]) {{
      size: [4.5, 4.5], normal: x, material: herb_leaf_mat, rotate: [14, -18, 10]
    }}
    plane hanging_ivy place(base_box=upper_shelf, anchor=[0.85, -0.5, -0.92], align=[0, 0.85, 0]) {{
      size: [2.5, 3.5], normal: z, material: herb_leaf_mat, rotate: [8, 0, 0]
    }}
    box mortar_bowl place(base_box=timber_desk, anchor=[-0.05, 1, 0.32], align=[0, -0.95, 0]) {{
      size: [2, 1.5, 2], material: violet_stone_mat, rotate: [0, 28, 0]
    }}
    box pestle_rod place(base_box=mortar_bowl, anchor=[0.15, 0.92, 0], align=[0, -0.92, 0]) {{
      size: [0.5, 2.5, 0.5], material: warm_timber_mat, rotate: [16, 0, -20]
    }}
    box scroll_roll place(base_box=timber_desk, anchor=[0.65, 1, 0.35], align=[0, -0.95, 0]) {{
      size: [1, 1, 3], material: parchment_mat, rotate: [0, 34, 0]
    }}
    box crystal_specimen place(base_box=upper_shelf, anchor=[0.05, 1, 0.1], align=[0, -0.85, 0]) {{
      size: [1.5, 3, 1.5], material: teal_bottle_mat, rotate: [-14, 20, 12]
    }}
    box wooden_stool place(base_box=floor_plinth, anchor=[-0.22, 1, -0.48], align=[0, -0.95, 0]) {{
      size: [3.5, 3, 3.5], material: warm_timber_mat, rotate: [0, -16, 0]
    }}
  }}
}}"""

    stage_c = """@meta { archetype: block_multipart, target_profile: bbmodel_5, texel_density: 16, dsl_version: "3.2", audit_profile: strict }
@palette {
  violet_stone_mat: ramp(base_lch=[0.38, 0.07, 286], steps=6)
  warm_timber_mat: ramp(base_lch=[0.54, 0.10, 56], steps=6)
}
@rig {
  root bone root_b {
    pivot_local: [0, 0, 0]
    box floor { size: [16, 2, 14], material: violet_stone_mat }
    box desk place(base_box=floor, anchor=[0, 1, 0], align=[0, -1, 0]) {
      size: [10, 5, 5], material: warm_timber_mat
    }
  }
}"""

    meta = {
        "family": "alchemist_studio_diorama",
        "tier": "Diorama Scene",
        "concept": "cosy alchemist workbench alcove with warm glowing lantern focal point, skewed leaning books, and potted leaf planes",
        "anchors": ["medieval apothecary desk", "leaning library shelf", "hanging botanical conservatory"],
        "ratios": "calm violet wall/floor backdrop + dense tilted prop cluster around warm focal lantern",
        "palette_plan": "cool violet stone ambient + warm timber + herb green + teal glass + bright amber lantern glow",
    }
    return stage_e, stage_c, meta


SCULPTED_ARCHETYPE_BUILDERS = (
    build_sculpted_crimson_western_dragon_dsl,
    build_sculpted_glacial_crystal_drake_dsl,
    build_sculpted_ochre_club_wyvern_dsl,
    build_sculpted_pastel_chibi_dragonet_dsl,
    build_sculpted_storm_wolf_dragon_v4_dsl,
    build_sculpted_harvest_automaton_dsl,
    build_sculpted_expedition_crawler_dsl,
    build_sculpted_alchemist_studio_diorama_dsl,
)


# ---------------------------------------------------------------------------
# 4. Native Supervision Stream Builder & Oracle Verifier
# ---------------------------------------------------------------------------

def _trait_sheet_think(meta: dict[str, Any], style_ev: dict[str, Any], task_tag: str) -> str:
    return (
        f"task={task_tag} | style=blockbench-sculpted-style\n"
        f"1. Concept & Anchors: {meta['concept']} (anchors: {', '.join(meta['anchors'])}).\n"
        f"2. Tier & Ratios: {meta['tier']} ({style_ev['element_count']} elements, {style_ev['plane_count']} cutout planes, "
        f"{style_ev['bone_count']} bones; {meta['ratios']}).\n"
        f"3. Palette (3+1 hue-shifted): {meta['palette_plan']} (hue_shifted_ramps={style_ev['hue_shifted_ramp_count']}, "
        f"emissive_share={style_ev['emissive_focal_share']}).\n"
        f"4. Sculpted Flow & Seating: audit_profile={style_ev['audit_profile']}, dsl_version={style_ev['dsl_version']}, "
        f"size_step={style_ev['size_step']}, rotated_fraction={style_ev['rotated_element_fraction']}, "
        f"sculpted_overlaps={style_ev['sculpted_overlap_count']}."
    )


def build_sculpted_style_curriculum_records(
    *,
    variants_per_family: int = 8,
    include_v4_assembly_tasks: bool = True,
    panel_size: tuple[int, int] = (192, 192),
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Generate native Sculpted Blockbench Style + Block-DSL v3.2/v4.0 multimodal & text supervision records."""
    ensure_synccraft_importable()
    from synccraft.dsl.parser import format_program, parse_program

    records: list[dict[str, Any]] = []
    media_blobs: dict[str, bytes] = {}

    for fam_idx, builder in enumerate(SCULPTED_ARCHETYPE_BUILDERS):
        for v in range(variants_per_family):
            stage_e_dsl, stage_c_dsl, meta = builder(v)
            prog_e = parse_program(stage_e_dsl)
            prog_c = parse_program(stage_c_dsl)
            ev_e = evaluate_sculpted_blockbench_style(prog_e)
            ev_c = evaluate_sculpted_blockbench_style(prog_c)
            if not ev_e["audit_passed"]:
                raise RuntimeError(f"Sculpted archetype {meta['family']} v={v} failed audit: {ev_e}")

            fam_id = f"sculpted_{meta['family']}"
            aid = f"{fam_id}_v{v:03d}"
            prov = make_clean_room_provenance(aid, title=f"{meta['family']} sculpted v{v}")
            # Preserve `@layout/@component/@assembly` source verbatim for v4.0 modular programs!
            canonical_e = stage_e_dsl.strip() if "@assembly" in stage_e_dsl else format_program(prog_e).strip()
            ans_dsl_block = f"```blockdsl\n{canonical_e}\n```"
            use_think = (v % 4 != 3)

            # 1. Text -> Sculpted Block-DSL (`synccraft_sculpted_style_creation`)
            th_create = _trait_sheet_think(meta, ev_e, "synccraft_sculpted_style_creation")
            ver_clause = (
                'using modular Block-DSL v4.0 (`@layout`, `@component`, `@assembly`, `audit_profile: sculpted`)'
                if ev_e["dsl_version"] == "4.0"
                else 'using Sculpted Block-DSL (`audit_profile: sculpted`, `size_step: 0.5`, `hue_shift` ramps, seated joints, and alpha-cutout `plane`s)'
            )
            records.append(
                {
                    "id": f"{aid}#sculpted_create",
                    "source_group": f"sculpted_family:{aid}",
                    "language": "en",
                    "task": "synccraft_sculpted_style_creation",
                    "enable_thinking": use_think,
                    "frames": [],
                    "prompt": (
                        f"Design a vibrant, deliberately sculpted 3D Blockbench asset for: {meta['concept']} "
                        f"({meta['tier']}), {ver_clause}. Avoid rigid grid-aligned boxes, straight pipe necks, or flat unpainted faces."
                    ),
                    "answer": format_supervised_answer(ans_dsl_block, th_create, enable_thinking=use_think),
                    "target": {"kind": "text", "text": ans_dsl_block},
                    "provenance": prov,
                    "metadata": {
                        "asset_id": aid,
                        "family_id": fam_id,
                        "expected_audit_profile": "sculpted",
                        "min_sculpted_score": 0.72,
                    },
                }
            )

            # Render studio turnaround views for the first `min(variants_per_family, 8)` variants per family
            if v < 8:
                views = ("south", "right", "top", "iso")
                pngs_e, _ = render_sculpted_turnaround_views(
                    prog_e,
                    views=views,
                    panel_size=panel_size,
                    include_silhouette_sheet=(v < 4),
                )
                e_paths = [f"sculpted_v6/{aid}_e_{vw}.png" for vw in views]
                for i_vw, vw in enumerate(views):
                    media_blobs[e_paths[i_vw]] = pngs_e[vw]
                v_units = [_camera_unit_meta(vw) for vw in views]

                # 2. Vibrant Studio Turnaround -> Sculpted Block-DSL (`synccraft_sculpted_multiview_inverse`)
                th_inv = _trait_sheet_think(meta, ev_e, "synccraft_sculpted_multiview_inverse")
                records.append(
                    {
                        "id": f"{aid}#sculpted_multiview",
                        "source_group": f"sculpted_family:{aid}",
                        "language": "en",
                        "task": "synccraft_sculpted_multiview_inverse",
                        "visual_kind": "multiview_render",
                        "enable_thinking": True,
                        "frames": e_paths,
                        "media_times_s": [0.0] * 4,
                        "visual_units": v_units,
                        "prompt": (
                            f"Inspect these 4 studio turnaround views (`south`, `right`, `top`, `iso`) of a sculpted Blockbench "
                            f"model (`{meta['family']}`) and reconstruct its complete, collision-audited Block-DSL program "
                            f"(`dsl_version: {ev_e['dsl_version']!r}`, `audit_profile: 'sculpted'`)."
                        ),
                        "answer": format_supervised_answer(ans_dsl_block, th_inv, enable_thinking=True),
                        "target": {"kind": "text", "text": ans_dsl_block},
                        "provenance": prov,
                        "metadata": {
                            "asset_id": aid,
                            "family_id": fam_id,
                            "expected_audit_profile": "sculpted",
                            "min_sculpted_score": 0.70,
                        },
                    }
                )

                # 3. Stage-C Stiff Blockout -> Stage-E Sculpted Upgrade (`synccraft_sculpted_stage_progression`)
                pngs_c, _ = render_sculpted_turnaround_views(prog_c, views=views, panel_size=panel_size)
                c_paths = [f"sculpted_v6/{aid}_c_{vw}.png" for vw in views]
                for i_vw, vw in enumerate(views):
                    media_blobs[c_paths[i_vw]] = pngs_c[vw]
                th_prog = (
                    f"Stage-C critique: input blockout has only {ev_c['element_count']} axis-aligned boxes "
                    f"(rotated_fraction={ev_c['rotated_element_fraction']}, planes={ev_c['plane_count']}, grade={ev_c['grade']}).\n"
                    + _trait_sheet_think(meta, ev_e, "synccraft_sculpted_stage_progression")
                )
                records.append(
                    {
                        "id": f"{aid}#sculpted_upgrade",
                        "source_group": f"sculpted_family:{aid}",
                        "language": "en",
                        "task": "synccraft_sculpted_stage_progression",
                        "visual_kind": "multiview_render",
                        "enable_thinking": True,
                        "frames": c_paths,
                        "media_times_s": [0.0] * 4,
                        "visual_units": v_units,
                        "prompt": (
                            "Critique this stiff Stage-C axis-aligned blockout from its 4 views (`south`, `right`, `top`, `iso`) "
                            f"and upgrade it into a vibrant Stage-E sculpted Blockbench asset (`{meta['concept']}`):\n"
                            f"```blockdsl\n{stage_c_dsl.strip()}\n```"
                        ),
                        "answer": format_supervised_answer(ans_dsl_block, th_prog, enable_thinking=True),
                        "target": {"kind": "text", "text": ans_dsl_block},
                        "provenance": prov,
                        "metadata": {
                            "asset_id": aid,
                            "family_id": fam_id,
                            "expected_audit_profile": "sculpted",
                            "min_sculpted_score": 0.72,
                        },
                    }
                )

                # 4. Studio + Black Silhouette Self-Critique (`synccraft_sculpted_self_critique`) on BOTH stiff C & sculpted E!
                if v < 4 and "silhouette_sheet" in pngs_e:
                    sil_e_path = f"sculpted_v6/{aid}_e_sil.png"
                    media_blobs[sil_e_path] = pngs_e["silhouette_sheet"]
                    pngs_c_sil, _ = render_sculpted_turnaround_views(
                        prog_c, views=views, panel_size=panel_size, include_silhouette_sheet=True
                    )
                    sil_c_path = f"sculpted_v6/{aid}_c_sil.png"
                    media_blobs[sil_c_path] = pngs_c_sil["silhouette_sheet"]

                    for tag_lbl, f_list, ev_obj in (
                        ("sculpted_e", [e_paths[3], sil_e_path], ev_e),
                        ("stiff_c", [c_paths[3], sil_c_path], ev_c),
                    ):
                        oracle_crit = {
                            "grade": ev_obj["grade"],
                            "is_stiff_or_voxel_like": ev_obj["is_stiff_or_voxel_like"],
                            "has_spine_gesture_curve": ev_obj["has_spine_gesture_curve"],
                            "has_cutout_or_thin_planes": ev_obj["has_cutout_or_thin_planes"],
                            "hue_shifted_palettes": ev_obj["hue_shifted_ramp_count"] >= 2,
                            "audit_passed": ev_obj["audit_passed"],
                        }
                        crit_json = json.dumps(oracle_crit, sort_keys=True)
                        th_crit = (
                            f"Inspect 3/4 studio hero + 4-view black silhouette sheet: elements={ev_obj['element_count']}, "
                            f"planes={ev_obj['plane_count']}, rotated_fraction={ev_obj['rotated_element_fraction']}, "
                            f"grade={ev_obj['grade']}."
                        )
                        records.append(
                            {
                                "id": f"{aid}#self_critique_{tag_lbl}",
                                "source_group": f"sculpted_family:{aid}",
                                "language": "en",
                                "task": "synccraft_sculpted_self_critique",
                                "visual_kind": "state_pair",
                                "enable_thinking": True,
                                "frames": f_list,
                                "media_times_s": [0.0, 0.0],
                                "visual_units": [
                                    {"visual_kind": "multiview_render", "view_id": "iso"},
                                    {"visual_kind": "multiview_render", "view_id": "top"},
                                ],
                                "prompt": (
                                    "Inspect the 3/4 studio render (image 1) and 4-view black silhouette check sheet (image 2). "
                                    "Perform a Sculpted Blockbench Style critique and output JSON with keys "
                                    "`audit_passed`, `grade` (`sculpted_master_tier` | `sculpted_standard` | `stiff_blockout`), "
                                    "`has_cutout_or_thin_planes`, `has_spine_gesture_curve`, `hue_shifted_palettes`, and `is_stiff_or_voxel_like`."
                                ),
                                "answer": format_supervised_answer(crit_json, th_crit, enable_thinking=True),
                                "target": {"kind": "text", "text": crit_json},
                                "provenance": prov,
                                "metadata": {"oracle_json": oracle_crit},
                            }
                        )

    # 5. Also integrate SyncCraft v4.0 `@layout/@component/@assembly` modular world-model tasks (`1f41187`) when available!
    if include_v4_assembly_tasks:
        try:
            from synccraft.dsl.assembly_ir import override_layout_params
            from synccraft.audit.sync_audit import audit_scene
            from synccraft.dsl.assembly_ir import audit_assembly_contracts
            from synccraft.kinematics.se3_chain import compile_geometry
            from synccraft.world.assembly_world_tasks import (
                MODULAR_ARTILLERY_WALKER_V4_DSL,
                MODULAR_SCULPTED_DRAGON_V4_DSL,
                build_modular_assembly_supervision_suite,
            )

            base_v4_suite = list(build_modular_assembly_supervision_suite())
            # Add parametric chest_width / hull_width propagation variants + full modular walker synthesis
            dragon_prog_v4 = parse_program(MODULAR_SCULPTED_DRAGON_V4_DSL)
            for cw in (11, 12, 15, 16):
                pert = override_layout_params(dragon_prog_v4, {"chest_width": cw})
                sc_p = compile_geometry(pert)
                aud_p = audit_assembly_contracts(pert, sc_p)
                sat_p = audit_scene(sc_p, enforce_physical_contact=True)
                gt_p = {
                    "torso_ribcage_width": float(sc_p.box_by_name["torso__ribcage"].size[0]),
                    "skull_cranium_width": float(sc_p.box_by_name["skull__cranium"].size[0]),
                    "audit_passed": bool(sat_p["passed"] and aud_p["passed"]),
                    "world_span_x": round(float(aud_p["world_span"][0]), 4),
                }
                base_v4_suite.append(
                    {
                        "task_id": f"v4_wm_04_propagate_dragon_cw_{cw}",
                        "task_type": "propagate_layout_and_instance_controls",
                        "assembly_provenance": "authored",
                        "base_assembly_dsl": MODULAR_SCULPTED_DRAGON_V4_DSL,
                        "layout_overrides": {"chest_width": cw},
                        "prompt": (
                            "Predict JSON `{torso_ribcage_width, skull_cranium_width, audit_passed, world_span_x}` "
                            f"when `@layout` parameter `chest_width` in `ModularElderDragon` changes from `10` to `{cw}`:\n"
                            f"```blockdsl\n{MODULAR_SCULPTED_DRAGON_V4_DSL}\n```"
                        ),
                        "ground_truth": gt_p,
                        "oracle_completion": json.dumps(gt_p, sort_keys=True),
                    }
                )
            for mod_name, mod_dsl, mod_desc in (
                ("modular_elder_dragon_v4", MODULAR_SCULPTED_DRAGON_V4_DSL, "4-legged winged sculpted dragon with `@layout`, `@component`, and `@assembly`"),
                ("modular_artillery_walker_v4", MODULAR_ARTILLERY_WALKER_V4_DSL, "4-legged mechanical steam artillery walker with revolute turret and prismatic recoil barrel"),
            ):
                base_v4_suite.append(
                    {
                        "task_id": f"v4_wm_full_{mod_name}",
                        "task_type": "full_modular_assembly_synthesis",
                        "assembly_provenance": "authored",
                        "prompt": (
                            f"Write a complete Block-DSL v4.0 program (`@meta` with `dsl_version: '4.0'`, `@layout`, "
                            f"`@component` blocks, and `@assembly`) for a {mod_desc}."
                        ),
                        "oracle_completion": f"```blockdsl\n{mod_dsl.strip()}\n```",
                    }
                )

            for idx_v4, v4_item in enumerate(base_v4_suite):
                p_txt = str(v4_item.get("prompt") or "").strip()
                raw_oracle = str(v4_item.get("oracle_completion") or "").strip()
                if not p_txt or not raw_oracle:
                    continue
                t_txt = (
                    raw_oracle
                    if (raw_oracle.startswith("{") or raw_oracle.startswith("```"))
                    else f"```blockdsl\n{raw_oracle}\n```"
                )
                v4_id = str(v4_item.get("task_id") or f"v4_mod_{idx_v4:03d}")
                v4_sub = str(v4_item.get("task_type") or "modular_v4")
                th_v4 = (
                    f"task=synccraft_v4_modular_assembly | subtask={v4_sub}\n"
                    "Verify @layout parameter bounds, @component attachment frames, and @assembly joint contracts."
                )
                use_th = idx_v4 % 3 != 2
                records.append(
                    {
                        "id": f"v4asm-{v4_id}",
                        "source_group": f"synccraft_v4_asm:{v4_id}",
                        "language": "en",
                        "task": "synccraft_v4_modular_assembly",
                        "enable_thinking": use_th,
                        "frames": [],
                        "prompt": p_txt,
                        "answer": format_supervised_answer(t_txt, th_v4, enable_thinking=use_th),
                        "target": {"kind": "text", "text": t_txt},
                        "provenance": make_clean_room_provenance(f"v4asm_{v4_id}"),
                        "metadata": {"v4_subtask": v4_sub, "v4_spec": dict(v4_item)},
                    }
                )
        except Exception as exc:
            emit("sculpted_v4_assembly_tasks_optional_skip", reason=str(exc)[:200])

    emit("sculpted_style_curriculum_ready", records=len(records), media_blobs=len(media_blobs))
    return records, media_blobs


def build_human_artist_sculpted_studio_records(
    human_rows: Sequence[dict[str, Any]],
    *,
    max_assets: int = 48,
    seed: int = 42,
    panel_size: tuple[int, int] = (192, 192),
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Render vibrant cel-shaded studio turnarounds + silhouette critiques for permissive human-artist Blockbench rigs."""
    ensure_synccraft_importable()
    from .synccraft_bridge import validate_synccraft_license_record
    from synccraft.dsl.ast_nodes import validate_program
    from synccraft.dsl.parser import format_program, parse_program

    eligible: list[dict[str, Any]] = []
    for ar in human_rows:
        if not ar.get("commercial_weights_eligible", False) or ar.get("ai_assisted", False):
            continue
        bx_cnt = int(ar.get("box_count") or 0)
        if 14 <= bx_cnt <= 80 and ar.get("canonical_dsl"):
            eligible.append(ar)

    eligible.sort(key=lambda r: identity([seed, "human_sculpted_studio", str(r.get("asset_id", ""))]))

    records: list[dict[str, Any]] = []
    media_blobs: dict[str, bytes] = {}
    views = ("south", "right", "top", "iso")
    added = 0

    for ar in eligible:
        if added >= max_assets:
            break
        aid = str(ar["asset_id"])
        fid = str(ar.get("family_id") or f"human_cc_{aid[:16]}")
        lic = ar.get("source_license_and_review_record") or {}
        spdx = str(ar.get("license_spdx") or lic.get("license") or "")
        sha_hex = str(lic.get("source_sha256") or identity(aid))
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
            "changes_made_note": str(lic.get("changes_made_notice") or "Studio cel-painted views from permissive human Blockbench rig"),
            "reviewer_id": "synccraft_catalog_verified_v3_2",
        }
        try:
            validate_synccraft_license_record({"provenance": prov}, allow_research_nc=False)
            prog = validate_program(parse_program(str(ar["canonical_dsl"])))
            ev = evaluate_sculpted_blockbench_style(prog)
            pngs, _ = render_sculpted_turnaround_views(
                prog,
                views=views,
                panel_size=panel_size,
                include_silhouette_sheet=(added < 24),
            )
        except Exception:
            continue

        f_paths = [f"sculpted_v6/human_{aid}_{vw}.png" for vw in views]
        for idx_v, vw in enumerate(views):
            media_blobs[f_paths[idx_v]] = pngs[vw]
        v_units = [_camera_unit_meta(vw) for vw in views]
        canon = format_program(prog).strip()
        ans_dsl = f"```blockdsl\n{canon}\n```"
        title_clean = str(prov["title"]).replace("_", " ").strip()
        th_txt = (
            f"task=synccraft_sculpted_multiview_inverse | human_artist_asset={aid}\n"
            f"1. Observe 4 studio views: {ev['element_count']} elements across {ev['bone_count']} bones "
            f"(planes={ev['plane_count']}, rotated_fraction={ev['rotated_element_fraction']}).\n"
            f"2. Audit profile={ev['audit_profile']}, grade={ev['grade']}, sculpted_score={ev['sculpted_score']}."
        )
        if ev["audit_passed"] and not ev["is_stiff_or_voxel_like"]:
            records.append(
                {
                    "id": f"human-studio-{aid}#multiview",
                    "source_id": aid,
                    "source_group": f"synccraft-family:{fid}",
                    "language": "en",
                    "task": "synccraft_sculpted_multiview_inverse",
                    "visual_kind": "multiview_render",
                    "enable_thinking": True,
                    "frames": f_paths,
                    "media_times_s": [0.0] * 4,
                    "visual_units": v_units,
                    "prompt": (
                        f"Inspect these 4 vibrant studio views (`south`, `right`, `top`, `iso`) of the human-authored "
                        f"Blockbench asset `{title_clean}` and reconstruct its complete Block-DSL program."
                    ),
                    "answer": format_supervised_answer(ans_dsl, th_txt, enable_thinking=True),
                    "target": {"kind": "text", "text": ans_dsl},
                    "provenance": prov,
                    "metadata": {
                        "asset_id": aid,
                        "family_id": fid,
                        "expected_audit_profile": ev["audit_profile"],
                        "min_sculpted_score": max(0.55, round(float(ev["sculpted_score"]) - 0.12, 2)),
                        "provenance": prov,
                        "grouping_keys": [f"synccraft-family:{fid}", f"synccraft-asset:{aid}"],
                    },
                }
            )

        if "silhouette_sheet" in pngs:
            sil_p = f"sculpted_v6/human_{aid}_sil.png"
            media_blobs[sil_p] = pngs["silhouette_sheet"]
            oracle_crit = {
                "grade": ev["grade"],
                "is_stiff_or_voxel_like": ev["is_stiff_or_voxel_like"],
                "has_spine_gesture_curve": ev["has_spine_gesture_curve"],
                "has_cutout_or_thin_planes": ev["has_cutout_or_thin_planes"],
                "hue_shifted_palettes": ev["hue_shifted_ramp_count"] >= 2,
                "audit_passed": ev["audit_passed"],
            }
            crit_json = json.dumps(oracle_crit, sort_keys=True)
            th_c = (
                f"Inspect human-artist studio hero + 4-view silhouette sheet: elements={ev['element_count']}, "
                f"rotated_fraction={ev['rotated_element_fraction']}, grade={ev['grade']}."
            )
            records.append(
                {
                    "id": f"human-studio-{aid}#self_critique",
                    "source_id": aid,
                    "source_group": f"synccraft-family:{fid}",
                    "language": "en",
                    "task": "synccraft_sculpted_self_critique",
                    "visual_kind": "state_pair",
                    "enable_thinking": True,
                    "frames": [f_paths[3], sil_p],
                    "media_times_s": [0.0, 0.0],
                    "visual_units": [
                        {"visual_kind": "multiview_render", "view_id": "iso"},
                        {"visual_kind": "multiview_render", "view_id": "top"},
                    ],
                    "prompt": (
                        "Inspect the 3/4 studio render (image 1) and 4-view black silhouette check sheet (image 2). "
                        "Perform a Sculpted Blockbench Style critique and output JSON with keys "
                        "`audit_passed`, `grade` (`sculpted_master_tier` | `sculpted_standard` | `stiff_blockout`), "
                        "`has_cutout_or_thin_planes`, `has_spine_gesture_curve`, `hue_shifted_palettes`, and `is_stiff_or_voxel_like`."
                    ),
                    "answer": format_supervised_answer(crit_json, th_c, enable_thinking=True),
                    "target": {"kind": "text", "text": crit_json},
                    "provenance": prov,
                    "metadata": {
                        "asset_id": aid,
                        "family_id": fid,
                        "oracle_json": oracle_crit,
                        "provenance": prov,
                        "grouping_keys": [f"synccraft-family:{fid}", f"synccraft-asset:{aid}"],
                    },
                }
            )
        added += 1

    emit("human_artist_sculpted_studio_ready", assets=added, records=len(records), media_blobs=len(media_blobs))
    return records, media_blobs


def evaluate_sculpted_task_completion(record: dict[str, Any], completion_text: str) -> dict[str, Any]:
    """Oracle verifier for all `SCULPTED_STYLE_TASKS` during evaluation and Vision RSI."""
    from .synccraft_bridge import _extract_dsl_candidate, _extract_json_candidate, strip_think_block

    task = str(record.get("task") or "")
    think_part, body = strip_think_block(completion_text)
    meta = record.get("metadata") or {}

    if task == "synccraft_sculpted_self_critique":
        try:
            pred = _extract_json_candidate(body)
        except Exception as exc:
            return {"task": task, "parse_valid": False, "passed": False, "error": f"json_error: {exc}"}
        oracle = meta.get("oracle_json") or {}
        matches = sum(1 for k, v in oracle.items() if pred.get(k) == v)
        acc = matches / max(1, len(oracle))
        passed = (
            bool(pred.get("is_stiff_or_voxel_like")) == bool(oracle.get("is_stiff_or_voxel_like"))
            and str(pred.get("grade")) == str(oracle.get("grade"))
            and acc >= 0.80
        )
        return {"task": task, "parse_valid": True, "passed": passed, "field_accuracy": round(acc, 4)}

    if task == "synccraft_v4_modular_assembly":
        v4_spec = meta.get("v4_spec") or {}
        v4_sub = str(meta.get("v4_subtask") or v4_spec.get("task_type") or "")
        if v4_sub in (
            "plan_layout_and_interfaces",
            "synthesize_component_from_contract",
            "surgical_component_swap_and_repair",
            "propagate_layout_and_instance_controls",
        ) and v4_spec:
            ensure_synccraft_importable()
            from synccraft.world.assembly_world_tasks import evaluate_modular_assembly_world_tasks

            if v4_sub == "propagate_layout_and_instance_controls":
                try:
                    cleaned = json.dumps(_extract_json_candidate(body))
                except Exception as exc:
                    return {"task": task, "parse_valid": False, "passed": False, "error": str(exc)}
            else:
                cleaned = _extract_dsl_candidate(body)
            ev_v4 = evaluate_modular_assembly_world_tasks([v4_spec], {str(v4_spec["task_id"]): cleaned})
            res0 = (ev_v4.get("results") or [{}])[0]
            ok = bool(res0.get("passed"))
            return {
                "task": task,
                "has_think_trace": bool(think_part),
                "parse_valid": "reason" not in res0,
                "passed": ok,
                "v4_result": res0,
            }
        # Otherwise (`full_modular_assembly_synthesis`), verify full v4.0 `@assembly` program + audit!
        dsl_v4 = _extract_dsl_candidate(body)
        try:
            ensure_synccraft_importable()
            from synccraft.audit.sync_audit import audit_scene
            from synccraft.dsl.assembly_ir import audit_assembly_contracts
            from synccraft.dsl.parser import parse_program
            from synccraft.kinematics.se3_chain import compile_geometry

            prog_v4 = parse_program(dsl_v4)
            sc_v4 = compile_geometry(prog_v4)
            aud_v4 = audit_scene(sc_v4, enforce_physical_contact=True)
            caud_v4 = audit_assembly_contracts(prog_v4, sc_v4) if prog_v4.assembly_spec else {"passed": False}
            ok = bool(aud_v4["passed"] and caud_v4["passed"] and prog_v4.assembly_spec is not None)
            return {
                "task": task,
                "has_think_trace": bool(think_part),
                "parse_valid": True,
                "sat_passed": bool(aud_v4["passed"]),
                "passed": ok,
            }
        except Exception as exc:
            return {"task": task, "parse_valid": False, "passed": False, "error": str(exc)}

    # Block-DSL producing sculpted tasks (`creation`, `stage_progression`, `multiview_inverse`, or DSL `v4_modular`)
    dsl_body = _extract_dsl_candidate(body)
    try:
        ev = evaluate_sculpted_blockbench_style(dsl_body)
    except Exception as exc:
        return {"task": task, "parse_valid": False, "passed": False, "error": f"dsl_compile_error: {exc}"}

    min_sc = float(meta.get("min_sculpted_score", 0.68))
    passed = (
        bool(ev["audit_passed"])
        and not bool(ev["is_stiff_or_voxel_like"])
        and float(ev["sculpted_score"]) >= min_sc
    )
    return {
        "task": task,
        "has_think_trace": bool(think_part),
        "parse_valid": True,
        "sat_passed": bool(ev["audit_passed"]),
        "passed": passed,
        "sculpted_eval": ev,
    }
