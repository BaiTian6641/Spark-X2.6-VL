"""Stage-11 `Blockbench_sculpted_style-v38.8` & SyncCraft `6031e58` 64-Archetype Verified Corpus (`CC0-1.0`).

Programmatically synthesizes and LIVE-VERIFIES **64 unique commercial-clean (`CC0-1.0`) Block-DSL v4.1
archetypes** across 8 anatomical & mechanical domains, exercising every new SyncCraft `6031e58` (`v38.8`)
construct:
1. **`D4-Aquatic` Underwater Swimmers (`8` species)**: `sculpt_group`, `inset_carve`, `uv_scale: 2.0`,
   `@muscles { scale_bulge }`, and native `@behavior` `swim { mode: leviathan_cruise, in_place: true, auto_bio_limits: true }`.
2. **`D4-Air` Origami-Wing Flyers & Multi-Wing Seraphs (`8` species)**: `sculpt_group`, `inset_carve`,
   `uv_scale: 2.0`, multi-segment folding origami wings (`mirror_x`), and `@behavior` `flight { mode: hover|cruise, in_place: true }`.
3. **`D3-Quadruped` Noble Wildlife & Mounts — Pole A Gentle (`8` species)**: open calm brow (`Rx <= 0°`),
   100%-closed rest mouth, `sculpt_group`, `inset_carve` eye (`uv_scale: 2.0`), `bio_joint`, grounded feet at `y=0`,
   `@behavior` `gait { pattern: lateral_sequence|trot, in_place: true, head_stabilize: 0.85, auto_bio_limits: true }`.
4. **`D4-Boss` Furious Apex Wyrms & Behemoths — Pole B Wrathful (`8` species)**: plunging `\\ /` brow
   (`Rx +20°`, `Rz -28°/+28°`), net downward predator glare, 100%-closed flush rest jaw (ZERO exposed teeth at rest),
   `sculpt_group`, `inset_carve` brow/eye (`uv_scale: 2.0`), grounded claws at `y=0`, `@muscles { scale_bulge }`,
   `@behavior` `gait` + `roar`.
5. **`D5-Vehicle` Rigid Wheeled Crawlers & Transporters (`8` machines)**: chassis-first wheelbase, `sculpt_group`
   armored hull, `inset_carve` recessed windshield/headlamps (`uv_scale: 2.0`), 4-band octagon wheels, Ackermann
   steering knuckles, `0%` `Sim(3)` metal bulge, grounded tires at `y=0`, `@behavior` `drive_roll`.
6. **`D5-Mecha` Articulated Utility & Battle Walkers (`8` machines)**: `inset_carve` glazed visor (`uv_scale: 2.0`),
   `sculpt_group` pauldrons, `bio_joint` (`ball_hip`, `hinge_knee`, `ball_shoulder`, `hinge_elbow`, `finger_flex`),
   `0%` `Sim(3)` bulge, grounded foot pads at `y=0`, `@behavior` `gait { pattern: biped_alternate, in_place: true }`.
7. **`D3-Humanoid` Stylized Artisans, Guardians & Golems (`8` characters)**: `pelvis -> chest -> neck -> head`,
   `sculpt_group` cuirass/pauldrons, `inset_carve` eyes (`uv_scale: 2.0`), `bio_joint` + `finger_flex`, grounded boots
   at `y=0`, `@behavior` `gait { pattern: biped_alternate, in_place: true, head_stabilize: 0.85, auto_bio_limits: true }`.
8. **`D2-Arthropod` Multi-Legged Crawlers & Reef Crustaceans (`8` creatures)**: segmented carapace (`sculpt_group`),
   `inset_carve` compound eyes (`uv_scale: 2.0`), 6 or 8 mirrored legs grounded at `y=0`, `@behavior`
   `gait { pattern: metachronal_wave, in_place: true, auto_bio_limits: true }`.

Hard Quality Invariants:
- 100% original commercial-clean (`CC0-1.0`); strictly `0%` overlap with the 4 `UNSEEN_V28_PROMPTS`.
- Every single emitted Block-DSL program is live-checked with SyncCraft's real `parse_program()`,
  `validate_program()`, `compile_geometry()`, and `audit_scene(audit_profile="sculpted")` — zero broken DSLs!
- Full-color shaded RGB studio turnaround renders ONLY (`clay=False`, foreground chroma > 10.0) with exact
  4-primitive `<point>`, `<box>`, `<dash>`, `<circle>` spatial annotations.
- Also generates 14-mutator `v38.8` Grammar/Semantic Repair records (including `sculpt_group`, `inset_carve`,
  `uv_scale`, `swim`, `flight`, `size_step=0.5`, and single-character punctuation errors).
"""
from __future__ import annotations

import hashlib
import io
import math
import random
import re
from typing import Any

import numpy as np
from PIL import Image

from .common import emit
from .reasoning_agentic_corpus import check_generation_collapse_health
from .spatial_3d_primitives_corpus import (
    STUDIO_LIGHTING_PRESETS,
    fmt_box,
    fmt_circle,
    fmt_dash,
    fmt_point,
    render_shaded_3d_with_spatial_annotations,
)
from .synccraft_bridge import ensure_synccraft_importable, make_clean_room_provenance
from .visual_primitives_corpus import BANNED_BENCHMARK_SUBSTRINGS


def _snap05(val: float, minimum: float = 1.0) -> float:
    return max(minimum, round(float(val) * 2.0) / 2.0)


def _fmt_num(val: float) -> str:
    v = round(float(val) * 2.0) / 2.0
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return f"{v:.1f}"


def _synccraft_supports_v38() -> bool:
    """Detect SyncCraft `6031e58` (`v38.8`) constructs (`sculpt_group`, `inset_carve`, `uv_scale`, `swim`, `flight`)."""
    ensure_synccraft_importable()
    try:
        from synccraft.kinematics import swim_solver  # noqa: F401
        return True
    except Exception:
        return False


def _ground_dsl_pivot_y(dsl: str) -> str:
    """Compile DSL via SyncCraft SE(3) engine and shift `pivot_local` Y so lowest box corner lands cleanly on `y=0`."""
    ensure_synccraft_importable()
    from synccraft.dsl.parser import parse_program
    from synccraft.kinematics.se3_chain import compile_geometry

    prog = parse_program(dsl)
    sc = compile_geometry(prog, validate=False)
    min_y = min(float(b.corners_world[:, 1].min()) for b in sc.boxes)
    if abs(min_y) < 0.05:
        return dsl
    m = re.search(r"pivot_local\s*:\s*\[\s*([-0-9.]+)\s*,\s*([-0-9.]+)\s*,\s*([-0-9.]+)\s*\]", dsl)
    if not m:
        return dsl
    px, py, pz = float(m.group(1)), float(m.group(2)), float(m.group(3))
    new_py = round((py - min_y) * 2.0) / 2.0
    return dsl[: m.start()] + f"pivot_local: [{_fmt_num(px)}, {_fmt_num(new_py)}, {_fmt_num(pz)}]" + dsl[m.end() :]


# ---------------------------------------------------------------------------
# 8 Parametric v38.8 Archetype Builders (8 Variants Each = 64 Unique Assets)
# ---------------------------------------------------------------------------

_PALETTE_BANKS: tuple[tuple[str, list[float], list[float], list[float], list[float]], ...] = (
    ("cobalt_amber", [0.46, 0.14, 248], [0.70, 0.17, 68], [0.28, 0.06, 255], [0.88, 0.19, 82]),
    ("crimson_obsidian", [0.40, 0.17, 24], [0.68, 0.14, 52], [0.22, 0.05, 265], [0.90, 0.20, 74]),
    ("jade_ivory", [0.48, 0.14, 156], [0.82, 0.06, 86], [0.30, 0.08, 178], [0.86, 0.18, 126]),
    ("terracotta_teal", [0.52, 0.15, 42], [0.48, 0.13, 196], [0.26, 0.05, 240], [0.88, 0.18, 78]),
    ("slate_gold", [0.44, 0.06, 238], [0.76, 0.16, 80], [0.25, 0.04, 250], [0.90, 0.18, 92]),
    ("ochre_indigo", [0.64, 0.15, 72], [0.38, 0.13, 268], [0.26, 0.06, 275], [0.89, 0.19, 86]),
    ("violet_copper", [0.44, 0.15, 298], [0.60, 0.16, 48], [0.24, 0.05, 290], [0.87, 0.18, 66]),
    ("sandstone_cerulean", [0.68, 0.11, 76], [0.48, 0.15, 232], [0.28, 0.06, 236], [0.88, 0.19, 88]),
)


def _header_and_palette(pal_idx: int, *, emissive_eye: bool = True) -> tuple[str, str]:
    ptag, p1, p2, p3, p4 = _PALETTE_BANKS[pal_idx % len(_PALETTE_BANKS)]
    em = ", emissive=true" if emissive_eye else ""
    hdr = f"""@meta {{ archetype: entity, target_profile: bedrock_entity, texel_density: 16, units_per_block: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }}
@palette {{
  primary: ramp(base_lch={p1}, steps=5, hue_shift=[-14, 16])
  secondary: ramp(base_lch={p2}, steps=4, hue_shift=[-12, 14])
  dark: ramp(base_lch={p3}, steps=4, hue_shift=[-10, 10])
  eye_glow: ramp(base_lch={p4}, steps=4, hue_shift=[-6, 8]{em})
}}"""
    return ptag, hdr


def _build_aquatic_swim_v38(idx: int, *, v38: bool) -> dict[str, Any]:
    specs = (
        ("abyssal_lantern_oarfish", 8.0, 6.0, 12.0, 0.50, 14.0, 12.0),
        ("coral_manta_strider", 10.0, 5.0, 11.0, 0.45, 12.0, 11.0),
        ("cobalt_reef_barracuda", 6.5, 5.5, 13.0, 0.65, 15.0, 10.0),
        ("saffron_kelp_sea_drake", 8.5, 6.5, 12.5, 0.50, 16.0, 13.0),
        ("opal_trench_nautilus_wyrm", 7.5, 6.0, 11.5, 0.55, 15.0, 12.0),
        ("jade_current_salamander", 7.5, 5.5, 12.0, 0.60, 14.0, 10.0),
        ("cinnabar_thermal_eel", 6.0, 5.0, 14.0, 0.70, 18.0, 11.0),
        ("azure_pelagic_pleiosaur", 9.0, 6.5, 12.0, 0.48, 13.0, 13.0),
    )
    name, bw, bh, bl, freq, wamp, buoy = specs[idx]
    ptag, hdr = _header_and_palette(idx)
    sg_root = "\n    sculpt_group: body_pack" if v38 else ""
    sg_box = ", sculpt_group: body_pack" if v38 else ""
    sg_head = "\n      sculpt_group: head_pack" if v38 else ""
    uv_eye = ", uv_scale: 2.0" if v38 else ""
    carve_dorsal = (
        f"box dorsal_keel inset_carve(base_box=pelvis, anchor=[0, 1, 0], protrude=1.5) {{ size: [{_fmt_num(_snap05(bw*0.55, 2.0))}, 2.5, {_fmt_num(_snap05(bl*0.75, 4.0))}], material: secondary{sg_box} }}"
        if v38
        else f"box dorsal_keel place(base_box=pelvis, anchor=[0, 1, 0], align=[0, -1, 0]) {{ size: [{_fmt_num(_snap05(bw*0.55, 2.0))}, 2, {_fmt_num(_snap05(bl*0.75, 4.0))}], material: secondary }}"
    )
    carve_eye_l = (
        f"box eye_l inset_carve(base_box=skull, anchor=[-1, 0.2, 0.25], protrude=0.5) {{ size: [1.5, 1.5, 1.5], material: eye_glow{uv_eye}, sculpt_group: head_pack }}"
        if v38
        else "box eye_l place(base_box=skull, anchor=[-1, 0.2, 0.25], align=[1, 0, 0]) { size: [1, 1.5, 1.5], material: eye_glow }"
    )
    carve_eye_r = (
        f"box eye_r inset_carve(base_box=skull, anchor=[1, 0.2, 0.25], protrude=0.5) {{ size: [1.5, 1.5, 1.5], material: eye_glow{uv_eye}, sculpt_group: head_pack }}"
        if v38
        else "box eye_r place(base_box=skull, anchor=[1, 0.2, 0.25], align=[-1, 0, 0]) { size: [1, 1.5, 1.5], material: eye_glow }"
    )
    muscles_blk = (
        """@muscles {
  muscle flank_myomere {
    origin_bone: pelvis_root,
    insertion_bone: tail_1,
    belly_box: pelvis,
    action_axis: y,
    moment_arm_deg: 2.0,
    scale_bulge: 0.03
  }
}"""
        if v38
        else ""
    )
    beh_blk = (
        f"""@behavior {{
  clip swim_cruise {{
    duration_s: 2.0, loop: true, in_place: true, auto_rig: true, auto_bio_limits: true
    swim leviathan_swim {{
      mode: leviathan_cruise, freq_hz: {freq}, wave_amp_deg: {wamp}, buoyancy_y: {buoy},
      heave_amp_u: 1.5, body_pitch_deg: -4.0, head_stabilize: 0.80,
      limb_mode: hydro_tuck, limb_tuck: 0.85, myomere_bulge: 0.80, fin_lag_deg: 12.0,
      in_place: true, auto_bio_limits: true, ground_clearance: true
    }}
    muscle_drive flank_myomere: activation(peak=0.70, freq_hz={freq}, baseline=0.15)
  }}
}}"""
        if v38
        else f"""@behavior {{
  clip swim_cruise {{
    duration_s: 2.0, loop: true
    track tail_1: wave(axis=y, amp_deg={int(wamp)}, freq_hz={freq})
    track tail_2: wave(axis=y, amp_deg={int(wamp+4)}, freq_hz={freq}, phase_deg=45)
  }}
}}"""
    )
    dsl = f"""{hdr}
@rig {{
  root bone pelvis_root {{
    pivot_local: [0, 8, 0]{sg_root}
    box pelvis {{ size: [{_fmt_num(bw)}, {_fmt_num(bh)}, {_fmt_num(bl)}], material: primary }}
    {carve_dorsal}
    bone chest attach(parent_box=pelvis, anchor=[0, 0, 1], align=[0, 0, -1]) {{
      box chest_core {{ size: [{_fmt_num(_snap05(bw*0.9, 5.0))}, {_fmt_num(_snap05(bh*0.9, 4.5))}, {_fmt_num(_snap05(bl*0.7, 6.0))}], material: primary }}
      bone head attach(parent_box=chest_core, anchor=[0, 0.1, 1], align=[0, 0, -1]) {{{sg_head}
        box skull {{ size: [{_fmt_num(_snap05(bw*0.7, 4.5))}, {_fmt_num(_snap05(bh*0.75, 4.0))}, 6], material: primary }}
        box snout place(base_box=skull, anchor=[0, -0.1, 1], align=[0, 0, -1]) {{ size: [{_fmt_num(_snap05(bw*0.5, 3.0))}, 2.5, 4], material: secondary }}
        {carve_eye_l}
        {carve_eye_r}
      }}
      bone pec_fin_l attach(parent_box=chest_core, anchor=[-1, -0.3, 0.1], align=[1, 0, 0]) mirror_x(pec_fin_r) {{
        box fin_blade {{ size: [6.5, 1, 4], material: secondary, rotate: [0, -14, -12] }}
      }}
    }}
    bone pelv_fin_l attach(parent_box=pelvis, anchor=[-1, -0.4, -0.2], align=[1, 0, 0]) mirror_x(pelv_fin_r) {{
      box pfin_blade {{ size: [4.5, 1, 3], material: dark, rotate: [0, -10, -10] }}
    }}
    bone tail_1 attach(parent_box=pelvis, anchor=[0, 0, -1], align=[0, 0, 1]) {{
      box t1 {{ size: [{_fmt_num(_snap05(bw*0.65, 3.5))}, {_fmt_num(_snap05(bh*0.7, 3.5))}, 9], material: primary }}
      bone tail_2 attach(parent_box=t1, anchor=[0, 0, -1], align=[0, 0, 1]) {{
        box t2 {{ size: [{_fmt_num(_snap05(bw*0.42, 2.5))}, {_fmt_num(_snap05(bh*0.5, 2.5))}, 8.5], material: secondary }}
        bone tail_fin attach(parent_box=t2, anchor=[0, 0, -1], align=[0, 0, 1]) {{
          box fluke {{ size: [1, 6.5, 6], material: dark }}
        }}
      }}
    }}
  }}
}}
{muscles_blk}
{beh_blk}"""
    return {
        "asset_id": f"v38_aquatic_{name}",
        "family": "D4_aquatic_swim",
        "domain": "D4",
        "mood": "M3",
        "polarity": "pole_a_aquatic_biomech",
        "palette_tag": ptag,
        "title": name.replace("_", " ").title(),
        "prompt": (
            f"Design an original Blockbench_sculpted_style-v38.8 aquatic creature in `Block-DSL v4.1` "
            f"(`dsl_version: \"4.1\"`, `audit_profile: sculpted`, `size_step: 0.5`): **{name.replace('_', ' ').title()}** "
            f"({ptag} palette). Use `sculpt_group` compound torso packing, `inset_carve` dorsal keel and recessed eyes "
            f"with `uv_scale: 2.0`, mirrored pectoral/pelvic flippers, tapered segmented tail fluke, `@muscles` with "
            f"`scale_bulge: 0.03`, and `@behavior` `swim` (`mode: leviathan_cruise`, `in_place: true`, `auto_bio_limits: true`)."
        ),
        "dsl": dsl.strip(),
        "clip_name": "swim_cruise",
    }


def _build_origami_flyer_v38(idx: int, *, v38: bool) -> dict[str, Any]:
    specs = (
        ("amber_ridge_condor", 6.5, 5.5, 10.0, 9.0, 10.0, "cruise", 0.70, 26),
        ("cobalt_gale_moth_guardian", 6.0, 5.0, 9.0, 8.5, 9.5, "hover", 0.80, 30),
        ("vermilion_sun_falcon", 5.5, 5.0, 9.5, 8.0, 9.0, "cruise", 0.75, 28),
        ("obsidian_monsoon_wyvern", 8.0, 6.5, 12.0, 11.0, 12.0, "cruise", 0.55, 24),
        ("ivory_cloud_seraph", 7.0, 6.0, 10.5, 10.0, 11.0, "hover", 0.62, 28),
        ("emerald_canopy_glider", 6.0, 4.5, 9.5, 9.0, 9.5, "cruise", 0.65, 22),
        ("copper_canyon_kestrel", 5.5, 4.5, 8.5, 7.5, 8.5, "hover", 0.85, 32),
        ("violet_aurora_bat_drake", 7.5, 6.0, 11.0, 10.5, 11.5, "cruise", 0.60, 26),
    )
    name, bw, bh, bl, w_in, w_out, fmode, fhz, sdeg = specs[idx]
    ptag, hdr = _header_and_palette(idx + 1)
    sg_root = "\n    sculpt_group: thorax_pack" if v38 else ""
    sg_box = ", sculpt_group: thorax_pack" if v38 else ""
    sg_head = "\n      sculpt_group: head_pack" if v38 else ""
    uv_eye = ", uv_scale: 2.0" if v38 else ""
    carve_sternum = (
        f"box keel_crest inset_carve(base_box=torso, anchor=[0, -1, 0.2], protrude=1.5) {{ size: [{_fmt_num(_snap05(bw*0.55, 2.5))}, 2, {_fmt_num(_snap05(bl*0.7, 5.0))}], material: secondary{sg_box} }}"
        if v38
        else f"box keel_crest place(base_box=torso, anchor=[0, -1, 0.2], align=[0, 1, 0]) {{ size: [{_fmt_num(_snap05(bw*0.55, 2.5))}, 1.5, {_fmt_num(_snap05(bl*0.7, 5.0))}], material: secondary }}"
    )
    eye_l = (
        f"box eye_l inset_carve(base_box=skull, anchor=[-1, 0.2, 0.2], protrude=0.5) {{ size: [1.5, 1.5, 1.5], material: eye_glow{uv_eye}, sculpt_group: head_pack }}"
        if v38
        else "box eye_l place(base_box=skull, anchor=[-1, 0.2, 0.2], align=[1, 0, 0]) { size: [1, 1.5, 1.5], material: eye_glow }"
    )
    eye_r = (
        f"box eye_r inset_carve(base_box=skull, anchor=[1, 0.2, 0.2], protrude=0.5) {{ size: [1.5, 1.5, 1.5], material: eye_glow{uv_eye}, sculpt_group: head_pack }}"
        if v38
        else "box eye_r place(base_box=skull, anchor=[1, 0.2, 0.2], align=[-1, 0, 0]) { size: [1, 1.5, 1.5], material: eye_glow }"
    )
    beh_blk = (
        f"""@behavior {{
  clip wing_flight {{
    duration_s: 1.6, loop: true, in_place: true, auto_bio_limits: true
    flight aerial_cycle {{
      mode: {fmode}, flap_hz: {fhz}, stroke_deg: {sdeg}, sweep_deg: 10,
      pronation_deg: 12, elbow_fold_deg: 14, lift_recoil_u: 2.2,
      body_pitch_deg: 10, head_stabilize: 0.85, in_place: true
    }}
  }}
}}"""
        if v38
        else f"""@behavior {{
  clip wing_flight {{
    duration_s: 1.6, loop: true
    track wing_l: wave(axis=z, amp_deg={sdeg}, freq_hz={fhz})
    track wing_r: wave(axis=z, amp_deg=-{sdeg}, freq_hz={fhz})
  }}
}}"""
    )
    dsl = f"""{hdr}
@rig {{
  root bone chest_root {{
    pivot_local: [0, 12, 0]{sg_root}
    box torso {{ size: [{_fmt_num(bw)}, {_fmt_num(bh)}, {_fmt_num(bl)}], material: primary }}
    {carve_sternum}
    bone neck attach(parent_box=torso, anchor=[0, 0.5, 1], align=[0, -0.3, -1]) {{
      box neck_seg {{ size: [3, 3, 4], material: primary }}
      bone head attach(parent_box=neck_seg, anchor=[0, 0.2, 1], align=[0, 0, -1]) {{{sg_head}
        box skull {{ size: [4.5, 4, 5], material: primary }}
        box beak place(base_box=skull, anchor=[0, -0.1, 1], align=[0, 0, -1]) {{ size: [2, 2, 3.5], material: secondary }}
        {eye_l}
        {eye_r}
      }}
    }}
    bone wing_l attach(parent_box=torso, anchor=[-1, 0.5, 0.2], align=[1, 0, 0]) mirror_x(wing_r) {{
      box wing_inner {{ size: [{_fmt_num(w_in)}, 1, 5.5], material: primary }}
      bone wing_outer attach(parent_box=wing_inner, anchor=[-1, 0, 0], align=[1, 0, 0]) {{
        box wing_tip {{ size: [{_fmt_num(w_out)}, 0.5, 4.5], material: secondary }}
      }}
    }}
    bone leg_l attach(parent_box=torso, anchor=[-0.6, -1, -0.2], align=[0, 1, 0]) mirror_x(leg_r) {{
      box thigh {{ size: [2, 4.5, 2], material: dark }}
      bone talon attach(parent_box=thigh, anchor=[0, -1, 0], align=[0, 1, 0]) {{
        box claw_pad {{ size: [2.5, 1.5, 3], material: secondary }}
      }}
    }}
    bone tail_1 attach(parent_box=torso, anchor=[0, 0, -1], align=[0, 0, 1]) {{
      box t1 {{ size: [3.5, 1.5, 6.5], material: secondary }}
      bone tail_fan attach(parent_box=t1, anchor=[0, 0, -1], align=[0, 0, 1]) {{
        box fan_blade {{ size: [5.5, 0.5, 5], material: dark }}
      }}
    }}
  }}
}}
{beh_blk}"""
    return {
        "asset_id": f"v38_flyer_{name}",
        "family": "D4_air_flight",
        "domain": "D4",
        "mood": "M3",
        "polarity": "pole_a_aerial_origami",
        "palette_tag": ptag,
        "title": name.replace("_", " ").title(),
        "prompt": (
            f"Design an original Blockbench_sculpted_style-v38.8 winged creature in `Block-DSL v4.1` "
            f"(`dsl_version: \"4.1\"`, `audit_profile: sculpted`, `size_step: 0.5`): **{name.replace('_', ' ').title()}** "
            f"({ptag} palette). Build compound-packed thorax (`sculpt_group`), `inset_carve` sternum crest and focal eyes "
            f"(`uv_scale: 2.0`), 2-stage mirrored origami wings (`0.5u` zero-blink outer membrane slab), fan tail, and "
            f"`@behavior` `flight` (`mode: {fmode}`, `in_place: true`, `head_stabilize: 0.85`)."
        ),
        "dsl": dsl.strip(),
        "clip_name": "wing_flight",
    }


def _build_quadruped_pole_a_v38(idx: int, *, v38: bool) -> dict[str, Any]:
    specs = (
        ("ochre_steppe_gazelle", 7.0, 6.0, 13.0, 5.5, 5.5, "trot", 0.30),
        ("slate_tundra_caribou", 8.5, 7.0, 15.0, 6.0, 6.0, "lateral_sequence", 0.22),
        ("terracotta_mesa_hound", 7.0, 6.0, 12.5, 4.5, 4.5, "trot", 0.28),
        ("ivory_orchard_ram", 8.0, 6.5, 13.5, 4.5, 4.5, "lateral_sequence", 0.20),
        ("moss_hollow_tapir", 8.5, 7.0, 14.0, 4.0, 4.0, "lateral_sequence", 0.18),
        ("sandstone_dune_oryx", 7.5, 6.5, 14.0, 5.5, 5.5, "trot", 0.26),
        ("indigo_highland_fox", 6.5, 5.5, 12.0, 4.0, 4.0, "trot", 0.30),
        ("cedar_grove_bear", 10.0, 8.0, 15.5, 5.0, 5.0, "lateral_sequence", 0.18),
    )
    name, bw, bh, bl, up_l, lo_l, gpat, fr = specs[idx]
    ptag, hdr = _header_and_palette(idx + 2)
    sg_root = "\n    sculpt_group: torso_pack" if v38 else ""
    sg_box = ", sculpt_group: torso_pack" if v38 else ""
    sg_head = "\n      sculpt_group: head_pack" if v38 else ""
    uv_eye = ", uv_scale: 2.0" if v38 else ""
    bio_s = "\n      bio_joint: ball_shoulder" if v38 else ""
    bio_e = "\n        bio_joint: hinge_elbow" if v38 else ""
    bio_h = "\n      bio_joint: ball_hip" if v38 else ""
    bio_k = "\n        bio_joint: hinge_knee" if v38 else ""
    wither = (
        f"box withers_pack inset_carve(base_box=torso_core, anchor=[0, 1, 0.3], protrude=1.5) {{ size: [{_fmt_num(_snap05(bw*0.75, 4.0))}, 2.5, {_fmt_num(_snap05(bl*0.45, 4.5))}], material: secondary{sg_box} }}"
        if v38
        else f"box withers_pack place(base_box=torso_core, anchor=[0, 1, 0.3], align=[0, -1, 0]) {{ size: [{_fmt_num(_snap05(bw*0.75, 4.0))}, 2, {_fmt_num(_snap05(bl*0.45, 4.5))}], material: secondary }}"
    )
    eye_l = (
        f"box eye_l inset_carve(base_box=skull, anchor=[-1, 0.15, 0.25], protrude=0.5) {{ size: [1.5, 1.5, 1.5], material: eye_glow{uv_eye}, sculpt_group: head_pack }}"
        if v38
        else "box eye_l place(base_box=skull, anchor=[-1, 0.15, 0.25], align=[1, 0, 0]) { size: [1, 1.5, 1.5], material: eye_glow }"
    )
    eye_r = (
        f"box eye_r inset_carve(base_box=skull, anchor=[1, 0.15, 0.25], protrude=0.5) {{ size: [1.5, 1.5, 1.5], material: eye_glow{uv_eye}, sculpt_group: head_pack }}"
        if v38
        else "box eye_r place(base_box=skull, anchor=[1, 0.15, 0.25], align=[-1, 0, 0]) { size: [1, 1.5, 1.5], material: eye_glow }"
    )
    clip_hdr = "duration_s: 1.4, loop: true, in_place: true, head_stabilize: 0.85, auto_bio_limits: true" if v38 else "duration_s: 1.4, loop: true"
    gait_opts = f"pattern: {gpat}, froude: {fr}, duty: 0.64, lift_ratio: 0.16, in_place: true, head_stabilize: 0.85" if v38 else f"pattern: {gpat}, froude: {fr}, duty: 0.64"
    dsl = f"""{hdr}
@rig {{
  root bone pelvis {{
    pivot_local: [0, 12, 0]{sg_root}
    box torso_core {{ size: [{_fmt_num(bw)}, {_fmt_num(bh)}, {_fmt_num(bl)}], material: primary }}
    {wither}
    bone neck attach(parent_box=torso_core, anchor=[0, 0.6, 1], align=[0, -0.5, -1]) {{
      box neck_core {{ size: [{_fmt_num(_snap05(bw*0.5, 3.5))}, 4, 4.5], material: primary, rotate: [-14, 0, 0] }}
      bone head attach(parent_box=neck_core, anchor=[0, 0.5, 1], align=[0, -0.3, -1]) {{{sg_head}
        box skull {{ size: [4.5, 4, 5], material: primary }}
        box muzzle place(base_box=skull, anchor=[0, -0.25, 1], align=[0, 0, -1]) {{ size: [3, 2.5, 3.5], material: secondary }}
        {eye_l}
        {eye_r}
        bone ear_l attach(parent_box=skull, anchor=[-0.85, 1, -0.3], align=[0, -1, 0]) mirror_x(ear_r) {{
          box ear_blade {{ size: [1.5, 3, 1.5], material: secondary, rotate: [-10, 0, -18] }}
        }}
      }}
    }}
    bone front_leg_l attach(parent_box=torso_core, anchor=[-0.85, -0.7, 0.65], align=[0.4, 1, 0]) mirror_x(front_leg_r) {{{bio_s}
      box upper_fl {{ size: [2.5, {_fmt_num(up_l)}, 3], material: primary }}
      bone lower_fl attach(parent_box=upper_fl, anchor=[0, -1, 0], align=[0, 1, 0]) {{{bio_e}
        box shin_fl {{ size: [2, {_fmt_num(lo_l)}, 2], material: secondary }}
        bone hoof_fl attach(parent_box=shin_fl, anchor=[0, -1, 0], align=[0, 1, 0]) {{
          box foot_fl {{ size: [2.5, 1.5, 2.5], material: dark }}
        }}
      }}
    }}
    bone hind_leg_l attach(parent_box=torso_core, anchor=[-0.85, -0.7, -0.65], align=[0.4, 1, 0]) mirror_x(hind_leg_r) {{{bio_h}
      box thigh_hl {{ size: [3, {_fmt_num(up_l)}, 3.5], material: primary }}
      bone lower_hl attach(parent_box=thigh_hl, anchor=[0, -1, 0], align=[0, 1, 0]) {{{bio_k}
        box shin_hl {{ size: [2, {_fmt_num(lo_l)}, 2], material: secondary }}
        bone hoof_hl attach(parent_box=shin_hl, anchor=[0, -1, 0], align=[0, 1, 0]) {{
          box foot_hl {{ size: [2.5, 1.5, 2.5], material: dark }}
        }}
      }}
    }}
    bone tail attach(parent_box=torso_core, anchor=[0, 0.3, -1], align=[0, 0, 1]) {{
      box tail_seg {{ size: [2, 2, 5.5], material: secondary, rotate: [18, 0, 0] }}
    }}
  }}
}}
@behavior {{
  clip walk_cycle {{
    {clip_hdr}
    gait quad_stride {{ {gait_opts} }}
  }}
}}"""
    return {
        "asset_id": f"v38_quad_pole_a_{name}",
        "family": "D3_quadruped_pole_a",
        "domain": "D3",
        "mood": "M3",
        "polarity": "pole_a_gentle_noble",
        "palette_tag": ptag,
        "title": name.replace("_", " ").title(),
        "prompt": (
            f"Design an original Blockbench_sculpted_style-v38.8 quadruped mount in `Block-DSL v4.1` "
            f"(`dsl_version: \"4.1\"`, `audit_profile: sculpted`, `size_step: 0.5`): **{name.replace('_', ' ').title()}** "
            f"— **Domain D3, Pole A (Gentle & Noble)** ({ptag} palette). Enforce open calm brow, 100%-closed rest muzzle, "
            f"`sculpt_group` withers packing, `inset_carve` recessed eyes (`uv_scale: 2.0`), `bio_joint` limb articulation, "
            f"4 grounded hooves at `y=0`, and `@behavior` Froude `gait` (`{gpat}`, `in_place: true`, `head_stabilize: 0.85`)."
        ),
        "dsl": dsl.strip(),
        "clip_name": "walk_cycle",
    }


def _build_boss_pole_b_v38(idx: int, *, v38: bool) -> dict[str, Any]:
    specs = (
        ("ashen_basalt_dreadwyrm", 11.0, 8.5, 16.0, 5.5, 5.0),
        ("crimson_caldera_manticore", 10.0, 8.0, 15.0, 5.0, 5.0),
        ("obsidian_rift_stalker", 9.5, 7.5, 15.5, 5.5, 5.0),
        ("pyrite_thunder_behemoth", 12.0, 9.0, 16.5, 5.5, 4.5),
        ("garnet_cinder_chimera", 10.5, 8.0, 15.5, 5.0, 5.0),
        ("iron_trench_hydra_beast", 11.0, 8.5, 16.0, 5.0, 5.0),
        ("sulfur_crag_basilisk", 9.5, 7.0, 16.5, 4.5, 4.5),
        ("onyx_eclipse_gargoyle", 10.0, 8.5, 14.5, 5.5, 5.0),
    )
    name, bw, bh, bl, up_l, lo_l = specs[idx]
    ptag, hdr = _header_and_palette(idx + 3)
    sg_root = "\n    sculpt_group: thorax_pack" if v38 else ""
    sg_box = ", sculpt_group: thorax_pack" if v38 else ""
    sg_head = "\n      sculpt_group: cranium_pack" if v38 else ""
    uv_eye = ", uv_scale: 2.0" if v38 else ""
    ridge = (
        f"box dorsal_ridge inset_carve(base_box=chest_core, anchor=[0, 1, 0], protrude=2.0) {{ size: [{_fmt_num(_snap05(bw*0.45, 3.5))}, 3, {_fmt_num(_snap05(bl*0.75, 8.0))}], material: dark{sg_box} }}"
        if v38
        else f"box dorsal_ridge place(base_box=chest_core, anchor=[0, 1, 0], align=[0, -1, 0]) {{ size: [{_fmt_num(_snap05(bw*0.45, 3.5))}, 2.5, {_fmt_num(_snap05(bl*0.75, 8.0))}], material: dark }}"
    )
    brow_l = (
        "box brow_l inset_carve(base_box=skull, anchor=[-1, 0.75, 0.3], protrude=1.0) { size: [2, 1.5, 4], material: dark, rotate: [20, 0, -28], sculpt_group: cranium_pack }"
        if v38
        else "box brow_l place(base_box=skull, anchor=[-1, 0.75, 0.3], align=[1, 0, 0]) { size: [1.5, 1.5, 4], material: dark, rotate: [20, 0, -28] }"
    )
    brow_r = (
        "box brow_r inset_carve(base_box=skull, anchor=[1, 0.75, 0.3], protrude=1.0) { size: [2, 1.5, 4], material: dark, rotate: [20, 0, 28], sculpt_group: cranium_pack }"
        if v38
        else "box brow_r place(base_box=skull, anchor=[1, 0.75, 0.3], align=[-1, 0, 0]) { size: [1.5, 1.5, 4], material: dark, rotate: [20, 0, 28] }"
    )
    eye_l = (
        f"box eye_l inset_carve(base_box=skull, anchor=[-1, 0.15, 0.25], protrude=0.5) {{ size: [1.5, 1.5, 1.5], material: eye_glow{uv_eye}, sculpt_group: cranium_pack }}"
        if v38
        else "box eye_l place(base_box=skull, anchor=[-1, 0.15, 0.25], align=[1, 0, 0]) { size: [1, 1.5, 1.5], material: eye_glow }"
    )
    eye_r = (
        f"box eye_r inset_carve(base_box=skull, anchor=[1, 0.15, 0.25], protrude=0.5) {{ size: [1.5, 1.5, 1.5], material: eye_glow{uv_eye}, sculpt_group: cranium_pack }}"
        if v38
        else "box eye_r place(base_box=skull, anchor=[1, 0.15, 0.25], align=[-1, 0, 0]) { size: [1, 1.5, 1.5], material: eye_glow }"
    )
    muscles_blk = (
        """@muscles {
  muscle chest_deltoid {
    origin_bone: pelvis,
    insertion_bone: front_leg_l,
    belly_box: chest_core,
    action_axis: x,
    moment_arm_deg: 2.5,
    scale_bulge: 0.04
  }
}"""
        if v38
        else ""
    )
    clip_hdr = "duration_s: 1.6, loop: true, in_place: true, head_stabilize: 0.85, auto_bio_limits: true" if v38 else "duration_s: 1.6, loop: true"
    gait_opts = "pattern: lateral_sequence, froude: 0.20, duty: 0.66, lift_ratio: 0.14, in_place: true, head_stabilize: 0.85" if v38 else "pattern: lateral_sequence, froude: 0.20, duty: 0.66"
    dsl = f"""{hdr}
@rig {{
  root bone pelvis {{
    pivot_local: [0, 12, 0]{sg_root}
    box chest_core {{ size: [{_fmt_num(bw)}, {_fmt_num(bh)}, {_fmt_num(bl)}], material: primary }}
    {ridge}
    bone neck attach(parent_box=chest_core, anchor=[0, 0.4, 1], align=[0, -0.3, -1]) {{
      box neck_mass {{ size: [5.5, 5, 5.5], material: primary, rotate: [-6, 0, 0] }}
      bone head attach(parent_box=neck_mass, anchor=[0, 0.2, 1], align=[0, 0, -1]) {{{sg_head}
        rest_euler: [14, 0, 0]
        box skull {{ size: [6, 5, 6.5], material: primary }}
        box upper_snout place(base_box=skull, anchor=[0, 0.1, 1], align=[0, 0, -1]) {{ size: [4.5, 2.5, 5], material: primary }}
        {brow_l}
        {brow_r}
        {eye_l}
        {eye_r}
        bone jaw attach(parent_box=skull, anchor=[0, -0.65, 0.8], align=[0, 0.5, -1]) {{
          rest_euler: [0, 0, 0]
          box mandible {{ size: [4, 1.5, 5], material: secondary }}
        }}
        bone horn_l attach(parent_box=skull, anchor=[-0.85, 0.85, -0.6], align=[0.4, -0.6, 1]) mirror_x(horn_r) {{
          box horn_shaft {{ size: [2, 2, 7.5], material: dark, rotate: [32, -16, 0] }}
        }}
      }}
    }}
    bone front_leg_l attach(parent_box=chest_core, anchor=[-0.9, -0.6, 0.6], align=[0.45, 1, 0]) mirror_x(front_leg_r) {{
      box arm_u {{ size: [3.5, {_fmt_num(up_l)}, 4], material: primary }}
      bone arm_l attach(parent_box=arm_u, anchor=[0, -1, 0], align=[0, 1, 0]) {{
        box forearm {{ size: [3, {_fmt_num(lo_l)}, 3], material: secondary }}
        bone claw_fl attach(parent_box=forearm, anchor=[0, -1, 0], align=[0, 1, 0]) {{
          box paw_fl {{ size: [4, 2, 4.5], material: dark }}
        }}
      }}
    }}
    bone hind_leg_l attach(parent_box=chest_core, anchor=[-0.9, -0.6, -0.65], align=[0.45, 1, 0]) mirror_x(hind_leg_r) {{
      box thigh_u {{ size: [4, {_fmt_num(up_l)}, 4.5], material: primary }}
      bone shin_l attach(parent_box=thigh_u, anchor=[0, -1, 0], align=[0, 1, 0]) {{
        box calf {{ size: [3, {_fmt_num(lo_l)}, 3], material: secondary }}
        bone claw_hl attach(parent_box=calf, anchor=[0, -1, 0], align=[0, 1, 0]) {{
          box paw_hl {{ size: [4, 2, 4.5], material: dark }}
        }}
      }}
    }}
    bone tail_1 attach(parent_box=chest_core, anchor=[0, -0.1, -1], align=[0, 0, 1]) {{
      box t1 {{ size: [4.5, 4, 9], material: primary }}
      bone tail_2 attach(parent_box=t1, anchor=[0, 0, -1], align=[0, 0, 1]) {{
        box t2 {{ size: [3, 2.5, 8], material: secondary }}
      }}
    }}
  }}
}}
{muscles_blk}
@behavior {{
  clip stalk_stride {{
    {clip_hdr}
    gait apex_stalk {{ {gait_opts} }}
  }}
  clip threat_roar {{
    duration_s: 1.5, loop: true
    track jaw: hinge(axis=x, amp_deg=40, freq_hz=0.65)
    track head: wave(axis=x, amp_deg=-10, freq_hz=0.65)
  }}
}}"""
    return {
        "asset_id": f"v38_boss_pole_b_{name}",
        "family": "D4_boss_pole_b",
        "domain": "D4",
        "mood": "M4",
        "polarity": "pole_b_furious_unapproachable",
        "palette_tag": ptag,
        "title": name.replace("_", " ").title(),
        "prompt": (
            f"Design an original Blockbench_sculpted_style-v38.8 apex boss beast in `Block-DSL v4.1` "
            f"(`dsl_version: \"4.1\"`, `audit_profile: sculpted`, `size_step: 0.5`): **{name.replace('_', ' ').title()}** "
            f"— **Domain D4, Mood M4, Pole B (Furious & Unapproachable)** ({ptag} palette). Enforce plunging `\\ /` "
            f"scowl brow (`Rx +20°`, `Rz -28°/+28°`), net downward `+8°` glare, 100%-closed flush rest jaw (`rest_euler:[0,0,0]`, "
            f"ZERO exposed teeth at rest) opening `40°` during `threat_roar`, `sculpt_group` + `inset_carve` eyes (`uv_scale: 2.0`), "
            f"grounded 4-paw stance at `y=0`, `@muscles` `scale_bulge: 0.04`, and `@behavior` `gait` + `threat_roar`."
        ),
        "dsl": dsl.strip(),
        "clip_name": "stalk_stride",
    }


def _build_vehicle_d5_v38(idx: int, *, v38: bool) -> dict[str, Any]:
    specs = (
        ("basalt_ore_transporter", 10.0, 22.0, 6.0),
        ("ochre_desert_recon_buggy", 9.0, 18.0, 5.5),
        ("cobalt_harbor_crane_truck", 10.5, 23.0, 6.0),
        ("olive_timber_forestry_rig", 10.0, 21.0, 6.5),
        ("terracotta_canyon_ambulift", 9.5, 20.0, 5.5),
        ("slate_rail_service_crawler", 10.0, 22.5, 6.0),
        ("copper_foundry_slag_hauler", 11.0, 22.0, 6.5),
        ("teal_coastal_radar_van", 9.5, 19.5, 5.5),
    )
    name, cw, cl, wdiam = specs[idx]
    ptag, hdr = _header_and_palette(idx + 4)
    sg_root = "\n    sculpt_group: hull_pack" if v38 else ""
    sg_cab = "\n      sculpt_group: cab_pack" if v38 else ""
    uv_lens = ", uv_scale: 2.0" if v38 else ""
    windshield = (
        f"box windshield inset_carve(base_box=cab_hull, anchor=[0, 0.35, 1], protrude=0.5) {{ size: [{_fmt_num(_snap05(cw*0.85, 6.0))}, 3, 1.5], material: eye_glow{uv_lens}, sculpt_group: cab_pack }}"
        if v38
        else f"box windshield place(base_box=cab_hull, anchor=[0, 0.35, 1], align=[0, 0, -1]) {{ size: [{_fmt_num(_snap05(cw*0.85, 6.0))}, 3, 1], material: eye_glow }}"
    )
    dsl = f"""{hdr}
@rig {{
  root bone chassis_root {{
    pivot_local: [0, 6, 0]{sg_root}
    box chassis_frame {{ size: [{_fmt_num(cw)}, 3, {_fmt_num(cl)}], material: primary }}
    bone cab_module attach(parent_box=chassis_frame, anchor=[0, 1, 0.4], align=[0, -1, 0]) {{{sg_cab}
      box cab_hull {{ size: [{_fmt_num(cw)}, 5.5, 8], material: primary }}
      {windshield}
    }}
    bone cargo_module attach(parent_box=chassis_frame, anchor=[0, 1, -0.45], align=[0, -1, 0]) {{
      box cargo_bed {{ size: [{_fmt_num(cw)}, 4.5, 9], material: secondary }}
      bone turret_yaw attach(parent_box=cargo_bed, anchor=[0, 1, 0], align=[0, -1, 0]) {{
        joint_limits: [[-2, 2], [-180, 180], [-2, 2]]
        box turret_head {{ size: [4.5, 2.5, 5], material: dark }}
      }}
    }}
    bone steer_fl attach(parent_box=chassis_frame, anchor=[-1, -0.5, 0.55], align=[1, 0, 0]) mirror_x(steer_fr) {{
      joint_limits: [[-2, 2], [-35, 35], [-2, 2]]
      box knuckle_f {{ size: [2, 1.5, 1.5], material: dark }}
      bone wheel_fl attach(parent_box=knuckle_f, anchor=[-1, 0, 0], align=[1, 0, 0]) {{
        joint_limits: [continuous, [-2, 2], [-2, 2]]
        box tire_f0 {{ size: [3, {_fmt_num(wdiam)}, 2.5], material: dark, rotate: [0, 0, 0], prism_group: "wf" }}
        box tire_f1 place(base_box=tire_f0, anchor=[0, 0, 0], align=[0, 0, 0]) {{ size: [3, {_fmt_num(wdiam)}, 2.5], material: dark, rotate: [45, 0, 0], prism_group: "wf" }}
        box tire_f2 place(base_box=tire_f0, anchor=[0, 0, 0], align=[0, 0, 0]) {{ size: [3, {_fmt_num(wdiam)}, 2.5], material: dark, rotate: [90, 0, 0], prism_group: "wf" }}
        box tire_f3 place(base_box=tire_f0, anchor=[0, 0, 0], align=[0, 0, 0]) {{ size: [3, {_fmt_num(wdiam)}, 2.5], material: dark, rotate: [135, 0, 0], prism_group: "wf" }}
      }}
    }}
    bone axle_rl attach(parent_box=chassis_frame, anchor=[-1, -0.5, -0.55], align=[1, 0, 0]) mirror_x(axle_rr) {{
      box hub_r {{ size: [2, 1.5, 1.5], material: dark }}
      bone wheel_rl attach(parent_box=hub_r, anchor=[-1, 0, 0], align=[1, 0, 0]) {{
        joint_limits: [continuous, [-2, 2], [-2, 2]]
        box tire_r0 {{ size: [3, {_fmt_num(wdiam)}, 2.5], material: dark, rotate: [0, 0, 0], prism_group: "wr" }}
        box tire_r1 place(base_box=tire_r0, anchor=[0, 0, 0], align=[0, 0, 0]) {{ size: [3, {_fmt_num(wdiam)}, 2.5], material: dark, rotate: [45, 0, 0], prism_group: "wr" }}
        box tire_r2 place(base_box=tire_r0, anchor=[0, 0, 0], align=[0, 0, 0]) {{ size: [3, {_fmt_num(wdiam)}, 2.5], material: dark, rotate: [90, 0, 0], prism_group: "wr" }}
        box tire_r3 place(base_box=tire_r0, anchor=[0, 0, 0], align=[0, 0, 0]) {{ size: [3, {_fmt_num(wdiam)}, 2.5], material: dark, rotate: [135, 0, 0], prism_group: "wr" }}
      }}
    }}
  }}
}}
@behavior {{
  clip drive_roll {{
    duration_s: 2.0, loop: true
    track wheel_fl: spin(axis=x, amp_deg=360, freq_hz=0.5)
    track steer_fr__wheel_fl: spin(axis=x, amp_deg=360, freq_hz=0.5)
    track wheel_rl: spin(axis=x, amp_deg=360, freq_hz=0.5)
    track axle_rr__wheel_rl: spin(axis=x, amp_deg=360, freq_hz=0.5)
    track steer_fl: wave(axis=y, amp_deg=12, freq_hz=0.5)
    track steer_fr: wave(axis=y, amp_deg=12, freq_hz=0.5)
    track turret_yaw: wave(axis=y, amp_deg=25, freq_hz=0.5)
  }}
}}"""
    return {
        "asset_id": f"v38_vehicle_{name}",
        "family": "D5_vehicle_rigid",
        "domain": "D5",
        "mood": "M2",
        "polarity": "mechanical_rigid",
        "palette_tag": ptag,
        "title": name.replace("_", " ").title(),
        "prompt": (
            f"Design an original Blockbench_sculpted_style-v38.8 rigid vehicle in `Block-DSL v4.1` "
            f"(`dsl_version: \"4.1\"`, `audit_profile: sculpted`, `size_step: 0.5`): **{name.replace('_', ' ').title()}** "
            f"— **Domain D5 (Vehicle), Mood M2** ({ptag} palette). Use chassis-first wheelbase, `inset_carve` recessed "
            f"windshield (`uv_scale: 2.0`), 4-band octagon tires grounded at `y=0`, Ackermann front steering knuckles, "
            f"rotating top turret, `0%` squishy `Sim(3)` metal bulge, and `@behavior` `drive_roll`."
        ),
        "dsl": dsl.strip(),
        "clip_name": "drive_roll",
    }


def _build_mecha_d5_v38(idx: int, *, v38: bool) -> dict[str, Any]:
    specs = (
        ("tungsten_dock_loader_mech", 9.0, 8.5, 6.5, 6.5),
        ("saffron_quarry_strider_bot", 8.5, 8.0, 7.0, 7.0),
        ("cobalt_bridge_welder_frame", 8.0, 8.0, 6.0, 6.5),
        ("crimson_firebreak_walker", 9.5, 9.0, 6.5, 6.5),
        ("slate_survey_biped_unit", 7.5, 7.5, 6.5, 6.5),
        ("amber_salvage_gantry_mech", 9.0, 8.5, 7.0, 6.5),
        ("jade_greenhouse_tender_bot", 7.5, 7.5, 6.0, 6.0),
        ("copper_foundry_anvil_frame", 10.0, 9.0, 6.5, 6.0),
    )
    name, tw, th, leg_u, leg_l = specs[idx]
    ptag, hdr = _header_and_palette(idx + 5)
    sg_chest = "\n      sculpt_group: armor_pack" if v38 else ""
    uv_vis = ", uv_scale: 2.0" if v38 else ""
    bio_sp = "\n      bio_joint: spine_vertebra" if v38 else ""
    bio_sh = "\n        bio_joint: ball_shoulder" if v38 else ""
    bio_el = "\n          bio_joint: hinge_elbow" if v38 else ""
    bio_fg = "\n            bio_joint: finger_flex" if v38 else ""
    bio_hp = "\n      bio_joint: ball_hip" if v38 else ""
    bio_kn = "\n        bio_joint: hinge_knee" if v38 else ""
    visor = (
        f"box cockpit_visor inset_carve(base_box=torso_armor, anchor=[0, 0.25, 1], protrude=0.5) {{ size: [{_fmt_num(_snap05(tw*0.6, 4.0))}, 3, 1.5], material: eye_glow{uv_vis}, sculpt_group: armor_pack }}"
        if v38
        else f"box cockpit_visor place(base_box=torso_armor, anchor=[0, 0.25, 1], align=[0, 0, -1]) {{ size: [{_fmt_num(_snap05(tw*0.6, 4.0))}, 3, 1], material: eye_glow }}"
    )
    clip_hdr = "duration_s: 1.4, loop: true, in_place: true, head_stabilize: 0.85, auto_bio_limits: true" if v38 else "duration_s: 1.4, loop: true"
    gait_opts = "pattern: biped_alternate, froude: 0.20, duty: 0.64, lift_ratio: 0.14, in_place: true, head_stabilize: 0.85" if v38 else "pattern: biped_alternate, froude: 0.20, duty: 0.64"
    dsl = f"""{hdr}
@rig {{
  root bone pelvis {{
    pivot_local: [0, 15, 0]
    box hip_frame {{ size: [{_fmt_num(_snap05(tw*0.8, 5.5))}, 3.5, 5], material: dark }}
    bone chest attach(parent_box=hip_frame, anchor=[0, 1, 0], align=[0, -1, 0]) {{{bio_sp}{sg_chest}
      box torso_armor {{ size: [{_fmt_num(tw)}, {_fmt_num(th)}, 6.5], material: primary }}
      {visor}
      bone arm_l attach(parent_box=torso_armor, anchor=[-1, 0.55, 0], align=[1, 0.5, 0]) mirror_x(arm_r) {{{bio_sh}
        box pauldron_arm {{ size: [3, 5.5, 3.5], material: secondary }}
        bone forearm attach(parent_box=pauldron_arm, anchor=[0, -1, 0], align=[0, 1, 0]) {{{bio_el}
          box actuator_arm {{ size: [2.5, 5.5, 2.5], material: dark }}
          bone clamp_finger attach(parent_box=actuator_arm, anchor=[0, -1, 0.3], align=[0, 1, 0]) {{{bio_fg}
            box gripper {{ size: [1.5, 2, 1.5], material: secondary }}
          }}
        }}
      }}
    }}
    bone leg_l attach(parent_box=hip_frame, anchor=[-0.6, -1, 0], align=[0, 1, 0]) mirror_x(leg_r) {{{bio_hp}
      box thigh_strut {{ size: [3, {_fmt_num(leg_u)}, 3.5], material: primary }}
      bone shin attach(parent_box=thigh_strut, anchor=[0, -1, 0], align=[0, 1, 0]) {{{bio_kn}
        rest_euler: [8, 0, 0]
        box lower_strut {{ size: [2.5, {_fmt_num(leg_l)}, 3], material: dark }}
        bone foot attach(parent_box=lower_strut, anchor=[0, -1, 0], align=[0, 1, 0]) {{
          box foot_pad {{ size: [3.5, 2, 5], material: secondary }}
        }}
      }}
    }}
  }}
}}
@behavior {{
  clip mech_march {{
    {clip_hdr}
    gait walker_stride {{ {gait_opts} }}
  }}
}}"""
    return {
        "asset_id": f"v38_mecha_{name}",
        "family": "D5_mecha_walker",
        "domain": "D5",
        "mood": "M2",
        "polarity": "mechanical_rigid",
        "palette_tag": ptag,
        "title": name.replace("_", " ").title(),
        "prompt": (
            f"Design an original Blockbench_sculpted_style-v38.8 bipedal utility mech in `Block-DSL v4.1` "
            f"(`dsl_version: \"4.1\"`, `audit_profile: sculpted`, `size_step: 0.5`): **{name.replace('_', ' ').title()}** "
            f"— **Domain D5 (Mecha), Mood M2** ({ptag} palette). Use `inset_carve` glazed cockpit visor (`uv_scale: 2.0`), "
            f"`sculpt_group` chest armor, `bio_joint` (`ball_hip`, `hinge_knee`, `ball_shoulder`, `hinge_elbow`, `finger_flex`), "
            f"grounded foot pads at `y=0`, `0%` `Sim(3)` metal bulge, and `@behavior` `gait` (`biped_alternate`, `in_place: true`)."
        ),
        "dsl": dsl.strip(),
        "clip_name": "mech_march",
    }


def _build_humanoid_d3_v38(idx: int, *, v38: bool) -> dict[str, Any]:
    specs = (
        ("terracotta_kiln_golem", 8.0, 8.5, 6.0, 6.0),
        ("cobalt_observatory_automaton", 7.0, 8.0, 6.5, 6.5),
        ("emerald_botanist_warden", 7.0, 8.0, 6.5, 6.5),
        ("ochre_lantern_sentinel", 7.5, 8.5, 6.5, 6.5),
        ("ivory_clockwork_scribe", 6.5, 7.5, 6.0, 6.0),
        ("crimson_blacksmith_knight", 8.5, 8.5, 6.5, 6.5),
        ("indigo_cartographer_sage", 6.5, 8.0, 6.5, 6.5),
        ("amber_apiary_keeper", 7.5, 8.0, 6.0, 6.0),
    )
    name, tw, th, leg_u, leg_l = specs[idx]
    ptag, hdr = _header_and_palette(idx + 6)
    sg_head = "\n          sculpt_group: face_pack" if v38 else ""
    uv_eye = ", uv_scale: 2.0" if v38 else ""
    bio_sp = "\n      bio_joint: spine_vertebra" if v38 else ""
    bio_nk = "\n        bio_joint: neck_vertebra" if v38 else ""
    bio_hd = "\n          bio_joint: head_atlas" if v38 else ""
    bio_sh = "\n        bio_joint: ball_shoulder" if v38 else ""
    bio_el = "\n          bio_joint: hinge_elbow" if v38 else ""
    bio_fg = "\n            bio_joint: finger_flex" if v38 else ""
    bio_hp = "\n      bio_joint: ball_hip" if v38 else ""
    bio_kn = "\n        bio_joint: hinge_knee" if v38 else ""
    eye_l = (
        f"box eye_l inset_carve(base_box=skull, anchor=[-0.45, 0.1, 1], protrude=0.5) {{ size: [1.5, 1, 1], material: eye_glow{uv_eye}, sculpt_group: face_pack }}"
        if v38
        else "box eye_l place(base_box=skull, anchor=[-0.45, 0.1, 1], align=[0, 0, -1]) { size: [1.5, 1, 0.5], material: eye_glow }"
    )
    eye_r = (
        f"box eye_r inset_carve(base_box=skull, anchor=[0.45, 0.1, 1], protrude=0.5) {{ size: [1.5, 1, 1], material: eye_glow{uv_eye}, sculpt_group: face_pack }}"
        if v38
        else "box eye_r place(base_box=skull, anchor=[0.45, 0.1, 1], align=[0, 0, -1]) { size: [1.5, 1, 0.5], material: eye_glow }"
    )
    clip_hdr = "duration_s: 1.2, loop: true, in_place: true, head_stabilize: 0.85, auto_bio_limits: true" if v38 else "duration_s: 1.2, loop: true"
    gait_opts = "pattern: biped_alternate, froude: 0.22, duty: 0.62, lift_ratio: 0.14, in_place: true, head_stabilize: 0.85" if v38 else "pattern: biped_alternate, froude: 0.22, duty: 0.62"
    dsl = f"""{hdr}
@rig {{
  root bone pelvis {{
    pivot_local: [0, 14, 0]
    box hip_core {{ size: [{_fmt_num(_snap05(tw*0.85, 5.5))}, 3, 4], material: dark }}
    bone chest attach(parent_box=hip_core, anchor=[0, 1, 0], align=[0, -1, 0]) {{{bio_sp}
      box torso_box {{ size: [{_fmt_num(tw)}, {_fmt_num(th)}, 4.5], material: primary }}
      bone neck attach(parent_box=torso_box, anchor=[0, 1, 0], align=[0, -1, 0]) {{{bio_nk}
        box neck_box {{ size: [3, 2, 3], material: secondary }}
        bone head attach(parent_box=neck_box, anchor=[0, 1, 0], align=[0, -1, 0]) {{{bio_hd}{sg_head}
          box skull {{ size: [5, 5, 5], material: primary }}
          {eye_l}
          {eye_r}
        }}
      }}
      bone arm_l attach(parent_box=torso_box, anchor=[-1, 0.7, 0], align=[1, 1, 0]) mirror_x(arm_r) {{{bio_sh}
        box upper_arm {{ size: [2.5, 5, 2.5], material: secondary }}
        bone forearm attach(parent_box=upper_arm, anchor=[0, -1, 0], align=[0, 1, 0]) {{{bio_el}
          box lower_arm {{ size: [2, 5, 2], material: primary }}
          bone finger_index attach(parent_box=lower_arm, anchor=[0, -1, 0.4], align=[0, 1, 0]) {{{bio_fg}
            box phalanx {{ size: [1, 2, 1], material: dark }}
          }}
        }}
      }}
    }}
    bone leg_l attach(parent_box=hip_core, anchor=[-0.55, -1, 0], align=[0, 1, 0]) mirror_x(leg_r) {{{bio_hp}
      box thigh {{ size: [2.5, {_fmt_num(leg_u)}, 2.5], material: primary }}
      bone shin attach(parent_box=thigh, anchor=[0, -1, 0], align=[0, 1, 0]) {{{bio_kn}
        rest_euler: [8, 0, 0]
        box calf {{ size: [2, {_fmt_num(leg_l)}, 2], material: dark }}
        bone boot attach(parent_box=calf, anchor=[0, -1, 0], align=[0, 1, 0]) {{
          box sole {{ size: [2.5, 1.5, 3.5], material: secondary }}
        }}
      }}
    }}
  }}
}}
@behavior {{
  clip walk_human {{
    {clip_hdr}
    gait human_stride {{ {gait_opts} }}
  }}
}}"""
    return {
        "asset_id": f"v38_humanoid_{name}",
        "family": "D3_humanoid_guardian",
        "domain": "D3",
        "mood": "M3",
        "polarity": "pole_a_stylized_humanoid",
        "palette_tag": ptag,
        "title": name.replace("_", " ").title(),
        "prompt": (
            f"Design an original Blockbench_sculpted_style-v38.8 stylized humanoid in `Block-DSL v4.1` "
            f"(`dsl_version: \"4.1\"`, `audit_profile: sculpted`, `size_step: 0.5`): **{name.replace('_', ' ').title()}** "
            f"({ptag} palette). Include full `bio_joint` annotations (`spine_vertebra`, `neck_vertebra`, `head_atlas`, "
            f"`ball_shoulder`, `hinge_elbow`, `finger_flex`, `ball_hip`, `hinge_knee`), `inset_carve` focal eyes "
            f"(`uv_scale: 2.0`), grounded boots at `y=0`, and `@behavior` `gait` (`biped_alternate`, `in_place: true`, `head_stabilize: 0.85`)."
        ),
        "dsl": dsl.strip(),
        "clip_name": "walk_human",
    }


def _build_arthropod_d2_v38(idx: int, *, v38: bool) -> dict[str, Any]:
    specs = (
        ("amber_ambergris_scarab", 8.0, 4.5, 11.0, 6),
        ("cobalt_tidepool_hermit_crab", 8.5, 5.0, 10.0, 6),
        ("jade_moss_centipede", 6.0, 3.5, 14.0, 8),
        ("copper_canyon_scorpion", 7.0, 4.0, 12.0, 8),
        ("violet_cavern_mantis", 6.5, 4.5, 11.5, 6),
        ("ochre_dune_trilobite", 9.0, 4.0, 12.5, 6),
        ("crimson_reef_lobster", 7.5, 4.5, 13.0, 8),
        ("slate_shale_isopod", 8.5, 4.5, 12.0, 8),
    )
    name, bw, bh, bl, n_legs = specs[idx]
    ptag, hdr = _header_and_palette(idx + 7)
    sg_root = "\n    sculpt_group: carapace_pack" if v38 else ""
    sg_box = ", sculpt_group: carapace_pack" if v38 else ""
    sg_head = "\n      sculpt_group: head_pack" if v38 else ""
    uv_eye = ", uv_scale: 2.0" if v38 else ""
    shell_ridge = (
        f"box dorsal_plate inset_carve(base_box=thorax, anchor=[0, 1, 0], protrude=1.5) {{ size: [{_fmt_num(_snap05(bw*0.7, 4.0))}, 2, {_fmt_num(_snap05(bl*0.7, 6.0))}], material: secondary{sg_box} }}"
        if v38
        else f"box dorsal_plate place(base_box=thorax, anchor=[0, 1, 0], align=[0, -1, 0]) {{ size: [{_fmt_num(_snap05(bw*0.7, 4.0))}, 1.5, {_fmt_num(_snap05(bl*0.7, 6.0))}], material: secondary }}"
    )
    eye_l = (
        f"box eye_l inset_carve(base_box=head_shell, anchor=[-1, 0.2, 0.4], protrude=0.5) {{ size: [1.5, 1.5, 1.5], material: eye_glow{uv_eye}, sculpt_group: head_pack }}"
        if v38
        else "box eye_l place(base_box=head_shell, anchor=[-1, 0.2, 0.4], align=[1, 0, 0]) { size: [1, 1.5, 1.5], material: eye_glow }"
    )
    eye_r = (
        f"box eye_r inset_carve(base_box=head_shell, anchor=[1, 0.2, 0.4], protrude=0.5) {{ size: [1.5, 1.5, 1.5], material: eye_glow{uv_eye}, sculpt_group: head_pack }}"
        if v38
        else "box eye_r place(base_box=head_shell, anchor=[1, 0.2, 0.4], align=[-1, 0, 0]) { size: [1, 1.5, 1.5], material: eye_glow }"
    )
    z_anchors = (0.65, 0.0, -0.65) if n_legs == 6 else (0.72, 0.24, -0.24, -0.72)
    leg_blocks: list[str] = []
    for li, za in enumerate(z_anchors, start=1):
        leg_blocks.append(
            f"""    bone leg{li}_l attach(parent_box=thorax, anchor=[-1, -0.4, {za}], align=[1, 0.3, 0]) mirror_x(leg{li}_r) {{
      box femur_{li} {{ size: [3.5, 1.5, 1.5], material: primary, rotate: [0, 0, 22] }}
      bone tibia_{li} attach(parent_box=femur_{li}, anchor=[-1, -0.5, 0], align=[0.5, 1, 0]) {{
        box tarsus_{li} {{ size: [1.5, 4.5, 1.5], material: dark }}
      }}
    }}"""
        )
    legs_str = "\n".join(leg_blocks)
    clip_hdr = "duration_s: 1.4, loop: true, in_place: true, auto_bio_limits: true" if v38 else "duration_s: 1.4, loop: true"
    gait_opts = "pattern: metachronal_wave, froude: 0.20, duty: 0.68, lift_ratio: 0.16, in_place: true" if v38 else "pattern: lateral_sequence, froude: 0.20, duty: 0.68"
    dsl = f"""{hdr}
@rig {{
  root bone thorax_root {{
    pivot_local: [0, 6, 0]{sg_root}
    box thorax {{ size: [{_fmt_num(bw)}, {_fmt_num(bh)}, {_fmt_num(bl)}], material: primary }}
    {shell_ridge}
    bone head attach(parent_box=thorax, anchor=[0, -0.1, 1], align=[0, 0, -1]) {{{sg_head}
      box head_shell {{ size: [{_fmt_num(_snap05(bw*0.65, 4.0))}, 3, 4], material: primary }}
      {eye_l}
      {eye_r}
    }}
{legs_str}
    bone abdomen attach(parent_box=thorax, anchor=[0, 0, -1], align=[0, 0, 1]) {{
      box ab_seg {{ size: [{_fmt_num(_snap05(bw*0.8, 4.5))}, {_fmt_num(_snap05(bh*0.85, 3.0))}, 7], material: secondary }}
    }}
  }}
}}
@behavior {{
  clip crawl_wave {{
    {clip_hdr}
    gait arthropod_crawl {{ {gait_opts} }}
  }}
}}"""
    return {
        "asset_id": f"v38_arthropod_{name}",
        "family": "D2_arthropod_metachronal",
        "domain": "D2",
        "mood": "M2",
        "polarity": "biomech_multi_legged",
        "palette_tag": ptag,
        "title": name.replace("_", " ").title(),
        "prompt": (
            f"Design an original Blockbench_sculpted_style-v38.8 {n_legs}-legged arthropod in `Block-DSL v4.1` "
            f"(`dsl_version: \"4.1\"`, `audit_profile: sculpted`, `size_step: 0.5`): **{name.replace('_', ' ').title()}** "
            f"({ptag} palette). Pack a segmented carapace with `sculpt_group` and `inset_carve` dorsal plate + compound eyes "
            f"(`uv_scale: 2.0`), attach {n_legs} mirrored articulated legs grounded at `y=0`, and animate via `@behavior` "
            f"`gait` (`metachronal_wave`, `in_place: true`, `auto_bio_limits: true`)."
        ),
        "dsl": dsl.strip(),
        "clip_name": "crawl_wave",
    }


def _build_all_64_archetype_specs() -> list[dict[str, Any]]:
    v38 = _synccraft_supports_v38()
    builders = (
        _build_aquatic_swim_v38,
        _build_origami_flyer_v38,
        _build_quadruped_pole_a_v38,
        _build_boss_pole_b_v38,
        _build_vehicle_d5_v38,
        _build_mecha_d5_v38,
        _build_humanoid_d3_v38,
        _build_arthropod_d2_v38,
    )
    out: list[dict[str, Any]] = []
    for bfn in builders:
        for idx in range(8):
            item = bfn(idx, v38=v38)
            item["dsl"] = _ground_dsl_pivot_y(item["dsl"])
            out.append(item)
    return out


def build_v38_64_archetype_corpus(
    *,
    seed: int = 20261170,
    max_visual_renders: int = 16,
    max_repairs: int = 140,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], dict[str, bytes]]], list[dict[str, Any]]]:
    """Build and LIVE-VERIFY all 64 unique `v38.8` archetypes, full-color shaded RGB visual pairs,
    and 14-mutator `v38.8` grammar/semantic repair records.

    Returns:
        `(text_dsl_records, visual_render_pairs, repair_records)`
    """
    ensure_synccraft_importable()
    from synccraft.audit.sync_audit import audit_scene
    from synccraft.dsl.ast_nodes import validate_program
    from synccraft.dsl.parser import parse_program
    from synccraft.kinematics.se3_chain import compile_geometry

    rng = random.Random(seed)
    v38_avail = _synccraft_supports_v38()
    specs = _build_all_64_archetype_specs()

    verified_specs: list[dict[str, Any]] = []
    text_records: list[dict[str, Any]] = []

    zh_prompt_templates = (
        "请用 SyncCraft `Block-DSL v4.1`（`audit_profile: sculpted`, `size_step: 0.5`）设计原创 `v38.8` 风格3D模型：**{title}**（{family}，{pal} 配色）。要求包含 `sculpt_group` 体块叠压、`inset_carve` 凹嵌结构、`uv_scale: 2.0` 高精度焦眼/镜片、`mirror_x` 精确镜像、接地 `y=0` 与 `@behavior` 生物力学动作。",
    )

    for idx, spec in enumerate(specs):
        dsl = spec["dsl"]
        prog = parse_program(dsl)
        validate_program(prog)
        scene = compile_geometry(prog)
        aud = audit_scene(scene, audit_profile="sculpted")
        if not aud.get("passed"):
            raise RuntimeError(f"v38.8 archetype {spec['asset_id']} failed sculpted audit_scene: {aud}")

        # Verify zero banned benchmark leakage
        low_blob = (spec["prompt"] + "\n" + dsl).lower()
        for bad in BANNED_BENCHMARK_SUBSTRINGS:
            if bad in low_blob:
                raise RuntimeError(f"Banned benchmark token {bad!r} in {spec['asset_id']}")

        verified_specs.append(spec)
        use_zh = (idx % 5 == 4)
        lang = "zh" if use_zh else "en"
        prompt = (
            zh_prompt_templates[0].format(title=spec["title"], family=spec["family"], pal=spec["palette_tag"])
            if use_zh
            else spec["prompt"]
        )
        think = (
            f"Design plan ({spec['domain']}/{spec['mood']}, {spec['polarity']}): "
            f"pack compound sculpted masses with `sculpt_group`, recess focal eyes/visors via `inset_carve` "
            f"with `uv_scale: 2.0`, quantize every box dimension strictly to `size_step: 0.5`, enforce exact "
            f"`mirror_x` symmetry and `y=0` ground contact, and bind `@behavior` clip `{spec['clip_name']}`."
        )
        answer = f"<think>\n{think}\n</think>\n```blockdsl\n{dsl}\n```"
        rec = {
            "id": f"stage11_{spec['asset_id']}",
            "source_group": f"stage11_v38_{spec['family']}",
            "language": lang,
            "task": "stage11_v38_sculpted_dsl",
            "visual_kind": None,
            "frames": [],
            "enable_thinking": True,
            "prompt": prompt,
            "answer": answer,
            "target": {"kind": "text", "text": f"```blockdsl\n{dsl}\n```"},
            "provenance": make_clean_room_provenance(
                asset_id=spec["asset_id"],
                title=spec["title"],
                license_spdx="CC0-1.0",
                author="SyncCraft Stage-11 v38.8 Parametric Generator",
            ),
            "metadata": {
                "source": "stage11_v38_64_archetypes",
                "license_spdx": "CC0-1.0",
                "family": spec["family"],
                "domain": spec["domain"],
                "mood": spec["mood"],
                "synccraft_v38_native": v38_avail,
                "real_text_corpus": True,
            },
        }
        text_records.append(rec)

    # Full-Color Shaded RGB Studio Turnaround Visual-Grounding Pairs (`clay=False`, chroma > 10!)
    visual_pairs: list[tuple[dict[str, Any], dict[str, bytes]]] = []
    if max_visual_renders > 0 and verified_specs:
        step = max(1, len(verified_specs) // max_visual_renders)
        picked = verified_specs[::step][:max_visual_renders]
        angles = ((34.0, 18.0), (-36.0, 20.0), (145.0, 22.0))
        for r_i, spec in enumerate(picked):
            prog = parse_program(spec["dsl"])
            az, el = angles[r_i % len(angles)]
            preset = STUDIO_LIGHTING_PRESETS[r_i % len(STUDIO_LIGHTING_PRESETS)]
            im, meta = render_shaded_3d_with_spatial_annotations(
                prog,
                azimuth_deg=az,
                elevation_deg=el,
                size=(384, 384),
                lighting_preset=preset,
            )
            arr = np.asarray(im.convert("RGB"), dtype=np.float32)
            chroma = float(np.abs(arr[:, :, 0] - arr[:, :, 1]).mean() + np.abs(arr[:, :, 1] - arr[:, :, 2]).mean())
            if chroma < 10.5:
                continue
            pts = meta.get("bone_points") or {}
            boxes_2d = meta.get("bone_boxes") or {}
            prim_lines: list[str] = []
            for bn, info in list(pts.items())[:5]:
                prim_lines.append(fmt_point(info["x"], info["y"], info.get("z_tier", "mid"), bn))
            for bn, bb in list(boxes_2d.items())[:3]:
                prim_lines.append(fmt_box(bb[0], bb[1], bb[2], bb[3], bn))
            pt_list = list(pts.items())
            if len(pt_list) >= 2:
                b0, p0 = pt_list[0]
                b1, p1 = pt_list[1]
                prim_lines.append(fmt_dash(p0["x"], p0["y"], p1["x"], p1["y"], f"{b0}_to_{b1}"))
                prim_lines.append(fmt_circle(p0["x"], p0["y"], 38, f"pivot_{b0}"))
            prims_str = " ".join(prim_lines)
            think = (
                f"Inspecting the full-color sculpted 3D render ({spec['domain']}/{spec['mood']}, {spec['polarity']}): "
                f"{prims_str}. The silhouette matches `{spec['title']}` ({spec['family']}) using `sculpt_group` mass-packing, "
                f"`inset_carve` focal eyes/visors (`uv_scale: 2.0`), and exact `size_step: 0.5` Block-DSL v4.1 cuboids."
            )
            ans = f"<think>\n{think}\n</think>\n```blockdsl\n{spec['dsl']}\n```"
            if not check_generation_collapse_health(ans, expect_thinking=True, max_rep_4gram=0.140)["healthy"]:
                continue
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            mem = f"v38_render_{spec['asset_id']}.png"
            vrow = {
                "id": f"stage11_v38_vis_{spec['asset_id']}",
                "source_group": f"stage11_v38_vis_{spec['family']}",
                "language": "en",
                "task": "stage11_v38_multiview_inverse_dsl",
                "visual_kind": "independent_image",
                "frames": [mem],
                "enable_thinking": True,
                "prompt": (
                    f"Analyze this full-color shaded 3D studio render of **{spec['title']}** ({spec['family']}) using "
                    "`<point>`, `<box>`, `<dash>`, and `<circle>` visual primitives inside `<think>...</think>`, then "
                    "reconstruct its compilable `Block-DSL v4.1` (`v38.8`, `size_step: 0.5`) program."
                ),
                "answer": ans,
                "target": {"kind": "text", "text": f"```blockdsl\n{spec['dsl']}\n```"},
                "metadata": {
                    "source": "stage11_v38_shaded_renders",
                    "license_spdx": "CC0-1.0",
                    "pixel_indispensable": True,
                    "chroma": round(chroma, 2),
                },
            }
            visual_pairs.append((vrow, {mem: buf.getvalue()}))

    # 14-Mutator Solver-Verified v38.8 Grammar & Punctuation Self-Repair Records
    def _mut_texel_size(s: str) -> tuple[str, str]:
        return s.replace("texel_density:", "texel_size:", 1), "Replace invalid `@meta` key `texel_size` with `texel_density`."

    def _mut_root_kw(s: str) -> tuple[str, str]:
        return re.sub(r"\broot\s+bone\b", "root_limb bone", s, count=1), "Use exact `root bone <name>`, never `root_limb bone`."

    def _mut_box_axis(s: str) -> tuple[str, str]:
        return re.sub(r"(box\s+[a-zA-Z0-9_]+[^{]*\{[^}]*size:\s*\[[^\]]+\])", r"\1, axis: [0, 1, 0]", s, count=1), "Remove illegal `axis` key from `box { ... }`."

    def _mut_palette_rgb(s: str) -> tuple[str, str]:
        return re.sub(r"\bramp\([^)]+\)", "rgb(120, 140, 190)", s, count=1), "Every `@palette` entry must use `ramp(base_lch=...)`, never `rgb(...)`."

    def _mut_meta_author(s: str) -> tuple[str, str]:
        return s.replace("audit_profile: sculpted", 'audit_profile: sculpted, author: "artist"', 1), "Remove unsupported `@meta` key `author`."

    def _mut_size_step_025(s: str) -> tuple[str, str]:
        return re.sub(
            r"size:\s*\[\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\]",
            lambda m: f"size: [{round(float(m.group(1))+0.25, 2)}, {m.group(2)}, {m.group(3)}]",
            s,
            count=1,
        ), "Quantize all `size: [x, y, z]` components to exact multiples of `size_step: 0.5`."

    def _mut_missing_comma(s: str) -> tuple[str, str]:
        return re.sub(r"(material:\s*[a-zA-Z0-9_]+)\s*,\s*(rotate:|prism_group:|uv_scale:|sculpt_group:)", r"\1 \2", s, count=1), "Separate `box` fields strictly with commas (`,`)."

    def _mut_mirror_colon(s: str) -> tuple[str, str]:
        return re.sub(r"\bmirror_x\(([a-zA-Z0-9_]+)\)", r"mirror_x: \1", s, count=1), "Use function syntax `mirror_x(<name>)` with parentheses, never `mirror_x: <name>`."

    def _mut_stray_bracket(s: str) -> tuple[str, str]:
        return re.sub(r"anchor=\[([^\]]+)\]", r"anchor=[\1]]", s, count=1), "Remove stray extra closing bracket `]` from `anchor=[...]` vector."

    def _mut_uv_scale_typo(s: str) -> tuple[str, str]:
        if "uv_scale:" in s:
            return s.replace("uv_scale:", "texel_scale:", 1), "Use exact `v38.8` field name `uv_scale: 2.0`, never `texel_scale`."
        return s.replace("units_per_block:", "block_units:", 1), "Use exact `@meta` key `units_per_block`, never `block_units`."

    def _mut_inset_typo(s: str) -> tuple[str, str]:
        if "inset_carve(" in s:
            return s.replace("inset_carve(", "carve_inset(", 1), "Use exact `v38.8` placement call `inset_carve(base_box=..., anchor=..., protrude=...)`."
        return re.sub(r"(\bbone\s+[a-zA-Z0-9_]+)(\s+attach\()", r"\1 and extra\2", s, count=1), "Remove illegal `and` from `bone` declaration."

    def _mut_sculpt_group_typo(s: str) -> tuple[str, str]:
        if "sculpt_group:" in s:
            return s.replace("sculpt_group:", "carve_group:", 1), "Use exact `v38.8` keyword `sculpt_group:`, never `carve_group:`."
        return s.replace("dsl_version:", "version:", 1), "Use exact `@meta` key `dsl_version`, never `version`."

    def _mut_trailing_semicolon_box(s: str) -> tuple[str, str]:
        return re.sub(r"(material:\s*[a-zA-Z0-9_]+)\s*\}", r"\1; }", s, count=1), "Do not place a stray semicolon `;` before `}` inside `box { ... }`."

    def _mut_dangling_brace(s: str) -> tuple[str, str]:
        return s + "\n}", "Remove unbalanced extra closing brace `}` at end of Block-DSL program."

    mutators = (
        ("texel_size", _mut_texel_size),
        ("root_kw", _mut_root_kw),
        ("box_axis", _mut_box_axis),
        ("palette_rgb", _mut_palette_rgb),
        ("meta_author", _mut_meta_author),
        ("size_step_025", _mut_size_step_025),
        ("missing_comma", _mut_missing_comma),
        ("mirror_colon", _mut_mirror_colon),
        ("stray_bracket", _mut_stray_bracket),
        ("uv_scale_typo", _mut_uv_scale_typo),
        ("inset_typo", _mut_inset_typo),
        ("sculpt_group_typo", _mut_sculpt_group_typo),
        ("semicolon_box", _mut_trailing_semicolon_box),
        ("dangling_brace", _mut_dangling_brace),
    )

    repair_records: list[dict[str, Any]] = []
    for s_i, spec in enumerate(verified_specs):
        dsl = spec["dsl"]
        if len(dsl) > 3600:
            continue
        # Pick 3 distinct mutators per archetype -> up to 192 verified repairs, capped to max_repairs
        for k_off in range(3):
            if len(repair_records) >= max_repairs:
                break
            mname, mfn = mutators[(s_i * 3 + k_off) % len(mutators)]
            broken, hint = mfn(dsl)
            if broken == dsl:
                continue
            try:
                pb = parse_program(broken)
                validate_program(pb)
                continue
            except Exception as exc:
                err_msg = str(exc).strip()[:240]
            pr = (
                f"[SyncCraft Block-DSL v4.1 Compiler Repair — `{mname}`]\n"
                f"The following `Block-DSL v4.1` (`v38.8`) draft failed SyncCraft verification with:\n"
                f"  Error: {err_msg}\n\n"
                f"```blockdsl\n{broken}\n```\n\n"
                "Diagnose the exact Block-DSL v4.1 violation inside `<think>...</think>` and output ONLY the repaired, compilable ```blockdsl ... ``` program."
            )
            th = (
                f"SyncCraft reported: `{err_msg}`.\n"
                f"Exact fix rule: {hint}\n"
                "Preserve all `sculpt_group`, `inset_carve`, `uv_scale`, `bio_joint`, symmetry, and `@behavior` blocks."
            )
            ans = f"<think>\n{th}\n</think>\n```blockdsl\n{dsl}\n```"
            if len(pr) + len(ans) > 9000:
                continue
            h = hashlib.sha256(f"v38_rep:{spec['asset_id']}:{mname}".encode("utf-8")).hexdigest()[:14]
            repair_records.append(
                {
                    "id": f"stage11_v38_repair_{mname}_{h}",
                    "source_group": f"stage11_v38_repair_{mname}",
                    "language": "en",
                    "task": "stage11_v38_grammar_repair",
                    "visual_kind": None,
                    "frames": [],
                    "enable_thinking": True,
                    "prompt": pr,
                    "answer": ans,
                    "target": {"kind": "text", "text": f"```blockdsl\n{dsl}\n```"},
                    "metadata": {
                        "source": "stage11_v38_synccraft_repair",
                        "license_spdx": "CC0-1.0",
                        "mutation": mname,
                        "real_text_corpus": True,
                    },
                }
            )
        if len(repair_records) >= max_repairs:
            break

    rng.shuffle(repair_records)
    emit(
        "stage11_v38_64_archetype_corpus_ready",
        archetypes=len(text_records),
        visual_renders=len(visual_pairs),
        repairs=len(repair_records),
        synccraft_v38_native=v38_avail,
    )
    return text_records, visual_pairs, repair_records
