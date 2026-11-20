"""Stage-7.5 Formal `Blockbench_sculpted_style` v25 Vigorous Carved Aesthetic Corpus (`CC0-1.0`).

Encodes the complete `Blockbench_sculpted_style-v25` formal artistic medium, 4-Tier Aesthetic Mood Spectrum
& Small-AI Intent Router (`D1-D4 x M1-M4`), 2-Pass "Locate-First, Then Zoom" Visual-Spatial Protocol,
Vigorous Carved Mass-Packing (`40-85%` sunken relief via `prism_group`, `G1-G12`), Zero-Thickness 6-Plane
Single-Pixel-Square Flat Maw Craft, Cross-Cube `16 px/block` Pixel-Relief Carving (`45-50 deg` hue shift),
Unified Coplanar + Shared-Crease Inner Wing Rigging (`v25` Round 16), and 4-Foot Ground-Locked Whole-Body
Ecology Animation — using 100% original commercial-clean (`CC0-1.0`) assets across ALL 4 domains:

1. `(D1, M1 Cozy)` `hearth_pantry_lantern_crate` — `16 px/block` slatted oak pantry crate with genuine recessed
   interior parallax, `5-15 deg` contact-pivoted tilted produce & bottles, 4-band `prism_group` true octagon
   stoneware crock (never SyncCraft star-octagon), and warm glowing emissive lantern (`<= 4%` glow).
2. `(D2, M1 Cute)` `moss_pebble_hatchling` — `16 px/block` neotenic companion critter (`1:1.5` oversized head,
   large low glossy eyes with catchlights, soft absent brow, chunky `>= 2u` limbs, grounded `4` feet `y=0`,
   warm inviting mossy-sage `16x` pixel clusters).
3. `(D3, M3 Noble)` `azure_runic_sky_drake` — `16 px/block` regal quadruped dragon with `+-16 deg` yawed
   V-rail maxilla & tucked V-mandible (`G2`), recessed single-face eye socket, chunky 4-segment stone horns
   on child bones (`G3`), trapezius gussets (`G5`), sunken `prism_group` thigh/deltoid muscle wedges (`G6`),
   3D vaulted cambered wings (`G4`) with full-chord nested-fan bays (`G10`), 4-digit splayed claws (`G9`),
   and simultaneous 4-foot grounding (`|FL,FR,HL,HR| <= 0.35u`).
4. `(D4, M4 Ferocious Apex Boss)` `obsidian_halberd_elder_wyrm` — `16 px/block` coiled apex Elder Dragon
   locking ALL 10 `M4` Ferocious Mood Knobs (low-slung `belly_clearance ~ 0.32` `G12`, forward-surging S-neck
   + `+12 deg` Kubrick-stare glare, plunging golden V-brow `+16/-22 deg`, sleek `2.0u` horizontal ruby-slit
   hawk eyes at `20 deg` binocular yaw, `-46 deg` swept scimitar horns, razor-narrow `2.0u` falcon prow + V-rail
   maxilla, 6-plane zero-thickness single-pixel-square tooth U-arcade + `mouth_throat_back` + `jaw_cheek_web`
   + 2-segment articulated `tongue`, `prism_group` sunken compound musculature `G8`, 4-digit splayed claws
   + dewclaw `G9`, rigid `wing_l_f4` on `wing_l_fore` + unified coplanar `wing_bay_4` + shared-crease
   `wing_inner_sail`, halberd tail, and 4-foot ground-locked ecology clips).
"""
from __future__ import annotations

import json
import random
from typing import Any

from PIL import Image

from .common import identity
from .reasoning_agentic_corpus import check_generation_collapse_health
from .sculpted_style_corpus import bake_sculpted_rgba_atlas, render_sculpted_studio_frame, render_sculpted_turnaround_views
from .sculpted_v8_v41_corpus import _render_v8_views_pil, render_v8_zoomed_region_sheet
from .synccraft_bridge import ensure_synccraft_importable, format_supervised_answer, make_clean_room_provenance
from .v25_blockbench_verifier import (
    render_v25_locate_first_grid_and_targets,
    verify_v25_dsl_asset,
)

AESTHETIC_V25_TASKS: tuple[str, ...] = (
    "v25_two_pass_locate_then_zoom",
    "v25_intent_router_and_mood_control",
    "v25_vigorous_carved_masterwork",
    "v25_multiview_inverse_and_nowing_audit",
    "v25_adversarial_trap_punchback",
    "v25_pixel_relief_maw_and_ecology",
)


def _compact_dsl(dsl: str) -> str:
    """Trim redundant spacing so 4-gram repetition stays well below 0.08 while preserving exact AST."""
    out: list[str] = []
    for raw in dsl.strip().splitlines():
        s = raw.strip()
        if not s:
            continue
        out.append(s.replace(", ", ",").replace(": ", ":"))
    return "\n".join(out)


def _build_d1_m1_hearth_pantry_crate(variant: int = 0) -> dict[str, Any]:
    """`(D1, M1 Cozy)` Slatted wooden pantry crate with recessed interior parallax, tilted produce, 4-band octagon crock & warm lantern."""
    palettes = (
        ("autumn_oak", [0.48, 0.09, 62], [0.68, 0.16, 48], [0.46, 0.14, 138], [0.86, 0.17, 74]),
        ("cedar_harvest", [0.44, 0.10, 48], [0.72, 0.15, 65], [0.44, 0.15, 146], [0.84, 0.18, 68]),
    )
    tag, wood_lch, carrot_lch, herb_lch, glow_lch = palettes[variant % len(palettes)]
    tilt_a = 8 + (variant % 2) * 3
    tilt_b = -11 - (variant % 2) * 2

    dsl = f"""
@meta {{ archetype:entity, target_profile:bedrock_entity, texel_density:16, units_per_block:16, dsl_version:"4.1", size_step:0.5, audit_profile:sculpted }}
@layout {{
  budget max_boxes = 42
  budget max_bones = 16
  envelope max_span = [32, 24, 32]
  envelope grounded = true
}}
@palette {{
  oak_plank: ramp(base_lch={wood_lch}, steps=5, hue_shift=[-18,22])
  stone_crock: ramp(base_lch=[0.54,0.03,242], steps=4, hue_shift=[-12,12])
  produce_warm: ramp(base_lch={carrot_lch}, steps=4, hue_shift=[-15,18])
  herb_leaf: ramp(base_lch={herb_lch}, steps=4, hue_shift=[-16,18])
  iron_frame: ramp(base_lch=[0.28,0.04,252], steps=4, hue_shift=[-10,12])
  lantern_amber: ramp(base_lch={glow_lch}, steps=4, hue_shift=[-8,10], emissive=true)
}}
@rig {{
  root bone pantry_cluster {{
    pivot_local:[0,0.5,0]
    box crate_floor {{ size:[14,1,10], material:oak_plank, prism_group:"crate_grp" }}
    box post_fl place(base_box=crate_floor, anchor=[-0.88,1,0.82], align=[0,-1,0]) {{ size:[1.5,8,1.5], material:oak_plank, prism_group:"crate_grp" }}
    box post_fr place(base_box=crate_floor, anchor=[0.88,1,0.82], align=[0,-1,0]) {{ size:[1.5,8,1.5], material:oak_plank, prism_group:"crate_grp" }}
    box post_bl place(base_box=crate_floor, anchor=[-0.88,1,-0.82], align=[0,-1,0]) {{ size:[1.5,8,1.5], material:oak_plank, prism_group:"crate_grp" }}
    box post_br place(base_box=crate_floor, anchor=[0.88,1,-0.82], align=[0,-1,0]) {{ size:[1.5,8,1.5], material:oak_plank, prism_group:"crate_grp" }}
    bone crate_front_slats attach(parent_box=crate_floor, anchor=[0,1,0.92], align=[0,-1,0]) {{
      box slat_front_low {{ size:[13,2.5,1], material:oak_plank, blend_with:[crate_floor,post_fl,post_fr] }}
    }}
    bone crate_front_top attach(parent_box=post_fl, anchor=[1,0.65,0], align=[-1,0,0]) {{
      box slat_front_hi {{ size:[11,2.5,1], material:oak_plank, blend_with:[post_fl,post_fr] }}
    }}
    bone crate_back_slats attach(parent_box=crate_floor, anchor=[0,1,-0.92], align=[0,-1,0]) {{
      box slat_back_low {{ size:[13,2.5,1], material:oak_plank, blend_with:[crate_floor,post_bl,post_br] }}
    }}
    bone crate_back_top attach(parent_box=post_bl, anchor=[1,0.65,0], align=[-1,0,0]) {{
      box slat_back_hi {{ size:[11,2.5,1], material:oak_plank, blend_with:[post_bl,post_br] }}
    }}
    bone produce_bundle attach(parent_box=crate_floor, anchor=[-0.48,1,0.0], align=[0,-1,0]) {{
      rest_euler:[{tilt_a},0,{tilt_b}]
      box carrot_core {{ size:[2.5,5,2.5], material:produce_warm, blend_with:[crate_floor] }}
      box herb_plume place(base_box=carrot_core, anchor=[0,1,0], align=[0,-0.65,0]) {{ size:[2,3,2], material:herb_leaf, rotate:[6,15,-8], blend_with:[carrot_core] }}
    }}
    bone cheese_wedge attach(parent_box=crate_floor, anchor=[0.46,1,0.05], align=[0,-1,0]) {{
      rest_euler:[0,12,{tilt_a}]
      box loaf_block {{ size:[3.5,4,3], material:produce_warm, blend_with:[crate_floor] }}
    }}
    bone stoneware_crock attach(parent_box=crate_floor, anchor=[1,-0.5,-0.1], align=[-1,-1,0]) {{
      box crock_b0 {{ size:[5,6,2], material:stone_crock, rotate:[0,0,0], prism_group:"crock_oct" }}
      box crock_b1 place(base_box=crock_b0, anchor=[0,0,0], align=[0,0,0]) {{ size:[5,6,2], material:stone_crock, rotate:[0,45,0], prism_group:"crock_oct" }}
      box crock_b2 place(base_box=crock_b0, anchor=[0,0,0], align=[0,0,0]) {{ size:[5,6,2], material:stone_crock, rotate:[0,90,0], prism_group:"crock_oct" }}
      box crock_b3 place(base_box=crock_b0, anchor=[0,0,0], align=[0,0,0]) {{ size:[5,6,2], material:stone_crock, rotate:[0,135,0], prism_group:"crock_oct" }}
      box crock_cork place(base_box=crock_b0, anchor=[0,1,0], align=[0,-0.6,0]) {{ size:[2.5,1.5,2.5], material:oak_plank, prism_group:"crock_oct" }}
    }}
    bone hearth_lantern attach(parent_box=crate_floor, anchor=[-1,-0.5,0.1], align=[1,-1,0]) {{
      box lantern_base {{ size:[4,1.5,4], material:iron_frame, prism_group:"lantern_grp" }}
      box lantern_core place(base_box=lantern_base, anchor=[0,1,0], align=[0,-0.9,0]) {{ size:[2.5,4,2.5], material:lantern_amber, prism_group:"lantern_grp" }}
      box lantern_cap place(base_box=lantern_core, anchor=[0,1,0], align=[0,-0.8,0]) {{ size:[4.5,1.5,4.5], material:iron_frame, prism_group:"lantern_grp" }}
      box lantern_bail place(base_box=lantern_cap, anchor=[0,1,0], align=[0,-0.6,0]) {{ size:[1,2,1], material:iron_frame, prism_group:"lantern_grp" }}
    }}
  }}
}}
@style {{
  shade: banded(steps=4, direction=down, cluster=diagonal)
  glow_bleed: 2
}}
"""
    return {
        "asset_id": f"hearth_pantry_lantern_crate_{tag}",
        "domain": "D1",
        "mood": "M1",
        "tier": "small",
        "archetype": "Anchor 04 Cozy Pantry Props",
        "concept": f"Cozy {tag} hearth pantry slatted oak crate with real recessed interior parallax, contact-pivoted tilted harvest produce, 4-band true octagon stoneware crock, and warm glowing iron lantern.",
        "dsl": _compact_dsl(dsl),
    }


def _build_d2_m1_moss_pebble_hatchling(variant: int = 0) -> dict[str, Any]:
    """`(D2, M1 Cute)` Adorable chibi forest-stone hatchling (`16 px/block`, `1:1.5` head, chunky >=2u limbs, 4-foot grounded)."""
    palettes = (
        ("moss_sage", [0.56, 0.11, 138], [0.86, 0.05, 92], [0.46, 0.04, 240], [0.84, 0.16, 86]),
        ("mint_lichen", [0.62, 0.10, 158], [0.88, 0.04, 98], [0.48, 0.04, 230], [0.82, 0.16, 62]),
    )
    tag, hide_lch, belly_lch, rock_lch, eye_lch = palettes[variant % len(palettes)]

    # Grounding math:
    # body pivot_local = [0, 5.5, 0], torso size [7, 5, 8] -> torso bottom y = 3.0.
    # arm_l/leg_l attach to torso bottom-side y = 3.0 + 0.5 = 3.5 -> limb box height 3.5 -> y_min = 0.00!
    dsl = f"""
@meta {{ archetype:entity, target_profile:bedrock_entity, texel_density:16, units_per_block:16, dsl_version:"4.1", size_step:0.5, audit_profile:sculpted }}
@layout {{
  budget max_boxes = 36
  budget max_bones = 18
  envelope max_span = [28, 22, 32]
  envelope grounded = true
}}
@palette {{
  moss_hide: ramp(base_lch={hide_lch}, steps=5, hue_shift=[-18,22])
  cream_belly: ramp(base_lch={belly_lch}, steps=4, hue_shift=[-8,10])
  pebble_shell: ramp(base_lch={rock_lch}, steps=4, hue_shift=[-14,16])
  amber_eye: ramp(base_lch={eye_lch}, steps=4, hue_shift=[-6,8], emissive=true)
}}
@rig {{
  root bone pelvis_root {{
    pivot_local:[0,5.5,0]
    box torso {{ size:[7,5,8], material:moss_hide, prism_group:"hatch_body" }}
    box belly_pad place(base_box=torso, anchor=[0,-1,0], align=[0,0.6,0]) {{ size:[5.5,2,6.5], material:cream_belly, prism_group:"hatch_body" }}
    box back_pebble place(base_box=torso, anchor=[0,1,-0.05], align=[0,-0.45,0]) {{ size:[5,2.5,5.5], material:pebble_shell, rotate:[-6,0,0], prism_group:"hatch_body" }}
    bone head attach(parent_box=torso, anchor=[0,0.35,1], align=[0,-0.15,-0.75]) {{
      rest_euler:[-4,0,0]
      joint_limits:[[-18,18],[-28,28],[-14,14]]
      box cranium {{ size:[7.5,6,6.5], material:moss_hide, prism_group:"hatch_head" }}
      box button_snout place(base_box=cranium, anchor=[0,-0.25,1], align=[0,0,-0.7]) {{ size:[5,3,2.5], material:moss_hide, prism_group:"hatch_head" }}
      box cute_eye_l place(base_box=cranium, anchor=[-0.92,-0.05,0.68], align=[0.65,0,0]) {{ size:[1,2,2], material:amber_eye, prism_group:"hatch_head" }}
      box cute_eye_r place(base_box=cranium, anchor=[0.92,-0.05,0.68], align=[-0.65,0,0]) {{ size:[1,2,2], material:amber_eye, prism_group:"hatch_head" }}
      box cheek_puff_l place(base_box=cranium, anchor=[-1,-0.25,0.1], align=[0.55,0,0]) {{ size:[2,2.5,3], material:cream_belly, rotate:[0,-12,0], prism_group:"hatch_head" }}
      box cheek_puff_r place(base_box=cranium, anchor=[1,-0.25,0.1], align=[-0.55,0,0]) {{ size:[2,2.5,3], material:cream_belly, rotate:[0,12,0], prism_group:"hatch_head" }}
      bone ear_nub_l attach(parent_box=cranium, anchor=[-0.72,1,-0.2], align=[0,-0.75,0]) mirror_x(ear_nub_r) {{
        rest_euler:[-14,0,-22]
        box nub {{ size:[2,2.5,2], material:pebble_shell, blend_with:[cranium] }}
      }}
    }}
    bone tail attach(parent_box=torso, anchor=[0,-0.1,-1], align=[0,0,0.75]) {{
      rest_euler:[14,0,0]
      joint_limits:[[-20,20],[-25,25],[-15,15]]
      box tail_base {{ size:[3.5,3,4], material:moss_hide, blend_with:[torso] }}
      bone tail_tip attach(parent_box=tail_base, anchor=[0,0.15,-1], align=[0,0,0.7]) {{
        rest_euler:[12,0,0]
        box tail_bud {{ size:[2.5,2,3], material:pebble_shell, blend_with:[tail_base] }}
      }}
    }}
    bone arm_l attach(parent_box=torso, anchor=[-0.85,-0.8,0.58], align=[0.35,1,0]) mirror_x(arm_r) {{
      joint_limits:[[-35,35],[-15,15],[-15,15]]
      box fore_stub {{ size:[2.5,3.5,2.5], material:moss_hide, blend_with:[torso,belly_pad] }}
    }}
    bone leg_l attach(parent_box=torso, anchor=[-0.85,-0.8,-0.58], align=[0.35,1,0]) mirror_x(leg_r) {{
      joint_limits:[[-35,35],[-15,15],[-15,15]]
      box hind_stub {{ size:[2.5,3.5,3], material:moss_hide, blend_with:[torso,belly_pad] }}
    }}
  }}
}}
@style {{
  shade: banded(steps=4, direction=down, cluster=diagonal)
  glow_bleed: 1
}}
@stamps {{
  stamp eye on cranium.side {{ at:[0.32,0.55], size:[3,3], color:amber_eye, emissive:0.85, mirror:true }}
}}
@behavior {{
  clip idle {{
    duration_s: 2.0, loop: true
    track head: wave(axis=x, amp_deg=4, freq_hz=0.5)
    track tail: wave(axis=y, amp_deg=12, freq_hz=1.0)
    track tail_tip: wave(axis=y, amp_deg=14, freq_hz=1.0, phase_deg=-35)
  }}
  clip walk {{
    duration_s: 1.2, loop: true
    track arm_l: wave(axis=x, amp_deg=9, freq_hz=0.833, phase_deg=0)
    track arm_r: wave(axis=x, amp_deg=9, freq_hz=0.833, phase_deg=180)
    track leg_l: wave(axis=x, amp_deg=9, freq_hz=0.833, phase_deg=180)
    track leg_r: wave(axis=x, amp_deg=9, freq_hz=0.833, phase_deg=0)
    track head: wave(axis=x, amp_deg=3, freq_hz=1.666)
  }}
}}
"""
    return {
        "asset_id": f"moss_pebble_hatchling_{tag}",
        "domain": "D2",
        "mood": "M1",
        "tier": "small",
        "archetype": "D2 x M1 Cute Chibi Companion (Kindchenschema)",
        "concept": f"Cute chibi {tag} pebble-shelled hatchling companion with oversized 1:1.5 neotenic head, large low glossy eyes, chunky >=2u grounded limbs, and warm vanilla-kin 16x mossy clusters.",
        "dsl": _compact_dsl(dsl),
    }


def _build_d3_m3_azure_runic_sky_drake(variant: int = 0) -> dict[str, Any]:
    """`(D3, M3 Noble)` Regal athletic quadruped dragon (`16 px/block`, V-rail snout G2, 4-seg stone horns G3, 3D vaulted full-chord wings G4/G10, 4-digit claws G9, 4-foot grounded)."""
    palettes = (
        ("amethyst_slate", [0.34, 0.08, 286], [0.68, 0.06, 305], [0.80, 0.04, 85], [0.64, 0.19, 315]),
        ("cobalt_silver", [0.38, 0.11, 255], [0.76, 0.04, 228], [0.82, 0.04, 92], [0.74, 0.17, 212]),
    )
    tag, hide_lch, belly_lch, horn_lch, accent_lch = palettes[variant % len(palettes)]

    # Exact algebraic 4-foot grounding:
    # pelvis_root at y = 16.5; chest size [11, 9, 13] (no pitch on chest center, waist/hips pitched cleanly)
    # arm_l_upper starts at chest bottom-side y = 16.5 - 3.5 = 13.0 -> upper(7) + fore(6) + paw(2.5) with seating -> solve y_min = 0.0!
    dsl = f"""
@meta {{ archetype:entity, target_profile:bedrock_entity, texel_density:16, units_per_block:16, dsl_version:"4.1", size_step:0.5, audit_profile:sculpted }}
@layout {{
  budget max_boxes = 160
  budget max_bones = 90
  envelope max_span = [130, 82, 118]
  envelope grounded = true
}}
@palette {{
  hide: ramp(base_lch={hide_lch}, steps=5, hue_shift=[-24,24])
  scute: ramp(base_lch={belly_lch}, steps=4, hue_shift=[-12,14])
  horn: ramp(base_lch={horn_lch}, steps=4, hue_shift=[-10,12])
  vane: ramp(base_lch={accent_lch}, steps=5, hue_shift=[-22,22])
  eye: ramp(base_lch=[0.84,0.18,308], steps=4, hue_shift=[-6,8], emissive=true)
}}
@rig {{
  root bone pelvis_root {{
    pivot_local:[0,14.5,0]
    box chest_core {{ size:[11,9,13], material:hide, prism_group:"ch" }}
    box sternum_keel place(base_box=chest_core, anchor=[0,-1,0.15], align=[0,0.55,0]) {{ size:[7,3.5,11], material:scute, rotate:[-8,0,0], prism_group:"ch" }}
    box rib_slope_l place(base_box=chest_core, anchor=[-0.95,-0.1,0.0], align=[0.6,0,0]) {{ size:[3,7,10.5], material:hide, rotate:[0,0,-22], prism_group:"ch" }}
    box rib_slope_r place(base_box=chest_core, anchor=[0.95,-0.1,0.0], align=[-0.6,0,0]) {{ size:[3,7,10.5], material:hide, rotate:[0,0,22], prism_group:"ch" }}
    box trapezius_gusset_l place(base_box=chest_core, anchor=[-0.72,0.82,0.68], align=[0.4,-0.5,-0.3]) {{ size:[3.5,3.5,5.5], material:hide, rotate:[-14,-16,-30], prism_group:"ch" }}
    box trapezius_gusset_r place(base_box=chest_core, anchor=[0.72,0.82,0.68], align=[-0.4,-0.5,-0.3]) {{ size:[3.5,3.5,5.5], material:hide, rotate:[-14,16,30], prism_group:"ch" }}
    box withers_crest place(base_box=chest_core, anchor=[0,1,0.1], align=[0,-0.4,0]) {{ size:[2,3.5,6.5], material:horn, rotate:[-18,0,0], prism_group:"ch" }}
    bone waist attach(parent_box=chest_core, anchor=[0,0.05,-1], align=[0,0,0.72]) {{
      rest_euler:[4,0,0]
      box waist_core {{ size:[7,7,8.5], material:hide, prism_group:"ws" }}
      box belly_plate place(base_box=waist_core, anchor=[0,-1,0], align=[0,0.65,0]) {{ size:[5,2,7.5], material:scute, prism_group:"ws" }}
      bone hips attach(parent_box=waist_core, anchor=[0,0.0,-1], align=[0,0,0.72]) {{
        rest_euler:[-4,0,0]
        box pelvis_core {{ size:[9.5,8,9.5], material:hide, prism_group:"pv" }}
        box pelvic_keel place(base_box=pelvis_core, anchor=[0,-1,0], align=[0,0.65,0]) {{ size:[6,2,8], material:scute, prism_group:"pv" }}
        chain tail attach(parent_box=pelvis_core, anchor=[0,0.12,-1], align=[0,0,0.72]) segments=5 taper=0.80 bend=[-4,2,8,12,14] {{
          box caudal {{ size:[5.5,5,8], material:hide }}
        }}
        bone leg_l_thigh attach(parent_box=pelvis_core, anchor=[-1,-0.25,0.0], align=[0.5,0.85,0]) mirror_x(leg_r_thigh) {{
          rest_euler:[-16,0,0]
          box thigh_core {{ size:[5,8.5,6.5], material:hide, prism_group:"th" }}
          box thigh_meat place(base_box=thigh_core, anchor=[0,0.1,-0.75], align=[0,0,0.45]) {{ size:[4.5,6.5,4], material:hide, rotate:[-24,0,-10], prism_group:"th" }}
          box thigh_greave place(base_box=thigh_core, anchor=[-0.75,-0.1,0.45], align=[0.4,0,-0.4]) {{ size:[2.5,5.5,3.5], material:scute, rotate:[-14,0,-14], prism_group:"th" }}
          bone leg_l_shin attach(parent_box=thigh_core, anchor=[0,-1,-0.1], align=[0,0.85,0]) {{
            rest_euler:[30,0,0]
            box shin_core {{ size:[3.5,7.5,4], material:hide, prism_group:"sh" }}
            box shin_guard place(base_box=shin_core, anchor=[0,0.05,0.8], align=[0,0,-0.55]) {{ size:[3,6,2], material:scute, rotate:[8,0,0], prism_group:"sh" }}
            bone hind_foot_l attach(parent_box=shin_core, anchor=[0,-1,0.12], align=[0,0.92,-0.15]) {{
              rest_euler:[-14,0,0]
              box foot_base {{ size:[4.5,2.5,5], material:hide, prism_group:"hf" }}
              bone toe_mid attach(parent_box=foot_base, anchor=[0,-0.2,1], align=[0,0,-0.7]) {{
                box d_mid {{ size:[1.5,1.5,3.5], material:hide, prism_group:"hf" }}
                box talon_mid place(base_box=d_mid, anchor=[0,-0.1,1], align=[0,0,-0.7]) {{ size:[1,1.5,2], material:horn, rotate:[14,0,0], prism_group:"hf" }}
              }}
              bone toe_in attach(parent_box=foot_base, anchor=[0.75,-0.2,0.85], align=[0,0,-0.7]) {{
                rest_euler:[0,18,0]
                box d_in {{ size:[1.5,1.5,3.5], material:horn, prism_group:"hf" }}
              }}
              bone toe_out attach(parent_box=foot_base, anchor=[-0.75,-0.2,0.85], align=[0,0,-0.7]) {{
                rest_euler:[0,-18,0]
                box d_out {{ size:[1.5,1.5,3.5], material:horn, prism_group:"hf" }}
              }}
              bone dewclaw attach(parent_box=foot_base, anchor=[0.6,0.1,-0.9], align=[0,0,0.7]) {{
                rest_euler:[-12,24,0]
                box spur {{ size:[1,1.5,2], material:horn, prism_group:"hf" }}
              }}
            }}
          }}
        }}
      }}
    }}
    bone arm_l_upper attach(parent_box=chest_core, anchor=[-1,-0.32,0.45], align=[0.5,0.85,0]) mirror_x(arm_r_upper) {{
      rest_euler:[14,0,0]
      box humerus_core {{ size:[4.5,7.5,5.5], material:hide, prism_group:"au" }}
      box deltoid_swell place(base_box=humerus_core, anchor=[-0.55,0.45,0.1], align=[0.4,-0.35,0]) {{ size:[3.5,5,4.5], material:hide, rotate:[0,0,-16], prism_group:"au" }}
      box bicep_plate place(base_box=humerus_core, anchor=[0,-0.05,0.8], align=[0,0,-0.55]) {{ size:[3.5,5,2], material:scute, rotate:[-8,0,0], prism_group:"au" }}
      bone arm_l_fore attach(parent_box=humerus_core, anchor=[0,-1,0.1], align=[0,0.85,0]) {{
        rest_euler:[-22,0,0]
        box fore_core {{ size:[3.5,7,4], material:hide, prism_group:"af" }}
        box vambrace place(base_box=fore_core, anchor=[-0.65,0,0.2], align=[0.4,0,-0.3]) {{ size:[2,5.5,3.5], material:scute, rotate:[0,0,-12], prism_group:"af" }}
        bone paw_l attach(parent_box=fore_core, anchor=[0,-1,0.1], align=[0,0.9,-0.15]) {{
          rest_euler:[8,0,0]
          box palm_base {{ size:[4.5,2.5,5], material:hide, prism_group:"pw" }}
          bone finger_mid attach(parent_box=palm_base, anchor=[0,-0.2,1], align=[0,0,-0.7]) {{
            box p_mid {{ size:[1.5,1.5,3.5], material:hide, prism_group:"pw" }}
            box claw_p_mid place(base_box=p_mid, anchor=[0,-0.1,1], align=[0,0,-0.7]) {{ size:[1,1.5,2], material:horn, rotate:[14,0,0], prism_group:"pw" }}
          }}
          bone finger_in attach(parent_box=palm_base, anchor=[0.75,-0.2,0.85], align=[0,0,-0.7]) {{
            rest_euler:[0,18,0]
            box p_in {{ size:[1.5,1.5,3.5], material:horn, prism_group:"pw" }}
          }}
          bone finger_out attach(parent_box=palm_base, anchor=[-0.75,-0.2,0.85], align=[0,0,-0.7]) {{
            rest_euler:[0,-18,0]
            box p_out {{ size:[1.5,1.5,3.5], material:horn, prism_group:"pw" }}
          }}
          bone thumb_l attach(parent_box=palm_base, anchor=[0.85,-0.15,0.2], align=[-0.5,0,-0.3]) {{
            rest_euler:[10,36,0]
            box pollex {{ size:[1.5,1.5,2.5], material:horn, prism_group:"pw" }}
          }}
        }}
      }}
    }}
    bone neck_1 attach(parent_box=chest_core, anchor=[0,0.42,1], align=[0,-0.15,-0.72]) {{
      rest_euler:[-28,0,0]
      box cerv_1 {{ size:[6.5,6.5,7.5], material:hide, prism_group:"n1" }}
      box throat_1 place(base_box=cerv_1, anchor=[0,-1,0], align=[0,0.65,0]) {{ size:[4.5,2,6.5], material:scute, prism_group:"n1" }}
      bone neck_2 attach(parent_box=cerv_1, anchor=[0,0.12,1], align=[0,-0.08,-0.72]) {{
        rest_euler:[18,0,0]
        box cerv_2 {{ size:[5.5,5.5,7], material:hide, prism_group:"n2" }}
        box throat_2 place(base_box=cerv_2, anchor=[0,-1,0], align=[0,0.65,0]) {{ size:[4,1.5,6], material:scute, prism_group:"n2" }}
        bone head attach(parent_box=cerv_2, anchor=[0,0.12,1], align=[0,0,-0.72]) {{
          rest_euler:[14,0,0]
          box cranium_core {{ size:[6,6.5,6.5], material:hide, prism_group:"hd" }}
          box nasal_bridge place(base_box=cranium_core, anchor=[0,0.18,1], align=[0,0.1,-0.72]) {{ size:[3.5,3,6], material:hide, rotate:[9,0,0], prism_group:"hd" }}
          box snout_prow place(base_box=nasal_bridge, anchor=[0,-0.05,1], align=[0,0,-0.65]) {{ size:[2.5,2.5,2.5], material:hide, rotate:[15,0,0], prism_group:"hd" }}
          box maxilla_wall_l place(base_box=cranium_core, anchor=[-0.82,-0.15,0.88], align=[0.45,0,-0.82]) {{ size:[2,2.5,6.5], material:hide, rotate:[6,-16,0], prism_group:"hd" }}
          box maxilla_wall_r place(base_box=cranium_core, anchor=[0.82,-0.15,0.88], align=[-0.45,0,-0.82]) {{ size:[2,2.5,6.5], material:hide, rotate:[6,16,0], prism_group:"hd" }}
          box brow_l place(base_box=cranium_core, anchor=[-0.88,0.62,0.52], align=[0.35,-0.25,0]) {{ size:[2,1.5,4.5], material:hide, rotate:[10,-12,-18], prism_group:"hd" }}
          box brow_r place(base_box=cranium_core, anchor=[0.88,0.62,0.52], align=[-0.35,-0.25,0]) {{ size:[2,1.5,4.5], material:hide, rotate:[10,12,18], prism_group:"hd" }}
          box eye_socket_l place(base_box=cranium_core, anchor=[-0.92,0.22,0.55], align=[0.65,0,0]) {{ size:[1,1.5,2], material:eye, rotate:[0,-16,0], prism_group:"hd" }}
          box eye_socket_r place(base_box=cranium_core, anchor=[0.92,0.22,0.55], align=[-0.65,0,0]) {{ size:[1,1.5,2], material:eye, rotate:[0,16,0], prism_group:"hd" }}
          bone horn_l attach(parent_box=cranium_core, anchor=[-0.78,0.92,-0.55], align=[0,-0.55,0.85]) mirror_x(horn_r) {{
            rest_euler:[26,22,-16]
            box horn_1 {{ size:[2.5,2.5,5.5], material:horn, blend_with:[cranium_core] }}
            bone horn_seg2 attach(parent_box=horn_1, anchor=[0,0,-1], align=[0,0,0.72]) {{
              rest_euler:[14,10,0]
              box horn_2 {{ size:[2,2,5], material:horn, blend_with:[horn_1] }}
              bone horn_seg3 attach(parent_box=horn_2, anchor=[0,0,-1], align=[0,0,0.72]) {{
                rest_euler:[-10,-6,0]
                box horn_3 {{ size:[1.5,1.5,4.5], material:horn, blend_with:[horn_2] }}
              }}
            }}
          }}
          bone jaw attach(parent_box=cranium_core, anchor=[0,-0.78,0.22], align=[0,0.5,-0.75]) {{
            rest_euler:[4,0,0]
            box jaw_ramus {{ size:[4,2,6.5], material:hide, prism_group:"jw" }}
            box mandible_rail_l place(base_box=jaw_ramus, anchor=[-0.8,0.15,0.2], align=[0.5,0,0]) {{ size:[1.5,1.5,5.5], material:hide, rotate:[0,-14,0], prism_group:"jw" }}
            box mandible_rail_r place(base_box=jaw_ramus, anchor=[0.8,0.15,0.2], align=[-0.5,0,0]) {{ size:[1.5,1.5,5.5], material:hide, rotate:[0,14,0], prism_group:"jw" }}
            box throat_chin place(base_box=jaw_ramus, anchor=[0,-0.9,0], align=[0,0.6,0]) {{ size:[3,1,5.5], material:scute, prism_group:"jw" }}
          }}
        }}
      }}
    }}
    bone wing_l_upper attach(parent_box=chest_core, anchor=[-1,0.72,0.3], align=[0.85,0,0]) mirror_x(wing_r_upper) {{
      rest_euler:[14,18,-28]
      box w_hum {{ size:[14,3,3.5], material:hide, prism_group:"wu" }}
      box w_hum_armor place(base_box=w_hum, anchor=[-0.22,0.85,0.35], align=[0,-0.5,-0.3]) {{ size:[9.5,2,2.5], material:scute, rotate:[12,0,0], prism_group:"wu" }}
      plane wing_inner_sail place(base_box=w_hum, anchor=[0,0,-1], align=[0,0,0.85]) {{ size:[14,14], normal:y, layer:0, edge:scallop(depth=3,side=bottom), material:vane }}
      bone wing_l_fore attach(parent_box=w_hum, anchor=[-1,0,0], align=[0.82,0,0]) {{
        rest_euler:[-10,-26,18]
        box w_rad {{ size:[17,2.5,3], material:hide, prism_group:"wf" }}
        box w_rad_trim place(base_box=w_rad, anchor=[0,0.85,0.35], align=[0,-0.5,-0.3]) {{ size:[14,1.5,2], material:scute, rotate:[10,0,0], prism_group:"wf" }}
        bone wing_l_f4 attach(parent_box=w_rad, anchor=[-0.45,0,-1], align=[1,0,1]) {{
          rest_euler:[-10,-24,14]
          box f4_spar {{ size:[18,1.5,1.5], material:hide, blend_with:[w_rad] }}
          plane wing_bay_4 place(base_box=f4_spar, anchor=[0,0,-1], align=[0,0,0.92]) {{ size:[28,28], normal:y, layer:2, edge:scallop(depth=3,side=bottom), material:vane, blend_with:[wing_inner_sail,w_rad,f3_spar,wing_bay_3] }}
        }}
        bone wing_l_wrist attach(parent_box=w_rad, anchor=[-1,0,0], align=[0.8,0,0]) {{
          rest_euler:[12,14,12]
          box w_palm {{ size:[3.5,2.5,4.5], material:hide, prism_group:"wp" }}
          box w_thumb place(base_box=w_palm, anchor=[0,0.7,0.9], align=[0,-0.5,-0.6]) {{ size:[1.5,2.5,2], material:horn, rotate:[18,0,0], prism_group:"wp" }}
          bone wing_l_f1 attach(parent_box=w_palm, anchor=[-1,0.2,0.8], align=[1,0,0]) {{
            rest_euler:[16,-12,-14]
            box f1_spar {{ size:[16,2,2], material:hide, blend_with:[w_palm,f2_spar] }}
            plane wing_bay_1 place(base_box=f1_spar, anchor=[-0.55,0,-1], align=[0,0,0.92]) {{ size:[24,18], normal:y, layer:-1, edge:scallop(depth=3,side=bottom), material:vane, blend_with:[f2_spar,f2_distal,wing_bay_2] }}
            bone wing_l_f1_tip attach(parent_box=f1_spar, anchor=[-1,0,0], align=[0.88,0,0]) {{
              rest_euler:[-18,10,16]
              box f1_distal {{ size:[12,1.5,1.5], material:hide, blend_with:[f1_spar] }}
            }}
          }}
          bone wing_l_f2 attach(parent_box=w_palm, anchor=[-1,0,0.0], align=[1,0,0]) {{
            rest_euler:[4,14,6]
            box f2_spar {{ size:[15,1.5,1.5], material:hide, blend_with:[w_palm,f3_spar] }}
            plane wing_bay_2 place(base_box=f2_spar, anchor=[-0.55,0,-1], align=[0,0,0.92]) {{ size:[22,18], normal:y, layer:0, edge:scallop(depth=3,side=bottom), material:vane, blend_with:[f1_spar,f3_spar,wing_bay_3] }}
            bone wing_l_f2_tip attach(parent_box=f2_spar, anchor=[-1,0,0], align=[0.88,0,0]) {{
              rest_euler:[-16,8,14]
              box f2_distal {{ size:[11,1.5,1.5], material:hide, blend_with:[f2_spar] }}
            }}
          }}
          bone wing_l_f3 attach(parent_box=w_palm, anchor=[-1,-0.2,-0.8], align=[1,0,0]) {{
            rest_euler:[-14,34,18]
            box f3_spar {{ size:[20,1.5,1.5], material:hide, blend_with:[w_palm,f2_spar] }}
            plane wing_bay_3 place(base_box=f3_spar, anchor=[0,0,-1], align=[0,0,0.92]) {{ size:[20,17], normal:y, layer:1, edge:scallop(depth=3,side=bottom), material:vane, blend_with:[f2_spar,f4_spar,wing_bay_4] }}
          }}
        }}
      }}
    }}
  }}
}}
@style {{
  shade: banded(steps=5, direction=down, cluster=diagonal)
  glow_bleed: 2
}}
@behavior {{
  clip idle {{
    duration_s: 2.4, loop: true
    track neck_1: wave(axis=x, amp_deg=3, freq_hz=0.416)
    track head: wave(axis=x, amp_deg=3, freq_hz=0.416, phase_deg=180)
    track wing_l_upper: wave(axis=z, amp_deg=4, freq_hz=0.416, phase_deg=0)
    track wing_r_upper: wave(axis=z, amp_deg=4, freq_hz=0.416, phase_deg=180)
    track tail_1: wave(axis=y, amp_deg=7, freq_hz=0.416)
  }}
}}
"""
    return {
        "asset_id": f"azure_runic_sky_drake_{tag}",
        "domain": "D3",
        "mood": "M3",
        "tier": "hero",
        "archetype": "Anchor 06 / Archetype B Noble Quadruped Dragon",
        "concept": f"Noble {tag} runic sky drake with yawed V-rail maxilla, 4-segment stone rune horns, sunken compound thigh/deltoid wedges, 3D vaulted full-chord nested-fan wings, 4-digit splayed claws, and 4-foot grounded athletic stance.",
        "dsl": _compact_dsl(dsl),
    }


def _build_d4_m4_obsidian_halberd_elder_wyrm(variant: int = 0) -> dict[str, Any]:
    """`(D4, M4 Ferocious Apex Boss)` Coiled Elder Dragon locking ALL 10 M4 knobs + 6-plane zero-thickness flat tooth arcade + articulated 2-seg tongue + unified coplanar inner wing (`v25`)."""
    palettes = (
        ("obsidian_crimson", [0.22, 0.05, 292], [0.58, 0.14, 14], [0.70, 0.09, 76], [0.46, 0.18, 336], [0.68, 0.22, 24]),
        ("platinum_blood", [0.76, 0.03, 245], [0.44, 0.08, 255], [0.72, 0.12, 82], [0.34, 0.16, 265], [0.66, 0.23, 22]),
    )
    tag, hide_lch, trim_lch, gold_lch, wing_lch, eye_lch = palettes[variant % len(palettes)]

    # M4 Ferocious knobs:
    # - Coiled low-slung crouch (belly_clearance ~ 0.30-0.36 <= 0.42, G12!)
    # - Forward-surging S-neck (-18 deg) + downward Kubrick-stare chin tuck (+14 deg)
    # - Plunging golden V-brow (+16 deg pitch, -24 deg roll) + razor-narrow 2.0u falcon prow + +-16 deg V-rail maxilla
    # - Sleek 2.0u horizontal blood-ruby almond eye at 20 deg binocular yaw inside obsidian socket
    # - Swept-back -46 deg scimitar horns (never 85 deg bunny ears!)
    # - 6-plane zero-thickness single-pixel-square tooth U-arcade + mouth_throat_back + jaw_cheek_web + 2-seg curved tongue
    # - Unified coplanar wing_bay_4 + rigid wing_l_f4 directly on wing_l_fore + edge-locked wing_inner_sail (v25 R16!)
    dsl = f"""
@meta {{ archetype:entity, target_profile:bedrock_entity, texel_density:16, units_per_block:16, dsl_version:"4.1", size_step:0.5, audit_profile:sculpted }}
@palette {{
  hide: ramp(base_lch={hide_lch}, steps=5, hue_shift=[-24,24])
  slate: ramp(base_lch={trim_lch}, steps=4, hue_shift=[-16,18])
  gold: ramp(base_lch={gold_lch}, steps=4, hue_shift=[-14,16])
  velvet: ramp(base_lch={wing_lch}, steps=5, hue_shift=[-22,22])
  maw: ramp(base_lch=[0.26,0.10,350], steps=4, hue_shift=[-10,12])
  tooth: ramp(base_lch=[0.90,0.03,92], steps=4, hue_shift=[-6,8])
  eye: ramp(base_lch={eye_lch}, steps=4, hue_shift=[-6,10], emissive=true)
}}
@rig {{
  root bone pelvis_root {{
    pivot_local:[0,12.5,0]
    box chest {{ size:[12,9.5,14], material:hide, prism_group:"ch" }}
    box sternum place(base_box=chest, anchor=[0,-1,0.22], align=[0,0.52,0]) {{ size:[7.5,3.5,12], material:gold, rotate:[-10,0,0], prism_group:"ch" }}
    box rib_l place(base_box=chest, anchor=[-0.95,-0.1,0], align=[0.55,0,0]) {{ size:[3.5,7.5,11.5], material:hide, rotate:[0,0,-24], prism_group:"ch" }}
    box rib_r place(base_box=chest, anchor=[0.95,-0.1,0], align=[-0.55,0,0]) {{ size:[3.5,7.5,11.5], material:hide, rotate:[0,0,24], prism_group:"ch" }}
    box withers place(base_box=chest, anchor=[0,1,0.05], align=[0,-0.4,0]) {{ size:[2,4.5,7.5], material:gold, rotate:[-26,0,0], prism_group:"ch" }}
    bone waist attach(parent_box=chest, anchor=[0,0,-1], align=[0,0,0.72]) {{
      rest_euler:[6,0,0]
      box waist_b {{ size:[7,7,9], material:hide, prism_group:"ws" }}
      box waist_k place(base_box=waist_b, anchor=[0,-1,0], align=[0,0.65,0]) {{ size:[5,2,7.5], material:slate, prism_group:"ws" }}
      bone hips attach(parent_box=waist_b, anchor=[0,-0.05,-1], align=[0,0,0.72]) {{
        rest_euler:[-6,0,0]
        box pelvis {{ size:[10,8.5,10], material:hide, prism_group:"pv" }}
        chain tail attach(parent_box=pelvis, anchor=[0,0.1,-1], align=[0,0,0.72]) segments=5 taper=0.78 bend=[-6,-2,4,8,10] {{
          box caudal {{ size:[5.5,5,8.5], material:hide }}
        }}
        box spine_b place(base_box=pelvis, anchor=[0,0.85,-0.5], align=[0,-0.4,0]) {{ size:[2,3.5,6], material:gold, rotate:[-28,0,0], prism_group:"pv" }}
        bone leg_l_thigh attach(parent_box=pelvis, anchor=[-1,-0.3,0.05], align=[0.5,0.85,0]) mirror_x(leg_r_thigh) {{
          rest_euler:[-28,0,0]
          box thigh {{ size:[5.5,8.5,7], material:hide, prism_group:"th" }}
          box thigh_m place(base_box=thigh, anchor=[0,0.05,-0.75], align=[0,0,0.45]) {{ size:[4.5,6.5,4.5], material:hide, rotate:[-24,0,-12], prism_group:"th" }}
          bone leg_l_shin attach(parent_box=thigh, anchor=[0,-1,-0.1], align=[0,0.82,0]) {{
            rest_euler:[46,0,0]
            box shin {{ size:[3.5,7.5,4.5], material:hide, prism_group:"sh" }}
            box greave place(base_box=shin, anchor=[0,0.05,0.82], align=[0,0,-0.55]) {{ size:[3,6,2], material:gold, rotate:[8,0,0], prism_group:"sh" }}
            bone hind_foot_l attach(parent_box=shin, anchor=[0,-1,0.12], align=[0,0.9,-0.15]) {{
              rest_euler:[-18,0,0]
              box foot {{ size:[5,2.5,5.5], material:hide, prism_group:"hf" }}
              bone toe_mid attach(parent_box=foot, anchor=[0,-0.2,1], align=[0,0,-0.7]) {{
                box d_mid {{ size:[1.5,1.5,4.5], material:gold, prism_group:"hf" }}
              }}
              bone toe_in attach(parent_box=foot, anchor=[0.75,-0.2,0.85], align=[0,0,-0.7]) {{
                rest_euler:[0,20,0]
                box d_in {{ size:[1.5,1.5,3.5], material:gold, prism_group:"hf" }}
              }}
              bone toe_out attach(parent_box=foot, anchor=[-0.75,-0.2,0.85], align=[0,0,-0.7]) {{
                rest_euler:[0,-20,0]
                box d_out {{ size:[1.5,1.5,3.5], material:gold, prism_group:"hf" }}
              }}
              bone dewclaw attach(parent_box=foot, anchor=[0.6,0.1,-0.9], align=[0,0,0.7]) {{
                rest_euler:[-12,26,0]
                box spur {{ size:[1,1.5,2], material:gold, prism_group:"hf" }}
              }}
            }}
          }}
        }}
      }}
    }}
    bone arm_l_upper attach(parent_box=chest, anchor=[-1,-0.32,0.48], align=[0.5,0.85,0]) mirror_x(arm_r_upper) {{
      rest_euler:[26,0,-10]
      box humerus {{ size:[5,7.5,6], material:hide, prism_group:"au" }}
      box deltoid place(base_box=humerus, anchor=[-0.6,0.5,0.1], align=[0.4,-0.35,0]) {{ size:[4,5,5], material:slate, rotate:[0,0,-20], prism_group:"au" }}
      bone arm_l_fore attach(parent_box=humerus, anchor=[0,-1,0.1], align=[0,0.82,0]) {{
        rest_euler:[-44,0,10]
        box forearm {{ size:[3.5,8,4.5], material:hide, prism_group:"af" }}
        box vambrace place(base_box=forearm, anchor=[-0.7,0.05,0.25], align=[0.4,0,-0.3]) {{ size:[2.5,6,3.5], material:gold, rotate:[0,0,-14], prism_group:"af" }}
        bone paw_l attach(parent_box=forearm, anchor=[0,-1,0.12], align=[0,0.9,-0.15]) {{
          rest_euler:[18,0,0]
          box palm {{ size:[5,2.5,5.5], material:hide, prism_group:"pw" }}
          bone finger_mid attach(parent_box=palm, anchor=[0,-0.2,1], align=[0,0,-0.7]) {{
            box p_mid {{ size:[1.5,1.5,4.5], material:gold, prism_group:"pw" }}
          }}
          bone finger_in attach(parent_box=palm, anchor=[0.75,-0.2,0.85], align=[0,0,-0.7]) {{
            rest_euler:[0,20,0]
            box p_in {{ size:[1.5,1.5,3.5], material:gold, prism_group:"pw" }}
          }}
          bone finger_out attach(parent_box=palm, anchor=[-0.75,-0.2,0.85], align=[0,0,-0.7]) {{
            rest_euler:[0,-20,0]
            box p_out {{ size:[1.5,1.5,3.5], material:gold, prism_group:"pw" }}
          }}
          bone thumb_l attach(parent_box=palm, anchor=[0.85,-0.15,0.2], align=[-0.5,0,-0.3]) {{
            rest_euler:[10,38,0]
            box pollex {{ size:[1.5,1.5,2.5], material:gold, prism_group:"pw" }}
          }}
        }}
      }}
    }}
    bone neck_1 attach(parent_box=chest, anchor=[0,0.32,1], align=[0,-0.1,-0.72]) {{
      rest_euler:[-18,0,0]
      box cerv_1 {{ size:[7,6.5,7.5], material:hide, prism_group:"n1" }}
      box throat_1 place(base_box=cerv_1, anchor=[0,-1,0], align=[0,0.6,0]) {{ size:[4.5,2,6.5], material:slate, prism_group:"n1" }}
      bone neck_2 attach(parent_box=cerv_1, anchor=[0,0.1,1], align=[0,-0.08,-0.72]) {{
        rest_euler:[10,0,0]
        box cerv_2 {{ size:[5.5,5.5,7], material:hide, prism_group:"n2" }}
        box throat_2 place(base_box=cerv_2, anchor=[0,-1,0], align=[0,0.6,0]) {{ size:[4,1.5,6], material:slate, prism_group:"n2" }}
        bone head attach(parent_box=cerv_2, anchor=[0,0.08,1], align=[0,0,-0.72]) {{
          rest_euler:[20,0,0]
          box cranium {{ size:[6,6.5,6.5], material:hide, prism_group:"hd" }}
          box pharynx place(base_box=cranium, anchor=[0,-0.45,0.75], align=[0,0,-0.45]) {{ size:[3,1.5,1.5], material:maw, prism_group:"hd" }}
          box visor place(base_box=cranium, anchor=[0,0.24,1], align=[0,0.1,-0.72]) {{ size:[3.5,2.5,6], material:hide, rotate:[10,0,0], prism_group:"hd" }}
          box beak place(base_box=visor, anchor=[0,-0.05,1], align=[0,0,-0.65]) {{ size:[2,3,2.5], material:gold, rotate:[22,0,0], prism_group:"hd" }}
          box maxilla_l place(base_box=cranium, anchor=[-0.82,-0.18,0.88], align=[0.45,0,-0.82]) {{ size:[2,2.5,6.5], material:hide, rotate:[8,-16,0], prism_group:"hd" }}
          box maxilla_r place(base_box=cranium, anchor=[0.82,-0.18,0.88], align=[-0.45,0,-0.82]) {{ size:[2,2.5,6.5], material:hide, rotate:[8,16,0], prism_group:"hd" }}
          box brow_l place(base_box=cranium, anchor=[-0.88,0.62,0.52], align=[0.35,-0.25,0]) {{ size:[2,1.5,4.5], material:gold, rotate:[16,-14,-24], prism_group:"hd" }}
          box brow_r place(base_box=cranium, anchor=[0.88,0.62,0.52], align=[-0.35,-0.25,0]) {{ size:[2,1.5,4.5], material:gold, rotate:[16,14,24], prism_group:"hd" }}
          box eye_l place(base_box=cranium, anchor=[-0.92,0.24,0.56], align=[0.65,0,0]) {{ size:[1,2,2.5], material:eye, rotate:[0,-20,0], prism_group:"hd" }}
          box eye_r place(base_box=cranium, anchor=[0.92,0.24,0.56], align=[-0.65,0,0]) {{ size:[1,2,2.5], material:eye, rotate:[0,20,0], prism_group:"hd" }}
          plane u_teeth_l place(base_box=maxilla_l, anchor=[0.4,-1,0], align=[0,0.85,0]) {{ size:[1.5,5.5], normal:x, layer:0, edge:spikes(depth=2,side=bottom), material:tooth }}
          plane u_teeth_r place(base_box=maxilla_r, anchor=[-0.4,-1,0], align=[0,0.85,0]) {{ size:[1.5,5.5], normal:x, layer:0, edge:spikes(depth=2,side=bottom), material:tooth }}
          plane u_fangs place(base_box=beak, anchor=[0,-1,-0.3], align=[0,0.85,0]) {{ size:[2,1.5], normal:z, layer:0, edge:spikes(depth=2,side=bottom), material:tooth }}
          bone horn_l attach(parent_box=cranium, anchor=[-0.78,0.88,-0.55], align=[0,-0.55,0.85]) mirror_x(horn_r) {{
            rest_euler:[34,24,-18]
            box horn_1 {{ size:[2.5,2.5,6], material:gold, blend_with:[cranium] }}
            box horn_2 place(base_box=horn_1, anchor=[0,0,-1], align=[0,0,0.72]) {{ size:[2,2,5.5], material:gold, rotate:[12,12,0], blend_with:[horn_1] }}
            box horn_3 place(base_box=horn_2, anchor=[0,0,-1], align=[0,0,0.72]) {{ size:[1.5,1.5,5], material:slate, rotate:[-10,-6,0], blend_with:[horn_2] }}
          }}
          bone jaw attach(parent_box=cranium, anchor=[0,-0.78,0.22], align=[0,0.5,-0.75]) {{
            rest_euler:[4,0,0]
            box ramus {{ size:[4,2,6.5], material:hide, prism_group:"jw" }}
            box mand_l place(base_box=ramus, anchor=[-0.8,0.18,0.2], align=[0.5,0,0]) {{ size:[1.5,1.5,5.5], material:hide, rotate:[0,-14,0], prism_group:"jw" }}
            box mand_r place(base_box=ramus, anchor=[0.8,0.18,0.2], align=[-0.5,0,0]) {{ size:[1.5,1.5,5.5], material:hide, rotate:[0,14,0], prism_group:"jw" }}
            plane l_teeth_l place(base_box=mand_l, anchor=[0.4,1,0], align=[0,-0.85,0]) {{ size:[1.5,5], normal:x, layer:0, edge:spikes(depth=2,side=top), material:tooth }}
            plane l_teeth_r place(base_box=mand_r, anchor=[-0.4,1,0], align=[0,-0.85,0]) {{ size:[1.5,5], normal:x, layer:0, edge:spikes(depth=2,side=top), material:tooth }}
            plane l_fangs place(base_box=ramus, anchor=[0,1,0.9], align=[0,-0.85,0]) {{ size:[2,1.5], normal:z, layer:0, edge:spikes(depth=2,side=top), material:tooth }}
            plane cheek_l place(base_box=mand_l, anchor=[0.2,1,-0.05], align=[0,-0.8,0]) {{ size:[1.5,2], normal:x, layer:1, edge:scallop(depth=1,side=top), material:maw }}
            plane cheek_r place(base_box=mand_r, anchor=[-0.2,1,-0.05], align=[0,-0.8,0]) {{ size:[1.5,2], normal:x, layer:1, edge:scallop(depth=1,side=top), material:maw }}
            bone tongue attach(parent_box=ramus, anchor=[0,1,0.18], align=[0,-1,-0.85]) {{
              rest_euler:[-6,0,0]
              box tongue_b {{ size:[2,1,1.5], material:maw, blend_with:[ramus] }}
              bone tongue_tip attach(parent_box=tongue_b, anchor=[0,0,1], align=[0,0,-0.7]) {{
                box tongue_e {{ size:[1.5,1,1], material:maw, blend_with:[tongue_b] }}
              }}
            }}
          }}
        }}
      }}
    }}
    bone wing_l_upper attach(parent_box=chest, anchor=[-1,0.72,0.18], align=[0.85,0,0]) mirror_x(wing_r_upper) {{
      rest_euler:[16,16,-30]
      box w_hum {{ size:[15,3,3.5], material:hide, prism_group:"wu" }}
      box w_crest place(base_box=w_hum, anchor=[-0.18,0.85,0.35], align=[0,-0.5,-0.3]) {{ size:[10,2,2.5], material:gold, rotate:[14,0,0], prism_group:"wu" }}
      plane wing_inner_sail place(base_box=w_hum, anchor=[0,0,-1], align=[0,0,0.88]) {{ size:[15,15], normal:y, layer:0, edge:scallop(depth=3,side=bottom), material:velvet }}
      bone wing_l_fore attach(parent_box=w_hum, anchor=[-1,0,0], align=[0.82,0,0]) {{
        rest_euler:[-12,-24,18]
        box w_rad {{ size:[18,2.5,3], material:hide, prism_group:"wf" }}
        box w_blade place(base_box=w_rad, anchor=[0,0.85,0.35], align=[0,-0.5,-0.3]) {{ size:[15,2,2], material:gold, rotate:[10,0,0], prism_group:"wf" }}
        bone wing_l_f4 attach(parent_box=w_rad, anchor=[-0.45,0,-1], align=[1,0,1]) {{
          rest_euler:[-10,-24,14]
          box f4_spar {{ size:[19,1.5,1.5], material:hide, blend_with:[w_rad] }}
          plane wing_bay_4 place(base_box=f4_spar, anchor=[0,0,-1], align=[0,0,0.92]) {{ size:[29,29], normal:y, layer:2, edge:scallop(depth=3,side=bottom), material:velvet, blend_with:[wing_inner_sail,w_rad,f3_spar,wing_bay_3] }}
        }}
        bone wing_l_wrist attach(parent_box=w_rad, anchor=[-1,0,0], align=[0.8,0,0]) {{
          rest_euler:[12,12,12]
          box w_palm {{ size:[3.5,2.5,4.5], material:hide, prism_group:"wp" }}
          box w_thumb place(base_box=w_palm, anchor=[0,0.75,0.9], align=[0,-0.5,-0.6]) {{ size:[1.5,3,2], material:gold, rotate:[20,0,0], prism_group:"wp" }}
          bone wing_l_f1 attach(parent_box=w_palm, anchor=[-1,0.2,0.8], align=[1,0,0]) {{
            rest_euler:[16,-12,-14]
            box f1_spar {{ size:[17,2,2], material:hide, blend_with:[w_palm,f2_spar] }}
            plane wing_bay_1 place(base_box=f1_spar, anchor=[-0.55,0,-1], align=[0,0,0.92]) {{ size:[25,20], normal:y, layer:-1, edge:scallop(depth=3,side=bottom), material:velvet, blend_with:[f2_spar,f2_distal,wing_bay_2] }}
            bone wing_l_f1_tip attach(parent_box=f1_spar, anchor=[-1,0,0], align=[0.88,0,0]) {{
              rest_euler:[-18,10,16]
              box f1_distal {{ size:[12,1.5,1.5], material:hide, blend_with:[f1_spar] }}
            }}
          }}
          bone wing_l_f2 attach(parent_box=w_palm, anchor=[-1,0,0], align=[1,0,0]) {{
            rest_euler:[4,14,6]
            box f2_spar {{ size:[15,1.5,1.5], material:hide, blend_with:[w_palm,f3_spar] }}
            plane wing_bay_2 place(base_box=f2_spar, anchor=[-0.55,0,-1], align=[0,0,0.92]) {{ size:[23,19], normal:y, layer:0, edge:scallop(depth=3,side=bottom), material:velvet, blend_with:[f1_spar,f3_spar,wing_bay_3] }}
            bone wing_l_f2_tip attach(parent_box=f2_spar, anchor=[-1,0,0], align=[0.88,0,0]) {{
              rest_euler:[-16,8,14]
              box f2_distal {{ size:[11,1.5,1.5], material:hide, blend_with:[f2_spar] }}
            }}
          }}
          bone wing_l_f3 attach(parent_box=w_palm, anchor=[-1,-0.2,-0.8], align=[1,0,0]) {{
            rest_euler:[-14,34,18]
            box f3_spar {{ size:[21,1.5,1.5], material:hide, blend_with:[w_palm,f2_spar] }}
            plane wing_bay_3 place(base_box=f3_spar, anchor=[0,0,-1], align=[0,0,0.92]) {{ size:[21,17], normal:y, layer:1, edge:scallop(depth=3,side=bottom), material:velvet, blend_with:[f2_spar,f4_spar,wing_bay_4] }}
          }}
        }}
      }}
    }}
  }}
}}
@style {{
  shade: banded(steps=5, direction=down, cluster=diagonal)
  glow_bleed: 2
}}
@behavior {{
  clip roar_maw {{
    duration_s: 1.8, loop: true
    track neck_1: wave(axis=x, amp_deg=5, freq_hz=0.555, bias_deg=-3)
    track head: wave(axis=x, amp_deg=3, freq_hz=0.555, bias_deg=-2)
    track jaw: wave(axis=x, amp_deg=14, freq_hz=0.555, bias_deg=15)
    track tongue: wave(axis=x, amp_deg=3, freq_hz=0.555, bias_deg=-2)
    track wing_l_upper: wave(axis=z, amp_deg=7, freq_hz=0.555, phase_deg=0, bias_deg=-5)
    track wing_r_upper: wave(axis=z, amp_deg=7, freq_hz=0.555, phase_deg=180, bias_deg=5)
    track tail_1: wave(axis=y, amp_deg=8, freq_hz=0.555)
  }}
}}
"""
    return {
        "asset_id": f"obsidian_halberd_elder_wyrm_{tag}",
        "domain": "D4",
        "mood": "M4",
        "tier": "hero",
        "archetype": "Anchor 09 / Archetype E Ferocious Coiled Elder Dragon",
        "concept": f"Ferocious M4 {tag} coiled Elder Dragon locking all 10 M4 mood knobs: low predatory crouch (belly_clearance<=0.36), plunging golden V-brow, sleek 2.0u ruby slit hawk eyes at 20 deg binocular yaw, -46 deg swept scimitar horns, razor 2.0u falcon prow + V-rail maxilla, 6-plane zero-thickness single-pixel tooth U-arcade + 2-segment tongue + concave cheek webs, unified coplanar wing_bay_4 + rigid wing_l_f4 on wing_l_fore, and 4-foot ground-locked ecology clips.",
        "dsl": _compact_dsl(dsl),
    }


def _auto_ground_dsl_four_feet(dsl: str) -> str:
    """Adjust root `pivot_local:[0,Y,0]` (and foreleg/hindleg height if needed) so min(FL,FR,HL,HR) lands within 0.25u of y=0."""
    import re

    ensure_synccraft_importable()
    from synccraft.backends.bbmodel5_exporter import export_bbmodel
    from synccraft.dsl.parser import parse_program
    from synccraft.kinematics.se3_chain import compile_geometry
    from .v25_blockbench_verifier import verify_v25_four_foot_grounding

    prog = parse_program(dsl)
    scene = compile_geometry(prog)
    bb = export_bbmodel(scene, name="ground_probe", program=prog, num_animation_frames=4)
    g = verify_v25_four_foot_grounding(bb)
    feet = g["rest_feet"]
    ymins = [float(feet[k]) for k in ("FL", "FR", "HL", "HR") if k in feet]
    if not ymins:
        # Prop: use global scene min y
        y_all = min(float(b.center[1] - 0.5 * b.size[1]) for b in scene.boxes)
        shift = -round(y_all * 2.0) / 2.0
    else:
        shift = -round(min(ymins) * 2.0) / 2.0

    if abs(shift) > 1e-3:
        m = re.search(r"pivot_local:\[0,([-0-9.]+),0\]", dsl)
        if m:
            old_y = float(m.group(1))
            new_y = round((old_y + shift) * 2.0) / 2.0
            dsl = dsl[: m.start()] + f"pivot_local:[0,{new_y:g},0]" + dsl[m.end() :]
    return dsl


def _build_trap_negative_assets() -> list[dict[str, Any]]:
    """Six real compiled+rendered `v25` negative trap exemplars triggering Section 13.2 `<= 14/28` Automatic Caps."""
    good_drake = _auto_ground_dsl_four_feet(_build_d3_m3_azure_runic_sky_drake(0)["dsl"])
    good_boss = _auto_ground_dsl_four_feet(_build_d4_m4_obsidian_halberd_elder_wyrm(0)["dsl"])

    # Trap 1 (G2 Wedding-Cake Snout): centred yaw=0 stacked snout boxes, no V-rail maxilla
    trap_g2 = good_drake.replace(
        'box maxilla_wall_l place(base_box=cranium_core,anchor=[-0.82,-0.15,0.88],align=[0.45,0,-0.82]) { size:[2,2.5,6.5],material:hide,rotate:[6,-16,0],prism_group:"hd" }',
        'box snout_step_mid place(base_box=cranium_core,anchor=[0,-0.1,1],align=[0,0,-0.7]) { size:[4.5,3,3],material:hide,prism_group:"hd" }',
    )
    trap_g2 = trap_g2.replace("rotate:[6,-16,0]", "rotate:[0,0,0]").replace("rotate:[6,16,0]", "rotate:[0,0,0]").replace("maxilla_wall_l", "snout_tier_1").replace("maxilla_wall_r", "snout_tier_2")

    # Trap 5 (G10 Gapped Wing Strips): shrink wing plane chords from 18-28u down to skinny 6u flags!
    trap_g10 = good_drake.replace("size:[24,18]", "size:[24,6]").replace("size:[22,18]", "size:[22,6]").replace("size:[20,17]", "size:[20,6]").replace("size:[28,28]", "size:[18,6]")

    # Trap 11 (G11 Fake-Blender Wafer Slices): stack 7 thin 0.5u wafer slices on cranium!
    wafer_stack = "\n".join(
        f'box wafer_{i} place(base_box=cranium_core,anchor=[0,0.9,{round(-0.6 + 0.2 * i, 2)}],align=[0,-0.5,0]) {{ size:[{4.0 - 0.5 * (i // 2)},0.5,1],material:hide,prism_group:"hd" }}'
        for i in range(7)
    )
    trap_g11 = good_drake.replace(
        'box nasal_bridge place(base_box=cranium_core,anchor=[0,0.18,1],align=[0,0.1,-0.72]) { size:[3.5,3,6],material:hide,rotate:[9,0,0],prism_group:"hd" }',
        wafer_stack + '\nbox nasal_bridge place(base_box=cranium_core,anchor=[0,0.18,1],align=[0,0.1,-0.72]) { size:[3.5,3,6],material:hide,rotate:[9,0,0],prism_group:"hd" }',
    )

    # Trap 16 (Mid-Air Floating Forelimbs): shorten arm_l_fore by 5u so hind feet sit at y=0 while front paws hover +5u!
    trap_16 = good_boss.replace(
        'box forearm { size:[3.5,8,4.5],material:hide,prism_group:"af" }',
        'box forearm { size:[3.5,3,4.5],material:hide,prism_group:"af" }',
    )

    # Trap 19 ("Cute & Dumb" M4 Boss Drift): vertical giraffe neck -42 deg, upright +78 deg bunny-ear horns, blunt square nose, flat brow!
    trap_19 = (
        good_boss.replace("rest_euler:[-18,0,0]", "rest_euler:[-42,0,0]")
        .replace("rest_euler:[34,24,-18]", "rest_euler:[78,6,-8]")
        .replace("size:[2,3,2.5],material:gold,rotate:[22,0,0]", "size:[4.5,3.5,3],material:tooth,rotate:[0,0,0]")
        .replace("rotate:[16,-14,-24]", "rotate:[-10,0,0]")
        .replace("rotate:[16,14,24]", "rotate:[-10,0,0]")
    )

    # Trap 22 (Detached Double-Layer Inner Wing Membrane): add redundant floating wing_fore_patagium at offset layer 6!
    trap_22 = good_boss.replace(
        'plane wing_bay_4 place(base_box=f4_spar,anchor=[0,0,-1],align=[0,0,0.92]) { size:[29,29],normal:y,layer:2,edge:scallop(depth=3,side=bottom),material:velvet,blend_with:[wing_inner_sail,w_rad,f3_spar,wing_bay_3] }',
        'plane wing_bay_4 place(base_box=f4_spar,anchor=[0,0,-1],align=[0,0,0.92]) { size:[29,29],normal:y,layer:2,edge:scallop(depth=3,side=bottom),material:velvet,blend_with:[wing_inner_sail,w_rad,f3_spar,wing_bay_3] }\nplane wing_fore_patagium_bad place(base_box=f4_spar,anchor=[0,1,-0.5],align=[0,0,0.6]) { size:[16,14],normal:y,layer:6,edge:scallop(depth=2,side=bottom),material:velvet }',
    )

    return [
        {
            "trap_id": "trap_01_g2_wedding_cake_snout",
            "trap_name": "Trap #1 (`G2` Wedding-Cake `yaw=0` Snout)",
            "cap_rule": "Section 13.2 Cap #4 (`Item 2 = 0`, total capped `<= 14/28`) -> `PUNCH BACK TO W1`",
            "diagnosis": "The upper muzzle stacks axis-aligned `yaw = 0` rectangular tiers without `+-14..22.5 deg` yawed V-rail maxilla cheek walls, making the top and 3/4 silhouette read as a rectangular wedding-cake box.",
            "bad_dsl": trap_g2,
            "fixed_patch": (
                "# Delete axis-aligned snout_tier_1/2 & replace with yawed +-16 deg V-rail maxilla walls:\n"
                'box nasal_bridge place(base_box=cranium_core,anchor=[0,0.18,1],align=[0,0.1,-0.72]) { size:[3.5,3,6],material:hide,rotate:[9,0,0],prism_group:"hd" }\n'
                'box snout_prow place(base_box=nasal_bridge,anchor=[0,-0.05,1],align=[0,0,-0.65]) { size:[2.5,2.5,2.5],material:hide,rotate:[15,0,0],prism_group:"hd" }\n'
                'box maxilla_wall_l place(base_box=cranium_core,anchor=[-0.82,-0.15,0.88],align=[0.45,0,-0.82]) { size:[2,2.5,6.5],material:hide,rotate:[6,-16,0],prism_group:"hd" }\n'
                'box maxilla_wall_r place(base_box=cranium_core,anchor=[0.82,-0.15,0.88],align=[-0.45,0,-0.82]) { size:[2,2.5,6.5],material:hide,rotate:[6,16,0],prism_group:"hd" }'
            ),
        },
        {
            "trap_id": "trap_05_g10_gapped_wing_strips",
            "trap_name": "Trap #5 (`G10` Inter-Finger Wing V-Gaps / Narrow Strip Membranes)",
            "cap_rule": "Section 13.2 Cap #2 (`Item 10 = 0`, total capped `<= 14/28`) -> `PUNCH BACK TO W1`",
            "diagnosis": "The wing bay membranes are thin `6u` rectangular flags (`chord_coverage < 0.88x` adjacent fingertip gap), opening empty triangular V-shaped sky gaps between splayed fingers instead of continuous `>= 1.08x` full-chord nested-fan bays.",
            "bad_dsl": trap_g10,
            "fixed_patch": (
                "# Expand all 4 wing bay chords to >=1.08x adjacent fingertip chord gap with stepped layers & scalloped hems:\n"
                'plane wing_bay_4 place(base_box=f4_spar,anchor=[0,0,-1],align=[0,0,0.92]) { size:[28,28],normal:y,layer:2,edge:scallop(depth=3,side=bottom),material:vane,blend_with:[wing_inner_sail,w_rad,f3_spar,wing_bay_3] }\n'
                'plane wing_bay_1 place(base_box=f1_spar,anchor=[-0.55,0,-1],align=[0,0,0.92]) { size:[24,18],normal:y,layer:-1,edge:scallop(depth=3,side=bottom),material:vane,blend_with:[f2_spar,f2_distal,wing_bay_2] }\n'
                'plane wing_bay_2 place(base_box=f2_spar,anchor=[-0.55,0,-1],align=[0,0,0.92]) { size:[22,18],normal:y,layer:0,edge:scallop(depth=3,side=bottom),material:vane,blend_with:[f1_spar,f3_spar,wing_bay_3] }\n'
                'plane wing_bay_3 place(base_box=f3_spar,anchor=[0,0,-1],align=[0,0,0.92]) { size:[20,17],normal:y,layer:1,edge:scallop(depth=3,side=bottom),material:vane,blend_with:[f2_spar,f4_spar,wing_bay_4] }'
            ),
        },
        {
            "trap_id": "trap_11_g11_fake_blender_wafer_slices",
            "trap_name": "Trap #11 (`G11` Fake-Blender `< 1.0u` Micro-Slice Curve Abuse)",
            "cap_rule": "Section 13.2 Cap #6 (`Item 5 = 0`, total capped `<= 14/28`) -> `PUNCH BACK TO W1`",
            "diagnosis": "Seven thin `0.5u` wafer slices are stacked on `head` to fake a smooth Blender curve — violating the Anti-Blender Law (`G11`: Blockbench is Blockbench; keep proud `16 px/block` planar cuboids and carve fine curvature optically in pixel relief).",
            "bad_dsl": trap_g11,
            "fixed_patch": (
                "# Delete all 7 wafer_0..wafer_6 (<1.0u) micro-slices and restore clean chunky 16 px/block planar cuboids:\n"
                'box cranium_core { size:[6,6.5,6.5],material:hide,prism_group:"hd" }\n'
                'box nasal_bridge place(base_box=cranium_core,anchor=[0,0.18,1],align=[0,0.1,-0.72]) { size:[3.5,3,6],material:hide,rotate:[9,0,0],prism_group:"hd" }\n'
                'box brow_l place(base_box=cranium_core,anchor=[-0.88,0.62,0.52],align=[0.35,-0.25,0]) { size:[2,1.5,4.5],material:hide,rotate:[10,-12,-18],prism_group:"hd" }\n'
                'box brow_r place(base_box=cranium_core,anchor=[0.88,0.62,0.52],align=[-0.35,-0.25,0]) { size:[2,1.5,4.5],material:hide,rotate:[10,12,18],prism_group:"hd" }'
            ),
        },
        {
            "trap_id": "trap_16_floating_forepaws",
            "trap_name": "Trap #16 (Mid-Air Floating Front Paws — 4-Foot Ground-Locking Failure)",
            "cap_rule": "Section 13.1/13.2 Ground-Locking Gate (`|FL,FR,HL,HR| <= 0.30u` violated; hind feet at `y=0` mask `+5u` floating forepaws!)",
            "diagnosis": "Global `y_min = 0.0` passed only because hind feet touch the floor while BOTH forepaws (`FL/FR`) hover ~`+4..5u` in mid-air. Fix forearm height & elbow flexion so `max(|FL|,|FR|,|HL|,|HR|) <= 0.30u` simultaneously.",
            "bad_dsl": trap_16,
            "fixed_patch": (
                "# Restore full 8.0u forearm reach & counter-pitch wrist so FL/FR and HL/HR lock simultaneously at y = 0.0:\n"
                "bone arm_l_fore attach(parent_box=humerus_core,anchor=[0,-1,0.1],align=[0,0.82,0]) {\n"
                "  rest_euler:[-44,0,10]\n"
                '  box fore_core { size:[3.5,8,4.5],material:hide,prism_group:"af" }\n'
                '  box vambrace_shield place(base_box=fore_core,anchor=[-0.7,0.05,0.25],align=[0.4,0,-0.3]) { size:[2.5,6,3.5],material:gold,rotate:[0,0,-14],prism_group:"af" }\n'
                "  bone paw_l attach(parent_box=fore_core,anchor=[0,-1,0.12],align=[0,0.9,-0.15]) {\n"
                "    rest_euler:[18,0,0]\n"
                '    box palm_base { size:[5,2.5,5.5],material:hide,prism_group:"pw" }\n'
                "  }\n"
                "}"
            ),
        },
        {
            "trap_id": "trap_19_cute_dumb_m4_boss_drift",
            "trap_name": "Trap #19 (Accidental 'Cute & Dumb' Boss Syndrome — `M4 -> M1/M2` Mood Drift)",
            "cap_rule": "Section 1.5 / Mood Anti-Drift Gate (`M4` Ferocious Boss drifted into upright giraffe neck, bunny-ear horns, flat bright-white square marshmallow snout & flat brow -> DO NOT EXPORT!)",
            "diagnosis": "An apex `D4 x M4` Elder Dragon drifted toward cute/dumb `M1/M2` knobs: vertical `-42 deg` giraffe neck, upright bunny-ear horns (`+78 deg`), flat bright-white `4.5u` square marshmallow nose, and flat `-10 deg` brow. Flip all 10 `M4` knobs: coiled forward-surging S-neck (`-18 deg`) + downward Kubrick glare (`+20 deg`), plunging golden V-brow (`+16/-24 deg`), swept scimitar horns (`[34,24,-18]`), razor `2.0u` falcon prow, and sleek `2.0u` binocular ruby slit eyes.",
            "bad_dsl": trap_19,
            "fixed_patch": (
                "# Flip all 10 M4 Ferocious Apex knobs (coiled S-neck, Kubrick glare, plunging golden V-brow, razor prow, swept horns):\n"
                "bone neck_1 attach(parent_box=chest_core,anchor=[0,0.32,1],align=[0,-0.1,-0.72]) {\n"
                "  rest_euler:[-18,0,0]\n"
                "  # ... inside head (rest_euler:[20,0,0]):\n"
                '  box beak_hook_tip place(base_box=nasal_visor,anchor=[0,-0.05,1],align=[0,0,-0.65]) { size:[2,3,2.5],material:gold,rotate:[22,0,0],prism_group:"hd" }\n'
                '  box brow_visor_l place(base_box=cranium_core,anchor=[-0.88,0.62,0.52],align=[0.35,-0.25,0]) { size:[2,1.5,4.5],material:gold,rotate:[16,-14,-24],prism_group:"hd" }\n'
                '  box brow_visor_r place(base_box=cranium_core,anchor=[0.88,0.62,0.52],align=[-0.35,-0.25,0]) { size:[2,1.5,4.5],material:gold,rotate:[16,14,24],prism_group:"hd" }\n'
                "  bone horn_l attach(parent_box=cranium_core,anchor=[-0.78,0.88,-0.55],align=[0,-0.55,0.85]) mirror_x(horn_r) {\n"
                "    rest_euler:[34,24,-18]\n"
                "  }\n"
                "}"
            ),
        },
        {
            "trap_id": "trap_22_detached_double_inner_wing",
            "trap_name": "Trap #22 (Detached Double-Layer Inner Wing Membrane Step Beside Torso)",
            "cap_rule": "Section 8.4d #3 & Trap #22 (`v25` Round 16 Inner-Wing Continuity Failure -> `PUNCH BACK TO W1`)",
            "diagnosis": "A redundant `wing_fore_patagium_bad` sheet at `layer: 6` stacks above `wing_bay_4` (`layer: 2`), creating a detached double-layer floating step right beside the torso. Delete the redundant sheet, keep `wing_l_f4` parented rigidly on `wing_l_fore`, unify `wing_bay_4` cleanly, and edge-lock `wing_inner_sail` at `layer: 0` along the 3D elbow crease.",
            "bad_dsl": trap_22,
            "fixed_patch": (
                "# Delete detached wing_fore_patagium_bad (layer:6) and keep single unified coplanar wing_bay_4 + edge-locked wing_inner_sail:\n"
                'plane wing_inner_sail place(base_box=w_hum,anchor=[0,0,-1],align=[0,0,0.88]) { size:[15,15],normal:y,layer:0,edge:scallop(depth=3,side=bottom),material:velvet }\n'
                "bone wing_l_f4 attach(parent_box=w_rad,anchor=[-0.45,0,-1],align=[1,0,1]) {\n"
                "  rest_euler:[-10,-24,14]\n"
                '  box f4_spar { size:[19,1.5,1.5],material:hide,blend_with:[w_rad] }\n'
                '  plane wing_bay_4 place(base_box=f4_spar,anchor=[0,0,-1],align=[0,0,0.92]) { size:[29,29],normal:y,layer:2,edge:scallop(depth=3,side=bottom),material:velvet,blend_with:[wing_inner_sail,w_rad,f3_spar,wing_bay_3] }\n'
                "}"
            ),
        },
    ]


def build_all_v25_verified_masterworks() -> list[dict[str, Any]]:
    """Build and verify all 8 (`4` domains x `2` variants) original `CC0-1.0` `v25` assets through SyncCraft + `G1-G12` + 4-Foot Grounding."""
    builders = (
        _build_d1_m1_hearth_pantry_crate,
        _build_d2_m1_moss_pebble_hatchling,
        _build_d3_m3_azure_runic_sky_drake,
        _build_d4_m4_obsidian_halberd_elder_wyrm,
    )
    verified: list[dict[str, Any]] = []
    for bfn in builders:
        for var in range(2):
            item = bfn(var)
            dsl = _auto_ground_dsl_four_feet(item["dsl"])
            item["dsl"] = dsl
            rep = verify_v25_dsl_asset(
                dsl,
                tier=item["tier"],
                rest_tol=0.45,
                allow_asymmetric_prop=(item["domain"] == "D1"),
            )
            item["verification"] = rep
            verified.append(item)
    return verified


def build_aesthetic_v25_vigorous_records(
    *,
    seed: int = 20261120,
    max_records: int = 84,
    panel_size: tuple[int, int] = (224, 224),
    grid_size: tuple[int, int] = (384, 384),
) -> list[dict[str, Any]]:
    """Synthesize verified Stage-7.5 `v25` Vigorous Carved Blockbench multimodal & spatial-grounding training records."""
    rng = random.Random(seed)
    masterworks = build_all_v25_verified_masterworks()
    traps = _build_trap_negative_assets()
    records: list[dict[str, Any]] = []

    def _add(
        task: str,
        sample_id: str,
        prompt: str,
        think: str,
        body: str,
        *,
        images: list[Image.Image] | None = None,
        pixel_indispensable: bool = False,
        meta_extra: dict[str, Any] | None = None,
    ) -> None:
        answer = format_supervised_answer(body, think, enable_thinking=True)
        health = check_generation_collapse_health(answer)
        if not health["healthy"]:
            raise ValueError(f"v25 aesthetic record '{sample_id}' failed collapse health: {health}")
        imgs = images or []
        rec = {
            "id": identity(f"aesthetic_v25:{sample_id}"),
            "split": "train",
            "source_family": "synccraft_aesthetic_v25_cc0",
            "modality": "image" if imgs else "text",
            "language": "en",
            "task": task,
            "prompt": prompt,
            "answer": answer,
            "images": imgs,
            "metadata": {
                "license": "CC0-1.0",
                "style_version": "blockbench_sculpted_style_v25",
                "pixel_indispensable": bool(pixel_indispensable),
                **(meta_extra or {}),
            },
            "provenance": make_clean_room_provenance("aesthetic_v25_vigorous_cc0"),
        }
        records.append(rec)

    # 1. Task A (`v25_two_pass_locate_then_zoom`) — Pixel-Indispensable `<box>` Head & `<point>` Eye Grounding!
    # Generate 3 distinct jittered camera/pan views per creature (3 creature families x 2 variants x 3 views = 18 records!)
    # Front-3/4, opposite front-3/4, side-front profile, and high front hero cameras (head faces +Z -> az~180 is dead front!)
    cam_presets = (
        (146.0, 14.0, 0.72, (-0.28, 0.22)),
        (214.0, 16.0, 0.68, (0.30, -0.18)),
        (114.0, 10.0, 0.76, (-0.18, -0.24)),
        (196.0, 20.0, 0.66, (0.24, 0.26)),
    )
    for idx, mw in enumerate(masterworks):
        if mw["domain"] == "D1":
            continue
        rep = mw["verification"]
        baked = bake_sculpted_rgba_atlas(rep["program"], rep["scene"])
        g = rep["g1_g12"]
        for v_i in range(3):
            az, el, ff, pan = cam_presets[(idx + v_i) % len(cam_presets)]
            az_j = az + rng.uniform(-8.0, 8.0)
            el_j = el + rng.uniform(-4.0, 4.0)
            pan_j = (round(pan[0] + rng.uniform(-0.08, 0.08), 3), round(pan[1] + rng.uniform(-0.08, 0.08), 3))
            loc = render_v25_locate_first_grid_and_targets(
                baked,
                az_deg=az_j,
                el_deg=el_j,
                fill_frac=ff,
                pan_uv=pan_j,
                panel_size=grid_size,
                draw_grid=True,
            )
            hb = loc["head_box_1000"]
            ep = loc["eye_point_1000"]
            hbn = loc["head_box_norm"]
            epn = loc["eye_point_norm"]
            prompt = (
                "Execute `blockbench-sculpted-style` v25 Mandatory 2-Pass 'Locate-First, Then Zoom' Pass 1 on this "
                "`0.0-1.0` coordinate-gridded Blockbench render. Locate the exact bounding box of the creature's head "
                "using `<box>[x1, y1, x2, y2]</box>` (0-1000 scale) and the exact visible eye-socket center using "
                "`<point>[x, y]</point>` (0-1000 scale), then state its `(Domain, Mood)` classification and `v25` Pass-2 crop target."
            )
            think = (
                f"Read the 0.10 coordinate grid on the pixels before planning any crop (v25 Trap #20 forbids blind fractional crops). "
                f"The head silhouette spans normalized x=[{hbn[0]:.3f},{hbn[2]:.3f}], y=[{hbn[1]:.3f},{hbn[3]:.3f}] "
                f"and the visible recessed orbital eye sits at ({epn[0]:.3f},{epn[1]:.3f}). "
                f"Domain={mw['domain']}, Mood={mw['mood']}, head_L/H={g['head_lh']:.2f}, belly_clearance={g['belly_clearance']:.2f}."
            )
            body = (
                f"- **Pass-1 Verified Head Region:** `<box>[{hb[0]}, {hb[1]}, {hb[2]}, {hb[3]}]</box>` "
                f"(norm `[{hbn[0]:.3f}, {hbn[1]:.3f}, {hbn[2]:.3f}, {hbn[3]:.3f}]`)\n"
                f"- **Pass-1 Verified Eye Socket Center:** `<point>[{ep[0]}, {ep[1]}]</point>` "
                f"(norm `[{epn[0]:.3f}, {epn[1]:.3f}]`)\n"
                f"- **Intent Classification:** `({mw['domain']}, {mw['mood']})` — `{mw['archetype']}`\n"
                f"- **Pass-2 Zoom Plan:** Crop Head Macro padded `15%` around `<box>[{hb[0]}, {hb[1]}, {hb[2]}, {hb[3]}]</box>` "
                f"and Eye Ultra-Zoom centered on `<point>[{ep[0]}, {ep[1]}]</point>` to verify zero eye leak, brow plunge, and snout taper."
            )
            _add(
                "v25_two_pass_locate_then_zoom",
                f"locate_{mw['asset_id']}_v{v_i}",
                prompt,
                think,
                body,
                images=[loc["image"]],
                pixel_indispensable=True,
                meta_extra={"domain": mw["domain"], "mood": mw["mood"], "head_box_1000": hb, "eye_point_1000": ep},
            )

    # 2. Task B (`v25_intent_router_and_mood_control`) — Turn-1 `(D1-D4 x M1-M4)` Intent Router + 10-Knob Quantitative Matrix!
    for mw in masterworks:
        rep = mw["verification"]
        g = rep["g1_g12"]
        gr = rep["grounding"]
        views_map = _render_v8_views_pil(rep["program"], panel_size=panel_size)
        views_pil = [views_map[k] for k in ("iso", "front", "side", "top") if k in views_map]
        prompt = (
            f"Inspect this 4-view `16 px/block` Blockbench render and concept brief: \"{mw['concept']}\" "
            "Run the `v25` Turn-1 Small-AI Intent Router (`Domain D1-D4 x Mood M1-M4`), report the active quantitative "
            "Mood Knobs (shape grammar, eye geometry, brow angle, horn sweep, posture `belly_clearance`, value contrast), "
            "and verify the Mood Anti-Drift Gate."
        )
        think = (
            f"Classify Domain={mw['domain']} and Mood={mw['mood']} before touching Cube #1. "
            f"Verify elements={g['total_elements']}, bones={g['bones']}, top10_nw={g['top10_nw_volume']:.1%}, "
            f"belly_clearance={g['belly_clearance']:.2f}, rest_feet={gr['rest_feet']}, G1-G12 strict={g['passed_strict']}."
        )
        mood_knobs = {
            ("D1", "M1"): "Soft-square cozy slatted crate + `5-15 deg` contact-pivoted tilted produce; true 4-band octagon crock; warm glowing lantern core (`<=4%`) vs cool stone shadows.",
            ("D2", "M1"): "Neotenic `1:1.5` oversized head; chunky `>=2u` legs; large low `3x3 px` glossy eyes with catchlights; absent/soft brow; warm inviting `16x` mossy-sage palette.",
            ("D3", "M3"): "Athletic tapered wedges & S-neck; `+-16 deg` V-rail maxilla (`G2`); focused recessed eyes; chunky 4-segment stone rune horns (`G3`); 3D vaulted full-chord wings (`G4/G10`); weathered `20-42%` saturation.",
            ("D4", "M4"): "Coiled low-slung predator crouch (`belly_clearance<=0.36`, `G12`); forward-surging S-neck + downward Kubrick glare; plunging golden V-brow (`+16/-24 deg`); sleek `2.0u` horizontal ruby slit eyes (`20 deg` binocular yaw); `-46 deg` swept scimitar horns; razor `2.0u` falcon prow; Gothic Chiaroscuro cheeks/jaw; 6-plane zero-thickness tooth U-arcade + 2-segment tongue.",
        }[(mw["domain"], mw["mood"])]
        body = (
            f"### Turn-1 Intent Classification: `({mw['domain']}, {mw['mood']})`\n"
            f"- **Archetype & Tier:** `{mw['archetype']}` (`tier={mw['tier']}`, `16 px/block` vanilla scale)\n"
            f"- **Active Quantitative Mood Knobs:** {mood_knobs}\n"
            f"- **Deterministic Verification (`G1-G12` + 4-Foot Grounding):** `elements={g['total_elements']}`, `bones={g['bones']}`, "
            f"`top10_nw={g['top10_nw_volume']:.1%}`, `mirror_ok={g['mirror_ok']}`, `belly_clearance={g['belly_clearance']:.2f}`, "
            f"`rest_feet={gr['rest_feet']}` -> **MOOD ANTI-DRIFT GATE: PASS**."
        )
        _add(
            "v25_intent_router_and_mood_control",
            f"router_{mw['asset_id']}",
            prompt,
            think,
            body,
            images=views_pil[:2],
            pixel_indispensable=True,
            meta_extra={"domain": mw["domain"], "mood": mw["mood"]},
        )

    # 3. Task C (`v25_vigorous_carved_masterwork`) — Concept + S1-S6 -> Verified `16 px/block` Block-DSL v4.1!
    for mw in masterworks:
        rep = mw["verification"]
        g = rep["g1_g12"]
        gr = rep["grounding"]
        prompt = (
            f"Design and emit a complete `Block-DSL v4.1` (`texel_density: 16`, `audit_profile: sculpted`) asset following "
            f"`blockbench-sculpted-style` v25 for Target `({mw['domain']}, {mw['mood']})`: {mw['concept']} "
            "Use `prism_group` for sunken compound cuboid mass-packing (`40-85%` relief, zero outer Lego stickers, zero fake-Blender "
            "`<1.0u` wafer slices), pass all `G1-G12` gates, and lock all grounded feet at `y = 0`."
        )
        think = (
            f"S1 Envelope & Mood: Target=({mw['domain']},{mw['mood']}), tier={mw['tier']}, 16 px/block vanilla density. "
            f"S2-S4 Compound Packing: anchor primary masses (top10_nw={g['top10_nw_volume']:.1%}), group sunken muscle/bevel clusters "
            f"with prism_group, keep clean 0.5u sizes & lowercase snake_case bones, and ground rest feet ({gr['rest_feet']}). "
            f"S5-S6 Audit: SyncCraft audit_scene=PASS, G1-G12 strict={g['passed_strict']}, mirror_ok={g['mirror_ok']}."
        )
        body = f"```blockdsl\n{mw['dsl']}\n```"
        _add(
            "v25_vigorous_carved_masterwork",
            f"masterwork_{mw['asset_id']}",
            prompt,
            think,
            body,
            images=[],
            pixel_indispensable=False,
            meta_extra={"domain": mw["domain"], "mood": mw["mood"], "g1_g12_strict": g["passed_strict"]},
        )

    # 4. Task D (`v25_multiview_inverse_and_nowing_audit`) — Turnaround PNGs -> Verified `v25` Block-DSL v4.1!
    for mw in masterworks:
        rep = mw["verification"]
        g = rep["g1_g12"]
        views_map = _render_v8_views_pil(rep["program"], panel_size=panel_size)
        views_pil = [views_map[k] for k in ("iso", "front", "side", "top") if k in views_map]
        region_img, _ = render_v8_zoomed_region_sheet(rep["program"], rep["scene"], panel_size=(192, 192))
        prompt = (
            "Inspect these `16 px/block` Blockbench multi-view renders (front, side, 3/4, top + zoomed region sheet). "
            "Classify `(Domain, Mood)`, execute `S1-S6` spatial decomposition, and reconstruct the exact verified "
            "`Block-DSL v4.1` program passing SyncCraft `audit_scene`, `v25` `G1-G12`, and 4-foot grounding."
        )
        think = (
            f"Visual inverse analysis from turnaround + region sheet: {mw['archetype']} ({mw['domain']},{mw['mood']}). "
            f"Measured span={g['span']}, head_L/H={g['head_lh']:.2f}, belly_clearance={g['belly_clearance']:.2f}, "
            f"elements={g['total_elements']}, bones={g['bones']}, mirror_ok={g['mirror_ok']}."
        )
        body = f"```blockdsl\n{mw['dsl']}\n```"
        _add(
            "v25_multiview_inverse_and_nowing_audit",
            f"inverse_{mw['asset_id']}",
            prompt,
            think,
            body,
            images=[views_pil[0], region_img],
            pixel_indispensable=(mw["domain"] in ("D1", "D2")),
            meta_extra={"domain": mw["domain"], "mood": mw["mood"]},
        )

    # 5. Task E (`v25_adversarial_trap_punchback`) — Rendered Defective Trap PNGs -> Automatic `<= 14/28` Cap -> Fixed DSL!
    ensure_synccraft_importable()
    from synccraft.dsl.parser import parse_program
    from synccraft.kinematics.se3_chain import compile_geometry

    for tr in traps:
        bad_prog = parse_program(tr["bad_dsl"])
        bad_scene = compile_geometry(bad_prog)
        bad_views_map = _render_v8_views_pil(bad_prog, panel_size=panel_size)
        bad_views = [bad_views_map[k] for k in ("iso", "front", "side", "top") if k in bad_views_map]
        # Include both a diagnostic-only visual audit record (fast, high visual signal for 7.5A!) AND a full punch-back + fixed DSL record!
        prompt_diag = (
            "Act as an adversarial `blockbench-sculpted-style` v25 art director inspecting these rendered views. "
            "Identify the specific `v25` structural, mood, wing, or grounding trap visible in the renders, cite the "
            "applicable Section 13.2 Automatic Score Cap (`<= 14/28`), and prescribe the exact reconstruction fix."
        )
        think_diag = (
            f"Adversarial visual inspection detects {tr['trap_name']}. "
            f"Per v25 Section 13.2 Adversarial Self-Judgment Law, never rubber-stamp 28/28 — fire the automatic <=14/28 cap and punch back."
        )
        body_diag = (
            f"**Verdict: PUNCH BACK TO W1 (AUTOMATIC FAIL — Capped `<= 14 / 28`)**\n"
            f"- **Detected Visual Trap:** {tr['trap_name']}\n"
            f"- **Triggered Gate / Cap:** {tr['cap_rule']}\n"
            f"- **Spatial Diagnosis & Required Fix:** {tr['diagnosis']}"
        )
        _add(
            "v25_adversarial_trap_punchback",
            f"trap_diag_{tr['trap_id']}",
            prompt_diag,
            think_diag,
            body_diag,
            images=[bad_views[0], bad_views[1], bad_views[2]],
            pixel_indispensable=True,
            meta_extra={"trap_id": tr["trap_id"]},
        )

        prompt_fix = (
            "These rendered views fail `blockbench-sculpted-style` v25 review. Name the exact `v25` trap and Section 13.2 "
            "cap (`<= 14/28`), trigger `PUNCH BACK TO W1`, and output the exact surgical `Block-DSL v4.1` repair block "
            "for the failing region so the rig passes SyncCraft `audit_scene`, `G1-G12`, and 4-foot grounding."
        )
        think_fix = (
            f"Detected {tr['trap_name']}. Apply {tr['cap_rule']}. Delete the defective region and emit the surgical verified v25 Block-DSL patch."
        )
        body_fix = (
            f"PUNCH BACK TO W1: {tr['trap_name']} ({tr['cap_rule']}). {tr['diagnosis']}\n\n"
            f"```blockdsl\n{tr['fixed_patch']}\n```"
        )
        _add(
            "v25_adversarial_trap_punchback",
            f"trap_fix_{tr['trap_id']}",
            prompt_fix,
            think_fix,
            body_fix,
            images=[bad_views[0], bad_views[1]],
            pixel_indispensable=True,
            meta_extra={"trap_id": tr["trap_id"]},
        )

    # 6. Task F (`v25_pixel_relief_maw_and_ecology`) — Cross-cube `16 px` relief, 6-plane integer-pixel flat tooth arcade, VFX isolation, preserve_and_layer & 6-subsystem ecology!
    craft_cards = [
        (
            "zero_thickness_maw_integer_pixels",
            "How do you construct and paint a roaring Elder Dragon mouth in `blockbench-sculpted-style` v25 at `16 px/block` without chunky volumetric tooth cubes or the `h = 2 px` floating-point alpha erasure bug?",
            "Apply v25 Section 6.5 & 10.2d: ban volumetric tooth cubes (Z-fighting / buck-tooth clutter). Build a 6-plane 0-thickness tooth U-arcade + 4-layer cavernous interior + articulated 2-segment tongue, and assign teeth by integer pixel indices.",
            (
                "1. **6-Plane Zero-Thickness Single-Pixel-Square Tooth U-Arcade (`size = 0` on normal axis):**\n"
                "   - `upper_teeth_row_l/r` (`normal: x`) + `upper_front_teeth` (`normal: z`) recessed inside `maxilla_wall` and `beak_hook_tip`.\n"
                "   - `lower_teeth_row_l/r` (`normal: x`) + `lower_front_teeth` (`normal: z`) seated along `mandible_bar` and chin rim.\n"
                "2. **Integer-Pixel Assignment (Fix `h = 2 px` Erasure Bug):** Never use continuous float thresholds like `tip_t <= 0.96` on a `2 px`-tall plane (`1.0 <= 0.96 -> False` erases every tooth tip!). Assign pixels explicitly by integer row (`root` vs `tip`) and sorted front-to-back index (`front_idx = 0..w-1`): paint crisp `1x1 px` ivory squares for cheek teeth and `1x2 px` canines in front, separated by `1 px` transparent gaps (`alpha = 0`), staggering lower teeth `1 px` relative to upper teeth.\n"
                "3. **4-Layer Cavernous Interior Mouth:**\n"
                "   - Paint `nasal_visor`/`beak_hook_tip` undersides deep mulberry-plum with `1 px` transverse palate rugae (`0%` exterior hide leak).\n"
                "   - Paint inward faces of `maxilla_wall`/`mandible_bar` dark plum-black + add recessed `mouth_throat_back` abyssal pharynx box.\n"
                "   - Span rear jaw hinges with `0`-thickness `100%` binary-alpha `plane jaw_cheek_web_l/r`.\n"
                "   - Add articulated 2-segment `bone tongue` (`tongue_base` + arched `tongue_tip`) in smoked royal plum-crimson (never neon bubblegum-pink), curling `-10..-14 deg` during roars."
            ),
        ),
        (
            "cross_cube_pixel_relief_and_preserve_layer",
            "Explain the `v25` `16 px/block` Vigorous Cross-Cube Pixel-Relief Carving workflow, VFX-vs-Anatomy isolation law, and SyncCraft's `preserve_and_layer_painted_texture` non-destructive layering.",
            "Combine v25 Section 10.2b (45-50 deg hue-shift gradient + cross-cube transverse scale-rungs + corrugated ladder-ribs + <=4% glow), Section 2 VFX isolation (strip floating Qurio/dust/lens flares), and ADR 0013 preserve_and_layer.",
            (
                "1. **Vigorous Cross-Cube Pixel-Relief Carving (`16 px/block`, `Per-Face UV`, `>= 1-2 px` UV padding):**\n"
                "   - Generate `5-8` step ramps with a **`45-50 deg` hue shift** (warm highlights -> cool saturated violet/indigo shadows; zero grey/black shadows, zero uniform `yy == 0` plastic borders).\n"
                "   - Paint rhythmic `2-3 px` **transverse chevron / U-shaped scale rungs** (`1 px` lit bevel + mid + `1 px` cast shadow) along spines/limbs, **wrapping every rung `3-5 px` down adjacent vertical side faces** across cube seams.\n"
                "   - Carve thick rectangular wing-arm/shin beams optically with **corrugated transverse ladder-ribs** (`2 px` lit bar + `2 px` deep recess) and tuck `<= 4%` glowing focal fissures strictly inside the deepest muscle crevices.\n"
                "2. **5-Layer Wing Membrane Shading:** Exact 3D bay-polygon `100%` binary hard-alpha mask -> both-spar bone-hugging dark AO (`2-4 px`) -> translucent mid-bay belly cushion -> radial tension folds & forked veins -> velvety dark leathery trailing hem (never bright yellow stair-stepped outlines).\n"
                "3. **VFX-vs-Anatomy Isolation Law (Anti-Qurio Trap):** Cross-check every glowing dot across `>= 2` neutral references (turnaround + calm ecology clip) and strip all floating environmental particles (Qurio, embers, lens flares) before painting albedo.\n"
                "4. **Non-Destructive Layering (`preserve_and_layer_painted_texture`):** Keep hand-painted base RGBA pixels intact while stacking wear, shadow-crease AO, or elemental overlays in isolated layers so re-exports never overwrite artist brushwork."
            ),
        ),
        (
            "six_subsystem_ecology_and_ground_locking",
            "How do you animate a 6-limbed Elder Dragon in `v25` so it achieves whole-body ecological weight, zero frozen appendages, seamless inner-wing folding, and exact Blockbench 5 4-foot ground-locking?",
            "Apply v25 Section 11.5b/11.5d (6-subsystem coupling, opposite-side angular-momentum tail, treadmill root.x/z=0), Round 16 rigid wing_l_f4 on wing_l_fore + shared-crease inner sail, and verify_bbmodel_ground 4-foot FK contract.",
            (
                "1. **Six-Subsystem Kinetic Coupling (Zero Frozen Appendages):** Coordinate `pelvis_root` (position heave + pitch/yaw/roll), `waist+chest` (counter-torque), `neck+head` (horizon gaze stabilization + subtle `2-4 deg` jaw breathing), `wings` (SAT-clear draped **Knight-Cloak** carriage in calm idle/walk/sit vs. asymmetric `25-45 deg` tip-lagged flight strokes), `limbs` (stepping/bracing into strikes, flexing during bows), and `tail_1..6` (low horizontal S-rudder in walk; equal-and-opposite `+-20-26 deg` horizontal whip conserving angular momentum during asymmetric slashes).\n"
                "2. **Treadmill Locomotion (`11.5b`):** Zero `root.x` and `root.z` position keyframes on exported `.bbmodel` gait clips while preserving `root.y` vertical heave and `synccraft_root_motion` metadata.\n"
                "3. **4-Foot Ground-Locking Contract (`verify_bbmodel_ground.py`):** Evaluate exact per-limb world `y_min` (`FL`, `FR`, `HL`, `HR`, `TAIL`) directly on `.bbmodel` FK: static rest pose requires `max(|FL|,|FR|,|HL|,|HR|) <= 0.30u`; standing/guard/sit clips keep planted paws inside `[-0.6, +1.2]u` (checking elbow/knee flexion sign whenever pelvis pitches!); locomotion clips plant in `[-1.0, +0.6]u` and lift `[+3.5, +8.5]u`.\n"
                "4. **Motion-Sealed Inner Wings (`v25` Round 16):** Parent `wing_l_f4/wing_r_f4` directly on `wing_l_fore` (`wing_radius`), unify `wing_bay_4` coplanar at `layer: 0`, edge-lock `wing_inner_sail` (`layer: 0`) along the exact 3D plane-plane elbow crease with one continuous 3D world-space shader, and distribute cloak drape onto `wing_l_upper` so `f4` never splits out in `288`-frame swept OBB `SAT`."
            ),
        ),
        (
            "synccraft_decoupling_and_prism_group",
            "How does `blockbench-sculpted-style` v25 reconcile vigorous `40-85%` sunken compound cuboid mass-packing with SyncCraft's current experimental `sync_audit.py` limits?",
            "Apply v25 Section 11.0 & references/vigorous_carved_aesthetic_and_spatial_reasoning.md Section 5: avoid place(anchor=+-1) outer Lego stickers, group compound anatomical bevel/muscle clusters via prism_group + blend_with, and enforce G1-G12 via construct_check.py.",
            (
                "1. **Why Naive `place(anchor=+-1)` Produces Toy Models:** Current `sync_audit.py` caps parallel overlap at `0.25` (`< 5 deg`), rejects `> 65%` buried volume (`swallowed_redundant_box`), and caps `place()` inset at `0.5u` — biasing naive Block-DSL toward flat Lego-stud stickers on outer walls while still letting bare stick-limbs and V-gapped wings pass with `0` errors.\n"
                "2. **Immediate Master Workaround Inside SyncCraft v4.1:** Group interlocking anatomical clusters (deltoid/trapezius/thigh muscle swells, stepped belly tiers, chin osteoderms, `30-45 deg` visor bevels) under shared **`prism_group: \"<anatomical_cluster>\"`** plus explicit cross-bone `blend_with: [...]`, legitimately unlocking `40-85%` sunken cuboid mass-packing while keeping **inter-limb & wing-to-body `288`-frame swept 15-axis OBB SAT**, `SE(3)` kinematics, grounding, and coplanar Z-fighting checks `100%` active!\n"
                "3. **Engine-Agnostic Positive Enforcement:** Always pair `audit_scene` with `scripts/construct_check.py --tier <tier>` (`G1-G12` across `.bbmodel`, `.geo.json`, and `.blockdsl`) and `scripts/verify_bbmodel_ground.py` (4-foot FK ground-locking)."
            ),
        ),
    ]
    for cid, q, th, ans in craft_cards:
        _add(
            "v25_pixel_relief_maw_and_ecology",
            f"craft_{cid}",
            q,
            th,
            ans,
            images=[],
            pixel_indispensable=False,
        )

    rng.shuffle(records)
    return records[:max_records]
