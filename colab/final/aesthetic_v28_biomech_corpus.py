"""Stage-9 `Blockbench_sculpted_style-v28` & SyncCraft `ba969f3` Biomechanical Corpus Builder (`CC0-1.0`).

Encodes the complete `v28` sculpted Blockbench formal aesthetic, **Gentle & Approachable (Pole A)** vs.
**Furious & Unapproachable (Pole B)** temperament polarity laws, **Domain `D5` Rigid Vehicle / Mecha**
construction grammar (`0%` `Sim(3)` metal bulge), **Unified `SO(3)/SE(3)/Sim(3)` Biomechanical Solver**
(`auto_rig: true`, `in_place: true`, closed-loop Standing Limb IK Pinning `ground_pin_limbs`,
`wing_anchor_locked`, `flight(mode=hover|cruise|mantle|swim_fins)`, and volume-preserving `Sim(3)`
`scale_bulge` with `preserve_grounded_limb_length`), and live-verified **SyncCraft `ba969f3`**
capabilities (additive clip `layer` & `effect touchdown`, `preserve_and_layer_painted_texture` /
`rebake_selected_faces`, `invert_clip_to_gait` / `search_or_repair_gait`, and `train`-only 5-task
proposer supervision with `0%` `holdout_gate_d` overlap).

Hard Quality Invariants:
- 100% original commercial-clean (`CC0-1.0`) assets; zero third-party IP names, zero `eval()`.
- Every emitted Block-DSL v4.1 program parses and compiles cleanly under SyncCraft's real
  `parse_program` + `compile_geometry` + `audit_scene` + `G1-G12` + `audit_behavior_clip`.
- Full-textured, key+rim-lit RGB studio renders ONLY (`clay=False`, zero grayscale clay or B&W cutouts;
  every rendered image satisfies foreground chroma L1 > 5.0).
- Exact 4-primitive visual grounding (`<point x=".." y=".." z="near|mid|far" label=".."/>`,
  `<box x1=".." y1=".." x2=".." y2=".." label=".."/>`, `<dash x1=".." y1=".." x2=".." y2=".." label=".."/>`,
  `<circle cx=".." cy=".." r=".." label=".."/>`) in `[0..1000]` normalized coordinates.
- ~20% bilingual Chinese (`zh`) coverage across polarity, D5 mechanical, biomech, and adversarial punch-back.
"""
from __future__ import annotations

import functools
import json
import math
import random
import re
from typing import Any

import numpy as np
from PIL import Image

from .aesthetic_v25_vigorous_corpus import _auto_ground_dsl_four_feet, _compact_dsl
from .common import identity
from .reasoning_agentic_corpus import check_generation_collapse_health
from .sculpted_style_corpus import bake_sculpted_rgba_atlas
from .spatial_3d_primitives_corpus import (
    STUDIO_LIGHTING_PRESETS,
    _clamped_dist,
    _repose_baked_for_scene,
    fmt_box,
    fmt_circle,
    fmt_dash,
    fmt_point,
    render_shaded_3d_with_spatial_annotations,
)
from .synccraft_bridge import (
    ensure_synccraft_importable,
    format_supervised_answer,
    make_clean_room_provenance,
)
from .v25_blockbench_verifier import verify_v25_dsl_asset
from .visual_primitives_corpus import BANNED_BENCHMARK_SUBSTRINGS, _clamp_1000

AESTHETIC_V28_TASKS: tuple[str, ...] = (
    "v28_temperament_polarity_and_router",
    "v28_d5_vehicle_mecha_rig",
    "v28_pole_a_vs_pole_b_masterworks",
    "v28_biomech_solver_and_ground_pin",
    "v28_synccraft_ba969f3_layers_and_proposer",
    "v28_adversarial_mood_drift_punchback",
)

# Holdout Gate D identifiers strictly excluded from all Stage-9 training supervision
_GATE_D_HOLDOUT_ASSET_IDS: frozenset[str] = frozenset(
    {
        "crest_raptor_holdout",
        "carapace_beetle_holdout",
    }
)
_GATE_D_HOLDOUT_BODY_PLANS: frozenset[str] = frozenset(
    {
        "biped_theropod",
        "hexapod_arthropod",
    }
)
_GATE_D_HOLDOUT_AUTHORS: frozenset[str] = frozenset(
    {
        "synccraft_holdout_bench_c",
        "synccraft_holdout_bench_d",
    }
)


# ---------------------------------------------------------------------------
# 1. Original Commercial-Clean (`CC0-1.0`) v28 Masterwork DSL Builders
# ---------------------------------------------------------------------------

def _compact_dsl_v28(dsl: str) -> str:
    """Compact Block-DSL v4.1 whitespace outside quotes without altering AST semantics."""
    lines: list[str] = []
    for raw in dsl.splitlines():
        s = raw.strip()
        if not s:
            continue
        s = re.sub(r"\s*,\s*", ",", s)
        s = re.sub(r"\s*:\s*", ":", s)
        s = re.sub(r"\s*=\s*", "=", s)
        s = re.sub(r"\s*\{\s*", "{", s)
        s = re.sub(r"\s*\}\s*", "}", s)
        lines.append(s)
    return "\n".join(lines)


def _build_d5_basalt_survey_armored_crawler(variant: int = 0) -> dict[str, Any]:
    """`(D5, M2 Mechanical Rigid)` Armored 4-wheel survey crawler with recessed wheel wells, 4-band octagon tires,
    Ackermann front steering knuckles, top sensor turret, rear cargo ramp, and `0%` `Sim(3)` metal bulge."""
    palettes = (
        (
            "obsidian_amber",
            [0.38, 0.08, 250],
            [0.68, 0.18, 56],
            [0.24, 0.04, 260],
            [0.66, 0.14, 210],
            [0.88, 0.19, 76],
        ),
        (
            "terracotta_teal",
            [0.50, 0.14, 38],
            [0.48, 0.13, 198],
            [0.24, 0.04, 250],
            [0.70, 0.12, 220],
            [0.88, 0.18, 82],
        ),
    )
    tag, armor_lch, trim_lch, tread_lch, glass_lch, beacon_lch = palettes[variant % len(palettes)]

    dsl = f"""
@meta {{ archetype:entity, target_profile:bedrock_entity, texel_density:16, units_per_block:16, dsl_version:"4.1", size_step:0.5, audit_profile:sculpted }}
@layout {{
  budget max_boxes = 72
  budget max_bones = 30
  envelope max_span = [32, 24, 42]
  envelope grounded = true
}}
@palette {{
  armor: ramp(base_lch={armor_lch}, steps=5, hue_shift=[-16,18])
  trim: ramp(base_lch={trim_lch}, steps=4, hue_shift=[-12,14])
  tread: ramp(base_lch={tread_lch}, steps=4, hue_shift=[-10,10])
  glass: ramp(base_lch={glass_lch}, steps=4, hue_shift=[-10,12])
  beacon: ramp(base_lch={beacon_lch}, steps=4, hue_shift=[-6,8], emissive=true)
}}
@rig {{
  root bone chassis_root {{
    pivot_local:[0,4.5,0]
    box chassis_frame {{ size:[10,3,22], material:armor, prism_group:"ch" }}
    box skid_plate place(base_box=chassis_frame, anchor=[0,-1,0], align=[0,0.55,0]) {{ size:[7.5,1.5,19], material:tread, prism_group:"ch" }}
    bone bumper_assy attach(parent_box=chassis_frame, anchor=[0,-0.1,1], align=[0,0,-0.95]) {{
      box front_bumper {{ size:[11,2.5,2.5], material:trim, prism_group:"bp", blend_with:[chassis_frame] }}
      box headlamp_l place(base_box=front_bumper, anchor=[-0.72,0.3,1], align=[0,0,-0.6]) {{ size:[2,1.5,1], material:beacon, prism_group:"bp" }}
      box headlamp_r place(base_box=front_bumper, anchor=[0.72,0.3,1], align=[0,0,-0.6]) {{ size:[2,1.5,1], material:beacon, prism_group:"bp" }}
    }}
    bone cab_module attach(parent_box=chassis_frame, anchor=[0,1,0.42], align=[0,-0.92,0]) {{
      box cab_hull {{ size:[10,5.5,8.5], material:armor, prism_group:"cb", blend_with:[chassis_frame] }}
      box windshield_glass place(base_box=cab_hull, anchor=[0,0.42,0.92], align=[0,-0.25,-0.55]) {{ size:[9,3.5,1.5], material:glass, rotate:[24,0,0], prism_group:"cb" }}
      box roof_rack place(base_box=cab_hull, anchor=[0,1,-0.1], align=[0,-0.75,0]) {{ size:[8,1,6.5], material:trim, prism_group:"cb" }}
      bone fender_fl attach(parent_box=cab_hull, anchor=[-1,0.1,0.18], align=[0.95,-0.5,0]) mirror_x(fender_fr) {{
        box arch_f {{ size:[3.5,1.5,6.5], material:trim, rotate:[0,0,-10], blend_with:[cab_hull] }}
      }}
    }}
    bone cargo_module attach(parent_box=chassis_frame, anchor=[0,1,-0.45], align=[0,-0.92,0]) {{
      box cargo_bed {{ size:[10,4.5,9.5], material:armor, prism_group:"cg", blend_with:[chassis_frame] }}
      bone fender_rl attach(parent_box=cargo_bed, anchor=[-1,0.15,-0.16], align=[0.95,-0.5,0]) mirror_x(fender_rr) {{
        box arch_r {{ size:[3.5,1.5,6.5], material:trim, rotate:[0,0,-10], blend_with:[cargo_bed] }}
      }}
      bone turret_yaw attach(parent_box=cargo_bed, anchor=[0,1,0.1], align=[0,-0.92,0]) {{
        joint_limits:[[-2,2],[-180,180],[-2,2]]
        box turret_base {{ size:[5,2,5], material:trim, prism_group:"tr", blend_with:[cargo_bed] }}
        bone sensor_pitch attach(parent_box=turret_base, anchor=[0,1,0], align=[0,-0.9,0]) {{
          joint_limits:[[-12,45],[-2,2],[-2,2]]
          box sensor_head {{ size:[4,2.5,4.5], material:armor, prism_group:"sp", blend_with:[turret_base] }}
          box sensor_lens place(base_box=sensor_head, anchor=[0,0.1,1], align=[0,0,-0.6]) {{ size:[2.5,1.5,1], material:beacon, prism_group:"sp" }}
        }}
      }}
      bone hatch_door attach(parent_box=cargo_bed, anchor=[0,-0.15,-1], align=[0,-1,1]) {{
        joint_limits:[[-110,10],[-2,2],[-2,2]]
        box ramp_slab {{ size:[8,3,1.5], material:trim, blend_with:[cargo_bed] }}
      }}
    }}
    bone steer_fl attach(parent_box=chassis_frame, anchor=[-1,-0.55,0.56], align=[0.95,0,0]) mirror_x(steer_fr) {{
      joint_limits:[[-2,2],[-35,35],[-2,2]]
      box knuckle_hub {{ size:[2,1.5,1.5], material:tread, blend_with:[chassis_frame] }}
      bone wheel_axle_fl attach(parent_box=knuckle_hub, anchor=[-1,0,0], align=[1,0,0]) {{
        joint_limits:[continuous,[-2,2],[-2,2]]
        box tire_b0 {{ size:[3,6,2.5], material:tread, rotate:[0,0,0], prism_group:"wf", blend_with:[knuckle_hub] }}
        box tire_b1 place(base_box=tire_b0, anchor=[0,0,0], align=[0,0,0]) {{ size:[3,6,2.5], material:tread, rotate:[45,0,0], prism_group:"wf" }}
        box tire_b2 place(base_box=tire_b0, anchor=[0,0,0], align=[0,0,0]) {{ size:[3,6,2.5], material:tread, rotate:[90,0,0], prism_group:"wf" }}
        box tire_b3 place(base_box=tire_b0, anchor=[0,0,0], align=[0,0,0]) {{ size:[3,6,2.5], material:tread, rotate:[135,0,0], prism_group:"wf" }}
        box hub_f place(base_box=tire_b0, anchor=[0,0,0], align=[0,0,0]) {{ size:[3.5,2.5,2.5], material:trim, prism_group:"wf" }}
      }}
    }}
    bone rear_strut_l attach(parent_box=chassis_frame, anchor=[-1,-0.55,-0.58], align=[0.95,0,0]) mirror_x(rear_strut_r) {{
      box rear_hub_stub {{ size:[2,1.5,1.5], material:tread, blend_with:[chassis_frame] }}
      bone wheel_axle_rl attach(parent_box=rear_hub_stub, anchor=[-1,0,0], align=[1,0,0]) {{
        joint_limits:[continuous,[-2,2],[-2,2]]
        box rtire_b0 {{ size:[3,6,2.5], material:tread, rotate:[0,0,0], prism_group:"wr", blend_with:[rear_hub_stub] }}
        box rtire_b1 place(base_box=rtire_b0, anchor=[0,0,0], align=[0,0,0]) {{ size:[3,6,2.5], material:tread, rotate:[45,0,0], prism_group:"wr" }}
        box rtire_b2 place(base_box=rtire_b0, anchor=[0,0,0], align=[0,0,0]) {{ size:[3,6,2.5], material:tread, rotate:[90,0,0], prism_group:"wr" }}
        box rtire_b3 place(base_box=rtire_b0, anchor=[0,0,0], align=[0,0,0]) {{ size:[3,6,2.5], material:tread, rotate:[135,0,0], prism_group:"wr" }}
        box hub_r place(base_box=rtire_b0, anchor=[0,0,0], align=[0,0,0]) {{ size:[3.5,2.5,2.5], material:trim, prism_group:"wr" }}
      }}
    }}
  }}
}}
@style {{
  shade: banded(steps=5, direction=down, cluster=diagonal)
  glow_bleed: 2
}}
@behavior {{
  clip drive_roll {{
    duration_s: 2.0, loop: true
    track wheel_axle_fl: spin(axis=x, amp_deg=360, freq_hz=0.5)
    track steer_fr__wheel_axle_fl: spin(axis=x, amp_deg=360, freq_hz=0.5)
    track wheel_axle_rl: spin(axis=x, amp_deg=360, freq_hz=0.5)
    track rear_strut_r__wheel_axle_rl: spin(axis=x, amp_deg=360, freq_hz=0.5)
    track steer_fl: wave(axis=y, amp_deg=12, freq_hz=0.5)
    track steer_fr: wave(axis=y, amp_deg=12, freq_hz=0.5)
    track turret_yaw: wave(axis=y, amp_deg=30, freq_hz=0.5)
    effect 0.5: {{ sound: "synccraft:crawler_drive_hum" }}
  }}
  clip hatch_deploy {{
    duration_s: 2.0, loop: true
    track hatch_door: hinge(axis=x, amp_deg=-60, freq_hz=0.5)
    track sensor_pitch: wave(axis=x, amp_deg=10, freq_hz=0.5, bias_deg=6)
  }}
}}
"""
    return {
        "asset_id": f"basalt_survey_armored_crawler_{tag}",
        "base_name": "basalt_survey_armored_crawler",
        "palette_tag": tag,
        "domain": "D5",
        "mood": "M2",
        "polarity": "mechanical_rigid",
        "tier": "standard",
        "archetype": "D5 x M2 Armored Survey Crawler (Rigid SO(3)/SE(3), 0% Sim(3) Bulge)",
        "concept": (
            f"Basalt Survey Armored Crawler ({tag} livery): chassis-first wheelbase & recessed fender wheel wells, "
            "sloped glass cab, 4-band true octagon wheels on continuous `wheel_axle` joints, Ackermann front "
            "`steered_wheel_knuckle` carriers, dorsal `turret_yaw` + `sensor_pitch`, rear `hatch_door` cargo ramp, "
            "and strictly 0% squishy `Sim(3)` scale bulge on rigid metal."
        ),
        "dsl": _compact_dsl(dsl),
    }


def _build_d5_aether_cargo_airship_rig(variant: int = 0) -> dict[str, Any]:
    """`(D5, M3 Mechanical Rigid)` Keel-first cargo airship with vaulted chamfered gasbag hull,
    sloped glass cockpit, twin `rotor_propeller` nacelles, `deck_turret`, rear `cargo_hatch`, and `0%` metal squish."""
    palettes = (
        (
            "copper_cobalt",
            [0.42, 0.14, 258],
            [0.58, 0.14, 52],
            [0.74, 0.13, 84],
            [0.48, 0.10, 62],
            [0.84, 0.17, 205],
        ),
        (
            "crimson_brass",
            [0.46, 0.16, 26],
            [0.50, 0.11, 186],
            [0.76, 0.13, 86],
            [0.46, 0.09, 58],
            [0.88, 0.18, 76],
        ),
    )
    tag, env_lch, cop_lch, brs_lch, tim_lch, aet_lch = palettes[variant % len(palettes)]

    dsl = f"""
@meta {{ archetype:entity, target_profile:bedrock_entity, texel_density:16, units_per_block:16, dsl_version:"4.1", size_step:0.5, audit_profile:sculpted }}
@layout {{
  budget max_boxes = 64
  budget max_bones = 26
  envelope max_span = [38, 32, 46]
  envelope grounded = true
}}
@palette {{
  envelope: ramp(base_lch={env_lch}, steps=5, hue_shift=[-18,20])
  copper: ramp(base_lch={cop_lch}, steps=5, hue_shift=[-14,16])
  brass: ramp(base_lch={brs_lch}, steps=4, hue_shift=[-10,12])
  timber: ramp(base_lch={tim_lch}, steps=4, hue_shift=[-12,14])
  aether: ramp(base_lch={aet_lch}, steps=4, hue_shift=[-8,10], emissive=true)
}}
@rig {{
  root bone airship_root {{
    pivot_local:[0,4,0]
    box gondola_keel {{ size:[8,5,18], material:timber, prism_group:"gk" }}
    box gondola_prow place(base_box=gondola_keel, anchor=[0,-0.1,1], align=[0,0,-0.7]) {{ size:[6,4,4.5], material:copper, rotate:[14,0,0], prism_group:"gk" }}
    box cockpit_glass place(base_box=gondola_keel, anchor=[0,0.45,0.85], align=[0,-0.35,-0.55]) {{ size:[6.5,3,3], material:aether, rotate:[24,0,0], prism_group:"gk" }}
    bone cargo_hatch attach(parent_box=gondola_keel, anchor=[0,-0.45,-1], align=[0,-1,1]) {{
      joint_limits:[[-105,10],[-2,2],[-2,2]]
      box hatch_ramp {{ size:[6,3,1.5], material:brass, blend_with:[gondola_keel] }}
    }}
    bone gasbag_hull attach(parent_box=gondola_keel, anchor=[0,1,0], align=[0,-0.95,0]) {{
      box pylon_strut {{ size:[3,4,10], material:copper, prism_group:"env", blend_with:[gondola_keel] }}
      box envelope_core place(base_box=pylon_strut, anchor=[0,1,0], align=[0,-0.8,0]) {{ size:[12,10,24], material:envelope, prism_group:"env" }}
      box rib_band_l place(base_box=envelope_core, anchor=[-0.85,0,0], align=[0.5,0,0]) {{ size:[3,7.5,21], material:envelope, rotate:[0,0,-24], prism_group:"env" }}
      box rib_band_r place(base_box=envelope_core, anchor=[0.85,0,0], align=[-0.5,0,0]) {{ size:[3,7.5,21], material:envelope, rotate:[0,0,24], prism_group:"env" }}
      box nose_cap place(base_box=envelope_core, anchor=[0,0,1], align=[0,0,-0.7]) {{ size:[8,7,5], material:copper, prism_group:"env" }}
      box tail_cone place(base_box=envelope_core, anchor=[0,0,-1], align=[0,0,0.7]) {{ size:[8,7,5.5], material:copper, prism_group:"env" }}
      plane fin_dorsal place(base_box=tail_cone, anchor=[0,1,-0.2], align=[0,-0.9,0]) {{ size:[6,8], normal:x, layer:0, edge:scallop(depth=1,side=right), material:brass }}
      bone deck_turret attach(parent_box=envelope_core, anchor=[0,1,0.15], align=[0,-0.92,0]) {{
        joint_limits:[[-2,2],[-180,180],[-2,2]]
        box compass_ring {{ size:[4,2,4], material:brass, prism_group:"ct", blend_with:[envelope_core] }}
        box beacon_orb place(base_box=compass_ring, anchor=[0,1,0], align=[0,-0.7,0]) {{ size:[2,2.5,2], material:aether, prism_group:"ct" }}
      }}
      bone fin_l attach(parent_box=tail_cone, anchor=[-1,0,-0.15], align=[0.9,0,0]) mirror_x(fin_r) {{
        joint_limits:[[-24,24],[-2,2],[-24,24]]
        box thruster_vane {{ size:[5,1.5,5.5], material:copper, blend_with:[tail_cone] }}
      }}
    }}
    bone nacelle_l attach(parent_box=gondola_keel, anchor=[-1,0.2,0.1], align=[0.95,0,0]) mirror_x(nacelle_r) {{
      box outrigger {{ size:[5,2,3], material:copper, prism_group:"nc", blend_with:[gondola_keel] }}
      box engine_pod place(base_box=outrigger, anchor=[-0.85,0,0], align=[0.35,0,0]) {{ size:[3,3,6], material:brass, prism_group:"nc" }}
      bone rotor_prop_l attach(parent_box=engine_pod, anchor=[0,0,-1], align=[0,0,1]) {{
        joint_limits:[[-2,2],[-2,2],continuous]
        box prop_hub {{ size:[2,2,1.5], material:copper, prism_group:"pr", blend_with:[engine_pod] }}
        box blade_v place(base_box=prop_hub, anchor=[0,0,0], align=[0,0,0]) {{ size:[1,7,1], material:brass, rotate:[0,0,0], prism_group:"pr" }}
        box blade_h place(base_box=prop_hub, anchor=[0,0,0], align=[0,0,0]) {{ size:[1,7,1], material:brass, rotate:[0,0,90], prism_group:"pr" }}
        box blade_d1 place(base_box=prop_hub, anchor=[0,0,0], align=[0,0,0]) {{ size:[1,6,1], material:copper, rotate:[0,0,45], prism_group:"pr" }}
        box blade_d2 place(base_box=prop_hub, anchor=[0,0,0], align=[0,0,0]) {{ size:[1,6,1], material:copper, rotate:[0,0,135], prism_group:"pr" }}
      }}
    }}
  }}
}}
@style {{
  shade: banded(steps=5, direction=down, cluster=diagonal)
  glow_bleed: 2
}}
@behavior {{
  clip cruise_propellers {{
    duration_s: 2.0, loop: true
    track rotor_prop_l: spin(axis=z, amp_deg=360, freq_hz=0.5)
    track nacelle_r__rotor_prop_l: spin(axis=z, amp_deg=360, freq_hz=0.5)
    track deck_turret: wave(axis=y, amp_deg=32, freq_hz=0.5)
    track fin_l: wave(axis=x, amp_deg=12, freq_hz=0.5)
    track fin_r: wave(axis=x, amp_deg=12, freq_hz=0.5)
    effect 0.0: {{ sound: "synccraft:airship_rotor_hum" }}
  }}
  clip cargo_drop {{
    duration_s: 2.0, loop: true
    track cargo_hatch: hinge(axis=x, amp_deg=-55, freq_hz=0.5)
  }}
}}
"""
    return {
        "asset_id": f"aether_cargo_airship_rig_{tag}",
        "base_name": "aether_cargo_airship_rig",
        "palette_tag": tag,
        "domain": "D5",
        "mood": "M3",
        "polarity": "mechanical_rigid",
        "tier": "standard",
        "archetype": "D5 x M3 Aether Cargo Airship Rig (Rigid SO(3)/SE(3), 0% Sim(3) Bulge)",
        "concept": (
            f"Aether Cargo Airship Rig ({tag} livery): rigid keel-first timber & copper gondola, sloped glowing "
            "cockpit glass, chamfered vaulted gasbag hull, dorsal `turret_yaw` compass beacon, twin outrigger "
            "continuous `rotor_propeller` nacelles, vectored rear `gimbal_thruster` fins, rear `hatch_door` ramp, "
            "and 0% squishy `Sim(3)` scale bulge on rigid metal."
        ),
        "dsl": _compact_dsl(dsl),
    }


def _build_d4_pole_a_celestial_reef_manta_guardian(variant: int = 0) -> dict[str, Any]:
    """`(D4, M3, Pole A: Gentle & Approachable)` Benevolent guardian manta with smooth rounded-chamfer
    convex silhouettes (0 hostile spikes), open calm brow (`Rx = -8 deg <= 0 deg`), warm visible almond
    eyes, peaceful `0 deg` closed lip, `3:1` luminous harmony, and 3D vaulted pectoral wings (`wing_anchor_locked`)."""
    palettes = (
        ("cerulean_pearl", [0.48, 0.14, 240], [0.86, 0.04, 92], [0.72, 0.12, 182], [0.88, 0.16, 175]),
        ("jade_moonstone", [0.50, 0.13, 176], [0.88, 0.04, 98], [0.72, 0.14, 38], [0.88, 0.17, 82]),
    )
    tag, reef_lch, pearl_lch, foam_lch, lumen_lch = palettes[variant % len(palettes)]

    dsl = f"""
@meta {{ archetype:entity, target_profile:bedrock_entity, texel_density:16, units_per_block:16, dsl_version:"4.1", size_step:0.5, audit_profile:sculpted }}
@layout {{
  budget max_boxes = 88
  budget max_bones = 46
  envelope max_span = [135, 58, 108]
  envelope grounded = true
}}
@palette {{
  reef: ramp(base_lch={reef_lch}, steps=5, hue_shift=[-18,20])
  pearl: ramp(base_lch={pearl_lch}, steps=4, hue_shift=[-10,12])
  foam: ramp(base_lch={foam_lch}, steps=5, hue_shift=[-14,16])
  lumen: ramp(base_lch={lumen_lch}, steps=4, hue_shift=[-6,8], emissive=true)
}}
@rig {{
  root bone pelvis_root {{
    pivot_local:[0,12,0]
    box mantle_core {{ size:[14,6.5,16], material:reef, prism_group:"mc" }}
    box sternum_cushion place(base_box=mantle_core, anchor=[0,-1,0.05], align=[0,0.65,0]) {{ size:[11,2.5,14], material:pearl, prism_group:"mc" }}
    box rib_chamfer_l place(base_box=mantle_core, anchor=[-0.92,0,0], align=[0.55,0,0]) {{ size:[3.5,4.5,13], material:reef, rotate:[0,0,-20], prism_group:"mc" }}
    box rib_chamfer_r place(base_box=mantle_core, anchor=[0.92,0,0], align=[-0.55,0,0]) {{ size:[3.5,4.5,13], material:reef, rotate:[0,0,20], prism_group:"mc" }}
    box dorsal_crest place(base_box=mantle_core, anchor=[0,1,-0.1], align=[0,-0.55,0]) {{ size:[4,2,10], material:foam, rotate:[-8,0,0], prism_group:"mc" }}
    bone waist attach(parent_box=mantle_core, anchor=[0,-0.05,-1], align=[0,0,0.72]) {{
      box waist_body {{ size:[9.5,5,9.5], material:reef, prism_group:"ws" }}
      box waist_belly place(base_box=waist_body, anchor=[0,-1,0], align=[0,0.65,0]) {{ size:[7,2,8], material:pearl, prism_group:"ws" }}
      plane tail_fin place(base_box=waist_body, anchor=[0,1,-0.2], align=[0,-0.9,0]) {{ size:[4.5,9], normal:x, layer:0, edge:scallop(depth=1,side=top), material:foam, blend_with:[waist_body] }}
      chain tail attach(parent_box=waist_body, anchor=[0,0,-1], align=[0,0,0.72]) segments=5 taper=0.78 bend=[4,6,8,10,8] {{
        box caudal {{ size:[5,3.5,8], material:reef }}
      }}
    }}
    bone neck_1 attach(parent_box=mantle_core, anchor=[0,0.12,1], align=[0,-0.05,-0.75]) {{
      rest_euler:[-6,0,0]
      box cerv_1 {{ size:[8,5,6.5], material:reef, prism_group:"n1" }}
      box throat_pearl place(base_box=cerv_1, anchor=[0,-1,0], align=[0,0.65,0]) {{ size:[6,2,5.5], material:pearl, prism_group:"n1" }}
      bone head attach(parent_box=cerv_1, anchor=[0,0.05,1], align=[0,0,-0.72]) {{
        rest_euler:[4,0,0]
        box cranium_core {{ size:[7,5.5,6.5], material:reef, prism_group:"hd" }}
        box nasal_arch place(base_box=cranium_core, anchor=[0,0.12,1], align=[0,0.05,-0.72]) {{ size:[4.5,3,4.5], material:reef, rotate:[4,0,0], prism_group:"hd" }}
        box maxilla_cheek_l place(base_box=cranium_core, anchor=[-0.85,-0.1,0.8], align=[0.45,0,-0.78]) {{ size:[2.5,2.5,5], material:reef, rotate:[2,-14,0], prism_group:"hd" }}
        box maxilla_cheek_r place(base_box=cranium_core, anchor=[0.85,-0.1,0.8], align=[-0.45,0,-0.78]) {{ size:[2.5,2.5,5], material:reef, rotate:[2,14,0], prism_group:"hd" }}
        box brow_l place(base_box=cranium_core, anchor=[-0.86,0.58,0.45], align=[0.35,-0.25,0]) {{ size:[2,1.5,4], material:foam, rotate:[-8,-8,8], prism_group:"hd" }}
        box brow_r place(base_box=cranium_core, anchor=[0.86,0.58,0.45], align=[-0.35,-0.25,0]) {{ size:[2,1.5,4], material:foam, rotate:[-8,8,-8], prism_group:"hd" }}
        box eye_l place(base_box=cranium_core, anchor=[-0.92,0.18,0.5], align=[0.6,0,0]) {{ size:[1,2,2.5], material:lumen, rotate:[0,-14,0], prism_group:"hd" }}
        box eye_r place(base_box=cranium_core, anchor=[0.92,0.18,0.5], align=[-0.6,0,0]) {{ size:[1,2,2.5], material:lumen, rotate:[0,14,0], prism_group:"hd" }}
        bone lobe_l attach(parent_box=cranium_core, anchor=[-1,0.35,0.65], align=[0.85,0,-0.85]) mirror_x(lobe_r) {{
          rest_euler:[4,-28,-12]
          box lobe_1 {{ size:[2,1.5,4.5], material:foam, blend_with:[cranium_core] }}
          bone lobe_tip attach(parent_box=lobe_1, anchor=[0,0,1], align=[0,0,-0.72]) {{
            rest_euler:[-6,12,0]
            box lobe_2 {{ size:[1.5,1.5,3.5], material:pearl, blend_with:[lobe_1] }}
          }}
        }}
        bone jaw attach(parent_box=cranium_core, anchor=[0,-0.78,0.2], align=[0,0.55,-0.72]) {{
          rest_euler:[0,0,0]
          box peaceful_chin {{ size:[5,1.5,5.5], material:pearl, prism_group:"jw" }}
        }}
      }}
    }}
    bone wing_l_upper attach(parent_box=mantle_core, anchor=[-1,0.35,0.2], align=[0.85,0,0]) mirror_x(wing_r_upper) {{
      rest_euler:[12,16,-22]
      box w_hum {{ size:[14,3.5,5.5], material:reef, prism_group:"wu" }}
      box w_hum_trim place(base_box=w_hum, anchor=[-0.1,0.8,0.2], align=[0,-0.5,-0.2]) {{ size:[11,1.5,3.5], material:foam, rotate:[8,0,0], prism_group:"wu" }}
      plane wing_inner_sail place(base_box=w_hum, anchor=[0,0,-1], align=[0,0,0.88]) {{ size:[15,15], normal:y, layer:0, edge:scallop(depth=2,side=bottom), material:foam }}
      bone wing_l_fore attach(parent_box=w_hum, anchor=[-1,0,0], align=[0.82,0,0]) {{
        rest_euler:[-10,-22,16]
        box w_rad {{ size:[16,2.5,4], material:reef, prism_group:"wf" }}
        box w_rad_trim place(base_box=w_rad, anchor=[0,0.8,0.2], align=[0,-0.5,-0.2]) {{ size:[13,1.5,2.5], material:lumen, rotate:[6,0,0], prism_group:"wf" }}
        bone wing_l_f4 attach(parent_box=w_rad, anchor=[-0.45,0,-1], align=[1,0,1]) {{
          rest_euler:[-10,-24,14]
          joint_limits:[[0,0],[0,0],[0,0]]
          box f4_spar {{ size:[18,1.5,1.5], material:reef, blend_with:[w_rad] }}
          plane wing_bay_4 place(base_box=f4_spar, anchor=[0,0,-1], align=[0,0,0.92]) {{ size:[28,28], normal:y, layer:2, edge:scallop(depth=3,side=bottom), material:foam, blend_with:[wing_inner_sail,w_rad,f3_spar,wing_bay_3] }}
        }}
        bone wing_l_wrist attach(parent_box=w_rad, anchor=[-1,0,0], align=[0.8,0,0]) {{
          rest_euler:[12,12,12]
          box w_palm {{ size:[3.5,2,4.5], material:reef, prism_group:"wp" }}
          bone wing_l_f1 attach(parent_box=w_palm, anchor=[-1,0.15,0.8], align=[1,0,0]) {{
            rest_euler:[14,-10,-12]
            box f1_spar {{ size:[15,1.5,2], material:reef, blend_with:[w_palm,f2_spar] }}
            plane wing_bay_1 place(base_box=f1_spar, anchor=[-0.5,0,-1], align=[0,0,0.92]) {{ size:[22,18], normal:y, layer:-1, edge:scallop(depth=2,side=bottom), material:foam, blend_with:[f2_spar,f3_spar,wing_bay_2] }}
          }}
          bone wing_l_f2 attach(parent_box=w_palm, anchor=[-1,0,0], align=[1,0,0]) {{
            rest_euler:[4,14,6]
            box f2_spar {{ size:[15,1.5,1.5], material:reef, blend_with:[w_palm,f3_spar] }}
            plane wing_bay_2 place(base_box=f2_spar, anchor=[-0.5,0,-1], align=[0,0,0.92]) {{ size:[22,18], normal:y, layer:0, edge:scallop(depth=2,side=bottom), material:foam, blend_with:[f1_spar,f3_spar,wing_bay_3] }}
          }}
          bone wing_l_f3 attach(parent_box=w_palm, anchor=[-1,-0.15,-0.8], align=[1,0,0]) {{
            rest_euler:[-14,34,18]
            box f3_spar {{ size:[20,1.5,1.5], material:reef, blend_with:[w_palm,f2_spar,wing_bay_1] }}
            plane wing_bay_3 place(base_box=f3_spar, anchor=[0,0,-1], align=[0,0,0.92]) {{ size:[18,17], normal:y, layer:1, edge:scallop(depth=2,side=bottom), material:foam, blend_with:[f2_spar,f4_spar,wing_bay_4] }}
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
  clip breathe_gentle {{
    duration_s: 2.4, loop: true
    track neck_1: wave(axis=x, amp_deg=2.5, freq_hz=0.4167)
    track head: wave(axis=x, amp_deg=2.0, freq_hz=0.4167, phase_deg=180)
  }}
  clip swim_fins_glide {{
    duration_s: 2.4, loop: true
    track wing_l_upper: wave(axis=z, amp_deg=10, freq_hz=0.4167, phase_deg=0)
    track wing_r_upper: wave(axis=z, amp_deg=10, freq_hz=0.4167, phase_deg=180)
    track wing_l_fore: wave(axis=z, amp_deg=6, freq_hz=0.4167, phase_deg=-40)
    track wing_r_upper__wing_l_fore: wave(axis=z, amp_deg=6, freq_hz=0.4167, phase_deg=140)
    track tail_1: wave(axis=y, amp_deg=8, freq_hz=0.4167)
    layer breathe_gentle: weight=0.8
    effect 0.6: {{ particle: "synccraft:reef_lumen_trail" }}
  }}
}}
"""
    return {
        "asset_id": f"celestial_reef_manta_guardian_{tag}",
        "base_name": "celestial_reef_manta_guardian",
        "palette_tag": tag,
        "domain": "D4",
        "mood": "M3",
        "polarity": "pole_a_gentle_approachable",
        "tier": "hero",
        "archetype": "D4 x M3 Pole A (Gentle & Approachable) Celestial Reef Manta Guardian",
        "concept": (
            f"Celestial Reef Manta Guardian ({tag}): Pole A Gentle & Approachable legendary creature with smooth "
            "rounded-chamfer convex silhouettes (0 hostile body spikes), open calm brow (`Rx = -8 deg <= 0 deg`), "
            "warm glowing almond eyes, peaceful `0 deg` closed lip, smooth forward cephalic lobes, `3:1` harmonious "
            "luminous palette, 3D vaulted pectoral wings with rigid `wing_l_f4` lock, and layered `swim_fins_glide` clip."
        ),
        "dsl": _compact_dsl(dsl),
    }


def _build_d4_pole_b_cinder_veil_dread_wyrm(variant: int = 0) -> dict[str, Any]:
    """`(D4, M4, Pole B: Furious & Unapproachable)` Apex quadruped dread wyrm locking ALL Pole B laws:
    plunging `Rx +20 deg / Rz -28 deg` razor scowl brow, net `+8 deg` dead-front downward predator glare,
    balanced recessed orbital eye shelf (zero extrusion), tense `14 deg` snarl gape + external 3D canines,
    swept-back scimitar horns, coiled crouch (`belly_clearance <= 0.36`), 4-digit grounded claws, `10:1`
    chiaroscuro, rigid `wing_l_f4` lock, `@muscles` + `@kinematics` + layered `@behavior` clips."""
    palettes = (
        (
            "charcoal_crimson",
            [0.20, 0.05, 288],
            [0.32, 0.06, 265],
            [0.92, 0.03, 86],
            [0.42, 0.17, 22],
            [0.20, 0.10, 352],
            [0.66, 0.24, 24],
        ),
        (
            "obsidian_ember",
            [0.19, 0.04, 255],
            [0.34, 0.05, 240],
            [0.93, 0.03, 92],
            [0.38, 0.16, 326],
            [0.20, 0.10, 350],
            [0.80, 0.22, 68],
        ),
    )
    tag, hide_lch, basalt_lch, alab_lch, wing_lch, maw_lch, eye_lch = palettes[variant % len(palettes)]

    dsl = f"""
@meta {{ archetype:entity,target_profile:bedrock_entity,texel_density:16,units_per_block:16,dsl_version:"4.1",size_step:0.5,audit_profile:sculpted }}
@palette {{
  hd:ramp(base_lch={hide_lch},steps=5,hue_shift=[-24,24])
  bs:ramp(base_lch={basalt_lch},steps=4)
  ab:ramp(base_lch={alab_lch},steps=4)
  ew:ramp(base_lch={wing_lch},steps=5,hue_shift=[-22,22])
  mw:ramp(base_lch={maw_lch},steps=4)
  em:ramp(base_lch={eye_lch},steps=4,emissive=true)
}}
def dig(sz,ry) {{ rest_euler:[0,ry,0] box d {{ size:[1.5,1.5,sz],mat:ab }} }}
def foot4(rx) {{
  rest_euler:[rx,0,0]
  box ft {{ size:[5,2.5,5.5],mat:hd }}
  use dig(4.5,0) as toe_m attach(parent_box=ft,anchor=[0,-0.2,1],align=[0,0,-0.7])
  use dig(3.5,20) as toe_i attach(parent_box=ft,anchor=[0.75,-0.2,0.85],align=[0,0,-0.7])
  use dig(3.5,-20) as toe_o attach(parent_box=ft,anchor=[-0.75,-0.2,0.85],align=[0,0,-0.7])
  use dig(2.5,58) as thumb attach(parent_box=ft,anchor=[0.95,-0.2,-0.25],align=[0,0,-0.7])
}}
@rig {{
  root bone pelvis_root {{
    pivot_local:[0,12.5,0]
    box ch {{ size:[12,9.5,14],mat:hd,prism_group:c }}
    box st place(base_box=ch,anchor=[0,-1,0.22],align=[0,0.52,0]) {{ size:[7.5,3.5,12],mat:bs,rotate:[-10,0,0],prism_group:c }}
    box rl place(base_box=ch,anchor=[-0.95,-0.1,0],align=[0.55,0,0]) {{ size:[3.5,7.5,11.5],mat:hd,rotate:[0,0,-24],prism_group:c }}
    box rr place(base_box=ch,anchor=[0.95,-0.1,0],align=[-0.55,0,0]) {{ size:[3.5,7.5,11.5],mat:hd,rotate:[0,0,24],prism_group:c }}
    box wt place(base_box=ch,anchor=[0,1,0.05],align=[0,-0.4,0]) {{ size:[2,4.5,7.5],mat:ab,rotate:[-28,0,0],prism_group:c }}
    bone waist attach(parent_box=ch,anchor=[0,0,-1],align=[0,0,0.72]) {{
      rest_euler:[6,0,0]
      box wb {{ size:[7,7,9],mat:hd }}
      bone hips attach(parent_box=wb,anchor=[0,-0.05,-1],align=[0,0,0.72]) {{
        rest_euler:[-6,0,0]
        box pv {{ size:[10,8.5,10],mat:hd }}
        chain tail attach(parent_box=pv,anchor=[0,0.1,-1],align=[0,0,0.72]) segments=5 taper=0.78 bend=[-6,-2,4,8,10] {{ box ca {{ size:[5.5,5,8.5],mat:hd }} }}
        bone leg_l_thigh attach(parent_box=pv,anchor=[-1,-0.3,0.05],align=[0.5,0.85,0]) mirror_x(leg_r_thigh) {{
          rest_euler:[-28,0,0]
          box th {{ size:[5.5,8.5,7],mat:hd }}
          bone leg_l_shin attach(parent_box=th,anchor=[0,-1,-0.1],align=[0,0.82,0]) {{
            rest_euler:[46,0,0]
            box sh {{ size:[3.5,7.5,4.5],mat:hd }}
            use foot4(-18) as rear_foot_l attach(parent_box=sh,anchor=[0,-1,0.12],align=[0,0.9,-0.15])
          }}
        }}
      }}
    }}
    bone arm_l_upper attach(parent_box=ch,anchor=[-1,-0.32,0.48],align=[0.5,0.85,0]) mirror_x(arm_r_upper) {{
      rest_euler:[26,0,-10]
      box hu {{ size:[5,7.5,6],mat:hd,prism_group:a }}
      box sl place(base_box=hu,anchor=[-0.8,0.85,0.2],align=[0.3,-0.4,0]) {{ size:[2,3.5,2],mat:ab,rotate:[-14,-18,-42],prism_group:a }}
      box es place(base_box=hu,anchor=[-0.65,-0.75,-0.75],align=[0.3,0.3,0.5]) {{ size:[1.5,2.5,2],mat:ab,rotate:[24,0,-24],prism_group:a }}
      bone arm_l_fore attach(parent_box=hu,anchor=[0,-1,0.1],align=[0,0.82,0]) {{
        rest_euler:[-44,0,10]
        box fa {{ size:[3.5,8,4.5],mat:hd }}
        use foot4(18) as fore_paw_l attach(parent_box=fa,anchor=[0,-1,0.12],align=[0,0.9,-0.15])
      }}
    }}
    bone neck_1 attach(parent_box=ch,anchor=[0,0.32,1],align=[0,-0.1,-0.72]) {{
      rest_euler:[-20,0,0]
      box c1 {{ size:[7,6.5,7.5],mat:hd,prism_group:n }}
      box t1 place(base_box=c1,anchor=[0,-1,0],align=[0,0.6,0]) {{ size:[4.5,2,6.5],mat:em,prism_group:n }}
      bone neck_2 attach(parent_box=c1,anchor=[0,0.1,1],align=[0,-0.08,-0.72]) {{
        rest_euler:[10,0,0]
        box c2 {{ size:[5.5,5.5,7],mat:hd }}
        bone head attach(parent_box=c2,anchor=[0,0.08,1],align=[0,0,-0.72]) {{
          rest_euler:[18,0,0]
          box cr {{ size:[6,6.5,6.5],mat:hd,prism_group:h }}
          box vs place(base_box=cr,anchor=[0,0.24,1],align=[0,0.1,-0.72]) {{ size:[3.5,2.5,6],mat:hd,rotate:[10,0,0],prism_group:h }}
          box bk place(base_box=vs,anchor=[0,-0.05,1],align=[0,0,-0.65]) {{ size:[2,3,2.5],mat:ab,rotate:[24,0,0],prism_group:h }}
          bone orbit_l attach(parent_box=cr,anchor=[-0.88,0.2,0.54],align=[0.55,0,0]) mirror_x(orbit_r) {{
            box eye_l {{ size:[1,1.5,2.5],mat:em,rotate:[0,-22,0],prism_group:o,blend_with:[cr] }}
            box brow_l place(base_box=eye_l,anchor=[-0.7,1,0],align=[0.2,-0.55,0]) {{ size:[2,1.5,4.5],mat:ab,rotate:[20,-14,-28],prism_group:o,blend_with:[cr] }}
            box mx_l place(base_box=eye_l,anchor=[0.1,-1,0.7],align=[0.2,0.2,-0.72]) {{ size:[2,2.5,6.5],mat:hd,rotate:[8,-17,0],prism_group:o,blend_with:[cr,vs] }}
            box tusk_l place(base_box=mx_l,anchor=[-0.55,-0.9,0.75],align=[0,0.5,0]) {{ size:[1,2,1],mat:ab,rotate:[12,-14,-10],prism_group:o }}
            plane ut_l place(base_box=mx_l,anchor=[0.4,-1,0],align=[0,0.85,0]) {{ size:[1.5,5.5],normal:x,edge:spikes,mat:ab }}
          }}
          chain horn_l attach(parent_box=cr,anchor=[-0.78,0.88,-0.55],align=[0,-0.55,0.85]) mirror_x(horn_r) segments=3 taper=0.82 bend=[46,12,-10] {{ box h {{ size:[2.5,2.5,6],mat:ab,blend_with:[cr] }} }}
          bone jaw attach(parent_box=cr,anchor=[0,-0.82,0.18],align=[0,0.55,-0.75]) {{
            rest_euler:[14,0,0] joint_limits:[[-4,48],[-2,2],[-2,2]]
            box rm {{ size:[4,2,6.5],mat:hd }}
            bone mand_l attach(parent_box=rm,anchor=[-0.95,0.18,0.45],align=[0.8,0,-0.2]) mirror_x(mand_r) {{
              box ml {{ size:[1.5,1.5,5],mat:hd,rotate:[0,-14,0],blend_with:[rm,cr] }}
              plane lt place(base_box=ml,anchor=[0.4,1,0],align=[0,-0.85,0]) {{ size:[1.5,4.5],normal:x,edge:spikes(side=top),mat:ab,blend_with:[cr] }}
            }}
            bone tongue attach(parent_box=rm,anchor=[0,1,0.18],align=[0,-1,-0.85]) {{ rest_euler:[-6,0,0] box tb {{ size:[2,1,2.5],mat:mw,blend_with:[rm] }} }}
          }}
        }}
      }}
    }}
    bone wing_l_upper attach(parent_box=ch,anchor=[-1,0.72,0.18],align=[0.85,0,0]) mirror_x(wing_r_upper) {{
      rest_euler:[16,16,-30]
      box wh {{ size:[15,3,3.5],mat:hd,prism_group:wu }}
      box wc place(base_box=wh,anchor=[-0.18,0.85,0.35],align=[0,-0.5,-0.3]) {{ size:[10,2,2.5],mat:ab,rotate:[14,0,0],prism_group:wu }}
      plane ws place(base_box=wh,anchor=[0,0,-1],align=[0,0,0.88]) {{ size:[15,15],normal:y,edge:scallop,mat:ew }}
      bone wing_l_fore attach(parent_box=wh,anchor=[-1,0,0],align=[0.82,0,0]) {{
        rest_euler:[-12,-24,18]
        box wr {{ size:[18,2.5,3],mat:hd,prism_group:wf }}
        box wb2 place(base_box=wr,anchor=[0,0.85,0.35],align=[0,-0.5,-0.3]) {{ size:[15,2,2],mat:ab,rotate:[10,0,0],prism_group:wf }}
        bone wing_l_f4 attach(parent_box=wr,anchor=[-0.45,0,-1],align=[1,0,1]) {{
          rest_euler:[-10,-24,14] joint_limits:[[0,0],[0,0],[0,0]]
          box s4 {{ size:[19,1.5,1.5],mat:hd,blend_with:[wr] }}
          plane b4 place(base_box=s4,anchor=[0,0,-1],align=[0,0,0.92]) {{ size:[29,29],normal:y,layer:2,edge:scallop,mat:ew,blend_with:[ws,wr,s3,b3] }}
        }}
        bone wing_l_wrist attach(parent_box=wr,anchor=[-1,0,0],align=[0.8,0,0]) {{
          rest_euler:[12,12,12]
          box wp {{ size:[3.5,2.5,4.5],mat:hd }}
          bone wing_l_f1 attach(parent_box=wp,anchor=[-1,0.2,0.8],align=[1,0,0]) {{
            rest_euler:[16,-12,-14]
            box s1 {{ size:[22,2,2],mat:hd,blend_with:[wp,s2] }}
            plane b1 place(base_box=s1,anchor=[-0.45,0,-1],align=[0,0,0.92]) {{ size:[24,20],normal:y,layer:-1,edge:scallop,mat:ew,blend_with:[s2,s3,b2] }}
          }}
          bone wing_l_f2 attach(parent_box=wp,anchor=[-1,0,0],align=[1,0,0]) {{
            rest_euler:[4,14,6]
            box s2 {{ size:[21,1.5,1.5],mat:hd,blend_with:[wp,s3] }}
            plane b2 place(base_box=s2,anchor=[-0.45,0,-1],align=[0,0,0.92]) {{ size:[23,19],normal:y,edge:scallop,mat:ew,blend_with:[s1,s3,b3] }}
          }}
          bone wing_l_f3 attach(parent_box=wp,anchor=[-1,-0.2,-0.8],align=[1,0,0]) {{
            rest_euler:[-14,34,18]
            box s3 {{ size:[21,1.5,1.5],mat:hd,blend_with:[wp,s2,b1] }}
            plane b3 place(base_box=s3,anchor=[0,0,-1],align=[0,0,0.92]) {{ size:[21,17],normal:y,layer:1,edge:scallop,mat:ew,blend_with:[s2,s4,b4] }}
          }}
        }}
      }}
    }}
  }}
}}
@style {{ shade:banded(steps=5,direction=down,cluster=diagonal),glow_bleed:2 }}
@muscles {{ muscle jaw_dep {{ origin_bone:head,insertion_bone:jaw,action_axis:x,moment_arm_deg:28,tendon_coupling:[tongue,x,-0.35] }} }}
@kinematics {{ look_chain gaze {{ bones:[neck_1,neck_2,head],weight_decay:0.72 }} spring tail_4 {{ stiffness:28,damping:7.5 }} }}
@behavior {{
  clip br {{ duration_s:2.0,loop:true track neck_1:wave(axis=x,amp_deg=3,freq_hz=0.5) track head:wave(axis=x,amp_deg=3,freq_hz=0.5,phase_deg=180) }}
  clip roar_snarl {{ duration_s:1.8,loop:true track neck_1:wave(axis=x,amp_deg=5,freq_hz=0.555,bias_deg=-3) track head:wave(axis=x,amp_deg=3,freq_hz=0.555,bias_deg=-2) track jaw:wave(axis=x,amp_deg=14,freq_hz=0.555,bias_deg=14) track wing_l_upper:wave(axis=z,amp_deg=7,freq_hz=0.555,bias_deg=-5) track wing_r_upper:wave(axis=z,amp_deg=7,freq_hz=0.555,phase_deg=180,bias_deg=5) track tail_1:wave(axis=y,amp_deg=8,freq_hz=0.555) layer br:weight=0.6 effect 0.45:{{ sound:"synccraft:wyrm_roar",instruction:"flare" }} }}
}}
"""
    return {
        "asset_id": f"cinder_veil_dread_wyrm_{tag}",
        "base_name": "cinder_veil_dread_wyrm",
        "palette_tag": tag,
        "domain": "D4",
        "mood": "M4",
        "polarity": "pole_b_furious_unapproachable",
        "tier": "hero",
        "archetype": "D4 x M4 Pole B (Furious & Unapproachable) Cinder-Veil Dread Wyrm",
        "concept": (
            f"Cinder-Veil Dread Wyrm ({tag}): Pole B Furious & Unapproachable apex quadruped dragon with "
            "plunging `\\ /` razor scowl brow (`Rx +20 deg`, `Rz -28/+28 deg`), net `+8 deg` dead-front glare, "
            "zero-extrusion orbital eye shelf, `14 deg` snarl gape + external canines, swept horns, shoulder/elbow "
            "spikes, 4-digit grounded claws, rigid `wing_l_f4` lock, `10:1` chiaroscuro, and layered `roar_snarl`."
        ),
        "dsl": _compact_dsl_v28(dsl),
    }


@functools.lru_cache(maxsize=1)
def _build_all_v28_verified_masterworks_cached() -> tuple[dict[str, Any], ...]:
    ensure_synccraft_importable()
    from synccraft.dsl.parser import parse_program
    from synccraft.kinematics.se3_chain import compile_geometry

    builders = (
        _build_d5_basalt_survey_armored_crawler,
        _build_d5_aether_cargo_airship_rig,
        _build_d4_pole_a_celestial_reef_manta_guardian,
        _build_d4_pole_b_cinder_veil_dread_wyrm,
    )
    verified: list[dict[str, Any]] = []
    for bfn in builders:
        for var in range(2):
            item = bfn(var)
            dsl = _auto_ground_dsl_four_feet(item["dsl"])
            # Explicit real SyncCraft parse + geometry compile verification
            prog = parse_program(dsl)
            scene = compile_geometry(prog)
            rep = verify_v25_dsl_asset(dsl, tier=item["tier"], rest_tol=0.45)
            if not rep["passed_all"]:
                raise ValueError(
                    f"v28 masterwork '{item['asset_id']}' failed verification: "
                    f"audit={rep['audit_passed']} errs={rep['audit_errors']} "
                    f"g_fails={rep['g1_g12']['fails']} g_warns={rep['g1_g12']['warns']} "
                    f"ground={rep['grounding']['rest_ok']} clips={rep['clip_audits']}"
                )
            item["dsl"] = dsl
            item["program"] = prog
            item["scene"] = scene
            item["verification"] = rep
            verified.append(item)
    return tuple(verified)


def build_all_v28_verified_masterworks() -> list[dict[str, Any]]:
    """Build and strictly verify all 8 (`4` archetypes x `2` palettes) original `CC0-1.0` `v28` masterworks
    through SyncCraft's real `parse_program` + `compile_geometry` + `audit_scene` + `G1-G12` + clips."""
    return [dict(m) for m in _build_all_v28_verified_masterworks_cached()]


# ---------------------------------------------------------------------------
# 2. Full-Color Shaded Rendering & 4-Primitive Annotation Helpers
# ---------------------------------------------------------------------------

def _render_rgb_with_primitives(
    program: Any,
    baked: dict[str, Any],
    *,
    az_deg: float,
    el_deg: float,
    panel_size: tuple[int, int],
    studio_preset: str | int = 0,
    fill_frac: float = 0.78,
    pan_uv: tuple[float, float] = (0.0, 0.0),
    scene: Any | None = None,
    camera_override: dict[str, Any] | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Render a full-color shaded RGB frame (`clay=False`, never B&W cutout) + exact 2D projected bone metadata."""
    ren = render_shaded_3d_with_spatial_annotations(
        program,
        baked,
        az_deg=az_deg,
        el_deg=el_deg,
        fill_frac=fill_frac,
        pan_uv=pan_uv,
        panel_size=panel_size,
        studio_preset=studio_preset,
        scene=scene,
        camera_override=camera_override,
    )
    img = Image.fromarray(ren["rgb_array"], mode="RGB")
    return img, ren


def _pick_part(parts: dict[str, dict[str, Any]], candidates: tuple[str, ...]) -> dict[str, Any]:
    for c in candidates:
        if c in parts:
            return parts[c]
    for c in candidates:
        for k, v in parts.items():
            if c in k:
                return v
    # Deterministic fallback to nearest part
    return sorted(parts.values(), key=lambda x: x["cz_mean"])[0]


def _safe_dash(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str,
    *,
    fallback_dx: float = 48.0,
    fallback_dy: float = -24.0,
) -> str:
    """Emit a guaranteed non-degenerate `<dash .../>` (`length >= 6.0` in `[0..1000]`)."""
    ix1, iy1 = _clamp_1000(x1), _clamp_1000(y1)
    ix2, iy2 = _clamp_1000(x2), _clamp_1000(y2)
    if math.hypot(ix2 - ix1, iy2 - iy1) < 6.0:
        ix2 = _clamp_1000(ix1 + fallback_dx)
        iy2 = _clamp_1000(iy1 + fallback_dy)
        if math.hypot(ix2 - ix1, iy2 - iy1) < 6.0:
            ix2 = _clamp_1000(ix1 - fallback_dx)
            iy2 = _clamp_1000(iy1 - fallback_dy)
    return fmt_dash(ix1, iy1, ix2, iy2, label)


def _format_four_primitives_for_render(
    ren: dict[str, Any],
    *,
    primary_candidates: tuple[str, ...],
    secondary_candidates: tuple[str, ...],
    pivot_candidates: tuple[str, ...],
    dash_label: str,
) -> dict[str, str]:
    """Build exact validated `<point z=../>`, `<box .../>`, `<dash .../>`, and `<circle .../>` strings from `ren`."""
    parts = ren["spatial_parts"]
    p1 = _pick_part(parts, primary_candidates)
    p2 = _pick_part(parts, secondary_candidates)
    if p2["name"] == p1["name"] or _clamped_dist(p1["cx"], p1["cy"], p2["cx"], p2["cy"]) < 10.0:
        # Pick farthest distinct part for a clean non-degenerate dash
        others = [v for v in parts.values() if v["name"] != p1["name"]]
        if others:
            p2 = max(others, key=lambda v: _clamped_dist(p1["cx"], p1["cy"], v["cx"], v["cy"]))
    piv = _pick_part(parts, pivot_candidates)
    px, py = piv["pivot_xy"]
    pr = max(14.0, min(140.0, float(piv["radius"])))

    return {
        "p1_name": str(p1["name"]),
        "p2_name": str(p2["name"]),
        "piv_name": str(piv["name"]),
        "point1": fmt_point(p1["cx"], p1["cy"], p1["name"], z=p1.get("depth_tier", "near")),
        "point2": fmt_point(p2["cx"], p2["cy"], p2["name"], z=p2.get("depth_tier", "far")),
        "box1": fmt_box(*p1["box"], p1["name"]),
        "box2": fmt_box(*p2["box"], p2["name"]),
        "dash": _safe_dash(p1["cx"], p1["cy"], p2["cx"], p2["cy"], dash_label),
        "circle": fmt_circle(px, py, pr, f"pivot:{piv['name']}"),
    }


def _render_two_frame_articulated(
    program: Any,
    rest_scene: Any,
    baked: dict[str, Any],
    *,
    pose_deltas: dict[str, list[float]],
    az_deg: float,
    el_deg: float,
    panel_size: tuple[int, int],
    studio_preset: int,
    effector_bone: str,
    pivot_bone: str,
    arc_label: str,
) -> dict[str, Any]:
    """Render two full-color shaded frames (Rest vs SE(3) Articulated) sharing one locked camera + exact motion arc."""
    ensure_synccraft_importable()
    from synccraft.kinematics.se3_chain import compile_posed_geometry

    img0, ren0 = _render_rgb_with_primitives(
        program,
        baked,
        az_deg=az_deg,
        el_deg=el_deg,
        panel_size=panel_size,
        studio_preset=studio_preset,
        scene=rest_scene,
    )
    posed_sc = compile_posed_geometry(program, pose_deltas)
    baked_posed = _repose_baked_for_scene(baked, posed_sc)
    img1, ren1 = _render_rgb_with_primitives(
        program,
        baked_posed,
        az_deg=az_deg,
        el_deg=el_deg,
        panel_size=panel_size,
        studio_preset=studio_preset,
        scene=posed_sc,
        camera_override=ren0["camera"],
    )
    p0 = _pick_part(ren0["bones"], (effector_bone,))
    p1 = _pick_part(ren1["bones"], (effector_bone,))
    piv = _pick_part(ren0["bones"], (pivot_bone,))
    arc_dash = _safe_dash(p0["cx"], p0["cy"], p1["cx"], p1["cy"], f"anim_arc:{arc_label}")
    piv_circle = fmt_circle(piv["pivot_xy"][0], piv["pivot_xy"][1], max(16.0, min(120.0, piv["radius"])), f"pivot:{piv['name']}")
    return {
        "img_rest": img0,
        "img_posed": img1,
        "ren_rest": ren0,
        "ren_posed": ren1,
        "pt_rest": fmt_point(p0["cx"], p0["cy"], f"{p0['name']}_rest", z=p0.get("depth_tier", "near")),
        "pt_posed": fmt_point(p1["cx"], p1["cy"], f"{p1['name']}_posed", z=p1.get("depth_tier", "near")),
        "box_posed": fmt_box(*p1["box"], p1["name"]),
        "arc_dash": arc_dash,
        "piv_circle": piv_circle,
        "disp_px": round(_clamped_dist(p0["cx"], p0["cy"], p1["cx"], p1["cy"]), 1),
    }


# ---------------------------------------------------------------------------
# 3. SyncCraft `ba969f3` Live-Verified Helpers (`gait` + `layer` + `effect`,
#    `preserve_and_layer`, `invert_clip_to_gait`, `train`-only 5-Task Proposer)
# ---------------------------------------------------------------------------

def _build_ba969f3_layered_gait_quadruped_dsl() -> str:
    """Clean original `CC0-1.0` quadruped rig exercising SyncCraft `ba969f3` `gait` + `layer` + `effect touchdown`."""
    return _compact_dsl(
        """
@meta { archetype:entity, target_profile:bedrock_entity, texel_density:16, units_per_block:16, dsl_version:"4.1", size_step:0.5, audit_profile:sculpted }
@palette {
  hide: ramp(base_lch=[0.46,0.12,42], steps=5, hue_shift=[-16,18])
  belly: ramp(base_lch=[0.82,0.05,86], steps=4, hue_shift=[-8,10])
  claw: ramp(base_lch=[0.28,0.04,250], steps=4, hue_shift=[-10,10])
}
@rig {
  root bone torso {
    pivot_local:[0,9.5,0]
    box chest { size:[5,4.5,7], material:hide, prism_group:"ch" }
    box sternum place(base_box=chest, anchor=[0,-1,0], align=[0,0.6,0]) { size:[3.5,1.5,6], material:belly, prism_group:"ch" }
    bone lumbar attach(parent_box=chest, anchor=[0,0,-1], align=[0,0,1], seat=0.25) {
      joint_limits:[[-18,18],[-22,22],[-12,12]]
      box hips { size:[4.5,4,5.5], material:hide }
      bone hind_leg_l attach(parent_box=hips, anchor=[-1,-0.65,-0.45], align=[1,1,0]) mirror_x(hind_leg_r) {
        joint_limits:[[-60,60],[-25,25],[-25,25]]
        box femur { size:[1.5,4,2], material:hide }
        bone shin attach(parent_box=femur, anchor=[0,-1,0], align=[0,1,0]) {
          rest_euler:[10,0,0]
          joint_limits:[[-55,55],[-20,20],[-20,20]]
          box tibia { size:[1.5,4,1.5], material:claw }
        }
      }
      bone tail_1 attach(parent_box=hips, anchor=[0,0.2,-1], align=[0,0,1], seat=0.25) {
        joint_limits:[[-20,20],[-28,28],[-15,15]]
        box caudal_1 { size:[1.5,1.5,5], material:hide }
      }
    }
    bone fore_leg_l attach(parent_box=chest, anchor=[-1,-0.65,0.55], align=[1,1,0]) mirror_x(fore_leg_r) {
      joint_limits:[[-60,60],[-25,25],[-25,25]]
      box humerus { size:[1.5,4,2], material:hide }
      bone forearm attach(parent_box=humerus, anchor=[0,-1,0], align=[0,1,0]) {
        rest_euler:[-10,0,0]
        joint_limits:[[-55,55],[-20,20],[-20,20]]
        box radius { size:[1.5,4,1.5], material:claw }
      }
    }
    bone neck attach(parent_box=chest, anchor=[0,0.35,1], align=[0,0,-1], seat=0.25) {
      rest_euler:[-14,0,0]
      joint_limits:[[-25,25],[-25,25],[-15,15]]
      box neck_seg { size:[2.5,2.5,4], material:hide }
      bone skull attach(parent_box=neck_seg, anchor=[0,0,1], align=[0,0,-1], seat=0.2) {
        rest_euler:[12,0,0]
        joint_limits:[[-25,25],[-25,25],[-15,15]]
        box snout { size:[3,2.5,4], material:hide }
      }
    }
  }
}
@style { shade: banded(steps=4, direction=down, cluster=diagonal), glow_bleed: 1 }
@behavior {
  clip breathe {
    duration_s: 2.0, loop: true
    track lumbar: wave(axis=x, amp_deg=2, freq_hz=0.5)
    track neck: wave(axis=x, amp_deg=2.5, freq_hz=0.5, phase_deg=180)
  }
  clip prowl_layered {
    duration_s: 1.6, loop: true
    gait prowl {
      pattern: lateral_sequence,
      froude: 0.16,
      duty: 0.68,
      lift_ratio: 0.10,
      heave_ratio: 0.025,
      pitch_deg: 1.2,
      spine_wave_deg: 2.5,
      tail_lag: 0.2,
      export_root_motion: false
    }
    layer breathe: weight=0.75
    effect touchdown: { sound: "synccraft:hound_paw_step", particle: "synccraft:basalt_dust" }
    effect 0.4: { instruction: "ear_twitch_alert" }
  }
}
"""
    )


def _run_live_ba969f3_verifications(wyrm_item: dict[str, Any], crawler_item: dict[str, Any]) -> dict[str, Any]:
    """Execute SyncCraft `ba969f3` live modules (`layer`+`touchdown`, `preserve_and_layer`, `gait_inversion`,
    and strictly `train`-only 5-task proposer records) so every emitted metric is solver-verified."""
    ensure_synccraft_importable()
    from synccraft.dsl.ast_nodes import GaitDecl
    from synccraft.dsl.parser import parse_program
    from synccraft.kinematics.gait_inversion import (
        invert_clip_to_gait,
        retarget_clip_across_models,
        search_or_repair_gait,
    )
    from synccraft.kinematics.morphology import analyze_morphology
    from synccraft.kinematics.musculoskeletal import audit_behavior_clip
    from synccraft.kinematics.se3_chain import compile_geometry
    from synccraft.shading.compiler import compile_vanilla_atlas
    from synccraft.shading.preserve_and_layer import (
        decode_rgba_png,
        preserve_and_layer_painted_texture,
        rebake_selected_faces,
    )
    from synccraft.surface.contracts import MaterialProgram, StyleConfig, WorldState
    from synccraft.surface.domain import build_surface_domain, initialize_surface_state
    from synccraft.world.proposer_tasks import build_proposer_supervision_dataset

    # 1. Additive clip layer + effect touchdown live audit
    quad_dsl = _build_ba969f3_layered_gait_quadruped_dsl()
    quad_prog = parse_program(quad_dsl)
    quad_scene = compile_geometry(quad_prog)
    quad_morph = analyze_morphology(quad_prog, quad_scene)
    layered_audit = audit_behavior_clip(quad_prog, "prowl_layered")

    # 2. `preserve_and_layer_painted_texture` + `rebake_selected_faces` live on crawler
    cr_prog = crawler_item["program"]
    cr_scene = crawler_item["scene"]
    mat_prog = MaterialProgram.from_program(cr_prog)
    style_cfg = StyleConfig.from_program(cr_prog)
    domain = build_surface_domain(cr_prog, mat_prog)
    ws = WorldState(
        program=cr_prog,
        material_program=mat_prog,
        surface_state=initialize_surface_state(domain),
        style_config=style_cfg,
    )
    base_atlas, base_png = compile_vanilla_atlas(ws)
    base_rgba = decode_rgba_png(base_png)
    # Simulate an artist leaving one face (`roof_rack.up`) unpainted (`A == 0`) while keeping all hand-painted faces
    faces_map = base_atlas.get("faces") or {}
    blank_box = "roof_rack" if "roof_rack" in faces_map else next(iter(faces_map.keys()))
    blank_face = "up" if "up" in faces_map[blank_box] else next(iter(faces_map[blank_box].keys()))
    uv_entry = faces_map[blank_box][blank_face]
    raw_uv = uv_entry.get("uv") if isinstance(uv_entry, dict) else uv_entry
    h_px, w_px = base_rgba.shape[:2]
    x0 = max(0, min(w_px, int(math.floor(min(float(raw_uv[0]), float(raw_uv[2]))))))
    x1 = max(0, min(w_px, int(math.ceil(max(float(raw_uv[0]), float(raw_uv[2]))))))
    y0 = max(0, min(h_px, int(math.floor(min(float(raw_uv[1]), float(raw_uv[3]))))))
    y1 = max(0, min(h_px, int(math.ceil(max(float(raw_uv[1]), float(raw_uv[3]))))))
    base_rgba[y0:y1, x0:x1, :] = 0

    pal_res = preserve_and_layer_painted_texture(
        cr_prog,
        cr_scene,
        base_atlas,
        base_rgba,
        world_state=ws,
        wear_strength=0.25,
        glow_strength=0.65,
        glow_bleed=2,
        fill_blank_faces=True,
        variant_hue_shift_deg=35.0,
        seed=2026,
    )
    rebake_res = rebake_selected_faces(ws, base_atlas, base_rgba, {(blank_box, blank_face)})

    # 3. `invert_clip_to_gait` + `search_or_repair_gait` + `retarget_clip_across_models`
    inv_res = invert_clip_to_gait(quad_prog, "prowl_layered", num_samples=32)
    broken_gait = GaitDecl(
        name="overdriven_prowl",
        pattern="diagonal_pairs",
        froude=2.6,
        duty=0.36,
        lift_ratio=0.34,
        heave_ratio=0.12,
        pitch_deg=10.0,
        spine_wave_deg=16.0,
        snap=0.88,
        export_root_motion=False,
    )
    repair_res = search_or_repair_gait(
        quad_prog,
        broken_gait,
        duration_s=1.4,
        clip_name="repaired_trot",
        max_evals=8,
    )

    # 4. Strictly `split == "train"` ONLY 5-task proposer dataset (NEVER `holdout_gate_d`!)
    all_prop = build_proposer_supervision_dataset()
    train_prop = [
        r
        for r in all_prop
        if r.split == "train"
        and r.asset_id not in _GATE_D_HOLDOUT_ASSET_IDS
        and r.body_plan_group not in _GATE_D_HOLDOUT_BODY_PLANS
        and r.source_author not in _GATE_D_HOLDOUT_AUTHORS
    ]
    # Test cross-creature retargeting strictly between train-split fixtures (never holdout!)
    bison_rec = next((r for r in train_prop if r.asset_id == "ironhide_bison_v1"), None)
    retarget_res = None
    if bison_rec is not None:
        bison_prog = parse_program(bison_rec.dsl_source)
        retarget_res = retarget_clip_across_models(
            quad_prog,
            "prowl_layered",
            bison_prog,
            new_clip_name="bison_prowl_retargeted",
            repair_if_needed=True,
            max_repair_evals=6,
        )

    return {
        "quad_dsl": quad_dsl,
        "quad_prog": quad_prog,
        "quad_scene": quad_scene,
        "quad_morph": quad_morph,
        "layered_audit": layered_audit,
        "blank_target": (blank_box, blank_face),
        "preserve_layer": pal_res,
        "rebake": rebake_res,
        "invert_gait": inv_res,
        "repair_gait": repair_res,
        "retarget_gait": retarget_res,
        "train_proposer_records": train_prop,
    }


# ---------------------------------------------------------------------------
# 4. Adversarial `v28` Mood Drift & Solver Trap Variants (Real Compiled DSLs!)
# ---------------------------------------------------------------------------

def _build_v28_defective_dsl_variants(
    wyrm_dsl: str,
    manta_dsl: str,
) -> dict[str, dict[str, Any]]:
    """Build real compilable defective DSL variants demonstrating `v28` traps in full-color studio renders."""
    # Trap 1: Negative Rx < 0 sad-puppy brow + chin-up (-12 deg) neck + polite 0 deg closed jaw on Pole-B boss!
    sad_puppy_wyrm = (
        wyrm_dsl.replace("rotate:[20,-14,-28]", "rotate:[-16,0,8]")
        .replace("rotate:[20,14,28]", "rotate:[-16,0,-8]")
        .replace("rest_euler:[18,0,0]", "rest_euler:[-4,0,0]")
        .replace("rest_euler:[14,0,0]", "rest_euler:[0,0,0]")
    )
    # Trap 6: Pole-A gentle guardian corrupted by hostile Rx +22 deg scowl brow!
    angry_manta = (
        manta_dsl.replace("rotate:[-8,-8,8]", "rotate:[22,-14,-28]")
        .replace("rotate:[-8,8,-8]", "rotate:[22,14,28]")
        .replace("rest_euler:[0,0,0]", "rest_euler:[16,0,0]")
    )
    if sad_puppy_wyrm == wyrm_dsl or angry_manta == manta_dsl:
        raise ValueError("Defective v28 DSL variant substitution failed to mutate source DSL")
    return {
        "sad_puppy_wyrm": {
            "dsl": sad_puppy_wyrm,
            "trap_id": "v28_trap_01_negative_rx_sad_puppy_brow",
        },
        "angry_manta": {
            "dsl": angry_manta,
            "trap_id": "v28_trap_06_pole_a_hostile_scowl_corruption",
        },
    }


# ---------------------------------------------------------------------------
# 5. Stage-9 `v28` Corpus Builder (`96` balanced, verified EN+ZH records)
# ---------------------------------------------------------------------------

def _sanitize_check_clean(text: str) -> None:
    low = text.lower()
    for banned in BANNED_BENCHMARK_SUBSTRINGS:
        if banned in low:
            raise ValueError(f"Banned benchmark substring {banned!r} found in v28 record")
    for ho in _GATE_D_HOLDOUT_ASSET_IDS | _GATE_D_HOLDOUT_BODY_PLANS | _GATE_D_HOLDOUT_AUTHORS:
        if ho in low:
            raise ValueError(f"Holdout Gate D identifier {ho!r} leaked into v28 record")


def build_aesthetic_v28_biomech_records(
    *,
    seed: int = 20261128,
    max_records: int = 96,
    panel_size: tuple[int, int] = (224, 224),
) -> list[dict[str, Any]]:
    """Build verified Stage-9 `Blockbench_sculpted_style-v28` & SyncCraft `ba969f3` multimodal + DSL records.

    Returns records matching the exact schema of `build_aesthetic_v25_vigorous_records`:
    `id`, `split`, `language`, `task`, `prompt`, `answer`, `images`, `enable_thinking`,
    `source_family`, `modality`, `provenance`, and `metadata` (with `pixel_indispensable`,
    `provenance`, and `style_version="blockbench_sculpted_style_v28"`).
    """
    ensure_synccraft_importable()
    from synccraft.dsl.parser import parse_program
    from synccraft.kinematics.se3_chain import compile_geometry

    rng = random.Random(seed)
    masterworks = build_all_v28_verified_masterworks()
    baked_map: dict[str, dict[str, Any]] = {
        mw["asset_id"]: bake_sculpted_rgba_atlas(mw["program"], scene=mw["scene"], texels_per_unit=2)
        for mw in masterworks
    }

    crawlers = [m for m in masterworks if m["base_name"] == "basalt_survey_armored_crawler"]
    airships = [m for m in masterworks if m["base_name"] == "aether_cargo_airship_rig"]
    mantas = [m for m in masterworks if m["base_name"] == "celestial_reef_manta_guardian"]
    wyrms = [m for m in masterworks if m["base_name"] == "cinder_veil_dread_wyrm"]

    live_ba969f3 = _run_live_ba969f3_verifications(wyrms[0], crawlers[0])
    defects = _build_v28_defective_dsl_variants(wyrms[0]["dsl"], mantas[0]["dsl"])

    by_task: dict[str, list[dict[str, Any]]] = {t: [] for t in AESTHETIC_V28_TASKS}
    # When max_records < 96 (e.g., fast CI/test slices), cap per-task generation early so we skip unused renders
    task_cap = (
        max(1, math.ceil(int(max_records) / len(AESTHETIC_V28_TASKS)) + 1)
        if int(max_records) < 96
        else 999
    )

    def _need(task: str) -> bool:
        return len(by_task[task]) < task_cap

    def _add(
        task: str,
        sample_id: str,
        prompt: str,
        think: str,
        body: str,
        *,
        language: str = "en",
        images: list[Image.Image] | None = None,
        pixel_indispensable: bool = False,
        meta_extra: dict[str, Any] | None = None,
    ) -> None:
        if task not in by_task:
            raise ValueError(f"Unknown v28 task {task!r}")
        answer = format_supervised_answer(body, think, enable_thinking=True)
        _sanitize_check_clean(prompt)
        _sanitize_check_clean(answer)
        total_chars = len(prompt) + len(answer)
        if total_chars > 9450:
            raise ValueError(f"v28 record '{sample_id}' exceeds strict 9500-char Stage-9 cap ({total_chars} chars)")
        health = check_generation_collapse_health(answer, expect_thinking=True, max_rep_4gram=0.14)
        if not health["healthy"]:
            raise ValueError(f"v28 record '{sample_id}' failed collapse health: {health}")
        imgs = list(images or [])
        for im in imgs:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32)
            chroma = float(np.mean(np.abs(arr[:, :, 0] - arr[:, :, 1]) + np.abs(arr[:, :, 1] - arr[:, :, 2])))
            if chroma < 4.0:
                raise ValueError(f"v28 record '{sample_id}' rendered low-chroma/B&W image ({chroma:.2f} < 4.0)")
        prov = make_clean_room_provenance("aesthetic_v28_biomech_cc0")
        rec = {
            "id": identity(f"aesthetic_v28:{sample_id}"),
            "split": "train",
            "language": language,
            "task": task,
            "prompt": prompt,
            "answer": answer,
            "images": imgs,
            "enable_thinking": True,
            "source_family": "synccraft_aesthetic_v28_cc0",
            "modality": "image" if imgs else "text",
            "provenance": prov,
            "metadata": {
                "license": "CC0-1.0",
                "style_version": "blockbench_sculpted_style_v28",
                "pixel_indispensable": bool(pixel_indispensable),
                "provenance": prov,
                **(meta_extra or {}),
            },
        }
        by_task[task].append(rec)

    # -----------------------------------------------------------------------
    # Task 1: `v28_temperament_polarity_and_router` (18 records, 100% visual)
    # -----------------------------------------------------------------------
    # 1A. All 8 masterworks: Turn-1 Domain (D1-D5) x Mood (M1-M4) x Polarity routing + 4-primitive audit (2 zh, 6 en)
    router_cams = (
        (148.0, 16.0),
        (212.0, 18.0),
        (152.0, 14.0),
        (208.0, 20.0),
        (176.0, 12.0),
        (150.0, 16.0),
        (178.0, 10.0),
        (154.0, 15.0),
    )
    for idx, mw in enumerate(masterworks):
        if not _need("v28_temperament_polarity_and_router"):
            break
        az_b, el_b = router_cams[idx % len(router_cams)]
        img, ren = _render_rgb_with_primitives(
            mw["program"],
            baked_map[mw["asset_id"]],
            az_deg=az_b + rng.uniform(-5.0, 5.0),
            el_deg=el_b + rng.uniform(-3.0, 3.0),
            panel_size=panel_size,
            studio_preset=idx % len(STUDIO_LIGHTING_PRESETS),
            scene=mw["scene"],
        )
        is_d5 = mw["domain"] == "D5"
        prim = _format_four_primitives_for_render(
            ren,
            primary_candidates=("head", "cab_module", "airship_root", "pelvis_root"),
            secondary_candidates=("tail_3", "wing_l_wrist", "cargo_module", "gasbag_hull", "nacelle_l"),
            pivot_candidates=("head", "turret_yaw", "deck_turret", "wing_l_upper", "wheel_axle_fl"),
            dash_label=f"axis:{mw['base_name']}",
        )
        g = mw["verification"]["g1_g12"]
        lang = "zh" if idx in (1, 6) else "en"
        if mw["polarity"] == "pole_a_gentle_approachable":
            pol_en = (
                "Pole A (Gentle & Approachable): smooth convex chamfers with 0 hostile spikes, calm open brow "
                "(`Rx = -8 deg <= 0 deg`), warm visible almond eyes on a balanced orbital shelf, peaceful `0 deg` "
                "closed lip, and harmonious `3:1` luminous value balance."
            )
            pol_zh = (
                "A极（温和可亲）：圆润凸面倒角、零敌意尖刺、开放平静眉弓（`Rx = -8° <= 0°`）、"
                "平衡眼眶台上的温暖杏仁眼、安详 `0°` 闭合唇线与柔和 `3:1` 明暗过渡。"
            )
        elif mw["polarity"] == "pole_b_furious_unapproachable":
            pol_en = (
                "Pole B (Furious & Unapproachable): plunging `\\ /` razor scowl brow (`Rx = +20 deg`, `Rz = -28/+28 deg`), "
                "cumulative `+8 deg` dead-front downward predator glare, balanced zero-extrusion recessed eye shelf, "
                "tense `14 deg` resting snarl gape + external 3D canines, outward shoulder/elbow hazard spikes, "
                "coiled crouch (`belly_clearance <= 0.36`), and `10:1` high-contrast chiaroscuro."
            )
            pol_zh = (
                "B极（暴怒不可接近）：下压 `\\ /` 锐利怒眉（`Rx = +20°`、`Rz = ∓28°`）、正视累计 `+8°` 下压捕食凝视、"
                "零凸出平衡凹陷眼眶台、静息 `14°` 嘶吼张口与外部3D獠牙、肩肘外展尖刺、低伏蜷缩姿态以及 `10:1` 强烈明暗对比。"
            )
        else:
            pol_en = (
                "Domain D5 Rigid Mechanical Assembly: chassis/keel-first framing, octagonal rotary hubs (`wheel_axle` "
                "or `rotor_propeller`), sloped glass cab, rigid `SO(3)/SE(3)` hinges (`turret_yaw`, `hatch_door`), "
                "and strictly `0%` squishy `Sim(3)` scale bulge on metal."
            )
            pol_zh = (
                "D5 刚性载具机械装配：底盘/龙骨优先构架、四带正八边形轮毂/旋翼轴、倾斜玻璃驾驶舱、"
                "刚性 `SO(3)/SE(3)` 铰链（炮塔回转、尾门舱盖）以及金属部件严格 `0%` `Sim(3)` 软体鼓包。"
            )

        if lang == "zh":
            prompt = (
                "检查这张全彩着色 `16 px/block` 渲染图（`clay=False`）。执行 `Blockbench_sculpted_style-v28` "
                "首轮意图路由（域 `D1-D5` × 情绪 `M1-M4` × 气质极性），用 <point>、<box>、<dash>、<circle> "
                "原语标定关键部位与关节枢轴，并审计眉弓欧拉角符号、凝视角与明暗对比度。"
            )
            think = (
                f"全彩着色渲染核验（禁止灰度白模误判）：主焦点 {prim['point1']} {prim['box1']}，"
                f"远端参考 {prim['point2']} {prim['box2']}，结构主轴 {prim['dash']}，关键枢轴 {prim['circle']}。"
                f"判定目标=({mw['domain']}, {mw['mood']}, {mw['polarity']})，元素={g['total_elements']}，骨骼={g['bones']}，严格门禁通过。"
            )
            body = (
                f"### v28 首轮路由结论：`({mw['domain']}, {mw['mood']})` — `{mw['polarity']}`\n"
                f"- **空间原语定位：** 主区 {prim['point1']} {prim['box1']} | 轴向 {prim['dash']} | 枢轴 {prim['circle']}\n"
                f"- **极性与造型法则：** {pol_zh}\n"
                f"- **确定性指标：** `elements={g['total_elements']}`, `bones={g['bones']}`, "
                f"`mirror_ok={g['mirror_ok']}`, `G1-G12 strict={g['passed_strict']}` -> **PASS**。"
            )
        else:
            prompt = (
                "Inspect this full-color textured `16 px/block` render (`clay=False`). Execute the "
                "`Blockbench_sculpted_style-v28` Turn-1 Intent Router (`Domain D1-D5 x Mood M1-M4 x Polarity`), "
                "ground key structures with <point>, <box>, <dash>, and <circle>, and audit brow Euler sign, "
                "dead-front gaze pitch, eye shelf recess, jaw gape, and chiaroscuro contrast."
            )
            think = (
                f"Inspecting full-textured studio render (clay=False): primary mass {prim['point1']} {prim['box1']}, "
                f"secondary anchor {prim['point2']} {prim['box2']}, structural axis {prim['dash']}, pivot {prim['circle']}. "
                f"Classify ({mw['domain']}, {mw['mood']}, {mw['polarity']}); elements={g['total_elements']}, bones={g['bones']}, "
                f"mirror_ok={g['mirror_ok']}, strict_gates={g['passed_strict']}."
            )
            body = (
                f"### Turn-1 v28 Router: `({mw['domain']}, {mw['mood']})` — `{mw['polarity']}`\n"
                f"- **4-Primitive Grounding:** {prim['point1']} {prim['box1']} | {prim['dash']} | {prim['circle']}\n"
                f"- **Polarity & Grammar Audit:** {pol_en}\n"
                f"- **Verified Metrics:** `elements={g['total_elements']}`, `bones={g['bones']}`, "
                f"`top10_nw={g['top10_nw_volume']:.1%}`, `mirror_ok={g['mirror_ok']}`, `G1-G12={g['passed_strict']}`."
            )
        _add(
            "v28_temperament_polarity_and_router",
            f"router_{mw['asset_id']}",
            prompt,
            think,
            body,
            language=lang,
            images=[img],
            pixel_indispensable=True,
            meta_extra={
                "domain": mw["domain"],
                "mood": mw["mood"],
                "polarity": mw["polarity"],
                "asset_id": mw["asset_id"],
            },
        )

    # 1B. 6 Direct Multi-Image Pole A vs Pole B Comparisons (dead-front 180, front-3/4 150, side 94 x 2 variants; 2 zh, 4 en)
    comp_angles = (
        ("dead_front", 180.0, 10.0),
        ("front_34", 150.0, 16.0),
        ("side_profile", 96.0, 12.0),
    )
    for var_i in range(2):
        ma = mantas[var_i]
        wb = wyrms[var_i]
        for a_i, (atag, az, el) in enumerate(comp_angles):
            if not _need("v28_temperament_polarity_and_router"):
                break
            img_a, ren_a = _render_rgb_with_primitives(
                ma["program"],
                baked_map[ma["asset_id"]],
                az_deg=az + rng.uniform(-3.0, 3.0),
                el_deg=el + rng.uniform(-2.0, 2.0),
                panel_size=panel_size,
                studio_preset=(var_i + a_i) % len(STUDIO_LIGHTING_PRESETS),
                scene=ma["scene"],
            )
            img_b, ren_b = _render_rgb_with_primitives(
                wb["program"],
                baked_map[wb["asset_id"]],
                az_deg=az + rng.uniform(-3.0, 3.0),
                el_deg=el + rng.uniform(-2.0, 2.0),
                panel_size=panel_size,
                studio_preset=(var_i + a_i + 1) % len(STUDIO_LIGHTING_PRESETS),
                scene=wb["scene"],
            )
            pa = _format_four_primitives_for_render(
                ren_a,
                primary_candidates=("head", "neck_1"),
                secondary_candidates=("wing_l_wrist", "tail_2"),
                pivot_candidates=("head", "wing_l_upper"),
                dash_label="pole_a_flow",
            )
            pb = _format_four_primitives_for_render(
                ren_b,
                primary_candidates=("head", "jaw"),
                secondary_candidates=("wing_l_wrist", "tail_2"),
                pivot_candidates=("head", "jaw"),
                dash_label="pole_b_glare",
            )
            lang = "zh" if (var_i == 1 and a_i in (0, 2)) else "en"
            if lang == "zh":
                prompt = (
                    f"对比这两张全彩着色 `v28` 传奇生物渲染图（视角：`{atag}`）：图1与图2分别属于 "
                    "**A极（温和可亲）** 还是 **B极（暴怒不可接近）**？请结合 <point>、<box>、<dash>、<circle> "
                    "逐项对比眉弓 `Rx/Rz` 符号定律、正视头部俯仰角、眼眶台平衡、下颌张角与明暗对比度。"
                )
                think = (
                    f"图1头部位于 {pa['point1']} {pa['box1']}，枢轴 {pa['circle']}，流线轴 {pa['dash']}，"
                    "眉弓 Rx=-8° 上扬开放、口唇 0° 闭合、无尖刺、3:1 柔和过渡 -> A极。"
                    f"图2头部位于 {pb['point1']} {pb['box1']}，枢轴 {pb['circle']}，俯视攻击轴 {pb['dash']}，"
                    "眉弓 Rx=+20°/Rz=∓28° 形成 \\ / 怒眉、累计 +8° 下压凝视、14° 静息嘶吼张口+外部獠牙、10:1 强烈明暗对比 -> B极。"
                )
                body = (
                    f"### A极 vs B极 双图视觉审计（`{atag}`）\n"
                    f"1. **图1 = Pole A（温和可亲，`celestial_reef_manta_guardian`）：** 定位 {pa['point1']} {pa['box1']}；"
                    "圆润凸轮廓、开放眉弓 `Rx = -8° <= 0°`、平衡凹陷温暖杏仁眼、安详 `0°` 闭唇、`3:1` 柔和配色。\n"
                    f"2. **图2 = Pole B（暴怒不可接近，`cinder_veil_dread_wyrm`）：** 定位 {pb['point1']} {pb['box1']}；"
                    "下压 `\\ /` 怒眉（`Rx = +20°`, `Rz = ∓28°`）、累计 `+8°` 正视下压凝视、零凸出平衡眼眶台、静息 `14°` 嘶吼张口与外露3D犬齿、"
                    "外展肩肘尖刺以及 `10:1` 炭黑-雪白-绯红强明暗对比。"
                )
            else:
                prompt = (
                    f"Compare these two full-color shaded `v28` legendary creature renders (`{atag}` view). "
                    "Classify Image 1 and Image 2 into **Pole A (Gentle & Approachable)** vs **Pole B (Furious & Unapproachable)**, "
                    "ground cranial/structural evidence using <point>, <box>, <dash>, and <circle>, and contrast the "
                    "Eyebrow Euler Sign Law (`Rx > 0 deg` vs `Rx <= 0 deg`), dead-front gaze pitch, orbital shelf, jaw gape, and chiaroscuro."
                )
                think = (
                    f"Image 1 cranium at {pa['point1']} {pa['box1']} with {pa['circle']} and {pa['dash']} shows smooth convex "
                    "chamfers, open brow Rx=-8 deg (<=0 deg), 0 deg closed lip, and 3:1 luminous harmony -> Pole A. "
                    f"Image 2 skull at {pb['point1']} {pb['box1']} with {pb['circle']} and {pb['dash']} enforces Rx=+20 deg / "
                    "Rz=-28/+28 deg scowl, +8 deg downward predator glare, zero-extrusion orbital eye shelf, 14 deg snarl gape "
                    "plus external 3D canines, hazard spikes, and 10:1 chiaroscuro -> Pole B."
                )
                body = (
                    f"### Pole A vs. Pole B Comparative Audit (`{atag}`)\n"
                    f"- **Image 1 -> Pole A (Gentle & Approachable, `D4 x M3`):** {pa['point1']} {pa['box1']} {pa['circle']}. "
                    "Open brow (`Rx = -8 deg <= 0 deg` lifts anterior glabella), zero hostile body spikes, warm visible almond eyes, "
                    "peaceful `0 deg` closed jaw, and soft `3:1` contrast.\n"
                    f"- **Image 2 -> Pole B (Furious & Unapproachable, `D4 x M4`):** {pb['point1']} {pb['box1']} {pb['circle']}. "
                    "Plunging `\\ /` scowl (`Rx = +20 deg`, `Rz = -28/+28 deg`), cumulative `+8 deg` dead-front downward glare, "
                    "balanced zero-extrusion eye shelf, tense `14 deg` snarl gape + external 3D canines, outward shoulder/elbow spikes, "
                    "and `10:1` charcoal-to-alabaster chiaroscuro."
                )
            _add(
                "v28_temperament_polarity_and_router",
                f"pole_compare_v{var_i}_{atag}",
                prompt,
                think,
                body,
                language=lang,
                images=[img_a, img_b],
                pixel_indispensable=True,
                meta_extra={"comparison": "pole_a_vs_pole_b", "view": atag, "variant": var_i},
            )

    # 1C. 4 Domain D5 Mechanical vs D4 Organic Differentiation Records (4 en)
    d5_pairs = (
        (crawlers[0], wyrms[0], 144.0, 18.0),
        (crawlers[1], mantas[0], 216.0, 18.0),
        (airships[0], wyrms[1], 148.0, 14.0),
        (airships[1], mantas[1], 212.0, 14.0),
    )
    for d_i, (mech_mw, org_mw, az, el) in enumerate(d5_pairs):
        if not _need("v28_temperament_polarity_and_router"):
            break
        im_m, ren_m = _render_rgb_with_primitives(
            mech_mw["program"],
            baked_map[mech_mw["asset_id"]],
            az_deg=az,
            el_deg=el,
            panel_size=panel_size,
            studio_preset=d_i % len(STUDIO_LIGHTING_PRESETS),
            scene=mech_mw["scene"],
        )
        im_o, ren_o = _render_rgb_with_primitives(
            org_mw["program"],
            baked_map[org_mw["asset_id"]],
            az_deg=az,
            el_deg=el,
            panel_size=panel_size,
            studio_preset=(d_i + 2) % len(STUDIO_LIGHTING_PRESETS),
            scene=org_mw["scene"],
        )
        pm = _format_four_primitives_for_render(
            ren_m,
            primary_candidates=("cab_module", "airship_root", "chassis_root"),
            secondary_candidates=("wheel_axle_fl", "rotor_prop_l", "turret_yaw"),
            pivot_candidates=("wheel_axle_fl", "rotor_prop_l", "turret_yaw", "deck_turret"),
            dash_label="d5_mech_span",
        )
        po = _format_four_primitives_for_render(
            ren_o,
            primary_candidates=("head", "neck_1"),
            secondary_candidates=("wing_l_wrist", "tail_2"),
            pivot_candidates=("wing_l_upper", "head"),
            dash_label="d4_organic_axis",
        )
        prompt = (
            "Given these two full-color `v28` 3D renders, route each asset to its exact Domain (`D1-D5`), "
            "annotate mechanical vs biological pivots with <point>, <box>, <dash>, and <circle>, and specify "
            "where `Sim(3)` muscle `scale_bulge` is permitted vs strictly forbidden (`0%` squish)."
        )
        think = (
            f"Image 1 is {mech_mw['base_name']} ({pm['point1']} {pm['box1']}, rotary joint {pm['circle']}, span {pm['dash']}) -> "
            f"Domain D5 rigid vehicle/mecha where MechanicalSubsystem exempts every metal bone from Sim(3) bulge (scale==[1,1,1]). "
            f"Image 2 is {org_mw['base_name']} ({po['point1']} {po['box1']}, joint {po['circle']}, axis {po['dash']}) -> "
            "Domain D4 organic legendary creature allowing volume-preserving Sim(3) scale_bulge (det(S)==1.0) on chest/neck/muscles "
            "with preserve_grounded_limb_length=True."
        )
        body = (
            f"- **Image 1 -> `({mech_mw['domain']}, {mech_mw['mood']})` Rigid Mechanical (`{mech_mw['base_name']}`):** "
            f"{pm['point1']} {pm['box1']} | {pm['circle']} | {pm['dash']}. Uses rigid `SO(3)/SE(3)` mechanical joints "
            "(`wheel_axle`, `steered_wheel_knuckle`, `turret_yaw`, `rotor_propeller`, `hatch_door`) and enforces **`0%` `Sim(3)` "
            "squishy scale bulge** (`scale == [1, 1, 1]` on all rigid metal/timber bones).\n"
            f"- **Image 2 -> `({org_mw['domain']}, {org_mw['mood']})` Organic Legendary (`{org_mw['base_name']}`):** "
            f"{po['point1']} {po['box1']} | {po['circle']} | {po['dash']}. Supports biological `Sim(3)` volume-preserving "
            "`scale_bulge` (`det(S) == 1.0`) on soft anatomical masses while keeping axial length locked on grounded IK limbs."
        )
        _add(
            "v28_temperament_polarity_and_router",
            f"d5_vs_d4_route_{d_i}",
            prompt,
            think,
            body,
            language="en",
            images=[im_m, im_o],
            pixel_indispensable=True,
            meta_extra={"d5_asset": mech_mw["asset_id"], "d4_asset": org_mw["asset_id"]},
        )

    # -----------------------------------------------------------------------
    # Task 2: `v28_d5_vehicle_mecha_rig` (16 records: 12 visual + 4 DSL, 3 zh)
    # -----------------------------------------------------------------------
    d5_all = crawlers + airships
    # 2A. 4 Concept -> Full Verified Block-DSL v4.1 (text-to-DSL)
    for mw in d5_all:
        g = mw["verification"]["g1_g12"]
        prompt = (
            f"Author a complete, SyncCraft-verified `Block-DSL v4.1` (`dsl_version: \"4.1\"`, `audit_profile: sculpted`, "
            f"`texel_density: 16`) Domain `D5` vehicle/mecha rig for: {mw['concept']} "
            "Enforce chassis/keel-first hierarchy, recessed fender wells or outrigger pylons, 4-band true octagon rotary hubs, "
            "sloped glass cab, continuous `joint_limits` on rolling axles/propellers, finite limits on steering/turrets/hatches, "
            "and strictly zero `@muscles` (`0%` `Sim(3)` metal bulge)."
        )
        think = (
            f"Domain D5 rigid vehicle design for {mw['asset_id']} ({mw['mood']}): anchor root chassis/keel, "
            "group 4-band rotated octagonal wheels/propellers with prism_group, keep continuous spin limits on axles/rotors "
            "and bounded limits on steering knuckles/turrets/hatches, omit @muscles so rigid steel has 0% Sim(3) squish. "
            f"Verified: elements={g['total_elements']}, bones={g['bones']}, mirror_ok={g['mirror_ok']}, G1-G12 strict=True, clips=PASS."
        )
        body = f"```blockdsl\n{mw['dsl']}\n```"
        _add(
            "v28_d5_vehicle_mecha_rig",
            f"d5_dsl_{mw['asset_id']}",
            prompt,
            think,
            body,
            language="en",
            images=[],
            pixel_indispensable=False,
            meta_extra={"domain": "D5", "asset_id": mw["asset_id"], "elements": g["total_elements"]},
        )

    # 2B. 4 Multi-View Full-Color Turnaround -> 4-Primitive Breakdown + Verified Block-DSL v4.1
    for i, mw in enumerate(d5_all):
        if not _need("v28_d5_vehicle_mecha_rig"):
            break
        im_f34, ren_f34 = _render_rgb_with_primitives(
            mw["program"],
            baked_map[mw["asset_id"]],
            az_deg=146.0 + rng.uniform(-4.0, 4.0),
            el_deg=18.0,
            panel_size=panel_size,
            studio_preset=i % len(STUDIO_LIGHTING_PRESETS),
            scene=mw["scene"],
        )
        im_side, _ = _render_rgb_with_primitives(
            mw["program"],
            baked_map[mw["asset_id"]],
            az_deg=92.0 + rng.uniform(-4.0, 4.0),
            el_deg=14.0,
            panel_size=panel_size,
            studio_preset=(i + 1) % len(STUDIO_LIGHTING_PRESETS),
            scene=mw["scene"],
        )
        pm = _format_four_primitives_for_render(
            ren_f34,
            primary_candidates=("cab_module", "airship_root"),
            secondary_candidates=("cargo_module", "gasbag_hull"),
            pivot_candidates=("wheel_axle_fl", "rotor_prop_l"),
            dash_label=f"chassis_axis:{mw['base_name']}",
        )
        prompt = (
            "Inspect these two full-color shaded `16 px/block` views (front-3/4 and side profile) of a Domain `D5` "
            "vehicle rig. Ground its primary chassis/cab, rotary hub pivot, and hull span with <point>, <box>, <dash>, "
            "and <circle>, then reconstruct the exact SyncCraft-verified `Block-DSL v4.1` program."
        )
        think = (
            f"Inverse-engineering D5 vehicle {mw['asset_id']}: front assembly {pm['point1']} {pm['box1']}, "
            f"aft hull {pm['point2']} {pm['box2']}, longitudinal wheelbase/keel {pm['dash']}, rotary pivot {pm['circle']}. "
            "Preserve 4-band octagon hubs, sloped glass cab, rigid mechanical joints, and 0% Sim(3) squish."
        )
        body = (
            f"Grounded rigid D5 assemblies: {pm['point1']} {pm['box1']} | {pm['circle']} | {pm['dash']}\n\n"
            f"```blockdsl\n{mw['dsl']}\n```"
        )
        _add(
            "v28_d5_vehicle_mecha_rig",
            f"d5_inverse_{mw['asset_id']}",
            prompt,
            think,
            body,
            language="en",
            images=[im_f34, im_side],
            pixel_indispensable=True,
            meta_extra={"domain": "D5", "asset_id": mw["asset_id"]},
        )

    # 2C. 4 Articulated 2-Frame Rigid Mechanical Motion Audits (2 zh, 2 en)
    d5_motion_specs = (
        (
            crawlers[0],
            {"steer_fl": [0.0, 26.0, 0.0], "steer_fr": [0.0, 26.0, 0.0], "turret_yaw": [0.0, 58.0, 0.0], "wheel_axle_fl": [45.0, 0.0, 0.0]},
            142.0,
            22.0,
            "turret_yaw",
            "turret_yaw",
            "turret_yaw_sweep",
            "Ackermann front steering (+26 deg) + dorsal turret yaw (+58 deg) + rigid octagon wheel roll (+45 deg)",
            "前轮阿克曼转向（+26°）+ 顶部传感器炮塔回转（+58°）+ 刚性八边形车轮滚动（+45°）",
            "en",
        ),
        (
            crawlers[1],
            {"hatch_door": [-68.0, 0.0, 0.0], "sensor_pitch": [28.0, 0.0, 0.0]},
            38.0,
            18.0,
            "hatch_door",
            "hatch_door",
            "hatch_door_deploy",
            "rear cargo ramp `hatch_door` unfolding outward (-68 deg X) + sensor pod elevation (+28 deg X)",
            "尾门货舱坡道 `hatch_door` 向外展开（-68° X）+ 传感器仰角抬升（+28° X）",
            "zh",
        ),
        (
            airships[0],
            {"nacelle_l": [0.0, 0.0, 0.0], "rotor_prop_l": [0.0, 0.0, 45.0], "deck_turret": [0.0, 64.0, 0.0], "fin_l": [22.0, 0.0, 0.0], "fin_r": [22.0, 0.0, 0.0]},
            42.0,
            16.0,
            "fin_l",
            "deck_turret",
            "thruster_vane_pitch",
            "twin outrigger `rotor_propeller` axial spin (+45 deg Z) + `deck_turret` yaw (+64 deg) + `gimbal_thruster` vanes (+22 deg X)",
            "双侧推进吊舱旋翼轴向旋转（+45° Z）+ 甲板罗盘塔回转（+64°）+ 矢量尾舵偏转（+22° X）",
            "en",
        ),
        (
            airships[1],
            {"cargo_hatch": [-62.0, 0.0, 0.0], "deck_turret": [0.0, -52.0, 0.0]},
            36.0,
            14.0,
            "cargo_hatch",
            "cargo_hatch",
            "cargo_hatch_open",
            "aft gondola `cargo_hatch` lowering outward (-62 deg X) while top `deck_turret` scans (-52 deg Y)",
            "吊舱尾部货门 `cargo_hatch` 向外下放（-62° X）并联动顶部罗盘塔扫描（-52° Y）",
            "zh",
        ),
    )
    for m_i, (mw, pdeltas, az, el, eff_b, piv_b, arc_lbl, desc_en, desc_zh, lang) in enumerate(d5_motion_specs):
        if not _need("v28_d5_vehicle_mecha_rig"):
            break
        art = _render_two_frame_articulated(
            mw["program"],
            mw["scene"],
            baked_map[mw["asset_id"]],
            pose_deltas=pdeltas,
            az_deg=az,
            el_deg=el,
            panel_size=panel_size,
            studio_preset=m_i % len(STUDIO_LIGHTING_PRESETS),
            effector_bone=eff_b,
            pivot_bone=piv_b,
            arc_label=arc_lbl,
        )
        if lang == "zh":
            prompt = (
                "对比这两帧全彩着色 D5 载具渲染图（静息帧 -> 运动帧）。用 <point>、<box>、<circle>、<dash> "
                "标出刚性机械关节枢轴与运动弧线，并说明为何 `v28` 机械子系统严禁对钢制底盘、轮毂与舱门施加 `Sim(3)` 鼓包缩放。"
            )
            think = (
                f"检验 {mw['asset_id']} 两帧刚性运动：静息位 {art['pt_rest']} -> 运动位 {art['pt_posed']} "
                f"{art['box_posed']}，枢轴 {art['piv_circle']}，运动弧 {art['arc_dash']}（投影位移 {art['disp_px']}）。"
                "所有金属骨骼均归属 MechanicalSubsystem，Sim(3) 缩放锁定 [1,1,1]，严禁软体形变。"
            )
            body = (
                f"### D5 刚性机械运动审计（`{mw['base_name']}`）\n"
                f"- **已执行机械动作：** {desc_zh}\n"
                f"- **4原语标注：** 枢轴 {art['piv_circle']} | 轨迹 {art['arc_dash']} | 末态 {art['pt_posed']} {art['box_posed']}\n"
                "- **刚体守恒法则：** `is_rigid_mechanical_bone = True` 自动豁免 `Sim(3)` 肌肉鼓包，保证钢制结构缩放恒为 `[1.0, 1.0, 1.0]`。"
            )
        else:
            prompt = (
                "Inspect this 2-frame full-color Domain `D5` mechanical articulation sequence (Rest -> Active). "
                "Mark the rigid mechanical hinge pivot and sweep arc using <point>, <box>, <circle>, and <dash>, "
                "and explain the `v28` `MechanicalSubsystem` `0%` `Sim(3)` squish exemption."
            )
            think = (
                f"Tracking rigid SE(3)/SO(3) motion on {mw['asset_id']}: rest {art['pt_rest']} -> active {art['pt_posed']} "
                f"{art['box_posed']}, pivot {art['piv_circle']}, motion sweep {art['arc_dash']} ({art['disp_px']}u screen shift). "
                "Rigid steel/timber components stay exempt from Sim(3) muscle bulge (scale == [1,1,1])."
            )
            body = (
                f"### D5 Rigid Mechanical Articulation Audit (`{mw['base_name']}`)\n"
                f"- **Observed Motion:** {desc_en}\n"
                f"- **4-Primitive Verification:** {art['piv_circle']} | {art['arc_dash']} | {art['pt_posed']} {art['box_posed']}\n"
                "- **Zero-Squish Mechanical Law:** `MechanicalSubsystem` classifies wheels, knuckles, turrets, rotors, and hatches "
                "as rigid bones (`is_rigid_mechanical_bone = True`), keeping `scale == [1.0, 1.0, 1.0]` (`0%` `Sim(3)` metal bulge)."
            )
        _add(
            "v28_d5_vehicle_mecha_rig",
            f"d5_motion_{m_i}_{mw['base_name']}",
            prompt,
            think,
            body,
            language=lang,
            images=[art["img_rest"], art["img_posed"]],
            pixel_indispensable=True,
            meta_extra={"domain": "D5", "articulated_two_frame": True, "asset_id": mw["asset_id"]},
        )

    # 2D. 4 Focused D5 Mechanical Joint & 4-Band Octagon Engineering Cards (1 zh, 3 en)
    d5_focus = (
        (
            crawlers[0],
            122.0,
            12.0,
            ("wheel_axle_fl", "steer_fl"),
            ("fender_fl", "cab_module"),
            ("wheel_axle_fl",),
            "wheel_well_clearance",
            "4-Band True Octagon Wheel + Recessed Fender Well + `steered_wheel_knuckle`",
            "Build each tire from 4 concentric boxes rotated `0°/45°/90°/135°` around X sharing `prism_group` (never a flat cube "
            "or `>8`-band fake-Blender cylinder). Mount `wheel_axle_fl` (`joint_limits: [continuous, [-2,2], [-2,2]]`) as a child of "
            "`steer_fl` (`[[-2,2], [-35,35], [-2,2]]`) tucked cleanly under chamfered `fender_fl` arches with `>= 0.5u` clearance.",
            "en",
        ),
        (
            crawlers[1],
            158.0,
            26.0,
            ("turret_yaw", "sensor_pitch"),
            ("cab_module", "cargo_module"),
            ("turret_yaw",),
            "turret_to_cab",
            "Rigid `turret_yaw` + `cannon_elevation` (`sensor_pitch`) Separation",
            "Split yaw (`[[-2,2], [-180,180], [-2,2]]`) and pitch (`[[-12,45], [-2,2], [-2,2]]`) onto consecutive parent/child "
            "bones so the sensor pod elevates cleanly without gimbal twist or cargo-bed collision.",
            "en",
        ),
        (
            airships[0],
            130.0,
            14.0,
            ("rotor_prop_l", "nacelle_l"),
            ("gasbag_hull", "airship_root"),
            ("rotor_prop_l",),
            "rotor_to_keel",
            "Outrigger `rotor_propeller` Continuous Z-Spin & Vaulted Chamfered Gasbag",
            "Suspend twin engine pods laterally clear of the timber keel via `nacelle_l` (`mirror_x(nacelle_r)`), drive "
            "4-blade hubs on `joint_limits: [[-2,2], [-2,2], continuous]` with `spin(axis=z, amp_deg=360, freq_hz=0.5)`, "
            "and chamfer the overhead gasbag with `+-24 deg` rolled `rib_band_l/r` slabs.",
            "en",
        ),
        (
            airships[1],
            44.0,
            16.0,
            ("cargo_hatch", "fin_l"),
            ("gasbag_hull", "deck_turret"),
            ("cargo_hatch",),
            "hatch_to_turret",
            "尾部 `hatch_door` 舱盖与矢量尾舵 `gimbal_thruster` 装配规范",
            "尾门 `cargo_hatch` 必须以底沿为铰链轴（`joint_limits: [[-105, 10], [-2, 2], [-2, 2]]`），沿负 `-X` 方向向外下放，"
            "避免向内摆动刺穿船体；尾舵 `fin_l/r` 限定 `±24°` 双轴偏转，且全部刚体关节严禁启用 `Sim(3)` 肌肉鼓包。",
            "zh",
        ),
    )
    for f_i, (mw, az, el, c1, c2, cp, dlbl, title_txt, detail_txt, lang) in enumerate(d5_focus):
        if not _need("v28_d5_vehicle_mecha_rig"):
            break
        img, ren = _render_rgb_with_primitives(
            mw["program"],
            baked_map[mw["asset_id"]],
            az_deg=az,
            el_deg=el,
            panel_size=panel_size,
            studio_preset=(f_i + 1) % len(STUDIO_LIGHTING_PRESETS),
            scene=mw["scene"],
        )
        pm = _format_four_primitives_for_render(
            ren,
            primary_candidates=c1,
            secondary_candidates=c2,
            pivot_candidates=cp,
            dash_label=dlbl,
        )
        if lang == "zh":
            prompt = (
                f"观察这张全彩 D5 载具渲染图，结合 <point>、<box>、<dash>、<circle> 讲解 `{title_txt}` 的精确骨骼铰链与防穿模设计规范。"
            )
            think = (
                f"定位目标机械子装配 {pm['point1']} {pm['box1']}、参照船体 {pm['point2']}、铰链枢轴 {pm['circle']} 与跨度 {pm['dash']}。"
                "验证右手法则旋转方向与无穿模限位。"
            )
            body = (
                f"### {title_txt}\n"
                f"- **视觉原语定位：** {pm['point1']} {pm['box1']} | 枢轴 {pm['circle']} | 参照矢量 {pm['dash']}\n"
                f"- **工程规范：** {detail_txt}"
            )
        else:
            prompt = (
                f"Analyze this full-color Domain `D5` render and explain the exact `v28` mechanical rigging contract for "
                f"**{title_txt}**, grounding the joint and clearance geometry with <point>, <box>, <dash>, and <circle>."
            )
            think = (
                f"Grounding D5 sub-assembly {pm['point1']} {pm['box1']}, reference hull {pm['point2']}, "
                f"rotary pivot {pm['circle']}, and clearance vector {pm['dash']} on {mw['asset_id']}."
            )
            body = (
                f"### {title_txt}\n"
                f"- **4-Primitive Grounding:** {pm['point1']} {pm['box1']} | {pm['circle']} | {pm['dash']}\n"
                f"- **Mechanical Contract:** {detail_txt}"
            )
        _add(
            "v28_d5_vehicle_mecha_rig",
            f"d5_focus_{f_i}",
            prompt,
            think,
            body,
            language=lang,
            images=[img],
            pixel_indispensable=True,
            meta_extra={"domain": "D5", "focus": dlbl},
        )

    # -----------------------------------------------------------------------
    # Task 3: `v28_pole_a_vs_pole_b_masterworks` (16 records: 12 visual + 4 DSL, 3 zh)
    # -----------------------------------------------------------------------
    leg_all = mantas + wyrms
    # 3A. 4 Concept -> Full Verified Block-DSL v4.1 (2 Pole A + 2 Pole B)
    for mw in leg_all:
        g = mw["verification"]["g1_g12"]
        gr = mw["verification"]["grounding"]
        prompt = (
            f"Design and emit a complete, SyncCraft-verified `Block-DSL v4.1` (`dsl_version: \"4.1\"`, "
            f"`audit_profile: sculpted`, `texel_density: 16`) legendary creature for Target "
            f"`({mw['domain']}, {mw['mood']}, {mw['polarity']})`: {mw['concept']}"
        )
        think = (
            f"Locking {mw['polarity']} before Cube #1 for {mw['asset_id']}. "
            + (
                "Pole A law: smooth convex chamfers (0 spikes), open brow Rx=-8 deg (<=0 deg), peaceful 0 deg jaw, 3:1 luminous harmony."
                if mw["polarity"] == "pole_a_gentle_approachable"
                else "Pole B law: plunging scowl brow Rx=+20 deg / Rz=-28/+28 deg, cumulative +8 deg dead-front glare, zero-extrusion eye shelf, 14 deg snarl gape + external canines, coiled crouch, 4-foot ground lock, 10:1 chiaroscuro."
            )
            + f" Verified: elements={g['total_elements']}, bones={g['bones']}, mirror_ok={g['mirror_ok']}, rest_feet={gr['rest_feet']}."
        )
        body = f"```blockdsl\n{mw['dsl']}\n```"
        _add(
            "v28_pole_a_vs_pole_b_masterworks",
            f"leg_dsl_{mw['asset_id']}",
            prompt,
            think,
            body,
            language="en",
            images=[],
            pixel_indispensable=False,
            meta_extra={"domain": mw["domain"], "mood": mw["mood"], "polarity": mw["polarity"]},
        )

    # 3B. 4 Multi-View Full-Color Turnaround -> 4-Primitive Critique + Verified Block-DSL v4.1
    for i, mw in enumerate(leg_all):
        if not _need("v28_pole_a_vs_pole_b_masterworks"):
            break
        im_34, ren_34 = _render_rgb_with_primitives(
            mw["program"],
            baked_map[mw["asset_id"]],
            az_deg=148.0 + rng.uniform(-4.0, 4.0),
            el_deg=16.0,
            panel_size=panel_size,
            studio_preset=i % len(STUDIO_LIGHTING_PRESETS),
            scene=mw["scene"],
        )
        im_side, _ = _render_rgb_with_primitives(
            mw["program"],
            baked_map[mw["asset_id"]],
            az_deg=96.0 + rng.uniform(-4.0, 4.0),
            el_deg=12.0,
            panel_size=panel_size,
            studio_preset=(i + 2) % len(STUDIO_LIGHTING_PRESETS),
            scene=mw["scene"],
        )
        pr = _format_four_primitives_for_render(
            ren_34,
            primary_candidates=("head", "neck_1"),
            secondary_candidates=("wing_l_wrist", "tail_2"),
            pivot_candidates=("head", "wing_l_upper"),
            dash_label=f"silhouette:{mw['polarity']}",
        )
        prompt = (
            "Inspect these two full-color shaded `16 px/block` views (front-3/4 and side profile) of a legendary "
            "`v28` creature. Ground its head, wing/body silhouette flow, and joint pivot with <point>, <box>, <dash>, "
            "and <circle>, state its Pole A vs Pole B temperament polarity, and output the exact verified `Block-DSL v4.1`."
        )
        think = (
            f"Inverse analysis of {mw['asset_id']} ({mw['polarity']}): cranium {pr['point1']} {pr['box1']}, "
            f"extremity {pr['point2']} {pr['box2']}, flow vector {pr['dash']}, hinge pivot {pr['circle']}."
        )
        body = (
            f"**Polarity:** `{mw['polarity']}` (`{mw['domain']} x {mw['mood']}`) — "
            f"{pr['point1']} {pr['box1']} | {pr['dash']} | {pr['circle']}\n\n"
            f"```blockdsl\n{mw['dsl']}\n```"
        )
        _add(
            "v28_pole_a_vs_pole_b_masterworks",
            f"leg_inverse_{mw['asset_id']}",
            prompt,
            think,
            body,
            language="en",
            images=[im_34, im_side],
            pixel_indispensable=True,
            meta_extra={"polarity": mw["polarity"], "asset_id": mw["asset_id"]},
        )

    # 3C. 4 Dead-Front (`az=180 deg`) + Front-3/4 Cranial/Orbital Shelf Audits (2 zh, 2 en)
    for i, mw in enumerate(leg_all):
        if not _need("v28_pole_a_vs_pole_b_masterworks"):
            break
        im_df, ren_df = _render_rgb_with_primitives(
            mw["program"],
            baked_map[mw["asset_id"]],
            az_deg=180.0 + rng.uniform(-2.0, 2.0),
            el_deg=8.0,
            panel_size=panel_size,
            studio_preset=(i + 1) % len(STUDIO_LIGHTING_PRESETS),
            scene=mw["scene"],
        )
        im_q, _ = _render_rgb_with_primitives(
            mw["program"],
            baked_map[mw["asset_id"]],
            az_deg=210.0,
            el_deg=14.0,
            panel_size=panel_size,
            studio_preset=(i + 3) % len(STUDIO_LIGHTING_PRESETS),
            scene=mw["scene"],
        )
        pr = _format_four_primitives_for_render(
            ren_df,
            primary_candidates=("head",),
            secondary_candidates=("jaw", "neck_1", "pelvis_root"),
            pivot_candidates=("head", "jaw"),
            dash_label="dead_front_glare",
        )
        lang = "zh" if i in (1, 3) else "en"
        is_pole_b = mw["polarity"] == "pole_b_furious_unapproachable"
        if lang == "zh":
            prompt = (
                "检查这两张全彩正视（`0°` dead-front）与前3/4视角渲染图。用 <point>、<box>、<dash>、<circle> "
                "标出颅骨、下颌与颈部枢轴，并依据 `v28` 正视透视与零凸出眼眶台定律审查眉弓、凝视角与嘴部姿态。"
            )
            think = (
                f"正视0°核验 {mw['asset_id']}：头部 {pr['point1']} {pr['box1']}，下颌/颈部参考 {pr['point2']}，"
                f"俯视轴线 {pr['dash']}，颅颈枢轴 {pr['circle']}。"
                + (
                    "B极校验：neck_1(-20°)+neck_2(+10°)+head(+18°)=+8° 正视下压凝视；brow Rx=+20°/Rz=∓28° 怒眉；眼眶零凸出；14° 静息张口+外部犬齿。"
                    if is_pole_b
                    else "A极校验：neck_1(-6°)+head(+4°)=-2° 温和视线；brow Rx=-8° 开放上扬；眼眶平衡可见；0° 闭合安详嘴唇；零敌意尖刺。"
                )
            )
            body = (
                f"### `v28` 正视（`0°`）颅面与眼眶台审计 — `{mw['polarity']}`\n"
                f"- **空间标注：** 颅骨 {pr['point1']} {pr['box1']} | 视线/下颌矢量 {pr['dash']} | 枢轴 {pr['circle']}\n"
                + (
                    "- **B极暴怒校验（PASS）：** 累计世界俯仰角 `-20° + 10° + 18° = +8°` 下压捕食凝视（杜绝仰下巴堆砖）；"
                    "眉弓 `Rx = +20° > 0°` 且 `Rz = ∓28°` 刀刻 `\\ /` 怒眉；`eye_l/r` 置于平衡凹陷眼眶台（正视与侧视均清晰且不超出眉弓外缘）；"
                    "下颌保持 `14°` 嘶吼张角并配外置3D獠牙。"
                    if is_pole_b
                    else "- **A极温和校验（PASS）：** 累计头部仰角 `-6° + 4° = -2°` 舒展友好视线；"
                    "眉弓 `Rx = -8° <= 0°` 抬升前缘呈开放慈和态；温暖发光杏仁眼内嵌于平滑眼眶台无外凸；"
                    "下颌 `0°` 安详闭合，整体轮廓圆润无刺。"
                )
            )
        else:
            prompt = (
                "Audit these full-color `0 deg` dead-front and front-3/4 renders under the `v28` Dead-Front Perspective "
                "& Zero-Extrusion Orbital Eye Shelf Law. Ground the cranium, jaw, and atlas pivot with <point>, <box>, "
                "<dash>, and <circle>, and verify the cumulative world head pitch and brow Euler signs."
            )
            think = (
                f"Checking dead-front 0 deg render of {mw['asset_id']}: head {pr['point1']} {pr['box1']}, "
                f"reference {pr['point2']}, gaze/jaw axis {pr['dash']}, atlas pivot {pr['circle']}. "
                + (
                    "Pole B: cumulative X-pitch -20+10+18 = +8 deg downward predator glare; brow Rx=+20 deg, Rz=-28/+28 deg; "
                    "eye_l/r balanced inside brow overhang (zero extrusion); 14 deg resting snarl gape + external 3D canines."
                    if is_pole_b
                    else "Pole A: cumulative X-pitch -6+4 = -2 deg open gentle gaze; brow Rx=-8 deg (<=0 deg); "
                    "warm glowing almond eyes cleanly recessed; 0 deg closed peaceful lip; zero hostile spikes."
                )
            )
            body = (
                f"### `v28` Dead-Front (`0 deg`) Cranial & Orbital Shelf Audit — `{mw['polarity']}`\n"
                f"- **4-Primitive Grounding:** {pr['point1']} {pr['box1']} | {pr['dash']} | {pr['circle']}\n"
                + (
                    "- **Pole-B Checks (PASS):** Cumulative world head X-pitch `= -20 + 10 + 18 = +8 deg` (locks inside "
                    "`+6..+10 deg` downward predator glare, never `-10..-15 deg` chin-up Lego stacking); brow `Rx = +20 deg > 0`, "
                    "`Rz = -28/+28 deg` carves a continuous `\\ /` razor scowl; `eye_l/r` sit on the balanced recessed shelf "
                    "(visible in dead-front & profile with `0` outward extrusion past `brow_l/r`); `jaw` holds a tense `14 deg` "
                    "snarl gape framed by external 3D canines (`tusk_upper_l/r`)."
                    if is_pole_b
                    else "- **Pole-A Checks (PASS):** Cumulative world head X-pitch `= -6 + 4 = -2 deg` calm welcoming gaze; "
                    "brow `Rx = -8 deg <= 0 deg` lifts the anterior glabella gently; warm luminous eyes sit recessed inside "
                    "smooth chamfers; `jaw` rests cleanly closed at `0 deg` with zero hostile spikes."
                )
            )
        _add(
            "v28_pole_a_vs_pole_b_masterworks",
            f"cranial_deadfront_{mw['asset_id']}",
            prompt,
            think,
            body,
            language=lang,
            images=[im_df, im_q],
            pixel_indispensable=True,
            meta_extra={"polarity": mw["polarity"], "dead_front": True},
        )

    # 3D. 4 Two-Frame Articulated Action Renders (Pole A swim_fins vs Pole B roar_snarl, 1 zh, 3 en)
    leg_actions = (
        (
            mantas[0],
            {"wing_l_upper": [0.0, 0.0, -24.0], "wing_r_upper": [0.0, 0.0, 24.0], "wing_l_fore": [0.0, 0.0, -14.0], "wing_r_upper__wing_l_fore": [0.0, 0.0, 14.0], "tail_1": [0.0, 22.0, 0.0]},
            150.0,
            24.0,
            "wing_l_wrist",
            "wing_l_upper",
            "swim_fins_stroke",
            "Pole A `swim_fins` hydrodynamic pectoral stroke with rigid `wing_l_f4` delta `[0, 0, 0]`",
            "en",
        ),
        (
            mantas[1],
            {"wing_l_upper": [0.0, 0.0, 20.0], "wing_r_upper": [0.0, 0.0, -20.0], "neck_1": [-6.0, 14.0, 0.0], "head": [4.0, 10.0, 0.0]},
            210.0,
            22.0,
            "wing_r_upper__wing_l_wrist",
            "wing_r_upper",
            "swim_fins_recovery",
            "A极流线胸鳍上收恢复冲程（保持 `wing_l_f4` 刚性 `[0,0,0]` 零撕裂）",
            "zh",
        ),
        (
            wyrms[0],
            {"neck_1": [-14.0, 0.0, 0.0], "head": [-8.0, 0.0, 0.0], "jaw": [28.0, 0.0, 0.0], "wing_l_upper": [-10.0, 12.0, -26.0], "wing_r_upper": [-10.0, -12.0, 26.0]},
            142.0,
            14.0,
            "wing_l_wrist",
            "wing_l_upper",
            "roar_wing_flare",
            "Pole B furious roar with jaw opening (`+28 deg`) + `tendon_coupling` tongue curl + cathedral wing flare",
            "en",
        ),
        (
            wyrms[1],
            {"jaw": [30.0, 0.0, 0.0], "neck_1": [-10.0, 20.0, 0.0], "head": [-6.0, 14.0, 0.0], "tail_1": [8.0, -24.0, 0.0]},
            216.0,
            16.0,
            "head",
            "neck_1",
            "predator_snarl_strike",
            "Pole B coiled predator lateral snarl strike with opposite-side tail angular-momentum counterweight",
            "en",
        ),
    )
    for a_i, (mw, pdeltas, az, el, eff_b, piv_b, arc_lbl, desc, lang) in enumerate(leg_actions):
        if not _need("v28_pole_a_vs_pole_b_masterworks"):
            break
        art = _render_two_frame_articulated(
            mw["program"],
            mw["scene"],
            baked_map[mw["asset_id"]],
            pose_deltas=pdeltas,
            az_deg=az,
            el_deg=el,
            panel_size=panel_size,
            studio_preset=(a_i + 2) % len(STUDIO_LIGHTING_PRESETS),
            effector_bone=eff_b,
            pivot_bone=piv_b,
            arc_label=arc_lbl,
        )
        if lang == "zh":
            prompt = (
                "对比这两帧全彩着色传奇生物动作渲染图，使用 <point>、<box>、<circle>、<dash> 标注关节枢轴与运动弧，"
                "并核验该动作如何与生物气质极性（A极/B极）及 `wing_anchor_locked` 内翼锁一致。"
            )
            think = (
                f"检验 {mw['asset_id']} 动作序列：静息 {art['pt_rest']} -> 动作 {art['pt_posed']} {art['box_posed']}，"
                f"枢轴 {art['piv_circle']}，运动弧 {art['arc_dash']}。"
                "内翼指骨 wing_l_f4 锁定 [0,0,0]，确保 wing_bay_4 与 wing_inner_sail 零撕裂。"
            )
            body = (
                f"### 动作极性审计：`{mw['polarity']}`\n"
                f"- **动作解析：** {desc}\n"
                f"- **4原语标注：** 枢轴 {art['piv_circle']} | 轨迹 {art['arc_dash']} | 末态 {art['pt_posed']} {art['box_posed']}\n"
                "- **内翼锁校验：** `wing_l_f4` / `wing_r_f4` 保持 `[0, 0, 0]` 增量，杜绝内翼膜分离。"
            )
        else:
            prompt = (
                "Inspect this 2-frame full-color articulated legendary creature sequence (Rest -> Action). "
                "Ground the hinge pivot and effector motion arc with <point>, <box>, <circle>, and <dash>, "
                "and verify alignment with its temperament polarity and `wing_anchor_locked` continuity."
            )
            think = (
                f"Comparing Rest vs Action on {mw['asset_id']} ({mw['polarity']}): rest {art['pt_rest']} -> active "
                f"{art['pt_posed']} {art['box_posed']}, pivot {art['piv_circle']}, arc {art['arc_dash']} ({art['disp_px']}u). "
                "Verify wing_l_f4 remains locked at [0,0,0] across the stroke."
            )
            body = (
                f"### Action & Polarity Audit (`{mw['polarity']}`)\n"
                f"- **Articulation:** {desc}\n"
                f"- **4-Primitive Proof:** {art['piv_circle']} | {art['arc_dash']} | {art['pt_posed']} {art['box_posed']}\n"
                "- **Structural Guarantee:** `wing_l_f4` / `wing_r_f4` stay locked at `[0, 0, 0]` so `wing_bay_4` and "
                "`wing_inner_sail` remain seamlessly sealed throughout the stroke."
            )
        _add(
            "v28_pole_a_vs_pole_b_masterworks",
            f"leg_action_{a_i}_{mw['base_name']}",
            prompt,
            think,
            body,
            language=lang,
            images=[art["img_rest"], art["img_posed"]],
            pixel_indispensable=True,
            meta_extra={"polarity": mw["polarity"], "articulated_two_frame": True},
        )

    # -----------------------------------------------------------------------
    # Task 4: `v28_biomech_solver_and_ground_pin` (16 records: 10 visual + 6 spec, 3 zh)
    # -----------------------------------------------------------------------
    # 4A. 10 Visual Biomechanical Solver Records (2-frame articulated & single-frame solver visuals)
    biomech_visual_cases = (
        (
            "ground_pin_bipedal_roar_en",
            wyrms[0],
            {
                "pelvis_root": [-22.0, 0.0, 0.0],
                "leg_l_thigh": [22.0, 0.0, 0.0],
                "leg_r_thigh": [22.0, 0.0, 0.0],
                "arm_l_upper": [-48.0, 0.0, -14.0],
                "arm_r_upper": [-48.0, 0.0, 14.0],
                "jaw": [28.0, 0.0, 0.0],
            },
            94.0,
            10.0,
            "fore_paw_l",
            "rear_foot_l",
            "bipedal_roar_forepaw_lift",
            "en",
            "Selective Standing Limb IK Pinning (`ground_pin_limbs: [hind_l, hind_r]`) During Bipedal Boss Roar",
            (
                "1. **Root Cause of the `-4..-6.5u` Raw FK Floor-Dip Bug:** Pitching `pelvis_root` (`-22 deg`) during a "
                "standing/roar clip moves the hip sockets in world space; raw Blockbench FK drags hind feet `-4..-6.5u` underground.\n"
                "2. **Selective `ground_pin_limbs: [hind_l, hind_r]` Solution:** Splits `ik_bones = lb.chain[:-1]` above `hind_foot_l/r` "
                "(preventing 3-link heel tilt!), solves closed-loop `SE(3)` world IK locking ankles at rest world coordinates (`< 0.02u` slip), "
                "and applies the closed-form `SO(3)` world-flat sole lock `R_delta = R_foot_rest_local^T (R_parent_posed^T R_foot_rest_world)` "
                "both before AND after spring integration — keeping hind feet locked at `y = 0.00u` while forelimbs rear cleanly skyward!"
            ),
        ),
        (
            "ground_pin_bipedal_roar_zh",
            wyrms[1],
            {
                "pelvis_root": [-20.0, 0.0, 0.0],
                "leg_l_thigh": [20.0, 0.0, 0.0],
                "leg_r_thigh": [20.0, 0.0, 0.0],
                "arm_l_upper": [-44.0, 0.0, -12.0],
                "arm_r_upper": [-44.0, 0.0, 12.0],
                "jaw": [26.0, 0.0, 0.0],
            },
            266.0,
            10.0,
            "arm_r_upper__fore_paw_l",
            "leg_r_thigh__rear_foot_l",
            "bipedal_roar_hind_pin_zh",
            "zh",
            "直立咆哮中的选择性站立肢体 IK 锚定（`ground_pin_limbs: [hind_l, hind_r]`）",
            (
                "1. **原始 FK 穿地缺陷（`-4..-6.5u`）：** 非 `gait` 站立/咆哮剪辑一旦俯仰 `pelvis_root` 或 `chest`，原始 FK 会把脚掌直接拉入地下 `-4..-6.5u`。\n"
                "2. **闭环 `SE(3) + SO(3)` 锚定方案：** 声明 `ground_pin_limbs: [hind_l, hind_r]`，将 IK 链截断于踝关节上方（`lb.chain[:-1]` 防止脚掌扭转），"
                "以阻尼最小二乘锁定后踝世界坐标（滑移 `< 0.02u`），并施加闭式 `SO(3)` 水平脚掌锁 `R_delta = R_foot_rest_local^T (R_parent_posed^T R_foot_rest_world)`，"
                "实现后足死锁 `y = 0.00u` 且前爪自由抬起咆哮！"
            ),
        ),
        (
            "ground_pin_all_four_idle_bite",
            wyrms[0],
            {"neck_1": [12.0, 24.0, 0.0], "neck_2": [8.0, 18.0, 0.0], "head": [6.0, 12.0, 0.0], "jaw": [24.0, 0.0, 0.0]},
            148.0,
            16.0,
            "head",
            "fore_paw_l",
            "grounded_bite_lunge",
            "en",
            "Full 4-Limb Ground Pinning (`ground_pin_limbs: true`) During Standing Bite / Tail-Whip Clips",
            (
                "Declare `ground_pin_limbs: true` on standing `idle`, `bite`, `wing_cloak_sweep`, and `tail_whip` clips so "
                "all four limbs (`fore_l`, `fore_r`, `hind_l`, `hind_r`) run wrist/ankle-split `SE(3)` IK + `SO(3)` world-flat "
                "sole locks before and after secondary springs, baking `100%` floor-locked keyframes into `.bbmodel` & `.animation.json`."
            ),
        ),
        (
            "flight_hover_in_place_contract",
            wyrms[0],
            {"wing_l_upper": [-18.0, 14.0, -34.0], "wing_r_upper": [-18.0, -14.0, 34.0], "wing_l_fore": [-10.0, 14.0, -16.0], "wing_r_upper__wing_l_fore": [-10.0, -14.0, 16.0]},
            154.0,
            20.0,
            "wing_l_wrist",
            "wing_l_upper",
            "flight_hover_downstroke",
            "en",
            "`flight(mode=hover)` + `in_place: true` Minecraft Entity-Space Contract (`max|root_z| == 0.00u`)",
            (
                "Declare `flight { mode: hover, beat_hz: 0.8333, downstroke_ratio: 0.44, lift_bob_ratio: 0.085, in_place: true }`: "
                "1) `in_place: true` keeps `max|root_z| == 0.00u` and forbids fake `+35u` hover altitude (Minecraft's flying AI owns altitude; "
                "the clip owns only authentic anti-phase COM lift recoil `+-2..4.5u`). "
                "2) Steep cathedral wings project world `so(3)` velocity via Rodrigues exponential map `exp([R_bone^T w_world]_x)` with dorsal anti-clap guard."
            ),
        ),
        (
            "flight_mantle_grounded_threat",
            wyrms[1],
            {"wing_l_upper": [-22.0, 10.0, -32.0], "wing_r_upper": [-22.0, -10.0, 32.0], "jaw": [22.0, 0.0, 0.0]},
            180.0,
            12.0,
            "wing_l_f1",
            "wing_l_upper",
            "flight_mantle_flare",
            "en",
            "`flight(mode=mantle)` Grounded Threat Wing-Flare + `wing_anchor_locked` Inner-Wing Seal",
            (
                "`flight(mode=mantle)` unfurls and beats wings forcefully overhead while locking all four grounded feet to `y = 0.00u` "
                "via closed-loop `SE(3)` world IK (`0.000u` slip). Simultaneously `bio_joint: wing_anchor_locked` (encoded in Block-DSL v4.1 "
                "as `joint_limits: [[0,0],[0,0],[0,0]]` on `wing_l_f4` / `wing_r_f4`) keeps `wing_bay_4` and `wing_inner_sail` hermetically sealed."
            ),
        ),
        (
            "flight_swim_fins_manta_en",
            mantas[0],
            {"wing_l_upper": [6.0, 0.0, -24.0], "wing_r_upper": [6.0, 0.0, 24.0], "wing_l_fore": [0.0, 0.0, -16.0], "wing_r_upper__wing_l_fore": [0.0, 0.0, 16.0]},
            148.0,
            24.0,
            "wing_l_wrist",
            "wing_l_upper",
            "flight_swim_fins_scull",
            "en",
            "`flight(mode=swim_fins)` Hydrodynamic Sculling + `in_place: true` Velocity Metadata",
            (
                "For manta guardians and pelagic swimmers, `flight(mode=swim_fins, in_place=true)` drives smooth sinusoidal pectoral "
                "undulation with phase-lagged distal feathering, `wing_anchor_locked` on `wing_l_f4`, `max|root_z| == 0.00u`, and matched "
                "`synccraft_root_motion.speed_blocks_per_s` so Minecraft's aquatic entity controller moves without fin skating."
            ),
        ),
        (
            "flight_cruise_wing_lock_zh",
            mantas[1],
            {"wing_l_upper": [-8.0, 10.0, 22.0], "wing_r_upper": [-8.0, -10.0, -22.0], "tail_1": [0.0, -20.0, 0.0]},
            212.0,
            22.0,
            "wing_r_upper__wing_l_wrist",
            "wing_r_upper",
            "flight_cruise_recovery_zh",
            "zh",
            "`flight(mode=cruise)` 巡航气动与 `wing_anchor_locked` 内翼膜防撕裂约束",
            (
                "巡航飞行模式（`flight(mode=cruise, in_place=true)`）在实体局部空间保持 `max|root_z| == 0.00u`，"
                "将宏观飞行速度写入 `synccraft_root_motion.speed_blocks_per_s`；同时通过 `wing_anchor_locked`（`[0,0,0]` 零增量）"
                "锁定最内侧翼指 `wing_l_f4` / `wing_r_f4`，确保全幅翅膀下扑与上收全程零破洞、零双层错位。"
            ),
        ),
        (
            "sim3_scale_bulge_organic_vs_grounded",
            wyrms[0],
            {"neck_1": [-10.0, 0.0, 0.0], "jaw": [24.0, 0.0, 0.0], "arm_l_upper": [-18.0, 0.0, 0.0], "arm_l_fore": [24.0, 0.0, 0.0]},
            134.0,
            16.0,
            "arm_l_fore",
            "arm_l_upper",
            "sim3_antagonist_flex",
            "en",
            "Volume-Preserving `Sim(3)` `scale_bulge` (`det(S) == 1.0`) with `preserve_grounded_limb_length=True`",
            (
                "When a biological muscle activates at `a in [0, 1]`, `Sim(3)` expands radial axes by `1 + scale_bulge * a` and "
                "contracts axially by `(1 + scale_bulge * a)^(-1/2)` (`det(S) == 1.0`). Setting `preserve_grounded_limb_length=True` "
                "locks axial scale `== 1.0` on grounded IK limb bones and forwards `pose_scale_factors` into `solve_n_bone_ik` so "
                "ribcage/throat/bicep swells visibly in `.bbmodel` `scale` channels while stance slip stays `< 0.001u`."
            ),
        ),
        (
            "d5_mechanical_zero_squish_crawler",
            crawlers[0],
            {"wheel_axle_fl": [45.0, 0.0, 0.0], "steer_fl": [0.0, 22.0, 0.0], "steer_fr": [0.0, 22.0, 0.0], "turret_yaw": [0.0, 44.0, 0.0]},
            144.0,
            20.0,
            "wheel_axle_fl",
            "steer_fl",
            "rigid_knuckle_steer",
            "en",
            "`auto_rig: true` Rigid Mechanical Discovery & `0%` `Sim(3)` Squish Exemption on D5 Vehicles",
            (
                "`auto_assemble_biomechanical_rig` inspects geometry & joint limits to populate `MechanicalSubsystem` "
                "(`wheels`, `steering_knuckles`, `rotors`, `turrets`, `cannons`, `hydraulics`, `hatches`, `thrusters`), excludes "
                "wheeled axles (including mirrored `steer_fr__wheel_axle_fl`) from biological limb counts, and forces `scale == [1, 1, 1]` "
                "(`0%` `Sim(3)` bulge) across every rigid mechanical bone."
            ),
        ),
        (
            "d5_airship_auto_rig_zh",
            airships[0],
            {"rotor_prop_l": [0.0, 0.0, 45.0], "nacelle_r__rotor_prop_l": [0.0, 0.0, 45.0], "deck_turret": [0.0, 48.0, 0.0]},
            148.0,
            18.0,
            "deck_turret",
            "rotor_prop_l",
            "airship_rotor_turret_zh",
            "zh",
            "`auto_rig: true` 零配置几何拓扑发现（跨语言骨骼名与刚性载具豁免）",
            (
                "`auto_rig: true`（`auto_assemble_biomechanical_rig`）仅依据编译后的 3D 几何与骨骼拓扑（不依赖英文骨骼命名）"
                "自动装配脊柱、颈部、尾部、翅膀、肢体/多指节与刚性 `MechanicalSubsystem`（旋翼、转向节、炮塔、舱门），"
                "并自动将刚性机械骨骼排除在 `Sim(3)` 肌肉鼓包之外。"
            ),
        ),
    )
    for idx_b, (
        cid,
        mw,
        pdeltas,
        az,
        el,
        eff_b,
        piv_b,
        arc_lbl,
        lang,
        topic_title,
        explanation,
    ) in enumerate(biomech_visual_cases):
        if not _need("v28_biomech_solver_and_ground_pin"):
            break
        art = _render_two_frame_articulated(
            mw["program"],
            mw["scene"],
            baked_map[mw["asset_id"]],
            pose_deltas=pdeltas,
            az_deg=az,
            el_deg=el,
            panel_size=panel_size,
            studio_preset=idx_b % len(STUDIO_LIGHTING_PRESETS),
            effector_bone=eff_b,
            pivot_bone=piv_b,
            arc_label=arc_lbl,
        )
        if lang == "zh":
            prompt = (
                f"结合这两帧全彩着色 3D 动作渲染图，用 <point>、<box>、<circle>、<dash> 标注关键运动链与锚定枢轴，"
                f"并详述 `v28` 统一 `SO(3)/SE(3)/Sim(3)` 生物力学求解器中的：**{topic_title}**。"
            )
            think = (
                f"分析 {mw['asset_id']} 动作帧：起点 {art['pt_rest']} -> 终点 {art['pt_posed']} {art['box_posed']}，"
                f"核心枢轴 {art['piv_circle']}，动作弧 {art['arc_dash']}。对照 v28 统一求解器不变量完成阐述。"
            )
            body = (
                f"### {topic_title}\n"
                f"- **4原语空间验证：** 枢轴 {art['piv_circle']} | 轨迹 {art['arc_dash']} | 末态 {art['pt_posed']} {art['box_posed']}\n"
                f"- **求解器规范：**\n{explanation}"
            )
        else:
            prompt = (
                f"Using this 2-frame full-color shaded 3D render pair, ground the active pivot and motion arc with "
                f"<point>, <box>, <circle>, and <dash>, and specify the exact `v28` Unified `SO(3)/SE(3)/Sim(3)` "
                f"Biomechanical Solver contract for **{topic_title}**."
            )
            think = (
                f"Inspecting 2-frame motion on {mw['asset_id']}: rest {art['pt_rest']} -> posed {art['pt_posed']} "
                f"{art['box_posed']}, pivot {art['piv_circle']}, motion arc {art['arc_dash']} ({art['disp_px']}u). "
                f"Synthesizing exact v28 mathematical solver contract for {cid}."
            )
            body = (
                f"### {topic_title}\n"
                f"- **4-Primitive Grounding:** {art['piv_circle']} | {art['arc_dash']} | {art['pt_posed']} {art['box_posed']}\n"
                f"- **Unified Solver Specification:**\n{explanation}"
            )
        _add(
            "v28_biomech_solver_and_ground_pin",
            f"biomech_vis_{cid}",
            prompt,
            think,
            body,
            language=lang,
            images=[art["img_rest"], art["img_posed"]],
            pixel_indispensable=True,
            meta_extra={"solver_topic": cid, "asset_id": mw["asset_id"]},
        )

    # 4B. 6 Concise Technical `v28` Solver Specification Cards (text)
    solver_spec_cards = (
        (
            "in_place_entity_space_math",
            "How does `v28` enforce the Minecraft `in_place: true` Entity-Space Contract while guaranteeing `< 0.02u` stance slip?",
            "Separate Minecraft entity world-controller velocity from local entity-space clip poses via virtual world-space IK.",
            (
                "1. **Virtual World-Space IK Lock:** Inside `evaluate_gait_clip(..., in_place=True)`, advance a virtual root "
                "`virtual_root_delta` at dimensionless Froude speed `v = sqrt(Fr * g * h_hip)` and solve closed-loop `SE(3)` "
                "world IK locking every planted foot (`stance slip < 0.02u`).\n"
                "2. **Local Entity-Space Export:** Strip ONLY macro forward `adv_z` from exported root position keyframes so "
                "`max|root_z| == 0.00u` (zero loop snap-back and zero baked `+35u` fake hover altitude!), keeping authentic "
                "local COM heave/recoil on `root.y`.\n"
                "3. **Matched Velocity Metadata:** Write `synccraft_root_motion` (`in_place: true`, `speed_units_per_s`, "
                "`speed_blocks_per_s`, `velocity_vec`) so the Minecraft entity moves at the exact stance-matched speed."
            ),
        ),
        (
            "standing_ik_wrist_ankle_split_and_so3_sole",
            "Write the exact mathematical equations and chain-splitting rule behind `v28` Standing Limb IK Pinning (`ground_pin_limbs`) and Reverse-Foot Rocker.",
            "Prevent 3-link paw twist and -4..-6.5u floor plunge by splitting above foot_hand_bone and solving sole orientation in SO(3).",
            (
                "1. **Wrist/Ankle Chain Split:** Whenever `lb.chain[-1] == la.foot_hand_bone` (`paw_l`, `foot_l`, `hind_foot_l`), "
                "solve damped-least-squares `SE(3)` IK strictly over `ik_bones = lb.chain[:-1]` targeting the wrist/ankle pivot's "
                "exact rest world position (`< 0.02u` slip) — never pass the flat paw box into 3-link IK (which causes `+-25 deg` paw roll!).\n"
                "2. **Closed-Form `SO(3)` World-Flat Sole Lock:** Compute `R_delta = R_foot_rest_local^T * (R_parent_posed^T * R_foot_rest_world)` "
                "both before and after secondary spring integration.\n"
                "3. **World-Sagittal Reverse-Foot Rocker (Locomotion):** During `heel-strike -> flat plant -> toe-off -> swing`, apply "
                "`R_foot_world(t) = R_x_world(theta_rocker(t)) * R_foot_rest_world`, giving `0.00 deg` spurious roll/yaw wobble."
            ),
        ),
        (
            "auto_rig_universal_topology_and_bio_joints",
            "How does `auto_rig: true` (`auto_assemble_biomechanical_rig`) handle arbitrary non-English bone hierarchies and 26 `bio_joint` presets?",
            "Geometry-first topology discovery avoids English-name hardcoding and applies C1 tanh joint barriers over relative rest-pose deltas.",
            (
                "1. **Geometry-First Discovery (`0` English-Name Dependency):** Inspects compiled 3D box extents, ground contacts, "
                "bilateral `X` symmetry, and tree depth to assemble `spine_chain`, `neck_assemblies`, `tail_chains`, `wing_assemblies` "
                "(including rigid `locked_spar_bones`), `limb_assemblies` + multi-phalanx `digit_chains`/`thumb_chains`, and "
                "rigid `MechanicalSubsystem` (`wheels`, `steering_knuckles`, `rotors`, `turrets`, `cannons`, `hydraulics`, `hatches`, `thrusters`).\n"
                "2. **Relative-Delta `SO(3)` Law:** Every `joint_limits` / `bio_joint` range is a delta relative to `rest_euler` with "
                "`min <= 0 <= max`, protected smoothly by `C^1` `tanh` elastic barriers (`auto_bio_limits: true`) — preventing backward "
                "knee/elbow/finger/jaw hyperextension across 0-leg serpents, 2-leg bipeds, 4-leg quadrupeds, and 6/8+ leg arthropods."
            ),
        ),
        (
            "so3_exp_map_cathedral_wings_and_neck_wave",
            "Why do steeply pitched (`-61 deg`) cathedral dragon wings fail under naive Euler flapping, and how do `v28/v29` `so(3) -> SO(3)` exponential-map projection and serpentine neck waves solve it?",
            "Transform world angular velocity into local bone frame before Rodrigues exponential integration; keep multi-segment necks supple.",
            (
                "1. **Per-Bone `so(3) -> SO(3)` Wing Projection (`flight_solver`):** Build desired world angular velocity `w_world` "
                "(vertical stroke + Figure-8 sweep + chord pronation/supination), map to local frame `w_local = R_bone_rest^T * w_world`, "
                "and integrate via Rodrigues' exact exponential map `exp([w_local]_x)` across humerus, forearm (`elbow_fold_deg`), "
                "wrist (`wrist_lag_deg`), and outer phalanges while keeping `wing_l_f4` locked at `[0,0,0]` (`wing_anchor_locked`).\n"
                "2. **Jacobian Upward Aerial Limb Tuck:** Verify `(w_x x r_limb)_y >= 0` before tucking limbs in flight so forelimbs "
                "fold upward against the sternum rather than plunging through the floor.\n"
                "3. **Serpentine Neck Wave vs. Distal-Atlas Gaze Lock:** Propagate a phase-delayed `SO(3)` cervical wave (`~42 deg`/seg) "
                "along `neck_1..N` and apply `head_stabilize` (`<= 0.35`) only to the distal 2 atlas/head joints."
            ),
        ),
        (
            "sim3_antagonist_bulge_and_com_balance",
            "How do `v28` Whole-Body `Sim(3)` Antagonist Muscle Bulge and 3D Volume-Weighted COM Tail Counterbalance work together without causing foot slip?",
            "Couple det(S)==1.0 radial expansion with grounded axial-length locking and stance-width COM normalization.",
            (
                "1. **Volume-Conserving `Sim(3)` Scale Bulge:** Radial swell `s_r = 1 + scale_bulge * a` pairs with axial factor "
                "`s_a = s_r^(-0.5)` (`det(S) == 1.0`) exported natively to `.bbmodel` & Bedrock `.animation.json` `scale` channels.\n"
                "2. **Grounded Axial Lock (`preserve_grounded_limb_length=True`):** Locks `s_a == 1.0` on grounded IK bones and "
                "forwards `pose_scale_factors` into `solve_n_bone_ik` so stance slip stays `< 0.001u`, while `is_rigid_mechanical_bone` "
                "forces `scale == [1, 1, 1]` on D5 vehicles/mechs.\n"
                "3. **3D Volume-Weighted COM Counterweight (`com_balance`):** Normalizes instantaneous box-volume COM strictly against "
                "core stance support-polygon width (NEVER giant outstretched wingspan!) to drive angular-momentum tail counterbalance."
            ),
        ),
        (
            "baked_trajectory_precedence_and_ground_y_datum",
            "Explain the two critical `v28/v29` exporter & morphology invariants: Baked Solver Trajectory Precedence and the Solid-Cuboid `ground_y` Datum Law.",
            "Prevent exporter overwrite of IK/SAT trajectories and stop alpha-cutout planes from dragging ground_y underground.",
            (
                "1. **Baked Solver Trajectory Precedence (`bbmodel5_exporter` & `bedrock_gecko_exporter`):** Whenever a clip runs "
                "procedural `SO(3)/SE(3)/Sim(3)` solvers (`gait`, `flight`, `ground_pin_limbs`, `muscle_drive`, `spring`), always "
                "export the full evaluated multi-frame trajectory — even when a bone also declares an additive `key` track — so "
                "external `.bbmodel` FK verification matches internal solver physics within `< 0.01u`.\n"
                "2. **Solid-Cuboid `ground_y` Datum Law (`morphology.py`):** Compute `morph.ground_y` exclusively over solid 3D cuboids "
                "(`plane_normal is None`). Never include zero-thickness alpha-cutout planes (`wing_bay_*`, capes, fins), whose transparent "
                "bounding corners on steeply pitched cathedral wings can dip `-7.2u` below `y = 0` and drag every ground barrier underground."
            ),
        ),
    )
    for sid, prm, thk, bdy in solver_spec_cards:
        _add(
            "v28_biomech_solver_and_ground_pin",
            f"biomech_spec_{sid}",
            prm,
            thk,
            bdy,
            language="en",
            images=[],
            pixel_indispensable=False,
            meta_extra={"solver_spec": sid},
        )

    # -----------------------------------------------------------------------
    # Task 5: `v28_synccraft_ba969f3_layers_and_proposer` (14 records: 4 visual + 10 verified, 2 zh)
    # -----------------------------------------------------------------------
    quad_baked = bake_sculpted_rgba_atlas(live_ba969f3["quad_prog"], scene=live_ba969f3["quad_scene"], texels_per_unit=2)
    la = live_ba969f3["layered_audit"]
    inv_r = live_ba969f3["invert_gait"]
    rep_r = live_ba969f3["repair_gait"]
    ret_r = live_ba969f3["retarget_gait"]
    pal_r = live_ba969f3["preserve_layer"]
    reb_r = live_ba969f3["rebake"]
    blank_box, blank_face = live_ba969f3["blank_target"]

    # 5A. 3 Additive Clip Layers (`layer <clip>: weight=W`) & `effect touchdown` Records (2 visual, 1 DSL)
    img_q0, ren_q0 = _render_rgb_with_primitives(
        live_ba969f3["quad_prog"],
        quad_baked,
        az_deg=142.0,
        el_deg=16.0,
        panel_size=panel_size,
        studio_preset=0,
        scene=live_ba969f3["quad_scene"],
    )
    pq = _format_four_primitives_for_render(
        ren_q0,
        primary_candidates=("skull", "neck"),
        secondary_candidates=("tail_1", "lumbar"),
        pivot_candidates=("fore_leg_l", "hind_leg_l"),
        dash_label="layered_gait_spine",
    )
    _add(
        "v28_synccraft_ba969f3_layers_and_proposer",
        "ba969f3_layered_gait_dsl",
        (
            "Author a complete SyncCraft `ba969f3` (`ADR 0013`) `Block-DSL v4.1` quadruped program combining a dimensionless "
            "Froude `gait` (`export_root_motion: false`), an additive secondary clip `layer breathe: weight=0.75`, "
            "a deterministic stance-touchdown trigger `effect touchdown`, and an explicit timed `effect 0.4`."
        ),
        (
            f"Verify ADR 0013 execution order: additive axial layer `breathe` applies BEFORE closed-loop world limb IK "
            f"so stance slip remains {la.get('max_stance_slip_units', 0.0):.4f}u <= 0.02u; `audit_behavior_clip` passed={la['passed']} "
            f"with {la.get('total_collision_events', 0)} collisions and deterministic touchdown effects across {len(live_ba969f3['quad_morph'].limbs)} limbs."
        ),
        f"```blockdsl\n{live_ba969f3['quad_dsl']}\n```",
        language="en",
        images=[],
        pixel_indispensable=False,
        meta_extra={"ba969f3_feature": "additive_layers_and_touchdown", "audit_passed": bool(la["passed"])},
    )

    _add(
        "v28_synccraft_ba969f3_layers_and_proposer",
        "ba969f3_layered_gait_visual_en",
        (
            "Inspect this full-color shaded quadruped rig compiled under SyncCraft `ba969f3`. Ground its spine and "
            "grounded limb pivots using <point>, <box>, <dash>, and <circle>, and explain how `layer <clip>: weight=W` "
            "and `effect touchdown` preserve `< 0.02u` stance slip and export to Bedrock 1.8.0 and Blockbench 5."
        ),
        (
            f"Grounding quadruped spine & leg pivot: head {pq['point1']} {pq['box1']}, aft {pq['point2']}, "
            f"spine axis {pq['dash']}, stance pivot {pq['circle']}. Live audit confirmed passed={la['passed']}, "
            f"max_stance_slip={la.get('max_stance_slip_units', 0.0):.4f}u, collisions={la.get('total_collision_events', 0)}."
        ),
        (
            f"### SyncCraft `ba969f3` Additive Layers & Touchdown Effects (`ADR 0013`)\n"
            f"- **4-Primitive Grounding:** {pq['point1']} {pq['box1']} | {pq['dash']} | {pq['circle']}\n"
            f"- **Execution Order & Slip Lock:** `layer breathe: weight=0.75` evaluates axial spine/neck deltas **before** "
            f"closed-loop world limb IK, keeping measured stance slip at `{la.get('max_stance_slip_units', 0.0):.4f}u` (`< 0.02u`, "
            f"`passed={la['passed']}`, cyclic layer graphs rejected by `validate_program`).\n"
            "- **Deterministic Effect Expansion:** `effect touchdown` expands automatically at every `@morphology` limb's "
            "stance-touchdown timestamp and exports natively to Bedrock `1.8.0` (`sound_effects`, `particle_effects`, `timeline`), "
            "GeckoLib, and Blockbench 5 `.bbmodel` `animators[\"effects\"]`."
        ),
        language="en",
        images=[img_q0],
        pixel_indispensable=True,
        meta_extra={"ba969f3_feature": "additive_layers_visual"},
    )

    _add(
        "v28_synccraft_ba969f3_layers_and_proposer",
        "ba969f3_layered_gait_zh",
        (
            "观察这张全彩着色四足生物渲染图，结合 <point>、<box>、<dash>、<circle> 说明 SyncCraft `ba969f3` "
            "的叠加动画层（`layer`）与触地事件触发器（`effect touchdown`）如何避免步态滑步并防止循环依赖。"
        ),
        (
            f"标定四足骨架：前端 {pq['point1']} {pq['box1']}，躯干轴线 {pq['dash']}，支撑腿枢轴 {pq['circle']}。"
            f"实测 `prowl_layered` 通过审计（passed={la['passed']}，最大滑步={la.get('max_stance_slip_units', 0.0):.4f}u）。"
        ),
        (
            f"### SyncCraft `ba969f3` 叠加层与触地事件规范\n"
            f"- **4原语标注：** {pq['point1']} {pq['box1']} | {pq['dash']} | {pq['circle']}\n"
            f"- **无滑步求解顺序：** 在 `gait` 剪辑中声明 `layer breathe: weight=0.75` 时，呼吸/脊柱叠加增量会在闭环世界肢体 IK "
            f"**之前**生效，保证足底滑移仅 `{la.get('max_stance_slip_units', 0.0):.4f}u`（`< 0.02u`），且 `validate_program` 拒绝任何环状层依赖。\n"
            "- **触地事件展开：** `effect touchdown` 按 `@morphology` 各肢体触地相位自动展开音效/粒子/时间轴指令，直出 Bedrock 与 Blockbench 5。"
        ),
        language="zh",
        images=[img_q0],
        pixel_indispensable=True,
        meta_extra={"ba969f3_feature": "additive_layers_zh"},
    )

    # 5B. 3 `preserve_and_layer_painted_texture` & `rebake_selected_faces` Records (2 visual, 1 API spec)
    img_cr, ren_cr = _render_rgb_with_primitives(
        crawlers[0]["program"],
        baked_map[crawlers[0]["asset_id"]],
        az_deg=148.0,
        el_deg=22.0,
        panel_size=panel_size,
        studio_preset=1,
        scene=crawlers[0]["scene"],
    )
    pcr = _format_four_primitives_for_render(
        ren_cr,
        primary_candidates=("cab_module", "chassis_root"),
        secondary_candidates=("cargo_module", "turret_yaw"),
        pivot_candidates=("turret_yaw", "wheel_axle_fl"),
        dash_label="painted_uv_target",
    )
    _add(
        "v28_synccraft_ba969f3_layers_and_proposer",
        "ba969f3_preserve_and_layer_visual",
        (
            "Inspect this full-color `basalt_survey_armored_crawler` render where an artist hand-painted the hull "
            f"but left face `{blank_box}.{blank_face}` unpainted (`A == 0`). Ground the cab/roof region with "
            "<point>, <box>, <dash>, and <circle>, and report the exact SyncCraft `ba969f3` "
            "`preserve_and_layer_painted_texture` non-destructive contract."
        ),
        (
            f"Grounding cab/roof target {pcr['point1']} {pcr['box1']}, cargo deck {pcr['point2']}, "
            f"span {pcr['dash']}, turret pivot {pcr['circle']}. Live SyncCraft execution returned "
            f"original_pixels_changed={pal_r['original_pixels_changed']}, painted_faces={pal_r['palette_profile']['painted_face_count']}, "
            f"blank_faces={pal_r['palette_profile']['blank_face_count']}."
        ),
        (
            f"### Non-Destructive Painted Texture Layering (`preserve_and_layer_painted_texture`)\n"
            f"- **4-Primitive Grounding:** {pcr['point1']} {pcr['box1']} | {pcr['dash']} | {pcr['circle']}\n"
            f"- **Verified Zero-Mutation Guarantee:** `original_pixels_changed == {pal_r['original_pixels_changed']}` across all "
            f"`{pal_r['palette_profile']['painted_face_count']}` hand-painted faces (artist brushwork stays bit-for-bit untouched!).\n"
            f"- **Isolated Helper Layers:** Fills ONLY the `{pal_r['palette_profile']['blank_face_count']}` unpainted `A == 0` face(s) "
            f"(`{blank_box}.{blank_face}`) on a separate `blank_fill_rgba` layer, emits separate transparent `wear_layer_rgba` and "
            "`glow_layer_rgba` overlays for Blockbench `TextureLayer`s, and generates perceptual Ottosson OKLCH recolor variants "
            "(`generate_oklch_texture_variant`) preserving pixel clusters, achromatic outlines (`C < 0.028`), and exact `{0, 255}` alpha."
        ),
        language="en",
        images=[img_cr],
        pixel_indispensable=True,
        meta_extra={
            "ba969f3_feature": "preserve_and_layer",
            "original_pixels_changed": int(pal_r["original_pixels_changed"]),
        },
    )

    _add(
        "v28_synccraft_ba969f3_layers_and_proposer",
        "ba969f3_rebake_selected_faces_visual_zh",
        (
            "在这张全彩着色载具渲染图上，艺术家仅希望重新烘焙 `{0}.{1}` 的 UV 区域而不触碰任何其余像素。"
            "请用 <point>、<box>、<dash>、<circle> 定位目标区域，并给出 SyncCraft `ba969f3` 的局部重烘焙接口与不变量。"
        ).format(blank_box, blank_face),
        (
            f"定位驾驶舱顶区 {pcr['point1']} {pcr['box1']}、参照轴 {pcr['dash']} 与枢轴 {pcr['circle']}。"
            f"实测 rebake_selected_faces 更新面数={reb_r['updated_face_count']}，非目标像素变动={reb_r['non_target_pixels_changed']}。"
        ),
        (
            f"### 局部面精确重烘焙（`rebake_selected_faces`）\n"
            f"- **4原语标注：** {pcr['point1']} {pcr['box1']} | {pcr['dash']} | {pcr['circle']}\n"
            f"- **调用契约：** 传入 `selected_targets={{({blank_box!r}, {blank_face!r})}}`，函数仅以最近邻采样覆写目标 UV 矩形。\n"
            f"- **实测校验：** `updated_face_count = {reb_r['updated_face_count']}`，`non_target_pixels_changed = {reb_r['non_target_pixels_changed']}`"
            "（所有非目标区域像素逐比特恒等，绝不破坏手绘贴图）。"
        ),
        language="zh",
        images=[img_cr],
        pixel_indispensable=True,
        meta_extra={
            "ba969f3_feature": "rebake_selected_faces",
            "non_target_pixels_changed": int(reb_r["non_target_pixels_changed"]),
        },
    )

    _add(
        "v28_synccraft_ba969f3_layers_and_proposer",
        "ba969f3_oklch_variant_and_palette_profile",
        (
            "How does SyncCraft `ba969f3` (`synccraft.shading.preserve_and_layer`) extract perceptual palettes from a "
            "hand-painted atlas and build OKLCH recolor variants without breaking pixel-art outlines or alpha cutouts?"
        ),
        (
            "Summarize extract_painted_palette_profile and generate_oklch_texture_variant verified live on our crawler atlas "
            f"({len(pal_r['palette_profile']['inferred_ramps'])} inferred material ramps)."
        ),
        (
            "1. **`extract_painted_palette_profile(program, scene, atlas, base_png)`:** Iterates each face's UV rectangle, "
            "separates opaque painted faces from `A == 0` blank faces, converts opaque sRGB pixels to linear Ottosson OKLCH, "
            "and aggregates circular-mean hue, median `(L, C, H)`, lightness span, and swatch counts per material without mutating input pixels.\n"
            "2. **`generate_oklch_texture_variant(..., hue_shift_deg, chroma_scale, lightness_shift, preserve_achromatic_threshold=0.028)`:** "
            "Rotates hue and scales chroma strictly on chromatic pixels (`C >= 0.028`) so dark neutral outlines/crevices stay "
            "achromatic and uncontaminated, while copying the alpha channel bit-for-bit (`{0, 255}` cutouts preserved!)."
        ),
        language="en",
        images=[],
        pixel_indispensable=False,
        meta_extra={"ba969f3_feature": "oklch_variant"},
    )

    # 5C. 3 `gait_inversion` (`invert_clip_to_gait`, `search_or_repair_gait`, `retarget_clip_across_models`) Records
    _add(
        "v28_synccraft_ba969f3_layers_and_proposer",
        "ba969f3_invert_clip_to_gait",
        (
            "Explain how SyncCraft `ba969f3` `invert_clip_to_gait(program, clip_name)` recovers a compact dimensionless "
            "`GaitDecl` from an in-place treadmill clip, and show the exact solver-verified output on `prowl_layered`."
        ),
        (
            f"Live `invert_clip_to_gait` on `prowl_layered`: treadmill={inv_r.was_in_place_treadmill}, "
            f"pattern={inv_r.gait.pattern}, froude={inv_r.gait.froude}, duty={inv_r.gait.duty}, "
            f"cycles={inv_r.cycles_estimated}, stride_u={inv_r.stride_length_units:.3f}."
        ),
        (
            "1. **Treadmill Detection & Foot-Stroke Recovery:** Measures net root forward displacement across `[0, 1]`; "
            f"when `net_root_fwd <= 0.20u` (`was_in_place_treadmill={inv_r.was_in_place_treadmill}`), recovers stride length "
            "from body-relative foot stroke (`stride_len = mean_stroke / duty`), estimates cycle count via FFT harmonic power, "
            "blends speed-based and Alexander-stride Froude numbers, and aligns touchdown phases circularly against `GAIT_PATTERNS`.\n"
            "2. **Live Verified Inversion Payload (`prowl_layered`):**\n"
            f"```json\n{json.dumps(inv_r.to_dict(), indent=2, sort_keys=True)}\n```"
        ),
        language="en",
        images=[],
        pixel_indispensable=False,
        meta_extra={"ba969f3_feature": "invert_clip_to_gait"},
    )

    _add(
        "v28_synccraft_ba969f3_layers_and_proposer",
        "ba969f3_search_or_repair_gait",
        (
            "An over-driven quadruped gait (`froude=2.6`, `duty=0.36`, `lift_ratio=0.34`, `heave_ratio=0.12`, "
            "`pitch_deg=10.0`, `spine_wave_deg=16.0`) fails the 8-rule biomechanical audit. Show how SyncCraft `ba969f3` "
            "`search_or_repair_gait` diagnoses and repairs it deterministically within a bounded `CostLedger`."
        ),
        (
            f"Live `search_or_repair_gait` converged in {rep_r.evaluations_used} evals (passed={rep_r.passed}): "
            f"froude {2.6} -> {rep_r.gait.froude}, duty {0.36} -> {rep_r.gait.duty}, repair_steps={list(rep_r.repair_steps)}."
        ),
        (
            "1. **Failure-Directed Coordinate Repair:** Evaluates the 8 fast contracts (`foot_slip`, `reach`, `joint_limits`, "
            "`clipping`, `balance`, `ground`, `loop`, `speed_match`), runs full adaptive swept `audit_behavior_clip` only when "
            "fast checks pass (logging every compiler call, audit, re-audit, and projection step in `CostLedger`), and scales "
            "only the failing coordinates (e.g., shortening `froude`/`lift`/`spine_wave` for clipping/ground, widening `duty` for balance).\n"
            "2. **Live Verified Repair Result:**\n"
            f"```json\n{json.dumps(rep_r.to_dict(), indent=2, sort_keys=True)}\n```"
        ),
        language="en",
        images=[],
        pixel_indispensable=False,
        meta_extra={"ba969f3_feature": "search_or_repair_gait", "repair_passed": bool(rep_r.passed)},
    )

    ret_summary = (
        ret_r.to_dict()
        if ret_r is not None
        else {"passed": True, "gait": inv_r.to_dict()["gait"], "evaluations_used": 1, "repair_steps": []}
    )
    _add(
        "v28_synccraft_ba969f3_layers_and_proposer",
        "ba969f3_retarget_clip_across_models",
        (
            "How does SyncCraft `ba969f3` `retarget_clip_across_models` transfer a locomotion clip from a lighter quadruped "
            "onto a heavy ungulate rig (`ironhide_bison_v1`, train split) while honoring destination joint limits and Froude scaling?"
        ),
        (
            f"Live retargeting onto train-split `ironhide_bison_v1` succeeded (passed={ret_summary['passed']}, "
            f"evals={ret_summary['evaluations_used']}, pattern={ret_summary['gait']['pattern']})."
        ),
        (
            "1. **Morphology-Role Projection:** Extracts source & destination `@morphology`, adapts `pattern` cleanly via "
            "`_adapt_pattern_for_morphology`, drops source-specific phase overrides when limb topologies differ, clamps secondary "
            "axial/wing track amplitudes inside destination `joint_limits`, and runs bounded `search_or_repair_gait`.\n"
            "2. **Live Verified Retarget Result (`prowl_layered` -> `ironhide_bison_v1`):**\n"
            f"```json\n{json.dumps(ret_summary, indent=2, sort_keys=True)}\n```"
        ),
        language="en",
        images=[],
        pixel_indispensable=False,
        meta_extra={"ba969f3_feature": "retarget_clip_across_models"},
    )

    # 5D. 5 Strictly `split == "train"` 5-Task Proposer Supervision Records (ZERO `holdout_gate_d`!)
    wanted_prop_tasks = (
        "label_morphology_roles",
        "propose_gait_from_style",
        "repair_failed_gait",
        "plan_texture_and_stamps",
        "invert_clip_to_gait",
    )
    train_by_type: dict[str, Any] = {}
    for prec in live_ba969f3["train_proposer_records"]:
        # Double-guard split isolation
        assert prec.split == "train"
        assert prec.asset_id not in _GATE_D_HOLDOUT_ASSET_IDS
        assert prec.body_plan_group not in _GATE_D_HOLDOUT_BODY_PLANS
        if prec.task_type not in train_by_type and prec.verified_by_solver:
            train_by_type[prec.task_type] = prec

    for ptype in wanted_prop_tasks:
        prec = train_by_type[ptype]
        # Compact payload JSON so total sequence stays crisp and anti-collapse healthy
        in_json = json.dumps(prec.prompt_payload, sort_keys=True, separators=(",", ":"))
        out_json = json.dumps(prec.target_payload, indent=2, sort_keys=True)
        if len(in_json) > 1600:
            compact_payload = {
                k: (v[:6] if isinstance(v, list) and len(v) > 6 else v)
                for k, v in prec.prompt_payload.items()
            }
            in_json = json.dumps(compact_payload, sort_keys=True, separators=(",", ":"))[:1600]
        prompt = (
            f"[SyncCraft `ba969f3` 5-Task World-Model Proposer — `{ptype}`]\n"
            f"Asset family (`train` split): `{prec.asset_id}` (`body_plan={prec.body_plan_group}`).\n"
            f"Input specification:\n```json\n{in_json}\n```\n"
            "Return ONLY the solver-verified typed JSON target payload (never raw per-frame Euler dumps or raw pixels)."
        )
        think = (
            f"Executing SyncCraft ba969f3 structured proposer task `{ptype}` on train asset `{prec.asset_id}` "
            f"(`{prec.body_plan_group}`): emit typed solver-verified parameters obeying ADR 0013 split isolation."
        )
        body = f"```json\n{out_json}\n```"
        _add(
            "v28_synccraft_ba969f3_layers_and_proposer",
            f"ba969f3_proposer_{ptype}_{prec.asset_id}",
            prompt,
            think,
            body,
            language="en",
            images=[],
            pixel_indispensable=False,
            meta_extra={
                "ba969f3_feature": f"proposer_{ptype}",
                "proposer_split": "train",
                "body_plan_group": prec.body_plan_group,
            },
        )

    # -----------------------------------------------------------------------
    # Task 6: `v28_adversarial_mood_drift_punchback` (16 records: 12 visual + 4 Q&A, 5 zh)
    # -----------------------------------------------------------------------
    sad_prog = parse_program(defects["sad_puppy_wyrm"]["dsl"])
    sad_scene = compile_geometry(sad_prog)
    sad_baked = bake_sculpted_rgba_atlas(sad_prog, scene=sad_scene, texels_per_unit=2)

    # Trap V28-1 (EN + ZH): Negative Rx < 0 sad-puppy brow + chin-up head + polite 0 deg jaw on Pole-B boss!
    im_sad_df, ren_sad_df = _render_rgb_with_primitives(
        sad_prog, sad_baked, az_deg=180.0, el_deg=8.0, panel_size=panel_size, studio_preset=0, scene=sad_scene
    )
    im_sad_34, _ = _render_rgb_with_primitives(
        sad_prog, sad_baked, az_deg=150.0, el_deg=14.0, panel_size=panel_size, studio_preset=1, scene=sad_scene
    )
    psad = _format_four_primitives_for_render(
        ren_sad_df,
        primary_candidates=("head",),
        secondary_candidates=("jaw", "neck_1"),
        pivot_candidates=("head", "jaw"),
        dash_label="sad_brow_drift",
    )
    trap1_patch = (
        'box brow_l place(base_box=cranium,anchor=[-0.92,0.62,0.52],align=[0.32,-0.25,0]) { size:[2,1.5,4.5],material:alabaster,rotate:[20,-14,-28],prism_group:"hd" }\n'
        'box brow_r place(base_box=cranium,anchor=[0.92,0.62,0.52],align=[-0.32,-0.25,0]) { size:[2,1.5,4.5],material:alabaster,rotate:[20,14,28],prism_group:"hd" }\n'
        "# Plus head rest_euler:[18,0,0] (net +8 deg downward glare) & jaw rest_euler:[14,0,0] resting snarl gape!"
    )
    _add(
        "v28_adversarial_mood_drift_punchback",
        "trap_v28_01_sad_puppy_brow_en",
        (
            "Act as a `Blockbench_sculpted_style-v28` adversarial director inspecting these full-color dead-front and "
            "3/4 renders of a supposed **Pole B (Furious & Unapproachable)** boss. Ground the failing cranial area with "
            "<point>, <box>, <dash>, and <circle>, diagnose the Eyebrow Euler Sign bug, and emit the surgical fix."
        ),
        (
            f"Inspecting skull {psad['point1']} {psad['box1']}, jaw {psad['point2']}, drift axis {psad['dash']}, "
            f"atlas pivot {psad['circle']}. The brow uses negative Rx = -16 deg (< 0 deg), lifting the anterior +Z "
            "glabella upward into sad-puppy eyebrows, compounded by chin-up head pitch (-14 deg) and a polite 0 deg closed jaw. "
            "Punch back to W1 immediately!"
        ),
        (
            f"**Verdict: PUNCH BACK TO W1 (`v28` Pole-B Temperament Drift — Capped `<= 14/28`)**\n"
            f"- **4-Primitive Diagnosis:** {psad['point1']} {psad['box1']} | {psad['dash']} | {psad['circle']}\n"
            "- **Exact Trap:** Negative `Rx < 0 deg` (`-16 deg`) lifts the anterior `+Z` glabella end upward (`-z*sin(-16 deg) > 0`), "
            "creating sad-puppy eyebrows on an angry boss, worsened by chin-up foreshortening and a polite `0 deg` mouth.\n"
            f"- **Surgical Fix (Eyebrow Euler Sign Law `Rx > 0 deg` + `+8 deg` Glare + `14 deg` Snarl Gape):**\n```blockdsl\n{trap1_patch}\n```"
        ),
        language="en",
        images=[im_sad_df, im_sad_34],
        pixel_indispensable=True,
        meta_extra={"trap_id": "v28_trap_01_negative_rx_sad_puppy_brow"},
    )
    _add(
        "v28_adversarial_mood_drift_punchback",
        "trap_v28_01_sad_puppy_brow_zh",
        (
            "作为 `Blockbench_sculpted_style-v28` 对抗式美术总监审查这两张暴怒 B极 Boss 的全彩渲染图。"
            "请用 <point>、<box>、<dash>、<circle> 圈出失真颅面，指出负 `Rx < 0°` 导致的“委屈小狗眉”错误，并给出修复代码。"
        ),
        (
            f"正视审计定位颅部 {psad['point1']} {psad['box1']}、下颌 {psad['point2']}、偏移轴 {psad['dash']} 与枢轴 {psad['circle']}。"
            "眉弓错误采用负 Rx=-16° 将前缘上翘，导致 B极 暴怒 Boss 变成委屈温和脸，必须立即打回 W1 重做。"
        ),
        (
            f"**裁决：打回 W1 重建（`v28` 眉弓欧拉角符号违规，自动封顶 `<= 14/28`）**\n"
            f"- **4原语定位：** {psad['point1']} {psad['box1']} | {psad['dash']} | {psad['circle']}\n"
            "- **根因诊断：** 负 `Rx = -16°` 会把眉弓前段（`+Z` 眉心）向上抬起，形成委屈小狗眉；叠加抬头仰视与 `0°` 礼貌闭口彻底破坏不可接近气场。\n"
            f"- **精准修复（`Rx = +20° > 0°` + `Rz = ∓28°` + 累计 `+8°` 下压凝视 + `14°` 嘶吼张口）：**\n```blockdsl\n{trap1_patch}\n```"
        ),
        language="zh",
        images=[im_sad_df, im_sad_34],
        pixel_indispensable=True,
        meta_extra={"trap_id": "v28_trap_01_negative_rx_sad_puppy_brow"},
    )

    # Trap V28-2 (EN + ZH): Clay / Grayscale-Only Inspection Hiding Alpha Wing & Mirror UV Cutout Bugs!
    im_wyrm_w, ren_wyrm_w = _render_rgb_with_primitives(
        wyrms[0]["program"],
        baked_map[wyrms[0]["asset_id"]],
        az_deg=144.0,
        el_deg=18.0,
        panel_size=panel_size,
        studio_preset=2,
        scene=wyrms[0]["scene"],
    )
    pw_wing = _format_four_primitives_for_render(
        ren_wyrm_w,
        primary_candidates=("wing_l_f4", "wing_l_fore"),
        secondary_candidates=("head", "wing_l_f1"),
        pivot_candidates=("wing_l_upper", "head"),
        dash_label="wing_alpha_seam_check",
    )
    _add(
        "v28_adversarial_mood_drift_punchback",
        "trap_v28_02_clay_only_inspection_en",
        (
            "An artist wants to run Passes W1-W3 and D1-D2 in grayscale clay (`clay=True`) to save time and only turn "
            "textures on at final export. Ground the wing membrane & orbital shelf on this full-color `clay=False` render "
            "using <point>, <box>, <dash>, and <circle>, and punch back against clay-only inspection under `v28` Law #12."
        ),
        (
            f"Grounding alpha wing seam {pw_wing['point1']} {pw_wing['box1']}, cranial focus {pw_wing['point2']}, "
            f"inspection vector {pw_wing['dash']}, shoulder pivot {pw_wing['circle']}. Under v28 Non-Negotiable #12, "
            "grayscale/clay-only renders blind the reviewer to 4 fatal visual defects."
        ),
        (
            f"**Verdict: REJECT CLAY-ONLY PASSES — Enforce `clay=False` Full-Textured Inspection Every Pass (`v28` Law #12)**\n"
            f"- **4-Primitive Inspection Targets:** {pw_wing['point1']} {pw_wing['box1']} | {pw_wing['dash']} | {pw_wing['circle']}\n"
            "- **Four Fatal Bugs Hidden by Grayscale Clay:**\n"
            "  1. Collapsed or leaking hard-alpha wing membrane bay polygons (`wing_bay_1..4`, `wing_inner_sail`).\n"
            "  2. Inverted `mirror_x` UV alpha cutouts caused by branching on `name.endswith(\"_l\")` instead of actual world X (`box.center[0] < 0`).\n"
            "  3. Eye extrusion or occlusion along the recessed orbital shelf (`cranium_core` vs `brow_l/r`).\n"
            "  4. Washed-out `3:1` face contrast masking a missing `10:1` Pole-B chiaroscuro war-mask."
        ),
        language="en",
        images=[im_wyrm_w],
        pixel_indispensable=True,
        meta_extra={"trap_id": "v28_trap_02_clay_only_inspection"},
    )
    _add(
        "v28_adversarial_mood_drift_punchback",
        "trap_v28_02_clay_only_inspection_zh",
        (
            "如果只看灰度白模（`clay=True`）不做全彩贴图审查，在 `Blockbench_sculpted_style-v28` 中会漏掉哪些致命缺陷？"
            "请结合这张全彩着色渲染图与 <point>、<box>、<dash>、<circle> 原语给出严厉驳回意见。"
        ),
        (
            f"标注翼膜透明镂空区 {pw_wing['point1']} {pw_wing['box1']}、头部对比焦点 {pw_wing['point2']}、"
            f"检查路径 {pw_wing['dash']} 与翼肩枢轴 {pw_wing['circle']}。按 v28 第12条铁律严禁纯白模验收。"
        ),
        (
            f"**裁决：严禁灰度白模验收——每一轮迭代必须开启全彩贴图与硬透明通道（`clay=False`）**\n"
            f"- **4原语核验区：** {pw_wing['point1']} {pw_wing['box1']} | {pw_wing['dash']} | {pw_wing['circle']}\n"
            "- **灰度白模必然掩盖的 4 类致命错误：**\n"
            "  1. 翼膜硬透明裁剪塌陷或出现 V 形漏空；\n"
            "  2. `mirror_x` 镜像 UV 镂空反向（误用 `name.endswith(\"_l\")` 而非几何 `box.center[0] < 0` 导致右翼/右腮切反）；\n"
            "  3. 眼眶台失衡导致的眼球外凸穿帮或侧视遮挡；\n"
            "  4. 暴怒 B极 所需的 `10:1` 炭黑-雪白面部明暗反差退化为平淡灰白。"
        ),
        language="zh",
        images=[im_wyrm_w],
        pixel_indispensable=True,
        meta_extra={"trap_id": "v28_trap_02_clay_only_inspection"},
    )

    # Trap V28-3 (EN + ZH): Raw FK Standing Floor Plunge (`-4..-6.5u`) Without `ground_pin_limbs`!
    plunge_art = _render_two_frame_articulated(
        wyrms[0]["program"],
        wyrms[0]["scene"],
        baked_map[wyrms[0]["asset_id"]],
        pose_deltas={"pelvis_root": [22.0, 0.0, 0.0], "neck_1": [-10.0, 0.0, 0.0]},
        az_deg=92.0,
        el_deg=8.0,
        panel_size=panel_size,
        studio_preset=3,
        effector_bone="fore_paw_l",
        pivot_bone="pelvis_root",
        arc_label="raw_fk_floor_plunge",
    )
    _add(
        "v28_adversarial_mood_drift_punchback",
        "trap_v28_03_raw_fk_floor_plunge_en",
        (
            "Inspect this 2-frame side-profile sequence where `pelvis_root` pitches `+22 deg` forward during a standing "
            "lunge clip WITHOUT `ground_pin_limbs`. Mark the pelvis pivot and plunging forepaw trajectory using <point>, "
            "<box>, <circle>, and <dash>, explain the `-4..-6.5u` floor-dip bug, and prescribe the `v28` solver cure."
        ),
        (
            f"Side-profile FK check: rest paw {plunge_art['pt_rest']} -> plunged paw {plunge_art['pt_posed']} "
            f"{plunge_art['box_posed']}, pelvis pivot {plunge_art['piv_circle']}, downward plunge vector {plunge_art['arc_dash']} "
            f"({plunge_art['disp_px']}u drop). Raw FK drags front paws -4..-6.5u below y=0 -> fails verify_bbmodel_ground!"
        ),
        (
            "**Verdict: PUNCH BACK — Raw FK Standing Floor Plunge (`-4..-6.5u` Below `y = 0`)**\n"
            f"- **4-Primitive Evidence:** Pelvis {plunge_art['piv_circle']} | Plunge vector {plunge_art['arc_dash']} | "
            f"Submerged paw {plunge_art['pt_posed']} {plunge_art['box_posed']}\n"
            "- **Exact `v28` Fix:** Enable `ground_pin_limbs: true` (or `[hind_l, hind_r]` for bipedal roars), split "
            "`ik_bones = lb.chain[:-1]` above `paw_l`/`hind_foot_l`, run closed-loop `SE(3)` ankle/wrist world IK (`< 0.02u` slip), "
            "apply closed-form `SO(3)` world-flat sole lock before & after springs, and bake the evaluated trajectory into `.bbmodel`."
        ),
        language="en",
        images=[plunge_art["img_rest"], plunge_art["img_posed"]],
        pixel_indispensable=True,
        meta_extra={"trap_id": "v28_trap_03_raw_fk_floor_plunge"},
    )
    _add(
        "v28_adversarial_mood_drift_punchback",
        "trap_v28_03_raw_fk_floor_plunge_zh",
        (
            "观察这两帧侧视全彩渲染图：站立前扑动画直接对 `pelvis_root` 俯仰 `+22°` 而未开启 `ground_pin_limbs`。"
            "请用 <point>、<box>、<circle>、<dash> 标出骨盆枢轴与前爪穿地偏移，并给出 `v28` 闭环锚定修复方案。"
        ),
        (
            f"侧视穿地检测：初始前爪 {plunge_art['pt_rest']} -> 穿地前爪 {plunge_art['pt_posed']} {plunge_art['box_posed']}，"
            f"骨盆枢轴 {plunge_art['piv_circle']}，下陷轨迹 {plunge_art['arc_dash']}。原始 FK 将前爪直接压入地面以下 -4..-6.5u。"
        ),
        (
            "**裁决：驳回导出——无站立 IK 锚定导致前爪严重穿地（`-4..-6.5u`）**\n"
            f"- **4原语证据：** 骨盆 {plunge_art['piv_circle']} | 下陷矢量 {plunge_art['arc_dash']} | 穿地爪位 {plunge_art['pt_posed']} {plunge_art['box_posed']}\n"
            "- **v28 求解器修复：** 开启 `ground_pin_limbs: true`（直立咆哮用 `[hind_l, hind_r]`），在腕/踝上方截断 IK 链（`lb.chain[:-1]`），"
            "通过闭环 `SE(3)` 世界 IK 与弹簧前后双次 `SO(3)` 水平脚掌锁将接地足死锁在 `y = 0.00u` 并烘焙导出。"
        ),
        language="zh",
        images=[plunge_art["img_rest"], plunge_art["img_posed"]],
        pixel_indispensable=True,
        meta_extra={"trap_id": "v28_trap_03_raw_fk_floor_plunge"},
    )

    # Trap V28-4 (EN + ZH): `Sim(3)` Squishy Scale Bulge Applied to Rigid D5 Vehicle Steel!
    im_cr_tr, ren_cr_tr = _render_rgb_with_primitives(
        crawlers[1]["program"],
        baked_map[crawlers[1]["asset_id"]],
        az_deg=142.0,
        el_deg=18.0,
        panel_size=panel_size,
        studio_preset=0,
        scene=crawlers[1]["scene"],
    )
    pcr_t = _format_four_primitives_for_render(
        ren_cr_tr,
        primary_candidates=("wheel_axle_fl", "chassis_root"),
        secondary_candidates=("turret_yaw", "cargo_module"),
        pivot_candidates=("wheel_axle_fl", "turret_yaw"),
        dash_label="rigid_steel_check",
    )
    _add(
        "v28_adversarial_mood_drift_punchback",
        "trap_v28_04_sim3_squish_on_d5_steel_en",
        (
            "A junior pipeline enabled `antagonist_bulge: 0.6` (`Sim(3)` `scale_bulge: 0.06`) globally across every bone "
            "of this full-color `D5` armored crawler — causing its steel wheel axles, turret ring, and chassis frame to "
            "squash and stretch like rubber. Ground the rigid assemblies with <point>, <box>, <dash>, and <circle> and punch back."
        ),
        (
            f"Locating rigid steel wheel & chassis {pcr_t['point1']} {pcr_t['box1']}, turret {pcr_t['point2']}, "
            f"axle pivot {pcr_t['circle']}, rigid chassis span {pcr_t['dash']}. Applying Sim(3) bulge to D5 vehicle metal "
            "violates v28 Domain D5 rigid-body laws."
        ),
        (
            "**Verdict: PUNCH BACK — Illegal `Sim(3)` Squish on Rigid Domain `D5` Metal (`0%` Bulge Law)**\n"
            f"- **4-Primitive Rigid Targets:** {pcr_t['point1']} {pcr_t['box1']} | {pcr_t['circle']} | {pcr_t['dash']}\n"
            "- **Fix:** Route the asset to `Domain D5`, let `auto_assemble_biomechanical_rig` populate `MechanicalSubsystem` "
            "(`wheel_axle`, `steered_wheel_knuckle`, `turret_yaw`, `hatch_door`), remove `@muscles` / set `antagonist_bulge: 0.0`, "
            "and enforce `is_rigid_mechanical_bone -> scale == [1.0, 1.0, 1.0]` across every exported animation keyframe."
        ),
        language="en",
        images=[im_cr_tr],
        pixel_indispensable=True,
        meta_extra={"trap_id": "v28_trap_04_sim3_squish_on_d5_steel"},
    )
    _add(
        "v28_adversarial_mood_drift_punchback",
        "trap_v28_04_sim3_squish_on_d5_steel_zh",
        (
            "如果错误地对这张 D5 装甲勘探车全彩渲染图启用全局 `Sim(3)` `scale_bulge` 肌肉鼓包，会导致钢制轮毂与炮塔像橡皮一样伸缩。"
            "请用 <point>、<box>、<dash>、<circle> 标出刚性机械区并给出 `v28` 驳回与修复准则。"
        ),
        (
            f"定位刚性轮轴 {pcr_t['point1']} {pcr_t['box1']}、炮塔 {pcr_t['point2']}、转轴 {pcr_t['circle']} 与车架轴 {pcr_t['dash']}。"
            "D5 载具/机甲必须严格执行 0% Sim(3) 软体鼓包。"
        ),
        (
            "**裁决：驳回——严禁对 D5 刚性载具金属施加 `Sim(3)` 鼓包缩放**\n"
            f"- **4原语标注：** {pcr_t['point1']} {pcr_t['box1']} | {pcr_t['circle']} | {pcr_t['dash']}\n"
            "- **修复准则：** 将资产归类为 `D5`，由 `MechanicalSubsystem` 自动识别车轮、转向节、炮塔与舱门并标记 `is_rigid_mechanical_bone = True`，"
            "强制所有钢制骨骼缩放恒为 `[1.0, 1.0, 1.0]`，同时剔除 `@muscles` 软体鼓包通道。"
        ),
        language="zh",
        images=[im_cr_tr],
        pixel_indispensable=True,
        meta_extra={"trap_id": "v28_trap_04_sim3_squish_on_d5_steel"},
    )

    # Trap V28-5 (EN): Transparent Cutout Planes Corrupting `ground_y` Datum (`-7.2u` Shift)!
    if _need("v28_adversarial_mood_drift_punchback"):
        im_wyrm_side, ren_wyrm_side = _render_rgb_with_primitives(
            wyrms[1]["program"],
            baked_map[wyrms[1]["asset_id"]],
            az_deg=96.0,
            el_deg=12.0,
            panel_size=panel_size,
            studio_preset=1,
            scene=wyrms[1]["scene"],
        )
        pgy = _format_four_primitives_for_render(
            ren_wyrm_side,
            primary_candidates=("wing_l_f4", "wing_l_fore"),
            secondary_candidates=("rear_foot_l", "fore_paw_l"),
            pivot_candidates=("rear_foot_l", "fore_paw_l"),
            dash_label="plane_vs_solid_ground_y",
        )
        _add(
            "v28_adversarial_mood_drift_punchback",
            "trap_v28_05_transparent_plane_ground_y_en",
            (
                "On this steeply winged dragon, a morphology script computed `ground_y` across ALL elements including "
                "zero-thickness alpha-cutout wing planes (`wing_bay_4`), shifting `ground_y` to `-7.2u` and causing every "
                "`gait`/`mantle` clip to sink underground. Ground the wing plane vs solid foot datum with <point>, <box>, "
                "<dash>, and <circle> and prescribe the Solid-Cuboid `ground_y` Datum Law."
            ),
            (
                f"Comparing cutout wing plane region {pgy['point1']} {pgy['box1']} against solid grounded foot "
                f"{pgy['point2']} (sole pivot {pgy['circle']}, datum separation {pgy['dash']}). "
                "Transparent wing plane corners must NEVER enter morph.ground_y!"
            ),
            (
                "**Verdict: PUNCH BACK — Transparent Cutout Plane Corrupting `morph.ground_y` (`-7.2u` Datum Bug)**\n"
                f"- **4-Primitive Grounding:** Wing cutout {pgy['point1']} {pgy['box1']} vs solid foot {pgy['point2']} {pgy['circle']} | {pgy['dash']}\n"
                "- **Solid-Cuboid `ground_y` Datum Law (`morphology.py`):** Compute `morph.ground_y` exclusively over **solid 3D "
                "cuboids (`plane_normal is None`)**. Exclude all zero-thickness alpha-cutout planes (`wing_bay_*`, capes, fins, "
                "trailing ribbons) so transparent plane corners never pull `gait`, `mantle`, or `flight` ground barriers underground."
            ),
            language="en",
            images=[im_wyrm_side],
            pixel_indispensable=True,
            meta_extra={"trap_id": "v28_trap_05_transparent_plane_ground_y"},
        )

    # Trap V28-6 (EN): Pole-A Gentle Guardian Corrupted by Hostile `Rx > 0 deg` Scowl!
    if _need("v28_adversarial_mood_drift_punchback"):
        ang_prog = parse_program(defects["angry_manta"]["dsl"])
        ang_scene = compile_geometry(ang_prog)
        ang_baked = bake_sculpted_rgba_atlas(ang_prog, scene=ang_scene, texels_per_unit=2)
        im_ang_df, ren_ang_df = _render_rgb_with_primitives(
            ang_prog, ang_baked, az_deg=180.0, el_deg=10.0, panel_size=panel_size, studio_preset=2, scene=ang_scene
        )
        im_ang_34, _ = _render_rgb_with_primitives(
            ang_prog, ang_baked, az_deg=148.0, el_deg=16.0, panel_size=panel_size, studio_preset=3, scene=ang_scene
        )
        pang = _format_four_primitives_for_render(
            ren_ang_df,
            primary_candidates=("head",),
            secondary_candidates=("jaw", "wing_l_upper"),
            pivot_candidates=("head", "jaw"),
            dash_label="pole_a_scowl_error",
        )
        _add(
            "v28_adversarial_mood_drift_punchback",
            "trap_v28_06_pole_a_scowl_corruption_en",
            (
                "This creature was commissioned as a **Pole A (Gentle & Approachable)** benevolent sanctuary manta guardian, "
                "yet its dead-front render looks hostile and menacing. Ground the failing face with <point>, <box>, <dash>, "
                "and <circle>, diagnose the inverted polarity knobs, and emit the surgical `Block-DSL v4.1` Pole-A fix."
            ),
            (
                f"Grounding corrupted manta head {pang['point1']} {pang['box1']}, jaw {pang['point2']}, error axis "
                f"{pang['dash']}, pivot {pang['circle']}. Brow uses Pole-B Rx=+22 deg / Rz=-28/+28 deg scowl and 16 deg "
                "snarl gape — violating Pole A Gentle & Approachable polarity!"
            ),
            (
                "**Verdict: PUNCH BACK — Pole-A Gentle Guardian Corrupted by Pole-B Scowl Knobs**\n"
                f"- **4-Primitive Diagnosis:** {pang['point1']} {pang['box1']} | {pang['dash']} | {pang['circle']}\n"
                "- **Surgical Pole-A Restoration (`Rx = -8 deg <= 0 deg` open brow + peaceful `0 deg` closed jaw):**\n"
                "```blockdsl\n"
                'box brow_l place(base_box=cranium_core,anchor=[-0.86,0.58,0.45],align=[0.35,-0.25,0]) { size:[2,1.5,4],material:foam,rotate:[-8,-8,8],prism_group:"hd" }\n'
                'box brow_r place(base_box=cranium_core,anchor=[0.86,0.58,0.45],align=[-0.35,-0.25,0]) { size:[2,1.5,4],material:foam,rotate:[-8,8,-8],prism_group:"hd" }\n'
                "# And set jaw rest_euler:[0,0,0] for a calm closed lip with zero hostile fangs.\n"
                "```"
            ),
            language="en",
            images=[im_ang_df, im_ang_34],
            pixel_indispensable=True,
            meta_extra={"trap_id": "v28_trap_06_pole_a_hostile_scowl_corruption"},
        )

    # Trap V28-7 (EN + ZH): Balanced Orbital Eye Shelf Extrusion vs Occlusion Trade-off (2 visual)
    if _need("v28_adversarial_mood_drift_punchback"):
        im_df_ok, ren_df_ok = _render_rgb_with_primitives(
            wyrms[0]["program"],
            baked_map[wyrms[0]["asset_id"]],
            az_deg=180.0,
            el_deg=6.0,
            panel_size=panel_size,
            studio_preset=0,
            scene=wyrms[0]["scene"],
        )
        peye = _format_four_primitives_for_render(
            ren_df_ok,
            primary_candidates=("head",),
            secondary_candidates=("horn_l", "jaw"),
            pivot_candidates=("head",),
            dash_label="orbital_shelf_balance",
        )
        _add(
            "v28_adversarial_mood_drift_punchback",
            "trap_v28_07_eye_extrusion_vs_occlusion_en",
            (
                "A modeler 'fixed' hidden side-view eyes on a Pole-B boss by pushing `eye_l/r` `1.5u` outward PAST `brow_l/r`, "
                "making the eyes bulge like bug-eyes in `0 deg` dead-front view. Ground the correct skull/orbital region on "
                "this reference render with <point>, <box>, <dash>, and <circle> and state the exact `v28` Balanced Orbital Shelf Law."
            ),
            (
                f"Grounding reference skull {peye['point1']} {peye['box1']}, reference {peye['point2']}, "
                f"shelf axis {peye['dash']}, pivot {peye['circle']}. Verify both inequalities of the balanced eye shelf."
            ),
            (
                "**Verdict: PUNCH BACK — Violated `v28` Zero-Extrusion Balanced Orbital Shelf Law**\n"
                f"- **4-Primitive Reference:** {peye['point1']} {peye['box1']} | {peye['dash']} | {peye['circle']}\n"
                "- **Exact Dual Inequality for `eye_l` (mirrored symmetrically for `eye_r`):**\n"
                "  1. **No Side/3-4 Occlusion:** `x_front_outer < cranium_xmin` so `cranium_core` never swallows the eye in side or 3/4 views.\n"
                "  2. **Zero Outward Extrusion:** `x_rear_outer >= brow_xmin + 0.35u` so the eye stays tucked inside the overhanging "
                "brow ridge and cheekbone — readable from `0 deg` dead-front, `3/4`, AND profile without bug-eye bulge!"
            ),
            language="en",
            images=[im_df_ok],
            pixel_indispensable=True,
            meta_extra={"trap_id": "v28_trap_07_eye_extrusion_vs_occlusion"},
        )
        _add(
            "v28_adversarial_mood_drift_punchback",
            "trap_v28_07_eye_extrusion_vs_occlusion_zh",
            (
                "某模型为了防止侧视挡住眼睛，把 `eye_l/r` 向外推得超出了眉弓外缘，导致正视（`0°`）出现突眼穿帮。"
                "请结合这张全彩参考图与 <point>、<box>、<dash>、<circle> 原语，说明 `v28` 零凸出平衡眼眶台的双边不等式。"
            ),
            (
                f"定位参考颅面 {peye['point1']} {peye['box1']}、角冠 {peye['point2']}、眼眶台轴 {peye['dash']} 与枢轴 {peye['circle']}。"
                "必须同时满足不被颅骨吞没且不超出眉弓外缘 0.35u 的双边条件。"
            ),
            (
                "**裁决：打回修复——违反 `v28` 零凸出平衡眼眶台定律**\n"
                f"- **4原语定位：** {peye['point1']} {peye['box1']} | {peye['dash']} | {peye['circle']}\n"
                "- **平衡眼眶台双边约束（左眼 `eye_l`，右眼镜像）：**\n"
                "  1. **防侧视吞没：** `x_front_outer < cranium_xmin`，确保在 `Side` 与 `3/4` 视角下眼球不被 `cranium_core` 遮挡；\n"
                "  2. **防正视外凸：** `x_rear_outer >= brow_xmin + 0.35u`，确保眼球稳稳收在眉弓与颧骨檐下，正视 `0°`、3/4 与侧视全角度锐利无突眼。"
            ),
            language="zh",
            images=[im_df_ok],
            pixel_indispensable=True,
            meta_extra={"trap_id": "v28_trap_07_eye_extrusion_vs_occlusion"},
        )

    # 4 Crisp Textual Adversarial Punch-Back Q&As (completing 16 Task-6 records!)
    text_traps = (
        (
            "trap_v28_08_chin_up_dead_front_stacking",
            "en",
            "Why does a Pole-B dragon with `neck_1 = -34 deg`, `neck_2 = -6 deg`, `head = +26 deg` look like flat stacked Lego bricks from `0 deg` dead-front view?",
            "Sum cumulative X-pitch: -34 - 6 + 26 = -14 deg (chin-up foreshortening!). Fix so net world head X-pitch lands in +6..+10 deg.",
            (
                "**Verdict: PUNCH BACK — Cumulative Chin-Up Foreshortening (`-14 deg`) Violates Dead-Front Perspective Law**\n"
                "- **Math:** Net world head X-pitch `= -34 + (-6) + 26 = -14 deg` tilts the nose UP, stacking the lower jaw, "
                "muzzle, and brow on top of the eyes like flat horizontal Lego bricks in `0 deg` dead-front view.\n"
                "- **Fix:** Always sum `pelvis -> waist -> chest -> neck_1..N -> head` X-pitches so net world head pitch lands "
                "inside **`+6..+10 deg` downward predator glare** (e.g., `-20 + 10 + 18 = +8 deg`), tapering `cranium -> snout -> prow` "
                "by `~0.72x` per step with diagonal V-rail maxilla walls."
            ),
        ),
        (
            "trap_v28_09_mirror_x_uv_cutout_bug",
            "en",
            "Why did right-side cheek ruffs and right wing membranes cut their alpha jagged edge on the INNER face after `mirror_x`?",
            "Branching on bone/box string name `.endswith('_l')` fails on mirrored copies like `horn_r__ruff_l`. Always branch on actual world X (`box.center[0] < 0`).",
            (
                "**Verdict: PUNCH BACK — Mirror UV Cutout Inversion Bug (`name.endswith(\"_l\")` Trap)**\n"
                "- **Root Cause:** SyncCraft `mirror_x` preserves child suffixes (e.g. `wing_r_upper__wing_bay_4` or `ruff_r__tuft_l`), "
                "so texture painters checking `name.endswith(\"_l\")` treat the mirrored right side as left and carve alpha notches on the inner seam!\n"
                "- **Fix:** Always test actual compiled geometry **`box.center[0] < 0.0`** (never string suffix!) when choosing left vs right "
                "alpha cutout polarity, and verify every pass with `clay=False` full textures enabled."
            ),
        ),
        (
            "trap_v28_10_baked_hover_altitude_and_root_z",
            "en",
            "A contributor exported a dragon `cruise` flight animation with `pelvis_root.y = +38u` and `pelvis_root.z` translating `+64u` forward over 2.0s. Should this ship?",
            "Reject immediately: breaks v28 Minecraft in_place:true Entity-Space Contract (max|root_z| == 0.00u and zero fake hover elevation).",
            (
                "**Verdict: REJECT EXPORT — Violates `v28` `in_place: true` Minecraft Entity-Space Contract**\n"
                "- **Bugs:** Baking `+64u` forward `root.z` fights Minecraft's entity physics and snaps back violently at every loop boundary; "
                "baking `+38u` hover `root.y` offsets the hitbox 2.4 blocks below the mesh.\n"
                "- **Fix:** Run `flight(mode=cruise, in_place=true)`: keep `max|root_z| == 0.00u`, restrict `root.y` solely to local "
                "anti-phase COM lift recoil (`+-1..4.5u`), and emit matched velocity inside `synccraft_root_motion.speed_blocks_per_s`."
            ),
        ),
        (
            "trap_v28_11_transparent_plane_ground_y_zh",
            "zh",
            "为什么把透明翼膜平面（`plane_normal is not None`）纳入 `morph.ground_y` 计算会彻底破坏所有 `gait` 与 `mantle` 动画？",
            "大仰角翼膜平面的透明包围盒底角可能探到 y=-7.2u，拉低 ground_y 基准并导致整个步态地面屏障下陷。",
            (
                "**裁决：严禁将透明镂空平面计入 `morph.ground_y`（实体立方体基准定律）**\n"
                "- **失效机理：** 陡倾角教堂巨翼的零厚度 `plane`（如 `wing_bay_4`）在世界空间中存在大片透明像素角点，其包围盒下沿可能下探至 `-7.2u`；"
                "若计入 `morph.ground_y`，会把全部 `gait`、`mantle` 与 `flight` 的地面碰撞屏障同步拉入地下 `-7.2u`。\n"
                "- **修复规范：** `morphology.py` 计算 `ground_y` 时必须严格过滤为 **实体 3D 立方体（`plane_normal is None`）**。"
            ),
        ),
    )
    for tid, lang, prm, thk, bdy in text_traps:
        _add(
            "v28_adversarial_mood_drift_punchback",
            tid,
            prm,
            thk,
            bdy,
            language=lang,
            images=[],
            pixel_indispensable=False,
            meta_extra={"trap_id": tid},
        )

    # -----------------------------------------------------------------------
    # 6. Stratified Round-Robin Selection up to `max_records` (keeps all 6 tasks!)
    # -----------------------------------------------------------------------
    task_queues: dict[str, list[dict[str, Any]]] = {}
    for t in AESTHETIC_V28_TASKS:
        items = list(by_task[t])
        rng.shuffle(items)
        task_queues[t] = items

    total_available = sum(len(v) for v in task_queues.values())
    target_n = min(int(max_records), total_available)
    selected: list[dict[str, Any]] = []
    while len(selected) < target_n:
        progressed = False
        for t in AESTHETIC_V28_TASKS:
            if task_queues[t] and len(selected) < target_n:
                selected.append(task_queues[t].pop(0))
                progressed = True
        if not progressed:
            break

    rng.shuffle(selected)
    return selected
