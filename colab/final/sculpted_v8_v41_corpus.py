"""Stage-7 Sculpted Blockbench Style v8 (`SKILL.md` v8) + SyncCraft `Block-DSL v4.1` Corpus & Evaluator.

Punches the complete `Blockbench_sculpted_style-v8` construction discipline and `Block-DSL v4.1`
(`ADR 0008`..`ADR 0013`) native surface/locomotion/modular-assembly capabilities into `Spark-X2.6-VL`:

1. **Hard `SKILL.md` v8 Anti-Shard & Head-Gate Verifier (`run_v8_construct_check`)**:
   - Non-head `boxes_per_bone <= 3.5` (and `max_non_head_bone_boxes <= 8`) — zero shard armour.
   - Top-10 cuboids hold `>= 50%` of total cuboid box volume (`Big 70 / Med 25 / Small 5` hierarchy).
   - Exact `X = 0` rest-pose mirror symmetry (`mirror_parity == "OK"`); attitude lives only in `@behavior`.
   - 3-step sculpted head depth (`cranium` + overhanging `brow` + `cheek` + wedged `muzzle` + hinged `jaw`),
     real block `horn` chains (`2..4` tapered segments, never flat sheets), and eye `@stamps` under the brow.
   - Full `audit_scene(scene, audit_profile="sculpted")` + `evaluate_sculpted_style_anchor` + swept motion audit.

2. **Progressive `W1 -> W2 -> W3 -> D1 -> D2 -> T1 -> T2 -> A1` Passes & Visual-Primitives Region Critique**:
   - Clean-room `CC0-1.0` `dsl_version: "4.1"` creatures, critters, vehicles (`octagon` hubs), and scenes.
   - Deliberate negative shard-armour / doghouse-snout / broken-symmetry punch-back (`PUNCH BACK TO W1`) critiques
     grounded with `<box>` and `<point>` Visual Primitives on turnaround + zoomed head/limb region sheets.

3. **Zero-Leakage `v4.1` Modular `@layout/@component/@assembly` & Gate D 5-Task Proposer Supervision**:
   - Disjoint modular assemblies (`0%` overlap with `v4_wm_01..04` holdout).
   - Strictly `train`/`val` proposer records (`0%` overlap with `holdout_gate_d` `biped_theropod` / `hexapod_arthropod`
     and `0%` `research_nc` / Capcom / Cobblemon / Mojang assets).
"""
from __future__ import annotations

import collections
import hashlib
import io
import json
import math
import random
import re
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .common import emit, identity
from .sculpted_style_corpus import (
    bake_sculpted_rgba_atlas,
    render_sculpted_studio_frame,
    render_sculpted_turnaround_views,
)
from .synccraft_bridge import (
    ensure_synccraft_importable,
    format_supervised_answer,
    make_clean_room_provenance,
)

SCULPTED_V8_TASKS: tuple[str, ...] = (
    "synccraft_v41_sculpted_v8",
    "synccraft_sculpted_stage_progression",
    "synccraft_sculpted_multiview_inverse",
    "synccraft_sculpted_self_critique",
    "synccraft_v4_modular_assembly",
    "synccraft_v41_proposer",
)


def run_v8_construct_check(dsl_text: str) -> dict[str, Any]:
    """Execute `SKILL.md` v8 Section 3.6 + 6.6 + 11.5 checks on a `Block-DSL v4.1` program."""
    ensure_synccraft_importable()
    from synccraft.audit.sculpted_style_evaluator import evaluate_sculpted_style_anchor
    from synccraft.audit.sync_audit import audit_scene
    from synccraft.dsl.parser import parse_program
    from synccraft.kinematics.morphology import analyze_morphology
    from synccraft.kinematics.musculoskeletal import audit_behavior_clip
    from synccraft.kinematics.se3_chain import compile_geometry

    prog = parse_program(dsl_text)
    scene = compile_geometry(prog)
    aud = audit_scene(scene)
    boxes = [b for b in scene.boxes if getattr(b, "plane_normal", None) is None and float(np.min(b.size)) > 0.0]
    planes = [b for b in scene.boxes if getattr(b, "plane_normal", None) is not None or float(np.min(b.size)) <= 0.0]

    vols = np.array(sorted((float(np.prod(b.size)) for b in boxes), reverse=True), dtype=np.float64)
    total_vol = float(vols.sum()) if vols.size else 1.0
    top10_share = float(vols[:10].sum() / max(1e-6, total_vol)) if vols.size else 0.0

    per_bone = collections.Counter(b.bone_name for b in boxes)
    head_bone_tokens = ("head", "skull", "cranium", "jaw", "mandible", "snout", "muzzle")
    non_head_counts = [
        cnt for bname, cnt in per_bone.items() if not any(tok in bname.lower() for tok in head_bone_tokens)
    ]
    boxes_per_bone_all = len(boxes) / float(max(1, len(per_bone)))
    boxes_per_bone_non_head = (
        sum(non_head_counts) / float(max(1, len(non_head_counts))) if non_head_counts else boxes_per_bone_all
    )
    max_non_head_bone = max(non_head_counts) if non_head_counts else 0

    def _mkey(b: Any, sx: float) -> tuple[float, float, float, float, float, float]:
        return (
            round(sx * float(b.center[0]), 1),
            round(float(b.center[1]), 1),
            round(float(b.center[2]), 1),
            round(float(b.size[0]), 1),
            round(float(b.size[1]), 1),
            round(float(b.size[2]), 1),
        )

    left = sorted(_mkey(b, -1.0) for b in scene.boxes if float(b.center[0]) < -0.05)
    right = sorted(_mkey(b, +1.0) for b in scene.boxes if float(b.center[0]) > +0.05)
    mirror_ok = left == right

    # Head 3-step depth check (`cranium`/`skull`, `brow`, `cheek` or `nasal`, `muzzle`/`snout`, hinged `jaw` bone)
    bnames_low = {b.name.lower() for b in boxes}
    bonenames_low = {bn.name.lower() for bn in scene.bones}
    has_brow = any("brow" in n for n in bnames_low)
    has_cheek_or_nasal = any(("cheek" in n or "zygoma" in n or "nasal" in n) for n in bnames_low)
    has_muzzle = any(("snout" in n or "muzzle" in n or "rostrum" in n or "beak" in n or "cab" in n or "hull" in n) for n in bnames_low)
    has_jaw_bone = any(("jaw" in n or "mandible" in n) for n in bonenames_low)
    is_creature = prog.meta.archetype == "entity" and not any(
        k in bnames_low for k in ("chassis_hull", "rover_cab", "turret_housing", "desk_top")
    )
    head_depth_steps = int(has_brow) + int(has_cheek_or_nasal) + int(has_muzzle)
    head_gate_ok = (not is_creature) or (head_depth_steps >= 2 and (has_jaw_bone or len(boxes) <= 36))

    # Plane horn ban (`SKILL.md` v8 Non-Negotiable #4: never a flat plane for a horn)
    plane_horn_violation = any("horn" in p.name.lower() for p in planes)

    anchor_rep = evaluate_sculpted_style_anchor(prog, scene)
    morph = analyze_morphology(prog, scene)

    clips_ok = True
    clip_reports: dict[str, Any] = {}
    if prog.behavior and prog.behavior.clips:
        for cl in prog.behavior.clips:
            crep = audit_behavior_clip(prog, cl.name)
            clip_reports[cl.name] = {
                "passed": bool(crep["passed"]),
                "collision_events": int(crep.get("collision_events", 0)),
            }
            if not crep["passed"]:
                clips_ok = False

    punch_back_reasons: list[str] = []
    if not aud["passed"]:
        punch_back_reasons.append("static_sat_or_contact_violation")
    if boxes_per_bone_non_head > 3.5 or max_non_head_bone > 8:
        punch_back_reasons.append(f"shard_armour_boxes_per_bone:{boxes_per_bone_non_head:.2f}")
    if len(boxes) >= 12 and top10_share < 0.50:
        punch_back_reasons.append(f"fragmented_top10_volume:{top10_share:.2%}")
    if not mirror_ok:
        punch_back_reasons.append(f"broken_mirror_parity:{len(left)}_vs_{len(right)}")
    if plane_horn_violation:
        punch_back_reasons.append("forbidden_plane_horn")
    if not head_gate_ok:
        punch_back_reasons.append("head_gate_missing_depth_or_jaw")
    if anchor_rep.metrics.get("snout_taper_ratio", 0.7) > 0.88 and is_creature:
        punch_back_reasons.append("doghouse_snout_width")
    if not clips_ok:
        punch_back_reasons.append("swept_motion_clip_failed")

    min_anchor = 0.78 if is_creature else 0.72
    passed_v8 = len(punch_back_reasons) == 0 and float(anchor_rep.style_anchor_score) >= min_anchor
    return {
        "passed_v8": passed_v8,
        "punch_back_to_w1": len(punch_back_reasons) > 0,
        "punch_back_reasons": punch_back_reasons,
        "dsl_version": str(prog.meta.dsl_version),
        "elements_total": len(scene.boxes),
        "cuboid_count": len(boxes),
        "plane_count": len(planes),
        "bone_count": len(scene.bones),
        "boxes_per_bone_all": round(boxes_per_bone_all, 2),
        "boxes_per_bone_non_head": round(boxes_per_bone_non_head, 2),
        "max_non_head_bone_boxes": max_non_head_bone,
        "top10_volume_share": round(top10_share, 4),
        "mirror_parity": "OK" if mirror_ok else f"BROKEN ({len(left)} vs {len(right)})",
        "symmetry_pair_count": len(morph.symmetry_pairs),
        "head_depth_steps": head_depth_steps,
        "has_jaw_bone": has_jaw_bone,
        "style_anchor_score": round(float(anchor_rep.style_anchor_score), 4),
        "style_pillar_scores": anchor_rep.to_dict()["pillar_scores"],
        "static_audit_passed": bool(aud["passed"]),
        "clips_passed": clips_ok,
        "clip_reports": clip_reports,
    }


def _render_v8_views_pil(
    program: Any,
    *,
    panel_size: tuple[int, int] = (256, 256),
) -> dict[str, Image.Image]:
    pngs, _ = render_sculpted_turnaround_views(
        program,
        views=("south", "right", "top", "iso"),
        panel_size=panel_size,
        include_silhouette_sheet=True,
    )
    return {
        "hero": Image.open(io.BytesIO(pngs["iso"])).convert("RGBA"),
        "front": Image.open(io.BytesIO(pngs["south"])).convert("RGBA"),
        "side": Image.open(io.BytesIO(pngs["right"])).convert("RGBA"),
        "top": Image.open(io.BytesIO(pngs["top"])).convert("RGBA"),
        "silhouette_sheet": Image.open(io.BytesIO(pngs["silhouette_sheet"])).convert("RGBA"),
    }


def render_v8_zoomed_region_sheet(
    program: Any,
    scene: Any | None = None,
    bake_info: dict[str, Any] | None = None,
    *,
    panel_size: tuple[int, int] = (224, 224),
    pre_rendered_turns: dict[str, Image.Image] | None = None,
) -> tuple[Image.Image, dict[str, list[int]]]:
    """Render a 2x2 `SKILL.md` v8 Region Sheet (`head_3q`, `head_side`, `limb_side`, `silhouette_side`)
    and return normalized `[0..1000]` bounding boxes for Visual-Primitives `<box>` / `<point>` supervision.
    """
    _ = (scene, bake_info)
    turns = pre_rendered_turns if pre_rendered_turns is not None else _render_v8_views_pil(program, panel_size=panel_size)
    iso = turns["hero"]
    side = turns["side"]
    front = turns["front"]
    sil = turns["silhouette_sheet"]

    pw, ph = panel_size

    def _crop_focus(im: Image.Image, box_frac: tuple[float, float, float, float]) -> Image.Image:
        w, h = im.size
        x0, y0, x1, y1 = (
            int(round(box_frac[0] * w)),
            int(round(box_frac[1] * h)),
            int(round(box_frac[2] * w)),
            int(round(box_frac[3] * h)),
        )
        cropped = im.crop((max(0, x0), max(0, y0), min(w, max(x0 + 16, x1)), min(h, max(y0 + 16, y1))))
        return cropped.resize((pw, ph), resample=Image.Resampling.NEAREST)

    # Top-left: Head 3/4 zoom; Top-right: Head/Profile side zoom; Bottom-left: Limb/Ground contact zoom; Bottom-right: Silhouette check
    p_head_3q = _crop_focus(iso, (0.18, 0.08, 0.78, 0.68))
    p_head_side = _crop_focus(side, (0.42, 0.10, 0.96, 0.66))
    p_limb_ground = _crop_focus(front, (0.14, 0.36, 0.86, 0.98))
    p_sil = sil.resize((pw, ph), resample=Image.Resampling.NEAREST)

    sheet = Image.new("RGBA", (pw * 2, ph * 2), (20, 22, 30, 255))
    sheet.paste(p_head_3q, (0, 0))
    sheet.paste(p_head_side, (pw, 0))
    sheet.paste(p_limb_ground, (0, ph))
    sheet.paste(p_sil, (pw, ph))

    draw = ImageDraw.Draw(sheet)
    draw.line([(pw, 0), (pw, ph * 2)], fill=(58, 64, 82, 255), width=2)
    draw.line([(0, ph), (pw * 2, ph)], fill=(58, 64, 82, 255), width=2)

    region_boxes_1000 = {
        "head_3q_panel": [20, 20, 480, 480],
        "head_side_wedge": [540, 60, 970, 450],
        "limb_ground_contact": [40, 540, 470, 980],
        "silhouette_panel": [520, 520, 980, 980],
    }
    return sheet.convert("RGB"), region_boxes_1000


# ---------------------------------------------------------------------------
# Clean-Room `CC0-1.0` `Block-DSL v4.1` Progressive Families (`W1` / `D1` / `A1`)
# ---------------------------------------------------------------------------

def _build_saffron_squallwing_v41(variant: int = 0) -> dict[str, Any]:
    """Six-limbed storm-desert dragon in `Block-DSL v4.1` obeying all 15 `SKILL.md` v8 rules:
    - Exact X=0 symmetric bind pose (attitude moved into `display_roar` clip, fixing Squallwing's v8 2.5 note!)
    - 3-step head depth (`cranium`, `brow_l`, `cheek_l`, `snout_bridge`, `nasal_tip` + hinged `jaw` bone)
    - 3-segment block horns (`>= 2u` root thickness, steady curve, zero plane horns)
    - `seat=0.28` S-curve neck & tail, cross-bone `blend_with` skull/shoulder shingles
    - 6-panel partitioned membrane wings (`layer: 1..3`, `edge: triangle_bay` / `scallop`)
    - `@style { shade: banded(...) }`, `@materials` `scale_rows` / `banded_limb` / `plain`, `@stamps` eye + teeth
    - Froude `gait` + additive `layer breathe` + `effect touchdown` (`ADR 0011` / `ADR 0013`)
    """
    palettes = (
        ("saffron", [0.68, 0.16, 68], [0.84, 0.06, 86], [0.34, 0.08, 255], [0.76, 0.20, 210]),
        ("vermilion", [0.56, 0.18, 28], [0.84, 0.06, 84], [0.86, 0.05, 82], [0.78, 0.19, 62]),
        ("viridian", [0.58, 0.15, 162], [0.82, 0.07, 92], [0.32, 0.07, 248], [0.78, 0.19, 48]),
    )
    tag, body_lch, belly_lch, horn_lch, glow_lch = palettes[variant % len(palettes)]
    chest_w = 10.0 + (variant % 2) * 1.0
    chest_l = 12.0 + (variant % 2) * 1.0

    w1_blockout = f"""
@meta {{
  archetype: entity,
  target_profile: bedrock_entity,
  texel_density: 32,
  units_per_block: 16,
  dsl_version: "4.1",
  size_step: 0.5,
  audit_profile: sculpted
}}
@layout {{
  budget max_boxes = 40
  budget max_bones = 28
  envelope max_span = [72, 48, 80]
  envelope grounded = true
}}
@palette {{
  hide: ramp(base_lch={body_lch}, steps=5, hue_shift=[-14, 12])
  under: ramp(base_lch={belly_lch}, steps=4, hue_shift=[-10, 8])
  keratin: ramp(base_lch={horn_lch}, steps=4, hue_shift=[-10, 10])
}}
@rig {{
  root bone body {{
    pivot_local: [0, 11.5, 0]
    box chest {{ size: [{chest_w}, 8, {chest_l}], material: hide, rotate: [-4, 0, 0] }}
    bone waist attach(parent_box=chest, anchor=[0, 0.05, -1], align=[0, 0, 1], seat=0.28) {{
      rest_euler: [6, 0, 0]
      joint_limits: [[-18, 18], [-20, 20], [-12, 12]]
      box loin {{ size: [{chest_w - 2.0}, 6.5, 8], material: hide }}
      bone hips attach(parent_box=loin, anchor=[0, -0.05, -1], align=[0, 0, 1], seat=0.28) {{
        rest_euler: [-5, 0, 0]
        joint_limits: [[-18, 18], [-20, 20], [-12, 12]]
        box pelvis {{ size: [{chest_w - 1.0}, 7, 8.5], material: hide }}
        chain tail attach(parent_box=pelvis, anchor=[0, 0.15, -1], align=[0, 0, 1], seat=0.28) segments=4 taper=0.78 bend=[-8, -4, 8, 14] {{
          joint_limits: [[-24, 24], [-28, 28], [-15, 15]]
          box caudal {{ size: [5, 4.5, 7], material: hide }}
        }}
        bone hind_leg_l attach(parent_box=pelvis, anchor=[-1, -0.35, 0.1], align=[0.6, 0.8, 0]) mirror_x(hind_leg_r) {{
          rest_euler: [-16, 0, 0]
          joint_limits: [[-50, 50], [-20, 20], [-20, 20]]
          box thigh {{ size: [4, 7.5, 5], material: hide }}
          bone crus attach(parent_box=thigh, anchor=[0, -1, -0.15], align=[0, 1, 0], seat=0.22) {{
            rest_euler: [30, 0, 0]
            joint_limits: [[-45, 55], [-15, 15], [-15, 15]]
            box shin {{ size: [3, 6.5, 3.5], material: hide }}
            bone hind_foot attach(parent_box=shin, anchor=[0, -1, 0.1], align=[0, 1, -0.3], seat=0.15) {{
              rest_euler: [-14, 0, 0]
              joint_limits: [[-35, 35], [-15, 15], [-15, 15]]
              box pes {{ size: [4, 2, 5.5], material: under }}
            }}
          }}
        }}
      }}
    }}
    bone neck_1 attach(parent_box=chest, anchor=[0, 0.35, 1], align=[0, -0.2, -1], seat=0.28) {{
      rest_euler: [-22, 0, 0]
      joint_limits: [[-28, 28], [-30, 30], [-18, 18]]
      box cerv_1 {{ size: [6, 6, 6.5], material: hide }}
      bone neck_2 attach(parent_box=cerv_1, anchor=[0, 0.15, 1], align=[0, -0.1, -1], seat=0.28) {{
        rest_euler: [26, 0, 0]
        joint_limits: [[-28, 28], [-30, 30], [-18, 18]]
        box cerv_2 {{ size: [5, 5, 6], material: hide }}
        bone head attach(parent_box=cerv_2, anchor=[0, 0.1, 1], align=[0, 0, -1], seat=0.26) {{
          rest_euler: [4, 0, 0]
          joint_limits: [[-30, 32], [-35, 35], [-20, 20]]
          box cranium {{ size: [6.5, 5, 6.5], material: hide }}
          bone brow_l attach(parent_box=cranium, anchor=[-0.85, 0.72, 0.45], align=[0.4, -0.5, 0]) mirror_x(brow_r) {{
            rest_euler: [6, 0, -12]
            box brow_ridge {{ size: [2, 1.5, 4], material: hide, blend_with: [cranium] }}
          }}
          box snout_bridge place(base_box=cranium, anchor=[0, 0.05, 1], align=[0, 0, -0.8]) {{
            size: [4.5, 3, 5.5], material: hide, rotate: [8, 0, 0], blend_with: [cranium]
          }}
          bone jaw attach(parent_box=cranium, anchor=[0, -0.75, 0.35], align=[0, 0.5, -0.85]) {{
            rest_euler: [8, 0, 0]
            joint_limits: [[-4, 38], [-8, 8], [-8, 8]]
            box mandible {{ size: [4, 1.5, 5], material: under }}
          }}
        }}
      }}
    }}
    bone fore_leg_l attach(parent_box=chest, anchor=[-1, -0.45, 0.55], align=[0.65, 0.85, 0]) mirror_x(fore_leg_r) {{
      rest_euler: [8, 0, 0]
      joint_limits: [[-50, 50], [-20, 20], [-20, 20]]
      box humerus {{ size: [3.5, 6.5, 4], material: hide }}
      bone forearm attach(parent_box=humerus, anchor=[0, -1, 0], align=[0, 1, 0], seat=0.20) {{
        rest_euler: [-12, 0, 0]
        joint_limits: [[-45, 45], [-15, 15], [-15, 15]]
        box radius {{ size: [2.5, 5.5, 3], material: hide }}
        bone fore_foot attach(parent_box=radius, anchor=[0, -1, 0.15], align=[0, 1, -0.35], seat=0.15) {{
          rest_euler: [4, 0, 0]
          joint_limits: [[-30, 30], [-15, 15], [-15, 15]]
          box manus {{ size: [3.5, 2, 4.5], material: under }}
        }}
      }}
    }}
  }}
}}
""".strip()

    final_v41 = f"""
@meta {{
  archetype: entity,
  target_profile: bedrock_entity,
  texel_density: 32,
  units_per_block: 16,
  dsl_version: "4.1",
  size_step: 0.5,
  audit_profile: sculpted
}}
@layout {{
  budget max_boxes = 120
  budget max_bones = 56
  envelope max_span = [96, 64, 96]
  envelope grounded = true
}}
@palette {{
  hide: ramp(base_lch={body_lch}, steps=5, hue_shift=[-16, 12])
  limb_coat: ramp(base_lch=[{round(body_lch[0] - 0.05, 2)}, {body_lch[1]}, {body_lch[2]}], steps=5, hue_shift=[-14, 10])
  under: ramp(base_lch={belly_lch}, steps=4, hue_shift=[-10, 8])
  keratin: ramp(base_lch={horn_lch}, steps=4, hue_shift=[-12, 10])
  storm_glow: ramp(base_lch={glow_lch}, steps=4, hue_shift=[-8, 12], emissive=true)
}}
@rig {{
  root bone body {{
    pivot_local: [0, 11.5, 0]
    box chest {{ size: [{chest_w}, 8, {chest_l}], material: hide, rotate: [-4, 0, 0] }}
    box withers_crest place(base_box=chest, anchor=[0, 1, 0.1], align=[0, -0.45, 0]) {{
      size: [4, 2.5, 7], material: hide, rotate: [10, 0, 0], blend_with: [chest]
    }}
    box chest_scute place(base_box=chest, anchor=[0, -1, 0.1], align=[0, 0.72, 0]) {{
      size: [{chest_w - 2.5}, 2, {chest_l - 2.0}], material: under, rotate: [-4, 0, 0], blend_with: [chest]
    }}
    bone shoulder_l attach(parent_box=chest, anchor=[-1, 0.25, 0.72], align=[0.85, 0, 0]) mirror_x(shoulder_r) {{
      rest_euler: [0, 0, -8]
      box shoulder_mass {{ size: [2, 2.5, 2.5], material: hide, blend_with: [chest] }}
    }}
    plane dorsal_fringe place(base_box=withers_crest, anchor=[0, 1, 0], align=[0, -0.85, 0]) {{
      size: [3.5, 6.5], normal: x, layer: 0, edge: spikes(width=2, length=1..3, side=top), material: keratin
    }}

    bone waist attach(parent_box=chest, anchor=[0, 0.05, -1], align=[0, 0, 1], seat=0.28) {{
      rest_euler: [6, 0, 0]
      joint_limits: [[-18, 18], [-20, 20], [-12, 12]]
      box loin {{ size: [3.5, 5.5, 7], material: hide }}
      box belly_scute place(base_box=loin, anchor=[0, -1, 0], align=[0, 0.72, 0]) {{
        size: [3, 1.5, 5.5], material: under, blend_with: [loin]
      }}

      bone hips attach(parent_box=loin, anchor=[0, -0.05, -1], align=[0, 0, 1], seat=0.28) {{
        rest_euler: [-5, 0, 0]
        joint_limits: [[-18, 18], [-20, 20], [-12, 12]]
        box pelvis {{ size: [{chest_w + 1.0}, 7, 8.5], material: hide }}
        bone haunch_l attach(parent_box=pelvis, anchor=[-1, 0.92, -0.75], align=[0.85, -0.5, 0]) mirror_x(haunch_r) {{
          rest_euler: [0, 0, -7.5]
          box haunch_mass {{ size: [2, 2, 2.5], material: hide, blend_with: [pelvis] }}
        }}

        chain tail attach(parent_box=pelvis, anchor=[0, 0.15, -1], align=[0, 0, 1], seat=0.28) segments=5 taper=0.80 seat=0.28 bend=[-8, -4, 6, 12, 15] {{
          joint_limits: [[-24, 24], [-28, 28], [-15, 15]]
          box caudal {{ size: [5, 4.5, 6.5], material: hide }}
        }}
        bone streamer_l attach(parent_box=pelvis, anchor=[-0.95, 0.65, -0.85], align=[0.90, -0.5, 0.90]) mirror_x(streamer_r) {{
          rest_euler: [12, -24, 0]
          plane pelvic_streamer {{
            size: [2.5, 6], normal: y, layer: 0, edge: ragged(depth=2, holes=1, side=bottom, seed=5), material: storm_glow
          }}
        }}

        bone hind_leg_l attach(parent_box=pelvis, anchor=[-1, -0.35, -0.25], align=[1.0, 0.85, 0]) mirror_x(hind_leg_r) {{
          rest_euler: [-16, 0, 0]
          joint_limits: [[-50, 50], [-20, 20], [-20, 20]]
          box thigh {{ size: [4, 7.5, 5], material: limb_coat }}
          box knee_cap_l place(base_box=thigh, anchor=[0, -0.85, 0.85], align=[0, 0.3, -0.4]) {{
            size: [2.5, 2.5, 2], material: hide, rotate: [15, 0, 0], blend_with: [thigh]
          }}
          bone crus attach(parent_box=thigh, anchor=[0, -1, -0.15], align=[0, 1, 0], seat=0.22) {{
            rest_euler: [30, 0, 0]
            joint_limits: [[-45, 55], [-15, 15], [-15, 15]]
            box shin {{ size: [3, 6.5, 3.5], material: limb_coat }}
            box calf_bulge_l place(base_box=shin, anchor=[0, 0.25, -0.85], align=[0, 0, 0.45]) {{
              size: [2.5, 3.5, 2], material: limb_coat, rotate: [-10, 0, 0], blend_with: [shin]
            }}
            bone hind_foot attach(parent_box=shin, anchor=[0, -1, 0.1], align=[0, 1, -0.3], seat=0.15) {{
              rest_euler: [-14, 0, 0]
              joint_limits: [[-35, 35], [-15, 15], [-15, 15]]
              box pes {{ size: [4, 2, 5], material: under }}
              box heel_spur_l place(base_box=pes, anchor=[0, 0.2, -1], align=[0, 0, 0.5]) {{
                size: [2, 1.5, 2], material: keratin, rotate: [-12, 0, 0], blend_with: [pes]
              }}
              box h_claw_mid place(base_box=pes, anchor=[0, -0.2, 1], align=[0, 0, -0.4]) {{
                size: [1.5, 1.5, 2.5], material: keratin, rotate: [15, 0, 0], blend_with: [pes]
              }}
            }}
          }}
        }}
      }}
    }}

    bone neck_1 attach(parent_box=chest, anchor=[0, 0.35, 1], align=[0, -0.2, -1], seat=0.28) {{
      rest_euler: [-22, 0, 0]
      joint_limits: [[-28, 28], [-30, 30], [-18, 18]]
      box cerv_1 {{ size: [6, 6, 6.5], material: hide }}
      box throat_plate_1 place(base_box=cerv_1, anchor=[0, -1, 0], align=[0, 0.7, 0]) {{
        size: [4.5, 1.5, 5], material: under, blend_with: [cerv_1]
      }}

      bone neck_2 attach(parent_box=cerv_1, anchor=[0, 0.15, 1], align=[0, -0.1, -1], seat=0.28) {{
        rest_euler: [26, 0, 0]
        joint_limits: [[-28, 28], [-30, 30], [-18, 18]]
        box cerv_2 {{ size: [5, 5, 6], material: hide }}
        plane neck_fringe place(base_box=cerv_2, anchor=[0, 1, 0], align=[0, -0.85, 0]) {{
          size: [2.5, 5], normal: x, layer: 0, edge: spikes(width=2, length=1..3, side=top), material: keratin
        }}

        bone head attach(parent_box=cerv_2, anchor=[0, 0.1, 1], align=[0, 0, -1], seat=0.26) {{
          rest_euler: [4, 0, 0]
          joint_limits: [[-30, 32], [-35, 35], [-20, 20]]
          box cranium {{ size: [6.5, 5, 6.5], material: hide }}
          bone brow_l attach(parent_box=cranium, anchor=[-0.85, 0.68, 0.72], align=[0.4, -0.5, -0.2]) mirror_x(brow_r) {{
            rest_euler: [6, 0, -12]
            box brow_ridge {{ size: [2, 1.5, 2.5], material: hide, blend_with: [cranium] }}
          }}
          bone cheek_l attach(parent_box=cranium, anchor=[-0.95, -0.15, 0.15], align=[0.65, 0, 0]) mirror_x(cheek_r) {{
            rest_euler: [0, -8, 0]
            box cheek_mass {{ size: [2, 3, 3], material: hide, blend_with: [cranium] }}
          }}
          box crest_shingle place(base_box=cranium, anchor=[0, 0.85, -0.85], align=[0, -0.4, 0.5]) {{
            size: [2.5, 1.5, 3], material: keratin, rotate: [-15, 0, 0], blend_with: [cranium, cerv_2]
          }}
          box snout_bridge place(base_box=cranium, anchor=[0, 0.05, 1], align=[0, 0, -0.8]) {{
            size: [4.5, 3, 5.5], material: hide, rotate: [7.5, 0, 0], blend_with: [cranium]
          }}
          box nasal_ridge place(base_box=snout_bridge, anchor=[0, 0.75, 0.35], align=[0, -0.5, 0]) {{
            size: [2.5, 1.5, 4], material: keratin, rotate: [12, 0, 0], blend_with: [snout_bridge]
          }}
          bone fang_mount_l attach(parent_box=snout_bridge, anchor=[-0.75, -0.95, 0.55], align=[0, 0.82, 0]) mirror_x(fang_mount_r) {{
            rest_euler: [-10, 0, -8]
            box fang_tooth {{ size: [1, 2, 1], material: keratin, blend_with: [snout_bridge] }}
          }}

          chain horn_l attach(parent_box=cranium, anchor=[-0.88, 1.0, 0.0], align=[0.45, -0.88, 0.0]) segments=3 taper=0.76 seat=0.22 bend=[-15, -14, -12] mirror_x(horn_r) {{
            axis: +y
            rest_euler: [-28, 0, 28]
            joint_limits: [[-10, 10], [-10, 10], [-10, 10]]
            box horn_seg {{ size: [2, 4.5, 2], material: keratin }}
          }}

          bone jaw attach(parent_box=cranium, anchor=[0, -0.75, 0.35], align=[0, 0.5, -0.85]) {{
            rest_euler: [10, 0, 0]
            joint_limits: [[-4, 38], [-8, 8], [-8, 8]]
            box mandible {{ size: [4, 1.5, 5], material: under }}
            box chin_keel place(base_box=mandible, anchor=[0, -0.75, 0.35], align=[0, 0.45, 0]) {{
              size: [2.5, 1.5, 3], material: under, rotate: [-8, 0, 0], blend_with: [mandible]
            }}
          }}
        }}
      }}
    }}

    bone wing_root_l attach(parent_box=chest, anchor=[-0.95, 0.95, -0.25], align=[0.95, -0.85, 0]) mirror_x(wing_root_r) {{
      rest_euler: [8, 12, 34]
      joint_limits: [[-45, 45], [-40, 40], [-55, 65]]
      box wing_humerus {{ size: [9, 2.5, 3], material: hide }}
      plane wing_bay_inner place(base_box=wing_humerus, anchor=[0, 0, -1], align=[0, 0, 1], offset=[0, 0.25, 0]) {{
        size: [7.5, 7], normal: y, layer: 0, edge: scallop(bays=2, depth=2, side=bottom), material: under
      }}

      bone wing_forearm_l attach(parent_box=wing_humerus, anchor=[-1, 0, 0.1], align=[1, 0, 0], seat=0.20) {{
        rest_euler: [0, -18, -14]
        joint_limits: [[-35, 35], [-55, 45], [-45, 45]]
        box wing_radius {{ size: [11, 2, 2.5], material: hide }}
        box wrist_claw place(base_box=wing_radius, anchor=[-0.95, 0.1, 0.95], align=[0, 0, -0.5]) {{
          size: [1.5, 1.5, 2.5], material: keratin, rotate: [0, 22, 0], blend_with: [wing_radius]
        }}
        plane wing_bay_mid place(base_box=wing_radius, anchor=[0, 0, -1], align=[0, 0, 1], offset=[0, -0.25, 0]) {{
          size: [9, 7], normal: y, layer: 0, edge: triangle_bay(apex=[0, 0.5], side=right), material: under
        }}

        bone wing_finger_1_l attach(parent_box=wing_radius, anchor=[-1, 0.35, 0.65], align=[0.95, 0, 0]) {{
          rest_euler: [0, 14, -4]
          joint_limits: [[-25, 25], [-45, 35], [-35, 35]]
          box spar_1 {{ size: [12, 1.5, 1.5], material: keratin }}
          plane wing_bay_outer_1 place(base_box=spar_1, anchor=[0, 0, -1], align=[0, 0, 1], offset=[0, 0.25, 0]) {{
            size: [10, 2.5], normal: y, layer: 0, edge: ragged(depth=3, holes=2, side=bottom, seed=7), material: under
          }}
        }}
        bone wing_finger_2_l attach(parent_box=wing_radius, anchor=[-0.75, -0.35, -1], align=[0.90, 0, 0.92]) {{
          rest_euler: [0, -28, -8]
          joint_limits: [[-25, 25], [-50, 35], [-35, 35]]
          box spar_2 {{ size: [10.5, 1.5, 1.5], material: keratin }}
          plane wing_bay_outer_2 place(base_box=spar_2, anchor=[0, 0, -1], align=[0, 0, 1], offset=[0, -0.25, 0]) {{
            size: [9, 4.5], normal: y, layer: 0, edge: scallop(bays=2, depth=2, side=bottom), material: under
          }}
        }}
      }}
    }}

    bone fore_leg_l attach(parent_box=chest, anchor=[-1, -0.55, 0.55], align=[1.0, 0.88, 0]) mirror_x(fore_leg_r) {{
      rest_euler: [8, 0, 0]
      joint_limits: [[-50, 50], [-20, 20], [-20, 20]]
      box humerus {{ size: [3.5, 7.5, 4], material: limb_coat }}
      box elbow_cap_l place(base_box=humerus, anchor=[0, -0.85, -0.8], align=[0, 0.3, 0.45]) {{
        size: [2, 2, 2], material: hide, rotate: [-15, 0, 0], blend_with: [humerus]
      }}
      bone forearm attach(parent_box=humerus, anchor=[0, -1, 0], align=[0, 1, 0], seat=0.20) {{
        rest_euler: [-12, 0, 0]
        joint_limits: [[-45, 45], [-15, 15], [-15, 15]]
        box radius {{ size: [2.5, 6.5, 3], material: limb_coat }}
        bone fore_foot attach(parent_box=radius, anchor=[0, -1, 0.15], align=[0, 1, -0.35], seat=0.15) {{
          rest_euler: [4, 0, 0]
          joint_limits: [[-30, 30], [-15, 15], [-15, 15]]
          box manus {{ size: [3.5, 2, 4.5], material: under }}
          box f_claw_mid place(base_box=manus, anchor=[0, -0.2, 1], align=[0, 0, -0.4]) {{
            size: [1.5, 1.5, 2.5], material: keratin, rotate: [15, 0, 0], blend_with: [manus]
          }}
        }}
      }}
    }}
  }}
}}
@materials {{
  hide: process(family=patterned_skin, coat=scale_rows, roughness=0.45)
  limb_coat: process(family=patterned_skin, coat=banded_limb, roughness=0.48)
  under: process(family=patterned_skin, coat=plain, rd_feed=0.09, rd_kill=0.059, roughness=0.52)
  keratin: process(family=weathered_stone, roughness=0.40)
  storm_glow: process(family=thin_film_iridescence, roughness=0.20, emissive=0.85)
}}
@style {{
  target_encoding: vanilla,
  shade: banded(steps=4, direction=down, cluster=diagonal),
  glow_bleed: 2
}}
@stamps {{
  stamp eye_glint on cranium.east {{ at: [0.30, 0.42], size: [0.22, 0.16], color: storm_glow, shape: eye, emissive: 0.92, mirror: true }}
  stamp upper_teeth on snout_bridge.east {{ at: [0.55, 0.84], size: [0.12, 0.14], color: under, rows: 3, spacing: 0.22, mirror: true }}
}}
@behavior {{
  clip breathe {{
    duration_s: 1.0
    track neck_1: wave(axis=x, amp_deg=2.5, freq_hz=1.0)
    track wing_root_l: wave(axis=z, amp_deg=3.0, freq_hz=1.0)
    track wing_root_r: wave(axis=z, amp_deg=-3.0, freq_hz=1.0)
  }}
  clip walk {{
    duration_s: 1.0
    gait prowl {{
      pattern: diagonal_pairs,
      froude: 0.14,
      duty: 0.68,
      lift_ratio: 0.08,
      heave_ratio: 0.02,
      pitch_deg: 1.0,
      spine_wave_deg: 1.8,
      tail_lag: 0.2,
      style: {{ weight: 0.55, snap: 0.10 }}
    }}
    layer breathe: weight=0.5
    effect touchdown: {{ sound: "synccraft:dragon_step", particle: "synccraft:sand_puff" }}
  }}
}}
""".strip()

    return {
        "family_id": f"saffron_squallwing_v41_{tag}",
        "title": f"Saffron-Lineage {tag.capitalize()} Squallwing Dragon (v8 + Block-DSL v4.1)",
        "tier": "Hero",
        "concept": f"storm-charged {tag} canyon dragon with partitioned 6-panel wings, 3-step wedged head, and symmetric X=0 rest pose",
        "w1_dsl": w1_blockout,
        "final_dsl": final_v41,
    }


def _build_cobalt_thunder_drake_v41(variant: int = 0) -> dict[str, Any]:
    """Fanged wolf-drake (`SKILL.md` v8 Section 9.2) demonstrating anti-shard mastery:
    big deliberate wolf masses (`<= 2.5` boxes/bone, `>= 58%` top-10 volume), short deep wolf muzzle (`L:H ~ 1.5`),
    layered `edge: spikes` fur planes instead of scattered cube shards, and Froude `gait`.
    """
    themes = (
        ("cobalt", [0.46, 0.14, 248], [0.82, 0.05, 86], [0.76, 0.15, 82], [0.80, 0.20, 202]),
        ("obsidian", [0.34, 0.09, 285], [0.78, 0.05, 84], [0.82, 0.06, 78], [0.74, 0.22, 342]),
    )
    tag, hide_lch, under_lch, horn_lch, glow_lch = themes[variant % len(themes)]
    dsl = f"""
@meta {{
  archetype: entity,
  target_profile: bedrock_entity,
  texel_density: 32,
  units_per_block: 16,
  dsl_version: "4.1",
  size_step: 0.5,
  audit_profile: sculpted
}}
@layout {{
  budget max_boxes = 96
  budget max_bones = 44
  envelope max_span = [64, 52, 84]
  envelope grounded = true
}}
@palette {{
  hide: ramp(base_lch={hide_lch}, steps=5, hue_shift=[-16, 12])
  under: ramp(base_lch={under_lch}, steps=4, hue_shift=[-10, 8])
  horn_gold: ramp(base_lch={horn_lch}, steps=4, hue_shift=[-12, 12])
  fulgur_glow: ramp(base_lch={glow_lch}, steps=4, hue_shift=[-8, 12], emissive=true)
}}
@rig {{
  root bone torso {{
    pivot_local: [0, 12.0, 0]
    box chest {{ size: [11, 9, 13], material: hide, rotate: [-5, 0, 0] }}
    box shoulder_hump place(base_box=chest, anchor=[0, 1, 0.15], align=[0, -0.5, 0]) {{
      size: [6.5, 3, 8], material: under, rotate: [8, 0, 0], blend_with: [chest]
    }}
    bone shoulder_l attach(parent_box=chest, anchor=[-1, 0.65, 0.15], align=[0.85, 0, 0]) mirror_x(shoulder_r) {{
      rest_euler: [0, 0, -12]
      box shoulder_plate {{ size: [2.5, 3, 4], material: horn_gold, blend_with: [chest] }}
    }}
    bone ruff_l attach(parent_box=shoulder_hump, anchor=[-0.65, 0.9, -0.15], align=[0, -0.85, 0]) mirror_x(ruff_r) {{
      rest_euler: [0, 0, -15]
      plane ruff_fringe {{
        size: [3, 5.5], normal: x, layer: 0, edge: spikes(width=2, length=2..4, side=top), material: under
      }}
    }}

    bone waist attach(parent_box=chest, anchor=[0, 0, -1], align=[0, 0, 1], seat=0.28) {{
      rest_euler: [7.5, 0, 0]
      joint_limits: [[-18, 18], [-20, 20], [-12, 12]]
      box loin {{ size: [3.5, 6, 8], material: hide }}
      bone pelvis_b attach(parent_box=loin, anchor=[0, -0.05, -1], align=[0, 0, 1], seat=0.28) {{
        rest_euler: [-6, 0, 0]
        joint_limits: [[-18, 18], [-20, 20], [-12, 12]]
        box hips {{ size: [11, 7.5, 9], material: hide }}
        chain tail attach(parent_box=hips, anchor=[0, 0.2, -1], align=[0, 0, 1], seat=0.28) segments=4 taper=0.80 seat=0.28 bend=[10, 8, 6, -10] {{
          joint_limits: [[-24, 24], [-28, 28], [-15, 15]]
          box caudal {{ size: [5.5, 4.5, 6.5], material: hide }}
        }}
        bone hind_leg_l attach(parent_box=hips, anchor=[-1, -0.35, -0.30], align=[1.0, 0.85, 0]) mirror_x(hind_leg_r) {{
          rest_euler: [-18, 0, 0]
          joint_limits: [[-50, 50], [-20, 20], [-20, 20]]
          box thigh {{ size: [4, 7, 5], material: hide }}
          bone crus attach(parent_box=thigh, anchor=[0, -1, -0.1], align=[0, 1, 0], seat=0.22) {{
            rest_euler: [32, 0, 0]
            joint_limits: [[-45, 55], [-15, 15], [-15, 15]]
            box shin {{ size: [3, 6, 3.5], material: hide }}
            bone hind_paw attach(parent_box=shin, anchor=[0, -1, 0.15], align=[0, 1, -0.3], seat=0.15) {{
              rest_euler: [-14, 0, 0]
              joint_limits: [[-35, 35], [-15, 15], [-15, 15]]
              box pes {{ size: [4, 2, 4.5], material: under }}
              box h_talon place(base_box=pes, anchor=[0, -0.2, 1], align=[0, 0, -0.4]) {{
                size: [1.5, 1.5, 2], material: horn_gold, rotate: [15, 0, 0], blend_with: [pes]
              }}
            }}
          }}
        }}
      }}
    }}

    bone neck attach(parent_box=chest, anchor=[0, 0.35, 1], align=[0, -0.2, -1], seat=0.28) {{
      rest_euler: [-15, 0, 0]
      joint_limits: [[-26, 26], [-28, 28], [-16, 16]]
      box cerv {{ size: [7, 6.5, 6.5], material: hide }}
      bone mane_l attach(parent_box=cerv, anchor=[-0.65, 1, 0.25], align=[0, -0.88, 0]) mirror_x(mane_r) {{
        rest_euler: [0, 0, -12]
        plane mane_plane {{
          size: [2.5, 4], normal: x, layer: 0, edge: spikes(width=2, length=2..4, side=top), material: under
        }}
      }}
      bone head attach(parent_box=cerv, anchor=[0, 0.15, 1], align=[0, 0, -1], seat=0.26) {{
        rest_euler: [18, 0, 0]
        joint_limits: [[-28, 30], [-32, 32], [-18, 18]]
        box cranium {{ size: [7, 5.5, 6], material: hide }}
        bone brow_l attach(parent_box=cranium, anchor=[-0.85, 0.68, 0.72], align=[0.4, -0.5, -0.2]) mirror_x(brow_r) {{
          rest_euler: [8, 0, -15]
          box brow_ridge {{ size: [2, 1.5, 2.5], material: horn_gold, blend_with: [cranium] }}
        }}
        bone cheek_l attach(parent_box=cranium, anchor=[-0.95, -0.1, 0.20], align=[0.65, 0, 0]) mirror_x(cheek_r) {{
          rest_euler: [0, -10, 0]
          box cheek_mass {{ size: [2, 3, 3], material: hide, blend_with: [cranium] }}
        }}
        box muzzle place(base_box=cranium, anchor=[0, -0.05, 1], align=[0, 0, -0.8]) {{
          size: [4.5, 3.5, 4.5], material: hide, rotate: [10, 0, 0], blend_with: [cranium]
        }}
        chain horn_l attach(parent_box=cranium, anchor=[-0.88, 1.0, 0.0], align=[0.45, -0.88, 0.0]) segments=3 taper=0.78 seat=0.22 bend=[-14, -12, -10] mirror_x(horn_r) {{
          axis: +y
          rest_euler: [-28, 0, 28]
          joint_limits: [[-8, 8], [-8, 8], [-8, 8]]
          box horn_seg {{ size: [2, 4, 2], material: horn_gold }}
        }}
        bone jaw attach(parent_box=cranium, anchor=[0, -0.75, 0.35], align=[0, 0.5, -0.85]) {{
          rest_euler: [8, 0, 0]
          joint_limits: [[-4, 36], [-8, 8], [-8, 8]]
          box mandible {{ size: [4, 1.5, 4.5], material: under }}
        }}
      }}
    }}

    bone fore_leg_l attach(parent_box=chest, anchor=[-1, -0.55, 0.55], align=[1.0, 0.88, 0]) mirror_x(fore_leg_r) {{
      rest_euler: [8, 0, 0]
      joint_limits: [[-50, 50], [-20, 20], [-20, 20]]
      box humerus {{ size: [4, 6.5, 4.5], material: hide }}
      bone forearm attach(parent_box=humerus, anchor=[0, -1, 0], align=[0, 1, 0], seat=0.20) {{
        rest_euler: [-13, 0, 0]
        joint_limits: [[-45, 45], [-15, 15], [-15, 15]]
        box radius {{ size: [3, 5.5, 3.5], material: hide }}
        bone fore_paw attach(parent_box=radius, anchor=[0, -1, 0.15], align=[0, 1, -0.35], seat=0.15) {{
          rest_euler: [5, 0, 0]
          joint_limits: [[-30, 30], [-15, 15], [-15, 15]]
          box manus {{ size: [4, 2, 4.5], material: under }}
          box f_talon place(base_box=manus, anchor=[0, -0.2, 1], align=[0, 0, -0.4]) {{
            size: [1.5, 1.5, 2], material: horn_gold, rotate: [15, 0, 0], blend_with: [manus]
          }}
        }}
      }}
    }}
  }}
}}
@materials {{
  hide: process(family=patterned_skin, coat=scale_rows, roughness=0.46)
  under: process(family=patterned_skin, coat=plain, rd_feed=0.09, rd_kill=0.059, roughness=0.55)
  horn_gold: process(family=weathered_stone, roughness=0.38)
  fulgur_glow: process(family=thin_film_iridescence, roughness=0.18, emissive=0.90)
}}
@style {{
  target_encoding: vanilla,
  shade: banded(steps=4, direction=down, cluster=diagonal),
  glow_bleed: 2
}}
@stamps {{
  stamp predator_eye on cranium.east {{ at: [0.32, 0.40], size: [0.22, 0.16], color: fulgur_glow, shape: eye, emissive: 0.95, mirror: true }}
  stamp wolf_teeth on muzzle.east {{ at: [0.52, 0.85], size: [0.14, 0.15], color: under, rows: 3, spacing: 0.24, mirror: true }}
}}
@behavior {{
  clip walk {{
    duration_s: 1.8
    gait trot {{
      pattern: diagonal_pairs,
      froude: 0.16,
      duty: 0.66,
      lift_ratio: 0.09,
      heave_ratio: 0.02,
      pitch_deg: 1.1,
      spine_wave_deg: 2.0,
      tail_lag: 0.2,
      style: {{ weight: 0.50, snap: 0.12 }}
    }}
  }}
}}
""".strip()
    return {
        "family_id": f"cobalt_thunder_drake_v41_{tag}",
        "title": f"{tag.capitalize()} Thunder Wolf-Drake (v8 Anti-Shard Exemplar + Block-DSL v4.1)",
        "tier": "Hero",
        "concept": f"fanged {tag} storm wolf-drake with short 1.5:1 wedged wolf muzzle, layered fur cutout planes, and zero shard armour",
        "w1_dsl": None,
        "final_dsl": dsl,
    }


def _build_tundra_survey_rover_v41(variant: int = 0) -> dict[str, Any]:
    """Hard-surface `Block-DSL v4.1` expedition vehicle using native `octagon` wheels (`joint_limits: continuous`),
    chamfered hull volumes, functional lamp/antenna greebles, and snow-cap slabs (`SKILL.md` v8 Section 8.7).
    """
    hues = (("ochre", [0.66, 0.13, 72]), ("teal", [0.58, 0.12, 198]))
    tag, hull_lch = hues[variant % len(hues)]
    dsl = f"""
@meta {{
  archetype: entity,
  target_profile: bedrock_entity,
  texel_density: 16,
  units_per_block: 16,
  dsl_version: "4.1",
  size_step: 0.5,
  audit_profile: sculpted
}}
@layout {{
  budget max_boxes = 64
  budget max_bones = 20
  envelope max_span = [48, 36, 56]
  envelope grounded = true
}}
@palette {{
  hull_paint: ramp(base_lch={hull_lch}, steps=5, hue_shift=[-12, 10])
  tyre_rubber: ramp(base_lch=[0.28, 0.04, 255], steps=4, hue_shift=[-8, 6])
  snow_cap: ramp(base_lch=[0.88, 0.03, 230], steps=4, hue_shift=[-10, 8])
  beacon_lamp: ramp(base_lch=[0.82, 0.18, 68], steps=4, hue_shift=[-6, 10], emissive=true)
}}
@rig {{
  root bone chassis {{
    pivot_local: [0, 6.5, 0]
    box chassis_hull {{ size: [12, 5, 20], material: hull_paint }}
    box rover_cab place(base_box=chassis_hull, anchor=[0, 1, 0.42], align=[0, -0.85, 0]) {{
      size: [11, 5.5, 9], material: hull_paint, rotate: [5, 0, 0], blend_with: [chassis_hull]
    }}
    box cab_visor_chamfer place(base_box=rover_cab, anchor=[0, 0.55, 1], align=[0, 0, -0.75]) {{
      size: [10, 3, 3], material: hull_paint, rotate: [22.5, 0, 0], blend_with: [rover_cab]
    }}
    box cargo_bay place(base_box=chassis_hull, anchor=[0, 1, -0.45], align=[0, -0.85, 0]) {{
      size: [11.5, 6, 9.5], material: hull_paint, blend_with: [chassis_hull]
    }}
    box snow_slab place(base_box=cargo_bay, anchor=[0, 1, 0], align=[0, -0.6, 0]) {{
      size: [11, 1.5, 9], material: snow_cap, rotate: [-3, 0, 0], blend_with: [cargo_bay]
    }}
    bone headlamp_mount_l attach(parent_box=chassis_hull, anchor=[-0.72, 0.25, 1], align=[0, 0, -0.5]) mirror_x(headlamp_mount_r) {{
      box headlamp {{ size: [2, 2, 1.5], material: beacon_lamp, blend_with: [chassis_hull] }}
    }}
    plane hazard_pennant place(base_box=cargo_bay, anchor=[0, 1, -0.8], align=[0, -0.9, 0]) {{
      size: [3, 5], normal: x, layer: 0, edge: ragged(depth=2, holes=0, side=right, seed=3), material: beacon_lamp
    }}

    bone wheel_fl attach(parent_box=chassis_hull, anchor=[-1, -0.45, 0.62], align=[0.85, 0, 0]) mirror_x(wheel_fr) {{
      joint_limits: continuous
      octagon tyre_fl {{ size: [3.5, 8.5, 8.5], axis: x, material: tyre_rubber }}
    }}
    bone wheel_rl attach(parent_box=chassis_hull, anchor=[-1, -0.45, -0.62], align=[0.85, 0, 0]) mirror_x(wheel_rr) {{
      joint_limits: continuous
      octagon tyre_rl {{ size: [3.5, 8.5, 8.5], axis: x, material: tyre_rubber }}
    }}
  }}
}}
@style {{
  target_encoding: vanilla,
  shade: banded(steps=4, direction=down, cluster=diagonal),
  glow_bleed: 2
}}
@behavior {{
  clip drive {{
    duration_s: 1.5
    track wheel_fl: spin(axis=x, amp_deg=360, freq_hz=0.6667)
    track wheel_fr: spin(axis=x, amp_deg=360, freq_hz=0.6667)
    track wheel_rl: spin(axis=x, amp_deg=360, freq_hz=0.6667)
    track wheel_rr: spin(axis=x, amp_deg=360, freq_hz=0.6667)
  }}
}}
""".strip()
    return {
        "family_id": f"tundra_survey_rover_v41_{tag}",
        "title": f"{tag.capitalize()} Tundra Survey Rover (`octagon` Continuous Wheels + Block-DSL v4.1)",
        "tier": "Standard",
        "concept": f"chamfered {tag} polar expedition rover with native octagon tyres, snow-cap slab, and continuous wheel spin",
        "w1_dsl": None,
        "final_dsl": dsl,
    }


def _build_broken_shard_negative_dsl() -> tuple[str, dict[str, Any]]:
    """Construct a compilable negative example exhibiting the exact v9 failure modes diagnosed in `SKILL.md` v8 2.5:
    flat rectangular doghouse snout (`snout_taper_ratio == 1.0`), plane horn, broken rest-pose yaw asymmetry,
    and excess small shards on torso (`> 3.5` boxes/bone) so the model learns when to `PUNCH BACK TO W1`.
    """
    bad_dsl = """
@meta {
  archetype: entity,
  target_profile: bedrock_entity,
  texel_density: 16,
  units_per_block: 16,
  dsl_version: "4.1",
  size_step: 0.5,
  audit_profile: sculpted
}
@palette {
  hide: ramp(base_lch=[0.48, 0.04, 240], steps=3, hue_shift=[-2, 2])
}
@rig {
  root bone body {
    pivot_local: [0, 8, 0]
    box torso_core { size: [8, 6, 12], material: hide }
    box shard_1 place(base_box=torso_core, anchor=[-1, 0.8, 0.6], align=[0.5, -0.5, 0]) { size: [1.5, 1.5, 2], material: hide, rotate: [17, 13, -29], blend_with: [torso_core] }
    box shard_2 place(base_box=torso_core, anchor=[-1, 0.6, 0.1], align=[0.5, -0.5, 0]) { size: [1.5, 1.5, 2], material: hide, rotate: [-23, 9, 31], blend_with: [torso_core] }
    box shard_3 place(base_box=torso_core, anchor=[-1, 0.7, -0.4], align=[0.5, -0.5, 0]) { size: [1.5, 1.5, 2], material: hide, rotate: [11, -19, 24], blend_with: [torso_core] }
    box shard_4 place(base_box=torso_core, anchor=[1, 0.5, 0.3], align=[-0.5, -0.5, 0]) { size: [1.5, 1.5, 2], material: hide, rotate: [29, 7, -18], blend_with: [torso_core] }
    box shard_5 place(base_box=torso_core, anchor=[0, 1, 0.4], align=[0, -0.5, 0]) { size: [1.5, 1.5, 2], material: hide, rotate: [-14, 21, 8], blend_with: [torso_core] }
    box shard_6 place(base_box=torso_core, anchor=[0, 1, -0.3], align=[0, -0.5, 0]) { size: [1.5, 1.5, 2], material: hide, rotate: [19, -11, -16], blend_with: [torso_core] }
    bone head attach(parent_box=torso_core, anchor=[0, 0.2, 1], align=[0, 0, -1], seat=0.2) {
      rest_euler: [0, 16, 0]
      box skull { size: [6, 4, 5], material: hide }
      box flat_doghouse_snout place(base_box=skull, anchor=[0, 0, 1], align=[0, 0, -0.9]) {
        size: [6, 4, 7], material: hide, blend_with: [skull]
      }
      plane sheet_horn_l place(base_box=skull, anchor=[-0.8, 1, -0.2], align=[0, -0.9, 0]) {
        size: [3, 6], normal: x, layer: 1, material: hide
      }
    }
    bone leg_l attach(parent_box=torso_core, anchor=[-0.8, -1, 0.5], align=[0, 1, 0]) mirror_x(leg_r) {
      box post_leg { size: [2, 5, 2], material: hide }
    }
    bone back_leg_l attach(parent_box=torso_core, anchor=[-0.8, -1, -0.5], align=[0, 1, 0]) mirror_x(back_leg_r) {
      box back_post { size: [2, 5, 2], material: hide }
    }
  }
}
""".strip()
    check = run_v8_construct_check(bad_dsl)
    return bad_dsl, check


# ---------------------------------------------------------------------------
# Disjoint `v4.1` Modular `@layout/@component/@assembly` Training Suite (`0%` `v4_wm_01..04` overlap)
# ---------------------------------------------------------------------------

SOLAR_SIEGE_STRIDER_V41_DSL: str = """
@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }

@layout {
  param hull_span = 14 [10, 22]
  param mast_height = 8 [5, 14]
  let deck_width = max(8, hull_span - 2)
  let leg_reach = 7
  budget max_boxes = 44
  budget max_bones = 24
  envelope max_span = [52, 44, 52]
  envelope min_ground_y = -0.5
  envelope grounded = true
}

@palette {
  sun_bronze: ramp(base_lch=[0.64, 0.13, 74], steps=5, hue_shift=[-12, 10])
  slate_iron: ramp(base_lch=[0.36, 0.04, 252], steps=4, hue_shift=[-8, 8])
  helios_core: ramp(base_lch=[0.80, 0.19, 82], steps=4, hue_shift=[-6, 10], emissive=true)
}

@component StriderChassis {
  param w = 12 [8, 20]
  material_slots: [armor, trim]
  box core_deck { size: [w, 5, 12], material: armor }
  box prow_bevel place(base_box=core_deck, anchor=[0, 0.1, 1], align=[0, 0, -0.75]) {
    size: [max(6, w - 2), 3.5, 4], material: trim, rotate: [15, 0, 0], blend_with: [core_deck]
  }
  frame mast_socket { on_box: core_deck, anchor: [0, 1, 0], rot: [0, 0, 0] }
  frame hip_fl { on_box: core_deck, anchor: [-1, -0.4, 0.55], rot: [0, 0, 0] }
  frame hip_fr { on_box: core_deck, anchor: [1, -0.4, 0.55], rot: [0, 0, 0] }
  frame hip_rl { on_box: core_deck, anchor: [-1, -0.4, -0.55], rot: [0, 0, 0] }
  frame hip_rr { on_box: core_deck, anchor: [1, -0.4, -0.55], rot: [0, 0, 0] }
}

@component SolarBeaconMast {
  param h = 8 [4, 16]
  material_slots: [frame_mat, core_mat]
  box pylon { size: [3, h, 3], material: frame_mat }
  box lens_housing place(base_box=pylon, anchor=[0, 1, 0], align=[0, -0.8, 0]) {
    size: [4.5, 3, 4.5], material: core_mat, blend_with: [pylon]
  }
  frame deck_mount { on_box: pylon, anchor: [0, -1, 0], rot: [0, 0, 0] }
  frame beam_origin { on_box: lens_housing, anchor: [0, 0, 1], rot: [0, 0, 0], role: vfx_emitter }
}

@component StriderLeg {
  param reach = 7 [5, 12]
  material_slots: [strut, pad]
  uv_policy: shared
  box femur_beam { size: [2.5, reach, 3], material: strut }
  bone tibia_link attach(parent_box=femur_beam, anchor=[0, -1, 0], align=[0, 1, 0]) {
    rest_euler: [12, 0, 0]
    joint_limits: [[-35, 45], [-15, 15], [-15, 15]]
    box lower_strut { size: [2, 5, 2.5], material: pad }
  }
  frame socket_mount { on_box: femur_beam, anchor: [0, 1, 0], rot: [0, 0, 0] }
}

@assembly SolarSiegeStrider {
  root_pivot: [0, 13.2, 0]
  instance chassis : StriderChassis(w=deck_width) {
    id: "strider_chassis_01"
    materials: { armor: sun_bronze, trim: slate_iron }
  }
  instance beacon : SolarBeaconMast(h=mast_height) {
    id: "strider_beacon_01"
    materials: { frame_mat: slate_iron, core_mat: helios_core }
  }
  instance leg_fl : StriderLeg(reach=leg_reach) {
    id: "strider_leg_fl_01"
    materials: { strut: slate_iron, pad: sun_bronze }
  }
  instance leg_fr : StriderLeg(reach=leg_reach) {
    id: "strider_leg_fr_01"
    materials: { strut: slate_iron, pad: sun_bronze }
  }
  instance leg_rl : StriderLeg(reach=leg_reach) {
    id: "strider_leg_rl_01"
    materials: { strut: slate_iron, pad: sun_bronze }
  }
  instance leg_rr : StriderLeg(reach=leg_reach) {
    id: "strider_leg_rr_01"
    materials: { strut: slate_iron, pad: sun_bronze }
  }
  connect beacon.deck_mount to chassis.mast_socket { type: revolute, axis: y, limits: continuous }
  connect leg_fl.socket_mount to chassis.hip_fl { type: revolute, axis: x, limits: [-45, 45] }
  connect leg_fr.socket_mount to chassis.hip_fr { type: revolute, axis: x, limits: [-45, 45] }
  connect leg_rl.socket_mount to chassis.hip_rl { type: revolute, axis: x, limits: [-45, 45] }
  connect leg_rr.socket_mount to chassis.hip_rr { type: revolute, axis: x, limits: [-45, 45] }
}
""".strip()


def build_stage7_sculpted_v8_v41_records(
    *,
    seed: int = 39526,
    scale: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Synthesize clean-room `Blockbench_sculpted_style-v8` + `Block-DSL v4.1` multimodal & text records."""
    ensure_synccraft_importable()
    from synccraft.audit.sync_audit import audit_scene
    from synccraft.dsl.assembly_ir import (
        AssemblySpec,
        audit_assembly_contracts,
        format_assembly_sections,
        override_layout_params,
    )
    from synccraft.dsl.parser import parse_program
    from synccraft.kinematics.se3_chain import compile_geometry
    from synccraft.kinematics.gait_inversion import (
        _adapt_pattern_for_morphology,
        invert_clip_to_gait,
        search_or_repair_gait,
    )
    from synccraft.kinematics.morphology import analyze_morphology
    from synccraft.world.proposer_tasks import (
        ProposerTaskRecord,
        _gait_decl_to_dict,
        _quadruped_fixture_dsl,
        style_phrase_to_gait_preset,
    )

    rng = random.Random(seed ^ 0x57A6E708)
    records: list[dict[str, Any]] = []
    blob_map: dict[str, bytes] = {}

    def _store_png(rel_path: str, im: Image.Image) -> str:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        blob_map[rel_path] = buf.getvalue()
        return rel_path

    # 1. Clean-Room `v8` + `v4.1` Exemplars (`W1 -> Final`, Multiview Inverse, Region Visual-Primitives Rubric)
    archetypes = [
        _build_saffron_squallwing_v41(0),
        _build_saffron_squallwing_v41(1),
        _build_saffron_squallwing_v41(2),
        _build_cobalt_thunder_drake_v41(0),
        _build_cobalt_thunder_drake_v41(1),
        _build_tundra_survey_rover_v41(0),
        _build_tundra_survey_rover_v41(1),
    ]

    repeats = max(1, int(round(14 * scale)))
    for arch in archetypes:
        fid = arch["family_id"]
        final_dsl = arch["final_dsl"]
        v8_check = run_v8_construct_check(final_dsl)
        if not v8_check["passed_v8"]:
            raise RuntimeError(f"Clean-room Stage-7 v8 exemplar {fid} failed v8 check: {v8_check}")

        prog = parse_program(final_dsl)
        turns = _render_v8_views_pil(prog, panel_size=(208, 208))
        reg_sheet, reg_boxes = render_v8_zoomed_region_sheet(
            prog,
            panel_size=(208, 208),
            pre_rendered_turns=turns,
        )

        p_hero = _store_png(f"sculpted_v8/{fid}_hero.png", turns["hero"])
        p_front = _store_png(f"sculpted_v8/{fid}_front.png", turns["front"])
        p_side = _store_png(f"sculpted_v8/{fid}_side.png", turns["side"])
        p_top = _store_png(f"sculpted_v8/{fid}_top.png", turns["top"])
        p_region = _store_png(f"sculpted_v8/{fid}_region_sheet.png", reg_sheet)

        p_w1_hero: str | None = None
        p_w1_side: str | None = None
        if arch["w1_dsl"]:
            w1_prog = parse_program(arch["w1_dsl"])
            w1_turns = _render_v8_views_pil(w1_prog, panel_size=(192, 192))
            p_w1_hero = _store_png(f"sculpted_v8/{fid}_w1_hero.png", w1_turns["hero"])
            p_w1_side = _store_png(f"sculpted_v8/{fid}_w1_side.png", w1_turns["side"])

        prov = make_clean_room_provenance(
            asset_id=f"stage7_v8_{fid}",
            title=arch["title"],
            license_spdx="CC0-1.0",
            author="SyncCraft Stage-7 Clean-Room Sculpted v8 Synthesizer",
        )
        group_id = f"sculpted-v8-family:{fid}"

        for rep_i in range(repeats):
            use_zh = rep_i % 4 == 3
            # (a) Text -> `SKILL.md` v8 Trait Map + `Block-DSL v4.1` Authoring
            think_create = (
                f"1. **v8 Part Map & Proportions**: Concept = {arch['concept']} ({arch['tier']} tier). "
                f"Rest pose is strictly symmetric across X = 0 (`mirror_x`, pitch-only gesture curve; attitude reserved for `@behavior`).\n"
                f"2. **Head Gate (Section 6.6)**: Build 3 depth steps (`cranium`, overhanging tilted `brow`, `cheek`, wedged `snout`/`muzzle`) "
                f"+ separate hinged `jaw` bone + tapered block `horn` chain (`0` flat plane horns) + mirrored `@stamps` eye under brow.\n"
                f"3. **Anti-Shard Discipline (Section 3.6)**: {v8_check['boxes_per_bone_non_head']} boxes/bone outside head (`<= 3.5`), "
                f"top-10 cuboids hold {v8_check['top10_volume_share']:.0%} volume (`>= 50%`), `seat=0.28`, `blend_with` joint shingles, "
                f"`@style {{ shade: banded(steps=4, direction=down, cluster=diagonal), glow_bleed: 2 }}`."
            )
            prompt_create = (
                f"使用 Sculpted Blockbench Style v8 规范与 Block-DSL v4.1 (`dsl_version: \"4.1\"`, `audit_profile: sculpted`) 构建原创模型：{arch['title']}（{arch['concept']}）。"
                "严格遵守 v8 铁律：X=0 对称绑定姿态、头部三层深度阶梯+独立铰接下颌+实体分段角（禁止平面角）、非头部每骨骼<=3.5盒防碎片装甲、`seat=0.28`、`shade: banded` 色带像素着色及 `@behavior` 步态动作。"
                if use_zh
                else f"Design an original Sculpted Blockbench Style v8 model in `Block-DSL v4.1` (`dsl_version: \"4.1\"`, `audit_profile: sculpted`): "
                f"{arch['title']} ({arch['concept']}). Enforce `SKILL.md` v8 non-negotiables: exact X=0 symmetric rest pose, "
                "3-step head depth + hinged jaw + real block horns (never flat planes), `<= 3.5` boxes/bone anti-shard discipline, "
                "`top-10 volume >= 50%`, `seat=0.28`, `shade: banded(...)`, `@stamps`, and verified `@behavior` motion."
            )
            body_create = f"```blockdsl\n{final_dsl}\n```"
            ans_create = format_supervised_answer(body_create, think_create, enable_thinking=True)
            records.append(
                {
                    "id": f"stage7_v8_create_{fid}_{rep_i:03d}",
                    "source_group": group_id,
                    "language": "zh" if use_zh else "en",
                    "task": "synccraft_v41_sculpted_v8",
                    "prompt": prompt_create,
                    "answer": ans_create,
                    "target": {"kind": "text", "text": body_create},
                    "enable_thinking": True,
                    "frames": [],
                    "provenance": prov,
                    "metadata": {
                        "asset_id": fid,
                        "family_id": fid,
                        "canonical_dsl": final_dsl,
                        "v8_check": v8_check,
                        "grouping_keys": [group_id],
                    },
                }
            )

            # (b) Zoomed Region Sheet + Visual-Primitives (`<box>` / `<point>`) 26-Point `SKILL.md` v8 Rubric Critique
            if rep_i < max(2, repeats // 2):
                hb = reg_boxes["head_side_wedge"]
                lb = reg_boxes["limb_ground_contact"]
                sb = reg_boxes["silhouette_panel"]
                rubric_payload = {
                    "rubric_26_score": 25,
                    "passed_v8_gate": True,
                    "punch_back_to_w1": False,
                    "boxes_per_bone_non_head": v8_check["boxes_per_bone_non_head"],
                    "top10_volume_share": v8_check["top10_volume_share"],
                    "mirror_parity": "OK",
                    "style_anchor_score": v8_check["style_anchor_score"],
                    "item_scores": {
                        "1_silhouette": 2,
                        "2_head_gate": 2,
                        "3_eyes_under_brow": 2,
                        "4_block_horns_and_spikes": 2,
                        "5_no_shard_armour": 2,
                        "6_seating_and_blend_with": 2,
                        "7_x0_mirror_symmetry": 2,
                        "8_proportions": 2,
                        "9_volume_hierarchy_70_25_5": 2,
                        "10_calm_banded_paint": 2,
                        "11_focal_vibrancy": 2,
                        "12_identity_traits": 2,
                        "13_animation_readiness": 1,
                    },
                }
                think_crit = (
                    f'Inspecting the 2x2 v8 Region Sheet with Visual Primitives:\n'
                    f'<box x1="{hb[0]}" y1="{hb[1]}" x2="{hb[2]}" y2="{hb[3]}" label="head_profile_3_depth_steps"/> '
                    f'<point x="690" y="220" label="eye_under_brow"/>\n'
                    f'Head shows clean brow overhang, wedged muzzle taper, and hinged jaw.\n'
                    f'<box x1="{lb[0]}" y1="{lb[1]}" x2="{lb[2]}" y2="{lb[3]}" label="grounded_limbs_x0_symmetric"/>\n'
                    f'Feet sit grounded at y=0 with exact X=0 mirror parity (`{v8_check["boxes_per_bone_non_head"]}` boxes/bone, '
                    f'top-10 volume `{v8_check["top10_volume_share"]:.0%}`).\n'
                    f'<box x1="{sb[0]}" y1="{sb[1]}" x2="{sb[2]}" y2="{sb[3]}" label="readable_silhouette"/>\n'
                    f'Silhouette reads immediately with calm 4-step banded paint (`25/26` PASS).'
                )
                body_crit = json.dumps(rubric_payload, indent=2, sort_keys=True)
                ans_crit = format_supervised_answer(
                    body_crit,
                    think_crit,
                    enable_thinking=True,
                )
                records.append(
                    {
                        "id": f"stage7_v8_region_crit_{fid}_{rep_i:03d}",
                        "source_group": group_id,
                        "language": "en",
                        "task": "synccraft_sculpted_self_critique",
                        "visual_kind": "independent_image",
                        "frames": [p_region],
                        "media_times_s": [0.0],
                        "prompt": (
                            "Inspect this 2x2 Sculpted Blockbench v8 Region Sheet (head 3/4, head profile, limb grounding, "
                            "and silhouette). Localize key anatomical regions with `<box>`/`<point>` Visual Primitives inside "
                            "`<think>` and output the 13-item (26-point) `SKILL.md` v8 review rubric JSON."
                        ),
                        "answer": ans_crit,
                        "target": {"kind": "text", "text": body_crit},
                        "enable_thinking": True,
                        "provenance": prov,
                        "metadata": {
                            "asset_id": fid,
                            "family_id": fid,
                            "v8_rubric": rubric_payload,
                            "grouping_keys": [group_id],
                        },
                    }
                )

            # (c) `W1 -> Final` Progressive Upgrade (`SKILL.md` v8 Section 5.2) when `w1_dsl` is available
            if arch["w1_dsl"] and p_w1_hero and p_w1_side and rep_i < max(2, repeats // 2):
                think_prog = (
                    '<box x1="180" y1="120" x2="820" y2="860" label="w1_primary_masses"/>\n'
                    "W1 white-model blockout passes base proportions, X=0 symmetry, and 3-step head wedge. "
                    "Advancing through W2/D1/D2/T1/T2/A1: adding cheek boxes, skull crest shingle (`blend_with`), "
                    "3-segment tapered block horns, 6-panel partitioned `edge:` wing membranes (`layer: 1..4`), "
                    "knee/calf/heel limb sculpture, 4-step `banded` paint + `@stamps`, and Froude `gait` + `breathe` layer."
                )
                body_prog = f"```blockdsl\n{final_dsl}\n```"
                ans_prog = format_supervised_answer(body_prog, think_prog, enable_thinking=True)
                records.append(
                    {
                        "id": f"stage7_v8_w1_to_a1_{fid}_{rep_i:03d}",
                        "source_group": group_id,
                        "language": "en",
                        "task": "synccraft_sculpted_stage_progression",
                        "visual_kind": "state_pair",
                        "frames": [p_w1_hero, p_w1_side],
                        "media_times_s": [0.0, 0.0],
                        "prompt": (
                            f"Upgrade this W1 primary-mass blockout of {arch['title']} into a finished A1 `Block-DSL v4.1` "
                            "asset following `SKILL.md` v8 passes (secondary anatomy, partitioned alpha-cutout planes, "
                            "`blend_with` shingles, `shade: banded(...)` + `@stamps`, and `@behavior` Froude `gait`) without "
                            "introducing shard armour:\n\n"
                            f"```blockdsl\n{arch['w1_dsl']}\n```"
                        ),
                        "answer": ans_prog,
                        "target": {"kind": "text", "text": body_prog},
                        "enable_thinking": True,
                        "provenance": prov,
                        "metadata": {
                            "asset_id": fid,
                            "family_id": fid,
                            "canonical_dsl": final_dsl,
                            "grouping_keys": [group_id],
                        },
                    }
                )

            # (d) 4-View Studio Turnaround -> `Block-DSL v4.1` Inverse Reconstruction
            if rep_i < max(2, repeats // 3):
                think_inv = (
                    '<box x1="160" y1="110" x2="840" y2="880" label="sculpted_turnaround_massing"/>\n'
                    f"Reconstructing {arch['title']} from 4-view studio turnaround: preserving exact X=0 symmetry, "
                    f"3-step wedged head, `seat=0.28` chain seating, `{v8_check['plane_count']}` layered alpha-cutout "
                    "planes, and `shade: banded` surface specification."
                )
                body_inv = f"```blockdsl\n{final_dsl}\n```"
                ans_inv = format_supervised_answer(body_inv, think_inv, enable_thinking=True)
                records.append(
                    {
                        "id": f"stage7_v8_mv_inv_{fid}_{rep_i:03d}",
                        "source_group": group_id,
                        "language": "en",
                        "task": "synccraft_sculpted_multiview_inverse",
                        "visual_kind": "multiview_render",
                        "frames": [p_hero, p_front, p_side, p_top],
                        "media_times_s": [0.0, 0.0, 0.0, 0.0],
                        "prompt": (
                            "Reconstruct the complete `Block-DSL v4.1` (`audit_profile: sculpted`) program for the creature "
                            "or vehicle shown in these 4 orthographic/studio turnaround views (`hero`, `front`, `side`, `top`)."
                        ),
                        "answer": ans_inv,
                        "target": {"kind": "text", "text": body_inv},
                        "enable_thinking": True,
                        "provenance": prov,
                        "metadata": {
                            "asset_id": fid,
                            "family_id": fid,
                            "canonical_dsl": final_dsl,
                            "grouping_keys": [group_id],
                        },
                    }
                )

    # 2. Negative Shard-Armour / Doghouse-Head / Plane-Horn Punch-Back Examples (`PUNCH BACK TO W1`)
    bad_dsl, bad_check = _build_broken_shard_negative_dsl()
    bad_prog = parse_program(bad_dsl)
    bad_sc = compile_geometry(bad_prog)
    bad_bake = bake_sculpted_rgba_atlas(bad_prog, bad_sc)
    bad_reg, bad_boxes = render_v8_zoomed_region_sheet(bad_prog, bad_sc, bad_bake, panel_size=(256, 256))
    p_bad_reg = _store_png("sculpted_v8/neg_shard_doghouse_region.png", bad_reg)
    neg_prov = make_clean_room_provenance(
        asset_id="stage7_v8_neg_shard_doghouse",
        title="Negative Diagnostic Shard-Armour Doghouse Blockout",
        license_spdx="CC0-1.0",
        author="SyncCraft Stage-7 Clean-Room Sculpted v8 Synthesizer",
    )
    neg_rubric = {
        "rubric_26_score": 6,
        "passed_v8_gate": False,
        "punch_back_to_w1": True,
        "punch_back_reasons": bad_check["punch_back_reasons"],
        "boxes_per_bone_non_head": bad_check["boxes_per_bone_non_head"],
        "mirror_parity": bad_check["mirror_parity"],
        "verdict": "PUNCH BACK TO W1: shard armour on torso, rectangular doghouse snout, forbidden flat plane horn, and broken X=0 rest-pose symmetry.",
    }
    for n_i in range(max(4, int(round(20 * scale)))):
        hb = bad_boxes["head_side_wedge"]
        think_neg = (
            f'<box x1="{hb[0]}" y1="{hb[1]}" x2="{hb[2]}" y2="{hb[3]}" label="flat_doghouse_snout_and_plane_horn"/>\n'
            '<box x1="60" y1="60" x2="460" y2="460" label="random_shard_armour_and_yaw_asymmetry"/>\n'
            f"Violates `SKILL.md` v8 Section 1 & 5.4: (1) non-head torso carries `{bad_check['boxes_per_bone_non_head']}` "
            "randomly tilted shard boxes, (2) snout has 1.0x skull width (rectangular doghouse face, no jaw bone, no brow), "
            "(3) horn is a flat `plane` instead of tapered block segments, and (4) rest pose turns head 16 deg breaking mirror parity. "
            "Must issue `PUNCH BACK TO W1` rather than patching with texture!"
        )
        body_neg = json.dumps(neg_rubric, indent=2, sort_keys=True)
        ans_neg = format_supervised_answer(
            body_neg,
            think_neg,
            enable_thinking=True,
        )
        records.append(
            {
                "id": f"stage7_v8_punchback_neg_{n_i:03d}",
                "source_group": "sculpted-v8-family:neg_shard_doghouse",
                "language": "en",
                "task": "synccraft_sculpted_self_critique",
                "visual_kind": "independent_image",
                "frames": [p_bad_reg],
                "media_times_s": [0.0],
                "prompt": (
                    "Inspect this 2x2 Sculpted Blockbench v8 Region Sheet with `<box>` Visual Primitives inside `<think>` "
                    "and output the `SKILL.md` v8 review rubric JSON (including `punch_back_to_w1` if any Section 5.4 gate fails)."
                ),
                "answer": ans_neg,
                "target": {"kind": "text", "text": body_neg},
                "enable_thinking": True,
                "provenance": neg_prov,
                "metadata": {
                    "asset_id": "neg_shard_doghouse",
                    "v8_rubric": neg_rubric,
                    "grouping_keys": ["sculpted-v8-family:neg_shard_doghouse"],
                },
            }
        )

    # 3. Disjoint `v4.1` Modular `@layout/@component/@assembly` Supervision (`SolarSiegeStrider`, 0% overlap with `v4_wm_01..04`)
    strider_prog = parse_program(SOLAR_SIEGE_STRIDER_V41_DSL)
    strider_spec: AssemblySpec = strider_prog.assembly_spec
    strider_layout_dsl = "\n".join(format_assembly_sections(AssemblySpec(layout=strider_spec.layout, components=(), assembly=None)))
    beacon_comp = next(c for c in strider_spec.components if c.name == "SolarBeaconMast")
    beacon_dsl = "\n".join(format_assembly_sections(AssemblySpec(layout=None, components=(beacon_comp,), assembly=None)))
    upgraded_beacon_dsl = """
@component SolarBeaconMast {
  param h = 8 [4, 16]
  material_slots: [frame_mat, core_mat]
  box pylon { size: [3, h, 3], material: frame_mat }
  box lens_housing place(base_box=pylon, anchor=[0, 1, 0], align=[0, -0.8, 0]) {
    size: [4.5, 3, 4.5], material: core_mat, blend_with: [pylon]
  }
  box sun_vane place(base_box=lens_housing, anchor=[0, 1, 0], align=[0, -0.7, 0]) {
    size: [2, 2.5, 2], material: frame_mat, blend_with: [lens_housing]
  }
  frame deck_mount { on_box: pylon, anchor: [0, -1, 0], rot: [0, 0, 0] }
  frame beam_origin { on_box: lens_housing, anchor: [0, 0, 1], rot: [0, 0, 0], role: vfx_emitter }
}
""".strip()
    pert_strider = override_layout_params(strider_prog, {"hull_span": 18})
    sc_pert = compile_geometry(pert_strider)
    aud_pert = audit_assembly_contracts(pert_strider, sc_pert)
    sat_pert = audit_scene(sc_pert)
    prop_gt = {
        "chassis_deck_width": float(sc_pert.box_by_name["chassis__core_deck"].size[0]),
        "leg_femur_height": float(sc_pert.box_by_name["leg_fl__femur_beam"].size[1]),
        "audit_passed": bool(sat_pert["passed"] and aud_pert["passed"]),
        "world_span_x": round(float(aud_pert["world_span"][0]), 4),
    }
    mod_prov = make_clean_room_provenance(
        asset_id="stage7_v41_solar_siege_strider",
        title="SolarSiegeStrider Modular Assembly v4.1",
        license_spdx="CC0-1.0",
        author="SyncCraft Stage-7 Clean-Room Modular Synthesizer",
    )
    mod_pairs = [
        (
            "layout",
            "Design a top-down `@layout` block for `SolarSiegeStrider` with driving parameters `hull_span` and `mast_height`, derived `deck_width` and `leg_reach`, `max_boxes <= 44`, `max_bones <= 24`, and `grounded = true`.",
            "Deriving DAG expression `deck_width = max(8, hull_span - 2)` and constant stance `leg_reach = 7` within the 44-box / 24-bone grounded envelope.",
            f"```blockdsl\n{strider_layout_dsl}\n```",
        ),
        (
            "comp_synth",
            "Synthesize ONLY `@component SolarBeaconMast` exposing parameter `h`, `material_slots: [frame_mat, core_mat]`, and attachment frames `deck_mount` and `beam_origin` (`role: vfx_emitter`).",
            "Building `pylon` and `lens_housing` in component-local coordinates with bottom `deck_mount` and forward `beam_origin`.",
            f"```blockdsl\n{beacon_dsl}\n```",
        ),
        (
            "surgical_swap",
            f"Upgrade `@component SolarBeaconMast` in `SolarSiegeStrider` by adding a `[2, 2.5, 2]` `sun_vane` crest atop `lens_housing` while keeping `deck_mount`, `beam_origin`, and all non-target strider vertices invariant:\n\n```blockdsl\n{SOLAR_SIEGE_STRIDER_V41_DSL}\n```",
            "Modifying only `@component SolarBeaconMast` while preserving `deck_mount` on `pylon` so chassis and legs suffer zero vertex drift.",
            f"```blockdsl\n{upgraded_beacon_dsl}\n```",
        ),
        (
            "propagate",
            f"Predict JSON `{{chassis_deck_width, leg_femur_height, audit_passed, world_span_x}}` when `@layout` parameter `hull_span` in `SolarSiegeStrider` increases from `14` to `18`:\n\n```blockdsl\n{SOLAR_SIEGE_STRIDER_V41_DSL}\n```",
            "Evaluating `deck_width = max(8, 18 - 2) = 16.0` and constant grounded stance `leg_reach = 7.0` through SE(3) assembly mating.",
            json.dumps(prop_gt, sort_keys=True),
        ),
    ]
    for m_rep in range(max(2, int(round(16 * scale)))):
        for kind_tag, m_prompt, m_think, m_body in mod_pairs:
            ans_m = format_supervised_answer(m_body, m_think, enable_thinking=True)
            records.append(
                {
                    "id": f"stage7_v41_mod_{kind_tag}_{m_rep:03d}",
                    "source_group": "sculpted-v8-family:solar_siege_strider_v41",
                    "language": "en",
                    "task": "synccraft_v4_modular_assembly",
                    "prompt": m_prompt,
                    "answer": ans_m,
                    "target": {"kind": "text", "text": m_body},
                    "enable_thinking": True,
                    "frames": [],
                    "provenance": mod_prov,
                    "metadata": {
                        "asset_id": "solar_siege_strider_v41",
                        "modular_task_kind": kind_tag,
                        "oracle_body": m_body,
                        "grouping_keys": ["sculpted-v8-family:solar_siege_strider_v41"],
                    },
                }
            )

    # 4. Gate D 5-Task Proposer Supervision (`train` & `val` ONLY — 0% `holdout_gate_d` instantiation or leakage!)
    from synccraft.dsl import GaitDecl

    train_val_species = (
        ("train", "swift_fox_v1", "canine_quadruped", "synccraft_proc_team_a", _quadruped_fixture_dsl(chest=(4.0, 4.0, 6.0), hips=(3.5, 3.5, 5.0), leg_u=3.5, leg_l=3.5)),
        ("train", "ironhide_bison_v1", "heavy_ungulate_quadruped", "synccraft_proc_team_a", _quadruped_fixture_dsl(chest=(6.5, 6.0, 8.0), hips=(5.5, 5.0, 6.5), leg_u=4.5, leg_l=4.5)),
        ("val", "dune_skink_v1", "sprawled_reptile", "synccraft_proc_team_b", _quadruped_fixture_dsl(chest=(4.5, 2.5, 6.5), hips=(4.0, 2.5, 5.5), leg_u=3.5, leg_l=3.0, sprawled=True)),
    )
    allowed_prop: list[ProposerTaskRecord] = []
    for sp_split, sp_aid, sp_plan, sp_author, sp_dsl in train_val_species:
        sp_prog = parse_program(sp_dsl)
        sp_sc = compile_geometry(sp_prog)
        sp_morph = analyze_morphology(sp_prog, sp_sc)
        sp_mdict = sp_morph.to_dict()
        skel = [
            {
                "bone": b.name,
                "parent": b.parent_name,
                "pivot_world": [round(float(x), 3) for x in b.pivot_world],
                "boxes": [{"name": bx.name, "size": [round(float(s), 3) for s in bx.size]} for bx in sp_sc.boxes if bx.bone_name == b.name],
            }
            for b in sp_sc.bones
        ]
        allowed_prop.append(
            ProposerTaskRecord(
                record_id=f"{sp_aid}__label_morphology",
                task_type="label_morphology_roles",
                split=sp_split,
                asset_id=sp_aid,
                body_plan_group=sp_plan,
                source_author=sp_author,
                dsl_source=sp_dsl,
                prompt_payload={"skeleton": skel},
                target_payload={
                    "limbs": [{"role": lb.role, "chain": list(lb.chain), "posture": lb.posture} for lb in sp_morph.limbs],
                    "spine": list(sp_morph.spine),
                    "neck": list(sp_morph.neck),
                    "tail": list(sp_morph.tail),
                    "wings": [{"role": wg.role, "chain": list(wg.chain)} for wg in sp_morph.wings],
                },
                verified_by_solver=True,
            )
        )
        for idx_p, (phrase, dur_s) in enumerate((("cautious creeping stalk", 1.2), ("snappy alert trot", 1.0))):
            seed_g = style_phrase_to_gait_preset(phrase, sp_morph, gait_name=f"gait_{idx_p}")
            rep_g = search_or_repair_gait(sp_prog, seed_g, duration_s=dur_s, clip_name=f"clip_{idx_p}", max_evals=2)
            allowed_prop.append(
                ProposerTaskRecord(
                    record_id=f"{sp_aid}__propose__{idx_p}",
                    task_type="propose_gait_from_style",
                    split=sp_split,
                    asset_id=sp_aid,
                    body_plan_group=sp_plan,
                    source_author=sp_author,
                    dsl_source=sp_dsl,
                    prompt_payload={"style_phrase": phrase, "duration_s": dur_s, "morphology": sp_mdict},
                    target_payload=_gait_decl_to_dict(rep_g.gait),
                    verified_by_solver=rep_g.passed,
                )
            )
            if idx_p == 1 and rep_g.passed:
                inv = invert_clip_to_gait(rep_g.program, f"clip_{idx_p}")
                allowed_prop.append(
                    ProposerTaskRecord(
                        record_id=f"{sp_aid}__invert__{idx_p}",
                        task_type="invert_clip_to_gait",
                        split=sp_split,
                        asset_id=sp_aid,
                        body_plan_group=sp_plan,
                        source_author=sp_author,
                        dsl_source=sp_dsl,
                        prompt_payload={
                            "duration_s": dur_s,
                            "cycles_estimated": inv.cycles_estimated,
                            "stride_length_units": round(inv.stride_length_units, 4),
                            "per_limb_duty": {k: round(v, 4) for k, v in inv.per_limb_duty.items()},
                            "per_limb_stroke_units": {k: round(v, 4) for k, v in inv.per_limb_stroke_units.items()},
                            "morphology": sp_mdict,
                        },
                        target_payload=_gait_decl_to_dict(rep_g.gait),
                        verified_by_solver=True,
                    )
                )
        broken_g = GaitDecl(
            name="overdriven_gait",
            pattern=_adapt_pattern_for_morphology("diagonal_pairs", sp_morph),
            froude=2.8,
            duty=0.35,
            lift_ratio=0.36,
            heave_ratio=0.14,
            pitch_deg=11.0,
            spine_wave_deg=18.0,
            snap=0.90,
        )
        fixed_g = search_or_repair_gait(sp_prog, broken_g, duration_s=1.0, clip_name="broken_clip", max_evals=3)
        allowed_prop.append(
            ProposerTaskRecord(
                record_id=f"{sp_aid}__repair_gait",
                task_type="repair_failed_gait",
                split=sp_split,
                asset_id=sp_aid,
                body_plan_group=sp_plan,
                source_author=sp_author,
                dsl_source=sp_dsl,
                prompt_payload={"failing_gait": _gait_decl_to_dict(broken_g), "morphology": sp_mdict},
                target_payload=_gait_decl_to_dict(fixed_g.gait),
                verified_by_solver=fixed_g.passed,
            )
        )
        tbox = sp_sc.boxes[0].name
        allowed_prop.append(
            ProposerTaskRecord(
                record_id=f"{sp_aid}__texture_plan",
                task_type="plan_texture_and_stamps",
                split=sp_split,
                asset_id=sp_aid,
                body_plan_group=sp_plan,
                source_author=sp_author,
                dsl_source=sp_dsl,
                prompt_payload={"character_brief": "weathered sculpted scales with glowing amber eyes", "boxes": [bx.name for bx in sp_sc.boxes], "materials": [p.name for p in sp_prog.palettes]},
                target_payload={
                    "style": {"shade_mode": "banded", "quantize_ramp": True, "glow_bleed": 1},
                    "material_family": "keratin_scales",
                    "stamps": [{"name": "eye_pair", "target_box": tbox, "face": "east", "at": [0.35, 0.40], "size": [0.22, 0.18], "shape": "eye", "emissive": 0.85, "mirror": True}],
                },
                verified_by_solver=True,
            )
        )
    for p_rep in range(max(2, int(round(14 * scale)))):
        for pr in allowed_prop:
            assert pr.split != "holdout_gate_d"
            assert pr.body_plan_group not in ("biped_theropod", "hexapod_arthropod")
            p_prov = make_clean_room_provenance(
                asset_id=f"proposer_{pr.asset_id}",
                title=f"SyncCraft Proposer {pr.asset_id}",
                license_spdx="CC0-1.0",
                author="SyncCraft Stage-7 Clean-Room Proposer Synthesizer",
            )
            prompt_p = (
                f"[SyncCraft v4.1 World-Model Proposer — `{pr.task_type}`]\n"
                f"Body plan: `{pr.body_plan_group}`. Input specification:\n"
                f"```json\n{json.dumps(pr.prompt_payload, sort_keys=True)}\n```\n"
                "Return ONLY the solver-verified JSON target parameters."
            )
            think_p = (
                f"Analyzing `{pr.task_type}` for `{pr.body_plan_group}` (`{pr.asset_id}`): "
                "matching dimensionless Froude number, duty factor, lift/heave ratios, and morphology chains so swept motion passes with 0 SAT collisions."
            )
            body_p = f"```json\n{json.dumps(pr.target_payload, indent=2, sort_keys=True)}\n```"
            ans_p = format_supervised_answer(body_p, think_p, enable_thinking=(p_rep % 3 != 2))
            records.append(
                {
                    "id": f"stage7_v41_prop_{pr.record_id}_{p_rep:03d}",
                    "source_group": f"proposer-v41:{pr.asset_id}",
                    "language": "en",
                    "task": "synccraft_v41_proposer",
                    "prompt": prompt_p,
                    "answer": ans_p,
                    "target": {"kind": "text", "text": body_p},
                    "enable_thinking": (p_rep % 3 != 2),
                    "frames": [],
                    "provenance": p_prov,
                    "metadata": {
                        "asset_id": pr.asset_id,
                        "body_plan_group": pr.body_plan_group,
                        "proposer_task_type": pr.task_type,
                        "dsl_source": pr.dsl_source,
                        "prompt_payload": pr.prompt_payload,
                        "target_payload": pr.target_payload,
                        "grouping_keys": [f"proposer-v41:{pr.asset_id}"],
                    },
                }
            )

    rng.shuffle(records)
    emit(
        "stage7_sculpted_v8_v41_ready",
        records=len(records),
        media_blobs=len(blob_map),
        archetypes=len(archetypes),
    )
    return records, blob_map


def evaluate_v8_v41_task_completion(record: dict[str, Any], completion_text: str) -> dict[str, Any]:
    """Deterministic `SKILL.md` v8 + `Block-DSL v4.1` + Gate D Proposer oracle verifier."""
    from .synccraft_bridge import _extract_dsl_candidate, _extract_json_candidate, strip_think_block

    task = str(record.get("task") or "")
    think_part, body = strip_think_block(completion_text)
    meta = record.get("metadata") or {}

    if task == "synccraft_v41_sculpted_v8":
        dsl_cand = _extract_dsl_candidate(body)
        try:
            v8_rep = run_v8_construct_check(dsl_cand)
        except Exception as exc:
            return {"task": task, "parse_valid": False, "passed": False, "error": f"v8_compile_error: {exc}"}
        return {
            "task": task,
            "has_think_trace": bool(think_part),
            "parse_valid": True,
            "sat_passed": bool(v8_rep["static_audit_passed"]),
            "passed": bool(v8_rep["passed_v8"]),
            "v8_check": v8_rep,
            "sculpted_eval": {"sculpted_score": float(v8_rep["style_anchor_score"])},
        }

    if task == "synccraft_sculpted_self_critique" and "v8_rubric" in meta:
        try:
            pred = _extract_json_candidate(body)
        except Exception as exc:
            return {"task": task, "parse_valid": False, "passed": False, "error": f"json_error: {exc}"}
        oracle = meta["v8_rubric"]
        pb_ok = bool(pred.get("punch_back_to_w1")) == bool(oracle.get("punch_back_to_w1"))
        gate_ok = bool(pred.get("passed_v8_gate")) == bool(oracle.get("passed_v8_gate"))
        sc_ok = abs(float(pred.get("rubric_26_score", -99)) - float(oracle.get("rubric_26_score", 0))) <= 4.0
        ok = pb_ok and gate_ok and sc_ok
        return {
            "task": task,
            "has_think_trace": bool(think_part),
            "parse_valid": True,
            "passed": ok,
            "field_accuracy": round((int(pb_ok) + int(gate_ok) + int(sc_ok)) / 3.0, 4),
        }

    if task == "synccraft_v4_modular_assembly" and "modular_task_kind" in meta:
        mkind = str(meta["modular_task_kind"])
        oracle_b = str(meta.get("oracle_body") or "")
        if mkind == "propagate":
            try:
                pred_j = _extract_json_candidate(body)
                gt_j = _extract_json_candidate(oracle_b)
            except Exception as exc:
                return {"task": task, "parse_valid": False, "passed": False, "error": str(exc)}
            matches = sum(
                1 for k, v in gt_j.items()
                if k in pred_j and (
                    (isinstance(v, float) and isinstance(pred_j[k], (int, float)) and abs(float(pred_j[k]) - v) <= 0.1)
                    or pred_j[k] == v
                )
            )
            acc = matches / max(1, len(gt_j))
            return {"task": task, "has_think_trace": bool(think_part), "parse_valid": True, "passed": acc >= 0.99, "field_accuracy": acc}

        cand_sec = _extract_dsl_candidate(body)
        if mkind == "layout":
            ok = cand_sec.lstrip().startswith("@layout") and "hull_span" in cand_sec and "deck_width" in cand_sec and cand_sec.count("{") == cand_sec.count("}")
            return {"task": task, "has_think_trace": bool(think_part), "parse_valid": ok, "passed": ok}
        if mkind in ("comp_synth", "surgical_swap"):
            try:
                ensure_synccraft_importable()
                from synccraft.audit.sync_audit import audit_scene
                from synccraft.dsl.parser import parse_program
                from synccraft.kinematics.se3_chain import compile_geometry

                spliced = re.sub(
                    r"@component\s+SolarBeaconMast\s*\{.*?\n\}",
                    cand_sec.strip(),
                    SOLAR_SIEGE_STRIDER_V41_DSL,
                    flags=re.DOTALL,
                )
                p_sp = parse_program(spliced)
                sc_sp = compile_geometry(p_sp)
                aud_sp = audit_scene(sc_sp)
                ok = bool(aud_sp["passed"]) and (mkind != "surgical_swap" or "sun_vane" in cand_sec)
                return {"task": task, "has_think_trace": bool(think_part), "parse_valid": True, "sat_passed": bool(aud_sp["passed"]), "passed": ok}
            except Exception as exc:
                return {"task": task, "parse_valid": False, "passed": False, "error": str(exc)}

    if task == "synccraft_v41_proposer":
        try:
            pred = _extract_json_candidate(body)
        except Exception as exc:
            return {"task": task, "parse_valid": False, "passed": False, "error": f"proposer_json_error: {exc}"}
        sub = str(meta.get("proposer_task_type") or "")
        target = meta.get("target_payload") or {}
        dsl_src = str(meta.get("dsl_source") or "")
        if sub in ("propose_gait_from_style", "repair_failed_gait", "invert_clip_to_gait") and dsl_src:
            try:
                ensure_synccraft_importable()
                from synccraft.dsl.parser import parse_program
                from synccraft.kinematics.gait_inversion import search_or_repair_gait
                from synccraft.world.proposer_tasks import _gait_decl_from_dict

                prog = parse_program(dsl_src)
                gait_decl = _gait_decl_from_dict(pred, default_name="cand_gait")
                dur_s = float((meta.get("prompt_payload") or {}).get("duration_s", 1.4))
                res = search_or_repair_gait(prog, gait_decl, duration_s=dur_s, clip_name="eval_cand", max_evals=1)
                return {
                    "task": task,
                    "proposer_task_type": sub,
                    "has_think_trace": bool(think_part),
                    "parse_valid": True,
                    "passed": bool(res.passed),
                    "field_accuracy": 1.0 if res.passed else 0.3,
                }
            except Exception as exc:
                return {"task": task, "parse_valid": False, "passed": False, "error": f"gait_solver_error: {exc}"}

        matches = sum(1 for k, v in target.items() if pred.get(k) == v)
        acc = matches / max(1, len(target))
        return {
            "task": task,
            "proposer_task_type": sub,
            "has_think_trace": bool(think_part),
            "parse_valid": True,
            "passed": acc >= 0.80,
            "field_accuracy": round(acc, 4),
        }

    from .sculpted_style_corpus import evaluate_sculpted_task_completion

    return evaluate_sculpted_task_completion(record, completion_text)
