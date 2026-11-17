"""End-to-end SyncCraft v3.2 3D World-Model & Working-Pipeline Asset Generator.

Connects ``SparkNative`` (or any multimodal generation callable) to SyncCraft's
shared SE(3) compiler, 15-axis SAT + ADR 0007 ``sculpted`` overlap auditor,
Gate G2 bounded repair policy (``max_reaudits <= 3``, integer ``size`` fixed during
PGD, monotonic ``CostLedger`` accounting), stateful PBR/OKLCH surface baker,
artist critique/upgrade loop, and multi-engine exporters:

- Blockbench 5.0 ``.bbmodel`` (geometry + embedded RGBA atlas + baked animations)
- Minecraft Bedrock 1.12.0 ``.geo.json`` + 1.8.0 ``.animation.json`` (with 8-corner
  post-serialization vertex re-audit ``max_err <= 1e-6``)
- GeckoLib 4 ``.geckolib.geo.json`` + ``.geckolib.animation.json``
- Vanilla Java ``java_modern`` / ``java_legacy`` block/item JSON (when non-entity
  and within ``[-16, 32]`` bounds)
- glTF 2.0 Metallic-Roughness PBR ``.gltf`` + ``base_color``/``normal_gl``/``orm``/``emissive`` PNGs
- Minecraft LabPBR 1.3 ``atlas.png`` / ``atlas_n.png`` / ``atlas_s.png``
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .common import write
from .synccraft_bridge import _camera_unit_meta, ensure_synccraft_importable, render_world_views_rgba


SUPPORTED_EXPORT_FORMATS = (
    "bbmodel",
    "bedrock",
    "geckolib",
    "java",
    "gltf_pbr",
    "labpbr",
)

SYNCCRAFT_V32_COMPACT_RULES = (
    "Mandatory SyncCraft Block-DSL v3.2 grammar & anti-toy rules:\n"
    "```blockdsl\n"
    "@meta { archetype: entity, target_profile: bbmodel_5, texel_density: 16, audit_profile: authored }\n"
    "@palette { primary_mat: ramp(base_lch=[0.56, 0.08, 45], steps=6) }\n"
    "@rig {\n"
    "  root bone root_core {\n"
    "    pivot_local: [0, 8, 0]\n"
    "    box trunk_box { size: [6, 4, 8], material: primary_mat }\n"
    "    bone left_limb attach(parent_box=trunk_box, anchor=[-1, -0.5, 0], align=[1, 0.5, 0]) mirror_x(right_limb) {\n"
    "      box limb_seg { size: [2, 3, 2], material: primary_mat }\n"
    "    }\n"
    "  }\n"
    "}\n"
    "```\n"
    "1. `@meta` ONLY accepts `archetype` (`entity` | `prop` | `architecture` | `vehicle`), `target_profile` "
    "(`bedrock_entity` | `bbmodel_5` | `java_modern` | `gltf_pbr`), `texel_density` (int), and optional `audit_profile` "
    "(`strict` | `sculpted` | `authored`) — NEVER `format_version`, `version`, `domain`, or `entity:`.\n"
    "2. Section order: `@meta`, `@palette`, `@rig` (`root bone <name> { box <name> { ... } }`), optional `@materials`, optional `@style`, optional `@muscles`, optional `@kinematics`, optional `@behavior`.\n"
    "3. `@palette` entries MUST use `<name>_mat: ramp(base_lch=[L, C, H], steps=6)` (`0<L<=1`) — ALWAYS suffix `@palette` identifiers with `_mat`.\n"
    "4. Box `size: [W, H, D]` MUST be 3 positive INTEGER literals in square brackets. NEVER use `root_bone: bone(...)`. "
    "For zero SAT collision in `place(base_box=B, anchor=[ax,ay,az], align=[lx,ly,lz])`, at least ONE axis MUST touch an outer face (`anchor[k]=1, align[k]=-1` or `anchor[k]=-1, align[k]=1`), and never stack two sibling boxes at the exact same `(base_box, anchor, align)`.\n"
    "5. CRITICAL `mirror_x` RULE: When a left bone declares `mirror_x(<right_name>)`, `<right_name>` is generated AUTOMATICALLY — NEVER also write a separate `bone <right_name> { ... }` block!\n"
    "6. Author rich 16–42 box artist silhouettes matching the requested subject: primary core + secondary taper, neck/head/focal assembly, appendages, 3-part limbs, and 4–6 contrasting `_mat` ramps."
)

SYNCCRAFT_SCULPTED_COMPACT_RULES = (
    "You author vibrant, deliberately sculpted 3D Blockbench models in SyncCraft Block-DSL (`blockbench-sculpted-style` v6). "
    "Obey these exact grammar & style rules (adapt all part/material names and anatomy to the requested subject):\n"
    "1. `@meta`: `@meta { archetype: entity, target_profile: bbmodel_5, texel_density: 16, dsl_version: \"3.2\", size_step: 0.5, audit_profile: sculpted }` (use `archetype: vehicle` or `prop` when appropriate).\n"
    "2. Vibrant 3+1 `@palette`: `<name>_mat: ramp(base_lch=[L, C, H], steps=6, hue_shift=[-14, 10])` so shadows shift cool and highlights drift warm; set `emissive=true` on 1 focal material (`3-15%` of elements).\n"
    "3. `@rig` hierarchy: `root bone <root_name> { pivot_local: [x, y, z] ... }` (`root bone` ONLY accepts `pivot_local:`, NEVER `rest_euler:`). "
    "Every child bone MUST use the keyword `bone` (NEVER `joint`!): `bone <name> attach(parent_box=<box>, anchor=[ax,ay,az], align=[lx,ly,lz]) { rest_euler: [rx, ry, rz] ... }` (optional `mirror_x(<right_name>)` — NEVER redeclare `<right_name>` explicitly, and NEVER nest a `box` inside another `box`!).\n"
    "4. `box` & `plane` syntax: ONLY the FIRST `box` inside each `bone { ... }` omits `place(...)`! Every subsequent sibling `box` inside the same `bone` MUST have an explicit `place(base_box=<box>, anchor=[ax,ay,az], align=[lx,ly,lz])` (`box` NEVER takes `offset:` or `position:` inside `{ ... }`). "
    "Zero-thickness cutout `plane`s MUST use 2-number `size: [W, H]` and axis letter `normal: x` | `y` | `z`: `plane <name> place(base_box=<box>, anchor=[ax,ay,az], align=[lx,ly,lz]) { size: [W, H], normal: x, material: <mat> }` (NEVER 3-number `size` or vector `normal: [0,0,1]`!).\n"
    "5. Deliberately sculpted 3D silhouette & scale: use normal Blockbench unit sizes (`0.5..16` units, NEVER `50..160`), NEVER repeat numbered sibling clones (`plate_2..plate_20`), and articulate across >= 5 child `bone`s (S-curved `neck` -> `head` -> `jaw`, `horn_l mirror_x(horn_r)`, `wing_l mirror_x(wing_r)` or `bogie_fl mirror_x(bogie_fr)`, `fl_leg mirror_x(fr_leg)`, `hl_leg mirror_x(hr_leg)`, `chain tail ... segments=4 taper=0.80`) with 6-34 deg organic tilts and outer-face seating (`|anchor[k]| >= 0.85, align[k] = -0.5 * sign(anchor[k])` so >=35% of every box protrudes)."
)

SYNCCRAFT_SCULPTED_SYSTEM_PROMPT = (
    "You author vibrant, deliberately sculpted 3D Blockbench models in SyncCraft Block-DSL (`blockbench-sculpted-style` v6). "
    "Combine flowing S-curved/tilted poses, 0.5-step chamfered proportions, hue-shifted cel ramps, seated joint overlaps, "
    "and RGBA alpha-cutout `plane`s (never stiff grid-aligned boxes or 5-9 box toys):\n"
    "```blockdsl\n"
    "@meta { archetype: entity, target_profile: bbmodel_5, texel_density: 16, dsl_version: \"3.2\", size_step: 0.5, audit_profile: sculpted }\n"
    "@palette {\n"
    "  hide_mat: ramp(base_lch=[0.58, 0.15, 22], steps=6, hue_shift=[-15, 10])\n"
    "  cream_mat: ramp(base_lch=[0.82, 0.06, 68], steps=6, hue_shift=[-12, 8])\n"
    "  horn_mat: ramp(base_lch=[0.84, 0.06, 78], steps=6, hue_shift=[-10, 6])\n"
    "  eye_glow_mat: ramp(base_lch=[0.84, 0.18, 58], steps=6, hue_shift=[-6, 12], emissive=true)\n"
    "}\n"
    "@rig {\n"
    "  root bone hips_root {\n"
    "    pivot_local: [0, 12, 3]\n"
    "    box pelvis_mass { size: [7, 6, 6.5], material: hide_mat, rotate: [-4, 0, 0] }\n"
    "    bone chest_seg attach(parent_box=pelvis_mass, anchor=[0, 0.15, -0.75], align=[0, 0, 0.8]) {\n"
    "      rest_euler: [8, 0, 0]\n"
    "      box chest_core { size: [9, 7, 8], material: hide_mat }\n"
    "      box sternum place(base_box=chest_core, anchor=[0, -0.84, -0.1], align=[0, 0.5, 0]) { size: [7, 1.5, 6.5], material: cream_mat, rotate: [-5, 0, 0] }\n"
    "      plane dorsal_fringe place(base_box=chest_core, anchor=[0, 1, 0], align=[0, -1, 0], offset=[0.25, 0, 0]) { size: [2.5, 6], normal: x, material: horn_mat }\n"
    "    }\n"
    "  }\n"
    "}\n"
    "```\n"
    + SYNCCRAFT_SCULPTED_COMPACT_RULES
)

SYNCCRAFT_V32_GRAMMAR_SYSTEM_PROMPT = (
    "You author valid, artist-grade SyncCraft Block-DSL v3.2 3D assets with rich structural hierarchy (never simplistic 5-9 box toys). "
    "Follow this exact grammar skeleton (adapt bone/box topology, proportions, and colors to the requested subject):\n"
    "```blockdsl\n"
    "@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, audit_profile: strict }\n"
    "@palette {\n"
    "  primary_mat: ramp(base_lch=[0.58, 0.10, 48], steps=6)\n"
    "  under_mat: ramp(base_lch=[0.76, 0.05, 68], steps=6)\n"
    "  armor_mat: ramp(base_lch=[0.42, 0.08, 245], steps=6)\n"
    "  dark_mat: ramp(base_lch=[0.26, 0.03, 35], steps=6)\n"
    "  accent_mat: ramp(base_lch=[0.80, 0.17, 78], steps=6, emissive=true)\n"
    "}\n"
    "@rig {\n"
    "  root bone root_core {\n"
    "    pivot_local: [0, 12, 0]\n"
    "    box trunk_core { size: [6, 6, 7], material: primary_mat }\n"
    "    box armor_shell place(base_box=trunk_core, anchor=[0, 1, -0.2], align=[0, -1, 0]) {\n"
    "      size: [6, 1, 5], material: armor_mat, inflate: 0.2\n"
    "    }\n"
    "    bone neck_joint attach(parent_box=trunk_core, anchor=[0, 0.5, -1], align=[0, -0.5, 1]) {\n"
    "      box neck_seg { size: [4, 4, 3], material: primary_mat }\n"
    "      bone head_joint attach(parent_box=neck_seg, anchor=[0, 0.3, -1], align=[0, -0.5, 1]) {\n"
    "        box cranium { size: [5, 4, 4], material: primary_mat }\n"
    "        box snout place(base_box=cranium, anchor=[0, 0.1, -1], align=[0, 0, 1]) { size: [3, 2, 3], material: under_mat }\n"
    "        bone left_crest attach(parent_box=cranium, anchor=[-0.8, 1, 0.4], align=[0, -1, 0]) mirror_x(right_crest) {\n"
    "          box crest_fin { size: [1, 3, 2], material: dark_mat }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "    bone fl_upper attach(parent_box=trunk_core, anchor=[-1, -0.5, -0.6], align=[1, 1, 0]) mirror_x(fr_upper) {\n"
    "      box fl_upper_seg { size: [2, 4, 2], material: armor_mat }\n"
    "      bone fl_lower attach(parent_box=fl_upper_seg, anchor=[0, -1, 0], align=[0, 1, 0]) {\n"
    "        box fl_lower_seg { size: [2, 4, 2], material: primary_mat }\n"
    "        box fl_foot place(base_box=fl_lower_seg, anchor=[0, -1, -0.25], align=[0, 1, 0]) { size: [2, 1, 3], material: dark_mat }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}\n"
    "@kinematics {\n"
    "  ik_chain fl_ik { upper: fl_upper, lower: fl_lower, effector: [0, -5, 0] }\n"
    "  ik_chain fr_ik { upper: fr_upper, lower: fr_upper__fl_lower, effector: [0, -5, 0] }\n"
    "}\n"
    "@behavior {\n"
    "  clip walk_cycle {\n"
    "    duration_s: 1.6\n"
    "    ik_drive fl_ik: foot_cycle(stride=1.5, lift=0.65, phase_deg=0)\n"
    "    ik_drive fr_ik: foot_cycle(stride=1.5, lift=0.65, phase_deg=180)\n"
    "  }\n"
    "}\n"
    "```\n"
    + SYNCCRAFT_V32_COMPACT_RULES
)


def evaluate_3d_asset_aesthetics(
    program_or_dsl: Any,
    *,
    prompt: str = "",
) -> dict[str, Any]:
    """Perform quantitative structural, anatomical, material, and anti-toy aesthetic judgment on a Block-DSL asset."""
    ensure_synccraft_importable()
    from synccraft.audit.sync_audit import audit_scene
    from synccraft.dsl.ast_nodes import Program, validate_program
    from synccraft.dsl.parser import parse_program
    from synccraft.kinematics.se3_chain import compile_geometry
    from synccraft.world.artist_loop import analyze_artist_project

    prog = validate_program(program_or_dsl) if isinstance(program_or_dsl, Program) else validate_program(parse_program(str(program_or_dsl)))
    scene = compile_geometry(prog)
    enforce_contact = getattr(prog.meta, "audit_profile", "strict") != "authored"
    aud = audit_scene(scene, enforce_physical_contact=enforce_contact)
    critique = analyze_artist_project(prog)

    n_boxes = len(scene.boxes)
    n_bones = len(scene.bones)
    max_depth = int(critique.max_bone_depth)
    n_palettes = len(prog.palettes)
    n_materials = len(getattr(prog, "materials", ()) or ())
    used_materials = {b.material for b in scene.boxes}
    unique_sizes = {tuple(round(float(x), 3) for x in b.size) for b in scene.boxes}

    all_names_low = " ".join([*(b.name.lower() for b in scene.boxes), *(bn.name.lower() for bn in scene.bones)])
    is_entity = prog.meta.archetype == "entity" or any(
        w in str(prompt or "").lower()
        for w in ("hound", "dog", "wolf", "wyvern", "dragon", "creature", "beast", "horse", "quadruped", "walker", "mech", "sentinel", "character", "bird")
    )

    has_head_group = any(
        k in all_names_low for k in ("head", "skull", "snout", "muzzle", "jaw", "beak", "helm", "turret", "cockpit", "visor", "crest", "eye")
    )
    has_axial_group = any(
        k in all_names_low for k in ("torso", "ribcage", "chest", "body", "pelvis", "spine", "hull", "chassis", "pedestal", "frame", "core")
    )
    has_limb_group = any(
        k in all_names_low for k in ("thigh", "shin", "leg", "arm", "wing", "paw", "foot", "hoof", "claw", "fin", "wheel", "link")
    )
    has_secondary_group = any(
        k in all_names_low for k in ("tail", "ear", "horn", "plate", "armor", "collar", "trim", "plume", "crest", "antenna", "banner", "brow", "nose")
    )

    critique_notes: list[str] = []
    if not aud["passed"]:
        critique_notes.append(
            f"Resolve {len(aud['collision_pairs'])} SAT collision(s) and {len(aud['contact_violations'])} contact violation(s)."
        )

    # 1. Complexity & Part Hierarchy Score (0..1)
    target_boxes = 18 if is_entity else 10
    complexity_score = min(1.0, n_boxes / float(target_boxes))
    if is_entity and n_boxes < 12:
        critique_notes.append(
            f"Asset has only {n_boxes} boxes (toy blockout); expand to >= 18 boxes with primary, secondary, and tertiary silhouette details."
        )
    elif not is_entity and n_boxes < 6:
        critique_notes.append(
            f"Asset has only {n_boxes} boxes; add secondary structural trim, bevels, or sub-assemblies."
        )

    hierarchy_score = min(1.0, 0.5 * min(1.0, n_bones / (8.0 if is_entity else 3.0)) + 0.5 * min(1.0, max_depth / (2.0 if is_entity else 1.0)))
    if is_entity and n_bones < 6:
        critique_notes.append(
            f"Only {n_bones} bone(s) declared; articulate separate neck, head, tail, and multi-segment limb chains."
        )

    # 2. Anatomical / Architectural Landmark Completeness (0..1)
    if is_entity:
        landmarks = [has_head_group, has_axial_group, has_limb_group, has_secondary_group]
        landmark_score = sum(1.0 for x in landmarks if x) / 4.0
        if not has_head_group:
            critique_notes.append("Missing cranial/head assembly (add neck + skull + muzzle/jaw/visor/crest).")
        if not has_secondary_group:
            critique_notes.append("Missing secondary silhouette landmarks (add ears/horns, multi-segment tail, chest/collar armor, or paws).")
    else:
        landmark_score = min(1.0, 0.5 * float(has_axial_group) + 0.5 * min(1.0, len(unique_sizes) / 3.0))

    # 3. Proportion & Material Contrast Score (0..1)
    variety_score = min(1.0, len(unique_sizes) / (5.0 if is_entity else 3.0))
    if len(unique_sizes) <= 2 and n_boxes >= 5:
        critique_notes.append(
            f"Only {len(unique_sizes)} unique integer box size(s) used; vary primary trunk, tapered pelvis/neck, and fine extremity dimensions."
        )
    mat_score = min(1.0, max(len(used_materials), n_palettes) / (4.0 if is_entity else 2.5))
    if len(used_materials) < 3 and is_entity:
        critique_notes.append(
            f"Only {len(used_materials)} material ramp(s) applied across boxes; use >= 4 contrasting OKLCH ramps (coat/hull, underbelly/trim, armor/metal, dark keratin/emissive)."
        )

    phys_score = 1.0 if aud["passed"] else max(0.0, 1.0 - 0.25 * len(aud["collision_pairs"]) - 0.2 * len(aud["contact_violations"]))
    aesthetic_score = round(
        0.26 * complexity_score
        + 0.18 * hierarchy_score
        + 0.22 * landmark_score
        + 0.14 * variety_score
        + 0.10 * mat_score
        + 0.10 * phys_score,
        4,
    )

    is_toy_like = bool(
        (is_entity and (n_boxes < 12 or not has_head_group or len(unique_sizes) <= 2))
        or (not is_entity and n_boxes < 4)
        or aesthetic_score < 0.58
    )
    if aesthetic_score >= 0.76 and not is_toy_like and aud["passed"]:
        grade = "artist_grade"
    elif not is_toy_like and aesthetic_score >= 0.60:
        grade = "intermediate_asset"
    else:
        grade = "toy_blockout"

    return {
        "aesthetic_score": aesthetic_score,
        "artist_readiness_score": round(float(critique.readiness_score), 4),
        "grade": grade,
        "is_toy_like": is_toy_like,
        "is_entity": is_entity,
        "static_audit_passed": bool(aud["passed"]),
        "box_count": n_boxes,
        "bone_count": n_bones,
        "max_bone_depth": max_depth,
        "palette_count": n_palettes,
        "used_material_count": len(used_materials),
        "physical_material_spec_count": n_materials,
        "unique_box_sizes": len(unique_sizes),
        "landmarks": {
            "head_or_focal": has_head_group,
            "axial_core": has_axial_group,
            "limbs_or_actuators": has_limb_group,
            "secondary_silhouette": has_secondary_group,
        },
        "critique_notes": critique_notes,
    }


def _extract_clean_blockdsl(raw_text: str) -> str:
    """Strip closed/unclosed `<think>` wrappers, salvage truncated trailing lines, and close Markdown fences."""
    txt = str(raw_text or "")
    # 1. Prefer an explicit ```blockdsl ... ``` code fence anywhere after `</think>` (or in full text)
    after_think = txt.split("</think>", 1)[1] if "</think>" in txt else re.sub(r"^<think>\s*", "", txt)
    fence = re.search(r"```(?:blockdsl|dsl|synccraft)?\s*\n(.*?)```", after_think, flags=re.DOTALL | re.IGNORECASE)
    if fence is not None:
        cand = fence.group(1).strip()
    else:
        open_fence = re.search(r"```(?:blockdsl|dsl|synccraft)\s*\n(.*)$", after_think, flags=re.DOTALL | re.IGNORECASE)
        if open_fence is not None:
            cand = open_fence.group(1).replace("```", "").strip()
        else:
            cleaned = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL | re.IGNORECASE).strip()
            meta_idx = cleaned.find("@meta")
            cand = cleaned[meta_idx:].replace("```", "").strip() if meta_idx >= 0 else cleaned.replace("```", "").strip()

    open_braces = cand.count("{")
    close_braces = cand.count("}")
    if open_braces > close_braces:
        # Drop any trailing incomplete line(s) cut off mid-declaration by max_new_tokens before closing open scopes
        lines = cand.splitlines()
        while lines and not lines[-1].rstrip().endswith("}"):
            lines.pop()
        if lines:
            cand = "\n".join(lines)
            open_braces = cand.count("{")
            close_braces = cand.count("}")
        if 0 < (open_braces - close_braces) <= 8:
            cand = cand + ("\n}" * (open_braces - close_braces))
    return cand


def _safe_slug(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", str(name).strip().lower()).strip("_")
    return cleaned or "synccraft_asset"


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (bytes, bytearray)):
        b = bytes(obj)
        return {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if hasattr(obj, "item") and callable(getattr(obj, "item", None)):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def _build_repair_prompt(
    original_prompt: str,
    candidate_dsl: str,
    feedback: Mapping[str, Any],
    attempt: int,
) -> str:
    is_sculpted_cand = ("audit_profile: sculpted" in candidate_dsl) or ("size_step: 0.5" in candidate_dsl) or ("sculpted" in original_prompt.lower())
    lines = [
        "Repair the SyncCraft Block-DSL asset below so it compiles cleanly and passes all audits.",
        (
            "Preserve `size_step: 0.5, audit_profile: sculpted`, `hue_shift` ramps, `plane` cutouts, and non-grid rotations while fixing syntax or overlap errors."
            if is_sculpted_cand
            else "Keep every integer box `size` tuple discrete and unchanged when fixing small pose/offset collisions unless syntax/adherence requires adding missing parts."
        ),
        f"Original request:\n{original_prompt.strip()}",
        f"Repair attempt: {attempt}",
    ]
    if feedback.get("error"):
        err_s = str(feedback["error"])
        lines.append(f"Compiler/audit error: {err_s}")
        if "got 'base'" in err_s or "expected 'parent_box'" in err_s or "expected 'anchor'" in err_s:
            lines.append(
                "CRITICAL ATTACH/PLACE FIX: Child bones MUST use `attach(parent_box=<box>, anchor=[ax,ay,az], align=[lx,ly,lz])` "
                "(use `parent_box=`, NEVER `base=`), and sibling boxes MUST use `place(base_box=<box>, anchor=[ax,ay,az], align=[lx,ly,lz])`!"
            )
        if "rest_euler" in err_s:
            lines.append(
                "CRITICAL ROOT BONE FIX: `root bone <name> { pivot_local: [x, y, z] ... }` ONLY accepts `pivot_local:` (NEVER `rest_euler:` on `root bone`; put `rotate: [...]` on the root box instead!)."
            )
        if "expected box, plane, locator, bone" in err_s or re.search(r"\bjoint\s+[A-Za-z_]", candidate_dsl):
            lines.append(
                "CRITICAL BONE KEYWORD FIX: Replace every `joint <name> attach(...)` with `bone <name> attach(...)` (Block-DSL uses `bone`, NEVER `joint`!), "
                "and NEVER nest a `box` inside another `box { ... }` (use `bone <name> attach(...) { box ... }` or sibling `box <name> place(...) { size: [...], material: ... }`)."
            )
        if "plane" in err_s or re.search(r"normal\s*:\s*\[", candidate_dsl):
            lines.append(
                "CRITICAL PLANE FIX: `plane <name> place(base_box=<box>, anchor=[...], align=[...]) { size: [W, H], normal: x, material: <mat> }` "
                "MUST use a 2-number `size: [W, H]` and single axis letter `normal: x` | `y` | `z` (NEVER 3-number `size: [W, H, D]` and NEVER vector `normal: [0, 0, 1]`)!"
            )
        if "subsequent same-bone boxes require an explicit place clause" in err_s:
            lines.append(
                "CRITICAL SAME-BONE PLACE FIX: Inside any `bone { ... }`, ONLY the very first `box` has no `place(...)`! "
                "Every 2nd, 3rd, ... `box` or `plane` in that `bone` MUST include `place(base_box=<first_box>, anchor=[ax,ay,az], align=[lx,ly,lz])` (or move it onto a child `bone <name> attach(...) { ... }`)."
            )
        if "field 'offset'" in err_s or "field 'position'" in err_s or "unexpected box" in err_s:
            lines.append(
                "CRITICAL BOX OFFSET FIX: `box <name> { ... }` NEVER accepts `offset:`, `position:`, or `hinge:` inside `{ ... }`! "
                "Position sibling boxes with `box <name> place(base_box=<box>, anchor=[ax,ay,az], align=[lx,ly,lz], offset=[ox,oy,oz]) { size: [...], material: ... }`, "
                "or put articulated parts on a child `bone <name> attach(parent_box=<box>, anchor=[...], align=[...]) { ... }`."
            )
        if "expected ',', got" in err_s or "unexpected @meta field" in err_s or "@part" in candidate_dsl:
            lines.append(
                "CRITICAL GRAMMAR FIX: Every vector MUST use comma-separated numbers `[a, b, c]` (NEVER space-separated `[a b c]`). "
                "Top-level sections MUST be `@meta { archetype: ..., target_profile: ..., texel_density: 16 }`, `@palette { ... }`, and `@rig { root bone <name> { ... } }` (NEVER `@part` or `@ramp`)."
            )
        if "expected integer literal" in err_s:
            lines.append(
                "CRITICAL SIZE_STEP FIX: Add `dsl_version: \"3.2\", size_step: 0.5, audit_profile: sculpted` inside `@meta { ... }` if using `0.5`-step sizes, "
                "or round every `size: [W, H, D]` to positive integers when `size_step` is `1`."
            )
        dup_m = re.search(r"duplicate identifier:\s*([A-Za-z0-9_]+)", err_s)
        if dup_m:
            dup_id = dup_m.group(1)
            if f"mirror_x({dup_id})" in candidate_dsl or re.search(rf"bone\s+{re.escape(dup_id)}\b.*mirror_x", candidate_dsl):
                lines.append(
                    f"CRITICAL MIRROR_X FIX: `{dup_id}` is ALREADY auto-generated by `mirror_x({dup_id})`! "
                    f"DELETE the explicit `bone {dup_id} ... {{ ... }}` block completely (keep ONLY the left bone with `mirror_x({dup_id})`), "
                    f"and in `@kinematics` reference any mirrored child lower bone as `{dup_id}__<left_lower_bone>`."
                )
            else:
                lines.append(
                    f"CRITICAL FIX: `{dup_id}` is declared more than once! Give each sibling box/bone a unique name (`{dup_id}_1`, `{dup_id}_2`, ...) "
                    f"and ensure `@palette` material names end with `_mat` (`{dup_id}_mat`) so every identifier in the program is globally unique."
                )
    collisions = list(feedback.get("collision_pairs") or ())
    if collisions:
        formatted_cols = ", ".join(
            f"{c.get('a')} vs {c.get('b')} (depth={float(c.get('depth', 0.0)):.4f}u)"
            for c in collisions[:8]
        )
        lines.append(f"15-axis SAT collision pairs: {formatted_cols}")
        lines.append(
            "SAT COLLISION FIX: Ensure every `place(base_box=B, anchor=[ax,ay,az], align=[lx,ly,lz])` has at least ONE outer-face contact axis "
            "(`anchor[k]=1, align[k]=-1` for +face or `anchor[k]=-1, align[k]=1` for -face) instead of interior `|anchor|<1` on all axes, "
            "and remove or stagger duplicate sibling boxes placed at the identical `(base_box, anchor, align)`."
        )
    contacts = list(feedback.get("contact_violations") or ())
    if contacts:
        formatted_ct = ", ".join(
            f"{c.get('violation') or c.get('kind', 'violation')}:{c.get('pair') or c.get('bone') or c.get('box') or c}"
            for c in contacts[:8]
        )
        lines.append(f"Physical contact violations: {formatted_ct}")
        lines.append(
            "CONTACT / BURIAL FIX: `swallowed_redundant_box` means a box is >65% buried inside its parent (e.g. `anchor=[0.5, 0.5, 0.5]`)! "
            "Use normal Blockbench unit sizes (`0.5..16`), seat sibling boxes on outer faces (`anchor=[0, 0.92, 0], align=[0, -0.5, 0]`), and put wheels/limbs/wings on child `bone <name> attach(parent_box=<box>, anchor=[-1, -0.7, -0.4], align=[1, 0, 0]) mirror_x(...)` so >=35% of every box protrudes."
        )
    adh = list(feedback.get("prompt_violations") or ())
    if adh:
        lines.append(f"Prompt adherence violations: {', '.join(str(v) for v in adh[:8])}")
    lines.append(f"Current candidate Block-DSL:\n```blockdsl\n{candidate_dsl.strip()}\n```")
    if is_sculpted_cand:
        lines.append(
            "Output ONLY the complete repaired ```blockdsl ... ``` program obeying `@meta { ..., size_step: 0.5, audit_profile: sculpted }`, "
            "`ramp(base_lch=[...], steps=6, hue_shift=[-14, 10])`, `size: [W, H, D]` multiples of `0.5` (`plane` `size: [W, H], normal: x|y|z`), "
            "and `attach(parent_box=..., anchor=[...], align=[...])`."
        )
    else:
        lines.append(
            "Output ONLY the complete repaired ```blockdsl ... ``` program obeying `@meta { archetype: ..., target_profile: ..., texel_density: ... }`, `ramp(base_lch=[...], steps=6)`, integer `size: [W, H, D]`, and `attach(parent_box=..., anchor=[...], align=[...])`."
        )
    return "\n\n".join(lines)


def _prepare_media_from_pil_images(
    images: Sequence[Any] | None,
    *,
    view_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    from PIL import Image
    from .inference import empty_media, visual_media

    if not images:
        return empty_media()
    loaded = [Image.open(raw) if isinstance(raw, (str, Path)) else raw for raw in images]
    vkind = "multiview_render" if len(loaded) > 1 else "independent_image"
    resolved_views = list(view_names) if view_names is not None else (
        ["south", "right", "top", "iso"] if len(loaded) == 4 else []
    )
    v_units = [_camera_unit_meta(v) for v in resolved_views] if len(resolved_views) == len(loaded) else None
    return visual_media(loaded, visual_kind=vkind, visual_units=v_units)


def build_default_world_state_for_program(program: Any, *, seed: int = 0) -> Any:
    """Build a SyncCraft WorldState with adaptive solver resolution for large rigs."""
    ensure_synccraft_importable()
    from synccraft.kinematics.se3_chain import compile_geometry
    from synccraft.surface.contracts import EnvironmentState, MaterialProgram, StyleConfig, WorldState
    from synccraft.surface.domain import build_surface_domain, initialize_surface_state

    scene = compile_geometry(program)
    subdiv = 1 if len(scene.boxes) > 20 else 2
    mat_prog = MaterialProgram.from_program(program, seed=int(seed))
    domain = build_surface_domain(program, mat_prog, subdivisions_per_unit=subdiv)
    return WorldState(
        program=program,
        material_program=mat_prog,
        surface_state=initialize_surface_state(domain),
        style_config=StyleConfig.from_program(program),
        environment=EnvironmentState(),
    )


def export_program_bundle(
    program: Any,
    *,
    asset_name: str = "synccraft_asset",
    output_dir: str | Path | None = None,
    formats: Sequence[str] = ("bbmodel", "bedrock", "gltf_pbr"),
    world_state: Any | None = None,
    seed: int = 0,
    num_animation_frames: int = 12,
    animation_mode: str = "keyframes",
    render_preview_views: bool = True,
) -> dict[str, Any]:
    """Compile, audit, bake textures, export multi-engine 3D artifacts, and re-audit serialized geometry."""
    ensure_synccraft_importable()
    from synccraft.audit.sync_audit import audit_scene
    from synccraft.backends.bbmodel5_exporter import export_bbmodel
    from synccraft.backends.bedrock_gecko_exporter import export_bedrock, export_geckolib
    from synccraft.backends.common import ExportError
    from synccraft.backends.gltf_exporter import export_gltf_pbr
    from synccraft.backends.java_exporter import export_java
    from synccraft.backends.labpbr_exporter import export_labpbr_1_3
    from synccraft.dsl.ast_nodes import validate_program
    from synccraft.dsl.parser import format_program
    from synccraft.kinematics.musculoskeletal import audit_behavior_clip
    from synccraft.kinematics.se3_chain import compile_geometry
    from synccraft.shading.compiler import compile_vanilla_atlas
    from synccraft.world.artist_loop import ingest_artist_project_to_program

    prog = validate_program(program)
    scene = compile_geometry(prog)
    enforce_contact = getattr(prog.meta, "audit_profile", "strict") != "authored"
    scene_audit = audit_scene(scene, enforce_physical_contact=enforce_contact)
    if not scene_audit["passed"]:
        raise ExportError(
            f"Refusing to export unaudited scene {asset_name!r}: "
            f"collisions={len(scene_audit['collision_pairs'])}, "
            f"contact_violations={len(scene_audit['contact_violations'])}"
        )

    behavior_audits: dict[str, dict[str, Any]] = {}
    if getattr(prog, "behavior", None) is not None and prog.behavior.clips:
        for clip in prog.behavior.clips:
            behavior_audits[clip.name] = audit_behavior_clip(
                prog,
                clip.name,
                num_frames=max(8, int(num_animation_frames)),
            )

    ws = world_state if world_state is not None else build_default_world_state_for_program(prog, seed=seed)
    atlas_desc, atlas_png = compile_vanilla_atlas(ws)

    slug = _safe_slug(asset_name or getattr(prog.meta, "domain", "") or "synccraft_asset")
    out_path = Path(output_dir) if output_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / f"{slug}.blockdsl").write_text(format_program(prog) + "\n", encoding="utf-8")
        (out_path / f"{slug}_atlas.png").write_bytes(atlas_png)

    exported: dict[str, Any] = {}
    written_files: dict[str, str] = {}
    if out_path is not None:
        written_files["blockdsl"] = str(out_path / f"{slug}.blockdsl")
        written_files["vanilla_atlas_png"] = str(out_path / f"{slug}_atlas.png")

    for fmt in formats:
        if fmt not in SUPPORTED_EXPORT_FORMATS:
            raise ValueError(f"Unsupported export format {fmt!r}; expected subset of {SUPPORTED_EXPORT_FORMATS}")

        if fmt == "bbmodel":
            bb_doc = export_bbmodel(
                scene,
                name=slug,
                atlas=atlas_desc,
                program=prog,
                num_animation_frames=num_animation_frames,
            )
            # Re-audit serialized Blockbench 5.0 roundtrip via independent decompiler
            roundtrip_prog = ingest_artist_project_to_program(bb_doc, format_kind="bbmodel")
            roundtrip_scene = compile_geometry(roundtrip_prog)
            bb_box_count_match = len(roundtrip_scene.boxes) == len(scene.boxes)
            exported["bbmodel"] = {
                "document": bb_doc,
                "elements": len(bb_doc.get("elements", ())),
                "animations": len(bb_doc.get("animations", ())),
                "roundtrip_box_count_match": bb_box_count_match,
            }
            if out_path is not None:
                bb_file = out_path / f"{slug}.bbmodel"
                bb_file.write_text(json.dumps(bb_doc, indent=2), encoding="utf-8")
                written_files["bbmodel"] = str(bb_file)

        elif fmt in ("bedrock", "geckolib"):
            req_profile = "bedrock_entity" if fmt == "bedrock" else "geckolib"
            target_prog = (
                prog
                if prog.meta.target_profile == req_profile
                else validate_program(replace(prog, meta=replace(prog.meta, target_profile=req_profile)))
            )
            target_scene = scene if target_prog is prog else compile_geometry(target_prog)
            exporter_fn = export_bedrock if fmt == "bedrock" else export_geckolib
            bundle = exporter_fn(
                target_scene,
                program=target_prog,
                atlas=atlas_desc,
                identifier=f"geometry.synccraft.{slug}",
                profile=req_profile,
                num_animation_frames=num_animation_frames,
                animation_mode=animation_mode,
            )
            max_v_err = float(bundle["serialized_vertex_max_error"])
            if max_v_err > 1e-6:
                raise ExportError(f"{fmt} post-serialization vertex error {max_v_err:.3e} > 1e-6")
            exported[fmt] = bundle
            if out_path is not None:
                suffix = "" if fmt == "bedrock" else ".geckolib"
                geo_file = out_path / f"{slug}{suffix}.geo.json"
                geo_file.write_text(json.dumps(bundle["geometry"], indent=2), encoding="utf-8")
                written_files[f"{fmt}_geo_json"] = str(geo_file)
                if bundle.get("animations") is not None:
                    anim_file = out_path / f"{slug}{suffix}.animation.json"
                    anim_file.write_text(json.dumps(bundle["animations"], indent=2), encoding="utf-8")
                    written_files[f"{fmt}_animation_json"] = str(anim_file)

        elif fmt == "java":
            if prog.meta.archetype == "entity":
                exported["java"] = {
                    "skipped": True,
                    "reason": "vanilla Java block/item JSON does not encode articulated entities; use bedrock/geckolib/bbmodel",
                }
            else:
                java_profile = prog.meta.target_profile if prog.meta.target_profile in ("java_modern", "java_legacy") else "java_modern"
                try:
                    java_doc = export_java(
                        scene,
                        texture_id=f"synccraft:block/{slug}",
                        atlas=atlas_desc,
                        profile=java_profile,
                    )
                    exported["java"] = {"skipped": False, "profile": java_profile, "document": java_doc}
                    if out_path is not None:
                        java_file = out_path / f"{slug}.java.json"
                        java_file.write_text(json.dumps(java_doc, indent=2), encoding="utf-8")
                        written_files["java_json"] = str(java_file)
                except ExportError as exc:
                    exported["java"] = {"skipped": True, "reason": str(exc)}

        elif fmt == "gltf_pbr":
            gltf_bundle = export_gltf_pbr(ws, name=slug, uv_layout=atlas_desc)
            exported["gltf_pbr"] = {
                "standard": gltf_bundle["standard"],
                "nodes": len(gltf_bundle["gltf"].get("nodes", ())),
                "meshes": len(gltf_bundle["gltf"].get("meshes", ())),
                "extensions_used": list(gltf_bundle["gltf"].get("extensionsUsed", ())),
                "gltf": gltf_bundle["gltf"],
                "textures_meta": {
                    k: {"bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()}
                    for k, v in gltf_bundle["textures_png"].items()
                },
            }
            if out_path is not None:
                gltf_dir = out_path / f"{slug}_gltf"
                gltf_dir.mkdir(parents=True, exist_ok=True)
                gltf_file = gltf_dir / f"{slug}.gltf"
                gltf_file.write_text(json.dumps(gltf_bundle["gltf"], indent=2), encoding="utf-8")
                written_files["gltf"] = str(gltf_file)
                for tex_name, png_bytes in gltf_bundle["textures_png"].items():
                    tex_path = gltf_dir / tex_name
                    tex_path.write_bytes(png_bytes)
                    written_files[f"gltf_{tex_name}"] = str(tex_path)

        elif fmt == "labpbr":
            lab_bundle = export_labpbr_1_3(ws, uv_layout=atlas_desc)
            exported["labpbr"] = {
                "standard": lab_bundle["standard"],
                "width": lab_bundle["width"],
                "height": lab_bundle["height"],
                "sha256": lab_bundle["sha256"],
                "channel_audit": lab_bundle["channel_audit"],
            }
            if out_path is not None:
                for key_name, suffix_name in (
                    ("atlas_png", f"{slug}_labpbr.png"),
                    ("atlas_n_png", f"{slug}_labpbr_n.png"),
                    ("atlas_s_png", f"{slug}_labpbr_s.png"),
                ):
                    if isinstance(lab_bundle.get(key_name), (bytes, bytearray)):
                        lp = out_path / suffix_name
                        lp.write_bytes(bytes(lab_bundle[key_name]))
                        written_files[f"labpbr_{key_name}"] = str(lp)

    preview_views: dict[str, str] = {}
    studio_views: dict[str, str] = {}
    if render_preview_views and out_path is not None:
        renders, _, _, _ = render_world_views_rgba(ws, panel_size=(192, 192))
        for vid, png_bytes in renders.items():
            vp = out_path / f"{slug}_preview_{vid}.png"
            vp.write_bytes(png_bytes)
            preview_views[vid] = str(vp)
        try:
            from .sculpted_style_corpus import render_sculpted_turnaround_views

            studio_pngs, baked_studio = render_sculpted_turnaround_views(
                prog,
                views=("south", "right", "top", "iso"),
                panel_size=(256, 256),
                include_silhouette_sheet=True,
            )
            studio_atlas_p = out_path / f"{slug}_studio_atlas.png"
            studio_atlas_p.write_bytes(baked_studio["atlas_png"])
            written_files["studio_rgba_atlas_png"] = str(studio_atlas_p)
            for k_v, b_png in studio_pngs.items():
                sp = out_path / f"{slug}_studio_{k_v}.png"
                sp.write_bytes(b_png)
                studio_views[k_v] = str(sp)
        except Exception:
            pass

    return {
        "asset_name": slug,
        "audit_profile": getattr(prog.meta, "audit_profile", "strict"),
        "dsl_version": getattr(prog.meta, "dsl_version", "3.2"),
        "size_step": float(getattr(prog.meta, "size_step", 1.0)),
        "target_profile": prog.meta.target_profile,
        "boxes": len(scene.boxes),
        "bones": len(scene.bones),
        "clips": [c.name for c in prog.behavior.clips] if getattr(prog, "behavior", None) else [],
        "scene_audit": {
            "passed": scene_audit["passed"],
            "collision_count": len(scene_audit["collision_pairs"]),
            "sculpted_overlap_count": len(scene_audit.get("sculpted_overlap_pairs", ())),
            "contact_violation_count": len(scene_audit["contact_violations"]),
        },
        "behavior_audits": {
            k: {
                "passed": bool(v.get("passed")),
                "collision_frames": int(v.get("collision_frames", 0)),
                "max_stance_foot_slip_units": float(v.get("max_stance_foot_slip_units", 0.0)),
            }
            for k, v in behavior_audits.items()
        },
        "atlas_resolution": [int(atlas_desc["width"]), int(atlas_desc["height"])],
        "exports": exported,
        "written_files": written_files,
        "preview_views": preview_views,
        "studio_views": studio_views,
    }


class SyncCraftWorldPipeline:
    """Production bridge pairing ``SparkNative`` (or callable) with SyncCraft's bounded repair & exporters."""

    def __init__(
        self,
        model_or_fn: Any,
        *,
        config: Mapping[str, Any] | None = None,
        default_max_new_tokens: int = 3072,
        enable_thinking: bool = True,
    ) -> None:
        self.model_or_fn = model_or_fn
        self.config = dict(config or {})
        self.default_max_new_tokens = int(default_max_new_tokens)
        self.enable_thinking = bool(enable_thinking)

    def _invoke_model(
        self,
        prompt: str,
        *,
        images: Sequence[Any] | None = None,
        attempt: int = 0,
        max_new_tokens: int | None = None,
    ) -> tuple[str, int]:
        max_toks = int(max_new_tokens or self.default_max_new_tokens)
        if hasattr(self.model_or_fn, "answer") and callable(self.model_or_fn.answer):
            from .inference import generate

            media = _prepare_media_from_pil_images(images)
            has_img = bool(media.get("images"))
            p_low = prompt.lower()
            p_head = prompt.split("```", 1)[0].lower()
            wants_strict = bool(re.search(r"audit_profile\s*[:=]\s*['\"]?strict['\"]?", p_head))
            has_sculpted_kw = any(
                kw in p_head
                for kw in (
                    "sculpted",
                    "blockbench",
                    "vibrant",
                    "dragon",
                    "wyvern",
                    "drake",
                    "diorama",
                    "size_step",
                    "stage-c",
                    "stage-e",
                    "dsl_version: \"4.0\"",
                    "dsl_version: '4.0'",
                    "@assembly",
                )
            )
            explicit_classic_v32 = (not has_sculpted_kw) and ("block-dsl v3.2" in p_head or "bedrock_entity" in p_head)
            wants_sculpted = (not wants_strict) and (not explicit_classic_v32) and (
                has_sculpted_kw or str(self.config.get("style_mode", "")).lower() == "sculpted"
            )
            if wants_sculpted:
                eff_prompt = prompt
                eff_sys = SYNCCRAFT_SCULPTED_COMPACT_RULES if attempt == 0 else None
            else:
                eff_prompt = (
                    f"{prompt.rstrip()}\n\n{SYNCCRAFT_V32_COMPACT_RULES}"
                    if (has_img and "Mandatory " not in prompt)
                    else prompt
                )
                eff_sys = None if has_img else SYNCCRAFT_V32_GRAMMAR_SYSTEM_PROMPT
            use_think = bool(
                self.enable_thinking
                and attempt == 0
                and (has_img or wants_sculpted or bool(self.config.get("pipeline_enable_thinking", False)))
            )
            is_repair_sample = int(attempt) >= 1
            is_stage_c_upgrade = wants_sculpted and has_img and ("stage-c" in p_low or "upgrade" in p_low or "blockout" in p_low)
            eff_task = (
                ("synccraft_sculpted_stage_progression" if is_stage_c_upgrade else ("synccraft_sculpted_multiview_inverse" if wants_sculpted else "synccraft_multiview_inverse_dsl"))
                if has_img
                else ("synccraft_sculpted_style_creation" if wants_sculpted else "synccraft_artist_assistant")
            )
            raw_text = str(
                generate(
                    self.model_or_fn,
                    eff_prompt,
                    media=media,
                    enable_thinking=use_think,
                    max_new_tokens=max_toks,
                    max_think_tokens=384 if use_think else None,
                    do_sample=is_repair_sample,
                    temperature=min(0.65, 0.35 + 0.10 * int(attempt)) if is_repair_sample else 1.0,
                    top_p=0.92 if is_repair_sample else 1.0,
                    repetition_penalty=1.06 if (wants_sculpted or is_repair_sample) else 1.04,
                    task=eff_task,
                    system_prompt=eff_sys,
                )
            )
            tok_count = len(self.model_or_fn.tokenizer.encode(raw_text, add_special_tokens=False))
            return _extract_clean_blockdsl(raw_text), max(1, tok_count)
        if callable(self.model_or_fn):
            out = self.model_or_fn(prompt, images, attempt)
            if isinstance(out, tuple) and len(out) == 2:
                raw_s = str(out[0])
                return _extract_clean_blockdsl(raw_s), int(out[1])
            raw_s = str(out)
            return _extract_clean_blockdsl(raw_s), max(1, (len(raw_s) + 3) // 4)
        raise TypeError(f"Unsupported model_or_fn {type(self.model_or_fn).__name__}; expected SparkNative or callable")

    def generate_3d_asset(
        self,
        prompt: str,
        *,
        images: Sequence[Any] | None = None,
        prompt_spec: Mapping[str, Any] | str | None = None,
        asset_name: str = "synccraft_generated_asset",
        output_dir: str | Path | None = None,
        export_formats: Sequence[str] = ("bbmodel", "bedrock", "gltf_pbr"),
        max_reaudits: int = 3,
        max_projection_steps_per_reaudit: int = 8,
        max_new_tokens: int | None = None,
        min_aesthetic_score: float | None = None,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Generate, repair (<= 3 re-audits + PGD + model feedback), aesthetic-critique, bake, and export a 3D asset."""
        ensure_synccraft_importable()
        import io
        from PIL import Image
        from synccraft.dsl.parser import format_program
        from synccraft.evaluation.budget import CostLedger
        from synccraft.modeling.repair_loop import run_bounded_repair_policy

        ledger = CostLedger()
        raw_generations: list[str] = []

        def _initial_gen(_spec: Any = None) -> tuple[str, int]:
            txt, toks = self._invoke_model(prompt, images=images, attempt=0, max_new_tokens=max_new_tokens)
            raw_generations.append(txt)
            return txt, toks

        def _repair_callback(cur_dsl: str, feedback: dict[str, Any], attempt_idx: int) -> tuple[str, int]:
            rep_prompt = _build_repair_prompt(prompt, cur_dsl, feedback, attempt_idx)
            txt, toks = self._invoke_model(rep_prompt, images=images, attempt=attempt_idx, max_new_tokens=max_new_tokens)
            raw_generations.append(txt)
            return txt, toks

        is_authored_prof = bool(
            re.search(r"audit_profile\s*[:=]\s*['\"]?authored['\"]?", prompt)
        )
        enforce_contact = not is_authored_prof

        repair_result = run_bounded_repair_policy(
            _initial_gen,
            prompt_spec if prompt_spec is not None else {},
            ledger,
            max_reaudits=int(max_reaudits),
            model_repair_fn=_repair_callback,
            max_projection_steps_per_reaudit=int(max_projection_steps_per_reaudit),
            enforce_physical_contact=enforce_contact,
        )

        aesthetic_eval: dict[str, Any] | None = None
        sculpted_eval: dict[str, Any] | None = None
        aesthetic_upgraded = False
        eff_min_aes = (
            float(min_aesthetic_score)
            if min_aesthetic_score is not None
            else (float(self.config["min_aesthetic_score"]) if "min_aesthetic_score" in self.config else None)
        )

        if repair_result["passed"] and repair_result["program"] is not None:
            aesthetic_eval = evaluate_3d_asset_aesthetics(repair_result["program"], prompt=prompt)
            try:
                from .sculpted_style_corpus import evaluate_sculpted_blockbench_style

                sculpted_eval = evaluate_sculpted_blockbench_style(repair_result["program"], prompt=prompt)
            except Exception:
                sculpted_eval = None
            p_head_g = prompt.split("```", 1)[0].lower()
            wants_sculpted_g = (not bool(re.search(r"audit_profile\s*[:=]\s*['\"]?strict['\"]?", p_head_g))) and (
                str(self.config.get("style_mode", "")).lower() == "sculpted"
                or any(kw in p_head_g for kw in ("sculpted", "blockbench", "vibrant", "dragon", "wyvern", "drake", "diorama", "size_step", "stage-c", "stage-e"))
            )
            needs_sculpted_upgrade = bool(
                wants_sculpted_g
                and sculpted_eval is not None
                and (sculpted_eval["is_stiff_or_voxel_like"] or float(sculpted_eval["sculpted_score"]) < 0.70)
            )
            needs_aes_upgrade = bool(
                eff_min_aes is not None
                and (aesthetic_eval["is_toy_like"] or float(aesthetic_eval["aesthetic_score"]) < eff_min_aes)
            )
            if (needs_sculpted_upgrade or needs_aes_upgrade) and int(repair_result["reaudits_used"]) < int(max_reaudits):
                rem_reaudits = max(1, int(max_reaudits) - int(repair_result["reaudits_used"]))
                preview_imgs = images
                if not preview_imgs:
                    try:
                        if wants_sculpted_g:
                            from .sculpted_style_corpus import render_sculpted_turnaround_views

                            png_map, _ = render_sculpted_turnaround_views(repair_result["program"], panel_size=(192, 192))
                        else:
                            ws_prev = build_default_world_state_for_program(repair_result["program"], seed=seed)
                            png_map, _, _, _ = render_world_views_rgba(ws_prev, panel_size=(144, 144))
                        preview_imgs = [Image.open(io.BytesIO(png_map[v])) for v in ("south", "right", "top", "iso")]
                    except Exception:
                        preview_imgs = None
                if wants_sculpted_g:
                    aes_prompt = (
                        "Critique this stiff Stage-C axis-aligned blockout from its 4 views (`south`, `right`, `top`, `iso`) "
                        f"and upgrade it into a vibrant Stage-E sculpted Blockbench asset ({prompt.strip()}):\n"
                        f"```blockdsl\n{format_program(repair_result['program']).strip()}\n```"
                    )
                else:
                    aes_prompt = (
                        f"Your initial SyncCraft v3.2 asset compiled cleanly ({aesthetic_eval['box_count']} boxes, "
                        f"aesthetic_score={aesthetic_eval['aesthetic_score']:.3f}, grade={aesthetic_eval['grade']!r}), "
                        "but failed the artist-grade richness gate:\n"
                        + "\n".join(f"- {n}" for n in aesthetic_eval["critique_notes"])
                        + f"\n\nOriginal request:\n{prompt.strip()}\n\n"
                        f"Initial Block-DSL blockout:\n```blockdsl\n{format_program(repair_result['program'])}\n```\n\n"
                        "Upgrade this blockout into an artist-grade 18–42 box articulated SyncCraft Block-DSL v3.2 asset "
                        "with complete cranial/head anatomy, tapered trunk/pelvis, secondary armor/trim, >=4 OKLCH `_mat` ramps, "
                        "and zero 15-axis SAT collisions."
                    )

                def _upgrade_gen(_spec: Any = None) -> tuple[str, int]:
                    txt, toks = self._invoke_model(aes_prompt, images=preview_imgs, attempt=0, max_new_tokens=max_new_tokens)
                    raw_generations.append(txt)
                    return txt, toks

                upgrade_res = run_bounded_repair_policy(
                    _upgrade_gen,
                    prompt_spec if prompt_spec is not None else {},
                    ledger,
                    max_reaudits=rem_reaudits - 1,
                    model_repair_fn=_repair_callback,
                    max_projection_steps_per_reaudit=int(max_projection_steps_per_reaudit),
                    enforce_physical_contact=enforce_contact,
                )
                if upgrade_res["passed"] and upgrade_res["program"] is not None:
                    cand_aes = evaluate_3d_asset_aesthetics(upgrade_res["program"], prompt=prompt)
                    cand_sc = None
                    try:
                        from .sculpted_style_corpus import evaluate_sculpted_blockbench_style

                        cand_sc = evaluate_sculpted_blockbench_style(upgrade_res["program"], prompt=prompt)
                    except Exception:
                        cand_sc = None
                    improved_sc = (
                        wants_sculpted_g
                        and cand_sc is not None
                        and sculpted_eval is not None
                        and float(cand_sc["sculpted_score"]) > float(sculpted_eval["sculpted_score"])
                    )
                    if improved_sc or float(cand_aes["aesthetic_score"]) > float(aesthetic_eval["aesthetic_score"]):
                        repair_result = {
                            **upgrade_res,
                            "reaudits_used": int(repair_result["reaudits_used"]) + 1 + int(upgrade_res["reaudits_used"]),
                        }
                        aesthetic_eval = cand_aes
                        sculpted_eval = cand_sc
                        aesthetic_upgraded = True

        bundle_report: dict[str, Any] | None = None
        if repair_result["passed"] and repair_result["program"] is not None:
            bundle_report = export_program_bundle(
                repair_result["program"],
                asset_name=asset_name,
                output_dir=output_dir,
                formats=export_formats,
                seed=seed,
            )

        summary = {
            "passed": bool(repair_result["passed"]),
            "initial_passed": bool(repair_result["initial_passed"]),
            "reaudits_used": int(repair_result["reaudits_used"]),
            "max_reaudits": int(repair_result["max_reaudits"]),
            "syntax_valid": bool(repair_result["syntax_valid"]),
            "audit_passed": bool(repair_result["audit_passed"]),
            "prompt_adherence_passed": bool(repair_result["prompt_adherence_passed"]),
            "failure_reason": repair_result["failure_reason"],
            "boxes": int(repair_result["boxes"]),
            "bones": int(repair_result["bones"]),
            "aesthetic_upgraded": aesthetic_upgraded,
            "aesthetic_evaluation": aesthetic_eval,
            "sculpted_evaluation": sculpted_eval,
            "canonical_ast_sha256": repair_result["canonical_ast_sha256"],
            "final_dsl": repair_result["final_dsl"],
            "raw_generation_count": len(raw_generations),
            "cost_ledger": ledger.to_dict(),
            "history": repair_result["history"],
            "bundle": bundle_report,
        }
        if output_dir is not None:
            write(Path(output_dir) / f"{_safe_slug(asset_name)}_pipeline_report.json", _json_safe(summary))
        return summary

    def critique_and_upgrade_asset(
        self,
        project: Any,
        *,
        format_kind: str = "blockdsl",
        animation_payload: dict[str, Any] | None = None,
        provenance: Any | None = None,
        license_policy: str | None = None,
        upgrade_goal: str = " Upgrade into an audited textured & articulated SyncCraft v3.2 asset.",
        use_model_first: bool = True,
        asset_name: str = "upgraded_asset",
        output_dir: str | Path | None = None,
        export_formats: Sequence[str] = ("bbmodel", "bedrock", "gltf_pbr"),
        max_reaudits: int = 3,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Ingest an existing artist .bbmodel / .geo.json / Java JSON / DSL, critique it, and upgrade + export."""
        ensure_synccraft_importable()
        from synccraft.dsl.parser import format_program
        from synccraft.world.artist_loop import (
            analyze_artist_project,
            ingest_artist_project_record,
            run_looped_artist_enhancement,
        )

        ingested = ingest_artist_project_record(
            project,
            format_kind=format_kind,
            animation_payload=animation_payload,
            provenance=provenance,
            license_policy=license_policy,
        )
        base_prog = ingested["program"]
        critique = analyze_artist_project(base_prog)
        critique_dict = critique.to_dict()

        if use_model_first:
            upgrade_prompt = (
                f"Upgrade this ingested {format_kind} artist blockout into a production SyncCraft v3.2 asset.\n"
                f"Goal: {upgrade_goal.strip()}\n"
                f"Critique summary: boxes={critique_dict['box_count']}, readiness={critique_dict['readiness_score']:.3f}, "
                f"static_audit_passed={critique_dict['static_audit_passed']}\n"
                f"Input Block-DSL:\n```blockdsl\n{format_program(base_prog)}\n```\n"
                "Think step-by-step inside <think>...</think> and emit the upgraded ```blockdsl ... ``` program."
            )
            model_res = self.generate_3d_asset(
                upgrade_prompt,
                prompt_spec={"min_boxes": max(1, int(critique_dict["box_count"]))},
                asset_name=asset_name,
                output_dir=output_dir,
                export_formats=export_formats,
                max_reaudits=max_reaudits,
                seed=seed,
            )
            if model_res["passed"]:
                return {
                    "route": "model_plus_bounded_repair",
                    "license_policy": ingested["license_policy"],
                    "commercial_weights_eligible": ingested["commercial_weights_eligible"],
                    "initial_critique": critique_dict,
                    "result": model_res,
                }

        # Deterministic SyncCraft artist-loop fallback guarantees a working output asset
        loop_res = run_looped_artist_enhancement(
            project,
            format_kind=format_kind,
            animation_payload=animation_payload,
            provenance=provenance,
            license_policy=license_policy,
            max_iterations=4,
        )
        upgraded_prog = loop_res["final_program"]
        bundle = export_program_bundle(
            upgraded_prog,
            asset_name=asset_name,
            output_dir=output_dir,
            formats=export_formats,
            seed=seed,
        )
        return {
            "route": "deterministic_artist_loop_fallback",
            "license_policy": ingested["license_policy"],
            "commercial_weights_eligible": ingested["commercial_weights_eligible"],
            "initial_critique": critique_dict,
            "final_critique": loop_res["final_critique"],
            "result": {
                "passed": True,
                "final_dsl": format_program(upgraded_prog),
                "boxes": bundle["boxes"],
                "bones": bundle["bones"],
                "bundle": bundle,
            },
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate & export SyncCraft v3.2 3D assets with Spark-X2.6-VL.")
    parser.add_argument("--config", required=True, help="Campaign JSON config path.")
    parser.add_argument("--checkpoint", default=None, help="Exported model dir (defaults to <work_root>/final-model).")
    parser.add_argument("--prompt", required=True, help="3D asset generation prompt.")
    parser.add_argument("--image", action="append", default=[], help="Optional reference image path(s).")
    parser.add_argument("--asset-name", default="spark_synccraft_asset", help="Output asset slug.")
    parser.add_argument("--out-dir", required=True, help="Directory to write .bbmodel / .geo.json / .gltf / PNGs.")
    parser.add_argument(
        "--formats",
        default="bbmodel,bedrock,gltf_pbr",
        help="Comma-separated subset of bbmodel,bedrock,geckolib,java,gltf_pbr,labpbr.",
    )
    parser.add_argument("--max-reaudits", type=int, default=3)
    args = parser.parse_args(list(argv) if argv is not None else None)

    from .common import local, read, validate_config
    from .inference import load

    cfg = validate_config(read(args.config))
    root = local(cfg["work_root"], execution_target=cfg.get("execution_target", "local_workstation"))
    ckpt = Path(args.checkpoint) if args.checkpoint else (root / "final-model")
    model = load(ckpt, device="cuda")
    pipeline = SyncCraftWorldPipeline(model, config=cfg)
    formats = tuple(x.strip() for x in args.formats.split(",") if x.strip())
    res = pipeline.generate_3d_asset(
        args.prompt,
        images=args.image or None,
        asset_name=args.asset_name,
        output_dir=args.out_dir,
        export_formats=formats,
        max_reaudits=args.max_reaudits,
    )
    print(
        json.dumps(
            {
                "passed": res["passed"],
                "reaudits_used": res["reaudits_used"],
                "boxes": res["boxes"],
                "bones": res["bones"],
                "cost_ledger": res["cost_ledger"],
                "written_files": (res["bundle"] or {}).get("written_files", {}),
            },
            indent=2,
        )
    )
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
