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

SYNCCRAFT_V32_GRAMMAR_SYSTEM_PROMPT = """You author valid SyncCraft Block-DSL v3.2 3D assets. Every Block-DSL v3.2 program MUST strictly obey this syntax:
```blockdsl
@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, audit_profile: strict }
@palette {
  pelt_mat: ramp(base_lch=[0.64, 0.11, 42], steps=6)
  hoof_mat: ramp(base_lch=[0.4, 0.06, 132], steps=6)
}
@rig {
  root bone body_root {
    pivot_local: [0, 10, 0]
    box torso { size: [6, 5, 9], material: pelt_mat }
    bone fl_thigh attach(parent_box=torso, anchor=[-1, -0.6, -0.6], align=[1, 1, 0]) mirror_x(fr_thigh) {
      box fl_thigh_seg { size: [2, 4, 2], material: pelt_mat }
      bone fl_shin attach(parent_box=fl_thigh_seg, anchor=[0, -1, 0], align=[0, 1, 0]) {
        box fl_shin_seg { size: [2, 4, 2], material: hoof_mat }
      }
    }
    bone hl_thigh attach(parent_box=torso, anchor=[-1, -0.6, 0.6], align=[1, 1, 0]) mirror_x(hr_thigh) {
      box hl_thigh_seg { size: [2, 4, 2], material: pelt_mat }
      bone hl_shin attach(parent_box=hl_thigh_seg, anchor=[0, -1, 0], align=[0, 1, 0]) {
        box hl_shin_seg { size: [2, 4, 2], material: hoof_mat }
      }
    }
  }
}
@kinematics {
  ik_chain fl_leg_ik { upper: fl_thigh, lower: fl_shin, effector: [0, -4, 0] }
  ik_chain fr_leg_ik { upper: fr_thigh, lower: fr_thigh__fl_shin, effector: [0, -4, 0] }
  ik_chain hl_leg_ik { upper: hl_thigh, lower: hl_shin, effector: [0, -4, 0] }
  ik_chain hr_leg_ik { upper: hr_thigh, lower: hr_thigh__hl_shin, effector: [0, -4, 0] }
}
@behavior {
  clip walk_cycle {
    duration_s: 1.6
    ik_drive fl_leg_ik: foot_cycle(stride=1.5, lift=0.65, phase_deg=0)
    ik_drive hr_leg_ik: foot_cycle(stride=1.5, lift=0.65, phase_deg=0)
    ik_drive fr_leg_ik: foot_cycle(stride=1.5, lift=0.65, phase_deg=180)
    ik_drive hl_leg_ik: foot_cycle(stride=1.5, lift=0.65, phase_deg=180)
  }
}
```
Mandatory syntax rules:
1. `@meta` ONLY accepts `archetype`, `target_profile`, `texel_density`, and optional `audit_profile` (`strict` | `sculpted` | `authored`) — never `version` or `domain`.
2. `@palette` entries MUST use `mat_name: ramp(base_lch=[L, C, H], steps=6)` — never `#hex` or `rgb(...)`.
3. Box `size: [W, H, D]` MUST be 3 positive INTEGER literals in square brackets `[W, H, D]` (never floats or parentheses).
4. Child bones MUST attach cleanly via `attach(parent_box=<box>, anchor=[ax, ay, az], align=[lx, ly, lz])` so faces touch with 0 SAT penetration and 0 air gap.
5. Every bone and box identifier MUST be unique; when a bone declares `mirror_x(right_name)`, NEVER declare a separate `bone right_name` block (mirror_x expands it automatically)."""


def _extract_clean_blockdsl(raw_text: str) -> str:
    """Strip closed/unclosed `<think>` wrappers and Markdown fences deterministically."""
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
    if 0 < (open_braces - close_braces) <= 4:
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
    lines = [
        "Repair the SyncCraft v3.2 Block-DSL asset below so it compiles cleanly and passes all audits.",
        "Keep every integer box `size` tuple discrete and unchanged when fixing small pose/offset collisions unless syntax/adherence requires adding missing parts.",
        f"Original request:\n{original_prompt.strip()}",
        f"Repair attempt: {attempt}",
    ]
    if feedback.get("error"):
        lines.append(f"Compiler/audit error: {feedback['error']}")
    collisions = list(feedback.get("collision_pairs") or ())
    if collisions:
        formatted_cols = ", ".join(
            f"{c.get('a')} vs {c.get('b')} (depth={float(c.get('depth', 0.0)):.4f}u)"
            for c in collisions[:8]
        )
        lines.append(f"15-axis SAT collision pairs: {formatted_cols}")
    contacts = list(feedback.get("contact_violations") or ())
    if contacts:
        formatted_ct = ", ".join(
            f"{c.get('kind', 'violation')}:{c.get('bone') or c.get('box') or c}"
            for c in contacts[:8]
        )
        lines.append(f"Physical contact violations: {formatted_ct}")
    adh = list(feedback.get("prompt_violations") or ())
    if adh:
        lines.append(f"Prompt adherence violations: {', '.join(str(v) for v in adh[:8])}")
    lines.append(f"Current candidate Block-DSL:\n```blockdsl\n{candidate_dsl.strip()}\n```")
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
    if render_preview_views and out_path is not None:
        renders, _, _, _ = render_world_views_rgba(ws, panel_size=(192, 192))
        for vid, png_bytes in renders.items():
            vp = out_path / f"{slug}_preview_{vid}.png"
            vp.write_bytes(png_bytes)
            preview_views[vid] = str(vp)

    return {
        "asset_name": slug,
        "audit_profile": getattr(prog.meta, "audit_profile", "strict"),
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
            # Direct non-thinking generation avoids exhausting token budget inside open-ended `<think>` blocks
            use_think = bool(self.enable_thinking and attempt == 0 and self.config.get("pipeline_enable_thinking", False))
            raw_text = str(
                generate(
                    self.model_or_fn,
                    prompt,
                    media=media,
                    enable_thinking=use_think,
                    max_new_tokens=max_toks,
                    task="synccraft_multiview_inverse_dsl" if has_img else "synccraft_artist_assistant",
                    system_prompt=SYNCCRAFT_V32_GRAMMAR_SYSTEM_PROMPT,
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
        seed: int = 0,
    ) -> dict[str, Any]:
        """Generate, repair (<= 3 re-audits + PGD + model feedback), audit, bake, and export a 3D asset."""
        ensure_synccraft_importable()
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

        repair_result = run_bounded_repair_policy(
            _initial_gen,
            prompt_spec if prompt_spec is not None else {},
            ledger,
            max_reaudits=int(max_reaudits),
            model_repair_fn=_repair_callback,
            max_projection_steps_per_reaudit=int(max_projection_steps_per_reaudit),
            enforce_physical_contact=True,
        )

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
