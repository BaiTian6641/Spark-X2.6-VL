"""Formal `Blockbench_sculpted_style` v25 Engine-Agnostic Verifier (`G1-G12`, 4-Foot Grounding, 2-Pass Locate-Grid).

Pure-Python + NumPy/PIL verifier aligned directly with `Blockbench_sculpted_style-v25`:
- `G1-G12` deterministic geometry, proportion & Vigorous Carved Blockbench style-grammar gates
  (ported cleanly from `scripts/construct_check.py` v20/v25 for in-memory `.bbmodel` / SyncCraft scenes).
- Blockbench 5 `.bbmodel` exact 4-foot (`FL`, `FR`, `HL`, `HR`, `TAIL`) rest-pose & animation FK ground-locking
  audit (ported cleanly from `scripts/verify_bbmodel_ground.py` v23/v25).
- Analytic 3D->2D camera projection for `v25` Mandatory 2-Pass "Locate-First, Then Zoom" (`0.0-1.0` grid
  overlay + exact ground-truth `<box>` for head & `<point>` for visible eye socket under jittered framing).
"""
from __future__ import annotations

import collections
import io
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .sculpted_style_corpus import FACES, _face_vertices_local, _uv_corners_px, bake_sculpted_rgba_atlas
from .synccraft_bridge import ensure_synccraft_importable

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]
EYE3: Mat3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


@dataclass
class _BoneNode:
    name: str
    parent_name: str | None
    local_pos: Vec3
    local_rot: Vec3
    world_pos: Vec3
    world_rot: Mat3
    mirrored: bool


@dataclass
class _BoxElem:
    name: str
    bone_name: str
    size: Vec3
    center: Vec3
    local_center: Vec3
    local_rot: Vec3
    material: str
    is_plane: bool = False
    plane_normal: str | None = None


def _vec3(seq: Sequence[float]) -> Vec3:
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(a: Vec3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _mat_vec(m: Mat3, v: Vec3) -> Vec3:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def _mat_mul(a: Mat3, b: Mat3) -> Mat3:
    return tuple(
        tuple(a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] for j in range(3))
        for i in range(3)
    )  # type: ignore[return-value]


def _rot_zyx(rx: float, ry: float, rz: float) -> Mat3:
    ax, ay, az = math.radians(rx), math.radians(ry), math.radians(rz)
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    rx_m: Mat3 = ((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx))
    ry_m: Mat3 = ((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy))
    rz_m: Mat3 = ((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0))
    return _mat_mul(_mat_mul(rz_m, ry_m), rx_m)


def _apply_bind_chain(pt: Vec3, chain: list[tuple[Vec3, Mat3]]) -> Vec3:
    cur = pt
    for piv, rmat in reversed(chain):
        cur = _add(piv, _mat_vec(rmat, _sub(cur, piv)))
    return cur


def _min_singular_value_3d(pts: list[Vec3]) -> float:
    n = len(pts)
    if n < 3:
        return 0.0
    arr = np.asarray(pts, dtype=np.float64)
    arr = arr - arr.mean(axis=0, keepdims=True)
    _, svals, _ = np.linalg.svd(arr, full_matrices=False)
    return float(svals[-1]) if len(svals) >= 3 else 0.0


def _std(vals: list[float]) -> float:
    if not vals:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def _parse_bbmodel_dict(data: dict[str, Any]) -> tuple[list[_BoneNode], list[_BoxElem], list[_BoxElem], int]:
    elem_by_uuid: dict[str, dict[str, Any]] = {}
    mesh_count = 0
    for el in data.get("elements", []):
        if el.get("type", "cube") == "mesh":
            mesh_count += 1
            continue
        uid = el.get("uuid")
        if uid:
            elem_by_uuid[uid] = el
    group_by_uuid = {g["uuid"]: g for g in data.get("groups", []) if isinstance(g, dict) and g.get("uuid")}

    bones: list[_BoneNode] = []
    boxes: list[_BoxElem] = []
    planes: list[_BoxElem] = []

    def _emit_el(el: dict[str, Any], bname: str, bone_bind_piv: Vec3, chain: list[tuple[Vec3, Mat3]]) -> None:
        frm = _vec3(el.get("from", (0.0, 0.0, 0.0)))
        to = _vec3(el.get("to", (0.0, 0.0, 0.0)))
        sz = (abs(to[0] - frm[0]), abs(to[1] - frm[1]), abs(to[2] - frm[2]))
        bind_ctr = (0.5 * (frm[0] + to[0]), 0.5 * (frm[1] + to[1]), 0.5 * (frm[2] + to[2]))
        e_orig = _vec3(el.get("origin") or bind_ctr)
        e_rot = _vec3(el.get("rotation") or (0.0, 0.0, 0.0))
        r_el = _rot_zyx(*e_rot)
        posed_bind = _add(e_orig, _mat_vec(r_el, _sub(bind_ctr, e_orig)))
        wctr = _apply_bind_chain(posed_bind, chain)
        lctr = _sub(bind_ctr, bone_bind_piv)
        is_plane = min(sz) <= 0.05
        pnorm = None
        if is_plane:
            pnorm = "x" if sz[0] <= 0.05 else ("y" if sz[1] <= 0.05 else "z")
        be = _BoxElem(
            name=str(el.get("name", "cube")),
            bone_name=bname,
            size=sz,
            center=wctr,
            local_center=lctr,
            local_rot=e_rot,
            material="",
            is_plane=is_plane,
            plane_normal=pnorm,
        )
        if is_plane:
            planes.append(be)
        else:
            boxes.append(be)

    def _walk(
        node: Any,
        parent_name: str | None,
        parent_bind_piv: Vec3,
        parent_wrot: Mat3,
        anc_chain: list[tuple[Vec3, Mat3]],
    ) -> None:
        if isinstance(node, str):
            el = elem_by_uuid.get(node)
            if el is not None:
                _emit_el(el, parent_name or "root", parent_bind_piv, anc_chain)
            return
        if not isinstance(node, dict):
            return
        uid = node.get("uuid")
        gmeta = group_by_uuid.get(uid, {}) if uid else {}
        bname = str(node.get("name") or gmeta.get("name") or f"group_{len(bones)}")
        bind_piv = _vec3(node.get("origin") or gmeta.get("origin") or (0.0, 0.0, 0.0))
        lrot = _vec3(node.get("rotation") or gmeta.get("rotation") or (0.0, 0.0, 0.0))
        rmat = _rot_zyx(*lrot)
        wpos = _apply_bind_chain(bind_piv, anc_chain)
        wrot = _mat_mul(parent_wrot, rmat)
        lpos = _sub(bind_piv, parent_bind_piv) if parent_name else bind_piv
        is_mir = ("__" in bname) or bname.endswith("_r") or ("_r_" in bname) or bname.startswith("right_")
        bones.append(_BoneNode(bname, parent_name, lpos, lrot, wpos, wrot, is_mir))
        next_chain = anc_chain + [(bind_piv, rmat)]
        for ch in node.get("children", []):
            _walk(ch, bname, bind_piv, wrot, next_chain)

    for root in data.get("outliner", []):
        _walk(root, None, (0.0, 0.0, 0.0), EYE3, [])
    return bones, boxes, planes, mesh_count


def _infer_tier(total_elems: int, explicit_tier: str) -> str:
    if explicit_tier in ("small", "standard", "hero", "epic_boss"):
        return explicit_tier
    if total_elems < 45:
        return "small"
    if total_elems < 140:
        return "standard"
    if total_elems < 240:
        return "hero"
    return "epic_boss"


def run_v25_construct_check_bbmodel(
    bb: dict[str, Any], *, tier: str = "auto", allow_asymmetric_prop: bool = False
) -> dict[str, Any]:
    """Run exact `Blockbench_sculpted_style` v25 `G1-G12` deterministic audit on a `.bbmodel` dict."""
    bones, boxes, planes, mesh_count = _parse_bbmodel_dict(bb)
    plane_count = len(planes)
    total_elems = len(boxes) + plane_count
    eff_tier = _infer_tier(total_elems, tier)

    par = {b.name: b.parent_name for b in bones}
    bone_map = {b.name: b for b in bones}
    children_map: dict[str, list[str]] = collections.defaultdict(list)
    for b in bones:
        if b.parent_name:
            children_map[b.parent_name].append(b.name)

    def in_wing(bn: str | None) -> bool:
        while bn:
            low = bn.lower()
            if any(low.startswith(k) or f"__{k}" in low for k in ("wing", "left_wing", "right_wing")):
                return True
            bn = par.get(bn)
        return False

    def subtree_bones(root_bn: str) -> list[str]:
        out = [root_bn]
        q = [root_bn]
        while q:
            cur = q.pop(0)
            for ch in children_map.get(cur, []):
                out.append(ch)
                q.append(ch)
        return out

    non_wing_boxes = [b for b in boxes if not in_wing(b.bone_name)]
    vol_all = sorted((b.size[0] * b.size[1] * b.size[2] for b in boxes), reverse=True)
    vol_nw = sorted((b.size[0] * b.size[1] * b.size[2] for b in non_wing_boxes), reverse=True)
    per_bone = collections.Counter(b.bone_name for b in boxes)
    non_head_bones = [cnt for bn, cnt in per_bone.items() if "head" not in bn.lower()]
    avg_non_head = (sum(non_head_bones) / len(non_head_bones)) if non_head_bones else 0.0

    key = lambda b, sx: (round(sx * b.center[0], 1), round(b.center[1], 1), round(b.center[2], 1))
    left = sorted(key(b, -1.0) for b in boxes if b.center[0] < -0.05)
    right = sorted(key(b, 1.0) for b in boxes if b.center[0] > 0.05)
    mirror_ok = left == right

    top10_nw = (sum(vol_nw[:10]) / sum(vol_nw)) if vol_nw else 0.0
    top10_all = (sum(vol_all[:10]) / sum(vol_all)) if vol_all else 0.0

    all_elems = boxes + planes
    if all_elems:
        xs_all = [e.center[0] - 0.5 * e.size[0] for e in all_elems] + [e.center[0] + 0.5 * e.size[0] for e in all_elems]
        ys_all = [e.center[1] - 0.5 * e.size[1] for e in all_elems] + [e.center[1] + 0.5 * e.size[1] for e in all_elems]
        zs_all = [e.center[2] - 0.5 * e.size[2] for e in all_elems] + [e.center[2] + 0.5 * e.size[2] for e in all_elems]
        y_ground = min(ys_all)
        span = (max(xs_all) - min(xs_all), max(ys_all) - y_ground, max(zs_all) - min(zs_all))
    else:
        y_ground = 0.0
        span = (0.0, 0.0, 0.0)

    head_core_boxes = [
        b
        for b in non_wing_boxes
        if any(k in b.bone_name.lower() for k in ("head", "jaw", "maxilla", "mandible", "snout", "muzzle", "cheek", "brow"))
        and not any(k in b.name.lower() for k in ("horn", "ear", "collar", "frill"))
    ]
    if head_core_boxes:
        hx = max(b.center[0] + 0.5 * b.size[0] for b in head_core_boxes) - min(b.center[0] - 0.5 * b.size[0] for b in head_core_boxes)
        hy = max(b.center[1] + 0.5 * b.size[1] for b in head_core_boxes) - min(b.center[1] - 0.5 * b.size[1] for b in head_core_boxes)
        hz = max(b.center[2] + 0.5 * b.size[2] for b in head_core_boxes) - min(b.center[2] - 0.5 * b.size[2] for b in head_core_boxes)
        head_lh = hz / max(1e-6, hy)
        head_wh = hx / max(1e-6, hy)
    else:
        head_lh = head_wh = 0.0

    chest_w = max((b.size[0] for b in non_wing_boxes if "chest" in b.bone_name.lower() or "chest" in b.name.lower()), default=0.0)
    waist_w = max((b.size[0] for b in non_wing_boxes if "waist" in b.bone_name.lower() or "loin" in b.name.lower()), default=0.0)
    v_taper = (chest_w / waist_w) if waist_w > 0 else 0.0

    torso_boxes = [
        b
        for b in non_wing_boxes
        if any(k in b.bone_name.lower() for k in ("chest", "torso", "body", "waist", "belly", "pelvis", "hips"))
    ]
    if torso_boxes:
        y_belly_min = min(b.center[1] - 0.5 * b.size[1] for b in torso_boxes)
        y_torso_top = max(b.center[1] + 0.5 * b.size[1] for b in torso_boxes)
        belly_clearance_ratio = (y_belly_min - y_ground) / max(1e-6, y_torso_top - y_ground)
    else:
        belly_clearance_ratio = 0.0

    findings: list[str] = []
    top10_floor = {"small": 0.55, "standard": 0.48, "hero": 0.36, "epic_boss": 0.26}.get(eff_tier, 0.45)
    avg_box_cap = 4.0 if eff_tier == "epic_boss" else 3.5

    # G1
    if top10_nw < top10_floor:
        findings.append(f"[FAIL G1: Volume Hierarchy] Top-10 non-wing={top10_nw:.1%} < {top10_floor:.0%} ({eff_tier})")
    if avg_non_head > avg_box_cap:
        findings.append(f"[FAIL G1: Shard Density] Non-head boxes/bone={avg_non_head:.2f} > {avg_box_cap:.1f}")
    for bn, cnt in per_bone.items():
        limit = 14 if "head" in bn.lower() else (10 if eff_tier == "epic_boss" else 8)
        if cnt > limit:
            findings.append(f"[WARN G1: Bone Bloat] '{bn}' has {cnt} boxes > {limit}")
    if not mirror_ok and not allow_asymmetric_prop:
        findings.append(f"[FAIL G1: Symmetry] Mirror parity broken ({len(left)}L vs {len(right)}R)")

    # G2: Wedding-cake snout
    snout_boxes = [
        b for b in boxes if any(k in b.name.lower() for k in ("snout", "muzzle", "maxilla")) and abs(b.local_center[0]) < 0.05
    ]
    if len(snout_boxes) >= 2 and all(abs(b.local_rot[1]) < 1.0 for b in snout_boxes):
        has_v_rail = any(
            any(k in b.name.lower() for k in ("maxilla", "snout_wall", "lip_rail", "snout_side")) and abs(b.local_rot[1]) >= 10.0
            for b in boxes
        )
        if not has_v_rail:
            findings.append("[WARN G2: Wedding-Cake Snout] Centred yaw=0 snout stack without yawed V-rail maxilla walls")

    # G3: Crown congestion
    head_horns = [
        b
        for b in boxes
        if "head" in b.bone_name.lower()
        and any(k in b.name.lower() for k in ("horn", "crest", "crystal"))
        and "collar" not in b.name.lower()
        and "jaw" not in b.name.lower()
    ]
    if len(head_horns) > 8:
        findings.append(f"[WARN G3: Crown Congestion] {len(head_horns)} horn/crest/crystal boxes on head")

    # G4: Wing 3D camber & knuckles
    left_wing_bones = [b for b in bones if in_wing(b.name) and b.world_pos[0] < -0.05 and "__" not in b.name]
    camber_ratio = None
    if len(left_wing_bones) >= 4:
        pts = [b.world_pos for b in left_wing_bones]
        span_w = max(p[0] for p in pts) - min(p[0] for p in pts)
        if span_w > 10.0:
            camber_ratio = _min_singular_value_3d(pts) / span_w
            if camber_ratio < 0.05:
                findings.append(f"[WARN G4: 2D Plank Wing] Camber ratio={camber_ratio:.3f} < 0.050")
        by_parent: dict[str, list[_BoneNode]] = collections.defaultdict(list)
        for b in left_wing_bones:
            if b.parent_name:
                by_parent[b.parent_name].append(b)
        for p_name, children in by_parent.items():
            f_kids = [c for c in children if any(k in c.name.lower() for k in ("f1", "f2", "f3", "f4", "finger", "wing_segment"))]
            if len(f_kids) >= 3:
                dists = [_norm(_sub(f_kids[i].local_pos, f_kids[i + 1].local_pos)) for i in range(len(f_kids) - 1)]
                if min(dists) < 1.5:
                    findings.append(f"[WARN G4: Co-located Wing Knuckles] '{p_name}' min spacing {min(dists):.2f}u < 1.5u")

    # G5: Crate-train spine
    torso_neck_boxes = [
        b for b in non_wing_boxes if any(k in b.bone_name.lower() for k in ("chest", "waist", "hips", "neck"))
    ]
    core_cubes = [
        b
        for b in torso_neck_boxes
        if "core" in b.name.lower()
        and max(b.size) / max(min(b.size), 1e-6) < 1.35
        and all(abs(r) < 1.0 for r in b.local_rot)
    ]
    if len(core_cubes) >= 4:
        has_gussets = any(
            any(k in b.name.lower() for k in ("gusset", "trapezius", "clavicle", "rib_slope", "lat_wedge"))
            for b in torso_neck_boxes
        )
        if not has_gussets:
            findings.append(f"[WARN G5: Crate-Train Spine] {len(core_cubes)} unchamfered cores without gussets")

    # G6: Hip-skirt & snowshoe
    for b in non_wing_boxes:
        low = b.name.lower()
        if any(k in low for k in ("haunch", "hip_plate", "hip_skirt")) and b.center[0] < -0.05:
            if abs(b.local_rot[0]) < 8.0 and abs(b.local_rot[2]) < 10.0:
                findings.append(f"[WARN G6: Pasted Hip-Skirt] '{b.name}' nearly upright")
        if any(k in low for k in ("palm", "foot_base", "sole")) and b.center[0] < -0.05:
            if b.size[1] / max(b.size[0], b.size[2], 1e-6) < 0.38:
                findings.append(f"[WARN G6: Snowshoe Plank Foot] '{b.name}' H/L < 0.38")

    # G7: Texel sizing & bone naming
    off_grid = [b.name for b in boxes if any(abs((d * 2.0) - round(d * 2.0)) > 1e-3 for d in b.size)]
    if off_grid:
        findings.append(f"[WARN G7: Off-Grid Texel Sizes] {len(off_grid)} box(es) non-0.5u")
    bad_bone_names = [b.name for b in bones if any(c.isupper() or c.isspace() for c in b.name)]
    if bad_bone_names:
        findings.append(f"[WARN G7: Bone Naming Hygiene] {len(bad_bone_names)} non-snake_case bone(s)")

    # G8: Epic floor & stick-limb check
    if eff_tier == "epic_boss" and total_elems < 240:
        findings.append(f"[FAIL G8: Under-Complex Epic Boss] total_elems={total_elems} < 240")
    if eff_tier in ("hero", "epic_boss"):
        stick_bones: list[str] = []
        shaft_keywords = ("arm_l_upper", "arm_l_fore", "leg_l_thigh", "leg_l_shin", "leg_l_ankle", "wing_l_upper", "wing_l_fore")
        for b in bones:
            low = b.name.lower()
            if "__" in low or b.world_pos[0] > -0.05:
                continue
            if any(k in low for k in shaft_keywords):
                direct_boxes = [bx for bx in boxes if bx.bone_name == b.name]
                orn_kids = [
                    ch
                    for ch in children_map.get(b.name, [])
                    if not any(sk in ch.lower() for sk in ("fore", "shin", "ankle", "paw", "hand", "foot", "palm", "f1", "f2", "f3", "f4", "thumb"))
                ]
                seg_boxes = direct_boxes + [bx for bx in boxes if bx.bone_name in orn_kids]
                max_len = max((max(bx.size) for bx in direct_boxes), default=0.0)
                if len(seg_boxes) == 1:
                    s0 = seg_boxes[0].size
                    if max(s0) / max(1e-6, sorted(s0)[1]) >= 2.6:
                        stick_bones.append(f"{b.name}({max(s0):.0f}u)")
                elif eff_tier == "epic_boss" and len(seg_boxes) == 2 and max_len >= 18.0:
                    stick_bones.append(f"{b.name}({max_len:.0f}u)")
        if stick_bones:
            sev = "FAIL" if eff_tier == "epic_boss" else "WARN"
            findings.append(f"[{sev} G8: Stick-Limb Shafts] {', '.join(stick_bones)}")

    # G9: Articulated extremities
    if eff_tier in ("hero", "epic_boss"):
        extremity_roots = [
            b
            for b in bones
            if "__" not in b.name
            and b.world_pos[0] < -0.05
            and any(b.name.lower().startswith(k) or b.name.lower().endswith(k) for k in ("paw_l", "hand_l", "foot_l", "front_foot_l", "hind_foot_l"))
        ]
        for ext in extremity_roots:
            sub_bns = subtree_bones(ext.name)
            digit_bones = [
                bone_map[bn]
                for bn in sub_bns
                if bn != ext.name and any(k in bn.lower() for k in ("toe", "finger", "digit", "thumb", "dewclaw", "hallux", "pollex"))
            ]
            root_digits = [d for d in digit_bones if d.parent_name == ext.name]
            min_digits = 4 if eff_tier == "epic_boss" else 3
            sev = "FAIL" if eff_tier == "epic_boss" else "WARN"
            if len(root_digits) < min_digits:
                findings.append(f"[{sev} G9: Under-Articulated Extremity] '{ext.name}' has {len(root_digits)} < {min_digits}")
            if len(root_digits) >= 3:
                yaws = [d.local_rot[1] + d.local_rot[2] for d in root_digits]
                if max(yaws) - min(yaws) < 20.0:
                    findings.append(f"[{sev} G9: Parallel Peg Digits] '{ext.name}' spread < 20 deg")

    # G10: Wing membrane chord coverage
    chord_coverage = None
    left_wing_planes = [p for p in planes if in_wing(p.bone_name) and p.center[0] < -0.05 and "__" not in p.name]
    if len(left_wing_bones) >= 4 and left_wing_planes:
        finger_tips: list[tuple[str, Vec3]] = []
        for idx_tag in ("f1", "f2", "f3", "f4", "wing_segment_1", "wing_segment_2", "wing_segment_3", "wing_segment_4"):
            f_boxes = [
                bx
                for bx in boxes
                if in_wing(bx.bone_name)
                and bx.center[0] < -0.05
                and "__" not in bx.name
                and idx_tag in bx.bone_name.lower()
                and "armor" not in bx.name.lower()
                and "claw" not in bx.name.lower()
            ]
            if f_boxes:
                distal = min(f_boxes, key=lambda bx: bx.center[0])
                bnode = bone_map.get(distal.bone_name)
                half_len = 0.5 * max(distal.size)
                if bnode is not None:
                    ax_x = (bnode.world_rot[0][0], bnode.world_rot[1][0], bnode.world_rot[2][0])
                    sign = -1.0 if ax_x[0] > 0 else 1.0
                    tip_pt = _add(distal.center, (sign * ax_x[0] * half_len, sign * ax_x[1] * half_len, sign * ax_x[2] * half_len))
                else:
                    tip_pt = distal.center
                finger_tips.append((idx_tag, tip_pt))
        if len(finger_tips) >= 3:
            tip_gaps = [_norm(_sub(finger_tips[i][1], finger_tips[i + 1][1])) for i in range(len(finger_tips) - 1)]
            max_tip_gap = max(tip_gaps)
            max_chord = max(sorted(p.size, reverse=True)[1] for p in left_wing_planes)
            chord_coverage = max_chord / max(1e-6, max_tip_gap)
            if chord_coverage < 0.88:
                sev = "FAIL" if eff_tier in ("hero", "epic_boss") else "WARN"
                findings.append(f"[{sev} G10: Gapped Wing Bays] chord_coverage={chord_coverage:.2f}x < 0.88x")

    # G11: Anti-Blender mesh/wafer ban
    if mesh_count > 0:
        findings.append(f"[FAIL G11: Fake-Blender Mesh] {mesh_count} polygon meshes")
    wafer_by_bone = collections.Counter(b.bone_name for b in boxes if 0.05 < min(b.size) < 1.0)
    wafer_abusers = [(bn, cnt) for bn, cnt in wafer_by_bone.items() if cnt >= 6 and "__" not in bn]
    if wafer_abusers:
        findings.append(f"[FAIL G11: Fake-Blender Micro-Slice Curve] {wafer_abusers[:3]}")

    # G12: Coiled predatory weight
    has_forelimbs = any(any(k in b.name.lower() for k in ("arm_l", "foreleg_l", "front_leg_l")) for b in bones)
    has_hindlimbs = any(any(k in b.name.lower() for k in ("leg_l", "hindleg_l", "thigh_l")) for b in bones)
    if has_forelimbs and has_hindlimbs and torso_boxes and belly_clearance_ratio > 0.58:
        findings.append(f"[WARN G12: Stilt-Leg Toy Posture] belly_clearance={belly_clearance_ratio:.2f} > 0.58")

    fails = [f for f in findings if f.startswith("[FAIL")]
    warns = [f for f in findings if f.startswith("[WARN")]
    return {
        "tier": eff_tier,
        "total_elements": total_elems,
        "boxes": len(boxes),
        "planes": plane_count,
        "bones": len(bones),
        "avg_non_head_boxes_per_bone": round(avg_non_head, 2),
        "top10_nw_volume": round(top10_nw, 4),
        "top10_all_volume": round(top10_all, 4),
        "mirror_ok": mirror_ok,
        "span": [round(span[0], 1), round(span[1], 1), round(span[2], 1)],
        "head_lh": round(head_lh, 2),
        "head_wh": round(head_wh, 2),
        "chest_waist": round(v_taper, 2),
        "belly_clearance": round(belly_clearance_ratio, 2),
        "camber_ratio": None if camber_ratio is None else round(camber_ratio, 3),
        "chord_coverage": None if chord_coverage is None else round(chord_coverage, 2),
        "fails": fails,
        "warns": warns,
        "passed_hard": len(fails) == 0,
        "passed_strict": len(findings) == 0,
    }


def _euler_xyz_np(rx: float, ry: float, rz: float) -> np.ndarray:
    ax, ay, az = math.radians(rx), math.radians(ry), math.radians(rz)
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _eval_anim_channel(kfs: list[dict[str, Any]], t: float) -> np.ndarray:
    if not kfs:
        return np.zeros(3, dtype=np.float64)
    pts = sorted(kfs, key=lambda k: float(k["time"]))
    if t <= float(pts[0]["time"]):
        dp = pts[0]["data_points"][0]
        return np.array([float(dp["x"]), float(dp["y"]), float(dp["z"])], dtype=np.float64)
    if t >= float(pts[-1]["time"]):
        dp = pts[-1]["data_points"][0]
        return np.array([float(dp["x"]), float(dp["y"]), float(dp["z"])], dtype=np.float64)
    for i in range(len(pts) - 1):
        t0, t1 = float(pts[i]["time"]), float(pts[i + 1]["time"])
        if t0 <= t <= t1:
            alpha = (t - t0) / max(1e-9, t1 - t0)
            d0, d1 = pts[i]["data_points"][0], pts[i + 1]["data_points"][0]
            v0 = np.array([float(d0["x"]), float(d0["y"]), float(d0["z"])], dtype=np.float64)
            v1 = np.array([float(d1["x"]), float(d1["y"]), float(d1["z"])], dtype=np.float64)
            return v0 * (1.0 - alpha) + v1 * alpha
    return np.zeros(3, dtype=np.float64)


def verify_v25_four_foot_grounding(bb: dict[str, Any], *, rest_tol: float = 0.45) -> dict[str, Any]:
    """Exact Blockbench 5 4-foot (`FL`, `FR`, `HL`, `HR`) rest-pose & animation FK grounding gate (`v25`)."""
    el_by_uuid = {e["uuid"]: e for e in bb.get("elements", [])}
    grp_by_uuid = {g["uuid"]: g for g in bb.get("groups", [])}

    def _el_corners(el: dict[str, Any]) -> np.ndarray:
        f0 = np.array(el["from"], dtype=np.float64)
        f1 = np.array(el["to"], dtype=np.float64)
        corners = np.array(
            [[x, y, z] for x in (f0[0], f1[0]) for y in (f0[1], f1[1]) for z in (f0[2], f1[2])],
            dtype=np.float64,
        )
        rot = el.get("rotation")
        if rot and any(abs(float(r)) > 1e-6 for r in rot):
            org = np.array(el.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
            R = _euler_xyz_np(float(rot[0]), float(rot[1]), float(rot[2]))
            corners = (corners - org) @ R.T + org
        return corners

    def _classify_limb(bone_path: list[str]) -> str | None:
        low = "/" + "/".join(b.lower() for b in bone_path)
        if any(k in low for k in ("/arm_r", "/foreleg_r", "/front_leg_r")):
            return "FR"
        if any(k in low for k in ("/arm_l", "/foreleg_l", "/front_leg_l")):
            return "FL"
        if any(k in low for k in ("/leg_r", "/hindleg_r", "/hind_leg_r", "/back_leg_r")):
            return "HR"
        if any(k in low for k in ("/leg_l", "/hindleg_l", "/hind_leg_l", "/back_leg_l")):
            return "HL"
        if "/tail" in low:
            return "TAIL"
        return None

    def _walk(
        nodes: list[Any],
        parent_R: np.ndarray,
        parent_t: np.ndarray,
        anim_pos: dict[str, np.ndarray],
        anim_rot: dict[str, np.ndarray],
        limb_ymins: dict[str, list[float]],
        bone_path: list[str],
    ) -> None:
        for nd in nodes:
            if isinstance(nd, str):
                el = el_by_uuid.get(nd)
                if el is None:
                    continue
                wc = _el_corners(el) @ parent_R.T + parent_t
                bucket = _classify_limb(bone_path)
                if bucket is not None:
                    limb_ymins[bucket].append(float(wc[:, 1].min()))
            elif isinstance(nd, dict):
                guid = nd.get("uuid", "")
                g = grp_by_uuid.get(guid, nd)
                gname = str(g.get("name", ""))
                org = np.array(g.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
                rest_r = g.get("rotation", [0.0, 0.0, 0.0])
                dpos = anim_pos.get(guid, np.zeros(3, dtype=np.float64))
                drot = anim_rot.get(guid, np.zeros(3, dtype=np.float64))
                R_rest = _euler_xyz_np(float(rest_r[0]), float(rest_r[1]), float(rest_r[2]))
                R_anim = _euler_xyz_np(float(drot[0]), float(drot[1]), float(drot[2]))
                R_bone = R_rest @ R_anim
                new_R = parent_R @ R_bone
                new_t = parent_R @ (org + dpos - R_bone @ org) + parent_t
                _walk(nd.get("children", []), new_R, new_t, anim_pos, anim_rot, limb_ymins, bone_path + [gname])

    def _eval_pose(anim_pos: dict[str, np.ndarray], anim_rot: dict[str, np.ndarray]) -> dict[str, float]:
        ly: dict[str, list[float]] = {"FL": [], "FR": [], "HL": [], "HR": [], "TAIL": []}
        _walk(bb.get("outliner", []), np.eye(3), np.zeros(3), anim_pos, anim_rot, ly, [])
        return {k: round(min(v), 2) for k, v in ly.items() if v}

    rest_feet = _eval_pose({}, {})
    feet_only = [rest_feet[k] for k in ("FL", "FR", "HL", "HR") if k in rest_feet]
    rest_ok = all(abs(v) <= rest_tol for v in feet_only) if feet_only else True
    clip_summary: dict[str, Any] = {}
    for anim in bb.get("animations", []):
        cname = str(anim.get("name", "clip"))
        dur = float(anim.get("length", 1.0))
        animators = anim.get("animators", {})
        res: dict[str, list[float]] = {"FL": [], "FR": [], "HL": [], "HR": [], "TAIL": []}
        for i in range(13):
            t = dur * (i / 12.0)
            apos: dict[str, np.ndarray] = {}
            arot: dict[str, np.ndarray] = {}
            for uid, adata in animators.items():
                kfs = adata.get("keyframes", [])
                pk = [k for k in kfs if k.get("channel") == "position"]
                rk = [k for k in kfs if k.get("channel") == "rotation"]
                if pk:
                    apos[uid] = _eval_anim_channel(pk, t)
                if rk:
                    arot[uid] = _eval_anim_channel(rk, t)
            ym = _eval_pose(apos, arot)
            for k in res:
                if k in ym:
                    res[k].append(ym[k])
        clip_summary[cname] = {k: [round(min(v), 1), round(max(v), 1)] for k, v in res.items() if v}
    return {"rest_feet": rest_feet, "rest_ok": rest_ok, "clips": clip_summary}


def verify_v25_dsl_asset(
    dsl_text: str,
    *,
    tier: str = "auto",
    rest_tol: float = 0.45,
    allow_asymmetric_prop: bool = False,
) -> dict[str, Any]:
    """Compile Block-DSL v4.1 -> SyncCraft `audit_scene` -> `.bbmodel` -> `v25` `G1-G12` + 4-foot grounding."""
    ensure_synccraft_importable()
    from synccraft.audit.sync_audit import audit_scene
    from synccraft.backends.bbmodel5_exporter import export_bbmodel
    from synccraft.dsl.parser import parse_program
    from synccraft.kinematics.musculoskeletal import audit_behavior_clip
    from synccraft.kinematics.se3_chain import compile_geometry

    prog = parse_program(dsl_text)
    scene = compile_geometry(prog)
    aud = audit_scene(scene)
    bb = export_bbmodel(scene, name="v25_asset", program=prog, num_animation_frames=12)
    g_rep = run_v25_construct_check_bbmodel(bb, tier=tier, allow_asymmetric_prop=allow_asymmetric_prop)
    ground_rep = verify_v25_four_foot_grounding(bb, rest_tol=rest_tol)

    clip_ok = True
    clip_audits: dict[str, Any] = {}
    if prog.behavior and prog.behavior.clips:
        for cl in prog.behavior.clips:
            cr = audit_behavior_clip(prog, cl.name)
            clip_audits[cl.name] = {
                "passed": bool(cr["passed"]),
                "collisions": int(cr.get("collision_events", 0)),
            }
            if not cr["passed"]:
                clip_ok = False

    passed_all = bool(aud["passed"]) and bool(g_rep["passed_strict"]) and bool(ground_rep["rest_ok"]) and clip_ok
    return {
        "passed_all": passed_all,
        "audit_passed": bool(aud["passed"]),
        "audit_errors": list(aud.get("errors", []))[:8],
        "g1_g12": g_rep,
        "grounding": ground_rep,
        "clips_ok": clip_ok,
        "clip_audits": clip_audits,
        "program": prog,
        "scene": scene,
        "bbmodel": bb,
    }


def render_v25_locate_first_grid_and_targets(
    baked: dict[str, Any],
    *,
    az_deg: float = 34.0,
    el_deg: float = 16.0,
    fill_frac: float = 0.78,
    pan_uv: tuple[float, float] = (0.0, 0.0),
    panel_size: tuple[int, int] = (384, 384),
    draw_grid: bool = True,
    head_keywords: tuple[str, ...] = ("head", "skull", "cranium", "snout", "muzzle", "jaw", "beak", "brow", "maxilla", "mandible"),
    eye_keywords: tuple[str, ...] = ("eye", "brow", "orbit", "cranium", "skull"),
) -> dict[str, Any]:
    """Render a jittered camera view with optional `0.0-1.0` v25 Pass-1 grid and exact 2D `<box>`/`<point>` targets."""
    width, height = int(panel_size[0]), int(panel_size[1])
    atlas_rgba: np.ndarray = baked["atlas_rgba"]
    emissive_map: np.ndarray = baked["emissive_map"]
    tex_h, tex_w = atlas_rgba.shape[:2]

    quads: list[tuple[np.ndarray, np.ndarray, bool]] = []
    head_pts_list: list[np.ndarray] = []
    eye_pts_list: list[np.ndarray] = []
    for rec in baked["elements"]:
        ctr = rec["world_center"]
        rot = rec["world_rotation"]
        half_ext = 0.5 * rec["eff_size"]
        is_pl = bool(rec["is_plane"])
        p_norm = rec["plane_normal"]
        local_faces = _face_vertices_local(half_ext)
        bname = str(rec.get("bone_name", "")).lower()
        ename = str(rec.get("name", "")).lower()

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

        stacked = np.concatenate(el_verts, axis=0)
        if (any(k in bname or k in ename for k in head_keywords)) and not any(
            ex in ename for ex in ("horn", "ear", "vane", "sail")
        ):
            head_pts_list.append(stacked)
        if any(k in ename for k in ("eye", "orbit")):
            eye_pts_list.append(stacked)

    if not eye_pts_list:
        for rec in baked["elements"]:
            ename = str(rec.get("name", "")).lower()
            if any(k in ename for k in ("brow", "cranium", "skull")):
                ctr = rec["world_center"]
                rot = rec["world_rotation"]
                half_ext = 0.5 * rec["eff_size"]
                local_faces = _face_vertices_local(half_ext)
                eye_pts_list.append(local_faces["west"] @ rot.T + ctr[None, :])
                eye_pts_list.append(local_faces["east"] @ rot.T + ctr[None, :])

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
    # Apply subtle viewport pan in camera plane so object location genuinely varies across images!
    pan_scale = cam_dist * half_tan_y * 0.45
    target = target - right * (float(pan_uv[0]) * pan_scale) - up * (float(pan_uv[1]) * pan_scale)
    cam_pos = target + cam_dir * cam_dist

    def _project_pts(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rel = pts - cam_pos[None, :]
        cx = rel @ right
        cy = rel @ up
        cz = rel @ fwd
        cz_safe = np.maximum(cz, 0.1)
        sx = (0.5 + 0.5 * (cx / (cz_safe * half_tan_x))) * (width - 1)
        sy = (0.5 - 0.5 * (cy / (cz_safe * half_tan_y))) * (height - 1)
        return sx, sy, cz

    # Rasterize studio frame with the exact camera above
    yy_bg, xx_bg = np.mgrid[0:height, 0:width]
    ry_n = (yy_bg / max(1.0, height - 1.0)) - 0.46
    rx_n = (xx_bg / max(1.0, width - 1.0)) - 0.50
    rad = np.sqrt(rx_n * rx_n + ry_n * ry_n)
    bg = np.clip(1.0 - 0.52 * rad, 0.35, 1.0)
    color_srgb = np.stack([18.0 + 18.0 * bg, 22.0 + 21.0 * bg, 32.0 + 28.0 * bg], axis=-1)
    zbuf = np.full((height, width), 1e9, dtype=np.float64)
    key_light = np.array([0.46, 0.78, -0.42], dtype=np.float64)
    key_light /= np.linalg.norm(key_light)
    rim_light = np.array([-0.55, 0.48, 0.68], dtype=np.float64)
    rim_light /= np.linalg.norm(rim_light)

    for verts, uv_c, double_sided in quads:
        sx, sy, cz = _project_pts(verts)
        if np.any(cz <= 0.1):
            continue
        e1 = verts[1] - verts[0]
        e2 = verts[2] - verts[0]
        fn = np.cross(e1, e2)
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
        shade = 0.56 + 0.42 * diff + 0.14 * rim

        for tri_idx in ((0, 1, 2), (0, 2, 3)):
            i0, i1, i2 = tri_idx
            x0, y0, z0 = sx[i0], sy[i0], cz[i0]
            x1, y1, z1 = sx[i1], sy[i1], cz[i1]
            x2, y2, z2 = sx[i2], sy[i2], cz[i2]
            min_x = max(0, int(math.floor(min(x0, x1, x2))))
            max_x = min(width - 1, int(math.ceil(max(x0, x1, x2))))
            min_y = max(0, int(math.floor(min(y0, y1, y2))))
            max_y = min(height - 1, int(math.ceil(max(y0, y1, y2))))
            if max_x < min_x or max_y < min_y:
                continue
            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) < 1e-7:
                continue
            gx = np.arange(min_x, max_x + 1, dtype=np.float64)[None, :] + 0.5
            gy = np.arange(min_y, max_y + 1, dtype=np.float64)[:, None] + 0.5
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
            u_px = np.clip((w0 * uv_c[i0, 0] + w1 * uv_c[i1, 0] + w2 * uv_c[i2, 0]).astype(np.int32), 0, tex_w - 1)
            v_px = np.clip((w0 * uv_c[i0, 1] + w1 * uv_c[i1, 1] + w2 * uv_c[i2, 1]).astype(np.int32), 0, tex_h - 1)
            rgba_samp = atlas_rgba[v_px, u_px]
            vis_alpha = vis & (rgba_samp[..., 3] >= 128)
            if not np.any(vis_alpha):
                continue
            em_samp = emissive_map[v_px, u_px]
            eff_shade = np.where(em_samp > 0.5, 1.08, shade)[..., None]
            rgb_lit = np.clip(rgba_samp[..., :3].astype(np.float64) * eff_shade, 0.0, 255.0)
            sub_z[vis_alpha] = z_interp[vis_alpha]
            sub_col = color_srgb[min_y : max_y + 1, min_x : max_x + 1]
            sub_col[vis_alpha] = rgb_lit[vis_alpha]

    img = Image.fromarray(np.clip(color_srgb, 0.0, 255.0).astype(np.uint8), mode="RGB")

    # Compute exact normalized [0, 1000] / [0.0, 1.0] head bbox & closest-to-camera eye point
    if head_pts_list:
        hp = np.concatenate(head_pts_list, axis=0)
        hsx, hsy, _ = _project_pts(hp)
        hx1 = float(np.clip(hsx.min() / max(1, width - 1), 0.0, 1.0))
        hy1 = float(np.clip(hsy.min() / max(1, height - 1), 0.0, 1.0))
        hx2 = float(np.clip(hsx.max() / max(1, width - 1), 0.0, 1.0))
        hy2 = float(np.clip(hsy.max() / max(1, height - 1), 0.0, 1.0))
    else:
        asx, asy, _ = _project_pts(all_pts)
        hx1 = float(np.clip(asx.min() / max(1, width - 1), 0.0, 1.0))
        hy1 = float(np.clip(asy.min() / max(1, height - 1), 0.0, 1.0))
        hx2 = float(np.clip(asx.max() / max(1, width - 1), 0.0, 1.0))
        hy2 = float(np.clip(asy.max() / max(1, height - 1), 0.0, 1.0))

    if eye_pts_list:
        # Pick the eye cluster closest to camera
        best_e = min(eye_pts_list, key=lambda pts: float(np.linalg.norm(pts.mean(axis=0) - cam_pos)))
        esx, esy, _ = _project_pts(best_e.mean(axis=0, keepdims=True))
        ex = float(np.clip(esx[0] / max(1, width - 1), 0.0, 1.0))
        ey = float(np.clip(esy[0] / max(1, height - 1), 0.0, 1.0))
    else:
        ex, ey = 0.5 * (hx1 + hx2), 0.5 * (hy1 + hy2)

    if draw_grid:
        drw = ImageDraw.Draw(img, "RGBA")
        for step in range(1, 10):
            gx = int(round(step * 0.1 * width))
            gy = int(round(step * 0.1 * height))
            drw.line([(gx, 0), (gx, height)], fill=(180, 210, 255, 72), width=1)
            drw.line([(0, gy), (width, gy)], fill=(180, 210, 255, 72), width=1)
            drw.text((gx + 2, 2), f".{step}", fill=(210, 230, 255, 190))
            drw.text((2, gy + 2), f".{step}", fill=(210, 230, 255, 190))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {
        "png_bytes": buf.getvalue(),
        "image": img,
        "head_box_norm": [round(hx1, 3), round(hy1, 3), round(hx2, 3), round(hy2, 3)],
        "head_box_1000": [int(round(hx1 * 1000)), int(round(hy1 * 1000)), int(round(hx2 * 1000)), int(round(hy2 * 1000))],
        "eye_point_norm": [round(ex, 3), round(ey, 3)],
        "eye_point_1000": [int(round(ex * 1000)), int(round(ey * 1000))],
    }
