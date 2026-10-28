"""Small, source-bound multimodal records and CPU-only validity checks.

Only ``prompt``, ``frames`` and timestamps belong in the model input.  Labels,
source paths and diagnostic overlays deliberately stay outside that interface.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image


REQUIRED = {
    "id", "split", "source_group", "language", "task", "prompt", "answer",
    "frames", "media_times_s", "metadata",
}


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_records(root: str | Path, split: str | None = None) -> list[dict[str, Any]]:
    root = Path(root)
    splits = [split] if split else ["train", "dev", "test"]
    if any(s not in {"train", "dev", "test"} for s in splits):
        raise ValueError("split must be train, dev or test")
    records = []
    for name in splits:
        with (root / f"{name}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get('split') != name:
                    raise ValueError(f"Record split disagrees with source file {name}.jsonl")
                records.append(record)
    return records


def resolve_media(root: str | Path, relative: str) -> Path:
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or Path(relative).is_absolute():
        raise ValueError(f"Media path must remain inside dataset root: {relative}")
    return path


def load_frames(record: dict[str, Any], root: str | Path) -> list[Image.Image]:
    frames = []
    for name in record["frames"]:
        with Image.open(resolve_media(root, name)) as image:
            frames.append(image.convert("RGB"))
    return frames


def model_input(record: dict[str, Any]) -> dict[str, Any]:
    """Return an explicit allowlist; never pass the whole record to a trainer."""
    result = {key: record[key] for key in ("prompt", "frames", "media_times_s")}
    if "capture_times_s" in record:
        result["capture_times_s"] = record["capture_times_s"]
    return result


def validate_dataset(root: str | Path, splits: tuple[str, ...] | list[str] | None = None) -> dict[str, Any]:
    """Validate file integrity, split isolation and rendered annotation geometry.

    This validates data wiring and controlled render targets, not model quality.
    """
    root = Path(root)
    selected_splits = tuple(splits) if splits is not None else ('train', 'dev', 'test')
    if not selected_splits or len(set(selected_splits)) != len(selected_splits):
        raise ValueError('Validation splits must be nonempty and distinct')
    records = [record for split in selected_splits for record in load_records(root, split)]
    errors = []
    groups: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, set[str]] = defaultdict(set)
    ids = set()
    by_split = Counter()
    by_task = Counter()
    by_language = Counter()
    source_counts = Counter()
    image_sizes = {}
    verified_media = {}
    expected_files = json.loads((root / "media_manifest.json").read_text())
    manifest = {item["path"]: item for item in expected_files}
    if len(manifest) != len(expected_files):
        errors.append('duplicate paths in media manifest')
    dataset_manifest = json.loads((root / "dataset_manifest.json").read_text())
    for name, digest in dataset_manifest["record_files"].items():
        if sha256_file(root / name) != digest:
            errors.append(f"record-file hash mismatch: {name}")
    if sha256_file(root / "media_manifest.json") != dataset_manifest["media_manifest_sha256"]:
        errors.append("media-manifest hash mismatch")
    for record in records:
        rid = record.get("id", "<missing>")
        if REQUIRED - record.keys():
            errors.append(f"{rid}: missing {sorted(REQUIRED - record.keys())}")
            continue
        if rid in ids:
            errors.append(f"duplicate record id {rid}")
        ids.add(rid)
        if not all(isinstance(record[k], str) and record[k].strip() for k in ('prompt', 'answer', 'source_group', 'task')):
            errors.append(f'{rid}: prompt, answer, source group and task must be nonempty strings')
        if record["split"] not in {"train", "dev", "test"}:
            errors.append(f"{rid}: invalid split")
        if record["language"] not in {"en", "zh"}:
            errors.append(f"{rid}: invalid language")
        by_split[record["split"]] += 1
        by_task[record["task"]] += 1
        by_language[record["language"]] += 1
        source_counts["synthetic" if record["metadata"]["synthetic"] else "real"] += 1
        groups[record["source_group"]].add(record["split"])
        n = len(record["frames"])
        text_only = record['task'] == 'text_replay'
        if (n == 0 and not text_only) or n != len(record["media_times_s"]):
            errors.append(f"{rid}: frame/time count mismatch")
        if text_only and (n != 0 or record.get('capture_times_s') is not None or record['metadata']['synthetic']):
            errors.append(f'{rid}: text replay requires genuine text, no media and no capture clock')
        times = record["media_times_s"]
        numeric_times = all(type(t) in {int, float} and math.isfinite(t) for t in times)
        if not numeric_times or any(b <= a for a, b in zip(times, times[1:])):
            errors.append(f"{rid}: media times must increase")
        capture = record.get('capture_times_s')
        if capture is not None:
            if len(capture) != n:
                errors.append(f"{rid}: capture time count mismatch")
            if any(t is not None and (type(t) not in {int, float} or not math.isfinite(t)) for t in capture):
                errors.append(f'{rid}: capture timestamps must be finite numbers or unknown')
        if "coordinate_convention" not in record["metadata"] or "license" not in record["metadata"]:
            errors.append(f"{rid}: missing coordinate/license provenance")
        for relative in record["frames"]:
            try:
                path = resolve_media(root, relative)
                if relative not in verified_media:
                    digest = sha256_file(path)
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        image_sizes[relative] = image.size
                    verified_media[relative] = digest
                digest = verified_media[relative]
                hashes[digest].add(record["split"])
                if relative not in manifest or manifest[relative]["sha256"] != digest:
                    errors.append(f"{rid}: media hash mismatch: {relative}")
            except Exception as exc:
                errors.append(f"{rid}: invalid media {relative}: {exc}")
        if record["metadata"]["synthetic"]:
            if any(relative not in image_sizes for relative in record["frames"]):
                continue
            truth = record["metadata"]["ground_truth"]
            if len(truth["frames"]) != n:
                errors.append(f"{rid}: rendered target/frame mismatch")
            for relative, target in zip(record["frames"], truth["frames"]):
                w, h = image_sizes[relative]
                x0, y0, x1, y1 = target["box_xyxy_pixels"]
                if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                    errors.append(f"{rid}: box outside image")
                cx, cy = target["point_xy_pixels"]
                if (cx, cy) != ((x0 + x1) / 2, (y0 + y1) / 2):
                    errors.append(f"{rid}: point/box center disagreement")
            json_tasks = {'point_box', 'occlusion', 'temporal_order', 'capture_elapsed', 'elapsed_capture_time'}
            answer = json.loads(record["answer"]) if record['task'] in json_tasks else None
            if record["task"] == "point_box":
                box = truth["frames"][-1]["box_xyxy_pixels"]
                w, h = image_sizes[record["frames"][-1]]
                expected_box = [round(v / scale, 4) for v, scale in zip(box, (w, h, w, h))]
                if answer["box"] != expected_box:
                    errors.append(f"{rid}: normalized target does not roundtrip")
                point = truth['frames'][-1]['point_xy_pixels']
                expected_point = [round(point[0] / w, 4), round(point[1] / h, 4)]
                if answer.get('point') != expected_point:
                    errors.append(f'{rid}: normalized point target does not roundtrip')
            if record["task"] == "occlusion":
                expected = [i for i, frame in enumerate(truth["frames"]) if not frame["visible"]]
                if answer["fully_occluded_frames"] != expected:
                    errors.append(f"{rid}: occlusion target mismatch")
            if record["task"] == "temporal_order" and answer["direction"] != truth["direction"]:
                errors.append(f"{rid}: temporal direction mismatch")
            if record['task'] in {'capture_elapsed', 'elapsed_capture_time'}:
                capture = record.get('capture_times_s')
                known = capture is not None and len(capture) == n and all(
                    type(t) in {int, float} and math.isfinite(t) for t in capture)
                expected = abs(capture[-1] - capture[0]) if known else None
                actual = answer.get('elapsed_seconds')
                if expected is None:
                    if actual is not None: errors.append(f'{rid}: unknown capture clock requires null target')
                elif type(actual) not in {int, float} or not math.isfinite(actual) or not math.isclose(actual, expected, abs_tol=1e-6):
                    errors.append(f'{rid}: elapsed capture target differs from supplied capture clock')
        if set(model_input(record)) & {"answer", "metadata", "source_group", "task", "id"}:
            errors.append(f"{rid}: target/provenance leaked into model interface")
    for group, splits in groups.items():
        if len(splits) > 1:
            errors.append(f"source group overlaps splits: {group}")
    for digest, splits in hashes.items():
        if len(splits) > 1:
            errors.append(f"identical image bytes overlap splits: {digest}")
    return {
        "valid": not errors,
        "validated_splits": list(selected_splits),
        "all_record_file_hashes_checked": True,
        "errors": errors,
        "record_count": len(records),
        "unique_source_groups": len(groups),
        "unique_media_files": len(manifest),
        "by_split": dict(by_split),
        "by_task": dict(by_task),
        "by_language": dict(by_language),
        "by_source_kind": dict(source_counts),
        "evidence_scope": "Data integrity and controlled rendering checks only; no model evaluation.",
    }
