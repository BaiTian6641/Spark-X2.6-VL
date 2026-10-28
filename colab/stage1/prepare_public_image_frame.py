#!/usr/bin/env python3
"""Build the formal-v1-style image/frame corpus from public Hugging Face TARs.

This is deliberately one-shot: it downloads the pinned DOCCI caption source and
the COCO/COCO-CN/controlled-geometry vision core, verifies every TAR member, and
writes a self-contained dataset that the reviewed image/frame trainer can read.

It does not download Open Images, video, audio or text replay, and it does not
pretend that a new train/dev partition is the frozen formal-v1 evaluation split.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from multimodal.data import validate_dataset  # noqa: E402

REPO = "TassP0lar/SparkX2.6-MM"
REVISION = "f472f980a6c95d3e9844f27e6045c8c09b21536a"
COMPONENTS = {
    "docci": "releases/docci-training-source-v1",
    "vision-core": "releases/full-ft-v1/vision-core",
}
DEV_PERCENT = 2.0
FIELDS = (
    "id", "split", "source_group", "source_id", "language", "task", "prompt",
    "answer", "frames", "media_times_s", "capture_times_s", "metadata",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def url(prefix: str, name: str) -> str:
    return f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{prefix}/{name}"


def download(prefix: str, name: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        return destination
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url(prefix, name), headers={"User-Agent": "Spark-Colab-one-shot/1"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            stream.write(block)
    if not partial.stat().st_size:
        raise RuntimeError(f"Empty download: {url(prefix, name)}")
    os.replace(partial, destination)
    return destination


def local_or_download(component: str, prefix: str, name: str, cache: Path) -> Path:
    local = cache / "releases" / component / name
    if local.exists():
        return local
    return download(prefix, name, local)


def resolve_release(component: str, prefix: str, cache: Path) -> dict:
    release_path = local_or_download(component, prefix, "RELEASE.json", cache)
    release = json.loads(release_path.read_text())
    if release.get("kind") != "spark_webdataset_release":
        raise ValueError(f"Unexpected release kind for {component}")
    if release.get("role") != "training":
        raise ValueError(f"Component {component} is not a training release")
    return release


def verify_members(component: str, release: dict, cache: Path) -> dict[str, dict]:
    """Return member path -> metadata, verifying every TAR byte before use."""
    members: dict[str, dict] = {}
    for shard in release["shards"]:
        name = shard.get("name") or shard.get("path") or shard.get("shard")
        if not name:
            # Release files use a synthetic member list without a shard path
            # only in older fixtures. The actual public release has paths.
            raise ValueError(f"Release shard lacks a path: {component}")
        path = Path(name)
        tar_path = cache / "releases" / component / path.name
        if not tar_path.exists():
            tar_path = local_or_download(component, COMPONENTS[component], path.name, cache)
        expected_bytes = shard.get("bytes")
        if expected_bytes is not None and tar_path.stat().st_size != expected_bytes:
            raise ValueError(f"TAR size mismatch: {component}/{path.name}")
        expected_sha = shard.get("sha256")
        if expected_sha and sha256_file(tar_path) != expected_sha:
            raise ValueError(f"TAR SHA256 mismatch: {component}/{path.name}")
        for item in shard["files"]:
            members[item["path"]] = {**item, "tar": tar_path, "shard": path.name}
    return members


def read_component(component: str, members: dict[str, dict], cache: Path,
                   output: Path) -> list[dict]:
    """Extract verified media, read JSON records and return flattened variants."""
    variants: list[dict] = []
    seen_ids: set[str] = set()
    by_tar: dict[Path, dict[str, dict]] = defaultdict(dict)
    for member_path, record in members.items():
        by_tar[record["tar"]][member_path] = record
    for tar_path, expected in sorted(by_tar.items()):
        with tarfile.open(tar_path) as archive:
            for member in archive:
                if member.isdir():
                    continue
                record = expected.get(member.name)
                if record is None:
                    raise ValueError(f"Undeclared TAR member {tar_path.name}/{member.name}")
                if not member.isfile() or member.issym() or member.islnk():
                    raise ValueError(f"Unsupported TAR member: {member.name}")
                payload = archive.extractfile(member).read()
                if len(payload) != record["bytes"] or sha256_bytes(payload) != record["sha256"]:
                    raise ValueError(f"Member hash mismatch: {component}/{member.name}")
                if member.name.endswith(".json"):
                    data = json.loads(payload)
                    base = {key: data[key] for key in FIELDS if key in data}
                    rows = data.get("task_variants") or [data]
                    if not rows:
                        raise ValueError(f"Empty task variants: {member.name}")
                    for row in rows:
                        merged = {**base, **row}
                        if not merged.get("frames") or not merged.get("prompt") or not merged.get("answer"):
                            raise ValueError(f"Incomplete training record: {member.name}")
                        merged["frames"] = [str(name) for name in merged["frames"]]
                        if not merged.get("source_group") or not merged.get("task") or not merged.get("language"):
                            raise ValueError(f"Record lacks task/group/language: {member.name}")
                        identity = str(merged.get("id") or f"{component}:{len(variants)}")
                        if identity in seen_ids:
                            identity = f"{identity}:{len(variants)}"
                        seen_ids.add(identity)
                        merged["id"] = identity
                        # The published unknown-clock variants keep the
                        # renderer's timestamps but teach a null answer. The
                        # reviewed collator treats any timestamp array as a
                        # known clock, so remove it for unknown targets.
                        if merged.get("task") in {"capture_elapsed", "elapsed_capture_time"}:
                            try:
                                elapsed = json.loads(merged["answer"]).get("elapsed_seconds")
                            except (KeyError, TypeError, ValueError):
                                elapsed = None
                            if elapsed is None:
                                merged.pop("capture_times_s", None)
                        merged["_component"] = component
                        variants.append(merged)
                    continue
                target = output / "media" / component / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_bytes(payload)
    return variants


def choose_splits(records: list[dict]) -> dict[str, str]:
    groups: dict[str, list[int]] = defaultdict(list)
    task_language: dict[tuple[str, str], set[str]] = defaultdict(set)
    for index, record in enumerate(records):
        group = record["source_group"]
        groups[group].append(index)
        task_language[(record["task"], record["language"])].add(group)
    split_by_group: dict[str, str] = {}
    for group in sorted(groups):
        bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10_000
        split_by_group[group] = "dev" if bucket < int(DEV_PERCENT * 100) else "train"
    # A task/language with no development group cannot produce a diagnostic.
    for key, group_set in task_language.items():
        if not any(split_by_group[group] == "dev" for group in group_set):
            split_by_group[sorted(group_set)[0]] = "dev"
    return split_by_group


def build(cache: Path, output: Path) -> dict:
    cache, output = cache.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []
    source_receipt = {"repo": REPO, "revision": REVISION, "components": {}}
    for component, prefix in COMPONENTS.items():
        release = resolve_release(component, prefix, cache)
        members = verify_members(component, release, cache)
        source_receipt["components"][component] = {
            "prefix": prefix,
            "release_sha256": sha256_file(cache / "releases" / component / "RELEASE.json"),
            "payload_bytes": release["payload_bytes"],
            "record_count": release["record_count"],
        }
        rows = read_component(component, members, cache, output)
        print(json.dumps({"event": "component_read", "component": component, "variants": len(rows)}), flush=True)
        all_records.extend(rows)
    split_by_group = choose_splits(all_records)
    for record in all_records:
        record["split"] = split_by_group[record["source_group"]]
        record["frames"] = [f"media/{record['_component']}/{name}" for name in record["frames"]]
        record.pop("_component", None)
    by_split = Counter(record["split"] for record in all_records)
    by_task = Counter(record["task"] for record in all_records)
    by_language = Counter(record["language"] for record in all_records)
    for split in ("train", "dev"):
        rows = sorted((record for record in all_records if record["split"] == split), key=lambda row: row["id"])
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in rows), encoding="utf-8"
        )
    media_manifest = []
    for path in sorted((output / "media").rglob("*")):
        if not path.is_file():
            continue
        from PIL import Image
        with Image.open(path) as image:
            width, height = image.size
        media_manifest.append({
            "path": path.relative_to(output).as_posix(),
            "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "width": width, "height": height,
        })
    write_json(output / "media_manifest.json", media_manifest)
    record_files = {f"{split}.jsonl": sha256_file(output / f"{split}.jsonl") for split in ("train", "dev")}
    source_path = output / "source_manifest.json"
    write_json(source_path, source_receipt)
    manifest = {
        "schema_version": 1,
        "dataset_version": "spark-public-image-frame-v1",
        "record_count": len(all_records),
        "by_split": dict(by_split),
        "by_task": dict(by_task),
        "by_language": dict(by_language),
        "record_files": record_files,
        "media_manifest_sha256": sha256_file(output / "media_manifest.json"),
        "source_manifest_sha256": sha256_file(source_path),
        "source": source_receipt,
        "split_policy": f"source-group hash, {DEV_PERCENT}% development; no sealed test split",
        "limitations": [
            "Public DOCCI and vision-core components only; not the frozen formal-v1 79,706-record corpus.",
            "No official or sealed test split; the development split is a deterministic source-group holdout.",
            "Human-source and pretraining overlap limits remain those of the published components.",
        ],
    }
    write_json(output / "dataset_manifest.json", manifest)
    (output / "DATA_CARD.md").write_text(
        "# Public image/frame training subset\n\n"
        f"Built from `{REPO}` revision `{REVISION}` components "
        "`docci` and `vision-core`. No test split is included. The development split "
        "is a deterministic source-group holdout, not the frozen formal-v1 evaluation split.\n"
    )
    validation = validate_dataset(output, splits=("train", "dev"))
    write_json(output / "validation.json", validation)
    if not validation["valid"]:
        raise ValueError("Built dataset validation failed: " + "; ".join(validation["errors"][:5]))
    receipt = {
        "schema_version": 1, "output": str(output), "records": len(all_records),
        "by_split": dict(by_split), "by_task": dict(by_task), "by_language": dict(by_language),
        "media_files": len(media_manifest), "media_bytes": sum(item["bytes"] for item in media_manifest),
        "dataset_manifest_sha256": sha256_file(output / "dataset_manifest.json"),
        "validation": validation,
    }
    write_json(output / "BUILD.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/content/spark-mm/public-cache"))
    parser.add_argument("--output", type=Path, default=Path("/content/spark-mm/datasets/public_image_frame"))
    args = parser.parse_args()
    receipt = build(args.cache, args.output)
    print(json.dumps({key: receipt[key] for key in ("records", "media_files", "media_bytes", "by_split")}, indent=2))


if __name__ == "__main__":
    main()
