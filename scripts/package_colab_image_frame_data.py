#!/usr/bin/env python3
"""Build a self-contained image/frame dataset TAR for Colab.

The training split and optional development split are copied without rewriting
record JSON. Media manifest entries are filtered to the copied files and the
dataset manifest is rebound to the new record/media hashes. Nothing is extracted
from the source dataset, and no source cache or sealed test split is included by
default.

Run on the workstation:
  python scripts/package_colab_image_frame_data.py \
    --dataset data/multimodal_formal_v1 \
    --output /tmp/multimodal_formal_v1_colab.tar
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multimodal.data import load_records, sha256_file, validate_dataset  # noqa: E402


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build(dataset: Path, output: Path, splits: tuple[str, ...]) -> dict:
    dataset = dataset.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("Splits must be non-empty and distinct")
    validation = validate_dataset(dataset, splits=splits)
    if not validation["valid"]:
        raise ValueError("Source dataset validation failed: " + "; ".join(validation["errors"][:5]))

    records = {split: load_records(dataset, split=split) for split in splits}
    media_paths = sorted({name for rows in records.values() for row in rows for name in row["frames"]})
    source_media = json.loads((dataset / "media_manifest.json").read_text())
    by_path = {item["path"]: item for item in source_media}
    missing = [path for path in media_paths if path not in by_path]
    if missing:
        raise ValueError(f"Media manifest lacks {len(missing)} referenced files, e.g. {missing[:3]}")

    source_manifest = json.loads((dataset / "dataset_manifest.json").read_text())
    by_task = Counter(row["task"] for rows in records.values() for row in rows)
    by_language = Counter(row["language"] for rows in records.values() for row in rows)
    by_split = Counter({split: len(rows) for split, rows in records.items()})

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spark-colab-package-", dir=output.parent) as temporary:
        staging = Path(temporary)
        record_hashes = {}
        for split in splits:
            source = dataset / f"{split}.jsonl"
            target = staging / source.name
            shutil.copyfile(source, target)
            if sha256_file(target) != sha256_file(source):
                raise ValueError(f"Record copy changed bytes: {source.name}")
            record_hashes[source.name] = sha256_file(target)

        for relative in media_paths:
            source = dataset / relative
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            expected = by_path[relative]
            if target.stat().st_size != expected["bytes"] or sha256_file(target) != expected["sha256"]:
                raise ValueError(f"Media copy changed bytes: {relative}")

        media_manifest = [by_path[path] for path in media_paths]
        media_manifest_path = staging / "media_manifest.json"
        write_json(media_manifest_path, media_manifest)

        manifest = dict(source_manifest)
        manifest.update({
            "record_count": sum(by_split.values()),
            "by_split": dict(by_split),
            "by_task": dict(by_task),
            "by_language": dict(by_language),
            "record_files": record_hashes,
            "media_manifest_sha256": sha256_file(media_manifest_path),
            "colab_package": {
                "schema_version": 1,
                "splits": list(splits),
                "sealed_test_included": "test" in splits,
                "source_dataset": str(dataset),
                "source_dataset_manifest_sha256": sha256_file(dataset / "dataset_manifest.json"),
                "media_files": len(media_manifest),
                "media_bytes": sum(item["bytes"] for item in media_manifest),
                "record_count": sum(by_split.values()),
            },
        })
        write_json(staging / "dataset_manifest.json", manifest)
        for name in ("DATA_CARD.md", "README.md"):
            source = dataset / name
            if source.exists():
                shutil.copyfile(source, staging / name)

        # Validate the package exactly as the Colab trainer will consume it.
        package_validation = validate_dataset(staging, splits=splits)
        if not package_validation["valid"]:
            raise ValueError("Packaged dataset validation failed: " + "; ".join(package_validation["errors"][:5]))
        write_json(staging / "validation.json", package_validation)

        with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.add(path, arcname=path.relative_to(staging).as_posix(), recursive=False)
        receipt = {
            "schema_version": 1,
            "archive": str(output),
            "bytes": output.stat().st_size,
            "sha256": digest(output),
            "splits": list(splits),
            "records": dict(by_split),
            "tasks": dict(by_task),
            "languages": dict(by_language),
            "media_files": len(media_manifest),
            "media_bytes": sum(item["bytes"] for item in media_manifest),
            "validation": package_validation,
        }
        write_json(output.with_suffix(output.suffix + ".json"), receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev"], choices=["train", "dev", "test"])
    args = parser.parse_args()
    receipt = build(args.dataset, args.output, tuple(args.splits))
    print(json.dumps({key: receipt[key] for key in ("archive", "bytes", "sha256", "records", "media_files")}, indent=2))


if __name__ == "__main__":
    main()
