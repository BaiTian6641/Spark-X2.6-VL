#!/usr/bin/env python3
"""Pin SigLIP 2 FixRes and extract its vision tower without loading the text tower.

Run using the existing Studio runtime and Spark's Transformers compatibility overlay:
  source scripts/env.sh
  PYTHONPATH="$BENCH_ROOT/runtime/transformers457${PYTHONPATH:+:$PYTHONPATH}" \
    "$BENCH_PYTHON" scripts/prepare_vision_component.py

This preparation is CPU-only. Its constant-input forward is a shape/finite-value
diagnostic, not a visual-understanding or training-quality evaluation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "google/siglip2-base-patch16-256"
REVISION = "3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab"
UPSTREAM_FILES = ("README.md", "config.json", "preprocessor_config.json", "model.safetensors")
EXPECTED_CHECKPOINT_SHA256 = "6125cacc01fa93bdc98a0c5101cefcd69b2ed1f8ab4f38d86f4ad5984f5dc863"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def file_record(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-files-only", action="store_true", help="Reuse an already downloaded pinned source.")
    parser.add_argument("--output", type=Path, default=ROOT / "models/SigLIP2-base-256")
    parser.add_argument("--manifest", type=Path, default=ROOT / "evidence/multimodal-s0-vision-source.json")
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / "runtime/hf"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    # No CUDA availability/device operation is needed for source preparation.
    import torch
    import transformers
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from safetensors import safe_open
    from transformers import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel

    torch.set_num_threads(8)
    source_dir = args.output / "upstream"
    vision_dir = args.output / "vision-only"
    source_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)
    for filename in UPSTREAM_FILES:
        hf_hub_download(
            repo_id=REPOSITORY,
            revision=REVISION,
            filename=filename,
            local_dir=source_dir,
            local_files_only=args.local_files_only,
        )

    checkpoint_path = source_dir / "model.safetensors"
    if digest(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Pinned checkpoint SHA256 does not match Hugging Face's LFS object identity")
    source_config = json.loads((source_dir / "config.json").read_text())
    if source_config["model_type"] != "siglip":
        raise RuntimeError("This preparation supports the pinned SigLIP 2 FixRes checkpoint only")
    vision_config = SiglipVisionConfig(**source_config["vision_config"], vision_use_head=False)
    model = SiglipVisionModel(vision_config)
    expected_keys = set(model.state_dict())
    state_dict = {}
    all_source_count = 0
    excluded_pooling_tensors = []
    excluded_pooling_parameter_count = 0
    with safe_open(checkpoint_path, framework="pt", device="cpu") as source:
        for key in source.keys():
            all_source_count += math.prod(source.get_slice(key).get_shape())
            if key.startswith("vision_model.head."):
                excluded_pooling_tensors.append(key)
                excluded_pooling_parameter_count += math.prod(source.get_slice(key).get_shape())
            elif key.startswith("vision_model."):
                # SiglipVisionModel itself owns vision_model.*, so preserve this
                # prefix. Never use loose strict=False to silently omit tensors.
                state_dict[key] = source.get_tensor(key)
    if set(state_dict) != expected_keys:
        raise RuntimeError({"missing": sorted(expected_keys - set(state_dict)), "unexpected": sorted(set(state_dict) - expected_keys)})
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.save_pretrained(vision_dir, safe_serialization=True)
    shutil.copyfile(source_dir / "preprocessor_config.json", vision_dir / "preprocessor_config.json")
    processor = SiglipImageProcessor.from_pretrained(vision_dir, local_files_only=True)
    diagnostic_pixels = processor(images=Image.new("RGB", (320, 180), (127, 127, 127)), return_tensors="pt")["pixel_values"]
    with torch.inference_mode():
        output = model(pixel_values=diagnostic_pixels)
    if not bool(torch.isfinite(output.last_hidden_state).all()):
        raise RuntimeError("CPU diagnostic produced non-finite vision features")
    # Reload the actual extracted artifact to verify its standalone loading path.
    reloaded = SiglipVisionModel.from_pretrained(vision_dir, local_files_only=True)
    reloaded.eval()
    with torch.inference_mode():
        reloaded_output = reloaded(pixel_values=diagnostic_pixels)
    max_reload_difference = float((output.last_hidden_state - reloaded_output.last_hidden_state).abs().max())
    if max_reload_difference != 0:
        raise RuntimeError(f"Extracted/reloaded CPU output differs by {max_reload_difference}")

    manifest = {
        "schema_version": 1,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Pinned semantic vision tower preparation for Spark multimodal S0; CPU structural diagnostic only",
        "source": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "model_card_url": f"https://huggingface.co/{REPOSITORY}/blob/{REVISION}/README.md",
            "license_metadata": "apache-2.0",
            "transformers_documentation_url": "https://huggingface.co/docs/transformers/v4.57.1/model_doc/siglip2",
            "fixres_implementation_url": "https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/siglip/modeling_siglip.py",
            "downloaded_files": [{**file_record(source_dir / filename), "url": f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{filename}"} for filename in UPSTREAM_FILES],
            "full_checkpoint_parameter_count": all_source_count,
            "full_checkpoint_sha256_verified_against_lfs": EXPECTED_CHECKPOINT_SHA256,
        },
        "derived_artifact": {
            "directory": str(vision_dir.relative_to(ROOT)),
            "class": "transformers.SiglipVisionModel",
            "processor_class": "transformers.SiglipImageProcessor",
            "model_type": vision_config.model_type,
            "extraction": "Set vision_use_head=False; keep exactly vision_model.* except vision_model.head.*, retaining prefix; strict state-dict equality and load; save_pretrained safetensors",
            "excluded": ["text_model.*", "logit_scale", "logit_bias", "vision_model.head.*"],
            "excluded_pooling_parameter_count": excluded_pooling_parameter_count,
            "excluded_pooling_tensor_keys": excluded_pooling_tensors,
            "files": [file_record(vision_dir / filename) for filename in ("config.json", "preprocessor_config.json", "model.safetensors")],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "vision_configuration": vision_config.to_dict(),
            "input_pixel_shape": list(diagnostic_pixels.shape),
            "dense_patch_output_shape": list(output.last_hidden_state.shape),
            "pooled_output_shape": None if output.pooler_output is None else list(output.pooler_output.shape),
            "patch_grid": [vision_config.image_size // vision_config.patch_size] * 2,
            "has_cls_token": False,
            "output_for_spark_projector": "last_hidden_state: batch x 256 x 768; retain dense features, do not use only pooler_output",
            "resize_geometry": "Processor resizes to 256x256 without keeping original aspect ratio; any point/box labels must follow this warp and preserve original dimensions for inverse mapping",
        },
        "validation": {
            "device": "cpu",
            "torch_dtype": str(next(model.parameters()).dtype),
            "input": "Constant gray image used only for shape and finite-value diagnostic",
            "all_output_features_finite": True,
            "standalone_reload_max_abs_difference": max_reload_difference,
            "quality_or_training_evidence": False,
        },
        "runtime": {"python": sys.version, "executable": sys.executable, "torch": torch.__version__, "transformers": transformers.__version__, "transformers_path": transformers.__file__},
        "preparation_script": file_record(Path(__file__)),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(args.manifest), "vision_directory": str(vision_dir), "parameters": manifest["derived_artifact"]["parameter_count"], "dense_shape": manifest["derived_artifact"]["dense_patch_output_shape"]}))


if __name__ == "__main__":
    main()
