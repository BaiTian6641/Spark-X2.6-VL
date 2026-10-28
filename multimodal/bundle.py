"""Load local, alias-aware Spark multimodal research checkpoints.

The saved full-wrapper tensors are the only weights read.  Meta construction
avoids allocating another randomly initialized 4B backbone, and assign=True
preserves the saved tensor dtypes, including Spark's FP32 normalization weights.
Checkpoint Python files are local research code and execute during loading.
Use the same tested Transformers runtime as training (currently 4.57.1).
"""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types

import torch

from .model import SparkMultimodalModel


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spark_classes(path: Path):
    """Import the checkpoint's custom Spark code without HF cache or network."""
    identity = hashlib.sha256((str(path.resolve()) + _sha256(path / "configuration_spark.py")
                               + _sha256(path / "modeling_spark.py")).encode()).hexdigest()[:16]
    package_name = f"_local_spark_bundle_{identity}"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(path)]
        package.__package__ = package_name
        sys.modules[package_name] = package
        try:
            for name in ("configuration_spark", "modeling_spark"):
                fullname = f"{package_name}.{name}"
                spec = importlib.util.spec_from_file_location(fullname, path / f"{name}.py")
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load local checkpoint implementation {name}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[fullname] = module
                spec.loader.exec_module(module)
        except Exception:
            for name in list(sys.modules):
                if name == package_name or name.startswith(package_name + "."):
                    del sys.modules[name]
            raise
    config_class = sys.modules[f"{package_name}.configuration_spark"].Spark2_5Config
    model_class = sys.modules[f"{package_name}.modeling_spark"].Spark2_5ForCausalLM
    return config_class, model_class


def _check_source(path: Path) -> dict:
    snapshot = path / "implementation" / "model.py"
    current = Path(__file__).with_name("model.py")
    sources = json.loads((path / "experiment_sources.json").read_text())
    recorded_hash = sources.get("source_hashes", {}).get("multimodal/model.py")
    if not recorded_hash:
        raise ValueError("Checkpoint lacks its recorded multimodal/model.py source hash")
    snapshot_hash = _sha256(snapshot)
    if snapshot_hash != recorded_hash:
        raise ValueError("Checkpoint wrapper snapshot does not match experiment_sources.json")
    if _sha256(current) != snapshot_hash:
        raise ValueError(
            "Current multimodal/model.py differs from the checkpoint implementation. "
            "Use the matching research implementation before loading this bundle."
        )
    return {
        "wrapper_sha256": snapshot_hash, "current_wrapper_matches_snapshot": True,
        "backbone_python_sha256": {name: _sha256(path / name)
                                   for name in ("configuration_spark.py", "modeling_spark.py")},
    }


def _expand_aliases(tensors: dict, metadata: dict, expected: set[str]) -> dict:
    unexpected = set(tensors) - expected
    if unexpected:
        raise ValueError(f"Unexpected checkpoint tensor keys: {sorted(unexpected)}")
    unknown_metadata = set(metadata) - expected - {"format"}
    if unknown_metadata:
        raise ValueError(f"Unknown checkpoint metadata/alias keys: {sorted(unknown_metadata)}")
    pending = expected - set(tensors)
    aliases = {}
    while pending:
        resolved = []
        for name in sorted(pending):
            target = metadata.get(name)
            if target in tensors:
                tensors[name] = tensors[target]
                aliases[name] = target
                resolved.append(name)
        if not resolved:
            raise ValueError(f"Missing checkpoint tensors or unresolved aliases: {sorted(pending)}")
        pending.difference_update(resolved)
    return aliases


def _materialize_nonpersistent_buffers(model) -> list[str]:
    repaired = []
    for name, buffer in list(model.named_buffers()):
        if not buffer.is_meta:
            continue
        if name != "vision_encoder.vision_model.embeddings.position_ids":
            raise ValueError(f"Unknown nonpersistent meta buffer cannot be reconstructed safely: {name}")
        embeddings = model.vision_encoder.vision_model.embeddings
        embeddings.position_ids = torch.arange(embeddings.num_positions, device="cpu", dtype=torch.long).expand(1, -1)
        repaired.append(name)
    return repaired


def load_bundle(path: str | Path, device: str | torch.device = "cpu"):
    """Return (model, tokenizer, image_processor) from a complete local bundle.

    This never calls from_pretrained for model weights, downloads files, casts
    all tensors to a single dtype or reads optimizer pickle files.  The wrapper
    implementation and architecture must match the saved experiment.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel

    path = Path(path).expanduser().resolve()
    target_device = torch.device(device)
    if target_device.type not in {"cpu", "cuda"}:
        raise ValueError("This research loader supports cpu or cuda (including ROCm devices)")
    required = [
        "model.safetensors", "config.json", "vision_config.json", "multimodal_config.json",
        "configuration_spark.py", "modeling_spark.py", "experiment_sources.json",
        "implementation/model.py", "tokenizer_config.json", "preprocessor_config.json",
    ]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete multimodal checkpoint: {missing}")
    source_receipt = _check_source(path)
    expected = json.loads((path / "multimodal_config.json").read_text())
    if expected.get("schema_version") != 1 or expected.get("architecture") != "SparkMultimodalModel":
        raise ValueError("Unsupported multimodal architecture/schema")
    config_data = json.loads((path / "config.json").read_text())
    vision_data = json.loads((path / "vision_config.json").read_text())
    if config_data.get("model_type") != "spark2_5" or not config_data.get("tie_word_embeddings"):
        raise ValueError("Expected the tied-embedding Spark2_5 backbone used in this experiment")
    if vision_data.get("model_type") != "siglip_vision_model" or vision_data.get("vision_use_head") is not False:
        raise ValueError("Expected the headless SigLIP vision configuration used in this experiment")
    config_class, model_class = _spark_classes(path)
    config = config_class.from_dict(config_data)
    vision_config = SiglipVisionConfig.from_dict(vision_data)
    with torch.device("meta"):
        language_model = model_class(config)
        vision_encoder = SiglipVisionModel(vision_config)
        model = SparkMultimodalModel(language_model, vision_encoder, pool_grid=expected["pool_grid"])
    actual = model.architecture_manifest()
    if actual != expected:
        differences = {key: {"saved": expected.get(key), "constructed": actual.get(key)}
                       for key in sorted(set(actual) | set(expected)) if actual.get(key) != expected.get(key)}
        raise ValueError(f"Reconstructed architecture differs from checkpoint: {differences}")
    weights = path / "model.safetensors"
    with safe_open(str(weights), framework="pt", device="cpu") as stream:
        metadata = stream.metadata() or {}
    if metadata.get("format") not in {None, "pt"}:
        raise ValueError("Checkpoint safetensors format is not PyTorch")
    state = load_file(str(weights), device="cpu")
    loaded_tensor_count = len(state)
    aliases = _expand_aliases(state, metadata, set(model.state_dict()))
    embedding = state["language_model.model.embedding.weight"]
    head = state["language_model.lm_head.weight"]
    if embedding.dtype != head.dtype or embedding.shape != head.shape or (
        embedding is not head and not torch.equal(embedding, head)
    ):
        raise ValueError("Checkpoint input/output embeddings disagree despite tie_word_embeddings=True")
    incompatibility = model.load_state_dict(state, strict=True, assign=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ValueError(f"Checkpoint key mismatch: {incompatibility}")
    # assign=True creates separate Parameter wrappers even for the same source
    # tensor.  Restore the exact identity required by tied-embedding training.
    model.language_model.lm_head.weight = model.language_model.get_input_embeddings().weight
    repaired = _materialize_nonpersistent_buffers(model)
    meta = [name for name, value in list(model.named_parameters()) + list(model.named_buffers()) if value.is_meta]
    if meta:
        raise ValueError(f"Checkpoint left unmaterialized meta tensors: {meta}")
    saved_dtypes = {name: tensor.dtype for name, tensor in state.items()}
    model.to(device=target_device)
    model.language_model.lm_head.weight = model.language_model.get_input_embeddings().weight
    if model.get_output_embeddings().weight is not model.get_input_embeddings().weight:
        raise AssertionError("Spark input/output embedding Parameter identity was not restored")
    for name, tensor in model.state_dict().items():
        if tensor.dtype != saved_dtypes[name]:
            raise AssertionError(f"Saved dtype changed for {name}")
        if tensor.device.type != target_device.type or (
            target_device.index is not None and tensor.device.index != target_device.index
        ):
            raise AssertionError(f"Wrong tensor device for {name}: {tensor.device}")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    processor = SiglipImageProcessor.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("Saved tokenizer has no pad token for the reserved visual prefix")
    model.bundle_load_receipt = {
        "checkpoint": str(path), "device": str(target_device), "loaded_unique_tensors": loaded_tensor_count,
        "expanded_aliases": aliases, "tied_embedding_parameter_identity": True,
        "nonpersistent_buffers_rebuilt": repaired, "strict_tensor_names": True,
        "per_tensor_dtypes_preserved": True, "meta_tensors_remaining": 0,
        "parameters_by_dtype": dict(Counter({str(dtype): sum(p.numel() for p in model.parameters() if p.dtype == dtype)
                                              for dtype in {p.dtype for p in model.parameters()}})),
        **source_receipt,
    }
    return model, tokenizer, processor


def _autocast(model):
    if model.dtype in (torch.bfloat16, torch.float16):
        return torch.autocast(device_type=model.device.type, dtype=model.dtype)
    return nullcontext()


@torch.inference_mode()
def greedy_generate(model, tokenizer, processor, frames, prompt: str,
                    media_times_s=None, capture_times_s=None, max_new_tokens: int = 24) -> dict:
    """Uncached greedy generation matching the S0 training prompt and prefix."""
    from .collator import clock_features

    if not frames or not prompt.strip():
        raise ValueError("Provide at least one image frame and a nonempty prompt")
    if not (1 <= max_new_tokens <= 512):
        raise ValueError("max_new_tokens must be between 1 and 512")
    frame_count = len(frames)
    media_known = media_times_s is not None
    media_times = list(media_times_s) if media_known else [0.0] * frame_count
    if len(media_times) != frame_count or (capture_times_s is not None and len(capture_times_s) != frame_count):
        raise ValueError("Timestamp lists must contain exactly one value per frame")
    clocks = clock_features(media_times, capture_times_s)
    before = tokenizer("User: " + prompt + "\nAssistant: ", add_special_tokens=False)["input_ids"]
    prefix = frame_count * model.tokens_per_frame
    ids = [tokenizer.pad_token_id] * prefix + before
    if len(ids) <= prefix:
        raise ValueError("Prompt produced no text tokens")
    max_positions = model.config.max_position_embeddings
    if len(ids) + max_new_tokens > max_positions:
        raise ValueError("Requested generation exceeds the saved Spark position limit")
    pixels = processor(images=frames, return_tensors="pt")["pixel_values"].unsqueeze(0).to(model.device)
    mask = torch.ones((1, frame_count), dtype=torch.bool, device=model.device)
    clock_tensor = torch.tensor([clocks], dtype=torch.float32, device=model.device)
    tokens = torch.tensor([ids], dtype=torch.long, device=model.device)
    generated = []
    eos = tokenizer.eos_token_id
    stopped_on_eos = False
    model.eval()
    for _ in range(max_new_tokens):
        with _autocast(model):
            output = model(
                input_ids=tokens, attention_mask=torch.ones_like(tokens), labels=None,
                pixel_values=pixels, frame_mask=mask, clock_features=clock_tensor, use_cache=False,
            )
        next_token = int(output.logits[0, -1].argmax().item())
        del output
        generated.append(next_token)
        tokens = torch.cat((tokens, torch.tensor([[next_token]], device=tokens.device, dtype=torch.long)), dim=1)
        if eos is not None and next_token == eos:
            stopped_on_eos = True
            break
    return {
        "text": tokenizer.decode(generated, skip_special_tokens=True), "generated_token_ids": generated,
        "generated_tokens": len(generated), "stopped_on_eos": stopped_on_eos,
        "input_frames": frame_count, "visual_positions": prefix, "prompt_positions": len(before),
        "media_times_s": media_times, "capture_times_s": capture_times_s,
        "media_times_supplied": media_known,
        "clock_note": "User-provided media timestamps" if media_known else
                      "No media durations supplied; zero time deltas and frame ordinals only",
        "generation_method": "greedy full-prefix regeneration; no KV cache",
        "evidence_scope": "Research checkpoint output; generation alone does not establish multimodal quality",
    }


def cpu_self_test() -> dict:
    """Round-trip a tiny actual Spark/SigLIP checkpoint; no GPU or base weights."""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file, save_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel

    root = Path(__file__).resolve().parents[1]
    temporary_root = root / "tmp"
    temporary_root.mkdir(exist_ok=True)
    import shutil
    with tempfile.TemporaryDirectory(prefix="multimodal-bundle-cpu-", dir=temporary_root) as directory:
        path = Path(directory)
        for name in ("configuration_spark.py", "modeling_spark.py"):
            shutil.copyfile(root / "models" / "Spark-X2.6-4B" / name, path / name)
        config_class, model_class = _spark_classes(path)
        torch.manual_seed(3407)
        config = config_class(
            vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2,
            tie_word_embeddings=True, headwise_attn_output_gate=True,
            layer_types=["sliding_attention", "full_attention"], sliding_window=8,
            rope_parameters={"full_attention": {"rope_theta": 10000, "partial_rotary_factor": 0.25},
                             "sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0}},
        )
        vision_config = SiglipVisionConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            image_size=16, patch_size=8, vision_use_head=False,
        )
        model = SparkMultimodalModel(model_class(config), SiglipVisionModel(vision_config), pool_grid=2).to(torch.bfloat16)
        for module in model.modules():
            if isinstance(module, torch.nn.LayerNorm) or type(module).__name__ == "Spark2_5RMSNorm":
                module.float()
        model.language_model.tie_weights()
        model.eval()
        vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
                 "User": 4, "Assistant": 5, ":": 6, "red": 7, "square": 8}
        core = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
        core.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]")
        processor = SiglipImageProcessor(size={"height": 16, "width": 16})
        tokenizer.save_pretrained(path)
        processor.save_pretrained(path)
        model.config.save_pretrained(path)
        model.vision_encoder.config.to_json_file(path / "vision_config.json")
        model.save_bundle_config(path)
        (path / "implementation").mkdir()
        wrapper = Path(__file__).with_name("model.py")
        shutil.copyfile(wrapper, path / "implementation" / "model.py")
        (path / "experiment_sources.json").write_text(json.dumps({"source_hashes": {"multimodal/model.py": _sha256(wrapper)}}))
        save_model(model, str(path / "model.safetensors"), metadata={"format": "pt"})
        with safe_open(str(path / "model.safetensors"), framework="pt") as stream:
            aliases = {key: value for key, value in stream.metadata().items() if key != "format"}
        assert aliases, "Test must exercise a dropped tied-embedding alias"
        batch = {
            "input_ids": torch.tensor([[0, 0, 0, 0, 4, 6, 7, 8]]),
            "attention_mask": torch.ones((1, 8), dtype=torch.long),
            "pixel_values": torch.randn((1, 1, 3, 16, 16)),
            "frame_mask": torch.ones((1, 1), dtype=torch.bool),
            "clock_features": torch.tensor([[[0., 0., 1., 0., 0., 0., 1., 0.]]]),
        }
        with torch.inference_mode(), _autocast(model):
            expected_logits = model(**batch, labels=None).logits.clone()
        loaded, loaded_tokenizer, loaded_processor = load_bundle(path, device="cpu")
        with torch.inference_mode(), _autocast(loaded):
            actual_logits = loaded(**batch, labels=None).logits
        assert torch.equal(expected_logits, actual_logits), "Checkpoint reload changed exact CPU logits"
        initial_dtypes = {name: parameter.dtype for name, parameter in model.named_parameters()}
        loaded_dtypes = {name: parameter.dtype for name, parameter in loaded.named_parameters()}
        assert initial_dtypes == loaded_dtypes
        assert loaded.get_input_embeddings().weight is loaded.get_output_embeddings().weight
        assert not any(value.is_meta for _, value in list(loaded.named_parameters()) + list(loaded.named_buffers()))
        from PIL import Image
        generated = greedy_generate(
            loaded, loaded_tokenizer, loaded_processor,
            [Image.new("RGB", (16, 16), "red"), Image.new("RGB", (16, 16), "blue")],
            "red square", media_times_s=[0., .5], capture_times_s=[0., 30.], max_new_tokens=2,
        )
        # Name and source-integrity negative checks exercise actual loader gates.
        original_weights = (path / "model.safetensors").read_bytes()
        invalid_tensors = load_file(str(path / "model.safetensors"))
        invalid_tensors["unexpected.tensor"] = torch.ones(1)
        save_file(invalid_tensors, str(path / "corrupt.safetensors"), metadata={"format": "pt", **aliases})
        (path / "corrupt.safetensors").replace(path / "model.safetensors")
        try:
            load_bundle(path)
        except ValueError as exc:
            assert "Unexpected checkpoint tensor" in str(exc)
        else:
            raise AssertionError("Unexpected tensor name was accepted")
        (path / "model.safetensors").write_bytes(original_weights)
        original_manifest = (path / "multimodal_config.json").read_text()
        invalid_manifest = json.loads(original_manifest)
        invalid_manifest["tokens_per_frame"] += 1
        (path / "multimodal_config.json").write_text(json.dumps(invalid_manifest))
        try:
            load_bundle(path)
        except ValueError as exc:
            assert "architecture differs" in str(exc)
        else:
            raise AssertionError("Corrupt architecture manifest was accepted")
        (path / "multimodal_config.json").write_text(original_manifest)
        (path / "implementation" / "model.py").write_text("# deliberately corrupted source\n")
        try:
            load_bundle(path)
        except ValueError as exc:
            assert "source" in str(exc) or "snapshot" in str(exc)
        else:
            raise AssertionError("Corrupt wrapper source snapshot was accepted")
        return {
            "passed": True, "device": "cpu", "actual_spark_and_siglip_architectures": True,
            "parameters": sum(p.numel() for p in loaded.parameters()), "safetensors_aliases": aliases,
            "exact_logits_match": True, "max_absolute_logit_difference": float((actual_logits.float() - expected_logits.float()).abs().max()),
            "mixed_bfloat16_fp32_dtypes_preserved": True, "tied_parameter_identity": True,
            "nonpersistent_position_ids_rebuilt": True, "meta_tensors_remaining": 0,
            "greedy_generation_exercised": True, "generated_tokens": generated["generated_tokens"],
            "multi_frame_and_capture_clocks_exercised": True, "unexpected_tensor_names_rejected": True,
            "architecture_and_source_corruption_rejected": True,
            "scope": "Small actual architecture checkpoint round trip; no production-model quality or GPU evidence",
        }
