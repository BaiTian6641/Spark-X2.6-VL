"""Lightweight, faithful Spark-X2.6 + SigLIP2 synthetic components for fast CPU regression testing.

Mirrors Spark2_5ForCausalLM's exact architectural invariants:
- Tied token embeddings (`embed_tokens` <-> `lm_head` via `tie_weights()`)
- Hybrid `sliding_attention` and `full_attention` layers
- `Spark2_5RMSNorm`, `q_k_v_proj`, `headwise_attn_output_gate`, `o_proj`
- Gated GELU MLP (`act_fn(gate_proj(x)) * up_proj(x)` -> `down_proj`, NOT SiLU)
- Dual `<think>` and non-thinking chat template formatting
- Patch-grid SigLIP2-like vision model (`patch_size=14`, `image_size=56` -> 4x4 patches -> 2x2 pixel-unshuffle)
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F


class FakeTokenizer:
    """Deterministic byte-level tokenizer supporting Spark's `<think>` and non-thinking chat templates."""

    eos_token = "<|im_end|>"
    eos_token_id = 2
    pad_token = "<|endoftext|>"
    pad_token_id = 0

    SPECIAL_TOKENS = {
        "<|endoftext|>": 0,
        "<|im_start|>": 1,
        "<|im_end|>": 2,
        "<think>": 3,
        "</think>": 4,
    }
    ID_TO_SPECIAL = {v: k for k, v in SPECIAL_TOKENS.items()}
    VOCAB_SIZE = 320

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
    ) -> SimpleNamespace:
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if return_tensors == "pt":
            return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))
        return SimpleNamespace(input_ids=ids)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(text):
            matched = False
            for tok_str, tok_id in self.SPECIAL_TOKENS.items():
                if text.startswith(tok_str, i):
                    out.append(tok_id)
                    i += len(tok_str)
                    matched = True
                    break
            if not matched:
                for b in text[i].encode("utf-8"):
                    out.append(16 + int(b))
                i += 1
        return out

    def decode(self, ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        chunks: list[bytes] = []
        out_parts: list[str] = []

        def _flush_bytes() -> None:
            if chunks:
                out_parts.append(b"".join(chunks).decode("utf-8", errors="replace"))
                chunks.clear()

        for raw in ids:
            tid = int(raw)
            if tid in self.ID_TO_SPECIAL:
                if tid in (3, 4) or not skip_special_tokens:
                    _flush_bytes()
                    out_parts.append(self.ID_TO_SPECIAL[tid])
            elif 16 <= tid < 272:
                chunks.append(bytes([tid - 16]))
        _flush_bytes()
        return "".join(out_parts)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
    ) -> str | list[int]:
        parts: list[str] = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "assistant" and not enable_thinking and not content.startswith("<think>"):
                parts.append(f"<|im_start|>assistant\n<think>\n\n</think>\n{content}<|im_end|>\n")
            else:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            if enable_thinking:
                parts.append("<|im_start|>assistant\n<think>\n")
            else:
                parts.append("<|im_start|>assistant\n<think>\n\n</think>\n")
        rendered = "".join(parts)
        return self.encode(rendered) if tokenize else rendered


class Spark2_5RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * rms).to(orig_dtype) * self.weight.to(orig_dtype)


class SyntheticSparkAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, layer_type: str, sliding_window: int = 8) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.layer_type = layer_type
        self.sliding_window = sliding_window
        self.q_k_v_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.headwise_attn_output_gate = nn.Linear(hidden_size, num_heads, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        bsz, q_len, _ = x.shape
        qkv = self.q_k_v_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        past_len = 0
        if past_kv is not None:
            past_len = past_kv[0].shape[-2]
            k = torch.cat([past_kv[0], k], dim=-2)
            v = torch.cat([past_kv[1], v], dim=-2)
        next_kv = (k, v) if use_cache else None

        kv_len = k.shape[-2]
        q_pos = torch.arange(past_len, past_len + q_len, device=x.device).unsqueeze(1)
        k_pos = torch.arange(0, kv_len, device=x.device).unsqueeze(0)
        allowed = k_pos <= q_pos
        if self.layer_type == "sliding_attention":
            allowed = allowed & ((q_pos - k_pos) < self.sliding_window)
        if attention_mask is not None:
            allowed = allowed.unsqueeze(0) & attention_mask[:, :kv_len].bool().unsqueeze(1)
        else:
            allowed = allowed.unsqueeze(0)

        attn_bias = torch.zeros((bsz, 1, q_len, kv_len), dtype=q.dtype, device=x.device)
        attn_bias = attn_bias.masked_fill(~allowed.unsqueeze(1), float("-inf"))
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, is_causal=False)
        gate = torch.sigmoid(self.headwise_attn_output_gate(x)).transpose(1, 2).unsqueeze(-1)
        attn_out = (attn_out * gate).transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_out), next_kv


class SyntheticSparkGatedGELUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SyntheticSparkDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, layer_type: str) -> None:
        super().__init__()
        self.input_layernorm = Spark2_5RMSNorm(hidden_size)
        self.self_attn = SyntheticSparkAttention(hidden_size, num_heads, layer_type=layer_type)
        self.post_attention_layernorm = Spark2_5RMSNorm(hidden_size)
        self.mlp = SyntheticSparkGatedGELUMLP(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, next_kv = self.self_attn(
            self.input_layernorm(x),
            attention_mask=attention_mask,
            past_kv=past_kv,
            use_cache=use_cache,
        )
        h = x + attn_out
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h, next_kv


class SyntheticSparkInnerModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, intermediate_size: int, num_heads: int, num_layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        layer_types = [
            "full_attention" if (i + 1) % 2 == 0 else "sliding_attention"
            for i in range(num_layers)
        ]
        self.layers = nn.ModuleList(
            [
                SyntheticSparkDecoderLayer(hidden_size, intermediate_size, num_heads, lt)
                for lt in layer_types
            ]
        )
        self.norm = Spark2_5RMSNorm(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **_kwargs: Any,
    ) -> SimpleNamespace:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Specify input_ids or inputs_embeds")
            h = self.embed_tokens(input_ids)
        else:
            h = inputs_embeds.to(self.embed_tokens.weight.dtype)

        hidden_states = [h] if output_hidden_states else None
        next_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if use_cache else None
        for idx, layer in enumerate(self.layers):
            pkv = past_key_values[idx] if past_key_values is not None else None
            h, layer_kv = layer(h, attention_mask=attention_mask, past_kv=pkv, use_cache=use_cache)
            if output_hidden_states and hidden_states is not None:
                hidden_states.append(h)
            if use_cache and next_cache is not None and layer_kv is not None:
                next_cache.append(layer_kv)
        h = self.norm(h)
        if output_hidden_states and hidden_states is not None:
            hidden_states[-1] = h
        return SimpleNamespace(
            last_hidden_state=h,
            past_key_values=next_cache,
            hidden_states=hidden_states,
        )


class SyntheticSparkCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = FakeTokenizer.VOCAB_SIZE,
        hidden_size: int = 128,
        intermediate_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 6,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            use_cache=False,
        )
        self.generation_config = SimpleNamespace(eos_token_id=FakeTokenizer.eos_token_id)
        self.model = SyntheticSparkInnerModel(vocab_size, hidden_size, intermediate_size, num_heads, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **_kwargs: Any,
    ) -> SimpleNamespace:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Specify input_ids or inputs_embeds")
            h = self.model.embed_tokens(input_ids)
        else:
            h = inputs_embeds.to(self.model.embed_tokens.weight.dtype)

        hidden_states = [h] if output_hidden_states else None
        next_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if use_cache else None
        for idx, layer in enumerate(self.model.layers):
            pkv = past_key_values[idx] if past_key_values is not None else None
            h, layer_kv = layer(h, attention_mask=attention_mask, past_kv=pkv, use_cache=use_cache)
            if output_hidden_states and hidden_states is not None:
                hidden_states.append(h)
            if use_cache and next_cache is not None and layer_kv is not None:
                next_cache.append(layer_kv)
        h = self.model.norm(h)
        if output_hidden_states and hidden_states is not None:
            hidden_states[-1] = h
        logits = self.lm_head(h)
        return SimpleNamespace(
            logits=logits,
            past_key_values=next_cache,
            hidden_states=hidden_states,
        )


class FakeImageProcessor:
    def __init__(self, image_size: int = 56, patch_size: int = 14) -> None:
        self.image_size = image_size
        self.patch_size = patch_size

    def __call__(
        self,
        images: Image.Image | list[Image.Image],
        return_tensors: str = "pt",
        max_num_patches: int | None = None,
    ) -> dict[str, torch.Tensor]:
        img_list = [images] if isinstance(images, Image.Image) else list(images)
        hh = self.image_size // self.patch_size
        ww = self.image_size // self.patch_size
        tensors = []
        for im in img_list:
            rgb = im.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
        bsz = len(tensors)
        return {
            "pixel_values": torch.stack(tensors, dim=0),
            "pixel_attention_mask": torch.ones((bsz, hh * ww), dtype=torch.long),
            "spatial_shapes": torch.tensor([[hh, ww]] * bsz, dtype=torch.long),
        }


class SyntheticSiglip2EncoderLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.gelu(self.fc1(self.norm1(x))))


class SyntheticSiglip2VisionModel(nn.Module):
    def __init__(self, hidden_size: int = 64, image_size: int = 56, patch_size: int = 14, num_layers: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, image_size=image_size, patch_size=patch_size)
        self.patch_embed = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)
        encoder_layers = nn.ModuleList([SyntheticSiglip2EncoderLayer(hidden_size) for _ in range(num_layers)])
        self.vision_model = nn.Module()
        self.vision_model.encoder = nn.Module()
        self.vision_model.encoder.layers = encoder_layers
        self.vision_model.post_layernorm = nn.LayerNorm(hidden_size)
        self.vision_model.use_head = False
        self.forward_calls = 0

    def forward(self, pixel_values: torch.Tensor, **_kwargs: Any) -> SimpleNamespace:
        self.forward_calls += 1
        x = self.patch_embed(pixel_values)
        x = x.flatten(2).transpose(1, 2)
        for layer in self.vision_model.encoder.layers:
            x = layer(x)
        x = self.vision_model.post_layernorm(x)
        return SimpleNamespace(last_hidden_state=x)


def make_synthetic_spark_native(
    extra_policy: dict[str, Any] | None = None,
    *,
    seed: int = 1234,
    hidden_size: int = 128,
    intermediate_size: int = 256,
):
    from colab.final.model import SparkNative

    torch.manual_seed(seed)
    spark = SyntheticSparkCausalLM(hidden_size=hidden_size, intermediate_size=intermediate_size)
    vision = SyntheticSiglip2VisionModel(hidden_size=64, image_size=56, patch_size=14, num_layers=4)
    tokenizer = FakeTokenizer()
    processor = FakeImageProcessor(image_size=56)

    policy: dict[str, Any] = {
        "modalities": "vision_text",
        "spark_dtype": "float32",
        "vision_dtype": "float32",
        "max_sequence": 512,
        "max_answer_tokens": 256,
        "max_images_per_example": 6,
        "detail_tiles": 4,
        "thumbnail_patches": 16,
        "tile_patches": 16,
        "max_visual_tokens": 256,
        "microbatch_size": 2,
        "visual_cross_attn_blocks": 0,
        "optimizer_profile": "apollo_coat",
        "apollo_rank": 16,
        "apollo_update_proj_gap": 2,
        "apollo_scale": 1.0,
        "apollo_min_dim": 64,
        "stochastic_rounding": True,
        "coat_activation_fp8": True,
        "coat_activation_group_size": 16,
        "coat_fp8_linear_forward": False,
        "merged_grad_norm_mode": "per_example",
        "backbone_lr": 2e-4,
        "adapter_lr": 5e-4,
        "vision_lr": 5e-5,
        "weight_decay": 0.01,
        "clip_norm": 1.0,
        "warmup_ratio": 0.1,
        "seed": seed,
    }
    if extra_policy:
        policy.update(extra_policy)

    return SparkNative(
        paths={},
        policy=policy,
        device="cpu",
        pretrained=False,
        components={
            "spark": spark,
            "tokenizer": tokenizer,
            "vision": vision,
            "image_processor": processor,
        },
    )
