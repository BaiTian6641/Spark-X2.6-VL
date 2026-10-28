"""Differentiable image/video embedding path into the unchanged Spark backbone.

This first stage uses ordered semantic frames and explicit clock features.
The dense V-JEPA motion branch and audio branches are subsequent experiments.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class FP32LayerNorm(nn.LayerNorm):
    """Keep small norm updates and checkpoint recomputation numerically stable."""
    def forward(self, inputs):
        dtype=inputs.dtype
        with torch.autocast(device_type=inputs.device.type,enabled=False):
            values=F.layer_norm(inputs.float(),self.normalized_shape,
                                self.weight.float() if self.weight is not None else None,
                                self.bias.float() if self.bias is not None else None,self.eps)
        return values.to(dtype)


def stabilize_layer_norms(module):
    for name,child in list(module.named_children()):
        if isinstance(child,nn.LayerNorm) and not isinstance(child,FP32LayerNorm):
            replacement=FP32LayerNorm(child.normalized_shape,eps=child.eps,
                elementwise_affine=child.elementwise_affine,bias=child.bias is not None,
                device=child.weight.device if child.weight is not None else None,dtype=torch.float32)
            replacement.load_state_dict(child.state_dict())
            setattr(module,name,replacement)
        else:
            stabilize_layer_norms(child)


class SparkMultimodalModel(nn.Module):
    main_input_name = "input_ids"

    def __init__(self, language_model, vision_encoder, pool_grid=4):
        super().__init__()
        self.language_model = language_model
        self.vision_encoder = vision_encoder
        stabilize_layer_norms(self.vision_encoder)
        self.config = language_model.config
        self.generation_config = language_model.generation_config
        self.name_or_path = getattr(language_model, "name_or_path", self.config._name_or_path)
        self.warnings_issued = {}
        self.pool_grid = int(pool_grid)
        self.tokens_per_frame = self.pool_grid ** 2
        width = vision_encoder.config.hidden_size
        hidden = self.config.hidden_size
        self.visual_projector = nn.Sequential(
            FP32LayerNorm(width), nn.Linear(width, 1024), nn.GELU(), nn.Linear(1024, hidden)
        )
        self.position_projector = nn.Sequential(nn.Linear(10, 128), nn.GELU(), nn.Linear(128, hidden))
        self.media_embedding = nn.Parameter(torch.empty(hidden))
        nn.init.normal_(self.media_embedding, std=.005)
        nn.init.normal_(self.visual_projector[-1].weight, std=.002)
        nn.init.zeros_(self.visual_projector[-1].bias)
        nn.init.normal_(self.position_projector[-1].weight, std=.002)
        nn.init.zeros_(self.position_projector[-1].bias)
        self._unsloth_full_finetuning = True
        self.is_loaded_in_4bit = False
        self.is_loaded_in_8bit = False
        self.last_batch_accounting = {}
        self._vision_gradient_probe = None

    @property
    def device(self):
        return next(self.language_model.parameters()).device

    @property
    def dtype(self):
        return self.get_input_embeddings().weight.dtype

    @property
    def is_gradient_checkpointing(self):
        return self.language_model.is_gradient_checkpointing

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )
        self.vision_encoder.gradient_checkpointing_enable(
            # Raw pixels have no required gradient. Reentrant checkpointing could
            # otherwise detach the first vision block and silently freeze it.
            gradient_checkpointing_kwargs={**(gradient_checkpointing_kwargs or {}), 'use_reentrant': False}
        )

    def gradient_checkpointing_disable(self):
        self.language_model.gradient_checkpointing_disable()
        self.vision_encoder.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        self.language_model.enable_input_require_grads()

    def for_training(self, use_gradient_checkpointing=True):
        self.train()
        if hasattr(self.language_model, 'for_training'):
            self.language_model.for_training(use_gradient_checkpointing=use_gradient_checkpointing)
        return self

    def component_name(self, parameter_name):
        return parameter_name.split('.', 1)[0]

    def encode_media(self, pixel_values, frame_mask, clock_features):
        """Return B,F,P,H features; invalid frames are not encoded or trained."""
        if pixel_values.ndim != 5:
            raise ValueError('pixel_values must have shape [batch, frames, channels, height, width]')
        b, frames, channels, h, w = pixel_values.shape
        if frame_mask.shape != (b, frames) or clock_features.shape != (b, frames, 8):
            raise ValueError('Frame mask/clock features do not align with media')
        valid = frame_mask.reshape(-1).bool()
        if not valid.any():
            raise ValueError('A media batch must contain at least one valid frame')
        inputs = pixel_values.reshape(b * frames, channels, h, w)[valid]
        inputs = inputs.to(dtype=next(self.vision_encoder.parameters()).dtype)
        features = self.vision_encoder(pixel_values=inputs, return_dict=True).last_hidden_state
        side = math.isqrt(features.shape[1])
        if side * side != features.shape[1]:
            raise ValueError('S0 pooling requires the fixed-resolution square patch grid')
        patches = features.transpose(1, 2).reshape(features.shape[0], features.shape[-1], side, side)
        patches = F.adaptive_avg_pool2d(patches, (self.pool_grid, self.pool_grid))
        patches = patches.flatten(2).transpose(1, 2)
        projected = self.visual_projector(patches)
        if self.training and projected.requires_grad:
            projected.register_hook(self._record_media_gradient)
        padded = torch.zeros((b * frames, self.tokens_per_frame, projected.shape[-1]),
                             device=projected.device, dtype=projected.dtype)
        padded[valid] = projected
        padded = padded.reshape(b, frames, self.tokens_per_frame, -1)
        coords = (torch.arange(self.pool_grid, device=projected.device, dtype=projected.dtype) + .5) / self.pool_grid
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        xy = torch.stack((xx, yy), -1).reshape(1, 1, self.tokens_per_frame, 2).expand(b, frames, -1, -1)
        clocks = clock_features.to(projected.dtype).unsqueeze(2).expand(-1, -1, self.tokens_per_frame, -1)
        positional = self.position_projector(torch.cat((xy, clocks), dim=-1))
        media = padded + positional + self.media_embedding
        return media * frame_mask[:, :, None, None].to(media.dtype)

    def _record_media_gradient(self, gradient):
        self._vision_gradient_probe = {
            'finite': bool(torch.isfinite(gradient).all().item()),
            'absolute_sum': float(gradient.detach().float().abs().sum().item()),
        }

    def build_spark_inputs(self, input_ids, attention_mask, pixel_values, frame_mask, clock_features, labels=None):
        media = self.encode_media(pixel_values, frame_mask, clock_features)
        b, frames, patches, hidden = media.shape
        prefix = frames * patches
        if input_ids.shape[0] != b or input_ids.shape[1] <= prefix:
            raise ValueError('Input IDs must reserve the visual prefix and contain text')
        if labels is not None and not (labels[:, :prefix] == -100).all():
            raise ValueError('Visual prefix positions must be masked from the language loss')
        expected = frame_mask[:, :, None].expand(-1, -1, patches).reshape(b, prefix)
        if not torch.equal(attention_mask[:, :prefix].bool(), expected.bool()):
            raise ValueError('Visual prefix attention mask does not match frame validity')
        text_embeddings = self.get_input_embeddings()(input_ids[:, prefix:])
        return torch.cat((media.reshape(b, prefix, hidden).to(text_embeddings.dtype), text_embeddings), dim=1)

    def forward(self, input_ids=None, attention_mask=None, labels=None, pixel_values=None,
                frame_mask=None, clock_features=None, use_cache=False, **kwargs):
        if use_cache:
            raise ValueError('S0 forward uses a complete media prefix; cached generation is not implemented')
        if pixel_values is None:
            raise ValueError('Use language_model directly for text-only controls')
        embeddings = self.build_spark_inputs(input_ids, attention_mask, pixel_values, frame_mask, clock_features, labels)
        self.last_batch_accounting = {
            'input_positions': int(attention_mask.sum().item()),
            'target_positions': int((labels != -100).sum().item()) if labels is not None else 0,
            'frames': int(frame_mask.sum().item()),
            'visual_positions': int(frame_mask.sum().item()) * self.tokens_per_frame,
        }
        # Spark's native output already contains a shifted causal LM loss and tied logits.
        return self.language_model(inputs_embeds=embeddings, attention_mask=attention_mask,
                                   labels=labels, use_cache=False, return_dict=True)

    def architecture_manifest(self):
        return {
            'schema_version': 1, 'architecture': 'SparkMultimodalModel',
            'stage': 'semantic_frame_and_clock_baseline', 'pool_grid': self.pool_grid,
            'tokens_per_frame': self.tokens_per_frame, 'spark_hidden_size': self.config.hidden_size,
            'vision_hidden_size': self.vision_encoder.config.hidden_size,
            'clock_features': ['log1p_media_delta', 'sin_media_delta', 'cos_media_delta',
                               'capture_known', 'log1p_capture_delta', 'sin_capture_delta',
                               'cos_capture_delta', 'normalized_frame_ordinal'],
            'coordinate_convention': 'normalized patch centers, x right and y down',
            'temporal_motion_encoder': None, 'audio_components': None,
            'fixed_vocabulary': True, 'visual_prefix_label': -100,
            'parameter_counts': {name: sum(p.numel() for p in module.parameters())
                                 for name, module in self.named_children()},
            'total_parameters': sum(p.numel() for p in self.parameters()),
        }

    def save_bundle_config(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path/'multimodal_config.json').write_text(json.dumps(self.architecture_manifest(), indent=2)+'\n')
