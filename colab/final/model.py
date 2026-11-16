"""Native differentiable multimodal Spark wrapper with pure-text bypass, RGBA/pose encoding, and gated cross-attention.

Key contracts:
- Pure-text requests (`not has_multimodal_context(media)`) bypass all visual/audio encoders,
  projectors, clocks, and cross-attention blocks, calling Spark directly with native KV cache,
  dual `<think>` / non-thinking chat formatting, and full generation controls.
- `modalities='vision_text'` loads only Spark + SigLIP2 (deferring Whisper/CLAP/V-JEPA/TTS/DiT)
  so full 4B finetuning with APOLLO + COAT + merged gradients fits comfortably in 24 GB VRAM.
- Visual encoding preserves RGBA alpha channels, detail tiles on multiviews/UVs/state-pairs,
  and explicit `visual_kind`, `view_id`, and orthographic camera basis conditioning.
"""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
import functools, hashlib, math, sys
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .common import HERE, read, sha
from .data import VIEW_IDS, VIEW_ID_TO_INDEX, VISUAL_KINDS, VISUAL_KIND_TO_INDEX, extract_image_rgb_and_alpha

_ACTIVE_VISUAL_CONTEXT: ContextVar[torch.Tensor | None] = ContextVar('_ACTIVE_VISUAL_CONTEXT', default=None)
_INSIDE_VIS_CKPT_WRAPPER: ContextVar[bool] = ContextVar('_INSIDE_VIS_CKPT_WRAPPER', default=False)
_SUPPRESS_XATTN_HOOK: ContextVar[bool] = ContextVar('_SUPPRESS_XATTN_HOOK', default=False)

def install_spark_sdpa(spark):
    """Replace only the reference attention kernel, keeping masks/RoPE/gates."""
    module=sys.modules[type(spark).__module__]
    current=getattr(module,'eager_attention_forward',None)
    if current is None or getattr(current,'_spark_native_sdpa',False): return
    reference=current
    def sdpa(attention,query,key,value,attention_mask=None,scaling=None,dropout=0.,**kwargs):
        key=module.repeat_kv(key,attention.num_key_value_groups)
        value=module.repeat_kv(value,attention.num_key_value_groups)
        mask=attention_mask[:,:,:,:key.shape[-2]] if attention_mask is not None else None
        result=F.scaled_dot_product_attention(query,key,value,attn_mask=mask,dropout_p=dropout if attention.training else 0.,is_causal=False,scale=scaling)
        return result,None
    sdpa._spark_native_sdpa=True; sdpa._spark_reference=reference
    module.eager_attention_forward=sdpa

SYSTEM_PROMPT = ('You are Spark, an English and Chinese multimodal assistant. Use the supplied media and clocks. '
                 'Report uncertainty when capture time or evidence is unknown.')


def has_multimodal_context(media:dict[str,Any]|None)->bool:
    """Return True iff any visual, motion, or input audio conditioning is present."""
    if not media:
        return False
    return bool(
        media.get('images')
        or media.get('motion')
        or media.get('audio') is not None
    )


class ChatFormat:
    """Exact Spark chat layout supporting both non-thinking and `<think>` modes plus native system prompts."""

    def __init__(self, tokenizer, system=SYSTEM_PROMPT):
        self.tok = tokenizer
        self.default_system = system
        self.layouts = {}
        for thinking in (False, True):
            self.layouts[thinking] = self._probe_layout(tokenizer, system, enable_thinking=thinking)
        # Retain top-level attributes for non-thinking backward compatibility.
        base = self.layouts[False]
        self.system_block = base['system_block']
        self.user_to_bot = base['user_to_bot']
        self.bot_to_user = base['bot_to_user']
        self.bot_close = base['bot_close']
        self.user_open = base['user_open']
        self.cache = {}

    @staticmethod
    def _probe_layout(tokenizer, system: str, *, enable_thinking: bool) -> dict[str, str]:
        marks = ['\x01S\x01', '\x01U1\x01', '\x01A1\x01', '\x01U2\x01', '\x01A2\x01']
        messages = [
            {'role': 'system', 'content': marks[0]},
            {'role': 'user', 'content': marks[1]},
            {'role': 'assistant', 'content': marks[2]},
            {'role': 'user', 'content': marks[3]},
            {'role': 'assistant', 'content': marks[4]},
        ]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=enable_thinking)
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        positions = [text.index(mark) for mark in marks]
        if positions != sorted(positions):
            raise ValueError('Unexpected Spark chat-template ordering')
        sys_prefix = text[:positions[0]]
        sys_suffix = text[positions[0] + len(marks[0]):positions[1]]
        system_block = sys_prefix + system + sys_suffix
        user_to_bot = text[positions[1] + len(marks[1]):positions[2]]
        bot_to_user = text[positions[2] + len(marks[2]):positions[3]]
        bot_close = text[positions[4] + len(marks[4]):]
        if not bot_to_user.startswith(bot_close):
            raise ValueError('Assistant close marker is inconsistent')
        user_open = bot_to_user[len(bot_close):]
        try:
            generated = tokenizer.apply_chat_template(
                messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
        except TypeError:
            generated = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
        # In thinking mode, add_generation_prompt may append `<think>\n` to `user_to_bot`.
        gen_prefix = sys_prefix + marks[0] + sys_suffix + marks[1]
        if not generated.startswith(gen_prefix):
            raise ValueError('Generation prompt prefix differs from the training layout')
        gen_user_to_bot = generated[len(gen_prefix):]
        if not enable_thinking and gen_user_to_bot != user_to_bot:
            raise ValueError('Generation prompt differs from the training layout')
        # In templates like Spark-X2.6 (`{% if reasoning_content %}<think>...{% else %}</think>{% endif %}`),
        # probing an assistant message without `reasoning_content` appends an empty-reasoning `</think>` to `user_to_bot`,
        # whereas supervised thinking answers already begin with `<think>...</think>`. Strip that empty `</think>`
        # so `user_to_bot + "<think>..."` matches `gen_user_to_bot` token-for-token!
        if enable_thinking and user_to_bot.endswith('</think>'):
            stripped = user_to_bot[:-len('</think>')]
            if stripped.endswith('<think>'):
                stripped = stripped[:-len('<think>')]
            if gen_user_to_bot.startswith(stripped):
                user_to_bot = stripped
        gen_open_prefix = gen_user_to_bot[len(user_to_bot):] if gen_user_to_bot.startswith(user_to_bot) else ''
        return {
            'sys_prefix': sys_prefix,
            'sys_suffix': sys_suffix,
            'system_block': system_block,
            'user_to_bot': user_to_bot,
            'gen_user_to_bot': gen_user_to_bot,
            'gen_open_prefix': gen_open_prefix,
            'bot_to_user': bot_to_user,
            'bot_close': bot_close,
            'user_open': user_open,
        }

    @staticmethod
    def restore_thinking_prefix(decoded: str, layout: dict[str, str], enable_thinking: bool) -> str:
        if not enable_thinking or decoded.lstrip().startswith('<think>'):
            return decoded
        open_pref = layout.get('gen_open_prefix', '')
        if open_pref == '<think>' and not decoded.startswith('\n'):
            return '<think>\n' + decoded
        return (open_pref or '<think>\n') + decoded

    def ids(self, text: str) -> list[int]:
        if text not in self.cache:
            self.cache[text] = self.tok(text, add_special_tokens=False).input_ids
        return list(self.cache[text])

    def encode(self, prompt: str, *, enable_thinking: bool = False, system: str | None = None, for_generation: bool = False):
        """Return `(pre_ids, post_ids)` for system/user-open and user-content/assistant-open."""
        layout = self.layouts[bool(enable_thinking)]
        sys_block = (
            layout['system_block']
            if system is None
            else (layout['sys_prefix'] + system + layout['sys_suffix'])
        )
        pre = self.ids(sys_block)
        turn_Transition = layout['gen_user_to_bot'] if for_generation else layout['user_to_bot']
        post = self.tok(prompt, add_special_tokens=False).input_ids + self.ids(turn_Transition)
        return pre, post


class Resampler(nn.Module):
    def __init__(self,dim,hidden,n,heads=8):
        super().__init__(); self.input=nn.Linear(dim,hidden); self.queries=nn.Parameter(torch.randn(n,hidden)*.02)
        self.norm=nn.LayerNorm(hidden); self.attn=nn.MultiheadAttention(hidden,heads,batch_first=True); self.out=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden))
    def forward(self,x,position=None):
        x=self.input(x)
        if position is not None: x=x+position.to(x.dtype)
        q=self.queries[None].expand(x.shape[0],-1,-1).to(x.dtype)
        return self.out(q+self.attn(q,self.norm(x),self.norm(x),need_weights=False)[0])


class GatedVisualCrossAttentionBlock(nn.Module):
    """Flamingo-style zero-initialized tanh-gated visual cross-attention + gated-GELU FFN adapter.

    At initialization (`gate_attn == 0`, `gate_ffn == 0`), outputs exact identity `x`.
    Completely skipped when no visual context is active.
    """
    def __init__(self, hidden_size: int, num_heads: int = 8, ffn_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.drop = nn.Dropout(float(dropout))
        self.norm_q = nn.LayerNorm(hidden_size)
        self.norm_kv = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=float(dropout), batch_first=True)
        self.gate_attn = nn.Parameter(torch.zeros(1))

        inner = hidden_size * ffn_mult
        self.norm_ffn = nn.LayerNorm(hidden_size)
        self.ffn_gate = nn.Linear(hidden_size, inner)
        self.ffn_up = nn.Linear(hidden_size, inner)
        self.ffn_down = nn.Linear(inner, hidden_size)
        self.gate_ffn = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states: torch.Tensor, visual_tokens: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        if hidden_states.is_cuda and torch.is_autocast_enabled():
            compute_dtype = torch.get_autocast_gpu_dtype()
        elif orig_dtype in (torch.bfloat16, torch.float16):
            compute_dtype = orig_dtype
        else:
            compute_dtype = self.norm_q.weight.dtype

        vis = visual_tokens.to(compute_dtype)
        if vis.shape[0] == 1 and hidden_states.shape[0] > 1:
            vis = vis.expand(hidden_states.shape[0], -1, -1)

        d = self.hidden_size
        head_dim = d // self.num_heads
        res = hidden_states
        q = F.layer_norm(
            res.float(),
            (d,),
            self.norm_q.weight.float(),
            self.norm_q.bias.float() if self.norm_q.bias is not None else None,
            self.norm_q.eps,
        ).to(compute_dtype)
        kv = F.layer_norm(
            vis.float(),
            (d,),
            self.norm_kv.weight.float(),
            self.norm_kv.bias.float() if self.norm_kv.bias is not None else None,
            self.norm_kv.eps,
        ).to(compute_dtype)

        w_in = self.attn.in_proj_weight.to(compute_dtype)
        b_in = self.attn.in_proj_bias.to(compute_dtype) if self.attn.in_proj_bias is not None else None
        q_proj = F.linear(q, w_in[:d], None if b_in is None else b_in[:d])
        kv_proj = F.linear(kv, w_in[d:], None if b_in is None else b_in[d:])
        k_proj, v_proj = kv_proj.chunk(2, dim=-1)

        bsz, q_len, _ = q_proj.shape
        kv_len = k_proj.shape[1]
        q_heads = q_proj.view(bsz, q_len, self.num_heads, head_dim).transpose(1, 2)
        k_heads = k_proj.view(bsz, kv_len, self.num_heads, head_dim).transpose(1, 2)
        v_heads = v_proj.view(bsz, kv_len, self.num_heads, head_dim).transpose(1, 2)
        attn_p = float(self.attn.dropout) if self.training else 0.0
        attn_ctx = F.scaled_dot_product_attention(q_heads, k_heads, v_heads, dropout_p=attn_p)
        attn_ctx = attn_ctx.transpose(1, 2).contiguous().view(bsz, q_len, d)

        w_out = self.attn.out_proj.weight.to(compute_dtype)
        b_out = self.attn.out_proj.bias.to(compute_dtype) if self.attn.out_proj.bias is not None else None
        attn_out = self.drop(F.linear(attn_ctx, w_out, b_out))
        res = res + (torch.tanh(self.gate_attn).float() * attn_out.float()).to(orig_dtype)

        h = F.layer_norm(
            res.float(),
            (d,),
            self.norm_ffn.weight.float(),
            self.norm_ffn.bias.float() if self.norm_ffn.bias is not None else None,
            self.norm_ffn.eps,
        ).to(compute_dtype)
        fg = F.linear(
            h,
            self.ffn_gate.weight.to(compute_dtype),
            None if self.ffn_gate.bias is None else self.ffn_gate.bias.to(compute_dtype),
        )
        fu = F.linear(
            h,
            self.ffn_up.weight.to(compute_dtype),
            None if self.ffn_up.bias is None else self.ffn_up.bias.to(compute_dtype),
        )
        ffn_hidden = self.drop(F.gelu(fg, approximate='tanh') * fu)
        ffn_out = self.drop(
            F.linear(
                ffn_hidden,
                self.ffn_down.weight.to(compute_dtype),
                None if self.ffn_down.bias is None else self.ffn_down.bias.to(compute_dtype),
            )
        )
        res = res + (torch.tanh(self.gate_ffn).float() * ffn_out.float()).to(orig_dtype)
        return res


def spatial_downshuffle(x,h,w):
    """Preserve the actual NaFlex grid; concatenate each real 2x2 neighborhood."""
    if h*w>x.shape[1] or h<1 or w<1: raise ValueError('Invalid native patch grid')
    x=x[:,:h*w].reshape(x.shape[0],h,w,x.shape[-1]); valid=torch.ones((h,w),device=x.device,dtype=torch.bool)
    if h%2: x=F.pad(x,(0,0,0,0,0,1)); valid=F.pad(valid,(0,0,0,1))
    if w%2: x=F.pad(x,(0,0,0,1)); valid=F.pad(valid,(0,1))
    hh,ww=x.shape[1:3]
    out=x.reshape(x.shape[0],hh//2,2,ww//2,2,x.shape[-1]).permute(0,1,3,2,4,5).flatten(3).flatten(1,2)
    mask=valid.reshape(hh//2,2,ww//2,2).permute(0,2,1,3).flatten(2).any(-1).flatten()
    return out[:,mask],mask


def compute_patch_alpha_transparency_features(
    alpha_hw: np.ndarray | None,
    hh: int,
    ww: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute 16-dim downshuffled transparency features per visual token.

    For completely opaque RGB images (`alpha_hw is None` or `alpha == 1.0` everywhere),
    returns exact zeros `[1, num_tokens, 16]` so zero-bias `alpha_projector` adds 0.0 shift!
    For RGBA cutouts / translucent textures, encodes `[mean_transp, max_transp, min_transp, edge_std]`
    per 2x2 sub-patch, cleanly separating opaque, cutout, and semi-transparent UV islands.
    """
    gh, gw = (hh + 1) // 2, (ww + 1) // 2
    if alpha_hw is None:
        return torch.zeros((1, gh * gw, 16), device=device, dtype=torch.float32)
    a_t = torch.as_tensor(alpha_hw, device=device, dtype=torch.float32)
    if a_t.ndim != 2 or a_t.numel() == 0 or float(a_t.min().item()) >= 1.0 - 1e-6:
        return torch.zeros((1, gh * gw, 16), device=device, dtype=torch.float32)

    transp = (1.0 - a_t.clamp(0.0, 1.0))[None, None]  # [1, 1, H, W]
    mean_t = F.adaptive_avg_pool2d(transp, (hh, ww))
    max_t = F.adaptive_max_pool2d(transp, (hh, ww))
    min_t = 1.0 - F.adaptive_max_pool2d(1.0 - transp, (hh, ww))
    sq_t = F.adaptive_avg_pool2d(transp * transp, (hh, ww))
    std_t = (sq_t - mean_t * mean_t).clamp_min(0.0).sqrt()

    feat = torch.cat([mean_t, max_t, min_t, std_t], dim=1)  # [1, 4, hh, ww]
    feat_tokens = feat.permute(0, 2, 3, 1).reshape(1, hh * ww, 4)
    downshuffled, _ = spatial_downshuffle(feat_tokens, hh, ww)
    return downshuffled


def _sample_next_token(
    logits: torch.Tensor,
    *,
    generated_ids: list[int],
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int | None = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """Select next token `[1, 1]` supporting exact greedy argmax and standard nucleus/top-k sampling."""
    scores = logits[:, -1, :].float().clone()
    if repetition_penalty is not None and repetition_penalty != 1.0 and generated_ids:
        seen = torch.tensor(sorted(set(generated_ids)), device=scores.device, dtype=torch.long)
        seen_scores = scores[:, seen]
        scores[:, seen] = torch.where(
            seen_scores < 0, seen_scores * repetition_penalty, seen_scores / repetition_penalty
        )
    if not do_sample or temperature <= 0.0:
        return scores.argmax(dim=-1, keepdim=True)
    if temperature != 1.0:
        scores = scores / float(temperature)
    if top_k is not None and top_k > 0 and top_k < scores.shape[-1]:
        kth = torch.topk(scores, int(top_k), dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < kth, float('-inf'))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        cutoff = cumprobs > float(top_p)
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, float('-inf'))
        scores = torch.zeros_like(scores).scatter(-1, sorted_indices, sorted_logits)
    probs = F.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def audio_resample(wav,sr,target):
    import scipy.signal
    if sr==target: return wav
    g=math.gcd(sr,target); return scipy.signal.resample_poly(wav,target//g,sr//g,axis=-1).astype('float32')


_FP8_TARGET_SUFFIXES = (
    '.self_attn.q_k_v_proj.weight',
    '.self_attn.out_proj.weight',
    '.self_attn.o_proj.weight',
    '.mlp.gate_proj.weight',
    '.mlp.up_proj.weight',
    '.mlp.down_proj.weight',
)


def load_spark_backbone(spark_path, *, spark_dtype: torch.dtype, pretrained: bool = True):
    """Load `XHToken/Spark-X2.6-4B` (BF16) or `XHToken/Spark-X2.6-4B-FP8` (compressed-tensors E4M3).

    Strips inference-only `quantization_config` in-memory so `Spark2_5ForCausalLM` builds
    trainable `nn.Linear` layers without requiring external `compressed-tensors`, and streams
    exact channelwise FP8 (`w_fp8 * weight_scale`) dequantization into `spark_dtype` master weights.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    spark_dir = Path(spark_path)
    cfg = AutoConfig.from_pretrained(spark_dir, trust_remote_code=True, local_files_only=True)
    quant_cfg = getattr(cfg, 'quantization_config', None) or cfg.__dict__.get('quantization_config')
    if hasattr(cfg, 'quantization_config'):
        try:
            delattr(cfg, 'quantization_config')
        except AttributeError:
            pass
    cfg.__dict__.pop('quantization_config', None)

    if not pretrained:
        return AutoModelForCausalLM.from_config(
            cfg,
            trust_remote_code=True,
            attn_implementation='eager',
        ).to(dtype=spark_dtype)

    single_sf = spark_dir / 'model.safetensors'
    is_fp8_safetensors = False
    if single_sf.is_file():
        from safetensors import safe_open
        with safe_open(str(single_sf), framework='pt', device='cpu') as f:
            keys = list(f.keys())
            is_fp8_safetensors = any(k.endswith('.weight_scale') for k in keys) or (quant_cfg is not None)

    if not is_fp8_safetensors:
        return AutoModelForCausalLM.from_pretrained(
            spark_dir,
            config=cfg,
            trust_remote_code=True,
            local_files_only=True,
            dtype=spark_dtype,
            attn_implementation='eager',
        )

    from safetensors import safe_open

    spark = AutoModelForCausalLM.from_config(
        cfg,
        trust_remote_code=True,
        attn_implementation='eager',
    ).to(dtype=spark_dtype)

    param_map = dict(spark.named_parameters())
    buf_map = dict(spark.named_buffers())
    loaded_keys: set[str] = set()
    fp8_dequant_count = 0

    with safe_open(str(single_sf), framework='pt', device='cpu') as f:
        sf_keys = set(f.keys())
        for key in sorted(sf_keys):
            if key.endswith(('.weight_scale', '.input_scale', '.weight_res_q4', '.weight_res_scale')):
                continue
            target = param_map.get(key)
            if target is None:
                target = buf_map.get(key)
            if target is None:
                raise KeyError(f'Unexpected tensor {key!r} in FP8 Spark checkpoint {single_sf}')
            raw_t = f.get_tensor(key)
            scale_key = key + '_scale'
            if raw_t.dtype == torch.float8_e4m3fn or scale_key in sf_keys:
                if scale_key not in sf_keys:
                    raise KeyError(f'Missing channelwise FP8 scale {scale_key!r} for {key!r}')
                w_scale = f.get_tensor(scale_key)
                deq = raw_t.to(torch.float32) * w_scale.to(torch.float32)
                if (key + '_res_q4') in sf_keys and (key + '_res_scale') in sf_keys:
                    deq = deq + _unpack_fp8_sub_lsb_residual(
                        f.get_tensor(key + '_res_q4'),
                        f.get_tensor(key + '_res_scale'),
                        int(deq.shape[0]),
                        int(deq.shape[1]),
                    )
                target.data.copy_(deq.to(dtype=target.dtype))
                fp8_dequant_count += 1
            else:
                target.data.copy_(raw_t.to(dtype=target.dtype))
            loaded_keys.add(key)

    expected_keys = set(param_map.keys())
    missing = expected_keys - loaded_keys - {'lm_head.weight'}
    if missing:
        raise RuntimeError(f'Missing parameters when loading FP8 Spark checkpoint: {sorted(missing)[:8]}')
    if hasattr(spark, 'tie_weights'):
        spark.tie_weights()
    spark._fp8_checkpoint_source = {
        'format': 'compressed-tensors-fp8-e4m3-channel',
        'dequantized_linear_tensors': fp8_dequant_count,
        'loaded_tensors': len(loaded_keys),
    }
    return spark


def _write_safetensors_dict(tensors: dict[str, torch.Tensor], dest: Path, metadata: dict[str, str] | None = None) -> None:
    try:
        from safetensors.torch import save_file
        save_file(tensors, str(dest), metadata=metadata)
        return
    except ModuleNotFoundError:
        pass
    import json, struct
    dtype_map = {
        torch.float32: 'F32',
        torch.bfloat16: 'BF16',
        torch.float16: 'F16',
        torch.float8_e4m3fn: 'F8_E4M3',
        torch.int64: 'I64',
        torch.int32: 'I32',
        torch.uint8: 'U8',
        torch.bool: 'BOOL',
    }
    header: dict[str, Any] = {'__metadata__': metadata} if metadata else {}
    blobs: list[bytes] = []
    offset = 0
    for name, t in tensors.items():
        cpu_t = t.detach().contiguous().cpu()
        dt = dtype_map[cpu_t.dtype]
        raw = bytes(cpu_t.view(torch.uint8).reshape(-1).numpy())
        end = offset + len(raw)
        header[name] = {'dtype': dt, 'shape': list(cpu_t.shape), 'data_offsets': [offset, end]}
        blobs.append(raw)
        offset = end
    hdr_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')
    hdr_bytes += b' ' * ((8 - (len(hdr_bytes) % 8)) % 8)
    with dest.open('wb') as f:
        f.write(struct.pack('<Q', len(hdr_bytes)))
        f.write(hdr_bytes)
        for b in blobs:
            f.write(b)


def _read_safetensors_dict(src: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors import safe_open
        out: dict[str, torch.Tensor] = {}
        with safe_open(str(src), framework='pt', device='cpu') as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
        return out
    except ModuleNotFoundError:
        pass
    import json, struct
    rev_dtype = {
        'F32': torch.float32,
        'BF16': torch.bfloat16,
        'F16': torch.float16,
        'F8_E4M3': torch.float8_e4m3fn,
        'I64': torch.int64,
        'I32': torch.int32,
        'U8': torch.uint8,
        'BOOL': torch.bool,
    }
    raw_all = src.read_bytes()
    (hlen,) = struct.unpack('<Q', raw_all[:8])
    hdr = json.loads(raw_all[8:8 + hlen].decode('utf-8'))
    body = memoryview(raw_all)[8 + hlen:]
    out = {}
    for name, meta in hdr.items():
        if name == '__metadata__':
            continue
        s, e = meta['data_offsets']
        dt = rev_dtype[meta['dtype']]
        chunk = bytes(body[s:e])
        t = torch.frombuffer(bytearray(chunk), dtype=dt).reshape(meta['shape']) if len(chunk) > 0 else torch.empty(meta['shape'], dtype=dt)
        out[name] = t
    return out


def _pack_fp8_sub_lsb_residual(res_f32: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack sub-LSB `F8_E4M3` quantization residual into symmetric group-128 signed INT4 (`U8` + `BF16` scale).

    Preserves full-rank APOLLO fine-tuning deltas (`~1e-4 .. 2e-3`) that otherwise snap back onto the 3-bit-mantissa
    `E4M3` grid (`~4.18e-4` relative error — ~10x finer than BF16 — while saving ~1.70 GiB vs BF16).
    """
    out_f, in_f = int(res_f32.shape[0]), int(res_f32.shape[1])
    padded_in = ((in_f + group_size - 1) // group_size) * group_size
    res_pad = F.pad(res_f32, (0, padded_in - in_f), value=0.0) if padded_in > in_f else res_f32
    num_groups = padded_in // group_size
    res_g = res_pad.reshape(out_f, num_groups, group_size)
    res_scale = (res_g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 7.0).to(torch.bfloat16)
    rq = (res_g / res_scale.to(torch.float32)).round().clamp(-7, 7).to(torch.int8)
    u = (rq + 8).to(torch.uint8).reshape(out_f, padded_in // 2, 2)
    packed = (u[..., 0] | (u[..., 1] << 4)).contiguous()
    return packed, res_scale.squeeze(-1).contiguous()


def _unpack_fp8_sub_lsb_residual(
    packed: torch.Tensor,
    res_scale: torch.Tensor,
    out_f: int,
    in_f: int,
    group_size: int = 128,
) -> torch.Tensor:
    """Reconstruct float32 residual `[out_f, in_f]` from packed `U8` nibbles + `BF16` group scales."""
    num_groups = int(res_scale.shape[1])
    lo = (packed & 0x0F).to(torch.int8) - 8
    hi = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    rq = torch.stack([lo, hi], dim=-1).reshape(out_f, num_groups, group_size)
    res_pad = (rq.to(torch.float32) * res_scale.unsqueeze(-1).to(torch.float32)).reshape(out_f, num_groups * group_size)
    return res_pad[:, :in_f]


def export_spark_native_fp8_safetensors(model: nn.Module, path: str | Path) -> dict[str, Any]:
    """Export `SparkNative` with Spark linear projections quantized to `F8_E4M3` + `weight_scale` + packed `INT4` sub-LSB residual."""
    fp8_max = 448.0
    out_tensors: dict[str, torch.Tensor] = {}
    seen_ptrs: set[tuple[int, tuple[int, ...], tuple[int, ...]]] = set()
    quantized_count = 0

    for name, tensor in model.state_dict().items():
        sig = (tensor.untyped_storage().data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))
        if sig in seen_ptrs:
            continue
        seen_ptrs.add(sig)
        t_cpu = tensor.detach().contiguous().cpu()
        if (
            name.startswith('spark.model.layers.')
            and any(name.endswith(sfx) for sfx in _FP8_TARGET_SUFFIXES)
            and t_cpu.ndim == 2
        ):
            w_f32 = t_cpu.to(torch.float32)
            scale = (w_f32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / fp8_max).to(torch.bfloat16)
            w_fp8 = (w_f32 / scale.to(torch.float32)).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
            res = w_f32 - w_fp8.to(torch.float32) * scale.to(torch.float32)
            res_q4, res_scale = _pack_fp8_sub_lsb_residual(res)
            out_tensors[name] = w_fp8.contiguous()
            out_tensors[name + '_scale'] = scale.contiguous()
            out_tensors[name + '_res_q4'] = res_q4
            out_tensors[name + '_res_scale'] = res_scale
            quantized_count += 1
        else:
            out_tensors[name] = t_cpu

    dest = Path(path)
    _write_safetensors_dict(
        out_tensors,
        dest,
        metadata={'format': 'pt', 'quant_method': 'compressed-tensors-fp8-e4m3-channel+q4g128-residual'},
    )
    return {'quantized_linear_tensors': quantized_count, 'residual_format': 'q4g128', 'bytes': dest.stat().st_size}


def load_spark_native_safetensors(model: nn.Module, path: str | Path) -> None:
    """Load standard BF16 `model.safetensors` or channelwise `F8_E4M3` (+ optional `q4g128` sub-LSB residual) `model.safetensors`."""
    sf_path = Path(path)
    aux_suffixes = ('.weight_scale', '.input_scale', '.weight_res_q4', '.weight_res_scale')
    try:
        from safetensors import safe_open
        from safetensors.torch import load_model
        with safe_open(str(sf_path), framework='pt', device='cpu') as f:
            sf_keys = set(f.keys())
            has_fp8 = any(k.endswith('.weight_scale') for k in sf_keys)
            if not has_fp8:
                strict = not hasattr(getattr(model, 'spark', None), 'tie_weights')
                load_model(model, str(sf_path), strict=strict, device='cpu')
                if hasattr(getattr(model, 'spark', None), 'tie_weights'):
                    model.spark.tie_weights()
                return

            param_map = dict(model.named_parameters())
            buf_map = dict(model.named_buffers())
            loaded: set[str] = set()
            for key in sorted(sf_keys):
                if key.endswith(aux_suffixes):
                    continue
                target = param_map.get(key)
                if target is None:
                    target = buf_map.get(key)
                if target is None:
                    continue
                raw_t = f.get_tensor(key)
                scale_key = key + '_scale'
                if raw_t.dtype == torch.float8_e4m3fn or scale_key in sf_keys:
                    w_scale = f.get_tensor(scale_key)
                    deq = raw_t.to(torch.float32) * w_scale.to(torch.float32)
                    if (key + '_res_q4') in sf_keys and (key + '_res_scale') in sf_keys:
                        deq = deq + _unpack_fp8_sub_lsb_residual(
                            f.get_tensor(key + '_res_q4'),
                            f.get_tensor(key + '_res_scale'),
                            int(deq.shape[0]),
                            int(deq.shape[1]),
                        )
                    target.data.copy_(deq.to(dtype=target.dtype, device=target.device))
                else:
                    target.data.copy_(raw_t.to(dtype=target.dtype, device=target.device))
                loaded.add(key)
    except ModuleNotFoundError:
        tensors = _read_safetensors_dict(sf_path)
        sf_keys = set(tensors.keys())
        param_map = dict(model.named_parameters())
        buf_map = dict(model.named_buffers())
        loaded = set()
        for key in sorted(sf_keys):
            if key.endswith(aux_suffixes):
                continue
            target = param_map.get(key)
            if target is None:
                target = buf_map.get(key)
            if target is None:
                continue
            raw_t = tensors[key]
            scale_key = key + '_scale'
            if raw_t.dtype == torch.float8_e4m3fn or scale_key in sf_keys:
                w_scale = tensors[scale_key]
                deq = raw_t.to(torch.float32) * w_scale.to(torch.float32)
                if (key + '_res_q4') in sf_keys and (key + '_res_scale') in sf_keys:
                    deq = deq + _unpack_fp8_sub_lsb_residual(
                        tensors[key + '_res_q4'],
                        tensors[key + '_res_scale'],
                        int(deq.shape[0]),
                        int(deq.shape[1]),
                    )
                target.data.copy_(deq.to(dtype=target.dtype, device=target.device))
            else:
                target.data.copy_(raw_t.to(dtype=target.dtype, device=target.device))
            loaded.add(key)

    if hasattr(getattr(model, 'spark', None), 'tie_weights'):
        model.spark.tie_weights()
    missing = set(param_map.keys()) - loaded - {'spark.lm_head.weight'}
    if missing:
        raise RuntimeError(f'Missing parameters when loading SparkNative safetensors: {sorted(missing)[:8]}')


class SparkNative(nn.Module):
    main_input_name='input_ids'
    supports_gradient_checkpointing=True
    def __init__(self,paths,policy,device='cuda',pretrained=True,components=None):
        super().__init__()
        self.policy=policy; self.paths={k:str(v) for k,v in (paths or {}).items()}; self.pretrained=pretrained
        self.modalities=policy.get('modalities','omni')
        default_backbone_dtype='float32' if self.modalities=='omni' or str(device).startswith('cpu') else 'bfloat16'
        spark_dtype=getattr(torch,policy.get('spark_dtype',default_backbone_dtype))
        vision_dtype=getattr(torch,policy.get('vision_dtype',default_backbone_dtype))

        if components is not None:
            self.spark=components['spark'].to(dtype=spark_dtype)
            install_spark_sdpa(self.spark)
            self.tokenizer=components['tokenizer']
            if getattr(self.tokenizer,'pad_token_id',None) is None and getattr(self.tokenizer,'eos_token',None) is not None:
                self.tokenizer.pad_token=self.tokenizer.eos_token
            self.chat = ChatFormat(self.tokenizer)
            self.config=self.spark.config; self.config.use_cache=False; self.generation_config=getattr(self.spark,'generation_config',None)
            self.vision=components['vision'].to(dtype=vision_dtype)
            if hasattr(getattr(self.vision,'vision_model',None),'head'): del self.vision.vision_model.head
            if hasattr(self.vision,'vision_model'): self.vision.vision_model.use_head=False
            self.image_processor=components['image_processor']
        else:
            from transformers import (AutoTokenizer,Siglip2VisionModel,Siglip2VisionConfig,AutoImageProcessor)
            self.spark=load_spark_backbone(paths['spark'],spark_dtype=spark_dtype,pretrained=pretrained)
            install_spark_sdpa(self.spark)
            try:
                self.tokenizer=AutoTokenizer.from_pretrained(paths['spark'],trust_remote_code=True,local_files_only=True,fix_mistral_regex=True)
            except TypeError:
                self.tokenizer=AutoTokenizer.from_pretrained(paths['spark'],trust_remote_code=True,local_files_only=True)
            if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token=self.tokenizer.eos_token
            self.chat = ChatFormat(self.tokenizer)
            self.config=self.spark.config; self.config.use_cache=False; self.generation_config=self.spark.generation_config

            self.vision=(Siglip2VisionModel.from_pretrained(paths['vision'],local_files_only=True,dtype=vision_dtype) if pretrained else Siglip2VisionModel(Siglip2VisionConfig.from_pretrained(paths['vision'],local_files_only=True)).to(dtype=vision_dtype))
            if hasattr(self.vision.vision_model,'head'): del self.vision.vision_model.head
            self.vision.vision_model.use_head=False
            self.image_processor=AutoImageProcessor.from_pretrained(paths['vision'],local_files_only=True)

        h=self.config.hidden_size; self.hidden=h
        self.vision_projector=nn.Sequential(nn.LayerNorm(self.vision.config.hidden_size*4),nn.Linear(self.vision.config.hidden_size*4,h),nn.GELU(),nn.Linear(h,h))
        # Zero-bias transparency projector: exact 0.0 on opaque RGB images, distinct on RGBA cutouts/translucency.
        self.alpha_projector=nn.Sequential(nn.Linear(16,h,bias=False),nn.GELU(),nn.Linear(h,h,bias=False))
        nn.init.normal_(self.alpha_projector[2].weight, mean=0.0, std=0.02)
        self.spatial_clock=nn.Sequential(nn.Linear(12,h),nn.GELU(),nn.Linear(h,h))
        self.camera_pose_encoder=nn.Sequential(nn.Linear(8,h,bias=False),nn.GELU(),nn.Linear(h,h,bias=False))
        nn.init.zeros_(self.camera_pose_encoder[2].weight)
        self.visual_kind_embedding=nn.Embedding(len(VISUAL_KINDS),h)
        nn.init.normal_(self.visual_kind_embedding.weight, mean=0.0, std=0.01)
        self.view_id_embedding=nn.Embedding(len(VIEW_IDS),h)
        nn.init.normal_(self.view_id_embedding.weight, mean=0.0, std=0.01)
        self.modality=nn.Embedding(5,h)
        self.visual_token_drop=nn.Dropout(float(policy.get('visual_token_dropout',0.0)))

        # Optional 0 / 3 / 6 Flamingo-style zero-init gated visual cross-attention blocks.
        self._active_visual_context: torch.Tensor | None = None
        self._xattn_hooks: list[Any] = []
        n_xattn = int(policy.get('visual_cross_attn_blocks', 0))
        if n_xattn not in (0, 3, 6):
            raise ValueError(f'visual_cross_attn_blocks must be 0, 3, or 6; got {n_xattn}')
        self.visual_cross_attn = nn.ModuleList()
        self.visual_cross_attn_layer_indices: list[int] = []
        if n_xattn > 0:
            self.install_visual_cross_attention(n_xattn)

        if self.modalities=='omni':
            from transformers import WhisperModel,WhisperConfig,WhisperFeatureExtractor,ClapFeatureExtractor
            whisper=(WhisperModel.from_pretrained(paths['speech_in'],local_files_only=True,dtype=torch.float32) if pretrained else WhisperModel(WhisperConfig.from_pretrained(paths['speech_in'],local_files_only=True)))
            self.speech_in=whisper.encoder; del whisper
            self.speech_processor=WhisperFeatureExtractor.from_pretrained(paths['speech_in'],local_files_only=True)
            self.sound_in=self.load_sound_in(paths['sound_in'],pretrained)
            self.sound_processor=ClapFeatureExtractor.from_pretrained(paths['sound_in'],local_files_only=True)
            self.motion=self.load_motion(paths['motion_source'],paths['motion'] if pretrained else None)
            self.motion_projector=Resampler(self.motion.embed_dim,h,policy['motion_tokens'])
            self.speech_projector=Resampler(self.speech_in.config.d_model,h,policy['audio_tokens'])
            self.sound_projector=Resampler(self.sound_in.config.hidden_size,h,policy['audio_tokens'])
            self.audio_clock=nn.Sequential(nn.Linear(4,h),nn.GELU(),nn.Linear(h,h))
            from .outputs import NativeSpeechOutput,NativeSoundOutput
            self.speech_out=NativeSpeechOutput(paths['speech_out'],h,policy,pretrained=pretrained)
            self.sound_out=NativeSoundOutput(paths['sound_out'],h,policy,pretrained=pretrained)
            self.to(device=device,dtype=torch.float32)
        else:
            self.speech_in=None; self.speech_processor=None
            self.sound_in=None; self.sound_processor=None
            self.motion=None; self.motion_projector=None
            self.speech_projector=None; self.sound_projector=None
            self.audio_clock=None; self.speech_out=None; self.sound_out=None
            # Move Spark + SigLIP2 preserving their configured dtypes; keep connectors/adapters in float32.
            self.spark.to(device=device,dtype=spark_dtype)
            if hasattr(self.spark,'tie_weights'): self.spark.tie_weights()
            self.vision.to(device=device,dtype=vision_dtype)
            for conn in self._connector_modules():
                conn.to(device=device,dtype=torch.float32)

        self.freeze_infrastructure()
        self.gradient_checkpointing_enable()

    def _connector_modules(self) -> list[nn.Module]:
        mods: list[nn.Module] = []
        for attr in (
            'vision_projector',
            'alpha_projector',
            'spatial_clock',
            'camera_pose_encoder',
            'visual_kind_embedding',
            'view_id_embedding',
            'modality',
            'visual_cross_attn',
        ):
            m = getattr(self, attr, None)
            if isinstance(m, nn.Module):
                mods.append(m)
        return mods

    def install_visual_cross_attention(self, num_blocks: int) -> list[int]:
        """Attach `num_blocks` (`3` or `6`) zero-init gated visual cross-attention adapters across Spark layers."""
        for handle in getattr(self, '_xattn_hooks', []):
            handle.remove()
        self._xattn_hooks = []
        layers = getattr(getattr(self.spark, 'model', None), 'layers', None)
        if layers is not None:
            for lyr in layers:
                if hasattr(lyr, '_spark_xattn_idx'):
                    delattr(lyr, '_spark_xattn_idx')
        if num_blocks == 0:
            self.visual_cross_attn = nn.ModuleList()
            self.visual_cross_attn_layer_indices = []
            return []
        if layers is None or len(layers) == 0:
            raise ValueError('Cannot attach gated visual cross-attention: spark.model.layers not found')
        n_layers = len(layers)
        step = max(1, n_layers // num_blocks)
        chosen = [min(n_layers - 1, (i + 1) * step - 1) for i in range(num_blocks)]
        # Deduplicate if tiny test model has fewer layers than num_blocks.
        unique_indices: list[int] = []
        for idx in chosen:
            if idx not in unique_indices:
                unique_indices.append(idx)
        heads = max(1, min(8, self.hidden // max(1, min(64, self.hidden // 2))))
        while self.hidden % heads != 0 and heads > 1:
            heads -= 1
        adapter_drop = float(self.policy.get('adapter_dropout', 0.0))
        self.visual_cross_attn = nn.ModuleList(
            [GatedVisualCrossAttentionBlock(self.hidden, num_heads=heads, dropout=adapter_drop) for _ in unique_indices]
        ).to(device=self.device, dtype=torch.float32)
        self.visual_cross_attn_layer_indices = unique_indices

        for block_idx, layer_idx in enumerate(unique_indices):
            block = self.visual_cross_attn[block_idx]
            layers[layer_idx]._spark_xattn_idx = block_idx

            def _make_hook(b: GatedVisualCrossAttentionBlock):
                def _hook(_module, _inputs, output):
                    if _SUPPRESS_XATTN_HOOK.get():
                        return output
                    vis_ctx = getattr(self, '_active_visual_context', None)
                    if vis_ctx is None:
                        vis_ctx = _ACTIVE_VISUAL_CONTEXT.get()
                    if vis_ctx is None:
                        return output
                    if isinstance(output, tuple):
                        hs = b(output[0], vis_ctx)
                        return (hs, *output[1:])
                    return b(output, vis_ctx)

                return _hook

            self._xattn_hooks.append(layers[layer_idx].register_forward_hook(_make_hook(block)))
        return unique_indices

    @contextmanager
    def _visual_cross_attn_scope(self, visual_tokens: torch.Tensor | None):
        prev = getattr(self, '_active_visual_context', None)
        active = (
            visual_tokens
            if (visual_tokens is not None and len(getattr(self, 'visual_cross_attn', ())) > 0)
            else None
        )
        self._active_visual_context = active
        ctx_tok = _ACTIVE_VISUAL_CONTEXT.set(active)
        try:
            yield
        finally:
            _ACTIVE_VISUAL_CONTEXT.reset(ctx_tok)
            self._active_visual_context = prev

    def _make_vis_aware_checkpoint_fn(self, base_ckpt):
        def _vis_aware_ckpt(fn, *args, **kwargs):
            if _INSIDE_VIS_CKPT_WRAPPER.get():
                return base_ckpt(fn, *args, **kwargs)
            vis_ctx = getattr(self, '_active_visual_context', None)
            if vis_ctx is None:
                vis_ctx = _ACTIVE_VISUAL_CONTEXT.get()
            bound_layer = getattr(fn, '__self__', None)
            xattn_idx = getattr(bound_layer, '_spark_xattn_idx', None)
            xattn_blk = (
                self.visual_cross_attn[xattn_idx]
                if (isinstance(xattn_idx, int) and 0 <= xattn_idx < len(self.visual_cross_attn))
                else None
            )
            guard_tok = _INSIDE_VIS_CKPT_WRAPPER.set(True)
            try:
                if vis_ctx is None:
                    def _no_vis_fn(*a, **kw):
                        sup_tok = _SUPPRESS_XATTN_HOOK.set(True)
                        try:
                            with self._visual_cross_attn_scope(None):
                                return fn(*a, **kw)
                        finally:
                            _SUPPRESS_XATTN_HOOK.reset(sup_tok)
                    return base_ckpt(_no_vis_fn, *args, **kwargs)

                if xattn_blk is not None:
                    # Disjoint sequential checkpoint segments: decoder_layer and xattn_blk
                    # never materialize autograd activations simultaneously during backward!
                    def _layer_only_fn(*a, **kw):
                        sup_tok = _SUPPRESS_XATTN_HOOK.set(True)
                        try:
                            with self._visual_cross_attn_scope(None):
                                return fn(*a, **kw)
                        finally:
                            _SUPPRESS_XATTN_HOOK.reset(sup_tok)

                    layer_out = base_ckpt(_layer_only_fn, *args, **kwargs)
                    if isinstance(layer_out, tuple):
                        hs_adapted = base_ckpt(xattn_blk, layer_out[0], vis_ctx)
                        return (hs_adapted, *layer_out[1:])
                    return base_ckpt(xattn_blk, layer_out, vis_ctx)

                def _with_vis_fn(captured_vis, *a, **kw):
                    with self._visual_cross_attn_scope(captured_vis):
                        return fn(*a, **kw)

                return base_ckpt(_with_vis_fn, vis_ctx, *args, **kwargs)
            finally:
                _INSIDE_VIS_CKPT_WRAPPER.reset(guard_tok)

        _vis_aware_ckpt._spark_vis_ctx_wrapped = True
        return _vis_aware_ckpt

    def _wrap_spark_gradient_checkpointing(self, opts: dict[str, Any]) -> None:
        spark = getattr(self, 'spark', None)
        if spark is None:
            return
        fallback_ckpt = functools.partial(torch.utils.checkpoint.checkpoint, **opts)
        for mod in spark.modules():
            fn = getattr(mod, '_gradient_checkpointing_func', None)
            if callable(fn):
                if not getattr(fn, '_spark_vis_ctx_wrapped', False):
                    mod._gradient_checkpointing_func = self._make_vis_aware_checkpoint_fn(fn)
            elif getattr(mod, 'gradient_checkpointing', False) is True:
                mod._gradient_checkpointing_func = self._make_vis_aware_checkpoint_fn(fallback_ckpt)

        spark_mod = sys.modules.get(type(spark).__module__)
        orig_mod_ckpt = getattr(spark_mod, 'checkpoint', None) if spark_mod is not None else None
        if callable(orig_mod_ckpt) and not getattr(orig_mod_ckpt, '_spark_vis_ctx_wrapped', False):
            def _mod_vis_ckpt(fn, *args, **kwargs):
                if _INSIDE_VIS_CKPT_WRAPPER.get():
                    return orig_mod_ckpt(fn, *args, **kwargs)
                vis_ctx = _ACTIVE_VISUAL_CONTEXT.get()
                if vis_ctx is None:
                    return orig_mod_ckpt(fn, *args, **kwargs)
                guard_tok = _INSIDE_VIS_CKPT_WRAPPER.set(True)
                try:
                    def _with_vis_mod(captured_vis, *a, **kw):
                        prev_tok = _ACTIVE_VISUAL_CONTEXT.set(captured_vis)
                        try:
                            return fn(*a, **kw)
                        finally:
                            _ACTIVE_VISUAL_CONTEXT.reset(prev_tok)

                    return orig_mod_ckpt(_with_vis_mod, vis_ctx, *args, **kwargs)
                finally:
                    _INSIDE_VIS_CKPT_WRAPPER.reset(guard_tok)

            _mod_vis_ckpt._spark_vis_ctx_wrapped = True
            spark_mod.checkpoint = _mod_vis_ckpt

    @staticmethod
    def load_sound_in(path,pretrained):
        from transformers import ClapModel,ClapAudioModel,ClapAudioConfig
        if pretrained:
            return ClapModel.from_pretrained(path,local_files_only=True,dtype=torch.float32).audio_model.audio_encoder
        return ClapAudioModel(ClapAudioConfig.from_pretrained(path,local_files_only=True)).audio_encoder
    @staticmethod
    def load_motion(source,checkpoint):
        sys.path.insert(0,str(source))
        from app.vjepa_2_1.models.vision_transformer import vit_base
        motion=vit_base(patch_size=16,img_size=(384,384),num_frames=64,tubelet_size=2,use_sdpa=True,use_SiLU=False,wide_SiLU=True,uniform_power=False,use_rope=True,img_temporal_dim_size=1,interpolate_rope=True)
        if checkpoint is not None:
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            if 'ema_encoder' not in state: raise ValueError('Wrong V-JEPA checkpoint key')
            clean={k.replace('module.','').replace('backbone.',''):v for k,v in state['ema_encoder'].items()}
            motion.load_state_dict(clean,strict=True); del state,clean
        motion.patch_embed_img=None; motion.img_temporal_dim_size=None
        if hasattr(motion,'img_mod_embed'): del motion.img_mod_embed
        if hasattr(motion,'use_activation_checkpointing'): motion.use_activation_checkpointing=True
        return motion
    @property
    def device(self): return next(self.spark.parameters()).device
    @property
    def embed_dtype(self): return self.get_input_embeddings().weight.dtype
    def get_input_embeddings(self): return self.spark.get_input_embeddings()
    def get_output_embeddings(self): return self.spark.get_output_embeddings()
    def gradient_checkpointing_enable(self,gradient_checkpointing_kwargs=None):
        opts=gradient_checkpointing_kwargs or {'use_reentrant':False}
        candidates=[
            getattr(self,'spark',None),
            getattr(self,'vision',None),
            getattr(self,'speech_in',None),
            getattr(self,'sound_in',None),
            getattr(getattr(self,'speech_out',None),'talker',None),
        ]
        for module in candidates:
            if module is not None and getattr(module,'supports_gradient_checkpointing',False):
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs=opts)
        self._wrap_spark_gradient_checkpointing(opts)
        sound_out=getattr(self,'sound_out',None)
        if sound_out is not None and hasattr(getattr(sound_out,'dit',None),'enable_gradient_checkpointing'):
            sound_out.dit.enable_gradient_checkpointing()
    def freeze_infrastructure(self):
        speech_out=getattr(self,'speech_out',None)
        if speech_out is not None: speech_out.freeze_codec()
        sound_out=getattr(self,'sound_out',None)
        if sound_out is not None:
            if hasattr(sound_out,'vae') and sound_out.vae is not None:
                sound_out.vae.requires_grad_(False)
            if hasattr(sound_out,'freeze_fixed_conditioning'):
                sound_out.freeze_fixed_conditioning()
    def train(self,mode=True):
        super().train(mode)
        vision=getattr(self,'vision',None)
        if mode and vision is not None and not any(p.requires_grad for p in vision.parameters()):
            vision.eval()
        speech_out=getattr(self,'speech_out',None)
        if speech_out is not None and getattr(speech_out,'codec_model',None) is not None:
            speech_out.codec_model.eval()
        sound_out=getattr(self,'sound_out',None)
        if sound_out is not None and getattr(sound_out,'vae',None) is not None:
            sound_out.vae.eval()
        return self

    def _unfreeze_late_vision_layers(self, last_n_layers: int) -> None:
        vision = getattr(self, 'vision', None)
        if vision is None or last_n_layers <= 0:
            return
        vision.requires_grad_(False)
        vmod = getattr(vision, 'vision_model', vision)
        enc_layers = getattr(getattr(vmod, 'encoder', None), 'layers', None)
        if enc_layers is not None and len(enc_layers) > 0:
            for layer in enc_layers[-int(last_n_layers):]:
                layer.requires_grad_(True)
        post_ln = getattr(vmod, 'post_layernorm', None)
        if post_ln is not None:
            post_ln.requires_grad_(True)

    def set_stage(
        self,
        alignment_only=False,
        *,
        stage_mode: str | None = None,
        unfreeze_vision_last_n_layers: int = 0,
    ):
        """Configure parameter trainability across staged connector, gated cross-attn, vision, and full-Spark runs."""
        mode = stage_mode or ('alignment_only' if alignment_only else 'full_spark')
        self.requires_grad_(True)
        self.freeze_infrastructure()

        if mode in ('alignment_only', 'frozen_spark_connector', 'frozen_spark_cross_attn'):
            for name in ('spark', 'vision', 'motion', 'speech_in', 'sound_in'):
                mod = getattr(self, name, None)
                if mod is not None:
                    mod.requires_grad_(False)
            speech_out = getattr(self, 'speech_out', None)
            if speech_out is not None and getattr(speech_out, 'talker', None) is not None:
                speech_out.talker.requires_grad_(False)
            sound_out = getattr(self, 'sound_out', None)
            if sound_out is not None and getattr(sound_out, 'dit', None) is not None:
                sound_out.dit.requires_grad_(False)
            if mode == 'frozen_spark_connector' and getattr(self, 'visual_cross_attn', None) is not None:
                self.visual_cross_attn.requires_grad_(False)
            if unfreeze_vision_last_n_layers > 0:
                self._unfreeze_late_vision_layers(unfreeze_vision_last_n_layers)
            return

        if mode == 'vision_late_unfreeze':
            if getattr(self, 'spark', None) is not None:
                self.spark.requires_grad_(False)
            if getattr(self, 'vision', None) is not None:
                self._unfreeze_late_vision_layers(max(1, int(unfreeze_vision_last_n_layers or 2)))
            return

        if mode == 'full_spark':
            # In vision_text 24GB profile, keep SigLIP2 frozen unless explicitly unfrozen via unfreeze_vision_last_n_layers.
            if getattr(self, 'modalities', 'omni') == 'vision_text' and getattr(self, 'vision', None) is not None:
                if unfreeze_vision_last_n_layers > 0:
                    self._unfreeze_late_vision_layers(unfreeze_vision_last_n_layers)
                elif not getattr(self, 'policy', {}).get('unfreeze_vision_in_full_spark', False):
                    self.vision.requires_grad_(False)
            if any(not p.requires_grad for p in self.spark.parameters()):
                raise AssertionError('Full Spark is not trainable')
            return

        raise ValueError(f'Unknown stage_mode {mode!r}')

    def frozen_parameter_hashes(self) -> dict[str, str]:
        """Compute exact raw-byte SHA256 hashes for all frozen (`not requires_grad`) parameters."""
        hashes: dict[str, str] = {}
        seen: set[int] = set()
        for name, p in self.named_parameters():
            if p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            raw = p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            hashes[name] = hashlib.sha256(raw).hexdigest()
        return hashes

    def named_groups(self):
        connector_prefixes = {
            'vision_projector',
            'alpha_projector',
            'spatial_clock',
            'camera_pose_encoder',
            'visual_kind_embedding',
            'view_id_embedding',
            'motion_projector',
            'speech_projector',
            'sound_projector',
            'modality',
            'audio_clock',
            'visual_cross_attn',
        }
        for name,p in self.named_parameters():
            prefix=name.split('.')[0]
            if prefix in connector_prefixes: group='connectors'
            elif prefix=='speech_out': group='speech_output'
            elif prefix=='sound_out': group='sound_output'
            else: group=prefix
            yield group,name,p

    def tile_views(self,image,video,*,alpha=None,is_pixel_art_atlas=False):
        from PIL import Image
        w,h=image.size; views=[(image,alpha,(0.,0.,1.,1.),self.policy['thumbnail_patches'])]
        if not video and self.policy['detail_tiles']:
            if self.policy['detail_tiles'] not in (2,4): raise ValueError('Qualified tile count is 0/2/4')
            nx,ny=(2,2) if self.policy['detail_tiles']==4 else ((2,1) if w>=h else (1,2))
            for j in range(ny):
                for i in range(nx):
                    a,b,c,d=round(i*w/nx),round(j*h/ny),round((i+1)*w/nx),round((j+1)*h/ny)
                    crop_im=image.crop((a,b,c,d))
                    crop_alpha=alpha[b:d,a:c] if alpha is not None else None
                    if is_pixel_art_atlas and min(c-a,d-b)<64:
                        # Crisp nearest-neighbor magnification on narrow/tiny pixel-art UV islands prevents bilinear blur!
                        scale=min(4,max(1,64//max(1,min(c-a,d-b))))
                        if scale>1:
                            crop_im=crop_im.resize(((c-a)*scale,(d-b)*scale),resample=Image.Resampling.NEAREST)
                            if crop_alpha is not None:
                                crop_alpha=np.repeat(np.repeat(crop_alpha,scale,axis=0),scale,axis=1)
                    views.append((crop_im,crop_alpha,(a/w,b/h,c/w,d/h),self.policy['tile_patches']))
        return views

    def image_tokens(self,images,times,capture,alphas=None,visual_units=None):
        if not images:
            return None
        # Only genuine temporal_frame sequences disable detail tiles; multiview renders,
        # UV atlases, and before/after state pairs preserve full detail tile crops!
        if visual_units:
            is_temporal=all(u.get('visual_kind')=='temporal_frame' for u in visual_units)
        else:
            is_temporal=len(images)>1 and any(float(t)!=0.0 for t in (times or []))

        units=[]
        for fi,raw_image in enumerate(images):
            rgb_image,derived_alpha=extract_image_rgb_and_alpha(raw_image)
            explicit_alpha=alphas[fi] if alphas is not None and fi<len(alphas) else None
            alpha_arr=explicit_alpha if explicit_alpha is not None else derived_alpha
            u_meta=visual_units[fi] if visual_units and fi<len(visual_units) else {
                'visual_kind':'temporal_frame' if is_temporal else ('multiview_render' if len(images)>1 else 'independent_image'),
                'visual_kind_index':VISUAL_KIND_TO_INDEX['temporal_frame' if is_temporal else ('multiview_render' if len(images)>1 else 'independent_image')],
                'view_id':'frame' if is_temporal else 'single',
                'view_id_index':VIEW_ID_TO_INDEX['frame' if is_temporal else 'single'],
                'camera_right':[1.0,0.0,0.0],
                'camera_up':[0.0,1.0,0.0],
                'ortho_scale':1.0,
                'timestamp_s':float(times[fi]) if times and fi<len(times) else 0.0,
                'has_timestamp':bool(is_temporal or (times and fi<len(times) and float(times[fi])!=0.0)),
                'capture_s':None if not capture or fi>=len(capture) or capture[fi] is None else float(capture[fi]),
                'is_pixel_art_atlas':False,
            }
            for vi,(view_rgb,view_alpha,box,patches) in enumerate(
                self.tile_views(rgb_image,is_temporal,alpha=alpha_arr,is_pixel_art_atlas=bool(u_meta.get('is_pixel_art_atlas',False)))
            ):
                batch=self.image_processor(images=view_rgb,return_tensors='pt',max_num_patches=patches)
                units.append({'frame':fi,'view':vi,'image':view_rgb,'alpha':view_alpha,'meta':u_meta,'box':box,'patches':patches,'batch':batch})

        def count(unit):
            hh,ww=(int(x) for x in unit['batch']['spatial_shapes'][0])
            return ((hh+1)//2)*((ww+1)//2)

        while sum(count(u) for u in units)>self.policy['max_visual_tokens']:
            candidates=[u for u in units if u['patches']>64]
            if not candidates: raise ValueError('Actual visual grids cannot fit the qualified token budget')
            u=max(candidates,key=count); u['patches']=max(64,u['patches']//2)
            u['batch']=self.image_processor(images=u['image'],return_tensors='pt',max_num_patches=u['patches'])

        outputs=[None]*len(units); groups={}
        for index,unit in enumerate(units): groups.setdefault(unit['patches'],[]).append((index,unit))
        vis_dtype=next(self.vision.parameters()).dtype
        proj_dtype=self.vision_projector[0].weight.dtype

        for _,items in sorted(groups.items()):
            batch={
                k:torch.cat([u['batch'][k] for _,u in items],dim=0).to(
                    self.device,
                    dtype=vis_dtype if k=='pixel_values' else None,
                )
                for k in ('pixel_values','pixel_attention_mask','spatial_shapes')
            }
            features=self.vision(**batch).last_hidden_state
            for position,(index,unit) in enumerate(items):
                fi,vi,box,u_meta=unit['frame'],unit['view'],unit['box'],unit['meta']
                hh,ww=(int(x) for x in batch['spatial_shapes'][position])
                tokens,_=spatial_downshuffle(features[position:position+1],hh,ww)
                tokens=self.vision_projector(tokens.to(proj_dtype))

                alpha_feat=compute_patch_alpha_transparency_features(unit['alpha'],hh,ww,self.device).to(proj_dtype)
                tokens=tokens+self.alpha_projector(alpha_feat)

                gh,gw=(hh+1)//2,(ww+1)//2
                yy,xx=torch.meshgrid(torch.arange(gh,device=self.device),torch.arange(gw,device=self.device),indexing='ij')
                xx,yy=xx.flatten().float(),yy.flatten().float(); xmin,ymin,xmax,ymax=box
                x0=2*xx/ww; x1=torch.clamp(2*xx+2,max=ww)/ww
                y0=2*yy/hh; y1=torch.clamp(2*yy+2,max=hh)/hh
                x=xmin+(x0+x1)/2*(xmax-xmin); y=ymin+(y0+y1)/2*(ymax-ymin)
                bw=(x1-x0)*(xmax-xmin); bh=(y1-y0)*(ymax-ymin)

                has_ts=bool(u_meta.get('has_timestamp',False))
                time_val=float(u_meta.get('timestamp_s',0.0)) if has_ts else 0.0
                cap=u_meta.get('capture_s')
                clock=[time_val/60,math.sin(time_val) if has_ts else 0.,math.cos(time_val) if has_ts else 1.,0. if cap is None else float(cap)/3600,0. if cap is None else 1.]
                coord=torch.stack([x,y,bw,bh,torch.full_like(x,float(fi)),*([torch.full_like(x,v) for v in clock]),torch.full_like(x,float(vi)),torch.full_like(x,float(vi==0))],dim=-1).to(proj_dtype)

                cam_vec=torch.tensor(
                    [
                        *u_meta.get('camera_right',[1.,0.,0.])[:3],
                        *u_meta.get('camera_up',[0.,1.,0.])[:3],
                        float(u_meta.get('ortho_scale',1.0)),
                        float(bool(u_meta.get('is_pixel_art_atlas',False))),
                    ],
                    device=self.device,
                    dtype=proj_dtype,
                ).view(1,1,8).expand(1,tokens.shape[1],8)

                vkind_idx=int(u_meta.get('visual_kind_index',VISUAL_KIND_TO_INDEX.get(u_meta.get('visual_kind','independent_image'),0)))
                vid_idx=int(u_meta.get('view_id_index',VIEW_ID_TO_INDEX.get(u_meta.get('view_id','single'),0)))

                token_out=(
                    tokens
                    + self.spatial_clock(coord[None])
                    + self.camera_pose_encoder(cam_vec)
                    + self.visual_kind_embedding.weight[vkind_idx].to(proj_dtype)
                    + self.view_id_embedding.weight[vid_idx].to(proj_dtype)
                    + self.modality.weight[0].to(proj_dtype)
                )
                token_out=self.visual_token_drop(token_out)
                outputs[index]=token_out.to(self.embed_dtype)
        return torch.cat(outputs,dim=1) if outputs else None

    def motion_tokens(self,frames):
        if not frames or getattr(self,'motion',None) is None: return None
        n=self.policy['motion_frames']; ids=np.linspace(0,len(frames)-1,n).round().astype(int)
        arrays=[np.asarray(frames[i].resize((self.policy['motion_size'],)*2),dtype=np.float32)/255 for i in ids]
        x=torch.from_numpy(np.stack(arrays)).permute(3,0,1,2)[None].to(self.device)
        mean=x.new_tensor([.485,.456,.406]).view(1,3,1,1,1); std=x.new_tensor([.229,.224,.225]).view(1,3,1,1,1)
        out=self.motion((x-mean)/std,training=True)
        if isinstance(out,(list,tuple)): out=out[-1]
        if out.shape[-1]!=self.motion.embed_dim: out=out.reshape(*out.shape[:-1],-1,self.motion.embed_dim).mean(-2)
        return (self.motion_projector(out)+self.modality.weight[1]).to(self.embed_dtype)

    def audio_tokens(self,audio,fam):
        if audio is None: return None
        if getattr(self,'speech_in',None) is None:
            raise ValueError('Audio modalities are deferred in vision_text mode')
        wav,sr=audio; wav=wav.mean(axis=0)
        if fam in ('speech_in','audiovisual'):
            mono=audio_resample(wav,sr,16000); parts=[]; clocks=[]
            for start in range(0,len(mono),30*16000):
                piece=mono[start:start+30*16000]
                f=self.speech_processor(piece,sampling_rate=16000,return_tensors='pt').input_features.to(self.device)
                out=self.speech_in(input_features=f).last_hidden_state
                out=out[:,:max(1,math.ceil(len(piece)/16000*50))]; parts.append(out)
                clocks.append(start/16000+torch.arange(out.shape[1],device=self.device)/50)
            times=torch.cat(clocks)
            position=self.audio_clock(torch.stack([times/30,torch.sin(times),torch.cos(times),torch.ones_like(times)],dim=-1))[None]
            return (self.speech_projector(torch.cat(parts,dim=1),position=position)+self.modality.weight[2]).to(self.embed_dtype)
        mono=audio_resample(wav,sr,48000); parts=[]; clocks=[]
        for start in range(0,len(mono),10*48000):
            piece=mono[start:start+10*48000]
            f=self.sound_processor(piece,sampling_rate=48000,return_tensors='pt',truncation='rand_trunc',padding='repeatpad')
            out=self.sound_in(**{k:v.to(self.device) for k,v in f.items()}).last_hidden_state
            if out.ndim!=4: raise ValueError('Expected CLAP frequency/time feature map')
            full_time=out.shape[-1]; valid=max(1,math.ceil(full_time*len(piece)/(10*48000)))
            out=out[...,:valid]; frequency=out.shape[-2]
            clocks.append((start/48000+torch.arange(valid,device=self.device)*10/full_time).repeat(frequency))
            parts.append(out.flatten(2).transpose(1,2))
        times=torch.cat(clocks)
        position=self.audio_clock(torch.stack([times/30,torch.sin(times),torch.cos(times),torch.ones_like(times)],dim=-1))[None]
        return (self.sound_projector(torch.cat(parts,dim=1),position=position)+self.modality.weight[3]).to(self.embed_dtype)

    def forward_text_only(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs):
        """Pure-text fast path calling `self.spark` directly with zero visual module overhead."""
        with self._visual_cross_attn_scope(None):
            return self.spark(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def prompt_ids_for_row(self, row: dict[str, Any], *, for_generation: bool = False) -> tuple[list[int], list[int]]:
        thinking = bool(row.get('enable_thinking', False))
        system = row.get('system_prompt')
        return self.chat.encode(
            row['prompt'],
            enable_thinking=thinking,
            system=system,
            for_generation=for_generation,
        )

    def prompt_parts(self,prompt,*,enable_thinking=False,system=None,for_generation=False):
        pre,post=self.chat.encode(prompt,enable_thinking=enable_thinking,system=system,for_generation=for_generation)
        return (torch.tensor(pre,device=self.device,dtype=torch.long)[None],
                torch.tensor(post,device=self.device,dtype=torch.long)[None])

    def _encode_visual_tokens_for_media(self, media: dict[str, Any]) -> torch.Tensor | None:
        img_fn = getattr(self, 'image_tokens', None)
        if img_fn is None:
            return None
        # Backward compatibility for mock lambdas in unit tests (`model.image_tokens = lambda *args: None`)
        try:
            return img_fn(
                media['images'],
                media['times'],
                media['capture_times'],
                media.get('alphas'),
                media.get('visual_units'),
            )
        except TypeError:
            return img_fn(media['images'], media['times'], media['capture_times'])

    def prefix_with_visual_context(self, row, media, *, for_generation: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        if row['_family'].endswith('_out') and any(media.get(k) is not None for k in ('audio','motion')):
            raise ValueError('Output target leaked into input encoder')
        pre,post=self.prompt_parts(
            row['prompt'],
            enable_thinking=bool(row.get('enable_thinking',False)),
            system=row.get('system_prompt'),
            for_generation=for_generation,
        )
        emb_dtype=self.embed_dtype
        parts=[self.get_input_embeddings()(pre)]
        vis_tokens = self._encode_visual_tokens_for_media(media) if media.get('images') else None
        mot_tokens = self.motion_tokens(media.get('motion')) if getattr(self, 'motion_tokens', None) else None
        aud_tokens = self.audio_tokens(media.get('audio'), row['_family']) if getattr(self, 'audio_tokens', None) else None
        for x in (vis_tokens, mot_tokens, aud_tokens):
            if x is not None:
                parts.append(x.to(emb_dtype))
        parts.append(self.get_input_embeddings()(post))
        return torch.cat(parts,dim=1), vis_tokens

    def prefix(self,row,media):
        embeds, _ = self.prefix_with_visual_context(row, media, for_generation=False)
        return embeds

    def hidden_states(self,embeds,visual_context=None):
        return self.hidden_batch(
            embeds,
            torch.ones(embeds.shape[:2],device=self.device,dtype=torch.long),
            visual_context=visual_context,
        )

    def hidden_batch(self,embeds,mask,visual_context=None):
        if embeds.shape[1]>self.policy['max_sequence']: raise ValueError('Input exceeds qualified context; no silent truncation')
        with self._visual_cross_attn_scope(visual_context):
            return self.spark.model(inputs_embeds=embeds,attention_mask=mask,use_cache=False,return_dict=True).last_hidden_state

    def _effective_supervised_row(self, row: dict[str, Any]) -> dict[str, Any]:
        want_think = bool(row.get('enable_thinking', False))
        has_think_body = str(row.get('answer', '')).lstrip().startswith('<think>')
        if want_think and not has_think_body:
            return {**row, 'enable_thinking': False}
        return row

    def text_loss(self,row,media):
        sup_row=self._effective_supervised_row(row)
        bot_close=self.chat.layouts[bool(sup_row.get('enable_thinking',False))]['bot_close']
        ids=self.tokenizer(sup_row['answer']+bot_close,add_special_tokens=False,return_tensors='pt').input_ids.to(self.device)
        if not has_multimodal_context(media):
            pre,post=self.prompt_ids_for_row(sup_row,for_generation=False)
            prefix_ids=torch.tensor(pre+post,device=self.device,dtype=torch.long)[None]
            if ids.shape[1]>self.policy['max_answer_tokens'] or prefix_ids.shape[1]+ids.shape[1]>self.policy['max_sequence']:
                raise ValueError('Supervision would be truncated; revise a versioned context policy')
            full_ids=torch.cat([prefix_ids,ids],dim=1)
            with self._visual_cross_attn_scope(None):
                hidden=self.spark.model(input_ids=full_ids,attention_mask=torch.ones_like(full_ids),use_cache=False,return_dict=True).last_hidden_state[:,prefix_ids.shape[1]-1:-1]
        else:
            prefix,vis_ctx=self.prefix_with_visual_context(sup_row,media,for_generation=False)
            if ids.shape[1]>self.policy['max_answer_tokens'] or prefix.shape[1]+ids.shape[1]>self.policy['max_sequence']:
                raise ValueError('Supervision would be truncated; revise a versioned context policy')
            embeds=torch.cat([prefix,self.get_input_embeddings()(ids)],dim=1)
            hidden=self.hidden_states(embeds,visual_context=vis_ctx)[:,prefix.shape[1]-1:-1]

        total=hidden.new_zeros((),dtype=torch.float32)
        ls=float(self.policy.get('label_smoothing',0.0)) if self.training else 0.0
        from torch.utils.checkpoint import checkpoint
        for start in range(0,ids.shape[1],128):
            hs_2d=hidden[:,start:start+128].reshape(-1,self.hidden); labels_1d=ids[:,start:start+128].reshape(-1)
            def ce(x,y): return F.cross_entropy(self.get_output_embeddings()(x).float().reshape(-1,self.config.vocab_size),y,reduction='sum',label_smoothing=ls)
            total=total+checkpoint(ce,hs_2d,labels_1d,use_reentrant=False)
        return total/ids.numel(),ids.numel()

    def text_batch_loss(self,examples):
        """One padded Spark forward for text-producing records; pure-text batches bypass all visual modules."""
        all_pure_text = all(not has_multimodal_context(media) for _, media in examples)
        has_active_xattn = len(getattr(self, 'visual_cross_attn', ())) > 0 and any(
            has_multimodal_context(media) and bool(media.get('images')) for _, media in examples
        )
        if has_active_xattn and len(examples) > 1:
            # Per-example visual context tokens differ across rows; evaluate cleanly per example.
            losses, units = [], []
            for row, media in examples:
                l_val, u_val = self.text_loss(row, media)
                losses.append(l_val)
                units.append(u_val)
            return losses, units

        if all_pure_text:
            prepared_ids=[]
            pad_id=int(self.tokenizer.pad_token_id or 0)
            for row,_ in examples:
                sup_row=self._effective_supervised_row(row)
                pre,post=self.prompt_ids_for_row(sup_row,for_generation=False)
                p_ids=torch.tensor(pre+post,device=self.device,dtype=torch.long)[None]
                bot_close=self.chat.layouts[bool(sup_row.get('enable_thinking',False))]['bot_close']
                a_ids=self.tokenizer(sup_row['answer']+bot_close,add_special_tokens=False,return_tensors='pt').input_ids.to(self.device)
                if a_ids.shape[1]>self.policy['max_answer_tokens'] or p_ids.shape[1]+a_ids.shape[1]>self.policy['max_sequence']:
                    raise ValueError('Supervision would be truncated; revise a versioned context policy')
                prepared_ids.append((p_ids,a_ids))
            width=max(p.shape[1]+a.shape[1] for p,a in prepared_ids)
            input_ids=torch.full((len(prepared_ids),width),pad_id,device=self.device,dtype=torch.long)
            mask=torch.zeros((len(prepared_ids),width),device=self.device,dtype=torch.long)
            for i,(p_ids,a_ids) in enumerate(prepared_ids):
                seq=torch.cat([p_ids,a_ids],dim=1)[0]
                input_ids[i,:seq.shape[0]]=seq
                mask[i,:seq.shape[0]]=1
            if width>self.policy['max_sequence']: raise ValueError('Input exceeds qualified context; no silent truncation')
            with self._visual_cross_attn_scope(None):
                hidden=self.spark.model(input_ids=input_ids,attention_mask=mask,use_cache=False,return_dict=True).last_hidden_state
            from torch.utils.checkpoint import checkpoint
            ls=float(self.policy.get('label_smoothing',0.0)) if self.training else 0.0
            losses=[]; units=[]
            for i,(p_ids,a_ids) in enumerate(prepared_ids):
                start=p_ids.shape[1]-1; count=a_ids.shape[1]
                hs=hidden[i:i+1,start:start+count]; total=hidden.new_zeros((),dtype=torch.float32)
                for offset in range(0,count,128):
                    hs_2d=hs[:,offset:offset+128].reshape(-1,self.hidden); lbl_1d=a_ids[:,offset:offset+128].reshape(-1)
                    def ce(x,y): return F.cross_entropy(self.get_output_embeddings()(x).float().reshape(-1,self.config.vocab_size),y,reduction='sum',label_smoothing=ls)
                    total=total+checkpoint(ce,hs_2d,lbl_1d,use_reentrant=False)
                losses.append(total/count); units.append(count)
            return losses,units

        prepared=[]
        single_vis_ctx=None
        for row,media in examples:
            sup_row=self._effective_supervised_row(row)
            prefix,vis_ctx=self.prefix_with_visual_context(sup_row,media,for_generation=False)
            if vis_ctx is not None: single_vis_ctx=vis_ctx
            bot_close=self.chat.layouts[bool(sup_row.get('enable_thinking',False))]['bot_close']
            ids=self.tokenizer(sup_row['answer']+bot_close,add_special_tokens=False,return_tensors='pt').input_ids.to(self.device)
            if ids.shape[1]>self.policy['max_answer_tokens'] or prefix.shape[1]+ids.shape[1]>self.policy['max_sequence']:
                raise ValueError('Supervision would be truncated; revise a versioned context policy')
            prepared.append((prefix,self.get_input_embeddings()(ids),ids))
        width=max(prefix.shape[1]+answer.shape[1] for prefix,answer,_ in prepared)
        embeds=torch.zeros(len(prepared),width,self.hidden,device=self.device,dtype=prepared[0][0].dtype)
        mask=torch.zeros(len(prepared),width,device=self.device,dtype=torch.long)
        for i,(prefix,answer,_) in enumerate(prepared):
            length=prefix.shape[1]+answer.shape[1]
            embeds[i,:length]=torch.cat([prefix,answer],dim=1)[0]
            mask[i,:length]=1
        hidden=self.hidden_batch(embeds,mask,visual_context=single_vis_ctx if len(prepared)==1 else None)
        from torch.utils.checkpoint import checkpoint
        ls=float(self.policy.get('label_smoothing',0.0)) if self.training else 0.0
        losses=[]; units=[]
        for i,(prefix,_,ids) in enumerate(prepared):
            start=prefix.shape[1]-1; count=ids.shape[1]
            hs=hidden[i:i+1,start:start+count]; total=hidden.new_zeros((),dtype=torch.float32)
            for offset in range(0,count,128):
                hs_2d=hs[:,offset:offset+128].reshape(-1,self.hidden); lbl_1d=ids[:,offset:offset+128].reshape(-1)
                def ce(x,y): return F.cross_entropy(self.get_output_embeddings()(x).float().reshape(-1,self.config.vocab_size),y,reduction='sum',label_smoothing=ls)
                total=total+checkpoint(ce,hs_2d,lbl_1d,use_reentrant=False)
            losses.append(total/count); units.append(count)
        return losses,units

    def forward(self,examples=None,**kwargs):
        if not examples or len(examples)>self.policy.get('microbatch_size',1):
            raise ValueError('Microbatch exceeds the qualified source-record bound')
        losses=[None]*len(examples); units=[0]*len(examples)
        text=[i for i,(row,_) in enumerate(examples) if not row['_family'].endswith('_out')]
        if text:
            values,counts=self.text_batch_loss([examples[i] for i in text])
            for i,value,count in zip(text,values,counts): losses[i],units[i]=value,count
        for i,(row,media) in enumerate(examples):
            if row['_family'].endswith('_out'): losses[i],units[i]=self.loss(row,media)
        per_example=torch.stack(losses)
        return {'loss':per_example.mean(),'per_example_loss':per_example,'supervised_units':sum(units)}

    def loss(self,row,media):
        fam=row['_family']
        if fam.endswith('_out'):
            hidden=self.hidden_states(self.prefix(row,media))
            if fam=='speech_out': return self.speech_out.loss(hidden,media['target_audio'],row)
            return self.sound_out.loss(hidden,media['target_audio'],row)
        return self.text_loss(row,media)

    @torch.no_grad()
    def generate_text_only_ids(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
    ) -> list[int]:
        """KV-cached pure-text decoding directly on `self.spark.model(input_ids=...)` with zero visual overhead."""
        generated: list[int] = []
        past = None
        cur_ids = input_ids
        length = int(input_ids.shape[1])
        max_seq = int(getattr(self, 'policy', {}).get('max_sequence', 8192))
        eos_id = self.tokenizer.eos_token_id
        with self._visual_cross_attn_scope(None):
            for _ in range(max_new_tokens):
                if length + len(generated) >= max_seq:
                    break
                output = self.spark.model(
                    input_ids=cur_ids,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = output.past_key_values
                logits = self.get_output_embeddings()(output.last_hidden_state[:, -1:])
                token = _sample_next_token(
                    logits,
                    generated_ids=generated,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )
                if eos_id is not None and token.item() == eos_id:
                    break
                generated.append(int(token.item()))
                cur_ids = token
        return generated

    @torch.no_grad()
    def answer(
        self,
        row,
        media,
        max_new_tokens=None,
        *,
        enable_thinking: bool | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
    ):
        if max_new_tokens is None:
            task = row.get('task', '')
            if str(task).startswith('synccraft_') or task in ('document_ocr', 'music_instrument_note_understanding'):
                max_new_tokens = 1024
            elif task in ('factual_caption', 'image_caption', 'video_caption', 'video_box_tracking', 'region_dense_video_caption'):
                max_new_tokens = 512
            else:
                max_new_tokens = 256

        effective_row = row if enable_thinking is None else {**row, 'enable_thinking': bool(enable_thinking)}

        # Pure-text fast path: call Spark directly via `input_ids` with zero visual-module execution!
        if not has_multimodal_context(media):
            if 'messages' in effective_row:
                thinking = bool(effective_row.get('enable_thinking', False))
                try:
                    rendered = self.tokenizer.apply_chat_template(
                        effective_row['messages'],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=thinking,
                    )
                except TypeError:
                    rendered = self.tokenizer.apply_chat_template(
                        effective_row['messages'],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                prompt_ids = self.tokenizer(rendered, add_special_tokens=False, return_tensors='pt').input_ids.to(self.device)
            else:
                # For non-thinking prompts without explicit `<think>` generation flag, keep exact layout compatibility.
                for_gen = bool(effective_row.get('enable_thinking', False))
                pre, post = self.prompt_ids_for_row(effective_row, for_generation=for_gen)
                prompt_ids = torch.tensor(pre + post, device=self.device, dtype=torch.long)[None]

            generated = self.generate_text_only_ids(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
            return self.chat.restore_thinking_prefix(
                decoded,
                self.chat.layouts[True],
                bool(effective_row.get('enable_thinking', False)),
            )

        embeds, vis_ctx = self.prefix_with_visual_context(
            effective_row,
            media,
            for_generation=True,
        )
        generated=[]; past=None; state=None; length=embeds.shape[1]
        with self._visual_cross_attn_scope(vis_ctx):
            for _ in range(max_new_tokens):
                if length+len(generated)>=self.policy['max_sequence']: break
                if past is None:
                    output=self.spark.model(inputs_embeds=embeds,use_cache=True,return_dict=True)
                else:
                    output=self.spark.model(inputs_embeds=state,past_key_values=past,use_cache=True,return_dict=True)
                past=output.past_key_values
                logits=self.get_output_embeddings()(output.last_hidden_state[:,-1:])
                token=_sample_next_token(
                    logits,
                    generated_ids=generated,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )
                if token.item()==self.tokenizer.eos_token_id: break
                generated.append(token.item()); state=self.get_input_embeddings()(token)
        decoded = self.tokenizer.decode(generated,skip_special_tokens=True)
        return self.chat.restore_thinking_prefix(
            decoded,
            self.chat.layouts[True],
            bool(effective_row.get('enable_thinking', False)),
        )

    @torch.no_grad()
    def generate_audio(self,row,media,seconds=None,steps=50):
        hidden=self.hidden_states(self.prefix(row,media))
        if row['_family']=='speech_out': return self.speech_out.generate(hidden,row)
        return self.sound_out.generate(hidden,seconds=seconds or 15,steps=steps)

    @torch.no_grad()
    def spoken_answer(self,row,media,max_new_tokens=128):
        if media.get('target_audio') is not None: raise ValueError('Inference cannot consume a target waveform')
        text=self.answer(row,media,max_new_tokens=max_new_tokens)
        answer_ids=self.tokenizer(text,add_special_tokens=False,return_tensors='pt').input_ids.to(self.device)
        context=torch.cat([self.prefix(row,media),self.get_input_embeddings()(answer_ids)],dim=1)
        hidden=self.hidden_states(context)
        request={'language':row['language'],'utterance':text,'prompt':'Speak this answer:\n'+text}
        audio,sr=self.speech_out.generate(hidden,request)
        return {'text':text,'audio':audio,'sample_rate':sr,'paired_audio_dialogue_training_verified':False}
