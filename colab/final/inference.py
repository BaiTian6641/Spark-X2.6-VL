"""Offline loading and generation helpers for Spark-X2.6-VL (`vision_text` and `omni`); no external API."""
from __future__ import annotations
from pathlib import Path
from typing import Any

def load(directory,device='cuda'):
    from .common import read,sha
    from .engine import reload_paths
    from .model import SparkNative,load_spark_native_safetensors
    directory=Path(directory)
    release=read(directory/'MODEL_RELEASE.json')
    for f in release['files']:
        path=(directory/f['path']).resolve()
        if not path.is_relative_to(directory.resolve()) or path.is_symlink() or path.stat().st_size!=f['bytes'] or sha(path)!=f['sha256']:
            raise ValueError('Export content differs from its release manifest')
    config=read(directory/'config.json')
    model=SparkNative(reload_paths(directory,config=config),config,device=device,pretrained=False)
    load_spark_native_safetensors(model,directory/'model.safetensors')
    model.eval()
    return model

def empty_media()->dict[str,Any]:
    return {
        'images':[],
        'alphas':[],
        'visual_kind':None,
        'visual_units':[],
        'times':[],
        'capture_times':None,
        'audio':None,
        'target_audio':None,
        'motion':None,
        'motion_times':None,
    }

def visual_media(
    images:list[Any],
    *,
    visual_kind:str='independent_image',
    visual_units:list[dict[str,Any]]|None=None,
    times:list[float]|None=None,
    capture_times:list[float|None]|None=None,
)->dict[str,Any]:
    """Build structured RGBA-preserving visual media payload without ever conflating multiviews with video."""
    from .data import extract_image_rgb_and_alpha,normalize_visual_units
    alphas=[]
    kept=[]
    for im in images:
        _,alpha=extract_image_rgb_and_alpha(im)
        kept.append(im)
        alphas.append(alpha)
    resolved_times=list(times) if times is not None else [0.0]*len(kept)
    row_stub={'visual_kind':visual_kind,'visual_units':visual_units or [],'frames':[f'img_{i}' for i in range(len(kept))],'media_times_s':resolved_times,'capture_times_s':capture_times}
    return {
        'images':kept,
        'alphas':alphas,
        'visual_kind':visual_kind if kept else None,
        'visual_units':normalize_visual_units(row_stub,len(kept),resolved_times,capture_times),
        'times':resolved_times,
        'capture_times':capture_times,
        'audio':None,
        'target_audio':None,
        'motion':None,
        'motion_times':None,
    }

def generate(
    model,
    prompt:str,
    *,
    media:dict[str,Any]|None=None,
    enable_thinking:bool=False,
    max_new_tokens:int=512,
    do_sample:bool=False,
    temperature:float=1.0,
    top_p:float=1.0,
    top_k:int|None=None,
    repetition_penalty:float=1.0,
    task:str='text_replay',
    language:str='en',
    system_prompt:str|None=None,
)->str:
    """Unified inference entrypoint routing pure-text calls directly to Spark and visual calls through connectors."""
    effective_media=media or empty_media()
    fam='image' if effective_media.get('images') else 'text'
    row={
        '_family':fam,
        'language':language,
        'task':task,
        'prompt':prompt,
        'enable_thinking':enable_thinking,
    }
    if system_prompt is not None:
        row['system_prompt']=system_prompt
    model.eval()
    return model.answer(
        row,
        effective_media,
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
