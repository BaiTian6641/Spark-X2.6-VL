"""Offline loading of the custom native package; no model/teacher API needed."""
from pathlib import Path
import sys

def load(directory,device='cuda'):
    from .common import read,sha
    from .engine import reload_paths
    from .model import SparkNative
    from safetensors.torch import load_model
    directory=Path(directory)
    release=read(directory/'MODEL_RELEASE.json')
    for f in release['files']:
        path=(directory/f['path']).resolve()
        if not path.is_relative_to(directory.resolve()) or path.is_symlink() or path.stat().st_size!=f['bytes'] or sha(path)!=f['sha256']: raise ValueError('Export content differs from its release manifest')
    model=SparkNative(reload_paths(directory),read(directory/'config.json'),device=device,pretrained=False)
    load_model(model,str(directory/'model.safetensors'),strict=True,device='cpu'); model.eval()
    return model

def empty_media(): return {'images':[],'times':[],'capture_times':None,'audio':None,'target_audio':None,'motion':None,'motion_times':None}
