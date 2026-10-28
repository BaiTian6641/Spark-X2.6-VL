"""Bound local checkpoint retention to two slots, with one verified fallback."""
from pathlib import Path
import shutil
from .common import read, emit
from .hf_checkpoints import receipt_path


def acknowledged(directory, binding):
    """Only the upload/restore verifier writes this deletion acknowledgement."""
    directory = Path(directory)
    receipt = receipt_path(directory)
    if directory.is_symlink() or not receipt.is_file() or not (directory/'BINDING.json').is_file():
        return False
    value = read(receipt)
    return (value.get('verified') is True and value.get('binding') == binding
            and read(directory/'BINDING.json') == binding)


def prune(root, binding, keep, *, incoming=None):
    """After verify: keep newest. Before restore/write: make only one free slot.

    Unknown or uncommitted directories stop the operation; they are never
    removed. A partially downloaded incoming restore may be retried in place.
    """
    base = Path(root).resolve()/'checkpoints'
    base.mkdir(parents=True, exist_ok=True)
    keep = {Path(p).resolve() for p in keep if p is not None}
    incoming = Path(incoming).resolve() if incoming is not None else None
    if any(p.parent != base for p in keep) or (incoming is not None and incoming.parent != base):
        raise ValueError('Checkpoint retention may only manage direct campaign checkpoint directories')
    dirs = [p for p in base.iterdir() if p.is_dir() or p.is_symlink()]
    if incoming is not None and not any(p.resolve() in keep for p in dirs):
        fallback = [p for p in dirs if p.resolve()!=incoming and acknowledged(p,binding)]
        if fallback:
            keep.add(max(fallback,key=lambda p:p.stat().st_mtime).resolve())
    # Validate the entire deletion set before performing any mutation.
    removing = [p for p in dirs if p.resolve() not in keep and p.resolve() != incoming]
    if any(not acknowledged(p, binding) for p in removing):
        raise RuntimeError('Unverified or foreign local checkpoint occupies a slot; preserve it and reconcile its upload')
    for old in removing:
        size = sum(p.stat().st_size for p in old.rglob('*') if p.is_file())
        shutil.rmtree(old)
        emit('local_checkpoint_purged', checkpoint=old.name, reclaimed_bytes=size,
             verified_remote_copy_retained=True)
    remaining = [p for p in base.iterdir() if p.is_dir()]
    if len(remaining) + (1 if incoming is not None and not incoming.exists() else 0) > 2:
        raise RuntimeError('Two local checkpoint slots would be exceeded')


def rebind_assets(model, directory):
    """Keep future snapshots independent of the old directory being reclaimed."""
    directory = Path(directory)
    # Preprocessors/codecs are read again when a later snapshot is assembled.
    # They must follow the retained snapshot, including after a restore.
    for key in ('spark', 'vision', 'speech_in', 'sound_in', 'speech_out', 'sound_out'):
        if key in model.paths:
            model.paths[key] = directory/'reload-assets'/key
