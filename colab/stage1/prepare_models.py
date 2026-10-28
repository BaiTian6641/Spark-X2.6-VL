"""Download only original pinned pretrained assets; never workstation checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from .common import now, sha256, write_json


def verify_asset(path, expected):
    path = Path(path)
    if path.stat().st_size != expected['bytes']:
        raise ValueError('Pinned asset length mismatch: ' + path.name)
    digest = sha256(path)
    if expected.get('lfs_sha256'):
        if digest != expected['lfs_sha256']:
            raise ValueError('Pinned LFS content hash mismatch: ' + path.name)
    else:
        payload = path.read_bytes()
        identity = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
        if identity != expected['git_blob_sha1']:
            raise ValueError('Pinned Git blob identity mismatch: ' + path.name)
    return {'path': path.name, 'bytes': expected['bytes'], 'sha256': digest}


def prepare(project):
    from huggingface_hub import hf_hub_download
    project = Path(project).resolve()
    sources = json.loads((project / 'colab/stage1/model_sources.json').read_text())['sources']
    if sum(item['bytes'] for item in sources) > 11 * 1024**3:
        raise ValueError('Unexpected original-model download budget')
    if shutil.disk_usage(project).free < 100 * 1024**3:
        raise RuntimeError('Insufficient local space for original assets and two qualification checkpoints')
    results = []
    for source in sources:
        directory = project / ('models/Spark-X2.6-4B' if source['key'] == 'spark'
                               else 'models/SigLIP2-base-256/upstream')
        directory.mkdir(parents=True, exist_ok=True)
        verified = []
        for expected in source['files']:
            name = expected['path']
            if '/' in name or '\\' in name or name in {'.', '..'}:
                raise ValueError('Model manifest must contain top-level filenames only')
            downloaded = hf_hub_download(repo_id=source['repository'], revision=source['revision'],
                filename=name, local_dir=directory, token=False)
            verified.append(verify_asset(downloaded, expected))
            print(json.dumps({'event': 'original_asset_verified', 'model': source['key'],
                              'file': name, 'bytes': expected['bytes']}), flush=True)
        receipt = {'repository': source['repository'], 'revision': source['revision'],
                   'downloaded_utc': now(), 'files': verified,
                   'original_pretrained_weights': True, 'workstation_checkpoint_used': False}
        if source['key'] == 'spark':
            write_json(project / 'evidence/Spark-X2.6-4B-source.json', receipt)
        else:
            write_json(project / 'evidence/colab-siglip-upstream.json', receipt)
        results.append(receipt)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.project)
