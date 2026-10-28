"""Infrastructure checks for Spark Colab. No installs, training, or teacher calls.

Google authorization belongs to the interactive notebook. All network writes are
new, uniquely named files below the explicitly selected project folder.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
import uuid

GIB = 1024 ** 3
PROJECT_STORAGE_CAP_BYTES = 2_000_000_000_000


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_digests(path):
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    length = 0
    with Path(path).open('rb') as handle:
        while block := handle.read(8 * 1024 * 1024):
            sha.update(block)
            md5.update(block)
            length += len(block)
    return {'bytes': length, 'sha256': sha.hexdigest(), 'md5': md5.hexdigest()}


def verify_local(path, expected_sha256, expected_bytes=None):
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
        raise ValueError('A lowercase 64-character SHA-256 is required')
    observed = file_digests(path)
    if observed['sha256'] != expected_sha256:
        raise ValueError('Downloaded payload SHA-256 mismatch')
    if expected_bytes is not None and observed['bytes'] != int(expected_bytes):
        raise ValueError('Downloaded payload length mismatch')
    return observed


def validate_drive_id(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]{10,}', str(value)):
        raise ValueError('Use a Drive file/folder ID, not a path or URL')
    return str(value)


def infrastructure_report(local_root='/content/spark_mm'):
    root = Path(local_root)
    root.mkdir(parents=True, exist_ok=True)
    report = {'schema_version': 1, 'created_utc': utc_now(),
              'session_id': str(uuid.uuid4()), 'scope': 'infrastructure_only',
              'training_ready': False, 'python': platform.python_version(),
              'platform': platform.platform(), 'local_root': str(root),
              'runtime_boot_id': (Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                                  if Path('/proc/sys/kernel/random/boot_id').exists() else None),
              'local_free_bytes': shutil.disk_usage(root).free,
              'packages': {}, 'gpu_check_passed': False,
              'tiny_bf16_backward_passed': False,
              'spark_forward_backward_tested': False,
              'fresh_session_model_restore_tested': False}
    for name in ['torch', 'torchvision', 'torchaudio', 'transformers', 'unsloth',
                 'unsloth-zoo', 'trl', 'bitsandbytes', 'triton', 'google-api-python-client']:
        try:
            report['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][name] = None
    meminfo = Path('/proc/meminfo')
    if meminfo.exists():
        wanted = {'MemTotal', 'MemAvailable'}
        report['host_memory_bytes'] = {
            line.split(':')[0]: int(line.split()[1]) * 1024
            for line in meminfo.read_text().splitlines()
            if line.split(':')[0] in wanted}
    if shutil.which('nvidia-smi'):
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                 '--format=csv,noheader'], text=True, capture_output=True,
                                timeout=20, check=False)
        report['nvidia_smi'] = result.stdout.strip() if result.returncode == 0 else 'query_failed'
    try:
        import torch
        report['torch_runtime'] = {'cuda': torch.version.cuda, 'hip': torch.version.hip,
                                   'cuda_available': torch.cuda.is_available()}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            report['gpu'] = {'name': props.name, 'total_bytes': props.total_memory,
                             'capability': list(torch.cuda.get_device_capability(0)),
                             'bf16_supported': bool(torch.cuda.is_bf16_supported())}
            is_target = ('RTX PRO 6000' in props.name.upper() and
                         'BLACKWELL' in props.name.upper() and props.total_memory >= 90 * GIB)
            report['gpu_check_passed'] = bool(is_target and torch.version.cuda and
                                              not torch.version.hip and torch.cuda.is_bf16_supported())
            if report['gpu_check_passed']:
                x = torch.randn(128, 128, device='cuda', dtype=torch.float32, requires_grad=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = (x @ x.T).float().square().mean()
                loss.backward()
                torch.cuda.synchronize()
                report['tiny_bf16_backward_passed'] = bool(torch.isfinite(loss).item() and
                                                          torch.isfinite(x.grad).all().item())
                report['gpu_check_passed'] = report['tiny_bf16_backward_passed']
                del x, loss
    except Exception as exc:
        # Exception text can contain paths/configuration; record only its type.
        report['torch_probe_error_type'] = type(exc).__name__
        report['gpu_check_passed'] = False
    return report


def drive_metadata(service, file_id):
    return service.files().get(fileId=validate_drive_id(file_id), supportsAllDrives=True,
        fields='id,name,mimeType,size,md5Checksum,trashed,capabilities(canAddChildren)').execute()


def check_drive_access(service, folder_id):
    folder = drive_metadata(service, folder_id)
    if folder.get('mimeType') != 'application/vnd.google-apps.folder' or folder.get('trashed'):
        raise ValueError('The selected project ID is not an active Drive folder')
    if not folder.get('capabilities', {}).get('canAddChildren', False):
        raise PermissionError('The authenticated account cannot write to this project folder')
    quota = service.about().get(fields='storageQuota').execute().get('storageQuota', {})
    limit = int(quota['limit']) if quota.get('limit') is not None else None
    usage = int(quota['usage']) if quota.get('usage') is not None else None
    free = max(0, limit - usage) if limit is not None and usage is not None else None
    return {'folder_id': folder_id, 'folder_name': folder['name'], 'can_write': True,
            'account_quota_limit_bytes': limit, 'account_usage_bytes': usage,
            'account_free_bytes': free, 'project_cap_bytes': PROJECT_STORAGE_CAP_BYTES,
            'quota_known': free is not None,
            'initial_500GB_free_check': None if free is None else free >= 500_000_000_000}


def upload_verified(service, local_path, folder_id, remote_name=None):
    from googleapiclient.http import MediaFileUpload
    local_path = Path(local_path)
    before = file_digests(local_path)
    # No retry loop: an uncertain create/upload is not blindly duplicated.
    file_id = service.files().generateIds(count=1, space='drive', type='files').execute()['ids'][0]
    request = service.files().create(
        body={'id': file_id, 'name': remote_name or local_path.name,
              'parents': [validate_drive_id(folder_id)],
              'appProperties': {'project': 'spark-multimodal', 'sha256': before['sha256']}},
        media_body=MediaFileUpload(str(local_path), mimetype='application/octet-stream',
                                  chunksize=8 * 1024 * 1024, resumable=True),
        fields='id,name,size,md5Checksum', supportsAllDrives=True)
    response = None
    started = time.monotonic()
    while response is None:
        _, response = request.next_chunk(num_retries=0)
    remote = drive_metadata(service, file_id)
    if int(remote.get('size', -1)) != before['bytes'] or remote.get('md5Checksum') != before['md5']:
        raise ValueError('Drive upload did not pass size and checksum verification')
    return {'file_id': file_id, 'name': remote['name'], **before,
            'upload_seconds': time.monotonic() - started,
            'remote_size_md5_verified': True}


def download_verified(service, file_id, destination, expected_sha256):
    from googleapiclient.http import MediaIoBaseDownload
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
        raise ValueError('A lowercase 64-character SHA-256 is required before downloading')
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Refusing to overwrite an existing local download')
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = drive_metadata(service, file_id)
    if metadata.get('trashed') or metadata.get('size') is None:
        raise ValueError('Expected a complete, ordinary binary Drive file')
    temporary = destination.with_name(destination.name + '.partial-' + uuid.uuid4().hex)
    started = time.monotonic()
    with temporary.open('xb') as handle:
        downloader = MediaIoBaseDownload(handle,
            service.files().get_media(fileId=validate_drive_id(file_id), supportsAllDrives=True),
            chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk(num_retries=0)
    observed = verify_local(temporary, expected_sha256, metadata['size'])
    if observed['md5'] != metadata.get('md5Checksum'):
        raise ValueError('Drive download MD5 mismatch')
    os.replace(temporary, destination)
    return {'file_id': file_id, **observed, 'download_seconds': time.monotonic() - started,
            'sha256_verified': True}


def drive_roundtrip_probe(service, folder_id, local_root, size_mib=64):
    if not isinstance(size_mib, int) or not 1 <= size_mib <= 1024:
        raise ValueError('Probe size must be 1..1024 MiB')
    root = Path(local_root)
    root.mkdir(parents=True, exist_ok=True)
    identity = uuid.uuid4().hex
    source = root / f'drive-probe-{identity}.bin'
    with source.open('xb') as handle:
        for _ in range(size_mib):
            handle.write(os.urandom(1024 * 1024))
    uploaded = upload_verified(service, source, folder_id)
    downloaded = download_verified(service, uploaded['file_id'],
        root / f'drive-probe-{identity}.download.bin', uploaded['sha256'])
    return {'upload': uploaded, 'download': downloaded,
            'same_session_roundtrip_passed': True, 'fresh_session_restore_tested': False,
            'scope': 'random infrastructure bytes, not model state or checkpoint resume'}


def save_receipt(service, folder_id, local_root, report):
    root = Path(local_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'preflight-receipt-{uuid.uuid4().hex}.json'
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return upload_verified(service, path, folder_id)
