"""Immutable qualification checkpoint uploads, with a commit receipt written last.

No shared latest pointer or automatic deletion. This is a single-operator
qualification store, not the future concurrent campaign controller.
"""
import json
from pathlib import Path
import shutil
import time
import uuid

from .common import now, safe_relative, sha256, write_json
from colab.preflight import file_digests, drive_metadata, download_verified, validate_drive_id

MAX_CHECKPOINT_BYTES = 40 * 1024**3
PROJECT_CAP_BYTES = 2_000_000_000_000
REQUIRED = {'model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json',
            'multimodal_config.json','experiment_sources.json','training_selection.json',
            'training_state_probe.json','parameter_samples_sha256.json'}


def project_usage(service, root_id):
    """Count project files recursively; this single-operator store has no reservations."""
    pending, seen, total = [validate_drive_id(root_id)], set(), 0
    while pending:
        folder = pending.pop()
        if folder in seen:
            raise ValueError('Unexpected repeated folder in project tree')
        seen.add(folder)
        token = None
        while True:
            page = service.files().list(q=f"'{folder}' in parents and trashed = false",
                pageSize=1000, pageToken=token,
                fields='nextPageToken,files(id,mimeType,size)').execute()
            for item in page.get('files', []):
                if item['mimeType'] == 'application/vnd.google-apps.folder':
                    pending.append(item['id'])
                else:
                    total += int(item.get('size', 0))
            token = page.get('nextPageToken')
            if not token:
                break
    return total


def snapshot_manifest(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    records = []
    for path in sorted(checkpoint.rglob('*')):
        if path.is_symlink():
            raise ValueError('Checkpoint symlinks are not accepted')
        if not path.is_file():
            continue
        name = path.relative_to(checkpoint).as_posix()
        safe_relative(name)
        if path.suffix == '.env' or any(part.lower() in {'secrets','credentials'} for part in path.parts):
            raise ValueError('Unexpected secret path inside qualification checkpoint')
        records.append({'path': name, **file_digests(path)})
    names = {record['path'] for record in records}
    total = sum(record['bytes'] for record in records)
    if not REQUIRED <= names or not 0 < total <= MAX_CHECKPOINT_BYTES:
        raise ValueError('Incomplete or oversized qualification checkpoint')
    return {'schema_version': 1, 'scope': 'new_colab_qualification_checkpoint',
            'total_bytes': total, 'files': records}


def upload_file(service, local_path, parent_id, state_path, remote_name):
    """Known file ID and private local upload state make same-VM retries safe."""
    from googleapiclient.http import MediaFileUpload
    local_path, state_path = Path(local_path), Path(state_path)
    local = file_digests(local_path)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['local'] != local or state['parent_id'] != parent_id:
            raise ValueError('Upload input changed after an upload intent was created')
        try:
            remote = drive_metadata(service, state['file_id'])
        except Exception as exc:
            if getattr(getattr(exc, 'resp', None), 'status', None) != 404:
                raise RuntimeError('Could not reconcile upload status: ' + type(exc).__name__) from None
            remote = None
        if remote and int(remote.get('size', -1)) == local['bytes'] and remote.get('md5Checksum') == local['md5']:
            return {'file_id': state['file_id'], **local, 'remote_size_md5_verified': True}
        if state.get('resumable_uri'):
            # The client requests server offset reconciliation before sending bytes.
            resume_uri = state['resumable_uri']
        else:
            raise RuntimeError('Uncertain upload without a saved resumable session; reconcile this file ID before retrying')
    else:
        file_id = service.files().generateIds(count=1, space='drive', type='files').execute()['ids'][0]
        state = {'file_id': file_id, 'parent_id': parent_id, 'local': local, 'resumable_uri': None}
        write_json(state_path, state)
        state_path.chmod(0o600)
        resume_uri = None
    request = service.files().create(body={'id': state['file_id'], 'name': remote_name,
        'parents': [validate_drive_id(parent_id)],
        'appProperties': {'project':'spark-multimodal','sha256':local['sha256']}},
        media_body=MediaFileUpload(str(local_path), mimetype='application/octet-stream',
                                  chunksize=16 * 1024**2, resumable=True),
        fields='id,size,md5Checksum', supportsAllDrives=True)
    if resume_uri:
        request.resumable_uri = resume_uri
        request._in_error_state = True
    response = None
    last_report = -1
    try:
        while response is None:
            progress, response = request.next_chunk(num_retries=3)
            state['resumable_uri'] = request.resumable_uri
            write_json(state_path, state)
            state_path.chmod(0o600)
            if progress is not None:
                tenth = int(progress.progress() * 10)
                if tenth > last_report:
                    print(json.dumps({'event':'checkpoint_upload_progress','file':remote_name,
                                      'percent':round(progress.progress()*100,1)}), flush=True)
                    last_report = tenth
    except Exception as exc:
        # Preserve session URI locally but keep it out of logs and Drive reports.
        if request.resumable_uri:
            state['resumable_uri'] = request.resumable_uri
            write_json(state_path, state)
            state_path.chmod(0o600)
        raise RuntimeError('Checkpoint upload interrupted: ' + type(exc).__name__) from None
    remote = drive_metadata(service, state['file_id'])
    if int(remote.get('size', -1)) != local['bytes'] or remote.get('md5Checksum') != local['md5']:
        raise ValueError('Completed Drive file failed size/MD5 verification')
    state['resumable_uri'] = None
    state['complete'] = True
    write_json(state_path, state)
    state_path.chmod(0o600)
    return {'file_id': state['file_id'], **local, 'remote_size_md5_verified': True}


def commit(service, checkpoint, parent_id, work, binding):
    checkpoint, work = Path(checkpoint).resolve(), Path(work).resolve()
    if not checkpoint.is_relative_to(work):
        raise ValueError('Only a new Colab checkpoint below the qualification work root may be uploaded')
    manifest = snapshot_manifest(checkpoint)
    identity = sha256(checkpoint / 'experiment_sources.json')[:16] + '-' + checkpoint.name
    state_dir = work / 'private-upload-state' / identity
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    transaction = state_dir / 'transaction.json'
    if transaction.exists():
        saved = json.loads(transaction.read_text())
        if saved['snapshot'] != manifest or saved['binding'] != binding or saved['parent_id'] != parent_id:
            raise ValueError('Checkpoint transaction binding changed')
        folder_id = saved['folder_id']
    else:
        if project_usage(service, parent_id) + manifest['total_bytes'] + 2 * 1024**3 > PROJECT_CAP_BYTES:
            raise RuntimeError('Project storage cap would be exceeded')
        quota = service.about().get(fields='storageQuota').execute()['storageQuota']
        if 'limit' in quota and int(quota['limit']) - int(quota.get('usage', 0)) < manifest['total_bytes'] + 2 * 1024**3:
            raise RuntimeError('Insufficient Drive quota for the complete checkpoint')
        folder_id = service.files().generateIds(count=1, space='drive', type='files').execute()['ids'][0]
        saved = {'folder_id':folder_id,'snapshot':manifest,'binding':binding,'parent_id':parent_id}
        write_json(transaction, saved)
    try:
        folder = drive_metadata(service, folder_id)
    except Exception as exc:
        if getattr(getattr(exc, 'resp', None), 'status', None) != 404:
            raise RuntimeError('Could not reconcile checkpoint folder: ' + type(exc).__name__) from None
        # Generated IDs make this folder creation retry address the same object.
        service.files().create(body={'id':folder_id,'name':'cuda-qualification-'+identity,
            'mimeType':'application/vnd.google-apps.folder','parents':[validate_drive_id(parent_id)]}, fields='id').execute()
        folder = drive_metadata(service, folder_id)
    if folder.get('mimeType') != 'application/vnd.google-apps.folder' or folder.get('trashed'):
        raise ValueError('Checkpoint transaction folder is invalid')
    payloads = []
    for index, record in enumerate(manifest['files']):
        uploaded = upload_file(service, checkpoint / record['path'], folder_id,
                               state_dir / f'payload-{index:03d}.json', record['path'].replace('/', '__'))
        if uploaded['sha256'] != record['sha256']:
            raise ValueError('Checkpoint changed while uploading')
        payloads.append({**record, 'file_id':uploaded['file_id']})
    committed = {'schema_version':1,'status':'COMMITTED','scope':'new_colab_qualification_checkpoint',
        'binding':binding,'total_bytes':manifest['total_bytes'],'files':payloads,
        'quality_promoted':False,'full_campaign_ready':False}
    path = state_dir / 'COMMITTED.json'
    write_json(path, committed)
    uploaded = upload_file(service, path, folder_id, state_dir / 'commit-upload.json', 'COMMITTED.json')
    return {'folder_id':folder_id,'commit_file_id':uploaded['file_id'],'commit_sha256':uploaded['sha256'],
            'payload_files':len(payloads),'payload_bytes':manifest['total_bytes'],
            'all_payload_remote_size_md5_verified':True,'fresh_vm_restore_tested':False}


def validate_commit(manifest, expected_source_sha256):
    if manifest.get('status') != 'COMMITTED' or manifest.get('schema_version') != 1:
        raise ValueError('Expected a complete qualification commit')
    if manifest.get('binding', {}).get('source_archive_sha256') != expected_source_sha256:
        raise ValueError('Checkpoint belongs to a different source release')
    files = manifest['files']
    names = [str(safe_relative(item['path'])) for item in files]
    if len(names) != len(set(names)) or not REQUIRED <= set(names):
        raise ValueError('Duplicate or missing checkpoint payload')
    if any(type(item['bytes']) is not int or item['bytes'] < 0 for item in files):
        raise ValueError('Invalid checkpoint size')
    total = sum(item['bytes'] for item in files)
    if not 0 < total <= MAX_CHECKPOINT_BYTES:
        raise ValueError('Invalid checkpoint size')
    if total != manifest['total_bytes']:
        raise ValueError('Checkpoint size accounting mismatch')
    return files


def restore(service, commit_id, commit_sha256, work, expected_source_sha256):
    work = Path(work).resolve()
    directory = work / 'restored' / validate_drive_id(commit_id)
    directory.mkdir(parents=True, exist_ok=True)
    commit_path = directory / 'COMMITTED.json'
    if not commit_path.exists():
        download_verified(service, commit_id, commit_path, commit_sha256)
    elif sha256(commit_path) != commit_sha256:
        raise ValueError('Previously restored commit differs')
    manifest = json.loads(commit_path.read_text())
    files = validate_commit(manifest, expected_source_sha256)
    if shutil.disk_usage(work).free < manifest['total_bytes'] + 65 * 1024**3:
        raise RuntimeError('Insufficient local disk for restore and the next complete snapshot')
    target = directory / 'checkpoint'
    for record in files:
        destination = target / record['path']
        if destination.is_symlink() or not destination.resolve().is_relative_to(target.resolve()):
            raise ValueError('Unsafe restore destination')
        if destination.exists():
            if destination.stat().st_size != record['bytes'] or sha256(destination) != record['sha256']:
                raise ValueError('Existing restore payload differs')
        else:
            download_verified(service, record['file_id'], destination, record['sha256'])
            if destination.stat().st_size != record['bytes']:
                raise ValueError('Downloaded payload differs from committed size')
    return {'checkpoint':str(target),'commit_file_id':commit_id,'payloads_verified':len(files),
            'origin_boot_id':manifest['binding'].get('runtime_boot_id')}
