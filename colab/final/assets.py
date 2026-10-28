"""Revision-pinned, content-verified direct downloads without an HF cache copy."""
from __future__ import annotations
import hashlib, os, shutil, zipfile
from pathlib import Path
from .common import HERE,read,write,sha,disk_guard,emit

def verified(path,entry):
    p=Path(path)
    if not p.is_file() or p.stat().st_size!=entry['bytes']: return False
    if entry.get('sha256'): return sha(p)==entry['sha256']
    h=hashlib.sha1(); h.update(f'blob {p.stat().st_size}\0'.encode())
    with p.open('rb') as source:
        for b in iter(lambda:source.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()==entry['git_blob_sha1']

def fetch(url,path,entry,token=None,reserve=0):
    import requests
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if verified(path,entry): return path
    disk_guard(path.parent,pending=entry['bytes'],reserve=reserve)
    partial=path.with_name(path.name+'.partial'); start=partial.stat().st_size if partial.exists() else 0
    if start>entry['bytes']: raise ValueError('Oversized partial download')
    headers={'Range':f'bytes={start}-'} if start else {}
    if token: headers['Authorization']='Bearer '+token
    session=requests.Session(); session.trust_env=False
    try:
        with session.get(url,headers=headers,stream=True,timeout=(30,120)) as response:
            if response.status_code not in (200,206): raise RuntimeError('Pinned asset download rejected (HTTP '+str(response.status_code)+'); verify access to the named model in the notebook')
            if start and response.status_code==206:
                if not response.headers.get('Content-Range','').startswith(f'bytes {start}-'): raise ValueError('Wrong resume range')
                mode='ab'
            else: mode='wb'; start=0
            with partial.open(mode) as dest:
                for block in response.iter_content(8*1024**2):
                    if block:
                        dest.write(block)
                        if dest.tell()>entry['bytes']: raise ValueError('Download exceeded frozen size')
        if not verified(partial,entry): raise ValueError('Pinned asset size or content hash differs')
        os.replace(partial,path); return path
    except requests.RequestException as exc:
        raise RuntimeError('Asset transport failed: '+type(exc).__name__) from None
    finally: session.close()

def check_gated_access(spec,token):
    """Check Hub authorization/identity without HEADing the signed CDN URL."""
    from huggingface_hub import get_hf_file_metadata,hf_hub_url
    from huggingface_hub.errors import GatedRepoError,HfHubHTTPError
    entry=next(f for f in spec['files'] if f['path'].endswith('.safetensors'))
    repo=spec['repository']; page=f'https://huggingface.co/{repo}'
    try:
        metadata=get_hf_file_metadata(hf_hub_url(repo,entry['path'],revision=spec['revision']),token=token,timeout=30)
    except GatedRepoError as exc:
        status=exc.response.status_code if exc.response is not None else 'unknown'
        raise RuntimeError(f'Model access denied (HTTP {status}, GatedRepo): {repo}. '
                           f'Using the same HF account as HF_TOKEN, review/request access at {page}. '
                           'After approval, ensure the fine-grained token permits reading this gated model, '
                           'then rerun CPU preflight. No license is accepted automatically.') from None
    except HfHubHTTPError as exc:
        status=exc.response.status_code if exc.response is not None else 'unknown'
        raise RuntimeError(f'Model metadata request failed (HTTP {status}): {repo}. '
                           'Check token read scope, pinned revision and Hub availability; '
                           'this response does not establish missing license approval.') from None
    except Exception as exc:
        raise RuntimeError(f'Model metadata transport failed for {repo}: {type(exc).__name__}. '
                           'Retry CPU preflight after resolving connectivity; access denial is not established.') from None
    if metadata.commit_hash!=spec['revision'] or metadata.size!=entry['bytes'] or metadata.etag!=entry['sha256']:
        raise ValueError(f'Gated model metadata differs from the frozen size, revision or SHA256: {repo}')
    emit('model_access_verified',repository=repo,revision=spec['revision'],weights_downloaded=False)


def prepare_assets(root,token,config,download_weights=True):
    from huggingface_hub import HfApi
    root=Path(root); lock=read(HERE/'components.lock.json'); paths={}
    api=HfApi(token=token)
    if not token: raise ValueError('HF_TOKEN must be enabled in Colab Secrets')
    who=api.whoami()
    role=(who.get('auth',{}).get('accessToken') or {}).get('role')
    if role=='read': raise ValueError('HF_TOKEN must permit writing the configured model/checkpoint repositories, not only reading datasets')
    if who.get('name')!='TassP0lar': raise ValueError('HF_TOKEN belongs to an unexpected publishing account')
    emit('hf_identity_verified',account=who['name'],token_role=role,publication_write_access_proven=False)
    # Check gated access before allocating a GPU or downloading other weights.
    check_gated_access(lock['sound_out'],token)
    remaining=sum(v['bytes'] for k,v in lock.items() if k!='motion')+lock['motion']['checkpoint']['bytes']
    if download_weights: disk_guard(root,pending=remaining,reserve=2*config['initial_checkpoint_reservation_bytes'])
    for key,spec in lock.items():
        if key=='motion': continue
        path=root/'assets'/key; paths[key]=path
        for f in spec['files']:
            is_weight=f['path'].endswith(('.safetensors','.bin'))
            if is_weight and not download_weights: continue
            fetch(f"https://huggingface.co/{spec['repository']}/resolve/{spec['revision']}/{f['path']}",path/f['path'],f,token,reserve=2*config['initial_checkpoint_reservation_bytes'] if download_weights else 0)
        emit('component_ready',component=key,revision=spec['revision'],weights=download_weights)
    motion=lock['motion']; archive=HERE/motion['source_archive']['path']
    if sha(archive)!=motion['source_archive']['sha256']: raise ValueError('V-JEPA source archive changed')
    source=root/'vendor'; source.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target=(source/item.filename).resolve()
            if not target.is_relative_to(source.resolve()) or item.file_size>100*1024**2: raise ValueError('Unsafe source archive member')
        z.extractall(source)
        prefix=z.namelist()[0].split('/')[0]
    paths['motion_source']=source/prefix
    paths['motion']=root/'assets/motion.pt'
    if download_weights: fetch(motion['checkpoint']['url'],paths['motion'],motion['checkpoint'],reserve=2*config['initial_checkpoint_reservation_bytes'])
    write(root/'asset-receipt.json',{'component_lock_sha256':sha(HERE/'components.lock.json'),'weights_downloaded':download_weights,'gated_access_checked':True,'paths':{k:str(v) for k,v in paths.items()}})
    return paths

def prepare_data(root,token,config):
    from .data import build_database
    root=Path(root); root.mkdir(exist_ok=True,parents=True); lock=read(HERE/'dataset.lock.json'); handoff=read(HERE.parent/'formal/HF_FULL_DATASET_HANDOFF.json')
    cat=root/'metadata/packets.jsonl.gz'; fetch(lock['catalogue_url'],cat,lock['catalogue'],token)
    for c in handoff['components']:
        f={'bytes':c['release_bytes'],'sha256':c['release_sha256']}
        fetch(f"https://huggingface.co/datasets/{handoff['repo_id']}/resolve/{handoff['revision']}/{c['prefix']}/RELEASE.json",root/'indices'/c['id']/'RELEASE.json',f,token)
    db=root/'metadata/corpus.sqlite'; expected={'catalogue_sha256':sha(cat),'split':config['split_percent'],'seed':config['seed']}
    if db.exists():
        if read(root/'metadata/BUILD.json')!=expected: raise ValueError('Existing dataset split differs; use a new campaign directory')
    else:
        report=build_database(cat,db,config)
        if report['original_instructions']!=304861: raise ValueError('Full corpus instruction coverage differs')
        write(root/'metadata/BUILD.json',expected)
    return read(db.with_suffix('.report.json'))
