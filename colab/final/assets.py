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

def fetch(url,path,entry,token=None,reserve=0,verified_existing=False):
    import requests
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if verified_existing or verified(path,entry): return path
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


def _active_component_keys(lock,config):
    spark_key=config.get('spark_component','spark')
    if config.get('modalities','omni')=='vision_text':
        return [k for k in (spark_key,'vision') if k in lock]
    return [k for k in lock.keys() if (k not in ('spark','spark_fp8') or k==spark_key)]


def prepare_assets(root,token,config,download_weights=True):
    root=Path(root); lock=read(HERE/'components.lock.json'); paths={}
    active_keys=_active_component_keys(lock,config)
    include_motion='motion' in active_keys
    is_local_workstation=(config.get('execution_target')=='local_workstation' and config.get('checkpoint_backend')=='local')

    checks={};remaining=0
    for key in active_keys:
        if key=='motion': continue
        spec=lock[key]
        for f in spec['files']:
            candidate=root/'assets'/key/f['path'];weight=f['path'].endswith(('.safetensors','.bin'))
            valid=verified(candidate,f) if download_weights or not weight else False
            checks[candidate]=valid
            if download_weights and weight and not valid: remaining+=f['bytes']
    motion_path=root/'assets/motion.pt'
    if include_motion:
        motion_valid=verified(motion_path,lock['motion']['checkpoint']) if download_weights else False
        checks[motion_path]=motion_valid
        if download_weights and not motion_valid: remaining+=lock['motion']['checkpoint']['bytes']

    all_local_verified=bool(checks) and all(checks.values())
    if not (is_local_workstation and all_local_verified):
        from huggingface_hub import HfApi
        api=HfApi(token=token)
        if not token: raise ValueError('HF_TOKEN must be enabled in Colab Secrets')
        who=api.whoami()
        role=(who.get('auth',{}).get('accessToken') or {}).get('role')
        if not is_local_workstation:
            if role=='read': raise ValueError('HF_TOKEN must permit writing the configured model/checkpoint repositories, not only reading datasets')
            if who.get('name')!='TassP0lar': raise ValueError('HF_TOKEN belongs to an unexpected publishing account')
        emit('hf_identity_verified',account=who.get('name'),token_role=role,publication_write_access_proven=False)
        if 'sound_out' in active_keys:
            # Check gated access before allocating a GPU or downloading other weights.
            check_gated_access(lock['sound_out'],token)

    from .run import directory_bytes
    allocated=directory_bytes(root/'checkpoints') if (root/'checkpoints').exists() else 0
    checkpoint_reserve=max(0,2*config['initial_checkpoint_reservation_bytes']-allocated) if download_weights else 0
    free_floor=config.get('local_free_floor_bytes',44_000_000_000)
    if download_weights: disk_guard(root,pending=remaining,reserve=checkpoint_reserve,min_free_bytes=free_floor)
    for key in active_keys:
        if key=='motion': continue
        spec=lock[key]
        alias='spark' if key in ('spark','spark_fp8') else key
        path=root/'assets'/key; paths[alias]=path
        for f in spec['files']:
            is_weight=f['path'].endswith(('.safetensors','.bin'))
            if is_weight and not download_weights: continue
            fetch(f"https://huggingface.co/{spec['repository']}/resolve/{spec['revision']}/{f['path']}",path/f['path'],f,token,reserve=checkpoint_reserve,verified_existing=checks[path/f['path']])
        emit('component_ready',component=key,revision=spec['revision'],weights=download_weights)
    if include_motion:
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
        paths['motion']=motion_path
        if download_weights: fetch(motion['checkpoint']['url'],paths['motion'],motion['checkpoint'],reserve=checkpoint_reserve,verified_existing=checks[motion_path])
    write(root/'asset-receipt.json',{'component_lock_sha256':sha(HERE/'components.lock.json'),'modalities':config.get('modalities','omni'),'spark_component':config.get('spark_component','spark'),'weights_downloaded':download_weights,'gated_access_checked':'sound_out' in active_keys,'paths':{k:str(v) for k,v in paths.items()}})
    return paths


def prepare_local_synccraft_data(root,config,*,include_animation=True,panel_size=(128,128)):
    """Build a verified local TAR shard + SQLite split corpus for clean-room SyncCraft + vision tasks."""
    import gzip,io,json,tarfile
    from PIL import Image
    from .common import identity
    from .data import build_database
    from .synccraft_bridge import (
        build_animated_quadruped_program,
        build_animation_clip_record,
        build_synccraft_curriculum_records,
    )

    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    db=root/'metadata/corpus.sqlite'
    cat=root/'metadata/packets.jsonl.gz'
    handoff_path=root/'metadata/HANDOFF.json'
    if db.exists() and cat.exists() and handoff_path.exists():
        expected={'catalogue_sha256':sha(cat),'split':config['split_percent'],'seed':config['seed']}
        if read(root/'metadata/BUILD.json')!=expected:
            raise ValueError('Existing dataset split differs; use a new campaign directory')
        return read(db.with_suffix('.report.json'))

    n_static_var=max(1,int(config.get('synccraft_num_variants',1)))
    n_anim_var=max(6,int(config.get('synccraft_animation_variants',6)))
    n_pilot_var=max(8,int(config.get('synccraft_pilot_groups',8)))

    pairs=list(build_synccraft_curriculum_records(include_animation=False,enable_thinking=True,panel_size=panel_size,num_variants=n_static_var))
    if include_animation:
        # Build >=6 independent animation families so `video:en` satisfies independent split coverage.
        for anim_idx in range(n_anim_var):
            prog=build_animated_quadruped_program(variant=anim_idx)
            rec,payloads=build_animation_clip_record(
                prog,
                clip_name='walk_cycle',
                asset_id=f'synccraft_anim_quad_{anim_idx:03d}',
                family_id=f'quadruped_gait_family_{anim_idx:03d}',
                num_keyframes=4,
                enable_thinking=True,
                panel_size=panel_size,
            )
            pairs.append((rec,payloads))

    # Add >=8 independent bilingual pure-text (`text_replay`) and RGBA/opaque visual grounding groups.
    for idx in range(n_pilot_var):
        for lang in ('en','zh'):
            txt_row={
                'id':f'spark_text_preserve_{lang}_{idx}',
                'source_group':f'spark_text_group_{lang}_{idx}',
                'language':lang,
                'task':'text_replay',
                'enable_thinking':bool(idx%2==0),
                'frames':[],
                'prompt':(
                    f'Compute {idx+3} * {idx+5} and state the integer result.'
                    if lang=='en'
                    else f'计算 {idx+3} 乘以 {idx+5} 并给出整数结果。'
                ),
                'answer':(
                    f'<think>\nMultiply {idx+3} by {idx+5}.\n</think>\n{(idx+3)*(idx+5)}'
                    if idx%2==0
                    else str((idx+3)*(idx+5))
                ),
                'target':{'kind':'text','text':str((idx+3)*(idx+5))},
                'metadata':{},
            }
            pairs.append((txt_row,{}))

            has_cutout=(idx%2==0)
            arr=__import__('numpy').zeros((64,64,4),dtype='uint8')
            arr[:,:,0]=(40+23*idx)%255; arr[:,:,1]=(90+17*idx)%255; arr[:,:,2]=(180+11*(0 if lang=='en' else 3))%255; arr[:,:,3]=255
            if has_cutout:
                arr[16:48,16:48,3]=0
            buf=io.BytesIO()
            Image.fromarray(arr,mode='RGBA').save(buf,format='PNG')
            png_bytes=buf.getvalue()
            member=f'pilot/{lang}_{idx}/texture.png'
            ans_txt='transparent_cutout' if has_cutout else 'opaque_solid'
            vis_row={
                'id':f'spark_rgba_pilot_{lang}_{idx}',
                'source_group':f'spark_rgba_group_{lang}_{idx}',
                'language':lang,
                'task':'rgba_visual_pilot',
                'visual_kind':'uv_atlas',
                'enable_thinking':False,
                'frames':[member],
                'media_times_s':[0.0],
                'visual_units':[{'visual_kind':'uv_atlas','view_id':'uv_vanilla','is_pixel_art_atlas':True}],
                'prompt':(
                    'Inspect the RGBA texture and state whether it has a transparent cutout region.'
                    if lang=='en'
                    else '检查该 RGBA 贴图并说明是否包含透明镂空区域。'
                ),
                'answer':ans_txt,
                'target':{'kind':'text','text':ans_txt},
                'metadata':{'has_alpha_cutout':has_cutout},
            }
            pairs.append((vis_row,{member:png_bytes}))

    shard_name='shard-00000.tar'
    cache_dir=root/'media-cache/synccraft_local'
    cache_dir.mkdir(parents=True,exist_ok=True)
    shard_path=cache_dir/shard_name
    blob_map={}
    for _row,payloads in pairs:
        for m_path,payload in payloads.items():
            if m_path in blob_map and blob_map[m_path]!=payload:
                raise ValueError(f'Conflicting TAR media member path {m_path!r} with distinct byte payloads')
            blob_map[m_path]=payload

    member_files=[]
    member_offsets={}
    with tarfile.open(shard_path,'w:') as tf:
        for m_path in sorted(blob_map):
            payload=blob_map[m_path]
            ti=tarfile.TarInfo(name=m_path)
            ti.size=len(payload)
            ti.mtime=0
            ti.mode=0o644
            tf.addfile(ti,io.BytesIO(payload))
    with tarfile.open(shard_path,'r:') as tf:
        for m in tf.getmembers():
            payload=blob_map[m.name]
            digest=hashlib.sha256(payload).hexdigest()
            member_offsets[m.name]={'member':m.name,'offset':m.offset_data,'length':m.size,'sha256':digest}
            member_files.append({'path':m.name,'bytes':m.size,'sha256':digest,'offset_data':m.offset_data})

    shard_bytes=shard_path.stat().st_size
    shard_sha=sha(shard_path)
    release_doc={
        'schema_version':1,
        'kind':'immutable_multimodal_dataset_shards',
        'role':'training',
        'dataset_manifest_sha256':identity({'items':len(pairs),'seed':config['seed']}),
        'media_manifest_sha256':identity(member_files),
        'payload_bytes':shard_bytes,
        'shards':[{'path':shard_name,'bytes':shard_bytes,'sha256':shard_sha,'files':member_files}],
    }
    release_path=root/'indices/synccraft_local/RELEASE.json'
    write(release_path,release_doc)
    release_sha=sha(release_path)
    handoff_doc={
        'repo_id':'local/synccraft-cleanroom',
        'revision':'22ff1605d75143870743d26fa5d9445a32d6e627',
        'components':[{'id':'synccraft_local','prefix':'synccraft_local','release_bytes':release_path.stat().st_size,'release_sha256':release_sha}],
    }
    write(handoff_path,handoff_doc)

    cat.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(cat,'wt',encoding='utf-8') as gz:
        for row,payloads in pairs:
            frames=row.get('frames') or []
            transport=None
            if frames:
                transport={'shard':shard_name,'media':[dict(member_offsets[m_name]) for m_name in frames]}
            packet={'component':'synccraft_local','record':row,'transport':transport}
            gz.write(json.dumps(packet,ensure_ascii=False,separators=(',',':'))+'\n')

    report=build_database(cat,db,config)
    write(root/'metadata/BUILD.json',{'catalogue_sha256':sha(cat),'split':config['split_percent'],'seed':config['seed']})
    return report


def prepare_data(root,token,config):
    from .data import build_database
    root=Path(root); root.mkdir(exist_ok=True,parents=True)
    if config.get('dataset_mode')=='hybrid_world_v32':
        from .hybrid_corpus import prepare_hybrid_world_v32_data
        return prepare_hybrid_world_v32_data(root,token,config)
    if config.get('dataset_mode')=='local_synccraft' or (root/'metadata/HANDOFF.json').is_file():
        return prepare_local_synccraft_data(root,config)
    lock=read(HERE/'dataset.lock.json'); handoff=read(HERE.parent/'formal/HF_FULL_DATASET_HANDOFF.json')
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
