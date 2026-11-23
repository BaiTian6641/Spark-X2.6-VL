"""Full-parameter & staged campaign engine supporting native APOLLO + COAT + merged-gradient training and Studio SFT."""
from __future__ import annotations
import contextlib, gc, importlib.metadata, math, os, random, secrets, shutil, signal, time
from collections import Counter,defaultdict
from pathlib import Path
from .common import HERE,OPTIMIZER_PROFILES,canonical,identity,read,write,sha,disk_guard,emit,ComputeBudget,runtime_boot_id
from .data import Catalogue,CommittedCursor,MediaReader,prepare_media,apply_dynamic_training_augmentation,FAMILIES,cursor_dataset
from .checkpoint_cache import acknowledged,prune,rebind_assets
from .hf_checkpoints import receipt_path


def rng_state():
    import numpy as np,torch
    s=np.random.get_state()
    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {'python':random.getstate(),'numpy':[s[0],s[1].tolist(),s[2],s[3],s[4]],'torch':torch.get_rng_state(),'cuda':cuda_rng}

def set_rng(s):
    import numpy as np,torch
    random.setstate(s['python']); n=s['numpy']; np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4])); torch.set_rng_state(s['torch'])
    if torch.cuda.is_available() and s.get('cuda'):
        torch.cuda.set_rng_state_all(s['cuda'])

def sample_parameters(model):
    if hasattr(model,'sync_intel_vision_to_host'):
        model.sync_intel_vision_to_host()
    return {n:p.detach().flatten()[:16].float().cpu().tolist() for n,p in model.named_parameters()}
def sample_hash(model): return identity(sample_parameters(model))
def census(model):
    groups=defaultdict(lambda:{'parameters':0,'trainable':0})
    for group,name,p in model.named_groups(): groups[group]['parameters']+=p.numel(); groups[group]['trainable']+=p.numel() if p.requires_grad else 0
    if hasattr(model,'has_intel_vision_bridge') and model.has_intel_vision_bridge() and 'vision' in groups:
        tel=model.intel_vision_telemetry()
        groups['vision']['intel_arc_xpu_trainable']=int(tel.get('trainable_parameters',0))
        groups['vision']['intel_arc_device']=str(tel.get('device_name',tel.get('device','xpu:0')))
    return dict(groups)

def binding(config,source_sha):
    operational=('observed_compute_balance','observed_cu_per_hour','instructor_route_eligibility_receipt',
                 'drive_project_folder','direct_wall_hours','enforce_compute_budget','run_phase')
    stable={k:v for k,v in config.items() if k not in operational}
    return {'source_archive_sha256':source_sha,'component_lock_sha256':sha(HERE/'components.lock.json'),'dataset_lock_sha256':sha(HERE/'dataset.lock.json'),'policy_sha256':identity(stable)}

def loss_weights(catalogue,order,config):
    classes={r[0]:(r[1]+':'+r[2]) for r in catalogue.select('train')}
    counts=Counter(classes[rid] for rid in order); n=len(order)
    return {key:min(config['max_loss_weight'],n/(len(counts)*count)) for key,count in counts.items()}

def _lr_for_group_and_name(group:str,name:str,config:dict)->float:
    base_lr=float(config.get('backbone_lr',config.get('learning_rate',1e-4)))
    conn_lr=float(config.get('adapter_lr',config.get('connector_learning_rate',base_lr)))
    engram_lr=float(config.get('engram_lr',config.get('engram_learning_rate',conn_lr)))
    vis_lr=float(config.get('vision_lr',config.get('vision_learning_rate',base_lr*0.5)))
    out_lr=float(config.get('output_lr',config.get('audio_output_learning_rate',base_lr)))
    early_n=int(config.get('early_spark_layers',0) or 0)
    early_lr=float(config.get('early_spark_lr',base_lr))
    if group=='engram_mhc' or name.startswith(('engram_anchors.','loop_controllers.')):
        return engram_lr
    if group in ('connectors','visual_adapters') or '.bridge.' in name or 'default_voice' in name or 'visual_cross_attn' in name:
        return conn_lr
    if group in ('speech_output','sound_output'):
        return out_lr
    if group in ('vision','motion','speech_in','sound_in'):
        return vis_lr
    if early_n>0 and name.startswith('spark.model.layers.'):
        parts=name.split('.')
        if len(parts)>3 and parts[3].isdigit() and int(parts[3])<early_n:
            return early_lr
    return base_lr

def optimizer_for(model,config,alignment=False):
    """Build the configured optimizer profile without ever applying stock CoatAdamW to APOLLO matrices."""
    import torch
    from .optimizer import ApolloMergedOptimizer, build_apollo_param_groups

    profile=config.get('optimizer_profile','adamw8bit')
    if profile not in OPTIMIZER_PROFILES:
        raise ValueError(f'Unknown optimizer_profile {profile!r}; expected one of {OPTIMIZER_PROFILES}')

    expected={id(p) for p in model.parameters() if p.requires_grad}

    if profile in ('apollo_merged','apollo_mini_merged','apollo_coat','apollo_coat_fp8_moments'):
        is_mini=(profile=='apollo_mini_merged')
        moment_codec='coat_fp8' if (profile=='apollo_coat_fp8_moments' or config.get('coat_moment_fp8',False)) else 'fp32'
        rank=1 if is_mini else int(config.get('apollo_rank',64))
        scale=float(config.get('apollo_mini_scale',128.0)) if is_mini else float(config.get('apollo_scale',1.0))
        cfg_apollo={
            **config,
            'apollo_scale':scale,
            'apollo_min_dim':int(config.get('apollo_min_dim',config.get('apollo_min_2d_dim',128))),
        }
        param_groups=build_apollo_param_groups(
            model,
            cfg_apollo,
            rank=rank,
            scale_type='tensor' if is_mini else 'channel',
            moment_codec=moment_codec,
        )
        optimizer=ApolloMergedOptimizer(
            param_groups,
            seed=int(config.get('seed',39526)),
            coat_group_size=int(config.get('coat_moment_group_size',128)),
        )
    elif profile=='bf16_adamw':
        buckets=defaultdict(list); seen=set()
        for group,name,p in model.named_groups():
            if not p.requires_grad or id(p) in seen: continue
            seen.add(id(p))
            no_decay=(p.ndim<2) or name.endswith('.bias') or any(k in name.lower() for k in ('norm','gate_attn','gate_ffn','early_mid_gate'))
            wd=0.0 if no_decay else float(config['weight_decay'])
            buckets[(_lr_for_group_and_name(group,name,config), wd)].append(p)
        optimizer=torch.optim.AdamW(
            [{'params':v,'lr':lr,'weight_decay':wd} for (lr,wd),v in buckets.items()],
            betas=(.9,.95),
            eps=1e-8,
            foreach=False,
        )
    else:
        import bitsandbytes as bnb
        buckets=defaultdict(list); seen=set()
        for group,name,p in model.named_groups():
            if not p.requires_grad or id(p) in seen: continue
            seen.add(id(p))
            no_decay=(p.ndim<2) or name.endswith('.bias') or any(k in name.lower() for k in ('norm','gate_attn','gate_ffn','early_mid_gate'))
            wd=0.0 if no_decay else float(config['weight_decay'])
            buckets[(_lr_for_group_and_name(group,name,config), wd)].append(p)
        optimizer=bnb.optim.AdamW8bit(
            [{'params':v,'lr':lr,'weight_decay':wd} for (lr,wd),v in buckets.items()],
            betas=(.9,.95),
            eps=1e-8,
        )

    actual={id(p) for g in optimizer.param_groups for p in g['params']}
    if actual!=expected: raise AssertionError('Optimizer omits or adds trainable parameters')
    return optimizer

def ensure_coat_for_model(model,config):
    """Install Spark COAT activation compression when configured; return controller or None."""
    profile=config.get('optimizer_profile','adamw8bit')
    want_coat=profile in ('apollo_coat','apollo_coat_fp8_moments') and bool(config.get('coat_activation_fp8',True))
    existing=getattr(model,'_coat_controller',None)
    if not want_coat:
        if existing is not None:
            existing.enabled=False
        return existing
    if existing is not None:
        existing.enabled=True
        return existing
    from .coat_spark import install_spark_coat
    has_grad_ckpt = bool(getattr(getattr(model.spark, 'model', model.spark), 'gradient_checkpointing', False))
    ctrl=install_spark_coat(
        model.spark,
        group_size=int(config.get('coat_activation_group_size',16)),
        min_features=int(config.get('apollo_min_dim',64)),
        use_fp8_matmul=bool(config.get('coat_fp8_linear_forward',False)),
        use_saved_tensors_hook=not has_grad_ckpt,
    )
    model._coat_controller=ctrl
    return ctrl

def copy_reload_assets(model,destination):
    destination=Path(destination); out=destination/'reload-assets'
    is_omni=(getattr(model,'modalities','omni')=='omni')
    for key,raw in model.paths.items():
        if key in ('motion','motion_source'): continue
        if not is_omni and key not in ('spark','vision'): continue
        root=Path(raw)
        if not root.exists(): continue
        for path in root.rglob('*'):
            if not path.is_file(): continue
            rel=path.relative_to(root)
            weight=path.suffix in ('.safetensors','.bin','.pt')
            if weight and not (key=='speech_out' and str(rel).startswith('speech_tokenizer/')): continue
            dest=out/key/rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,dest)
    if is_omni and (HERE/'vendor/vjepa2-source.zip').is_file():
        vendor=out/'motion-source.zip'; vendor.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(HERE/'vendor/vjepa2-source.zip',vendor)
    # Every saved package carries the reviewed inference/runtime source and lock.
    runtime=destination/'runtime'
    release=HERE.parents[1]
    include=[p for p in (release/'colab').rglob('*.py') if '__pycache__' not in p.parts and (p.parent==HERE or p.name=='__init__.py' or p.relative_to(release).as_posix() in ('colab/preflight.py','colab/stage1/common.py','colab/stage1/drive_checkpoint.py','colab/stage1/requirements.in','colab/formal/rolling_cache.py','colab/formal/shard_dataset.py','colab/formal/local_storage.py','colab/formal/teacher_readiness.py'))]
    lock_candidates=[HERE/'components.lock.json',HERE/'dataset.lock.json',HERE/'bootstrap.lock.json',HERE/'requirements.lock',HERE/'requirements-preflight.lock',HERE/'requirements.in',HERE/'requirements-preflight.in',HERE/'requirements-flash-attn.in',HERE/'requirements-flash-attn.lock',HERE.parent/'stage1/requirements.in',HERE.parent/'formal/HF_FULL_DATASET_HANDOFF.json']
    include += [p for p in lock_candidates if p.is_file()]
    for src in include:
        dest=runtime/src.relative_to(release); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dest)


def merge_instructor_ledgers(current,restored):
    """Union request IDs while allowing only intent -> outcome progression."""
    merged=dict(restored); merged.update(current)
    requests=[dict(entry) for entry in current.get('requests',[])]
    positions={entry['id']:i for i,entry in enumerate(requests)}
    for entry in restored.get('requests',[]):
        if entry['id'] not in positions:
            positions[entry['id']]=len(requests); requests.append(dict(entry))
            continue
        local=requests[positions[entry['id']]]
        if local.get('status')=='reserved_ambiguous' and entry.get('status') not in (None,'reserved_ambiguous'):
            requests[positions[entry['id']]]=dict(entry)
    merged['requests']=requests
    return merged

def deferred_checkpoint_receipt(directory,bound):
    from .storage import manifest
    directory=Path(directory);snapshot=manifest(directory)
    receipt={'backend':'local_deferred','directory':str(directory),'kind':'checkpoint',
             'commit_sha256':identity(snapshot),'total_bytes':snapshot['total_bytes'],
             'verified':False,'binding':bound,
             'runtime_boot_id':runtime_boot_id()}
    write(receipt_path(directory),receipt)
    return receipt


def _save_model_safetensors(model, path):
    try:
        from safetensors.torch import save_model
        save_model(model, str(path))
        return
    except ModuleNotFoundError:
        pass
    import json, struct, torch
    dtype_map = {
        torch.float32: 'F32',
        torch.bfloat16: 'BF16',
        torch.float16: 'F16',
        torch.int64: 'I64',
        torch.int32: 'I32',
        torch.uint8: 'U8',
        torch.bool: 'BOOL',
    }
    state = model.state_dict()
    seen_ptrs = set()
    header = {}
    blobs = []
    offset = 0
    for name, tensor in state.items():
        ptr = tensor.untyped_storage().data_ptr()
        if (ptr, tensor.shape, tensor.stride()) in seen_ptrs:
            continue
        seen_ptrs.add((ptr, tensor.shape, tensor.stride()))
        cpu_t = tensor.detach().contiguous().cpu()
        dt = dtype_map[cpu_t.dtype]
        raw = bytes(cpu_t.reshape(-1).view(torch.uint8).numpy())
        end = offset + len(raw)
        header[name] = {'dtype': dt, 'shape': list(cpu_t.shape), 'data_offsets': [offset, end]}
        blobs.append(raw)
        offset = end
    header_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')
    pad = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b' ' * pad
    with Path(path).open('wb') as f:
        f.write(struct.pack('<Q', len(header_bytes)))
        f.write(header_bytes)
        for raw in blobs:
            f.write(raw)


def _load_model_safetensors(model, path, *, strict=True, device='cpu'):
    try:
        from safetensors.torch import load_model
        load_model(model, str(path), strict=strict, device=device)
        return
    except ModuleNotFoundError:
        pass
    import json, struct, torch
    rev_dtype = {
        'F32': torch.float32,
        'BF16': torch.bfloat16,
        'F16': torch.float16,
        'I64': torch.int64,
        'I32': torch.int32,
        'U8': torch.uint8,
        'BOOL': torch.bool,
    }
    raw_all = Path(path).read_bytes()
    (header_len,) = struct.unpack('<Q', raw_all[:8])
    header = json.loads(raw_all[8:8 + header_len].decode('utf-8'))
    body = memoryview(raw_all)[8 + header_len:]
    state = {}
    for name, meta in header.items():
        if name == '__metadata__':
            continue
        start, end = meta['data_offsets']
        dt = rev_dtype[meta['dtype']]
        chunk = bytes(body[start:end])
        t = torch.frombuffer(bytearray(chunk), dtype=dt).reshape(meta['shape']).to(device=device) if len(chunk) > 0 else torch.empty(meta['shape'], dtype=dt, device=device)
        state[name] = t
    model.load_state_dict(state, strict=strict)


def _build_scheduler(optimizer, kind='cosine', num_warmup_steps=0, num_training_steps=1):
    import torch
    try:
        if kind == 'constant':
            from transformers import get_constant_schedule
            return get_constant_schedule(optimizer)
        from transformers import get_cosine_schedule_with_warmup
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max(1, num_training_steps))
    except ModuleNotFoundError:
        if kind == 'constant':
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        total = max(1, int(num_training_steps))
        warm = max(0, int(num_warmup_steps))
        def _lr_lambda(step: int) -> float:
            if step < warm:
                return float(step) / float(max(1, warm))
            progress = float(step - warm) / float(max(1, total - warm))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


def checkpoint(model,optimizer,scheduler,state,root,config,bound,store,defer_upload=False):
    import torch
    root=Path(root); name=f"{state['stage']}-{state.get('updates',0):08d}"; directory=root/'checkpoints'/name
    if not (root/'compute-budget.json').is_file(): raise RuntimeError('Missing compute-budget.json; persist campaign compute accounting before checkpointing')
    estimate=max(state.get('qualified_checkpoint_reservation_bytes',config['initial_checkpoint_reservation_bytes']),int(state.get('checkpoint_bytes',0)))
    free_floor=config.get('local_free_floor_bytes',44_000_000_000)
    if directory.exists():
        names={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
        required={'model.safetensors','optimizer.pt','scheduler.pt','rng.pt','STATE.json','config.json','BINDING.json','PARAMETERS.json','CLOSED.json','COMPUTE_BUDGET.json'}
        if required<=names:
            if read(directory/'BINDING.json')!=bound or read(directory/'PARAMETERS.json')['sample_sha256']!=sample_hash(model): raise ValueError('Checkpoint name collides with a changed model/source')
            receipt_file=receipt_path(directory)
            local_bytes=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
            if local_bytes>estimate: raise RuntimeError('Measured checkpoint exceeds its reservation; preserve it and revise the disk profile before further writes')
            if acknowledged(directory,bound):
                result=read(receipt_file)
            elif defer_upload:
                result=read(receipt_file) if receipt_file.exists() else deferred_checkpoint_receipt(directory,bound)
            else:
                result=store.commit(directory)
            state.update(last_checkpoint=result,last_checkpoint_local=str(directory),checkpoint_bytes=local_bytes)
            rebind_assets(model,directory)
            if not defer_upload: prune(root,bound,[directory])
            return result
        intent=directory/'INTENT.json'
        if not intent.exists() or read(intent).get('binding')!=bound: raise RuntimeError('Unknown incomplete local artifact; no automatic overwrite')
        shutil.rmtree(directory)
    # Reserve the second complete slot before writing a new closed snapshot.
    prune(root,bound,[state.get('last_checkpoint_local')],incoming=directory)
    existing=[p for p in (root/'checkpoints').glob('*') if p.is_dir()]
    disk_guard(root,pending=estimate,reserve=estimate if not existing else 0,min_free_bytes=free_floor)
    directory.mkdir(parents=True); write(directory/'INTENT.json',{'binding':bound,'stage':state['stage'],'updates':state.get('updates',0)}); started=time.monotonic()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    if hasattr(model,'sync_intel_vision_to_host'):
        model.sync_intel_vision_to_host()
    _save_model_safetensors(model, directory/'model.safetensors')
    torch.save(optimizer.state_dict(),directory/'optimizer.pt'); torch.save(scheduler.state_dict(),directory/'scheduler.pt'); torch.save(rng_state(),directory/'rng.pt')
    if hasattr(model,'has_tier1b_state') and model.has_tier1b_state():
        torch.save(model.tier1b_state_dict(),directory/'tier1b_optim.pt')
    if hasattr(model,'has_intel_vision_optim_state') and model.has_intel_vision_optim_state():
        torch.save(model.intel_vision_optim_state_dict(),directory/'intel_vision_optim.pt')
    write(directory/'STATE.json',state); write(directory/'config.json',config); write(directory/'BINDING.json',bound); write(directory/'PARAMETERS.json',{'sample_sha256':sample_hash(model),'census':census(model)})
    copy_reload_assets(model,directory)
    for name in ('corpus.report.json','BUILD.json'):
        src=root/'metadata'/name
        if src.exists(): shutil.copy2(src,directory/name)
    for name in ('QUALIFICATION.json','FULL_RECOVERY_PROOF.json','BASELINE.json','INSTRUCTOR_FORMAT_PROBE.json'):
        if (root/name).exists():
            dest=directory/'provenance'/name; dest.parent.mkdir(exist_ok=True); shutil.copy2(root/name,dest)
    if (root/'instructor/LEDGER.json').exists(): shutil.copy2(root/'instructor/LEDGER.json',directory/'INSTRUCTOR_LEDGER.json')
    shutil.copy2(root/'compute-budget.json',directory/'COMPUTE_BUDGET.json')
    write(directory/'CLOSED.json',{'binding':bound,'parameter_samples_sha256':sample_hash(model),'complete':True})
    local_bytes=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
    if local_bytes>estimate: raise RuntimeError('Measured checkpoint exceeds its reservation; preserve it and revise the disk profile before further writes')
    receipt=deferred_checkpoint_receipt(directory,bound) if defer_upload else store.commit(directory)
    state['checkpoint_bytes']=local_bytes;state['last_checkpoint']=receipt;state['last_checkpoint_local']=str(directory)
    emit('checkpoint_measured',bytes=local_bytes,write_seconds=time.monotonic()-started,receipt=receipt)
    rebind_assets(model,directory)
    if not defer_upload: prune(root,bound,[directory])
    return receipt

def reload_paths(directory,config=None):
    import zipfile
    directory=Path(directory); assets=directory/'reload-assets'
    cfg=config or (read(directory/'config.json') if (directory/'config.json').is_file() else {})
    if cfg.get('modalities','omni')=='vision_text' and not (assets/'motion-source.zip').exists():
        return {'spark':assets/'spark','vision':assets/'vision'}
    source=directory/'vendor'
    source.mkdir(exist_ok=True)
    archive=assets/'motion-source.zip'
    if sha(archive)!=read(HERE/'components.lock.json')['motion']['source_archive']['sha256']: raise ValueError('Saved motion source differs')
    with zipfile.ZipFile(archive) as z:
        for p in z.namelist():
            if not (source/p).resolve().is_relative_to(source.resolve()): raise ValueError('Unsafe source member')
        z.extractall(source); prefix=z.namelist()[0].split('/')[0]
    return {**{k:assets/k for k in ('spark','vision','speech_in','sound_in','speech_out','sound_out')},'motion_source':source/prefix,'motion':None}

def load_checkpoint(directory,config,bound,model=None,load_optimizer=True,device='cuda'):
    import torch
    from .model import SparkNative
    directory=Path(directory)
    if read(directory/'BINDING.json')!=bound: raise ValueError('Checkpoint source/data/policy binding differs')
    state=read(directory/'STATE.json')
    if model is None:
        eff_device=device if (device!='cuda' or torch.cuda.is_available()) else 'cpu'
        model=SparkNative(reload_paths(directory,config=config),config,device=eff_device,pretrained=False)
    model.set_stage(
        state.get('alignment_only',False),
        stage_mode=state.get('stage_mode'),
        unfreeze_vision_last_n_layers=int(state.get('unfreeze_vision_last_n_layers',0)),
    )
    has_tie=hasattr(getattr(model,'spark',None),'tie_weights')
    _load_model_safetensors(model, directory/'model.safetensors', strict=not has_tie, device='cpu')
    if has_tie: model.spark.tie_weights()
    if hasattr(model,'sync_host_vision_to_intel'):
        model.sync_host_vision_to_intel()
    # Rebuild any cached COAT FP8 weights immediately after loading model weights!
    coat_ctrl=ensure_coat_for_model(model,config)
    if coat_ctrl is not None:
        coat_ctrl.rebuild_weight_caches()
    rebind_assets(model,directory)
    if sample_hash(model)!=read(directory/'PARAMETERS.json')['sample_sha256']: raise ValueError('Reloaded parameter samples differ')
    eff_opt_cfg={**config,**(state.get('stage_opt_overrides') or {})}
    optimizer=optimizer_for(model,eff_opt_cfg)
    scheduler=_build_scheduler(
        optimizer,
        kind=state.get('scheduler_kind','cosine'),
        num_warmup_steps=state.get('warmup_steps',0),
        num_training_steps=max(1,state.get('planned_updates',1)),
    )
    if load_optimizer:
        optimizer.load_state_dict(torch.load(directory/'optimizer.pt',map_location='cpu',weights_only=True)); scheduler.load_state_dict(torch.load(directory/'scheduler.pt',map_location='cpu',weights_only=True))
        if (directory/'tier1b_optim.pt').is_file() and hasattr(model,'load_tier1b_state_dict'):
            model.load_tier1b_state_dict(torch.load(directory/'tier1b_optim.pt',map_location='cpu',weights_only=True))
        if (directory/'intel_vision_optim.pt').is_file() and hasattr(model,'load_intel_vision_optim_state_dict'):
            model.load_intel_vision_optim_state_dict(torch.load(directory/'intel_vision_optim.pt',map_location='cpu',weights_only=True))
    set_rng(torch.load(directory/'rng.pt',map_location='cpu',weights_only=True))
    return model,optimizer,scheduler,state

class MediaCollator:
    """Open independent catalogue/cache handles in each DataLoader process."""
    def __init__(self,campaign,dynamic_seed:int=0,row_overrides:dict|None=None):
        self.root=campaign.root; self.config=campaign.c; self.token=campaign._hf_token
        local_handoff=self.root/'metadata/HANDOFF.json'
        self.handoff=read(local_handoff) if local_handoff.is_file() else read(HERE.parent/'formal/HF_FULL_DATASET_HANDOFF.json')
        self.slot=campaign.reader.checkpoint_slot_bytes; self.worker=None
        self.dynamic_seed=int(dynamic_seed)
        self.row_overrides=dict(row_overrides) if row_overrides else {}
    def __call__(self,rows):
        import torch
        if self.worker is None:
            catalogue=Catalogue(self.root/'metadata/corpus.sqlite')
            reader=MediaReader(self.root,self.handoff,self.config,self.token,reserve=2*self.config['initial_checkpoint_reservation_bytes'])
            reader.checkpoint_slot_bytes=self.slot
            self.worker=(catalogue,reader)
        catalogue,reader=self.worker; examples=[]
        use_dyn=bool(self.config.get('dynamic_training_augmentation',False))
        for item in rows:
            row,packet=catalogue.row(item['rid'])
            with reader.packet(packet) as get: media=prepare_media(row,get,self.config)
            override=self.row_overrides.get(item['rid'])
            if override is not None:
                row=dict(row)
                row.update(override)
            elif use_dyn:
                aug_seed=int(identity([self.dynamic_seed,int(item['_index']),item['rid']])[:16],16)
                row,media=apply_dynamic_training_augmentation(row,media,random.Random(aug_seed),self.config)
            examples.append((row,media))
        # Accelerate inspects only the first mapping value for batch size.
        # Keep native PIL/array examples behind a tensor batch-size anchor.
        return {'source_indices':torch.tensor([item['_index'] for item in rows],dtype=torch.long),'examples':examples}


class Campaign:
    def __init__(self,config,paths,source_sha,token=None,key=None):
        self.c=config; self.root=Path(config['work_root']); self.paths=paths; self.bound=binding(config,source_sha)
        from .storage import ColdStore
        self.store=ColdStore(self.root,config,self.bound,token); self.catalogue=Catalogue(self.root/'metadata/corpus.sqlite',modalities=config.get('modalities','omni'))
        local_handoff=self.root/'metadata/HANDOFF.json'
        handoff_doc=read(local_handoff) if local_handoff.is_file() else read(HERE.parent/'formal/HF_FULL_DATASET_HANDOFF.json')
        self.reader=MediaReader(self.root,handoff_doc,config,token,reserve=2*config['initial_checkpoint_reservation_bytes'])
        self.key=key; self._hf_token=token; self.model=None; self.state={'stage':'initial','updates':0,'completed_stages':[],'rsi_rounds':[]}
        self.deferred_uploads=[];self.deferred_receipts={}
        self.budget=ComputeBudget(config,read(self.root/'compute-budget.json') if (self.root/'compute-budget.json').exists() else None)
        if config['observed_compute_balance'] and config['observed_cu_per_hour']: self.budget.observe(config['observed_compute_balance'],config['observed_cu_per_hour'])
        self.stop_reason=None; self.last_checkpoint=time.monotonic(); self.effective_checkpoint_minutes=config['checkpoint_minutes']
        self.direct_deadline=(time.monotonic()+float(config['direct_wall_hours'])*3600
                              if config.get('direct_wall_hours') else None)
        signal.signal(signal.SIGTERM,lambda *a:setattr(self,'stop_reason','SIGTERM'))
        signal.signal(signal.SIGINT,lambda *a:setattr(self,'stop_reason','operator interrupt'))
    def media(self,rid):
        row,x=self.catalogue.row(rid)
        with self.reader.packet(x) as get: media=prepare_media(row,get,self.c)
        return row,media
    def check_wall_budget(self,seconds=0):
        self.budget.tick(); limit=self.c.get('max_wall_hours',110)*3600
        remaining=limit-self.budget.state['active_seconds']
        if seconds>remaining:
            raise RuntimeError(f'The measured curriculum needs {seconds/3600:.2f} h but only {max(0,remaining)/3600:.2f} h remains of the {self.c.get("max_wall_hours",110)} h wall budget; no stage or epoch will be shortened')
        return remaining
    def budget_check(self,seconds=0):
        try:
            self.budget.check(seconds); self.check_wall_budget(seconds)
        except RuntimeError as exc: self.stop_reason=str(exc)
        write(self.root/'compute-budget.json',self.budget.state)
    def flush_deferred_uploads(self):
        for local in list(self.deferred_uploads):
            result=self.store.commit(Path(local['directory']))
            self.deferred_receipts[local['directory']]=result
            self.deferred_uploads.remove(local)
            if self.state.get('last_checkpoint_local')==local['directory']:
                self.state['last_checkpoint']=result
    def save(self,optimizer,scheduler,cursor=None,defer_upload=False):
        if not defer_upload: self.flush_deferred_uploads()
        if cursor: self.state['cursor']=cursor.state()
        self.budget_check(); write(self.root/'CAMPAIGN.json',self.state)
        result=checkpoint(self.model,optimizer,scheduler,self.state,self.root,self.c,self.bound,self.store,defer_upload=defer_upload)
        self.deferred_receipts[self.state['last_checkpoint_local']]=result
        if defer_upload and result.get('backend')=='local_deferred': self.deferred_uploads.append(result)
        write(self.root/'CAMPAIGN.json',self.state); self.last_checkpoint=time.monotonic(); return result
    def restore(self,receipt):
        directory=self.root/'checkpoints'/('restored-'+receipt['commit_sha256'][:12])
        prune(self.root,self.bound,[self.state.get('last_checkpoint_local')],incoming=directory)
        commit=self.store.restore(receipt,directory)
        self.model,optimizer,scheduler,self.state=load_checkpoint(directory,self.c,self.bound,model=self.model)
        self.state['last_checkpoint_local']=str(directory); self.state['last_checkpoint']=receipt
        for p in (directory/'provenance').glob('*.json'):
            if p.name not in ('QUALIFICATION.json','FULL_RECOVERY_PROOF.json','BASELINE.json','INSTRUCTOR_FORMAT_PROBE.json'): raise ValueError('Unexpected restored provenance')
            shutil.copy2(p,self.root/p.name)
        if (directory/'INSTRUCTOR_LEDGER.json').exists():
            p=self.root/'instructor/LEDGER.json'; p.parent.mkdir(exist_ok=True)
            current=read(p) if p.exists() else {}
            write(p,merge_instructor_ledgers(current,read(directory/'INSTRUCTOR_LEDGER.json')))
        if (directory/'COMPUTE_BUDGET.json').exists():
            # Replaying an older RSI parent must not rewind campaign spending.
            self.budget.merge_checkpoint(read(directory/'COMPUTE_BUDGET.json'))
            write(self.root/'compute-budget.json',self.budget.state)
        prune(self.root,self.bound,[directory])
        foreign=commit['runtime_boot_id']!=runtime_boot_id()
        proof={'schema_version':1,'passed':True,'fresh_vm':foreign,'parameter_samples_match':True,'optimizer_restored':True,'scheduler_restored':True,'rng_restored':True,'receipt':receipt,'binding':self.bound}
        write(self.root/'RESTORE_PROOF.json',proof); self.store.small_file(self.root/'RESTORE_PROOF.json','restore-proof')
        return optimizer,scheduler,proof

    def train_order_native(self,stage,order,resume=False,dynamic_seed:int=0,row_overrides:dict|None=None):
        """Native PyTorch APOLLO + COAT + merged-gradient training loop for 24 GB SM120 workstations and Colab."""
        import torch
        from torch.utils.data import DataLoader
        from .optimizer import MergedGradientAccumulator

        stage_mode=stage.get('stage_mode')
        unfreeze_vis=int(stage.get('unfreeze_vision_last_n_layers',0))
        is_frozen_spark=(
            bool(stage.get('alignment_only',False))
            or stage_mode in ('alignment_only','frozen_spark_connector','frozen_spark_cross_attn','vision_late_unfreeze','defrost_loop_engram')
        )
        stage_opt_overrides={
            k:stage[k]
            for k in (
                'learning_rate','backbone_lr','apollo_learning_rate',
                'connector_learning_rate','adapter_lr','apollo_connector_learning_rate',
                'vision_lr','vision_learning_rate',
                'engram_lr','engram_learning_rate',
                'warmup_ratio','clip_norm',
            )
            if k in stage
        }
        stage_cfg={**self.c,**stage_opt_overrides}
        self.model._intel_vision_base_lr=_lr_for_group_and_name('vision','vision.vision_model',stage_cfg)
        self.model.set_stage(
            stage.get('alignment_only',False),
            stage_mode=stage_mode,
            unfreeze_vision_last_n_layers=unfreeze_vis,
        )
        self.model.train()
        coat_ctrl=ensure_coat_for_model(self.model,stage_cfg)

        microbatch=self.c['microbatch_size']; accumulation=self.c['gradient_accumulation']//microbatch
        cursor=CommittedCursor(order,self.state.get('cursor') if resume else None)
        planned=math.ceil(len(order)/(microbatch*accumulation))
        remaining=math.ceil((len(order)-cursor.committed)/(microbatch*accumulation))
        if remaining==0:
            self.state['cursor']=None
            if stage['name'] not in self.state['completed_stages']: self.state['completed_stages'].append(stage['name'])
            return

        warmup=round(planned*float(stage_cfg['warmup_ratio']))
        optimizer=optimizer_for(self.model,stage_cfg)
        scheduler=_build_scheduler(optimizer,kind='cosine',num_warmup_steps=warmup,num_training_steps=planned)
        if stage.get('inherit_parent_optimizer') and not resume:
            lrs=[g['lr'] for g in optimizer.param_groups]
            saved=Path(self.state['last_checkpoint_local'])
            optimizer.load_state_dict(torch.load(saved/'optimizer.pt',map_location='cpu',weights_only=True))
            for group,lr in zip(optimizer.param_groups,lrs): group['lr']=lr; group['initial_lr']=lr
            if (saved/'tier1b_optim.pt').is_file() and hasattr(self.model,'load_tier1b_state_dict'):
                self.model.load_tier1b_state_dict(torch.load(saved/'tier1b_optim.pt',map_location='cpu',weights_only=True))
            if (saved/'intel_vision_optim.pt').is_file() and hasattr(self.model,'load_intel_vision_optim_state_dict'):
                self.model.load_intel_vision_optim_state_dict(torch.load(saved/'intel_vision_optim.pt',map_location='cpu',weights_only=True))
        if resume:
            saved=Path(self.state['last_checkpoint_local'])
            optimizer.load_state_dict(torch.load(saved/'optimizer.pt',map_location='cpu',weights_only=True))
            scheduler.load_state_dict(torch.load(saved/'scheduler.pt',map_location='cpu',weights_only=True))
            if (saved/'tier1b_optim.pt').is_file() and hasattr(self.model,'load_tier1b_state_dict'):
                self.model.load_tier1b_state_dict(torch.load(saved/'tier1b_optim.pt',map_location='cpu',weights_only=True))
            if (saved/'intel_vision_optim.pt').is_file() and hasattr(self.model,'load_intel_vision_optim_state_dict'):
                self.model.load_intel_vision_optim_state_dict(torch.load(saved/'intel_vision_optim.pt',map_location='cpu',weights_only=True))
            set_rng(torch.load(saved/'rng.pt',map_location='cpu',weights_only=True))

        self.state.update(
            stage=stage['name'],
            alignment_only=stage.get('alignment_only',False),
            stage_mode=stage_mode,
            unfreeze_vision_last_n_layers=unfreeze_vis,
            stage_opt_overrides=stage_opt_overrides,
            planned_updates=planned,
            warmup_steps=warmup,
            scheduler_kind='cosine',
            optimizer_profile=stage_cfg.get('optimizer_profile','apollo_coat'),
            order_sha256=cursor.order_sha256,
        )
        weights=stage.get('fixed_loss_weights') or loss_weights(self.catalogue,order,self.c)
        self.state['loss_weights']=weights
        offset=self.state.get('updates',0)

        buf_policy='fp32' if stage.get('alignment_only',False) else 'bf16_apollo_fp32_small'
        accumulator=MergedGradientAccumulator(
            self.model,
            accum_mode=buf_policy,
            min_bf16_accum_dim=int(self.c.get('apollo_min_dim',128)),
            allow_embeddings_in_apollo=bool(self.c.get('apollo_include_embeddings',False)),
        )
        norm_mode=str(self.c.get('merged_grad_norm_mode','per_example'))

        workers=int(self.c.get('dataloader_workers',0))
        loader_kwargs=(
            {
                'prefetch_factor':int(self.c.get('dataloader_prefetch',2)),
                'persistent_workers':False,
                'multiprocessing_context':'spawn',
            }
            if workers>0
            else {}
        )
        loader=DataLoader(
            cursor_dataset(order,cursor),
            batch_size=microbatch,
            num_workers=workers,
            collate_fn=MediaCollator(self,dynamic_seed=dynamic_seed,row_overrides=row_overrides),
            pin_memory=False,
            **loader_kwargs,
        )

        use_cuda_amp=torch.cuda.is_available() and str(self.model.device).startswith('cuda')
        amp_ctx=(lambda: torch.autocast('cuda',dtype=torch.bfloat16)) if use_cuda_amp else contextlib.nullcontext
        coat_ctx=(lambda: coat_ctrl.activation_context(training=True)) if (coat_ctrl is not None and coat_ctrl.enabled) else contextlib.nullcontext
        inv_fn=(coat_ctrl.invalidate_weight_caches if (coat_ctrl is not None and coat_ctrl.enabled) else None)

        epochs=max(1,int(stage.get('epochs',1)))
        eval_every_epoch=bool(self.c.get('eval_every_epoch',False)) and epochs>1 and len(order)>=epochs
        epoch_size=max(1,len(order)//epochs)
        evaluated_epoch=cursor.committed//epoch_size
        patience=int(stage.get('early_stopping_patience',self.c.get('early_stopping_patience',0)))
        min_rel_delta=float(stage.get('early_stopping_min_rel_delta',self.c.get('early_stopping_min_rel_delta',0.005)))
        best_dev_score=float('inf')
        best_epoch=0
        no_improve_epochs=0
        best_trainable_cpu=None
        early_stopped=False
        cur_epoch=evaluated_epoch
        stage_fams=set(stage.get('families',()))
        if stage.get('alignment_only',False):
            stage_fams.update(('image','video'))

        window_microbatches=0
        window_records=0
        window_tokens=0
        step_index=0
        exposure=Counter(self.state.get('exposure',{}))
        losses_by_stratum=defaultdict(lambda:[0.,0])

        from concurrent.futures import ThreadPoolExecutor
        use_vis_prefetch = (
            hasattr(self.model, 'has_intel_vision_bridge')
            and self.model.has_intel_vision_bridge()
            and getattr(self.model, '_intel_vision_trainable', False)
            and hasattr(self.model, 'prefetch_examples_vision')
        )
        prefetch_pool = ThreadPoolExecutor(max_workers=1) if use_vis_prefetch else None
        prefetched_next = None
        prefetch_fut = None

        try:
            batches_iter=iter(loader)
            while True:
                if prefetched_next is not None:
                    if prefetch_fut is not None:
                        prefetch_fut.result()
                        prefetch_fut = None
                    batch = prefetched_next
                    prefetched_next = None
                else:
                    try:
                        batch=next(batches_iter)
                    except StopIteration:
                        batch=None

                # Overlap `Microbatch m+1` CPU tiling + Intel Arc Pro B50 27L ViT forward behind
                # `Microbatch m` NVIDIA Blackwell 48L forward + backward (strictly within the same
                # accumulation window so vision weights never step between prefetch and consumption!).
                if (
                    batch is not None
                    and prefetch_pool is not None
                    and (window_microbatches + 1) < accumulation
                    and not self.stop_reason
                ):
                    try:
                        prefetched_next = next(batches_iter)
                    except StopIteration:
                        prefetched_next = None
                    if prefetched_next is not None:
                        prefetch_fut = prefetch_pool.submit(
                            self.model.prefetch_examples_vision, prefetched_next['examples']
                        )

                if batch is not None:
                    is_long_mb = use_cuda_amp and any(
                        len(str(rw.get('prompt', ''))) + len(str(rw.get('answer', ''))) > 8000
                        for rw, _ in batch['examples']
                    )
                    if is_long_mb:
                        torch.cuda.empty_cache()
                    with amp_ctx(), coat_ctx():
                        result=self.model(examples=batch['examples'])
                        per=result.get('per_example_loss')
                        if per is None or len(per)!=len(batch['examples']):
                            raise ValueError('Per-record loss count differs from the source microbatch')
                        weighted=[]
                        for raw_index,(row,_),loss in zip(batch['source_indices'],batch['examples'],per):
                            cursor.record_computed(int(raw_index))
                            window_records+=1
                            exposure[row['_family']]+=1
                            key=row['_family']+':'+row['language']
                            losses_by_stratum[key][0]+=float(loss.detach())
                            losses_by_stratum[key][1]+=1
                            weighted.append(loss*weights[key])
                        micro_loss_sum=torch.stack(weighted).sum()

                    if not torch.isfinite(micro_loss_sum):
                        accumulator.zero_buffers()
                        if hasattr(self.model,'zero_tier1b_buffers'): self.model.zero_tier1b_buffers()
                        raise RuntimeError('Nonfinite microbatch loss encountered; rolled back merged gradient buffer before step')

                    mb_units=max(1,int(result.get('supervised_units',len(batch['examples']))))
                    result=per=weighted=None
                    window_tokens+=mb_units
                    mb_weight=(mb_units/max(1,len(batch['examples']))) if norm_mode=='per_token' else 1.0
                    with accumulator.microbatch(weight=mb_weight,supervised_units=mb_units):
                        if micro_loss_sum.requires_grad:
                            micro_loss_sum.backward()
                        elif not is_frozen_spark or stage_mode=='defrost_loop_engram':
                            raise RuntimeError('Full-Spark or Loop-Engram microbatch loss is detached from autograd graph')
                    micro_loss_sum=None
                    if is_long_mb:
                        torch.cuda.empty_cache()
                    window_microbatches+=1

                should_step=(
                    (batch is not None and window_microbatches>=accumulation)
                    or (batch is None and window_microbatches>0)
                )
                if not should_step:
                    if batch is None:
                        break
                    continue

                if not is_frozen_spark:
                    spark_buf_norm=0.0
                    for group,_name,p in self.model.named_groups():
                        if group=='spark' and id(p) in accumulator.buffers:
                            spark_buf_norm+=float(accumulator.buffers[id(p)].detach().float().norm())
                    if spark_buf_norm<=0.0:
                        accumulator.zero_buffers()
                        if hasattr(self.model,'zero_tier1b_buffers'): self.model.zero_tier1b_buffers()
                        raise RuntimeError('Spark full-parameter gradient path is absent')
                elif stage_mode=='defrost_loop_engram':
                    eng_buf_norm=0.0
                    for group,_name,p in self.model.named_groups():
                        if group=='engram_mhc' and id(p) in accumulator.buffers:
                            eng_buf_norm+=float(accumulator.buffers[id(p)].detach().float().norm())
                    if eng_buf_norm<=0.0:
                        accumulator.zero_buffers()
                        if hasattr(self.model,'zero_tier1b_buffers'): self.model.zero_tier1b_buffers()
                        raise RuntimeError('Loop-Engram defrost gradient path is absent')

                denom=float(max(1,window_tokens if norm_mode=='per_token' else window_records))
                base_engram_lr=float(stage_cfg.get('engram_lr',stage_cfg.get('engram_learning_rate',stage_cfg.get('connector_learning_rate',1.2e-4))))
                lr_ratio=1.0
                if optimizer.param_groups:
                    g0=optimizer.param_groups[0]
                    if float(g0.get('initial_lr',0.0))>0.0:
                        lr_ratio=float(g0['lr'])/float(g0['initial_lr'])
                step_stats=accumulator.step(
                    optimizer,
                    scheduler,
                    max_grad_norm=float(stage_cfg['clip_norm']),
                    post_scale=1.0/denom,
                    invalidate_caches_fn=inv_fn,
                )
                if not step_stats.finite or not step_stats.applied:
                    if hasattr(self.model,'zero_tier1b_buffers'): self.model.zero_tier1b_buffers()
                    raise RuntimeError('Nonfinite merged gradient norm detected; optimizer step aborted cleanly')
                if hasattr(self.model,'step_tier1b_sparse_adam'):
                    self.model.step_tier1b_sparse_adam(lr=base_engram_lr*lr_ratio,denom=denom)

                # Commit cursor ONLY at completed optimizer step boundaries!
                cursor.commit(window_records)
                window_records=0
                window_tokens=0
                window_microbatches=0
                step_index+=1
                self.state.update(
                    updates=offset+step_index,
                    cursor=cursor.state(),
                    exposure=dict(exposure),
                    last_grad_norm=step_stats.grad_norm,
                )
                self.budget_check()
                if self.direct_deadline and time.monotonic()>self.direct_deadline:
                    self.stop_reason='direct_wall_hours'
                if step_index%10==0 or batch is None or self.stop_reason:
                    extra_gpu_kw={}
                    if hasattr(self.model,'has_intel_vision_bridge') and self.model.has_intel_vision_bridge():
                        tel=self.model.intel_vision_telemetry()
                        lst=tel.get('last_step') or {}
                        extra_gpu_kw['intel_arc_vision']={
                            'device':tel.get('device_name',tel.get('device','xpu:0')),
                            'trainable_params':tel.get('trainable_parameters',0),
                            'vision_grad_norm':round(float(lst.get('vision_grad_norm',0.0)),5),
                            'xpu_peak_gib':round(float(lst.get('xpu_peak_bytes',0))/(1024**3),3),
                        }
                    emit(
                        'training',
                        stage=stage['name'],
                        update=offset+step_index,
                        cursor=cursor.committed,
                        records=len(order),
                        grad_norm=round(step_stats.grad_norm,5),
                        raw_stratum_losses={k:v[0]/v[1] for k,v in losses_by_stratum.items() if v[1]>0},
                        **extra_gpu_kw,
                    )
                    losses_by_stratum.clear()

                cur_epoch=min(epochs,cursor.committed//epoch_size)
                if eval_every_epoch and cur_epoch>evaluated_epoch and not self.stop_reason:
                    evaluated_epoch=cur_epoch
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    saved_rng=rng_state()
                    ep_eval=self.evaluate('dev',f"{stage['name']}-ep{cur_epoch}",generate=False)
                    set_rng(saved_rng)
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    self.model.train()
                    by_st=ep_eval.get('by_stratum',{}) if isinstance(ep_eval,dict) else {}
                    def _rec_count(v): return int(v.get('records',v.get('n',0)))
                    active_items=[v for k,v in by_st.items() if k.split(':',1)[0] in stage_fams and _rec_count(v)>0]
                    if not active_items:
                        active_items=[v for v in by_st.values() if _rec_count(v)>0]
                    if active_items:
                        total_n=sum(_rec_count(v) for v in active_items)
                        dev_score=sum(float(v['mean_loss'])*_rec_count(v) for v in active_items)/max(1,total_n)
                        if dev_score<best_dev_score*(1.0-min_rel_delta):
                            best_dev_score=dev_score
                            best_epoch=cur_epoch
                            no_improve_epochs=0
                            if hasattr(self.model,'sync_intel_vision_to_host'):
                                self.model.sync_intel_vision_to_host()
                            offload_vis_tr=hasattr(self.model,'has_intel_vision_bridge') and self.model.has_intel_vision_bridge() and getattr(self.model,'_intel_vision_trainable',False)
                            best_trainable_cpu={
                                n:p.detach().cpu().clone()
                                for n,p in self.model.named_parameters()
                                if p.requires_grad or (offload_vis_tr and n.startswith('vision.'))
                            }
                        else:
                            no_improve_epochs+=1
                        ep_entry={
                            'epoch':cur_epoch,
                            'epochs':epochs,
                            'update':offset+step_index,
                            'dev_loss':round(dev_score,6),
                            'best_dev_loss':round(best_dev_score,6),
                            'best_epoch':best_epoch,
                            'no_improve_epochs':no_improve_epochs,
                        }
                        self.state.setdefault('stage_epoch_history',{}).setdefault(stage['name'],[]).append(ep_entry)
                        emit('stage_epoch_eval',stage=stage['name'],**ep_entry)
                        if patience>0 and no_improve_epochs>=patience and cur_epoch<epochs:
                            early_stopped=True
                            emit(
                                'stage_early_stopped',
                                stage=stage['name'],
                                stopped_at_epoch=cur_epoch,
                                best_epoch=best_epoch,
                                best_dev_loss=round(best_dev_score,6),
                            )
                            break

                if self.stop_reason or time.monotonic()-self.last_checkpoint>self.effective_checkpoint_minutes*60:
                    self.save(optimizer,scheduler,cursor)
                if self.stop_reason or batch is None:
                    break
        finally:
            if prefetch_fut is not None:
                try:
                    prefetch_fut.result()
                except Exception:
                    pass
            if prefetch_pool is not None:
                prefetch_pool.shutdown(wait=True)
            batches_iter=None
            loader=None
            accumulator.close()
            del accumulator
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        if not self.stop_reason and best_trainable_cpu is not None and (early_stopped or best_epoch<epochs):
            with torch.no_grad():
                param_map=dict(self.model.named_parameters())
                for name,cpu_t in best_trainable_cpu.items():
                    if name in param_map:
                        param_map[name].copy_(cpu_t.to(device=param_map[name].device,dtype=param_map[name].dtype))
                if hasattr(getattr(self.model,'spark',None),'tie_weights'):
                    self.model.spark.tie_weights()
                if hasattr(self.model,'sync_host_vision_to_intel'):
                    self.model.sync_host_vision_to_intel()
                if inv_fn is not None:
                    inv_fn()
            emit(
                'stage_best_weights_restored',
                stage=stage['name'],
                best_epoch=best_epoch,
                best_dev_loss=round(best_dev_score,6),
                early_stopped=early_stopped,
            )
        best_trainable_cpu=None

        if early_stopped:
            self.state['cursor']=None
            self.state.setdefault('early_stopped_stages',{})[stage['name']]={
                'stopped_at_epoch':cur_epoch,
                'best_epoch':best_epoch,
                'best_dev_loss':best_dev_score,
                'committed_records':cursor.committed,
                'planned_records':len(order),
            }
            self.save(optimizer,scheduler,cursor=None)
        else:
            self.save(optimizer,scheduler,cursor)
            if cursor.committed!=len(order):
                if self.stop_reason: return
                raise RuntimeError('Resumable stop: '+str(self.stop_reason or 'incomplete curriculum'))
            self.state['cursor']=None
        if stage['name'] not in self.state.setdefault('completed_stages',[]):
            self.state['completed_stages'].append(stage['name'])
        write(self.root/'CAMPAIGN.json',self.state)
        del optimizer,scheduler; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def train_order(self,stage,order,resume=False,dynamic_seed:int=0,row_overrides:dict|None=None):
        profile=self.c.get('optimizer_profile','adamw8bit')
        use_native=(
            self.c.get('trainer_backend')=='native_merged'
            or self.c.get('execution_target')=='local_workstation'
            or profile!='adamw8bit'
        )
        if use_native:
            return self.train_order_native(stage,order,resume=resume,dynamic_seed=dynamic_seed,row_overrides=row_overrides)

        import torch,unsloth
        import studio.backend.core.training.trainer as backend
        from transformers import TrainerCallback,get_cosine_schedule_with_warmup
        campaign=self; self.model.set_stage(stage.get('alignment_only',False)); self.model.train()
        microbatch=self.c['microbatch_size']; accumulation=self.c['gradient_accumulation']//microbatch
        cursor=CommittedCursor(order,self.state.get('cursor') if resume else None)
        planned=math.ceil(len(order)/(microbatch*accumulation)); remaining=math.ceil((len(order)-cursor.committed)/(microbatch*accumulation))
        if remaining==0:
            self.state['cursor']=None
            if stage['name'] not in self.state['completed_stages']: self.state['completed_stages'].append(stage['name'])
            return
        warmup=round(planned*self.c['warmup_ratio']); optimizer=optimizer_for(self.model,self.c)
        scheduler=get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=warmup,num_training_steps=planned)
        if stage.get('inherit_parent_optimizer') and not resume:
            lrs=[g['lr'] for g in optimizer.param_groups]
            saved=Path(self.state['last_checkpoint_local'])
            optimizer.load_state_dict(torch.load(saved/'optimizer.pt',map_location='cpu',weights_only=True))
            for group,lr in zip(optimizer.param_groups,lrs): group['lr']=lr; group['initial_lr']=lr
        if resume:
            saved=Path(self.state['last_checkpoint_local'])
            optimizer.load_state_dict(torch.load(saved/'optimizer.pt',map_location='cpu',weights_only=True)); scheduler.load_state_dict(torch.load(saved/'scheduler.pt',map_location='cpu',weights_only=True))
            set_rng(torch.load(saved/'rng.pt',map_location='cpu',weights_only=True))
        self.state.update(stage=stage['name'],alignment_only=stage.get('alignment_only',False),planned_updates=planned,warmup_steps=warmup,scheduler_kind='cosine')
        self.state['order_sha256']=cursor.order_sha256; processed={'count':0,'families':Counter(),'losses':defaultdict(lambda:[0.,0])}; offset=self.state.get('updates',0)
        weights=stage.get('fixed_loss_weights') or loss_weights(self.catalogue,order,self.c)
        self.state['loss_weights']=weights
        training_optim=(optimizer,scheduler)
        collate=MediaCollator(self)
        class NativeSFT(backend.SFTTrainer):
            def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
                result=model(examples=inputs['examples']); per=result.get('per_example_loss')
                if per is None or len(per)!=len(inputs['examples']): raise ValueError('Per-record loss count differs from the source microbatch')
                weighted=[]
                for raw_index,(row,_),loss in zip(inputs['source_indices'],inputs['examples'],per):
                    index=int(raw_index)
                    cursor.record_computed(index)
                    processed['count']+=1; processed['families'][row['_family']]+=1
                    key=row['_family']+':'+row['language']; bucket=processed['losses'][key]
                    bucket[0]+=float(loss.detach()); bucket[1]+=1
                    weighted.append(loss*weights[key])
                value=torch.stack(weighted).mean()
                return (value,{**result,'loss':value}) if return_outputs else value
            def _prepare_inputs(self,inputs): return inputs
        class Audit(TrainerCallback):
            def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
                norms=defaultdict(float)
                for group,name,p in model.named_groups():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all(): raise RuntimeError('Nonfinite gradient in '+group)
                        norms[group]+=float(p.grad.detach().float().norm())
                if not norms or (not stage.get('alignment_only') and norms.get('spark',0)<=0): raise RuntimeError('Spark full-parameter gradient path is absent')
            def on_step_end(self,args,state,control,**kwargs):
                cursor.commit(processed['count']); processed['count']=0
                campaign.state.update(updates=offset+state.global_step,cursor=cursor.state(),exposure=dict(processed['families']))
                campaign.budget_check()
                if campaign.direct_deadline and time.monotonic()>campaign.direct_deadline:
                    campaign.stop_reason='direct_wall_hours'
                if campaign.stop_reason or time.monotonic()-campaign.last_checkpoint>campaign.effective_checkpoint_minutes*60:
                    campaign.save(training_optim[0],training_optim[1],cursor)
                if campaign.stop_reason: control.should_training_stop=True
            def on_log(self,args,state,control,logs=None,**kwargs):
                if logs:
                    emit('training',stage=stage['name'],update=offset+state.global_step,cursor=cursor.committed,records=len(order),raw_stratum_losses={k:v[0]/v[1] for k,v in processed['losses'].items()},**{k:v for k,v in logs.items() if isinstance(v,(int,float))})
                    processed['losses'].clear()
        os.environ.update(UNSLOTH_MIXED_PRECISION='float32',ACCELERATE_MIXED_PRECISION='bf16')
        self.model._unsloth_full_finetuning=True
        self.model._unsloth_forced_float32=False; self.model._unsloth_user_float32=False
        self.model.config.torch_dtype=torch.float32
        workers=self.c['dataloader_workers']
        loader_options={'dataloader_prefetch_factor':self.c['dataloader_prefetch'],'dataloader_persistent_workers':True} if workers else {}
        args=backend.SFTConfig(output_dir=str(self.root/'trainer-scratch'),per_device_train_batch_size=microbatch,gradient_accumulation_steps=accumulation,max_steps=remaining,bf16=True,fp16=False,gradient_checkpointing=False,remove_unused_columns=False,max_length=self.c['max_sequence'],packing=False,accelerator_config={'dispatch_batches':False},dataset_kwargs={'skip_prepare_dataset':True},dataloader_num_workers=workers,dataloader_pin_memory=False,save_strategy='no',logging_steps=10,report_to='none',max_grad_norm=self.c['clip_norm'],warmup_ratio=0,seed=self.c['seed'],ignore_data_skip=True,lr_scheduler_type='cosine',**loader_options)
        class FormalStudio(backend.UnslothTrainer):
            def run_formal(self):
                self.model=campaign.model; self.tokenizer=campaign.model.tokenizer; self.load_in_4bit=False; self.is_vlm=False; self.is_audio=False
                self.trainer=NativeSFT(model=self.model,args=args,train_dataset=cursor_dataset(order,cursor),data_collator=collate,processing_class=self.tokenizer,optimizers=training_optim,callbacks=[Audit()])
                self.trainer.model_accepts_loss_kwargs=False
                if self.trainer.accelerator.mixed_precision!='bf16' or any(p.dtype!=torch.float32 for p in self.model.parameters()): raise RuntimeError('Studio altered the reviewed FP32-parameter/BF16-compute precision policy')
                self.is_training=True
                try: return self.trainer.train()
                finally: self.is_training=False
        bench=FormalStudio()
        write(self.root/'studio-backend.json',{'module':str(Path(backend.__file__).resolve()),'sha256':sha(backend.__file__),'sft_class':backend.SFTTrainer.__module__+'.'+backend.SFTTrainer.__name__,'unsloth':importlib.metadata.version('unsloth'),'integration':'actual headless UnslothTrainer owner and Studio-imported SFTTrainer; no UI submission claim'})
        bench.run_formal(); self.save(optimizer,scheduler,cursor)
        if cursor.committed!=len(order):
            if self.stop_reason: return
            raise RuntimeError('Resumable stop: '+str(self.stop_reason or 'incomplete curriculum'))
        self.state['cursor']=None; self.state.setdefault('completed_stages',[]).append(stage['name']); write(self.root/'CAMPAIGN.json',self.state)
        del bench,optimizer,scheduler; training_optim=None; gc.collect(); torch.cuda.empty_cache()

    def evaluate(self,split,name,generate=False):
        from .evaluate import evaluate
        self.budget_check()
        if self.stop_reason: raise RuntimeError(self.stop_reason)
        folder=self.root/'evaluations'/name
        report=evaluate(self.model,self.catalogue,self.reader,self.c,split,folder,generate=generate)
        receipt=self.store.commit(folder,kind='evaluation')
        self.state.setdefault('evaluations',{})[name]=receipt
        write(self.root/'CAMPAIGN.json',self.state)
        return report
    def sync_instructor_ledger(self):
        from colab.preflight import download_verified
        entries=self.store.discover(); latest=None; folder=self.root/'ledger-reconciliation'; folder.mkdir(exist_ok=True)
        for item in entries:
            name=item.get('name',''); kinds=('instructor-intent','instructor-receipt')
            if item.get('kind') not in kinds and not any(name.startswith(kind+'-') or '-'+kind+'-' in name for kind in kinds): continue
            commit=item.get('commit')
            if commit is None:
                target=folder/(item['commit_sha256']+'.json')
                if not target.exists(): download_verified(self.store.service,item['commit_file_id'],target,item['commit_sha256'])
                commit=read(target)
            if commit['binding']!=self.bound: continue
            if latest is None or commit['created_unix']>latest[0]: latest=(commit['created_unix'],item)
        if latest:
            destination=folder/('latest-'+latest[1]['commit_sha256'][:12]); self.store.restore(latest[1],destination)
            p=self.root/'instructor/LEDGER.json'; p.parent.mkdir(exist_ok=True)
            current=read(p) if p.exists() else {}
            write(p,merge_instructor_ledgers(current,read(destination/'LEDGER.json')))
    def instructed_rsi(self,round_no):
        import copy,torch
        from .teacher import Instructor,replay_orders
        from .evaluate import promotion
        self.sync_instructor_ledger(); session=self.state.get('orchestration')
        if session is None:
            master=copy.deepcopy(self.state); parent=master['last_checkpoint']
            pool=self.catalogue.select('train'); strata=defaultdict(list)
            for r in pool: strata[(r[1],r[2])].append(r[0])
            rng=random.Random(self.c['seed']+round_no); selected=[]
            for key in sorted(strata): rng.shuffle(strata[key])
            while len(selected)<self.c['instructor_records_per_round']:
                for key in sorted(strata):
                    if strata[key] and len(selected)<self.c['instructor_records_per_round']: selected.append(strata[key].pop())
            teacher=Instructor(self.key,self.root/'instructor',self.c,self.store); reviewed=[]
            self.model.eval()
            for rid in selected:
                row,media=self.media(rid)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    if row['_family'].endswith('_out'):
                        import numpy as np,soundfile as sf
                        inference=dict(media); inference['target_audio']=None
                        wav,sr=self.model.generate_audio(row,inference,seconds=15)
                        wav=np.asarray(wav)
                        if wav.ndim==2 and wav.shape[0]<=2: wav=wav.T
                        folder=self.root/'instructor/student-audio'; folder.mkdir(exist_ok=True)
                        file=folder/(identity([round_no,rid])[:24]+'.wav'); sf.write(file,wav,sr,subtype='FLOAT')
                        student={'native_audio_file':str(file),'native_audio_sha256':sha(file),'description':'Generated student waveform; compare to the source request. Voice identity need not match.'}
                    else: student=self.model.answer(row,media)
                row['_parent_checkpoint']=parent['commit_sha256']
                decision=teacher.review(row,media,student,round_no); reviewed.append((rid,decision))
            candidate,control=replay_orders(reviewed,self.c['seed']+round_no,self.c['rsi_epochs'])
            parent_eval=self.evaluate('promotion',f'rsi-{round_no}-parent')
            session={'round':round_no,'master':master,'parent_receipt':parent,'parent_eval':parent_eval,'candidate_order':candidate,'control_order':control,'loss_weights':loss_weights(self.catalogue,control,self.c),'phase':'candidate','reviews':reviewed}
            self.state['orchestration']=session
            write(self.root/'RSI_ORCHESTRATION.json',session); self.store.small_file(self.root/'RSI_ORCHESTRATION.json','rsi-orchestration')
        if session['round']!=round_no: raise ValueError('RSI round state differs')
        for phase in ('candidate','control'):
            if phase+'_receipt' in session: continue
            stage={'name':f'rsi-{round_no}-{phase}','alignment_only':False,'inherit_parent_optimizer':True,'fixed_loss_weights':session['loss_weights']}
            resume=self.state.get('stage')==stage['name'] and self.state.get('cursor') is not None
            if not resume:
                opt,sched,_=self.restore(session['parent_receipt']); del opt,sched
                parent_local=self.state['last_checkpoint_local']
                self.state=copy.deepcopy(session['master']); self.state['last_checkpoint_local']=parent_local; self.state['orchestration']=session
            self.train_order(stage,session[phase+'_order'],resume=resume)
            if self.stop_reason:
                self.state['orchestration']=session
                self.state.update(training_complete=False,stopped_reason=self.stop_reason)
                write(self.root/'CAMPAIGN.json',self.state); write(self.root/'RSI_ORCHESTRATION.json',session)
                return {'stopped':True,'reason':self.stop_reason,'stage':stage['name'],'round':round_no,
                        'last_checkpoint':self.state.get('last_checkpoint')}
            session[phase+'_receipt']=self.state['last_checkpoint']; session[phase+'_eval']=self.evaluate('promotion',f'rsi-{round_no}-{phase}')
            self.state['orchestration']=session
            write(self.root/'RSI_ORCHESTRATION.json',session); self.store.small_file(self.root/'RSI_ORCHESTRATION.json','rsi-orchestration')
        cp=promotion(session['parent_eval'],session['candidate_eval'],self.c); bp=promotion(session['parent_eval'],session['control_eval'],self.c)
        keys=session['candidate_eval']['by_stratum']; better=sum(session['candidate_eval']['by_stratum'][k]['mean_loss']<session['control_eval']['by_stratum'][k]['mean_loss'] for k in keys)
        candidate_mean=sum(session['candidate_eval']['by_stratum'][k]['mean_loss']/session['control_eval']['by_stratum'][k]['mean_loss'] for k in keys)/len(keys)
        choice='candidate' if cp['accepted'] and better and candidate_mean<1 else ('control' if bp['accepted'] else 'parent')
        receipt=session[choice+'_receipt']; opt,sched,_=self.restore(receipt)
        self.state.update(completed_stages=session['master']['completed_stages'],rsi_rounds=session['master'].get('rsi_rounds',[]),orchestration=None,stage=f'rsi-{round_no}-selected',cursor=None)
        result={'round':round_no,'selected':choice,'candidate_vs_parent':cp,'control_vs_parent':bp,'candidate_mean_relative_to_control':candidate_mean,'candidate_won_this_comparison':choice=='candidate','general_improvement_established':False,'scope':'Small controlled instructor-guided replay experiment; no architecture self-modification or broad autonomous-improvement claim'}
        self.state['rsi_rounds'].append(result); self.save(opt,sched); del opt,sched
        write(self.root/f'RSI-{round_no}.json',result); self.store.small_file(self.root/f'RSI-{round_no}.json','rsi-result')
        return result
    def vision_self_improving_rsi(self,round_no):
        from .vision_rsi import run_vision_self_improvement_round
        return run_vision_self_improvement_round(self,round_no)
    def teacher_subagent_rsi(self,round_no):
        from .vision_rsi import run_vision_self_improvement_round
        return run_vision_self_improvement_round(self,round_no)
    def qualify_instructor_format(self):
        from .teacher import Instructor
        import torch
        if (self.root/'INSTRUCTOR_FORMAT_PROBE.json').exists(): return
        self.sync_instructor_ledger(); teacher=Instructor(self.key,self.root/'instructor',self.c,self.store)
        self.model.eval(); reviewed=[]
        probe_pairs=[('image','en')] if self.c.get('modalities')=='vision_text' else [('image','en'),('speech_in','zh')]
        for fam,lang in probe_pairs:
            rowid=self.catalogue.select('train',[fam],lang)[0][0]
            row,media=self.media(rowid); row['_parent_checkpoint']=self.state.get('last_checkpoint',{}).get('commit_sha256','direct-initial')
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): student=self.model.answer(row,media,32)
            decision=teacher.review(row,media,student,0); reviewed.append({'id':rowid,'family':fam,'language':lang,'decision':decision})
        result={'passed':True,'requests':len(reviewed),'scope':'Current provider key/model and actual image/audio request schema/accounting only; teacher competence and route eligibility remain separate','reviewed_training_records':reviewed}
        write(self.root/'INSTRUCTOR_FORMAT_PROBE.json',result); self.store.small_file(self.root/'INSTRUCTOR_FORMAT_PROBE.json','instructor-format-probe')
    def run_formal(self,profile,require_recovery=True):
        from .teacher import eligibility
        if profile.get('binding')!=self.bound: raise ValueError('Qualification source/data/policy binding differs')
        eligibility(self.c)
        if require_recovery:
            proof=read(self.root/'FULL_RECOVERY_PROOF.json') if (self.root/'FULL_RECOVERY_PROOF.json').exists() else {}
            if (proof.get('binding')!=self.bound or not proof.get('passed') or not proof.get('fresh_process')
                    or not proof.get('complete_optimizer_recovery') or proof.get('native_updates')!=2
                    or not proof.get('initial_reset_complete') or not proof.get('initial_reset_state_sha256')
                    or proof.get('initial_reset_receipt')!=profile['qualification_initial_receipt']):
                raise RuntimeError('Actual new-process checkpoint continuation gate has not passed')
        if self.model is None: raise RuntimeError('Restore the reviewed initial or resumable campaign checkpoint first')
        slot=math.ceil(profile['complete_checkpoint_bytes']*1.10)
        self.reader.checkpoint_slot_bytes=slot; self.state['qualified_checkpoint_reservation_bytes']=slot
        transfer_minutes=math.ceil(profile['qualification_complete_receipt'].get('seconds',0)*1.2*4/60/10)*10
        self.effective_checkpoint_minutes=max(self.c['checkpoint_minutes'],transfer_minutes)
        if self.effective_checkpoint_minutes>self.c['max_checkpoint_minutes']: raise RuntimeError('Measured checkpoint transfer overhead exceeds the reviewed maximum interval')
        write(self.root/'STORAGE_PROFILE.json',{'checkpoint_backend':self.c.get('checkpoint_backend','huggingface'),'checkpoint_repo':self.c['checkpoint_repo'],'measured_checkpoint_bytes':profile['complete_checkpoint_bytes'],'slot_with_margin':slot,'effective_checkpoint_minutes':self.effective_checkpoint_minutes,'requested_checkpoint_minutes':self.c['checkpoint_minutes'],'transfer_overhead_minutes_requirement':transfer_minutes,'drive_small_artifacts_cap_bytes':self.c['drive_cap_bytes'],'local_retained_after_commit':1,'local_peak_checkpoint_slots':2,'remote_history_squashed':False})
        emit('storage_profile',**read(self.root/'STORAGE_PROFILE.json'))
        if not self.state.get('completed_stages') and self.state['stage']=='initial':
            self.budget.check(profile['projected_hours']*3600)
            self.check_wall_budget(profile['projected_hours']*3600)
        phase=self.c.get('run_phase','full')
        if phase not in ('full','train','rsi','audit'): raise ValueError('Expected run_phase full, train, rsi or audit')
        if phase in ('rsi','audit'):
            missing={s['name'] for s in self.c['stages']}-set(self.state.get('completed_stages',[]))
            if missing: raise RuntimeError('Full fine-tune stages are incomplete: '+','.join(sorted(missing)))
        is_vision_rsi=(self.c.get('rsi_mode') in ('vision_self_improvement','teacher_subagent_v8') or int(self.c.get('vision_rsi_rounds',0))>0)
        target_rsi_rounds=int(self.c.get('vision_rsi_rounds',0)) if is_vision_rsi else int(self.c.get('instructor_rounds',0))
        if phase=='audit' and len(self.state.get('rsi_rounds',[]))!=target_rsi_rounds:
            raise RuntimeError('All configured RSI rounds must complete before result checking')
        if not is_vision_rsi and (phase=='rsi' or (phase=='full' and int(self.c.get('instructor_rounds',0))>0)):
            self.qualify_instructor_format()
        if not (self.root/'BASELINE.json').exists():
            base=self.evaluate('dev','pretrained-native-baseline',generate=True); write(self.root/'BASELINE.json',base); self.store.small_file(self.root/'BASELINE.json','baseline')
        if phase in ('full','train'):
            for i,stage in enumerate(self.c['stages']):
                if stage['name'] in self.state.get('completed_stages',[]): continue
                stage_seed=int(self.c['seed'])+i
                if self.c.get('dynamic_shuffle',False):
                    seeds_map=self.state.setdefault('stage_shuffle_seeds',{})
                    if stage['name'] not in seeds_map:
                        seeds_map[stage['name']]=int((stage_seed^secrets.randbits(31))&0x7FFFFFFF)
                        write(self.root/'CAMPAIGN.json',self.state)
                    stage_seed=int(seeds_map[stage['name']])
                order=self.catalogue.plan(stage,stage_seed)
                resume=self.state.get('stage')==stage['name'] and self.state.get('cursor') is not None
                parent_receipt=self.state.get('last_checkpoint')
                self.train_order(stage,order,resume=resume,dynamic_seed=stage_seed)
                if self.stop_reason:
                    self.state.update(training_complete=False,stopped_reason=self.stop_reason)
                    write(self.root/'CAMPAIGN.json',self.state)
                    emit('campaign_stopped',reason=self.stop_reason,stage=stage['name'],
                         curriculum_complete=False,capability_promoted=False)
                    return {'stopped':True,'reason':self.stop_reason,'stage':stage['name'],
                            'last_checkpoint':self.state.get('last_checkpoint')}
                stage_eval=self.evaluate('dev',stage['name'],generate=False)
                # Best-checkpoint promotion guard: never silently overwrite a better parent checkpoint on regression.
                if self.c.get('enforce_stage_promotion_gate') and parent_receipt and isinstance(stage_eval,dict) and (self.root/'BASELINE.json').exists():
                    from .evaluate import promotion
                    prev_eval=self.state.get('best_dev_eval') or read(self.root/'BASELINE.json')
                    if isinstance(prev_eval,dict) and 'by_stratum' in prev_eval and 'by_stratum' in stage_eval:
                        gate=promotion(prev_eval,stage_eval,self.c)
                        if gate['accepted']:
                            self.state['best_checkpoint']=self.state.get('last_checkpoint')
                            self.state['best_dev_eval']=stage_eval
                        else:
                            emit('stage_checkpoint_rejected',stage=stage['name'],reasons=gate['reasons'],restoring_best=True)
                            opt,sched,_=self.restore(self.state.get('best_checkpoint') or parent_receipt)
                            del opt,sched
        if phase in ('full','rsi'):
            if is_vision_rsi:
                for round_no in range(1,target_rsi_rounds+1):
                    if any(r['round']==round_no for r in self.state.get('rsi_rounds',[])): continue
                    result=self.vision_self_improving_rsi(round_no)
                    if isinstance(result,dict) and result.get('stopped'):
                        emit('campaign_stopped',reason=self.stop_reason,stage=result['stage'],
                             curriculum_complete=False,capability_promoted=False)
                        return result
                if len(self.state.get('rsi_rounds',[]))!=target_rsi_rounds:
                    raise RuntimeError('Required vision-RSI rounds are incomplete')
            else:
                for round_no in range(1,self.c['instructor_rounds']+1):
                    if any(r['round']==round_no for r in self.state.get('rsi_rounds',[])): continue
                    result=self.instructed_rsi(round_no)
                    if isinstance(result,dict) and result.get('stopped'):
                        emit('campaign_stopped',reason=self.stop_reason,stage=result['stage'],
                             curriculum_complete=False,capability_promoted=False)
                        return result
                if len(self.state.get('rsi_rounds',[]))!=self.c['instructor_rounds']: raise RuntimeError('Required instructed-RSI comparisons are incomplete')
        if phase in ('full','audit'):
            self.evaluate('audit','final-protected-audit',generate=True)
            audio_review='not_applicable_vision_text' if self.c.get('modalities')=='vision_text' else 'pending'
            self.state.update(training_complete=True,final_audit_complete=True,human_audio_review=audio_review,capability_claims_verified=False,phase_result='audit_complete')
            write(self.root/'CAMPAIGN.json',self.state); self.store.small_file(self.root/'CAMPAIGN.json','campaign-completion')
            from .publish import export_and_upload
            return export_and_upload(self,self.c)
        self.state['phase_result']=phase+'_complete'
        write(self.root/'CAMPAIGN.json',self.state);self.store.small_file(self.root/'CAMPAIGN.json','phase-completion')
        return {'phase':self.state['phase_result'],'last_checkpoint':self.state.get('last_checkpoint')}
