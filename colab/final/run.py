"""CLI entry point. Secrets are received through stdin, never argv or files."""
from __future__ import annotations
import argparse,json,os,re,sys,fcntl
from pathlib import Path
from .common import HERE,read,write,emit,validate_config

def doctor(root):
    import importlib.metadata,subprocess
    import unsloth  # Must precede Transformers/Studio imports.
    import torch
    import studio.backend.core.training.trainer as backend
    from .common import sha
    if sys.version_info[:2]!=(3,13) or torch.__version__!='2.11.0+cu130' or torch.version.hip or not torch.cuda.is_available(): raise RuntimeError('Unqualified Python/PyTorch/CUDA runtime')
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('Native BF16 support is required')
    if torch.cuda.get_device_capability()!=(12,0): raise RuntimeError('GPU is not the reviewed SM120 Blackwell target')
    subprocess.run([sys.executable,'-m','pip','check'],check=True)
    # Real CUDA SDPA forward/backward sanity; this is not full-model proof.
    q=torch.randn(1,4,32,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    result=torch.nn.functional.scaled_dot_product_attention(q,q,q); result.float().square().mean().backward()
    if not torch.isfinite(q.grad).all() or not q.grad.abs().sum(): raise RuntimeError('CUDA attention gradient check failed')
    report={'python':sys.version.split()[0],'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),'bf16':True,'studio_backend_file':str(Path(backend.__file__).resolve()),'studio_backend_sha256':sha(backend.__file__),'versions':{k:importlib.metadata.version(k) for k in ('unsloth','unsloth-zoo','transformers','trl','accelerate','bitsandbytes','diffusers','qwen-tts')},'actual_cuda_sdpa_backward':True,'full_native_model_qualified':False}
    write(Path(root)/'ENVIRONMENT.json',report); emit('environment_ready',**report); return report

def direct_doctor(root):
    import importlib.metadata,subprocess
    import unsloth  # Must precede Transformers/Studio imports.
    import torch
    import studio.backend.core.training.trainer as backend
    from .common import sha
    if sys.version_info[:2]!=(3,13) or torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('Direct mode requires CPython3.13 and a CUDA GPU')
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('Native BF16 support is required')
    prop=torch.cuda.get_device_properties(0)
    if prop.total_memory < 90_000_000_000:
        raise RuntimeError(f'Full native multimodal training needs at least 90 GB GPU memory; found {prop.total_memory/2**30:.1f} GiB')
    subprocess.run([sys.executable,'-m','pip','check'],check=True)
    q=torch.randn(1,4,32,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    result=torch.nn.functional.scaled_dot_product_attention(q,q,q); result.float().square().mean().backward()
    if not torch.isfinite(q.grad).all() or not q.grad.abs().sum(): raise RuntimeError('CUDA attention gradient check failed')
    report={'python':sys.version.split()[0],'torch':torch.__version__,'cuda':torch.version.cuda,
            'gpu':prop.name,'gpu_bytes':prop.total_memory,'capability':list(torch.cuda.get_device_capability()),
            'bf16':True,'studio_backend_file':str(Path(backend.__file__).resolve()),
            'studio_backend_sha256':sha(backend.__file__),'direct_mode':True}
    write(Path(root)/'ENVIRONMENT.json',report); emit('environment_ready',**report); return report

DIRECT_QUALIFICATION_EXIT=10
DIRECT_SESSION_EXIT=3

def directory_bytes(root):
    seen=set(); total=0
    for path in Path(root).rglob('*'):
        if not path.is_file() or {'checkpoints','media-cache'} & set(path.relative_to(root).parts): continue
        stat=path.stat(); identity=(stat.st_dev,stat.st_ino)
        if identity not in seen:
            seen.add(identity); total+=stat.st_size
    return total

def storage_envelope(root,config,slot_bytes):
    """Conservative 230-GB envelope, including two slots and the 44-GB floor."""
    pending=2*slot_bytes+10*config['cache_bytes_per_component']+4_000_000_000+44_000_000_000
    used=directory_bytes(root); free=__import__('shutil').disk_usage(root).free
    if used+pending>230_000_000_000 or free<pending:
        raise RuntimeError(f'230-GB local envelope failed: used={used/1e9:.1f} GB, required={pending/1e9:.1f} GB, free={free/1e9:.1f} GB')
    emit('local_storage_envelope',used_gb=round(used/1e9,2),reserved_gb=round(pending/1e9,2),checkpoint_slot_gb=round(slot_bytes/1e9,2),capacity_gb=230)

def direct(config,token,key,source_sha):
    if not key: raise ValueError('OPENCODE_API_KEY is required: the campaign includes two instructor RSI rounds')
    from .teacher import eligibility
    eligibility(config)
    from .assets import prepare_assets,prepare_data
    from .common import ComputeBudget
    from .engine import Campaign
    from .qualify import qualify,recovery_check
    root=Path(config['work_root']); root.mkdir(parents=True,exist_ok=True)
    early=ComputeBudget(config,read(root/'compute-budget.json') if (root/'compute-budget.json').exists() else None)
    if config['observed_compute_balance'] and config['observed_cu_per_hour']:
        early.observe(config['observed_compute_balance'],config['observed_cu_per_hour'])
    write(root/'compute-budget.json',early.state)
    direct_doctor(root)
    emit('direct_assets_start',download_weights=True,full_dataset=True)
    paths=prepare_assets(root,token,config,download_weights=True)
    report=prepare_data(root,token,config)
    emit('direct_data_ready',instructions=report.get('instructions'),original_instructions=report.get('original_instructions'),split_membership_sha256=report.get('split_membership_sha256'))
    storage_envelope(root,config,config['initial_checkpoint_reservation_bytes'])
    campaign=Campaign(config,paths,source_sha,token,key); campaign.budget=early
    profile_path=root/'QUALIFICATION.json'
    if not profile_path.exists():
        profile=qualify(campaign)
        storage_envelope(root,config,__import__('math').ceil(profile['complete_checkpoint_bytes']*1.10))
        write(root/'compute-budget.json',early.state)
        emit('direct_qualification_complete',projected_hours=profile['projected_hours'],next_process='checkpoint recovery',exit_code=DIRECT_QUALIFICATION_EXIT)
        return DIRECT_QUALIFICATION_EXIT
    profile=read(profile_path)
    if not profile.get('passed') or profile.get('binding')!=campaign.bound:
        raise RuntimeError('Qualification belongs to a different source/data/policy; run a new campaign identity')
    storage_envelope(root,config,__import__('math').ceil(profile['complete_checkpoint_bytes']*1.10))
    if not (root/'FULL_RECOVERY_PROOF.json').exists():
        result=recovery_check(campaign,profile); emit('direct_recovery_complete',**result)
    state=read(root/'CAMPAIGN.json') if (root/'CAMPAIGN.json').exists() else {}
    selected=state.get('last_checkpoint') or profile['qualification_initial_receipt']
    optimizer,scheduler,_=campaign.restore(selected); del optimizer,scheduler
    emit('direct_campaign_starting',campaign=config['campaign_id'],stages=[s['name'] for s in config['stages']],instructor_rounds=config['instructor_rounds'],max_wall_hours=config['max_wall_hours'],curriculum_will_not_be_shortened=True)
    result=campaign.run_formal(profile)
    write(root/'compute-budget.json',early.state)
    if isinstance(result,dict) and result.get('stopped'):
        if result.get('reason')=='direct_wall_hours': return DIRECT_SESSION_EXIT
        raise RuntimeError('Campaign stopped before completion: '+str(result.get('reason')))
    return 0

def main(secrets=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=('cpu','doctor','direct','qualify','discover','upload-checkpoint','recover','train','resume')); p.add_argument('--config',required=True);p.add_argument('--source-sha',required=True); p.add_argument('--profile-receipt');p.add_argument('--resume-receipt');p.add_argument('--checkpoint-directory');a=p.parse_args()
    if not re.fullmatch('[0-9a-f]{64}',a.source_sha): raise ValueError('Source archive SHA256 is required')
    config=validate_config(read(a.config)); root=Path(config['work_root']); root.mkdir(parents=True,exist_ok=True)
    # Training, restore, retry-upload and cleanup share the same local slots.
    # A second cell/process must never reclaim files from a running campaign.
    with (root/'.campaign.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another campaign process is active; wait for it to exit before retrying, restoring or cleaning checkpoints') from None
        return dispatch(a,config,secrets)

def dispatch(a,config,secrets):
    root=Path(config['work_root'])
    os.environ.update(HF_HOME=str(root/'hf-metadata-cache'),HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_XET='1',TORCH_HOME=str(root/'torch-metadata-cache'),WANDB_DISABLED='true',TOKENIZERS_PARALLELISM='false',UNSLOTH_COMPILE_DISABLE='1')
    if secrets is None: secrets=json.loads(sys.stdin.readline())
    token=secrets.get('HF_TOKEN'); key=secrets.get('OPENCODE_API_KEY')
    if not token: raise ValueError('HF_TOKEN is unavailable')
    if a.mode=='cpu':
        from .preflight import run
        run(config,token,a.source_sha); return
    if a.mode=='doctor': doctor(root);return
    if a.mode=='direct':
        direct(config,token,key,a.source_sha); return
    from .engine import Campaign
    receipt=read(root/'asset-receipt.json') if (root/'asset-receipt.json').exists() else None
    preflight=read(root/'PREFLIGHT.json') if (root/'PREFLIGHT.json').exists() else {}
    if not receipt or not preflight.get('passed') or preflight.get('source_archive_sha256')!=a.source_sha:
        raise RuntimeError('Run a successful CPU preflight for this source release on this VM before GPU work')
    paths={k:Path(v) for k,v in receipt['paths'].items()}
    if a.mode=='qualify':
        doctor(root)
        from .assets import prepare_assets
        paths=prepare_assets(root,token,config,download_weights=True)
    elif a.mode not in ('discover','upload-checkpoint'): doctor(root)
    campaign=Campaign(config,paths,a.source_sha,token,key)
    if a.mode=='discover': emit('drive_artifacts',commits=campaign.store.discover()); return
    if a.mode=='upload-checkpoint':
        if not a.checkpoint_directory: raise ValueError('Select the exact closed local checkpoint directory')
        directory=Path(a.checkpoint_directory).resolve()
        if directory.parent!=(root/'checkpoints').resolve() or not (directory/'CLOSED.json').exists():
            raise ValueError('Only a closed direct child of the campaign checkpoint directory may be retried')
        receipt=campaign.store.commit(directory)
        from .checkpoint_cache import prune
        prune(root,campaign.bound,[directory])
        emit('checkpoint_upload_recovered',receipt=receipt,gpu_training_performed=False)
        return
    if a.mode=='qualify':
        from .qualify import qualify
        qualify(campaign); return
    profile=None
    if a.profile_receipt:
        receipt=read(a.profile_receipt); dest=root/'qualification-profile'
        campaign.store.restore(receipt,dest); profile=read(dest/'QUALIFICATION.json'); write(root/'QUALIFICATION.json',profile)
    elif (root/'QUALIFICATION.json').exists(): profile=read(root/'QUALIFICATION.json')
    if not profile or not profile.get('passed'): raise RuntimeError('The complete-model qualification profile is required')
    if a.mode in ('recover','train') and not (root/'FULL_RECOVERY_PROOF.json').exists():
        from .qualify import recovery_check
        result=recovery_check(campaign,profile); emit('checkpoint_recovery_passed',**result)
    if a.mode=='recover': return
    if a.mode=='resume':
        if not a.resume_receipt: raise ValueError('Select an exact committed checkpoint receipt')
        optimizer,scheduler,proof=campaign.restore(read(a.resume_receipt)); del optimizer,scheduler
        emit('resumable_checkpoint_loaded',proof=proof,stage=campaign.state['stage'])
    elif campaign.model is None:
        state=read(root/'CAMPAIGN.json') if (root/'CAMPAIGN.json').exists() else None
        selected=state.get('last_checkpoint') if state else profile['qualification_initial_receipt']
        optimizer,scheduler,_=campaign.restore(selected); del optimizer,scheduler
    emit('formal_campaign_starting',campaign=config['campaign_id'],projected_hours=profile['projected_hours'],curriculum_will_not_be_shortened=True)
    campaign.run_formal(profile)

if __name__=='__main__':
    secrets={}
    try:
        secrets=json.loads(sys.stdin.readline())
        result=main(secrets)
        if isinstance(result,int) and result: raise SystemExit(result)
    except (Exception,KeyboardInterrupt) as exc:
        # No traceback containing response bodies, credentials or signed URLs.
        from .launcher import sanitize
        message=sanitize(str(exc),secrets.values())
        emit('stopped',error_type=type(exc).__name__,reason=message[:1500],training_completion_claimed=False)
        raise SystemExit(2)
