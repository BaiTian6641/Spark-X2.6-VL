"""CLI entry point. Secrets are received through stdin, never argv or files."""
from __future__ import annotations
import argparse,json,os,re,sys
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

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=('cpu','doctor','qualify','discover','recover','train','resume')); p.add_argument('--config',required=True);p.add_argument('--source-sha',required=True); p.add_argument('--profile-receipt');p.add_argument('--resume-receipt');a=p.parse_args()
    if not re.fullmatch('[0-9a-f]{64}',a.source_sha): raise ValueError('Source archive SHA256 is required')
    config=validate_config(read(a.config)); root=Path(config['work_root']); root.mkdir(parents=True,exist_ok=True)
    os.environ.update(HF_HOME=str(root/'hf-metadata-cache'),HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_XET='1',TORCH_HOME=str(root/'torch-metadata-cache'),WANDB_DISABLED='true',TOKENIZERS_PARALLELISM='false',UNSLOTH_COMPILE_DISABLE='1')
    secrets=json.loads(sys.stdin.readline()); token=secrets.get('HF_TOKEN'); key=secrets.get('OPENCODE_API_KEY')
    if not token: raise ValueError('HF_TOKEN is unavailable')
    if a.mode=='cpu':
        from .preflight import run
        run(config,token,a.source_sha); return
    if a.mode=='doctor': doctor(root);return
    from .engine import Campaign
    receipt=read(root/'asset-receipt.json') if (root/'asset-receipt.json').exists() else None
    if not receipt or not (root/'PREFLIGHT.json').exists(): raise RuntimeError('Run the CPU preflight on this VM before GPU work')
    paths={k:Path(v) for k,v in receipt['paths'].items()}
    if a.mode=='qualify':
        doctor(root)
        from .assets import prepare_assets
        paths=prepare_assets(root,token,config,download_weights=True)
    elif a.mode not in ('discover',): doctor(root)
    campaign=Campaign(config,paths,a.source_sha,token,key)
    if a.mode=='discover': emit('drive_artifacts',commits=campaign.store.discover()); return
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
        result=recovery_check(campaign,profile); emit('fresh_vm_recovery_passed',**result)
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
    try: main()
    except (Exception,KeyboardInterrupt) as exc:
        # No traceback containing response bodies, credentials or signed URLs.
        message=str(exc)
        if any(marker in message.lower() for marker in ('bearer ','hf_','sk-','token=')): message='Sensitive transport detail suppressed; inspect the sanitized local ledger/receipt.'
        emit('stopped',error_type=type(exc).__name__,reason=message[:1500],training_completion_claimed=False)
        raise SystemExit(2)
