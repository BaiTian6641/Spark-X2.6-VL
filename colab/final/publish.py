"""Private model publication with a narrow allowlist; optimizer stays on Drive."""
import gc,os,shutil
from pathlib import Path
from .common import HERE,read,write,sha,emit,identity
from .storage import manifest


def card(config,state,bound):
    lock=read(HERE/'components.lock.json'); dataset=read(HERE/'dataset.lock.json')
    rows='\n'.join(f"| {key} | [{value['repository']}](https://huggingface.co/{value['repository']}/tree/{value['revision']}) | {value.get('license','see source')} |" for key,value in lock.items() if key!='motion')
    return f'''---
language: [en, zh]
library_name: pytorch
pipeline_tag: any-to-any
license: other
license_name: composite-see-component-licenses
license_link: README.md
base_model: XHToken/Spark-X2.6-4B
datasets:
- TassP0lar/SparkX2.6-MM
private: true
---
# Spark X2.6 multimodal formal prototype

This custom PyTorch package contains a fully parameter-tuned Spark decoder with native image/video, speech/sound inputs and native waveform output branches. It requires the included runtime; `AutoModelForCausalLM` alone does not reconstruct this composite model. It is a private experimental artifact. Completion of the configured curriculum is distinct from broad capability acceptance. Human audio listening remains **{state.get('human_audio_review','pending')}**. See `evaluation-summary.json` for the actual measured results and limitations.

All configured formal stages completed: {state.get('completed_stages',[])}. Instructor-controlled rounds: {len(state.get('rsi_rounds',[]))}. A rejected RSI candidate is recorded and the selected parent/control is retained. No autonomous architecture self-modification is claimed.

| Branch | Pinned source | License |
|---|---|---|
{rows}
| Motion | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2/tree/{lock['motion']['revision']}) | MIT; archive license included |

Stable Audio Open 1.0 has separate Stability Community License terms. The composite is not relicensed wholesale under Apache-2.0. Original component cards/notices and V-JEPA source license are retained under `reload-assets`. Training sources have their own licenses; the [dataset card and attribution](https://huggingface.co/datasets/{dataset['repo_id']}/tree/{dataset['media_revision']}) remain authoritative. Internal source-held-out evaluations do not establish public-benchmark independence; pretraining overlap is unknown. Sound labels are weak, music is predominantly classical, AV acting and NASA time-lapse coverage are narrow.

To load after downloading this private repository with your own HF token:

```python
import sys
from pathlib import Path
package = Path('/content/downloaded-model')
sys.path.insert(0, str(package / 'runtime'))
from colab.final.inference import load, empty_media
model = load(package)
# Images / timestamped frames: populate media['images'] and media['times'].
# Text-to-speech: row['_family']='speech_out', row['utterance']=requested text,
# and include that exact text in row['prompt']; no reference waveform is used.
# Sound/music: row['_family']='sound_out'; model.generate_audio(row, empty_media(), seconds=15).
```

Install the included hashed dependency lock in an isolated CPython3.13 CUDA13.0 environment as described in `RUN.md`. The export contains model weights, preprocessing/configuration, exact runtime source and original tokenizer/codec infrastructure. Optimizer, scheduler, private instructor ledger, credentials and raw training media are excluded. Resumable training snapshots remain in the operator's Drive project.

Source release: `{bound['source_archive_sha256']}`. Dataset media revision: `{dataset['media_revision']}`. Metadata revision: `{dataset['catalogue_revision']}`.
'''

def export_and_upload(campaign,config):
    from huggingface_hub import HfApi
    state=campaign.state
    if not state.get('training_complete') or not state.get('final_audit_complete'): raise RuntimeError('Incomplete campaigns cannot be published as the final model')
    if set(s['name'] for s in config['stages'])-set(state['completed_stages']): raise ValueError('Missing full-parameter stage')
    source=Path(state['last_checkpoint_local']); out=campaign.root/'final-model'; out.mkdir(exist_ok=True)
    if (out/'MODEL_RELEASE.json').exists():
        release=read(out/'MODEL_RELEASE.json')
        if release['binding']!=campaign.bound or next(f['sha256'] for f in release['files'] if f['path']=='model.safetensors')!=sha(source/'model.safetensors'): raise ValueError('Existing closed export differs from the selected final checkpoint')
        return upload_closed_export(campaign,config,out)
    keep={'model.safetensors','BINDING.json','PARAMETERS.json'}
    for path in source.rglob('*'):
        if not path.is_file(): continue
        rel=path.relative_to(source)
        if rel.as_posix() not in keep and rel.parts[0] not in ('reload-assets','runtime'): continue
        dest=out/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            if sha(dest)!=sha(path): raise ValueError('Existing final export belongs to a different model')
        else: os.link(path,dest)  # Closed immutable weights; no extra 30GB staging copy.
    inference_policy={k:v for k,v in config.items() if k not in ('drive_project_folder','observed_compute_balance','observed_cu_per_hour','instructor_route_eligibility_receipt')}
    write(out/'config.json',inference_policy)
    summary={'training_complete':True,'full_parameter_stages':state['completed_stages'],'rsi_rounds':state['rsi_rounds'],'human_audio_review':state.get('human_audio_review'),'broad_capability_passed':False,'evaluations':list(state.get('evaluations',{})),'frozen_infrastructure':['Qwen speech tokenizer/codec','Stable Audio Oobleck autoencoder'],'source_binding':campaign.bound}
    write(out/'evaluation-summary.json',summary)
    reports=out/'evaluation'; reports.mkdir(exist_ok=True)
    for name in ('final-protected-audit','pretrained-native-baseline'):
        p=campaign.root/'evaluations'/name/'REPORT.json'
        if p.exists(): shutil.copy2(p,reports/(name+'.json'))
    (out/'README.md').write_text(card(config,state,campaign.bound))
    guide=(HERE/'RUN.md').read_text()
    design=out/'design'; design.mkdir(exist_ok=True)
    for name in ('COLAB_CAMPAIGN_PLAN.md','REFINEMENT_MULTIMODAL_REVIEW_2026-10-27.md'):
        src=HERE.parents[1]/'reports/spark_multimodal'/name
        if src.exists(): shutil.copy2(src,design/name)
        guide=guide.replace('../../reports/spark_multimodal/'+name,'design/'+name)
    (out/'RUN.md').write_text(guide)
    architecture={'kind':'SparkNative','inference_entrypoint':'runtime/colab/final/inference.py','vision':'SigLIP2 So400m NaFlex +0/2/4tiles +2x2 spatial downshuffle','motion':'V-JEPA2.1 base384, 16-frame adaptation','speech_input':'multilingual Whisper-small encoder','sound_input':'CLAP audio encoder','speech_output':'Qwen3-TTS0.6B pretrained native talker +16-group12.5Hz codec, Spark-state bridge','sound_output':'Stable Audio Open1.0 DiT v-prediction +Oobleck, Spark-state bridge','native_output':'generator losses backpropagate into Spark; no inference API'}
    write(out/'multimodal_architecture.json',architecture)
    # Scan all publishable text for the exact in-memory credentials, without
    # printing either value. Binary weights are produced only by safetensors.
    forbidden=('optimizer.pt','scheduler.pt','rng.pt','LEDGER.json','STATE.json','.ipynb')
    for path in out.rglob('*'):
        if not path.is_file(): continue
        if any(v in path.name for v in forbidden): raise ValueError('Private training artifact in final export')
        if path.suffix in ('.json','.md','.py','.txt','.lock'):
            payload=path.read_bytes()
            for secret in (campaign._hf_token,campaign.key):
                if secret and secret.encode() in payload: raise ValueError('Credential detected in final text artifact')
    files=manifest(out); write(out/'MODEL_RELEASE.json',{'schema_version':1,'campaign_id':config['campaign_id'],'binding':campaign.bound,'training_complete':True,**files})
    return upload_closed_export(campaign,config,out)

def upload_closed_export(campaign,config,out):
    from huggingface_hub import HfApi
    release=read(out/'MODEL_RELEASE.json')
    allowed=[f['path'] for f in release['files']]+['MODEL_RELEASE.json']
    # Verify a genuinely new object graph using only exported configs/weights.
    import torch
    from .inference import load
    original=campaign.model; campaign.model=None; del original; gc.collect(); torch.cuda.empty_cache()
    restored=load(out); emit('final_export_offline_reload',passed=True,parameters=sum(p.numel() for p in restored.parameters()))
    campaign.model=restored
    api=HfApi(token=campaign._hf_token)
    if api.whoami().get('name')!='TassP0lar': raise ValueError('Unexpected HF publishing account')
    exists=api.repo_exists(config['model_repo'],repo_type='model')
    if exists:
        info=api.model_info(config['model_repo'])
        if not info.private: raise RuntimeError('Existing destination is public; this reviewed release targets a private model repository')
        names=api.list_repo_files(config['model_repo'],repo_type='model')
        if names and not set(names)<={'.gitattributes'} and 'MODEL_RELEASE.json' not in names: raise RuntimeError('Existing model repository is unrelated; choose a new explicit destination')
        if set(names)-set(allowed)-{'.gitattributes'}: raise RuntimeError('Existing model repository contains unrelated files; use a clean model destination')
        if 'MODEL_RELEASE.json' in names:
            import requests
            session=requests.Session();session.trust_env=False
            try:
                response=session.get(f'https://huggingface.co/{config["model_repo"]}/resolve/{info.sha}/MODEL_RELEASE.json',headers={'Authorization':'Bearer '+campaign._hf_token},timeout=60)
                if response.status_code!=200 or len(response.content)>2*1024**2: raise RuntimeError('Could not verify existing model ownership binding')
                previous=response.json()
                if previous.get('campaign_id')!=config['campaign_id'] or previous.get('binding')!=campaign.bound: raise RuntimeError('Existing private model belongs to another campaign/source release')
            finally: session.close()
    else: api.create_repo(config['model_repo'],repo_type='model',private=True)
    result=api.upload_folder(repo_id=config['model_repo'],repo_type='model',folder_path=out,allow_patterns=allowed,ignore_patterns=['.cache/**'],commit_message='Publish completed formal native multimodal Spark prototype')
    info=api.model_info(config['model_repo'],revision=result.oid,files_metadata=True)
    remote={f.rfilename:f for f in info.siblings}
    expected=sha(out/'model.safetensors'); obj=remote.get('model.safetensors'); lfs=getattr(obj,'lfs',None)
    actual=getattr(lfs,'sha256',None) if lfs is not None else None
    if actual!=expected: raise RuntimeError('Final remote model SHA256 verification failed')
    receipt={'repo_id':config['model_repo'],'repo_type':'model','private':True,'commit':result.oid,'url':f'https://huggingface.co/{config["model_repo"]}/tree/{result.oid}','model_sha256':expected,'model_lfs_sha256_verified':True,'training_complete':True,'broad_capability_passed':False}
    write(campaign.root/'MODEL_UPLOAD_RECEIPT.json',receipt); campaign.store.small_file(campaign.root/'MODEL_UPLOAD_RECEIPT.json','model-publication'); emit('final_model_uploaded',**receipt)
    return receipt
