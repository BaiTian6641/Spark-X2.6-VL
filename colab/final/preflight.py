"""CPU-first formal preflight; no GPU training or instructor API requests."""
import json,platform,shutil,time
from collections import Counter
from pathlib import Path
from .common import HERE,identity,read,write,emit,validate_config
from .assets import prepare_assets,prepare_data

def context_audit(root,config):
    from transformers import AutoTokenizer
    from .data import training_views,family
    import sqlite3
    tokenizer=AutoTokenizer.from_pretrained(Path(root)/'assets/spark',trust_remote_code=True,local_files_only=True)
    db=sqlite3.connect(Path(root)/'metadata/corpus.sqlite'); counts=Counter(); maximum=Counter(); failures=[]; original=0; total=0
    for body, in db.execute('SELECT body FROM packets ORDER BY pid'):
        packet=json.loads(body)['record']
        for row in packet.get('task_variants',[packet]):
            original+=1; views=training_views(row)
            for v in views:
                fam=family(v)
                prompt=tokenizer.apply_chat_template([{'role':'system','content':'You are Spark, an English and Chinese multimodal assistant. Use the supplied media and clocks. Report uncertainty when capture time or evidence is unknown.'},{'role':'user','content':v['prompt']}],tokenize=True,add_generation_prompt=True)
                answer=tokenizer(v['answer']+tokenizer.eos_token,add_special_tokens=False).input_ids
                # Conservative native-media token maxima, never actual allocation.
                media={'image':config['max_visual_tokens'],'video':config['max_visual_tokens']+config['motion_tokens'],'audiovisual':config['max_visual_tokens']+config['motion_tokens']+config['audio_tokens'],'speech_in':config['audio_tokens'],'sound_in':config['audio_tokens']}.get(fam,0)
                seq=len(prompt)+(0 if fam.endswith('_out') else len(answer))+media
                maximum[fam]=max(maximum[fam],seq); counts[fam]+=1; total+=1
                if seq>config['max_sequence'] or (not fam.endswith('_out') and len(answer)>config['max_answer_tokens']): failures.append({'id':v['id'],'family':fam,'sequence':seq,'answer_tokens':len(answer)})
                if fam=='speech_out' and (not v['metadata'].get('text') or v['metadata']['text'] not in v['prompt']): failures.append({'id':v['id'],'reason':'TTS utterance missing from prompt'})
    report={'schema_version':1,'passed':not failures,'original_instructions':original,'derived_training_views':total,'counts':dict(counts),'max_conservative_sequences':dict(maximum),'failures':failures[:100],'failure_count':len(failures),'adapter':'24-event complete score views; full waveform retained; source groups unchanged'}
    write(Path(root)/'CONTEXT_AUDIT.json',report)
    if failures: raise RuntimeError('Full-corpus context/role audit failed; no examples were silently truncated or dropped')
    return report

def run(config,token,source_sha):
    validate_config(config); root=Path(config['work_root']); root.mkdir(parents=True,exist_ok=True)
    if platform.python_version_tuple()[:2]!=('3','13'): raise RuntimeError('This reviewed environment is CPython3.13; do not install the CUDA lock on an unqualified Python version')
    prepare_assets(root,token,config,download_weights=False)
    data=prepare_data(root,token,config); audit=context_audit(root,config)
    from .engine import binding
    # engine imports no torch until a training/recovery method executes.
    from .storage import ColdStore
    store=ColdStore(root,config,binding(config,source_sha)); drive=store.service.about().get(fields='storageQuota').execute()['storageQuota']
    free=shutil.disk_usage(root).free
    report={'schema_version':1,'passed':True,'source_archive_sha256':source_sha,'dataset_split_sha256':data['split_membership_sha256'],'context':audit,'local_free_bytes':free,'drive_quota':drive,'gated_audio_model_access':True,'teacher_api_calls':0,'gpu_training_performed':False,'training_ready':False,'remaining_gates':['whole-model CUDA profile and two real native continuation updates after fresh-VM recovery','provider/account eligibility for OpenCode Go dataset review','observed Colab CU balance/rate and complete-curriculum budget fit'],'created_unix':time.time()}
    write(root/'PREFLIGHT.json',report); receipt=store.small_file(root/'PREFLIGHT.json','formal-preflight'); emit('cpu_preflight_complete',receipt=receipt,**{k:v for k,v in report.items() if k not in ('context','drive_quota')})
    return report
