"""Full-parameter campaign driven by the installed Unsloth Studio SFT backend."""
from __future__ import annotations
import contextlib, gc, importlib.metadata, math, os, random, shutil, signal, time
from collections import Counter,defaultdict
from pathlib import Path
from .common import HERE,canonical,identity,read,write,sha,disk_guard,emit,ComputeBudget
from .data import Catalogue,CommittedCursor,MediaReader,prepare_media,FAMILIES,cursor_rows


def rng_state():
    import numpy as np,torch
    s=np.random.get_state()
    return {'python':random.getstate(),'numpy':[s[0],s[1].tolist(),s[2],s[3],s[4]],'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def set_rng(s):
    import numpy as np,torch
    random.setstate(s['python']); n=s['numpy']; np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4])); torch.set_rng_state(s['torch']); torch.cuda.set_rng_state_all(s['cuda'])
def sample_parameters(model):
    return {n:p.detach().flatten()[:16].float().cpu().tolist() for n,p in model.named_parameters()}
def sample_hash(model): return identity(sample_parameters(model))
def census(model):
    groups=defaultdict(lambda:{'parameters':0,'trainable':0})
    for group,name,p in model.named_groups(): groups[group]['parameters']+=p.numel(); groups[group]['trainable']+=p.numel() if p.requires_grad else 0
    return dict(groups)

def binding(config,source_sha):
    stable={k:v for k,v in config.items() if k not in ('observed_compute_balance','observed_cu_per_hour','instructor_route_eligibility_receipt')}
    return {'source_archive_sha256':source_sha,'component_lock_sha256':sha(HERE/'components.lock.json'),'dataset_lock_sha256':sha(HERE/'dataset.lock.json'),'policy_sha256':identity(stable)}

def loss_weights(catalogue,order,config):
    classes={r[0]:(r[1]+':'+r[2]) for r in catalogue.select('train')}
    counts=Counter(classes[rid] for rid in order); n=len(order)
    return {key:min(config['max_loss_weight'],n/(len(counts)*count)) for key,count in counts.items()}

def optimizer_for(model,config,alignment=False):
    import bitsandbytes as bnb
    buckets=defaultdict(list); seen=set()
    for group,name,p in model.named_groups():
        if not p.requires_grad or id(p) in seen: continue
        seen.add(id(p)); lr=config['connector_learning_rate'] if group=='connectors' or '.bridge.' in name or 'default_voice' in name else (config['audio_output_learning_rate'] if group in ('speech_output','sound_output') else config['learning_rate'])
        buckets[lr].append(p)
    optimizer=bnb.optim.AdamW8bit([{'params':v,'lr':lr,'weight_decay':config['weight_decay']} for lr,v in buckets.items()],betas=(.9,.95),eps=1e-8)
    expected={id(p) for p in model.parameters() if p.requires_grad}
    actual={id(p) for g in optimizer.param_groups for p in g['params']}
    if actual!=expected: raise AssertionError('Optimizer omits or adds trainable parameters')
    return optimizer

def copy_reload_assets(model,destination):
    destination=Path(destination); out=destination/'reload-assets'
    for key,raw in model.paths.items():
        if key in ('motion','motion_source'): continue
        root=Path(raw)
        for path in root.rglob('*'):
            if not path.is_file(): continue
            rel=path.relative_to(root)
            weight=path.suffix in ('.safetensors','.bin','.pt')
            if weight and not (key=='speech_out' and str(rel).startswith('speech_tokenizer/')): continue
            dest=out/key/rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,dest)
    vendor=out/'motion-source.zip'; vendor.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(HERE/'vendor/vjepa2-source.zip',vendor)
    # Every saved package carries the reviewed inference/runtime source and lock.
    runtime=destination/'runtime'
    release=HERE.parents[1]
    include=[p for p in (release/'colab').rglob('*.py') if '__pycache__' not in p.parts and (p.parent==HERE or p.name=='__init__.py' or p.relative_to(release).as_posix() in ('colab/preflight.py','colab/stage1/common.py','colab/stage1/drive_checkpoint.py','colab/formal/rolling_cache.py','colab/formal/shard_dataset.py','colab/formal/local_storage.py'))]
    include += [HERE/'components.lock.json',HERE/'dataset.lock.json',HERE/'requirements.lock',HERE.parent/'formal/HF_FULL_DATASET_HANDOFF.json']
    for src in include:
        dest=runtime/src.relative_to(release); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dest)

def checkpoint(model,optimizer,scheduler,state,root,config,bound,store):
    import torch
    from safetensors.torch import save_model
    root=Path(root); name=f"{state['stage']}-{state.get('updates',0):08d}"; directory=root/'checkpoints'/name
    if directory.exists():
        names={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
        required={'model.safetensors','optimizer.pt','scheduler.pt','rng.pt','STATE.json','config.json','BINDING.json','PARAMETERS.json','CLOSED.json'}
        if required<=names:
            if read(directory/'BINDING.json')!=bound or read(directory/'PARAMETERS.json')['sample_sha256']!=sample_hash(model): raise ValueError('Checkpoint name collides with a changed model/source')
            receipt_path=directory.with_name(directory.name+'.drive-receipt.json')
            result=read(receipt_path) if receipt_path.exists() else store.commit(directory)
            state.update(last_checkpoint=result,last_checkpoint_local=str(directory),checkpoint_bytes=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()))
            return result
        intent=directory/'INTENT.json'
        if not intent.exists() or read(intent).get('binding')!=bound: raise RuntimeError('Unknown incomplete local artifact; no automatic overwrite')
        # A writer-owned snapshot lacking required optimizer/state files cannot
        # be resumed. The current in-memory optimizer state is still authoritative.
        shutil.rmtree(directory)
    # Reserve the second complete slot before writing a new closed snapshot.
    estimate=max(state.get('qualified_checkpoint_reservation_bytes',config['initial_checkpoint_reservation_bytes']),int(state.get('checkpoint_bytes',0)))
    existing=[p for p in (root/'checkpoints').glob('*') if p.is_dir()]
    if len(existing)>=2:
        for old in sorted(existing,key=lambda p:p.stat().st_mtime)[:-1]:
            if not old.with_name(old.name+'.drive-receipt.json').exists() or read(old/'BINDING.json')!=bound: raise RuntimeError('Uncommitted or foreign local checkpoint occupies a slot')
            shutil.rmtree(old)
        existing=[p for p in (root/'checkpoints').glob('*') if p.is_dir()]
    disk_guard(root,pending=estimate,reserve=estimate if not existing else 0)
    directory.mkdir(parents=True); write(directory/'INTENT.json',{'binding':bound,'stage':state['stage'],'updates':state.get('updates',0)}); started=time.monotonic()
    torch.cuda.synchronize(); save_model(model,str(directory/'model.safetensors'))
    torch.save(optimizer.state_dict(),directory/'optimizer.pt'); torch.save(scheduler.state_dict(),directory/'scheduler.pt'); torch.save(rng_state(),directory/'rng.pt')
    write(directory/'STATE.json',state); write(directory/'config.json',config); write(directory/'BINDING.json',bound); write(directory/'PARAMETERS.json',{'sample_sha256':sample_hash(model),'census':census(model)})
    copy_reload_assets(model,directory)
    for name in ('corpus.report.json','BUILD.json'):
        src=root/'metadata'/name
        if src.exists(): shutil.copy2(src,directory/name)
    for name in ('QUALIFICATION.json','FULL_RECOVERY_PROOF.json','BASELINE.json','INSTRUCTOR_FORMAT_PROBE.json'):
        if (root/name).exists():
            dest=directory/'provenance'/name; dest.parent.mkdir(exist_ok=True); shutil.copy2(root/name,dest)
    if (root/'instructor/LEDGER.json').exists(): shutil.copy2(root/'instructor/LEDGER.json',directory/'INSTRUCTOR_LEDGER.json')
    if (root/'compute-budget.json').exists(): shutil.copy2(root/'compute-budget.json',directory/'COMPUTE_BUDGET.json')
    write(directory/'CLOSED.json',{'binding':bound,'parameter_samples_sha256':sample_hash(model),'complete':True})
    local_bytes=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
    if local_bytes>estimate: raise RuntimeError('Measured checkpoint exceeds its reservation; preserve it and revise the disk profile before further writes')
    receipt=store.commit(directory); state['checkpoint_bytes']=local_bytes; state['last_checkpoint']=receipt; state['last_checkpoint_local']=str(directory)
    emit('checkpoint_measured',bytes=local_bytes,write_seconds=time.monotonic()-started,receipt=receipt)
    # Reclaim only an older, campaign-owned, hash-verified durable local snapshot.
    for old in existing:
        receipt_path=old.with_name(old.name+'.drive-receipt.json')
        if not receipt_path.exists(): raise RuntimeError('Refusing to reclaim a checkpoint without a durable receipt')
        old_binding=read(old/'BINDING.json')
        if old_binding!=bound: raise RuntimeError('Refusing to reclaim another campaign snapshot')
        shutil.rmtree(old)
    return receipt

def reload_paths(directory):
    import zipfile
    directory=Path(directory); assets=directory/'reload-assets'; source=directory/'vendor'
    source.mkdir(exist_ok=True)
    archive=assets/'motion-source.zip'
    if sha(archive)!=read(HERE/'components.lock.json')['motion']['source_archive']['sha256']: raise ValueError('Saved motion source differs')
    with zipfile.ZipFile(archive) as z:
        for p in z.namelist():
            if not (source/p).resolve().is_relative_to(source.resolve()): raise ValueError('Unsafe source member')
        z.extractall(source); prefix=z.namelist()[0].split('/')[0]
    return {**{k:assets/k for k in ('spark','vision','speech_in','sound_in','speech_out','sound_out')},'motion_source':source/prefix,'motion':None}

def load_checkpoint(directory,config,bound,model=None,load_optimizer=True):
    import torch
    from safetensors.torch import load_model
    from .model import SparkNative
    directory=Path(directory)
    if read(directory/'BINDING.json')!=bound: raise ValueError('Checkpoint source/data/policy binding differs')
    state=read(directory/'STATE.json')
    if model is None: model=SparkNative(reload_paths(directory),config,pretrained=False)
    model.set_stage(state.get('alignment_only',False)); load_model(model,str(directory/'model.safetensors'),strict=True,device='cpu')
    if sample_hash(model)!=read(directory/'PARAMETERS.json')['sample_sha256']: raise ValueError('Reloaded parameter samples differ')
    optimizer=optimizer_for(model,config)
    from transformers import get_cosine_schedule_with_warmup
    if state.get('scheduler_kind')=='constant':
        from transformers import get_constant_schedule
        scheduler=get_constant_schedule(optimizer)
    else: scheduler=get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=state.get('warmup_steps',0),num_training_steps=max(1,state.get('planned_updates',1)))
    if load_optimizer:
        optimizer.load_state_dict(torch.load(directory/'optimizer.pt',map_location='cpu',weights_only=True)); scheduler.load_state_dict(torch.load(directory/'scheduler.pt',map_location='cpu',weights_only=True))
    set_rng(torch.load(directory/'rng.pt',map_location='cpu',weights_only=True))
    return model,optimizer,scheduler,state

class Campaign:
    def __init__(self,config,paths,source_sha,token=None,key=None):
        self.c=config; self.root=Path(config['work_root']); self.paths=paths; self.bound=binding(config,source_sha)
        from .storage import ColdStore
        self.store=ColdStore(self.root,config,self.bound); self.catalogue=Catalogue(self.root/'metadata/corpus.sqlite')
        self.reader=MediaReader(self.root,read(HERE.parent/'formal/HF_FULL_DATASET_HANDOFF.json'),config,token,reserve=2*config['initial_checkpoint_reservation_bytes'])
        self.key=key; self._hf_token=token; self.model=None; self.state={'stage':'initial','updates':0,'completed_stages':[],'rsi_rounds':[]}
        self.budget=ComputeBudget(config,read(self.root/'compute-budget.json') if (self.root/'compute-budget.json').exists() else None)
        if config['observed_compute_balance'] and config['observed_cu_per_hour']: self.budget.observe(config['observed_compute_balance'],config['observed_cu_per_hour'])
        self.stop_reason=None; self.last_checkpoint=time.monotonic(); self.effective_checkpoint_minutes=config['checkpoint_minutes']
        signal.signal(signal.SIGTERM,lambda *a:setattr(self,'stop_reason','SIGTERM'))
        signal.signal(signal.SIGINT,lambda *a:setattr(self,'stop_reason','operator interrupt'))
    def media(self,rid):
        row,x=self.catalogue.row(rid)
        with self.reader.packet(x) as get: media=prepare_media(row,get,self.c)
        return row,media
    def budget_check(self,seconds=0):
        try: self.budget.check(seconds)
        except RuntimeError as exc: self.stop_reason=str(exc)
        write(self.root/'compute-budget.json',self.budget.state)
    def save(self,optimizer,scheduler,cursor=None):
        if cursor: self.state['cursor']=cursor.state()
        self.budget_check(); write(self.root/'CAMPAIGN.json',self.state)
        result=checkpoint(self.model,optimizer,scheduler,self.state,self.root,self.c,self.bound,self.store)
        write(self.root/'CAMPAIGN.json',self.state); self.last_checkpoint=time.monotonic(); return result
    def restore(self,receipt):
        directory=self.root/'checkpoints'/('restored-'+receipt['commit_sha256'][:12])
        commit=self.store.restore(receipt,directory)
        self.model,optimizer,scheduler,self.state=load_checkpoint(directory,self.c,self.bound,model=self.model)
        self.state['last_checkpoint_local']=str(directory); self.state['last_checkpoint']=receipt
        for p in (directory/'provenance').glob('*.json'):
            if p.name not in ('QUALIFICATION.json','FULL_RECOVERY_PROOF.json','BASELINE.json','INSTRUCTOR_FORMAT_PROBE.json'): raise ValueError('Unexpected restored provenance')
            shutil.copy2(p,self.root/p.name)
        if (directory/'INSTRUCTOR_LEDGER.json').exists():
            p=self.root/'instructor/LEDGER.json'; p.parent.mkdir(exist_ok=True); shutil.copy2(directory/'INSTRUCTOR_LEDGER.json',p)
        if (directory/'COMPUTE_BUDGET.json').exists():
            self.budget=ComputeBudget(self.c,read(directory/'COMPUTE_BUDGET.json')); self.budget.observe(self.c['observed_compute_balance'],self.c['observed_cu_per_hour'])
        foreign=commit['runtime_boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        proof={'schema_version':1,'passed':True,'fresh_vm':foreign,'parameter_samples_match':True,'optimizer_restored':True,'scheduler_restored':True,'rng_restored':True,'receipt':receipt,'binding':self.bound}
        write(self.root/'RESTORE_PROOF.json',proof); self.store.small_file(self.root/'RESTORE_PROOF.json','restore-proof')
        return optimizer,scheduler,proof
    def train_order(self,stage,order,resume=False):
        import torch,unsloth
        import studio.backend.core.training.trainer as backend
        from torch.utils.data import IterableDataset
        from transformers import TrainerCallback,get_cosine_schedule_with_warmup
        campaign=self; self.model.set_stage(stage.get('alignment_only',False)); self.model.train()
        cursor=CommittedCursor(order,self.state.get('cursor') if resume else None)
        planned=math.ceil(len(order)/self.c['gradient_accumulation']); remaining=math.ceil((len(order)-cursor.committed)/self.c['gradient_accumulation'])
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
        class Stream(IterableDataset):
            def __iter__(self):
                yield from cursor_rows(order,cursor)
        def collate(rows): return {'examples':[campaign.media(r['rid']) for r in rows],'source_indices':[r['_index'] for r in rows]}
        class NativeSFT(backend.SFTTrainer):
            def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
                result=model(examples=inputs['examples'])
                for index in inputs['source_indices']: cursor.record_computed(index)
                processed['count']+=len(inputs['examples'])
                for row,_ in inputs['examples']:
                    processed['families'][row['_family']]+=1
                    key=row['_family']+':'+row['language']; bucket=processed['losses'][key]
                    bucket[0]+=float(result['loss'].detach());bucket[1]+=1
                weighted=result['loss']*weights[key]
                return (weighted,result) if return_outputs else weighted
            def _prepare_inputs(self,inputs): return inputs  # PIL/arrays are intentionally CPU until the native router.
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
                if campaign.stop_reason or time.monotonic()-campaign.last_checkpoint>campaign.effective_checkpoint_minutes*60:
                    campaign.save(training_optim[0],training_optim[1],cursor)
                if campaign.stop_reason: control.should_training_stop=True
            def on_log(self,args,state,control,logs=None,**kwargs):
                if logs:
                    emit('training',stage=stage['name'],update=offset+state.global_step,cursor=cursor.committed,records=len(order),raw_stratum_losses={k:v[0]/v[1] for k,v in processed['losses'].items()},**{k:v for k,v in logs.items() if isinstance(v,(int,float))})
                    processed['losses'].clear()
        # Unsloth's generated SFT config must retain BF16 AMP with FP32 params.
        os.environ.update(UNSLOTH_MIXED_PRECISION='float32',ACCELERATE_MIXED_PRECISION='bf16')
        self.model._unsloth_full_finetuning=True
        self.model._unsloth_forced_float32=False; self.model._unsloth_user_float32=False
        self.model.config.torch_dtype=torch.float32
        args=backend.SFTConfig(output_dir=str(self.root/'trainer-scratch'),per_device_train_batch_size=1,gradient_accumulation_steps=self.c['gradient_accumulation'],max_steps=remaining,bf16=True,fp16=False,gradient_checkpointing=False,remove_unused_columns=False,max_length=self.c['max_sequence'],packing=False,dataset_kwargs={'skip_prepare_dataset':True},dataloader_num_workers=0,dataloader_pin_memory=False,save_strategy='no',logging_steps=10,report_to='none',max_grad_norm=self.c['clip_norm'],warmup_ratio=0,seed=self.c['seed'],ignore_data_skip=True,lr_scheduler_type='cosine')
        class FormalStudio(backend.UnslothTrainer):
            def run_formal(self):
                self.model=campaign.model; self.tokenizer=campaign.model.tokenizer; self.load_in_4bit=False; self.is_vlm=False; self.is_audio=False
                self.trainer=NativeSFT(model=self.model,args=args,train_dataset=Stream(),data_collator=collate,processing_class=self.tokenizer,optimizers=training_optim,callbacks=[Audit()])
                self.trainer.model_accepts_loss_kwargs=False
                if self.trainer.accelerator.mixed_precision!='bf16' or any(p.dtype!=torch.float32 for p in self.model.parameters()): raise RuntimeError('Studio altered the reviewed FP32-parameter/BF16-compute precision policy')
                self.is_training=True
                try: return self.trainer.train()
                finally: self.is_training=False
        bench=FormalStudio()
        write(self.root/'studio-backend.json',{'module':str(Path(backend.__file__).resolve()),'sha256':sha(backend.__file__),'sft_class':backend.SFTTrainer.__module__+'.'+backend.SFTTrainer.__name__,'unsloth':importlib.metadata.version('unsloth'),'integration':'actual headless UnslothTrainer owner and Studio-imported SFTTrainer; no UI submission claim'})
        bench.run_formal(); self.save(optimizer,scheduler,cursor)
        if cursor.committed!=len(order): raise RuntimeError('Resumable stop: '+str(self.stop_reason or 'incomplete curriculum'))
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
        # Reconcile the newest committed intent OR receipt across VM loss.
        from colab.preflight import download_verified
        entries=self.store.discover(); latest=None; folder=self.root/'ledger-reconciliation'; folder.mkdir(exist_ok=True)
        for item in entries:
            if not any('-'+kind+'-' in item['name'] for kind in ('instructor-intent','instructor-receipt')): continue
            target=folder/(item['commit_sha256']+'.json')
            if not target.exists(): download_verified(self.store.service,item['commit_file_id'],target,item['commit_sha256'])
            commit=read(target)
            if commit['binding']!=self.bound: continue
            if latest is None or commit['created_unix']>latest[0]: latest=(commit['created_unix'],item)
        if latest:
            destination=folder/('latest-'+latest[1]['commit_sha256'][:12]); self.store.restore(latest[1],destination)
            p=self.root/'instructor/LEDGER.json'; p.parent.mkdir(exist_ok=True); shutil.copy2(destination/'LEDGER.json',p)
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
    def qualify_instructor_format(self):
        from .teacher import Instructor
        import torch
        if (self.root/'INSTRUCTOR_FORMAT_PROBE.json').exists(): return
        self.sync_instructor_ledger(); teacher=Instructor(self.key,self.root/'instructor',self.c,self.store)
        self.model.eval(); reviewed=[]
        for fam,lang in [('image','en'),('speech_in','zh')]:
            rowid=self.catalogue.select('train',[fam],lang)[0][0]
            row,media=self.media(rowid); row['_parent_checkpoint']=self.state['last_checkpoint']['commit_sha256']
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): student=self.model.answer(row,media,32)
            decision=teacher.review(row,media,student,0); reviewed.append({'id':rowid,'family':fam,'language':lang,'decision':decision})
        result={'passed':True,'requests':2,'scope':'Current provider key/model and actual image/audio request schema/accounting only; teacher competence and route eligibility remain separate','reviewed_training_records':reviewed}
        write(self.root/'INSTRUCTOR_FORMAT_PROBE.json',result); self.store.small_file(self.root/'INSTRUCTOR_FORMAT_PROBE.json','instructor-format-probe')
    def run_formal(self,profile):
        from .teacher import eligibility
        eligibility(self.c)
        if not (self.root/'FULL_RECOVERY_PROOF.json').exists() or read(self.root/'FULL_RECOVERY_PROOF.json').get('binding')!=self.bound: raise RuntimeError('Actual fresh-VM recovery gate has not passed')
        if self.model is None: raise RuntimeError('Restore the reviewed initial or resumable campaign checkpoint first')
        slot=math.ceil(profile['complete_checkpoint_bytes']*1.10)
        self.reader.checkpoint_slot_bytes=slot; self.state['qualified_checkpoint_reservation_bytes']=slot
        from colab.stage1.drive_checkpoint import project_usage
        remaining_drive=self.c['drive_cap_bytes']-project_usage(self.store.service,self.c['drive_project_folder'])-100_000_000_000
        # Fixed stage/RSI snapshots plus periodic commits must fit cold storage.
        fixed=len(self.c['stages'])+3*self.c['instructor_rounds']
        periodic_slots=math.floor(remaining_drive/slot)-fixed
        if periodic_slots<1: raise RuntimeError('Drive project cannot retain the declared stage/RSI checkpoints and a periodic recovery history')
        required_minutes=math.ceil(profile['projected_hours']*60/periodic_slots/10)*10
        transfer_minutes=math.ceil(profile['qualification_complete_receipt'].get('seconds',0)*1.2*4/60/10)*10
        self.effective_checkpoint_minutes=max(self.c['checkpoint_minutes'],required_minutes,transfer_minutes)
        if self.effective_checkpoint_minutes>self.c['max_checkpoint_minutes']: raise RuntimeError('Cold-storage capacity requires a checkpoint interval beyond the reviewed maximum')
        write(self.root/'STORAGE_PROFILE.json',{'measured_checkpoint_bytes':profile['complete_checkpoint_bytes'],'slot_with_margin':slot,'effective_checkpoint_minutes':self.effective_checkpoint_minutes,'requested_checkpoint_minutes':self.c['checkpoint_minutes'],'transfer_overhead_minutes_requirement':transfer_minutes,'cold_project_cap_bytes':self.c['drive_cap_bytes'],'no_cold_artifacts_deleted':True})
        emit('storage_profile',**read(self.root/'STORAGE_PROFILE.json'))
        if not self.state.get('completed_stages') and self.state['stage']=='initial':
            self.budget.check(profile['projected_hours']*3600)
        self.qualify_instructor_format()
        if not (self.root/'BASELINE.json').exists():
            base=self.evaluate('dev','pretrained-native-baseline',generate=True); write(self.root/'BASELINE.json',base); self.store.small_file(self.root/'BASELINE.json','baseline')
        for i,stage in enumerate(self.c['stages']):
            if stage['name'] in self.state.get('completed_stages',[]): continue
            order=self.catalogue.plan(stage,self.c['seed']+i)
            resume=self.state.get('stage')==stage['name'] and self.state.get('cursor') is not None
            self.train_order(stage,order,resume=resume)
            self.evaluate('dev',stage['name'],generate=False)
        for round_no in range(1,self.c['instructor_rounds']+1):
            if any(r['round']==round_no for r in self.state.get('rsi_rounds',[])): continue
            self.instructed_rsi(round_no)
        if len(self.state.get('rsi_rounds',[]))!=self.c['instructor_rounds']: raise RuntimeError('Required instructed-RSI comparisons are incomplete')
        self.evaluate('audit','final-protected-audit',generate=True)
        self.state.update(training_complete=True,final_audit_complete=True,human_audio_review='pending',capability_claims_verified=False)
        write(self.root/'CAMPAIGN.json',self.state); self.store.small_file(self.root/'CAMPAIGN.json','campaign-completion')
        from .publish import export_and_upload
        return export_and_upload(self,self.c)
