"""Real complete-model CUDA qualification. Host tests cannot produce this PASS."""
import gc, math, os, random, statistics, time
from collections import defaultdict
from pathlib import Path
from .common import HERE,identity,read,write,emit,disk_guard
from .data import FAMILIES
from .engine import optimizer_for,census,sample_parameters,checkpoint,load_checkpoint

def process_identity():
    # Start ticks disambiguate a reused PID on the same Linux VM.
    stat=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    return {'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'pid':os.getpid(),'start_ticks':int(stat[19])}

def require_new_process(profile):
    previous=profile.get('qualification_process')
    current=process_identity()
    if not previous or previous==current:
        raise RuntimeError('Run recovery in a new campaign subprocess after qualification finishes; keep the same G4 VM')
    return current

def qualify(campaign):
    import numpy as np,psutil,torch
    from transformers import get_constant_schedule
    from .model import SparkNative
    c=campaign.c; root=campaign.root
    if not torch.cuda.is_available() or torch.version.hip: raise RuntimeError('This release requires the reviewed CUDA runtime, not the workstation ROCm environment')
    prop=torch.cuda.get_device_properties(0)
    if prop.total_memory<90_000_000_000 or psutil.virtual_memory().total<60_000_000_000: raise RuntimeError('Full native qualification requires at least 90 GB GPU memory and 60 GB system RAM; no automatic quantization/LoRA substitution')
    if torch.cuda.get_device_capability(0)!=(12,0): raise RuntimeError('Assigned GPU differs from the reviewed Blackwell SM120 target; qualify a separate environment release')
    torch.manual_seed(c['seed']); np.random.seed(c['seed']); random.seed(c['seed']); torch.set_autocast_cache_enabled(False)
    campaign.model=SparkNative(campaign.paths,c); model=campaign.model; model.set_stage(False); model.train()
    optimizer=optimizer_for(model,c); scheduler=get_constant_schedule(optimizer)
    campaign.state.update(stage='qualification-initial',alignment_only=False,updates=0,planned_updates=1,warmup_steps=0,scheduler_kind='constant')
    initial=campaign.save(optimizer,scheduler)
    # Compare the custom kernel adapter against the exact original full
    # decoder on real image data before timing or making optimizer updates.
    import sys
    module=sys.modules[type(model.spark).__module__]; adapter=module.eager_attention_forward
    rid=campaign.catalogue.select('train',['image'],'en')[0][0]; row,media=campaign.media(rid)
    model.eval(); optimizer.zero_grad(set_to_none=True)
    try:
        module.eager_attention_forward=adapter._spark_reference
        with torch.autocast('cuda',dtype=torch.bfloat16): reference_loss,_=model.loss(row,media)
        reference_loss.backward()
        reference_grads={n:p.grad.detach().flatten()[:16].float().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
    finally: module.eager_attention_forward=adapter
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16): candidate_loss,_=model.loss(row,media)
    candidate_loss.backward()
    if abs(float(reference_loss)-float(candidate_loss))>.01: raise RuntimeError('Full Spark BF16 attention loss parity failed')
    for name,p in model.named_parameters():
        if name in reference_grads:
            if p.grad is None: raise RuntimeError('Kernel adapter removed a gradient')
            torch.testing.assert_close(reference_grads[name],p.grad.detach().flatten()[:16].float().cpu(),rtol=.15,atol=2e-4)
    kernel_parity={'passed':True,'real_training_record':rid,'reference_loss':float(reference_loss),'sdpa_loss':float(candidate_loss),'gradient_sample_tensors':len(reference_grads),'scope':'Full resident Spark+vision BF16 loss and sampled-gradient comparison; CPU tests compare all tiny-decoder gradients'}
    del reference_grads,reference_loss,candidate_loss; optimizer.zero_grad(set_to_none=True); model.train()
    before=sample_parameters(model); seen=set(); timings=defaultdict(list); rows=campaign.catalogue.select('train'); by_family=defaultdict(list)
    for r in rows: by_family[r[1]].append(r[0])
    # Exercise Studio's actual patched SFT loop before the longer profile.
    # Every branch participates in this one accumulated full-parameter update.
    optimizer.zero_grad(set_to_none=True); del optimizer,scheduler; gc.collect(); torch.cuda.empty_cache()
    mixed_order=[by_family[fam][0] for fam in FAMILIES]
    campaign.train_order({'name':'qualification-studio','alignment_only':False},mixed_order)
    studio_receipt=campaign.state['last_checkpoint']
    model,optimizer,_,campaign.state=load_checkpoint(campaign.state['last_checkpoint_local'],c,campaign.bound,model=model)
    for group in optimizer.param_groups: group['lr']=group['initial_lr']
    scheduler=get_constant_schedule(optimizer)
    campaign.state.update(scheduler_kind='constant',planned_updates=1,warmup_steps=0)
    torch.cuda.reset_peak_memory_stats(); exposure=defaultdict(int)
    for fam in FAMILIES:
        pool=sorted(by_family[fam],key=lambda x:identity(['qualification',x]))
        if not pool: raise RuntimeError('Missing qualification family '+fam)
        for i in range(35):
            campaign.budget_check()
            if campaign.stop_reason: raise RuntimeError('Qualification stopped before complete proof: '+campaign.stop_reason)
            row,media=campaign.media(pool[i%len(pool)]); optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); started=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16): loss,units=model.loss(row,media)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite native loss in '+fam)
            loss.backward(); active=defaultdict(float)
            for group,name,p in model.named_groups():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all(): raise RuntimeError('Nonfinite gradient in '+name)
                    norm=float(p.grad.detach().float().norm()); active[group]+=norm
                    if norm>0: seen.add(name)
            if active.get('spark',0)<=0: raise RuntimeError('Native '+fam+' objective does not reach Spark')
            required={'image':'vision','video':'motion','audiovisual':'speech_in','speech_in':'speech_in','sound_in':'sound_in','speech_out':'speech_output','sound_out':'sound_output','text':'spark'}[fam]
            if active.get(required,0)<=0: raise RuntimeError('Native component receives no gradient: '+required)
            torch.nn.utils.clip_grad_norm_(model.parameters(),c['clip_norm']); optimizer.step(); scheduler.step()
            torch.cuda.synchronize(); seconds=time.perf_counter()-started
            if i>=5: timings[fam].append(seconds)
            exposure[fam]+=1
            emit('qualification_step',family=fam,step=i+1,loss=float(loss),supervised_units=units,seconds=seconds)
    after=sample_parameters(model); changed=defaultdict(int)
    for group,name,p in model.named_groups():
        if before[name]!=after[name]: changed[group]+=1
    missing_spark=[n for n,p in model.spark.named_parameters() if 'spark.'+n not in seen]
    if missing_spark: raise RuntimeError('Spark parameters absent from the complete backward cycle: '+','.join(missing_spark[:10]))
    for group,values in census(model).items():
        if values['trainable'] and not changed[group]: raise RuntimeError('No actual parameter writeback in '+group)
    campaign.state.update(stage='qualification-complete',updates=1+sum(exposure.values()),planned_updates=1,scheduler_kind='constant')
    full=campaign.save(optimizer,scheduler)
    bytes_=campaign.state['checkpoint_bytes']; disk_guard(root,reserve=bytes_)
    # Reconstruct all branches from local configs/code and saved state. The
    # command-line entry point also performs this in a NEW process/VM on resume.
    profile={'schema_version':1,'passed':True,'scope':'actual whole-model CUDA profile; new-process recovery is still a separate gate','device':prop.name,'gpu_bytes':prop.total_memory,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'host_available':psutil.virtual_memory().available,'census':census(model),'kernel_parity':kernel_parity,'changed_tensors_by_group':dict(changed),'spark_all_parameters_reached':True,'family_steps':dict(exposure),'studio_full_optimizer_updates':1,'studio_mixed_microbatches':len(mixed_order),'studio_checkpoint_receipt':studio_receipt,'warmup_updates_each':5,'timed_updates_each':30,'seconds_per_record_p90':{f:sorted(v)[math.ceil(.9*len(v))-1] for f,v in timings.items()},'complete_checkpoint_bytes':bytes_,'qualification_initial_receipt':initial,'qualification_complete_receipt':full,'initial_parameter_samples_sha256':identity(before),'full_parameter_samples_sha256':identity(after),'fresh_vm_recovery':False}
    profile['qualification_process']=process_identity()
    profile['binding']=campaign.bound
    profile['scope']='actual whole-model CUDA profile; new-process checkpoint continuation is a separate gate'
    # Record two actual continuation updates as the recovery oracle.
    next_ids=[by_family['image'][0],by_family['sound_out'][0]]; continuation=[]
    for rid in next_ids:
        row,media=campaign.media(rid); optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): loss,_=model.loss(row,media)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),c['clip_norm']); optimizer.step(); scheduler.step()
        continuation.append(float(loss))
    profile['continuation']={'ids':next_ids,'losses':continuation,'parameter_samples':sample_parameters(model),'tolerance':1e-4}
    projected=0.; stages=[]
    family_by_id={r[0]:r[1] for r in rows}
    for i,stage in enumerate(c['stages']):
        order=campaign.catalogue.plan(stage,c['seed']+i); counts=defaultdict(int)
        for rid in order: counts[family_by_id[rid]]+=1
        seconds=sum(n*profile['seconds_per_record_p90'][fam] for fam,n in counts.items()); projected+=seconds
        stages.append({'stage':stage['name'],'records':len(order),'by_family':dict(counts),'conservative_training_seconds':seconds})
    # Reserve 25% for checkpoint transfers, evaluation, teacher waits and controls.
    rate=c.get('observed_cu_per_hour')
    profile.update(stages=stages,projected_hours=projected*1.25/3600,projected_cu=(projected*1.25/3600*rate if rate else None),projection_policy='p90 full-resident per-record forward/backward/step, plus25%; not a guarantee or automatic curriculum trim')
    write(root/'QUALIFICATION.json',profile); profile_receipt=campaign.store.small_file(root/'QUALIFICATION.json','qualification-profile')
    write(root/'QUALIFICATION_RECEIPT.json',profile_receipt)
    emit('qualification_complete',initial_receipt=initial,profile_receipt=profile_receipt,complete_checkpoint_receipt=full,projected_hours=profile['projected_hours'],projected_cu=profile['projected_cu'],next_action='Keep this G4 VM and run train in the next cell. A new subprocess verifies checkpoint continuation, then restores the initial snapshot before formal training.')
    return profile

def recovery_check(campaign,profile):
    import torch
    from .engine import sample_parameters
    current_process=require_new_process(profile)
    optimizer,scheduler,restore=campaign.restore(profile['qualification_complete_receipt'])
    expected=profile['continuation']; losses=[]
    campaign.model.train()
    for rid in expected['ids']:
        row,media=campaign.media(rid); optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): loss,_=campaign.model.loss(row,media)
        loss.backward(); torch.nn.utils.clip_grad_norm_(campaign.model.parameters(),campaign.c['clip_norm']); optimizer.step(); scheduler.step(); losses.append(float(loss))
    tolerance=expected['tolerance']; current=sample_parameters(campaign.model)
    error=max(abs(v-w) for name,values in current.items() for v,w in zip(values,expected['parameter_samples'][name]))
    loss_error=max(abs(a-b) for a,b in zip(losses,expected['losses']))
    if error>tolerance or loss_error>tolerance: raise RuntimeError('Checkpoint continuation differs from the saved optimizer/RNG oracle')
    proof={'schema_version':1,'passed':True,'fresh_vm':restore['fresh_vm'],'fresh_process':True,'recovery_process':current_process,'binding':campaign.bound,'native_updates':2,'max_parameter_sample_error':error,'max_loss_error':loss_error,'tolerance':tolerance,'complete_optimizer_recovery':True,'dataset_order_sha256':read(campaign.root/'metadata/corpus.report.json')['split_membership_sha256']}
    write(campaign.root/'FULL_RECOVERY_PROOF.json',proof); receipt=campaign.store.small_file(campaign.root/'FULL_RECOVERY_PROOF.json','full-recovery-proof')
    del optimizer,scheduler; gc.collect(); torch.cuda.empty_cache()
    # Main campaign begins from the original source lineage, not profile updates.
    optimizer,scheduler,_=campaign.restore(profile['qualification_initial_receipt'])
    del optimizer,scheduler; gc.collect(); torch.cuda.empty_cache()
    campaign.state.update(stage='initial',updates=0,completed_stages=[],rsi_rounds=[],cursor=None)
    write(campaign.root/'CAMPAIGN.json',campaign.state)
    return {'proof':proof,'receipt':receipt}
