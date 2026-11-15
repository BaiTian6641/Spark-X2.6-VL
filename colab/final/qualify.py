"""Real complete-model CUDA qualification + 4-rung APOLLO/COAT/merged-gradient comparison."""
from __future__ import annotations
import contextlib, copy, gc, hashlib, math, os, random, subprocess, time
from collections import defaultdict
from pathlib import Path
from typing import Any
from .common import HARDWARE_PROFILES,HERE,identity,read,write,emit,disk_guard,runtime_boot_id
from .data import FAMILIES
from .engine import ensure_coat_for_model,optimizer_for,census,sample_parameters,checkpoint,load_checkpoint

def process_identity():
    # Start ticks disambiguate a reused PID on the same Linux VM or macOS host.
    stat_file=Path('/proc/self/stat')
    if stat_file.is_file():
        stat=stat_file.read_text().rsplit(')',1)[1].split()
        ticks=int(stat[19])
    else:
        try:
            lstart=subprocess.check_output(['ps','-o','lstart=','-p',str(os.getpid())],text=True).strip()
        except Exception:
            lstart=str(os.getpid())
        ticks=int(hashlib.sha256(lstart.encode()).hexdigest()[:12],16)
    return {'boot_id':runtime_boot_id(),'pid':os.getpid(),'start_ticks':ticks}

def require_new_process(profile):
    previous=profile.get('qualification_process')
    current=process_identity()
    if not previous or previous==current:
        raise RuntimeError('Run recovery in a new campaign subprocess after qualification finishes; keep the same G4 VM')
    return current

KERNEL_GRADIENT_ATOL=5e-4
KERNEL_GRADIENT_RTOL=.15

def check_kernel_gradient_parity(reference_grads,reference_norms,named_parameters):
    checked=0;max_sample_error=0.;max_sample_l2=0.;worst=None;max_norm_error=0.
    for name,p in named_parameters:
        if name not in reference_grads: continue
        if p.grad is None: raise RuntimeError('Kernel adapter removed a gradient: '+name)
        reference=reference_grads[name];candidate=p.grad.detach().flatten()[:16].float().cpu()
        difference=reference-candidate;sample_l2=float(difference.norm());reference_sample_norm=float(reference.norm())
        sample_allowed=KERNEL_GRADIENT_ATOL*math.sqrt(reference.numel())+KERNEL_GRADIENT_RTOL*reference_sample_norm
        if sample_l2>sample_allowed:
            raise RuntimeError(f'Spark BF16 gradient sample L2 parity failed for {name}: error={sample_l2} allowed={sample_allowed} max_abs={float(difference.abs().max())}')
        max_sample_l2=max(max_sample_l2,sample_l2)
        error=float(difference.abs().max())
        if error>max_sample_error: max_sample_error=error;worst=name
        reference_norm=reference_norms[name];candidate_norm=float(p.grad.detach().float().norm())
        if not math.isfinite(reference_norm) or not math.isfinite(candidate_norm):
            raise RuntimeError('Nonfinite kernel parity gradient norm: '+name)
        norm_error=abs(reference_norm-candidate_norm);max_norm_error=max(max_norm_error,norm_error)
        allowed=KERNEL_GRADIENT_ATOL+KERNEL_GRADIENT_RTOL*reference_norm
        if norm_error>allowed:
            raise RuntimeError(f'Spark BF16 full-tensor gradient norm parity failed for {name}: error={norm_error} allowed={allowed} sample_l2={sample_l2} sample_allowed={sample_allowed}')
        checked+=1
    return {'gradient_sample_tensors':checked,'gradient_sample_atol':KERNEL_GRADIENT_ATOL,
            'gradient_sample_rtol':KERNEL_GRADIENT_RTOL,'max_sample_absolute_error':max_sample_error,
            'max_sample_l2_error':max_sample_l2,'worst_gradient_parameter':worst,'max_full_tensor_norm_absolute_error':max_norm_error}

def audit_parameter_coverage(model,before,after,nonzero_gradients):
    missing=[name for name,p in model.named_parameters() if p.requires_grad and name not in nonzero_gradients]
    if missing: raise RuntimeError('Trainable parameters absent from the complete backward cycle or have only zero gradients: '+','.join(missing[:10]))
    changed=defaultdict(int); writeback={}
    for group,name,p in model.named_groups():
        updated=before[name]!=after[name]
        if p.requires_grad: writeback[name]=updated
        if updated: changed[group]+=1
    for group,values in census(model).items():
        if values['trainable'] and not changed[group]: raise RuntimeError('No actual parameter writeback in '+group)
    return dict(changed),writeback


def _optimizer_state_bytes(optimizer)->int:
    if hasattr(optimizer,'memory_breakdown_bytes'):
        bd=optimizer.memory_breakdown_bytes()
        return int(bd['apollo_moment_bytes']+bd['adamw_moment_bytes']+bd['master_weight_bytes'])
    import torch
    total=0
    seen:set[int]=set()
    for state in optimizer.state.values():
        if not isinstance(state,dict): continue
        for val in state.values():
            if isinstance(val,torch.Tensor) and id(val) not in seen:
                seen.add(id(val))
                total+=val.numel()*val.element_size()
            elif isinstance(val,dict):
                for sub in val.values():
                    if isinstance(sub,torch.Tensor) and id(sub) not in seen:
                        seen.add(id(sub))
                        total+=sub.numel()*sub.element_size()
    return total


def _parameter_bytes(model)->dict[str,int]:
    trainable_b=0; frozen_b=0; seen:set[int]=set()
    for p in model.parameters():
        if id(p) in seen: continue
        seen.add(id(p))
        nbytes=p.numel()*p.element_size()
        if p.requires_grad: trainable_b+=nbytes
        else: frozen_b+=nbytes
    return {'trainable_param_bytes':trainable_b,'frozen_param_bytes':frozen_b,'total_param_bytes':trainable_b+frozen_b}


def qualify_matched_profiles(
    model_builder,
    examples:list[tuple[dict[str,Any],dict[str,Any]]],
    base_config:dict[str,Any],
    *,
    profiles:tuple[str,...]=('bf16_adamw','apollo_merged','apollo_coat','apollo_coat_fp8_moments'),
    rank_candidates:tuple[int,...]=(64,32),
    microbatch_size:int=1,
    accumulation_steps:int=2,
    num_updates:int=2,
    device:str='cpu',
)->dict[str,Any]:
    """Compare optimizer/precision rungs & APOLLO ranks on identical data batches with honest telemetry."""
    import torch
    from .optimizer import MergedGradientAccumulator

    is_cuda=str(device).startswith('cuda') and torch.cuda.is_available()
    amp_ctx=(lambda: torch.autocast('cuda',dtype=torch.bfloat16)) if is_cuda else contextlib.nullcontext
    results:dict[str,Any]={}

    runs:list[tuple[str,dict[str,Any]]]=[]
    for prof in profiles:
        if prof=='apollo_merged' and len(rank_candidates)>1:
            for r in rank_candidates:
                runs.append((f'{prof}_r{r}',{**base_config,'optimizer_profile':prof,'apollo_rank':int(r)}))
        else:
            runs.append((prof,{**base_config,'optimizer_profile':prof}))

    for label,cfg in runs:
        seed=int(cfg.get('seed',39526))
        torch.manual_seed(seed); random.seed(seed)
        if is_cuda:
            torch.cuda.manual_seed_all(seed)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        model=opt=acc=coat_ctrl=out=mb_loss=chunk=None
        try:
            model=model_builder(cfg)
            model.set_stage(
                bool(cfg.get('alignment_only',False)),
                stage_mode=cfg.get('stage_mode'),
                unfreeze_vision_last_n_layers=int(cfg.get('unfreeze_vision_last_n_layers',0)),
            )
            model.train()
            frozen_before=model.frozen_parameter_hashes()
            coat_ctrl=ensure_coat_for_model(model,cfg)
            if coat_ctrl is not None:
                coat_ctrl.telemetry.reset_step()

            opt=optimizer_for(model,cfg)
            buf_policy='fp32' if cfg.get('alignment_only',False) else 'bf16_apollo_fp32_small'
            acc=MergedGradientAccumulator(
                model,
                accum_mode=buf_policy,
                min_bf16_accum_dim=int(cfg.get('apollo_min_dim',128)),
                allow_embeddings_in_apollo=bool(cfg.get('apollo_include_embeddings',False)),
            )
            losses:list[float]=[]
            grad_norms:list[float]=[]
            step_seconds:list[float]=[]
            cursor_idx=0
            coat_ctx=(lambda: coat_ctrl.activation_context(training=True)) if (coat_ctrl is not None and coat_ctrl.enabled) else contextlib.nullcontext
            inv_fn=(coat_ctrl.invalidate_weight_caches if (coat_ctrl is not None and coat_ctrl.enabled) else None)

            try:
                for _upd in range(num_updates):
                    t0=time.perf_counter()
                    upd_losses=[]
                    n_examples_total=0
                    for _mb in range(accumulation_steps):
                        chunk=[examples[(cursor_idx+k)%len(examples)] for k in range(microbatch_size)]
                        cursor_idx+=microbatch_size
                        with amp_ctx(), coat_ctx():
                            out=model(examples=chunk)
                            mb_loss=out['per_example_loss'].sum()
                        units=max(1,int(out.get('supervised_units',len(chunk))))
                        with acc.microbatch(weight=1.0,supervised_units=units):
                            mb_loss.backward()
                        n_examples_total+=len(chunk)
                        upd_losses.append(float(out['loss'].detach()))
                        out=mb_loss=None
                    stats=acc.step(
                        opt,
                        None,
                        max_grad_norm=float(cfg.get('clip_norm',1.0)),
                        post_scale=1.0/max(1,n_examples_total),
                        invalidate_caches_fn=inv_fn,
                    )
                    if not stats.finite or not stats.applied:
                        raise RuntimeError(f'Nonfinite merged gradient in profile {label}')
                    if is_cuda: torch.cuda.synchronize()
                    step_seconds.append(time.perf_counter()-t0)
                    losses.append(sum(upd_losses)/len(upd_losses))
                    grad_norms.append(stats.grad_norm)
            finally:
                merged_buf_bytes=acc.buffer_bytes()
                acc.close()

            frozen_after=model.frozen_parameter_hashes()
            if frozen_before!=frozen_after:
                raise RuntimeError(f'Frozen parameters mutated during profile {label}')

            p_bytes=_parameter_bytes(model)
            opt_bytes=_optimizer_state_bytes(opt)
            entry={
                'status':'passed',
                'optimizer_profile':cfg['optimizer_profile'],
                'apollo_rank':cfg.get('apollo_rank'),
                'device_type':'cuda' if is_cuda else 'cpu',
                **p_bytes,
                'merged_grad_buffer_bytes':merged_buf_bytes,
                'optimizer_state_bytes':opt_bytes,
                'total_static_training_bytes':p_bytes['total_param_bytes']+merged_buf_bytes+opt_bytes,
                'cuda_peak_allocated_bytes':int(torch.cuda.max_memory_allocated()) if is_cuda else None,
                'cuda_peak_reserved_bytes':int(torch.cuda.max_memory_reserved()) if is_cuda else None,
                'losses':losses,
                'grad_norms':grad_norms,
                'mean_step_seconds':sum(step_seconds)/max(1,len(step_seconds)),
                'frozen_parameters_unchanged':True,
                'coat_telemetry':coat_ctrl.telemetry.summary() if coat_ctrl is not None else None,
            }
            results[label]=entry
        except RuntimeError as exc:
            is_oom=isinstance(exc,getattr(torch,'OutOfMemoryError',())) or 'out of memory' in str(exc).lower()
            if is_oom:
                reason=str(exc)[:400]
                exc.__traceback__=None
                results[label]={'status':'cuda_oom','optimizer_profile':cfg['optimizer_profile'],'reason':reason}
            else:
                raise
        finally:
            if acc is not None:
                try: acc.close()
                except Exception: pass
            model=opt=acc=coat_ctrl=out=mb_loss=chunk=None
            gc.collect()
            if is_cuda: torch.cuda.empty_cache()

    return {
        'schema_version':1,
        'device':device,
        'is_live_cuda_measurement':is_cuda,
        'microbatch_size':microbatch_size,
        'accumulation_steps':accumulation_steps,
        'num_updates':num_updates,
        'profiles':results,
    }


def qualify(campaign):
    import numpy as np,psutil,torch
    from transformers import get_constant_schedule
    from .coat_spark import qualify_coat_kernels
    from .model import SparkNative
    from .optimizer import MergedGradientAccumulator

    c=campaign.c; root=campaign.root
    if not torch.cuda.is_available() or torch.version.hip:
        raise RuntimeError('This release requires the reviewed CUDA runtime, not ROCm or CPU')
    prop=torch.cuda.get_device_properties(0)
    default_hw='rtx_pro_4000_blackwell_24gb' if c.get('execution_target')=='local_workstation' else 'colab_g4_96gb'
    hw_name=c.get('hardware_profile',default_hw)
    hw=HARDWARE_PROFILES[hw_name]
    if prop.total_memory<hw['min_gpu_bytes'] or psutil.virtual_memory().total<hw['min_ram_bytes']:
        raise RuntimeError(
            f'Qualification for {hw_name} requires at least {hw["min_gpu_bytes"]/1e9:.1f} GB GPU memory '
            f'and {hw["min_ram_bytes"]/1e9:.1f} GB system RAM (found GPU={prop.total_memory/1e9:.2f} GB)'
        )
    if torch.cuda.get_device_capability(0)!=hw['capability']:
        raise RuntimeError(f'Assigned GPU capability {torch.cuda.get_device_capability(0)} differs from reviewed Blackwell SM120 target {hw["capability"]}')

    torch.manual_seed(c['seed']); np.random.seed(c['seed']); random.seed(c['seed']); torch.set_autocast_cache_enabled(False)
    coat_kernel_report=qualify_coat_kernels('cuda') if c.get('optimizer_profile','adamw8bit') in ('apollo_coat','apollo_coat_fp8_moments') else None

    campaign.model=SparkNative(campaign.paths,c); model=campaign.model; model.set_stage(False); model.train()
    frozen_hashes_initial=model.frozen_parameter_hashes()
    ensure_coat_for_model(model,c)

    optimizer=optimizer_for(model,c); scheduler=get_constant_schedule(optimizer)
    campaign.state.update(stage='qualification-initial',alignment_only=False,updates=0,planned_updates=1,warmup_steps=0,scheduler_kind='constant')
    initial=campaign.save(optimizer,scheduler,defer_upload=True)
    initial_key=campaign.state['last_checkpoint_local']

    import sys
    from types import SimpleNamespace
    module=sys.modules[type(model.spark).__module__]; adapter=module.eager_attention_forward
    cand_image_rids=[r[0] for r in campaign.catalogue.select('train',['image'],'en')]
    def _parity_record_len(r_id:str)->tuple[int,int,str]:
        rw,_=campaign.catalogue.row(r_id)
        return (len(rw.get('frames') or ()), len(str(rw.get('prompt','')))+len(str(rw.get('answer',''))), r_id)
    rid=min(cand_image_rids,key=_parity_record_len)
    row,media=campaign.media(rid)
    optimizer.zero_grad(set_to_none=True)
    if c.get('modalities')=='vision_text':
        # Keep model.spark.train() so HF gradient_checkpointing remains active on 24 GB VRAM;
        # temporarily zero any Dropout / MultiheadAttention dropout so reference vs SDPA forward/backward are deterministic.
        saved_dropouts=[(m,m.p) for m in model.modules() if isinstance(m,torch.nn.Dropout)]
        saved_mha_dropouts=[(m,m.dropout) for m in model.modules() if isinstance(m,torch.nn.MultiheadAttention)]
        for m,_ in saved_dropouts: m.p=0.0
        for m,_ in saved_mha_dropouts: m.dropout=0.0
        model.train()
        if model.vision is not None: model.vision.eval()
        acc_parity=MergedGradientAccumulator(
            model,
            accum_mode='bf16_apollo_fp32_small',
            min_bf16_accum_dim=int(c.get('apollo_min_dim',128)),
            allow_embeddings_in_apollo=bool(c.get('apollo_include_embeddings',False)),
        )
        try:
            try:
                torch.manual_seed(c['seed'])
                module.eager_attention_forward=adapter._spark_reference
                with torch.autocast('cuda',dtype=torch.bfloat16): reference_loss,_=model.loss(row,media)
                reference_value=float(reference_loss.detach())
                with acc_parity.microbatch(weight=1.0,supervised_units=1):
                    reference_loss.backward()
                del reference_loss
                reference_grads={n:acc_parity.buffers[id(p)].detach().flatten()[:16].float().cpu().clone() for n,p in model.named_parameters() if id(p) in acc_parity.buffers}
                reference_norms={n:float(acc_parity.buffers[id(p)].detach().float().norm()) for n,p in model.named_parameters() if id(p) in acc_parity.buffers}
            finally:
                module.eager_attention_forward=adapter
            acc_parity.zero_buffers()
            torch.manual_seed(c['seed'])
            with torch.autocast('cuda',dtype=torch.bfloat16): candidate_loss,_=model.loss(row,media)
            candidate_value=float(candidate_loss.detach())
            with acc_parity.microbatch(weight=1.0,supervised_units=1):
                candidate_loss.backward()
            del candidate_loss
            if abs(reference_value-candidate_value)>.01: raise RuntimeError('Full Spark BF16 attention loss parity failed')
            cand_named=[(n,SimpleNamespace(grad=acc_parity.buffers.get(id(p)))) for n,p in model.named_parameters()]
            parity=check_kernel_gradient_parity(reference_grads,reference_norms,cand_named)
            del cand_named
        finally:
            acc_parity.close(); del acc_parity
            for m,orig_p in saved_dropouts: m.p=orig_p
            for m,orig_d in saved_mha_dropouts: m.dropout=orig_d
            torch.cuda.empty_cache()
        del reference_grads,reference_norms
    else:
        model.eval()
        try:
            module.eager_attention_forward=adapter._spark_reference
            with torch.autocast('cuda',dtype=torch.bfloat16): reference_loss,_=model.loss(row,media)
            reference_loss.backward()
            reference_grads={n:p.grad.detach().flatten()[:16].float().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
            reference_norms={n:float(p.grad.detach().float().norm()) for n,p in model.named_parameters() if p.grad is not None}
        finally: module.eager_attention_forward=adapter
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): candidate_loss,_=model.loss(row,media)
        candidate_loss.backward()
        reference_value=float(reference_loss.detach());candidate_value=float(candidate_loss.detach())
        if abs(reference_value-candidate_value)>.01: raise RuntimeError('Full Spark BF16 attention loss parity failed')
        parity=check_kernel_gradient_parity(reference_grads,reference_norms,model.named_parameters())
        del reference_grads,reference_norms,reference_loss,candidate_loss
    optimizer.zero_grad(set_to_none=True); model.train()
    kernel_parity={'passed':True,'real_training_record':rid,'reference_loss':reference_value,'sdpa_loss':candidate_value,**parity,'scope':'Full resident Spark+vision BF16 loss, sampled-gradient and full-tensor gradient-norm comparison; CPU tests compare all tiny-decoder gradients'}

    before=sample_parameters(model); seen=set(); timings=defaultdict(list); rows=campaign.catalogue.select('train'); by_family=defaultdict(list)
    for r in rows: by_family[r[1]].append(r[0])

    active_families=[f for f in ('image','video','text') if by_family.get(f)] if c.get('modalities')=='vision_text' else list(FAMILIES)
    if not active_families:
        raise RuntimeError('No active training families found in corpus for qualification')

    optimizer.zero_grad(set_to_none=True); del optimizer,scheduler; gc.collect(); torch.cuda.empty_cache()
    mixed_order=[by_family[fam][0] for fam in active_families]
    while len(mixed_order)<c['gradient_accumulation']:
        mixed_order.append(by_family[active_families[len(mixed_order)%len(active_families)]][0])
    campaign.train_order({'name':'qualification-accumulated','alignment_only':False,'stage_mode':'full_spark'},mixed_order)
    initial=campaign.deferred_receipts[initial_key]
    studio_receipt=campaign.state['last_checkpoint']
    model,optimizer,_,campaign.state=load_checkpoint(campaign.state['last_checkpoint_local'],c,campaign.bound,model=model)
    for group in optimizer.param_groups: group['lr']=group['initial_lr']
    scheduler=get_constant_schedule(optimizer)
    campaign.state.update(scheduler_kind='constant',planned_updates=1,warmup_steps=0)
    torch.cuda.reset_peak_memory_stats(); exposure=defaultdict(int)

    coat_ctrl=ensure_coat_for_model(model,c)
    coat_ctx=(lambda: coat_ctrl.activation_context(training=True)) if (coat_ctrl is not None and coat_ctrl.enabled) else contextlib.nullcontext
    inv_fn=(coat_ctrl.invalidate_weight_caches if (coat_ctrl is not None and coat_ctrl.enabled) else None)
    accumulator=MergedGradientAccumulator(
        model,
        accum_mode='bf16_apollo_fp32_small',
        min_bf16_accum_dim=int(c.get('apollo_min_dim',128)),
        allow_embeddings_in_apollo=bool(c.get('apollo_include_embeddings',False)),
    )

    try:
        for fam in active_families:
            pool=sorted(by_family[fam],key=lambda x:identity(['qualification',x]))
            if not pool: raise RuntimeError('Missing qualification family '+fam)
            for i in range(35):
                campaign.budget_check()
                if campaign.stop_reason: raise RuntimeError('Qualification stopped before complete proof: '+campaign.stop_reason)
                row,media=campaign.media(pool[i%len(pool)])
                torch.cuda.synchronize(); started=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16), coat_ctx():
                    loss,units=model.loss(row,media)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite native loss in '+fam)
                with accumulator.microbatch(weight=1.0,supervised_units=max(1,int(units or 1))):
                    loss.backward()
                active=defaultdict(float)
                for group,name,p in model.named_groups():
                    if id(p) in accumulator.buffers:
                        norm=float(accumulator.buffers[id(p)].detach().float().norm()); active[group]+=norm
                        if norm>0: seen.add(name)
                if active.get('spark',0)<=0: raise RuntimeError('Native '+fam+' objective does not reach Spark')
                if c.get('modalities')!='vision_text':
                    required={'image':'vision','video':'motion','audiovisual':'speech_in','speech_in':'speech_in','sound_in':'sound_in','speech_out':'speech_output','sound_output':'sound_output','sound_out':'sound_output','text':'spark'}[fam]
                else:
                    required={'image':'connectors','video':'connectors','text':'spark'}[fam]
                if active.get(required,0)<=0: raise RuntimeError('Native component receives no gradient: '+required)
                step_stats=accumulator.step(
                    optimizer,
                    scheduler,
                    max_grad_norm=float(c['clip_norm']),
                    post_scale=1.0,
                    invalidate_caches_fn=inv_fn,
                )
                if not step_stats.finite or not step_stats.applied:
                    raise RuntimeError('Nonfinite merged gradient during qualification in '+fam)
                torch.cuda.synchronize(); seconds=time.perf_counter()-started
                if i>=5: timings[fam].append(seconds)
                exposure[fam]+=1
                emit('qualification_step',family=fam,step=i+1,loss=float(loss.detach()),supervised_units=units,seconds=seconds)
    finally:
        accumulator.close()
        del accumulator
        gc.collect()
        torch.cuda.empty_cache()

    if model.frozen_parameter_hashes()!=frozen_hashes_initial:
        raise RuntimeError('Frozen parameters (e.g., SigLIP2) mutated during full-Spark qualification')

    after=sample_parameters(model)
    changed,writeback=audit_parameter_coverage(model,before,after,seen)
    campaign.state.update(stage='qualification-complete',updates=1+sum(exposure.values()),planned_updates=1,scheduler_kind='constant')
    full=campaign.save(optimizer,scheduler)
    bytes_=campaign.state['checkpoint_bytes']
    disk_guard(root,reserve=bytes_,min_free_bytes=c.get('local_free_floor_bytes',44_000_000_000))

    profile={
        'schema_version':1,
        'passed':True,
        'hardware_profile':hw_name,
        'optimizer_profile':c.get('optimizer_profile','adamw8bit'),
        'modalities':c.get('modalities','omni'),
        'device':prop.name,
        'gpu_bytes':prop.total_memory,
        'peak_allocated':torch.cuda.max_memory_allocated(),
        'peak_reserved':torch.cuda.max_memory_reserved(),
        'host_available':psutil.virtual_memory().available,
        'census':census(model),
        'coat_kernels':coat_kernel_report,
        'kernel_parity':kernel_parity,
        'frozen_parameters_verified_unchanged':True,
        'changed_tensors_by_group':changed,
        'parameter_writeback_by_name':writeback,
        'spark_all_parameters_reached':True,
        'all_trainable_parameters_reached':True,
        'family_steps':dict(exposure),
        'studio_full_optimizer_updates':1,
        'studio_mixed_microbatches':len(mixed_order),
        'studio_checkpoint_receipt':studio_receipt,
        'warmup_updates_each':5,
        'timed_updates_each':30,
        'seconds_per_record_p90':{f:sorted(v)[math.ceil(.9*len(v))-1] for f,v in timings.items()},
        'complete_checkpoint_bytes':bytes_,
        'qualification_initial_receipt':initial,
        'qualification_complete_receipt':full,
        'initial_parameter_samples_sha256':identity(before),
        'full_parameter_samples_sha256':identity(after),
        'fresh_vm_recovery':False,
    }
    profile['qualification_process']=process_identity()
    profile['binding']=campaign.bound
    profile['scope']='actual whole-model CUDA profile; new-process checkpoint continuation is a separate gate'

    second_fam='sound_out' if 'sound_out' in active_families else active_families[-1]
    next_ids=[by_family['image'][0],by_family[second_fam][0]]; continuation=[]
    acc_cont=MergedGradientAccumulator(
        model,
        accum_mode='bf16_apollo_fp32_small',
        min_bf16_accum_dim=int(c.get('apollo_min_dim',128)),
        allow_embeddings_in_apollo=bool(c.get('apollo_include_embeddings',False)),
    )
    try:
        for rid in next_ids:
            row,media=campaign.media(rid)
            with torch.autocast('cuda',dtype=torch.bfloat16), coat_ctx():
                loss,u=model.loss(row,media)
            with acc_cont.microbatch(weight=1.0,supervised_units=max(1,int(u or 1))):
                loss.backward()
            acc_cont.step(
                optimizer,
                scheduler,
                max_grad_norm=float(c['clip_norm']),
                post_scale=1.0,
                invalidate_caches_fn=inv_fn,
            )
            continuation.append(float(loss.detach()))
    finally:
        acc_cont.close()
        del acc_cont
        gc.collect()
        torch.cuda.empty_cache()

    tol=5e-3 if c.get('optimizer_profile','').startswith('apollo') else 1e-4
    profile['continuation']={'ids':next_ids,'losses':continuation,'parameter_samples':sample_parameters(model),'tolerance':tol}
    projected=0.; stages=[]
    family_by_id={r[0]:r[1] for r in rows}
    for i,stage in enumerate(c['stages']):
        order=campaign.catalogue.plan(stage,c['seed']+i); counts=defaultdict(int)
        for rid in order: counts[family_by_id[rid]]+=1
        seconds=sum(n*profile['seconds_per_record_p90'].get(fam,next(iter(profile['seconds_per_record_p90'].values()))) for fam,n in counts.items()); projected+=seconds
        stages.append({'stage':stage['name'],'records':len(order),'by_family':dict(counts),'conservative_training_seconds':seconds})
    rate=c.get('observed_cu_per_hour')
    profile.update(stages=stages,projected_hours=projected*1.25/3600,projected_cu=(projected*1.25/3600*rate if rate else None),projection_policy='p90 full-resident per-record forward/backward/step, plus 25%; not a guarantee or automatic curriculum trim')
    write(root/'QUALIFICATION.json',profile); profile_receipt=campaign.store.small_file(root/'QUALIFICATION.json','qualification-profile')
    write(root/'QUALIFICATION_RECEIPT.json',profile_receipt)
    emit('qualification_complete',initial_receipt=initial,profile_receipt=profile_receipt,complete_checkpoint_receipt=full,projected_hours=profile['projected_hours'],projected_cu=profile['projected_cu'],next_action='Run recovery check in a fresh subprocess, then launch formal training.')
    return profile

def recovery_check(campaign,profile):
    import torch
    from .engine import ensure_coat_for_model,sample_parameters
    from .optimizer import MergedGradientAccumulator
    current_process=require_new_process(profile)
    optimizer,scheduler,restore=campaign.restore(profile['qualification_complete_receipt'])
    expected=profile['continuation']; losses=[]
    campaign.model.train()
    use_apollo=isinstance(campaign.c,dict) and str(campaign.c.get('optimizer_profile','')).startswith('apollo')
    if use_apollo:
        coat_ctrl=ensure_coat_for_model(campaign.model,campaign.c)
        coat_ctx=(lambda: coat_ctrl.activation_context(training=True)) if (coat_ctrl is not None and coat_ctrl.enabled) else contextlib.nullcontext
        inv_fn=(coat_ctrl.invalidate_weight_caches if (coat_ctrl is not None and coat_ctrl.enabled) else None)
        acc=MergedGradientAccumulator(
            campaign.model,
            accum_mode='bf16_apollo_fp32_small',
            min_bf16_accum_dim=int(campaign.c.get('apollo_min_dim',128)),
            allow_embeddings_in_apollo=bool(campaign.c.get('apollo_include_embeddings',False)),
        )
        try:
            for rid in expected['ids']:
                row,media=campaign.media(rid)
                with torch.autocast('cuda',dtype=torch.bfloat16), coat_ctx():
                    loss,u=campaign.model.loss(row,media)
                with acc.microbatch(weight=1.0,supervised_units=max(1,int(u or 1))):
                    loss.backward()
                acc.step(
                    optimizer,
                    scheduler,
                    max_grad_norm=float(campaign.c['clip_norm']),
                    post_scale=1.0,
                    invalidate_caches_fn=inv_fn,
                )
                losses.append(float(loss.detach()))
        finally:
            acc.close()
            del acc
            gc.collect()
            torch.cuda.empty_cache()
    else:
        for rid in expected['ids']:
            row,media=campaign.media(rid); optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): loss,_=campaign.model.loss(row,media)
            loss.backward(); torch.nn.utils.clip_grad_norm_(campaign.model.parameters(),campaign.c['clip_norm']); optimizer.step(); scheduler.step(); losses.append(float(loss.detach()))
    tolerance=expected['tolerance']; current=sample_parameters(campaign.model)
    error=max(abs(v-w) for name,values in current.items() for v,w in zip(values,expected['parameter_samples'][name]))
    loss_error=max(abs(a-b) for a,b in zip(losses,expected['losses']))
    if error>tolerance or loss_error>tolerance: raise RuntimeError('Checkpoint continuation differs from the saved optimizer/RNG oracle')
    del optimizer,scheduler; gc.collect(); torch.cuda.empty_cache()
    optimizer,scheduler,initial_restore=campaign.restore(profile['qualification_initial_receipt'])
    del optimizer,scheduler; gc.collect(); torch.cuda.empty_cache()
    campaign.state.update(stage='initial',updates=0,completed_stages=[],rsi_rounds=[],cursor=None)
    write(campaign.root/'CAMPAIGN.json',campaign.state)
    reset_state=read(campaign.root/'CAMPAIGN.json')
    proof={'schema_version':1,'passed':True,'fresh_vm':restore['fresh_vm'],'fresh_process':True,'recovery_process':current_process,'binding':campaign.bound,'native_updates':2,'max_parameter_sample_error':error,'max_loss_error':loss_error,'tolerance':tolerance,'complete_optimizer_recovery':True,'dataset_order_sha256':read(campaign.root/'metadata/corpus.report.json')['split_membership_sha256'],'initial_reset_complete':True,'initial_reset_receipt':profile['qualification_initial_receipt'],'initial_reset_state_sha256':identity(reset_state),'initial_reset_state':reset_state,'initial_restore_proof':initial_restore}
    write(campaign.root/'FULL_RECOVERY_PROOF.json',proof); receipt=campaign.store.small_file(campaign.root/'FULL_RECOVERY_PROOF.json','full-recovery-proof')
    return {'proof':proof,'receipt':receipt}
