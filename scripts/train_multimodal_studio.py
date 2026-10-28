"""Bounded full-parameter Spark vision experiment through installed Studio.

Local subclass hooks supply a custom collator and alias-aware checkpoint saving.
No installed Studio/Unsloth/Spark source is edited. This is not a Studio UI job.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import random
import signal
import shutil
import statistics
import sys
import time
import traceback
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SCRIPT_STARTED=time.monotonic()


def select_training_records(records, limit=None, seed=3407):
    """Select without replacement, round-robin over seeded task/language strata.

    With no limit, preserve the legacy record order and Trainer sampling path.
    With a limit, the returned order is suitable for a sequential sampler: a
    bounded prefix of actual microbatches stays mixed across available strata.
    """
    if not records:
        raise ValueError('Training task selection produced an empty dataset')
    if any(record.get('split')!='train' for record in records):
        raise ValueError('Only records declared in the training split may be selected for updates')
    if limit is None:
        return list(records)
    if limit < 1:
        raise ValueError('--train-samples must be positive')
    groups=collections.defaultdict(list)
    for record in records:
        groups[(record['task'],record['language'])].append(record)
    generator=random.Random(seed)
    for key in sorted(groups):
        values=groups[key]
        values.sort(key=lambda record:record['id'])
        generator.shuffle(values)
    chosen=[]
    while groups and len(chosen)<min(limit,len(records)):
        for key in sorted(groups):
            chosen.append(groups[key].pop())
            if not groups[key]:del groups[key]
            if len(chosen)>=limit:break
    return chosen


def training_selection_receipt(records, available, limit, seed):
    ids=[record['id'] for record in records]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate record IDs in selected training records')
    serialized=json.dumps(records,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
    return {'available_after_task_filter':available,'requested_limit':limit,'selected_examples':len(records),
        'selection_seed':seed,'policy':('legacy_dataset_order_and_trainer_sampler' if limit is None
            else 'seeded_task_language_round_robin_without_replacement_with_sequential_sampler'),
        'record_ids':ids,'record_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        'selected_records_sha256':hashlib.sha256(serialized).hexdigest(),
        'unique_source_groups':len({r['source_group'] for r in records}),
        'task_counts':dict(collections.Counter(r['task'] for r in records)),
        'language_counts':dict(collections.Counter(r['language'] for r in records))}


def epoch_record_indices(records, seed, epoch):
    """A complete epoch with each shuffled stratum spread over its full length."""
    groups=collections.defaultdict(list)
    for index,record in enumerate(records):groups[(record['task'],record['language'])].append(index)
    generator=random.Random(seed+epoch)
    scheduled=[]
    for key in sorted(groups):
        values=groups[key]
        values.sort(key=lambda index:records[index]['id'])
        generator.shuffle(values)
        phase=generator.random()
        scheduled.extend(((ordinal+phase)/len(values),key,index) for ordinal,index in enumerate(values))
    return [index for _,_,index in sorted(scheduled)]


def objective_weights(records, policy):
    if policy=='none':return [1.]*len(records)
    if policy!='task_language':raise ValueError('Unknown objective balance policy')
    counts=collections.Counter((r['task'],r['language']) for r in records)
    # Mean weight is one; each task/language stratum has equal total weight.
    return [len(records)/(len(counts)*counts[(r['task'],r['language'])]) for r in records]


def planned_epoch_updates(examples, accumulation, epochs):
    if min(examples,accumulation,epochs)<1:raise ValueError('Epoch plan must be positive')
    return math.ceil(examples/accumulation)*epochs


def prefix_layout_migration(previous, current, allowed):
    differences={key:{'old':previous.get(key),'new':current.get(key)}
        for key in set(previous)|set(current)
        if key not in previous or key not in current or previous[key]!=current[key]}
    if differences and (not allowed or set(differences)-{'pool_grid','tokens_per_frame'}):
        raise ValueError(f'Weights-only architecture mismatch: {differences}')
    return differences


def configure_amp_environment(environment):
    """Select FP32 storage + BF16 AMP rather than Unsloth pure-BF16 mode."""
    requested={'UNSLOTH_MIXED_PRECISION':'float32','ACCELERATE_MIXED_PRECISION':'bf16'}
    previous={key:environment.get(key) for key in requested}
    environment.update(requested)
    return {'before':previous,'requested':requested}


parser=argparse.ArgumentParser()
parser.add_argument('--run-id',required=True)
parser.add_argument('--data-root',type=Path,default=ROOT/'data/multimodal_s0',
                    help='Versioned dataset directory; training always uses train.jsonl')
parser.add_argument('--train-samples',type=int,help='Optional deterministic task/language-balanced training subset')
parser.add_argument('--selection-seed',type=int,default=3407)
parser.add_argument('--steps',type=int,help='Legacy bounded optimizer-step count; default 24; formal mode uses epochs')
parser.add_argument('--accum',type=int,help='Gradient accumulation; default 1, or 8 in formal mode')
parser.add_argument('--formal',action='store_true',help='Full-data epoch run with periodic checkpoint/dev diagnostics and <=72h active wall budget')
parser.add_argument('--epochs',type=int,default=1,help='Complete dataset epochs in formal mode, 1..3')
parser.add_argument('--wall-hours',type=float,default=72.,help='Formal active wall budget across optimizer resumes, maximum 72h')
parser.add_argument('--loss-balance',choices=['none','task_language'],help='Formal default task_language; mean-one inverse-frequency example weighting')
parser.add_argument('--dev-every',type=int,help='Periodic dev diagnostic every N optimizer steps; formal default 250')
parser.add_argument('--save-total-limit',type=int,help='Retain latest N checkpoints; formal default 3, legacy default 2')
parser.add_argument('--result-every',type=int,help='Rewrite result.json every N steps; formal default 25, legacy default 1')
parser.add_argument('--stop-file',type=Path,help='Formal run stops and checkpoints at next optimizer boundary when this path exists')
parser.add_argument('--learning-rate',type=float,default=1e-5)
parser.add_argument('--parameter-precision',choices=['fp32','bf16'],help='Parameter storage: formal default fp32, legacy bf16; compute remains BF16 AMP')
parser.add_argument('--scheduler',choices=['constant','cosine'],help='Learning-rate schedule; formal default cosine, legacy constant')
parser.add_argument('--warmup-ratio',type=float,help='Warmup share of optimizer horizon, 0..0.1; formal default .03, legacy 0')
parser.add_argument('--max-length',type=int,default=512)
parser.add_argument('--pool-grid',type=int,default=4)
parser.add_argument('--resume-from')
parser.add_argument('--initialize-from',help='Load weights only, beginning a new optimizer/curriculum stage')
parser.add_argument('--allow-prefix-layout-change',action='store_true',help='Explicit weights-only migration: only pool_grid/tokens_per_frame architecture fields may differ')
parser.add_argument('--stage',choices=['full','alignment'],default='full')
parser.add_argument('--tasks',nargs='*',help='Optional exact task filter for this bounded stage')
parser.add_argument('--eval-limit',type=int,default=12)
parser.add_argument('--save-every',type=int)
parser.add_argument('--no-save',action='store_true')
parser.add_argument('--skip-locked-test',action='store_true')
args=parser.parse_args()
args.accum=args.accum if args.accum is not None else (8 if args.formal else 1)
args.save_every=args.save_every if args.save_every is not None else (250 if args.formal else 0)
args.dev_every=args.dev_every if args.dev_every is not None else (250 if args.formal else 0)
args.save_total_limit=args.save_total_limit if args.save_total_limit is not None else (3 if args.formal else 2)
args.result_every=args.result_every if args.result_every is not None else (25 if args.formal else 1)
args.loss_balance=args.loss_balance or ('task_language' if args.formal else 'none')
args.parameter_precision=args.parameter_precision or ('fp32' if args.formal else 'bf16')
args.scheduler=args.scheduler or ('cosine' if args.formal else 'constant')
args.warmup_ratio=args.warmup_ratio if args.warmup_ratio is not None else (.03 if args.formal else 0.)
if args.formal:
    if args.steps is not None or args.train_samples is not None:
        parser.error('Formal mode uses complete dataset epochs; omit --steps and --train-samples')
    if not (1<=args.epochs<=3 and math.isfinite(args.wall_hours) and 0<args.wall_hours<=72):
        parser.error('Formal run requires 1..3 epochs and a finite positive wall-hours<=72')
    if args.no_save or args.save_every<1:
        parser.error('Formal runs require periodic and final resumable checkpoints')
    args.skip_locked_test=True
else:
    args.steps=args.steps if args.steps is not None else 24
    if not 1<=args.steps<=500:parser.error('Legacy bounded run requires 1..500 steps; use --formal for complete epochs')
    if args.loss_balance!='none' or args.stop_file:
        parser.error('Loss weighting and stop-file are formal-mode options')
if not (1<=args.accum<=16 and 1<=args.max_length<=2048 and 1<=args.eval_limit<=64 and 1<=args.pool_grid<=16
        and args.save_every>=0 and args.dev_every>=0 and 1<=args.save_total_limit<=10 and args.result_every>=1
        and math.isfinite(args.learning_rate) and args.learning_rate>0
        and math.isfinite(args.warmup_ratio) and 0<=args.warmup_ratio<=.1):
    parser.error('Require positive accum<=16, max-length<=2048, eval-limit<=64, learning-rate; valid checkpoint/dev/result intervals')
if args.train_samples is not None and args.train_samples<1:
    parser.error('--train-samples must be positive')
args.data_root=args.data_root.expanduser().resolve()
if args.stop_file:args.stop_file=args.stop_file.expanduser().resolve()
if args.resume_from and args.initialize_from:
    parser.error('Choose optimizer resume or weights-only initialization, not both')
if args.allow_prefix_layout_change and (not args.initialize_from or args.resume_from):
    parser.error('Prefix layout migration requires --initialize-from and cannot use optimizer resume')
run_dir=ROOT/'results'/args.run_id
run_dir.mkdir(parents=True,exist_ok=False)
events=(run_dir/'events.jsonl').open('a',buffering=1)
result={'status':'starting','arguments':vars(args),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'backend':'Installed Unsloth Studio load_model and _train_worker, local custom-wrapper/collator/checkpoint hooks; not UI submission',
        'scope':'Versioned semantic image/frame and clock training; V-JEPA and audio not yet attached',
        'training_method':('full_parameter_finetuning' if args.stage=='full' else 'connector_alignment_frozen_spark_and_vision'),
        'steps':[],'logs':[],'checkpoints':[]}
stop_requested={'reason':None}
if args.formal:
    def request_stop(signum,frame):
        stop_requested['reason']=f'signal_{signum}'
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)

def emit(kind,**data):
    item={'event':kind,'utc_epoch':time.time(),**data}
    events.write(json.dumps(item,default=str)+'\n')
    print(json.dumps(item,default=str),flush=True)

def save_result():
    (run_dir/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')

save_result()
try:
    from unsloth_cli._inference import ensure_studio_backend_path
    ensure_studio_backend_path()
    from studio.backend.core.training.trainer import UnslothTrainer
    from utils.paths.storage_roots import outputs_root
    import torch
    from datasets import Dataset
    from safetensors.torch import save_model as save_tensor_model, load_model as load_tensor_model
    from safetensors import safe_open
    from transformers import TrainerCallback, SiglipVisionModel, SiglipImageProcessor
    from multimodal.model import SparkMultimodalModel
    from multimodal.collator import MultimodalCollator
    from multimodal.data import load_records, validate_dataset

    torch.set_num_threads(8)
    torch.manual_seed(3407)
    assert torch.cuda.is_available() and torch.version.hip
    result['versions']={k:md.version(k) for k in ['torch','transformers','unsloth','unsloth-zoo','trl','bitsandbytes']}
    result['gpu']=str(torch.cuda.get_device_properties(0))
    result['hip']=torch.version.hip
    result['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [Path(__file__),ROOT/'multimodal/model.py',ROOT/'multimodal/collator.py',ROOT/'multimodal/data.py']}
    data_root=args.data_root
    validation_splits=('train','dev') if args.skip_locked_test else ('train','dev','test')
    result['data_validation']=validate_dataset(data_root,splits=validation_splits)
    assert result['data_validation']['valid'],result['data_validation']['errors']
    manifest_bytes=(data_root/'dataset_manifest.json').read_bytes()
    manifest=json.loads(manifest_bytes)
    result['dataset_manifest_sha256']=hashlib.sha256(manifest_bytes).hexdigest()
    records=load_records(data_root,split='train')
    available_tasks={r['task'] for r in records}
    if args.tasks:
        unknown=set(args.tasks)-available_tasks
        if unknown:raise ValueError(f'Unknown training tasks: {sorted(unknown)}')
        records=[r for r in records if r['task'] in args.tasks]
    available=len(records)
    if args.formal and not any(r['frames'] for r in records):
        raise ValueError('Formal multimodal training requires visual records; text_replay is replay only')
    records=(sorted(records,key=lambda r:r['id']) if args.formal else
             select_training_records(records,args.train_samples,args.selection_seed))
    result['training_selection']=training_selection_receipt(records,available,args.train_samples,args.selection_seed)
    if args.formal:
        result['training_selection']['policy']='complete_epoch_seeded_stratum_even_spread_v1'
    sample_weights=objective_weights(records,args.loss_balance)
    planned_updates=planned_epoch_updates(len(records),args.accum,args.epochs) if args.formal else args.steps
    warmup_steps=math.ceil(planned_updates*args.warmup_ratio)
    result['training_plan']={'formal':args.formal,'optimizer_update_horizon':planned_updates,
        'epochs':args.epochs if args.formal else None,'examples_per_epoch':len(records),
        'planned_example_presentations':len(records)*args.epochs if args.formal else args.steps*args.accum,
        'loss_balance':args.loss_balance,'minimum_example_weight':min(sample_weights),
        'maximum_example_weight':max(sample_weights),'mean_example_weight':statistics.mean(sample_weights),
        'wall_hours':args.wall_hours if args.formal else None,
        'scheduler':args.scheduler,'warmup_ratio':args.warmup_ratio,'warmup_steps':warmup_steps,
        'sampling_note':'Complete coverage once per epoch; shuffled strata spread proportionately over the full epoch; balancing changes objective contribution, not example multiplicity' if args.formal else 'Bounded optimizer steps'}
    result['dataset_binding']={'data_root':str(data_root),'schema_version':manifest.get('schema_version'),
        'dataset_version':manifest.get('dataset_version'),'train_split':'train','development_split':'dev',
        'validated_splits':list(validation_splits),'locked_test_evaluation_requested':not args.skip_locked_test,
        'dataset_manifest_sha256':result['dataset_manifest_sha256'],
        'record_files_sha256':{name:(entry['sha256'] if isinstance(entry,dict) else entry)
                               for name,entry in manifest['record_files'].items()},
        'media_manifest_sha256':manifest['media_manifest_sha256'],
        'selected_records_sha256':result['training_selection']['selected_records_sha256'],
        'selected_record_ids_sha256':result['training_selection']['record_ids_sha256']}
    training_contract={'schema_version':1,'formal':args.formal,'stage':args.stage,'epochs':args.epochs,
        'accum':args.accum,'batch_size':1,'max_length':args.max_length,'pool_grid':args.pool_grid,
        'learning_rate':args.learning_rate,'loss_balance':args.loss_balance,'selection_seed':args.selection_seed,
        'parameter_precision':args.parameter_precision,'compute_precision':'bf16_amp',
        'autocast_weight_cache':args.parameter_precision!='fp32',
        'optimizer':'adamw_8bit','weight_decay':.001,'scheduler':args.scheduler,
        'warmup_ratio':args.warmup_ratio,'warmup_steps':warmup_steps,'optimizer_update_horizon':planned_updates,
        'save_every':args.save_every,'dev_every':args.dev_every,'eval_limit':args.eval_limit,
        'dataset_manifest_sha256':result['dataset_manifest_sha256'],
        'selected_records_sha256':result['training_selection']['selected_records_sha256'],
        'sampling_policy':result['training_selection']['policy'],'source_hashes':result['source_hashes'],
        'versions':result['versions']}
    result['training_contract']=training_contract
    exposure={'schema_version':1,'examples_forwarded':0,'target_tokens_including_eos':0,
        'presentation_counts':[0]*len(records),'task_counts':{},'language_counts':{},
        'sequence_positions':0,'frame_presentations':0,'rolling_sequence_sha256':'0'*64,
        'active_wall_seconds_before_session':0.,'resume_origin':None}
    exposure_log=(run_dir/'microbatch_exposure.jsonl').open('a',buffering=1) if args.formal else None
    def exposure_summary():
        return {key:exposure[key] for key in ('examples_forwarded','target_tokens_including_eos','sequence_positions',
            'frame_presentations','task_counts','language_counts','rolling_sequence_sha256')}|{
            'unique_records_forwarded':sum(count>0 for count in exposure['presentation_counts']),
            'minimum_presentations_per_record':min(exposure['presentation_counts']),
            'maximum_presentations_per_record':max(exposure['presentation_counts']),
            'active_wall_seconds':exposure['active_wall_seconds_before_session']+time.monotonic()-SCRIPT_STARTED,
            'measurement_boundary':'Completed training loss forwards; checkpoint at optimizer boundaries; includes EOS targets, excludes diagnostic dev forwards'}
    if args.resume_from:
        saved_sources=json.loads((Path(args.resume_from)/'experiment_sources.json').read_text())
        if saved_sources.get('dataset_manifest_sha256')!=result['dataset_manifest_sha256']:
            raise ValueError('Optimizer resume requires the original dataset manifest; use --initialize-from for a new curriculum')
        if 'training_selection' in saved_sources and saved_sources['training_selection']!=result['training_selection']:
            raise ValueError('Optimizer resume requires the same selected records, seed and sampling policy')
        if 'training_selection' not in saved_sources and (args.train_samples is not None or args.tasks):
            raise ValueError('Legacy checkpoint has no selection receipt; use unchanged full dataset for optimizer resume')
        if args.formal:
            if saved_sources.get('training_contract')!=training_contract:
                raise ValueError('Formal optimizer resume requires the exact original training contract, sources and runtime versions')
            saved_exposure=json.loads((Path(args.resume_from)/'data_exposure.json').read_text())
            if len(saved_exposure['presentation_counts'])!=len(records):raise ValueError('Exposure record identity mismatch')
            if (saved_exposure.get('selected_record_ids_sha256')!=result['training_selection']['record_ids_sha256'] or
                saved_exposure.get('selected_records_sha256')!=result['training_selection']['selected_records_sha256'] or
                any(type(count)!=int or count<0 for count in saved_exposure['presentation_counts']) or
                sum(saved_exposure['presentation_counts'])!=saved_exposure['examples_forwarded']):
                raise ValueError('Checkpoint exposure identity/count binding is invalid')
            exposure.update(saved_exposure)
            exposure['active_wall_seconds_before_session']=saved_exposure['active_wall_seconds_at_checkpoint']
            exposure['resume_origin']=str(Path(args.resume_from).resolve())
    emit('dataset_bound',**result['dataset_binding'])
    save_result()
    vision_path=ROOT/'models/SigLIP2-base-256/vision-only'

    def sample_values(model):
        return {n:p.detach().reshape(-1)[::max(1,p.numel()//32)][:32].float().cpu().clone()
                for n,p in model.named_parameters()}

    def sample_hashes(model):
        return {n:hashlib.sha256(v.numpy().tobytes()).hexdigest() for n,v in sample_values(model).items()}

    def optimizer_probe(model,optimizer):
        names={id(p):n for n,p in model.named_parameters()}
        values={}
        for parameter,fields in optimizer.state.items():
            row={}
            for key,value in fields.items():
                if torch.is_tensor(value):
                    sampled=value.detach().reshape(-1)[::max(1,value.numel()//16)][:16].float().cpu()
                    row[key]={'shape':list(value.shape),'dtype':str(value.dtype),
                              'sample_sha256':hashlib.sha256(sampled.numpy().tobytes()).hexdigest()}
                elif isinstance(value,(int,float,bool,str)) or value is None:
                    row[key]=value
            values[names[id(parameter)]]=row
        return values

    class Audit(TrainerCallback):
        def on_train_begin(self,training_args,state,control,model=None,optimizer=None,**kw):
            self.before=sample_values(model)
            self.initial_global_step=state.global_step
            self.gradient_checked=False
            ps=list(model.named_parameters())
            opt_ids={id(p) for g in optimizer.param_groups for p in g['params']}
            assert opt_ids=={id(p) for _,p in ps if p.requires_grad},'Optimizer parameter coverage mismatch'
            if args.stage=='full':
                assert all(p.requires_grad for _,p in ps),'Full-parameter stage unexpectedly froze a tensor'
            else:
                assert all(not p.requires_grad for p in model.language_model.parameters())
                assert all(not p.requires_grad for p in model.vision_encoder.parameters())
                assert all(p.requires_grad for p in model.visual_projector.parameters())
            self.trainable_components={model.component_name(n) for n,p in ps if p.requires_grad}
            self.frozen_names={n for n,p in ps if not p.requires_grad}
            assert all(p.device.type=='cuda' for _,p in ps),'Unexpected parameter offload'
            assert not any('lora_' in n for n,_ in ps)
            assert training_args.get_warmup_steps(state.max_steps)==warmup_steps,'Effective warmup differs from the explicit plan'
            assert getattr(training_args.lr_scheduler_type,'value',training_args.lr_scheduler_type)==args.scheduler
            if args.parameter_precision=='fp32':
                assert all(p.dtype==torch.float32 for _,p in ps),'FP32 parameter storage was not preserved'
                assert not torch.is_autocast_cache_enabled(),'FP32 parameter run unexpectedly caches BF16 weight copies'
            result['effective_training_args']={'scheduler':getattr(training_args.lr_scheduler_type,'value',training_args.lr_scheduler_type),
                'warmup_steps':training_args.get_warmup_steps(state.max_steps),'warmup_ratio':training_args.warmup_ratio,
                'optimizer_update_horizon':state.max_steps,'bf16':training_args.bf16,'fp16':training_args.fp16,
                'parameter_precision':args.parameter_precision,'autocast_cache_enabled':torch.is_autocast_cache_enabled(),
                'initial_optimizer_group_learning_rates':[float(group['lr']) for group in optimizer.param_groups]}
            if not args.resume_from and warmup_steps==0:
                assert all(float(group['lr'])==args.learning_rate for group in optimizer.param_groups),'Hidden optimizer warmup at the first update'
            if args.resume_from:
                saved=json.loads((Path(args.resume_from)/'training_state_probe.json').read_text())
                current=optimizer_probe(model,optimizer)
                assert current and current==saved['optimizer'],'Optimizer state differs after checkpoint resume'
                assert state.global_step==saved['global_step'],'Trainer step differs after checkpoint resume'
                assert bench.trainer.lr_scheduler.state_dict()==saved['scheduler'],'Scheduler state differs after checkpoint resume'
                result['optimizer_scheduler_resume_audit']={'optimizer_state_entries':len(current),
                    'all_sampled_optimizer_fields_match':True,'scheduler_state_matches':True,
                    'global_step_matches':True}
            component=collections.Counter()
            dtypes=collections.Counter()
            for n,p in ps:
                component[model.component_name(n)]+=p.numel();dtypes[str(p.dtype)]+=p.numel()
            result['parameter_audit']={'total':sum(p.numel() for _,p in ps),'trainable':sum(p.numel() for _,p in ps if p.requires_grad),
                'tensors':len(ps),'components':dict(component),'dtypes':dict(dtypes),'optimizer_covers_all_trainable':True,
                'all_parameters_on_gpu':True,'no_lora':True,'frozen_tensors':len(self.frozen_names),
                'training_method':result['training_method']}
            self.started=time.perf_counter()
            emit('train_begin',global_step=state.global_step,**result['parameter_audit'])

        def on_step_begin(self,training_args,state,control,**kw):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            self.start=time.perf_counter()
            self.window_visual=False
            self.learning_rates=[float(group['lr']) for group in bench.trainer.optimizer.param_groups]

        def on_pre_optimizer_step(self,training_args,state,control,model=None,**kw):
            if self.gradient_checked:return
            if args.formal and not self.window_visual:
                emit('gradient_audit_deferred',reason='Text-only optimizer window; wait for actual visual gradient')
                return
            ps=list(model.named_parameters())
            missing=[n for n,p in ps if p.requires_grad and p.grad is None]
            component=collections.defaultdict(lambda:{'tensors':0,'finite_probes':True,'sample_abs_sum':0.})
            for n,p in ps:
                if p.grad is None:continue
                g=p.grad.detach().reshape(-1)[::max(1,p.numel()//32)][:32].float()
                row=component[model.component_name(n)]
                row['tensors']+=1;row['finite_probes']&=bool(torch.isfinite(g).all().item())
                row['sample_abs_sum']+=float(g.abs().sum().item())
            result['gradient_audit']={'missing':missing,'covered_parameters':sum(p.numel() for _,p in ps if p.grad is not None),
                                      'components':dict(component),'media_to_spark_gradient':model._vision_gradient_probe}
            emit('gradient_audit',**result['gradient_audit'])
            assert not missing,missing
            assert all(x['finite_probes'] and x['sample_abs_sum']>0 for x in component.values())
            assert model._vision_gradient_probe and model._vision_gradient_probe['finite'] and model._vision_gradient_probe['absolute_sum']>0
            self.gradient_checked=True

        def on_optimizer_step(self,training_args,state,control,optimizer=None,**kw):
            if 'optimizer_state' in result:return
            counts=collections.Counter()
            for state_value in optimizer.state.values():
                for k,v in state_value.items():
                    if torch.is_tensor(v):counts[f'{k}:{v.dtype}:{v.device.type}']+=v.numel()*v.element_size()
            result['optimizer_state']={'bytes_by_field_dtype_device':dict(counts),'parameter_state_entries':len(optimizer.state)}

        def on_step_end(self,training_args,state,control,model=None,**kw):
            torch.cuda.synchronize()
            row={'step':state.global_step,'seconds':time.perf_counter()-self.start,
                 'optimizer_group_learning_rates_used':self.learning_rates,
                 'warmup':len(result['steps'])<3,'last_microbatch':model.last_batch_accounting,
                 'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            result['steps'].append(row);emit('step',**row)
            if args.formal:
                result['data_exposure']=exposure_summary()
            if args.formal:
                active=exposure_summary()['active_wall_seconds']
                if args.stop_file and args.stop_file.exists():stop_requested['reason']='stop_file'
                if active>=max(0.,args.wall_hours*3600-120):stop_requested['reason']='active_wall_budget_save_margin'
            if args.dev_every and state.global_step%args.dev_every==0 and not stop_requested['reason']:
                started=time.perf_counter()
                diagnostic=evaluate('dev',args.eval_limit,ablate=False)
                diagnostic.update({'global_step':state.global_step,'seconds':time.perf_counter()-started,
                    'capability_promotion':False})
                result.setdefault('periodic_dev',[]).append(diagnostic)
                emit('periodic_dev',**{key:value for key,value in diagnostic.items() if key!='rows'})
            if args.formal:
                active=exposure_summary()['active_wall_seconds']
                if active>=max(0.,args.wall_hours*3600-120):stop_requested['reason']='active_wall_budget_save_margin'
                if stop_requested['reason']:
                    control.should_training_stop=True;control.should_save=True
                    result['termination_reason']=stop_requested['reason']
                    emit('graceful_stop_requested',reason=stop_requested['reason'],active_wall_seconds=active)
            if state.global_step%args.result_every==0:save_result()

        def on_log(self,training_args,state,control,logs=None,**kw):
            if logs:
                assert 'loss' not in logs or math.isfinite(logs['loss']),logs
                result['logs'].append({'step':state.global_step,**logs});emit('trainer_log',step=state.global_step,**logs)

        def on_train_end(self,training_args,state,control,model=None,**kw):
            torch.cuda.synchronize()
            assert self.gradient_checked,'No visual optimizer window passed the full gradient audit'
            result['training_loop_seconds']=time.perf_counter()-self.started
            after=sample_values(model)
            components=collections.defaultdict(lambda:{'sampled_tensors':0,'changed_tensors':0})
            for n,values in after.items():
                row=components[model.component_name(n)];row['sampled_tensors']+=1
                row['changed_tensors']+=int(not torch.equal(values,self.before[n]))
            result['update_audit']={'components':dict(components),'values_sampled_per_tensor_max':32,
                                    'initial_step':self.initial_global_step,'final_step':state.global_step}
            assert all(components[name]['changed_tensors']>0 for name in self.trainable_components),components
            frozen_changed=[n for n in self.frozen_names if not torch.equal(after[n],self.before[n])]
            assert not frozen_changed,frozen_changed
            result['update_audit']['frozen_sampled_tensors_changed']=len(frozen_changed)
            emit('train_end',**result['update_audit'])

    audit=Audit()
    collator=None
    class MultimodalStudio(UnslothTrainer):
        def _create_progress_callback(self):
            # This hook executes after SFT construction but before Studio preflight.
            trainer=self.trainer
            trainer.data_collator=collator
            trainer.args.remove_unused_columns=False
            trainer.model_accepts_loss_kwargs=False
            trainer.args.save_total_limit=args.save_total_limit
            trainer.args.max_grad_norm=1.0
            # Unsloth's generated SFTConfig defaults warmup_ratio to .1. An
            # explicit zero warmup_steps otherwise falls through to that ratio.
            trainer.args.warmup_ratio=0.0
            trainer.args.warmup_steps=warmup_steps
            trainer.args.lr_scheduler_type=type(trainer.args.lr_scheduler_type)(args.scheduler)
            assert trainer.args.get_warmup_steps(planned_updates)==warmup_steps
            assert trainer.args.bf16 and not trainer.args.fp16,'This workstation experiment requires BF16 AMP compute'
            trainer.add_callback(audit)
            if args.formal:
                from torch.utils.data import Sampler
                assert len(trainer.train_dataset)==len(records),'Studio changed formal epoch coverage'
                assert list(trainer.train_dataset['_record_index'])==list(range(len(records))),'Studio reordered formal row identities'
                class CompleteEpochSampler(Sampler):
                    def __init__(self):self.epoch=0
                    def __len__(self):return len(records)
                    def set_epoch(self,epoch):self.epoch=int(epoch)
                    def __iter__(self):return iter(epoch_record_indices(records,args.selection_seed,self.epoch))
                def formal_sampler(tr,train_dataset=None):return CompleteEpochSampler()
                trainer._get_train_sampler=types.MethodType(formal_sampler,trainer)
                def formal_loss(tr,model,inputs,return_outputs=False,num_items_in_batch=None):
                    indices=inputs.pop('_record_indices',None)
                    if indices is None or len(indices)!=1:
                        raise ValueError('Formal training requires exactly one bound record identity per microbatch')
                    index=int(indices[0]);record=records[index]
                    targets=int((inputs['labels'][:,1:]!=-100).sum().item())
                    positions=int(inputs['attention_mask'].sum().item())
                    text_only=inputs.pop('_text_only',False)
                    if bool(text_only)!=(record['task']=='text_replay' and not record['frames']):
                        raise ValueError('Text-only route differs from immutable record identity')
                    active_model=model.language_model if text_only else model
                    compute_device=next(active_model.parameters()).device.type
                    with torch.autocast(compute_device,dtype=torch.bfloat16,
                                        cache_enabled=args.parameter_precision!='fp32'):
                        enabled=torch.is_autocast_enabled(compute_device)
                        compute_dtype=torch.get_autocast_dtype(compute_device)
                        assert enabled and compute_dtype==torch.bfloat16,'Requested BF16 AMP context is not active'
                        if 'compute_precision_audit' not in result:
                            result['compute_precision_audit']={'device_type':compute_device,
                                'autocast_enabled_during_forward':enabled,'autocast_dtype':str(compute_dtype),
                                'parameter_dtype':str(next(active_model.parameters()).dtype),
                                'autocast_cache_enabled':torch.is_autocast_cache_enabled(),
                                'text_only_first_forward':bool(text_only)}
                            emit('compute_precision_audit',**result['compute_precision_audit'])
                        outputs=active_model(**inputs,use_cache=False)
                    if text_only:
                        model.last_batch_accounting={'input_positions':positions,'target_positions':targets,
                            'frames':0,'visual_positions':0,'text_only':True}
                    elif model.training and torch.is_grad_enabled():audit.window_visual=True
                    loss=outputs.loss
                    if not bool(torch.isfinite(loss.detach()).item()):raise ValueError('Nonfinite formal training loss')
                    weight=sample_weights[index]
                    loss=loss*weight
                    if model.training and torch.is_grad_enabled():
                        item={'record_index':index,'record_id':record['id'],'task':record['task'],
                            'language':record['language'],'optimizer_step_before_forward':tr.state.global_step,
                            'target_tokens_including_eos':targets,'sequence_positions':positions,
                            'frames':len(record['frames']),'objective_weight':weight}
                        exposure['examples_forwarded']+=1
                        exposure['target_tokens_including_eos']+=targets
                        exposure['sequence_positions']+=positions
                        exposure['frame_presentations']+=len(record['frames'])
                        exposure['presentation_counts'][index]+=1
                        for field,value in [('task_counts',record['task']),('language_counts',record['language'])]:
                            exposure[field][value]=exposure[field].get(value,0)+1
                        encoded=json.dumps(item,sort_keys=True,separators=(',',':')).encode()
                        exposure['rolling_sequence_sha256']=hashlib.sha256(exposure['rolling_sequence_sha256'].encode()+encoded).hexdigest()
                        exposure_log.write(json.dumps(item,separators=(',',':'))+'\n')
                    return (loss,outputs) if return_outputs else loss
                trainer.compute_loss=types.MethodType(formal_loss,trainer)
            elif args.train_samples is not None:
                from torch.utils.data import SequentialSampler
                def balanced_sampler(tr,train_dataset=None):
                    return SequentialSampler(tr.train_dataset if train_dataset is None else train_dataset)
                trainer._get_train_sampler=types.MethodType(balanced_sampler,trainer)
            if not args.formal and any(record['task']=='text_replay' for record in records):
                def mixed_loss(tr,model,inputs,return_outputs=False,num_items_in_batch=None):
                    text_only=inputs.pop('_text_only',False)
                    active_model=model.language_model if text_only else model
                    with torch.autocast(next(active_model.parameters()).device.type,dtype=torch.bfloat16,
                                        cache_enabled=args.parameter_precision!='fp32'):
                        outputs=active_model(**inputs,use_cache=False)
                    if text_only:
                        model.last_batch_accounting={'input_positions':int(inputs['attention_mask'].sum().item()),
                            'target_positions':int((inputs['labels'][:,1:]!=-100).sum().item()),
                            'frames':0,'visual_positions':0,'text_only':True}
                    return (outputs.loss,outputs) if return_outputs else outputs.loss
                trainer.compute_loss=types.MethodType(mixed_loss,trainer)

            def save_hook(tr, output_dir=None, state_dict=None):
                target=Path(output_dir or tr.args.output_dir)
                target.mkdir(parents=True,exist_ok=True)
                started=time.perf_counter()
                for relative,expected_hash in result['source_hashes'].items():
                    if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected_hash:
                        raise ValueError(f'Source changed during training; cannot certify checkpoint: {relative}')
                # Alias-aware saver handles Spark's tied embedding/lm_head.
                save_tensor_model(tr.model,str(target/'model.safetensors'),metadata={'format':'pt'})
                tr.model.save_bundle_config(target)
                tr.model.config.save_pretrained(target)
                tr.processing_class.save_pretrained(target)
                image_processor.save_pretrained(target)
                torch.save(tr.args,target/'training_args.bin')
                fingerprints=sample_hashes(tr.model)
                (target/'parameter_samples_sha256.json').write_text(json.dumps(fingerprints,indent=2)+'\n')
                (target/'training_state_probe.json').write_text(json.dumps({
                    'global_step':tr.state.global_step,'optimizer':optimizer_probe(tr.model,tr.optimizer),
                    'scheduler':tr.lr_scheduler.state_dict(),
                },indent=2)+'\n')
                tr.model.vision_encoder.config.to_json_file(target/'vision_config.json')
                # A reconstructable research bundle includes its original custom
                # backbone implementation and a snapshot of the local wrapper.
                for filename in ['configuration_spark.py','modeling_spark.py']:
                    shutil.copyfile(ROOT/'models/Spark-X2.6-4B'/filename,target/filename)
                implementation=target/'implementation'
                implementation.mkdir(exist_ok=True)
                for filename in ['__init__.py','model.py','collator.py','data.py']:
                    shutil.copyfile(ROOT/'multimodal'/filename,implementation/filename)
                shutil.copyfile(Path(__file__),implementation/'train_multimodal_studio.py')
                (target/'dataset_manifest_snapshot.json').write_bytes(manifest_bytes)
                (target/'training_selection.json').write_text(json.dumps(result['training_selection'],indent=2)+'\n')
                if args.formal:
                    snapshot={**exposure,'active_wall_seconds_at_checkpoint':exposure_summary()['active_wall_seconds'],
                        'completed_optimizer_step':tr.state.global_step,
                        'selected_record_ids_sha256':result['training_selection']['record_ids_sha256'],
                        'selected_records_sha256':result['training_selection']['selected_records_sha256']}
                    (target/'data_exposure.json').write_text(json.dumps(snapshot,indent=2)+'\n')
                (target/'experiment_sources.json').write_text(json.dumps({
                    'source_hashes':result['source_hashes'],'dataset_manifest_sha256':result['dataset_manifest_sha256'],
                    'dataset_binding':result['dataset_binding'],'training_selection':result['training_selection'],
                    'training_arguments':{key:(str(value) if isinstance(value,Path) else value)
                                          for key,value in vars(args).items()},
                    'training_contract':training_contract,
                    'base_model_source':json.loads((ROOT/'evidence/Spark-X2.6-4B-source.json').read_text()),
                    'vision_source':json.loads((ROOT/'evidence/multimodal-s0-vision-source.json').read_text()),
                },indent=2)+'\n')
                elapsed=time.perf_counter()-started
                emit('model_saved',path=str(target),seconds=elapsed)

            def load_hook(tr, resume_from_checkpoint, model=None):
                target=Path(resume_from_checkpoint)
                expected=json.loads((target/'multimodal_config.json').read_text())
                assert expected['pool_grid']==tr.model.pool_grid,'Incompatible visual prefix layout'
                started=time.perf_counter()
                missing,unexpected=load_tensor_model(model or tr.model,str(target/'model.safetensors'),strict=True,device='cpu')
                assert not missing and not unexpected,(missing,unexpected)
                actual=sample_hashes(model or tr.model)
                saved=json.loads((target/'parameter_samples_sha256.json').read_text())
                assert actual==saved,'Reloaded parameter samples differ from the saved checkpoint'
                result['checkpoint_reload']={'path':str(target),'seconds':time.perf_counter()-started,
                    'strict_tensor_load':True,'all_parameter_samples_match':True,'sampled_tensors':len(actual)}
                emit('checkpoint_reloaded',**result['checkpoint_reload'])

            trainer._save=types.MethodType(save_hook,trainer)
            trainer._load_from_checkpoint=types.MethodType(load_hook,trainer)
            original_checkpoint_saver=trainer._save_checkpoint
            def complete_checkpoint(tr,model,trial):
                started=time.perf_counter()
                original_checkpoint_saver(model,trial)
                target=Path(tr.args.output_dir)/f'checkpoint-{tr.state.global_step}'
                required=['model.safetensors','optimizer.pt','scheduler.pt','trainer_state.json','rng_state.pth','multimodal_config.json']
                if args.formal:required+=['data_exposure.json','experiment_sources.json','training_selection.json']
                if not all((target/name).is_file() for name in required):raise ValueError('Incomplete optimizer checkpoint')
                row={'path':str(target),'global_step':tr.state.global_step,'seconds':time.perf_counter()-started,
                    'bytes':sum(p.stat().st_size for p in target.rglob('*') if p.is_file()),'required_files_present':required}
                result['checkpoints']=[entry for entry in result['checkpoints'] if entry['path']!=str(target)]+[row]
                retained=[p for p in Path(tr.args.output_dir).glob('checkpoint-*') if p.is_dir()]
                for entry in result['checkpoints']:entry['retained']=Path(entry['path']) in retained
                pointer={'schema_version':1,'path':str(target),'global_step':tr.state.global_step,
                    'optimizer_resume_files_verified':True,'capability_promoted':False,
                    'checkpoint_sources_sha256':hashlib.sha256((target/'experiment_sources.json').read_bytes()).hexdigest()}
                (run_dir/'latest_checkpoint.json').write_text(json.dumps(pointer,indent=2)+'\n')
                candidates=[]
                for diagnostic in result.get('periodic_dev',[]):
                    checkpoint=Path(tr.args.output_dir)/f"checkpoint-{diagnostic['global_step']}"
                    if checkpoint in retained:candidates.append((diagnostic['token_weighted_loss'],checkpoint,diagnostic['global_step']))
                if candidates:
                    value,checkpoint,step=min(candidates)
                    (run_dir/'best_retained_dev_checkpoint.json').write_text(json.dumps({
                        'path':str(checkpoint),'global_step':step,'token_weighted_dev_nll':value,
                        'scope':'Lowest periodic dev NLL among currently retained checkpoints only; no automatic reload or capability promotion',
                        'capability_promoted':False},indent=2)+'\n')
                emit('checkpoint_complete',**row);save_result()
            trainer._save_checkpoint=types.MethodType(complete_checkpoint,trainer)
            original_rng_loader=trainer._load_rng_state
            def rng_hook(tr,checkpoint):
                original_rng_loader(checkpoint)
                saved=torch.load(Path(checkpoint)/'rng_state.pth',map_location='cpu',weights_only=False)
                cpu_match=torch.equal(torch.random.get_rng_state(),saved['cpu'])
                python_match=random.getstate()==saved['python']
                cuda_value=saved.get('cuda')
                cuda_match=(torch.equal(torch.cuda.random.get_rng_state(),cuda_value)
                            if torch.is_tensor(cuda_value) else None)
                assert cpu_match and python_match and cuda_match is not False,'RNG restore mismatch'
                result['rng_resume_audit']={'cpu_rng_matches':cpu_match,'python_rng_matches':python_match,
                                            'cuda_rng_matches':cuda_match}
                emit('rng_restored',**result['rng_resume_audit'])
            trainer._load_rng_state=types.MethodType(rng_hook,trainer)
            return super()._create_progress_callback()

        def _finalize_training(self,output_dir,label=''):
            if not args.no_save:
                start=time.perf_counter()
                path=Path(output_dir)/f'checkpoint-{self.trainer.state.global_step}'
                if not any(entry['path']==str(path) for entry in result['checkpoints']):
                    self.trainer._save_checkpoint(self.trainer.model,trial=None)
                required=['model.safetensors','optimizer.pt','scheduler.pt','trainer_state.json','rng_state.pth','multimodal_config.json']
                assert all((path/x).exists() for x in required),list(path.iterdir())
                pointer={'path':str(path),'global_step':self.trainer.state.global_step,
                    'final_for_this_session':True,'epoch_plan_complete':self.trainer.state.global_step>=planned_updates,
                    'termination_reason':result.get('termination_reason'),'capability_promoted':False}
                (run_dir/'final_checkpoint.json').write_text(json.dumps(pointer,indent=2)+'\n')
                emit('checkpoint_saved',**pointer)
            self._update_progress(is_training=False,is_completed=True,status_message='Bounded multimodal full-parameter experiment completed')

    bench=MultimodalStudio()
    start=time.perf_counter()
    loaded=bench.load_model(model_name=str(ROOT/'models/Spark-X2.6-4B'),max_seq_length=args.max_length,
        load_in_4bit=False,full_finetuning=True,trust_remote_code=True,use_gradient_checkpointing='unsloth',
        local_files_only=True,actual_model_repo_id='XHToken/Spark-X2.6-4B')
    if not loaded:raise RuntimeError(f'Studio load failed: {bench.model_load_error}')
    assert bench.prepare_model_for_training(use_lora=False)
    image_processor=SiglipImageProcessor.from_pretrained(vision_path,local_files_only=True)
    vision=SiglipVisionModel.from_pretrained(vision_path,local_files_only=True,torch_dtype=torch.bfloat16).to('cuda')
    assert not vision.vision_model.use_head,'Unused pooling head would invalidate full gradient coverage'
    wrapper=SparkMultimodalModel(bench.model,vision,pool_grid=args.pool_grid)
    for module in [wrapper.visual_projector,wrapper.position_projector]:module.to(device='cuda',dtype=torch.bfloat16)
    wrapper.media_embedding.data=wrapper.media_embedding.data.to(device='cuda',dtype=torch.bfloat16)
    # BF16 gamma values near 1 cannot represent a 1e-5 update. Preserve norm
    # parameters in FP32, as the working Spark runtime already does for its norms.
    for module in [wrapper.vision_encoder,wrapper.visual_projector,wrapper.position_projector]:
        for child in module.modules():
            if isinstance(child,torch.nn.LayerNorm):child.float()
    if args.initialize_from:
        source=Path(args.initialize_from)
        previous=json.loads((source/'multimodal_config.json').read_text())
        differences=prefix_layout_migration(previous,wrapper.architecture_manifest(),args.allow_prefix_layout_change)
        checkpoint_sources=json.loads((source/'experiment_sources.json').read_text())
        if checkpoint_sources['source_hashes']['multimodal/model.py']!=result['source_hashes']['multimodal/model.py']:
            raise ValueError('Weights-only source wrapper differs from current model.py')
        with safe_open(str(source/'model.safetensors'),framework='pt',device='cpu') as stream:
            embedding_name='language_model.model.embedding.weight'
            if embedding_name not in stream.keys():embedding_name=(stream.metadata() or {}).get(embedding_name)
            if embedding_name not in stream.keys():raise ValueError('Checkpoint lacks a resolvable tied embedding tensor')
            source_storage=stream.get_slice(embedding_name).get_dtype()
        if source_storage=='F32':
            if args.parameter_precision!='fp32':
                raise ValueError('FP32 checkpoint initialization requires --parameter-precision fp32 to avoid silent downcasting')
            # Preserve an FP32 source before loading; a BF16 temporary would
            # irreversibly discard its small accumulated updates.
            wrapper.float()
        missing,unexpected=load_tensor_model(wrapper,str(source/'model.safetensors'),strict=True,device='cpu')
        assert not missing and not unexpected
        expected=json.loads((source/'parameter_samples_sha256.json').read_text())
        assert sample_hashes(wrapper)==expected,'Initialized parameter samples mismatch'
        result['weights_only_initialization']={'checkpoint':str(source),'sampled_parameters_match':True,
                                               'optimizer_scheduler_rng':'new stage, not resumed'}
        if differences:
            result['weights_only_initialization']['prefix_layout_migration']={
                'explicitly_allowed':True,'architecture_differences':differences,
                'strict_tensor_load':True,'all_parameter_fingerprints_match':True,
                'new_or_reshaped_parameter_tensors':0,'optimizer_state_reused':False}
    if args.parameter_precision=='fp32':
        wrapper.float()
        wrapper.config.torch_dtype=torch.float32
        wrapper.vision_encoder.config.torch_dtype=torch.float32
        torch.set_autocast_cache_enabled(False)
    result['precision_policy']={'parameter_storage':args.parameter_precision,'compute':'BF16 automatic mixed precision',
        'autocast_cache_enabled':torch.is_autocast_cache_enabled(),
        'parameter_role':'FP32 parameters themselves retain small updates; no separate master copy' if args.parameter_precision=='fp32' else
            'BF16 parameter writeback without FP32 master copy; small updates may round away',
        'optimizer_state':'AdamW8bit with recorded FP32 exceptions',
        'checkpoint_initialization_verified_before_requested_storage_upcast':bool(args.initialize_from)}
    # The installed full-finetuning loader leaves pure-BF16 mode in the process
    # environment. Generated SFTTrainer then disables AMP despite bf16=True.
    # Set the supported mixed-precision policy before SFT/Accelerator creation.
    result['precision_policy']['environment_transition']=configure_amp_environment(os.environ)
    wrapper._unsloth_full_finetuning=True
    wrapper._unsloth_forced_float32=False
    wrapper._unsloth_user_float32=False
    if args.stage=='alignment':
        wrapper.language_model.requires_grad_(False)
        wrapper.vision_encoder.requires_grad_(False)
    bench.model=wrapper
    bench.is_vlm=False
    result['load_seconds']=time.perf_counter()-start
    result['architecture']=wrapper.architecture_manifest()
    (run_dir/'architecture.json').write_text(json.dumps(result['architecture'],indent=2)+'\n')
    emit('model_loaded',seconds=result['load_seconds'],**result['architecture'])
    class BoundTrainingCollator(MultimodalCollator):
        def __call__(self,rows):
            if not rows[0]['frames']:
                if len(rows)!=1 or rows[0].get('_task')!='text_replay':
                    raise ValueError('Zero-media batches are allowed only for batch-one true text_replay')
                row=rows[0]
                batch={key:torch.tensor([row[key]],dtype=torch.long) for key in ('input_ids','attention_mask','labels')}
                batch['_text_only']=True
            else:batch=super().__call__(rows)
            if args.formal and '_record_index' in rows[0]:
                if not all('_record_index' in row for row in rows):raise ValueError('Mixed training/dev identity batch')
                batch['_record_indices']=torch.tensor([row['_record_index'] for row in rows],dtype=torch.long)
            return batch
        def dataset_record(self,record):
            encoded=super().dataset_record(record)
            encoded['_task']=record['task']
            if not record['frames']:
                if record['task']!='text_replay':raise ValueError('Only text_replay may omit media')
            return encoded
    collator=BoundTrainingCollator(bench.tokenizer,image_processor,data_root,wrapper.tokens_per_frame,args.max_length)
    encoded_records=[collator.dataset_record(r) for r in records]
    if args.formal:
        for index,row in enumerate(encoded_records):row['_record_index']=index
    dataset=Dataset.from_list(encoded_records)
    result['dataset']={'train_examples':len(dataset),'sequence_lengths':[len(x) for x in dataset['input_ids']],
                       'task_counts':dict(collections.Counter(r['task'] for r in records)),
                       'language_counts':dict(collections.Counter(r['language'] for r in records))}
    result['dataset']['target_tokens_per_complete_epoch_including_eos']=sum(sum(token!=-100 for token in row['labels'][1:]) for row in encoded_records)
    result['dataset']['planned_target_tokens_including_eos']=(result['dataset']['target_tokens_per_complete_epoch_including_eos']*args.epochs if args.formal else None)

    @torch.no_grad()
    def evaluate(split,limit,ablate=False):
        wrapper.eval()
        rows=[]
        selected=load_records(data_root,split=split)
        if args.tasks:selected=[r for r in selected if r['task'] in args.tasks]
        if not selected:raise ValueError(f'No {split} records for the selected tasks')
        # Round-robin task/language strata rather than selecting only early image rows.
        groups=collections.defaultdict(list)
        for r in selected:groups[(r['task'],r['language'])].append(r)
        selected=[]
        while groups and len(selected)<limit:
            for key in list(groups):
                selected.append(groups[key].pop(0))
                if not groups[key]:del groups[key]
                if len(selected)>=limit:break
        for record in selected:
            batch={k:(v.to('cuda') if torch.is_tensor(v) else v) for k,v in collator([collator.dataset_record(record)]).items()}
            text_only=batch.pop('_text_only',False)
            active_model=wrapper.language_model if text_only else wrapper
            with torch.autocast('cuda',dtype=torch.bfloat16):
                normal=active_model(**batch)
            row={'id':record['id'],'task':record['task'],'language':record['language'],'target_tokens':int((batch['labels']!=-100).sum().item()),
                 'loss':float(normal.loss.item()),'text_only':bool(text_only)}
            del normal
            if ablate and not text_only:
                other={k:v.clone() for k,v in batch.items()}
                other['pixel_values'].zero_()
                with torch.autocast('cuda',dtype=torch.bfloat16):out=wrapper(**other)
                row['blank_media_loss']=float(out.loss.item());del out
                if len(record['frames'])>1:
                    other['pixel_values']=batch['pixel_values'].flip(1)
                    with torch.autocast('cuda',dtype=torch.bfloat16):out=wrapper(**other)
                    row['reversed_frames_loss']=float(out.loss.item());del out
            rows.append(row)
        wrapper.train()
        targets=sum(r['target_tokens'] for r in rows)
        out={'split':split,'examples':len(rows),'target_tokens':targets,
             'token_weighted_loss':sum(r['loss']*r['target_tokens'] for r in rows)/targets,'rows':rows,
             'note':'Teacher-forced target likelihood diagnostic, not free-generation accuracy'}
        if ablate:
            visual=[r for r in rows if not r['text_only']]
            out['blank_media_examples']=len(visual)
            out['blank_media_loss_delta']=(sum((r['blank_media_loss']-r['loss'])*r['target_tokens'] for r in visual)/
                sum(r['target_tokens'] for r in visual)) if visual else None
        return out

    if not args.resume_from:
        result['before_dev']=evaluate('dev',args.eval_limit,ablate=True)
        emit('before_dev',**{k:v for k,v in result['before_dev'].items() if k!='rows'})
    bench.is_training=True
    outdir=outputs_root()/args.run_id
    result['trainer_output_dir']=str(outdir)
    save_result()
    bench._train_worker({'dataset':dataset,'final_format':'pretokenized'},
        output_dir=str(outdir),batch_size=1,gradient_accumulation_steps=args.accum,
        max_steps=0 if args.formal else args.steps,max_seq_length=args.max_length,num_epochs=args.epochs if args.formal else 1,
        learning_rate=args.learning_rate,warmup_steps=warmup_steps,warmup_ratio=0.0,save_steps=args.save_every,
        random_seed=3407,weight_decay=.001,optim='adamw_8bit',lr_scheduler_type=args.scheduler,
        packing=False,train_on_completions=False,enable_wandb=False,enable_tensorboard=False,
        eval_steps=0.0,resume_from_checkpoint=args.resume_from)
    progress=bench.get_training_progress()
    result['studio_progress']=dataclasses.asdict(progress) if dataclasses.is_dataclass(progress) else str(progress)
    if getattr(progress,'error',None):raise RuntimeError(progress.error)
    assert getattr(progress,'is_completed',False),'Studio did not mark completion'
    assert result['steps'],'No optimizer step was measured'
    if args.resume_from:
        assert result.get('checkpoint_reload',{}).get('all_parameter_samples_match')
        previous=json.loads((Path(args.resume_from)/'trainer_state.json').read_text())['global_step']
        assert result['steps'][-1]['step']>previous,'Checkpoint did not resume training'
        result['resume_verification']={'previous_global_step':previous,'continued_to_global_step':result['steps'][-1]['step'],
            'optimizer_and_scheduler_restore':result.get('optimizer_scheduler_resume_audit'),
            'rng_restore':result.get('rng_resume_audit'),'parameters_matched_before_resume':True,
            'trajectory_equivalence_to_uninterrupted_run':'not tested'}
    if not (args.formal and result.get('termination_reason')):
        result['after_dev']=evaluate('dev',args.eval_limit,ablate=True)
    else:result['after_dev_skipped']={'reason':'Cooperative stop; preserve time for optimizer checkpoint'}
    if not args.skip_locked_test:
        result['after_test']=evaluate('test',args.eval_limit,ablate=True)
    if 'after_dev' in result:emit('after_dev',**{k:v for k,v in result['after_dev'].items() if k!='rows'})
    if 'after_test' in result:
        emit('after_test',**{k:v for k,v in result['after_test'].items() if k!='rows'})
    measured=[x for x in result['steps'] if not x['warmup']] or result['steps']
    result['summary']={'optimizer_steps_executed':len(result['steps']),
        'mean_optimizer_step_seconds':statistics.mean(x['seconds'] for x in measured),
        'median_optimizer_step_seconds':statistics.median(x['seconds'] for x in measured),
        'peak_allocated_GiB':max(x['peak_allocated_bytes'] for x in result['steps'])/2**30,
        'peak_reserved_GiB':max(x['peak_reserved_bytes'] for x in result['steps'])/2**30,
        'throughput_note':'Variable sequence/frame lengths; per-step microbatch accounting retained; do not equate to prior fixed-length text token/s'}
    if args.formal:
        result['data_exposure']=exposure_summary()
        covered=all(count==args.epochs for count in exposure['presentation_counts'])
        result['summary'].update({'planned_optimizer_updates':planned_updates,'epoch_plan_complete':covered,
            'active_wall_hours':result['data_exposure']['active_wall_seconds']/3600,
            'capability_promoted':False})
        if not covered and not result.get('termination_reason'):
            raise ValueError('Formal epoch training ended without complete record coverage or a documented graceful stop')
        result['status']='completed' if covered else 'stopped'
    else:result['status']='completed'
    emit(result['status'],**result['summary'])
except BaseException as exc:
    result['status']='failed';result['error']=repr(exc);result['traceback']=traceback.format_exc()
    emit('failed',error=repr(exc));traceback.print_exc()
finally:
    result['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    save_result();events.close()
    if 'exposure_log' in globals() and exposure_log is not None:exposure_log.close()
if result['status'] not in {'completed','stopped'}:sys.exit(1)
