"""Durability, split, role and curriculum contracts for the formal runner."""
import gzip,json,sqlite3
from pathlib import Path
import pytest
from colab.final.common import ComputeBudget,read,identity
from colab.final.data import Catalogue,CommittedCursor,build_database,conditioning_members,family,training_views
from colab.final.storage import manifest

CONFIG=read('colab/final/campaign.json')

def test_optimizer_commit_rewinds_prefetched_examples():
    c=CommittedCursor(list('abcde')); assert [c.issue() for _ in range(5)]==list('abcde')
    c.commit(2); saved=c.state(); assert saved['committed']==2
    restored=CommittedCursor(list('abcde'),saved); assert restored.issue()=='c'
    with pytest.raises(ValueError): CommittedCursor(list('bcdea'),saved)



def test_cursor_dataset_supports_unsloth_probe_and_dataloader_resume():
    from torch.utils.data import DataLoader
    from colab.final.data import cursor_dataset
    cursor=CommittedCursor(list('abcde'));cursor.record_computed(0);cursor.record_computed(1);cursor.commit(2)
    stream=cursor_dataset(cursor.order,cursor)
    assert len(stream)==3 and stream[0]=={'rid':'c','_index':2}
    with pytest.raises(IndexError): stream[3]
    batches=list(DataLoader(stream,batch_size=2,num_workers=0))
    assert [batch['rid'] for batch in batches]==[['c','d'],['e']]
    assert [batch['_index'].tolist() for batch in batches]==[[2,3],[4]]


def test_video_decoder_uses_decoded_frame_presentation_clock():
    import av,io,numpy as np
    from colab.final.data import decode_video
    buffer=io.BytesIO()
    with av.open(buffer,'w',format='mp4') as output:
        stream=output.add_stream('mpeg4',rate=4);stream.width=32;stream.height=24;stream.pix_fmt='yuv420p'
        for index in range(8):
            frame=av.VideoFrame.from_ndarray(np.full((24,32,3),index*30,dtype=np.uint8),format='rgb24')
            for packet in stream.encode(frame): output.mux(packet)
        for packet in stream.encode(): output.mux(packet)
    frames,times=decode_video(buffer.getvalue(),4)
    assert len(frames)==4 and all(frame.size==(32,24) for frame in frames)
    assert times==pytest.approx([0.0,0.5,1.25,1.75])


def test_media_collator_has_tensor_batch_size_anchor(tmp_path):
    from accelerate.utils.operations import find_batch_size
    from contextlib import contextmanager
    from types import SimpleNamespace
    from colab.final.engine import MediaCollator
    row={'_family':'text','language':'en','task':'text_replay','prompt':'Answer','answer':'One'}
    class Rows:
        def row(self,rid): return row,{'transport':None}
    class Reader:
        @contextmanager
        def packet(self,value):
            yield lambda name: (_ for _ in ()).throw(ValueError('Text has no media'))
    campaign=SimpleNamespace(root=tmp_path,c=CONFIG,_hf_token=None,reader=SimpleNamespace(checkpoint_slot_bytes=1))
    collator=MediaCollator(campaign);collator.worker=(Rows(),Reader())
    batch=collator([{'rid':'row','_index':4}])
    assert find_batch_size(batch)==1 and batch['source_indices'].tolist()==[4]
    assert batch['examples'][0][0]['_family']=='text'


def test_native_ragged_batches_work_without_accelerate_dispatch():
    from accelerate.data_loader import prepare_data_loader
    from torch.utils.data import DataLoader
    from colab.final.data import cursor_dataset
    cursor=CommittedCursor(['a','b','c']);dataset=cursor_dataset(cursor.order,cursor)
    def collate(rows):
        import torch
        return {'source_indices':torch.tensor([row['_index'] for row in rows]),
                'examples':[{'rid':row['rid'],'text':'native-string'} for row in rows]}
    loader=DataLoader(dataset,batch_size=2,num_workers=0,collate_fn=collate)
    prepared=prepare_data_loader(loader,num_processes=1,dispatch_batches=False,put_on_device=False)
    batches=list(prepared)
    assert [batch['source_indices'].tolist() for batch in batches]==[[0,1],[2]]
    assert [[row['rid'] for row in batch['examples']] for batch in batches]==[['a','b'],['c']]

def test_output_targets_never_condition_recognition_or_generation():
    row={'task':'speech_synthesis','audio':['voice.flac'],'audio_direction':'text_to_audio','media_role':'target','target':{'kind':'waveform','path':'voice.flac'}}
    assert family(row)=='speech_out'; assert conditioning_members(row)==[]
    bad={**row,'media_role':'input'}
    with pytest.raises(ValueError): family(bad)
    row.update(task='sound_generation'); assert family(row)=='sound_out'


def test_complete_score_views_preserve_every_event_and_full_audio():
    notes=[{'clip_start_seconds':i/10,'clip_end_seconds':i/10+.05,'gm_program_1based':41,'midi_note':60+i%12} for i in range(61)]
    row={'id':'notes','task':'music_instrument_note_understanding','language':'en','audio':['music.flac'],'answer':json.dumps({'active_instruments':[{'gm_program_1based':41}],'score_aligned_note_events':notes})}
    views=training_views(row); assert len(views)==3
    assert sum(len(json.loads(v['answer'])['notes']) for v in views)==61
    assert all(v['audio']==['music.flac'] and v['_derived_task']['waveform_crop'] is False for v in views)
    assert json.loads(views[-1]['answer'])['notes'][-1][0]==6.0


def test_shard_local_interleaving_does_not_thrash_same_source(tmp_path):
    path=tmp_path/'corpus.sqlite';c=sqlite3.connect(path)
    c.executescript('CREATE TABLE packets(pid INTEGER,component TEXT,shard TEXT); CREATE TABLE rows(rid TEXT,pid INTEGER,family TEXT,language TEXT,split TEXT);')
    for component in ('a','b'):
        for shard in range(3):
            pid=len(c.execute('select * from packets').fetchall())+1
            c.execute('INSERT INTO packets VALUES(?,?,?)',(pid,component,str(shard)))
            c.executemany('INSERT INTO rows VALUES(?,?,?,?,?)',[(f'{component}{shard}-{i}',pid,'image','en','train') for i in range(300)])
    c.commit(); c.close(); order=Catalogue(path).plan({'families':['image'],'epochs':1},123)
    assert len(order)==1800 and len(set(order))==1800
    for component in ('a','b'):
        sequence=[r[1] for r in order if r.startswith(component)]
        transitions=[s for i,s in enumerate(sequence) if i==0 or sequence[i-1]!=s]
        assert len(transitions)==3 and len(set(transitions))==3


def test_shared_article_and_task_variants_stay_in_one_split(tmp_path):
    path=tmp_path/'packets.gz'
    with gzip.open(path,'wt') as f:
        for i in range(40):
            for modality in ('image','text'):
                row={'id':f'{modality}-{i}','source_group':f'wikipedia-en:{i}','source_id':f'wikipedia-en:{i}','language':'en','task':'document_ocr' if modality=='image' else 'text_replay','frames':['page.png'] if modality=='image' else [],'audio':[],'video':[],'prompt':'Read','answer':'Content','metadata':{}}
                f.write(json.dumps({'component':modality,'transport':None,'record':row})+'\n')
    db=tmp_path/'c.sqlite'; report=build_database(path,db,CONFIG); c=sqlite3.connect(db)
    assert report['original_instructions']==80
    assert c.execute('SELECT count(*) FROM (SELECT group_id FROM rows GROUP BY group_id HAVING count(distinct split)>1)').fetchone()[0]==0
    assert {r[0] for r in c.execute('select distinct split from rows')}=={'train','dev','promotion','audit'}


def test_missing_or_insufficient_compute_never_counts_as_completion():
    b=ComputeBudget(CONFIG)
    with pytest.raises(RuntimeError): b.check()
    b.observe(50,8.71)
    with pytest.raises(RuntimeError): b.check(100*3600)
    assert b.check(0)<50


def test_checkpoint_continuation_requires_new_process_but_not_new_vm():
    import subprocess,sys
    from colab.final.qualify import process_identity,require_new_process
    current=process_identity()
    with pytest.raises(RuntimeError,match='new campaign subprocess'):
        require_new_process({'qualification_process':current})
    with pytest.raises(RuntimeError):require_new_process({})
    code='import json,sys; from colab.final.qualify import require_new_process; print(json.dumps(require_new_process(json.loads(sys.argv[1]))))'
    output=subprocess.check_output([sys.executable,'-c',code,json.dumps({'qualification_process':current})],text=True)
    restored=json.loads(output)
    assert restored['boot_id']==current['boot_id'] and restored['pid']!=current['pid']


def test_bf16_gradient_parity_allows_reduction_noise_but_not_coherent_bias():
    import torch
    from colab.final.qualify import check_kernel_gradient_parity
    reference=torch.zeros(16);reference[14]=0.00023815
    candidate=reference.clone();candidate[14]+=0.0002667596563696861
    parameter=torch.nn.Parameter(torch.zeros(16));parameter.grad=candidate.clone()
    report=check_kernel_gradient_parity({'weight':reference},{'weight':float(reference.norm())},[('weight',parameter)])
    assert report['max_sample_absolute_error']==pytest.approx(0.0002667596563696861)
    # Each element is below the absolute floor, but their coherent direction
    # creates a full-tensor norm error that must still reject the gradient.
    wrong=torch.nn.Parameter(torch.zeros(16));wrong.grad=torch.full((16,),0.00049)
    with pytest.raises(RuntimeError,match='full-tensor gradient norm'):
        check_kernel_gradient_parity({'weight':torch.zeros(16)},{'weight':0.0},[('weight',wrong)])


def test_bf16_parity_distinguishes_vector_noise_from_coherent_error():
    import torch
    from colab.final.qualify import check_kernel_gradient_parity
    reference=torch.full((16,),0.01);reference[8]=0.004577636
    candidate=reference.clone();candidate[8]+=0.00164794921875
    parameter=torch.nn.Parameter(torch.zeros(16));parameter.grad=candidate.clone()
    check_kernel_gradient_parity({'weight':reference},{'weight':float(reference.norm())},[('weight',parameter)])
    isolated=torch.zeros(16);isolated[8]=0.004577636
    wrong=torch.nn.Parameter(torch.zeros(16));wrong.grad=isolated.clone()
    wrong.grad[8]+=0.00164794921875
    with pytest.raises(RuntimeError,match='full-tensor gradient norm'):
        check_kernel_gradient_parity({'weight':isolated},{'weight':float(isolated.norm())},[('weight',wrong)])


def test_closed_artifact_rejects_secret_paths(tmp_path):
    (tmp_path/'credential.json').write_text('{}')
    with pytest.raises(ValueError): manifest(tmp_path)


def test_teacher_decisions_never_create_new_ground_truth():
    from colab.final.teacher import replay_orders,eligibility
    selected,control=replay_orders([('a',{'training_action':'replay_original','priority':3}),('b',{'training_action':'replay_original','priority':0})],3,8)
    assert len(selected)==len(control)==16 and set(selected)<=set(control)=={'a','b'}
    with pytest.raises(RuntimeError): eligibility({**CONFIG,'instructor_enabled':False})

def test_family_language_objective_balances_supervision_without_dropping_rows():
    from colab.final.engine import loss_weights
    class SourceCatalogue:
        def select(self,split):
            return [(f'en-{i}','image','en','','') for i in range(100)]+[(f'zh-{i}','image','zh','','') for i in range(10)]+[(f't-{i}','text','en','','') for i in range(20)]
    rows=SourceCatalogue().select('train');order=[r[0] for r in rows]
    weights=loss_weights(SourceCatalogue(),order,CONFIG)
    assert len(order)==130
    assert 100*weights['image:en']==pytest.approx(10*weights['image:zh'])
    assert 100*weights['image:en']==pytest.approx(20*weights['text:en'])
    assert max(weights.values())<=CONFIG['max_loss_weight']

def test_dataset_format_peeks_and_prefetch_do_not_skip_training_records():
    from colab.final.data import cursor_rows
    cursor=CommittedCursor(['a','b','c','d'])
    assert next(cursor_rows(cursor.order,cursor))['rid']=='a'
    assert next(cursor_rows(cursor.order,cursor))['rid']=='a'
    assert cursor.issued==cursor.committed==0
    iterator=cursor_rows(cursor.order,cursor);prefetched=[next(iterator) for _ in range(3)]
    assert cursor.issued==0
    for item in prefetched[:2]:cursor.record_computed(item['_index'])
    cursor.commit(2);restored=CommittedCursor(cursor.order,cursor.state())
    assert next(cursor_rows(restored.order,restored))=={'rid':'c','_index':2}


def test_qualification_requires_each_named_gradient_and_records_writeback():
    import torch
    from colab.final.qualify import audit_parameter_coverage
    class Tiny(torch.nn.Linear):
        def named_groups(self):
            for name,p in self.named_parameters(): yield 'spark',name,p
    model=Tiny(2,1)
    before={'weight':[0.0],'bias':[0.0]}; after={'weight':[1.0],'bias':[0.0]}
    with pytest.raises(RuntimeError,match='only zero gradients'):
        audit_parameter_coverage(model,before,after,{'weight'})
    changed,writeback=audit_parameter_coverage(model,before,after,{'weight','bias'})
    assert changed=={'spark':1}
    assert writeback=={'weight':True,'bias':False}


def test_recovery_proof_is_published_only_after_initial_reset(tmp_path,monkeypatch):
    import torch
    from contextlib import nullcontext
    from colab.final.common import write
    from colab.final.qualify import recovery_check
    monkeypatch.setattr('colab.final.qualify.require_new_process',lambda profile:{'pid':123})
    monkeypatch.setattr(torch,'autocast',lambda *args,**kwargs:nullcontext())
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight=torch.nn.Parameter(torch.zeros(1))
        def loss(self,row,media): return self.weight.sum(),None
    initial={'commit_sha256':'a'*64}; complete={'commit_sha256':'b'*64}
    class Store:
        def __init__(self): self.uploads=[]
        def small_file(self,path,kind):
            self.uploads.append(kind); return {'kind':kind}
    class FakeCampaign:
        def __init__(self,fail_initial):
            self.root=tmp_path; self.c={'clip_norm':1.0}; self.bound={'source':'test'}
            self.model=Model(); self.optimizer=torch.optim.SGD(self.model.parameters(),lr=0.0)
            self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lambda step:1.0)
            self.store=Store(); self.fail_initial=fail_initial; self.calls=[]; self.state={}
        def media(self,rid): return None,None
        def restore(self,receipt):
            self.calls.append(receipt)
            if receipt==initial and self.fail_initial: raise RuntimeError('simulated initial restore failure')
            self.state={'stage':'restored','updates':9,'completed_stages':['stale'],'rsi_rounds':[{'round':9}],'cursor':{'stale':True},'last_checkpoint':receipt}
            return self.optimizer,self.scheduler,{'fresh_vm':False,'receipt':receipt}
    write(tmp_path/'metadata/corpus.report.json',{'split_membership_sha256':'dataset'})
    profile={'qualification_initial_receipt':initial,'qualification_complete_receipt':complete,
             'continuation':{'ids':['one','two'],'losses':[0.0,0.0],
                             'parameter_samples':{'weight':[0.0]},'tolerance':1e-4}}
    failed=FakeCampaign(True)
    with pytest.raises(RuntimeError,match='simulated initial restore failure'):
        recovery_check(failed,profile)
    assert failed.calls==[complete,initial]
    assert not (tmp_path/'FULL_RECOVERY_PROOF.json').exists() and failed.store.uploads==[]
    campaign=FakeCampaign(False); result=recovery_check(campaign,profile)
    proof=read(tmp_path/'FULL_RECOVERY_PROOF.json'); reset=read(tmp_path/'CAMPAIGN.json')
    assert campaign.calls==[complete,initial] and campaign.store.uploads==['full-recovery-proof']
    assert result['proof']==proof and proof['initial_reset_complete'] is True
    assert proof['initial_reset_receipt']==initial and proof['initial_restore_proof']['receipt']==initial
    assert proof['initial_reset_state']==reset and proof['initial_reset_state_sha256']==identity(reset)
    assert reset['stage']=='initial' and reset['updates']==0 and reset['cursor'] is None
    assert reset['completed_stages']==reset['rsi_rounds']==[]


@pytest.mark.parametrize('phase',['candidate','control'])
def test_rsi_stop_skips_evaluation_and_phase_completion(tmp_path,phase):
    from colab.final.engine import Campaign
    campaign=object.__new__(Campaign); campaign.root=tmp_path; campaign.stop_reason='direct_wall_hours'
    session={'round':1,'master':{'completed_stages':[],'rsi_rounds':[]},
             'candidate_order':['a'],'control_order':['b'],'loss_weights':{}}
    if phase=='control': session.update(candidate_receipt={'commit_sha256':'c'*64},candidate_eval={})
    campaign.state={'stage':f'rsi-1-{phase}','cursor':{'committed':0},'orchestration':session,
                    'last_checkpoint':{'commit_sha256':'d'*64}}
    campaign.sync_instructor_ledger=lambda:None
    trained=[]; campaign.train_order=lambda stage,order,resume:trained.append((stage['name'],resume))
    campaign.evaluate=lambda *args,**kwargs:pytest.fail('a stopped RSI phase must not be evaluated')
    result=campaign.instructed_rsi(1)
    assert result['stopped'] is True and result['reason']=='direct_wall_hours' and result['stage']==f'rsi-1-{phase}'
    assert trained==[(f'rsi-1-{phase}',True)]
    assert phase+'_receipt' not in session and phase+'_eval' not in session
    assert read(tmp_path/'CAMPAIGN.json')['stopped_reason']=='direct_wall_hours'


def test_existing_checkpoint_retry_enforces_measured_reservation(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from colab.final import engine
    from colab.final.common import write
    class Model:
        def named_parameters(self): return []
    model=Model(); bound={'source':'test'}
    write(tmp_path/'compute-budget.json',{'used_cu':0,'active_seconds':0})
    directory=tmp_path/'checkpoints/unit-00000000'; directory.mkdir(parents=True)
    for name in ('model.safetensors','optimizer.pt','scheduler.pt','rng.pt','STATE.json','config.json','CLOSED.json','COMPUTE_BUDGET.json'):
        (directory/name).write_bytes(b'x')
    write(directory/'BINDING.json',bound); write(directory/'PARAMETERS.json',{'sample_sha256':engine.sample_hash(model)})
    monkeypatch.setattr(engine,'acknowledged',lambda *args:False)
    store=SimpleNamespace(commit=lambda path:pytest.fail('an oversized retry must not be committed'))
    state={'stage':'unit','updates':0,'qualified_checkpoint_reservation_bytes':1,'checkpoint_bytes':0}
    with pytest.raises(RuntimeError,match='exceeds its reservation'):
        engine.checkpoint(model,None,None,state,tmp_path,{'initial_checkpoint_reservation_bytes':1},bound,store)
    assert directory.exists()


def test_checkpoint_requires_persistent_compute_budget(tmp_path):
    from colab.final.engine import checkpoint
    state={'stage':'unit','updates':0,'checkpoint_bytes':0}
    with pytest.raises(RuntimeError,match='Missing compute-budget.json'):
        checkpoint(object(),None,None,state,tmp_path,{'initial_checkpoint_reservation_bytes':1},{},object())
    assert not (tmp_path/'checkpoints').exists()


def test_runtime_snapshot_includes_flash_attention_requirements(tmp_path):
    from types import SimpleNamespace
    from colab.final.engine import copy_reload_assets
    assets=tmp_path/'assets'; assets.mkdir(); (assets/'tokenizer.json').write_text('{}')
    copy_reload_assets(SimpleNamespace(paths={'spark':assets}),tmp_path/'snapshot')
    for suffix in ('in','lock'):
        assert (tmp_path/'snapshot/runtime/colab/final'/f'requirements-flash-attn.{suffix}').is_file()


def test_instructor_ledger_merge_never_rewinds_current_requests():
    from colab.final.engine import merge_instructor_ledgers
    accounted={'id':'a','status':'accounted','decision':{'priority':3}}
    current={'ceiling_usd':10,'requests':[accounted,{'id':'b','status':'reserved_ambiguous'}]}
    restored={'ceiling_usd':1,'requests':[{'id':'a','status':'reserved_ambiguous'},
              {'id':'b','status':'accounted','decision':{'priority':2}}, {'id':'c','status':'accounted'}]}
    merged=merge_instructor_ledgers(current,restored)
    assert merged['ceiling_usd']==10 and [r['id'] for r in merged['requests']]==['a','b','c']
    assert merged['requests'][0]==accounted
    assert merged['requests'][1]['status']=='accounted' and merged['requests'][1]['decision']=={'priority':2}
    assert current['requests'][1]['status']=='reserved_ambiguous'


def test_hf_instructor_sync_uses_embedded_commit_without_drive(tmp_path):
    from colab.final.common import write
    from colab.final.engine import Campaign
    bound={'source':'test'}
    current={'requests':[{'id':'local','status':'accounted','decision':{'priority':3}}]}
    restored={'requests':[{'id':'local','status':'reserved_ambiguous'}, {'id':'remote','status':'accounted'}]}
    class Store:
        def discover(self):
            return [{'name':'instructor-receipt-remote','commit_sha256':'f'*64,'kind':'instructor-receipt',
                     'backend':'huggingface_artifact','commit':{'binding':bound,'created_unix':10}}]
        def restore(self,receipt,destination):
            write(destination/'LEDGER.json',restored)
    campaign=object.__new__(Campaign); campaign.root=tmp_path; campaign.bound=bound; campaign.store=Store()
    write(tmp_path/'instructor/LEDGER.json',current)
    campaign.sync_instructor_ledger()
    merged=read(tmp_path/'instructor/LEDGER.json')
    assert [entry['id'] for entry in merged['requests']]==['local','remote']
    assert merged['requests'][0]['status']=='accounted' and merged['requests'][0]['decision']=={'priority':3}
