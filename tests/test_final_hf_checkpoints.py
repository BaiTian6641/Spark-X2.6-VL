"""Failure-injected Hub contract tests. These are not live HF/GPU qualification."""
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from huggingface_hub import CommitOperationAdd,CommitOperationDelete
from huggingface_hub.hf_api import RepoFile
from colab.final.common import read,write,canonical
from colab.final.hf_checkpoints import HFCheckpointStore,MARKER,PREFIX,receipt_path
from colab.final.checkpoint_cache import prune,acknowledged,rebind_assets
from colab.final.storage import ColdStore,REQUIRED

CONFIG=read('colab/final/campaign.json')
BOUND={'source_archive_sha256':'a'*64,'policy_sha256':'b'*64}


class Hub:
    """In-memory remote boundary; uses the pinned library's real operation/types."""
    def __init__(self):
        self.exists=False; self.private=True; self.history={}; self.head='0'*40
        self.fail=False; self.lose_response=False; self.corrupt=None; self.missing_lfs=None; self.calls=0
    def whoami(self): return {'name':'TassP0lar'}
    def repo_exists(self,*a,**kw): return self.exists
    def create_repo(self,*a,**kw):
        self.private=kw['private']; self.exists=True; self.history[self.head]={'.gitattributes':b'*.pt filter=lfs'}
    def model_info(self,*a,**kw): return SimpleNamespace(private=self.private,sha=self.head)
    def list_repo_files(self,*a,revision=None,**kw): return list(self.history[revision or self.head])
    def list_repo_commits(self,*a,**kw): return [SimpleNamespace(commit_id=key) for key in self.history]
    def create_commit(self,repo,operations,parent_commit=None,**kw):
        self.calls+=1
        if self.fail: raise RuntimeError('simulated upload failure')
        assert parent_commit==self.head
        contents=dict(self.history[self.head])
        for op in operations:
            if isinstance(op,CommitOperationAdd):
                with op.as_file() as f: contents[op.path_in_repo]=f.read()
            elif isinstance(op,CommitOperationDelete): contents.pop(op.path_in_repo)
            else: raise AssertionError(type(op))
        self.head=hashlib.sha1(canonical({k:hashlib.sha256(v).hexdigest() for k,v in contents.items()})).hexdigest()
        self.history[self.head]=contents
        if self.lose_response:
            self.lose_response=False; raise RuntimeError('simulated lost commit response')
        return SimpleNamespace(oid=self.head)
    def list_repo_tree(self,repo,revision,path_in_repo,**kw):
        self.expand_seen=kw.get('expand',False)
        for name,data in self.history[revision].items():
            if not name.startswith(path_in_repo+'/'): continue
            digest=hashlib.sha256(data).hexdigest()
            if name==self.corrupt: digest='f'*64
            lfs={'size':len(data),'oid':digest,'pointerSize':128} if name.endswith(('.pt','.safetensors','.zip')) and name!=self.missing_lfs else None
            yield RepoFile(path=name,size=len(data),oid=hashlib.sha1(data).hexdigest(),lfs=lfs)


@pytest.fixture
def store(tmp_path):
    hub=Hub(); c={**CONFIG,'work_root':str(tmp_path)}
    value=HFCheckpointStore(tmp_path,c,BOUND,'test-only-secret',api=hub)
    value._read_remote=lambda name,revision,limit=2*1024**2: hub.history[revision][name]
    return value,hub


def snapshot(root,name='step-1',weight=b'weights'):
    directory=root/'checkpoints'/name;directory.mkdir(parents=True)
    for filename in REQUIRED: (directory/filename).write_bytes(weight if filename.endswith(('.pt','.safetensors')) else b'{}')
    write(directory/'BINDING.json',BOUND)
    write(directory/'COMPUTE_BUDGET.json',{'used_cu':1.25,'active_seconds':60.0,'observed_balance':'private-account','rate':8.9})
    (directory/'reload-assets').mkdir();(directory/'reload-assets/tokenizer.json').write_bytes(b'{"vocab":1}')
    return directory


def acknowledge(directory,receipt):
    write(receipt_path(directory),receipt)


def test_atomic_versioned_upload_verifies_all_files_and_restores_old_revision(store,monkeypatch):
    hf,hub=store; first=snapshot(hf.root)
    r1=hf.commit(first);assert r1['verified'] and r1['private']
    assert hub.expand_seen is True
    second=snapshot(hf.root,'step-2',b'new weights')
    (second/'reload-assets/tokenizer.json').unlink()
    r2=hf.commit(second)
    assert r1['revision']!=r2['revision']
    assert 'checkpoint/reload-assets/tokenizer.json' not in hub.history[r2['revision']]
    assert 'checkpoint/reload-assets/tokenizer.json' in hub.history[r1['revision']]
    assert hub.history[r1['revision']]['checkpoint/model.safetensors']==b'weights'
    assert not receipt_path(first).exists(), 'HF verification alone must not authorize cleanup before receipt backup'
    # Restore from the older exact commit even after the mutable branch changes.
    def fetch(url,path,entry,token):
        revision,filename=url.split('/resolve/')[1].split('/',1)
        assert revision==r1['revision'] and token=='test-only-secret'
        data=hub.history[revision][filename]
        assert hashlib.sha256(data).hexdigest()==entry['sha256']
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    monkeypatch.setattr('colab.final.hf_checkpoints.fetch',fetch)
    monkeypatch.setattr('colab.final.hf_checkpoints.disk_guard',lambda *a,**kw:None)
    recovered=hf.root/'checkpoints/restored'
    hf.restore(r1,recovered)
    assert (recovered/'model.safetensors').read_bytes()==b'weights'
    assert acknowledged(recovered,BOUND)
    assert not (hf.root/'hub').exists()
    assert 'test-only-secret' not in json.dumps(hub.history[r2['revision']],default=lambda b:b.decode(errors='ignore'))


def test_public_checkpoint_is_sanitized_and_restores(store,monkeypatch):
    hf,hub=store; hf.c['checkpoint_repo_private']=False; hf.c['allow_public_artifacts']=True
    d=snapshot(hf.root)
    write(d/'STATE.json',{'orchestration':{'parent_eval':{'by_stratum':{'image:en':{'mean_loss':1.0}},'rows':[{'reference':'protected-answer'}]},'candidate_order':['rid-1'],'reviews':[{'decision':{'priority':3,'training_action':'replay_original','rationale':'protected-rationale'}}]}})
    write(d/'config.json',{'observed_compute_balance':'private-account','instructor_route_eligibility_receipt':{'private':'evidence'}})
    write(d/'INSTRUCTOR_LEDGER.json',{'requests':[{'status':'accounted','reservation_usd':0.5,'decision':{'priority':3,'training_action':'replay_original','rationale':'protected-rationale'}}]})
    (d/'provenance').mkdir();write(d/'provenance/BASELINE.json',{'by_stratum':{'image:en':{'mean_loss':1.0}},'rows':[{'reference':'protected-answer'}]})
    write(d/'COMPUTE_BUDGET.json',{'used_cu':1.25,'active_seconds':60.0,'observed_balance':'private-account','rate':8.9})
    receipt=hf.commit(d)
    assert receipt['private'] is False and hub.private is False
    history=hub.history[receipt['revision']]
    public_text=json.dumps(history,default=lambda value:value.decode(errors='ignore') if isinstance(value,bytes) else value)
    assert 'protected-answer' not in public_text and 'protected-rationale' not in public_text and 'private-account' not in public_text
    assert json.loads(history['checkpoint/COMPUTE_BUDGET.json'])=={'used_cu':1.25,'active_seconds':60.0}
    state=json.loads(history['checkpoint/STATE.json'])
    assert state['orchestration']['candidate_order']==['rid-1']
    assert 'rows' not in state['orchestration']['parent_eval']
    assert 'rationale' not in state['orchestration']['reviews'][0]['decision']
    assert json.loads(history['checkpoint/config.json'])['observed_compute_balance'] is None
    def fetch(url,path,entry,token):
        revision,filename=url.split('/resolve/')[1].split('/',1)
        data=hub.history[revision][filename]
        assert hashlib.sha256(data).hexdigest()==entry['sha256']
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    monkeypatch.setattr('colab.final.hf_checkpoints.fetch',fetch)
    monkeypatch.setattr('colab.final.hf_checkpoints.disk_guard',lambda *a,**kw:None)
    recovered=hf.root/'checkpoints/public-restored'
    hf.restore(receipt,recovered)
    assert json.loads((recovered/'STATE.json').read_text())['orchestration']['candidate_order']==['rid-1']
    assert 'rows' not in json.loads((recovered/'provenance/BASELINE.json').read_text())
    from colab.final.common import ComputeBudget
    budget=ComputeBudget(CONFIG);budget.merge_checkpoint(read(recovered/'COMPUTE_BUDGET.json'))
    assert budget.state['used_cu']>=1.25 and budget.state['active_seconds']>=60.0

def test_lost_response_reconciles_same_transaction_without_another_commit(store):
    hf,hub=store;hf.ensure_repository();d=snapshot(hf.root)
    hub.lose_response=True
    with pytest.raises(RuntimeError,match='lost commit'):hf.commit(d)
    oid=hub.head;calls=hub.calls
    receipt=hf.commit(d)
    assert receipt['revision']==oid and hub.calls==calls


def test_remote_hash_failure_cannot_acknowledge_local_deletion(store):
    hf,hub=store;d=snapshot(hf.root);hub.corrupt='checkpoint/optimizer.pt'
    with pytest.raises(ValueError,match='LFS hash'):hf.commit(d)
    assert d.exists() and not acknowledged(d,BOUND)
    with pytest.raises(RuntimeError,match='Unverified'):prune(hf.root,BOUND,[])
    assert (d/'optimizer.pt').exists()


def test_large_regular_blob_uses_head_and_stream_hash(store,monkeypatch):
    hf,hub=store;d=snapshot(hf.root)
    payload=b'{\"payload\":\"'+b'x'*(8*1024**2)+b'\"}'
    (d/'PARAMETERS.json').write_bytes(payload)
    monkeypatch.setattr(hf,'_head_remote',lambda path,revision,prefix=PREFIX:SimpleNamespace(etag=hashlib.sha1(payload).hexdigest(),size=len(payload)),raising=False)
    streamed=[]
    def stream_hash(path,revision,expected_bytes,prefix=PREFIX):
        streamed.append((path,expected_bytes));assert hub.history[revision][PREFIX+path]==payload
        return hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(hf,'_stream_remote_sha256',stream_hash,raising=False)
    receipt=hf.commit(d)
    assert receipt['verified'] and streamed==[('PARAMETERS.json',len(payload))]


def test_missing_tree_lfs_uses_head_sha256_without_large_download(store,monkeypatch):
    hf,hub=store;d=snapshot(hf.root);payload=b'm'*(8*1024**2+1)
    (d/'model.safetensors').write_bytes(payload);hub.missing_lfs='checkpoint/model.safetensors'
    monkeypatch.setattr(hf,'_head_remote',lambda path,revision,prefix=PREFIX:SimpleNamespace(etag=hashlib.sha256(payload).hexdigest(),size=len(payload)),raising=False)
    def forbidden_download(*args):raise AssertionError('LFS content must be verified by HEAD, not downloaded')
    monkeypatch.setattr(hf,'_stream_remote_sha256',forbidden_download,raising=False)
    assert hf.commit(d)['verified']


def test_public_or_foreign_repository_is_never_overwritten(store):
    hf,hub=store;hf.ensure_repository();hub.private=False
    with pytest.raises(RuntimeError,match='private'):hf.ensure_repository()
    hub.private=True
    hub.history[hub.head]['unrelated.txt']=b'keep'
    with pytest.raises(RuntimeError,match='Unrelated'):hf.commit(snapshot(hf.root))
    assert hub.history[hub.head]['unrelated.txt']==b'keep'


def test_unused_same_campaign_destination_can_follow_setup_source_repair(store):
    hf,hub=store;original=hf.ensure_repository()
    hf.owner={**hf.owner,'binding':{**BOUND,'source_archive_sha256':'c'*64}}
    updated=hf.ensure_repository()
    assert updated!=original
    assert all(not name.startswith(PREFIX) for name in hub.history[updated])
    assert json.loads(hub.history[original]['CHECKPOINT_REPOSITORY.json'])['binding']==BOUND


def test_source_repair_never_rebinds_existing_or_historically_trained_repo(store):
    hf,hub=store;hf.commit(snapshot(hf.root))
    hf.owner={**hf.owner,'binding':{**BOUND,'source_archive_sha256':'c'*64}}
    with pytest.raises(RuntimeError,match='different campaign/source'):hf.ensure_repository()
    # Even if an operator cleared current files, old trained history is protected.
    hub.create_commit(hf.repo,parent_commit=hub.head,operations=[CommitOperationDelete(n) for n in hub.history[hub.head] if n.startswith(PREFIX)])
    with pytest.raises(RuntimeError,match='trained or unverified'):hf.ensure_repository()


def test_drive_receipt_failure_preserves_both_slots_and_retry_reuses_hf_commit(store):
    hf,hub=store;old=snapshot(hf.root,'old');acknowledge(old,hf.commit(old))
    new=snapshot(hf.root,'new',b'changed');cold=object.__new__(ColdStore)
    cold.root=hf.root;cold.binding=BOUND;cold.hf=hf
    def fail(*a,**kw):raise RuntimeError('simulated Drive receipt failure')
    cold.small_file=fail
    with pytest.raises(RuntimeError,match='Drive receipt'):cold.commit(new)
    oid=hub.head
    assert old.exists() and new.exists() and not acknowledged(new,BOUND)
    cold.small_file=lambda *a,**kw:{'commit_file_id':'index','commit_sha256':'d'*64}
    receipt=cold.commit(new)
    assert receipt['revision']==oid and acknowledged(new,BOUND)
    prune(hf.root,BOUND,[new])
    assert new.exists() and not old.exists()
    assert receipt_path(old).exists(), 'Small receipts remain after large local payloads are purged'


def test_retention_rejects_foreign_uncommitted_and_symlink_paths_without_deleting(store,tmp_path):
    hf,hub=store;old=snapshot(hf.root,'old');acknowledge(old,hf.commit(old))
    unknown=snapshot(hf.root,'unknown')
    with pytest.raises(RuntimeError,match='Unverified'):prune(hf.root,BOUND,[])
    assert old.exists() and unknown.exists()
    outside=tmp_path/'outside';outside.mkdir()
    link=hf.root/'checkpoints/link';link.symlink_to(outside,target_is_directory=True)
    with pytest.raises(RuntimeError,match='Unverified'):prune(hf.root,BOUND,[old,unknown])
    assert outside.exists()


def test_fresh_restore_makes_room_without_reserving_a_third_checkpoint(store):
    hf,hub=store;first=snapshot(hf.root,'first');acknowledge(first,hf.commit(first))
    second=snapshot(hf.root,'second');acknowledge(second,hf.commit(second))
    incoming=hf.root/'checkpoints/restored-third'
    prune(hf.root,BOUND,[second],incoming=incoming)
    assert not first.exists() and second.exists() and not incoming.exists()


def test_next_checkpoint_uses_retained_reload_assets_after_old_is_deleted(store):
    hf,hub=store;first=snapshot(hf.root,'first');acknowledge(first,hf.commit(first))
    second=snapshot(hf.root,'second');acknowledge(second,hf.commit(second))
    model=SimpleNamespace(paths={'spark':first/'reload-assets/spark','motion_source':Path('unused')})
    rebind_assets(model,second);prune(hf.root,BOUND,[second])
    assert model.paths['spark']==second/'reload-assets/spark' and second.exists() and not first.exists()


def test_new_process_keeps_verified_local_fallback_until_restore_finishes(store):
    hf,hub=store;fallback=snapshot(hf.root);acknowledge(fallback,hf.commit(fallback))
    prune(hf.root,BOUND,[None],incoming=hf.root/'checkpoints/incoming')
    assert fallback.exists()


def test_rsi_parent_restore_never_rewinds_compute_accounting():
    from colab.final.common import ComputeBudget
    budget=ComputeBudget(CONFIG);budget.observe(900,8.71)
    budget.state['used_cu']=42;budget.state['active_seconds']=100
    observation=budget.state['observed_at']
    budget.merge_checkpoint({'used_cu':10,'active_seconds':50,'observed_at':observation+1000,'rate':.01})
    assert budget.state['used_cu']>=42 and budget.state['active_seconds']>=100
    assert budget.state['observed_at']==observation and budget.state['rate']==8.71
    budget.merge_checkpoint({'used_cu':60,'active_seconds':200})
    assert budget.state['used_cu']==60 and budget.state['active_seconds']==200


def test_real_cpu_optimizer_snapshots_retry_then_purge_and_can_save_again(store,monkeypatch):
    """Exercise the actual engine writer; only Hub/Drive and CUDA sync are replaced."""
    import torch
    from colab.final.engine import checkpoint
    hf,hub=store
    class TinyModel(torch.nn.Linear):
        def named_groups(self):
            for name,p in self.named_parameters(): yield 'spark',name,p
    model=TinyModel(3,2);assets=hf.root/'assets/spark';assets.mkdir(parents=True)
    (assets/'tokenizer.json').write_text('{"persist":true}')
    model.paths={'spark':assets}
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    cold=object.__new__(ColdStore);cold.root=hf.root;cold.binding=BOUND;cold.hf=hf
    cold.small_file=lambda *a,**kw:{'commit_file_id':'test-index','commit_sha256':'d'*64}
    config={**hf.c,'initial_checkpoint_reservation_bytes':64*1024**2}
    monkeypatch.setattr('colab.final.engine.disk_guard',lambda *a,**kw:None)
    state={'stage':'unit-training','updates':0,'completed_stages':[]}
    write(hf.root/'compute-budget.json',{'used_cu':0.0,'active_seconds':0.0})
    def update():
        optimizer.zero_grad();model(torch.ones(1,3)).square().mean().backward()
        optimizer.step();scheduler.step();state['updates']+=1
    update();r1=checkpoint(model,optimizer,scheduler,state,hf.root,config,BOUND,cold)
    first=Path(state['last_checkpoint_local'])
    loaded=torch.load(first/'optimizer.pt',weights_only=True)
    assert loaded['state'] and acknowledged(first,BOUND)
    update();hub.fail=True
    with pytest.raises(RuntimeError,match='simulated upload failure'):
        checkpoint(model,optimizer,scheduler,state,hf.root,config,BOUND,cold)
    assert first.exists() and len([p for p in (hf.root/'checkpoints').glob('unit-*') if p.is_dir()])==2
    hub.fail=False
    r2=checkpoint(model,optimizer,scheduler,state,hf.root,config,BOUND,cold)
    second=Path(state['last_checkpoint_local'])
    assert r2['revision']!=r1['revision'] and second.exists() and not first.exists()
    assert (model.paths['spark']/'tokenizer.json').is_file()
    update();checkpoint(model,optimizer,scheduler,state,hf.root,config,BOUND,cold)
    assert not second.exists() and (model.paths['spark']/'tokenizer.json').is_file()
    assert len([p for p in (hf.root/'checkpoints').glob('unit-*') if p.is_dir()])==1


def test_deferred_initial_checkpoint_uploads_only_after_studio_success(store,monkeypatch):
    import torch
    from colab.final.engine import checkpoint,Campaign
    hf,hub=store
    class TinyModel(torch.nn.Linear):
        def named_groups(self):
            for name,p in self.named_parameters(): yield 'spark',name,p
    model=TinyModel(3,2);assets=hf.root/'assets/spark';assets.mkdir(parents=True)
    (assets/'tokenizer.json').write_text('{"persist":true}');model.paths={'spark':assets}
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr('colab.final.engine.disk_guard',lambda *a,**kw:None)
    cold=object.__new__(ColdStore);cold.root=hf.root;cold.binding=BOUND;cold.hf=hf
    cold.small_file=lambda *a,**kw:{'commit_file_id':'test-index','commit_sha256':'d'*64}
    config={**hf.c,'initial_checkpoint_reservation_bytes':64*1024**2}
    state={'stage':'deferred-initial','updates':0,'completed_stages':[]}
    write(hf.root/'compute-budget.json',{'used_cu':0.0,'active_seconds':0.0})
    deferred=checkpoint(model,optimizer,scheduler,state,hf.root,config,BOUND,cold,defer_upload=True)
    assert deferred['backend']=='local_deferred' and not deferred['verified'] and hub.calls==0
    campaign=object.__new__(Campaign);campaign.store=cold;campaign.state=state
    campaign.deferred_uploads=[deferred];campaign.deferred_receipts={}
    campaign.flush_deferred_uploads()
    assert campaign.deferred_uploads==[] and hub.calls>0
    assert state['last_checkpoint']['backend']=='huggingface' and state['last_checkpoint']['verified']


def test_upload_only_cli_is_cpu_and_refuses_a_concurrent_campaign(store,monkeypatch,capsys):
    import fcntl,sys
    from colab.final import run,engine
    hf,hub=store;directory=snapshot(hf.root)
    cold=object.__new__(ColdStore);cold.root=hf.root;cold.binding=BOUND;cold.hf=hf
    cold.small_file=lambda *a,**kw:{'commit_file_id':'test-index','commit_sha256':'d'*64}
    monkeypatch.setattr(engine,'Campaign',lambda *a,**kw:SimpleNamespace(store=cold,bound=BOUND))
    monkeypatch.setattr(run,'validate_config',lambda c:c)
    def forbidden(*a,**kw): raise AssertionError('Upload retry must not initialize CUDA')
    monkeypatch.setattr(run,'doctor',forbidden)
    monkeypatch.setattr(run.os,'environ',{})
    write(hf.root/'asset-receipt.json',{'paths':{}})
    path=hf.root/'config.json';write(path,hf.c)
    monkeypatch.setattr(sys,'argv',['run','upload-checkpoint','--config',str(path),'--source-sha','a'*64,'--checkpoint-directory',str(directory)])
    with (hf.root/'.campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='Another campaign process'):run.main({'HF_TOKEN':'test-only-secret'})
        assert not hub.exists
    run.main({'HF_TOKEN':'test-only-secret'})
    assert acknowledged(directory,BOUND)
    assert 'checkpoint_upload_recovered' in capsys.readouterr().out


def _fetch_hub_bytes(hub,url,path,entry,token):
    revision,filename=url.split('/resolve/')[1].split('/',1)
    data=hub.history[revision][filename]
    assert hashlib.sha256(data).hexdigest()==entry['sha256']
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)


def test_hf_durable_artifact_roundtrip_discovery_and_lost_response(store,monkeypatch):
    hf,hub=store;directory=hf.root/'durable-ledgers'/'qualification-profile-test';directory.mkdir(parents=True)
    hf.ensure_repository()
    write(directory/'QUALIFICATION.json',{'binding':BOUND,'passed':True,'continuation':{'ids':['image-1']}})
    hub.lose_response=True
    with pytest.raises(RuntimeError,match='lost commit'):hf.commit_artifact(directory,'qualification-profile')
    revision=hub.head;calls=hub.calls
    receipt=hf.commit_artifact(directory,'qualification-profile')
    assert receipt['revision']==revision and hub.calls==calls and receipt['verified']
    found=hf.discover_artifacts()
    assert len(found)==1 and found[0]['commit']['binding']==BOUND
    monkeypatch.setattr('colab.final.hf_checkpoints.fetch',lambda url,path,entry,token:_fetch_hub_bytes(hub,url,path,entry,token))
    monkeypatch.setattr('colab.final.hf_checkpoints.disk_guard',lambda *a,**kw:None)
    restored=hf.root/'recovered-profile';value=hf.restore_artifact(receipt,restored)
    assert value['kind']=='qualification-profile'
    assert read(restored/'QUALIFICATION.json')['continuation']['ids']==['image-1']


def test_public_durable_artifact_sanitizes_instructor_rationale(store):
    hf,hub=store;hf.c['checkpoint_repo_private']=False;hf.c['allow_public_artifacts']=True
    directory=hf.root/'durable-ledgers'/'instructor-intent-test';directory.mkdir(parents=True)
    write(directory/'LEDGER.json',{'observed_compute_balance':'private-account','requests':[{'id':'request-1','status':'reserved_ambiguous','decision':{'priority':1,'rationale':'protected-rationale'}}]})
    receipt=hf.commit_artifact(directory,'instructor-intent')
    history=hub.history[receipt['revision']]
    text=json.dumps(history,default=lambda value:value.decode(errors='ignore') if isinstance(value,bytes) else value)
    assert 'private-account' not in text and 'protected-rationale' not in text
    payload=json.loads(history['ledger/instructor-intent-test/LEDGER.json'])
    assert payload['requests'][0]['id']=='request-1' and payload['observed_compute_balance'] is None
    assert 'rationale' not in payload['requests'][0]['decision']
