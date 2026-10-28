"""Fake-Hub publication tests; no credentials, network, or live GPU qualification."""
import hashlib
from types import SimpleNamespace

import pytest

from colab.final.common import read,write
from colab.final.storage import manifest
from colab.final import publish

BOUND={'source_archive_sha256':'a'*64,'policy_sha256':'b'*64}
CONFIG={'campaign_id':'publication-test','model_repo':'TassP0lar/publication-test','model_repo_private':True}


class Hub:
    def __init__(self):
        self.exists=False
        self.private=True
        self.head='0'*40
        self.history={}
        self.uploads=0
        self.lose_response=False
        self.api_tokens=[]

    def whoami(self):
        return {'name':'TassP0lar'}

    def repo_exists(self,*args,**kwargs):
        return self.exists

    def create_repo(self,*args,**kwargs):
        self.exists=True
        self.private=kwargs['private']
        self.history[self.head]={'.gitattributes':b'*.safetensors filter=lfs'}

    def list_repo_files(self,*args,revision=None,**kwargs):
        return list(self.history[revision or self.head])

    def model_info(self,*args,revision=None,files_metadata=False,**kwargs):
        revision=revision or self.head
        siblings=[]
        if files_metadata:
            for name,payload in self.history[revision].items():
                lfs=SimpleNamespace(sha256=hashlib.sha256(payload).hexdigest()) if name.endswith('.safetensors') else None
                siblings.append(SimpleNamespace(rfilename=name,size=len(payload),lfs=lfs))
        return SimpleNamespace(private=self.private,sha=revision,siblings=siblings)

    def upload_folder(self,repo_id,folder_path,allow_patterns,**kwargs):
        self.uploads+=1
        contents=dict(self.history.get(self.head,{}))
        for name in allow_patterns:
            contents[name]=(folder_path/name).read_bytes()
        identity=hashlib.sha256(repr(sorted((name,hashlib.sha256(data).hexdigest()) for name,data in contents.items())).encode()).hexdigest()
        self.head=hashlib.sha1((self.head+identity).encode()).hexdigest()
        self.history[self.head]=contents
        if self.lose_response:
            self.lose_response=False
            raise RuntimeError('simulated lost upload response')
        return SimpleNamespace(oid=self.head)

    def read(self,path,revision,limit):
        payload=self.history[revision][path]
        assert len(payload)<=limit
        return payload


class Store:
    def __init__(self):
        self.receipts=[]

    def small_file(self,path,kind):
        self.receipts.append((read(path),kind))


@pytest.fixture
def publication(tmp_path,monkeypatch):
    out=tmp_path/'final-model'
    (out/'runtime').mkdir(parents=True)
    (out/'model.safetensors').write_bytes(b'closed-weights')
    (out/'config.json').write_bytes(b'{"mode":"clean"}')
    (out/'runtime/inference.py').write_bytes(b'def load():\n    return None\n')
    release={'schema_version':1,'campaign_id':CONFIG['campaign_id'],'binding':BOUND,'training_complete':True,**manifest(out)}
    write(out/'MODEL_RELEASE.json',release)
    campaign=SimpleNamespace(root=tmp_path,bound=BOUND,_hf_token='test-only-token',model=object(),store=Store())
    hub=Hub()

    def api(token):
        hub.api_tokens.append(token)
        return hub

    monkeypatch.setattr('huggingface_hub.HfApi',api)
    monkeypatch.setattr(publish,'_remote_bytes',lambda repo,path,revision,token,limit:hub.read(path,revision,limit))
    monkeypatch.setitem(__import__('sys').modules,'colab.final.inference',SimpleNamespace(load=lambda path:SimpleNamespace(parameters=lambda:[])))
    monkeypatch.setattr('torch.cuda.empty_cache',lambda:None)
    return campaign,hub,out


def test_upload_constructs_hfapi_and_verifies_complete_release(publication):
    campaign,hub,out=publication
    receipt=publish.upload_closed_export(campaign,CONFIG,out)
    assert hub.api_tokens==['test-only-token']
    assert hub.uploads==1
    assert receipt['remote_manifest_verified'] and receipt['remote_inventory_verified']
    assert receipt['model_content_sha256_verified'] and receipt['model_lfs_sha256_verified']
    assert receipt['files_verified']==3 and not receipt['already_committed']
    assert set(hub.history[receipt['commit']])=={'.gitattributes','MODEL_RELEASE.json','model.safetensors','config.json','runtime/inference.py'}
    assert campaign.store.receipts[0][0]==receipt


@pytest.mark.parametrize('damage',['corrupt','missing'])
def test_retry_rejects_corrupt_or_missing_nonweight_without_reupload(publication,damage):
    campaign,hub,out=publication
    revision=publish.upload_closed_export(campaign,CONFIG,out)['commit']
    if damage=='corrupt':
        # Equal length isolates content SHA256 from the separate size check.
        hub.history[revision]['config.json']=b'{"mode":"dirty"}'
        match='Git blob SHA256'
    else:
        del hub.history[revision]['config.json']
        match='inventory'
    (campaign.root/'MODEL_UPLOAD_RECEIPT.json').unlink()
    with pytest.raises(RuntimeError,match=match):
        publish.upload_closed_export(campaign,CONFIG,out)
    assert hub.uploads==1
    assert not (campaign.root/'MODEL_UPLOAD_RECEIPT.json').exists()


def test_lost_response_reconciles_exact_commit_without_second_upload(publication):
    campaign,hub,out=publication
    hub.lose_response=True
    with pytest.raises(RuntimeError,match='lost upload response'):
        publish.upload_closed_export(campaign,CONFIG,out)
    committed=hub.head
    receipt=publish.upload_closed_export(campaign,CONFIG,out)
    assert receipt['commit']==committed
    assert receipt['already_committed']
    assert hub.uploads==1
