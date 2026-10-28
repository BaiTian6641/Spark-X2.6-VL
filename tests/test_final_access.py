"""Gate failures must name the required action without inventing their cause."""
from types import SimpleNamespace
import pytest
import requests
import huggingface_hub as hub
from huggingface_hub.errors import GatedRepoError,HfHubHTTPError
from colab.final.assets import check_gated_access
from colab.final.common import read,write

SPEC={'repository':'stabilityai/stable-audio-open-1.0','revision':'a'*40,
      'files':[{'path':'model.safetensors','bytes':42,'sha256':'b'*64}]}


def test_access_uses_metadata_without_following_the_signed_weight_redirect(monkeypatch,capsys):
    def metadata(url,**kwargs):
        assert url.endswith('/'+'a'*40+'/model.safetensors')
        assert kwargs['token']=='test-secret'
        return SimpleNamespace(commit_hash='a'*40,size=42,etag='b'*64)
    monkeypatch.setattr(hub,'get_hf_file_metadata',metadata)
    check_gated_access(SPEC,'test-secret')
    assert 'model_access_verified' in capsys.readouterr().out


@pytest.mark.parametrize('kind,status,match',[(GatedRepoError,403,'GatedRepo.*review/request access'),
                                          (HfHubHTTPError,503,'HTTP 503.*does not establish missing license')])
def test_access_denial_is_distinct_from_hub_failure(monkeypatch,kind,status,match):
    response=requests.Response();response.status_code=status
    def metadata(*a,**kw): raise kind('sensitive transport response',response=response)
    monkeypatch.setattr(hub,'get_hf_file_metadata',metadata)
    with pytest.raises(RuntimeError,match=match) as exc: check_gated_access(SPEC,'test-secret')
    assert 'sensitive transport response' not in str(exc.value)


def test_access_success_does_not_accept_a_changed_pinned_object(monkeypatch):
    monkeypatch.setattr(hub,'get_hf_file_metadata',lambda *a,**kw:SimpleNamespace(commit_hash='a'*40,size=43,etag='b'*64))
    with pytest.raises(ValueError,match='frozen size'): check_gated_access(SPEC,'test-secret')


def test_failed_preflight_invalidates_old_success_even_after_partial_write(tmp_path,monkeypatch):
    from colab.final import preflight
    config={'work_root':str(tmp_path)};write(tmp_path/'PREFLIGHT.json',{'passed':True})
    monkeypatch.setattr(preflight,'validate_config',lambda c:c)
    def failure(*a):
        write(tmp_path/'PREFLIGHT.json',{'passed':True})
        raise RuntimeError('Drive upload failed after HF_TOKEN check test-secret')
    monkeypatch.setattr(preflight,'_run',failure)
    with pytest.raises(RuntimeError): preflight.run(config,'test-secret','a'*64)
    receipt=read(tmp_path/'PREFLIGHT.json')
    assert not receipt['passed'] and receipt['status']=='stopped'
    assert 'HF_TOKEN check' in receipt['reason'] and 'test-secret' not in receipt['reason']


def test_repeat_asset_preparation_reserves_only_missing_bytes(tmp_path,monkeypatch):
    from colab.final import assets
    from colab.final.common import sha
    payload=b'test';(tmp_path/'missing').write_bytes(payload);digest=sha(tmp_path/'missing')
    (tmp_path/'assets/sound_out').mkdir(parents=True);(tmp_path/'assets/sound_out/model.safetensors').write_bytes(payload)
    (tmp_path/'assets/motion.pt').write_bytes(payload);(tmp_path/'checkpoints').mkdir();(tmp_path/'checkpoints/existing').write_bytes(b'0123456789')
    real=read('colab/final/components.lock.json')
    lock={'sound_out':{'repository':'owner/model','revision':'a'*40,'files':[{'path':'model.safetensors','bytes':len(payload),'sha256':digest}]},
          'motion':{**real['motion'],'checkpoint':{'url':'https://example.invalid/motion.pt','bytes':len(payload),'sha256':digest}}}
    monkeypatch.setattr(assets,'read',lambda path:lock)
    class Api:
        def __init__(self,token): pass
        def whoami(self): return {'name':'TassP0lar','auth':{'accessToken':{'role':'write'}}}
    monkeypatch.setattr(hub,'HfApi',Api);monkeypatch.setattr(assets,'check_gated_access',lambda *args:None)
    guards=[];fetches=[]
    monkeypatch.setattr(assets,'disk_guard',lambda root,pending=0,reserve=0:guards.append((pending,reserve)))
    def fetch(url,path,entry,token=None,reserve=0,verified_existing=False):
        fetches.append((path,reserve,verified_existing));return path
    monkeypatch.setattr(assets,'fetch',fetch)
    paths=assets.prepare_assets(tmp_path,'test-secret',{'initial_checkpoint_reservation_bytes':10},download_weights=True)
    assert guards==[(0,10)] and all(valid for _,_,valid in fetches)
    assert paths['motion']==tmp_path/'assets/motion.pt'


def test_cpu_preflight_allows_no_drive_and_reports_actual_visibility(tmp_path,monkeypatch,capsys):
    from colab.final import preflight
    monkeypatch.setattr(preflight,'prepare_assets',lambda *args,**kwargs:{})
    monkeypatch.setattr(preflight,'prepare_data',lambda *args,**kwargs:{'split_membership_sha256':'s'*64})
    monkeypatch.setattr(preflight,'context_audit',lambda *args,**kwargs:{})
    monkeypatch.setattr('colab.final.engine.binding',lambda config,source:'bound')
    class Store:
        def __init__(self,*args):
            self.service=None;self.hf=SimpleNamespace(ensure_repository=lambda:'h'*40)
        def small_file(self,path,kind): return {'backend':'huggingface_artifact'}
    monkeypatch.setattr('colab.final.storage.ColdStore',Store)
    report=preflight._run({'work_root':str(tmp_path),'checkpoint_repo':'TassP0lar/test','checkpoint_repo_private':False},'test-secret','a'*64)
    assert report['drive_quota'] is None and report['passed']
    assert '"private": false' in capsys.readouterr().out
