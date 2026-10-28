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
