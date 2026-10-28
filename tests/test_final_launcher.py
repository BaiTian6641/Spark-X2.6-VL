"""Real subprocess visibility, credential redaction and graceful interruption."""
import json
import os
import sys
from urllib.parse import quote

import pytest

from colab.final.launcher import run_process,sanitize


def test_child_stopped_reason_is_visible_and_preserved_without_credentials(tmp_path,capsys):
    token='hf_testSecretAbc123';instructor='special/secret+value'
    child="""import json,sys,os
s=json.loads(sys.stdin.readline())
assert all(k not in os.environ for k in s)
print('normal stdout',flush=True)
print('stderr '+s['HF_TOKEN']+' '+s['OPENCODE_API_KEY'],file=sys.stderr,flush=True)
print(json.dumps({'event':'stopped','error_type':'RuntimeError','reason':'GatedRepo at https://huggingface.co/stabilityai/stable-audio-open-1.0; HF_TOKEN needs model read access'}),flush=True)
sys.exit(2)
"""
    env=dict(os.environ,HF_TOKEN=token,OPENCODE_API_KEY=instructor)
    log=tmp_path/'failure.log'
    with pytest.raises(RuntimeError,match=r'exit 2.*GatedRepo') as exc:
        run_process([sys.executable,'-c',child],cwd=tmp_path,env=env,
                    secrets={'HF_TOKEN':token,'OPENCODE_API_KEY':instructor},log_path=log)
    evidence=capsys.readouterr().out+log.read_text()+str(exc.value)
    assert 'normal stdout' in evidence and 'stderr [REDACTED] [REDACTED]' in evidence
    assert token not in evidence and instructor not in evidence
    assert 'HF_TOKEN needs model read access' in evidence


def test_sanitizer_preserves_actions_but_removes_signed_urls_and_encoded_values():
    secret='secret/with+reserved'
    text=f'HF_TOKEN needs read access {secret} {quote(secret,safe="")} https://cdn.test/model?signature=private&token=abc Bearer unknownsecret'
    clean=sanitize(text,(secret,))
    assert clean=='HF_TOKEN needs read access [REDACTED] [REDACTED] https://cdn.test/model?[query redacted] Bearer [REDACTED]'


def test_signal_death_without_json_keeps_exit_code_and_stderr(tmp_path,capsys):
    child="import sys,os,signal; sys.stdin.readline(); print('last native diagnostic',file=sys.stderr,flush=True); os.kill(os.getpid(),signal.SIGKILL)"
    with pytest.raises(RuntimeError,match=r'exit -9.*last native diagnostic'):
        run_process([sys.executable,'-c',child],cwd=tmp_path,env=os.environ,secrets={},log_path=tmp_path/'killed.log')


def test_interrupt_drains_real_child_until_checkpoint_receipt(tmp_path):
    child="""import json,sys,signal,time
sys.stdin.readline()
def stop(*args):
 print(json.dumps({'event':'checkpoint_committed','test_child':True}),flush=True)
 sys.exit(130)
signal.signal(signal.SIGINT,stop)
print('ready for interrupt',flush=True)
time.sleep(20)
"""
    messages=[]
    def output(line,**kw):
        messages.append(line)
        if line=='ready for interrupt': raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt,match='Runner exited'):
        run_process([sys.executable,'-c',child],cwd=tmp_path,env=os.environ,secrets={},log_path=tmp_path/'interrupt.log',output=output)
    assert any('checkpoint_committed' in s for s in messages)
    assert any('exit code: 130' in s for s in messages)


def test_oversized_line_cannot_leak_a_secret_across_read_boundary(tmp_path,capsys):
    secret='uniqueSecretValue'
    child="import json,sys;s=json.loads(sys.stdin.readline());print('x'*65530+s['HF_TOKEN'],flush=True);print('after oversized line',flush=True)"
    run_process([sys.executable,'-c',child],cwd=tmp_path,env=os.environ,secrets={'HF_TOKEN':secret},log_path=tmp_path/'long.log')
    evidence=capsys.readouterr().out+(tmp_path/'long.log').read_text()
    assert 'Oversized child output line omitted' in evidence and 'after oversized line' in evidence
    assert 'unique' not in evidence and 'SecretValue' not in evidence


def test_flashattention_tuple_output_backpropagates_through_output_tensor():
    import torch
    from colab.final.run import _check_flash_attention_backward
    torch.manual_seed(0)
    q=torch.randn(1,2,8,8,dtype=torch.float32,requires_grad=True)
    result=(torch.nn.functional.scaled_dot_product_attention(q,q,q),None)
    _check_flash_attention_backward(torch,result,q)
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum()>0
