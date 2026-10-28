"""Real CPython 3.13 bootstrap/recovery tests; require public PyPI access.

Use a local-disk --basetemp with at least 46GB free. These install CPU tools only;
they do not download models, use a GPU, contact Drive or call the instructor.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
from colab.final.bootstrap import install_environment

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(sys.version_info[:2] != (3, 13), reason="Reviewed CPython 3.13 environment")


def test_actual_python_without_pip_recovers_and_repeated_setup_preserves_work(tmp_path):
    work = tmp_path / 'work'
    target = work / 'env-cpu'
    # This is the state left by a failed venv ensurepip: Python works, pip is absent.
    subprocess.run([sys.executable,'-m','venv','--without-pip',str(target)],check=True)
    python = target / 'bin/python'
    subprocess.run([str(python),'-I','-c',"import importlib.util; assert importlib.util.find_spec('pip') is None"],check=True)
    saved = work / 'checkpoints' / 'preserved.bin'
    saved.parent.mkdir(); saved.write_bytes(b'resumable-artifact-must-survive')
    result = install_environment(work, ROOT, 'cpu')
    assert result == str(python)
    receipt = json.loads((target/'.spark-environment-ready.json').read_text())
    assert receipt['ready'] and receipt['recovered_python_without_pip']
    subprocess.run([result,'-I','-c',"import importlib.metadata as m; import requests, googleapiclient, transformers; assert m.version('pip') == '26.2.2'; assert m.version('transformers') == '4.57.3'"],check=True)
    # Tokenizer-only CPU installs do not inherit Torch's Jinja dependency.
    # Exercise the actual Transformers method used by the corpus audit.
    subprocess.run([result,'-I','-c',"""
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
backend=Tokenizer(models.WordLevel({'[UNK]':0,'hello':1},unk_token='[UNK]'))
backend.pre_tokenizer=pre_tokenizers.Whitespace()
tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]')
tokenizer.chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}"
assert tokenizer.apply_chat_template([{'role':'user','content':'hello'}],tokenize=True)==[1]
"""],check=True)
    pip_source = target/'lib/python3.13/site-packages/pip/__init__.py'
    before = pip_source.stat().st_mtime_ns
    install_environment(work, ROOT, 'cpu')
    assert pip_source.stat().st_mtime_ns == before
    assert saved.read_bytes() == b'resumable-artifact-must-survive'
    # A failed retry cannot retain the previous successful readiness receipt.
    bad_source = tmp_path / 'bad-source' / 'colab/final'
    bad_source.mkdir(parents=True)
    (bad_source/'bootstrap.lock.json').write_bytes((ROOT/'colab/final/bootstrap.lock.json').read_bytes())
    (bad_source/'requirements-preflight.lock').write_text('pip==25.3 --hash=sha256:'+'0'*64+'\n')
    with pytest.raises(subprocess.CalledProcessError):
        install_environment(work, bad_source.parents[1], 'cpu')
    assert not (target/'.spark-environment-ready.json').exists()
    assert saved.read_bytes() == b'resumable-artifact-must-survive'
    install_environment(work, ROOT, 'cpu')
    assert json.loads((target/'.spark-environment-ready.json').read_text())['ready']


def test_fresh_pipless_environment_is_created_without_stdlib_seeding(tmp_path):
    source = tmp_path/'minimal-source'
    final = source/'colab/final'; final.mkdir(parents=True)
    (final/'bootstrap.lock.json').write_bytes((ROOT/'colab/final/bootstrap.lock.json').read_bytes())
    (final/'requirements-preflight.lock').write_text('pip==26.2.2 --hash=sha256:fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70\n')
    work = tmp_path/'fresh-work'
    result = install_environment(work, source, 'cpu')
    subprocess.run([result,'-I','-m','pip','check'],check=True)
    assert json.loads((work/'env-cpu/.spark-environment-ready.json').read_text())['ready']


def test_foreign_directory_and_symlink_environment_are_preserved(tmp_path):
    work = tmp_path/'work'; target=work/'env-cpu'; target.mkdir(parents=True)
    sentinel=target/'important.bin'; sentinel.write_bytes(b'keep')
    with pytest.raises(ValueError,match='Unrecognized existing'):
        install_environment(work, ROOT, 'cpu')
    assert sentinel.read_bytes()==b'keep'
    second=tmp_path/'another-work'; second.mkdir()
    (second/'env-cpu').symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError,match='escapes|symlink'):
        install_environment(second, ROOT, 'cpu')
    assert sentinel.read_bytes()==b'keep'
