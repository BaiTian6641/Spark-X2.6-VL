"""Install a hash-pinned candidate runtime without changing the Colab kernel."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile

from .common import now, sha256, write_json


def child_environment(project, work):
    project, work = Path(project).resolve(), Path(work).resolve()
    environment = os.environ.copy()
    # The public model preparation/qualification process does not need secrets.
    for key in ['PYTHONPATH', 'PYTHONHOME', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN',
                'OPENCODE_API_KEY', 'WANDB_API_KEY', 'VIRTUAL_ENV']:
        environment.pop(key, None)
    environment.update({
        'HF_HOME': str(work / 'hf'), 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
        'HF_HUB_DISABLE_XET': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
        'DO_NOT_TRACK': '1', 'TOKENIZERS_PARALLELISM': 'false',
        'UNSLOTH_STUDIO_HOME': str(work / 'studio'),
        'UNSLOTH_COMPILE_LOCATION': str(work / 'unsloth-compiled'),
        'TORCHINDUCTOR_CACHE_DIR': str(work / 'torchinductor'),
        'TRITON_CACHE_DIR': str(work / 'triton'),
        'XDG_CACHE_HOME': str(work / 'cache'), 'UV_CACHE_DIR': str(work / 'uv-cache'),
        'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
        'PYTHONUNBUFFERED': '1', 'UNSLOTH_SKIP_AUTOSTART': '1',
    })
    return environment


def run_logged(command, log_path, cwd, environment, timeout=2700):
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        result = subprocess.run([str(part) for part in command], cwd=cwd, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'Command failed with code {result.returncode}; inspect {path}')


def bootstrap(project, work):
    project, work = Path(project).resolve(), Path(work).resolve()
    if platform.system() != 'Linux' or platform.machine() != 'x86_64' or sys.version_info[:2] != (3, 13):
        raise RuntimeError('This candidate requires Linux x86_64 and CPython 3.13; stop on runtime drift')
    if project.is_relative_to('/content/drive') or work.is_relative_to('/content/drive'):
        raise ValueError('The active environment must use Colab local disk')
    spec = json.loads((project / 'colab/stage1/environment_spec.json').read_text())
    lock = project / 'colab/stage1/candidate-cu130-py313.lock'
    lock_hash = sha256(lock)
    work.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(work).free < 110 * 1024**3:
        raise RuntimeError('At least 110 GiB free local disk is required before installation and qualification')
    environment = child_environment(project, work)
    tools = work / 'tools'
    tools.mkdir(exist_ok=True)
    uv = tools / 'uv'
    uv_record = tools / 'uv-source.json'
    if not uv.exists():
        with urllib.request.urlopen(spec['uv']['url'], timeout=90) as response:
            payload = response.read(spec['uv']['bytes'] + 1)
        if len(payload) != spec['uv']['bytes'] or hashlib.sha256(payload).hexdigest() != spec['uv']['sha256']:
            raise ValueError('uv bootstrap wheel failed integrity verification')
        with zipfile.ZipFile(io.BytesIO(payload)) as wheel:
            binaries = [name for name in wheel.namelist() if name.endswith('.data/scripts/uv')]
            if len(binaries) != 1:
                raise ValueError('Expected exactly one uv binary in the pinned wheel')
            with uv.open('xb') as stream:
                stream.write(wheel.read(binaries[0]))
        uv.chmod(0o700)
        write_json(uv_record, {'wheel': spec['uv'], 'binary_sha256': sha256(uv)})
    if not uv_record.exists() or sha256(uv) != json.loads(uv_record.read_text())['binary_sha256']:
        raise ValueError('uv bootstrap binary identity changed')
    python = work / 'venv/bin/python'
    marker = work / 'environment-install.json'
    if marker.exists() and json.loads(marker.read_text()).get('requirements_lock_sha256') != lock_hash:
        raise ValueError('Use a new environment directory for a changed dependency lock')
    if not python.exists():
        run_logged([uv, 'venv', '--python', sys.executable, work / 'venv'],
                   work / 'logs/create-venv.log', project, environment)
    run_logged([uv, 'pip', 'sync', '--python', python, '--require-hashes', '--only-binary', ':all:', lock],
               work / 'logs/install.log', project, environment)
    run_logged([uv, 'pip', 'check', '--python', python], work / 'logs/pip-check.log', project, environment)
    run_logged([uv, 'pip', 'freeze', '--python', python], work / 'resolved-installed.txt', project, environment)
    receipt = {'created_utc': now(), 'scope': 'dependency_installation_only', 'python': str(python),
               'python_version': platform.python_version(), 'requirements_lock_sha256': lock_hash,
               'installed_freeze_sha256': sha256(work / 'resolved-installed.txt'),
               'dependency_check_passed': True, 'colab_kernel_packages_changed': False,
               'cuda_model_qualified': False, 'full_campaign_ready': False}
    write_json(marker, receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(bootstrap(args.project, args.work), indent=2))
