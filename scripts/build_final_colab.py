"""Build reviewed source, then a small launcher pinned to its GitHub commit.

First use --runtime-only and commit the runtime ZIP and sources to GitHub.
Then use --runtime-revision FULL_COMMIT to build the lightweight notebook.
"""
import argparse,ast,hashlib,io,json,re,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FINAL=ROOT/'colab/final'
def digest(x): return hashlib.sha256(x).hexdigest()
def build(runtime_revision=None, runtime_only=False):
    selected=[p for p in FINAL.rglob('*') if p.is_file() and ('research' not in p.relative_to(FINAL).parts) and '__pycache__' not in p.parts and p.suffix not in ('.log','.partial')]
    extras=['reports/spark_multimodal/COLAB_CAMPAIGN_PLAN.md','reports/spark_multimodal/REFINEMENT_MULTIMODAL_REVIEW_2026-10-27.md','colab/__init__.py','colab/preflight.py','colab/stage1/__init__.py','colab/stage1/common.py','colab/stage1/drive_checkpoint.py','colab/stage1/requirements.in','colab/formal/__init__.py','colab/formal/rolling_cache.py','colab/formal/shard_dataset.py','colab/formal/local_storage.py','colab/formal/teacher_readiness.py','colab/formal/HF_FULL_DATASET_HANDOFF.json']
    selected+= [ROOT/p for p in extras]
    selected=sorted(set(selected)); files=[]; output=io.BytesIO()
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in selected:
            payload=p.read_bytes(); name=p.relative_to(ROOT).as_posix()
            if p.is_symlink(): raise ValueError('No source symlinks')
            if len(payload)>8*1024**2: raise ValueError('Unexpected large source payload')
            info=zipfile.ZipInfo(name,date_time=(2026,10,27,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            z.writestr(info,payload);files.append({'path':name,'bytes':len(payload),'sha256':digest(payload)})
        manifest={'schema_version':1,'scope':'formal native multimodal runtime source only','files':files};payload=json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()
        info=zipfile.ZipInfo('SOURCE_MANIFEST.json',date_time=(2026,10,27,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,payload)
    archive=output.getvalue(); source_sha=digest(archive); out=ROOT/'output/colab-formal-final';out.mkdir(parents=True,exist_ok=True)
    (out/'spark-formal-runtime.zip').write_bytes(archive)
    (out/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if runtime_only:
        result={'source_archive_sha256':source_sha,'bytes':len(archive),'source_files':len(files)}
        print(json.dumps(result,indent=2)); return result
    if not runtime_revision or not re.fullmatch('[0-9a-f]{40}',runtime_revision):
        raise ValueError('Use --runtime-only first, then supply the full reviewed GitHub commit with --runtime-revision')
    source_url=f'https://raw.githubusercontent.com/BaiTian6641/Spark-X2.6-VL/{runtime_revision}/releases/spark-formal-runtime.zip'
    cells=[]
    def md(s): cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(keepends=True),'id':f'cell-{len(cells):03d}'})
    def code(s):
        ast.parse(s)
        cells.append({'cell_type':'code','metadata':{},'source':s.splitlines(keepends=True),'outputs':[],'execution_count':None,'id':f'cell-{len(cells):03d}'})
    md('''# Spark X2.6 native multimodal formal full fine-tune

This is the formal source-bound campaign notebook, not the earlier image-only qualification. It implements SigLIP2 So400m NaFlex with tiles and spatial downshuffle, V-JEPA2.1 motion, Whisper/CLAP audio input, Qwen native speech output, Stable Audio native sound/music output, full-parameter stages, bounded instructor-guided replay comparisons, resumable Drive checkpoints and private HF model publication.

**Start on CPU.** Keep model/data random I/O on VM-local disk. Drive stores closed artifacts only. The released dataset has304,861 original instructions; complete bounded music-score views produce315,359 supervised views before source-group splitting. The local media cache is23GB; the free floor is44GB; two complete checkpoint slots are reserved. A full Git LFS dataset clone is prohibited by this disk design.

**Evidence boundary:** the source/CPU component tests do not certify the complete CUDA model. This notebook must earn its whole-model and fresh-VM recovery gates. A single notebook does not guarantee one uninterrupted VM session or successful multimodal quality within1000CU. It never silently reduces the curriculum to fit.

Default final destination is the **private model repository** `TassP0lar/SparkX2.6-MM`; the public dataset namespace is separate. No instructor call or training happens merely by opening this notebook. Read the downloaded `RUN.md` after bootstrap.

The runtime is downloaded from a pinned commit of [BaiTian6641/Spark-X2.6-VL](https://github.com/BaiTian6641/Spark-X2.6-VL). Its SHA256 and exact source inventory are checked before execution. The notebook contains no embedded source archive. Both CPU and CUDA setup recover an earlier Python-without-pip environment without relying on system `ensurepip`. Existing local data and Drive checkpoints stay in their original locations.''')
    code(f'''# Verified source bootstrap: standard library only, no credentials.
import hashlib, io, json, os, pathlib, sys, urllib.request, zipfile
SOURCE_SHA256 = {source_sha!r}
SOURCE_COMMIT = {runtime_revision!r}
SOURCE_URL = {source_url!r}
SOURCE_BYTES = {len(archive)!r}
# A distinct source directory allows repair alongside the previous notebook.
SOURCE_ROOT = pathlib.Path('/content/spark-formal-source') / SOURCE_SHA256[:16]
request = urllib.request.Request(SOURCE_URL, headers={{'User-Agent':'Spark-Colab-launcher/2'}})
with urllib.request.urlopen(request, timeout=60) as response:
    _source_bytes = response.read(SOURCE_BYTES + 1)
if len(_source_bytes) != SOURCE_BYTES or hashlib.sha256(_source_bytes).hexdigest() != SOURCE_SHA256:
    raise RuntimeError('GitHub runtime size or SHA256 mismatch; no runtime source was executed')
with zipfile.ZipFile(io.BytesIO(_source_bytes)) as z:
    names = z.namelist()
    assert len(names) == len(set(names))
    manifest = json.loads(z.read('SOURCE_MANIFEST.json'))
    assert set(names) == {{f['path'] for f in manifest['files']}} | {{'SOURCE_MANIFEST.json'}}
    assert sum(f['bytes'] for f in manifest['files']) < 16*1024**2
    for f in manifest['files']:
        relative = pathlib.PurePosixPath(f['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        target = SOURCE_ROOT / relative
        assert target.resolve().is_relative_to(SOURCE_ROOT.resolve()) and not target.is_symlink()
        payload = z.read(f['path'])
        assert len(payload) == f['bytes'] and hashlib.sha256(payload).hexdigest() == f['sha256']
        if target.exists(): assert target.read_bytes() == payload, 'Changed source: use a new reviewed release'
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    (SOURCE_ROOT/'SOURCE_MANIFEST.json').write_bytes(z.read('SOURCE_MANIFEST.json'))
del _source_bytes
print('Reviewed source:', SOURCE_SHA256)
print('GitHub commit:', SOURCE_COMMIT)
print((SOURCE_ROOT/'colab/final/RUN.md').read_text())
''')
    md('''## Operator configuration

Keep Secrets in Colab's key panel, not in these cells. Enable access for `HF_TOKEN` and `OPENCODE_API_KEY`. The HF account must have access to [Stable Audio Open1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) before the CPU preflight passes. This notebook never accepts a model license on your behalf.

Fill the instructor eligibility receipt from provider/account evidence before formal training. OpenCode Go API readiness and the key's existence do not establish permission or quota for dataset review. The implementation stops at that gate instead of guessing. Keep its $50 cumulative ceiling.

For GPU work, replace the two `None` usage fields with the **actual current** Colab balance and CU/hour display. Re-run this cell and refresh the observation at least every12h. Changing architecture, splits or stage epochs creates a different checkpoint policy binding.''')
    code('''CONFIG = json.loads((SOURCE_ROOT/'colab/final/campaign.json').read_text())
# Stable campaign name/work root across VM replacements.
CONFIG['model_repo'] = 'TassP0lar/SparkX2.6-MM'
CONFIG['model_repo_private'] = True
CONFIG['drive_project_folder'] = '1ELmWWHxpCb3IGM2LZDQV-u7NiLxK12T_'
CONFIG['observed_compute_balance'] = None  # Enter currently displayed balance before GPU commands.
CONFIG['observed_cu_per_hour'] = None     # Enter currently displayed active rate (do not assume a quoted price).
# Example SHAPE only: replace with actual provider/account evidence. Never invent an approval.
# CONFIG['instructor_route_eligibility_receipt'] = {
#   'provider':'OpenCode Go', 'model':'mimo-v2.6-pro',
#   'endpoint':'https://opencode.ai/inference/go/openai/v1/chat/completions',
#   'permits_dataset_review':True, 'evidence_url':'PROVIDER_EVIDENCE', 'checked_at':'ACTUAL_DATE'
# }
WORK = pathlib.Path(CONFIG['work_root'])
assert WORK.resolve().is_relative_to('/content') and 'drive' not in WORK.parts
WORK.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = WORK/'operator-config.json'
CONFIG_PATH.write_text(json.dumps(CONFIG, indent=2))
print({k:CONFIG[k] for k in ('campaign_id','model_repo','model_repo_private','max_compute_units','instructor_budget_usd')})
''')
    md('''## Google access and in-memory Secrets

Authorize this reviewed notebook for the intended Drive project. Authentication uses Colab's normal credential handling; credentials are never copied into a checkpoint or model upload. Optional Drive mounting below is for browsing cold results only.''')
    code('''from google.colab import auth, userdata
# The returned secrets remain in kernel memory. Do not print these variables.
auth.authenticate_user()
HF_SECRET = userdata.get('HF_TOKEN')
try:
    INSTRUCTOR_SECRET = userdata.get('OPENCODE_API_KEY')
except Exception:
    INSTRUCTOR_SECRET = None
assert HF_SECRET, 'Enable notebook access to HF_TOKEN in Colab Secrets'
print('Google authentication finished; secret presence checked without displaying values.')
''')
    code('''# Optional cold-storage browsing. No training path uses this mount.
MOUNT_DRIVE_FOR_BROWSING = False
if MOUNT_DRIVE_FOR_BROWSING:
    from google.colab import drive
    drive.mount('/content/drive')
''')
    md('''## Install isolated CPU tooling, then preflight

This cell installs into a separate environment. It does not replace the notebook kernel's PyTorch. A wheel-and-binary-hash-verified `uv` installer creates pipless virtual environments and synchronizes complete package locks, including a pinned `pip`. System `ensurepip` is not needed. An earlier isolated environment with Python but no pip is repaired in place. Setup records readiness only after the complete installation and dependency check pass. On a new VM repeat this setup and CPU preflight before GPU recovery.

Campaign stdout and stderr stream into this cell, with secret values and signed URL queries removed. Each invocation also saves a bounded sanitized log under `/content/spark-formal/logs`. A failure reports its exit code and actual reason. If the reason is `GatedRepo`, review the Stable Audio access form using the HF account that owns your token; once access is granted, rerun this cell without resetting the VM. A fine-grained token must also permit reading that gated model.''')
    code('''import importlib.util, subprocess, shutil
assert sys.version_info[:2] == (3,13), 'This reviewed lock targets CPython3.13; qualify another release if Colab changes Python'
# Load only this release's standard-library helper, even after an older setup.
_spec = importlib.util.spec_from_file_location('spark_environment_bootstrap', SOURCE_ROOT/'colab/final/bootstrap.py')
_bootstrap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bootstrap)
CPU_PYTHON = _bootstrap.install_environment(WORK, SOURCE_ROOT, 'cpu', base_python=sys.executable)
_spec = importlib.util.spec_from_file_location('spark_campaign_launcher', SOURCE_ROOT/'colab/final/launcher.py')
_launcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_launcher)

def run_campaign(mode, gpu=False, profile_receipt=None, resume_receipt=None):
    python = str(WORK/('env-cuda' if gpu else 'env-cpu')/'bin/python')
    args = [python,'-m','colab.final.run',mode,'--config',str(CONFIG_PATH),'--source-sha',SOURCE_SHA256]
    for option, value in (('--profile-receipt',profile_receipt),('--resume-receipt',resume_receipt)):
        if value is not None:
            receipt_path = WORK/(option[2:]+'.json')
            receipt_path.write_text(json.dumps(value))
            args += [option,str(receipt_path)]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(SOURCE_ROOT)
    log_path = WORK/'logs'/(mode+'-'+__import__('uuid').uuid4().hex+'.log')
    return _launcher.run_process(args, cwd=SOURCE_ROOT, env=env,
        secrets={'HF_TOKEN':HF_SECRET,'OPENCODE_API_KEY':INSTRUCTOR_SECRET}, log_path=log_path)

run_campaign('cpu')
''')
    md('''## GPU environment and qualification

Choose the G4 Blackwell runtime only after CPU preflight. Record its current usage balance/rate in configuration. This release targets SM120, at least90GB GPU memory and60GB RAM. It fails on a different assignment instead of silently switching to LoRA or low-bit training.

The actual Studio backend ships inside the pinned Unsloth wheel. Its full graphical UI extra conflicts with the reviewed Transformers version, so this campaign uses Studio's real headless training backend. Torch SDPA, BF16 compute, gradient checkpointing and AdamW8bit are active; custom Spark speed is measured, not assumed. This cell may take time and incurs runtime charges while a GPU is attached.''')
    code('''INSTALL_CUDA_ENVIRONMENT = False  # Set True after switching to G4 and entering current usage observations.
if INSTALL_CUDA_ENVIRONMENT:
    assert CONFIG['observed_compute_balance'] and CONFIG['observed_cu_per_hour'], 'Update and rerun operator configuration'
    _preflight = json.loads((WORK/'PREFLIGHT.json').read_text())
    assert _preflight.get('passed') and _preflight.get('source_archive_sha256') == SOURCE_SHA256, 'CPU preflight must pass for this source on this VM first'
    # Optional native SoX executable used by the upstream package; decoding in
    # this pipeline uses SoundFile/PyAV and preserves the released waveforms.
    if not shutil.which('sox'):
        subprocess.run(['apt-get','update','-qq'], check=True)
        subprocess.run(['apt-get','install','-y','-qq','sox','libsndfile1'], check=True)
    CUDA_PYTHON = _bootstrap.install_environment(WORK, SOURCE_ROOT, 'cuda', base_python=sys.executable)
    run_campaign('doctor', gpu=True)
''')
    code('''RUN_QUALIFICATION = False  # Run once on the first GPU VM; do not repeat after it passes.
if RUN_QUALIFICATION:
    assert INSTRUCTOR_SECRET, 'Enable OPENCODE_API_KEY before the formal campaign'
    run_campaign('qualify', gpu=True)
    print('SAVE THE EMITTED QUALIFICATION PROFILE RECEIPT before replacing this VM.')
''')
    md('''## Fresh-VM recovery, formal training, and later resume

After qualification, save its emitted **profile receipt**, reconnect a fresh VM, run source/bootstrap/config/auth/CPU setup again and install the CUDA environment. Enter that profile receipt below. `train` first restores the fully populated qualification checkpoint, reproduces two native continuation updates, and verifies recovery across boot IDs. It then restores the original initial snapshot and begins formal stages. This is the deliberate recovery qualification before a long campaign.

If a later session ends, use `resume` with the last exact **checkpoint receipt** (not a profile/evaluation/unfinished upload). Optimizer-committed cursor recovery prevents skipped rows. Instructor intents are reconciled from Drive; an ambiguous POST requires operator/provider reconciliation. The $50 ceiling includes accumulated provider cash/list/allowance exposure.

The full-stage projection must fit the current authorized CU balance plus reserve. The notebook will not promise a one-session result or remove modalities/epochs to make the estimate fit. Watch `training`, `drive_committed`, `qualification_complete` and `final_model_uploaded` events. A stop event is not completion.''')
    code('''PROFILE_RECEIPT = None  # Paste the nonsecret committed qualification-profile receipt dict.
RESUME_RECEIPT = None   # For a later interruption, paste an exact committed checkpoint receipt dict.
RUN_MODE = 'inspect'    # 'inspect', 'train', 'resume', or 'recover' (recovery only).
if RUN_MODE == 'inspect':
    print('No GPU training launched. Select a mode explicitly when ready.')
elif RUN_MODE in ('train','resume','recover'):
    assert PROFILE_RECEIPT, 'The qualification-profile receipt is required'
    assert CONFIG['observed_compute_balance'] and CONFIG['observed_cu_per_hour'], 'Refresh current usage observations'
    assert CONFIG['instructor_route_eligibility_receipt'], 'Resolve the provider/account dataset-review eligibility before formal training'
    assert INSTRUCTOR_SECRET, 'Enable OPENCODE_API_KEY'
    run_campaign(RUN_MODE,gpu=True,profile_receipt=PROFILE_RECEIPT,resume_receipt=RESUME_RECEIPT)
else:
    raise ValueError('Unknown explicit run mode')
''')
    md('''## Receipts and results

A finished campaign uploads the selected **trained private model** using `HF_TOKEN`, verifies the remote model LFS checksum and writes `MODEL_UPLOAD_RECEIPT.json` to local disk and cold Drive. The export is also reloaded from only its own saved configs/code/weights before uploading. It excludes optimizer state, credentials, raw training data and private instructor ledgers. Complete resumable checkpoints remain in Drive.

The card reports actual internal results, the selected/rejected RSI candidates and pending human audio review. A completed curriculum is not proof of broad vision/video/music capability. Listen to the final native samples and their zero-bridge controls; examine input ablations, temporal reversal, bilingual retention and independent-group counts. Stronger downstream acceptance still needs the documented domain/public benchmarks and qualified listeners.''')
    code('''# Read only sanitized status/receipts. Never display credentials or upload-state files.
for name in ('PREFLIGHT.json','ENVIRONMENT.json','FULL_RECOVERY_PROOF.json','MODEL_UPLOAD_RECEIPT.json'):
    path=WORK/name
    if path.exists():
        value=json.loads(path.read_text())
        if name=='PREFLIGHT.json': value={k:v for k,v in value.items() if k not in ('context','drive_quota')}
        print(name, json.dumps(value,indent=2))
# To find cold commits after VM loss, with configuration/auth/CPU setup completed:
# run_campaign('discover')
''')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'Spark_Formal_Full_Finetune.ipynb','provenance':[]},'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python'},'spark_formal':{'source_sha256':source_sha,'source_commit':runtime_revision,'source_url':source_url,'training_executed':False,'teacher_requests_made':0,'dataset_revision':json.loads((FINAL/'dataset.lock.json').read_text())['media_revision']}},'cells':cells}
    notebook=ROOT/'colab/Spark_Formal_Full_Finetune.ipynb'; notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n');shipped=out/notebook.name;shipped.write_bytes(notebook.read_bytes())
    if notebook.stat().st_size > 64*1024: raise ValueError('Launcher unexpectedly exceeds 64KiB')
    release={'schema_version':2,'notebook':{'path':str(notebook.relative_to(ROOT)),'bytes':notebook.stat().st_size,'sha256':digest(notebook.read_bytes())},'source_archive':{'path':'releases/spark-formal-runtime.zip','bytes':len(archive),'sha256':source_sha,'github_commit':runtime_revision,'url':source_url},'source_files':len(files),'training_executed':False,'instructor_requests_made':0}
    (out/'RELEASE.json').write_text(json.dumps(release,indent=2)+'\n');print(json.dumps(release,indent=2))
    return release
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-only',action='store_true')
    parser.add_argument('--runtime-revision')
    args=parser.parse_args()
    build(args.runtime_revision,args.runtime_only)
