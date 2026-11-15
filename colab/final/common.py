"""Small standard-library contracts shared by the formal runner, workstation CLI, and notebook."""
from __future__ import annotations
import hashlib, json, os, platform, re, shutil, subprocess, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
SCHEMA=1

OPTIMIZER_PROFILES=(
    'bf16_adamw',
    'adamw8bit',
    'apollo_merged',
    'apollo_mini_merged',
    'apollo_coat',
    'apollo_coat_fp8_moments',
)

HARDWARE_PROFILES={
    'colab_g4_96gb':{
        'min_gpu_bytes':90_000_000_000,
        'min_ram_bytes':60_000_000_000,
        'capability':(12,0),
    },
    'rtx_pro_4000_blackwell_24gb':{
        'min_gpu_bytes':22_000_000_000,
        'min_ram_bytes':24_000_000_000,
        'capability':(12,0),
    },
}

_BOOT_ID_CACHE:str|None=None

def runtime_boot_id()->str:
    global _BOOT_ID_CACHE
    if _BOOT_ID_CACHE is not None: return _BOOT_ID_CACHE
    proc_boot=Path('/proc/sys/kernel/random/boot_id')
    if proc_boot.is_file():
        _BOOT_ID_CACHE=proc_boot.read_text().strip()
        return _BOOT_ID_CACHE
    try:
        out=subprocess.check_output(['sysctl','-n','kern.bootsuuid'],stderr=subprocess.DEVNULL,text=True).strip()
        if out:
            _BOOT_ID_CACHE=out
            return _BOOT_ID_CACHE
    except Exception:
        pass
    _BOOT_ID_CACHE='host-'+hashlib.sha256(f'{platform.node()}:{platform.system()}:{platform.release()}'.encode()).hexdigest()[:32]
    return _BOOT_ID_CACHE

def canonical(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def identity(x): return hashlib.sha256(canonical(x)).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()
def write(path,x):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial'); tmp.write_bytes(canonical(x)+b'\n'); os.replace(tmp,path)
def read(path): return json.loads(Path(path).read_text())
def emit(event,**kw): print(json.dumps({'event':event,**kw},ensure_ascii=False),flush=True)

def local(path,execution_target='colab'):
    p=Path(path).resolve()
    if any(x in {'drive','gdrive','MyDrive'} for x in p.parts):
        raise ValueError('Active work must stay outside mounted Drive')
    if execution_target=='colab' and not p.is_relative_to('/content'):
        raise ValueError('Active work must be local /content, outside mounted Drive')
    if execution_target not in ('colab','local_workstation'):
        raise ValueError(f'Unknown execution_target {execution_target!r}')
    return p

def disk_guard(root,pending=0,reserve=0,min_free_bytes=44_000_000_000):
    free=shutil.disk_usage(root).free
    if free-pending-reserve < min_free_bytes:
        raise RuntimeError(f'Local free floor would be violated: free={free}, pending={pending}, reserved={reserve}')
    return free

def validate_config(c):
    if c.get('schema_version')!=SCHEMA or not re.fullmatch(r'[a-zA-Z0-9_-]{3,64}',c.get('campaign_id','')): raise ValueError('Invalid campaign identity')
    target=c.get('execution_target','colab')
    local(c['work_root'],execution_target=target)
    if not re.fullmatch(r'TassP0lar/[A-Za-z0-9_.-]+',c['model_repo']): raise ValueError('Use an explicit model repository in the intended account')
    public_ok=c.get('allow_public_artifacts') is True
    if type(c['model_repo_private']) is not bool or type(c['checkpoint_repo_private']) is not bool:
        raise ValueError('Repository visibility must be an explicit boolean')
    if not c['model_repo_private'] and not public_ok:
        raise ValueError('Public model upload requires allow_public_artifacts=true and sanitized exports')
    allowed_backends=('huggingface','local') if target=='local_workstation' else ('huggingface',)
    if c.get('checkpoint_backend') not in allowed_backends or type(c['checkpoint_repo_private']) is not bool:
        raise ValueError('Formal checkpoints require the reviewed Hugging Face storage policy')
    if c.get('checkpoint_backend')=='huggingface' and not c['checkpoint_repo_private'] and not public_ok:
        raise ValueError('Public checkpoints require allow_public_artifacts=true and sanitized snapshots')
    if not 0 < c['instructor_budget_usd'] <= 50 or not 0 < c['max_compute_units'] <= 1000: raise ValueError('Campaign exceeds the authorized budget')
    if c['cache_bytes_per_component']*10 > 24_000_000_000: raise ValueError('Global media cache exceeds its reservation')
    if c['checkpoint_slots']!=2 or c['drive_cap_bytes']>2_000_000_000_000: raise ValueError('Storage contract changed')
    if c['gradient_accumulation']<1 or c['max_sequence']>8192: raise ValueError('Unqualified batch/sequence policy')
    microbatch=c.get('microbatch_size',1)
    if microbatch not in (1,2,4,8) or c['gradient_accumulation']%microbatch:
        raise ValueError('Microbatch must divide the effective eight-record optimizer batch')
    if not 0<=c.get('dataloader_workers',0)<=16 or not 1<=c.get('dataloader_prefetch',1)<=8:
        raise ValueError('Invalid DataLoader worker/prefetch policy')
    max_wall_cap=720 if target=='local_workstation' else 110
    if not 0<c.get('max_wall_hours',110)<=max_wall_cap or not 0<=c.get('direct_wall_hours',0)<=max(24,max_wall_cap if target=='local_workstation' else 24):
        raise ValueError('Campaign wall budget is 110 hours; each Colab session is at most 24 hours')
    if 'optimizer_profile' in c and c['optimizer_profile'] not in OPTIMIZER_PROFILES:
        raise ValueError(f"Unknown optimizer_profile {c['optimizer_profile']!r}")
    if 'hardware_profile' in c and c['hardware_profile'] not in HARDWARE_PROFILES:
        raise ValueError(f"Unknown hardware_profile {c['hardware_profile']!r}")
    if 'modalities' in c and c['modalities'] not in ('vision_text','omni'):
        raise ValueError(f"Unknown modalities {c['modalities']!r}")
    if 'spark_component' in c and c['spark_component'] not in ('spark','spark_fp8'):
        raise ValueError(f"Unknown spark_component {c['spark_component']!r}")
    if 'visual_cross_attn_blocks' in c and c['visual_cross_attn_blocks'] not in (0,3,6):
        raise ValueError('visual_cross_attn_blocks must be one of (0, 3, 6)')
    if 'apollo_rank' in c and (type(c['apollo_rank']) is not int or c['apollo_rank']<1):
        raise ValueError('apollo_rank must be a positive integer')
    if any(k in c for k in ('HF_TOKEN','OPENCODE_API_KEY','token','api_key')): raise ValueError('Secrets do not belong in configuration')
    return c

class ComputeBudget:
    """Conservative rate integration; operator balance observations are mandatory."""
    def __init__(self,c,state=None):
        self.c=c; self.state=state or {'used_cu':0.,'active_seconds':0.,'observed_at':None,'observed_balance':None,'rate':None}
        self.started=time.monotonic()
    def observe(self,balance,rate):
        if balance is None or rate is None or balance<=0 or rate<=0: raise ValueError('Enter the actual Colab CU balance and active hourly CU rate')
        self.tick(); self.state.update(observed_at=time.time(),observed_balance=float(balance),rate=float(rate))
    def tick(self):
        current=time.monotonic(); dt=current-self.started; self.started=current
        self.state['active_seconds']+=dt
        if self.state['rate'] is not None: self.state['used_cu']+=dt*self.state['rate']/3600
        return self.state
    def merge_checkpoint(self,saved):
        self.tick()
        for key in ('used_cu','active_seconds'):
            self.state[key]=max(self.state.get(key,0),saved.get(key,0))
        # Current operator observations remain current; old checkpoints may
        # supply accounting history but cannot refresh a stale observation.
        if self.state['observed_at'] is None:
            for key in ('observed_at','observed_balance','rate'): self.state[key]=saved.get(key)
        return self.state
    def check(self,needed_seconds=0):
        self.tick(); s=self.state
        if self.c.get('enforce_compute_budget') is False or self.c.get('execution_target')=='local_workstation': return None
        if s['rate'] is None or s['observed_at'] is None: raise RuntimeError('A fresh Colab usage observation is required')
        elapsed=(time.time()-s['observed_at'])/3600
        if elapsed>self.c['balance_refresh_hours']: raise RuntimeError('Refresh the Colab CU balance/rate observation before continuing')
        estimate=needed_seconds*s['rate']/3600
        available=min(self.c['max_compute_units']-s['used_cu'],s['observed_balance']-elapsed*s['rate'])
        if available < estimate+self.c['compute_reserve_cu']: raise RuntimeError('Compute reserve reached; save resumably, do not declare completion')
        return available
