"""Small standard-library contracts shared by the formal runner and notebook."""
from __future__ import annotations
import hashlib, json, os, re, shutil, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
SCHEMA=1

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
def local(path):
    p=Path(path).resolve()
    if not p.is_relative_to('/content') or any(x in {'drive','gdrive','MyDrive'} for x in p.parts):
        raise ValueError('Active work must be local /content, outside mounted Drive')
    return p

def disk_guard(root,pending=0,reserve=0):
    free=shutil.disk_usage(root).free
    if free-pending-reserve < 44_000_000_000:
        raise RuntimeError(f'Local free floor would be violated: free={free}, pending={pending}, reserved={reserve}')
    return free

def validate_config(c):
    if c.get('schema_version')!=SCHEMA or not re.fullmatch(r'[a-zA-Z0-9_-]{3,64}',c.get('campaign_id','')): raise ValueError('Invalid campaign identity')
    local(c['work_root'])
    if not re.fullmatch(r'TassP0lar/[A-Za-z0-9_.-]+',c['model_repo']): raise ValueError('Use an explicit model repository in the intended account')
    public_ok=c.get('allow_public_artifacts') is True
    if type(c['model_repo_private']) is not bool or type(c['checkpoint_repo_private']) is not bool:
        raise ValueError('Repository visibility must be an explicit boolean')
    if not c['model_repo_private'] and not public_ok:
        raise ValueError('Public model upload requires allow_public_artifacts=true and sanitized exports')
    if c.get('checkpoint_backend')!='huggingface' or type(c['checkpoint_repo_private']) is not bool:
        raise ValueError('Formal checkpoints require the reviewed Hugging Face storage policy')
    if not c['checkpoint_repo_private'] and not public_ok:
        raise ValueError('Public checkpoints require allow_public_artifacts=true and sanitized snapshots')
    if not 0 < c['instructor_budget_usd'] <= 50 or not 0 < c['max_compute_units'] <= 1000: raise ValueError('Campaign exceeds the authorized budget')
    if c['cache_bytes_per_component']*10 > 24_000_000_000: raise ValueError('Global media cache exceeds its reservation')
    if c['checkpoint_slots']!=2 or c['drive_cap_bytes']>2_000_000_000_000: raise ValueError('Storage contract changed')
    if c['gradient_accumulation']<1 or c['max_sequence']>4096: raise ValueError('Unqualified batch/sequence policy')
    microbatch=c.get('microbatch_size',1)
    if microbatch not in (1,2,4,8) or c['gradient_accumulation']%microbatch:
        raise ValueError('Microbatch must divide the effective eight-record optimizer batch')
    if not 0<=c.get('dataloader_workers',0)<=16 or not 1<=c.get('dataloader_prefetch',1)<=8:
        raise ValueError('Invalid DataLoader worker/prefetch policy')
    if not 0<c.get('max_wall_hours',110)<=110 or not 0<=c.get('direct_wall_hours',0)<=24:
        raise ValueError('Campaign wall budget is 110 hours; each Colab session is at most 24 hours')
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
        if self.c.get('enforce_compute_budget') is False: return None
        if s['rate'] is None or s['observed_at'] is None: raise RuntimeError('A fresh Colab usage observation is required')
        elapsed=(time.time()-s['observed_at'])/3600
        if elapsed>self.c['balance_refresh_hours']: raise RuntimeError('Refresh the Colab CU balance/rate observation before continuing')
        estimate=needed_seconds*s['rate']/3600
        available=min(self.c['max_compute_units']-s['used_cu'],s['observed_balance']-elapsed*s['rate'])
        if available < estimate+self.c['compute_reserve_cu']: raise RuntimeError('Compute reserve reached; save resumably, do not declare completion')
        return available
