"""HF full checkpoints and small immutable Drive artifacts; local active I/O."""
from __future__ import annotations
import json, os, shutil, time
from pathlib import Path
from .common import canonical,identity,read,write,sha,disk_guard,emit
from colab.stage1.common import safe_relative
from colab.stage1.drive_checkpoint import upload_file,project_usage
from colab.preflight import download_verified,drive_metadata,validate_drive_id

REQUIRED={'model.safetensors','optimizer.pt','scheduler.pt','rng.pt','STATE.json','config.json','BINDING.json','PARAMETERS.json','CLOSED.json'}

def manifest(directory):
    directory=Path(directory); files=[]
    for p in sorted(directory.rglob('*')):
        if p.is_symlink(): raise ValueError('Artifact symlinks are forbidden')
        if not p.is_file(): continue
        name=p.relative_to(directory).as_posix(); safe_relative(name)
        if any(v in name.lower() for v in ('.env','credential','secret','private-upload','token.txt')): raise ValueError('Secret path in artifact')
        files.append({'path':name,'bytes':p.stat().st_size,'sha256':sha(p)})
    return {'files':files,'total_bytes':sum(f['bytes'] for f in files)}

def drive_service():
    import google.auth,google_auth_httplib2
    from googleapiclient.discovery import build
    from googleapiclient.http import build_http
    credentials,_=google.auth.default(scopes=['https://www.googleapis.com/auth/drive'])
    return build('drive','v3',http=google_auth_httplib2.AuthorizedHttp(credentials,http=build_http()),cache_discovery=False)

class ColdStore:
    def __init__(self,root,config,binding,token=None):
        self.root=Path(root); self.c=config; self.binding=binding; self.service=drive_service(); self.parent=validate_drive_id(config['drive_project_folder'])
        self.private=self.root/'private-upload-state'; self.private.mkdir(exist_ok=True,mode=0o700)
        item=drive_metadata(self.service,self.parent)
        if item.get('mimeType')!='application/vnd.google-apps.folder' or item.get('trashed'): raise ValueError('Invalid Drive project folder')
        self.hf=None
        if config.get('checkpoint_backend')=='huggingface':
            from .hf_checkpoints import HFCheckpointStore
            self.hf=HFCheckpointStore(root,config,binding,token)
    def commit(self,directory,kind='checkpoint'):
        directory=Path(directory).resolve()
        if not directory.is_relative_to(self.root.resolve()): raise ValueError('Only campaign-owned local artifacts may be uploaded')
        if kind=='checkpoint' and self.hf is not None:
            from .hf_checkpoints import receipt_path
            receipt=self.hf.commit(directory)
            # Back up the small exact-revision receipt before local reclamation.
            index=self.root/'checkpoint-receipts'/('hf-'+receipt['revision']+'.json')
            write(index,receipt); cold=self.small_file(index,'hf-checkpoint-receipt')
            receipt={**receipt,'drive_index_receipt':cold}
            write(receipt_path(directory),receipt)
            emit('hf_checkpoint_committed',**receipt)
            return receipt
        snapshot=manifest(directory); names={f['path'] for f in snapshot['files']}
        if kind=='checkpoint' and not REQUIRED<=names: raise ValueError('Incomplete resumable checkpoint')
        tx=identity([kind,directory.name,self.binding,snapshot]); state=self.private/tx; state.mkdir(exist_ok=True,mode=0o700)
        transaction=state/'TRANSACTION.json'
        if transaction.exists():
            saved=read(transaction)
            if saved['snapshot']!=snapshot: raise ValueError('Closed snapshot changed during upload')
            folder=saved['folder_id']
        else:
            usage=project_usage(self.service,self.parent)
            quota=self.service.about().get(fields='storageQuota').execute()['storageQuota']
            if usage+snapshot['total_bytes']+2_000_000_000>self.c['drive_cap_bytes']: raise RuntimeError('Drive project cap reached; preserve local checkpoint and stop')
            if 'limit' in quota and int(quota['limit'])-int(quota.get('usage',0))<snapshot['total_bytes']+2_000_000_000: raise RuntimeError('Drive account free quota is insufficient')
            folder=self.service.files().generateIds(count=1,space='drive',type='files').execute()['ids'][0]
            saved={'folder_id':folder,'snapshot':snapshot,'binding':self.binding}; write(transaction,saved); transaction.chmod(0o600)
        try: drive_metadata(self.service,folder)
        except Exception as exc:
            if getattr(getattr(exc,'resp',None),'status',None)!=404: raise RuntimeError('Drive transaction reconciliation failed') from None
            self.service.files().create(body={'id':folder,'name':self.c['campaign_id']+'-'+kind+'-'+directory.name,'parents':[self.parent],'mimeType':'application/vnd.google-apps.folder','appProperties':{'campaign':self.c['campaign_id'],'transaction':tx}},fields='id').execute()
        entries=[]; started=time.monotonic()
        for i,f in enumerate(snapshot['files']):
            receipt=upload_file(self.service,directory/f['path'],folder,state/f'{i:05d}.json',f'{i:05d}--'+Path(f['path']).name)
            if receipt['sha256']!=f['sha256']: raise ValueError('Artifact mutated while uploading')
            entries.append({**f,'file_id':receipt['file_id']})
        value={'schema_version':1,'status':'COMMITTED','kind':kind,'campaign_id':self.c['campaign_id'],'binding':self.binding,'files':entries,'total_bytes':snapshot['total_bytes'],'created_unix':time.time(),'runtime_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
        marker=state/'COMMITTED.json'; write(marker,value)
        result=upload_file(self.service,marker,folder,state/'commit-upload.json','COMMITTED.json')
        receipt={'folder_id':folder,'commit_file_id':result['file_id'],'commit_sha256':result['sha256'],'total_bytes':snapshot['total_bytes'],'seconds':time.monotonic()-started,'kind':kind}
        write(directory.with_name(directory.name+'.drive-receipt.json'),receipt)
        if kind=='checkpoint':
            from .hf_checkpoints import receipt_path
            receipt={**receipt,'backend':'drive','verified':True,'binding':self.binding}
            write(receipt_path(directory),receipt)
        emit('drive_committed',**receipt); return receipt
    def restore(self,receipt,destination):
        if receipt.get('backend')=='huggingface':
            if self.hf is None: raise ValueError('HF checkpoint restore is not configured')
            return self.hf.restore(receipt,destination)
        destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
        marker=destination.parent/(destination.name+'.COMMITTED.json')
        if not marker.exists(): download_verified(self.service,receipt['commit_file_id'],marker,receipt['commit_sha256'])
        if sha(marker)!=receipt['commit_sha256']: raise ValueError('Cold commit hash differs')
        commit=read(marker)
        if commit['status']!='COMMITTED' or commit['binding']!=self.binding or commit['campaign_id']!=self.c['campaign_id']: raise ValueError('Cold artifact is incomplete or belongs to another campaign/source')
        files=commit['files']; names=[str(safe_relative(f['path'])) for f in files]
        if len(names)!=len(set(names)) or sum(f['bytes'] for f in files)!=commit['total_bytes']: raise ValueError('Cold manifest accounting differs')
        if commit['kind']=='checkpoint' and not REQUIRED<=set(names): raise ValueError('Incomplete checkpoint')
        pending=sum(f['bytes'] for f in files if not (destination/f['path']).exists())
        disk_guard(self.root,pending=pending,reserve=max(commit['total_bytes'],self.c['initial_checkpoint_reservation_bytes']))
        for f in files:
            p=destination/f['path']
            if not p.resolve().is_relative_to(destination.resolve()): raise ValueError('Unsafe restore path')
            if not p.exists(): download_verified(self.service,f['file_id'],p,f['sha256'])
            if p.stat().st_size!=f['bytes'] or sha(p)!=f['sha256']: raise ValueError('Restored checkpoint content differs')
        write(destination.with_name(destination.name+'.drive-receipt.json'),receipt)
        if commit['kind']=='checkpoint':
            from .hf_checkpoints import receipt_path
            write(receipt_path(destination),{**receipt,'backend':'drive','verified':True,'binding':self.binding})
        return commit
    def small_file(self,path,kind):
        # Small ledgers are closed in uniquely named artifact directories. Keys,
        # upload session URLs and ADC credentials never enter these directories.
        src=Path(path); directory=self.root/'durable-ledgers'/(kind+'-'+identity(src.read_bytes().hex())[:16])
        directory.mkdir(parents=True,exist_ok=True); target=directory/src.name
        if not target.exists(): shutil.copy2(src,target)
        return self.commit(directory,kind=kind)
    def discover(self):
        """List campaign commits; no unauthenticated mutable 'latest' pointer."""
        items=[]; token=None
        while True:
            page=self.service.files().list(q=f"'{self.parent}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'",fields='nextPageToken,files(id,name,appProperties)',pageSize=1000,pageToken=token).execute()
            for folder in page.get('files',[]):
                if folder.get('appProperties',{}).get('campaign')!=self.c['campaign_id']: continue
                found=self.service.files().list(q=f"'{folder['id']}' in parents and trashed=false and name='COMMITTED.json'",fields='files(id,appProperties,size)',pageSize=100).execute().get('files',[])
                if len(found)==1:
                    f=found[0]; h=f.get('appProperties',{}).get('sha256')
                    if h: items.append({'folder_id':folder['id'],'commit_file_id':f['id'],'commit_sha256':h,'name':folder['name']})
            token=page.get('nextPageToken')
            if not token: break
        if self.hf is not None:
            latest=self.hf.latest()
            if latest: items.append({**latest,'name':'HF latest verified resumable checkpoint'})
        return items
