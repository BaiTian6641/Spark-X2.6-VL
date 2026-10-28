"""Variant-aware source splits and leased, bounded local TAR streaming."""
from __future__ import annotations
import gzip, hashlib, io, json, math, random, re, sqlite3
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from .common import HERE, canonical, identity, read, sha, write
from colab.formal.rolling_cache import DatasetContract, HFHTTPTransport, RollingShardCache

FAMILIES=('image','video','audiovisual','speech_in','sound_in','speech_out','sound_out','text')

def variants(packet):
    rows=packet.get('task_variants',[packet])
    if len(rows)!=packet.get('instruction_count',1): raise ValueError('Grouped variant count differs')
    return rows

def training_views(row):
    """Bound long score transcription without dropping any annotated note.

    The original 15-second waveform remains intact. Each compact supervised
    view requests a declared interval of the canonical ordered event list.
    This is a versioned derived task, not silent token or waveform truncation.
    """
    if row['task']!='music_instrument_note_understanding': return [row]
    answer=json.loads(row['answer'])
    events=sorted(answer['score_aligned_note_events'],key=lambda e:(e['clip_start_seconds'],e['gm_program_1based'],e['midi_note'],e['clip_end_seconds']))
    output=[]
    for offset in range(0,max(1,len(events)),24):
        view=dict(row); view['id']=row['id']+f'#notes-{offset//24:03d}'
        request=(f'Identify the active instruments in the full 15-second clip. Return annotated note events {offset+1} through {min(offset+24,len(events))} of {len(events)}, sorted by (start, GM program, MIDI note, end). Use JSON keys instruments (GM programs) and notes (arrays [start_seconds,end_seconds,GM_program_1based,MIDI_note]).' if row['language']=='en' else f'识别完整15秒片段中的乐器。音符按（开始时间、GM乐器号、MIDI音高、结束时间）排序，返回总计{len(events)}条中的第{offset+1}至{min(offset+24,len(events))}条。JSON键为instruments（GM乐器号）及notes（[开始秒,结束秒,GM乐器号,MIDI音高]数组）。')
        view['prompt']=request
        view['answer']=json.dumps({'instruments':[i['gm_program_1based'] for i in answer['active_instruments']], 'notes':[[e['clip_start_seconds'],e['clip_end_seconds'],e['gm_program_1based'],e['midi_note']] for e in events[offset:offset+24]]},separators=(',',':'))
        view['target']={'kind':'text','text':view['answer']}
        view['_derived_task']={'kind':'bounded_complete_score_event_views_v1','source_record':row['id'],'event_offset':offset,'event_total':len(events),'no_events_dropped':True,'waveform_crop':False}
        output.append(view)
    return output

def family(row):
    target=row.get('target',{}); direction=row.get('audio_direction'); role=row.get('media_role')
    output=direction=='text_to_audio' or role=='target' or target.get('kind')=='waveform'
    if output:
        if direction!='text_to_audio' or role!='target' or target.get('kind')!='waveform' or not target.get('path'):
            raise ValueError('Ambiguous waveform target role')
        if target['path'] not in row.get('audio',[]): raise ValueError('Output target is not in paired audio members')
        return 'speech_out' if row['task']=='speech_synthesis' else 'sound_out'
    if direction not in (None,'audio_to_text') or role not in (None,'input'): raise ValueError('Unrecognized media direction')
    if row.get('video') or len(row.get('frames',[]))>1:
        return 'audiovisual' if row.get('audio') else 'video'
    if row.get('audio'): return 'speech_in' if row['task']=='speech_recognition' else 'sound_in'
    if row.get('frames'): return 'image'
    if row['task']=='text_replay': return 'text'
    raise ValueError('No implemented modality for row '+row['task'])

def conditioning_members(row):
    """Targets are excluded by construction, including target entries in audio[]."""
    fam=family(row)
    return [] if fam.endswith('_out') else row.get('frames',[])+row.get('video',[])+row.get('audio',[])

class Union:
    def __init__(self): self.p={}
    def find(self,x):
        p=self.p; p.setdefault(x,x)
        while p[x]!=x: p[x]=p[p[x]]; x=p[x]
        return x
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b: self.p[max(a,b)]=min(a,b)

def group_keys(packet,transport):
    keys={packet['source_group']}
    for row in variants(packet):
        keys.add(row['source_group']); m=row.get('metadata',{})
        keys.update(m.get('grouping_keys',[]))
        for k in ('source_id',):
            val=row.get(k,'')
            if val.startswith('wikipedia-'): keys.add(val)
        for k in ('canonical_flickr_photo_id',):
            if m.get(k): keys.add('flickr:'+str(m[k]))
        for value in [m.get('flickr_url',''),m.get('source_url',''),m.get('source_provenance',{}).get('flickr_url','')]:
            match=re.search(r'flickr\.com/(?:photos/[^/]+/|.*?)([0-9]{5,})',str(value))
            if match: keys.add('flickr:'+match[1])
    if transport and transport['media']:
        # Whole media sets also join reversed/translated packet variants, without
        # joining unrelated synthetic sequences through a shared blank frame.
        keys.add('media-set:'+identity(sorted(x['sha256'] for x in transport['media'])))
    return sorted(keys)

def build_database(catalogue,db_path,config):
    db_path=Path(db_path)
    if db_path.exists(): raise FileExistsError('Do not replace a frozen split database')
    db=sqlite3.connect(db_path); db.executescript('''
    PRAGMA journal_mode=DELETE;
    CREATE TABLE packets(pid INTEGER PRIMARY KEY, component TEXT, shard TEXT, body TEXT, group_id TEXT);
    CREATE TABLE rows(rid TEXT PRIMARY KEY,pid INTEGER,variant INTEGER,unit INTEGER,family TEXT,language TEXT,task TEXT,split TEXT,group_id TEXT);
    ''')
    u=Union(); packet_keys={}; total=0; original_total=0
    with gzip.open(catalogue,'rt') as stream:
        for pid,line in enumerate(stream,1):
            x=json.loads(line); packet=x['record']; t=x['transport']; keys=group_keys(packet,t)
            for key in keys[1:]: u.join(keys[0],key)
            packet_keys[pid]=keys[0]
            db.execute('INSERT INTO packets VALUES(?,?,?,?,NULL)',(pid,x['component'],t['shard'] if t else 'text',json.dumps(x,ensure_ascii=False,separators=(',',':'))))
            for vi,original in enumerate(variants(packet)):
                original_total+=1
                for unit,r in enumerate(training_views(original)):
                    fam=family(r); conditioning_members(r)
                    db.execute('INSERT INTO rows VALUES(?,?,?,?,?,?,?,NULL,NULL)',(r['id'],pid,vi,unit,fam,r['language'],r['task'])); total+=1
    group_families=defaultdict(set); group_counts=defaultdict(Counter)
    for pid,fam,lang in db.execute('SELECT pid,family,language FROM rows'):
        group=u.find(packet_keys[pid]); group_families[group].add(fam+':'+lang); group_counts[group][fam+':'+lang]+=1
    cuts=[]; s=0
    for name,percent in config['split_percent'].items(): s+=percent; cuts.append((s,name))
    if s!=100: raise ValueError('Split percentages must sum to 100')
    assigned={g:next(name for cut,name in cuts if int(identity([config['seed'],g])[:8],16)%100<cut) for g in group_families}
    # Reserve at least one independent group per family/split when possible.
    for fam in sorted(set(f for fs in group_families.values() for f in fs)):
        groups=sorted((g for g,fs in group_families.items() if fam in fs),key=lambda g:identity([config['seed'],g]))
        if len(groups)<4: raise ValueError('Too few independent groups for '+fam)
        for split in config['split_percent']:
            if any(assigned[g]==split for g in groups): continue
            counts=Counter(assigned[g] for g in groups)
            candidate=next((g for g in groups if counts[assigned[g]]>1 and all(sum(assigned[h]==assigned[g] and f in group_families[h] for h in group_families)>1 for f in group_families[g])),None)
            if candidate is None: raise ValueError('Cannot form independent split coverage for '+fam)
            assigned[candidate]=split
    # Avoid a nominal bilingual holdout containing only one or two utterances.
    # Whole groups move; no labels/learner predictions participate in selection.
    totals=Counter(f for counts in group_counts.values() for f in counts for _ in range(counts[f]))
    for fam in sorted(totals):
        minimum=min(config['evaluation_per_family_language'],max(1,totals[fam]//20))
        for split in ('dev','promotion','audit'):
            have=sum(counts[fam] for g,counts in group_counts.items() if assigned[g]==split)
            while have<minimum:
                deficit=minimum-have
                candidates=[g for g,counts in group_counts.items() if assigned[g]=='train' and counts[fam]>0 and counts[fam]<totals[fam]/2]
                if not candidates: raise ValueError('Insufficient independent holdout support for '+fam)
                g=min(candidates,key=lambda g:(abs(group_counts[g][fam]-deficit),identity([config['seed'],g])))
                assigned[g]=split; have+=group_counts[g][fam]
    db.execute('CREATE INDEX row_packet ON rows(pid)')
    for pid,key in packet_keys.items():
        group=u.find(key); db.execute('UPDATE packets SET group_id=? WHERE pid=?',(group,pid))
        db.execute('UPDATE rows SET split=?,group_id=? WHERE pid=?',(assigned[group],group,pid))
    db.executescript('CREATE INDEX selection ON rows(split,family,language); CREATE INDEX packet_group ON packets(group_id);'); db.commit()
    report={'schema_version':1,'catalogue_sha256':sha(catalogue),'config_split':config['split_percent'],'seed':config['seed'],'instructions':total,'original_instructions':original_total,'adapter':'bounded_complete_score_event_views_v1',
            'groups':len(assigned),'counts':[dict(zip(('split','family','language','rows'),x)) for x in db.execute('SELECT split,family,language,COUNT(*) FROM rows GROUP BY split,family,language')],
            'split_membership_sha256':identity(list(db.execute('SELECT rid,split,group_id FROM rows ORDER BY rid'))),
            'independent_group_counts':[dict(zip(('split','family','language','groups'),x)) for x in db.execute('SELECT split,family,language,COUNT(DISTINCT group_id) FROM rows GROUP BY split,family,language')],
            'policy':'Source groups, declared grouping keys, shared Wikipedia article identity, known Flickr identity and exact whole-media sets connected before split; pretraining overlap and unprovided perceptual duplicates remain unknown.',
            'evaluation_scope':'New internal source-held-out diagnosis/promotion/audit; not a clean official public benchmark. Prior local DOCCI/COCO exposure is recorded in source metadata.'}
    db.close(); write(db_path.with_suffix('.report.json'),report); return report

class Catalogue:
    def __init__(self,db): self.db=sqlite3.connect(db); self.path=Path(db)
    def row(self,rid):
        p=self.db.execute('SELECT p.body,r.variant,r.split,r.family,r.unit FROM rows r JOIN packets p USING(pid) WHERE rid=?',(rid,)).fetchone()
        if p is None: raise KeyError(rid)
        x=json.loads(p[0]); row=training_views(variants(x['record'])[p[1]])[p[4]].copy(); row['_split']=p[2]; row['_family']=p[3]
        return row,x
    def select(self,split,families=FAMILIES,language=None):
        sql='SELECT rid,family,language,p.component,p.shard FROM rows r JOIN packets p USING(pid) WHERE split=? AND family IN ('+','.join('?' for _ in families)+')'
        args=[split,*families]
        if language: sql+=' AND language=?'; args.append(language)
        return list(self.db.execute(sql+' ORDER BY rid',args))
    def evaluation_ids(self,split,per=32):
        items=self.select(split); buckets=defaultdict(list)
        for r in items: buckets[(r[1],r[2])].append(r[0])
        chosen=[rid for key in sorted(buckets) for rid in sorted(buckets[key],key=lambda x:identity([split,x]))[:per]]
        locations={r[0]:(r[3],r[4],r[0]) for r in items}
        return sorted(chosen,key=lambda rid:locations[rid])
    def plan(self,stage,seed):
        rows=self.select('train',stage['families']); cap=stage.get('max_records_per_family')
        if cap:
            buckets=defaultdict(list)
            for r in rows: buckets[r[1]].append(r)
            rows=[r for fam in sorted(buckets) for r in sorted(buckets[fam],key=lambda x:identity([seed,x[0]]))[:cap]]
        if stage.get('replay_fraction'):
            pool=[r for r in self.select('train') if r[1] not in stage['families']]
            random.Random(seed).shuffle(pool); rows+=pool[:round(len(rows)*stage['replay_fraction'])]
        order=[]
        for epoch in range(stage['epochs']):
            buckets=defaultdict(list); rng=random.Random(seed+epoch)
            for r in rows: buckets[(r[3],r[4])].append(r[0])
            queues=defaultdict(list)
            shard_keys=sorted(buckets); rng.shuffle(shard_keys)
            for key in shard_keys:
                rng.shuffle(buckets[key]); queues[key[0]].extend([buckets[key][i:i+128] for i in range(0,len(buckets[key]),128)])
            queues={key:deque(windows) for key,windows in queues.items()}
            # Interleave sources while finishing each source's current shard.
            # Randomizing all shard windows globally would redownload gigabytes
            # for each128 rows with a small cache, wasting GPU rental time.
            while queues:
                component=rng.choice(sorted(queues)); order.extend(queues[component].popleft())
                if not queues[component]: del queues[component]
        return order

class CommittedCursor:
    """Issued microbatches are speculative until their optimizer update commits."""
    def __init__(self,order,state=None):
        self.order=order; self.order_sha256=identity(order); self.committed=0; self.issued=0
        if state:
            if state['order_sha256']!=self.order_sha256: raise ValueError('Training order changed across resume')
            self.committed=int(state['committed']); self.issued=self.committed
    def issue(self):
        if self.issued>=len(self.order): raise StopIteration
        rid=self.order[self.issued]; self.issued+=1; return rid
    def record_computed(self,index):
        if index!=self.issued: raise ValueError('Model consumed a skipped or repeated source position')
        self.issued+=1
    def commit(self,count=None):
        target=self.issued if count is None else self.committed+count
        if not self.committed<=target<=self.issued: raise ValueError('Invalid optimizer commitment')
        self.committed=target
    def state(self): return {'order_sha256':self.order_sha256,'committed':self.committed,'records':len(self.order)}

def cursor_rows(order,cursor):
    """A metadata peek/iterator construction must not consume training state."""
    start=cursor.committed
    for index in range(start,len(order)):
        yield {'rid':order[index],'_index':index}

class MediaReader:
    def __init__(self,root,handoff,config,token=None,reserve=0):
        self.root=Path(root); self.config=config; self.checkpoint_slot_bytes=config['initial_checkpoint_reservation_bytes']; self.caches={}; self.contracts={}
        for c in handoff['components']:
            contract=DatasetContract(self.root/'indices'/c['id']/'RELEASE.json',repo_id=handoff['repo_id'],commit=handoff['revision'],index_sha256=c['release_sha256'],prefix=c['prefix'])
            self.contracts[c['id']]=contract
            self.caches[c['id']]=RollingShardCache(contract,HFHTTPTransport(contract,token),root=self.root/'media-cache'/c['id'],max_bytes=config['cache_bytes_per_component'],reserve_free_bytes=44_000_000_000+reserve,lookahead=0)
    @contextmanager
    def packet(self,x):
        if not x['transport']:
            yield lambda name: (_ for _ in ()).throw(ValueError('Text has no media')); return
        component=x['component']; t=x['transport']; c=self.contracts[component]
        allocated=sum(p.stat().st_size for p in (self.root/'checkpoints').rglob('*') if p.is_file()) if (self.root/'checkpoints').exists() else 0
        self.caches[component].reserve_free_bytes=44_000_000_000+max(0,2*self.checkpoint_slot_bytes-allocated)
        with self.caches[component].lease(t['shard']) as lease:
            members={v['member']:v for v in t['media']}
            def get(name):
                if name not in members: raise ValueError('Requested member is not in the released packet')
                m=members[name]; rec=c.member_records[name]
                if m['length']!=rec['bytes'] or m['sha256']!=rec['sha256']: raise ValueError('Member binding differs')
                # Read exact bytes under a lease; no TAR extraction or path traversal.
                with lease.open(name,offset_data=m['offset']) as source:
                    payload=source.read(m['length'])
                if hashlib.sha256(payload).hexdigest()!=m['sha256']: raise ValueError('Member hash differs')
                return payload
            yield get

def decode_audio(payload):
    import numpy as np, soundfile as sf
    wav,sr=sf.read(io.BytesIO(payload),dtype='float32',always_2d=True)
    if not len(wav) or not np.isfinite(wav).all(): raise ValueError('Invalid decoded audio')
    return wav.T.copy(),int(sr)

def decode_video(payload,count,origin_seconds=None):
    """Bounded sequential two-pass decode using actual PTS, not guessed FPS."""
    import av, numpy as np
    pts=[]
    with av.open(io.BytesIO(payload)) as source:
        for frame in source.decode(video=0):
            if frame.pts is None or frame.time_base is None: raise ValueError('Unknown video PTS')
            pts.append(float(frame.pts*frame.time_base))
    if not pts or any(b<=a for a,b in zip(pts,pts[1:])): raise ValueError('Unusable presentation clock')
    ids=np.linspace(0,len(pts)-1,count).round().astype(int).tolist(); wanted=set(ids); frames={}
    with av.open(io.BytesIO(payload)) as source:
        for i,frame in enumerate(source.decode(video=0)):
            if i in wanted: frames[i]=frame.to_image().convert('RGB')
            if i>=max(ids): break
    origin=pts[0] if origin_seconds is None else float(origin_seconds)
    return [frames[i] for i in ids],[pts[i]-origin for i in ids]

def prepare_media(row,get,config):
    from PIL import Image
    conditioning_members(row); result={'images':[],'times':[],'capture_times':None,'audio':None,'target_audio':None,'motion':None,'motion_times':None}
    if row['_family'].endswith('_out'):
        result['target_audio']=decode_audio(get(row['target']['path'])); return result
    result['images']=[Image.open(io.BytesIO(get(p))).convert('RGB') for p in row.get('frames',[])]
    result['times']=row.get('media_times_s',[0.]*len(result['images'])); result['capture_times']=row.get('capture_times_s')
    if row.get('video'):
        if len(row['video'])!=1: raise ValueError('Only one paired video is qualified per record')
        origin=None
        if row['_family']=='audiovisual':
            clock=row.get('metadata',{}).get('clock',{})
            if 'waveform_origin_on_media_clock_seconds' not in clock: raise ValueError('Audiovisual waveform/media clock origin is unknown')
            origin=clock['waveform_origin_on_media_clock_seconds']
        frames,times=decode_video(get(row['video'][0]),config['motion_frames'],origin_seconds=origin)
        result['motion']=frames; result['motion_times']=times
        ids=[round(i*(len(frames)-1)/(config['video_semantic_frames']-1)) for i in range(config['video_semantic_frames'])]
        result['images']=[frames[i] for i in ids]; result['times']=[times[i] for i in ids]
        result['capture_times']=None  # No invented capture clock for a derivative.
    elif len(result['images'])>1:
        result['motion']=result['images']; result['motion_times']=result['times']
    if row.get('audio'):
        if len(row['audio'])!=1: raise ValueError('Only one explicitly paired input waveform is qualified')
        result['audio']=decode_audio(get(row['audio'][0]))
    return result
