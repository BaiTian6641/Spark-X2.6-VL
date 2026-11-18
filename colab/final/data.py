"""Variant-aware source splits, explicit visual-input contracts, RGBA preservation, and bounded TAR streaming."""
from __future__ import annotations
import gzip, hashlib, io, json, math, random, re, sqlite3
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from .common import HERE, canonical, identity, read, sha, write
from colab.formal.rolling_cache import DatasetContract, HFHTTPTransport, RollingShardCache

FAMILIES=('image','video','audiovisual','speech_in','sound_in','speech_out','sound_out','text')

VISUAL_KINDS=(
    'independent_image',
    'multiview_render',
    'uv_atlas',
    'state_pair',
    'temporal_frame',
)

VISUAL_KIND_TO_INDEX={name:idx for idx,name in enumerate(VISUAL_KINDS)}

VIEW_IDS=(
    'single',
    'front',
    'south',
    'right',
    'top',
    'iso',
    'uv_vanilla',
    'uv_pbr',
    'before',
    'after',
    'frame',
)

VIEW_ID_TO_INDEX={name:idx for idx,name in enumerate(VIEW_IDS)}

TEMPORAL_TASKS={
    'video_caption',
    'video_box_tracking',
    'region_dense_video_caption',
    'temporal_order',
    'capture_elapsed',
    'source_capture_elapsed',
    'synccraft_behavior_animation',
}

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

def infer_visual_kind(row:dict[str,Any])->str:
    """Resolve explicit visual input kind without turning static multiviews into video."""
    explicit=row.get('visual_kind')
    if explicit is not None:
        if explicit not in VISUAL_KINDS:
            raise ValueError(f'Unknown visual_kind {explicit!r}; expected one of {VISUAL_KINDS}')
        return explicit
    if row.get('video') or row.get('media_kind')=='video_frames':
        return 'temporal_frame'
    task=str(row.get('task',''))
    frames=row.get('frames',[])
    if task in TEMPORAL_TASKS or task.startswith('video_'):
        return 'temporal_frame'
    if len(frames)>1:
        times=row.get('media_times_s') or []
        captures=row.get('capture_times_s') or []
        if any(float(t)!=0.0 for t in times) or any(c is not None for c in captures):
            return 'temporal_frame'
        return 'multiview_render'
    return 'independent_image'

PURE_TEXT_TASKS={
    'text_replay',
    'general_reasoning',
    'fable5_swe_agent',
    'fable5_reasoning',
    'claude_code_agent',
    'terminal_bench_agent',
    'terminal_bench_3_agent',
    'terminal_bench_4_agent',
    'deepswe_agent',
    'deepswe_repo_agent',
    'swe_bench_pro_agent',
    'swebench_pro_patch',
    'r2e_swe_agent',
    'swe_gym_agent',
    'toucan_interleaved_tool',
    'toucan_interleaved_tool_agent',
    'xlam_interleaved_tool',
    'openr1_math',
    'openr1_verified_math',
    'openscience_reasoning',
    'openscience_reasoning_2',
    'spark_234k_native_reasoning',
    'supergpqa_graduate_science',
    'world_knowledge_2026',
    'arc_agi2_symbolic',
    'arc_agi_2_symbolic_induction',
    'no_robots_prose',
    'helpsteer2_prose',
    'helpsteer2_human_prose',
    'openthoughts_reasoning',
    'gpqa_extended_science',
    'hle_text_reasoning',
    'synccraft_sculpted_style_creation',
    'synccraft_v4_modular_assembly',
    'synccraft_v41_sculpted_v8',
}

def family(row):
    target=row.get('target',{}); direction=row.get('audio_direction'); role=row.get('media_role')
    output=direction=='text_to_audio' or role=='target' or target.get('kind')=='waveform'
    if output:
        if direction!='text_to_audio' or role!='target' or target.get('kind')!='waveform' or not target.get('path'):
            raise ValueError('Ambiguous waveform target role')
        if target['path'] not in row.get('audio',[]): raise ValueError('Output target is not in paired audio members')
        return 'speech_out' if row['task']=='speech_synthesis' else 'sound_out'
    if direction not in (None,'audio_to_text') or role not in (None,'input'): raise ValueError('Unrecognized media direction')
    vkind=infer_visual_kind(row) if (row.get('video') or row.get('frames')) else None
    if row.get('video') or (len(row.get('frames',[]))>1 and vkind=='temporal_frame'):
        return 'audiovisual' if row.get('audio') else 'video'
    if row.get('audio'): return 'speech_in' if row['task']=='speech_recognition' else 'sound_in'
    if row.get('frames'): return 'image'
    if row['task'] in PURE_TEXT_TASKS or str(row.get('task','')).startswith('synccraft_'): return 'text'
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

def group_keys(packet,transport,allow_research_nc=False):
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
        if row.get('provenance') or m.get('provenance') or str(row.get('task','')).startswith('synccraft_'):
            from .synccraft_bridge import synccraft_group_keys,validate_synccraft_license_record
            validate_synccraft_license_record(row,allow_research_nc=allow_research_nc)
            keys.update(synccraft_group_keys(row))
        elif m.get('asset_id') or m.get('family_id'):
            from .synccraft_bridge import synccraft_group_keys
            keys.update(synccraft_group_keys(row))
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
    allow_nc=bool(config.get('allow_synccraft_research_nc',False))
    with gzip.open(catalogue,'rt') as stream:
        for pid,line in enumerate(stream,1):
            x=json.loads(line); packet=x['record']; t=x['transport']; keys=group_keys(packet,t,allow_research_nc=allow_nc)
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
            'policy':'Source groups, declared grouping keys, SyncCraft asset/family lineage, shared Wikipedia article identity, known Flickr identity and exact whole-media sets connected before split; pretraining overlap and unprovided perceptual duplicates remain unknown.',
            'evaluation_scope':'New internal source-held-out diagnosis/promotion/audit; not a clean official public benchmark. Prior local DOCCI/COCO exposure is recorded in source metadata.'}
    db.close(); write(db_path.with_suffix('.report.json'),report); return report

def _stratified_group_shuffle(rids, group_by_rid, rng):
    """Dynamic anti-clustering shuffle: spread sibling tasks of the same asset group across waves."""
    by_group = defaultdict(list)
    for rid in rids:
        by_group[group_by_rid.get(rid, rid)].append(rid)
    group_keys = sorted(by_group)
    for g in group_keys:
        rng.shuffle(by_group[g])
    out = []
    while group_keys:
        wave = []
        next_keys = []
        for g in group_keys:
            items = by_group[g]
            if items:
                wave.append(items.pop())
            if items:
                next_keys.append(g)
        rng.shuffle(wave)
        out.extend(wave)
        group_keys = next_keys
    return out

def _stratified_epoch_slice(
    fam_rows,
    target_n,
    task_by_rid,
    seed_prefix,
    epoch,
    task_weight_multipliers=None,
    task_repeat_cap=None,
):
    """Draw a deterministic, task-balanced, cyclic-disjoint slice of size `target_n` for `epoch`."""
    target_n = int(target_n)
    mults = dict(task_weight_multipliers or {})
    rep_caps = dict(task_repeat_cap or {})
    if target_n <= 0 or (not mults and not rep_caps and len(fam_rows) <= target_n):
        return list(fam_rows)
    by_task = defaultdict(list)
    for r in fam_rows:
        by_task[str(task_by_rid.get(r[0], '') or 'default')].append(r)
    tasks = sorted(by_task)
    pools = {
        t: sorted(by_task[t], key=lambda x: identity([*seed_prefix, 'task_pool', t, x[0]]))
        for t in tasks
    }
    caps = {
        t: len(pools[t]) * max(1, int(rep_caps.get(t, 1)))
        for t in tasks
    }
    if sum(caps.values()) <= target_n:
        target_n = sum(caps.values())
    quotas = {t: 0 for t in tasks}
    # Minimum floor so rare high-signal tasks are represented every epoch
    floor_each = max(1, target_n // max(2, 2 * len(tasks)))
    remaining = target_n
    for t in tasks:
        give = min(caps[t], floor_each, remaining)
        quotas[t] = give
        remaining -= give
    # Distribute remaining budget proportional to sqrt(pool_size) * priority multiplier
    while remaining > 0:
        avail = [t for t in tasks if quotas[t] < caps[t]]
        if not avail:
            break
        weights = {
            t: math.sqrt(len(pools[t])) * max(0.05, float(mults.get(t, 1.0)))
            for t in avail
        }
        w_sum = sum(weights.values()) or 1.0
        progressed = False
        for t in sorted(avail, key=lambda k: (-weights[k], k)):
            if remaining <= 0:
                break
            share = max(1, int(round(remaining * weights[t] / w_sum)))
            add = min(caps[t] - quotas[t], share, remaining)
            if add > 0:
                quotas[t] += add
                remaining -= add
                progressed = True
        if not progressed:
            break
    selected = []
    for t in tasks:
        q = quotas[t]
        if q <= 0:
            continue
        pool = pools[t]
        ln = len(pool)
        start = (epoch * q) % ln
        selected.extend(pool[(start + j) % ln] for j in range(q))
    return selected


class Catalogue:
    def __init__(self,db,modalities='omni'): self.db=sqlite3.connect(db); self.path=Path(db); self.modalities=modalities
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
    def plan(self,stage,seed,modalities=None):
        eff_mod=modalities or getattr(self,'modalities','omni')
        rows=self.select('train',stage['families']); cap=stage.get('max_records_per_family')
        dyn_cap=stage.get('dynamic_epoch_records_per_family')
        cols={col[1] for col in self.db.execute('PRAGMA table_info(rows)')}
        task_by_rid=dict(self.db.execute("SELECT rid,task FROM rows WHERE split='train'")) if 'task' in cols else {}
        allowed_sc=set(stage['synccraft_tasks']) if stage.get('synccraft_tasks') else None
        exclude_sc_anim=(stage.get('include_synccraft_animation') is False)
        if (allowed_sc is not None or exclude_sc_anim) and task_by_rid:
            filtered=[]
            for r in rows:
                t=str(task_by_rid.get(r[0],'') or '')
                if t.startswith('synccraft_'):
                    if exclude_sc_anim and ('animation' in t or 'motion' in t): continue
                    if allowed_sc is not None and t not in allowed_sc: continue
                filtered.append(r)
            if filtered: rows=filtered
        if cap:
            buckets=defaultdict(list)
            for r in rows: buckets[r[1]].append(r)
            rows=[r for fam in sorted(buckets) for r in sorted(buckets[fam],key=lambda x:identity([seed,x[0]]))[:cap]]
        by_fam_all=defaultdict(list)
        for r in rows: by_fam_all[r[1]].append(r)
        replay_pool=[]
        if stage.get('replay_fraction'):
            replay_fams=('image','video','text') if eff_mod=='vision_text' else FAMILIES
            if stage.get('alignment_only'):
                replay_fams=tuple(f for f in replay_fams if f!='text')
            replay_pool=[r for r in self.select('train',replay_fams) if r[1] not in stage['families']] if replay_fams else []
        group_by_rid=dict(self.db.execute("SELECT rid,group_id FROM rows WHERE split='train'")) if 'group_id' in cols else {}
        order=[]
        for epoch in range(stage['epochs']):
            ep_seed=int(identity([seed,stage.get('name',''),'epoch',epoch])[:16],16)
            buckets=defaultdict(list); rng=random.Random(ep_seed)
            if dyn_cap:
                epoch_rows=[]
                for fam in sorted(by_fam_all):
                    f_cap=dyn_cap.get(fam) if isinstance(dyn_cap,dict) else int(dyn_cap)
                    if f_cap:
                        epoch_rows.extend(
                            _stratified_epoch_slice(
                                by_fam_all[fam],
                                f_cap,
                                task_by_rid,
                                [seed,stage.get('name',''),fam],
                                epoch,
                                task_weight_multipliers=stage.get('task_weight_multipliers'),
                                task_repeat_cap=stage.get('task_repeat_cap'),
                            )
                        )
                    else:
                        epoch_rows.extend(by_fam_all[fam])
            else:
                epoch_rows=list(rows)
            replay_count=min(len(replay_pool),round(len(epoch_rows)*stage['replay_fraction'])) if stage.get('replay_fraction') else 0
            if replay_count>0:
                pool_copy=list(replay_pool)
                rng.shuffle(pool_copy)
                epoch_rows.extend(pool_copy[:replay_count])
            for r in epoch_rows: buckets[(r[3],r[4])].append(r[0])
            queues=defaultdict(list)
            shard_keys=sorted(buckets); rng.shuffle(shard_keys)
            for key in shard_keys:
                shuffled_bucket=_stratified_group_shuffle(buckets[key],group_by_rid,rng)
                queues[key[0]].extend([shuffled_bucket[i:i+128] for i in range(0,len(shuffled_bucket),128)])
            queues={key:deque(windows) for key,windows in queues.items()}
            # Interleave sources while finishing each source's current shard.
            # Randomizing all shard windows globally would redownload gigabytes
            # for each 128 rows with a small cache, wasting GPU rental time.
            while queues:
                component=rng.choice(sorted(queues)); order.extend(queues[component].popleft())
                if not queues[component]: del queues[component]
        return order

class CommittedCursor:
    """Worker-issued media may finish out of order; commits stay contiguous."""
    def __init__(self,order,state=None):
        self.order=order; self.order_sha256=identity(order); self.committed=0; self.issued=0; self.pending=set()
        if state:
            if state['order_sha256']!=self.order_sha256: raise ValueError('Training order changed across resume')
            self.committed=int(state['committed']); self.issued=self.committed
    def _computed(self,index):
        if not self.committed<=index<len(self.order) or index in self.pending: raise ValueError('Model consumed a skipped or repeated source position')
        self.pending.add(index); self.issued=max(self.issued,index+1)
    def issue(self):
        if self.issued>=len(self.order): raise StopIteration
        index=self.issued; self._computed(index); return self.order[index]
    def record_computed(self,index): self._computed(index)
    def commit(self,count=None):
        if count is not None and not 0<=count<=len(self.pending): raise ValueError('Optimizer acknowledgement does not cover the computed microbatches')
        advanced=0
        while self.committed in self.pending and (count is None or advanced<count):
            self.pending.remove(self.committed); self.committed+=1; advanced+=1
        if count is not None and advanced!=count: raise ValueError('Cannot commit across an uncomputed source position')
        self.issued=max(self.issued,self.committed)
    def state(self): return {'order_sha256':self.order_sha256,'committed':self.committed,'records':len(self.order)}

try:
    from torch.utils.data import IterableDataset as _TorchIterableDataset
except ImportError:  # pragma: no cover
    _TorchIterableDataset = object


class CursorStreamDataset(_TorchIterableDataset):
    """Picklable top-level IterableDataset safe for `multiprocessing_context='spawn'`."""

    def __init__(self, order: list[str], start: int):
        self.order = list(order)
        self.start = int(start)
        self.remaining = max(0, len(self.order) - self.start)

    def __len__(self) -> int:
        return self.remaining

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not isinstance(index, int):
            raise TypeError('Cursor dataset indices must be integers')
        if not 0 <= index < self.remaining:
            raise IndexError(index)
        absolute = self.start + index
        return {'rid': self.order[absolute], '_index': absolute}

    def __iter__(self):
        from torch.utils.data import get_worker_info
        worker = get_worker_info()
        step = worker.num_workers if worker else 1
        offset = worker.id if worker else 0
        for index in range(self.start + offset, len(self.order), step):
            yield {'rid': self.order[index], '_index': index}


def cursor_rows(order,cursor):
    """Shard metadata by worker while retaining a contiguous committed prefix."""
    from torch.utils.data import get_worker_info
    worker=get_worker_info(); start=cursor.committed
    step=worker.num_workers if worker else 1; offset=worker.id if worker else 0
    for index in range(start+offset,len(order),step):
        yield {'rid':order[index],'_index':index}

def cursor_dataset(order,cursor):
    return CursorStreamDataset(order, cursor.committed)


class MediaReader:
    def __init__(self,root,handoff,config,token=None,reserve=0):
        self.root=Path(root); self.config=config; self.checkpoint_slot_bytes=config['initial_checkpoint_reservation_bytes']; self.caches={}; self.contracts={}
        free_floor=config.get('local_free_floor_bytes',44_000_000_000)
        for c in handoff['components']:
            c_repo=c.get('repo_id',handoff['repo_id'])
            c_rev=c.get('revision',handoff['revision'])
            contract=DatasetContract(self.root/'indices'/c['id']/'RELEASE.json',repo_id=c_repo,commit=c_rev,index_sha256=c['release_sha256'],prefix=c['prefix'])
            self.contracts[c['id']]=contract
            self.caches[c['id']]=RollingShardCache(contract,HFHTTPTransport(contract,token),root=self.root/'media-cache'/c['id'],max_bytes=config['cache_bytes_per_component'],reserve_free_bytes=free_floor+reserve,lookahead=0)
    @contextmanager
    def packet(self,x):
        if not x['transport']:
            yield lambda name: (_ for _ in ()).throw(ValueError('Text has no media')); return
        component=x['component']; t=x['transport']; c=self.contracts[component]
        allocated=sum(p.stat().st_size for p in (self.root/'checkpoints').rglob('*') if p.is_file()) if (self.root/'checkpoints').exists() else 0
        free_floor=self.config.get('local_free_floor_bytes',44_000_000_000)
        self.caches[component].reserve_free_bytes=free_floor+max(0,2*self.checkpoint_slot_bytes-allocated)
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
    """Read decoded presentation timestamps, then decode only selected images without global swscale slice-threads."""
    import av, numpy as np
    from PIL import Image
    pts=[]
    with av.open(io.BytesIO(payload)) as source:
        vstream=source.streams.video[0]
        vstream.thread_type='NONE'
        vstream.thread_count=1
        for frame in source.decode(video=0):
            if frame.pts is None or frame.time_base is None: raise ValueError('Unknown video PTS')
            pts.append(float(frame.pts*frame.time_base))
    if not pts or any(b<=a for a,b in zip(pts,pts[1:])): raise ValueError('Unusable presentation clock')
    ids=np.linspace(0,len(pts)-1,count).round().astype(int).tolist(); wanted=set(ids); frames={}
    reformatter=av.video.reformatter.VideoReformatter()
    with av.open(io.BytesIO(payload)) as source:
        vstream=source.streams.video[0]
        vstream.thread_type='NONE'
        vstream.thread_count=1
        for i,frame in enumerate(source.decode(video=0)):
            if i in wanted:
                rgb_arr=reformatter.reformat(frame,format='rgb24',threads=1).to_ndarray()
                frames[i]=Image.fromarray(rgb_arr,mode='RGB')
            if i>=max(ids): break
    if len(frames)!=len(wanted): raise ValueError('Video decode did not produce the selected frames')
    origin=pts[0] if origin_seconds is None else float(origin_seconds)
    return [frames[i] for i in ids],[pts[i]-origin for i in ids]

def decode_visual_image(payload:bytes):
    """Decode an image while preserving RGBA/alpha channels whenever present."""
    import numpy as np
    from PIL import Image
    raw=Image.open(io.BytesIO(payload))
    has_alpha=raw.mode in ('RGBA','LA') or ('transparency' in raw.info)
    if has_alpha:
        rgba=raw.convert('RGBA')
        alpha=np.asarray(rgba.getchannel('A'),dtype=np.float32)/255.0
        return rgba,alpha
    rgb=raw.convert('RGB')
    return rgb,None

def extract_image_rgb_and_alpha(image):
    """Split a PIL RGB or RGBA image into an RGB PIL image plus float32 alpha array `[H, W]` (or None)."""
    import numpy as np
    if image.mode in ('RGBA','LA') or ('transparency' in getattr(image,'info',{})):
        rgba=image.convert('RGBA')
        rgb=Image.merge('RGB',rgba.split()[:3]) if False else rgba.convert('RGB')
        alpha=np.asarray(rgba.getchannel('A'),dtype=np.float32)/255.0
        return rgb,alpha
    return image.convert('RGB'),None

def normalize_visual_units(row:dict[str,Any],n_images:int,times:list[float],capture_times:list[float|None]|None)->list[dict[str,Any]]:
    """Build explicit per-image metadata (`visual_kind`, `view_id`, camera basis, timestamps, UV flag)."""
    if n_images==0: return []
    vkind=infer_visual_kind(row)
    declared=row.get('visual_units') or []
    units=[]
    default_multiview=('south','right','top','iso')
    for idx in range(n_images):
        raw=declared[idx] if idx<len(declared) and isinstance(declared[idx],dict) else {}
        ukind=raw.get('visual_kind',vkind)
        if ukind not in VISUAL_KINDS:
            raise ValueError(f'Invalid unit visual_kind {ukind!r}')
        default_view=(
            'frame' if ukind=='temporal_frame'
            else ('before' if ukind=='state_pair' and idx==0 else ('after' if ukind=='state_pair' else (
                'uv_vanilla' if ukind=='uv_atlas' and idx==0 else (
                    default_multiview[idx] if ukind=='multiview_render' and idx<len(default_multiview) else 'single'
                )
            )))
        )
        view_id=raw.get('view_id',default_view)
        if view_id not in VIEW_ID_TO_INDEX:
            view_id='single'
        cam_r=list(raw.get('camera_right',(1.0,0.0,0.0)))[:3]
        cam_u=list(raw.get('camera_up',(0.0,1.0,0.0)))[:3]
        while len(cam_r)<3: cam_r.append(0.0)
        while len(cam_u)<3: cam_u.append(0.0)
        t_val=float(raw['timestamp_s']) if 'timestamp_s' in raw else (float(times[idx]) if idx<len(times) else 0.0)
        has_ts=bool(raw.get('has_timestamp',ukind=='temporal_frame' or t_val!=0.0))
        cap_val=capture_times[idx] if capture_times is not None and idx<len(capture_times) else None
        is_atlas=bool(raw.get('is_pixel_art_atlas',view_id in ('uv_vanilla','uv_pbr')))
        units.append({
            'visual_kind':ukind,
            'visual_kind_index':VISUAL_KIND_TO_INDEX[ukind],
            'view_id':view_id,
            'view_id_index':VIEW_ID_TO_INDEX[view_id],
            'camera_right':[float(x) for x in cam_r],
            'camera_up':[float(x) for x in cam_u],
            'ortho_scale':float(raw.get('ortho_scale',1.0)),
            'timestamp_s':t_val if has_ts else 0.0,
            'has_timestamp':has_ts,
            'capture_s':None if cap_val is None else float(cap_val),
            'is_pixel_art_atlas':is_atlas,
        })
    return units

def prepare_media(row,get,config):
    conditioning_members(row)
    result={
        'images':[],
        'alphas':[],
        'visual_kind':None,
        'visual_units':[],
        'times':[],
        'capture_times':None,
        'audio':None,
        'target_audio':None,
        'motion':None,
        'motion_times':None,
    }
    if row['_family'].endswith('_out'):
        result['target_audio']=decode_audio(get(row['target']['path'])); return result

    decoded=[decode_visual_image(get(p)) for p in row.get('frames',[])]
    result['images']=[im for im,_ in decoded]
    result['alphas']=[al for _,al in decoded]
    result['times']=list(row.get('media_times_s',[0.]*len(result['images'])))
    result['capture_times']=row.get('capture_times_s')

    if row.get('video'):
        if len(row['video'])!=1: raise ValueError('Only one paired video is qualified per record')
        origin=None
        if row['_family']=='audiovisual':
            clock=row.get('metadata',{}).get('clock',{})
            if 'waveform_origin_on_media_clock_seconds' not in clock: raise ValueError('Audiovisual waveform/media clock origin is unknown')
            origin=clock['waveform_origin_on_media_clock_seconds']
        frames,times=decode_video(get(row['video'][0]),config['motion_frames'],origin_seconds=origin)
        if config.get('modalities','omni')=='omni':
            result['motion']=frames; result['motion_times']=times
        ids=[round(i*(len(frames)-1)/(config['video_semantic_frames']-1)) for i in range(config['video_semantic_frames'])]
        result['images']=[frames[i] for i in ids]
        result['alphas']=[None]*len(result['images'])
        result['times']=[times[i] for i in ids]
        result['capture_times']=None  # No invented capture clock for a derivative.
    else:
        vkind=infer_visual_kind(row) if result['images'] else None
        # Only genuine temporal_frame sequences enter the motion tower, and only in omni mode.
        # Static multiview renders, UV atlases, and before/after state pairs NEVER become video!
        if len(result['images'])>1 and vkind=='temporal_frame' and config.get('modalities','omni')=='omni':
            result['motion']=[im.convert('RGB') for im in result['images']]
            result['motion_times']=result['times']

    if result['images']:
        result['visual_kind']=infer_visual_kind(row)
        result['visual_units']=normalize_visual_units(row,len(result['images']),result['times'],result['capture_times'])

    if row.get('audio'):
        if len(row['audio'])!=1: raise ValueError('Only one explicitly paired input waveform is qualified')
        result['audio']=decode_audio(get(row['audio'][0]))
    return result


def apply_dynamic_training_augmentation(row, media, rng, config):
    """Training-only dynamic view permutation, pilot synthesis, and subtle photometric jitter.

    Never called during evaluation (`evaluate.py`), preserving frozen held-out benchmarks while
    preventing template/slot/pixel memorization across training epochs.
    """
    import numpy as np
    from PIL import Image

    if not config.get('dynamic_training_augmentation', False):
        return row, media

    aug_row = dict(row)
    aug_media = dict(media)
    task = str(aug_row.get('task', '') or '')

    # 1. Dynamic on-the-fly synthesis ONLY for synthetic pilot `text_replay` rows (never real SparkX2.6-MM encyclopedia rows)
    if config.get('dynamic_pilot_synthesis', True) and task == 'text_replay' and not (aug_row.get('metadata') or {}).get('real_text_corpus'):
        a = rng.randint(4, 49)
        b = rng.randint(4, 49)
        c = rng.randint(2, 31)
        op_mode = rng.randint(0, 2)
        lang = aug_row.get('language', 'en')
        thinking = bool(aug_row.get('enable_thinking', False))
        if op_mode == 0:
            val = a * b
            prompt = f'Compute {a} * {b} and state the integer result.' if lang == 'en' else f'计算 {a} 乘以 {b} 并给出整数结果。'
            trace = f'Multiply {a} by {b} to get {val}.'
        elif op_mode == 1:
            val = a * b + c
            prompt = f'Evaluate {a} * {b} + {c} and return only the integer value.' if lang == 'en' else f'计算 {a} * {b} + {c} 并仅给出整数结果。'
            trace = f'First {a} * {b} = {a * b}, then add {c} -> {val}.'
        else:
            val = (a + b) * c
            prompt = f'Compute ({a} + {b}) * {c} and give the exact integer.' if lang == 'en' else f'计算 ({a} + {b}) * {c} 并给出准确整数结果。'
            trace = f'Sum {a} + {b} = {a + b}, multiply by {c} -> {val}.'
        aug_row['prompt'] = prompt
        aug_row['answer'] = f'<think>\n{trace}\n</think>\n{val}' if thinking else str(val)
        aug_row['target'] = {'kind': 'text', 'text': str(val)}
        return aug_row, aug_media

    # 2. Dynamic on-the-fly RGBA cutout/opaque texture synthesis for `rgba_visual_pilot`
    if config.get('dynamic_pilot_synthesis', True) and task == 'rgba_visual_pilot' and aug_media.get('images'):
        has_cutout = bool(rng.getrandbits(1))
        h_px, w_px = 64, 64
        arr = np.zeros((h_px, w_px, 4), dtype=np.uint8)
        r_c, g_c, b_c = rng.randint(25, 235), rng.randint(25, 235), rng.randint(25, 235)
        arr[:, :, 0] = r_c
        arr[:, :, 1] = g_c
        arr[:, :, 2] = b_c
        # Add subtle procedural border/stripe RGB pattern so opaque vs cutout cannot rely on flat RGB stats
        stripe = rng.randint(4, 12)
        arr[::stripe, :, :3] = np.clip(arr[::stripe, :, :3].astype(np.int16) + rng.randint(-30, 30), 0, 255).astype(np.uint8)
        arr[:, :, 3] = 255
        if has_cutout:
            y0 = rng.randint(8, 24)
            y1 = rng.randint(y0 + 12, 56)
            x0 = rng.randint(8, 24)
            x1 = rng.randint(x0 + 12, 56)
            arr[y0:y1, x0:x1, 3] = 0
        rgb_img = Image.fromarray(arr[:, :, :3], mode='RGB')
        alpha_arr = arr[:, :, 3].astype(np.float32) / 255.0
        aug_media['images'] = [rgb_img]
        aug_media['alphas'] = [alpha_arr]
        ans_txt = 'transparent_cutout' if has_cutout else 'opaque_solid'
        aug_row['answer'] = ans_txt
        aug_row['target'] = {'kind': 'text', 'text': ans_txt}
        return aug_row, aug_media

    # 3. Dynamic calibrated multi-view camera permutation (`multiview_render`)
    vkind = aug_media.get('visual_kind')
    imgs = list(aug_media.get('images') or [])
    units = list(aug_media.get('visual_units') or [])
    if (
        config.get('dynamic_multiview_shuffle', True)
        and vkind == 'multiview_render'
        and len(imgs) > 1
        and len(units) == len(imgs)
    ):
        orig_views = [str(u.get('view_id', '')) for u in units]
        old_phrase = f"orthographic views ({', '.join(orig_views)})"
        if all(orig_views) and old_phrase in str(aug_row.get('prompt', '')):
            perm = list(range(len(imgs)))
            rng.shuffle(perm)
            if perm != list(range(len(imgs))):
                new_views = [orig_views[i] for i in perm]
                new_phrase = f"orthographic views ({', '.join(new_views)})"
                aug_media['images'] = [imgs[i] for i in perm]
                if aug_media.get('alphas') and len(aug_media['alphas']) == len(imgs):
                    aug_media['alphas'] = [aug_media['alphas'][i] for i in perm]
                aug_media['visual_units'] = [dict(units[i]) for i in perm]
                if aug_media.get('times') and len(aug_media['times']) == len(imgs):
                    aug_media['times'] = [aug_media['times'][i] for i in perm]
                aug_row['prompt'] = str(aug_row['prompt']).replace(old_phrase, new_phrase, 1)
                old_think_views = f"views={orig_views!r}"
                new_think_views = f"views={new_views!r}"
                if old_think_views in str(aug_row.get('answer', '')):
                    aug_row['answer'] = str(aug_row['answer']).replace(old_think_views, new_think_views, 1)
                imgs = aug_media['images']

    # 4. Subtle photometric jitter on 3D renders / temporal frames ONLY (never `uv_atlas`!)
    if (
        config.get('dynamic_photometric_jitter', True)
        and imgs
        and vkind in ('multiview_render', 'independent_image', 'state_pair', 'temporal_frame')
    ):
        gain = rng.uniform(0.96, 1.04)
        bias = rng.uniform(-3.0, 3.0)
        jittered = []
        for im in imgs:
            rgb_np = np.asarray(im.convert('RGB'), dtype=np.float32)
            out_np = np.clip(rgb_np * gain + bias, 0.0, 255.0).round().astype(np.uint8)
            jittered.append(Image.fromarray(out_np, mode='RGB'))
        aug_media['images'] = jittered

    return aug_row, aug_media
