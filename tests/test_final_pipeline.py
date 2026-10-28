"""Durability, split, role and curriculum contracts for the formal runner."""
import gzip,json,sqlite3
from pathlib import Path
import pytest
from colab.final.common import ComputeBudget,read,identity
from colab.final.data import Catalogue,CommittedCursor,build_database,conditioning_members,family,training_views
from colab.final.storage import manifest

CONFIG=read('colab/final/campaign.json')

def test_optimizer_commit_rewinds_prefetched_examples():
    c=CommittedCursor(list('abcde')); assert [c.issue() for _ in range(5)]==list('abcde')
    c.commit(2); saved=c.state(); assert saved['committed']==2
    restored=CommittedCursor(list('abcde'),saved); assert restored.issue()=='c'
    with pytest.raises(ValueError): CommittedCursor(list('bcdea'),saved)


def test_output_targets_never_condition_recognition_or_generation():
    row={'task':'speech_synthesis','audio':['voice.flac'],'audio_direction':'text_to_audio','media_role':'target','target':{'kind':'waveform','path':'voice.flac'}}
    assert family(row)=='speech_out'; assert conditioning_members(row)==[]
    bad={**row,'media_role':'input'}
    with pytest.raises(ValueError): family(bad)
    row.update(task='sound_generation'); assert family(row)=='sound_out'


def test_complete_score_views_preserve_every_event_and_full_audio():
    notes=[{'clip_start_seconds':i/10,'clip_end_seconds':i/10+.05,'gm_program_1based':41,'midi_note':60+i%12} for i in range(61)]
    row={'id':'notes','task':'music_instrument_note_understanding','language':'en','audio':['music.flac'],'answer':json.dumps({'active_instruments':[{'gm_program_1based':41}],'score_aligned_note_events':notes})}
    views=training_views(row); assert len(views)==3
    assert sum(len(json.loads(v['answer'])['notes']) for v in views)==61
    assert all(v['audio']==['music.flac'] and v['_derived_task']['waveform_crop'] is False for v in views)
    assert json.loads(views[-1]['answer'])['notes'][-1][0]==6.0


def test_shard_local_interleaving_does_not_thrash_same_source(tmp_path):
    path=tmp_path/'corpus.sqlite';c=sqlite3.connect(path)
    c.executescript('CREATE TABLE packets(pid INTEGER,component TEXT,shard TEXT); CREATE TABLE rows(rid TEXT,pid INTEGER,family TEXT,language TEXT,split TEXT);')
    for component in ('a','b'):
        for shard in range(3):
            pid=len(c.execute('select * from packets').fetchall())+1
            c.execute('INSERT INTO packets VALUES(?,?,?)',(pid,component,str(shard)))
            c.executemany('INSERT INTO rows VALUES(?,?,?,?,?)',[(f'{component}{shard}-{i}',pid,'image','en','train') for i in range(300)])
    c.commit(); c.close(); order=Catalogue(path).plan({'families':['image'],'epochs':1},123)
    assert len(order)==1800 and len(set(order))==1800
    for component in ('a','b'):
        sequence=[r[1] for r in order if r.startswith(component)]
        transitions=[s for i,s in enumerate(sequence) if i==0 or sequence[i-1]!=s]
        assert len(transitions)==3 and len(set(transitions))==3


def test_shared_article_and_task_variants_stay_in_one_split(tmp_path):
    path=tmp_path/'packets.gz'
    with gzip.open(path,'wt') as f:
        for i in range(40):
            for modality in ('image','text'):
                row={'id':f'{modality}-{i}','source_group':f'wikipedia-en:{i}','source_id':f'wikipedia-en:{i}','language':'en','task':'document_ocr' if modality=='image' else 'text_replay','frames':['page.png'] if modality=='image' else [],'audio':[],'video':[],'prompt':'Read','answer':'Content','metadata':{}}
                f.write(json.dumps({'component':modality,'transport':None,'record':row})+'\n')
    db=tmp_path/'c.sqlite'; report=build_database(path,db,CONFIG); c=sqlite3.connect(db)
    assert report['original_instructions']==80
    assert c.execute('SELECT count(*) FROM (SELECT group_id FROM rows GROUP BY group_id HAVING count(distinct split)>1)').fetchone()[0]==0
    assert {r[0] for r in c.execute('select distinct split from rows')}=={'train','dev','promotion','audit'}


def test_missing_or_insufficient_compute_never_counts_as_completion():
    b=ComputeBudget(CONFIG)
    with pytest.raises(RuntimeError): b.check()
    b.observe(50,8.71)
    with pytest.raises(RuntimeError): b.check(100*3600)
    assert b.check(0)<50


def test_checkpoint_continuation_requires_new_process_but_not_new_vm():
    import subprocess,sys
    from colab.final.qualify import process_identity,require_new_process
    current=process_identity()
    with pytest.raises(RuntimeError,match='new campaign subprocess'):
        require_new_process({'qualification_process':current})
    with pytest.raises(RuntimeError):require_new_process({})
    code='import json,sys; from colab.final.qualify import require_new_process; print(json.dumps(require_new_process(json.loads(sys.argv[1]))))'
    output=subprocess.check_output([sys.executable,'-c',code,json.dumps({'qualification_process':current})],text=True)
    restored=json.loads(output)
    assert restored['boot_id']==current['boot_id'] and restored['pid']!=current['pid']


def test_closed_artifact_rejects_secret_paths(tmp_path):
    (tmp_path/'credential.json').write_text('{}')
    with pytest.raises(ValueError): manifest(tmp_path)


def test_teacher_decisions_never_create_new_ground_truth():
    from colab.final.teacher import replay_orders,eligibility
    selected,control=replay_orders([('a',{'training_action':'replay_original','priority':3}),('b',{'training_action':'replay_original','priority':0})],3,8)
    assert len(selected)==len(control)==16 and set(selected)<=set(control)=={'a','b'}
    with pytest.raises(RuntimeError): eligibility({**CONFIG,'instructor_enabled':False})

def test_family_language_objective_balances_supervision_without_dropping_rows():
    from colab.final.engine import loss_weights
    class SourceCatalogue:
        def select(self,split):
            return [(f'en-{i}','image','en','','') for i in range(100)]+[(f'zh-{i}','image','zh','','') for i in range(10)]+[(f't-{i}','text','en','','') for i in range(20)]
    rows=SourceCatalogue().select('train');order=[r[0] for r in rows]
    weights=loss_weights(SourceCatalogue(),order,CONFIG)
    assert len(order)==130
    assert 100*weights['image:en']==pytest.approx(10*weights['image:zh'])
    assert 100*weights['image:en']==pytest.approx(20*weights['text:en'])
    assert max(weights.values())<=CONFIG['max_loss_weight']

def test_dataset_format_peeks_and_prefetch_do_not_skip_training_records():
    from colab.final.data import cursor_rows
    cursor=CommittedCursor(['a','b','c','d'])
    assert next(cursor_rows(cursor.order,cursor))['rid']=='a'
    assert next(cursor_rows(cursor.order,cursor))['rid']=='a'
    assert cursor.issued==cursor.committed==0
    iterator=cursor_rows(cursor.order,cursor);prefetched=[next(iterator) for _ in range(3)]
    assert cursor.issued==0
    for item in prefetched[:2]:cursor.record_computed(item['_index'])
    cursor.commit(2);restored=CommittedCursor(cursor.order,cursor.state())
    assert next(cursor_rows(restored.order,restored))=={'rid':'c','_index':2}
