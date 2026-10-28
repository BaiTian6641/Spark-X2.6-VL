"""Fixed internal holdouts, native inference and media ablations; no teacher labels."""
import contextlib, copy, math, re, time
from collections import defaultdict
from pathlib import Path
from .common import identity,write
from .data import prepare_media

def edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        out=[i]
        for j,y in enumerate(b,1): out.append(min(out[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=out
    return row[-1]

def error_rate(reference,hypothesis,language):
    # Explicit diagnostic normalization; keep original predictions as evidence.
    norm=lambda x: re.sub(r'[^\w\s]','',x.casefold())
    a,b=norm(reference),norm(hypothesis)
    a,b=(list(a.replace(' ','')),list(b.replace(' ',''))) if language=='zh' else (a.split(),b.split())
    return {'errors':edit_distance(a,b),'reference_units':len(a),'rate':edit_distance(a,b)/max(1,len(a))}

def evaluate(model,catalogue,reader,config,split,path,generate=False,ids=None):
    import numpy as np, soundfile as sf, torch
    path=Path(path); path.mkdir(parents=True,exist_ok=True); model.eval(); results=[]; started=time.monotonic()
    ids=ids or catalogue.evaluation_ids(split,config['evaluation_per_family_language'])
    counts=defaultdict(int)
    for rid in ids:
        row,x=catalogue.row(rid)
        if row['_split']!=split: raise ValueError('Protected evaluation split mismatch')
        with reader.packet(x) as get: media=prepare_media(row,get,config)
        seed=int(identity(['evaluation',rid])[:8],16)
        with torch.random.fork_rng(devices=[model.device.index or 0]),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            loss,units=model.loss(row,media)
            item={'id':rid,'family':row['_family'],'language':row['language'],'task':row['task'],'source_group':row['source_group'],'loss':float(loss),'units':units}
            if not math.isfinite(item['loss']): raise RuntimeError('Nonfinite evaluation loss')
            key=(row['_family'],row['language'],x['component']); counts[key]+=1
            if not row['_family'].endswith('_out') and (media['images'] or media['audio'] is not None):
                blank=copy.copy(media); blank['images']=[__import__('PIL.Image',fromlist=['Image']).new('RGB',im.size) for im in media['images']]
                blank['motion']=[__import__('PIL.Image',fromlist=['Image']).new('RGB',im.size) for im in media['motion']] if media['motion'] else None
                blank['audio']=(np.zeros_like(media['audio'][0]),media['audio'][1]) if media['audio'] is not None else None
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                item['ablated_loss']=float(model.loss(row,blank)[0])
                item['media_loss_advantage']=item['ablated_loss']-item['loss']
            if media['motion'] and len(media['images'])>1:
                reverse=copy.copy(media); reverse['images']=list(reversed(media['images'])); reverse['motion']=list(reversed(media['motion']))
                # PTS positions remain increasing while content order changes.
                torch.manual_seed(seed); item['reversal_loss']=float(model.loss(row,reverse)[0])
            if generate and counts[key]<=2:
                if row['_family'].endswith('_out'):
                    # The target waveform is cleared before inference. Duration
                    # comes from the explicit task duration, never codec targets.
                    inference=copy.copy(media); inference['target_audio']=None
                    target=row.get('target',{})
                    seconds=(target['num_frames']/target['sample_rate']) if target.get('num_frames') and target.get('sample_rate') else 15
                    wav,sr=model.generate_audio(row,inference,seconds=min(30,float(seconds)))
                    wav=np.asarray(wav)
                    if wav.ndim==2 and wav.shape[0]<=2: wav=wav.T
                    if not np.isfinite(wav).all(): raise RuntimeError('Nonfinite generated audio')
                    name=identity(rid)[:20]+'.wav'; sf.write(path/name,wav,sr,subtype='FLOAT')
                    item.update(audio_file=name,sample_rate=sr,peak=float(np.abs(wav).max()),human_listening='pending',native_generation=True)
                    # Bridge ablation uses the same random seed and no target audio.
                    prefix=model.hidden_states(model.prefix(row,inference)); zeros=torch.zeros_like(prefix)
                    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                    if row['_family']=='speech_out': detached=model.speech_out.generate(zeros,row)[0]
                    else: detached=model.sound_out.generate(zeros,seconds=min(30,float(seconds)))[0]
                    detached=np.asarray(detached)
                    if detached.ndim==2 and detached.shape[0]<=2: detached=detached.T
                    sf.write(path/(identity(rid)[:20]+'-zero-bridge.wav'),detached,sr,subtype='FLOAT')
                    item['zero_bridge_comparison']='saved for prompt-match/listening review; waveform difference alone is not competence'
                else:
                    prediction=model.answer(row,media,config.get('evaluation_max_new_tokens')); item.update(prediction=prediction,reference=row['answer'])
                    if row['task'] in ('speech_recognition','document_ocr','region_ocr'):
                        item['edit_error']=error_rate(row['answer'],prediction,row['language'])
                    elif row['task'] in ('real_count','capture_elapsed','capture_clock_calibration','source_capture_elapsed','temporal_order','occlusion'):
                        item['exact_match']=prediction.strip()==row['answer'].strip()
        results.append(item)
    grouped=defaultdict(list)
    for r in results: grouped[r['family']+':'+r['language']].append(r)
    report={'schema_version':1,'split':split,'membership_sha256':identity(ids),'rows':results,'seconds':time.monotonic()-started,
            'by_stratum':{k:{'records':len(v),'mean_loss':sum(x['loss'] for x in v)/len(v),'mean_media_advantage':sum(x.get('media_loss_advantage',0) for x in v)/len(v)} for k,v in grouped.items()},
            'human_audio_review':'pending' if generate else 'not performed','broad_multimodal_capability_passed':False,
            'scope':'Fixed internal source-held-out losses; limited native samples and ablations. Cross-family losses have different units and are not averaged together. Not a public benchmark or human audio acceptance.'}
    write(path/'REPORT.json',report); return report

def promotion(parent,candidate,config):
    if parent['membership_sha256']!=candidate['membership_sha256']: raise ValueError('Promotion membership changed')
    reasons=[]
    for key,base in parent['by_stratum'].items():
        child=candidate['by_stratum'].get(key)
        if not child or child['mean_loss']>base['mean_loss']*(1+config['promotion_max_relative_loss_regression']): reasons.append(key+' loss regression')
    improved=[key for key,b in parent['by_stratum'].items() if key in candidate['by_stratum'] and candidate['by_stratum'][key]['mean_loss']<b['mean_loss']]
    if not improved: reasons.append('no improved stratum')
    return {'accepted':not reasons,'reasons':reasons,'improved_strata':improved,'scope':'Automated internal loss gate only; human audio review and broader benchmarks remain separate'}
