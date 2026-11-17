"""Fixed internal holdouts, pure-text parity, blank/shuffled visual ablations, and SyncCraft oracle checks."""
from __future__ import annotations
import contextlib, copy, math, re, time
from collections import defaultdict
from pathlib import Path
from typing import Any
from .common import identity,write
from .data import prepare_media
from .inference import empty_media

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


def verify_text_only_parity(
    model,
    reference_spark,
    prompts: list[str],
    *,
    enable_thinking_modes: tuple[bool, ...] = (False, True),
    max_new_tokens: int = 8,
    atol: float = 0.0,
) -> dict[str, Any]:
    """Verify exact text-only logit & generation parity against standalone `Spark`."""
    import torch
    model.eval()
    reference_spark.eval()
    max_logit_diff = 0.0
    checked = 0
    for prompt in prompts:
        for thinking in enable_thinking_modes:
            row = {
                '_family': 'text',
                'language': 'en',
                'task': 'text_replay',
                'prompt': prompt,
                'enable_thinking': thinking,
            }
            pre, post = model.prompt_ids_for_row(row, for_generation=thinking)
            ids = torch.tensor(pre + post, device=model.device, dtype=torch.long)[None]
            with torch.no_grad():
                wrapped_logits = model.forward_text_only(ids).logits.float()
                ref_logits = reference_spark(input_ids=ids).logits.float()
                diff = float((wrapped_logits - ref_logits).abs().max().item())
                max_logit_diff = max(max_logit_diff, diff)
                if diff > atol:
                    raise RuntimeError(
                        f'Text-only logit parity failed (thinking={thinking}): max_abs_diff={diff} > atol={atol}'
                    )
                wrapped_ans = model.answer(row, empty_media(), max_new_tokens=max_new_tokens, enable_thinking=thinking)
                # Reference greedy decode over exact same prompt_ids
                past = None
                cur = ids
                gen_ids: list[int] = []
                eos_id = model.tokenizer.eos_token_id
                for _ in range(max_new_tokens):
                    out = reference_spark.model(input_ids=cur, past_key_values=past, use_cache=True, return_dict=True)
                    past = out.past_key_values
                    tok = reference_spark.get_output_embeddings()(out.last_hidden_state[:, -1:]).float().argmax(dim=-1)
                    if eos_id is not None and int(tok.item()) == eos_id:
                        break
                    gen_ids.append(int(tok.item()))
                    cur = tok
                ref_ans = model.chat.restore_thinking_prefix(
                    model.tokenizer.decode(gen_ids, skip_special_tokens=True),
                    model.chat.layouts[True],
                    thinking,
                )
                if wrapped_ans != ref_ans:
                    raise RuntimeError(
                        f'Text-only greedy generation parity failed (thinking={thinking}): {wrapped_ans!r} != {ref_ans!r}'
                    )
            checked += 1
    return {
        'passed': True,
        'cases_checked': checked,
        'max_abs_logit_diff': max_logit_diff,
        'exact_generation_match': True,
    }


def evaluate(model,catalogue,reader,config,split,path,generate=False,ids=None):
    import numpy as np, torch
    from PIL import Image
    path=Path(path); path.mkdir(parents=True,exist_ok=True); model.eval(); results=[]; started=time.monotonic()
    modalities=config.get('modalities','omni')
    active_families=('image','video','text') if modalities=='vision_text' else None
    if ids is None:
        all_ids=catalogue.evaluation_ids(split,config['evaluation_per_family_language'])
        if active_families is not None:
            allowed_rids={r[0] for r in catalogue.select(split,active_families)}
            ids=[rid for rid in all_ids if rid in allowed_rids]
        else:
            ids=all_ids
    counts=defaultdict(int)

    # Pre-collect a deterministic donor image pool for shuffled-image cross-record control.
    donor_media_by_id:dict[str,dict[str,Any]]={}
    image_eval_ids=[]
    for rid in ids:
        row,x=catalogue.row(rid)
        if row['_family']=='image' and len(image_eval_ids)<8:
            with reader.packet(x) as get: donor_media_by_id[rid]=prepare_media(row,get,config)
            if donor_media_by_id[rid]['images']:
                image_eval_ids.append(rid)

    is_cuda=str(model.device).startswith('cuda') and torch.cuda.is_available()
    fork_devices=[model.device.index or 0] if is_cuda else []
    amp_ctx=(lambda: torch.autocast('cuda',dtype=torch.bfloat16)) if is_cuda else contextlib.nullcontext

    synccraft_evals:list[dict[str,Any]]=[]

    for rid in ids:
        row,x=catalogue.row(rid)
        if row['_split']!=split: raise ValueError('Protected evaluation split mismatch')
        media=donor_media_by_id[rid] if rid in donor_media_by_id else None
        if media is None:
            with reader.packet(x) as get: media=prepare_media(row,get,config)
        seed=int(identity(['evaluation',rid])[:8],16)
        with torch.random.fork_rng(devices=fork_devices),torch.no_grad(),amp_ctx():
            torch.manual_seed(seed)
            if is_cuda: torch.cuda.manual_seed_all(seed)
            loss,units=model.loss(row,media)
            item={'id':rid,'family':row['_family'],'language':row['language'],'task':row['task'],'source_group':row['source_group'],'loss':float(loss),'units':units}
            if not math.isfinite(item['loss']): raise RuntimeError('Nonfinite evaluation loss')
            key=(row['_family'],row['language'],x['component']); counts[key]+=1

            # 1. Blank-media ablation control
            if not row['_family'].endswith('_out') and (media['images'] or media['audio'] is not None):
                blank=copy.copy(media)
                blank['images']=[Image.new('RGB',im.size) for im in media['images']]
                blank['alphas']=[None]*len(media['images'])
                blank['motion']=[Image.new('RGB',im.size) for im in media['motion']] if media['motion'] else None
                blank['audio']=(np.zeros_like(media['audio'][0]),media['audio'][1]) if media['audio'] is not None else None
                torch.manual_seed(seed)
                if is_cuda: torch.cuda.manual_seed_all(seed)
                item['ablated_loss']=float(model.loss(row,blank)[0])
                item['media_loss_advantage']=item['ablated_loss']-item['loss']

            # 2. Shuffled-image / permuted-view control (tests genuine visual & camera/UV/state grounding)
            if not row['_family'].endswith('_out') and media['images']:
                shuffled=copy.copy(media)
                if len(media['images'])>1:
                    # Cyclic shift of images & alphas while keeping camera/view_id/timestamp slots fixed!
                    shuffled['images']=media['images'][1:]+media['images'][:1]
                    if media.get('alphas'):
                        shuffled['alphas']=media['alphas'][1:]+media['alphas'][:1]
                    if media.get('motion'):
                        shuffled['motion']=list(reversed(media['motion']))
                elif len(image_eval_ids)>1:
                    donor_rid=next((d for d in image_eval_ids if d!=rid),None)
                    if donor_rid is not None:
                        donor=donor_media_by_id[donor_rid]
                        shuffled['images']=[donor['images'][0]]
                        shuffled['alphas']=[donor['alphas'][0]] if donor.get('alphas') else [None]
                if shuffled['images'] is not media['images']:
                    torch.manual_seed(seed)
                    if is_cuda: torch.cuda.manual_seed_all(seed)
                    item['shuffled_loss']=float(model.loss(row,shuffled)[0])
                    item['shuffled_loss_advantage']=item['shuffled_loss']-item['loss']

            if media['motion'] and len(media['images'])>1:
                reverse=copy.copy(media); reverse['images']=list(reversed(media['images'])); reverse['motion']=list(reversed(media['motion']))
                torch.manual_seed(seed)
                if is_cuda: torch.cuda.manual_seed_all(seed)
                item['reversal_loss']=float(model.loss(row,reverse)[0])

            is_synccraft=str(row.get('task','')).startswith('synccraft_')
            if generate and (counts[key]<=2 or is_synccraft):
                if row['_family'].endswith('_out'):
                    import soundfile as sf
                    inference=copy.copy(media); inference['target_audio']=None
                    target=row.get('target',{})
                    seconds=(target['num_frames']/target['sample_rate']) if target.get('num_frames') and target.get('sample_rate') else 15
                    wav,sr=model.generate_audio(row,inference,seconds=min(30,float(seconds)))
                    wav=np.asarray(wav)
                    if wav.ndim==2 and wav.shape[0]<=2: wav=wav.T
                    if not np.isfinite(wav).all(): raise RuntimeError('Nonfinite generated audio')
                    name=identity(rid)[:20]+'.wav'; sf.write(path/name,wav,sr,subtype='FLOAT')
                    item.update(audio_file=name,sample_rate=sr,peak=float(np.abs(wav).max()),human_listening='pending',native_generation=True)
                    prefix=model.hidden_states(model.prefix(row,inference)); zeros=torch.zeros_like(prefix)
                    torch.manual_seed(seed)
                    if is_cuda: torch.cuda.manual_seed_all(seed)
                    if row['_family']=='speech_out': detached=model.speech_out.generate(zeros,row)[0]
                    else: detached=model.sound_out.generate(zeros,seconds=min(30,float(seconds)))[0]
                    detached=np.asarray(detached)
                    if detached.ndim==2 and detached.shape[0]<=2: detached=detached.T
                    sf.write(path/(identity(rid)[:20]+'-zero-bridge.wav'),detached,sr,subtype='FLOAT')
                    item['zero_bridge_comparison']='saved for prompt-match/listening review; waveform difference alone is not competence'
                else:
                    cfg_max=config.get('evaluation_max_new_tokens')
                    eff_eval_max=(max(int(cfg_max),1536) if (is_synccraft and cfg_max is not None and int(cfg_max)>=128) else cfg_max)
                    prediction=model.answer(row,media,eff_eval_max)
                    item.update(prediction=prediction,reference=row['answer'])
                    if is_synccraft:
                        from .synccraft_bridge import evaluate_synccraft_completion
                        sc_eval=evaluate_synccraft_completion(row,prediction)
                        item['synccraft_eval']=sc_eval
                        synccraft_evals.append(sc_eval)
                    elif row['task'] in ('speech_recognition','document_ocr','region_ocr'):
                        item['edit_error']=error_rate(row['answer'],prediction,row['language'])
                    elif row['task'] in ('real_count','capture_elapsed','capture_clock_calibration','source_capture_elapsed','temporal_order','occlusion'):
                        item['exact_match']=prediction.strip()==row['answer'].strip()
        results.append(item)

    grouped=defaultdict(list)
    for r in results: grouped[r['family']+':'+r['language']].append(r)
    by_stratum={}
    for k,v in grouped.items():
        shuf_vals=[x['shuffled_loss_advantage'] for x in v if 'shuffled_loss_advantage' in x]
        by_stratum[k]={
            'records':len(v),
            'mean_loss':sum(x['loss'] for x in v)/len(v),
            'mean_media_advantage':sum(x.get('media_loss_advantage',0) for x in v)/len(v),
            'mean_shuffled_advantage':(sum(shuf_vals)/len(shuf_vals)) if shuf_vals else 0.0,
        }

    synccraft_summary=None
    if synccraft_evals:
        ious=[x['silhouette_iou'] for x in synccraft_evals if 'silhouette_iou' in x]
        sat_items=[x for x in synccraft_evals if 'sat_passed' in x]
        synccraft_summary={
            'evaluated_records':len(synccraft_evals),
            'parse_valid_rate':sum(1 for x in synccraft_evals if x.get('parse_valid'))/len(synccraft_evals),
            'oracle_pass_rate':sum(1 for x in synccraft_evals if x.get('passed'))/len(synccraft_evals),
            'sat_pass_rate':(sum(1 for x in sat_items if x.get('sat_passed'))/len(sat_items)) if sat_items else None,
            'mean_silhouette_iou':(sum(ious)/len(ious)) if ious else None,
        }

    report={
        'schema_version':1,
        'split':split,
        'membership_sha256':identity(ids),
        'rows':results,
        'seconds':time.monotonic()-started,
        'by_stratum':by_stratum,
        'synccraft_summary':synccraft_summary,
        'human_audio_review':('not_applicable_vision_text' if modalities=='vision_text' else ('pending' if generate else 'not performed')),
        'broad_multimodal_capability_passed':False,
        'scope':'Fixed internal source-held-out losses, blank-image and shuffled-image controls, and SyncCraft compiler/SAT/IoU oracle checks. Cross-family losses have different units and are not averaged together.',
    }
    write(path/'REPORT.json',report); return report

def promotion(parent,candidate,config):
    if parent['membership_sha256']!=candidate['membership_sha256']: raise ValueError('Promotion membership changed')
    reasons=[]
    abs_tol=float(config.get('promotion_min_abs_loss_tolerance',0.0))
    for key,base in parent['by_stratum'].items():
        child=candidate['by_stratum'].get(key)
        allowed=base['mean_loss']*(1+config['promotion_max_relative_loss_regression'])+abs_tol
        if not child or child['mean_loss']>allowed: reasons.append(key+' loss regression')
    improved=[key for key,b in parent['by_stratum'].items() if key in candidate['by_stratum'] and candidate['by_stratum'][key]['mean_loss']<b['mean_loss']]
    if not improved: reasons.append('no improved stratum')
    return {'accepted':not reasons,'reasons':reasons,'improved_strata':improved,'scope':'Automated internal loss gate only; human audio review and broader benchmarks remain separate'}
