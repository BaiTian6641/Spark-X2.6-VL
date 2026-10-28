"""Loss-masked multimodal batching; target/provenance fields never become inputs."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from PIL import Image


def clock_features(media_times_s, capture_times_s=None):
    if not media_times_s or not all(math.isfinite(float(t)) for t in media_times_s):
        raise ValueError('Media timestamps must be finite and nonempty')
    # Deltas preserve duration without exposing an arbitrary absolute acquisition epoch.
    media = [float(t)-float(media_times_s[0]) for t in media_times_s]
    known = capture_times_s is not None and all(t is not None for t in capture_times_s)
    if known and len(capture_times_s) != len(media):
        raise ValueError('Capture and presentation timestamps must align')
    capture = [float(t)-float(capture_times_s[0]) for t in capture_times_s] if known else [0.] * len(media)
    if not all(math.isfinite(t) for t in capture):
        raise ValueError('Capture timestamps must be finite when provided')
    slog = lambda t: math.copysign(math.log1p(abs(t)), t)
    return [[slog(m), math.sin(m), math.cos(m), float(known),
             slog(c), math.sin(c), math.cos(c), i/max(1,len(media)-1)]
            for i,(m,c) in enumerate(zip(media,capture))]


class MultimodalCollator:
    def __init__(self, tokenizer, image_processor, data_root, tokens_per_frame=16, max_length=512):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data_root = Path(data_root).resolve()
        self.tokens_per_frame = int(tokens_per_frame)
        self.max_length = int(max_length)

    def tokenize_record(self, record):
        # Explicit allow-list: no filenames, IDs, metadata, class labels or ground truth in the prompt.
        prompt = 'User: '+record['prompt']+'\nAssistant: '
        before = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
        target = self.tokenizer(record['answer'], add_special_tokens=False)['input_ids']
        target = target + [self.tokenizer.eos_token_id]
        if not target:
            raise ValueError('An example must contain answer targets')
        return before, target

    def dataset_record(self, record):
        prompt, answer = self.tokenize_record(record)
        n = len(record['frames']) * self.tokens_per_frame
        ids = [self.tokenizer.pad_token_id]*n + prompt + answer
        labels = [-100]*(n+len(prompt)) + answer
        if len(ids) > self.max_length:
            raise ValueError(f"Example {record['id']} has {len(ids)} positions, exceeding {self.max_length}; no silent truncation")
        return {'input_ids':ids,'attention_mask':[1]*len(ids),'labels':labels,
                'frames':record['frames'],'media_times_s':record['media_times_s'],
                'capture_times_s':record.get('capture_times_s')}

    def _frame(self, filename):
        path = (self.data_root/filename).resolve()
        if self.data_root not in path.parents:
            raise ValueError('Media path escapes the selected dataset')
        with Image.open(path) as image:
            return image.convert('RGB')

    def __call__(self, records):
        if not records:
            raise ValueError('Empty multimodal batch')
        max_frames = max(len(r['frames']) for r in records)
        if len({len(r['frames']) for r in records}) != 1:
            raise ValueError('S0 requires equal frame counts per batch; bucket examples before batching to preserve Spark RoPE distances')
        rows=[]; all_pixels=[]; masks=[]; clocks=[]
        for r in records:
            f=len(r['frames']); old_prefix=f*self.tokens_per_frame
            new_prefix=max_frames*self.tokens_per_frame
            ids=[self.tokenizer.pad_token_id]*new_prefix + r['input_ids'][old_prefix:]
            labels=[-100]*new_prefix + r['labels'][old_prefix:]
            attention=[1]*old_prefix + [0]*(new_prefix-old_prefix) + r['attention_mask'][old_prefix:]
            rows.append((ids,labels,attention))
            px=self.image_processor(images=[self._frame(p) for p in r['frames']],return_tensors='pt')['pixel_values']
            if f<max_frames:
                px=torch.cat((px,torch.zeros((max_frames-f,*px.shape[1:]),dtype=px.dtype)),dim=0)
            all_pixels.append(px)
            masks.append([1]*f+[0]*(max_frames-f))
            clocks.append(clock_features(r['media_times_s'],r.get('capture_times_s'))+[[0.]*8]*(max_frames-f))
        length=max(len(r[0]) for r in rows)
        if length>self.max_length:
            raise ValueError('Padded frame prefix exceeds the sequence budget')
        ids=[];labels=[];attention=[]
        for a,b,c in rows:
            pad=length-len(a)
            ids.append(a+[self.tokenizer.pad_token_id]*pad)
            labels.append(b+[-100]*pad)
            attention.append(c+[0]*pad)
        return {'input_ids':torch.tensor(ids,dtype=torch.long),
                'labels':torch.tensor(labels,dtype=torch.long),
                'attention_mask':torch.tensor(attention,dtype=torch.long),
                'pixel_values':torch.stack(all_pixels),
                'frame_mask':torch.tensor(masks,dtype=torch.bool),
                'clock_features':torch.tensor(clocks,dtype=torch.float32)}
