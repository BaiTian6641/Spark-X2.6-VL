"""Native differentiable multimodal Spark wrapper. No external generation API.

The Spark decoder is called directly: this custom backbone does not return
hidden_states through its CausalLM wrapper. Targets never enter input encoders.
"""
from __future__ import annotations
import math, sys, zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .common import HERE, read, sha

def install_spark_sdpa(spark):
    """Replace only the reference attention kernel, keeping masks/RoPE/gates.

    Spark's original class does not advertise the Transformers `sdpa` loader
    flag. Load its declared eager interface, then install this tested kernel
    adapter. The original immutable source files are never modified.
    """
    module=sys.modules[type(spark).__module__]
    current=module.eager_attention_forward
    if getattr(current,'_spark_native_sdpa',False): return
    reference=current
    def sdpa(attention,query,key,value,attention_mask=None,scaling=None,dropout=0.,**kwargs):
        key=module.repeat_kv(key,attention.num_key_value_groups)
        value=module.repeat_kv(value,attention.num_key_value_groups)
        mask=attention_mask[:,:,:,:key.shape[-2]] if attention_mask is not None else None
        result=F.scaled_dot_product_attention(query,key,value,attn_mask=mask,dropout_p=dropout if attention.training else 0.,is_causal=False,scale=scaling)
        return result,None
    sdpa._spark_native_sdpa=True; sdpa._spark_reference=reference
    module.eager_attention_forward=sdpa

class Resampler(nn.Module):
    def __init__(self,dim,hidden,n,heads=8):
        super().__init__(); self.input=nn.Linear(dim,hidden); self.queries=nn.Parameter(torch.randn(n,hidden)*.02)
        self.norm=nn.LayerNorm(hidden); self.attn=nn.MultiheadAttention(hidden,heads,batch_first=True); self.out=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden))
    def forward(self,x,position=None):
        x=self.input(x)
        if position is not None: x=x+position.to(x.dtype)
        q=self.queries[None].expand(x.shape[0],-1,-1).to(x.dtype)
        return self.out(q+self.attn(q,self.norm(x),self.norm(x),need_weights=False)[0])

def spatial_downshuffle(x,h,w):
    """Preserve the actual NaFlex grid; concatenate each real 2x2 neighborhood."""
    if h*w>x.shape[1] or h<1 or w<1: raise ValueError('Invalid native patch grid')
    x=x[:,:h*w].reshape(x.shape[0],h,w,x.shape[-1]); valid=torch.ones((h,w),device=x.device,dtype=torch.bool)
    if h%2: x=F.pad(x,(0,0,0,0,0,1)); valid=F.pad(valid,(0,0,0,1))
    if w%2: x=F.pad(x,(0,0,0,1)); valid=F.pad(valid,(0,1))
    hh,ww=x.shape[1:3]
    out=x.reshape(x.shape[0],hh//2,2,ww//2,2,x.shape[-1]).permute(0,1,3,2,4,5).flatten(3).flatten(1,2)
    mask=valid.reshape(hh//2,2,ww//2,2).permute(0,2,1,3).flatten(2).any(-1).flatten()
    return out[:,mask],mask

def audio_resample(wav,sr,target):
    import scipy.signal
    if sr==target: return wav
    g=math.gcd(sr,target); return scipy.signal.resample_poly(wav,target//g,sr//g,axis=-1).astype('float32')

class SparkNative(nn.Module):
    main_input_name='input_ids'
    supports_gradient_checkpointing=True
    def __init__(self,paths,policy,device='cuda',pretrained=True):
        super().__init__()
        from transformers import (AutoConfig,AutoModelForCausalLM,AutoTokenizer,Siglip2VisionModel,Siglip2VisionConfig,AutoImageProcessor,WhisperModel,WhisperConfig,WhisperFeatureExtractor,ClapAudioModel,ClapAudioConfig,ClapFeatureExtractor)
        self.policy=policy; self.paths={k:str(v) for k,v in paths.items()}; self.pretrained=pretrained
        self.spark=(AutoModelForCausalLM.from_pretrained(paths['spark'],trust_remote_code=True,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager') if pretrained else AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(paths['spark'],trust_remote_code=True,local_files_only=True),trust_remote_code=True,attn_implementation='eager'))
        install_spark_sdpa(self.spark)
        self.tokenizer=AutoTokenizer.from_pretrained(paths['spark'],trust_remote_code=True,local_files_only=True)
        if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token=self.tokenizer.eos_token
        self.config=self.spark.config; self.config.use_cache=False; self.generation_config=self.spark.generation_config
        self.vision=(Siglip2VisionModel.from_pretrained(paths['vision'],local_files_only=True,torch_dtype=torch.float32) if pretrained else Siglip2VisionModel(Siglip2VisionConfig.from_pretrained(paths['vision'],local_files_only=True)))
        # Pooling heads are not part of the dense patch interface.
        if hasattr(self.vision.vision_model,'head'): del self.vision.vision_model.head
        self.vision.vision_model.use_head=False
        self.image_processor=AutoImageProcessor.from_pretrained(paths['vision'],local_files_only=True)
        whisper=(WhisperModel.from_pretrained(paths['speech_in'],local_files_only=True,torch_dtype=torch.float32) if pretrained else WhisperModel(WhisperConfig.from_pretrained(paths['speech_in'],local_files_only=True)))
        self.speech_in=whisper.encoder; del whisper
        self.speech_processor=WhisperFeatureExtractor.from_pretrained(paths['speech_in'],local_files_only=True)
        self.sound_in=(ClapAudioModel.from_pretrained(paths['sound_in'],local_files_only=True,torch_dtype=torch.float32) if pretrained else ClapAudioModel(ClapAudioConfig.from_pretrained(paths['sound_in'],local_files_only=True)))
        self.sound_processor=ClapFeatureExtractor.from_pretrained(paths['sound_in'],local_files_only=True)
        self.motion=self.load_motion(paths['motion_source'],paths['motion'] if pretrained else None)
        h=self.config.hidden_size; self.hidden=h
        self.vision_projector=nn.Sequential(nn.LayerNorm(self.vision.config.hidden_size*4),nn.Linear(self.vision.config.hidden_size*4,h),nn.GELU(),nn.Linear(h,h))
        self.spatial_clock=nn.Sequential(nn.Linear(12,h),nn.GELU(),nn.Linear(h,h))
        self.motion_projector=Resampler(self.motion.embed_dim,h,policy['motion_tokens'])
        self.speech_projector=Resampler(self.speech_in.config.d_model,h,policy['audio_tokens'])
        self.sound_projector=Resampler(self.sound_in.config.hidden_size,h,policy['audio_tokens'])
        self.modality=nn.Embedding(5,h)
        self.audio_clock=nn.Sequential(nn.Linear(4,h),nn.GELU(),nn.Linear(h,h))
        from .outputs import NativeSpeechOutput,NativeSoundOutput
        self.speech_out=NativeSpeechOutput(paths['speech_out'],h,policy,pretrained=pretrained)
        self.sound_out=NativeSoundOutput(paths['sound_out'],h,policy,pretrained=pretrained)
        self.to(device=device,dtype=torch.float32); self.freeze_infrastructure(); self.gradient_checkpointing_enable()
    @staticmethod
    def load_motion(source,checkpoint):
        sys.path.insert(0,str(source))
        # Direct construction avoids the upstream localhost pretrained URL and
        # never allocates an unused predictor/teacher network.
        from app.vjepa_2_1.models.vision_transformer import vit_base
        motion=vit_base(patch_size=16,img_size=(384,384),num_frames=64,tubelet_size=2,use_sdpa=True,use_SiLU=False,wide_SiLU=True,uniform_power=False,use_rope=True,img_temporal_dim_size=1,interpolate_rope=True)
        if checkpoint is not None:
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            if 'ema_encoder' not in state: raise ValueError('Wrong V-JEPA checkpoint key')
            clean={k.replace('module.','').replace('backbone.',''):v for k,v in state['ema_encoder'].items()}
            motion.load_state_dict(clean,strict=True); del state,clean
        # The image-only stem is absent from our fixed 16-frame motion route;
        # still-image features come from SigLIP2. Remove unused source parameters.
        motion.patch_embed_img=None; motion.img_temporal_dim_size=None
        if hasattr(motion,'img_mod_embed'): del motion.img_mod_embed
        if hasattr(motion,'use_activation_checkpointing'): motion.use_activation_checkpointing=True
        return motion
    @property
    def device(self): return next(self.spark.parameters()).device
    def get_input_embeddings(self): return self.spark.get_input_embeddings()
    def get_output_embeddings(self): return self.spark.get_output_embeddings()
    def gradient_checkpointing_enable(self,gradient_checkpointing_kwargs=None):
        opts=gradient_checkpointing_kwargs or {'use_reentrant':False}
        for module in (self.spark,self.vision,self.speech_in,self.sound_in,self.speech_out.talker):
            if getattr(module,'supports_gradient_checkpointing',False): module.gradient_checkpointing_enable(gradient_checkpointing_kwargs=opts)
        if hasattr(self.sound_out.dit,'enable_gradient_checkpointing'): self.sound_out.dit.enable_gradient_checkpointing()
    def freeze_infrastructure(self):
        self.speech_out.freeze_codec(); self.sound_out.vae.requires_grad_(False)
    def train(self,mode=True):
        super().train(mode); self.speech_out.codec_model.eval(); self.sound_out.vae.eval(); return self
    def set_stage(self,alignment_only=False):
        self.requires_grad_(True); self.freeze_infrastructure()
        if alignment_only:
            for name in ('spark','vision','motion','speech_in','sound_in'):
                getattr(self,name).requires_grad_(False)
            self.speech_out.talker.requires_grad_(False); self.sound_out.dit.requires_grad_(False)
        if not alignment_only and any(not p.requires_grad for p in self.spark.parameters()): raise AssertionError('Full Spark is not trainable')
    def named_groups(self):
        for name,p in self.named_parameters():
            prefix=name.split('.')[0]
            if prefix in ('vision_projector','spatial_clock','motion_projector','speech_projector','sound_projector','modality','audio_clock'): group='connectors'
            elif prefix=='speech_out': group='speech_output'
            elif prefix=='sound_out': group='sound_output'
            else: group=prefix
            yield group,name,p
    def tile_views(self,image,video):
        w,h=image.size; views=[(image,(0.,0.,1.,1.),self.policy['thumbnail_patches'])]
        if not video and self.policy['detail_tiles']:
            if self.policy['detail_tiles'] not in (2,4): raise ValueError('Qualified tile count is 0/2/4')
            nx,ny=(2,2) if self.policy['detail_tiles']==4 else ((2,1) if w>=h else (1,2))
            for j in range(ny):
                for i in range(nx):
                    a,b,c,d=round(i*w/nx),round(j*h/ny),round((i+1)*w/nx),round((j+1)*h/ny)
                    views.append((image.crop((a,b,c,d)),(a/w,b/h,c/w,d/h),self.policy['tile_patches']))
        return views
    def image_tokens(self,images,times,capture):
        outputs=[]; video=len(images)>1; units=[]
        for fi,image in enumerate(images):
            for vi,(view,box,patches) in enumerate(self.tile_views(image,video)):
                batch=self.image_processor(images=view,return_tensors='pt',max_num_patches=patches)
                units.append({'frame':fi,'view':vi,'image':view,'box':box,'patches':patches,'batch':batch})
        def count(unit):
            hh,ww=(int(x) for x in unit['batch']['spatial_shapes'][0])
            return ((hh+1)//2)*((ww+1)//2)
        while sum(count(u) for u in units)>self.policy['max_visual_tokens']:
            candidates=[u for u in units if u['patches']>64]
            if not candidates: raise ValueError('Actual visual grids cannot fit the qualified token budget')
            u=max(candidates,key=count); u['patches']=max(64,u['patches']//2)
            u['batch']=self.image_processor(images=u['image'],return_tensors='pt',max_num_patches=u['patches'])
        for unit in units:
            fi,vi,box=unit['frame'],unit['view'],unit['box']
            batch={k:v.to(self.device) for k,v in unit['batch'].items()}
            features=self.vision(**batch).last_hidden_state
            hh,ww=(int(x) for x in batch['spatial_shapes'][0])
            tokens,_=spatial_downshuffle(features,hh,ww); tokens=self.vision_projector(tokens)
            gh,gw=(hh+1)//2,(ww+1)//2
            yy,xx=torch.meshgrid(torch.arange(gh,device=self.device),torch.arange(gw,device=self.device),indexing='ij')
            xx,yy=xx.flatten().float(),yy.flatten().float(); xmin,ymin,xmax,ymax=box
            x0=2*xx/ww; x1=torch.clamp(2*xx+2,max=ww)/ww
            y0=2*yy/hh; y1=torch.clamp(2*yy+2,max=hh)/hh
            x=xmin+(x0+x1)/2*(xmax-xmin); y=ymin+(y0+y1)/2*(ymax-ymin)
            bw=(x1-x0)*(xmax-xmin); bh=(y1-y0)*(ymax-ymin)
            time=float(times[fi]); cap=capture[fi] if capture is not None and fi<len(capture) else None
            clock=[time/60,math.sin(time),math.cos(time),0. if cap is None else float(cap)/3600,0. if cap is None else 1.]
            coord=torch.stack([x,y,bw,bh,torch.full_like(x,float(fi)),*([torch.full_like(x,v) for v in clock]),torch.full_like(x,float(vi)),torch.full_like(x,float(vi==0))],dim=-1)
            outputs.append(tokens+self.spatial_clock(coord[None])+self.modality.weight[0])
        return torch.cat(outputs,dim=1) if outputs else None
    def motion_tokens(self,frames):
        if not frames: return None
        n=self.policy['motion_frames']; ids=np.linspace(0,len(frames)-1,n).round().astype(int)
        arrays=[np.asarray(frames[i].resize((self.policy['motion_size'],)*2),dtype=np.float32)/255 for i in ids]
        x=torch.from_numpy(np.stack(arrays)).permute(3,0,1,2)[None].to(self.device)
        mean=x.new_tensor([.485,.456,.406]).view(1,3,1,1,1); std=x.new_tensor([.229,.224,.225]).view(1,3,1,1,1)
        out=self.motion((x-mean)/std,training=True)
        if isinstance(out,(list,tuple)): out=out[-1]
        if out.shape[-1]!=self.motion.embed_dim: out=out.reshape(*out.shape[:-1],-1,self.motion.embed_dim).mean(-2)
        return self.motion_projector(out)+self.modality.weight[1]
    def audio_tokens(self,audio,fam):
        if audio is None: return None
        wav,sr=audio; wav=wav.mean(axis=0)
        if fam in ('speech_in','audiovisual'):
            mono=audio_resample(wav,sr,16000); parts=[]; clocks=[]
            for start in range(0,len(mono),30*16000):
                piece=mono[start:start+30*16000]
                f=self.speech_processor(piece,sampling_rate=16000,return_tensors='pt').input_features.to(self.device)
                out=self.speech_in(input_features=f).last_hidden_state
                out=out[:,:max(1,math.ceil(len(piece)/16000*50))]; parts.append(out)
                clocks.append(start/16000+torch.arange(out.shape[1],device=self.device)/50)
            times=torch.cat(clocks)
            position=self.audio_clock(torch.stack([times/30,torch.sin(times),torch.cos(times),torch.ones_like(times)],dim=-1))[None]
            return self.speech_projector(torch.cat(parts,dim=1),position=position)+self.modality.weight[2]
        mono=audio_resample(wav,sr,48000); parts=[]; clocks=[]
        for start in range(0,len(mono),10*48000):
            piece=mono[start:start+10*48000]
            f=self.sound_processor(piece,sampling_rate=48000,return_tensors='pt',truncation='rand_trunc',padding='repeatpad')
            out=self.sound_in(**{k:v.to(self.device) for k,v in f.items()}).last_hidden_state
            if out.ndim!=4: raise ValueError('Expected CLAP frequency/time feature map')
            full_time=out.shape[-1]; valid=max(1,math.ceil(full_time*len(piece)/(10*48000)))
            out=out[...,:valid]; frequency=out.shape[-2]
            clocks.append((start/48000+torch.arange(valid,device=self.device)*10/full_time).repeat(frequency))
            parts.append(out.flatten(2).transpose(1,2))
        times=torch.cat(clocks)
        position=self.audio_clock(torch.stack([times/30,torch.sin(times),torch.cos(times),torch.ones_like(times)],dim=-1))[None]
        return self.sound_projector(torch.cat(parts,dim=1),position=position)+self.modality.weight[3]
    def prompt_ids(self,prompt):
        messages=[{'role':'system','content':'You are Spark, an English and Chinese multimodal assistant. Use the supplied media and clocks. Report uncertainty when capture time or evidence is unknown.'},{'role':'user','content':prompt}]
        return self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt').to(self.device)
    def prefix(self,row,media):
        if row['_family'].endswith('_out') and any(media[k] is not None for k in ('audio','motion')): raise ValueError('Output target leaked into input encoder')
        parts=[]
        for x in (self.image_tokens(media['images'],media['times'],media['capture_times']),self.motion_tokens(media['motion']),self.audio_tokens(media['audio'],row['_family'])):
            if x is not None: parts.append(x)
        ids=self.prompt_ids(row['prompt']); parts.append(self.get_input_embeddings()(ids)); return torch.cat(parts,dim=1)
    def hidden_states(self,embeds):
        if embeds.shape[1]>self.policy['max_sequence']: raise ValueError('Input exceeds qualified context; no silent truncation')
        return self.spark.model(inputs_embeds=embeds,attention_mask=torch.ones(embeds.shape[:2],device=self.device,dtype=torch.long),use_cache=False,return_dict=True).last_hidden_state
    def text_loss(self,row,media):
        prefix=self.prefix(row,media)
        ids=self.tokenizer(row['answer']+self.tokenizer.eos_token,add_special_tokens=False,return_tensors='pt').input_ids.to(self.device)
        if ids.shape[1]>self.policy['max_answer_tokens'] or prefix.shape[1]+ids.shape[1]>self.policy['max_sequence']: raise ValueError('Supervision would be truncated; revise a versioned context policy')
        embeds=torch.cat([prefix,self.get_input_embeddings()(ids)],dim=1)
        hidden=self.hidden_states(embeds)[:,prefix.shape[1]-1:-1]; total=hidden.new_zeros((),dtype=torch.float32)
        # Recompute chunked logits during backward; do not hold L*vocab logits.
        from torch.utils.checkpoint import checkpoint
        for start in range(0,ids.shape[1],128):
            hs=hidden[:,start:start+128]; labels=ids[:,start:start+128]
            def ce(x,y): return F.cross_entropy(self.get_output_embeddings()(x).float().reshape(-1,self.config.vocab_size),y.flatten(),reduction='sum')
            total=total+checkpoint(ce,hs,labels,use_reentrant=False)
        return total/ids.numel(),ids.numel()
    def loss(self,row,media):
        fam=row['_family']
        if fam.endswith('_out'):
            hidden=self.hidden_states(self.prefix(row,media))
            if fam=='speech_out': return self.speech_out.loss(hidden,media['target_audio'],row)
            return self.sound_out.loss(hidden,media['target_audio'],row)
        return self.text_loss(row,media)
    def forward(self,examples=None,**kwargs):
        if not examples or len(examples)!=1: raise ValueError('Qualified microbatch is one source record')
        row,media=examples[0]; loss,n=self.loss(row,media)
        return {'loss':loss,'supervised_units':n}
    @torch.no_grad()
    def answer(self,row,media,max_new_tokens=None):
        if max_new_tokens is None:
            max_new_tokens=1024 if row['task'] in ('document_ocr','music_instrument_note_understanding') else (512 if row['task'] in ('factual_caption','image_caption','video_caption','video_box_tracking','region_dense_video_caption') else 256)
        embeds=self.prefix(row,media); generated=[]
        for _ in range(max_new_tokens):
            if embeds.shape[1]>=self.policy['max_sequence']: break
            hidden=self.hidden_states(embeds); token=self.get_output_embeddings()(hidden[:,-1:]).argmax(-1)
            if token.item()==self.tokenizer.eos_token_id: break
            generated.append(token.item()); embeds=torch.cat([embeds,self.get_input_embeddings()(token)],dim=1)
        return self.tokenizer.decode(generated,skip_special_tokens=True)
    @torch.no_grad()
    def generate_audio(self,row,media,seconds=None,steps=50):
        hidden=self.hidden_states(self.prefix(row,media))
        if row['_family']=='speech_out': return self.speech_out.generate(hidden,row)
        return self.sound_out.generate(hidden,seconds=seconds or 15,steps=steps)

    @torch.no_grad()
    def spoken_answer(self,row,media,max_new_tokens=128):
        """Spark text decoding plus its native, state-conditioned speech head.

        This inference composition is not evidence of paired audiovisual-dialogue
        training. The generation head never receives a reference waveform.
        """
        if media.get('target_audio') is not None: raise ValueError('Inference cannot consume a target waveform')
        text=self.answer(row,media,max_new_tokens=max_new_tokens)
        answer_ids=self.tokenizer(text,add_special_tokens=False,return_tensors='pt').input_ids.to(self.device)
        context=torch.cat([self.prefix(row,media),self.get_input_embeddings()(answer_ids)],dim=1)
        hidden=self.hidden_states(context)
        request={'language':row['language'],'utterance':text,'prompt':'Speak this answer:\n'+text}
        audio,sr=self.speech_out.generate(hidden,request)
        return {'text':text,'audio':audio,'sample_rate':sr,'paired_audio_dialogue_training_verified':False}
