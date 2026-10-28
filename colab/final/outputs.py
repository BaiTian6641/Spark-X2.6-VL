"""Pretrained native audio decoders conditioned differentiably on Spark states."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .model import Resampler,audio_resample

class NativeSpeechOutput(nn.Module):
    def __init__(self,path,spark_hidden,policy,pretrained=True):
        super().__init__()
        from qwen_tts.core.models import Qwen3TTSForConditionalGeneration,Qwen3TTSConfig
        from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
        from transformers import AutoTokenizer
        if pretrained:
            base=Qwen3TTSForConditionalGeneration.from_pretrained(path,local_files_only=True,dtype=torch.float32,attn_implementation='sdpa')
        else:
            base=Qwen3TTSForConditionalGeneration(Qwen3TTSConfig.from_pretrained(path,local_files_only=True))
            base.speech_tokenizer=Qwen3TTSTokenizer.from_pretrained(str(__import__('pathlib').Path(path)/'speech_tokenizer'),device_map='cpu')
        self.talker=base.talker; self.qconfig=base.config
        self.codec=base.speech_tokenizer; self.codec_model=self.codec.model
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.bridge=Resampler(spark_hidden,self.talker.config.hidden_size,policy['generator_prefix_tokens'])
        self.default_voice=nn.Parameter(torch.zeros(1,1,self.talker.config.hidden_size))
        # Speaker encoder is deliberately absent: no target-waveform speaker cue.
        del base; self.freeze_codec()
    @property
    def device(self): return next(self.talker.parameters()).device
    def freeze_codec(self): self.codec_model.requires_grad_(False); self.codec_model.eval()
    def text(self,row):
        text=row.get('utterance') or row.get('metadata',{}).get('text')
        if not text or text not in row['prompt']: raise ValueError('TTS lexical text must be explicitly present in the user request')
        return text
    def _text_ids(self,text):
        ids=self.tokenizer(f'<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n',
                           add_special_tokens=False,return_tensors='pt').input_ids.to(self.device)
        if ids.shape[1]<9: raise ValueError('TTS text layout is too short')
        return ids[:,:3],ids[:,3:-5]
    def conditioning(self,hidden,row):
        c=self.talker.config; cfg=self.qconfig
        role_ids,text_ids=self._text_ids(self.text(row))
        temb=self.talker.get_text_embeddings(); proj=self.talker.text_projection
        special=torch.tensor([[cfg.tts_bos_token_id,cfg.tts_eos_token_id,cfg.tts_pad_token_id]],device=self.device)
        bos,eos,pad=proj(temb(special)).chunk(3,dim=1)
        language='chinese' if row['language']=='zh' else 'english'
        cemb=self.talker.get_input_embeddings()
        control=cemb(torch.tensor([[c.codec_think_id,c.codec_think_bos_id,c.codec_language_id[language],c.codec_think_eos_id]],device=self.device))
        pad_bos=cemb(torch.tensor([[c.codec_pad_id,c.codec_bos_id]],device=self.device))
        codec=torch.cat([control,self.default_voice.to(control.dtype),pad_bos],dim=1)
        role=proj(temb(role_ids))
        head=torch.cat([pad.expand(-1,codec.shape[1]-2,-1),bos],dim=1)+codec[:,:-1]
        body=torch.cat([proj(temb(text_ids)),eos],dim=1)+cemb(torch.full((1,text_ids.shape[1]+1),c.codec_pad_id,device=self.device))
        start=pad+cemb(torch.tensor([[c.codec_bos_id]],device=self.device))
        prefix=torch.cat([self.bridge(hidden),role,head,body,start],dim=1)
        return prefix,pad
    def frame_embeddings(self,codes,pad):
        embeds=self.talker.get_input_embeddings()(codes[None,:,0])
        for i in range(1,self.talker.config.num_code_groups): embeds=embeds+self.talker.code_predictor.get_input_embeddings()[i-1](codes[None,:,i])
        return embeds+pad
    def codec_loss(self,hidden,codes,row):
        prefix,pad=self.conditioning(hidden,row)
        embeds=torch.cat([prefix,self.frame_embeddings(codes,pad)],dim=1)
        out=self.talker.model(inputs_embeds=embeds,use_cache=False,return_dict=True).last_hidden_state[:,prefix.shape[1]-1:]
        labels=torch.cat([codes[:,0],codes.new_tensor([self.talker.config.codec_eos_token_id])])[None]
        loss=F.cross_entropy(self.talker.codec_head(out).float().flatten(0,1),labels.flatten())
        # Official SFT alignment: frame i's residual codebooks use the hidden
        # state after frame i, not the preceding codec-BOS/frame-i-1 state.
        _,sub=self.talker.forward_sub_talker_finetune(codes,out[0,1:len(codes)+1])
        return loss+.3*sub,int(codes.numel()+1)
    def loss(self,hidden,audio,row):
        wav,sr=audio; mono=wav.mean(axis=0)
        self.codec.device=next(self.codec_model.parameters()).device
        with torch.no_grad():
            codes=self.codec.encode([mono],sr=sr).audio_codes[0].to(self.device).long().clone()
        if codes.ndim!=2 or codes.shape[1]!=16: raise ValueError('Expected native T x 16 codec groups')
        return self.codec_loss(hidden,codes,row)
    @torch.no_grad()
    def generate(self,hidden,row,max_seconds=30):
        prefix,pad=self.conditioning(hidden,row); codes=[]
        output=self.talker.model(inputs_embeds=prefix,use_cache=True,return_dict=True)
        state=output.last_hidden_state[:,-1:]; past=output.past_key_values
        for _ in range(math.ceil(max_seconds*12.5)):
            logits=self.talker.codec_head(state).float(); allowed=logits.new_full(logits.shape,float('-inf'))
            allowed[...,:2048]=logits[...,:2048]; allowed[...,self.talker.config.codec_eos_token_id]=logits[...,self.talker.config.codec_eos_token_id]
            first=allowed.argmax(-1)
            if first.item()==self.talker.config.codec_eos_token_id: break
            rest=self.talker.code_predictor.generate(inputs_embeds=torch.cat([state,self.talker.get_input_embeddings()(first)],dim=1),max_new_tokens=15,do_sample=False,output_hidden_states=False,return_dict_in_generate=True).sequences
            if rest.shape[-1]!=15: raise RuntimeError('Sub-talker returned a different codec group count')
            frame=torch.cat([first,rest],dim=-1)[0]; codes.append(frame)
            output=self.talker.model(inputs_embeds=self.frame_embeddings(frame[None],pad),past_key_values=past,use_cache=True,return_dict=True)
            state=output.last_hidden_state[:,-1:]; past=output.past_key_values
        if not codes: raise RuntimeError('Speech decoder emitted no waveform codes')
        self.codec.device=next(self.codec_model.parameters()).device
        waveforms,sr=self.codec.decode([{'audio_codes':torch.stack(codes)}])
        wav=waveforms[0]
        if isinstance(wav,torch.Tensor): wav=wav.detach().float().cpu().numpy()
        return wav,int(sr)

class NativeSoundOutput(nn.Module):
    def __init__(self,path,spark_hidden,policy,pretrained=True):
        super().__init__()
        from diffusers import StableAudioDiTModel,AutoencoderOobleck
        from diffusers.pipelines.stable_audio.modeling_stable_audio import StableAudioProjectionModel
        self.dit=(StableAudioDiTModel.from_pretrained(path,subfolder='transformer',local_files_only=True,dtype=torch.float32) if pretrained else StableAudioDiTModel.from_config(StableAudioDiTModel.load_config(path,subfolder='transformer',local_files_only=True)))
        self.vae=(AutoencoderOobleck.from_pretrained(path,subfolder='vae',local_files_only=True,dtype=torch.float32) if pretrained else AutoencoderOobleck.from_config(AutoencoderOobleck.load_config(path,subfolder='vae',local_files_only=True)))
        self.projection=(StableAudioProjectionModel.from_pretrained(path,subfolder='projection_model',local_files_only=True,dtype=torch.float32) if pretrained else StableAudioProjectionModel.from_config(StableAudioProjectionModel.load_config(path,subfolder='projection_model',local_files_only=True)))
        dim=self.projection.config.text_encoder_dim
        self.bridge=Resampler(spark_hidden,dim,policy['generator_prefix_tokens'])
        from diffusers import EDMDPMSolverMultistepScheduler
        self.scheduler_config=dict(EDMDPMSolverMultistepScheduler.from_pretrained(path,subfolder='scheduler',local_files_only=True).config)
        self.vae.requires_grad_(False); self.vae.eval()
    @property
    def device(self): return next(self.dit.parameters()).device
    def conditioning(self,hidden,seconds):
        duration=torch.tensor([seconds],device=self.device,dtype=torch.float32)
        output=self.projection(text_hidden_states=self.bridge(hidden),start_seconds=torch.zeros_like(duration),end_seconds=duration)
        cross=torch.cat([output.text_hidden_states,output.seconds_start_hidden_states,output.seconds_end_hidden_states],dim=1)
        glob=torch.cat([output.seconds_start_hidden_states,output.seconds_end_hidden_states],dim=2)
        return cross,glob
    def velocity(self,z,t,cross,glob):
        from diffusers.models.embeddings import get_1d_rotary_pos_embed
        rope=get_1d_rotary_pos_embed(self.dit.config.attention_head_dim//2,z.shape[2]+glob.shape[1],use_real=True,repeat_interleave_real=False)
        rope=tuple(x.to(self.device) for x in rope)
        return self.dit(hidden_states=z,timestep=t,encoder_hidden_states=cross,global_hidden_states=glob,rotary_embedding=rope,return_dict=True).sample
    def target(self,audio):
        wav,sr=audio; duration=wav.shape[-1]/sr
        if not 0<duration<=47: raise ValueError('Target exceeds the reviewed full-clip generator duration; no weak-label cropping')
        rate=int(self.vae.config.sampling_rate); wav=audio_resample(wav,sr,rate)
        channels=int(self.vae.config.audio_channels)
        if channels==2 and wav.shape[0]==1: wav=wav.repeat(2,axis=0)
        if wav.shape[0]!=channels: raise ValueError('Unsupported generator target channel layout')
        factor=math.prod(self.vae.config.downsampling_ratios); length=math.ceil(wav.shape[-1]/factor)*factor
        x=torch.as_tensor(wav,device=self.device,dtype=torch.float32)[None]; x=F.pad(x,(0,length-x.shape[-1]))
        with torch.no_grad(): z=self.vae.encode(x).latent_dist.sample()
        return z,duration
    def scheduler(self):
        from diffusers import EDMDPMSolverMultistepScheduler
        return EDMDPMSolverMultistepScheduler.from_config(self.scheduler_config)
    def diffusion(self,z,noise,sigma):
        """EDM v-prediction matching the pinned Diffusers pipeline."""
        sigma=sigma.to(device=z.device,dtype=z.dtype)
        sigma_data=float(self.scheduler_config.get('sigma_data',1.0))
        view=sigma[:,None,None]
        scale=torch.rsqrt(view.square()+sigma_data**2)
        scaled=(z+view*noise)*scale
        timestep=.25*torch.log(sigma)
        target=(noise-view*z)*scale
        return scaled,timestep,target
    def loss(self,hidden,audio,row):
        z,seconds=self.target(audio); noise=torch.randn_like(z)
        lo=math.log(float(self.scheduler_config['sigma_min'])); hi=math.log(float(self.scheduler_config['sigma_max']))
        sigma=torch.rand(z.shape[0],device=z.device,dtype=z.dtype).mul_(hi-lo).add_(lo).exp_()
        scaled,timestep,target=self.diffusion(z,noise,sigma)
        cross,glob=self.conditioning(hidden,seconds); pred=self.velocity(scaled,timestep,cross,glob)
        return F.mse_loss(pred.float(),target.float()),int(z.numel())
    @torch.no_grad()
    def generate(self,hidden,seconds=15,steps=50):
        if not 0<seconds<=47: raise ValueError('Generation duration must be within (0,47] seconds')
        scheduler=self.scheduler(); scheduler.set_timesteps(steps,device=self.device)
        rate=int(self.vae.config.sampling_rate); factor=math.prod(self.vae.config.downsampling_ratios)
        n=math.ceil(seconds*rate/factor); z=torch.randn((1,self.dit.config.in_channels,n),device=self.device,dtype=torch.float32)*scheduler.init_noise_sigma
        cross,glob=self.conditioning(hidden,seconds)
        for t in scheduler.timesteps:
            scaled=scheduler.scale_model_input(z,t)
            velocity=self.velocity(scaled,t.expand(z.shape[0]),cross,glob)
            z=scheduler.step(velocity,t,z).prev_sample
        wav=self.vae.decode(z).sample[0,:,:round(seconds*rate)].float().cpu().numpy().T
        return wav,rate
