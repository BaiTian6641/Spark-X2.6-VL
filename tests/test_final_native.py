"""CPU component/API tests. These do not certify the full CUDA campaign."""
import json,types
from pathlib import Path
import pytest
import torch
from torch import nn
from colab.final.model import spatial_downshuffle,Resampler
from colab.final.outputs import NativeSpeechOutput,NativeSoundOutput


def test_non_square_native_grid_padding_and_gradient():
    x=torch.arange(15.,requires_grad=True).reshape(1,15,1)
    out,mask=spatial_downshuffle(x,3,5)
    assert out.shape==(1,6,4)
    assert out[0,0].tolist()==[0,1,5,6]
    assert out[0,-1].tolist()==[14,0,0,0]
    out.sum().backward()
    assert mask.all()


def test_real_naflex_patch_interface_backward():
    from transformers import Siglip2VisionConfig,Siglip2VisionModel
    cfg=Siglip2VisionConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_patches=256,patch_size=16,vision_use_head=False)
    model=Siglip2VisionModel(cfg)
    patches=torch.randn(1,20,768,requires_grad=True); mask=torch.tensor([[1]*15+[0]*5])
    out=model(pixel_values=patches,pixel_attention_mask=mask,spatial_shapes=torch.tensor([[3,5]])).last_hidden_state
    reduced,_=spatial_downshuffle(out,3,5); reduced.square().mean().backward()
    assert torch.isfinite(patches.grad).all() and patches.grad[:,:15].abs().sum()>0
    assert patches.grad[:,15:].abs().sum()==0

def test_clap_pretrained_loader_uses_audio_model_subtree(monkeypatch):
    from colab.final.model import SparkNative
    from transformers import ClapModel
    marker=object()
    monkeypatch.setattr(ClapModel,'from_pretrained',lambda *args,**kwargs: types.SimpleNamespace(
        audio_model=types.SimpleNamespace(audio_encoder=marker)))
    assert SparkNative.load_sound_in('clap-checkpoint',True) is marker

def test_clap_loader_branches_have_matching_audio_encoder_keys(monkeypatch,tmp_path):
    from colab.final.model import SparkNative
    from transformers import ClapAudioConfig,ClapAudioModel,ClapModel
    config=ClapAudioConfig(hidden_size=8,projection_dim=4,depths=[1],num_attention_heads=[1],
        num_hidden_layers=1,patch_embeds_hidden_size=8,window_size=2,spec_size=64,mlp_ratio=1.0)
    config.save_pretrained(tmp_path)
    monkeypatch.setattr(ClapModel,'from_pretrained',lambda *args,**kwargs: types.SimpleNamespace(
        audio_model=ClapAudioModel(config)))
    pretrained=SparkNative.load_sound_in(tmp_path,True)
    from_config=SparkNative.load_sound_in(tmp_path,False)
    assert pretrained.state_dict().keys()==from_config.state_dict().keys()


def tiny_talker():
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration
    cfg=Qwen3TTSTalkerConfig(vocab_size=2200,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,text_vocab_size=128,text_hidden_size=64,num_code_groups=16,rope_scaling={'rope_type':'default','mrope_section':[2,2,4],'interleaved':True},codec_eos_token_id=2150,codec_think_id=2154,codec_think_bos_id=2156,codec_think_eos_id=2157,codec_pad_id=2148,codec_bos_id=2149,codec_language_id={'english':2050,'chinese':2055},code_predictor_config={'vocab_size':2048,'hidden_size':64,'intermediate_size':128,'num_hidden_layers':2,'num_attention_heads':4,'num_key_value_heads':2,'head_dim':16,'num_code_groups':16})
    cfg._attn_implementation='sdpa'
    wrapper=NativeSpeechOutput.__new__(NativeSpeechOutput); nn.Module.__init__(wrapper)
    wrapper.talker=Qwen3TTSTalkerForConditionalGeneration(cfg)
    wrapper.qconfig=types.SimpleNamespace(tts_pad_token_id=10,tts_bos_token_id=11,tts_eos_token_id=12)
    class TinyTokenizer:
        def __call__(self,text,add_special_tokens=False,return_tensors='pt'):
            assert text=='<|im_start|>assistant\nhi<|im_end|>\n<|im_start|>assistant\n'
            return types.SimpleNamespace(input_ids=torch.tensor([[1,2,3,4,5,6,7,8,9,10,11]]))
    wrapper.tokenizer=TinyTokenizer()
    wrapper.bridge=Resampler(48,64,4);wrapper.default_voice=nn.Parameter(torch.zeros(1,1,64))
    return wrapper


def test_native_talker_official_layout_subtalker_alignment_and_gradients():
    model=tiny_talker(); model.talker.eval(); hidden=torch.randn(1,7,48,requires_grad=True)
    prefix,pad=model.conditioning(hidden,{'language':'en','prompt':'Say hi','utterance':'hi'})
    codes=torch.randint(0,32,(3,16))
    assert prefix.shape[1]==18  # bridge 4 + role 3 + control 6 + text/eos 4 + codec BOS 1
    temb=model.talker.get_text_embeddings(); projected=model.talker.text_projection(temb(torch.tensor([[1,2,3]])))
    torch.testing.assert_close(prefix[0,4:7],projected[0])
    embeds=torch.cat([prefix,model.frame_embeddings(codes,pad)],dim=1)
    expected=model.talker.model(inputs_embeds=embeds,use_cache=False,return_dict=True).last_hidden_state[:,prefix.shape[1]-1:]
    original=model.talker.forward_sub_talker_finetune; seen={}
    def probe(codec_ids,talker_hidden_states):
        seen['hidden']=talker_hidden_states.detach().clone()
        return original(codec_ids,talker_hidden_states)
    model.talker.forward_sub_talker_finetune=probe
    loss,_=model.codec_loss(hidden,codes,{'language':'en','prompt':'Say hi','utterance':'hi'})
    torch.testing.assert_close(seen['hidden'],expected[0,1:4])
    loss.backward()
    assert torch.isfinite(hidden.grad).all() and hidden.grad.abs().sum()>0
    assert all(head.weight.grad is not None and head.weight.grad.abs().sum()>0 for head in model.talker.code_predictor.lm_head)
    assert model.talker.model.layers[0].self_attn.q_proj.weight.grad.abs().sum()>0

def test_native_talker_cached_frames_match_full_prefill():
    model=tiny_talker(); model.talker.eval(); hidden=torch.randn(1,7,48)
    prefix,pad=model.conditioning(hidden,{'language':'en','prompt':'Say hi','utterance':'hi'})
    codes=torch.randint(0,32,(3,16))
    with torch.no_grad():
        embeds=torch.cat([prefix,model.frame_embeddings(codes,pad)],dim=1)
        expected=model.talker.model(inputs_embeds=embeds,use_cache=False,return_dict=True).last_hidden_state
        output=model.talker.model(inputs_embeds=prefix,use_cache=True,return_dict=True)
        past=output.past_key_values; actual=[output.last_hidden_state]
        for frame in codes:
            output=model.talker.model(inputs_embeds=model.frame_embeddings(frame[None],pad),past_key_values=past,use_cache=True,return_dict=True)
            past=output.past_key_values; actual.append(output.last_hidden_state)
        torch.testing.assert_close(torch.cat(actual,dim=1),expected,atol=1e-5,rtol=1e-5)


def test_native_dit_v_prediction_reaches_spark_conditioning():
    from diffusers import StableAudioDiTModel
    from diffusers.pipelines.stable_audio.modeling_stable_audio import StableAudioProjectionModel
    model=NativeSoundOutput.__new__(NativeSoundOutput);nn.Module.__init__(model)
    model.dit=StableAudioDiTModel(sample_size=16,in_channels=8,out_channels=8,num_layers=2,attention_head_dim=16,num_attention_heads=4,num_key_value_attention_heads=2,cross_attention_dim=32,cross_attention_input_dim=32,time_proj_dim=16,global_states_input_dim=64)
    model.projection=StableAudioProjectionModel(text_encoder_dim=32,conditioning_dim=32,min_value=0,max_value=512)
    model.bridge=Resampler(48,32,4)
    model.scheduler_config={'sigma_min':.3,'sigma_max':500.,'sigma_data':1.0}
    model.freeze_fixed_conditioning()
    assert not model.projection.start_number_conditioner.time_positional_embedding[0].weights.requires_grad
    hidden=torch.randn(1,7,48,requires_grad=True);z=torch.randn(1,8,16);noise=torch.randn_like(z)
    scaled,timestep,target=model.diffusion(z,noise,torch.tensor([2.0]))
    torch.testing.assert_close(timestep,torch.tensor([.25*__import__('math').log(2.)]))
    torch.testing.assert_close(scaled,(z+2*noise)/__import__('math').sqrt(5))
    torch.testing.assert_close(target,(noise-2*z)/__import__('math').sqrt(5))
    cross,glob=model.conditioning(hidden,15);out=model.velocity(scaled,timestep,cross,glob)
    torch.nn.functional.mse_loss(out,target).backward()
    assert torch.isfinite(hidden.grad).all() and hidden.grad.abs().sum()>0
    assert model.dit.proj_out.weight.grad.abs().sum()>0
    assert model.projection.start_number_conditioner.time_positional_embedding[0].weights.grad is None
    assert model.projection.start_number_conditioner.time_positional_embedding[1].weight.grad.abs().sum()>0
    assert model.projection.end_number_conditioner.time_positional_embedding[0].weights.grad.abs().sum()>0
    assert type(model.scheduler()).__name__=='EDMDPMSolverMultistepScheduler'

def test_diffusers_pretrained_loaders_accept_torch_dtype(tmp_path):
    from diffusers import StableAudioDiTModel,AutoencoderOobleck
    from diffusers.pipelines.stable_audio.modeling_stable_audio import StableAudioProjectionModel
    dit=StableAudioDiTModel(sample_size=16,in_channels=8,out_channels=8,num_layers=1,attention_head_dim=16,num_attention_heads=4,num_key_value_attention_heads=2,cross_attention_dim=32,cross_attention_input_dim=32,time_proj_dim=16,global_states_input_dim=64)
    dit.save_pretrained(tmp_path/'dit')
    loaded=StableAudioDiTModel.from_pretrained(tmp_path/'dit',torch_dtype=torch.float32,local_files_only=True)
    assert next(loaded.parameters()).dtype==torch.float32
    vae=AutoencoderOobleck(encoder_hidden_size=128,downsampling_ratios=[2],channel_multiples=[1],decoder_channels=128,decoder_input_channels=64,audio_channels=2,sampling_rate=44100)
    vae.save_pretrained(tmp_path/'vae')
    loaded_vae=AutoencoderOobleck.from_pretrained(tmp_path/'vae',torch_dtype=torch.float32,local_files_only=True)
    assert next(loaded_vae.parameters()).dtype==torch.float32
    projection=StableAudioProjectionModel(text_encoder_dim=32,conditioning_dim=32,min_value=0,max_value=512)
    projection.save_pretrained(tmp_path/'projection')
    loaded_projection=StableAudioProjectionModel.from_pretrained(tmp_path/'projection',torch_dtype=torch.float32,local_files_only=True)
    assert next(loaded_projection.parameters()).dtype==torch.float32

def tiny_original_spark(adapt=True):
    """The pinned custom decoder source, dimension-reduced for CPU graph tests."""
    import hashlib,importlib,sys,types
    root=Path('models/Spark-X2.6-4B').resolve(); lock=json.load(open('colab/final/components.lock.json'))
    for name in ('configuration_spark.py','modeling_spark.py'):
        payload=(root/name).read_bytes(); expected=next(f for f in lock['spark']['files'] if f['path']==name)
        assert hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()==expected['git_blob_sha1']
    package=types.ModuleType('spark_formal_tiny');package.__path__=[str(root)];sys.modules['spark_formal_tiny']=package
    config=importlib.import_module('spark_formal_tiny.configuration_spark').Spark2_5Config(vocab_size=131072,hidden_size=48,intermediate_size=96,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=12,max_position_embeddings=128,pad_token_id=0,tie_word_embeddings=True,headwise_attn_output_gate=True,sliding_window=8,layer_types=['sliding_attention','full_attention'],rope_parameters={'sliding_attention':{'rope_theta':10000,'partial_rotary_factor':1.0},'full_attention':{'rope_theta':10000,'partial_rotary_factor':1.0}})
    config._attn_implementation='eager'
    module=importlib.import_module('spark_formal_tiny.modeling_spark')
    module.eager_attention_forward=getattr(module.eager_attention_forward,'_spark_reference',module.eager_attention_forward)
    model=module.Spark2_5ForCausalLM(config)
    if adapt:
        from colab.final.model import install_spark_sdpa
        install_spark_sdpa(model)
    return model


def test_native_audio_objective_updates_the_actual_custom_spark_decoder():
    spark=tiny_original_spark(); talker=tiny_talker();opt=torch.optim.AdamW(spark.parameters(),lr=1e-3)
    ids=torch.tensor([[1,4,5,6,2]]); h=spark.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True).last_hidden_state
    codes=torch.randint(0,32,(3,16))
    loss,_=talker.codec_loss(h,codes,{'language':'en','prompt':'Say hi','utterance':'hi'}); loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in spark.parameters())
    assert all(p.grad.abs().sum()>0 for p in spark.parameters())
    weight=spark.model.layers[0].self_attn.q_k_v_proj.weight;before=weight.detach().clone();opt.step()
    assert not torch.equal(before,weight)

def test_spark_kernel_adapter_matches_reference_masks_forward_and_backward():
    from colab.final.model import install_spark_sdpa
    model=tiny_original_spark(adapt=False); model.eval(); ids=torch.tensor([[1,5,6,7,8,9,10,11,12,13,14,2]])
    a=model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True).last_hidden_state
    a.float().square().mean().backward(); expected={n:p.grad.clone() for n,p in model.named_parameters()}; model.zero_grad(set_to_none=True)
    install_spark_sdpa(model)
    b=model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True).last_hidden_state
    b.float().square().mean().backward()
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
    for n,p in model.named_parameters(): torch.testing.assert_close(expected[n],p.grad,atol=2e-6,rtol=2e-4)

def test_spark_text_microbatch_matches_single_record_losses():
    from transformers import AutoTokenizer
    from colab.final.model import SparkNative,ChatFormat
    from colab.final.inference import empty_media
    model=SparkNative.__new__(SparkNative); nn.Module.__init__(model)
    model.spark=tiny_original_spark(); model.config=model.spark.config
    model.tokenizer=AutoTokenizer.from_pretrained('models/Spark-X2.6-4B',trust_remote_code=True,local_files_only=True)
    model.chat=ChatFormat(model.tokenizer); model.hidden=model.config.hidden_size
    model.policy={'max_sequence':128,'max_answer_tokens':64,'microbatch_size':2}
    model.image_tokens=lambda *args: None; model.motion_tokens=lambda *args: None; model.audio_tokens=lambda *args: None
    rows=[{'_family':'text','language':'en','task':'text_replay','prompt':'Answer briefly.','answer':'One.'},
          {'_family':'text','language':'en','task':'text_replay','prompt':'Give a number.','answer':'Two.'}]
    singles=[model.text_loss(row,empty_media())[0] for row in rows]
    result=model(examples=[(row,empty_media()) for row in rows])
    torch.testing.assert_close(result['per_example_loss'],torch.stack(singles),atol=1e-5,rtol=1e-5)
    result['loss'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.spark.parameters())
    with torch.no_grad():
        cached=model.answer(rows[0],empty_media(),max_new_tokens=3)
        embeds=model.prefix(rows[0],empty_media()); expected=[]
        for _ in range(3):
            hidden=model.hidden_states(embeds)
            token=model.get_output_embeddings()(hidden[:,-1:]).argmax(-1)
            if token.item()==model.tokenizer.eos_token_id: break
            expected.append(token.item()); embeds=torch.cat([embeds,model.get_input_embeddings()(token)],dim=1)
        assert cached==model.tokenizer.decode(expected,skip_special_tokens=True)

def test_temporal_resampler_keeps_event_order_when_absolute_clock_is_fixed():
    torch.manual_seed(99)
    model=Resampler(16,32,4); clock=nn.Linear(4,32)
    events=torch.randn(1,5,16,requires_grad=True); times=torch.arange(5.).reshape(1,5,1)
    position=clock(torch.cat([times,times.sin(),times.cos(),torch.ones_like(times)],dim=-1))
    a=model(events,position=position); b=model(events.flip(1),position=position)
    assert not torch.allclose(a,b,atol=1e-6)
    a.square().mean().backward(); assert clock.weight.grad.abs().sum()>0 and events.grad.abs().sum()>0
