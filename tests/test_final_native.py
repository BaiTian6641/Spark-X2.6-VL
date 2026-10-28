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


def tiny_talker():
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration
    from tokenizers import Tokenizer,models,pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    cfg=Qwen3TTSTalkerConfig(vocab_size=2200,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,text_vocab_size=128,text_hidden_size=64,num_code_groups=16,rope_scaling={'rope_type':'default','mrope_section':[2,2,4],'interleaved':True},codec_eos_token_id=2150,codec_think_id=2154,codec_think_bos_id=2156,codec_think_eos_id=2157,codec_pad_id=2148,codec_bos_id=2149,codec_language_id={'english':2050,'chinese':2055},code_predictor_config={'vocab_size':2048,'hidden_size':64,'intermediate_size':128,'num_hidden_layers':2,'num_attention_heads':4,'num_key_value_heads':2,'head_dim':16,'num_code_groups':16})
    cfg._attn_implementation='sdpa'
    wrapper=NativeSpeechOutput.__new__(NativeSpeechOutput); nn.Module.__init__(wrapper)
    wrapper.talker=Qwen3TTSTalkerForConditionalGeneration(cfg)
    wrapper.qconfig=types.SimpleNamespace(tts_pad_token_id=10,tts_bos_token_id=11,tts_eos_token_id=12)
    raw=Tokenizer(models.WordLevel({'[UNK]':0,'hi':1},unk_token='[UNK]'));raw.pre_tokenizer=pre_tokenizers.Whitespace()
    wrapper.tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]')
    wrapper.bridge=Resampler(48,64,4);wrapper.default_voice=nn.Parameter(torch.zeros(1,1,64))
    return wrapper


def test_native_talker_and_subtalker_objectives_reach_conditioning():
    model=tiny_talker(); hidden=torch.randn(1,7,48,requires_grad=True)
    prefix,pad=model.conditioning(hidden,{'language':'en','prompt':'Say hi','utterance':'hi'})
    codes=torch.randint(0,32,(3,16));embeds=torch.cat([prefix,model.frame_embeddings(codes,pad)],dim=1)
    output=model.talker.model(inputs_embeds=embeds,use_cache=False,return_dict=True).last_hidden_state[:,prefix.shape[1]-1:]
    labels=torch.cat([codes[:,0],torch.tensor([2150])])
    first=torch.nn.functional.cross_entropy(model.talker.codec_head(output).float().flatten(0,1),labels)
    _,rest=model.talker.forward_sub_talker_finetune(codes,output[0,:3]);(first+.3*rest).backward()
    assert torch.isfinite(hidden.grad).all() and hidden.grad.abs().sum()>0
    assert model.talker.code_predictor.lm_head[0].weight.grad.abs().sum()>0
    assert model.talker.model.layers[0].self_attn.q_proj.weight.grad.abs().sum()>0


def test_native_dit_v_prediction_reaches_spark_conditioning():
    from diffusers import StableAudioDiTModel
    from diffusers.pipelines.stable_audio.modeling_stable_audio import StableAudioProjectionModel
    model=NativeSoundOutput.__new__(NativeSoundOutput);nn.Module.__init__(model)
    model.dit=StableAudioDiTModel(sample_size=16,in_channels=8,out_channels=8,num_layers=2,attention_head_dim=16,num_attention_heads=4,num_key_value_attention_heads=2,cross_attention_dim=32,cross_attention_input_dim=32,time_proj_dim=16,global_states_input_dim=64)
    model.projection=StableAudioProjectionModel(text_encoder_dim=32,conditioning_dim=32,min_value=0,max_value=512)
    model.bridge=Resampler(48,32,4)
    hidden=torch.randn(1,7,48,requires_grad=True);z=torch.randn(1,8,16);noise=torch.randn_like(z);t=torch.tensor([.5]);a=torch.cos(t*torch.pi/2)[:,None,None];s=torch.sin(t*torch.pi/2)[:,None,None]
    cross,glob=model.conditioning(hidden,15);out=model.velocity(a*z+s*noise,t,cross,glob)
    torch.nn.functional.mse_loss(out,a*noise-s*z).backward()
    assert torch.isfinite(hidden.grad).all() and hidden.grad.abs().sum()>0
    assert model.dit.proj_out.weight.grad.abs().sum()>0

def tiny_original_spark(adapt=True):
    """The pinned custom decoder source, dimension-reduced for CPU graph tests."""
    import hashlib,importlib,sys,types
    root=Path('models/Spark-X2.6-4B').resolve(); lock=json.load(open('colab/final/components.lock.json'))
    for name in ('configuration_spark.py','modeling_spark.py'):
        payload=(root/name).read_bytes(); expected=next(f for f in lock['spark']['files'] if f['path']==name)
        assert hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()==expected['git_blob_sha1']
    package=types.ModuleType('spark_formal_tiny');package.__path__=[str(root)];sys.modules['spark_formal_tiny']=package
    config=importlib.import_module('spark_formal_tiny.configuration_spark').Spark2_5Config(vocab_size=128,hidden_size=48,intermediate_size=96,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=12,max_position_embeddings=128,pad_token_id=0,tie_word_embeddings=True,headwise_attn_output_gate=True,sliding_window=8,layer_types=['sliding_attention','full_attention'],rope_parameters={'sliding_attention':{'rope_theta':10000,'partial_rotary_factor':1.0},'full_attention':{'rope_theta':10000,'partial_rotary_factor':1.0}})
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
    prefix,pad=talker.conditioning(h,{'language':'en','prompt':'Say hi','utterance':'hi'});codes=torch.randint(0,32,(3,16))
    embeds=torch.cat([prefix,talker.frame_embeddings(codes,pad)],dim=1)
    out=talker.talker.model(inputs_embeds=embeds,use_cache=False,return_dict=True).last_hidden_state[:,prefix.shape[1]-1:]
    first=torch.nn.functional.cross_entropy(talker.talker.codec_head(out).flatten(0,1),torch.cat([codes[:,0],torch.tensor([2150])]))
    _,sub=talker.talker.forward_sub_talker_finetune(codes,out[0,:3]);(first+.3*sub).backward()
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

def test_temporal_resampler_keeps_event_order_when_absolute_clock_is_fixed():
    torch.manual_seed(99)
    model=Resampler(16,32,4); clock=nn.Linear(4,32)
    events=torch.randn(1,5,16,requires_grad=True); times=torch.arange(5.).reshape(1,5,1)
    position=clock(torch.cat([times,times.sin(),times.cos(),torch.ones_like(times)],dim=-1))
    a=model(events,position=position); b=model(events.flip(1),position=position)
    assert not torch.allclose(a,b,atol=1e-6)
    a.square().mean().backward(); assert clock.weight.grad.abs().sum()>0 and events.grad.abs().sum()>0
