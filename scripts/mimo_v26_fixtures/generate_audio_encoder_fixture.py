from pathlib import Path
import importlib.util,sys,json,hashlib
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from fixture_paths import paths
out, source = paths('audio-encoder')
spec=importlib.util.spec_from_file_location('audio_reference',source);a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
cfg=dict(audio_channels=3,group_size=4,input_local_dim=16,input_local_layers=2,input_local_attn_heads=4,input_local_head_dim=4,input_local_intermediate_size=32,input_full_attention=True,out_hidden_size=16,rope_theta=640000,partial_rotary_factor=1.0,projection_layers=2,add_post_norm=True,audio_segment_size=6000,speech_vocab_size='16',speech_zeroemb_idx='15')
config_path=out/'audio-encoder-config.json';config_path.write_text(json.dumps(cfg,indent=2)+'\n')
c=a.AudioEncoderConfig(**{k:v for k,v in cfg.items() if k not in ('speech_vocab_size','speech_zeroemb_idx')});c.speech_vocab_size=[16]*3;c.speech_zeroemb_idx=[15]*3
model=a.MiMoAudioEncoderMLX(c);weights={}
for name,x in tree_flatten(model.parameters()):
    v=((np.arange(x.size,dtype=np.float32)+sum(name.encode())%19)%29-14)*np.float32(.02)
    if 'norm' in name: v=v*.1+1
    weights[name]=mx.array(v.reshape(x.shape)).astype(mx.bfloat16)
model.load_weights(list(weights.items()),strict=True)
codes=mx.array((np.arange(21,dtype=np.int32)%16).reshape(7,3))
expected=model(codes);mx.eval(expected)
arrays={'weight.'+k:v for k,v in weights.items()};arrays.update(codes=codes,expected=expected)
mx.save_safetensors(str(out/'audio-encoder-reference.safetensors'),arrays)
(out/'audio-encoder-reference.json').write_text(json.dumps(dict(reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),dtype='bfloat16',output_shape=list(expected.shape)),indent=2)+'\n')
print((out/'audio-encoder-reference.json').read_text())
