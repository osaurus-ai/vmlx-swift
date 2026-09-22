from pathlib import Path
import importlib.util,sys,json,hashlib
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from fixture_paths import paths
# The production encoder/RVQ path uses precise CPU F32. Avoid generating
# hardware-dependent TF32 golden features from a GPU default stream.
mx.set_default_device(mx.cpu)
out, source = paths('audio-tokenizer')
spec=importlib.util.spec_from_file_location('audio_reference',source);a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
cfg=dict(d_model=16,encoder_layers=3,encoder_attention_heads=4,encoder_ffn_dim=32,encoder_skip_layer_id=2,encoder_causal=True,encoder_attn_window_size=[2,0],hybrid_attention=True,swa_per_block=2,kernel_size=3,stride_size=2,avg_pooler=2,rope_theta=10000,n_mels=4,num_quantizers=3,codebook_size=[8,8,8],ln_type='LayerNorm',activation_function='gelu')
(out/'audio-tokenizer-config.json').write_text(json.dumps(cfg,indent=2)+'\n')
model=a.AudioTokenizerEncoderMLX(a.AudioTokenizerConfig(**cfg));weights={}
for name,x in tree_flatten(model.parameters()):
    if name.startswith('codebooks'): continue
    v=((np.arange(x.size,dtype=np.float32)+sum(name.encode())%23)%41-20)*np.float32(.012)
    if 'norm' in name and name.endswith('weight'): v=v*.1+1
    weights[name]=mx.array(v.reshape(x.shape))
model.load_weights(list(weights.items()),strict=False)
rng=np.random.default_rng(260923)
model.codebooks=[mx.array(rng.normal(0,.6,(8,16)).astype(np.float32)) for _ in range(3)]
mels=[rng.normal(0,1,(n,4)).astype(np.float32) for n in (11,16)]
features=model.features(mels);codes=model.tokenize(mels);mx.eval(features,codes)
arrays={'weight.'+k:v for k,v in weights.items()}
for i,book in enumerate(model.codebooks): arrays[f'weight.codebooks.{i}.weight']=book
for i in range(2): arrays.update({f'mel{i}':mx.array(mels[i]),f'features{i}':features[i],f'codes{i}':codes[i]})
mx.save_safetensors(str(out/'audio-tokenizer-reference.safetensors'),arrays)
(out/'audio-tokenizer-reference.json').write_text(json.dumps(dict(reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),dtype='float32',device='cpu',feature_shapes=[list(x.shape) for x in features],codes=[np.array(x).tolist() for x in codes]),indent=2)+'\n')
print((out/'audio-tokenizer-reference.json').read_text())
