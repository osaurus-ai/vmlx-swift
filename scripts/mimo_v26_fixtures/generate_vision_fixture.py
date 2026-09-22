import importlib.util, sys, json, hashlib
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
from fixture_paths import paths
root, source = paths('vision')
spec=importlib.util.spec_from_file_location('mimo_reference',source)
v=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v;spec.loader.exec_module(v)
cfg=dict(depth=4,hidden_size=16,intermediate_size=32,num_heads=4,num_key_value_heads=2,qk_channels=4,out_hidden_size=16,patch_size=2,temporal_patch_size=2,spatial_merge_size=2,in_channels=3,rms_norm_eps=1e-6,fullatt_block_indexes=[0,3],vit_window_attn_types=[-1,0,1,-1],visual_token_window_size=2,use_sink=True)
tower=v.MiMoVisionTower(v.MiMoV26VisionConfig.from_dict(cfg))
weights={}
for name,a in tree_flatten(tower.parameters()):
    phase=sum(name.encode())%17
    values=((np.arange(a.size,dtype=np.float32)+phase)%31-15)*np.float32(.011)
    if 'norm' in name or 'ln_q' in name: values=values*np.float32(.1)+1
    if name.endswith('sinks'): values=np.arange(a.size,dtype=np.float32)*.7-1
    weights[name]=mx.array(values.reshape(a.shape))
tower.load_weights(list(weights.items()),strict=True)
grid=np.array([[1,4,6],[2,2,4]])
pixels=mx.array((np.arange(40*24,dtype=np.float32)%53-26).reshape(40,24)*.04)
out=tower(pixels,grid);mx.eval(out)
# Preserve the default-GPU reference and independently evaluate the same
# reference implementation under CPU F32. M5 default GPU GEMM may use TF32;
# strict-F32 and non-TF32 devices must not compare against that rounded golden.
with mx.stream(mx.cpu):
    out_f32=tower(pixels,grid)
    mx.eval(out_f32)
cos,sin=v.rotary_cos_sin(grid,4,2)
raw=(np.arange(3*5*7,dtype=np.float32)%251).reshape(1,3,5,7)
resized=v.resize_bilinear(raw,8,12)
normalized=(resized-v.PIXEL_MEAN[None,:,None,None])/v.PIXEL_STD[None,:,None,None]
patches,patch_grid=v.flatten_patches(np.repeat(normalized,2,axis=0),2,2,2)
arrays={'weight.'+k:a for k,a in weights.items()}
arrays.update(pixels=pixels,expected=out,expected_f32=out_f32,cosine=mx.array(cos),sine=mx.array(sin),columns=mx.array(v.window_index_col(grid)),raw=mx.array(raw),resized=mx.array(resized),patches=mx.array(patches))
mx.save_safetensors(str(root/'vision-reference.safetensors'),arrays)
(root/'vision-config.json').write_text(json.dumps(cfg,indent=2)+'\n')
(root/'vision-reference.json').write_text(json.dumps(dict(reference_sha256=hashlib.sha256((source).read_bytes()).hexdigest(),grid=grid.tolist(),patch_grid=list(patch_grid),mlx_version=mx.__version__,generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),output_shape=list(out.shape),reference_math=dict(expected="default_gpu",expected_f32="cpu_f32"),gpu_device=mx.metal.device_info()),indent=2)+'\n')
print((root/'vision-reference.json').read_text())
