from pathlib import Path
import importlib.util, sys, json, hashlib
import numpy as np
import mlx.core as mx
from fixture_paths import paths
out, source = paths('audio-features')
spec=importlib.util.spec_from_file_location('audio_reference',source)
a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
rng=np.random.default_rng(260922)
t=np.arange(2205)/22050
raw=(.3*np.sin(2*np.pi*437*t)+.15*np.cos(2*np.pi*1789*t)+.01*rng.standard_normal(len(t))).astype(np.float32)
resampled=a.resample(raw,22050,24000)
mel=a.log_mel(resampled)
mx.save_safetensors(str(out/'audio-features-reference.safetensors'),dict(raw=mx.array(raw),resampled=mx.array(resampled),mel=mx.array(mel)))
config=dict(sampling_rate=24000,nfft=960,hop_length=240,window_size=960,n_mels=128,fmin=0,fmax=None)
(out/'audio-features-config.json').write_text(json.dumps(config,indent=2)+'\n')
(out/'audio-features-reference.json').write_text(json.dumps(dict(reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_rate=22050,input_samples=len(raw),resampled_samples=len(resampled),mel_shape=list(mel.shape)),indent=2)+'\n')
print((out/'audio-features-reference.json').read_text())
