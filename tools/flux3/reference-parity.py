#!/usr/bin/env python3
"""Run the pinned local mflux reference or compare its tensors/PNG with Swift.

Use a fresh process for each run. No model downloads, server calls or sampler
changes. The same prompt, seed, dimensions, steps and ordered references must be
supplied to Flux3KleinRuntimeTests. Omitted edit size follows mflux reference size.
"""
import argparse
import json
import time
from pathlib import Path


def compare(args):
    import numpy as np
    from PIL import Image
    swift = json.loads((args.swift / 'swift-result.json').read_text())
    reference = json.loads((args.out / 'mflux-result.json').read_text())
    for key in ['prompt', 'seed', 'steps', 'width', 'height', 'references']:
        if swift[key] != reference[key]:
            raise ValueError(f'Cannot compare different {key}: {swift[key]} vs {reference[key]}')
    result = {}
    for name in ['noise', 'text', 'references', 'step0-velocity', 'latents']:
        lhs, rhs = args.swift / f'swift-{name}.safetensors', args.out / f'mflux-{name}.safetensors'
        if not lhs.exists() and not rhs.exists():
            continue
        def read(path):
            data = path.read_bytes()
            n = int.from_bytes(data[:8], 'little')
            header = json.loads(data[8:8+n])[name]
            start, end = header['data_offsets']
            payload = data[8+n+start:8+n+end]
            if header['dtype'] == 'BF16':
                values = (np.frombuffer(payload, dtype='<u2').astype(np.uint32) << 16).view(np.float32)
            elif header['dtype'] == 'F32':
                values = np.frombuffer(payload, dtype='<f4')
            else:
                raise ValueError(f"Unexpected tensor dtype: {header['dtype']}")
            return values.reshape(header['shape'])
        a, b = read(lhs), read(rhs)
        if a.shape != b.shape:
            raise ValueError(f'{name} shape mismatch: {a.shape} vs {b.shape}')
        delta = np.abs(a - b)
        result[name] = {'shape': list(a.shape), 'mean_abs': float(delta.mean()), 'max_abs': float(delta.max()),
                        'finite': bool(np.isfinite(a).all() and np.isfinite(b).all())}
    a, b = np.asarray(Image.open(swift['png']).convert('RGB'), dtype=np.float32), np.asarray(Image.open(reference['png']).convert('RGB'), dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f'PNG shape mismatch: {a.shape} vs {b.shape}')
    result['pixels_rgb8'] = {'mean_abs': float(np.abs(a-b).mean()), 'max_abs': float(np.abs(a-b).max())}
    (args.swift / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def generate(args):
    import mlx.core as mx
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux3.variants import Flux3Klein, Flux3KleinEdit
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cls = Flux3KleinEdit if args.reference else Flux3Klein
    model = cls(model_config=ModelConfig.flux3_image_klein(), model_path=str(args.bundle))
    load_seconds = time.perf_counter() - started
    def save(name, value):
        mx.save_safetensors(str(args.out / f'mflux-{name}.safetensors'), {name: value})
    original_encode = model._encode_prompt
    def encode(prompt):
        text, ids = original_encode(prompt)
        save('text', text)
        return text, ids
    model._encode_prompt = encode
    original_predict = model._predict
    def predictor(transformer):
        forward = original_predict(transformer)
        first = True
        def predict(**kwargs):
            nonlocal first
            velocity = forward(**kwargs)
            if first:
                save('noise', kwargs['latents'])
                save('step0-velocity', velocity)
                if kwargs['cond_tokens'] is not None:
                    save('references', kwargs['cond_tokens'])
                first = False
            return velocity
        return predict
    model._predict = predictor
    original_decode = model._decode_latents
    def decode(latents, h, w):
        save('latents', latents)
        return original_decode(latents, h, w)
    model._decode_latents = decode
    operation_start = time.perf_counter()
    kwargs = dict(seed=args.seed, prompt=args.prompt, num_inference_steps=args.steps)
    if args.reference:
        kwargs.update(image_paths=[str(p) for p in args.reference], width=args.width, height=args.height)
    else:
        kwargs.update(width=args.width or 512, height=args.height or 512)
    image = model.generate_image(**kwargs)
    png = args.out / 'mflux.png'
    image.save(str(png))
    seconds = time.perf_counter() - operation_start
    record = dict(bundle=str(args.bundle), prompt=args.prompt, seed=args.seed, steps=args.steps,
                  width=image.image.width, height=image.image.height, references=[str(p) for p in args.reference],
                  load_seconds=load_seconds, operation_seconds=seconds, images_per_second=1/seconds, png=str(png))
    (args.out / 'mflux-result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    del model, image
    mx.clear_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=Path('/Volumes/EricMLWork/models/image/staging/FLUX-3-Image-Klein-mflux-4bit'))
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--swift', type=Path, help='Compare with this Swift output directory instead of running a model')
    p.add_argument('--prompt', default='A red ceramic teapot on a wooden table, soft morning window light, detailed photograph')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--width', type=int)
    p.add_argument('--height', type=int)
    p.add_argument('--reference', type=Path, action='append', default=[])
    args = p.parse_args()
    compare(args) if args.swift else generate(args)


if __name__ == '__main__':
    main()
