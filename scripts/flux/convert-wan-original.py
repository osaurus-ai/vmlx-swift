#!/usr/bin/env python3
"""Convert original Wan 2.1 pickle components into the native safetensors layout.

No remote code is loaded. Input is an already-downloaded original Wan directory;
output must be new. Transformer values, names, dtypes and axes are preserved.
Requires torch, safetensors and a local UMT5 tokenizer folder.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.source / "config.json").read_text())
    if config.get("model_type") not in ("t2v", "i2v"):
        raise ValueError("Only original Wan 2.1 t2v/i2v configurations are supported")
    if args.output.exists():
        raise ValueError("Output must be a new directory; existing bundles are never overwritten")
    if not (args.tokenizer / "tokenizer.json").is_file():
        raise ValueError("A complete local UMT5 tokenizer directory is required")
    args.output.mkdir(parents=True)
    shutil.copy2(args.source / "config.json", args.output / "config.json")
    transformer = args.output / "transformer"
    transformer.mkdir()
    shards = sorted(args.source.glob("diffusion_pytorch_model*.safetensors"))
    if not shards:
        raise ValueError("No original transformer safetensor shards found")
    for shard in shards:
        shutil.copy2(shard, transformer / shard.name)
    components = [("models_t5_umt5-xxl-enc-bf16.pth", "text_encoder"), ("Wan2.1_VAE.pth", "vae")]
    if config["model_type"] == "i2v":
        components.append(("models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", "image_encoder"))
    for filename, component in components:
        tensors = torch.load(args.source / filename, map_location="cpu", weights_only=True)
        if not isinstance(tensors, dict) or not all(isinstance(t, torch.Tensor) for t in tensors.values()):
            raise ValueError(f"{filename} is not a plain tensor state dictionary")
        destination = args.output / component
        destination.mkdir()
        save_file({key: value.contiguous().clone() for key, value in tensors.items()}, str(destination / "model.safetensors"))
        del tensors
    shutil.copytree(args.tokenizer, args.output / "tokenizer")
    (args.output / "wan_native_format.json").write_text(json.dumps({
        "format": "wan-original-safetensors-v1", "source": args.source.name,
        "convolution_axes": "out,in,time,height,width", "scheduler": "flow-euler",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
