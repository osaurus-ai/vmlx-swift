#!/usr/bin/env python3
"""Convert Nemotron-Labs-Audex-2B to a vMLX affine bundle.

The dense text decoder is quantized; NV-Whisper and the audio projector stay
in their source precision so the first correctness bundle does not trade audio
quality for size. The result remains loadable by the native Audex VLM class.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx


def should_quantize(name: str, value: mx.array) -> bool:
    if value.ndim != 2:
        return False
    if not (name.startswith("model.") or name == "lm_head.weight"):
        return False
    return not name.endswith("norm.weight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shard_names = sorted(set(index["weight_map"].values()))
    output_map: dict[str, str] = {}
    total_size = 0

    for shard_name in shard_names:
        source_shard = source / shard_name
        weights = mx.load(str(source_shard))
        converted: dict[str, mx.array] = {}
        for name, value in weights.items():
            if should_quantize(name, value):
                weight, scales, biases = mx.quantize(
                    value, group_size=args.group_size, bits=args.bits)
                converted[name] = weight
                converted[name.removesuffix(".weight") + ".scales"] = scales
                converted[name.removesuffix(".weight") + ".biases"] = biases
            else:
                converted[name] = value

        destination = output / shard_name
        mx.save_safetensors(str(destination), converted, metadata={"format": "mlx"})
        total_size += destination.stat().st_size
        for name in converted:
            output_map[name] = shard_name
        print(f"wrote {destination.name}: {destination.stat().st_size / 2**30:.3f} GiB")

    for item in source.iterdir():
        if item.name.startswith("model-") and item.suffix == ".safetensors":
            continue
        if item.name == "model.safetensors.index.json":
            continue
        destination = output / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)

    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization"] = {
        "group_size": args.group_size,
        "bits": args.bits,
        "mode": "affine",
    }
    config["weight_format"] = "affine"
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    output_index = {
        "metadata": {
            "total_size": total_size,
            "source_model": "nvidia/Nemotron-Labs-Audex-2B",
            "audio_encoder_precision": "source",
            "audio_projector_precision": "source",
        },
        "weight_map": dict(sorted(output_map.items())),
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(output_index, indent=2) + "\n"
    )
    print(f"complete: {output} ({total_size / 2**30:.3f} GiB weights)")


if __name__ == "__main__":
    main()
