#!/usr/bin/env python3
"""Create reproducible affine vMLX bundles for Nemotron-Labs-Audex.

The language decoder is quantized. NV-Whisper and the audio projector remain
in source precision. Nemotron-H routed experts are stacked before quantization
to match the native SwitchLinear layout used by vMLX and mlx-lm.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import mlx.core as mx


SUPPORTED_MODEL_TYPES = {"nemotron_dense_audex", "nemotron_h_audex"}
EXPERT_KEY = re.compile(
    r"^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\."
    r"(up_proj|down_proj)\.weight$"
)
PROJECTION_NAME = {"up_proj": "fc1", "down_proj": "fc2"}


def should_quantize(name: str, value: mx.array, model_type: str) -> bool:
    if value.ndim != 2 or not name.endswith(".weight"):
        return False
    if model_type == "nemotron_dense_audex":
        return (name.startswith("model.") or name == "lm_head.weight") and not name.endswith(
            "norm.weight"
        )
    if not (name.startswith("backbone.") or name == "lm_head.weight"):
        return False
    if name.endswith("norm.weight") or ".norm_f.weight" in name:
        return False
    if name.endswith(".mixer.gate.weight"):
        return False
    return not any(marker in name for marker in (".A_log", ".D", ".dt_bias", "conv1d."))


class ShardWriter:
    def __init__(self, output: Path, max_bytes: int) -> None:
        self.output = output
        self.max_bytes = max_bytes
        self.buffer: dict[str, mx.array] = {}
        self.buffer_bytes = 0
        self.shard_count = 0
        self.temporary_map: dict[str, str] = {}

    def add(self, tensors: dict[str, mx.array]) -> None:
        incoming_bytes = sum(value.nbytes for value in tensors.values())
        if self.buffer and self.buffer_bytes + incoming_bytes > self.max_bytes:
            self.flush()
        duplicate = set(self.buffer).intersection(tensors)
        if duplicate:
            raise RuntimeError(f"duplicate output tensor(s): {sorted(duplicate)}")
        self.buffer.update(tensors)
        self.buffer_bytes += incoming_bytes
        if self.buffer_bytes >= self.max_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        self.shard_count += 1
        name = f"model-{self.shard_count:05d}-of-XXXXX.safetensors"
        path = self.output / name
        mx.save_safetensors(str(path), self.buffer, metadata={"format": "mlx"})
        for key in self.buffer:
            self.temporary_map[key] = name
        print(
            f"wrote {name}: {len(self.buffer)} tensors, "
            f"{path.stat().st_size / 2**30:.3f} GiB",
            flush=True,
        )
        self.buffer = {}
        self.buffer_bytes = 0
        gc.collect()
        mx.clear_cache()

    def finish(self) -> dict[str, str]:
        self.flush()
        final_map: dict[str, str] = {}
        for key, temporary_name in self.temporary_map.items():
            final_name = temporary_name.replace("XXXXX", f"{self.shard_count:05d}")
            final_map[key] = final_name
        for index in range(1, self.shard_count + 1):
            old = self.output / f"model-{index:05d}-of-XXXXX.safetensors"
            new = self.output / f"model-{index:05d}-of-{self.shard_count:05d}.safetensors"
            old.rename(new)
        return final_map


def quantize_tensor(name: str, value: mx.array, bits: int, group_size: int) -> dict[str, mx.array]:
    weight, scales, biases = mx.quantize(value, group_size=group_size, bits=bits)
    mx.eval(weight, scales, biases)
    base = name.removesuffix(".weight")
    return {
        f"{base}.weight": weight,
        f"{base}.scales": scales,
        f"{base}.biases": biases,
    }


def copy_support_files(source: Path, output: Path, source_shards: set[str]) -> None:
    for item in source.iterdir():
        if item.name in source_shards or item.name == "model.safetensors.index.json":
            continue
        if item.name == ".cache":
            continue
        destination = output / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bits", type=int, choices=(4, 6, 8), default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--max-shard-bytes", type=int, default=1_000_000_000)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    config_path = source / "config.json"
    index_path = source / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise SystemExit(f"source must contain config.json and model.safetensors.index.json: {source}")
    config = json.loads(config_path.read_text())
    model_type = config.get("model_type")
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise SystemExit(f"unsupported Audex model_type: {model_type!r}")

    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    source_shards = set(weight_map.values())
    missing_shards = sorted(name for name in source_shards if not (source / name).is_file())
    if missing_shards:
        raise SystemExit(f"source is incomplete; missing shards: {missing_shards}")

    writer = ShardWriter(output, args.max_shard_bytes)
    expert_groups: dict[tuple[int, str], dict[int, tuple[str, str]]] = defaultdict(dict)
    for name, shard_name in weight_map.items():
        match = EXPERT_KEY.match(name)
        if match:
            layer, expert, projection = match.groups()
            expert_groups[(int(layer), projection)][int(expert)] = (name, shard_name)

    affine_simple = 0
    passthrough = 0
    dropped = 0
    for shard_name in sorted(source_shards):
        weights = mx.load(str(source / shard_name))
        for name in sorted(weights):
            value = weights[name]
            if EXPERT_KEY.match(name):
                continue
            if name.startswith("mtp.") or name.endswith(".importance"):
                dropped += 1
                continue
            if should_quantize(name, value, model_type):
                writer.add(quantize_tensor(name, value, args.bits, args.group_size))
                affine_simple += 1
            else:
                writer.add({name: value})
                passthrough += 1
        del weights
        gc.collect()

    affine_expert_groups = 0
    if model_type == "nemotron_h_audex":
        expert_count = int(config["n_routed_experts"])
        if not expert_groups:
            raise SystemExit("Nemotron-H source has no routed expert tensors")
        for (layer, projection), group in sorted(expert_groups.items()):
            expected = set(range(expert_count))
            if set(group) != expected:
                missing = sorted(expected.difference(group))
                raise SystemExit(
                    f"layer {layer} {projection} is incomplete: "
                    f"{len(group)}/{expert_count}, missing={missing}"
                )
            by_shard: dict[str, list[tuple[int, str]]] = defaultdict(list)
            for expert, (name, shard_name) in group.items():
                by_shard[shard_name].append((expert, name))
            experts: dict[int, mx.array] = {}
            for shard_name, entries in sorted(by_shard.items()):
                weights = mx.load(str(source / shard_name))
                for expert, name in entries:
                    experts[expert] = weights[name]
                del weights
            stacked = mx.stack([experts[expert] for expert in range(expert_count)], axis=0)
            fc_name = PROJECTION_NAME[projection]
            destination = f"backbone.layers.{layer}.mixer.switch_mlp.{fc_name}.weight"
            writer.add(quantize_tensor(destination, stacked, args.bits, args.group_size))
            affine_expert_groups += 1
            del experts, stacked
            gc.collect()

    output_map = writer.finish()
    copy_support_files(source, output, source_shards)

    output_config_path = output / "config.json"
    output_config = json.loads(output_config_path.read_text())
    config_normalizations: list[dict[str, object]] = []
    time_step_limit = output_config.get("time_step_limit")
    if isinstance(time_step_limit, list) and any(
        isinstance(value, float) and not (-float("inf") < value < float("inf"))
        for value in time_step_limit
    ):
        output_config.pop("time_step_limit")
        config_normalizations.append(
            {
                "field": "time_step_limit",
                "source_value": ["0.0", "Infinity"],
                "output": "omitted",
                "runtime_semantics": "NemotronHConfiguration defaults to [0.001, +infinity]",
                "reason": "strict JSON does not permit non-finite numeric literals",
            }
        )
    output_config["quantization"] = {
        "group_size": args.group_size,
        "bits": args.bits,
        "mode": "affine",
    }
    output_config["weight_format"] = "affine"
    output_config_path.write_text(json.dumps(output_config, indent=2) + "\n")

    final_shards = sorted(output.glob("model-*-of-*.safetensors"))
    total_size = sum(path.stat().st_size for path in final_shards)
    output_index = {
        "metadata": {
            "total_size": total_size,
            "source_model": args.source_model,
            "source_revision": args.source_revision,
            "audio_encoder_precision": "source",
            "audio_projector_precision": "source",
        },
        "weight_map": dict(sorted(output_map.items())),
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(output_index, indent=2) + "\n"
    )
    conversion = {
        "format": "vmlx-affine-audex",
        "source_model": args.source_model,
        "source_revision": args.source_revision,
        "source_model_type": model_type,
        "quantization": {
            "method": "affine",
            "bits": args.bits,
            "group_size": args.group_size,
            "language_decoder": "quantized",
            "audio_encoder": "source_precision",
            "audio_projector": "source_precision",
        },
        "conversion_counts": {
            "affine_simple_tensors": affine_simple,
            "affine_expert_groups": affine_expert_groups,
            "passthrough_tensors": passthrough,
            "dropped_runtime_unused_tensors": dropped,
        },
        "weight_shards": len(final_shards),
        "weight_bytes": total_size,
        "config_normalizations": config_normalizations,
    }
    (output / "vmlx_conversion.json").write_text(json.dumps(conversion, indent=2) + "\n")
    print(
        f"complete: {output} ({total_size / 2**30:.3f} GiB weights, "
        f"{affine_simple} simple affine tensors, "
        f"{affine_expert_groups} stacked expert groups)",
        flush=True,
    )


if __name__ == "__main__":
    main()
