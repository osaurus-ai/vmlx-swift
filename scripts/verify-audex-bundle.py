#!/usr/bin/env python3
"""Validate an Audex affine bundle without loading model weights into RAM."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path


SUPPORTED_MODEL_TYPES = {"nemotron_dense_audex", "nemotron_h_audex"}


def safetensors_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"invalid safetensors header: {path}")
        header_size = struct.unpack("<Q", raw_size)[0]
        header = json.loads(handle.read(header_size))
    return header, 8 + header_size


def tensor_signature(path: Path, header: dict, data_start: int, key: str) -> tuple:
    descriptor = header[key]
    start, end = descriptor["data_offsets"]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(data_start + start)
        remaining = end - start
        while remaining:
            chunk = handle.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"truncated tensor data for {key}: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return descriptor["dtype"], tuple(descriptor["shape"]), digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--expect-bits", type=int, choices=(4, 6, 8))
    parser.add_argument(
        "--source",
        type=Path,
        help="exact source snapshot used to prove raw audio tensor dtype/shape/content parity",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    config = json.loads((bundle / "config.json").read_text())
    conversion = json.loads((bundle / "vmlx_conversion.json").read_text())
    index = json.loads((bundle / "model.safetensors.index.json").read_text())
    model_type = config.get("model_type")
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise SystemExit(f"unsupported model_type: {model_type!r}")
    quantization = config.get("quantization", {})
    if quantization.get("mode") != "affine" or config.get("weight_format") != "affine":
        raise SystemExit("bundle does not declare affine quantization")
    if args.expect_bits is not None and quantization.get("bits") != args.expect_bits:
        raise SystemExit(
            f"expected {args.expect_bits} bits, config declares {quantization.get('bits')!r}"
        )
    if conversion.get("source_model_type") != model_type:
        raise SystemExit("conversion provenance model_type does not match config.json")
    if conversion.get("quantization", {}).get("audio_encoder") != "source_precision":
        raise SystemExit("audio encoder source-precision provenance is missing")
    if conversion.get("quantization", {}).get("audio_projector") != "source_precision":
        raise SystemExit("audio projector source-precision provenance is missing")

    weight_map: dict[str, str] = index.get("weight_map", {})
    if not weight_map:
        raise SystemExit("weight map is empty")
    shard_names = sorted(set(weight_map.values()))
    missing_shards = [name for name in shard_names if not (bundle / name).is_file()]
    if missing_shards:
        raise SystemExit(f"missing weight shards: {missing_shards}")
    incomplete_files = sorted(
        str(path.relative_to(bundle))
        for path in bundle.rglob("*")
        if path.is_file() and (path.name.endswith(".incomplete") or path.name.endswith(".lock"))
    )
    if incomplete_files:
        raise SystemExit(f"incomplete/lock files present: {incomplete_files}")

    actual_by_shard: dict[str, set[str]] = {}
    actual_headers: dict[str, tuple[dict, int]] = {}
    occurrences: Counter[str] = Counter()
    for shard_name in shard_names:
        header, data_start = safetensors_header(bundle / shard_name)
        keys = set(header).difference({"__metadata__"})
        actual_by_shard[shard_name] = keys
        actual_headers[shard_name] = (header, data_start)
        occurrences.update(keys)
    duplicate_keys = sorted(key for key, count in occurrences.items() if count != 1)
    if duplicate_keys:
        raise SystemExit(f"duplicate tensor keys across shards: {duplicate_keys[:20]}")
    actual_keys = set(occurrences)
    indexed_keys = set(weight_map)
    if actual_keys != indexed_keys:
        raise SystemExit(
            f"header/index mismatch: unindexed={sorted(actual_keys - indexed_keys)[:20]}, "
            f"missing={sorted(indexed_keys - actual_keys)[:20]}"
        )
    wrong_shard = sorted(
        key for key, shard_name in weight_map.items() if key not in actual_by_shard[shard_name]
    )
    if wrong_shard:
        raise SystemExit(f"weight map points at wrong shard: {wrong_shard[:20]}")

    shard_bytes = sum((bundle / name).stat().st_size for name in shard_names)
    if index.get("metadata", {}).get("total_size") != shard_bytes:
        raise SystemExit("index total_size does not match actual shard bytes")
    if conversion.get("weight_bytes") != shard_bytes:
        raise SystemExit("conversion weight_bytes does not match actual shard bytes")
    if conversion.get("weight_shards") != len(shard_names):
        raise SystemExit("conversion weight_shards does not match actual shard count")

    audio_keys = [
        key for key in actual_keys if key.startswith("audio_encoder.") or key.startswith("audio_projector.")
    ]
    if not any(key.startswith("audio_encoder.") for key in audio_keys):
        raise SystemExit("bundle has no audio_encoder tensors")
    if not any(key.startswith("audio_projector.") for key in audio_keys):
        raise SystemExit("bundle has no audio_projector tensors")
    quantized_audio = sorted(
        key for key in audio_keys if key.endswith(".scales") or key.endswith(".biases")
    )
    if quantized_audio:
        raise SystemExit(f"audio tensors were unexpectedly quantized: {quantized_audio[:20]}")

    audio_source_parity = "UNVERIFIED"
    if args.source is not None:
        source = args.source.resolve()
        source_index = json.loads((source / "model.safetensors.index.json").read_text())
        source_map: dict[str, str] = source_index.get("weight_map", {})
        missing_source_keys = sorted(set(audio_keys) - set(source_map))
        if missing_source_keys:
            raise SystemExit(f"source is missing audio tensors: {missing_source_keys[:20]}")
        source_headers: dict[str, tuple[dict, int]] = {}
        mismatches = []
        for key in sorted(audio_keys):
            output_shard = weight_map[key]
            output_header, output_data_start = actual_headers[output_shard]
            source_shard = source_map[key]
            if source_shard not in source_headers:
                source_headers[source_shard] = safetensors_header(source / source_shard)
            source_header, source_data_start = source_headers[source_shard]
            output_signature = tensor_signature(
                bundle / output_shard, output_header, output_data_start, key
            )
            source_signature = tensor_signature(
                source / source_shard, source_header, source_data_start, key
            )
            if output_signature != source_signature:
                mismatches.append(key)
        if mismatches:
            raise SystemExit(
                f"audio tensor dtype/shape/content differs from source: {mismatches[:20]}"
            )
        audio_source_parity = "PASS"

    scale_keys = {key for key in actual_keys if key.endswith(".scales")}
    bias_keys = {key for key in actual_keys if key.endswith(".biases")}
    quantized_bases = {key.removesuffix(".scales") for key in scale_keys}
    for base in quantized_bases:
        required = {f"{base}.weight", f"{base}.scales", f"{base}.biases"}
        if not required.issubset(actual_keys):
            raise SystemExit(f"incomplete quantized tensor triplet: {base}")
    if {key.removesuffix(".biases") for key in bias_keys} != quantized_bases:
        raise SystemExit("quantized scales/biases base sets do not match")

    expert_group_count = 0
    if model_type == "nemotron_h_audex":
        normalizations = conversion.get("config_normalizations", [])
        if config.get("time_step_limit") is not None:
            raise SystemExit("Nemotron-H output retained a non-portable time_step_limit")
        if not any(item.get("field") == "time_step_limit" for item in normalizations):
            raise SystemExit("Nemotron-H time_step_limit normalization provenance is missing")
        if any(".mixer.experts." in key for key in actual_keys):
            raise SystemExit("unstacked routed expert tensors remain in Nemotron-H output")
        expert_layers = config["hybrid_override_pattern"].count("E")
        expected_bases = {
            f"backbone.layers.{layer}.mixer.switch_mlp.{fc}"
            for layer, block in enumerate(config["hybrid_override_pattern"])
            if block == "E"
            for fc in ("fc1", "fc2")
        }
        missing_groups = sorted(expected_bases - quantized_bases)
        if missing_groups:
            raise SystemExit(f"missing stacked expert groups: {missing_groups}")
        expert_group_count = len(expected_bases)
        declared = conversion.get("conversion_counts", {}).get("affine_expert_groups")
        if declared != expert_group_count:
            raise SystemExit(
                f"declared expert group count {declared} does not match {expert_group_count}"
            )

    report = {
        "status": "PASS",
        "bundle": str(bundle),
        "model_type": model_type,
        "source_model": conversion["source_model"],
        "source_revision": conversion["source_revision"],
        "quantization": quantization,
        "weight_shards": len(shard_names),
        "weight_bytes": shard_bytes,
        "tensor_count": len(actual_keys),
        "quantized_module_count": len(quantized_bases),
        "stacked_expert_group_count": expert_group_count,
        "audio_tensor_count": len(audio_keys),
        "audio_precision": "source",
        "audio_source_parity": audio_source_parity,
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
