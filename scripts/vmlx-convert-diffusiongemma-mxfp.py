#!/usr/bin/env python3
"""Convert DiffusionGemma BF16 source to native MLX MXFP4/MXFP8 bundles.

This is a first-party converter for google/diffusiongemma-26B-A4B-it. It does
not emit old affine 4-bit weights. Quantized tensors are produced with
``mx.quantize(..., mode="mxfp4"|"mxfp8")`` and stored as MLX ``weight`` /
``scales`` companions.

Dry-run mode intentionally avoids importing MLX so source/policy checks can run
in the system Python. Real conversion requires a Python environment with MLX.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - cosmetic fallback
    tqdm = None


MAX_SHARD_BYTES = 1_000_000_000
DEFAULT_SOURCE = Path("/Users/eric/models/google/diffusiongemma-26B-A4B-it")
DEFAULT_OUTPUT_ROOT = Path("/Users/eric/models/OsaurusAI")
TARGET_BASE_NAME = "diffusiongemma-26B-A4B-it"

SIDECAR_FILES = [
    "README.md",
    "LICENSE",
    "chat_template.jinja",
    "chat_template.json",
    "generation_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


@dataclass(frozen=True)
class QuantPolicy:
    method: str  # "mxfp" | "passthrough" | "skip"
    bits: int
    role: str
    reason: str


@dataclass(frozen=True)
class TensorItem:
    name: str
    shape: tuple[int, ...]
    shard: Path
    dtype: str | None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    return header, 8 + header_size


def _load_bf16_tensor(path: Path, tensor_name: str, shape: tuple[int, ...]) -> np.ndarray:
    header, data_start = _read_safetensors_header(path)
    info = header[tensor_name]
    dtype = str(info.get("dtype", "")).upper()
    if dtype != "BF16":
        raise TypeError(f"manual loader only handles BF16, got {dtype} for {tensor_name}")
    start, end = info["data_offsets"]
    with path.open("rb") as f:
        f.seek(data_start + start)
        raw = f.read(end - start)
    values = np.frombuffer(raw, dtype="<u2").copy()
    fp32_bits = values.astype(np.uint32) << 16
    return fp32_bits.view(np.float32).reshape(shape)


def _load_tensor(path: Path, tensor_name: str, shape: tuple[int, ...]) -> np.ndarray:
    with safe_open(str(path), framework="numpy") as f:
        try:
            tensor = f.get_tensor(tensor_name)
            if not isinstance(tensor, np.ndarray):
                tensor = np.array(tensor)
        except Exception:
            tensor = _load_bf16_tensor(path, tensor_name, shape)
    if tensor.dtype != np.float32:
        tensor = tensor.astype(np.float32)
    return tensor


def _scan_source(src: Path) -> list[TensorItem]:
    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing model.safetensors.index.json: {index_path}")

    index = _read_json(index_path)
    by_shard: dict[str, list[str]] = {}
    for key, shard in index.get("weight_map", {}).items():
        by_shard.setdefault(shard, []).append(key)

    items: list[TensorItem] = []
    for shard_name, keys in sorted(by_shard.items()):
        shard = src / shard_name
        if not shard.is_file():
            raise FileNotFoundError(f"missing source shard: {shard}")
        header, _ = _read_safetensors_header(shard)
        for key in sorted(keys):
            if key.endswith("_scale_inv"):
                continue
            info = header.get(key)
            if not isinstance(info, dict):
                raise KeyError(f"{key} missing from {shard}")
            items.append(
                TensorItem(
                    name=key,
                    shape=tuple(int(v) for v in info.get("shape", [])),
                    shard=shard,
                    dtype=info.get("dtype"),
                )
            )
    return items


def _is_norm_or_scalar_control(name: str) -> bool:
    low = name.lower()
    return (
        "norm" in low
        or name.endswith(".bias")
        or name.endswith(".layer_scalar")
        or name.endswith("router.scale")
        or name.endswith("router.per_expert_scale")
        or name.endswith("embed_scale")
        or name.endswith("pos_embedding")
    )


def _is_decoder_attention(name: str) -> bool:
    return (
        name.startswith("model.decoder.layers.")
        and ".self_attn." in name
        and name.endswith(".weight")
        and any(f".{proj}_proj.weight" in name for proj in ("q", "k", "v", "o"))
    )


def _is_decoder_dense_mlp(name: str) -> bool:
    return (
        name.startswith("model.decoder.layers.")
        and ".mlp." in name
        and name.endswith(".weight")
        and any(f".{proj}_proj.weight" in name for proj in ("gate", "up", "down"))
    )


def _is_decoder_router(name: str) -> bool:
    return (
        name.startswith("model.decoder.layers.")
        and name.endswith(".router.proj.weight")
    )


def _is_decoder_expert(name: str) -> bool:
    return (
        name.startswith("model.decoder.layers.")
        and (
            name.endswith(".experts.gate_up_proj")
            or name.endswith(".experts.down_proj")
            or name.endswith(".experts.gate_up_proj.weight")
            or name.endswith(".experts.down_proj.weight")
        )
    )


def _is_self_conditioning(name: str) -> bool:
    return name.startswith("model.decoder.self_conditioning.")


def quant_policy(name: str, shape: tuple[int, ...], profile_bits: int) -> QuantPolicy:
    if name.endswith("_scale_inv"):
        return QuantPolicy("skip", 0, "scale_inv", "stale source-side quant metadata")

    if len(shape) < 2:
        return QuantPolicy("passthrough", 16, "scalar", "rank<2")

    if _is_norm_or_scalar_control(name):
        return QuantPolicy("passthrough", 16, "control", "norm/bias/router scalar")

    if name.endswith("model.decoder.embed_tokens.weight"):
        return QuantPolicy("passthrough", 16, "embedding", "tied embedding/output projection")

    if name.startswith("model.encoder."):
        return QuantPolicy("passthrough", 16, "vision_encoder", "VL encoder path preserved")

    if _is_self_conditioning(name):
        return QuantPolicy("passthrough", 16, "self_conditioning", "block diffusion parity guard")

    if _is_decoder_router(name):
        return QuantPolicy("mxfp", 8, "router", "router logits kept MXFP8")

    if _is_decoder_dense_mlp(name):
        bits = 8 if profile_bits == 4 else 8
        return QuantPolicy("mxfp", bits, "dense_mlp", f"dense MLP MXFP{bits}")

    if _is_decoder_attention(name):
        return QuantPolicy("mxfp", profile_bits, "attention", f"attention MXFP{profile_bits}")

    if _is_decoder_expert(name):
        return QuantPolicy("mxfp", profile_bits, "experts", f"MoE experts MXFP{profile_bits}")

    if name.startswith("model.decoder.") and name.endswith(".weight"):
        return QuantPolicy("mxfp", profile_bits, "decoder_other", f"decoder linear MXFP{profile_bits}")

    return QuantPolicy("passthrough", 16, "unmatched", "not a recognized decoder linear")


def _quant_base_name(name: str) -> str:
    if name.endswith(".weight"):
        return name[: -len(".weight")]
    return name


def _assert_quantizable_shape(name: str, tensor: np.ndarray, group_size: int) -> None:
    if tensor.ndim < 2:
        raise ValueError(f"{name}: MXFP quantization expects rank >= 2, got {tensor.shape}")
    if tensor.shape[-1] % group_size != 0:
        raise ValueError(
            f"{name}: last dim {tensor.shape[-1]} is not divisible by group_size={group_size}"
        )


def _lazy_import_mlx():
    try:
        import mlx.core as mx  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "MLX is required for conversion. Use a Python env with mlx installed, "
            "for example /Users/eric/jang/jang-tools/.venv/bin/python."
        ) from exc
    return mx


def _mxfp_quantize(
    tensor: np.ndarray,
    *,
    bits: int,
    group_size: int,
    mx: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    _assert_quantizable_shape("tensor", tensor, group_size)
    original_shape = tensor.shape if tensor.ndim >= 3 else None
    matrix = tensor.reshape(-1, tensor.shape[-1]) if original_shape is not None else tensor

    q_weights: list[np.ndarray] = []
    q_scales: list[np.ndarray] = []
    q_biases: list[np.ndarray] = []
    mode = f"mxfp{bits}"
    chunk_rows = max(1, min(matrix.shape[0], 100_000_000 // max(1, matrix.shape[1])))
    for start in range(0, matrix.shape[0], chunk_rows):
        chunk = mx.array(matrix[start : start + chunk_rows].astype(np.float16))
        quantized = mx.quantize(chunk, group_size=group_size, bits=bits, mode=mode)
        qw, qs = quantized[:2]
        qb = quantized[2] if len(quantized) > 2 else None
        mx.eval(qw, qs, *([] if qb is None else [qb]))
        q_weights.append(np.array(qw))
        q_scales.append(np.array(qs))
        if qb is not None:
            q_biases.append(np.array(qb))
        del chunk, qw, qs, qb

    weight = np.concatenate(q_weights, axis=0)
    scales = np.concatenate(q_scales, axis=0)
    biases = np.concatenate(q_biases, axis=0) if q_biases else None
    if original_shape is not None:
        weight = weight.reshape(*original_shape[:-1], weight.shape[-1])
        scales = scales.reshape(*original_shape[:-1], scales.shape[-1])
        if biases is not None:
            biases = biases.reshape(*original_shape[:-1], biases.shape[-1])
    return weight, scales, biases


def _copy_sidecars(src: Path, out: Path) -> None:
    for name in SIDECAR_FILES:
        src_file = src / name
        if src_file.exists():
            shutil.copy2(src_file, out / name)
    tok_cfg = out / "tokenizer_config.json"
    template = out / "chat_template.jinja"
    if tok_cfg.exists() and template.exists():
        cfg = _read_json(tok_cfg)
        cfg["chat_template"] = template.read_text(encoding="utf-8")
        _write_json(tok_cfg, cfg)


def _remove_stale_output(out: Path) -> None:
    if not out.exists():
        return
    for path in out.glob("model-*.safetensors"):
        path.unlink()
    for name in ("model.safetensors", "model.safetensors.index.json", "config.json", "jang_config.json"):
        path = out / name
        if path.exists():
            path.unlink()


def _validate_source(src: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cfg = _read_json(src / "config.json")
    gen = _read_json(src / "generation_config.json") if (src / "generation_config.json").exists() else {}
    proc = _read_json(src / "processor_config.json") if (src / "processor_config.json").exists() else {}
    if cfg.get("model_type") != "diffusion_gemma":
        raise SystemExit(f"expected model_type='diffusion_gemma', got {cfg.get('model_type')!r}")
    arch = cfg.get("architectures") or []
    if "DiffusionGemmaForBlockDiffusion" not in arch:
        raise SystemExit(f"expected DiffusionGemmaForBlockDiffusion architecture, got {arch!r}")
    return cfg, gen, proc


def _target_from_args(src: Path, out: Path | None, out_root: Path | None, bits: int) -> Path:
    if out is not None:
        return out.expanduser()
    root = (out_root or DEFAULT_OUTPUT_ROOT).expanduser()
    return root / f"{TARGET_BASE_NAME}-MXFP{bits}"


def _policy_summary(items: list[TensorItem], bits: int) -> dict[str, Any]:
    counts: dict[str, int] = {}
    roles: dict[str, int] = {}
    estimated_quantized_source_bytes = 0
    estimated_passthrough_source_bytes = 0
    overrides: list[str] = []
    skips: list[str] = []
    for item in items:
        policy = quant_policy(item.name, item.shape, bits)
        counts[f"{policy.method}-{policy.bits}"] = counts.get(f"{policy.method}-{policy.bits}", 0) + 1
        roles[policy.role] = roles.get(policy.role, 0) + 1
        source_bytes = int(np.prod(item.shape, dtype=np.int64)) * 2 if item.shape else 0
        if policy.method == "mxfp":
            estimated_quantized_source_bytes += source_bytes
            if bits == 4 and policy.bits != bits:
                overrides.append(_quant_base_name(item.name))
        elif policy.method == "passthrough":
            estimated_passthrough_source_bytes += source_bytes
            if item.name.endswith(".weight"):
                skips.append(_quant_base_name(item.name))
    return {
        "counts": counts,
        "roles": roles,
        "mxfp8_overrides_for_mxfp4": sorted(set(overrides)),
        "passthrough_weight_bases": sorted(set(skips)),
        "estimated_source_gb_by_policy": {
            "quantized": round(estimated_quantized_source_bytes / (1024 ** 3), 2),
            "passthrough_fp16": round(estimated_passthrough_source_bytes / (1024 ** 3), 2),
        },
    }


def _build_quant_config(
    *,
    bits: int,
    group_size: int,
    quantized_policies: dict[str, QuantPolicy],
    passthrough_bases: set[str],
) -> dict[str, Any]:
    weight_format = f"mxfp{bits}"
    config: dict[str, Any] = {
        "bits": bits,
        "group_size": group_size,
        "mode": weight_format,
        "quantization_backend": "mx.quantize",
        "family": "diffusion_gemma",
        "canvas_length": 256,
        "quantized_roles": sorted({p.role for p in quantized_policies.values()}),
        "passthrough_roles": [
            "embedding",
            "norms",
            "router_scalars",
            "self_conditioning",
            "vision_encoder",
        ],
    }
    for base, policy in sorted(quantized_policies.items()):
        if bits == 4 and policy.bits != bits:
            config[base] = {"bits": policy.bits, "group_size": group_size, "mode": f"mxfp{policy.bits}"}
    for base in sorted(passthrough_bases):
        config[base] = False
    return config


def _write_configs(
    *,
    src: Path,
    out: Path,
    source_config: dict[str, Any],
    generation_config: dict[str, Any],
    processor_config: dict[str, Any],
    bits: int,
    group_size: int,
    shard_map: dict[str, str],
    total_size: int,
    quantized_policies: dict[str, QuantPolicy],
    passthrough_bases: set[str],
    quantized_count: int,
    passthrough_count: int,
    skipped_count: int,
) -> None:
    weight_format = f"mxfp{bits}"
    cfg = dict(source_config)
    cfg.pop("quantization_config", None)
    cfg["weight_format"] = weight_format
    cfg["quantization"] = _build_quant_config(
        bits=bits,
        group_size=group_size,
        quantized_policies=quantized_policies,
        passthrough_bases=passthrough_bases,
    )
    cfg["modalities"] = {
        "text": True,
        "image": bool(source_config.get("vision_config")),
        "vision": bool(source_config.get("vision_config")),
        "video": bool(processor_config.get("video_processor")),
        "audio": False,
    }
    cfg["has_vision"] = bool(source_config.get("vision_config"))
    cfg["has_video"] = bool(processor_config.get("video_processor"))
    cfg["has_audio"] = False
    cfg["modality_notes"] = {
        "image": "vision_config plus image_token_id are present; runtime proof still required",
        "video": "processor_config has video_processor, but config has no video_token_id; proof required",
        "audio": "processor_config has feature_extractor metadata, but config has no audio_config/audio_token_id",
    }
    _write_json(out / "config.json", cfg)

    jang_config = {
        "version": 2,
        "weight_format": weight_format,
        "profile": f"MXFP{bits}",
        "source_model": {
            "repo_id": "google/diffusiongemma-26B-A4B-it",
            "path": str(src),
            "architecture": source_config.get("architectures"),
            "model_type": source_config.get("model_type"),
            "text_model_type": (source_config.get("text_config") or {}).get("model_type"),
        },
        "modalities": cfg["modalities"],
        "has_vision": cfg["has_vision"],
        "has_video": cfg["has_video"],
        "has_audio": cfg["has_audio"],
        "quantization": {
            "method": weight_format,
            "mode": weight_format,
            "bits": bits,
            "group_size": group_size,
            "quantization_backend": "mx.quantize",
            "native_mx_format": True,
            "passthrough_tensor_count": passthrough_count,
            "quantized_tensor_count": quantized_count,
            "skipped_tensor_count": skipped_count,
            "mxfp8_override_count": sum(1 for p in quantized_policies.values() if p.bits == 8 and bits == 4),
        },
        "runtime": {
            "total_weight_bytes": total_size,
            "total_weight_gb": round(total_size / (1024 ** 3), 2),
            "canvas_length": source_config.get("canvas_length"),
            "max_new_tokens": generation_config.get("max_new_tokens"),
            "max_denoising_steps": generation_config.get("max_denoising_steps"),
            "sampler_config": generation_config.get("sampler_config"),
            "cache": "encoder_kv_plus_bidirectional_denoising_canvas",
        },
    }
    _write_json(out / "jang_config.json", jang_config)

    manifest = {
        "source": str(src),
        "output": str(out),
        "weight_format": weight_format,
        "group_size": group_size,
        "total_size": total_size,
        "shards": sorted(set(shard_map.values())),
        "tensor_counts": {
            "quantized": quantized_count,
            "passthrough": passthrough_count,
            "skipped": skipped_count,
            "indexed_output_tensors": len(shard_map),
        },
        "modalities": cfg["modalities"],
    }
    _write_json(out / "diffusiongemma_mxfp_manifest.json", manifest)


def convert(args: argparse.Namespace) -> int:
    src = args.src.expanduser()
    if not src.exists():
        raise SystemExit(f"source does not exist: {src}")
    source_config, generation_config, processor_config = _validate_source(src)
    out = _target_from_args(src, args.out, args.out_root, args.bits)
    weight_format = f"mxfp{args.bits}"

    items = _scan_source(src)
    policy = _policy_summary(items, args.bits)
    summary = {
        "source": str(src),
        "output": str(out),
        "weight_format": weight_format,
        "group_size": args.group_size,
        "tensors": len(items),
        "policy": policy,
        "modalities": {
            "image": bool(source_config.get("vision_config")),
            "video_processor": bool(processor_config.get("video_processor")),
            "video_token_id": source_config.get("video_token_id"),
            "audio_config": source_config.get("audio_config"),
            "audio_token_id": source_config.get("audio_token_id"),
        },
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    mx = _lazy_import_mlx()
    out.mkdir(parents=True, exist_ok=True)
    if args.replace:
        _remove_stale_output(out)
    elif any(out.glob("model-*.safetensors")) or (out / "model.safetensors.index.json").exists():
        raise SystemExit(f"output already has model artifacts; pass --replace: {out}")

    _copy_sidecars(src, out)

    shard_idx = 0
    shard_tensors: dict[str, np.ndarray] = {}
    shard_bytes = 0
    shard_map: dict[str, str] = {}
    quantized_policies: dict[str, QuantPolicy] = {}
    passthrough_bases: set[str] = set()
    quantized_count = 0
    passthrough_count = 0
    skipped_count = 0

    def flush_shard() -> None:
        nonlocal shard_idx, shard_tensors, shard_bytes
        if not shard_tensors:
            return
        shard_idx += 1
        name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        save_file(shard_tensors, str(out / name), metadata={"format": "mlx"})
        for key in shard_tensors:
            shard_map[key] = name
        if not args.quiet:
            print(f"shard {shard_idx}: {len(shard_tensors)} tensors, {shard_bytes / 1e9:.2f} GB")
        shard_tensors = {}
        shard_bytes = 0

    def add_tensor(name: str, array: np.ndarray) -> None:
        nonlocal shard_bytes
        shard_tensors[name] = np.ascontiguousarray(array)
        shard_bytes += int(shard_tensors[name].nbytes)
        if shard_bytes >= MAX_SHARD_BYTES:
            flush_shard()

    iterator = items
    if tqdm is not None and not args.quiet:
        iterator = tqdm(items, desc=f"converting {weight_format}")  # type: ignore[assignment]

    for item in iterator:
        policy_item = quant_policy(item.name, item.shape, args.bits)
        if policy_item.method == "skip":
            skipped_count += 1
            continue

        tensor = _load_tensor(item.shard, item.name, item.shape)
        if policy_item.method == "passthrough":
            add_tensor(item.name, tensor.astype(np.float16))
            if item.name.endswith(".weight") or _is_self_conditioning(item.name):
                passthrough_bases.add(_quant_base_name(item.name))
            passthrough_count += 1
        else:
            _assert_quantizable_shape(item.name, tensor, args.group_size)
            qw, qs, qb = _mxfp_quantize(tensor, bits=policy_item.bits, group_size=args.group_size, mx=mx)
            base = _quant_base_name(item.name)
            add_tensor(f"{base}.weight", qw)
            add_tensor(f"{base}.scales", qs)
            if qb is not None:
                add_tensor(f"{base}.biases", qb)
            quantized_policies[base] = policy_item
            quantized_count += 1
            del qw, qs, qb

        del tensor
        if (quantized_count + passthrough_count) % 100 == 0:
            gc.collect()
            mx.clear_cache()

    flush_shard()
    for idx in range(1, shard_idx + 1):
        old = out / f"model-{idx:05d}-of-XXXXX.safetensors"
        new = out / f"model-{idx:05d}-of-{shard_idx:05d}.safetensors"
        if old.exists():
            old.rename(new)
    shard_map = {key: value.replace("XXXXX", f"{shard_idx:05d}") for key, value in shard_map.items()}
    total_size = sum((out / shard).stat().st_size for shard in set(shard_map.values()))
    _write_json(
        out / "model.safetensors.index.json",
        {"metadata": {"format": weight_format, "total_size": total_size}, "weight_map": shard_map},
    )
    _write_configs(
        src=src,
        out=out,
        source_config=source_config,
        generation_config=generation_config,
        processor_config=processor_config,
        bits=args.bits,
        group_size=args.group_size,
        shard_map=shard_map,
        total_size=total_size,
        quantized_policies=quantized_policies,
        passthrough_bases=passthrough_bases,
        quantized_count=quantized_count,
        passthrough_count=passthrough_count,
        skipped_count=skipped_count,
    )

    print(
        json.dumps(
            {
                "output": str(out),
                "weight_format": weight_format,
                "shards": shard_idx,
                "total_weight_gb": round(total_size / (1024 ** 3), 2),
                "quantized_tensors": quantized_count,
                "passthrough_tensors": passthrough_count,
                "skipped_tensors": skipped_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DiffusionGemma BF16 source to first-party native MLX MXFP4/MXFP8."
    )
    parser.add_argument("--src", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return convert(parse_args(argv or sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
