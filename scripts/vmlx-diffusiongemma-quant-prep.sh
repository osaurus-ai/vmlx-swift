#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/vmlx-diffusiongemma-quant-prep.sh [--src DIR] [--out-root DIR] [--artifact-root DIR] [--allow-incomplete]

Preflight DiffusionGemma BF16 source and write first-party native MLX
MXFP4/MXFP8 quantization prep manifests. This does not quantize weights; it
refuses to pretend the bundle is ready when source shards or modality metadata
are missing.
USAGE
  exit 64
}

SRC="/Users/eric/models/google/diffusiongemma-26B-A4B-it"
OUT_ROOT="/Users/eric/models/OsaurusAI"
ARTIFACT_ROOT=""
ALLOW_INCOMPLETE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) SRC="${2:-}"; shift 2 ;;
    --out-root) OUT_ROOT="${2:-}"; shift 2 ;;
    --artifact-root) ARTIFACT_ROOT="${2:-}"; shift 2 ;;
    --allow-incomplete) ALLOW_INCOMPLETE=1; shift ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

if [[ -z "$ARTIFACT_ROOT" ]]; then
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  ARTIFACT_ROOT="docs/local/diffusiongemma-quant-prep/$STAMP"
fi

python3 - "$SRC" "$OUT_ROOT" "$ARTIFACT_ROOT" "$ALLOW_INCOMPLETE" <<'PY'
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

src = Path(sys.argv[1]).expanduser()
out_root = Path(sys.argv[2]).expanduser()
artifact_root = Path(sys.argv[3])
allow_incomplete = sys.argv[4] == "1"

required_small = [
    "README.md",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]

missing = [name for name in required_small if not (src / name).is_file()]
config_path = src / "config.json"
if not config_path.is_file():
    raise SystemExit(f"missing config.json: {config_path}")

config = json.loads(config_path.read_text())
generation_config = {}
if (src / "generation_config.json").is_file():
    generation_config = json.loads((src / "generation_config.json").read_text())

index = None
if (src / "model.safetensors.index.json").is_file():
    index = json.loads((src / "model.safetensors.index.json").read_text())

text_config = config.get("text_config") or {}
vision_config = config.get("vision_config")
audio_config = config.get("audio_config")
model_type = config.get("model_type")
arch = config.get("architectures") or []
canvas_length = config.get("canvas_length")

if model_type != "diffusion_gemma":
    raise SystemExit(f"unexpected model_type={model_type!r}; refusing DiffusionGemma prep")
if "DiffusionGemmaForBlockDiffusion" not in arch:
    raise SystemExit(f"unexpected architectures={arch!r}; refusing DiffusionGemma prep")

shard_files = sorted(src.glob("model-*-of-*.safetensors"))
expected_shards = 0
if index:
    expected_shards = len(set(index.get("weight_map", {}).values()))
else:
    expected_shards = 11

complete = not missing and len(shard_files) == expected_shards
if not complete and not allow_incomplete:
    lines = [
        "DiffusionGemma source is incomplete.",
        f"src={src}",
        f"missing={missing}",
        f"shards={len(shard_files)}/{expected_shards}",
        "Use --allow-incomplete only for metadata-only prep.",
    ]
    raise SystemExit("\n".join(lines))

artifact_root.mkdir(parents=True, exist_ok=True)
target_base = "diffusiongemma-26B-A4B-it"
targets = {
    "MXFP4": out_root / f"{target_base}-MXFP4",
    "MXFP8": out_root / f"{target_base}-MXFP8",
}

weight_counts = Counter()
if index:
    for key in index.get("weight_map", {}):
        if key.endswith(".weight"):
            weight_counts["weights"] += 1
        elif key.endswith(".scales"):
            weight_counts["scales"] += 1
        elif key.endswith(".biases"):
            weight_counts["biases"] += 1
        else:
            weight_counts["other"] += 1

tensor_policy = {
    "native_mx_quantize": [
        "uses mx.quantize(..., mode='mxfp4'|'mxfp8')",
        "emits MLX weight/scales companions, not affine 4-bit biases",
    ],
    "quantize_mxfp4_profile": [
        "model.decoder.layers.*.self_attn.{q,k,v,o}_proj.weight",
        "model.decoder.layers.*.experts.{gate_up,down}_proj.weight",
        "model.decoder.layers.*.mlp.{gate,up,down}_proj.weight as MXFP8 override",
        "model.decoder.layers.*.router.proj.weight as MXFP8 override",
    ],
    "quantize_mxfp8_profile": [
        "model.decoder.layers.*.self_attn.{q,k,v,o}_proj.weight",
        "model.decoder.layers.*.experts.{gate_up,down}_proj.weight",
        "model.decoder.layers.*.mlp.{gate,up,down}_proj.weight",
        "model.decoder.layers.*.router.proj.weight",
    ],
    "keep_fp16_or_bf16": [
        "norms, layer_scalar, per_expert_scale, router.scale",
        "token embeddings and LM head for first parity pass",
        "vision tower, multimodal projector/embedder, image soft-token path",
        "self_conditioning projections until text parity passes",
    ],
    "blocked_until_engine": [
        "do not benchmark MXFP bundles with autoregressive TokenIterator",
        "do not advertise audio; source has no audio_config/audio_token_id and MLX-VLM reference rejects audio",
        "do not claim video until processor/video-token/runtime evidence passes",
    ],
}

base_manifest = {
    "source": {
        "path": str(src),
        "repo_id": "google/diffusiongemma-26B-A4B-it",
        "complete": complete,
        "missing": missing,
        "shards_present": len(shard_files),
        "shards_expected": expected_shards,
    },
    "model": {
        "model_type": model_type,
        "architectures": arch,
        "canvas_length": canvas_length,
        "max_new_tokens": generation_config.get("max_new_tokens"),
        "max_denoising_steps": generation_config.get("max_denoising_steps"),
        "sampler_config": generation_config.get("sampler_config"),
        "confidence_threshold": generation_config.get("confidence_threshold"),
        "stability_threshold": generation_config.get("stability_threshold"),
        "text_model_type": text_config.get("model_type"),
        "hidden_size": text_config.get("hidden_size"),
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "num_experts": text_config.get("num_experts"),
        "top_k_experts": text_config.get("top_k_experts"),
        "sliding_window": text_config.get("sliding_window"),
        "layer_types": text_config.get("layer_types"),
        "vision_model_type": (vision_config or {}).get("model_type") if isinstance(vision_config, dict) else None,
        "image_token_id": config.get("image_token_id"),
        "vision_soft_tokens_per_image": config.get("vision_soft_tokens_per_image"),
        "audio_config_present": audio_config is not None,
        "audio_token_id": config.get("audio_token_id"),
        "video_token_id": config.get("video_token_id"),
    },
    "weight_index": {
        "present": index is not None,
        "counts": dict(weight_counts),
    },
    "tensor_policy": tensor_policy,
    "targets": {},
}

for profile, out_dir in targets.items():
    bits = 4 if profile == "MXFP4" else 8
    manifest = dict(base_manifest)
    manifest["quantization"] = {
        "profile": profile,
        "mode": profile.lower(),
        "bits": bits,
        "group_size": 32,
        "weight_format": profile.lower(),
        "first_party": True,
        "source_converter_status": "ready",
        "source_converter": "scripts/vmlx-convert-diffusiongemma-mxfp.py",
        "example_command": (
            "scripts/vmlx-convert-diffusiongemma-mxfp.py "
            f"--src {src} --out {out_dir} --bits {bits} --group-size 32 --replace"
        ),
    }
    manifest["output"] = {
        "path": str(out_dir),
        "estimated_min_free_bytes_before_run": int(1.25 * (src.stat().st_size if src.is_file() else sum(p.stat().st_size for p in src.rglob('*') if p.is_file()))),
    }
    manifest["targets"] = {profile: str(out_dir)}
    (artifact_root / f"{profile.lower()}-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

summary = {
    "source": str(src),
    "complete": complete,
    "missing": missing,
    "shards_present": len(shard_files),
    "shards_expected": expected_shards,
    "artifact_root": str(artifact_root),
    "targets": {k: str(v) for k, v in targets.items()},
    "vl": {
        "status": "required",
        "vision_config_present": vision_config is not None,
        "image_token_id": config.get("image_token_id"),
        "vision_soft_tokens_per_image": config.get("vision_soft_tokens_per_image"),
    },
    "audio": {
        "status": "unsupported_by_source_bundle_and_reference_processor",
        "audio_config_present": audio_config is not None,
        "audio_token_id": config.get("audio_token_id"),
    },
    "video": {
        "status": "processor_present_runtime_proof_required",
        "video_processor_present": isinstance(json.loads((src / "processor_config.json").read_text()).get("video_processor"), dict) if (src / "processor_config.json").is_file() else False,
        "video_token_id": config.get("video_token_id"),
    },
}
(artifact_root / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

print(json.dumps(summary, indent=2, sort_keys=True))
PY
