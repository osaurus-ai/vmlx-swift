#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LFM="$ROOT/Libraries/MLXVLM/Models/LFM2VL.swift"
failures=0

require_text() {
  local file="$1"
  local pattern="$2"
  local label="$3"
  if ! grep -Fq "$pattern" "$file"; then
    printf 'FAIL: %s missing %s\n' "$file" "$label" >&2
    failures=$((failures + 1))
  fi
}

reject_text() {
  local file="$1"
  local pattern="$2"
  local label="$3"
  if grep -Fq "$pattern" "$file"; then
    printf 'FAIL: %s still contains forbidden %s\n' "$file" "$label" >&2
    failures=$((failures + 1))
  fi
}

require_text "$LFM" 'LFM2-VL text config hidden_size must be divisible by num_attention_heads.' \
  'text attention-head divisibility validation'
require_text "$LFM" 'LFM2-VL text config num_attention_heads must be divisible by num_key_value_heads.' \
  'text KV-head divisibility validation'
require_text "$LFM" 'LFM2-VL text config full_attn_idxs entries must be within num_hidden_layers.' \
  'full-attention index bounds validation'
require_text "$LFM" 'LFM2-VL vision config hidden_size must be divisible by num_attention_heads.' \
  'vision attention-head divisibility validation'
require_text "$LFM" 'LFM2-VL vision config num_patches must be a square.' \
  'vision positional embedding square validation'
require_text "$LFM" 'LFM2-VL config vision_feature_layer is outside vision num_hidden_layers.' \
  'vision feature layer bounds validation'
require_text "$LFM" 'try validateRGBTuple(imageMean, key: ._imageMean, in: container)' \
  'processor image_mean tuple validation'
require_text "$LFM" 'try validateRGBTuple(imageStd, key: ._imageStd, in: container)' \
  'processor image_std tuple validation'
require_text "$LFM" 'throw VLMError.processing(' \
  'typed image feature/token mismatch failure'
require_text "$LFM" 'let inputEmbeddings = try getInputEmbeddings(' \
  'prepare propagates typed embedding failure'
reject_text "$LFM" 'fatalError(' \
  'process-fatal LFM2-VL path'
reject_text "$LFM" 'precondition(' \
  'process-fatal LFM2-VL constructor validation'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: LFM2-VL config/init fatal boundaries are guarded at decode/prepare time.\n'
