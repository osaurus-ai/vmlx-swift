#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PALI="$ROOT/Libraries/MLXVLM/Models/Paligemma.swift"
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

require_text "$PALI" 'PaliGemma text config hidden_size must be divisible by num_attention_heads.' \
  'text head divisibility validation'
require_text "$PALI" 'PaliGemma text config num_attention_heads must be divisible by num_key_value_heads.' \
  'text KV-head validation'
require_text "$PALI" 'PaliGemma vision config model_type must be siglip_vision_model.' \
  'vision model_type validation'
require_text "$PALI" 'PaliGemma vision config hidden_size must be divisible by num_attention_heads.' \
  'vision head divisibility validation'
require_text "$PALI" 'PaliGemma vision config image_size must be divisible by patch_size.' \
  'vision patch divisibility validation'
require_text "$PALI" 'PaliGemma config image_token_index must be less than vocab_size.' \
  'image token bounds validation'
require_text "$PALI" 'try validateRGBTuple(imageMean, key: .imageMean, in: container)' \
  'processor image_mean RGB tuple validation'
require_text "$PALI" 'try validateRGBTuple(imageStd, key: .imageStd, in: container)' \
  'processor image_std RGB tuple validation'
reject_text "$PALI" 'fatalError(' \
  'process-fatal PaliGemma path'
reject_text "$PALI" 'precondition(' \
  'process-fatal PaliGemma precondition'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: PaliGemma config/init fatal boundaries are guarded at decode time.\n'
