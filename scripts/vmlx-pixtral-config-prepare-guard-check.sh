#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIXTRAL="$ROOT/Libraries/MLXVLM/Models/Pixtral.swift"
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

require_text "$PIXTRAL" 'Pixtral vision config hidden_size must equal num_attention_heads * head_dim.' \
  'vision head-dim validation'
require_text "$PIXTRAL" 'Pixtral text config hidden_size must equal num_attention_heads * head_dim.' \
  'text head-dim validation'
require_text "$PIXTRAL" 'Pixtral text config num_attention_heads must be divisible by num_key_value_heads.' \
  'text KV-head validation'
require_text "$PIXTRAL" 'Pixtral config image_token_index must be less than vocab_size.' \
  'image-token bounds validation'
require_text "$PIXTRAL" 'Pixtral config vision_feature_layer is outside vision num_hidden_layers.' \
  'vision feature layer bounds validation'
require_text "$PIXTRAL" 'throw VLMError.processing("Pixtral inputIds required when no pixelValues are provided.")' \
  'typed prepare failure for missing text input'
require_text "$PIXTRAL" 'throw VLMError.processing("Pixtral inputIds required when pixelValues are provided.")' \
  'typed prepare failure for missing text with pixels'
require_text "$PIXTRAL" 'Pixtral image token count' \
  'typed image token/feature count validation'
require_text "$PIXTRAL" 'try validateRGBTuple(imageMean, key: .imageMean, in: container)' \
  'processor image_mean RGB tuple validation'
require_text "$PIXTRAL" 'try validateRGBTuple(imageStd, key: .imageStd, in: container)' \
  'processor image_std RGB tuple validation'
reject_text "$PIXTRAL" 'fatalError(' \
  'Pixtral process-fatal path'
reject_text "$PIXTRAL" 'precondition(' \
  'Pixtral process-fatal precondition'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: Pixtral config and VLM prepare fatal boundaries are guarded.\n'
