#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MISTRAL="$ROOT/Libraries/MLXVLM/Models/Mistral3.swift"
JANGTQ="$ROOT/Libraries/MLXVLM/Models/Mistral3VLMJANGTQ.swift"
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

require_text "$MISTRAL" 'Mistral3 VLM text config rope_parameters.rope_theta is required and must be > 0.' \
  'decode-time rope_parameters.rope_theta validation'
require_text "$MISTRAL" 'Mistral3 VLM text config hidden_size must equal num_attention_heads * head_dim.' \
  'text head dimension validation'
require_text "$MISTRAL" 'Mistral3 VLM text config num_attention_heads must be divisible by num_key_value_heads.' \
  'text KV-head divisibility validation'
require_text "$MISTRAL" 'try validateRGBTuple(imageMean, key: .imageMean, in: container)' \
  'processor image_mean RGB tuple validation'
require_text "$MISTRAL" 'try validateRGBTuple(imageStd, key: .imageStd, in: container)' \
  'processor image_std RGB tuple validation'
require_text "$MISTRAL" 'Mistral3 VLM processor config rescale_factor must be finite and > 0.' \
  'processor rescale factor validation'
require_text "$MISTRAL" 'base: config.ropeTheta,' \
  'normal attention consumes validated ropeTheta'
require_text "$JANGTQ" 'base: config.ropeTheta,' \
  'JANGTQ attention consumes validated ropeTheta'
reject_text "$MISTRAL" 'fatalError("rope_parameters' \
  'normal Mistral3 rope fatalError'
reject_text "$JANGTQ" 'fatalError("rope_parameters' \
  'JANGTQ Mistral3 rope fatalError'
reject_text "$JANGTQ" 'let ropeTheta = ropeParams["rope_theta"]?.asFloat()' \
  'JANGTQ constructor ropeTheta force validation'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: Mistral3 VLM rope/RGB config fatal boundaries are guarded at decode time.\n'
