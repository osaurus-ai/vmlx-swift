#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEMMA3="$ROOT/Libraries/MLXVLM/Models/Gemma3.swift"
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

require_text "$GEMMA3" 'Gemma3 text config hidden_size must equal num_attention_heads * head_dim.' \
  'text head-dim validation'
require_text "$GEMMA3" 'Gemma3 text config num_attention_heads must be divisible by num_key_value_heads.' \
  'text KV-head validation'
require_text "$GEMMA3" 'Gemma3 vision config hidden_size must be divisible by num_attention_heads.' \
  'vision head divisibility validation'
require_text "$GEMMA3" 'Gemma3 vision config image_size must be divisible by patch_size.' \
  'vision patch divisibility validation'
require_text "$GEMMA3" 'Gemma3 config image_token_id must be less than vocab_size.' \
  'top-level image token bounds validation'
require_text "$GEMMA3" 'try validateRGBTuple(imageMean, key: .imageMean, in: container)' \
  'processor image_mean RGB tuple validation'
require_text "$GEMMA3" 'try validateRGBTuple(imageStd, key: .imageStd, in: container)' \
  'processor image_std RGB tuple validation'
require_text "$GEMMA3" 'Gemma3 processor config rescale_factor must be finite and > 0.' \
  'processor rescale validation'
reject_text "$GEMMA3" 'fatalError("The input feature dimensions should be divisible by the number of heads")' \
  'old vision attention head-divisibility fatalError'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: Gemma3 VLM config/processor fatal boundaries are guarded at decode time.\n'
