#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IDEFICS="$ROOT/Libraries/MLXVLM/Models/Idefics3.swift"
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

require_text "$IDEFICS" 'Idefics3 text config hidden_size must be divisible by num_attention_heads.' \
  'text head divisibility validation'
require_text "$IDEFICS" 'Idefics3 text config num_attention_heads must be divisible by num_key_value_heads.' \
  'text KV-head validation'
require_text "$IDEFICS" 'Idefics3 vision config hidden_size must be divisible by num_attention_heads.' \
  'vision head divisibility validation'
require_text "$IDEFICS" 'Idefics3 vision config image_size must be divisible by patch_size.' \
  'vision patch divisibility validation'
require_text "$IDEFICS" 'Idefics3 config image_token_index must be less than vocab_size.' \
  'top-level image-token bounds validation'
require_text "$IDEFICS" 'throw VLMError.processing("Idefics3 inputIds required when no pixelValues are provided.")' \
  'typed prepare failure for missing inputIds without pixels'
require_text "$IDEFICS" 'throw VLMError.processing("Idefics3 inputIds and pixelValues are required for multimodal prepare.")' \
  'typed prepare failure for malformed multimodal inputs'
require_text "$IDEFICS" 'Idefics3 image token count' \
  'typed image token/chunk alignment validation'
require_text "$IDEFICS" 'try validateRGBTuple(imageMean, key: .imageMean, in: container)' \
  'processor image_mean RGB tuple validation'
require_text "$IDEFICS" 'try validateRGBTuple(imageStd, key: .imageStd, in: container)' \
  'processor image_std RGB tuple validation'
reject_text "$IDEFICS" 'fatalError("inputIds required if no pixelValues")' \
  'old prepare inputIds fatalError'
reject_text "$IDEFICS" 'fatalError("inputIds and pixelValues required")' \
  'old multimodal prepare fatalError'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: Idefics3 config and VLM prepare fatal boundaries are guarded.\n'
