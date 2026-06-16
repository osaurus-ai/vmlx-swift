#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GLM="$ROOT/Libraries/MLXVLM/Models/GlmOcr.swift"
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

require_text "$GLM" 'GlmOcr rope_parameters.mrope_section must contain positive integers.' \
  'M-RoPE section validation'
require_text "$GLM" 'GlmOcr rope_parameters.mrope_section split indices must fit within the rotary frequency dimension.' \
  'M-RoPE split bounds validation'
require_text "$GLM" 'GlmOcr text config hidden_size must equal num_attention_heads * head_dim.' \
  'text attention dimension validation'
require_text "$GLM" 'GlmOcr text config num_attention_heads must be divisible by num_key_value_heads.' \
  'text KV-head validation'
require_text "$GLM" 'GlmOcr vision config hidden_size must be divisible by num_heads.' \
  'vision head divisibility validation'
require_text "$GLM" 'GlmOcr image/video token ids must be less than vocab_size.' \
  'image/video token bounds validation'
require_text "$GLM" 'GlmOcr image prefill only supports batch size 1; received batch size' \
  'batch-size VLMError'
require_text "$GLM" 'GlmOcr image token count \(imageTokenCount) does not match image feature count \(imageFeatureCount).' \
  'image token/feature count validation'
require_text "$GLM" 'try Self.validateRGBTuple(imageMean, key: .imageMean, in: container)' \
  'processor image_mean RGB tuple validation'
require_text "$GLM" 'try Self.validateRGBTuple(imageStd, key: .imageStd, in: container)' \
  'processor image_std RGB tuple validation'
reject_text "$GLM" 'precondition(args.vocabularySize > 0)' \
  'text vocabulary constructor precondition'
reject_text "$GLM" 'precondition(batchSize == 1' \
  'batch-size getRopeIndex precondition'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: GlmOcr config and prepare fatal boundaries are guarded.\n'
