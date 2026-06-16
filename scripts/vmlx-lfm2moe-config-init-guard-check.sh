#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LFM2MOE="$ROOT/Libraries/MLXLLM/Models/LFM2MoE.swift"
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

reject_text "$LFM2MOE" 'precondition(args.vocabularySize > 0)' \
  'vocabulary precondition'

require_text "$LFM2MOE" 'try validateDecodedFields(container: container)' \
  'decode validation call'
require_text "$LFM2MOE" 'try validatePositive(vocabularySize, key: .vocabularySize, in: container)' \
  'vocab_size validation'
require_text "$LFM2MOE" 'LFM2MoE hidden_size must be divisible by num_attention_heads.' \
  'hidden/head validation'
require_text "$LFM2MOE" 'LFM2MoE num_attention_heads must be divisible by num_key_value_heads.' \
  'KV-head validation'
require_text "$LFM2MOE" 'LFM2MoE num_experts_per_tok must be <= num_experts.' \
  'expert top-k validation'
require_text "$LFM2MOE" 'LFM2MoE num_dense_layers must be >= 0 and <= num_hidden_layers.' \
  'dense layer bounds validation'
require_text "$LFM2MOE" 'LFM2MoE full_attn_idxs entries must be within layer bounds.' \
  'full attention index validation'
require_text "$LFM2MOE" 'LFM2MoE layer_types count must equal num_hidden_layers.' \
  'layer type count validation'
require_text "$LFM2MOE" 'LFM2MoE layer_types entries must be full_attention or short_conv.' \
  'layer type value validation'
require_text "$LFM2MOE" 'debugDescription: "LFM2MoE \(key.rawValue) must be finite and > 0."' \
  'finite positive float validation'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: LFM2MoE config/init fatal boundaries are guarded at decode time.\n'
