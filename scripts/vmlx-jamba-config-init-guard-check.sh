#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JAMBA="$ROOT/Libraries/MLXLLM/Models/Jamba.swift"
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

reject_text "$JAMBA" 'precondition(config.vocabSize > 0)' \
  'vocabulary precondition'

require_text "$JAMBA" 'try validateDecodedFields(container: container)' \
  'decode validation call'
require_text "$JAMBA" 'try validatePositive(vocabSize, key: .vocabSize, in: container)' \
  'vocab_size validation'
require_text "$JAMBA" 'Jamba hidden_size must be divisible by num_attention_heads.' \
  'attention head divisibility validation'
require_text "$JAMBA" 'Jamba num_attention_heads must be divisible by num_key_value_heads.' \
  'KV-head validation'
require_text "$JAMBA" 'Jamba attn_layer_period and expert_layer_period must be positive.' \
  'period validation'
require_text "$JAMBA" 'Jamba layers_block_type count must equal num_hidden_layers.' \
  'layer block count validation'
require_text "$JAMBA" 'Jamba layers_block_type must include at least one attention and one mamba layer.' \
  'attention/mamba presence validation'
require_text "$JAMBA" 'Jamba layers_block_type entries must be attention or mamba.' \
  'layer block value validation'
require_text "$JAMBA" 'Jamba mamba_dt_rank must be positive.' \
  'mamba dt rank validation'
require_text "$JAMBA" 'debugDescription: "Jamba \(key.rawValue) must be finite and > 0."' \
  'finite positive float validation'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: Jamba config/init fatal boundaries are guarded at decode time.\n'
