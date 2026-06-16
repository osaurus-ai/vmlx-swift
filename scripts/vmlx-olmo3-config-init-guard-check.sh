#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLMO3="$ROOT/Libraries/MLXLLM/Models/Olmo3.swift"
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

reject_text "$OLMO3" 'precondition(args.vocabularySize > 0)' \
  'vocabulary precondition'

require_text "$OLMO3" 'try Self.validatePositive(hiddenLayers, key: .hiddenLayers, in: container)' \
  'pre-layer generation hidden layer validation'
require_text "$OLMO3" 'try validateDecodedFields(container: container)' \
  'decode validation call'
require_text "$OLMO3" 'try Self.validatePositive(vocabularySize, key: .vocabularySize, in: container)' \
  'vocab_size validation'
require_text "$OLMO3" 'Olmo3 hidden_size must equal num_attention_heads * head_dim.' \
  'hidden/head validation'
require_text "$OLMO3" 'Olmo3 num_attention_heads must be divisible by num_key_value_heads.' \
  'KV-head validation'
require_text "$OLMO3" 'Olmo3 layer_types count must equal num_hidden_layers.' \
  'layer type count validation'
require_text "$OLMO3" 'Olmo3 layer_types entries must be full_attention or sliding_attention.' \
  'layer type value validation'
require_text "$OLMO3" 'Olmo3 rope_scaling.factor must be finite and > 0.' \
  'rope factor validation'
require_text "$OLMO3" 'debugDescription: "Olmo3 \(key.rawValue) must be finite and > 0."' \
  'finite positive float validation'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: Olmo3 config/init fatal boundaries are guarded at decode time.\n'
