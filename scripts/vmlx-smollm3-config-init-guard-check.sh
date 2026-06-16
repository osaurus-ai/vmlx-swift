#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOLLM3="$ROOT/Libraries/MLXLLM/Models/SmolLM3.swift"
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

reject_text "$SMOLLM3" 'precondition(args.vocabularySize > 0)' \
  'vocabulary precondition'

require_text "$SMOLLM3" 'try validateDecodedFields(container: container)' \
  'decode validation call'
require_text "$SMOLLM3" 'try validatePositive(vocabularySize, key: .vocabularySize, in: container)' \
  'vocab_size validation'
require_text "$SMOLLM3" 'SmolLM3 hidden_size must equal num_attention_heads * head_dim.' \
  'hidden/head validation'
require_text "$SMOLLM3" 'SmolLM3 num_attention_heads must be divisible by num_key_value_heads.' \
  'KV-head validation'
require_text "$SMOLLM3" 'SmolLM3 no_rope_layer_interval must be positive.' \
  'no-rope interval validation'
require_text "$SMOLLM3" 'SmolLM3 no_rope_layers count must equal num_hidden_layers.' \
  'no-rope layer count validation'
require_text "$SMOLLM3" 'SmolLM3 no_rope_layers entries must be 0 or 1.' \
  'no-rope layer value validation'
require_text "$SMOLLM3" 'SmolLM3 rope_scaling.factor must be finite and > 0.' \
  'rope factor validation'
require_text "$SMOLLM3" 'debugDescription: "SmolLM3 \(key.rawValue) must be finite and > 0."' \
  'finite positive float validation'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: SmolLM3 config/init fatal boundaries are guarded at decode time.\n'
