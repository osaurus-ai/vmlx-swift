#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLMOE="$ROOT/Libraries/MLXLLM/Models/OlmoE.swift"
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

reject_text "$OLMOE" 'precondition(args.vocabularySize > 0)' \
  'vocabulary precondition'

require_text "$OLMOE" 'try validateDecodedFields(container: container)' \
  'decode validation call'
require_text "$OLMOE" 'try validatePositive(vocabularySize, key: .vocabularySize, in: container)' \
  'vocab_size validation'
require_text "$OLMOE" 'OlmoE hidden_size must equal num_attention_heads * head_dim.' \
  'hidden/head validation'
require_text "$OLMOE" 'OlmoE num_attention_heads must be divisible by num_key_value_heads.' \
  'KV-head validation'
require_text "$OLMOE" 'OlmoE num_experts_per_tok must be less than or equal to num_experts.' \
  'MoE top-k validation'
require_text "$OLMOE" 'OlmoE rope_scaling.factor must be finite and > 0.' \
  'rope factor validation'
require_text "$OLMOE" 'debugDescription: "OlmoE \(key.rawValue) must be finite and > 0."' \
  'finite positive float validation'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: OlmoE config/init fatal boundaries are guarded at decode time.\n'
