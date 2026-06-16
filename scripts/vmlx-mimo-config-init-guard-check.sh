#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMO="$ROOT/Libraries/MLXLLM/Models/MiMo.swift"
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

reject_text "$MIMO" 'precondition(args.vocabularySize > 0)' \
  'vocabulary precondition'

require_text "$MIMO" 'try validateDecodedFields(container: container)' \
  'decode validation call'
require_text "$MIMO" 'try validatePositive(vocabularySize, key: .vocabularySize, in: container)' \
  'vocab_size validation'
require_text "$MIMO" 'MiMo hidden_size must be divisible by num_attention_heads.' \
  'hidden/head validation'
require_text "$MIMO" 'MiMo num_attention_heads must be divisible by num_key_value_heads.' \
  'KV-head validation'
require_text "$MIMO" 'MiMo num_nextn_predict_layers must be >= 0.' \
  'next-token layer validation'
require_text "$MIMO" 'MiMo rope_scaling.factor must be finite and > 0.' \
  'rope factor validation'
require_text "$MIMO" 'debugDescription: "MiMo \(key.rawValue) must be finite and > 0."' \
  'finite positive float validation'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: MiMo config/init fatal boundaries are guarded at decode time.\n'
