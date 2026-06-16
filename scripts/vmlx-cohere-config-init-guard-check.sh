#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COHERE="$ROOT/Libraries/MLXLLM/Models/Cohere.swift"
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

reject_text "$COHERE" 'precondition(args.vocabularySize > 0)' \
  'vocabulary precondition'

require_text "$COHERE" 'try validateDecodedFields(container: container)' \
  'decode validation call'
require_text "$COHERE" 'try validatePositive(vocabularySize, key: .vocabularySize, in: container)' \
  'vocab_size validation'
require_text "$COHERE" 'Cohere hidden_size must be divisible by num_attention_heads.' \
  'hidden/head validation'
require_text "$COHERE" 'Cohere num_attention_heads must be divisible by num_key_value_heads.' \
  'KV-head validation'
require_text "$COHERE" 'Cohere rope_scaling.factor must be finite and > 0.' \
  'rope factor validation'
require_text "$COHERE" 'debugDescription: "Cohere \(key.rawValue) must be finite and > 0."' \
  'finite positive float validation'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: Cohere config/init fatal boundaries are guarded at decode time.\n'
