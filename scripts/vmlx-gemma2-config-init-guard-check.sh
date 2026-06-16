#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEMMA2="$ROOT/Libraries/MLXLLM/Models/Gemma2.swift"
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

reject_text "$GEMMA2" 'precondition(args.vocabularySize > 0)' \
  'vocabulary precondition'

require_text "$GEMMA2" 'try validateDecodedFields(container: container)' \
  'decode validation call'
require_text "$GEMMA2" 'try validatePositive(vocabularySize, key: .vocabularySize, in: container)' \
  'vocab_size validation'
require_text "$GEMMA2" 'try validatePositive(queryPreAttnScalar, key: .queryPreAttnScalar, in: container)' \
  'query pre-attention scalar validation'
require_text "$GEMMA2" 'Gemma2 hidden_size must equal num_attention_heads * head_dim.' \
  'hidden/head validation'
require_text "$GEMMA2" 'Gemma2 num_attention_heads must be divisible by num_key_value_heads.' \
  'KV-head validation'
require_text "$GEMMA2" 'debugDescription: "Gemma2 \(key.rawValue) must be finite and > 0."' \
  'finite positive float validation'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: Gemma2 config/init fatal boundaries are guarded at decode time.\n'
