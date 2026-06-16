#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRANITE="$ROOT/Libraries/MLXLLM/Models/Granite.swift"
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

reject_text "$GRANITE" 'precondition(args.vocabularySize > 0)' \
  'vocabulary precondition'

require_text "$GRANITE" 'try validateDecodedFields(container: container)' \
  'decode validation call'
require_text "$GRANITE" 'try validatePositive(vocabularySize, key: .vocabularySize, in: container)' \
  'vocab_size validation'
require_text "$GRANITE" 'Granite hidden_size must be divisible by num_attention_heads.' \
  'hidden/head validation'
require_text "$GRANITE" 'Granite num_attention_heads must be divisible by num_key_value_heads.' \
  'KV-head validation'
require_text "$GRANITE" 'Granite rope_scaling.factor must be finite and > 0.' \
  'rope factor validation'
require_text "$GRANITE" 'debugDescription: "Granite \(key.rawValue) must be finite and > 0."' \
  'finite positive float validation'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: Granite config/init fatal boundaries are guarded at decode time.\n'
