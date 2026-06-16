#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NANOCHAT="$ROOT/Libraries/MLXLLM/Models/NanoChat.swift"
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

reject_text "$NANOCHAT" 'precondition(headDim % 2 == 0, "Head dimension must be even for rotary embeddings.")' \
  'headDim rotary precondition'
reject_text "$NANOCHAT" 'precondition(config.vocabularySize > 0)' \
  'vocabulary precondition'

require_text "$NANOCHAT" 'try validateDecodedFields(container: container)' \
  'decode validation call'
require_text "$NANOCHAT" 'try validatePositive(vocabularySize, key: .vocabularySize, in: container)' \
  'vocab_size validation'
require_text "$NANOCHAT" 'NanoChat hidden_size must be divisible by num_attention_heads.' \
  'hidden/head divisibility validation'
require_text "$NANOCHAT" 'NanoChat head dimension must be even for rotary embeddings.' \
  'rotary head-dim validation'
require_text "$NANOCHAT" 'NanoChat num_attention_heads must be divisible by num_key_value_heads.' \
  'KV-head validation'
require_text "$NANOCHAT" 'debugDescription: "NanoChat \(key.rawValue) must be finite and > 0."' \
  'finite positive float validation'
require_text "$NANOCHAT" 'debugDescription: "NanoChat \(key.rawValue) must be finite and >= 0."' \
  'finite nonnegative float validation'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: NanoChat config/init fatal boundaries are guarded at decode time.\n'
