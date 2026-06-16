#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAONE4="$ROOT/Libraries/MLXLLM/Models/Exaone4.swift"
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

reject_text "$EXAONE4" 'precondition(args.vocabularySize > 0)' \
  'vocabulary precondition'

require_text "$EXAONE4" 'try validateDecodedFields(container: container)' \
  'decode validation call'
require_text "$EXAONE4" 'try validatePositive(vocabularySize, key: .vocabularySize, in: container)' \
  'vocab_size validation'
require_text "$EXAONE4" 'Exaone4 hidden_size must equal num_attention_heads * head_dim.' \
  'hidden/head validation'
require_text "$EXAONE4" 'Exaone4 num_attention_heads must be divisible by num_key_value_heads.' \
  'KV-head validation'
require_text "$EXAONE4" 'Exaone4 sliding_window_pattern must not be empty.' \
  'sliding window pattern empty validation'
require_text "$EXAONE4" 'Exaone4 sliding_window_pattern entries must be L or G.' \
  'sliding window pattern value validation'
require_text "$EXAONE4" 'Exaone4 sliding_window must be positive when sliding_window_pattern is present.' \
  'sliding window validation'
require_text "$EXAONE4" 'Exaone4 rope_scaling.factor must be finite and > 0.' \
  'rope factor validation'
require_text "$EXAONE4" 'debugDescription: "Exaone4 \(key.rawValue) must be finite and > 0."' \
  'finite positive float validation'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: Exaone4 config/init fatal boundaries are guarded at decode time.\n'
