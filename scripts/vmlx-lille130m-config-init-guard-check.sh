#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LILLE="$ROOT/Libraries/MLXLLM/Models/Lille130m.swift"
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

reject_text "$LILLE" 'precondition(args.vocabularySize > 0)' \
  'vocabulary precondition'

require_text "$LILLE" 'try validateDecodedFields(container: container)' \
  'decode validation call'
require_text "$LILLE" 'try validatePositive(vocabularySize, key: .vocabularySize, in: container)' \
  'vocab_size validation'
require_text "$LILLE" 'Lille130m hidden_size must be divisible by n_head.' \
  'hidden/head divisibility validation'
require_text "$LILLE" 'Lille130m n_head must be divisible by n_kv_heads.' \
  'KV-head validation'
require_text "$LILLE" 'debugDescription: "Lille130m \(key.rawValue) must be finite and > 0."' \
  'finite positive float validation'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: Lille130m config/init fatal boundaries are guarded at decode time.\n'
