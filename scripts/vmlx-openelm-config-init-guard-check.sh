#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENELM="$ROOT/Libraries/MLXLLM/Models/OpenELM.swift"
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

reject_text "$OPENELM" 'precondition(args.vocabularySize > 0)' \
  'vocabulary precondition'

require_text "$OPENELM" 'try validateBaseFields(container: container)' \
  'decode validation call'
require_text "$OPENELM" 'try validatePositive(vocabularySize, key: .vocabularySize, in: container)' \
  'vocab_size validation'
require_text "$OPENELM" 'OpenELM num_transformer_layers must be greater than 1.' \
  'layer count validation'
require_text "$OPENELM" 'OpenELM model_dim must be divisible by head_dim.' \
  'model/head divisibility validation'
require_text "$OPENELM" 'OpenELM derived query dimensions must be positive and divisible by head_dim.' \
  'derived query dimension validation'
require_text "$OPENELM" 'OpenELM derived num_key_value_heads entries must be positive.' \
  'derived KV-head validation'
require_text "$OPENELM" 'OpenELM ffn_multipliers must contain two positive finite values.' \
  'FFN multiplier validation'
require_text "$OPENELM" 'OpenELM qkv multipliers must contain two positive finite values.' \
  'QKV multiplier validation'
require_text "$OPENELM" 'debugDescription: "OpenELM \(key.rawValue) must be finite and > 0."' \
  'finite positive float validation'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: OpenELM config/init fatal boundaries are guarded at decode time.\n'
