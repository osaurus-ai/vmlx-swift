#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHI="$ROOT/Libraries/MLXLLM/Models/Phi.swift"
PHI3="$ROOT/Libraries/MLXLLM/Models/Phi3.swift"
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

reject_text "$PHI" 'fatalError("hidden_size must be divisible by num_heads")' \
  'Phi hidden/head fatal'
reject_text "$PHI3" 'precondition(args.vocabularySize > 0)' \
  'Phi3 vocabulary precondition'

require_text "$PHI" 'Phi hidden_size must be divisible by num_attention_heads.' \
  'Phi hidden/head validation'
require_text "$PHI" 'Phi num_attention_heads must be divisible by num_key_value_heads.' \
  'Phi KV-head validation'
require_text "$PHI" 'Phi rotary dimension must be positive and no larger than head_dim.' \
  'Phi rotary validation'

require_text "$PHI3" 'Phi3 hidden_size must be divisible by num_attention_heads.' \
  'Phi3 hidden/head validation'
require_text "$PHI3" 'Phi3 num_attention_heads must be divisible by num_key_value_heads.' \
  'Phi3 KV-head validation'
require_text "$PHI3" 'Phi3 rotary dimension must be positive and no larger than head_dim.' \
  'Phi3 rotary validation'
require_text "$PHI3" 'Phi3 rope_scaling.factor must be a positive float.' \
  'Phi3 linear rope factor validation'
require_text "$PHI3" 'Phi3 longrope/su rope_scaling must include positive short_factor and long_factor arrays.' \
  'Phi3 longrope/su factor presence validation'
require_text "$PHI3" 'Phi3 longrope/su factor arrays must match half the attention head dimension.' \
  'Phi3 longrope/su factor length validation'
require_text "$PHI3" 'Phi3 rope_scaling.type must be linear, su, or longrope.' \
  'Phi3 rope scaling type validation'
require_text "$PHI3" 'Model configuration error: Neither tied embeddings nor lm_head is available' \
  'documented Phi3 lm_head internal invariant'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: Phi/Phi3 config/init fatal boundaries are guarded at decode time.\n'
