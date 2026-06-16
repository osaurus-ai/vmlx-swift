#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRANITE="$ROOT/Libraries/MLXLLM/Models/GraniteMoeHybrid.swift"
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

reject_text "$GRANITE" 'GraniteMoeHybridMamba2Mixer requires Mamba parameters' \
  'Mamba missing-parameter fatal'
reject_text "$GRANITE" 'GraniteMoeHybridMoE requires MoE parameters' \
  'MoE missing-parameter fatal'
reject_text "$GRANITE" 'GraniteMoeHybridSharedMLP requires shared_intermediate_size' \
  'shared MLP missing-parameter fatal'
reject_text "$GRANITE" 'precondition(args.vocabularySize > 0)' \
  'vocabulary precondition'

require_text "$GRANITE" 'GraniteMoeHybrid hidden_size must be divisible by num_attention_heads.' \
  'hidden/head validation'
require_text "$GRANITE" 'GraniteMoeHybrid num_attention_heads must be divisible by num_key_value_heads.' \
  'KV-head validation'
require_text "$GRANITE" 'GraniteMoeHybrid layer_types count must match num_hidden_layers.' \
  'layer type count validation'
require_text "$GRANITE" 'GraniteMoeHybrid layer_types entries must be mamba or attention.' \
  'layer type enum validation'
require_text "$GRANITE" 'GraniteMoeHybrid mamba layers require complete Mamba parameters.' \
  'Mamba required metadata validation'
require_text "$GRANITE" 'GraniteMoeHybrid MoE layers require num_local_experts, num_experts_per_tok, and shared_intermediate_size.' \
  'MoE required metadata validation'
require_text "$GRANITE" 'GraniteMoeHybrid num_experts_per_tok must be less than or equal to num_local_experts.' \
  'MoE top-k validation'
require_text "$GRANITE" 'GraniteMoeHybrid time_step_limit must contain two positive ascending values.' \
  'time step validation'
require_text "$GRANITE" 'GraniteMoeHybrid position_embedding_type must be rope or nope.' \
  'position embedding validation'
require_text "$GRANITE" 'let numHeads = args.mambaHeads!' \
  'post-decode Mamba metadata use'
require_text "$GRANITE" 'let numExperts = args.numLocalExperts!' \
  'post-decode MoE metadata use'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: GraniteMoeHybrid config/init fatal boundaries are guarded at decode time.\n'
