#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEMMA3N="$ROOT/Libraries/MLXLLM/Models/Gemma3nText.swift"
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

reject_text "$GEMMA3N" "Layer type 'sliding_attention' not found in layer_types" \
  'missing sliding_attention fatal'
reject_text "$GEMMA3N" "Layer type 'full_attention' not found in layer_types" \
  'missing full_attention fatal'
reject_text "$GEMMA3N" 'assert(vocabSize > 0)' \
  'vocabulary assert'

require_text "$GEMMA3N" 'try requirePositive(hiddenSize, .hiddenSize, "hidden_size")' \
  'hidden_size validation'
require_text "$GEMMA3N" 'try requirePositive(vocabSize, .vocabSize, "vocab_size")' \
  'vocab_size validation'
require_text "$GEMMA3N" 'Gemma3n text config intermediate_size must be a scalar or have num_hidden_layers entries.' \
  'intermediate_size length validation'
require_text "$GEMMA3N" 'Gemma3n text config num_attention_heads must be divisible by num_key_value_heads.' \
  'KV-head validation'
require_text "$GEMMA3N" 'Gemma3n text config num_kv_shared_layers must be in 0..<num_hidden_layers.' \
  'shared KV layer validation'
require_text "$GEMMA3N" 'Gemma3n text config altup_active_idx must be within altup_num_inputs.' \
  'AltUp active index validation'
require_text "$GEMMA3N" 'Gemma3n text config activation_sparsity_pattern must have num_hidden_layers entries.' \
  'activation sparsity length validation'
require_text "$GEMMA3N" 'Gemma3n text config layer_types is required because this runtime needs explicit ' \
  'required layer_types validation'
require_text "$GEMMA3N" 'Gemma3n text config layer_types count must match num_hidden_layers.' \
  'layer_types count validation'
require_text "$GEMMA3N" 'Gemma3n text config layer_types entries must be full_attention or sliding_attention.' \
  'layer_types enum validation'
require_text "$GEMMA3N" 'Gemma3n text config layer_types must include sliding_attention.' \
  'sliding_attention presence validation'
require_text "$GEMMA3N" 'Gemma3n text config layer_types must include full_attention.' \
  'full_attention presence validation'
require_text "$GEMMA3N" 'layerTypes.firstIndex(of: "sliding_attention")!' \
  'post-decode sliding index use'
require_text "$GEMMA3N" 'layerTypes.firstIndex(of: "full_attention")!' \
  'post-decode full index use'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: Gemma3nText config/init fatal boundaries are guarded at decode time.\n'
