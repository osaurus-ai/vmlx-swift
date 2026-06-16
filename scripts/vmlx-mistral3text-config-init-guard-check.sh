#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEXT="$ROOT/Libraries/MLXLLM/Models/Mistral3Text.swift"
JANGTQ="$ROOT/Libraries/MLXLLM/Models/Mistral3TextJANGTQ.swift"
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

reject_text "$TEXT" 'precondition(args.vocabularySize > 0)' \
  'vanilla vocabulary precondition'
reject_text "$JANGTQ" 'precondition(args.vocabularySize > 0)' \
  'JANGTQ vocabulary precondition'

require_text "$TEXT" 'try validateDecodedFields(container: container)' \
  'decode validation call'
require_text "$TEXT" 'try validatePositive(vocabularySize, key: .vocabularySize, in: container)' \
  'vocab_size validation'
require_text "$TEXT" 'Mistral3Text hidden_size must equal num_attention_heads * head_dim.' \
  'hidden/head validation'
require_text "$TEXT" 'Mistral3Text num_attention_heads must be divisible by num_key_value_heads.' \
  'KV-head validation'
require_text "$TEXT" 'Mistral3Text layer_types count must match num_hidden_layers.' \
  'layer_types count validation'
require_text "$TEXT" 'Mistral3Text layer_types entries must be full_attention or sliding_attention.' \
  'layer_types enum validation'
require_text "$TEXT" 'Mistral3Text sliding_attention layers require sliding_window.' \
  'sliding_window validation'
require_text "$TEXT" 'Mistral3Text rope_parameters.rope_theta must be finite and > 0.' \
  'rope theta validation'
require_text "$TEXT" 'Mistral3Text rope_parameters.original_max_position_embeddings must be > 0.' \
  'llama4 original positions validation'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: Mistral3Text config/init fatal boundaries are guarded at decode time.\n'
