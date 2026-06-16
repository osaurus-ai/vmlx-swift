#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AFMOE="$ROOT/Libraries/MLXLLM/Models/AfMoE.swift"
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

reject_text "$AFMOE" 'precondition(args.vocabularySize > 0)' \
  'vocabulary precondition'

require_text "$AFMOE" 'try validateDecodedFields(container: container)' \
  'decode validation call'
require_text "$AFMOE" 'try validatePositive(vocabularySize, key: .vocabularySize, in: container)' \
  'vocab_size validation'
require_text "$AFMOE" 'AfMoE hidden_size must equal num_attention_heads * head_dim.' \
  'hidden/head validation'
require_text "$AFMOE" 'AfMoE num_attention_heads must be divisible by num_key_value_heads.' \
  'KV-head validation'
require_text "$AFMOE" 'AfMoE num_experts_per_tok must be <= num_experts.' \
  'expert top-k validation'
require_text "$AFMOE" 'AfMoE num_experts must be divisible by n_group.' \
  'expert group divisibility validation'
require_text "$AFMOE" 'AfMoE topk_group must be > 0 and <= n_group.' \
  'router group bounds validation'
require_text "$AFMOE" 'AfMoE layer_types count must equal num_hidden_layers.' \
  'layer type count validation'
require_text "$AFMOE" 'AfMoE layer_types entries must be full_attention or sliding_attention.' \
  'layer type value validation'
require_text "$AFMOE" 'let droppedGroups = nGroup - topkGroup' \
  'router no-drop group branch'
require_text "$AFMOE" 'debugDescription: "AfMoE \(key.rawValue) must be finite and > 0."' \
  'finite positive float validation'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: AfMoE config/init fatal boundaries are guarded at decode time.\n'
