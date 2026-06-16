#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN="$ROOT/Libraries/MLXLLM/Models/Qwen35JANGTQ.swift"

require_text() {
  local file="$1"
  local text="$2"
  local label="$3"
  if ! grep -Fq "$text" "$file"; then
    printf 'FAIL: %s missing %s\n' "$file" "$label" >&2
    return 1
  fi
}

reject_text() {
  local file="$1"
  local text="$2"
  local label="$3"
  if grep -Fq "$text" "$file"; then
    printf 'FAIL: %s still contains forbidden %s\n' "$file" "$label" >&2
    return 1
  fi
}

reject_text "$QWEN" 'precondition(args.vocabularySize > 0)' \
  'Qwen35JANGTQ vocabulary precondition'

require_text "$QWEN" 'try validateDecodedFields(container: container)' \
  'Qwen35JANGTQ decode validation call'
require_text "$QWEN" 'try Self.validatePositive(vocabularySize, key: .vocabularySize, in: container)' \
  'Qwen35JANGTQ vocabulary validation'
require_text "$QWEN" 'Qwen35JANGTQ mxtq_bits must be 2 or 4.' \
  'Qwen35JANGTQ mxtq bit validation'
require_text "$QWEN" 'try Self.validateNonNegative(mxtqSeed, key: .mxtqSeed, in: container)' \
  'Qwen35JANGTQ mxtq seed validation'
require_text "$QWEN" 'Qwen35JANGTQ hidden_size must be divisible by num_attention_heads.' \
  'Qwen35JANGTQ hidden/head validation'
require_text "$QWEN" 'Qwen35JANGTQ num_attention_heads must be divisible by num_key_value_heads.' \
  'Qwen35JANGTQ KV-head validation'
require_text "$QWEN" 'Qwen35JANGTQ linear_num_value_heads must be divisible by linear_num_key_heads.' \
  'Qwen35JANGTQ linear head validation'
require_text "$QWEN" 'Qwen35JANGTQ MoE config incoherent: num_experts_per_tok must be in 1...num_experts when enabled.' \
  'Qwen35JANGTQ MoE top-k validation'

printf 'PASS: Qwen35JANGTQ config/init fatal boundaries are guarded at decode time.\n'
