#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FILES=(
  "$ROOT/Libraries/MLXLLM/Models/Qwen2.swift"
  "$ROOT/Libraries/MLXLLM/Models/Qwen3.swift"
  "$ROOT/Libraries/MLXLLM/Models/Qwen3MoE.swift"
  "$ROOT/Libraries/MLXLLM/Models/Internlm2.swift"
)
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

for file in "${FILES[@]}"; do
  reject_text "$file" 'fatalError("ropeScaling.factor must be a float")' \
    'ropeScaling factor fatal'
  reject_text "$file" 'precondition(args.vocabularySize > 0)' \
    'vocabulary precondition'
  require_text "$file" 'var ropeScale: Float' \
    'validated ropeScale property'
done

require_text "${FILES[0]}" 'Qwen2 rope_scaling.factor must be a positive float.' \
  'Qwen2 rope factor validation'
require_text "${FILES[0]}" 'Qwen2 hidden_size must be divisible by num_attention_heads.' \
  'Qwen2 hidden/head validation'
require_text "${FILES[0]}" 'Qwen2 num_attention_heads must be divisible by num_key_value_heads.' \
  'Qwen2 KV-head validation'

require_text "${FILES[1]}" 'Qwen3 rope_scaling.factor must be a positive float.' \
  'Qwen3 rope factor validation'
require_text "${FILES[1]}" 'Qwen3 hidden_size must equal num_attention_heads * head_dim.' \
  'Qwen3 hidden/head_dim validation'
require_text "${FILES[1]}" 'Qwen3 num_attention_heads must be divisible by num_key_value_heads.' \
  'Qwen3 KV-head validation'

require_text "${FILES[2]}" 'Qwen3MoE rope_scaling.factor must be a positive float.' \
  'Qwen3MoE rope factor validation'
require_text "${FILES[2]}" 'Qwen3MoE num_experts_per_tok must be less than or equal to num_experts.' \
  'Qwen3MoE top-k expert validation'
require_text "${FILES[2]}" 'Qwen3MoE mlp_only_layers entries must be valid layer indexes.' \
  'Qwen3MoE mlp_only_layers validation'

require_text "${FILES[3]}" 'InternLM2 rope_scaling.factor must be a positive float.' \
  'InternLM2 rope factor validation'
require_text "${FILES[3]}" 'InternLM2 hidden_size must be divisible by num_attention_heads.' \
  'InternLM2 hidden/head validation'
require_text "${FILES[3]}" 'InternLM2 num_attention_heads must be divisible by num_key_value_heads.' \
  'InternLM2 KV-head validation'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: Qwen-family LLM rope/vocab config fatal boundaries are guarded.\n'
