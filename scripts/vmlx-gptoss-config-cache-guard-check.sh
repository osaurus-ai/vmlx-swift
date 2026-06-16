#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPTOSS="$ROOT/Libraries/MLXLLM/Models/GPTOSS.swift"
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

reject_text "$GPTOSS" 'fatalError("Quantized attention does not support non-zero sinks.")' \
  'old GPTOSS QuantizedKVCache sinks fatal'
reject_text "$GPTOSS" 'blocks.shape=\(blocks.shape) does not match scales.shape=\(scales.shape)' \
  'packed MoE shape precondition'
reject_text "$GPTOSS" 'out.reshaped(prefixShape.count, G * B * 2)' \
  'packed MoE wrong prefix reshape'

require_text "$GPTOSS" 'GPTOSS num_attention_heads must be divisible by num_key_value_heads.' \
  'GPTOSS attention/KV validation'
require_text "$GPTOSS" 'GPTOSS num_experts_per_tok must be less than or equal to num_local_experts.' \
  'GPTOSS expert top-k validation'
require_text "$GPTOSS" 'GPTOSS layer_types count must match num_hidden_layers.' \
  'GPTOSS layer type count validation'
require_text "$GPTOSS" 'GPTOSS layer_types entries must be sliding_attention or full_attention.' \
  'GPTOSS layer type enum validation'
require_text "$GPTOSS" 'GPTOSS rope_scaling.factor must be a positive float.' \
  'GPTOSS rope factor validation'
require_text "$GPTOSS" 'let floatCache = quantizedCache.toUnquantized()' \
  'GPTOSS built-in QuantizedKVCache sink fallback'
require_text "$GPTOSS" 'quantizedCache.state = refreshed.state' \
  'GPTOSS requantized cache state restore'
require_text "$GPTOSS" 'private func convertMoePackedTensors(blocks: MLXArray, scales: MLXArray) -> MLXArray?' \
  'GPTOSS packed MoE optional conversion'
require_text "$GPTOSS" 'guard blocks.shape.dropLast() == scales.shape else {' \
  'GPTOSS packed MoE shape guard'
require_text "$GPTOSS" 'out.reshaped(prefixShape + [G * B * 2])' \
  'GPTOSS packed MoE prefix reshape'
require_text "$GPTOSS" 'GPTOSS non-zero sinks require concrete QuantizedKVCache fallback support.' \
  'documented custom QuantizedKVCacheProtocol invariant'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: GPTOSS config/cache/sanitize fatal boundaries are guarded.\n'
