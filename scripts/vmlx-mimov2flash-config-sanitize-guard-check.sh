#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMO="$ROOT/Libraries/MLXLLM/Models/MiMoV2Flash.swift"
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

reject_text "$MIMO" 'MiMoV2FlashMoEGate requires nRoutedExperts' \
  'MoE gate routed-experts fatal'
reject_text "$MIMO" 'MiMoV2FlashMoE requires nRoutedExperts' \
  'MoE routed-experts fatal'
reject_text "$MIMO" 'precondition(config.topkMethod == "noaux_tc"' \
  'topk method precondition'
reject_text "$MIMO" 'MiMoV2Flash missing routed expert FP8 pair' \
  'routed expert FP8 checkpoint fatal'
reject_text "$MIMO" 'MiMoV2Flash affine quantization produced no biases' \
  'routed expert affine quantization fatal'
reject_text "$MIMO" 'unsupported MiMo routed expert bits=' \
  'routed expert bit precondition'
reject_text "$MIMO" 'MiMo routed expert group size must be positive' \
  'routed expert group-size precondition'
reject_text "$MIMO" 'precondition(sinks == nil, "Quantized SDPA does not support attention sinks.")' \
  'quantized attention sink precondition'

require_text "$MIMO" 'self.nRoutedExperts = try container.decode(Int.self, forKey: .nRoutedExperts)' \
  'required routed experts decode'
require_text "$MIMO" 'MiMoV2Flash hybrid_layer_pattern and moe_layer_freq must match num_hidden_layers.' \
  'layer-pattern length validation'
require_text "$MIMO" 'MiMoV2Flash topk_method must be noaux_tc.' \
  'topk method validation'
require_text "$MIMO" 'MiMoV2Flash n_routed_experts must be divisible by n_group.' \
  'routed group divisibility validation'
require_text "$MIMO" 'MiMoV2Flash num_experts_per_tok must be less than or equal to n_routed_experts.' \
  'top-k expert bounds validation'
require_text "$MIMO" 'MiMoV2Flash rotary dimensions must be positive and no larger than the head dimensions.' \
  'rotary dimension validation'
require_text "$MIMO" 'hasCompleteExpertSet(' \
  'routed expert checkpoint completeness check'
require_text "$MIMO" 'MiMoV2Flash routed expert bits must use supported projections and bit widths.' \
  'routed expert bit validation'
require_text "$MIMO" 'let floatCache = quantizedCache.toUnquantized()' \
  'built-in QuantizedKVCache sink fallback'
require_text "$MIMO" 'quantizedCache.state = refreshed.state' \
  'requantized cache state restore'
require_text "$MIMO" 'MiMoV2Flash attention sinks require concrete QuantizedKVCache fallback support.' \
  'documented custom QuantizedKVCacheProtocol invariant'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: MiMoV2Flash config/cache/sanitize fatal boundaries are guarded.\n'
