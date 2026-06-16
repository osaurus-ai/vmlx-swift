#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

llm="$root/Libraries/MLXLLM/Models/Gemma4Text.swift"
vlm="$root/Libraries/MLXVLM/Models/Gemma4.swift"

fail=0

require() {
  local file="$1"
  local pattern="$2"
  local message="$3"
  if ! rg -q "$pattern" "$file"; then
    printf 'FAIL: %s\n  missing pattern: %s\n  file: %s\n' "$message" "$pattern" "$file" >&2
    fail=1
  fi
}

require "$llm" 'validateGemma4MoERouterConfig' \
  'Gemma4TextConfiguration must validate MoE router top-k before Gemma4Router indexes.'
require "$vlm" 'validateGemma4MoERouterConfig' \
  'G4TextConfig must validate MoE router top-k before VLM TextRouter indexes.'

require "$llm" 'topKExperts > 0' \
  'Gemma4 LLM must reject enabled MoE with non-positive top_k_experts.'
require "$vlm" 'topKExperts > 0' \
  'Gemma4 VLM must reject enabled MoE with non-positive top_k_experts.'

require "$llm" 'topKExperts <= numExperts' \
  'Gemma4 LLM must reject top_k_experts greater than num_experts.'
require "$vlm" 'topKExperts <= numExperts' \
  'Gemma4 VLM must reject top_k_experts greater than num_experts.'

require "$llm" 'Gemma4 MoE config incoherent' \
  'Gemma4 LLM MoE validation must surface a typed config error.'
require "$vlm" 'Gemma4 MoE config incoherent' \
  'Gemma4 VLM MoE validation must surface a typed config error.'
require "$vlm" 'Gemma4 text config.*must be greater than zero' \
  'Gemma4 VLM text config must reject non-positive dimensions before router construction.'
require "$vlm" 'layer_types count' \
  'Gemma4 VLM text config must reject layer_types count mismatches before per-layer routing.'
require "$vlm" 'Gemma4 vision config.*must be greater than zero' \
  'Gemma4 VLM vision config must reject non-positive dimensions before vision construction.'

require "$root/Libraries/MLXLMCommon/Load.swift" 'LoadedWeightsValidatingModel' \
  'loadWeights must support post-load model-specific checkpoint validation.'
require "$root/Libraries/MLXLMCommon/Load.swift" 'validatingModel\.validateLoadedWeights' \
  'loadWeights must run Gemma4 router checkpoint validation after parameters are loaded.'
require "$llm" 'validateGemma4RouterLoadedWeights' \
  'Gemma4 LLM must validate loaded router/projection/per-expert checkpoint shapes.'
require "$vlm" 'validateGemma4VLMRouterLoadedWeights' \
  'Gemma4 VLM must validate loaded router/projection/per-expert checkpoint shapes.'
require "$llm" 'per_expert_scale' \
  'Gemma4 LLM must reject per_expert_scale shape mismatches before TextRouter indexes.'
require "$vlm" 'per_expert_scale' \
  'Gemma4 VLM must reject per_expert_scale shape mismatches before TextRouter indexes.'
require "$llm" 'proj\.weight.*expected first dimension' \
  'Gemma4 LLM must reject router projection output-dimension mismatches before TextRouter indexes.'
require "$vlm" 'proj\.weight.*expected first dimension' \
  'Gemma4 VLM must reject router projection output-dimension mismatches before TextRouter indexes.'

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi

printf 'PASS: Gemma4 MoE router config and checkpoint-shape guards are present in LLM and VLM paths.\n'
