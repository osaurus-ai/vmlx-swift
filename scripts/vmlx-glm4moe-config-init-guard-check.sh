#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOE="$ROOT/Libraries/MLXLLM/Models/GLM4MOE.swift"
LITE="$ROOT/Libraries/MLXLLM/Models/GLM4MOELite.swift"
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

for file in "$MOE" "$LITE"; do
  reject_text "$file" 'requires nRoutedExperts' 'nRoutedExperts fatal'
  reject_text "$file" 'precondition(config.topkMethod == "noaux_tc"' 'topk method precondition'
  reject_text "$file" 'precondition(args.vocabularySize > 0)' 'vocabulary precondition'
  reject_text "$file" 'precondition(config.vocabularySize > 0)' 'vocabulary precondition'
done

require_text "$MOE" 'self.nRoutedExperts = try container.decode(Int.self, forKey: .nRoutedExperts)' \
  'GLM4MoE required routed experts decode'
require_text "$MOE" 'GLM4MoE topk_method must be noaux_tc.' \
  'GLM4MoE topk method validation'
require_text "$MOE" 'GLM4MoE n_routed_experts must be divisible by n_group.' \
  'GLM4MoE routed group divisibility validation'
require_text "$MOE" 'GLM4MoE num_experts_per_tok must be less than or equal to n_routed_experts.' \
  'GLM4MoE top-k expert bounds validation'
require_text "$MOE" 'GLM4MoE hidden_size must equal num_attention_heads * head_dim.' \
  'GLM4MoE attention dimension validation'
require_text "$MOE" 'GLM4MoE rotary dimension must be positive and no larger than head_dim.' \
  'GLM4MoE rotary dimension validation'

require_text "$LITE" 'self.nRoutedExperts = try container.decode(Int.self, forKey: .nRoutedExperts)' \
  'GLM4MoELite required routed experts decode'
require_text "$LITE" 'GLM4MoELite topk_method must be noaux_tc.' \
  'GLM4MoELite topk method validation'
require_text "$LITE" 'GLM4MoELite n_routed_experts must be divisible by n_group.' \
  'GLM4MoELite routed group divisibility validation'
require_text "$LITE" 'GLM4MoELite num_experts_per_tok must be less than or equal to n_routed_experts.' \
  'GLM4MoELite top-k expert bounds validation'
require_text "$LITE" 'GLM4MoELite rotary dimension must be positive and no larger than qk_rope_head_dim.' \
  'GLM4MoELite rotary dimension validation'
require_text "$LITE" 'fatalError("Module must be MultiLinear or QuantizedMultiLinear")' \
  'documented internal MultiLinear type invariant'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: GLM4 MoE config/init fatal boundaries are guarded at decode time.\n'
