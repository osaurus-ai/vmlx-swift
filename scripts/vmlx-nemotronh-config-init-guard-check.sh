#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEMOTRON="$ROOT/Libraries/MLXLLM/Models/NemotronH.swift"
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

require_text "$NEMOTRON" 'static func decode(_ char: Character) -> NemotronHBlockType?' \
  'failable NemotronH block-type decode'
require_text "$NEMOTRON" 'internal var blockTypes: [NemotronHBlockType]' \
  'single validated block-type source'
require_text "$NEMOTRON" 'NemotronH config hybrid_override_pattern length must equal num_hidden_layers.' \
  'hybrid_override_pattern length validation'
require_text "$NEMOTRON" 'NemotronH config hybrid_override_pattern must contain only M, *, -, or E.' \
  'hybrid_override_pattern character validation'
require_text "$NEMOTRON" 'NemotronH config hidden_size must be divisible by num_attention_heads.' \
  'attention head divisibility validation'
require_text "$NEMOTRON" 'NemotronH config num_attention_heads must be divisible by num_key_value_heads.' \
  'KV head divisibility validation'
require_text "$NEMOTRON" 'NemotronH config mamba_num_heads * mamba_head_dim must be divisible by n_groups.' \
  'Mamba grouped norm divisibility validation'
require_text "$NEMOTRON" 'NemotronH config num_experts_per_tok must not exceed n_routed_experts.' \
  'MoE top-k validation'
require_text "$NEMOTRON" 'NemotronH config time_step_limit max must be >= min.' \
  'time-step range validation'
require_text "$NEMOTRON" 'return configuration.blockTypes.compactMap' \
  'newCache uses validated block types'
require_text "$NEMOTRON" 'self.kvHeads = args.blockTypes.compactMap' \
  'kvHeads uses validated block types'
reject_text "$NEMOTRON" 'fatalError("Unknown NemotronH block type' \
  'old unknown block-type fatalError'
reject_text "$NEMOTRON" 'NemotronHBlockType(from:' \
  'old raw block-type force decode'
reject_text "$NEMOTRON" 'precondition(args.vocabSize > 0' \
  'constructor vocabulary-size precondition'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: NemotronH config/init fatal boundaries are guarded at decode time.\n'
