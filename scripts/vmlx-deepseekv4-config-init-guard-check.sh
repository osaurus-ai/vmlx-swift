#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/Libraries/MLXLLM/Models/DeepseekV4Configuration.swift"

require_text() {
  local file="$1"
  local text="$2"
  local label="$3"
  if ! grep -Fq "$text" "$file"; then
    printf 'FAIL: %s missing %s\n' "$file" "$label" >&2
    return 1
  fi
}

require_text "$CONFIG" 'try validateDecodedFields(container: c)' \
  'DeepseekV4 decode validation call'
require_text "$CONFIG" 'try Self.validatePositive(vocabSize, key: .vocabSize, in: c)' \
  'DeepseekV4 vocabulary validation'
require_text "$CONFIG" 'DeepseekV4 qk_rope_head_dim must be in 1...head_dim.' \
  'DeepseekV4 RoPE/head-dim validation'
require_text "$CONFIG" 'DeepseekV4 compress_ratios entries must be one of 0, 4, or 128.' \
  'DeepseekV4 compressor ratio validation'
require_text "$CONFIG" 'DeepseekV4 compress_ratios count must match num_hidden_layers when provided.' \
  'DeepseekV4 per-layer ratio count validation'
require_text "$CONFIG" 'DeepseekV4 num_experts_per_tok must be in 1...n_routed_experts.' \
  'DeepseekV4 MoE top-k validation'
require_text "$CONFIG" 'DeepseekV4 num_hash_layers must be in 0...num_hidden_layers.' \
  'DeepseekV4 hash-layer validation'
require_text "$CONFIG" 'DeepseekV4 routed expert bits must be 2 or 4.' \
  'DeepseekV4 routed expert bit validation'
require_text "$CONFIG" 'try Self.validatePositive(ropeTheta, key: .ropeTheta, in: c)' \
  'DeepseekV4 rope theta validation'
require_text "$CONFIG" 'try Self.validatePositive(compressRopeTheta, key: .compressRopeTheta, in: c)' \
  'DeepseekV4 compress rope theta validation'

printf 'PASS: DeepseekV4 config/init fatal boundaries are guarded at decode time.\n'
