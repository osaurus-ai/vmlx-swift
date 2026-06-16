#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BAILING="$ROOT/Libraries/MLXLLM/Models/BailingMoe.swift"
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

reject_text "$BAILING" 'precondition(args.vocabularySize > 0)' \
  'vocabulary precondition'

require_text "$BAILING" 'try validateDecodedFields(container: container)' \
  'decode validation call'
require_text "$BAILING" 'try validatePositive(vocabularySize, key: .vocabularySize, in: container)' \
  'vocab_size validation'
require_text "$BAILING" 'BailingMoe hidden_size must be divisible by num_attention_heads.' \
  'hidden/head validation'
require_text "$BAILING" 'BailingMoe num_attention_heads must be divisible by num_key_value_heads.' \
  'KV-head validation'
require_text "$BAILING" 'BailingMoe rotary dimension must be positive and no larger than head_dim.' \
  'rotary dimension validation'
require_text "$BAILING" 'BailingMoe num_experts_per_tok must be <= num_experts.' \
  'expert top-k validation'
require_text "$BAILING" 'BailingMoe num_experts must be divisible by n_group.' \
  'expert group divisibility validation'
require_text "$BAILING" 'BailingMoe topk_group must be > 0 and <= n_group.' \
  'router group bounds validation'
require_text "$BAILING" 'let droppedGroups = nGroup - topkGroup' \
  'router no-drop group branch'
require_text "$BAILING" 'debugDescription: "BailingMoe \(key.rawValue) must be finite and > 0."' \
  'finite positive float validation'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: BailingMoe config/init fatal boundaries are guarded at decode time.\n'
