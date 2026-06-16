#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$ROOT/Libraries/MLXVLM/Models/Qwen3VL.swift"

require_text() {
  local file="$1"
  local text="$2"
  local label="$3"
  if ! grep -Fq "$text" "$file"; then
    printf 'FAIL: %s missing %s\n' "$file" "$label" >&2
    exit 1
  fi
}

reject_text() {
  local file="$1"
  local text="$2"
  local label="$3"
  if grep -Fq "$text" "$file"; then
    printf 'FAIL: %s still contains forbidden %s\n' "$file" "$label" >&2
    exit 1
  fi
}

reject_text "$MODEL" 'fatalError("Either input ids or embeddings must be provided")' \
  'Qwen3-VL missing input fatal'
reject_text "$MODEL" '_ inputIds: MLXArray?,' \
  'optional private Qwen3-VL input id surface'
reject_text "$MODEL" 'inputEmbeddings!' \
  'force-unwrapped Qwen3-VL input embeddings'
reject_text "$MODEL" 'inputIds ?? inputEmbeddings!' \
  'nil/nil Qwen3-VL dimension fallback'

require_text "$MODEL" 'hidden = inputEmbeddings ?? embedTokens(inputIds)' \
  'nonoptional input ids or explicit embeddings path'

printf 'PASS: Qwen3-VL private prepare path cannot enter nil input-id/native fatal state.\n'
