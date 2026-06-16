#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$ROOT/Libraries/MLXVLM/Models/Gemma3.swift"

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

reject_text "$MODEL" 'fatalError("Either inputs or inputEmbedding must be provided")' \
  'Gemma3 missing input fatal'
reject_text "$MODEL" '_ inputs: MLXArray? = nil,' \
  'optional private Gemma3 token input surface'
reject_text "$MODEL" 'nil,  // Pass nil for tokens when using embeddings' \
  'Gemma3 nil-token embedding prefill call'

require_text "$MODEL" 'func embeddingsForward(' \
  'throwing Gemma3 embedding forward path'
require_text "$MODEL" 'throw VLMError.processing(' \
  'typed Gemma3 prepare failure'

printf 'PASS: Gemma3 private prepare path cannot enter nil token/embedding native fatal state.\n'
