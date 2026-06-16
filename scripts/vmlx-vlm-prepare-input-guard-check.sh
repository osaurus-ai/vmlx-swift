#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GLM="$ROOT/Libraries/MLXVLM/Models/GlmOcr.swift"
FAST="$ROOT/Libraries/MLXVLM/Models/FastVLM.swift"

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

for file in "$GLM" "$FAST"; do
  reject_text "$file" 'fatalError("one of inputs or inputEmbedding must be non-nil")' \
    'missing text/inputEmbedding fatal'
  reject_text "$file" '_ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil' \
    'nil/nil private language model call surface'
  reject_text "$file" 'languageModel(nil, cache: cache, inputEmbedding:' \
    'nil-token VLM prepare call'
  require_text "$file" 'func embeddingsForward(' \
    'explicit embedding forward path'
  require_text "$file" 'throw VLMError.processing(' \
    'typed VLM prepare failure'
done

printf 'PASS: GlmOcr/FastVLM private prepare paths cannot enter nil token/embedding native fatal state.\n'
