#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
Q2="$ROOT/Libraries/MLXVLM/Models/Qwen2VL.swift"
Q25="$ROOT/Libraries/MLXVLM/Models/Qwen25VL.swift"

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

for file in "$Q2" "$Q25"; do
  reject_text "$file" 'fatalError("one of inputs or inputEmbedding must be non-nil")' \
    'missing text/inputEmbedding fatal'
  reject_text "$file" '_ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil' \
    'nil/nil private language model call surface'
  require_text "$file" 'func embeddingsForward(' \
    'throwing embedding forward path'
  require_text "$file" 'throw VLMError.processing(' \
    'typed VLM prepare failure'
done

printf 'PASS: Qwen2/Qwen2.5-VL missing prepare inputs throw typed VLM errors.\n'
