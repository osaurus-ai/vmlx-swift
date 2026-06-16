#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEMO="$ROOT/Libraries/MLXLLM/Models/NemotronHJANGTQ.swift"
CACHE="$ROOT/Libraries/MLXLMCommon/JANGTQKernels.swift"

require_text() {
  local file="$1"
  local text="$2"
  local label="$3"
  if ! grep -Fq "$text" "$file"; then
    printf 'FAIL: %s missing %s\n' "$file" "$label" >&2
    return 1
  fi
}

reject_text() {
  local file="$1"
  local text="$2"
  local label="$3"
  if grep -Fq "$text" "$file"; then
    printf 'FAIL: %s still contains forbidden %s\n' "$file" "$label" >&2
    return 1
  fi
}

reject_text "$NEMO" 'fatalError("JANGTQ sidecar missing signs.' \
  'Nemotron JANGTQ missing-signs fatal'
reject_text "$NEMO" 'fatalError("JANGTQ sidecar missing codebook.' \
  'Nemotron JANGTQ missing-codebook fatal'

require_text "$CACHE" 'public func requiredSigns(inFeatures: Int, seed: Int) -> MLXArray' \
  'deterministic required signs API'
require_text "$CACHE" 'public func requiredCodebook(inFeatures: Int, bits: Int) -> MLXArray' \
  'deterministic required codebook API'
require_text "$NEMO" 'JANGTQRuntimeCache.shared.requiredSigns(' \
  'Nemotron required signs use'
require_text "$NEMO" 'JANGTQRuntimeCache.shared.requiredCodebook(' \
  'Nemotron required codebook use'

printf 'PASS: NemotronH JANGTQ missing sidecar arrays use deterministic runtime fallback.\n'
