#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DENSE="$ROOT/Libraries/MLXLMCommon/JANGTQDenseLinear.swift"
CACHE="$ROOT/Libraries/MLXLMCommon/JANGTQKernels.swift"

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

reject_text "$DENSE" 'fatalError("JANGTQ runtime sidecar not loaded' \
  'JANGTQDenseLinear missing-signs fatal'
reject_text "$DENSE" 'fatalError("JANGTQ codebook missing' \
  'JANGTQDenseLinear missing-codebook fatal'
reject_text "$DENSE" 'fatalError(' \
  'process-fatal JANGTQDenseLinear path'

require_text "$CACHE" 'public func requiredSigns(inFeatures: Int, seed: Int) -> MLXArray' \
  'deterministic requiredSigns accessor'
require_text "$CACHE" 'public func requiredCodebook(inFeatures: Int, bits: Int) -> MLXArray' \
  'deterministic requiredCodebook accessor'
require_text "$DENSE" 'JANGTQRuntimeCache.shared.requiredSigns(' \
  'JANGTQDenseLinear requiredSigns use'
require_text "$DENSE" 'JANGTQRuntimeCache.shared.requiredCodebook(' \
  'JANGTQDenseLinear requiredCodebook use'

printf 'PASS: JANGTQDenseLinear sidecar paths use deterministic runtime fallback.\n'
