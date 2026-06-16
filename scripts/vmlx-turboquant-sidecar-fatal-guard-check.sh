#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TQ="$ROOT/Libraries/MLXLMCommon/TurboQuantSwitchLinear.swift"
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

reject_text "$TQ" 'fatalError("JANGTQ runtime sidecar not loaded' \
  'TurboQuantSwitchLinear missing-signs fatal'
reject_text "$TQ" 'fatalError("JANGTQ codebook missing' \
  'TurboQuantSwitchLinear missing-codebook fatal'
reject_text "$TQ" 'fatalError(' \
  'TurboQuantSwitchGLU sidecar fatal'

require_text "$CACHE" 'public func requiredSigns(inFeatures: Int, seed: Int) -> MLXArray' \
  'deterministic required signs API'
require_text "$CACHE" 'public func requiredCodebook(inFeatures: Int, bits: Int) -> MLXArray' \
  'deterministic required codebook API'
require_text "$TQ" 'JANGTQRuntimeCache.shared.requiredSigns(' \
  'TurboQuant required signs use'
require_text "$TQ" 'JANGTQRuntimeCache.shared.requiredCodebook(' \
  'TurboQuant required codebook use'

printf 'PASS: shared TurboQuant JANGTQ sidecar paths use deterministic runtime fallback.\n'
