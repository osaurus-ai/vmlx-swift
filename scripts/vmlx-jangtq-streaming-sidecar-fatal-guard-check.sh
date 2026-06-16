#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STREAMING="$ROOT/Libraries/MLXLMCommon/JANGTQStreamingExperts.swift"
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

reject_text "$STREAMING" 'JANGTQRuntimeCache.shared.signs(' \
  'optional signs sidecar lookup'
reject_text "$STREAMING" 'JANGTQRuntimeCache.shared.codebook(' \
  'optional codebook sidecar lookup'
reject_text "$STREAMING" 'missing active Nemotron JANGTQ sidecar arrays' \
  'Nemotron streaming sidecar fatal'
reject_text "$STREAMING" 'missing JANGTQ sidecar array(s)' \
  'generic streaming sidecar fatal'

require_text "$CACHE" 'public func requiredSigns(inFeatures: Int, seed: Int) -> MLXArray' \
  'deterministic requiredSigns accessor'
require_text "$CACHE" 'public func requiredCodebook(inFeatures: Int, bits: Int) -> MLXArray' \
  'deterministic requiredCodebook accessor'
require_text "$STREAMING" 'JANGTQRuntimeCache.shared.requiredSigns(' \
  'streaming requiredSigns use'
require_text "$STREAMING" 'JANGTQRuntimeCache.shared.requiredCodebook(' \
  'streaming requiredCodebook use'

printf 'PASS: JANGTQ streaming sidecar paths use deterministic runtime fallback.\n'
