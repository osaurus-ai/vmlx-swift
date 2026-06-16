#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OMNI="$ROOT/Libraries/MLXVLM/Models/NemotronHOmni/NemotronHOmni.swift"
failures=0

require_text() {
  local pattern="$1"
  local label="$2"
  if ! grep -Fq "$pattern" "$OMNI"; then
    printf 'FAIL: missing %s\n  pattern: %s\n  file: %s\n' "$label" "$pattern" "$OMNI" >&2
    failures=$((failures + 1))
  fi
}

require_text 'try validateRadioPixelValues(pixelValues, modality: "image", expectedChannels: 3)' \
  'image RADIO pixel guard before extraction'
require_text 'try validateRadioPixelValues(video.pixels, modality: "video", expectedChannels: 6)' \
  'video RADIO pixel guard before extraction'
require_text 'private func validateRadioPixelValues(' \
  'NemotronHOmni RADIO pixel guard helper'
require_text 'NemotronHOmni \(modality) pixels must have rank 4 [N,C,H,W]' \
  'rank failure is a typed VLMError'
require_text 'height % config.visionPatchSize == 0, width % config.visionPatchSize == 0' \
  'patch divisibility guard'
require_text 'gridH <= config.visionMaxGrid, gridW <= config.visionMaxGrid' \
  'RADIO max-grid guard before positional interpolation'
require_text 'resize media before prepare' \
  'user/actionable oversized media diagnostic'
require_text 'throw VLMError.processing(' \
  'NemotronHOmni uses typed VLMError processing failures'
require_text 'NemotronHOmni.prepare expects single-sequence input (batch=1)' \
  'batched text-only input fails gracefully instead of fatalError'
require_text 'NemotronHOmni RADIO patch count must be a perfect square' \
  'RADIO image patch mismatch is a typed VLMError'
require_text 'NemotronHOmni RADIO video patch grid mismatch' \
  'RADIO video grid mismatch is a typed VLMError'
require_text 'NemotronHOmni multimodal placeholder count' \
  'placeholder/replacement mismatch is a typed VLMError'
require_text 'NemotronHOmni spliceAtToken currently supports batch=1 only' \
  'multimodal splice batch mismatch is a typed VLMError'

if grep -Eq 'fatalError\(|precondition\(' "$OMNI"; then
  printf 'FAIL: NemotronHOmni still has process-fatal validation paths.\n  file: %s\n' "$OMNI" >&2
  failures=$((failures + 1))
fi

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: NemotronHOmni validates RADIO media tensor shape before tower eval.\n'
