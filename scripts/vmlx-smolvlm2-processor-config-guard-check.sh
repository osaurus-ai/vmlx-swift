#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOL="$ROOT/Libraries/MLXVLM/Models/SmolVLM2.swift"
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

require_text "$SMOL" 'try validateRGBTuple(imageMean, key: .imageMean, in: container)' \
  'processor image_mean RGB tuple validation'
require_text "$SMOL" 'try validateRGBTuple(imageStd, key: .imageStd, in: container)' \
  'processor image_std RGB tuple validation'
require_text "$SMOL" 'try validatePositive(size.longestEdge, key: .size, in: container)' \
  'processor size validation'
require_text "$SMOL" 'try validatePositive(maxImageSize.longestEdge, key: .maxImageSize, in: container)' \
  'processor max_image_size validation'
require_text "$SMOL" 'try validatePositive(videoSampling.fps, key: .videoSampling, in: container)' \
  'video fps validation'
require_text "$SMOL" 'try validatePositive(videoSampling.maxFrames, key: .videoSampling, in: container)' \
  'video max_frames validation'
require_text "$SMOL" 'try validatePositive(imageSequenceLength, key: ._imageSequenceLength, in: container)' \
  'image sequence length validation'
require_text "$SMOL" 'SmolVLM2 processor config \(key.rawValue) must contain exactly 3 RGB values.' \
  'RGB tuple error message'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: SmolVLM2 processor config tuple/numeric bounds are guarded at decode time.\n'
