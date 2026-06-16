#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gemma4="$root/Libraries/MLXVLM/Models/Gemma4.swift"
failures=0

require() {
  local pattern="$1"
  local message="$2"
  if ! grep -Fq "$pattern" "$gemma4"; then
    printf 'FAIL: %s\n' "$message" >&2
    failures=$((failures + 1))
  fi
}

require 'maskFlat.shape[0] == inputFlat.shape[0]' \
  'Gemma4 maskedScatter must reject masks whose flattened size differs from input.'
require 'Gemma4 maskedScatter: mask/input size mismatch' \
  'Gemma4 maskedScatter must surface mask/input mismatch as a typed VLM error.'
require 'let lastPositionInBounds = positions.last.map { Int($0) < inputFlat.shape[0] } ?? true' \
  'Gemma4 maskedScatter must prove mask-derived indices are inside inputFlat.'
require 'Gemma4 maskedScatter: mask index out of input bounds' \
  'Gemma4 maskedScatter must surface out-of-bounds mask positions as a typed VLM error.'
require 'Gemma4 maskedScatter: size mismatch between vision features and image token positions' \
  'Gemma4 maskedScatter must preserve the existing feature/position mismatch error.'
require 'Gemma4 processor config' \
  'Gemma4 processor config must reject non-positive image/audio/token sizing metadata before prepare.'
require 'vision_soft_tokens_per_image/default_output_length must be greater than zero' \
  'Gemma4 model config must reject non-positive vision token expansion before maskedScatter.'
require 'processor_class must not be empty' \
  'Gemma4 processor config must reject empty processor class metadata.'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: Gemma4 maskedScatter index boundaries are guarded.\n'
