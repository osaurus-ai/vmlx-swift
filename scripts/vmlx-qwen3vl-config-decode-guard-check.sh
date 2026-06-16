#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_file="$root/Libraries/MLXVLM/Models/Qwen3VL.swift"

fail=0

require() {
  local pattern="$1"
  local message="$2"
  if ! rg -q "$pattern" "$source_file"; then
    printf 'FAIL: %s\n' "$message" >&2
    fail=1
  fi
}

reject() {
  local pattern="$1"
  local message="$2"
  if rg -q "$pattern" "$source_file"; then
    printf 'FAIL: %s\n' "$message" >&2
    fail=1
  fi
}

require 'public init\(from decoder: any Swift\.Decoder\) throws' \
  'Qwen3-VL configuration types must decode through throwing validation.'
require 'hiddenSize == numAttentionHeads \* headDim' \
  'Qwen3-VL text hidden/head dimensions must be validated before attention init.'
require 'numAttentionHeads % numKeyValueHeads == 0' \
  'Qwen3-VL text KV-head divisibility must be validated before attention init.'
require 'ropeScaling\?\.mropeSection' \
  'Qwen3-VL mrope_section must be validated when present.'
require 'hiddenSize % numHeads == 0' \
  'Qwen3-VL vision hidden/head dimensions must be validated before vision init.'
require 'patchSize % spatialMergeSize == 0' \
  'Qwen3-VL vision patch/merge divisibility must be validated before preprocessing.'
require 'vocabSize, key: \.vocabSize' \
  'Qwen3-VL vocab size must be validated before embedding construction.'
require 'validateRGBTriplet\(imageMean, key: \.imageMean' \
  'Qwen3-VL processor image_mean must be validated before RGB tuple indexing.'
require 'validateRGBTriplet\(imageStd, key: \.imageStd' \
  'Qwen3-VL processor image_std must be validated before RGB tuple indexing.'
require 'patchSize % mergeSize == 0' \
  'Qwen3-VL processor patch/merge divisibility must be validated before preprocessing.'
require 'maxPixels >= minPixels' \
  'Qwen3-VL processor pixel budget must be validated before resize math.'
require 'video max_frames must be greater than or equal to min_frames' \
  'Qwen3-VL video processor frame budget must be validated before video preprocessing.'
reject 'precondition\(config\.vocabSize > 0\)' \
  'Qwen3-VL must not rely on a process-fatal precondition for vocabulary size.'

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi

printf 'PASS: Qwen3-VL invalid bundle metadata fails during config decode, not model init fatal paths.\n'
