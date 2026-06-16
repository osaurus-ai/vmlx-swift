#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_file="$root/Libraries/MLXVLM/Models/Qwen2VL.swift"

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
  'Qwen2-VL configuration types must decode through throwing validation.'
require 'ropeScaling\?\["mrope_section"\]\?\.asInts\(\)' \
  'Qwen2-VL text config must validate rope_scaling.mrope_section before model init.'
require 'Qwen2-VL rope_scaling\.mrope_section must be an array of positive integers' \
  'Qwen2-VL invalid mrope_section must become a configuration decoding error.'
require 'hiddenSize % attentionHeads == 0' \
  'Qwen2-VL text hidden/head divisibility must be validated before attention init.'
require 'attentionHeads % kvHeads == 0' \
  'Qwen2-VL text KV-head divisibility must be validated before attention init.'
require 'embedDimensions % numHeads == 0' \
  'Qwen2-VL vision embed/head divisibility must be validated before vision attention init.'
require 'patchSize % spatialMergeSize == 0' \
  'Qwen2-VL vision patch/merge divisibility must be validated before preprocessing.'
require 'vocabularySize, key: \.vocabularySize' \
  'Qwen2-VL vocab size must be validated before embedding construction.'
require 'validateRGBTriplet\(imageMean, key: \.imageMean' \
  'Qwen2-VL processor image_mean must be validated before RGB tuple indexing.'
require 'validateRGBTriplet\(imageStd, key: \.imageStd' \
  'Qwen2-VL processor image_std must be validated before RGB tuple indexing.'
require 'patchSize % mergeSize == 0' \
  'Qwen2-VL processor patch/merge divisibility must be validated before preprocessing.'
require 'maxPixels >= minPixels' \
  'Qwen2-VL processor pixel budget must be validated before resize math.'
reject 'fatalError\("rope_scaling' \
  'Qwen2-VL must not fatalError on invalid rope_scaling metadata.'
reject 'precondition\(args\.vocabularySize > 0\)' \
  'Qwen2-VL must not rely on a process-fatal precondition for vocabulary size.'

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi

printf 'PASS: Qwen2-VL invalid bundle metadata fails during config decode, not model init fatal paths.\n'
