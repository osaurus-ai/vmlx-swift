#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAST="$ROOT/Libraries/MLXVLM/Models/FastVLM.swift"
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

reject_text() {
  local file="$1"
  local pattern="$2"
  local label="$3"
  if grep -Fq "$pattern" "$file"; then
    printf 'FAIL: %s still contains forbidden %s\n' "$file" "$label" >&2
    failures=$((failures + 1))
  fi
}

require_text "$FAST" 'FastVLM vision config stage arrays must all match layers count.' \
  'vision stage-array length validation'
require_text "$FAST" 'FastVLM vision config token_mixers values must be repmixer or attention.' \
  'token mixer validation'
require_text "$FAST" 'FastVLM attention token mixer embed_dim must be divisible by 32.' \
  'attention token mixer head-dim validation'
require_text "$FAST" 'FastVLM vision config pos_embs_shapes entries must be positive 2D shapes.' \
  'position embedding shape validation'
require_text "$FAST" 'FastVLM vision config image_size must be divisible by patch_size.' \
  'vision patch divisibility validation'
require_text "$FAST" 'FastVLM base config tokenizer_padding_side must be left or right.' \
  'tokenizer padding-side validation'
require_text "$FAST" 'FastVLM config image_token_index must be less than vocab_size.' \
  'image token bounds validation'
require_text "$FAST" 'FastVLM vision projection_dim must match mm_hidden_size.' \
  'projector dimension validation'
require_text "$FAST" 'try Self.validateRGBTuple(imageMean, key: .imageMean, in: container)' \
  'processor image_mean RGB tuple validation'
require_text "$FAST" 'try Self.validateRGBTuple(imageStd, key: .imageStd, in: container)' \
  'processor image_std RGB tuple validation'
reject_text "$FAST" 'precondition(args.vocabularySize > 0)' \
  'text vocabulary constructor precondition'
reject_text "$FAST" 'precondition(dim % headDim == 0' \
  'vision MHSA constructor precondition'
reject_text "$FAST" 'precondition(mlpRatio > 0' \
  'vision MLP ratio constructor precondition'
reject_text "$FAST" 'fatalError("Token mixer type:' \
  'unsupported token mixer fatal'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: FastVLM config/init fatal boundaries are guarded at decode time.\n'
