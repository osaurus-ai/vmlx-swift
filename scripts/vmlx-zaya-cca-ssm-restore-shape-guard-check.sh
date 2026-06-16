#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_HELPERS="$ROOT_DIR/Libraries/MLXLMCommon/Cache/CacheHelpers.swift"

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  exit 1
}

[[ -f "$CACHE_HELPERS" ]] || fail "missing CacheHelpers.swift"

grep -q 'private func restoreZayaCCACompanionState' "$CACHE_HELPERS" \
  || fail 'ZAYA CCA SSM companion restore must validate shapes before writeCCA'

[[ "$(grep -c 'guard restoreZayaCCACompanionState(conv: conv, prev: prev, into: zaya) else {' "$CACHE_HELPERS")" -ge 2 ]] \
  || fail 'top-level ZAYA companion restore must miss/stop before unsafe writeCCA on bad shapes'

[[ "$(grep -c 'zaya.writeCCA(conv: conv, prev: prev)' "$CACHE_HELPERS")" -eq 1 ]] \
  || fail 'ZAYA companion writeCCA should only be called from the validated helper'

grep -q 'conv.shape == \[zaya.batchSize, zaya.convChannels, 2\]' "$CACHE_HELPERS" \
  || fail 'ZAYA companion restore must validate conv_state shape'

grep -q 'prev.shape == \[zaya.batchSize, zaya.hiddenSize\]' "$CACHE_HELPERS" \
  || fail 'ZAYA companion restore must validate prev_hs shape'

printf 'PASS: ZAYA CCA SSM companion restore validates shapes before writeCCA.\n'
