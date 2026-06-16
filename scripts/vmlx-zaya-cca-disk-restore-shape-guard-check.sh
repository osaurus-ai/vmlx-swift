#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_HELPERS="$ROOT_DIR/Libraries/MLXLMCommon/Cache/CacheHelpers.swift"

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  exit 1
}

[[ -f "$CACHE_HELPERS" ]] || fail "missing CacheHelpers.swift"

grep -q 'guard restoreZayaCCALayer(comp, into: cache\[i\]) else { continue }' "$CACHE_HELPERS" \
  || fail 'ZAYA CCA disk restore must report a miss when the live cache cannot accept restored state'

grep -q 'private func isCompatibleZayaCCAState' "$CACHE_HELPERS" \
  || fail 'ZAYA CCA disk restore must validate path-dependent state shapes before assignment'

grep -q 'comp.convState.shape == \[zaya.batchSize, zaya.convChannels, 2\]' "$CACHE_HELPERS" \
  || fail 'ZAYA CCA restore must validate conv_state shape against the live cache'

grep -q 'comp.prevHS.shape == \[zaya.batchSize, zaya.hiddenSize\]' "$CACHE_HELPERS" \
  || fail 'ZAYA CCA restore must validate prev_hs shape against the live cache'

grep -q 'comp.keys.shape.count >= 3 && comp.values.shape.count >= 3' "$CACHE_HELPERS" \
  || fail 'ZAYA CCA restore must validate KV tensor rank before restoring'

printf 'PASS: ZAYA CCA disk restore validates live cache compatibility before reporting a hit.\n'
