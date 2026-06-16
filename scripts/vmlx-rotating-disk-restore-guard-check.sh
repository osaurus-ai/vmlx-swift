#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_HELPERS="$ROOT_DIR/Libraries/MLXLMCommon/Cache/CacheHelpers.swift"

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  exit 1
}

[[ -f "$CACHE_HELPERS" ]] || fail "missing CacheHelpers.swift"

grep -q 'guard restoreRotatingLayer(comp, into: cache\[i\]) else { continue }' "$CACHE_HELPERS" \
  || fail 'Rotating disk restore must report a miss when no compatible rotating cache exists'

grep -q 'private func isCompatibleRotatingState' "$CACHE_HELPERS" \
  || fail 'Rotating disk restore must validate serialized rotating state before assignment'

grep -q 'comp.keys.shape.count >= 3 && comp.values.shape.count >= 3' "$CACHE_HELPERS" \
  || fail 'Rotating disk restore must validate KV tensor rank before restoring'

grep -q 'comp.keys.shape == comp.values.shape' "$CACHE_HELPERS" \
  || fail 'Rotating disk restore must validate matching key/value shapes'

grep -q 'comp.maxSize > 0' "$CACHE_HELPERS" \
  || fail 'Rotating disk restore must validate positive max cache size'

grep -q 'comp.idx >= 0 && comp.idx < comp.maxSize' "$CACHE_HELPERS" \
  || fail 'Rotating disk restore must validate ring index bounds'

printf 'PASS: rotating disk restore validates cache compatibility before reporting a hit.\n'
