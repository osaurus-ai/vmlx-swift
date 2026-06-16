#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_HELPERS="$ROOT_DIR/Libraries/MLXLMCommon/Cache/CacheHelpers.swift"

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  exit 1
}

[[ -f "$CACHE_HELPERS" ]] || fail "missing CacheHelpers.swift"

grep -q 'guard restoreDeepseekV4Layer(comp, into: cache\[i\]) else { continue }' "$CACHE_HELPERS" \
  || fail 'DSV4 disk restore must report a miss when the live cache cannot accept restored hybrid state'

grep -q 'private func isCompatibleDeepseekV4State' "$CACHE_HELPERS" \
  || fail 'DSV4 disk restore must validate hybrid cache identity before assignment'

grep -q 'comp.compressRatio == hybrid.compressRatio' "$CACHE_HELPERS" \
  || fail 'DSV4 restore must validate compressRatio against the live cache'

grep -q 'comp.slidingWindow == hybrid.slidingWindow' "$CACHE_HELPERS" \
  || fail 'DSV4 restore must validate slidingWindow against the live cache'

grep -q 'comp.keys.shape.count >= 3 && comp.values.shape.count >= 3' "$CACHE_HELPERS" \
  || fail 'DSV4 restore must validate rotating KV rank before restoring'

grep -q 'comp.keys.shape == comp.values.shape' "$CACHE_HELPERS" \
  || fail 'DSV4 restore must validate matching rotating key/value shapes'

printf 'PASS: DSV4 hybrid disk restore validates live cache compatibility before reporting a hit.\n'
