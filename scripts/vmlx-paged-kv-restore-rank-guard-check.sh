#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_HELPERS="$ROOT_DIR/Libraries/MLXLMCommon/Cache/CacheHelpers.swift"

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  exit 1
}

[[ -f "$CACHE_HELPERS" ]] || fail "missing CacheHelpers.swift"

grep -q 'private func isCompatibleDecodedKVState' "$CACHE_HELPERS" \
  || fail 'paged/legacy KV restore must share a decoded KV rank/shape validator'

grep -q 'keySlices.allSatisfy({ $0.shape.count >= 3 })' "$CACHE_HELPERS" \
  || fail 'paged restore must validate key slice rank before axis-2 concat'

grep -q 'valueSlices.allSatisfy({ $0.shape.count >= 3 })' "$CACHE_HELPERS" \
  || fail 'paged restore must validate value slice rank before axis-2 concat'

grep -q 'guard isCompatibleDecodedKVState(keys: restoredKeys, values: restoredValues)' "$CACHE_HELPERS" \
  || fail 'paged/legacy restore must validate decoded KV shape before cache assignment'

printf 'PASS: paged and legacy decoded KV restore validate rank/shape before cache mutation.\n'
