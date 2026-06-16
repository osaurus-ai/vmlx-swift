#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_HELPERS="$ROOT_DIR/Libraries/MLXLMCommon/Cache/CacheHelpers.swift"

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  exit 1
}

[[ -f "$CACHE_HELPERS" ]] || fail "missing CacheHelpers.swift"

if ! grep -q 'private func isValidQKVStateArrayCount' "$CACHE_HELPERS"; then
  fail 'restoreQKVLayer must validate serialized QuantizedKVCache state array count before assignment'
fi

if grep -q 'qkv\.state = comp\.stateArrays' "$CACHE_HELPERS"; then
  fail 'restoreQKVLayer still assigns comp.stateArrays directly to QuantizedKVCache.state'
fi

if ! grep -q 'guard isValidQKVStateArrayCount(comp.stateArrays.count)' "$CACHE_HELPERS"; then
  fail 'restoreQKVLayer must cleanly reject malformed QKV disk state counts'
fi

printf 'PASS: QKV disk restore validates state array count before nonthrowing cache setters.\n'
