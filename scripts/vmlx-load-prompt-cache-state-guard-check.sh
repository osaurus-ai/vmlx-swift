#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KVCACHE="$ROOT_DIR/Libraries/MLXLMCommon/KVCache.swift"

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  exit 1
}

[[ -f "$KVCACHE" ]] || fail "missing KVCache.swift"

grep -q 'try validatePromptCacheState' "$KVCACHE" \
  || fail 'loadPromptCache must validate persisted cache state before cache.state assignment'

grep -q 'if !cacheData\[i\].isEmpty {' "$KVCACHE" \
  || fail 'loadPromptCache must not assign empty state to non-empty-only cache setters'

grep -q 'private func validatePromptCacheState' "$KVCACHE" \
  || fail 'prompt cache state validator is missing'

grep -q 'prompt cache state must have 0 or 2 arrays' "$KVCACHE" \
  || fail 'KVCacheSimple persisted state count must be validated'

grep -q 'QuantizedKVCache prompt cache state must have 0, 4, or 6 arrays' "$KVCACHE" \
  || fail 'QuantizedKVCache persisted state count must be validated'

grep -q 'RotatingKVCache prompt cache metaState must have 5 values' "$KVCACHE" \
  || fail 'RotatingKVCache persisted metaState must be validated before setter'

grep -q 'CacheList prompt cache loading only supports empty state' "$KVCACHE" \
  || fail 'unsupported CacheList prompt cache payloads must throw instead of silently misloading'

printf 'PASS: loadPromptCache validates persisted state before nonthrowing cache setters.\n'
