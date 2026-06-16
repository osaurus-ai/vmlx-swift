#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL="$ROOT/Libraries/MLXLMCommon/Evaluate.swift"

require_text() {
  local file="$1"
  local text="$2"
  local label="$3"
  if ! grep -Fq "$text" "$file"; then
    printf 'FAIL: %s missing %s\n' "$file" "$label" >&2
    exit 1
  fi
}

reject_text() {
  local file="$1"
  local text="$2"
  local label="$3"
  if grep -Fq "$text" "$file"; then
    printf 'FAIL: %s still contains forbidden %s\n' "$file" "$label" >&2
    exit 1
  fi
}

reject_text "$EVAL" 'precondition(capacity > 0)' \
  'TokenRing capacity precondition'
require_text "$EVAL" 'self.capacity = max(0, capacity)' \
  'nonnegative TokenRing capacity normalization'
require_text "$EVAL" 'guard capacity > 0 else { return nil }' \
  'zero-capacity validTokens no-op'
require_text "$EVAL" 'guard capacity > 0 else { return }' \
  'zero-capacity load/append no-op'

printf 'PASS: penalty TokenRing zero-capacity contexts no-op instead of process-aborting.\n'
