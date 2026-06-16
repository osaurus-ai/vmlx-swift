#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BATCH_ENGINE="$ROOT_DIR/Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift"

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  exit 1
}

[[ -f "$BATCH_ENGINE" ]] || fail "missing BatchEngine.swift"

if grep -q 'precondition(maxBatchSize > 0' "$BATCH_ENGINE"; then
  fail 'BatchEngine public init must not process-fatal on user/app maxBatchSize <= 0'
fi

grep -q 'let resolvedMaxBatchSize = max(1, maxBatchSize)' "$BATCH_ENGINE" \
  || fail 'BatchEngine init must normalize invalid construction maxBatchSize to a safe single-slot engine'

grep -q 'self.maxBatchSize = resolvedMaxBatchSize' "$BATCH_ENGINE" \
  || fail 'BatchEngine init must store the normalized maxBatchSize'

grep -q 'initialAdmissionCoalescingNanos = resolvedMaxBatchSize > 1' "$BATCH_ENGINE" \
  || fail 'initial admission coalescing must derive from the normalized maxBatchSize'

grep -q 'throw BatchEngineConfigurationError.invalidMaxBatchSize(newMaxBatchSize)' "$BATCH_ENGINE" \
  || fail 'runtime resize must keep the typed invalidMaxBatchSize error path'

printf 'PASS: BatchEngine construction cannot abort on invalid maxBatchSize and resize remains typed.\n'
