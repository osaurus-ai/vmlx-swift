#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOAD="$ROOT/Libraries/MLXLMCommon/Load.swift"
failures=0

require_text() {
  local pattern="$1"
  local label="$2"
  if ! grep -Fq "$pattern" "$LOAD"; then
    printf 'FAIL: missing %s\n  pattern: %s\n  file: %s\n' "$label" "$pattern" "$LOAD" >&2
    failures=$((failures + 1))
  fi
}

require_text 'private enum MLXLoadMaterializationLock' \
  'process-wide load materialization lock'
require_text 'static let shared = NSRecursiveLock()' \
  'recursive load materialization lock'
require_text 'private func withMLXLoadMaterializationLock' \
  'load materialization lock helper'
require_text 'Sentry APPLE-MACOS-25/31/5M' \
  'Sentry BF16 load crash rationale'
require_text 'withMLXLoadMaterializationLock {' \
  'BF16 conversion/final eval lock scope'

if ! awk '
  /withMLXLoadMaterializationLock \{/ { in_scope=1; depth=0 }
  in_scope {
    depth += gsub(/\{/, "{")
    depth -= gsub(/\}/, "}")
    if (index($0, "convertToBFloat16(model: model)") > 0) saw_convert=1
    if (index($0, "eval(model)") > 0) saw_eval=1
    if (index($0, "MLX.Memory.clearCache()") > 0) saw_clear=1
    if (depth == 0) in_scope=0
  }
  END { exit(saw_convert && saw_eval && saw_clear ? 0 : 1) }
' "$LOAD"; then
  printf 'FAIL: load materialization lock must cover BF16 conversion, final eval, and cache clear\n' >&2
  failures=$((failures + 1))
fi

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: load-time BF16 materialization is serialized across concurrent loads.\n'
