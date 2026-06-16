#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG="$ROOT/Package.swift"
PROBE="$ROOT/tools/LoadRaceProbe/main.swift"
failures=0

require_text() {
  local file="$1"
  local pattern="$2"
  local label="$3"
  if ! grep -Fq "$pattern" "$file"; then
    printf 'FAIL: %s missing %s\n  pattern: %s\n' "$file" "$label" "$pattern" >&2
    failures=$((failures + 1))
  fi
}

require_text "$PKG" '.executable(name: "LoadRaceProbe", targets: ["LoadRaceProbe"])' \
  'LoadRaceProbe product'
require_text "$PKG" 'name: "LoadRaceProbe"' \
  'LoadRaceProbe executable target'
require_text "$PROBE" 'withThrowingTaskGroup' \
  'single-process concurrent load tasks'
require_text "$PROBE" 'MLXLMCommon.loadModel(' \
  'real vMLX loadModel path'
require_text "$PROBE" 'MLX.eval(context.model)' \
  'post-load materialization proof point'
require_text "$PROBE" 'Sentry APPLE-MACOS-25/31/5M' \
  'Sentry reproduction rationale'
require_text "$PROBE" 'LOAD_JOB_BEGIN' \
  'per-job timing output'
require_text "$PROBE" 'LOAD_RACE_OK' \
  'success output'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: LoadRaceProbe exercises duplicate loadModel tasks in one process.\n'
