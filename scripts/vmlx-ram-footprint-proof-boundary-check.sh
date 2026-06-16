#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_DIR="${VMLX_QWEN_PROOF_DIR:-/tmp/vmlx-qwen35-jangtq-turnmatrix-post-vlfix-20260524-1545}"
GEMMA_DIR="${VMLX_GEMMA_PROOF_DIR:-/tmp/vmlx-gemma4-turnmatrix-current-20260524-1516}"
RELEASE_LEDGER="$ROOT/.agents/vmlx-osaurus/codex/RELEASE-READINESS.md"
PR_CHECKLIST="$ROOT/.agents/vmlx-osaurus/codex/PR-PROOF-CHECKLIST.md"
RUNBENCH="$ROOT/RunBench/Bench.swift"

fail=0
pass() { echo "PASS $*"; }
fail_msg() { echo "FAIL $*" >&2; fail=1; }

require_dir() {
  local dir="$1" label="$2"
  if [[ -d "$dir" ]]; then
    pass "$label exists: $dir"
  else
    fail_msg "missing $label: $dir"
  fi
}

require_file() {
  local file="$1" label="$2"
  if [[ -f "$file" ]]; then
    pass "$label exists"
  else
    fail_msg "missing $label: $file"
  fi
}

require_text() {
  local path="$1" pattern="$2" label="$3"
  if rg -qi "$pattern" "$path"; then
    pass "$label"
  else
    fail_msg "missing $label in $path"
  fi
}

reject_text() {
  local path="$1" pattern="$2" label="$3"
  if rg -n -i "$pattern" "$path"; then
    fail_msg "forbidden $label in $path"
  else
    pass "no $label"
  fi
}

require_dir "$QWEN_DIR" "Qwen proof artifact"
require_dir "$GEMMA_DIR" "Gemma proof artifact"
require_file "$RELEASE_LEDGER" "release readiness ledger"
require_file "$PR_CHECKLIST" "PR proof checklist"
require_file "$RUNBENCH" "RunBench harness"

echo "--- artifact telemetry boundary ---"
require_text "$QWEN_DIR" 'tokps|tok/s|decodeTok/s|token/s' \
  "Qwen artifact has token-rate telemetry"
require_text "$GEMMA_DIR" 'tokps|tok/s|decodeTok/s|token/s' \
  "Gemma artifact has token-rate telemetry"
require_text "$QWEN_DIR" 'RSS|peakRSS|rssMiB' \
  "Qwen artifact has RSS telemetry"
require_text "$GEMMA_DIR" 'RSS|peakRSS|rssMiB' \
  "Gemma artifact has RSS telemetry"
require_text "$QWEN_DIR" 'footprintMiB|phys_footprint|peak_footprint' \
  "Qwen artifact has at least one physical-footprint telemetry row"

if rg -qi 'footprintMiB|phys_footprint|peak_footprint' "$GEMMA_DIR"; then
  pass "Gemma artifact has physical-footprint telemetry"
else
  echo "WARN Gemma artifact has RSS/token-rate proof but no physical-footprint telemetry; do not use it as Activity Monitor promotion proof" >&2
fi

echo "--- source capability for future phys-footprint proof ---"
require_text "$RUNBENCH" 'currentPhysFootprintMiB' \
  "RunBench can sample task_vm_info phys_footprint"
require_text "$RUNBENCH" 'phys_footprint' \
  "RunBench uses phys_footprint field"
require_text "$RUNBENCH" 'peak_footprint_mib|footprint_mib|footprintMiB' \
  "RunBench can emit footprint telemetry"
require_text "$RUNBENCH" 'PERF_MEMORY|PERF_RUN|BENCH_PERF' \
  "RunBench has PERF memory rows for future live proof"

echo "--- no-overclaim docs ---"
require_text "$PR_CHECKLIST" 'Activity Monitor physical footprint|physical footprint' \
  "PR checklist requires physical-footprint proof"
require_text "$PR_CHECKLIST" 'Qwen/Gemma artifacts include token/s and RSS evidence; physical-footprint promotion proof remains live/app gated' \
  "PR checklist distinguishes RSS artifacts from physical-footprint promotion proof"
require_text "$RELEASE_LEDGER" 'Qwen/Gemma text artifacts include token/s and RSS evidence, but physical-footprint promotion proof remains live/app gated' \
  "release ledger distinguishes RSS artifacts from physical-footprint promotion proof"
require_text "$RELEASE_LEDGER" 'Qwen35 RAM/OOM user crash path is not fully proven fixed end-to-end' \
  "release ledger refuses Qwen RAM/OOM overclaim"
reject_text "$RELEASE_LEDGER" 'Qwen/Gemma artifacts include token/s and RAM evidence\.|Qwen/Gemma artifacts include token/s and physical-footprint evidence|Qwen35 RAM/OOM user crash path is fully proven|Qwen35.*release-ready: yes' \
  "RAM/footprint overclaim wording"

if [[ "$fail" -ne 0 ]]; then
  echo "RAM/physical-footprint boundary guard failed." >&2
  exit 1
fi

echo "RAM/physical-footprint boundary guard passed."
