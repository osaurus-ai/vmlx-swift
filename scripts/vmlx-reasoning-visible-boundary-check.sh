#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fail=0

pass() { echo "PASS $*"; }
fail_msg() { echo "FAIL $*" >&2; fail=1; }

require_file() {
  local file="$1" label="$2"
  if [[ -f "$file" ]]; then
    pass "$label exists"
  else
    fail_msg "missing $label: $file"
  fi
}

require_text() {
  local file="$1" pattern="$2" label="$3"
  if rg -q "$pattern" "$file"; then
    pass "$label"
  else
    fail_msg "missing $label in ${file#$ROOT/}"
  fi
}

reject_text() {
  local file="$1" pattern="$2" label="$3"
  if rg -n "$pattern" "$file"; then
    fail_msg "forbidden $label in ${file#$ROOT/}"
  else
    pass "no $label"
  fi
}

BENCH="$ROOT/RunBench/Bench.swift"
OMNI="$ROOT/RunBench/OmniBench.swift"
FOCUSED="$ROOT/Tests/MLXLMCommonFocusedTests/NoHiddenReasoningCloseBiasFocusedTests.swift"
PROOF="$ROOT/scripts/vmlx-qwen-gemma-proof-check.sh"
PR_CHECKLIST="$ROOT/.agents/vmlx-osaurus/codex/PR-PROOF-CHECKLIST.md"
RELEASE_LEDGER="$ROOT/.agents/vmlx-osaurus/codex/RELEASE-READINESS.md"

for pair in \
  "$BENCH:RunBench" \
  "$OMNI:OmniBench" \
  "$FOCUSED:NoHiddenReasoningCloseBiasFocusedTests" \
  "$PROOF:Qwen/Gemma proof checker" \
  "$PR_CHECKLIST:PR proof checklist" \
  "$RELEASE_LEDGER:release readiness ledger"; do
  require_file "${pair%%:*}" "${pair#*:}"
done

echo "--- source boundary ---"
require_text "$BENCH" '\[empty visible; reasoning chars=' \
  "RunBench reports empty visible previews explicitly"
require_text "$OMNI" 'let visible = text\.trimmingCharacters' \
  "OmniBench computes visible text from chunk output"
require_text "$OMNI" 'visible,' \
  "OmniBench returns visible text separately from reasoning"
require_text "$FOCUSED" 'visible\.isEmpty \? reasoning : visible' \
  "focused test rejects visible/reasoning substitution"
require_text "$FOCUSED" 'visible\.isEmpty \? reasoning\.trimmingCharacters' \
  "focused test rejects OmniBench reasoning substitution"
require_text "$PROOF" 'OmniBench reasoning-only output counted as visible answer' \
  "proof checker guards OmniBench reasoning substitution"

reject_text "$BENCH" \
  'text\.isEmpty \? reasoning : text|visible\.isEmpty \? reasoning : visible|visible\.isEmpty \? r\.reasoning : visible|r\.text\.isEmpty \? r\.reasoning : r\.text|reasoning\.isEmpty \? r\.text : r\.reasoning|let combined = text \+ reasoning' \
  "RunBench reasoning-only output counted/displayed as visible"
reject_text "$OMNI" \
  'visible\.isEmpty \? reasoning\.trimmingCharacters|text\.isEmpty \? reasoning : text' \
  "OmniBench reasoning-only output counted/displayed as visible"

echo "--- documentation boundary ---"
require_text "$PR_CHECKLIST" 'Reasoning-only rows cannot masquerade as visible answers' \
  "PR checklist tracks reasoning-only preview boundary"
require_text "$RELEASE_LEDGER" 'Reasoning-only output must not be counted or displayed as visible answer proof' \
  "release ledger tracks reasoning-only preview boundary"
require_text "$RELEASE_LEDGER" 'No reasoning-as-visible proof previews' \
  "release proof map tracks no reasoning-as-visible previews"

if [[ "$fail" -ne 0 ]]; then
  echo "Reasoning/visible boundary guard failed." >&2
  exit 1
fi

echo "Reasoning/visible boundary guard passed."
