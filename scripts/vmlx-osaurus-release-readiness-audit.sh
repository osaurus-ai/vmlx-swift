#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OSAURUS_ROOT="${OSAURUS_ROOT:-/Users/eric/osaurus-staging}"
QWEN_DIR="${VMLX_QWEN_PROOF_DIR:-/tmp/vmlx-qwen35-jangtq-turnmatrix-post-vlfix-20260524-1545}"
GEMMA_DIR="${VMLX_GEMMA_PROOF_DIR:-/tmp/vmlx-gemma4-turnmatrix-post-thoughtfix-20260524}"
LOG_ROOT="${VMLX_RELEASE_AUDIT_LOG_ROOT:-/tmp/vmlx-osaurus-release-readiness-audit-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$LOG_ROOT"
: >"$LOG_ROOT/gates.tsv"

fail=0
pass() { echo "PASS $*"; }
fail_msg() { echo "FAIL $*" >&2; fail=1; }

run_gate() {
  local label="$1"
  shift
  local log="$LOG_ROOT/${label}.log"
  echo "--- $label ---"
  if "$@" >"$log" 2>&1; then
    pass "$label"
    printf '%s\tPASS\t%s\n' "$label" "$log" >>"$LOG_ROOT/gates.tsv"
  else
    fail_msg "$label"
    printf '%s\tFAIL\t%s\n' "$label" "$log" >>"$LOG_ROOT/gates.tsv"
  fi
  echo "log=$log"
  tail -40 "$log" || true
}

require_clean_process_baseline() {
  local log="$LOG_ROOT/process-baseline.log"
  {
    ps -axo pid,ppid,rss,etime,command || true
  } >"$log"

  local active
  active="$(rg -i 'CodeSigningHelper|xcodebuild|codesign( |$)|notarytool|/usr/bin/security( |$)|swift( |$)|swift-driver|swift-frontend|RunBench|vmlx-live-model-matrix|vmlx_engine\.cli|uvicorn|python.*vmlx|/Applications/osaurus\.app|/Users/eric/osaurus-staging.*(swift-test|xcrun swift|swift test|swift build|PackagePlugin|\.build/.*/Cmlx\.build|/usr/bin/clang .*osaurus-staging)' "$log" \
    | rg -v 'rg -i|vmlx-osaurus-release-readiness-audit' || true)"
  if [[ -n "$active" ]]; then
    fail_msg "clean process baseline"
    echo "$active" >&2
    printf '%s\n' "$active" >"$LOG_ROOT/process-blockers.txt"
  else
    pass "clean process baseline"
    : >"$LOG_ROOT/process-blockers.txt"
  fi
}

require_lock_clear() {
  if [[ -e /tmp/vmlx-runbench-live.lock ]]; then
    fail_msg "vMLX RunBench live lock clear"
    cat /tmp/vmlx-runbench-live.lock/pid 2>/dev/null || true
  else
    pass "vMLX RunBench live lock clear"
  fi
  if [[ -e /tmp/osaurus-live-proof.lock ]]; then
    fail_msg "Osaurus live proof lock clear"
    cat /tmp/osaurus-live-proof.lock/pid 2>/dev/null || true
  else
    pass "Osaurus live proof lock clear"
  fi
}

require_ledger() {
  local ledger="$ROOT/.agents/vmlx-osaurus/codex/RELEASE-READINESS.md"
  if [[ -f "$ledger" ]]; then
    pass "release readiness ledger exists"
  else
    fail_msg "release readiness ledger exists"
    return
  fi
  for text in \
    'Gemma4 parser' \
    'Reasoning on/off' \
    'No forced thinking tags' \
    'Hybrid SSM' \
    'DSV4 CSA/HSA/SWA' \
    'Qwen35 JANGTQ RAM overload' \
    'Big-model load cancellation' \
    'No keychain prompts' \
    'HF import compatibility' \
    'ChatEngine reasoning delta' \
    'Required final release gates before Eric can merge'; do
    if rg -q --fixed-strings "$text" "$ledger"; then
      pass "ledger tracks: $text"
    else
      fail_msg "ledger tracks: $text"
    fi
  done
}

require_current_state_consistency() {
  local release_ledger="$ROOT/.agents/vmlx-osaurus/codex/RELEASE-READINESS.md"
  local pr_ledger="$ROOT/.agents/vmlx-osaurus/codex/PR-READINESS.md"
  local vmlx_head osaurus_head osaurus_pin

  vmlx_head="$(git -C "$ROOT" rev-parse HEAD)"
  osaurus_head="$(git -C "$OSAURUS_ROOT" rev-parse HEAD)"
  osaurus_pin="$(rg -o 'revision: "[0-9a-f]{40}"' "$OSAURUS_ROOT/Packages/OsaurusCore/Package.swift" | head -1 | sed -E 's/.*"([0-9a-f]{40})"/\1/')"

  if rg -q --fixed-strings "HEAD \`$vmlx_head\`" "$release_ledger"; then
    pass "release ledger current vMLX HEAD"
  else
    fail_msg "release ledger current vMLX HEAD"
  fi

  if rg -q --fixed-strings "HEAD \`$osaurus_head\`" "$release_ledger"; then
    pass "release ledger current Osaurus HEAD"
  else
    fail_msg "release ledger current Osaurus HEAD"
  fi

  if [[ "$osaurus_pin" == "$vmlx_head" ]]; then
    pass "Osaurus Package.swift pin matches current vMLX HEAD"
  else
    fail_msg "Osaurus Package.swift pin matches current vMLX HEAD"
  fi

  if [[ -f "$pr_ledger" ]] && rg -q --fixed-strings "Current pushed Osaurus branch head is \`${osaurus_head:0:8}\`" "$pr_ledger"; then
    pass "PR readiness ledger current Osaurus branch head"
  else
    fail_msg "PR readiness ledger current Osaurus branch head"
  fi

  if [[ -f "$pr_ledger" ]] && rg -q --fixed-strings "Current pushed vMLX revision for Osaurus is \`$vmlx_head\`" "$pr_ledger"; then
    pass "PR readiness ledger current vMLX pin"
  else
    fail_msg "PR readiness ledger current vMLX pin"
  fi
}

require_no_partial_family_promotion() {
  local ledger="$ROOT/.agents/vmlx-osaurus/codex/RELEASE-READINESS.md"
  if rg -q 'DSV4.*failed/partial|DSV4.*partial/failed' "$ledger" && rg -q 'ZAYA.*partial/failed|ZAYA.*failed/partial' "$ledger"; then
    pass "DSV4/ZAYA remain explicitly unpromoted"
  else
    fail_msg "DSV4/ZAYA remain explicitly unpromoted"
  fi
}

require_no_live_overclaim() {
  local ledger="$ROOT/.agents/vmlx-osaurus/codex/RELEASE-READINESS.md"
  if rg -q 'Qwen35 JANGTQ RAM overload.*not end-to-end proven fixed|Qwen35 RAM/OOM user crash path is not fully proven fixed end-to-end' "$ledger"; then
    pass "Qwen35 RAM/OOM remains explicitly unpromoted"
  else
    fail_msg "Qwen35 RAM/OOM remains explicitly unpromoted"
  fi

  if rg -q 'Big-model load cancellation.*live proof blocked|Big-model load cancellation.*no live rebuilt-app proof|Big-model load cancellation.*blocked' "$ledger"; then
    pass "big-model load cancellation remains live-proof gated"
  else
    fail_msg "big-model load cancellation remains live-proof gated"
  fi

  if rg -q 'No SwiftPM/Xcode/app launch/signing/model-load lane was run|Osaurus live app/build/test/signing/model-load proof: blocked' "$ledger"; then
    pass "ledger states no live app/model proof promotion"
  else
    fail_msg "ledger states no live app/model proof promotion"
  fi
}

require_clean_process_baseline
require_lock_clear
require_ledger
require_current_state_consistency
require_no_partial_family_promotion
require_no_live_overclaim

if [[ -x "$ROOT/scripts/vmlx-push-readiness-scope-check.sh" ]]; then
  run_gate vmlx-push-readiness "$ROOT/scripts/vmlx-push-readiness-scope-check.sh"
else
  fail_msg "vmlx-push-readiness gate exists"
fi

if [[ -x "$ROOT/scripts/vmlx-release-dirty-scope-classifier.sh" ]]; then
  run_gate vmlx-dirty-scope "$ROOT/scripts/vmlx-release-dirty-scope-classifier.sh" "$LOG_ROOT/vmlx-dirty-scope.md"
else
  fail_msg "vMLX dirty-scope classifier exists"
fi

if [[ -x "$ROOT/scripts/vmlx-architecture-cache-proof-check.sh" ]]; then
  run_gate vmlx-architecture-cache "$ROOT/scripts/vmlx-architecture-cache-proof-check.sh"
else
  fail_msg "vmlx architecture cache gate exists"
fi

if [[ -x "$ROOT/scripts/vmlx-qwen-gemma-proof-check.sh" ]]; then
  run_gate vmlx-qwen-gemma "$ROOT/scripts/vmlx-qwen-gemma-proof-check.sh" \
    --qwen-dir "$QWEN_DIR" \
    --gemma-dir "$GEMMA_DIR" \
    --osaurus-root "$OSAURUS_ROOT"
else
  fail_msg "vmlx Qwen/Gemma proof gate exists"
fi

if [[ -x "$ROOT/scripts/vmlx-family-matrix-coverage-check.sh" ]]; then
  run_gate vmlx-family-matrix-coverage "$ROOT/scripts/vmlx-family-matrix-coverage-check.sh"
else
  fail_msg "vmlx family matrix coverage gate exists"
fi

if [[ -x "$OSAURUS_ROOT/scripts/live-proof/assert-keychain-free-proof-path.sh" ]]; then
  run_gate osaurus-keychain-free "$OSAURUS_ROOT/scripts/live-proof/assert-keychain-free-proof-path.sh"
else
  fail_msg "Osaurus keychain-free proof gate exists"
fi

if [[ -x "$OSAURUS_ROOT/scripts/live-proof/assert-keychain-disabled-source-coverage.sh" ]]; then
  run_gate osaurus-keychain-disabled-source "$OSAURUS_ROOT/scripts/live-proof/assert-keychain-disabled-source-coverage.sh"
else
  fail_msg "Osaurus keychain-disabled source guard exists"
fi

if [[ -x "$OSAURUS_ROOT/scripts/live-proof/assert-osaurus-pr-hygiene.sh" ]]; then
  run_gate osaurus-pr-hygiene "$OSAURUS_ROOT/scripts/live-proof/assert-osaurus-pr-hygiene.sh"
else
  fail_msg "Osaurus PR hygiene guard exists"
fi

if [[ -x "$OSAURUS_ROOT/scripts/live-proof/assert-osaurus-vmlx-pr-readiness.sh" ]]; then
  run_gate osaurus-vmlx-pr-readiness "$OSAURUS_ROOT/scripts/live-proof/assert-osaurus-vmlx-pr-readiness.sh"
else
  fail_msg "Osaurus vMLX PR readiness guard exists"
fi

if [[ -x "$OSAURUS_ROOT/scripts/live-proof/assert-vmlx-gemma4-parser-fix-wired.sh" ]]; then
  run_gate osaurus-gemma-parser-wire "$OSAURUS_ROOT/scripts/live-proof/assert-vmlx-gemma4-parser-fix-wired.sh"
else
  fail_msg "Osaurus Gemma parser wire guard exists"
fi

if [[ -x "$OSAURUS_ROOT/scripts/live-proof/assert-osaurus-no-forced-behavior-pr.sh" ]]; then
  run_gate osaurus-no-forced-behavior "$OSAURUS_ROOT/scripts/live-proof/assert-osaurus-no-forced-behavior-pr.sh"
else
  fail_msg "Osaurus no-forced-behavior guard exists"
fi

if [[ -x "$OSAURUS_ROOT/scripts/live-proof/assert-no-hidden-local-sampler-defaults.sh" ]]; then
  run_gate osaurus-no-hidden-sampler-defaults "$OSAURUS_ROOT/scripts/live-proof/assert-no-hidden-local-sampler-defaults.sh"
else
  fail_msg "Osaurus no-hidden-local-sampler-defaults guard exists"
fi

if [[ -x "$OSAURUS_ROOT/scripts/live-proof/assert-openresponses-cache-proof-wiring.sh" ]]; then
  run_gate osaurus-openresponses-cache "$OSAURUS_ROOT/scripts/live-proof/assert-openresponses-cache-proof-wiring.sh"
else
  fail_msg "Osaurus OpenResponses/cache guard exists"
fi

if [[ -x "$OSAURUS_ROOT/scripts/live-proof/assert-server-settings-runtime-wiring.sh" ]]; then
  run_gate osaurus-server-settings "$OSAURUS_ROOT/scripts/live-proof/assert-server-settings-runtime-wiring.sh"
else
  fail_msg "Osaurus server-settings runtime guard exists"
fi

if [[ -x "$OSAURUS_ROOT/scripts/live-proof/assert-chat-reasoning-delta-routing.sh" ]]; then
  run_gate osaurus-chat-reasoning "$OSAURUS_ROOT/scripts/live-proof/assert-chat-reasoning-delta-routing.sh"
else
  fail_msg "Osaurus chat reasoning delta guard exists"
fi

if [[ -x "$OSAURUS_ROOT/scripts/live-proof/assert-chat-ui-reasoning-routing.sh" ]]; then
  run_gate osaurus-chat-ui-reasoning "$OSAURUS_ROOT/scripts/live-proof/assert-chat-ui-reasoning-routing.sh"
else
  fail_msg "Osaurus chat UI reasoning guard exists"
fi

if [[ -x "$OSAURUS_ROOT/scripts/live-proof/assert-http-channel-load-cancellation.sh" ]]; then
  run_gate osaurus-http-cancellation "$OSAURUS_ROOT/scripts/live-proof/assert-http-channel-load-cancellation.sh"
else
  fail_msg "Osaurus HTTP cancellation guard exists"
fi

if [[ -x "$OSAURUS_ROOT/scripts/live-proof/assert-tool-choice-required-routing.sh" ]]; then
  run_gate osaurus-tool-choice "$OSAURUS_ROOT/scripts/live-proof/assert-tool-choice-required-routing.sh"
else
  fail_msg "Osaurus tool-choice guard exists"
fi

if [[ -x "$OSAURUS_ROOT/scripts/live-proof/assert-hf-import-compatibility.sh" ]]; then
  run_gate osaurus-hf-import "$OSAURUS_ROOT/scripts/live-proof/assert-hf-import-compatibility.sh"
else
  fail_msg "Osaurus HF import compatibility guard exists"
fi

if [[ -x "$OSAURUS_ROOT/scripts/live-proof/classify-osaurus-pr-dirty-scope.sh" ]]; then
  run_gate osaurus-dirty-scope "$OSAURUS_ROOT/scripts/live-proof/classify-osaurus-pr-dirty-scope.sh" "$LOG_ROOT/osaurus-pr-dirty-scope.md"
else
  fail_msg "Osaurus PR dirty-scope classifier exists"
fi

VMLX_BRANCH="$(git -C "$ROOT" branch --show-current 2>/dev/null || true)"
VMLX_HEAD="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
OSAURUS_BRANCH="$(git -C "$OSAURUS_ROOT" branch --show-current 2>/dev/null || true)"
OSAURUS_HEAD="$(git -C "$OSAURUS_ROOT" rev-parse HEAD 2>/dev/null || true)"
VMLX_DIRTY_COUNT="$({ git -C "$ROOT" status --short 2>/dev/null || true; } | wc -l | tr -d ' ')"
OSAURUS_DIRTY_COUNT="$({ git -C "$OSAURUS_ROOT" status --short 2>/dev/null || true; } | wc -l | tr -d ' ')"
VMLX_DIRTY_SAMPLE="$LOG_ROOT/vmlx-dirty-sample.txt"
OSAURUS_DIRTY_SAMPLE="$LOG_ROOT/osaurus-dirty-sample.txt"
git -C "$ROOT" status --short 2>/dev/null | sed -n '1,80p' >"$VMLX_DIRTY_SAMPLE" || true
git -C "$OSAURUS_ROOT" status --short 2>/dev/null | sed -n '1,80p' >"$OSAURUS_DIRTY_SAMPLE" || true

cat > "$LOG_ROOT/SUMMARY.md" <<SUMMARY
# vMLX / Osaurus release readiness audit

Result: $([[ "$fail" -eq 0 ]] && echo PASS || echo FAIL)

Inputs:

- vMLX root: $ROOT
- Osaurus root: $OSAURUS_ROOT
- Qwen artifact dir: $QWEN_DIR
- Gemma artifact dir: $GEMMA_DIR

Repos:

- vMLX branch/head: ${VMLX_BRANCH:-unknown} ${VMLX_HEAD:-unknown}
- Osaurus branch/head: ${OSAURUS_BRANCH:-unknown} ${OSAURUS_HEAD:-unknown}
- vMLX dirty entries: $VMLX_DIRTY_COUNT (sample: $VMLX_DIRTY_SAMPLE)
- Osaurus dirty entries: $OSAURUS_DIRTY_COUNT (sample: $OSAURUS_DIRTY_SAMPLE)

Logs: $LOG_ROOT

Gate table:

\`\`\`tsv
$(cat "$LOG_ROOT/gates.tsv")
\`\`\`

Process blockers:

\`\`\`
$(cat "$LOG_ROOT/process-blockers.txt")
\`\`\`

A PASS means shell/source/artifact gates are clean. It still does not replace any explicitly approved live rebuilt-app/model proof that Eric requires.
SUMMARY

echo "summary=$LOG_ROOT/SUMMARY.md"
if [[ "$fail" -ne 0 ]]; then
  echo "release readiness audit failed; do not merge or stop work" >&2
  exit 1
fi

echo "release readiness audit passed"
