#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OSAURUS_ROOT="${OSAURUS_ROOT:-/Users/eric/osaurus-staging}"
QWEN_DIR="${VMLX_QWEN_PROOF_DIR:-/tmp/vmlx-qwen35-jangtq-turnmatrix-post-vlfix-20260524-1545}"
GEMMA_DIR="${VMLX_GEMMA_PROOF_DIR:-/tmp/vmlx-gemma4-turnmatrix-current-20260524-1516}"
ZAYA_DIR="${VMLX_ZAYA_PROOF_DIR:-/tmp/vmlx-zaya-text-turnmatrix-20260524-1456}"
DSV4_DIR="${VMLX_DSV4_PROOF_DIR:-/tmp/vmlx-dsv4-band-pe2-turnmatrix-20260524-1445}"
DSV4_JANGTQ2_DIR="${VMLX_DSV4_JANGTQ2_PROOF_DIR:-/tmp/vmlx-dsv4-jangtq2-final-20260524}"
DSV4_AGENTIC_DIR="${VMLX_DSV4_AGENTIC_PROOF_DIR:-/tmp/vmlx-dsv4-jangtq2-agentic-tool-current-20260525}"
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

require_dir() {
  local dir="$1" label="$2"
  if [[ -d "$dir" ]]; then
    pass "$label exists: $dir"
  else
    fail_msg "missing $label: $dir"
  fi
}

require_text() {
  local path="$1" pattern="$2" label="$3"
  if rg -q "$pattern" "$path"; then
    pass "$label"
  else
    fail_msg "missing $label in $path"
  fi
}

reject_text() {
  local path="$1" pattern="$2" label="$3"
  if rg -n "$pattern" "$path"; then
    fail_msg "forbidden $label in $path"
  else
    pass "no $label"
  fi
}

MATRIX="$ROOT/scripts/vmlx-live-model-matrix.sh"
RELEASE_LEDGER="$ROOT/.agents/vmlx-osaurus/codex/RELEASE-READINESS.md"
PR_CHECKLIST="$ROOT/.agents/vmlx-osaurus/codex/PR-PROOF-CHECKLIST.md"
RUNBENCH="$ROOT/RunBench/Bench.swift"
NO_HIDDEN_TESTS="$ROOT/Tests/MLXLMCommonFocusedTests/NoHiddenReasoningCloseBiasFocusedTests.swift"
ARCH_GUARD="$ROOT/scripts/vmlx-architecture-cache-proof-check.sh"
QG_GUARD="$ROOT/scripts/vmlx-qwen-gemma-proof-check.sh"
OSAURUS_UI_GUARD="$OSAURUS_ROOT/scripts/live-proof/assert-chat-ui-reasoning-routing.sh"

require_file "$MATRIX" "live model matrix"
require_file "$RUNBENCH" "RunBench live harness"
require_file "$NO_HIDDEN_TESTS" "focused no-hidden-reasoning tests"
require_file "$RELEASE_LEDGER" "release readiness ledger"
require_file "$PR_CHECKLIST" "PR proof checklist"
require_file "$ARCH_GUARD" "architecture cache guard"
require_file "$QG_GUARD" "Qwen/Gemma proof guard"
require_file "$OSAURUS_UI_GUARD" "Osaurus Chat UI reasoning guard"

echo "--- matrix family lanes ---"
require_text "$MATRIX" 'qwen_multiturn_tool' \
  "matrix includes Qwen multi-turn tool row"
require_text "$MATRIX" 'BENCH_GROWING_CHAT_CACHE=1' \
  "matrix includes growing chat/prefix cache row"
require_text "$MATRIX" 'BENCH_BATCH_DISK_RESTORE=1' \
  "matrix includes disk L2 restore row"
require_text "$MATRIX" 'BENCH_BATCH_TQ_B2=1' \
  "matrix includes TurboQuant KV B=2 row"
require_text "$MATRIX" 'n-a:deepseek-v4-uses-swa-csa-hsa-hybrid-pool-cache-not-turboquant-kv' \
  "matrix excludes generic TurboQuant substitution for DSV4"
require_text "$MATRIX" 'vl_mixed_text_image_video' \
  "matrix includes VL/video media payload row"
require_text "$MATRIX" 'sampler_defaults' \
  "matrix includes sampler defaults row"
require_text "$MATRIX" 'fail:missing-bundle-sampler-defaults-would-use-engine-fallback' \
  "matrix fails missing bundle sampler defaults"
require_text "$MATRIX" 'Refusing to start matrix while another RunBench live row is active' \
  "matrix refuses contaminated concurrent RunBench lanes"
require_text "$MATRIX" 'Matrix live proof must not reuse a stale RunBench binary' \
  "matrix refuses stale RunBench binaries"
require_text "$RUNBENCH" 'requiresToolCall' \
  "Qwen multi-turn tool row marks tool-required turns"
require_text "$RUNBENCH" '\$0\.requiresToolCall && \$0\.toolCalls == 0' \
  "Qwen multi-turn tool row fails missing structured tool events"
require_text "$RUNBENCH" 'tool-required turn .* produced no structured \.toolCall event' \
  "Qwen multi-turn tool row reports missing structured tool event"
require_text "$NO_HIDDEN_TESTS" '\$0\.requiresToolCall && \$0\.toolCalls == 0' \
  "focused tests guard Qwen tool-required event failure"

echo "--- source guard coverage ---"
require_text "$ARCH_GUARD" 'DSV4.*CSA/HSA/SWA|CSA HSA pools|dsv4_0_pool_comp' \
  "architecture guard covers DSV4 hybrid pools"
require_text "$ARCH_GUARD" 'ZAYA.*CCA|zayaCCA' \
  "architecture guard covers ZAYA CCA"
require_text "$ARCH_GUARD" 'hybrid SSM|SSM companion' \
  "architecture guard covers hybrid SSM companion"
require_text "$ARCH_GUARD" 'Gemma4.*SWA|rotating/SWA' \
  "architecture guard covers Gemma SWA/rotating"
require_text "$ARCH_GUARD" 'media salt|mediaSalt' \
  "architecture guard covers media cache salt"
require_text "$QG_GUARD" 'check_qwen_hybrid_specific_artifacts' \
  "Qwen/Gemma guard has Qwen hybrid-specific artifact gate"
require_text "$QG_GUARD" 'check_gemma_cache_specific_artifacts' \
  "Qwen/Gemma guard has Gemma cache-specific artifact gate"
require_text "$OSAURUS_UI_GUARD" 'ChatView routes reasoning to processor reasoning path' \
  "Osaurus UI guard covers Think panel reasoning route"

echo "--- artifact and no-overclaim coverage ---"
require_dir "$QWEN_DIR" "Qwen artifact dir"
require_dir "$GEMMA_DIR" "Gemma artifact dir"
require_dir "$ZAYA_DIR" "ZAYA artifact dir"
require_dir "$DSV4_DIR" "DSV4 artifact dir"
require_dir "$DSV4_JANGTQ2_DIR" "DSV4 JANGTQ2 coherence artifact dir"
require_dir "$DSV4_AGENTIC_DIR" "DSV4 JANGTQ2 agentic DSML artifact dir"
require_text "$QWEN_DIR" 'qwen_multiturn_tool|tool' \
  "Qwen artifact includes multi-turn/tool evidence"
require_text "$QWEN_DIR" 'ssm\{hits=[1-9]|companion=ssm|ssm_companion' \
  "Qwen artifact includes SSM companion evidence"
require_text "$QWEN_DIR" 'tqCompressionsA=[1-9]|tqCompressionsB=[1-9]|BatchEngine TurboQuant B=2' \
  "Qwen artifact includes TurboQuant KV evidence"
require_text "$GEMMA_DIR" 'Gemma|gemma' \
  "Gemma artifact is identifiable"
require_text "$GEMMA_DIR" 'rotatingLayers=[1-9]|SWA|sliding' \
  "Gemma artifact includes SWA/rotating evidence"
require_text "$GEMMA_DIR" 'no raw markers leaked to \.chunk|reasoning= *0.*tools=0|reasoning ON/OFF rows must not leak' \
  "Gemma artifact includes no-leak parser evidence"
require_text "$ZAYA_DIR" 'fail|failed|partial|blocked|missing bundle sampler defaults|reasoning-only|TurboQuant B=2 did not compress' \
  "ZAYA artifact remains partial/failed rather than promoted"
require_text "$DSV4_DIR" 'fail|failed|partial|blocked|café|UTF-8|coherence' \
  "DSV4 artifact remains partial/failed rather than promoted"
require_text "$DSV4_JANGTQ2_DIR/template_kwargs.out" 'BENCH_DSV4_TEMPLATE_KWARGS: passed' \
  "DSV4 JANGTQ2 template kwargs proof passed"
require_text "$DSV4_JANGTQ2_DIR/coherence_chat.out" 'BENCH_DSV4_COHERENCE: PASS' \
  "DSV4 JANGTQ2 chat coherence row passed"
require_text "$DSV4_JANGTQ2_DIR/coherence_chat.out" 'Sapphire-42|SAPPHIRE-42' \
  "DSV4 JANGTQ2 chat recall evidence"
require_text "$DSV4_JANGTQ2_DIR/coherence_reasoning.out" 'DSV4_REASONING_ON_REASONING_BEGIN' \
  "DSV4 JANGTQ2 reasoning channel evidence"
require_text "$DSV4_JANGTQ2_DIR/coherence_reasoning.out" 'BENCH_DSV4_COHERENCE: PASS' \
  "DSV4 JANGTQ2 reasoning coherence row passed"
require_text "$DSV4_JANGTQ2_DIR/coherence_long.out" 'CERULEAN RIVER and OSLO' \
  "DSV4 JANGTQ2 long-context semantic recall evidence"
require_text "$DSV4_JANGTQ2_DIR/dsv4_cache_disk_roundtrip.out" 'Test run with 4 tests in 1 suite passed' \
  "DSV4 JANGTQ2 L2 disk round-trip source proof passed"
require_text "$DSV4_JANGTQ2_DIR/dsv4_cache_topology.out" 'DSV4 paged-incompatible cache skips paged blocks and restores CSA HSA pools from disk.*passed|Test run with 1 test in 1 suite passed' \
  "DSV4 JANGTQ2 CSA/HSA pool disk restore proof passed"
require_text "$DSV4_JANGTQ2_DIR/coherence_long.err" 'peak memory footprint' \
  "DSV4 JANGTQ2 long-context footprint caveat captured"
require_text "$DSV4_AGENTIC_DIR" 'BENCH_DSV4_AGENTIC_TOOL: PASS' \
  "DSV4 JANGTQ2 agentic DSML tool row passed"
require_text "$DSV4_AGENTIC_DIR" 'Tool format: dsml' \
  "DSV4 JANGTQ2 agentic row uses DSML tool format"
require_text "$DSV4_AGENTIC_DIR" 'DSV4_AGENTIC_TOOL_CALL name=lookup_launch_status.*DSV4-77' \
  "DSV4 JANGTQ2 agentic row emitted structured DSML tool call"
require_text "$DSV4_AGENTIC_DIR" 'turn2-tool-result-summary: text=.*toolCalls=0' \
  "DSV4 JANGTQ2 agentic row summarized tool result without recursive tool call"
require_text "$DSV4_AGENTIC_DIR" 'turn3-tool-history-recall: text=.*DSV4-77' \
  "DSV4 JANGTQ2 agentic row recalled tool history"
require_text "$DSV4_AGENTIC_DIR" 'DSV4_AGENTIC_CACHE_STATS.*pagedIncompatible=true.*disk\{hits=[1-9]' \
  "DSV4 JANGTQ2 agentic row proves paged-incompatible disk L2 hit"
require_text "$DSV4_AGENTIC_DIR" 'rep=nil' \
  "DSV4 JANGTQ2 agentic row keeps repetition penalty unset"
require_text "$DSV4_AGENTIC_DIR" 'peak memory footprint' \
  "DSV4 JANGTQ2 agentic footprint caveat captured"

echo "--- release ledger/checklist no-overclaim coverage ---"
require_text "$RELEASE_LEDGER" 'Qwen35 RAM/OOM user crash path is not fully proven fixed end-to-end' \
  "ledger refuses Qwen RAM/OOM overclaim"
require_text "$RELEASE_LEDGER" 'Big-model load cancellation.*live proof blocked|Big-model load cancellation.*Source-covered; live proof blocked' \
  "ledger keeps first-load cancellation live-proof gated"
require_text "$RELEASE_LEDGER" 'ZAYA.*partial/failed|ZAYA/CCA/VL remains partial/failed' \
  "ledger keeps ZAYA partial/failed"
require_text "$RELEASE_LEDGER" 'DSV4 Flash JANGTQ2 coherence is live-proven in local vMLX' \
  "ledger records DSV4 JANGTQ2 coherence promotion"
require_text "$RELEASE_LEDGER" 'DSV4 JANGTQ2 agentic DSML tool/cache proof' \
  "ledger records DSV4 JANGTQ2 agentic DSML/cache proof"
require_text "$RELEASE_LEDGER" 'DSV4 long-context low-footprint remains partial' \
  "ledger keeps DSV4 low-footprint partial"
require_text "$RELEASE_LEDGER" 'Chat UI Think panel routing' \
  "ledger includes Chat UI reasoning route"
require_text "$PR_CHECKLIST" 'Low-level diagnostic rows with raw markers are explicitly excluded' \
  "PR checklist excludes raw diagnostic markers from parser proof"
require_text "$PR_CHECKLIST" 'DSS?V4|DSV4' \
  "PR checklist names DSV4 family"
require_text "$PR_CHECKLIST" 'DSML agentic|agentic DSML|tool history' \
  "PR checklist names DSV4 agentic DSML/tool-history proof"
require_text "$PR_CHECKLIST" 'ZAYA/CCA|CCA/HY3' \
  "PR checklist names CCA/ZAYA family"
require_text "$PR_CHECKLIST" 'Qwen35 JANGTQ RAM/OOM' \
  "PR checklist names Qwen35 RAM/OOM"
reject_text "$RELEASE_LEDGER" 'Qwen35 RAM/OOM user crash path is fully proven|Qwen35.*is release-ready|Qwen35.*release-ready: yes|ZAYA.*is promoted|DSV4.*is release-ready|ZAYA.*is release-ready' \
  "release overclaim wording"

active="$({ ps -axo pid,ppid,rss,etime,command || true; } \
  | rg -i 'CodeSigningHelper|xcodebuild|codesign( |$)|notarytool|/usr/bin/security( |$)|vmlx_engine\\.cli|RunBench|vmlx-live-model-matrix|/Users/eric/osaurus-staging.*(swift-test|xcrun swift|swift test|swift build|swift-driver|swift-frontend|PackagePlugin|\\.build/.*/Cmlx\\.build|/usr/bin/clang .*osaurus-staging)' \
  | rg -v 'rg -i|vmlx-family-matrix-coverage-check' || true)"
if [[ -n "$active" ]]; then
  echo "$active" >&2
  fail_msg "active model/build/keychain process detected; family coverage assertions above are still useful but do not promote live readiness"
else
  pass "no active model/build/keychain process"
fi

if [[ "$fail" -ne 0 ]]; then
  echo "Family matrix coverage guard failed." >&2
  exit 1
fi

echo "Family matrix coverage guard passed."
