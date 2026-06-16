#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fail=0

pass() { echo "PASS $*"; }
fail_msg() { echo "FAIL $*" >&2; fail=1; }
warn() { echo "WARN $*" >&2; }

require_file() {
  local file="$1"
  local label="$2"
  if [[ -f "$file" ]]; then
    pass "$label exists"
  else
    fail_msg "missing $label: $file"
  fi
}

require_text() {
  local file="$1"
  local pattern="$2"
  local label="$3"
  if rg -q "$pattern" "$file"; then
    pass "$label"
  else
    fail_msg "missing $label in ${file#$ROOT/}"
  fi
}

require_clean_status() {
  local dir="$1"
  local label="$2"
  if [[ ! -f "$dir/status.tsv" ]]; then
    fail_msg "$label status.tsv missing: $dir/status.tsv"
    return
  fi
  if awk -F '\t' 'tolower($0) ~ /(^|\t)(fail|failed|error|blocked|invalid|partial)(\t|$)/ { bad=1 } END { exit bad ? 0 : 1 }' "$dir/status.tsv"; then
    fail_msg "$label status.tsv contains failed/blocked/partial/invalid row"
  else
    pass "$label status.tsv clean"
  fi
}

require_tracked() {
  local path="$1"
  local label="$2"
  if git -C "$ROOT" ls-files --error-unmatch "$path" >/dev/null 2>&1; then
    pass "$label is tracked"
  else
    fail_msg "$label is not tracked: $path"
  fi
}

is_allowed_push_path() {
  case "$1" in
    .gitignore) return 0 ;;
    AGENTS.md) return 0 ;;
    Libraries/MLXLMCommon/ReasoningParser.swift) return 0 ;;
    Tests/MLXLMCommonFocusedTests/Gemma4ThoughtChannelParserFocusedTests.swift) return 0 ;;
    scripts/vmlx-qwen-gemma-proof-check.sh) return 0 ;;
    scripts/vmlx-architecture-cache-proof-check.sh) return 0 ;;
    scripts/vmlx-push-readiness-scope-check.sh) return 0 ;;
    scripts/vmlx-osaurus-release-readiness-audit.sh) return 0 ;;
    scripts/vmlx-release-dirty-scope-classifier.sh) return 0 ;;
    scripts/vmlx-family-matrix-coverage-check.sh) return 0 ;;
    scripts/vmlx-downloaded-model-inventory-check.sh) return 0 ;;
    scripts/vmlx-ram-footprint-proof-boundary-check.sh) return 0 ;;
    scripts/vmlx-reasoning-visible-boundary-check.sh) return 0 ;;
    .agents/vmlx-osaurus/codex/PROGRESS.md) return 0 ;;
    .agents/vmlx-osaurus/codex/TEST-RESULTS.md) return 0 ;;
    .agents/vmlx-osaurus/codex/MATRIX-LEDGER.md) return 0 ;;
    .agents/vmlx-osaurus/codex/PR-READINESS.md) return 0 ;;
    .agents/vmlx-osaurus/codex/PR-COMMENT-CURRENT.md) return 0 ;;
    .agents/vmlx-osaurus/codex/PR-PROOF-CHECKLIST.md) return 0 ;;
    .agents/vmlx-osaurus/codex/RELEASE-READINESS.md) return 0 ;;
    .agents/vmlx-osaurus/codex/2026-05-*.md) return 0 ;;
    .agents/vmlx-osaurus/codex/outbox/2026-05-*.md) return 0 ;;
    *) return 1 ;;
  esac
}

echo "--- process baseline ---"
active="$({ ps -axo pid,ppid,rss,etime,command || true; } \
  | rg -i 'RunBench|vmlx-live-model-matrix|vmlx_engine\.cli|CodeSigningHelper|xcodebuild|codesign( |$)|notarytool|/usr/bin/security( |$)|/Users/eric/osaurus-staging.*(swift-test|xcrun swift|swift test|swift build|swift-driver|swift-frontend|PackagePlugin|\\.build/.*/Cmlx\\.build|/usr/bin/clang .*osaurus-staging)' \
  | rg -v 'rg -i|vmlx-push-readiness-scope-check' || true)"
if [[ -n "$active" ]]; then
  echo "$active" >&2
  fail_msg "active model/build/signing/keychain-sensitive process detected"
else
  pass "no active model/build/signing/keychain-sensitive process"
fi

if [[ -e /tmp/vmlx-runbench-live.lock ]]; then
  fail_msg "/tmp/vmlx-runbench-live.lock exists"
else
  pass "RunBench live lock clear"
fi

echo "--- proven source files ---"
require_file "$ROOT/Libraries/MLXLMCommon/ReasoningParser.swift" "ReasoningParser"
require_text "$ROOT/Libraries/MLXLMCommon/ReasoningParser.swift" 'stripIdentifierOnlyAtEnd: true\)' \
  "Gemma empty-thought parser fix"
require_file "$ROOT/Tests/MLXLMCommonFocusedTests/Gemma4ThoughtChannelParserFocusedTests.swift" \
  "Gemma focused parser regression"
require_text "$ROOT/Tests/MLXLMCommonFocusedTests/Gemma4ThoughtChannelParserFocusedTests.swift" \
  'empty thought channel without newline does not surface thought' \
  "Gemma empty-thought parser regression"
require_text "$ROOT/Tests/MLXLMCommonFocusedTests/Gemma4ThoughtChannelParserFocusedTests.swift" \
  'pre<\|channel>thought<channel\|>answer' \
  "Gemma empty-thought fixture"

echo "--- proof artifacts ---"
QWEN_DIR="${VMLX_QWEN_PROOF_DIR:-/tmp/vmlx-qwen35-jangtq-turnmatrix-post-vlfix-20260524-1545}"
GEMMA_DIR="${VMLX_GEMMA_PROOF_DIR:-/tmp/vmlx-gemma4-turnmatrix-current-20260524-1516}"
require_file "$QWEN_DIR/REPORT.md" "Qwen proof REPORT"
require_file "$GEMMA_DIR/REPORT.md" "Gemma proof REPORT"
require_clean_status "$QWEN_DIR" "Qwen proof"
require_clean_status "$GEMMA_DIR" "Gemma proof"

echo "--- dirty scope ---"
dirty_paths="$(
  {
    git -C "$ROOT" diff --name-only
    git -C "$ROOT" diff --cached --name-only
    git -C "$ROOT" ls-files --others --exclude-standard
  } | sort -u
)"

if [[ -z "$dirty_paths" ]]; then
  pass "worktree clean"
else
  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    if is_allowed_push_path "$path"; then
      pass "allowed dirty path: $path"
    else
      fail_msg "dirty path outside proven push scope: $path"
    fi
  done <<< "$dirty_paths"
fi

echo "--- reproducibility boundary ---"
head_rev="$(git -C "$ROOT" rev-parse HEAD)"
require_tracked "Libraries/MLXLMCommon/ReasoningParser.swift" "ReasoningParser"
require_tracked "Tests/MLXLMCommonFocusedTests/Gemma4ThoughtChannelParserFocusedTests.swift" \
  "Gemma focused parser regression"

if git -C "$ROOT" diff --quiet -- Libraries/MLXLMCommon/ReasoningParser.swift \
    Tests/MLXLMCommonFocusedTests/Gemma4ThoughtChannelParserFocusedTests.swift \
  && git -C "$ROOT" show "HEAD:Libraries/MLXLMCommon/ReasoningParser.swift" \
      | rg -q 'stripIdentifierOnlyAtEnd: true\)' \
  && git -C "$ROOT" show "HEAD:Tests/MLXLMCommonFocusedTests/Gemma4ThoughtChannelParserFocusedTests.swift" 2>/dev/null \
      | rg -q 'empty thought channel without newline does not surface thought'; then
  pass "parser fix/regression are reproducible from HEAD $head_rev"
else
  warn "parser fix/regression are not fully reproducible from HEAD $head_rev; Osaurus cannot reproduce them by pinning current HEAD"
  fail=1
fi

if [[ "$fail" -ne 0 ]]; then
  cat >&2 <<'EOF'
vMLX push readiness is BLOCKED.

Only push after either:
- unrelated/unowned dirty paths are isolated away, and the proven parser/guard changes are committed in a scoped commit; or
- Eric explicitly approves a broader publish scope.

Do not bump Osaurus pins until this guard passes against the intended vMLX revision.
EOF
  exit 1
fi

echo "vMLX push readiness scope guard passed."
