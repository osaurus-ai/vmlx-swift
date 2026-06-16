#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/vmlx-qwen-gemma-proof-check.sh --qwen-dir DIR --gemma-dir DIR [--osaurus-root DIR]

Keychain-free verifier for current Qwen/Gemma promotion evidence. This script
only reads source/artifact files; it does not build, launch apps, sign, or touch
Keychain. It fails closed when artifacts are missing or too weak.
USAGE
  exit 64
}

QWEN_DIR="${VMLX_QWEN_PROOF_DIR:-}"
GEMMA_DIR="${VMLX_GEMMA_PROOF_DIR:-}"
OSAURUS_ROOT="${OSAURUS_ROOT:-/Users/eric/osaurus-staging}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --qwen-dir) QWEN_DIR="${2:-}"; shift 2 ;;
    --gemma-dir) GEMMA_DIR="${2:-}"; shift 2 ;;
    --osaurus-root) OSAURUS_ROOT="${2:-}"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

fail=0
pass() { echo "PASS $*"; }
fail_msg() { echo "FAIL $*" >&2; fail=1; }

check_active_processes() {
  local active
  active="$({ ps -axo pid,ppid,rss,etime,command || true; } \
    | rg -i 'CodeSigningHelper|xcodebuild|codesign( |$)|notarytool|/usr/bin/security( |$)|DerivedData|vmlx_engine\.cli|python.*vmlx|RunBench|vmlx-live-model-matrix|/Users/eric/osaurus-staging.*(swift-test|xcrun swift|swift test|swift build|swift-driver|swift-frontend|PackagePlugin|\\.build/.*/Cmlx\\.build|/usr/bin/clang .*osaurus-staging)' \
    | rg -v 'rg -i|vmlx-qwen-gemma-proof-check|assert-osaurus-vmlx-pr-readiness|assert-keychain-free-proof-path|assert-vmlx-gemma4-parser-fix-wired|assert-no-hidden-local-sampler-defaults|assert-openresponses-cache-proof-wiring' || true)"
  if [[ -n "$active" ]]; then
    fail_msg "active heavy/keychain-sensitive process detected before proof classification"
    echo "$active" >&2
  else
    pass "no active model/build/signing/keychain process"
  fi
}

require_dir() {
  local dir="$1" label="$2"
  if [[ -z "$dir" ]]; then
    fail_msg "$label proof dir not provided"
    return 1
  fi
  if [[ ! -d "$dir" ]]; then
    fail_msg "$label proof dir missing: $dir"
    return 1
  fi
  pass "$label proof dir exists: $dir"
}

require_file() {
  local file="$1" label="$2"
  if [[ ! -f "$file" ]]; then
    fail_msg "missing $label: $file"
    return 1
  fi
  pass "$label exists"
}

require_text() {
  local dir="$1" pattern="$2" label="$3"
  if rg -qi "$pattern" "$dir"; then
    pass "$label"
  else
    fail_msg "missing evidence for $label in $dir"
  fi
}

require_fixed_text() {
  local file="$1" pattern="$2" label="$3"
  if rg -Fq "$pattern" "$file"; then
    pass "$label"
  else
    fail_msg "missing evidence for $label in $file"
  fi
}

reject_text() {
  local dir="$1" pattern="$2" label="$3"
  if rg -n -i "$pattern" "$dir"; then
    fail_msg "forbidden evidence found for $label in $dir"
  else
    pass "no $label"
  fi
}

check_status_clean() {
  local status="$1" label="$2"
  if [[ ! -f "$status" ]]; then
    fail_msg "$label status.tsv missing"
    return
  fi
  if awk -F '\t' 'tolower($0) ~ /(^|\t)(fail|failed|error|blocked|invalid|partial)(\t|$)/ { bad=1 } END { exit bad ? 0 : 1 }' "$status"; then
    fail_msg "$label status.tsv contains failed/blocked/partial/invalid row"
  else
    pass "$label status.tsv has no failed/blocked/partial/invalid row markers"
  fi
}

check_family() {
  local dir="$1" label="$2" arch_pattern="$3"
  require_dir "$dir" "$label" || return
  local report="$dir/REPORT.md"
  local status="$dir/status.tsv"
  require_file "$report" "$label REPORT.md"
  require_file "$status" "$label status.tsv"
  check_status_clean "$status" "$label"
  require_text "$dir" 'reasoning|think' "$label reasoning on/off evidence"
  require_text "$dir" 'tool' "$label tool/parser evidence"
  require_text "$dir" 'generation_config|generative config|sampler|temperature|top_p|topP|top-k|topK|min_p|minP|repetition' "$label generation default evidence"
  require_text "$dir" 'prefix|cache hit|l2|disk' "$label prefix/L2 cache evidence"
  require_text "$dir" 'token/s|tok/s|tokens/s' "$label token/s evidence"
  require_text "$dir" 'RSS|phys_footprint|peakRSS|memory' "$label RAM evidence"
  require_text "$dir" "$arch_pattern" "$label architecture-specific cache evidence"
  require_text "$dir" 'leakedThinkMarkers=false|zero reasoning envelope markers|no raw markers leaked|no raw marker leak|leak=false' "$label parser no-leak evidence"
  reject_text "$dir" 'leakedThinkMarkers=true|forced thinking|close-token bias|hidden repetition|rep-penalty rescue|synthetic temperature|parser repair' "$label forced-behavior/parser-leak markers"
  if rg -n '<think>|</think>' "$dir" | rg -v 'template\\.out|rendered tail|directTail|tail=|stored prompt window|Reasoning prompt toggle|BEGIN_FULL_TEXT|END_FULL_TEXT' >/tmp/vmlx-proof-lowlevel-think.$$.txt; then
    echo "WARN $label low-level full-text artifacts include raw think markers; do not use those rows as parser no-leak proof" >&2
    cat /tmp/vmlx-proof-lowlevel-think.$$.txt >&2
  fi
  rm -f /tmp/vmlx-proof-lowlevel-think.$$.txt
}

check_qwen_hybrid_specific_artifacts() {
  local dir="$1"
  require_dir "$dir" "Qwen hybrid-specific" || return
  require_text "$dir" 'companion=ssm' "Qwen cache topology names SSM companion"
  require_text "$dir" 'ssm_companion' "Qwen disk restore writes SSM companion state"
  require_text "$dir" 'ssm\{hits=[1-9]' "Qwen SSM companion cache hit evidence"
  require_text "$dir" 'disk\{hits=[1-9]' "Qwen disk L2 hit evidence"
  require_text "$dir" 'Coord.*HIT.*disk|Coordinator probe.*HIT.*disk' "Qwen coordinator disk-hit probe evidence"
  require_text "$dir" 'BatchEngine TurboQuant B=2' "Qwen TurboQuant B=2 row exists"
  require_text "$dir" 'Slot [01] \(TQ|TQ\(4,4\)\)' "Qwen TurboQuant encoded slot evidence"
  require_text "$dir" 'tqCompressionsA=[1-9]|tqCompressionsB=[1-9]' "Qwen TurboQuant compression counter evidence"
}

check_gemma_cache_specific_artifacts() {
  local dir="$1"
  require_dir "$dir" "Gemma cache-specific" || return
  require_text "$dir" 'rotatingLayers=[1-9]|SWA|sliding' "Gemma rotating/SWA topology evidence"
  require_text "$dir" 'disk\{hits=[1-9]' "Gemma disk L2 hit evidence"
  require_text "$dir" 'BatchEngine TurboQuant B=2' "Gemma TurboQuant B=2 row exists"
  require_text "$dir" 'Slot [01] \(TQ|TQ\(4,4\)\)' "Gemma TurboQuant encoded slot evidence"
  require_text "$dir" 'tqCompressionsA=[1-9]|tqCompressionsB=[1-9]' "Gemma TurboQuant compression counter evidence"
}

check_osaurus_source() {
  local root="$1"
  if [[ ! -d "$root" ]]; then
    fail_msg "Osaurus root missing: $root"
    return
  fi
  local handler="$root/Packages/OsaurusCore/Networking/HTTPHandler.swift"
  local tests="$root/Packages/OsaurusCore/Tests/Service/RuntimePolicySourceTests.swift"
  local launcher="$root/scripts/live-proof/launch-keychain-free-osaurus.sh"
  require_file "$handler" "Osaurus HTTPHandler.swift"
  require_file "$tests" "Osaurus RuntimePolicySourceTests.swift"
  require_file "$launcher" "Osaurus keychain-free launcher"
  require_text "$handler" 'ChannelEvent\.inputClosed|requestTasks\.cancelAll|Task\.checkCancellation|unloadModel' "Osaurus streaming cancellation/load cleanup source wiring"
  require_text "$handler" 'v1/responses|chat/completions|messages|/chat' "Osaurus all streaming endpoints present"
  require_text "$tests" 'inputClosed|responses|chat/completions|messages|Ollama|OSAURUS_DISABLE_KEYCHAIN_FOR_TESTS' "Osaurus source regression assertions"
  require_text "$launcher" 'OSAURUS_DISABLE_KEYCHAIN_FOR_TESTS=1|OSAURUS_TEST_ROOT' "Osaurus keychain-free live launcher env"
}

extract_osaurus_vmlx_pin() {
  local manifest="$1"
  python3 - "$manifest" <<'PY'
import pathlib
import re
import sys

source = pathlib.Path(sys.argv[1]).read_text()
match = re.search(
    r'url:\s*"https://github\.com/osaurus-ai/vmlx-swift",\s*revision:\s*"([0-9a-f]{40})"',
    source,
    re.S,
)
if not match:
    raise SystemExit(1)
print(match.group(1))
PY
}

check_osaurus_vmlx_pin_reproducible() {
  local root="$1"
  local manifest="$root/Packages/OsaurusCore/Package.swift"
  local tests="$root/Packages/OsaurusCore/Tests/Service/RuntimePolicySourceTests.swift"
  require_file "$manifest" "Osaurus Package.swift"
  require_file "$tests" "Osaurus RuntimePolicySourceTests.swift"

  local pin head
  if ! pin="$(extract_osaurus_vmlx_pin "$manifest")"; then
    fail_msg "could not extract Osaurus vmlx-swift revision pin from $manifest"
    return
  fi
  head="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  if [[ "$pin" == "$head" ]]; then
    pass "Osaurus vmlx-swift pin matches local vMLX HEAD: $pin"
  else
    fail_msg "Osaurus vmlx-swift pin $pin does not match local vMLX HEAD $head"
  fi

  if rg -q "$pin" "$tests"; then
    pass "Osaurus runtime source guard names the pinned vMLX revision"
  else
    fail_msg "Osaurus runtime source guard does not name pinned vMLX revision $pin"
  fi

  if git -C "$REPO_ROOT" diff --quiet -- \
      Package.swift \
      Libraries/MLXLMCommon/ReasoningParser.swift \
      Libraries/MLXLMCommon/Tool/Parsers/GemmaFunctionParser.swift \
      Tests/MLXLMCommonFocusedTests/Gemma4ThoughtChannelParserFocusedTests.swift; then
    pass "vMLX parser fix/regression files are committed at the pinned revision"
  else
    fail_msg "vMLX parser fix/regression files have uncommitted diffs; Osaurus pin $pin cannot reproduce local parser proof"
  fi

  pinned_parser="$(git -C "$REPO_ROOT" show "HEAD:Libraries/MLXLMCommon/ReasoningParser.swift" 2>/dev/null || true)"
  if rg -Fq 'channelName == "thought" || channelName == "thinking"' <<<"$pinned_parser" \
      && rg -Fq 'harmonyChannelShouldStripName = false' <<<"$pinned_parser"; then
    pass "pinned vMLX HEAD contains Gemma empty thought-channel parser fix"
  else
    fail_msg "pinned vMLX HEAD lacks Gemma empty thought-channel parser fix"
  fi

  if git -C "$REPO_ROOT" show "HEAD:Tests/MLXLMCommonFocusedTests/Gemma4ThoughtChannelParserFocusedTests.swift" 2>/dev/null \
      | rg -q 'empty thought channel without newline does not surface thought'; then
    pass "pinned vMLX HEAD contains Gemma parser regression"
  else
    fail_msg "pinned vMLX HEAD lacks Gemma parser regression"
  fi

  if git -C "$REPO_ROOT" show "HEAD:Libraries/MLXLMCommon/Tool/Parsers/GemmaFunctionParser.swift" 2>/dev/null \
      | rg -q 'trimmingCharacters\(in: \.whitespacesAndNewlines\)'; then
    pass "pinned vMLX HEAD contains Gemma tool whitespace parser fix"
  else
    fail_msg "pinned vMLX HEAD lacks Gemma tool whitespace parser fix"
  fi

  if git -C "$REPO_ROOT" show "HEAD:Tests/MLXLMTests/ToolCallEdgeCasesTests.swift" 2>/dev/null \
      | rg -q 'Gemma-4 tool-call parser trims whitespace around function names and keys'; then
    pass "pinned vMLX HEAD contains Gemma tool whitespace regression"
  else
    fail_msg "pinned vMLX HEAD lacks Gemma tool whitespace regression"
  fi

  if git -C "$REPO_ROOT" show "HEAD:Package.swift" \
      | rg -q 'Gemma4ThoughtChannelParserFocusedTests\.swift'; then
    pass "pinned vMLX HEAD includes Gemma parser regression in the focused test target"
  else
    fail_msg "pinned vMLX HEAD does not include Gemma parser regression in the focused test target"
  fi
}

check_vmlx_gemma_parser_source() {
  local parser="$REPO_ROOT/Libraries/MLXLMCommon/ReasoningParser.swift"
  local tool_parser="$REPO_ROOT/Libraries/MLXLMCommon/Tool/Parsers/GemmaFunctionParser.swift"
  local tests="$REPO_ROOT/Tests/MLXLMCommonFocusedTests/Gemma4ThoughtChannelParserFocusedTests.swift"
  local tool_tests="$REPO_ROOT/Tests/MLXLMTests/ToolCallEdgeCasesTests.swift"
  require_file "$parser" "vMLX ReasoningParser.swift"
  require_file "$tool_parser" "vMLX GemmaFunctionParser.swift"
  require_file "$tests" "vMLX Gemma parser focused tests"
  require_file "$tool_tests" "vMLX Gemma tool parser edge-case tests"
  require_fixed_text "$parser" 'channelName == "thought" || channelName == "thinking"' "vMLX Gemma bare thought-channel source fix"
  require_fixed_text "$parser" 'harmonyChannelShouldStripName = false' "vMLX Gemma empty thought-channel source fix"
  require_text "$tests" 'empty thought channel without newline does not surface thought' "vMLX Gemma no-newline thought regression"
  require_text "$tests" 'pre<\|channel>thought<channel\|>answer' "vMLX Gemma no-newline thought fixture"
  require_text "$REPO_ROOT/Package.swift" 'Gemma4ThoughtChannelParserFocusedTests\.swift' "vMLX Gemma parser regression target wiring"
  require_text "$tool_parser" 'trimmingCharacters\(in: \.whitespacesAndNewlines\)' "vMLX Gemma tool parser trims whitespace"
  require_text "$tool_tests" 'Gemma-4 tool-call parser trims whitespace around function names and keys' "vMLX Gemma tool whitespace regression"
  require_text "$tool_tests" 'arguments\[" location"\] == nil' "vMLX Gemma tool whitespace negative-key assertion"
}

check_vmlx_no_hidden_defaults_source() {
  local no_hidden="$REPO_ROOT/Tests/MLXLMCommonFocusedTests/NoHiddenReasoningCloseBiasFocusedTests.swift"
  local matrix="$REPO_ROOT/scripts/vmlx-live-model-matrix.sh"
  local jangpress="$REPO_ROOT/RunBench/JangPressRegressionBench.swift"
  local bench="$REPO_ROOT/RunBench/Bench.swift"

  require_file "$no_hidden" "vMLX no-hidden-behavior focused tests"
  require_file "$matrix" "vMLX live model matrix script"
  require_file "$jangpress" "vMLX JangPress regression bench"
  require_file "$bench" "vMLX RunBench"

  require_text "$no_hidden" 'sampler_defaults' "vMLX source guard requires bundle sampler defaults"
  require_text "$no_hidden" 'fail:missing-bundle-sampler-defaults-would-use-engine-fallback' "vMLX source guard fails missing sampler defaults"
  require_text "$no_hidden" 'missing bundle sampler defaults are failing evidence' "vMLX source guard rejects engine-fallback sampler promotion"
  require_text "$matrix" 'sampler_defaults' "live matrix records sampler defaults"
  require_text "$matrix" 'fail:missing-bundle-sampler-defaults-would-use-engine-fallback' "live matrix fails missing bundle sampler defaults"
  require_text "$matrix" 'bundle-derived' "live matrix reports bundle-derived defaults"
  reject_text "$jangpress" 'var p = GenerateParameters\(maxTokens: maxNewTokens, temperature: 0\)' "JangPress forced temperature zero"
  reject_text "$bench" 'BENCH_DSV4_REPETITION_PENALTY"\] \?\? "1\.0"|BENCH_DSV4_MAX_REPETITION_PENALTY"\] \?\? "1\.05"|dsv4MaxReasoningRepetitionPenalty' "DSV4 hidden repetition-penalty rescue"
  reject_text "$bench" 'text\.isEmpty \? reasoning : text|visible\.isEmpty \? reasoning : visible|visible\.isEmpty \? r\.reasoning : visible|r\.text\.isEmpty \? r\.reasoning : r\.text|reasoning\.isEmpty \? r\.text : r\.reasoning|let combined = text \+ reasoning' "reasoning-only output counted as visible answer"
  reject_text "$REPO_ROOT/RunBench/OmniBench.swift" 'visible\.isEmpty \? reasoning\.trimmingCharacters|text\.isEmpty \? reasoning : text' "OmniBench reasoning-only output counted as visible answer"
}

check_active_processes
check_vmlx_gemma_parser_source
check_vmlx_no_hidden_defaults_source
check_family "$QWEN_DIR" "Qwen" 'SSM|MTP|TurboQuant|TQ|KV|hybrid|rederive'
check_qwen_hybrid_specific_artifacts "$QWEN_DIR"
check_family "$GEMMA_DIR" "Gemma" 'SWA|sliding|rotating|KV|TurboQuant|TQ'
check_gemma_cache_specific_artifacts "$GEMMA_DIR"
check_osaurus_source "$OSAURUS_ROOT"
check_osaurus_vmlx_pin_reproducible "$OSAURUS_ROOT"

if [[ "$fail" -ne 0 ]]; then
  echo "proof check failed; do not promote Qwen/Gemma/Osaurus row" >&2
  exit 1
fi

echo "proof check passed; Qwen/Gemma artifacts and Osaurus source wiring meet this verifier's minimum evidence gate"
