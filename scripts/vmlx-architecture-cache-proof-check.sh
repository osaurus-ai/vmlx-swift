#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fail=0

pass() {
  echo "PASS $*"
}

fail_msg() {
  echo "FAIL $*" >&2
  fail=1
}

require_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    pass "file exists: ${file#$ROOT/}"
  else
    fail_msg "missing file: ${file#$ROOT/}"
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

require_fixed() {
  local file="$1"
  local text="$2"
  local label="$3"
  if rg -q --fixed-strings "$text" "$file"; then
    pass "$label"
  else
    fail_msg "missing $label in ${file#$ROOT/}"
  fi
}

reject_text() {
  local file="$1"
  local pattern="$2"
  local label="$3"
  if rg -n "$pattern" "$file"; then
    fail_msg "forbidden $label in ${file#$ROOT/}"
  else
    pass "no $label"
  fi
}

MATRIX="$ROOT/scripts/vmlx-live-model-matrix.sh"
DSV4_DISK="$ROOT/Tests/MLXLMTests/DeepseekV4CacheDiskRoundTripTests.swift"
DSV4_SMOKE="$ROOT/Tests/MLXLMTests/DeepseekV4ModelSmokeTests.swift"
ZAYA_DISK="$ROOT/Tests/MLXLMTests/ZayaCCACacheDiskRoundTripTests.swift"
TOPO="$ROOT/Tests/MLXLMCommonFocusedTests/CacheCoordinatorTopologyFocusedTests.swift"
MEDIA="$ROOT/Tests/MLXLMTests/CacheCoordinatorMediaSaltTests.swift"
TQ_PROBE="$ROOT/Tests/MLXLMTests/TurboQuantCompileProbeTests.swift"
TQ_LIST="$ROOT/Tests/MLXLMTests/CompilableCacheListTests.swift"
MIMO_TOPO="$ROOT/Tests/MLXLMTests/MiMoV2FlashCacheTopologyTests.swift"
NOHIDDEN="$ROOT/Tests/MLXLMCommonFocusedTests/NoHiddenReasoningCloseBiasFocusedTests.swift"
GEMMA_PARSER="$ROOT/Tests/MLXLMCommonFocusedTests/Gemma4ThoughtChannelParserFocusedTests.swift"
SETTINGS_TESTS="$ROOT/Tests/MLXLMCommonFocusedTests/VMLXServerRuntimeSettingsTests.swift"
PARSER="$ROOT/Libraries/MLXLMCommon/ReasoningParser.swift"
QG_PROOF="$ROOT/scripts/vmlx-qwen-gemma-proof-check.sh"

for file in \
  "$MATRIX" "$DSV4_DISK" "$DSV4_SMOKE" "$ZAYA_DISK" "$TOPO" "$MEDIA" \
  "$TQ_PROBE" "$TQ_LIST" "$MIMO_TOPO" "$NOHIDDEN" "$GEMMA_PARSER" "$SETTINGS_TESTS" "$PARSER" "$QG_PROOF"; do
  require_file "$file"
done

require_text "$MATRIX" 'Full-attention models need KV/prefix/L2 proof' \
  "matrix requires full-attention KV/prefix/L2 proof"
require_text "$MATRIX" 'hybrid SSM models' "matrix names hybrid SSM models"
require_text "$MATRIX" 'need attention KV plus SSM companion proof' \
  "matrix requires hybrid SSM companion proof"
require_text "$MATRIX" 'CCA/HY3 style models need companion' "matrix names CCA/HY3 companion caches"
require_text "$MATRIX" 'cache/pooling proof' "matrix requires CCA/HY3 cache/pooling proof"
require_text "$MATRIX" 'DeepSeek-V4 needs CSA/HSA/SWA pool restore proof' \
  "matrix requires DSV4 CSA/HSA/SWA pool proof"
require_text "$MATRIX" 'MiMo v2.5 needs full-attention KV plus rotating SWA L2 proof' \
  "matrix requires MiMo v2.5 full/SWA KV proof"
require_text "$MATRIX" 'VL/video models need media payload' \
  "matrix requires VL/video media cache proof"
require_fixed "$MATRIX" "n-a:deepseek-v4-uses-swa-csa-hsa-hybrid-pool-cache-not-turboquant-kv" \
  "DSV4 excludes generic TurboQuant KV substitution"
require_text "$MATRIX" 'BENCH_GROWING_CHAT_CACHE=1' \
  "matrix runs growing chat prefix/cache row"
require_text "$MATRIX" 'BENCH_BATCH_DISK_RESTORE=1' \
  "matrix runs disk-L2 restore row"
require_text "$MATRIX" 'BENCH_BATCH_TQ_B2=1' \
  "matrix runs TurboQuant B=2 row where applicable"
require_text "$MATRIX" 'qwen_multiturn_tool' \
  "matrix runs Qwen multi-turn tool row"
require_text "$MATRIX" 'cache rows must match architecture: KV/TurboQuant KV, hybrid SSM companion, CCA/HY3 companion, DSV4 CSA/HSA/SWA' \
  "matrix report prints architecture cache acceptance boundary"

require_text "$DSV4_DISK" 'dsv4_0_pool_comp' "DSV4 disk round-trip stores compressor pool"
require_text "$DSV4_DISK" 'dsv4_0_pool_idx' "DSV4 disk round-trip stores indexer pool"
require_text "$DSV4_DISK" 'RotatingKVCacheWrapper' "DSV4 cache wrapper conformance covered"
require_text "$DSV4_DISK" 'Mixed per-layer|mixed per-layer|mixed cache' "DSV4 mixed per-layer cache covered"
require_text "$TOPO" 'DSV4 paged-incompatible cache skips paged blocks and restores CSA HSA pools from disk' \
  "DSV4 topology guard rejects paged false positives"
require_text "$TOPO" 'dsv4_0_pool_comp' "DSV4 topology guard checks disk pool compressor"
require_text "$TOPO" 'dsv4_0_pool_idx' "DSV4 topology guard checks disk pool indexer"

require_text "$DSV4_SMOKE" 'newCache keeps DSV4 hybrid cache even when request asks for TurboQuant' \
  "DSV4 newCache ignores generic TurboQuant request"
require_text "$DSV4_SMOKE" '!cache.contains \{ \$0 is TurboQuantKVCache \}' \
  "DSV4 default cache rejects TurboQuantKVCache"
require_text "$DSV4_SMOKE" 'DSV4_KV_MODE=tq is the explicit diagnostic simple-cache override' \
  "DSV4 TurboQuant simple cache remains explicit diagnostic only"

require_text "$ZAYA_DISK" 'LayerKind\.zayaCCA' "ZAYA CCA disk layer kind covered"
require_text "$ZAYA_DISK" 'readCCA\(\)\.conv' "ZAYA CCA conv state round-trip covered"
require_text "$ZAYA_DISK" 'readCCA\(\)\.prev' "ZAYA CCA previous-state round-trip covered"
require_text "$ZAYA_DISK" 'Mixed per-layer round-trip: KVCacheSimple \+ ZayaCCACache \+ RotatingKVCache' \
  "ZAYA mixed per-layer cache round-trip covered"
require_text "$TOPO" 'ZAYA CCA format-v2 disk payload is enough for hybrid cache hit' \
  "ZAYA topology guard accepts CCA v2 payload as companion cache"
require_text "$TOPO" 'ZAYA CCA v2 disk payload already carries path-dependent state' \
  "ZAYA topology guard documents path-dependent CCA state"

require_text "$TOPO" 'hybridPagedHitRequiresSSMCompanion' \
  "hybrid SSM paged hit requires companion state"
require_text "$TOPO" 'hybridPagedHitRejectsPartialSSMCompanion' \
  "hybrid SSM paged hit rejects partial companion"
require_text "$TOPO" 'hybridDiskHitRejectsPartialSSMCompanion' \
  "hybrid SSM disk hit rejects partial companion"
require_text "$NOHIDDEN" 'history-boundary rederive feeds remaining tokens batch-first' \
  "SSM/CCA history-boundary rederive batch-first regression covered"
require_text "$SETTINGS_TESTS" '#expect\(settings\.cache\.enableSSMReDerive\)' \
  "server runtime defaults keep SSM rederive enabled"
require_text "$SETTINGS_TESTS" 'settings\.cache\.enableSSMReDerive = true' \
  "Osaurus-facing SSM rederive default wiring source guard covered"

require_text "$TOPO" 'Gemma4 cache topology focused contracts' \
  "Gemma4 topology suite exists"
require_text "$TOPO" 'Mixed Rotating\+Simple Gemma4 cache classifies as heterogeneous' \
  "Gemma4 mixed rotating/simple topology covered"
require_text "$TOPO" 'All-Rotating Gemma4 cache classifies as rotating' \
  "Gemma4 all-rotating topology covered"
require_text "$TOPO" 'actual Gemma4TextModel newCache matches mixed and maxKVSize topologies' \
  "Gemma4 newCache topology covered"
require_text "$TOPO" 'TokenIterator compiled decode promotes all-rotating SWA caches' \
  "Gemma4 SWA compile promotion covered"
require_text "$GEMMA_PARSER" 'empty thought channel without newline does not surface thought' \
  "Gemma4 no-newline thought regression covered"
require_text "$GEMMA_PARSER" 'pre<\|channel>thought<channel\|>answer' \
  "Gemma4 no-newline thought fixture covered"
require_fixed "$PARSER" 'channelName == "thought" || channelName == "thinking"' \
  "Gemma4 bare thought-channel source fix guarded"
require_fixed "$PARSER" 'harmonyChannelShouldStripName = false' \
  "Gemma4 empty thought-channel source fix guarded"

require_text "$MIMO_TOPO" 'MiMo V2.5 hybrid full/SWA cache topology' \
  "MiMo v2.5 topology suite exists"
require_text "$MIMO_TOPO" 'per-layer KV heads and cache classes match full/SWA layer pattern' \
  "MiMo v2.5 per-layer full/SWA cache classes covered"
require_text "$MIMO_TOPO" 'TurboQuant KV only promotes full-attention cache layers' \
  "MiMo v2.5 TurboQuant KV full-layer-only promotion covered"
require_text "$MIMO_TOPO" 'cache topology snapshot marks MiMo full/SWA KV as disk-backed rotating topology' \
  "MiMo v2.5 topology snapshot marks rotating restore boundary"
require_text "$MIMO_TOPO" 'L2 disk round trip preserves hybrid full/SWA cache kinds' \
  "MiMo v2.5 L2 disk native full/SWA round-trip covered"
require_text "$MIMO_TOPO" 'L2 disk round trip preserves TurboQuant full layers plus rotating SWA' \
  "MiMo v2.5 L2 disk TurboQuant plus rotating SWA round-trip covered"
require_text "$MIMO_TOPO" 'CacheCoordinator L2 prefix hit preserves MiMo full/SWA cache topology' \
  "MiMo v2.5 coordinator L2 prefix hit covered"
require_text "$SETTINGS_TESTS" '\("mimo_v2", \.xmlFunction, "think_xml"\)' \
  "server runtime family policy includes MiMo v2.5 reasoning/tool autodetect"

require_text "$MEDIA" 'mediaSalt is not folded into the paged block hash chain' \
  "media salt paged-prefix isolation covered"
require_text "$MEDIA" 'nil mediaSalt must be a distinct domain' \
  "nil media salt isolation covered"
require_text "$TOPO" 'mediaSalt' "topology tests include media salt cache scope"

require_text "$TQ_PROBE" 'TurboQuantKVCache' "TurboQuant KV compile probe exists"
require_text "$TQ_LIST" 'TurboQuantKVCache' "TurboQuant KV cache-list serializer coverage exists"
require_text "$TQ_LIST" 'CompilableCacheList' "compilable cache-list coverage exists"

require_text "$QG_PROOF" "check_family.*Qwen.*SSM\\|MTP\\|TurboQuant\\|TQ\\|KV\\|hybrid\\|rederive" \
  "Qwen proof checker requires hybrid SSM/TQ cache evidence"
require_text "$QG_PROOF" 'check_qwen_hybrid_specific_artifacts' \
  "Qwen proof checker has explicit hybrid-specific artifact gate"
require_text "$QG_PROOF" 'Qwen cache topology names SSM companion' \
  "Qwen proof checker requires SSM companion topology evidence"
require_text "$QG_PROOF" 'Qwen disk restore writes SSM companion state' \
  "Qwen proof checker requires SSM companion disk state"
require_text "$QG_PROOF" 'Qwen SSM companion cache hit evidence' \
  "Qwen proof checker requires SSM companion hit evidence"
require_text "$QG_PROOF" 'Qwen TurboQuant compression counter evidence' \
  "Qwen proof checker requires TurboQuant compression counters"
require_text "$QG_PROOF" "check_family.*Gemma.*SWA\\|sliding\\|rotating\\|KV\\|TurboQuant\\|TQ" \
  "Gemma proof checker requires SWA/TQ cache evidence"
require_text "$QG_PROOF" 'check_gemma_cache_specific_artifacts' \
  "Gemma proof checker has explicit cache-specific artifact gate"
require_text "$QG_PROOF" 'Gemma rotating/SWA topology evidence' \
  "Gemma proof checker requires rotating/SWA topology evidence"
require_text "$QG_PROOF" 'Gemma TurboQuant compression counter evidence' \
  "Gemma proof checker requires TurboQuant compression counters"

reject_text "$MATRIX" 'deepseek-v4.*BENCH_BATCH_TQ_B2=1|BENCH_BATCH_TQ_B2=1.*deepseek-v4' \
  "DSV4 generic TurboQuant KV row"

active_forbidden="$({ ps -axo pid,ppid,rss,etime,command || true; } \
  | rg -i 'CodeSigningHelper|xcodebuild|codesign( |$)|notarytool|/usr/bin/security( |$)|/Users/eric/osaurus-staging.*(swift-test|xcrun swift|swift test|swift build|swift-driver|swift-frontend|PackagePlugin|\\.build/.*/Cmlx\\.build|/usr/bin/clang .*osaurus-staging)' \
  | rg -v 'rg -i|vmlx-architecture-cache-proof-check|assert-keychain-free-proof-path' || true)"
if [[ -n "$active_forbidden" ]]; then
  echo "FAIL active Osaurus keychain-sensitive validation process detected:" >&2
  echo "$active_forbidden" >&2
  fail=1
else
  pass "no active Osaurus keychain-sensitive validation process"
fi

exit "$fail"
