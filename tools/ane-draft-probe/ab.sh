#!/usr/bin/env bash
# A/B the Neural Engine MTP drafter against the GPU head with RunBench.
#
# Same bundle, prompt, greedy sampling, token budget and depth in every
# arm; each arm is warmup + N measured runs (median). The NativeMTP stats
# line (acceptance, forwards, phase seconds) and the PERF line land in
# $OUT/<arm>.log; a SHA of the generated text is printed per arm so
# output equivalence is checked, not assumed.
#
#   BENCH_MODEL=/path/to/bundle tools/ane-draft-probe/ab.sh [ar gpu ane ane64k]
#
# Env knobs (defaults): DEPTH=3 TOKENS=256 RUNS=3 PROMPT=<prose> OUT=/tmp/ane-ab
set -euo pipefail
cd "$(dirname "$0")/../.."

: "${BENCH_MODEL:?set BENCH_MODEL to a bundle directory}"
DEPTH=${DEPTH:-3}
TOKENS=${TOKENS:-256}
RUNS=${RUNS:-3}
PROMPT=${PROMPT:-"Write one long paragraph describing ocean waves. Be verbose and detailed."}
OUT=${OUT:-/tmp/ane-ab}
BIN=${BIN:-.build/release/RunBench}
mkdir -p "$OUT"

arms=("$@")
[ ${#arms[@]} -eq 0 ] && arms=(ar gpu ane)

run_arm() {
  local arm=$1
  local -a envs=(
    BENCH_PERF=1
    BENCH_MODEL="$BENCH_MODEL"
    BENCH_MAX_TOKENS="$TOKENS"
    BENCH_PERF_RUNS="$RUNS"
    BENCH_PERF_WARMUP=1
    BENCH_PERF_TEMP=0
    BENCH_PERF_PROMPT="$PROMPT"
    BENCH_PERF_VARIANT="$arm"
    BENCH_PERF_FULL_TEXT=1
    BENCH_PERF_ALLOCATOR_CACHE_BYTES=8589934592
    VMLX_ANE_MTP=0
  )
  case $arm in
    ar) ;;
    gpu) envs+=(BENCH_PERF_NATIVE_MTP_DEPTH="$DEPTH") ;;
    ane) envs+=(BENCH_PERF_NATIVE_MTP_DEPTH="$DEPTH" VMLX_ANE_MTP=1) ;;
    ane64k) envs+=(BENCH_PERF_NATIVE_MTP_DEPTH="$DEPTH" VMLX_ANE_MTP=1 VMLX_ANE_MTP_VOCAB=65536) ;;
    anefull) envs+=(BENCH_PERF_NATIVE_MTP_DEPTH="$DEPTH" VMLX_ANE_MTP=1 VMLX_ANE_MTP_VOCAB=1000000) ;;
    anefp16) envs+=(BENCH_PERF_NATIVE_MTP_DEPTH="$DEPTH" VMLX_ANE_MTP=1 VMLX_ANE_MTP_WEIGHTS=fp16) ;;
    anefp16full) envs+=(BENCH_PERF_NATIVE_MTP_DEPTH="$DEPTH" VMLX_ANE_MTP=1 VMLX_ANE_MTP_WEIGHTS=fp16 VMLX_ANE_MTP_VOCAB=1000000) ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
  echo "=== arm $arm"
  env "${envs[@]}" "$BIN" > "$OUT/$arm.log" 2>&1 || { echo "arm $arm FAILED"; tail -20 "$OUT/$arm.log"; return 1; }
  grep -E "^PERF model=|\[NativeMTP\] depth=|ANE drafter" "$OUT/$arm.log" | sed -E 's/(tokps_median=[0-9.]+|tokps_best=[0-9.]+|acceptedByDepth=[^ ]+|mtpForwards=[0-9]+|aneForwards=[0-9]+|drafter=[a-z]+|mtpDraftSec=[0-9.]+|targetVerifySec=[0-9.]+|iteratorWallSec=[0-9.]+|rejected=[0-9]+|bonus=[0-9]+)/\n    \1/g' | grep -E "^PERF|^    |ANE drafter" | head -40
  echo "    text sha (last run): $(awk '/BEGIN_FULL_TEXT/{buf="";on=1;next} /END_FULL_TEXT/{on=0;last=buf} on{buf=buf $0 "\n"} END{printf "%s", last}' "$OUT/$arm.log" | shasum | cut -c1-12)"
}

for arm in "${arms[@]}"; do run_arm "$arm"; done
