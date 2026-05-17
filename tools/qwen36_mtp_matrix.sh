#!/usr/bin/env zsh
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RUNBENCH="${RUNBENCH:-$ROOT/.build/release/RunBench}"
STAMP="${STAMP:-$(date +%Y%m%dT%H%M%S)}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$ROOT/docs/local/qwen36-mtp-matrix/$STAMP}"
SUMMARY="$ARTIFACT_DIR/summary.tsv"

mkdir -p "$ARTIFACT_DIR"
printf "label\tstatus\tlog\n" > "$SUMMARY"

if [[ ! -x "$RUNBENCH" ]]; then
  echo "RunBench not found at $RUNBENCH. Build it first: swift build -c release --product RunBench --jobs 2" >&2
  exit 2
fi

models=(
  "qwen36_27b_jang4m|/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP"
  "qwen36_27b_mxfp4|/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP4-MTP"
  "qwen36_27b_mxfp8|/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP8-MTP"
  "qwen36_35b_jang2k|/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-JANG_2K-MTP"
  "qwen36_35b_mxfp4|/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP4-MTP"
  "qwen36_35b_mxfp8|/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP"
)

base_env=(
  "MLXPRESS_ALIGN_SAFETENSORS=0"
  "JANGPRESS_ALIGN_SAFETENSORS=0"
  "MLXPRESS_PRESTACK=0"
  "JANGPRESS_PRESTACK=0"
  "BENCH_PERF_PROMPT=Count from 1 to 50 in order, separated by commas."
  "BENCH_PERF_FULL_TEXT=1"
  "BENCH_MAX_TOKENS=192"
  "BENCH_PERF_WARMUP=0"
  "BENCH_PERF_RUNS=1"
  "BENCH_PERF_TEMP=0"
  "BENCH_PERF_TOP_P=1"
  "BENCH_PERF_TOP_K=0"
)

run_logged() {
  local label="$1"
  shift
  local log="$ARTIFACT_DIR/$label.log"
  echo "[$(date '+%H:%M:%S')] $label"
  set +e
  (
    cd "$ROOT"
    env "$@" "$RUNBENCH"
  ) >"$log" 2>&1
  local exit_code=$?
  set -e
  printf "%s\t%d\t%s\n" "$label" "$exit_code" "$log" >> "$SUMMARY"
  tail -n 12 "$log" || true
  echo
  return "$exit_code"
}

for item in "${models[@]}"; do
  label="${item%%|*}"
  model="${item#*|}"
  if [[ ! -d "$model" ]]; then
    missing="$ARTIFACT_DIR/${label}_missing.log"
    echo "missing model path: $model" > "$missing"
    printf "%s\t%d\t%s\n" "${label}_missing" 66 "$missing" >> "$SUMMARY"
    continue
  fi

  run_logged "${label}_census" \
    "BENCH_MODEL=$model" \
    "BENCH_MTP_CENSUS=1" || true

  run_logged "${label}_ar_text" \
    "${base_env[@]}" \
    "BENCH_MODEL=$model" \
    "BENCH_PERF=1" \
    "BENCH_PERF_VARIANT=${label}_ar_text" || true

  run_logged "${label}_mtp_d3_text" \
    "${base_env[@]}" \
    "BENCH_MODEL=$model" \
    "BENCH_PERF=1" \
    "BENCH_PERF_NATIVE_MTP_DEPTH=3" \
    "BENCH_PERF_VARIANT=${label}_mtp_d3_text" || true

  cache_dir="$ARTIFACT_DIR/cache/$label"
  rm -rf "$cache_dir"
  mkdir -p "$cache_dir"
  run_logged "${label}_mtp_d3_cache_repeat" \
    "${base_env[@]}" \
    "BENCH_MODEL=$model" \
    "BENCH_PERF=1" \
    "BENCH_PERF_NATIVE_MTP_DEPTH=3" \
    "BENCH_PERF_CACHE_COORDINATOR=1" \
    "BENCH_PERF_CACHE_HYBRID=1" \
    "BENCH_PERF_CACHE_DISK=1" \
    "BENCH_PERF_CACHE_SSM_REDERIVE=1" \
    "BENCH_PERF_CACHE_DIR=$cache_dir" \
    "BENCH_PERF_CACHE_DISK_MAX_GB=2" \
    "BENCH_PERF_RUNS=2" \
    "BENCH_PERF_VARIANT=${label}_mtp_d3_cache_repeat" || true

  prod_cache="$ARTIFACT_DIR/prod-cache/$label"
  rm -rf "$prod_cache"
  mkdir -p "$prod_cache"
  run_logged "${label}_mtp_d3_multiturn_reasoning" \
    "BENCH_MODEL=$model" \
    "BENCH_PROD=1" \
    "BENCH_PROD_NATIVE_MTP_DEPTH=3" \
    "BENCH_PROD_COORD=1" \
    "BENCH_PROD_CACHE_HYBRID=1" \
    "BENCH_PROD_CACHE_DIR=$prod_cache" \
    "BENCH_MAX_TOKENS=128" || true

  run_logged "${label}_vl_chat_cache_ar" \
    "BENCH_MODEL=$model" \
    "BENCH_VL_CHAT_CACHE=1" \
    "BENCH_MAX_TOKENS=96" || true

  run_logged "${label}_vl_chat_cache_mtp_d3" \
    "BENCH_MODEL=$model" \
    "BENCH_VL_CHAT_CACHE=1" \
    "BENCH_VL_NATIVE_MTP_DEPTH=3" \
    "BENCH_MAX_TOKENS=96" || true
done

echo "summary: $SUMMARY"
