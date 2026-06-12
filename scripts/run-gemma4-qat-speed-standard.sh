#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNBENCH="$ROOT/.build/arm64-apple-macosx/release/RunBench"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/tmp/vmlx-gemma4-qat-speed-standard-$STAMP}"
MAX_TOKENS="${MAX_TOKENS:-128}"
RUNS="${RUNS:-3}"
WARMUP="${WARMUP:-1}"
PROMPT="${PROMPT:-Write one long paragraph describing ocean waves. Be verbose and detailed.}"

DEFAULT_MODELS=(
  "/Users/eric/models/OsaurusAI--gemma-4-E2B-it-qat-MXFP4"
  "/Users/eric/models/OsaurusAI--gemma-4-E2B-it-qat-JANG_4M"
)

MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=("${DEFAULT_MODELS[@]}")
fi

mkdir -p "$ARTIFACT_ROOT"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  swift build -c release --product RunBench --jobs "${SWIFT_JOBS:-4}"
fi

MLXPRESS_BUILD_CONFIGURATION=release "$ROOT/scripts/prepare-mlx-metal.sh" >/dev/null

if [[ ! -x "$RUNBENCH" ]]; then
  echo "missing RunBench executable: $RUNBENCH" >&2
  exit 1
fi

run_row() {
  local model="$1"
  local mode="$2"
  local name variant out err
  name="$(basename "$model")"
  variant="$name-$mode"
  out="$ARTIFACT_ROOT/$variant.out"
  err="$ARTIFACT_ROOT/$variant.err"

  if [[ ! -d "$model" ]]; then
    echo "missing model directory: $model" | tee "$ARTIFACT_ROOT/$variant.missing"
    return 1
  fi

  case "$mode" in
    deterministic)
      if ! BENCH_PERF=1 \
      BENCH_MODEL="$model" \
      BENCH_PERF_VARIANT="$variant" \
      BENCH_PERF_PROMPT="$PROMPT" \
      BENCH_MAX_TOKENS="$MAX_TOKENS" \
      BENCH_PERF_RUNS="$RUNS" \
      BENCH_PERF_WARMUP="$WARMUP" \
      BENCH_PERF_PATH=batch \
      BENCH_PERF_TEMP=0 \
      BENCH_PERF_TOP_P=1 \
      BENCH_PERF_TOP_K=0 \
      BENCH_PERF_MIN_P=0 \
      "$RUNBENCH" >"$out" 2>"$err"; then
        echo "FAILED $variant" | tee -a "$ARTIFACT_ROOT/SUMMARY.txt"
        tail -40 "$err" | tee -a "$ARTIFACT_ROOT/SUMMARY.txt"
        return 1
      fi
      ;;
    bundle-defaults)
      if ! BENCH_PERF=1 \
      BENCH_MODEL="$model" \
      BENCH_PERF_VARIANT="$variant" \
      BENCH_PERF_PROMPT="$PROMPT" \
      BENCH_MAX_TOKENS="$MAX_TOKENS" \
      BENCH_PERF_RUNS="$RUNS" \
      BENCH_PERF_WARMUP="$WARMUP" \
      BENCH_PERF_PATH=batch \
      BENCH_PERF_USE_GENERATION_CONFIG=1 \
      BENCH_PERF_SEED="${BENCH_PERF_SEED:-1234}" \
      "$RUNBENCH" >"$out" 2>"$err"; then
        echo "FAILED $variant" | tee -a "$ARTIFACT_ROOT/SUMMARY.txt"
        tail -40 "$err" | tee -a "$ARTIFACT_ROOT/SUMMARY.txt"
        return 1
      fi
      ;;
    *)
      echo "unknown mode: $mode" >&2
      return 2
      ;;
  esac

  if ! rg '^PERF( |_)|^  PERF_RUN' "$out" | tee -a "$ARTIFACT_ROOT/SUMMARY.txt"; then
    echo "FAILED $variant: no PERF lines emitted" | tee -a "$ARTIFACT_ROOT/SUMMARY.txt"
    tail -40 "$out" | tee -a "$ARTIFACT_ROOT/SUMMARY.txt"
    return 1
  fi
}

{
  echo "artifact_root=$ARTIFACT_ROOT"
  echo "max_tokens=$MAX_TOKENS"
  echo "runs=$RUNS"
  echo "warmup=$WARMUP"
  echo "prompt=$PROMPT"
  echo "runbench=$RUNBENCH"
  git -C "$ROOT" rev-parse HEAD | sed 's/^/vmlx_head=/'
} > "$ARTIFACT_ROOT/METADATA.txt"

: > "$ARTIFACT_ROOT/SUMMARY.txt"
for model in "${MODELS[@]}"; do
  run_row "$model" deterministic
  run_row "$model" bundle-defaults
done

echo "artifact_root=$ARTIFACT_ROOT"
