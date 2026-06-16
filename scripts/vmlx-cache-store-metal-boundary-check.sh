#!/usr/bin/env bash
set -euo pipefail

failures=0

require_contains() {
  local file="$1"
  local needle="$2"
  local label="$3"
  if ! grep -Fq "$needle" "$file"; then
    printf 'FAIL: %s missing %s\n' "$file" "$label" >&2
    failures=$((failures + 1))
  fi
}

require_contains \
  "Libraries/MLXLMCommon/Cache/DiskCache.swift" \
  "NSRecursiveLock" \
  "recursive process-wide MLX disk cache IO lock"
require_contains \
  "Libraries/MLXLMCommon/Cache/DiskCache.swift" \
  "func withMLXDiskCacheIOLock" \
  "reentrant cache IO lock helper"

for file in \
  "Libraries/MLXLMCommon/Evaluate.swift" \
  "Libraries/MLXLMCommon/SpecDec/NativeMTPTokenIterator.swift" \
  "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift"
do
  require_contains "$file" "withMLXDiskCacheIOLock {" \
    "outer cache-store MLX IO serialization"
done

require_contains \
  "RunBench/Bench.swift" \
  "BENCH_DISK_CACHE_STRESS" \
  "cache-store MLX safetensors stress entry point"
require_contains \
  "RunBench/Bench.swift" \
  "runDiskCacheMetalStress" \
  "cache-store MLX safetensors stress implementation"

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: post-generation cache-store MLX IO boundaries and stress harness are present.\n'
