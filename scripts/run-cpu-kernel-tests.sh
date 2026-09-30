#!/usr/bin/env bash
# Copyright © 2026 Osaurus AI. All rights reserved.
# SPDX-License-Identifier: MIT
#
# run-cpu-kernel-tests.sh: build the Highway test build and run VMLXCPUKernelTests twice, with
# MLX_CPU_THREADS=1 and with the default pool, as CI does. Linux only.
#
#   scripts/run-cpu-kernel-tests.sh [--expect "<Highway targets>"] [--sde <chip>] [--skip-build]
#                                    [--filter <swift-testing filter>] [--once] [--stall <seconds>]
#
# The test build (VMLX_HWY_ALL_TARGETS=1, release) compiles every attainable Highway target, EMU128
# included. VMLX_EXPECT_HWY_TARGETS, the targets this host must execute, is --expect's value, or
# else Highway's own detection of this CPU (scripts/hwy-supported-targets.cc, compiled with $CXX, by
# default g++) intersected with the targets Highway 1.4.0 attains on this architecture, less the SVE
# family that Package.swift disables on arm64, plus EMU128. It never comes from what the build
# contains.
# --sde runs the probe and the tests under Intel SDE, emulating <chip> (for example spr).
# --filter narrows the tests (default: the whole target); --once runs only the default pool, for
# emulated runs, which are slow. A run whose log stops growing for --stall seconds (default 1200) is
# killed with everything it started: a hung kernel must not hold a runner forever.
# VMLX_KERNEL_TESTS_SCRATCH moves the build (default .build/hwy-all), for a mutated build beside it.
#
# Exits 0 when every configuration passed. Otherwise with the first failing configuration's status:
# 124 when it stalled, 69 when no test matched the filter, and else the test executable's own (1 for
# a failed test). Exits 2 for a usage error, a missing test executable or a missing probe compiler.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SCRATCH="${VMLX_KERNEL_TESTS_SCRATCH:-$ROOT/.build/hwy-all}"
EXPECT=""
SDE=""
BUILD=1
FILTER=VMLXCPUKernelTests
CONFIGURATIONS="1 default"
STALL=1200
usage() {
  echo "usage: $0 [--expect '<targets>'] [--sde <chip>] [--skip-build] [--filter <filter>] [--once] [--stall <s>]" >&2
  exit 2
}
while [ $# -gt 0 ]; do
  case "$1" in --expect | --sde | --filter | --stall) [ $# -ge 2 ] || usage ;; esac
  case "$1" in
    --expect) EXPECT=$2; shift 2 ;;
    --sde) SDE=$2; shift 2 ;;
    --skip-build) BUILD=0; shift ;;
    --filter) FILTER=$2; shift 2 ;;
    --once) CONFIGURATIONS=default; shift ;;
    --stall) STALL=$2; shift 2 ;;
    *) usage ;;
  esac
done
# The tests expect the int8 switch off at the start, and judge the polynomial bounds, which
# VMLX_ULP_BASELINE=1 would only measure.
unset MLX_CPU_QUANTIZED_INT8 VMLX_ULP_BASELINE
SWIFT_FLAGS=(--package-path "$ROOT" --scratch-path "$SCRATCH" -c release -Xswiftc -enable-testing)
export VMLX_HWY_ALL_TARGETS=1
if [ "$BUILD" = 1 ]; then
  swift build "${SWIFT_FLAGS[@]}" --build-tests
fi
# The test executable, run directly: `swift test` would start it in a process group of its own, out
# of watched's reach. Swift Build (SwiftPM 6.4's default) links one per test target, beside its .so;
# the native build system linked one per package. The newest, should a stale one remain.
BIN=$(swift build "${SWIFT_FLAGS[@]}" --show-bin-path)
TEST_BIN=$(find "$BIN" -maxdepth 1 -type f -perm -u+x \( -name 'VMLXCPUKernelTests-test-runner' -o -name '*PackageTests.xctest' \) -printf '%T@ %p\n' 2> /dev/null | sort -rn | head -1 | cut -d' ' -f2- || true)
[ -n "$TEST_BIN" ] || { echo "no test executable in $BIN" >&2; exit 2; }
echo "Tests: $TEST_BIN"
# The emulator's command prefix, empty without --sde.
SDE_RUN=()
if [ -n "$SDE" ]; then SDE_RUN=(sde64 "-$SDE" --); fi
# watched <log> <program...>: runs the program line-buffered into <log> under setsid, which makes it
# the leader of a new process group, and kills that group if the log stops growing for $STALL
# seconds, so that no hung child (SDE starts several) outlives the run. Returns the program's exit
# status, and sets STALLED to 1 after a kill.
watched() {
  local log=$1
  shift
  STALLED=0
  setsid stdbuf -oL -eL "$@" > "$log" 2>&1 &
  local pid=$! last=-1 idle=0 size rc=0
  while kill -0 "$pid" 2> /dev/null; do
    sleep 15
    size=$(stat -c %s "$log")
    if [ "$size" = "$last" ]; then idle=$((idle + 15)); else idle=0; last=$size; fi
    if [ "$idle" -ge "$STALL" ]; then
      echo "STALLED: no output for $STALL s" >> "$log"
      kill -9 -- "-$pid" 2> /dev/null || true
      STALLED=1
      break
    fi
  done
  wait "$pid" || rc=$?
  return "$rc"
}
echo "CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- || true)"
if [ -z "$EXPECT" ]; then
  case "$(uname -m)" in
    # Highway 1.4.0 marks AVX10_2 broken below Clang 23 (hwy/detect_targets.h), so Swift 6.4's
    # Clang 21 attains the rest. Add AVX10_2 when the toolchain's Clang reaches 23.
    x86_64) ATTAINABLE="AVX3_SPR AVX3_ZEN4 AVX3_DL AVX3 AVX2 SSE4 SSSE3 SSE2" ;;
    # No SVE: the build disables it (Package.swift) until the kernels are vector-length agnostic.
    aarch64) ATTAINABLE="NEON_BF16 NEON NEON_WITHOUT_AES" ;;
    *) ATTAINABLE="" ;; # no Highway kernels elsewhere
  esac
  if [ -n "$ATTAINABLE" ]; then
    H="$ROOT/Source/Cmlx/highway"
    PROBE_CXX=${CXX:-g++}
    command -v "$PROBE_CXX" > /dev/null || {
      echo "no C++ compiler for the target probe: $PROBE_CXX (install g++, or set CXX)" >&2
      exit 2
    }
    mkdir -p "$SCRATCH"
    "$PROBE_CXX" -std=c++17 -O1 -DHWY_DISABLE_PCLMUL_AES -I "$H" \
      "$ROOT/scripts/hwy-supported-targets.cc" "$H/hwy/targets.cc" "$H/hwy/abort.cc" \
      "$H/hwy/print.cc" "$H/hwy/per_target.cc" -o "$SCRATCH/hwy-supported-targets"
    SUPPORTED=" $("${SDE_RUN[@]}" "$SCRATCH/hwy-supported-targets") "
    EXPECT=""
    for t in $ATTAINABLE; do
      case "$SUPPORTED" in *" $t "*) EXPECT="$EXPECT$t " ;; esac
    done
    EXPECT="${EXPECT}EMU128"
  else
    EXPECT="none"
  fi
fi
export VMLX_EXPECT_HWY_TARGETS="$EXPECT"
echo "VMLX_EXPECT_HWY_TARGETS=$EXPECT"
status=0
for threads in $CONFIGURATIONS; do
  log="$SCRATCH/kernel-tests-$threads.log"
  if [ "$threads" = 1 ]; then export MLX_CPU_THREADS=1; else unset MLX_CPU_THREADS; fi
  rc=0
  watched "$log" "${SDE_RUN[@]}" "$TEST_BIN" --testing-library swift-testing --filter "$FILTER" || rc=$?
  echo "== MLX_CPU_THREADS=$threads: exit $rc, $(grep -E 'Test run with' "$log" | tail -1)"
  if [ "$rc" = 0 ] && grep -Eq 'Test run with [1-9][0-9]* tests? .*passed' "$log"; then continue; fi
  # Each recorded issue with the values swift-testing prints under it (↳), and any stall or crash.
  awk '/recorded an issue/ {show = 1; print; next} show && /^↳/ {print; next} {show = 0}
    /STALLED|[Ff]atal error|error:/ {print}' "$log" | head -40 || true
  if [ "$STALLED" = 1 ]; then
    rc=124
  elif [ "$rc" = 69 ]; then
    # The executable exits 69 when the filter matches no test, where `swift test` exits 0.
    echo "no test matches --filter '$FILTER'"
  elif [ "$rc" = 0 ]; then
    rc=1 # exit 0 without a passing summary line
  fi
  if [ "$status" = 0 ]; then status=$rc; fi
done
exit "$status"
