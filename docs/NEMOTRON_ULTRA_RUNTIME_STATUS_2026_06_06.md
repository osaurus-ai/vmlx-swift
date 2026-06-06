# Nemotron Ultra Runtime Status - 2026-06-06

## Scope

Model: `/Users/eric/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L`

This note records the current vMLX Swift status for the Ultra JANGTQ_1L
runtime after rechecking the Python-doc `8 tok/s` claim against live Swift
resident and mmap paths.

## Code Changes

- Fixed `BENCH_PERF_MMAP=1` in `RunBench` so the perf harness explicitly uses
  `LoadConfiguration(useMmapSafetensors: true)`.
- Preserved the original resident load call when `BENCH_PERF_MMAP=0`.
  Passing `LoadConfiguration(useMmapSafetensors: false)` is not equivalent to
  the original resident path and regressed decode to about `0.6 tok/s`.
- Added source coverage that keeps the rejected stacked scored down-projection
  experiment out of the default Nemotron-H JANGTQ path.

No sampler, prompt, generation-config, reasoning parser, or tool parser
behavior was changed.

## Rejected Experiment

The attempted stacked scored down-projection kernel was removed from the patch.
It looked plausible because it avoided materializing `(tokens, K, hidden)` for
the final weighted reduction, but live resident rows proved it was slower:

- `/tmp/vmlx-nemotron-compiled-weighted-perf-resident-20260606-034324.log`
  - `tokps_median=3.0`
  - `peak_footprint_mib=102031`
- `/tmp/vmlx-nemotron-scored-weighted-perf-resident-rebuilt-20260606-034903.log`
  - `tokps_median=0.6`
  - `peak_footprint_mib=102019`

The default runtime keeps the previously proven `weightedDecode` shape:

```swift
let y = callAsFunction(x, indices)
return (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
```

## Validation

Focused source/compile coverage:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift test --filter NemotronHJANGTQDispatchFocusedTests \
  --jobs 1 --no-parallel
```

Artifact: `/tmp/vmlx-nemotron-restored-weighted-focused-20260606-035551.log`

Result: passed, 10 tests.

RunBench build after the harness fix:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift build --product RunBench --jobs 1
```

Artifact:
`/tmp/vmlx-nemotron-runbench-rebuild-resident-load-fix-20260606-040134.log`

Result: passed.

## Live Rows

Resident Swift speed row:

```sh
BENCH_MODEL=/Users/eric/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L \
BENCH_PERF=1 \
BENCH_PERF_VARIANT=nemotron_resident_original_load \
BENCH_MAX_TOKENS=32 \
BENCH_PERF_WARMUP=1 \
BENCH_PERF_RUNS=1 \
BENCH_PERF_USE_GENERATION_CONFIG=1 \
BENCH_PERF_SEED=42 \
BENCH_PERF_MMAP=0 \
.build/debug/RunBench
```

Artifact: `/tmp/vmlx-nemotron-resident-original-load-20260606-040225.log`

Result:

- `tokps_median=8.1`
- `peak_footprint_mib=102736`
- `samplingSource=bundle-defaults`
- `temp=1.00 topP=0.95 topK=0 rep=nil`
- coherent visible text, no loop, no parser marker leak

Explicit mmap Swift row:

```sh
BENCH_MODEL=/Users/eric/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L \
BENCH_PERF=1 \
BENCH_PERF_VARIANT=nemotron_mmap_explicit_load \
BENCH_MAX_TOKENS=16 \
BENCH_PERF_WARMUP=0 \
BENCH_PERF_RUNS=1 \
BENCH_PERF_USE_GENERATION_CONFIG=1 \
BENCH_PERF_SEED=42 \
BENCH_PERF_MMAP=1 \
.build/debug/RunBench
```

Artifact: `/tmp/vmlx-nemotron-mmap-explicit-load-20260606-040324.log`

Result:

- `tokps_median=3.9`
- `peak_footprint_mib=1353`
- `samplingSource=bundle-defaults`
- `temp=1.00 topP=0.95 topK=0 rep=nil`
- coherent visible text, no loop, no parser marker leak

Low-footprint JPREG row from the same worktree:

Artifact: `/tmp/vmlx-nemotron-scored-weighted-jpreg-20260606-033442.log`

Result:

- post-load footprint `0.3 GB`
- post-quiesce footprint `4.2 GB`
- `avgApiDecode=3.8 tok/s`
- three-turn text row coherent, `looping=no`
- `TQ disk round-trip: PASS`
- thinking-on probe remained partial: `offReasoning=0c onReasoning=812c`

## Current Verdict

PARTIAL.

Fixed/proven:

- The perf harness now distinguishes the resident and mmap load paths.
- Current Swift resident decode confirms the documented `8 tok/s` class:
  `8.1 tok/s` with bundle generation defaults.
- Current Swift low-footprint mmap decode is coherent and stays around
  `1.35 GB` footprint.
- The attempted scored-kernel optimization was proven slower and removed from
  the default path.

Still not complete:

- The release-friendly low-footprint mmap path is `3.8-3.9 tok/s`, not
  `8-10 tok/s`.
- The resident `8.1 tok/s` row uses about `100 GB` physical footprint.
- Thinking-on parser behavior is still partial in the short JPREG row because
  the model emitted reasoning but no visible answer within the token budget.
- The current rows are text-only. Full Osaurus chat, tool-call, and hybrid SSM
  companion prefix-cache proof still need a separate live app/API pass.
