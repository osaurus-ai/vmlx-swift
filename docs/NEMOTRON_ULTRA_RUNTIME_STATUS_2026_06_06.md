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
- Added `BENCH_GROWING_MMAP=1` so the growing-chat cache harness can run the
  same low-footprint mmap load path as the perf harness.
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

RunBench build after the growing-cache mmap knob:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift build --product RunBench --jobs 1
```

Artifact:
`/tmp/vmlx-nemotron-runbench-rebuild-growing-mmap-20260606-040923.log`

Result: passed.

Clean-main cache topology source coverage:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift test --filter CacheCoordinatorTopologyFocusedTests \
  --jobs 1 --no-parallel
```

Artifact:
`/tmp/vmlx-nemotron-cache-topology-main-20260606-050752.log`

Result: passed, 31 tests across 5 suites.

Coverage includes hybrid companion-state requirements, partial companion-state
rejection, disk-tier longest-prefix restore, path-dependent disk-backed restore,
and reasoning/tool-choice/KV-policy cache-salt isolation.

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

Clean-main mmap graph-stats probe:

```sh
BENCH_MODEL=/Users/eric/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L \
BENCH_PERF=1 \
BENCH_PERF_VARIANT=nemotron_mmap_graphstats_main \
BENCH_MAX_TOKENS=4 \
BENCH_PERF_WARMUP=0 \
BENCH_PERF_RUNS=1 \
BENCH_PERF_USE_GENERATION_CONFIG=1 \
BENCH_PERF_SEED=42 \
BENCH_PERF_MMAP=1 \
BENCH_GRAPH_STATS=1 \
.build/debug/RunBench
```

Artifact: `/tmp/vmlx-nemotron-mmap-graphstats-main-20260606-051005.log`

Result:

- `commit=4ccceed`
- `tokps_median=4.4`
- `peak_footprint_mib=1366`
- `graphNodes=5711`
- `asType=1152`
- `samplingSource=bundle-defaults`
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

Hybrid SSM exact-replay cache row:

Artifact: `/tmp/vmlx-nemotron-mmap-cache-hybrid-ssm-20260606-040758.log`

Result:

- mmap load, bundle generation defaults
- `tokps_median=4.5`
- `peak_footprint_mib=2131`
- run 0 stores disk state: `disk{hits=0,misses=1,stores=1}`
- run 1 restores disk plus SSM companion state:
  `disk{hits=1,misses=2,stores=2}` and `ssm{hits=1,misses=0,reDerives=0}`
- `hybrid=true pagedIncompatible=true`, so Nemotron-H does not accept unsafe
  paged-only cache hits
- coherent visible text, no loop, no parser marker leak

Growing-chat post-answer cache row:

Artifact: `/tmp/vmlx-nemotron-growing-cache-mmap-20260606-041017.log`

Result:

- `Load mode: mmap`
- topology: `layers=60,kvLayers=12,mambaLayers=48,companion=ssm,restore=disk-backed`
- turn 1: `prompt=30 gen=5 finish=stop promptTime=1.420s`, visible
  `vmlx-cache-green`
- prompt-boundary salted probe hit: `matched=30/31`
- post-answer salted probe hit: `matched=35/36`
- nil-salt probes missed, proving salt isolation
- before turn 2: `disk{hits=2,misses=8,stores=2}` and
  `ssm{hits=2,misses=0,reDerives=0}`
- turn 2 growing prompt: `prompt=62 gen=5 finish=stop promptTime=1.215s`,
  visible `vmlx-cache-green`
- after turn 2: `disk{hits=4,misses=11,stores=4}` and
  `ssm{hits=4,misses=0,reDerives=0}`

## Current Verdict

PARTIAL.

Release-safe wording:

- Current Swift resident decode reaches the documented `8 tok/s` class on
  Nemotron Ultra JANGTQ_1L: `8.1 tok/s`, bundle generation defaults, coherent
  visible output, no loop, and no parser marker leak.
- Do not describe the low-footprint mmap/JangPress path as `8-10 tok/s`.
  Current mmap rows are coherent and cache-correct, but they are still in the
  `3.8-4.5 tok/s` class.

Fixed/proven:

- The perf harness now distinguishes the resident and mmap load paths.
- Current Swift resident decode confirms the documented `8 tok/s` class:
  `8.1 tok/s` with bundle generation defaults.
- Current Swift low-footprint mmap decode is coherent and stays around
  `1.35 GB` footprint.
- Hybrid SSM disk-backed prefix cache hits are proven for exact replay and
  growing chat, including SSM companion-state hits and salt isolation.
- The attempted scored-kernel optimization was proven slower and removed from
  the default path.

Still not complete:

- The release-friendly low-footprint mmap path is `3.8-3.9 tok/s`, not
  `8-10 tok/s`.
- A clean-main graph-stats probe still shows `5711` decode graph nodes and
  `1152` AsType nodes on the mmap path; closing the speed gap needs a real
  graph/kernel or resident-compute/reclaim improvement, not a template,
  sampler, or parser workaround.
- The resident `8.1 tok/s` row uses about `100 GB` physical footprint.
- Thinking-on parser behavior is still partial in the short JPREG row because
  the model emitted reasoning but no visible answer within the token budget.
- Live async SSM rederive was not triggered in these cache rows:
  `reDerives=0`. The proven path is disk-backed SSM companion restore/hit.
- The current rows are vMLX harness rows. Full Osaurus chat and tool-call proof
  still need a separate live app/API pass.
