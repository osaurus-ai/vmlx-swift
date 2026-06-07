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

Clean-main mmap DOT primitive histogram:

```sh
BENCH_MODEL=/Users/eric/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L \
BENCH_PERF=1 \
BENCH_PERF_VARIANT=nemotron_mmap_graph_dot \
BENCH_MAX_TOKENS=4 \
BENCH_PERF_WARMUP=0 \
BENCH_PERF_RUNS=1 \
BENCH_PERF_USE_GENERATION_CONFIG=1 \
BENCH_PERF_SEED=42 \
BENCH_PERF_MMAP=1 \
BENCH_GRAPH_STATS=1 \
VMLINUX_GRAPH_DOT_PATH=/tmp/vmlx-nemotron-mmap-dot-20260606-051930.dot \
.build/debug/RunBench
```

Artifact: `/tmp/vmlx-nemotron-mmap-dot-20260606-051930.log`

Result:

- `commit=ab207fa`
- `tokps_median=5.0`
- `peak_footprint_mib=1366`
- `graphNodes=5711`
- `asType=1152`
- `samplingSource=bundle-defaults`
- coherent visible text, no loop, no parser marker leak

Top DOT primitive counts:

| Primitive | Count |
|---|---:|
| `AsType` | 1152 |
| `Broadcast` | 672 |
| `Reshape` | 626 |
| `Multiply` | 384 |
| `Add` | 348 |
| `Transpose` | 241 |
| `Flatten` | 241 |
| `CustomKernel` | 240 |
| `Matmul` | 193 |
| `QuantizedMatmul` | 192 |
| `RMSNorm` | 157 |
| `Sigmoid` | 144 |
| `Maximum` | 144 |
| `Sum` | 96 |
| `Convolution` | 48 |
| `ArgPartition` | 48 |
| `ScaledDotProductAttention` | 12 |

The histogram matches the 48-Mamba / 12-attention topology. The remaining mmap
speed gap is dominated by Mamba and routed-MoE graph shape, not by
generation-config, chat-template, tool parser, or reasoning parser behavior.

Low-footprint mmap JPREG row from the same worktree:

Artifact: `/tmp/vmlx-nemotron-scored-weighted-jpreg-20260606-033442.log`

Result:

- The saved JPREG row is a low-footprint mmap row, not an enabled JangPress
  advisory row.
- `JangPress: enabled=false`
- `RouterAdvice: enabled=false`
- mmap tracked Metal buffers: `94.4 GB`
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
- Do not describe the low-footprint mmap path as `8-10 tok/s`.
  Current mmap rows are coherent and cache-correct, and the latest default
  auto-BF16 mmap row is `5.3 tok/s`, but that is still below the `8-10 tok/s`
  target.

Fixed/proven:

- The perf harness now distinguishes the resident and mmap load paths.
- Current Swift resident decode confirms the documented `8 tok/s` class:
  `8.1 tok/s` with bundle generation defaults.
- Current Swift low-footprint mmap decode is coherent and stays around
  `1.35-2.1 GB` footprint in the perf rows. The latest default auto-BF16 mmap
  row is `5.3 tok/s` with `1926 MiB` peak physical footprint. The saved JPREG
  row stays low-footprint too, but its own artifact proves JangPress advisory
  state was disabled.
- Hybrid SSM disk-backed prefix cache hits are proven for exact replay and
  growing chat, including SSM companion-state hits and salt isolation.
- The attempted scored-kernel optimization was proven slower and removed from
  the default path.
- Loader dtype policy now preserves JANGTQ `tq_packed` / `tq_norms` raw while
  allowing non-TQ tensors to promote out of fp16 AsType-heavy decode. This is
  now automatic for mmap-loaded native Nemotron-H JANGTQ bundles only; other
  native JANGTQ families remain env opt-in through
  `VMLINUX_JANGTQ_BF16_MMAP` / `MLX_JANGTQ_BF16_MMAP`.
- Role-level `mxtq_bits` now accepts both Nemotron Ultra Mamba projection
  spellings: `mamba_proj` and `mamba_projection`. Focused coverage proves the
  longer spelling still applies the 8-bit affine override to
  `backbone.layers.*.mixer.{in,out}_proj` instead of falling through to a
  stale global quantization default.

Still not complete:

- The release-friendly low-footprint mmap path is `5.3 tok/s`, not
  `8-10 tok/s`.
- The saved JPREG artifact is not proof that enabled JangPress router/expert
  advice is speed-neutral or production-ready; it reports
  `JangPress: enabled=false` and `RouterAdvice: enabled=false`.
- The latest auto-BF16 mmap graph-stats probe shows `4799` decode graph nodes
  and `480` AsType nodes; closing the remaining speed gap still needs a real
  graph/kernel or resident-compute/reclaim improvement, not a template,
  sampler, or parser workaround.
- The DOT histogram confirms the mmap graph is dominated by the expected
  Nemotron-H topology: 48 recurrent Mamba layers plus 12 attention layers and
  routed-MoE work. The next optimization target is Mamba / routed-MoE graph
  reduction, not Osaurus wiring or generation defaults.
- The resident `8.1 tok/s` row uses about `100 GB` physical footprint.
- Thinking-on parser behavior is still partial in the short JPREG row because
  the model emitted reasoning but no visible answer within the token budget.
- Live prompt-boundary SSM rederive was not triggered in these cache rows:
  `reDerives=0`. The proven path is disk-backed SSM companion restore/hit.
  Detached async SSM rederive is intentionally not a production path.
- The vMLX rows above are harness rows. A separate Osaurus no-sign app/API pass
  was also run against the same model id and current vMLX pin:
  `/tmp/osaurus-bbd5d5ce-nemotron-ultra-tool-cache-warm-20260606-071921`.
  That warm relaunch row passed required-tool parsing, tool-history replay, no
  reasoning/tool marker leak, disk L2 hits, and SSM companion-state hits on the
  low-footprint mmap app path.
- The selective non-TQ bf16 loader policy is now source/test-proven and
  live-proven for Nemotron-H JANGTQ mmap. It improves the mmap graph shape and
  default token/s, but it does not complete the low-footprint speed target.

## Follow-Up Trace - 2026-06-06 08:40 PDT

Added a source-level Ultra metadata guard for a real alias gap:

- `jang_config.json["mxtq_bits"]` may stamp Mamba projection roles as either
  `mamba_proj` or `mamba_projection`.
- The shape walker previously recognized only `mamba_proj` for
  `backbone.layers.*.mixer.in_proj` / `out_proj`.
- The loader now accepts both aliases without changing sampler, parser,
  generation defaults, router precision, or quantized matmul math.

Focused verification:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift test \
  --filter 'MTPRuntimeFocusedTests/(jangConfigParsesRoleLevelMXTQMetadata|shapeWalkHonorsRoleLevelMXTQMetadata)' \
  --jobs 1 --no-parallel
```

Artifact:
`/tmp/vmlx-nemotron-mamba-projection-alias-focused-20260606-083927.log`

Result: passed, 2 tests.

## Follow-Up Trace - 2026-06-06 08:45 PDT

Reviewed the Swift runtime against the Python JANG loader patch in
`jang_tools/load_jangtq.py`.

Rejected shortcuts:

- Do not remove or demote the fp32 MoE router sigmoid cast. The source contract
  and `NemotronGroupExpertSelectFP32SigmoidTests` intentionally pin
  `sigmoid(gates.asType(.float32))` because bf16 router sigmoid can change
  expert selection.
- Do not change `JANGTQKernels.hadamardRotate` to return half from the current
  wrapper. Swift `gatherTQ` / `fusedGateUpSwiGLU` currently accept fp32 rotated
  inputs, and `JANGTQKernelsTests` pins the fp32 shape/dtype contract. Python's
  MPP/NAX half-rotated path is a different kernel family; copying only the
  dtype cast into the current Swift gather path would be an unproven kernel
  change.
- Do not re-enable Nemotron Ultra active streaming by default. The clean-main
  source keeps it explicit-only because prior live rows were slower than the
  ordinary JANGTQ path.

Current root-cause target remains the same:

- The mmap row is coherent and cache-correct, but the DOT histogram still shows
  `5711` graph nodes and `1152` `AsType` nodes. The next useful runtime work is
  a real Mamba/projection or JANGTQ kernel-family cleanup that preserves the
  fp32 router-selection floor and the existing JANGTQ math contract.

## Follow-Up Trace - 2026-06-06 09:06 PDT

The saved DOT histogram was rechecked against the local Ultra bundle headers.
The high `AsType` count is not evidence that Osaurus, the chat template, or the
JANGTQ routed-expert path is misrouting the model:

- DOT primitive count: `QuantizedMatmul=192`, `AsType=1152`.
- `192` quantized matmuls matches the expected affine-8 path:
  - 96 Mamba projections: 48 `mixer.in_proj` + 48 `mixer.out_proj`.
  - 96 shared-expert projections: 48 shared `up_proj` + 48 shared `down_proj`.
- The local bundle headers for representative affine-8 tensors are consistent:
  - `backbone.layers.0.mixer.in_proj.weight`: `U32`
  - `backbone.layers.0.mixer.in_proj.scales` / `.biases`: `F16`
  - `backbone.layers.0.mixer.out_proj.weight`: `U32`
  - `backbone.layers.1.mixer.shared_experts.{up,down}_proj.weight`: `U32`
  - shared/Mamba affine scales and biases: `F16`
- Precision-critical control-plane tensors remain unquantized as intended:
  - `backbone.layers.1.mixer.gate.weight`: `F32`
  - `backbone.layers.1.mixer.fc1_latent_proj.weight`: `BF16`
  - `backbone.layers.1.mixer.fc2_latent_proj.weight`: `BF16`

Interpretation:

- Do not "fix" this by demoting router/latent/control tensors or inventing
  generation defaults.
- Do not treat `QuantizedMatmul=192` as an accidental routed-expert fallback;
  it is the expected Mamba/shared-expert affine-8 surface.
- The remaining low-footprint speed gap is now narrowed to generic affine
  quantized-matmul graph overhead / mmap residency behavior, or a future
  resident-compute/reclaim mechanism that can keep Activity Monitor footprint
  low without per-token mmap pressure.

## Follow-Up Trace - 2026-06-06 09:56 PDT

Ran the JANG-side no-load Ultra proof tools against the saved log bundle in
`/Users/eric/jang/docs/runtime/logs` so this vMLX note stays aligned with the
source-model handoff before any new heavy model run:

```sh
PYTHONPATH=/Users/eric/jang/jang-tools \
  /Users/eric/jang/jang-tools/.venv/bin/python \
  /Users/eric/jang/jang-tools/examples/nemotron_ultra/runtime_status_report.py \
  --log-dir /Users/eric/jang/docs/runtime/logs

PYTHONPATH=/Users/eric/jang/jang-tools \
  /Users/eric/jang/jang-tools/.venv/bin/python \
  /Users/eric/jang/jang-tools/examples/nemotron_ultra/speed_experiment_plan.py \
  --log-dir /Users/eric/jang/docs/runtime/logs \
  --out /tmp/nemotron-ultra-speed-experiment-plan-current.md

PYTHONPATH=/Users/eric/jang/jang-tools \
  /Users/eric/jang/jang-tools/.venv/bin/python \
  /Users/eric/jang/jang-tools/examples/nemotron_ultra/runtime_speed_gate.py \
  --log-dir /Users/eric/jang/docs/runtime/logs \
  --out /tmp/nemotron-ultra-runtime-speed-gate-current.md

PYTHONPATH=/Users/eric/jang/jang-tools \
  /Users/eric/jang/jang-tools/.venv/bin/python \
  /Users/eric/jang/jang-tools/examples/nemotron_ultra/validate_runtime_log_bundle.py \
  --log-dir /Users/eric/jang/docs/runtime/logs
```

Results:

- Log bundle validation: `FIXED`; all required saved logs are present.
- Runtime speed gate: `PARTIAL`.
- Best saved live speed: `8.335 tok/s`, which clears the resident
  `8 tok/s` floor.
- Manual synchronized decode: `143.237 ms/token`, implied `6.981 tok/s`.
- MoE remains a major bucket: `65.773 ms` across 48 layers.
- Mamba remains a major bucket: `64.157 ms` across 48 layers.
- Attention and final head are not the first bottleneck:
  attention `8.990 ms`, norm/lm_head `4.317 ms`.
- Coherence remains partial in the JANG saved rows because visible
  `</think>` leakage, repeated n-gram fractions, and one no-EOS row are still
  recorded there. The Osaurus warm app/API row is separate and did not show
  tool/reasoning marker leakage in visible output.

Ranked next experiments from JANG's saved-log planner:

1. MoE routed/shared scheduling or fused decode kernel.
2. Mamba fused decode kernel / lower-overhead projection-state path.
3. Joint MoE+Mamba scheduling path.
4. Ahead-of-time warmup for startup/TTFT predictability, not steady tok/s.

Negative controls confirmed by the planner:

- Do not chase attention first.
- Do not dequantize the 8-bit Mamba/shared projections; current probes say
  quantized affine is faster.
- Do not lower router top-k as the main fix; top-k 8 did not materially
  improve decode.
- Do not replace the normal generation loop with a manual Python-style argmax
  loop.
- Do not hide coherence/parser issues with prompt suffixes, forced tags, or
  sampler tricks.

Source interpretation:

- Swift active-expert streaming for Nemotron Ultra remains explicit-only by
  design. `JANGTQStreamingExperts.shouldAutoEnableNemotronUltra` returns
  `false` unless the operator explicitly sets the streaming env override,
  because prior live rows showed the stacked/offset backend could be slower
  than the ordinary JANGTQ path.
- The next aligned runtime work is not Osaurus wiring, generation defaults,
  reasoning parser, or tool parser. It is a real MoE/Mamba graph or kernel
  reduction that preserves the existing JANGTQ math contract and the fp32
  router-selection floor.

## Follow-Up Trace - 2026-06-06 13:45 PDT

Aligned Osaurus production load policy with the proven mmap harness boundary:

- `LoadConfiguration.default` already uses mmap safetensors and 70% memory
  caps while keeping MLXPress/JangPress disabled unless explicitly requested.
- The saved low-footprint vMLX rows above use that disabled-cold-tier mmap path.
  After the auto-BF16 default, the latest mmap row decodes at `5.3 tok/s`.
- The Osaurus app proof used the `osaurusProduction` preset, which previously
  aliased `experimentalJangPressAuto`. On the 98 GB Nemotron Ultra routed
  bundle, that can auto-enable the experimental cold-tier because the bundle is
  larger than half of physical RAM.
- Since the current enabled-cold-tier path is not speed-proven for Nemotron
  Ultra and prior active-streaming rows were slower, `osaurusProduction` now
  aliases `LoadConfiguration.default`.
- `experimentalJangPressAuto` remains available for explicit host/settings
  opt-in after a bundle family has live proof.

This change does not alter sampler defaults, prompt templates, reasoning/tool
parsers, quantized matmul math, JANGTQ kernels, or SSM cache semantics. It keeps
the Osaurus default on the proven mmap + memory-cap behavior instead of
silently selecting an experimental cold-tier path for large routed bundles.

## Follow-Up Trace - 2026-06-06 14:52 PDT

Added a narrow Nemotron-H decode-only depthwise-conv fast path:

- Scope: Mamba `conv1d` only when `seqLen == 1`, the rolling Mamba conv state is
  available or initialized, and the bundle has a conv bias. Prefill and no-bias
  variants still use generic `Conv1d`.
- Math contract: the kernel computes the same depthwise convolution over
  `[previous_state, current_token]`, applies the same SiLU, and emits the next
  rolling conv state. It does not change SSM recurrence, MoE routing, JANGTQ
  kernels, generation defaults, templates, reasoning parsers, or tool parsers.
- Diagnostic opt-out:
  `VMLINUX_DISABLE_NEMOTRON_MAMBA_CONV_FASTPATH=1`.

Focused validation:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift test --filter NemotronHJANGTQDispatchFocusedTests \
  --jobs 1 --no-parallel
```

Artifact:
`/tmp/vmlx-nemotron-depthwise-conv-focused-rerun2-20260606-144944.log`

Result: passed, 11 tests. The new numerical parity test compares the custom
decode kernel against generic `conv1d + bias + SiLU` and verifies the emitted
rolling state.

Fresh mmap speed row:

- Artifact:
  `/tmp/vmlx-nemotron-mmap-depthwise-conv-fastpath-20260606-145009.log`
- Result: `tokps_median=4.1`, `peak_footprint_mib=1360`, bundle generation
  defaults, coherent visible output, no reasoning/tool leaks.

Disabled fast-path control:

- Artifact:
  `/tmp/vmlx-nemotron-mmap-depthwise-conv-disabled-control-20260606-145034.log`
- Result: `tokps_median=4.0`, `peak_footprint_mib=1362`, same visible output.

Resident control:

- Artifact:
  `/tmp/vmlx-nemotron-resident-depthwise-conv-fastpath-20260606-145051.log`
- Result: `tokps_median=8.8`, `peak_footprint_mib=101573`, coherent visible
  output, no reasoning/tool leaks.

Graph-stats row:

- Artifact:
  `/tmp/vmlx-nemotron-mmap-depthwise-conv-graphstats-20260606-145155.log`
- DOT:
  `/tmp/vmlx-nemotron-mmap-depthwise-conv-dot-20260606-145155.dot`
- Result: `decodeNodes=5375`, `asType=1056`, `tokps_median=5.0` for the
  4-token graph probe.

Interpretation:

- This is a real graph reduction versus the prior clean-main graph row
  (`5711` decode nodes, `1152` `AsType` nodes), and it preserves the low
  footprint and coherency envelope.
- It is not the final low-footprint speed fix. The fresh 16-token mmap row is
  still about `4 tok/s`, not `8-10 tok/s`.
- The remaining low-footprint gap is still MoE/Mamba graph/kernel work or a
  resident-compute/reclaim mechanism, not Osaurus wiring or parser/default
  behavior.

## Follow-Up Trace - 2026-06-06 15:55 PDT

Merged the scoped auto-BF16 mmap load policy for Nemotron-H JANGTQ:

- vMLX PR: `https://github.com/osaurus-ai/vmlx-swift/pull/28`
- vMLX main revision: `9717a4562cd52a3b91156fb627389fdbc1911013`
- Commit: `920e3e3 Auto BF16 mmap conversion for Nemotron JANGTQ`

Change:

- Mmap-loaded native Nemotron-H JANGTQ bundles automatically promote non-TQ
  tensors out of fp16 AsType-heavy decode.
- `.tq_packed` and `.tq_norms` stay raw.
- Other native JANGTQ families remain opt-in through
  `VMLINUX_JANGTQ_BF16_MMAP` / `MLX_JANGTQ_BF16_MMAP`.
- No sampler, prompt, template, parser, or forced-output behavior changed.

Focused checks:

- `/tmp/vmlx-loadconfig-nemotron-auto-bf16-20260606-154840.log`
  - `LoadConfigurationTests/jangtqLoadDoesNotSkipWholeModelBFloat16Conversion`
    passed.
- `/tmp/vmlx-nemotron-dispatch-focused-auto-bf16-20260606-155138.log`
  - `NemotronHJANGTQDispatchFocusedTests`, 11 tests passed.

Live low-footprint proof without BF16 env flags:

- Artifact:
  `/tmp/vmlx-nemotron-mmap-auto-bf16-64tok-sustained-20260606-155110.log`
- Result:
  - `tokps_median=5.3`
  - `peak_footprint_mib=1926`
  - `decodeNodes=4799`
  - `asType=480`
  - bundle generation defaults:
    `temp=1.00 topP=0.95 topK=0 rep=nil`
  - coherent visible text, no loop, no reasoning/tool leaks.

Control:

- `/tmp/vmlx-nemotron-mmap-streaming-explicit-16tok-control-20260606-154539.log`
  showed the explicit MLXPress streaming-expert path remains diagnostic-only:
  `tokps_median=0.3`, coherent but far too slow.

Interpretation:

- The low-footprint default is improved and still under 2 GB physical footprint
  in the 64-token row.
- The low-footprint speed gate remains PARTIAL: `5.3 tok/s` is not the
  requested `8-10 tok/s`.
- The next real speed lane is regular stacked Nemotron MoE/Mamba graph and
  kernel work, not JangPress streaming, top-k reduction, prompt changes, or
  sampler changes.

## Follow-Up Trace - 2026-06-06 19:35 PDT

Rechecked current `vmlx-origin/main` after the scoped Nemotron merges and
rejected two more non-fixes:

- Compiled SwitchMLP reduced the graph only under an unsafe compile lane and
  did not move default decode enough:
  - control `/tmp/vmlx-nemotron-ultra-compiled-switch-control-20260606-185207.log`:
    `6.9 tok/s`
  - candidate `/tmp/vmlx-nemotron-ultra-compiled-switch-candidate-20260606-185231.log`:
    `6.9 tok/s`
  - unsafe diagnostic `/tmp/vmlx-nemotron-ultra-compiled-switch-unsafe-20260606-185308.log`:
    `7.1 tok/s`
- Explicit JangPress/streaming-expert mode is still rejected for the production
  default:
  `/tmp/vmlx-nemotron-ultra-explicit-streaming-diagnostic-20260606-185920.log`
  reports about `0.7 tok/s`.
- A scored-offset down-projection candidate was coherent but slower and was
  reverted:
  `/tmp/vmlx-nemotron-ultra-scored-offset-capital-20260606-191817.log`
  reports `4.4 tok/s`.

Fresh current-main low-footprint rows after the revert:

- `/tmp/vmlx-nemotron-ultra-reverted-tq33-capital-20260606-192246.log`
  - `tokps_median=7.0`
  - `tail_tokps_est=9.9`
  - `first_decode_ms=761`
  - `peak_footprint_mib=1932`
  - bundle defaults, coherent visible answer, no parser leak, no loop
- `/tmp/vmlx-nemotron-ultra-reverted-graphstats-capital-20260606-192305.log`
  - `decodeNodes=4799`
  - `asType=480`
  - `tokps_median=6.6`
  - `tail_tokps_est=19.8`
  - visible answer: `Tokyo is the capital of Japan.`
- `/tmp/vmlx-nemotron-ultra-warmup-tq33-capital-20260606-192411.log`
  - warmup did not remove the first-decode cost:
    `tokps_median=7.0`, `tail_tokps_est=9.9`, `first_decode_ms=765`
- `/tmp/vmlx-nemotron-ultra-batch-tq33-capital-20260606-192442.log`
  - BatchEngine path matches the iterator path:
    `tokps_median=7.0`, `tail_tokps_est=9.8`, `first_decode_ms=759`

Python comparison from the same local model:

- `/tmp/nemotron-ultra-python-live-short-20260606-190521.json`
  - math: `8.808 tok/s` excluding first decode
  - capital: `8.684 tok/s` excluding first decode

Interpretation:

- Swift low-footprint sustained/tail decode is now in the same `8-10 tok/s`
  class as the Python documentation when compared on the same exclude-first
  basis.
- Full-run short prompts remain around `7 tok/s` because the first decode step
  costs about `760 ms`.
- Osaurus path selection is not the speed gap; BatchEngine and direct iterator
  rows are equivalent.
- The remaining work is model-forward dispatch in Nemotron-H MoE/Mamba, not
  sampler, template, tool parser, reasoning parser, top-k reduction, attention,
  or JangPress streaming.

Added current diagnostic instrumentation:

- `RunBench` now prints `first_decode_ms` and `tail_tokps_est` in `PERF_RUN`
  rows so the resident/mmap/Python comparison is not hidden by the first-decode
  cost.
- `VMLINUX_NEMOTRON_LAYER_PROFILE=1` enables an explicit diagnostic profiler
  for Nemotron-H block and MoE subcomponent timings. The flag is default-off and
  inserts synchronization only when enabled.

Diagnostic profiler rows:

- `/tmp/vmlx-nemotron-ultra-layer-profile-20260606-192922.log`
  shows decode-token work dominated by Mamba and MoE mixers, with attention
  around `9-10 ms`.
- `/tmp/vmlx-nemotron-ultra-moe-subprofile-20260606-193036.log`
  shows MoE work split mainly across `moe.switch_mlp`, `moe.shared`, gate, and
  latent projections.

Focused verification:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift test --filter NemotronHJANGTQDispatchFocusedTests \
  --jobs 1 --no-parallel
```

Result: passed, 11 tests.
