# Qwen3.6 Native MTP Overnight Speed Loop - 2026-05-17

This note is the autonomous work prompt and execution contract for improving
native MTP speed in `vmlx-swift`. It is intentionally focused on runtime
performance and correctness, not UI policy wiring.

## Current Target

Required MXFP scope:

```text
/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP4-MTP
/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP8-MTP
/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP4-MTP
/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP
```

Only these MXFP variants are in scope for this overnight pass.

## 2026-05-17 Live Harness Update

Artifact directory:

```text
docs/local/qwen36-mtp-overnight/20260517T075020Z-sampler-vl/
```

Changes landed in the working tree for this pass:

- `RunBench` perf rows can now opt into bundle sampling defaults with
  `BENCH_PERF_USE_GENERATION_CONFIG=1`.
- Explicit `BENCH_PERF_TEMP`, `BENCH_PERF_TOP_P`, `BENCH_PERF_TOP_K`,
  `BENCH_PERF_MIN_P`, and `BENCH_PERF_REPETITION_PENALTY` still override the
  bundle values. This is for controlled measurement, not hidden model policy.
- `BENCH_PERF_SEED` now wires into `GenerateParameters.randomSeed` and is
  printed in `PERF_RUN` / `PERF` output so stochastic rows are reproducible.
- The focused source gate is
  `MTPRuntimeFocusedTests/runBenchPerfCanUseBundleGenerationDefaults`.

Live rows from this pass:

| Row | Result |
|---|---|
| `27b-mxfp8-d3-explicit-greedy-192.log` | Native MTP D3, explicit greedy, coherent `1..50`, `stop=stop`, no loop, `25.2 tok/s`. |
| `27b-mxfp8-ar-bundle-seed7.log` | AR bundle defaults, `seed=7`, `temp=1.0 top_p=0.95 top_k=20`, coherent blue-sky answer. |
| `27b-mxfp8-d3-bundle-seed7.log` | Native MTP D3 bundle defaults, `seed=7`, exact-pq, coherent blue-sky answer. |
| `27b-mxfp8-d3-bundle-generation-config.log` | Unseeded D3 bundle-default row produced incoherent text and `loop=YES`; kept as a real failure artifact. |
| `27b-mxfp8-d3-bundle-generation-config-rerun.log` | Unseeded rerun with same settings produced coherent answer, proving stochastic rows need seed control before diagnosis. |
| `27b-mxfp8-d3-bundle-generation-config-topk0.log` | Bundle defaults with only `topK=0` override produced coherent answer. |
| `27b-mxfp8-vl-batch-mediasalt.log` | `Qwen3VLProcessor` loaded; image tensor shape `[196, 1536]`; same-image cache probe HIT; different-image media salt probe MISS. |

Important current interpretation:

- Bundle `generation_config.json` is being applied in the perf harness. The
  27B MXFP8 generation config resolves to `temperature=1.0`, `top_p=0.95`,
  `top_k=20`.
- The Swift 27B MXFP8 greedy D3 row is coherent but still below the desired
  Python/source-server speed target. Do not hide this with sampler changes.
- Stochastic exact-pq native MTP is not yet a production-blessed quality path.
  Seeded row `seed=7` was coherent, but an unseeded row produced a real
  incoherent output. Further work must diagnose exact-pq/top-k acceptance
  behavior with seeded repeats rather than clamping top-k or temperature.
- The VL media-salt row proves VLM processing and cache isolation for the 27B
  MXFP8 bundle, but it does not yet prove native MTP plus VL in the same
  generation loop.

## 2026-05-17 Sampler Top-K and VL Native-MTP Update

Artifact directory:

```text
docs/local/qwen36-mtp-overnight/20260517T080234Z-sampler-topk-mxfp/
```

Seeded bundle-default rows show that `top_k` is a real MTP acceptance/speed
variable, not a cosmetic setting:

| Model | Row | top_k | tok/s | Verify calls | Rejects | Output |
|---|---|---:|---:|---:|---:|---|
| 27B MXFP8 | `27b-mxfp8-d3-bundle-seed11-topk20-count.log` | 20 | 26.1 | 58 | 25 | exact `1..50`, stop, no loop |
| 27B MXFP8 | `27b-mxfp8-d3-bundle-seed11-topk0-count.log` | 0 | 23.3 | 65 | 31 | exact `1..50`, stop, no loop |
| 27B MXFP4 | `27b-mxfp4-d3-bundle-seed11-topk20-count.log` | 20 | 34.2 | 70 | 48 | exact `1..50`, stop, no loop |
| 27B MXFP4 | `27b-mxfp4-d3-bundle-seed11-topk0-count.log` | 0 | 27.3 | 88 | 69 | exact `1..50`, stop, no loop |
| 35B MXFP8 | `35b-mxfp8-d3-bundle-seed11-topk20-count.log` | 20 | 101.9 | 48 | 2 | exact `1..50`, stop, no loop |
| 35B MXFP8 | `35b-mxfp8-d3-bundle-seed11-topk0-count.log` | 0 | 104.1 | 48 | 2 | exact `1..50`, stop, no loop |
| 35B MXFP4 | `35b-mxfp4-d3-bundle-seed11-topk20-count.log` | 20 | 124.8 | 49 | 5 | exact `1..50`, stop, no loop |
| 35B MXFP4 | `35b-mxfp4-d3-bundle-seed11-topk0-count.log` | 0 | 128.8 | 49 | 5 | exact `1..50`, stop, no loop |

Interpretation:

- The bundle's native `generation_config.json` default
  `temperature=1.0, top_p=0.95, top_k=20` can improve 27B D3 acceptance on
  this prompt and seed. Do not clamp it away.
- The 35B MXFP variants are already near-perfect on this prompt, so `top_k=20`
  and `top_k=0` produce the same accept histograms there; the small speed
  difference is single-row noise.
- Greedy rows remain separate kernel/runtime measurements. Do not compare a
  stochastic bundle-default row to a greedy row without recording the sampler
  source and seed.
- This is not a fake guard or hidden policy. It is the bundle's own stamped
  sampling configuration, with explicit env overrides only for controlled
  experiments.

The same artifact set also adds the first local 27B MXFP8 BatchEngine VL row
with native MTP D3 enabled:

```text
27b-mxfp8-vl-chat-cache-native-mtp-d3.log
```

That row loaded `Qwen3VLProcessor`, generated a correct image answer, replayed
the same image with a disk-backed cache hit, missed on a different image, and
answered the text-only follow-up `Red and blue.` without marker leakage. This
closes the previous local gap where VL media-salt proof existed only without
native MTP active.

The same artifact set adds env-gated Qwen3.6 native-MTP phase diagnostics for
the draft sidecar:

```text
llm_mtp_block
llm_mtp_lm_head
vlm_mtp_block
vlm_mtp_lm_head
```

Short 27B/35B MXFP8 diagnostic rows on `Count from 1 to 20...` show the
remaining 27B gap is primarily target verifier/backbone phase cost, not the MTP
sidecar:

| Model | Artifact | Verify sec | MTP block sec | MTP LM-head sec | Output |
|---|---|---:|---:|---:|---|
| 27B MXFP8 | `27b-mxfp8-d3-phase-diag-topk20-count20-v2.log` | 2.470 | 0.077 | 0.150 | exact `1..20` |
| 35B MXFP8 | `35b-mxfp8-d3-phase-diag-topk20-count20.log` | 0.835 | 0.040 | 0.067 | exact `1..20` |

The 27B MXFP8 quant-dispatch trace shows small-M quantized matmuls are taking
the real MXFP8 `qmv` route:

```text
27b-mxfp8-d3-quant-dispatch-trace.log
[QuantDispatch] primitive=QuantizedMatmul path=qmv M=1 ... bits=8 groupSize=32 mode=mxfp8
```

So the next root-cause lane is not "MTP sidecar missing" or "fp fallback".
It is whether the target verifier's small-M MXFP8 qmv/backbone path needs a
real optimized verifier kernel or compiled small-M path.

## 2026-05-17 Min-P Sweep Update

Artifact directory:

```text
docs/local/qwen36-mtp-overnight/20260517T081510Z-minp-sweep/
```

Seeded 27B D3 rows with bundle defaults plus `top_k=20` show that `min_p` is
not currently a useful speed fix:

| Model | min_p=0.00 | min_p=0.02 | min_p=0.05 | min_p=0.10 |
|---|---:|---:|---:|---:|
| 27B MXFP8 | 26.1 tok/s, 58 verify, 25 rejects | 26.1 tok/s, 58 verify, 25 rejects | 24.5 tok/s, 60 verify, 30 rejects | 23.5 tok/s, 58 verify, 25 rejects |
| 27B MXFP4 | 34.5 tok/s, 70 verify, 48 rejects | 34.2 tok/s, 70 verify, 49 rejects | 32.6 tok/s, 68 verify, 46 rejects | 28.3 tok/s, 69 verify, 46 rejects |

All rows produced exact `1..50`, stopped normally, and did not loop. The
result is therefore not a quality failure. It is evidence that `min_p` should
remain a normal request or bundle input, not a hidden MTP speed guard.

The useful sampler conclusion so far is narrower:

- honor `generation_config.json` defaults, including `top_k=20`, because they
  can materially affect exact-pq acceptance on 27B rows;
- record sampler source and seed for stochastic speed rows;
- do not force `min_p`, temperature, top-p, top-k, or repetition penalty as
  compensating runtime policy.

## 2026-05-17 Lazy-Repair Probe

Artifact directory:

```text
docs/local/qwen36-mtp-overnight/20260517T082844Z-lazy-repair-topk/
```

The 27B MXFP8 D3 row was rerun with bundle defaults, `seed=11`, and
`VMLX_NATIVE_MTP_HYBRID_VERIFY=chunk_lazy_repair`.

| Mode | tok/s | Verify calls | Rejects | Replay forwards | Output |
|---|---:|---:|---:|---:|---|
| `chunk_commit` | 26.1 | 58 | 25 | 0 | exact `1..50` |
| `chunk_lazy_repair` | 19.5 | 57 | 21 | 45 | exact `1..50` |

`chunk_lazy_repair` is correctness-preserving on this row, but it is slower
because partial rejections trigger 45 replay forwards and 2.846 seconds of
replay time. Do not pursue lazy repair as the 27B MXFP8 speed path unless a
later kernel change alters the replay cost profile.

Current next root-cause lane: keep `chunk_commit` semantics and focus on the
target verifier/backbone small-M MXFP8 path, because the phase diagnostics and
lazy-repair row both point away from sampler policy and accepted-prefix capture
as the primary remaining cost.

## 2026-05-17 Backbone / Scheduler Triage

New artifacts:

```text
docs/local/qwen36-mtp-overnight/20260517T083241Z-gdn-phase-diag/
docs/local/qwen36-mtp-overnight/20260517T083538Z-ar-phase-snapshot/
docs/local/qwen36-mtp-overnight/20260517T083628Z-iter-vs-batch/
```

The GDN replay diagnostic row keeps 27B MXFP8 D3 on `chunk_commit`, bundle
defaults, and `seed=11`. It produced exact `1..20`. Accepted-prefix replay is
measurable but not the main gap:

```text
gdnReplayCalls=864
gdnReplayStates=2592
gdnReplaySec=0.081
phaseDiag=vlm_mlp:1280/1.508,vlm_gdn:960/0.891,vlm_attention:320/0.262,...
```

So GDN replay is not worth replacing with lazy repair; lazy repair was already
slower, and capture replay is only a small fraction of the verifier cost.

RunBench now has an explicit opt-in AR phase snapshot:

```text
BENCH_PERF_PHASE_SNAPSHOT=1
```

This is diagnostic instrumentation only. It prints `PERF_PHASE` for regular
AR rows, but the extra layer-level eval points mean those rows are not direct
production speed gates.

AR diagnostic comparison on the same count `1..20` prompt:

| Model | tok/s | Phase split |
|---|---:|---|
| 27B MXFP8 | 11.2 | `vlm_mlp:4544/3.410`, `vlm_gdn:3408/2.032`, `vlm_attention:1136/0.684` |
| 35B MXFP8 | 31.0 | `vlm_mlp:2840/1.203`, `vlm_gdn:2130/0.840`, `vlm_attention:710/0.311` |

This confirms the 27B gap is inherited from the dense backbone before MTP is
involved. The 35B A3B path is sparse and has fewer/lighter phase calls.

Direct iterator versus BatchEngine was also checked for the 27B MXFP8 D3
count `1..50` row:

| Path | tok/s | Verify calls | Accept histogram | Target verify sec |
|---|---:|---:|---|---:|
| BatchEngine prior | 26.1 | 58 | `0:6,1:6,2:13,3:33` | 6.252 |
| Direct iterator | 25.9 | 58 | `0:6,1:6,2:13,3:33` | 6.292 |

Scheduler/stream overhead is therefore not the remaining speed gap for this
row.

Current narrowed target: 27B MXFP8 dense verifier MLP/GDN execution, especially
small-M MXFP8 matmul and a possible compiled/tuned verifier path. Do not spend
more time on min-p, lazy repair, or BatchEngine overhead unless new evidence
appears.

## 2026-05-17 QMM Dispatch Probe

Artifact directory:

```text
docs/local/qwen36-mtp-overnight/20260517T0920-qmm-dispatch-probe/
```

The small-M MXFP8 verifier hypothesis was tested directly by temporarily
routing `M=4` MXFP8 quantized matmuls through `qmm` instead of the default
`qmv` path. The row preserved the bundle's real generation config
(`temperature=1.0`, `top_p=0.95`, `top_k=20`, no `min_p`, no repetition
penalty), `seed=11`, native MTP D3, and `chunk_commit`.

The trace proved the diagnostic override worked:

```text
[QuantDispatch] primitive=QuantizedMatmul path=qmm M=4 ... vectorLimit=4 bits=8 groupSize=32 mode=mxfp8
```

But it was slower:

| Row | Dispatch for M=4 | tok/s | Verify calls | Rejects | Target verify sec | Output |
|---|---|---:|---:|---:|---:|---|
| Prior baseline | `qmv`, vectorLimit=10 | 26.1 | 58 | 25 | 6.252 | exact `1..50`, stop, no loop |
| `27b-mxfp8-d3-bundle-seed11-topk20-qmm-min4.log` | `qmm`, vectorLimit=4 | 16.3 | 66 | 43 | 10.397 | exact `1..50`, stop, no loop |

Decision: reject forced qmm. The temporary override was removed. Keep the
existing dispatch trace, but do not add a production qmm-min-M switch based on
this evidence.

Next target remains a real tuned small-M MXFP8 verifier path or cache-aware
compiled verifier path. It must improve the same seeded bundle-default row and
then rerun the VL/cache matrix; sampler guards are not an acceptable substitute.

## 2026-05-17 27B MXFP8 D2 vs D3 Clean Repeat

Artifact directory:

```text
docs/local/qwen36-mtp-overnight/20260517T0928-27b-mxfp8-d2-d3-bundle-repeat/
```

Both rows used the current rebuilt binary, bundle generation defaults
(`temperature=1.0`, `top_p=0.95`, `top_k=20`, no `min_p`, no repetition
penalty), `seed=11`, and `chunk_commit`.

| Depth | tok/s | Verify calls | Accepted by depth | Rejects | Target verify sec | MTP draft sec | Sampling sec | Materialize/sync sec | Output |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| D2 | 28.3 | 65 | `0:1,1:5,2:59` | 6 | 5.885 | 0.601 | 0.167 | 0.983 | exact `1..50`, stop, no loop |
| D3 | 25.3 | 58 | `0:6,1:6,2:13,3:33` | 25 | 6.436 | 0.803 | 0.209 | 1.216 | exact `1..50`, stop, no loop |

Interpretation: D3 is live and coherent for 27B MXFP8, but it is currently
slower than D2 on this controlled bundle-default row. D3 saves seven verifier
calls, but the `M=4` verifier cost plus additional draft/sampling/materialize
work loses the row. Keep D2 as the current 27B speed choice until a real
verifier hot-path change moves D3.

`claude -p` was attempted for a second opinion on the verifier lane. The GNU
`timeout` wrapper is not available on this macOS install; the retried Perl
alarm wrapper produced zero bytes before the 180-second alarm. No external
recommendation is used here.

## 2026-05-17 27B MXFP8 Top-P Sweep

Artifact directory:

```text
docs/local/qwen36-mtp-overnight/20260517T093618Z-27b-mxfp8-top-p-sweep/
```

This row changed only `top_p` on the current 27B MXFP8 D3 bundle-default
prompt. `temperature=1.0`, bundle `top_k=20`, `seed=11`, no `min_p`, no
repetition penalty, native MTP D3, and `chunk_commit` stayed fixed.

| top_p | tok/s | Verify calls | Rejects | Accepted by depth | Output |
|---:|---:|---:|---:|---|---|
| 1.00 | 21.6 | 69 | 48 | `0:12,1:16,2:20,3:21` | exact `1..50`, stop, no loop |
| 0.95 | 25.4 | 58 | 25 | `0:6,1:6,2:13,3:33` | exact `1..50`, stop, no loop |
| 0.90 | 23.9 | 58 | 25 | `0:6,1:6,2:13,3:33` | exact `1..50`, stop, no loop |
| 0.85 | 22.6 | 58 | 25 | `0:6,1:6,2:13,3:33` | exact `1..50`, stop, no loop |

Decision: do not force a `top_p` override. The bundle default `top_p=0.95`
remains the best row in this bounded sweep. `top_p=1.0` materially worsened
acceptance, and lower `top_p` values did not improve the accept histogram.
This keeps the remaining speed target on the verifier/backbone hot path rather
than sampler policy.

Focused source/runtime-setting verification after this sampler pass:

```text
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift test --filter VMLXServerRuntimeSettingsTests --jobs 2
```

Result: `9` tests passed. This covers bundle generation config applying before
explicit server overrides and nil server fields not adding hidden sampler
guards.

Focused VL/cache topology verification:

```text
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift test --filter 'MediaCachePlaceholderTests|CacheCoordinatorModeKeyIsolationTests|CacheCoordinatorTopologyFocusedTests' --jobs 2
```

Result: `9` tests passed. The matching Swift Testing suites covered media
placeholder suffix handling, media-salt isolation, hybrid companion-state
requirements, disk-backed hybrid media-salt hits, DSV4 CSA/HSA pool restore,
and path-dependent cache restore. The older
`CacheCoordinatorModeKeyIsolationTests` name did not resolve to a runnable test
case under this filter in the current checkout, so it is not counted as
executed.

The currently investigated 27B MXFP8 Swift rows have been too slow for
production confidence:

```text
AR  ~= 16-18 tok/s
D1  ~= 21-27 tok/s depending on run
D2  ~= 24-31 tok/s depending on run
D3  ~= 23-30 tok/s depending on run
```

Python-side reference for the same class of Qwen3.6 27B MXFP8 MTP runtime was
closer to:

```text
AR  ~= 15.8 tok/s
D1  ~= 24.7 tok/s
D2  ~= 28.8 tok/s
D3  ~= 28.9 tok/s
```

Some later source-server rows also showed about `34-37 tok/s` for true MXFP8
D3 after the norm fix, so the working ambition is:

```text
Minimum useful target: close the repeatable gap to Python.
Stretch target: about 2.0x AR, roughly 33-36 tok/s on 27B MXFP8.
```

Do not stop at "D2 is recommended." Depth policy is only useful after the Swift
runtime path is actually fast.

## Non-Negotiables

- MTP must only activate from real config and real `mtp.*` tensor evidence.
  Never infer MTP from a directory name.
- CRACK models are not MTP unless tensor inspection proves otherwise.
- No fake quality guards:
  - no forced repetition penalty;
  - no forced temperature/top-p/top-k/min-p;
  - no forced thinking closure;
  - no length-cap fake pass;
  - no output monkeypatch.
- A speed row counts only if visible output is coherent, no loop, no marker
  leak, stop reason is sane, and token/s is reported.
- Exact output equivalence beats subjective coherence. For the count prompt,
  exact `1..50` output is required.
- One heavy model process at a time. Check process state before every real
  model run.
- Do not touch unrelated Flux work or other-agent dirty files.

## Required Hot-Path Accounting

Each MTP row must report enough timing to explain where wall time is going:

| Metric | Purpose |
|---|---|
| wall time | End-to-end row timing. |
| scheduler/decode time | Detect BatchEngine or stream overhead outside model calls. |
| seed main forward count/time | Time spent bridging prompt prefill into MTP state. |
| verifier main forward count/time | Main target verifier forward over `[primary, d1, ...]`. |
| replay/repair main forward count/time | Cost of rollback/repair after reject or partial accept. |
| MTP draft forward count/time | Cost of recursive draft heads. |
| sample/logits time | Argmax/sampling/logit processor overhead. |
| cache snapshot/restore/replay time | Hybrid SSM/KV checkpoint or prefix-state overhead. |
| token materialization / CPU sync time | `.item`, `.asArray`, `MLX.eval`, decode sync costs. |
| accepted tokens per cycle | Real average committed tokens per verify call. |
| acceptance by depth | `0`, `1`, `2`, `3` accept histogram. |
| forward counts | `seed_main`, `verify_main`, `replay_main`, `mtp`. |

If any row cannot report these, mark it as incomplete rather than claiming a
speed result.

## Bottlenecks To Prove Or Rule Out

Investigate these in order, with artifacts:

1. MXFP4/MXFP8 matmul path slower than Python.
2. Verifier chunk path using a slow route for small-M MXFP kernels.
3. MTP heads doing unnecessary full-logit materialization.
4. Excessive CPU/GPU syncs from `.item`, `MLX.eval`, `asArray`, or detokenizing.
5. Cache snapshot, prefix-state capture, restore, or replay overhead.
6. BatchEngine/streaming overhead versus direct iterator overhead.
7. D3 reject overhead making D2 better for this prompt.
8. MTP tensors falling back to fp16/affine path instead of MXFP dispatch.
9. VL/MRoPE/hybrid cache path forcing a generic slower language route.
10. Repeated tensor conversion, shape walk, or mode detection inside decode.

Known result to keep: forcing qmm/NAX at M>=4 was slower on this shape, so do
not promote that without new evidence.

## Measurement Matrix

For each required MXFP artifact, run:

```text
/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP4-MTP
/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP8-MTP
/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP4-MTP
/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP
```

For every model path above, run:

```text
AR
D1 strict chunk
D2 strict chunk
D3 strict chunk
```

Use this prompt first:

```text
Count from 1 to 50 in order, separated by commas.
```

Parameters:

```text
temperature=0
top_p=1
top_k=0
repetition_penalty=nil
max_tokens=192 or enough to finish
```

Then run at least one second prompt after a speed improvement:

```text
What is the capital of France? Answer with one word.
```

Cache-on proof is required before blessing a policy:

```text
same prompt twice
record disk/SSM hit counters
record exact output both runs
record D1/D2/D3 timing if applicable
```

The target is approximately `2x` each model's own AR baseline where the real
runtime can reach it. If a model cannot reach `2x`, the report must show the
best proven depth, exact output status, acceptance histogram, and the measured
bottleneck rather than hiding behind an automatic depth recommendation.

## Candidate Work Loop

Use this loop for each hypothesis:

1. Create an artifact directory:

```sh
mkdir -p docs/local/qwen36-mtp-overnight/<timestamp>
```

2. Record process and repo state:

```sh
ps -axo pid,ppid,stat,lstart,etime,command | rg -i 'RunBench|swift build|swift-build|swift-driver|swift-frontend|swiftc|xcodebuild|vMLX|claude -p' || true
git status --short
```

3. Pick one hypothesis only.

4. Patch minimally. Do not bundle unrelated cleanup.

5. Build:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift build -c release --product RunBench --jobs 2
```

6. Run focused tests:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift test --filter 'MTPRuntimeFocusedTests|QuantizationTests' --jobs 2
```

7. Run AR/D1/D2/D3 rows and save raw logs.

8. Compare exact output, token count, speed, acceptance, and timing.

9. If failed, revert only that experiment's own code and record why.

10. If improved, run a second prompt and cache-on repeat row.

11. Write `REPORT.md` in the artifact directory before the next experiment.

## Recommended Experiment Order

### 1. Complete NativeMTP Accounting

Add missing counters before deeper optimization:

```text
seedMainForwardCount
seedMainForwardTime
verifyMainForwardCount
verifyMainForwardTime
replayMainForwardCount
replayMainForwardTime
mtpForwardCount
mtpForwardTime
materializeSyncTime
cacheSnapshotRestoreTime
decodeWallTime
```

This is necessary to stop guessing.

### 2. Greedy Sampling Fast Path

For `temperature=0`, verify whether Swift is sampling each verifier position
with separate argmax/eval/item syncs. If so, batch argmax over verifier logits
and materialize once per cycle. This must not change sampled tokens.

### 3. MTP Draft Logits Narrowing

Check whether MTP draft heads compute full vocab logits at every recursive step
when only argmax is needed. If a narrower real kernel path is possible, measure
it. Do not fake acceptance.

### 4. Compile NativeMTP Verifier

Investigate a compiled verifier path for Qwen3.6 hybrid cache:

- strict chunk path first;
- no lazy rollback path as default, because it adds repair forwards;
- compile only if cache state is exact;
- prove exact output equivalence against uncompiled strict chunk;
- include cache semantics for hybrid SSM.

This is likely the largest real speed path, but it is also the riskiest.

### 5. MXFP Kernel Dispatch Audit

Confirm all of these:

- base verifier uses the artifact's real MXFP mode, bits, and group size;
- MTP layers use the same quantized dispatch;
- `lm_head`, projections, and MTP projections are not falling back to fp16
  unexpectedly;
- norms use the repaired Qwen3.6 convention and are not shifted twice;
- no decode-time shape walk or tensor conversion happens after load.

### 6. Depth Policy Only After Runtime Work

Once runtime speed is close enough, write local MXFP policy:

| Model | Policy |
|---|---|
| 27B MXFP4 | Measure; D2 can beat D3. |
| 27B MXFP8 | Do not bless until speed gap is explained. |
| 35B MXFP4 | D3 likely best, but remeasure. |
| 35B MXFP8 | D3 likely best, but remeasure. |

## 2026-05-17 Generation Config / Top-K Checkpoint

The active MXFP MTP bundles must be measured with their real sampling defaults
unless an experiment explicitly says otherwise. Current local files show:

| Bundle | `generation_config.json` sampling defaults |
|---|---|
| `/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP4-MTP` | `do_sample=true`, `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=nil`, `repetition_penalty=nil` |
| `/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP8-MTP` | `do_sample=true`, `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=nil`, `repetition_penalty=nil` |
| `/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP4-MTP` | `do_sample=true`, `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=nil`, `repetition_penalty=nil` |
| `/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP` | `do_sample=true`, `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=nil`, `repetition_penalty=nil` |

Quantization metadata is not identical across the files named "MXFP4":

- 27B MXFP4 is stamped as `mode=mxfp4`, `bits=4`, `group_size=32`.
- 35B A3B "MXFP4" is stamped as `mode=affine`, `bits=4`,
  `group_size=32`, `quantization_backend=mx.quantize`.
- Both MXFP8 bundles are stamped as `mode=mxfp8`, `bits=8`,
  `group_size=32`.

The same 35B A3B path on `erics-m5-max.local` is also stamped
`mode=affine`, so copying that folder would not create a true 35B MXFP4 proof.
Current 35B A3B "MXFP4" speed rows are 4-bit affine rows unless a newly
converted true-MXFP4 artifact is provided. Do not describe those rows as true
MXFP4 kernel proof without inspecting the actual `config.json` and dispatch
trace.

The active server/settings contract is:

- nil server fields do not add temperature floors, repetition penalties,
  top-p/top-k clamps, or family rescue settings;
- bundle defaults apply before explicit server/UI overrides;
- explicit env rows must print the override in `PERF_RUN`;
- native-MTP non-greedy accept/reject uses
  `SpeculativeSamplingController(parameters:)`, so `top_k`, `top_p`, and
  `min_p` alter both the draft and verifier probability distributions.

New focused proof:

```text
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift test --filter VMLXServerRuntimeSettingsTests --jobs 2
```

Result: `10` tests passed, including
`bundle top-k reaches speculative sampler probabilities`. This is the
speed-relevant bridge: bundle `top_k=2` masks out non-top-k probability mass in
the same sampler native MTP uses. This is not a default change and not a guard.

Prior live rows already show why this matters:

- 27B MXFP8 D3, seed 11, bundle `top_k=20`: `26.1 tok/s`, `58` verifier
  calls, `25` rejects;
- 27B MXFP8 D3, same row with explicit `top_k=0`: `23.3 tok/s`, `65`
  verifier calls, `31` rejects;
- 27B MXFP4 D3, seed 11, bundle `top_k=20`: `34.2 tok/s`, `70` verifier
  calls, `48` rejects;
- 27B MXFP4 D3, same row with explicit `top_k=0`: `27.3 tok/s`, `88`
  verifier calls, `69` rejects.

Therefore, future speed rows must name the sampling source and print
`temp/topP/topK/minP/rep`. If the row is meant to test raw greedy AR, use an
explicit label such as `explicit-greedy`; otherwise use
`BENCH_PERF_USE_GENERATION_CONFIG=1` and record the bundle path.

Current-checkout VL regate:

```text
docs/local/qwen36-mtp-overnight/20260517T084520Z-vl-current-regate/REPORT.md
```

Result: PASS. The 27B MXFP8 BatchEngine native-MTP D3 row loaded
`Qwen3VLProcessor`, answered the image coherently, hit disk cache for the same
image (`84/84` tokens), missed for a different image, and answered a text-only
follow-up as `Red and blue.` Each turn emitted native-MTP telemetry with
`verifierMode=chunk_commit`.

Future 27B/35B MXFP rows still need the same VL matrix when sampling or cache
behavior changes, because MRoPE/VL processing can affect prompt tokens, media
salt, cache hits, and the verifier continuation state.

## 2026-05-17 Batched Exact-PQ Rejection

Artifact directory:

```text
docs/local/qwen36-mtp-overnight/20260517T085729Z-batched-exactpq/
```

A temporary `NativeMTPTokenIterator` experiment batched exact-pq verifier
target probability rows for the no-processor path. It preserved correctness but
did not improve the 27B MXFP8 D3 speed row:

| Row | tok/s | Verify calls | Rejects | Sampling sec | Materialize/sync sec | Output |
|---|---:|---:|---:|---:|---:|---|
| Baseline `top_k=20`, `seed=11` | 26.1 | 58 | 25 | 0.189 | 1.166 | exact `1..50`, stop, no loop |
| Batched exact-pq | 26.2 | 58 | 25 | 0.110 | 3.432 | exact `1..50`, stop, no loop |

The identical accept histogram (`0:6,1:6,2:13,3:33`) means this was only a
materialization/sampler plumbing experiment, not a quality or acceptance
change. The patch was reverted. Post-revert focused tests passed:
`VMLXServerRuntimeSettingsTests` `10/10` and `MTPRuntimeFocusedTests`
`38/38`.

Decision: do not pursue exact-pq batching as a production speed path. The next
real lane remains dense 27B verifier/backbone execution: small-M MXFP8 `qmv`,
fixed-shape verifier execution, compiled verifier graphs, or lower-cost target
hidden/logit materialization.

## 2026-05-17 Compiled-Flag Probe

Artifact directory:

```text
docs/local/qwen36-mtp-overnight/20260517T0905-compiled-flag-probe/
```

`BENCH_PERF_COMPILED=1` was tested against the exact 27B MXFP8 D3/top-k
baseline prompt. It is a no-op for native MTP:

| Row | compiled flag | tok/s | Verify calls | Rejects | Output |
|---|---|---:|---:|---:|---|
| Baseline `top_k=20`, `seed=11` | off | 26.1 | 58 | 25 | exact `1..50`, stop, no loop |
| Compiled flag exact prompt after release rebuild | on | 25.8 | 58 | 25 | exact `1..50`, stop, no loop |

Source read confirms the path: `BatchEngine.generate` routes native MTP into
`startSoloFastPath`, which constructs `NativeMTPTokenIterator` directly. The
normal `TokenIterator` compiled decode closure is not used by native MTP.

Decision: do not spend more time on generic compiled decode flags for native
MTP. A compiled verifier speed path must be implemented inside the native-MTP
verify loop with explicit Qwen hybrid SSM cache commit/rollback semantics.

## Report Template

Each experiment report must include:

```text
# Experiment <N>: <hypothesis>

## Patch
- Files changed:
- Summary:

## Commands
- Build:
- Tests:
- Bench rows:

## Results
| Mode | tok/s | exact output | tokens | accept depth | target verify s | mtp draft s | materialize/sync s | notes |
|---|---:|---|---:|---|---:|---:|---:|---|

## Bottleneck Finding
<one concrete statement>

## Decision
PASS / FAIL / PARTIAL

## Next
<next hypothesis>
```

## Short Prompt For An Overnight Agent

```text
Work in /Users/eric/vmlx-swift on Qwen3.6 native MTP speed. Do not treat this as only a depth-policy task. The goal is to get the MXFP MTP variants as close as possible to 2x their own AR baselines while preserving exact output and cache correctness.

Use these four models only for this pass: /Users/eric/models/JANGQ/Qwen3.6-27B-MXFP4-MTP, /Users/eric/models/JANGQ/Qwen3.6-27B-MXFP8-MTP, /Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP4-MTP, and /Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP. Add or use hot-path accounting for wall time, seed/verify/replay main forwards, MTP draft forwards, sampling/logits, cache snapshot/restore/replay, token materialization/CPU sync, acceptance by depth, accepted tokens per cycle, and forward counts.

For each MXFP model, run AR/D1/D2/D3 on the count 1..50 prompt with temp=0, top_p=1, top_k=0, rep=nil. Save raw logs under docs/local/qwen36-mtp-overnight/<timestamp>/ and write REPORT.md after each experiment. A speed row only counts if output is exact, token count is sane, no loop, no marker leak, and token/s plus acceptance stats are present.

Investigate real causes: MXFP kernel dispatch, verifier small-M route, MTP head full-logit cost, CPU/GPU syncs, cache snapshot/restore/replay, scheduler overhead, depth rejection overhead, MTP tensor quant dispatch, VL/MRoPE/hybrid cache slow path, and decode-time tensor conversion. Do not add fake guards, forced sampling defaults, forced repetition penalty, forced think closure, or model-name MTP activation.

Patch one hypothesis at a time, build RunBench release, run focused MTP/quantization tests, then rebench AR/D1/D2/D3. Revert only your own failed experiment. Do not touch unrelated Flux files. Do not auto-enable MTP globally. Continue until you have a proven bottleneck, a real patch, before/after speed, and remaining gap vs Python.
```
