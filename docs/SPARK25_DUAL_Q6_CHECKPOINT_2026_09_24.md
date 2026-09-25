# Spark dual q6 projection checkpoint — 2026-09-24

Separate candidate from merged c3227f4c. No release or tag. The rejected R32 attention kernel is not included.

The candidate computes the q6/g64 gate/up projections from one shared input load, preserves both FP32 reduction sequences and BF16 projection rounding, then applies the existing exact erf GELU/BF16 multiplication before writing. It reuses the merged GELU Metal source. Dispatch is limited to exact QuantizedLinear instances with BF16 affine6/g64 metadata, no additive bias, shape[1,1,2560] to10240, and untraced Metal execution. Other formats, batching, shapes, devices and transformations retain the reference path. Packed weights are unchanged; no overlay, requantization or sampler changes.

A paired-dispatch-only prototype was slower and rejected. The fuller fusion passed exact synthetic and unchanged installed layer0/17/35 tensor probes. Actual guarded helper serial36-layer MLP median4.524ms versus4.959ms (8.77% component latency reduction); this is not whole-model tokens/s. All12focused tests across2suites passed, including existing exhaustive GELU tests, random q6/strided parity, fallback contracts and compiled output parity. Isolated test package copies the actual source with only its shim module import changed.

Private proof: `/Users/eric/vmlx-private-evidence/raptor06-speed-2026-09-23/spark-projection-r33/`: `component-review.json`, `fixtures/source.json`, `complete-tests-run-test.log`, `r6-production-real0/`. First prototype build actor-isolation failure retained inr1; fixed harness inr2.

Remaining: exact-source full runtime build; paired native >=128token decode with median/p95/max, complete/coherent outputs, long/growing/restarted cache equality and realistic prefill. Only promote if a useful full-model gain survives. Then isolated dev-app UI proof and Osaurus CI before pin merge. No broad model family, all-chip, low-RAM, or new cache topology claim from these component probes.

## Runtime evidence and current decision
Exact runtime source56d358f94c942117c88a44a8e7272f626a177e2b built in409.8s with no changed source. Binaryca8ec4c186a23ada37724e508faac0f2ea56bbfd45ff5e1be19fcb0dd04ea2cf. Later0d6073c0 only makes the GPU parity test Metal-conditional; all12tests passed again in0.147s.

Native7032token sequence: candidate59.3, same-binary reference63.0, archived mergedbaseline70.7, candidate repeat74.2tok/s. Every25113character output is identical, native stop, no detected loop, reasoning closed. First pair is a5.9%regression and is retained. Candidate repeat and archived baseline show substantial temporal variation, so these runs do not establish a stable gain or identify the cause. `full-model-drift-review.json` retains all rows and host-delivery median/p95/max; no GPU synchronization-count claim.

An improved component probe uses324unchanged tensors/2,300,313,600bytes across all36distinct installed MLP layers. Exact output; median5.177ms fused versus5.535reference (6.46% component reduction). Reusing one layer did not fully explain the component/full-model discrepancy. `r7-distinct/`.

CacheABBA completed8native turns; all full outputs and685retained checkpoint tensors per comparison exact. Snapshot/finalization tails4.745–5.549s remain separate from decode. Paged cache off, same rotating/full topology as baseline; no new cache topology or low-RAM-family claim. `cache-parity-review.json`, `cache-abba-timing-review.json`.

Do not promote or pin this candidate yet: stable useful whole-model timing and dev-app UI qualification remain missing. No PR, no release/tag. All owned jobs finished. Next isolate cache finalization phases in a separate lane, retaining this candidate and its failed/noisy performance rows rather than presenting component timings as user-visible gains.
