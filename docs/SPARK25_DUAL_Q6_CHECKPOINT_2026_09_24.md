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

## Serialized R38 recheck and current-engine integration

On the original qualified R33 binary, native short ABBA (1027tokens/turn) now measures reference97.6/94.6tok/s versus fused101.5/98.5: mean decode latency3.88%lower, throughput4.04%higher. Original long workload ABBA (7032tokens/turn) measures reference90.9/89.9 versus fused93.2/93.1: latency2.91%lower, throughput3.00%higher. n=2 per arm; all full outputs exact, native stop, no length cap or template/sampler changes. Raw submit protocol is not a parsed UI stream. Host-delivery percentiles are recorded; GPU synchronization counts remain unmeasured. Earlier regressions and drift are retained; no causal explanation for them is claimed.

The native long answer remains identical to baseline, including factual imprecision in its hash-table explanation. These are runtime/output-equivalence checks, not a claim of factual correctness for every generated statement.

Evidence: private raptor-dual-q6-recheck-r38/{short-abba-review,long-abba-review}.json and full-model raw logs. Integration branch is based on merged cache enginea2a45e86 with the same kernel source cherry-picked. Combined exact-head build, native/cache proof and app UI/CI still pending; not merge-ready. Other model families retain explicit fallbacks; no generic/all-chip performance claim.

## Integrated engine qualification
Runtime6d5ae291 built successfully in386.46s with no source mutation, binaryf70930277f06a75ba104ae9fec8e970b3008652a98d0ff07398429c50eac4745. Kernel/dispatch/test files are byte-identical to the12focused test inputs. Integrated native1027-token ABBA candidate103.4/102.9 versusreference99.9/91.5, all outputs exact; the reference drift makes the larger apparent gap unreliable, so retain the earlier conservative~3%long-workload result.

Integrated growing-cache ABBA passes8native turns and full raw output/685retained tensors per comparison under2.2GiB quota. Native-equivalent thinking=true matches the bundle's Jinja defaulttrue. A separate parsed batched2 check uses omitted template overrides and passes2turns plus new-process SSD restore:89.79/101.72/101.82tok/s, accepted1183tokens/36layers on restart. Captured1024checkpoint is reused for1155boundary; cold/warm fallback behavior retained. Physical peaks3.93GB for growing rows and1.71GB for batched proof; no low-RAM-family claim.

Private `raptor-dual-q6-recheck-r38/integrated-review.json` contains receipts/traces; all guards passed. Metal System Trace10s capture failed to finish saving before60s; its partial document could not export. Instrumented rates are excluded; no GPU synchronization-count claim.

Engine CI is waived by user instruction, not claimed green. Osaurus still needs its separate exact pin/build/UI/hosted CI before merge. No release or tag.

Final runtime8db8c655 removes only trailing whitespace from30blank Metal-header lines. Exact rebuild passed112.45s with no source mutation and produces the byte-identical RunBench SHA f70930277f06a75ba104ae9fec8e970b3008652a98d0ff07398429c50eac4745. Integrated runtime/cache proof therefore targets the same executable; `git diff --check` passes. `raptor-dual-q6-recheck-r38/final-binary-equivalence.json`.
