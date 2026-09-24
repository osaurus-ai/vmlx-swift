# Spark dual q6 projection checkpoint — 2026-09-24

Separate candidate from merged c3227f4c. No release or tag. The rejected R32 attention kernel is not included.

The candidate computes the q6/g64 gate/up projections from one shared input load, preserves both FP32 reduction sequences and BF16 projection rounding, then applies the existing exact erf GELU/BF16 multiplication before writing. It reuses the merged GELU Metal source. Dispatch is limited to exact QuantizedLinear instances with BF16 affine6/g64 metadata, no additive bias, shape[1,1,2560] to10240, and untraced Metal execution. Other formats, batching, shapes, devices and transformations retain the reference path. Packed weights are unchanged; no overlay, requantization or sampler changes.

A paired-dispatch-only prototype was slower and rejected. The fuller fusion passed exact synthetic and unchanged installed layer0/17/35 tensor probes. Actual guarded helper serial36-layer MLP median4.524ms versus4.959ms (8.77% component latency reduction); this is not whole-model tokens/s. All12focused tests across2suites passed, including existing exhaustive GELU tests, random q6/strided parity, fallback contracts and compiled output parity. Isolated test package copies the actual source with only its shim module import changed.

Private proof: `/Users/eric/vmlx-private-evidence/raptor06-speed-2026-09-23/spark-projection-r33/`: `component-review.json`, `fixtures/source.json`, `complete-tests-run-test.log`, `r6-production-real0/`. First prototype build actor-isolation failure retained inr1; fixed harness inr2.

Remaining: exact-source full runtime build; paired native >=128token decode with median/p95/max, complete/coherent outputs, long/growing/restarted cache equality and realistic prefill. Only promote if a useful full-model gain survives. Then isolated dev-app UI proof and Osaurus CI before pin merge. No broad model family, all-chip, low-RAM, or new cache topology claim from these component probes.
