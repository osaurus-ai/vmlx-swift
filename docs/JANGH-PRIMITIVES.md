# JANGH metadata and numerical primitives

This change adds strict format validation and experimental Metal projection primitives. It does not register a JANGH model loader, alter generation settings, or claim a speedup.

The validator recognizes version 2, LSB bitstream packing, F16 row scales, odd-cubic codebooks, and 2/3/4/6/8-bit projections. It checks finite coefficients and levels, complete gate/up/down groups, compatible input rotation, and packed tensor geometry. These codes are not affine weights and must not enter an affine fallback.

The single-projection primitive performs guarded packed QMV with F32 accumulation, optional H32 input rotation, and explicit invalid-route handling. Its H32 casts back to the input dtype. The separate fused gate/up primitive supports independent bit widths and codebooks, raw gate/up clamping, SwiGLU, and an optional H32 epilogue preparing the down projection's input. Fused decode preparation preserves F32 after H32 for F16/BF16 source activations; activated hidden rows and the down-input rotation remain F32. These are distinct precision contracts, covered separately by tests.

## Executed numerical evidence

On integration build `df99ffece51054251e30cee17ccdfc5c6ee6a585`, `JANGHProjectionKernelTests` passed all five methods, and `JANGHFusedGateUpKernelTests` passed all five methods on Metal. Coverage includes independent packed decoding and scalar activation oracles, mixed bit widths, nonzero cubic terms, dtype boundaries, odd output dimensions, input tails, invalid routes, multiple H32 blocks, repeated top-eight routing, and finite extreme activations. The optional private packed-range fixture test was supplied and passed, rather than skipped. Model weight excerpts are not included in this repository.

The format contract also has standalone Foundation XCTest evidence. The Metal results came from a combined integration build, not an independently rebuilt head of this branch. They establish bounded primitive numerical behavior, not model load support, coherent generation, cache correctness, low-memory operation, chip-wide coverage, or faster prefill/decode.

## Remaining integration work

A future loader must validate the complete custom format before ordinary quantization decoding, reconcile quantization aliases, check index and tensor headers, and explicitly exclude custom projections from affine fallback. Legacy JANGTQ names must not select these kernels by spelling alone. Model-specific construction, complete expert composition, sorted prefill kernels, resource behavior, and live multi-turn tests are required before enabling a factory route. Naive requires its own architecture implementation; GLM support does not imply it.
