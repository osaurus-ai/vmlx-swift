# JANGH weighted-down decode primitive

Status: five focused Metal test methods passed with zero failures or skips on
combined integration build `df567e81`. No loader or factory calls this primitive.
Model integration, speed, and model-family readiness remain unproved.

`JANGHWeightedDownKernel` performs the down projection for each selected route and
accumulates router-weighted token outputs without expanding the packed expert bank.
It implements odd-cubic codebooks for 2/3/4/6/8-bit LSB-packed projections with
per-output-row F16 scales.

The API requires F32 hidden rows `[tokens * routes, hidden]` **already in the down
projection input basis**. The caller must explicitly assert that basis; this
primitive performs no rotation. Route IDs are U32 and weights F32, both shaped
`[tokens, routes]`. Dot products and route accumulation remain F32. Only the final
`[tokens, output]` result is cast to the requested F16, BF16, or F32 dtype. No
normalization or sign restriction is imposed on router weights.

Geometry and dtype mismatches throw before dispatch. An out-of-range expert ID
produces NaNs for its entire token without reading that expert's bank or scales,
including when its score is zero. This is a deterministic diagnostic policy;
production integration must decide how to surface invalid routing safely.

Prepared tests use an independent bit-by-bit unpack and Float64 arithmetic oracle,
covering all five bit widths, input tails beyond a 512-column block, nine-row output
tails, repeated/out-of-order expert IDs, zero/non-normalized/signed scores, and all
three result dtypes, top-eight routing, and coefficient/basis kernel identity. A cancellation-sensitive two-route case distinguishes F32
hidden/accumulation from premature BF16/F16 rounding and from a second rotation.
Invalid tensor layouts and invalid expert IDs have separate tests. Tests use the
shared Metal lock.

The first Metal run caught implicit Float-to-BF16 output assignments rejected by
Metal. Explicit output-type casts now cover both final results and invalid-route
NaNs. Accumulation remains F32, and the kernel identity was advanced to version 2.
The passing regression covers F16, BF16, and F32 outputs. A separate executed
composition test covers mixed gate/up/down widths and basis transitions.

Packed and scale banks must already be available and row contiguous. Metadata-only
admission rejects lazy or strided banks rather than copying them; automatic kernel
repacking is disabled. Small activations and routing inputs may be made contiguous.
Actual mapped-loader admission passed with a tiny synthetic shard. A real model
path still needs matched decode/prefill, cache, multi-turn, and physical-memory proof.
These tests establish no end-to-end speedup.
