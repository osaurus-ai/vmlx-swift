# Ornith 122B affine decode candidate

Status: draft, not qualified for release or an Osaurus pin. Keep open for review.

## Trigger and implementation

The actual Ornith-1.5-122B-JANG_4M app run reported
`fused_affine_moe_decode=construction_rejected gate=3072x1024:4b:g64`.
The existing Ornith shape is 2048/512/top-8; it does not cover this model.

This candidate adds 3072/1024/top-8 to the existing paired gate/up and weighted
down Metal implementation. It admits only q4, group 64, BF16 scales and biases
on all three projections. Other layouts retain the generic path. There is no
new Metal arithmetic, packed-weight copy, router change, sampler change, or
cache dtype change. The ordinary limit stays at four rows; this candidate does
not extend the Flash-only eight-row verification scope.

This is an affine routed-expert candidate. It does not qualify JANGH2 Hadamard
banks, shared-expert row exactness, attention exactness, or the adaptive governor.
It enables the matching shape on this draft branch; do not merge on fixture
coverage alone. The existing `VMLX_QWEN4_EXP_FUSED_AFFINE_MOE=0` provides an A/B
control in a separate process without editing model files or generation settings.

## Focused numerical evidence

`Ornith122BFusedAffineMoETests` exercises the real Metal reducer at this geometry
using q4g64 BF16 banks, eight selected experts out of sixteen, strided inputs and
indices, and both BF16/F32 routing scores. This is a numerical fixture, not a
256-expert model residency or performance measurement.

- Single rows are checked against independently dequantized FP32 matmul and
  against the generic gathered quantized path's error relative to that reference.
- Two- and four-row outputs must match serial fused rows byte for byte.
- Five-row input and F16 metadata are rejected at this new geometry.
- Test keys are deterministic fixture inputs only; user generation seeds are unchanged.

Command:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift test --build-system swiftbuild --filter Ornith122BFusedAffineMoETests
```

The final expanded focused run passed (one XCTest, no failures): fused relative
FP32 error 0.00403–0.00428; generic gathered relative error 0.00875–0.01113.
W2/W4 serial equivalence, five-row rejection and F16-metadata rejection all passed.
These numerical tolerances do not establish whole-model output equivalence.

## Required before promotion

1. Same-bundle model logits/quality, including long prompts and the existing MTP
   multi-row discrepancy. A 0.4% tensor error is not a full-model parity guarantee.
2. Matched Release Osaurus A/B, existing switch on/off, same prompt/settings,
   logged GPU clock and warm/cold load state. Record prose, easy code, tools,
   actual speculative dispatch/acceptance, prefill, TTFT and decode separately.
3. RAM/residency, SSD checkpoint restore and write cost, followed by connected
   tool/media turns. Preserve the generic fallback for larger prefill batches.
4. Do not use the earlier AR app cache receipts as proof of this candidate.

The existing 128k policy transition and the remaining shared-expert/attention
MTP row drift are separate unresolved findings. This candidate does not claim
to fix either or to deliver an adaptive-MTP speedup.
