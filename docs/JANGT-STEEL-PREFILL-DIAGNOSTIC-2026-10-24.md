# JANGT Steel matrix prefill diagnostic

The current default JANGT matrix prefill requires NAX. Other GPUs use the
decode kernels for prompt batches. This prototype adds a packed Steel MMA
path behind the explicit process-launch setting
`VMLX_JANGT_PREFILL_BACKEND=steel`; it does not change default dispatch.
`VMLX_JANGT_PREFILL=0` still disables matrix prefill entirely.

The Steel tile is 16 by 32 by 64 with 128 threads. It reads the existing
trellis bitstream and FP32 scales into small threadgroup tiles, without a
decoded expert bank or permanent overlay. Gate/up activation, outliers and
optional Hadamard-32 output rotation use the existing separate operations.
This preserves the bundle's generation and RAM Safety policies.

Local Max2 validation covers 37 distinct test methods across the low-bit and
general JANGT suites. Twenty one-hot configurations cover FP16/BF16, widths
1/1.5/2/2.5/3, input widths 640/2560, and full/partial output tiles:
2,867,200 values agree exactly with the CPU decoder after the output dtype
round trip. Composed gate/up/outlier/down cases cover ragged expert segments
and column scales against the existing dense CPU oracle and thresholds.
Additional cases cover tiny index arrays, partial row tiles, invalid routes,
clamped activation and Hadamard-32 output rotation.

The first edge fixtures used 37 output rows and were correctly rejected by
the format's multiple-of-eight admission rule. Corrected fixtures use 40;
the admission rule and numerical thresholds are unchanged. Initial failures
and corrected-run receipts are retained separately.

This is **partial**: full-model Release speed, coherent multi-turn output,
cache restoration/cancellation, and Activity Monitor physical footprint still
need qualification on this prototype. Forcing Steel on Max2 can exercise its
math but cannot certify M3 or M5 Pro performance. Keep default-off and do not
claim that the reported 11 tok/s problem or the Metal interactivity error is
fixed by these kernel tests.
