# Portable Steel verification diagnostic

Small dense JANGH verification windows already use split-K on the NAX path.
The portable Steel path instead executes each output tile over the full K.
This candidate adds a packed 16x32x32 Steel split-K kernel for 1–16 rows,
reduces FP32 partials, and casts once to the input dtype. It retains the
original packed bank; unavailable or strided banks are declined rather than
implicitly materialized. Ragged output columns are supported.

The candidate is default-off. `VMLX_JANGH_STEEL_SPLITK=1` enables it only on
the Steel path under the existing dense-owner and split-K gates.
`VMLX_JANGH_PREFILL_BACKEND=steel` conservatively selects the portable JANGH
prefill path for local diagnostics. It never forces NAX on unsupported GPUs.
The separate JANGT Steel matrix prototype still needs its own explicit flag.
Neither flag changes the reported physical GPU architecture.

Initial numerical and layout/admission tests pass in native and forced
portable dispatch modes. The selected 14-test suite has one explicit timing
skip on native dispatch and additionally skips the NAX-only test when
forced to Steel. Twelve packed-Trellis/general-kernel methods also pass with
JANGH/trellis Steel and generic g15 core dispatch controls. The latter is a
software-path diagnostic on raw g17s hardware, not older physical hardware.

The expanded native and forced-portable reruns also pass all 40 independent
CPU-golden configurations across the five supported 2/3/4/6/8-bit code
widths, FP16/BF16, 1/3/8/16 rows, ragged columns and uneven K partitions.
The 6-bit case includes a one-column output, exercising tiny-scale padding.
A first operator timing probe overlaps app compilation and is not performance
clearance. Quiescent timing and full-model Release proof remain pending. This is not a physical M3/M4 benchmark, an end-to-end MTP
speedup claim, or a diagnosis of the reported M5 Pro 11 tok/s. No generation
or RAM Safety defaults change. Do not promote this candidate before
coherent multi-turn/cache/lifecycle and physical-footprint qualification.
