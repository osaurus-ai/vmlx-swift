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

## Call-site scope correction and native integration

The new split-K entry point is called only by the dense JANGH owner in
`JANGHDenseModelPreparation`. The two audited Allosaurus bundles declare
custom quantization on routed expert projections; their other linears remain
ordinary affine modules. These bundles do not reach this dense entry point.
Do not interpret an Allosaurus run with the split-K flag as evidence that
this kernel executed or accelerated native MTP. The planned third
Allosaurus comparison is skipped because the flag is ineffective there.

Quiescent synthetic operator repeats on Max2 gave baseline/split-K ratios
1.553, 1.575 and 1.054 for rows=4, K=2560, N=512/2560/12288. GPU median 338 MHz;
these are operator diagnostics, not a dense-family full-model qualification.

The same Release binary also contains the separate Steel Trellis prefill
prototype. With generic core g15 dispatch and JANGH Steel on actual g17s
Max2, the legacy versus Trellis-Steel arms returned correct answers at
430.89 versus 422.47 cache-inclusive prompt tok/s on the 3669-token task.
There is no demonstrated full-portable prefill gain in this pair. Native
code/growing turns completed at 67.93/53.66 and 65.96/54.07 tok/s, with
natural stops, closed reasoning, four generated tests and 2,000 independent
function/input-preservation cases per arm. Both 150-value arrays and sums
validate. Reasoning lengths/token counts differ; these unseeded turns are
regression evidence, not matched-output decode speedup. Activity Monitor
during the growing turns showed 31.49 and 32.79 GB.

Both processes exited normally; owned clock/log collectors were stopped.
The exact Release binary SHA-256 is
97ad57136888ae07703d79bb2343e11fd887983b7d94e781c759828f6bc8173a.
All work remains local Max2. The candidate stays draft/default-off pending
a reachable dense-family runtime and physical-footprint qualification.
