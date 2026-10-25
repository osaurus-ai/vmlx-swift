# Preserve the base-GPU architecture suffix

`JANGHPrefillKernel.nativeBackend` split Metal architecture names on `g`.
For `applegpu_g17g`, the last nonempty component becomes `17`, losing the
GPU kind. The selector incorrectly returns Steel, and JANGT's NAX matrix
prefill availability also becomes false.

Parse the final GPU kind separately from the numeric generation. Preserve
the existing macOS 26.2 availability gate and generation thresholds. Older
M3/M4 classes still use Steel; malformed metadata does not enable NAX.
No weights, kernels, generation defaults or RAM Safety policy change.

The unchanged parser fails two base-GPU assertions and one malformed-name
assertion. Corrected parser fixtures pass, including M3 g15 and M4 g16
families; the seven existing JANGH kernel tests also pass locally on Max2.

This is a **base-M5** defect. It is not an explanation of the reported M5
Pro slowdown: [an upstream M5 Pro measurement](https://github.com/ml-explore/mlx/pull/3295)
reports `applegpu_g17s`, which the old selector already handles. Pro and
Max cannot be distinguished by that suffix. A C++ architecture override is
also not a physical-chip benchmark and does not override this Swift raw
MTLDevice query; the attempted local setup was rejected and retained.

Release app qualification is pending. Actual base-M5 and older-chip speed
remain unmeasured; do not claim that the 11 tok/s reports or the separate
Metal interactivity error are fixed by the selector regression tests.
