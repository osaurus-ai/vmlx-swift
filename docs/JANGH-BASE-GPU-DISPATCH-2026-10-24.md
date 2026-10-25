# Preserve the base-GPU architecture suffix

`JANGHPrefillKernel.nativeBackend` split Metal architecture names on `g`.
For `applegpu_g17g`, the last nonempty component becomes `17`, losing the
GPU kind. The selector incorrectly returns Steel, and JANGT's NAX matrix
prefill availability also becomes false.

Parse the final GPU kind separately from the numeric generation. Preserve
the existing macOS 26.2 availability gate and generation thresholds. Older
M3/M4 classes still use Steel; the tested malformed suffixes do not enable NAX.
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

Local Release integration consumes runtime c2ab69f771 in app source
bbb8fd4dc on raw g17s Max2 hardware. Native v0.2 history and a validated
200-value odd-number array complete naturally at 45.69 and 78.83 tok/s;
v0.1 model-switch history and a validated 150-value multiples array complete
at 50.58 and 73.73 tok/s. All four turns close reasoning normally. Activity
Monitor shows 32.03 / 39.12 GB during the two sustained code turns; full-request
GPU clock medians are 1585.5 / 1590.5 MHz. These are regression/integration
rows on an unchanged g17s backend, not a measured base-M5 speedup.

Strict-mode settings Save unloads the resident to 798.8 MB in Activity
Monitor. The native UI then refuses v0.1 with its measured 46,214,708,950-byte
estimate against an explicit 36,077,725,286-byte diagnostic budget, matching
the recorded load decision. Safe Auto and Strict retain their existing
different admission semantics; this selector fix changes neither.

Exact Release binary SHA-256:
205266d8baa1f2ccd3e902aecd70415018e549cfd2683fde95d542dad1ad3404.
The proof artifact is isolated and has its Sentry DSN removed. Original
screenshots, process-footprint/clock samples and responses remain private.
Actual base-M5 and older-chip speed remain unmeasured. Do not claim the
reported 11 tok/s or Metal interactivity workload is fixed by this selector.
