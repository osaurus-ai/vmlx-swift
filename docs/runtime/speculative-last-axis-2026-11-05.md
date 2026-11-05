# Speculative sampler vocabulary-axis checkpoint

The public speculative sampler applied top-k indexing to axis 1 and allocated
only a single probability row. A `[1, 1, vocabulary]` input therefore crashed
in `put_along_axis` because the destination rank did not match its indices.
The unchanged baseline reproduced that Metal-backed failure twice under the
normal Xcode retry mechanism; both attempts are retained.

Use the final vocabulary axis for top-k selection, min-p thresholds and token
probability lookup, and preserve the input shape when scattering probabilities.
Ordinary two-dimensional native-MTP callers already slice model logits before
sampling. This is a public leaf API shape bug; it does not establish a cause
for any previously observed model-quality failure or app crash. Acceptance and
residual correction remain single-token operations, not batch acceptance APIs.
Generation defaults, request RNG ownership, adaptive depth and filter policy
are unchanged. No AgentLoop or app tool code changes.

## Validation

The selected normal macOS Xcode run passes 112 tests with no failures or skips:
107 Swift Testing functions across four suites and five XCTest cases. It
includes all four explicitly selected new/updated methods, not an empty
selector match. GPU controls cover Float32/BFloat16, vector/matrix/singleton
3D logits, distinct rows, fast and generic top-k paths, temperature-before-
nucleus filtering, independent host distribution references, categorical draws,
acceptance and residual correction. Existing TopP shape, JANG defaults, MTP
runtime and acceptance-boundary controls also pass.

An earlier incorrectly spelled method selection ran only adjacent tests;
that receipt is marked invalid for the new methods and is superseded by the
corrected selection above. The wider focused target remains red: current-main
refresh recorded 14 failed tests / 19 Swift Testing issues. A deleted legacy
benchmark source guard also fails the entire NoHiddenReasoningCloseBias suite;
it is preserved, not skipped or made to pass by creating a placeholder file.

Fresh Release dev-app regression controls are pending. This checkpoint is not
a speedup, universal family-readiness, low-RAM, multimodal or native-MTP app
qualification. Canonical models are read-only; bundle sampling parameters
remain authoritative. No release/tag.

SOURCE EVIDENCE: `Libraries/MLXLMCommon/Evaluate.swift` and
`Tests/MLXLMCommonFocusedTests/NoHiddenReasoningCloseBiasFocusedTests.swift`.
LIVE EVIDENCE: local `goal-storage-cache-resume-20261104/` artifacts
`full-focused-xcode/sampler-order-existing-guard` retain the real pre-fix crash;
`full-focused-xcode/speculative-last-axis-affected-v2/Focused.xcresult`,
`SUMMARY.json`, `COMMAND.json` and `test.log` bind the corrected passing
selection. These are real MLX component tests, not Osaurus GUI proof.
