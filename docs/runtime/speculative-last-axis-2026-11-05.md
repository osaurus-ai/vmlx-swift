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

Fresh Release dev-app regression controls are recorded below. This checkpoint
is not a speedup, universal family-readiness, low-RAM or multimodal qualification. Canonical models are read-only; bundle sampling parameters
remain authoritative. No release/tag.

SOURCE EVIDENCE: `Libraries/MLXLMCommon/Evaluate.swift` and
`Tests/MLXLMCommonFocusedTests/NoHiddenReasoningCloseBiasFocusedTests.swift`.
LIVE EVIDENCE: local `goal-storage-cache-resume-20261104/` artifacts
`full-focused-xcode/sampler-order-existing-guard` retain the real pre-fix crash;
`full-focused-xcode/speculative-last-axis-affected-v2/Focused.xcresult`,
`SUMMARY.json`, `COMMAND.json` and `test.log` bind the corrected passing
selection. These are real MLX component tests, not Osaurus GUI proof.


## Exact-source Release39 GUI controls

Runtime `6f450646c7f68caea72019b320650ee8a8385c6c` was built through the normal
Osaurus Release workspace, consumer `53a44ee29990d1474bb9d94cab7e59e6881d2943`,
MLX core `3b39b5e7925fa8860ab8ccd79b054349872e24a5`. The isolated locally signed
dev app used fresh SSD cache and canonical external-drive model aliases.
No model weights, generation config, tuning sidecars or prompts were patched.
All seven actual GUI requests have empty generation override objects and
natural stop receipts; reasoning was expanded and inspected.

| Model | Visible short controls | UI decode tokens/s | Disk restore |
|---|---|---|---|
| Raptor0.7 JANG_6M | Five/$22, then six/$27/+$5 | 98.0 / 101.0 | 346 tokens |
| Diplodocus preview4 JANG_6M | Five/$22, then six/$27/+$5 | 61.5 / 69.9 | 233 tokens |
| Flash-Next JANG_2L | 42, then126, then100 after restart | 40.7 / 61.4 / 80.5 | 182 growing /339 restart |

Flash-Next used the normal selected Adaptive setting and existing bundle
tuning. Runtime traces confirm native MTP on the actual requests. The final
process-restart row records nine verification calls, three residual corrections,
`samplingMode=exact-pq`, `activeDepth=3` and `adaptiveCeiling=7`; depth was not
pinned. Its bundle uses T1/p.95/top-k20. Two app processes and two physical-memory
samplers exited zero. Canonical metadata hashes and Flash-Next weight file
sizes/mtimes remain unchanged. Weight payload checksums were not recomputed.

These short rates are unpaired observations, not a speed comparison. Activity
request rates differ from UI decode rates and are retained separately. Owned
compute residency peaked at54,187,185,952bytes in the first process and
53,600,409,184bytes after restart; low RAM is not qualified. Diplodocus's second
reasoning initially miscounted folders and then explicitly corrected itself.
Its correct final answer does not make every intermediate reasoning calculation
correct. That trace and payload are retained as a training note; no causal
attribution to this sampler fix or matched quality improvement claim.

The full same-target candidate run records1249total/1199passed/13failed/37skipped,
including1003SwiftTesting functions with18issues. Exact failing method identity
comparison against main shows no new failures. The corrected sampler source
guard removes one old failure; remaining unrelated failures are still open.

LIVE EVIDENCE: `sampler-native/FINAL-CANDIDATE-NATIVE-RECEIPT.json`, actual
AX/PNG/reasoning/SQLite/cache/physical receipts in `candidate39` and
`candidate39-reopen`, and `APP-IDENTITY-BUILD39-SAMPLER.json` bind seven GUI turns
to the exact binary/source. `full-focused-xcode/speculative-last-axis-full-candidate`
retains the complete red suite and baseline comparison. The leaf rank crash is
reproduced by component tests; existing app callers already slice logits to2D.
No universal family, tax/document, batching or multimodal readiness claim.
