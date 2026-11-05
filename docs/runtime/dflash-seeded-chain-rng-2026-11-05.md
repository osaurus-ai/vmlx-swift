# Seeded DFlash chain sampling

An explicit request seed reached the target sampler and the tree acceptance
sampler, but chain proposal, rejection and residual draws still used implicit
process-global MLX randomness. Another operation could therefore change the
generated chain even when the request seed, prompt, model and fixed width were
the same.

`DFlash2TokenIterator` now owns an optional random state for seeded cycles. Its
seed is separated from the target sampler's seed using the existing speculative
sampling stream offset. The cycle scope includes proposals, acceptance,
residual sampling and prefetched proposals. It does not reseed the global RNG.
Requests without an explicit seed retain their previous path. Sampling filters,
model defaults, kernel routes, cache policy and the adaptive width ladder are
unchanged.

## Validation

The actual local Qwen3.8-27B JANGH2 target and bundled DFlash drafter reproduced
the issue with request seed 42, fixed width 8, bundle sampling defaults and a
96-token diagnostic budget. Changing only the global MLX seed from 8 to 555
changed the unpatched chain output. Both rows recorded zero tree cycles and
positive chain verification counts. A separate default-mode control produced
identical output; this is not a failure claim against that mode.

The new opt-in repository regression independently reproduces the failure on
the old runtime. It passes against a fresh Swift 6 build of all 286
`MLXLMCommon` source files, with the same request seed and global perturbation.
It verifies raw token equality, active chain verification, terminal counts,
absence of runtime failure and positive finite native token rates. Its real
model work uses the existing process-wide MLX test lock. Eleven adjacent
tree/top-k acceptance tests and five width-admission tests also pass against
the rebuilt module. Unchanged dependency objects and the native app shader
were reused; this is not a fresh build of the entire package or app.

```sh
VMLX_DFLASH2_SEEDED_TARGET=/path/to/existing/target-with-bundled-dflash2 \
  swift test --filter DFlash2SeededRequestTests
```

The test deliberately uses a short token budget to examine reproducibility.
A length stop in that test is not a completed-answer or model-quality pass.
Recorded token rates are observations, not a speed comparison.

## Native consumer validation

A fresh isolated Osaurus Release development app consumed runtime
`6396715051b0d43ab8a112601eb0d6940330c54d` from consumer
`f6278f5a873accf3b7c415a10222498495589147`. The real agent editor saved custom
seed 42 and explicit reasoning None; navigation and normal app restarts
preserved the values. Eleven actual Chat UI requests completed five bounded
workflows: calculator, dependent result, restart continuation, an actual
45-second shell command followed by a dependent calculator, and another
restart continuation with normal tree/adaptive planning. Exact tool call IDs,
arguments, consumed results, visible answers and positive native token rates
are retained. All five workflows passed; no protocol markers leaked. Seven
internal 48-token load warmups are excluded from that score.

The first two app phases explicitly exercised chain/fixed-width diagnostics.
The last phase removed those flags and logged active tree verification and
widths 5 and 16. This is a bounded compatibility check, not a speed comparison.
The inherited bundle sampler remained temperature 1, top-p .95 and top-k 20;
the app's existing output-token budget was not changed.

Actual HTTP requests with seed 42, then 99, then 42 produced identical same-seed
visible output and different alternate-seed output when all three restored
the same 21-token disk prefix and prefilled the same seven remaining prompt
tokens. All three naturally stopped with visible answers and no reasoning.
Native rates were 39.43, 43.22 and 39.19 token/s. The last content-to-EOF gap was
under .25 ms in each of these three streams.

The earlier cold/warm triple **failed** exact same-seed equality: its first
request prefilled all 28 prompt tokens, while the later requests restored
21. That failure remains recorded separately. This fix does not establish
numerical equality across different cache/prefill plans or diagnose a cache
serialization defect.

Telemetry reported 64 layers (16 KV and 48 SSM), disk-backed restore, paged KV
off and zero TurboQuant KV layers. Typed payload headers contain BF16
convolution and F32 recurrent companion state. This is not full numerical cache
parity proof. Physical footprint peaked at 22.40 GB, exceeding primary weights
of 15.30 GB: the low-RAM gate **failed**. Owned loading remains unchanged.
The three owned app/sampler pairs exited normally. All 48 protected agent/tool
source hashes remained unchanged. Consumer source/pin contract suites passed
115 checks; they do not substitute for the native proof above.

## Remaining gates and limits

Normal PR/consumer merges remain pending. Hosted Jinja, both Linux SwiftPM and
four Linux container jobs passed on the tested runtime head. The advisory
repository formatter job failed on existing formatting; the new test has no
formatter diff and iterator formatting suggestions do not overlap changed
lines. Optional self-hosted Mac and CUDA jobs remain queued, not passed. Local
actual macOS module/test/app proof is recorded above.

Adaptive planning uses measured costs, so a seed alone is not a promise of
bit-identical output when the verification plan changes. The regression pins
the plan rather than changing production defaults.

This fix does not resolve or establish the cause of the earlier long-reasoning
length-stop failure. Private chain arithmetic controls and sampled tree-capable
requests are separate evidence surfaces. Neither disabling optimized kernels
nor narrowing the production width ladder is justified by those controls.
No agent loop, orchestration, in-app tool, prompt, reasoning marker, sampler
default, release or tag changes are part of this fix.

Private evidence index: `DFLASH-CHAIN-SEED-BASELINE-RED-20261105.json` and
`qwen27-concurrency/seeded-request-regression/`, under the existing
`goal-storage-cache-resume-20261104` evidence directory. Failed harness/build
attempts and the original native quality/resource failures remain retained.
Native evidence: `seeded-chain-native/NATIVE-RECEIPT.json`, the actual exported
`build29-seeded-default-native.jsonl`, raw SSE events and app resource logs in
the same private directory. The disposable profile retains the explicit test
settings; the user's ordinary dev app and profile were left alone.
