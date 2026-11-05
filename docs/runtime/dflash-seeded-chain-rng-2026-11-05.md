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

## Remaining gates and limits

Actual dev-app consumer validation, PR CI and normal merge remain pending.
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
