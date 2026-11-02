# Native generation contract: reasoning and block diffusion

Status: source changes and regression fixtures prepared; focused runtime tests pending.
This is a bounded follow-up to the all-family advanced-settings audit, separate
from the output-limit correction in PR #719. No real model inference, speed
qualification, UI qualification, release, or Studio deployment has been performed
for this change.

## Policy defects and origin

The former `MinimumReasoningFloor` automatically suppressed the reasoning closer
for 16 sampled tokens on the DSV4-0731 Low/High/Max rails. Its own source described
the purpose as preventing immediate closure after native adjacent thinking tags;
it was a rescue policy rather than a model-native sampling requirement. Both
`BatchEngine.submit` and `BatchEngine.generate` injected it. The public leading
suppression processor additionally applied an unrequested survivor-relative
probability cutoff of 0.05 and rewrote the caller's logit array through an alias.

The block-diffusion iterator similarly suppressed first-canvas, first-position EOS
to obtain nonempty answers. Its comment explicitly made the mutation of the
self-conditioning logits part of the rescue. The iterator's documented native
algorithm instead temperature-schedules decoder logits, samples and computes
entropy from them, reuses them as self-conditioning, then truncates the final
canvas at the first EOS. The rescue guard is removed; immediate native EOS is a
valid termination, including when it produces no visible answer. Such a real
model quality failure must be reported rather than masked.

## Change and compatibility

- Remove the automatic reasoning floor and both engine arming sites. The
  tokenizer, existing chat encoder, reasoning parser, explicit reasoning ceiling,
  and ordinary sampler controls remain intact.
- Retain the public `GenerateParameters.initialSuppressTokens`,
  `initialSuppressCount`, and `InitialSuppressTokensProcessor` APIs as deprecated,
  explicit-caller-only compatibility controls. Repository-wide Swift reference
  inspection found the removed floor to be their only automatic producer.
  Defaults remain an empty token list and a zero-length window. Explicit callers
  retain exactly their requested token/window mask; the hidden 0.05 cutoff is
  removed. No model family, prompt, or effort setting enables these controls.
- Preserve bundle/user `suppress_tokens` semantics while constructing the mask
  out of place so caller-owned logits and aliases are unchanged. Invalid token
  IDs are ignored as before. This allocates mask expressions proportional to
  vocabulary size and explicit token count; the performance impact has not been
  measured and this patch makes no speedup claim.
- Diffusion categorical sampling, argmax, entropy, and self-conditioning use the
  same unmasked native temperature-scheduled logits. Native EOS truncation,
  prompt/cache commit, output limits, and denoising schedule remain intact.

## Validation and remaining work

`NativeReasoningLogitsFocusedTests` covers unchanged default logits, immediate
closer eligibility, exact explicit suppression with unchanged survivor
probabilities, invalid IDs, caller aliases, leading-window expiration, and both
engine entry points' absence of automatic injection.

`BlockDiffusionEOSContractFocusedTests` uses a tiny constant-logit CPU fixture
through the real `BlockDiffusionTokenIterator`. It covers native EOS in the first
and later positions, unchanged source logits and prompt arrays, native
self-conditioning, and committing only content preceding EOS. It records fixture
token/s; these rates do not represent a real model or hardware performance proof.
The old tests that required rescuing EOS or forcing minimum reasoning are removed.

Completed: `swiftc -frontend -parse` for changed Swift files; `git diff --check`.
Pending: coordinated focused-test build/run, review and normal PR/CI/merge flow,
then consumer-pin and real-model/UI qualification where appropriate. Source
checks and synthetic fixtures cannot qualify every model family's behavior.

Separate source finding: `DeepseekV4ChatEncoder.swift` describes its DSV4-0731
Low/High/Max effort prefaces as enforcement prose, including a deliberate
deviation from the official Low rail. Those prefaces are unchanged in this
bounded decode-mask patch and require their own native-template contract audit.
Removing the floor does not establish that the remaining DSV4 prompt policy is
fully compliant.
