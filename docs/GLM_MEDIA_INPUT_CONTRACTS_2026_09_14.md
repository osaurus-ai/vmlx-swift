# GLM media input contracts

The installed-model audit in osaurus-ai/osaurus#2772 inventoried 79 bundles
and exercised all 33 admitted image bundles across ten architectures. It
exposed two GLM failures: a rank-three text input during warm-up and rejection
of the second image in a conversation. This follow-up addresses the GLM
processor/runtime contracts found by that broad audit.

## Changes and source trace

- `Glm5NextProcessor.prepare` emits an unbatched token sequence. Image and
  video prefill add the batch axis at their model-owned decoder call.
- `Glm5Next.prepare` also accepts the batched text fragments supplied by cache
  boundary splitting, then returns a flat tail to generic generation. It
  preserves token IDs and flattens the per-token mask. Unsupported ranks or
  multiple sequences are rejected at the throwing prepare boundary.
- Multiple images expand distinct original placeholders in conversation
  order, with matching concatenated patch tensors and grids. A placeholder
  count mismatch throws instead of dropping or misplacing an attachment.
- Prepared media declares its image placeholder token ID. The existing cache
  predicates can then distinguish a safe text suffix from a suffix that
  still contains image features. No cache safety predicate is relaxed.
- Decoded video frames provide their own size and timestamps; the processor
  no longer tries to read an AVAsset at `/dev/null` for this input form.
- The tiny text-only construction test expects four decoder layers. The
  previous five-layer assertion predates opt-in MTP construction (#414).

The processor's extra batch axis, single-image guard, and missing placeholder
metadata originate in #371 (`0237f63f9a11db5bd027a313347bf97d656d97f7`).
The short-text return of an unchanged batched fragment is in the chunked
prefill implementation from #448 (`64db8f984`).

## Current evidence

Evidence root:
`/Users/eric/vmlx-private-evidence/ornith-vision-2026-09-14/`.

Engine base `beb8176eee5ce878346c95e99d2b60c6ddacb826` has the same tree as
Osaurus's previous pin `5b0c8e6b8b29a7ead21fe785688bc0621580cc62`.
The diagnostic Osaurus host is
`3fd0e69a35d42c987eb80d841249ada0c2710c2b`, with a local engine dependency
override. All other Evals dependency pins match the earlier locked baseline.

| Evidence | Observed result |
| --- | --- |
| `glm-input-contract-before-five-2.log` | Processor rank failure, two-image rejection, and constant-zero logits from malformed short text. |
| `glm-cache-fragment-before-2.log` | Batched cache fragment reproduces `[1, 1, 2]` decoder input and zero logits. |
| `glm-media-boundary-before.log` | Missing placeholder metadata prevents safe-boundary capture and text-only suffix recognition. |
| `glm-input-contract-regressions-v3.log` | 27 tests in eight suites passed, including real Metal numerical prefill and ordered image patches. |
| `glm-input-mtp-live-v2.json` | Seven image/stream/agent/history requests returned expected visible colors; whole case failed exact replay cache reuse. |
| `glm-input-nonmtp-live-v2.json` | Same seven requests returned expected colors without the earlier crash; whole case failed exact replay cache reuse. |
| `glm-input-live-receipt-v3.json` | Exact current source-patch, source-file, and diagnostic executable hashes. |

The v2 runs contained no `forward failed` log entries. Peak physical
footprint was 97613.766 MiB for MTP and 99845.643 MiB for non-MTP. These
full-model-footprint rows do **not** qualify low-RAM behavior. Sampler values
were not overridden; the Vision suite used its existing 1024-token output
cap and retained normal-finish, visible-color, throughput, and cache checks.

The first v3 run overlapped a Release build and slowed under memory pressure.
It recorded a disk hit at the safe 55-token boundary, but was interrupted
and is not a passing qualification row. The process identities, memory
observation, and sample are retained in
`glm-v3-concurrent-run-interruption.json` and `glm-v3-wait.sample.txt`.

## Remaining gates

Repeat both complete GLM Vision cases serially after the build, then exercise
the final Release app with real attachments and retained/changed-image
history. Record the final source and binary identity before claiming this
follow-up qualified. Video has numerical frame-input coverage here, not a
real-weight video understanding claim.

The pre-existing nonthrowing GLM generation overload still substitutes zero
logits for unexpected decoder errors. This change prevents the reproduced
valid-input shape error from reaching that fallback; it does not claim a
general throwing-generation error contract. Other installed-model failures
from the broad audit remain visible in the Osaurus qualification document.
