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

## Cache continuation findings after v3

Both v3 serial qualification cases failed: the MTP agent reached its output
limit and non-MTP returned no visible changed-image answer. A fetched disk
file was being counted as a hit even though its complete state was not
restored. `glm-cache-restore-before.log` reproduces two underlying defects:

1. Assigning populated indexed state into a fresh GLM cache used the
   destination's zero array count, seating keys as the indexer and losing KV.
2. The disk serializer skipped that custom cache but seated recurrent layers.
   Restore returned zero while leaving a Mamba layer at offset four. The
   caller then prefills the entire prompt into partly restored state.

The follow-up corrects indexed-state layout and introduces an opt-in
`DiskCacheStateProviding` contract. Model-owned tensors, metadata, offset,
and a versioned runtime-mode identifier round-trip as one record. Missing
arrays, incompatible modes and mismatched companion lengths are refused.
The identifier enters runtime topology tags and thus Osaurus cache keys,
separating complete records from the previous lossy format. No model-name
list is involved in selecting this persistence path.

Generation calls now set `requirePromptBoundary: true` at every production
disk-restore call site: TokenIterator, BatchEngine, NativeMTP, DFlash2, block
diffusion and paged companion restore. A staged pass rejects incomplete or
zero-boundary records before live layers mutate, then applies a validated
record in place to preserve retained layer references. Low-level snapshot
callers retain their existing valid zero-offset round-trip behavior.

An initial transaction that replaced live cache objects broke retained layer
references; the TQ and ZAYA tests caught this and the failing logs remain.
The first attempt to require a positive boundary for low-level snapshots
also failed existing empty-state tests; separating snapshot and generation
contracts preserves those checks. Neither failure was promoted as a pass.

The Mamba malformed-file test also rewrote its input file before lazy tensor
reads finished. Materializing those arrays before rewriting metadata allows
the whole ten-test suite to complete; no corruption assertions were removed.

`glm-custom-codec-roundtrip.log` contains thirteen passing focused tests,
including actual safetensors media-cache continuation with numerical logits
comparison, damaged payloads, and runtime-mode refusal.
`glm-input-mtp-live-v4.json` and `glm-input-nonmtp-live-v4.json` each passed
all seven real-image requests in the intermediate build. Peak physical
footprint remains full-model size; these do not qualify low-RAM operation.
The source and executable are identified in `glm-input-live-receipt-v4.json`.
Final source adds in-place restoration, topology isolation and explicit
accepted-restore tracing; its qualification remains pending.

PR #475 is a draft. Its initial lint job failed with repository-wide
formatter changes beyond this patch (`glm-pr475-lint-failure.log`). This is
not a green CI or merge-ready claim.

The final shared-cache run (`glm-final-shared-v7-receipt.json`) contains
61 passing tests: 13 GLM input/copy, 20 TQ serializer, eight ZAYA, ten Mamba,
five architecture damage-matrix and five QSA persistence tests. An empty
Mamba child in CacheList is now tagged as skip when it has neither tokens
nor tensors; the existing empty-composite regression exposed this earlier
serializer defect. No normal-generation or empty-snapshot checks were
removed. The subsequent evidence below supersedes this pending status for engine
`ffee904d4f4f0680aa6a2c39c3dedff9cedae010`.


## September 15 follow-up and process-limit regression

At engine `ffee904d4f4f0680aa6a2c39c3dedff9cedae010`, the additional
TokenIterator progress and hybrid boundary suites passed four tests, bringing
the focused cache/input total to 65 (`glm-final-pipeline-regressions.json`).
The 11-case representative run covered all ten installed architectures:
ten passed and ZAYA's changed-image history failed. Both complete GLM cases
passed all seven image/stream/agent/history checks and logged three accepted
restores each (`glm-v7-architecture-receipt.json`).

A fresh public-dependency build at Osaurus `a4d3405223bfae79b82ee496e414b44585b75c5c`
then repeated the full installed sweep: 79 inventoried, 33 selected, 28 passed,
four failed, and one GLM MTP case was deliberately interrupted. Five declared
vision bundles remained rejected by installed evidence; none were excluded by
header audit. The failures were Ornith 9B 2D, both ZAYA quantizations, and
CRACK Qwen3.8 27B 2D. These were already failing models; this does not erase
previous failures or qualify every installed bundle.
`glm-public-full-matrix-receipt.json` retains each report and the interruption.

The public non-MTP GLM case returned all seven expected answers, but early
requests ran at about 0.67 tok/s and later history at 15.6–15.9 tok/s.
The Release UI also returned Red and Blue for actual changed attachments;
a retained-image response was cancelled after 171 reasoning tokens with no
visible answer. The cancellation and next-image recovery were exercised
through the real controls. Peak UI physical footprint was 99,902 MiB; this
is not low-RAM qualification. Evidence: `glm-ui-v7-receipt.json`,
`glm-ui-history-v7.json`, and `glm-release-ui-v7.log`.

The UI model switch from LFM to GLM exposed a separate policy application
mismatch: `LoadBundleFacts.requiresUncappedResidentPools` selected GLM, but
`ModelFactory` applied the corresponding process-global reset only to DSV4.
Resolving `.unlimited` produces no integer assignment, leaving LFM's 70% cap
active. The added parameterized test reproduces six failing assertions for
GLM's two aliases, while DSV4 takes the reset path
(`glm-resident-policy-red.log`). The loader now uses the existing shared
property for both policy resolution and application, and records previous
and applied limits when cache tracing is enabled. The native ceiling formula
and wired-memory reserve are unchanged; ordinary capped models retain their
existing behavior.

This mismatch does not explain the whole CLI slowdown: both compared CLI
builds already had the native 95% ceiling. The 70% observation belongs to the
UI model-switch run. `glm-stale-limit-investigation.json` records that
correction. No new throughput or low-RAM claim follows from this change.
The final process-policy candidate still requires its focused tests, public
pin rebuild, complete installed matrix and Release UI model-switch proof.

The ZAYA native-processor diagnostic is not included. It failed all seven
answer checks for both installed quantizations, with locked dependencies and
no test sampler overrides (`zaya-native-locked/results.json`). Publisher
processor/template/generation metadata matches the installed metadata, but
that comparison does not prove weight or runtime correctness.

The policy candidate re-ran all 65 cache/input regressions and nine wired-memory
and safety-level tests successfully (`glm-policy-regressions.json`). Its first
full LoadConfiguration run passed the new live MLX process-limit assertions
but exposed one pre-existing source-string assertion still expecting a private
helper. That helper became internal in `163798115` on September 5. The
assertion now checks the same signature without the obsolete visibility;
all dtype/residency assertions remain (`glm-resident-policy-green.log`).

The complete LoadConfiguration rerun passed 42 tests in two suites
(`glm-resident-policy-green-2.log`). Public-pin and UI evidence for this
additional process-policy correction remains pending.
