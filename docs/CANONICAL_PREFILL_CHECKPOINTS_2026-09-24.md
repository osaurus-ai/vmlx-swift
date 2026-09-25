# Durable canonical prefill checkpoints

Warm or restarted chats can reuse ordinary history state for generation but still
recompute the entire prefix when sealing a cache boundary. Retain a separately
namespaced full-chunk prefill checkpoint so boundary reconstruction computes only
remaining canonical chunks and the final tail.

## Correctness contract

- Only a cold, full-input prefill or a verified canonical continuation establishes
  provenance. An aligned offset alone is insufficient. The original chunk size,
  token prefix, model identity and request salt must match.
- Preserve standard KV dtype and the original prefill partition. The batch path requires
  at least one rotating KV layer and permits ordinary full KV companions; media,
  masks, custom preparation keys and recurrent companions are excluded.
- Ordinary history and canonical snapshots have separate payload identities.
  Optional nullable `replay_chunk_size` metadata extends schema v2 without a new
  entry kind or version bump. Older readers can continue ordinary writes and
  eviction. Extension failure disables optional checkpoint persistence.
- Persist once after normal resume stores and only on native completion. Account
  bytes under the existing combined quota lock. Skip optional writes that cannot
  fit beside the active resume row, including a conservative header allowance.
- Protect at most one current canonical checkpoint per chain. Superseded optional
  rows are disposable. Cold optional checkpoints yield before active rewind rows;
  active optional checkpoints yield before real cold chat tips and stable roots.
  The hard cap wins. Optional rows never count as actual resume points or clear
  lost-tip warnings.
- Reject malformed or incompatible restores and permit their replacement. Candidate
  rejection must not decrement unrelated normal cache-hit telemetry.

This does not change the ordinary generation prefix lookup, quantization,
sampling, templates, memory limits or cache topology. Reusing these checkpoints
for initial generation itself remains separate work.

## Validation

Evidence bundle: `canonical-checkpoints-r47` under the private Raptor proof archive.
All model runs were serialized on the authorized M5 Max, with unchanged host and
physical-footprint guards. No build ran alongside a model benchmark.

- 168 tests in 10 suites passed. Coverage includes additive schema migration,
  legacy writes, read-only and contended indexes, future schema rejection,
  namespace isolation, malformed metadata, dtype preservation, rejection repair,
  write suppression, shared quota accounting, ownership, pressure warnings,
  eviction order, and existing rotating boundary replay regressions.
- Fresh-engine fixtures restore 96 canonical tokens and reconstruct 165 using
  multiple 16-token chunks, matching independent cold state exactly. Existing
  replay work counts decrease from 231/249 to 199/217 with unchanged state checks.
- Final release benchmark build completed with no source drift; binary SHA-256:
  `3e51757c3b216376cdd9e59382bba384bbc17245ce925596775121111230f8a6`.
- Installed Raptor 0.6 JANG_6M, 10,031-token cold prompt and 10,058-token follow-up:
  final 2.2 GiB quota ABBA completed eight native-stop turns. Full outputs and all
  common retained ordinary checkpoint tensors match; the latest history resume
  boundary is retained and indexed usage remains under quota. Candidate physical
  footprint peaks below 4.44 GB. Checkpoint restore skips 9,728 tokens while
  preserving 512-token chunking.
- Final 2.2 GiB means: two-turn stream completion 9.771 -> 7.519 seconds (23.05%
  less), warm finalization 3.124 -> 0.824 seconds (73.64% less). Two samples per
  arm; these are whole-turn/cache timings, not a raw decode speedup.
- Earlier 8 GiB ABBA: warm finalization 3.067 -> 0.917 seconds, but cold
  finalization added 343 ms on average and total two-turn time was 9.714 -> 9.822
  seconds: no aggregate improvement. The first candidate's 5.324-second prefill
  outlier remains included and unexplained.
- Earlier fresh-process pair: combined finalization 7.871 -> 1.568 seconds, with
  exact full outputs and ordinary state. This is one pair, not a statistical claim.
- Bounded other-family checks completed nine native-stop turns across dense full
  KV (Nanbeige 4.2), MoE/KDA (Raptor 0.5), and dense convolution (LFM2.5), including
  accepted disk restores after process restart. These and the 8 GiB/restart rows
  precede the final quota-priority refinement; the final refinement is covered by
  the 168-test suite and final 2.2 GiB full-model ABBA.

Recorded artifacts: `runtime-test-r10.log`, `native-build-r3-receipt.json`,
`pressure-final-review.json`, per-run `comparison.json`, `native-abba-review.json`,
`restart-review.json`, and `cross-family/review.json`. Earlier failed builds and
fixtures remain in the private archive with corrections recorded.

## Limits and app integration

Engine qualification is distinct from Osaurus consumption. An Osaurus pin update,
exact-build UI checks, restart/Stop/cache controls and required app CI remain
necessary before claiming this change is proven in the app. No media, tool-call,
all-chip, low-RAM, generic-family speedup or release-to-current speed claim follows
from these bounded text benchmarks. No release is authorized.
