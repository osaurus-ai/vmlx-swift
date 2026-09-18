# Qwen3.5 / Bonsai2 media-prefill follow-up

Status: candidate implemented, no measured speedup or current live-model proof.
Base:87a686e929c4bbd9e99728126b30095339d0df5b. Separate from packed-load PR481.

## Source mechanism

`Qwen35.prepare` chunks pure text but runs all media embeddings through the
language trunk in one forward, regardless of the requested prefill window.
The retained Bonsai8049-token image/history run exceeded its20GiB supervisor
cap. That observation motivates bounding the language prefill; it does not
prove every byte of that peak came from this path.

Splitting raw media inputs into ordinary text chunks would be wrong. The
language model computes M-RoPE coordinates and the decode delta from the full
image/video grid. Its hybrid cache contains both attention KV and GDN conv /
recurrent state. Media scattering already separates image and video rows in
conversation order; preserve that correction.

## Bounded implementation

1. Encode/scatter media exactly once using the existing path.
2. Resolve full-prompt M-RoPE positions exactly once with the existing helper,
   retaining the resulting decode delta. Slice those positions along with
   token IDs and merged embeddings; never recompute a grid for each chunk.
3. Evaluate each prefix chunk's KV and GDN state before reporting actual
   progress/releasing transient allocations. Keep final-tail logits lazy as
   on the existing text path. Honor cancellation before media work, between
   chunks, and before the final tail.
4. Keep unchunked behavior for empty caches, nonpositive/large windows and masks for which
   the existing position resolver cannot produce full positions. Do not alter
   model-native samplers, templates, schemas, media order or disk-cache keys.

## Proof required

- Tiny actual vision + hybrid forward: image and video-first/image-later;
  chunks ending before/inside/after media tokens; requested-window progress.
- Same weights, one-shot vs chunked last logits, attention KV, conv/recurrent
  state and subsequent decode logits; zero/nonzero initial cache offsets.
- All-ones2D mask parity; cancellation before work and between chunks; fallback
  behavior and no chunking when the full prompt fits.
- Then a repinned app, both Bonsai storages sequentially, native image/tool
  history/cache continuation and the retained long-image workload under the
  unchanged resource guard. Measure load separately from residual prefill,
  TTFT, decode and physical footprint. Tiny tests are not full-model proof.

Current full-model runtime authorization for the follow-up is pending.
The current component result is recorded below; no native model was loaded.

Candidate and actual tiny-vision regression source parse successfully with
`swiftc -frontend -parse`; this does not typecheck or execute the tests.
`Qwen35MediaPrefillTests` covers image/video-first inputs, windows3/4/8,
zero/nonzero initial offsets, a2D mask, KV/GDN/next-decode comparisons and
Stop before work/at the final-tail boundary. Initial source-parse evidence is
superseded by the executed component result below, not by a full-model claim.

Source review also added a cache-free fallback regression: `castCache([])`
means no attention or recurrent state survives between forwards, so slicing
that public prepare call would discard the prefix. The candidate only chunks
when a cache is supplied. Exact-window and no-cache cases retain full logits.

## Executed component result,2026-09-18

SOURCE EVIDENCE:a30e86a99c72e870a78354d6d2a8d3890e983051, based on
87a686e9. `Qwen35.prepare` media branch and `Qwen35MediaPrefillTests` are the
candidate. No packed-loader changes are included in this branch.

LIVE EVIDENCE: serialized, bounded Max2 component run03:31:55–03:36:40PDT;
artifacts under `/Users/eric/vmlx-private-evidence/runtime-followup-2026-09-18/`:

- `SWIFTTEST_QwenMediaA30eComponents1__033155.log`: build completed,
  4XCTest cases in ChunkedPrefillVLMTests passed, then14Swift Testing functions
  in4suites passed, with0failures/0skips. These use tiny synthetic weights,
  actual vision/GDN forward operations and real cache state, not full bundles.
- `media-a30e86a99c72e870a78354d6d2a8d3890e983051-one-swift-testing.xml`
  contains the14Swift Testing results; the4XCTest results are in the log.
  SwiftPM did not emit the unsuffixed XML, so do not cite it as an artifact.
- Image and video-first/image-later parity passed at windows3/4/8 with
  offsets0/3 and an all-ones2D mask on the latter. Last logits, attention KV,
  GDN convolution/recurrent states and two subsequent decode-token logits
  matched within1e-4+1e-4*referenceScale. Long270-token history/window128,
  progress, cancellation, cache-free and fitting/disabled-window cases passed.
- Existing Hadamard routing, hybrid disk reopen/continuation and VLM GDN
  batch-offset/compiled-tail tests also passed. This does not prove a real
  media cache hit or full-model coherence in the app.
- Guard exit0, peak tracked physical footprint3.02GiB, swap1.67GiB unchanged,
  final cleanup verified0group/0tracked/0watchdog survivors. Exact source,
  vendor, lockfile and Metal artifact identities checked before/after.

PARTIAL: no actual Bonsai prefill time, whole-app peak-memory improvement or
native long-image/tool-history proof is claimed. The retained8049-token
guard failure is still open until the same workload is rerun in the app.
