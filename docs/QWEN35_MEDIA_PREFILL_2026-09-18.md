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
4. Keep unchunked behavior for nonpositive/large windows and masks for which
   the existing position resolver cannot produce full positions. Do not alter
   model-native samplers, templates, schemas, media order or disk-cache keys.

## Proof required

- Tiny actual vision + hybrid forward: image and video-first/image-later;
  chunks ending before/inside/after media tokens; requested-window progress.
- Same weights, one-shot vs chunked last logits, attention KV, conv/recurrent
  state and subsequent decode logits; zero/nonzero initial cache offsets.
- Full2D mask parity; cancellation before work and between chunks; fallback
  behavior and no chunking when the full prompt fits.
- Then a repinned app, both Bonsai storages sequentially, native image/tool
  history/cache continuation and the retained long-image workload under the
  unchanged resource guard. Measure load separately from residual prefill,
  TTFT, decode and physical footprint. Tiny tests are not full-model proof.

Current runtime authorization for the follow-up is pending. The already
running handoff app build is serialized ahead of Metal component tests.

Candidate and actual tiny-vision regression source parse successfully with
`swiftc -frontend -parse`; this does not typecheck or execute the tests.
`Qwen35MediaPrefillTests` covers image/video-first inputs, windows3/4/8,
zero/nonzero initial offsets, a2D mask, KV/GDN/next-decode comparisons and
Stop before work/at the final-tail boundary. Execution remains pending.
