# Qwen3.6 Native MTP Matrix - 2026-05-17

This note records the current Swift engine status for the six local Qwen3.6
MTP/VL bundles. It is intentionally not a production-ready claim. Rows that
only load, only hit a cache, or only pass by exhausting a token budget are not
counted as working.

Local artifact roots:

- `docs/local/qwen36-mtp-matrix/20260517T042923Z-six-variant-matrix/`
- `docs/local/qwen36-mtp-matrix/20260517T043859Z-vl-mtp-strict-rerun/`

`docs/local/` is gitignored; the artifact paths are local evidence references.

## Runtime Contract

- Native MTP is detected from real `mtp.*` tensor/config evidence, never from
  model names. All six bundles report `bundleHasMTP=true`,
  `mode=preserved_enabled`, and `complete=true`.
- MTP is still not auto-launched by name or bundle family. `canAutoLaunch=false`
  remains correct until Osaurus explicitly opts into native MTP for a row that
  has passed the relevant text/VL/cache gates.
- Hybrid/Mamba/GatedDelta-style verifier cache state is not sliceable by
  keeping an all-or-nothing verifier cache. For those caches, Swift now uses a
  correctness-first sequential verifier path: advance the backbone through the
  primary and accepted drafts one token at a time, stop before rejected draft
  state, and only draft again from the committed state.
- `BatchEngine.submit` intentionally rejects raw batched native MTP. VL+MTP proof
  must use `BatchEngine.generate` or `Evaluate.generate`, which route through the
  exclusive solo native-MTP iterator. Osaurus should wire native MTP as an
  exclusive generate path until true batched/paged native-MTP scheduling lands.
- The VL cache matrix now fails zero-token, empty-visible, and max-token-exhausted
  rows. A row that only "passes" because `max_tokens` ended it is a failure.

## Code Changes Behind This Matrix

- `JangLoader` shape inference now handles Qwen3.6 35B JANG2K mixed quant
  layouts: 3/5/6-bit group-128 candidates, hidden-input routed projections,
  linear-attention value-dim hints, and hidden-size guided shared-expert gate
  dequantization.
- Qwen3.6 MTP/VL loading isolates MTP-only tensors from base AR load while
  preserving them when `nativeMTP=true`.
- Qwen3.6 base/MTP norm-shift detection ignores MTP-side tensors when deciding
  whether base norms need sanitizing.
- Cached multi-token forwards now create offset-aware causal masks when the
  cache already has an offset.
- Native MTP verifier semantics now support partial commit on hybrid caches
  without storing rejected draft state.
- `VLBench.runChatCacheMatrix` now exercises the real native-MTP generate path
  and fails length-cap pseudo-passes.

## Text Decode Matrix

All six text rows below produced the exact visible answer `1, 2, ... 50`,
`stop=stop`, `unclosedReasoning=NO`, `loop=NO`, and no leaks.

| Bundle | MTP tensors | Vision tensors | AR tok/s | MTP D3 tok/s | MTP footprint MiB | Cache repeat |
|---|---:|---:|---:|---:|---:|---|
| 27B JANG_4M | 31 | 333 | 26.5 | 22.3 | 3152 | PASS, disk+SSM hit |
| 27B MXFP4 | 23 | 333 | 29.7 | 25.1 | 15091 | PASS, disk+SSM hit |
| 27B MXFP8 | 23 | 333 | 16.8 | 14.8 | 1516 | PASS, disk+SSM hit |
| 35B JANG_2K | 44 | 333 | 119.7 | 79.8 | 8317 | PASS, disk+SSM hit |
| 35B MXFP4 | 42 | 333 | 104.8 | 81.7 | 18678 | PASS, disk+SSM hit |
| 35B MXFP8 | 31 | 333 | 77.9 | 63.3 | 1309 | PASS, disk+SSM hit |

Important speed note: D3 is currently coherent but not faster than AR on these
Swift rows. The correctness path is working; the speed path still needs
compiled/small-M verifier work and possibly a capture/commit verifier cache path
instead of sequential verification for hybrid caches.

## Multi-Turn Reasoning

The MTP D3 production multi-turn/reasoning gate still fails on every bundle.
The common failure mode is thinking-enabled prompts spending the answer inside
reasoning and then hitting the short budget with no visible answer. This is not
papered over by forcing a close tag or clamping sampling. It remains a real
template/reasoning/runtime policy bug to fix before a production claim.

| Bundle | Result |
|---|---|
| 27B JANG_4M | 3 pass / 4 fail |
| 27B MXFP4 | 4 pass / 3 fail |
| 27B MXFP8 | 6 pass / 1 fail |
| 35B JANG_2K | 4 pass / 3 fail |
| 35B MXFP4 | 3 pass / 4 fail |
| 35B MXFP8 | 5 pass / 2 fail |

Each row still recorded disk L2 plus SSM companion cache hits, so the cache stack
is being exercised even when the visible reasoning behavior fails.

## VL + MTP Matrix

After routing native MTP through `BatchEngine.generate`, four of six strict VL
rows pass with same-media disk hits, different-media misses, and coherent
follow-up answers.

| Bundle | Strict VL+MTP result | Finding |
|---|---|---|
| 27B JANG_4M | PASS | Red/blue gradient described; replay hit; follow-up `Red and blue.` |
| 27B MXFP4 | PASS | Red/blue gradient described; replay hit; follow-up `Red and blue.` |
| 27B MXFP8 | PASS | Red/blue gradient described; replay hit; follow-up `Red and blue.` |
| 35B JANG_2K | FAIL | Cold image response exhausted 96/96 tokens; length-cap pass rejected. |
| 35B MXFP4 | FAIL | Cold image response looped and exhausted 96/96 tokens; length-cap pass rejected. |
| 35B MXFP8 | PASS | Red/blue gradient described; replay hit; follow-up `Red and blue.` |

The earlier six-variant matrix showed all VL+MTP rows as zero-token failures
because the bench used `BatchEngine.submit`, which rejects native MTP by design.
That was a harness-path bug. The strict rerun is the current evidence.

## Still Open Before Production

- Native MTP speed is coherent but slower than AR on the tested text rows. Do
  not claim the 45-50 tok/s target is met for 27B, and do not claim a speedup
  for 35B despite high absolute tok/s.
- Reasoning-on mode must be fixed at the template/runtime boundary so answers
  appear in visible content without forced closing tags or hidden output moves.
- 35B JANG2K and 35B MXFP4 VL+MTP need root-cause work. The strict bench now
  blocks their length-exhausted/looping outputs instead of accepting them.
- Raw batched native-MTP scheduling is not implemented. Osaurus must route MTP
  as an exclusive generate path or keep MTP disabled for concurrent batched
  server mode until the scheduler owns draft/verify/cache state per slot.
- The sequential hybrid verifier is the correctness path. The future speed path
  needs a measured capture/commit implementation with per-accepted-prefix cache
  commits, plus row-vs-step equivalence tests for MRoPE/VL and SSM state.
