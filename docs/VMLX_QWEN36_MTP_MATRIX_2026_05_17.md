# Qwen3.6 Native MTP Matrix - 2026-05-17

This note records the current Swift engine status for the six local Qwen3.6
MTP/VL bundles. It is intentionally not a production-ready claim. Rows that
only load, only hit a cache, or only pass by exhausting a token budget are not
counted as working.

Local artifact roots:

- `docs/local/qwen36-mtp-matrix/20260517T042923Z-six-variant-matrix/`
- `docs/local/qwen36-mtp-matrix/20260517T043859Z-vl-mtp-strict-rerun/`
- `docs/local/qwen36-mtp-current/*-prod-budget384/`
- `docs/local/qwen36-mtp-current/20260517T124945Z-35b-mxfp4-vl-mtp-budget384/`

`docs/local/` is gitignored; the artifact paths are local evidence references.

## Runtime Contract

- Native MTP is detected from real `mtp.*` tensor/config evidence, never from
  model names. All six bundles report `bundleHasMTP=true`,
  `mode=preserved_enabled`, and `complete=true`.
- MTP is still not auto-launched by name or bundle family. `canAutoLaunch=false`
  remains correct until Osaurus explicitly opts into native MTP for a row that
  has passed the relevant text/VL/cache gates.
- Hybrid/Mamba/GatedDelta-style verifier cache state is not sliceable by
  keeping an all-or-nothing verifier cache. The default correctness path is
  sequential repair: advance the backbone through the primary and accepted
  drafts one token at a time, stop before rejected draft state, and only draft
  again from the committed state. The speed path remains opt-in
  `VMLINUX_NATIVE_MTP_HYBRID_VERIFY=chunk_commit` for greedy rows, but
  non-greedy exact-pq over hybrid SSM now always routes to `sequential_repair`
  because the chunk verifier produced incoherent residual-correction output on
  `Qwen3.6-35B-A3B-MXFP4-MTP`.
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

The first MTP D3 production multi-turn/reasoning gate used a short budget and
failed visible-answer checks when thinking-enabled prompts naturally spent the
early output inside reasoning. That was not fixed with forced close tags,
sampler clamps, or output movement.

Fresh MXFP reruns with `BENCH_MAX_TOKENS=384` now pass the production gate:

| Bundle | Result | Artifact |
|---|---:|---|
| 27B MXFP4 | 7/7 PASS | `docs/local/qwen36-mtp-current/20260517T124139Z-27b-mxfp4-prod-budget384/prod_mtp_d3_chunk_budget384.log` |
| 27B MXFP8 | 7/7 PASS | `docs/local/qwen36-mtp-current/20260517T124237Z-27b-mxfp8-prod-budget384/prod_mtp_d3_chunk_budget384.log` |
| 35B MXFP4 | 7/7 PASS | `docs/local/qwen36-mtp-current/20260517T124323Z-35b-mxfp4-prod-budget384/prod_mtp_d3_chunk_budget384.log` |
| 35B MXFP8 | 7/7 PASS | `docs/local/qwen36-mtp-current/20260517T124351Z-35b-mxfp8-prod-budget384/prod_mtp_d3_chunk_budget384.log` |

Each fresh row used bundle defaults
`temp=1.000 topP=0.950 topK=20 minP=0.000 rep=nil`, D3 native MTP, explicit
`chunk_commit`, L2 disk cache, and SSM companion state. Every row emitted
visible answers, normal stops, no loop/leak, a disk hit, and an SSM hit.

Growing-chat exact-pq rerun after the hybrid verifier policy fix:

```text
docs/local/qwen36-mtp-current/20260517T131024Z-35b-mxfp4-growing-chat-mtp-d3-exact-postfix/growing_chat_mtp_d3_bundle_defaults_postfix.log
docs/local/qwen36-mtp-current/20260517T131050Z-mxfp-growing-chat-mtp-d3-exact-postfix/
```

Those rows use bundle defaults plus `BENCH_GROWING_NATIVE_MTP_DEPTH=3` and even
with `VMLINUX_NATIVE_MTP_HYBRID_VERIFY=chunk_commit` they report
`verifierMode=sequential_repair`. All four MXFP rows stop normally, answer
`vmlx-cache-green` across two turns, hit a canonical growing-chat disk prefix
before turn 2, and increment SSM companion hits. The matched prefix is the
real rendered history boundary (`18/53`), not a fabricated full-prompt hit.

JANG_4M and JANG_2K were not rerun in this current MXFP-only pass. The earlier
short-budget failures remain historical evidence only for those old settings,
not proof that the current runtime needs fake reasoning guards.

## VL + MTP Matrix

After routing native MTP through `BatchEngine.generate`, five of six strict VL
rows pass with same-media disk hits, different-media misses, and coherent
follow-up answers.

| Bundle | Strict VL+MTP result | Finding |
|---|---|---|
| 27B JANG_4M | PASS | Red/blue gradient described; replay hit; follow-up `Red and blue.` |
| 27B MXFP4 | PASS | Red/blue gradient described; replay hit; follow-up `Red and blue.` |
| 27B MXFP8 | PASS | Red/blue gradient described; replay hit; follow-up `Red and blue.` |
| 35B JANG_2K | FAIL | Cold image response exhausted 96/96 tokens; length-cap pass rejected. |
| 35B MXFP4 | PASS | Fresh 384-token strict rerun described the red/blue gradient, replay hit `84/84`, different-media miss was correct, and text-only follow-up returned `Red and blue`. |
| 35B MXFP8 | PASS | Red/blue gradient described; replay hit; follow-up `Red and blue.` |

The earlier six-variant matrix showed all VL+MTP rows as zero-token failures
because the bench used `BatchEngine.submit`, which rejects native MTP by design.
That was a harness-path bug. The 96-token strict rerun remains valid evidence
for 35B JANG_2K. The 35B MXFP4 rerun with `BENCH_MAX_TOKENS=384` is the current
evidence for that MXFP row, because the 96-token row was a budget artifact rather
than a demonstrated runtime failure.

## Still Open Before Production

- Native MTP speed in the initial six-variant D3 text matrix is coherent but
  slower than AR. Separate optimized count-prompt speed rows reach the 45-50
  tok/s target for 27B JANG_4M / 27B MXFP4 and higher absolute throughput for
  35B MXFP4/MXFP8; do not use the initial matrix alone as a speedup claim.
- Production gates must use enough max tokens for thinking-enabled prompts.
  Short-budget visible-answer failures are not runtime failures by themselves;
  they must be rerun with a sufficient budget before changing code.
- 35B JANG2K VL+MTP still needs root-cause work. The strict bench blocks its
  length-exhausted output instead of accepting it.
- Raw batched native-MTP scheduling is not implemented. Osaurus must route MTP
  as an exclusive generate path or keep MTP disabled for concurrent batched
  server mode until the scheduler owns draft/verify/cache state per slot.
- The sequential hybrid verifier remains the stochastic correctness path.
  Explicit `chunk_commit` remains a greedy speed path only until a per-artifact
  equivalence gate proves stochastic chunk probabilities are safe.
