# DFlash 2

Block-diffusion speculative decoding. One drafter forward proposes a
whole block of tokens; one target forward verifies all of them.

Upstream: [z-lab/dflash](https://github.com/z-lab/dflash) ·
[blog](https://inco.ai/blog/dflash2/) · checkpoints under
[`z-lab/dflash-2`](https://huggingface.co/collections/z-lab/dflash-2).
The Swift code here is a port of the authors' own MLX implementation
(`dflash/model_mlx.py`), which is what the parity tests compare against.

## What it is

A DFlash 2 drafter is a 5-layer Qwen3-shaped transformer that ships
**no embedding and no LM head** — it borrows the target's. What it
does own:

| tensor | role |
|---|---|
| `fc`, `hidden_norm` | project the target's hidden states at `target_layer_ids` into the drafter's K/V context |
| `layers.N.self_attn` | queries from the drafted block, K/V from the context **and** the block |
| `layers.N.{attention,mlp}_conv` | two-tap dynamic causal conv, per-token kernel from `kernel_projection` plus a learned `base_kernel` |
| `candidate_selector` | low-rank bigram scorer that traces one path through the per-position candidate sets |

Three properties follow from that, and each one shaped the port:

1. **It reads the target's internals.** `target_layer_ids: [5, 19, 33,
   47, 61]` for Qwen3.8-27B. The target must conform to
   `HiddenStateCaptureModel`, and every verify forward hands the
   drafter the hidden states of the tokens it just committed.
2. **The block is bidirectional.** `is_causal: false`, so every
   position in the block attends to every other one. That is the
   "diffusion" — the whole block is drafted in one pass rather than
   left to right.
3. **Acceptance is a path, not a token.** The drafter keeps
   `selector_top_k: 16` candidates per position and threads one
   coherent path through them; the target then verifies that path.

## Losslessness

Greedy DFlash 2 emits exactly what the target would have emitted alone.
That is not a quality claim, it is the correctness invariant, and
`DFlash2LosslessSmokeTests` asserts it as string equality against the
plain iterator on a real Qwen3.8-27B. Under sampling,
`DFlash2Sampling.acceptSampled` is standard speculative rejection
sampling over the candidate-restricted draft distribution, so the
emitted distribution is the target's.

The iterator **refuses** rather than approximates when the request
cannot honour that: repetition/presence/frequency penalties, token
suppression, reasoning budgets and `min_p` all need per-step logits that
a pre-drafted block does not have. Those requests fall back to ordinary
decoding.

## Rollback on a hybrid target

Qwen3.8-27B is 48 GatedDeltaNet layers plus 16 full-attention layers.
Attention layers trim; recurrent layers cannot, because their state is
path-dependent — so a rejected block would otherwise leave the SSM state
describing tokens that were never emitted.

The reference solves this by monkey-patching GatedDeltaNet to capture
its inputs and recomputing the state over the accepted prefix. This
package already solved it better for native MTP: recurrent layers
**record** their state at each step of the verify forward
(`recordPrefixCommitStates`), so committing the accepted prefix is
`MambaCache.commitRecordedPrefix(length:)` — a lookup, not a
recomputation. DFlash 2 reuses that path verbatim.

A target with non-trimmable caches that does **not** support recording
is refused up front (`targetLacksPrefixCommitRecording`) rather than
silently corrupted mid-turn.

## Selection and supersession

DFlash 2 and native MTP are alternatives, not layers. When a user points
the runtime at a drafter folder, `resolvedMTPDraftStrategy` returns
`.dflash2` and the model's own MTP head does not run.

A drafter is trained against ONE target, and because it borrows that
target's embedding and LM head a mismatch does not error — it produces
fluent tokens from the wrong vocabulary. `VMLXDFlash2DrafterInfo`
therefore checks vocabulary, depth and hidden size against the loaded
bundle's `config.json` *before* a request, and reports a sentence the
settings pane can show. A mismatch, a missing folder, or a DFlash 1
checkpoint all degrade to ordinary decoding; none of them fail a
request.

## Where it is dispatched

Four entry points, and all four have to name it. Wiring only the obvious
one produces a feature that is configured, persisted, and inert:

| entry point | who calls it |
|---|---|
| `Evaluate.generate` | direct-engine callers, tests |
| `Evaluate.generateTokensTask` | raw token-stream callers |
| `BatchEngine.generate` | **the host chat window** — routes to `startSoloFastPath`, which builds the iterator |
| `BatchEngine.submit` | raw batched path — refuses, because it cannot express draft/verify/rollback |

DFlash 2 is exclusive for the same reason native MTP is: it owns the
target's cache across a cycle, so it cannot share a batched decode slot.

This was found the hard way. The first live run resolved the strategy
correctly, persisted it, passed it into `GenerateParameters` — and then
`BatchEngine.generate` dispatched only on `usesBlockDiffusion` (false for
`.dflash2`), so the turn fell through to batched decode. It answered
correctly, at ordinary speed, with an empty trace. Nothing failed.
`DFlash2DispatchReachabilityTests` now reads all four sites as source and
fails if a branch goes missing.

## Measured

M5 Max 128 GB, greedy, 192-token generations, interleaved rounds, free
memory logged per leg (`DFlash2SpeedSweepTests`); bf16 measured earlier
via the lossless smoke at 160 tokens.

| target | baseline | dflash2 b8 | b6 | b4 |
|---|---|---|---|---|
| Qwen3.8-27B bf16 (52 GB) | 7.6 tok/s | **1.65×** (accLen 5.65) | — | — |
| Qwen3.8-27B JANG_6D (24 GB) | 14.3 tok/s | 1.23× | 1.32× | **1.53×** |
| Qwen3.8-27B JANG_4D (17 GB) | 16.7 tok/s | 1.13× | 1.30× | **1.36×** |

Getting the quantized rows from a LOSS (0.62–0.81× in the first
implementation) to the wins above took three fixes, in order of measured
impact:

1. **One-shot lazy rollback instead of per-prefix state recording**
   (the dominant one — this is what flipped the sign). The first
   implementation verified under `capture_commit`, whose recording
   re-runs the GDN scan once per prefix WITH an `MLX.eval` per record:
   48 layers × 7 prefixes = 336 extra launches + 336 graph flushes per
   cycle, ~70% of a verify that cost 5.1× a decode step. The
   `input_capture` mode stashes layer input REFERENCES during verify
   (zero cost) and replays only the accepted rows in one kernel per
   layer, only on rejection — the reference's `_GDNStateCapture` design.
   Replay is audited bit-exact against chained decode
   (`VMLX_DFLASH2_ROLLBACK_AUDIT=1`, maxdiff 0.0).

   The one trap, and it is a repo-wide one: `ArraysCache`'s subscript
   setter MUTATES the stored MLXArray in place (`_updateInternal`), so a
   bare reference to `cache[1]` becomes the post-block state the moment
   the forward commits. The stash pins the pre-mutation value with a
   `* 1` graph node — still no eval.

2. **One host sync per cycle** (oMLX Lightning MTP's headline lever):
   verify input built on-graph from the drafter output, greedy argmax
   folded into the verify flush, `asyncEval` dispatch.

3. **Block size.** Verify cost grows faster with row count than
   acceptance does on quantized targets (z-lab/dflash#151 found the
   same): block 4 beats the checkpoint's 8 by ~20% on both JANG quants.
   bf16 tolerates 8.

Remaining headroom, still real but now secondary: the vendored MLX
0.31.1 qmm at M=8 costs 1.3–2.9× M=1 at these shapes (lm_head worst);
mlx 0.32.1 measures ~15–30% better at the same shapes, and oMLX-style
split-K verify-shape kernels (their lm_head 2.6–3.3×) would close the
rest. That is the path from ~1.4–1.5× to ~2×+ on quants.

### On losslessness under quantization

Byte-equality with the plain iterator holds on bf16 and was also
measured on JANG_6D at 96 tokens. On JANG_4D at 192 tokens the arms
diverge at a greedy near-tie a few hundred characters in: quantized
matmul reduction order depends on M, so verify logits sit a few ulp from
decode logits, and a near-tie can flip. The guarantee that survives is
that every emitted token is the verify forward's own argmax.

## Testing

| test | needs | what it pins |
|---|---|---|
| `DFlash2DrafterSelectionTests` | nothing | supersession, mismatch handling, settings round-trip |
| `DFlash2StageProbeTests` | drafter + `probe.py` golden | every intermediate of layer 0 and each layer output |
| `DFlash2ReferenceParityTests` | drafter + `dflash2_reference_dump.py` golden | the traced path, against BOTH reference precisions |
| `DFlash2LosslessSmokeTests` | target + drafter | greedy output equality, single-turn and warm-prefix multiturn |

### On the parity tolerance

This backbone runs its residual stream to ~3e6 in bf16 before the final
RMSNorm compresses it to ~19. One ulp at the top is worth whole tokens
at the bottom, and the reference disagrees with **itself** by 8.7% on
hidden states and 4 of 7 path tokens when the only change is bf16 →
fp32.

So the parity test does not assert a hand-picked epsilon. It generates
both reference precisions and requires the port to be no further from
the reference than the reference is from its own higher-precision self,
and it requires exact agreement at every path position where the two
reference precisions already agreed. Measured: 0 such mismatches, and
logit drift below the reference's own noise on both cases.

Stage-level, the port is tighter than that summary suggests — the conv
is bit-exact and attention lands within a single bf16 ulp:

```
l0_ln      rel=0.0000%     l0_attn      rel=0.3817%
l0_convin  rel=0.0000%     l0_attnconv  rel=0.5556%
l0_kern    rel=0.0000%     l0           rel=0.3788%
```

Regenerate the goldens with:

```bash
python -m venv venv && venv/bin/pip install mlx mlx-lm numpy
git clone https://github.com/z-lab/dflash.git
venv/bin/python dflash2_reference_dump.py   # scratchpad script
```

## Compiled decode on the hybrid target

The plain-decode arm of every table above was CPU-bound, not GPU-bound:
19 ms/token of per-token graph rebuild that never overlapped the GPU —
the entire gap between this runtime (17.3 tok/s) and python mlx-lm
(24.6) on the identical JANG_4D bundle. The generic compiled-decode
path (`enableCompiledDecode` + `VMLX_ENABLE_UNSAFE_COMPILE=1`) now
works for the hybrid GDN families and closes most of it:

| arm (greedy, 192 tok, hot) | plain build | compiled build |
| --- | --- | --- |
| baseline | 17.32 tok/s | **23.13 tok/s (+33%)** |
| dflash2-b4 | 22.84 | 22.89 (own iterator — unaffected) |

The compiled baseline's greedy text is **byte-identical** to plain
decode (dump both with `VMLX_SWEEP_DUMP` and diff).

What had to change, and why (each of these is a class, not a one-off —
audit any new hybrid the same way before enabling compile):

1. **`MambaCache` passes through `setupCompiledDecode` unpromoted.**
   Fixed-shape state, `innerState()`-tracked, `_updateInternal` in-place
   updates — exactly the compile state-tracker contract. Requiring
   promotability of every layer silently disabled compiled decode for
   the families that needed it most.
2. **No host `.offset` reads inside the traced forward.** A promoted
   cache's `offset` getter is `.item()` — fatal during the trace, and
   conceptually a constant-bake even where it wouldn't crash. Positions
   come from `graphOffsetArray(for:)`; masks come from the cache's own
   `makeMask` (already graph-visible).
3. **The kvSeqLen mask slice must not run** against a Compilable cache:
   `update()` returns the FULL static buffer and `makeMask` spans it.
4. **No nested `compile()`.** The per-layer micrographs (sigmoid-multiply,
   swiglu, precise-swiglu, compute_g) fall back to their plain body under
   `CompiledDecodeTrace.isActive`; the ops fuse into the outer graph.
5. **Recurrent offsets are host bookkeeping.** `cache.offset += S` runs
   once at trace time and never again; `TokenIterator` rewrites
   `MambaCache` offsets from a step counter after each compiled call.
6. **The static buffers are sized from `promptOffset + maxTokens`**
   (explicit `compiledMaxCacheLength` wins) so a long run cannot outgrow
   them silently.

Debugging note: when the trace aborts with "[eval] … during function
transformations", lldb on the fatal shows only the ErrorHandler frame.
Break at `ErrorHandler.dispatch`'s *entry* line instead — dispatch is
synchronous, so the full Swift stack up through the offending `.item()`
is visible there.

DFlash 2's own iterator still builds its graphs eagerly; compiling the
fixed-shape S=1+b verify forward the same way is the remaining lever
(the draft loop's shapes vary with context length and do not trace).

## Block size is the dominant tuning knob (and 8 was the wrong default)

Acceptance is a property of the CONTENT DOMAIN, not just the target's
quant, and the block size that maximises throughput follows it. Measured
on Qwen3.8-27B-JANG_4D, greedy, 160 tokens, settled machine (PhysMem
logged per leg; a loaded box swings these by 30%, so re-measure before
trusting any single cell):

| prompt | effort | plain | b4 | b8 | b15 |
| --- | --- | --- | --- | --- | --- |
| general | xhigh | 17.2 | **32.8** | 21.4 | 23.9 |
| general | medium | 17.3 | **29.0** | 22.4 | 25.5 |
| general | off | 17.3 | **29.9** | 21.8 | 22.5 |
| coding | xhigh | 16.9 | 34.5 | 27.1 | **34.2** |
| coding | medium | 17.2 | 35.6 | 33.3 | **43.3** |
| coding | off | 17.2 | 34.0 | 30.5 | **36.7** |

Acceptance length rises monotonically with the block (coding/medium:
5.71 at b8, 6.96 at b12, 7.27 at b15) because the drafter saturates a
narrow block — at b8 it was already landing 5.71 of a possible 8. The
quantized matmul cost grows sublinearly over the same range, so wide
blocks pay for themselves on high-acceptance content and cost more than
they return on low-acceptance content.

**b8 — the drafter config's own default — was the worst of the three in
6 of 6 cells.** Prose wants b4, code wants b15. Reasoning effort matters
too: `medium` drafts better than `xhigh` on code (accLen 5.71 vs 4.71),
because tighter reasoning is more predictable.

### Adaptive block size (opt-in: `VMLX_DFLASH2_ADAPTIVE_BLOCK=1`)

A controller that probes each ladder size and settles on the measured
best (tokens per verify-second) lands within 3-4% of the per-domain
oracle on code and BEATS every fixed size on prose. It is opt-in because
switching size mid-turn still interacts badly with the compiled staged
verify — both the staging slots and the trace are shaped by the block
length, and a switch can crash the process.

Note for anyone re-deriving this: an acceptance-RATE threshold looks like
the natural control signal and is wrong. The rate falls mechanically as
the block widens (more drafts to get right), so fixed thresholds
equilibrate mid-ladder — measured at 34.5 tok/s where fixed b15 gave
41.9. Measure throughput, do not infer it.

### Compiled verify status (opt-in: `VMLX_DFLASH2_COMPILED_VERIFY=1`)

Proven token-identical (per-cycle emitted and drafted ids hash-identical
to the eager path across b8/b6/b4 and both rounds) and materially
faster. NOT yet default: the compiled traces capture the cache array and
`DFlash2TokenIterator` is a struct, so copies retain traces
independently and releasing one copy's table does not free them — the
process segfaults at exit. Reproduce with `DFlash2LosslessSmokeTests`:
tests pass either way, the process exits with signal 11 only when the
compiled verify is on. Fixing it likely means moving the trace table
into a reference-typed box owned by the run.
