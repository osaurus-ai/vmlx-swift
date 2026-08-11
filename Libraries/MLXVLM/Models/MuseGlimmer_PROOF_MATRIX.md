# Muse Glimmer 30B — live proof matrix

Nothing in this file is proven yet. It is the gate Muse Glimmer has to clear in
the osaurus dev-app GUI before the runtime merges. Proof means a screenshot plus
the matching trace lines, per turn — not "it looked fine".

Run every leg on **JANG_4M** and **JANG_6M**. The bf16 bundle is the numeric
reference when a leg disagrees between quants.

## Why the matrix is shaped this way

Muse Glimmer renders three things into the **system prefix**:

- `Reasoning strength: high|medium|low.` (from `reasoning_strength`)
- the full tool-definition block, when tools are offered
- the date line

That makes reasoning-effort changes and tool-toggle changes the **same failure
class** for this family, and a harsher one than elsewhere: both mutate a token
near the *start* of the prompt, so the entire prefix is invalidated and no
partial suffix match is possible. A family whose effort lives in a suffix can
still reuse most of its ladder; Muse cannot. Two consequences to prove, not
assume:

1. The scope salt must cover `reasoning_strength` **and** tool availability. If
   it does not, two genuinely different prefixes collide on one cache key and
   the model is fed another configuration's KV — wrong reuse, not a slow miss.
2. Every effort flip is a full re-prefill by construction. The measurement that
   matters is that the *next* turn at that effort hits, i.e. each effort builds
   its own ladder rather than fighting for one.

## Per-turn evidence (every leg, every turn)

| Field | Source |
|---|---|
| scope salt | `[vmlx][cache/salt]` — raw policy string + scope + media |
| fetch verdict | `[vmlx][cache/disk-fetch]` — `HIT boundary=N remaining=M` or `MISS` |
| store | `[vmlx][cache/disk-store]`, `[vmlx][cache/store-boundary]` |
| store timing | `[vmlx][cache/store-phase]` — eval vs save split |
| boundaries | `[vmlx][cache/boundaries]` — prompt, stable rungs |
| TTFT | first visible token, wall clock from send |
| decode rate | sustained tok/s — **gate: ≥ 25 tok/s** |
| cancellation damage | `[vmlx][cache/rederive-failed]` — must stay 0 |
| invalid stores | `REFUSED offset/key mismatch` — must stay 0 |

## Legs

### A — varying multiturn, one conversation

The point is that state changes *mid-conversation*, not across fresh chats.

1. reasoning **high**, no tools, pure text
2. same turn shape, **+ image** (`<|patch|>`) — reasoning still high
3. **+ tools** offered, reasoning high — expect a tool call, ATEM envelope
4. change effort **high → low** mid-conversation, tools still on
5. change effort **low → high** back, tools still on
6. **turn tools off**, pure text, reasoning high
7. **+ video** (`<|video|>`), reasoning high
8. back to **pure text**, reasoning **off**
9. pure text, reasoning **on** again

Confirm per turn: correct scope salt, expected HIT/MISS, coherent answer,
correct EOS, ≥ 25 tok/s.

Expected shape: 2, 3, 4, 5, 6 all change the system prefix → MISS on the turn
that changes it, HIT on the following turn at the same configuration. A HIT on
the turn that changed the prefix is a **bug** (collision), not a win.

### B — EOS and stopping

- model stops on `<|eot|>` / `<|eom|>`; no run-on, no leaked special tokens
- generation prompt is `<|start|>assistant` with **no** trailing `<|message|>`
- long answer that ends naturally, not by hitting max tokens
- required-tool turn ends cleanly after the ATEM envelope closes

### C — stop mid-turn, then continue

Press Stop mid-generation, then send another turn in the same chat.

- next turn still hits its prefix
- interrupted turn stored nothing invalid (0 refusals)
- 0 `rederive-failed`
- continuation stays coherent — no repetition, no topic drift
- repeat with a tool call interrupted mid-envelope: the partial ATEM block must
  not be stored as if it were a completed call

### D — cache blocks on disk

- blocks are actually **written** (not just matched in memory)
- a cold boot **reuses** them — cross-process, not just cross-turn
- longest-prefix best-match reuse when the suffix differs
- no malformed/corrupt blocks; shard/rung lengths agree with their headers
- eviction behaves at the configured L2 size, and the setting displays truthfully

### E — parser robustness on live output

- no JSON/XML parser failures on any real generation
- stray `<` inside a tool argument does not desync (unit-covered; confirm live)
- multiple `<atem:invoke>` blocks in one envelope all execute
- orphan closers never reach visible text
- reasoning text and tool envelopes never bleed into each other

### F — sustained performance

- **≥ 25 tok/s** sustained, hot *and* cool — a hot-only number is a thermal
  artifact, so measure ABAB
- TTFT on a warm prefix vs cold
- prefill pp/s at short and long context

### Decode ceiling — measured 2026-08-10, UNDER GPU CONTENTION

**Read this first: every absolute number below was taken while an unrelated
inference workload was running on the same GPU.** They are therefore *lower
bounds*, not ceilings, and the "25 tok/s is unreachable" conclusion is NOT
established — it needs a re-measure on an idle machine before anyone acts on it.

What does survive the contention is everything expressed as a ratio, because
those pairs were measured minutes apart under the same load:

- the full model runs at **~100% of what its own bare quantized kernel reaches**,
  so there is no overhead left in the surrounding port;
- **compiled decode adds 3.7%** here, against +72% on a dispatch-bound model —
  this family is not dispatch-bound;
- the **4-bit path reaches 55-80% of fp16** bandwidth on the same shape, so
  dequantization is a real cost and a faster GEMV is the lever that matters.

The absolute figures, as a contended-machine baseline:

### The 25 tok/s gate at 4-bit — contended measurement, 2026-08-10

Live app: **12-13 tok/s**. Isolated forward pass, no sampler/detokenizer/cache/UI:
**12.2-13.6 tok/s**, so the serving stack costs essentially nothing and the
whole budget is the forward pass.

Weight bytes actually read per decode token, from the JANG_4M shard headers
(the vision tower is not touched during text decode):

| component | GB | share |
|---|---|---|
| text MLP | 11.66 | 53.8% |
| text attn qkvo | 3.20 | 14.8% |
| lm_head | 1.43 | 6.6% |
| attn output gate (this family's extra) | 0.80 | 3.7% |
| embeddings | 0.76 | 3.5% |
| **text total** | **17.85** | |
| vision tower (idle during decode) | 3.84 | |

That puts measured decode at **~245 GB/s**. Bare matvec at this model's own MLP
shape (19968x6656):

| kernel | GB/s |
|---|---|
| fp16 | 342 |
| 4-bit quantized, chained | 241-273 |
| **full model decode** | **245** |

The model already runs at ~100% of what the bare quantized kernel reaches, so
there is no surrounding overhead left to reclaim. Two things follow:

1. **The ceiling is the kernel, not the port.** Compiled decode buys +3.7% here
   (not the +72% it gave Gemma E2B — that model was dispatch-bound, this one is
   not). Folding the centered-norm `+1` into the weights at load removes 208
   tensor ops per token and cannot be slower, but the effect is inside the
   ~11% run-to-run variance, so no speedup is claimed for it.
2. **25 tok/s needs 446 GB/s**, which is above even the *fp16* rate on this
   shape. A perfect zero-overhead implementation running at fp16 speed would
   cap at 342/17.85 = **19 tok/s**. At the quantized kernel's real rate the cap
   is **~15 tok/s**, and 12-13 measured is 80-89% of that.

So on this measurement the gate is not met by optimizing the port — but the
absolute rates were taken under GPU contention, so re-measure idle before
concluding the gate is unreachable. What the ratios do establish is that the
remaining headroom is in the quantized kernel, not in the port. Beyond that it
needs fewer bytes per token — the 2-bit JANG_2D bundle (~9 GB/token, implying ~30 tok/s) or a smaller
model — or a faster 4-bit GEMV kernel, since the current one reaches only 55-80%
of fp16 bandwidth. Measure before attempting either: thermals move these numbers
~11% run to run, so any claim needs ABAB attribution.

**No speculative decoding.** The `-assistant` draft bundle is out of scope for
now, so 25 tok/s has to come from the base decode path alone. Budget: ~30B
dense at 4-bit is ~17–18 GB of weight reads per token, which puts 25 tok/s at
roughly 75–80% of achievable memory bandwidth. The levers left are the ones
inside the forward pass — the 2048-token sliding window on 39 of 52 layers
keeping KV traffic and attention work down, quantized KV cache, and fusing the
per-layer work (the centered norms, the QK-norm, the output gate) so decode
does not pay dispatch overhead 52 times per token.

## Status

| Leg | State |
|---|---|
| A varying multiturn | not run |
| B EOS / stopping | not run |
| C stop mid-turn | not run |
| D disk cache blocks | not run |
| E parser robustness | unit tests pass; live not run |
| F performance ≥ 25 tok/s | not run |

Blocked on: vision tower (legs A2/A7), osaurus integration + repin.

Out of scope for this pass: speculative decoding via the `-assistant` bundle,
and the JANG_2L quant.

## Suspected reasoning-effort wiring defects — UNCONFIRMED, must be checked

Muse Glimmer's official contract: **reasoning strength is part of the system
prompt**, written as `Reasoning strength: <value>`, with four levels —
**low / medium / high / xhigh** (`high`/`xhigh` intended for complex problem
solving, coding, and agentic work). The shipped template's `render_reasoning()`
defaults to `high` when `reasoning_strength` is undefined.

Note the level set is **four**, not three. Anything in osaurus that models
effort as low/medium/high only cannot express `xhigh`, and a UI that maps its
top setting to `high` silently caps the model below its intended ceiling for
exactly the agentic work this model is meant for.

Four suspected defects, none verified yet:

1. **Card and template can disagree.** Selecting "low" in the reasoning card may
   coexist with a template-injected "high", leaving two conflicting
   `Reasoning strength:` statements in one system prompt. Check the rendered
   prompt for a duplicate — and note whichever the model honours is then a
   coin-flip that also changes the cache prefix.
2. **`enable_thinking=false` may be dead on this path.** Muse has no think tags,
   so a flag written for tag-gated families likely has nothing to switch off.
   If it is inert, turning reasoning "off" in the UI would not actually reduce
   reasoning — it would just look like it did.
3. **`reasoning_effort` may be dead too.** The template reads
   `reasoning_strength`. If osaurus sends `reasoning_effort`, the template never
   sees it and silently falls back to its `high` default — so every turn reasons
   at high regardless of the setting.
4. **Low reasoning may cut reasoning mass materially.** If (1)–(3) are wrong and
   the level does land, confirm the output difference is intended rather than a
   collapse in answer quality.

Why this matters beyond correctness: every one of these changes the **system
prefix**, so a defect here is simultaneously a cache-key defect. Two turns that
look identical in the UI but render different strength lines must not share a
cache entry, and two that render the same line must not miss.

**How to check:** dump the fully rendered system prompt for each card setting
and grep for `Reasoning strength:` — count the occurrences and read the value.
That single check settles 1–3.

## Vision: broken, then fixed — 2026-08-10

Vision did not work, and the test that said otherwise accepted any non-empty
answer containing a word from a list the prompt itself supplied. Rebuilt around
matched opposite stimuli — so a constant answer scores half, not full — and run
against Qwen3.6-27B as a positive control, because a probe no model passes
measures the probe.

| probe (scored as opposite pairs) | before | after | Qwen3.6 control |
|---|---|---|---|
| six-trial shape battery | 3/6 (constant "circle" = chance) | **6/6** | 6/6 |
| solid red field / solid blue field | "green" / "green" | **red / blue** | 2/2 |
| bar top / bottom | top / bottom | top / bottom | top / bottom |
| bar left / right | "right" / "right" | **left / right** | left / right |

Both quants pass: JANG_6M and JANG_4M each score 6/6 on shapes and name both
solid fields correctly.

The model's own description of a centred black circle, before and after:

> before: "an amorphous, irregular blob… muted yellow-olive… sits in the lower
> part of the frame"
>
> after: "a perfectly round disc, filled in solid… the disc is black… centred in
> the frame with a roughly even margin on all four sides"

### What was actually wrong

Four defects, all invisible to shape checks because every one of them preserves
tensor dimensions:

1. **RoPE axis order.** The checkpoint interleaves its head dimension as
   `[w, h, w, h]`; the port built `[h, w, h, w]`. Every patch had its row and
   column coordinates exchanged. This is what produced the diagnostic signature
   — vertical position readable, horizontal not — because vision tokens arrive
   raster-ordered, so rows survive in token order even when the positional
   signal is wrong, while columns have no such backup.
2. **RoPE coordinate offset.** Coordinates are 1-based, not 0-based.
3. **Grid order.** The encoder runs over the raster grid, not over merge blocks.
   `QwenVL.patchify` groups its rows by merge block, so the rows are ungrouped
   on entry and regrouped at the end.
4. **The 2x2 merge is feature-major.** The merged vector is all `mergeUnit`
   values of feature 0, then feature 1, and so on — not one patch vector
   concatenated after another. Same width either way.

Plus the patch vector layout fixed earlier: `[temporal][channel][h][w]`, not
`[channel][temporal][h][w]`, identified from the weights (temporal-pair cosine
0.990 against 0.492) before the reference confirmed it.

The window size inferred as `pos_emb_height * patch_size` turned out to match
the reference; the adapter's trailing activation, removed at one point on the
strength of a Qwen analogy and then restored, is also correct.

### Dead ends worth not repeating

Two internal metrics looked like proof and were withdrawn once controlled:
the layerwise contrast collapse (46 to 0.13 through the stack — Qwen's working
tower scores 0.515 on the same statistic) and a gradient row-order correlation
(Muse 0.42, Qwen **-0.07**, i.e. Muse scored better). Neither could separate a
working tower from a broken one. `MuseGlimmerMetricControl` keeps that closed.

Rejected by running them: disabling rope, the interleaved rope pairing,
removing the adapter's trailing activation, and window sizes 224/112/56.

## Post-merge live evidence (2026-08-10 session, merged build)

- **SSD prefix HIT proven**: `HIT disk boundary=2819 remaining=7786 tokens=10605`
  — a 10.6k-token prompt restored 2,819 tokens from disk and prefilled only the
  remainder. 74 disk stores this session, 0 `rederive-failed`.
- **Reasoning strengths live**: `strength=high ×6, low ×3, medium ×3` in the
  scope salts, distinct salt per strength.
- **Two refused stores, benign**: `REFUSED offset/key mismatch tokens=N
  offsets=[N, N+1]` twice. The hybrid 13-standard/39-rotating topology left
  layer offsets one apart at a store boundary and the guard declined to write
  the inconsistent entry. Correctness preserved (that is the guard's job — the
  LFM2.5 per-layer-offset lesson); cost is one missed store opportunity per
  occurrence. Root-causing the off-by-one belongs to the optimization pass:
  suspect the first post-prompt sampled token advancing standard and rotating
  caches asymmetrically.
