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

## VISION IS NOT WORKING — confirmed against a positive control, 2026-08-10

The merged claim that the vision path works is **wrong**, and the test that
supported it was too weak to catch this. That test asserted only that the answer
was non-empty and contained a word from a list including "image", "colour" and
"band" — a model that cannot see satisfies every one of those by echoing the
prompt's own wording.

### The evidence, with a control

Each probe uses stimuli whose ground truth is known exactly and which the prompt
never names, and each is scored as a **matched opposite pair** so a constant
answer scores 1/2 rather than 1/1. The same probes were then run unchanged
against Qwen3.6-27B — a working VL model in this same runtime — to prove the
probes are passable:

| Probe | Muse Glimmer 30B (JANG_6M) | Qwen3.6-27B (control) |
|---|---|---|
| solid red field / solid blue field | "green" / "green" — **0/2** | **2/2** |
| black circle / black square | "circle" / "circle" — **1/2** | **2/2** |

A working model passes both trivially. Telling a square from a circle is the
easiest task a vision model is ever given.

(A third probe on two macOS desktop-picture *thumbnails* returned "mottled
texture" and "abstract shape". That one is **not** evidence: those files come
from a hidden `.thumbnails` directory and their actual content was never
verified, so a vague description may well be accurate. It is recorded here only
so it is not mistaken for a finding later.)

### What is verified CORRECT (so the fault is not here)

- all 809 vision tensors load and match the checkpoint bit for bit (`verify: .all`)
- channel order: channel 0 carries the red value, checked against raw PNG pixels
- normalization matches `processor_config.json` exactly (mean/std 0.5, 1/255)
- quantization correctly skips the vision tower (no `.scales` in the checkpoint)
- adapter and projection are `bias: false`, matching a checkpoint with no bias
- geometry: 256 `<|patch|>` placeholders for 256 vision tokens, 221 for 221
- projected vision tokens sit at RMS 0.83 against normed text tokens at 1.0
- patch content enters the embedder ~56x stronger than the position term

### A dead end, recorded so it is not repeated

Feeding a half-black / half-white field and measuring how strongly the two
halves stay separated against the spread inside one uniform half looked
damning — the ratio falls from 46 after the embedder to 0.13 by layer 49, with
the position-driven spread growing from 0.014 to 15.6.

**That measurement proves nothing.** Qwen3.6's working tower scores **0.515** on
the identical statistic — the same low band. The metric cannot separate a good
tower from a bad one, so no verdict can be drawn from it. `MuseGlimmerMetricControl`
pins this down permanently.

Also ruled out by measurement, before the metric itself was invalidated:
disabling rope entirely and switching to the interleaved rope pairing. Neither
repaired the end-to-end behaviour.

### One confirmed defect, fixed: the patch vector layout

`patch_embedding.weight` is `(1536, 1176)` and 1176 = 3 channels x 2 temporal x
14 x 14. Nothing states the order of those factors and both candidates have the
same width, so every shape check passes either way. The weights settle it — a
still image duplicates the frame, so the two temporal slots always see identical
pixels and their weights co-adapt:

| pairing | temporal-pair cosine |
|---|---|
| `[channel][temporal]` (what the port fed) | 0.492 |
| **`[temporal][channel]`** | **0.990** |
| unrelated-slice baseline | 0.740 |

Under the assumed order the "pairs" correlate *below* the unrelated baseline.
`temporalMajor` now reorders patchify's output before the projection, and
`MuseGlimmerPatchLayout` fails if that evidence ever flips.

Note what this fix does **not** do: it did not change the end-to-end answers.
Colour is still misnamed. That is consistent — the reorder only matters when the
channels differ, and greyscale (the circle/square probe) is untouched by it — so
at least one further defect remains.

Also tested and rejected, by running it: removing the adapter's trailing GELU
(the Qwen reference has no trailing activation) left the shape probe at 1/2
unchanged, so it was reverted rather than kept on style grounds.

Checked and correct, so not the cause: `image_token_id` 200092 / `video_token_id`
200091 both decode and match the rendered placeholders; position interpolation
and merge-block reordering agree with the working Qwen3VL implementation, and at
a 32x32 grid both reduce to exact identity; the window partition is a single
window at that size; the attention block, rotary pairing and masks all match the
reference.

### The probe, properly calibrated

The six-trial battery (three circles, three squares, varying size and position)
is the yardstick, and it took two corrections before it measured anything:
scoring required the opposite word to be *absent*, which threw away "a square,
not a circle"; and `maxTokens` was low enough to truncate the model's thinking,
which returned an empty visible answer indistinguishable from a refusal. Both
were caught because the **control** failed, not the subject.

| model | score |
|---|---|
| Qwen3.6-27B (control) | **6/6** |
| Muse Glimmer, window 448 (default) | 3/6 — constant "circle", i.e. chance |
| Muse Glimmer, window 224 | 3/6 — constant "circle" |
| Muse Glimmer, window 112 | **0/6 — every answer inverted** |
| Muse Glimmer, window 56 | 3/6 — mixed |

The 0/6 is the interesting row: a clean inversion on six trials happens by
chance once in 64, so at 112 the shape information is reaching the model
reliably and coming out with the labels swapped, while at 448 and 224 nothing
reaches it at all. That is a lead, not an answer — 112 was not adopted, because
a setting that scores 0/6 is worse than the one that scores at chance, and
colour stays wrong at every window size tested.

### The sharpest finding: vertical works, horizontal does not

Bars filling only one quarter of the frame, each forced choice scored against
its opposite, with Qwen3.6 as the control:

| probe | Muse Glimmer | Qwen3.6 (control) |
|---|---|---|
| bar in the top quarter | top | top |
| bar in the bottom quarter | bottom | bottom |
| bar on the left | **right** | left |
| bar on the right | right | right |

Vertical is correct. Horizontal is a constant answer — the two opposite images
are indistinguishable to the model.

That split is a clue, not a curiosity. The vision tokens arrive raster-ordered,
so **row** position is recoverable from token order alone even with no
positional signal at all, while **column** position can only reach the language
model through the vision position embedding. Losing exactly the axis that
depends on that embedding, and keeping the one that does not, points at it
directly.

The magnitudes agree, against the control:

| | content : position magnitude |
|---|---|
| Qwen3.6 (resolves both axes) | **4.8** |
| Muse Glimmer | **55.8** |

Muse's position term is roughly 11.5x weaker relative to patch content than the
working reference — small enough to be effectively invisible. Whether the cause
is a missing scale factor, the wrong insertion point relative to `ln_pre`, or a
table that is genuinely small in a tower that also carries rope, is not yet
settled; the tower's own vertical/horizontal asymmetry is mild (1.33x against
the control's 0.91x), so the loss is not a gross scrambling inside the blocks.

The model's own words match. Asked to describe a centred black circle it
reports "an amorphous, irregular blob", "muted yellow-olive", "sits in the lower
part of the frame" — a picture read without usable position.

### Scaling the position term: it moves the needle, but is not the fix

`ln_pre` follows the sum, and LayerNorm is scale-invariant, so only the
content:position **ratio** matters — multiplying the position term by k is the
same as dividing content by k. Tested live:

| position x | top bar | bottom bar | left bar | right bar |
|---|---|---|---|---|
| x1 (shipped) | top | bottom | right | right |
| x12 (ratio ~4.7, matching the control) | bottom | top | left | left |
| x30 (ratio ~1.9) | bottom | bottom | **left** | **right** |

At x30 horizontal becomes correct for the first time — the axis that had been
unreadable at every window size. So the position term genuinely does carry
column information and is simply too faint to be used at the shipped ratio.

But every amplification breaks vertical, which was correct at x1, so a uniform
scale is not the answer: it trades one axis for the other rather than fixing
either. Something about how the two terms are combined is wrong in a way a
single multiplier cannot express, and the knob was removed rather than shipped
at a value that scores well on one probe and badly on another.

### Still unknown

The defective operation has not been located. There is no reference
implementation on this machine — the bundles ship no `modeling_*.py` and
transformers has no `muse_glimmer` — so locating it needs a reference to diff
activations against, layer by layer, rather than another whole-tower statistic.

Text-only Muse Glimmer is unaffected: reasoning strengths, ATEM tool parsing,
EOS and the prefix cache were proven separately and do not involve the tower.

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
