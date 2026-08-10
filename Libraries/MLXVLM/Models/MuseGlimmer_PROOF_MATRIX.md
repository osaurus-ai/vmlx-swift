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
