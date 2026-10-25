# DeepSeek-V4.1-Flash runtime in vmlx-swift (2026-10-25)

Status and contract for DeepSeek-V4.1-Flash (`model_type: deepseek_v41`, text config
`deepseek_v41_text`) on the DSV4 code path, as served by Osaurus. The Python runtime in jangq
(`jang_tools/dsv41/`) is the numerical reference: see its `RUNTIME_SERVING.md` and
`DECODE_DSPARK_METHODS.md`. The authoritative prompt format is the bundle's
`encoding/encoding.py`.

## Family isolation (DSV4-0731 must not change)

`DeepseekV4ChatFamily(modelType:)` selects the contract. Anything that isn't explicitly
`deepseek_v41*` stays on 0731:

| behavior | 0731 (`deepseek_v4`) | V4.1 (`deepseek_v41`) |
|---|---|---|
| reasoning effort | named preface (`low` = none) | numeric `Reasoning Effort: N` inside `<｜System｜>` (ladder: none 25, low 50, medium 62, high 75, xhigh 88, max 100) |
| system messages | 0731 rendering | System-led; mid-conversation system turns supported |
| tool tags | 0731 DSML names | V4.1 tags with a leading space (`<｜DSML｜ calls>`, ` invoke`, ` parameter`); `ToolCallFormat.dsmlV41` |
| indexer weights | loaded (`.indexer.*`) | V4.1 indexer (`self_attn.v41_indexer.*`) |
| compressed-KV pools | 8-bit | 4-bit, two per byte |
| resident safetensors load | unchanged | required |
| reasoning floor | 0731 markers | never arms on a numeric V4.1 prompt |

Hosts must render with the bundle's family. Osaurus does this in
`SwiftTransformersTokenizerLoader.applyDeepseekV4NativeTemplate`
(`DeepseekV4ChatEncoder(family:)`, plus `reasoningBudget(from:)` for V4.1 thinking turns). An
absent effort renders as low (50). The bundle reference default is high (75).

## Engram (n-gram embedding tables)

- Tables are located from `jang_config` `runtime.engram_dir` (default `<bundle>/engram`).
  No environment variable is needed. Load throws if a V4.1 bundle's tables are missing or
  don't match the config. It never runs without Engram.
- Rows are read in place from the safetensors shards with concurrent `pread`: 64 lanes,
  256-row tasks, rows and scales together. The OS page cache keeps hot rows; nothing is wired.
  The reads for layer 14 overlap the compute of layers 0–13. Cold random rows read about
  130k rows/s this way, against 8–11k rows/s serially.
- The n-gram history travels with the cache (first compressed layer): copy, trim, metaState
  and disk records. A V4.1 disk record without history is a cache miss, never padded. The
  runtime holds no per-sequence state.

## KV cache size

V4.1 compressed pools are 4-bit (about 0.625 B per element against 1.125 for 8-bit). The disk
format records the width and refuses to restore into a cache of the other width.
`VMLX_DSV41_KVBYTES=1` logs total cache bytes, bytes per token and pool bits after each
prepare. Target: ~0.9 GB for a 1M-token context, resident and stored.

## Long context: indexer

Port of the Python indexer. Layers 2/8/14/20 build the shared index keys from compressor
output. The other index layers choose their own top-512. Layer 20 sets the candidate blocks
that layers 24+ select from, and layers in between reuse the previous selection. With 512 or
fewer visible compressed rows every row is attended, as in the reference. Without the indexer,
output diverges past ~512 compressed rows and prefill is quadratic.

## Measured baseline (PR #657 head `661e9173`, AR, before these fixes)

M5 Ultra 256 GB, Osaurus Release build, `DeepSeek-V4.1-Flash-1017-JANG-Q2Q3-IMATRIX`:

| metric | value |
|---|---|
| load + first reply | 35.5 s |
| phys_footprint after load (peak) | 196.7 GB (197.4) |
| cold prefill, 1.2k tokens | 488 pp/s |
| decode (thinking on) | 26.6–26.9 tok/s |
| TTFT | 0.79–0.91 s |
| SSD L2 repeat prompt | 23.7 s → 0.47 s; hits survive a restart |
| tool calls | broken: raw V4.1 DSML in content (fixed by `dsmlV41`) |

`usage.prompt_tokens_details.cached_tokens` reports 0 even on an L2 hit. That's a reporting
gap, not a cache miss.

## GPU verification checklist (for the fixes above)

1. 0731: prompts byte-identical to before (low preface, no `<｜System｜>`); indexer weights
   load; generation unchanged.
2. V4.1 loads with no `VMLX_DSV41_ENGRAM_DIR` and is coherent; renaming `engram/` makes
   loading throw a "DeepSeek-V4.1 engram" error.
3. V4.1 loads `self_attn.v41_indexer.*` (`wq_b` as 8-bit, group 64).
4. A V4.1 tool prompt returns populated `tool_calls` and no DSML in content.
5. `xhigh` renders `Reasoning Effort: 88`; "Say hello in exactly one word." does not loop.
6. A multi-turn prefix-cache hit continues exactly like a cold run; the "n-gram history
   missing" log never appears.
7. `VMLX_DSV41_KVBYTES=1` shows `pool bits 4` and about half the bytes per token of 8-bit.
8. Past 2K tokens output stays coherent and prefill speed does not degrade quadratically.
9. Cold prefill of a 1K-token unique prompt is clearly faster than the baseline.

## DSpark speculative decode (branch `feat/dsv41-dspark-swift-v2`)

Port of the Python `DSparkRunner` (methods M1 and M2 of `DECODE_DSPARK_METHODS.md`) and its
wall-clock `_Governor`. It replaces the first Swift attempt (`feat/dsv41-dspark-swift`, which drafted
through the native-MTP iterator). Native MTP is not touched: DSV4 has no native-MTP cold-start
depth, and DSV4-0731 never attaches a head.

### Guarantee

Greedy by construction: every committed token is a backbone argmax. The output is identical to
greedy AR except at near-ties. A multi-row verify forward rounds differently from a one-row decode
step, so a top-2 margin below about 0.05 can flip (the Python runtime measured first divergences at
AR margins of 0.004–0.04 over 256 tokens). Both the verify and AR paths take fp32 logits from the
quantized head: `DeepseekV4Math.lmHeadFp32` casts the hidden to fp32 before `quantizedMM`. That has
been the default for both DSV4 families, so 0731 is unchanged.

### Where it runs

- `DraftStrategy.dspark(drafterPath:)` runs inside `TokenIterator`, as Uno does. Prefill, prefix-cache
  fetch and store, boundary snapshots and stop handling are the ordinary AR code. `BatchEngine.generate`
  takes the exclusive solo path. With another request in flight, and on `submit`, the strategy is
  dropped and the request decodes AR in a batched slot.
- `VMLXServerRuntimeSettings.resolvedMTPDraftStrategy` returns `.dspark` only when the config's
  `model_type` is `deepseek_v41` or `deepseek_v41_text`, `<bundle>/dspark/dspark.safetensors` exists,
  and `mtp.mode != .off`. It is on by default (`familyDefault`). The host must pass `bundleDirectory`.
- The head (8.9 GB: prestacked 4-bit experts, bf16 attention and Markov head) loads on the first
  greedy `.dspark` request, before that request's prefill. A sampled request never loads it. Each
  `dspark_*` key is read from the bundle config. With the head resident, the default prefill outer
  chunk is 1024 (see RUNTIME_SERVING.md).
- Per request it runs only when all of these hold: temperature 0, no logit processor (penalties,
  reasoning budget, JSON schema), no media, no KV quantization, no compiled decode, and every cache
  layer supports an in-place checkpoint. Otherwise the request logs why (`dSpark requested but not
  used for this request`) and decodes AR.

### Loop (`MLXLMCommon/SpecDec/DSpark.swift`)

The state is the same as ordinary decoding: the cache holds prompt plus committed tokens, and `y`
(p0, at c+1 == cache offset) is pending. Each step:

1. Read p0 back. This closes the previous step's timing, then the governor picks spec or AR.
2. AR: an ordinary `TokenIterator` step.
3. Spec: draft d1…d5 for c+2…c+6 (seeded with p0, windows ending at c, which is the critical
   off-by-one). W = clamp(round(5 · acceptEMA), 1, 5), capped at the budget. Take an in-place cache
   checkpoint and run one verify forward of [p0, d1…dW]. k is the length of the longest prefix
   where each dj+1 matches the argmax.
4. k == W: keep the verify cache (no refeed), commit p0…dW, and argmax(row W) becomes the next p0.
   One forward per step.
5. k < W: restore in place, run one commit forward of exactly [p0, d1…dk, truth_k], and take the
   next p0 from its last row.
6. A block is cut at the first stop token and at `maxTokens`. The cache never runs ahead of the
   returned tokens, so the post-answer prefix-cache boundary stays valid.

The governor is a port of `_Governor`. It keeps an EMA (α 0.3) of seconds per emitted token for
spec steps and for AR steps, measured from one step start to the next. It speculates while spec is
cheaper than AR, sends one spec probe every 16 AR tokens (one good probe switches back), and
re-measures AR every 64 spec steps. There is no fixed depth and no terminal state.

### Cache and prefix-cache safety

- `SpeculativeCacheListCheckpoint` stores reference snapshots and restores them into the same
  objects (`RotatingKVCache`, `KVCacheSimple`, `DeepseekV4Cache`: local window, both pools,
  incomplete-window buffers, deferred decode inputs, Engram n-gram history, hidden tail). Nothing
  deep-copies the cache, and cache identity never changes.
- The draft windows have no identity-keyed state. While a head is attached, every V4.1 forward
  records the target layers' input hidden (layers 37/38/39, mean over hc, after Engram) for its last
  128 positions into `DeepseekV4Cache.auxiliaryHiddenTail` on the history holder layer. The tail
  travels with copy, trim, restore and the disk record (`dsv4_<i>_dspark_tail`, optional; a record
  without it restores normally). A per-request `DSparkV41Session` syncs its windows from the tail by
  absolute position at draft time. A prefix hit (including SSD L2 after a restart) therefore
  drafts with a full 128-row window, the same as the Python `dspark_tail`. If the tail does not end
  at the cache offset, the session drafts with empty windows. That lowers acceptance, never
  correctness.
- Engram history: the holder's history is part of the checkpoint, and every forward re-records it
  from its start position, so verify, restore and commit hash exactly like AR.

### Cost per step

There is one cache checkpoint (array references only). The draft does one host sync, and the
Markov chain runs on the device. The Markov head is cast to fp32 and transposed once at load. The
verify readback is one sync. The windows project only the new rows.

### Environment

| variable | effect |
|---|---|
| `VMLX_DSV41_DSPARK=0` (`off`/`false`/`no`) | dSpark off everywhere. It can only turn dSpark off, never on over Off. |
| `VMLX_DSV41_DSPARK_STATS=1` | one `[dSpark]` NSLog line per request: steps, spec/AR steps, drafted/accepted, acceptance, full accepts, commit forwards, mean emit, ms/token spec and AR (measured and governor EMA), draft/verify/commit ms, tok/s. A skipped request logs `[dSpark] off for this request: <reason>`. |
| `VMLX_DSV41_DSPARK_PREFILL_STEP=<n>` | prefill outer chunk while the head is resident (default 1024; `DSV4_PREFILL_OUTER_CHUNK` still wins) |

### CPU tests (`MLXLMCommonFocusedTests`)

- `DSparkSpeculationFocusedTests`:
  - greedy identity against AR on a synthetic attention target (partial, zero and full acceptance), with caches equal to AR's;
  - restore after the rotating window wrapped;
  - stop-token and budget cuts;
  - the governor under a synthetic clock (port of `dsv41_dspark_governor_test.py`);
  - the hidden tail;
  - `TokenIterator` end to end.
- `DeepseekV41DSparkFocusedTests`:
  - `DeepseekV4Cache` in-place restore;
  - the tail through copy, trim and the disk record;
  - head load with every parameter verified;
  - incremental window sync equals one-shot sync;
  - the Markov chain;
  - the strategy and model gates for 0731 and V4.1;
  - native-MTP cold start off.

### GPU verification checklist (lead; one model load at a time)

1. Load V4.1 in Osaurus with defaults and run one greedy (temperature 0) request. The log shows
   `VMLX DSV4.1 dSpark head attached ... 3 stages, block 5, targets [37, 38, 39]` once (a few
   seconds). phys_footprint is about 206 GB (backbone plus 8.9 GB).
2. With `VMLX_DSV41_DSPARK_STATS=1`, a thinking-off code prompt of about 300 tokens should give
   `accept` ≥ 0.6, `mean_emit` ≥ 3, `ar` steps only from probes, and decode about 45–50 tok/s
   (Python reference: code 44.7, code2 39.3, structured 76.9, prose 28.6; AR about 30). Prose should
   be at or above AR (27 tok/s): the governor shows mostly `ar` steps with spec probes.
3. Identity: run the same greedy prompt with `VMLX_DSV41_DSPARK=0`. The tokens match the dSpark run
   up to the first near-tie. Check any divergence against the AR top-2 margin (expect < 0.05).
4. 0731: no head attaches, there are no `[dSpark]` lines, prompts and generation are unchanged.
5. Off: with Speculative Decoding Off, no head attaches, even with `VMLX_DSV41_DSPARK=1`.
6. Prefix cache: send a multi-turn chat and a tool turn. The second turn hits (RAM, then SSD L2 after
   a restart), `[dSpark]` acceptance on turn 2 is in the same range as turn 1 (full window from the
   persisted tail), the output matches a cold run up to near-ties, and the n-gram history-missing
   log never appears.
7. Sampled request (temperature > 0): `[dSpark] off for this request: sampled decoding`, decode
   unchanged. With the default sampler, check whether Osaurus sends a reasoning-budget or penalty
   processor on V4.1 turns. If it does, dSpark stays off for those turns (see the log reason).
8. Cold prefill of a 12k-token prompt with the head resident is about 700 pp/s (1024 chunks), not
   about 545.
9. Long output (2k+ tokens): tok/s does not fall off over the run, and the `ms/tok ar` estimate grows
   only with context.
