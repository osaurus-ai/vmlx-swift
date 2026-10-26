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
  The reads for layer 14 overlap the compute of layers 0–13: the forward `asyncEval`s the
  hidden state before it waits for them. Cold random rows read about 130k rows/s this way,
  against 8–11k rows/s serially.
- Chunked prefill passes the next forward's tokens on the history-holder cache, and each
  chunk starts reading the next chunk's rows once its own are resolved (reference
  `engram_lookahead`). The record is used only on an exact start/ids/history match.
- The n-gram history travels with the cache (first compressed layer): copy, trim, metaState
  and disk records. A V4.1 disk record without history is a cache miss, never padded. The
  runtime holds no per-sequence state.

## KV cache size

Only the kv-source layers (2, 8, 14, 20; ratios 2, 2, 2, 1) hold pools: compressed KV
(512 wide) and index keys (128 wide). Every other layer holds only its 128-row window.

- Pools are 4-bit from the first row (reference `QuantGrowBuffer`): one contiguous slab per
  pool, grown in 2048-row steps and written in place. There is no BF16 hot tier for V4.1.
  Before 2026-10-25 every pool under 64 MiB stayed BF16, which measured 3.2 KB/token below
  ~65k tokens (42k tokens: 139-160 MiB).
- Row cost: 4-bit codes plus fp16 scale and minimum per 32 elements, 320 B per compressed row
  and 80 B per index row, so 1000 B/token of pools against the reference's 900 (group 64 with
  bf16 scale/bias). 42k tokens ≈ 52 MiB including the 10 MiB of windows; 1M ≈ 1.05 GB.
- After a multi-token update the local window keeps only its last 128 rows (the reference's
  `cache.window = raw[:, -win:]`); previously a 2048-token chunk left ~2175 rows per layer.
- Disk records store each pool as one segment of its live 4-bit rows, about the resident size.
  Older BF16 records are re-encoded on restore, with the same codes appending would produce.
- `VMLX_DSV41_POOL_LAYOUT=segmented` restores the old BF16-hot-tier layout for A/B runs.
- The disk format records the width and refuses to restore into a cache of the other width.
  `VMLX_DSV41_KVBYTES=1` logs total cache bytes, bytes per token and pool bits after each
  prepare.

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

## Measured live (combined branch `264f6c4c`, osaurus `fix/dsv41-thinking-toggle`)

M5 Ultra 256 GB, Osaurus Release, DSpark on (default), thinking off, unique cold prompts:

| metric | value |
|---|---|
| load + first reply | 39.7 s |
| phys_footprint after load / peak (42k context) | 205.0 GB (incl. the 8.9 GB dSpark head) / 212.4 GB |
| decode code / code2 / structured / prose | 65.6 / 58.7 / 105.2 / 34.3 tok/s (AR ≈ 34 tok/s, 28–30 ms/token) |
| TTFT | 0.30–0.42 s |
| cold prefill 1.6k (first request) / 6.5k / 42k | 363 / 896 / 654 pp/s |
| long-context retrieval at 42k | correct |
| SSD L2 repeat prompt | 1.77 → 0.31 s, `cached_tokens` 3420 reported |
| tool call + tool turn 2 | populated `tool_calls`; turn 2 370 tokens cached |
| KV bytes at 42k (`VMLX_DSV41_KVBYTES=1`) | 53.1 MiB = 1.32 KB/token incl. fixed windows (4-bit pools, group 32); ~1.1–1.3 GB at 1M |
| DSpark stats (code, per request) | accept 0.5–0.7, mean emit 2.8–3.9, prefix commits 5–6 vs commit forwards 0–2, spec 14–20 ms/token vs AR 28–30 |

Multimodal, live on the vision branch (`7dc30630`): red 112×112 PNG → "Red" (928-token prompt, 184 image-span
tokens at 13×13, one image prefill chunk, dSpark off for the media turn); text follow-up → "Apple", 930/944 tokens
cached, 0.5 s; blue image → "Blue", cache miss. Text decode unchanged (code 70.1, code2 58.6, structured 104.8,
prose 34.4 tok/s), footprint 205.9 GB with the vision tower.

Group-64 pools (`b34b1789`, reference `QuantGrowBuffer` group): KV at 42k tokens 48.9 MiB including the 40
full sliding windows (~10.5 MB) and slab slack, i.e. 900 B/token marginal → ~0.91 GB at 1M context. Quality
and speed unchanged (long-context retrieval correct, Red / Apple / Blue, code 65.6 tok/s).

Long-context prefill (unique text, retrieval answer checked). The indexer is the cost that grows with
context: 8 index layers score every compressed row per query. Two exact-up-to-ties changes (python reference
jangq 3c64d3a / dc09b99):
- `DeepseekV41FusedIndexScores.swift`: vMLX's `dsa_pool_index_scores` Metal kernel (`sum_h w * relu(q . k)`,
  fp32, one pass) for prefill index scoring (> 8 rows, >= 32 keys); `VMLX_DSV41_FUSED_INDEX=0` keeps the stock path.
- `DeepseekV41IndexerMath.topK`: block top-k for rows of 65536+ keys (top-512 blocks of 64 by maximum, then the
  exact selection over their entries) instead of MLX's full-row sort.
Final build (`1cf69395`, block top-k for prefill rows only): cold prefill 1.6k / 6.5k / 42k / 49k / 100k = 843 / 918 /
727 / 700 / 594 pp/s, retrieval correct; decode code / code2 / structured / prose 67.7 / 52.8 / 105.6 / 32.6 tok/s.
Same build A/B (`a436089e`): 49k tokens 593 → 680 pp/s, 100k 458 → 593 pp/s, retrieval correct in all runs;
42k in the full acceptance 627 → 727 pp/s. Decode unchanged (code 67.4, code2 50.4, structured 105.6, prose
32.5 tok/s). The per-layer eval barrier inherited from 0731 (above 24,576 tokens) costs only ~3%
(`DSV4_LAYERWISE_PREFILL=0`) and stays on for its memory protection.

DSV4-0731 regression on the same builds (`b34b1789` and `a436089e`; `DeepSeek-V4-Flash-0731-JANG`): decode 42 tok/s,
prefill 660–1024 pp/s, tools / multiturn / reasoning / SSD L2 pass, image input rejected (text-only bundle). `xhigh` effort is
rejected by the 0731 contract (low/high/max), identical to `main`.

Before these fixes (PR head `661e9173`, AR): decode 26.8 tok/s, 1.2k prefill 488 pp/s, tool calls broken,
KV ~3.3 KB/token.

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
5. 0 < k < W: commit in place (Python `commit_prefix`, jangq 95ef4ea), with no second forward.
   - The verify forward ran with a capture on the holder cache. Each attention layer recorded its
     post-RoPE local KV rows, and kv-source layers also recorded their compressor input.
   - `DeepseekV4Model.dsparkCommitPrefix` restores the checkpoint in place. Per layer it re-appends
     the first k+1 KV rows and replays the compressor on the kept inputs, with the same pending
     tail, completed groups, compressed/index pool rows and RoPE positions as a forward of only
     those tokens.
   - The n-gram history and the hidden tail keep their verify records up to start + k + 1.
   - It emits [p0, d1…dk]; truth_k (argmax of row k) becomes the next p0.
6. k == 0, or `VMLX_DSV41_DSPARK_REFEED=1`: restore in place, run one commit forward of
   [p0, truth_0], and take the next p0 from its last row.
7. A block is cut at the first stop token and at `maxTokens`. The cache never runs ahead of the
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
| `VMLX_DSV41_DSPARK_REFEED=1` | partial accepts restore and refeed instead of committing in place (A/B) |
| `VMLX_DSV41_DSPARK_PREFILL_STEP=<n>` | prefill outer chunk while the head is resident (default 1024; `DSV4_PREFILL_OUTER_CHUNK` still wins) |

### CPU tests (`MLXLMCommonFocusedTests`)

- `DSparkSpeculationFocusedTests`:
  - greedy identity against AR on a synthetic attention target (partial acceptance refed and committed in place, zero and full acceptance), with caches equal to AR's;
  - restore after the rotating window wrapped;
  - stop-token and budget cuts;
  - the governor under a synthetic clock (port of `dsv41_dspark_governor_test.py`);
  - the hidden tail;
  - `TokenIterator` end to end.
- `DeepseekV41DSparkFocusedTests`:
  - `DeepseekV4Cache` in-place restore;
  - `commitCapturedPrefix` on a V4.1 kv-source/index-owner layer at ratio 2 and 1: keeping all
    tokens equals the verify's own state bit for bit, and keeping 1–5 equals a forward of only
    those tokens (row counts and metadata exact, values within bf16 batch tolerance);
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
   (Python reference with the in-place partial commit: code 57.5, code2 48.7, Rust 59.1, prose 28.8;
   with refeed: code 44.9, code2 41.3; AR about 30). `prefixCommit` should account for most
   non-full steps, and `commitForwards` only for the k == 0 ones. A/B with
   `VMLX_DSV41_DSPARK_REFEED=1`. Prose should
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

## Multimodal (image input, branch `feat/dsv41-vision-swift`)

Port of the working vmlx-py path: `jang_tools/dsv41/vision.py` (tower and preprocessing),
`DSV41.forward(images=, token_types=)` / `Model.merge_image_embeddings` (splice, `bias_vl`,
Engram masking), and the engine rules in `vmlx_engine/models/deepseek_v4_1/dsv41_vl_preprocess.py`
and `utils/single_batch_generator.py`. Images stay in the turn they were sent in. A text
continuation re-sends them and hits the prefix cache. A different image misses.

### Design

- **Tower** (`MLXLLM/Models/DeepseekV41Vision.swift`):
  - the ViT runs 32 blocks with full bidirectional attention over one image's patch grid, 2D RoPE
    (row positions in the first half of the rotary dims, column positions in the second) and fp32
    RMSNorm;
  - the Aligner zero-pads to a multiple of 3, does a 3x3 merge (column order `c*9 + kh*3 + kw`, as
    `F.unfold`) and runs `w2(gelu_erf(w1(x)))`;
  - weights load from the bundle names under `v41_vision.` (`vision.*`, `aligner.*`,
    `image_{start,end,newline}`) as BF16 passthrough, about 1 GB resident.
- **Preprocessing** (CPU, per image):
  - `plan_image_grid`: images smaller than `min_pixels` are scaled up with `int()` truncation, the
    canvas rounds up to multiples of 14, and over `max_image_tokens` the canvas is re-solved;
  - `ImageOps.pad(color=127)` with Pillow's exact BICUBIC (22-bit fixed point, an 8-bit
    intermediate, half-to-even rounding), then `(x/255 - 0.5)/0.5` and patches `[n, 3, 14, 14]`;
  - request images decode through an unmanaged software `CIContext`, which keeps the stored RGB
    values, as PIL does.
- **Prompt** (`LLMUserInputProcessor`, V4.1 with its tower only):
  - each image renders as `<｜deepseek_image｜>` (token 129264) in its own message, in content-part
    order. Without an explicit order the images go first. Parts are joined with `\n\n`, which is
    `encoding.py process_image_messages`;
  - rendering goes through the same chat template as text, so an image turn and its text
    continuation share one prompt format;
  - each placeholder expands to `llmH*(llmW+1)+2` copies of the id (START, (IMAGE x w, NEW_LINE) x h,
    END) and the cache boundaries are remapped into the expanded space;
  - `LMInput.image` carries the bf16 patches with one `THW(1, vitH, vitW)` per image, and
    `mediaTokenIds = [129264]`.
- **Prefill** (`DeepseekV4Model.prepareImages`):
  - spans are found from the THW grids. The tokens hold either every image or, after a restore,
    the last k images;
  - chunks follow the engine's `_dsv41_vl_chunk_len`: a chunk end that falls inside a span moves
    back to stop before the image, or forward to take the whole image when the image starts the
    chunk. Chunks never split a span;
  - each image is encoded in the chunk that holds it and its rows overwrite the span
    chunk-relative;
  - inside that chunk, image tokens choose experts with `gate.bias_vl` (eager selector) and text
    tokens with `bias`. Image tokens hash DEAD in Engram: they take part in no n-gram, they block
    the n-grams that span them, and the DEAD ids travel in the cache-carried history. Their Engram
    gate is zeroed;
  - the remainder handed to the caller's final forward is text-only, at most one prefill step.
- **Prefix cache**:
  - the media salt hashes the patch bytes and grids, combined with scope and KV-policy salts;
  - `DeepseekV4Model` makes no pre-media promise (`PrefixCacheInvariantModel` is not adopted),
    so every boundary of an image request is keyed under the whole-request salt, like the
    engine's `dsv41_vl_media`;
  - a text follow-up re-sends the same image, gets the same salt and hits the previous turn's
    boundary. Its suffix holds no placeholder, so it prefills as plain text;
  - a different image misses every boundary, including the text before the image.
- **dSpark**: decided on the input that is actually prefilled. After a hit that covers every
  image the tail is text-only and dSpark serves it. A request that prefills an image decodes AR
  (`[dSpark] off ... media inputs`).

### Gating

Everything above requires DeepSeek-V4.1 (`isDeepseekV41`), a `vision_config` with layers, and
images in the request. Without them nothing changes:

- **DSV4-0731**: never decodes `vision_config`, never builds a tower or `bias_vl`, and sanitize
  drops the vision tensors as before.
- **Text-only V4.1** (no images, or every image inside a restored prefix): takes the unchanged
  text prefill (`prefillOuterStep` is an extraction of the old code).
- **Text-only construction**: `VMLX_DSV41_VL=0`, a caller that requests modalities without
  `.vision`, and the JANGTQ variant all build no tower. Images are then ignored, as before.
- **Modalities**: `DeepseekV4Model` reports `[.text, .vision]` through `ModalityBearing` only
  when the tower exists.

### Osaurus (branch `feat/dsv41-vision`)

- `LocalVisionEvidence` grants vision to a `deepseek_v41` bundle when all of these hold:
  - the linked vMLX's `DeepseekV4Model` is `ModalityBearing`;
  - `vision_config.num_hidden_layers` is set and `image_token_id` is present;
  - every `vision.blocks.N` plus `patch_embed`, `vision.norm`, `aligner.w1/w2` and the three
    delimiters are backed by tensors.
- The bundle needs no processor file. `VLMDetection.isVLM(modelType: "deepseek_v41")` follows the
  same runtime check.
- `VMLX_DSV41_VL=0` turns the evidence off too, so the text-only construction is admitted.
- Under the current vMLX pin (`264f6c4c`) the check is false, so V4.1 stays text-only instead of
  being refused at load ("loaded runtime has no vision tower"). Bump the pin to a vMLX revision
  containing this branch to turn images on.
- Image parts already reach the VL path through `mapOpenAIChatToMLX`:
  - each `Chat.Message` keeps its own images and `contentParts`;
  - `validateRuntimePolicy` and the composer read the same evidence.

### Environment

| variable | effect |
|---|---|
| `VMLX_DSV41_VL=0` (`false`/`off`/`no`) | build V4.1 text-only (no tower, no `bias_vl`) |
| `VMLX_DSV41_VL_LOG=1` | NSLog `VMLX DSV4.1 VL preprocess: N tokens, S image-span tokens, k image(s) WxH->hxw` and one `VMLX DSV4.1 VL prefill chunk [a, b): image i at r (hxw, n tokens)` line per image chunk |
| `VMLX_CACHE_FETCH_TRACE=1` | existing salt trace: `[vmlx][cache/salt] ... media=<12 hex> image=[[1521, 3, 14, 14] bfloat16 grids=1x39x39 ...]` |
| `VMLX_DSV41_DSPARK_STATS=1` | existing: shows `[dSpark] off for this request: media inputs` on an image prefill |

### CPU tests (`DeepseekV41VisionFocusedTests`, 22 tests)

- **Python parity**:
  - grid plans and token counts for 12 sizes match `plan_image_grid`;
  - Pillow BICUBIC resize is byte-identical in 4 cases, both up and down;
  - pad, normalize and patchify are bit-identical to `load_image` (canvas SHA-256 and float32
    patch SHA-256) in 4 cases;
  - red and blue 64x64 images give 39x39 patches and 184 tokens;
  - PNG decodes round-trip exactly through `ciImage` and file URLs.
- **Bundle contract** (safetensors headers only): every tower parameter maps to a BF16 bundle
  tensor of the same name and shape (266 tensors), and every layer has a `bias_vl` of the gate
  bias's shape.
- **Tower**: on tiny deterministic weights, ViT + Aligner match the NumPy port of the vision.py
  math within 2e-5. Span rows are laid out START / rows + NEW_LINE / END.
- **Prompt**: `encode_messages` goldens for the image turn, the follow-up, a mid-conversation
  image and two images, in thinking and chat modes. Placeholder text is refused, loose images
  go to the last user turn, and expansion and boundary remapping are checked.
- **Splice**:
  - span location covers full and trailing sets, adjacent images and partial spans;
  - the never-split rule holds, and 21 chunk plans keep whole spans with a text-only remainder;
  - chunk-relative splice and the masks are checked;
  - a tiny V4.1 backbone (synthetic Engram bundle) gives the same image-prefill logits for outer
    chunks of 4, 5, 6, 9, 13 and 64 (< 1e-4);
  - other pixels change the logits, and a span-free suffix takes the text path.
- **Routing and Engram**: `bias_vl` routes image tokens. An all-text mask equals the compiled
  gate. Engram DEAD hashing is checked, and chunked hashing with the carried history equals
  one-shot.
- **Cache keys**: the same image gives the same salt across turns and a different image gives a
  different salt. The plan keys every boundary under the request salt. The boundary and
  suffix-media checks pass.
- **Unchanged paths**:
  - text-only V4.1 logits are bit-identical with and without the tower;
  - 0731 builds no tower or `bias_vl`, and sanitize drops the vision tensors;
  - `VMLX_DSV41_VL` and config decode and round-trip are checked.
- **Regression**: all V4.1 / dSpark / DSV4 focused suites (82 tests in 11 suites) and the DSV4
  disk round-trip, partial-restore and media-prefix suites still pass.

### Not verified (needs the GPU and the real weights)

- Image understanding on the real tower (bf16, 1521 patches, Aligner into 5120). The only check
  is CPU parity of the math on tiny weights.
- Pixels of images with alpha or an ICC profile. CoreImage premultiplies and PIL does not, so
  transparent PNGs can differ from the reference. Opaque sRGB images match.
- ViT time and memory per image, and the image chunk inside the 2048 (1024 with dSpark) outer
  chunk.
- Tool-result images: the engine nests them in `tool_result` blocks; Swift does not render them.
- Batched multi-request slots with images (the solo `TokenIterator` path is the target).

### GPU verification checklist (lead; one model load at a time)

Load V4.1 in Osaurus once the vMLX pin includes this branch, with `VMLX_DSV41_VL_LOG=1`,
`VMLX_CACHE_FETCH_TRACE=1` and `VMLX_DSV41_DSPARK_STATS=1`. Use temperature 0 and thinking off
(chat rail) for fast answers.

1. Load: the model loads with no "no vision tower" error, the picker shows image input, and
   phys_footprint is about 1 GB above the text-only load.
2. Image turn: send a 64x64 solid red PNG with "What color is this image? Answer in one word."
   - The log shows `VL preprocess: ... 184 image-span tokens, 1 image(s) 546x546->13x13` and one
     `VL prefill chunk ... image 0 at <r> (13x13, 184 tokens)`.
   - The answer is "Red".
   - `[dSpark] off for this request: media inputs`, and cached tokens are 0 (cold).
3. Follow-up: in the same chat send "Name a fruit that has this color." (no new image).
   - The answer is a red fruit (apple, strawberry, cherry).
   - Usage `cached_tokens` > 0 and covers the image span, so it is at least about 190.
   - The salt trace shows the same `media=` digest as turn 2.
   - There is no `VL prefill chunk` line, since the tail is text-only.
   - With dSpark on, `[dSpark]` stats appear, because dSpark serves the text-only tail.
4. Different image: in a new chat send a 64x64 solid blue PNG with the same question.
   - The answer is "Blue".
   - The `media=` digest differs and `cached_tokens` = 0: a cache miss, even for the shared text
     prefix.
5. Mid-conversation image: text turn, then an image turn, then a text turn.
   - The image turn misses (new media salt) and answers correctly.
   - The third turn hits with `cached_tokens` > 0 and still refers to the image.
6. Long prompt: put the image after more than 2048 prompt tokens.
   - The chunk log shows the image whole inside one chunk, never split.
   - The answer is correct.
7. Text-only regression: V4.1 without images gives the same tokens and tok/s as before (dSpark
   stats in the same range). 0731 is unchanged.
8. `VMLX_DSV41_VL=0` (set for the Osaurus process): V4.1 loads text-only with no tower error,
   and the picker shows no image input. Osaurus reads the same switch.
