# MiniMax-M3 (mm3) — E2E Verification & Confirmation Spec

> **STATUS (2026-06-21):** WIRED + BUILD GREEN, ready for user live testing.
> vmlx `codex/mm3-runtime`@`409d2c47` rebased on main (incl cache B1/B2). osaurus
> `codex/minimax-m3-integration` rebased + repinned to `5139e84a`; Release build
> SUCCEEDED (MM3 engine integrates). The §6 script + §7 scan against the 95 GB
> model is the remaining merge gate — RAM-gated, user-run.

How to confirm and live-prove the MiniMax-M3 engine + osaurus integration end-to-end before merge.
Paired PRs: **vmlx-swift #75** (`codex/mm3-runtime`, engine) + **osaurus #1576** (`codex/minimax-m3-integration`).
Model: `MiniMax-M3-REAP40-d3-JANG_2L` (REAP 128→77 experts, 95 GB) at `~/models/JANGQ-AI/MiniMax-M3-REAP40-d3-JANG_2L`.
REAP-pruned → gate is **coherent non-garbled text + correct mechanics**, NOT answer correctness.

---

## 0. Architecture recap (what makes M3 special — drives the test matrix)

- **60 decoder layers, two attention regimes:**
  - **Layers 0–2:** dense full-causal attention → stock `KVCacheSimple`.
  - **Layers 3–59:** **MiniMax Sparse Attention (MSA)** via a **Lightning Indexer** → `MiniMaxM3SparseCache`.
- **GQA:** `n_kv=4`, `head_dim=128`. **Partial RoPE** (`rotaryDim=64`, `traditional:false`/NeoX half-split, base=`ropeTheta`); manual `rope→SDPA` (not `attentionWithCacheUpdate`).
- **Lightning Indexer / MSA:** scores fresh `idx_q` against ALL cached `idx_k`, **max-pools per 128-token block**, **max over heads**, selects **top-k blocks** per query → additive `[B,1,Sq,Sk]` keep-bias (0 kept / −inf else). The query's **own (local) block always wins**.
  - **Below `topk*block = 2048` tokens:** every block visible → indexer returns `nil` → caller uses plain causal (full attention). **Past 2048:** sparse selection fires.
  - `idx_q` is **recomputed every step, never cached**. Only `idx_keys` are cached.
  - **Blocks anchored to ABSOLUTE position** (`block = pos/128`) → cache is **append-only / trim-and-replay only** (never re-key on reuse).
- **`MiniMaxM3SparseCache` = 3 lanes that MUST move together:** `keys [B,4,S,128]`, `values [B,4,S,128]`, `idx_keys [B,1,S,128]`, sharing one `offset`. `copy()` returns a `MiniMaxM3SparseCache`; `state`/`metaState` round-trip all three through disk/prefix tiers. **DEBUG asserts `idx_keys.len == offset`.**
  - **THE failure mode:** any reuse layer that downcasts to a plain KVCache copies only `(keys,values)` and **drops `idx_keys`** → indexer scores against corrupt/empty keys → **decode loops**. (This was the Python loop bug.) Every cache path must keep it first-class.
- **Reasoning envelope:** `<mm:think>…</mm:think>` (NOT m2's `<think>`) → dedicated `minimax_m3` reasoning stamp (`ReasoningParser.swift`).
- **Tool calls:** `MiniMaxM3ToolCallParser` (registered in `ToolCallFormat`).
- **Quant (mixed, all gs=64):** embed=6, attn/lm_head=8, routed gate/up=2, routed/shared down=3 or 6, indexer/router-gate/norms fp16. Loader resolves `(bits,gs)` from each module's true input dim (see [[mm3-quant-bits-gs-ambiguity-fix]]).
- **Runtime constraints for M3 (must be enforced + verified):** **paged OFF**, **TurboQuant-KV skip**, **JIT/compile OFF**. Gated via `ModelContainer.requiresMiniMaxM3SparseState`.

---

## 1. Pre-flight (do before any live test)

| # | Step | Pass criteria |
|---|------|---------------|
| P1 | **Rebase `codex/mm3-runtime` onto current vmlx main** (now includes the merged disk-cache B1/B2 fixes, commit `1b7d3ef`). MM3 touches `CacheHelpers/TQDiskSerializer/KVCache/Load/ModelContainer`; B1/B2 touched `DiskCache/Evaluate/BatchEngine` → **no file overlap, expect clean rebase**. | rebase applies with no conflicts; `git diff --name-only` overlap check = none |
| P2 | **Build vmlx** (Xcode toolchain) | compiles clean |
| P3 | **Autodetect / load** — model appears in `/v1/models`; loads without the bits×gs crash | listed; `loadModel` succeeds; no `[quantized_matmul] ... does not match` fatal |
| P4 | **Cache layout** assertion | 60 caches: layers 0–2 = `KVCacheSimple`, layers 3–59 = `MiniMaxM3SparseCache` |
| P5 | **Constraints enforced** | paged off, TQ-KV skipped, JIT off for M3 (assert via the coordinator config / `requiresMiniMaxM3SparseState`) |
| P6 | **RAM safety** — 95 GB model on 128 GB box; single residency; no OOM | loads; RSS/unified-mem within budget; no eviction thrash |

---

## 2. Engine smoke (vmlx, gated test) — load once, harvest all

Run `MiniMaxM3SmokeTests` (one `loadModel`, RAM-heavy):
```
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer MM3_MSA_TRACE=1 \
  swift test --filter MiniMaxM3SmokeTests
```
(metallib gotcha: copy `mlx/default.metallib` into the `.build` lookup sites — see [[mm3-live-test-recipe]].)

| # | Signal | Pass criteria |
|---|--------|---------------|
| S1 | Short forward (< 2048 ctx) | runs; indexer returns `nil` (full attention); coherent logits |
| S2 | **MSA fires past 2048** | `MM3_MSA_TRACE` prints `[MM3-MSA] full attention …` then `[MM3-MSA] SPARSE SELECTION FIRED (… keep=N of topk=…)`; proven before at Sk≈2304 (16/18 blocks) |
| S3 | BatchEngine generation | `[MM3-GEN] visible=<<<…>>>` is coherent, no garble, no loop |
| S4 | **SparseCache round-trip** (`MiniMaxM3SparseCacheTests`) | `copy()/state/metaState/trim` preserve `idx_keys`; DEBUG assert `idx_keys.len == offset` holds across append/trim/restore |
| S5 | **Tool parser** (`MiniMaxM3ToolCallParserTests`) | parses the M3 tool envelope; no false positives on prose |

---

## 3. LIVE E2E MATRIX (osaurus, the real chat-engine path) — the gate

Drive headless via osaurus `/v1/chat/completions` + `/agents/default/run`. Read **every** response for: garble, looping, incoherency, markup leak, and **streaming content-delta correctness**. Varying prompt per turn.

### A. iRoPE / partial-rope + position correctness
- **A1** Short prompt → coherent (rope correct at low positions).
- **A2** Long prompt that crosses 2048 → coherent (positions stay absolute & monotonic across the full→sparse transition; partial-rotary 64/128 applied to the right dims).
- **A3** After cache trim/reuse, positions resume correctly (no off-by-one at the prefill→decode or reuse boundary).
- **Pass:** no degradation/repetition as context grows; the full→sparse crossover at ~2048 produces no discontinuity.

### B. MSA / Lightning-Indexer (block-sparse attention)
- **B1** Force context > 2048 (long system prompt or multi-turn) and confirm **`SPARSE SELECTION FIRED`** in the engine trace AND output stays coherent (sparse selection didn't drop needed context).
- **B2** **Local-block-wins:** the most recent tokens are always attended (the model can quote/continue the immediately preceding text verbatim past 2048).
- **B3** **Recall across the sparse window:** a fact placed early (block 0) and queried after >2048 tokens is still retrievable (top-k selects the right block) — within REAP capability.
- **Pass:** indexer fires, no loop, coherent recall of both local and selected-distant blocks.

### C. Reasoning matrix (`<mm:think>`)
- **C1** reasoning ON → `<mm:think>…</mm:think>` is parsed to `reasoning_content` (NOT leaked into `content`); content is the post-think answer; coherent.
- **C2** reasoning OFF → no think envelope; coherent answer.
- **C3** reasoning across **multiturn** (think on turn 2 after a cache-hit turn 1) → think still parsed, no `<mm:think>` leak, no loop.
- **Pass:** `<mm:think>` always routed to reasoning_content; never the `<think>` (m2) mis-stamp; no CoT leak; no loop.

### D. Tool-call matrix
- **D1** with tools → emits a structured tool_call (M3 envelope parsed by `MiniMaxM3ToolCallParser`); no markup leak into content.
- **D2** **tool round-trip:** model calls tool → tool result fed back → model produces coherent final answer.
- **D3** **reasoning + tool** (think then call) → `<mm:think>` parsed AND tool_call parsed in the same turn; correct order; no leak/loop.
- **D4** tool_choice=required → valid call; no-tools → coherent prose (no spurious tool markup).
- **Pass:** structured calls, clean round-trip, reasoning+tool coexist.

### E. System-prompt injection
- **E1** Custom system prompt is injected correctly for M3 (osaurus chat-context compose) and honored in output.
- **E2** With tools, the tool block + system prompt render in the right order (no duplication, single BOS, correct M3 turn markers).
- **E3** Capture the **rendered prompt** (debug dump) and confirm: one BOS, correct M3 chat template, tools in the M3 format, reasoning prefill correct for the reasoning toggle.
- **Pass:** rendered prompt is well-formed; output reflects the system prompt.

### F. Prefix / disk-L2 cache reuse — **the critical 3-lane invariant**
- **F1** **Multiturn warm reuse:** turn 1 (cold) → turn 2 (same session, growing) reuses turn-1 prefix. Confirm a cache hit AND that the restored cache is a `MiniMaxM3SparseCache` with `idx_keys` intact (NOT downcast). **Output coherent, zero loops.**
- **F2** **idx_keys round-trip through disk:** after an unload/reload (or cross-process disk hit), the restored sparse cache still carries `idx_keys` (state/metaState serialized all 3 lanes). The indexer past 2048 still selects correctly post-restore.
- **F3** **Trim-replay:** a reuse that requires trimming to a history boundary replays append-only (blocks stay absolute-positioned); no re-keying; coherent.
- **F4** **Negative guard:** confirm NO generic reuse path downcasts the sparse cache to plain KVCache (the loop bug). Greppable invariant + live: a warm sparse-context turn does not loop.
- **Pass:** warm reuse on M3 = coherent, idx_keys preserved on every path (paged-off, disk, trim).

### G. Cache syncing — between layers, processes, and rederive type
- **G1** **Cross-layer sync:** dense layers (0–2, KVCacheSimple) and sparse layers (3–59, MiniMaxM3SparseCache) advance on a **shared offset**; a restore lands ALL 60 layers at the same boundary (no dense/sparse desync → would corrupt).
- **G2** **Cross-process / disk sync:** an entry stored in one run is fetched in another (disk-L2) and both lanes (dense KV + sparse 3-lane) restore consistently; modelKey isolation holds (no cross-model alias).
- **G3** **Async / rederive type:** M3 sparse cache is **append-only / trim-replay** — confirm there is **no async detached rederive** racing decode (unlike SSM); the trim path is synchronous/actor-serialized. (M3 has no conv/SSM state; verify nothing schedules a background rederive that could read half-written `idx_keys`.)
- **G4** **Multi-cache-type coexistence:** the 60-layer cache (mixed KVCacheSimple + MiniMaxM3SparseCache) is handled first-class by the coordinator (protocol-driven, no type allow-list); store/fetch/trim treat the composite correctly.
- **Pass:** all 60 layers stay synced through every reuse; no desync, no race, no allow-list drop.

### H. osaurus cache-window processing + wiring
- **H1** **KV window/cap:** the osaurus memory-safety slider / KV cap governs M3's live context correctly (the resolved cap reaches the coordinator); a long session respects the window without garbling.
- **H2** **Constraints wired osaurus-side:** paged OFF, TQ-KV skip, JIT off are applied for M3 via the runtime config (not just engine-side); confirm `/health` / coordinator tags reflect it.
- **H3** **Reasoning/tool wiring:** osaurus surfaces M3 `reasoning_content` + `tool_calls` correctly (picks up the engine's minimax_m3 stamps via jang_config/capabilities); model autodetect sets the right reasoning + tool format.
- **H4** **Prefill-progress / streaming:** prefill counter + streaming deltas behave for M3 (no frozen counter, monotonic, content deltas contiguous).
- **Pass:** osaurus drives M3 with correct window, constraints, reasoning/tool surfacing, and streaming.

### I. THE MERGE GATE — multiturn cache-reuse, zero loops, coherent
A ≥6-turn session on `MiniMax-M3-REAP40-d3-JANG_2L` (paged off, TQ-KV skip, JIT off) that interleaves: a long turn crossing 2048 (MSA fires), a reasoning turn (`<mm:think>`), a tool round-trip, and cache-hit (warm) turns. **Every cache-hit turn must be coherent, zero loops, zero incoherency, no markup leak.** Scan all deltas.
- **Pass = the whole session is loop-free and coherent, with the sparse cache reused (idx_keys preserved) on the warm turns.** This is the not-to-main bar.

### J. Streaming-delta + stress + RAM
- **J1** Streaming content deltas are contiguous/monotonic, no dup/garble, reasoning vs content split correct.
- **J2** Per-turn-varied multirun (≥10) at non-zero temperature → scan every output for loop/garble/leak (catch non-deterministic faults).
- **J3** RAM: sustained session → no leak, no crash, disk-L2 bounded (ties to B1 orphan-row fix already on main).

---

## 4. Run-book (commands)

- **Engine smoke / MSA trace / sparse-cache / tool-parser tests:** §2 command above.
- **Live osaurus matrix:** build osaurus against the rebased vmlx (repin `codex/minimax-m3-integration` → the mm3 vmlx commit), launch with a test root, drive `/v1/chat/completions` (reasoning toggle via `enable_thinking`/`reasoning_effort`, tools via `tools`) and `/agents/default/run`; capture rendered prompt with the `VMLX_DUMP_PROMPT` env (swift-transformers Tokenizer render site) for §E; inspect the disk-cache SQLite for §F/§G accounting; `MM3_MSA_TRACE=1` for §B.
- **Cache accounting helper:** `cacheq.sh <root>` (rows vs files vs SUM(file_size), delta=0).

## 5. Definition of done (both PRs out of WIP)
- [ ] P1–P6 pre-flight green (rebase clean, builds, loads, layout, constraints, RAM).
- [ ] S1–S5 engine smoke green (MSA fires, sparse-cache round-trip, tool parser).
- [ ] A–H live matrix green on the real model (iRoPE, MSA, reasoning, tools, sys-prompt, prefix/disk reuse with idx_keys preserved, cross-layer/process sync, osaurus window/wiring/streaming).
- [ ] **I merge gate**: ≥6-turn multiturn, zero loops, coherent, sparse cache reused warm.
- [ ] J streaming/stress/RAM green.
- [ ] vmlx #75 + osaurus #1576 rebased onto current mains, CI green, flipped out of `[WIP]`, repin osaurus → merged mm3 vmlx commit.

---

## 6. CONCRETE multiturn live script (drive this exact session, same `session_id`)

Run on `MiniMax-M3-REAP40-d3-JANG_2L`, paged off / TQ-KV skip / JIT off. **Read every turn deeply per §7.** Vary the prompt TYPE each turn; keep one long system prompt so context crosses 2048 by mid-session.

System prompt (long, ~600+ tokens so the session crosses the 2048 MSA threshold by turn 3-4): a detailed assistant persona + a planted fact: *"Remember this session key: ORBIT-7731."*

| Turn | Type | Prompt (vary each run) | What it exercises | Specific pass check |
|---|---|---|---|---|
| 1 | short text, reason OFF | "In one sentence, what is photosynthesis?" | cold prefill, rope@low-pos, baseline | 1 coherent sentence; no `<mm:think>` leak; no junk |
| 2 | text, **reason ON** | "Explain why the sky is blue. Think first." | `<mm:think>` parse, warm reuse of turn-1 prefix | think→reasoning_content (NOT content); answer coherent; **warm cache hit** (TTFT drop); no loop |
| 3 | **long-form, reason OFF** | "Write ~400 words on the history of computing, with specifics." | crosses 2048 → **MSA fires**; long coherent gen | `MM3_MSA_TRACE` shows SPARSE SELECTION FIRED; text stays coherent to the end (no late-context degeneration/repetition) |
| 4 | **tool + reason ON** | "What's the weather in Tokyo? Think, then use the get_weather tool." | reasoning + tool same turn; M3 tool parse | `<mm:think>` parsed AND a structured `get_weather` tool_call; no markup leak; correct order |
| 5 | **tool round-trip** | (feed tool result `{"temp":"18C"}` back) | tool result consumed → final answer | coherent final answer using 18C; no re-loop of the call; no leak |
| 6 | **recall, reason OFF** | "What was the session key I gave you at the start?" | cross-turn recall past 2048 (idx_keys preserved + local/distant block selection) | returns **ORBIT-7731** (top-k selected block 0) AND warm reuse coherent; **proves sparse cache reused, idx_keys intact** |
| 7 | **code, reason ON** | "Write a Python function to reverse a linked list. Think about edge cases." | reasoning + code formatting, warm reuse | think parsed; valid code in content; no `<mm:think>`/backtick-fence corruption; no loop |
| 8 | **long + tool, reason ON** | "Summarize our whole conversation, then use get_weather for London." | deep context + sparse reuse + tool, max stress | coherent summary referencing earlier turns; structured tool_call; no loop/leak across the longest context |

**Repeat the whole 8-turn session ≥3×** at temperature 0.7-0.9 (sampling) with **different prompts each run** to catch non-deterministic faults. Then a **≥10-iteration single-turn multirun** mixing all types at temp 0.9.

## 7. DEEP output scanning (apply to EVERY turn — this is the crucial gate)

For each response, scan `content` AND `reasoning_content` AND the raw stream:

**A. Looping** (any → FAIL):
- Repeated n-gram: any 12–40 char window appearing ≥4× (excluding legit list bullets/markdown).
- Repeated sentence/line ≥3×.
- Repeated special/reasoning token: `<mm:think>`, `</mm:think>`, `<|`, channel/think markers ≥3×.
- Degenerate tail: the last 20% is near-identical repetition of earlier text.
- **M3-specific:** the indexer-loop signature — generation that stalls into repeating the same phrase past 2048 tokens = the idx_keys-dropped bug. Treat as CRITICAL.

**B. Incoherency** (any → FAIL):
- Non-ASCII junk: control/garbage chars beyond legit unicode (count > a few).
- **Broken words / char corruption:** inserted/dropped letters (e.g. `stateFcile`, `Amerin`, `rtificial`) — scan for mid-word breaks, impossible letter runs. (This was the gemma symptom; watch for it on M3 too, esp. after a cache hit or past 2048.)
- Abrupt topic break / non-sequitur mid-response.
- Empty content with `finish=length` (all output went to a discarded bucket — reasoning/special).

**C. Leaking** (any → FAIL):
- `<mm:think>` / `</mm:think>` appearing in **content** (reasoning leaked) — must be in reasoning_content only.
- Tool envelope markup (the M3 tool tags / `<tool…`) in content instead of a parsed `tool_calls`.
- Special tokens (`<|`, BOS/EOS literals, turn markers) in content.
- The wrong reasoning tag: `<think>` (m2) instead of `<mm:think>` (m3) → indicates the minimax (not minimax_m3) stamp fired = wiring bug.

**D. Streaming-delta correctness:**
- Content deltas concatenate to a clean string (no dup/overlap/gap between chunks).
- reasoning_content vs content split is clean (no bleed at the `</mm:think>` boundary).
- Monotonic; finish_reason present; no truncated UTF-8 mid-grapheme.

**E. Cache/mechanics (per warm turn):**
- Warm turn shows a TTFT drop (cache hit) vs the equivalent cold.
- `cacheq.sh` accounting delta = 0 across the session (no orphan rows — B1).
- No `[MM3] downcast`/plain-KVCache restore in logs (idx_keys preserved).
- `MM3_MSA_TRACE`: full→sparse transition fires once, cleanly.

## 8. Fix playbook (what each failure points to)

| Symptom | Most-likely cause | Where to look |
|---|---|---|
| Loop/repeat past 2048, esp. after warm reuse | idx_keys dropped (sparse cache downcast to plain KVCache on a reuse path) | `MiniMaxM3SparseCache.copy()/state/metaState`; the coordinator fetch/restore path; assert `idx_keys.len==offset` |
| `<think>` instead of `<mm:think>` parsed / CoT leak | minimax (m2) stamp instead of minimax_m3 | engine `ReasoningParser` stamp (`minimax_m3`); osaurus `isMiniMaxFamily` sites (OpenAIAPI:155, MLXBatchAdapter:829) — verify M3 isn't forced to m2 handling |
| Tool markup leaks / not parsed | M3 tool format not resolved | `ToolCallFormat.infer(model_type=minimax_m3)`; osaurus `supportsLocalToolCalling`/`resolvedToolCallFormat` (jang_config tool_parser=None → infer path) |
| Char corruption / junk (esp. past 2048 or post-hit) | quant (bits×gs) or RoPE/MSA numeric | quant resolve at `Load.swift` (mm3 mixed bits); partial-rope dims; indexer max-pool/top-k math |
| Garble only on cache-hit turns | trim-replay / restore mis-keys sparse blocks (absolute pos) | sparse cache `trim`; disk round-trip of all 3 lanes; cross-layer (dense 0-2 vs sparse 3-59) offset sync |
| Empty content, finish=length | everything routed to reasoning (open `<mm:think>` never closed) | template reasoning prefill + the `<mm:think>` close; reasoning parser tail-aware init |
| Late-context degeneration | KV window/cap too small OR MSA dropping needed blocks | osaurus memory-safety slider → coordinator KV cap for M3; indexer top-k `keep` value; local-block-wins |
| Cross-process garble | disk-L2 didn't round-trip all 3 lanes / modelKey alias | `MiniMaxM3SparseCache.state`/`metaState` serialize; disk modelKey tag includes M3 topology |

## 9. osaurus wiring status (this PR)
- vmlx `codex/mm3-runtime` rebased onto main (`5139e84a`, includes merged cache B1/B2). osaurus `codex/minimax-m3-integration` rebased + repinned (Package.swift + 2 Package.resolved + RuntimePolicySource hardening test all → `5139e84a`).
- Engine owns: model, MSA/indexer, sparse cache, `minimax_m3` reasoning + tool parsers, quant resolve, registration.
- osaurus family handling for M3 (verify live, do NOT speculatively change): `isMiniMaxFamily` → `shouldEnableCompiledBatchDecode=false` (matches JIT-off ✅), `enable_thinking` toggle (MLXBatchAdapter:829), architecture metadata (OpenAIAPI:155). jang_config has reasoning_parser/tool_parser = None → engine infers from `model_type=minimax_m3`.
- **Open verification (live, RAM-gated, user-run): the entire §3 matrix + §6 script + §7 scan.** Build (Release) compiling now = the static integration check.
