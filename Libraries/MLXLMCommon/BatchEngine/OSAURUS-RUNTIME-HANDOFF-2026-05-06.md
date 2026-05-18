# Osaurus Runtime Handoff - 2026-05-06

> ⚠️ **DEPRECATED 2026-05-10.** Superseded by [`docs/OSAURUS-INTEGRATION-HANDOFF-2026-05-09.md`](../../../docs/OSAURUS-INTEGRATION-HANDOFF-2026-05-09.md).
>
> **Read the 2026-05-09 handoff first.** It covers the current pin (`7273ba2`),
> ZAYA reasoning-split correction (this doc still describes ZAYA as
> "served as non-reasoning" — that framing was corrected), Hy3 native
> runtime, ZAYA1-VL native-adapter status, cache-scope salt, generation_config
> defaults, JANGTQ top-k override plumbing, and B>1 admission coalescing.
>
> Kept here as a historical record for the DSV4 / Ling / Bailing / Gemma /
> Laguna / Qwen / MiniMax / Nemotron-Omni cache-tier and TurboQuant KV
> notes that were accurate at 2026-05-06.
>
> **MiniMax speed supersession 2026-05-10:** `a5a0e37` had no committed
> `soloFastPath*` branch and routed `BatchEngine.generate` through
> actor-managed `submit(...)`. It also lacked the JANGTQ Hadamard `newv[8]`
> and cached-meta optimization that stale notes claimed were already present.
> Both are restored in the current working tree. Fresh Release rows show
> `MiniMax-M2.7-JANGTQ` at 46.6 tok/s through `BatchEngine.generate`, and
> 46.4 tok/s with a production-style `CacheCoordinator` attached. See
> `docs/MINIMAX-OSAURUS-DECODE-SPEED-DISCREPANCY-2026-05-10.md`. The
> 74 GB `JANGTQ_K` / CRACK rows still need a follow-up run in a clear memory
> window.

Audience: osaurus agents wiring `vmlx-swift-lm` into the production runtime.

Scope: text runtime, chat templates, reasoning streams, DSV4, Ling/Bailing,
Gemma/Laguna/Qwen/MiniMax/Nemotron-Omni, BatchEngine cache tiers, SSM companion
state, and TurboQuant KV correctness. Distributed inference and JangPress speed
work are intentionally out of scope for this handoff.

## Current Position

Use `BatchEngine.generate(input:parameters:)` as the production path for normal
osaurus chat/runtime traffic. Use `UserInput(chat:)` for chat models, not a raw
`"User:\nAssistant:"` transcript, whenever template semantics matter.

The runtime is coherent at the minimum bar across the currently tested local
families. Speed tuning remains open, especially MiniMax M2.7 throughput and
Ling MXFP4 affine decode after removing the oversized fused gate/up cache. Do
not block integration on speed unless product policy requires a token/s
threshold.

## Current Validation - 2026-05-07

Current commit under test: `88fc352`.

### Stop-hook full matrix - 2026-05-07 14:45 PDT

This is the newest local evidence set and supersedes older provisional or
background-active speed rows below. The pass used the current release
`RunBench` and one model process at a time.

Strict speed/graph rows that are green:

- Qwen3.5-35B-A3B 4bit:
  `/tmp/vmlx_stop_hook_seq_qwen35_20260507.log`, TTFT 62 ms, 97.5 tok/s,
  3854 graph nodes, 90 `AsType`, no loop/leak.
- Gemma4 E2B:
  `/tmp/vmlx_stop_hook_seq_gemma4_e2b_20260507.log`, TTFT 32 ms,
  164.9 tok/s, 1704 graph nodes, 89 `AsType`, no loop/leak.
- Gemma4 26B A4B:
  `/tmp/vmlx_stop_hook_seq_gemma4_26b_20260507.log`, TTFT 85 ms,
  86.8 tok/s, 3274 graph nodes, 151 `AsType`, no loop/leak. It passes the
  explicit >80 tok/s target, but `AsType` cleanup remains.

Functional rows with open speed or `AsType` work:

- Gemma4 E4B: 106.4 tok/s, 2210 graph nodes, 112 `AsType`.
- Nemotron Cascade: 114.1 tok/s, 2062 graph nodes, 161 `AsType`.
- Nemotron Super: 42.2 tok/s, 3599 graph nodes, 280 `AsType`.
- Ling JANGTQ2: 35.4 tok/s, 5 graph nodes, 3 `AsType`.
- Ling legacy JANGTQ: 35.8 tok/s, 5 graph nodes, 3 `AsType`.
- Ling MXFP4: 9.9 tok/s, 5 graph nodes, 3 `AsType`.
- ZAYA JANGTQ2: 55.6 tok/s, 8831 graph nodes, 1365 `AsType`.
- ZAYA JANGTQ4: 53.7 tok/s, 8831 graph nodes, 1365 `AsType`.
- ZAYA MXFP4: 65.0 tok/s, 8791 graph nodes, 1285 `AsType`.

ZAYA and Ling library integration status:

- ZAYA JANGTQ2/JANGTQ4/MXFP4 contract rows pass. The tested bundles advertise
  `cacheSubtype=zaya_cca`, hybrid cache, expected EOS/template metadata, and
  the expected TQ sidecar state for JANGTQ rows.
- ZAYA template smoke passes. `enable_thinking=false` closes the think block
  while tools remain enabled through `zaya_xml`.
- ZAYA disk restore passes with a fresh coordinator disk hit and
  `ssm_companion`. ZAYA paged-prefix cache correctly reports not applicable
  because CCA topology is paged-incompatible.
- Ling JANGTQ2 passes compile off/on multi-turn recall, paged-prefix HIT,
  disk restore with `ssm_companion`, and TurboQuant B=2 isolation.
- Gemma4 E2B BatchEngine B=2 and B=4 concurrent rows pass; B=4 slot 0 matched
  the B=1 solo reference exactly.

VL status:

- The hook-listed Qwen3.5 VL 35B/122B and Gemma4 VL bundles are not locally
  loadable on this machine. The Qwen VL HF cache entries on the mounted
  migration volume are only tiny metadata snapshots with no safetensor shards.
- Available local Nemotron-Omni VL substitutes pass structured chat/cache on
  JANGTQ, JANGTQ4, and MXFP4 rows. The full Omni matrix passes 17/17, covering
  text, image, video encoder, audio encoder, media salt isolation, reasoning
  toggle, hybrid SSM warm pass, and BatchEngine text/image/audio rows.

Missing or incomplete hook rows:

- `~/models/Mistral-Small-4-119B-JANG_2L` is present but incomplete: config,
  template, images, and cache metadata exist, but no safetensor shards or
  `model.safetensors.index.json` are present.
- `~/.mlxstudio/models/Qwen3.5-VL-35B-A3B-JANG_4K-CRACK`,
  `~/.mlxstudio/models/Qwen3.5-VL-122B-A10B-JANG_4K-CRACK`, and
  `~/.mlxstudio/models/MiniMax-M2.5-JANG_2L-CRACK` are missing.
- Gemma4 E2B/E4B VL and ZAYA BF16 source were not found under the audited local
  model roots. Converted ZAYA bundles were tested from `~/models/Zyphra`.

Do not expose the following as default Osaurus behavior from this evidence:

- ZAYA paged-prefix hits, compiled decode, reasoning-on/max defaults, or
  TurboQuant KV defaults.
- Global TurboQuant KV defaulting. TQ KV remains functionally stable in the
  tested rows but fails the within-5-percent speed target on Qwen35, Gemma4
  E2B, and Nemotron Cascade long-context comparisons.
- A global "all models meet Python speed" claim. Several rows are functional
  but still below speed or `AsType` targets.

### Stability re-audit - 2026-05-07 14:15 PDT

This pass focused on the Osaurus-facing stability surface rather than raw
token/s: reasoning on/off recognition, chat-template routing, prefix/paged/L2
cache behavior, BatchEngine shutdown/admission semantics, and ZAYA/Ling hybrid
cache contracts.

Two false-positive sources were found and corrected before declaring the
surface clean:

- `gpt_oss` is now a Harmony-reasoning family. Older reasoning audit tests still
  classified it as `"none"`; the tests and comments now match the runtime
  contract (`reasoningStampFromModelType("gpt_oss") == "harmony"`).
- The synthetic `TestTokenizer` returned a random ID for every unknown special
  token and used EOS/unknown IDs inside tiny test model vocabularies. That made
  BatchEngine tests nondeterministically stop on ordinary generated tokens. The
  fixture now maps only known tokens and keeps EOS/unknown outside generated
  vocab ranges.

Fresh verification from the current tree:

| Surface | Result |
|---|---|
| Release `RunBench` build | PASS. `/tmp/vmlx_release_runbench_build_stability_ready_20260507.log`. |
| Reasoning parser/stamp/prompt-tail matrix | PASS. 33 Swift Testing/XCTest rows, including GPT-OSS Harmony, Bailing/Ling `think_xml`, ZAYA/Zyphra tool format, prompt-tail open/closed think detection. `/tmp/vmlx_reasoning_parser_surface_after_fixture_20260507.log`. |
| BatchEngine + CacheCoordinator unit/integration surface | PASS. 35 rows covering resize/shutdown, TQ B=2 isolation, multi-turn plain/TQ cache hits, prompt extension, no cross-prompt contamination, media salt, and KV policy defaults. `/tmp/vmlx_cache_batch_surface_after_fixture_20260507.log`. |
| Cross-prompt cache contamination flake check | PASS 10/10 isolated repeats after fixing `TestTokenizer`. `/tmp/vmlx_repeat_no_cross_prompt_summary_20260507.log`. |
| ZAYA focused real-bundle suite | PASS. 33 rows covering real JANGTQ2/JANGTQ4/MXFP4 load+forward, CCA cache state/disk round-trip, B=2 CCA isolation, ZAYA XML parser, and BatchEngine B=2 visible chunks. `/tmp/vmlx_zaya_focused_after_stability_fixture_20260507.log`. |
| Ling/Bailing unit + real-bundle processor toggle | PASS. Bailing directive unit tests plus real Ling JANGTQ2 load through `LLMUserInputProcessor`; default/off render `detailed thinking off`, explicit `enable_thinking=true` renders `detailed thinking on`. `/tmp/vmlx_ling_processor_toggle_real_bundle_20260507.log`. |

Fresh live model rows:

| Model | Current status |
|---|---|
| ZAYA1-8B JANGTQ2 contract | PASS. `cacheSubtype=zaya_cca`, `cacheType=hybrid`, 40 CCA layers, 120 TQ groups, sidecar present, effective EOS `[106]`, template present. `/tmp/vmlx_live_zaya_jangtq2_contract_ready_20260507.log`. |
| ZAYA1-8B JANGTQ2 template | PASS. `enable_thinking=false` closes `<think></think>` and keeps tool rows; explicit thinking-on opens the tail for diagnostics. `/tmp/vmlx_live_zaya_jangtq2_template_ready_20260507.log`. |
| ZAYA1-8B JANGTQ2 BatchEngine chat | PASS runtime; compile off/on both emit visible chunks without raw thinking markers. Generic favorite-color recall remains weak model behavior, not a cache/runtime failure. `/tmp/vmlx_live_zaya_jangtq2_batch_chat_ready_20260507.log`. |
| ZAYA1-8B JANGTQ2 paged prefix | PASS as not-applicable. ZAYA CCA is paged-incompatible by design; Osaurus should not expect paged-prefix hits for ZAYA. `/tmp/vmlx_live_zaya_jangtq2_cache_hit_ready_20260507.log`. |
| ZAYA1-8B JANGTQ2 disk restore | PASS. Fresh coordinator hit L2 disk, `matched=142/142`, `diskArrays=yes`, `ssm_companion` present. `/tmp/vmlx_live_zaya_jangtq2_disk_restore_ready_20260507.log`. |
| Ling 2.6 Flash JANGTQ2 config/template | PASS. `modelType=bailing_hybrid`, sidecar present, effective EOS/BOS covered. Raw tokenizer smoke is stable; production thinking toggle is verified by the processor test above. `/tmp/vmlx_live_ling_jangtq2_config_ready_20260507.log`, `/tmp/vmlx_live_ling_jangtq2_template_ready_20260507.log`. |
| Ling 2.6 Flash JANGTQ2 BatchEngine chat | PASS. Compile off/on both recall the multi-turn blue/cool prompts with visible chunks. `/tmp/vmlx_live_ling_jangtq2_batch_chat_ready_20260507.log`. |
| Ling 2.6 Flash JANGTQ2 paged prefix | PASS. Paged tier hit `matched=128/166`; hybrid partial-hit rollback is expected and correct. `/tmp/vmlx_live_ling_jangtq2_cache_hit_ready_20260507.log`. |
| Ling 2.6 Flash JANGTQ2 disk restore | PASS. Fresh coordinator hit L2 disk, `matched=143/143`, `diskArrays=yes`, `ssm_companion` present. `/tmp/vmlx_live_ling_jangtq2_disk_restore_ready_20260507.log`. |
| Ling 2.6 Flash JANGTQ2 TQ B=2 | PASS. Plain slot beside TQ neighbor matched the B=1 reference exactly; both TQ slots completed coherent output. `/tmp/vmlx_live_ling_jangtq2_tq_b2_ready_20260507.log`. |

Osaurus guidance from this pass:

- Use `UserInput(chat:)` through the production processor. Do not render raw
  tokenizer templates in Osaurus and expect Bailing/Ling thinking directives to
  be injected.
- ZAYA is production-safe for default thinking-off chat, ZAYA XML tool parsing,
  B=2 isolation, and L2 disk restore. Paged-prefix hits are intentionally
  disabled for its CCA cache topology.
- Ling is production-safe for default thinking-off chat, explicit thinking
  toggle prompt rendering, paged-prefix hits, L2 disk restore, and B=2/TQ
  isolation. Speed is still optimization work.
- TurboQuant KV remains opt-in/diagnostic for speed. It is functionally stable
  in the rows above, but the broader hook sweep still showed TQ slower than
  float KV on Qwen/Gemma/Nemotron rows.
- The huge mixed `swift test --filter ...` runner can still make SwiftPM's
  testing helper unstable when XCTest and Swift Testing MLX-heavy suites are
  interleaved. Split the stability surface into the focused commands above.

Correction: some token/s rows in this subsection were captured while separate
MiniMax and/or Osaurus model jobs were active. Treat the speed values here as
provisional except where superseded by the clean rebench rows below. The
functional pass/fail, cache behavior, graph stats, stop reasons, and
missing-path audits remain useful because they exercised the current runtime,
but do not use provisional token/s numbers as production targets.

| Row | Current result |
|---|---|
| Qwen3.5-35B-A3B 4bit | PASS. 93.9 tok/s cold single run, 105.3 tok/s in the later float-KV comparison, TTFT 56-64 ms, `decodeNodes=3854`, `asType=90`, no loop/leak. |
| Gemma4 E2B 4bit | PASS. 155.2 tok/s cold hook row; 184.1 tok/s in the later float-KV comparison, `decodeNodes=1704`, `asType=89`, no loop/leak. |
| Gemma4 E4B 4bit | PASS. 102.5 tok/s, TTFT 50 ms, `decodeNodes=2210`, `asType=112`, no loop/leak. |
| Gemma4 26B A4B 4bit | PASS after background-job cleanup. 82.4 tok/s, TTFT 92 ms, `decodeNodes=3274`, `asType=151`, no loop/leak. |
| Nemotron Cascade JANG_2L | PASS but slower than the earlier clean hook row. 59.9 tok/s, `decodeNodes=2062`, `asType=161`, visible coherent text. |
| Nemotron Super JANG_2L | PASS functionally, speed/graph cleanup remains. 35.7 tok/s, `decodeNodes=3599`, `asType=280`. Generic coherent bench routes recall into `.reasoning` with empty visible chunks. |
| MiniMax M2.7 JANGTQ | PARTIAL. TokenIterator path works at 38.2 tok/s for 32 tokens, `decodeNodes=4045`, `asType=372`, coherent text. BatchEngine perf path hit a Metal GPU timeout and is a production blocker for this bundle/path. MiniMax is standard KV/MoE, not a hybrid SSM/CCA model. |
| Ling JANGTQ2 | PASS functionally. 31.3 tok/s, `decodeNodes=5`, `asType=3`, multi-turn recall passes compile off/on. Paged-prefix hit, disk L2 restore with `ssm_companion`, and TurboQuant KV B=2 isolation pass. Speed target remains open. |
| Ling legacy JANGTQ | PASS functionally. 29.4 tok/s, `decodeNodes=5`, `asType=3`, coherent short decode. |
| Ling MXFP4 | PASS functionally but not production-preferred. 9.9 tok/s, coherent short decode, high memory/low speed. |
| ZAYA JANGTQ2 | PASS contract/cache/decode, weak generic chat recall. Contract sees `cacheSubtype=zaya_cca`, 40 CCA layers, 120 TQ groups, sidecar present. 62.4 tok/s, `decodeNodes=8831`, `asType=1365`. Disk L2 restore passes; paged-prefix cache correctly reports not applicable. Generic favorite-color multi-turn ends cleanly but does not recall the color. |
| ZAYA JANGTQ4 | PASS short decode. 55.5 tok/s, `decodeNodes=8831`, `asType=1365`. |
| ZAYA MXFP4 | PASS short decode. 62.2 tok/s, `decodeNodes=8791`, `asType=1285`. |
| ZAYA BF16 source | NOT TESTED. No local `ZAYA1-8B` source bundle was found under the audited ZAYA roots in this pass. |
| Nemotron Omni Nano JANGTQ | PASS. Text row 73.7 tok/s, `decodeNodes=1947`, `asType=46`; structured VL cache matrix passed; full Omni matrix passed 17/17 including text, image, video encoder, audio encoder, reasoning toggle, media-salt isolation, hybrid SSM warm-pass, and BatchEngine text/image/audio rows. |
| DSV4 Flash JANGTQ | PASS production gate. Template kwargs pass. Chat recall, reasoning off/on/max, and 5,568-token long-context semantic recall all pass with `unclosedReasoning=false`. Short perf is 13.2 tok/s, `decodeNodes=30143`, `asType=1325`; disk L2 restore passes; paged-prefix cache correctly reports not applicable. |

Clean rebench rows from 2026-05-07 10:52 PDT, current `RunBench`, one vmlx
model process at a time. These supersede the matching token/s values above:

| Row | Clean result |
|---|---|
| Qwen3.5-35B-A3B 4bit | PASS. 91.5 tok/s, TTFT 73 ms, `decodeNodes=3854`, `asType=90`, coherent, no loop/leak. Log: `/tmp/vmlx_rebench_clean_qwen35_35b_20260507_105241.log`. |
| Gemma4 E2B 4bit | PASS. 149.8 tok/s, TTFT 36 ms, `decodeNodes=1704`, `asType=89`, coherent, no loop/leak. Log: `/tmp/vmlx_rebench_clean_gemma4_e2b_20260507_105241.log`. |
| Gemma4 E4B 4bit | PASS. 97.4 tok/s, TTFT 50 ms, `decodeNodes=2210`, `asType=112`, coherent, no loop/leak. Log: `/tmp/vmlx_rebench_clean_gemma4_e4b_20260507_105241.log`. |
| Gemma4 26B A4B 4bit | PASS. 77.9 tok/s, TTFT 90 ms, `decodeNodes=3274`, `asType=151`, coherent, no loop/leak. Log: `/tmp/vmlx_rebench_clean_gemma4_26b_20260507_105241.log`. |
| Nemotron Cascade JANG_2L | PASS. 99.2 tok/s, TTFT 100 ms, `decodeNodes=2062`, `asType=161`, coherent, no loop/leak. Log: `/tmp/vmlx_rebench_clean_nemotron_cascade_20260507_105241.log`. |
| Nemotron Super JANG_2L | PASS functionally. 28.4 tok/s, TTFT 221 ms, `decodeNodes=3599`, `asType=280`, coherent, no loop/leak; speed/AsType cleanup remains. Log: `/tmp/vmlx_rebench_clean_nemotron_super_20260507_105241.log`. |
| MiniMax M2.7 JANGTQ | PASS functionally on iterator path. 26.5 tok/s, TTFT 688 ms, `decodeNodes=4045`, `asType=372`, coherent, no loop/leak; BatchEngine timeout/speed cleanup remains. Log: `/tmp/vmlx_rebench_clean_minimax_m27_20260507_105241.log`. |
| Ling JANGTQ2 | PASS functionally. 22.2 tok/s, TTFT 258 ms, `decodeNodes=5`, `asType=3`, coherent, no loop/leak; speed target remains open. Log: `/tmp/vmlx_rebench_clean_ling_jangtq2_20260507_105241.log`. |
| ZAYA JANGTQ2 | PASS short decode. 49.4 tok/s, TTFT 74 ms, `decodeNodes=8831`, `asType=1365`, coherent, no loop/leak; chat-quality and AsType cleanup remain. Log: `/tmp/vmlx_rebench_clean_zaya_jangtq2_20260507_105241.log`. |
| Nemotron Omni Nano JANGTQ | PASS. 59.6 tok/s, TTFT 137 ms, `decodeNodes=1947`, `asType=46`, coherent, no loop/leak. Log: `/tmp/vmlx_rebench_clean_omni_jangtq_20260507_105241.log`. |
| DSV4 Flash JANGTQ | Not rerun in the 10:52 clean speed set. Use the production gate row above until a later clean DSV4 rebench is recorded. |

Current missing/incomplete hook rows:

- `~/models/Mistral-Small-4-119B-JANG_2L` is an incomplete local download:
  config/template/images plus incomplete cache blobs, no weights/index.
- `~/.mlxstudio/models/Qwen3.5-VL-35B-A3B-JANG_4K-CRACK` is not present.
- `~/.mlxstudio/models/Qwen3.5-VL-122B-A10B-JANG_4K-CRACK` is not present.
- `~/.mlxstudio/models/MiniMax-M2.5-JANG_2L-CRACK` is not present; local
  MiniMax M2.7 was tested instead.
- Gemma4 E2B/E4B VL bundles were not found under the audited local model roots.
- The hook's old `~/jang/models/Zyphra/ZAYA1-8B*` paths are stale/missing on
  this host. The valid converted ZAYA bundles are under `~/models/Zyphra`.

TurboQuant KV status from the current pass:

| Model | Float KV | TQ33 | TQ44 | Status |
|---|---:|---:|---:|---|
| Qwen3.5-35B-A3B 4bit | 105.3 tok/s | 45.7 tok/s | 45.9 tok/s | Functionally coherent, not speed-valid. |
| Gemma4 E2B 4bit | 184.1 tok/s | 144.0 tok/s | 141.9 tok/s | Functionally coherent, slower than the 5% target. |
| Nemotron Omni Nano JANGTQ | 73.7 tok/s | 70.1 tok/s | 69.6 tok/s | TQ33 is within about 5%; TQ44 is slightly slower than target. |
| Gemma4 E2B long prompt | 5,220 prompt tokens; float, TQ33, and TQ44 all completed coherently with no loop/leak. |

Batching status from the current pass:

- Gemma4 E2B `BENCH_BATCH=1`: B=1, compile, TurboQuant, and B=2 smoke passed.
- Gemma4 E2B `BENCH_BATCH_CONCURRENT=1`: B=2 concurrent passed with isolated
  prompt outputs.
- Gemma4 E2B `BENCH_BATCH_B4=1`: B=4 concurrent stress passed; slot 0 matched
  the B=1 solo reference exactly.

Open production blockers found in this pass:

- MiniMax M2.7 BatchEngine perf can GPU-timeout on Metal. The iterator path is
  coherent, so this is a BatchEngine/model-profile/runtime issue, not missing
  weights.
- Qwen/Nemotron/MiniMax generic coherent bench can route useful recall into
  `.reasoning` with empty visible `.chunk` when thinking is enabled by the
  template/profile. Osaurus must use explicit reasoning-off profiles for normal
  chat and handle reasoning-only/length finishes deliberately.
- ZAYA JANGTQ2 is cache/runtime-ready with thinking off, but the generic
  multi-turn chat-quality row is weak. Do not expose ZAYA reasoning-on/max,
  paged-prefix hits, compiled decode, or TurboQuant KV defaults yet.
- TurboQuant KV is not uniformly speed-valid. It works functionally on tested
  normal KV models, but Qwen and Gemma are materially slower than float KV.

Background-active rebench note: at user direction, additional rows were run
while leaving a concurrent MiniMax evaluation and Osaurus runtime alone. These
are useful only to prove functional output under contention; they are not clean
token/s baselines.

| Row | Background-active result |
|---|---|
| Qwen3.5-35B-A3B 4bit | Coherent, no loop/leak, `asType=90`, but only 12.5 tok/s with TTFT 4.7 s. |
| Gemma4 E2B 4bit | Coherent, no loop/leak, `asType=89`, 133.7 tok/s, below the clean target because the host was busy. |
| Gemma4 E4B 4bit | Coherent, no loop/leak, `asType=112`, 42.0 tok/s under contention. |
| Gemma4 26B A4B 4bit | Coherent, no loop/leak, `asType=151`, 31.3 tok/s under contention. |
| ZAYA JANGTQ2 | Coherent short decode, no loop/leak, `asType=1365`, 21.8 tok/s under contention. |
| Ling JANGTQ2 | Coherent short decode, no loop/leak, `asType=3`, 12.4 tok/s under contention. |
| Nemotron Omni Nano JANGTQ | Coherent short decode, no loop/leak, `asType=46`, 23.5 tok/s under contention. |

Do not compare these background-active rows against Python or product speed
targets. Use them only as proof that current runtime paths still complete
without loops while another large model job is present.

## Focus Status - MiniMax, Ling, ZAYA

These are the current rows to use when deciding what belongs in the first
osaurus-side integration PR.

| Model | Speed from live RunBench | Cache/runtime status | Osaurus PR status |
|---|---:|---|---|
| MiniMax M2.7 JANGTQ | BatchEngine 31.3 tok/s median, TokenIterator 34.9 tok/s median, target 45-50 | Coherent visible text, no loop/leak. Paged prefix hit PASS, disk L2 restore PASS, TurboQuant KV B=2 isolation PASS. | Functionally ready for runtime integration. Keep speed below-target as a follow-up optimization/model-profile audit. |
| Ling 2.6 flash JANGTQ2 | BatchEngine 29.8 tok/s median in the latest gate, TokenIterator about 29.9 tok/s, target 80-90 | Multi-turn recall PASS, compile off/on. Paged prefix and disk L2 hit with SSM companion PASS. TurboQuant KV B=2 isolation PASS after the BailingHybrid per-slot offset and asymmetric K/V TurboQuant fix. | Functionally ready for runtime integration with speed below target. Keep Ling cache ownership inside vmlx; do not add app-layer prefill/cache guards. |
| ZAYA1 8B BF16/JANGTQ2/JANGTQ4/MXFP4 | Current short decode: BF16 57.9 tok/s, JANGTQ2 49.9 tok/s, JANGTQ4 54.7 tok/s, MXFP4 66.8 tok/s | Real bundles load through `LLMModelFactory`; forward smoke passes for all four variants. `ZayaCCACache`, disk round-trip, live exact disk restore, B=2 CCA slot isolation, live B=2 raw decode, production chat-stream B=2, BatchEngine hybrid admission, and `BatchZayaCCACache` are tested. ZAYA remains paged-prefix/compile-incompatible for now because CCA state is path-dependent. | Runtime/cache integration is ready with thinking off. Prefer JANGTQ4 or MXFP4 for chat-quality rows today; JANGTQ2 is runtime-ready but weak on the generic favorite-color multi-turn recall after the default-thinking fix, so treat it as a model-artifact follow-up before broad chat serving. Keep `zaya_xml` tools enabled. Do not expose paged-prefix hits, compiled decode, reasoning-on/max defaults, or TurboQuant KV defaults for ZAYA until CCA/reasoning parity is proven in those tiers. |

Fresh source logs for this focused pass:

- MiniMax perf: `/tmp/vmlx_focus_minimax_m27_batch_20260506.log`,
  `/tmp/vmlx_focus_minimax_m27_iter_20260506.log`.
- MiniMax cache: `/tmp/vmlx_focus_minimax_m27_cache_hit_20260506.log`,
  `/tmp/vmlx_focus_minimax_m27_disk_restore_20260506.log`,
  `/tmp/vmlx_focus_minimax_m27_tq_b2_20260506.log`.
- Ling perf/coherence: `/tmp/vmlx_focus_ling_jangtq2_batch_20260506.log`,
  `/tmp/vmlx_focus_ling_jangtq2_iter_20260506.log`,
  `/tmp/vmlx_focus_ling_jangtq2_coherent_20260506.log`.
- Ling cache: `/tmp/vmlx_focus_ling_jangtq2_cache_hit_20260506.log`,
  `/tmp/vmlx_focus_ling_jangtq2_disk_restore_20260506.log`,
  `/tmp/vmlx_focus_ling_jangtq2_tq_b2_20260506.log`,
  `/tmp/vmlx_focus_ling_jangtq2_tq_b2_after_tq_asym_20260506.log`.
- ZAYA contract/template: `/tmp/vmlx_focus_zaya_JANGTQ2_contract_20260506.log`,
  `/tmp/vmlx_focus_zaya_JANGTQ4_contract_20260506.log`,
  `/tmp/vmlx_focus_zaya_MXFP4_contract_20260506.log`, and matching
  `*_template_20260506.log` files.
- ZAYA post-port contract: `/tmp/vmlx_zaya_jangtq2_contract_after_port_20260506.log`,
  `/tmp/vmlx_zaya_jangtq4_contract_after_port_20260506.log`, and
  `/tmp/vmlx_zaya_mxfp4_contract_after_port_20260506.log`.
- ZAYA post-port live decode: `/tmp/vmlx_zaya_bf16_short_decode_20260506.log`,
  `/tmp/vmlx_zaya_jangtq2_short_decode_20260506.log`,
  `/tmp/vmlx_zaya_jangtq4_short_decode_20260506.log`, and
  `/tmp/vmlx_zaya_mxfp4_short_decode_20260506.log`.
- ZAYA B=2-fix decode smoke:
  `/tmp/vmlx_zaya_jangtq2_short_decode_after_b2fix_20260506.log`.
- ZAYA post-port cache/reasoning:
  `/tmp/vmlx_zaya_jangtq2_b2_concurrent_20260506.log`,
  `/tmp/vmlx_zaya_jangtq2_disk_restore_20260506.log`,
  `/tmp/vmlx_zaya_jangtq2_cache_hit_20260506.log`,
  `/tmp/vmlx_zaya_jangtq2_cache_hit_after_harness_fix_20260506.log`,
  `/tmp/vmlx_zaya_jangtq2_think_off_probe_20260506.log`, and
  `/tmp/vmlx_zaya_jangtq2_think_on_probe_20260506.log`.
- ZAYA 2026-05-07 metadata/runtime refresh after marking converted bundles
  `supports_thinking=false`:
  `/tmp/vmlx_zaya_jangtq2_contract_metadata_patch_20260507.log`,
  `/tmp/vmlx_zaya_jangtq2_template_metadata_patch_20260507.log`,
  `/tmp/vmlx_zaya_jangtq2_perf_metadata_patch_20260507.log`,
  `/tmp/vmlx_zaya_jangtq2_disk_restore_metadata_patch_20260507.log`, and
  `/tmp/vmlx_zaya_jangtq2_cache_hit_metadata_patch_20260507.log`.
- ZAYA/Ling stop-hook gate refresh after adding both families to the local
  completion matrix:
  `/tmp/vmlx_gate_zaya_contract_20260507.log`,
  `/tmp/vmlx_gate_zaya_template_20260507.log`,
  `/tmp/vmlx_gate_zaya_perf_20260507.log`,
  `/tmp/vmlx_gate_zaya_disk_restore_20260507.log`,
  `/tmp/vmlx_gate_zaya_cache_hit_20260507.log`,
  `/tmp/vmlx_gate_ling_perf_20260507.log`,
  `/tmp/vmlx_gate_ling_coherent_20260507.log`,
  `/tmp/vmlx_gate_ling_cache_hit_20260507.log`,
  `/tmp/vmlx_gate_ling_disk_restore_20260507.log`, and
  `/tmp/vmlx_gate_ling_tq_b2_20260507.log`.

## ZAYA Production Notes - 2026-05-07

ZAYA is a hybrid CCA topology: even decoder layers carry
`ZayaCCACache` (KV + convolution state + previous hidden state) while odd MoE
layers use `KVCacheSimple` stubs. Treat it as a path-dependent hybrid model,
not as a normal full-attention KV-only model.

Verified gates from the latest local pass:

| Area | Status |
|---|---|
| Bundle contract | PASS for JANGTQ2 current refresh: `weightFormat=mxtq`, `cacheSubtype=zaya_cca`, `cacheType=hybrid`, 40 CCA layers, 40 MoE layers, 120 TQ packed groups, sidecar present, EOS 106. Prior JANGTQ4/MXFP4 contract rows also PASS. |
| Real model load/forward | PASS in focused Swift tests for BF16, JANGTQ2, JANGTQ4, and MXFP4. |
| Chat template | PASS. `enable_thinking=false` renders a closed empty think block; `enable_thinking=true` still opens `<think>`. |
| Thinking off | PASS for production text row. Current JANGTQ2 live row generated `Paris.`, `stop=stop`, no loop, no marker leak, `unclosedReasoning=NO`. |
| Thinking on/max | NOT PRODUCTION-READY. Live JANGTQ2 can stop inside reasoning with no visible answer even with a larger token budget. Keep Osaurus catalog/profile `supportsThinking=false` for ZAYA until a real thinking-on row closes reasoning and emits visible content. |
| Tool calls | Parser support is ready via `ToolCallFormat.zayaXml`; keep `supports_tools=true` / `tool_parser=zaya_xml`. |
| BatchEngine B=2 | PASS in production-shaped chat-stream test with isolated visible chunks and no thinking/tool marker leaks. |
| L2 disk cache | PASS. Current JANGTQ2 disk restore hit exact disk arrays from a fresh coordinator, matched 140/140 prompt tokens, and completed both sessions. |
| Paged prefix cache | INTENTIONALLY DISABLED. The coordinator marks ZAYA paged-incompatible; prefix-extension hits are not applicable because CCA conv/prev-hs state is path-dependent. |
| TurboQuant KV | Do not enable as a default for ZAYA. The model weights can be JANGTQ, but runtime KV quantization for the hybrid CCA stack is separate and must remain opt-in until exact B=2 + disk + restore parity is proven. |

Local converted bundle metadata was patched in both `config.json` and
`jang_config.json` for the JANGTQ2, JANGTQ4, MXFP4, and OsaurusAI mirror
copies: `supports_thinking=false`, while `reasoning_parser=qwen3`,
`think_in_template=true`, and `tool_parser=zaya_xml` are preserved. The
reasoning parser remains useful for forced/debug rows; product wiring should
not expose a user-facing thinking toggle for ZAYA yet.

The generic chat processor now also seeds `enable_thinking=false` for ZAYA /
Zyphra topology and for bundles whose capabilities declare
`supports_thinking=false`. This is a library-level template default, not an
osaurus app-layer prompt workaround; explicit request context still overrides
it for diagnostics. After this fix, JANGTQ4 and MXFP4 passed the generic
favorite-color multi-turn recall probe. JANGTQ2 and BF16 no longer leak thinking
markers, but their recall answer remained weak, so prefer JANGTQ4/MXFP4 for
production chat-quality validation while the JANGTQ2 artifact is reviewed.

## Current Local Pass - 2026-05-06

This continuation prioritized complete bundles already present under
`~/models` and `~/models/JANGQ`. Mistral Small was intentionally deferred; the
partial `~/models/Mistral-Small-4-119B-JANG_2L` download is not part of this
matrix.

Hook compatibility rerun after commit `a138f47`, all available rows tested
sequentially with `BENCH_PERF=1 BENCH_GRAPH_STATS=1 BENCH_PERF_PATH=batch`,
one model process at a time:

| Hook row | Status |
|---|---|
| `~/models/Qwen3.5-35B-A3B-4bit` | PASS speed/coherence. 100.2 tok/s median, best 100.7, `decodeNodes=3854`, `asType=300`, max RSS about 20.6 GB. No loop, no marker leak, visible coherent text. |
| `~/osaurus_models/finished/gemma-4-e2b-it-4bit` | PASS speed/coherence. 168.5 tok/s median, best 168.7, `decodeNodes=1704`, `asType=300`, max RSS about 3.3 GB. No loop, no marker leak. |
| `~/osaurus_models/finished/gemma-4-e4b-it-4bit` | PASS coherence. 108.0 tok/s median, best 109.7, `decodeNodes=2210`, `asType=365`, max RSS about 5.0 GB. No loop, no marker leak. |
| `~/osaurus_models/finished/gemma-4-26b-a4b-it-4bit` | PASS speed/coherence. 87.0 tok/s median, best 87.6, `decodeNodes=3334`, `asType=362`, max RSS about 15.9 GB. No loop, no marker leak. |
| `~/.mlxstudio/models/Nemotron-Cascade-2-30B-A3B-JANG_2L` | PASS speed/coherence. 118.6 tok/s median, best 119.8, `decodeNodes=2062`, `asType=161`, max RSS about 9.3 GB. No loop, no marker leak. |
| `~/.mlxstudio/models/Nemotron-3-Super-120B-A12B-JANG_2L` | PASS coherence, speed lower than smaller Cascade. 41.9 tok/s median/best, `decodeNodes=3599`, `asType=280`, max RSS about 41.8 GB. No loop, no marker leak. |
| `~/models/Mistral-Small-4-119B-JANG_2L` | NOT TESTED. Local folder is incomplete: config/template/readme/images are present, but no safetensor shards or index. This row also remains deprioritized per the latest user direction. |
| `~/.mlxstudio/models/Qwen3.5-VL-35B-A3B-JANG_4K-CRACK` | NOT TESTED. Path is not present on this host. |
| `~/.mlxstudio/models/Qwen3.5-VL-122B-A10B-JANG_4K-CRACK` | NOT TESTED. Path is not present on this host. |
| `~/.mlxstudio/models/MiniMax-M2.5-JANG_2L-CRACK` | NOT TESTED. Path is not present on this host. |
| Gemma4 E2B/E4B VL | NOT TESTED. No matching local bundle found under `~/models`, `~/osaurus_models`, or `~/.mlxstudio/models`. |

Interpretation: throughput for Qwen3.5 and Gemma4 meets or exceeds the hook
targets, and all available rows emitted coherent text without loop/leak. The
initial graph-stats rows used a substring `AsType` counter; the exact counter
rerun below supersedes those counts.

Follow-up exact graph-counter rerun: `CmlxGraphShim` was checked with DOT dumps
and the previous substring counter was found to overcount fused node labels such
as `CompiledAsType...`. Exact `label="AsType"` counts from the current rerun
are:

| Hook row | Median tok/s | Exact `AsType` | Output status | Log |
|---|---:|---:|---|---|
| Qwen3.5-35B-A3B 4bit | 101.5 | 90 | Coherent text preview, no loop/leak. Meets speed and `<100` graph gate. | `/tmp/vmlx_hook_now_01_qwen35_35b_4bit_20260506.log` |
| Gemma4 E2B 4bit | 172.3 | 89 | Coherent text preview, no loop/leak. Meets speed and `<100` graph gate. | `/tmp/vmlx_hook_now_02_gemma4_e2b_20260506.log` |
| Gemma4 E4B 4bit | 109.2 | 112 | Coherent text preview, no loop/leak. Minor graph cleanup remains. | `/tmp/vmlx_hook_now_03_gemma4_e4b_20260506.log` |
| Gemma4 26B A4B 4bit | 88.5 | 151 | Coherent text preview, no loop/leak. Above speed target; graph cleanup remains. | `/tmp/vmlx_hook_now_04_gemma4_26b_20260506.log` |
| Nemotron Cascade JANG_2L | 118.5 | 161 | Coherent text preview, no loop/leak. Fast; graph cleanup remains. | `/tmp/vmlx_hook_now_05_nemotron_cascade_20260506.log` |
| Nemotron Super JANG_2L | 42.1 | 280 | Coherent text preview, no loop/leak. Speed/graph cleanup remain. | `/tmp/vmlx_hook_now_06_nemotron_super_20260506.log` |

Hook compatibility rerun on 2026-05-07 after the current ZAYA/Ling cache
work, again one model process at a time with
`BENCH_PERF=1 BENCH_GRAPH_STATS=1 BENCH_PERF_PATH=batch`. MLXShot's GPU helper
was present during this run, so token/s should be treated as current local
evidence, not a clean-room maximum.

| Hook row | Status | Log |
|---|---|---|
| `~/models/Qwen3.5-35B-A3B-4bit` | PASS speed/coherence. 96.0 tok/s median, best 97.6, exact `AsType=90`, no loop/leak, visible coherent text. Meets the >90 tok/s and <100 AsType hook gate. | `/tmp/vmlx_hook_seq_qwen35_35b_20260507.log` |
| `~/osaurus_models/finished/gemma-4-e2b-it-4bit` | PASS speed/coherence. 163.5 tok/s median, best 165.8, exact `AsType=89`, no loop/leak, visible coherent text. Meets the >150 tok/s and <100 AsType hook gate. | `/tmp/vmlx_hook_seq_gemma4_e2b_20260507.log` |
| `~/osaurus_models/finished/gemma-4-e4b-it-4bit` | PASS coherence. 103.9 tok/s median, best 106.3, exact `AsType=112`, no loop/leak. Graph cleanup remains above the <100 target. | `/tmp/vmlx_hook_seq_gemma4_e4b_20260507.log` |
| `~/osaurus_models/finished/gemma-4-26b-a4b-it-4bit` | PASS speed/coherence. 84.4 tok/s median, best 84.6, exact `AsType=151`, no loop/leak. Meets the >80 tok/s speed target; graph cleanup remains. | `/tmp/vmlx_hook_seq_gemma4_26b_20260507.log` |
| `~/.mlxstudio/models/Nemotron-Cascade-2-30B-A3B-JANG_2L` | PASS speed/coherence. 112.7 tok/s median, best 115.8, exact `AsType=161`, no loop/leak. Graph cleanup remains. | `/tmp/vmlx_hook_seq_nemotron_cascade_20260507.log` |
| `~/.mlxstudio/models/Nemotron-3-Super-120B-A12B-JANG_2L` | PASS coherence, slower than Cascade. 40.9 tok/s median, best 41.0, exact `AsType=280`, no loop/leak. Speed/graph cleanup remain. | `/tmp/vmlx_hook_seq_nemotron_super_20260507.log` |
| `~/models/Mistral-Small-4-119B-JANG_2L` | NOT TESTED. Local folder is still incomplete: config/template/readme/images only, no safetensor shards and no index. Rechecked exact path and mounted volumes on 2026-05-07. | local filesystem audit |
| `~/.mlxstudio/models/Qwen3.5-VL-35B-A3B-JANG_4K-CRACK` | NOT TESTED. Path is not present on this host. A separate text/VL-adjacent Qwen3.6 JANGTQ bundle exists under `~/models/dealign.ai`, but it is not this hook path. Rechecked exact path and mounted volumes on 2026-05-07. | local filesystem audit |
| `~/.mlxstudio/models/Qwen3.5-VL-122B-A10B-JANG_4K-CRACK` | NOT TESTED. Path is not present on this host. Rechecked exact path and mounted volumes on 2026-05-07. | local filesystem audit |
| `~/.mlxstudio/models/MiniMax-M2.5-JANG_2L-CRACK` | NOT TESTED. Exact hook path is not present on this host. Rechecked exact path and mounted volumes on 2026-05-07. Local MiniMax M2.7 JANGTQ was tested separately below. | local filesystem audit |
| Gemma4 E2B/E4B VL | NOT TESTED. No matching local Gemma VL bundle found under `~/models`, `~/osaurus_models`, `~/.mlxstudio/models`, or mounted-volume checks. | local filesystem audit |

Additional local-model row requested outside the exact hook list:

| Model | Status | Log |
|---|---|---|
| `~/models/JANGQ/MiniMax-M2.7-JANGTQ` | PASS coherence, speed below target. 30.5 tok/s median, best 30.6, exact `AsType=372`, no loop/leak, visible coherent text. Keep MiniMax M2.7 speed and AsType cleanup as active optimization work. | `/tmp/vmlx_extra_minimax_m27_jangtq_20260507.log` |

Runtime fix made during this pass:

- Low-level `generateTask(...)` now reconstructs the decoded prompt tail from
  `TokenIterator.promptTokenIds` when the caller does not pass `promptTail`.
  This keeps `ReasoningParser.forPrompt(...)` using the actual rendered prompt
  state instead of falling back to a family stamp. Live impact: Ling/Bailing
  ChatSession multi-turn output now streams visible answers through `.chunk`
  when the prompt tail has no `<think>` opener, instead of routing the whole
  answer to `.reasoning`.

Current live local generation results:

| Model | Result |
|---|---|
| Qwen3.6 35B A3B JANGTQ | PASS coherent. 75.8 tok/s median, 61-63 ms TTFT, peak footprint about 11.2 GB, `decodeNodes=2525`, `asType=260`. Reasoning channel check passed with 95 reasoning deltas and no `<think>` leakage. |
| Qwen3.6 27B JANG_4M | PASS coherent. 26.1 tok/s median, peak footprint about 18.1 GB, `decodeNodes=4424`, `asType=480`. |
| Qwen3.6 27B MXFP4 | PASS coherent. 30.1 tok/s median, peak footprint about 15.8 GB, `decodeNodes=4422`, `asType=480`. Tokenizer shim path is required. |
| Gemma4 26B A4B JANG_4M | PASS coherent. 79.3 tok/s median against the 80 tok/s target, peak footprint about 25.2 GB, `decodeNodes=3334`, `asType=362`. Harmony marker check passed; selected prompt did not elicit reasoning deltas. |
| Laguna XS.2 JANGTQ | PASS coherent, no loop/leak. 27.9 tok/s median, peak footprint about 12.1 GB, `decodeNodes=3677`, `asType=275`. Speed remains below target. |
| Ling 2.6 Flash JANGTQ | PASS coherent. 30.1 tok/s median, peak footprint about 30.3 GB, `decodeNodes=5`, `asType=3` on the recurrentGLA graph probe. After the prompt-tail fix, 3-turn ChatSession recall emits visible `.chunk` text for compile off/on. |
| Ling 2.6 Flash MXFP4 | PASS coherent but not production-preferred. 9.5 tok/s median, peak footprint about 66.7 GB. JANGTQ2/JANGTQ is the better Ling production choice today. |
| MiniMax M2.7 Small JANGTQ | PASS coherent. 31.4 tok/s median, peak footprint about 70.8 GB, `decodeNodes=4046`, `asType=373`. Bundle warning: tokenizer BOS `200034` differs from configured BOS `[200019]`. |
| MiniMax M2.7 JANGTQ | PASS coherent. 31.5 tok/s median, peak footprint about 61.3 GB, `decodeNodes=4045`, `asType=372`. Same BOS warning. |
| MiniMax M2.7 JANGTQ_K | PASS coherent. 30.9 tok/s median, peak footprint about 80.0 GB, `decodeNodes=4045`, `asType=372`. JANGTQ_K routing detected as `gateUp:2,down:4`; same BOS warning. |
| Nemotron Omni Nano JANGTQ2 | PASS coherent. 67.3 tok/s median, peak footprint about 15.1 GB, `decodeNodes=1947`, `asType=46`. |
| Nemotron Omni Nano JANGTQ4 | PASS coherent. 65.6 tok/s median, peak footprint about 29.8 GB, `decodeNodes=1947`, `asType=46`. |
| Nemotron Omni Nano MXFP4 | PASS coherent and fastest local Omni path. 129.1 tok/s median, peak footprint about 23.0 GB, `decodeNodes=2062`, `asType=161`. |
| DSV4 Flash JANGTQ | PASS coherent. Short perf: 12.1 tok/s median, peak footprint about 86.4 GB, `decodeNodes=30143`, `asType=1325`. Long-context row below is the acceptance gate; speed remains below 20 tok/s target. |

DSV4 production rows on the current local bundle:

- Config smoke: PASS. `modelType=deepseek_v4`, dispatch `deepseek_v4`,
  `weightFormat=mxtq`, `sidecar=true`, effective EOS includes `128804`.
- Template kwargs: PASS. `enable_thinking=false` closes `</think>`,
  `enable_thinking=true` opens `<think>`, and `reasoning_effort=max` is gated
  behind thinking.
- Chat multi-turn: PASS. Recalled `sapphire-42`; follow-up answered `Yes.`;
  no reasoning leak, `unclosedReasoning=false`.
- Reasoning modes: PASS. Thinking off/on/max all answered `12`; thinking rows
  split internal text to `.reasoning` and final text to `.chunk`.
- Long context: PASS. 5,568-token semantic recall returned
  `CERULEAN RIVER / OSLO`, `stop=stop`, no loop, no reasoning leak. Peak
  footprint was about 117.0 GB, so this row fits on the local M5 Max but is
  close to the memory ceiling.

Cache and batching checks from this pass:

| Row | Result |
|---|---|
| Qwen3.6 35B JANGTQ paged prefix | PASS. Coordinator probe hit paged tier, matched 128/161 tokens. Because the model has hybrid/linear-attention state, the warm prefill ratio is informational and rollback semantics are correct. |
| Qwen3.6 35B JANGTQ disk L2 | PASS. Fresh coordinator hit disk, `diskArrays=yes`, and `ssm_companion` was written. |
| Qwen3.6 35B JANGTQ TurboQuant KV B=2 | PASS. Plain-KV slot stayed byte-identical beside a TurboQuant(4,4) slot; two TQ slots also completed. |
| Qwen3.6 35B JANGTQ B=4 | FUNCTIONAL / PERF GAP. Four slots completed and slot 0 stayed byte-identical to solo, but B=4 wall time was about 96% of serial projection at both 24-token and 64-token budgets. Continuous batching isolation is correct; throughput gain for this hybrid/JANGTQ path is not yet there. |
| Gemma4 26B B=4 | PASS. Slot 0 matched solo output; wall ratio about 0.51 with uneven stop lengths, so speed assertion was skipped by the harness but correctness passed. |
| Nemotron Omni MXFP4 paged prefix | PASS. Coordinator probe hit paged tier, matched 128/173 tokens; `ssm_companion` handling remained enabled. |
| Nemotron Omni MXFP4 disk L2 | PASS. Fresh coordinator hit disk and restored with `diskArrays=yes`; `ssm_companion` was written. |
| DSV4 Flash disk L2 | PASS. Fresh coordinator hit disk using the DSV4 hybrid serializer path; prompt time dropped from 6.710 s cold to 0.097 s warm. This is the DSV4 cache path Osaurus should use. Do not globally replace DSV4 CSA/HSA/SWA with TurboQuant KV. |

Current known gaps:

- Speed: MiniMax M2.7, Laguna, Ling JANGTQ, Nemotron JANGTQ, and DSV4 remain
  below target. Qwen3.6 35B and Gemma4 26B are near target but still show high
  graph `AsType` counts. Nemotron MXFP4 is already above target.
- Ling TurboQuant KV: the BailingHybrid plain+TQ B=2 shape crash is fixed by
  using per-slot offsets for batched recurrent RoPE and separate TurboQuant
  encoder states for asymmetric MLA K/V dimensions. Keep the focused unit and
  live B=2 rows in the regression gate.
- Batching: Qwen3.6 hybrid/JANGTQ B=4 is functionally isolated but not faster
  than serial projection. Treat this as a BatchEngine throughput optimization,
  not a correctness blocker.
- DSV4 memory: the 5,568-token long row passes, but peak footprint is close to
  host ceiling. Larger stress rows still need memory hardening.
- Tests: `swift build -c release --product RunBench` passes. Focused
  prompt-tail tests also pass with the Xcode toolchain:
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test -c release --filter PromptTailDecodeTests --no-parallel`.
  The Command Line Tools Swift environment on this host still fails before
  test execution when importing `XCTest` from existing tests, so use the Xcode
  toolchain for release test runs.

## Osaurus PR Agent Notes

This is the handoff document to attach or cite from the first osaurus PR that
wires this library. The PR should be a runtime-coherence integration PR, not a
speed PR and not a JangPress PR.

Ownership guidance:

- Fix defects in the lowest owning layer. If the issue belongs in
  osaurus-owned MLX, mlx-swift, Swift Jinja, model metadata, or JANG conversion
  files, fix that layer instead of hiding it in `vmlx-swift-lm`.
- Do not add app-layer guards, prompt monkeypatches, or duplicate cache paths to
  paper over runtime bugs. Prefer explicit model/cache functions with tests and
  live rows.
- Keep per-agent notes current while changing runtime behavior. `.claude/` and
  `docs/agent-logs/` are ignored locally; durable integration facts belong in
  this tracked handoff.

PR scope:

- Use `vmlx-swift-lm` as the library/runtime source of truth.
- Wire `BatchEngine.generate(input:parameters:)` for production chat.
- Build production requests with `UserInput(chat:)` and explicit
  `additionalContext`.
- Consume `.chunk`, `.reasoning`, `.toolCall`, and `.info` as separate stream
  events.
- Surface reasoning-only length finishes cleanly in osaurus UI/API telemetry.
- Keep cache ownership in vmlx via `CacheCoordinator` and model-specific cache
  topology.

Do not include in the first PR:

- JangPress default-on behavior.
- Distributed inference.
- Speed-target enforcement.
- App-layer guards for DSV4/Ling cache behavior.
- Osaurus-side parsing of raw `<think>`, harmony, DSML, Qwen XML tool, or
  MiniMax tool markers from visible text.

Production-ready minimum as of this handoff:

- Ling JANGTQ/JANGTQ2/MXFP4 load and answer coherently in multi-turn recall.
  JANGTQ2 is the preferred Ling production bundle for memory. MXFP4 is now
  memory-bounded but slow.
- DSV4 Flash loads, routes chat/reasoning/max modes, and stores/restores
  hybrid cache state. Short multi-turn chat and the 5,568-token long-context
  row are revalidated cleanly after the local sidecar rebuild and RunBench
  teardown fix.
- Qwen 35B, Qwen 27B, Gemma4 26B, Laguna XS.2, MiniMax M2.7, and Nemotron Omni
  have live coherent-output coverage without marker leakage in the tested rows.
- Paged/prefix cache, L2 disk restore, SSM companion state, and TurboQuant KV
  disk round-trip have targeted coverage.

Hygiene notes for PR agents:

- `RunBench/` and `docs/` are gitignored in this checkout. Local benchmark
  harness edits and raw logs are audit artifacts unless explicitly force-added.
- Keep PR code to library/runtime files plus this handoff. Do not commit stale
  local logs by accident.
- There is no new duplicate runtime path for the first PR. Use the existing
  `BatchEngine`, `CacheCoordinator`, `ReasoningParser`, `ToolCallProcessor`,
  model loaders, and tokenizer processor APIs.
- Treat old JangPress docs that say "production ready" as historical for this
  phase. JangPress is deferred until runtime coherence is wired.
- Treat old speed docs as constraints for future optimization, not acceptance
  blockers for this PR.

## 2026-05-06 Local Revalidation

This pass revalidated the integration surface after pinning Swift-Jinja through
the osaurus-owned dependency chain and fixing RunBench's tokenizer smoke path.

| Area | Result |
|---|---|
| Package dependency graph | PASS. `Package.swift` resolves `osaurus-ai/Jinja` tag `2.3.5` and `osaurus-ai/swift-transformers` revision `b4a094b34b997167549c7f45bde16c80f18ed5a8`; no active `huggingface/swift-jinja` edge remains. |
| Release build | PASS. `swift build -c release --product RunBench`. |
| Focused Swift tests | PASS with Xcode toolchain and `--no-parallel`: paged/prefix cache, cache coordinator, SSM state cache, TQ disk serializer CacheList, interleaved reasoning/tool calls, Gemma4 template probes, and CompilableTurboQuantKVCache parity. |
| Kimi K2.6 Small JANGTQ config/template | PASS after RunBench switched to the production tokenizer substitution path. Remaining warning: tokenizer EOS `163585` is not in effective EOS `[163586]`. |
| DSV4 template kwargs | PASS. `enable_thinking=false` renders the DSV4 prompt tail as `<｜Assistant｜></think>`, `enable_thinking=true` opens `<think>`, and `reasoning_effort=max` is honored only while thinking is enabled. |
| DSV4 production chat row | PASS. Full-weight `BENCH_DSV4_COHERENCE BENCH_DSV4_ROW=chat` recalled `sapphire-42`, answered the sapphire/blue follow-up, emitted visible `.chunk` text with `unclosedReasoning=false`, and clean-exited after explicit engine shutdown. |
| Laguna XS.2 JANGTQ live loop probe | PASS. The bridge engaged the native Poolside/Laguna template, and the strict 512-token loop probe stopped cleanly in both thinking modes with no raw marker leak. Peak memory footprint about 12.0 GB. |
| Qwen3.6 35B JANGTQ live reasoning gate | PASS. No `<think>` leakage in `.chunk`; reasoning deltas populated. Cosmetic VLM factory warning still appears before LLM factory succeeds. |
| Gemma4 26B JANG live harmony gate | PASS. No harmony markers in `.chunk`; the selected prompt did not elicit `.reasoning` deltas. Peak memory footprint about 27.9 GB. |
| Qwen cache stack live probes | PASS. Paged-prefix hit fired with hybrid rollback semantics; disk L2 restore hit with `ssm_companion`; TurboQuant KV B=2 kept plain-slot output byte-identical beside a TQ slot. |
| Ling 2.6 flash JANGTQ/JANGTQ2/MXFP4 live ChatSession | PASS compile off/on. Multi-turn recall stayed coherent through `ArraysCache` reuse. JANGTQ2 peak footprint about 30.4 GB; MXFP4 peak footprint fixed from ~110 GB to ~66.8 GB by disabling oversized fused gate/up cache materialization. |
| MiniMax M2.7 JANGTQ live perf/coherence probe | PASS for coherent visible text, no loop, no raw marker leakage. Throughput remains below target and is deferred to speed work. |

### 2026-05-06 Cache/SSM WIP Recheck

This recheck was run in `~/vmlx-swift-lm` after the Ling
`enable_thinking` fix landed at `82ce729`.

| Row | Result |
|---|---|
| `swift test -c release --filter SSMStateCacheTests --no-parallel` | PASS with Xcode toolchain after copying `default.metallib` / `mlx.metallib` beside the release test bundle. 7 tests passed, including SSM companion disk round-trip, media-salt isolation, and over-cap eviction. |
| `swift build -c release --product RunBench` | PASS with existing distributed/VL/model warnings only. |
| Ling JANGTQ `BENCH_COHERENT=1` | PASS. Compile off/on both recalled favorite color, identified blue as cool, and produced no loop or raw marker leak. |
| DSV4 Flash short `BENCH_DSV4_COHERENCE BENCH_DSV4_ROW=chat` | PASS after the DSV4 fallback-template fix. Three turns recalled `sapphire-42`; visible answer content arrived through `.chunk` with `unclosedReasoning=false`. |
| DSV4 Flash long row | PASS after local sidecar rebuild and RunBench teardown fix. The 5,568-token row recalled `CERULEAN RIVER / OSLO` with visible final answer, `stop=stop`, no loop, `unclosedReasoning=false`, and clean process exit. |
| Qwen3.6 35B JANGTQ `BENCH_BATCH_TQ_B2=1` | PASS. Plain KV slot stayed byte-identical beside a TurboQuant KV neighbor; both TQ slots decoded coherent text. |
| MiniMax M2.7 Small JANGTQ `BENCH_BATCH_DISK_RESTORE=1` | PASS. Fresh coordinator hit disk L2; prompt time dropped from 13.608s to 0.179s. |
| MiniMax M2.7 Small JANGTQ `BENCH_BATCH_CACHE_HIT=1` | PASS. Paged prefix probe hit 128/186 tokens and warm/cold prompt ratio was 0.36. |
| MiniMax M2.7 full-size speed row | NOT COMPLETED. The process never entered a real heavy load path and was stopped after several minutes at ~300 MiB RSS. Keep prior MiniMax speed status as open. |

Runtime change made in this pass: `CacheCoordinatorConfig.enableSSMReDerive`
now gates the extra synchronous prompt-boundary SSM companion rederive/store
pass in both `Evaluate` and `BatchEngine`. Direct prompt-end SSM seed handling
remains enabled for correctness. Completion `.info` now emits before the
cache store/re-derive pass; detached async SSM rederive is still not a
production path because the prior async helper raced Metal command encoders.

### 2026-05-06 Live Model Continuation

Follow-up rows were run after the DSV4 redownload completed enough to restore
the model shards. These rows are live model checks unless explicitly marked as
metadata/template only.

| Row | Result |
|---|---|
| Qwen3.6 35B JANGTQ `BENCH_QWEN_THINKING_CHECK=1` | PASS. 63 reasoning deltas, empty `.chunk`, no `<think>` marker leak. Peak footprint about 11.2 GB. |
| Qwen3.6 35B JANGTQ `BENCH_QWEN_MULTITURN_TOOL=1` | PASS. Three prompt/tool-style turns had zero reasoning-envelope marker leakage in `.chunk`. The selected budget stayed inside `.reasoning`, which is acceptable for the leak gate. |
| Gemma4 26B JANG_4M `BENCH_HARMONY_CHECK=1` | PASS. Coherent README-template visible text, no harmony markers in `.chunk`. The prompt did not elicit reasoning deltas. |
| Laguna XS.2 JANGTQ `BENCH_LAGUNA_LOOP=1` | PASS. Thinking off/on both produced coherent folder summaries, reached `finish=stop`, and reported `loop=NO`, `unclosedReasoning=NO`, and `leaks=none`. |
| MiniMax M2.7 JANGTQ `BENCH_COHERENT=1` | PASS. Compile off/on both recalled blue and answered cool color correctly. Peak footprint about 61.2 GB. |
| MiniMax M2.7 JANGTQ_K `BENCH_COHERENT=1` | COHERENT / PERF ISSUE. Compile off/on both recalled blue. The first cold run had 163s TTFT under cold shader/cache conditions; rerun after warmup had about 4.2s first TTFT and warm turns around 350 ms. Peak footprint stayed about 80.0 GB, so memory/speed remain optimization work. |
| Qwen3.6 27B JANG_4M `BENCH_COHERENT=1` | PASS with explicit visible-answer policy. The generic 48-token row stayed entirely in reasoning, but `BENCH_THINK_LOOP_PROBE=1 THINK=0` produced visible content with zero reasoning chars, EOS stop, no loop, and no marker leak. |
| Qwen3.6 27B JANG_4M `BENCH_QWEN_THINKING_CHECK=1` | PASS for reasoning split. 95 reasoning deltas, empty `.chunk`, no `<think>` marker leak. Osaurus should set explicit `enable_thinking=false` for normal visible-answer mode on Qwen-family traffic. |
| Ling 2.6 flash JANGTQ2 `BENCH_COHERENT=1` | PASS. Compile off/on both recalled blue and cool color. Peak footprint about 30.4 GB. |
| Ling 2.6 flash MXFP4 `BENCH_COHERENT=1` | PASS. Compile off/on both recalled blue and cool color. Peak footprint about 66.8 GB, not the earlier ~110 GB failure mode. |
| Nemotron Omni Nano JANGTQ `BENCH_OMNI=1 BENCH_OMNI_BATCH=1` | PASS. 17/17 rows passed: text single/multi-turn, image, video encoder, audio encoder, video/audio LMInput, reasoning ON/OFF toggle, mixed image+audio, media-salt isolation, hybrid SSM warm-pass, BatchEngine B=1/B=2/image/audio. |
| Production bundle `BENCH_CONFIG_SMOKE=1` sweep | PASS for Qwen 35B, Qwen 27B, Gemma4, Laguna, MiniMax JANGTQ/JANGTQ_K, Ling JANGTQ2/MXFP4, Nemotron Omni JANGTQ, and DSV4 Flash. DSV4 reports `sidecar=true` after local sidecar rebuild. |
| Production bundle `BENCH_TEMPLATE_SMOKE=1` sweep | PASS for the same set. DSV4 now works with both the local bundle template and the Swift `DSV4Minimal` fallback: `thinking_false` closes `</think>`, `thinking_true` opens `<think>`, and max-effort preface is gated by `enable_thinking=true`. Laguna uses the Swift `LagunaMinimal` Poolside template when the bundle exposes only an include wrapper. Qwen/MiniMax/Nemotron close thinking for `thinking_false`. Ling renders the Bailing "detailed thinking off" system hint in all tested toggle rows. |
| DSV4 Flash `BENCH_DSV4_COHERENCE BENCH_DSV4_ROW=reasoning` | HISTORICAL PASS. Reasoning-off, reasoning-on, and max-effort all answered `12`; thinking rows routed thought text through `.reasoning`, closed reasoning, stopped by EOS/stop, and did not leak raw `<think>` markers. The historical max-effort row used a larger `BENCH_DSV4_REASONING_MAX_TOKENS=384` budget and an explicit repetition-penalty override. Current Osaurus wiring must not turn that artifact into a hidden default; preserve request/bundle sampling and report failures honestly. |

Runtime fix made after this matrix: `MiniMaxJANGTQConfiguration` now decodes
the real JANGTQ_K nested bit map directly from `mxtq_bits.routed_expert`, not
only from factory-normalized `mxtq_gate_up_bits` / `mxtq_down_bits` fields.
Focused unit coverage verifies uniform bits, gate/up/down projection bits,
`quantization` fallback, and explicit field precedence. Targeted smoke on the
local MiniMax M2.7 JANGTQ_K bundle passes with
`routedBits=gateUp:2,down:4`; template smoke also passes for thinking on/off,
multi-turn, and tool rows.

DSV4 model-file issue found during this continuation: the copied
`~/models/JANGQ/DeepSeek-V4-Flash-JANGTQ` bundle had all 78 model
shards, `model.safetensors.index.json`, and tokenizer files, but it was missing
`jangtq_runtime.safetensors` while `jang_config.json` declared
`weight_format="mxtq"`. The safetensors index confirms this DSV4 bundle is a
prestacked JANGTQ layout (`format="jangtq"`,
`rebundled_layout="prestacked-switch_mlp"`) with `tq_packed` / `tq_norms` /
`tq_bits` tensors in the model shards. The small runtime sidecar still must
exist because the TurboQuant kernels need deterministic `signs.{dim}.42` and
`codebook.{dim}.2` tensors. Local sidecar rebuilt with keys:
`signs.4096.42`, `codebook.4096.2`, `signs.2048.42`, and `codebook.2048.2`,
then copied back to `/Volumes/eric-1/models/JANGQ/DeepSeek-V4-Flash-JANGTQ`.
Both copies have SHA-256
`f488d42982781d5653f5bbd6e6d6bd6d93416c9759e2dceaabde4a9817ad571c`.
Model publishing should include that sidecar in the DSV4 Flash bundle.

DSV4 long-context retry status: after the sidecar rebuild and RunBench teardown
fix, DSV4 loaded as `DeepseekV4JANGTQModel` and the strict 5,568-token row
passed cleanly:

- command: `BENCH_DSV4_COHERENCE=1 BENCH_DSV4_ROW=all BENCH_MAX_TOKENS=128 BENCH_DSV4_LONG_REPEAT=220 BENCH_DSV4_LONG_MAX_TOKENS=96 BENCH_DSV4_REASONING_MAX_TOKENS=384`
- long prompt: 5,568 tokens
- answer: `CERULEAN RIVER / OSLO`
- finish: `stop=stop`, `unclosedReasoning=false`, no loop
- memory: max RSS about 69.1 GB; peak footprint about 111.9 GB
- wall time: about 132 s for the full chat + reasoning + long row

Latest speed/coherence sample from the local ignored `RunBench` perf harness
after adding output previews to the audit logs. `BatchEngine observed` is the
production batch path (`BENCH_PERF_PATH=batch`). `Simple observed` is the
single-request `TokenIterator` path (`BENCH_PERF_PATH=iter`), used to separate
model/kernel speed from batching overhead. These are the live speed-progress
numbers from this pass, not final speed acceptance for the slower families:

| Model | Target | BatchEngine observed | Simple observed | Coherence status |
|---|---:|---:|---:|---|
| Qwen3.6 35B JANGTQ | 80 tok/s | 78.8 tok/s | 94.2 tok/s | Coherent visible text, no loop, no leaks. |
| Qwen3.6 27B JANG_4M | 25 tok/s | 25.5 tok/s | not rerun | Coherent visible text, no loop, no leaks. |
| Gemma4 26B JANG_4M | 80 tok/s | 79.1 tok/s | 94.4 tok/s | Coherent visible text, no loop, no leaks. |
| Laguna XS.2 JANGTQ | 80 tok/s | 28.3 tok/s | 31.1 tok/s | Coherent visible text, no loop, no leaks. QKV is now fused at sanitize/load time. |
| MiniMax M2.7 JANGTQ | 45-50 tok/s | 30.9 tok/s | 34.5 tok/s | Coherent visible text, no loop, no leaks. SwitchGLU compile regression removed from default. |
| Ling 2.6 flash JANGTQ2 | 80 tok/s | 29.6 tok/s | 29.5 tok/s | Coherent visible text, no loop, no leaks. |
| Nemotron Omni Nano JANGTQ2 | 90 tok/s | 65.4 tok/s | 76.3 tok/s | Coherent visible text, no loop, no leaks. |
| DSV4 Flash JANGTQ | 20 tok/s | 11.9 tok/s | 13.7 tok/s | Coherent visible text on the perf prompt, no loop, no leaks. Reasoning/long rows are separately passing below. |

Raw local logs are under `docs/benchmarks/speed-2026-05-06/` but that directory
is gitignored. Recreate them with `BENCH_PERF=1` if the PR needs fresh numbers.
Use `BENCH_PERF_PATH=batch` for the production BatchEngine row and
`BENCH_PERF_PATH=iter` for the simple TokenIterator row. `BENCH_SIMPLE=1`
remains the quick single-load sanity check; for Ling JANGTQ it generated 64
tokens at prompt length 30 with about 29.1 GB peak RSS.

Additional HQ continuation after this table:

- Qwen3.6 35B JANGTQ `BENCH_PERF` BatchEngine, 64 tokens:
  **80.4 tok/s**, coherent text, no loop, no leaks, and no factory fallback noise.
- Nemotron Omni Nano JANGTQ2 `BENCH_PERF`, 64 tokens:
  BatchEngine **65.4 tok/s** median / 65.9 best; simple TokenIterator
  **76.3 tok/s** median / 76.9 best. Both runs emitted coherent visible text
  with no loop or marker leaks. Peak footprint was about 15.1 GB. Logs:
  `/tmp/vmlx_omni_jangtq_perf_batch_20260506.log` and
  `/tmp/vmlx_omni_jangtq_perf_iter_20260506.log`.
- Laguna XS.2 JANGTQ `BENCH_TEMPLATE_SMOKE=1 VMLX_CHAT_TEMPLATE_FALLBACK_LOG=1`:
  PASS. The tokenizer bridge selected `LagunaMinimal`; thinking-off rendered a
  closed `</think>` prompt tail, thinking-on opened `<think>`, and assistant
  reasoning history rendered as `<think>...</think>` followed by visible
  content.
- Laguna XS.2 JANGTQ strict `BENCH_LAGUNA_LOOP=1 BENCH_MAX_TOKENS=512`:
  PASS. Thinking-off generated 374 tokens and stopped; thinking-on generated
  171 tokens and stopped; no loop, unclosed reasoning, or marker leak. Log:
  `/tmp/vmlx_laguna_loop_strict_default_after_20260506.log`.
- MiniMax M2.7 JANGTQ TokenIterator, 64 tokens: baseline **34.7 tok/s**;
  `VMLX_MINIMAX_ROUTER_COMPILE=1` **34.6 tok/s**;
  `VMLX_TQ_SWITCH_GLU_COMPILE=0` **35.1 tok/s**. The current MiniMax gap is
  therefore not a simple router-compile or SwitchGLU-compile default issue.
- Speed continuation after the JANGTQ runtime audit:
  - `TurboQuantSwitchGLU` whole-path compile is now opt-in via
    `VMLX_TQ_SWITCH_GLU_COMPILE=1`. Default compiled SwitchGLU regressed
    MiniMax M2.7 simple decode into the high-20s / low-30s tok/s band; the
    plain custom Metal chain restored MiniMax to **34.5 tok/s** simple and
    **30.9 tok/s** BatchEngine with identical coherent text.
  - DSV4 was also checked both ways. On the production BatchEngine path,
    compile-off/default measured **11.9 tok/s** while forced compile measured
    lower in the same pass. Keep DSV4 limited-SwiGLU correctness in the Metal
    kernel, but do not silently enable whole-SwitchGLU compile for serving.
  - Laguna attention now fuses affine q/k/v at sanitize time into `qkv_proj`,
    matching the Python JANGTQ P18 optimization and MiniMaxJANGTQ's Swift
    sanitize path. It is coherent and gives a small simple-path lift; the large
    80 tok/s gap remains routed-MoE/kernel work, not template or loop failure.
  - The local full-size MiniMax bundle reports 256 local experts and affine
    `group_size=64`; the historical 45-50 tok/s MiniMax reference was for a
    different 139B/154-expert/gs=128 profile. Treat any remaining MiniMax speed
    gap as both runtime-kernel and model-file/profile audit work.
- Qwen cache rows after a local ignored RunBench engine-shutdown cleanup:
  `BENCH_BATCH_CACHE_HIT`, `BENCH_BATCH_DISK_RESTORE`, and
  `BENCH_BATCH_TQ_B2` all PASS and exit 0. Before this cleanup the cache-hit row
  printed PASS but could return 139 during process teardown. `RunBench/` remains
  gitignored in this checkout, so treat that harness cleanup as local validation
  hygiene unless the benchmark target is intentionally promoted into source.

The requested Python file
`~/mlx/vllm-mlx/docs/DSV4_FIX_NUANCES.md` was not present locally.
The equivalent local Python-runtime notes are in
`~/mlx/vllm-mlx/docs/DSV4_RUNTIME_REGRESSION_TRACE.md` and
`~/mlx/vllm-mlx/docs/DSV4-PYTHON-AUDIT-2026-05-03.md`; their relevant
requirements are reflected here: preserve DSV4 prompt-mode kwargs, keep DSV4
prefill single-shot, preserve SWA+CSA+HSA hybrid cache state, and never route
DSV4 through paged KV ownership.

## Late 2026-05-06 Production Matrix

This continuation was run on the local M5 Max MacBook only. The stop-hook model
paths under `~/.mlxstudio` and `~/osaurus_models` were not present on this
machine, so the rows below use the available local production-equivalent
bundles under `~/models/dealign.ai` and `~/models/JANGQ`.

Runtime fixes made in this continuation:

- Rotating/sliding-window cache topologies are now marked paged-incompatible in
  both `BatchEngine` and `TokenIterator`. Gemma4/SWA prefix reuse therefore
  restores via the disk serializer, which carries `.rotating` layer metadata,
  instead of the paged tier, which only stores full-history KV blocks.
- DSV4 chat-template context strips `reasoning_effort` when
  `enable_thinking=false`, and the DSV4 fallback templates gate the max-effort
  preface on `enable_thinking=true`.
- `BENCH_BATCH_LONG_CONTEXT` now applies the same EOS-prefix comparison used by
  the short cross-engine validator; raw `TokenIterator` can yield EOS as a
  token, while `BatchEngine` correctly stops before surfacing it.
- `BENCH_PERF` now prints `promptTokens=...` so long-context/TurboQuant logs
  prove the actual context size.

Fresh live rows:

| Row | Result |
|---|---|
| DSV4 Flash full coherence `BENCH_DSV4_COHERENCE=1 BENCH_DSV4_ROW=all` | PASS. Three-turn chat recalled `sapphire-42`; reasoning off/on/max all answered `12`; 5,568-token long-context row recalled `CERULEAN RIVER / OSLO`; `stop=stop`, `unclosedReasoning=false`. Log: `/tmp/vmlx_hook_dsv4_coherence_all_20260506.log`. |
| DSV4 template kwargs after fix | PASS. Chat+max suppresses the max preface while `enable_thinking=false`; thinking+max keeps the max preface. Log: `/tmp/vmlx_hook_dsv4_template_kwargs_after_fix_20260506.log`. |
| DSV4 reasoning after fix | PASS. Reasoning off/on/max still route correctly after the context coercion. Log: `/tmp/vmlx_hook_dsv4_reasoning_after_template_fix_20260506.log`. |
| Gemma4 26B production matrix after SWA cache fix | PASS 7/7. Same-prompt cache-hit row returned `4` instead of the prior repeated-text corruption; TTFT improved from 351 ms to 59 ms on the hit row. Log: `/tmp/vmlx_hook_gemma4_26b_prod_after_swa_cache_fix_20260506.log`. |
| Ling 2.6 flash JANGTQ2 production matrix | PASS 7/7. Reasoning toggles, cache-hit row, and UTF-8 row passed; model load delta was about 28 GB and peak process RSS about 57 GB in this harness. Log: `/tmp/vmlx_hook_ling_prod_20260506.log`. |
| Qwen3.6 long-context cross-engine | PASS. 2,048-token synthetic prompt matched the BatchEngine prefix, then BatchEngine stopped at EOS token `248046` that raw TokenIterator continued through. Log: `/tmp/vmlx_hook_qwen36_long_context_2048_after_stopfix_20260506.log`. |
| Nemotron Omni JANGTQ multimodal matrix | PASS 17/17. Text, image, video encoder, audio encoder, video/audio LMInput, reasoning toggle, mixed image+audio, media-salt isolation, hybrid SSM warm-pass, and BatchEngine text/image/audio rows passed. Log: `/tmp/vmlx_hook_nemotron_omni_matrix_20260506.log`. |
| Config smoke sweep | PASS for Qwen3.6 35B, Gemma4 26B, Nemotron Omni, Ling JANGTQ2, MiniMax M2.7, Laguna XS.2, and DSV4 Flash. Warnings remain for known BOS/EOS overlap/mismatch on Qwen, MiniMax, and Laguna. Logs: `/tmp/vmlx_hook_*_config_smoke_20260506.log`. |
| Template smoke sweep | PASS for Qwen3.6, Ling, and DSV4. Logs: `/tmp/vmlx_hook_*_template_smoke_20260506.log`. |

ZAYA addendum:

- Local ZAYA bundles inspected under `~/jang/models/Zyphra`: source
  `ZAYA1-8B`, `ZAYA1-8B-JANGTQ2`, `ZAYA1-8B-JANGTQ4`, and `ZAYA1-8B-MXFP4`.
- Metadata smoke passed for JANGTQ2/JANGTQ4/MXFP4 after fixing the local
  bundle `generation_config.json` EOS from `1` to tokenizer/config EOS `106`.
  JANGTQ bundles report `model_type=zaya`, 80 layers, `weight_format=mxtq`,
  per-role `mxtq_bits`, `tq_in_features` count 120, and
  `jangtq_runtime.safetensors` present. MXFP4 reports `weight_format=mxfp4`.
  Logs:
  `/tmp/vmlx_zaya_JANGTQ2_config_smoke_after_eos_20260506.log`,
  `/tmp/vmlx_zaya_JANGTQ4_config_smoke_after_eos_20260506.log`, and
  `/tmp/vmlx_zaya_MXFP4_config_smoke_after_eos_20260506.log`.
- ZAYA CCA contract gate passed for JANGTQ2/JANGTQ4/MXFP4. The gate asserts
  40 even CCA attention layers, 40 odd pre-stacked MoE layers, `cache_subtype`
  `zaya_cca`, hybrid cache metadata, `conv_qk`/`temp`/`linear_q`/`linear_k`/
  `val_proj1`/`val_proj2` counts, sidecar/TQ tensor counts, `tq_in_features`,
  tokenizer template presence, and effective EOS coverage. Logs:
  `/tmp/vmlx_focus_zaya_JANGTQ2_contract_20260506.log`,
  `/tmp/vmlx_focus_zaya_JANGTQ4_contract_20260506.log`, and
  `/tmp/vmlx_focus_zaya_MXFP4_contract_20260506.log`.
- Template smoke passed for JANGTQ2/JANGTQ4/MXFP4. The bundle template renders
  the Gemma-style `<|im_start|>` transcript and closes thinking when
  `enable_thinking=false`. Logs:
  `/tmp/vmlx_focus_zaya_JANGTQ2_template_20260506.log`,
  `/tmp/vmlx_focus_zaya_JANGTQ4_template_20260506.log`, and
  `/tmp/vmlx_focus_zaya_MXFP4_template_20260506.log`.
- Current vmlx-swift-lm status is now a functional text decoder, not the
  previous explicit unsupported route. `LLMModelFactory.dispatchZaya` maps
  BF16, JANGTQ2, JANGTQ4, and MXFP4 bundles to one `ZayaModel` implementation.
  The CCA decoder follows the Zyphra reference order: q/k projections, q/k mean
  residuals, two causal `conv_qk` kernels with no activation between them,
  delayed `prev_hs` value projection, fp32 q/k L2 normalization, key
  temperature, RoPE, then standard GQA attention.
- The important JANGTQ correctness fix is fp32 L2 normalization before casting
  q/k back to the model dtype. The previous fp16 path could overflow
  `(k * k).sum` inside CCA and corrupt JANGTQ2/JANGTQ4 next-token logits.
- Real forward smoke passes for BF16, JANGTQ2, JANGTQ4, and MXFP4. Focused
  release tests pass for `Zaya` plus tool parser coverage:
  93 selected tests, 0 failures.
- Short live decode on the France prompt:
  - BF16: `Paris.`, 49.0 tok/s, `stop=stop`.
  - JANGTQ2: `Paris.`, 45.9 tok/s, `stop=stop`.
  - JANGTQ4: correct but verbose answer, 47.7 tok/s, `stop=stop`.
  - MXFP4: correct but verbose answer, 59.8 tok/s, `stop=length` at 48 tokens.
- ZAYA tool parser is wired as `ToolCallFormat.zayaXml` /
  `capabilities.tool_parser = "zaya_xml"`. It uses Zyphra wrapper tags
  `<zyphra_tool_call>...</zyphra_tool_call>` around the existing
  `<function=...><parameter=...>` XML body.
- Live B=2 raw decode with JANGTQ2 passes and exercises the
  `BatchZayaCCACache` path. A production-shaped real-bundle Swift test now also
  passes: `UserInput(chat:)`, `enable_thinking=false`, `BatchEngine.generate`,
  two different prompt lengths, no visible thinking/tool marker leakage. The
  fix was to build the batch attention mask from pre-update per-slot offsets
  and use `BatchZayaCCACache.offsetArray` for RoPE; this avoids the prior
  `(2,1,1,33)` vs `(2,8,1,32)` mask/KV broadcast crash.
- Live exact disk restore with JANGTQ2 passes: the second coordinator hit disk
  for a 140/140-token exact prompt and reduced prompt time from 0.172 s to
  0.037 s. Prefix-extension cache hit intentionally does not pass for ZAYA
  today; a generic prefix probe returned miss for a 140-token prefix inside a
  163-token prompt before the bench harness was made capability-aware. The
  current `BENCH_BATCH_CACHE_HIT=1` rerun reports `not applicable
  (paged-incompatible topology)` after `BatchEngine` flips
  `CacheCoordinator.isPagedIncompatible` on first ZAYA admission. Keep prefix
  hits disabled until CCA state can be restored at the exact matched prefix
  boundary.
- Thinking-off JANGTQ2 probe routes content with no unclosed reasoning. On the
  validation thinking-on probe, ZAYA stopped at EOS inside reasoning after 21
  tokens and emitted no visible content. Keep ZAYA thinking-on/max gated until
  a live row proves close-tag/content transition.
- Prefix caching remains disabled for ZAYA in this port. Paged KV may only
  cover the standard K/V tensors and must not report a complete prefix hit
  unless the matching CCA state is restored for the exact same prefix length.
- TurboQuant KV, if enabled later, should compress only standard K/V pages.
  Keep CCA `conv_state` and `prev_hs` float32 until single-shot versus
  chunked-prefill parity and cache-restore parity are proven.
- Bundle issue fixed locally: converted ZAYA `generation_config.json` files now
  set `eos_token_id=106` to match `config.json` and tokenizer
  `<|im_end|>`. Keep that fix in the published model bundles to avoid
  stop-condition drift.

ZAYA remaining production checklist:

| Axis | Current status / required behavior before broad serving |
|---|---|
| Reasoning off | Template smoke closes thinking for `enable_thinking=false`; short live decode emits visible `.chunk` text with no raw `<think>` leakage on the tested prompt. |
| Reasoning on/max | Not ready. Template smoke opens thinking, but the live thinking-on probe stopped inside reasoning and emitted no visible content. Needs rows proving `.reasoning` / `.chunk` split, bounded max-effort behavior, and EOS stop with no unclosed reasoning. |
| Async rederive / warm pass | Existing SSM rederive is for Mamba/Arrays recurrence and must not be reused blindly. ZAYA needs a CCA rederive path that captures KV plus `conv_state` and `prev_hs` at prompt/block boundaries, or cache hits must stay disabled. |
| Paged prefix | Disabled/incompatible today. A KV-only page hit would be wrong because CCA state is path-dependent. |
| Disk L2 | Unit disk round-trip for `LayerKind.zayaCCA` passes, including mixed KV/ZAYA/rotating layers. Live exact same-prompt disk restore passes. Prefix-extension restore remains disabled until exact-boundary CCA restore is implemented. |
| TurboQuant KV | Not a default. It may apply only to ordinary K/V tensors after CCA parity is proven; CCA `conv_state` and `prev_hs` stay float32. |
| Continuous batching | Per-slot `BatchZayaCCACache` gather/scatter isolation passes in unit tests. Live B=2 raw decode and production chat-stream B=2 pass after the pre-update-mask / offsetArray RoPE fix. |
| JANGTQ weights | JANGTQ2/JANGTQ4 load pre-stacked odd-layer `switch_mlp` experts with per-role `mxtq_bits`, sidecar, and 120 `tq_in_features`; no per-expert tensor re-expansion on load. |

TurboQuant KV cache rows:

| Model | Row | Result |
|---|---|---|
| Qwen3.6 35B JANGTQ | `BENCH_BATCH_TQ_B2=1` | PASS. Plain slot stayed byte-identical beside TQ(4,4). Log: `/tmp/vmlx_hook_qwen36_tq_b2_20260506.log`. |
| Gemma4 26B JANG_4M | `BENCH_BATCH_TQ_B2=1` | PASS. Plain slot stayed byte-identical beside TQ(4,4). Log: `/tmp/vmlx_hook_gemma4_26b_tq_b2_20260506.log`. |
| Nemotron Omni JANGTQ | `BENCH_BATCH_TQ_B2=1` | PASS. Plain slot stayed byte-identical beside TQ(4,4). Log: `/tmp/vmlx_hook_nemotron_omni_tq_b2_20260506.log`. |
| Qwen3.6 35B JANGTQ | TQ(3,3) long context | PASS. `promptTokens=7464`, 32 generated tokens, 56.0 tok/s, no loop or marker leak. Log: `/tmp/vmlx_hook_qwen36_tq33_long_perf_after_promptcount_20260506.log`. |
| Qwen3.6 35B JANGTQ | TQ(4,4) long context | PASS. `promptTokens=7464`, 32 generated tokens, 55.1 tok/s, no loop or marker leak. Log: `/tmp/vmlx_hook_qwen36_tq44_long_perf_20260506.log`. |

KV-mode speed spot checks from the same build:

| Model | Float KV | TQ(3,3) | TQ(4,4) | Policy note |
|---|---:|---:|---:|---|
| Qwen3.6 35B JANGTQ | 76.5 tok/s | 71.2 tok/s | 70.8 tok/s | Functionally good; ~7% speed cost in this one-run probe. |
| Nemotron Omni JANGTQ | 66.6 tok/s | 63.4 tok/s | 63.3 tok/s | Functionally good; within about 5%. |
| Gemma4 26B JANG_4M | 78.2 tok/s | 46.2 tok/s | 46.3 tok/s | Functionally correct but not a production default for SWA/rotating Gemma4 until a compressed rotating-cache path exists. |

Batching rows:

- Qwen3.6 B=2: PASS, slot 0 byte-identical, batched/serial ratio 0.93.
  Log: `/tmp/vmlx_hook_qwen36_batch_b2_20260506.log`.
- Qwen3.6 B=4: correctness PASS but throughput gate FAIL by the harness's
  strict cutoff (`ratio=0.95`). Treat as a speed/scheduler efficiency item, not
  cross-slot corruption. Log: `/tmp/vmlx_hook_qwen36_batch_b4_20260506.log`.
- Gemma4 B=4: PASS, slot 0 byte-identical, ratio 0.45; throughput assertion
  skipped because token counts were intentionally uneven. Log:
  `/tmp/vmlx_hook_gemma4_26b_batch_b4_20260506.log`.

External dirty state to keep out of this repo:

- `~/jang` contains unrelated DSV4/ZAYA/model-tool edits from another
  agent, including local model metadata patches. Do not infer vmlx-swift-lm
  source truth from that dirty tree without a separate review.
- `~/vmlx/swift` is also dirty with app/runtime changes from another
  agent. This handoff and the commit from this pass touch only
  `vmlx-swift-lm`.

## 2026-05-06 Hook Model-Gate Continuation

The local stop hook requires exact paths that are not present on this M5 host:

- `~/models/Qwen3.5-35B-A3B-4bit`
- `~/osaurus_models/finished/gemma-4-e2b-it-4bit`
- `~/osaurus_models/finished/gemma-4-e4b-it-4bit`
- `~/osaurus_models/finished/gemma-4-26b-a4b-it-4bit`
- `~/.mlxstudio/models/Nemotron-Cascade-2-30B-A3B-JANG_2L`
- `~/.mlxstudio/models/Nemotron-3-Super-120B-A12B-JANG_2L`
- `~/models/Mistral-Small-4-119B-JANG_2L`
- `~/.mlxstudio/models/Qwen3.5-VL-35B-A3B-JANG_4K-CRACK`
- `~/.mlxstudio/models/Qwen3.5-VL-122B-A10B-JANG_4K-CRACK`
- `~/.mlxstudio/models/MiniMax-M2.5-JANG_2L-CRACK`

No acceptance claim is made for those exact bundles. The closest available
local substitutes were tested sequentially with:

```bash
BENCH_PERF=1 BENCH_PERF_PATH=batch BENCH_PERF_WARMUP=0 \
BENCH_PERF_RUNS=1 BENCH_MAX_TOKENS=64
```

| Model | TTFT | Decode | Output status | Memory |
|---|---:|---:|---|---:|
| Qwen3.6-35B-A3B-JANGTQ-CRACK | 82 ms | 75.4 tok/s | coherent, no loop, no marker leaks | 10.2 GiB max RSS |
| Gemma-4-26B-A4B-it-JANG_4M-CRACK | 325 ms | 75.0 tok/s | coherent, no loop, no marker leaks | 12.8 GiB max RSS |
| Nemotron-Omni-Nano-JANGTQ-CRACK | 159 ms | 63.8 tok/s | coherent, no loop, no marker leaks | 7.2 GiB max RSS |
| MiniMax-M2.7-JANGTQ | 1,018 ms | 30.6 tok/s | coherent, no loop, no marker leaks | 52.6 GiB max RSS |
| Ling-2.6-flash-JANGTQ2-CRACK | 340 ms | 29.0 tok/s | coherent, no loop, no marker leaks | 27.8 GiB max RSS |
| Laguna-XS.2-JANGTQ | 160 ms | 27.8 tok/s | coherent, no loop, no marker leaks | 8.8 GiB max RSS |
| DeepSeek-V4-Flash-JANGTQ | 9,097 ms | 8.4 tok/s | coherent, no loop, no marker leaks | 62.3 GiB max RSS |

Logs:

- `/tmp/vmlx_hook_available_qwen36_35b_perf_20260506.log`
- `/tmp/vmlx_hook_available_gemma4_26b_perf_20260506.log`
- `/tmp/vmlx_hook_available_nemotron_omni_perf_20260506.log`
- `/tmp/vmlx_hook_available_minimax_m27_perf_20260506.log`
- `/tmp/vmlx_hook_available_ling_jangtq2_perf_20260506.log`
- `/tmp/vmlx_hook_available_laguna_xs2_perf_20260506.log`
- `/tmp/vmlx_hook_available_dsv4_flash_perf_20260506.log`

The current `RunBench` perf harness reports TTFT, decode tok/s, stop reason,
visible/reasoning character counts, loop/leak checks, and process memory via
`/usr/bin/time -l`. It does not currently emit graph-node or `AsType`
primitive counts, so those hook fields remain unverified until an MLX graph
instrumentation path is added.

## Speed / Dtype Contract For New Runtime Work

The previous "speed stuck" cluster was real and should be treated as production
history, not old notes:

| Date/commit cluster | What it fixed | Current contract |
|---|---|---|
| 2026-04-11 to 2026-04-14, `2859808`, `06721aa`, `a8a6a6f`, `d0706af` | Float32 scalar/`AsType` cascades, universal bf16 conversion, and the JANGTQ-native bf16 bypass. | New cache/runtime code must not introduce untyped floating `MLXArray(...)` scalars in decode/prefill paths. Non-JANGTQ parameters convert to bf16 at load; JANGTQ-native keeps fp16 TurboQuant norms because the Metal kernels use norm dtype in their signature. |
| 2026-04-13, `cf55f6d`, `21176a4`, `d4e4e45`, `0e36d38` | Compile micro-fusion islands and Qwen/GatedDelta dtype cleanup. | Keep existing compiled helper paths and `asyncEval` decode ordering intact. Do not add `.item()` or synchronous `eval` inside compiled traces or per-token loops except at the sampler/EOS boundary. |
| 2026-04-15, `fb46fbd` | Fused int4 MoE gate/up gather and SwiGLU path. | Routed JANGTQ MoE should use `TurboQuantSwitchGLU` / JANGTQ kernels, including `gateUpBits` and `downBits` for JANGTQ_K. Do not re-expand per-expert tensors in forward paths. |
| 2026-05-02, `102d80c` | MiniMax router-gate fp32 precision parity with Python. | Router precision exceptions must be model-specific and documented. Do not blindly remove every `.asType(.float32)`; keep MiniMax router fp32, DSV4 mHC wide reductions fp32, MLA single-token fp32 SDPA, and Bailing recurrent GLA fp32 state math. |

Audit result for the new cache/JangPress stack on 2026-05-06:

- `CacheCoordinator`, paged cache, SSM companion, and disk L2 paths use `MLX.eval`
  for cache materialization/store/restore boundaries, not as hidden per-token
  decode work.
- `BatchEngine` keeps the April decode perf contract: B=1 bypass, conditional
  `Task.yield()`, and `asyncEval(logits)` / `asyncEval(sampledTokens)` before
  token readback.
- `TQDiskSerializer` readbacks are metadata/header reads for serialization, not
  model-forward dtype promotion.
- `LoadTimeStacking` materializes load-time per-expert JANGTQ stacks so large 3D
  routed tensors do not retain every per-expert source tensor until final eval.
  It must not be used in model forward paths.
- `ModelFactory` fallback attempts are quiet by default. Set
  `VMLINUX_MODEL_FACTORY_TRACE=1` only when diagnosing factory routing; thrown
  load errors still preserve the most informative factory failure.
- JangPress mmap/prestack is a residency policy. It must not cast weights, change
  JANGTQ norm dtypes, or replace `TurboQuantSwitchGLU`.
- JangPress router advice is default-off because it reads router indices back to
  CPU. It is exact and correct, but it is not yet tok/s-neutral; enable only for
  experiments until the readback path is replaced or proven neutral.

When adding new model/cache code, run the speed-contract grep before calling it
production-ready:

```bash
rg -n "MLXArray\\((0\\.0|1\\.0|[0-9]+\\.?[0-9]*|Float\\(|Double\\()" Libraries
rg -n "softmax\\([^\\n]*asType\\(\\.float32\\)|sigmoid\\([^\\n]*asType\\(\\.float32\\)" Libraries
rg -n "\\.item\\(|MLX\\.eval\\(|Memory\\.clearCache" Libraries/MLXLMCommon/BatchEngine Libraries/MLXLMCommon/Cache
```

Allowed hits must be explained by one of the contracts above. Otherwise treat
them as likely speed regressions until proven with a live model run.

## Non-Negotiable Integration Contract

1. Load through vmlx model loaders.

   Use `loadModel(from:using:loadConfiguration:)` or the existing
   `MLXLMCommon.loadModel(from:using:)` wrappers. Do not bypass the loader for
   JANG/JANGTQ bundles. The loader stamps:

   - `ModelConfiguration.toolCallFormat`
   - `ModelConfiguration.reasoningParserName`
   - effective EOS IDs and extra EOS strings
   - tokenizer fallback for weights-only JANG bundles
   - JANG/JANGTQ metadata, bits, and sidecar paths

2. Build chat requests with `UserInput(chat:)`.

   Production chat should flow:

   ```swift
   var input = UserInput(chat: messages)
   input.additionalContext = [
       "enable_thinking": enableThinking,
       "reasoning_effort": reasoningEffort
   ]
   let lmInput = try await context.processor.prepare(input: input)
   let stream = await batchEngine.generate(input: lmInput, parameters: params)
   ```

   Raw text prompts are acceptable for benchmark/perf probes, FIM/code
   completions, and deliberately template-free tests. They are not a substitute
   for production chat-template coverage.

3. Consume library stream events directly.

   `Generation.chunk(String)` is user-visible assistant text.
   `Generation.reasoning(String)` is a separate reasoning delta.
   `Generation.toolCall(ToolCall)` is authoritative tool-call output.
   `Generation.info(GenerateCompletionInfo)` is telemetry.

   Osaurus should not re-parse `<think>`, harmony channels, DSML, Qwen XML
   function calls, MiniMax tool syntax, or Gemma tool envelopes from `.chunk`.
   If raw markers appear in `.chunk`, treat that as a vmlx bug.

4. Keep request-level reasoning policy explicit.

   Always set `additionalContext["enable_thinking"]` for families with
   template-controlled thinking. When a model supports effort levels, set
   `additionalContext["reasoning_effort"]` to `"high"` or `"max"` only for
   requests that actually need it.

5. Keep cache ownership in vmlx.

   Osaurus should configure `CacheCoordinatorConfig`; vmlx should decide the
   per-model cache topology. Do not bolt on app-layer cache guards around
   DSV4/Ling/SSM. If a prefix hit is unsafe for a topology, BatchEngine must
   roll back or route to the correct tier.

## Reasoning And Tool Streaming

The stream split is library-level:

```swift
for await event in stream {
    switch event {
    case .chunk(let text):
        appendVisibleAssistantText(text)
    case .reasoning(let delta):
        appendReasoningDelta(delta)
    case .toolCall(let call):
        enqueueToolCall(call)
    case .info(let info):
        recordTelemetry(info)
    }
}
```

Important behavior:

- `.reasoning` can be non-empty on thinking-enabled rows and should always be
  rendered separately from visible answer text.
- `.chunk` can be empty at `max_tokens` on thinking-enabled short-budget rows
  while `.reasoning` contains useful text. This is a length/state policy issue,
  not marker leakage. The UI should surface a reasoning-only or length-finished
  state instead of treating it as no output.
- Thinking-on short prompts may end before the model closes `</think>`. Do not
  assume every thinking turn produces visible answer text within tiny budgets.
- Tool calls belong in `.toolCall`; do not scan visible chunks for raw tool
  syntax.

Relevant files:

- `Libraries/MLXLMCommon/ReasoningParser.swift`
- `Libraries/MLXLMCommon/Tool/ToolCallProcessor.swift`
- `Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift`
- `Libraries/MLXLMCommon/BatchEngine/REASONING-STREAM-EVENT.md`
- `Libraries/MLXLMCommon/BatchEngine/STOP-SEQUENCES-CONTRACT.md`

## Chat Templates And Jinja

The tokenizer/processor path owns template rendering. Use `UserInput(chat:)`
plus `additionalContext`; do not manually splice role markers.

Family notes:

| Family | Template/runtime note |
|---|---|
| DSV4 | Uses DSV4 fallback/encoder semantics with `<｜User｜>`, `<｜Assistant｜>`, `<think>`, DSML tools, and `reasoning_effort`. |
| Qwen 3.x / 3.6 | `enable_thinking` affects prompt tail; parser uses decoded prompt tail via `ReasoningParser.forPrompt`. |
| Gemma 4 | Harmony parser is channel based. Use Gemma4 fallback templates when native Swift Jinja compatibility is not enough. |
| MiniMax M2.7 | JANG fallback detects MiniMax tokens by BOS/EOS, not by fragile `convertTokenToId` checks. |
| Laguna | Poolside/Laguna bundles can expose only `{% include 'chat_template.jinja' %}`. vmlx selects `LagunaMinimal` from BOS/EOS and `<assistant>` / `<think>` sentinels; loop probe is the production smoke. |
| Nemotron-Omni | Text/image/video/audio rows go through Omni processor; reasoning parser strips raw markers from validation summaries. |

Swift-Jinja now resolves through the osaurus-owned chain:
`osaurus-ai/Jinja` carries the HuggingFace 2.3.5 code, and
`osaurus-ai/swift-transformers` is pinned to the osaurus fork that depends on
that package. Do not reintroduce a direct `huggingface/swift-jinja` edge.

## DSV4 Production Notes

DSV4 is its own architecture: SWA + CSA/HSA compressor/indexer hybrid attention,
not ordinary KV, and not Mamba/SSM.

Required runtime behavior:

- Default `newCache` uses:
  - `RotatingKVCache(window=128)` for `compress_ratio == 0` layers.
  - `DeepseekV4Cache(window=128, compressRatio=cr)` for compressed layers.
- `BatchEngine` marks DSV4 hybrid-pool caches as paged-incompatible so the
  paged tier does not claim unsafe prefix hits.
- Disk/L2 serialization uses `TQDiskSerializer` with `LayerKind.deepseekV4`,
  including rotating window state plus compressor/indexer buffers and nil masks.
- DSV4 prefill is single-shot for hybrid-pool caches. Do not chunk DSV4 prompt
  prefill at the app layer.
- `DSV4_KV_MODE=full|tq` is diagnostic/operational override only. Production
  should default to the hybrid cache unless the operator deliberately opts into
  the memory/quality tradeoff.

Reasoning modes:

- `enable_thinking=false`: plain-answer mode. The current Swift fallback closes
  the thinking block at the prompt tail and the strict live DSV4 row produced
  visible `.chunk` text with `unclosedReasoning=false`.
- `enable_thinking=true`: normal reasoning stream.
- `reasoning_effort="max"`: template preface is applied and can consume more
  budget before visible answer. Use a larger budget and the explicit
  request/bundle sampling values; do not add a hidden repetition penalty or
  sampler clamp because a historical diagnostic used one.

Live DSV4 gate added:

```bash
BENCH_DSV4_COHERENCE=1 \
  BENCH_DSV4_ROW=chat|reasoning|long|all \
  BENCH_MODEL=~/models/JANGQ/DeepSeek-V4-Flash-JANGTQ \
  .build/release/RunBench
```

Observed on 2026-05-06:

| Row | Result |
|---|---|
| `chat` | PASS, clean exit. Multi-turn recalled `sapphire-42`; turn 3 answered sapphire/blue follow-up. All three turns had visible `.chunk` text and `unclosedReasoning=false`. |
| `reasoning` | PASS, clean exit. Reasoning off/on/max routed without raw `<think>` in `.chunk`; arithmetic answer was present in visible output; thinking rows closed reasoning. |
| `long` repeat=120 | PASS, clean exit. 3,068 prompt tokens, recalled buried `CERULEAN RIVER, OSLO`. |
| `long` repeat=220 | PASS, clean exit. 5,568 prompt tokens, recalled buried `CERULEAN RIVER, OSLO`, `stop=stop`, no loop, `unclosedReasoning=false`. |
| `long` repeat=650 | OOM during long prefill/decode. Treat as a memory ceiling until DSV4 long-context memory is hardened. |

For production, use DSV4 long-context chat confidently past the 128-token local
window at the validated 5,568-token scale. Treat much larger prompts, including
the 650-filler stress row, as a live memory-pressure area on a 128 GB M5 Max
until further hardening lands.

## Ling / Bailing Runtime Notes

The Ling RAM/stall issue had multiple causes; do not reduce it to "missing 3D
stacked tensors."

The original multi-turn coherence bug was the MLA cache path:

- Bailing MLA manually updates the MLA/KV cache.
- It must then call the no-update MLA SDPA helper.
- Calling the helper that also updates cache appends keys/values twice, grows
  cache offsets incorrectly, stalls, and breaks multi-turn recall.

The 2026-05-06 MXFP4 110 GB peak was a separate decode-fusion residency bug:

- Ling MXFP4 stores routed expert banks as large pre-quantized 3D
  `switch_mlp.{gate,up,down}_proj.{weight,scales,biases}` tensors.
- `SwitchGLU.ensureFusedGateUp()` previously concatenated gate+up into a
  persistent fused cache on first forward. For Ling MXFP4 this was about 1 GiB
  per routed layer and effectively kept a second expert bank resident.
- The fix keeps gate/up fusion for normal-sized MoE layers but skips persistent
  fusion when the fused cache would exceed 512 MiB
  (`VMLX_FUSED_GATE_UP_CACHE_LIMIT_MB` / `_BYTES` override available). Ling
  MXFP4 now uses the regular two-projection path for correctness and memory.
- The loader also now builds `QuantizedLinear` / `QuantizedSwitchLinear`
  modules directly from pre-quantized safetensor arrays, drops the staging
  weight dictionary before post-load materialization, chunk-converts bf16
  casts, and clears the allocator cache after load.

Current expected behavior:

- Do not reset `ArraysCache` on each ChatSession turn.
- Do not cap `ChatSession` prefill at the app layer for Ling.
- Recurrent GLA memory control belongs in the Bailing/Ling recurrent path, not
  in a generic chat-session guard.
- Load-time JANGTQ restacking materializes 3D routed expert tensors before
  source tensors are dropped.
- Batched recurrent decode must preserve per-slot logical positions. The
  `BatchArraysCache` wrapper exposes `offsetArray` for Ling/Bailing RoPE and
  writes per-slot offsets back in `splitBack()` instead of collapsing every
  slot to the batch maximum.
- Bailing MLA uses asymmetric K/V cache dimensions (`K=192`, `V=128` on the
  local Ling JANGTQ2 bundle). TurboQuant KV must keep separate key-side and
  value-side encoder states; a single key-derived state corrupts value
  encode/decode and reproduces the former `(1,32,24,128)` vs `(192)` crash.

Live validation already recorded:

- `Ling-2.6-flash-JANGTQ BENCH_SIMPLE BENCH_PROMPT_LEN=30 BENCH_MAX_TOKENS=64`:
  PASS, around 29.1 GB peak RSS.
- `Ling-2.6-flash-JANGTQ BENCH_COHERENT BENCH_MAX_TOKENS=64`: PASS, recalled
  `blue` and classified it as cool.
- `Ling-2.6-flash-JANGTQ2-CRACK BENCH_COHERENT BENCH_MAX_TOKENS=96`: PASS
  compile off/on, peak footprint ~30.4 GB, max RSS ~29.8 GB.
- `Ling-2.6-flash-MXFP4-CRACK BENCH_COHERENT BENCH_MAX_TOKENS=96`: PASS
  compile off/on, peak footprint ~66.8 GB, max RSS ~49.2 GB. Before the
  fusion-cache fix this same row peaked at ~110 GB footprint.
- Ling JANGTQ2 perf, `BENCH_PERF_PATH=batch`, 160 tokens: median ~23.7 tok/s.
- Ling JANGTQ2 perf, `BENCH_PERF_PATH=iter`, 160 tokens: median ~28.5 tok/s.
- Ling MXFP4 perf, 160 tokens: median ~6.0 tok/s after memory fix. This is the
  expected tradeoff until a memory-safe fused/streamed MXFP4 gate+up path exists.
- `Ling-2.6-flash-JANGTQ2-CRACK BENCH_BATCH_TQ_B2=1 BENCH_MAX_TOKENS=16`:
  PASS after the batched offset and asymmetric TurboQuant fix. Plain slot with
  a TQ neighbor matched the plain-solo reference exactly; two TQ slots also
  completed with coherent text. Log:
  `/tmp/vmlx_focus_ling_jangtq2_tq_b2_after_tq_asym_20260506.log`.

Osaurus implication: let vmlx own Ling cache lifecycle. If Ling regresses, fix
`BailingHybrid` / cache topology, not an osaurus-side prompt-size guard.

## Cache Stack

Recommended production coordinator shape:

```swift
let config = CacheCoordinatorConfig(
    usePagedCache: true,
    enableDiskCache: true,
    pagedBlockSize: 256,
    maxCacheBlocks: 1024,
    diskCacheMaxGB: 10,
    diskCacheDir: kvDir,
    ssmMaxEntries: 64,
    enableSSMReDerive: true,
    modelKey: modelKey,
    defaultKVMode: .none,
    defaultMaxKVSize: nil,
    longPromptMultiplier: 2.0
)
let coordinator = CacheCoordinator(config: config)
let engine = BatchEngine(context: context, maxBatchSize: 1, cacheCoordinator: coordinator)
```

Use `defaultKVMode: .turboQuant(keyBits:valueBits:)` only when the deployment
wants KV memory reduction and has validated the family. TurboQuant KV is a
correctness feature first; current live runs did not show decode-speed wins.

Tier behavior:

| Tier | Purpose | Notes |
|---|---|---|
| Paged L1 | Shared-prefix reuse for ordinary KV models | Exact block prefix hits. Unsafe partial hits roll back for VL/SSM. Disabled for DSV4 hybrid-pool caches. |
| Disk L2 | Session replay/restart and long-lived prefix cache | Uses TQDiskSerializer. DSV4 and SSM companion state serialize here. |
| SSM companion | Mamba/Arrays hidden-state sidecar | Stores prompt-boundary and block-boundary states so KV hits do not lose SSM recurrence state. Controlled by `enableSSMReDerive`; completion `.info` is yielded before store/re-derive, and detached async rederive is not used. |
| TurboQuant KV | Optional compressed KV cache | Batch path supports `.turboQuant(keyBits:valueBits:)`; disk round-trip covered by stability and batch probes. |

Cache semantics:

- Exact same prompt across coordinators should hit disk.
- Prefix extension on dense KV can hit paged L1 and prefill only suffix.
- Prefix extension on hybrid SSM or VL may report a coordinator hit but roll
  back to full prefill for correctness.
- DSV4 paged prefix hits are considered incompatible; disk/L2 serializer owns
  DSV4 cache restore.
- Cache store must use prompt-boundary snapshots, not post-decode mutated cache
  state, for prompt-only keys.

Relevant files:

- `Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift`
- `Libraries/MLXLMCommon/BatchEngine/BatchScheduler.swift`
- `Libraries/MLXLMCommon/Cache/TQDiskSerializer.swift`
- `Libraries/MLXLMCommon/Cache/SSMStateCache.swift`
- `Libraries/MLXLMCommon/Cache/SSMCompanionDiskStore.swift`
- `Libraries/MLXLMCommon/Cache/SSMReDerive.swift`

## TurboQuant / JANGTQ Notes

JANGTQ model weights and TurboQuant KV are separate concepts:

- JANGTQ weights: routed/expert codebook weight format used by model layers.
- TurboQuant KV: optional KV-cache quantization for runtime memory reduction.

Do not conflate the two in osaurus settings. A JANGTQ model can run with plain
KV, and a non-JANGTQ ordinary KV model can request TurboQuant KV if supported.

Current live correctness:

- Qwen batch cache hit: PASS.
- Qwen disk restore with SSM companion: PASS.
- Qwen `BENCH_BATCH_TQ_B2`: PASS, plain slot byte-identical with TQ neighbor.
- MiniMax M2.7 full-size paged prefix, disk L2, and `BENCH_BATCH_TQ_B2`:
  PASS in the focused rerun. This covers the full local JANGTQ bundle, not only
  MiniMax-small.
- Ling JANGTQ2 paged prefix and disk L2: PASS with SSM companion and hybrid
  rollback semantics. Ling `BENCH_BATCH_TQ_B2`: PASS after the per-slot
  `BatchArraysCache.offsetArray` and asymmetric TurboQuant K/V encoder-state
  fix. The previous `[broadcast_shapes] Shapes (1,32,24,128) and (192)` crash
  was the value path reusing a key-width TurboQuant state.
- MiniMax-small JP regression: PASS, 3 coherent turns and TQ disk round-trip.
- DSV4 disk restore: PASS at short budget with DSV4 nil-mask fix.

Current speed note:

- TurboQuant KV did not reproduce a decode-speed win in the tested Qwen and
  MiniMax-small settings. It increased TTFT and was neutral/slower for tok/s.
  Treat speed as future tuning, not a correctness gate.
- Ling MXFP4 is memory-correct after disabling oversized persistent gate/up
  fusion, but decode speed is much lower than JANGTQ2. Production should prefer
  Ling JANGTQ2 for serving unless MXFP4 is specifically required.

## Family Minimum Coherence Matrix

| Family/model | Minimum current status |
|---|---|
| DSV4 Flash JANGTQ | PASS for production chat row, reasoning row, and 5,568-token semantic long-context row. Much larger long-agent prompts remain memory-hardening work. |
| Ling/Bailing JANGTQ/JANGTQ2/MXFP4 | PASS for multi-turn recall after MLA no-double-update fix and live ArraysCache reuse. Paged/disk hybrid-cache rows pass for JANGTQ2, and TurboQuant KV B=2 now passes after the batched-offset/asymmetric-KV fix. |
| Qwen3.6 35B JANGTQ | PASS for thinking marker routing, tool-call multi-turn, paged prefix, disk restore, TQ B=2. Latest marker gate clean-exited with no raw `<think>` in `.chunk`. |
| Gemma4 26B JANG | PASS for harmony parser marker stripping and live perf/coherence smoke. Latest harmony gate clean-exited with no raw harmony markers in `.chunk`. |
| Laguna XS.2 JANGTQ | PASS for strict thinking-off/on loop probe. Both modes reached `stop`, produced visible content, and had no marker leak or loop. |
| MiniMax M2.7 JANGTQ | PASS for full-size ChatSession multi-turn coherence, compile off/on, clean exit. Speed remains open. MiniMax-small also passed JP regression and TQ disk round-trip. |
| Nemotron-Omni Nano JANGTQ | PASS for Omni matrix covering text, image, video, audio, media salt, hybrid SSM warm-pass, B=1/B=2. |
| ZAYA1 8B BF16/JANGTQ2/JANGTQ4/MXFP4 | PASS for factory load, forward smoke, CCA cache state, disk serializer round-trip, live exact disk restore, B=2 CCA state isolation, live B=2 raw decode, production chat-stream B=2, `zaya_xml` tool parser, and short live France decode. JANGTQ4/MXFP4 pass the generic favorite-color multi-turn recall after the default-thinking fix; JANGTQ2/BF16 are marker-clean but weak on that chat-quality probe. Paged-prefix, compiled decode, TurboQuant KV default, and reasoning-on/max rows remain gated follow-ups. |

Mistral 3.5 was requested, but no local model directory existed under
`~/models` during this pass.

## Toolchain Notes

Use Xcode's toolchain for Swift tests on this host:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --no-parallel
```

The Command Line Tools toolchain at `/Library/Developer/CommandLineTools` lacks
`xctest` here, so default `swift test` with CLT fails before it reaches package
logic. GPU-backed SwiftPM tests and live smokes also need MLX's default metallib
available beside the SwiftPM binaries. If it is missing:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild -downloadComponent MetalToolchain

(cd .build/checkouts/mlx-swift && \
  DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild build -scheme MLX -destination 'platform=macOS,arch=arm64')

cp ~/Library/Developer/Xcode/DerivedData/mlx-swift-*/Build/Products/Debug/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib \
  .build/arm64-apple-macosx/debug/default.metallib
cp .build/arm64-apple-macosx/debug/default.metallib \
  .build/arm64-apple-macosx/debug/mlx.metallib
cp .build/arm64-apple-macosx/debug/default.metallib \
  .build/arm64-apple-macosx/release/default.metallib
cp .build/arm64-apple-macosx/debug/default.metallib \
  .build/arm64-apple-macosx/release/mlx.metallib
```

Run MLX/GPU Swift tests serially with `--no-parallel`. A parallel mixed
Swift-Testing/XCTest run hit a Swift test helper signal 11 after individual
target tests had passed; the serial reruns were clean.

## Commands Agents Should Reuse

Build:

```bash
swift build -c release --product RunBench
```

Speed progress, batch vs simple:

```bash
BENCH_PERF=1 BENCH_PERF_PATH=batch BENCH_MAX_TOKENS=64 \
  BENCH_MODEL=~/models/dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK \
  .build/release/RunBench

BENCH_PERF=1 BENCH_PERF_PATH=iter BENCH_MAX_TOKENS=64 \
  BENCH_MODEL=~/models/dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK \
  .build/release/RunBench

BENCH_SIMPLE=1 BENCH_PROMPT_LEN=30 BENCH_MAX_TOKENS=64 \
  BENCH_MODEL=~/models/JANGQ/Ling-2.6-flash-JANGTQ \
  .build/release/RunBench

BENCH_PERF=1 BENCH_PERF_PATH=batch BENCH_MAX_TOKENS=64 \
  BENCH_MODEL=~/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK \
  .build/release/RunBench

BENCH_PERF=1 BENCH_PERF_PATH=iter BENCH_MAX_TOKENS=64 \
  BENCH_MODEL=~/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK \
  .build/release/RunBench

BENCH_PERF=1 BENCH_PERF_PATH=batch BENCH_MAX_TOKENS=96 \
  BENCH_MODEL=~/models/JANGQ/MiniMax-M2.7-JANGTQ \
  .build/release/RunBench

BENCH_PERF=1 BENCH_PERF_PATH=iter BENCH_MAX_TOKENS=96 \
  BENCH_MODEL=~/models/JANGQ/MiniMax-M2.7-JANGTQ \
  .build/release/RunBench

BENCH_PERF=1 BENCH_PERF_PATH=batch BENCH_MAX_TOKENS=96 \
  BENCH_MODEL=~/models/dealign.ai/Ling-2.6-flash-JANGTQ2-CRACK \
  .build/release/RunBench

BENCH_PERF=1 BENCH_PERF_PATH=iter BENCH_MAX_TOKENS=96 \
  BENCH_MODEL=~/models/dealign.ai/Ling-2.6-flash-JANGTQ2-CRACK \
  .build/release/RunBench

BENCH_PERF=1 BENCH_PERF_PATH=batch BENCH_MAX_TOKENS=96 \
  BENCH_MODEL=~/models/JANGQ/Laguna-XS.2-JANGTQ \
  .build/release/RunBench

BENCH_PERF=1 BENCH_PERF_PATH=iter BENCH_MAX_TOKENS=96 \
  BENCH_MODEL=~/models/JANGQ/Laguna-XS.2-JANGTQ \
  .build/release/RunBench

BENCH_PERF=1 BENCH_PERF_PATH=batch BENCH_MAX_TOKENS=64 \
  BENCH_MODEL=~/models/JANGQ/DeepSeek-V4-Flash-JANGTQ \
  .build/release/RunBench

BENCH_PERF=1 BENCH_PERF_PATH=iter BENCH_MAX_TOKENS=64 \
  BENCH_MODEL=~/models/JANGQ/DeepSeek-V4-Flash-JANGTQ \
  .build/release/RunBench
```

DSV4 production gates:

```bash
BENCH_DSV4_COHERENCE=1 BENCH_DSV4_ROW=chat \
  BENCH_MAX_TOKENS=128 \
  BENCH_MODEL=~/models/JANGQ/DeepSeek-V4-Flash-JANGTQ \
  .build/release/RunBench

BENCH_DSV4_COHERENCE=1 BENCH_DSV4_ROW=reasoning \
  BENCH_MAX_TOKENS=128 BENCH_DSV4_REASONING_MAX_TOKENS=384 \
  BENCH_MODEL=~/models/JANGQ/DeepSeek-V4-Flash-JANGTQ \
  .build/release/RunBench

BENCH_DSV4_COHERENCE=1 BENCH_DSV4_ROW=long \
  BENCH_MAX_TOKENS=128 BENCH_DSV4_LONG_REPEAT=220 \
  BENCH_DSV4_LONG_MAX_TOKENS=96 BENCH_DSV4_REASONING_MAX_TOKENS=384 \
  BENCH_MODEL=~/models/JANGQ/DeepSeek-V4-Flash-JANGTQ \
  .build/release/RunBench
```

Ling:

```bash
BENCH_COHERENT=1 BENCH_MAX_TOKENS=64 \
  BENCH_MODEL=~/models/JANGQ/Ling-2.6-flash-JANGTQ \
  .build/release/RunBench

BENCH_COHERENT=1 BENCH_MAX_TOKENS=96 \
  BENCH_MODEL=~/models/dealign.ai/Ling-2.6-flash-JANGTQ2-CRACK \
  .build/release/RunBench

BENCH_COHERENT=1 BENCH_MAX_TOKENS=96 \
  BENCH_MODEL=~/models/dealign.ai/Ling-2.6-flash-MXFP4-CRACK \
  .build/release/RunBench
```

MiniMax full-size:

```bash
BENCH_COHERENT=1 BENCH_MAX_TOKENS=64 \
  BENCH_MODEL=~/models/JANGQ/MiniMax-M2.7-JANGTQ \
  .build/release/RunBench
```

Qwen reasoning marker gate:

```bash
BENCH_QWEN_THINKING_CHECK=1 BENCH_MAX_TOKENS=64 \
  BENCH_MODEL=~/models/dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK \
  .build/release/RunBench
```

Gemma harmony gate:

```bash
BENCH_HARMONY_CHECK=1 BENCH_MAX_TOKENS=96 \
  BENCH_MODEL=~/models/dealign.ai/Gemma-4-26B-A4B-it-JANG_4M-CRACK \
  .build/release/RunBench
```

Qwen cache stack:

```bash
BENCH_BATCH_CACHE_HIT=1 BENCH_MAX_TOKENS=8 \
  BENCH_MODEL=~/models/dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK \
  .build/release/RunBench

BENCH_BATCH_DISK_RESTORE=1 BENCH_MAX_TOKENS=8 \
  BENCH_MODEL=~/models/dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK \
  .build/release/RunBench

BENCH_BATCH_TQ_B2=1 BENCH_MAX_TOKENS=8 \
  BENCH_MODEL=~/models/dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK \
  .build/release/RunBench
```

Laguna:

```bash
BENCH_LAGUNA_LOOP=1 BENCH_MAX_TOKENS=512 \
  BENCH_MODEL=~/models/JANGQ/Laguna-XS.2-JANGTQ \
  .build/release/RunBench
```

Omni:

```bash
BENCH_OMNI=1 BENCH_OMNI_BATCH=1 BENCH_MAX_TOKENS=24 \
  BENCH_MODEL=~/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK \
  .build/release/RunBench
```

## 2026-05-07 ZAYA and Ling Hook Integration Gate

The local stop hook now includes Ling and ZAYA explicitly. The hook file lives
under ignored `.claude/`, so this section is the tracked record to keep future
osaurus PR agents from relying on the older truncated model list.

Focused release tests:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
swift test -c release \
  --filter 'Zaya|BailingThinkingTemplateContextTests|BatchArraysCacheTests|BatchKVCacheWithTQSlotsTests' \
  --no-parallel
```

Result after the default-thinking processor fix: PASS. The focused run covered
39 Swift-Testing tests plus the 5 Bailing XCTest cases. Coverage includes
Bailing thinking-template context, batched ArraysCache offsets, asymmetric
TurboQuant KV slot caches, `ZayaCCACache` state/disk round-trip,
`BatchZayaCCACache` isolation, ZAYA config decode, `zaya_xml` parser mapping,
default ZAYA thinking-off template context, and real BF16/JANGTQ2/JANGTQ4/MXFP4
ZAYA load-forward smoke.

Fresh ZAYA JANGTQ2 gate rows:

| Row | Result |
|---|---|
| Contract | PASS. `cacheSubtype=zaya_cca`, `cacheType=hybrid`, 40 CCA layers, 40 MoE layers, 120 packed TQ groups, sidecar present, effective EOS 106. Log: `/tmp/vmlx_gate_zaya_contract_20260507.log`. |
| Template smoke | PASS for plain, thinking false/true, reasoning max, tools, multi-turn, and reasoning-history rows. Log: `/tmp/vmlx_gate_zaya_template_20260507.log`. |
| Live thinking-off decode | PASS. Answered `Paris.`, `stop=stop`, `unclosedReasoning=NO`, no loop/leak, 49.9 tok/s on the two-token tiny row. Log: `/tmp/vmlx_gate_zaya_perf_20260507.log`. |
| L2 disk restore | PASS. Fresh coordinator hit disk arrays, matched 140/140 prompt tokens, and completed both sessions. Log: `/tmp/vmlx_gate_zaya_disk_restore_20260507.log`. |
| Paged prefix cache | PASS as not applicable. ZAYA remains paged-incompatible because CCA `conv_state` and `prev_hs` are path-dependent. Log: `/tmp/vmlx_gate_zaya_cache_hit_20260507.log`. |

ZAYA variant refresh after the library-level default-thinking fix:

| Variant | Speed / graph row | Generic multi-turn chat row |
|---|---|---|
| BF16 source | PASS, 57.9 tok/s, `decodeNodes=9587`, `asType=1285`. Log: `/tmp/vmlx_gate_zaya_bf16_perf_20260507.log`. | Marker-clean after default-thinking fix, but still weak/mixed on favorite-color recall. Log: `/tmp/vmlx_gate_zaya_bf16_coherent_after_default_20260507.log`. |
| JANGTQ2 | PASS, 49.9 tok/s tiny row. Log: `/tmp/vmlx_gate_zaya_perf_20260507.log`. | Marker-clean after default-thinking fix, but weak on favorite-color recall. Treat as runtime/cache-ready and model-quality follow-up. Log: `/tmp/vmlx_gate_zaya_jangtq2_coherent_after_default_20260507.log`. |
| JANGTQ4 | PASS, 54.7 tok/s, `decodeNodes=8831`, `asType=1365`. Log: `/tmp/vmlx_gate_zaya_jangtq4_perf_20260507.log`. | PASS. Recalled blue/cool with visible chunks and no thinking marker leak. Log: `/tmp/vmlx_gate_zaya_jangtq4_coherent_after_default_20260507.log`. |
| MXFP4 | PASS, 66.8 tok/s, `decodeNodes=8791`, `asType=1285`. Log: `/tmp/vmlx_gate_zaya_mxfp4_perf_20260507.log`. | PASS. Recalled blue/cool with visible chunks and no thinking marker leak. Log: `/tmp/vmlx_gate_zaya_mxfp4_coherent_after_default_20260507.log`. |

BatchEngine chat-stream rows after the same default-context fix:

- JANGTQ2: no crash, no marker leak, raw-transcript style harness remained
  weak on recall. Log:
  `/tmp/vmlx_gate_zaya_jangtq2_batch_chat_after_default_20260507.log`.
- JANGTQ4: no crash, no marker leak, recovered blue/cool by turn 3. Log:
  `/tmp/vmlx_gate_zaya_jangtq4_batch_chat_after_default_20260507.log`.
- MXFP4: no crash, no marker leak, blue/cool recall acceptable. Log:
  `/tmp/vmlx_gate_zaya_mxfp4_batch_chat_after_default_20260507.log`.

Fresh Ling JANGTQ2 gate rows:

| Row | Result |
|---|---|
| Perf/graph | PASS. BatchEngine median 29.8 tok/s, best 29.9, `decodeNodes=5`, `asType=3`, coherent preview, no loop/leak. Log: `/tmp/vmlx_gate_ling_perf_20260507.log`. |
| Multi-turn coherence | PASS compile off/on. The model recalled the favorite color and answered that blue is cool. Log: `/tmp/vmlx_gate_ling_coherent_20260507.log`. |
| Paged-prefix cache | PASS. Coordinator hit paged tier, matched 128/166 prompt tokens, then completed the warm turn. Log: `/tmp/vmlx_gate_ling_cache_hit_20260507.log`. |
| L2 disk restore | PASS. Fresh coordinator hit disk arrays, matched 143/143 prompt tokens, and completed both sessions. Log: `/tmp/vmlx_gate_ling_disk_restore_20260507.log`. |
| TurboQuant KV B=2 | PASS. Plain KV slot stayed identical beside a TurboQuant neighbor; both TQ slots completed. Log: `/tmp/vmlx_gate_ling_tq_b2_20260507.log`. |

Additional Ling variants:

| Variant | Result |
|---|---|
| Legacy JANGTQ | PASS perf/graph at 28.6 tok/s, `decodeNodes=5`, `asType=3`; multi-turn recall passed compile off/on. Logs: `/tmp/vmlx_gate_ling_legacy_perf_20260507.log`, `/tmp/vmlx_gate_ling_legacy_coherent_20260507.log`. |
| MXFP4 | PASS perf/graph at 9.2 tok/s, `decodeNodes=5`, `asType=3`; multi-turn recall passed compile off/on. Logs: `/tmp/vmlx_gate_ling_mxfp4_perf_20260507.log`, `/tmp/vmlx_gate_ling_mxfp4_coherent_20260507.log`. |

Current cache and batching gate refresh:

| Row | Result |
|---|---|
| Qwen3.5 35B TurboQuant KV B=2 | PASS. Plain slot stayed byte-identical beside TQ(4,4); both TQ slots completed. Log: `/tmp/vmlx_gate_qwen35_tq_b2_20260507.log`. |
| Gemma4 E2B TurboQuant KV B=2 | PASS. Plain slot stayed byte-identical beside TQ(4,4); both TQ slots completed. Log: `/tmp/vmlx_gate_gemma4_e2b_tq_b2_20260507.log`. |
| Qwen3.5 35B B=4 | PASS. Slot 0 stayed byte-identical to solo; 4 slots completed 16 tokens; batched/serial ratio 0.33. Log: `/tmp/vmlx_gate_qwen35_b4_20260507.log`. |
| Gemma4 E2B B=4 | PASS for cross-slot isolation. Slot 0 stayed byte-identical to solo; throughput assertion was skipped because stop lengths were intentionally uneven. Log: `/tmp/vmlx_gate_gemma4_e2b_b4_20260507.log`. |

Current multi-turn chat refresh:

| Family | Result |
|---|---|
| Qwen3.5 35B | Cache/history tracked in reasoning; visible answer stayed empty at the 64-token budget because the model remained in reasoning. No raw marker leak. Log: `/tmp/vmlx_gate_qwen35_coherent_20260507.log`. |
| Gemma4 E2B/E4B/26B | PASS. Visible blue/cool recall with compile off/on. Logs: `/tmp/vmlx_gate_gemma4_e2b_coherent_20260507.log`, `/tmp/vmlx_gate_gemma4_e4b_coherent_20260507.log`, `/tmp/vmlx_gate_gemma4_26b_coherent_20260507.log`. |
| Nemotron Cascade | Reasoning captured blue/cool and visible output included the final answer. Log: `/tmp/vmlx_gate_nemotron_cascade_coherent_20260507.log`. |
| Nemotron Super | Reasoning captured blue/cool but visible chunks stayed empty at the 64-token budget. Treat as reasoning-heavy budget behavior, not a cache failure. Log: `/tmp/vmlx_gate_nemotron_super_coherent_20260507.log`. |

Production interpretation:

- Ling is functionally ready for Osaurus runtime wiring with the current
  BailingHybrid cache topology. Speed is still below target, but the cache
  stack now has live coverage for prefix, disk, and TurboQuant KV B=2.
- ZAYA runtime/cache integration is ready only with thinking off. Prefer
  JANGTQ4 or MXFP4 for production chat-quality validation today; JANGTQ2 is
  load/cache/decode-ready but weak on the generic favorite-color recall probe.
  Keep `supports_thinking=false`, keep tools through `zaya_xml`, and keep
  paged-prefix/compiled-decode/runtime TurboQuant KV defaults off until CCA
  parity is proven for those tiers.
- Do not add osaurus app-layer guards for either family. If a future failure is
  in MLX kernels, mlx-swift API behavior, Swift Jinja template rendering, model
  metadata, or JANG conversion output, fix that owning osaurus repo or model
  artifact directly.

## 2026-05-07 VL / Omni Cache Follow-Up

Runtime changes:

- `LMInput.hasMediaContent` is now the shared media-topology predicate for
  image, video, and audio prompts.
- `BatchEngine` uses that predicate for cache-hit rollback instead of checking
  only image/video. Partial prefix hits that would split an image/video/audio
  placeholder span fall back to full prefill.
- `TokenIterator` now mirrors the same media/SSM rollback policy on
  coordinator hits. This closes the older gap where the iterator could restore
  a media or hybrid-SSM prefix and then feed only remaining text tokens.
- `UserInput` convenience initializers now preserve audio on the wrapped
  `Chat.Message`, expose audio for raw-message inputs, and extract media when
  callers pass a prebuilt `.chat` prompt enum.
- `NemotronHOmniProcessor` now strips one-token Qwen-style image/video parts
  before injecting expanded NVLM placeholders, so structured chats do not get
  duplicate media markers. Existing text content is preserved when media is
  prepended.
- `RunBench` has a new `BENCH_VL_CHAT_CACHE=1` row that exercises
  `UserInput(chat:)`, same-media cache hits, different-media misses, and a
  follow-up turn through `BatchEngine`.

Live VL-capable local model results:

| Bundle | Result |
|---|---|
| Nemotron-Omni-Nano-JANGTQ | PASS structured VL chat/cache. Same image hit paged 293/293, different image missed, follow-up emitted visible text with no raw media/reasoning markers. Log: `/tmp/vmlx_vl_chat_cache_omni_jangtq_20260507.log`. |
| Nemotron-Omni-Nano-JANGTQ | PASS full Omni matrix 17/17. Covered text, image, video encoder, audio encoder, video/audio LMInput, reasoning toggle, mixed image+audio, audio media-salt isolation, hybrid SSM warm-pass, and BatchEngine text/image/audio rows. Log: `/tmp/vmlx_omni_matrix_jangtq_20260507.log`. |
| Nemotron-Omni-Nano-JANGTQ4 | PASS structured VL chat/cache. Same image hit paged 293/293, different image missed, follow-up emitted visible text with no raw media/reasoning markers. Log: `/tmp/vmlx_vl_chat_cache_omni_jangtq4_20260507.log`. |
| Nemotron-Omni-Nano-MXFP4 | PASS structured VL chat/cache. Same image hit paged 293/293, different image missed, follow-up emitted visible text with no raw media/reasoning markers. Log: `/tmp/vmlx_vl_chat_cache_omni_mxfp4_20260507.log`. |

Verification:

```bash
swift build -c release --product RunBench

BENCH_VL_CHAT_CACHE=1 BENCH_MAX_TOKENS=16 \
  BENCH_MODEL=~/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK \
  .build/release/RunBench

BENCH_OMNI=1 BENCH_OMNI_BATCH=1 BENCH_MAX_TOKENS=16 \
  BENCH_MODEL=~/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK \
  .build/release/RunBench
```

SwiftPM XCTest note: this host currently uses Command Line Tools Swift at
`/Library/Developer/CommandLineTools/usr/bin/swift`, and `swift -e 'import
XCTest'` fails with `no such module 'XCTest'`. Because of that local toolchain
state, the runtime/library changes were compile-checked by `RunBench` build
and validated through live model benches. The focused XCTest rows themselves
did not run on this host.

## Open Risks

- DSV4 passed the 5,568-token semantic long-context row cleanly on this host,
  but the 650-filler stress row OOMed. Long DSV4 memory hardening is still
  needed for very long production agent loops.
- DSV4 thinking-enabled short-budget turns can still finish with useful text in
  `.reasoning` and empty visible content. Osaurus must expose/handle
  reasoning-only or length finishes.
- MiniMax M2.7 full-size speed improved after removing default whole-SwitchGLU
  compile, but is still below the expected 45-50 tok/s band in the latest live
  run. This is a speed/model-profile task, not a minimum coherence blocker.
- Ling MXFP4 speed is now low (~6 tok/s in BatchEngine perf) because the
  previous faster path kept a second fused gate/up expert bank resident and
  pushed footprint to ~110 GB. Reintroduce speed only with a memory-safe fused
  or streamed MXFP4 path.
- Distributed XCTest targets are opt-in with
  `VMLINUX_ENABLE_DISTRIBUTED_TESTS=1 swift test`; default `swift test` covers
  the active local runtime package surface.
- Factory fallback tracing is opt-in via `VMLINUX_MODEL_FACTORY_TRACE=1`.

## License

Repo-level license is MIT. The root `LICENSE` keeps the upstream
`2024 ml-explore` notice and now also includes `2026 Osaurus contributors` for
the local vmlx-swift-lm additions. Do not remove upstream copyright notices when
syncing or pushing.

## What Not To Do

- Do not add osaurus-side prefill guards for Ling.
- Do not reset ArraysCache per Ling chat turn.
- Do not parse reasoning/tool-call markup in osaurus from visible chunks.
- Do not use raw transcript prompts to certify DSV4 chat-template behavior.
- Do not enable DSV4 `DSV4_KV_MODE=full|tq` as a silent default.
- Do not treat JANGTQ weight format as the same setting as TurboQuant KV cache.
- Do not require JangPress for this phase. JangPress can be wired later as a
  memory/residency policy after runtime coherence is stable.
