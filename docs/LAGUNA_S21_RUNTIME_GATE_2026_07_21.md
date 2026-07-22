# Laguna S 2.1 vMLX Swift runtime gate (2026-07-21)

Status: **PARTIAL — focused source tests and direct actual-weight 2L/4M
cache/tool/reasoning runs have evidence; no Osaurus Release UI row is accepted
yet, and 4M has not emitted a non-empty reasoning delta in the thinking-on
probe.**

This lane targets the local released text-only bundles:

- `/Users/eric/models/JANGQ-AI/Laguna-S-2.1-JANG_2L`
- `/Users/eric/models/JANGQ-AI/Laguna-S-2.1-JANG_4M`

The authoritative port contract is
`/Users/eric/jang/docs/runtime/laguna-s21/NOTES.md`, with the Swift-specific
ordering in `SWIFT.md`. MXFP4 is not part of this gate.

## Current bundle truth

- 48 layers: 12 full attention / 36 sliding-window attention.
- Full layers use 48 query heads, YaRN theta 500k, factor 128, and 64/128
  rotary dimensions. SWA layers use 72 query heads, theta 10k, full rotary,
  and window 512.
- Layer 0 is dense. Layers 1...47 use 256 routed experts, top 10, plus one
  shared expert.
- Router math is sigmoid in fp32, correction bias for selection only,
  normalized unbiased selected weights, routed contribution times 2.5, shared
  contribution unscaled.
- `g_proj` is per-head and uses fp32 softplus.
- Activation stream must be bfloat16. Float16 overflows near layer 46 and the
  characteristic failure is repeated unknown tokens.
- EOS is `[2, 24]`; the chat template owns BOS token 2.
- Vendor thinking default is on through
  `generation_config.default_chat_template_kwargs.enable_thinking=true`, also
  mirrored in `jang_config.chat.template_kwargs_defaults`.
- Tool parser is `glm47` (Swift `.glm4`); reasoning parser is `deepseek_r1`.
- JANG_2L is affine mixed 2/3/6/8-bit. JANG_4M is affine mixed 4/6/8-bit.
- Full attention owns ordinary growing KV state. SWA owns rotating KV state
  with `maxSize=512, keep=0`. TurboQuant KV is optional and may compress only
  the compatible full-attention KV component.

## Root causes found in the pre-fix Swift source

1. `LagunaMoE` registered its routed expert leaf under `mlp.experts`, while
   the released S-2.1 tensors and per-module quantization declarations use
   `mlp.switch_mlp`. This prevents the production loader from pairing the
   released weights with the model leaves.
2. `LagunaModel.callAsFunction` returned `.none` for the SWA mask whenever
   `cache == nil`. Prompts longer than 512 therefore let SWA layers attend the
   full prefix on the no-cache path.
3. Default no-TurboQuant cache construction used a full-attention rotating
   cache with `keep=4`. The S-2.1 reference has no attention sinks; full
   attention grows normally and SWA alone rotates with `keep=0`.
4. `generation_config.default_chat_template_kwargs` and
   `jang_config.chat.template_kwargs_defaults` were not decoded or passed to
   tokenizer rendering. A silent request therefore selected the Jinja
   fallback (thinking off) instead of the bundle's declared default (on).
5. The tokenizer bridge deliberately replaces Laguna's unresolved
   `{% include 'chat_template.jinja' %}` with `lagunaMinimal`, but that Swift
   fallback had drifted from S-2.1: it omitted Poolside's default system
   message, changed trained turn whitespace, exposed stored reasoning while
   thinking was off, and injected required-tool instructions absent from the
   bundle template.

## Candidate source changes

- Make `switch_mlp` the canonical routed-expert module path. Normalize legacy
  `experts.*` JANGTQ payloads during Laguna sanitization.
- Build the SWA band mask for no-cache and cached prefills through one tested
  production helper.
- Construct `KVCacheSimple` for full layers and explicit
  `RotatingKVCache(maxSize: 512, keep: 0)` for SWA.
- Decode typed chat-template defaults from both bundle metadata files. Apply
  generation config first, JANG mirror second, and let request/UI context
  overwrite either value. `supports_thinking=false` remains authoritative.
- Align the Laguna-only Swift-Jinja fallback to the released S-2.1 template
  contract while retaining exactly one template-owned BOS and removing only
  the unsupported `{% generation %}` wrapper. Do not add synthetic tool or
  sampler instructions.

## Current evidence

- `/tmp/vmlx-laguna-model-tests-20260721.log`: 10 Laguna configuration,
  sanitization, mask, cache-topology, quant-declaration, and thinking-default
  tests executed and passed after the Metal test library was installed beside
  the XCTest runner.
- `/tmp/vmlx-laguna-source-tests-20260721.log`: 14 XCTest plus 16
  Swift-Testing cases executed and passed after the fallback was aligned.
- `/tmp/vmlx-laguna-2l-config-smoke-20260721.log`: actual JANG_2L metadata
  resolved `model_type=laguna`, 48 layers, effective EOS `[2,24]`, tokenizer
  BOS/EOS id 2, affine `JANG_2L`, and no JANGTQ tensors.
- `/tmp/vmlx-laguna-2l-template-smoke-20260721.log`: actual JANG_2L tokenizer
  exercised plain, explicit thinking off/on, two- and nine-tool schemas,
  multi-turn thinking-off history, and reasoning history. Every row passed;
  rendered prompts contain one leading `〈|EOS|〉` and the expected Poolside
  assistant rail.
- `/tmp/vmlx-laguna-focused-current-v2-20260721.log`: the current-source
  focused gate executed 15 XCTest cases and 102 Swift Testing cases with zero
  failures. This includes Laguna module/mask/template contracts, non-aligned
  partial-leaf lookup, paged eviction to disk, mixed TQ+rotating restore,
  ZAYA CCA restore, and other hybrid companion regressions.
- `/tmp/vmlx-laguna-release-build-current-20260721.log`: the current source
  builds the Release `RunBench` product successfully. The three emitted
  unhandled-file warnings pre-exist this lane and are recorded rather than
  treated as test execution.
- Default KV, paged RAM off, SSD L2 on:
  `/tmp/vmlx-laguna-2l-defaultkv-safe-seed-{cold,restart}-20260721.log` and
  `/tmp/vmlx-laguna-4m-defaultkv-safe-seed-{cold,restart}-20260721.log` show
  fresh-process disk matches at 607/608 prompt tokens and coherent
  `vmlx-cache-green` output. 2L prompt processing changed 3.302s cold to
  1.426s after restart; 4M changed 6.983s to 3.646s. TurboQuant compression
  remained zero in these default rows.
- Explicit TQ KV, paged RAM off, SSD L2 on:
  `/tmp/vmlx-laguna-2l-tq44-safe-seed-{cold-v2,restart-v2}-20260721.log` and
  `/tmp/vmlx-laguna-4m-tq44-safe-seed-{cold,restart}-20260721.log` show only
  the 12 full-KV layers transitioning to TurboQuant while all 36 SWA layers
  remain rotating. Fresh processes restore 607/608 tokens from disk and the
  output remains coherent.
- Explicit TQ KV with paged RAM on:
  `/tmp/vmlx-laguna-{2l,4m}-tq44-paged-hot-20260721.log` records a paged
  612/635 partial hit. The paired `paged-evict` logs force 13 evictions and
  then record a disk 612/635 fallback, demonstrating RAM-hot / SSD-warm tier
  order in the direct runtime.
- `/tmp/vmlx-laguna-2l-reasoning-matrix-20260721.log`: direct 2L weight run
  exercised thinking off/low/max. Off emitted zero reasoning; both on rows
  emitted 1,042 reasoning characters, kept visible content separate, and had
  344-353ms warm TTFT.
- `/tmp/vmlx-laguna-4m-reasoning-matrix-20260721.log` did not satisfy the
  strict reasoning-on gate because both on rows emitted zero reasoning.
  `/tmp/vmlx-laguna-4m-think-raw-route-20260721.log` shows a clean immediate
  reasoning close followed by content with no marker leakage, so this is not
  evidence of a content-delta routing loss, but non-empty 4M reasoning remains
  unproven.
- `/tmp/vmlx-laguna-{2l,4m}-toolcall-current-20260721.log` contains parsed
  `get_weather({"location":"Tokyo"})` calls without raw protocol markers.
  `/tmp/vmlx-laguna-2l-agentic-tool-v2-20260721.log` and
  `/tmp/vmlx-laguna-4m-agentic-tool-20260721.log` contain parsed tool calls,
  supplied-result continuations, and later `LAGUNA-77` recall. These are
  direct runtime rows, not Osaurus UI proof.
- Direct cache-store budget telemetry during the actual-weight rows reported
  roughly 68-70 GB process physical footprint for both bundles. This is not
  being promoted as a low-memory pass; the isolated app must show the resident
  process in Activity Monitor and the app memory UI before the row can close.
- `/tmp/vmlx-laguna-full-swift-test-20260721.log` is not a passing full-suite
  artifact. The XCTest phase progressed, but the Swift Testing phase stopped
  making progress after starting concurrent MLX tests. A process sample at
  `/tmp/vmlx-laguna-full-test-sample-20260721.txt` shows a lock-order deadlock:
  legacy `lockSerializedMLXTest()` owns `mlx.metal.test.serializer` while
  waiting for its token, and `MLXMetalTestLock` owns the process-wide POSIX
  semaphore while waiting for that same dispatch queue. The run was
  terminated and must not be counted as green.

## Required proof matrix

No row may move to verified without its named artifact and measured output.

| Row | Required evidence | Status |
|---|---|---|
| Focused config/module/mask/cache/parser tests | Test log with actual pass markers | VERIFIED-SOURCE (`/tmp/vmlx-laguna-model-tests-20260721.log`, `/tmp/vmlx-laguna-source-tests-20260721.log`) |
| Actual 2L config + tokenizer/template contract | Local bundle metadata and rendered prompt rows | VERIFIED-RUNTIME-NO-WEIGHTS (`/tmp/vmlx-laguna-2l-config-smoke-20260721.log`, `/tmp/vmlx-laguna-2l-template-smoke-20260721.log`) |
| Full Swift test/build regression gate | Complete test/build log | BLOCKED — existing test-lock inversion; focused current-source gate passes, full run was terminated (`/tmp/vmlx-laguna-full-swift-test-20260721.log`, `/tmp/vmlx-laguna-full-test-sample-20260721.txt`) |
| 2L no-cache short smoke | Coherent text, no unknown-token spam, token/s, physical footprint | OPEN |
| 2L cached short smoke | Coherent text, token/s, cache topology | VERIFIED-DIRECT; UI/footprint pending |
| 2L >512 teacher-forced cache parity | Pre/post-window agreement, not only greedy text equality | OPEN |
| 2L multi-turn chat, thinking default/on/off | Visible reasoning/content separation and token/s | VERIFIED-DIRECT; UI pending |
| 2L GLM tool call and post-tool continuation | Parsed call, clean deltas, coherent continuation | VERIFIED-DIRECT; UI pending |
| 2L SSD L2 cold/store/restart/partial restore | Counters, restored token count, TTFT/prefill | VERIFIED-DIRECT; UI pending |
| 2L paged RAM off/on | Effective setting and cache telemetry | VERIFIED-DIRECT; UI pending |
| 2L TurboQuant KV off/on | Only compatible full KV encoded; SWA remains rotating | VERIFIED-DIRECT; UI pending |
| 4M repeat of correctness/cache/tool rows | Same evidence plus token/s/footprint | PARTIAL — cache/tool direct rows pass; non-empty thinking-on reasoning and UI/footprint pending |
| Isolated Release Osaurus build and pin | Exact vMLX revision in resolved graph | OPEN |
| Live Osaurus UI settings/model/chat/tool matrix | Computer Use screenshots/telemetry/log artifacts | OPEN |

Expected reference throughput on the local 128 GB M5 Max is approximately
48.3 token/s for 2L and 32.6 token/s for 4M with the wired-memory policy in
effect. Those are Python reference measurements, not Swift results. Swift
numbers will be recorded rather than assumed.

## Regression questions to answer before promotion

- Does canonicalizing `switch_mlp` preserve legacy Laguna JANGTQ fused expert
  payloads and their per-projection bit widths?
- Does the banded no-cache mask agree with cached prefill across token 512?
- Does `keep=0` remove boundary divergence without changing full-attention
  history?
- Do generation-config defaults reach ordinary chat, agent/tool continuation,
  and spawned/delegated requests, while explicit UI on/off wins every turn?
- Does EOS 24 stop both ordinary answers and post-tool continuations without
  leaking protocol markers?
- Does bfloat16 remain the effective activation/quant-metadata stream through
  layer 46 on both 2L and 4M?
- With paged RAM cache off, can disk L2 restore full and partial stable prompt
  blocks after a new chat and process restart?
- With paged RAM cache on, does RAM act as hot tier and SSD as warm fallback?
- With TurboQuant KV on, are only the 12 full-attention KV layers compressed,
  with the 36 SWA layers and disk restore remaining coherent?
- Does the wired-memory setting improve decode without violating Osaurus RAM
  safety or changing bundle generation parameters?
