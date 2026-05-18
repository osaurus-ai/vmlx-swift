# vmlx-swift-lm ↔ osaurus integration notes

**Link this file** from osaurus issues / Discord / PRs. Standalone, short, directly addresses the two referenced osaurus docs.

**Looking for the full public-API surface?** See `OSAURUS-API-SURFACE.md` in this directory — per-symbol reference with shape + which osaurus file consumes it, cross-checked against osaurus `main` and PR #893.

## Doc map

| Doc | For |
|---|---|
| `OSAURUS-INTEGRATION.md` (this file) | Narrative overview — what osaurus flipped / closed / is blocked on. |
| **`TPAE-2026-04-20-TRIAGE.md`** | **Start here if you're reading tpae's Slack thread.** Line-by-line map from every message (1:58 AM through 3:17 AM) to its resolution — commit, doc, real-model verification row. |
| `OSAURUS-API-SURFACE.md` | Per-symbol public API surface. The canonical reference osaurus integrators link against. |
| **`docs/OSAURUS-INTEGRATION-HANDOFF-2026-05-09.md`** | **Current production-runtime handoff for osaurus agents** (supersedes the 2026-05-06 doc below). Pin-bump roadmap, cache-scope salt, generation_config defaults, ZAYA reasoning split, Hy3 native runtime, ZAYA1-VL native/cache status, JANGTQ top-k override plumbing. Read this first. |
| `OSAURUS-RUNTIME-HANDOFF-2026-05-06.md` | **DEPRECATED 2026-05-10** — kept for historical context (DSV4/Ling/Bailing notes, the 2026-05-07 ZAYA "served as non-reasoning" framing that has since been corrected). Current ZAYA wiring: reasoning-capable, default `enable_thinking=false`, honors explicit `disableThinking=false` opt-in. |
| `OSAURUS-SPECDEC.md` (one level up) | `GenerateParameters.draftStrategy` contract — DFlash / DDTree speculative decoding. |
| `GEMMA4-SLIDING-WINDOW-CRASH.md` | 2026-04-20 fix for tpae's `broadcast_shapes` crash on Gemma-4 at prompts past `sliding_window=1024`. Real-model verification matrix included. |
| `REASONING-STREAM-EVENT.md` | 2026-04-20 `Generation.reasoning(String)` library-level streaming channel. Closes tpae's "thinking parsers should be handled at library level". Updated 2026-04-20 PM with harmony (Gemma-4) + `startInReasoning` (Qwen 3.x enable_thinking prefill) support. |
| `STOP-SEQUENCES-CONTRACT.md` | 2026-04-20 `GenerateParameters.extraStopStrings` field + `StopStringMatcher`. Closes tpae's "what should happen to text-level stop sequences". |
| **`OMNI-OSAURUS-HOOKUP.md`** | **2026-04-29 Nemotron-3-Nano-Omni multimodal integration guide.** Bundle detection, four-tower wrapper (LLM + RADIO + Parakeet + projectors), image/video/audio splice contracts, hybrid Mamba/Attn cache topology, TQ KV interaction, EVS pruning, JANGTQ × all-three-quants routing, cross-VLM `VLMVideoUtils.swift` shared library, full audio-LMInput integration, API endpoint hand-off contract, hybrid-SSM warm-pass parity, full migration checklist. **Read this first** if osaurus is wiring any omni bundle. |
| **`OMNI-VOICE-INTEGRATION.md`** | **2026-04-29 voice/audio/video product-engineering guide for osaurus app + UI workers.** How to wire `NemotronHOmniMicRecorder` (live mic) + `NemotronHOmniSpeaker` (system TTS) + drag-drop file UX + push-to-talk + voice-mode UX patterns into the chat app. RADIO + Parakeet + mel STFT + NVLM tile reference for engineers doing custom work on the encoders. Honest scope boundary on neural TTS (not in bundle — BYOM). Read this for **product-level** voice/audio/video features beyond the API surface contract. |
| `FORK-SYNC-PROCESS.md` | 2026-04-20 upstream-sync procedure (`ml-explore/mlx-swift-lm` → `osaurus-ai/vmlx-swift-lm`). **Note:** `osaurus-ai/mlx-swift-lm` is deprecated — osaurus consumes vmlx directly. Closes tpae's "are we keeping this up to date". |
| `BATCH_ENGINE.md` (next to these) | Internal iter log — architecture decisions, per-iter rationale, the ~2100-line deep dive. |

**Status** (2026-04-20): **production-ready**. All four tpae-reported issues fixed, documented, unit-tested, AND real-model-verified against the actual crashing `gemma-4-26b-a4b-it` architecture at tpae's exact 2152 → 3715 → 7869 → 8362 prompt progression. See `GEMMA4-SLIDING-WINDOW-CRASH.md` §"Real-model verification" for the run-by-run numbers.

**Iter 66 closes tpae's tool-call-parsing request** — `BatchEngine.generate()` + `Evaluate.generate()` now emit authoritative `.toolCall(ToolCall)` events for every supported family (JSON, Qwen xml_function, Qwen 3.6 interleaved thinking, Mistral, GLM-4, LFM2, Kimi K2, Gemma-3/4, MiniMax M2). Osaurus no longer needs its own tool-call parser at the app layer. See §4 below.

---

## Addressing tpae's two references

### 1. `osaurus/docs/INFERENCE_RUNTIME.md`

| INFERENCE_RUNTIME.md concern | vmlx-swift-lm resolution |
|---|---|
| "`mlxBatchEngine` — default OFF until blockers close" | All `BatchEnginePlan.openBlockers` closed — see §2 below. Ready to flip default ON. |
| "`mlxBatchEngineMaxBatchSize` default 4" | Verified at B=4 AND B=8: slot 0 byte-identical to solo-B=1 reference on Qwen3-0.6B. |
| "`mlxAllowConcurrentStreams` — caution, MLX-vs-MLX unvalidated" | vmlx-side thread-safety covered: `DiskCache.store` now locks across `MLX.eval` + `save` + SQLite (iter 61 fix). Coordinator paged cache + SSM state + setHybrid flag all thread-safe under 128-way concurrent fuzz. MLX-vs-MLX at the Metal layer is still osaurus's operational call per its own doc. |
| "`cooperativeYield` — StreamAccumulator yield between tokens" | Osaurus-side only. vmlx emits tokens via `AsyncStream`; osaurus's `StreamAccumulator` chooses yield policy. |
| Freeze / Metal crash classes | Nine production bug classes fixed — listed in §3 below. |

### 2. `osaurus/Packages/OsaurusCore/Services/ModelRuntime/BatchEnginePlan.swift`

Both `Blocker` cases closed upstream:

```swift
public enum Blocker: String, CaseIterable {
    case kvQuantization    // CLOSED by vmlx iter 17 (Stage 0 TurboQuant)
    case compileSupport    // CLOSED by vmlx iters 9-22 (Stages 1A-5 compile path)
}
```

**Proposed osaurus change**: `public static var openBlockers: [Blocker] { [] }`.

**Why closed**:

- **kvQuantization** → `Libraries/MLXLMCommon/BatchEngine/BatchQuantize.swift` runs `wrapNewCacheIfNeeded` at admission + `maybeCompress` post-prefill. Supported `kvMode: .turboQuant(keyBits:, valueBits:)`. Legacy affine (`kvBits`, `kvMode: .affine`) deliberately not supported under batch — requires quantized-tuple attention sites out of Stage-0 scope; affine requests run with float KV and log a warning. Verified by `BENCH_BATCH_TQ_B2`: slot 0 plain with a concurrent TQ(4,4) neighbour is byte-identical to solo reference (no cross-slot contamination).
- **compileSupport** → `Libraries/MLXLMCommon/BatchEngine/BatchCompile.swift` classifies per-slot cache topology (`.simple` / `.turboQuant` / `.rotating` / `.cacheList` / `.mamba` / `.heterogeneous`); eligible families promote via `maybePromoteToCompiledDecode` at prefill end. Heterogeneous (Gemma-4 SWA, Qwen3.5-MoE mix), `.mamba`, and model families with known compiled-trace parity failures fall through to uncompiled — correct-by-design. Verified by `CompilableKVCache|TurboQuant|Rotating|CacheList|Mamba` probe suites at 5e-7 abs diff. `BENCH_BATCH_CHAT` compile ON ≡ compile OFF remains a required family gate; Laguna is currently denied compiled decode after a current-head live row exposed a compile-on loop.

---

## What osaurus integrators need to know

### Load-time env-var shims (opt-in, zero impact when unset)

- **`VMLX_CHAT_TEMPLATE_OVERRIDE=/path/to/template.jinja`** — tokenizer bridge substitutes the shipped `chat_template.jinja` with this file's contents. Needed for **Gemma-4** because its native template trips a swift-jinja 1.3.0 interaction bug. Ship two compatible templates:
  - `Libraries/MLXLMCommon/ChatTemplates/Gemma4Minimal.jinja` — text + image/video/audio content parts
  - `Libraries/MLXLMCommon/ChatTemplates/Gemma4WithTools.jinja` — adds `tool_calls` + `tool_responses`
- **`VMLX_TOKENIZER_CLASS_OVERRIDE=Qwen2Tokenizer`** — auto-rewrites `tokenizer_class` at load. Default map includes `TokenizersBackend` → `Qwen2Tokenizer` (unblocks `mlx-community/Qwen3.5-VL-9B-8bit`).

### Auto-detected behaviour (no env var needed)

- **`CacheCoordinator.isHybrid` auto-flip**: when the first slot's cache contains a Mamba/SSM layer, `BatchEngine.admitPendingRequests` calls `coordinator.setHybrid(true)` automatically. Osaurus no longer needs to remember per-model.
- **Coordinator-owned KV sizing** (2026-04-21 contract, see `KV-SIZING-CONTRACT.md`): when `CacheCoordinatorConfig.defaultKVMode` / `.defaultMaxKVSize` are set, `BatchEngine.admitPendingRequests` fills those values into any request that submitted `kvMode: .none` / `maxKVSize: nil`. Explicit per-request values always win. Recommended osaurus config for memory-bounded inference across arbitrary prompt lengths:
  ```swift
  CacheCoordinatorConfig(
      usePagedCache: true,
      enableDiskCache: true,
      modelKey: modelID,
      defaultKVMode: .turboQuant(keyBits: 3, valueBits: 3),  // ~5× KV savings
      defaultMaxKVSize: 8192,                                 // 8K ring window
      longPromptMultiplier: 2.0                               // >16K → cap
  )
  ```
  This closes the gap that ferebee's 55K-token translation crash exposed: osaurus 0.17.0 removed its per-request `maxKVSize` UI knob assuming vmlx would own KV sizing; now vmlx actually does.
- **JANG weights-only tokenizer fallback**: `JangLoader.resolveTokenizerDirectory` redirects to the cached source-model snapshot when a JANG bundle ships without tokenizer files (MiniMax JANGTQ et al.).
- **VL / hybrid SSM partial cache-hit rollback**: when a prefix-extend cache hit would split the vision-token region or interrupt the SSM recurrence, the engine rolls back to full prefill instead of producing corrupted output. Log line: `rolling back to full prefill (VL vision-token region can't be split)` or `(hybrid SSM recurrence path-dependent on full prefix)`.

### Multi-turn cache behaviour osaurus should expect

| Scenario | Outcome |
|---|---|
| Turn-2 tokens = Turn-1 tokens (session replay) | Full hit. Dense: 40-70% prefill speedup. VL: vision tower skipped. Hybrid SSM: SSM state restored via `ssmStateCache`. |
| Turn-2 = Turn-1 + new tokens, dense | Paged hit on shared prefix; remaining tokens prefill normally. 62% observed speedup on Qwen3-0.6B test harness. |
| Turn-2 = Turn-1 + new tokens, VL or hybrid SSM | Partial hit reported by coordinator, **engine rolls back to full prefill** for correctness. No speedup; no corruption. |

---

## 4. Tool-call parsing: authoritative at library level (iter 66)

### Contract

`BatchEngine.generate(input:parameters:)` and `Evaluate.generate(...)` emit four event cases. Every tool call the model emits is surfaced as `.toolCall(ToolCall)` and every `<think>…</think>` block as `.reasoning(String)` — **osaurus should not parse tool calls or reasoning at its level**.

```swift
public enum Generation: Sendable {
    case chunk(String)            // pure user-visible text; no <think>, no <tool_call>
    case reasoning(String)        // streaming chain-of-thought delta (think-pane)
    case toolCall(ToolCall)       // fully-parsed tool call, authoritative
    case info(GenerateCompletionInfo)
}
```

### Why this matters

Before iter 66, `BatchEngine.generate` only emitted `.chunk(String)` — raw detokenized text including tool-call markers. Osaurus was forced to re-parse at the app layer, which meant:

- **Two sources of truth** — vmlx's `ToolCallProcessor` (used on the `Evaluate` path) and osaurus's own parser. When a model-family wire format drifted, one or the other broke.
- **Conflicting formats** — tpae reported "gemma4 is outputting harmony format" because the library wasn't extracting tool calls and osaurus's parser didn't know Gemma-4's `<|tool_call>call:name{k:<|"|>v<|"|>}<tool_call|>` syntax.

After iter 66, the library does all parsing. Both paths (single-stream `Evaluate.generate` AND `BatchEngine.generate`) run identical pipelines:

```
detokenized chunk
    → ReasoningParser.feed(_:)                    (emits .reasoning + forwards content)
    → ToolCallProcessor.processChunk(_:)          (extracts .toolCall + pure text)
    → emit .chunk / .reasoning / .toolCall
```

### Auto-pickup for JANG / MLX / VL models

Both `LLMModelFactory` and `VLMModelFactory` stamp the two capability fields on `ModelConfiguration` in this priority:

1. Caller-supplied override (`ModelConfiguration.toolCallFormat` / `.reasoningParserName`)
2. JANG `capabilities.tool_parser` / `capabilities.reasoning_parser` from `jang_config.json`
3. `ToolCallFormat.infer(from: modelType)` / model-type reasoning heuristic

`ToolCallFormat.fromCapabilityName` accepts every short alias the JANG converter produces (`qwen`, `qwen3_6`, `minimax`, `glm47`, `deepseek`, `nemotron`, etc.), so JANG-stamped models pick the right parser automatically.

### Supported families

| Family | `toolCallFormat` | Reasoning stamp | Interleaved thinking |
|---|---|---|---|
| Qwen 3 / 3.5 / 3.6 dense / 3 Coder / Nemotron | `.xmlFunction` | `qwen3` / `think_xml` | ✓ (Qwen 3.6 wire format) |
| MiniMax M2 / M2.5 | `.minimaxM2` | `minimax` | ✓ |
| GLM 4.x / DeepSeek-R1 | `.glm4` | `glm4` / `deepseek_r1` | ✓ |
| **DeepSeek-V4 (incl. JANGTQ)** | **`.dsml`** | **`think_xml`** | ✓ |
| Kimi K2 | `.kimiK2` | `think_xml` | ✓ |
| Gemma 3 | `.gemma` | none | — |
| Gemma 4 (incl. JANG Gemma4WithTools) | `.gemma4` | none | — |
| Mistral (any variant) | `.mistral` | none | — |
| LFM2 | `.lfm2` | none | — |
| Standard JSON / unknown | `.json` | heuristic | ✓ if `<think>` present |

### Verification

| Test | Coverage |
|---|---|
| `Tests/MLXLMTests/ToolCallEdgeCasesTests.swift` (22 passes) | Qwen 3.6 interleaved thinking, MiniMax M2 interleaved thinking, Gemma-4 escape-marker regression, Gemma-4 channel + tool-call coexistence, char-by-char streaming, back-to-back calls, JANG stamp → format mapping, canonical rawValue round-trip, `reasoningParserName` plumb-through |
| `Tests/MLXLMTests/ToolTests.swift` (42 passes, pre-existing) | Baseline per-family parsers |
| `Tests/MLXLMTests/ReasoningParserTests.swift` (passes) | Streaming `<think>` strip + whole-string `split(_:)` |
| `BENCH_BATCH_TOOLCALL=1` (manual, real model) | End-to-end on Qwen3-0.6B / Qwen3.6-35B-JANGTQ2 / Gemma-4-E2B — zero raw markers in `.chunk` |

### What osaurus should do

- **Remove or bypass** the app-layer tool-call parser for the MLX runtime path.
- Consume `.toolCall(ToolCall)` events directly from the stream and forward to the OpenAI `tool_calls` response field.
- Consume `.reasoning(String)` events on a separate UI channel (think-pane, collapsed thought bubble, whatever renders chain-of-thought for this product). Each `.reasoning(String)` is a streaming delta — concatenate them to rebuild the full reasoning transcript; the library does NOT buffer until end-of-think.
- Continue rendering `.chunk(String)` as assistant text — it is already reasoning-stripped and tool-call-stripped.

See `GEMMA4-SLIDING-WINDOW-CRASH.md` and `REASONING-STREAM-EVENT.md` for the per-crash fix log and the reasoning-event contract.

Callers that only need the final answer can still exhaustively match `.chunk` / `.toolCall` / `.info` and ignore `.reasoning` — but because the `Generation` enum is not `@frozen`, add `case .reasoning: break` (or a default) to keep the switch exhaustive. No model-level flag disables reasoning emission; if `reasoningParserName` is `"none"` (Gemma-3/4, Mistral) the library never emits `.reasoning` events in the first place.

### API parity with upstream ml-explore/mlx-swift-lm

`Libraries/MLXLMCommon/Tool/ToolCallProcessor.swift` is **byte-identical** to [ml-explore/mlx-swift-lm `main`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Tool/ToolCallProcessor.swift) as of 2026-04-19 — the same stream-state machine, the same `jsonBracesBalanced`-gated inline-format buffering, the same `separateToken` / `partialMatch` helpers. Osaurus's `StreamAccumulator.swift` consumes this public API today:

```swift
// osaurus/Packages/OsaurusCore/Services/ModelRuntime/StreamAccumulator.swift
let processor = ToolCallProcessor(format: toolCallFormat, tools: toolsSpec)
let displayText = processor.processChunk(token)       // user-visible text
processor.processEOS()                                // flush at end of stream
for toolCall in processor.toolCalls { … }             // authoritative tool calls
```

Every method, every state transition, every return-value nullability matches upstream. Osaurus can pin to either repo without drift.

### What's additive vs upstream (and why)

| vmlx-swift-lm only | Why |
|---|---|
| `ToolCallFormat.gemma4` + `GemmaFunctionParser(startTag:"<|tool_call>", endTag:"<tool_call|>", escapeMarker:"<|\"|>")` | Gemma-4 ships a *different* envelope from Gemma-3; upstream only has `.gemma`. Fixes tpae's "gemma4 is outputting harmony format" issue. |
| `ToolCallFormat.fromCapabilityName(_:)` | Accepts the short stamps the JANG converter writes (`qwen`, `qwen3_6`, `minimax`, `glm47`, `deepseek`, `nemotron`, `gemma4`, `mistral`, `lfm2`, `kimi_k2`). Osaurus's `JANGReasoningResolver` calls this via `ParserResolution`. |
| `ReasoningParser` (+ `fromCapabilityName` + `split(_:)`) | Upstream has no streaming `<think>` parser. Osaurus's `StreamingDeltaProcessor` holds a `ReasoningParser?` instance and feeds chunks through `feed(_:) -> [ReasoningSegment]` + `flush()`. |
| `JangCapabilities` + `JangLoader.loadConfig` + `ParserResolution` | JANG / JANGTQ bundle metadata — osaurus reads `capabilities.reasoning_parser` and `capabilities.tool_parser` stamps for auto-pickup. |
| `ModelConfiguration.reasoningParserName` | Capability stamp carried through `ResolvedModelConfiguration` so Evaluate + BatchEngine can resolve a `ReasoningParser` without re-reading disk. |

All additions are purely additive — `ToolCallFormat` still has every upstream case (including `.llama3` + `Llama3ToolCallParser`) and `infer(from: modelType, configData: Data? = nil)` keeps the upstream secondary-signal path for Llama 3 `rope_scaling.rope_type == "llama3"` / `vocab_size >= 128000` detection.

### Osaurus PR #893 consumer map (2026-04-19)

tpae's WIP PR [osaurus-ai/osaurus#893](https://github.com/osaurus-ai/osaurus/pull/893) "Deprecate Work Mode (migrate into single Chat/Agent system)" touches four sites that consume vmlx APIs. All four are covered by this branch:

| Site | API it uses | Provided by |
|---|---|---|
| `StreamAccumulator.swift` | `ToolCallProcessor(format:tools:)`, `processChunk(_:)`, `processEOS()`, `toolCalls` | `Libraries/MLXLMCommon/Tool/ToolCallProcessor.swift` — byte-identical with upstream |
| `BatchEngineAdapter.swift` | `ModelConfiguration.toolCallFormat` + new `toolCallFormatOverride: ToolCallFormat?` | `ModelConfiguration.toolCallFormat` field stamped by LLM/VLM factories with JANG stamp priority |
| `JANGReasoningResolver.swift` | `JangLoader.loadConfig`, `JangCapabilities`, `ParserResolution.reasoning/.toolCall`, `ReasoningParser`, `ToolCallFormat` | all public on `MLXLMCommon` |
| `StreamingDeltaProcessor.swift` | `ReasoningParser` value, `parser.feed(_:) -> [ReasoningSegment]`, `.content`/`.reasoning` cases, `parser.flush()`, `ReasoningParser()` init | `Libraries/MLXLMCommon/ReasoningParser.swift` |

No osaurus change depends on an API we don't ship. If tpae flips `mlxBatchEngine=YES` default + merges #893, the MLX path is fully covered.

---

## Production bugs fixed this session (iters 28-64)

Each would crash or silently corrupt under osaurus production load.

| Bug class | Closing commit |
|---|---|
| `BatchEngine.generate()` hung across turns under real HF tokenizer (`while let detokenizer.next()` infinite loop) | `16b72d7` (iter 28) |
| `UserInput(prompt:, images:)` silently dropped images — `didSet` doesn't fire in init | `16b72d7` (iter 45) |
| VL partial cache-hit crashed MLX vision-feature merge (`SmallVector out of range`) | `16b72d7` (iter 48) |
| JANGTQ4 bundles crashed at first forward (`JANGTQ runtime sidecar not loaded` — bits=2 default on VL-wrapped configs) | `16b72d7` (iter 49) |
| Hybrid SSM partial cache-hit silently degraded output | `16b72d7` (iter 57) |
| Coordinator `isHybrid` had to be manually set per hybrid model | `16b72d7` (iter 57) |
| JANG weights-only bundles had no chat template | `bca0786` (iter 29) |
| `mlx-community/Qwen3.5-VL-9B-8bit` unsupported `TokenizersBackend` tokenizer class | `cc1ee54` (iter 59) |
| `DiskCache.store` not thread-safe across `MLX.eval` + `save` + SQLite (MTL command-buffer crash under concurrent writers) | `30e00a1` (iter 61) |
| `BatchKVCache.makeMask` sized mask key-dim at `offset+n`, ignoring `RotatingKVCache` slot's `maxSize` cap — crashed any sliding-window family (Gemma-3/4, Mistral-3/4, MiMoV2Flash, BaichuanM1) with `broadcast_shapes (1,1,1,offset+1)` vs `(1,H,1,maxSize)` on the first batched decode step after prefilling past the window. See `GEMMA4-SLIDING-WINDOW-CRASH.md`. | `fix/osaurus-integration-issues` |

---

## How to verify before flipping the default

From this repo's root:

```bash
# Unit tests (~20s) — 121 expected, 4 skipped, 0 failed
./scripts/verify-engine.sh --tests-only

# Quick model sweep (~5 min) — skips 35B hybrid
./scripts/verify-engine.sh --quick

# Full sweep (~15 min) — 25 scenarios across 5 families
./scripts/verify-engine.sh

# 1-hour rotating soak (manual op) — flags any crash / hang / silent regression
./scripts/soak-engine.sh --duration 3600
```

Osaurus-side integration smoke (after flag flip):

1. `/v1/chat/completions` × 1 request — warm path.
2. `/v1/chat/completions` × 4 concurrent (same model) — stress `mlxBatchEngineMaxBatchSize`.
3. Close chat window mid-stream — stress `ModelLease.wait-for-release`.
4. Model swap under `strictSingleModel=true` mid-stream — stress lease eviction deferral.

All four should complete without Metal crashes — those are the crash classes osaurus's `ModelLease` + `MetalGate` layers exist to close; vmlx-swift-lm's `cancel()` path and per-actor serialization match osaurus's assumptions.

---

## DeepSeek-V4 — runtime knobs

DSV4-Flash (43-layer MLA + sqrtsoftplus MoE + per-layer rope + Compressor/Indexer hybrid attention) ships with three cache-mode knobs to balance memory vs. long-context coherence. Auto-detected paths:

- Reasoning parser: `think_xml` (auto, `deepseek_*` model_type).
- Tool format: `.dsml` (auto on model_type `deepseek_v4`; also via JANG `chat.tool_calling.parser = "dsml"`).
- Chat template: use `UserInput(chat:)` plus `additionalContext` so `enable_thinking` (bool) and `reasoning_effort` (`'max'` / `nil`) flow into the tokenizer template or Swift `DSV4Minimal` fallback. The current local Flash bundle has no tokenizer `chat_template`, so the fallback is the production path; it auto-prepends the max-effort preface and toggles `<think>` open vs `</think>` closed at the prompt tail.

Cache-mode env knobs (osaurus can set at process start; defaults preserve the
production SWA+CSA+HSA topology):

| Env var | Effect | Memory @ 8K output | Use when |
|---|---|---|---|
| (default) | Hybrid cache: layer 0/last use `RotatingKVCache(128)`; every `compress_ratio > 0` layer uses `DeepseekV4Cache(128, cr)` with CSA/HSA pools. | context-dependent; roughly a 90% KV-row cut at long context | Chat, reasoning, normal traffic. |
| `DSV4_KV_MODE=full` | `KVCacheSimple` on every layer. No rotation, no CSA/HSA pool. Memory grows linearly with sequence length. | ~360 MB | Diagnostic full-KV runs that fit in memory. |
| `DSV4_KV_MODE=tq` | `KVCacheSimple` at construction; BatchEngine's `BatchQuantize.maybeCompress` swaps to `TurboQuantKVCache` once the offset crosses the min-token threshold. **Caller must also set `GenerateParameters.kvMode = .turboQuant(3, 3)`** for the swap to fire. | ~14 MB | Diagnostic memory-constrained runs that explicitly trade away the DSV4 pool. |

TurboQuant policy: a global `parameters.kvMode = .turboQuant(...)` or
coordinator `defaultKVMode = .turboQuant(...)` does not auto-select the `tq`
override for DSV4. The default model cache has no `KVCacheSimple` layers for
BatchEngine to promote, so the SWA+CSA+HSA cache stays intact unless
`DSV4_KV_MODE=tq` is set deliberately.

Bundle dispatch / quantization knobs:

| Env var | Effect |
|---|---|
| `DSV4_FORCE_JANGTQ=1` | Forces the JANGTQ model class for bundles whose `weight_format` stamp is `bf16` (mislabeled — we've seen this on JANGTQ_2L bundles in the wild). |
| `DSV4_JANGTQ_BITS={2\|4}` | Routed-MoE codebook bits override. Use `4` for JANGTQ_4 bundles. Default resolution: `weight_format` stamp → config.json `quantization.bits` (only when in `{2, 4}`) → `2`. |
| `DSV4_KV_MODE=sliding` or unset | Production path. Uses `RotatingKVCache(128)` on SWA-only layers and `DeepseekV4Cache(128, cr)` on CSA/HSA layers. This is the default and should be used for osaurus serving. |
| `DSV4_KV_MODE=full` | Diagnostic only. Forces plain `KVCacheSimple` and drops DSV4's CSA/HSA pools. Do not use as an osaurus default. |
| `DSV4_KV_MODE=tq` | Diagnostic only. Enables the simple-cache shape that can later be swapped to `TurboQuantKVCache` when the request also sets `.turboQuant(...)`. This deliberately trades away the DSV4 hybrid pool and must not be selected by global cache defaults. |
| `VMLX_CHAT_TEMPLATE_OVERRIDE=/path/to/template.jinja` | Bypass the tokenizer's bundled chat template (e.g. when swift-jinja chokes on a specific construct). |
| `VMLX_CHAT_TEMPLATE_FALLBACK_DISABLE=1` | Opt out of in-process `DSV4Minimal` fallback when a legacy bundle ships no `chat_template`. |

`DSV4_LONG_CTX` is removed. There is no short-context fallback anymore;
DSV4 always uses the SWA+CSA+HSA cache topology unless the operator sets
one of the explicit diagnostic `DSV4_KV_MODE` overrides above.

Loader robustness (2026-04-25): the per-layer quantization inference walks every `.scales` key in the bundle and chooses `(bits, group_size)` from a fixed empirical preference order — `(8,32) (8,64) (8,128) (4,32) (4,64) (4,128) (2,32) (2,64) (2,128) (3,32) (6,32)`. This is shape-authoritative: bundles whose `config.json` got re-stamped with mismatched per-layer overrides (or uniform `bits: 8` while routed experts are actually `bits: 2`) load correctly without manual config patching. Idempotent — clean configs add zero overrides.

**Prompt-mode guidance (live-tested 2026-04-25 on DSV4-Flash JANGTQ):**

| Caller mode | Result |
|---|---|
| `applyChatTemplate(...additionalContext: ["enable_thinking": false])` | Clean & correct. Short questions complete in `<10` tokens with proper EOS. |
| `applyChatTemplate(...additionalContext: ["enable_thinking": true])` + `max_tokens >= 1024` | Correct. Model emits `<think>…</think>` then the answer. |
| `applyChatTemplate(...additionalContext: ["enable_thinking": true, "reasoning_effort": "max"])` + `max_tokens >= 4096` | Canonical config for hard reasoning / HumanEval-style benchmarks. T=1.0 top_p=1.0 per the DeepSeek paper. |
| `applyChatTemplate(...additionalContext: ["enable_thinking": true])` + `max_tokens < 512` | ⚠ Mid-think truncation. Model gets cut off before closing `</think>`. Cap higher or use `enable_thinking=false` for short-output workloads. |
| Raw FIM (no chat template) | ⚠ **Avoid for general chat.** No stop signal → model loops the answer then degenerates into character-level slop on long outputs. Only safe when the prompt itself constrains the continuation (function signature → body). |

The reasoning parser strips `<think>…</think>` from `.chunk` events automatically when consumers want post-thinking content only — see the reasoning channel section above.

**Multi-turn + reasoning gotcha (live-verified 2026-04-25):** when `enable_thinking=true` and `max_tokens` is too low for the model to close `</think>` before the cap, the truncated mid-reasoning text gets injected verbatim into the next turn's prompt and the model goes off-distribution (gibberish output, leaked `</think>` markers in `.chunk`). Two valid mitigations:

1. **Raise `max_tokens` per turn** to ≥ 4096. DSV4 reasoning traces routinely exceed 1K tokens on non-trivial problems.
2. **Set `drop_thinking: true` on prior assistant messages** before re-applying the chat template. The bundled DSV4 Jinja honors `message.get('drop_thinking', false)` for non-last messages — passing it strips prior reasoning blocks from the rendered prompt and the model only sees the final answers + the new user turn. This is the production pattern for long agent loops.

**Short-prompt thinking quirk (live-verified 2026-04-26):** with `enable_thinking=true` on trivial questions ("What is 7+5?", "Hi", etc.), DSV4 sometimes generates the answer INLINE inside the `<think>` block and emits EOS without ever closing `</think>`. The streaming pipeline correctly routes everything before EOS to `.reasoning(String)` events; `.chunk(String)` is empty; `finish: stop`. This is model behaviour, not a runtime bug — DSV4 was trained to match its own EOS pattern even mid-think on short prompts.

If your UI only renders `.chunk` events, the user sees an empty bubble while the thinking pane has the actual answer. Two consumer-side handling patterns:

1. **Auto-toggle thinking off for casual chat** (recommended). Maintain a per-model `AutoThinkingProfile` that defaults DSV4 to `enable_thinking=false` for chat workloads, only flipping on for explicit reasoning intents. The chat-template kwarg is set per-request via `UserInput.additionalContext: ["enable_thinking": false]` and the model emits a normal `</think>` at the prompt tail → output flows through `.chunk` cleanly.
2. **Mirror `.reasoning` to displayed content when content is empty + `finish: stop`**. UI-level fallback that surfaces the answer when the model's reasoning trace contains it but `</think>` was never emitted. Less ideal because the displayed text retains a "thinking" tone ("Okay, the user asks 7+5, the answer is 12") rather than a direct answer.

Cross-runtime fp32-internal mHC RMSnorm cast is locked at both layers — Python (`jang_tools/dsv4/mlx_model.py:1346-1382`) and Swift (`Libraries/MLXLLM/Models/DeepseekV4.swift:559-572`, commit `5f64495`). Required for M3 Ultra correctness; M4 happens to keep fp32 in SIMD lanes by accident but the cast is mandatory parity guard. **Do NOT remove the `.asType(.float32)` on `xFlat` and `hcRMSOnes` in `collapse()`** — see `research/DSV4-HC-PRE-FP32-CAST-FIX-2026-04-25.md` for the diagnosis trail.

Cache + L2 disk: `CacheCoordinator` paged tier handles cross-turn prefix reuse; `DiskCache` (TQDiskSerializer v2 with `LayerKind.kvSimple` / `.tqCompressed` / `.qkv` / `.mamba` / `.rotating` tags) round-trips all layer types including DSV4's per-layer mix. The 2026-04-24 `BatchEngine.swift:733` trim fix ensures full-cache disk hits don't re-feed the last token at the wrong RoPE position.

---

## Pointer to the detailed iter log

Architectural decisions, per-iteration addenda, and the full real-model verification log live at `Libraries/MLXLMCommon/BatchEngine/BATCH_ENGINE.md`. ~2100 lines, chronological. Read this file for a 10-minute overview; read `BATCH_ENGINE.md` when you need the why behind a specific design choice.
