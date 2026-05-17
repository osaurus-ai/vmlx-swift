# vmlx-swift-lm → osaurus public API surface

Authoritative reference for every `MLXLMCommon` / `MLXLLM` / `MLXVLM` symbol
that osaurus's `OsaurusCore` target consumes.  Maintained so an osaurus
contributor (or agent) can land a change without reverse-engineering which
vmlx symbol still exists, which shape it has, and which module it lives in.

Cross-checked against osaurus `main` and PR [#893
`chore/tool-calling-updates`](https://github.com/osaurus-ai/osaurus/pull/893)
on 2026-04-19. Every row is a live import in one of those branches.

## 1. Loading a model

| vmlx symbol | Osaurus consumer | Shape |
|---|---|---|
| `public func loadModelContainer(from: any Downloader, using: any TokenizerLoader, configuration: ModelConfiguration, useLatest:, progressHandler:) async throws -> sending ModelContainer` | `ModelRuntime.swift` | Primary load entry. Factories iterate automatically — no manual VLM vs LLM dispatch needed. |
| `public func loadModelContainer(from: URL, using: any TokenizerLoader) async throws -> sending ModelContainer` | `ModelRuntime.swift` fallback | Local directory overload. |
| `public func loadModel(...) async throws -> sending ModelContext` | `RunBench` + (future osaurus) | Loads without actor isolation when the caller owns serialization. |
| `public let LLMModelFactory.shared` | `ModelRuntime.swift:28` | Registry of LLM model creators. Factory iteration is automatic via `loadModelContainer` — this is mostly a sanity handle for osaurus's own factory-ordering check. |
| `public let VLMModelFactory.shared` | `ModelRuntime.swift:27` | Same for VLM. |
| `public protocol TokenizerLoader: Sendable { func load(from: URL) async throws -> any Tokenizer }` | `SwiftTransformersTokenizerLoader.swift:13` | Osaurus bridges swift-transformers' `AutoTokenizer` → `MLXLMCommon.Tokenizer` by conforming to this protocol. |
| `public protocol Tokenizer: Sendable` | `SwiftTransformersTokenizerLoader.swift` (`TokenizerBridge`) | encode/decode/convertTokenToId/convertIdToToken, bosToken/eosToken/unknownToken, `applyChatTemplate(messages:tools:additionalContext:)`. |

## 2. ModelContainer + ModelContext

| Symbol | Shape |
|---|---|
| `public actor ModelContainer` | Owns a `ModelContext` under serial access. |
| `ModelContainer.perform<R: Sendable>((ModelContext) throws -> R) async throws -> R` (+ sending / async variants) | Osaurus's `BatchEngineAdapter.swift:210` + `MLXGenerationEngine.swift:94` run tokenization + prefill inside `perform` so the non-Sendable `ModelContext` never crosses an actor boundary. |
| `ModelContainer.cacheCoordinator: CacheCoordinator?` | Read in `MLXGenerationEngine.swift:160`. Package-level multi-tier KV cache handle. |
| `ModelContainer.enableCaching(config: CacheCoordinatorConfig = .init())` + `enableCachingAsync()` | Called by osaurus's `installCacheCoordinator` to wire paged + disk caches at load time. |
| `ModelContainer.makeBatchEngine(maxBatchSize: Int = 8, memoryPurgeInterval: Int = 256) async -> BatchEngine` | Osaurus's `BatchEngineAdapter.swift:59` → `container.makeBatchEngine(maxBatchSize: maxBatchSize)`. |
| `BatchEngine.updateMaxBatchSize(_:) async throws` | Runtime admission-limit update. Increasing admits queued work immediately; decreasing waits for active slots to drain below the new limit. Throws after terminal shutdown. |
| `public struct ModelContext: Sendable` | `configuration: ModelConfiguration`, `model: any LanguageModel`, `tokenizer: any Tokenizer`, `processor: any UserInputProcessor`. |

## 3. Configuration

| Symbol | Shape |
|---|---|
| `public struct ModelConfiguration: Sendable, Equatable` | `id`, `tokenizerSource`, `defaultPrompt`, `extraEOSTokens`, `eosTokenIds`, `toolCallFormat: ToolCallFormat?`, **`reasoningParserName: String?`** (iter 66). |
| `public enum TokenizerSource: Sendable, Equatable { .id(String, revision:), .directory(URL) }` | |
| `public struct ResolvedModelConfiguration: Sendable` | Same fields, all URLs resolved. Factories return this inside `ModelContext`. |
| `public enum ToolCallFormat: String, Sendable, CaseIterable` | `.json`, `.lfm2`, `.xmlFunction`, `.glm4`, `.gemma`, `.gemma4`, `.kimiK2`, `.minimaxM2`, `.mistral`, `.llama3`, `.dsml`, `.zayaXml`, `.hunyuan`. |
| `public static func ToolCallFormat.infer(from: String, configData: Data? = nil) -> ToolCallFormat?` | Model-type heuristic with Llama-3 secondary-signal (`vocab_size >= 128000` or `rope_scaling.rope_type == "llama3"`). |
| `public static func ToolCallFormat.fromCapabilityName(_: String?) -> ToolCallFormat?` | JANG-stamp → canonical enum. Accepts `qwen` / `qwen3_6` / `minimax` / `glm47` / `glm5*` / `deepseek_v3` / `deepseek_v4*` / `nemotron` / `gemma4` / `mistral` / `mistral4*` / `pixtral*` / `lfm2` / `kimi_k2` / `dsml` / `zaya_xml` / `hunyuan` / `hy3` plus every direct rawValue. DeepSeek V4 aliases route to DSML before the generic DeepSeek/GLM parser. Returns nil for unknown. |
| `public func ToolCallFormat.createParser() -> any ToolCallParser` | Factory used by `ToolCallProcessor.init`. |

## 4. Tool-call parsing — **byte-identical with upstream ml-explore/mlx-swift-lm**

> `Libraries/MLXLMCommon/Tool/ToolCallProcessor.swift` is byte-identical with
> the upstream `main` branch as of 2026-04-19, including the
> `jsonBracesBalanced`-gated inline-format buffering. Osaurus pins against
> either repo without drift.

| Symbol | Used by |
|---|---|
| `public class ToolCallProcessor { init(format: ToolCallFormat = .json, tools: [[String: any Sendable]]? = nil); func processChunk(_: String) -> String?; func processEOS(); var toolCalls: [ToolCall] }` | `StreamAccumulator.swift:197` (field), `:351` (feed), `:432` + `:515` (EOS flush), `:352+` (drain). |
| `public protocol ToolCallParser: Sendable { var startTag: String? / var endTag: String?; func parse(content:tools:) -> ToolCall?; func parseEOS(_:tools:) -> [ToolCall] }` (default parseEOS provided) | Parser implementations + custom consumers. |
| `public struct ToolCall: Hashable, Codable, Sendable { let function: Function }` | `StreamAccumulator.drainNewToolCalls()` remaps to osaurus's internal `ToolCall`. |
| `ToolCall.Function { let name: String; let arguments: [String: JSONValue]; init(name:, arguments: [String: JSONValue]); init(name:, arguments: [String: any Sendable]) }` | Both inits are public. |
| `public enum JSONValue` (+ `JSONValue.from(_: any Sendable)` converter, `anyValue` readback) | `StreamAccumulator` serializes arguments via `serializeArguments` → JSON string. |
| All family-specific parsers (`JSONToolCallParser`, `XMLFunctionParser`, `GemmaFunctionParser`, `GLM4ToolCallParser`, `KimiK2ToolCallParser`, `MiniMaxM2ToolCallParser`, `MistralToolCallParser`, `PythonicToolCallParser`, `Llama3ToolCallParser`, `DSMLToolCallParser`, `HunyuanToolCallParser`) | Constructed via `ToolCallFormat.createParser()`, never touched directly. ZAYA reuses `XMLFunctionParser` with `<zyphra_tool_call>` wrapper tags. |

### Iter-67 parity fixes

- `PythonicToolCallParser.parseEOS(_:tools:)` now overrides the default so `[search(q='a'), fetch(url='b')]` surfaces **both** calls (upstream behaviour; default impl returned one).
- `ToolCall.Function.init(name:, arguments: [String: JSONValue])` restored (was dropped locally; upstream has it).
- `GemmaFunctionParser(startTag:, endTag:, escapeMarker: "<|\"|>")` init retained — vmlx extension for Gemma-4's different envelope, keyed off `ToolCallFormat.gemma4`.

## 5. Reasoning (`<think>...</think>`) streaming

| Symbol | Used by |
|---|---|
| `public struct ReasoningParser: Sendable` — `init(startTag: String = "<think>", endTag: String = "</think>")` | `StreamingDeltaProcessor.swift` holds an optional instance. Value type — osaurus copies into the processor, writes back after `.feed`. |
| `parser.feed(_: String) -> [ReasoningSegment]` | Per-chunk streaming. |
| `parser.flush() -> [ReasoningSegment]` | End of stream. |
| `public enum ReasoningSegment: Sendable, Equatable { .content(String), .reasoning(String) }` | Osaurus routes `.content` → visible answer, `.reasoning` → think pane. |
| `static ReasoningParser.fromCapabilityName(_: String?) -> ReasoningParser?` | JANG-stamp → parser (or nil for `"none"`/`"mistral"`/`"gemma"`). Accepts canonical stamps plus suffixed direct capability aliases: `gemma4*` / `gpt_oss*` → Harmony, `glm4_moe*` / `glm5*` / `deepseek*` / `laguna*` → think XML, and explicit `mistral4*` → `[THINK]...[/THINK]`. |
| `static ReasoningParser.split(_: String, startTag:, endTag:) -> (reasoning: String, content: String)` | Whole-string convenience for non-streaming reveal. |

## 6. JANG capability stamps

| Symbol | Used by |
|---|---|
| `public struct JangConfig: Sendable { format, formatVersion, quantization, sourceModel, architecture, runtime, capabilities: JangCapabilities? }` | `JangLoader.loadConfig(at:)` return. |
| `public struct JangCapabilities: Sendable { reasoningParser, toolParser, thinkInTemplate, supportsTools, supportsThinking, family, modality, cacheType }` | `JANGReasoningResolver.swift:119` reads this. |
| `public enum JangCapabilities.ResolutionSource: String { jangStamped, modelTypeHeuristic, none }` | `JANGReasoningResolver.Resolution.reasoningSource` / `toolCallSource`. |
| `public enum JangLoader { static func isJangModel(at: URL) -> Bool; static func loadConfig(at: URL) throws -> JangConfig; static func resolveTokenizerDirectory(for: URL) -> URL; static func resolveTokenizerClassSubstitution(for: URL) -> URL }` | `JANGReasoningResolver.swift:123` calls `loadConfig`. |
| `public enum ParserResolution { static func reasoning(capabilities:, modelType:) -> (parser: ReasoningParser?, source: ResolutionSource); static func toolCall(...) -> (format: ToolCallFormat?, source: ResolutionSource) }` | `JANGReasoningResolver.swift:134-138` calls both. |

`ModelConfiguration.toolCallFormat` + `ModelConfiguration.reasoningParserName` are also stamped automatically by `LLMModelFactory._load` + `VLMModelFactory._load` with this priority: **caller override → JANG stamp → model_type heuristic**. This is why osaurus's `BatchEngineAdapter.swift:261` reads `context.configuration.toolCallFormat` as the vmlx-authoritative value and warns on disagreement.

## 7. Input + output types

| Symbol | Shape |
|---|---|
| `public enum Chat { public struct Message { var role: Role; var content: String; var images: [UserInput.Image]; var videos: [UserInput.Video]; init(role:, content:, images: [] = , videos: [] = ); static func system/assistant/user/tool(...) } }` | `MLXGenerationEngine.swift:51` builds messages. Role: `.user`/`.assistant`/`.system`/`.tool`. |
| `public struct UserInput` — `init(chat: [Chat.Message], processing: Processing = .init(), tools: [ToolSpec]? = nil, additionalContext: [String: any Sendable]? = nil)` | `BatchEngineAdapter.swift:224` + `MLXGenerationEngine.swift:118`. |
| `public struct UserInput.Image { case ciImage(CIImage), array(MLXArray), url(URL) }` | |
| `public enum UserInput.Video { case asset(AVAsset), url(URL), frames([VideoFrame]) }` | |
| `public struct UserInput.Processing: Sendable { var resize: CGSize? }` | |
| `public protocol UserInputProcessor: Sendable { func prepare(input: UserInput) async throws -> LMInput }` | Called as `context.processor.prepare(input: userInput)`. |
| `public struct LMInput: Sendable` | Carries `text: LMInput.Text` + optional vision tensors. |
| `public struct GenerateParameters: Sendable` | Temperature, topP, topK, minP, maxTokens, prefillStepSize, kvBits, kvGroupSize, quantizedKVStart, kvMode, repetition/presence/frequency penalty contexts, `extraStopStrings: [String]` (text-level stop sequences — matches `.chunk` only; see `STOP-SEQUENCES-CONTRACT.md`). Osaurus builds via `ModelRuntime.makeGenerateParameters`. |
| `public enum Generation: Sendable { case chunk(String), case reasoning(String), case info(GenerateCompletionInfo), case toolCall(ToolCall) }` | Returned by `BatchEngine.generate` + `Evaluate.generate` + `SpecDecStream.streamDflashLinear` / `streamDDTree`. `.reasoning` is a streaming delta — concat consecutive ones for the full think-block. See `REASONING-STREAM-EVENT.md`. |
| `public enum TokenGeneration: Sendable { case token(Int), case info(GenerateCompletionInfo) }` | Returned by `BatchEngine.submit` + `generateTokenTask`. Raw token IDs. |
| `public enum BatchGeneration: Sendable { case token(Int), case info(GenerateCompletionInfo) }` | Same shape as `TokenGeneration`, emitted by `BatchEngine.submit`. |
| `public struct GenerateCompletionInfo: Sendable` | `promptTokenCount`, `generationTokenCount`, `promptTime`, `generationTime`, `stopReason: StopReason`, plus `tokensPerSecond` computed var. |

## 8. BatchEngine

```swift
public actor BatchEngine {
    public init(context: ModelContext,
                maxBatchSize: Int = 8,
                memoryPurgeInterval: Int = 256,
                cacheCoordinator: CacheCoordinator? = nil)

    public func updateMaxBatchSize(_ newMaxBatchSize: Int) throws

    @discardableResult
    public func submit(input: consuming sending LMInput,
                       parameters: GenerateParameters)
        -> (id: BatchRequestID, stream: AsyncStream<BatchGeneration>)

    public func generate(input: consuming sending LMInput,
                         parameters: GenerateParameters)
        -> AsyncStream<Generation>

    public func cancel(_ id: BatchRequestID)
    public func shutdown()

    public var pendingCount: Int { get }
    public var activeCount: Int { get }
    public var isRunning: Bool { get }
    public var isShutdown: Bool { get }
    public var isAcceptingRequests: Bool { get }
}

public struct BatchRequestID: Hashable, Sendable, CustomStringConvertible
```

Osaurus's `BatchEngineAdapter.swift:152` uses `.submit` + its own `StreamAccumulator` wrapper today. `BatchEngine.generate` is the higher-level path that handles detokenization + reasoning strip + tool-call extraction internally — osaurus can migrate to it to drop app-layer tool parsing.

`shutdown()` is terminal. Once shutdown begins, `isShutdown == true`,
`isAcceptingRequests == false`, and later `submit`/`generate` calls return a
completed `.info(.cancelled)` stream without re-entering the scheduling loop.
`updateMaxBatchSize(_:)` also rejects the stale handle. This lets host
registries fail closed if a stale engine handle escapes during model unload.
The decode loop yields at bounded intervals even for B=1, so `shutdown()` and
`cancel(_:)` are not starved behind a long single-slot decode.

## 9. Cache coordinator

| Symbol | Shape |
|---|---|
| `public actor CacheCoordinator` | `config: CacheCoordinatorConfig`, `pagedCache: PagedCacheManager?`, `diskCache: DiskCache?`, `ssmStateCache: SSMStateCache`. |
| `setHybrid(_: Bool)` / `var isHybrid: Bool` | Osaurus *does not* need to call this manually — `BatchEngine.admitPendingRequests` flips it automatically on first Mamba/SSM slot. |
| `fetch(tokens: [Int], mediaSalt: String? = nil) -> CacheFetchResult` | Used inside `BatchEngine` + `TokenIterator`; osaurus does not call directly. |
| `storeAfterGeneration(...)` | Called by the library's `cacheStoreAction`. |
| `public struct CacheCoordinatorConfig: Sendable { usePagedCache, enableDiskCache, pagedBlockSize, maxCacheBlocks, diskCacheMaxGB, diskCacheDir: URL?, ssmMaxEntries, modelKey: String?, defaultKVMode: KVQuantizationMode = .none, defaultMaxKVSize: Int? = nil, longPromptMultiplier: Double = 2.0 }` | Osaurus builds via `ModelRuntime.buildCacheCoordinatorConfig`. The three trailing fields implement the **coordinator-owned KV sizing contract** (2026-04-21) — see `KV-SIZING-CONTRACT.md`. Setting `defaultKVMode: .turboQuant(keyBits: 3, valueBits: 3)` gives ordinary `KVCacheSimple` attention slots ~5× KV memory savings unless the caller set its own `kvMode`; DSV4's default `RotatingKVCache + DeepseekV4Cache` topology is not promoted because it has no simple layers and must keep CSA/HSA pools. Setting `defaultMaxKVSize: 8192` caps KV growth on prompts larger than `longPromptMultiplier × defaultMaxKVSize`. Callers' explicit `GenerateParameters.kvMode` / `maxKVSize` always win — the coordinator only fills gaps. |
| `CacheCoordinatorConfig.resolveKVPolicy(kvMode:maxKVSize:promptTokenCount:)` | Pure function that returns the effective `(kvMode, maxKVSize)` a slot should run under, given a request's values and the coordinator's defaults. Called from `BatchEngine.admitPendingRequests` before `context.model.newCache(...)`. 10 unit tests in `Tests/MLXLMTests/CacheCoordinatorKVPolicyTests.swift` lock the resolution rules. |

## 10. Direct single-stream path (TokenIterator)

For the path that bypasses `BatchEngine`:

```swift
let iterator = try TokenIterator(
    input: fullLMInput,
    model: context.model,
    parameters: parameters,
    cacheCoordinator: container.cacheCoordinator
)
let (stream, task) = MLXLMCommon.generateTokenTask(
    promptTokenCount: newPromptTokens.count,
    modelConfiguration: context.configuration,
    tokenizer: context.tokenizer,
    iterator: iterator,
    wiredMemoryTicket: wiredMemoryTicket
)
```

Both symbols match `MLXGenerationEngine.swift:160`+`:175` byte-for-byte.

## 11. Environment shims

- `VMLX_CHAT_TEMPLATE_OVERRIDE=/path/to/template.jinja` — substitutes the tokenizer's shipped `chat_template` at load.  Ship templates:
  - `Libraries/MLXLMCommon/ChatTemplates/Gemma4Minimal.jinja`
  - `Libraries/MLXLMCommon/ChatTemplates/Gemma4WithTools.jinja`
- `VMLX_TOKENIZER_CLASS_OVERRIDE=Qwen2Tokenizer` — rewrites unsupported `tokenizer_class` at load.

Both are no-ops when unset.

## 12. Migration path to drop osaurus's own tool-call parser

Current osaurus flow (`StreamAccumulator`):

```
BatchEngine.submit(...) → AsyncStream<BatchGeneration>
    → StubTokenizer.decode() → String chunks
    → osaurus's ToolCallProcessor.processChunk() → .toolInvocation events
```

Drop-in migration to drop the app-layer processor:

```swift
let stream = await engine.generate(input: input, parameters: params)
for await event in stream {
    switch event {
    case .chunk(let text):     // pure user text, reasoning-stripped, tool-stripped
        accumulator.appendAssistantText(text)
    case .reasoning(let thought): // streaming <think>…</think> delta
        accumulator.appendReasoning(thought)
    case .toolCall(let call):  // authoritative — no re-parse needed
        accumulator.emitToolCall(name: call.function.name,
                                 arguments: call.function.arguments)
    case .info(let info):      // completion metrics
        accumulator.finish(stopReason: info.stopReason)
    }
}
```

`BatchEngine.generate` internally runs `ReasoningParser.fromCapabilityName(modelConfiguration.reasoningParserName)` (set by the factory via JANG stamp + model-type heuristic) and `ToolCallProcessor(format: toolCallFormat)`. Result: `.chunk` is pure text and `.toolCall` is authoritative on every supported family.

## 13. Speculative decoding (DFlash + DDTree) — opt-in

Block-diffusion speculative decoding landed on branch `feature/ddtree-spec-dec`. Zero API churn for callers using `.none` / `nil`.

```swift
public enum DraftStrategy: Sendable {
    case none
    case autoregressive(draftModel: any LanguageModel, numDraftTokens: Int)
    case dflash(drafterPath: URL, blockSize: Int)
    case ddtree(drafterPath: URL, branchingBudget: Int, blockSize: Int)
}

public var GenerateParameters.draftStrategy: DraftStrategy?
```

Set `draftStrategy` on `GenerateParameters` and the same `Evaluate.generate` / `BatchEngine.generate` entry points honor the request. Block-diffusion strategies dispatch through `SpecDecStream.streamViaStrategy`. Native MTP dispatches through the real `NativeMTPTokenIterator` when the loaded model exposes an active MTP head.

Current native-MTP boundary for Osaurus:

- `Evaluate.generate` supports `.nativeMTP(depth:)`.
- `BatchEngine.generate` supports `.nativeMTP(depth:)` only as an exclusive solo lane. It does not claim multi-slot paged native-MTP batching yet.
- `BatchEngine.submit` rejects native MTP instead of silently running ordinary AR batch decode.

The returned `AsyncStream<Generation>` emits the same `.chunk(String)` + `.toolCall(ToolCall)` + `.info(GenerateCompletionInfo)` events the non-speculative path does.

Invariant: **at temperature 0, output is byte-identical to greedy autoregressive decode**. The drafter affects speed (mean acceptance length), not output.

Per-symbol reference:

| Symbol | Purpose |
|---|---|
| `DraftStrategy` | enum on `GenerateParameters.draftStrategy` |
| `SpecDecStream.streamDflashLinear / streamDDTree / streamViaStrategy` | wraps runtime → `AsyncStream<Generation>` |
| `SpecDecDrafterResolver` | actor cache for loaded drafters; `shared` singleton |
| `DFlashDraftModel` | block-diffusion drafter ported from z-lab/dflash `dflash.py` |
| `DFlashDrafterLoader.load(from:)` | local HF snapshot → `DFlashDraftModel` |
| `HiddenStateCaptureModel` | protocol that target models implement to expose per-layer hidden states (Qwen3 conforms) |
| `TokenEmbedderModel` | protocol that target models implement to expose `embed(_:)` + `projectToLogits(_:)` (Qwen3 conforms) |
| `TreeBuilder` / `TreeCompile` / `TreeVerify` | DDTree algorithm ports of `tree.py` / `compile.py` / `verify.py` |
| `SpecDecRuntimeLinear.run` / `SpecDecRuntimeDDTree.run` | stateless runtime entry points |
| `SpecDecRuntime` actor | long-lived wrapper for BatchEngine integration |

Full guide: `Libraries/MLXLMCommon/SpecDec/OSAURUS-SPECDEC.md`.

## 14. Integration smoke: what the library verifies pre-ship

- `./scripts/verify-engine.sh --tests-only` → 121 unit tests, 0 failures.
- `./scripts/verify-engine.sh --quick` → 20 scenarios green.
- `./scripts/verify-engine.sh` → 25 scenarios including Qwen3.6-35B hybrid SSM.
- `BENCH_BATCH_TOOLCALL=1 BENCH_MODEL=<path> RunBench` (gitignored bench) → zero raw tool-call / reasoning markers leak to `.chunk` on Qwen3, Qwen3.6-35B-JANGTQ2, Gemma-4-E2B.

If osaurus-side smoke fails on a symbol not listed above, file an issue — that means a new osaurus consumer was added and this doc needs updating.
