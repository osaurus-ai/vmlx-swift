# vMLX Swift Osaurus Server/API Panel Spec - 2026-05-19

This is the source-backed spec for building an Osaurus server/API panel on top
of the current `vmlx-swift` checkout. It intentionally avoids names from the
separate Python/server engine. Every symbol below is either defined in this
Swift repo or explicitly marked as host-owned work.

Source anchors:

- `Libraries/MLXLMCommon/ServerRuntimeSettings.swift`
- `Libraries/MLXLMCommon/ModelContainer.swift`
- `Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift`
- `Libraries/MLXLMCommon/BatchEngine/ModelContainerBatch.swift`
- `Libraries/MLXLMCommon/BatchEngine/BatchCompile.swift`
- `Libraries/MLXLMCommon/BatchEngine/BatchQuantize.swift`
- `Libraries/MLXLMCommon/Cache/CacheCoordinatorConfig.swift`
- `Libraries/MLXLMCommon/Cache/CacheCoordinator.swift`
- `Libraries/MLXLMCommon/Cache/CacheHelpers.swift`
- `Libraries/MLXLMCommon/Cache/TQDiskSerializer.swift`
- `Libraries/MLXLMCommon/KVCache.swift`
- `Libraries/MLXLMCommon/Cache/ZayaCCACache.swift`
- `Libraries/MLXLMCommon/Evaluate.swift`
- `Libraries/MLXLMCommon/ModelRuntimeCapabilitySnapshot.swift`
- `Libraries/MLXLLM/LLMModelFactory.swift`
- `Libraries/MLXVLM/VLMModelFactory.swift`

## Status Tags

Use these labels in the panel spec and implementation tickets:

- `[ENGINE-READY]`: the Swift type/function exists and is wired into the runtime path.
- `[HOST-WIRE]`: Osaurus owns the HTTP route, lifecycle, registry, or UI behavior; vMLX supplies lower-level symbols.
- `[NEEDS BRIDGE]`: the Swift settings field exists and validates, but the current runtime does not consume it end-to-end.
- `[FUTURE/VERIFY]`: no source-backed serving path was traced in this checkout, or live proof is still required.
- `[DO NOT USE]`: stale or foreign-engine names that should not appear in the Osaurus panel contract.

## Review Of The Existing Draft

The draft structure is directionally right: server/model settings, API gateway,
usage/cache/perf stats, saved sessions, and architecture-aware cache defaults
are the right product surfaces.

Corrections needed before handing this to Osaurus:

- `[ENGINE-READY]` The source of truth for settings is
  `VMLXServerRuntimeSettings`, not a Python engine config.
- `[ENGINE-READY]` `BatchEngine.maxBatchSize` exists, but the UI should wire it
  through `ModelContainer.makeBatchEngine(maxBatchSize:memoryPurgeInterval:)`
  at engine creation and `BatchEngine.updateMaxBatchSize(_:)` for live updates.
- `[ENGINE-READY]` Cache settings bridge through
  `VMLXServerRuntimeSettings.cacheCoordinatorConfig(modelKey:diskCacheDirectory:ssmMaxEntries:)`
  into `CacheCoordinatorConfig`.
- `[HOST-WIRE]` Single-model-per-session, swap-on-model-id, endpoint routing,
  HTTP auth, CORS, LAN binding, saved sessions, and endpoint catalog are
  Osaurus host responsibilities. vMLX supplies model/container/engine/cache
  primitives.
- `[NEEDS BRIDGE]` Some fields in `VMLXServerRuntimeSettings` validate and
  persist but are not fully consumed by the runtime today. The panel must show
  these as planned/host-wired, not as finished engine behavior.
- `[FUTURE/VERIFY]` Do not list image generation, audio generation, embeddings,
  rerank, or arbitrary non-chat endpoints as ready unless Osaurus has an
  explicit route backed by a traced engine path. This checkout has a text/VL
  generation stack and an `MLXEmbedders` product, but this spec did not trace a
  serving endpoint for embeddings/rerank/image/audio generation.

## Hard Rule: Auto Means Model Topology Selected

The default panel mode for cache, template, parsers, MTP, and modality support
must be `Auto`. In `Auto`, Osaurus should use the model's actual resolved
configuration and cache objects:

- `ModelConfiguration.toolCallFormat`
- `ModelConfiguration.reasoningParserName`
- `ModelConfiguration.generationDefaults`
- `ModelConfiguration.mtpStatus`
- `JangConfig.capabilities`
- `ModelRuntimeCapabilitySnapshot`
- `context.model.newCache(parameters:)`
- `CacheFamily.classify(_:)`
- `cacheContainsPathDependentState(_:)`
- `cacheRequiresDiskBackedCoordinatorRestore(_:)`

Manual controls may narrow or disable behavior, but they must not force an
incompatible cache representation onto a model. A model with `MambaCache`,
`ArraysCache`, `ZayaCCACache`, `CacheList`, `RotatingKVCache`, `HybridPoolCache`,
`TurboQuantKVCache`, or `QuantizedKVCache` is not the same cache topology as a
plain `KVCacheSimple` model.

## Panel 1: Server / Model Settings

### Network And Server Identity

`[ENGINE-READY]` Persist these fields from `VMLXServerNetworkSettings`:

| UI control | Swift field |
| --- | --- |
| Host | `network.host` |
| Port | `network.port` |
| API key | `network.apiKey` |
| Served model name | `network.servedModelName` |
| Rate limit | `network.rateLimitRequestsPerMinute` |
| Request timeout | `network.timeoutSeconds` |
| Log level | `network.logLevel` as `VMLXServerLogLevel` |
| CORS origins | `network.corsOrigins` |

`[HOST-WIRE]` The actual HTTP server, LAN binding, auth enforcement, CORS
middleware, endpoint catalog, and model-name routing live in Osaurus. vMLX only
defines the settings shape and validation.

### Model Loading, Residency, And Swapping

`[ENGINE-READY]` The load/generation ownership surface is:

- `ModelContainer`
- `loadModelContainer(...)`
- `LoadConfiguration`
- `ModelContainer.enableCaching(config:)`
- `ModelContainer.enableCachingAsync(config:)`
- `ModelContainer.disableCaching()`
- `ModelContainer.cacheCoordinator`
- `ModelContainer.makeBatchEngine(maxBatchSize:memoryPurgeInterval:)`
- `BatchEngine.shutdown()`

`[HOST-WIRE]` A saved "server session" is not a vMLX type. Osaurus should own a
session record containing:

- stable model id or bundle URL;
- optional `network.servedModelName`;
- `VMLXServerRuntimeSettings`;
- resolved `ModelRuntimeCapabilitySnapshot`;
- stable cache `modelKey`;
- selected load status and last runtime error;
- host-owned endpoint and API compatibility settings.

`[HOST-WIRE]` For single-model swap-on-model-id:

1. reject or drain in-flight requests;
2. call `BatchEngine.shutdown()` on the old engine;
3. call `ModelContainer.disableCaching()` on the old container;
4. release the old `BatchEngine` and `ModelContainer`;
5. create the new `LoadConfiguration`;
6. load the new container;
7. enable caching with the new stable `modelKey`;
8. create a new `BatchEngine`.

The panel should show that repeated swaps must release MLX/Metal/wired memory.
Do not represent a model swap as only changing `network.servedModelName`.

### Concurrency And Continuous Batching

`[ENGINE-READY]` The real batch engine symbols are:

- `public actor BatchEngine`
- `public private(set) var maxBatchSize: Int`
- `public func updateMaxBatchSize(_ newMaxBatchSize: Int) throws`
- `public func submit(input:parameters:) -> (id: BatchRequestID, stream: AsyncStream<BatchGeneration>)`
- `public func generate(input:parameters:) -> AsyncStream<Generation>`
- `public func cancel(_ id: BatchRequestID)`
- `public func shutdown()`
- `public var pendingCount: Int`
- `public var activeCount: Int`
- `public var isRunning: Bool`
- `public var isShutdown: Bool`
- `public var isAcceptingRequests: Bool`

`[ENGINE-READY]` Wire:

| UI control | Swift field/function | Runtime meaning |
| --- | --- | --- |
| Max concurrent sequences | `concurrency.maxConcurrentSequences` -> `makeBatchEngine(maxBatchSize:)` / `updateMaxBatchSize(_:)` | request admission limit |
| Prefill step size | `concurrency.prefillStepSize` -> `GenerateParameters.prefillStepSize` | prompt chunk size |
| Continuous batching | `concurrency.continuousBatching` | host chooses batched route vs single-stream route |
| Active requests | `BatchEngine.activeCount` | current active slots |
| Queue depth | `BatchEngine.pendingCount` | waiting requests |
| High-water active slots | `BatchEngine.activeCountHighWatermarkForDiagnostics` | proof that B>1 actually ran |
| Decode compatibility splits | `BatchEngine.decodeCompatibilitySplitCountForDiagnostics` | batch decode had to split incompatible caches |
| TurboQuant compressions | `BatchEngine.turboQuantCompressionCountForDiagnostics` | TQ cache compression actually fired |

`[NEEDS BRIDGE]` `concurrency.prefillBatchSize`,
`concurrency.completionBatchSize`, and `concurrency.smeltMode` exist and
validate, but this spec did not trace an end-to-end runtime consumer for them
equivalent to `maxConcurrentSequences` or `prefillStepSize`.

`[HOST-WIRE]` BatchEngine concurrency is request admission and batched decode
through the MLX/Metal runtime. Do not market it as independent parallel Metal
compute.

### Cache Controls

`[ENGINE-READY]` The panel should persist `VMLXServerCacheSettings` and build
`CacheCoordinatorConfig` by calling:

```swift
let config = settings.cacheCoordinatorConfig(
    modelKey: modelKey,
    diskCacheDirectory: nil,
    ssmMaxEntries: 50
)
await container.enableCachingAsync(config: config)
```

`[ENGINE-READY]` The real bridge fields are:

| UI control | Swift field | Consumed by |
| --- | --- | --- |
| Prefix cache master toggle | `cache.prefix.enabled` | `cacheCoordinatorConfig(...)` gates `usePagedCache` and `enableDiskCache` |
| Paged KV enabled | `cache.pagedKV.enabled` | `CacheCoordinatorConfig.usePagedCache` |
| Paged block size | `cache.pagedKV.blockSize` | `CacheCoordinatorConfig.pagedBlockSize` |
| Max paged blocks | `cache.pagedKV.maxBlocks` | `CacheCoordinatorConfig.maxCacheBlocks` |
| Legacy disk enabled | `cache.legacyDisk.enabled` | `CacheCoordinatorConfig.enableDiskCache` only when paged KV is off |
| Block disk L2 enabled | `cache.blockDisk.enabled` | `CacheCoordinatorConfig.enableDiskCache` only when paged KV is on |
| Disk max GB | `cache.legacyDisk.maxSizeGB` or `cache.blockDisk.maxSizeGB` | `CacheCoordinatorConfig.diskCacheMaxGB` |
| Disk directory | `cache.legacyDisk.directory` or `cache.blockDisk.directory` | `CacheCoordinatorConfig.diskCacheDir` |
| Live KV codec | `cache.liveKVCodec` | `CacheCoordinatorConfig.defaultKVMode` for TurboQuant only |
| TurboQuant key/value bits | `cache.turboQuantKeyBits`, `cache.turboQuantValueBits` | `KVQuantizationMode.turboQuant(keyBits:valueBits:)` |
| Default max KV size | `cache.defaultMaxKVSize` | `CacheCoordinatorConfig.defaultMaxKVSize` |
| Long prompt multiplier | `cache.longPromptMultiplier` | `CacheCoordinatorConfig.longPromptMultiplier` |
| SSM rederive | `cache.enableSSMReDerive` | `CacheCoordinatorConfig.enableSSMReDerive` |

Important validator rules:

- `cache.pagedKV.enabled` and `cache.legacyDisk.enabled` cannot both be true.
- `cache.prefix.enabled == false` disables paged and disk reuse in
  `cacheCoordinatorConfig(...)` even if stale child toggles remain on.
- `cache.liveKVCodec == .turboQuant` requires both
  `cache.turboQuantKeyBits` and `cache.turboQuantValueBits`.
- `multimodal.requireMediaSaltForCache == false` is invalid while any reuse tier
  is enabled.

`[NEEDS BRIDGE]` These fields exist and validate, but are not fully enforced by
`cacheCoordinatorConfig(...)` today:

- `cache.prefix.legacyEntryCountCache`
- `cache.prefix.memoryLimitMB`
- `cache.prefix.memoryPercent`
- `cache.prefix.ttlMinutes`
- `cache.storedKVCodec`

Show them as session metadata or future engine work unless Osaurus adds a real
consumer.

### Architecture-Aware Cache Matrix

`[ENGINE-READY]` Cache topology is derived from the model's actual cache objects,
not from a user choosing a family in the UI. `ModelContainer.enableCachingAsync`
checks `context.model.newCache(parameters:nil)` and eagerly marks hybrid state
for `MambaCache`, `ArraysCache`, and `ZayaCCACache`. `BatchEngine` also checks
each admitted slot and updates the coordinator through `setHybrid(_:)` and
`setPagedIncompatible(_:)`.

| Detected cache/architecture | Real Swift signal | Required panel behavior |
| --- | --- | --- |
| Plain dense attention | `KVCacheSimple`, `CacheFamily.simple` | Prefix/paged/block disk are valid; TurboQuant KV can be offered if explicit bit widths are set. |
| Sliding/window attention | `RotatingKVCache` or `RotatingKVCacheWrapper` | Do not treat paged KV as a complete restore path; `cacheRequiresDiskBackedCoordinatorRestore(_:)` forces disk-backed restoration. |
| TurboQuant KV | `TurboQuantKVCache`, `KVQuantizationMode.turboQuant` | Show live TQ codec and TQ compression stats; disk uses `TQDiskSerializer` v2 layer tags. |
| Legacy affine KV | `QuantizedKVCache`, `KVQuantizationMode.affine` or `kvBits` | Do not promote as a server-batch-ready default. `BatchQuantize` keeps unsupported affine paths out of batched TQ behavior. |
| Hybrid SSM / linear attention | `MambaCache`, `ArraysCache`, or recursive `CacheList` containing them | Enable hybrid status, SSM companion cache, and `enableSSMReDerive`; KV-only hits are not valid. |
| ZAYA CCA | `ZayaCCACache` | Treat as path-dependent CCA state, not plain KV. Requires companion state/disk v2 restore. |
| DeepSeek V4 SWA/CSA/HSA | `HybridPoolCache` / DSV4 cache implementation | Set `isPagedIncompatible`; prefix reuse must use disk v2 `LayerKind.deepseekV4`, not paged KV blocks. |
| Composite cache | `CacheList` | Recursively inspect sub-caches; do not show one generic KV status for the whole model. |
| VL/video/audio models | non-empty `UserInput.images`, `videos`, or `audios`; `ModelRuntimeCapabilitySnapshot` modality support | Require `mediaSalt`; show processor/media-cache behavior separately from text-only cache hits. |
| Native MTP | `MTPBundleStatus`, `NativeMTPAutoDecodePolicy`, `DraftStrategy.nativeMTP(depth:)` | Exclusive solo generate lane only; draft cache stays separate and only accepted tokens enter base cache. |

`[ENGINE-READY]` Expose these cache stats in the panel:

- `CacheCoordinator.snapshotStats()`
- `CacheCoordinatorStatsSnapshot.pagedEnabled`
- `CacheCoordinatorStatsSnapshot.pagedStats`
- `CacheCoordinatorStatsSnapshot.diskEnabled`
- `CacheCoordinatorStatsSnapshot.diskStats`
- `CacheCoordinatorStatsSnapshot.ssmStats`
- `CacheCoordinatorStatsSnapshot.isHybrid`
- `CacheCoordinatorStatsSnapshot.isPagedIncompatible`
- `PagedCacheManager.snapshotStats()`
- `DiskCache.snapshotStats()`
- `SSMStateCache.snapshotStats()`

`[HOST-WIRE]` For each loaded model, display "Auto detected: ..." using the
runtime stats above plus the first observed cache family. Do not let users
hand-pair incompatible cache state with a model. If a control is invalid for
the detected cache family, disable it and show the reason.

### Generation Defaults

`[ENGINE-READY]` Use `VMLXServerGenerationDefaults`:

- `generation.streamInterval`
- `generation.maxTokens`
- `generation.temperature`
- `generation.topP`
- `generation.topK`
- `generation.minP`
- `generation.repetitionPenalty`

`[ENGINE-READY]` Build request params with:

```swift
let modelConfiguration = await container.configuration
var parameters = settings.resolvedGenerateParameters(
    generationConfig: modelConfiguration.generationDefaults,
    fallback: GenerateParameters()
)
if let step = settings.concurrency.prefillStepSize {
    parameters.prefillStepSize = step
}
```

`[HOST-WIRE]` `generation.streamInterval` is a server streaming/coalescing
setting. It is validated by `VMLXServerRuntimeSettings` but not consumed by
`GenerateParameters`.

### Chat Templates, Tool Parsers, And Reasoning Parsers

`[ENGINE-READY]` Parser auto-selection is already model-config-driven:

- LLM/VLM factories set `ModelConfiguration.toolCallFormat`.
- LLM/VLM factories set `ModelConfiguration.reasoningParserName`.
- Priority is caller override, then JANG capability stamp, then `model_type`
  heuristic.
- `BatchEngine.generate` and `Evaluate.generate` emit `Generation.chunk`,
  `Generation.reasoning`, `Generation.toolCall`, and `Generation.info`.

`[ENGINE-READY]` The user input types are:

- `Chat.Message`
- `Chat.Message.reasoningContent`
- `Chat.Message.toolCalls`
- `Chat.Message.toolCallId`
- `UserInput(prompt:images:videos:audios:tools:additionalContext:)`
- `UserInput(chat:processing:tools:additionalContext:)`
- `UserInput.Image.ciImage`, `.url`, `.array`
- `UserInput.Video.avAsset`, `.url`, `.frames`
- `UserInput.Audio.url`, `.samples`, `.array`, `.preEncoded`
- `UserInput.Processing`

`[NEEDS BRIDGE]` The settings fields below exist and validate, but this spec did
not trace an automatic end-to-end consumer inside `VMLXServerRuntimeSettings`:

- `tools.mcpConfigFile`
- `tools.enableAutoToolChoice`
- `tools.toolParserOverride`
- `tools.reasoningParserOverride`
- `tools.customChatTemplate`

If Osaurus exposes overrides, it should apply them before load through
`ModelConfiguration` or through a clearly defined host-side route. Do not
mutate parser/template state invisibly after the model has been loaded.

`[HOST-WIRE]` Reasoning is not a generic cache setting. Multi-turn reasoning
must be represented in messages with `Chat.Message.reasoningContent` and routed
through the model's stamped `ReasoningParser`. Hybrid SSM rederive is a cache
state requirement; it should not be conflated with visible thinking UI.

### Multimodal Support

`[ENGINE-READY]` Use:

- `VMLXServerMultimodalSettings`
- `VMLXVLMServerMode`
- `ModelRuntimeCapabilitySnapshot`
- `ModelRuntimeCapabilityRequest(input:usesReasoning:usesNativeMTP:)`
- `VMLXServerRuntimeSettings.validateRequest(_:capabilitySnapshot:unknownPolicy:)`

`[ENGINE-READY]` The request validator rejects:

- unsupported model modalities;
- unknown modalities when `unknownPolicy == .rejectUnknown`;
- server-disabled modalities from `multimodal.vlmMode`, `enableVideo`,
  `enableAudio`, or `mtp.mode`.

`[ENGINE-READY]` Media-aware cache correctness requires `mediaSalt`. The
coordinator also has:

- `recordPostPrepareCacheKeyAlias(rawTokens:effectiveTokens:mediaSalt:)`
- `resolvePostPrepareCacheKeyAlias(rawTokens:mediaSalt:)`

These are needed for media processors whose final cache key is only known after
preparation.

### MTP / Speculative Decode

`[ENGINE-READY]` The real MTP/server symbols are:

- `VMLXServerMTPSettings`
- `VMLXMTPServerMode`
- `VMLXMTPLaunchMode`
- `VMLXResolvedMTPLaunch`
- `MTPBundleStatus`
- `MTPBundleStatusSnapshot`
- `NativeMTPAutoDecodePolicy`
- `DraftStrategy.nativeMTP(depth:)`
- `GenerateParameters.draftStrategy`
- `LoadConfiguration.nativeMTP`

`[ENGINE-READY]` Resolve load and request decisions from the same bundle
evidence:

```swift
let loadConfig = settings.resolvedLoadConfiguration(
    base: .default,
    configData: configData,
    jangConfig: jangConfig,
    status: mtpStatus
)
let draft = settings.resolvedMTPDraftStrategy(
    configData: configData,
    jangConfig: jangConfig,
    status: mtpStatus
)
```

`[ENGINE-READY]` `BatchEngine.submit` rejects native MTP; `BatchEngine.generate`
supports native MTP only as an exclusive solo lane. Multi-slot native-MTP
batching should be `[FUTURE/VERIFY]`.

### Power Management

`[ENGINE-READY]` Persist and validate `VMLXServerPowerSettings`:

- `power.autoSleepEnabled`
- `power.lightSleepAfterSeconds`
- `power.deepSleepAfterSeconds`
- `power.wakeOnRequest`
- `power.jitLoad`

`[HOST-WIRE]` Actual light sleep, deep sleep, wake-on-request, process lifetime,
and memory pressure behavior are host lifecycle features. They must not mutate
sampling, parser selection, cache topology, or MTP mode.

## Panel 2: API Gateway

### Gateway Ownership

`[HOST-WIRE]` vMLX Swift does not define HTTP route handlers here. Osaurus owns:

- OpenAI-compatible routes;
- Anthropic-compatible routes;
- Ollama-compatible routes;
- API auth;
- CORS;
- LAN exposure;
- quick-start snippets;
- one-click coding-tool setup;
- endpoint catalog;
- usage accounting.

vMLX supplies:

- model loading and model context;
- `UserInput` and `Chat.Message`;
- `GenerateParameters`;
- `BatchEngine.generate`;
- `BatchEngine.submit`;
- `Generation` events;
- model capability validation;
- cache and batch diagnostics.

### Supported Gateway Shape

Recommended routing model:

1. Parse the request's model name.
2. Find the host session by `network.servedModelName` or stable model id.
3. Convert HTTP messages to `[Chat.Message]`.
4. Build `UserInput(chat:processing:tools:additionalContext:)`.
5. Build `ModelRuntimeCapabilityRequest`.
6. Validate with `settings.validateRequest(...)`.
7. Prepare `LMInput` through `context.processor.prepare(input:)`.
8. Stream with `BatchEngine.generate(input:parameters:)`.
9. Map `Generation.chunk`, `.reasoning`, `.toolCall`, and `.info` to the
   selected compatibility protocol.

Minimal Swift-side shape:

```swift
let userInput = UserInput(
    chat: messages,
    processing: .init(),
    tools: tools,
    additionalContext: additionalContext
)

let usesNativeMTP = parameters.draftStrategy?.usesNativeMTP == true
let request = ModelRuntimeCapabilityRequest(
    input: userInput,
    usesReasoning: wantsReasoning,
    usesNativeMTP: usesNativeMTP
)
let validation = settings.validateRequest(
    request,
    capabilitySnapshot: capabilitySnapshot,
    unknownPolicy: .rejectUnknown
)
guard validation.allowed else {
    throw GatewayValidationError(validation)
}

let input = try await container.perform { context in
    try await context.processor.prepare(input: userInput)
}

let stream = await engine.generate(input: input, parameters: parameters)
for await event in stream {
    switch event {
    case .chunk(let text):
        emitVisibleText(text)
    case .reasoning(let text):
        emitReasoningText(text)
    case .toolCall(let call):
        emitToolCall(call)
    case .info(let info):
        emitUsage(info)
    }
}
```

`GatewayValidationError`, `emitVisibleText`, `emitReasoningText`,
`emitToolCall`, and `emitUsage` are host-owned examples, not vMLX symbols.

### Endpoint Catalog

Use this status table in the Osaurus panel:

| Endpoint group | Status | Backing |
| --- | --- | --- |
| Chat/text completions | `[HOST-WIRE]` | Back with `BatchEngine.generate` or `BatchEngine.submit`. |
| Model list/status | `[HOST-WIRE]` | Back with host session registry plus `ModelRuntimeCapabilitySnapshot`. |
| Admin cache stats | `[HOST-WIRE]` | Back with `CacheCoordinator.snapshotStats()` and batch diagnostics. |
| Usage/perf stats | `[HOST-WIRE]` | Back with `GenerateCompletionInfo` and host counters. |
| Tool calls | `[HOST-WIRE]` | Back with `Generation.toolCall` and model `ToolCallFormat`. |
| Reasoning stream | `[HOST-WIRE]` | Back with `Generation.reasoning` and `GenerateCompletionInfo.unclosedReasoning`. |
| Vision input | `[HOST-WIRE]` | Back with `UserInput.Image`, VLM processors, capability validation, and `mediaSalt`. |
| Video/audio input | `[FUTURE/VERIFY]` per model | Types exist; require model-specific live validation and cache proof before listing as ready. |
| Embeddings/rerank | `[FUTURE/VERIFY]` | Do not list as ready from this text/VL generation spec alone. |
| Image/audio generation | `[FUTURE/VERIFY]` | Do not list endpoints that would 404. |

## Stats And Usage View

`[ENGINE-READY]` Generation metrics:

- `GenerateCompletionInfo.promptTokenCount`
- `GenerateCompletionInfo.generationTokenCount`
- `GenerateCompletionInfo.promptTime`
- `GenerateCompletionInfo.generateTime`
- `GenerateCompletionInfo.promptTokensPerSecond`
- `GenerateCompletionInfo.tokensPerSecond`
- `GenerateCompletionInfo.stopReason`
- `GenerateCompletionInfo.unclosedReasoning`

`[ENGINE-READY]` Batch metrics:

- `BatchEngine.pendingCount`
- `BatchEngine.activeCount`
- `BatchEngine.activeCountHighWatermarkForDiagnostics`
- `BatchEngine.decodeCompatibilitySplitCountForDiagnostics`
- `BatchEngine.turboQuantCompressionCountForDiagnostics`
- `BatchEngine.isRunning`
- `BatchEngine.isAcceptingRequests`
- `BatchEngine.isShutdown`

`[ENGINE-READY]` Cache metrics:

- `CacheCoordinator.snapshotStats()`
- `CacheStats.cacheHits`
- `CacheStats.cacheMisses`
- `CacheStats.evictions`
- `DiskCacheStats.hits`
- `DiskCacheStats.misses`
- `DiskCacheStats.stores`
- `SSMStateCacheStats.hits`
- `SSMStateCacheStats.misses`
- `SSMStateCacheStats.reDerives`

`[HOST-WIRE]` API usage counts, per-key accounting, per-route request history,
LAN clients, and rate-limit accounting are Osaurus-owned.

## Saved Sessions

`[HOST-WIRE]` There is no traced `SavedSession` type in this checkout. Define it
in Osaurus as a host record containing:

```json
{
  "model_id": "stable-model-or-bundle-id",
  "model_directory": "/path/or/bookmark",
  "served_model_name": "optional-api-name",
  "model_key": "stable-cache-key",
  "server_settings": "VMLXServerRuntimeSettings JSON",
  "capability_snapshot": "ModelRuntimeCapabilitySnapshot JSON",
  "created_at": "host timestamp",
  "last_used_at": "host timestamp"
}
```

Do not save live KV arrays as a session preference. Cache reuse is keyed by
`modelKey`, tokens, and optional `mediaSalt`, and stored by `CacheCoordinator`.

## Stable Cache Keys

`[ENGINE-READY]` `CacheCoordinatorConfig.modelKey` scopes paged, disk, and SSM
hashes. `DiskCache.hashTokens(_:modelKey:mediaSalt:)`,
`SSMStateCache.makeKey(tokens:boundary:mediaSalt:modelKey:)`, and paged block
hashing all include model-specific keying.

`[HOST-WIRE]` Osaurus must build a stable per-model cache key. Recommended
components:

- canonical model id or local bundle id;
- JANG/TurboQuant variant id when relevant;
- weight/config revision or digest if available;
- runtime MoE top-k override if active, matching
  `RuntimeMoETopKOverride.cacheScopedModelKey(_:)`;
- processor/template identity when it can change token/media layout.

Never use only the served API alias as `modelKey`. Two different models can
share an alias over time, and one model can have multiple aliases.

## Symbol Hygiene: Names To Avoid

`[DO NOT USE]` Do not put these stale or foreign-engine names in the spec/API
contract unless a real Swift definition is added:

- `MLXEngine`
- `ServerConfig`
- `prefix_cache`
- `paged_cache`
- `kv_cache_quantization`
- `CacheCoordinatorConfig.prefixCache`
- `CacheCoordinatorConfig.pagedCache`
- `LMInput.Image.pixels`
- `UserInput.Video.asset`
- `Generation.text`
- `Generation.completion`
- `StopReason`
- `generationTime` as a `GenerateCompletionInfo` property
- `public actor ModelContainer`
- `public actor CacheCoordinator`

Use these real Swift names instead:

- `VMLXServerRuntimeSettings`
- `VMLXServerCacheSettings`
- `CacheCoordinatorConfig.usePagedCache`
- `CacheCoordinatorConfig.enableDiskCache`
- `KVQuantizationMode.turboQuant(keyBits:valueBits:)`
- `UserInput.Image.ciImage`, `.url`, `.array`
- `UserInput.Video.avAsset`, `.url`, `.frames`
- `Generation.chunk`
- `Generation.reasoning`
- `Generation.toolCall`
- `Generation.info`
- `GenerateStopReason`
- `GenerateCompletionInfo.generateTime`
- `public final class ModelContainer`
- `public final class CacheCoordinator`

## Build Sequence For Osaurus

Recommended implementation order:

1. Persist `VMLXServerRuntimeSettings` exactly.
2. Add validation UI from `validationIssues(...)`.
3. Add model capability status from `ModelRuntimeCapabilitySnapshot`.
4. Create sessions with a stable `modelKey`.
5. Load a model with `LoadConfiguration` resolved from MTP settings.
6. Enable cache through `settings.cacheCoordinatorConfig(...)` and
   `container.enableCachingAsync(config:)`.
7. Create `BatchEngine` from `maxConcurrentSequences`.
8. Route chat/text requests through `BatchEngine.generate`.
9. Display generation, batch, and cache stats.
10. Add disabled/manual states for topology-incompatible cache controls.
11. Add gateway compatibility snippets only for endpoints that are actually
    mounted by Osaurus.

## Handoff Prompt

Build the Osaurus Server/API panel using current `vmlx-swift` symbols only:
persist `VMLXServerRuntimeSettings`, validate with `validationIssues(...)`
and `validateRequest(...)`, load models into `ModelContainer`, enable cache via
`cacheCoordinatorConfig(modelKey:)` plus `enableCachingAsync(config:)`, create
`BatchEngine` using `maxConcurrentSequences`, stream through
`BatchEngine.generate`, and expose `GenerateCompletionInfo`, `BatchEngine`
diagnostics, and `CacheCoordinator.snapshotStats()`. Defaults for cache,
template, parsers, multimodal, and MTP must be `Auto` and derived from
`ModelConfiguration`, `JangConfig.capabilities`, `ModelRuntimeCapabilitySnapshot`,
and the model's real `newCache(parameters:)` topology. Do not let users pair a
plain paged/prefix/TurboQuant setting with incompatible cache families such as
Mamba/Arrays SSM, ZAYA CCA, `CacheList`, sliding `RotatingKVCache`, DSV4
`HybridPoolCache`, VL media processors, or native MTP. Mark HTTP gateway routes,
saved sessions, lifecycle sleep/wake, API usage accounting, custom template
application, MCP execution, and endpoint catalog as Osaurus host work unless a
real vMLX Swift consumer exists. Do not list image/audio generation,
embeddings, rerank, or other endpoints as ready unless their routes and engine
paths are actually implemented and verified.
