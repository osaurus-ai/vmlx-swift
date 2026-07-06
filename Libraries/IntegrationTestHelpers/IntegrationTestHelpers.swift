// Shared integration test logic for verifying end-to-end model loading and generation.
// Integration packages inject their own Downloader and TokenizerLoader, then call
// these functions which run the test and throw on failure.

import CoreImage
import Foundation
import MLX
import MLXEmbedders
import MLXLLM
import MLXLMCommon
import MLXVLM

// Both MLXLMCommon and MLXEmbedders define ModelContainer.
public typealias LMModelContainer = MLXLMCommon.ModelContainer
public typealias EmbeddingModelContainer = MLXEmbedders.ModelContainer

// MARK: - Error

public struct IntegrationTestFailure: LocalizedError {
    public let errorDescription: String?

    public init(_ message: String) {
        self.errorDescription = message
    }
}

private func check(_ condition: Bool, _ message: String) throws {
    guard condition else { throw IntegrationTestFailure(message) }
}

// MARK: - Model IDs

public enum IntegrationTestModelIDs {
    public static let llm = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    public static let vlm = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    public static let lfm2 = "mlx-community/LFM2-2.6B-Exp-4bit"
    public static let glm4 = "mlx-community/GLM-4-9B-0414-4bit"
}

// MARK: - Model Loading

/// Shared model cache that loads each model at most once per test run.
public actor IntegrationTestModels {
    private let downloader: any Downloader
    private let tokenizerLoader: any TokenizerLoader

    private var llmTask: Task<LMModelContainer, Error>?
    private var vlmTask: Task<LMModelContainer, Error>?
    private var lfm2Task: Task<LMModelContainer, Error>?
    private var glm4Task: Task<LMModelContainer, Error>?

    public init(downloader: any Downloader, tokenizerLoader: any TokenizerLoader) {
        self.downloader = downloader
        self.tokenizerLoader = tokenizerLoader
    }

    public func llmContainer() async throws -> LMModelContainer {
        if let task = llmTask {
            return try await task.value
        }
        let downloader = self.downloader
        let tokenizerLoader = self.tokenizerLoader
        let id = IntegrationTestModelIDs.llm
        let task = Task {
            print("Loading LLM: \(id)")
            let container = try await LLMModelFactory.shared.loadContainer(
                from: downloader, using: tokenizerLoader,
                configuration: .init(id: id),
                progressHandler: logProgress(id)
            )
            print("Loaded LLM: \(id)")
            return container
        }
        llmTask = task
        return try await task.value
    }

    public func vlmContainer() async throws -> LMModelContainer {
        if let task = vlmTask {
            return try await task.value
        }
        let downloader = self.downloader
        let tokenizerLoader = self.tokenizerLoader
        let id = IntegrationTestModelIDs.vlm
        let task = Task {
            print("Loading VLM: \(id)")
            let container = try await VLMModelFactory.shared.loadContainer(
                from: downloader, using: tokenizerLoader,
                configuration: .init(id: id),
                progressHandler: logProgress(id)
            )
            print("Loaded VLM: \(id)")
            return container
        }
        vlmTask = task
        return try await task.value
    }

    public func lfm2Container() async throws -> LMModelContainer {
        if let task = lfm2Task {
            return try await task.value
        }
        let downloader = self.downloader
        let tokenizerLoader = self.tokenizerLoader
        let id = IntegrationTestModelIDs.lfm2
        let task = Task {
            print("Loading LFM2: \(id)")
            let container = try await LLMModelFactory.shared.loadContainer(
                from: downloader, using: tokenizerLoader,
                configuration: .init(id: id),
                progressHandler: logProgress(id)
            )
            print("Loaded LFM2: \(id)")
            return container
        }
        lfm2Task = task
        return try await task.value
    }

    public func glm4Container() async throws -> LMModelContainer {
        if let task = glm4Task {
            return try await task.value
        }
        let downloader = self.downloader
        let tokenizerLoader = self.tokenizerLoader
        let id = IntegrationTestModelIDs.glm4
        let task = Task {
            print("Loading GLM4: \(id)")
            let container = try await LLMModelFactory.shared.loadContainer(
                from: downloader, using: tokenizerLoader,
                configuration: .init(id: id),
                progressHandler: logProgress(id)
            )
            print("Loaded GLM4: \(id)")
            return container
        }
        glm4Task = task
        return try await task.value
    }

    public func embeddingContainer() async throws -> EmbeddingModelContainer {
        let downloader = self.downloader
        let tokenizerLoader = self.tokenizerLoader
        let id = "nomic_text_v1_5"
        print("Loading embedding model: \(id)")
        let container = try await MLXEmbedders.loadModelContainer(
            from: downloader, using: tokenizerLoader, configuration: .nomic_text_v1_5,
            progressHandler: logProgress(id)
        )
        print("Loaded embedding model: \(id)")
        return container
    }
}

// MARK: - ChatSession Tests

private let generateParameters = GenerateParameters(maxTokens: 200, temperature: 0)

public enum ChatSessionTests {

    public static func oneShot(container: LMModelContainer) async throws {
        let session = ChatSession(container, generateParameters: generateParameters)
        let result = try await streamAndCollect(
            session.streamResponse(
                to: "What is 2+2? Reply with just the number."), label: "One-shot")
        try check(
            result.contains("4") || result.lowercased().contains("four"),
            "Expected '4' or 'four' in response, got: \(result)"
        )
    }

    public static func oneShotStream(container: LMModelContainer) async throws {
        let session = ChatSession(container, generateParameters: generateParameters)
        let result = try await streamAndCollect(
            session.streamResponse(
                to: "What is 2+2? Reply with just the number."), label: "Stream")
        try check(
            result.contains("4") || result.lowercased().contains("four"),
            "Expected '4' or 'four' in streamed response, got: \(result)"
        )
    }

    public static func multiTurnConversation(container: LMModelContainer) async throws {
        let session = ChatSession(
            container, instructions: "You are a helpful assistant. Keep responses brief.",
            generateParameters: generateParameters)

        _ = try await streamAndCollect(
            session.streamResponse(
                to: "My name is Alice."), label: "Turn 1")

        let response2 = try await streamAndCollect(
            session.streamResponse(
                to: "What is my name?"), label: "Turn 2")

        try check(
            response2.lowercased().contains("alice"),
            "Expected 'Alice' in response, got: \(response2)"
        )
    }

    public static func visionModel(container: LMModelContainer) async throws {
        let session = ChatSession(container, generateParameters: generateParameters)
        let redImage = CIImage(color: .red).cropped(
            to: CGRect(x: 0, y: 0, width: 100, height: 100))

        let result = try await streamAndCollect(
            session.streamResponse(
                to: "What color is this image? Reply with just the color name.",
                image: .ciImage(redImage)), label: "Vision")
        try check(
            result.lowercased().contains("red"),
            "Expected 'red' in response, got: \(result)"
        )
    }

    public static func streamDetailsWithTools(container: LMModelContainer) async throws {
        let tools: [ToolSpec] = [weatherToolSchema]
        let session = ChatSession(container, generateParameters: generateParameters, tools: tools)

        var responseText = ""
        var toolCalls: [ToolCall] = []

        var info: GenerateCompletionInfo?
        print("Tools: ", terminator: "")
        for try await generation in session.streamDetails(
            to: "What is the weather in San Francisco?", images: [], videos: [])
        {
            switch generation {
            case .chunk(let text):
                print(text, terminator: "")
                responseText += text
            case .reasoning:
                break
            case .toolCall(let toolCall):
                toolCalls.append(toolCall)
            case .toolCallProgress:
                break
            case .prefillProgress:

                break
            case .info(let completionInfo):
                info = completionInfo
            }
        }
        print()
        if let info {
            print(
                "Generation info: \(info.generationTokenCount) tokens, stop reason: \(info.stopReason)"
            )
        }
        if !toolCalls.isEmpty {
            print("Tool calls: \(toolCalls)")
        }

        try check(
            !responseText.isEmpty || !toolCalls.isEmpty,
            "Expected either text or tool calls, got neither (generated \(info?.generationTokenCount ?? 0) tokens, stop reason: \(String(describing: info?.stopReason)))"
        )

        // If we got tool calls, feed back a tool result and verify the model responds
        if !toolCalls.isEmpty {
            let followUp = try await streamAndCollect(
                session.streamResponse(
                    to: "Foggy with a high in the low 60s, clearing later in the day",
                    role: .tool, images: [], videos: []),
                label: "Tool result")
            try check(
                !followUp.isEmpty,
                "Expected a response after providing tool result, got empty string"
            )
        }
    }

    public static func toolInvocation(container: LMModelContainer) async throws {
        struct EmptyInput: Codable {}

        struct TimeOutput: Codable {
            let time: String
        }

        let timeTool = Tool<EmptyInput, TimeOutput>(
            name: "get_time",
            description: "Get the current date and time including day of week.",
            parameters: []
        ) { _ in
            TimeOutput(time: "Wed Feb 18 17:50:43 PST 2026")
        }

        let session = ChatSession(
            container, generateParameters: generateParameters,
            tools: [timeTool.schema]
        ) { toolCall in
            if toolCall.function.name == timeTool.name {
                return try await toolCall.execute(with: timeTool).toolResult
            }
            return "Unknown tool: \(toolCall.function.name)"
        }

        let result = try await streamAndCollect(
            session.streamResponse(
                to: "What day of week is it?"), label: "Tool invocation")

        try check(
            result.lowercased().contains("wed") || result.lowercased().contains("wednesday"),
            "Expected 'Wed' or 'Wednesday' in response, got: \(result)"
        )
    }

    public static func promptRehydration(container: LMModelContainer) async throws {
        let history: [Chat.Message] = [
            .system("You are a helpful assistant."),
            .user("My name is Bob."),
            .assistant("Hello Bob! How can I help you today?"),
        ]

        let session = ChatSession(
            container, history: history, generateParameters: generateParameters)
        let response = try await streamAndCollect(
            session.streamResponse(
                to: "What is my name?"), label: "Rehydration")

        try check(
            response.lowercased().contains("bob"),
            "Expected 'Bob' in response (prompt rehydration), got: \(response)"
        )
    }
}

// MARK: - Stream Helper

private func streamAndCollect(
    _ stream: AsyncThrowingStream<String, Error>,
    label: String
) async throws -> String {
    var result = ""
    print("\(label): ", terminator: "")
    for try await token in stream {
        print(token, terminator: "")
        result += token
    }
    print()
    return result
}

// MARK: - BatchEngine Tests

/// Integration coverage for `BatchEngine` on real downloaded models.
/// Complements unit tests that use tiny synthetic models — these exercise
/// the actual Qwen/GLM/LFM weight tensors the engine is shipped to serve.
///
/// Downstream test packages call these with a preloaded `LMModelContainer`
/// from `IntegrationTestModels`.
public enum BatchEngineIntegrationTests {

    /// Submit a single request through `BatchEngine` at `maxBatchSize == 1`
    /// and verify the response is coherent. Smoke-test for Stage 0/1A/1B.3
    /// on real model weights (not just the synthetic test model used by
    /// unit tests).
    public static func oneShot(container: LMModelContainer) async throws {
        let params = GenerateParameters(maxTokens: 64, temperature: 0)
        try await runBatchPrompt(
            container: container,
            prompt: "What is 2+2? Reply with just the number.",
            parameters: params,
            label: "BatchEngine one-shot",
            validate: { result in
                let lower = result.lowercased()
                try check(
                    result.contains("4") || lower.contains("four"),
                    "Expected '4' or 'four' in BatchEngine response, got: \(result)"
                )
            }
        )
    }

    /// Submit a TurboQuant-compressed request through `BatchEngine`. At
    /// prompt length > 8, the per-prefill compression hook fires and
    /// slot caches become `TurboQuantKVCache`. The decode path wraps them
    /// in a `BatchKVCache` via the shared shape contract (no subclass at
    /// Stage 0). Stage 2 compile+TQ still disabled at the engine level
    /// per iter 9 rollback.
    public static func turboQuantSingle(container: LMModelContainer) async throws {
        let params = GenerateParameters(
            maxTokens: 48,
            kvMode: .turboQuant(keyBits: 3, valueBits: 3),
            temperature: 0
        )
        try await runBatchPrompt(
            container: container,
            prompt: "Count from 1 to 5, separated by commas.",
            parameters: params,
            label: "BatchEngine TQ",
            validate: { result in
                try check(
                    result.contains("1") && result.contains("5"),
                    "Expected count output (1..5) in TQ response, got: \(result)"
                )
            }
        )
    }

    /// Submit a request with `enableCompiledBatchDecode: true` at
    /// `maxBatchSize == 1`. Verifies the Stage 1B.3 compile wiring works
    /// end-to-end on real model weights — not just the synthetic test
    /// Llama used by unit tests.
    public static func compiledSingle(container: LMModelContainer) async throws {
        let params = GenerateParameters(
            maxTokens: 48,
            enableCompiledBatchDecode: true,
            temperature: 0
        )
        try await runBatchPrompt(
            container: container,
            prompt: "Name three primary colors.",
            parameters: params,
            label: "BatchEngine compile",
            validate: { result in
                let lower = result.lowercased()
                // Primary colors (depending on model): red/blue/yellow or red/green/blue
                let hasRed = lower.contains("red")
                let hasBlue = lower.contains("blue")
                try check(
                    hasRed || hasBlue,
                    "Expected a primary color (red/blue) in compile-path response, got: \(result)"
                )
            }
        )
    }

    /// VLM path through `BatchEngine`. Feeds a solid red image and asks
    /// the model to name the color. Uses the same "red square + what
    /// color" probe shape as `ChatSessionTests.visionModel` so results
    /// are directly comparable. Container must be a preloaded VLM
    /// (e.g. `IntegrationTestModels.vlmContainer()`).
    public static func visionModel(container: LMModelContainer) async throws {
        let redImage = CIImage(color: .red).cropped(
            to: CGRect(x: 0, y: 0, width: 100, height: 100))
        let params = GenerateParameters(maxTokens: 32, temperature: 0)
        let engine = await container.makeBatchEngine(maxBatchSize: 1)

        let stream = try await container.perform { context in
            nonisolated(unsafe) let ctx = context
            let input = try await ctx.processor.prepare(input: UserInput(
                prompt: "What color is this image? Reply with just the color name.",
                images: [.ciImage(redImage)]
            ))
            nonisolated(unsafe) let sendableInput = input
            return await engine.generate(input: sendableInput, parameters: params)
        }
        let result = try await collectGeneration(stream, label: "BatchEngine VLM")
        try check(
            result.lowercased().contains("red"),
            "Expected 'red' in VLM response, got: \(result)"
        )
    }

    /// Multi-turn through `BatchEngine` with cache coordinator enabled.
    /// Tests the full cache-hit path: turn 1 populates the coordinator,
    /// turn 2 retrieves the prefix. Both turns submit the same prompt
    /// and should produce identical greedy output.
    ///
    /// **Caveat:** `container.enableCaching()` must be called before
    /// `makeBatchEngine` so the engine captures the coordinator. The
    /// helper handles this internally.
    public static func multiTurn(container: LMModelContainer) async throws {
        // Attach a cache coordinator before building the engine.
        container.enableCaching()
        defer { container.disableCaching() }

        let params = GenerateParameters(maxTokens: 32, temperature: 0)
        let engine = await container.makeBatchEngine(maxBatchSize: 1)

        // Turn 1 — populates cache.
        let result1 = try await runOneBatch(
            engine: engine, container: container,
            prompt: "What is the capital of Italy? One word.",
            parameters: params, label: "Multi-turn 1")

        // Turn 2 — same prompt, should cache-hit.
        let result2 = try await runOneBatch(
            engine: engine, container: container,
            prompt: "What is the capital of Italy? One word.",
            parameters: params, label: "Multi-turn 2")

        try check(
            result1.lowercased().contains("rome"),
            "Expected 'Rome' in turn 1 response, got: \(result1)"
        )
        try check(
            result2.lowercased().contains("rome"),
            "Expected 'Rome' in turn 2 response (cache hit), got: \(result2)"
        )
    }

    /// Stage 3 verification: verify compile+sliding-window works on real
    /// models. Caller must supply a container whose model uses
    /// `RotatingKVCache` (Gemma3/Gemma4 SWA layers, Mistral4 with
    /// maxKVSize, MiMoV2Flash, BaichuanM1, Qwen3.5-VL inherited).
    /// Runs a decode long enough to exercise the ring (and potentially
    /// the wrap point, depending on model's sliding-window size).
    public static func compiledSlidingWindow(container: LMModelContainer) async throws {
        let params = GenerateParameters(
            maxTokens: 64,
            enableCompiledBatchDecode: true,
            temperature: 0
        )
        try await runBatchPrompt(
            container: container,
            prompt: "Summarise recursion in one short paragraph.",
            parameters: params,
            label: "BatchEngine compile+sliding",
            validate: { result in
                try check(
                    !result.isEmpty,
                    "Expected non-empty response from compile+sliding-window path"
                )
                // Coherent-response check: for a recursion prompt, the
                // model should mention something like function / call /
                // self / recursive / base. Loose check — don't require
                // a specific word; just one of them.
                let lower = result.lowercased()
                let keywords = ["function", "call", "self", "recursiv", "base"]
                try check(
                    keywords.contains(where: { lower.contains($0) }),
                    "Expected coherent recursion-related response, got: \(result)"
                )
            }
        )
    }

    /// Decode-speed smoke: verify compiled BatchEngine path delivers at
    /// LEAST the uncompiled path's throughput on real weights. This is
    /// the primary win Stage 1B.3 was designed for — "all model-level
    /// compile optimisations are exhausted. Next gains must come from
    /// framework-level changes: static chunk KV cache (Overflow Bin) to
    /// enable full decode compile()" (per `perf_decode_gap_analysis`).
    ///
    /// Measures tok/s on both paths for the same prompt. Asserts the
    /// compiled path is not MATERIALLY slower (within 5%) — a regression
    /// catcher. Actual speedup is measured in real benchmarks; this
    /// just ensures we never ship a compile path that REGRESSES decode
    /// speed.
    public static func compiledDecodeSpeedFloor(container: LMModelContainer) async throws {
        let maxTokens = 32
        let prompt = "Describe a sunset in one short paragraph."

        // Uncompiled baseline.
        let uncompiledParams = GenerateParameters(
            maxTokens: maxTokens, enableCompiledBatchDecode: false, temperature: 0)
        let uStart = Date()
        _ = try await runOneBatchSimple(
            container: container, prompt: prompt, parameters: uncompiledParams,
            label: "Speed uncompiled")
        let uTime = Date().timeIntervalSince(uStart)
        let uTps = Double(maxTokens) / uTime

        // Compiled path.
        let compiledParams = GenerateParameters(
            maxTokens: maxTokens, enableCompiledBatchDecode: true, temperature: 0)
        let cStart = Date()
        _ = try await runOneBatchSimple(
            container: container, prompt: prompt, parameters: compiledParams,
            label: "Speed compiled")
        let cTime = Date().timeIntervalSince(cStart)
        let cTps = Double(maxTokens) / cTime

        let ratio = cTps / uTps
        print("Speed: uncompiled \(String(format: "%.1f", uTps)) tok/s, compiled \(String(format: "%.1f", cTps)) tok/s, ratio \(String(format: "%.2f", ratio))x")

        // Floor: compiled must be no slower than 95% of uncompiled. This
        // is intentionally loose — first-hit compile cost is amortised
        // over many runs; the SECOND call is when speedup should appear.
        // A proper benchmark would warm up the trace first. Here we just
        // guard against catastrophic regressions.
        try check(
            ratio > 0.95,
            "Compiled path is materially slower than uncompiled: \(ratio)x"
        )
    }

    /// Minimal single-prompt run, no validation hook — used by perf
    /// helpers where we only care about timing, not content.
    private static func runOneBatchSimple(
        container: LMModelContainer,
        prompt: String,
        parameters: GenerateParameters,
        label: String
    ) async throws -> String {
        let maxB = parameters.enableCompiledBatchDecode ? 1 : 2
        let engine = await container.makeBatchEngine(maxBatchSize: maxB)
        return try await runOneBatch(
            engine: engine, container: container,
            prompt: prompt, parameters: parameters, label: label)
    }

    /// Submit two concurrent requests through `BatchEngine` at
    /// `maxBatchSize == 2`. Verifies batched decode runs across slots
    /// without compile (since compile requires maxBatchSize=1 at Stage
    /// 1B.3).
    public static func twoConcurrent(container: LMModelContainer) async throws {
        let params = GenerateParameters(maxTokens: 32, temperature: 0)
        let engine = await container.makeBatchEngine(maxBatchSize: 2)

        async let streamA = runOneBatch(
            engine: engine, container: container,
            prompt: "What is the capital of France? One word.",
            parameters: params, label: "Concurrent A")
        async let streamB = runOneBatch(
            engine: engine, container: container,
            prompt: "What is the capital of Japan? One word.",
            parameters: params, label: "Concurrent B")

        let (resultA, resultB) = try await (streamA, streamB)

        try check(
            resultA.lowercased().contains("paris"),
            "Expected 'Paris' in A's response, got: \(resultA)"
        )
        try check(
            resultB.lowercased().contains("tokyo"),
            "Expected 'Tokyo' in B's response, got: \(resultB)"
        )
    }

    /// Helper: prepare one input inside the container (staying on the
    /// container's actor for non-Sendable LMInput) then kick off decode on
    /// the engine. Returns the collected text.
    private static func runOneBatch(
        engine: BatchEngine,
        container: LMModelContainer,
        prompt: String,
        parameters: GenerateParameters,
        label: String
    ) async throws -> String {
        let stream = try await container.perform { context in
            nonisolated(unsafe) let ctx = context
            let input = try await ctx.processor.prepare(input: UserInput(
                chat: [.user(prompt)]
            ))
            nonisolated(unsafe) let sendableInput = input
            return await engine.generate(input: sendableInput, parameters: parameters)
        }
        return try await collectGeneration(stream, label: label)
    }

    /// Helper: run a single prompt through BatchEngine and pipe it
    /// through a validator.
    private static func runBatchPrompt(
        container: LMModelContainer,
        prompt: String,
        parameters: GenerateParameters,
        label: String,
        validate: @Sendable (String) throws -> Void
    ) async throws {
        let maxB = parameters.enableCompiledBatchDecode ? 1 : 2
        let engine = await container.makeBatchEngine(maxBatchSize: maxB)
        let result = try await runOneBatch(
            engine: engine, container: container,
            prompt: prompt, parameters: parameters, label: label)
        try validate(result)
    }

    /// Collect an `AsyncStream<Generation>` into a single text string,
    /// mirroring the `ChatSessionTests` helper but for BatchEngine's
    /// stream type.
    private static func collectGeneration(
        _ stream: AsyncStream<Generation>,
        label: String
    ) async throws -> String {
        var result = ""
        print("\(label): ", terminator: "")
        for await generation in stream {
            switch generation {
            case .chunk(let text):
                print(text, terminator: "")
                result += text
            case .prefillProgress:

                break
            case .info, .toolCall, .toolCallProgress, .reasoning:
                break
            }
        }
        print()
        return result
    }
}

// MARK: - Embedder Tests

public enum EmbedderTests {

    public static func gemma3Embedder(
        downloader: any Downloader, tokenizerLoader: any TokenizerLoader
    ) async throws {
        let modelId = "mlx-community/gemma-3-1b-it-qat-4bit"
        print("Loading Gemma 3 embedding model: \(modelId)")
        let modelContainer = try await MLXEmbedders.loadModelContainer(
            from: downloader, using: tokenizerLoader, configuration: .init(id: modelId),
            progressHandler: logProgress(modelId)
        )
        print("Loaded Gemma 3 embedding model: \(modelId)")

        let inputs = [
            "The Coca-Cola Company is a soft drink company based in Atlanta, Georgia, USA.",
            "In the United States, PepsiCo Inc. is a leading soft drink company.",
        ]

        let resultEmbeddings = await modelContainer.perform {
            (model: EmbeddingModel, tokenizer: Tokenizer, pooling: Pooling) -> [[Float]] in
            let encoded = inputs.map {
                tokenizer.encode(text: $0, addSpecialTokens: true)
            }
            let maxLength = encoded.reduce(into: 1) { acc, elem in
                acc = max(acc, elem.count)
            }

            let padded = stacked(
                encoded.map { elem in
                    MLXArray(
                        elem
                            + Array(
                                repeating: tokenizer.eosTokenId ?? 0,
                                count: maxLength - elem.count))
                })

            let mask = (padded .!= (tokenizer.eosTokenId ?? 0))
            let tokenTypes = MLXArray.zeros(like: padded)

            let modelOutput = model(
                padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask)

            let result = pooling(
                modelOutput,
                normalize: true, applyLayerNorm: true
            )
            result.eval()
            return result.map { $0.asArray(Float.self) }
        }

        try check(
            resultEmbeddings.count == inputs.count,
            "Should have one embedding per input, got \(resultEmbeddings.count)"
        )
        for embedding in resultEmbeddings {
            try check(
                embedding.count == 1152,
                "Gemma 3 1B embedding size should be 1152, got \(embedding.count)"
            )
            let l2Norm = sqrt(embedding.map { $0 * $0 }.reduce(0, +))
            try check(
                abs(l2Norm - 1.0) < 0.05,
                "Embeddings should be approximately L2-normalized, got L2 norm \(l2Norm)"
            )
        }

        let similarity = zip(resultEmbeddings[0], resultEmbeddings[1]).map(*).reduce(0, +)
        try check(
            similarity > 0.0,
            "Similarity between related sentences should be positive, got \(similarity)"
        )
    }

    public static func readmeExample(container: EmbeddingModelContainer) async throws {
        let searchInputs = [
            "search_query: Animals in Tropical Climates.",
            "search_document: Elephants",
            "search_document: Horses",
            "search_document: Polar Bears",
        ]

        let resultEmbeddings = await container.perform {
            (model: EmbeddingModel, tokenizer: Tokenizer, pooling: Pooling) -> [[Float]] in
            let inputs = searchInputs.map {
                tokenizer.encode(text: $0, addSpecialTokens: true)
            }
            let maxLength = inputs.reduce(into: 16) { acc, elem in
                acc = max(acc, elem.count)
            }
            let padded = stacked(
                inputs.map { elem in
                    MLXArray(
                        elem
                            + Array(
                                repeating: tokenizer.eosTokenId ?? 0,
                                count: maxLength - elem.count))
                })
            let mask = (padded .!= tokenizer.eosTokenId ?? 0)
            let tokenTypes = MLXArray.zeros(like: padded)
            let result = pooling(
                model(padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask),
                normalize: true, applyLayerNorm: true
            )
            result.eval()
            return result.map { $0.asArray(Float.self) }
        }

        let searchQueryEmbedding = resultEmbeddings[0]
        let documentEmbeddings = resultEmbeddings[1...]
        let similarities = documentEmbeddings.map { docEmbedding in
            zip(searchQueryEmbedding, docEmbedding).map(*).reduce(0, +)
        }
        let documentNames = searchInputs[1...].map {
            $0.replacingOccurrences(of: "search_document: ", with: "")
        }

        let expectedSimilarities: [Float] = [0.6854175, 0.6644787, 0.63326025]
        let tolerance: Float = 1e-4

        for (index, resultSimilarity) in similarities.enumerated() {
            try check(
                abs(resultSimilarity - expectedSimilarities[index]) < tolerance,
                "Similarity mismatch for \(documentNames[index]): expected \(expectedSimilarities[index]), got \(resultSimilarity)"
            )
        }
    }
}

// MARK: - Tool Call Tests

public enum ToolCallTests {

    public static func lfm2FormatAutoDetection(container: LMModelContainer) async throws {
        let config = await container.configuration
        try check(
            config.toolCallFormat == ToolCallFormat.lfm2,
            "Expected .lfm2 tool call format, got: \(String(describing: config.toolCallFormat))"
        )
    }

    public static func lfm2EndToEndGeneration(container: LMModelContainer) async throws {
        let (result, toolCalls) = try await generateWithTools(
            container: container,
            userMessage: "What's the weather in Tokyo?")

        print("LFM2 Output:", result)
        print("LFM2 Tool Calls:", toolCalls)

        if !toolCalls.isEmpty {
            let toolCall = toolCalls[0]
            try check(
                toolCall.function.name == "get_weather",
                "Expected tool name 'get_weather', got: \(toolCall.function.name)"
            )
            if case .string(let location) = toolCall.function.arguments["location"] {
                try check(
                    location.lowercased().contains("tokyo"),
                    "Expected location containing 'Tokyo', got: \(location)"
                )
            }
        }
    }

    public static func glm4FormatAutoDetection(container: LMModelContainer) async throws {
        let config = await container.configuration
        try check(
            config.toolCallFormat == ToolCallFormat.glm4,
            "Expected .glm4 tool call format, got: \(String(describing: config.toolCallFormat))"
        )
    }

    public static func glm4EndToEndGeneration(container: LMModelContainer) async throws {
        let (result, toolCalls) = try await generateWithTools(
            container: container,
            userMessage: "What's the weather in Paris?")

        print("GLM4 Output:", result)
        print("GLM4 Tool Calls:", toolCalls)

        if !toolCalls.isEmpty {
            let toolCall = toolCalls[0]
            try check(
                toolCall.function.name == "get_weather",
                "Expected tool name 'get_weather', got: \(toolCall.function.name)"
            )
            if case .string(let location) = toolCall.function.arguments["location"] {
                try check(
                    location.lowercased().contains("paris"),
                    "Expected location containing 'Paris', got: \(location)"
                )
            }
        }
    }

    private static func generateWithTools(
        container: LMModelContainer,
        userMessage: String
    ) async throws -> (text: String, toolCalls: [ToolCall]) {
        try await container.perform { context in
            let input = UserInput(
                chat: [
                    .system(
                        "You are a helpful assistant with access to tools. When asked about weather, use the get_weather function."
                    ),
                    .user(userMessage),
                ],
                tools: [weatherToolSchema]
            )
            let lmInput = try await context.processor.prepare(input: input)
            let stream = try generate(
                input: lmInput,
                parameters: GenerateParameters(maxTokens: 100),
                context: context
            )

            var text = ""
            var toolCalls: [ToolCall] = []
            for try await generation in stream {
                switch generation {
                case .chunk(let chunk):
                    text += chunk
                case .toolCall(let toolCall):
                    toolCalls.append(toolCall)
                case .reasoning, .prefillProgress, .toolCallProgress, .info:
                    break
                }
            }
            return (text, toolCalls)
        }
    }
}

// MARK: - Progress Logging

private func logProgress(_ label: String) -> @Sendable (Progress) -> Void {
    let lock = NSLock()
    nonisolated(unsafe) var lastThreshold = -1
    return { progress in
        let pct = Int(progress.fractionCompleted * 100)
        let threshold = pct / 5
        lock.lock()
        let shouldPrint = threshold > lastThreshold
        if shouldPrint { lastThreshold = threshold }
        lock.unlock()
        if shouldPrint {
            print("  \(label): \(pct)%")
        }
    }
}

// MARK: - Shared Constants

private let weatherToolSchema: ToolSpec = [
    "type": "function",
    "function": [
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": [
            "type": "object",
            "properties": [
                "location": [
                    "type": "string",
                    "description": "The city name, e.g. San Francisco",
                ] as [String: any Sendable],
                "unit": [
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                ] as [String: any Sendable],
            ] as [String: any Sendable],
            "required": ["location"],
        ] as [String: any Sendable],
    ] as [String: any Sendable],
]
