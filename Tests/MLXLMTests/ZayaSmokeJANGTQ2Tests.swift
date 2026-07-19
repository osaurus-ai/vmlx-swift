// ZAYA1-8B-JANGTQ2 smoke (gated on bundle present at the canonical path).
// Verifies the bundle loads through LLMModelFactory.shared and a single
// forward pass produces logits of the expected shape.

import BenchmarkHelpers
import Foundation
import MLX
@preconcurrency import VMLXTokenizers
@testable import MLXHuggingFace
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("ZAYA1-8B JANGTQ2 smoke", .serialized)
struct ZayaSmokeJANGTQ2Tests {
    private struct TracingTokenizerLoader: TokenizerLoader {
        let base: any TokenizerLoader

        func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
            let tokenizerURL = directory.appendingPathComponent("tokenizer.json")
            print(
                "ZAYA_TOKENIZER_LOAD directory=\(directory.path) tokenizerExists=\(FileManager.default.fileExists(atPath: tokenizerURL.path))"
            )
            return try await base.load(from: directory)
        }
    }
    static let bundleRoot: String = {
        if let override = ProcessInfo.processInfo.environment["VMLX_ZAYA_BUNDLE_ROOT"],
           !override.isEmpty {
            return override
        }
        let home = FileManager.default.homeDirectoryForCurrentUser
        let candidates = [
            home.appendingPathComponent("jang/models/Zyphra"),
            home.appendingPathComponent("models/Zyphra"),
        ]
        return candidates.first {
            FileManager.default.fileExists(
                atPath: $0.appendingPathComponent("ZAYA1-8B-JANGTQ2/config.json").path)
        }?.path ?? candidates[0].path
    }()
    static let bundlePath: String = {
        if let exact = ProcessInfo.processInfo.environment["VMLX_ZAYA_EXACT_BUNDLE"],
           !exact.isEmpty {
            return exact
        }
        return bundleRoot + "/ZAYA1-8B-JANGTQ2"
    }()
    static let jangTQ4BundlePath = bundleRoot + "/ZAYA1-8B-JANGTQ4"
    static let mxfp4BundlePath = bundleRoot + "/ZAYA1-8B-MXFP4"
    static let bf16BundlePath = bundleRoot + "/ZAYA1-8B"

    static var bundlePresent: Bool {
        FileManager.default.fileExists(atPath: bundlePath + "/config.json")
    }
    static var jangTQ4BundlePresent: Bool {
        FileManager.default.fileExists(atPath: jangTQ4BundlePath + "/config.json")
    }
    static var mxfp4BundlePresent: Bool {
        FileManager.default.fileExists(atPath: mxfp4BundlePath + "/config.json")
    }
    static var bf16BundlePresent: Bool {
        FileManager.default.fileExists(atPath: bf16BundlePath + "/config.json")
    }
    static var selectiveTQCacheGateEnabled: Bool {
        bundlePresent &&
            ProcessInfo.processInfo.environment["VMLX_ZAYA_SELECTIVE_TQ_LIVE"] == "1"
    }

    @Test("Bundle loads through factory and forward returns [1,T,vocab]",
          .enabled(if: ZayaSmokeJANGTQ2Tests.bundlePresent))
    func loadAndForward() async throws {
        try await Self.loadAndForward(bundlePath: Self.bundlePath)
    }

    @Test("Bundle binds real ZAYA layer parameters",
          .enabled(if: ZayaSmokeJANGTQ2Tests.bundlePresent))
    func bindsRealLayerParameters() async throws {
        let url = URL(fileURLWithPath: Self.bundlePath)
        let factory = LLMModelFactory.shared
        let container = try await factory.loadContainer(from: url, using: NoOpTokenizerLoader())

        let probe: (hasTemp: Bool, temp0: Float, hasRouter: Bool, hasExpert: Bool) =
            await container.perform { context in
                let flat = Dictionary(uniqueKeysWithValues: context.model.parameters().flattened())
                let tempKey = "model.layers.0.sub.qkv.temp"
                let routerKey = "model.layers.1.sub.router.rmsnorm_eda.weight"
                let expertKey = "model.layers.1.sub.experts.switch_mlp.gate_proj.tq_packed"
                let temp = flat[tempKey]
                let temp0 = temp?[0].item(Float.self) ?? 0
                return (temp != nil, temp0, flat[routerKey] != nil, flat[expertKey] != nil)
            }

        #expect(probe.hasTemp)
        #expect(abs(probe.temp0 - 11.625) < 0.001)
        #expect(probe.hasRouter)
        #expect(probe.hasExpert)
    }

    @Test("Factory stamps ZAYA parser capabilities from jang_config",
          .enabled(if: ZayaSmokeJANGTQ2Tests.bundlePresent))
    func stampsZayaParserCapabilities() async throws {
        let url = URL(fileURLWithPath: Self.bundlePath)
        let factory = LLMModelFactory.shared
        let container = try await factory.loadContainer(from: url, using: NoOpTokenizerLoader())

        let stamps: (tool: ToolCallFormat?, reasoning: String?) =
            await container.perform { context in
                (context.configuration.toolCallFormat, context.configuration.reasoningParserName)
            }

        #expect(stamps.tool == .zayaXml)
        #expect(stamps.reasoning == "qwen3")
    }

    @Test("Converted ZAYA metadata supports opt-in reasoning while defaulting prompts off",
          .enabled(if: ZayaSmokeJANGTQ2Tests.bundlePresent))
    func convertedMetadataSupportsOptInReasoningButDefaultsOff() throws {
        for path in [Self.bundlePath, Self.jangTQ4BundlePath, Self.mxfp4BundlePath]
            where FileManager.default.fileExists(atPath: path + "/config.json")
        {
            let url = URL(fileURLWithPath: path)
            let jang = try JangLoader.loadConfig(at: url)
            #expect(jang.capabilities?.supportsThinking == true)
            #expect(jang.capabilities?.thinkInTemplate == false)
            #expect(jang.capabilities?.supportsTools == true)
            #expect(jang.capabilities?.toolParser == "zaya_xml")
            #expect(jang.capabilities?.reasoningParser == "qwen3")

            // Raw converted bundles should be restamped at the source.
            // Runtime code deliberately trusts ZAYA capabilities instead
            // of silently repairing them.
            let configURL = url.appendingPathComponent("config.json")
            let configData = try Data(contentsOf: configURL)
            let config = try #require(
                JSONSerialization.jsonObject(with: configData) as? [String: Any])
            let caps = try #require(config["capabilities"] as? [String: Any])
            #expect(caps["supports_tools"] as? Bool == true)
            #expect(caps["tool_parser"] as? String == "zaya_xml")
            #expect(caps["reasoning_parser"] as? String == "qwen3")
        }
    }

    @Test("ZAYA input processor defaults thinking off from metadata",
          .enabled(if: ZayaSmokeJANGTQ2Tests.bundlePresent))
    func inputProcessorDefaultsThinkingOffFromMetadata() async throws {
        let url = URL(fileURLWithPath: Self.bundlePath)
        let context = try await MLXLMCommon.loadModel(
            from: url, using: #huggingFaceTokenizerLoader())

        let defaultInput = try await context.processor.prepare(input: UserInput(
            chat: [.user("Say OK.")]))
        let defaultTokens = defaultInput.text.tokens.reshaped(-1).asArray(Int.self)
        let defaultPrompt = context.tokenizer.decode(
            tokenIds: defaultTokens, skipSpecialTokens: false)
        #expect(defaultPrompt.contains("</think>"))
        #expect(!defaultPrompt.hasSuffix("<think>\n"))

        let explicitThinkingInput = try await context.processor.prepare(input: UserInput(
            chat: [.user("Say OK.")],
            additionalContext: ["enable_thinking": true]))
        let explicitTokens = explicitThinkingInput.text.tokens.reshaped(-1).asArray(Int.self)
        let explicitPrompt = context.tokenizer.decode(
            tokenIds: explicitTokens, skipSpecialTokens: false)
        #expect(explicitPrompt.hasSuffix("<think>\n"))
    }

    @Test("zaya_xml parser extracts Zyphra-wrapped XML function calls",
          .enabled(if: ZayaSmokeJANGTQ2Tests.bundlePresent))
    func zayaXMLParserExtractsToolCall() async throws {
        let parser = ToolCallFormat.zayaXml.createParser()
        let content = """
            <zyphra_tool_call>
            <function=search_web>
            <parameter=query>
            Swift MLX ZAYA
            </parameter>
            </function>
            </zyphra_tool_call>
            """

        let call = try #require(parser.parse(content: content, tools: nil))

        #expect(call.function.name == "search_web")
        #expect(call.function.arguments["query"] == .string("Swift MLX ZAYA"))
    }

    @Test("BatchEngine B=2 chat stream emits isolated visible chunks without thinking markers",
          .enabled(if: ZayaSmokeJANGTQ2Tests.bundlePresent))
    func batchEngineB2ChatStreamNoThinkingLeak() async throws {
        let url = URL(fileURLWithPath: Self.bundlePath)
        let context = try await MLXLMCommon.loadModel(
            from: url, using: #huggingFaceTokenizerLoader())
        nonisolated(unsafe) let ctx = context
        let engine = BatchEngine(context: ctx, maxBatchSize: 2)
        let params = GenerateParameters(maxTokens: 24, temperature: 0, prefillStepSize: 512)

        let input0 = try await context.processor.prepare(input: UserInput(
            chat: [.user("What is the capital of France? Answer in one short sentence.")],
            additionalContext: ["enable_thinking": false]))
        let input1 = try await context.processor.prepare(input: UserInput(
            chat: [.user("List two prime numbers greater than 10.")],
            additionalContext: ["enable_thinking": false]))

        nonisolated(unsafe) let send0 = input0
        nonisolated(unsafe) let send1 = input1
        let stream0 = await engine.generate(input: send0, parameters: params)
        let stream1 = await engine.generate(input: send1, parameters: params)

        async let result0 = Self.collectGeneration(stream0)
        async let result1 = Self.collectGeneration(stream1)
        let results = await [result0, result1]
        await engine.shutdown()

        #expect(results[0].text.contains("Paris"))
        #expect(!results[1].text.isEmpty || !results[1].reasoning.isEmpty)
        for result in results {
            #expect(result.toolCalls == 0)
            #expect(!result.unclosedReasoning)
            #expect(!result.text.contains("<think>"))
            #expect(!result.text.contains("</think>"))
            #expect(!result.text.contains("<|im_end|>"))
            #expect(!result.text.contains("<zyphra_tool_call>"))
        }
    }

    @Test("JANGTQ4 bundle loads through factory and forward returns [1,T,vocab]",
          .enabled(if: ZayaSmokeJANGTQ2Tests.jangTQ4BundlePresent))
    func loadAndForwardJANGTQ4() async throws {
        try await Self.loadAndForward(bundlePath: Self.jangTQ4BundlePath)
    }

    @Test("MXFP4 bundle loads through factory and forward returns [1,T,vocab]",
          .enabled(if: ZayaSmokeJANGTQ2Tests.mxfp4BundlePresent))
    func loadAndForwardMXFP4() async throws {
        try await Self.loadAndForward(bundlePath: Self.mxfp4BundlePath)
    }

    @Test("BF16 source bundle loads through factory and forward returns [1,T,vocab]",
          .enabled(if: ZayaSmokeJANGTQ2Tests.bf16BundlePresent))
    func loadAndForwardBF16() async throws {
        try await Self.loadAndForward(bundlePath: Self.bf16BundlePath)
    }

    private static func loadAndForward(bundlePath: String) async throws {
        let url = URL(fileURLWithPath: bundlePath)
        let factory = LLMModelFactory.shared
        let container = try await factory.loadContainer(from: url, using: NoOpTokenizerLoader())

        let logitsShape: [Int] = await container.perform { context in
            let model = context.model
            let tokens = MLXArray([1, 2, 3, 4]).reshaped([1, 4])
            let cache = model.newCache(parameters: nil)
            let logits = model(tokens, cache: cache)
            // Force materialization through the MLX evaluator.
            MLX.eval(logits)
            return logits.shape
        }

        #expect(logitsShape.count == 3)
        #expect(logitsShape[0] == 1)
        #expect(logitsShape[1] == 4)
        #expect(logitsShape[2] == 262_272)
    }

    @Test("newCache returns 80 entries: even ZayaCCACache, odd explicit MoE placeholder",
          .enabled(if: ZayaSmokeJANGTQ2Tests.bundlePresent))
    func newCacheShape() async throws {
        let url = URL(fileURLWithPath: Self.bundlePath)
        let factory = LLMModelFactory.shared
        let container = try await factory.loadContainer(from: url, using: NoOpTokenizerLoader())
        let layout: (count: Int, evenIsZaya: Bool, oddIsPlaceholder: Bool) =
            await container.perform { context in
                let cache = context.model.newCache(parameters: nil)
                let count = cache.count
                let evenIsZaya = (cache[0] is ZayaCCACache)
                let oddIsPlaceholder = (cache[1] is ZayaMoEPlaceholderCache)
                return (count, evenIsZaya, oddIsPlaceholder)
            }
        #expect(layout.count == 80)
        #expect(layout.evenIsZaya)
        #expect(layout.oddIsPlaceholder)
    }

    @Test(
        "Exact ZAYA bundle uses selective TQ attention KV across partial SSD restore",
        .enabled(if: ZayaSmokeJANGTQ2Tests.selectiveTQCacheGateEnabled))
    func selectiveTQAttentionKVPartialDiskRestore() async throws {
        struct Result {
            var text = ""
            var reasoning = ""
            var toolCalls = 0
            var info: GenerateCompletionInfo?
        }

        let modelURL = URL(fileURLWithPath: Self.bundlePath)
        let diskURL = URL(fileURLWithPath:
            ProcessInfo.processInfo.environment["VMLX_ZAYA_SELECTIVE_TQ_CACHE_DIR"]
                ?? "/tmp/vmlx-zaya-selective-tq-live")
        try? FileManager.default.removeItem(at: diskURL)
        try FileManager.default.createDirectory(
            at: diskURL, withIntermediateDirectories: true)
        let keepCache =
            ProcessInfo.processInfo.environment["VMLX_ZAYA_SELECTIVE_TQ_KEEP_CACHE"] == "1"
        defer {
            if !keepCache {
                try? FileManager.default.removeItem(at: diskURL)
            }
        }

        let context = try await MLXLMCommon.loadModel(
            from: modelURL,
            using: TracingTokenizerLoader(base: #huggingFaceTokenizerLoader()))
        nonisolated(unsafe) let ctx = context

        var params = GenerateParameters(
            generationConfig: context.configuration.generationDefaults,
            fallback: GenerateParameters(maxTokens: 64, prefillStepSize: 512))
        params.maxTokens = 64
        params.prefillStepSize = 512
        params.randomSeed = 47
        let testKVBits = Int(
            ProcessInfo.processInfo.environment["VMLX_ZAYA_SELECTIVE_TQ_BITS"] ?? "3") ?? 3
        let expectEncodedDisk = testKVBits >= 4
        params.kvMode = testKVBits > 0
            ? .turboQuant(keyBits: testKVBits, valueBits: testKVBits)
            : .none
        params.enableCompiledBatchDecode = false
        params.extraStopStrings = ["<END>"]

        let generationPromptSuffix: [Int] = {
            guard let tokenizer = context.tokenizer as? GenerationPromptControllableTokenizer
            else { return [] }
            let dummy: [[String: any Sendable]] = [["role": "user", "content": "x"]]
            guard
                let withGenerationPrompt = try? tokenizer.applyChatTemplate(
                    messages: dummy,
                    tools: nil,
                    additionalContext: nil,
                    addGenerationPrompt: true),
                let withoutGenerationPrompt = try? tokenizer.applyChatTemplate(
                    messages: dummy,
                    tools: nil,
                    additionalContext: nil,
                    addGenerationPrompt: false)
            else { return [] }
            var common = 0
            let limit = min(withGenerationPrompt.count, withoutGenerationPrompt.count)
            while common < limit,
                  withGenerationPrompt[common] == withoutGenerationPrompt[common] {
                common += 1
            }
            return Array(withGenerationPrompt[common...])
        }()
        #expect((1...64).contains(generationPromptSuffix.count))

        func makeCoordinator() -> CacheCoordinator {
            var config = CacheCoordinatorConfig()
            config.usePagedCache = false
            config.enableDiskCache = true
            config.diskCacheDir = diskURL
            config.diskCacheMaxGB = 2
            config.modelKey = "zaya-jang6m-selective-tq-v1"
            let coordinator = CacheCoordinator(config: config)
            coordinator.setHybrid(true)
            coordinator.setPagedIncompatible(true)
            coordinator.setGenPromptSuffixTokens(generationPromptSuffix)
            return coordinator
        }

        func clone(_ input: LMInput) -> LMInput {
            LMInput(
                text: LMInput.Text(
                    tokens: input.text.tokens,
                    mask: input.text.mask,
                    tokenIds: input.text.tokenIds),
                image: input.image,
                video: input.video,
                audio: input.audio,
                mediaTokenIds: input.mediaTokenIds,
                cacheScopeSalt: input.cacheScopeSalt,
                cachePrefixTokenCounts: input.cachePrefixTokenCounts,
                toolSchemas: input.toolSchemas)
        }

        func run(_ engine: BatchEngine, _ input: sending LMInput) async -> Result {
            var result = Result()
            let stream = await engine.generate(input: input, parameters: params)
            for await event in stream {
                switch event {
                case .chunk(let chunk):
                    result.text += chunk
                case .reasoning(let chunk):
                    result.reasoning += chunk
                case .toolCall:
                    result.toolCalls += 1
                case .info(let info):
                    result.info = info
                case .toolCallProgress, .prefillProgress:
                    break
                }
            }
            return result
        }

        func assertClean(_ result: Result, label: String) {
            #expect(!result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                    "\(label) emitted no visible answer")
            #expect(result.reasoning.isEmpty, "\(label) routed reasoning while thinking was off")
            #expect(result.toolCalls == 0, "\(label) emitted an unrequested tool call")
            #expect(result.info != nil, "\(label) emitted no completion info")
            #expect(result.info?.unclosedReasoning == false,
                    "\(label) ended inside reasoning")
            #expect(result.info?.stopReason != .length,
                    "\(label) only stopped at the test token limit")
            for marker in ["<think>", "</think>", "<zyphra_tool_call>", "<|im_end|>"] {
                #expect(!result.text.contains(marker), "\(label) leaked \(marker)")
            }
        }

        let system = String(repeating: """
            This is stable cache-proof context. Keep exact identifiers unchanged,
            follow the latest user instruction, and answer concisely. The repeated
            prose exists only to make the prompt large enough for a real encoded
            KV middle region; do not summarize or discuss it.
            """, count: 5)
        var chat: [Chat.Message] = [
            .system(system),
            .user("Remember the exact access key CERULEAN-47. Reply only with saved, then <END>."),
        ]
        let prepared1 = try await context.processor.prepare(input: UserInput(
            chat: chat,
            additionalContext: ["enable_thinking": false]))
        let prompt1Tokens = prepared1.text.tokens.reshaped(-1).asArray(Int.self)
        #expect(prompt1Tokens.count > 96)

        let coordinatorA = makeCoordinator()
        let engineA = BatchEngine(
            context: ctx, maxBatchSize: 1, cacheCoordinator: coordinatorA)
        nonisolated(unsafe) let prepared1Send = clone(prepared1)
        let coldBoundary = await run(engineA, prepared1Send)
        await engineA.shutdown()
        assertClean(coldBoundary, label: "cold boundary")

        if testKVBits > 0 {
            let transition = try #require(coldBoundary.info?.turboQuantCacheTransition)
            #expect(coldBoundary.info?.turboQuantCompressions == 1)
            #expect(transition.before.layerCount == 80)
            #expect(transition.before.kvLayerCount == 40)
            #expect(transition.before.zayaCCALayerCount == 40)
            #expect(transition.before.turboQuantKVLayerCount == 0)
            #expect(transition.after.layerCount == 80)
            #expect(transition.after.kvLayerCount == 0)
            #expect(transition.after.zayaCCALayerCount == 40)
            #expect(transition.after.turboQuantKVLayerCount == 40)
            #expect(transition.convertedTurboQuantKVLayerCount == 40)
        } else {
            #expect(coldBoundary.info?.turboQuantCompressions == 0)
            #expect(coldBoundary.info?.turboQuantCacheTransition == nil)
        }

        chat.append(.assistant(coldBoundary.text))
        chat.append(.user(
            "What exact access key did I ask you to remember? Reply only with the key, then <END>."))
        let prepared2 = try await context.processor.prepare(input: UserInput(
            chat: chat,
            additionalContext: ["enable_thinking": false]))
        let prompt2Tokens = prepared2.text.tokens.reshaped(-1).asArray(Int.self)
        let salt2 = computeCacheSalt(for: prepared2, parameters: params)

        let coordinatorB = makeCoordinator()
        let probe = coordinatorB.fetch(
            tokens: prompt2Tokens,
            mediaSalt: salt2,
            skipExactDiskBoundary: true)
        let arrays: [String: MLXArray]
        let matched: Int
        let remaining: Int
        switch probe {
        case .hit(let matchedTokens, let remainingTokens, let detail, let blocks, _, let disk):
            coordinatorB.release(blocks: blocks)
            #expect(detail == .disk)
            arrays = try #require(disk)
            matched = matchedTokens
            remaining = remainingTokens.count
        case .miss:
            Issue.record("fresh coordinator missed the prior turn's SSD prefix")
            return
        }
        #expect(matched > 0)
        #expect(matched < prompt2Tokens.count)
        #expect(remaining == prompt2Tokens.count - matched)

        let indexed = TQDiskSerializer.deserializeIndexed(arrays)
        var zayaTQ = 0
        var zayaRaw = 0
        var skipped = 0
        var requiredMiss = 0
        for entry in indexed {
            switch entry.data {
            case .zayaCCATQ: zayaTQ += 1
            case .zayaCCA: zayaRaw += 1
            case .skip: skipped += 1
            case .requiredMiss: requiredMiss += 1
            default: break
            }
        }
        #expect(TQDiskSerializer.formatVersion(of: arrays) == 2)
        #expect(indexed.count == 80)
        #expect(zayaTQ == (expectEncodedDisk ? 40 : 0))
        #expect(zayaRaw == (expectEncodedDisk ? 0 : 40))
        #expect(skipped == 40)
        #expect(requiredMiss == 0)
        #expect(arrays.keys.filter { $0.hasPrefix("zaya_") && $0.hasSuffix("_conv_state") }.count == 40)
        #expect(arrays.keys.filter { $0.hasPrefix("zaya_") && $0.hasSuffix("_prev_hs") }.count == 40)
        let rawKeyCount = arrays.keys.filter {
            $0.hasPrefix("zaya_") && $0.hasSuffix("_keys")
        }.count
        let rawValueCount = arrays.keys.filter {
            $0.hasPrefix("zaya_") && $0.hasSuffix("_values")
        }.count
        #expect(rawKeyCount == (expectEncodedDisk ? 0 : 40))
        #expect(rawValueCount == (expectEncodedDisk ? 0 : 40))

        let safetensors = (try? FileManager.default.contentsOfDirectory(
            at: diskURL,
            includingPropertiesForKeys: [.fileSizeKey]))?.filter {
                $0.pathExtension == "safetensors"
            } ?? []
        let diskBytes = safetensors.reduce(Int64(0)) { total, url in
            total + Int64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        #expect(!safetensors.isEmpty)
        #expect(diskBytes > 0)
        print(String(format:
            "ZAYA_SELECTIVE_TQ_DISK bits=%d matched=%d/%d remaining=%d files=%d bytes=%lld gib=%.6f kinds{zayaTQ=%d,zayaRaw=%d,skip=%d}",
            testKVBits, matched, prompt2Tokens.count, remaining, safetensors.count, diskBytes,
            Double(diskBytes) / 1_073_741_824.0, zayaTQ, zayaRaw, skipped))

        let engineB = BatchEngine(
            context: ctx, maxBatchSize: 1, cacheCoordinator: coordinatorB)
        nonisolated(unsafe) let prepared2WarmSend = clone(prepared2)
        let warm = await run(engineB, prepared2WarmSend)
        await engineB.shutdown()
        assertClean(warm, label: "warm partial SSD")

        let engineCold = BatchEngine(context: ctx, maxBatchSize: 1)
        nonisolated(unsafe) let prepared2ColdSend = clone(prepared2)
        let cold = await run(engineCold, prepared2ColdSend)
        await engineCold.shutdown()
        assertClean(cold, label: "cold comparison")

        #expect(warm.text.uppercased().contains("CERULEAN-47"))
        #expect(cold.text.uppercased().contains("CERULEAN-47"))
        let stats = coordinatorB.snapshotStats()
        #expect(stats.diskStats?.hits ?? 0 >= 2)
        print(String(format:
            "ZAYA_SELECTIVE_TQ_RESULT warm{ttft=%.3f,tokps=%.2f,text=%@} cold{ttft=%.3f,tokps=%.2f,text=%@} diskHits=%d",
            warm.info?.promptTime ?? -1,
            warm.info?.tokensPerSecond ?? 0,
            warm.text.replacingOccurrences(of: "\n", with: "\\n"),
            cold.info?.promptTime ?? -1,
            cold.info?.tokensPerSecond ?? 0,
            cold.text.replacingOccurrences(of: "\n", with: "\\n"),
            stats.diskStats?.hits ?? 0))
    }

    private static func collectGeneration(_ stream: AsyncStream<Generation>)
        async -> (text: String, reasoning: String, toolCalls: Int, unclosedReasoning: Bool)
    {
        var text = ""
        var reasoning = ""
        var toolCalls = 0
        var unclosedReasoning = false
        for await event in stream {
            switch event {
            case .prefillProgress:
                break
            case .chunk(let chunk):
                text += chunk
            case .reasoning(let chunk):
                reasoning += chunk
            case .toolCall:
                toolCalls += 1
            case .toolCallProgress:
                break
            case .prefillProgress:

                break
            case .info(let info):
                unclosedReasoning = info.unclosedReasoning
            }
        }
        return (text, reasoning, toolCalls, unclosedReasoning)
    }
}
