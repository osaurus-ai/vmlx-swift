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
    static let bundlePath = bundleRoot + "/ZAYA1-8B-JANGTQ2"
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

    @Test("newCache returns 80 entries: even ZayaCCACache, odd KVCacheSimple",
          .enabled(if: ZayaSmokeJANGTQ2Tests.bundlePresent))
    func newCacheShape() async throws {
        let url = URL(fileURLWithPath: Self.bundlePath)
        let factory = LLMModelFactory.shared
        let container = try await factory.loadContainer(from: url, using: NoOpTokenizerLoader())
        let layout: (count: Int, evenIsZaya: Bool, oddIsSimple: Bool) =
            await container.perform { context in
                let cache = context.model.newCache(parameters: nil)
                let count = cache.count
                let evenIsZaya = (cache[0] is ZayaCCACache)
                let oddIsSimple = (cache[1] is KVCacheSimple)
                return (count, evenIsZaya, oddIsSimple)
            }
        #expect(layout.count == 80)
        #expect(layout.evenIsZaya)
        #expect(layout.oddIsSimple)
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
            case .prefillProgress:

                break
            case .info(let info):
                unclosedReasoning = info.unclosedReasoning
            }
        }
        return (text, reasoning, toolCalls, unclosedReasoning)
    }
}
