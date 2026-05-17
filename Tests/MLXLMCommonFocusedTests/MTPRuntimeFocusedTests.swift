// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@Suite("MTP runtime metadata")
struct MTPRuntimeFocusedTests {
    @Test("Qwen-style preserved MTP bundle is detected but not auto-enabled")
    func qwenPreservedMTPBundleIsDetectedButNotAutoEnabled() throws {
        let root = try makeTemporaryBundle(name: "qwen-mtp-detected")
        defer { try? FileManager.default.removeItem(at: root) }

        try writeJSON([
            "model_type": "qwen3_vl",
            "text_config": [
                "model_type": "qwen3",
                "num_hidden_layers": 48,
                "mtp_num_hidden_layers": 1,
            ] as [String: Any],
        ], to: root.appendingPathComponent("config.json"))
        try writeJSON([
            "runtime": [
                "total_weight_bytes": 17_820_460_160,
                "total_weight_gb": 16.6,
                "bundle_has_mtp": true,
                "mtp_layers": 1,
                "mtp_mode": "preserved_enabled",
            ] as [String: Any],
        ], to: root.appendingPathComponent("jang_config.json"))
        try writeJSON([
            "weight_map": [
                "mtp.fc.weight": "model-00029-of-00029.safetensors",
                "mtp.layers.0.self_attn.q_proj.weight": "model-00029-of-00029.safetensors",
                "mtp.layers.0.mlp.down_proj.weight": "model-00029-of-00029.safetensors",
                "vision_tower.blocks.0.attn.qkv.weight": "model-00001-of-00029.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00029.safetensors",
            ] as [String: Any],
        ], to: root.appendingPathComponent("model.safetensors.index.json"))

        let status = try MTPBundleInspector.inspect(modelDirectory: root)

        #expect(status.bundleHasMTP)
        #expect(status.configuredLayers == 1)
        #expect(status.tensorCount == 3)
        #expect(status.visionTensorCount == 1)
        #expect(status.mode == .preservedEnabled)
        #expect(status.hasCompleteMTPArtifact)
        #expect(status.requiresAcceptRejectBeforeEnable)
        #expect(!status.speculativeDecodeEnabled)
        #expect(!status.canAutoLaunchMTP)
        #expect(status.configEvidence.contains("text_config.mtp_num_hidden_layers=1"))
        #expect(status.statusLine.contains("accept/reject required"))
    }

    @Test("MTP config without tensors is reported as metadata-only")
    func configOnlyMTPIsMetadataOnlyMissingWeights() throws {
        let root = try makeTemporaryBundle(name: "qwen-mtp-missing-weights")
        defer { try? FileManager.default.removeItem(at: root) }

        try writeJSON([
            "text_config": [
                "mtp_num_hidden_layers": 1,
            ] as [String: Any],
        ], to: root.appendingPathComponent("config.json"))
        try writeJSON([
            "weight_map": [
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
            ] as [String: Any],
        ], to: root.appendingPathComponent("model.safetensors.index.json"))

        let status = try MTPBundleInspector.inspect(modelDirectory: root)

        #expect(!status.bundleHasMTP)
        #expect(status.configuredLayers == 1)
        #expect(status.tensorCount == 0)
        #expect(status.mode == .metadataOnlyMissingWeights)
        #expect(!status.hasCompleteMTPArtifact)
        #expect(!status.speculativeDecodeEnabled)
        #expect(!status.canAutoLaunchMTP)
    }

    @Test("inactive native MTP scrub does not touch generic nextn metadata")
    func inactiveNativeMTPScrubDoesNotTouchGenericNextnMetadata() throws {
        let config = """
        {
          "model_type": "deepseek_v4",
          "mtp_num_hidden_layers": 1,
          "num_nextn_predict_layers": 7,
          "text_config": {
            "model_type": "qwen3_5",
            "mtp_num_hidden_layers": 1,
            "num_nextn_predict_layers": 3
          }
        }
        """.data(using: .utf8)!

        let scrubbed = try NativeMTPActivation.scrubInactiveMTPConfig(config)
        let object = try #require(
            JSONSerialization.jsonObject(with: scrubbed) as? [String: Any])
        let textConfig = try #require(object["text_config"] as? [String: Any])

        #expect(object["mtp_num_hidden_layers"] as? Int == 0)
        #expect(object["num_nextn_predict_layers"] as? Int == 7)
        #expect(textConfig["mtp_num_hidden_layers"] as? Int == 0)
        #expect(textConfig["num_nextn_predict_layers"] as? Int == 3)
    }

    @Test("native MTP activation supports Qwen3.5 MoE only with explicit tensor evidence")
    func nativeMTPActivationSupportsQwen35MoEWithTensorEvidence() throws {
        setenv("VMLINUX_NATIVE_MTP", "1", 1)
        defer { unsetenv("VMLINUX_NATIVE_MTP") }

        let config = """
        {
          "model_type": "qwen3_5_moe",
          "text_config": {
            "model_type": "qwen3_5_moe_text",
            "mtp_num_hidden_layers": 1
          }
        }
        """.data(using: .utf8)!
        let status = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 1,
            tensorCount: 42,
            visionTensorCount: 333,
            mode: .preservedEnabled)

        let shouldLoad = try NativeMTPActivation.shouldLoadNativeMTPWeights(
            configData: config,
            baseModelType: "qwen3_5_moe",
            status: status)

        #expect(shouldLoad)
    }

    @Test("JANG MTP metadata without tensor evidence is not treated as an MTP bundle")
    func jangMTPMetadataWithoutTensorEvidenceIsMissingWeights() throws {
        let root = try makeTemporaryBundle(name: "named-mtp-but-no-mtp-tensors")
        defer { try? FileManager.default.removeItem(at: root) }

        try writeJSON([
            "model_type": "qwen3_5",
            "text_config": [
                "num_hidden_layers": 64,
                "mtp_num_hidden_layers": 1,
            ] as [String: Any],
        ], to: root.appendingPathComponent("config.json"))
        try writeJSON([
            "format": "jang",
            "format_version": "2.0",
            "runtime": [
                "bundle_has_mtp": true,
                "mtp_layers": 1,
                "mtp_mode": "preserved_enabled",
            ] as [String: Any],
        ], to: root.appendingPathComponent("jang_config.json"))
        try writeJSON([
            "weight_map": [
                "model.embed_tokens.weight": "model-00001-of-00001.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
                "model.layers.63.mlp.down_proj.weight": "model-00001-of-00001.safetensors",
            ] as [String: Any],
        ], to: root.appendingPathComponent("model.safetensors.index.json"))

        let status = try MTPBundleInspector.inspect(modelDirectory: root)

        #expect(!status.bundleHasMTP)
        #expect(status.configuredLayers == 1)
        #expect(status.tensorCount == 0)
        #expect(status.mode == .metadataOnlyMissingWeights)
        #expect(!status.hasCompleteMTPArtifact)
        #expect(!status.canAutoLaunchMTP)
        #expect(status.configEvidence.contains("jang_config.runtime.bundle_has_mtp=true"))
    }

    @Test("JANG runtime parses MTP activation metadata")
    func jangRuntimeParsesMTPActivationMetadata() throws {
        let config = try JangLoader.parseConfig(from: [
            "runtime": [
                "total_weight_bytes": 17_820_460_160,
                "total_weight_gb": 16.6,
                "bundle_has_mtp": true,
                "mtp_layers": 1,
                "mtp_mode": "preserved_enabled",
            ] as [String: Any],
        ])

        #expect(config.runtime.totalWeightBytes == 17_820_460_160)
        #expect(config.runtime.bundleHasMTP)
        #expect(config.runtime.mtpLayers == 1)
        #expect(config.runtime.mtpMode == .preservedEnabled)
    }

    @Test("ModelConfiguration carries MTP status into resolved configuration")
    func modelConfigurationCarriesMTPStatusIntoResolvedConfiguration() {
        let root = URL(fileURLWithPath: "/tmp/qwen-mtp")
        let status = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 1,
            tensorCount: 31,
            visionTensorCount: 333,
            mode: .preservedEnabled,
            tensorSamples: ["mtp.fc.weight"],
            visionTensorSamples: ["vision_tower.blocks.0.attn.qkv.weight"],
            configEvidence: ["text_config.mtp_num_hidden_layers=1"])
        let configuration = ModelConfiguration(
            directory: root,
            mtpStatus: status)

        let resolved = configuration.resolved(modelDirectory: root, tokenizerDirectory: root)

        #expect(configuration.mtpStatus == status)
        #expect(resolved.mtpStatus == status)
        #expect(resolved.mtpStatus?.requiresAcceptRejectBeforeEnable == true)
    }

    @Test("recursive MTP contract models D3 hidden-state draft verify")
    func recursiveMTPContractModelsD3HiddenStateDraftVerify() {
        let contract = MTPRecursiveDraftContract.mtplxDepth3

        #expect(contract.depth == 3)
        #expect(contract.draftStepReturnsHiddenState)
        #expect(contract.draftCacheIsPrivate)
        #expect(contract.backboneCacheCommitPolicy == .acceptedVerifierTokensOnly)
        #expect(contract.verifierPositionsPerCycle == 4)
        #expect(contract.minAcceptedDraftTokensPerVerify == 0)
        #expect(contract.maxAcceptedDraftTokensPerVerify == 3)
        #expect(contract.requiresVariablePrefixCommit)
        #expect(contract.partialAcceptCommitStrategy == .captureCommit)
        #expect(contract.maxCommittedTokensPerVerify == 4)
        #expect(contract.fullAcceptanceVerifyCycles(forOutputTokens: 256) == 64)
        #expect(contract.speedBenchRequirements.requiresARBaseline)
        #expect(contract.speedBenchRequirements.requiresVerifyCalls)
        #expect(contract.speedBenchRequirements.requiresAcceptedDraftedByDepth)
        #expect(contract.speedBenchRequirements.requiresPhaseTiming)
        #expect(contract.speedBenchRequirements.requiresOutputTailReview)
    }

    @Test("Qwen3.5 text SSM cache records accepted-prefix offsets")
    func qwen35TextSSMCacheRecordsAcceptedPrefixOffsets() throws {
        let source = try Self.source("Libraries/MLXLLM/Models/Qwen35.swift")

        #expect(source.contains("cache.offset += S"))
        #expect(source.contains("offset: baseOffset + prefixLength"))
        #expect(!source.contains("offset: baseOffset)"))
    }

    @Test("native MTP partial reject refreshes private draft cache")
    func nativeMTPPartialRejectRefreshesPrivateDraftCache() throws {
        let source = try Self.source(
            "Libraries/MLXLMCommon/SpecDec/NativeMTPTokenIterator.swift")

        #expect(source.contains("mtpCache = model.makeNativeMTPCache()"))
        #expect(source.contains("mtpCacheRefreshCount += 1"))
        #expect(!source.contains("trimPromptCache(mtpCache"))
    }

    @Test("BatchEngine.generate rejects native MTP without an active MTP head")
    func batchEngineGenerateRejectsNativeMTPWithoutActiveHead() async throws {
        try await FocusedMLXTestSupport.withLock {
            let model = try Qwen35TextModel(Self.tinyQwen35Config(mtpLayers: 0))
            let context = Self.nativeMTPDispatchContext(model: model)
            let engine = BatchEngine(context: context, maxBatchSize: 2)
            var params = GenerateParameters(maxTokens: 4, temperature: 0)
            params.draftStrategy = .nativeMTP(depth: 3)

            let stream = await engine.generate(
                input: LMInput(tokens: MLXArray([3, 5, 7])),
                parameters: params)
            let info = await Self.collectInfo(from: stream)

            #expect(info.count == 1)
            #expect(info.cancelled == 1)
        }
    }

    @Test("BatchEngine.generate routes active native MTP through the exclusive lane")
    func batchEngineGenerateRunsActiveNativeMTP() async throws {
        try await FocusedMLXTestSupport.withLock {
            let model = FocusedNativeMTPProbeTarget()
            #expect(model.nativeMTPAvailable)
            let context = Self.nativeMTPDispatchContext(model: model)
            let engine = BatchEngine(context: context, maxBatchSize: 2)
            var params = GenerateParameters(maxTokens: 4, temperature: 0)
            params.draftStrategy = .nativeMTP(depth: 3)

            let stream = await engine.generate(
                input: LMInput(tokens: MLXArray([3, 5, 7])),
                parameters: params)
            let info = await Self.collectInfo(from: stream)

            #expect(info.count == 1)
            #expect(info.cancelled == 0)
        }
    }

    @Test("BatchEngine.submit rejects native MTP instead of silently batching AR")
    func batchEngineSubmitRejectsNativeMTP() async throws {
        try await FocusedMLXTestSupport.withLock {
            let model = FocusedNativeMTPProbeTarget()
            let context = Self.nativeMTPDispatchContext(model: model)
            let engine = BatchEngine(context: context, maxBatchSize: 2)
            var params = GenerateParameters(maxTokens: 4, temperature: 0)
            params.draftStrategy = .nativeMTP(depth: 3)

            let (_, stream) = await engine.submit(
                input: LMInput(tokens: MLXArray([3, 5, 7])),
                parameters: params)
            var cancelled = 0
            for await event in stream {
                if case .info(let info) = event, info.stopReason == .cancelled {
                    cancelled += 1
                }
            }

            #expect(cancelled == 1)
        }
    }

    @Test("shape-walk quantization preserves MXFP4 mode")
    func shapeWalkQuantizationPreservesMXFP4Mode() {
        let weights: [String: MLXArray] = [
            "model.layers.0.mlp.down_proj.weight": MLXArray.zeros([2, 16], dtype: .uint32),
            "model.layers.0.mlp.down_proj.scales": MLXArray.zeros([2, 4], dtype: .float32),
            "model.layers.1.mlp.down_proj.weight": MLXArray.zeros([2, 32], dtype: .uint32),
            "model.layers.1.mlp.down_proj.scales": MLXArray.zeros([2, 4], dtype: .float32),
        ]

        let inferred = JangLoader.inferPerLayerQuantizationFromShapes(
            weights: weights,
            defaultBits: 4,
            defaultGroupSize: 32,
            defaultMode: .mxfp4)

        #expect(inferred?.quantization?.mode == .mxfp4)
        if case .quantize(let override)? =
            inferred?.perLayerQuantization["model.layers.1.mlp.down_proj"]
        {
            #expect(override.bits == 8)
            #expect(override.groupSize == 32)
            #expect(override.mode == .mxfp4)
        } else {
            Issue.record("Expected 8-bit MXFP4 per-layer override")
        }
    }

    @Test("Qwen3.5 sanitize does not shift base norms just because MTP tensors exist")
    func qwen35SanitizeDoesNotShiftBaseNormsForPreservedMTP() throws {
        let configData = """
        {
          "hidden_size": 4,
          "num_hidden_layers": 1,
          "intermediate_size": 8,
          "num_attention_heads": 1,
          "num_key_value_heads": 1,
          "linear_num_value_heads": 1,
          "linear_num_key_heads": 1,
          "linear_key_head_dim": 4,
          "linear_value_head_dim": 4,
          "linear_conv_kernel_dim": 4,
          "head_dim": 4,
          "vocab_size": 16,
          "tie_word_embeddings": false
        }
        """.data(using: .utf8)!
        let configuration = try JSONDecoder().decode(Qwen35TextConfiguration.self, from: configData)
        let model = Qwen35TextModel(configuration)
        let norm = MLXArray([Float](repeating: 0.5, count: 4))

        let sanitized = model.sanitize(weights: [
            "mtp.layers.0.linear_attn.conv1d.weight": MLXArray.zeros([4, 4, 4], dtype: .float32),
            "mtp.fc.weight": MLXArray.zeros([4, 4], dtype: .float32),
            "model.norm.weight": norm,
        ])

        #expect(sanitized["mtp.fc.weight"] == nil)
        #expect(sanitized["mtp.layers.0.linear_attn.conv1d.weight"] == nil)
        #expect(sanitized["model.norm.weight"]?.asArray(Float.self) == [0.5, 0.5, 0.5, 0.5])
    }

    @Test("Qwen3.5 JANGTQ sanitize also ignores MTP sidecar conv when deciding norm shifts")
    func qwen35JANGTQSanitizeDoesNotShiftBaseNormsForPreservedMTP() throws {
        let configData = """
        {
          "hidden_size": 4,
          "num_hidden_layers": 1,
          "intermediate_size": 8,
          "num_attention_heads": 1,
          "num_key_value_heads": 1,
          "linear_num_value_heads": 1,
          "linear_num_key_heads": 1,
          "linear_key_head_dim": 4,
          "linear_value_head_dim": 4,
          "linear_conv_kernel_dim": 4,
          "head_dim": 4,
          "vocab_size": 16,
          "tie_word_embeddings": false,
          "num_experts": 0,
          "num_experts_per_tok": 0,
          "weight_format": "mxtq",
          "mxtq_bits": 4
        }
        """.data(using: .utf8)!
        let configuration = try JSONDecoder().decode(
            Qwen35JANGTQTextConfiguration.self, from: configData)
        let model = Qwen35JANGTQTextModel(configuration)
        let norm = MLXArray([Float](repeating: 0.5, count: 4))

        let sanitized = model.sanitize(weights: [
            "model.mtp_layers.0.linear_attn.conv1d.weight": MLXArray.zeros(
                [4, 4, 4], dtype: .float32),
            "mtp.fc.weight": MLXArray.zeros([4, 4], dtype: .float32),
            "model.norm.weight": norm,
        ])

        #expect(sanitized["model.mtp_layers.0.linear_attn.conv1d.weight"] == nil)
        #expect(sanitized["mtp.fc.weight"] == nil)
        #expect(sanitized["model.norm.weight"]?.asArray(Float.self) == [0.5, 0.5, 0.5, 0.5])
    }

    @Test("optional real local MTP bundle inspection")
    func optionalRealLocalMTPBundleInspection() throws {
        guard let path = ProcessInfo.processInfo.environment["VMLX_MTP_REAL_BUNDLE"],
            !path.isEmpty
        else {
            return
        }

        let status = try MTPBundleInspector.inspect(
            modelDirectory: URL(fileURLWithPath: path))

        #expect(status.bundleHasMTP)
        #expect(status.configuredLayers > 0)
        #expect(status.tensorCount > 0)
        #expect(status.hasCompleteMTPArtifact)
        #expect(!status.canAutoLaunchMTP)
        if ProcessInfo.processInfo.environment["VMLX_MTP_REAL_BUNDLE_EXPECTS_VL"] == "1" {
            #expect(status.visionTensorCount > 0)
            #expect(status.bundleHasVision)
        }
    }

    private func makeTemporaryBundle(name: String) throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "\(name)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func writeJSON(_ object: [String: Any], to url: URL) throws {
        let data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url)
    }

    private static func source(_ relativePath: String) throws -> String {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        let url = root.appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }

    private static func tinyQwen35Config(mtpLayers: Int) throws -> Qwen35TextConfiguration {
        let data = """
        {
          "hidden_size": 16,
          "num_hidden_layers": 1,
          "intermediate_size": 32,
          "num_attention_heads": 2,
          "num_key_value_heads": 2,
          "linear_num_value_heads": 1,
          "linear_num_key_heads": 1,
          "linear_key_head_dim": 8,
          "linear_value_head_dim": 8,
          "linear_conv_kernel_dim": 4,
          "head_dim": 8,
          "vocab_size": 32,
          "tie_word_embeddings": false,
          "mtp_num_hidden_layers": \(mtpLayers)
        }
        """.data(using: .utf8)!
        return try JSONDecoder().decode(Qwen35TextConfiguration.self, from: data)
    }

    private static func nativeMTPDispatchContext(model: any LanguageModel) -> ModelContext {
        let tokenizer = FocusedMTPTokenizer()
        return ModelContext(
            configuration: ModelConfiguration(id: "focused-native-mtp"),
            model: model,
            processor: FocusedMTPProcessor(tokenizer: tokenizer),
            tokenizer: tokenizer)
    }

    private static func collectInfo(
        from stream: AsyncStream<Generation>
    ) async -> (count: Int, cancelled: Int) {
        var count = 0
        var cancelled = 0
        for await event in stream {
            if case .info(let info) = event {
                count += 1
                if info.stopReason == .cancelled {
                    cancelled += 1
                }
            }
        }
        return (count, cancelled)
    }
}

private struct FocusedMTPTokenizer: Tokenizer {
    let vocabularySize = 64
    let eosTokenId: Int? = 60
    let unknownTokenId: Int? = 61
    let bosToken: String? = nil
    let eosToken: String? = nil
    let unknownToken: String? = nil

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        [3, 5, 7]
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { String(id) }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        [3, 5, 7]
    }
}

private struct FocusedMTPProcessor: UserInputProcessor {
    let tokenizer: any Tokenizer

    func prepare(input: UserInput) async throws -> LMInput {
        LMInput(tokens: MLXArray([3, 5, 7]))
    }
}

private final class FocusedNativeMTPProbeTarget: Module, LanguageModel, NativeMTPModel,
    KVCacheDimensionProvider, @unchecked Sendable
{
    var kvHeads: [Int] { [1] }
    var nativeMTPAvailable: Bool { true }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        preconditionFailure("BatchEngine native MTP dispatch fell through to AR callAsFunction")
    }

    func makeNativeMTPCache() -> [KVCache] {
        newCache(parameters: nil)
    }

    func nativeBackboneForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        NativeMTPForwardResult(
            logits: logits(for: inputs),
            hiddenStates: hiddenStates(for: inputs))
    }

    func nativeMTPForward(
        hiddenStates: MLXArray,
        nextTokenIds: MLXArray,
        cache: [KVCache]?
    ) -> NativeMTPForwardResult {
        NativeMTPForwardResult(
            logits: logits(for: nextTokenIds),
            hiddenStates: self.hiddenStates(for: nextTokenIds))
    }

    private func sequenceLength(_ inputs: MLXArray) -> Int {
        inputs.ndim >= 2 ? inputs.dim(1) : inputs.size
    }

    private func logits(for inputs: MLXArray) -> MLXArray {
        MLXArray.zeros([1, sequenceLength(inputs), 16])
    }

    private func hiddenStates(for inputs: MLXArray) -> MLXArray {
        MLXArray.zeros([1, sequenceLength(inputs), 4])
    }
}
