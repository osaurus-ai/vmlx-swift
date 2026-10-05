#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif
import Foundation
import MLX
import MLXLLM
import MLXNN
import XCTest
@testable import MLXLMCommon
@testable import MLXVLM

/// Metadata and real typed-cache ownership controls. No model forward or sampler.
final class GlmOrdinaryStableRederiveTests: XCTestCase {
    private static let ids = Array(0..<1300)
    private static func input(policy: LMInput.CacheRestorePolicy = .standard,
                              intent: LMInput.CachePromptIntent = .generation,
                              mask: MLXArray? = nil,
                              context: CanonicalRequiredToolContext? = nil,
                              tools: [ToolSpec]? = nil) -> LMInput {
        LMInput(tokens: MLXArray(ids.map(Int32.init)), mask: mask, tokenIds: ids,
            cachePrefixTokenCounts: [700, 800, 1200], cacheStablePrefixTokenCounts: [700, 800],
            cachePromptIntent: intent, cacheRestorePolicy: policy,
            toolSchemas: tools, canonicalRequiredToolContext: context)
    }
    private static func target(_ input: LMInput, _ model: Glm5Next, _ cache: [KVCache],
                               prepareInput: LMInput? = nil,
                               parameters: GenerateParameters = .init(prefillStepSize: 512),
                               missing: [Int] = [799, 699], structural: Int? = 1200) -> Int? {
        canonicalOrdinaryStableRederiveTarget(input: input, inputForPrepare: prepareInput ?? input,
            promptTokens: ids, cache: cache, model: model, parameters: parameters,
            structuralBoundary: structural, missingTargets: missing)
    }
    private static func capture(_ input: LMInput, _ model: Glm5Next, _ cache: [KVCache],
                                target: Int? = 699, salt: String = "request-a") throws -> CanonicalStablePrefillCapture {
        try XCTUnwrap(CanonicalStablePrefillCapture(input: input, promptTokens: ids,
            cache: cache, chunkSize: 512, targets: [target ?? 699], salt: salt,
            canonicalModel: model, exactReplayTarget: target))
    }
    private static func fill(_ cache: [KVCache], model: Glm5Next, count: Int, value: Float) {
        let config = model.config.textConfig
        for (kind, layer) in zip(model.schedule, cache) {
            switch kind {
            case .linearAttention:
                let mamba = layer as! MambaCache
                mamba[0] = MLXArray.full([1, config.linearAttnConfig.shortConvKernelSize - 1,
                    3 * config.linearAttnConfig.numHeads * config.linearAttnConfig.headDim], values: MLXArray(value), dtype: .bfloat16)
                mamba[1] = MLXArray.full([1, config.linearAttnConfig.numHeads,
                    config.linearAttnConfig.headDim, config.linearAttnConfig.headDim], values: MLXArray(value), dtype: .float32)
                mamba.offset = count
            case .deepseekSparseAttention:
                (layer as! Glm5NextIndexedKVCache).state = [
                    MLXArray.full([1, 1, count, config.kvLoraRank], values: MLXArray(value), dtype: .bfloat16),
                    MLXArray.full([1, 1, count, 1], values: MLXArray(value), dtype: .bfloat16),
                    MLXArray.full([1, count, 2 * config.indexHeadDim + 1], values: MLXArray(value), dtype: .bfloat16)]
            }
        }
        MLX.eval(cache)
    }
    private struct ArrayView: Equatable {
        let shape: [Int]; let dtype: String; let data: Data
        init(_ array: MLXArray) { shape = array.shape; dtype = String(describing: array.dtype); data = array.asData(access: .copy).data }
    }
    private struct CacheView: Equatable {
        let kind: String; let offset: Int; let meta: [String]; let persistent: Int?; let arrays: [ArrayView?]
        init(_ cache: KVCache) {
            kind = String(reflecting: type(of: cache)); offset = cache.offset; meta = cache.metaState
            persistent = (cache as? MambaCache)?.persistentSlotCount
            if let slots = cache as? ArraysCache { arrays = (0..<slots.slotCount).map { slots[$0].map(ArrayView.init) } }
            else { arrays = cache.state.map(ArrayView.init) }
        }
    }
    private static func views(_ cache: [KVCache]) -> [CacheView] { cache.map(CacheView.init) }

    private final class UnqualifiedModel: Module, CanonicalRequiredToolCacheModel {
        var canonicalRequiredToolCacheIdentity: String { "other-model" }
        func canonicalRequiredToolChunkSize(parameters: GenerateParameters) -> Int? { 512 }
        func validateCanonicalRequiredToolCache(_ cache: [KVCache], boundary: Int) -> Bool { true }
        func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            XCTFail("Default opt-out must not execute prepare"); return .tokens(input.text)
        }
        func callAsFunction(_ input: MLXArray, cache: [KVCache]?) -> MLXArray {
            XCTFail("Default opt-out must not execute forward"); return input
        }
    }
    func testColdSelectsOnlyMissingProcessorStableNMinusOne() throws {
        try MLXMetalTestLock.withLock {
            try Self.withFixture { model in
                let input = Self.input(), cache = model.newCache(parameters: nil)
                XCTAssertTrue(model.supportsOrdinaryStablePrefixRederive)
                let other = UnqualifiedModel()
                XCTAssertFalse(other.supportsOrdinaryStablePrefixRederive)
                XCTAssertNil(canonicalOrdinaryStableRederiveTarget(input: input, inputForPrepare: input,
                    promptTokens: Self.ids, cache: cache, model: other, parameters: .init(prefillStepSize: 512),
                    structuralBoundary: 1200, missingTargets: [699]))
                XCTAssertEqual(Self.target(input, model, cache), 699)
                XCTAssertEqual(Self.target(input, model, cache, missing: [799]), 799)
                XCTAssertNil(Self.target(input, model, cache, missing: []), "Already-durable targets are omitted by the caller")
                XCTAssertNil(Self.target(input, model, cache, missing: [698, 1200]))
                XCTAssertNil(Self.target(input, model, cache, structural: nil))
                XCTAssertNil(Self.target(input, model, cache, structural: 699))
                XCTAssertNil(Self.target(input, model, cache, parameters: .init(prefillStepSize: 256)))
                let capture = try Self.capture(input, model, cache)
                XCTAssertEqual(capture.seedCount, 512)
                XCTAssertNil(capture.snapshot, "Alignment alone is not cold-chunk provenance")
            }
        }
    }
    func testWarmMasksMediaAndUnqualifiedMathKeepFallback() throws {
        try MLXMetalTestLock.withLock {
            try Self.withFixture { model in
                let input = Self.input(), cache = model.newCache(parameters: nil)
                let suffix = LMInput(tokens: MLXArray(Self.ids.dropFirst(512).map(Int32.init)), tokenIds: Array(Self.ids.dropFirst(512)))
                XCTAssertNil(Self.target(input, model, cache, prepareInput: suffix))
                XCTAssertNil(Self.target(Self.input(intent: .auxiliary), model, cache))
                XCTAssertNil(Self.target(Self.input(intent: .reusablePrefixWarmup), model, cache))
                XCTAssertNil(Self.target(Self.input(mask: MLXArray.ones([1300], dtype: .int32)), model, cache))
                let image = LMInput(text: input.text, image: .init(pixels: MLXArray.zeros([1, 1, 1, 3])),
                    cacheStablePrefixTokenCounts: [700])
                XCTAssertNil(Self.target(image, model, cache))
                var wrong = cache; wrong[1] = KVCacheSimple()
                XCTAssertNil(Self.target(input, model, wrong))
                let embedding = try XCTUnwrap(model.languageModel.embedTokens as? QuantizedEmbedding)
                embedding.outputDType = .float16
                XCTAssertNil(Self.target(input, model, cache))
                embedding.outputDType = .bfloat16
                setenv("MLX_ENABLE_TF32", "0", 1)
                XCTAssertNil(Self.target(input, model, cache))
                unsetenv("MLX_ENABLE_TF32")
                Self.fill(cache, model: model, count: 512, value: 1)
                XCTAssertNil(Self.target(input, model, cache), "An aligned populated cache is not cold origin")
            }
        }
    }
    func testRequiredAndNamedCaptureRemainSeparate() throws {
        try MLXMetalTestLock.withLock {
            try Self.withFixture { model in
                let function: [String: any Sendable] = ["name": "file_read", "parameters": ["type": "object"] as [String: any Sendable]]
                let tools: [ToolSpec] = [["type": "function", "function": function]]
                let parameters = GenerateParameters(prefillStepSize: 512)
                let choices: [[String: any Sendable]] = [["tool_choice": "required"], ["tool_choice": "required", "tool_choice_name": "file_read"]]
                for additional in choices {
                    let scope = try XCTUnwrap(CanonicalRequiredToolContext(additionalContext: additional, tools: tools))
                    let input = Self.input(policy: .freshRequiredToolSelection, context: scope, tools: tools)
                    let cache = model.newCache(parameters: parameters)
                    XCTAssertNil(Self.target(input, model, cache))
                    let salt = try XCTUnwrap(canonicalRequiredToolSalt(input: input, model: model,
                        parameters: parameters, cache: cache, ordinarySalt: "ordinary"))
                    let capture = try Self.capture(input, model, cache, target: nil, salt: salt)
                    Self.fill(cache, model: model, count: 512, value: 1)
                    capture.receive(input: input, cache: cache, chunkSize: 512, completed: 512, beganWithEmptyCache: true)
                    XCTAssertNotNil(capture.snapshot)
                    XCTAssertNotNil(capture.copySeed(for: Array(Self.ids.prefix(700)), salt: salt, chunkSize: 512),
                        "Existing required collector has no new ordinary exact-target restriction")
                    XCTAssertNil(capture.copySeed(for: Array(Self.ids.prefix(699)), salt: "ordinary", chunkSize: 512))
                }
                let unknown = Self.input(policy: .freshRequiredToolSelection)
                XCTAssertNil(Self.target(unknown, model, model.newCache(parameters: parameters)))
                XCTAssertNotNil(CanonicalStablePrefillCapture(input: Self.input(), promptTokens: Self.ids,
                    cache: [MambaCache()], chunkSize: 512, targets: [699], salt: "legacy"),
                    "Existing legacy topology remains eligible without model capability")
                XCTAssertNil(CanonicalStablePrefillCapture(input: Self.input(), promptTokens: Self.ids,
                    cache: model.newCache(parameters: parameters), chunkSize: 512, targets: [699], salt: "legacy"),
                    "GLM IndexedKV requires explicit typed model capability")
            }
        }
    }
    func testOwningSnapshotRejectsWrongOriginAndUnselectedTargets() throws {
        try MLXMetalTestLock.withLock {
            try Self.withFixture { model in
                let input = Self.input(), cache = model.newCache(parameters: nil)
                let capture = try Self.capture(input, model, cache)
                Self.fill(cache, model: model, count: 512, value: 1)
                capture.receive(input: input, cache: cache.map { $0.copy() }, chunkSize: 512, completed: 512, beganWithEmptyCache: true)
                XCTAssertNil(capture.snapshot)
                capture.receive(input: input, cache: cache, chunkSize: 512, completed: 512, beganWithEmptyCache: false)
                XCTAssertNil(capture.snapshot)
                capture.receive(input: input, cache: cache, chunkSize: 256, completed: 512, beganWithEmptyCache: true)
                XCTAssertNil(capture.snapshot)
                let changed = LMInput(tokens: input.text.tokens, tokenIds: [-1] + Array(Self.ids.dropFirst()))
                capture.receive(input: changed, cache: cache, chunkSize: 512, completed: 512, beganWithEmptyCache: true)
                XCTAssertNil(capture.snapshot)
                let mamba = cache[0] as! MambaCache
                let correctConv = MLXArray(data: try XCTUnwrap(mamba[0]).asData(access: .copy))
                mamba[0] = correctConv.asType(.float16)
                capture.receive(input: input, cache: cache, chunkSize: 512, completed: 512, beganWithEmptyCache: true)
                XCTAssertNil(capture.snapshot, "Typed dtype contract is checked at the reported chunk")
                mamba[0] = correctConv
                capture.receive(input: input, cache: cache, chunkSize: 512, completed: 512, beganWithEmptyCache: true)
                let seed = try XCTUnwrap(capture.snapshot), baseline = Self.views(seed)
                XCTAssertEqual(baseline, Self.views(cache))
                Self.fill(cache, model: model, count: 699, value: 2)
                XCTAssertEqual(Self.views(seed), baseline)
                XCTAssertNil(capture.copySeed(for: Array(Self.ids.prefix(700)), salt: "request-a", chunkSize: 512))
                XCTAssertNil(capture.copySeed(for: Array(Self.ids.prefix(699)), salt: "request-b", chunkSize: 512))
                XCTAssertNil(capture.copySeed(for: Array(Self.ids.prefix(699)), salt: "request-a", chunkSize: 256))
                let replay = try XCTUnwrap(capture.copySeed(for: Array(Self.ids.prefix(699)), salt: "request-a", chunkSize: 512))
                XCTAssertEqual(Self.views(replay), baseline)
                Self.fill(replay, model: model, count: 699, value: 3)
                XCTAssertEqual(Self.views(seed), baseline, "Replay cannot mutate retained seed")
            }
        }
    }
    private final class CancellationBox: @unchecked Sendable {
        // Sole child access under the process-wide Metal lock, parent joins it.
        let pending: CanonicalStablePrefillCapture; let sealed: CanonicalStablePrefillCapture
        let cache: [KVCache]; let input: LMInput; let before: [CacheView]
        init(_ pending: CanonicalStablePrefillCapture, _ sealed: CanonicalStablePrefillCapture, _ cache: [KVCache], _ input: LMInput, _ before: [CacheView]) {
            self.pending = pending; self.sealed = sealed; self.cache = cache; self.input = input; self.before = before
        }
    }
    func testCancellationDropsPendingCaptureAndReplayClone() async throws {
        try await MLXMetalTestLock.withLock {
            let box = try Self.withFixture { model in
                let input = Self.input(), cache = model.newCache(parameters: nil)
                let pending = try Self.capture(input, model, cache), sealed = try Self.capture(input, model, cache)
                Self.fill(cache, model: model, count: 512, value: 1)
                sealed.receive(input: input, cache: cache, chunkSize: 512, completed: 512, beganWithEmptyCache: true)
                return CancellationBox(pending, sealed, cache, input, Self.views(try XCTUnwrap(sealed.snapshot)))
            }
            let rejected = await Task { @Sendable in
                withUnsafeCurrentTask { $0?.cancel() }
                box.pending.receive(input: box.input, cache: box.cache, chunkSize: 512, completed: 512, beganWithEmptyCache: true)
                return box.pending.snapshot == nil
                    && box.sealed.copySeed(for: Array(Self.ids.prefix(699)), salt: "request-a", chunkSize: 512) == nil
            }.value
            XCTAssertTrue(rejected)
            XCTAssertEqual(Self.views(try XCTUnwrap(box.sealed.snapshot)), box.before)
        }
    }
    func testMalformedResidualUsesThrowingReplayContract() throws {
        try MLXMetalTestLock.withLock {
            try Self.withFixture { model in
                let cache = model.newCache(parameters: nil), before = Self.views(model.newCache(parameters: nil))
                XCTAssertThrowsError(try model.replayForward(MLXArray.zeros([1, 1, 1], dtype: .int32), cache: cache)) { error in
                    XCTAssertTrue(error is Glm5NextInputShapeError)
                }
                XCTAssertEqual(Self.views(cache), before, "Malformed replay must leave no partial target state")
            }
        }
    }
    private final class FixtureBank: Module, WeightedRoutedExpertLayer {
        func callRouted(_ input: MLXArray, indices: MLXArray, scores: MLXArray) -> MLXArray {
            XCTFail("Eligibility tests must not execute a model forward")
            return MLXArray.zeros(input.shape, dtype: input.dtype)
        }
    }
    private static func withFixture<T>(_ body: (Glm5Next) throws -> T) throws -> T {
        let priorAbsorb = Glm5NextIndexerRuntime.absorbMLA, priorGather = Glm5NextIndexerRuntime.gatherSelected
        let priorPool = Glm5NextIndexerRuntime.poolFP32, priorConv = KDAConvRuntime.enabled
        let names = ["MLX_ENABLE_TF32", "VMLX_GLM5_PREFILL_STEP"]
        let priorEnvironment = names.map { name -> String? in getenv(name).map { String(cString: $0) } }
        Glm5NextIndexerRuntime.absorbMLA = true; Glm5NextIndexerRuntime.gatherSelected = true
        Glm5NextIndexerRuntime.poolFP32 = false; KDAConvRuntime.enabled = true
        names.forEach { unsetenv($0) }
        defer {
            Glm5NextIndexerRuntime.absorbMLA = priorAbsorb; Glm5NextIndexerRuntime.gatherSelected = priorGather
            Glm5NextIndexerRuntime.poolFP32 = priorPool; KDAConvRuntime.enabled = priorConv
            for (name, prior) in zip(names, priorEnvironment) {
                if let prior { setenv(name, prior, 1) } else { unsetenv(name) }
            }
        }
        let config = try JSONDecoder().decode(Glm5NextConfiguration.self, from: Data(tinyJSON.utf8))
        var banks: [Int: any WeightedRoutedExpertLayer] = [:]
        for layer in 0..<config.textConfig.numHiddenLayers where config.textConfig.mlpLayerTypes[layer] == .sparse {
            banks[layer] = FixtureBank()
        }
        let exclusions = Set(banks.keys.flatMap { layer in
            ["gate_proj", "up_proj", "down_proj"].flatMap { role in
                ["tq2_packed", "tq2_scales"].map { "model.layers.\(layer).mlp.switch_mlp.\(role).\($0)" }
            }
        })
        let model = try Glm5Next(config, requesting: [.text], routedExperts: banks, customRoutedTensorNames: exclusions)
        let embedding = QuantizedEmbedding(
            weight: MLXArray.zeros([128, 16], dtype: .uint32),
            scales: MLXArray.ones([128, 1], dtype: .bfloat16),
            biases: MLXArray.zeros([128, 1], dtype: .bfloat16), groupSize: 64, bits: 8)
        model.languageModel.update(modules: ModuleChildren.unflattened([("embed_tokens", embedding)]))
        return try body(model)
    }
    private static let tinyJSON = #"""
        {"model_type":"glm5_next","image_token_id":9,"video_token_id":10,
         "text_config":{"model_type":"glm5_next_text","hidden_size":64,
           "num_hidden_layers":4,"intermediate_size":128,"num_attention_heads":4,
           "num_key_value_heads":4,"vocab_size":128,"rms_norm_eps":1e-05,
           "max_position_embeddings":4096,"kv_lora_rank":16,"q_lora_rank":32,
           "qk_nope_head_dim":16,"qk_rope_head_dim":0,"v_head_dim":16,"mla_use_nope":true,
           "n_routed_experts":8,"n_shared_experts":1,"num_experts_per_tok":2,
           "moe_intermediate_size":32,"first_k_dense_replace":2,"scoring_func":"sigmoid",
           "topk_method":"noaux_tc","routed_scaling_factor":2.5,"norm_topk_prob":true,
           "n_group":1,"topk_group":1,"mhc":true,"hc_mult":4,"hc_sinkhorn_iters":20,"hc_eps":1e-06,
           "index_head_dim":16,"index_n_heads":2,"index_topk":2048,"index_kpool":4,
           "index_kpool_compress":true,"index_kpool_always_select_tail":true,
           "num_nextn_predict_layers":0,"swiglu_limit":10.0,"tie_word_embeddings":false,
           "linear_attn_config":{"num_heads":4,"gate_lower_bound":-5.0,"head_dim":16,
             "short_conv_kernel_size":4,"kda_layers":[0,2,3],"full_attn_layers":[1]},
           "layer_types":["linear_attention","deepseek_sparse_attention","linear_attention","linear_attention"],
           "mlp_layer_types":["dense","dense","sparse","sparse"]}}
        """#
}
