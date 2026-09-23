// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The converted V2.6 contract, identified by representation rather than a repository name.
public typealias MiMoV26Contract = MiMoV26BundleContract

enum MiMoV26Error: Error {
    case invalidConfiguration(String)
}

/// Keeps the converted fused QKV module intact, including its per-module quantization.
final class MiMoV26Attention: Module {
    @ModuleInfo(key: "qkv_proj") var qkv: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    @ParameterInfo(key: "attention_sink_bias") var sink: MLXArray?
    let heads: Int
    let kvHeads: Int
    let keyDim: Int
    let valueDim: Int
    let valueScale: Float
    let rope: RoPE

    init(_ config: MiMoV2FlashConfiguration, layer: Int) {
        let sliding = config.isSlidingLayer(layer)
        heads = sliding ? config.swaAttentionHeads : config.attentionHeads
        kvHeads = config.kvHeadsForLayer(layer)
        keyDim = sliding ? config.swaHeadDim : config.headDim
        valueDim = sliding ? config.swaVHeadDim : config.vHeadDim
        valueScale = config.attentionValueScale
        _qkv.wrappedValue = Linear(
            config.hiddenSize, heads * keyDim + kvHeads * (keyDim + valueDim), bias: false)
        _output.wrappedValue = Linear(heads * valueDim, config.hiddenSize, bias: false)
        let hasSink = sliding ? config.addSwaAttentionSinkBias : config.addFullAttentionSinkBias
        _sink.wrappedValue = hasSink ? MLXArray.zeros([heads]) : nil
        rope = RoPE(
            dimensions: Int(Float(keyDim) * config.partialRotaryFactor), traditional: false,
            base: sliding ? config.swaRopeTheta : config.ropeTheta)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let batch = x.dim(0), length = x.dim(1)
        let parts = split(qkv(x), indices: [heads * keyDim, (heads + kvHeads) * keyDim], axis: -1)
        var q = parts[0].reshaped(batch, length, heads, keyDim).transposed(0, 2, 1, 3)
        var k = parts[1].reshaped(batch, length, kvHeads, keyDim).transposed(0, 2, 1, 3)
        let v = parts[2].reshaped(batch, length, kvHeads, valueDim).transposed(0, 2, 1, 3)
            * valueScale
        q = applyRotaryPosition(rope, to: q, cache: cache)
        k = applyRotaryPosition(rope, to: k, cache: cache)
        let attended = attentionWithCacheUpdateAndSinks(
            queries: q, keys: k, values: v, cache: cache,
            scale: pow(Float(keyDim), -0.5), mask: mask, sinks: sink?.asType(q.dtype))
        return output(attended.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class MiMoV26Router: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correction: MLXArray
    let topK: Int
    let normalize: Bool
    let scale: Float

    init(_ config: MiMoV2FlashConfiguration) {
        _weight.wrappedValue = MLXArray.zeros([config.nRoutedExperts!, config.hiddenSize])
        _correction.wrappedValue = MLXArray.zeros([config.nRoutedExperts!])
        topK = config.numExpertsPerTok
        normalize = config.normTopkProb
        scale = config.routedScalingFactor ?? 1
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let scores = sigmoid(x.asType(.float32).matmul(weight.asType(.float32).T))
        let choice = scores + correction.asType(.float32)
        let indices = stopGradient(argPartition(-choice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK])
        var selected = takeAlong(scores, indices, axis: -1)
        if topK > 1 && normalize {
            selected = selected / (selected.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (indices, selected * scale)
    }
}

final class MiMoV26MoE: Module, UnaryLayer {
    let gate: MiMoV26Router
    @ModuleInfo(key: "switch_mlp") var experts: Module & SwitchGLULayer
    private var compiledResidentDecode: (@Sendable ([MLXArray]) -> [MLXArray])?

    init(_ config: MiMoV2FlashConfiguration) {
        gate = MiMoV26Router(config)
        _experts.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize, hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts!, allowFusedGateUpCache: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let compiledResidentDecode, !CompiledDecodeTrace.isActive, x.dim(-2) == 1 {
            return compiledResidentDecode([x])[0]
        }
        let (indices, scores) = gate(x)
        if let mixed = experts as? MixedQuantizedSwitchGLU,
            let result = mixed.fusedWeightedOutput(x, indices, scores: scores) {
            return result
        }
        let values = experts(x, indices).asType(.float32)
        return (values * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(x.dtype)
    }

    /// Cache-free MoE region only: attention and rotating KV remain eager.
    /// Capture child modules, never self, so unloading releases the full bank.
    func configureCompiledResidentDecode() {
        guard HardwareInfo.isCompiledDecodeSupported,
            RuntimeEnvironment.value("VMLX_MIMO_COMPILE_MOE") == "1",
            (experts as? MixedQuantizedSwitchGLU)?.usesResidentGPURouting == true else { return }
        let router = gate, projections = experts
        compiledResidentDecode = vmlxTrustedCompile(inputs: [router]) { args in
            let x = args[0]
            let (indices, scores) = router(x)
            let values = projections(x, indices).asType(.float32)
            return [(values * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(x.dtype)]
        }
    }
}

final class MiMoV26Layer: Module {
    @ModuleInfo(key: "self_attn") var attention: MiMoV26Attention
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    let mlp: UnaryLayer

    init(_ config: MiMoV2FlashConfiguration, layer: Int) {
        _attention.wrappedValue = MiMoV26Attention(config, layer: layer)
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.layernormEpsilon)
        _postNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.layernormEpsilon)
        mlp = config.moeLayerFreq[layer] == 1 ? MiMoV26MoE(config) : MiMoV2FlashMLP(config)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputNorm(x), mask: mask, cache: cache)
        return h + mlp(postNorm(h))
    }
}

final class MiMoV26Backbone: Module {
    @ModuleInfo(key: "embed_tokens") var embeddings: Embedding
    let layers: [MiMoV26Layer]
    let norm: RMSNorm

    init(_ config: MiMoV2FlashConfiguration) {
        _embeddings.wrappedValue = Embedding(embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        layers = (0..<config.hiddenLayers).map { MiMoV26Layer(config, layer: $0) }
        norm = RMSNorm(dimensions: config.hiddenSize, eps: config.layernormEpsilon)
    }
}

/// Text backbone shared by text-only requests and the V2.6 multimodal embedding path.
public final class MiMoV26TextModel: Module, LLMModel, KVCacheDimensionProvider,
    SafetensorsLoadKeyExcluding
{
    public var preservesCheckpointParameterDTypes: Bool { true }
    // Auxiliary towers share shards with the text backbone. Keep them out of
    // the text load before allocating. The mapped diagnostic uses exact spans.
    public var requiresExactTensorMmapBuffers: Bool { true }
    // Owned weights are the normal path. The explicit mapped override exists
    // for controlled residency comparisons, never as an automatic low-RAM fallback.
    private let expertStorage: MixedQuantizedExpertCatalog.Storage =
        RuntimeEnvironment.value("VMLX_MIMO_EXPERT_STORAGE") == "mapped" ? .mapped : .resident
    public var requiresResidentSafetensorsWeights: Bool { expertStorage == .resident }
    private var exactExpertRegions = false
    // Retain eager rotating-cache semantics until whole-forward
    // compiled window masks are qualified across ring wraps.
    public var supportsWholeForwardCompilation: Bool { false }

    /// Load indexed packed expert banks before admitting the model for generation.
    public func configure(modelDirectory: URL) throws {
        let index = modelDirectory.appendingPathComponent("model.safetensors.index.json")
        guard FileManager.default.fileExists(atPath: index.path) else { return }
        let routed = configuration.moeLayerFreq.enumerated().compactMap { $0.element == 1 ? $0.offset : nil }
        guard !routed.isEmpty else { return }
        let catalog = try MixedQuantizedExpertCatalog(
            directory: modelDirectory, layerIndices: routed,
            expertCount: configuration.nRoutedExperts!,
            inputDimensions: configuration.hiddenSize, hiddenDimensions: configuration.moeIntermediateSize)
        // Construct everything before mutating the model: a failed mapping
        // leaves the existing parameter graph intact.
        let replacements = try routed.map { index in
            (index, try MixedQuantizedSwitchGLU(catalog: catalog, layer: index,
                                               inputDimensions: configuration.hiddenSize,
                                               storage: expertStorage))
        }
        for (index, replacement) in replacements {
            let moe = model.layers[index].mlp as! MiMoV26MoE
            try moe.update(modules: .unflattened([("switch_mlp", replacement as Module)]),
                           verify: .noUnusedKeys)
            moe.configureCompiledResidentDecode()
        }
        exactExpertRegions = true
    }

    public func excludeFromGenericSafetensorsLoad(key: String) -> Bool {
        if exactExpertRegions {
            let parts = key.split(separator: ".")
            if parts.count == 7, parts[0] == "model", parts[1] == "layers",
                let layer = Int(parts[2]), configuration.moeLayerFreq.indices.contains(layer),
                configuration.moeLayerFreq[layer] == 1, parts[3] == "mlp",
                parts[4] == "switch_mlp" { return true }
        }
        return !(key.hasPrefix("lm_head.")
            || (key.hasPrefix("model.") && !key.hasPrefix("model.mtp")))
    }
    public let modelType: String
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let hiddenSize: Int
    let configuration: MiMoV2FlashConfiguration
    let model: MiMoV26Backbone
    @ModuleInfo(key: "lm_head") var head: Linear

    public init(_ config: MiMoV2FlashConfiguration) throws {
        guard config.hiddenLayers > 0,
            config.hybridLayerPattern.count == config.hiddenLayers,
            config.moeLayerFreq.count == config.hiddenLayers,
            config.hybridLayerPattern.allSatisfy({ $0 == 0 || $0 == 1 }),
            config.moeLayerFreq.allSatisfy({ $0 == 0 || $0 == 1 }),
            let experts = config.nRoutedExperts, experts >= config.numExpertsPerTok,
            config.numExpertsPerTok > 0, config.nSharedExperts == nil,
            config.nGroup == 1, config.topkGroup == 1,
            config.topkMethod == "noaux_tc", config.scoringFunc == "sigmoid"
        else { throw MiMoV26Error.invalidConfiguration("Invalid V2.6 layer or router configuration") }
        for dim in [config.headDim, config.swaHeadDim] {
            let rotary = Int(Float(dim) * config.partialRotaryFactor)
            guard rotary > 0, rotary <= dim, rotary.isMultiple(of: 2) else {
                throw MiMoV26Error.invalidConfiguration("Invalid partial rotary dimensions")
            }
        }
        configuration = config
        modelType = config.modelType
        vocabularySize = config.vocabularySize
        hiddenSize = config.hiddenSize
        kvHeads = (0..<config.hiddenLayers).map { config.kvHeadsForLayer($0) }
        model = MiMoV26Backbone(config)
        _head.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }

    public func embed(_ tokens: MLXArray) -> MLXArray { model.embeddings(tokens) }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(embeddings: embed(inputs), cache: cache)
    }

    public func callAsFunction(embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embeddings
        for (index, layer) in model.layers.enumerated() {
            let slot = cache?[index]
            let mask = createAttentionMask(
                h: h, cache: slot,
                windowSize: configuration.isSlidingLayer(index) ? configuration.slidingWindowSize : nil)
            h = layer(h, mask: mask, cache: slot)
        }
        return head(model.norm(h))
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        configuration.hybridLayerPattern.map {
            $0 == 1 ? RotatingKVCache(maxSize: configuration.slidingWindowSize, keep: 0) : KVCacheSimple()
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Already converted: retain fused QKV and native mixed expert bytes.
        // Media/MTP are owned by their separate modules, not the text backbone.
        weights.filter { !excludeFromGenericSafetensorsLoad(key: $0.key) }
    }

    public var loraLayers: [Module] { model.layers }
}
