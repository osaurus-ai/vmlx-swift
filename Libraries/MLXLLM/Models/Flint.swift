// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Flint's converted affine/JANG bundle contract. No sampler or reasoning overrides.
public struct FlintConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let moeIntermediateSize: Int
    public let sharedExpertIntermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let numExperts: Int
    public let topK: Int
    public let vocabularySize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let slidingWindow: Int
    public let layerTypes: [String]
    public let noRopeLayerIndices: [Int]
    public let mlpOnlyLayers: [Int]
    public let normTopKProb: Bool
    public let routerScoreFunction: String
    public let softmaxType: String
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers", intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case attentionHeads = "num_attention_heads", kvHeads = "num_key_value_heads"
        case headDim = "head_dim", numExperts = "num_experts", topK = "num_experts_per_tok"
        case vocabularySize = "vocab_size", rmsNormEps = "rms_norm_eps", ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window", layerTypes = "layer_types"
        case noRopeLayerIndices = "no_rope_layer_indices", mlpOnlyLayers = "mlp_only_layers"
        case normTopKProb = "norm_topk_prob", routerScoreFunction = "router_score_function"
        case softmaxType = "softmax_type", tieWordEmbeddings = "tie_word_embeddings"
    }

    func validate() throws {
        guard modelType == "flint", hiddenSize > 0, hiddenLayers > 0,
            intermediateSize > 0, moeIntermediateSize > 0, sharedExpertIntermediateSize > 0,
            attentionHeads > 0, kvHeads > 0, attentionHeads.isMultiple(of: kvHeads),
            headDim > 0, headDim.isMultiple(of: 2), vocabularySize > 0,
            numExperts >= topK, topK > 0, slidingWindow > 0,
            rmsNormEps.isFinite, rmsNormEps > 0, ropeTheta.isFinite, ropeTheta > 0,
            layerTypes.count == hiddenLayers,
            layerTypes.allSatisfy({ $0 == "full_attention" || $0 == "sliding_attention" }),
            (noRopeLayerIndices + mlpOnlyLayers).allSatisfy({ (0..<hiddenLayers).contains($0) }),
            normTopKProb, routerScoreFunction == "sigmoid", softmaxType == "learnable",
            !tieWordEmbeddings
        else { throw FlintError.invalidConfiguration }
    }
}

enum FlintError: Error { case invalidConfiguration }

final class FlintAttention: Module {
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    @ParameterInfo(key: "softmax_offset") var sink: MLXArray
    let rope: RoPE?
    let heads: Int, kvHeads: Int, headDim: Int

    init(_ c: FlintConfiguration, layer: Int) {
        heads = c.attentionHeads; kvHeads = c.kvHeads; headDim = c.headDim
        _query.wrappedValue = Linear(c.hiddenSize, heads * headDim, bias: false)
        _key.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: false)
        _value.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: false)
        _output.wrappedValue = Linear(heads * headDim, c.hiddenSize, bias: false)
        _queryNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.rmsNormEps)
        _keyNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.rmsNormEps)
        _sink.wrappedValue = MLXArray.zeros([heads], dtype: .bfloat16)
        // NoPE is independent of layer_types; the checkpoint explicitly declares it.
        rope = c.noRopeLayerIndices.contains(layer) ? nil
            : RoPE(dimensions: headDim, traditional: false, base: c.ropeTheta)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
                        cache: KVCache?) -> MLXArray {
        let b = x.dim(0), n = x.dim(1)
        var q = queryNorm(query(x).reshaped(b, n, heads, headDim)).transposed(0, 2, 1, 3)
        var k = keyNorm(key(x).reshaped(b, n, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let v = value(x).reshaped(b, n, kvHeads, headDim).transposed(0, 2, 1, 3)
        if let rope {
            q = applyRotaryPosition(rope, to: q, cache: cache)
            k = applyRotaryPosition(rope, to: k, cache: cache)
        }
        let y = attentionWithCacheUpdateAndSinks(queries: q, keys: k, values: v,
            cache: cache, scale: pow(Float(headDim), -0.5), mask: mask, sinks: sink.asType(q.dtype))
        return output(y.transposed(0, 2, 1, 3).reshaped(b, n, heads * headDim))
    }
}

/// Not a Linear: the bundle's router must never inherit affine quantization.
final class FlintRouter: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "expert_bias") var bias: MLXArray
    let topK: Int
    init(_ c: FlintConfiguration) {
        _weight.wrappedValue = MLXArray.zeros([c.numExperts, c.hiddenSize], dtype: .bfloat16)
        _bias.wrappedValue = MLXArray.zeros([c.numExperts], dtype: .float32)
        topK = c.topK
    }
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        // Reference: BF16 projection, then FP32 sigmoid/selection/normalization.
        let scores = sigmoid(x.matmul(weight.T).asType(.float32))
        let indices = stopGradient(argPartition(-(scores + bias.asType(.float32)),
                                                kth: topK - 1, axis: -1)[.ellipsis, ..<topK])
        let selected = takeAlong(scores, indices, axis: -1)
        return (indices, selected / (selected.sum(axis: -1, keepDims: true) + 1e-20))
    }
}

final class FlintMoE: Module, UnaryLayer {
    let gate: FlintRouter
    @ModuleInfo(key: "switch_mlp") var experts: Module
    @ModuleInfo(key: "shared_expert") var shared: Qwen3MoEMLP
    init(_ c: FlintConfiguration, routed: (any WeightedRoutedExpertLayer)? = nil) {
        gate = FlintRouter(c)
        // Avoid retaining a second fused expert bank during initial integration.
        _experts.wrappedValue = routed ?? SwitchGLU(inputDims: c.hiddenSize,
            hiddenDims: c.moeIntermediateSize, numExperts: c.numExperts, allowFusedGateUpCache: false)
        _shared.wrappedValue = Qwen3MoEMLP(dimensions: c.hiddenSize,
                                        hiddenDimensions: c.sharedExpertIntermediateSize)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = gate(x)
        let routed: MLXArray
        if let affine = experts as? any SwitchGLULayer {
            routed = (affine(x, indices) * weights.asType(x.dtype)[.ellipsis, .newAxis]).sum(axis: -2)
        } else {
            routed = (experts as! any WeightedRoutedExpertLayer).callRouted(
                x, indices: indices, scores: weights.asType(x.dtype))
        }
        return routed + shared(x)
    }
}

final class FlintLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: FlintAttention
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    let mlp: UnaryLayer
    init(_ c: FlintConfiguration, layer: Int, routed: (any WeightedRoutedExpertLayer)? = nil) {
        _attention.wrappedValue = FlintAttention(c, layer: layer)
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        mlp = c.mlpOnlyLayers.contains(layer)
            ? Qwen3MoEMLP(dimensions: c.hiddenSize, hiddenDimensions: c.intermediateSize) : FlintMoE(c, routed: routed)
    }
    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
                        cache: KVCache?) -> MLXArray {
        let h = x + attention(inputNorm(x), mask: mask, cache: cache)
        return h + mlp(postNorm(h))
    }
}

final class FlintBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embeddings: Embedding
    let layers: [FlintLayer]
    let norm: RMSNorm
    init(_ c: FlintConfiguration, routed: [Int: any WeightedRoutedExpertLayer] = [:]) {
        _embeddings.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        layers = (0..<c.hiddenLayers).map { FlintLayer(c, layer: $0, routed: routed[$0]) }
        norm = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }
}

public final class FlintModel: Module, LLMModel, KVCacheDimensionProvider, SafetensorsLoadKeyExcluding, PrefixCacheInvariantModel {
    private var customTensorNames: Set<String> = []
    let configuration: FlintConfiguration
    let model: FlintBackbone
    @ModuleInfo(key: "lm_head") var head: Linear
    public var vocabularySize: Int { configuration.vocabularySize }
    public var kvHeads: [Int] { Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers) }
    public var loraLayers: [Module] { model.layers }
    public var preservesCheckpointParameterDTypes: Bool { true }
    public var cacheStorageDTypeIdentity: String? { "flint-bf16-sinks-v1" }
    public var supportsWholeForwardCompilation: Bool { false }
    public var statesBeforeFirstMediaTokenAreMediaIndependent: Bool { false }
    // Text-only causal attention. Serialized prefix state (including wrapped
    // SWA rings and learned sinks) is exact for each subsequent suffix; see
    // FlintRuntimeTests.chunkedAndDiskRestore before broadening this contract.
    public var exactPrefixRestoreIsSuffixIndependent: Bool { true }

    public init(_ c: FlintConfiguration) throws {
        try c.validate()
        configuration = c
        model = FlintBackbone(c)
        _head.wrappedValue = Linear(c.hiddenSize, c.vocabularySize, bias: false)
    }

    init(_ c: FlintConfiguration, routed: [Int: any WeightedRoutedExpertLayer],
         excluded: Set<String>) throws {
        try c.validate()
        let sparse = Set(0..<c.hiddenLayers).subtracting(c.mlpOnlyLayers)
        guard Set(routed.keys) == sparse,
            JANGHModelPreparation.isCanonicalRoutedExclusionSet(excluded, routedLayers: sparse)
        else { throw FlintError.invalidConfiguration }
        configuration = c
        customTensorNames = excluded
        model = FlintBackbone(c, routed: routed)
        _head.wrappedValue = Linear(c.hiddenSize, c.vocabularySize, bias: false)
    }

    public func excludeFromGenericSafetensorsLoad(key: String) -> Bool { customTensorNames.contains(key) }

    /// Full-position logits remain available for numerical parity and teacher forcing.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        head(model.norm(hiddenStates(inputs, cache: cache)))
    }

    /// Generation consumes only the last prompt position. Project that row instead
    /// of allocating [batch, prompt, vocabulary] logits, as REC's prefill does.
    /// Cache updates still process every input position; no token is skipped.
    public func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        let hidden = hiddenStates(input.tokens, cache: cache)
        return LMOutput(logits: head(model.norm(hidden[0..., (hidden.dim(1) - 1)..<hidden.dim(1), 0...])))
    }

    private func hiddenStates(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // FP16 affine embedding metadata does not define the activation dtype.
        // Super experts overflow FP16; preserve stored weights and cast activations only.
        var h = model.embeddings(inputs).asType(.bfloat16)
        for (i, layer) in model.layers.enumerated() {
            let slot = cache?[i]
            let mask = createAttentionMask(h: h, cache: slot,
                windowSize: configuration.layerTypes[i] == "sliding_attention"
                    ? configuration.slidingWindow : nil)
            h = layer(h, mask: mask, cache: slot)
        }
        return h
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        configuration.layerTypes.map {
            $0 == "sliding_attention" ? RotatingKVCache(maxSize: configuration.slidingWindow, keep: 0)
                : AttentionSinkKVCache(headCount: configuration.kvHeads, headDimension: configuration.headDim)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights
        for i in 0..<configuration.hiddenLayers where !configuration.mlpOnlyLayers.contains(i) {
            let key = "model.layers.\(i).mlp.gate.expert_bias"
            if result[key] == nil { result[key] = MLXArray.zeros([configuration.numExperts], dtype: .float32) }
        }
        return result
    }
}
