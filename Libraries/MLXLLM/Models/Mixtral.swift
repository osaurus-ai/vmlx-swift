import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mixtral.py
//
// Mixtral = vanilla Llama/Mistral decoder (RMSNorm, plain RoPE, GQA, no biases) with a sparse
// 8-expert top-2 MoE FFN (`block_sparse_moe`) in place of the dense MLP. Structurally identical to
// PhiMoE except the attention/norm flavor — so this mirrors PhiMoE's decoder/model/sanitize and
// swaps in Llama-style attention. The expert-weight sanitize (block_sparse_moe.experts.N.{w1,w2,w3}
// → stacked switch_mlp) is exactly Mixtral's HF layout.

public struct MixtralConfiguration: Codable, Sendable {
    var modelType: String = "mixtral"
    var vocabularySize: Int = 32000
    var hiddenSize: Int = 4096
    var intermediateSize: Int = 14336
    var hiddenLayers: Int = 32
    var attentionHeads: Int = 32
    var kvHeads: Int = 8
    var maxPositionEmbeddings: Int = 32768
    var rmsNormEps: Float = 1e-5
    var ropeTheta: Float = 1_000_000.0
    var numLocalExperts: Int = 8
    var numExpertsPerToken: Int = 2

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case numLocalExperts = "num_local_experts"
        case numExpertsPerToken = "num_experts_per_tok"
    }
}

private class MixtralAttention: Module {
    let args: MixtralConfiguration
    let scale: Float
    let headDim: Int

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPELayer

    init(_ args: MixtralConfiguration) {
        self.args = args
        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads
        self.headDim = dim / heads
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        self.rope = initializeRope(
            dims: headDim, base: args.ropeTheta,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: args.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, -1, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, -1, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, -1, headDim).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

private class MixtralSparseMoeBlock: Module {
    let layerIdx: Int
    let numExperts: Int
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    init(_ args: MixtralConfiguration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self.numExperts = args.numLocalExperts
        self.topK = args.numExpertsPerToken
        self._gate.wrappedValue = Linear(args.hiddenSize, args.numLocalExperts, bias: false)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.intermediateSize,
            numExperts: args.numLocalExperts)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = gate(x)
        let k = self.topK
        let inds = MLX.stopGradient(
            MLX.argPartition(-gates, kth: k - 1, axis: -1)[.ellipsis, ..<k])
        let scores = MLX.softmax(MLX.takeAlong(gates, inds, axis: -1), axis: -1, precise: true)
        JangPressCanonicalExpertAdvisor.shared.observe(layer: layerIdx, indices: inds)
        let y = switchMLP(x, inds)
        return (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

private class MixtralDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MixtralAttention
    @ModuleInfo(key: "block_sparse_moe") var blockSparseMoe: MixtralSparseMoeBlock
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: MixtralConfiguration, layerIdx: Int) {
        self._selfAttn.wrappedValue = MixtralAttention(args)
        self._blockSparseMoe.wrappedValue = MixtralSparseMoeBlock(args, layerIdx: layerIdx)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        h = h + blockSparseMoe(postAttentionLayerNorm(h))
        return h
    }
}

public class MixtralModelInner: Module {
    let args: MixtralConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [MixtralDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ args: MixtralConfiguration) {
        self.args = args
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { MixtralDecoderLayer(args, layerIdx: $0) }
        self._norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

public class MixtralModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: MixtralModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ args: MixtralConfiguration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.model = MixtralModelInner(args)
        self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Some conversions (VLM-style / JANG) wrap the LLM under a `language_model.` prefix. Strip it so
        // keys match this model's modules (model.*, lm_head.*). The shared helper carries the collision
        // guard: a mixed-provenance re-bake can ship both spellings of a key, and picking between them
        // by dictionary iteration order binds a different tensor from run to run.
        var sanitizedWeights = Weights.stripLanguageModelPrefix(weights)

        // Only fuse if the per-expert weights are still separate (skip if already a JANG-stacked bundle).
        if sanitizedWeights["model.layers.0.block_sparse_moe.experts.0.w1.weight"] == nil {
            return sanitizedWeights
        }
        for l in 0 ..< model.args.hiddenLayers {
            let prefix = "model.layers.\(l)"
            for (n, m) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                for k in ["weight", "scales", "biases"] {
                    guard sanitizedWeights["\(prefix).block_sparse_moe.experts.0.\(n).\(k)"] != nil
                    else { continue }
                    let expertKeys = (0 ..< model.args.numLocalExperts).map { e in
                        "\(prefix).block_sparse_moe.experts.\(e).\(n).\(k)"
                    }
                    // A truncated or malformed bundle can carry expert 0 but be missing a later shard.
                    // Stacking blind would trap. Leave the keys in place instead: the module bind then
                    // reports them, so a bad checkpoint yields a load error rather than a crash.
                    guard expertKeys.allSatisfy({ sanitizedWeights[$0] != nil }) else { continue }
                    let toJoin = expertKeys.compactMap { sanitizedWeights.removeValue(forKey: $0) }
                    sanitizedWeights["\(prefix).block_sparse_moe.switch_mlp.\(m).\(k)"] =
                        MLX.stacked(toJoin)
                }
            }
        }
        return sanitizedWeights
    }
}

// MARK: - LoRA

extension MixtralModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
