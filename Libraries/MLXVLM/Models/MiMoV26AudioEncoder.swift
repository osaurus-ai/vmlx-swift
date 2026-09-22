// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN

struct MiMoV26AudioConfiguration: Decodable, Sendable {
    let channels: Int
    let groupSize: Int
    let hiddenSize: Int
    let layers: Int
    let heads: Int
    let intermediateSize: Int
    let outputSize: Int
    let fullAttention: Bool
    let postNorm: Bool
    let projectionLayers: Int
    let ropeTheta: Float
    let vocabulary: [Int]
    let segmentSize: Int

    enum CodingKeys: String, CodingKey {
        case channels = "audio_channels", groupSize = "group_size", hiddenSize = "input_local_dim"
        case layers = "input_local_layers", heads = "input_local_attn_heads"
        case headDim = "input_local_head_dim", intermediateSize = "input_local_intermediate_size"
        case outputSize = "out_hidden_size", fullAttention = "input_full_attention"
        case postNorm = "add_post_norm", projectionLayers = "projection_layers"
        case ropeTheta = "rope_theta", vocabulary = "speech_vocab_size"
        case partialRotary = "partial_rotary_factor", segmentSize = "audio_segment_size"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        channels = try c.decode(Int.self, forKey: .channels)
        groupSize = try c.decode(Int.self, forKey: .groupSize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        layers = try c.decode(Int.self, forKey: .layers)
        heads = try c.decode(Int.self, forKey: .heads)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        outputSize = try c.decode(Int.self, forKey: .outputSize)
        fullAttention = try c.decode(Bool.self, forKey: .fullAttention)
        postNorm = try c.decode(Bool.self, forKey: .postNorm)
        projectionLayers = try c.decode(Int.self, forKey: .projectionLayers)
        ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        segmentSize = try c.decode(Int.self, forKey: .segmentSize)
        let sizes: [Int]
        if let number = try? c.decode(Int.self, forKey: .vocabulary) { sizes = [number] }
        else {
            let value = try c.decode(String.self, forKey: .vocabulary)
            sizes = value.split(separator: "-").compactMap { Int($0) }
            guard sizes.count == value.split(separator: "-").count else {
                throw VLMError.processing("Invalid MiMo speech vocabulary")
            }
        }
        vocabulary = sizes.count == 1 ? Array(repeating: sizes[0], count: max(0, channels)) : sizes
        let headDim = try c.decode(Int.self, forKey: .headDim)
        let partial = try c.decode(Float.self, forKey: .partialRotary)
        guard channels > 0, groupSize > 0, hiddenSize > 0, layers > 0,
            heads > 0, hiddenSize.isMultiple(of: heads), hiddenSize / heads == headDim,
            headDim.isMultiple(of: 2), intermediateSize > 0, outputSize > 0,
            partial == 1, [1, 2].contains(projectionLayers), ropeTheta > 0,
            vocabulary.count == channels, vocabulary.allSatisfy({ $0 > 0 }), segmentSize > 0 else {
            throw VLMError.processing("Unsupported MiMo audio encoder configuration")
        }
    }
}

enum MiMoV26AudioMath {
    static func gelu(_ x: MLXArray) -> MLXArray { MLXNN.gelu(x.asType(.float32)).asType(x.dtype) }
    static func silu(_ x: MLXArray) -> MLXArray { MLXNN.silu(x.asType(.float32)).asType(x.dtype) }

    static func rotary(length: Int, dimensions: Int, theta: Float, dtype: DType,
                       roundFrequencies: Bool = false) -> (MLXArray, MLXArray) {
        var inverse = MLXArray(stride(from: 0, to: dimensions, by: 2).map {
            1 / pow(theta, Float($0) / Float(dimensions))
        })
        if roundFrequencies { inverse = inverse.asType(dtype).asType(.float32) }
        let positions = MLXArray(Array(0..<length)).asType(.float32)
        let frequencies = positions[0..., .newAxis] * inverse[.newAxis, 0...]
        let phase = concatenated([frequencies, frequencies], axis: -1)
        return (cos(phase).asType(dtype), sin(phase).asType(dtype))
    }

    static func rotate(_ x: MLXArray, cosine: MLXArray, sine: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let rotated = concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
        return x * cosine[.newAxis, .newAxis] + rotated * sine[.newAxis, .newAxis]
    }
}

final class MiMoV26LocalAudioAttention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    let heads: Int
    let headDim: Int
    init(_ c: MiMoV26AudioConfiguration) {
        heads = c.heads; headDim = c.hiddenSize / c.heads
        _q.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        _k.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        _v.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        _output.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
    }
    func callAsFunction(_ x: MLXArray, cosine: MLXArray, sine: MLXArray, causal: Bool) -> MLXArray {
        let shape = [x.dim(0), x.dim(1), heads, headDim]
        let query = MiMoV26AudioMath.rotate(q(x).reshaped(shape).transposed(0, 2, 1, 3), cosine: cosine, sine: sine)
        let key = MiMoV26AudioMath.rotate(k(x).reshaped(shape).transposed(0, 2, 1, 3), cosine: cosine, sine: sine)
        let value = v(x).reshaped(shape).transposed(0, 2, 1, 3)
        let result = MLXFast.scaledDotProductAttention(queries: query, keys: key, values: value,
            scale: pow(Float(headDim), -0.5), mask: causal ? .causal : .none)
        return output(result.transposed(0, 2, 1, 3).reshaped(x.shape))
    }
}

final class MiMoV26LocalAudioMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ c: MiMoV26AudioConfiguration) {
        _gate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _up.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(MiMoV26AudioMath.silu(gate(x)) * up(x)) }
}

final class MiMoV26LocalAudioLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: MiMoV26LocalAudioAttention
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    let mlp: MiMoV26LocalAudioMLP
    init(_ c: MiMoV26AudioConfiguration) {
        _attention.wrappedValue = MiMoV26LocalAudioAttention(c)
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: 1e-6)
        _postNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: 1e-6)
        mlp = MiMoV26LocalAudioMLP(c)
    }
    func callAsFunction(_ x: MLXArray, cosine: MLXArray, sine: MLXArray, causal: Bool) -> MLXArray {
        let h = x + attention(inputNorm(x), cosine: cosine, sine: sine, causal: causal)
        return h + mlp(postNorm(h))
    }
}

final class MiMoV26LocalAudioTransformer: Module {
    let layers: [MiMoV26LocalAudioLayer]
    let norm: RMSNorm?
    init(_ c: MiMoV26AudioConfiguration) {
        layers = (0..<c.layers).map { _ in MiMoV26LocalAudioLayer(c) }
        norm = c.postNorm ? RMSNorm(dimensions: c.hiddenSize, eps: 1e-6) : nil
    }
}

final class MiMoV26AudioProjection: Module, UnaryLayer {
    let fc1: Linear
    let fc2: Linear
    init(input: Int, output: Int) {
        fc1 = Linear(input, input * 4, bias: false)
        fc2 = Linear(input * 4, output, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(MiMoV26AudioMath.gelu(fc1(x))) }
}

/// Codebook rows are grouped per clip before the bidirectional local encoder.
/// A clip's trailing padding must never borrow rows from the next clip.
final class MiMoV26AudioEncoder: Module {
    @ModuleInfo(key: "speech_embeddings") var embeddings: [Embedding]
    @ModuleInfo(key: "input_local_transformer") var transformer: MiMoV26LocalAudioTransformer
    let projection: UnaryLayer
    let configuration: MiMoV26AudioConfiguration
    init(_ c: MiMoV26AudioConfiguration) {
        configuration = c
        _embeddings.wrappedValue = c.vocabulary.map { Embedding(embeddingCount: $0, dimensions: c.hiddenSize) }
        _transformer.wrappedValue = MiMoV26LocalAudioTransformer(c)
        let input = c.groupSize * c.hiddenSize
        projection = c.projectionLayers == 2
            ? MiMoV26AudioProjection(input: input, output: c.outputSize)
            : Linear(input, c.outputSize, bias: false)
    }
    func callAsFunction(_ input: MLXArray) throws -> MLXArray {
        let c = configuration
        guard input.ndim == 2, input.dim(0) > 0, input.dim(1) >= c.channels else {
            throw VLMError.processing("MiMo audio codes have invalid clip dimensions")
        }
        var codes = input[0..., ..<c.channels].asType(.int32)
        let padding = (c.groupSize - codes.dim(0) % c.groupSize) % c.groupSize
        if padding > 0 {
            let last = codes[(codes.dim(0) - 1)..<codes.dim(0), 0...]
            codes = concatenated([codes, repeated(last, count: padding, axis: 0)], axis: 0)
        }
        let grouped = codes.reshaped(-1, c.groupSize, c.channels)
        var x = MLXArray.zeros([grouped.dim(0), c.groupSize, c.hiddenSize], dtype: embeddings[0].weight.dtype)
        for channel in 0..<c.channels { x = x + embeddings[channel](grouped[0..., 0..., channel]) }
        let (cosine, sine) = MiMoV26AudioMath.rotary(length: c.groupSize, dimensions: c.hiddenSize / c.heads,
                                                  theta: c.ropeTheta, dtype: x.dtype)
        for layer in transformer.layers {
            x = layer(x, cosine: cosine, sine: sine, causal: !c.fullAttention)
        }
        if let norm = transformer.norm { x = norm(x) }
        return projection(x.reshaped(grouped.dim(0), -1))
    }
}
