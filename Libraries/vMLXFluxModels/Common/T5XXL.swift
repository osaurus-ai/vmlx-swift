//
//  T5XXL.swift
//  vMLXFluxModels
//
//  T5-XXL encoder (the FLUX/flux2/wan text encoder, `text_encoder_2`).
//  Ported from mflux `flux_text_encoder/t5_encoder/`. Produces per-token hidden
//  states (1, seq, 4096) used as the transformer's text conditioning.
//
//  Arch: Embedding(32128, 4096) → 24 × T5Block → T5LayerNorm(final).
//  Each block: h += SelfAttention(RMSNorm(h)); h += DenseReluDense(RMSNorm(h)).
//  SelfAttention: 64 heads × 64 dim, NO 1/sqrt(d) scaling, additive T5 relative
//  position bias (bucketed). FFN: gated GELU, 4096→10240→4096.
//

import Foundation
@preconcurrency import MLX
import vMLXFluxKit

final class T5XXLEncoder {
    private let component: String
    private let shared: MFluxEmbedding
    private let blocks: [T5Block]
    private let finalNorm: MFluxRMSNorm

    private static let dim = 4096
    private static let heads = 64
    private static let headDim = 64
    private static let layers = 24

    init(store: MFluxStore, component: String = "text_encoder_2") throws {
        self.component = component
        self.shared = try store.embedding(component, "shared", dimensions: T5XXLEncoder.dim)
        var b: [T5Block] = []
        b.reserveCapacity(T5XXLEncoder.layers)
        for i in 0 ..< T5XXLEncoder.layers {
            b.append(try T5Block(store: store, component: component, index: i))
        }
        self.blocks = b
        self.finalNorm = try store.rmsNorm(component, "final_layer_norm", eps: 1e-6)
    }

    /// inputIds: (1, seq) Int32. Returns (1, seq, 4096) per-token embeddings.
    func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        var h = shared(inputIds)
        let seq = h.dim(1)
        let positionBias = T5Block.computeBias(seqLength: seq, biasEmbedding: blocks[0].relativeAttentionBias)
        for block in blocks {
            h = block(h, positionBias: positionBias)
        }
        return finalNorm(h)
    }
}

private final class T5Block {
    private let attnNorm: MFluxRMSNorm
    private let q: MFluxLinear
    private let k: MFluxLinear
    private let v: MFluxLinear
    private let o: MFluxLinear
    let relativeAttentionBias: MFluxEmbedding
    private let ffNorm: MFluxRMSNorm
    private let wi0: MFluxLinear
    private let wi1: MFluxLinear
    private let wo: MFluxLinear

    private let dim = 4096
    private let heads = 64
    private let headDim = 64

    init(store: MFluxStore, component: String, index: Int) throws {
        let a = "t5_blocks.\(index).attention"
        self.attnNorm = try store.rmsNorm(component, "\(a).layer_norm", eps: 1e-6)
        let sa = "\(a).SelfAttention"
        self.q = try store.linear(component, "\(sa).q", inputDimensions: dim, outputDimensions: dim)
        self.k = try store.linear(component, "\(sa).k", inputDimensions: dim, outputDimensions: dim)
        self.v = try store.linear(component, "\(sa).v", inputDimensions: dim, outputDimensions: dim)
        self.o = try store.linear(component, "\(sa).o", inputDimensions: dim, outputDimensions: dim)
        self.relativeAttentionBias = try store.embedding(component, "\(sa).relative_attention_bias", dimensions: heads)
        let f = "t5_blocks.\(index).ff"
        self.ffNorm = try store.rmsNorm(component, "\(f).layer_norm", eps: 1e-6)
        self.wi0 = try store.linear(component, "\(f).DenseReluDense.wi_0", inputDimensions: dim, outputDimensions: 10240)
        self.wi1 = try store.linear(component, "\(f).DenseReluDense.wi_1", inputDimensions: dim, outputDimensions: 10240)
        self.wo = try store.linear(component, "\(f).DenseReluDense.wo", inputDimensions: 10240, outputDimensions: dim)
    }

    func callAsFunction(_ hidden: MLXArray, positionBias: MLXArray) -> MLXArray {
        var h = hidden + selfAttention(attnNorm(hidden), positionBias: positionBias)
        h = h + denseReluDense(ffNorm(h))
        return h
    }

    private func selfAttention(_ x: MLXArray, positionBias: MLXArray) -> MLXArray {
        let seq = x.dim(1)
        let qs = shape(q(x), seq: seq)
        let ks = shape(k(x), seq: seq)
        let vs = shape(v(x), seq: seq)
        var scores = matmul(qs, ks.transposed(0, 1, 3, 2))  // (1, heads, seq, seq) — NO scaling (T5)
        scores = scores + positionBias
        let weights = softmax(scores, axis: -1)
        let attended = matmul(weights, vs)  // (1, heads, seq, headDim)
        let merged = attended.transposed(0, 2, 1, 3).reshaped([1, seq, dim])
        return o(merged)
    }

    private func shape(_ x: MLXArray, seq: Int) -> MLXArray {
        x.reshaped([1, seq, heads, headDim]).transposed(0, 2, 1, 3)
    }

    private func denseReluDense(_ x: MLXArray) -> MLXArray {
        wo(newGELU(wi0(x)) * wi1(x))
    }

    private func newGELU(_ x: MLXArray) -> MLXArray {
        let c = Float(0.7978845608028654)  // sqrt(2/pi)
        return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * (x * x * x))))
    }

    /// Bucketed T5 relative position bias → (1, heads, seq, seq).
    static func computeBias(seqLength: Int, biasEmbedding: MFluxEmbedding) -> MLXArray {
        var buckets = [Int32](repeating: 0, count: seqLength * seqLength)
        for ctx in 0 ..< seqLength {
            for mem in 0 ..< seqLength {
                buckets[ctx * seqLength + mem] = Int32(relativeBucket(memory: mem, context: ctx))
            }
        }
        let bucketArray = MLXArray(buckets, [seqLength, seqLength])
        let values = biasEmbedding(bucketArray)         // (seq, seq, heads)
        let bias = values.transposed(2, 0, 1)           // (heads, seq, seq)
        return bias.reshaped([1, bias.dim(0), bias.dim(1), bias.dim(2)])
    }

    /// HF/mflux T5 relative-position bucketing (bidirectional, 32 buckets, max 128).
    private static func relativeBucket(memory: Int, context: Int) -> Int {
        let numBuckets = 16  // 32 // 2 for bidirectional
        let maxDistance = 128
        var relative = memory - context
        var bucket = 0
        if relative > 0 { bucket += numBuckets }
        relative = abs(relative)
        let maxExact = numBuckets / 2  // 8
        if relative < maxExact {
            bucket += relative
        } else {
            let ratio = log(Double(relative) / Double(maxExact)) / log(Double(maxDistance) / Double(maxExact))
            var large = maxExact + Int(Double(numBuckets - maxExact) * ratio)
            if large > numBuckets - 1 { large = numBuckets - 1 }
            bucket += large
        }
        return bucket
    }
}
