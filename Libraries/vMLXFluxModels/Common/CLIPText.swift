//
//  CLIPText.swift
//  vMLXFluxModels
//
//  CLIP-L text encoder (FLUX `text_encoder`). Ported from mflux
//  `flux_text_encoder/clip_encoder/`. Produces the POOLED conditioning vector
//  (768) that feeds the transformer's `vector_in`. pooled = the final-layernorm
//  hidden state at the EOS token (argmax of token ids).
//
//  Arch: token_emb(49408,768)+pos_emb(77,768) → 12 × CLIPEncoderLayer → LayerNorm.
//  Layer: residual{LN1→SDPA(12h×64, causal, scale 1/sqrt(64))}, residual{LN2→MLP}.
//  MLP: fc1(768→3072) → quick_gelu(x·sigmoid(1.702x)) → fc2(3072→768).
//

import Foundation
@preconcurrency import MLX
import vMLXFluxKit

final class CLIPTextEncoder {
    private let tokenEmbedding: MFluxEmbedding
    private let positionEmbedding: MFluxEmbedding
    private let layers: [CLIPLayer]
    private let finalNorm: MFluxLayerNorm

    private static let dim = 768

    init(store: MFluxStore, component: String = "text_encoder", numLayers: Int = 12) throws {
        let e = "text_model.embeddings"
        self.tokenEmbedding = try store.embedding(component, "\(e).token_embedding", dimensions: CLIPTextEncoder.dim)
        self.positionEmbedding = try store.embedding(component, "\(e).position_embedding", dimensions: CLIPTextEncoder.dim)
        var ls: [CLIPLayer] = []
        for i in 0 ..< numLayers {
            ls.append(try CLIPLayer(store: store, component: component, index: i))
        }
        self.layers = ls
        self.finalNorm = try store.layerNorm(component, "text_model.final_layer_norm", eps: 1e-5)
    }

    /// inputIds: (1, seq) Int32. Returns the pooled (1, 768) conditioning vector.
    func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        let seq = inputIds.dim(1)
        let positionIds = MLXArray(Array(Int32(0) ..< Int32(seq)), [1, seq])
        var h = tokenEmbedding(inputIds) + positionEmbedding(positionIds)
        let mask = CLIPTextEncoder.causalMask(seq: seq, dtype: h.dtype)
        for layer in layers {
            h = layer(h, mask: mask)
        }
        h = finalNorm(h)
        // pooled = EOS-token row (CLIP EOS id is the max token id).
        let eos = argMax(inputIds, axis: -1).item(Int.self)
        return h[0, eos].reshaped([1, CLIPTextEncoder.dim])
    }

    private static func causalMask(seq: Int, dtype: DType) -> MLXArray {
        var m = [Float](repeating: 0, count: seq * seq)
        for i in 0 ..< seq {
            for j in 0 ..< seq where j > i {
                m[i * seq + j] = -3.4e38
            }
        }
        return MLXArray(m, [1, 1, seq, seq]).asType(dtype)
    }
}

private final class CLIPLayer {
    private let ln1: MFluxLayerNorm
    private let ln2: MFluxLayerNorm
    private let qProj: MFluxLinear
    private let kProj: MFluxLinear
    private let vProj: MFluxLinear
    private let outProj: MFluxLinear
    private let fc1: MFluxLinear
    private let fc2: MFluxLinear

    private let dim = 768
    private let heads = 12
    private let headDim = 64

    init(store: MFluxStore, component: String, index: Int) throws {
        let p = "text_model.encoder.layers.\(index)"
        self.ln1 = try store.layerNorm(component, "\(p).layer_norm1", eps: 1e-5)
        self.ln2 = try store.layerNorm(component, "\(p).layer_norm2", eps: 1e-5)
        self.qProj = try store.linear(component, "\(p).self_attn.q_proj", inputDimensions: dim, outputDimensions: dim, bias: true)
        self.kProj = try store.linear(component, "\(p).self_attn.k_proj", inputDimensions: dim, outputDimensions: dim, bias: true)
        self.vProj = try store.linear(component, "\(p).self_attn.v_proj", inputDimensions: dim, outputDimensions: dim, bias: true)
        self.outProj = try store.linear(component, "\(p).self_attn.out_proj", inputDimensions: dim, outputDimensions: dim, bias: true)
        self.fc1 = try store.linear(component, "\(p).mlp.fc1", inputDimensions: dim, outputDimensions: 3072, bias: true)
        self.fc2 = try store.linear(component, "\(p).mlp.fc2", inputDimensions: 3072, outputDimensions: dim, bias: true)
    }

    func callAsFunction(_ hidden: MLXArray, mask: MLXArray) -> MLXArray {
        var h = hidden + attention(ln1(hidden), mask: mask)
        h = h + mlp(ln2(h))
        return h
    }

    private func attention(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let seq = x.dim(1)
        let q = shape(qProj(x), seq: seq)
        let k = shape(kProj(x), seq: seq)
        let v = shape(vProj(x), seq: seq)
        let scale = Float(1.0 / sqrt(Double(headDim)))
        let attended = MLX.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        let merged = attended.transposed(0, 2, 1, 3).reshaped([1, seq, dim])
        return outProj(merged)
    }

    private func shape(_ x: MLXArray, seq: Int) -> MLXArray {
        x.reshaped([1, seq, heads, headDim]).transposed(0, 2, 1, 3)
    }

    private func mlp(_ x: MLXArray) -> MLXArray {
        let h = fc1(x)
        return fc2(h * sigmoid(1.702 * h))
    }
}
