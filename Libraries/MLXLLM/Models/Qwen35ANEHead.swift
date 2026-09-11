// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The Qwen3.5-family native-MTP head as an `ANEHeadWeightSource`: hands the
// generic ANE emitter dequantized matrices, norm gains, RoPE tables and
// embeddings straight from the loaded `Qwen35MTPModule`, so the Neural
// Engine program is built from the same weights the GPU head runs.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Dequantized `[out, in]` weight of a Linear, whatever its storage.
func aneDenseWeight(_ linear: Linear) -> MLXArray {
    if let q = linear as? QuantizedLinear {
        return dequantized(q.weight, scales: q.scales, biases: q.biases,
                           groupSize: q.groupSize, bits: q.bits, mode: q.mode)
    }
    return linear.weight
}

func aneDenseEmbedding(_ embedding: Embedding) -> MLXArray {
    if let q = embedding as? QuantizedEmbedding {
        return dequantized(q.weight, scales: q.scales, biases: q.biases,
                           groupSize: q.groupSize, bits: q.bits, mode: q.mode)
    }
    return embedding.weight
}

struct Qwen35ANEHeadWeightSource: ANEHeadWeightSource {
    let geometry: ANEHeadGeometry
    private let mtp: Qwen35MTPModule
    private let layer: Qwen35MTPDecoderLayer
    private let mlp: Qwen3NextMLP
    private let embedTokens: Embedding
    private let lmHead: Linear?

    /// - Parameters:
    ///   - draftVocab: lm_head rows the ANE carries (ids `0 ..< draftVocab`).
    ///   - window: head KV window in positions.
    init?(model: Qwen35TextModel, args: Qwen35TextConfiguration, draftVocab: Int, window: Int) {
        guard let mtp = model.mtp, let layer = mtp.layers.first, layer.mlp is Qwen3NextMLP,
              mtp.layers.count == 1 else { return nil }
        let headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
        self.mtp = mtp
        self.layer = layer
        self.mlp = layer.mlp as! Qwen3NextMLP
        self.embedTokens = model.model.embedTokens
        self.lmHead = model.lmHead
        self.geometry = ANEHeadGeometry(
            hidden: args.hiddenSize, heads: args.attentionHeads, kvHeads: args.kvHeads,
            headDim: headDim, rotaryDims: max(2, Int(Float(headDim) * args.partialRotaryFactor)),
            intermediate: args.intermediateSize,
            draftVocab: min(draftVocab, args.vocabularySize), window: window, eps: args.rmsNormEps)
    }

    func matrix(_ which: ANEHeadMatrix) -> MLXArray {
        switch which {
        case .fc: return aneDenseWeight(mtp.fc)
        case .qProj: return aneDenseWeight(layer.selfAttn.qProj)
        case .kProj: return aneDenseWeight(layer.selfAttn.kProj)
        case .vProj: return aneDenseWeight(layer.selfAttn.vProj)
        case .oProj: return aneDenseWeight(layer.selfAttn.oProj)
        case .gateProj: return aneDenseWeight(mlp.gateProj)
        case .upProj: return aneDenseWeight(mlp.upProj)
        case .downProj: return aneDenseWeight(mlp.downProj)
        }
    }

    func normWeight(_ which: ANEHeadNorm) -> MLXArray {
        switch which {
        case .preFCHidden: return mtp.preFCNormHidden.weight
        case .preFCEmbedding: return mtp.preFCNormEmbedding.weight
        case .inputLayerNorm: return layer.inputLayerNorm.weight
        case .postAttentionLayerNorm: return layer.postAttentionLayerNorm.weight
        case .qNorm: return layer.selfAttn.qNorm.weight
        case .kNorm: return layer.selfAttn.kNorm.weight
        case .final: return mtp.norm.weight
        }
    }

    func lmHeadRows(_ range: Range<Int>) -> MLXArray {
        if let lmHead {
            return aneDenseWeight(lmHead)[range, 0...]
        }
        return aneDenseEmbedding(embedTokens)[range, 0...]
    }

    /// RoPE tables recovered from the layer's own rope: rotating a vector of
    /// ones gives `cos − sin` in the first half and `cos + sin` in the second,
    /// so any variant the model uses (base, scaling, mscale) is captured.
    func ropeTables(positions: [Int]) -> (cos: MLXArray, sin: MLXArray) {
        let rot = geometry.rotaryDims, half = rot / 2
        var cosRows: [MLXArray] = [], sinRows: [MLXArray] = []
        for p in positions {
            let ones = MLXArray.ones([1, 1, 1, geometry.headDim], dtype: .float32)
            let out = layer.selfAttn.rope(ones, offset: p).reshaped(geometry.headDim)
            let a = out[0 ..< half]            // cos - sin
            let c = out[half ..< rot]          // cos + sin
            cosRows.append((a + c) / 2)
            sinRows.append((c - a) / 2)
        }
        return (stacked(cosRows, axis: 0), stacked(sinRows, axis: 0))
    }

    func embedding(token: Int) -> MLXArray {
        embedTokens(MLXArray([Int32(token)])).reshaped(geometry.hidden)
    }

    func embeddingRows(_ range: Range<Int>) -> MLXArray {
        aneDenseEmbedding(embedTokens)[range, 0...]
    }

    var vocabularySize: Int { embedTokens.weight.dim(0) }
}

extension Qwen35TextModel {
    /// Weight source for an ANE drafter, or nil when this head is not the
    /// dense single-layer shape the emitter supports.
    public func aneHeadWeightSource(draftVocab: Int, window: Int) -> ANEHeadWeightSource? {
        Qwen35ANEHeadWeightSource(model: self, args: configuration, draftVocab: draftVocab, window: window)
    }
}

extension Qwen35TextModel: ANEDraftableModel {}

extension Qwen35Model: ANEDraftableModel {
    public func aneHeadWeightSource(draftVocab: Int, window: Int) -> ANEHeadWeightSource? {
        languageModel.aneHeadWeightSource(draftVocab: draftVocab, window: window)
    }
}
