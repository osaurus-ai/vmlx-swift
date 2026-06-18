//
//  QwenImageNative.swift
//  vMLXFluxModels
//
//  Native Qwen-Image (txt2img) pipeline. Ported from mflux
//  `models/qwen/`. Three parts: a Qwen2.5-VL language-model text encoder
//  (this file, first), an MM-DiT transformer, and a 3D causal-conv VAE.
//  Text encoder is bf16 (not quantized); transformer is 4-bit mflux-quant.
//  See FLUX/QWEN_IMAGE_PORT_PLAN.md for the grounded spec.
//

import Foundation
@preconcurrency import MLX
import MLXNN
import MLXRandom
import VMLXTokenizers
import vMLXFluxKit

// MARK: - Qwen2.5 LM text encoder (component "text_encoder")
//
// hidden 3584, 28 layers, 28 attn heads / 4 KV heads (GQA), head_dim 128,
// intermediate 18944, RMSNorm eps 1e-6, theta 1e6, SwiGLU. Causal decoder.
// mRoPE collapses to standard RoPE for text-only input. The qwen-image
// prompt template prepends a system prefix; the first `dropIdx` (34) hidden
// states are dropped, leaving the prompt-conditioning embeddings.

final class QwenTextEncoder {
    private let embedTokens: MFluxEmbedding
    private let layers: [QwenLMLayer]
    private let finalNorm: MFluxRMSNorm
    private let dropIdx: Int

    static let hidden = 3584
    static let heads = 28
    static let kvHeads = 4
    static let headDim = 128
    static let theta: Float = 1_000_000

    init(store: MFluxStore, component: String = "text_encoder", layers: Int = 28, dropIdx: Int = 34) throws {
        self.dropIdx = dropIdx
        self.embedTokens = try store.embedding(component, "encoder.embed_tokens", dimensions: QwenTextEncoder.hidden)
        var ls: [QwenLMLayer] = []
        for i in 0 ..< layers {
            ls.append(try QwenLMLayer(store: store, component: component, index: i))
        }
        self.layers = ls
        self.finalNorm = try store.rmsNorm(component, "encoder.norm", eps: 1e-6)
    }

    /// inputIds (1, seq) Int32 → prompt embeds (1, seq-dropIdx, 3584).
    func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        let seq = inputIds.dim(1)
        let h = encode(inputEmbeddings: embedTokens(inputIds), sequenceLength: seq)
        // drop the template prefix → prompt conditioning.
        let keep = max(0, seq - dropIdx)
        return h[0..., dropIdx ..< (dropIdx + keep), 0...]
    }

    func encodeVisionLanguage(
        inputIDs: MLXArray,
        attentionMask: MLXArray,
        imageFeatures: QwenImageEditVisionFeatures,
        templateDropIndex: Int
    ) throws -> QwenImageEditPromptEmbeddings {
        let seq = inputIDs.dim(1)
        let ids = inputIDs.asArray(Int32.self)
        let imageTokenCount = ids.reduce(0) { count, id in
            count + (id == QwenImageEditPreprocessor.imageTokenID ? 1 : 0)
        }
        try imageFeatures.validateMatches(promptImageTokenCount: imageTokenCount)

        let tokenEmbeds = embedTokens(inputIDs)
        var pieces: [MLXArray] = []
        pieces.reserveCapacity(seq)
        var imageIndex = 0
        for index in 0 ..< seq {
            if ids[index] == QwenImageEditPreprocessor.imageTokenID {
                pieces.append(imageFeatures.imageFeatures[imageIndex, 0...])
                imageIndex += 1
            } else {
                pieces.append(tokenEmbeds[0, index, 0...])
            }
        }
        let inputEmbeddings = stacked(pieces, axis: 0).reshaped([1, seq, QwenTextEncoder.hidden])
        let hidden = encode(inputEmbeddings: inputEmbeddings, sequenceLength: seq)
        let validLength = max(0, min(seq, attentionMask.asArray(Int32.self).reduce(0) { $0 + Int($1) }))
        let keep = max(0, validLength - templateDropIndex)
        let promptEmbeds = hidden[0..., templateDropIndex ..< (templateDropIndex + keep), 0...]
        let promptMask = MLXArray([Int32](repeating: 1, count: keep)).reshaped([1, keep])
        let result = QwenImageEditPromptEmbeddings(
            promptEmbeds: promptEmbeds,
            attentionMask: promptMask,
            templateDropIndex: templateDropIndex,
            sourceSequenceLength: validLength)
        try result.validate()
        return result
    }

    private func encode(inputEmbeddings: MLXArray, sequenceLength seq: Int) -> MLXArray {
        var h = inputEmbeddings
        let (cos, sin) = QwenTextEncoder.ropeCosSin(seq: seq, dtype: h.dtype)
        let mask = QwenTextEncoder.causalMask(seq: seq, dtype: h.dtype)
        for layer in layers {
            h = layer(h, cos: cos, sin: sin, mask: mask)
        }
        return finalNorm(h)
    }

    /// Standard RoPE cos/sin, (seq, headDim).
    static func ropeCosSin(seq: Int, dtype: DType) -> (MLXArray, MLXArray) {
        let half = headDim / 2
        let invFreq = MLXArray((0 ..< half).map { Float(1) / pow(theta, Float(2 * $0) / Float(headDim)) })
        let pos = MLXArray((0 ..< seq).map { Float($0) }).reshaped([seq, 1])
        let freqs = pos * invFreq.reshaped([1, half])           // (seq, half)
        let emb = concatenated([freqs, freqs], axis: -1)        // (seq, headDim)
        return (cos(emb).asType(dtype), sin(emb).asType(dtype))
    }

    static func causalMask(seq: Int, dtype: DType) -> MLXArray {
        var m = [Float](repeating: 0, count: seq * seq)
        for i in 0 ..< seq {
            for j in 0 ..< seq where j > i { m[i * seq + j] = -Float.greatestFiniteMagnitude }
        }
        return MLXArray(m, [1, 1, seq, seq]).asType(dtype)
    }
}

private final class QwenLMLayer {
    private let inputNorm: MFluxRMSNorm
    private let postNorm: MFluxRMSNorm
    private let qProj: MFluxLinear
    private let kProj: MFluxLinear
    private let vProj: MFluxLinear
    private let oProj: MFluxLinear
    private let gate: MFluxLinear
    private let up: MFluxLinear
    private let down: MFluxLinear

    private let heads = QwenTextEncoder.heads
    private let kvHeads = QwenTextEncoder.kvHeads
    private let headDim = QwenTextEncoder.headDim

    init(store: MFluxStore, component: String, index: Int) throws {
        let p = "encoder.layers.\(index)"
        inputNorm = try store.rmsNorm(component, "\(p).input_layernorm", eps: 1e-6)
        postNorm = try store.rmsNorm(component, "\(p).post_attention_layernorm", eps: 1e-6)
        let hidden = QwenTextEncoder.hidden
        qProj = try store.linear(component, "\(p).self_attn.q_proj", inputDimensions: hidden, outputDimensions: heads * headDim, bias: true)
        kProj = try store.linear(component, "\(p).self_attn.k_proj", inputDimensions: hidden, outputDimensions: kvHeads * headDim, bias: true)
        vProj = try store.linear(component, "\(p).self_attn.v_proj", inputDimensions: hidden, outputDimensions: kvHeads * headDim, bias: true)
        oProj = try store.linear(component, "\(p).self_attn.o_proj", inputDimensions: heads * headDim, outputDimensions: hidden, bias: false)
        gate = try store.linear(component, "\(p).mlp.gate_proj", inputDimensions: hidden, outputDimensions: 18944, bias: false)
        up = try store.linear(component, "\(p).mlp.up_proj", inputDimensions: hidden, outputDimensions: 18944, bias: false)
        down = try store.linear(component, "\(p).mlp.down_proj", inputDimensions: 18944, outputDimensions: hidden, bias: false)
    }

    func callAsFunction(_ hidden: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray) -> MLXArray {
        var h = hidden + attention(inputNorm(hidden), cos: cos, sin: sin, mask: mask)
        h = h + mlp(postNorm(h))
        return h
    }

    private func attention(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray) -> MLXArray {
        let seq = x.dim(1)
        var q = qProj(x).reshaped([1, seq, heads, headDim]).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped([1, seq, kvHeads, headDim]).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped([1, seq, kvHeads, headDim]).transposed(0, 2, 1, 3)
        q = QwenLMLayer.applyRope(q, cos: cos, sin: sin)
        k = QwenLMLayer.applyRope(k, cos: cos, sin: sin)
        let kr = repeatKV(k, n: heads / kvHeads)
        let vr = repeatKV(v, n: heads / kvHeads)
        let scale = Float(1.0 / sqrt(Double(headDim)))
        let att = MLX.scaledDotProductAttention(queries: q, keys: kr, values: vr, scale: scale, mask: mask)
        let merged = att.transposed(0, 2, 1, 3).reshaped([1, seq, heads * headDim])
        return oProj(merged)
    }

    private func repeatKV(_ x: MLXArray, n: Int) -> MLXArray {
        if n == 1 { return x }
        let (b, kv, s, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let expanded = x.reshaped([b, kv, 1, s, d])
        let tiled = concatenated(Array(repeating: expanded, count: n), axis: 2)  // (b, kv, n, s, d)
        return tiled.reshaped([b, kv * n, s, d])
    }

    static func applyRope(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        // x (1, H, S, D); cos/sin (S, D) → broadcast over heads.
        let s = x.dim(2), d = x.dim(3)
        let c = cos.reshaped([1, 1, s, d]), sn = sin.reshaped([1, 1, s, d])
        let half = d / 2
        let x1 = x[.ellipsis, 0 ..< half], x2 = x[.ellipsis, half ..< d]
        let rotated = concatenated([-x2, x1], axis: -1)
        return x * c + rotated * sn
    }

    private func mlp(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

// MARK: - Qwen MM-DiT transformer (component "transformer", 4-bit quantized)
//
// 60 dual-stream blocks, inner 3072 (24 heads × 128). img_in 64→3072,
// txt_in 3584→3072 (after txt_norm RMSNorm). Timestep-only modulation.
// 3-axis RoPE (axes [16,56,56], theta 1e4, scale_rope) over img grid + text.

enum QwenRoPE {
    static let theta: Float = 10000
    static let axesDim = [16, 56, 56]

    /// Per-axis (cos,sin) frequency table for positions `indices`, dim `dim`.
    /// Returns flat [pos][dim/2] cos and sin as Swift arrays.
    private static func ropeParams(_ indices: [Float], dim: Int) -> (cos: [[Float]], sin: [[Float]]) {
        let half = dim / 2
        let omega = (0 ..< half).map { Float(1) / pow(theta, Float(2 * $0) / Float(dim)) }
        var c = [[Float]](), s = [[Float]]()
        for p in indices {
            var cr = [Float](repeating: 0, count: half), sr = [Float](repeating: 0, count: half)
            for k in 0 ..< half { let f = p * omega[k]; cr[k] = cos(f); sr[k] = sin(f) }
            c.append(cr); s.append(sr)
        }
        return (c, s)
    }

    /// Build img + txt rope tables. Returns (imgCos,imgSin) shape (imgSeq,64),
    /// (txtCos,txtSin) shape (txtSeq,64). frame=1 for txt2img.
    static func freqs(latentH: Int, latentW: Int, txtLen: Int, dtype: DType)
        -> ((MLXArray, MLXArray), (MLXArray, MLXArray)) {
        freqs(
            imageShapes: [(frame: 1, height: latentH, width: latentW)],
            txtLen: txtLen,
            dtype: dtype)
    }

    static func freqs(
        imageShapes: [(frame: Int, height: Int, width: Int)],
        txtLen: Int,
        dtype: DType
    ) -> ((MLXArray, MLXArray), (MLXArray, MLXArray)) {
        let posIdx = (0 ..< 4096).map { Float($0) }
        let negIdx = (0 ..< 4096).map { Float(-($0) - 1) }.reversed().map { $0 }  // reversed neg
        // per-axis pos/neg tables
        let pf0 = ropeParams(posIdx, dim: axesDim[0])  // frame, 8 freqs
        let pf1 = ropeParams(posIdx, dim: axesDim[1])  // height, 28
        let pf2 = ropeParams(posIdx, dim: axesDim[2])  // width, 28
        let nf1 = ropeParams(negIdx, dim: axesDim[1])
        let nf2 = ropeParams(negIdx, dim: axesDim[2])

        // height index list (scale_rope: center): neg[-(h-h/2):] + pos[:h/2]
        func centeredRows(_ pos: ([[Float]], [[Float]]), _ neg: ([[Float]], [[Float]]), n: Int) -> ([[Float]], [[Float]]) {
            let lo = n - n / 2
            var c = Array(neg.0.suffix(lo)); c += Array(pos.0.prefix(n / 2))
            var s = Array(neg.1.suffix(lo)); s += Array(pos.1.prefix(n / 2))
            return (c, s)
        }

        var imgCos = [Float](), imgSin = [Float]()
        let imageSeqLen = imageShapes.reduce(0) { total, shape in
            total + shape.frame * shape.height * shape.width
        }
        imgCos.reserveCapacity(imageSeqLen * 64)
        imgSin.reserveCapacity(imageSeqLen * 64)

        var maxVid = 0
        for (index, shape) in imageShapes.enumerated() {
            let frame = shape.frame
            let h = shape.height
            let w = shape.width
            maxVid = max(maxVid, h / 2, w / 2)
            let (hc, hs) = centeredRows(pf1, nf1, n: h)   // (h, 28)
            let (wc, ws) = centeredRows(pf2, nf2, n: w)   // (w, 28)
            for frameIndex in 0 ..< frame {
                let frameFreqIndex = index + frameIndex
                for r in 0 ..< h {
                    for c in 0 ..< w {
                        imgCos += pf0.cos[frameFreqIndex]; imgCos += hc[r]; imgCos += wc[c]
                        imgSin += pf0.sin[frameFreqIndex]; imgSin += hs[r]; imgSin += ws[c]
                    }
                }
            }
        }
        let imgCosA = MLXArray(imgCos, [imageSeqLen, 64]).asType(dtype)
        let imgSinA = MLXArray(imgSin, [imageSeqLen, 64]).asType(dtype)

        // txt: pos_freqs[maxVidIndex : +txtLen] across all axes (8+28+28=64)
        var txtCos = [Float](), txtSin = [Float]()
        for j in 0 ..< txtLen {
            let i = maxVid + j
            txtCos += pf0.cos[i]; txtCos += pf1.cos[i]; txtCos += pf2.cos[i]
            txtSin += pf0.sin[i]; txtSin += pf1.sin[i]; txtSin += pf2.sin[i]
        }
        let txtCosA = MLXArray(txtCos, [txtLen, 64]).asType(dtype)
        let txtSinA = MLXArray(txtSin, [txtLen, 64]).asType(dtype)
        return ((imgCosA, imgSinA), (txtCosA, txtSinA))
    }

    /// Apply complex-pair RoPE. x (1, seq, heads, headDim); cos/sin (seq, headDim/2).
    static func apply(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let s = x.shape
        let pairs = xf.reshaped(Array(s.dropLast()) + [-1, 2])  // (1, seq, h, hd/2, 2)
        let real = pairs[.ellipsis, 0], imag = pairs[.ellipsis, 1]
        let c = cos.reshaped([1, cos.dim(0), 1, cos.dim(1)])
        let sn = sin.reshaped([1, sin.dim(0), 1, sin.dim(1)])
        let outReal = real * c - imag * sn
        let outImag = real * sn + imag * c
        let stacked = stacked([outReal, outImag], axis: -1)  // (...,hd/2,2)
        return stacked.reshaped(s).asType(x.dtype)
    }
}

enum QwenGuidance {
    static func computeGuidedNoise(
        positive: MLXArray,
        negative: MLXArray,
        guidance: Float
    ) -> MLXArray {
        let combined = negative + MLXArray(guidance) * (positive - negative)
        let positiveNorm = sqrt(sum(positive * positive, axis: -1, keepDims: true) + MLXArray(Float(1e-12)))
        let combinedNorm = sqrt(sum(combined * combined, axis: -1, keepDims: true) + MLXArray(Float(1e-12)))
        return combined * (positiveNorm / combinedNorm)
    }
}

final class QwenTimeEmbed {
    private let l1: MFluxLinear
    private let l2: MFluxLinear
    init(store: MFluxStore, component: String) throws {
        l1 = try store.linear(component, "time_text_embed.timestep_embedder.linear_1", inputDimensions: 256, outputDimensions: 3072, bias: true)
        l2 = try store.linear(component, "time_text_embed.timestep_embedder.linear_2", inputDimensions: 3072, outputDimensions: 3072, bias: true)
    }
    func callAsFunction(_ timestep: Float) -> MLXArray {
        let proj = QwenTimeEmbed.timeProj(timestep)
        return l2(silu(l1(proj)))
    }
    static func timeProj(_ t: Float) -> MLXArray {
        let half = 128
        let exponent = (0 ..< half).map { -log(Float(10000)) * Float($0) / Float(half) }
        let emb = exponent.map { exp($0) }
        let e = emb.map { Float(1000) * t * $0 }   // scale 1000
        var sc = e.map { sin($0) } + e.map { cos($0) }
        sc = Array(sc[half ..< 2 * half]) + Array(sc[0 ..< half])  // flip sin/cos
        return MLXArray(sc, [1, 256])
    }
}

private final class QwenAttn {
    private let toQ: MFluxLinear, toK: MFluxLinear, toV: MFluxLinear
    private let addQ: MFluxLinear, addK: MFluxLinear, addV: MFluxLinear
    private let normQ: MFluxRMSNorm, normK: MFluxRMSNorm, normAddQ: MFluxRMSNorm, normAddK: MFluxRMSNorm
    private let toOut: MFluxLinear, toAddOut: MFluxLinear
    private let heads = 24, headDim = 128

    init(store: MFluxStore, component: String, p: String) throws {
        let a = "\(p).attn"
        toQ = try store.linear(component, "\(a).to_q", inputDimensions: 3072, outputDimensions: 3072, bias: true)
        toK = try store.linear(component, "\(a).to_k", inputDimensions: 3072, outputDimensions: 3072, bias: true)
        toV = try store.linear(component, "\(a).to_v", inputDimensions: 3072, outputDimensions: 3072, bias: true)
        addQ = try store.linear(component, "\(a).add_q_proj", inputDimensions: 3072, outputDimensions: 3072, bias: true)
        addK = try store.linear(component, "\(a).add_k_proj", inputDimensions: 3072, outputDimensions: 3072, bias: true)
        addV = try store.linear(component, "\(a).add_v_proj", inputDimensions: 3072, outputDimensions: 3072, bias: true)
        normQ = try store.rmsNorm(component, "\(a).norm_q", eps: 1e-6)
        normK = try store.rmsNorm(component, "\(a).norm_k", eps: 1e-6)
        normAddQ = try store.rmsNorm(component, "\(a).norm_added_q", eps: 1e-6)
        normAddK = try store.rmsNorm(component, "\(a).norm_added_k", eps: 1e-6)
        toOut = try store.linear(component, "\(a).attn_to_out.0", inputDimensions: 3072, outputDimensions: 3072, bias: true)
        toAddOut = try store.linear(component, "\(a).to_add_out", inputDimensions: 3072, outputDimensions: 3072, bias: true)
    }

    /// img (1, imgSeq, 3072), txt (1, txtSeq, 3072). Returns (imgOut, txtOut).
    func callAsFunction(_ img: MLXArray, _ txt: MLXArray, imgCos: MLXArray, imgSin: MLXArray,
                        txtCos: MLXArray, txtSin: MLXArray) -> (MLXArray, MLXArray) {
        let imgSeq = img.dim(1), txtSeq = txt.dim(1)
        func heads4(_ x: MLXArray, _ seq: Int) -> MLXArray { x.reshaped([1, seq, heads, headDim]) }
        var iq = normQ(heads4(toQ(img), imgSeq)), ik = normK(heads4(toK(img), imgSeq))
        let iv = heads4(toV(img), imgSeq)
        var tq = normAddQ(heads4(addQ(txt), txtSeq)), tk = normAddK(heads4(addK(txt), txtSeq))
        let tv = heads4(addV(txt), txtSeq)
        iq = QwenRoPE.apply(iq, cos: imgCos, sin: imgSin); ik = QwenRoPE.apply(ik, cos: imgCos, sin: imgSin)
        tq = QwenRoPE.apply(tq, cos: txtCos, sin: txtSin); tk = QwenRoPE.apply(tk, cos: txtCos, sin: txtSin)
        // concat [txt, img] on seq, transpose to (1, heads, seq, headDim)
        let q = concatenated([tq, iq], axis: 1).transposed(0, 2, 1, 3)
        let k = concatenated([tk, ik], axis: 1).transposed(0, 2, 1, 3)
        let v = concatenated([tv, iv], axis: 1).transposed(0, 2, 1, 3)
        let scale = Float(1.0 / sqrt(Double(headDim)))
        let att = MLX.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        let merged = att.transposed(0, 2, 1, 3).reshaped([1, txtSeq + imgSeq, 3072])
        let txtOut = toAddOut(merged[0..., 0 ..< txtSeq, 0...])
        let imgOut = toOut(merged[0..., txtSeq..., 0...])
        return (imgOut, txtOut)
    }
}

private final class QwenFF {
    private let mlpIn: MFluxLinear, mlpOut: MFluxLinear
    init(store: MFluxStore, component: String, prefix: String) throws {
        mlpIn = try store.linear(component, "\(prefix).mlp_in", inputDimensions: 3072, outputDimensions: 12288, bias: true)
        mlpOut = try store.linear(component, "\(prefix).mlp_out", inputDimensions: 12288, outputDimensions: 3072, bias: true)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { mlpOut(geluApproximate(mlpIn(x))) }
}

private final class QwenBlock {
    private let imgMod: MFluxLinear, txtMod: MFluxLinear
    private let attn: QwenAttn
    private let imgFF: QwenFF, txtFF: QwenFF
    init(store: MFluxStore, component: String, index: Int) throws {
        let p = "transformer_blocks.\(index)"
        imgMod = try store.linear(
            component,
            prefixes: ["\(p).img_mod_linear", "\(p).img_norm1.mod_linear"],
            inputDimensions: 3072,
            outputDimensions: 18432,
            bias: true)
        txtMod = try store.linear(
            component,
            prefixes: ["\(p).txt_mod_linear", "\(p).txt_norm1.mod_linear"],
            inputDimensions: 3072,
            outputDimensions: 18432,
            bias: true)
        attn = try QwenAttn(store: store, component: component, p: p)
        imgFF = try QwenFF(store: store, component: component, prefix: "\(p).img_ff")
        txtFF = try QwenFF(store: store, component: component, prefix: "\(p).txt_ff")
    }

    func callAsFunction(_ img0: MLXArray, _ txt0: MLXArray, text: MLXArray,
                        imgCos: MLXArray, imgSin: MLXArray, txtCos: MLXArray, txtSin: MLXArray) -> (MLXArray, MLXArray) {
        var img = img0, txt = txt0
        let im = imgMod(silu(text)), tm = txtMod(silu(text))  // (1, 18432)
        let (im1, im2) = (im[0..., 0 ..< 9216], im[0..., 9216 ..< 18432])
        let (tm1, tm2) = (tm[0..., 0 ..< 9216], tm[0..., 9216 ..< 18432])
        let (imgN, imgGate1) = QwenBlock.modulate(QwenBlock.ln(img), im1)
        let (txtN, txtGate1) = QwenBlock.modulate(QwenBlock.ln(txt), tm1)
        let (imgAttn, txtAttn) = attn(imgN, txtN, imgCos: imgCos, imgSin: imgSin, txtCos: txtCos, txtSin: txtSin)
        img = img + imgGate1 * imgAttn
        txt = txt + txtGate1 * txtAttn
        let (imgN2, imgGate2) = QwenBlock.modulate(QwenBlock.ln(img), im2)
        img = img + imgGate2 * imgFF(imgN2)
        let (txtN2, txtGate2) = QwenBlock.modulate(QwenBlock.ln(txt), tm2)
        txt = txt + txtGate2 * txtFF(txtN2)
        return (img, txt)
    }

    static func ln(_ x: MLXArray) -> MLXArray {  // LayerNorm affine=false, eps 1e-6
        let m = mean(x, axis: -1, keepDims: true)
        let c = x - m
        let v = mean(c * c, axis: -1, keepDims: true)
        return c * rsqrt(v + MLXArray(Float(1e-6)))
    }
    /// mod (1, 9216) = shift|scale|gate (3072 each). Returns (modulated, gate).
    static func modulate(_ x: MLXArray, _ mod: MLXArray) -> (MLXArray, MLXArray) {
        let shift = mod[0..., 0 ..< 3072], scale = mod[0..., 3072 ..< 6144], gate = mod[0..., 6144 ..< 9216]
        let out = x * (1 + scale.expandedDimensions(axis: 1)) + shift.expandedDimensions(axis: 1)
        return (out, gate.expandedDimensions(axis: 1))
    }
}

final class QwenTransformer {
    private let imgIn: MFluxLinear
    private let txtNorm: MFluxRMSNorm
    private let txtIn: MFluxLinear
    private let timeEmbed: QwenTimeEmbed
    private let blocks: [QwenBlock]
    private let normOut: FluxAdaNormContinuous
    private let projOut: MFluxLinear

    init(store: MFluxStore, component: String = "transformer", layers: Int = 60) throws {
        imgIn = try store.linear(component, "img_in", inputDimensions: 64, outputDimensions: 3072, bias: true)
        txtNorm = try store.rmsNorm(component, "txt_norm", eps: 1e-6)
        txtIn = try store.linear(component, "txt_in", inputDimensions: 3584, outputDimensions: 3072, bias: true)
        timeEmbed = try QwenTimeEmbed(store: store, component: component)
        blocks = try (0 ..< layers).map { try QwenBlock(store: store, component: component, index: $0) }
        normOut = try FluxAdaNormContinuous(store: store, component: component, prefix: "norm_out")
        projOut = try store.linear(component, "proj_out", inputDimensions: 3072, outputDimensions: 64, bias: true)
    }

    /// latents (1, imgSeq, 64), promptEmbeds (1, txtSeq, 3584), timestep scalar,
    /// latentH/W (in latent units = px//16). Returns (1, imgSeq, 64).
    func callAsFunction(latents: MLXArray, promptEmbeds: MLXArray, timestep: Float,
                        latentH: Int, latentW: Int) -> MLXArray {
        callAsFunction(
            latents: latents,
            promptEmbeds: promptEmbeds,
            timestep: timestep,
            imageShapes: [(frame: 1, height: latentH, width: latentW)])
    }

    /// Edit path variant: `latents` can include target image latents followed by
    /// static conditioning-image latents. `imageShapes` must describe those
    /// image-token grids in the same order, matching mflux `cond_image_grid`.
    func callAsFunction(
        latents: MLXArray,
        promptEmbeds: MLXArray,
        timestep: Float,
        imageShapes: [(frame: Int, height: Int, width: Int)]
    ) -> MLXArray {
        var img = imgIn(latents)
        var txt = txtIn(txtNorm(promptEmbeds))
        let text = timeEmbed(timestep)
        let txtSeq = txt.dim(1)
        let ((imgCos, imgSin), (txtCos, txtSin)) =
            QwenRoPE.freqs(imageShapes: imageShapes, txtLen: txtSeq, dtype: img.dtype)
        for block in blocks {
            (img, txt) = block(img, txt, text: text, imgCos: imgCos, imgSin: imgSin, txtCos: txtCos, txtSin: txtSin)
        }
        return projOut(normOut(img, text: text))
    }
}

// MARK: - Qwen 3D VAE decoder (operated in 2D since txt2img temporal dim = 1)
//
// Each CausalConv3D(kt) at T=1 reduces to a 2D conv using the LAST temporal
// kernel slice (causal front-pad makes only kernel[..,kt-1,..] reach the single
// frame). Resamplers do spatial nearest-2x then a conv that HALVES channels.
// QwenImageRMSNorm = L2-normalize over channels × sqrt(C) × weight.

import MLXRandom

private final class QVConv {
    private let weight: MLXArray   // (out, kh, kw, in) MLX 2D layout
    private let bias: MLXArray?
    private let padding: Int
    init(store: MFluxStore, prefix: String, padding: Int) throws {
        // mflux stores conv3d weight in MLX channels-last layout: (out, kt, kh, kw, in).
        // At T=1 the causal conv reduces to a 2D conv using the LAST temporal kernel slice.
        let w5 = try store.tensor("vae", "\(prefix).weight")  // (out, kt, kh, kw, in)
        let kt = w5.dim(1)
        self.weight = w5[0..., (kt - 1), 0..., 0..., 0...]    // (out, kh, kw, in) — already MLX layout
        self.bias = store.optionalTensor("vae", "\(prefix).bias")
        self.padding = padding
    }
    /// x (b, c, h, w) NCHW → (b, out, h, w).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let nhwc = x.transposed(0, 2, 3, 1)
        var y = conv2d(nhwc, weight, stride: IntOrPair(1), padding: IntOrPair(padding))
        if let bias { y = y + bias }
        return y.transposed(0, 3, 1, 2)
    }
}

private func qvL2Norm(_ x: MLXArray, weight: MLXArray, eps: Float = 1e-12) -> MLXArray {
    // L2 over channel axis (1), scale by sqrt(C), per-channel weight.
    let c = x.dim(1)
    let l2 = sqrt(sum(x * x, axis: 1, keepDims: true))
    let denom = maximum(l2, MLXArray(eps))
    return (x / denom) * MLXArray(Float(c).squareRoot()) * weight.reshaped([1, c, 1, 1])
}

private final class QVResBlock {
    private let n1: MLXArray, n2: MLXArray
    private let c1: QVConv, c2: QVConv
    private let skip: QVConv?
    init(store: MFluxStore, prefix: String) throws {
        n1 = try store.tensor("vae", "\(prefix).norm1.weight")
        n2 = try store.tensor("vae", "\(prefix).norm2.weight")
        c1 = try QVConv(store: store, prefix: "\(prefix).conv1.conv3d", padding: 1)
        c2 = try QVConv(store: store, prefix: "\(prefix).conv2.conv3d", padding: 1)
        skip = store.hasKey("vae", "\(prefix).skip_conv.conv3d.weight")
            ? try QVConv(store: store, prefix: "\(prefix).skip_conv.conv3d", padding: 0) : nil
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = c1(silu(qvL2Norm(x, weight: n1)))
        h = c2(silu(qvL2Norm(h, weight: n2)))
        return h + (skip?(x) ?? x)
    }
}

private final class QVAttn {
    private let norm: MLXArray
    private let toQKV: QVConv1x1
    private let proj: QVConv1x1
    init(store: MFluxStore, prefix: String) throws {
        norm = try store.tensor("vae", "\(prefix).norm.weight")
        toQKV = try QVConv1x1(store: store, prefix: "\(prefix).to_qkv")
        proj = try QVConv1x1(store: store, prefix: "\(prefix).proj")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), c = x.dim(1), h = x.dim(2), w = x.dim(3)
        let normed = qvL2Norm(x, weight: norm)
        let qkv = toQKV(normed)                                  // (b, 3c, h, w)
        let flat = qkv.reshaped([b, 3 * c, h * w]).transposed(0, 2, 1)  // (b, hw, 3c)
        let q = flat[0..., 0..., 0 ..< c]
        let k = flat[0..., 0..., c ..< 2 * c]
        let v = flat[0..., 0..., 2 * c ..< 3 * c]
        let scale = Float(1.0 / sqrt(Double(c)))
        let scores = softmax(matmul(q, k.transposed(0, 2, 1)) * scale, axis: -1)  // (b, hw, hw)
        let out = matmul(scores, v).transposed(0, 2, 1).reshaped([b, c, h, w])
        return proj(out) + x
    }
}

// 1x1 conv (from conv2d weight, not conv3d) — for attention qkv/proj.
private final class QVConv1x1 {
    private let weight: MLXArray
    private let bias: MLXArray?
    init(store: MFluxStore, prefix: String) throws {
        let w = try store.tensor("vae", "\(prefix).weight")   // (out, 1, 1, in) MLX layout
        self.weight = w.reshaped([w.dim(0), w.dim(3)])         // (out, in)
        self.bias = store.optionalTensor("vae", "\(prefix).bias")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), c = x.dim(1), h = x.dim(2), w = x.dim(3)
        let flat = x.reshaped([b, c, h * w])                  // (b, in, hw)
        var y = matmul(weight, flat)                           // (out, ?)... per batch
        // matmul broadcasts: weight (out,in) x flat (b,in,hw) -> (b,out,hw)
        y = y.reshaped([b, weight.dim(0), h, w])
        if let bias { y = y + bias.reshaped([1, bias.dim(0), 1, 1]) }
        return y
    }
}

private final class QVResample {
    private let conv: QVConv1x1OrSpatial
    private let mode: String
    init(store: MFluxStore, prefix: String, mode: String) throws {
        self.mode = mode
        conv = try QVConv1x1OrSpatial(store: store, prefix: "\(prefix).resample_conv")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        switch mode {
        case "upsample2d", "upsample3d", "up":
            // mflux's 3D upsample path reshapes T into batch, nearest-2x spatially,
            // then applies the 2D resample_conv. At T=1 this is pure NCHW 2D.
            let up = repeated(repeated(x, count: 2, axis: 2), count: 2, axis: 3)
            return conv(up)
        case "downsample2d", "downsample3d", "down":
            // mflux pads bottom/right by one pixel before stride-2 resample_conv.
            let b = x.dim(0), c = x.dim(1), h = x.dim(2), w = x.dim(3)
            let right = MLXArray.zeros([b, c, h, 1], dtype: x.dtype)
            var padded = concatenated([x, right], axis: 3)
            let bottom = MLXArray.zeros([b, c, 1, w + 1], dtype: x.dtype)
            padded = concatenated([padded, bottom], axis: 2)
            return conv(padded, stride: 2, padding: 0)
        default:
            preconditionFailure("unsupported Qwen VAE resample mode \(mode)")
        }
    }
}

// resample_conv is a plain nn.Conv2d (k3, pad1) with a 2D weight (out,in,kh,kw).
private final class QVConv1x1OrSpatial {
    private let weight: MLXArray
    private let bias: MLXArray?
    init(store: MFluxStore, prefix: String) throws {
        let w = try store.tensor("vae", "\(prefix).weight")   // (out,kh,kw,in) MLX layout already
        self.weight = w
        self.bias = store.optionalTensor("vae", "\(prefix).bias")
    }
    func callAsFunction(_ x: MLXArray, stride: Int = 1, padding: Int? = nil) -> MLXArray {
        let pad = padding ?? (weight.dim(1) / 2)
        var y = conv2d(x.transposed(0, 2, 3, 1), weight, stride: IntOrPair(stride), padding: IntOrPair(pad))
        if let bias { y = y + bias }
        return y.transposed(0, 3, 1, 2)
    }
}

private final class QVUpBlock {
    private let resnets: [QVResBlock]
    private let resample: QVResample?
    init(store: MFluxStore, index: Int, numRes: Int, upsample: Bool) throws {
        var rs: [QVResBlock] = []
        for l in 0 ..< (numRes + 1) {
            rs.append(try QVResBlock(store: store, prefix: "decoder.up_block\(index).resnets.\(l)"))
        }
        resnets = rs
        resample = upsample
            ? try QVResample(store: store, prefix: "decoder.up_block\(index).upsamplers.0", mode: "up") : nil
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for r in resnets { h = r(h) }
        if let resample { h = resample(h) }
        return h
    }

}

private final class QVDownBlock {
    private let resnets: [QVResBlock]
    private let resample: QVResample?
    init(store: MFluxStore, index: Int, numRes: Int, downsampleMode: String?) throws {
        var rs: [QVResBlock] = []
        for l in 0 ..< numRes {
            rs.append(try QVResBlock(store: store, prefix: "encoder.down_blocks.\(index).resnets.\(l)"))
        }
        resnets = rs
        resample = try downsampleMode.map {
            try QVResample(store: store, prefix: "encoder.down_blocks.\(index).downsamplers.0", mode: $0)
        }
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for r in resnets { h = r(h) }
        if let resample { h = resample(h) }
        return h
    }

    func snapshots(_ x: MLXArray, prefix: String) -> ([QwenImageEditLatentSnapshot], MLXArray) {
        var h = x
        var output: [QwenImageEditLatentSnapshot] = []
        func snap(_ name: String, _ value: MLXArray) {
            eval(value)
            output.append(QwenImageEditLatentSnapshot(name: name, value: value))
        }
        for (index, r) in resnets.enumerated() {
            h = r(h)
            snap("\(prefix)_res_\(index)", h)
        }
        if let resample {
            h = resample(h)
            snap("\(prefix)_resample", h)
        }
        return (output, h)
    }
}

private enum Qwen3DVAEStats {
    static let mean: [Float] = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
    static let std: [Float] = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916]
}

final class Qwen3DVAEEncoder {
    private let convIn: QVConv
    private let down: [QVDownBlock]
    private let midRes0: QVResBlock, midAttn: QVAttn, midRes1: QVResBlock
    private let normOut: MLXArray
    private let convOut: QVConv
    private let quant: QVConv

    init(store: MFluxStore) throws {
        convIn = try QVConv(store: store, prefix: "encoder.conv_in.conv3d", padding: 1)
        down = [
            try QVDownBlock(store: store, index: 0, numRes: 2, downsampleMode: "downsample2d"),
            try QVDownBlock(store: store, index: 1, numRes: 2, downsampleMode: "downsample2d"),
            try QVDownBlock(store: store, index: 2, numRes: 2, downsampleMode: "downsample3d"),
            try QVDownBlock(store: store, index: 3, numRes: 2, downsampleMode: nil),
        ]
        midRes0 = try QVResBlock(store: store, prefix: "encoder.mid_block.resnets.0")
        midAttn = try QVAttn(store: store, prefix: "encoder.mid_block.attentions.0")
        midRes1 = try QVResBlock(store: store, prefix: "encoder.mid_block.resnets.1")
        normOut = try store.tensor("vae", "encoder.norm_out.weight")
        convOut = try QVConv(store: store, prefix: "encoder.conv_out.conv3d", padding: 1)
        quant = try QVConv(store: store, prefix: "quant_conv.conv3d", padding: 0)
    }

    /// image (1, 3, H, W) in [-1,1] NCHW -> normalized latents (1, 16, H/8, W/8).
    func encode(_ image: MLXArray) -> MLXArray {
        var h = convIn(image)
        for d in down { h = d(h) }
        h = midRes1(midAttn(midRes0(h)))
        h = convOut(silu(qvL2Norm(h, weight: normOut)))
        h = quant(h)
        h = h[0..., 0 ..< 16, 0..., 0...]
        let meanA = MLXArray(Qwen3DVAEStats.mean, [1, 16, 1, 1])
        let stdA = MLXArray(Qwen3DVAEStats.std, [1, 16, 1, 1])
        return (h - meanA) / stdA
    }

    func encodeSnapshots(_ image: MLXArray) -> [QwenImageEditLatentSnapshot] {
        var snapshots: [QwenImageEditLatentSnapshot] = []
        func snap(_ name: String, _ value: MLXArray) {
            eval(value)
            snapshots.append(QwenImageEditLatentSnapshot(name: name, value: value))
        }
        var h = convIn(image)
        snap("conv_in", h)
        for (index, d) in down.enumerated() {
            let (downSnapshots, next) = d.snapshots(h, prefix: "down_\(index)")
            h = next
            snapshots.append(contentsOf: downSnapshots)
            snap("down_\(index)", h)
        }
        h = midRes1(midAttn(midRes0(h)))
        snap("mid_block", h)
        h = qvL2Norm(h, weight: normOut)
        snap("norm_out", h)
        h = silu(h)
        snap("silu_out", h)
        h = convOut(h)
        snap("conv_out", h)
        h = quant(h)
        snap("quant", h)
        h = h[0..., 0 ..< 16, 0..., 0...]
        snap("quant_first16", h)
        let meanA = MLXArray(Qwen3DVAEStats.mean, [1, 16, 1, 1])
        let stdA = MLXArray(Qwen3DVAEStats.std, [1, 16, 1, 1])
        let normalized = (h - meanA) / stdA
        snap("normalized", normalized)
        return snapshots
    }
}

final class Qwen3DVAEDecoder {
    private let postQuant: QVConv
    private let convIn: QVConv
    private let midRes0: QVResBlock, midAttn: QVAttn, midRes1: QVResBlock
    private let up: [QVUpBlock]
    private let normOut: MLXArray
    private let convOut: QVConv

    init(store: MFluxStore) throws {
        postQuant = try QVConv(store: store, prefix: "post_quant_conv.conv3d", padding: 0)
        convIn = try QVConv(store: store, prefix: "decoder.conv_in.conv3d", padding: 1)
        midRes0 = try QVResBlock(store: store, prefix: "decoder.mid_block.resnets.0")
        midAttn = try QVAttn(store: store, prefix: "decoder.mid_block.attentions.0")
        midRes1 = try QVResBlock(store: store, prefix: "decoder.mid_block.resnets.1")
        up = [
            try QVUpBlock(store: store, index: 0, numRes: 2, upsample: true),
            try QVUpBlock(store: store, index: 1, numRes: 2, upsample: true),
            try QVUpBlock(store: store, index: 2, numRes: 2, upsample: true),
            try QVUpBlock(store: store, index: 3, numRes: 2, upsample: false),
        ]
        normOut = try store.tensor("vae", "decoder.norm_out.weight")
        convOut = try QVConv(store: store, prefix: "decoder.conv_out.conv3d", padding: 1)
    }

    /// latents (1, 16, h/8, w/8) NCHW → image (1, 3, H, W) in [0,1].
    func decode(_ latents: MLXArray) -> MLXArray {
        let meanA = MLXArray(Qwen3DVAEStats.mean, [1, 16, 1, 1])
        let stdA = MLXArray(Qwen3DVAEStats.std, [1, 16, 1, 1])
        var h = latents * stdA + meanA
        h = postQuant(h)
        h = convIn(h)
        h = midRes1(midAttn(midRes0(h)))
        for u in up { h = u(h) }
        h = convOut(silu(qvL2Norm(h, weight: normOut)))
        return VAEDecoder.postprocess(h)
    }
}

// MARK: - Tokenizer + pipeline

private let qwenGenTemplate = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n%@<|im_end|>\n<|im_start|>assistant\n"

private final class QwenImageTokenizer {
    private let tok: any VMLXTokenizers.Tokenizer
    init(modelPath: URL) async throws {
        tok = try await AutoTokenizer.from(modelFolder: modelPath.appendingPathComponent("tokenizer"), strict: false)
    }
    func ids(_ prompt: String) -> MLXArray {
        let formatted = String(format: qwenGenTemplate, prompt)
        let t = tok.encode(text: formatted, addSpecialTokens: false)
        return MLXArray(t.map(Int32.init)).reshaped([1, t.count])
    }
}

final class QwenImagePipeline {
    private let encoder: QwenTextEncoder
    private let transformer: QwenTransformer
    private let vae: Qwen3DVAEDecoder
    private let tokenizer: QwenImageTokenizer

    init(modelPath: URL) async throws {
        let store = MFluxStore(try WeightLoader.load(from: modelPath))
        encoder = try QwenTextEncoder(store: store)
        transformer = try QwenTransformer(store: store)
        vae = try Qwen3DVAEDecoder(store: store)
        tokenizer = try await QwenImageTokenizer(modelPath: modelPath)
    }

    private static func mark(_ s: String) {
        FileHandle.standardError.write("[qwen] >> \(s)\n".data(using: .utf8)!)
    }
    private static func dbg(_ label: String, _ x: MLXArray) {
        FileHandle.standardError.write("[qwen] \(label) shape=\(x.shape) (evaluating...)\n".data(using: .utf8)!)
        eval(x)
        let f = x.asType(.float32)
        let mn = mean(f).item(Float.self), mx = MLX.max(f).item(Float.self)
        FileHandle.standardError.write("[qwen] \(label) mean=\(mn) max=\(mx) finite=\(mn.isFinite && mx.isFinite)\n".data(using: .utf8)!)
    }

    func generate(prompt: String, negativePrompt: String?, width: Int, height: Int, steps: Int,
                  guidance: Float, seed: UInt64?, progress: (Int, Int, Double?) -> Void) throws -> MLXArray {
        guard width % 16 == 0, height % 16 == 0 else {
            throw FluxError.invalidRequest("Qwen width/height must be divisible by 16")
        }
        QwenImagePipeline.mark("tokenize+encode")
        let promptEmbeds = encoder(tokenizer.ids(prompt))            // (1, txt, 3584)
        let negEmbeds = encoder(tokenizer.ids(negativePrompt ?? " "))
        QwenImagePipeline.dbg("textpos", promptEmbeds)

        let latH = height / 16, latW = width / 16
        let hw = latH * latW
        var latents = MLXRandom.normal([1, hw, 64], key: seed.map(MLXRandom.key))
            .asType(.float32)
        let scheduler = FlowMatchEulerScheduler.qwenImage(steps: steps, imageSeqLen: hw)

        let start = Date()
        for step in 0 ..< steps {
            // QwenTimeEmbed applies the ×1000 scale internally (QwenTimesteps scale=1000),
            // so pass the RAW sigma here (not sigma×1000) to avoid double-scaling.
            let t = scheduler.sigmas[step]
            let np = transformer(latents: latents, promptEmbeds: promptEmbeds, timestep: t, latentH: latH, latentW: latW)
            let nn = transformer(latents: latents, promptEmbeds: negEmbeds, timestep: t, latentH: latH, latentW: latW)
            let guided = QwenGuidance.computeGuidedNoise(
                positive: np,
                negative: nn,
                guidance: guidance)
            latents = scheduler.step(latent: latents, velocity: guided, stepIndex: step)
            eval(latents)
            if step == 0 { QwenImagePipeline.dbg("step0", latents) }
            let el = Date().timeIntervalSince(start)
            progress(step + 1, steps, el / Double(step + 1) * Double(steps - step - 1))
        }
        // unpack flux-style → (1,16,h/8,w/8)
        var unpacked = latents.reshaped([1, latH, latW, 16, 2, 2])
        unpacked = unpacked.transposed(0, 3, 1, 4, 2, 5).reshaped([1, 16, latH * 2, latW * 2])
        QwenImagePipeline.dbg("unpacked", unpacked)
        let image = vae.decode(unpacked)
        QwenImagePipeline.dbg("image", image)
        return image
    }
}
