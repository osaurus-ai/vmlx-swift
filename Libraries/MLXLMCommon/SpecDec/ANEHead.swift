// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Accelerate
import Foundation
import MLX

/// Static geometry of one Qwen3.5-family native-MTP head as the ANE runs it.
public struct ANEHeadGeometry: Sendable, Equatable {
    public var hidden: Int
    public var heads: Int
    public var kvHeads: Int
    public var headDim: Int
    /// RoPE dims per head (partial rotary: `headDim * partial_rotary_factor`).
    public var rotaryDims: Int
    public var intermediate: Int
    /// Rows of the lm_head the ANE carries (the draft vocabulary = ids `0 ..< draftVocab`).
    public var draftVocab: Int
    /// KV window positions the head attends over (ring of committed + speculative rows).
    public var window: Int
    public var eps: Float
    /// ANE tile: rows per eval (fp16 plane pitch contract). Row 0 is the draft step.
    public var rows: Int = 32

    public init(hidden: Int, heads: Int, kvHeads: Int, headDim: Int, rotaryDims: Int,
                intermediate: Int, draftVocab: Int, window: Int, eps: Float) {
        self.hidden = hidden; self.heads = heads; self.kvHeads = kvHeads; self.headDim = headDim
        self.rotaryDims = rotaryDims; self.intermediate = intermediate; self.draftVocab = draftVocab
        self.window = window; self.eps = eps
    }

    /// lm_head output-channel chunk: indices must stay exact in fp16 (≤ 16384).
    public static let lmHeadChunk = 16384
    public var lmHeadChunks: Int { (draftVocab + Self.lmHeadChunk - 1) / Self.lmHeadChunk }
}

/// Which head matrix an emitter is asking for. All are `[out, in]` and
/// already dequantized (fp16/bf16/fp32 MLXArray); the emitter requantizes
/// int8 per row for the ANE.
public enum ANEHeadMatrix: Sendable, Hashable {
    case fc            // [H, 2H]   input = concat(norm(embed), norm(hidden))
    case qProj         // [heads*headDim*2, H]  (query | gate per head)
    case kProj         // [kvHeads*headDim, H]
    case vProj         // [kvHeads*headDim, H]
    case oProj         // [H, heads*headDim]
    case gateProj      // [I, H]
    case upProj        // [I, H]
    case downProj      // [H, I]
}

public enum ANEHeadNorm: Sendable, Hashable {
    case preFCHidden, preFCEmbedding, inputLayerNorm, postAttentionLayerNorm, qNorm, kNorm, final
}

/// What a model family supplies so the generic emitter can build its head.
public protocol ANEHeadWeightSource {
    var geometry: ANEHeadGeometry { get }
    func matrix(_ which: ANEHeadMatrix) -> MLXArray
    func normWeight(_ which: ANEHeadNorm) -> MLXArray
    /// lm_head rows `range` as `[rows, H]`.
    func lmHeadRows(_ range: Range<Int>) -> MLXArray
    /// RoPE cos/sin for absolute positions, each `[positions.count, rotaryDims/2]`.
    func ropeTables(positions: [Int]) -> (cos: MLXArray, sin: MLXArray)
    /// Input embedding of one token as `[H]` (whatever the trunk feeds the head).
    func embedding(token: Int) -> MLXArray
}

/// Builds the MIL program + weight blob for one head step from a weight source.
public enum ANEHeadEmitter {

    /// Per-row int8 requantization of an `[n, k]` matrix.
    static func int8Rows(_ m: MLXArray) -> (q: [Int8], scales: [Float16]) {
        let n = m.dim(0), k = m.dim(1)
        let f = m.asType(.float32).asArray(Float.self)
        var q = [Int8](repeating: 0, count: n * k)
        var scales = [Float16](repeating: 0, count: n)
        f.withUnsafeBufferPointer { fp in
            q.withUnsafeMutableBufferPointer { qp in
                var scaled = [Float](repeating: 0, count: k)
                for i in 0 ..< n {
                    let row = UnsafePointer(fp.baseAddress! + i * k)
                    var absmax: Float = 0
                    vDSP_maxmgv(row, 1, &absmax, vDSP_Length(k))
                    let scale = absmax > 0 ? absmax / 127 : 1
                    var inv = 1 / scale
                    vDSP_vsmul(row, 1, &inv, &scaled, 1, vDSP_Length(k))
                    for j in 0 ..< k {
                        qp[i * k + j] = Int8(clamping: Int(scaled[j].rounded()))
                    }
                    scales[i] = Float16(scale)
                }
            }
        }
        return (q, scales)
    }

    /// Emitted program plus the plane byte sizes the caller must allocate,
    /// in the alphabetical input order the ANE binds them.
    public struct Emission {
        public let mil: String
        public let weights: Data
        public let inputByteCounts: [Int]
        public let outputByteCounts: [Int]
        public let geometry: ANEHeadGeometry
    }

    public static func emit(from source: ANEHeadWeightSource) -> Emission {
        let g = source.geometry
        let H = g.hidden, NH = g.heads, KVH = g.kvHeads, HD = g.headDim, R = g.rows, W = g.window
        let ROT = g.rotaryDims, G = NH / KVH
        var b = ANEMILBuilder()

        // Inputs: alphabetical names == binding order.
        let aHidden = b.input("a_hidden", shape: [1, H, 1, R])
        let bEmbed = b.input("b_embed", shape: [1, H, 1, R])
        let cK = b.input("c_kcache", shape: [KVH, 1, HD, W])
        let dV = b.input("d_vcache", shape: [KVH, 1, W, HD])
        let eMask = b.input("e_mask", shape: [1, 1, R, W + R])
        let fCos = b.input("f_cos", shape: [1, 1, ROT / 2, R])
        let gSin = b.input("g_sin", shape: [1, 1, ROT / 2, R])

        func normConst(_ which: ANEHeadNorm, dim: Int, shape: [Int]) -> ANEMILBuilder.Value {
            let w = source.normWeight(which).asType(.float32).asArray(Float.self).map { Float16($0) }
            precondition(w.count == dim)
            return b.constFP16(w, shape: shape, "nw")
        }
        func rmsnorm(_ x: ANEMILBuilder.Value, axis: Int, weight: ANEMILBuilder.Value, _ tag: String) -> ANEMILBuilder.Value {
            let sq = b.mul(x, x, "\(tag)_sq")
            let mean = b.reduceMean(sq, axis: axis, "\(tag)_mean")
            let r = b.rsqrt(mean, epsilon: g.eps, "\(tag)_rsqrt")
            let n = b.mul(x, r, "\(tag)_n")
            return b.mul(n, weight, "\(tag)_w")
        }
        func linear(_ x: ANEMILBuilder.Value, _ m: MLXArray, _ tag: String) -> ANEMILBuilder.Value {
            let n = m.dim(0), k = m.dim(1)
            let kChunks = max(1, (k + 4607) / 4608)
            let kc = k / kChunks
            precondition(kc * kChunks == k, "K \(k) must split evenly into \(kChunks) chunks")
            var acc: ANEMILBuilder.Value?
            for c in 0 ..< kChunks {
                let slab = kChunks == 1 ? m : m[0..., (c * kc) ..< ((c + 1) * kc)]
                let (q, s) = int8Rows(slab)
                let w = b.int8Weight(q, scales: s, n: n, k: kc, "\(tag)_w\(c)")
                let xin = kChunks == 1 ? x : b.slice(x, begin: [0, c * kc, 0, 0], size: [1, kc, 1, R], "\(tag)_x\(c)")
                let p = b.conv(xin, weight: w, "\(tag)_p\(c)")
                acc = acc.map { b.add($0, p, "\(tag)_s\(c)") } ?? p
            }
            return acc!
        }

        // fc over the fused, normalized (embed | hidden) input.
        let nE = rmsnorm(bEmbed, axis: 1, weight: normConst(.preFCEmbedding, dim: H, shape: [1, H, 1, 1]), "pre_e")
        let nH = rmsnorm(aHidden, axis: 1, weight: normConst(.preFCHidden, dim: H, shape: [1, H, 1, 1]), "pre_h")
        let fused = b.concat([nE, nH], axis: 1, "fused")
        let x = linear(fused, source.matrix(.fc), "fc")

        // Attention.
        let n1 = rmsnorm(x, axis: 1, weight: normConst(.inputLayerNorm, dim: H, shape: [1, H, 1, 1]), "ln1")
        let qg = linear(n1, source.matrix(.qProj), "q")          // [1, NH*HD*2, 1, R]
        let qg4 = b.reshape(qg, [1, NH, 2 * HD, R], "qg4")
        let q4 = b.slice(qg4, begin: [0, 0, 0, 0], size: [1, NH, HD, R], "q4")
        let gate4 = b.slice(qg4, begin: [0, 0, HD, 0], size: [1, NH, HD, R], "gate4")
        let k4 = b.reshape(linear(n1, source.matrix(.kProj), "k"), [1, KVH, HD, R], "k4")
        let v4 = b.reshape(linear(n1, source.matrix(.vProj), "v"), [1, KVH, HD, R], "v4")
        let qn = rmsnorm(q4, axis: 2, weight: normConst(.qNorm, dim: HD, shape: [1, 1, HD, 1]), "qn")
        let kn = rmsnorm(k4, axis: 2, weight: normConst(.kNorm, dim: HD, shape: [1, 1, HD, 1]), "kn")

        func rope(_ t: ANEMILBuilder.Value, heads: Int, _ tag: String) -> ANEMILBuilder.Value {
            let half = ROT / 2
            let x1 = b.slice(t, begin: [0, 0, 0, 0], size: [1, heads, half, R], "\(tag)_x1")
            let x2 = b.slice(t, begin: [0, 0, half, 0], size: [1, heads, half, R], "\(tag)_x2")
            let o1 = b.sub(b.mul(x1, fCos, "\(tag)_x1c"), b.mul(x2, gSin, "\(tag)_x2s"), "\(tag)_o1")
            let o2 = b.add(b.mul(x2, fCos, "\(tag)_x2c"), b.mul(x1, gSin, "\(tag)_x1s"), "\(tag)_o2")
            if ROT == HD { return b.concat([o1, o2], axis: 2, "\(tag)_r") }
            let rest = b.slice(t, begin: [0, 0, ROT, 0], size: [1, heads, HD - ROT, R], "\(tag)_rest")
            return b.concat([o1, o2, rest], axis: 2, "\(tag)_r")
        }
        let qr = rope(qn, heads: NH, "qr")                          // [1, NH, HD, R]
        let kr = rope(kn, heads: KVH, "kr")                         // [1, KVH, HD, R]

        // New rows as outputs (the host appends them into the window planes).
        let kNew = b.reshape(kr, [1, KVH * HD, 1, R], "o1_knew")
        let vNew = b.reshape(v4, [1, KVH * HD, 1, R], "o2_vnew")

        // Window + in-tile keys/values.
        let kTile = b.reshape(kr, [KVH, 1, HD, R], "k_tile")
        let kAll = b.concat([cK, kTile], axis: 3, "k_all")          // [KVH, 1, HD, W+R]
        let vTile = b.transpose(b.reshape(v4, [KVH, 1, HD, R], "v_tile4"), perm: [0, 1, 3, 2], "v_tile")
        let vAll = b.concat([dV, vTile], axis: 2, "v_all")          // [KVH, 1, W+R, HD]

        let qGroup = b.reshape(qr, [KVH, G, HD, R], "q_group")
        let qT = b.transpose(qGroup, perm: [0, 1, 3, 2], "q_t")    // [KVH, G, R, HD]
        var scores = b.matmul(qT, kAll, "scores")                   // [KVH, G, R, W+R]
        scores = b.mul(scores, scalar: 1 / Float(HD).squareRoot(), "scores_scaled")
        scores = b.add(scores, eMask, "scores_masked")
        let probs = b.softmax(scores, axis: 3, "probs")
        let ctx = b.matmul(probs, vAll, "ctx")                      // [KVH, G, R, HD]
        let ctxT = b.transpose(ctx, perm: [0, 1, 3, 2], "ctx_t")   // [KVH, G, HD, R]
        let ctxFlat = b.reshape(ctxT, [1, NH * HD, 1, R], "ctx_flat")
        let gateFlat = b.reshape(gate4, [1, NH * HD, 1, R], "gate_flat")
        let gated = b.mul(ctxFlat, b.sigmoid(gateFlat, "gate_sig"), "gated")
        let o = linear(gated, source.matrix(.oProj), "o")
        let h1 = b.add(x, o, "h1")

        // MLP.
        let n2 = rmsnorm(h1, axis: 1, weight: normConst(.postAttentionLayerNorm, dim: H, shape: [1, H, 1, 1]), "ln2")
        let gateP = linear(n2, source.matrix(.gateProj), "gate")
        let upP = linear(n2, source.matrix(.upProj), "up")
        var act = b.mul(b.silu(gateP, "silu"), upP, "act")
        act = b.mul(act, scalar: 1.0 / 16, "act_s")                 // fp16 accumulator headroom
        var down = linear(act, source.matrix(.downProj), "down")
        down = b.mul(down, scalar: 16, "down_s")
        // Outputs bind alphabetically like inputs: name them in unpack order.
        let hOut = b.add(h1, down, "o0_hidden")

        // lm_head over the draft vocab, argmax per chunk on the ANE.
        let nf = rmsnorm(hOut, axis: 1, weight: normConst(.final, dim: H, shape: [1, H, 1, 1]), "lnf")
        var maxes: [ANEMILBuilder.Value] = [], idxs: [ANEMILBuilder.Value] = []
        var rows = 0
        var chunk = 0
        while rows < g.draftVocab {
            let n = min(ANEHeadGeometry.lmHeadChunk, g.draftVocab - rows)
            let logits = linear(nf, source.lmHeadRows(rows ..< (rows + n)), "lm\(chunk)")
            let m = b.reduceMax(logits, axis: 1, "am_max\(chunk)")
            var d = b.sub(logits, m, "am_d\(chunk)")
            d = b.mul(d, scalar: 1024, "am_ds\(chunk)")
            d = b.add(d, scalar: 1, "am_d1\(chunk)")
            let mask = b.clip(d, low: 0, high: 1, "am_mask\(chunk)")
            let ramp = b.constFP16((0 ..< n).map { Float16(Float($0)) }, shape: [1, n, 1, 1], "ramp\(chunk)")
            let idx = b.reduceMax(b.mul(mask, ramp, "am_mr\(chunk)"), axis: 1, "am_idx\(chunk)")
            maxes.append(m); idxs.append(idx)
            rows += n; chunk += 1
        }
        let maxOut = b.concat(maxes, axis: 1, "o3_max")
        let idxOut = b.concat(idxs, axis: 1, "o4_idx")

        let mil = b.program(returning: [hOut, kNew, vNew, maxOut, idxOut])
        let inputs = [H * R, H * R, KVH * HD * W, KVH * W * HD, R * (W + R), ROT / 2 * R, ROT / 2 * R].map { $0 * 2 }
        let outputs = [H * R, KVH * HD * R, KVH * HD * R, chunk * R, chunk * R].map { $0 * 2 }
        return Emission(mil: mil, weights: b.weights, inputByteCounts: inputs, outputByteCounts: outputs, geometry: g)
    }
}
