//
//  QwenImage21Native.swift
//  vMLXFluxModels
//
//  Native Qwen-Image-2.1 (generation + reference editing). Ported from mflux
//  `models/qwen21` (the `QwenImage21Edit` variant, which owns the complete
//  checkpoint layout: transformer + Qwen3-VL text/vision encoder + 64-channel
//  RGBA VAE). Text-to-image is the same pipeline with zero reference images.
//
//  This is a separate architecture from Qwen-Image 1.x (QwenImageNative.swift):
//  a 7B single-stream block-causal DiT, a Qwen3-VL encoder whose hidden states
//  are taken BEFORE the final norm, and a 16x residual VAE.
//
//  Numerical contract mirrors mflux 0.20 / main `QwenImage21Edit`:
//   - activations bf16, VAE fp32
//   - text tokens read the t=0 modulation row, target tokens the sampled row
//   - text prefix K/V cached after the first denoise step
//   - timestep quantized through bf16 before the time embedding
//

import CoreFoundation
import Foundation
@preconcurrency import MLX
import MLXNN
import MLXRandom
import VMLXTokenizers
import vMLXFluxKit

// MARK: - Config

struct QwenImage21Config {
    // transformer
    var numLayers = 32
    var heads = 32
    var headDim = 128
    var contextInDim = 4096
    var inChannels = 64
    var outChannels = 64
    var mlpRatio = 3
    var axes: [Int] = [16, 56, 56]
    // text
    var vocab = 151936
    var hidden = 4096
    var intermediate = 12288
    var textLayers = 36
    var textHeads = 32
    var kvHeads = 8
    var textHeadDim = 128
    var ropeTheta: Float = 5_000_000
    var mropeSection: [Int] = [24, 20, 20]
    var rmsEps: Float = 1e-6
    var imageTokenID = 151655
    // vision
    var vDepth = 27
    var vHidden = 1152
    var vHeads = 16
    var vIntermediate = 4304
    var vPatch = 16
    var vTemporal = 2
    var vMerge = 2
    var vNumPositions = 2304
    var vOut = 4096
    var deepstack: [Int] = [8, 16, 24]
    // vae
    var vaeBaseDim = 96
    var vaeDecoderBaseDim = 144
    var zDim = 64
    var dimMult: [Int] = [1, 2, 4, 8, 8]
    var numResBlocks = 2
    var temporalDownsample: [Bool] = [false, true, true, true]
    var vaeIn = 4
    var vaeOut = 4
    var latentsMean: [Float] = []
    var latentsStd: [Float] = []
    // processor
    var imageMean: [Float] = [0.5, 0.5, 0.5]
    var imageStd: [Float] = [0.5, 0.5, 0.5]
    var minPixels = 65536
    var maxPixels = 16_777_216

    static func load(_ root: URL) throws -> QwenImage21Config {
        var c = QwenImage21Config()
        func json(_ rel: String) -> [String: Any]? {
            let url = root.appendingPathComponent(rel)
            guard let data = try? Data(contentsOf: url) else { return nil }
            return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        }
        func int(_ d: [String: Any]?, _ k: String) -> Int? { (d?[k] as? NSNumber)?.intValue }
        func flt(_ d: [String: Any]?, _ k: String) -> Float? { (d?[k] as? NSNumber)?.floatValue }
        func ints(_ d: [String: Any]?, _ k: String) -> [Int]? { (d?[k] as? [NSNumber])?.map(\.intValue) }
        func flts(_ d: [String: Any]?, _ k: String) -> [Float]? { (d?[k] as? [NSNumber])?.map(\.floatValue) }

        guard let t = json("transformer/config.json") else {
            throw FluxError.invalidRequest("Qwen-Image-2.1: missing transformer/config.json")
        }
        c.numLayers = int(t, "num_layers") ?? c.numLayers
        c.heads = int(t, "num_attention_heads") ?? c.heads
        c.headDim = int(t, "attention_head_dim") ?? c.headDim
        c.contextInDim = int(t, "context_in_dim") ?? c.contextInDim
        c.inChannels = int(t, "in_channels") ?? c.inChannels
        c.outChannels = int(t, "out_channels") ?? c.outChannels
        c.mlpRatio = int(t, "mlp_ratio") ?? c.mlpRatio
        c.axes = ints(t, "axes_dims_rope") ?? c.axes

        guard let te = json("text_encoder/config.json") else {
            throw FluxError.invalidRequest("Qwen-Image-2.1: missing text_encoder/config.json")
        }
        let text = te["text_config"] as? [String: Any]
        c.vocab = int(text, "vocab_size") ?? c.vocab
        c.hidden = int(text, "hidden_size") ?? c.hidden
        c.intermediate = int(text, "intermediate_size") ?? c.intermediate
        c.textLayers = int(text, "num_hidden_layers") ?? c.textLayers
        c.textHeads = int(text, "num_attention_heads") ?? c.textHeads
        c.kvHeads = int(text, "num_key_value_heads") ?? c.kvHeads
        c.textHeadDim = int(text, "head_dim") ?? c.textHeadDim
        c.ropeTheta = flt(text, "rope_theta") ?? c.ropeTheta
        c.rmsEps = flt(text, "rms_norm_eps") ?? c.rmsEps
        if let scaling = text?["rope_scaling"] as? [String: Any], let section = ints(scaling, "mrope_section") {
            c.mropeSection = section
        }
        c.imageTokenID = int(te, "image_token_id") ?? c.imageTokenID
        let vision = te["vision_config"] as? [String: Any]
        c.vDepth = int(vision, "depth") ?? c.vDepth
        c.vHidden = int(vision, "hidden_size") ?? c.vHidden
        c.vHeads = int(vision, "num_heads") ?? c.vHeads
        c.vIntermediate = int(vision, "intermediate_size") ?? c.vIntermediate
        c.vPatch = int(vision, "patch_size") ?? c.vPatch
        c.vTemporal = int(vision, "temporal_patch_size") ?? c.vTemporal
        c.vMerge = int(vision, "spatial_merge_size") ?? c.vMerge
        c.vNumPositions = int(vision, "num_position_embeddings") ?? c.vNumPositions
        c.vOut = int(vision, "out_hidden_size") ?? c.vOut
        c.deepstack = ints(vision, "deepstack_visual_indexes") ?? c.deepstack

        let vae = json("vae/config.json")
        c.vaeBaseDim = int(vae, "base_dim") ?? c.vaeBaseDim
        c.vaeDecoderBaseDim = int(vae, "decoder_base_dim") ?? c.vaeDecoderBaseDim
        c.zDim = int(vae, "z_dim") ?? c.zDim
        c.dimMult = ints(vae, "dim_mult") ?? c.dimMult
        c.numResBlocks = int(vae, "num_res_blocks") ?? c.numResBlocks
        if let td = (vae?["temporal_downsample"] ?? vae?["temperal_downsample"]) as? [NSNumber] {
            c.temporalDownsample = td.map(\.boolValue)
        }
        c.vaeIn = int(vae, "in_channels") ?? c.vaeIn
        c.vaeOut = int(vae, "out_channels") ?? c.vaeOut
        c.latentsMean = flts(vae, "latents_mean") ?? QwenImage21Config.defaultMean
        c.latentsStd = flts(vae, "latents_std") ?? QwenImage21Config.defaultStd
        guard c.latentsMean.count == c.zDim, c.latentsStd.count == c.zDim else {
            throw FluxError.invalidRequest("Qwen-Image-2.1 VAE latent mean/std must have z_dim entries")
        }

        let pre = json("processor/preprocessor_config.json")
        c.imageMean = flts(pre, "image_mean") ?? c.imageMean
        c.imageStd = flts(pre, "image_std") ?? c.imageStd
        if let size = pre?["size"] as? [String: Any] {
            c.minPixels = int(size, "shortest_edge") ?? c.minPixels
            c.maxPixels = int(size, "longest_edge") ?? c.maxPixels
        }
        return c
    }

    static let defaultMean: [Float] = [
        0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722,
        1.8873, -1.7177, -0.651, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363,
        2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
        -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955,
        -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.107, 1.7376,
        -1.0914, -1.9435, -0.2784, -1.368, 0.4809, -0.4433, 0.3764, 0.5729,
        -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505,
        0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1566, 0.0304,
    ]
    static let defaultStd: [Float] = [
        3.2001, 3.2936, 3.4321, 3.0091, 3.106, 4.0379, 4.0705, 3.791,
        3.0785, 3.65, 3.9308, 3.0904, 2.8778, 3.7675, 3.732, 5.0756,
        3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
        3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467,
        3.9037, 3.2947, 5.077, 3.5075, 3.27, 3.4767, 2.8063, 5.1125,
        3.532, 4.7833, 3.1284, 4.181, 3.8527, 3.8317, 3.5603, 4.3867,
        3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.44, 3.4957,
        3.8747, 3.7608, 3.5735, 3.149, 3.7662, 3.6746, 3.4563, 3.8161,
    ]
}

// MARK: - Small shared ops

func q21Range(_ start: Int, _ end: Int) -> MLXArray {
    MLXArray((start ..< end).map { Int32($0) })
}

private func q21Gather(_ source: MLXArray, rows: [Int32]) -> MLXArray {
    // source (1, N, D) → (1, rows.count, D)
    take(source, MLXArray(rows), axis: 1)
}

private func q21Scaled(_ x: MLXArray, _ scale: MLXArray) -> MLXArray {
    x * (scale + 1)
}

private func q21Gelu(_ x: MLXArray) -> MLXArray {
    gelu(x.asType(.float32)).asType(x.dtype)
}

private func q21GeluTanh(_ x: MLXArray) -> MLXArray {
    geluApproximate(x.asType(.float32)).asType(x.dtype)
}

// MARK: - Layout (joint text/reference/target sequence)

struct QwenImage21Layout {
    let gatherRows: [Int32]        // per final position: text row (<T) or T + image row
    let targetMask: [Bool]
    let cos: MLXArray              // (L, headDim/2) fp32
    let sin: MLXArray
    let segments: [(start: Int, end: Int, isText: Bool)]
    let targetTokens: Int
    let prefixLength: Int
    let length: Int

    /// - Parameters:
    ///   - slots: one flag per prompt-embedding row: true where the row is an image placeholder.
    ///   - shapes: (t, h, w) latent grids, references first, target last.
    static func create(slots: [Bool], shapes: [(Int, Int, Int)], axes: [Int]) throws -> QwenImage21Layout {
        guard let last = shapes.last else { throw FluxError.invalidRequest("layout needs a target shape") }
        let targetTokens = last.0 * last.1 * last.2
        guard targetTokens % 4 == 0 else {
            throw FluxError.invalidRequest("The target must contain a multiple of four latent tokens.")
        }
        let textCount = slots.count
        let allSlots = slots + [Bool](repeating: true, count: targetTokens / 4)
        var indices: [Int] = []
        indices.reserveCapacity(allSlots.count * 2)
        for (i, s) in allSlots.enumerated() {
            for _ in 0 ..< (s ? 4 : 1) { indices.append(i) }
        }
        let imageMask = indices.map { allSlots[$0] }
        let imagePositions = imageMask.enumerated().compactMap { $0.element ? $0.offset : nil }
        let lengths = shapes.map { $0.0 * $0.1 * $0.2 }
        guard lengths.reduce(0, +) == imagePositions.count else {
            throw FluxError.invalidRequest("Reference image slots do not match the VAE latent shapes.")
        }
        // gather rows: text positions read the text embedding row, image positions read image rows in order
        var gather = [Int32](repeating: 0, count: indices.count)
        var imageIDs = [Int](repeating: -1, count: indices.count)
        var k = 0
        var cursorImage = 0
        for (pos, slot) in indices.enumerated() {
            if imageMask[pos] {
                gather[pos] = Int32(textCount + k)
                k += 1
            } else {
                gather[pos] = Int32(slot)
            }
        }
        // image ids per image position
        var offset = 0
        for (imageIndex, length) in lengths.enumerated() {
            for j in 0 ..< length { imageIDs[imagePositions[offset + j]] = imageIndex }
            offset += length
        }
        _ = cursorImage
        let length = indices.count
        let prefix = length - targetTokens
        var targetMask = [Bool](repeating: false, count: length)
        for i in prefix ..< length { targetMask[i] = true }

        var frame = [Float](repeating: 0, count: length)
        var height = [Float](repeating: 0, count: length)
        var width = [Float](repeating: 0, count: length)
        var cursor = 0, position = 0
        offset = 0
        for (shape, len) in zip(shapes, lengths) {
            let (_, h, w) = shape
            let start = imagePositions[offset]
            for p in start ..< (start + len) where !imageMask[p] {
                throw FluxError.invalidRequest("Reference image tokens must form contiguous blocks.")
            }
            for (j, p) in (cursor ..< start).enumerated() {
                let v = Float(position + j)
                frame[p] = v; height[p] = v; width[p] = v
            }
            position += start - cursor
            let hStart = -(h - h / 2), wStart = -(w - w / 2)
            for row in 0 ..< h {
                for col in 0 ..< w {
                    let p = start + row * w + col
                    frame[p] = Float(position)
                    height[p] = Float(hStart + row)
                    width[p] = Float(wStart + col)
                }
            }
            position += max(h, w)
            cursor = start + len
            offset += len
        }

        // angles: concat over axes of axis[:, None] * 10000^(-arange(0, dim, 2)/dim), fp32 like numpy
        let half = axes.reduce(0, +) / 2
        var angles = [Float](repeating: 0, count: length * half)
        var col = 0
        for (axisValues, dim) in zip([frame, height, width], axes) {
            let freqs: [Float] = stride(from: 0, to: dim, by: 2).map { i in
                Float(pow(10000.0 as Float, -(Float(i) / Float(dim))))
            }
            for p in 0 ..< length {
                for (j, f) in freqs.enumerated() {
                    angles[p * half + col + j] = axisValues[p] * f
                }
            }
            col += freqs.count
        }
        let angleArray = MLXArray(angles, [length, half])

        var segments: [(Int, Int, Bool)] = []
        var segStart = 0
        if prefix > 0 {
            for end in 1 ... prefix {
                if end == prefix || imageIDs[end] != imageIDs[segStart] {
                    segments.append((segStart, end, imageIDs[segStart] < 0))
                    segStart = end
                }
            }
        }
        return QwenImage21Layout(
            gatherRows: gather,
            targetMask: targetMask,
            cos: MLX.cos(angleArray),
            sin: MLX.sin(angleArray),
            segments: segments.map { (start: $0.0, end: $0.1, isText: $0.2) },
            targetTokens: targetTokens,
            prefixLength: prefix,
            length: length)
    }

    /// (x real/imag interleaved pairs) rotation in fp32, cast back to x's dtype.
    static func rotate(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let shape = x.shape
        var pairShape = Array(shape.dropLast())
        pairShape.append(shape.last! / 2)
        pairShape.append(2)
        let pairs = x.asType(.float32).reshaped(pairShape)
        let real = pairs[.ellipsis, 0]
        let imag = pairs[.ellipsis, 1]
        let c = cos.expandedDimensions(axes: [0, 1])
        let s = sin.expandedDimensions(axes: [0, 1])
        let out = stacked([real * c - imag * s, imag * c + real * s], axis: -1)
        return out.reshaped(shape).asType(x.dtype)
    }
}

// MARK: - Transformer

private final class Q21Attention {
    let toQ, toK, toV, toOut: MFluxLinear
    let normQ, normK: MLXArray
    let heads: Int, headDim: Int

    init(store: MFluxStore, prefix p: String, dim: Int, heads: Int, headDim: Int) throws {
        self.heads = heads
        self.headDim = headDim
        let inner = heads * headDim
        toQ = try store.linear("transformer", "\(p).to_q", inputDimensions: dim, outputDimensions: inner)
        toK = try store.linear("transformer", "\(p).to_k", inputDimensions: dim, outputDimensions: inner)
        toV = try store.linear("transformer", "\(p).to_v", inputDimensions: dim, outputDimensions: inner)
        toOut = try store.linear("transformer", "\(p).to_out.0", inputDimensions: inner, outputDimensions: dim)
        normQ = try store.tensor("transformer", "\(p).norm_q.weight")
        normK = try store.tensor("transformer", "\(p).norm_k.weight")
    }

    func callAsFunction(
        _ x: MLXArray, layout: QwenImage21Layout, cos: MLXArray, sin: MLXArray,
        cached: (MLXArray, MLXArray)?, extract: Bool
    ) -> (MLXArray, (MLXArray, MLXArray)?) {
        let b = x.dim(0), length = x.dim(1), dim = x.dim(2)
        func split(_ y: MLXArray) -> MLXArray {
            y.reshaped([b, length, heads, headDim]).transposed(0, 2, 1, 3)
        }
        var q = split(toQ(x))
        var k = split(toK(x))
        var v = split(toV(x))
        q = QwenImage21Layout.rotate(MLXFast.rmsNorm(q, weight: normQ, eps: 1e-6).asType(v.dtype), cos: cos, sin: sin)
        k = QwenImage21Layout.rotate(MLXFast.rmsNorm(k, weight: normK, eps: 1e-6).asType(v.dtype), cos: cos, sin: sin)
        var stored: (MLXArray, MLXArray)?
        if let cached {
            k = concatenated([cached.0, k], axis: 2)
            v = concatenated([cached.1, v], axis: 2)
        } else if extract {
            stored = (k[0..., 0..., 0 ..< layout.prefixLength, 0...], v[0..., 0..., 0 ..< layout.prefixLength, 0...])
        }
        let scale = 1 / Float(headDim).squareRoot()
        let output: MLXArray
        if cached != nil {
            output = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        } else {
            var outs: [MLXArray] = []
            let segs = layout.segments + [(start: layout.prefixLength, end: length, isText: false)]
            for seg in segs {
                let qs = q[0..., 0..., seg.start ..< seg.end, 0...]
                let ks = k[0..., 0..., 0 ..< seg.end, 0...]
                let vs = v[0..., 0..., 0 ..< seg.end, 0...]
                if seg.isText {
                    let rows = q21Range(seg.start, seg.end).expandedDimensions(axis: 1)
                    let cols = q21Range(0, seg.end).expandedDimensions(axis: 0)
                    let mask = cols .<= rows
                    outs.append(MLXFast.scaledDotProductAttention(
                        queries: qs, keys: ks, values: vs, scale: scale, mask: .array(mask)))
                } else {
                    outs.append(MLXFast.scaledDotProductAttention(
                        queries: qs, keys: ks, values: vs, scale: scale, mask: .none))
                }
            }
            output = concatenated(outs, axis: 2)
        }
        let merged = output.transposed(0, 2, 1, 3).reshaped([b, length, dim])
        return (toOut(merged), stored)
    }
}

private final class Q21Block {
    let attn: Q21Attention
    let proj, out, gate: MFluxLinear

    init(store: MFluxStore, index: Int, cfg: QwenImage21Config) throws {
        let p = "transformer_blocks.\(index)"
        let dim = cfg.heads * cfg.headDim
        attn = try Q21Attention(store: store, prefix: "\(p).attn", dim: dim, heads: cfg.heads, headDim: cfg.headDim)
        let mlpHidden = dim * cfg.mlpRatio
        proj = try store.linear("transformer", "\(p).img_mlp.proj", inputDimensions: dim, outputDimensions: mlpHidden)
        gate = try store.linear("transformer", "\(p).img_mlp.gate_layer", inputDimensions: dim, outputDimensions: mlpHidden)
        out = try store.linear("transformer", "\(p).img_mlp.out", inputDimensions: mlpHidden, outputDimensions: dim)
    }

    func callAsFunction(
        _ x: MLXArray, mods: [MLXArray], layout: QwenImage21Layout, cos: MLXArray, sin: MLXArray,
        cached: (MLXArray, MLXArray)?, extract: Bool
    ) -> (MLXArray, (MLXArray, MLXArray)?) {
        let (a, stored) = attn(
            q21Scaled(MLXFast.layerNorm(x, weight: nil, bias: nil, eps: 1e-6), mods[0]),
            layout: layout, cos: cos, sin: sin, cached: cached, extract: extract)
        var h = x + tanh(mods[1]) * a
        let m = q21Scaled(MLXFast.layerNorm(h, weight: nil, bias: nil, eps: 1e-6), mods[2])
        h = h + tanh(mods[3]) * out(silu(gate(m)) * proj(m))
        return (h, stored)
    }
}

final class QwenImage21Transformer {
    private let cfg: QwenImage21Config
    private let timeL1, timeL2: MFluxLinear
    private let textNorm: MLXArray
    private let txtIn, txtOut: MFluxLinear
    private let imgIn: MFluxLinear
    private let modulation: MFluxLinear
    private let blocks: [Q21Block]
    private let normOutLinear: MFluxLinear
    private let projOut: MFluxLinear

    init(store: MFluxStore, cfg: QwenImage21Config) throws {
        self.cfg = cfg
        let dim = cfg.heads * cfg.headDim
        timeL1 = try store.linear("transformer", "time_text_embed.timestep_embedder.linear_1", inputDimensions: 256, outputDimensions: dim)
        timeL2 = try store.linear("transformer", "time_text_embed.timestep_embedder.linear_2", inputDimensions: dim, outputDimensions: dim)
        textNorm = try store.tensor("transformer", "txt_in.text_norm.weight")
        txtIn = try store.linear("transformer", "txt_in.in_layer", inputDimensions: cfg.contextInDim, outputDimensions: dim)
        txtOut = try store.linear("transformer", "txt_in.out_layer", inputDimensions: dim, outputDimensions: dim)
        imgIn = try store.linear("transformer", "img_in", inputDimensions: cfg.inChannels, outputDimensions: dim)
        modulation = try store.linear(
            "transformer", prefixes: ["modulation.layers.1", "modulation.1"],
            inputDimensions: dim, outputDimensions: 4 * dim)
        blocks = try (0 ..< cfg.numLayers).map { try Q21Block(store: store, index: $0, cfg: cfg) }
        normOutLinear = try store.linear("transformer", "norm_out.linear", inputDimensions: dim, outputDimensions: dim)
        projOut = try store.linear("transformer", "proj_out", inputDimensions: dim, outputDimensions: cfg.outChannels)
    }

    private func timeEmbed(_ t: MLXArray, dtype: DType) -> MLXArray {
        let half = 128
        let freqs = exp(-log(MLXArray(Float(10000))) * q21Range(0, half).asType(.float32) / Float(half))
        let args = t.asType(.float32).expandedDimensions(axis: 1) * Float(1000) * freqs.expandedDimensions(axis: 0)
        let proj = concatenated([MLX.cos(args), MLX.sin(args)], axis: -1).asType(dtype)
        return timeL2(silu(timeL1(proj)))
    }

    private func textProjection(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        let rrms = rsqrt(mean(f * f, axis: -1, keepDims: true) + Float(1e-6))
        let normed = (f * rrms * (textNorm.asType(.float32) + 1)).asType(x.dtype)
        return txtOut(geluApproximate(txtIn(normed)))
    }

    private static func selectRows(_ value: MLXArray, mask: MLXArray) -> MLXArray {
        // value (2, D): row 0 = sampled t, row 1 = t=0; mask (L) bool → (1, L, D)
        let target = value[0 ..< 1].expandedDimensions(axis: 1)
        let condition = value[1 ..< 2].expandedDimensions(axis: 1)
        return which(mask.reshaped([1, -1, 1]), target, condition)
    }

    /// - Parameters:
    ///   - hidden: (1, imageTokens, 64) packed latents, references first, target last.
    ///   - encoder: (1, T, 4096) prompt embeddings.
    ///   - timestep: (1,) sigma in the activation dtype.
    ///   - cache: per-block prefix K/V; empty on the first step (filled), reused after.
    func callAsFunction(
        hidden: MLXArray, encoder: MLXArray, timestep: MLXArray, layout: QwenImage21Layout,
        cache: inout [(MLXArray, MLXArray)]?
    ) -> MLXArray {
        let cachedPass = cache.map { !$0.isEmpty } ?? false
        let extract = cache != nil && !cachedPass
        var x: MLXArray
        var maskValues = layout.targetMask
        var cos = layout.cos, sin = layout.sin
        if cachedPass {
            let n = hidden.dim(1)
            x = imgIn(hidden[0..., (n - layout.targetTokens) ..< n, 0...])
            maskValues = Array(layout.targetMask[layout.prefixLength...])
            cos = cos[layout.prefixLength...]
            sin = sin[layout.prefixLength...]
        } else {
            let images = imgIn(hidden)
            let text = textProjection(encoder)
            x = q21Gather(concatenated([text, images], axis: 1), rows: layout.gatherRows)
        }
        let dtype = x.dtype
        let t = concatenated([timestep.asType(dtype).reshaped([-1]), MLXArray.zeros([1], dtype: dtype)])
        let time = timeEmbed(t, dtype: dtype)
        let mod = modulation(silu(time))
        let mask = MLXArray(maskValues)
        let mods = split(mod, parts: 4, axis: -1).map { Self.selectRows($0, mask: mask) }
        for (index, block) in blocks.enumerated() {
            let (next, stored) = block(
                x, mods: mods, layout: layout, cos: cos, sin: sin,
                cached: cachedPass ? cache![index] : nil, extract: extract)
            x = next
            if let stored { cache!.append(stored) }
            eval(x)
        }
        let scale = Self.selectRows(normOutLinear(silu(time)), mask: mask)
        let out = projOut(q21Scaled(MLXFast.layerNorm(x, weight: nil, bias: nil, eps: 1e-6), scale))
        let n = out.dim(1)
        return out[0..., (n - layout.targetTokens) ..< n, 0...]
    }
}

// MARK: - Qwen3-VL text encoder (hidden states before the final norm)

private func q21TextRMS(_ x: MLXArray, weight: MLXArray, eps: Float) -> MLXArray {
    let f = x.asType(.float32)
    let normed = f * rsqrt(mean(f * f, axis: -1, keepDims: true) + eps)
    return normed.asType(x.dtype) * weight
}

private func q21RotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
}

private final class Q21TextLayer {
    let inputNorm, postNorm, qNorm, kNorm: MLXArray
    let q, k, v, o, gate, up, down: MFluxLinear
    let heads, kvHeads, headDim: Int
    let eps: Float

    init(store: MFluxStore, index: Int, cfg: QwenImage21Config) throws {
        let p = "language_model.layers.\(index)"
        heads = cfg.textHeads; kvHeads = cfg.kvHeads; headDim = cfg.textHeadDim; eps = cfg.rmsEps
        inputNorm = try store.tensor("text_encoder", "\(p).input_layernorm.weight")
        postNorm = try store.tensor("text_encoder", "\(p).post_attention_layernorm.weight")
        qNorm = try store.tensor("text_encoder", "\(p).self_attn.q_norm.weight")
        kNorm = try store.tensor("text_encoder", "\(p).self_attn.k_norm.weight")
        let h = cfg.hidden
        q = try store.linear("text_encoder", "\(p).self_attn.q_proj", inputDimensions: h, outputDimensions: heads * headDim)
        k = try store.linear("text_encoder", "\(p).self_attn.k_proj", inputDimensions: h, outputDimensions: kvHeads * headDim)
        v = try store.linear("text_encoder", "\(p).self_attn.v_proj", inputDimensions: h, outputDimensions: kvHeads * headDim)
        o = try store.linear("text_encoder", "\(p).self_attn.o_proj", inputDimensions: heads * headDim, outputDimensions: h)
        gate = try store.linear("text_encoder", "\(p).mlp.gate_proj", inputDimensions: h, outputDimensions: cfg.intermediate)
        up = try store.linear("text_encoder", "\(p).mlp.up_proj", inputDimensions: h, outputDimensions: cfg.intermediate)
        down = try store.linear("text_encoder", "\(p).mlp.down_proj", inputDimensions: cfg.intermediate, outputDimensions: h)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray) -> MLXArray {
        let b = x.dim(0), s = x.dim(1)
        let normed = q21TextRMS(x, weight: inputNorm, eps: eps)
        var qs = q21TextRMS(q(normed).reshaped([b, s, heads, headDim]), weight: qNorm, eps: eps).transposed(0, 2, 1, 3)
        var ks = q21TextRMS(k(normed).reshaped([b, s, kvHeads, headDim]), weight: kNorm, eps: eps).transposed(0, 2, 1, 3)
        var vs = v(normed).reshaped([b, s, kvHeads, headDim]).transposed(0, 2, 1, 3)
        let c = cos.expandedDimensions(axis: 1), sn = sin.expandedDimensions(axis: 1)
        qs = qs * c + q21RotateHalf(qs) * sn
        ks = ks * c + q21RotateHalf(ks) * sn
        let rep = heads / kvHeads
        if rep > 1 {
            ks = repeated(ks.expandedDimensions(axis: 2), count: rep, axis: 2).reshaped([b, heads, s, headDim])
            vs = repeated(vs.expandedDimensions(axis: 2), count: rep, axis: 2).reshaped([b, heads, s, headDim])
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: qs.asType(.float32), keys: ks.asType(.float32), values: vs.asType(.float32),
            scale: 1 / Float(headDim).squareRoot(), mask: .array(mask)
        ).asType(qs.dtype)
        let h = x + o(attended.transposed(0, 2, 1, 3).reshaped([b, s, heads * headDim]))
        let m = q21TextRMS(h, weight: postNorm, eps: eps)
        return h + down(silu(gate(m)) * up(m))
    }
}

final class QwenImage21TextEncoder {
    private let cfg: QwenImage21Config
    private let embed: MFluxEmbedding
    private let layers: [Q21TextLayer]
    private let invFreq: MLXArray
    let vision: QwenImage21Vision

    init(store: MFluxStore, cfg: QwenImage21Config) throws {
        self.cfg = cfg
        embed = try store.embedding("text_encoder", "language_model.embed_tokens", dimensions: cfg.hidden)
        layers = try (0 ..< cfg.textLayers).map { try Q21TextLayer(store: store, index: $0, cfg: cfg) }
        if let stored = store.optionalTensor("text_encoder", "language_model.rotary_emb.inv_freq") {
            invFreq = stored.asType(.float32)
        } else {
            let exponents = MLXArray(stride(from: 0, to: cfg.textHeadDim, by: 2).map { Float($0) }) / Float(cfg.textHeadDim)
            invFreq = 1 / pow(MLXArray(cfg.ropeTheta), exponents)
        }
        vision = try QwenImage21Vision(store: store, cfg: cfg)
    }

    /// Qwen3-VL 3-axis positions; image placeholders use (t, h, w) of their merged grid.
    func positionIDs(ids: [Int32], grids: [[Int]]) -> [[Int32]] {
        var result = [[Int32]](repeating: [Int32](repeating: 0, count: ids.count), count: 3)
        var cursor = 0, position = 0
        for grid in grids {
            let t = grid[0], h = grid[1] / cfg.vMerge, w = grid[2] / cfg.vMerge
            guard let start = ids[cursor...].firstIndex(of: Int32(cfg.imageTokenID)) else { break }
            for (j, p) in (cursor ..< start).enumerated() {
                for a in 0 ..< 3 { result[a][p] = Int32(position + j) }
            }
            position += start - cursor
            let length = t * h * w
            for idx in 0 ..< length {
                let tt = idx / (h * w), hh = (idx / w) % h, ww = idx % w
                result[0][start + idx] = Int32(tt + position)
                result[1][start + idx] = Int32(hh + position)
                result[2][start + idx] = Int32(ww + position)
            }
            cursor = start + length
            position += max(h, w)
        }
        for (j, p) in (cursor ..< ids.count).enumerated() {
            for a in 0 ..< 3 { result[a][p] = Int32(position + j) }
        }
        return result
    }

    private func rope(positions: [[Int32]], dtype: DType) -> (MLXArray, MLXArray) {
        let s = positions[0].count
        let freqs = (0 ..< 3).map { a -> MLXArray in
            MLXArray(positions[a]).asType(.float32).reshaped([s, 1]) * invFreq.reshaped([1, -1])
        }
        // interleaved mRoPE: axis 1 at offsets 1,4,7..., axis 2 at offsets 2,5,8... (up to section*3)
        let half = invFreq.dim(0)
        var source = [Int32](repeating: 0, count: half)
        for (axis, offset) in [(1, 1), (2, 2)] {
            let limit = cfg.mropeSection[axis] * 3
            for i in stride(from: offset, to: limit, by: 3) where i < half { source[i] = Int32(axis) }
        }
        let all = stacked(freqs, axis: 0)                       // (3, s, half)
        let selector = MLXArray(source).reshaped([1, 1, half])
        let picked = takeAlong(all, broadcast(selector, to: [1, s, half]).asType(.int32), axis: 0).squeezed(axis: 0)
        let emb = concatenated([picked, picked], axis: -1).expandedDimensions(axis: 0)  // (1, s, dim)
        return (MLX.cos(emb).asType(dtype), MLX.sin(emb).asType(dtype))
    }

    /// Returns (1, S, hidden) pre-norm states.
    func encode(ids: [Int32], pixelValues: MLXArray?, grids: [[Int]]) -> MLXArray {
        let s = ids.count
        var hidden = embed(MLXArray(ids).reshaped([1, s]))
        let imagePositions = ids.enumerated().compactMap { $0.element == Int32(cfg.imageTokenID) ? $0.offset : nil }
        var deepstack: [MLXArray] = []
        var isImageRows: [Int32] = []
        if let pixelValues, !imagePositions.isEmpty {
            let (features, stack) = vision(pixelValues.asType(hidden.dtype), grids: grids)
            deepstack = stack
            var rows = (0 ..< s).map { Int32($0) }
            for (k, p) in imagePositions.enumerated() { rows[p] = Int32(s + k) }
            hidden = q21Gather(concatenated([hidden, features.asType(hidden.dtype).expandedDimensions(axis: 0)], axis: 1), rows: rows)
            isImageRows = [Int32](repeating: 0, count: s)
            for (k, p) in imagePositions.enumerated() { isImageRows[p] = Int32(1 + k) }
        }
        let positions = positionIDs(ids: ids, grids: grids)
        let (cos, sin) = rope(positions: positions, dtype: hidden.dtype)
        let idx = q21Range(0, s)
        let mask = (idx.expandedDimensions(axis: 1) .>= idx.expandedDimensions(axis: 0)).reshaped([1, 1, s, s])
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, cos: cos, sin: sin, mask: mask)
            if index < deepstack.count {
                let zeros = MLXArray.zeros([1, 1, hidden.dim(2)], dtype: hidden.dtype)
                let add = q21Gather(concatenated([zeros, deepstack[index].asType(hidden.dtype).expandedDimensions(axis: 0)], axis: 1), rows: isImageRows)
                hidden = hidden + add
            }
            eval(hidden)
        }
        return hidden
    }
}

// MARK: - Qwen3-VL vision tower

private final class Q21Merger {
    let normW, normB: MLXArray
    let fc1, fc2: MFluxLinear
    let postShuffle: Bool
    let width: Int

    init(store: MFluxStore, prefix p: String, cfg: QwenImage21Config, postShuffle: Bool) throws {
        self.postShuffle = postShuffle
        width = cfg.vHidden * cfg.vMerge * cfg.vMerge
        normW = try store.tensor("text_encoder", "\(p).norm.weight")
        normB = try store.tensor("text_encoder", "\(p).norm.bias")
        fc1 = try store.linear("text_encoder", "\(p).linear_fc1", inputDimensions: width, outputDimensions: width, bias: true)
        fc2 = try store.linear("text_encoder", "\(p).linear_fc2", inputDimensions: width, outputDimensions: cfg.vOut, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y: MLXArray
        if postShuffle {
            y = MLXFast.layerNorm(x.reshaped([-1, width]), weight: normW, bias: normB, eps: 1e-6)
        } else {
            y = MLXFast.layerNorm(x, weight: normW, bias: normB, eps: 1e-6).reshaped([-1, width])
        }
        return fc2(q21Gelu(fc1(y)))
    }
}

private final class Q21VisionBlock {
    let n1w, n1b, n2w, n2b: MLXArray
    let qkv, proj, fc1, fc2: MFluxLinear
    let heads, headDim, dim: Int

    init(store: MFluxStore, index: Int, cfg: QwenImage21Config) throws {
        let p = "visual.blocks.\(index)"
        dim = cfg.vHidden; heads = cfg.vHeads; headDim = cfg.vHidden / cfg.vHeads
        n1w = try store.tensor("text_encoder", "\(p).norm1.weight")
        n1b = try store.tensor("text_encoder", "\(p).norm1.bias")
        n2w = try store.tensor("text_encoder", "\(p).norm2.weight")
        n2b = try store.tensor("text_encoder", "\(p).norm2.bias")
        qkv = try store.linear("text_encoder", "\(p).attn.qkv", inputDimensions: dim, outputDimensions: dim * 3, bias: true)
        proj = try store.linear("text_encoder", "\(p).attn.proj", inputDimensions: dim, outputDimensions: dim, bias: true)
        fc1 = try store.linear("text_encoder", "\(p).mlp.linear_fc1", inputDimensions: dim, outputDimensions: cfg.vIntermediate, bias: true)
        fc2 = try store.linear("text_encoder", "\(p).mlp.linear_fc2", inputDimensions: cfg.vIntermediate, outputDimensions: dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, lengths: [Int]) -> MLXArray {
        let n = x.dim(0)
        let normed = MLXFast.layerNorm(x, weight: n1w, bias: n1b, eps: 1e-6)
        let packed = qkv(normed).reshaped([n, 3, heads, headDim]).transposed(1, 0, 2, 3)
        var q = packed[0], k = packed[1]
        let v = packed[2]
        let c = cos.expandedDimensions(axis: 1), s = sin.expandedDimensions(axis: 1)
        q = (q * c + q21RotateHalf(q) * s).asType(v.dtype)
        k = (k * c + q21RotateHalf(k) * s).asType(v.dtype)
        var chunks: [MLXArray] = []
        var offset = 0
        let scale = 1 / Float(headDim).squareRoot()
        for length in lengths {
            let qc = q[offset ..< (offset + length)].transposed(1, 0, 2).expandedDimensions(axis: 0)
            let kc = k[offset ..< (offset + length)].transposed(1, 0, 2).expandedDimensions(axis: 0)
            let vc = v[offset ..< (offset + length)].transposed(1, 0, 2).expandedDimensions(axis: 0)
            let o = MLXFast.scaledDotProductAttention(queries: qc, keys: kc, values: vc, scale: scale, mask: .none)
            chunks.append(o.squeezed(axis: 0).transposed(1, 0, 2))
            offset += length
        }
        let attended = concatenated(chunks, axis: 0).reshaped([n, dim])
        var h = x + proj(attended)
        let m = MLXFast.layerNorm(h, weight: n2w, bias: n2b, eps: 1e-6)
        h = h + fc2(q21GeluTanh(fc1(m)))
        return h
    }
}

final class QwenImage21Vision {
    private let cfg: QwenImage21Config
    private let patchW: MLXArray     // (hidden, temporal*patch*patch*C) in (tp, ph, pw, C) order
    private let patchB: MLXArray
    private let posEmbed: MFluxEmbedding
    private let invFreq: MLXArray
    private let blocks: [Q21VisionBlock]
    private let merger: Q21Merger
    private let deepMergers: [Q21Merger]
    /// Parity-dump stages, only populated when VMLX_QWEN21_DUMP is set.
    var debug: [String: MLXArray] = [:]
    private let dumpEnabled = ProcessInfo.processInfo.environment["VMLX_QWEN21_DUMP"] != nil

    init(store: MFluxStore, cfg: QwenImage21Config) throws {
        self.cfg = cfg
        let w = try store.tensor("text_encoder", "visual.patch_embed.proj.weight")  // (O, tp, ph, pw, C)
        patchW = w.reshaped([w.dim(0), -1])
        patchB = try store.tensor("text_encoder", "visual.patch_embed.proj.bias")
        posEmbed = try store.embedding("text_encoder", "visual.pos_embed", dimensions: cfg.vHidden)
        let headDim = cfg.vHidden / cfg.vHeads
        if let stored = store.optionalTensor("text_encoder", "visual.rotary_pos_emb.inv_freq") {
            invFreq = stored.asType(.float32)
        } else {
            let dim = headDim / 2
            invFreq = MLXArray(stride(from: 0, to: dim, by: 2).map { 1 / pow(Float(10000), Float($0) / Float(dim)) })
        }
        blocks = try (0 ..< cfg.vDepth).map { try Q21VisionBlock(store: store, index: $0, cfg: cfg) }
        merger = try Q21Merger(store: store, prefix: "visual.merger", cfg: cfg, postShuffle: false)
        deepMergers = try cfg.deepstack.indices.map {
            try Q21Merger(store: store, prefix: "visual.deepstack_merger_list.\($0)", cfg: cfg, postShuffle: true)
        }
    }

    private func positionEmbeddings(grids: [[Int]], dtype: DType) -> MLXArray {
        let side = Int(Double(cfg.vNumPositions).squareRoot())
        var idx = [[Int32]](repeating: [], count: 4)
        var wts = [[Float]](repeating: [], count: 4)
        for g in grids {
            let h = g[1], w = g[2]
            func lin(_ n: Int) -> [Float] {
                let delta = n > 1 ? Float(Double(side - 1) / Double(n - 1)) : 0
                return (0 ..< n).map { Float($0) * delta }
            }
            let hs = lin(h), ws = lin(w)
            for hv in hs {
                let hf = Int32(hv), hc = min(hf + 1, Int32(side - 1)), dh = hv - Float(hf)
                for wv in ws {
                    let wf = Int32(wv), wc = min(wf + 1, Int32(side - 1)), dw = wv - Float(wf)
                    idx[0].append(hf * Int32(side) + wf); wts[0].append((1 - dh) * (1 - dw))
                    idx[1].append(hf * Int32(side) + wc); wts[1].append((1 - dh) * dw)
                    idx[2].append(hc * Int32(side) + wf); wts[2].append(dh * (1 - dw))
                    idx[3].append(hc * Int32(side) + wc); wts[3].append(dh * dw)
                }
            }
        }
        var total: MLXArray?
        for i in 0 ..< 4 {
            let e = posEmbed(MLXArray(idx[i])) * MLXArray(wts[i]).reshaped([-1, 1])
            total = total.map { $0 + e } ?? e
        }
        var pieces: [MLXArray] = []
        var start = 0
        let m = cfg.vMerge
        for g in grids {
            let t = g[0], h = g[1], w = g[2]
            var p = total![start ..< (start + h * w)]
            start += h * w
            if t > 1 { p = tiled(p, repetitions: [t, 1]) }
            p = p.reshaped([t, h / m, m, w / m, m, -1]).transposed(0, 1, 3, 2, 4, 5).reshaped([-1, cfg.vHidden])
            pieces.append(p)
        }
        return concatenated(pieces, axis: 0).asType(dtype)
    }

    private func rotary(grids: [[Int]]) -> (MLXArray, MLXArray) {
        let m = cfg.vMerge
        var hIdx: [Int32] = [], wIdx: [Int32] = []
        for g in grids {
            let t = g[0], h = g[1], w = g[2]
            var hh: [Int32] = [], ww: [Int32] = []
            for bh in 0 ..< (h / m) {
                for bw in 0 ..< (w / m) {
                    for ih in 0 ..< m {
                        for iw in 0 ..< m {
                            hh.append(Int32(bh * m + ih)); ww.append(Int32(bw * m + iw))
                        }
                    }
                }
            }
            for _ in 0 ..< t { hIdx += hh; wIdx += ww }
        }
        let maxSize = grids.map { max($0[1], $0[2]) }.max() ?? 1
        let full = q21Range(0, maxSize).asType(.float32).reshaped([-1, 1]) * invFreq.reshaped([1, -1])
        let rot = concatenated([take(full, MLXArray(hIdx), axis: 0), take(full, MLXArray(wIdx), axis: 0)], axis: -1)
        let emb = concatenated([rot, rot], axis: -1)
        return (MLX.cos(emb), MLX.sin(emb))
    }

    /// pixels (N, C*tp*ph*pw) in (C, tp, ph, pw) order → (merged tokens, deepstack list)
    func callAsFunction(_ pixels: MLXArray, grids: [[Int]]) -> (MLXArray, [MLXArray]) {
        let n = pixels.dim(0)
        let x5 = pixels.reshaped([n, 3, cfg.vTemporal, cfg.vPatch, cfg.vPatch]).transposed(0, 2, 3, 4, 1).reshaped([n, -1])
        var h = matmul(x5, patchW.transposed()) + patchB
        if dumpEnabled { debug["v_patch"] = h }
        let pos = positionEmbeddings(grids: grids, dtype: h.dtype)
        if dumpEnabled { debug["v_pos"] = pos }
        h = h + pos
        let (cos, sin) = rotary(grids: grids)
        if dumpEnabled { debug["v_cos"] = cos }
        let lengths = grids.flatMap { g in [Int](repeating: g[1] * g[2], count: g[0]) }
        var stack: [MLXArray] = []
        for (index, block) in blocks.enumerated() {
            h = block(h, cos: cos, sin: sin, lengths: lengths)
            if dumpEnabled, index == 0 { debug["v_block0"] = h }
            if let k = cfg.deepstack.firstIndex(of: index) {
                stack.append(deepMergers[k](h))
            }
            eval(h)
        }
        return (merger(h), stack)
    }
}

// MARK: - VAE (64-channel residual, single-frame, NHWC, fp32)

private final class Q21Conv {
    let weight: MLXArray, bias: MLXArray?
    let stride: Int, padding: Int
    init(store: MFluxStore, prefix: String, stride: Int = 1, padding: Int? = nil) throws {
        weight = try store.tensor("vae", "\(prefix).weight").asType(.float32)
        bias = store.optionalTensor("vae", "\(prefix).bias")?.asType(.float32)
        self.stride = stride
        self.padding = padding ?? (weight.dim(1) / 2)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv2d(x, weight, stride: IntOrPair(stride), padding: IntOrPair(padding))
        if let bias { y = y + bias }
        return y
    }
}

private func q21ChannelNorm(_ x: MLXArray, gamma: MLXArray) -> MLXArray {
    let v = x.asType(.float32)
    let norm = maximum(sqrt(sum(v * v, axis: -1, keepDims: true)), MLXArray(Float(1e-12)))
    return (v / norm).asType(x.dtype) * Float(x.dim(-1)).squareRoot() * gamma
}

private final class Q21Res {
    let g1, g2: MLXArray
    let c1, c2: Q21Conv
    let shortcut: Q21Conv?
    init(store: MFluxStore, prefix p: String) throws {
        g1 = try store.tensor("vae", "\(p).norm1.gamma").asType(.float32).reshaped([-1])
        g2 = try store.tensor("vae", "\(p).norm2.gamma").asType(.float32).reshaped([-1])
        c1 = try Q21Conv(store: store, prefix: "\(p).conv1")
        c2 = try Q21Conv(store: store, prefix: "\(p).conv2")
        shortcut = store.hasKey("vae", "\(p).conv_shortcut.weight")
            ? try Q21Conv(store: store, prefix: "\(p).conv_shortcut", padding: 0) : nil
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let s = shortcut.map { $0(x) } ?? x
        let h = c2(silu(q21ChannelNorm(c1(silu(q21ChannelNorm(x, gamma: g1))), gamma: g2)))
        return s + h
    }
}

private final class Q21VAEAttn {
    let gamma: MLXArray
    let qkv, proj: Q21Conv
    init(store: MFluxStore, prefix p: String) throws {
        gamma = try store.tensor("vae", "\(p).norm.gamma").asType(.float32).reshaped([-1])
        qkv = try Q21Conv(store: store, prefix: "\(p).to_qkv", padding: 0)
        proj = try Q21Conv(store: store, prefix: "\(p).proj", padding: 0)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3)
        let packed = qkv(q21ChannelNorm(x, gamma: gamma)).reshaped([b, 1, h * w, 3 * c])
        let parts = split(packed, parts: 3, axis: -1)
        let o = MLXFast.scaledDotProductAttention(
            queries: parts[0], keys: parts[1], values: parts[2],
            scale: 1 / Float(c).squareRoot(), mask: .none)
        return x + proj(o.reshaped([b, h, w, c]))
    }
}

private final class Q21Mid {
    let r0, r1: Q21Res
    let attn: Q21VAEAttn
    init(store: MFluxStore, prefix p: String) throws {
        r0 = try Q21Res(store: store, prefix: "\(p).resnets.0")
        r1 = try Q21Res(store: store, prefix: "\(p).resnets.1")
        attn = try Q21VAEAttn(store: store, prefix: "\(p).attentions.0")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { r1(attn(r0(x))) }
}

/// mflux Qwen21AvgDown on NCHW; x is NHWC here.
private func q21AvgDown(_ xNHWC: MLXArray, inC: Int, outC: Int, ft: Int, fs: Int) -> MLXArray {
    let factor = ft * fs * fs
    let group = inC * factor / outC
    if factor == 1 && group == 1 { return xNHWC }
    var x = xNHWC.transposed(0, 3, 1, 2)                      // (b, c, h, w)
    let b = x.dim(0), c = x.dim(1), h = x.dim(2), w = x.dim(3)
    x = x.expandedDimensions(axis: 2)                         // (b, c, 1, h, w)
    if ft == 2 {
        x = concatenated([MLXArray.zeros(like: x), x], axis: 2)
    }
    x = x.reshaped([b, c, 1, ft, h / fs, fs, w / fs, fs])
    x = x.transposed(0, 1, 3, 5, 7, 2, 4, 6)
    x = x.reshaped([b, c * factor, h / fs, w / fs]).reshaped([b, outC, group, h / fs, w / fs])
    return mean(x, axis: 2).transposed(0, 2, 3, 1)
}

/// mflux Qwen21DupUp on NCHW; x is NHWC here.
private func q21DupUp(_ xNHWC: MLXArray, inC: Int, outC: Int, ft: Int, fs: Int = 2) -> MLXArray {
    let factor = ft * fs * fs
    let repeats = outC * factor / inC
    var x = xNHWC.transposed(0, 3, 1, 2)
    let b = x.dim(0), h = x.dim(2), w = x.dim(3)
    x = repeated(x, count: repeats, axis: 1)
    x = x.reshaped([b, outC, ft, fs, fs, 1, h, w])
    x = x.transposed(0, 1, 5, 2, 6, 3, 7, 4)
    x = x.reshaped([b, outC, ft, h * fs, w * fs])
    x = x[0..., 0..., (ft - 1)..., 0..., 0...].squeezed(axis: 2)
    return x.transposed(0, 2, 3, 1)
}

private final class Q21Down {
    let resnets: [Q21Res]
    let downsample: Q21Conv?
    let inC, outC, ft, fs: Int
    init(store: MFluxStore, index: Int, inC: Int, outC: Int, numRes: Int, down: Bool, temporal: Bool) throws {
        let p = "encoder.down_blocks.\(index)"
        resnets = try (0 ..< numRes).map { try Q21Res(store: store, prefix: "\(p).resnets.\($0)") }
        downsample = down ? try Q21Conv(store: store, prefix: "\(p).downsampler.resample.1", stride: 2, padding: 0) : nil
        self.inC = inC; self.outC = outC; ft = temporal ? 2 : 1; fs = down ? 2 : 1
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shortcut = q21AvgDown(x, inC: inC, outC: outC, ft: ft, fs: fs)
        var h = x
        for r in resnets { h = r(h) }
        if let downsample {
            h = downsample(padded(h, widths: [IntOrPair(0), IntOrPair((0, 1)), IntOrPair((0, 1)), IntOrPair(0)]))
        }
        return h + shortcut
    }
}

private final class Q21Up {
    let resnets: [Q21Res]
    let upsample: Q21Conv?
    let inC, outC, ft: Int
    init(store: MFluxStore, index: Int, inC: Int, outC: Int, numRes: Int, up: Bool, temporal: Bool) throws {
        let p = "decoder.up_blocks.\(index)"
        resnets = try (0 ..< (numRes + 1)).map { try Q21Res(store: store, prefix: "\(p).resnets.\($0)") }
        upsample = up ? try Q21Conv(store: store, prefix: "\(p).upsampler.resample.1", padding: 1) : nil
        self.inC = inC; self.outC = outC; ft = temporal ? 2 : 1
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for r in resnets { h = r(h) }
        guard let upsample else { return h }
        let shortcut = q21DupUp(x, inC: inC, outC: outC, ft: ft)
        let doubled = repeated(repeated(h, count: 2, axis: 1), count: 2, axis: 2)
        return upsample(doubled) + shortcut
    }
}

final class QwenImage21VAE {
    private let cfg: QwenImage21Config
    private let eConvIn, eConvOut, quantConv: Q21Conv
    private let downs: [Q21Down]
    private let eMid: Q21Mid
    private let eNorm: MLXArray
    private let postQuant, dConvIn, dConvOut: Q21Conv
    private let dMid: Q21Mid
    private let ups: [Q21Up]
    private let dNorm: MLXArray
    private let mean: MLXArray, std: MLXArray

    init(store: MFluxStore, cfg: QwenImage21Config) throws {
        self.cfg = cfg
        let encDims = ([1] + cfg.dimMult).map { cfg.vaeBaseDim * $0 }
        eConvIn = try Q21Conv(store: store, prefix: "encoder.conv_in", padding: 1)
        var ds: [Q21Down] = []
        for i in 0 ..< (encDims.count - 1) {
            ds.append(try Q21Down(
                store: store, index: i, inC: encDims[i], outC: encDims[i + 1], numRes: cfg.numResBlocks,
                down: i < encDims.count - 2,
                temporal: i < cfg.temporalDownsample.count ? cfg.temporalDownsample[i] : false))
        }
        downs = ds
        eMid = try Q21Mid(store: store, prefix: "encoder.mid_block")
        eNorm = try store.tensor("vae", "encoder.norm_out.gamma").asType(.float32).reshaped([-1])
        eConvOut = try Q21Conv(store: store, prefix: "encoder.conv_out", padding: 1)
        quantConv = try Q21Conv(store: store, prefix: "quant_conv", padding: 0)

        let decDims = ([cfg.dimMult.last!] + cfg.dimMult.reversed()).map { cfg.vaeDecoderBaseDim * $0 }
        let temporal = Array(cfg.temporalDownsample.reversed())
        postQuant = try Q21Conv(store: store, prefix: "post_quant_conv", padding: 0)
        dConvIn = try Q21Conv(store: store, prefix: "decoder.conv_in", padding: 1)
        dMid = try Q21Mid(store: store, prefix: "decoder.mid_block")
        var us: [Q21Up] = []
        for i in 0 ..< (decDims.count - 1) {
            us.append(try Q21Up(
                store: store, index: i, inC: decDims[i], outC: decDims[i + 1], numRes: cfg.numResBlocks,
                up: i < decDims.count - 2,
                temporal: i < temporal.count ? temporal[i] : false))
        }
        ups = us
        dNorm = try store.tensor("vae", "decoder.norm_out.gamma").asType(.float32).reshaped([-1])
        dConvOut = try Q21Conv(store: store, prefix: "decoder.conv_out", padding: 1)
        mean = MLXArray(cfg.latentsMean)
        std = MLXArray(cfg.latentsStd)
    }

    /// image (1, H, W, 4) in [-1, 1] fp32 → packed latents (1, H/16 * W/16, 64) normalized.
    func encode(_ image: MLXArray) -> MLXArray {
        var h = eConvIn(image)
        for d in downs { h = d(h); eval(h) }
        h = eConvOut(silu(q21ChannelNorm(eMid(h), gamma: eNorm)))
        let moments = quantConv(h)
        let z = moments[.ellipsis, 0 ..< cfg.zDim]
        let normalized = (z - mean) / std
        return normalized.reshaped([1, -1, cfg.zDim])
    }

    /// latents (1, h, w, 64) normalized → image (1, H, W, 4) in [-1, 1].
    func decode(_ latents: MLXArray) -> MLXArray {
        var h = postQuant(latents * std + mean)
        h = dMid(dConvIn(h))
        for u in ups { h = u(h); eval(h) }
        h = dConvOut(silu(q21ChannelNorm(h, gamma: dNorm)))
        return clip(h, min: Float(-1), max: Float(1))
    }
}

// MARK: - Scheduler

enum QwenImage21Schedule {
    /// The bundle's fixed sigma grid, `model_index.json` `sample_sigmas` (Qwen-Image-2.1-Turbo: 8 distilled
    /// nodes). Diffusers (PR #14950) passes it to FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas:);
    /// with the Turbo scheduler config (shift 1.0, no dynamic shifting, no terminal shift) the nodes are
    /// used unchanged and a terminal 0 is appended. Nil when the bundle defines no grid (Qwen-Image-2.1).
    static func sampleSigmas(modelPath: URL) throws -> [Float]? {
        let url = modelPath.appendingPathComponent("model_index.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw FluxError.invalidRequest("Qwen-Image-2.1 model_index.json must be an object.")
        }
        guard let raw = json["sample_sigmas"] else { return nil }
        guard let values = raw as? [NSNumber], values.count >= 2,
            values.allSatisfy({ CFGetTypeID($0) != CFBooleanGetTypeID() })
        else {
            throw FluxError.invalidRequest("sample_sigmas must contain at least two numeric sigma nodes.")
        }
        let grid = values.map { $0.floatValue }
        guard grid.allSatisfy({ $0.isFinite && $0 > 0 && $0 <= 1 }),
            zip(grid, grid.dropFirst()).allSatisfy({ $0.0 > $0.1 })
        else {
            throw FluxError.invalidRequest("sample_sigmas must be finite, strictly decreasing nodes in (0, 1].")
        }
        return grid
    }

    /// Fixed grid + terminal 0, unshifted.
    static func sigmas(grid: [Float]) -> MLXArray {
        MLXArray(grid + [0])
    }

    /// mflux LinearScheduler with resolution shift (base 0.5@256 → max 0.9@8192) and terminal 0.02.
    static func sigmas(steps: Int, width: Int, height: Int) -> MLXArray {
        let n = steps
        let delta = Float((1.0 / Double(n) - 1.0) / Double(max(n - 1, 1)))
        let raw = q21Range(0, n).asType(.float32) * delta + Float(1)
        let m = (0.9 - 0.5) / Double(8192 - 256)
        let b = 0.5 - m * 256
        let mu = MLXArray(Float(m * Double(width * height) / 256 + b))
        var shifted = exp(mu) / (exp(mu) + (1 / raw - 1))
        let oneMinus = 1 - shifted
        let scale = oneMinus[(n - 1) ..< n] / Float(1 - 0.02)
        shifted = 1 - oneMinus / scale
        return concatenated([shifted, MLXArray.zeros([1], dtype: .float32)])
    }
}

// MARK: - Tokenizer / processor

final class QwenImage21Processor {
    static let system = "Comprehend and analyze the provided prompt."
    private let tokenizer: any VMLXTokenizers.Tokenizer
    let cfg: QwenImage21Config
    let dropCount: Int

    init(modelPath: URL, cfg: QwenImage21Config) async throws {
        self.cfg = cfg
        let processorDir = modelPath.appendingPathComponent("processor")
        let folder = FileManager.default.fileExists(atPath: processorDir.appendingPathComponent("tokenizer.json").path)
            ? processorDir : modelPath.appendingPathComponent("tokenizer")
        tokenizer = try await AutoTokenizer.from(modelFolder: folder, strict: false)
        dropCount = tokenizer.encode(
            text: "<|im_start|>system\n\(Self.system)<|im_end|>\n", addSpecialTokens: false).count
    }

    /// Prompt token ids with each image placeholder expanded to its merged-token count.
    func promptIDs(_ prompt: String, imageTokenCounts: [Int]) -> [Int32] {
        let prefix = imageTokenCounts.indices.map { i in
            "<image\(i + 1)><|vision_start|>" + String(repeating: "<|image_pad|>", count: imageTokenCounts[i]) + "<|vision_end|>"
        }.joined(separator: " ")
        let body = prompt.isEmpty ? " " : prompt
        let text = "<|im_start|>system\n\(Self.system)<|im_end|>\n<|im_start|>user\n\(prefix)\(body)<|im_end|>\n<|im_start|>assistant\n"
        return tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
    }

    /// RGB [0,1] planar (3, H, W) → Qwen2-VL patches (gridT*gh*gw, C*tp*ps*ps) + grid.
    func visionPatches(rgb: MLXArray, width: Int, height: Int) -> (MLXArray, [Int]) {
        let ps = cfg.vPatch, tp = cfg.vTemporal, m = cfg.vMerge
        let meanA = MLXArray(cfg.imageMean).reshaped([3, 1, 1])
        let stdA = MLXArray(cfg.imageStd).reshaped([3, 1, 1])
        let normalized = (rgb - meanA) / stdA                          // (3, H, W)
        let frames = stacked([MLXArray](repeating: normalized, count: tp), axis: 0)  // (tp, 3, H, W)
        let gh = height / ps, gw = width / ps
        var p = frames.reshaped([1, tp, 3, gh / m, m, ps, gw / m, m, ps])
        p = p.transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return (p.reshaped([gh * gw, 3 * tp * ps * ps]), [1, gh, gw])
    }
}

// MARK: - Pipeline

final class QwenImage21Pipeline: @unchecked Sendable {
    let cfg: QwenImage21Config
    private let transformer: QwenImage21Transformer
    private let textEncoder: QwenImage21TextEncoder
    private let vae: QwenImage21VAE
    private let processor: QwenImage21Processor
    private var debugDump: [String: MLXArray] = [:]

    /// Fixed sampling grid from the bundle (Turbo); nil = shifted linear schedule over `steps`.
    let sampleSigmas: [Float]?

    init(modelPath: URL, loaded: LoadedWeights, sampleSigmas: [Float]?) async throws {
        cfg = try QwenImage21Config.load(modelPath)
        self.sampleSigmas = sampleSigmas
        let store = MFluxStore(loaded)
        transformer = try QwenImage21Transformer(store: store, cfg: cfg)
        textEncoder = try QwenImage21TextEncoder(store: store, cfg: cfg)
        vae = try QwenImage21VAE(store: store, cfg: cfg)
        processor = try await QwenImage21Processor(modelPath: modelPath, cfg: cfg)
    }

    static func dimensions(resolution: Int, ratio: Double) -> (Int, Int) {
        let width = (Double(resolution * resolution) * ratio).squareRoot()
        let w = max(32, Int((width / 32).rounded(.toNearestOrEven)) * 32)
        let h = max(32, Int((width / ratio / 32).rounded(.toNearestOrEven)) * 32)
        return (w, h)
    }

    private func encodePrompt(_ prompt: String, references: [(rgb: MLXArray, width: Int, height: Int)])
        -> (MLXArray, [Bool])
    {
        var pixels: [MLXArray] = []
        var grids: [[Int]] = []
        for ref in references {
            let (p, g) = processor.visionPatches(rgb: ref.rgb, width: ref.width, height: ref.height)
            pixels.append(p); grids.append(g)
        }
        if ProcessInfo.processInfo.environment["VMLX_QWEN21_DUMP"] != nil, !pixels.isEmpty {
            let px = concatenated(pixels, axis: 0)
            let (features, stack) = textEncoder.vision(px.asType(.bfloat16), grids: grids)
            debugDump["pixel_values"] = px
            debugDump["vision_features"] = features
            if let first = stack.first { debugDump["deepstack0"] = first }
            for (k, v) in textEncoder.vision.debug { debugDump[k] = v }
        }
        let counts = grids.map { $0[0] * $0[1] * $0[2] / (cfg.vMerge * cfg.vMerge) }
        let ids = processor.promptIDs(prompt, imageTokenCounts: counts)
        let hidden = textEncoder.encode(
            ids: ids, pixelValues: pixels.isEmpty ? nil : concatenated(pixels, axis: 0), grids: grids)
        let drop = processor.dropCount
        let embeds = hidden[0..., drop..., 0...]
        let slots = ids[drop...].map { $0 == Int32(cfg.imageTokenID) }
        return (embeds, slots)
    }

    func generate(
        prompt: String, negativePrompt: String?, references: [URL],
        width requestedWidth: Int?, height requestedHeight: Int?,
        steps requestedSteps: Int, guidance: Float, seed: UInt64, outputResolution: Int = 1024,
        progress: (Int, Int, Double?) -> Void
    ) throws -> MLXArray {
        // A bundle with a fixed grid (Turbo) samples on it: the grid sets the step count, as in diffusers.
        let steps = sampleSigmas?.count ?? requestedSteps
        guard references.count <= 10 else {
            throw FluxError.invalidRequest("Qwen-Image-2.1 supports at most 10 reference images.")
        }
        guard steps >= 2 else {
            throw FluxError.invalidRequest("Qwen-Image-2.1 requires at least two steps.")
        }
        // references: resize to the area budget at their own aspect ratio (multiples of 32)
        var refs: [(rgb: MLXArray, width: Int, height: Int)] = []
        var lastRatio = 1.0
        for url in references {
            let size = try ImageIO.dimensions(of: url)
            lastRatio = Double(size.width) / Double(size.height)
            let (w, h) = Self.dimensions(resolution: outputResolution, ratio: lastRatio)
            guard cfg.minPixels <= w * h, w * h <= cfg.maxPixels else {
                throw FluxError.invalidRequest("Reference image area falls outside the vision processor's range.")
            }
            let values = try ImageIO.readRGBValues(url, width: w, height: h, normalization: .zeroToOne)
            refs.append((MLXArray(values, [3, h, w]), w, h))
        }
        let (defaultW, defaultH) = Self.dimensions(resolution: outputResolution, ratio: references.isEmpty ? 1 : lastRatio)
        let width = requestedWidth ?? defaultW
        let height = requestedHeight ?? defaultH
        guard width >= 32, height >= 32, width % 32 == 0, height % 32 == 0 else {
            throw FluxError.invalidRequest("Qwen-Image-2.1 dimensions must be positive multiples of 32.")
        }

        let (embeds, slots) = encodePrompt(prompt, references: refs)
        var negative: (MLXArray, [Bool])?
        if guidance > 1 {
            negative = encodePrompt(negativePrompt ?? "", references: refs)
        }
        eval(embeds)
        let dtype = embeds.dtype
        let shapes = refs.map { (1, $0.height / 16, $0.width / 16) } + [(1, height / 16, width / 16)]
        let layout = try QwenImage21Layout.create(slots: slots, shapes: shapes, axes: cfg.axes)
        let negativeLayout = try negative.map { try QwenImage21Layout.create(slots: $0.1, shapes: shapes, axes: cfg.axes) }

        var condition: MLXArray?
        if !refs.isEmpty {
            let latents = refs.map { ref -> MLXArray in
                let rgb = ref.rgb * 2 - 1                                        // (3, H, W)
                let alpha = MLXArray.ones([1, ref.height, ref.width], dtype: .float32)
                let rgba = concatenated([rgb, alpha], axis: 0).transposed(1, 2, 0).expandedDimensions(axis: 0)
                return vae.encode(rgba).asType(dtype)
            }
            condition = concatenated(latents, axis: 1)
        }

        // Same values as mflux's `mx.random.seed(seed); mx.random.normal(...)` (the global
        // key sequence hands out split(key(seed))[1] first) WITHOUT touching MLX's global
        // PRNG, which an LLM in the same process may be sampling from concurrently.
        let noiseKey = MLXRandom.split(key: MLXRandom.key(seed)).1
        let hLat = height / 16, wLat = width / 16
        var latents = MLXRandom.normal([1, cfg.zDim, 1, hLat, wLat], key: noiseKey).asType(dtype)
        latents = latents[0..., 0..., 0, 0..., 0...].transposed(0, 2, 3, 1).reshaped([1, hLat * wLat, cfg.zDim])
        let sigmas = sampleSigmas.map { QwenImage21Schedule.sigmas(grid: $0) }
            ?? QwenImage21Schedule.sigmas(steps: steps, width: width, height: height)
        eval(sigmas)
        let dumpURL = ProcessInfo.processInfo.environment["VMLX_QWEN21_DUMP"].map { URL(fileURLWithPath: $0) }
        var dump: [String: MLXArray] = debugDump
        if dumpURL != nil {
            dump["prompt_embeds"] = embeds
            dump["slots"] = MLXArray(slots.map { Int32($0 ? 1 : 0) })
            dump["latents_init"] = latents
            dump["sigmas"] = sigmas
            if let condition { dump["condition"] = condition }
        }

        var cache: [(MLXArray, MLXArray)]? = []
        var negativeCache: [(MLXArray, MLXArray)]? = []
        let start = Date()
        for step in 0 ..< steps {
            let input = condition.map { concatenated([$0, latents], axis: 1) } ?? latents
            let timestep = (sigmas[step ..< (step + 1)] * 1000).asType(dtype) / 1000
            var noise = transformer(hidden: input, encoder: embeds, timestep: timestep, layout: layout, cache: &cache)
            if let negative, let negativeLayout {
                let uncond = transformer(
                    hidden: input, encoder: negative.0, timestep: timestep, layout: negativeLayout, cache: &negativeCache)
                noise = uncond + guidance * (noise - uncond)
            }
            let dt = sigmas[step + 1] - sigmas[step]
            latents = (latents.asType(.float32) + dt * noise.asType(.float32)).asType(dtype)
            eval(latents)
            if dumpURL != nil, step == 0 { dump["noise_step0"] = noise }
            let elapsed = Date().timeIntervalSince(start)
            progress(step + 1, steps, elapsed / Double(step + 1) * Double(steps - step - 1))
        }
        cache = nil
        negativeCache = nil
        let unpacked = latents.reshaped([1, hLat, wLat, cfg.zDim]).asType(.float32)
        let decoded = vae.decode(unpacked)                                    // (1, H, W, 4)
        let rgb = decoded[.ellipsis, 0 ..< 3].transposed(0, 3, 1, 2)          // (1, 3, H, W)
        if let dumpURL {
            dump["latents_final"] = latents
            dump["decoded"] = decoded
            try save(arrays: dump.mapValues { $0.asType(.float32) }, url: dumpURL)
        }
        return VAEDecoder.postprocess(rgb)
    }
}
