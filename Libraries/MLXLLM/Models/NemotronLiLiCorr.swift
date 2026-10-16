// Copyright © 2026 Apple Inc.
//
// NVIDIA LiLiCorr drafter for Nemotron-3.5-Bolt (`lilicorr/` in the bundle), mlx-swift port.
//
// Semantics follow SGLang's merged LiLiCorr (sgl-project/sglang#37462: srt/models/lilicorr.py,
// srt/models/dflash.py, srt/speculative/lilicorr_utils.py) through the vMLX Python port
// `vmlx_engine/models/nemotron_lilicorr.py`, which is the numerics reference for this file.
//
// Drafter body (DFlashDraftModel):
//   * context feature = hidden_norm(fc(concat(target taps))), taps = OUTPUTS of the target blocks
//     dflash_config.target_layer_ids (absolute indices of the backbone stack, before `norm_f`);
//     context K/V per draft layer = rope(k_norm(k_proj(ctx))) and v_proj(ctx), no conv, no layer
//     norm (the projected feature feeds every layer);
//   * block = [anchor token, mask x (block_size - 1)] embedded with the TARGET embedding table;
//   * pre-norm layers: input_layernorm -> grouped dynamic conv (prepare) -> attention (per-head
//     q/k RMSNorm, neox RoPE, per-head attention_sink_bias, NON-causal inside a left window of
//     sliding_window) -> conv (finish) -> residual; post_attention_layernorm -> conv prepare ->
//     SwiGLU MLP -> conv finish -> residual; final norm;
//   * grouped dynamic conv: K taps over the block's own positions, coefficients = base_kernel[side]
//     + a per-token delta from kernel_projection(input) (groups of conv_group_size channels), tap t
//     reads position p - t only when p mod block_size >= t.
// LiLiCorr head (LiLiCorrHead.score + lilicorr_sample_path):
//   * candidates per draft slot = top-k of log_softmax(lm_head(draft hidden)) over the FULL vocab;
//   * candidate state = token_proj(target_embed(candidate)) + pass_hidden_proj(draft hidden) +
//     feature_mlp(5 features: log p, p, log p - max, rank fraction, is-top1) + slot / rank
//     embeddings; lattice layers (relative-slot + same-slot attention bias); anchor =
//     context_proj(context feature at the last committed position);
//   * factor edges: start = <anchor_out, in_vec(slot 0)>, pair = <out_vec(slot s), in_vec(slot s+1)>,
//     x logit_scale; the path is walked left to right (greedy: argmax).
// NVFP4 modules: y = (W_nvfp4 x) * weight_scale_2 (modelopt's per-tensor global scale) [+ bias].
//
// Numerics mirror the Python port exactly: bf16 for every projection / norm / attention input, fp32
// for log-probabilities, the five candidate features (cast to bf16 only for feature_norm), the
// L2-normalised factor vectors and the start / pair scores.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Errors

public enum NemotronLiLiCorrError: LocalizedError {
    case missingFile(String)
    case malformedConfig(String)
    case missingTensor(String)
    case unsupportedQuantization(module: String, mode: String)
    case malformedTensor(String)
    case contextPositionMismatch(expected: Int, got: Int)
    case noAnchorContext

    public var errorDescription: String? {
        switch self {
        case .missingFile(let p): return "LiLiCorr: missing file \(p)"
        case .malformedConfig(let m): return "LiLiCorr: malformed config.json: \(m)"
        case .missingTensor(let n): return "LiLiCorr: checkpoint is missing tensor \(n)"
        case .unsupportedQuantization(let module, let mode):
            return "LiLiCorr: module \(module) declares quantization mode '\(mode)'; only nvfp4 is supported"
        case .malformedTensor(let m): return "LiLiCorr: malformed tensor: \(m)"
        case .contextPositionMismatch(let expected, let got):
            return "LiLiCorr: context position \(got) does not continue the drafter context (ends at \(expected))"
        case .noAnchorContext:
            return "LiLiCorr: the lattice head needs at least one context position (anchor) before proposing"
        }
    }
}

// MARK: - Configuration

/// `lilicorr/config.json`. Only the fields the drafter consumes are modelled; everything is read
/// from the live file, nothing is inferred from the family name.
public struct NemotronLiLiCorrConfiguration {
    public struct QuantizationSpec {
        public var mode: String
        public var groupSize: Int
        public var bits: Int
        /// Name of the per-tensor global scale (`<module>.weight_scale_2`).
        public var globalScale: String?
    }

    public var blockSize: Int
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var intermediateSize: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var vocabSize: Int
    /// Left attention window (`dflash_config.swa_window_size`, else `sliding_window`).
    public var window: Int
    public var maskTokenId: Int
    public var targetLayerIds: [Int]
    public var convKernelSize: Int
    public var convGroupSize: Int
    public var lilicorrEnabled: Bool
    public var lilicorrCandidateTopK: Int
    public var lilicorrHiddenSize: Int
    public var lilicorrNumLayers: Int
    public var lilicorrNumHeads: Int
    public var lilicorrVectorEps: Float
    public var lilicorrLogitScale: Float
    /// Per-module quantization (`quantization` block, module name -> spec). Modules absent from
    /// the map are plain bf16 linears unless the checkpoint carries `<module>.scales`, in which
    /// case `defaultQuantization` applies (the Python port keys on the tensors alone).
    public var quantization: [String: QuantizationSpec]
    public var defaultQuantization: QuantizationSpec?

    public init(
        blockSize: Int, hiddenSize: Int, numHiddenLayers: Int, numAttentionHeads: Int,
        numKeyValueHeads: Int, headDim: Int, intermediateSize: Int, rmsNormEps: Float,
        ropeTheta: Float, vocabSize: Int, window: Int, maskTokenId: Int, targetLayerIds: [Int],
        convKernelSize: Int, convGroupSize: Int, lilicorrEnabled: Bool,
        lilicorrCandidateTopK: Int, lilicorrHiddenSize: Int, lilicorrNumLayers: Int,
        lilicorrNumHeads: Int, lilicorrVectorEps: Float, lilicorrLogitScale: Float,
        quantization: [String: QuantizationSpec], defaultQuantization: QuantizationSpec?
    ) {
        self.blockSize = blockSize
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.intermediateSize = intermediateSize
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.vocabSize = vocabSize
        self.window = window
        self.maskTokenId = maskTokenId
        self.targetLayerIds = targetLayerIds
        self.convKernelSize = convKernelSize
        self.convGroupSize = convGroupSize
        self.lilicorrEnabled = lilicorrEnabled
        self.lilicorrCandidateTopK = lilicorrCandidateTopK
        self.lilicorrHiddenSize = lilicorrHiddenSize
        self.lilicorrNumLayers = lilicorrNumLayers
        self.lilicorrNumHeads = lilicorrNumHeads
        self.lilicorrVectorEps = lilicorrVectorEps
        self.lilicorrLogitScale = lilicorrLogitScale
        self.quantization = quantization
        self.defaultQuantization = defaultQuantization
    }

    public init(json root: [String: Any]) throws {
        func int(_ d: [String: Any], _ k: String) throws -> Int {
            guard let v = d[k] else { throw NemotronLiLiCorrError.malformedConfig("missing \(k)") }
            if let i = v as? Int { return i }
            if let n = v as? NSNumber { return n.intValue }
            throw NemotronLiLiCorrError.malformedConfig("\(k) is not an integer")
        }
        func float(_ d: [String: Any], _ k: String, default def: Float? = nil) throws -> Float {
            guard let v = d[k], !(v is NSNull) else {
                if let def { return def }
                throw NemotronLiLiCorrError.malformedConfig("missing \(k)")
            }
            if let n = v as? NSNumber { return n.floatValue }
            throw NemotronLiLiCorrError.malformedConfig("\(k) is not a number")
        }
        func optInt(_ d: [String: Any], _ k: String) -> Int? {
            guard let v = d[k], !(v is NSNull) else { return nil }
            return (v as? NSNumber)?.intValue
        }
        guard let dc = root["dflash_config"] as? [String: Any] else {
            throw NemotronLiLiCorrError.malformedConfig("missing dflash_config")
        }
        blockSize = optInt(root, "block_size") ?? 8
        hiddenSize = try int(root, "hidden_size")
        numHiddenLayers = try int(root, "num_hidden_layers")
        numAttentionHeads = try int(root, "num_attention_heads")
        numKeyValueHeads = try int(root, "num_key_value_heads")
        headDim = optInt(root, "head_dim") ?? (hiddenSize / numAttentionHeads)
        intermediateSize = try int(root, "intermediate_size")
        rmsNormEps = try float(root, "rms_norm_eps", default: 1e-6)
        ropeTheta = try float(root, "rope_theta", default: 10000)
        vocabSize = try int(root, "vocab_size")
        // Python: int(dc.get("swa_window_size") or c.get("sliding_window") or 1 << 30)
        window = optInt(dc, "swa_window_size").flatMap { $0 > 0 ? $0 : nil }
            ?? optInt(root, "sliding_window").flatMap { $0 > 0 ? $0 : nil } ?? (1 << 30)
        maskTokenId = try int(dc, "mask_token_id")
        guard let ids = dc["target_layer_ids"] as? [Any] else {
            throw NemotronLiLiCorrError.malformedConfig("missing dflash_config.target_layer_ids")
        }
        targetLayerIds = try ids.map {
            guard let n = $0 as? NSNumber else {
                throw NemotronLiLiCorrError.malformedConfig("target_layer_ids must be integers")
            }
            return n.intValue
        }
        convKernelSize = optInt(dc, "conv_kernel_size") ?? 0
        convGroupSize = optInt(dc, "conv_group_size") ?? 0
        lilicorrEnabled = (dc["lilicorr_enabled"] as? Bool) ?? false
        if lilicorrEnabled {
            lilicorrCandidateTopK = try int(dc, "lilicorr_candidate_topk")
            lilicorrHiddenSize = try int(dc, "lilicorr_hidden_size")
            lilicorrNumLayers = try int(dc, "lilicorr_num_layers")
            lilicorrNumHeads = try int(dc, "lilicorr_num_heads")
            lilicorrVectorEps = try float(dc, "lilicorr_vector_eps")
            lilicorrLogitScale = try float(dc, "lilicorr_logit_scale")
        } else {
            lilicorrCandidateTopK = 0
            lilicorrHiddenSize = 0
            lilicorrNumLayers = 0
            lilicorrNumHeads = 0
            lilicorrVectorEps = 0
            lilicorrLogitScale = 0
        }
        var quant: [String: QuantizationSpec] = [:]
        var defaultSpec: QuantizationSpec? = nil
        if let q = root["quantization"] as? [String: Any] {
            func spec(_ d: [String: Any]) -> QuantizationSpec {
                QuantizationSpec(
                    mode: (d["mode"] as? String) ?? "affine",
                    groupSize: optInt(d, "group_size") ?? 64,
                    bits: optInt(d, "bits") ?? 4,
                    globalScale: d["global_scale"] as? String)
            }
            var top: [String: Any] = [:]
            for (k, v) in q {
                if let d = v as? [String: Any] {
                    quant[k] = spec(d)
                } else {
                    top[k] = v
                }
            }
            if !top.isEmpty { defaultSpec = spec(top) }
        }
        quantization = quant
        defaultQuantization = defaultSpec
    }

    public static func load(url: URL) throws -> NemotronLiLiCorrConfiguration {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw NemotronLiLiCorrError.missingFile(url.path)
        }
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NemotronLiLiCorrError.malformedConfig("top level is not an object")
        }
        return try NemotronLiLiCorrConfiguration(json: root)
    }
}

// MARK: - Weight access

struct LiLiCorrWeights {
    let tensors: [String: MLXArray]
    let configuration: NemotronLiLiCorrConfiguration

    func take(_ name: String) throws -> MLXArray {
        guard let t = tensors[name] else { throw NemotronLiLiCorrError.missingTensor(name) }
        return t
    }

    func optional(_ name: String) -> MLXArray? { tensors[name] }

    /// Python `_linear`: NVFP4 when the module is quantized, else a plain bf16 `Linear`.
    func linear(_ name: String) throws -> Module {
        let spec = configuration.quantization[name]
            ?? (tensors["\(name).scales"] != nil ? configuration.defaultQuantization : nil)
        if let spec {
            guard spec.mode == "nvfp4" else {
                throw NemotronLiLiCorrError.unsupportedQuantization(module: name, mode: spec.mode)
            }
            let scaleName = spec.globalScale ?? "\(name).weight_scale_2"
            return NVFP4Linear(
                weight: try take("\(name).weight"), scales: try take("\(name).scales"),
                weightScale2: try take(scaleName), bias: optional("\(name).bias"),
                groupSize: spec.groupSize, bits: spec.bits)
        }
        if tensors["\(name).weight_scale_2"] != nil || tensors["\(name).scales"] != nil {
            throw NemotronLiLiCorrError.malformedTensor(
                "\(name) carries nvfp4 tensors but no quantization entry and no default quantization")
        }
        return Linear(weight: try take("\(name).weight"), bias: optional("\(name).bias"))
    }

    func rmsNorm(_ name: String) throws -> LiLiCorrRMSNorm {
        LiLiCorrRMSNorm(weight: try take("\(name).weight"), eps: configuration.rmsNormEps)
    }
}

@inline(__always)
func liliApply(_ module: Module, _ x: MLXArray) -> MLXArray {
    (module as! UnaryLayer)(x)
}

// MARK: - Modules

/// modelopt W4A16 NVFP4: `y = quantized_matmul(x, W, scales, mode=nvfp4) * weight_scale_2 [+ bias]`.
/// The per-tensor global scale is applied AFTER the matmul, in the output dtype, exactly as the
/// Python port does (it is not folded into the kernel's `globalScale` argument).
public final class NVFP4Linear: Module, UnaryLayer {
    public let weight: MLXArray
    public let scales: MLXArray
    /// fp32 scalar (shape `[1]`).
    public let weightScale2: MLXArray
    public let bias: MLXArray?
    public let groupSize: Int
    public let bits: Int

    public init(
        weight: MLXArray, scales: MLXArray, weightScale2: MLXArray, bias: MLXArray?,
        groupSize: Int = 16, bits: Int = 4
    ) {
        self.weight = weight
        self.scales = scales
        self.weightScale2 = weightScale2.asType(.float32)
        self.bias = bias
        self.groupSize = groupSize
        self.bits = bits
        super.init()
    }

    /// Input features (`weight` packs 32 / bits values per uint32).
    public var inputDimensions: Int { weight.dim(1) * (32 / bits) }
    public var outputDimensions: Int { weight.dim(0) }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = quantizedMM(
            x, weight, scales: scales, biases: nil, transpose: true, groupSize: groupSize,
            bits: bits, mode: .nvfp4)
        y = y * weightScale2.asType(y.dtype)
        if let bias { y = y + bias }
        return y
    }
}

public final class LiLiCorrRMSNorm: Module, UnaryLayer {
    public let weight: MLXArray
    public let eps: Float

    public init(weight: MLXArray, eps: Float) {
        self.weight = weight
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

/// Grouped dynamic convolution over the block's own positions.
public final class LiLiCorrGroupedConv: Module {
    /// `(2, taps, D)`: side 0 = prepare, side 1 = finish.
    public let baseKernel: MLXArray
    let kernelProjection: Module
    public let blockSize: Int
    public let taps: Int
    public let groupSize: Int

    init(weights w: LiLiCorrWeights, name: String, blockSize: Int, taps: Int, groupSize: Int) throws {
        self.baseKernel = try w.take("\(name).base_kernel")
        self.kernelProjection = try w.linear("\(name).kernel_projection")
        self.blockSize = blockSize
        self.taps = taps
        self.groupSize = groupSize
        super.init()
        guard baseKernel.ndim == 3, baseKernel.dim(0) == 2, baseKernel.dim(1) == taps else {
            throw NemotronLiLiCorrError.malformedTensor(
                "\(name).base_kernel shape \(baseKernel.shape) != [2, \(taps), D]")
        }
    }

    /// `x (N, D)` flattened block rows, `delta (N, taps, G)` -> `(N, D)`.
    func convolve(_ x: MLXArray, delta: MLXArray, side: Int) -> MLXArray {
        let N = x.dim(0)
        let D = x.dim(1)
        let G = D / groupSize
        let blocks = x.reshaped([N, G, groupSize])
        let coeff =
            baseKernel[side].reshaped([1, taps, G, groupSize]) + expandedDimensions(delta, axis: -1)
        var out = coeff[0..., 0] * blocks
        if taps > 1 {
            let pos = MLXArray(0 ..< Int32(N)) % Int32(blockSize)
            for t in 1 ..< taps {
                let shifted = concatenated(
                    [MLXArray.zeros([t, G, groupSize], dtype: blocks.dtype), blocks[0 ..< (N - t)]],
                    axis: 0)
                let gate = (pos .>= Int32(t)).asType(blocks.dtype).reshaped([-1, 1, 1])
                out = out + coeff[0..., t] * shifted * gate
            }
        }
        return out.reshaped([N, D])
    }

    /// Returns the prepared rows and the finish-side coefficients `(N, taps, G)`.
    public func prepare(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let N = x.dim(0)
        let D = x.dim(1)
        let c = liliApply(kernelProjection, x).reshaped([N, 2, taps, D / groupSize])
        return (convolve(x, delta: c[0..., 0], side: 0), c[0..., 1])
    }

    public func finish(_ x: MLXArray, coefficients: MLXArray) -> MLXArray {
        convolve(x, delta: coefficients, side: 1)
    }
}

public final class LiLiCorrDraftAttention: Module {
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    let qProj: Module
    let kProj: Module
    let vProj: Module
    let oProj: Module
    let qNorm: LiLiCorrRMSNorm
    let kNorm: LiLiCorrRMSNorm
    /// Per-head attention sink bias `(H,)`.
    public let sinks: MLXArray
    public let theta: Float
    public let scale: Float

    init(weights w: LiLiCorrWeights, name: String) throws {
        let c = w.configuration
        numHeads = c.numAttentionHeads
        numKVHeads = c.numKeyValueHeads
        headDim = c.headDim
        qProj = try w.linear("\(name).q_proj")
        kProj = try w.linear("\(name).k_proj")
        vProj = try w.linear("\(name).v_proj")
        oProj = try w.linear("\(name).o_proj")
        qNorm = try w.rmsNorm("\(name).q_norm")
        kNorm = try w.rmsNorm("\(name).k_norm")
        sinks = try w.take("\(name).attention_sink_bias")
        theta = c.ropeTheta
        scale = pow(Float(c.headDim), -0.5)
        super.init()
        guard sinks.ndim == 1, sinks.dim(0) == numHeads else {
            throw NemotronLiLiCorrError.malformedTensor(
                "\(name).attention_sink_bias shape \(sinks.shape) != [\(numHeads)]")
        }
    }

    /// Test-only construction from explicit arrays.
    init(
        numHeads: Int, numKVHeads: Int, headDim: Int, qProj: Module, kProj: Module, vProj: Module,
        oProj: Module, qNorm: LiLiCorrRMSNorm, kNorm: LiLiCorrRMSNorm, sinks: MLXArray, theta: Float
    ) {
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.oProj = oProj
        self.qNorm = qNorm
        self.kNorm = kNorm
        self.sinks = sinks
        self.theta = theta
        self.scale = pow(Float(headDim), -0.5)
        super.init()
    }

    func rope(_ x: MLXArray, position: Int) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: headDim, traditional: false, base: theta, scale: 1.0, offset: position)
    }

    /// Context K/V from the projected context feature `ctx (1, n, D)` at absolute positions
    /// `position ..< position + n`: `(1, KV, n, hd)` each.
    public func contextKV(_ ctx: MLXArray, position: Int) -> (MLXArray, MLXArray) {
        let n = ctx.dim(1)
        let k = kNorm(liliApply(kProj, ctx).reshaped([1, n, numKVHeads, headDim]))
            .transposed(0, 2, 1, 3)
        let v = liliApply(vProj, ctx).reshaped([1, n, numKVHeads, headDim]).transposed(0, 2, 1, 3)
        return (rope(k, position: position), v)
    }

    /// `h (B, S, D)` block rows at absolute positions `position ..< position + S`; `contextK/V`
    /// `(1, KV, n, hd)` or nil; `mask` boolean `(1, 1, S, n + S)` (true = attend).
    public func callAsFunction(
        _ h: MLXArray, contextK: MLXArray?, contextV: MLXArray?, position: Int, mask: MLXArray?
    ) -> MLXArray {
        let B = h.dim(0)
        let S = h.dim(1)
        var q = qNorm(liliApply(qProj, h).reshaped([B, S, numHeads, headDim])).transposed(0, 2, 1, 3)
        var k = kNorm(liliApply(kProj, h).reshaped([B, S, numKVHeads, headDim])).transposed(0, 2, 1, 3)
        var v = liliApply(vProj, h).reshaped([B, S, numKVHeads, headDim]).transposed(0, 2, 1, 3)
        q = rope(q, position: position)
        k = rope(k, position: position)
        if let contextK, let contextV {
            k = concatenated([contextK, k], axis: 2)
            v = concatenated([contextV, v], axis: 2)
        }
        let o = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask, sinks: sinks.asType(q.dtype))
        return liliApply(oProj, o.transposed(0, 2, 1, 3).reshaped([B, S, -1]))
    }
}

public final class LiLiCorrDraftLayer: Module {
    let inputLayerNorm: LiLiCorrRMSNorm
    let postAttentionLayerNorm: LiLiCorrRMSNorm
    public let selfAttn: LiLiCorrDraftAttention
    let gateProj: Module
    let upProj: Module
    let downProj: Module
    public let attentionConv: LiLiCorrGroupedConv?
    public let mlpConv: LiLiCorrGroupedConv?

    init(weights w: LiLiCorrWeights, index i: Int) throws {
        let c = w.configuration
        let p = "layers.\(i)"
        inputLayerNorm = try w.rmsNorm("\(p).input_layernorm")
        postAttentionLayerNorm = try w.rmsNorm("\(p).post_attention_layernorm")
        selfAttn = try LiLiCorrDraftAttention(weights: w, name: "\(p).self_attn")
        gateProj = try w.linear("\(p).mlp.gate_proj")
        upProj = try w.linear("\(p).mlp.up_proj")
        downProj = try w.linear("\(p).mlp.down_proj")
        if c.convKernelSize > 0 {
            guard c.convGroupSize > 0, c.hiddenSize % c.convGroupSize == 0 else {
                throw NemotronLiLiCorrError.malformedConfig(
                    "conv_group_size \(c.convGroupSize) must divide hidden_size \(c.hiddenSize)")
            }
            attentionConv = try LiLiCorrGroupedConv(
                weights: w, name: "\(p).attention_conv", blockSize: c.blockSize,
                taps: c.convKernelSize, groupSize: c.convGroupSize)
            mlpConv = try LiLiCorrGroupedConv(
                weights: w, name: "\(p).mlp_conv", blockSize: c.blockSize, taps: c.convKernelSize,
                groupSize: c.convGroupSize)
        } else {
            attentionConv = nil
            mlpConv = nil
        }
        super.init()
    }

    public func callAsFunction(
        _ h: MLXArray, contextK: MLXArray?, contextV: MLXArray?, position: Int, mask: MLXArray?
    ) -> MLXArray {
        let B = h.dim(0)
        let S = h.dim(1)
        let D = h.dim(2)
        var x = inputLayerNorm(h).reshaped([B * S, D])
        var kern: MLXArray? = nil
        if let attentionConv {
            (x, kern) = attentionConv.prepare(x)
        }
        var a = selfAttn(
            x.reshaped([B, S, D]), contextK: contextK, contextV: contextV, position: position,
            mask: mask
        ).reshaped([B * S, D])
        if let attentionConv, let kern {
            a = attentionConv.finish(a, coefficients: kern)
        }
        let h1 = h + a.reshaped([B, S, D])
        x = postAttentionLayerNorm(h1).reshaped([B * S, D])
        kern = nil
        if let mlpConv {
            (x, kern) = mlpConv.prepare(x)
        }
        var m = liliApply(downProj, silu(liliApply(gateProj, x)) * liliApply(upProj, x))
        if let mlpConv, let kern {
            m = mlpConv.finish(m, coefficients: kern)
        }
        return h1 + m.reshaped([B, S, D])
    }
}

public final class LiLiCorrLatticeLayer: Module {
    let attnNorm: LiLiCorrRMSNorm
    let mlpNorm: LiLiCorrRMSNorm
    let inWeight: MLXArray
    let inBias: MLXArray
    let outProj: Module
    let up: Module
    let down: Module
    public let heads: Int

    init(weights w: LiLiCorrWeights, prefix p: String, heads: Int) throws {
        attnNorm = try w.rmsNorm("\(p).attn_norm")
        mlpNorm = try w.rmsNorm("\(p).mlp_norm")
        inWeight = try w.take("\(p).attn.in_proj_weight")
        inBias = try w.take("\(p).attn.in_proj_bias")
        outProj = try w.linear("\(p).attn.out_proj")
        up = try w.linear("\(p).mlp.up_proj")
        down = try w.linear("\(p).mlp.down_proj")
        self.heads = heads
        super.init()
    }

    /// `h (B, L, D)`, `bias (1, heads, L, L)` additive attention bias in `h.dtype`.
    public func callAsFunction(_ h: MLXArray, bias: MLXArray) -> MLXArray {
        let B = h.dim(0)
        let L = h.dim(1)
        let D = h.dim(2)
        let x = attnNorm(h)
        let qkv = matmul(x, inWeight.T) + inBias
        let parts = qkv.split(parts: 3, axis: -1)
        let hd = D / heads
        let q = parts[0].reshaped([B, L, heads, hd]).transposed(0, 2, 1, 3)
        let k = parts[1].reshaped([B, L, heads, hd]).transposed(0, 2, 1, 3)
        let v = parts[2].reshaped([B, L, heads, hd]).transposed(0, 2, 1, 3)
        let o = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: pow(Float(hd), -0.5), mask: bias)
        let h1 = h + liliApply(outProj, o.transposed(0, 2, 1, 3).reshaped([B, L, D]))
        return h1 + liliApply(down, silu(liliApply(up, mlpNorm(h1))))
    }
}

/// fp32 L2 normalisation with the SGLang floor: `x / max(||x||, eps)`.
func liliL2Normalize(_ x: MLXArray, eps: Float) -> MLXArray {
    let xf = x.asType(.float32)
    let n = sqrt(sum(xf * xf, axis: -1, keepDims: true))
    return xf / maximum(n, MLXArray(eps))
}

public final class LiLiCorrHead: Module {
    public let topK: Int
    public let slots: Int
    public let heads: Int
    public let hidden: Int
    public let vectorEps: Float
    public let logitScale: Float
    let tokenProj: Module
    let passHiddenProj: Module
    let featureNormWeight: MLXArray
    let featureNormBias: MLXArray
    let featureUp: Module
    let featureDown: Module
    let slotEmbedding: MLXArray
    let rankEmbedding: MLXArray
    let contextProj: Module
    let layers: [LiLiCorrLatticeLayer]
    let outputNorm: LiLiCorrRMSNorm
    let anchorNorm: LiLiCorrRMSNorm
    let fSelf: MLXArray
    let fAnchor: MLXArray
    let fCross: MLXArray
    let fBias: MLXArray
    let outHead: Module
    let inHead: Module
    let anchorOutHead: Module
    /// `(1, heads, slots*K, slots*K)` fp32: relative-slot + same-slot bias over (slot, rank) rows.
    public let attnBias: MLXArray
    let rankFrac: MLXArray
    let isTop1: MLXArray

    init(weights w: LiLiCorrWeights) throws {
        let c = w.configuration
        let p = "lilicorr"
        topK = c.lilicorrCandidateTopK
        slots = c.blockSize - 1
        heads = c.lilicorrNumHeads
        hidden = c.lilicorrHiddenSize
        vectorEps = c.lilicorrVectorEps
        logitScale = c.lilicorrLogitScale
        tokenProj = try w.linear("\(p).token_proj")
        passHiddenProj = try w.linear("\(p).pass_hidden_proj")
        featureNormWeight = try w.take("\(p).feature_norm.weight")
        featureNormBias = try w.take("\(p).feature_norm.bias")
        featureUp = try w.linear("\(p).feature_mlp.up_proj")
        featureDown = try w.linear("\(p).feature_mlp.down_proj")
        slotEmbedding = try w.take("\(p).slot_embedding")
        rankEmbedding = try w.take("\(p).rank_embedding")
        contextProj = try w.linear("\(p).context_proj")
        layers = try (0 ..< c.lilicorrNumLayers).map {
            try LiLiCorrLatticeLayer(weights: w, prefix: "\(p).layers.\($0)", heads: c.lilicorrNumHeads)
        }
        outputNorm = try w.rmsNorm("\(p).output_norm")
        anchorNorm = try w.rmsNorm("\(p).anchor_norm")
        let fw = try w.take("\(p).factor_input_proj.weight")
        guard fw.ndim == 2, fw.dim(1) == 3 * hidden else {
            throw NemotronLiLiCorrError.malformedTensor(
                "\(p).factor_input_proj.weight shape \(fw.shape) != [*, \(3 * hidden)]")
        }
        fSelf = fw[0..., 0 ..< hidden]
        fAnchor = fw[0..., hidden ..< (2 * hidden)]
        fCross = fw[0..., (2 * hidden)...]
        fBias = try w.take("\(p).factor_input_proj.bias")
        outHead = try w.linear("\(p).out_head")
        inHead = try w.linear("\(p).in_head")
        anchorOutHead = try w.linear("\(p).anchor_out_head")

        // attention bias [1, heads, L, L] over (slot, rank) rows: relative-slot + same-slot
        let relBias = try w.take("\(p).relative_slot_bias")
        let same = try w.take("\(p).same_slot_bias")
        let block = c.blockSize
        guard relBias.ndim == 2, relBias.dim(0) == heads, relBias.dim(1) == 2 * block - 1,
            same.ndim == 1, same.dim(0) == heads
        else {
            throw NemotronLiLiCorrError.malformedTensor(
                "\(p).relative_slot_bias \(relBias.shape) / same_slot_bias \(same.shape) do not match heads=\(heads), block=\(block)")
        }
        let rel = relBias.asType(.float32).asArray(Float.self)
        let sameHost = same.asType(.float32).asArray(Float.self)
        let L = slots * topK
        var bias = [Float](repeating: 0, count: heads * L * L)
        let width = 2 * block - 1
        for h in 0 ..< heads {
            for i in 0 ..< L {
                let si = i / topK
                for j in 0 ..< L {
                    let sj = j / topK
                    let r = min(max(si - sj, -(block - 1)), block - 1)
                    var b = rel[h * width + r + block - 1]
                    if si == sj { b += sameHost[h] }
                    bias[(h * L + i) * L + j] = b
                }
            }
        }
        attnBias = MLXArray(bias, [1, heads, L, L])
        let k = topK
        rankFrac = MLXArray((0 ..< k).map { Float($0) / Float(max(1, k - 1)) })
        isTop1 = MLXArray((0 ..< k).map { $0 == 0 ? Float(1) : Float(0) })
        super.init()
    }

    /// `tokenEmb (S, K, Dm)`, `candLogp (S, K)` fp32, `passHidden (S, Dm)`, `anchorCtx (Dm,)`
    /// -> `start (K,)`, `pair (S-1, K, K)` (fp32).
    public func score(
        tokenEmbeddings tokenEmb: MLXArray, candidateLogProbs candLogp: MLXArray,
        passHidden: MLXArray, anchorContext anchorCtx: MLXArray
    ) -> (start: MLXArray, pair: MLXArray) {
        let S = candLogp.dim(0)
        let K = candLogp.dim(1)
        let tok = liliApply(tokenProj, tokenEmb.asType(.bfloat16))
        let pas = expandedDimensions(liliApply(passHiddenProj, passHidden.asType(.bfloat16)), axis: 1)
        let lp = candLogp.asType(.float32)
        let feats = stacked(
            [
                lp, exp(lp), lp - lp.max(axis: -1, keepDims: true),
                broadcast(rankFrac, to: lp.shape), broadcast(isTop1, to: lp.shape),
            ], axis: -1)
        let fn = MLXFast.layerNorm(
            feats.asType(.bfloat16), weight: featureNormWeight, bias: featureNormBias, eps: 1e-5)
        let fm = liliApply(featureDown, silu(liliApply(featureUp, fn)))
        var h = tok + pas + fm + slotEmbedding[0, 0, ..<S] + rankEmbedding[0, 0]
        h = h.reshaped([1, S * K, hidden])
        let bias = attnBias[0..., 0..., ..<(S * K), ..<(S * K)].asType(h.dtype)
        for layer in layers {
            h = layer(h, bias: bias)
        }
        h = outputNorm(h).reshaped([S, K, hidden])
        let anchor = anchorNorm(
            liliApply(contextProj, expandedDimensions(anchorCtx.asType(.bfloat16), axis: 0)))[0]
        let pre =
            matmul(h, fSelf.T) + fBias + matmul(anchor, fAnchor.T) + matmul(h * anchor, fCross.T)
        let fh = silu(pre)
        let outVec = liliL2Normalize(liliApply(outHead, fh), eps: vectorEps)
        let inVec = liliL2Normalize(liliApply(inHead, fh), eps: vectorEps)
        let anchorOut = liliL2Normalize(
            liliApply(anchorOutHead, expandedDimensions(anchor, axis: 0)), eps: vectorEps)[0]
        let start = sum(inVec[0] * anchorOut, axis: -1) * logitScale
        let pair = matmul(outVec[0 ..< (S - 1)], inVec[1...].transposed(0, 2, 1)) * logitScale
        return (start, pair)
    }
}

// MARK: - Lattice walk

public enum LiLiCorrLattice {
    /// Greedy left-to-right walk entirely on the graph: slot 0 by `start`, slot s by
    /// `pair[s-1][previous]`. Returns `(S,)` int32 token ids; nothing is read back to the host.
    public static func greedyWalk(start: MLXArray, pair: MLXArray, candidates: MLXArray) -> MLXArray {
        var prev = argMax(start, axis: -1).reshaped([1])
        var tokens = [candidates[0].take(prev, axis: 0)]
        for s in 0 ..< pair.dim(0) {
            let row = pair[s].take(prev, axis: 0)  // (1, K)
            prev = argMax(row, axis: -1)  // (1,)
            tokens.append(candidates[s + 1].take(prev, axis: 0))
        }
        return concatenated(tokens, axis: 0).asType(.int32)
    }

    /// Host reference of ``greedyWalk(start:pair:candidates:)`` (tests / debugging).
    public static func greedyWalkHost(start: [Float], pair: [[[Float]]], candidates: [[Int32]]) -> [Int32] {
        func argmax(_ v: [Float]) -> Int {
            var best = 0
            for i in 1 ..< v.count where v[i] > v[best] { best = i }
            return best
        }
        var prev = argmax(start)
        var out = [candidates[0][prev]]
        for s in 0 ..< pair.count {
            prev = argmax(pair[s][prev])
            out.append(candidates[s + 1][prev])
        }
        return out
    }

    /// Top-K candidates per slot in DESCENDING log-probability order (rank order matters for the
    /// head's rank features): partial selection, then a sort of the K.
    public static func topKCandidates(logProbs logp: MLXArray, k: Int) -> (ids: MLXArray, logProbs: MLXArray) {
        let part = argPartition(-logp, kth: k - 1, axis: -1)[0..., ..<k]
        let partLp = takeAlong(logp, part, axis: -1)
        let order = argSort(-partLp, axis: -1)
        let ids = takeAlong(part, order, axis: -1)
        let candLp = takeAlong(partLp, order, axis: -1)
        return (ids, candLp)
    }
}

// MARK: - Drafter

/// The stateful drafter: context K/V (last `window` committed target positions), block drafting
/// and the LiLiCorr proposal. Bolt shares the TARGET's embedding table and `lm_head`; both come
/// in as closures.
public final class NemotronLiLiCorrDrafter: Module {
    public let configuration: NemotronLiLiCorrConfiguration
    public var blockSize: Int { configuration.blockSize }
    public var maskTokenId: Int { configuration.maskTokenId }
    public var targetLayerIds: [Int] { configuration.targetLayerIds }
    public var window: Int { configuration.window }
    /// Expected width of a tap row (`len(target_layer_ids) * target hidden`), from `fc`.
    public let tapDimension: Int

    let fc: Module
    let hiddenNorm: LiLiCorrRMSNorm
    public let layers: [LiLiCorrDraftLayer]
    let norm: LiLiCorrRMSNorm
    public let lilicorr: LiLiCorrHead?

    // Context state (positions `contextEnd - contextRows ..< contextEnd`).
    private(set) var contextK: [MLXArray?]
    private(set) var contextV: [MLXArray?]
    /// Projected context features `(n, D)` for the kept window (lets `truncate` recover the anchor).
    private(set) var contextFeatures: MLXArray?
    public private(set) var contextEnd: Int = 0
    public private(set) var lastContext: MLXArray?

    public var contextRows: Int { contextFeatures?.dim(0) ?? 0 }

    public init(configuration: NemotronLiLiCorrConfiguration, weights tensors: [String: MLXArray]) throws {
        self.configuration = configuration
        let w = LiLiCorrWeights(tensors: tensors, configuration: configuration)
        fc = try w.linear("fc")
        if let q = fc as? NVFP4Linear {
            tapDimension = q.inputDimensions
        } else if let l = fc as? Linear {
            tapDimension = l.weight.dim(1)
        } else {
            throw NemotronLiLiCorrError.malformedTensor("fc is neither Linear nor NVFP4Linear")
        }
        hiddenNorm = try w.rmsNorm("hidden_norm")
        layers = try (0 ..< configuration.numHiddenLayers).map {
            try LiLiCorrDraftLayer(weights: w, index: $0)
        }
        norm = try w.rmsNorm("norm")
        lilicorr = configuration.lilicorrEnabled ? try LiLiCorrHead(weights: w) : nil
        contextK = Array(repeating: nil, count: configuration.numHiddenLayers)
        contextV = Array(repeating: nil, count: configuration.numHiddenLayers)
        super.init()
        eval(Array(tensors.values))
    }

    /// Load `<directory>/config.json` + every `*.safetensors` in `directory` (the bundle's
    /// `lilicorr/` folder).
    public static func load(directory: URL) throws -> NemotronLiLiCorrDrafter {
        let config = try NemotronLiLiCorrConfiguration.load(
            url: directory.appendingPathComponent("config.json"))
        let files = try FileManager.default.contentsOfDirectory(atPath: directory.path)
            .filter { $0.hasSuffix(".safetensors") }.sorted()
        guard !files.isEmpty else {
            throw NemotronLiLiCorrError.missingFile("\(directory.path)/*.safetensors")
        }
        var tensors: [String: MLXArray] = [:]
        for f in files {
            let loaded = try loadArrays(url: directory.appendingPathComponent(f))
            tensors.merge(loaded) { _, new in new }
        }
        return try NemotronLiLiCorrDrafter(configuration: config, weights: tensors)
    }

    // MARK: Context

    /// Forget every context row; the next `addContext` must start at `contextEnd`.
    public func reset(contextEnd: Int = 0) {
        contextK = Array(repeating: nil, count: layers.count)
        contextV = Array(repeating: nil, count: layers.count)
        contextFeatures = nil
        lastContext = nil
        self.contextEnd = contextEnd
    }

    /// Append the taps `(n, tapDimension)` of target positions `position ..< position + n`
    /// (`position` must equal `contextEnd`). Keeps only the last `window` positions.
    public func addContext(taps: MLXArray, position: Int) throws {
        guard position == contextEnd else {
            throw NemotronLiLiCorrError.contextPositionMismatch(expected: contextEnd, got: position)
        }
        guard taps.ndim == 2, taps.dim(1) == tapDimension else {
            throw NemotronLiLiCorrError.malformedTensor(
                "taps shape \(taps.shape) != [n, \(tapDimension)]")
        }
        let n = taps.dim(0)
        guard n > 0 else { return }
        let ctx = hiddenNorm(liliApply(fc, expandedDimensions(taps.asType(.bfloat16), axis: 0)))
        var rows = ctx[0]
        if let existing = contextFeatures {
            rows = concatenated([existing, rows], axis: 0)
        }
        if rows.dim(0) > window {
            rows = rows[(rows.dim(0) - window)...]
        }
        contextFeatures = rows
        lastContext = ctx[0, n - 1]
        var pending: [MLXArray] = [rows, lastContext!]
        for (i, layer) in layers.enumerated() {
            var (k, v) = layer.selfAttn.contextKV(ctx, position: position)
            if let kc = contextK[i], let vc = contextV[i] {
                k = concatenated([kc, k], axis: 2)
                v = concatenated([vc, v], axis: 2)
            }
            if k.dim(2) > window {
                k = k[0..., 0..., (k.dim(2) - window)...]
                v = v[0..., 0..., (v.dim(2) - window)...]
            }
            contextK[i] = k
            contextV[i] = v
            pending.append(k)
            pending.append(v)
        }
        contextEnd = position + n
        // Bounded graph without a host wait (Python 3e93c98a uses mx.async_eval here): the next
        // draftHidden reads this state on the same stream.
        asyncEval(pending)
    }

    /// Drop every context row at positions `>= position` (a rejected verify). When the kept
    /// window no longer reaches back to `position` the context becomes empty at `position` and
    /// the lane must re-add taps from its recorder before drafting.
    public func truncate(to position: Int) {
        guard position < contextEnd else { return }
        let rows = contextRows
        let drop = contextEnd - position
        if drop >= rows {
            reset(contextEnd: position)
            return
        }
        let keep = rows - drop
        contextFeatures = contextFeatures![0 ..< keep]
        lastContext = contextFeatures![keep - 1]
        for i in 0 ..< layers.count {
            contextK[i] = contextK[i]?[0..., 0..., 0 ..< keep]
            contextV[i] = contextV[i]?[0..., 0..., 0 ..< keep]
        }
        contextEnd = position
    }

    // MARK: Drafting

    /// Boolean `(1, 1, B, n + B)` mask: key position `kpos` is visible from block query `qpos`
    /// iff `kpos > qpos - window` (non-causal inside the block, left window over the context).
    public static func attentionMask(contextRows n: Int, position: Int, blockLength B: Int, window: Int)
        -> MLXArray
    {
        let kpos = MLXArray(
            (Int32(position - n) ..< Int32(position)).map { $0 }
                + (Int32(position) ..< Int32(position + B)).map { $0 })
        let qpos = MLXArray((Int32(position) ..< Int32(position + B)).map { $0 })
        let mask = expandedDimensions(kpos, axis: 0) .> (expandedDimensions(qpos, axis: 1) - Int32(window))
        return expandedDimensions(mask, axes: [0, 1])
    }

    /// `blockEmbeddings (1, B, D)` at positions `position ..< position + B` (`position ==
    /// contextEnd`) -> final draft hidden `(1, B, D)`.
    public func draftHidden(blockEmbeddings: MLXArray, position: Int) throws -> MLXArray {
        guard position == contextEnd else {
            throw NemotronLiLiCorrError.contextPositionMismatch(expected: contextEnd, got: position)
        }
        let B = blockEmbeddings.dim(1)
        let n = contextK.first??.dim(2) ?? 0
        let mask = Self.attentionMask(contextRows: n, position: position, blockLength: B, window: window)
        var h = blockEmbeddings
        for (i, layer) in layers.enumerated() {
            h = layer(h, contextK: contextK[i], contextV: contextV[i], position: position, mask: mask)
        }
        return norm(h)
    }

    /// `hidden (1, B, D)` -> (LiLiCorr path tokens `(B-1,)` int32 or nil, plain top-1 `(B-1,)`)
    /// for slots `1 ..< B`. Both stay on the graph (the lattice walk runs as lazy ops, nothing is
    /// read back here); the caller evaluates both in ONE sync, see ``proposeTokens``.
    public func propose(
        hidden: MLXArray, lmHead: (MLXArray) -> MLXArray, embed: (MLXArray) -> MLXArray,
        useLiLiCorr: Bool = true
    ) throws -> (lilicorrPath: MLXArray?, top1: MLXArray) {
        let passHidden = hidden[0, 1...]
        let logits = lmHead(passHidden).asType(.float32)
        let logp = logits - logits.logSumExp(axis: -1, keepDims: true)
        let top1 = argMax(logp, axis: -1).asType(.int32)
        guard let lilicorr, useLiLiCorr else { return (nil, top1) }
        guard let lastContext else { throw NemotronLiLiCorrError.noAnchorContext }
        let (ids, candLp) = LiLiCorrLattice.topKCandidates(logProbs: logp, k: lilicorr.topK)
        let idsInt = ids.asType(.int32)
        let (start, pair) = lilicorr.score(
            tokenEmbeddings: embed(idsInt), candidateLogProbs: candLp, passHidden: passHidden,
            anchorContext: lastContext)
        return (LiLiCorrLattice.greedyWalk(start: start, pair: pair, candidates: idsInt), top1)
    }

    /// ``propose(hidden:lmHead:embed:useLiLiCorr:)`` with exactly one device sync for everything
    /// the host needs (the Python port's `mx.eval(start, pair, ids, top1)` collapsed further: the
    /// walk itself is on the graph, so only the `(B-1,)` path and top-1 ids come back).
    public func proposeTokens(
        hidden: MLXArray, lmHead: (MLXArray) -> MLXArray, embed: (MLXArray) -> MLXArray,
        useLiLiCorr: Bool = true
    ) throws -> (lilicorrPath: [Int32]?, top1: [Int32]) {
        let (path, top1) = try propose(hidden: hidden, lmHead: lmHead, embed: embed, useLiLiCorr: useLiLiCorr)
        if let path {
            eval(path, top1)
            return (path.asArray(Int32.self), top1.asArray(Int32.self))
        }
        eval(top1)
        return (nil, top1.asArray(Int32.self))
    }
}

// MARK: - Target tap hook

/// Records the OUTPUTS of selected backbone blocks (before the final norm) while a target forward
/// runs. Nil by default on the backbone; costs one optional check per block when unset.
public final class NemotronHBlockOutputTap {
    public let layerIndices: Set<Int>
    public let record: (_ layerIndex: Int, _ output: MLXArray) -> Void

    public init(layerIndices: Set<Int>, record: @escaping (_ layerIndex: Int, _ output: MLXArray) -> Void) {
        self.layerIndices = layerIndices
        self.record = record
    }
}

/// A model whose backbone can expose block outputs (Nemotron-H and its VLM wrappers).
public protocol NemotronHBlockOutputTappable: AnyObject {
    var blockOutputTap: NemotronHBlockOutputTap? { get set }
}
