//
//  Flux1Native.swift
//  vMLXFluxModels
//
//  Native FLUX.1 DiT transformer (the `transformer` component), ported from
//  mflux `flux/model/flux_transformer/`. Dual-stream (19 JointTransformerBlock)
//  then fused (38 SingleTransformerBlock), 3072-dim, 24 heads × 128, flow-matching.
//  Text conditioning: T5 per-token (prompt_embeds, 4096→3072) + CLIP pooled
//  (768) folded into the timestep modulation. See FLUX_SCHNELL_PORT_PLAN.md.
//

import Foundation
@preconcurrency import MLX
import MLXNN
import MLXRandom
import VMLXTokenizers
import vMLXFluxKit

// MARK: - 3-axis RoPE (EmbedND)

enum FluxRoPE {
    static let axesDim = [16, 56, 56]
    static let theta: Float = 10000

    /// ids: (1, seq, 3). Returns freqs (1, 1, seq, 64, 2, 2).
    static func embed(_ ids: MLXArray) -> MLXArray {
        let parts = (0 ..< 3).map { rope(ids[0..., 0..., $0], dim: axesDim[$0]) }  // each (1, seq, d/2, 2, 2)
        let emb = concatenated(parts, axis: -3)  // (1, seq, 64, 2, 2)
        return emb.reshaped([1, 1, emb.dim(1), emb.dim(2), 2, 2])
    }

    private static func rope(_ pos: MLXArray, dim: Int) -> MLXArray {
        let scale = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) / Float(dim) })
        let omega = MLXArray(Float(1)) / pow(MLXArray(theta), scale)  // (dim/2,)
        let seq = pos.dim(1)
        let out = pos.reshaped([1, seq, 1]) * omega.reshaped([1, 1, dim / 2])  // (1, seq, dim/2)
        let c = cos(out), s = sin(out)
        let stacked = stacked([c, -s, s, c], axis: -1)  // (1, seq, dim/2, 4)
        return stacked.reshaped([1, seq, dim / 2, 2, 2])
    }

    /// Apply rope to q/k of shape (1, H, S, 128). freqs (1,1,S,64,2,2).
    static func apply(_ x: MLXArray, freqs: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let s = x.shape
        let x_ = xf.reshaped(Array(s.dropLast()) + [-1, 1, 2])  // (1,H,S,64,1,2)
        let f0 = freqs[.ellipsis, 0]  // (1,1,S,64,2)
        let f1 = freqs[.ellipsis, 1]
        let out = f0 * x_[.ellipsis, 0] + f1 * x_[.ellipsis, 1]  // (1,H,S,64,2)
        return out.reshaped(s).asType(.float32)
    }
}

// MARK: - helpers

private func layerNormNoAffine(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let m = mean(x, axis: -1, keepDims: true)
    let c = x - m
    let v = mean(c * c, axis: -1, keepDims: true)
    return c * rsqrt(v + MLXArray(eps))
}

private func sdpaHeads(_ x: MLXArray, heads: Int, headDim: Int) -> MLXArray {
    let seq = x.dim(1)
    return x.reshaped([1, seq, heads, headDim]).transposed(0, 2, 1, 3)
}

private func mergeHeads(_ x: MLXArray, dim: Int) -> MLXArray {
    let seq = x.dim(2)
    return x.transposed(0, 2, 1, 3).reshaped([1, seq, dim])
}

// MARK: - Time/text conditioning

final class FluxTimeTextEmbed {
    private let tsLin1: MFluxLinear
    private let tsLin2: MFluxLinear
    private let txtLin1: MFluxLinear
    private let txtLin2: MFluxLinear

    init(store: MFluxStore, component: String) throws {
        let t = "time_text_embed"
        tsLin1 = try store.linear(component, "\(t).timestep_embedder.linear_1", inputDimensions: 256, outputDimensions: 3072, bias: true)
        tsLin2 = try store.linear(component, "\(t).timestep_embedder.linear_2", inputDimensions: 3072, outputDimensions: 3072, bias: true)
        txtLin1 = try store.linear(component, "\(t).text_embedder.linear_1", inputDimensions: 768, outputDimensions: 3072, bias: true)
        txtLin2 = try store.linear(component, "\(t).text_embedder.linear_2", inputDimensions: 3072, outputDimensions: 3072, bias: true)
    }

    /// timestep scalar, pooled (1,768) → conditioning (1, 3072). Schnell: no guidance.
    func callAsFunction(timestep: Float, pooled: MLXArray) -> MLXArray {
        let proj = FluxTimeTextEmbed.timeProj(timestep)  // (1,256)
        let timeEmb = tsLin2(silu(tsLin1(proj)))
        let textEmb = txtLin2(silu(txtLin1(pooled)))
        return timeEmb + textEmb
    }

    static func timeProj(_ t: Float) -> MLXArray {
        let half = 128
        let maxPeriod = Float(10000)
        let exponent = MLXArray((0 ..< half).map { -log(maxPeriod) * Float($0) / Float(half) })
        let emb = exp(exponent)  // (128,)
        let e = MLXArray(t) * emb  // (128,)
        let sc = concatenated([sin(e), cos(e)], axis: -1)  // (256,)
        // swap halves: [cos | sin]
        let swapped = concatenated([sc[half ..< 2 * half], sc[0 ..< half]], axis: -1)
        return swapped.reshaped([1, 256])
    }
}

// MARK: - AdaLayerNorm variants

/// AdaLayerNormZero: 6 modulation params (shift/scale/gate for msa+mlp).
final class FluxAdaNormZero {
    private let linear: MFluxLinear
    init(store: MFluxStore, component: String, prefix: String) throws {
        linear = try store.linear(component, "\(prefix).linear", inputDimensions: 3072, outputDimensions: 18432, bias: true)
    }
    /// returns (normed, gateMSA, shiftMLP, scaleMLP, gateMLP)
    func callAsFunction(_ h: MLXArray, text: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
        let m = linear(silu(text))  // (1, 18432)
        let cs = 3072
        func chunk(_ i: Int) -> MLXArray { m[0..., (i * cs) ..< ((i + 1) * cs)] }
        let shiftMSA = chunk(0), scaleMSA = chunk(1), gateMSA = chunk(2)
        let shiftMLP = chunk(3), scaleMLP = chunk(4), gateMLP = chunk(5)
        let normed = layerNormNoAffine(h) * (1 + scaleMSA.expandedDimensions(axis: 1)) + shiftMSA.expandedDimensions(axis: 1)
        return (normed, gateMSA, shiftMLP, scaleMLP, gateMLP)
    }
}

/// AdaLayerNormZeroSingle: 3 modulation params.
final class FluxAdaNormZeroSingle {
    private let linear: MFluxLinear
    init(store: MFluxStore, component: String, prefix: String) throws {
        linear = try store.linear(component, "\(prefix).linear", inputDimensions: 3072, outputDimensions: 9216, bias: true)
    }
    func callAsFunction(_ h: MLXArray, text: MLXArray) -> (MLXArray, MLXArray) {
        let m = linear(silu(text))
        let cs = 3072
        let shift = m[0..., 0 ..< cs], scale = m[0..., cs ..< 2 * cs], gate = m[0..., 2 * cs ..< 3 * cs]
        let normed = layerNormNoAffine(h) * (1 + scale.expandedDimensions(axis: 1)) + shift.expandedDimensions(axis: 1)
        return (normed, gate)
    }
}

/// AdaLayerNormContinuous: scale THEN shift, no-bias linear.
final class FluxAdaNormContinuous {
    private let linear: MFluxLinear
    init(store: MFluxStore, component: String, prefix: String) throws {
        linear = try store.linear(component, "\(prefix).linear", inputDimensions: 3072, outputDimensions: 6144, bias: false)
    }
    func callAsFunction(_ x: MLXArray, text: MLXArray) -> MLXArray {
        let m = linear(silu(text))
        let cs = 3072
        let scale = m[0..., 0 ..< cs], shift = m[0..., cs ..< 2 * cs]
        return layerNormNoAffine(x) * (1 + scale).expandedDimensions(axis: 1) + shift.expandedDimensions(axis: 1)
    }
}

// MARK: - Feed forward

private final class FluxFeedForward {
    private let l1: MFluxLinear
    private let l2: MFluxLinear
    private let approx: Bool
    init(store: MFluxStore, component: String, prefix: String, approx: Bool) throws {
        l1 = try store.linear(component, "\(prefix).linear1", inputDimensions: 3072, outputDimensions: 12288, bias: true)
        l2 = try store.linear(component, "\(prefix).linear2", inputDimensions: 12288, outputDimensions: 3072, bias: true)
        self.approx = approx
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = l1(x)
        let a = approx ? geluApproximate(h) : gelu(h)
        return l2(a)
    }
}

// MARK: - Attention

private struct FluxAttnProj {
    let q: MFluxLinear, k: MFluxLinear, v: MFluxLinear
    let normQ: MFluxRMSNorm, normK: MFluxRMSNorm
}

private func processQKV(_ x: MLXArray, _ p: FluxAttnProj, heads: Int, headDim: Int) -> (MLXArray, MLXArray, MLXArray) {
    var q = sdpaHeads(p.q(x), heads: heads, headDim: headDim)
    var k = sdpaHeads(p.k(x), heads: heads, headDim: headDim)
    let v = sdpaHeads(p.v(x), heads: heads, headDim: headDim)
    q = p.normQ(q.asType(.float32)).asType(q.dtype)
    k = p.normK(k.asType(.float32)).asType(k.dtype)
    return (q, k, v)
}

final class FluxJointAttention {
    private let img: FluxAttnProj
    private let txt: FluxAttnProj
    private let toOut: MFluxLinear
    private let toAddOut: MFluxLinear
    private let heads = 24
    private let headDim = 128

    init(store: MFluxStore, component: String, prefix: String) throws {
        let a = "\(prefix).attn"
        img = FluxAttnProj(
            q: try store.linear(component, "\(a).to_q", inputDimensions: 3072, outputDimensions: 3072, bias: true),
            k: try store.linear(component, "\(a).to_k", inputDimensions: 3072, outputDimensions: 3072, bias: true),
            v: try store.linear(component, "\(a).to_v", inputDimensions: 3072, outputDimensions: 3072, bias: true),
            normQ: try store.rmsNorm(component, "\(a).norm_q", eps: 1e-6),
            normK: try store.rmsNorm(component, "\(a).norm_k", eps: 1e-6))
        txt = FluxAttnProj(
            q: try store.linear(component, "\(a).add_q_proj", inputDimensions: 3072, outputDimensions: 3072, bias: true),
            k: try store.linear(component, "\(a).add_k_proj", inputDimensions: 3072, outputDimensions: 3072, bias: true),
            v: try store.linear(component, "\(a).add_v_proj", inputDimensions: 3072, outputDimensions: 3072, bias: true),
            normQ: try store.rmsNorm(component, "\(a).norm_added_q", eps: 1e-6),
            normK: try store.rmsNorm(component, "\(a).norm_added_k", eps: 1e-6))
        toOut = try store.linear(component, "\(a).to_out.0", inputDimensions: 3072, outputDimensions: 3072, bias: true)
        toAddOut = try store.linear(component, "\(a).to_add_out", inputDimensions: 3072, outputDimensions: 3072, bias: true)
    }

    /// returns (imgOut, txtOut)
    func callAsFunction(_ hidden: MLXArray, encoder: MLXArray, freqs: MLXArray) -> (MLXArray, MLXArray) {
        let txtLen = encoder.dim(1)
        let (iq, ik, iv) = processQKV(hidden, img, heads: heads, headDim: headDim)
        let (tq, tk, tv) = processQKV(encoder, txt, heads: heads, headDim: headDim)
        var q = concatenated([tq, iq], axis: 2)
        var k = concatenated([tk, ik], axis: 2)
        let v = concatenated([tv, iv], axis: 2)
        q = FluxRoPE.apply(q, freqs: freqs)
        k = FluxRoPE.apply(k, freqs: freqs)
        let scale = Float(1.0 / sqrt(Double(headDim)))
        let attended = MLX.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        let merged = mergeHeads(attended, dim: 3072)  // (1, S, 3072)
        let txtOut = toAddOut(merged[0..., 0 ..< txtLen, 0...])
        let imgOut = toOut(merged[0..., txtLen..., 0...])
        return (imgOut, txtOut)
    }
}

final class FluxSingleAttention {
    private let p: FluxAttnProj
    private let heads = 24
    private let headDim = 128
    init(store: MFluxStore, component: String, prefix: String) throws {
        let a = "\(prefix).attn"
        p = FluxAttnProj(
            q: try store.linear(component, "\(a).to_q", inputDimensions: 3072, outputDimensions: 3072, bias: true),
            k: try store.linear(component, "\(a).to_k", inputDimensions: 3072, outputDimensions: 3072, bias: true),
            v: try store.linear(component, "\(a).to_v", inputDimensions: 3072, outputDimensions: 3072, bias: true),
            normQ: try store.rmsNorm(component, "\(a).norm_q", eps: 1e-6),
            normK: try store.rmsNorm(component, "\(a).norm_k", eps: 1e-6))
    }
    func callAsFunction(_ x: MLXArray, freqs: MLXArray) -> MLXArray {
        var (q, k, v) = processQKV(x, p, heads: heads, headDim: headDim)
        q = FluxRoPE.apply(q, freqs: freqs)
        k = FluxRoPE.apply(k, freqs: freqs)
        let scale = Float(1.0 / sqrt(Double(headDim)))
        let attended = MLX.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        return mergeHeads(attended, dim: 3072)
    }
}

// MARK: - Blocks

final class FluxJointBlock {
    private let norm1: FluxAdaNormZero
    private let norm1Context: FluxAdaNormZero
    private let attn: FluxJointAttention
    private let ff: FluxFeedForward
    private let ffContext: FluxFeedForward

    init(store: MFluxStore, component: String, index: Int) throws {
        let p = "transformer_blocks.\(index)"
        norm1 = try FluxAdaNormZero(store: store, component: component, prefix: "\(p).norm1")
        norm1Context = try FluxAdaNormZero(store: store, component: component, prefix: "\(p).norm1_context")
        attn = try FluxJointAttention(store: store, component: component, prefix: p)
        ff = try FluxFeedForward(store: store, component: component, prefix: "\(p).ff", approx: false)
        ffContext = try FluxFeedForward(store: store, component: component, prefix: "\(p).ff_context", approx: true)
    }

    func callAsFunction(_ hidden: MLXArray, encoder: MLXArray, text: MLXArray, freqs: MLXArray) -> (MLXArray, MLXArray) {
        let (nh, gMSA, sMLP, scMLP, gMLP) = norm1(hidden, text: text)
        let (nenc, cgMSA, csMLP, cscMLP, cgMLP) = norm1Context(encoder, text: text)
        let (attnOut, ctxOut) = attn(nh, encoder: nenc, freqs: freqs)
        let newHidden = applyFF(hidden, attnOut, gMSA, sMLP, scMLP, gMLP, ff)
        let newEnc = applyFF(encoder, ctxOut, cgMSA, csMLP, cscMLP, cgMLP, ffContext)
        return (newEnc, newHidden)
    }

    private func applyFF(_ h0: MLXArray, _ attnOut: MLXArray, _ gateMSA: MLXArray, _ shiftMLP: MLXArray,
                         _ scaleMLP: MLXArray, _ gateMLP: MLXArray, _ ff: FluxFeedForward) -> MLXArray {
        var h = h0 + gateMSA.expandedDimensions(axis: 1) * attnOut
        var n = layerNormNoAffine(h)
        n = n * (1 + scaleMLP.expandedDimensions(axis: 1)) + shiftMLP.expandedDimensions(axis: 1)
        h = h + gateMLP.expandedDimensions(axis: 1) * ff(n)
        return h
    }
}

final class FluxSingleBlock {
    private let norm: FluxAdaNormZeroSingle
    private let attn: FluxSingleAttention
    private let projMLP: MFluxLinear
    private let projOut: MFluxLinear

    init(store: MFluxStore, component: String, index: Int) throws {
        let p = "single_transformer_blocks.\(index)"
        norm = try FluxAdaNormZeroSingle(store: store, component: component, prefix: "\(p).norm")
        attn = try FluxSingleAttention(store: store, component: component, prefix: p)
        projMLP = try store.linear(component, "\(p).proj_mlp", inputDimensions: 3072, outputDimensions: 12288, bias: true)
        projOut = try store.linear(component, "\(p).proj_out", inputDimensions: 15360, outputDimensions: 3072, bias: true)
    }

    func callAsFunction(_ hidden: MLXArray, text: MLXArray, freqs: MLXArray) -> MLXArray {
        let residual = hidden
        let (nh, gate) = norm(hidden, text: text)
        let attnOut = attn(nh, freqs: freqs)
        let ff = geluApproximate(projMLP(nh))
        let cat = concatenated([attnOut, ff], axis: 2)  // (1, S, 15360)
        let out = gate.expandedDimensions(axis: 1) * projOut(cat)
        return residual + out
    }
}

// MARK: - Transformer assembler

final class FluxTransformer {
    private let component: String
    private let xEmbedder: MFluxLinear
    private let contextEmbedder: MFluxLinear
    private let timeText: FluxTimeTextEmbed
    private let jointBlocks: [FluxJointBlock]
    private let singleBlocks: [FluxSingleBlock]
    private let normOut: FluxAdaNormContinuous
    private let projOut: MFluxLinear

    init(store: MFluxStore, component: String = "transformer", joint: Int = 19, single: Int = 38) throws {
        self.component = component
        xEmbedder = try store.linear(component, "x_embedder", inputDimensions: 64, outputDimensions: 3072, bias: true)
        contextEmbedder = try store.linear(component, "context_embedder", inputDimensions: 4096, outputDimensions: 3072, bias: true)
        timeText = try FluxTimeTextEmbed(store: store, component: component)
        jointBlocks = try (0 ..< joint).map { try FluxJointBlock(store: store, component: component, index: $0) }
        singleBlocks = try (0 ..< single).map { try FluxSingleBlock(store: store, component: component, index: $0) }
        normOut = try FluxAdaNormContinuous(store: store, component: component, prefix: "norm_out")
        projOut = try store.linear(component, "proj_out", inputDimensions: 3072, outputDimensions: 64, bias: true)
    }

    /// latents (1, hw, 64), promptEmbeds (1, Tt, 4096), pooled (1,768), timestep scalar,
    /// height/width in pixels. Returns predicted noise (1, hw, 64).
    func callAsFunction(latents: MLXArray, promptEmbeds: MLXArray, pooled: MLXArray,
                        timestep: Float, height: Int, width: Int) -> MLXArray {
        var hidden = xEmbedder(latents)
        var encoder = contextEmbedder(promptEmbeds)
        let text = timeText(timestep: timestep, pooled: pooled)
        let freqs = FluxTransformer.ropeFreqs(txtLen: promptEmbeds.dim(1), height: height, width: width)
        for block in jointBlocks {
            (encoder, hidden) = block(hidden, encoder: encoder, text: text, freqs: freqs)
        }
        let txtLen = encoder.dim(1)
        var joined = concatenated([encoder, hidden], axis: 1)
        for block in singleBlocks {
            joined = block(joined, text: text, freqs: freqs)
        }
        var out = joined[0..., txtLen..., 0...]
        out = normOut(out, text: text)
        return projOut(out)
    }

    static func ropeFreqs(txtLen: Int, height: Int, width: Int) -> MLXArray {
        let lh = height / 16, lw = width / 16
        // txt ids = zeros(1, txtLen, 3)
        let txtIds = MLXArray.zeros([1, txtLen, 3])
        // img ids: [.,1]=row, [.,2]=col over lh×lw grid → (1, lh*lw, 3)
        var imgVals = [Float](repeating: 0, count: lh * lw * 3)
        for r in 0 ..< lh {
            for c in 0 ..< lw {
                let base = (r * lw + c) * 3
                imgVals[base + 1] = Float(r)
                imgVals[base + 2] = Float(c)
            }
        }
        let imgIds = MLXArray(imgVals, [1, lh * lw, 3])
        let ids = concatenated([txtIds, imgIds], axis: 1)
        return FluxRoPE.embed(ids)
    }
}

// MARK: - VAE decoder (AutoencoderKL — identical family to z-image, flux conv2d keys)

private final class FluxVAEResnet {
    private let norm1: MFluxGroupNorm
    private let conv1: MFluxConv2D
    private let norm2: MFluxGroupNorm
    private let conv2: MFluxConv2D
    private let shortcut: MFluxConv2D?
    init(store: MFluxStore, prefix: String) throws {
        norm1 = try store.groupNorm("vae", "\(prefix).norm1")
        conv1 = try store.conv2d("vae", "\(prefix).conv1", padding: 1)
        norm2 = try store.groupNorm("vae", "\(prefix).norm2")
        conv2 = try store.conv2d("vae", "\(prefix).conv2", padding: 1)
        shortcut = store.hasKey("vae", "\(prefix).conv_shortcut.weight")
            ? try store.conv2d("vae", "\(prefix).conv_shortcut") : nil
    }
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let nhwc = input.transposed(0, 2, 3, 1)
        var h = conv1(silu(norm1(nhwc)))
        h = conv2(silu(norm2(h)))
        let residual = shortcut?(nhwc) ?? nhwc
        return (residual + h).transposed(0, 3, 1, 2)
    }
}

private final class FluxVAEAttention {
    private let groupNorm: MFluxGroupNorm
    private let toQ: MFluxLinear, toK: MFluxLinear, toV: MFluxLinear, toOut: MFluxLinear
    private let channels: Int
    init(store: MFluxStore, prefix: String, channels: Int = 512) throws {
        self.channels = channels
        groupNorm = try store.groupNorm("vae", "\(prefix).group_norm")
        toQ = try store.linear("vae", "\(prefix).to_q", inputDimensions: channels, outputDimensions: channels, bias: true)
        toK = try store.linear("vae", "\(prefix).to_k", inputDimensions: channels, outputDimensions: channels, bias: true)
        toV = try store.linear("vae", "\(prefix).to_v", inputDimensions: channels, outputDimensions: channels, bias: true)
        toOut = try store.linear("vae", "\(prefix).to_out.0", inputDimensions: channels, outputDimensions: channels, bias: true)
    }
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let nhwc = input.transposed(0, 2, 3, 1)
        let b = nhwc.dim(0), h = nhwc.dim(1), w = nhwc.dim(2)
        let normed = groupNorm(nhwc.asType(.float32)).asType(input.dtype)
        let q = toQ(normed).reshaped([b, h * w, 1, channels]).transposed(0, 2, 1, 3)
        let k = toK(normed).reshaped([b, h * w, 1, channels]).transposed(0, 2, 1, 3)
        let v = toV(normed).reshaped([b, h * w, 1, channels]).transposed(0, 2, 1, 3)
        let scale = Float(1.0 / sqrt(Float(channels)))
        let att = MLX.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        let out = att.transposed(0, 2, 1, 3).reshaped([b, h, w, channels])
        return (nhwc + toOut(out)).transposed(0, 3, 1, 2)
    }
}

private final class FluxVAEUpsampler {
    private let conv: MFluxConv2D
    init(store: MFluxStore, prefix: String) throws { conv = try store.conv2d("vae", "\(prefix).conv", padding: 1) }
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let up = repeated(repeated(input, count: 2, axis: 2), count: 2, axis: 3)
        return conv(up.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
    }
}

private final class FluxVAEDecoder {
    private let convIn: MFluxConv2D
    private let midResnet0: FluxVAEResnet
    private let midAttn: FluxVAEAttention
    private let midResnet1: FluxVAEResnet
    private let upBlocks: [(resnets: [FluxVAEResnet], up: FluxVAEUpsampler?)]
    private let normOut: MFluxGroupNorm
    private let convOut: MFluxConv2D

    init(store: MFluxStore) throws {
        convIn = try store.conv2d("vae", "decoder.conv_in.conv2d", padding: 1)
        midResnet0 = try FluxVAEResnet(store: store, prefix: "decoder.mid_block.resnets.0")
        midAttn = try FluxVAEAttention(store: store, prefix: "decoder.mid_block.attentions.0")
        midResnet1 = try FluxVAEResnet(store: store, prefix: "decoder.mid_block.resnets.1")
        var blocks: [(resnets: [FluxVAEResnet], up: FluxVAEUpsampler?)] = []
        for i in 0 ..< 4 {
            var resnets: [FluxVAEResnet] = []
            for l in 0 ..< 3 {
                resnets.append(try FluxVAEResnet(store: store, prefix: "decoder.up_blocks.\(i).resnets.\(l)"))
            }
            let up = i < 3 ? try FluxVAEUpsampler(store: store, prefix: "decoder.up_blocks.\(i).upsamplers.0") : nil
            blocks.append((resnets, up))
        }
        upBlocks = blocks
        normOut = try store.groupNorm("vae", "decoder.conv_norm_out.norm")
        convOut = try store.conv2d("vae", "decoder.conv_out.conv2d", padding: 1)
    }

    /// latents (1,16,h/8,w/8) → image (1,3,H,W) in [0,1].
    func decode(_ latents: MLXArray) -> MLXArray {
        let scaled = latents / MLXArray(Float(0.3611)) + MLXArray(Float(0.1159))
        var h = convIn(scaled.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
        h = midResnet1(midAttn(midResnet0(h)))
        for block in upBlocks {
            for r in block.resnets { h = r(h) }
            if let up = block.up { h = up(h) }
        }
        h = normOut(h.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
        h = silu(h)
        h = convOut(h.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
        return VAEDecoder.postprocess(h)
    }
}

// MARK: - Tokenizers

private final class FluxTokenizers {
    let clip: any VMLXTokenizers.Tokenizer
    let t5: any VMLXTokenizers.Tokenizer
    init(modelPath: URL) async throws {
        clip = try await AutoTokenizer.from(modelFolder: modelPath.appendingPathComponent("tokenizer"), strict: false)
        t5 = try await AutoTokenizer.from(modelFolder: modelPath.appendingPathComponent("tokenizer_2"), strict: false)
    }
    /// CLIP: pad/truncate to 77. Returns (1, 77) Int32.
    func clipIds(_ prompt: String) -> MLXArray {
        padTo(clip.encode(text: prompt, addSpecialTokens: true), length: 77, pad: clip.eosTokenId ?? 0)
    }
    /// T5: pad/truncate to 256. Returns (1, 256) Int32.
    func t5Ids(_ prompt: String) -> MLXArray {
        padTo(t5.encode(text: prompt, addSpecialTokens: true), length: 256, pad: 0)
    }
    private func padTo(_ tokens: [Int], length: Int, pad: Int) -> MLXArray {
        var t = tokens
        if t.count > length { t = Array(t.prefix(length)) }
        else { t += Array(repeating: pad, count: length - t.count) }
        return MLXArray(t.map(Int32.init)).reshaped([1, length])
    }
}

// MARK: - Pipeline

final class FluxSchnellPipeline {
    private let t5: T5XXLEncoder
    private let clip: CLIPTextEncoder
    private let transformer: FluxTransformer
    private let vae: FluxVAEDecoder
    private let tokenizers: FluxTokenizers

    init(modelPath: URL) async throws {
        let loaded = try WeightLoader.load(from: modelPath)
        let store = MFluxStore(loaded)
        self.t5 = try T5XXLEncoder(store: store)
        self.clip = try CLIPTextEncoder(store: store)
        self.transformer = try FluxTransformer(store: store)
        self.vae = try FluxVAEDecoder(store: store)
        self.tokenizers = try await FluxTokenizers(modelPath: modelPath)
    }

    private static func dbg(_ label: String, _ x: MLXArray) {
        eval(x)
        let f = x.asType(.float32)
        let mn = mean(f).item(Float.self)
        let mx = MLX.max(f).item(Float.self)
        let finite = mn.isFinite && mx.isFinite
        FileHandle.standardError.write("[flux] \(label) shape=\(x.shape) mean=\(mn) max=\(mx) finite=\(finite)\n".data(using: .utf8)!)
    }

    func generate(prompt: String, width: Int, height: Int, steps: Int, seed: UInt64?,
                  progress: (Int, Int, Double?) -> Void) throws -> MLXArray {
        guard width % 16 == 0, height % 16 == 0 else {
            throw FluxError.invalidRequest("Flux width/height must be divisible by 16")
        }
        let promptEmbeds = t5(tokenizers.t5Ids(prompt))       // (1, 256, 4096)
        let pooled = clip(tokenizers.clipIds(prompt))          // (1, 768)
        FluxSchnellPipeline.dbg("t5", promptEmbeds)
        FluxSchnellPipeline.dbg("clip", pooled)

        let hw = (height / 16) * (width / 16)
        if let seed { MLXRandom.seed(seed) }
        var latents = MLXRandom.normal([1, hw, 64]).asType(.float32)
        let scheduler = FlowMatchEulerScheduler(steps: steps, imageSeqLen: hw)

        let start = Date()
        for step in 0 ..< steps {
            let timestep = scheduler.sigmas[step] * 1000.0
            let noise = transformer(latents: latents, promptEmbeds: promptEmbeds, pooled: pooled,
                                    timestep: timestep, height: height, width: width)
            latents = scheduler.step(latent: latents, velocity: noise, stepIndex: step)
            eval(latents)
            if step == 0 { FluxSchnellPipeline.dbg("step0", latents) }
            let elapsed = Date().timeIntervalSince(start)
            progress(step + 1, steps, elapsed / Double(step + 1) * Double(steps - step - 1))
        }
        // unpack (1, hw, 64) → (1, 16, h/8, w/8)
        let lh = height / 16, lw = width / 16
        var unpacked = latents.reshaped([1, lh, lw, 16, 2, 2])
        unpacked = unpacked.transposed(0, 3, 1, 4, 2, 5).reshaped([1, 16, lh * 2, lw * 2])
        FluxSchnellPipeline.dbg("unpacked", unpacked)
        let image = vae.decode(unpacked)
        FluxSchnellPipeline.dbg("image", image)
        return image
    }
}
