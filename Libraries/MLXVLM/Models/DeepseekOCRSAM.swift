//
//  DeepseekOCRSAM.swift
//  mlx-swift-lm
//
//  SAM-ViT-B vision encoder for the DeepSeek-OCR DeepEncoder (the `sam_model`
//  branch). Faithful port of mlx-vlm's deepseekocr/sam.py.
//
//  Architecture (fixed by SAMViTConfig): image_size 1024, patch_size 16,
//  embed_dim 768, depth 12, heads 12, window_size 14,
//  global_attn_indexes {2,5,8,11}, neck out_chans 256, then net_2 (256->512)
//  and net_3 (512->1024) each stride-2 — total 16x downsample of the 64x64
//  patch grid down to a 16x16 (final_out_chans 1024) feature map in NHWC.
//
//  All spatial tensors flow in NHWC (B, H, W, C) to match MLX Conv2d and the
//  Python reference exactly.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Utility functions (1:1 with sam.py)

/// Interpolate absolute positional embeddings to target size.
/// abs_pos: (1, src, src, C). When src == tgt this is the identity (the only
/// case that occurs for the fixed 1024/16 = 64 geometry, so we keep it simple
/// and assert the identity branch). NOTE: bicubic interpolation branch from
/// sam.py is omitted because tgt_size always equals src_size here; if a
/// different image size is ever used this must grow a bicubic path.
private func getAbsPosSAM(_ absPos: MLXArray, tgtSize: Int) -> MLXArray {
    // absPos: (1, src, src, C). The SAM pos embed is the 64x64 grid for the
    // 1024px global view (1024/16). The 640px CROP tiles produce a 40x40 patch
    // grid, so src(64) != tgt(40) and the embedding MUST be bicubically resized
    // — matching deepencoder.py `get_abs_pos_sam`:
    //   permute(0,3,1,2) -> float32 -> F.interpolate(bicubic, antialias,
    //   align_corners=False) -> permute(0,2,3,1).
    // Returning identity here (the prior behavior) left crop tiles with the
    // wrong positional encoding (and a shape mismatch on the add), collapsing
    // multi-tile OCR.
    let srcSize = absPos.dim(1)
    if srcSize == tgtSize {
        return absPos
    }
    let dtype = absPos.dtype
    // (1, src, src, C) -> (1, C, src, src)
    var p = absPos.transposed(0, 3, 1, 2).asType(.float32)
    p = bicubicInterpolate(p, size: (tgtSize, tgtSize))
    // (1, C, tgt, tgt) -> (1, tgt, tgt, C)
    return p.asType(dtype).transposed(0, 2, 3, 1)
}

/// Partition (B, H, W, C) into non-overlapping windows with padding if needed.
/// Returns the windows (B*nW, win, win, C) and the padded (Hp, Wp).
private func windowPartition(_ xIn: MLXArray, windowSize: Int) -> (MLXArray, (Int, Int)) {
    var x = xIn
    let (B, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))

    let padH = (windowSize - H % windowSize) % windowSize
    let padW = (windowSize - W % windowSize) % windowSize

    if padH > 0 || padW > 0 {
        x = padded(x, widths: [[0, 0], [0, padH], [0, padW], [0, 0]])
    }
    let Hp = H + padH
    let Wp = W + padW

    x = x.reshaped(B, Hp / windowSize, windowSize, Wp / windowSize, windowSize, C)
    let windows = x.transposed(0, 1, 3, 2, 4, 5).reshaped(-1, windowSize, windowSize, C)
    return (windows, (Hp, Wp))
}

/// Reverse of windowPartition: stitch windows back to (B, H, W, C), dropping padding.
private func windowUnpartition(
    _ windows: MLXArray, windowSize: Int, padHW: (Int, Int), hw: (Int, Int)
) -> MLXArray {
    let (Hp, Wp) = padHW
    let (H, W) = hw
    let B = windows.dim(0) / (Hp * Wp / windowSize / windowSize)

    var x = windows.reshaped(
        B, Hp / windowSize, Wp / windowSize, windowSize, windowSize, -1)
    x = x.transposed(0, 1, 3, 2, 4, 5).reshaped(B, Hp, Wp, -1)

    if Hp > H || Wp > W {
        x = x[0..., ..<H, ..<W, 0...]
    }
    return x
}

/// Get relative positional embeddings for the given query / key sizes.
/// rel_pos: (L, C). When L already equals 2*max(q,k)-1 (true for the fixed
/// geometry, where q==k==window_size or q==k==64) this just builds the gather
/// index. NOTE: the linear-interpolation resize branch from sam.py is included
/// for fidelity but is not hit at the fixed DeepSeek-OCR geometry.
private func getRelPos(qSize: Int, kSize: Int, relPos: MLXArray) -> MLXArray {
    let maxRelDist = 2 * max(qSize, kSize) - 1

    var relPosResized: MLXArray
    if relPos.dim(0) != maxRelDist {
        let dtype = relPos.dtype
        var rp = relPos.asType(.float32)
        // (1, L, C) -> (1, C, L)
        rp = rp.reshaped(1, relPos.dim(0), -1).transposed(0, 2, 1)
        let scale = Float(rp.dim(2)) / Float(maxRelDist)
        let indices = MLXArray(0 ..< maxRelDist).asType(.float32) * scale
        let idxFloor = floor(indices).asType(.int32)
        let idxCeil = minimum(idxFloor + 1, rp.dim(2) - 1)
        let weight = indices - idxFloor.asType(.float32)

        let gatheredFloor = rp.take(idxFloor, axis: 2)
        let gatheredCeil = rp.take(idxCeil, axis: 2)
        rp = (gatheredFloor * (1 - weight) + gatheredCeil * weight).asType(dtype)
        relPosResized = rp.reshaped(-1, maxRelDist).transposed(1, 0)
    } else {
        relPosResized = relPos
    }

    // Scale the coords with short length if q and k differ.
    let qScale = max(Float(kSize) / Float(qSize), 1.0)
    let kScale = max(Float(qSize) / Float(kSize), 1.0)
    let qCoords = MLXArray(0 ..< qSize).asType(.float32).reshaped(qSize, 1) * qScale
    let kCoords = MLXArray(0 ..< kSize).asType(.float32).reshaped(1, kSize) * kScale
    let relativeCoords = (qCoords - kCoords) + Float(kSize - 1) * kScale

    return relPosResized[relativeCoords.asType(.int32)]
}

/// Decomposed relative position bias (rel_h, rel_w) as in sam.py.
/// q: (B, q_h*q_w, C). Returns (rel_h: (B, q_h*q_w, k_h, 1), rel_w: (B, q_h*q_w, 1, k_w)).
private func addDecomposedRelPos(
    _ q: MLXArray, relPosH: MLXArray, relPosW: MLXArray,
    qSize: (Int, Int), kSize: (Int, Int)
) -> (MLXArray, MLXArray) {
    let (qH, qW) = qSize
    let (kH, kW) = kSize

    let Rh = getRelPos(qSize: qH, kSize: kH, relPos: relPosH)
    let Rw = getRelPos(qSize: qW, kSize: kW, relPos: relPosW)

    let B = q.dim(0)
    let dim = q.dim(2)
    let rq = q.reshaped(B, qH, qW, dim)

    var relH = einsum("bhwc,hkc->bhwk", rq, Rh)
    var relW = einsum("bhwc,wkc->bhwk", rq, Rw)
    relH = relH[.ellipsis, .newAxis]          // (B, qH, qW, kH, 1)
    relW = relW[.ellipsis, .newAxis, 0...]    // (B, qH, qW, 1, kW)
    relH = relH.reshaped(B, qH * qW, kH, 1)
    relW = relW.reshaped(B, qH * qW, 1, kW)
    return (relH, relW)
}

// MARK: - MLP block

private class MLPBlock: Module, UnaryLayer {
    @ModuleInfo var lin1: Linear
    @ModuleInfo var lin2: Linear
    let act: GELU

    init(embeddingDim: Int, mlpDim: Int) {
        self._lin1.wrappedValue = Linear(embeddingDim, mlpDim)
        self._lin2.wrappedValue = Linear(mlpDim, embeddingDim)
        self.act = GELU()
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        lin2(act(lin1(x)))
    }
}

// MARK: - Attention with relative position embeddings

private class Attention: Module {
    let numHeads: Int
    let scale: Float
    let useRelPos: Bool

    @ModuleInfo var qkv: Linear
    @ModuleInfo var proj: Linear

    @ParameterInfo(key: "rel_pos_h") var relPosH: MLXArray
    @ParameterInfo(key: "rel_pos_w") var relPosW: MLXArray

    init(
        dim: Int, numHeads: Int = 8, qkvBias: Bool = true,
        useRelPos: Bool = false, inputSize: (Int, Int)? = nil
    ) {
        self.numHeads = numHeads
        let headDim = dim / numHeads
        self.scale = pow(Float(headDim), -0.5)
        self.useRelPos = useRelPos

        self._qkv.wrappedValue = Linear(dim, dim * 3, bias: qkvBias)
        self._proj.wrappedValue = Linear(dim, dim)

        if useRelPos {
            precondition(
                inputSize != nil,
                "Input size must be provided if using relative positional encoding.")
            let (ih, iw) = inputSize!
            self._relPosH.wrappedValue = MLXArray.zeros([2 * ih - 1, headDim])
            self._relPosW.wrappedValue = MLXArray.zeros([2 * iw - 1, headDim])
        } else {
            // Unused, but the property must be initialized.
            self._relPosH.wrappedValue = MLXArray.zeros([1, headDim])
            self._relPosW.wrappedValue = MLXArray.zeros([1, headDim])
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, H, W) = (x.dim(0), x.dim(1), x.dim(2))

        // (B, H*W, 3, nHeads, headDim) -> (3, B, nHeads, H*W, headDim)
        var qkvOut = qkv(x)
            .reshaped(B, H * W, 3, numHeads, -1)
            .transposed(2, 0, 3, 1, 4)

        // (3, B*nHeads, H*W, headDim)
        qkvOut = qkvOut.reshaped(3, B * numHeads, H * W, -1)
        let q = qkvOut[0]
        let k = qkvOut[1]
        let v = qkvOut[2]

        let qH = q.reshaped(B, numHeads, H * W, -1)
        let kH = k.reshaped(B, numHeads, H * W, -1)
        let vH = v.reshaped(B, numHeads, H * W, -1)

        var out: MLXArray
        if useRelPos {
            let (relH, relW) = addDecomposedRelPos(
                q, relPosH: relPosH, relPosW: relPosW, qSize: (H, W), kSize: (H, W))
            // relH: (B*nHeads, H*W, kH, 1), relW: (B*nHeads, H*W, 1, kW)
            let relH5 = relH.reshaped(B, numHeads, relH.dim(1), relH.dim(2), relH.dim(3))
            let relW5 = relW.reshaped(B, numHeads, relW.dim(1), relW.dim(2), relW.dim(3))
            // attn_bias: (B, nHeads, H*W, kH*kW)
            let attnBias = (relH5 + relW5).reshaped(
                B, numHeads, relH5.dim(2), relH5.dim(3) * relW5.dim(4))
            out = MLXFast.scaledDotProductAttention(
                queries: qH, keys: kH, values: vH, scale: scale, mask: .array(attnBias))
        } else {
            out = MLXFast.scaledDotProductAttention(
                queries: qH, keys: kH, values: vH, scale: scale, mask: .none)
        }

        // (B, nHeads, H, W, headDim) -> (B, H, W, C)
        out = out.reshaped(B, numHeads, H, W, -1)
            .transposed(0, 2, 3, 1, 4)
            .reshaped(B, H, W, -1)
        return proj(out)
    }
}

// MARK: - Transformer block (window or global attention)

private class Block: Module {
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var attn: Attention
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var mlp: MLPBlock

    let windowSize: Int

    init(
        dim: Int, numHeads: Int, mlpRatio: Float = 4.0, qkvBias: Bool = true,
        useRelPos: Bool = false, windowSize: Int = 0, inputSize: (Int, Int)? = nil
    ) {
        self._norm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._attn.wrappedValue = Attention(
            dim: dim, numHeads: numHeads, qkvBias: qkvBias, useRelPos: useRelPos,
            inputSize: windowSize == 0 ? inputSize : (windowSize, windowSize))
        self._norm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._mlp.wrappedValue = MLPBlock(embeddingDim: dim, mlpDim: Int(Float(dim) * mlpRatio))
        self.windowSize = windowSize
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shortcut = x
        var h = norm1(x)

        var padHW: (Int, Int) = (0, 0)
        var origHW: (Int, Int) = (0, 0)
        if windowSize > 0 {
            origHW = (h.dim(1), h.dim(2))
            let (windows, pad) = windowPartition(h, windowSize: windowSize)
            h = windows
            padHW = pad
        }

        h = attn(h)

        if windowSize > 0 {
            h = windowUnpartition(h, windowSize: windowSize, padHW: padHW, hw: origHW)
        }

        let x1 = shortcut + h
        return x1 + mlp(norm2(x1))
    }
}

// MARK: - Patch embedding

private class PatchEmbed: Module {
    @ModuleInfo var proj: Conv2d

    init(
        kernelSize: Int = 16, stride: Int = 16, inChans: Int = 3, embedDim: Int = 768
    ) {
        self._proj.wrappedValue = Conv2d(
            inputChannels: inChans, outputChannels: embedDim,
            kernelSize: IntOrPair(kernelSize), stride: IntOrPair(stride))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        proj(x)
    }
}

// MARK: - LayerNorm2d (channel-axis norm in NHWC)

/// Channel-wise LayerNorm matching deepencoder.py `LayerNorm2d`.
///
/// The official PyTorch code runs the neck in NCHW and normalizes over axis 1
/// (channels) — `u = x.mean(1)`, then scales by `weight[:,None,None]`. Our SAM
/// tower flows in NHWC, where the channel axis is the LAST axis, so a normal
/// channel-last LayerNorm reduction is numerically identical. We keep an
/// explicit small module (rather than reusing MLXNN.LayerNorm) so the
/// `weight`/`bias` parameter keys resolve directly under the neck container
/// (`neck.1.weight`, `neck.1.bias`, …) and the reduction axis is unambiguous.
private class LayerNorm2dNHWC: Module, UnaryLayer {
    var weight: MLXArray
    var bias: MLXArray
    let eps: Float

    init(_ numChannels: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([numChannels])
        self.bias = MLXArray.zeros([numChannels])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, H, W, C) NHWC — normalize over the channel (last) axis.
        let u = mean(x, axis: -1, keepDims: true)
        let d = x - u
        let s = mean(d * d, axis: -1, keepDims: true)
        let normed = d * rsqrt(s + eps)
        return weight * normed + bias
    }
}

// MARK: - SAM encoder

/// SAM-ViT-B encoder. Wrapped as `sam_model` by the top model so weight keys
/// resolve to `sam_model.patch_embed.*`, `sam_model.pos_embed`,
/// `sam_model.blocks.N.*`, `sam_model.neck.*`, `sam_model.net_2`,
/// `sam_model.net_3`.
public class DeepseekOCRSAMEncoder: Module {

    let imgSize: Int
    let useAbsPos: Bool

    // NOTE: compile-fix — fileprivate so these private-typed members can live in a public class.
    @ModuleInfo(key: "patch_embed") fileprivate var patchEmbed: PatchEmbed
    @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray
    @ModuleInfo fileprivate var blocks: [Block]

    // The neck is a heterogeneous Sequential (Conv2d, LayerNorm2d, Conv2d,
    // LayerNorm2d) in sam.py, serialized with numeric paths (`neck.0.weight`,
    // `neck.1.weight`, …). An @ModuleInfo array keyed "neck" produces exactly
    // those `neck.<index>.*` paths and the checkpoint's array structure lines
    // up element-for-element. Conv2d and LayerNorm2dNHWC both conform to
    // UnaryLayer, so the heterogeneous stack fits one array.
    @ModuleInfo(key: "neck") fileprivate var neck: [UnaryLayer]

    @ModuleInfo(key: "net_2") var net2: Conv2d     // 3x3 stride-2, 256 -> 512
    @ModuleInfo(key: "net_3") var net3: Conv2d     // 3x3 stride-2, 512 -> final_out_chans

    public convenience init(_ config: DeepseekOCRConfiguration.SAMViTConfiguration) {
        self.init(
            imgSize: config.imageSize,
            patchSize: config.patchSize,
            embedDim: config.width,
            depth: config.layers,
            numHeads: config.heads,
            outChans: config.promptEmbedDim,
            windowSize: config.windowSize,
            globalAttnIndexes: config.globalAttnIndexes,
            finalOutChans: config.downsampleChannels.last ?? 1024)
    }

    public init(
        imgSize: Int = 1024,
        patchSize: Int = 16,
        inChans: Int = 3,
        embedDim: Int = 768,
        depth: Int = 12,
        numHeads: Int = 12,
        mlpRatio: Float = 4.0,
        outChans: Int = 256,
        qkvBias: Bool = true,
        useAbsPos: Bool = true,
        useRelPos: Bool = true,
        windowSize: Int = 14,
        globalAttnIndexes: [Int] = [2, 5, 8, 11],
        finalOutChans: Int = 1024
    ) {
        self.imgSize = imgSize
        self.useAbsPos = useAbsPos

        self._patchEmbed.wrappedValue = PatchEmbed(
            kernelSize: patchSize, stride: patchSize, inChans: inChans, embedDim: embedDim)

        let grid = imgSize / patchSize
        // (1, grid, grid, embedDim)
        self._posEmbed.wrappedValue = MLXArray.zeros([1, grid, grid, embedDim])

        let globalSet = Set(globalAttnIndexes)
        self._blocks.wrappedValue = (0 ..< depth).map { i in
            Block(
                dim: embedDim, numHeads: numHeads, mlpRatio: mlpRatio, qkvBias: qkvBias,
                useRelPos: useRelPos,
                windowSize: globalSet.contains(i) ? 0 : windowSize,
                inputSize: (grid, grid))
        }

        // Neck: matches nn.Conv2d/LayerNorm2d Sequential in sam.py (4 elements).
        self._neck.wrappedValue = [
            Conv2d(
                inputChannels: embedDim, outputChannels: outChans,
                kernelSize: IntOrPair(1), bias: false),                 // neck.0
            LayerNorm2dNHWC(outChans),                                  // neck.1
            Conv2d(
                inputChannels: outChans, outputChannels: outChans,
                kernelSize: IntOrPair(3), padding: IntOrPair(1), bias: false),  // neck.2
            LayerNorm2dNHWC(outChans),                                  // neck.3
        ]

        self._net2.wrappedValue = Conv2d(
            inputChannels: 256, outputChannels: 512,
            kernelSize: IntOrPair(3), stride: IntOrPair(2), padding: IntOrPair(1), bias: false)
        self._net3.wrappedValue = Conv2d(
            inputChannels: 512, outputChannels: finalOutChans,
            kernelSize: IntOrPair(3), stride: IntOrPair(2), padding: IntOrPair(1), bias: false)

        super.init()
    }

    /// x: (B, 1024, 1024, 3) NHWC. Returns the (B, 16, 16, final_out_chans)
    /// spatial feature map (NHWC), exactly as sam.py returns it.
    public func callAsFunction(_ xIn: MLXArray) -> MLXArray {
        // Patch embed -> (B, 64, 64, embedDim) NHWC
        var x = patchEmbed(xIn)

        if useAbsPos {
            x = x + getAbsPosSAM(posEmbed, tgtSize: x.dim(1))
        }

        for blk in blocks {
            x = blk(x)
        }

        // Neck: Conv2d operate in NHWC; LayerNorm2dNHWC normalizes the last
        // (channel) axis, matching sam.py where the neck LayerNorm2d acts over
        // channels (axis 1 in its NCHW layout == last axis here in NHWC).
        for layer in neck { x = layer(x) }

        // Additional 2x + 2x downsampling -> (B, 16, 16, final_out_chans)
        x = net2(x)
        x = net3(x)

        return x
    }

    /// Convert PyTorch conv weights ([out, in, kH, kW]) to MLX NHWC layout
    /// ([out, kH, kW, in]). The top model invokes this for its `sam_model.*`
    /// slice. Conv weight keys: patch_embed.proj.weight, neck.0/neck.2.weight,
    /// net_2.weight, net_3.weight. `pos_embed` and `rel_pos_*` ship already in
    /// the expected layout and are passed through.
    public static func sanitize(weights: [String: MLXArray], prefix: String = "sam_model.")
        -> [String: MLXArray]
    {
        var out = weights
        let convWeightSuffixes = [
            "patch_embed.proj.weight",
            "neck.0.weight",
            "neck.2.weight",
            "net_2.weight",
            "net_3.weight",
        ]
        for (k, v) in weights {
            guard k.hasPrefix(prefix), v.ndim == 4 else { continue }
            if convWeightSuffixes.contains(where: { k.hasSuffix($0) }) {
                // Skip if already NHWC (out, kH, kW, in) with square kernel.
                let s = v.shape
                let alreadyMLX = (s[0] >= s[1]) && (s[0] >= s[2]) && (s[1] == s[2])
                out[k] = alreadyMLX ? v : v.transposed(0, 2, 3, 1)
            }
        }
        return out
    }
}
