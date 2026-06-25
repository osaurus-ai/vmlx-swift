//
//  DeepseekOCRVision.swift
//  mlx-swift-lm
//
//  CLIP-L/14-224 vision tower for the DeepSeek-OCR DeepEncoder.
//
//  Faithful port of
//  https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/deepseekocr (vision.py)
//
//  Architecture: CLIP-L/14-224 — width 1024, layers 24, heads 16,
//  patch_size 14, image_size 224, layer_norm_eps 1e-6.
//
//  This branch is the `vision_model` sub-module of DeepseekOCRForCausalLM.
//  Its distinguishing feature vs. a stock CLIP ViT is the `patchEmbeds`
//  fusion: when the SAM encoder's patch tokens are supplied, they are used
//  in place of running the Conv2d patch embedding (the SAM tower already
//  produced the per-patch features); otherwise the Conv2d patch embedding
//  runs on the raw NHWC image.
//
//  Weight keys (after the top model's `sanitize`, the prefix is
//  `vision_model.`, matching `model.vision_model.*` in the safetensors):
//    vision_model.embeddings.class_embedding
//    vision_model.embeddings.patch_embedding.weight
//    vision_model.embeddings.position_embedding.weight
//    vision_model.pre_layrnorm.{weight,bias}            (note: "layrnorm")
//    vision_model.transformer.layers.{i}.layer_norm1.{weight,bias}
//    vision_model.transformer.layers.{i}.layer_norm2.{weight,bias}
//    vision_model.transformer.layers.{i}.mlp.fc1.{weight,bias}
//    vision_model.transformer.layers.{i}.mlp.fc2.{weight,bias}
//    vision_model.transformer.layers.{i}.self_attn.qkv_proj.{weight,bias}
//    vision_model.transformer.layers.{i}.self_attn.out_proj.{weight,bias}
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Attention (fused QKV)

/// CLIP-style attention with a single fused `qkv_proj` (`dims -> dims * 3`)
/// and `out_proj`. Mirrors vision.py `Attention`.
private class DeepseekOCRVisionAttention: Module {
    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let numHeads: Int
    let scale: Float

    init(dims: Int, numHeads: Int, qkvBias: Bool = true) {
        if dims % numHeads != 0 {
            fatalError(
                "The input feature dimensions should be divisible by the number of heads "
                    + "(\(dims) % \(numHeads)) != 0")
        }
        self.numHeads = numHeads
        let headDim = dims / numHeads
        self.scale = pow(Float(headDim), -0.5)

        self._qkvProj.wrappedValue = Linear(dims, dims * 3, bias: qkvBias)
        self._outProj.wrappedValue = Linear(dims, dims, bias: true)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none)
        -> MLXArray
    {
        let qkv = qkvProj(x)
        let parts = split(qkv, parts: 3, axis: -1)
        var queries = parts[0]
        var keys = parts[1]
        var values = parts[2]

        let (B, L) = (queries.dim(0), queries.dim(1))
        let S = keys.dim(1)

        queries = queries.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, S, numHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, S, numHeads, -1).transposed(0, 2, 1, 3)

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return outProj(output)
    }
}

// MARK: - MLP

/// Mirrors vision.py `MLP`: fc1 -> GELU -> fc2, with bias.
private class DeepseekOCRVisionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    let activationFn: GELU

    init(config: DeepseekOCRConfiguration.VisionConfiguration, bias: Bool = true) {
        // intermediate_size = round(width * mlp_ratio) — CLIP-L: round(1024 * 3.7362) = 3826.
        let intermediateSize = Int((Float(config.width) * config.mlpRatio).rounded())
        self.activationFn = GELU()
        self._fc1.wrappedValue = Linear(config.width, intermediateSize, bias: bias)
        self._fc2.wrappedValue = Linear(intermediateSize, config.width, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(activationFn(fc1(x)))
    }
}

// MARK: - Encoder layer

/// Pre-norm transformer block. Mirrors vision.py `EncoderLayer`.
private class DeepseekOCREncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DeepseekOCRVisionAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo var mlp: DeepseekOCRVisionMLP
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

    init(config: DeepseekOCRConfiguration.VisionConfiguration) {
        let embedDim = config.width
        self._selfAttn.wrappedValue = DeepseekOCRVisionAttention(
            dims: config.width, numHeads: config.heads, qkvBias: true)
        self._layerNorm1.wrappedValue = LayerNorm(dimensions: embedDim, eps: config.layerNormEps)
        self.mlp = DeepseekOCRVisionMLP(config: config)
        self._layerNorm2.wrappedValue = LayerNorm(dimensions: embedDim, eps: config.layerNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none)
        -> MLXArray
    {
        var y = layerNorm1(x)
        y = selfAttn(y, mask: mask)
        let h = x + y
        let r = mlp(layerNorm2(h))
        return h + r
    }
}

// MARK: - Transformer

/// Mirrors vision.py `NoTPTransformer`: a plain stack of encoder layers,
/// returning the final layer's output (no layer selection).
private class DeepseekOCRTransformer: Module {
    @ModuleInfo var layers: [DeepseekOCREncoderLayer]

    init(config: DeepseekOCRConfiguration.VisionConfiguration) {
        self._layers.wrappedValue = (0 ..< config.layers).map { _ in
            DeepseekOCREncoderLayer(config: config)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h, mask: .none)
        }
        return h
    }
}

// MARK: - Embeddings (class token + patch embed + position embed)

/// Mirrors vision.py `VisionEmbeddings`.
private class DeepseekOCRVisionEmbeddings: Module {
    @ParameterInfo(key: "class_embedding") var classEmbedding: MLXArray
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    let embedDim: Int
    let imageSize: Int
    let patchSize: Int
    let numPatches: Int
    let numPositions: Int

    init(config: DeepseekOCRConfiguration.VisionConfiguration) {
        self.embedDim = config.width
        // NOTE: image_size is hard-coded to 224 in vision.py (not read from config).
        self.imageSize = 224
        self.patchSize = config.patchSize

        // class_embedding is a raw learned parameter of shape (embed_dim,),
        // initialized random in the reference; loaded from weights at runtime.
        self._classEmbedding.wrappedValue = MLXArray.zeros([embedDim])

        self._patchEmbedding.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: embedDim,
            kernelSize: IntOrPair(patchSize),
            stride: IntOrPair(patchSize),
            bias: false
        )

        self.numPatches = (imageSize / patchSize) * (imageSize / patchSize)
        self.numPositions = numPatches + 1
        self._positionEmbedding.wrappedValue = Embedding(
            embeddingCount: numPositions, dimensions: embedDim)
    }

    /// Mirrors vision.py `_get_abs_pos`: resize absolute positional embeddings
    /// to match the current token grid. For the fixed 224/14 geometry the
    /// source grid (16x16) equals the target grid, so the identity branch is
    /// taken and `absPos` is returned unchanged. The resize branch is ported
    /// for fidelity (bicubic, matching the reference's `interpolate` default).
    private func getAbsPos(_ absPos: MLXArray, tgtSize: Int) -> MLXArray {
        let dim = absPos.dim(-1)
        let absPosNew = squeezed(absPos, axis: 0)  // (L, C)
        let total = absPosNew.dim(0)
        let clsToken = absPosNew[0 ..< 1]
        let oldPosEmbed = absPosNew[1...]
        let srcSize = Int(Double(total - 1).squareRoot())
        let tgtSize2D = Int(Double(tgtSize).squareRoot())
        let dtype = absPos.dtype

        if srcSize != tgtSize2D {
            // (1, src, src, dim) -> (1, dim, src, src)
            var resized =
                oldPosEmbed
                .reshaped(1, srcSize, srcSize, dim)
                .transposed(0, 3, 1, 2)
                .asType(.float32)

            resized = bicubicInterpolate(resized, size: (tgtSize2D, tgtSize2D))

            // (1, dim, tgt, tgt) -> (tgt*tgt, dim)
            var newPosEmbed =
                resized
                .asType(dtype)
                .transposed(0, 2, 3, 1)
                .reshaped(tgtSize2D * tgtSize2D, dim)

            newPosEmbed = concatenated([clsToken, newPosEmbed], axis: 0)
            return newPosEmbed.reshaped(1, tgtSize2D * tgtSize2D + 1, dim)
        } else {
            return absPos
        }
    }

    /// `x` is NHWC (B, H, W, 3). `patchEmbeds` is the SAM encoder output; when
    /// supplied it replaces the Conv2d patch embedding (already NHWC-laid-out
    /// per-patch features). Returns (B, seq, embed_dim) with the CLS token at
    /// position 0.
    func callAsFunction(_ x: MLXArray, patchEmbeds: MLXArray? = nil) -> MLXArray {
        let batchSize = x.dim(0)
        let targetDtype = positionEmbedding.weight.dtype

        let patchEmbeddings: MLXArray
        if let patchEmbeds {
            patchEmbeddings = patchEmbeds
        } else {
            patchEmbeddings = patchEmbedding(x)
        }

        // Flatten spatial dims: (B, H', W', C) -> (B, H'*W', C)
        let flatPatches = flattened(patchEmbeddings, start: 1, end: 2)

        // Broadcast class embedding to (B, 1, embed_dim)
        let classEmbeds = broadcast(
            classEmbedding, to: [batchSize, 1, embedDim]
        ).asType(targetDtype)

        var embeddings = concatenated([classEmbeds, flatPatches], axis: 1)

        // Position IDs 0..<numPositions
        let positionIds = MLXArray(Array(0 ..< numPositions))[.newAxis, 0...]
        let absPos = getAbsPos(positionEmbedding(positionIds), tgtSize: embeddings.dim(1))
        embeddings = embeddings + absPos.asType(targetDtype)

        return embeddings
    }
}

// MARK: - Vision model

/// CLIP-L/14 vision tower (the `vision_model` branch of the DeepEncoder).
/// Mirrors vision.py `VisionModel`.
public class DeepseekOCRVisionModel: Module {
    @ModuleInfo var embeddings: DeepseekOCRVisionEmbeddings
    @ModuleInfo(key: "pre_layrnorm") var preLayerNorm: LayerNorm
    @ModuleInfo var transformer: DeepseekOCRTransformer

    public init(_ config: DeepseekOCRConfiguration.VisionConfiguration) {
        self.embeddings = DeepseekOCRVisionEmbeddings(config: config)
        // NOTE: pre_layrnorm uses the default LayerNorm eps (1e-5) in the
        // reference (nn.LayerNorm(hidden_size) — no eps passed), unlike the
        // encoder-layer norms which use config.layer_norm_eps (1e-6).
        self._preLayerNorm.wrappedValue = LayerNorm(dimensions: config.width)
        self.transformer = DeepseekOCRTransformer(config: config)
        super.init()
    }

    /// `x`: NHWC image (B, H, W, 3) — the caller passes `image.transpose(0,2,3,1)`.
    /// `patchEmbeds`: the SAM encoder output, fused into the patch tokens.
    /// Returns (B, seq, width); the top model selects `result[:, 1:]` to drop CLS.
    public func callAsFunction(_ x: MLXArray, patchEmbeds: MLXArray) -> MLXArray {
        var h = embeddings(x, patchEmbeds: patchEmbeds)
        h = preLayerNorm(h)
        return transformer(h)
    }
}
