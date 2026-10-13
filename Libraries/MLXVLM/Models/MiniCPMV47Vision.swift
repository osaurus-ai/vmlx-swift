// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// MiniCPM-V 4.7 vision path: a NaViT-style SigLIP tower (27 layers, hidden 1152, patch 14, learned 70x70
// position table resampled by nearest bucket), a 2x2 window-attention merger spliced in after encoder layer
// `insert_layer_id` (halves each patch grid), a post layer norm, and the top-level `merger` MLP that folds another
// 2x2 of merged patches into one LLM embedding — 16 pixels patches per LLM token in `downsample_mode = 16x`.
// Reference: transformers 5.18 `models/minicpmv4_7/modeling_minicpmv4_7.py` (4.7 overrides the window merger to
// keep window order, so the output is the merged grid in row-major order without an argsort back).
//
// Canvas M-RoPE (`mrope_mode = canvas`): each picture is a thumbnail stretched over a canvas plus optional
// detail slices tiling that canvas; the `<image>`/`</image>`, `<slice>`/`</slice>` and row newline markers are
// pinned to canvas corners and edges. Ported from `MiniCPMV4_7Model.get_rope_index` token for token.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct MiniCPMV47VisionConfiguration: Codable, Sendable {
    public var hiddenSize: Int = 1152
    public var intermediateSize: Int = 4304
    public var numHiddenLayers: Int = 27
    public var numAttentionHeads: Int = 16
    public var numChannels: Int = 3
    public var imageSize: Int = 980
    public var patchSize: Int = 14
    public var hiddenAct: String = "gelu_pytorch_tanh"
    public var layerNormEps: Float = 1e-6
    public var windowKernelSize: [Int] = [2, 2]

    public var numPatchesPerSide: Int { imageSize / patchSize }
    public var headDim: Int { hiddenSize / numAttentionHeads }
    public var windowHiddenSize: Int { hiddenSize * windowKernelSize[0] * windowKernelSize[1] }
    public var windowIntermediateSize: Int { intermediateSize * windowKernelSize[0] * windowKernelSize[1] }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numChannels = "num_channels"
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case hiddenAct = "hidden_act"
        case layerNormEps = "layer_norm_eps"
        case windowKernelSize = "window_kernel_size"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4304
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 27
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numChannels = try c.decodeIfPresent(Int.self, forKey: .numChannels) ?? 3
        imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 980
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu_pytorch_tanh"
        layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
        windowKernelSize = try c.decodeIfPresent([Int].self, forKey: .windowKernelSize) ?? [2, 2]
    }
}

/// Top-level `config.json` fields that drive the merger and the canvas layout. The marker ids are resolved by
/// the conversion script and stored in the config; they are required for image input.
public struct MiniCPMV47CanvasConfiguration: Codable, Sendable, Equatable {
    public var insertLayerId: Int = 6
    public var downsampleMode: String = "16x"
    public var mropeMode: String = "canvas"
    public var mergeKernelSize: [Int] = [2, 2]
    public var mergerTimes: Int = 1
    public var imageStartId: Int?
    public var imageEndId: Int?
    public var sliceStartId: Int?
    public var sliceEndId: Int?
    public var newlineId: Int?

    /// Patch-to-LLM-token merge factor per axis: 4 in `16x` (window merger 2x2 then merger 2x2), 2 in `4x`.
    public var mergeFactor: Int { downsampleMode == "4x" ? 2 : 4 }
    public var tokensPerPatchDivisor: Int { mergeFactor * mergeFactor }
    public var usesViTMerger: Bool { downsampleMode != "4x" }

    enum CodingKeys: String, CodingKey {
        case insertLayerId = "insert_layer_id"
        case downsampleMode = "downsample_mode"
        case mropeMode = "mrope_mode"
        case mergeKernelSize = "merge_kernel_size"
        case mergerTimes = "merger_times"
        case imageStartId = "image_start_id"
        case imageEndId = "image_end_id"
        case sliceStartId = "slice_start_id"
        case sliceEndId = "slice_end_id"
        case newlineId = "newline_id"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        insertLayerId = try c.decodeIfPresent(Int.self, forKey: .insertLayerId) ?? 6
        downsampleMode = try c.decodeIfPresent(String.self, forKey: .downsampleMode) ?? "16x"
        mropeMode = try c.decodeIfPresent(String.self, forKey: .mropeMode) ?? "canvas"
        mergeKernelSize = try c.decodeIfPresent([Int].self, forKey: .mergeKernelSize) ?? [2, 2]
        mergerTimes = try c.decodeIfPresent(Int.self, forKey: .mergerTimes) ?? 1
        imageStartId = try c.decodeIfPresent(Int.self, forKey: .imageStartId)
        imageEndId = try c.decodeIfPresent(Int.self, forKey: .imageEndId)
        sliceStartId = try c.decodeIfPresent(Int.self, forKey: .sliceStartId)
        sliceEndId = try c.decodeIfPresent(Int.self, forKey: .sliceEndId)
        newlineId = try c.decodeIfPresent(Int.self, forKey: .newlineId)
    }

    /// All five marker ids, or nil when the bundle did not resolve them (then images cannot be laid out).
    public var markerIds: (imageStart: Int, imageEnd: Int, sliceStart: Int, sliceEnd: Int, newline: Int)? {
        guard let imageStartId, let imageEndId, let sliceStartId, let sliceEndId, let newlineId else {
            return nil
        }
        return (imageStartId, imageEndId, sliceStartId, sliceEndId, newlineId)
    }
}

// MARK: - Tower

enum MiniCPMV47Vision {

    /// Nearest-bucket resampling of the learned `numPatchesPerSide^2` position table onto an `h x w` patch
    /// grid, exactly as `get_vision_nearest_position_ids`: `bucketize(arange(n)/n, arange(1/S, 1, 1/S),
    /// right=True)` in float32.
    static func nearestPositionIds(height: Int, width: Int, numPatchesPerSide side: Int) -> [Int32] {
        // torch.arange(1/S, 1.0, 1/S) sizes itself as ceil((end - start) / step) in double and evaluates
        // start + i*step in double before narrowing to float32.
        let step = 1.0 / Double(side)
        let count = Int(ceil((1.0 - step) / step))
        let boundaries: [Float] = (0 ..< count).map { Float(step + Double($0) * step) }
        func bucket(_ i: Int, _ n: Int) -> Int32 {
            let coord = Float(i) / Float(n)
            var b: Int32 = 0
            for boundary in boundaries where boundary <= coord { b += 1 }
            return b
        }
        let rows = (0 ..< height).map { bucket($0, height) }
        let cols = (0 ..< width).map { bucket($0, width) }
        var out: [Int32] = []
        out.reserveCapacity(height * width)
        for r in rows { for c in cols { out.append(r * Int32(side) + c) } }
        return out
    }

    /// Window-attention gather order for a `(h, w)` patch grid with a `k x k` window: windows in row-major
    /// order, patches inside a window in row-major order. Grids are multiples of `k` (the processor rounds
    /// every crop to a multiple of `patch * 4`), so no padding slot survives.
    static func windowIndex(height: Int, width: Int, window k: Int, offset: Int) -> [Int32] {
        var index: [Int32] = []
        index.reserveCapacity(height * width)
        for wi in 0 ..< height / k {
            for wj in 0 ..< width / k {
                for r in 0 ..< k {
                    for c in 0 ..< k {
                        index.append(Int32(offset + (wi * k + r) * width + (wj * k + c)))
                    }
                }
            }
        }
        return index
    }

    final class Embeddings: Module {
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
        let numPatchesPerSide: Int
        let patchSize: Int

        init(_ config: MiniCPMV47VisionConfiguration) {
            numPatchesPerSide = config.numPatchesPerSide
            patchSize = config.patchSize
            _patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.numChannels, outputChannels: config.hiddenSize,
                kernelSize: IntOrPair(config.patchSize), stride: IntOrPair(config.patchSize), bias: true)
            _positionEmbedding.wrappedValue = Embedding(
                embeddingCount: config.numPatchesPerSide * config.numPatchesPerSide,
                dimensions: config.hiddenSize)
        }

        /// - Parameter patches: `[N, patch, patch, C]` pixel patches, crop after crop in row-major order.
        func callAsFunction(_ patches: MLXArray, grids: [THW]) -> MLXArray {
            var embeds = patchEmbedding(patches)  // [N, 1, 1, D]
            embeds = embeds.reshaped(patches.dim(0), -1)
            var positions: [Int32] = []
            positions.reserveCapacity(patches.dim(0))
            for grid in grids {
                positions.append(contentsOf: MiniCPMV47Vision.nearestPositionIds(
                    height: grid.h, width: grid.w, numPatchesPerSide: numPatchesPerSide))
            }
            return embeds + positionEmbedding(MLXArray(positions))
        }
    }

    final class Attention: Module {
        let numHeads: Int
        let headDim: Int
        let scale: Float

        /// Diagnostics only (parity tests): replace the fused SDPA kernel with explicit matmul + softmax so the
        /// runtime's own float32 noise floor can be measured against the reference.
        nonisolated(unsafe) static var useExplicitSoftmax = false

        private func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
            if Self.useExplicitSoftmax {
                let scores = matmul(q * scale, k.transposed(0, 1, 3, 2))
                return matmul(softmax(scores, axis: -1, precise: true), v)
            }
            return MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        }

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: MiniCPMV47VisionConfiguration) {
            numHeads = config.numAttentionHeads
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)
            _qProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _kProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _vProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        }

        /// Bidirectional attention inside each segment. `segmentLengths` partition the `N` rows of `x`; every
        /// segment attends only to itself (the reference's `cu_seqlens`). Equal-length segments run as one
        /// batched SDPA, ragged ones (one crop per segment) run per segment.
        func callAsFunction(_ x: MLXArray, segmentLengths: [Int]) -> MLXArray {
            let n = x.dim(0)
            let q = qProj(x)
            let k = kProj(x)
            let v = vProj(x)
            let uniform = segmentLengths.allSatisfy { $0 == segmentLengths[0] }
            let attended: MLXArray
            if uniform, let length = segmentLengths.first, length > 0, n % length == 0 {
                let segments = n / length
                func heads(_ t: MLXArray) -> MLXArray {
                    t.reshaped(segments, length, numHeads, headDim).transposed(0, 2, 1, 3)
                }
                attended = attend(heads(q), heads(k), heads(v)).transposed(0, 2, 1, 3).reshaped(n, -1)
            } else {
                var parts: [MLXArray] = []
                var start = 0
                for length in segmentLengths {
                    let end = start + length
                    func heads(_ t: MLXArray) -> MLXArray {
                        t[start ..< end].reshaped(1, length, numHeads, headDim).transposed(0, 2, 1, 3)
                    }
                    parts.append(attend(heads(q), heads(k), heads(v)).transposed(0, 2, 1, 3).reshaped(length, -1))
                    start = end
                }
                attended = concatenated(parts)
            }
            return outProj(attended)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "fc1") var fc1: Linear
        @ModuleInfo(key: "fc2") var fc2: Linear
        let activation: (MLXArray) -> MLXArray

        init(_ config: MiniCPMV47VisionConfiguration) {
            _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
            _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
            activation = MiniCPMV47Vision.activation(named: config.hiddenAct)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(activation(fc1(x))) }
    }

    /// `gelu_pytorch_tanh` is torch's tanh approximation (MLXNN `geluApproximate`); plain `gelu` is the erf form.
    static func activation(named name: String) -> (MLXArray) -> MLXArray {
        switch name {
        case "gelu_pytorch_tanh", "gelu_new", "gelu_fast": return { geluApproximate($0) }
        case "quick_gelu": return { geluFastApproximate($0) }
        case "silu", "swish": return { silu($0) }
        case "relu": return { relu($0) }
        default: return { gelu($0) }
        }
    }

    final class EncoderLayer: Module {
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo(key: "self_attn") var selfAttn: Attention
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
        @ModuleInfo(key: "mlp") var mlp: MLP

        init(_ config: MiniCPMV47VisionConfiguration) {
            _layerNorm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            _selfAttn.wrappedValue = Attention(config)
            _layerNorm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            _mlp.wrappedValue = MLP(config)
        }

        func callAsFunction(_ x: MLXArray, segmentLengths: [Int]) -> MLXArray {
            var h = x + selfAttn(layerNorm1(x), segmentLengths: segmentLengths)
            h = h + mlp(layerNorm2(h))
            return h
        }
    }

    final class Encoder: Module {
        @ModuleInfo(key: "layers") var layers: [EncoderLayer]
        init(_ config: MiniCPMV47VisionConfiguration) {
            _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in EncoderLayer(config) }
        }
    }

    /// `MiniCPMV4_7ViTWindowAttentionMerger`: attention inside 2x2 windows, then the four window patches are
    /// concatenated and projected to one patch, plus the window mean as residual. Output stays in window order,
    /// which is the row-major order of the halved grid.
    final class WindowAttentionMerger: Module {
        @ModuleInfo(key: "self_attn") var selfAttn: Attention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
        @ModuleInfo(key: "linear_1") var linear1: Linear
        @ModuleInfo(key: "linear_2") var linear2: Linear
        let window: Int

        init(_ config: MiniCPMV47VisionConfiguration) {
            precondition(
                config.windowKernelSize.count == 2 && config.windowKernelSize[0] == config.windowKernelSize[1],
                "window_kernel_size must be square")
            window = config.windowKernelSize[0]
            _selfAttn.wrappedValue = Attention(config)
            _layerNorm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            _preNorm.wrappedValue = LayerNorm(dimensions: config.windowHiddenSize, eps: config.layerNormEps)
            _linear1.wrappedValue = Linear(config.windowHiddenSize, config.windowIntermediateSize, bias: true)
            _linear2.wrappedValue = Linear(config.windowIntermediateSize, config.hiddenSize, bias: true)
        }

        func callAsFunction(_ x: MLXArray, grids: [THW]) -> MLXArray {
            var index: [Int32] = []
            index.reserveCapacity(x.dim(0))
            var offset = 0
            for grid in grids {
                index.append(contentsOf: MiniCPMV47Vision.windowIndex(
                    height: grid.h, width: grid.w, window: window, offset: offset))
                offset += grid.h * grid.w
            }
            let gather = MLXArray(index)
            let windowSize = window * window
            let normed = layerNorm1(x)[gather]
            let attended = selfAttn(normed, segmentLengths: Array(repeating: windowSize, count: x.dim(0) / windowSize))
            let h = x[gather] + attended
            let embedDim = h.dim(-1)
            let patch = h.reshaped(-1, windowSize, embedDim)
            let flat = patch.reshaped(-1, windowSize * embedDim)
            let patchResidual = patch.mean(axis: 1)
            var out = preNorm(flat)
            out = linear1(out)
            out = geluApproximate(out)
            out = linear2(out)
            return out + patchResidual
        }
    }

    /// `MiniCPMV4_7DownsampleMLP`: folds a 2x2 of merged patches (4D) into one vector; GELU is the erf form.
    final class DownsampleMLP: Module, UnaryLayer {
        @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
        @ModuleInfo(key: "linear_1") var linear1: Linear
        @ModuleInfo(key: "linear_2") var linear2: Linear

        init(hiddenSize: Int, outputSize: Int) {
            let merged = hiddenSize * 4
            _preNorm.wrappedValue = LayerNorm(dimensions: merged, eps: 1e-6)
            _linear1.wrappedValue = Linear(merged, merged, bias: true)
            _linear2.wrappedValue = Linear(merged, outputSize, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            linear2(gelu(linear1(preNorm(x))))
        }
    }

    /// Top-level `merger`: per crop, group the merged grid 2x2 row-major and project into the LLM width.
    final class Merger: Module {
        @ModuleInfo(key: "mlp") var mlp: [DownsampleMLP]
        let mergeKernel: [Int]
        let mergerTimes: Int

        init(canvas: MiniCPMV47CanvasConfiguration, visionHidden: Int, llmHidden: Int) {
            mergeKernel = canvas.mergeKernelSize
            mergerTimes = max(1, canvas.mergerTimes)
            var mlps = (0 ..< mergerTimes - 1).map { _ in DownsampleMLP(hiddenSize: visionHidden, outputSize: visionHidden) }
            mlps.append(DownsampleMLP(hiddenSize: visionHidden, outputSize: llmHidden))
            _mlp.wrappedValue = mlps
        }

        /// - Parameter grids: per-crop `(h, w)` of the rows in `x` (already halved by the window merger).
        func callAsFunction(_ x: MLXArray, grids: [THW]) -> MLXArray {
            let (mh, mw) = (mergeKernel[0], mergeKernel[1])
            var outputs: [MLXArray] = []
            var start = 0
            for grid in grids {
                var height = grid.h
                var width = grid.w
                let count = height * width
                let embed = x.dim(-1)
                var h = x[start ..< start + count]
                    .reshaped(height / mh, mh, width / mw, mw, embed)
                    .transposed(0, 2, 1, 3, 4)
                    .reshaped((height / mh) * (width / mw), mh * mw * embed)
                h = mlp[0](h)
                for i in 1 ..< mergerTimes {
                    height /= mh
                    width /= mw
                    let inner = h.dim(-1)
                    h = h.reshaped(height / mh, mh, width / mw, mw, inner)
                        .transposed(0, 2, 1, 3, 4)
                        .reshaped((height / mh) * (width / mw), mh * mw * inner)
                    h = mlp[i](h)
                }
                outputs.append(h)
                start += count
            }
            return concatenated(outputs)
        }
    }

    /// `vision_tower.*` plus the top-level `merger.*` (remapped under the tower by `sanitize`, since the
    /// language model's `vision_tower` slot is the only vision module slot it has).
    final class Tower: Module, Qwen35VisionPath {
        @ModuleInfo(key: "embeddings") var embeddings: Embeddings
        @ModuleInfo(key: "encoder") var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm
        @ModuleInfo(key: "vit_merger") var vitMerger: WindowAttentionMerger
        @ModuleInfo(key: "merger") var merger: Merger

        let config: MiniCPMV47VisionConfiguration
        let canvas: MiniCPMV47CanvasConfiguration

        init(_ config: MiniCPMV47VisionConfiguration, canvas: MiniCPMV47CanvasConfiguration, llmHidden: Int) {
            self.config = config
            self.canvas = canvas
            _embeddings.wrappedValue = Embeddings(config)
            _encoder.wrappedValue = Encoder(config)
            _postLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            _vitMerger.wrappedValue = WindowAttentionMerger(config)
            _merger.wrappedValue = Merger(canvas: canvas, visionHidden: config.hiddenSize, llmHidden: llmHidden)
        }

        var visionDType: DType { embeddings.patchEmbedding.weight.dtype }

        /// Encoder output after `post_layernorm` (the reference `last_hidden_state`), with the per-crop grids
        /// those rows are laid out in (halved when the window merger ran).
        func encode(_ patches: MLXArray, grids: [THW]) -> (MLXArray, [THW]) {
            var h = embeddings(patches, grids: grids)
            var currentGrids = grids
            var segments = grids.map { $0.h * $0.w }
            let insert = canvas.usesViTMerger ? canvas.insertLayerId : -1
            for (index, layer) in encoder.layers.enumerated() {
                h = layer(h, segmentLengths: segments)
                if index == insert {
                    h = vitMerger(h, grids: currentGrids)
                    let k = vitMerger.window
                    currentGrids = currentGrids.map { THW($0.t, $0.h / k, $0.w / k) }
                    segments = currentGrids.map { $0.h * $0.w }
                }
            }
            return (postLayerNorm(h), currentGrids)
        }

        /// One LLM-embedding row per `mergeFactor x mergeFactor` patch block, crops in input order.
        func features(pixels: MLXArray, frames: [THW]) -> MLXArray {
            let (hidden, grids) = encode(pixels, grids: frames)
            return merger(hidden, grids: grids)
        }

        func mergedRowCount(for frames: [THW]) -> Int {
            let divisor = canvas.tokensPerPatchDivisor
            return frames.reduce(0) { $0 + ($1.t * $1.h * $1.w) / divisor }
        }

        /// The bundle stores `patch_embedding.weight` in MLX conv layout `[O, kh, kw, C]` already; a torch-layout
        /// `[O, C, kh, kw]` tensor is transposed. The top-level `merger.*` moves under `vision_tower.merger.*`.
        func sanitizeVision(weights: [String: MLXArray]) -> [String: MLXArray] {
            var out: [String: MLXArray] = [:]
            out.reserveCapacity(weights.count)
            for (key, value) in weights {
                var key = key
                var value = value
                if key.hasPrefix("merger.") {
                    key = "vision_tower." + key
                }
                if key == "vision_tower.embeddings.patch_embedding.weight", value.ndim == 4,
                    value.dim(1) == config.numChannels, value.dim(-1) != config.numChannels
                {
                    value = value.transposed(0, 2, 3, 1)
                }
                out[key] = value
            }
            return out
        }
    }
}

extension MiniCPMV47Vision.Tower: ModelCapabilityProviding {
    /// An image encoder. Video is not claimed: the reference video path stacks frames onto one canvas image
    /// with separators (`concat_frames_as_image`), which this runtime does not implement.
    public var providedModalities: Set<ModelRuntimeRequestModality> { [.vision] }
}

// MARK: - Canvas M-RoPE

enum MiniCPMV47Canvas {

    struct Frame {
        let modality: Int
        let thumbStart: Int
        let thumbEnd: Int
        var frameEnd: Int
        let slices: [(Int, Int)]
    }

    struct Group {
        let start: Int
        let end: Int
        let frames: [Frame]
    }

    /// Port of `MiniCPMV4_7Model._group_visual_frames`: maximal runs of placeholder tokens are crops; a crop
    /// preceded by `<slice>` joins the open frame, otherwise it opens a frame (thumbnail); frames separated by
    /// markers only share a group.
    static func groupVisualFrames(
        inputIds: [Int], tokenTypes: [Int], markers: (imageStart: Int, imageEnd: Int, sliceStart: Int, sliceEnd: Int, newline: Int)
    ) -> [Group] {
        let markerSet: Set<Int> = [markers.sliceStart, markers.sliceEnd, markers.imageStart, markers.imageEnd, markers.newline]
        let seqLen = inputIds.count
        var groups: [(Int, [(Int, Int, Int, [(Int, Int)])])] = []  // (groupStart, frames[(modality, thumbStart, thumbEnd, slices)])
        var openSlices: [(Int, Int)]? = nil
        var openFrameIndex: (Int, Int)? = nil  // (group index, frame index) owning openSlices
        var previousEnd = 0
        var i = 0
        while i < seqLen {
            let modality = tokenTypes[i]
            var j = i
            while j < seqLen && tokenTypes[j] == modality { j += 1 }
            defer { i = j }
            if modality == 0 { continue }
            let crop = (i, j)
            if openSlices != nil, crop.0 > 0, inputIds[crop.0 - 1] == markers.sliceStart, let owner = openFrameIndex {
                groups[owner.0].1[owner.1].3.append(crop)
            } else {
                let gap = Array(inputIds[previousEnd ..< max(previousEnd, crop.0 - 1)])
                let adjacent = !groups.isEmpty && gap.allSatisfy { markerSet.contains($0) }
                let frame = (modality, crop.0, crop.1, [(Int, Int)]())
                if adjacent {
                    groups[groups.count - 1].1.append(frame)
                } else {
                    groups.append((crop.0 - 1, [frame]))
                }
                openSlices = []
                openFrameIndex = (groups.count - 1, groups[groups.count - 1].1.count - 1)
            }
            previousEnd = crop.1
        }
        func frameEnd(after lastCropEnd: Int) -> Int {
            var end = lastCropEnd
            while end < seqLen && markerSet.contains(inputIds[end]) { end += 1 }
            return end
        }
        return groups.map { groupStart, frames in
            let built = frames.map { modality, thumbStart, thumbEnd, slices in
                Frame(
                    modality: modality, thumbStart: thumbStart, thumbEnd: thumbEnd,
                    frameEnd: frameEnd(after: slices.last?.1 ?? thumbEnd), slices: slices)
            }
            return Group(start: groupStart, end: built.last!.frameEnd, frames: built)
        }
    }

    /// `(3, llmH*llmW)` positions for one crop placed on a canvas: `linspace(0, canvas-1, llm)` rounded
    /// (torch `round` = half to even) plus the offset, all shifted by `start`. Zero canvas = own grid = arange.
    static func visionPositionIds(
        start: Int, gridH: Int, gridW: Int, canvasHeight: Int = 0, canvasWidth: Int = 0,
        hOffset: Int = 0, wOffset: Int = 0, mergeFactor: Int
    ) -> [[Int32]] {
        let llmH = gridH / mergeFactor
        let llmW = gridW / mergeFactor
        let ch = canvasHeight == 0 ? llmH : canvasHeight
        let cw = canvasWidth == 0 ? llmW : canvasWidth
        func coords(_ canvas: Int, _ n: Int, _ offset: Int) -> [Int] {
            // torch.linspace(0, canvas - 1, n) in float32, then round half-to-even.
            if n == 1 { return [offset] }
            let step = Float(canvas - 1) / Float(n - 1)
            return (0 ..< n).map { i in
                // torch linspace: first half start + i*step, second half end - (n-1-i)*step
                let v: Float = i < n / 2 ? Float(i) * step : Float(canvas - 1) - Float(n - 1 - i) * step
                return Int(v.rounded(.toNearestOrEven)) + offset
            }
        }
        let hs = coords(ch, llmH, hOffset)
        let ws = coords(cw, llmW, wOffset)
        var t: [Int32] = []
        var h: [Int32] = []
        var w: [Int32] = []
        t.reserveCapacity(llmH * llmW); h.reserveCapacity(llmH * llmW); w.reserveCapacity(llmH * llmW)
        for hv in hs {
            for wv in ws {
                t.append(Int32(start))
                h.append(Int32(hv + start))
                w.append(Int32(wv + start))
            }
        }
        return [t, h, w]
    }

    /// Port of `MiniCPMV4_7Model.get_rope_index` for one unpadded sequence. Returns `[3][S]` positions and the
    /// decode delta `max + 1 - S`.
    static func ropeIndex(
        inputIds: [Int], imageGrids: [THW], videoGrids: [THW], imageTokenId: Int, videoTokenId: Int,
        markers: (imageStart: Int, imageEnd: Int, sliceStart: Int, sliceEnd: Int, newline: Int), mergeFactor: Int
    ) -> (positions: [[Int32]], delta: Int32) {
        let seqLen = inputIds.count
        let tokenTypes = inputIds.map { $0 == imageTokenId ? 1 : ($0 == videoTokenId ? 2 : 0) }
        var pos: [[Int32]] = (0 ..< 3).map { _ in (0 ..< seqLen).map { Int32($0) } }
        var gridCursor = [1: 0, 2: 0]
        func nextGrid(_ modality: Int) -> (Int, Int) {
            let list = modality == 1 ? imageGrids : videoGrids
            let k = gridCursor[modality]!
            gridCursor[modality] = k + 1
            precondition(k < list.count, "MiniCPM-V canvas: more visual crops in the prompt than grids")
            return (list[k].h, list[k].w)
        }
        func fill(_ range: Range<Int>, from base: Int) {
            for (offset, idx) in range.enumerated() {
                for d in 0 ..< 3 { pos[d][idx] = Int32(base + offset) }
            }
        }

        var currentPos = 0
        var currentIdx = 0
        for group in groupVisualFrames(inputIds: inputIds, tokenTypes: tokenTypes, markers: markers) {
            if group.start > currentIdx {
                fill(currentIdx ..< group.start, from: currentPos)
                currentPos += group.start - currentIdx
            }
            var frameCursor = group.start
            for frame in group.frames {
                let thumb = nextGrid(frame.modality)
                let frameStart = frame.thumbStart - 1
                if frameStart > frameCursor {
                    fill(frameCursor ..< frameStart, from: currentPos)
                    currentPos += (frameStart - frameCursor) + 1
                }
                let frameEnd = min(frame.frameEnd, group.end)
                let canvasOrigin = currentPos
                let haloBeforeCanvas = max(canvasOrigin - 1, 0)

                var llmSliceH = 0, llmSliceW = 0
                var numRows = 0, numCols = 0
                var canvasHeight = thumb.0 / mergeFactor
                var canvasWidth = thumb.1 / mergeFactor
                var firstSlice: (Int, Int)? = nil
                if !frame.slices.isEmpty {
                    numCols = frame.slices.count
                    for k in 0 ..< frame.slices.count - 1 where frame.slices[k + 1].0 - frame.slices[k].1 > 2 {
                        numCols = k + 1
                        break
                    }
                    numRows = numCols > 0 ? frame.slices.count / numCols : 1
                    if numRows * numCols != frame.slices.count {
                        numRows = 1
                        numCols = frame.slices.count
                    }
                    let first = nextGrid(frame.modality)
                    firstSlice = first
                    llmSliceH = first.0 / mergeFactor
                    llmSliceW = first.1 / mergeFactor
                    canvasHeight = numRows * llmSliceH
                    canvasWidth = numCols * llmSliceW
                }

                for idx in frameStart ..< frameEnd { for d in 0 ..< 3 { pos[d][idx] = Int32(canvasOrigin) } }
                pos[1][frameStart] = Int32(haloBeforeCanvas)
                pos[2][frameStart] = Int32(haloBeforeCanvas)

                if frame.thumbEnd < frameEnd {
                    pos[1][frame.thumbEnd] = Int32(canvasOrigin + canvasHeight)
                    pos[2][frame.thumbEnd] = Int32(canvasOrigin + canvasWidth)
                }

                let thumbPositions = visionPositionIds(
                    start: canvasOrigin, gridH: thumb.0, gridW: thumb.1,
                    canvasHeight: canvasHeight, canvasWidth: canvasWidth, mergeFactor: mergeFactor)
                precondition(thumbPositions[0].count == frame.thumbEnd - frame.thumbStart,
                    "MiniCPM-V canvas: thumbnail placeholder count \(frame.thumbEnd - frame.thumbStart) != grid rows \(thumbPositions[0].count)")
                for d in 0 ..< 3 { pos[d].replaceSubrange(frame.thumbStart ..< frame.thumbEnd, with: thumbPositions[d]) }

                for (k, slice) in frame.slices.enumerated() {
                    let hOff = (k / numCols) * llmSliceH
                    let wOff = (k % numCols) * llmSliceW
                    let grid = k == 0 ? firstSlice! : nextGrid(frame.modality)
                    let sliceH = grid.0 / mergeFactor
                    let sliceW = grid.1 / mergeFactor
                    let startMarker = slice.0 - 1
                    if startMarker >= frameStart {
                        pos[1][startMarker] = Int32(canvasOrigin + hOff)
                        pos[2][startMarker] = Int32(canvasOrigin + wOff)
                    }
                    if slice.1 < frameEnd {
                        pos[1][slice.1] = Int32(canvasOrigin + hOff + sliceH - 1)
                        pos[2][slice.1] = Int32(canvasOrigin + wOff + sliceW - 1)
                    }
                    let slicePositions = visionPositionIds(
                        start: canvasOrigin, gridH: grid.0, gridW: grid.1,
                        hOffset: hOff, wOffset: wOff, mergeFactor: mergeFactor)
                    precondition(slicePositions[0].count == slice.1 - slice.0,
                        "MiniCPM-V canvas: slice placeholder count \(slice.1 - slice.0) != grid rows \(slicePositions[0].count)")
                    for d in 0 ..< 3 { pos[d].replaceSubrange(slice.0 ..< slice.1, with: slicePositions[d]) }
                }

                if frame.slices.count > 1 {
                    for k in 0 ..< frame.slices.count - 1 {
                        let gapStart = frame.slices[k].1
                        let gapEnd = frame.slices[k + 1].0
                        if gapEnd - gapStart <= 2 { continue }
                        let boundaryH = (k / numCols + 1) * llmSliceH - 1
                        let rightEdgeW = numCols * llmSliceW
                        for newlineIdx in (gapStart + 1) ..< (gapEnd - 1) {
                            pos[1][newlineIdx] = Int32(canvasOrigin + boundaryH)
                            pos[2][newlineIdx] = Int32(canvasOrigin + rightEdgeW)
                        }
                    }
                }

                currentPos = canvasOrigin + max(canvasHeight, canvasWidth) + 1
                frameCursor = frameEnd
            }
            if frameCursor < group.end {
                fill(frameCursor ..< group.end, from: currentPos)
                currentPos += group.end - frameCursor
            }
            currentIdx = group.end
        }
        if currentIdx < seqLen {
            fill(currentIdx ..< seqLen, from: currentPos)
        }
        let maxPos = pos.flatMap { $0 }.max() ?? Int32(seqLen - 1)
        return (pos, maxPos + 1 - Int32(seqLen))
    }

    /// `[3, B, S]` int32 positions and `[B]` deltas for a batch of rows, each laid out independently.
    static func ropeIndex(
        inputIds: MLXArray, imageGrids: [THW]?, videoGrids: [THW]?, imageTokenId: Int, videoTokenId: Int,
        canvas: MiniCPMV47CanvasConfiguration
    ) -> (MLXArray, MLXArray) {
        let batch = inputIds.dim(0)
        let seqLen = inputIds.dim(1)
        guard let markers = canvas.markerIds else {
            let flat = MLXArray(0 ..< seqLen).asType(.int32)
            return (
                broadcast(flat[.newAxis, .newAxis, 0...], to: [3, batch, seqLen]),
                MLXArray.zeros([batch], dtype: .int32)
            )
        }
        let rows = inputIds.asArray(Int32.self)
        var positions: [Int32] = []
        positions.reserveCapacity(3 * batch * seqLen)
        var deltas: [Int32] = []
        var perBatch: [[[Int32]]] = []
        for b in 0 ..< batch {
            let ids = rows[(b * seqLen) ..< ((b + 1) * seqLen)].map { Int($0) }
            // One shared grid list per batch is the Qwen3-VL convention this family inherits: all crops of
            // the batch are consumed in order across rows.
            let (pos, delta) = ropeIndex(
                inputIds: ids, imageGrids: imageGrids ?? [], videoGrids: videoGrids ?? [],
                imageTokenId: imageTokenId, videoTokenId: videoTokenId, markers: markers,
                mergeFactor: canvas.mergeFactor)
            perBatch.append(pos)
            deltas.append(delta)
        }
        for d in 0 ..< 3 {
            for b in 0 ..< batch { positions.append(contentsOf: perBatch[b][d]) }
        }
        return (MLXArray(positions, [3, batch, seqLen]), MLXArray(deltas))
    }
}
