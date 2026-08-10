//
//  MuseGlimmerVision.swift
//  vmlx-swift
//
//  Vision tower for Muse Glimmer 30B. Confirmed shapes and the two load-time
//  hazards are in `MuseGlimmer_ARCHITECTURE.md`.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct MuseGlimmerVisionConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHeads: Int
    public let numLayers: Int
    public let patchSize: Int
    public let patchTemporal: Int
    public let mergeSize: Int
    public let posEmbHeight: Int
    public let posEmbWidth: Int
    public let layerNormEps: Float
    public let ropeTheta: Float
    public let layerTypes: [String]
    public let inChannels: Int = 3

    public func isWindowed(_ layer: Int) -> Bool {
        layerTypes.indices.contains(layer) ? layerTypes[layer] == "window_attention" : true
    }

    /// The reference derives the attention window from the position-embedding
    /// grid: `pos_emb_height * patch_size` pixels.
    /// **Not in the checkpoint.** `config.json` declares `layer_types` with 39
    /// `window_attention` layers but never states the window size, so this is
    /// inferred: the position table covers a 32x32 patch grid, i.e. 448px, the
    /// model's native resolution.
    ///
    /// That inference is suspect — at 448 a native-resolution image is a single
    /// window, which makes those 39 layers behave as full attention and defeats
    /// the point of having them. A sweep against the six-trial shape battery
    /// (see MuseGlimmer_PROOF_MATRIX.md) found 448 and 224 give a constant
    /// answer (3/6, chance) while 112 discriminates the shapes but reports them
    /// inverted (0/6, which chance produces once in 64). None is correct, so
    /// the value is left at the inferred default and overridable for the next
    /// investigation rather than changed to one that scores worse.
    public var windowSize: Int {
        if let raw = ProcessInfo.processInfo.environment["VMLX_MUSE_VISION_WINDOW"],
            let value = Int(raw), value > 0
        {
            return value
        }
        return posEmbHeight * patchSize
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHeads = "num_attention_heads"
        case numLayers = "num_hidden_layers"
        case patchSize = "patch_size"
        case patchTemporal = "patch_temporal"
        case mergeSize = "merge_size"
        case posEmbHeight = "pos_emb_height"
        case posEmbWidth = "pos_emb_width"
        case layerNormEps = "layer_norm_eps"
        case layerTypes = "layer_types"
    }

    private enum RopeKeys: String, CodingKey {
        case ropeParameters = "rope_parameters"
        case ropeTheta = "rope_theta"
    }

    private struct RopeParameters: Codable {
        let ropeTheta: Float?
        enum CodingKeys: String, CodingKey { case ropeTheta = "rope_theta" }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1536
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 8960
        numHeads = try c.decodeIfPresent(Int.self, forKey: .numHeads) ?? 16
        numLayers = try c.decodeIfPresent(Int.self, forKey: .numLayers) ?? 50
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        patchTemporal = try c.decodeIfPresent(Int.self, forKey: .patchTemporal) ?? 2
        mergeSize = try c.decodeIfPresent(Int.self, forKey: .mergeSize) ?? 2
        posEmbHeight = try c.decodeIfPresent(Int.self, forKey: .posEmbHeight) ?? 32
        posEmbWidth = try c.decodeIfPresent(Int.self, forKey: .posEmbWidth) ?? 32
        layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5

        let layers = numLayers
        if let types = try c.decodeIfPresent([String].self, forKey: .layerTypes) {
            layerTypes = types
        } else {
            layerTypes = (0 ..< layers).map {
                ($0 + 1) % 4 == 0 ? "full_attention" : "window_attention"
            }
        }

        let ropeContainer = try decoder.container(keyedBy: RopeKeys.self)
        let params = try ropeContainer.decodeIfPresent(
            RopeParameters.self, forKey: .ropeParameters)
        let flat = try ropeContainer.decodeIfPresent(Float.self, forKey: .ropeTheta)
        ropeTheta = params?.ropeTheta ?? flat ?? 10_000.0
    }
}

// MARK: - Patch embedder

/// Linear patch projection plus a learned 32×32 position grid, bilinearly
/// resampled onto the image's actual patch grid.
///
/// The position table is *not* interpolated with a generic image resize: the
/// reference computes the four corner indices and their weights explicitly to
/// match `F.grid_sample(align_corners=false, padding="zeros")`, and notes that
/// the accumulated numerical error of the library path is itself load-bearing.
/// Out-of-range corners contribute weight 0 rather than being clamped into a
/// neighbour, which is what "padding=zeros" means here — clamping the index but
/// keeping its weight would smear the border embeddings inward.
private class MuseGlimmerPatchEmbedder: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    @ModuleInfo(key: "position_embedding_table") var positionTable: Embedding

    let side: Int
    let channels: Int
    let temporal: Int
    let patchSize: Int

    init(_ config: MuseGlimmerVisionConfiguration) {
        self.channels = config.inChannels
        self.temporal = config.patchTemporal
        self.patchSize = config.patchSize
        let patchDim =
            config.patchTemporal * config.inChannels * config.patchSize * config.patchSize
        self._patchEmbedding.wrappedValue = Linear(patchDim, config.hiddenSize, bias: false)
        self._positionTable.wrappedValue = Embedding(
            embeddingCount: config.posEmbHeight * config.posEmbWidth,
            dimensions: config.hiddenSize)
        self.side = config.posEmbHeight
        super.init()
    }

    func callAsFunction(_ patches: MLXArray, frames: [THW]) -> MLXArray {
        let embeddings = patchEmbedding(temporalMajor(patches))
        let positions = interpolatedPositions(frames: frames)
        return embeddings + positions.asType(embeddings.dtype)
    }

    /// `QwenVL.patchify` lays each patch vector out as `[channel][temporal][h][w]`,
    /// but this checkpoint's `patch_embedding` was trained on
    /// `[temporal][channel][h][w]`. Both orders have the same width, so the
    /// mistake is invisible in every shape check, and it is invisible again in
    /// greyscale — a still image duplicates the frame, so for equal channels
    /// the two orders permute identical blocks and produce the same vector.
    /// It only shows up in colour, where it interleaves the channels across the
    /// temporal slots.
    ///
    /// The weights identify the true order: under `[temporal][channel]` the two
    /// temporal slices of a channel have cosine 0.99 — they co-adapt because
    /// still images always feed them the same pixels — against 0.49 under the
    /// other reading, which is below the 0.74 baseline for unrelated slices.
    private func temporalMajor(_ patches: MLXArray) -> MLXArray {
        let spatial = patchSize * patchSize
        guard patches.dim(-1) == channels * temporal * spatial else { return patches }
        return patches.reshaped(patches.dim(0), channels, temporal, spatial)
            .transposed(0, 2, 1, 3)
            .reshaped(patches.dim(0), -1)
    }

    /// Four gathers plus a weighted sum — the explicit form of a bilinear
    /// sample over the `side × side` table.
    private func interpolatedPositions(frames: [THW]) -> MLXArray {
        var cornerIndices: [[Int32]] = [[], [], [], []]
        var cornerWeights: [[Float]] = [[], [], [], []]

        for frame in frames {
            let (t, h, w) = frame.values

            // Cell centers mapped into table space, align_corners=false.
            var hFloor = [Int](repeating: 0, count: h)
            var hCeil = [Int](repeating: 0, count: h)
            var hFrac = [Float](repeating: 0, count: h)
            var hFloorValid = [Bool](repeating: false, count: h)
            var hCeilValid = [Bool](repeating: false, count: h)
            for i in 0 ..< h {
                let g = (Float(i) + 0.5) * (Float(side) / Float(h)) - 0.5
                let f = g.rounded(.down)
                hFrac[i] = g - f
                let fi = Int(f)
                hFloorValid[i] = fi >= 0 && fi <= side - 1
                hCeilValid[i] = fi + 1 >= 0 && fi + 1 <= side - 1
                hFloor[i] = min(max(fi, 0), side - 1)
                hCeil[i] = min(max(fi + 1, 0), side - 1)
            }

            var wFloor = [Int](repeating: 0, count: w)
            var wCeil = [Int](repeating: 0, count: w)
            var wFrac = [Float](repeating: 0, count: w)
            var wFloorValid = [Bool](repeating: false, count: w)
            var wCeilValid = [Bool](repeating: false, count: w)
            for j in 0 ..< w {
                let g = (Float(j) + 0.5) * (Float(side) / Float(w)) - 0.5
                let f = g.rounded(.down)
                wFrac[j] = g - f
                let fj = Int(f)
                wFloorValid[j] = fj >= 0 && fj <= side - 1
                wCeilValid[j] = fj + 1 >= 0 && fj + 1 <= side - 1
                wFloor[j] = min(max(fj, 0), side - 1)
                wCeil[j] = min(max(fj + 1, 0), side - 1)
            }

            // Row-major (h, w) corner tables for this frame.
            var idx: [[Int32]] = [[], [], [], []]
            var wgt: [[Float]] = [[], [], [], []]
            for i in 0 ..< h {
                for j in 0 ..< w {
                    let hf = hFrac[i]
                    let wf = wFrac[j]
                    idx[0].append(Int32(hFloor[i] * side + wFloor[j]))
                    idx[1].append(Int32(hFloor[i] * side + wCeil[j]))
                    idx[2].append(Int32(hCeil[i] * side + wFloor[j]))
                    idx[3].append(Int32(hCeil[i] * side + wCeil[j]))
                    wgt[0].append(
                        (1 - hf) * (1 - wf) * ((hFloorValid[i] && wFloorValid[j]) ? 1 : 0))
                    wgt[1].append((1 - hf) * wf * ((hFloorValid[i] && wCeilValid[j]) ? 1 : 0))
                    wgt[2].append(hf * (1 - wf) * ((hCeilValid[i] && wFloorValid[j]) ? 1 : 0))
                    wgt[3].append(hf * wf * ((hCeilValid[i] && wCeilValid[j]) ? 1 : 0))
                }
            }

            // Raster order. The tower runs over individual patches, not over
            // merge blocks — the 2x2 merge happens at the very end — so the
            // position rows are walked straight down the grid.
            for _ in 0 ..< t {
                for corner in 0 ..< 4 {
                    cornerIndices[corner].append(contentsOf: idx[corner])
                    cornerWeights[corner].append(contentsOf: wgt[corner])
                }
            }
        }

        var accumulated: MLXArray?
        for corner in 0 ..< 4 {
            let gathered = positionTable(MLXArray(cornerIndices[corner]))
            let weights = MLXArray(cornerWeights[corner]).expandedDimensions(axis: 1)
            let term = gathered.asType(.float32) * weights
            accumulated = accumulated.map { $0 + term } ?? term
        }
        return accumulated ?? MLXArray.zeros([0])
    }
}

// MARK: - Blocks

private class MuseGlimmerVisionAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "proj") var outProj: Linear

    init(_ config: MuseGlimmerVisionConfiguration) {
        let dim = config.hiddenSize
        self.numHeads = config.numHeads
        self.headDim = dim / config.numHeads
        self.scale = pow(Float(headDim), -0.5)
        // Every projection here carries a bias, unlike the text tower.
        self._qProj.wrappedValue = Linear(dim, dim, bias: true)
        self._kProj.wrappedValue = Linear(dim, dim, bias: true)
        self._vProj.wrappedValue = Linear(dim, dim, bias: true)
        self._outProj.wrappedValue = Linear(dim, dim, bias: true)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray, rotary: MLXArray
    ) -> MLXArray {
        let L = x.dim(0)

        var q = qProj(x).reshaped(L, numHeads, headDim)
        var k = kProj(x).reshaped(L, numHeads, headDim)
        let v = vProj(x).reshaped(L, numHeads, headDim)

        q = MuseGlimmerVisionRoPE.apply(q, freqs: rotary)
        k = MuseGlimmerVisionRoPE.apply(k, freqs: rotary)

        let qh = q.reshaped(1, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let kh = k.reshaped(1, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let vh = v.reshaped(1, L, numHeads, headDim).transposed(0, 2, 1, 3)

        // Non-causal: visibility comes entirely from the window mask.
        let out = MLXFast.scaledDotProductAttention(
            queries: qh, keys: kh, values: vh, scale: scale,
            mask: .array(mask)
        )
        .transposed(0, 2, 1, 3)
        .reshaped(L, numHeads * headDim)

        return outProj(out)
    }
}

private class MuseGlimmerVisionMLP: Module, UnaryLayer {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self.fc1 = Linear(dimensions, hiddenDimensions, bias: true)
        self.fc2 = Linear(hiddenDimensions, dimensions, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(geluApproximate(fc1(x)))
    }
}

private class MuseGlimmerVisionBlock: Module {
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo(key: "attn") var attention: MuseGlimmerVisionAttention
    @ModuleInfo var mlp: MuseGlimmerVisionMLP

    init(_ config: MuseGlimmerVisionConfiguration) {
        // LayerNorm with bias here, not the text tower's centered RMSNorm.
        self.norm1 = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.norm2 = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._attention.wrappedValue = MuseGlimmerVisionAttention(config)
        self.mlp = MuseGlimmerVisionMLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, rotary: MLXArray) -> MLXArray {
        var h = x + attention(norm1(x), mask: mask, rotary: rotary)
        h = h + mlp(norm2(h))
        return h
    }
}

// MARK: - Rotary

enum MuseGlimmerVisionRoPE {
    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let x1 = x[.ellipsis, 0 ..< half]
        let x2 = x[.ellipsis, half...]
        return concatenated([-x2, x1], axis: -1)
    }

    /// `freqs` is `(L, headDim/2)`; duplicated to `(L, headDim)` and applied
    /// per head. This is the split-half (NeoX/Qwen) pairing: element `i` is
    /// rotated against element `i + headDim/2`.
    static func apply(_ x: MLXArray, freqs: MLXArray) -> MLXArray {
        let doubled = concatenated([freqs, freqs], axis: -1).expandedDimensions(axis: 1)
        let cosT = cos(doubled).asType(x.dtype)
        let sinT = sin(doubled).asType(x.dtype)
        return (x * cosT) + (rotateHalf(x) * sinT)
    }

}

// MARK: - Vision model

public class MuseGlimmerVisionModel: Module {
    @ModuleInfo(key: "patch_embedder") fileprivate var patchEmbedder: MuseGlimmerPatchEmbedder
    @ModuleInfo(key: "layers") fileprivate var layers: [MuseGlimmerVisionBlock]
    @ModuleInfo(key: "ln_pre") var lnPre: LayerNorm
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm

    // `mergeUnit` is read by `callAsFunction`; `rotary` is not a Module, so
    // both are plain stored properties set before `super.init()`.

    let config: MuseGlimmerVisionConfiguration
    private let rotary: QwenVL.VisionRotaryEmbedding
    private let mergeUnit: Int

    public init(_ config: MuseGlimmerVisionConfiguration) {
        self.config = config
        self.mergeUnit = config.mergeSize * config.mergeSize
        self._patchEmbedder.wrappedValue = MuseGlimmerPatchEmbedder(config)
        self._layers.wrappedValue = (0 ..< config.numLayers).map { _ in
            MuseGlimmerVisionBlock(config)
        }
        self._lnPre.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._lnPost.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.rotary = QwenVL.VisionRotaryEmbedding(
            dimensions: (config.hiddenSize / config.numHeads) / 2, theta: config.ropeTheta)
        super.init()
    }

    /// Returns `(L, hiddenSize * mergeSize²)` — the merged patch grid, ready
    /// for the adapter. Reordering back to raster order happens here so the
    /// caller never sees window order.
    public func callAsFunction(_ patches: MLXArray, frames: [THW]) -> MLXArray {
        callAsFunction(patches, frames: frames, probe: nil)
    }

    /// `probe` is called with the running activation after the embedder, after
    /// `ln_pre`, and after every block. A tower can produce well-shaped output
    /// that carries no usable picture, and that failure is only localizable by
    /// watching the signal layer by layer — there is no other seam to measure.
    public func callAsFunction(
        _ patches: MLXArray, frames: [THW], probe: ((String, MLXArray) -> Void)?
    ) -> MLXArray {
        // `QwenVL.patchify` groups its rows by merge block; this tower is
        // defined over the raster grid, so the rows are ungrouped first and
        // regrouped at the end by `pixelShuffle`.
        var h = patchEmbedder(rasterOrder(patches, frames: frames), frames: frames)
        probe?("embedder", h)
        h = lnPre(h)
        probe?("ln_pre", h)

        let seqLen = h.dim(0)
        let rotaryEmb = rotaryPositionEmbedding(frames)
        let (windowIndex, cuWindowSeqlens) = windowIndexAndSeqlens(frames)

        var cuSeqlens = [0]
        for frame in frames {
            let frameLen = frame.h * frame.w
            for _ in 0 ..< frame.t { cuSeqlens.append(cuSeqlens.last! + frameLen) }
        }

        let fullMask = blockDiagonalMask(sequenceLength: seqLen, cuSeqlens: cuSeqlens)
        let windowMask = blockDiagonalMask(
            sequenceLength: seqLen, cuSeqlens: cuWindowSeqlens)

        // Permute into window order, carrying the rotary rows with it. Both are
        // per-patch, so the permutation indexes rows directly.
        let order = MLXArray(windowIndex)
        h = h[order, 0...]
        let rot = rotaryEmb[order, 0...]

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: config.isWindowed(i) ? windowMask : fullMask, rotary: rot)
            probe?("layer\(i)", h)
        }

        // Undo the window permutation first, so `ln_post` and the merge both
        // run on raster-ordered patches.
        h = h[argSort(order, axis: 0), 0...]
        h = lnPost(h)
        return pixelShuffle(h, frames: frames)
    }

    /// `QwenVL.patchify` emits rows grouped as `(hb, wb, di, dj)`; this returns
    /// them in raster order `(row, col)`, which is the order the tower's
    /// positions, rotary coordinates and window partition are all defined over.
    private func rasterOrder(_ patches: MLXArray, frames: [THW]) -> MLXArray {
        let merge = config.mergeSize
        var gather: [Int32] = []
        var base = 0
        for frame in frames {
            let (t, h, w) = frame.values
            for _ in 0 ..< t {
                for row in 0 ..< h {
                    for col in 0 ..< w {
                        let block = (row / merge) * (w / merge) + (col / merge)
                        let within = (row % merge) * merge + (col % merge)
                        gather.append(Int32(base + block * merge * merge + within))
                    }
                }
                base += h * w
            }
        }
        guard gather.count == patches.dim(0) else { return patches }
        return patches[MLXArray(gather), 0...]
    }

    /// Groups each `mergeSize x mergeSize` block of raster-ordered patches into
    /// one token. The concatenation is **feature-major** — all `mergeUnit`
    /// values of feature 0, then of feature 1, and so on — not one patch vector
    /// after another. Both orderings have the same width, so getting it wrong
    /// still feeds the adapter a correctly shaped, permuted vector.
    private func pixelShuffle(_ h: MLXArray, frames: [THW]) -> MLXArray {
        let merge = config.mergeSize
        let dim = h.dim(-1)
        var out: [MLXArray] = []
        var base = 0
        for frame in frames {
            let (t, gh, gw) = frame.values
            var gather: [Int32] = []
            for f in 0 ..< t {
                let frameBase = base + f * gh * gw
                for hb in 0 ..< (gh / merge) {
                    for wb in 0 ..< (gw / merge) {
                        for di in 0 ..< merge {
                            for dj in 0 ..< merge {
                                gather.append(
                                    Int32(frameBase + (hb * merge + di) * gw + (wb * merge + dj)))
                            }
                        }
                    }
                }
            }
            let cells = t * (gh / merge) * (gw / merge)
            let grouped = h[MLXArray(gather), 0...].reshaped(cells, merge * merge, dim)
            out.append(grouped.transposed(0, 2, 1).reshaped(cells, dim * merge * merge))
            base += t * gh * gw
        }
        return out.count == 1 ? out[0] : concatenated(out, axis: 0)
    }

    // MARK: Position / window bookkeeping

    /// Rotary coordinates, in raster order, as `(column, row)` — **not**
    /// `(row, column)`. The checkpoint interleaves its head dimension as
    /// `[w, h, w, h]`, so feeding `(row, column)` transposes every patch's
    /// coordinates: the tower still produces well-formed features, but rows and
    /// columns are exchanged, which is invisible on a symmetric picture and
    /// scrambles everything else.
    ///
    /// Coordinates are 1-based; the trained offset is 1, not 0.
    private func rotaryPositionEmbedding(_ frames: [THW]) -> MLXArray {
        var positionIds = [MLXArray]()

        for frame in frames {
            let (t, h, w) = frame.values

            var rows = expandedDimensions(MLXArray(0 ..< h), axis: 1)
            rows = repeated(rows, count: w, axis: 1).flattened() + 1

            var cols = expandedDimensions(MLXArray(0 ..< w), axis: 0)
            cols = repeated(cols, count: h, axis: 0).flattened() + 1

            positionIds.append(tiled(stacked([cols, rows], axis: -1), repetitions: [t, 1]))
        }

        let indices = concatenated(positionIds, axis: 0)
        // +2 so the 1-based coordinates stay inside the table.
        let maxSide = (frames.lazy.map { max($0.h, $0.w) }.max() ?? 0) + 2
        return rotary(sequenceLength: maxSide)[indices].reshaped(indices.dim(0), -1)
    }

    /// Window partition over the merged grid, mirroring the Qwen2.5-VL scheme:
    /// pad the merged grid up to a whole number of windows, mark the padding
    /// so it can be dropped, and emit cumulative lengths for the mask.
    private func windowIndexAndSeqlens(_ frames: [THW]) -> ([Int32], [Int]) {
        var windowIndex: [Int32] = []
        var cuWindowSeqlens = [0]
        var offset = 0
        // The partition is over individual patches: the 2x2 merge happens after
        // the encoder, so nothing here is expressed in merged cells.
        let windowCells = config.windowSize / config.patchSize

        for frame in frames {
            let (t, gridH, gridW) = frame.values
            let llmH = gridH
            let llmW = gridW

            let padH = (windowCells - llmH % windowCells) % windowCells
            let padW = (windowCells - llmW % windowCells) % windowCells
            let windowsH = (llmH + padH) / windowCells
            let windowsW = (llmW + padW) / windowCells

            for frameIdx in 0 ..< t {
                let base = offset + frameIdx * llmH * llmW
                for wh in 0 ..< windowsH {
                    for ww in 0 ..< windowsW {
                        var count = 0
                        for r in 0 ..< windowCells {
                            let row = wh * windowCells + r
                            if row >= llmH { continue }
                            for c in 0 ..< windowCells {
                                let col = ww * windowCells + c
                                if col >= llmW { continue }
                                windowIndex.append(Int32(base + row * llmW + col))
                                count += 1
                            }
                        }
                        if count > 0 {
                            cuWindowSeqlens.append(cuWindowSeqlens.last! + count)
                        }
                    }
                }
            }
            offset += t * llmH * llmW
        }

        return (windowIndex, cuWindowSeqlens)
    }

    /// `true` where a query may see a key. Built as one host-side buffer
    /// because the per-segment MLX slice assignment costs a kernel launch per
    /// window, and a large image has hundreds of them.
    private func blockDiagonalMask(sequenceLength: Int, cuSeqlens: [Int]) -> MLXArray {
        var flags = [Bool](repeating: false, count: sequenceLength * sequenceLength)
        for i in 1 ..< cuSeqlens.count {
            let start = cuSeqlens[i - 1]
            let end = min(cuSeqlens[i], sequenceLength)
            if start >= end { continue }
            for row in start ..< end {
                let rowBase = row * sequenceLength
                for col in start ..< end { flags[rowBase + col] = true }
            }
        }
        return MLXArray(flags).reshaped(1, 1, sequenceLength, sequenceLength)
    }
}

// MARK: - Projector

/// `vision_adapter` (fc1 → act → fc2 → act) followed by `vision_projection`
/// into the text width. The activation runs after *both* adapter layers, which
/// is easy to miss reading the block as a standard two-layer MLP.
public class MuseGlimmerVisionProjector: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(inputDimensions: Int, hiddenDimensions: Int) {
        self._fc1.wrappedValue = Linear(inputDimensions, hiddenDimensions, bias: false)
        self._fc2.wrappedValue = Linear(hiddenDimensions, hiddenDimensions, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        geluApproximate(fc2(geluApproximate(fc1(x))))
    }
}
