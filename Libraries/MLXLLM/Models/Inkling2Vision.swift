// Copyright © 2026 Osaurus AI. All rights reserved.
//
// Inkling-2 image input: the ViT vision tower, its image preprocessing and prompt expansion.
//
// Reference: transformers `Inkling2VisionModel` / `Inkling2ImageProcessor` (and the matching MLX
// port). An image is cut, at native resolution, into 40 x 40 pixel tiles (2 x 2 native 20 px
// patches); each tile becomes one language-model token. The tower embeds the native patches, adds a
// bilinearly resized learned position grid, prepends a class token, runs 23 pre-norm blocks with
// axial 2-D rotary attention and layer scale, drops the class token, concatenates each tile's four
// patch features and projects them to the text width (Linear-GELU-Linear). The text model's
// embedding (embed + embed_norm) is computed for the whole prompt and the image-token rows are then
// replaced by these features (they do not pass through embed_norm).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

#if canImport(CoreImage)
    import CoreImage
#endif

// MARK: - Configuration

public struct Inkling2VisionConfiguration: Codable, Sendable {
    public var hiddenSize = 1024
    public var numHiddenLayers = 23
    public var numAttentionHeads = 16
    public var numChannels = 3
    /// Native patch side in pixels (`native_patch_size` / `patch_size`).
    public var patchSize = 20
    /// Native patches per tile side (`adapter_merge_size` / `spatial_merge_size`).
    public var mergeSize = 2
    public var posembGridSize = 32
    public var projectionHiddenDim = 4096
    public var projectorBias = false
    public var textHiddenSize = 2304
    public var layerNormEps: Float = 1e-5
    public var mlpRatio: Float = 4
    public var ropeTheta: Float = 10000
    /// `<|unused_200054|>`.
    public var imageTokenId = 200054

    /// Pixels per tile side (one language-model token).
    public var tileSize: Int { patchSize * mergeSize }
    public var headDim: Int { hiddenSize / numAttentionHeads }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numChannels = "num_channels"
        case nativePatchSize = "native_patch_size"
        case patchSize = "patch_size"
        case adapterMergeSize = "adapter_merge_size"
        case spatialMergeSize = "spatial_merge_size"
        case posembGridSize = "posemb_grid_size"
        case projectionHiddenDim = "projection_hidden_dim"
        case projectorBias = "projector_bias"
        case textHiddenSize = "text_hidden_size"
        case layerNormEps = "layer_norm_eps"
        case mlpRatio = "mlp_ratio"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
    }

    struct RopeParameters: Codable { var ropeTheta: Float?; enum CodingKeys: String, CodingKey { case ropeTheta = "rope_theta" } }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? numHiddenLayers
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? numAttentionHeads
        numChannels = try c.decodeIfPresent(Int.self, forKey: .numChannels) ?? numChannels
        patchSize = try c.decodeIfPresent(Int.self, forKey: .nativePatchSize)
            ?? c.decodeIfPresent(Int.self, forKey: .patchSize) ?? patchSize
        mergeSize = try c.decodeIfPresent(Int.self, forKey: .adapterMergeSize)
            ?? c.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? mergeSize
        posembGridSize = try c.decodeIfPresent(Int.self, forKey: .posembGridSize) ?? posembGridSize
        projectionHiddenDim = try c.decodeIfPresent(Int.self, forKey: .projectionHiddenDim) ?? projectionHiddenDim
        projectorBias = try c.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? projectorBias
        textHiddenSize = try c.decodeIfPresent(Int.self, forKey: .textHiddenSize) ?? textHiddenSize
        layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? layerNormEps
        mlpRatio = try c.decodeIfPresent(Float.self, forKey: .mlpRatio) ?? mlpRatio
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
            ?? (try? c.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters))??.ropeTheta
            ?? ropeTheta
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try c.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try c.encode(patchSize, forKey: .nativePatchSize)
        try c.encode(mergeSize, forKey: .adapterMergeSize)
        try c.encode(posembGridSize, forKey: .posembGridSize)
        try c.encode(projectionHiddenDim, forKey: .projectionHiddenDim)
        try c.encode(textHiddenSize, forKey: .textHiddenSize)
    }

    /// Image input is on unless `VMLX_INKLING2_VL=0` (text-only construction: no tower is built).
    static var enabled: Bool { ProcessInfo.processInfo.environment["VMLX_INKLING2_VL"] != "0" }
}

// MARK: - Tower

/// `(x1, x2)` channel pairs -> `(-x2, x1)`.
private func inklingRotatePairs(_ x: MLXArray) -> MLXArray {
    let pairs = x.reshaped(Array(x.shape.dropLast()) + [-1, 2])
    return MLX.stacked([-pairs[.ellipsis, 1], pairs[.ellipsis, 0]], axis: -1).reshaped(x.shape)
}

final class Inkling2VisionAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    let heads: Int
    let headDim: Int

    init(_ c: Inkling2VisionConfiguration) {
        heads = c.numAttentionHeads
        headDim = c.headDim
        _qProj.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        _kProj.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        _vProj.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        _outProj.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
    }

    /// `x` `[N, H]`; `cos` / `sin` `[N, headDim]` fp32.
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let n = x.dim(0)
        func split(_ a: MLXArray) -> MLXArray { a.reshaped([1, n, heads, headDim]) }
        let c = cos.reshaped([1, n, 1, headDim])
        let s = sin.reshaped([1, n, 1, headDim])
        func rope(_ a: MLXArray) -> MLXArray {
            let f = a.asType(.float32)
            return (f * c + inklingRotatePairs(f) * s).asType(a.dtype)
        }
        let q = rope(split(qProj(x))).transposed(0, 2, 1, 3)
        let k = rope(split(kProj(x))).transposed(0, 2, 1, 3)
        let v = split(vProj(x)).transposed(0, 2, 1, 3)
        let o = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / Float(headDim).squareRoot(), mask: .none)
        return outProj(o.transposed(0, 2, 1, 3).reshaped([n, -1]))
    }
}

final class Inkling2VisionMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ c: Inkling2VisionConfiguration) {
        let inner = Int(Float(c.hiddenSize) * c.mlpRatio)
        _fc1.wrappedValue = Linear(c.hiddenSize, inner)
        _fc2.wrappedValue = Linear(inner, c.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

final class Inkling2VisionEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Inkling2VisionAttention
    @ModuleInfo(key: "mlp") var mlp: Inkling2VisionMLP
    @ModuleInfo(key: "layernorm_before") var normBefore: LayerNorm
    @ModuleInfo(key: "layernorm_after") var normAfter: LayerNorm
    @ParameterInfo(key: "lambda_1") var lambda1: MLXArray
    @ParameterInfo(key: "lambda_2") var lambda2: MLXArray

    init(_ c: Inkling2VisionConfiguration) {
        _attention.wrappedValue = Inkling2VisionAttention(c)
        _mlp.wrappedValue = Inkling2VisionMLP(c)
        _normBefore.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _normAfter.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _lambda1.wrappedValue = MLXArray.ones([c.hiddenSize])
        _lambda2.wrappedValue = MLXArray.ones([c.hiddenSize])
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let h = x + lambda1 * attention(normBefore(x), cos: cos, sin: sin)
        return h + lambda2 * mlp(normAfter(h))
    }
}

final class Inkling2VisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    @ParameterInfo(key: "position_embedding") var positionEmbedding: MLXArray
    @ParameterInfo(key: "class_embedding") var classEmbedding: MLXArray
    @ParameterInfo(key: "cls_position_embedding") var clsPositionEmbedding: MLXArray

    init(_ c: Inkling2VisionConfiguration) {
        _patchEmbedding.wrappedValue = Linear(
            c.numChannels * c.patchSize * c.patchSize, c.hiddenSize, bias: false)
        _positionEmbedding.wrappedValue = MLXArray.zeros([c.posembGridSize * c.posembGridSize, c.hiddenSize])
        _classEmbedding.wrappedValue = MLXArray.zeros([c.hiddenSize])
        _clsPositionEmbedding.wrappedValue = MLXArray.zeros([c.hiddenSize])
    }
}

final class Inkling2VisionMerger: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ c: Inkling2VisionConfiguration) {
        _linear1.wrappedValue = Linear(
            c.mergeSize * c.mergeSize * c.hiddenSize, c.projectionHiddenDim, bias: c.projectorBias)
        _linear2.wrappedValue = Linear(c.projectionHiddenDim, c.textHiddenSize, bias: c.projectorBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(gelu(linear1(x))) }
}

public final class Inkling2VisionModel: Module {
    @ModuleInfo(key: "embeddings") var embeddings: Inkling2VisionEmbeddings
    @ModuleInfo(key: "pre_layernorm") var preLayerNorm: LayerNorm
    @ModuleInfo(key: "layers") var layers: [Inkling2VisionEncoderLayer]
    @ModuleInfo(key: "merger") var merger: Inkling2VisionMerger
    public let config: Inkling2VisionConfiguration

    init(_ c: Inkling2VisionConfiguration) {
        config = c
        _embeddings.wrappedValue = Inkling2VisionEmbeddings(c)
        _preLayerNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _layers.wrappedValue = (0 ..< c.numHiddenLayers).map { _ in Inkling2VisionEncoderLayer(c) }
        _merger.wrappedValue = Inkling2VisionMerger(c)
    }

    /// Bilinear resampling weights `[out, in]` (PyTorch `align_corners=False`: half-pixel centres,
    /// source clamped at 0, the upper neighbour clamped to the last row).
    static func bilinearWeights(out: Int, in size: Int) -> MLXArray {
        var w = [Float](repeating: 0, count: out * size)
        let scale = Float(size) / Float(out)
        for o in 0 ..< out {
            let src = max((Float(o) + 0.5) * scale - 0.5, 0)
            let i0 = min(Int(src), size - 1)
            let i1 = min(i0 + 1, size - 1)
            let f = src - Float(i0)
            w[o * size + i0] += 1 - f
            w[o * size + i1] += f
        }
        return MLXArray(w, [out, size])
    }

    /// Features of one image: `tiles` `[rows * cols, tile, tile, C]` (normalized pixels, tile-major)
    /// -> `[rows * cols, textHidden]`.
    public func features(tiles: MLXArray, rows: Int, cols: Int) -> MLXArray {
        let c = config
        let (m, p, H) = (c.mergeSize, c.patchSize, c.hiddenSize)
        let count = rows * cols
        let dtype = preLayerNorm.weight?.dtype ?? .bfloat16
        // [count, m, p, m, p, C] -> [count, m, m, p, p, C] -> one row per native patch.
        let patches = tiles.asType(dtype).reshaped([count, m, p, m, p, c.numChannels])
            .transposed(0, 1, 3, 2, 4, 5).reshaped([count * m * m, -1])
        var x = embeddings.patchEmbedding(patches)

        // Learned [G, G, H] grid resized to the native patch grid, in the same tile-major order.
        let G = c.posembGridSize
        let (gh, gw) = (rows * m, cols * m)
        let table = embeddings.positionEmbedding.asType(.float32).reshaped([G, G * H])
        let wy = Self.bilinearWeights(out: gh, in: G)
        let wx = Self.bilinearWeights(out: gw, in: G)
        var pos = MLX.matmul(wy, table).reshaped([gh, G, H])  // rows resized
        pos = MLX.matmul(wx, pos.transposed(1, 0, 2).reshaped([G, gh * H]))  // [gw, gh*H]
            .reshaped([gw, gh, H]).transposed(1, 0, 2)  // [gh, gw, H]
        pos = pos.reshaped([rows, m, cols, m, H]).transposed(0, 2, 1, 3, 4).reshaped([count * m * m, H])
        x = x + pos.asType(x.dtype)

        let cls = (embeddings.classEmbedding + embeddings.clsPositionEmbedding).asType(x.dtype)
        x = preLayerNorm(MLX.concatenated([cls.reshaped([1, H]), x], axis: 0))

        // Axial rotary angles: (col + 1, row + 1) per patch, 0 for the class token; width
        // frequencies then height frequencies, each repeated for its channel pair.
        let half = c.headDim / 2
        let inv = (0 ..< half / 2).map { 1 / pow(c.ropeTheta, Float(2 * $0) / Float(half)) }
        var angles = [Float](repeating: 0, count: (count * m * m + 1) * c.headDim)
        var row = 1
        for ty in 0 ..< rows {
            for tx in 0 ..< cols {
                for my in 0 ..< m {
                    for mx in 0 ..< m {
                        let px = Float(tx * m + mx + 1)
                        let py = Float(ty * m + my + 1)
                        for (j, f) in inv.enumerated() {
                            let base = row * c.headDim
                            angles[base + 2 * j] = px * f
                            angles[base + 2 * j + 1] = px * f
                            angles[base + half + 2 * j] = py * f
                            angles[base + half + 2 * j + 1] = py * f
                        }
                        row += 1
                    }
                }
            }
        }
        let a = MLXArray(angles, [count * m * m + 1, c.headDim])
        let cos = MLX.cos(a)
        let sin = MLX.sin(a)
        for layer in layers {
            x = layer(x, cos: cos, sin: sin)
        }
        x = x[1...].reshaped([count, m * m * H])
        return merger(x)
    }
}

// MARK: - Preprocessing

public struct Inkling2ProcessedImage: Sendable {
    /// Normalized tiles `[rows * cols, tile, tile, 3]`, tile-major (row of tiles, then column).
    public let tiles: [Float]
    public let rows: Int
    public let cols: Int
    public let tileSize: Int
    public var tokenCount: Int { rows * cols }
}

public enum Inkling2VisionError: Error, CustomStringConvertible {
    case unreadableImage(String)
    case placeholderMismatch(found: Int, images: Int)
    case spanMismatch(String)

    public var description: String {
        switch self {
        case .unreadableImage(let why): return "Inkling-2 image could not be read: \(why)"
        case .placeholderMismatch(let found, let images):
            return "Inkling-2 prompt has \(found) image placeholders for \(images) images"
        case .spanMismatch(let why): return "Inkling-2 image tokens do not match the images: \(why)"
        }
    }
}

public enum Inkling2ImagePreprocessor {
    /// `Inkling2ImageProcessor.preprocess`: no resizing; the image is padded (bottom / right) to whole
    /// tiles with the raw value -1, rescaled by 1/255 and normalized with mean = std = 0.5, then cut
    /// into tiles. Transparent pixels are composited over the reference's 8 px grey checkerboard.
    public static func process(
        rgba: [UInt8], width: Int, height: Int, premultiplied: Bool, tile: Int
    ) -> Inkling2ProcessedImage {
        let rows = (height + tile - 1) / tile
        let cols = (width + tile - 1) / tile
        let paddedW = cols * tile
        let fill: Float = (-1 / 255 - 0.5) / 0.5
        var out = [Float](repeating: fill, count: rows * cols * tile * tile * 3)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let i = (y * width + x) * 4
                let a = Int(rgba[i + 3])
                var rgb = (Int(rgba[i]), Int(rgba[i + 1]), Int(rgba[i + 2]))
                if a < 255 {
                    let bg = ((y / 8 + x / 8) % 2 == 0) ? 204 : 153
                    func comp(_ v: Int) -> Int {
                        let straightTimesA = premultiplied ? v * 255 : v * a
                        return (straightTimesA + bg * (255 - a) + 127) / 255
                    }
                    rgb = (comp(rgb.0), comp(rgb.1), comp(rgb.2))
                }
                // Tile-major destination.
                let (ty, tx, iy, ix) = (y / tile, x / tile, y % tile, x % tile)
                let t = ty * cols + tx
                let d = ((t * tile + iy) * tile + ix) * 3
                out[d] = (Float(rgb.0) / 255 - 0.5) / 0.5
                out[d + 1] = (Float(rgb.1) / 255 - 0.5) / 0.5
                out[d + 2] = (Float(rgb.2) / 255 - 0.5) / 0.5
            }
        }
        _ = paddedW
        return Inkling2ProcessedImage(tiles: out, rows: rows, cols: cols, tileSize: tile)
    }

    #if canImport(CoreImage)
        private static let context = CIContext(options: [.useSoftwareRenderer: false])

        public static func process(_ image: UserInput.Image, tile: Int) throws -> Inkling2ProcessedImage {
            let ci = try image.asCIImage()
            let extent = ci.extent
            guard !extent.isEmpty, !extent.isInfinite else {
                throw Inkling2VisionError.unreadableImage("empty or infinite image extent")
            }
            let w = Int(extent.width.rounded())
            let h = Int(extent.height.rounded())
            var rgba = [UInt8](repeating: 0, count: w * h * 4)
            rgba.withUnsafeMutableBytes { ptr in
                context.render(
                    ci, toBitmap: ptr.baseAddress!, rowBytes: w * 4, bounds: extent, format: .RGBA8,
                    colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
            }
            return process(rgba: rgba, width: w, height: h, premultiplied: true, tile: tile)
        }
    #else
        public static func process(_ image: UserInput.Image, tile: Int) throws -> Inkling2ProcessedImage {
            throw Inkling2VisionError.unreadableImage("image decoding needs CoreImage")
        }
    #endif

    /// The images of one request as `LMInput.ProcessedImage`: all tiles concatenated, one frame
    /// `(1, rows, cols)` per image.
    public static func processedImage(_ images: [Inkling2ProcessedImage]) -> LMInput.ProcessedImage {
        let tile = images.first?.tileSize ?? 40
        let all = images.flatMap(\.tiles)
        let n = images.reduce(0) { $0 + $1.tokenCount }
        return LMInput.ProcessedImage(
            pixels: MLXArray(all, [n, tile, tile, 3]),
            frames: images.map { THW(1, $0.rows, $0.cols) })
    }
}

// MARK: - Prompt

public enum Inkling2VisionPrompt {
    /// Images ride their own message as `{"type": "image"}` parts before its text, which the chat
    /// template renders as `<|content_image|><|unused_200054|><|end_message|>`. Images not attached to
    /// a chat message go on the last user message.
    public static func anchorImages(
        chat: [Chat.Message]?, looseImages: [UserInput.Image], generated: [Message]
    ) -> (messages: [Message], images: [UserInput.Image]) {
        func parts(_ text: String, _ n: Int) -> [[String: any Sendable]] {
            var p: [[String: any Sendable]] = Array(repeating: ["type": "image"], count: n)
            if !text.isEmpty { p.append(["type": "text", "text": text]) }
            return p
        }
        var out = generated
        if let chat, chat.count == generated.count {
            var images: [UserInput.Image] = []
            for (i, message) in chat.enumerated() where !message.images.isEmpty {
                let text = (generated[i]["content"] as? String) ?? message.content
                out[i]["content"] = parts(text, message.images.count)
                images.append(contentsOf: message.images)
            }
            return (out, images)
        }
        let all = chat.map { $0.flatMap(\.images) } ?? looseImages
        guard !all.isEmpty else { return (out, []) }
        if let last = out.lastIndex(where: { ($0["role"] as? String) == "user" }) {
            out[last]["content"] = parts((out[last]["content"] as? String) ?? "", all.count)
        } else {
            out.append(["role": "user", "content": parts("", all.count)])
        }
        return (out, all)
    }

    /// Each image placeholder becomes `count` placeholders (one per tile). Returns the expanded tokens
    /// and, for every raw index (and the end), its expanded index.
    public static func expand(tokens: [Int], imageTokenId: Int, counts: [Int]) throws
        -> (tokens: [Int], rawToExpanded: [Int])
    {
        let found = tokens.reduce(0) { $0 + ($1 == imageTokenId ? 1 : 0) }
        guard found == counts.count else {
            throw Inkling2VisionError.placeholderMismatch(found: found, images: counts.count)
        }
        var out: [Int] = []
        out.reserveCapacity(tokens.count + counts.reduce(0, +))
        var map: [Int] = []
        map.reserveCapacity(tokens.count + 1)
        var next = 0
        for t in tokens {
            map.append(out.count)
            if t == imageTokenId {
                out.append(contentsOf: repeatElement(imageTokenId, count: counts[next]))
                next += 1
            } else {
                out.append(t)
            }
        }
        map.append(out.count)
        return (out, map)
    }

    public static func remapBoundaries(_ boundaries: [Int], rawToExpanded: [Int]) -> [Int] {
        boundaries.compactMap { b in (b >= 0 && b < rawToExpanded.count) ? rawToExpanded[b] : nil }
    }
}

// MARK: - Spans in a (possibly cache-trimmed) prompt

public struct Inkling2ImageSpan: Equatable, Sendable {
    /// Index of the first placeholder in the token run being prefilled.
    public let start: Int
    /// Placeholders of this image inside that run (the leading part may be in the restored prefix).
    public let length: Int
    /// The image's first tile in `LMInput.image.pixels`.
    public let tileStart: Int
    /// Tiles of this image skipped because the restored prefix already holds them.
    public let skippedTiles: Int
    public let rows: Int
    public let cols: Int
}

public enum Inkling2VisionLayout {
    /// The prompt being prefilled is the suffix after a cache restore, so its placeholders belong to
    /// the trailing images, and the first of them may already be partly in the restored prefix.
    public static func spans(tokens: [Int], imageTokenId: Int, grids: [(rows: Int, cols: Int)]) throws
        -> [Inkling2ImageSpan]
    {
        let found = tokens.reduce(0) { $0 + ($1 == imageTokenId ? 1 : 0) }
        if found == 0 { return [] }
        var tileStarts: [Int] = [0]
        for g in grids { tileStarts.append(tileStarts.last! + g.rows * g.cols) }
        let total = tileStarts.last!
        guard found <= total else {
            throw Inkling2VisionError.spanMismatch("\(found) placeholders for \(total) tiles")
        }
        // Walk backwards: the suffix holds the last `found` tiles of the image sequence.
        var firstTile = total - found
        var image = grids.count - 1
        while image > 0 && tileStarts[image] > firstTile { image -= 1 }
        var out: [Inkling2ImageSpan] = []
        var i = 0
        while i < tokens.count {
            guard tokens[i] == imageTokenId else { i += 1; continue }
            guard image < grids.count else {
                throw Inkling2VisionError.spanMismatch("more placeholder runs than images")
            }
            let skipped = firstTile - tileStarts[image]
            let need = grids[image].rows * grids[image].cols - skipped
            let end = i + need
            guard end <= tokens.count, tokens[i ..< end].allSatisfy({ $0 == imageTokenId }) else {
                throw Inkling2VisionError.spanMismatch("image \(image) needs \(need) contiguous placeholders at \(i)")
            }
            out.append(Inkling2ImageSpan(
                start: i, length: need, tileStart: tileStarts[image], skippedTiles: skipped,
                rows: grids[image].rows, cols: grids[image].cols))
            image += 1
            firstTile = image < grids.count ? tileStarts[image] : total
            i = end
        }
        guard image == grids.count else {
            throw Inkling2VisionError.spanMismatch("\(grids.count - image) image(s) without placeholders")
        }
        return out
    }
}
