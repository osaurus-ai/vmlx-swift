// DeepSeek-V4.1-Flash image input: the vision tower (ViT + Aligner), image preprocessing, the
// prompt-side placeholder expansion and the prefill-side span layout.
//
// Port of jang_tools/dsv41/vision.py (the working vmlx-py path, itself a port of the release
// `inference/vision.py` + `inference/image_processor.py`) and of the engine rules in
// vmlx_engine/models/deepseek_v4_1/dsv41_vl_preprocess.py + utils/single_batch_generator.py:
//
//   * The ViT runs full bidirectional attention over ONE image's patch grid with 2D RoPE; the
//     Aligner does a 3x3 spatial merge (9x token reduction) and projects to the text hidden size.
//     Weights are BF16 passthrough in the bundle (`vision.*`, `aligner.*`,
//     `image_{start,end,newline}`).
//   * An image becomes a span of `image_token_id` tokens: START, (IMAGE x w, NEW_LINE) x h, END.
//     The backbone overwrites those embedding rows with the aligner rows / learned delimiters,
//     routes image tokens through `gate.bias_vl`, and keeps them out of every Engram n-gram.
//   * An image is spliced into the prefill chunk that contains it (chunk-relative) and a chunk
//     never splits an image span.
//
// Everything here is reached only for a DeepSeek-V4.1 bundle whose config carries a
// `vision_config` (and only when a request carries images). DSV4-0731 and text-only V4.1 never
// touch it.

#if canImport(CoreImage)
    import CoreImage
#endif
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// `vision_config` of a DeepSeek-V4.1 bundle, plus the two top-level fields the image path needs
/// (`image_token_id` and the text hidden size the Aligner projects to).
public struct DeepseekV41VisionConfiguration: Codable, Sendable, Equatable {
    public var numHiddenLayers: Int
    public var hiddenSize: Int
    public var numAttentionHeads: Int
    public var intermediateSize: Int
    public var patchSize: Int
    public var ropeTheta: Float
    public var downsampleRatio: Int
    public var maxImageTokens: Int
    public var minPixels: Int
    public var maxWHRatio: Double?
    public var eps: Float = 1e-6
    /// Top-level `image_token_id` (129264 on the 1017 bundles).
    public var imageTokenId: Int = 129_264
    /// Text hidden size (`text_config.hidden_size`).
    public var textHiddenSize: Int = 0

    public var headDim: Int { hiddenSize / numAttentionHeads }
    /// 2D RoPE table width: half the head dim (h positions in the first half, w in the second).
    public var ropeDim: Int { headDim / 2 }

    enum CodingKeys: String, CodingKey {
        case numHiddenLayers = "num_hidden_layers"
        case hiddenSize = "hidden_size"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case patchSize = "patch_size"
        case ropeTheta = "rope_theta"
        case downsampleRatio = "downsample_ratio"
        case maxImageTokens = "max_image_tokens"
        case minPixels = "min_pixels"
        case maxWHRatio = "max_wh_ratio"
    }

    public init(
        numHiddenLayers: Int, hiddenSize: Int, numAttentionHeads: Int, intermediateSize: Int,
        patchSize: Int, ropeTheta: Float = 10_000, downsampleRatio: Int, maxImageTokens: Int = 1024,
        minPixels: Int = 544 * 544, maxWHRatio: Double? = nil, imageTokenId: Int = 129_264,
        textHiddenSize: Int
    ) {
        self.numHiddenLayers = numHiddenLayers
        self.hiddenSize = hiddenSize
        self.numAttentionHeads = numAttentionHeads
        self.intermediateSize = intermediateSize
        self.patchSize = patchSize
        self.ropeTheta = ropeTheta
        self.downsampleRatio = downsampleRatio
        self.maxImageTokens = maxImageTokens
        self.minPixels = minPixels
        self.maxWHRatio = maxWHRatio
        self.imageTokenId = imageTokenId
        self.textHiddenSize = textHiddenSize
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        patchSize = try c.decode(Int.self, forKey: .patchSize)
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        downsampleRatio = try c.decode(Int.self, forKey: .downsampleRatio)
        maxImageTokens = try c.decodeIfPresent(Int.self, forKey: .maxImageTokens) ?? 1024
        minPixels = try c.decodeIfPresent(Int.self, forKey: .minPixels) ?? 544 * 544
        maxWHRatio = try c.decodeIfPresent(Double.self, forKey: .maxWHRatio)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(patchSize, forKey: .patchSize)
        try c.encode(ropeTheta, forKey: .ropeTheta)
        try c.encode(downsampleRatio, forKey: .downsampleRatio)
        try c.encode(maxImageTokens, forKey: .maxImageTokens)
        try c.encode(minPixels, forKey: .minPixels)
        try c.encodeIfPresent(maxWHRatio, forKey: .maxWHRatio)
    }

    /// `VMLX_DSV41_VL=0|false|off|no` builds a vision-config V4.1 bundle text-only (no tower, no
    /// `bias_vl`; the processor then ignores request images and renders the text alone, and hosts
    /// read the same switch to stop offering image input). Any other value, or unset, keeps the tower.
    /// Same switch as the Python engine.
    public static func enabled(
        environment env: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        let raw = (env["VMLX_DSV41_VL"] ?? "").trimmingCharacters(in: .whitespaces).lowercased()
        return !["0", "false", "off", "no"].contains(raw)
    }
}

// MARK: - Token types

/// Per-token type inside an image span (reference `TEXT=-1, IMAGE_START, IMAGE, IMAGE_NEW_LINE,
/// IMAGE_END = range(4)`).
public enum DeepseekV41ImageTokenType: Int, Sendable {
    case text = -1
    case start = 0
    case image = 1
    case newline = 2
    case end = 3
}

// MARK: - Grid planning (pure; exact port of vision.py)

/// The geometry of one preprocessed image: the padded pixel canvas, the ViT patch grid and the
/// LLM grid after the `downsample_ratio` merge.
public struct DeepseekV41ImageGrid: Sendable, Equatable {
    public let bestW: Int
    public let bestH: Int
    public let vitH: Int
    public let vitW: Int
    public let llmH: Int
    public let llmW: Int

    /// Tokens the image occupies in the prompt: `llmH * (llmW + 1) + 2`.
    public var tokenCount: Int { DeepseekV41ImagePlanner.numImageTokens(llmH: llmH, llmW: llmW) }
    public var patchCount: Int { vitH * vitW }

    /// Span token types in order (reference `image_token_types`).
    public var tokenTypes: [DeepseekV41ImageTokenType] {
        var types: [DeepseekV41ImageTokenType] = [.start]
        types.reserveCapacity(tokenCount)
        for _ in 0 ..< llmH {
            types.append(contentsOf: repeatElement(.image, count: llmW))
            types.append(.newline)
        }
        types.append(.end)
        return types
    }

    /// The grid a ViT patch grid implies (ceil division by the merge ratio).
    public init(vitH: Int, vitW: Int, patchSize: Int, downsampleRatio r: Int) {
        self.vitH = vitH
        self.vitW = vitW
        self.bestH = vitH * patchSize
        self.bestW = vitW * patchSize
        self.llmH = (vitH + r - 1) / r
        self.llmW = (vitW + r - 1) / r
    }

    public init(bestW: Int, bestH: Int, vitH: Int, vitW: Int, llmH: Int, llmW: Int) {
        self.bestW = bestW
        self.bestH = bestH
        self.vitH = vitH
        self.vitW = vitW
        self.llmH = llmH
        self.llmW = llmW
    }
}

public enum DeepseekV41ImagePlanner {
    public static func numImageTokens(llmH: Int, llmW: Int) -> Int { llmH * (llmW + 1) + 2 }

    /// `math.ceil((best // patch) / r)` per side.
    static func llmGrid(bestH: Int, bestW: Int, patch: Int, r: Int) -> (Int, Int) {
        let h = bestH / patch
        let w = bestW / patch
        return ((h + r - 1) / r, (w + r - 1) / r)
    }

    /// `solve_resize_ratio`: the largest canvas whose token count fits `maxTokens`.
    static func solveResizeRatio(
        height: Double, width: Double, patch: Int, r: Int, maxTokens: Int
    ) -> (bestH: Int, bestW: Int) {
        let ratio = height / width
        let maxW = (Double(maxTokens - 2) / ratio + 0.25).squareRoot() - 0.5
        let maxH = maxW * ratio
        let cell = patch * r
        if maxW < 1.0 { return ((maxTokens - 2) / 2 * cell, cell) }
        if maxH < 1.0 { return (cell, (maxTokens - 3) * cell) }
        let beta = min(
            maxW.rounded(.down) * Double(cell) / width, maxH.rounded(.down) * Double(cell) / height)
        return (
            Int((height * beta / Double(patch)).rounded(.down)) * patch,
            Int((width * beta / Double(patch)).rounded(.down)) * patch)
    }

    /// `plan_image_grid` for a decoded image of `width` x `height` pixels.
    public static func plan(
        width: Int, height: Int, config: DeepseekV41VisionConfiguration
    ) -> DeepseekV41ImageGrid {
        let p = config.patchSize
        let r = config.downsampleRatio
        var w = Double(width)
        var h = Double(height)
        if let maxRatio = config.maxWHRatio, w > h * maxRatio {
            w = h * maxRatio
        }
        if w * h > 0, w * h < Double(config.minPixels) {
            let ratio = pow(Double(config.minPixels) / (w * h), 0.5)
            w = Double(Int(w * ratio))  // Python int(): truncation toward zero
            h = Double(Int(h * ratio))
        }
        var bestW = Int((w / Double(p)).rounded(.up)) * p
        var bestH = Int((h / Double(p)).rounded(.up)) * p
        var (llmH, llmW) = llmGrid(bestH: bestH, bestW: bestW, patch: p, r: r)
        if numImageTokens(llmH: llmH, llmW: llmW) > config.maxImageTokens {
            (bestH, bestW) = solveResizeRatio(
                height: h, width: w, patch: p, r: r, maxTokens: config.maxImageTokens)
            (llmH, llmW) = llmGrid(bestH: bestH, bestW: bestW, patch: p, r: r)
        }
        return DeepseekV41ImageGrid(
            bestW: bestW, bestH: bestH, vitH: bestH / p, vitW: bestW / p, llmH: llmH, llmW: llmW)
    }
}

// MARK: - Pixels (PIL-exact resize + pad, then normalize + patchify)

/// An 8-bit RGB raster, row-major, top row first.
public struct DeepseekV41RGBImage: Sendable, Equatable {
    public let width: Int
    public let height: Int
    /// `width * height * 3` bytes.
    public var pixels: [UInt8]

    public init(width: Int, height: Int, pixels: [UInt8]) {
        precondition(pixels.count == width * height * 3, "RGB buffer size mismatch")
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    public init(width: Int, height: Int, fill: (UInt8, UInt8, UInt8)) {
        self.width = width
        self.height = height
        var px = [UInt8](repeating: 0, count: width * height * 3)
        for i in stride(from: 0, to: px.count, by: 3) {
            px[i] = fill.0
            px[i + 1] = fill.1
            px[i + 2] = fill.2
        }
        self.pixels = px
    }
}

/// Pillow's `Image.resize(..., BICUBIC)` for 8-bit RGB (libImaging/Resample.c: separable
/// convolution, horizontal pass first, 22-bit fixed-point coefficients, an 8-bit intermediate),
/// and `ImageOps.pad` / `ImageOps.contain`. Pure CPU so the planned canvas, and the pixels, match
/// the reference without CoreImage resampling (which differs, and may run on the GPU).
public enum DeepseekV41PILResample {
    static let precisionBits = 32 - 8 - 2

    static func bicubic(_ x: Double) -> Double {
        let a = -0.5
        let x = abs(x)
        if x < 1.0 { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1 }
        if x < 2.0 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0
    }

    /// `precompute_coeffs` + `normalize_coeffs_8bpc`: per output index the first input index, the
    /// tap count, and the fixed-point taps.
    static func coefficients(inSize: Int, outSize: Int) -> (bounds: [(Int, Int)], ksize: Int, kk: [Int]) {
        let support0 = 2.0  // bicubic
        let scale = Double(inSize) / Double(outSize)
        let filterscale = max(scale, 1.0)
        let support = support0 * filterscale
        let ksize = Int(support.rounded(.up)) * 2 + 1
        var bounds: [(Int, Int)] = []
        bounds.reserveCapacity(outSize)
        var kk = [Int](repeating: 0, count: outSize * ksize)
        var k = [Double](repeating: 0, count: ksize)
        for xx in 0 ..< outSize {
            let center = (Double(xx) + 0.5) * scale
            var ww = 0.0
            let ss = 1.0 / filterscale
            var xmin = Int(center - support + 0.5)  // C (int): truncation toward zero
            if xmin < 0 { xmin = 0 }
            var xmax = Int(center + support + 0.5)
            if xmax > inSize { xmax = inSize }
            xmax -= xmin
            for x in 0 ..< ksize { k[x] = 0 }
            for x in 0 ..< xmax {
                let w = bicubic((Double(x + xmin) - center + 0.5) * ss)
                k[x] = w
                ww += w
            }
            for x in 0 ..< xmax where ww != 0.0 { k[x] /= ww }
            let one = Double(1 << precisionBits)
            for x in 0 ..< ksize {
                kk[xx * ksize + x] = k[x] < 0 ? Int(-0.5 + k[x] * one) : Int(0.5 + k[x] * one)
            }
            bounds.append((xmin, xmax))
        }
        return (bounds, ksize, kk)
    }

    @inline(__always) static func clip8(_ v: Int) -> UInt8 {
        let s = v >> precisionBits
        return UInt8(clamping: min(max(s, 0), 255))
    }

    /// `Image.resize((outW, outH), BICUBIC)`.
    public static func resize(_ image: DeepseekV41RGBImage, width outW: Int, height outH: Int)
        -> DeepseekV41RGBImage
    {
        if image.width == outW && image.height == outH { return image }
        let needH = outW != image.width
        let needV = outH != image.height
        let horiz = coefficients(inSize: image.width, outSize: outW)
        let vert = coefficients(inSize: image.height, outSize: outH)
        let half = 1 << (precisionBits - 1)

        var src = image.pixels
        var srcW = image.width
        var vbounds = vert.bounds
        if needH {
            let yFirst = vert.bounds[0].0
            let yLast = vert.bounds[outH - 1].0 + vert.bounds[outH - 1].1
            vbounds = vert.bounds.map { ($0.0 - yFirst, $0.1) }
            let rows = yLast - yFirst
            var tmp = [UInt8](repeating: 0, count: outW * rows * 3)
            for yy in 0 ..< rows {
                let rowBase = (yy + yFirst) * srcW * 3
                for xx in 0 ..< outW {
                    let (xmin, xmax) = horiz.bounds[xx]
                    let kBase = xx * horiz.ksize
                    var s0 = half, s1 = half, s2 = half
                    for x in 0 ..< xmax {
                        let kv = horiz.kk[kBase + x]
                        let p = rowBase + (x + xmin) * 3
                        s0 += Int(src[p]) * kv
                        s1 += Int(src[p + 1]) * kv
                        s2 += Int(src[p + 2]) * kv
                    }
                    let o = (yy * outW + xx) * 3
                    tmp[o] = clip8(s0)
                    tmp[o + 1] = clip8(s1)
                    tmp[o + 2] = clip8(s2)
                }
            }
            src = tmp
            srcW = outW
        }
        if needV {
            var out = [UInt8](repeating: 0, count: srcW * outH * 3)
            for yy in 0 ..< outH {
                let (ymin, ymax) = vbounds[yy]
                let kBase = yy * vert.ksize
                for xx in 0 ..< srcW {
                    var s0 = half, s1 = half, s2 = half
                    for y in 0 ..< ymax {
                        let kv = vert.kk[kBase + y]
                        let p = ((y + ymin) * srcW + xx) * 3
                        s0 += Int(src[p]) * kv
                        s1 += Int(src[p + 1]) * kv
                        s2 += Int(src[p + 2]) * kv
                    }
                    let o = (yy * srcW + xx) * 3
                    out[o] = clip8(s0)
                    out[o + 1] = clip8(s1)
                    out[o + 2] = clip8(s2)
                }
            }
            return DeepseekV41RGBImage(width: srcW, height: outH, pixels: out)
        }
        return DeepseekV41RGBImage(width: srcW, height: image.height, pixels: src)
    }

    /// Python's `round()` (half to even).
    static func pyRound(_ x: Double) -> Int { Int(x.rounded(.toNearestOrEven)) }

    /// `ImageOps.contain` target size.
    static func containSize(width: Int, height: Int, target: (Int, Int)) -> (Int, Int) {
        let imRatio = Double(width) / Double(height)
        let destRatio = Double(target.0) / Double(target.1)
        var size = target
        if imRatio != destRatio {
            if imRatio > destRatio {
                let newH = pyRound(Double(height) / Double(width) * Double(target.0))
                if newH != target.1 { size = (target.0, newH) }
            } else {
                let newW = pyRound(Double(width) / Double(height) * Double(target.1))
                if newW != target.0 { size = (newW, target.1) }
            }
        }
        return size
    }

    /// `ImageOps.pad(image, (width, height), color=(127,127,127))` (bicubic, centered).
    public static func pad(_ image: DeepseekV41RGBImage, width: Int, height: Int, gray: UInt8 = 127)
        -> DeepseekV41RGBImage
    {
        let (cw, ch) = containSize(width: image.width, height: image.height, target: (width, height))
        let resized = resize(image, width: cw, height: ch)
        if cw == width && ch == height { return resized }
        var out = DeepseekV41RGBImage(width: width, height: height, fill: (gray, gray, gray))
        var x0 = 0, y0 = 0
        if cw != width {
            x0 = pyRound(Double(width - cw) * 0.5)
        } else {
            y0 = pyRound(Double(height - ch) * 0.5)
        }
        for y in 0 ..< ch {
            let dy = y + y0
            guard dy >= 0 && dy < height else { continue }
            for x in 0 ..< cw {
                let dx = x + x0
                guard dx >= 0 && dx < width else { continue }
                let s = (y * cw + x) * 3
                let d = (dy * width + dx) * 3
                out.pixels[d] = resized.pixels[s]
                out.pixels[d + 1] = resized.pixels[s + 1]
                out.pixels[d + 2] = resized.pixels[s + 2]
            }
        }
        return out
    }
}

/// One image after preprocessing: its grid and its ViT patches (`[patchCount, 3, p, p]`, values in
/// [-1, 1], row-major over the patch grid; channel-major inside a patch like the reference).
public struct DeepseekV41PreprocessedImage: Sendable {
    public let grid: DeepseekV41ImageGrid
    /// `patchCount * 3 * p * p` floats.
    public let patches: [Float]
}

public enum DeepseekV41ImagePreprocessor {
    /// Reference `load_image` on a decoded RGB raster.
    public static func preprocess(
        _ image: DeepseekV41RGBImage, config: DeepseekV41VisionConfiguration
    ) -> DeepseekV41PreprocessedImage {
        let grid = DeepseekV41ImagePlanner.plan(
            width: image.width, height: image.height, config: config)
        let canvas: DeepseekV41RGBImage
        if let maxRatio = config.maxWHRatio, Double(image.width) >= maxRatio * Double(image.height) {
            canvas = DeepseekV41PILResample.resize(image, width: grid.bestW, height: grid.bestH)
        } else {
            canvas = DeepseekV41PILResample.pad(image, width: grid.bestW, height: grid.bestH)
        }
        return DeepseekV41PreprocessedImage(grid: grid, patches: patchify(canvas, grid: grid, patch: config.patchSize))
    }

    /// `(x/255 - 0.5)/0.5` then `[3,H,W] -> [nH, nW, 3, p, p]` flattened.
    static func patchify(_ canvas: DeepseekV41RGBImage, grid: DeepseekV41ImageGrid, patch p: Int) -> [Float] {
        let nH = grid.vitH, nW = grid.vitW
        let W = canvas.width
        var out = [Float](repeating: 0, count: nH * nW * 3 * p * p)
        var o = 0
        for ph in 0 ..< nH {
            for pw in 0 ..< nW {
                for c in 0 ..< 3 {
                    for y in 0 ..< p {
                        let row = (ph * p + y) * W
                        for x in 0 ..< p {
                            let v = Float(canvas.pixels[(row + pw * p + x) * 3 + c]) / 255.0
                            out[o] = (v - 0.5) / 0.5
                            o += 1
                        }
                    }
                }
            }
        }
        return out
    }

    #if canImport(CoreImage)
    private static let unmanagedSoftwareContext = CIContext(options: [
        .workingColorSpace: NSNull(), .outputColorSpace: NSNull(), .useSoftwareRenderer: true,
    ])

    /// Decode a request image to 8-bit RGB without colour management (PIL `convert("RGB")` keeps
    /// the stored values; it never applies an ICC profile). Rendered on the CPU.
    public static func decodeRGB(_ image: UserInput.Image) throws -> DeepseekV41RGBImage {
        let ci = try image.asCIImage()
        let extent = ci.extent
        guard !extent.isEmpty, !extent.isInfinite else {
            throw DeepseekV41VisionError.unreadableImage("empty or infinite image extent")
        }
        let w = Int(extent.width.rounded())
        let h = Int(extent.height.rounded())
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        rgba.withUnsafeMutableBytes { ptr in
            unmanagedSoftwareContext.render(
                ci, toBitmap: ptr.baseAddress!, rowBytes: w * 4, bounds: extent,
                format: .RGBA8, colorSpace: nil)
        }
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0 ..< (w * h) {
            rgb[i * 3] = rgba[i * 4]
            rgb[i * 3 + 1] = rgba[i * 4 + 1]
            rgb[i * 3 + 2] = rgba[i * 4 + 2]
        }
        return DeepseekV41RGBImage(width: w, height: h, pixels: rgb)
    }
    #else
        /// Image decoding needs CoreImage (Apple platforms); elsewhere image requests are refused.
        public static func decodeRGB(_ image: UserInput.Image) throws -> DeepseekV41RGBImage {
            throw DeepseekV41VisionError.unreadableImage("image decoding needs CoreImage")
        }
    #endif
}

public enum DeepseekV41VisionError: Error, CustomStringConvertible {
    case unreadableImage(String)
    case placeholderMismatch(found: Int, images: Int)
    case placeholderInText
    case spanMismatch(String)
    case imageAtPromptEnd

    public var description: String {
        switch self {
        case .unreadableImage(let why):
            return "DeepSeek-V4.1 image input: cannot read the image (\(why))"
        case .placeholderMismatch(let found, let images):
            return "DeepSeek-V4.1 image input: the prompt has \(found) image placeholder(s) but the request carries \(images) image(s)"
        case .placeholderInText:
            return "DeepSeek-V4.1 image input: message text contains the image placeholder token; send images as image content parts"
        case .spanMismatch(let why):
            return "DeepSeek-V4.1 image input: image spans do not match the prompt tokens (\(why))"
        case .imageAtPromptEnd:
            return "DeepSeek-V4.1 image input: an image span ends the prompt; the final prompt token must be text"
        }
    }
}

// MARK: - Prompt assembly (bundle encoding.py `process_image_messages`)

public enum DeepseekV41VisionPrompt {
    /// `encoding.py` IMAGE_PLACEHOLDER (token 129264).
    public static let placeholder = "<｜deepseek_image｜>"

    /// The text an image-bearing message renders: every text part and one placeholder per image,
    /// in content order, joined with "\n\n" (`process_image_messages`: image blocks become
    /// placeholder text blocks, then `content = "\n\n".join(texts)`). Without an explicit part
    /// order the images go first, then the text (the engine's re-anchoring rule).
    public static func renderContent(
        text: String, imageCount: Int, parts: [Chat.ContentPart]?
    ) throws -> String {
        if let parts, parts.filter({ $0 == .image }).count == imageCount {
            var texts: [String] = []
            for part in parts {
                switch part {
                case .text(let t):
                    if t.contains(placeholder) { throw DeepseekV41VisionError.placeholderInText }
                    texts.append(t)
                case .image:
                    texts.append(placeholder)
                case .video, .audio:
                    continue
                }
            }
            return texts.joined(separator: "\n\n")
        }
        if text.contains(placeholder) { throw DeepseekV41VisionError.placeholderInText }
        var texts = Array(repeating: placeholder, count: imageCount)
        if !text.isEmpty { texts.append(text) }
        return texts.joined(separator: "\n\n")
    }

    /// Rewrite the generated message dicts so each image appears as a placeholder in the turn it
    /// was sent in (`images stay in the turn they were sent in`), and return the images in prompt
    /// order. `chat` and `generated` correspond 1:1 (the DSV4 generator is the default one); when
    /// they do not, every image goes into the last user turn (the engine's fallback).
    public static func anchorImages(
        chat: [Chat.Message]?, looseImages: [UserInput.Image], generated: [Message]
    ) throws -> (messages: [Message], images: [UserInput.Image]) {
        var out = generated
        var images: [UserInput.Image] = []
        if let chat, chat.count == generated.count {
            for (i, message) in chat.enumerated() where !message.images.isEmpty {
                if message.role == .tool {
                    // Tool-role images would need tool_result block nesting; not rendered.
                    continue
                }
                let text = (generated[i]["content"] as? String) ?? message.content
                out[i]["content"] = try renderContent(
                    text: text, imageCount: message.images.count, parts: message.contentParts)
                images.append(contentsOf: message.images)
            }
            return (out, images)
        }
        let all = chat.map { $0.filter { $0.role != .tool }.flatMap(\.images) } ?? looseImages
        guard !all.isEmpty else { return (out, []) }
        if let last = out.lastIndex(where: { ($0["role"] as? String) == "user" }) {
            let text = (out[last]["content"] as? String) ?? ""
            out[last]["content"] = try renderContent(text: text, imageCount: all.count, parts: nil)
        } else {
            out.append([
                "role": "user",
                "content": try renderContent(text: "", imageCount: all.count, parts: nil),
            ])
        }
        return (out, all)
    }

    /// Expand each placeholder token into its image span (`prepare_vl_inputs`): returns the
    /// expanded tokens and, for every raw index, its expanded index (to remap cache boundaries).
    public static func expand(
        tokens: [Int], imageTokenId: Int, grids: [DeepseekV41ImageGrid]
    ) throws -> (tokens: [Int], rawToExpanded: [Int]) {
        let found = tokens.filter { $0 == imageTokenId }.count
        guard found == grids.count else {
            throw DeepseekV41VisionError.placeholderMismatch(found: found, images: grids.count)
        }
        var out: [Int] = []
        out.reserveCapacity(tokens.count + grids.reduce(0) { $0 + $1.tokenCount })
        var map: [Int] = []
        map.reserveCapacity(tokens.count + 1)
        var next = 0
        for t in tokens {
            map.append(out.count)
            if t == imageTokenId {
                out.append(contentsOf: repeatElement(imageTokenId, count: grids[next].tokenCount))
                next += 1
            } else {
                out.append(t)
            }
        }
        map.append(out.count)
        return (out, map)
    }

    /// Cache boundaries are raw-token prefix lengths; a boundary `b` maps to `rawToExpanded[b]`.
    public static func remapBoundaries(_ boundaries: [Int], rawToExpanded: [Int]) -> [Int] {
        boundaries.compactMap { b in (b >= 0 && b < rawToExpanded.count) ? rawToExpanded[b] : nil }
    }

    /// The LMInput image payload: every image's patches concatenated (`[sum patches, 3, p, p]`,
    /// bf16 like the reference) and one THW grid per image (t = 1). The media salt hashes exactly
    /// these bytes and grids.
    public static func processedImage(
        _ images: [DeepseekV41PreprocessedImage], patchSize p: Int
    ) -> LMInput.ProcessedImage {
        let total = images.reduce(0) { $0 + $1.grid.patchCount }
        var flat: [Float] = []
        flat.reserveCapacity(total * 3 * p * p)
        for image in images { flat.append(contentsOf: image.patches) }
        let pixels = MLXArray(flat, [total, 3, p, p]).asType(.bfloat16)
        return LMInput.ProcessedImage(
            pixels: pixels, frames: images.map { THW(1, $0.grid.vitH, $0.grid.vitW) })
    }
}

// MARK: - Prefill span layout

/// One image span inside a token list.
public struct DeepseekV41ImageSpan: Sendable, Equatable {
    /// Index of the span's first token in the token list.
    public let start: Int
    public let grid: DeepseekV41ImageGrid
    /// Which image of the request (index into its frames).
    public let imageIndex: Int
    /// Rows of the concatenated pixel tensor holding this image's patches.
    public let patchRows: Range<Int>

    public var end: Int { start + grid.tokenCount }
}

public enum DeepseekV41VisionLayout {
    /// Grids of the request's images, from the `ProcessedImage` frames.
    public static func grids(
        frames: [THW], config: DeepseekV41VisionConfiguration
    ) -> [DeepseekV41ImageGrid] {
        frames.map {
            DeepseekV41ImageGrid(
                vitH: $0.h, vitW: $0.w, patchSize: config.patchSize,
                downsampleRatio: config.downsampleRatio)
        }
    }

    /// Locate the image spans in `tokens`. The token list holds either every image of the request
    /// or (a suffix after a cache restore) only the LAST k of them; spans are matched in order and
    /// each must be a complete run of `imageTokenId`. Adjacent images form one run.
    public static func spans(
        tokens: [Int], imageTokenId: Int, grids: [DeepseekV41ImageGrid]
    ) throws -> [DeepseekV41ImageSpan] {
        let placeholderCount = tokens.reduce(0) { $0 + ($1 == imageTokenId ? 1 : 0) }
        if placeholderCount == 0 { return [] }
        // Which trailing subset of the images do these tokens carry?
        var firstImage: Int? = nil
        var suffixTokens = 0
        for k in stride(from: grids.count - 1, through: 0, by: -1) {
            suffixTokens += grids[k].tokenCount
            if suffixTokens == placeholderCount { firstImage = k; break }
            if suffixTokens > placeholderCount { break }
        }
        guard let firstImage else {
            throw DeepseekV41VisionError.spanMismatch(
                "\(placeholderCount) placeholder tokens match no trailing set of the \(grids.count) image(s)")
        }
        var patchOffsets: [Int] = [0]
        for g in grids { patchOffsets.append(patchOffsets.last! + g.patchCount) }

        var out: [DeepseekV41ImageSpan] = []
        var image = firstImage
        var i = 0
        while i < tokens.count {
            guard tokens[i] == imageTokenId else { i += 1; continue }
            guard image < grids.count else {
                throw DeepseekV41VisionError.spanMismatch("more placeholder runs than images")
            }
            let g = grids[image]
            let end = i + g.tokenCount
            guard end <= tokens.count, tokens[i ..< end].allSatisfy({ $0 == imageTokenId }) else {
                throw DeepseekV41VisionError.spanMismatch(
                    "image \(image) needs \(g.tokenCount) contiguous placeholders at \(i)")
            }
            out.append(DeepseekV41ImageSpan(
                start: i, grid: g, imageIndex: image,
                patchRows: patchOffsets[image] ..< patchOffsets[image + 1]))
            image += 1
            i = end
        }
        guard image == grids.count else {
            throw DeepseekV41VisionError.spanMismatch("\(grids.count - image) image(s) without a span")
        }
        return out
    }

    /// The engine's `_dsv41_vl_chunk_len`: a chunk `[pos, pos + n)` whose end would fall inside an
    /// image span stops right before that image, or (when the image starts the chunk) takes the
    /// whole image. Returns the adjusted length (>= 1).
    public static func chunkLength(pos: Int, proposed n: Int, spans: [DeepseekV41ImageSpan]) -> Int {
        var end = pos + n
        for span in spans where span.start < end && end < span.end {
            end = span.start > pos ? span.start : span.end
        }
        return max(1, end - pos)
    }

    /// Prefill chunk lengths over `tokenCount` prompt tokens: every chunk the model's `prepare`
    /// runs, leaving a text-only remainder of at most `stepSize` tokens (>= 1) for the caller's final
    /// forward. Throws when an image span reaches the final prompt token.
    public static func prefillChunks(
        tokenCount: Int, stepSize: Int, outerStep: Int, spans: [DeepseekV41ImageSpan]
    ) throws -> [Range<Int>] {
        let lastImageEnd = spans.map(\.end).max() ?? 0
        guard lastImageEnd < tokenCount else { throw DeepseekV41VisionError.imageAtPromptEnd }
        var chunks: [Range<Int>] = []
        var pos = 0
        while tokenCount - pos > stepSize || pos < lastImageEnd {
            let proposed = min(outerStep, tokenCount - pos - 1)
            let n = chunkLength(pos: pos, proposed: proposed, spans: spans)
            guard pos + n <= tokenCount - 1 else { throw DeepseekV41VisionError.imageAtPromptEnd }
            chunks.append(pos ..< pos + n)
            pos += n
        }
        return chunks
    }
}

// MARK: - Vision tower (module names mirror the bundle: vision.*, aligner.*, image_*)

/// Reference `_RMSNorm`: fp32 normalize, weight multiply in fp32, cast back.
final class DeepseekV41VisionRMSNorm: Module, UnaryLayer {
    let eps: Float
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let n = xf * rsqrt(mean(xf * xf, axis: -1, keepDims: true) + eps)
        return (weight * n).asType(x.dtype)
    }
}

final class DeepseekV41PatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Linear

    init(_ c: DeepseekV41VisionConfiguration) {
        self._proj.wrappedValue = Linear(3 * c.patchSize * c.patchSize, c.hiddenSize, bias: true)
    }

    func callAsFunction(_ patches: MLXArray) -> MLXArray {
        proj(patches.reshaped(patches.dim(0), -1))
    }
}

final class DeepseekV41VisionAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float
    @ModuleInfo(key: "wqkv") var wqkv: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    init(_ c: DeepseekV41VisionConfiguration) {
        heads = c.numAttentionHeads
        headDim = c.headDim
        scale = pow(Float(c.headDim), -0.5)
        self._wqkv.wrappedValue = Linear(c.hiddenSize, 3 * c.hiddenSize, bias: true)
        self._wo.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let n = x.dim(0)
        let parts = split(wqkv(x), parts: 3, axis: -1)
        let q = DeepseekV41VisionMath.rotary(parts[0].reshaped(n, heads, headDim), cos: cos, sin: sin)
        let k = DeepseekV41VisionMath.rotary(parts[1].reshaped(n, heads, headDim), cos: cos, sin: sin)
        let v = parts[2].reshaped(n, heads, headDim)
        // [1, heads, n, hd]: full bidirectional attention over one image.
        let o = MLXFast.scaledDotProductAttention(
            queries: q.transposed(1, 0, 2).expandedDimensions(axis: 0),
            keys: k.transposed(1, 0, 2).expandedDimensions(axis: 0),
            values: v.transposed(1, 0, 2).expandedDimensions(axis: 0),
            scale: scale, mask: nil)
        return wo(o[0].transposed(1, 0, 2).reshaped(n, -1))
    }
}

final class DeepseekV41VisionMLP: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear

    init(_ c: DeepseekV41VisionConfiguration) {
        self._w1.wrappedValue = Linear(c.hiddenSize, 2 * c.intermediateSize, bias: false)
        self._w2.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gu = split(w1(x), parts: 2, axis: -1)
        return w2(silu(gu[0]) * gu[1])
    }
}

final class DeepseekV41VisionBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: DeepseekV41VisionRMSNorm
    @ModuleInfo(key: "attn") var attn: DeepseekV41VisionAttention
    @ModuleInfo(key: "norm2") var norm2: DeepseekV41VisionRMSNorm
    @ModuleInfo(key: "mlp") var mlp: DeepseekV41VisionMLP

    init(_ c: DeepseekV41VisionConfiguration) {
        self._norm1.wrappedValue = DeepseekV41VisionRMSNorm(dimensions: c.hiddenSize, eps: c.eps)
        self._attn.wrappedValue = DeepseekV41VisionAttention(c)
        self._norm2.wrappedValue = DeepseekV41VisionRMSNorm(dimensions: c.hiddenSize, eps: c.eps)
        self._mlp.wrappedValue = DeepseekV41VisionMLP(c)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let h = x + attn(norm1(x), cos: cos, sin: sin)
        return h + mlp(norm2(h))
    }
}

final class DeepseekV41ViT: Module {
    let ropeDim: Int
    let ropeTheta: Float
    @ModuleInfo(key: "patch_embed") var patchEmbed: DeepseekV41PatchEmbed
    @ModuleInfo(key: "blocks") var blocks: [DeepseekV41VisionBlock]
    @ModuleInfo(key: "norm") var norm: DeepseekV41VisionRMSNorm

    init(_ c: DeepseekV41VisionConfiguration) {
        ropeDim = c.ropeDim
        ropeTheta = c.ropeTheta
        self._patchEmbed.wrappedValue = DeepseekV41PatchEmbed(c)
        self._blocks.wrappedValue = (0 ..< c.numHiddenLayers).map { _ in DeepseekV41VisionBlock(c) }
        self._norm.wrappedValue = DeepseekV41VisionRMSNorm(dimensions: c.hiddenSize, eps: c.eps)
    }

    func callAsFunction(_ patches: MLXArray, vitH: Int, vitW: Int) -> MLXArray {
        var x = patchEmbed(patches)
        let (cos, sin) = DeepseekV41VisionMath.cosSin(
            vitH: vitH, vitW: vitW, dim: ropeDim, theta: ropeTheta)
        for block in blocks { x = block(x, cos: cos, sin: sin) }
        return norm(x)
    }
}

/// 3x3 spatial merge (release `F.unfold`) + two-layer projector to the text hidden size.
final class DeepseekV41Aligner: Module {
    let r: Int
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear

    init(_ c: DeepseekV41VisionConfiguration) {
        r = c.downsampleRatio
        self._w1.wrappedValue = Linear(c.hiddenSize * r * r, c.textHiddenSize, bias: true)
        self._w2.wrappedValue = Linear(c.textHiddenSize, c.textHiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray, vitH: Int, vitW: Int) -> MLXArray {
        w2(DeepseekV41VisionMath.geluErf(w1(DeepseekV41VisionMath.merge(x, vitH: vitH, vitW: vitW, r: r))))
    }
}

enum DeepseekV41VisionMath {
    /// 2D RoPE tables (reference `_vision_cos_sin`): `[n, 1, dim]` with the first dim/2 columns
    /// rotating by the patch row and the last dim/2 by the patch column. fp32.
    static func cosSin(vitH: Int, vitW: Int, dim: Int, theta: Float) -> (MLXArray, MLXArray) {
        let half = dim / 2
        let inv: [Float] = (0 ..< half).map { 1.0 / pow(theta, Float(2 * $0) / Float(dim)) }
        let n = vitH * vitW
        var cosv = [Float](repeating: 0, count: n * dim)
        var sinv = [Float](repeating: 0, count: n * dim)
        for hp in 0 ..< vitH {
            for wp in 0 ..< vitW {
                let row = (hp * vitW + wp) * dim
                for j in 0 ..< half {
                    let fh = Float(hp) * inv[j]
                    let fw = Float(wp) * inv[j]
                    cosv[row + j] = Foundation.cos(fh)
                    sinv[row + j] = Foundation.sin(fh)
                    cosv[row + half + j] = Foundation.cos(fw)
                    sinv[row + half + j] = Foundation.sin(fw)
                }
            }
        }
        return (MLXArray(cosv, [n, 1, dim]), MLXArray(sinv, [n, 1, dim]))
    }

    /// x `[n, heads, headDim]`; fp32 rotate-half, cast back.
    static func rotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let halves = split(xf, parts: 2, axis: -1)
        let x1 = halves[0], x2 = halves[1]
        return concatenated([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis: -1).asType(x.dtype)
    }

    static func geluErf(_ x: MLXArray) -> MLXArray {
        0.5 * x * (1.0 + erf(x / Float(2.0).squareRoot()))
    }

    /// `[n, C]` -> `[ceil(h/r) * ceil(w/r), C*r*r]`, zero-padding bottom/right; column order
    /// `c*r*r + kh*r + kw` (F.unfold on `[C, H, W]`).
    static func merge(_ x: MLXArray, vitH: Int, vitW: Int, r: Int) -> MLXArray {
        let C = x.dim(-1)
        var g = x.reshaped(vitH, vitW, C)
        let ph = (r - vitH % r) % r
        let pw = (r - vitW % r) % r
        if ph > 0 || pw > 0 {
            g = padded(g, widths: [.init((0, ph)), .init((0, pw)), .init((0, 0))])
        }
        let H = vitH + ph, W = vitW + pw
        g = g.reshaped(H / r, r, W / r, r, C).transposed(0, 2, 4, 1, 3)
        return g.reshaped((H / r) * (W / r), C * r * r)
    }
}

/// The bundle's vision side as one module: `vision` (ViT), `aligner`, and the three learned
/// delimiter rows. Its parameter paths are the bundle tensor names (`vision.blocks.0.attn.wqkv.weight`
/// etc.) under this module's key.
public final class DeepseekV41VisionModel: Module {
    public let config: DeepseekV41VisionConfiguration
    @ModuleInfo(key: "vision") var vision: DeepseekV41ViT
    @ModuleInfo(key: "aligner") var aligner: DeepseekV41Aligner
    @ParameterInfo(key: "image_start") var imageStart: MLXArray
    @ParameterInfo(key: "image_end") var imageEnd: MLXArray
    @ParameterInfo(key: "image_newline") var imageNewline: MLXArray

    public init(config: DeepseekV41VisionConfiguration) {
        self.config = config
        self._vision.wrappedValue = DeepseekV41ViT(config)
        self._aligner.wrappedValue = DeepseekV41Aligner(config)
        self._imageStart.wrappedValue = MLXArray.zeros([config.textHiddenSize])
        self._imageEnd.wrappedValue = MLXArray.zeros([config.textHiddenSize])
        self._imageNewline.wrappedValue = MLXArray.zeros([config.textHiddenSize])
    }

    /// Aligner rows for one image: `[llmH * llmW, textHidden]`.
    func encode(patches: MLXArray, grid: DeepseekV41ImageGrid) -> MLXArray {
        aligner(vision(patches, vitH: grid.vitH, vitW: grid.vitW), vitH: grid.vitH, vitW: grid.vitW)
    }

    /// The embedding rows of one image span (reference `merge_image_embeddings`): START, then per
    /// LLM row the `llmW` aligner rows and a NEW_LINE, then END. `[tokenCount, textHidden]`.
    func spanRows(patches: MLXArray, grid: DeepseekV41ImageGrid, dtype: DType) -> MLXArray {
        let feats = encode(patches: patches, grid: grid).asType(dtype)
        let d = feats.dim(-1)
        let rows = feats.reshaped(grid.llmH, grid.llmW, d)
        let newline = broadcast(
            imageNewline.asType(dtype).reshaped(1, 1, d), to: [grid.llmH, 1, d])
        let body = concatenated([rows, newline], axis: 1).reshaped(grid.llmH * (grid.llmW + 1), d)
        return concatenated(
            [imageStart.asType(dtype).reshaped(1, d), body, imageEnd.asType(dtype).reshaped(1, d)],
            axis: 0)
    }

    /// The bundle tensor names this module loads (`vision.`, `aligner.`, `image_` prefixes).
    public static func ownsBundleKey(_ key: String) -> Bool {
        key.hasPrefix("vision.") || key.hasPrefix("aligner.") || key == "image_start"
            || key == "image_end" || key == "image_newline"
    }
}

/// One prefill chunk's image inputs, built by `DeepseekV4Model.prepare` and consumed by the
/// backbone forward: the embedding stream with the spans spliced in, the image mask (bias_vl
/// routing, Engram gate) and the Engram keep vector (image tokens take part in no n-gram).
struct DeepseekV41VisionChunk {
    /// `[1, L, D]`.
    let embeddings: MLXArray
    /// `[1, L]` bool, true inside image spans.
    let imageMask: MLXArray
    /// `[1, L]` float, 1 on text and 0 on image tokens (Engram gate mask).
    let textMask: MLXArray
    /// Per-token keep for the Engram hasher (false on image tokens).
    let keep: [Bool]

    /// Splice `spans` (chunk-relative starts, each wholly inside the chunk) into `embeddings`.
    static func make(
        textEmbeddings: MLXArray, spans: [(start: Int, rows: MLXArray)], length L: Int
    ) -> DeepseekV41VisionChunk {
        var pieces: [MLXArray] = []
        var keep = [Bool](repeating: true, count: L)
        var cursor = 0
        for span in spans.sorted(by: { $0.start < $1.start }) {
            if span.start > cursor { pieces.append(textEmbeddings[0, cursor ..< span.start]) }
            pieces.append(span.rows)
            let n = span.rows.dim(0)
            for i in span.start ..< span.start + n { keep[i] = false }
            cursor = span.start + n
        }
        if cursor < L { pieces.append(textEmbeddings[0, cursor ..< L]) }
        let embeddings = concatenated(pieces, axis: 0).expandedDimensions(axis: 0)
        let mask = MLXArray(keep.map { !$0 }).reshaped(1, L)
        let textMask = MLXArray(keep.map { $0 ? Float(1) : Float(0) }).reshaped(1, L)
        return DeepseekV41VisionChunk(
            embeddings: embeddings, imageMask: mask, textMask: textMask, keep: keep)
    }
}
