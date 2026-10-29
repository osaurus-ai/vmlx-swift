// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MiniMax-M3 / M3.1 image input (`minimax_m3_vl`). The reference is vMLX Python's validated route (VMLX_M3_VL:
// `vmlx_engine/models/minimax_m3/m3_vl_preprocess.py` + `minimax_m3_vl.py`, graded 12-image set), which runs the
// bundle's own processor (`image_processor.py` / `processing_minimax.py`):
//
// * preprocessing: decoded RGB bytes, `smart_resize` to multiples of patch x merge (28) within `max_pixels` (451,584,
//   the processor class default; the bundle's `max_image_resolution` is not read by it) and the fixed 3,136 minimum,
//   torch's antialiased bicubic resize of the uint8 image (int16 fixed-point weights, width then height:
//   `MiniMaxM3TorchResample`), `(x - 255 mean) / (255 std)` in float32, the frame repeated to the temporal patch (2),
//   Qwen2-VL patch order (merge windows contiguous): `[t h w, 3 x 2 x 14 x 14]`, handed to the model as bfloat16;
// * the vision stack (`vision_tower.vision_model.*`, `multi_modal_projector.*`, `patch_merge_mlp.*`, stored BF16 with a
//   float32 patch embedding): Conv3d patch embedding, 3-D (t, h, w) rotary positions on 78 of the 80 head dims,
//   `pre_layrnorm`, CLIP encoder layers with full attention per image, the projector per patch, the 2 x 2 merge
//   (four consecutive patch rows concatenated) and `patch_merge_mlp`. The float32 patch embedding promotes the whole
//   stack to float32 activations (as in Python); the features are cast to the token embedding's dtype at the splice;
// * the prompt: the chat template renders `]<]image[>[` for each `{"type": "image"}` content item (images first in
//   their message, as vMLX's preprocessing places them); each becomes `]<]start of image[>[`, `t h w / 4` image tokens
//   (200025), `]<]end of image[>[` (`processing_minimax.py`), and the features replace the image tokens' embeddings.
//
// Serving: the media prefill keeps every run of image tokens inside one forward (`MiniMaxM3ImageSplice.unsplit`); the
// prefix cache keys a media request by its pixels and grids (`computeMediaSalt`) and restores only text suffixes
// (`cacheHitSuffixRequiresMediaRollback`), so a text follow-up after an image turn reuses the image turn's rows; the
// engine leaves DSpark off while the prefilled input carries media (`setupDSpark` on the prefilled input).

// MARK: - Configuration

/// `config.json`'s vision fields (top level of a `minimax_m3_vl` config): `vision_config`, `image_token_index`,
/// `video_token_index`, `projector_hidden_size`.
public struct MiniMaxM3VisionConfiguration: Sendable, Equatable {
    public var hiddenSize: Int
    public var attentionHeads: Int
    public var layers: Int
    public var intermediateSize: Int
    public var patchSize: Int
    public var channels: Int
    public var layerNormEps: Float
    public var ropeTheta: Float
    public var mergeSize: Int
    public var temporalPatchSize: Int
    public var projectorHiddenSize: Int
    public var imageTokenId: Int
    public var videoTokenId: Int

    public var headDim: Int { hiddenSize / attentionHeads }

    /// nil when the config carries no `vision_config` (a text-only conversion), or one that does not name the stack's
    /// shape (`hidden_size`, `num_attention_heads`, `num_hidden_layers`, `intermediate_size`, as vMLX's stack reads
    /// them): the text model is built and the reason is written to the runtime diagnostics, never a load failure.
    public static func decode(configuration data: Data) throws -> Self? {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let vision = root["vision_config"] as? [String: Any]
        else { return nil }
        func int(_ d: [String: Any], _ k: String) -> Int? { (d[k] as? NSNumber)?.intValue }
        func float(_ d: [String: Any], _ k: String) -> Float? { (d[k] as? NSNumber)?.floatValue }
        let compression = vision["img_token_compression_config"] as? [String: Any] ?? [:]
        guard let hidden = int(vision, "hidden_size"), let heads = int(vision, "num_attention_heads"),
            let layers = int(vision, "num_hidden_layers"), let intermediate = int(vision, "intermediate_size"),
            hidden > 0, heads > 0, hidden % heads == 0, layers > 0
        else {
            RuntimeDiagnostics.write(Data(
                ("[vmlx][minimax-m3] vision_config lacks hidden_size / num_attention_heads / num_hidden_layers / "
                    + "intermediate_size: the vision stack is not built (text only)\n").utf8))
            return nil
        }
        let text = root["text_config"] as? [String: Any] ?? root
        return Self(
            hiddenSize: hidden, attentionHeads: heads, layers: layers, intermediateSize: intermediate,
            patchSize: int(vision, "patch_size") ?? 14, channels: int(vision, "num_channels") ?? 3,
            layerNormEps: float(vision, "layer_norm_eps") ?? 1e-5, ropeTheta: float(vision, "rope_theta") ?? 10_000,
            mergeSize: int(compression, "spatial_merge_size") ?? 2,
            temporalPatchSize: int(compression, "temporal_patch_size") ?? 2,
            projectorHiddenSize: int(root, "projector_hidden_size") ?? int(text, "hidden_size") ?? 6144,
            imageTokenId: int(root, "image_token_index") ?? 200_025,
            videoTokenId: int(root, "video_token_index") ?? 200_026)
    }
}

/// The bundle's `preprocessor_config.json` as the processor class reads it: `image_mean`, `image_std`, `patch_size`
/// (and `merge_size` / `temporal_patch_size` / `max_pixels` when present; the class defaults otherwise). `min_pixels`
/// is `smart_resize`'s own default (the processor never passes one).
public struct MiniMaxM3ImageProcessorConfiguration: Sendable, Equatable {
    public var imageMean: [Float] = [0.48145466, 0.4578275, 0.40821073]
    public var imageStd: [Float] = [0.26862954, 0.26130258, 0.27577711]
    public var patchSize = 14
    public var mergeSize = 2
    public var temporalPatchSize = 2
    public var maxPixels = 451_584
    public var minPixels = 4 * 28 * 28

    public init() {}

    public static func load(directory: URL) -> Self {
        var config = Self()
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("preprocessor_config.json")),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return config }
        func floats(_ k: String) -> [Float]? {
            (root[k] as? [NSNumber]).map { $0.map(\.floatValue) }.flatMap { $0.count == 3 ? $0 : nil }
        }
        func int(_ k: String) -> Int? { (root[k] as? NSNumber)?.intValue }
        if let v = floats("image_mean") { config.imageMean = v }
        if let v = floats("image_std") { config.imageStd = v }
        if let v = int("patch_size") { config.patchSize = v }
        if let v = int("merge_size") { config.mergeSize = v }
        if let v = int("temporal_patch_size") { config.temporalPatchSize = v }
        if let v = int("max_pixels") { config.maxPixels = v }
        return config
    }
}

/// What the input processor needs from a loaded M3 model with its vision stack.
public struct MiniMaxM3VisionPromptConfiguration: Sendable, Equatable {
    public var processor: MiniMaxM3ImageProcessorConfiguration
    public var imageTokenId: Int
}

public enum MiniMaxM3VisionError: Error, CustomStringConvertible {
    case configuration(String)
    case image(String)
    case placeholderMismatch(found: Int, images: Int)
    case spanMismatch(String)
    case missingToken(String)

    public var description: String {
        switch self {
        case .configuration(let why): return "MiniMax-M3 vision configuration: \(why)"
        case .image(let why): return "MiniMax-M3 image input: \(why)"
        case .placeholderMismatch(let found, let images):
            return "MiniMax-M3 image input: the prompt has \(found) image placeholder(s) but the request carries "
                + "\(images) image(s)"
        case .spanMismatch(let why): return "MiniMax-M3 image input: image tokens do not match the images (\(why))"
        case .missingToken(let token): return "MiniMax-M3 image input: the tokenizer has no \(token)"
        }
    }
}

// MARK: - Preprocessing

/// torch's antialiased bicubic resize of a uint8 image on the CPU (`F.interpolate(..., mode="bicubic",
/// antialias=True)` on a uint8 tensor, which the bundle's fast image processor calls): separable, width first, then
/// height (a dimension of unchanged size is skipped), each output a convolution of `ceil(2 scale) * 2 + 1` taps of
/// Keys' cubic (a = -0.5) widened by the downscale factor, normalized, then rounded to int16 at the widest precision
/// that keeps the largest tap below 2^15; integer accumulation from half a unit, shifted and clamped to 0...255 after
/// each pass. Bit-identical to torch 2.14 on the graded image set and the fixtures (CPU capability DEFAULT and AVX2
/// share these weights).
public enum MiniMaxM3TorchResample {
    static func cubic(_ x: Double) -> Double {
        let a = -0.5
        let x = abs(x)
        if x < 1.0 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2.0 { return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a }
        return 0
    }

    /// Per output index its first input index and int16 taps, and the precision (bits) of the taps.
    static func weights(inSize: Int, outSize: Int) -> (taps: [(start: Int, weights: [Int])], precision: Int) {
        let scale = Double(inSize) / Double(outSize)
        let support = scale >= 1.0 ? 2.0 * scale : 2.0
        let invScale = scale >= 1.0 ? 1.0 / scale : 1.0
        var rows: [(Int, [Double])] = []
        rows.reserveCapacity(outSize)
        var maxWeight = 0.0
        for i in 0 ..< outSize {
            let center = scale * (Double(i) + 0.5)
            let start = max(Int(center - support + 0.5), 0)
            let size = min(Int(center + support + 0.5), inSize) - start
            var w = (0 ..< max(size, 0)).map { j in cubic((Double(j + start) - center + 0.5) * invScale) }
            let total = w.reduce(0, +)
            if total != 0.0 {
                w = w.map { $0 / total }
                maxWeight = w.reduce(maxWeight, max)
            }
            rows.append((start, w))
        }
        var precision = 0
        while precision < 22 {
            if Int(0.5 + maxWeight * Double(1 << (precision + 1))) >= (1 << 15) { break }
            precision += 1
        }
        let unit = Double(1 << precision)
        let taps = rows.map { row in
            (start: row.0, weights: row.1.map { v -> Int in
                let x = v * unit
                return Int(Int16(truncatingIfNeeded: Int(x < 0 ? x - 0.5 : x + 0.5)))
            })
        }
        return (taps, precision)
    }

    /// One pass over `axis` (0 = height, 1 = width) of an `[h, w, 3]` uint8 buffer.
    static func pass(_ src: [UInt8], width: Int, height: Int, outSize: Int, horizontal: Bool) -> [UInt8] {
        let inSize = horizontal ? width : height
        let (taps, precision) = weights(inSize: inSize, outSize: outSize)
        let half = 1 << (precision - 1)
        let outW = horizontal ? outSize : width
        let outH = horizontal ? height : outSize
        var out = [UInt8](repeating: 0, count: outW * outH * 3)
        src.withUnsafeBufferPointer { s in
            out.withUnsafeMutableBufferPointer { o in
                for y in 0 ..< outH {
                    for x in 0 ..< outW {
                        let tap = taps[horizontal ? x : y]
                        var a0 = half, a1 = half, a2 = half
                        for (j, w) in tap.weights.enumerated() {
                            let p = horizontal ? (y * width + tap.start + j) * 3 : ((tap.start + j) * width + x) * 3
                            a0 += Int(s[p]) * w
                            a1 += Int(s[p + 1]) * w
                            a2 += Int(s[p + 2]) * w
                        }
                        let q = (y * outW + x) * 3
                        o[q] = UInt8(clamping: min(max(a0 >> precision, 0), 255))
                        o[q + 1] = UInt8(clamping: min(max(a1 >> precision, 0), 255))
                        o[q + 2] = UInt8(clamping: min(max(a2 >> precision, 0), 255))
                    }
                }
            }
        }
        return out
    }

    public static func resize(_ image: DeepseekV41RGBImage, width outW: Int, height outH: Int) -> DeepseekV41RGBImage {
        var pixels = image.pixels
        var w = image.width
        if outW != w {
            pixels = pass(pixels, width: w, height: image.height, outSize: outW, horizontal: true)
            w = outW
        }
        if outH != image.height {
            pixels = pass(pixels, width: w, height: image.height, outSize: outH, horizontal: false)
        }
        return DeepseekV41RGBImage(width: w, height: outH, pixels: pixels)
    }
}

public enum MiniMaxM3ImagePreprocessor {
    /// Python's `round()` (half to even).
    static func pyRound(_ x: Double) -> Int { Int(x.rounded(.toNearestOrEven)) }

    /// The bundle's `smart_resize`: (height, width) multiples of `factor`.
    public static func smartResize(
        height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int
    ) throws -> (height: Int, width: Int) {
        guard height > 0, width > 0 else { throw MiniMaxM3VisionError.image("empty image (\(width) x \(height))") }
        // The processor's own error (absolute aspect ratio must be smaller than 200), reproduced.
        guard Double(max(height, width)) / Double(min(height, width)) <= 200 else {
            throw MiniMaxM3VisionError.image("aspect ratio of \(width) x \(height) exceeds 200 (the bundle's processor)")
        }
        let (h, w, f) = (Double(height), Double(width), Double(factor))
        var hBar = max(factor, pyRound(h / f) * factor)
        var wBar = max(factor, pyRound(w / f) * factor)
        if hBar * wBar > maxPixels {
            let beta = (h * w / Double(maxPixels)).squareRoot()
            hBar = Int((h / beta / f).rounded(.down)) * factor
            wBar = Int((w / beta / f).rounded(.down)) * factor
        } else if hBar * wBar < minPixels {
            let beta = (Double(minPixels) / (h * w)).squareRoot()
            hBar = Int((h * beta / f).rounded(.up)) * factor
            wBar = Int((w * beta / f).rounded(.up)) * factor
        }
        guard hBar > 0, wBar > 0 else {
            throw MiniMaxM3VisionError.image("\(width) x \(height) resizes to nothing")
        }
        return (hBar, wBar)
    }

    /// One image: float32 patches `[t h w, C x tp x p x p]` (row-major, the processor's values) and its grid.
    public static func process(
        _ image: DeepseekV41RGBImage, config: MiniMaxM3ImageProcessorConfiguration
    ) throws -> (patches: [Float], grid: THW) {
        let p = config.patchSize, m = config.mergeSize, tp = config.temporalPatchSize
        let (rh, rw) = try smartResize(
            height: image.height, width: image.width, factor: p * m, minPixels: config.minPixels,
            maxPixels: config.maxPixels)
        let resized = MiniMaxM3TorchResample.resize(image, width: rw, height: rh)
        // The fast processor folds the rescale into the normalization: (x - mean / (1/255)) / (std / (1/255)), float32.
        let mean = config.imageMean.map { $0 * Float(255.0) }
        let std = config.imageStd.map { $0 * Float(255.0) }
        let gh = rh / p, gw = rw / p
        let rowWidth = 3 * tp * p * p
        var out = [Float](repeating: 0, count: gh * gw * rowWidth)
        var o = 0
        resized.pixels.withUnsafeBufferPointer { px in
            for bh in 0 ..< gh / m {
                for bw in 0 ..< gw / m {
                    for mh in 0 ..< m {
                        for mw in 0 ..< m {
                            let y0 = (bh * m + mh) * p, x0 = (bw * m + mw) * p
                            for c in 0 ..< 3 {
                                // A still image is its frame repeated to the temporal patch.
                                for _ in 0 ..< tp {
                                    for y in 0 ..< p {
                                        let row = (y0 + y) * rw
                                        for x in 0 ..< p {
                                            out[o] = (Float(px[(row + x0 + x) * 3 + c]) - mean[c]) / std[c]
                                            o += 1
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return (out, THW(1, gh, gw))
    }

    /// The `LMInput` image payload: every image's patches concatenated, as bfloat16 (what vMLX hands the model), one
    /// grid per image. The media salt hashes exactly these bytes and grids.
    public static func processedImage(
        _ images: [(patches: [Float], grid: THW)], rowWidth: Int
    ) -> LMInput.ProcessedImage {
        var flat: [Float] = []
        flat.reserveCapacity(images.reduce(0) { $0 + $1.patches.count })
        for image in images { flat.append(contentsOf: image.patches) }
        let pixels = MLXArray(flat, [flat.count / rowWidth, rowWidth]).asType(.bfloat16)
        return LMInput.ProcessedImage(pixels: pixels, frames: images.map(\.grid))
    }
}

// MARK: - Prompt

public enum MiniMaxM3VisionPrompt {
    public static let imageToken = "]<]image[>["
    public static let startToken = "]<]start of image[>["
    public static let endToken = "]<]end of image[>["

    /// A message's content with its images: one `{"type": "image"}` item per image, then the text (vMLX's
    /// `_normalize_messages_for_template` puts the image placeholders before the text of their turn).
    static func content(text: String, imageCount: Int) -> [[String: String]] {
        Array(repeating: ["type": "image"], count: imageCount) + [["type": "text", "text": text]]
    }

    /// Rewrite the generated message dicts so each image is an image item of the message it was sent in (tool results
    /// included: the template renders their image items), and return the images in prompt order. `chat` and
    /// `generated` correspond 1:1 (the default generator); otherwise every image goes to the last user turn (vMLX's
    /// out-of-band fallback).
    public static func anchorImages(
        chat: [Chat.Message]?, looseImages: [UserInput.Image], generated: [Message]
    ) -> (messages: [Message], images: [UserInput.Image]) {
        var out = generated
        var images: [UserInput.Image] = []
        if let chat, chat.count == generated.count {
            for (i, message) in chat.enumerated() where !message.images.isEmpty {
                let text = (generated[i]["content"] as? String) ?? message.content
                out[i]["content"] = content(text: text, imageCount: message.images.count)
                images.append(contentsOf: message.images)
            }
            return (out, images)
        }
        let all = chat.map { $0.flatMap(\.images) } ?? looseImages
        guard !all.isEmpty else { return (out, []) }
        if let last = out.lastIndex(where: { ($0["role"] as? String) == "user" }) {
            out[last]["content"] = content(text: (out[last]["content"] as? String) ?? "", imageCount: all.count)
        } else {
            out.append(["role": "user", "content": content(text: "", imageCount: all.count)])
        }
        return (out, all)
    }

    /// `processing_minimax.py`'s expansion at the token level: each image token becomes `start`, `count` image tokens,
    /// `end`. Returns the expanded tokens and, for every raw index (and the end), its expanded index.
    public static func expand(
        tokens: [Int], imageTokenId: Int, startTokenId: Int, endTokenId: Int, counts: [Int]
    ) throws -> (tokens: [Int], rawToExpanded: [Int]) {
        let found = tokens.reduce(0) { $0 + ($1 == imageTokenId ? 1 : 0) }
        guard found == counts.count else {
            throw MiniMaxM3VisionError.placeholderMismatch(found: found, images: counts.count)
        }
        var out: [Int] = []
        out.reserveCapacity(tokens.count + counts.reduce(0) { $0 + $1 + 1 })
        var map: [Int] = []
        map.reserveCapacity(tokens.count + 1)
        var next = 0
        for t in tokens {
            map.append(out.count)
            if t == imageTokenId {
                out.append(startTokenId)
                out.append(contentsOf: repeatElement(imageTokenId, count: counts[next]))
                out.append(endTokenId)
                next += 1
            } else {
                out.append(t)
            }
        }
        map.append(out.count)
        return (out, map)
    }

    /// Cache boundaries are raw-token prefix lengths; `b` maps to `rawToExpanded[b]`.
    public static func remapBoundaries(_ boundaries: [Int], rawToExpanded: [Int]) -> [Int] {
        boundaries.compactMap { b in (b >= 0 && b < rawToExpanded.count) ? rawToExpanded[b] : nil }
    }
}

// MARK: - Vision stack (module names mirror the bundle)

final class MiniMaxM3VisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv3d
    let channels: Int, temporal: Int, patch: Int, hidden: Int

    init(_ c: MiniMaxM3VisionConfiguration) {
        channels = c.channels
        temporal = c.temporalPatchSize
        patch = c.patchSize
        hidden = c.hiddenSize
        _patchEmbedding.wrappedValue = Conv3d(
            inputChannels: c.channels, outputChannels: c.hiddenSize,
            kernelSize: .init((c.temporalPatchSize, c.patchSize, c.patchSize)),
            stride: .init((c.temporalPatchSize, c.patchSize, c.patchSize)), bias: false)
    }

    /// `[N, C x T x p x p]` -> `[N, hidden]` (MLX's Conv3d takes channels last).
    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let x = pixels.reshaped(-1, channels, temporal, patch, patch).movedAxis(source: 1, destination: 4)
        return patchEmbedding(x).reshaped(-1, hidden)
    }
}

final class MiniMaxM3VisionAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    let heads: Int, headDim: Int
    let scale: Float

    init(_ c: MiniMaxM3VisionConfiguration) {
        heads = c.attentionHeads
        headDim = c.headDim
        // Python `head_dim ** -0.5` (a double) handed to the kernel as a float.
        scale = Float(pow(Double(c.headDim), -0.5))
        _qProj.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
        _kProj.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
        _vProj.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
        _outProj.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray, segments: [Range<Int>], cos: MLXArray, sin: MLXArray) -> MLXArray {
        let L = x.dim(0)
        let q = MiniMaxM3VisionMath.rope(qProj(x).reshaped(1, L, heads, headDim), cos: cos, sin: sin)[0]
        let k = MiniMaxM3VisionMath.rope(kProj(x).reshaped(1, L, heads, headDim), cos: cos, sin: sin)[0]
        let v = vProj(x).reshaped(L, heads, headDim)
        let qh = q.transposed(1, 0, 2).expandedDimensions(axis: 0)
        let kh = k.transposed(1, 0, 2).expandedDimensions(axis: 0)
        let vh = v.transposed(1, 0, 2).expandedDimensions(axis: 0)
        // Full attention inside each image, none across images (`cu_seqlens`).
        let outs = segments.filter { !$0.isEmpty }.map { s in
            MLXFast.scaledDotProductAttention(
                queries: qh[0..., 0..., s, 0...], keys: kh[0..., 0..., s, 0...], values: vh[0..., 0..., s, 0...],
                scale: scale, mask: .none)
        }
        let o = concatenated(outs, axis: 2)[0].transposed(1, 0, 2).reshaped(L, -1)
        return outProj(o)
    }
}

final class MiniMaxM3VisionMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ c: MiniMaxM3VisionConfiguration) {
        _fc1.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: true)
        _fc2.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

final class MiniMaxM3VisionLayer: Module {
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttention: MiniMaxM3VisionAttention
    @ModuleInfo(key: "mlp") var mlp: MiniMaxM3VisionMLP

    init(_ c: MiniMaxM3VisionConfiguration) {
        _layerNorm1.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _layerNorm2.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _selfAttention.wrappedValue = MiniMaxM3VisionAttention(c)
        _mlp.wrappedValue = MiniMaxM3VisionMLP(c)
    }

    func callAsFunction(_ x: MLXArray, segments: [Range<Int>], cos: MLXArray, sin: MLXArray) -> MLXArray {
        let h = x + selfAttention(layerNorm1(x), segments: segments, cos: cos, sin: sin)
        return h + mlp(layerNorm2(h))
    }
}

final class MiniMaxM3VisionEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [MiniMaxM3VisionLayer]

    init(_ c: MiniMaxM3VisionConfiguration) {
        _layers.wrappedValue = (0 ..< c.layers).map { _ in MiniMaxM3VisionLayer(c) }
    }
}

final class MiniMaxM3VisionModel: Module {
    @ModuleInfo(key: "embeddings") var embeddings: MiniMaxM3VisionEmbeddings
    @ModuleInfo(key: "pre_layrnorm") var preLayerNorm: LayerNorm
    @ModuleInfo(key: "encoder") var encoder: MiniMaxM3VisionEncoder
    let config: MiniMaxM3VisionConfiguration

    init(_ c: MiniMaxM3VisionConfiguration) {
        config = c
        _embeddings.wrappedValue = MiniMaxM3VisionEmbeddings(c)
        _preLayerNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _encoder.wrappedValue = MiniMaxM3VisionEncoder(c)
    }

    func callAsFunction(_ pixels: MLXArray, grids: [THW]) -> MLXArray {
        var x = preLayerNorm(embeddings(pixels))
        let (cos, sin) = MiniMaxM3VisionMath.rotary(
            grids: grids, mergeSize: config.mergeSize, headDim: config.headDim, theta: config.ropeTheta)
        var segments: [Range<Int>] = []
        var at = 0
        for g in grids {
            segments.append(at ..< at + g.t * g.h * g.w)
            at += g.t * g.h * g.w
        }
        for layer in encoder.layers { x = layer(x, segments: segments, cos: cos, sin: sin) }
        return x
    }
}

final class MiniMaxM3VisionTower: Module {
    @ModuleInfo(key: "vision_model") var visionModel: MiniMaxM3VisionModel

    init(_ c: MiniMaxM3VisionConfiguration) { _visionModel.wrappedValue = MiniMaxM3VisionModel(c) }
}

/// `linear_1 -> GELU -> linear_2` (`multi_modal_projector`, `patch_merge_mlp`).
final class MiniMaxM3VisionMLP2: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ input: Int, _ hidden: Int, _ output: Int) {
        _linear1.wrappedValue = Linear(input, hidden, bias: true)
        _linear2.wrappedValue = Linear(hidden, output, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(gelu(linear1(x))) }
}

/// The whole image path: tower, projector per patch, 2 x 2 merge, patch-merge MLP. Module key `vision` on
/// `MiniMaxM3Model` (the bundle's `vision_tower.*`, `multi_modal_projector.*`, `patch_merge_mlp.*` under it).
public final class MiniMaxM3VisionStack: Module {
    @ModuleInfo(key: "vision_tower") var visionTower: MiniMaxM3VisionTower
    @ModuleInfo(key: "multi_modal_projector") var projector: MiniMaxM3VisionMLP2
    @ModuleInfo(key: "patch_merge_mlp") var patchMerge: MiniMaxM3VisionMLP2
    public let config: MiniMaxM3VisionConfiguration

    public init(_ c: MiniMaxM3VisionConfiguration) {
        config = c
        let merge = c.mergeSize * c.mergeSize
        _visionTower.wrappedValue = MiniMaxM3VisionTower(c)
        _projector.wrappedValue = MiniMaxM3VisionMLP2(c.hiddenSize, c.projectorHiddenSize, c.projectorHiddenSize)
        _patchMerge.wrappedValue = MiniMaxM3VisionMLP2(
            c.projectorHiddenSize * merge, c.projectorHiddenSize, c.projectorHiddenSize)
    }

    /// `[sum t h w, C T p p]` pixels of `grids` -> `[sum t h w / merge^2, projector_hidden_size]` (float32 for the
    /// bundle: the float32 patch embedding promotes the stack, as in Python).
    public func callAsFunction(_ pixels: MLXArray, grids: [THW]) -> MLXArray {
        let projected = projector(visionTower.visionModel(pixels, grids: grids))
        let merge = config.mergeSize * config.mergeSize
        let n = projected.dim(0)
        return patchMerge(projected.reshaped(n / merge, merge * projected.dim(-1)))
    }

    /// The bundle's names (`vision_tower.*`, `multi_modal_projector.*`, `patch_merge_mlp.*`) under `vision.`; the
    /// patch embedding from PyTorch's (out, C, T, H, W) to MLX's (out, T, H, W, C).
    static func sanitize(key: String, value: MLXArray) -> (String, MLXArray)? {
        guard key.hasPrefix("vision_tower.") || key.hasPrefix("multi_modal_projector.")
            || key.hasPrefix("patch_merge_mlp.")
        else { return nil }
        var value = value
        if key.hasSuffix("patch_embedding.weight"), value.ndim == 5 {
            value = value.movedAxis(source: 1, destination: 4)
        }
        return ("vision." + key, value)
    }
}

enum MiniMaxM3VisionMath {
    /// Python `rotate_half` + `apply_rope_vision`: the first `cos.dim(-1)` head dims rotate (`x * cos +
    /// rotate_half(x) * sin`), the rest pass through. `x`: `[1, L, H, d]`.
    static func rope(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let rot = cos.dim(-1)
        let c = cos.expandedDimensions(axis: 0).expandedDimensions(axis: 2)
        let s = sin.expandedDimensions(axis: 0).expandedDimensions(axis: 2)
        let xr = x[.ellipsis, ..<rot]
        let xp = x[.ellipsis, rot...]
        let half = rot / 2
        let rotated = concatenated([-xr[.ellipsis, half...], xr[.ellipsis, ..<half]], axis: -1)
        return concatenated([xr * c + rotated * s, xp], axis: -1)
    }

    /// Python `rot_pos_emb`: (t, h, w) per patch in merge-window order, `axis_dim = 2 ((2 (d / 2) / 3) / 2)` rotary dims
    /// per axis, `inv_freq = 1 / theta^(arange(0, axis_dim, 2) / axis_dim)`; returns cos / sin of
    /// `concat(freqs, freqs)`, `[N, 3 axis_dim]`, float32, built with the same MLX ops.
    static func rotary(grids: [THW], mergeSize m: Int, headDim: Int, theta: Float) -> (MLXArray, MLXArray) {
        let axisDim = 2 * (((2 * (headDim / 2)) / 3) / 2)
        var coords: [Float] = []
        for g in grids {
            for t in 0 ..< g.t {
                for bh in 0 ..< g.h / m {
                    for bw in 0 ..< g.w / m {
                        for mh in 0 ..< m {
                            for mw in 0 ..< m {
                                coords += [Float(t), Float(bh * m + mh), Float(bw * m + mw)]
                            }
                        }
                    }
                }
            }
        }
        let c = MLXArray(coords, [coords.count / 3, 3])
        let exponents = MLXArray(stride(from: 0, to: axisDim, by: 2).map { Float($0) }) / Float(axisDim)
        let invFreq = MLXArray(Float(1.0)) / MLX.pow(MLXArray(theta), exponents)
        let freqs = concatenated((0 ..< 3).map { c[0..., $0 ..< ($0 + 1)] * invFreq }, axis: -1)
        let emb = concatenated([freqs, freqs], axis: -1)
        return (MLX.cos(emb), MLX.sin(emb))
    }
}

// MARK: - Splice at the image tokens (prefill)

/// One request's image features and where they go, for `MiniMaxM3Model.prepare`.
struct MiniMaxM3ImageSplice {
    /// Absolute positions of the image tokens in prompt order, and the feature row of each.
    let positions: [Int]
    /// `[positions.count, projector_hidden_size]`.
    let features: MLXArray
    /// Absolute runs of image tokens (one per image).
    let runs: [Range<Int>]

    /// The image tokens of `tokens` (absolute positions from `start`) matched to the request's images: the token list
    /// holds every image of the request or, after a cache restore, only the LAST k of them (each a complete run).
    /// nil when it holds no image token.
    static func make(
        tokens: [Int], start: Int, image: LMInput.ProcessedImage, vision: MiniMaxM3VisionStack
    ) throws -> MiniMaxM3ImageSplice? {
        let id = vision.config.imageTokenId
        let found = tokens.reduce(0) { $0 + ($1 == id ? 1 : 0) }
        guard found > 0 else { return nil }
        let grids = image.frames ?? []
        let merge = vision.config.mergeSize * vision.config.mergeSize
        let counts = grids.map { $0.t * $0.h * $0.w / merge }
        var first: Int?
        var total = 0
        for k in stride(from: counts.count - 1, through: 0, by: -1) {
            total += counts[k]
            if total == found { first = k; break }
            if total > found { break }
        }
        guard let first else {
            throw MiniMaxM3VisionError.spanMismatch(
                "\(found) image tokens match no trailing set of the \(grids.count) image(s) (\(counts))")
        }
        var runs: [Range<Int>] = []
        var next = first
        var i = 0
        while i < tokens.count {
            guard tokens[i] == id else { i += 1; continue }
            guard next < counts.count, i + counts[next] <= tokens.count,
                tokens[i ..< i + counts[next]].allSatisfy({ $0 == id })
            else {
                throw MiniMaxM3VisionError.spanMismatch("image \(next) needs a run of its tokens at \(i)")
            }
            runs.append((start + i) ..< (start + i + counts[next]))
            i += counts[next]
            next += 1
        }
        guard next == counts.count else {
            throw MiniMaxM3VisionError.spanMismatch("\(counts.count - next) image(s) without their tokens")
        }
        var patchStart = 0
        for k in 0 ..< first { patchStart += grids[k].t * grids[k].h * grids[k].w }
        let pixels = first == 0 ? image.pixels : image.pixels[patchStart...]
        let started = Date()
        let features = vision(pixels, grids: Array(grids[first...]))
        eval(features)
        RuntimeReceipts.emit(
            "m3vision stage=prefill images=\(counts.count - first)/\(counts.count) runs="
                + runs.map { "\($0.lowerBound)-\($0.upperBound)" }.joined(separator: ",")
                + " tower_ms=\(Int(Date().timeIntervalSince(started) * 1000)) dtype=\(features.dtype)")
        return MiniMaxM3ImageSplice(positions: runs.flatMap { Array($0) }, features: features, runs: runs)
    }

    /// `pieces` (absolute, consecutive) with every cut strictly inside an image run removed: a run never spans two
    /// forwards. Cuts at a run's edges stay.
    func unsplit(_ pieces: [Range<Int>]) -> [Range<Int>] {
        var out: [Range<Int>] = []
        for piece in pieces {
            if let last = out.last, runs.contains(where: { $0.lowerBound < last.upperBound && last.upperBound < $0.upperBound }) {
                out[out.count - 1] = last.lowerBound ..< piece.upperBound
            } else {
                out.append(piece)
            }
        }
        return out
    }

    func touches(_ piece: Range<Int>) -> Bool { runs.contains { $0.overlaps(piece) } }

    /// The input embeddings of `piece` (`ids` its tokens, `[1, L]`) with the image rows replaced; nil when it holds no
    /// image token.
    func embeddings(for piece: Range<Int>, ids: MLXArray, embed: (MLXArray) -> MLXArray) -> MLXArray? {
        guard touches(piece) else { return nil }
        let text = embed(ids)
        var local: [Int32] = []
        var rows: [Int32] = []
        for (row, p) in positions.enumerated() where piece.contains(p) {
            local.append(Int32(p - piece.lowerBound))
            rows.append(Int32(row))
        }
        let flat = text.reshaped(-1, text.dim(-1))
        flat[MLXArray(local)] = features[MLXArray(rows)].asType(flat.dtype)
        return flat.reshaped(text.shape)
    }
}
