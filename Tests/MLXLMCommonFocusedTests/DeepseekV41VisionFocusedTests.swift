// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 image input (CPU only, tiny synthetic weights, no bundle weights, no GPU):
//   - grid planning, PIL-exact bicubic resize / pad, normalization and patch layout against the
//     Python reference (values produced CPU-only from jang_tools/dsv41/vision.py + Pillow 12.3,
//     hard-coded below);
//   - the ViT + Aligner on tiny deterministic weights against a NumPy port of the vision.py math;
//   - span rows (START, rows + NEW_LINE, END), placeholder expansion, cache-boundary remapping;
//   - prompt assembly against the bundle encoder (`encoding.py encode_messages`) goldens;
//   - span location, the engine's never-split chunk rule, chunk-relative splicing, and a tiny
//     V4.1 backbone whose image prefill is invariant to the chunk plan;
//   - `bias_vl` routing for image tokens and Engram DEAD hashing;
//   - media-salted cache keys (same image hits, different image misses, text-only unsalted);
//   - DSV4-0731 and text-only V4.1 unchanged (no tower, no bias_vl, identical text prefill).

import CoreGraphics
import CoreImage
import CryptoKit
import Foundation
import ImageIO
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("DeepSeek-V4.1 vision (CPU)", .serialized)
struct DeepseekV41VisionFocusedTests {

    // MARK: Fixtures

    static let bundleVision = DeepseekV41VisionConfiguration(
        numHiddenLayers: 32, hiddenSize: 1024, numAttentionHeads: 16, intermediateSize: 2816,
        patchSize: 14, ropeTheta: 10_000, downsampleRatio: 3, maxImageTokens: 1024,
        minPixels: 295_936, maxWHRatio: nil, imageTokenId: 129_264, textHiddenSize: 5120)

    /// The reference test pattern: (x*7 + y*13, x*3 + y*5 + 17, x*x + y) mod 256.
    static func pattern(_ w: Int, _ h: Int) -> DeepseekV41RGBImage {
        var px = [UInt8](repeating: 0, count: w * h * 3)
        for y in 0 ..< h {
            for x in 0 ..< w {
                let o = (y * w + x) * 3
                px[o] = UInt8((x * 7 + y * 13) % 256)
                px[o + 1] = UInt8((x * 3 + y * 5 + 17) % 256)
                px[o + 2] = UInt8((x * x + y) % 256)
            }
        }
        return DeepseekV41RGBImage(width: w, height: h, pixels: px)
    }

    static func sha256(_ bytes: [UInt8]) -> String {
        SHA256.hash(data: Data(bytes)).map { String(format: "%02x", $0) }.joined()
    }

    static func sha256(_ floats: [Float]) -> String {
        floats.withUnsafeBufferPointer { buf in
            SHA256.hash(data: Data(buffer: buf)).map { String(format: "%02x", $0) }.joined()
        }
    }

    // MARK: Planning, resize, preprocessing (Python reference values)

    /// `plan_image_grid` -> [best_w, best_h, vit_h, vit_w, llm_h, llm_w, tokens].
    static let referencePlans: [String: [Int]] = [
        "64x64": [546, 546, 39, 39, 13, 13, 184], "640x480": [644, 490, 35, 46, 12, 16, 206],
        "1920x1080": [1708, 966, 69, 122, 23, 41, 968], "100x3000": [112, 3010, 215, 8, 72, 3, 290],
        "4000x50": [4872, 70, 5, 348, 2, 116, 236], "544x544": [546, 546, 39, 39, 13, 13, 184],
        "1000x1000": [1008, 1008, 72, 72, 24, 24, 602], "13x7": [742, 406, 29, 53, 10, 18, 192],
        "900x600": [910, 602, 43, 65, 15, 22, 347], "3000x2000": [1568, 1050, 75, 112, 25, 38, 977],
        "301x200": [672, 448, 32, 48, 11, 16, 189], "700x50": [2044, 154, 11, 146, 4, 49, 202],
    ]

    @Test("grid planning and image token counts match vision.py", arguments: Array(referencePlans.keys))
    func planning(_ key: String) {
        let dims = key.split(separator: "x").map { Int($0)! }
        let g = DeepseekV41ImagePlanner.plan(width: dims[0], height: dims[1], config: Self.bundleVision)
        let expected = Self.referencePlans[key]!
        #expect([g.bestW, g.bestH, g.vitH, g.vitW, g.llmH, g.llmW, g.tokenCount] == expected)
        #expect(g.tokenCount <= Self.bundleVision.maxImageTokens)
        // The grid a frame implies (what the backbone re-derives from THW) is the same grid.
        #expect(DeepseekV41ImageGrid(vitH: g.vitH, vitW: g.vitW, patchSize: 14, downsampleRatio: 3)
            .tokenCount == g.tokenCount)
        let types = g.tokenTypes
        #expect(types.count == g.tokenCount)
        #expect(types.first == .start && types.last == .end)
        #expect(types.filter { $0 == .image }.count == g.llmH * g.llmW)
        #expect(types.filter { $0 == .newline }.count == g.llmH)
    }

    /// `Image.resize(size, BICUBIC)` of the pattern -> [sha256 of RGB bytes, byte sum].
    static let referenceResizes: [String: (String, Int)] = [
        "97x61->40x25": ("d3d0433b61229686760843e471d9b8f268b1807d5f666ff566115ede34c08e0e", 371_573),
        "23x17->50x31": ("cc64613d6dec2ab7353bf4f11b64904aac998f687d254aa2afd9c51207117b32", 491_522),
        "40x30->40x55": ("8b17df731f35e7220ba42ab9e36d3c198b2aaa4e35be2068ffb3fedcbdc72cf1", 829_220),
        "40x30->71x30": ("21be79ebe89a64e86532fd22de6d504cb7b173ca7eae1edd243fc45778e9bbe8", 802_483),
    ]

    @Test("bicubic resize is byte-identical to Pillow", arguments: Array(referenceResizes.keys))
    func pilResize(_ key: String) {
        let parts = key.split(separator: ">").map { $0.trimmingCharacters(in: CharacterSet(charactersIn: "-")) }
        let src = parts[0].split(separator: "x").map { Int($0)! }
        let dst = parts[1].split(separator: "x").map { Int($0)! }
        let out = DeepseekV41PILResample.resize(Self.pattern(src[0], src[1]), width: dst[0], height: dst[1])
        #expect(out.width == dst[0] && out.height == dst[1])
        let (sha, sum) = Self.referenceResizes[key]!
        #expect(Self.sha256(out.pixels) == sha)
        #expect(out.pixels.reduce(0) { $0 + Int($1) } == sum)
    }

    /// `load_image` on the pattern: canvas size + sha, patch tensor sha (float32 bytes), grid.
    static let referencePreprocess: [String: (Int, Int, String, String, [Int])] = [
        "301x200": (672, 448, "b361ba0c512b76ca2a824641e13e1d6feb8869ff1a93a4b8816e27da980b46ab",
                    "9726e4e6b04bb997809d15ed15797b141342f45e758a2eb61b843fb22f2f7b65", [32, 48, 11, 16]),
        "900x600": (910, 602, "0a7c1273d44c56c7d00c34eafba049a50017cdb86e1596e39aa8631c78c3e701",
                    "87395e91b405b7a3ba4b6ca9e98ebbefc9650179a1c39b3a8d2010541f99e098", [43, 65, 15, 22]),
        "64x64": (546, 546, "bcbb704f4db743381ad25a3cef6ecc83d10be3f8b0fb61da3f1cbb913afd895c",
                  "19bdeb5e234adf92b4d185d7aad6ffab8e8de3343384d1ca31ec6d9e8dfb1393", [39, 39, 13, 13]),
        "700x50": (2044, 154, "993f121f94c5a07338cebd158837ef23481adad8e2824f3a71240e1901bb3790",
                   "2432661b4cad96e1e13842a3faa3435d8bdf847afd57daab53371571884073e9", [11, 146, 4, 49]),
    ]

    @Test("pad + normalize + patchify match vision.py load_image bit for bit",
          arguments: Array(referencePreprocess.keys))
    func preprocessing(_ key: String) {
        let dims = key.split(separator: "x").map { Int($0)! }
        let image = Self.pattern(dims[0], dims[1])
        let (cw, ch, canvasSHA, patchSHA, grid) = Self.referencePreprocess[key]!
        let g = DeepseekV41ImagePlanner.plan(width: dims[0], height: dims[1], config: Self.bundleVision)
        let canvas = DeepseekV41PILResample.pad(image, width: g.bestW, height: g.bestH)
        #expect(canvas.width == cw && canvas.height == ch)
        #expect(Self.sha256(canvas.pixels) == canvasSHA)
        let pre = DeepseekV41ImagePreprocessor.preprocess(image, config: Self.bundleVision)
        #expect([pre.grid.vitH, pre.grid.vitW, pre.grid.llmH, pre.grid.llmW] == grid)
        #expect(pre.patches.count == grid[0] * grid[1] * 3 * 14 * 14)
        #expect(Self.sha256(pre.patches) == patchSHA)
    }

    @Test("solid red / blue 64x64 images: 39x39 patches, 184 tokens, same patch sum (-298116)")
    func solidColorImages() {
        for color: (UInt8, UInt8, UInt8) in [(255, 0, 0), (0, 0, 255)] {
            let pre = DeepseekV41ImagePreprocessor.preprocess(
                DeepseekV41RGBImage(width: 64, height: 64, fill: color), config: Self.bundleVision)
            let grid: [Int] = [pre.grid.vitH, pre.grid.vitW, pre.grid.llmH, pre.grid.llmW]
            #expect(grid == [39, 39, 13, 13] as [Int])
            #expect(pre.grid.tokenCount == 184)
            #expect(pre.patches.reduce(0.0) { $0 + Double($1) } == -298_116)
        }
    }

    /// A PNG of `image` written through ImageIO.
    static func png(_ image: DeepseekV41RGBImage) throws -> Data {
        var rgba = [UInt8](repeating: 255, count: image.width * image.height * 4)
        for i in 0 ..< image.width * image.height {
            rgba[i * 4] = image.pixels[i * 3]
            rgba[i * 4 + 1] = image.pixels[i * 3 + 1]
            rgba[i * 4 + 2] = image.pixels[i * 3 + 2]
        }
        let provider = CGDataProvider(data: Data(rgba) as CFData)!
        let cg = CGImage(
            width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: image.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let out = NSMutableData()
        let dest = CGImageDestinationCreateWithData(out as CFMutableData, "public.png" as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, cg, nil)
        #expect(CGImageDestinationFinalize(dest))
        return out as Data
    }

    @Test("request images decode to the stored RGB bytes, top row first (ciImage and file URL)")
    func decodeRoundTrip() throws {
        let image = Self.pattern(37, 23)
        let data = try Self.png(image)
        let fromCI = try DeepseekV41ImagePreprocessor.decodeRGB(.ciImage(CIImage(data: data)!))
        #expect(fromCI == image)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("dsv41-vl-\(UUID().uuidString).png")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(try DeepseekV41ImagePreprocessor.decodeRGB(.url(url)) == image)
    }

    // MARK: Vision tower (tiny deterministic weights vs the NumPy port of vision.py)

    static let tinyVision = DeepseekV41VisionConfiguration(
        numHiddenLayers: 2, hiddenSize: 32, numAttentionHeads: 2, intermediateSize: 24, patchSize: 2,
        ropeTheta: 10_000, downsampleRatio: 3, maxImageTokens: 1024, minPixels: 16,
        imageTokenId: 120, textHiddenSize: 20)

    /// Parameters in the reference order with `sin(0.37 k + 1.3 j + 0.1)` values (weights x0.2,
    /// biases x0.05, norms 1 + 0.1 sin), exactly as the NumPy reference generated them.
    static func tinyVisionParameters(_ c: DeepseekV41VisionConfiguration) -> [(String, [Int], MLXArray)] {
        let D = c.hiddenSize, P = c.patchSize, R = c.downsampleRatio, TD = c.textHiddenSize
        var order: [(String, [Int], String)] = [
            ("vision.patch_embed.proj.weight", [D, 3 * P * P], "w"), ("vision.patch_embed.proj.bias", [D], "b"),
        ]
        for i in 0 ..< c.numHiddenLayers {
            let b = "vision.blocks.\(i)"
            order += [
                ("\(b).norm1.weight", [D], "n"), ("\(b).attn.wqkv.weight", [3 * D, D], "w"),
                ("\(b).attn.wqkv.bias", [3 * D], "b"), ("\(b).attn.wo.weight", [D, D], "w"),
                ("\(b).attn.wo.bias", [D], "b"), ("\(b).norm2.weight", [D], "n"),
                ("\(b).mlp.w1.weight", [2 * c.intermediateSize, D], "w"),
                ("\(b).mlp.w2.weight", [D, c.intermediateSize], "w"),
            ]
        }
        order += [
            ("vision.norm.weight", [D], "n"), ("aligner.w1.weight", [TD, D * R * R], "w"),
            ("aligner.w1.bias", [TD], "b"), ("aligner.w2.weight", [TD, TD], "w"), ("aligner.w2.bias", [TD], "b"),
            ("image_start", [TD], "b"), ("image_end", [TD], "b"), ("image_newline", [TD], "b"),
        ]
        return order.enumerated().map { j, entry in
            let n = entry.1.reduce(1, *)
            let values: [Float] = (0 ..< n).map { k in
                let s = Foundation.sin(0.37 * Double(k) + 1.3 * Double(j) + 0.1)
                switch entry.2 {
                case "n": return Float(1.0 + 0.1 * s)
                case "b": return Float(0.05 * s)
                default: return Float(0.2 * s)
                }
            }
            return (entry.0, entry.1, MLXArray(values, entry.1))
        }
    }

    static func tinyVisionModel() -> DeepseekV41VisionModel {
        let model = DeepseekV41VisionModel(config: tinyVision)
        let params = tinyVisionParameters(tinyVision).map { ($0.0, $0.2) }
        try! model.update(parameters: ModuleParameters.unflattened(params), verify: [.all])
        eval(model)
        return model
    }

    /// NumPy reference (vision.py math in float32): aligner rows [4, 20] for a 4x5 patch grid.
    static let referenceFeats: [Float] = [
        -0.1412852, -0.5436373, -0.3708084, 0.1730224, 0.4732179, 0.1953993, -0.3394452, -0.5166212,
        -0.1198821, 0.4234059, 0.5197007, 0.0735409, -0.4069755, -0.3816979, 0.1149609, 0.5134826,
        0.3503585, -0.2092683, -0.5546021, -0.312677, -0.5020343, 0.62622, 1.016016, 0.2195416,
        -0.8728051, -1.031709, -0.06971282, 0.9470684, 0.8941799, -0.1508554, -0.9980428, -0.6834024,
        0.446857, 1.124138, 0.5818885, -0.582813, -1.078124, -0.3658871, 0.736511, 0.9763702,
        0.2129786, -0.04655926, -0.2890877, -0.2523789, 0.01837996, 0.2218646, 0.1386053, -0.123791,
        -0.2533833, -0.08651759, 0.2059509, 0.308276, 0.1126593, -0.1606641, -0.2108064, 0.006720163,
        0.2316471, 0.1933731, -0.08273611, -0.3014473, -0.5345377, -0.5669652, 0.001983408, 0.523324,
        0.4076737, -0.2123908, -0.6315715, -0.3650537, 0.3051834, 0.6446609, 0.2886969, -0.3503263,
        -0.5477433, -0.08129735, 0.5192084, 0.5676455, -0.006383152, -0.5763273, -0.519806, 0.08490141,
    ]

    static func tinyPatches(_ n: Int, patch p: Int, phase: Double = 0.5) -> MLXArray {
        let count = n * 3 * p * p
        let values = (0 ..< count).map { Float(Foundation.sin(0.11 * Double($0) + phase)) }
        return MLXArray(values, [n, 3, p, p])
    }

    @Test("ViT + Aligner on tiny weights match the NumPy port of vision.py (fp32)")
    func visionTowerMatchesReference() {
        Device.withDefaultDevice(.cpu) {
            let model = Self.tinyVisionModel()
            let grid = DeepseekV41ImageGrid(vitH: 4, vitW: 5, patchSize: 2, downsampleRatio: 3)
            #expect(grid.llmH == 2 && grid.llmW == 2)
            let feats = model.encode(patches: Self.tinyPatches(20, patch: 2), grid: grid)
            #expect(feats.shape == [4, 20])
            let got = feats.asType(.float32).asArray(Float.self)
            let maxErr = zip(got, Self.referenceFeats).map { abs($0 - $1) }.max() ?? .infinity
            #expect(maxErr < 2e-5, "max |swift - numpy| = \(maxErr)")
        }
    }

    @Test("span rows: START, (llmW rows, NEW_LINE) per LLM row, END")
    func spanRowsLayout() {
        Device.withDefaultDevice(.cpu) {
            let model = Self.tinyVisionModel()
            let grid = DeepseekV41ImageGrid(vitH: 4, vitW: 5, patchSize: 2, downsampleRatio: 3)
            let patches = Self.tinyPatches(20, patch: 2)
            let feats = model.encode(patches: patches, grid: grid)
            let rows = model.spanRows(patches: patches, grid: grid, dtype: .float32)
            #expect(rows.shape == [grid.tokenCount, 20])  // 2 * 3 + 2 = 8
            func row(_ a: MLXArray, _ i: Int) -> [Float] { a[i].asArray(Float.self) }
            let start = model.imageStart.asArray(Float.self)
            let end = model.imageEnd.asArray(Float.self)
            let newline = model.imageNewline.asArray(Float.self)
            let expected: [[Float]] = [start, row(feats, 0), row(feats, 1), newline,
                                       row(feats, 2), row(feats, 3), newline, end]
            for (i, e) in expected.enumerated() { #expect(row(rows, i) == e, "row \(i)") }
            let types = grid.tokenTypes
            #expect(types == [.start, .image, .image, .newline, .image, .image, .newline, .end])
        }
    }

    @Test("every tower parameter matches a bundle tensor name and shape (safetensors header only)")
    func bundleTensorContract() throws {
        let bundle = URL(fileURLWithPath: "/Users/eric/models/DeepSeek-V4.1-Flash-1017-JANG-Q2Q3-IMATRIX")
        let index = bundle.appendingPathComponent("model.safetensors.index.json")
        guard FileManager.default.fileExists(atPath: index.path) else { return }
        let config = try JSONDecoder().decode(
            DeepseekV4Configuration.self, from: Data(contentsOf: bundle.appendingPathComponent("config.json")))
        let vision = try #require(config.vision)
        #expect(vision == Self.bundleVision)
        let map = try #require(
            (try JSONSerialization.jsonObject(with: Data(contentsOf: index)) as? [String: Any])?["weight_map"]
                as? [String: String])
        var headers: [String: [String: EngramSafetensors.Entry]] = [:]
        func entry(_ key: String) throws -> EngramSafetensors.Entry? {
            guard let file = map[key] else { return nil }
            if headers[file] == nil {
                headers[file] = try EngramSafetensors.header(bundle.appendingPathComponent(file)).entries
            }
            return headers[file]?[key]
        }
        let tower = DeepseekV41VisionModel(config: vision)
        let params = tower.parameters().flattened()
        #expect(params.count == 2 + 32 * 8 + 1 + 4 + 3)
        for (key, value) in params {
            let e = try entry(key)
            #expect(e?.shape == value.shape, "\(key): bundle \(e?.shape ?? []) vs module \(value.shape)")
            #expect(e?.dtype == "BF16", "\(key)")
        }
        // The bundle carries no tower tensor the module would leave unloaded.
        let bundleTower = Set(map.keys.filter { DeepseekV41VisionModel.ownsBundleKey($0) })
        #expect(bundleTower == Set(params.map(\.0)))
        // One VL routing bias per layer, shaped like the gate bias.
        for layer in 0 ..< config.numHiddenLayers {
            let e = try entry("layers.\(layer).ffn.gate.bias_vl")
            #expect(e?.shape == [config.nRoutedExperts], "layer \(layer)")
        }
    }

    // MARK: Prompt assembly (bundle encoder goldens)

    /// `encoding.py encode_messages(..., reasoning_effort="high", return_multi_modal_data=True)`.
    static let goldenPrompts: [String: String] = {
        let json = #"""
{"image_turn_thinking": "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜><｜deepseek_image｜>\n\nWhat color is this image? Answer in one word.<｜Assistant｜><think>", "image_turn_chat": "<｜begin▁of▁sentence｜><｜User｜><｜deepseek_image｜>\n\nWhat color is this image? Answer in one word.<｜Assistant｜></think>", "follow_up_thinking": "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜><｜deepseek_image｜>\n\nWhat color is this image? Answer in one word.<｜Assistant｜></think>Red<｜end▁of▁sentence｜><｜User｜>Name a fruit that has this color.<｜Assistant｜><think>", "follow_up_chat": "<｜begin▁of▁sentence｜><｜User｜><｜deepseek_image｜>\n\nWhat color is this image? Answer in one word.<｜Assistant｜></think>Red<｜end▁of▁sentence｜><｜User｜>Name a fruit that has this color.<｜Assistant｜></think>", "mid_conversation_thinking": "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>Hi<｜Assistant｜></think>Hello!<｜end▁of▁sentence｜><｜User｜>Look:\n\n<｜deepseek_image｜>\n\nWhat is it?<｜Assistant｜><think>", "mid_conversation_chat": "<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜></think>Hello!<｜end▁of▁sentence｜><｜User｜>Look:\n\n<｜deepseek_image｜>\n\nWhat is it?<｜Assistant｜></think>", "two_images_thinking": "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\nYou compare images.<｜User｜><｜deepseek_image｜>\n\n<｜deepseek_image｜>\n\nSame color?<｜Assistant｜><think>", "two_images_chat": "<｜begin▁of▁sentence｜><｜System｜>You compare images.<｜User｜><｜deepseek_image｜>\n\n<｜deepseek_image｜>\n\nSame color?<｜Assistant｜></think>"}
"""#
        return try! JSONDecoder().decode([String: String].self, from: Data(json.utf8))
    }()

    nonisolated(unsafe) static let red = UserInput.Image.url(URL(fileURLWithPath: "/nonexistent/red.png"))
    nonisolated(unsafe) static let blue = UserInput.Image.url(URL(fileURLWithPath: "/nonexistent/blue.png"))
    static let question = "What color is this image? Answer in one word."

    nonisolated(unsafe) static let promptCases: [String: [Chat.Message]] = [
        "image_turn": [.user(question, images: [red])],
        "follow_up": [
            .user(question, images: [red]), .assistant("Red"), .user("Name a fruit that has this color."),
        ],
        "mid_conversation": [
            .user("Hi"), .assistant("Hello!"),
            Chat.Message(
                role: .user, content: "Look:\n\nWhat is it?", images: [red],
                contentParts: [.text("Look:"), .image, .text("What is it?")]),
        ],
        "two_images": [.system("You compare images."), .user("Same color?", images: [red, blue])],
    ]

    @Test("image placeholders render in their own turn exactly like encoding.py",
          arguments: Array(promptCases.keys), ["thinking", "chat"])
    func promptAssembly(_ name: String, _ mode: String) throws {
        let chat = Self.promptCases[name]!
        let generated = DefaultMessageGenerator().generate(messages: chat)
        let anchored = try DeepseekV41VisionPrompt.anchorImages(
            chat: chat, looseImages: [], generated: generated)
        #expect(anchored.images.count == chat.reduce(0) { $0 + $1.images.count })
        let context: [String: any Sendable] = mode == "thinking"
            ? ["reasoning_effort": "high"] : ["enable_thinking": false]
        let prompt = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: anchored.messages, tools: nil, additionalContext: context,
            addGenerationPrompt: true, family: .deepseekV41)
        #expect(prompt == Self.goldenPrompts["\(name)_\(mode)"])
    }

    @Test("placeholder text in a message is refused; loose images go to the last user turn")
    func promptEdgeCases() throws {
        let bad: [Chat.Message] = [.user("see \(DeepseekV41VisionPrompt.placeholder)", images: [Self.red])]
        #expect(throws: DeepseekV41VisionError.self) {
            _ = try DeepseekV41VisionPrompt.anchorImages(
                chat: bad, looseImages: [], generated: DefaultMessageGenerator().generate(messages: bad))
        }
        let raw: [Message] = [["role": "user", "content": "first"], ["role": "assistant", "content": "ok"],
                              ["role": "user", "content": "second"]]
        let loose = try DeepseekV41VisionPrompt.anchorImages(
            chat: nil, looseImages: [Self.red, Self.blue], generated: raw)
        #expect(loose.images.count == 2)
        #expect(loose.messages[0]["content"] as? String == "first")
        let p = DeepseekV41VisionPrompt.placeholder
        #expect(loose.messages[2]["content"] as? String == "\(p)\n\n\(p)\n\nsecond")
    }

    @Test("placeholder expansion and cache-boundary remapping (prepare_vl_inputs)")
    func expansion() throws {
        let img = 120
        let g1 = DeepseekV41ImageGrid(vitH: 4, vitW: 5, patchSize: 2, downsampleRatio: 3)  // 8 tokens
        let g2 = DeepseekV41ImageGrid(vitH: 3, vitW: 3, patchSize: 2, downsampleRatio: 3)  // 1x1 -> 4
        let raw = [0, 5, img, 6, 7, img, 8]
        let (tokens, map) = try DeepseekV41VisionPrompt.expand(tokens: raw, imageTokenId: img, grids: [g1, g2])
        #expect(tokens == [0, 5] + Array(repeating: img, count: 8) + [6, 7] + Array(repeating: img, count: 4) + [8])
        #expect(map == [0, 1, 2, 10, 11, 12, 16, 17])
        // A boundary after turn 1 (raw 4 = through token 6) moves past the first span.
        #expect(DeepseekV41VisionPrompt.remapBoundaries([2, 4, 7, 99], rawToExpanded: map) == [2, 11, 17])
        #expect(throws: DeepseekV41VisionError.self) {
            _ = try DeepseekV41VisionPrompt.expand(tokens: raw, imageTokenId: img, grids: [g1])
        }
    }

    // MARK: Span layout and chunking

    static let g8 = DeepseekV41ImageGrid(vitH: 4, vitW: 5, patchSize: 2, downsampleRatio: 3)  // 8 tokens, 20 patches
    static let g4 = DeepseekV41ImageGrid(vitH: 3, vitW: 3, patchSize: 2, downsampleRatio: 3)  // 4 tokens, 9 patches

    @Test("spans are located in order, adjacent images split by their lengths, suffixes match the last images")
    func spanLocation() throws {
        let img = 120
        let t = Array(repeating: img, count: 8)
        let u = Array(repeating: img, count: 4)
        let full = [1, 2] + t + [3] + u + u + [4]
        let spans = try DeepseekV41VisionLayout.spans(tokens: full, imageTokenId: img, grids: [Self.g8, Self.g4, Self.g4])
        #expect(spans.map(\.start) == [2, 11, 15])
        #expect(spans.map(\.imageIndex) == [0, 1, 2])
        #expect(spans.map(\.patchRows) == [0 ..< 20, 20 ..< 29, 29 ..< 38])
        // A restored-prefix suffix carrying only the last image.
        let suffix = [9] + u + [4]
        let tail = try DeepseekV41VisionLayout.spans(tokens: suffix, imageTokenId: img, grids: [Self.g8, Self.g4, Self.g4])
        #expect(tail.map(\.start) == [1] && tail.map(\.imageIndex) == [2] && tail[0].patchRows == 29 ..< 38)
        // No placeholders: no spans (every image is inside the restored prefix).
        #expect(try DeepseekV41VisionLayout.spans(tokens: [1, 2, 3], imageTokenId: img, grids: [Self.g8]).isEmpty)
        // A partial span is refused.
        #expect(throws: DeepseekV41VisionError.self) {
            _ = try DeepseekV41VisionLayout.spans(tokens: [1] + Array(t.prefix(5)) + [2], imageTokenId: img, grids: [Self.g8])
        }
    }

    static func span(_ start: Int, _ grid: DeepseekV41ImageGrid) -> DeepseekV41ImageSpan {
        DeepseekV41ImageSpan(start: start, grid: grid, imageIndex: 0, patchRows: 0 ..< grid.patchCount)
    }

    @Test("a chunk never splits an image: stop before it, or take all of it when it starts the chunk")
    func chunkRule() {
        let spans = [Self.span(10, Self.g8)]  // [10, 18)
        #expect(DeepseekV41VisionLayout.chunkLength(pos: 0, proposed: 14, spans: spans) == 10)
        #expect(DeepseekV41VisionLayout.chunkLength(pos: 10, proposed: 3, spans: spans) == 8)
        #expect(DeepseekV41VisionLayout.chunkLength(pos: 0, proposed: 18, spans: spans) == 18)
        #expect(DeepseekV41VisionLayout.chunkLength(pos: 0, proposed: 10, spans: spans) == 10)
        #expect(DeepseekV41VisionLayout.chunkLength(pos: 0, proposed: 5, spans: spans) == 5)
    }

    @Test("prefill chunk plans: whole spans per chunk, a text-only remainder within one step",
          arguments: [1, 2, 3, 5, 7, 9, 64], [2, 4, 16])
    func prefillPlans(outer: Int, step: Int) throws {
        let T = 40
        let spans = [Self.span(3, Self.g8), Self.span(11, Self.g4), Self.span(25, Self.g8)]  // [3,11) [11,15) [25,33)
        let chunks = try DeepseekV41VisionLayout.prefillChunks(
            tokenCount: T, stepSize: step, outerStep: max(step, outer), spans: spans)
        var pos = 0
        for c in chunks {
            #expect(c.lowerBound == pos && !c.isEmpty)
            for s in spans {
                let inside = s.start >= c.lowerBound && s.end <= c.upperBound
                let disjoint = s.end <= c.lowerBound || s.start >= c.upperBound
                #expect(inside || disjoint, "chunk \(c) splits span \(s.start)..<\(s.end)")
            }
            pos = c.upperBound
        }
        #expect(pos >= 33)  // every image consumed
        #expect(T - pos >= 1 && T - pos <= max(step, 1))
        #expect(throws: DeepseekV41VisionError.self) {
            _ = try DeepseekV41VisionLayout.prefillChunks(
                tokenCount: 33, stepSize: step, outerStep: outer, spans: spans)
        }
    }

    @Test("chunk splice places each span at its chunk-relative position; masks mark exactly the spans")
    func chunkSplice() {
        Device.withDefaultDevice(.cpu) {
            let L = 12, D = 3
            let text = MLXArray((0 ..< L * D).map { Float($0) }, [1, L, D])
            let a = MLXArray.ones([4, D]) * Float(-1)
            let b = MLXArray.ones([2, D]) * Float(-2)
            let chunk = DeepseekV41VisionChunk.make(
                textEmbeddings: text, spans: [(start: 7, rows: b), (start: 1, rows: a)], length: L)
            #expect(chunk.embeddings.shape == [1, L, D])
            let rows = (0 ..< L).map { chunk.embeddings[0, $0].asArray(Float.self) }
            for i in 0 ..< L {
                let expected: [Float]
                if (1 ..< 5).contains(i) { expected = [-1, -1, -1] }
                else if (7 ..< 9).contains(i) { expected = [-2, -2, -2] }
                else { expected = (0 ..< D).map { Float(i * D + $0) } }
                #expect(rows[i] == expected, "row \(i)")
            }
            let keep = (0 ..< L).map { !((1 ..< 5).contains($0) || (7 ..< 9).contains($0)) }
            #expect(chunk.keep == keep)
            #expect(chunk.imageMask.asArray(Bool.self) == keep.map { !$0 })
            #expect(chunk.textMask.asArray(Float.self) == keep.map { $0 ? 1 : 0 })
        }
    }

    // MARK: Engram + routing

    @Test("image tokens hash DEAD: they and every n-gram spanning them read pad")
    func engramDeadHashing() {
        let hasher = EngramNgramHasher(EngramV41FocusedTests.syntheticAux())
        let ids: [Int32] = [5, 6, 7, 120, 120, 120, 8, 9, 10, 11]
        let keep = ids.map { $0 != 120 }
        let comp = hasher.compress(ids, keep: keep)
        #expect(comp[3 ..< 6].allSatisfy { $0 == EngramNgramHasher.DEAD })
        #expect(comp[0 ..< 3] == hasher.compress(Array(ids[0 ..< 3]))[0 ..< 3])
        let masked = hasher.hashes(comp, startPos: 0, previous: [])
        let plain = hasher.hashes(hasher.compress(ids), startPos: 0, previous: [])
        let per = hasher.aux.multipliers.count * hasher.aux.cols
        // Tokens before the image hash as text; the image and the n-grams reaching back into it
        // (maxNgram 4: positions 6, 7, 8 still see a DEAD within 3 back) differ; position 9 is clear.
        #expect(masked[0 ..< (3 * per)] == plain[0 ..< (3 * per)])
        #expect(masked[(6 * per) ..< (9 * per)] != plain[(6 * per) ..< (9 * per)])
        #expect(masked[(9 * per)...] == plain[(9 * per)...])
        // Chunked with the DEAD-carrying history equals one shot (restore safety).
        let head = hasher.hashes(Array(comp[..<5]), startPos: 0, previous: [])
        let tail = hasher.hashes(Array(comp[5...]), startPos: 5, previous: Array(comp[2 ..< 5]))
        #expect(head + tail == masked)
    }

    @Test("image tokens select experts with bias_vl, text tokens with bias")
    func biasVLRouting() {
        Device.withDefaultDevice(.cpu) {
            var c = Self.tinyBackboneConfig(vision: true)
            c.nRoutedExperts = 4
            c.numExpertsPerTok = 1
            let gate = DeepseekV4MoEGate(config: c, layerIdx: 1)
            #expect(gate.biasVl != nil)
            try! gate.update(
                parameters: ModuleParameters.unflattened([
                    ("weight", MLXArray.zeros([4, c.hiddenSize])),
                    ("bias", MLXArray([Float(5), 0, 0, 0])),
                    ("bias_vl", MLXArray([Float(0), 0, 0, 5])),
                ]), verify: [])
            let x = DeepseekV41DSparkFocusedTests.random([1, 4, c.hiddenSize], seed: 3)
            let mask = MLXArray([false, true, true, false]).reshaped(1, 4)
            let (idx, _) = gate(x, inputIds: nil, imageMask: mask)
            #expect(idx.reshaped(-1).asArray(UInt32.self) == [0, 3, 3, 0])
            let (plain, w0) = gate(x, inputIds: nil)
            #expect(plain.reshaped(-1).asArray(UInt32.self) == [0, 0, 0, 0])
            // An all-text mask selects exactly like the unmasked (compiled) gate.
            let (none, w1) = gate(x, inputIds: nil, imageMask: MLXArray.zeros([1, 4], type: Bool.self))
            #expect(none.asArray(UInt32.self) == plain.asArray(UInt32.self))
            #expect(MLX.abs(w1 - w0).max().item(Float.self) < 1e-6)
            // DSV4-0731 and text-only V4.1 gates carry no bias_vl.
            #expect(DeepseekV4MoEGate(config: Self.tinyBackboneConfig(vision: false), layerIdx: 1).biasVl == nil)
            var v4 = DeepseekV41DSparkFocusedTests.tinyConfig(v41: false)
            v4.vision = Self.tinyVision
            #expect(DeepseekV4MoEGate(config: v4, layerIdx: 1).biasVl == nil)
        }
    }

    // MARK: Tiny V4.1 backbone with engram + vision

    static func tinyBackboneConfig(vision: Bool) -> DeepseekV4Configuration {
        var c = DeepseekV41DSparkFocusedTests.tinyConfig()
        c.numHashLayers = 0
        c.engramLayerIds = [1]
        c.engramNHeads = 2
        c.engramHeadDim = 8
        c.engramMaxNgramSize = 3
        if vision {
            var v = tinyVision
            v.textHiddenSize = c.hiddenSize
            c.vision = v
        }
        return c
    }

    /// Minimal safetensors writer (header JSON + raw bytes).
    static func writeSafetensors(
        _ url: URL, _ tensors: [(name: String, dtype: String, shape: [Int], bytes: [UInt8])],
        metadata: [String: String]? = nil
    ) throws {
        var header: [String: Any] = [:]
        var offset = 0
        for t in tensors {
            header[t.name] = ["dtype": t.dtype, "shape": t.shape, "data_offsets": [offset, offset + t.bytes.count]]
            offset += t.bytes.count
        }
        if let metadata { header["__metadata__"] = metadata }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while json.count % 8 != 0 { json.append(0x20) }
        var data = Data()
        var len = UInt64(json.count).littleEndian
        data.append(Data(bytes: &len, count: 8))
        data.append(json)
        for t in tensors { data.append(contentsOf: t.bytes) }
        try data.write(to: url)
    }

    static func bytes<T>(_ values: [T]) -> [UInt8] {
        values.withUnsafeBytes { Array($0) }
    }

    /// A bundle directory with a synthetic engram aux + one FP8 table (layer 1), as the loader
    /// resolves them (`engram_aux.safetensors` + `engram/`).
    static func writeEngramBundle(vocab: Int) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dsv41-vl-bundle-\(UUID().uuidString)", isDirectory: true)
        let engram = dir.appendingPathComponent("engram", isDirectory: true)
        try FileManager.default.createDirectory(at: engram, withIntermediateDirectories: true)
        let nHeads = 2, maxNgram = 3, headDim = 8, cols = (maxNgram - 1) * nHeads
        var rng = SplitMix64(seed: 41)
        let tokenMap = (0 ..< vocab).map { _ in Int32(rng.next() % 50) }
        let multipliers = (0 ..< maxNgram).map { _ in Int64(rng.next() % 100_003) * 2 + 1 }
        let primes: [Int64] = [11, 13, 17, 19]  // [1][maxNgram-1][nHeads]
        let offsets = (0 ..< cols).map { Int64($0 * 20) }
        try writeSafetensors(dir.appendingPathComponent("engram_aux.safetensors"), [
            ("token_map", "I32", [vocab], bytes(tokenMap)),
            ("multipliers", "I64", [1, maxNgram], bytes(multipliers)),
            ("primes", "I64", [1, maxNgram - 1, nHeads], bytes(primes)),
            ("offsets", "I64", [1, cols], bytes(offsets)),
            ("pad_id", "I64", [1], bytes([Int64(49)])),
        ], metadata: ["head_dim": "\(headDim)", "layer_ids": "1", "compressed_vocab_size": "50"])
        let rows = cols * 20
        let weight = (0 ..< rows * headDim).map { _ in UInt8(0x28 + rng.next() % 16) | UInt8((rng.next() % 2) << 7) }
        let scale = [UInt8](repeating: 127, count: rows)  // E8M0 127 = 1.0, one scale per row
        try writeSafetensors(engram.appendingPathComponent("engram-00000.safetensors"), [
            ("layers.1.engram.embed.weight", "F8_E4M3", [rows, headDim], weight),
            ("layers.1.engram.embed.scale", "F8_E8M0", [rows, 1], scale),
        ])
        return dir
    }

    /// Every parameter of `model` set from a seeded generator (norm-like vectors near 1).
    static func randomize(_ model: Module, seed: UInt64) {
        var updates: [(String, MLXArray)] = []
        for (i, (key, value)) in model.parameters().flattened().sorted(by: { $0.0 < $1.0 }).enumerated() {
            var a = DeepseekV41DSparkFocusedTests.random(value.shape, seed: seed &+ UInt64(i), scale: 0.08)
            if key.hasSuffix("norm.weight") || key.hasSuffix("layernorm.weight") || key.hasSuffix("_weight") {
                a = a + 1
            }
            updates.append((key, a.asType(value.dtype)))
        }
        try! model.update(parameters: ModuleParameters.unflattened(updates), verify: [.all])
        eval(model)
    }

    static func tinyModel(vision: Bool, bundle: URL, seed: UInt64 = 900) throws -> DeepseekV4Model {
        let model = DeepseekV4Model(tinyBackboneConfig(vision: vision))
        try model.configureBundle(modelDirectory: bundle)
        randomize(model, seed: seed)
        return model
    }

    /// Image prompt over the tiny backbone: text, image 0 (8 tokens), text, image 1 (4 tokens), text.
    static func imagePrompt(phase: Double = 0.5) -> LMInput {
        let img = 120
        let tokens = [0, 11, 12] + Array(repeating: img, count: 8) + [13, 14, 15, 16, 17]
            + Array(repeating: img, count: 4) + [18, 19, 20, 21, 22, 23]
        let pixels = concatenated(
            [tinyPatches(20, patch: 2, phase: phase), tinyPatches(9, patch: 2, phase: phase + 1)], axis: 0)
        return LMInput(
            text: .init(tokens: MLXArray(tokens.map(Int32.init)), tokenIds: tokens),
            image: .init(pixels: pixels, frames: [THW(1, 4, 5), THW(1, 3, 3)]),
            mediaTokenIds: [img])
    }

    /// Prefill through `prepare` (VL path with an explicit outer chunk), then the caller's final
    /// forward; returns the last-position logits.
    static func prefillLogits(
        _ model: DeepseekV4Model, _ input: LMInput, outer: Int?, window: Int = 4
    ) throws -> (logits: [Float], remainder: Int) {
        let cache = model.newCache(parameters: nil)
        let result: PrepareResult
        if let image = input.image, let vision = model.v41Vision, let outer {
            result = try model.prepareImages(
                input, image: image, vision: vision, cache: cache, windowSize: window, outerStep: outer)!
        } else {
            result = try model.prepare(input, cache: cache, windowSize: window)
        }
        guard case .tokens(let rest) = result else { fatalError("expected tokens") }
        let logits = model(rest.tokens.reshaped(1, -1), cache: cache)
        let last = logits[0, logits.dim(1) - 1].asType(.float32)
        return (last.asArray(Float.self), rest.tokens.size)
    }

    @Test("image prefill is invariant to the chunk plan (chunk-relative splice, never-split rule)")
    func backbonePrefillChunkInvariant() throws {
        try Device.withDefaultDevice(.cpu) {
            let bundle = try Self.writeEngramBundle(vocab: 128)
            defer { try? FileManager.default.removeItem(at: bundle) }
            let model = try Self.tinyModel(vision: true, bundle: bundle)
            let input = Self.imagePrompt()
            let (reference, rest) = try Self.prefillLogits(model, input, outer: 64)
            #expect(rest >= 1 && rest <= 4)
            for outer in [4, 5, 6, 9, 13] {
                let (got, r) = try Self.prefillLogits(model, input, outer: outer)
                #expect(r >= 1 && r <= 4)
                let err = zip(got, reference).map { abs($0 - $1) }.max() ?? .infinity
                #expect(err < 1e-4, "outer \(outer): max |delta| \(err)")
            }
            // The pixels reach the backbone: a different image changes the logits.
            let (other, _) = try Self.prefillLogits(model, Self.imagePrompt(phase: 2.0), outer: 64)
            #expect((zip(other, reference).map { abs($0 - $1) }.max() ?? 0) > 1e-4)
            // Tokens of an image-covering restore carry no span: the text prefill runs (nil).
            let suffix = LMInput(
                text: .init(tokens: MLXArray([Int32(18), 19, 20])), image: input.image, mediaTokenIds: [120])
            #expect(try model.prepareImages(
                suffix, image: input.image!, vision: model.v41Vision!,
                cache: model.newCache(parameters: nil), windowSize: 4) == nil)
        }
    }

    @Test("text-only V4.1 prefill is identical with and without the vision tower; 0731 builds none")
    func textPathUnchanged() throws {
        try Device.withDefaultDevice(.cpu) {
            let bundle = try Self.writeEngramBundle(vocab: 128)
            defer { try? FileManager.default.removeItem(at: bundle) }
            let withVision = try Self.tinyModel(vision: true, bundle: bundle)
            let textOnly = DeepseekV4Model(Self.tinyBackboneConfig(vision: false))
            try textOnly.configureBundle(modelDirectory: bundle)
            // Same backbone weights: copy every parameter the text-only model has.
            let shared = withVision.parameters().flattened().filter { key, _ in
                !key.hasPrefix("v41_vision.") && !key.hasSuffix(".bias_vl")
            }
            try textOnly.update(parameters: ModuleParameters.unflattened(shared), verify: [.all])
            #expect(withVision.v41Vision != nil && textOnly.v41Vision == nil)
            #expect(withVision.modalities == [.text, .vision] && textOnly.modalities == [.text])
            let tokens = (0 ..< 23).map { Int32(($0 * 7 + 3) % 110) }
            let input = LMInput(tokens: MLXArray(tokens))
            let (a, _) = try Self.prefillLogits(withVision, input, outer: nil)
            let (b, _) = try Self.prefillLogits(textOnly, input, outer: nil)
            #expect(a == b)

            // DSV4-0731: never a tower or bias_vl, even if a vision block were present; sanitize
            // drops the vision tensors exactly as before.
            var v4 = DeepseekV41DSparkFocusedTests.tinyConfig(v41: false)
            v4.vision = Self.tinyVision
            let m0731 = DeepseekV4Model(v4)
            #expect(m0731.v41Vision == nil && m0731.modalities == [.text])
            #expect(!m0731.parameters().flattened().contains { $0.0.contains("bias_vl") || $0.0.contains("vision") })
            let raw: [String: MLXArray] = [
                "vision.norm.weight": MLXArray.ones([4]), "aligner.w1.bias": MLXArray.ones([4]),
                "image_start": MLXArray.ones([4]), "layers.0.ffn.gate.bias_vl": MLXArray.ones([4]),
            ]
            #expect(m0731.sanitize(weights: raw).isEmpty)
            #expect(textOnly.sanitize(weights: raw).isEmpty)
            let mapped = withVision.sanitize(weights: raw)
            #expect(Set(mapped.keys) == [
                "v41_vision.vision.norm.weight", "v41_vision.aligner.w1.bias", "v41_vision.image_start",
                "model.layers.0.mlp.gate.bias_vl",
            ])
        }
    }

    @Test("VMLX_DSV41_VL=0 builds text-only; vision_config decodes only for V4.1 bundles")
    func gatingAndConfig() throws {
        #expect(DeepseekV41VisionConfiguration.enabled(environment: [:]))
        #expect(DeepseekV41VisionConfiguration.enabled(environment: ["VMLX_DSV41_VL": "1"]))
        for off in ["0", "false", "OFF", "no"] {
            #expect(!DeepseekV41VisionConfiguration.enabled(environment: ["VMLX_DSV41_VL": off]))
        }
        let v41 = #"""
        {"model_type": "deepseek_v41", "image_token_id": 129264,
         "vision_config": {"model_type": "deepseek_v41_vision", "num_hidden_layers": 32, "hidden_size": 1024,
           "num_attention_heads": 16, "intermediate_size": 2816, "patch_size": 14, "rope_theta": 10000,
           "downsample_ratio": 3, "max_image_tokens": 1024, "min_pixels": 295936, "max_wh_ratio": null},
         "text_config": {"model_type": "deepseek_v41_text", "hidden_size": 5120, "engram_layer_ids": [1, 14]}}
        """#
        let c = try JSONDecoder().decode(DeepseekV4Configuration.self, from: Data(v41.utf8))
        #expect(c.vision == Self.bundleVision)
        let round = try JSONDecoder().decode(DeepseekV4Configuration.self, from: JSONEncoder().encode(c))
        #expect(round.vision == Self.bundleVision)
        let v4 = #"""
        {"model_type": "deepseek_v4", "image_token_id": 5,
         "vision_config": {"num_hidden_layers": 2, "hidden_size": 8, "num_attention_heads": 2,
           "intermediate_size": 8, "patch_size": 2, "downsample_ratio": 3}}
        """#
        let c4 = try JSONDecoder().decode(DeepseekV4Configuration.self, from: Data(v4.utf8))
        #expect(c4.vision == nil)
        let encoded = String(decoding: try JSONEncoder().encode(c4), as: UTF8.self)
        #expect(!encoded.contains("vision_config") && !encoded.contains("image_token_id"))
    }

    // MARK: Cache keys

    static func imageInput(_ color: (UInt8, UInt8, UInt8), tokens: [Int]) -> LMInput {
        let pre = DeepseekV41ImagePreprocessor.preprocess(
            DeepseekV41RGBImage(width: 64, height: 64, fill: color), config: Self.bundleVision)
        return LMInput(
            text: .init(tokens: MLXArray(tokens.map(Int32.init)), tokenIds: tokens),
            image: DeepseekV41VisionPrompt.processedImage([pre], patchSize: 14),
            mediaTokenIds: [129_264], cacheScopeSalt: "reasoning=on")
    }

    @Test("cache keys are salted with the image pixels: same image hits, a different image misses")
    func cacheKeySalting() {
        Device.withDefaultDevice(.cpu) {
            let span = Array(repeating: 129_264, count: 184)
            let turn1 = [0, 128803] + span + [271, 3085, 128804, 128822]
            let turn2 = turn1 + [33, 1, 128803, 4609, 128804]
            let params = GenerateParameters()
            let red1 = Self.imageInput((255, 0, 0), tokens: turn1)
            let red2 = Self.imageInput((255, 0, 0), tokens: turn2)
            let blue = Self.imageInput((0, 0, 255), tokens: turn1)
            let textOnly = LMInput(
                text: .init(tokens: MLXArray([Int32(0), 1]), tokenIds: [0, 1]), cacheScopeSalt: "reasoning=on")
            #expect(red1.image!.pixels.shape == [1521, 3, 14, 14])
            #expect(red1.image!.pixels.dtype == .bfloat16)
            let sRed1 = computeCacheSalt(for: red1, parameters: params)
            let sRed2 = computeCacheSalt(for: red2, parameters: params)
            let sBlue = computeCacheSalt(for: blue, parameters: params)
            // The follow-up text turn carries the same image in its first turn: same salt, so the
            // first turn's stored prefix is found; a different image changes every key.
            #expect(sRed1 == sRed2)
            #expect(sRed1 != sBlue)
            #expect(computeMediaSalt(for: red1) != nil && computeMediaSalt(for: textOnly) == nil)
            #expect(computeMediaSalt(for: red1) != computeMediaSalt(for: blue))
            #expect(sRed1 != computeCacheSalt(for: textOnly, parameters: params))
            // DeepseekV4Model makes no pre-media promise: every boundary keeps the media salt, so a
            // different image misses even the text before it (the engine's whole-request salt).
            let model = DeepseekV4Model(Self.tinyBackboneConfig(vision: true))
            let plan = CacheBoundarySaltPlan.make(
                requestSalt: sRed1, input: red1, lookupTokenIds: turn1, model: model, parameters: params)
            #expect(plan.mediaIndependentPrefix == nil)
            #expect(plan.salt(forBoundary: 1) == sRed1 && plan.salt(forBoundary: turn1.count) == sRed1)
            // A boundary after the image keeps the image with the head (stored with its pixels); a
            // boundary inside the span is refused; a suffix past the span needs no media.
            #expect(red2.inputForCacheBoundary(tokens: turn1, fullPromptTokenIds: turn2)?.image != nil)
            #expect(red2.inputForCacheBoundary(tokens: Array(turn1.prefix(50)), fullPromptTokenIds: turn2) == nil)
            #expect(!red2.cacheHitSuffixContainsMediaPlaceholder(Array(turn2.dropFirst(turn1.count))))
            #expect(red2.cacheHitSuffixContainsMediaPlaceholder(Array(turn2.dropFirst(10))))
        }
    }

    @Test("dSpark is decided on the prefilled input (text-only tail after an image-covering hit)")
    func dsparkTailGateSource() throws {
        let source = try String(
            contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent()
                .deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("Libraries/MLXLMCommon/Evaluate.swift"),
            encoding: .utf8)
        #expect(source.contains("setupDSpark(\n                input: inputForPrepare,"))
        #expect(source.contains("if state != nil || input.image != nil || input.video != nil { return \"media inputs\" }"))
    }
}
