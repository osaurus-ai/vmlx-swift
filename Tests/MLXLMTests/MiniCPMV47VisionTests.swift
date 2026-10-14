// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// MiniCPM-V 4.7 vision path against transformers 5.18 reference dumps (`Resources/minicpmv47/*.json`, produced by
// the PIL-backend `MiniCPMV4_7Processor` and `MiniCPMV4_7Model.get_rope_index` on the synthetic PNGs next to them).
// The large binary dumps (pixel values, float32 tower features) live outside the repo; those rows run only when
// `MINICPMV47_FIXTURES` (or the default private evidence folder) is present, and the tower row additionally needs
// the 1B bundle's vision weights.

import CoreImage
import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXNN
import Testing
@preconcurrency import VMLXTokenizers

@testable import MLXVLM

struct MiniCPMV47Manifest: Decodable {
    let prompt: String
    let enable_thinking: Bool?
    let images: [String]
    let image_u8_chw_shapes: [[Int]]
    let chat_text: String
    let input_ids: [Int]
    let mm_token_type_ids: [Int]
    let target_sizes: [[Int]]
    let grids: [[Int]]
    let num_patches_per_image: [Int]
    let pixel_values_shape: [Int]
    let position_ids: [[Int]]
    let rope_delta: Int
    let features_shape: [Int]
    let image_token_count: Int
    let nearest_position_ids: [Int]
    let window_index: [Int]
}

enum MiniCPMV47Fixtures {
    static let cases = ["scene", "odd", "portrait", "odd_nothink", "two_images"]
    static let markers = (imageStart: 248078, imageEnd: 248079, sliceStart: 248088, sliceEnd: 248089, newline: 198)
    static let imageTokenId = 248056
    static let videoTokenId = 248057

    static func manifest(_ name: String) throws -> MiniCPMV47Manifest {
        let url = try #require(Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "minicpmv47"))
        return try JSONDecoder().decode(MiniCPMV47Manifest.self, from: Data(contentsOf: url))
    }

    static func imageURL(_ file: String) throws -> URL {
        let name = (file as NSString).deletingPathExtension
        return try #require(Bundle.module.url(forResource: name, withExtension: "png", subdirectory: "minicpmv47"))
    }

    /// Folder with the reference `.bin` dumps, or nil when absent.
    static var binaryRoot: URL? {
        let env = ProcessInfo.processInfo.environment["MINICPMV47_FIXTURES"]
        let fallback = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("vmlx-private-evidence/minicpm-vision/fixtures").path
        let path = env ?? fallback
        return FileManager.default.fileExists(atPath: path) ? URL(fileURLWithPath: path) : nil
    }

    static var bundleDirectory: URL? {
        let env = ProcessInfo.processInfo.environment["MINICPMV47_BUNDLE"]
        let fallback = "/Volumes/EricMLWork/models/JANGQ-AI/MiniCPM-V-4.7-1B-JANG_8M"
        let path = env ?? fallback
        return FileManager.default.fileExists(atPath: path + "/config.json") ? URL(fileURLWithPath: path) : nil
    }

    static func floats(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    static func bytes(_ url: URL) throws -> [UInt8] {
        [UInt8](try Data(contentsOf: url))
    }

    /// Reference `pixel_values[0]` `[3, 14, N*14]` as `[N, 14, 14, 3]` patches.
    static func referencePatches(_ root: URL, _ name: String, _ manifest: MiniCPMV47Manifest) throws -> MLXArray {
        let values = try floats(root.appendingPathComponent("\(name)/pixel_values_f32.bin"))
        let shape = manifest.pixel_values_shape  // [1, 3, 14, N*14]
        let n = shape[3] / 14
        return MLXArray(values, [3, 14, n, 14]).transposed(2, 1, 3, 0)
    }

    static func crops(_ manifest: MiniCPMV47Manifest) -> [THW] {
        manifest.target_sizes.map { THW(1, $0[0], $0[1]) }
    }
}

@Suite("MiniCPM-V 4.7 canvas M-RoPE and slicing math", .serialized)
struct MiniCPMV47CanvasTests {

    @Test(arguments: MiniCPMV47Fixtures.cases)
    func canvasRopeIndexMatchesReference(name: String) throws {
        let m = try MiniCPMV47Fixtures.manifest(name)
        let (positions, delta) = MiniCPMV47Canvas.ropeIndex(
            inputIds: m.input_ids, imageGrids: MiniCPMV47Fixtures.crops(m), videoGrids: [],
            imageTokenId: MiniCPMV47Fixtures.imageTokenId, videoTokenId: MiniCPMV47Fixtures.videoTokenId,
            markers: MiniCPMV47Fixtures.markers, mergeFactor: 4)
        for d in 0 ..< 3 {
            let expected = m.position_ids[d].map(Int32.init)
            let mismatches = zip(positions[d], expected).enumerated().filter { $0.element.0 != $0.element.1 }
            #expect(mismatches.isEmpty, "\(name) axis \(d): \(mismatches.count) mismatches, first \(mismatches.prefix(5).map { ($0.offset, $0.element.0, $0.element.1) })")
        }
        #expect(delta == Int32(m.rope_delta), "\(name) rope delta")
    }

    @Test(arguments: MiniCPMV47Fixtures.cases)
    func batchedRopeIndexMatchesReference(name: String) throws {
        let m = try MiniCPMV47Fixtures.manifest(name)
        var canvas = MiniCPMV47CanvasConfiguration()
        canvas.imageStartId = 248078
        canvas.imageEndId = 248079
        canvas.sliceStartId = 248088
        canvas.sliceEndId = 248089
        canvas.newlineId = 198
        let ids = MLXArray(m.input_ids.map(Int32.init), [1, m.input_ids.count])
        let (positions, deltas) = MiniCPMV47Canvas.ropeIndex(
            inputIds: ids, imageGrids: MiniCPMV47Fixtures.crops(m), videoGrids: nil,
            imageTokenId: MiniCPMV47Fixtures.imageTokenId, videoTokenId: MiniCPMV47Fixtures.videoTokenId,
            canvas: canvas)
        #expect(positions.shape == [3, 1, m.input_ids.count])
        let flat = positions.asArray(Int32.self)
        let expected = m.position_ids.flatMap { $0.map(Int32.init) }
        #expect(flat == expected, "\(name) batched positions")
        #expect(deltas.asArray(Int32.self) == [Int32(m.rope_delta)])
    }

    @Test(arguments: MiniCPMV47Fixtures.cases)
    func slicingMathMatchesReference(name: String) throws {
        let m = try MiniCPMV47Fixtures.manifest(name)
        var expectedCrops = m.target_sizes.makeIterator()
        for (shape, grid) in zip(m.image_u8_chw_shapes, m.grids) {
            let (h, w) = (shape[1], shape[2])
            let sliced = MiniCPMV47ImageProcessing.slicedGrid(height: h, width: w, maxSliceNums: 9, scaleResolution: 448)
            #expect((sliced?.rows ?? 0) == grid[0] && (sliced?.cols ?? 0) == grid[1], "\(name) grid for \(w)x\(h)")
            let (th, tw) = MiniCPMV47ImageProcessing.findBestResize(
                height: Double(h), width: Double(w), scaleResolution: 448, patchSize: 14, allowUpscale: sliced == nil)
            #expect([th / 14, tw / 14] == expectedCrops.next()!, "\(name) thumbnail grid")
            if let sliced {
                let (rh, rw) = MiniCPMV47ImageProcessing.refineSize(
                    height: h, width: w, grid: sliced, scaleResolution: 448, patchSize: 14, allowUpscale: true)
                for _ in 0 ..< sliced.rows * sliced.cols {
                    #expect([rh / sliced.rows / 14, rw / sliced.cols / 14] == expectedCrops.next()!, "\(name) slice grid")
                }
            }
        }
        #expect(expectedCrops.next() == nil)
    }

    @Test(arguments: MiniCPMV47Fixtures.cases)
    func nearestPositionIdsAndWindowIndexMatchReference(name: String) throws {
        let m = try MiniCPMV47Fixtures.manifest(name)
        var nearest: [Int32] = []
        var window: [Int32] = []
        var offset = 0
        for crop in m.target_sizes {
            nearest.append(contentsOf: MiniCPMV47Vision.nearestPositionIds(height: crop[0], width: crop[1], numPatchesPerSide: 70))
            window.append(contentsOf: MiniCPMV47Vision.windowIndex(height: crop[0], width: crop[1], window: 2, offset: offset))
            offset += crop[0] * crop[1]
        }
        #expect(nearest == m.nearest_position_ids.map(Int32.init), "\(name) nearest position ids")
        #expect(window == m.window_index.map(Int32.init), "\(name) window index")
    }

    @Test func placeholderLayoutMatchesReferenceWithoutTokenizer() throws {
        // `scene`: one image, 2x4 slices. Rebuild the placeholder span from the grids and compare with the
        // reference token span between the first <image_id> and the text.
        let m = try MiniCPMV47Fixtures.manifest("scene")
        let crops = MiniCPMV47ImageProcessing.ProcessedCrops(
            patches: MLXArray.zeros([1]), grids: MiniCPMV47Fixtures.crops(m), sliceGrid: (2, 4))
        let tokens = MiniCPMV47ImageProcessing.placeholderTokens(
            crops: crops, localIndex: 0, useImageId: true, padTokenId: 248056, tokensPerPatchDivisor: 16,
            markers: MiniCPMV47Fixtures.markers, imageIdStart: 248090, imageIdEnd: 248091, digitTokens: { _ in [15] })
        let start = try #require(m.input_ids.firstIndex(of: 248090))
        let end = try #require(m.input_ids.lastIndex(of: 248089)) + 1
        #expect(Array(m.input_ids[start ..< end]) == tokens)
    }

    @Test func ensureDivideUsesPythonRounding() {
        // round(2.5) == 2 and round(3.5) == 4 in Python; 140/56 = 2.5 -> 112, 196/56 = 3.5 -> 224.
        #expect(MiniCPMV47ImageProcessing.ensureDivide(140, 56) == 112)
        #expect(MiniCPMV47ImageProcessing.ensureDivide(196, 56) == 224)
        #expect(MiniCPMV47ImageProcessing.ensureDivide(3, 56) == 56)
    }
}

@Suite("MiniCPM-V 4.7 pixel preprocessing (bit-exact vs PIL reference)", .serialized,
       .enabled(if: MiniCPMV47Fixtures.binaryRoot != nil))
struct MiniCPMV47PixelTests {

    @Test(arguments: ["scene", "odd", "portrait"])
    func decodedBytesAndPatchesMatchReference(name: String) throws {
        let root = try #require(MiniCPMV47Fixtures.binaryRoot)
        let m = try MiniCPMV47Fixtures.manifest(name)
        let image = try #require(CIImage(contentsOf: try MiniCPMV47Fixtures.imageURL(m.images[0])))
        let rgb = try MiniCPMV47ImageProcessing.rgb8(from: image)
        let shape = m.image_u8_chw_shapes[0]
        #expect(rgb.height == shape[1] && rgb.width == shape[2])
        let reference = try MiniCPMV47Fixtures.bytes(root.appendingPathComponent("\(name)/image0_u8_chw.bin"))
        var byteMismatches = 0
        let plane = shape[1] * shape[2]
        for y in 0 ..< rgb.height {
            for x in 0 ..< rgb.width {
                for c in 0 ..< 3 where rgb[y, x, c] != reference[c * plane + y * rgb.width + x] {
                    byteMismatches += 1
                }
            }
        }
        #expect(byteMismatches == 0, "\(name): \(byteMismatches) decoded bytes differ from PIL")

        let processed = MiniCPMV47ImageProcessing.process(
            rgb, maxSliceNums: 9, scaleResolution: 448, patchSize: 14, sliceMode: true,
            mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5])
        #expect(processed.grids.map { [$0.h, $0.w] } == m.target_sizes)
        let expected = try MiniCPMV47Fixtures.referencePatches(root, name, m)
        #expect(processed.patches.shape == expected.shape)
        let diff = abs(processed.patches - expected)
        let maxAbs = diff.max().item(Float.self)
        let differing = (diff .> 0).sum().item(Int.self)
        #expect(maxAbs == 0, "\(name): pixel max-abs \(maxAbs), \(differing) of \(diff.size) values differ (1/255 = 0.0078 per uint8 step)")
        print("[minicpmv47 pixels] \(name): max-abs \(maxAbs) differing \(differing)/\(diff.size)")
    }
}

@Suite("MiniCPM-V 4.7 processor (real tokenizer)", .serialized,
       .enabled(if: MiniCPMV47Fixtures.bundleDirectory != nil))
struct MiniCPMV47ProcessorTests {

    @Test(arguments: MiniCPMV47Fixtures.cases)
    func prepareMatchesReferenceTokens(name: String) async throws {
        let bundle = try #require(MiniCPMV47Fixtures.bundleDirectory)
        let m = try MiniCPMV47Fixtures.manifest(name)
        let tokenizer = try await #huggingFaceTokenizerLoader().load(from: bundle)
        let processor = MiniCPMV47Processor(try processorConfiguration(bundle), tokenizer: tokenizer)
        let images = try m.images.map { UserInput.Image.url(try MiniCPMV47Fixtures.imageURL($0)) }
        var input = UserInput(chat: [.user(m.prompt, images: images)])
        if let thinking = m.enable_thinking { input.additionalContext = ["enable_thinking": thinking] }
        let prepared = try await processor.prepare(input: input)
        let ids = try #require(prepared.text.tokenIds)
        if ids != m.input_ids {
            let firstDiff = zip(ids, m.input_ids).enumerated().first { $0.element.0 != $0.element.1 }?.offset ?? min(ids.count, m.input_ids.count)
            Issue.record("\(name): tokens differ at \(firstDiff) (swift \(ids.count) vs ref \(m.input_ids.count)); swift[\(firstDiff)...] = \(ids.dropFirst(firstDiff).prefix(8)), ref = \(m.input_ids.dropFirst(firstDiff).prefix(8))")
        }
        #expect(prepared.image?.frames?.map { [$0.h, $0.w] } == m.target_sizes)
        #expect(prepared.image?.pixels.shape == [m.pixel_values_shape[3] / 14, 14, 14, 3])
        #expect(prepared.mediaTokenIds == [MiniCPMV47Fixtures.imageTokenId])
    }

    /// Video (and video + image) prompts against the reference run on the SAME AVFoundation-decoded frames:
    /// token ids, modality labels, per-crop grids, canvas positions and rope delta must be identical.
    @Test(arguments: MiniCPMV47VideoFixtures.cases)
    func prepareMatchesReferenceVideoTokens(name: String) async throws {
        let bundle = try #require(MiniCPMV47Fixtures.bundleDirectory)
        let m = try MiniCPMV47VideoFixtures.manifest(name)
        let tokenizer = try await #huggingFaceTokenizerLoader().load(from: bundle)
        let processor = MiniCPMV47Processor(try processorConfiguration(bundle), tokenizer: tokenizer)
        let video = UserInput.Video.url(try MiniCPMV47VideoFixtures.videoURL(name))
        let images = try m.images.map { UserInput.Image.url(try MiniCPMV47Fixtures.imageURL($0)) }
        var input = UserInput(chat: [.user(m.prompt, images: images, videos: [video])])
        if let thinking = m.enable_thinking { input.additionalContext = ["enable_thinking": thinking] }
        let prepared = try await processor.prepare(input: input)
        let ids = try #require(prepared.text.tokenIds)
        if ids != m.input_ids {
            let firstDiff = zip(ids, m.input_ids).enumerated().first { $0.element.0 != $0.element.1 }?.offset ?? min(ids.count, m.input_ids.count)
            Issue.record("\(name): tokens differ at \(firstDiff) (swift \(ids.count) vs ref \(m.input_ids.count)); swift[\(firstDiff)...] = \(ids.dropFirst(firstDiff).prefix(8)), ref = \(m.input_ids.dropFirst(firstDiff).prefix(8))")
        }
        let types = ids.map { $0 == MiniCPMV47Fixtures.imageTokenId ? 1 : ($0 == MiniCPMV47Fixtures.videoTokenId ? 2 : 0) }
        #expect(types == m.mm_token_type_ids)
        #expect(prepared.video?.frames?.map { [$0.h, $0.w] } == m.target_sizes_videos)
        #expect(prepared.video?.pixels.shape == [m.pixel_values_videos_shape[3] / 14, 14, 14, 3])
        if let targetSizes = m.target_sizes {
            #expect(prepared.image?.frames?.map { [$0.h, $0.w] } == targetSizes)
            #expect(prepared.mediaTokenIds == [MiniCPMV47Fixtures.imageTokenId, MiniCPMV47Fixtures.videoTokenId])
        } else {
            #expect(prepared.image == nil)
            #expect(prepared.mediaTokenIds == [MiniCPMV47Fixtures.videoTokenId])
        }
        let (positions, delta) = MiniCPMV47Canvas.ropeIndex(
            inputIds: ids, imageGrids: prepared.image?.frames ?? [], videoGrids: prepared.video?.frames ?? [],
            imageTokenId: MiniCPMV47Fixtures.imageTokenId, videoTokenId: MiniCPMV47Fixtures.videoTokenId,
            markers: MiniCPMV47Fixtures.markers, mergeFactor: 4)
        for d in 0 ..< 3 {
            let mismatches = zip(positions[d].map(Int.init), m.position_ids[d]).enumerated().filter { $0.element.0 != $0.element.1 }
            #expect(mismatches.isEmpty, "\(name) axis \(d): \(mismatches.count) mismatches, first \(mismatches.prefix(5).map { ($0.offset, $0.element.0, $0.element.1) })")
        }
        #expect(Int(delta) == m.rope_delta, "\(name) rope delta")
    }

    private func processorConfiguration(_ bundle: URL) throws -> MiniCPMV47ProcessorConfiguration {
        try JSONDecoder().decode(
            MiniCPMV47ProcessorConfiguration.self,
            from: Data(contentsOf: bundle.appendingPathComponent("preprocessor_config.json")))
    }
}

struct MiniCPMV47VideoManifest: Decodable {
    let prompt: String
    let enable_thinking: Bool?
    let images: [String]
    let image_first: Bool
    let video_fps: Double
    let video_total_frames: Int
    let frame_indices: [Int]
    let sampled_frames_shape: [Int]
    let input_ids: [Int]
    let mm_token_type_ids: [Int]
    let target_sizes_videos: [[Int]]
    let grids_videos: [[Int]]
    let target_sizes: [[Int]]?
    let pixel_values_videos_shape: [Int]
    let position_ids: [[Int]]
    let rope_delta: Int
    let video_token_count: Int
}

enum MiniCPMV47VideoFixtures {
    /// Reference manifests `video_<case>.json` (transformers 5.18 `MiniCPMV4_6VideoProcessor`, torchvision
    /// backend, fed the frames AVFoundation decodes from the H.264 clips next to them). A mixed message is
    /// covered image-first only: `Qwen3VLMessageGenerator` renders a message's images before its videos (the
    /// same convention Qwen3-VL uses here), so a video-then-image content order is not expressible.
    static let cases = ["clipA", "clipB", "image_then_clipB"]
    static let clips = ["clipA": "clipA_640x360_30fps_h264", "clipB": "clipB_518x346_24fps_h264"]

    static func manifest(_ name: String) throws -> MiniCPMV47VideoManifest {
        let url = try #require(Bundle.module.url(forResource: "video_\(name)", withExtension: "json", subdirectory: "minicpmv47"))
        return try JSONDecoder().decode(MiniCPMV47VideoManifest.self, from: Data(contentsOf: url))
    }

    static func clipName(_ name: String) -> String { name.contains("clipA") ? "clipA" : "clipB" }

    static func videoURL(_ name: String) throws -> URL {
        try #require(Bundle.module.url(forResource: clips[clipName(name)]!, withExtension: "mp4", subdirectory: "minicpmv47"))
    }

    /// Reference `pixel_values_videos[0]` `[3, 14, N*14]` as `[N, 14, 14, 3]` patches.
    static func referencePatches(_ root: URL, _ name: String, _ m: MiniCPMV47VideoManifest) throws -> MLXArray {
        let values = try MiniCPMV47Fixtures.floats(root.appendingPathComponent("\(name)/pixel_values_videos_f32.bin"))
        let n = m.pixel_values_videos_shape[3] / 14
        return MLXArray(values, [3, 14, n, 14]).transposed(2, 1, 3, 0)
    }
}

@Suite("MiniCPM-V 4.7 video sampling and torchvision resize", .serialized)
struct MiniCPMV47VideoMathTests {

    /// `MiniCPMV4_6VideoProcessor.sample_frames` on the same (total, fps) pairs: one per started second, and the
    /// 0.1 s + linspace thinning past `max_num_frames` seconds (expected lists printed by the reference).
    @Test func sampleFrameIndicesMatchReference() {
        func check(_ total: Int, _ fps: Double, count: Int, first: [Int], last: [Int], sum: Int) {
            let idx = MiniCPMV47VideoProcessing.sampleFrameIndices(totalFrames: total, fps: fps, maxNumFrames: 128)
            #expect(idx.count == count, "\(total)@\(fps) count")
            #expect(Array(idx.prefix(first.count)) == first, "\(total)@\(fps) first")
            #expect(Array(idx.suffix(last.count)) == last, "\(total)@\(fps) last")
            #expect(idx.reduce(0, +) == sum, "\(total)@\(fps) sum")
        }
        check(90, 30.0, count: 3, first: [0, 30, 60], last: [0, 30, 60], sum: 90)
        check(72, 24.0, count: 3, first: [0, 24, 48], last: [0, 24, 48], sum: 72)
        check(100, 29.97, count: 4, first: [0, 29, 59, 89], last: [0, 29, 59, 89], sum: 177)
        check(45, 15.0, count: 3, first: [0, 15, 30], last: [0, 15, 30], sum: 45)
        check(1, 30.0, count: 1, first: [0], last: [0], sum: 0)
        check(6000, 30.0, count: 128, first: [0, 45, 93, 141, 186, 234, 282, 330], last: [5853, 5901, 5949, 5997], sum: 383615)
        check(4000, 25.0, count: 128, first: [0, 30, 62, 92, 125, 155, 187, 220], last: [3902, 3932, 3965, 3997], sum: 255648)
        check(3853, 29.97002997, count: 128, first: [0, 29, 59, 89, 119, 149, 179, 209], last: [3755, 3785, 3815, 3848], sum: 246029)
        check(3907, 30.0, count: 128, first: [0, 30, 60, 90, 120, 153, 183, 213], last: [3810, 3840, 3870, 3903], sum: 249601)
    }

    /// `tvF.resize(uint8, BICUBIC, antialias=True)` on a 346x518 synthetic pattern (`(x*7 + y*13 + c*101) % 256`):
    /// pixel sums and sampled values printed by torchvision 0.29.
    @Test func torchAntialiasedBicubicMatchesTorchvision() {
        let (h, w) = (346, 518)
        var data = [UInt8](repeating: 0, count: h * w * 3)
        for y in 0 ..< h {
            for x in 0 ..< w {
                for c in 0 ..< 3 { data[(y * w + x) * 3 + c] = UInt8((x * 7 + y * 13 + c * 101) % 256) }
            }
        }
        let image = MiniCPMV47ImageProcessing.RGB8(width: w, height: h, data: data)
        let expectations: [(Int, Int, Int, [Int])] = [
            (392, 560, 83961912, [0, 114, 225, 89, 128, 232]),
            (336, 616, 79171441, [0, 114, 228, 106, 116, 253]),
            (448, 448, 76769839, [0, 114, 227, 71, 150, 237]),
            (280, 504, 53976644, [2, 112, 241, 120, 124, 246]),
        ]
        for (oh, ow, sum, samples) in expectations {
            let r = MiniCPMV47VideoProcessing.resizeBicubicTorchAA(image, toWidth: ow, height: oh)
            #expect(r.width == ow && r.height == oh)
            #expect(r.data.reduce(0) { $0 + Int($1) } == sum, "\(oh)x\(ow) sum")
            let points = [(0, 0, 0), (oh - 1, ow - 1, 2), (oh / 2, ow / 3, 1), (7, ow - 5, 0), (oh - 3, 11, 2), (oh / 3, ow / 2, 0)]
            #expect(points.map { Int(r[$0.0, $0.1, $0.2]) } == samples, "\(oh)x\(ow) samples")
        }
    }

    /// Frame placeholders: `<image>pad*thumb</image>` + slice rows per frame, back to back, no `<image_id>`.
    @Test func videoPlaceholderLayout() {
        let markers = MiniCPMV47Fixtures.markers
        let sliced = MiniCPMV47ImageProcessing.ProcessedCrops(
            patches: MLXArray.zeros([1]), grids: [THW(1, 24, 44), THW(1, 32, 32), THW(1, 32, 32)], sliceGrid: (1, 2))
        let plain = MiniCPMV47ImageProcessing.ProcessedCrops(patches: MLXArray.zeros([1]), grids: [THW(1, 28, 40)], sliceGrid: nil)
        let tokens = MiniCPMV47VideoProcessing.placeholderTokens(
            frames: [sliced, plain], padTokenId: 7, tokensPerPatchDivisor: 16, markers: markers)
        var expected: [Int] = [markers.imageStart] + Array(repeating: 7, count: 66) + [markers.imageEnd]
        for _ in 0 ..< 2 { expected += [markers.sliceStart] + Array(repeating: 7, count: 64) + [markers.sliceEnd] }
        expected += [markers.imageStart] + Array(repeating: 7, count: 70) + [markers.imageEnd]
        #expect(tokens == expected)
    }
}

@Suite("MiniCPM-V 4.7 video decode and pixel parity", .serialized,
       .enabled(if: MiniCPMV47Fixtures.binaryRoot != nil))
struct MiniCPMV47VideoPixelTests {

    /// The library's AVFoundation decode of the sampled frames must equal the dump the reference consumed
    /// (`videos/<clip>.mp4.avf_frames_u8_thwc.bin`, all frames), and the sampled patches the reference's
    /// `pixel_values_videos`.
    @Test(arguments: ["clipA", "clipB"])
    func decodedFramesAndPatchesMatchReference(name: String) async throws {
        let root = try #require(MiniCPMV47Fixtures.binaryRoot)
        let m = try MiniCPMV47VideoFixtures.manifest(name)
        let url = try MiniCPMV47VideoFixtures.videoURL(name)
        let sampled = try await MiniCPMV47VideoProcessing.sample(.url(url), maxNumFrames: 128)
        #expect(sampled.totalFrames == m.video_total_frames)
        #expect(sampled.fps == m.video_fps)
        #expect(sampled.indices == m.frame_indices)
        let (t, h, w) = (m.sampled_frames_shape[0], m.sampled_frames_shape[1], m.sampled_frames_shape[2])
        #expect(sampled.frames.count == t)
        let dump = try MiniCPMV47Fixtures.bytes(
            root.appendingPathComponent("videos/\(MiniCPMV47VideoFixtures.clips[MiniCPMV47VideoFixtures.clipName(name)]!).mp4.avf_frames_u8_thwc.bin"))
        #expect(dump.count == m.video_total_frames * h * w * 3)
        var byteMismatches = 0
        for (k, index) in sampled.indices.enumerated() {
            let frame = sampled.frames[k]
            #expect(frame.height == h && frame.width == w)
            let base = index * h * w * 3
            for i in 0 ..< h * w * 3 where frame.data[i] != dump[base + i] { byteMismatches += 1 }
        }
        #expect(byteMismatches == 0, "\(name): \(byteMismatches) decoded bytes differ from the AVFoundation dump")

        let crops = sampled.frames.map {
            MiniCPMV47VideoProcessing.process(
                frame: $0, maxSliceNums: 9, scaleResolution: 448, patchSize: 14, sliceMode: true,
                mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5])
        }
        #expect(crops.flatMap { $0.grids.map { [$0.h, $0.w] } } == m.target_sizes_videos)
        let patches = concatenated(crops.map { $0.patches })
        let expected = try MiniCPMV47VideoFixtures.referencePatches(root, name, m)
        #expect(patches.shape == expected.shape)
        let diff = abs(patches - expected)
        let maxAbs = diff.max().item(Float.self)
        let differing = (diff .> 0).sum().item(Int.self)
        #expect(maxAbs == 0, "\(name): video pixel max-abs \(maxAbs), \(differing) of \(diff.size) values differ")
        print("[minicpmv47 video pixels] \(name): max-abs \(maxAbs) differing \(differing)/\(diff.size)")
    }
}

@Suite("MiniCPM-V 4.7 vision tower float32 parity", .serialized,
       .enabled(if: MiniCPMV47Fixtures.bundleDirectory != nil && MiniCPMV47Fixtures.binaryRoot != nil))
struct MiniCPMV47TowerParityTests {

    /// Build the tower from the bundle config and load its bf16 vision weights as float32.
    static func loadTower(_ bundle: URL) throws -> MiniCPMV47Vision.Tower {
        let configData = try Data(contentsOf: bundle.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Qwen35Configuration.self, from: configData)
        let vision = try #require(config.miniCPMVisionConfiguration)
        let canvas = try #require(config.miniCPMCanvas)
        let tower = MiniCPMV47Vision.Tower(vision, canvas: canvas, llmHidden: config.textConfiguration.hiddenSize)
        let index = try JSONSerialization.jsonObject(
            with: Data(contentsOf: bundle.appendingPathComponent("model.safetensors.index.json"))) as? [String: Any]
        let weightMap = try #require(index?["weight_map"] as? [String: String])
        var weights: [String: MLXArray] = [:]
        for file in Set(weightMap.values) {
            let arrays = try MLX.loadArrays(url: bundle.appendingPathComponent(file))
            for (key, value) in arrays where key.hasPrefix("vision_tower.") || key.hasPrefix("merger.") {
                weights[key] = value.asType(.float32)
            }
        }
        let sanitized = tower.sanitizeVision(weights: weights)
        var towerWeights: [String: MLXArray] = [:]
        for (key, value) in sanitized where key.hasPrefix("vision_tower.") {
            towerWeights[String(key.dropFirst("vision_tower.".count))] = value
        }
        try tower.update(parameters: ModuleParameters.unflattened(towerWeights), verify: .all)
        MLX.eval(tower)
        return tower
    }

    @Test(arguments: ["scene", "odd", "portrait"])
    func featuresMatchTorchFloat32(name: String) throws {
        let bundle = try #require(MiniCPMV47Fixtures.bundleDirectory)
        let root = try #require(MiniCPMV47Fixtures.binaryRoot)
        let m = try MiniCPMV47Fixtures.manifest(name)
        let tower = try Self.loadTower(bundle)
        let pixels = try MiniCPMV47Fixtures.referencePatches(root, name, m)
        let grids = MiniCPMV47Fixtures.crops(m)

        func compare(_ label: String, _ actual: MLXArray, _ file: String) throws -> (Float, Float) {
            let expectedValues = try MiniCPMV47Fixtures.floats(root.appendingPathComponent("\(name)/\(file)"))
            let expected = MLXArray(expectedValues, actual.shape)
            let a = actual.asType(.float32)
            let maxAbs = abs(a - expected).max().item(Float.self)
            let cosine = (sum(a * expected) / (sqrt(sum(a * a)) * sqrt(sum(expected * expected)))).item(Float.self)
            let refMax = abs(expected).max().item(Float.self)
            print("[minicpmv47 tower] \(name) \(label): shape \(actual.shape) max-abs \(maxAbs) (ref max |x| \(refMax)) cosine \(cosine)")
            return (maxAbs, cosine)
        }

        let embeds = tower.embeddings(pixels, grids: grids)
        _ = try compare("embeddings", embeds, "stage_embeddings_f32.bin")
        // Stage walk: layers 0...insert, the window merger, then the remaining layers — to tell gradual fp32
        // accumulation drift from a semantic step.
        var h = embeds
        var segments = grids.map { $0.h * $0.w }
        for (index, layer) in tower.encoder.layers.enumerated() {
            h = layer(h, segmentLengths: segments)
            if index == tower.canvas.insertLayerId {
                _ = try compare("layer\(index) (pre_merger)", h, "stage_pre_merger_f32.bin")
                h = tower.vitMerger(h, grids: grids)
                _ = try compare("vit_merger (post_merger)", h, "stage_post_merger_f32.bin")
                segments = grids.map { ($0.h / 2) * ($0.w / 2) }
            }
        }
        let (hidden, mergedGrids) = tower.encode(pixels, grids: grids)
        _ = try compare("post_layernorm", hidden, "stage_post_layernorm_f32.bin")
        let features = tower.merger(hidden, grids: mergedGrids)
        // Runtime noise floor: the same tower with explicit matmul+softmax attention instead of the fused kernel.
        MiniCPMV47Vision.Attention.useExplicitSoftmax = true
        let (hiddenExplicit, _) = tower.encode(pixels, grids: grids)
        let featuresExplicit = tower.merger(hiddenExplicit, grids: mergedGrids)
        MiniCPMV47Vision.Attention.useExplicitSoftmax = false
        print("[minicpmv47 tower] \(name) fused-vs-explicit attention: post_layernorm max-abs \(abs(hidden - hiddenExplicit).max().item(Float.self)) features max-abs \(abs(features - featuresExplicit).max().item(Float.self))")
        _ = try compare("features (explicit softmax)", featuresExplicit, "features_f32.bin")
        #expect(features.shape == m.features_shape)
        #expect(tower.mergedRowCount(for: grids) == m.features_shape[0])
        let (maxAbs, cosine) = try compare("features", features, "features_f32.bin")
        #expect(cosine > 0.9999, "\(name) features cosine \(cosine)")
        #expect(maxAbs < 5e-2, "\(name) features max-abs \(maxAbs)")
    }
}
