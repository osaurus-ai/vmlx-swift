// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import MLXLMCommon
@testable import MLXVLM
import Testing

@Suite("MiMo V2.6 vision reference parity", .serialized)
struct MiMoV26VisionTests {
    static let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("Fixtures/MiMoV26")

    func arrays() throws -> [String: MLXArray] {
        try MLX.loadArrays(url: Self.fixtures.appendingPathComponent("vision-reference.safetensors"))
    }

    /// Both goldens come from the independent reference tower with identical
    /// weights and inputs. Default GPU math can use TF32 on supported devices;
    /// an explicitly strict process must match CPU F32, never the TF32 golden.
    /// The complete output must match one policy at the original tolerance.
    static func matchesReference(_ actual: MLXArray, tensors: [String: MLXArray],
        selecting: (MLXArray) -> MLXArray = { $0 }) throws -> Bool {
        let keys = ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0"
            ? ["expected_f32"] : ["expected_f32", "expected"]
        return try keys.contains { key in
            allClose(actual, selecting(try #require(tensors[key])), rtol: 2e-4, atol: 2e-5).item(Bool.self)
        }
    }

    @Test("Flattened patches with different geometry cannot reuse media cache state")
    func cacheGeometry() {
        let pixels = MLXArray((0..<96).map { Float($0) }).reshaped(8, 12)
        func salt(_ frames: [THW]?, video: Bool = false) -> String? {
            let text = LMInput.Text(tokens: MLXArray([Int32(1)]))
            return computeMediaSalt(for: video
                ? LMInput(text: text, video: .init(pixels: pixels, frames: frames))
                : LMInput(text: text, image: .init(pixels: pixels, frames: frames)))
        }
        for video in [false, true] {
            let grid = [THW(1, 2, 4)]
            #expect(salt(grid, video: video) == salt(grid, video: video))
            #expect(salt(grid, video: video) != salt([THW(1, 4, 2)], video: video))
            #expect(salt(grid, video: video) != salt([THW(2, 2, 2)], video: video))
            #expect(salt(grid, video: video) != salt([THW(1, 2, 2), THW(1, 2, 2)], video: video))
            #expect(salt(grid, video: video) != salt(nil, video: video))
        }
        #expect(salt([THW(1, 2, 4)]) != salt([THW(1, 2, 4)], video: true))
    }

    @Test("Mixed full/row/column attention with nonzero sinks matches the reference tower")
    func towerReference() throws {
        let tensors = try arrays()
        let config = try JSONDecoder().decode(MiMoV26VisionConfiguration.self,
            from: Data(contentsOf: Self.fixtures.appendingPathComponent("vision-config.json")))
        let tower = MiMoV26VisionTower(config)
        let weights = Dictionary(uniqueKeysWithValues: tensors.compactMap { key, value in
            key.hasPrefix("weight.") ? (String(key.dropFirst(7)), value) : nil
        })
        try tower.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let grid = [THW(1, 4, 6), THW(2, 2, 4)]
        let result = try tower(#require(tensors["pixels"]), grid: grid)
        eval(result)
        #expect(result.shape == [10, 16])
        #expect(try Self.matchesReference(result, tensors: tensors))
        // A separate item/temporal frame must never attend to earlier frames.
        let isolated = try tower(#require(tensors["pixels"])[24...], grid: [THW(2, 2, 4)])
        #expect(allClose(result[6...], isolated, rtol: 2e-4, atol: 2e-5).item(Bool.self))
    }

    @Test("Merge-unit permutations and 2D rotary positions match the reference")
    func layoutReference() throws {
        let tensors = try arrays()
        let layout = try MiMoV26VisionLayout(grid: [THW(1, 4, 6), THW(2, 2, 4)], headDim: 4, merge: 2)
        #expect(layout.sequenceLengths == [24, 8, 8])
        #expect(arrayEqual(layout.columnIndices, try #require(tensors["columns"])).item(Bool.self))
        #expect(allClose(layout.cosine, try #require(tensors["cosine"]), atol: 1e-7).item(Bool.self))
        #expect(allClose(layout.sine, try #require(tensors["sine"]), atol: 1e-7).item(Bool.self))
    }

    @Test("Non-antialiased bilinear pixels and C,T,P,P packing match reference")
    func pixelsReference() throws {
        let tensors = try arrays()
        let raw = try #require(tensors["raw"])
        let resized = try MiMoV26Pixels.resize(raw, height: 8, width: 12)
        #expect(allClose(resized, try #require(tensors["resized"]), rtol: 1e-6, atol: 1e-5).item(Bool.self))
        let (patches, grid) = try MiMoV26Pixels.patches(raw, height: 8, width: 12, patch: 2, merge: 2, temporal: 2)
        #expect(grid.t == 1 && grid.h == 4 && grid.w == 6)
        #expect(allClose(patches, try #require(tensors["patches"]), rtol: 1e-6, atol: 1e-6).item(Bool.self))
    }

    @Test("Banker rounding and video sampling preserve native geometry")
    func geometry() throws {
        let (h, w) = try MiMoV26Pixels.targetSize(height: 80, width: 144, factor: 32, minimum: 1024, maximum: 1_000_000)
        #expect(h == 64 && w == 128)
        #expect(try MiMoV26Pixels.videoSamples(total: 300, sourceFPS: 30, targetFPS: 1, minimum: 8, maximum: 3600)
            == [0, 33, 66, 99, 132, 166, 199, 232, 265, 299])
        #expect(try MiMoV26Pixels.timestamp(125.9) == "02:05")
    }
}
