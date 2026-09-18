// Copyright © 2026 osaurus-eval contributors
// SPDX-License-Identifier: MIT
//
// A quantized position table has to interpolate like a float one.
//
// The Qwen3-VL vision tower adds a learned position embedding to every patch, bilinearly
// interpolated from a `num_grid_per_side` x `num_grid_per_side` table onto the image's patch grid.
// It built the four corner weights, and the accumulator they are summed into, in
// `posEmbed.weight.dtype`. For a float table that is the table's dtype. A bundle that stores
// `vision_tower.pos_embed` quantized loads it as a `QuantizedEmbedding`, whose `weight` is the
// PACKED uint32 array, so every corner weight below 1 truncated to 0. Only a patch that lands
// exactly on a table row kept its position. Nothing failed: the features were finite and had the
// right shape.
//
// The oracle is the same table dequantized to float. A `QuantizedEmbedding` returns exactly those
// rows, so a correct interpolation cannot tell the two apart.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

@Suite("Qwen3-VL quantized position table")
struct Qwen3VLQuantizedPosEmbedTests {

    /// An 8 x 8 position table, 64 wide, so each row holds two quantization groups of 32.
    static let configJSON = """
        {
            "model_type": "qwen3_vl",
            "depth": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "out_hidden_size": 32,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 64,
            "in_channels": 3
        }
        """

    /// 6 x 10 patches on the 8 x 8 table. Both axes land between table rows everywhere except at
    /// their ends, so only the four corner patches sit exactly on a row.
    static let grid = THW(1, 6, 10)
    static let patches = 60

    static let groupSize = 32
    static let bits = 4

    static func tower() throws -> Qwen3VLVision.VisionModel {
        let config = try JSONDecoder().decode(
            Qwen3VLConfiguration.VisionConfiguration.self, from: Data(configJSON.utf8))
        return Qwen3VLVision.VisionModel(config)
    }

    /// `table` quantized the way a checkpoint stores it, built with the loader's constructor for
    /// pre-quantized tensors, and the float table those codes decode to.
    static func quantize(_ table: MLXArray) -> (quantized: QuantizedEmbedding, decoded: Embedding) {
        let (wq, scales, biases) = MLX.quantized(table, groupSize: groupSize, bits: bits)
        let quantized = QuantizedEmbedding(
            weight: wq, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
        let decoded = Embedding(
            weight: dequantized(
                wq, scales: scales, biases: biases, groupSize: groupSize, bits: bits))
        return (quantized, decoded)
    }

    /// Swaps the tower's position table, as the loader does when it finds `pos_embed.scales`.
    static func install(_ table: Embedding, in tower: Qwen3VLVision.VisionModel) {
        tower.update(modules: ModuleChildren.unflattened([("pos_embed", table)]))
    }

    /// The largest absolute difference, in float32 whatever the operands' dtype.
    static func largestDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// THE DEFECT. Every patch must get a position, and the one the float table gives it.
    @Test("a quantized table interpolates exactly like the float table it decodes to")
    func quantizedTableMatchesItsDecoding() throws {
        try MLXMetalTestLock.withLock {
            let tower = try Self.tower()
            let table = Self.quantize(MLXRandom.normal([64, 64], key: MLXRandom.key(164)))

            Self.install(table.decoded, in: tower)
            let expected = tower.positionalEmbeddings([Self.grid])
            Self.install(table.quantized, in: tower)
            let actual = tower.positionalEmbeddings([Self.grid])

            #expect(actual.shape == [Self.patches, 64])
            let unplaced = (abs(actual).max(axis: -1) .== 0).asType(.int32).sum().item(Int.self)
            #expect(unplaced == 0, "\(unplaced) of \(Self.patches) patches have no position")
            let worst = Self.largestDifference(actual, expected)
            #expect(worst <= 1e-6, "off by up to \(worst) from the decoded float table")
        }
    }

    /// Against the ORIGINAL float table. Interpolation is a convex combination of four rows, so no
    /// element can move further than the table's own worst quantization error.
    @Test("a quantized table stays within its quantization error of the float original")
    func quantizedTableStaysWithinQuantizationError() throws {
        try MLXMetalTestLock.withLock {
            let tower = try Self.tower()
            let original = MLXRandom.normal([64, 64], key: MLXRandom.key(165))
            let table = Self.quantize(original)
            let quantizationError = Self.largestDifference(table.decoded.weight, original)

            Self.install(Embedding(weight: original), in: tower)
            let expected = tower.positionalEmbeddings([Self.grid])
            Self.install(table.quantized, in: tower)
            let actual = tower.positionalEmbeddings([Self.grid])

            let worst = Self.largestDifference(actual, expected)
            #expect(
                worst <= quantizationError * 1.001 + 1e-6,
                "off by up to \(worst); the table's quantization error is \(quantizationError)")
        }
    }

    /// The same comparison through the forward pass, which is what an image actually runs.
    @Test("the tower's features do not depend on how the position table is stored")
    func towerFeaturesMatch() throws {
        try MLXMetalTestLock.withLock {
            let tower = try Self.tower()
            let table = Self.quantize(MLXRandom.normal([64, 64], key: MLXRandom.key(166)))
            // One row per patch: channels x temporal patch x patch x patch.
            let pixels = MLXRandom.normal([Self.patches, 3 * 2 * 2 * 2], key: MLXRandom.key(167))

            Self.install(table.decoded, in: tower)
            let (expected, _) = tower(pixels, gridTHW: [Self.grid])
            Self.install(table.quantized, in: tower)
            let (actual, _) = tower(pixels, gridTHW: [Self.grid])

            // 60 patches merged 2 x 2 into 15 tokens of `out_hidden_size`.
            #expect(actual.shape == [15, 32])
            let worst = Self.largestDifference(actual, expected)
            #expect(worst <= 1e-5, "features differ by up to \(worst)")
        }
    }

    /// A converter may keep a quantized table's scales in float16 inside a bfloat16 tower. Its
    /// rows then dequantize to float16, and float16 + bfloat16 promotes to float32, which would
    /// widen every block after the position embedding.
    @Test("float16 scales do not widen a bfloat16 tower to float32")
    func bfloat16TowerStaysBFloat16() throws {
        try MLXMetalTestLock.withLock {
            let tower = try Self.tower()
            tower.apply { $0.asType(.bfloat16) }
            let table = Self.quantize(
                MLXRandom.normal([64, 64], key: MLXRandom.key(168)).asType(.float16))
            #expect(table.quantized.scales.dtype == .float16)
            Self.install(table.quantized, in: tower)

            let pixels = MLXRandom.normal([Self.patches, 3 * 2 * 2 * 2], key: MLXRandom.key(169))
                .asType(.bfloat16)
            let (features, _) = tower(pixels, gridTHW: [Self.grid])

            #expect(features.dtype == .bfloat16)
            let largest = abs(features.asType(.float32)).max().item(Float.self)
            #expect(largest.isFinite)
        }
    }
}
