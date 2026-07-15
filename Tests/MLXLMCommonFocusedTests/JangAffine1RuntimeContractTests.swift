// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("JANG affine-1 runtime contract", .serialized)
struct JangAffine1RuntimeContractTests {
    @Test("schema-2 contract selects only affine one-bit modules")
    func contractSelectsOnlyAffineOneBitModules() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("jang-affine1-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let config: [String: Any] = [
            "format": "jang",
            "format_version": "2.0",
            "runtime": ["requires_jang_affine1_expansion": true],
            "quantization": [
                "affine1_runtime_expansion": [
                    "lossless": true,
                    "runtime_bits": 2,
                    "scales_biases_unchanged": true,
                    "storage_bits": 1,
                ],
                "tensor_quantization_manifest_schema": 2,
                "tensor_quantization_manifest_count": 2,
                "tensor_quantization_manifest": [
                    "model.binary": [
                        "bits": 1, "storage_bits": 1, "group_size": 128,
                        "mode": "affine",
                    ],
                    "model.four_bit": [
                        "bits": 4, "storage_bits": 4, "group_size": 64,
                        "mode": "affine",
                    ],
                ],
            ],
        ]
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
            .write(to: root.appendingPathComponent("jang_config.json"))

        let loadedContract = try JangLoader.loadAffine1RuntimeContract(at: root)
        let contract = try #require(loadedContract)
        #expect(contract.storageBits == 1)
        #expect(contract.runtimeBits == 2)
        #expect(contract.modulePaths == ["model.binary"])

        let loadedManifest = try JangLoader.loadTensorQuantizationManifest(at: root)
        let manifest = try #require(loadedManifest)
        #expect(manifest.entries["model.binary"]?.bits == 1)
        #expect(manifest.entries["model.binary"]?.groupSize == 128)
        #expect(manifest.entries["model.four_bit"]?.bits == 4)
        #expect(manifest.entries["model.four_bit"]?.groupSize == 64)
    }

    @Test("required malformed contract fails closed")
    func malformedContractFailsClosed() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("jang-affine1-invalid-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try JSONSerialization.data(withJSONObject: [
            "runtime": ["requires_jang_affine1_expansion": true],
            "quantization": [:],
        ]).write(to: root.appendingPathComponent("jang_config.json"))

        #expect(throws: JangLoaderError.self) {
            try JangLoader.loadAffine1RuntimeContract(at: root)
        }
    }

    @Test("native one-bit qmv matches lossless two-bit representation")
    func nativeOneBitQMVMatchesLosslessTwoBit() {
        compareNativeOneBitToLosslessTwoBit(inputRows: 1, columns: 512)
    }

    @Test("native one-bit fast qmv matches lossless two-bit representation")
    func nativeOneBitFastQMVMatchesLosslessTwoBit() {
        compareNativeOneBitToLosslessTwoBit(inputRows: 1, columns: 1024)
    }

    @Test("native one-bit qmm matches lossless two-bit representation")
    func nativeOneBitQMMMatchesLosslessTwoBit() {
        compareNativeOneBitToLosslessTwoBit(inputRows: 16, columns: 512)
    }

    @Test("schema-2 manifest resolves ambiguous vision packing")
    func manifestResolvesAmbiguousVisionPacking() {
        let base = "vision_tower.blocks.0.attn.qkv"
        let weights: [String: MLXArray] = [
            "\(base).weight": MLXArray.zeros([3456, 144], dtype: .uint32),
            "\(base).scales": MLXArray.zeros([3456, 18], dtype: .float32),
            "\(base).biases": MLXArray.zeros([3456, 18], dtype: .float32),
        ]
        let exact = BaseConfiguration.Quantization(
            groupSize: 64, bits: 4, mode: .affine)
        let inferred = JangLoader.inferPerLayerQuantization(
            weights: weights,
            jangConfig: JangConfig(
                quantization: JangQuantization(
                    blockSize: 128, bitWidthsUsed: [1, 4])),
            declaredManifestQuantization: [base: exact])

        guard case .quantize(let actual)? = inferred.perLayerQuantization[base] else {
            Issue.record("Expected an exact vision quantization override")
            return
        }
        #expect(actual == exact)
    }

    private func compareNativeOneBitToLosslessTwoBit(inputRows: Int, columns: Int) {
        Device.withDefaultDevice(.gpu) {
            let rows = 8
            let packedOneBit = (0 ..< (rows * columns / 32)).map {
                UInt32(truncatingIfNeeded: 0x9E37_79B9 &* UInt32($0 + 1))
            }
            let packedTwoBit = packedOneBit.flatMap { value in
                [spread16(value & 0xFFFF), spread16(value >> 16)]
            }
            let input = MLXArray(
                (0 ..< (inputRows * columns)).map { Float(($0 % 17) - 8) * 0.0625 },
                [inputRows, columns])
            let groupsPerRow = columns / 128
            let scales = MLXArray(
                (0 ..< (rows * groupsPerRow)).map { Float($0 + 1) * 0.03125 },
                [rows, groupsPerRow])
            let biases = MLXArray(
                (0 ..< (rows * groupsPerRow)).map { Float($0 - 4) * 0.015625 },
                [rows, groupsPerRow])

            let oneBit = quantizedMM(
                input,
                MLXArray(packedOneBit, [rows, columns / 32]),
                scales: scales,
                biases: biases,
                groupSize: 128,
                bits: 1)
            let twoBit = quantizedMM(
                input,
                MLXArray(packedTwoBit, [rows, columns / 16]),
                scales: scales,
                biases: biases,
                groupSize: 128,
                bits: 2)
            eval(oneBit, twoBit)

            let oneBitValues = oneBit.asArray(Float.self)
            let twoBitValues = twoBit.asArray(Float.self)
            #expect(oneBitValues.count == twoBitValues.count)
            for (actual, expected) in zip(oneBitValues, twoBitValues) {
                #expect(abs(actual - expected) < 1e-5)
            }
        }
    }

    private func spread16(_ input: UInt32) -> UInt32 {
        var result: UInt32 = 0
        for bit in 0 ..< 16 {
            result |= ((input >> UInt32(bit)) & 1) << UInt32(bit * 2)
        }
        return result
    }
}
