// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
@testable import MLXLMCommon
import XCTest

final class JANGTQHadamardShuffleTests: XCTestCase {
    override func setUpWithError() throws {
        prepareMLXMetallibForFocusedTests()
    }

    func testMiniMaxSizedShufflePathMatchesCPUReference() {
        MLX.Device.withDefaultDevice(.gpu) {
            let dim = 1536
            let xFloats = (0..<dim).map { Float(($0 % 37) - 18) / 19.0 }
            let signsFloats = (0..<dim).map { i -> Float in
                i.isMultiple(of: 3) ? -1.0 : 1.0
            }
            let x = MLXArray(xFloats).reshaped([1, dim])
            let signs = MLXArray(signsFloats)

            let rotated = JANGTQKernels.hadamardRotate(x, signs: signs, dim: dim)
            MLX.eval(rotated)

            let actual = rotated.asArray(Float.self)
            let expected = hadamardReference(xFloats, signs: signsFloats, blocks: [1024, 512])
            XCTAssertEqual(actual.count, expected.count)

            let maxDiff = zip(actual, expected)
                .map { abs($0 - $1) }
                .max() ?? 0
            XCTAssertLessThan(
                maxDiff, 1e-4,
                "MiniMax 1536-dim SIMD-shuffle Hadamard path must match CPU reference; maxDiff=\(maxDiff)")
        }
    }

    func testHadamardRotatePreservesRankThreeAndRankFourShapes() {
        MLX.Device.withDefaultDevice(.gpu) {
            let rank3Dim = 128
            let rank3 = MLXArray.zeros([2, 4, rank3Dim], dtype: .float32)
            let rank3Signs = MLXArray(Array(repeating: Float(1), count: rank3Dim))
            let rank3Rotated = JANGTQKernels.hadamardRotate(
                rank3, signs: rank3Signs, dim: rank3Dim)
            MLX.eval(rank3Rotated)
            XCTAssertEqual(rank3Rotated.shape, [2, 4, rank3Dim])
            XCTAssertEqual(rank3Rotated.dtype, .float32)

            let rank4Dim = 64
            let rank4 = MLXArray.zeros([1, 2, 3, rank4Dim], dtype: .float32)
            let rank4Signs = MLXArray(Array(repeating: Float(1), count: rank4Dim))
            let rank4Rotated = JANGTQKernels.hadamardRotate(
                rank4, signs: rank4Signs, dim: rank4Dim)
            MLX.eval(rank4Rotated)
            XCTAssertEqual(rank4Rotated.shape, [1, 2, 3, rank4Dim])
            XCTAssertEqual(rank4Rotated.dtype, .float32)
        }
    }

    func testDenseJANGTQMatmulAcceptsRankTwoAndRankThreeInputs() {
        MLX.Device.withDefaultDevice(.gpu) {
            let linear = JANGTQDenseLinear(
                inFeatures: 8, outFeatures: 5, bits: 2, seed: 42)

            let rank2 = MLXArray((0..<24).map { Float($0) / 10 })
                .reshaped([3, 8])
            let rank2Out = linear(rank2)
            MLX.eval(rank2Out)
            XCTAssertEqual(rank2Out.shape, [3, 5])
            XCTAssertEqual(rank2Out.dtype, .float32)

            let rank3 = MLXArray((0..<48).map { Float($0) / 10 })
                .reshaped([2, 3, 8])
                .asType(.float16)
            let rank3Out = linear(rank3)
            MLX.eval(rank3Out)
            XCTAssertEqual(rank3Out.shape, [2, 3, 5])
            XCTAssertEqual(rank3Out.dtype, .float16)
        }
    }

    func testTurboQuantKVHadamardPreservesRankFourShape() {
        MLX.Device.withDefaultDevice(.gpu) {
            let dim = 64
            let kv = MLXArray((0..<384).map { Float($0 % 23) / 23 })
                .reshaped([1, 2, 3, dim])
            let signs = TQHadamard.generateRandomSigns(dim: dim, seed: 7)

            let rotated = TQHadamard.hadamardRotate(kv, signs: signs)
            let restored = TQHadamard.hadamardInverse(rotated, signs: signs)
            MLX.eval(rotated, restored)

            XCTAssertEqual(rotated.shape, [1, 2, 3, dim])
            XCTAssertEqual(restored.shape, kv.shape)
            let maxDiff = (restored - kv).abs().max().item(Float.self)
            XCTAssertLessThan(maxDiff, 1e-4)
        }
    }

    func testOffsetGatherMatchesStackedGatherWithExpertMajorGaps() {
        MLX.Device.withDefaultDevice(.gpu) {
            let bits = 2
            let inFeatures = 8
            let outFeatures = 3
            let batchTokens = 2
            let k = 2
            let packedRows: [[UInt32]] = [
                [0, 1, 2, 3, 0, 1, 2, 3],
                [3, 2, 1, 0, 3, 2, 1, 0],
                [1, 1, 2, 2, 3, 3, 0, 0],
                [2, 0, 2, 0, 2, 0, 2, 0],
                [0, 3, 0, 3, 1, 2, 1, 2],
                [3, 3, 2, 2, 1, 1, 0, 0],
            ].map { [packJANGTQCodes($0, bits: bits)] }
            let packedStack = MLXArray(packedRows.flatMap { $0 })
                .reshaped([2, outFeatures, 1])
            let normsStack = MLXArray([
                Float(0.5), Float(1.0), Float(1.5),
                Float(1.25), Float(0.75), Float(0.25),
            ]).reshaped([2, outFeatures])

            let packedOffsets: [UInt32] = [2, 9]
            let normOffsets: [UInt32] = [1, 7]
            let xRot = MLXArray([
                Float(1), Float(2), Float(-1), Float(0.5),
                Float(3), Float(-2), Float(0.25), Float(1.5),
                Float(-1.5), Float(0.75), Float(2.5), Float(-0.5),
                Float(1.25), Float(0.0), Float(-2.25), Float(3.5),
            ]).reshaped([batchTokens, inFeatures])
            let codebook = MLXArray([Float(-1.0), Float(-0.25), Float(0.5), Float(1.75)])
            let rhsIndices = MLXArray([UInt32(1), UInt32(0), UInt32(0), UInt32(1)])

            let expected = JANGTQKernels.gatherTQTopK(
                xRot: xRot,
                packed: packedStack,
                norms: normsStack,
                codebook: codebook,
                rhsIndices: rhsIndices,
                batchTokens: batchTokens,
                K: k,
                inFeatures: inFeatures,
                outFeatures: outFeatures,
                bits: bits)
            let actual = JANGTQKernels.gatherTQTopKOffsets(
                xRot: xRot,
                packed: MLXArray(makeOffsetSpan(
                    rows: packedRows, offsets: packedOffsets, outFeatures: outFeatures)),
                packedOffsets: MLXArray(packedOffsets),
                norms: MLXArray(makeOffsetSpan(
                    rows: normsStack.asArray(Float.self),
                    offsets: normOffsets,
                    outFeatures: outFeatures)),
                normOffsets: MLXArray(normOffsets),
                codebook: codebook,
                rhsIndices: rhsIndices,
                batchTokens: batchTokens,
                K: k,
                inFeatures: inFeatures,
                outFeatures: outFeatures,
                bits: bits)
            assertClose(actual, expected, tolerance: 1e-5)
        }
    }

    func testOffsetGatherSplitShardSentinelsCanBeSummed() {
        MLX.Device.withDefaultDevice(.gpu) {
            let bits = 2
            let inFeatures = 8
            let outFeatures = 3
            let batchTokens = 2
            let k = 2
            let packedRows: [[UInt32]] = [
                [0, 1, 2, 3, 0, 1, 2, 3],
                [3, 2, 1, 0, 3, 2, 1, 0],
                [1, 1, 2, 2, 3, 3, 0, 0],
                [2, 0, 2, 0, 2, 0, 2, 0],
                [0, 3, 0, 3, 1, 2, 1, 2],
                [3, 3, 2, 2, 1, 1, 0, 0],
            ].map { [packJANGTQCodes($0, bits: bits)] }
            let packedStack = MLXArray(packedRows.flatMap { $0 })
                .reshaped([2, outFeatures, 1])
            let normsStack = MLXArray([
                Float(0.5), Float(1.0), Float(1.5),
                Float(1.25), Float(0.75), Float(0.25),
            ]).reshaped([2, outFeatures])
            let xRot = MLXArray([
                Float(1), Float(2), Float(-1), Float(0.5),
                Float(3), Float(-2), Float(0.25), Float(1.5),
                Float(-1.5), Float(0.75), Float(2.5), Float(-0.5),
                Float(1.25), Float(0.0), Float(-2.25), Float(3.5),
            ]).reshaped([batchTokens, inFeatures])
            let codebook = MLXArray([Float(-1.0), Float(-0.25), Float(0.5), Float(1.75)])
            let rhsIndices = MLXArray([UInt32(1), UInt32(0), UInt32(0), UInt32(1)])

            let expected = JANGTQKernels.gatherTQTopK(
                xRot: xRot,
                packed: packedStack,
                norms: normsStack,
                codebook: codebook,
                rhsIndices: rhsIndices,
                batchTokens: batchTokens,
                K: k,
                inFeatures: inFeatures,
                outFeatures: outFeatures,
                bits: bits)
            let shard0 = JANGTQKernels.gatherTQTopKOffsets(
                xRot: xRot,
                packed: MLXArray(makeOffsetSpan(
                    rows: packedRows, offsets: [2, UInt32.max], outFeatures: outFeatures)),
                packedOffsets: MLXArray([UInt32(2), UInt32.max]),
                norms: MLXArray(makeOffsetSpan(
                    rows: normsStack.asArray(Float.self),
                    offsets: [1, UInt32.max],
                    outFeatures: outFeatures)),
                normOffsets: MLXArray([UInt32(1), UInt32.max]),
                codebook: codebook,
                rhsIndices: rhsIndices,
                batchTokens: batchTokens,
                K: k,
                inFeatures: inFeatures,
                outFeatures: outFeatures,
                bits: bits)
            let shard1 = JANGTQKernels.gatherTQTopKOffsets(
                xRot: xRot,
                packed: MLXArray(makeOffsetSpan(
                    rows: packedRows, offsets: [UInt32.max, 4], outFeatures: outFeatures)),
                packedOffsets: MLXArray([UInt32.max, UInt32(4)]),
                norms: MLXArray(makeOffsetSpan(
                    rows: normsStack.asArray(Float.self),
                    offsets: [UInt32.max, 3],
                    outFeatures: outFeatures)),
                normOffsets: MLXArray([UInt32.max, UInt32(3)]),
                codebook: codebook,
                rhsIndices: rhsIndices,
                batchTokens: batchTokens,
                K: k,
                inFeatures: inFeatures,
                outFeatures: outFeatures,
                bits: bits)
            assertClose(shard0 + shard1, expected, tolerance: 1e-5)
        }
    }

    func testOffsetGatherScoredMatchesStackedGatherScoreSum() {
        MLX.Device.withDefaultDevice(.gpu) {
            let bits = 2
            let inFeatures = 8
            let outFeatures = 3
            let batchTokens = 2
            let k = 2
            let packedRows: [[UInt32]] = [
                [0, 1, 2, 3, 0, 1, 2, 3],
                [3, 2, 1, 0, 3, 2, 1, 0],
                [1, 1, 2, 2, 3, 3, 0, 0],
                [2, 0, 2, 0, 2, 0, 2, 0],
                [0, 3, 0, 3, 1, 2, 1, 2],
                [3, 3, 2, 2, 1, 1, 0, 0],
            ].map { [packJANGTQCodes($0, bits: bits)] }
            let packedStack = MLXArray(packedRows.flatMap { $0 })
                .reshaped([2, outFeatures, 1])
            let normsStack = MLXArray([
                Float(0.5), Float(1.0), Float(1.5),
                Float(1.25), Float(0.75), Float(0.25),
            ]).reshaped([2, outFeatures])
            let packedOffsets: [UInt32] = [2, 9]
            let normOffsets: [UInt32] = [1, 7]
            let xRot = MLXArray([
                Float(1), Float(2), Float(-1), Float(0.5),
                Float(3), Float(-2), Float(0.25), Float(1.5),
                Float(-1.5), Float(0.75), Float(2.5), Float(-0.5),
                Float(1.25), Float(0.0), Float(-2.25), Float(3.5),
                Float(0.5), Float(-1.25), Float(1.75), Float(2.25),
                Float(-3), Float(0.5), Float(1.0), Float(-0.75),
                Float(2.0), Float(1.25), Float(-0.25), Float(-1.0),
                Float(0.0), Float(1.5), Float(-2.0), Float(0.25),
            ]).reshaped([batchTokens * k, inFeatures])
            let codebook = MLXArray([Float(-1.0), Float(-0.25), Float(0.5), Float(1.75)])
            let rhsIndices = MLXArray([UInt32(1), UInt32(0), UInt32(0), UInt32(1)])
            let scores = MLXArray([Float(0.75), Float(0.25), Float(0.4), Float(0.6)])
                .reshaped([batchTokens, k])

            let gathered = JANGTQKernels.gatherTQ(
                xRot: xRot,
                packed: packedStack,
                norms: normsStack,
                codebook: codebook,
                rhsIndices: rhsIndices,
                nRows: batchTokens * k,
                inFeatures: inFeatures,
                outFeatures: outFeatures,
                bits: bits)
                .reshaped([batchTokens, k, outFeatures])
            let expected = (gathered * scores[.ellipsis, .newAxis]).sum(axis: -2)
            let actual = JANGTQKernels.gatherTQTopKOffsetsScored(
                xRot: xRot,
                packed: MLXArray(makeOffsetSpan(
                    rows: packedRows, offsets: packedOffsets, outFeatures: outFeatures)),
                packedOffsets: MLXArray(packedOffsets),
                norms: MLXArray(makeOffsetSpan(
                    rows: normsStack.asArray(Float.self),
                    offsets: normOffsets,
                    outFeatures: outFeatures)),
                normOffsets: MLXArray(normOffsets),
                codebook: codebook,
                rhsIndices: rhsIndices,
                scores: scores.reshaped([-1]),
                batchTokens: batchTokens,
                K: k,
                inFeatures: inFeatures,
                outFeatures: outFeatures,
                bits: bits)
            assertClose(actual, expected, tolerance: 1e-5)
        }
    }

    func testOffsetGatherScoredSplitShardSentinelsCanBeSummed() {
        MLX.Device.withDefaultDevice(.gpu) {
            let bits = 2
            let inFeatures = 8
            let outFeatures = 3
            let batchTokens = 2
            let k = 2
            let packedRows: [[UInt32]] = [
                [0, 1, 2, 3, 0, 1, 2, 3],
                [3, 2, 1, 0, 3, 2, 1, 0],
                [1, 1, 2, 2, 3, 3, 0, 0],
                [2, 0, 2, 0, 2, 0, 2, 0],
                [0, 3, 0, 3, 1, 2, 1, 2],
                [3, 3, 2, 2, 1, 1, 0, 0],
            ].map { [packJANGTQCodes($0, bits: bits)] }
            let packedStack = MLXArray(packedRows.flatMap { $0 })
                .reshaped([2, outFeatures, 1])
            let normsStack = MLXArray([
                Float(0.5), Float(1.0), Float(1.5),
                Float(1.25), Float(0.75), Float(0.25),
            ]).reshaped([2, outFeatures])
            let xRot = MLXArray([
                Float(1), Float(2), Float(-1), Float(0.5),
                Float(3), Float(-2), Float(0.25), Float(1.5),
                Float(-1.5), Float(0.75), Float(2.5), Float(-0.5),
                Float(1.25), Float(0.0), Float(-2.25), Float(3.5),
                Float(0.5), Float(-1.25), Float(1.75), Float(2.25),
                Float(-3), Float(0.5), Float(1.0), Float(-0.75),
                Float(2.0), Float(1.25), Float(-0.25), Float(-1.0),
                Float(0.0), Float(1.5), Float(-2.0), Float(0.25),
            ]).reshaped([batchTokens * k, inFeatures])
            let codebook = MLXArray([Float(-1.0), Float(-0.25), Float(0.5), Float(1.75)])
            let rhsIndices = MLXArray([UInt32(1), UInt32(0), UInt32(0), UInt32(1)])
            let scores = MLXArray([Float(0.75), Float(0.25), Float(0.4), Float(0.6)])
                .reshaped([batchTokens, k])

            let gathered = JANGTQKernels.gatherTQ(
                xRot: xRot,
                packed: packedStack,
                norms: normsStack,
                codebook: codebook,
                rhsIndices: rhsIndices,
                nRows: batchTokens * k,
                inFeatures: inFeatures,
                outFeatures: outFeatures,
                bits: bits)
                .reshaped([batchTokens, k, outFeatures])
            let expected = (gathered * scores[.ellipsis, .newAxis]).sum(axis: -2)
            let shard0 = JANGTQKernels.gatherTQTopKOffsetsScored(
                xRot: xRot,
                packed: MLXArray(makeOffsetSpan(
                    rows: packedRows, offsets: [2, UInt32.max], outFeatures: outFeatures)),
                packedOffsets: MLXArray([UInt32(2), UInt32.max]),
                norms: MLXArray(makeOffsetSpan(
                    rows: normsStack.asArray(Float.self),
                    offsets: [1, UInt32.max],
                    outFeatures: outFeatures)),
                normOffsets: MLXArray([UInt32(1), UInt32.max]),
                codebook: codebook,
                rhsIndices: rhsIndices,
                scores: scores.reshaped([-1]),
                batchTokens: batchTokens,
                K: k,
                inFeatures: inFeatures,
                outFeatures: outFeatures,
                bits: bits)
            let shard1 = JANGTQKernels.gatherTQTopKOffsetsScored(
                xRot: xRot,
                packed: MLXArray(makeOffsetSpan(
                    rows: packedRows, offsets: [UInt32.max, 4], outFeatures: outFeatures)),
                packedOffsets: MLXArray([UInt32.max, UInt32(4)]),
                norms: MLXArray(makeOffsetSpan(
                    rows: normsStack.asArray(Float.self),
                    offsets: [UInt32.max, 3],
                    outFeatures: outFeatures)),
                normOffsets: MLXArray([UInt32.max, UInt32(3)]),
                codebook: codebook,
                rhsIndices: rhsIndices,
                scores: scores.reshaped([-1]),
                batchTokens: batchTokens,
                K: k,
                inFeatures: inFeatures,
                outFeatures: outFeatures,
                bits: bits)
            assertClose(shard0 + shard1, expected, tolerance: 1e-5)
        }
    }

    func testStackedGatherScoredMatchesStackedGatherScoreSum() {
        MLX.Device.withDefaultDevice(.gpu) {
            let bits = 2
            let inFeatures = 8
            let outFeatures = 3
            let batchTokens = 2
            let k = 2
            let packedRows: [[UInt32]] = [
                [0, 1, 2, 3, 0, 1, 2, 3],
                [3, 2, 1, 0, 3, 2, 1, 0],
                [1, 1, 2, 2, 3, 3, 0, 0],
                [2, 0, 2, 0, 2, 0, 2, 0],
                [0, 3, 0, 3, 1, 2, 1, 2],
                [3, 3, 2, 2, 1, 1, 0, 0],
            ].map { [packJANGTQCodes($0, bits: bits)] }
            let packedStack = MLXArray(packedRows.flatMap { $0 })
                .reshaped([2, outFeatures, 1])
            let normsStack = MLXArray([
                Float(0.5), Float(1.0), Float(1.5),
                Float(1.25), Float(0.75), Float(0.25),
            ]).reshaped([2, outFeatures])
            let xRot = MLXArray([
                Float(1), Float(2), Float(-1), Float(0.5),
                Float(3), Float(-2), Float(0.25), Float(1.5),
                Float(-1.5), Float(0.75), Float(2.5), Float(-0.5),
                Float(1.25), Float(0.0), Float(-2.25), Float(3.5),
                Float(0.5), Float(-1.25), Float(1.75), Float(2.25),
                Float(-3), Float(0.5), Float(1.0), Float(-0.75),
                Float(2.0), Float(1.25), Float(-0.25), Float(-1.0),
                Float(0.0), Float(1.5), Float(-2.0), Float(0.25),
            ]).reshaped([batchTokens * k, inFeatures])
            let codebook = MLXArray([Float(-1.0), Float(-0.25), Float(0.5), Float(1.75)])
            let rhsIndices = MLXArray([UInt32(1), UInt32(0), UInt32(0), UInt32(1)])
            let scores = MLXArray([Float(0.75), Float(0.25), Float(0.4), Float(0.6)])
                .reshaped([batchTokens, k])

            let gathered = JANGTQKernels.gatherTQ(
                xRot: xRot,
                packed: packedStack,
                norms: normsStack,
                codebook: codebook,
                rhsIndices: rhsIndices,
                nRows: batchTokens * k,
                inFeatures: inFeatures,
                outFeatures: outFeatures,
                bits: bits)
                .reshaped([batchTokens, k, outFeatures])
            let expected = (gathered * scores[.ellipsis, .newAxis]).sum(axis: -2)
            let actual = JANGTQKernels.gatherTQTopKScored(
                xRot: xRot,
                packed: packedStack,
                norms: normsStack,
                codebook: codebook,
                rhsIndices: rhsIndices,
                scores: scores.reshaped([-1]),
                batchTokens: batchTokens,
                K: k,
                inFeatures: inFeatures,
                outFeatures: outFeatures,
                bits: bits)
            assertClose(actual, expected, tolerance: 1e-5)
        }
    }

    func testSlots8GatherScoredMatchesStackedGatherForDecodeToken() {
        MLX.Device.withDefaultDevice(.gpu) {
            let bits = 2
            let inFeatures = 8
            let outFeatures = 3
            let batchTokens = 1
            let k = 2
            let packedRows: [[UInt32]] = [
                [0, 1, 2, 3, 0, 1, 2, 3],
                [3, 2, 1, 0, 3, 2, 1, 0],
                [1, 1, 2, 2, 3, 3, 0, 0],
                [2, 0, 2, 0, 2, 0, 2, 0],
                [0, 3, 0, 3, 1, 2, 1, 2],
                [3, 3, 2, 2, 1, 1, 0, 0],
            ].map { [packJANGTQCodes($0, bits: bits)] }
            let packedStack = MLXArray(packedRows.flatMap { $0 })
                .reshaped([2, outFeatures, 1])
            let normsStack = MLXArray([
                Float(0.5), Float(1.0), Float(1.5),
                Float(1.25), Float(0.75), Float(0.25),
            ]).reshaped([2, outFeatures])
            let xRot = MLXArray([
                Float(0.5), Float(-1.25), Float(1.75), Float(2.25),
                Float(-3), Float(0.5), Float(1.0), Float(-0.75),
                Float(2.0), Float(1.25), Float(-0.25), Float(-1.0),
                Float(0.0), Float(1.5), Float(-2.0), Float(0.25),
            ]).reshaped([batchTokens * k, inFeatures])
            let codebook = MLXArray([Float(-1.0), Float(-0.25), Float(0.5), Float(1.75)])
            let rhsIndices = MLXArray([UInt32(1), UInt32(0)])
            let scores = MLXArray([Float(0.75), Float(0.25)])

            let gathered = JANGTQKernels.gatherTQ(
                xRot: xRot,
                packed: packedStack,
                norms: normsStack,
                codebook: codebook,
                rhsIndices: rhsIndices,
                nRows: batchTokens * k,
                inFeatures: inFeatures,
                outFeatures: outFeatures,
                bits: bits)
                .reshaped([batchTokens, k, outFeatures])
            let expected = (gathered * scores.reshaped([batchTokens, k, 1])).sum(axis: -2)
            let actual = JANGTQKernels.gatherTQTopKSlots8Scored(
                xRot: xRot,
                packed: [
                    MLXArray(Array(packedRows[3..<6]).flatMap { $0 }).reshaped([outFeatures, 1]),
                    MLXArray(Array(packedRows[0..<3]).flatMap { $0 }).reshaped([outFeatures, 1]),
                ],
                norms: [
                    MLXArray([Float(1.25), Float(0.75), Float(0.25)]),
                    MLXArray([Float(0.5), Float(1.0), Float(1.5)]),
                ],
                codebook: codebook,
                scores: scores,
                batchTokens: batchTokens,
                K: k,
                inFeatures: inFeatures,
                outFeatures: outFeatures,
                bits: bits)
            assertClose(actual, expected, tolerance: 1e-5)
        }
    }

    func testOffsetFusedGateUpMatchesStackedGateUpWithExpertMajorGaps() {
        MLX.Device.withDefaultDevice(.gpu) {
            let bits = 2
            let inFeatures = 8
            let outFeatures = 3
            let batchTokens = 2
            let k = 2
            let gateRows: [[UInt32]] = [
                [0, 1, 2, 3, 0, 1, 2, 3],
                [3, 2, 1, 0, 3, 2, 1, 0],
                [1, 1, 2, 2, 3, 3, 0, 0],
                [2, 0, 2, 0, 2, 0, 2, 0],
                [0, 3, 0, 3, 1, 2, 1, 2],
                [3, 3, 2, 2, 1, 1, 0, 0],
            ].map { [packJANGTQCodes($0, bits: bits)] }
            let upRows: [[UInt32]] = [
                [3, 1, 0, 2, 3, 1, 0, 2],
                [0, 2, 3, 1, 0, 2, 3, 1],
                [1, 2, 1, 2, 3, 0, 3, 0],
                [2, 2, 0, 0, 1, 1, 3, 3],
                [3, 0, 3, 0, 2, 1, 2, 1],
                [0, 1, 0, 1, 2, 3, 2, 3],
            ].map { [packJANGTQCodes($0, bits: bits)] }
            let packedGateStack = MLXArray(gateRows.flatMap { $0 })
                .reshaped([2, outFeatures, 1])
            let packedUpStack = MLXArray(upRows.flatMap { $0 })
                .reshaped([2, outFeatures, 1])
            let normsGateStack = MLXArray([
                Float(0.5), Float(1.0), Float(1.5),
                Float(1.25), Float(0.75), Float(0.25),
            ]).reshaped([2, outFeatures])
            let normsUpStack = MLXArray([
                Float(1.5), Float(0.25), Float(1.0),
                Float(0.5), Float(1.75), Float(0.75),
            ]).reshaped([2, outFeatures])

            let packedGateOffsets: [UInt32] = [2, 9]
            let packedUpOffsets: [UInt32] = [3, 11]
            let normsGateOffsets: [UInt32] = [1, 7]
            let normsUpOffsets: [UInt32] = [2, 8]
            let xRot = MLXArray([
                Float(1), Float(2), Float(-1), Float(0.5),
                Float(3), Float(-2), Float(0.25), Float(1.5),
                Float(-1.5), Float(0.75), Float(2.5), Float(-0.5),
                Float(1.25), Float(0.0), Float(-2.25), Float(3.5),
            ]).reshaped([batchTokens, inFeatures])
            let codebook = MLXArray([Float(-1.0), Float(-0.25), Float(0.5), Float(1.75)])
            let rhsIndices = MLXArray([UInt32(1), UInt32(0), UInt32(0), UInt32(1)])

            let expected = JANGTQKernels.fusedGateUpSwiGLU(
                xRot: xRot,
                packedGate: packedGateStack,
                normsGate: normsGateStack,
                packedUp: packedUpStack,
                normsUp: normsUpStack,
                codebook: codebook,
                rhsIndices: rhsIndices,
                batchTokens: batchTokens,
                K: k,
                inFeatures: inFeatures,
                outFeatures: outFeatures,
                bits: bits)
            let actual = JANGTQKernels.fusedGateUpSwiGLUOffsets(
                xRot: xRot,
                packedGate: MLXArray(makeOffsetSpan(
                    rows: gateRows, offsets: packedGateOffsets, outFeatures: outFeatures)),
                packedGateOffsets: MLXArray(packedGateOffsets),
                normsGate: MLXArray(makeOffsetSpan(
                    rows: normsGateStack.asArray(Float.self),
                    offsets: normsGateOffsets,
                    outFeatures: outFeatures)),
                normsGateOffsets: MLXArray(normsGateOffsets),
                packedUp: MLXArray(makeOffsetSpan(
                    rows: upRows, offsets: packedUpOffsets, outFeatures: outFeatures)),
                packedUpOffsets: MLXArray(packedUpOffsets),
                normsUp: MLXArray(makeOffsetSpan(
                    rows: normsUpStack.asArray(Float.self),
                    offsets: normsUpOffsets,
                    outFeatures: outFeatures)),
                normsUpOffsets: MLXArray(normsUpOffsets),
                codebook: codebook,
                rhsIndices: rhsIndices,
                batchTokens: batchTokens,
                K: k,
                inFeatures: inFeatures,
                outFeatures: outFeatures,
                bits: bits)
            assertClose(actual, expected, tolerance: 1e-5)
        }
    }

    func testOffsetFusedGateUpSplitShardSentinelsCanBeSummed() {
        MLX.Device.withDefaultDevice(.gpu) {
            let bits = 2
            let inFeatures = 8
            let outFeatures = 3
            let batchTokens = 2
            let k = 2
            let gateRows: [[UInt32]] = [
                [0, 1, 2, 3, 0, 1, 2, 3],
                [3, 2, 1, 0, 3, 2, 1, 0],
                [1, 1, 2, 2, 3, 3, 0, 0],
                [2, 0, 2, 0, 2, 0, 2, 0],
                [0, 3, 0, 3, 1, 2, 1, 2],
                [3, 3, 2, 2, 1, 1, 0, 0],
            ].map { [packJANGTQCodes($0, bits: bits)] }
            let upRows: [[UInt32]] = [
                [3, 1, 0, 2, 3, 1, 0, 2],
                [0, 2, 3, 1, 0, 2, 3, 1],
                [1, 2, 1, 2, 3, 0, 3, 0],
                [2, 2, 0, 0, 1, 1, 3, 3],
                [3, 0, 3, 0, 2, 1, 2, 1],
                [0, 1, 0, 1, 2, 3, 2, 3],
            ].map { [packJANGTQCodes($0, bits: bits)] }
            let packedGateStack = MLXArray(gateRows.flatMap { $0 })
                .reshaped([2, outFeatures, 1])
            let packedUpStack = MLXArray(upRows.flatMap { $0 })
                .reshaped([2, outFeatures, 1])
            let normsGateStack = MLXArray([
                Float(0.5), Float(1.0), Float(1.5),
                Float(1.25), Float(0.75), Float(0.25),
            ]).reshaped([2, outFeatures])
            let normsUpStack = MLXArray([
                Float(1.5), Float(0.25), Float(1.0),
                Float(0.5), Float(1.75), Float(0.75),
            ]).reshaped([2, outFeatures])
            let xRot = MLXArray([
                Float(1), Float(2), Float(-1), Float(0.5),
                Float(3), Float(-2), Float(0.25), Float(1.5),
                Float(-1.5), Float(0.75), Float(2.5), Float(-0.5),
                Float(1.25), Float(0.0), Float(-2.25), Float(3.5),
            ]).reshaped([batchTokens, inFeatures])
            let codebook = MLXArray([Float(-1.0), Float(-0.25), Float(0.5), Float(1.75)])
            let rhsIndices = MLXArray([UInt32(1), UInt32(0), UInt32(0), UInt32(1)])

            let expected = JANGTQKernels.fusedGateUpSwiGLU(
                xRot: xRot,
                packedGate: packedGateStack,
                normsGate: normsGateStack,
                packedUp: packedUpStack,
                normsUp: normsUpStack,
                codebook: codebook,
                rhsIndices: rhsIndices,
                batchTokens: batchTokens,
                K: k,
                inFeatures: inFeatures,
                outFeatures: outFeatures,
                bits: bits)
            let shard0 = JANGTQKernels.fusedGateUpSwiGLUOffsets(
                xRot: xRot,
                packedGate: MLXArray(makeOffsetSpan(
                    rows: gateRows, offsets: [2, UInt32.max], outFeatures: outFeatures)),
                packedGateOffsets: MLXArray([UInt32(2), UInt32.max]),
                normsGate: MLXArray(makeOffsetSpan(
                    rows: normsGateStack.asArray(Float.self),
                    offsets: [1, UInt32.max],
                    outFeatures: outFeatures)),
                normsGateOffsets: MLXArray([UInt32(1), UInt32.max]),
                packedUp: MLXArray(makeOffsetSpan(
                    rows: upRows, offsets: [3, UInt32.max], outFeatures: outFeatures)),
                packedUpOffsets: MLXArray([UInt32(3), UInt32.max]),
                normsUp: MLXArray(makeOffsetSpan(
                    rows: normsUpStack.asArray(Float.self),
                    offsets: [2, UInt32.max],
                    outFeatures: outFeatures)),
                normsUpOffsets: MLXArray([UInt32(2), UInt32.max]),
                codebook: codebook,
                rhsIndices: rhsIndices,
                batchTokens: batchTokens,
                K: k,
                inFeatures: inFeatures,
                outFeatures: outFeatures,
                bits: bits)
            let shard1 = JANGTQKernels.fusedGateUpSwiGLUOffsets(
                xRot: xRot,
                packedGate: MLXArray(makeOffsetSpan(
                    rows: gateRows, offsets: [UInt32.max, 4], outFeatures: outFeatures)),
                packedGateOffsets: MLXArray([UInt32.max, UInt32(4)]),
                normsGate: MLXArray(makeOffsetSpan(
                    rows: normsGateStack.asArray(Float.self),
                    offsets: [UInt32.max, 3],
                    outFeatures: outFeatures)),
                normsGateOffsets: MLXArray([UInt32.max, UInt32(3)]),
                packedUp: MLXArray(makeOffsetSpan(
                    rows: upRows, offsets: [UInt32.max, 5], outFeatures: outFeatures)),
                packedUpOffsets: MLXArray([UInt32.max, UInt32(5)]),
                normsUp: MLXArray(makeOffsetSpan(
                    rows: normsUpStack.asArray(Float.self),
                    offsets: [UInt32.max, 4],
                    outFeatures: outFeatures)),
                normsUpOffsets: MLXArray([UInt32.max, UInt32(4)]),
                codebook: codebook,
                rhsIndices: rhsIndices,
                batchTokens: batchTokens,
                K: k,
                inFeatures: inFeatures,
                outFeatures: outFeatures,
                bits: bits)
            assertClose(shard0 + shard1, expected, tolerance: 1e-5)
        }
    }

    func testSlots8FusedGateUpMatchesStackedGateUpForDecodeToken() {
        MLX.Device.withDefaultDevice(.gpu) {
            let bits = 2
            let inFeatures = 8
            let outFeatures = 3
            let batchTokens = 1
            let k = 2
            let gateRows: [[UInt32]] = [
                [0, 1, 2, 3, 0, 1, 2, 3],
                [3, 2, 1, 0, 3, 2, 1, 0],
                [1, 1, 2, 2, 3, 3, 0, 0],
                [2, 0, 2, 0, 2, 0, 2, 0],
                [0, 3, 0, 3, 1, 2, 1, 2],
                [3, 3, 2, 2, 1, 1, 0, 0],
            ].map { [packJANGTQCodes($0, bits: bits)] }
            let upRows: [[UInt32]] = [
                [3, 1, 0, 2, 3, 1, 0, 2],
                [0, 2, 3, 1, 0, 2, 3, 1],
                [1, 2, 1, 2, 3, 0, 3, 0],
                [2, 2, 0, 0, 1, 1, 3, 3],
                [3, 0, 3, 0, 2, 1, 2, 1],
                [0, 1, 0, 1, 2, 3, 2, 3],
            ].map { [packJANGTQCodes($0, bits: bits)] }
            let packedGateStack = MLXArray(gateRows.flatMap { $0 })
                .reshaped([2, outFeatures, 1])
            let packedUpStack = MLXArray(upRows.flatMap { $0 })
                .reshaped([2, outFeatures, 1])
            let normsGateStack = MLXArray([
                Float(0.5), Float(1.0), Float(1.5),
                Float(1.25), Float(0.75), Float(0.25),
            ]).reshaped([2, outFeatures])
            let normsUpStack = MLXArray([
                Float(1.5), Float(0.25), Float(1.0),
                Float(0.5), Float(1.75), Float(0.75),
            ]).reshaped([2, outFeatures])
            let xRot = MLXArray([
                Float(1), Float(2), Float(-1), Float(0.5),
                Float(3), Float(-2), Float(0.25), Float(1.5),
            ]).reshaped([batchTokens, inFeatures])
            let codebook = MLXArray([Float(-1.0), Float(-0.25), Float(0.5), Float(1.75)])
            let rhsIndices = MLXArray([UInt32(1), UInt32(0)])

            let expected = JANGTQKernels.fusedGateUpSwiGLU(
                xRot: xRot,
                packedGate: packedGateStack,
                normsGate: normsGateStack,
                packedUp: packedUpStack,
                normsUp: normsUpStack,
                codebook: codebook,
                rhsIndices: rhsIndices,
                batchTokens: batchTokens,
                K: k,
                inFeatures: inFeatures,
                outFeatures: outFeatures,
                bits: bits)
            let actual = JANGTQKernels.fusedGateUpSwiGLUSlots8(
                xRot: xRot,
                packedGate: [
                    MLXArray(Array(gateRows[3..<6]).flatMap { $0 }).reshaped([outFeatures, 1]),
                    MLXArray(Array(gateRows[0..<3]).flatMap { $0 }).reshaped([outFeatures, 1]),
                ],
                normsGate: [
                    MLXArray([Float(1.25), Float(0.75), Float(0.25)]),
                    MLXArray([Float(0.5), Float(1.0), Float(1.5)]),
                ],
                packedUp: [
                    MLXArray(Array(upRows[3..<6]).flatMap { $0 }).reshaped([outFeatures, 1]),
                    MLXArray(Array(upRows[0..<3]).flatMap { $0 }).reshaped([outFeatures, 1]),
                ],
                normsUp: [
                    MLXArray([Float(0.5), Float(1.75), Float(0.75)]),
                    MLXArray([Float(1.5), Float(0.25), Float(1.0)]),
                ],
                codebook: codebook,
                batchTokens: batchTokens,
                K: k,
                inFeatures: inFeatures,
                outFeatures: outFeatures,
                bits: bits)
            assertClose(actual, expected, tolerance: 1e-5)
        }
    }
}

private func hadamardReference(_ x: [Float], signs: [Float], blocks: [Int]) -> [Float] {
    var output: [Float] = []
    output.reserveCapacity(x.count)
    var offset = 0
    for block in blocks {
        var values = (0..<block).map { x[offset + $0] * signs[offset + $0] }
        var half = 1
        while half < block {
            let stride = half * 2
            var base = 0
            while base < block {
                for i in 0..<half {
                    let lower = values[base + i]
                    let upper = values[base + i + half]
                    values[base + i] = lower + upper
                    values[base + i + half] = lower - upper
                }
                base += stride
            }
            half *= 2
        }
        let norm = Float(1.0 / sqrt(Double(block)))
        output.append(contentsOf: values.map { $0 * norm })
        offset += block
    }
    return output
}

private func packJANGTQCodes(_ codes: [UInt32], bits: Int) -> UInt32 {
    var value: UInt32 = 0
    let mask = UInt32((1 << bits) - 1)
    for (index, code) in codes.enumerated() {
        value |= (code & mask) << UInt32(index * bits)
    }
    return value
}

private func makeOffsetSpan(
    rows: [[UInt32]],
    offsets: [UInt32],
    outFeatures: Int
) -> [UInt32] {
    let count = offsets
        .filter { $0 != UInt32.max }
        .map { Int($0) + outFeatures }
        .max() ?? 0
    var span = [UInt32](repeating: 0xFFFF_FFFF, count: count)
    for expert in 0..<offsets.count {
        guard offsets[expert] != UInt32.max else { continue }
        for output in 0..<outFeatures {
            span[Int(offsets[expert]) + output] = rows[expert * outFeatures + output][0]
        }
    }
    return span
}

private func makeOffsetSpan(
    rows: [Float],
    offsets: [UInt32],
    outFeatures: Int
) -> [Float] {
    let count = offsets
        .filter { $0 != UInt32.max }
        .map { Int($0) + outFeatures }
        .max() ?? 0
    var span = [Float](repeating: -7, count: count)
    for expert in 0..<offsets.count {
        guard offsets[expert] != UInt32.max else { continue }
        for output in 0..<outFeatures {
            span[Int(offsets[expert]) + output] = rows[expert * outFeatures + output]
        }
    }
    return span
}

private func assertClose(_ actual: MLXArray, _ expected: MLXArray, tolerance: Float) {
    MLX.eval(actual, expected)
    let actualValues = actual.asArray(Float.self)
    let expectedValues = expected.asArray(Float.self)
    XCTAssertEqual(actualValues.count, expectedValues.count)
    let maxDiff = zip(actualValues, expectedValues)
        .map { abs($0 - $1) }
        .max() ?? 0
    XCTAssertLessThanOrEqual(maxDiff, tolerance, "maxDiff=\(maxDiff)")
}

private let focusedTestRepoRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .standardizedFileURL

private let focusedMetallibSourceDirectory: URL? = {
    let sourceDirectories = [
        focusedTestRepoRoot.appendingPathComponent(".build/arm64-apple-macosx/debug"),
        focusedTestRepoRoot.appendingPathComponent(".build/debug"),
    ]
    return sourceDirectories.first {
        FileManager.default.fileExists(atPath: $0.appendingPathComponent("default.metallib").path)
    }
}()

private final class FocusedTestBundleProbe {}

private let focusedMetallibPrepared: Void = {
    guard let sourceDirectory = focusedMetallibSourceDirectory else { return }
    let fileManager = FileManager.default
    let source = sourceDirectory.appendingPathComponent("default.metallib")

    var targetDirectories: [URL] = []
    if let executableURL = Bundle.main.executableURL {
        targetDirectories.append(executableURL.deletingLastPathComponent())
    }
    if let resourceURL = Bundle.main.resourceURL {
        targetDirectories.append(resourceURL)
    }
    let testBundle = Bundle(for: FocusedTestBundleProbe.self)
    if let executableURL = testBundle.executableURL {
        targetDirectories.append(executableURL.deletingLastPathComponent())
    }
    if let resourceURL = testBundle.resourceURL {
        targetDirectories.append(resourceURL)
    }
    if let firstArgument = CommandLine.arguments.first, !firstArgument.isEmpty {
        targetDirectories.append(URL(fileURLWithPath: firstArgument).deletingLastPathComponent())
    }

    var scanned = Set<String>()
    for candidate in targetDirectories {
        var directory = candidate.standardizedFileURL
        for _ in 0..<4 {
            if scanned.insert(directory.path).inserted {
                try? fileManager.copyMetallibIfMissing(from: source, into: directory)
            }
            directory.deleteLastPathComponent()
        }
    }
}()

private func prepareMLXMetallibForFocusedTests() {
    _ = focusedMetallibPrepared
}

private extension FileManager {
    func copyMetallibIfMissing(from source: URL, into directory: URL) throws {
        try createDirectory(at: directory, withIntermediateDirectories: true)
        for name in ["default.metallib", "mlx.metallib"] {
            let destination = directory.appendingPathComponent(name)
            if !fileExists(atPath: destination.path) {
                try copyItem(at: source, to: destination)
            }
        }
    }
}
