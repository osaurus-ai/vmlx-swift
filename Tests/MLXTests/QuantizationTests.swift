// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

class QuantizationTests: XCTestCase {
    func testQuantizedLinearShapeDesc() {
        let linear1 = Linear(512, 1024)
        let quantized1 = linear1.toQuantized(groupSize: 64, bits: 4)
        XCTAssertEqual(
            quantized1.describeExtra(0), "(inputDimensions=512, outputDimensions=1024, bias=true)")
        let linear2 = Linear(1024, 512, bias: false)
        let quantized2 = linear2.toQuantized(groupSize: 128, bits: 8)
        XCTAssertEqual(
            quantized2.describeExtra(0), "(inputDimensions=1024, outputDimensions=512, bias=false)")
        let linear3 = Linear(512, 1024)
        let quantized3 = linear3.toQuantized(groupSize: 32, bits: 4, mode: .mxfp4)
        XCTAssertEqual(
            quantized3.describeExtra(0), "(inputDimensions=512, outputDimensions=1024, bias=true)")
    }

    func testQuantizedEmbeddingShapeDesc() {
        let embedding1 = Embedding(embeddingCount: 512, dimensions: 1024)
        let quantized1 = embedding1.toQuantized(groupSize: 64, bits: 4)
        XCTAssertEqual(quantized1.describeExtra(0), "(embeddingCount=512, dimensions=1024)")
        let embedding2 = Embedding(embeddingCount: 1024, dimensions: 512)
        let quantized2 = embedding2.toQuantized(groupSize: 128, bits: 8)
        XCTAssertEqual(
            quantized2.describeExtra(0), "(embeddingCount=1024, dimensions=512)")
        let embedding3 = Embedding(embeddingCount: 512, dimensions: 1024)
        let quantized3 = embedding3.toQuantized(groupSize: 32, bits: 4, mode: .mxfp4)
        XCTAssertEqual(
            quantized3.describeExtra(0), "(embeddingCount=512, dimensions=1024)")
    }

    func testQuantizedEmbeddingCheckpointInitializerRestoresUnpackedDimension() {
        let dense = MLXRandom.normal([16, 64])
        let (weight, scales, biases) = MLX.quantized(
            dense, groupSize: 32, bits: 4, mode: .affine)
        let embedding = QuantizedEmbedding(
            weight: weight,
            scales: scales,
            biases: biases,
            groupSize: 32,
            bits: 4,
            mode: .affine)

        let lookedUp = embedding(MLXArray([0, 7, 15]))
        XCTAssertEqual(lookedUp.shape, [3, 64])

        let projected = embedding.asLinear(MLXArray.zeros([2, 64]))
        XCTAssertEqual(projected.shape, [2, 16])
    }

    func testQuantizedLinearInitializerPreservesMXFP8Mode() {
        let linear = Linear(64, 32, bias: false)
        let quantized = QuantizedLinear(linear, groupSize: 32, bits: 8, mode: .mxfp8)

        XCTAssertEqual(quantized.mode, .mxfp8)
        XCTAssertNil(quantized.biases)
    }
}
