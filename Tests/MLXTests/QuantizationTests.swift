// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

class QuantizationTests: XCTestCase {

    func testOneBitAffineNegativeEndpointsAndSmallKThroughCBridge() {
        for dtype in [DType.float32, .float16, .bfloat16] {
            for width in [64, 128] {
                for group in [32, 64, 128] where width.isMultiple(of: group) {
                    for stream in [StreamOrDevice.cpu, .gpu] {
                        let weights = MLXArray(
                            (0 ..< 16 * width).map { $0.isMultiple(of: 2) ? Float(-2) : Float(-1) },
                            [16, width]
                        ).asType(dtype)
                        let (packed, scales, biases) = MLX.quantized(
                            weights, groupSize: group, bits: 1, mode: .affine, stream: stream)
                        let restored = MLX.dequantized(
                            packed, scales: scales, biases: biases, groupSize: group,
                            bits: 1, mode: .affine, stream: stream)
                        XCTAssertEqual(MLX.abs(restored - weights).max().item(Float.self), 0)
                        let input = MLXArray.ones([1, width], dtype: dtype)
                        let actual = MLX.quantizedMM(
                            input, packed, scales: scales, biases: biases, transpose: true,
                            groupSize: group, bits: 1, mode: .affine, stream: stream)
                        let expected = MLXArray.full([1, 16], values: Float(-width * 3 / 2))
                        XCTAssertEqual(MLX.abs(actual - expected).max().item(Float.self), 0)
                    }
                }
            }
        }
    }
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

    func testCPUAffineGatherQuantizedMMKeepsBitsAndGroupSizeOrder() {
        Device.withDefaultDevice(.cpu) {
            Stream.withNewDefaultStream(device: .cpu) {
                let dense =
                    (MLXArray(0 ..< (3 * 16 * 32), [3, 16, 32]).asType(.float32) / 2048)
                    - 0.5
                let x =
                    (MLXArray(0 ..< (2 * 1 * 32), [2, 1, 32]).asType(.float32) / 128)
                    - 0.25
                let (weight, scales, biases) = MLX.quantized(
                    dense,
                    groupSize: 32,
                    bits: 4,
                    mode: .affine,
                    stream: .cpu)

                let gathered = MLX.gatherQuantizedMM(
                    x,
                    weight,
                    scales: scales,
                    biases: biases,
                    rhsIndices: MLXArray([0, 2]),
                    transpose: true,
                    groupSize: 32,
                    bits: 4,
                    mode: .affine,
                    stream: .cpu)
                MLX.eval(gathered)

                let dequantized = MLX.dequantized(
                    weight,
                    scales: scales,
                    biases: biases,
                    groupSize: 32,
                    bits: 4,
                    mode: .affine,
                    stream: .cpu)
                let expected = concatenated(
                    [
                        matmul(x[0 ..< 1], dequantized[0].transposed()),
                        matmul(x[1 ..< 2], dequantized[2].transposed()),
                    ], axis: 0, stream: .cpu)

                assertEqual(gathered, expected, rtol: 1e-3, atol: 1e-3)
            }
        }
    }
}
