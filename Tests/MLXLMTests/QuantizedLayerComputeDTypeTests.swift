// Copyright © 2026 osaurus-eval contributors
// SPDX-License-Identifier: MIT
//
// A quantized layer's `weight` is not a float tensor.
//
// Several models took a compute dtype from `layer.weight.dtype`: to cast an input before calling the
// layer, or to type a scalar. For a float layer that is the layer's dtype. A bundle that ships the
// layer quantized loads it as a `QuantizedLinear` or `QuantizedEmbedding`, whose `weight` holds the
// PACKED codes, so the dtype is uint32 and whatever was cast to it was truncated to an integer.
// `computeDType` names the dtype a layer actually computes in, and every such site now asks for it.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

@Suite("Compute dtype of possibly-quantized layers")
struct QuantizedLayerComputeDTypeTests {

    @Test("a float layer computes in its weight's dtype")
    func floatLayers() throws {
        try MLXMetalTestLock.withLock {
            for dtype in [DType.float32, .bfloat16, .float16] {
                let linear = Linear(64, 32)
                linear.apply { $0.asType(dtype) }
                #expect(linear.computeDType == dtype)
                let embedding = Embedding(embeddingCount: 16, dimensions: 64)
                embedding.apply { $0.asType(dtype) }
                #expect(embedding.computeDType == dtype)
            }
        }
    }

    /// The oracle is MLX's own `dequantized`: a quantized layer computes in the dtype its codes
    /// decode to, which is not the packed weight's.
    @Test("a quantized layer computes in the dtype its codes dequantize to")
    func quantizedLayers() throws {
        try MLXMetalTestLock.withLock {
            let cases: [(QuantizationMode, Int, Int, DType)] = [
                (.affine, 32, 4, .float16), (.affine, 32, 4, .bfloat16), (.affine, 64, 8, .float32),
                (.mxfp4, 32, 4, .float16),
            ]
            for (mode, groupSize, bits, dtype) in cases {
                let w = MLXRandom.normal([32, 64], key: MLXRandom.key(170)).asType(dtype)
                let (wq, scales, biases) = MLX.quantized(
                    w, groupSize: groupSize, bits: bits, mode: mode)
                let decoded = dequantized(
                    wq, scales: scales, biases: biases, groupSize: groupSize, bits: bits,
                    mode: mode
                ).dtype

                let linear = QuantizedLinear(
                    weight: wq, bias: nil, scales: scales, biases: biases, groupSize: groupSize,
                    bits: bits, mode: mode)
                #expect(linear.weight.dtype == .uint32)
                #expect(linear.computeDType == decoded, "\(mode) from \(dtype)")

                let embedding = QuantizedEmbedding(
                    weight: wq, scales: scales, biases: biases, groupSize: groupSize, bits: bits,
                    mode: mode)
                #expect(embedding.computeDType == decoded, "\(mode) from \(dtype)")
                #expect(embedding(MLXArray([Int32(0), 1])).dtype == decoded)
            }
        }
    }

    /// The loader pins some embeddings' output dtype, so their rows come back in that dtype.
    @Test("an embedding with a pinned output dtype computes in it")
    func pinnedEmbedding() throws {
        try MLXMetalTestLock.withLock {
            let w = MLXRandom.normal([16, 64], key: MLXRandom.key(171)).asType(.float16)
            let (wq, scales, biases) = MLX.quantized(w, groupSize: 32, bits: 4)
            let embedding = QuantizedEmbedding(
                weight: wq, scales: scales, biases: biases, groupSize: 32, bits: 4)
            embedding.outputDType = .bfloat16
            #expect(embedding.computeDType == .bfloat16)
            #expect(embedding(MLXArray([Int32(3)])).dtype == .bfloat16)
        }
    }
}
