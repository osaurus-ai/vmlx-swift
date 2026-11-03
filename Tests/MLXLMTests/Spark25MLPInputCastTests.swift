import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

enum Spark25InputCastFixture {
    static func layers() -> (QuantizedLinear, QuantizedLinear) {
        func layer(_ seed: UInt64) -> QuantizedLinear {
            QuantizedLinear(
                weight: (MLXRandom.normal([10240, 2560], key: MLXRandom.key(seed)) * 0.02)
                    .asType(.bfloat16), bias: nil, groupSize: 64, bits: 6)
        }
        return (layer(7), layer(19))
    }
}

final class Spark25MLPInputCastTests: XCTestCase {
    private func requireCandidate() throws {
        #if !canImport(Metal)
            throw XCTSkip("Metal unavailable")
        #else
            let env = ProcessInfo.processInfo.environment
            try XCTSkipIf(
                env["VMLX_SPARK_MLP_INPUT_CAST_REFERENCE"] == "1"
                    || env["VMLX_SPARK_DUAL_Q6_REFERENCE"] == "1",
                "Reference-only diagnostic requested")
        #endif
    }

    private func reference(_ x: MLXArray, gate: Linear, up: Linear) -> MLXArray {
        let cast = x.asType(.bfloat16)
        return Spark25DualProjection.apply(cast, gate: gate, up: up) ?? gelu(gate(cast)) * up(cast)
    }

    private func expectBits(_ actual: MLXArray, _ expected: MLXArray) {
        eval(actual, expected)
        XCTAssertEqual(actual.shape, expected.shape)
        XCTAssertEqual(actual.dtype, .bfloat16)
        XCTAssertEqual(expected.dtype, .bfloat16)
        let a = actual.view(dtype: .uint16).asArray(UInt16.self)
        let b = expected.view(dtype: .uint16).asArray(UInt16.self)
        // Fixed-input parity is strict, including NaN payload and signed zero.
        XCTAssertEqual(a, b)
    }

    func testNativeNormSeed1004AndOtherSeeds() throws {
        try requireCandidate()
        try MLXMetalTestLock.withLock {
            try MLX.Stream.withNewDefaultStream(device: .gpu) {
                let (g, u) = Spark25InputCastFixture.layers()
                let normWeight = (MLXRandom.normal([2560], key: MLXRandom.key(313)) * 0.1 + 1)
                    .asType(.bfloat16)
                eval(g.parameters(), u.parameters(), normWeight)
                for seed: UInt64 in [1004, 7, 19, 31] {
                    let base = MLXRandom.normal([16, 1, 2560], key: MLXRandom.key(seed))
                    for i in 0 ..< 16 {
                        // Preserve the native FP32 normalization even for the
                        // seed that rejected the separate custom norm kernel.
                        let x = MLXFast.rmsNorm(
                            base[i ..< i + 1, 0..., 0...], weight: normWeight, eps: 1e-6)
                        eval(x)
                        let before = x.view(dtype: .uint32).asArray(UInt32.self)
                        let actual = try XCTUnwrap(
                            Spark25DualProjection.applyNormalizedFP32(
                                x, inputDType: .bfloat16, gate: g, up: u))
                        expectBits(actual, reference(x, gate: g, up: u))
                        XCTAssertEqual(x.view(dtype: .uint32).asArray(UInt32.self), before)
                    }
                }
            }
        }
    }

    func testLazyNativeNormAdmitsBeforeEvaluation() throws {
        try requireCandidate()
        try MLXMetalTestLock.withLock {
            try MLX.Stream.withNewDefaultStream(device: .gpu) {
                let (g, u) = Spark25InputCastFixture.layers()
                let normWeight = (MLXRandom.normal([2560], key: MLXRandom.key(313)) * 0.1 + 1)
                    .asType(.bfloat16)
                eval(g.parameters(), u.parameters(), normWeight)
                let input = MLXRandom.normal([1, 1, 2560], key: MLXRandom.key(1004))
                let normalized = MLXFast.rmsNorm(input, weight: normWeight, eps: 1e-6)
                // This unwrap precedes every eval/readback of the normalized
                // result, exactly as the ordinary decoder constructs its MLP.
                let actual = try XCTUnwrap(
                    Spark25DualProjection.applyNormalizedFP32(
                        normalized, inputDType: .bfloat16, gate: g, up: u))
                expectBits(actual, reference(normalized, gate: g, up: u))
            }
        }
    }

    func testRoundTiesPartialSumsSignedZeroAndNonfinite() throws {
        try requireCandidate()
        try MLXMetalTestLock.withLock {
            try MLX.Stream.withNewDefaultStream(device: .gpu) {
                let (g, u) = Spark25InputCastFixture.layers()
                eval(g.parameters(), u.parameters())
                let patterns: [[UInt32]] = [
                    [
                        0x3f80_7fff, 0x3f80_8000, 0x3f80_8001, 0xbf80_7fff, 0xbf80_8000,
                        0xbf80_8001, 0x4181_0001, 0xc180_ffff,
                    ],
                    [
                        0, 0x8000_0000, 1, 0x8000_0001, 0x0080_0000, 0x8080_0000, 0x3a00_0001,
                        0xba00_0001,
                    ],
                    [
                        0x7f80_0000, 0xff80_0000, 0x7fc0_0001, 0, 0x8000_0000, 0x3f80_0000,
                        0xbf80_0000, 0x7f7f_ffff,
                    ],
                ]
                for pattern in patterns {
                    let values = (0 ..< 2560).map { pattern[$0 % pattern.count] }
                    let x = MLXArray(values).view(dtype: .float32).reshaped([1, 1, 2560])
                    eval(x)
                    let actual = try XCTUnwrap(
                        Spark25DualProjection.applyNormalizedFP32(
                            x, inputDType: .bfloat16, gate: g, up: u))
                    expectBits(actual, reference(x, gate: g, up: u))
                }
            }
        }
    }

    func testStridedFallbackAndLazyValues() throws {
        try requireCandidate()
        try MLXMetalTestLock.withLock {
            try MLX.Stream.withNewDefaultStream(device: .gpu) {
                let (g, u) = Spark25InputCastFixture.layers()
                eval(g.parameters(), u.parameters())
                for step in [2, -2] {
                    let base = MLXRandom.normal([1, 1, 5120], key: MLXRandom.key(43))
                    let x = base[0..., 0..., .stride(by: step)]
                    let lazy =
                        Spark25DualProjection.applyNormalizedFP32(
                            x, inputDType: .bfloat16, gate: g, up: u)
                        ?? reference(x, gate: g, up: u)
                    expectBits(lazy, reference(x, gate: g, up: u))
                    eval(x)
                    XCTAssertNil(
                        Spark25DualProjection.applyNormalizedFP32(
                            x, inputDType: .bfloat16, gate: g, up: u))
                }
            }
        }
    }

    func testCPUShapeDtypeAndProjectionFallbacks() {
        MLXMetalTestLock.withLock {
            let (g, u) = Spark25InputCastFixture.layers()
            let x = MLXArray.zeros([1, 1, 2560])
            MLX.Stream.withNewDefaultStream(device: .cpu) {
                XCTAssertNil(
                    Spark25DualProjection.applyNormalizedFP32(
                        x, inputDType: .bfloat16, gate: g, up: u))
            }
            for dtype: DType in [.float16, .float32] {
                XCTAssertNil(
                    Spark25DualProjection.applyNormalizedFP32(x, inputDType: dtype, gate: g, up: u))
            }
            for dtype: DType in [.float16, .bfloat16] {
                XCTAssertNil(
                    Spark25DualProjection.applyNormalizedFP32(
                        x.asType(dtype), inputDType: .bfloat16, gate: g, up: u))
            }
            for shape in [[1, 2, 2560], [2, 1, 2560], [1, 2560], [1, 1, 1280]] {
                XCTAssertNil(
                    Spark25DualProjection.applyNormalizedFP32(
                        MLXArray.zeros(shape), inputDType: .bfloat16, gate: g, up: u))
            }
            let biased = QuantizedLinear(
                weight: g.weight, bias: MLXArray.zeros([10240], dtype: .bfloat16), scales: g.scales,
                biases: g.biases, groupSize: 64, bits: 6)
            XCTAssertNil(
                Spark25DualProjection.applyNormalizedFP32(
                    x, inputDType: .bfloat16, gate: biased, up: u))
            let metadata = QuantizedLinear(
                weight: g.weight, scales: g.scales.asType(.float16),
                biases: g.biases?.asType(.float16), groupSize: 64, bits: 6)
            XCTAssertNil(
                Spark25DualProjection.applyNormalizedFP32(
                    x, inputDType: .bfloat16, gate: metadata, up: u))
        }
    }

    func testCompiledFallbackUsesOriginalCast() throws {
        try MLXMetalTestLock.withLock {
            let (g, u) = Spark25InputCastFixture.layers()
            eval(g.parameters(), u.parameters())
            let x = MLXRandom.normal([1, 1, 2560], key: MLXRandom.key(101))
            let compiled = compile { (a: MLXArray) in
                Spark25DualProjection.applyNormalizedFP32(a, inputDType: .bfloat16, gate: g, up: u)
                    ?? self.reference(a, gate: g, up: u)
            }
            expectBits(compiled(x), reference(x, gate: g, up: u))
        }
    }

    func testDerivativeFallbackUsesOriginalCast() {
        MLXMetalTestLock.withLock {
            let (g, u) = Spark25InputCastFixture.layers()
            eval(g.parameters(), u.parameters())
            let x = MLXRandom.normal([1, 1, 2560], key: MLXRandom.key(103))
            let cotangent = MLXArray.ones([1, 1, 10240], dtype: .bfloat16)
            let expected = vjp(
                { [self.reference($0[0], gate: g, up: u)] }, primals: [x], cotangents: [cotangent])
            let actual = vjp(
                {
                    [
                        Spark25DualProjection.applyNormalizedFP32(
                            $0[0], inputDType: .bfloat16, gate: g, up: u)
                            ?? self.reference($0[0], gate: g, up: u)
                    ]
                }, primals: [x], cotangents: [cotangent])
            for (a, b) in zip(actual.1, expected.1) {
                eval(a, b)
                XCTAssertEqual(a.dtype, b.dtype)
                XCTAssertEqual(
                    a.view(dtype: .uint32).asArray(UInt32.self),
                    b.view(dtype: .uint32).asArray(UInt32.self))
            }
        }
    }

    func testOrdinaryMLPIntegrationAndFallbackInputDType() throws {
        try requireCandidate()
        try MLXMetalTestLock.withLock {
            try MLX.Stream.withNewDefaultStream(device: .gpu) {
                let data = try JSONSerialization.data(withJSONObject: [
                    "hidden_size": 2560, "num_hidden_layers": 1,
                    "intermediate_size": 10240, "num_attention_heads": 1, "num_key_value_heads": 1,
                    "head_dim": 32, "vocab_size": 16,
                ])
                let mlp = Spark25MLP(
                    try JSONDecoder().decode(Spark25Configuration.self, from: data))
                let (g, u) = Spark25InputCastFixture.layers()
                let down = QuantizedLinear(
                    weight: (MLXRandom.normal([2560, 10240], key: MLXRandom.key(109)) * 0.02)
                        .asType(.bfloat16), bias: nil, groupSize: 64, bits: 6)
                mlp.update(modules: ModuleChildren.unflattened([
                    ("gate_proj", g), ("up_proj", u), ("down_proj", down),
                ]))
                XCTAssertTrue(mlp.gate === g && mlp.up === u && mlp.down === down)
                eval(mlp.parameters())
                for shape in [[1, 1, 2560], [1, 2, 2560]] {
                    let x = MLXRandom.normal(shape, key: MLXRandom.key(113))
                    eval(x)
                    expectBits(
                        mlp.callWithNormalizedInput(x, inputDType: .bfloat16),
                        mlp(x.asType(.bfloat16)))
                }
                // Other norm dtypes take the exact original explicit cast.
                let x = MLXRandom.normal([1, 1, 2560], key: MLXRandom.key(127))
                let actual = mlp.callWithNormalizedInput(x, inputDType: .float32)
                let expected = mlp(x.asType(.float32))
                eval(actual, expected)
                XCTAssertEqual(actual.dtype, expected.dtype)
                XCTAssertTrue(all(actual .== expected).item(Bool.self))
            }
        }
    }
}
