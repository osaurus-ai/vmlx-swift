import CmlxGraphShim
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class K2GroupedNormFusionTests: XCTestCase {
    private func reference(_ x: MLXArray, weight: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(
            x.reshaped([1, 1, 4, 1024]),
            weight: MLXArray.ones([1024], dtype: .bfloat16), eps: 1e-6
        ).reshaped(x.shape) * weight
    }

    private func bits(_ x: MLXArray) -> [UInt16] {
        x.view(dtype: .uint16).asArray(UInt16.self)
    }

    private func expectSameBits(
        _ actual: MLXArray, _ expected: MLXArray,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        eval(actual, expected)
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertEqual(actual.dtype, expected.dtype, file: file, line: line)
        let a = bits(actual)
        let e = bits(expected)
        func isNaN(_ b: UInt16) -> Bool { b & 0x7f80 == 0x7f80 && b & 0x007f != 0 }
        // Compare finite values, infinities and signed zeros bit for bit.
        // A NaN must remain a NaN; its payload need not survive arithmetic.
        let differences = zip(a, e).filter {
            $0 != $1 && !(isNaN($0) && isNaN($1))
        }.count
        XCTAssertEqual(differences, 0, file: file, line: line)
    }

    private func requireFusionEnabled() throws {
        #if !canImport(Metal)
            throw XCTSkip("Metal kernel unavailable on this platform")
        #else
            try XCTSkipIf(
                K2GroupedNormFusion.referenceOverride,
                "Reference-only A/B process requested")
        #endif
    }

    func testExactGroupedReductionAndInputWeightAliases() throws {
        try requireFusionEnabled()
        try MLXMetalTestLock.withLock {
            try Stream.withNewDefaultStream(device: .gpu) {
                for seed: UInt64 in [7, 19, 43] {
                    let scales = MLXArray([Float(0.003), 1, 64, 1000])
                        .reshaped([1, 1, 4, 1])
                    let x =
                        (MLXRandom.normal([1, 1, 4, 1024], key: MLXRandom.key(seed))
                        * scales).reshaped([1, 1, 4096]).asType(.bfloat16)
                    let weight = (MLXRandom.normal([4096], key: MLXRandom.key(seed + 1)) * 4)
                        .asType(.bfloat16)
                    eval(x, weight)
                    let inputAlias = x
                    let weightAlias = weight
                    let inputBefore = bits(x)
                    let weightBefore = bits(weight)
                    let actual = try XCTUnwrap(
                        K2GroupedNormFusion.apply(x, weight: weight, groups: 4, eps: 1e-6))
                    expectSameBits(actual, reference(x, weight: weight))
                    XCTAssertEqual(bits(inputAlias), inputBefore)
                    XCTAssertEqual(bits(weightAlias), weightBefore)
                }
            }
        }
    }

    func testNormalizationRoundsBeforeCheckpointWeight() throws {
        try requireFusionEnabled()
        try MLXMetalTestLock.withLock {
            try Stream.withNewDefaultStream(device: .gpu) {
                let x = MLXRandom.normal([1, 1, 4096], key: MLXRandom.key(73))
                    .asType(.bfloat16)
                let weight = (MLXRandom.normal([4096], key: MLXRandom.key(79)) * 3)
                    .asType(.bfloat16)
                eval(x, weight)
                let expected = reference(x, weight: weight)
                let unrounded =
                    (MLXFast.rmsNorm(
                        x.asType(.float32).reshaped([1, 1, 4, 1024]),
                        weight: MLXArray.ones([1024]), eps: 1e-6
                    ).reshaped(x.shape) * weight.asType(.float32)).asType(.bfloat16)
                eval(expected, unrounded)
                XCTAssertFalse(
                    all(expected .== unrounded).item(Bool.self),
                    "Fixture must detect loss of the intermediate BF16 rounding")
                let actual = try XCTUnwrap(
                    K2GroupedNormFusion.apply(x, weight: weight, groups: 4, eps: 1e-6))
                expectSameBits(actual, expected)
            }
        }
    }

    func testSignedZeroInfinityAndNaNSemantics() throws {
        try requireFusionEnabled()
        try MLXMetalTestLock.withLock {
            try Stream.withNewDefaultStream(device: .gpu) {
                var inputBits = (0 ..< 4096).map { UInt16($0.isMultiple(of: 2) ? 0 : 0x8000) }
                var weightBits = (0 ..< 4096).map { UInt16($0.isMultiple(of: 3) ? 0xbf80 : 0x3f80) }
                // Group 0 isolates positive/negative zero. Group 1 gives
                // finite normalized values nonfinite checkpoint weights.
                for i in 1024 ..< 4096 { inputBits[i] = i.isMultiple(of: 2) ? 0x3f80 : 0xbf80 }
                weightBits[1024] = 0x7f80
                weightBits[1025] = 0xff80
                weightBits[1026] = 0x7fc1
                weightBits[1027] = 0x0000
                // One infinite input makes its group's inverse RMS zero;
                // one NaN propagates only within its own contiguous group.
                inputBits[2048] = 0x7f80
                inputBits[3072] = 0x7fc1
                let x = MLXArray(inputBits).view(dtype: .bfloat16).reshaped([1, 1, 4096])
                let weight = MLXArray(weightBits).view(dtype: .bfloat16)
                eval(x, weight)
                let actual = try XCTUnwrap(
                    K2GroupedNormFusion.apply(x, weight: weight, groups: 4, eps: 1e-6))
                expectSameBits(actual, reference(x, weight: weight))
            }
        }
    }

    func testUnsupportedGeometryDtypesAndExplicitReferenceDecline() {
        MLXMetalTestLock.withLock {
            let x = MLXArray.zeros([1, 1, 4096], dtype: .bfloat16)
            let weight = MLXArray.ones([4096], dtype: .bfloat16)
            XCTAssertNil(
                K2GroupedNormFusion.apply(
                    x, weight: weight, groups: 4, eps: 1e-6, forceReference: true))
            for groups in [1, 2, 8] {
                XCTAssertNil(
                    K2GroupedNormFusion.apply(x, weight: weight, groups: groups, eps: 1e-6))
            }
            for eps: Float in [0, 1e-5, .nan] {
                XCTAssertNil(K2GroupedNormFusion.apply(x, weight: weight, groups: 4, eps: eps))
            }
            for shape in [[1, 2, 4096], [2, 1, 4096], [1, 4096], [1, 1, 2048]] {
                XCTAssertNil(
                    K2GroupedNormFusion.apply(
                        MLXArray.zeros(shape, dtype: .bfloat16), weight: weight, groups: 4,
                        eps: 1e-6))
            }
            for dtype: DType in [.float16, .float32] {
                XCTAssertNil(
                    K2GroupedNormFusion.apply(
                        x.asType(dtype), weight: weight, groups: 4, eps: 1e-6))
                XCTAssertNil(
                    K2GroupedNormFusion.apply(
                        x, weight: weight.asType(dtype), groups: 4, eps: 1e-6))
            }
            XCTAssertNil(
                K2GroupedNormFusion.apply(
                    x, weight: weight.reshaped([1, 4096]), groups: 4, eps: 1e-6))
        }
    }

    func testEvaluatedStridedAndReversedInputsDecline() throws {
        try requireFusionEnabled()
        MLXMetalTestLock.withLock {
            Stream.withNewDefaultStream(device: .gpu) {
                let base = MLXRandom.normal([1, 1, 8192], key: MLXRandom.key(97)).asType(.bfloat16)
                let weights = MLXRandom.normal([8192], key: MLXRandom.key(101)).asType(.bfloat16)
                let contiguous = MLXArray.ones([1, 1, 4096], dtype: .bfloat16)
                let weight = MLXArray.ones([4096], dtype: .bfloat16)
                eval(base, weights, contiguous, weight)
                for stride in [2, -2] {
                    let x = base[0..., 0..., .stride(by: stride)]
                    let w = weights[.stride(by: stride)]
                    eval(x, w)
                    XCTAssertNil(K2GroupedNormFusion.apply(x, weight: weight, groups: 4, eps: 1e-6))
                    XCTAssertNil(
                        K2GroupedNormFusion.apply(contiguous, weight: w, groups: 4, eps: 1e-6))
                }
            }
        }
    }

    func testLazyStridedInputRetainsReferenceValues() {
        MLXMetalTestLock.withLock {
            Stream.withNewDefaultStream(device: .gpu) {
                for stride in [2, -2] {
                    let base = MLXRandom.normal([1, 1, 8192], key: MLXRandom.key(107))
                        .asType(.bfloat16)
                    let x = base[0..., 0..., .stride(by: stride)]
                    let weight = MLXRandom.normal([4096], key: MLXRandom.key(109)).asType(.bfloat16)
                    // Do not evaluate views before admission. Depending on
                    // deferred layout metadata this may fuse or decline.
                    let actual =
                        K2GroupedNormFusion.apply(x, weight: weight, groups: 4, eps: 1e-6)
                        ?? reference(x, weight: weight)
                    expectSameBits(actual, reference(x, weight: weight))
                }
            }
        }
    }

    func testCPUStreamAndCompiledDecodeScopeDecline() {
        MLXMetalTestLock.withLock {
            let x = MLXArray.ones([1, 1, 4096], dtype: .bfloat16)
            let weight = MLXArray.ones([4096], dtype: .bfloat16)
            Stream.withNewDefaultStream(device: .cpu) {
                XCTAssertNil(K2GroupedNormFusion.apply(x, weight: weight, groups: 4, eps: 1e-6))
            }
            CompiledDecodeTrace.withActive {
                XCTAssertNil(K2GroupedNormFusion.apply(x, weight: weight, groups: 4, eps: 1e-6))
            }
        }
    }

    func testOrdinaryCPUCallRetainsNativeExpression() throws {
        try MLXMetalTestLock.withLock {
            try Stream.withNewDefaultStream(device: .cpu) {
                let norm = K2GroupedRMSNorm(dimensions: 4096, groups: 4, eps: 1e-6)
                let weight = MLXRandom.normal([4096], key: MLXRandom.key(139)).asType(.bfloat16)
                try norm.update(
                    parameters: ModuleParameters.unflattened(["weight": weight]), verify: [.all])
                let x = MLXRandom.normal([1, 1, 4096], key: MLXRandom.key(149)).asType(.bfloat16)
                expectSameBits(norm(x), reference(x, weight: norm.weight))
            }
        }
    }

    func testCompiledAndDerivativeFallbacksPreserveReference() {
        MLXMetalTestLock.withLock {
            let x = MLXRandom.normal([1, 1, 4096], key: MLXRandom.key(113)).asType(.bfloat16)
            let weight = MLXRandom.normal([4096], key: MLXRandom.key(127)).asType(.bfloat16)
            let compiled = compile { (a: MLXArray, w: MLXArray) in
                K2GroupedNormFusion.apply(a, weight: w, groups: 4, eps: 1e-6)
                    ?? self.reference(a, weight: w)
            }
            expectSameBits(compiled(x, weight), reference(x, weight: weight))
            let cotangent = MLXArray.ones(x.shape, dtype: .bfloat16)
            let native = vjp(
                { [self.reference($0[0], weight: $0[1])] }, primals: [x, weight],
                cotangents: [cotangent])
            let actual = vjp(
                {
                    [
                        K2GroupedNormFusion.apply($0[0], weight: $0[1], groups: 4, eps: 1e-6)
                            ?? self.reference($0[0], weight: $0[1])
                    ]
                },
                primals: [x, weight], cotangents: [cotangent])
            for (a, e) in zip(actual.1, native.1) { expectSameBits(a, e) }
        }
    }

    func testOrdinaryNormCallReachesCandidateGraph() throws {
        try requireFusionEnabled()
        try MLXMetalTestLock.withLock {
            try Stream.withNewDefaultStream(device: .gpu) {
                let norm = K2GroupedRMSNorm(dimensions: 4096, groups: 4, eps: 1e-6)
                let weight = MLXRandom.normal([4096], key: MLXRandom.key(131)).asType(.bfloat16)
                try norm.update(
                    parameters: ModuleParameters.unflattened(["weight": weight]), verify: [.all])
                let x = MLXRandom.normal([1, 1, 4096], key: MLXRandom.key(137)).asType(.bfloat16)
                eval(x, norm.weight)
                let direct = try XCTUnwrap(
                    K2GroupedNormFusion.apply(x, weight: norm.weight, groups: 4, eps: 1e-6))
                let ordinary = norm(x)
                let native = reference(x, weight: norm.weight)
                func nodes(_ value: MLXArray) -> Int32 {
                    var count: Int32 = -1
                    var casts: Int32 = -1
                    XCTAssertEqual(vmlx_graph_stats(value.ctx.ctx, &count, &casts), 0)
                    return count
                }
                let directNodes = nodes(direct)
                XCTAssertGreaterThan(directNodes, 0)
                XCTAssertEqual(nodes(ordinary), directNodes)
                XCTAssertGreaterThan(
                    nodes(native), directNodes, "Probe must distinguish the reference graph")
                expectSameBits(ordinary, native)
            }
        }
    }
}
