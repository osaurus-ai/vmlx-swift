import MLX
import MLXLMCommon
import MLXNN
import MLXRandom
import Testing

@testable import MLXVLM

@Suite("Flash shared-expert compiled metadata", .serialized)
struct Qwen4ExpSharedExpertCompileTests {
    private struct Fixture {
        let bits: Int
        let groupSize: Int
        let arguments: [MLXArray]

        init(
            bits: Int = 8, groupSize: Int = 64, dtype: DType = .float16,
            hidden: Int = 256, intermediate: Int = 128
        ) {
            self.bits = bits
            self.groupSize = groupSize
            // Bound construction, including the real-shape single shared expert.
            // Default layout cases stay under 100k parameters; the actual-shape
            // case stays under 5M, with no routed bank or complete model.
            precondition(hidden > 0 && hidden <= 2560)
            precondition(intermediate > 0 && intermediate <= 640)
            precondition(3 * hidden * intermediate < 5_000_000)
            func source(_ shape: [Int], seed: UInt64, dtype: DType) -> MLXArray {
                MLXRandom.uniform(
                    low: -0.1, high: 0.1, shape,
                    key: MLXRandom.key(seed)
                ).asType(dtype)
            }
            func projection(_ output: Int, _ input: Int, seed: UInt64) -> [MLXArray] {
                let (w, s, b) = quantized(
                    source([output, input], seed: seed, dtype: dtype),
                    groupSize: groupSize, bits: bits, mode: .affine)
                return [w, s, b!]
            }
            arguments =
                [source([1, 1, hidden], seed: 1201, dtype: .bfloat16)]
                + projection(intermediate, hidden, seed: 1202)
                + projection(intermediate, hidden, seed: 1203)
                + projection(hidden, intermediate, seed: 1204)
                + [source([1, hidden], seed: 1205, dtype: .bfloat16)]
        }

        func compiled(_ replacement: [MLXArray]? = nil) -> MLXArray? {
            let a = replacement ?? arguments
            return Qwen4ExpCompiledMoE.sharedExpert(
                a[0], gateWeight: a[1], gateScales: a[2], gateBiases: a[3],
                upWeight: a[4], upScales: a[5], upBiases: a[6],
                downWeight: a[7], downScales: a[8], downBiases: a[9],
                sharedGateWeight: a[10], groupSize: groupSize, bits: bits, mode: .affine)
        }

        func eager(_ replacement: [MLXArray]? = nil) -> MLXArray {
            let a = replacement ?? arguments
            let gate = quantizedMM(
                a[0], a[1], scales: a[2], biases: a[3],
                transpose: true, groupSize: groupSize, bits: bits, mode: .affine)
            let up = quantizedMM(
                a[0], a[4], scales: a[5], biases: a[6],
                transpose: true, groupSize: groupSize, bits: bits, mode: .affine)
            let down = quantizedMM(
                silu(gate) * up, a[7], scales: a[8], biases: a[9],
                transpose: true, groupSize: groupSize, bits: bits, mode: .affine)
            return sigmoid(matmul(a[0], a[10].transposed())) * down
        }
    }

    @Test("F16 affine metadata enters the existing compiled shared-expert path")
    func f16MetadataIsEligible() throws {
        try MLXMetalTestLock.withLock {
            let fixture = Fixture(hidden: 2560, intermediate: 640)
            let candidate = fixture.compiled()
            let actual = try #require(candidate)
            let expected = fixture.eager()
            eval(actual, expected)
            // Existing mixed-BF16/F16 QMV keeps BF16 only for K % 512 == 0.
            // This bundle's down projection has K=640, so the unchanged eager
            // contract returns FP32. Compiling must preserve that contract,
            // not cast the result to satisfy the model's layer-output dtype.
            #expect(expected.dtype == .float32)
            #expect(actual.dtype == expected.dtype)
            #expect(isFinite(actual).all().item(Bool.self))
            #expect(abs(actual - expected).max().item(Float.self) == 0)
        }
    }

    @Test("independent batch members retain the eager shared-expert result")
    func batchMemberParity() throws {
        try MLXMetalTestLock.withLock {
            let fixture = Fixture()
            var args = fixture.arguments
            args[0] = concatenated([args[0], -args[0]], axis: 0)
            let candidate = fixture.compiled(args)
            let actual = try #require(candidate)
            let expected = fixture.eager(args)
            eval(actual, expected)
            #expect(actual.shape == [2, 1, 256])
            #expect(actual.dtype == expected.dtype)
            #expect(abs(actual - expected).max().item(Float.self) == 0)
        }
    }

    @Test("outer decode traces do not enter a nested compiled shared-expert region")
    func outerTraceFallsBack() {
        MLXMetalTestLock.withLock {
            let fixture = Fixture()
            let candidate = CompiledDecodeTrace.withActive { fixture.compiled() }
            #expect(candidate == nil)
        }
    }

    @Test("compiled shared expert preserves each supported bit/group and metadata layout")
    func quantizationAndMetadataParity() throws {
        try MLXMetalTestLock.withLock {
            for dtype in [DType.float16, .bfloat16] {
                for bits in [2, 3, 4, 5, 6, 8] {
                    for group in [32, 64, 128] {
                        let fixture = Fixture(bits: bits, groupSize: group, dtype: dtype)
                        let candidate = fixture.compiled()
                        let actual = try #require(candidate)
                        let expected = fixture.eager()
                        eval(actual, expected)
                        #expect(actual.dtype == expected.dtype)
                        #expect(isFinite(actual).all().item(Bool.self))
                        let error = abs(actual - expected).max().item(Float.self)
                        #expect(
                            error == 0,
                            "bits=\(bits) group=\(group) metadata=\(dtype) maxAbs=\(error)")
                    }
                }
            }
        }
    }

    @Test("unsupported metadata and non-decode inputs retain the generic fallback")
    func rejectedInputs() {
        MLXMetalTestLock.withLock {
            let fixture = Fixture()
            for index in [2, 3, 5, 6, 8, 9] {
                var args = fixture.arguments
                args[index] = args[index].asType(.bfloat16)
                #expect(fixture.compiled(args) == nil, "mixed metadata at \(index)")
            }
            var fp32 = fixture.arguments
            for index in [2, 3, 5, 6, 8, 9] { fp32[index] = fp32[index].asType(.float32) }
            #expect(fixture.compiled(fp32) == nil)
            for index in [0, 10] {
                var args = fixture.arguments
                args[index] = args[index].asType(.float32)
                #expect(fixture.compiled(args) == nil)
            }
            for rows in [2, 3, 4] {
                var args = fixture.arguments
                args[0] = repeated(args[0], count: rows, axis: 1)
                #expect(fixture.compiled(args) == nil)
            }
            var flat = fixture.arguments
            flat[0] = flat[0].reshaped(1, 256)
            #expect(fixture.compiled(flat) == nil)
        }
    }

    @Test("same-shape replacement weights do not reuse a previous layer's constants")
    func replacementWeightIdentity() throws {
        try MLXMetalTestLock.withLock {
            let fixture = Fixture()
            let firstCandidate = fixture.compiled()
            let first = try #require(firstCandidate)
            eval(first)
            var args = fixture.arguments
            args[10] = -args[10]
            // Same shape/layout/cache key, distinct weight values.
            let secondCandidate = fixture.compiled(args)
            let second = try #require(secondCandidate)
            let expected = fixture.eager(args)
            eval(second, expected)
            #expect(abs(second - first).max().item(Float.self) > 0)
            #expect(abs(second - expected).max().item(Float.self) == 0)
        }
    }
}
