import MLX
import MLXLMCommon
import MLXNN
import Testing

@Suite("Qwen4Exp native BF16 affine kernels", .serialized)
struct Qwen4ExpBF16AffineTests {
    @Test("Qwen4Exp BF16 JANG weights stay owned, not file-backed, even when mmap is requested")
    func ownedWeightPolicyDeclinesMmap() {
        let facts = LoadBundleFacts(
            totalSafetensorsBytes: 96 << 30,
            isRouted: true,
            physicalMemory: 128 << 30,
            modelType: "qwen4_exp",
            jangFormat: "jang_v2",
            declaredComputeDType: "bfloat16",
            numRoutedExperts: 512,
            topK: 10)

        #expect(facts.isQwen4ExpBF16JANG)
        #expect(!facts.requiresResidentSafetensors)
        // File-backed model-weight ownership is retired (1747b2bd): a request for mmap is declined.
        #expect(!facts.resolveMmapSafetensors(requested: true))
    }

    @Test("Qwen4Exp classification does not infer JANG or BF16 from family name")
    func computePolicyFailsClosedWithoutConfigEvidence() {
        let missingJang = LoadBundleFacts(
            totalSafetensorsBytes: 96 << 30,
            isRouted: true,
            physicalMemory: 128 << 30,
            modelType: "qwen4_exp",
            declaredComputeDType: "bfloat16",
            numRoutedExperts: 512,
            topK: 10)
        let wrongDType = LoadBundleFacts(
            totalSafetensorsBytes: 96 << 30,
            isRouted: true,
            physicalMemory: 128 << 30,
            modelType: "qwen4_exp",
            jangFormat: "jang_v2",
            declaredComputeDType: "float16",
            numRoutedExperts: 512,
            topK: 10)

        #expect(!missingJang.isQwen4ExpBF16JANG)
        #expect(!wrongDType.isQwen4ExpBF16JANG)
    }

    @Test("MLX core dense QMV keeps BF16 compute with F16 affine metadata")
    func coreMixedDenseParity() {
        let input = (MLXArray(0 ..< 512, [1, 1, 512]).asType(.float32) / 511)
            .asType(.bfloat16)
        let denseWeight = (MLXArray(0 ..< 4096, [8, 512]).asType(.float32) / 4095) - 0.5
        let (weight, scales, biases) = quantized(
            denseWeight.asType(.float16), groupSize: 64, bits: 4, mode: .affine)
        let actual = quantizedMM(
            input, weight, scales: scales, biases: biases,
            transpose: true, groupSize: 64, bits: 4, mode: .affine)
        let expected = quantizedMM(
            input.asType(.float32), weight,
            scales: scales.asType(.float32), biases: biases?.asType(.float32),
            transpose: true, groupSize: 64, bits: 4, mode: .affine)
            .asType(.bfloat16)
        MLX.eval(actual, expected)
        #expect(actual.dtype == .bfloat16)
        let delta = abs(actual.asType(.float32) - expected.asType(.float32))
        let relativeError = (delta / maximum(abs(expected.asType(.float32)), 1)).max()
            .item(Float.self)
        // The production QMV and the F32 reference reduce lanes in different
        // orders. Require agreement within one BF16 quantum at unit scale.
        #expect(relativeError <= 0.0078125)
    }

    @Test("MLX core gathered QMV keeps BF16 compute with F16 affine metadata")
    func coreMixedGatheredParity() {
        let input = (MLXArray(0 ..< 1024, [2, 1, 512]).asType(.float32) / 1023)
            .asType(.bfloat16)
        let denseWeight = (MLXArray(0 ..< 12288, [3, 8, 512]).asType(.float32) / 12287) - 0.5
        let (weight, scales, biases) = quantized(
            denseWeight.asType(.float16), groupSize: 64, bits: 4, mode: .affine)
        let indices = MLXArray([0, 2, 1, 0], [2, 2]).asType(.uint32)
        let actual = gatherQuantizedMM(
            input, weight, scales: scales, biases: biases,
            rhsIndices: indices, transpose: true,
            groupSize: 64, bits: 4, mode: .affine)
        let expected = gatherQuantizedMM(
            input.asType(.float32), weight,
            scales: scales.asType(.float32), biases: biases?.asType(.float32),
            rhsIndices: indices, transpose: true,
            groupSize: 64, bits: 4, mode: .affine)
            .asType(.bfloat16)
        MLX.eval(actual, expected)
        #expect(actual.dtype == .bfloat16)
        #expect(actual.shape == [2, 2, 1, 8])
        #expect(MLX.isFinite(actual).all().item(Bool.self))
        let delta = abs(actual.asType(.float32) - expected.asType(.float32))
        let relativeError = (delta / maximum(abs(expected.asType(.float32)), 1)).max()
            .item(Float.self)
        #expect(relativeError <= 0.0078125)
    }

    @Test("dense 4-bit and 8-bit projections keep BF16 I/O with F16 metadata")
    func denseParity() {
        for bits in [4, 8] {
            let input = (MLXArray(0 ..< 128, [2, 64]).asType(.float32) / 127)
                .asType(.bfloat16)
            let denseWeight = (MLXArray(0 ..< 320, [5, 64]).asType(.float32) / 319) - 0.5
            let (weight, scales, biases) = quantized(
                denseWeight.asType(.float16), groupSize: 64, bits: bits, mode: .affine)
            let expected = quantizedMM(
                input.asType(.float32), weight,
                scales: scales.asType(.float32), biases: biases?.asType(.float32),
                transpose: true, groupSize: 64, bits: bits, mode: .affine)
                .asType(.bfloat16)
            let actual = Qwen4ExpBF16Affine.dense(
                input, weight, scales: scales, biases: biases,
                groupSize: 64, bits: bits, mode: .affine)
            MLX.eval(expected, actual)
            #expect(actual.dtype == .bfloat16)
            let error = abs(actual.asType(.float32) - expected.asType(.float32)).max()
                .item(Float.self)
            #expect(error <= 0.015625)
        }
    }

    @Test("dense affine output is exactly invariant to verifier row batching")
    func denseRowBatchingIsExact() {
        let input = (MLXArray(0 ..< 256, [4, 64]).asType(.float32) / 255)
            .asType(.bfloat16)
        let denseWeight =
            (MLXArray(0 ..< 448, [7, 64]).asType(.float32) / 447) - 0.5
        let (weight, scales, biases) = quantized(
            denseWeight.asType(.float16), groupSize: 64, bits: 4, mode: .affine)

        let batched = Qwen4ExpBF16Affine.dense(
            input, weight, scales: scales, biases: biases,
            groupSize: 64, bits: 4, mode: .affine)
        let rowWise = concatenated(
            (0 ..< 4).map { row in
                Qwen4ExpBF16Affine.dense(
                    input[row ..< (row + 1), 0...], weight,
                    scales: scales, biases: biases,
                    groupSize: 64, bits: 4, mode: .affine)
            },
            axis: 0)
        MLX.eval(batched, rowWise)

        #expect(batched.shape == rowWise.shape)
        #expect(abs(batched - rowWise).max().item(Float.self) == 0)
    }

    @Test("gathered routed projection keeps BF16 I/O")
    func gatheredParity() throws {
        let input = (MLXArray(0 ..< 128, [2, 64]).asType(.float32) / 127)
            .asType(.bfloat16)
        let denseWeight = (MLXArray(0 ..< 768, [3, 4, 64]).asType(.float32) / 767) - 0.5
        let (weight, scales, biases) = quantized(
            denseWeight.asType(.float16), groupSize: 64, bits: 4, mode: .affine)
        let indices = MLXArray([0, 2, 1, 0], [2, 2]).asType(.uint32)
        let actual = try #require(Qwen4ExpBF16Affine.gathered(
            input, weight, scales: scales, biases: biases, indices: indices,
            groupSize: 64, bits: 4, mode: .affine))
        MLX.eval(actual)
        #expect(actual.dtype == .bfloat16)
        #expect(actual.shape == [2, 2, 1, 4])
    }
}
