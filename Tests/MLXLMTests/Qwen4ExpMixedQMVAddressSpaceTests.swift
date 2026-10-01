import MLX
import MLXLMCommon
import Testing

/// Generated, bounded binary-exact fixtures; no model or bundle is loaded.
/// Metadata counts straddle MLX custom Metal's constant/device threshold8.
@Suite("Qwen mixed affine QMV metadata pointer deduction", .serialized)
struct Qwen4ExpMixedQMVAddressSpaceTests {
    private struct Fixture {
        let input: MLXArray
        let weight: MLXArray
        let scales: MLXArray
        let biases: MLXArray
        let packed: [UInt32]
        let inputIntegers: [Int]
        let scaleIntegers: [Int]
        let biasIntegers: [Int]
    }

    private func fixture(k: Int, n: Int, bits: Int, group: Int = 64,
                         experts: Int = 1, bank: Bool = false,
                         inputType: DType = .bfloat16) -> Fixture {
        let packedColumns = k * bits / 32
        let groups = k / group
        var packed: [UInt32] = []
        packed.reserveCapacity(experts * n * packedColumns)
        for index in 0 ..< (experts * n * packedColumns) {
            // Every UInt32 is a valid packed affine payload. Include high/low
            // codes and nontrivial expert/row offsets without quantizer math.
            let value = UInt32(truncatingIfNeeded: UInt64(index + 1) &* 0x9e37_79b9)
            packed.append(value ^ 0xa5c3_6e19)
        }
        var inputIntegers: [Int] = []
        var inputValues: [Float] = []
        for index in 0 ..< k {
            let integer = (index * 3 + 1) % 5 - 2
            inputIntegers.append(integer)
            inputValues.append(Float(integer) / 512)
        }
        var scaleIntegers: [Int] = []
        var biasIntegers: [Int] = []
        var scaleValues: [Float] = []
        var biasValues: [Float] = []
        for index in 0 ..< (experts * n * groups) {
            let scale = 1 + index % 2
            let bias = index % 3 - 1
            scaleIntegers.append(scale)
            biasIntegers.append(bias)
            scaleValues.append(Float(scale) / 1024)
            biasValues.append(Float(bias) / 1024)
        }
        let isBank = bank || experts > 1
        let weightShape = isBank ? [experts, n, packedColumns] : [n, packedColumns]
        let metadataShape = isBank ? [experts, n, groups] : [n, groups]
        return Fixture(
            input: MLXArray(inputValues, [1, 1, k]).asType(inputType),
            weight: MLXArray(packed, weightShape),
            scales: MLXArray(scaleValues, metadataShape).asType(.float16),
            biases: MLXArray(biasValues, metadataShape).asType(.float16),
            packed: packed, inputIntegers: inputIntegers,
            scaleIntegers: scaleIntegers, biasIntegers: biasIntegers)
    }

    /// Independent host unpack/reduction for the helper's admitted q4/q8.
    /// Integer products/sums are bounded below2^24 at these K, so reduction
    /// order/FMA cannot justify numerical slack for these dyadic inputs.
    private func hostReference(_ fixture: Fixture, k: Int, n: Int, bits: Int,
                               expert: Int = 0) -> MLXArray {
        let pack = 32 / bits
        let columns = k / pack
        let groups = k / 64
        let mask = UInt32((1 << bits) - 1)
        var output: [Float] = []
        for row in 0 ..< n {
            var sum = 0
            for index in 0 ..< k {
                let wordIndex = (expert * n + row) * columns + index / pack
                let shift = UInt32((index % pack) * bits)
                let code = Int((fixture.packed[wordIndex] >> shift) & mask)
                let metadataIndex = (expert * n + row) * groups + index / 64
                let affineInteger = code * fixture.scaleIntegers[metadataIndex]
                    + fixture.biasIntegers[metadataIndex]
                sum += fixture.inputIntegers[index] * affineInteger
            }
            output.append(Float(sum) / 524_288)
        }
        return MLXArray(output, [1, 1, n]).asType(fixture.input.dtype)
    }

    private func expectExact(_ actual: MLXArray, _ expected: MLXArray, _ context: String) {
        #expect(actual.shape == expected.shape, "\(context) shape")
        #expect(actual.dtype == expected.dtype, "\(context) dtype")
        #expect(isFinite(actual).all().item(Bool.self), "\(context) finite")
        let actualBits = actual.asType(.float32).asArray(Float.self).map(\.bitPattern)
        let expectedBits = expected.asType(.float32).asArray(Float.self).map(\.bitPattern)
        #expect(actualBits == expectedBits, "\(context) strict bit equality")
    }

    @Test("dense q4/q8 compile with constant and device metadata and preserve native arithmetic")
    func denseConstantAndDevicePointers() {
        MLXMetalTestLock.withLock {
            for bits in [4, 8] {
                for (k, n) in [(256, 1), (256, 2), (256, 3), (256, 7),
                               (256, 9), (512, 1), (512, 8),
                               (2560, 48), (6144, 8)] {
                    let values = fixture(k: k, n: n, bits: bits)
                    #expect(Qwen4ExpBF16Affine.supports(
                        input: values.input, weight: values.weight, scales: values.scales,
                        biases: values.biases, groupSize: 64, bits: bits, mode: .affine))
                    let actual = Qwen4ExpBF16Affine.dense(
                        values.input, values.weight, scales: values.scales, biases: values.biases,
                        groupSize: 64, bits: bits, mode: .affine)
                    let native = quantizedMM(
                        values.input, values.weight, scales: values.scales, biases: values.biases,
                        transpose: true, groupSize: 64, bits: bits, mode: .affine)
                        .asType(.bfloat16)
                    let host = hostReference(values, k: k, n: n, bits: bits)
                    MLX.eval(actual, native, host)
                    let context = "q\(bits) K\(k) N\(n) metadata_count\(values.scales.size)"
                    expectExact(actual, native, context + " native")
                    expectExact(actual, host, context + " independent host")
                    #expect(actual.dtype == .bfloat16)
                }
            }
            print("[mixed-qmv-address-space] dense_cases=18 strict_bits=1 tokens_per_second=NA reason=no_generation")
        }
    }

    @Test("gathered q4/q8 deduce metadata pointee address space and preserve expert offsets")
    func gatheredConstantAndDevicePointers() throws {
        try MLXMetalTestLock.withLock {
            for bits in [4, 8] {
                for (k, n, experts) in [(256, 1, 1), (256, 1, 2), (256, 3, 2),
                                       (512, 8, 3), (2560, 48, 3), (6144, 8, 3)] {
                    let values = fixture(k: k, n: n, bits: bits, experts: experts, bank: true)
                    let routes = [experts - 1, 0, experts - 1]
                    let indices = MLXArray(routes.map(UInt32.init), [1, routes.count])
                    let actual = try #require(Qwen4ExpBF16Affine.gathered(
                        values.input, values.weight, scales: values.scales, biases: values.biases,
                        indices: indices, groupSize: 64, bits: bits, mode: .affine))
                    let native = gatherQuantizedMM(
                        values.input, values.weight, scales: values.scales, biases: values.biases,
                        rhsIndices: indices, transpose: true, groupSize: 64, bits: bits, mode: .affine)
                        .asType(.bfloat16)
                    var references: [MLXArray] = []
                    for expert in routes {
                        references.append(hostReference(values, k: k, n: n, bits: bits, expert: expert))
                    }
                    let host = stacked(references, axis: 1)
                    MLX.eval(actual, native, host)
                    let context = "gather q\(bits) K\(k) N\(n) E\(experts) metadata_count\(values.scales.size)"
                    expectExact(actual, native, context + " native")
                    expectExact(actual, host, context + " independent host")
                    #expect(actual.shape == [1, routes.count, 1, n])
                }
            }
            print("[mixed-qmv-address-space] gathered_cases=12 strict_bits=1 tokens_per_second=NA reason=no_generation")
        }
    }

    @Test("B1 rows q2/q3/q4/q6/q8 retain wrapper dtype and stock fallback precision")
    func allQuantWidthsAndActivationDTypes() {
        MLXMetalTestLock.withLock {
            for bits in [2, 3, 4, 6, 8] {
                for inputType in [DType.bfloat16, .float16] {
                    for group in [32, 64] {
                        let values = fixture(k: 512, n: 8, bits: bits, group: group,
                                             inputType: inputType)
                        let admitted = inputType == .bfloat16 && group == 64 && (bits == 4 || bits == 8)
                        #expect(Qwen4ExpBF16Affine.supports(
                            input: values.input, weight: values.weight, scales: values.scales,
                            biases: values.biases, groupSize: group, bits: bits, mode: .affine) == admitted)
                        let actual = Qwen4ExpBF16Affine.dense(
                            values.input, values.weight, scales: values.scales, biases: values.biases,
                            groupSize: group, bits: bits, mode: .affine)
                        let rawNative = quantizedMM(
                            values.input, values.weight, scales: values.scales, biases: values.biases,
                            transpose: true, groupSize: group, bits: bits, mode: .affine)
                        let expected = rawNative.asType(inputType)
                        MLX.eval(actual, rawNative, expected)
                        expectExact(actual, expected, "q\(bits) \(inputType) group\(group) B1M1")
                        #expect(actual.shape == [1, 1, 8])
                        #expect(actual.dtype == inputType)
                        let expectedRawType: DType = inputType == .float16 ? .float16
                            : (admitted ? .bfloat16 : .float32)
                        #expect(rawNative.dtype == expectedRawType)
                    }
                }
            }
            print("[mixed-qmv-address-space] all_quant_cases=20 strict_bits=1 tokens_per_second=NA reason=no_generation")
        }
    }
}
