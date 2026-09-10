import MLX
import MLXLMCommon
import MLXNN
import Testing

@Suite(.serialized)
struct Qwen4ExpMixedQuantDispatchTests {
    @Test("each projection selects its own quantization contract, including fallback")
    func mixedProjectionContracts() throws {
        try MLXMetalTestLock.withLock {
            for bits in [2, 3, 4, 5, 6, 8] {
                for groupSize in [32, 64, 128] {
                    let input = ((MLXArray(0..<1024, [2, 512]).asType(.float32) % 23) / 23 - 0.5)
                        .asType(.bfloat16)
                    let source = ((MLXArray(0..<3584, [7, 512]).asType(.float32) % 31) / 31 - 0.5)
                        .asType(.float16)
                    let (weight, scales, biases) = quantized(
                        source, groupSize: groupSize, bits: bits, mode: .affine)
                    let native = Qwen4ExpBF16Affine.supports(
                        input: input, weight: weight, scales: scales, biases: biases,
                        groupSize: groupSize, bits: bits, mode: .affine)
                    #expect(native == (groupSize == 64 && (bits == 4 || bits == 8)))
                    let actual = Qwen4ExpBF16Affine.dense(
                        input, weight, scales: scales, biases: biases,
                        groupSize: groupSize, bits: bits, mode: .affine)
                    let reference = quantizedMM(
                        input, weight, scales: scales, biases: biases, transpose: true,
                        groupSize: groupSize, bits: bits, mode: .affine).asType(input.dtype)
                    MLX.eval(actual, reference)
                    #expect(actual.shape == [2, 7])
                    #expect(actual.dtype == input.dtype)
                    #expect(MLX.isFinite(actual).all().item(Bool.self))
                    let error = abs(actual.asType(.float32) - reference.asType(.float32)).max().item(Float.self)
                    // Existing native-kernel BF16 tolerance; fallback must be exact.
                    #expect(error <= (native ? 0.015625 : 0), "bits=\(bits) group=\(groupSize) error=\(error)")
                    let gathered = Qwen4ExpBF16Affine.gathered(
                        input, weight.expandedDimensions(axis: 0), scales: scales.expandedDimensions(axis: 0),
                        biases: biases?.expandedDimensions(axis: 0), indices: MLXArray([UInt32(0), 0], [2, 1]),
                        groupSize: groupSize, bits: bits, mode: .affine)
                    #expect((gathered != nil) == native)
                    if let gathered {
                        MLX.eval(gathered)
                        #expect(abs(gathered.reshaped([2, 7]) - actual).max().item(Float.self) == 0)
                    }
                }
            }
        }
    }
}
