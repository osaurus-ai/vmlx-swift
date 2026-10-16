import MLX
@testable import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

final class NemotronStagedMambaTests: XCTestCase {
    func testFusedDecodeConvolutionPreservesActivationRounding() throws {
        let channels = 256
        for dtype: DType in [.bfloat16, .float16, .float32] {
            let input = sin(MLXArray(0 ..< 2 * channels).asType(.float32) * 0.193)
                .asType(dtype).reshaped([2, 1, channels])
            let state = cos(MLXArray(0 ..< 2 * 3 * channels).asType(.float32) * 0.071)
                .asType(dtype).reshaped([2, 3, channels])
            let weight = sin(MLXArray(0 ..< channels * 4).asType(.float32) * 0.137)
                .asType(dtype).reshaped([channels, 4, 1])
            let bias = (cos(MLXArray(0 ..< channels).asType(.float32)) * 0.7).asType(dtype)
            let padded = concatenated([state, input], axis: 1)
            let expected = silu(conv1d(padded, weight, groups: channels) + bias)
            let (actual, next) = try XCTUnwrap(nemotronHMambaDepthwiseDecodeConv(
                input: input, state: state, weight: weight, bias: bias,
                channels: channels, kernelSize: 4))
            XCTAssertEqual(abs(actual - expected).max().item(Float.self), 0, "\(dtype)")
            XCTAssertEqual(abs(next - padded[0..., 1..., 0...]).max().item(Float.self), 0)
        }
    }

    func testGatedNormMatchesPythonFusedBFloat16Reference() {
        let norm = NemotronHRMSNormGated(dimensions: 32, eps: 1e-5, groupSize: 8)
        norm.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray.ones([32], dtype: .bfloat16),
        ]))
        let indices = MLXArray(0 ..< 32).asType(.float32)
        let x = (indices * 0.071 - 1).asType(.bfloat16).reshaped([1, 1, 32])
        let gate = (indices * 0.193 - 2).asType(.bfloat16).reshaped([1, 1, 32])
        // Current mlx-lm MambaRMSNormGated, computed independently in Python.
        let expected: [Float] = [
            1.203125, 1.1953125, 1.15625, 1.0859375, 0.99609375, 0.87890625, 0.73046875, 0.5625,
            2.0625, 1.109375, 0.265625, -0.38671875, -0.734375, -0.66015625, -0.07666015625, 1.1171875,
            0.140625, 0.263671875, 0.423828125, 0.62109375, 0.859375, 1.125, 1.453125, 1.796875,
            0.52734375, 0.6328125, 0.74609375, 0.87109375, 1, 1.140625, 1.28125, 1.4375,
        ]
        XCTAssertEqual(norm(x, gate: gate).asType(.float32).asArray(Float.self), expected)
    }

    func testAcceptedPrefixRestoresRecurrentStateAndNextToken() throws {
        let config = NemotronHConfiguration(
            vocabSize: 32, hiddenSize: 32, numHiddenLayers: 1,
            numAttentionHeads: 4, numKeyValueHeads: 2,
            mambaNumHeads: 4, mambaHeadDim: 8, ssmStateSize: 128,
            convKernel: 4, nGroups: 2, intermediateSize: 64,
            moeIntermediateSize: 32, moeSharedExpertIntermediateSize: 32,
            nRoutedExperts: 4, numExpertsPerTok: 2, hybridOverridePattern: "M",
            layerNormEpsilon: 1e-5, nGroup: 1, topkGroup: 1)
        let mixer = NemotronHMamba2Mixer(config)
        // Exactly representable inputs/projection isolate rollback from the
        // TF32 GEMM versus float32 GEMV precision difference across row counts.
        let projection = (MLXArray(0 ..< mixer.inProj.weight.size) % 5 - 2)
            .asType(.float32) * 0.03125
        mixer.inProj.update(parameters: ModuleParameters.unflattened([
            "weight": projection.reshaped(mixer.inProj.weight.shape),
        ]))
        let input = ((MLXArray(0 ..< 10 * 32) % 7 - 3).asType(.float32) * 0.125)
            .reshaped([1, 10, 32])
        func forward(_ x: MLXArray, _ cache: MambaCache) -> MLXArray {
            mixer(x, attentionMask: .none, ssmMask: nil, cache: cache)
        }
        for accepted in 1 ... 4 {
            let staged = MambaCache()
            let reference = MambaCache()
            eval(forward(input[0..., ..<5], staged), forward(input[0..., ..<5], reference))
            let committed = staged.state.map { $0 * 1 }
            eval(committed)
            let verified = NativeMTPVerifierStatePolicy.withVerifierMode("input_capture_staged") {
                forward(input[0..., 5 ..< 9], staged)
            }
            eval(verified)
            XCTAssertEqual(staged.offset, 5)
            for (actual, before) in zip(staged.state, committed) {
                XCTAssertEqual(abs(actual - before).max().item(Float.self), 0)
            }
            XCTAssertFalse(mixer.commitVerifyStaged(cache: staged, acceptedInputs: 0, blockLength: 4))
            XCTAssertEqual(staged.offset, 5)
            XCTAssertTrue(mixer.commitVerifyStaged(cache: staged, acceptedInputs: accepted, blockLength: 4))
            staged.clearVerifyStaging()
            for token in 5 ..< 5 + accepted {
                eval(forward(input[0..., token ..< token + 1], reference))
            }
            XCTAssertEqual(staged.offset, reference.offset)
            XCTAssertEqual(staged[1]?.dtype, .float32)
            for (actual, expected) in zip(staged.state, reference.state) {
                XCTAssertLessThan(abs(actual - expected).max().item(Float.self), 1e-4,
                    "accepted=\(accepted), shape=\(actual.shape)")
            }
            let next = forward(input[0..., 9 ..< 10], staged)
            let expected = forward(input[0..., 9 ..< 10], reference)
            XCTAssertLessThan(abs(next - expected).max().item(Float.self), 1e-4,
                "Next token after accepted=\(accepted)")
        }
    }
}
