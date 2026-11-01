import MLX
import MLXLMCommon
import MLXNN
import XCTest

#if os(macOS)
    /// Fails while materializing a lazy graph, without submitting an invalid GPU
    /// memory access or triggering an actual GPU watchdog/driver reset.
    private final class LazyFailureSSMModel: Module, LanguageModel, @unchecked Sendable {
        private(set) var forwardCalls = 0

        func newCache(parameters: GenerateParameters?) -> [KVCache] { [MambaCache()] }

        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            .tokens(input.text)
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            forwardCalls += 1
            let kernel = MLXFast.metalKernel(
                name: "ssm_materialization_failure",
                inputNames: ["input"], outputNames: ["output"],
                source: "output[0] = missing_ssm_materialization_fixture_symbol;")
            let state = kernel(
                [MLXArray([Float(1)])], grid: (1, 1, 1), threadGroup: (1, 1, 1),
                outputShapes: [[1, 1, 1]], outputDTypes: [.float32])[0]
            if let mamba = cache?.first as? MambaCache {
                mamba.state = [state, MLXArray.zeros([1, 1, 1])]
                mamba.offset += inputs.size
            }
            return MLXArray.zeros([1, max(inputs.size, 1), 1])
        }
    }

    final class SSMReDeriveEvaluationFailureTests: XCTestCase {
        func testLazyEvaluationFailureThrowsInsteadOfReturningAnInvalidCompanion() {
            let model = LazyFailureSSMModel()
            var threw = false
            var capturedMessage = ""
            // The outer scope makes the previous implementation safe to exercise:
            // it records its otherwise-unhandled error, but must not be mistaken
            // for the rederive function itself throwing at the failing evaluation.
            try? withError { scope in
                do {
                    _ = try reDeriveSSMStates(model: model, tokens: [1, 2, 3], prefillStepSize: 1)
                } catch {
                    threw = true
                    capturedMessage = String(describing: error)
                }
                if capturedMessage.isEmpty, let error = scope.firstError {
                    capturedMessage = String(describing: error)
                }
            }
            XCTAssertTrue(
                capturedMessage.contains("missing_ssm_materialization_fixture_symbol"),
                "Must exercise the intended lazy compile error: \(capturedMessage)")
            XCTAssertTrue(threw, "The rederive function must throw before returning invalid state")
            XCTAssertEqual(model.forwardCalls, 1)
        }

        func testLazyEvaluationFailureStopsBeforeReplayingMoreChunks() {
            let model = LazyFailureSSMModel()
            var threw = false
            var capturedMessage = ""
            try? withError { scope in
                do {
                    _ = try reDeriveSSMStatesAtBoundaries(
                        model: model, tokens: [1, 2, 3],
                        boundaries: [2, 3], prefillStepSize: 1)
                } catch {
                    threw = true
                    capturedMessage = String(describing: error)
                }
                if capturedMessage.isEmpty, let error = scope.firstError {
                    capturedMessage = String(describing: error)
                }
            }
            XCTAssertTrue(
                capturedMessage.contains("missing_ssm_materialization_fixture_symbol"),
                "Must exercise the intended lazy compile error: \(capturedMessage)")
            XCTAssertTrue(threw, "A materialization error must abort the replay")
            XCTAssertEqual(
                model.forwardCalls, 1, "Do not keep submitting work after a failed evaluation")
        }
    }
#endif
