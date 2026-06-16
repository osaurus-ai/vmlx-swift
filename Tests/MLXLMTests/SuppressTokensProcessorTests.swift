import MLX
import MLXLMCommon
import XCTest

final class SuppressTokensProcessorTests: XCTestCase {
    func testGenerationConfigSuppressTokensReachGenerateParameters() {
        let config = GenerationConfigFile(
            temperature: 1.0,
            topK: 64,
            doSample: true,
            suppressTokens: [258883, 258882])

        let params = GenerateParameters(
            generationConfig: config,
            fallback: GenerateParameters(suppressTokens: [7]))

        XCTAssertEqual(params.suppressTokens, [258883, 258882])
        XCTAssertNotNil(params.processor())
    }

    func testSuppressTokensProcessorMasksConfiguredLogits() {
        let params = GenerateParameters(suppressTokens: [1, 3, 99])
        var processor = params.processor()
        XCTAssertNotNil(processor)

        let logits = MLXArray([0.0 as Float, 1.0, 2.0, 3.0])[.newAxis, .ellipsis]
        let processed = processor!.process(logits: logits)[0].asArray(Float.self)

        XCTAssertEqual(processed[0], 0.0)
        XCTAssertTrue(processed[1].isInfinite && processed[1].sign == .minus)
        XCTAssertEqual(processed[2], 2.0)
        XCTAssertTrue(processed[3].isInfinite && processed[3].sign == .minus)
    }

    func testSuppressTokensComposeWithPenaltyProcessor() {
        let params = GenerateParameters(
            repetitionPenalty: 1.1,
            repetitionContextSize: 20,
            suppressTokens: [2])
        var processor = params.processor()
        XCTAssertNotNil(processor)

        processor!.prompt(MLXArray([0, 1, 2, 3]))
        let logits = MLXArray([0.0 as Float, 1.0, 2.0, 3.0])[.newAxis, .ellipsis]
        let processed = processor!.process(logits: logits)[0].asArray(Float.self)

        XCTAssertTrue(processed[2].isInfinite && processed[2].sign == .minus)
    }
}
