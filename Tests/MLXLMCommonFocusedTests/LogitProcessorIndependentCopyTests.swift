// Copyright © 2026 osaurus-eval contributors

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Isolates the copy mechanism from the speculative-decoding logic that consumes it.
@Suite("LogitProcessor independent copy")
struct LogitProcessorIndependentCopyTests {

    private func makeProcessor() -> LogitProcessor {
        GenerateParameters(
            maxTokens: 8, temperature: 0,
            repetitionPenalty: nil, presencePenalty: 0.5, frequencyPenalty: 0.2
        ).processor()!
    }

    @Test("recording into a copy does not reach the original")
    func copyDoesNotShareCounts() throws {
        let vocab = 16
        var original = makeProcessor()
        // Force the counts vector to exist: it allocates on first sight of logits.
        _ = original.process(logits: MLXArray.zeros([1, vocab], type: Float32.self))

        var copy = original.independentCopy()
        for _ in 0 ..< 4 { copy.didSample(token: MLXArray([Int32(3)])) }

        // The original never sampled anything, so its penalty must still be a no-op.
        let flat = MLXArray.zeros([1, vocab], type: Float32.self)
        let afterOriginal = original.process(logits: flat)
        eval(afterOriginal)
        let values = afterOriginal.asArray(Float.self)
        #expect(
            values.allSatisfy { $0 == 0 },
            "the original was penalised by tokens only the copy sampled: \(values)")
    }

    @Test("the copy still works — it is a copy, not a stub")
    func copyIsFunctional() throws {
        let vocab = 16
        var copy = makeProcessor().independentCopy()
        _ = copy.process(logits: MLXArray.zeros([1, vocab], type: Float32.self))
        for _ in 0 ..< 4 { copy.didSample(token: MLXArray([Int32(3)])) }
        let after = copy.process(logits: MLXArray.zeros([1, vocab], type: Float32.self))
        eval(after)
        let values = after.asArray(Float.self)
        #expect(values[3] != 0, "the copy must still accumulate its own penalty")
    }
}
