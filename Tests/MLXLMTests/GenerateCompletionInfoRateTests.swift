// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// A speed is a measurement, not a formality. When a phase did not measurably
/// run there is no speed to report, and the honest answer is zero — not `+inf`,
/// and not `NaN`.
///
/// Both of these are reachable in normal operation, not just in theory:
///   - a cancelled stream is constructed with `generationTokenCount: 0` and
///     `generationTime: 0` → `0/0` → `NaN`
///   - a full prefix-cache hit can round `promptTime` to zero → `n/0` → `+inf`
///
/// Neither value is filtered on its way out: it is formatted straight into the
/// stats wire hint and rendered in the chat UI.
@Suite("Generation rate reporting")
struct GenerateCompletionInfoRateTests {

    @Test("A cancelled stream reports zero, not NaN")
    func cancelledStreamIsNotNaN() {
        // Exactly what `cancelledBatchStream` / `cancelledGenerationStream` build.
        let info = GenerateCompletionInfo(
            promptTokenCount: 128,
            generationTokenCount: 0,
            promptTime: 0,
            generationTime: 0,
            stopReason: .cancelled
        )

        #expect(!info.tokensPerSecond.isNaN, "0 tokens in 0 seconds must not be 0/0")
        #expect(info.tokensPerSecond.isFinite)
        #expect(info.tokensPerSecond == 0)

        #expect(!info.promptTokensPerSecond.isNaN)
        #expect(info.promptTokensPerSecond.isFinite, "128 tokens in 0 seconds must not be +inf")
        #expect(info.promptTokensPerSecond == 0)
    }

    @Test("A zero-duration prefill reports zero, not infinity")
    func instantPrefillIsNotInfinite() {
        // A full cache hit: the prompt was never re-processed, so its "speed" is
        // not a very large number — it is undefined.
        let info = GenerateCompletionInfo(
            promptTokenCount: 4096,
            generationTokenCount: 32,
            promptTime: 0,
            generationTime: 0.5
        )
        #expect(info.promptTokensPerSecond.isFinite)
        #expect(info.promptTokensPerSecond == 0)
        // The generation phase DID run, so its rate is a real measurement.
        #expect(info.tokensPerSecond == 64)
    }

    @Test("A real generation still reports its real rate")
    func realGenerationIsUnchanged() {
        let info = GenerateCompletionInfo(
            promptTokenCount: 100,
            generationTokenCount: 50,
            promptTime: 0.25,
            generationTime: 2.0
        )
        #expect(info.tokensPerSecond == 25)
        #expect(info.promptTokensPerSecond == 400)
    }
}
