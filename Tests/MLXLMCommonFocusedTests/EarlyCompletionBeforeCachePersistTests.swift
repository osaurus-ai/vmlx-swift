// Copyright © 2026 Apple Inc.

// Regression tests for the end-of-output hang: `generateLoopTask` (the
// shared terminal path for EVERY local model) used to hold the `.info`
// completion event — the user-visible "generation finished" signal —
// behind the GPU drains, the cache snapshot/store, and the advisor
// drain, so large models visibly stalled for seconds after the last
// token rendered. The rewritten contract emits `.info` right after the
// detokenizer tail flush and the CPU-only stats finalize, while the
// STREAM END (`continuation.finish()`) still waits for cleanup — that
// is the serialization point hosts (the osaurus adapter's model lease /
// Metal gate) key on before starting the next decode.

import Foundation
import MLX
import Testing
import os

@testable import MLXLMCommon

/// Iterator whose cache store blocks for a configurable delay and records
/// when the store started/ended, so the test can prove `.info` arrived
/// while the store was still running — not merely before some later event.
private struct ClockState: Sendable {
    var storeStarted: Date?
    var storeEnded: Date?
    var statsFinalizedAt: Date?
}

private final class DelayedStoreClock: Sendable {
    let state = OSAllocatedUnfairLock(initialState: ClockState())
}

private struct DelayedStoreIterator: TokenIteratorProtocol {
    let maxTokens: Int? = nil
    var tokenCount = 0
    let promptPrefillTime: TimeInterval = 0
    let clock: DelayedStoreClock
    let storeDelay: TimeInterval
    private var emitted = 0
    private let tokens = [11, 12, 13]

    init(clock: DelayedStoreClock, storeDelay: TimeInterval) {
        self.clock = clock
        self.storeDelay = storeDelay
    }

    mutating func next() -> Int? {
        guard emitted < tokens.count else { return nil }
        defer { emitted += 1 }
        return tokens[emitted]
    }

    mutating func finalizeGenerationStats(generatedTokenIds: [Int]) {
        clock.state.withLock { $0.statsFinalizedAt = Date() }
    }

    mutating func storeCacheAfterGeneration(
        generatedTokenIds: [Int], includeGeneratedBoundary: Bool
    ) {
        clock.state.withLock { $0.storeStarted = Date() }
        Thread.sleep(forTimeInterval: storeDelay)
        clock.state.withLock { $0.storeEnded = Date() }
    }
}

private struct TailTestTokenizer: Tokenizer {
    let vocabularySize = 64
    let eosTokenId: Int? = 60
    let unknownTokenId: Int? = 61
    let bosToken: String? = nil
    let eosToken: String? = nil
    let unknownToken: String? = nil

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [3, 5, 7] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map(String.init).joined(separator: " ")
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { String(id) }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [3, 5, 7] }
}

/// `.info` must arrive while an artificially delayed cache store is STILL
/// RUNNING (spinner-off precedes persistence), and the stream must not
/// finish until the store has completed (the next generation still
/// serializes safely on stream termination).
@Test func infoArrivesWhileDelayedCacheStoreStillRunning() async throws {
    let clock = DelayedStoreClock()
    let delay: TimeInterval = 1.5
    let tokenizer = TailTestTokenizer()

    let (stream, task) = generateTaskDeferred(
        promptTokenCount: 3,
        modelConfiguration: ModelConfiguration(id: "tail-test/none"),
        tokenizer: tokenizer,
        promptTokenIds: [3, 5, 7],
        makeIterator: { DelayedStoreIterator(clock: clock, storeDelay: delay) }
    )

    var infoAt: Date?
    var infoCount = 0
    for await event in stream {
        if case .info = event {
            infoAt = Date()
            infoCount += 1
        }
    }
    let streamEndedAt = Date()
    _ = await task.result

    let snapshot = clock.state.withLock { $0 }
    let storeStarted = snapshot.storeStarted
    let storeEnded = snapshot.storeEnded
    let statsAt = snapshot.statsFinalizedAt

    #expect(infoCount == 1)
    let info = try #require(infoAt)
    let started = try #require(storeStarted)
    let ended = try #require(storeEnded)
    #expect(statsAt != nil, "CPU-only stats finalize must run")
    // The user-visible completion precedes even the START of persistence.
    #expect(info <= started, "info \(info) must precede store start \(started)")
    // And by construction precedes its end by ~the injected delay.
    #expect(ended.timeIntervalSince(info) >= delay * 0.9)
    // The stream itself (the host's serialization point) does NOT end
    // until the store has fully completed.
    #expect(streamEndedAt >= ended)
}

/// The idempotency backstop: an iterator whose stats finalize runs twice
/// (once from the loop, once from a legacy direct `storeCacheAfterGeneration`
/// call) must not double-finalize. Uses the native-MTP iterator's contract
/// indirectly via the default no-op — here we just pin that the loop calls
/// finalize exactly once per generation.
@Test func loopFinalizesStatsExactlyOnce() async throws {
    let counter = FinalizeCounter()
    let (stream, task) = generateTaskDeferred(
        promptTokenCount: 3,
        modelConfiguration: ModelConfiguration(id: "tail-test/none"),
        tokenizer: TailTestTokenizer(),
        promptTokenIds: [3, 5, 7],
        makeIterator: { CountingIterator(counter: counter) }
    )
    for await _ in stream {}
    _ = await task.result
    #expect(counter.count.withLock { $0 } == 1)
}

private final class FinalizeCounter: Sendable {
    let count = OSAllocatedUnfairLock(initialState: 0)
}

private struct CountingIterator: TokenIteratorProtocol {
    let maxTokens: Int? = nil
    var tokenCount = 0
    let promptPrefillTime: TimeInterval = 0
    let counter: FinalizeCounter
    private var emitted = 0

    init(counter: FinalizeCounter) { self.counter = counter }

    mutating func next() -> Int? {
        guard emitted < 1 else { return nil }
        emitted += 1
        return 9
    }

    mutating func finalizeGenerationStats(generatedTokenIds: [Int]) {
        counter.count.withLock { $0 += 1 }
    }
}
