// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Regression for the force-unwrap crash at BatchEngine stepBatchDecode
// `activeSlots[$0].nextToken!`.
//
// Scenario: sampling EOS as the FIRST decode token. `stepPrefill`
// transitions the slot to `.decode` BEFORE checking for EOS, calls
// `finishSlot(..., .stop)`, sets `isFinished = true`, but NEVER
// assigns `nextToken` (that branch is EOS-only). Before this fix,
// `step()`'s decodeIndices filter picked up the finished slot and
// `stepBatchDecode` force-unwrapped the nil, crashing.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
@preconcurrency import VMLXTokenizers
import XCTest

class BatchEngineEOSFirstTokenTests: XCTestCase {

    /// Build an engine whose stop-token set is the ENTIRE vocabulary.
    /// Guarantees the first sampled token hits an EOS — exercises the
    /// finish-on-first-token branch of stepPrefill.
    private func makeAllEOSEngine() -> BatchEngine {
        let vocabSize = 100
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 2, intermediateSize: 128,
            attentionHeads: 4, rmsNormEps: 1e-5, vocabularySize: vocabSize,
            kvHeads: 2)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)
        MLX.eval(model)

        var modelConfig = ModelConfiguration(id: "test-all-eos")
        // Every possible token id is an EOS - first sampled token is
        // guaranteed to match, firing the crash path.
        modelConfig.eosTokenIds = Set(0 ..< vocabSize)

        let processor = TestInputProcessor(
            tokenizer: TestTokenizer(),
            configuration: modelConfig,
            messageGenerator: DefaultMessageGenerator()
        )
        nonisolated(unsafe) let context = ModelContext(
            configuration: modelConfig,
            model: model,
            processor: processor,
            tokenizer: processor.tokenizer
        )
        return BatchEngine(context: context, maxBatchSize: 2)
    }

    /// Force-unwrap regression. Submits a slot where the sampler is
    /// guaranteed to emit EOS as its first token. Pre-fix: engine
    /// crashes in `stepBatchDecode` when building the batch tensor.
    /// Post-fix: engine finishes the slot cleanly with `stopReason`
    /// `.stop` and zero generated tokens.
    func testEOSOnFirstTokenDoesNotCrash() async throws {
        let engine = makeAllEOSEngine()

        let input = LMInput(tokens: MLXArray(Int32(1) ..< Int32(6)))
        let params = GenerateParameters(maxTokens: 16, temperature: 0)

        let (_, stream) = await engine.submit(input: input, parameters: params)

        var tokenCount = 0
        var stopReason: GenerateStopReason?
        for await event in stream {
            switch event {
            case .prefillProgress:
                break
            case .token:
                tokenCount += 1
            case .prefillProgress:

                break
            case .info(let info):
                stopReason = info.stopReason
            }
        }

        XCTAssertEqual(tokenCount, 0,
            "EOS on the first decode token must not surface to client")
        XCTAssertEqual(stopReason, .stop,
            "Expected .stop as the finish reason (got \(String(describing: stopReason)))")
    }

    /// Two concurrent slots, both guaranteed to EOS on first token.
    /// Catches the more-subtle case where stepBatchDecode's filter
    /// might look fine on one slot but break when the batch dim is
    /// non-trivial and the finished-but-still-present slot happens to
    /// be index 0 of the batch.
    func testConcurrentEOSFirstTokenBoth() async throws {
        let engine = makeAllEOSEngine()

        let input1 = LMInput(tokens: MLXArray(Int32(1) ..< Int32(4)))
        let input2 = LMInput(tokens: MLXArray(Int32(10) ..< Int32(14)))
        let params = GenerateParameters(maxTokens: 8, temperature: 0)

        let (_, s1) = await engine.submit(input: input1, parameters: params)
        let (_, s2) = await engine.submit(input: input2, parameters: params)

        // Collect sequentially — async-let on methods bound to `self`
        // trips Swift 6 strict-concurrency sending rules, and the test
        // doesn't need true parallelism: the engine's scheduling loop
        // is what exercises concurrency internally.
        var r1: GenerateStopReason?
        for await ev in s1 {
            if case .info(let info) = ev { r1 = info.stopReason }
        }
        var r2: GenerateStopReason?
        for await ev in s2 {
            if case .info(let info) = ev { r2 = info.stopReason }
        }

        XCTAssertEqual(r1, .stop, "Slot 1 should finish .stop on first-token EOS")
        XCTAssertEqual(r2, .stop, "Slot 2 should finish .stop on first-token EOS")
    }

    private func collectStop(
        from stream: AsyncStream<BatchGeneration>
    ) async -> GenerateStopReason? {
        var reason: GenerateStopReason?
        for await ev in stream {
            if case .info(let info) = ev {
                reason = info.stopReason
            }
        }
        return reason
    }
}
