// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Native reasoning logits without automatic rescue")
struct NativeReasoningLogitsFocusedTests {
    @Test("default generation leaves an immediate reasoning closer and the tail unchanged")
    func defaultDistributionIsUnchanged() {
        Device.withDefaultDevice(.cpu) {
            let parameters = GenerateParameters()
            #expect(parameters.processor() == nil)
            #expect(parameters.initialSuppressTokens.isEmpty)
            #expect(parameters.initialSuppressCount == 0)
            #expect(parameters.reasoningBudgetTokens == nil)
            #expect(parameters.requestedReasoningBudgetTokens == nil)
            let logits = MLXArray([Float(100), 3, -20, -100]).reshaped(1, 4)
            let alias = logits
            let result = parameters.processor()?.process(logits: logits) ?? logits
            #expect(result.asArray(Float.self) == [100, 3, -20, -100])
            #expect(alias.asArray(Float.self) == [100, 3, -20, -100])
            #expect(argMax(result, axis: -1).item(Int.self) == 0)
        }
    }

    @Test("default batch slot samples an immediate closer without mutating model logits")
    func batchDefaultDoesNotInjectLeadingSuppression() throws {
        try Device.withDefaultDevice(.cpu) {
            let stream = AsyncStream<BatchGeneration>.makeStream()
            defer { stream.continuation.finish() }
            let request = BatchPendingRequest(
                input: LMInput(tokens: MLXArray([Int32(1)])),
                parameters: GenerateParameters(maxTokens: 2, temperature: 0),
                continuation: stream.continuation)
            var slot = BatchSlot(from: request, cache: [], stopTokenIDs: [])
            #expect(slot.processor == nil)
            let logits = MLXArray([Float(100), 3, -20, -100]).reshaped(1, 4)
            let alias = logits
            let start = Date()
            let token = try slot.sampleToken(from: logits).item(Int.self)
            #expect(token == 0)
            #expect(alias.asArray(Float.self) == [100, 3, -20, -100])
            print("NATIVE_BATCH_SYNTHETIC tokens=1 tokens_per_second=\(1 / max(Date().timeIntervalSince(start), 1e-9)) CPU=true real_model_proof=false")
        }
    }

    @Test("explicit bundle suppression preserves surviving logits and caller aliases")
    func explicitSuppressionIsExactAndImmutable() throws {
        try Device.withDefaultDevice(.cpu) {
            var parameters = GenerateParameters()
            parameters.suppressTokens = [-1, 0, 0, 4, 200]
            let processor = try #require(parameters.processor())
            let logits = MLXArray([Float(100), 3, 0, -2]).reshaped(1, 4)
            let alias = logits
            let processed = processor.process(logits: logits)
            #expect(processed.asArray(Float.self) == [-Float.infinity, 3, 0, -2])
            let probabilities = softmax(processed, axis: -1).asArray(Float.self)
            #expect(probabilities[0] == 0)
            #expect(probabilities[2] > 0)
            #expect(probabilities[3] > 0)
            #expect(abs(probabilities[3] / probabilities[1] - exp(-Float(5))) < 1e-6)
            #expect(logits.asArray(Float.self) == [100, 3, 0, -2])
            #expect(alias.asArray(Float.self) == [100, 3, 0, -2])
            let invalidOnly = SuppressTokensProcessor(tokens: [-1, 4, 200])
            #expect(invalidOnly.process(logits: logits) === logits)
        }
    }

    @Test("legacy explicit leading suppression has only its requested mask and window")
    func compatibilityWindowHasNoHiddenSampler() throws {
        try Device.withDefaultDevice(.cpu) {
            var parameters = GenerateParameters(temperature: 0)
            parameters.initialSuppressTokens = [0]
            parameters.initialSuppressCount = 2
            #expect(!parameters.isNativeMTPLosslessGreedyEligible)
            var processor = try #require(parameters.processor())
            let logits = MLXArray([Float(100), 3, -20, -100]).reshaped(1, 4)
            let alias = logits
            for _ in 0 ..< 2 {
                #expect(processor.process(logits: logits).asArray(Float.self)
                    == [-Float.infinity, 3, -20, -100])
                processor.didSample(token: MLXArray(Int32(1)))
            }
            #expect(processor.process(logits: logits).asArray(Float.self) == [100, 3, -20, -100])
            #expect(alias.asArray(Float.self) == [100, 3, -20, -100])
            let disabled = InitialSuppressTokensProcessor(tokens: [0], count: 0)
            #expect(disabled.process(logits: logits) === logits)
        }
    }

    @Test("both application entry points leave native reasoning output unmasked")
    func defaultEngineReachability() throws {
        let engine = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift", encoding: .utf8)
        #expect(!engine.contains("MinimumReasoningFloor"))
        #expect(!engine.contains("initialSuppressTokens ="))
        #expect(!engine.contains("initialSuppressCount ="))
        let evaluate = try String(contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift", encoding: .utf8)
        #expect(!evaluate.contains("survivorRelativeFloor"))
        #expect(!evaluate.contains("VMLX_REASONING_FLOOR_TRACE"))
        // The separate caller-requested ceiling remains explicit on both routes.
        #expect(engine.components(separatedBy: "ReasoningBudget.armIfNeeded(").count - 1 == 2)
        #expect(engine.components(separatedBy: "parameters.requestedReasoningBudgetTokens.flatMap").count - 1 == 2)
    }
}
