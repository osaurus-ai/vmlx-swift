// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Minimum reasoning floor")
struct MinimumReasoningFloorFocusedTests {
    @Test("initial suppress masks the token for the leading window then releases it")
    func initialSuppressMasksThenReleases() throws {
        try FocusedMLXTestSupport.withLock {
            let banned = 3
            let vocab = 8
            var processor = InitialSuppressTokensProcessor(tokens: [banned], count: 4)

            // Logits where the banned token is the clear argmax.
            func makeLogits() -> MLXArray {
                var values = [Float](repeating: 0, count: vocab)
                values[banned] = 10
                return MLXArray(values).reshaped(1, vocab)
            }

            for step in 0..<4 {
                let masked = processor.process(logits: makeLogits())
                let bannedLogit = masked[0, banned].item(Float.self)
                #expect(
                    bannedLogit == -Float.infinity,
                    "step \(step): banned token must stay masked inside the window")
                let keptLogit = masked[0, 0].item(Float.self)
                #expect(keptLogit == 0, "step \(step): other logits must be untouched")
                processor.didSample(token: MLXArray([Int32(0)]))
            }

            let released = processor.process(logits: makeLogits())
            #expect(
                released[0, banned].item(Float.self) == 10,
                "after the window the ban must lift — the floor only DELAYS the close")
        }
    }

    @Test("floor arms on every enforced-effort thinking rail and nowhere else")
    func railMatchingIsExact() {
        let lowHead = DeepseekV4ChatEncoder.reasoningEffortLowPreface + "You are helpful."
        let highHead = DeepseekV4ChatEncoder.reasoningEffortHighPreface + "You are helpful."
        let maxHead = DeepseekV4ChatEncoder.reasoningEffortMaxPreface + "You are helpful."
        let bareHead = "You are helpful."
        let thinkTail = "<｜Assistant｜><think>"
        let chatTail = "<｜Assistant｜><think></think>"

        #expect(
            MinimumReasoningFloor.railRequiresFloor(promptHead: lowHead, promptTail: thinkTail))
        #expect(
            MinimumReasoningFloor.railRequiresFloor(promptHead: highHead, promptTail: thinkTail))
        #expect(
            MinimumReasoningFloor.railRequiresFloor(promptHead: maxHead, promptTail: thinkTail))
        #expect(
            !MinimumReasoningFloor.railRequiresFloor(promptHead: bareHead, promptTail: thinkTail),
            "thinking rails without an effort preface must never be floored")
        #expect(
            !MinimumReasoningFloor.railRequiresFloor(promptHead: lowHead, promptTail: chatTail),
            "the chat rail (closed think) must never be floored")
        #expect(!MinimumReasoningFloor.railRequiresFloor(promptHead: nil, promptTail: thinkTail))
        #expect(!MinimumReasoningFloor.railRequiresFloor(promptHead: lowHead, promptTail: nil))
    }

    @Test("arming markers are the first lines of the enforced-effort prefaces")
    func markersDeriveFromEncoderPrefaces() {
        let markers = MinimumReasoningFloor.enforcedEffortMarkers
        #expect(markers.count == 3)
        for (marker, preface) in zip(
            markers,
            [
                DeepseekV4ChatEncoder.reasoningEffortLowPreface,
                DeepseekV4ChatEncoder.reasoningEffortHighPreface,
                DeepseekV4ChatEncoder.reasoningEffortMaxPreface,
            ])
        {
            #expect(!marker.isEmpty)
            #expect(
                preface.hasPrefix(marker + "\n"),
                "markers must stay in lockstep with the encoder prefaces")
        }
    }

    @Test("armed floor disqualifies native MTP lossless greedy")
    func armedFloorDisqualifiesNativeMTP() {
        var parameters = GenerateParameters(temperature: 0)
        #expect(parameters.isNativeMTPLosslessGreedyEligible)

        parameters.initialSuppressTokens = [5]
        parameters.initialSuppressCount = MinimumReasoningFloor.tokenCount
        #expect(
            !parameters.isNativeMTPLosslessGreedyEligible,
            "drafted MTP tokens would bypass the per-token logit mask")
    }

    @Test("floor arming is reached in both BatchEngine entry points")
    func floorArmingIsReachedInBothEntryPoints() throws {
        // A locally-correct guard the system never reaches is worthless:
        // pin the arming call sites in BOTH generate() and submit() so a
        // refactor cannot silently orphan the floor.
        let engine = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)
        let armingCalls = engine.components(
            separatedBy: "MinimumReasoningFloor.armIfNeeded("
        ).count - 1
        #expect(armingCalls == 2, "expected arming in generate() AND submit(), found \(armingCalls)")

        // The floor must remain mask-only: a leading -inf window, never a
        // boost or forced emission. This is the inverse of the removed
        // hidden close bias and must never regress into one.
        let floor = try String(
            contentsOfFile: "Libraries/MLXLMCommon/MinimumReasoningFloor.swift",
            encoding: .utf8)
        #expect(floor.contains("never forces or biases toward"))
        let evaluate = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)
        #expect(evaluate.contains("initialSuppressTokens.isEmpty"))
    }
}
