// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Acceptance is a property of the CONTENT DOMAIN and the target's quant,
// not of the runtime: structured spans (code, lists) draft far better
// than free prose. This matrix measures acceptance length and throughput
// across prompt domain × reasoning effort × block size on one model
// load, with a plain-decode reference per condition.
//
//   VMLX_DFLASH2_MATRIX_TARGET=<bundle> swift test --filter DFlash2AcceptanceMatrixTests

import Foundation
import MLX
import XCTest
@preconcurrency import VMLXTokenizers
@testable import MLXHuggingFace
@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

final class DFlash2AcceptanceMatrixTests: XCTestCase {

    private static var targetPath: String? {
        ProcessInfo.processInfo.environment["VMLX_DFLASH2_MATRIX_TARGET"]
    }

    private static var drafterURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("models/Qwen3.8-27B-DFlash2")
    }

    func testAcceptanceByDomainAndEffort() async throws {
        guard let targetPath = Self.targetPath else {
            throw XCTSkip("Set VMLX_DFLASH2_MATRIX_TARGET to a Qwen3.8-27B bundle")
        }
        guard DFlash2Loader.looksLikeDFlash2Drafter(at: Self.drafterURL) else {
            throw XCTSkip("No DFlash 2 drafter at \(Self.drafterURL.path)")
        }

        let context = try await MLXLMCommon.loadModel(
            from: URL(fileURLWithPath: targetPath),
            using: #huggingFaceTokenizerLoader())
        nonisolated(unsafe) let ctx = context
        let drafter = try DFlash2Loader.load(from: Self.drafterURL)
        let maxTokens = 160

        let prompts: [(name: String, text: String)] = [
            ("general", "Explain why the sky is blue and why sunsets turn red, in about 150 words."),
            ("coding", "Write a Swift function that parses an ISO-8601 date string (YYYY-MM-DD) into year, month and day Ints without Foundation, returning nil on malformed input. Include a few example calls."),
        ]
        // (label, enable_thinking, reasoning_effort or nil)
        let efforts: [(name: String, thinking: Bool, effort: String?)] = [
            ("xhigh", true, nil),  // template default
            ("medium", true, "medium"),
            ("off", false, nil),
        ]

        func prepare(_ prompt: String, thinking: Bool, effort: String?) async throws -> LMInput {
            var userInput = UserInput(chat: [.user(prompt)])
            var extra: [String: any Sendable] = ["enable_thinking": thinking]
            if let effort { extra["reasoning_effort"] = effort }
            userInput.additionalContext = extra
            return try await ctx.processor.prepare(input: userInput)
        }

        var params = GenerateParameters(maxTokens: maxTokens, temperature: 0)
        params.prefillStepSize = 1024

        // Block size is a real tuning axis, not a constant: acceptance can
        // SATURATE the block (coding at b8 accepts 5.71 of a max 8), and
        // the quantized matmul cost is nearly flat from M=8 to M=16, so a
        // wider block can buy tokens almost for free. Overridable so the
        // sweep can be widened without an edit.
        let blockSizes: [Int] =
            ProcessInfo.processInfo.environment["VMLX_DFLASH2_BLOCKS"]
            .map { $0.split(separator: ",").compactMap { Int($0) } }
            ?? [8, 4]

        print("[accept-matrix] prompt   effort  arm      tok    s    tok/s  accLen  acc%  cycles  ar")
        for (pName, pText) in prompts {
            for (eName, thinking, effort) in efforts {
                // Plain reference for this condition.
                do {
                    let input = try await prepare(pText, thinking: thinking, effort: effort)
                    var p = params
                    p.draftStrategy = nil
                    var tokens = 0
                    let start = Date()
                    for await item in try MLXLMCommon.generate(input: input, parameters: p, context: ctx) {
                        if case .info(let info) = item { tokens = info.generationTokenCount }
                    }
                    let secs = Date().timeIntervalSince(start)
                    print(String(
                        format: "[accept-matrix] %-8s %-7s plain   %4d %5.2f  %6.2f      --    --      --  --",
                        (pName as NSString).utf8String!, (eName as NSString).utf8String!,
                        tokens, secs, Double(tokens) / max(secs, 0.001)))
                }
                for bs in blockSizes {
                    let input = try await prepare(pText, thinking: thinking, effort: effort)
                    // 0 means "let the adaptive controller pick" — a nil
                    // block size is how a real request arrives.
                    var iterator = try DFlash2TokenIterator(
                        input: input, target: ctx.model as! any DFlash2Target,
                        drafter: drafter, blockSize: bs == 0 ? nil : bs, parameters: params,
                        cacheCoordinator: nil)
                    var produced = 0
                    let start = Date()
                    while produced < maxTokens, iterator.next() != nil { produced += 1 }
                    let secs = Date().timeIntervalSince(start)
                    let s = iterator.dflash2Stats
                    let accRate = s.draftedTokens > 0
                        ? Double(s.acceptedTokens) / Double(s.draftedTokens) * 100 : 0
                    print(String(
                        format: "[accept-matrix] %-8s %-7s %-7s  %4d %5.2f  %6.2f   %5.2f  %4.0f  %6d  %2d",
                        (pName as NSString).utf8String!, (eName as NSString).utf8String!,
                        ((bs == 0 ? "adaptive" : "b\(bs)") as NSString).utf8String!,
                        produced, secs, Double(produced) / max(secs, 0.001),
                        s.acceptanceLength, accRate, s.verifyCalls,
                        s.autoregressiveFallbackTokens))
                    XCTAssertGreaterThan(produced, 0)
                }
            }
        }
    }
}
