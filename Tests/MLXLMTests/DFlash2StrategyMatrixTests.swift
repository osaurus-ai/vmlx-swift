// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The whole decode-strategy matrix on one model, one load, one process:
//
//   { plain, native MTP d1, dflash2 b8, dflash2 b4 } × { reasoning on, off }
//
// Native MTP and DFlash 2 are mutually exclusive alternatives; this is
// the head-to-head that says which one a given bundle should use. The
// model is loaded ONCE with the MTP sidecar enabled — the loader
// isolates MTP tensors from the base AR weights, so the plain and
// DFlash 2 arms are byte-identical to a non-MTP load.
//
// Reasoning is toggled the way the host does it:
// `additionalContext["enable_thinking"]` through the chat template.
//
// Two interleaved rounds; round 2 reported. Output snippets printed per
// arm so the runs double as behavioural examples.
//
//   VMLX_DFLASH2_MATRIX_TARGET=$HOME/models/JANGQ-AI/Qwen3.8-27B-JANG_4D \
//   swift test --filter DFlash2StrategyMatrixTests

import Foundation
import MLX
import XCTest
@preconcurrency import VMLXTokenizers
@testable import MLXHuggingFace
@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

final class DFlash2StrategyMatrixTests: XCTestCase {

    private static var targetPath: String? {
        ProcessInfo.processInfo.environment["VMLX_DFLASH2_MATRIX_TARGET"]
    }

    private static var drafterURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("models/Qwen3.8-27B-DFlash2")
    }

    // VMLX_DFLASH2_MATRIX_PROMPT=code|prose swaps the prompt — structured
    // output accepts drafts far better than free prose (the same domain
    // split every MTP stack reports), and the prose prompt runs long
    // enough to fill a 320-token budget instead of stopping at 94.
    private static let prompt: String = {
        switch ProcessInfo.processInfo.environment["VMLX_DFLASH2_MATRIX_PROMPT"] {
        case "code":
            return "Write a Swift function that parses a CSV line into fields, handling quoted fields with embedded commas. Code only, no explanation."
        case "prose":
            return "Write a detailed explanation of how tides work, covering the moon's role, spring and neap tides, and why some coasts see larger tides than others."
        default:
            return "List the first eight prime numbers, then explain in two sentences why 1 is not prime."
        }
    }()

    func testMatrix() async throws {
        guard let targetPath = Self.targetPath else {
            throw XCTSkip("Set VMLX_DFLASH2_MATRIX_TARGET to a Qwen3.8-27B bundle")
        }
        guard DFlash2Loader.looksLikeDFlash2Drafter(at: Self.drafterURL) else {
            throw XCTSkip("No DFlash 2 drafter at \(Self.drafterURL.path)")
        }

        // One load serves every arm. nativeMTP: true keeps the sidecar so
        // the MTP arm can run; base AR weights are unaffected.
        var loadConfig = LoadConfiguration.default
        loadConfig.nativeMTP = true
        let (context, _) = try await MLXLMCommon.loadModel(
            from: URL(fileURLWithPath: targetPath),
            using: #huggingFaceTokenizerLoader(),
            loadConfiguration: loadConfig)
        nonisolated(unsafe) let ctx = context
        if let mtpModel = ctx.model as? any NativeMTPModel {
            print("[matrix] nativeMTPAvailable=\(mtpModel.nativeMTPAvailable)")
        }
        let maxTokens = ProcessInfo.processInfo.environment["VMLX_DFLASH2_MATRIX_MAXTOKENS"]
            .flatMap(Int.init) ?? 160

        struct Arm { let name: String; let strategy: DraftStrategy? }
        let allArms: [Arm] = [
            Arm(name: "plain", strategy: nil),
            Arm(name: "mtp-d1", strategy: .nativeMTP(depth: 1)),
            Arm(name: "mtp-d2", strategy: .nativeMTP(depth: 2)),
            Arm(name: "mtp-d3", strategy: .nativeMTP(depth: 3)),
            Arm(name: "mtp-d4", strategy: .nativeMTP(depth: 4)),
            Arm(name: "dflash2-b8", strategy: .dflash2(drafterPath: Self.drafterURL, blockSize: 8)),
            Arm(name: "dflash2-b4", strategy: .dflash2(drafterPath: Self.drafterURL, blockSize: 4)),
            // nil = the runtime default (width 5 since the speed-recipe port).
            Arm(name: "dflash2-auto", strategy: .dflash2(drafterPath: Self.drafterURL, blockSize: nil)),
        ]
        // VMLX_DFLASH2_MATRIX_ARMS=plain,mtp-d3 runs a subset;
        // VMLX_DFLASH2_MATRIX_THINK=on|off runs one leg.
        let armFilter = ProcessInfo.processInfo.environment["VMLX_DFLASH2_MATRIX_ARMS"]?
            .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        let arms = allArms.filter { armFilter?.contains($0.name) ?? true }
        let thinkLegs: [Bool]
        switch ProcessInfo.processInfo.environment["VMLX_DFLASH2_MATRIX_THINK"]?.lowercased() {
        case "on": thinkLegs = [true]
        case "off": thinkLegs = [false]
        default: thinkLegs = [true, false]
        }

        func run(_ arm: Arm, thinking: Bool) async throws
            -> (reasoning: String, content: String, seconds: Double, tokens: Int)
        {
            var userInput = UserInput(chat: [.user(Self.prompt)])
            userInput.additionalContext = ["enable_thinking": thinking]
            let input = try await ctx.processor.prepare(input: userInput)

            var p = GenerateParameters(
                generationConfig: ctx.configuration.generationDefaults,
                fallback: GenerateParameters(maxTokens: maxTokens, prefillStepSize: 1024))
            p.maxTokens = maxTokens
            p.prefillStepSize = 1024
            p.temperature = 0
            p.topP = 1
            p.topK = 0
            p.minP = 0
            p.draftStrategy = arm.strategy

            var reasoning = ""
            var content = ""
            var tokens = 0
            let start = Date()
            for await item in try MLXLMCommon.generate(
                input: input, parameters: p, context: ctx)
            {
                switch item {
                case .chunk(let c): content += c
                case .reasoning(let c): reasoning += c
                case .info(let info): tokens = info.generationTokenCount
                default: break
                }
            }
            return (reasoning, content, Date().timeIntervalSince(start), tokens)
        }

        for thinking in thinkLegs {
            var round2: [String: (reasoning: String, content: String, seconds: Double, tokens: Int)] = [:]
            for round in 1 ... 2 {
                for arm in arms {
                    Memory.clearCache()
                    let result = try await run(arm, thinking: thinking)
                    let tokPerSec = Double(result.tokens) / Swift.max(result.seconds, 0.001)
                    print(String(
                        format: "[matrix r%d think=%@] %-10s  %3d tok  %6.2fs  %6.2f tok/s",
                        round, thinking ? "ON " : "OFF", (arm.name as NSString).utf8String!,
                        result.tokens, result.seconds, tokPerSec))
                    if round == 2 { round2[arm.name] = result }
                }
            }

            guard let plain = round2["plain"] else { continue }
            print("[matrix think=\(thinking ? "ON" : "OFF")] plain reasoning: \"\(plain.reasoning.prefix(90))\"")
            print("[matrix think=\(thinking ? "ON" : "OFF")] plain content  : \"\(plain.content.prefix(90))\"")
            if thinking {
                XCTAssertFalse(plain.reasoning.isEmpty, "reasoning ON must produce a think block")
            } else {
                XCTAssertTrue(
                    plain.reasoning.isEmpty,
                    "reasoning OFF must not produce a think block, got: \(plain.reasoning.prefix(120))")
                XCTAssertFalse(plain.content.isEmpty, "reasoning OFF must answer in content")
            }
            for arm in arms where arm.name != "plain" {
                let result = try XCTUnwrap(round2[arm.name])
                let speedup = plain.seconds / Swift.max(result.seconds, 0.001)
                let all = result.reasoning + result.content
                let plainAll = plain.reasoning + plain.content
                let shared = zip(all, plainAll).prefix { $0 == $1 }.count
                print(String(
                    format: "[matrix think=%@] %-10s speedup=%.2fx sharedPrefix=%d/%d  out: \"%@\"",
                    thinking ? "ON " : "OFF", (arm.name as NSString).utf8String!, speedup,
                    shared, plainAll.count, String(all.prefix(70))))
                XCTAssertFalse(all.isEmpty, "\(arm.name) produced nothing")
                XCTAssertGreaterThan(
                    shared, 60,
                    "\(arm.name) diverged from plain decode almost immediately — broken, not a near-tie")
            }
        }
    }
}
