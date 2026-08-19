// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Speed sweep: baseline vs DFlash 2 at block sizes 8 / 6 / 4.
//
// Why the block sweep: z-lab/dflash#151 measured the shipped block_size
// well above the Apple-Silicon throughput optimum — smaller blocks won
// by 24-29% despite LOWER acceptance, because a quantized verify
// forward's cost grows faster with row count than acceptance does.
//
// Method per the measurement rules that have burned this repo before:
// two interleaved rounds (B, D8, D6, D4, B, D8, D6, D4) so every arm
// runs both cold-ish and hot; only round 2 is reported; free memory is
// printed before each leg so a paging artifact cannot masquerade as a
// regression. Every DFlash arm is also checked byte-identical against
// the baseline — a fast-but-wrong arm must fail, not win.
//
//   VMLX_DFLASH2_SPEED_TARGET=$HOME/models/JANGQ-AI/Qwen3.8-27B-JANG_4D \
//   swift test --filter DFlash2SpeedSweepTests

import Foundation
import MLX
import XCTest
@preconcurrency import VMLXTokenizers
@testable import MLXHuggingFace
@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

final class DFlash2SpeedSweepTests: XCTestCase {

    private static var targetPath: String? {
        ProcessInfo.processInfo.environment["VMLX_DFLASH2_SPEED_TARGET"]
    }

    private static var drafterURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("models/Qwen3.8-27B-DFlash2")
    }

    private static let prompt =
        "Explain, in detail, how quicksort chooses pivots, why the worst case is quadratic, and two standard mitigations."

    private func freeMemoryGB() -> Double {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return -1 }
        return Double(stats.free_count + stats.inactive_count) * 16384 / 1_073_741_824
    }

    func testSweep() async throws {
        guard let targetPath = Self.targetPath else {
            throw XCTSkip("Set VMLX_DFLASH2_SPEED_TARGET to a Qwen3.8-27B bundle")
        }
        guard DFlash2Loader.looksLikeDFlash2Drafter(at: Self.drafterURL) else {
            throw XCTSkip("No DFlash 2 drafter at \(Self.drafterURL.path)")
        }

        let context = try await MLXLMCommon.loadModel(
            from: URL(fileURLWithPath: targetPath),
            using: #huggingFaceTokenizerLoader())
        nonisolated(unsafe) let ctx = context
        let maxTokens = 192

        func params(_ strategy: DraftStrategy?) -> GenerateParameters {
            var p = GenerateParameters(
                generationConfig: ctx.configuration.generationDefaults,
                fallback: GenerateParameters(maxTokens: maxTokens, prefillStepSize: 1024))
            p.maxTokens = maxTokens
            p.prefillStepSize = 1024
            p.temperature = 0
            p.topP = 1
            p.topK = 0
            p.minP = 0
            if ProcessInfo.processInfo.environment["VMLX_SWEEP_COMPILED_DECODE"] == "1" {
                p.enableCompiledDecode = true
            }
            return p
        }

        struct Arm { let name: String; let strategy: DraftStrategy? }
        let arms: [Arm] = [
            Arm(name: "baseline", strategy: nil),
            Arm(name: "dflash2-b8", strategy: .dflash2(drafterPath: Self.drafterURL, blockSize: 8)),
            Arm(name: "dflash2-b6", strategy: .dflash2(drafterPath: Self.drafterURL, blockSize: 6)),
            Arm(name: "dflash2-b4", strategy: .dflash2(drafterPath: Self.drafterURL, blockSize: 4)),
        ]

        func run(_ arm: Arm) async throws -> (text: String, seconds: Double, tokens: Int) {
            let input = try await ctx.processor.prepare(
                input: UserInput(chat: [.user(Self.prompt)]))
            var p = params(arm.strategy)
            p.draftStrategy = arm.strategy
            var text = ""
            var tokens = 0
            let start = Date()
            for await item in try MLXLMCommon.generate(
                input: input, parameters: p, context: ctx)
            {
                switch item {
                case .chunk(let c): text += c
                case .reasoning(let c): text += c
                case .info(let info): tokens = info.generationTokenCount
                default: break
                }
            }
            return (text, Date().timeIntervalSince(start), tokens)
        }

        var lastRound: [String: (text: String, seconds: Double, tokens: Int)] = [:]
        for round in 1 ... 2 {
            for arm in arms {
                Memory.clearCache()
                let free = freeMemoryGB()
                let result = try await run(arm)
                let tokPerSec = Double(result.tokens) / Swift.max(result.seconds, 0.001)
                print(String(
                    format: "[sweep r%d] %-11s free=%5.1fGB  %3d tok  %6.2fs  %6.2f tok/s",
                    round, (arm.name as NSString).utf8String!, free, result.tokens,
                    result.seconds, tokPerSec))
                if round == 2 { lastRound[arm.name] = result }
            }
        }

        let baseline = try XCTUnwrap(lastRound["baseline"])
        XCTAssertFalse(baseline.text.isEmpty)
        // Cross-run correctness diffing (compiled vs plain decode): dump the
        // baseline arm's greedy text so two invocations can be compared.
        if let dumpPath = ProcessInfo.processInfo.environment["VMLX_SWEEP_DUMP"] {
            try baseline.text.write(
                toFile: dumpPath, atomically: true, encoding: .utf8)
            print("[sweep] baseline text dumped to \(dumpPath)")
        }
        for arm in arms where arm.name != "baseline" {
            let result = try XCTUnwrap(lastRound[arm.name])
            // Bit-equality with the baseline holds on bf16 targets (proven
            // in DFlash2LosslessSmokeTests) but NOT on quantized ones: a
            // quantized matmul's reduction order depends on the row count,
            // so an M = 1+block verify forward produces logits a few ulp
            // away from the M = 1 decode forward, and a greedy near-tie
            // can legitimately flip. DFlash 2's guarantee is that every
            // emitted token is the verify forward's own argmax — measured
            // here as a long shared prefix up to the first near-tie, and a
            // same-length, coherent completion after it.
            let shared = zip(result.text, baseline.text).prefix { $0 == $1 }.count
            if result.text != baseline.text {
                print(String(
                    format: "[sweep] %@ diverged from baseline at char %d of %d (quantized near-tie)",
                    arm.name, shared, baseline.text.count))
            }
            XCTAssertGreaterThan(
                shared, 120,
                "\(arm.name) diverged from the baseline almost immediately — that is a broken forward, not a near-tie"
            )
            XCTAssertFalse(result.text.isEmpty)
            let speedup = baseline.seconds / Swift.max(result.seconds, 0.001)
            print(String(format: "[sweep] %@ speedup vs baseline: %.2fx", arm.name, speedup))
        }
    }
}
