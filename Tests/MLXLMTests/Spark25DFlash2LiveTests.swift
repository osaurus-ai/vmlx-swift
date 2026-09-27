// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// A Spark2.5 bundle with its drafter in `<bundle>/dflash`, end to end.
//
// Correctness is checked as a property rather than as string equality. On a
// quantized target a verify forward over 1+block rows reduces in a different
// order than a one-row decode step, so a greedy near-tie may flip and the two
// texts then legitimately differ. What must hold for ANY emitted token is
// that it is the target's own choice given the history actually emitted:
// the argmax of one teacher-forced forward over prompt + output, up to a
// near-tie margin. A sliding-window cache holding rejected rows breaks that
// with large margins from the first rollback after the window fills, which
// is why the prompts sit on both sides of the window. A negative control
// runs the same check against a corrupted history and must fail it.
//
//   VMLX_SPARK25_DFLASH_BUNDLE=<bundle with dflash/> \
//   swift test --filter Spark25DFlash2LiveTests
//
// Optional: VMLX_SPARK25_DFLASH_OUT=<dir> writes the measurements as JSON.

import Foundation
import MLX
@preconcurrency import VMLXTokenizers
import XCTest

@testable import MLXHuggingFace
@testable import MLXLLM
@testable import MLXLMCommon

final class Spark25DFlash2LiveTests: XCTestCase {

    private static var bundle: URL? {
        ProcessInfo.processInfo.environment["VMLX_SPARK25_DFLASH_BUNDLE"].map {
            URL(fileURLWithPath: $0)
        }
    }

    /// Largest top-1 minus emitted-token logit gap still treated as a tie.
    private static let tieMargin: Float = 0.75

    struct Check: Codable {
        var tokens: Int
        var mismatches: Int
        var violations: Int
        var worstMargin: Float
    }

    /// Teacher-forced audit of `generated` after `prompt`.
    private func audit(
        _ model: any LanguageModel, prompt: [Int], generated: [Int]
    ) -> Check {
        let all = MLXArray((prompt + generated).map(Int32.init)).reshaped(1, -1)
        let full = model(all, cache: nil)
        let rows = MLXArray(Int32(prompt.count - 1) ..< Int32(prompt.count - 1 + generated.count))
        let logits = full[0].take(rows, axis: 0).asType(.float32)
        precondition(
            logits.shape == [generated.count, full.dim(-1)],
            "audit forward shape \(full.shape) for \(prompt.count)+\(generated.count) tokens")
        let top = logits.max(axis: -1)
        let chosen = takeAlong(
            logits, MLXArray(generated.map(Int32.init)).reshaped(-1, 1), axis: -1
        ).reshaped(-1)
        let margins = (top - chosen).asArray(Float.self)
        let mismatches = margins.filter { $0 > 0 }.count
        let violations = margins.filter { $0 > Self.tieMargin }.count
        return Check(
            tokens: generated.count, mismatches: mismatches, violations: violations,
            worstMargin: margins.max() ?? 0)
    }

    struct Run {
        var tokens: [Int]
        var decodeSeconds: Double
        var stats: DFlash2GenerationStats?
    }

    private func plain(
        _ ctx: ModelContext, _ input: LMInput, _ parameters: GenerateParameters
    ) throws -> Run {
        var iterator = try TokenIterator(input: input, model: ctx.model, parameters: parameters)
        var tokens: [Int] = []
        let start = Date()
        while tokens.count < (parameters.maxTokens ?? 256), let t = iterator.next() {
            tokens.append(t)
        }
        return Run(tokens: tokens, decodeSeconds: Date().timeIntervalSince(start), stats: nil)
    }

    private func dflash(
        _ ctx: ModelContext, _ drafter: DFlash2DraftModel, _ input: LMInput,
        _ parameters: GenerateParameters, block: Int?
    ) throws -> Run {
        var iterator = try DFlash2TokenIterator(
            input: input, target: ctx.model as! any DFlash2Target, drafter: drafter,
            blockSize: block, parameters: parameters, cacheCoordinator: nil)
        var tokens: [Int] = []
        let start = Date()
        while tokens.count < (parameters.maxTokens ?? 256), let t = iterator.next() {
            tokens.append(t)
        }
        return Run(
            tokens: tokens, decodeSeconds: Date().timeIntervalSince(start),
            stats: iterator.dflash2Stats)
    }

    func testBundledDrafterIsLosslessAndMeasured() async throws {
        guard let bundle = Self.bundle else {
            throw XCTSkip("Set VMLX_SPARK25_DFLASH_BUNDLE to a Spark2.5 bundle with dflash/")
        }
        let configData = try Data(contentsOf: bundle.appendingPathComponent("config.json"))
        let selection = try XCTUnwrap(
            VMLXServerRuntimeSettings().resolvedDFlash2Selection(
                configData: configData, modelDirectory: bundle),
            "the bundled drafter must resolve with default settings")

        let context = try await MLXLMCommon.loadModel(
            from: bundle, using: #huggingFaceTokenizerLoader())
        nonisolated(unsafe) let ctx = context
        XCTAssertTrue(ctx.model is Spark25Model)
        let drafter = try DFlash2DrafterResolver.shared.drafter(
            at: URL(fileURLWithPath: selection.path),
            vocabularySize: selection.vocabularySize, targetLayerCount: 36)

        func prepare(_ text: String) async throws -> LMInput {
            try await ctx.processor.prepare(input: UserInput(chat: [.user(text)]))
        }
        var greedy = GenerateParameters(maxTokens: 640, temperature: 0)
        greedy.prefillStepSize = 1024

        // Short prompt: the window fills DURING decode. Long prompt: it is
        // full before the first verify. 37 filler sentences keep the length
        // off any power of two.
        let filler = (1 ... 37).map {
            "Note \($0): the relay in bay \($0 % 7) reported a nominal reading at tick \($0 * 13)."
        }.joined(separator: " ")
        let prompts: [(String, String)] = [
            (
                "short",
                "Write a Python function that merges two sorted lists, then explain its complexity."
            ),
            ("long", filler + "\n\nSummarise the notes above as a table of bay, count of reports."),
        ]

        var report: [String: Any] = [:]
        for (name, text) in prompts {
            let input = try await prepare(text)
            let promptTokens = input.text.tokens.reshaped(-1).asArray(Int.self)
            func mark(_ s: String) {
                FileHandle.standardError.write(Data("[raptor-live] \(name): \(s)\n".utf8))
            }
            mark("prompt \(promptTokens.count) tokens; plain decode")
            let base = try plain(ctx, input, greedy)
            mark("plain produced \(base.tokens.count); dflash decode")
            let blocks: [Int?] =
                ProcessInfo.processInfo.environment["VMLX_SPARK25_DFLASH_AUDIT_BLOCKS"]
                .map { $0.split(separator: ",").map { Int($0) } } ?? [nil]
            let baseCheck = audit(ctx.model, prompt: promptTokens, generated: base.tokens)
            XCTAssertEqual(baseCheck.violations, 0, "\(name): plain decode is the harness baseline")
            var arms: [String: Any] = [
                "plain": [
                    "tokens": base.tokens.count, "mismatches": baseCheck.mismatches,
                    "violations": baseCheck.violations, "worst_margin": baseCheck.worstMargin,
                ]
            ]
            for block in blocks {
                let label = block.map { "b\($0)" } ?? "default"
                mark("dflash \(label) decode")
                let spec = try dflash(ctx, drafter, input, greedy, block: block)
                let specCheck = audit(ctx.model, prompt: promptTokens, generated: spec.tokens)
                // Control: the DFlash output audited against a history
                // missing its first 64 prompt tokens must show violations.
                let control = audit(
                    ctx.model, prompt: Array(promptTokens.dropFirst(64)), generated: spec.tokens)
                let shared = zip(base.tokens, spec.tokens).prefix { $0 == $1 }.count
                let stats = try XCTUnwrap(spec.stats)
                print(
                    "[raptor-live] \(name) \(label) prompt=\(promptTokens.count) plain=\(base.tokens.count) dflash=\(spec.tokens.count) shared=\(shared) accLen=\(String(format: "%.2f", stats.acceptanceLength)) base=\(baseCheck) dflash=\(specCheck) control=\(control)"
                )
                XCTAssertGreaterThan(promptTokens.count + spec.tokens.count, 512 + 64)
                XCTAssertEqual(
                    specCheck.violations, 0,
                    "\(name) \(label): DFlash emitted a token the target would not")
                XCTAssertGreaterThan(
                    control.violations, 0, "\(name): the audit cannot see a wrong history")
                XCTAssertGreaterThan(stats.acceptanceLength, 1.2)
                arms[label] = [
                    "tokens": spec.tokens.count, "shared_prefix_tokens": shared,
                    "mismatches": specCheck.mismatches, "violations": specCheck.violations,
                    "worst_margin": specCheck.worstMargin,
                    "acceptance_length": stats.acceptanceLength,
                    "verify_calls": stats.verifyCalls, "block": stats.blockSize,
                    "ar_fallback": stats.autoregressiveFallbackTokens,
                    "control_violations": control.violations,
                ]
            }
            report[name] = ["prompt_tokens": promptTokens.count, "arms": arms]
        }

        // Throughput: one warm-up, then ≥3 probes per arm, interleaved,
        // median reported. Greedy and the bundle's own sampler.
        let probes = Int(ProcessInfo.processInfo.environment["VMLX_SPARK25_DFLASH_PROBES"] ?? "3")!
        let input = try await prepare(prompts[0].1)
        var sampled = GenerateParameters(
            generationConfig: ctx.configuration.generationDefaults,
            fallback: GenerateParameters(maxTokens: 256))
        sampled.maxTokens = 256
        sampled.prefillStepSize = 1024
        var greedy256 = greedy
        greedy256.maxTokens = 256
        let arms: [(String, GenerateParameters, Int?)] = [
            ("greedy-plain", greedy256, -1), ("greedy-b8", greedy256, 8),
            ("greedy-b6", greedy256, 6), ("greedy-b5", greedy256, 5), ("greedy-b4", greedy256, 4),
            ("sampled-plain", sampled, -1), ("sampled-b8", sampled, 8), ("sampled-b5", sampled, 5),
            ("sampled-b4", sampled, 4),
        ]
        var rates: [String: [Double]] = [:]
        var acceptance: [String: [Double]] = [:]
        for probe in 0 ... probes {
            for (name, parameters, block) in arms {
                Memory.clearCache()
                let run =
                    try block == -1
                    ? plain(ctx, input, parameters)
                    : dflash(ctx, drafter, input, parameters, block: block)
                let rate = Double(run.tokens.count) / Swift.max(run.decodeSeconds, 1e-3)
                let st = run.stats
                let cycles = Double(Swift.max(st?.verifyCalls ?? 1, 1))
                print(
                    String(
                        format:
                            "[raptor-speed] probe %d %-13@ %4d tok %6.2f tok/s accLen %.2f  per-cycle ms draft %.2f verify %.2f commit %.2f  cycles %d ar %d",
                        probe, name as NSString, run.tokens.count, rate, st?.acceptanceLength ?? 1,
                        (st?.draftSeconds ?? 0) / cycles * 1000,
                        (st?.verifySeconds ?? 0) / cycles * 1000,
                        (st?.commitSeconds ?? 0) / cycles * 1000, st?.verifyCalls ?? 0,
                        st?.autoregressiveFallbackTokens ?? 0))
                guard probe > 0 else { continue }  // discard the warm-up probe
                rates[name, default: []].append(rate)
                acceptance[name, default: []].append(run.stats?.acceptanceLength ?? 1)
            }
        }
        func median(_ xs: [Double]) -> Double {
            let s = xs.sorted()
            return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
        }
        var speed: [String: Any] = [:]
        for (name, _, _) in arms {
            let m = median(rates[name] ?? [0])
            let base = median(
                rates[name.hasPrefix("greedy") ? "greedy-plain" : "sampled-plain"] ?? [1])
            print(
                String(
                    format:
                        "[raptor-speed] %-13@ median %6.2f tok/s  %.2fx  accLen %.2f  probes %@",
                    name as NSString, m, m / base, median(acceptance[name] ?? [1]),
                    (rates[name] ?? []).map { String(format: "%.1f", $0) }.joined(separator: ",")
                        as NSString))
            speed[name] = [
                "median_tok_s": m, "speedup": m / base, "probes": rates[name] ?? [],
                "acceptance_length": median(acceptance[name] ?? [1]),
            ]
        }
        report["speed"] = speed
        if let out = ProcessInfo.processInfo.environment["VMLX_SPARK25_DFLASH_OUT"] {
            let url = URL(fileURLWithPath: out).appendingPathComponent(
                "live-\(Int(Date().timeIntervalSince1970)).json")
            try JSONSerialization.data(
                withJSONObject: report, options: [.prettyPrinted, .sortedKeys]
            )
            .write(to: url)
            print("[raptor-live] wrote \(url.path)")
        }
    }
}
