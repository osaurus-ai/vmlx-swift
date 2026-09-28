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
import MLXRandom
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
        let skipAudit = ProcessInfo.processInfo.environment["VMLX_SPARK25_DFLASH_SKIP_AUDIT"] == "1"
        for (name, text) in prompts where !skipAudit {
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

        // The host sees which path ran: `.info` carries DFlash 2 stats for a
        // DFlash request through the public generate API, and none for plain.
        for (label, strategy) in [
            (
                "dflash",
                DraftStrategy.dflash2(
                    drafterPath: URL(fileURLWithPath: selection.path), blockSize: nil)
            ),
            ("plain", nil),
        ] as [(String, DraftStrategy?)] {
            var parameters = GenerateParameters(maxTokens: 48, temperature: 0)
            parameters.draftStrategy = strategy
            var info: GenerateCompletionInfo?
            for await item in try MLXLMCommon.generate(
                input: try await prepare(prompts[0].1), parameters: parameters, context: ctx)
            {
                if case .info(let i) = item { info = i }
            }
            let stats = try XCTUnwrap(info).dflash2Stats
            print(
                "[raptor-live] completion info \(label): dflash2Stats=\(String(describing: stats))")
            if strategy == nil {
                XCTAssertNil(stats)
            } else {
                XCTAssertEqual(stats?.blockSize, 5)
                XCTAssertGreaterThan(stats?.verifyCalls ?? 0, 0)
            }
        }

        // Throughput: one warm-up, then ≥3 probes per arm, interleaved,
        // median reported. Greedy and the bundle's own sampler.
        let probes = Int(ProcessInfo.processInfo.environment["VMLX_SPARK25_DFLASH_PROBES"] ?? "3")!
        let input = try await prepare(prompts[0].1)
        var sampled = GenerateParameters(
            generationConfig: ctx.configuration.generationDefaults,
            fallback: GenerateParameters(maxTokens: 256))
        let speedTokens = Int(
            ProcessInfo.processInfo.environment["VMLX_SPARK25_DFLASH_SPEED_TOKENS"] ?? "256")!
        sampled.maxTokens = speedTokens
        sampled.prefillStepSize = 1024
        var greedy256 = greedy
        greedy256.maxTokens = speedTokens
        let arms: [(String, GenerateParameters, Int?)] = [
            ("greedy-plain", greedy256, -1), ("greedy-b8", greedy256, 8),
            ("greedy-b6", greedy256, 6), ("greedy-b5", greedy256, 5), ("greedy-b4", greedy256, 4),
            ("sampled-plain", sampled, -1), ("sampled-b8", sampled, 8), ("sampled-b5", sampled, 5),
            ("sampled-b4", sampled, 4),
        ]
        let armFilter = ProcessInfo.processInfo.environment["VMLX_SPARK25_DFLASH_ARMS"].map {
            Set($0.split(separator: ",").map(String.init))
        }
        let selectedArms = arms.filter { armFilter?.contains($0.0) ?? true }
        var rates: [String: [Double]] = [:]
        var acceptance: [String: [Double]] = [:]
        for probe in 0 ... probes {
            for (name, parameters, block) in selectedArms {
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
                            "[raptor-speed] probe %d %-13@ %4d tok %6.2f tok/s accLen %.2f  per-cycle ms draft %.2f verify %.2f commit %.2f  cycles %d ar %d  paused %d tok @ %.1f tok/s",
                        probe, name as NSString, run.tokens.count, rate, st?.acceptanceLength ?? 1,
                        (st?.draftSeconds ?? 0) / cycles * 1000,
                        (st?.verifySeconds ?? 0) / cycles * 1000,
                        (st?.commitSeconds ?? 0) / cycles * 1000, st?.verifyCalls ?? 0,
                        st?.autoregressiveFallbackTokens ?? 0, st?.throughputPausedTokens ?? 0,
                        Double(st?.throughputPausedTokens ?? 0)
                            / Swift.max(st?.throughputPausedSeconds ?? 0, 1e-6)))
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
        for (name, _, _) in selectedArms {
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

    /// Compare the same user request with and without an exported app system
    /// prompt. An optional JSON token array replays the exact rendered app input.
    /// Private app content stays outside the repository.
    func testAppPromptAcceptance() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let bundle = Self.bundle,
            let systemPath = env["VMLX_SPARK25_DFLASH_SYSTEM_PROMPT"]
        else {
            throw XCTSkip("Set bundle and VMLX_SPARK25_DFLASH_SYSTEM_PROMPT")
        }
        let system = try String(contentsOfFile: systemPath, encoding: .utf8)
        XCTAssertFalse(system.isEmpty)
        let context = try await MLXLMCommon.loadModel(
            from: bundle, using: #huggingFaceTokenizerLoader())
        nonisolated(unsafe) let ctx = context
        let drafter = try DFlash2DrafterResolver.shared.drafter(
            at: bundle.appendingPathComponent("dflash"))
        let user =
            "Write a Python function that merges two sorted lists, then explain its complexity."
        var inputs: [(String, LMInput)] = []
        for (name, messages) in [
            ("short", [Chat.Message.user(user)]),
            ("app-system", [.system(system), .user(user)]),
        ] {
            inputs.append((name, try await ctx.processor.prepare(input: UserInput(chat: messages))))
        }
        if let path = env["VMLX_SPARK25_DFLASH_REPLAY_TOKENS"] {
            let ids = try JSONDecoder().decode(
                [Int32].self, from: Data(contentsOf: URL(fileURLWithPath: path)))
            guard !ids.isEmpty, ids.allSatisfy({ $0 >= 0 && Int($0) < drafter.config.vocabSize })
            else {
                throw NSError(
                    domain: "DFlashReplay", code: 1,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "Replay tokens are empty or outside the vocabulary"
                    ])
            }
            inputs.append(("app-replay", LMInput(tokens: MLXArray(ids))))
        }
        var parameters = GenerateParameters(
            generationConfig: ctx.configuration.generationDefaults,
            fallback: GenerateParameters(maxTokens: 256))
        parameters.maxTokens = 256
        parameters.prefillStepSize = 1024
        let probes = max(3, Int(env["VMLX_SPARK25_DFLASH_PROBES"] ?? "3") ?? 3)
        var rows: [[String: Any]] = []
        for probe in 0 ... probes {
            for (name, input) in inputs {
                // Reverse order on alternating probes to expose thermal drift.
                for speculative in (probe.isMultiple(of: 2) ? [false, true] : [true, false]) {
                    Memory.clearCache()
                    if let seed = env["VMLX_SPARK25_DFLASH_SEED"].flatMap(UInt64.init) {
                        parameters.randomSeed = seed + UInt64(probe)
                        MLXRandom.seed(seed + UInt64(probe))
                    }
                    let run =
                        try speculative
                        ? dflash(ctx, drafter, input, parameters, block: 5)
                        : plain(ctx, input, parameters)
                    let rate = Double(run.tokens.count) / max(run.decodeSeconds, 1e-6)
                    let row: [String: Any] = [
                        "probe": probe, "warmup": probe == 0, "input": name,
                        "prompt_tokens": input.text.tokens.size,
                        "path": speculative ? "dflash" : "plain", "tokens": run.tokens,
                        "tok_s": rate, "acceptance_length": run.stats?.acceptanceLength ?? 1,
                        "verify_calls": run.stats?.verifyCalls ?? 0,
                        "temperature": parameters.temperature, "top_p": parameters.topP,
                    ]
                    rows.append(row)
                    print(
                        "[app-acceptance] probe=\(probe) input=\(name) dflash=\(speculative) prompt=\(input.text.tokens.size) tok/s=\(rate) acceptance=\(run.stats?.acceptanceLength ?? 1)"
                    )
                }
            }
        }
        if let out = env["VMLX_SPARK25_DFLASH_OUT"] {
            let url = URL(fileURLWithPath: out).appendingPathComponent(
                "app-acceptance-\(UUID().uuidString).json")
            try JSONSerialization.data(
                withJSONObject: rows, options: [.prettyPrinted, .sortedKeys]
            ).write(to: url)
        }
    }

    func testDiskContextSurvivesAuxiliaryChurn() async throws {
        guard let bundle = Self.bundle,
            let replay = ProcessInfo.processInfo.environment["VMLX_SPARK25_DFLASH_REPLAY_TOKENS"]
        else { throw XCTSkip("Provide the bundle and an app-sized token replay") }
        let ids = try JSONDecoder().decode([Int32].self, from: Data(contentsOf: URL(fileURLWithPath: replay)))
        XCTAssertGreaterThan(ids.count, 2048)
        let context = try await MLXLMCommon.loadModel(from: bundle, using: #huggingFaceTokenizerLoader())
        nonisolated(unsafe) let ctx = context
        let drafter = try DFlash2DrafterResolver.shared.drafter(at: bundle.appendingPathComponent("dflash"))
        let input = LMInput(tokens: MLXArray(ids))
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("dflash-live-disk-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let config = CacheCoordinatorConfig(usePagedCache: false, enableDiskCache: true,
                                           diskCacheMaxGB: 2, diskCacheDir: root,
                                           modelKey: "spark25-dflash-live-disk")
        var parameters = GenerateParameters(generationConfig: ctx.configuration.generationDefaults,
                                            fallback: GenerateParameters(maxTokens: 96))
        parameters.maxTokens = 96
        parameters.prefillStepSize = 1024
        parameters.randomSeed = 417
        let cacheSalt = computeCacheSalt(for: input, parameters: parameters)
        func run(_ name: String, _ input: LMInput, _ coordinator: CacheCoordinator, count: Int) throws -> DFlash2GenerationStats {
            MLXRandom.seed(417)
            var p = parameters
            p.maxTokens = count
            let start = Date.timeIntervalSinceReferenceDate
            var iterator = try DFlash2TokenIterator(input: input, target: ctx.model as! any DFlash2Target,
                                                    drafter: drafter, blockSize: 5, parameters: p,
                                                    cacheCoordinator: coordinator)
            let decodeStart = Date.timeIntervalSinceReferenceDate
            var tokens: [Int] = []
            while tokens.count < count, let token = iterator.next() { tokens.append(token) }
            let decodeSeconds = Date.timeIntervalSinceReferenceDate - decodeStart
            iterator.storeCacheAfterGeneration(generatedTokenIds: tokens, includeGeneratedBoundary: true)
            let stats = iterator.dflash2Stats!
            print("[dflash-disk] name=\(name) prompt=\(input.text.tokens.size) prefill_s=\(decodeStart-start) tok/s=\(Double(tokens.count)/max(decodeSeconds,1e-6)) seeded=\(stats.seededContextRows) recomputed=\(stats.recomputedContextRows) acceptance=\(stats.acceptanceLength)")
            return stats
        }
        drafter.contextStore.removeAll()
        let writer = CacheCoordinator(config: config)
        let cold = try run("cold", input, writer, count: 96)
        for i in 0 ..< 6 {
            let auxiliary = try await ctx.processor.prepare(input: UserInput(chat: [
                .user("Give a short title for a conversation about sorting lists. Variant \(i).")
            ]))
            _ = try run("aux-\(i)", auxiliary.withCachePromptIntent(.auxiliary), writer, count: 32)
        }
        XCTAssertNotNil(drafter.contextStore.rows(endingAt: ids.count, of: ids.map(Int.init), salt: cacheSalt,
                                                 minimumRows: cold.seededContextRows))
        drafter.contextStore.removeAll()
        let reader = CacheCoordinator(config: config)
        guard case .hit(let boundary, _, let detail, _, _, let arrays) = reader.fetch(tokens: ids.map(Int.init), mediaSalt: cacheSalt) else {
            return XCTFail("Expected disk target boundary after reopening")
        }
        XCTAssertEqual(detail, .disk)
        XCTAssertEqual(boundary, ids.count)
        XCTAssertNotNil(try XCTUnwrap(arrays)["dflash2_context_rows"])
        let restored = try run("reopened-disk", input, reader, count: 96)
        XCTAssertEqual(restored.seededContextRows, cold.seededContextRows)
        XCTAssertEqual(restored.recomputedContextRows, 0)
    }

    /// A prompt-cache hit restores the target's prefix without re-running it.
    /// The drafter's context must still cover that prefix, or it drafts from
    /// the new suffix alone. Same request cold and after a hit on a shorter
    /// prompt it extends (partial hit) and on itself (full hit).
    func testDrafterKeepsContextAcrossPromptCacheHits() async throws {
        guard let bundle = Self.bundle else {
            throw XCTSkip("Set VMLX_SPARK25_DFLASH_BUNDLE to a Spark2.5 bundle with dflash/")
        }
        let context = try await MLXLMCommon.loadModel(
            from: bundle, using: #huggingFaceTokenizerLoader())
        nonisolated(unsafe) let ctx = context
        let drafter = try DFlash2DrafterResolver.shared.drafter(
            at: bundle.appendingPathComponent("dflash"))
        let system = (1 ... 40).map {
            "Rule \($0): when asked about item \($0 % 9), answer with its code, K-\($0 * 7)."
        }.joined(separator: " ")
        func prepare(_ turns: [Chat.Message]) async throws -> LMInput {
            try await ctx.processor.prepare(input: UserInput(chat: turns))
        }
        let first = try await prepare([
            .system(system), .user("Write a Python function that reverses a linked list."),
        ])
        let second = try await prepare([
            .system(system), .user("Write a Python function that reverses a linked list."),
            .assistant(
                "def reverse(head):\n    prev = None\n    while head:\n        head.next, prev, head = prev, head, head.next\n    return prev"
            ),
            .user("Now write the recursive version and explain its stack depth."),
        ])
        var p = GenerateParameters(maxTokens: 384, temperature: 0)
        p.prefillStepSize = 1024
        func coordinator() -> CacheCoordinator {
            CacheCoordinator(
                config: CacheCoordinatorConfig(
                    usePagedCache: true, enableDiskCache: false,
                    modelKey: "spark25-dflash-hit-\(UUID().uuidString)"))
        }
        func run(_ input: LMInput, _ coord: CacheCoordinator?) throws -> DFlash2GenerationStats {
            var it = try DFlash2TokenIterator(
                input: input, target: ctx.model as! any DFlash2Target, drafter: drafter,
                blockSize: nil, parameters: p, cacheCoordinator: coord)
            var ids: [Int] = []
            let decodeStart = Date.timeIntervalSinceReferenceDate
            while ids.count < 384, let t = it.next() { ids.append(t) }
            let decodeSeconds = Date.timeIntervalSinceReferenceDate - decodeStart
            print("[dflash-hit-speed] tokens=\(ids.count) tok/s=\(Double(ids.count) / max(decodeSeconds, 1e-6))")
            it.storeCacheAfterGeneration(generatedTokenIds: ids, includeGeneratedBoundary: true)
            return it.dflash2Stats!
        }
        let cold = try run(second, nil)
        let warmCoord = coordinator()
        _ = try run(first, warmCoord)
        let partial = try run(second, warmCoord)
        let full = try run(second, warmCoord)
        // A prefix restored with nothing kept for it (the disk tier after a
        // relaunch): the rows are recomputed.
        drafter.contextStore.removeAll()
        let recomputed = try run(second, warmCoord)
        let promptRows = second.text.tokens.size
        for (name, s) in [
            ("cold", cold), ("partial-hit", partial), ("full-hit", full),
            ("recomputed", recomputed),
        ] {
            print(
                String(
                    format:
                        "[dflash-hit] %-11@ prompt=%d seededContextRows=%d recomputed=%d tokPerVerify=%.2f accepted/drafted=%d/%d",
                    name as NSString, promptRows, s.seededContextRows, s.recomputedContextRows,
                    s.acceptanceLength,
                    s.acceptedTokens, s.draftedTokens))
        }
        XCTAssertGreaterThan(
            partial.acceptanceLength, cold.acceptanceLength * 0.9,
            "a partial prompt-cache hit must not starve the drafter of context")
        XCTAssertGreaterThan(
            full.acceptanceLength, cold.acceptanceLength * 0.9,
            "a full prompt-cache hit must not starve the drafter of context")
        XCTAssertGreaterThan(recomputed.recomputedContextRows, 0)
        XCTAssertGreaterThan(
            recomputed.acceptanceLength, cold.acceptanceLength * 0.85,
            "rows recomputed for an unkept prefix must restore most of the context")
    }
}
