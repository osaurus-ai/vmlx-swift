import Foundation
import MLX
import XCTest
@preconcurrency import VMLXTokenizers
@testable import MLXHuggingFace
@testable import MLXLLM
@testable import MLXLMCommon

/// Opt-in real-bundle evidence. Exact rendered prompt IDs are shared with the
/// BF16 comparison; no test rewrites model sampling defaults on disk.
final class DiplodocusQualificationTests: XCTestCase {
    private struct Probe: Decodable {
        var name: String
        var effort: String
        var ids: [Int]
        var max_tokens: Int
    }

    func testRealBundleGreedyAndDFlash() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["DIPLODOC_TARGET"], let probesPath = env["DIPLODOC_PROBES"],
            let outputPath = env["DIPLODOC_OUTPUT"]
        else { throw XCTSkip("Set DIPLODOC_TARGET, DIPLODOC_PROBES and DIPLODOC_OUTPUT") }
        let context = try await MLXLMCommon.loadModel(
            from: URL(fileURLWithPath: path), using: #huggingFaceTokenizerLoader())
        nonisolated(unsafe) let ctx = context
        let target = try XCTUnwrap(ctx.model as? any DFlash2Target)
        let drafter = try env["DIPLODOC_DRAFTER"].map {
            try DFlash2Loader.load(from: URL(fileURLWithPath: $0))
        }
        var probes = try JSONDecoder().decode(
            [Probe].self, from: Data(contentsOf: URL(fileURLWithPath: probesPath)))
        if let names = env["DIPLODOC_CASES"] {
            let wanted = Set(names.split(separator: ",").map(String.init))
            probes = probes.filter { wanted.contains($0.name) }
        }
        XCTAssertFalse(probes.isEmpty)
        let directory = URL(fileURLWithPath: outputPath)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // DFlash installs optimized target projections in place. Capture every
        // ordinary AR baseline before that mutation, so later cases cannot
        // accidentally compare DFlash with an already-modified AR target.
        var baselines: [String: [Int]] = [:]
        for arm in (drafter == nil ? ["ar"] : ["ar", "dflash2"]) {
          for probe in probes {
            let cap = Int(env["DIPLODOC_MAX_TOKENS"] ?? "") ?? probe.max_tokens
            var parameters = GenerateParameters(
                generationConfig: ctx.configuration.generationDefaults,
                fallback: GenerateParameters(maxTokens: cap, prefillStepSize: 1024))
            parameters.maxTokens = cap
            parameters.prefillStepSize = 1024
            parameters.temperature = 0  // explicit matched greedy diagnostic
            parameters.topP = 1
            parameters.topK = 0
            parameters.minP = 0
            parameters.draftStrategy = nil
            let input = LMInput(tokens: MLXArray(probe.ids), tokenIds: probe.ids)
            var baseline = baselines[probe.name] ?? []
                let start = Date()
                var first: TimeInterval?
                var ids: [Int] = []
                var statistics: [String: Any] = [:]
                if arm == "ar" {
                    var iterator = try TokenIterator(
                        input: input, model: target, parameters: parameters)
                    while ids.count < cap, let token = iterator.next() {
                        if first == nil { first = Date().timeIntervalSince(start) }
                        ids.append(token)
                        if token == 1 || token == 250019 { break }
                    }
                    baseline = ids
                    baselines[probe.name] = ids
                } else {
                    var iterator = try DFlash2TokenIterator(
                        input: input, target: target, drafter: drafter!, blockSize: 8,
                        parameters: parameters, cacheCoordinator: nil)
                    while ids.count < cap, let token = iterator.next() {
                        if first == nil { first = Date().timeIntervalSince(start) }
                        ids.append(token)
                        if token == 1 || token == 250019 { break }
                    }
                    let s = iterator.dflash2Stats
                    statistics = ["drafted": s.draftedTokens, "accepted": s.acceptedTokens,
                        "sequential_target_steps": s.sequentialVerificationSteps,
                        "verify_calls": s.verifyCalls, "emitted": s.emittedTokens,
                        "acceptance_length": s.acceptanceLength, "fallback_tokens": s.autoregressiveFallbackTokens,
                        "draft_seconds": s.draftSeconds, "verify_seconds": s.verifySeconds,
                        "commit_seconds": s.commitSeconds, "block_size": s.blockSize]
                }
                let seconds = Date().timeIntervalSince(start)
                let stopped = ids.last == 1 || ids.last == 250019
                let divergence = zip(baseline, ids).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                var divergenceDiagnostic: [String: Any] = [:]
                if arm == "dflash2", let divergence {
                    // Diagnostic only: a fresh chunked replay is not the original
                    // verifier execution and cannot waive a token mismatch.
                    let prefix = probe.ids + baseline.prefix(divergence)
                    let cache = target.newCache(parameters: parameters)
                    var last: MLXArray?
                    for offset in stride(from: 0, to: prefix.count, by: 1024) {
                        let end = min(prefix.count, offset + 1024)
                        let logits = target(
                            MLXArray(Array(prefix[offset..<end])).expandedDimensions(axis: 0),
                            cache: cache)
                        last = logits[0, -1].asType(.float32)
                        eval(last!)
                    }
                    let ranked = argSort(-last!)[0..<5].asArray(Int.self)
                    let values = last!.asArray(Float.self)
                    divergenceDiagnostic = [
                        "scope": "fresh chunked replay; original mismatch remains a failure",
                        "top_ids": ranked, "top_logits": ranked.map { values[$0] },
                        "ar_token": baseline[divergence], "speculative_token": ids[divergence],
                        "replay_top2_margin": values[ranked[0]] - values[ranked[1]],
                    ]
                }
                let record: [String: Any] = [
                    "target": path, "drafter": env["DIPLODOC_DRAFTER"] ?? "",
                    "case": probe.name, "effort": probe.effort, "arm": arm,
                    "prompt_tokens": probe.ids.count, "ids": ids,
                    "text": ctx.tokenizer.decode(tokenIds: ids), "seconds": seconds,
                    "ttft_seconds": first ?? seconds,
                    "total_tokens_per_second": Double(ids.count) / max(seconds, 0.001),
                    "decode_tokens_per_second": Double(max(0, ids.count - 1)) / max(seconds - (first ?? seconds), 0.001),
                    "finish": stopped ? "stop" : "length", "statistics": statistics,
                    "exact_greedy_match": ids == baseline,
                    "first_divergence": divergence.map { $0 as Any } ?? NSNull(),
                    "divergence_diagnostic": divergenceDiagnostic,
                    "sampling": ["temperature": 0, "top_p": 1, "top_k": 0, "min_p": 0],
                ]
                let data = try JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys])
                try data.write(to: directory.appendingPathComponent("\(probe.name)-\(arm).json"), options: .atomic)
                print("DIPLODOC case=\(probe.name) arm=\(arm) tokens=\(ids.count) seconds=\(seconds) stopped=\(stopped) exact=\(ids == baseline)")
                XCTAssertFalse(ids.isEmpty)
                if env["DIPLODOC_REQUIRE_STOP"] == "1" { XCTAssertTrue(stopped, probe.name) }
                if arm == "dflash2" {
                    XCTAssertTrue(ids == baseline, "\(probe.name): greedy divergence at \(String(describing: divergence)); inspect receipt")
                }
                Memory.clearCache()
            }
        }
    }
}
