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

    /// Opt-in: native sampling (bundle generation defaults) through ordinary
    /// decoding and DFlash2. Sampled verification is batched; this records
    /// throughput, acceptance and per-phase seconds. Set
    /// VMLX_DFLASH2_SYNC_PHASE_TIMING=1 for GPU-accurate phase seconds.
    func testRealBundleSampledDFlash() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["DIPLODOC_TARGET"], let drafterPath = env["DIPLODOC_DRAFTER"],
            let probesPath = env["DIPLODOC_PROBES"], let outputPath = env["DIPLODOC_OUTPUT"],
            env["DIPLODOC_SAMPLED"] == "1"
        else { throw XCTSkip("Set DIPLODOC_SAMPLED=1 with the DIPLODOC_* paths") }
        let context = try await MLXLMCommon.loadModel(
            from: URL(fileURLWithPath: path), using: #huggingFaceTokenizerLoader())
        nonisolated(unsafe) let ctx = context
        let target = try XCTUnwrap(ctx.model as? any DFlash2Target)
        let drafter = try DFlash2Loader.load(from: URL(fileURLWithPath: drafterPath))
        var probes = try JSONDecoder().decode(
            [Probe].self, from: Data(contentsOf: URL(fileURLWithPath: probesPath)))
        if let names = env["DIPLODOC_CASES"] {
            let wanted = Set(names.split(separator: ",").map(String.init))
            probes = probes.filter { wanted.contains($0.name) }
        }
        XCTAssertFalse(probes.isEmpty)
        let directory = URL(fileURLWithPath: outputPath)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let rounds = Int(env["DIPLODOC_ROUNDS"] ?? "") ?? 1
        // AR first for every probe: DFlash installs the lane matmul in place.
        for arm in ["ar", "dflash2"] {
            for round in 0 ..< rounds {
                for probe in probes {
                    let cap = Int(env["DIPLODOC_MAX_TOKENS"] ?? "") ?? probe.max_tokens
                    var parameters = GenerateParameters(
                        generationConfig: ctx.configuration.generationDefaults,
                        fallback: GenerateParameters(maxTokens: cap, prefillStepSize: 1024))
                    parameters.maxTokens = cap
                    parameters.prefillStepSize = 1024
                    parameters.draftStrategy = nil
                    XCTAssertGreaterThan(parameters.temperature, 0, "bundle defaults must be sampled")
                    let input = LMInput(tokens: MLXArray(probe.ids), tokenIds: probe.ids)
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
                    } else {
                        var iterator = try DFlash2TokenIterator(
                            input: input, target: target, drafter: drafter,
                            blockSize: Int(env["DIPLODOC_BLOCK"] ?? "") ?? 8,
                            parameters: parameters, cacheCoordinator: nil)
                        while ids.count < cap, let token = iterator.next() {
                            if first == nil { first = Date().timeIntervalSince(start) }
                            ids.append(token)
                            if token == 1 || token == 250019 { break }
                        }
                        let s = iterator.dflash2Stats
                        statistics = [
                            "drafted": s.draftedTokens, "accepted": s.acceptedTokens,
                            "sequential_target_steps": s.sequentialVerificationSteps,
                            "verify_calls": s.verifyCalls, "emitted": s.emittedTokens,
                            "acceptance_length": s.acceptanceLength,
                            "fallback_tokens": s.autoregressiveFallbackTokens,
                            "draft_seconds": s.draftSeconds, "verify_seconds": s.verifySeconds,
                            "commit_seconds": s.commitSeconds, "block_size": s.blockSize,
                        ]
                    }
                    let seconds = Date().timeIntervalSince(start)
                    let decode = Double(max(0, ids.count - 1)) / max(seconds - (first ?? seconds), 0.001)
                    let record: [String: Any] = [
                        "case": probe.name, "arm": arm, "round": round, "tokens": ids.count,
                        "seconds": seconds, "ttft_seconds": first ?? seconds,
                        "decode_tokens_per_second": decode, "statistics": statistics,
                        "text": ctx.tokenizer.decode(tokenIds: ids),
                        "sampling": ["temperature": parameters.temperature, "top_p": parameters.topP],
                    ]
                    let data = try JSONSerialization.data(
                        withJSONObject: record, options: [.prettyPrinted, .sortedKeys])
                    try data.write(
                        to: directory.appendingPathComponent("\(probe.name)-\(arm)-r\(round).json"),
                        options: .atomic)
                    print("DIPLODOC_SAMPLED case=\(probe.name) arm=\(arm) round=\(round) tokens=\(ids.count) decode=\(decode)")
                    XCTAssertFalse(ids.isEmpty)
                    Memory.clearCache()
                }
            }
        }
    }

    /// Opt-in component costs for one DFlash2 cycle on the real bundle:
    /// target LM head over the 7 draft rows, the 250k-vocab partition, the
    /// drafter backbone, and multi-row target forwards (1...8 rows). Every
    /// timing drains the GPU; medians of DIPLODOC_ITERS runs after warm-up.
    func testRealBundleDFlashComponentCosts() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["DIPLODOC_TARGET"], let drafterPath = env["DIPLODOC_DRAFTER"],
            env["DIPLODOC_COMPONENTS"] == "1"
        else { throw XCTSkip("Set DIPLODOC_COMPONENTS=1 with DIPLODOC_TARGET/DRAFTER") }
        let context = try await MLXLMCommon.loadModel(
            from: URL(fileURLWithPath: path), using: #huggingFaceTokenizerLoader())
        nonisolated(unsafe) let ctx = context
        let target = try XCTUnwrap(ctx.model as? any DFlash2Target)
        let drafter = try DFlash2Loader.load(from: URL(fileURLWithPath: drafterPath))
        let iters = Int(env["DIPLODOC_ITERS"] ?? "") ?? 30
        func median(_ body: () -> MLXArray) -> Double {
            for _ in 0 ..< 3 { eval(body()) }
            var times: [Double] = []
            for _ in 0 ..< iters {
                let t0 = Date()
                eval(body())
                times.append(Date().timeIntervalSince(t0) * 1000)
            }
            return times.sorted()[times.count / 2]
        }
        let hidden = MLXRandom.normal([1, 7, 4096]).asType(.bfloat16)
        let head = median { target.projectToLogits(hidden) }
        let logits = target.projectToLogits(hidden)
        eval(logits)
        let vocab = logits.dim(-1)
        let partition = median { argPartition(logits, kth: vocab - 16, axis: -1) }
        // Prefill a real prompt so the target cache and captured context exist.
        let ids = (0 ..< 512).map { 1000 + ($0 * 37) % 20000 }
        let cache = target.newCache(parameters: nil)
        let layerIDs = drafter.config.dflash.targetLayerIds
        let (_, captured) = target.callAsFunction(
            MLXArray(ids).expandedDimensions(axis: 0), cache: cache,
            captureLayerIDs: Set(layerIDs), recordPrefixCommitStates: false)
        let context0 = extractContextFeature(captured: captured, targetLayerIDs: layerIDs)
        eval(context0)
        let draftCache = drafter.makeCache()
        let block = MLXArray([Int32(ids.last!)] + Array(repeating: Int32(250062), count: 7)).reshaped(1, 8)
        // Prime the drafter cache with the whole context once, as the first
        // cycle after prefill does; steady cycles then add ~2 committed rows.
        eval(drafter.hiddenStates(
            inputs: block, targetHidden: context0, cache: draftCache, embedder: target,
            logitsStart: 1))
        let primed = draftCache.map(\.offset)
        let steadyContext = context0[0..., (-2)..., 0...]
        let backbone = median {
            for (c, o) in zip(draftCache, primed) where c.isTrimmable && c.offset > o { _ = c.trim(c.offset - o) }
            return drafter.hiddenStates(
                inputs: block, targetHidden: steadyContext, cache: draftCache, embedder: target,
                logitsStart: 1)
        }
        let backboneFull = median {
            for (c, o) in zip(draftCache, primed) where c.isTrimmable && c.offset > o { _ = c.trim(c.offset - o) }
            return drafter.propose(
                inputs: block, targetHidden: steadyContext, cache: draftCache, embedder: target,
                temperature: 1.0, logitsStart: 1).tokens
        }
        // The drafter's own projections (8-bit, group 64) are lane-eligible.
        let drafterLane = LaneQMM.install(model: drafter)
        let backboneLane = median {
            for (c, o) in zip(draftCache, primed) where c.isTrimmable && c.offset > o { _ = c.trim(c.offset - o) }
            return drafter.hiddenStates(
                inputs: block, targetHidden: steadyContext, cache: draftCache, embedder: target,
                logitsStart: 1)
        }
        let proposeLane = median {
            for (c, o) in zip(draftCache, primed) where c.isTrimmable && c.offset > o { _ = c.trim(c.offset - o) }
            return drafter.propose(
                inputs: block, targetHidden: steadyContext, cache: draftCache, embedder: target,
                temperature: 1.0, logitsStart: 1).tokens
        }
        print("DIPLODOC_DRAFTER_LANE lane=\(drafterLane.lane) tiled=\(drafterLane.tiled) skipped=\(drafterLane.skipped) backbone_ms=\(String(format: "%.2f", backboneLane)) propose_ms=\(String(format: "%.2f", proposeLane))")
        var rows: [String] = []
        for (pass, width) in [("mlx", 1), ("mlx", 2), ("mlx", 4), ("mlx", 8), ("lane", 1), ("lane", 2), ("lane", 4), ("lane", 8)] {
            if pass == "lane", width == 1 { LaneQMM.installForDFlash2Target(target) }
            let input = MLXArray(Array(ids.suffix(width))).reshaped(1, width)
            let ms = median {
                let out = target.callAsFunction(
                    input, cache: cache, captureLayerIDs: Set(layerIDs),
                    recordPrefixCommitStates: false).0
                for c in cache { _ = c.trim(width) }
                return out
            }
            rows.append("\(pass)-w\(width)=\(String(format: "%.2f", ms))")
        }
        print("DIPLODOC_COMPONENTS head7_ms=\(String(format: "%.2f", head)) partition7_ms=\(String(format: "%.2f", partition)) drafter_backbone_ms=\(String(format: "%.2f", backbone)) drafter_propose_ms=\(String(format: "%.2f", backboneFull)) primed_offset=\(primed.first ?? -1) lane_report=\(LaneQMM.lastInstallReport.map { "lane \($0.lane) skipped \($0.skipped)" } ?? "none") target_rows_ms \(rows.joined(separator: " ")) lane=\(env["VMLX_LANE_QMM"] ?? "default")")
    }
}
