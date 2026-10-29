// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Test harness (not shipped): the agentic prefix-cache loop and the long-context runs on the real MiniMax-M3.1 bundle.

import CryptoKit
import Foundation
import MLX
import MLXNN

@testable import MLXLLM
@testable import MLXLMCommon

// MARK: - recorder

final class LoopRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var _logits: [MLXArray] = []
    private var _restores: [AlignedPrefillRestoreRecord] = []
    private var _prefills: [AlignedPrefillPrefillRecord] = []
    var logits: [MLXArray] { lock.withLock { _logits } }
    var restores: [AlignedPrefillRestoreRecord] { lock.withLock { _restores } }
    var prefills: [AlignedPrefillPrefillRecord] { lock.withLock { _prefills } }

    func install() {
        AlignedPrefill.hooks.logitsObserver = { [self] x in
            let row = x[0, -1].asType(.float32)
            eval(row)
            lock.withLock { _logits.append(row) }
        }
        AlignedPrefill.hooks.restoreObserver = { [self] r in lock.withLock { _restores.append(r) } }
        AlignedPrefill.hooks.prefillObserver = { [self] r in lock.withLock { _prefills.append(r) } }
    }
}

struct LoopRun {
    let tokens: [Int]
    let logits: [MLXArray]
    let restores: [AlignedPrefillRestoreRecord]
    let prefills: [AlignedPrefillPrefillRecord]
    let prefillMs: Double
    let totalMs: Double
    let stats: DSparkGenerationStats?
    var cached: Int { prefills.first?.start ?? -1 }
}

func loopRun(
    _ model: MiniMaxM3Model, _ input: LMInput, coordinator: CacheCoordinator?, recorder: LoopRecorder,
    maxTokens: Int, drafter: URL?
) throws -> LoopRun {
    let before = (recorder.logits.count, recorder.restores.count, recorder.prefills.count)
    var parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0, prefillStepSize: 2048)
    if let drafter { parameters.draftStrategy = .dspark(drafterPath: drafter) }
    let t0 = nowMs()
    var iterator = try TokenIterator(input: input, model: model, parameters: parameters, cacheCoordinator: coordinator)
    var out: [Int] = []
    var t1 = t0
    while let token = iterator.next() {
        if out.isEmpty { t1 = nowMs() }
        out.append(token)
        if token == eosToken { break }
    }
    let t2 = nowMs()
    if coordinator != nil {
        iterator.storeCacheAfterGeneration(generatedTokenIds: out, includeGeneratedBoundary: false)
    }
    return LoopRun(
        tokens: out, logits: Array(recorder.logits[before.0...]), restores: Array(recorder.restores[before.1...]),
        prefills: Array(recorder.prefills[before.2...]), prefillMs: t1 - t0, totalMs: t2 - t0,
        stats: iterator.dsparkGenerationStats)
}

func lcp(_ a: [Int], _ b: [Int]) -> Int {
    var i = 0
    while i < min(a.count, b.count), a[i] == b[i] { i += 1 }
    return i
}

/// Deterministic terminal output of about `lines` lines (build / test / ls / grep style).
func terminalOutput(lines: Int, seed: Int) -> String {
    var rng = UInt64(seed &* 2_654_435_761 &+ 97)
    func next() -> Int {
        rng = rng &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Int((rng >> 33) % 10_000)
    }
    let modules = ["CacheKit", "NetCore", "Router", "Storage", "Telemetry", "Auth"]
    var out: [String] = []
    for i in 0 ..< lines {
        let m = modules[next() % modules.count]
        switch next() % 5 {
        case 0:
            out.append("/work/repo/Sources/\(m)/File\(next() % 40).swift:\(next() % 400 + 1):\(next() % 60 + 1): warning: "
                + "initialization of immutable value 'tmp\(i)' was never used; consider replacing with '_'")
        case 1:
            out.append("Test Case '-[\(m)Tests.\(m)Tests test\(next() % 90)]' passed (\(Double(next() % 900) / 1000) seconds).")
        case 2:
            out.append("-rw-r--r--  1 eric  staff  \(next() * 7 % 99999) Oct 28 1\(next() % 10):\(next() % 50 + 10) \(m.lowercased())_\(next() % 300).json")
        case 3:
            out.append("[\(next() % 900 + 100)/\(1000)] Compiling \(m) \(m)Store\(next() % 30).swift")
        default:
            out.append("grep: ./build/\(m)/obj\(next() % 70).o: binary file matches (pid \(next() % 90000 + 1000))")
        }
    }
    return out.joined(separator: "\n")
}

// MARK: - cacheloop

func taskCacheLoop() async throws {
    let dir = bundleURL()
    let (model, context, _) = try await productionLoad(dir)
    let lane = env["T5_LANE"] ?? "plain"
    let drafter = lane == "dspark" ? try attachDrafter(model, dir) : nil
    if lane == "dspark" && drafter == nil { fatalError("no drafter resolved") }
    let maxTokens = Int(env["T5_MAX_TOKENS"] ?? "48")!
    let step = Int(env["T5_STEP"] ?? "2048")!
    let markers = model.alignedPrefillTurnMarkers
    log("cacheloop \(lane): step \(step), turn markers \(markers.map(String.init(describing:)) ?? "none (v1 rule)"), schema \(model.alignedPrefillSchema ?? "off")")
    /// What the rule allows a request reusing `common` shared rows of a prompt stored with boundary `previousK`.
    func ruleReuse(_ ids: [Int], k: Int, common: Int, previousK: Int) -> Int {
        let cuts = AlignedPrefillGeometry.turnCuts(ids, markers: markers, end: k)
        return AlignedPrefillGeometry.reuseLimit(
            matched: min(common, previousK), boundary: k, step: step, exactGeometry: false, turnCuts: cuts)
    }
    let diskDir = outDir.appendingPathComponent("diskcache-\(lane)-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: diskDir) }
    var config = VMLXServerRuntimeSettings().cacheCoordinatorConfig(
        modelKey: "minimax-m31-real-\(lane)", diskCacheDirectory: diskDir)
    config.enableDiskCache = true
    let coordinator = CacheCoordinator(config: config)
    let suffix = generationPromptSuffix(context.tokenizer)
    coordinator.setGenPromptSuffixTokens(suffix)
    let recorder = LoopRecorder()
    recorder.install()

    // Terminal-agent loop: one scripted shell call per turn, tool results of different sizes.
    let toolLines = [21, 13, 50, 3, 82, 33]
    let commands = [
        "swift build 2>&1 | tail -40", "ls -la Sources/CacheKit", "swift test 2>&1 | tail -80",
        "git status --short", "grep -rn \"DiskStore\" Sources Tests", "cat Package.swift",
    ]
    var messages: [Chat.Message] = [
        .system(
            "You are an autonomous coding agent working in a Linux terminal on the repository at /work/repo. Use "
                + "run_shell to inspect and change it. Think briefly, then act."),
        .user("The release build of CacheKit fails and two tests are flaky. Investigate and fix the build first."),
    ]
    var report: [String: Any] = [
        "lane": lane, "max_tokens": maxTokens, "step": step, "gen_suffix": suffix,
        "turn_markers": markers.map(String.init(describing:)) ?? "none", "schema": model.alignedPrefillSchema ?? "off",
    ]
    var turns: [[String: Any]] = []
    var previous: [Int] = []
    var previousK = 0
    var allSame = true
    func boundary(_ ids: [Int], _ input: LMInput) -> Int {
        AlignedPrefillGeometry.storeBoundary(
            promptTokenIds: ids, cachePrefixTokenCounts: input.cachePrefixTokenCounts ?? [],
            generationPromptSuffix: suffix) ?? 0
    }
    func compare(_ warm: LoopRun, _ cold: LoopRun) -> (Bool, Bool, Float) {
        let tokensSame = warm.tokens == cold.tokens
        // Plain lane: every sampling forward's logits; DSpark lane: the prompt's last row (the governor's
        // wall-clock choices make the later forward shapes differ between arms; the tokens must not).
        let n = lane == "plain" ? max(warm.logits.count, cold.logits.count) : 1
        var logitsSame = warm.logits.count >= n && cold.logits.count >= n
        var gap: Float = 0
        if logitsSame {
            for i in 0 ..< n {
                let d = abs(warm.logits[i] - cold.logits[i]).max().item(Float.self)
                gap = max(gap, d)
                if !(warm.logits[i].shape == cold.logits[i].shape && arrayEqual(warm.logits[i], cold.logits[i]).item(Bool.self)) {
                    logitsSame = false
                }
            }
        }
        return (tokensSame, logitsSame, gap)
    }
    var lastInput: LMInput?
    for turn in 0 ..< 6 {
        let input = try await prepare(context, messages, tools: agentTools)
        let ids = tokenIds(input)
        let k = boundary(ids, input)
        let warm = try loopRun(model, input, coordinator: coordinator, recorder: recorder, maxTokens: maxTokens, drafter: drafter)
        let cold = try loopRun(model, input, coordinator: nil, recorder: recorder, maxTokens: maxTokens, drafter: drafter)
        let (tokensSame, logitsSame, gap) = compare(warm, cold)
        let common = lcp(ids, previous)
        let expectedMax = turn == 0 ? 0 : ruleReuse(ids, k: k, common: common, previousK: previousK)
        let ok = tokensSame && logitsSame && cold.cached == 0 && !cold.tokens.isEmpty
        allSame = allSame && ok
        turns.append([
            "turn": turn + 1, "prompt": ids.count, "K": k, "lcp_with_previous_prompt": common,
            "previous_K": previousK, "reuse": warm.cached, "rule_max_reuse": expectedMax,
            "reuse_is_chunk_cut_or_K": warm.cached % step == 0 || warm.cached == k
                || AlignedPrefillGeometry.turnCuts(ids, markers: markers, end: k).contains(warm.cached),
            "v1_rule_reuse": turn == 0 ? 0 : (min(common, previousK) / step) * step,
            "reuse_equals_rule": warm.cached == expectedMax, "prefilled": ids.count - max(0, warm.cached),
            "restore_details": warm.restores.map {
                "\($0.detail) matched \($0.matched) reuse \($0.reuse) exact \($0.exactGeometry) offsets \(Set($0.offsets)) "
                    + "lanes \(Set($0.laneRows))"
            },
            "restore_lanes_lockstep": warm.restores.allSatisfy { r in
                r.reuse == 0 || (Set(r.offsets) == [r.reuse] && Set(r.laneRows) == [r.reuse])
            },
            "pieces": warm.prefills.first.map { $0.pieces.map { "\($0.lowerBound)..<\($0.upperBound)" } } ?? [],
            "warm_prefill_ms": warm.prefillMs, "cold_prefill_ms": cold.prefillMs,
            "warm_total_ms": warm.totalMs, "cold_total_ms": cold.totalMs,
            "tokens_identical": tokensSame, "logits_identical": logitsSame, "logits_compared": lane == "plain" ? warm.logits.count : 1,
            "max_logit_gap": gap, "generated": warm.tokens.count,
            "dspark_warm": statsJSON(warm.stats), "dspark_cold": statsJSON(cold.stats),
        ])
        log("loop \(lane) turn \(turn + 1): prompt \(ids.count) K \(k) lcp \(common) reuse \(warm.cached) (rule \(expectedMax)) prefill warm \(Int(warm.prefillMs)) cold \(Int(cold.prefillMs)) ms; same tokens \(tokensSame) logits \(logitsSame) gap \(gap)")
        previous = ids
        previousK = k
        lastInput = input
        if turn < 5 {
            let text = context.tokenizer.decode(tokenIds: warm.tokens.filter { $0 != eosToken }, skipSpecialTokens: true)
            messages.append(
                Chat.Message(
                    role: .assistant, content: text,
                    toolCalls: [ToolCall(function: .init(name: "run_shell", arguments: ["cmd": commands[turn]]))]))
            messages.append(.tool(terminalOutput(lines: toolLines[turn], seed: turn)))
        }
    }
    report["turns"] = turns
    _ = coordinator.waitForPendingDiskWrites()

    // Exact repeat of turn 6.
    if let lastInput {
        let ids = tokenIds(lastInput)
        let warm = try loopRun(model, lastInput, coordinator: coordinator, recorder: recorder, maxTokens: maxTokens, drafter: drafter)
        let cold = try loopRun(model, lastInput, coordinator: nil, recorder: recorder, maxTokens: maxTokens, drafter: drafter)
        let (t, l, g) = compare(warm, cold)
        report["exact_repeat"] = [
            "restore_lanes_lockstep": warm.restores.allSatisfy { r in
                r.reuse == 0 || (Set(r.offsets) == [r.reuse] && Set(r.laneRows) == [r.reuse])
            },
            "dspark_warm": statsJSON(warm.stats), "prefilled_after_restore": ids.count - warm.cached,
            "prompt": ids.count, "K": boundary(ids, lastInput), "reuse": warm.cached, "tokens_identical": t,
            "logits_identical": l, "max_logit_gap": g, "warm_prefill_ms": warm.prefillMs, "cold_prefill_ms": cold.prefillMs,
        ]
        allSame = allSame && t && l
        log("loop \(lane) exact repeat: reuse \(warm.cached) of \(ids.count); same \(t) \(l)")
    }
    // Partial hit: turn 6 with the last tool result replaced (same prefix, different suffix).
    var variant = messages
    variant[variant.count - 1] = .tool(terminalOutput(lines: 41, seed: 1234))
    let input = try await prepare(context, variant, tools: agentTools)
    let ids = tokenIds(input)
    let common = lcp(ids, previous)
    let warm = try loopRun(model, input, coordinator: coordinator, recorder: recorder, maxTokens: maxTokens, drafter: drafter)
    let cold = try loopRun(model, input, coordinator: nil, recorder: recorder, maxTokens: maxTokens, drafter: drafter)
    let (t, l, g) = compare(warm, cold)
    let rule = ruleReuse(ids, k: boundary(ids, input), common: common, previousK: previousK)
    report["partial_hit"] = [
        "prompt": ids.count, "divergence": common, "divergence_mod_step": common % step, "reuse": warm.cached,
        "rule_max_reuse": rule, "tokens_identical": t, "logits_identical": l, "max_logit_gap": g,
        "warm_prefill_ms": warm.prefillMs, "cold_prefill_ms": cold.prefillMs,
    ]
    allSame = allSame && t && l
    log("loop \(lane) partial: diverges at \(common), reuse \(warm.cached) (rule \(rule)); same \(t) \(l)")
    report["paged_incompatible"] = coordinator.isPagedIncompatible
    report["all_cached_equal_cold"] = allSame
    try writeJSON(report, "cacheloop_\(tag)_\(lane).json")
}

// MARK: - longctx

func taskLongContext() async throws {
    let dir = bundleURL()
    let (model, context, _) = try await productionLoad(dir)
    let activeLoaded = Memory.activeMemory
    let targets = (env["T5_LONG_LENS"] ?? "32845,65411,100003,131001").split(separator: ",").map { Int($0)! }
    let depths = [0.1, 0.5, 0.9]
    let decodeSteps = Int(env["T5_STEPS"] ?? "64")!
    let mode = env["T5_THINKING"] ?? "disabled"
    // Haystack: the held-out token stream decoded in 2,000-token pieces (special tokens skipped).
    let stream = try loadTokens().map(Int.init)
    var pieces: [(String, Int)] = []
    var s = 0
    while s < stream.count {
        let e = min(s + 2000, stream.count)
        let text = context.tokenizer.decode(tokenIds: Array(stream[s ..< e]), skipSpecialTokens: true)
        pieces.append((text, context.tokenizer.encode(text: text, addSpecialTokens: false).count))
        s = e
    }
    func prompt(target: Int, depth: Double, code: String) async throws -> (LMInput, Int) {
        var budget = target - 160
        var parts: [String] = []
        var i = 0
        while budget > 0, i < pieces.count {
            let (text, n) = pieces[i]
            if n <= budget {
                parts.append(text)
                budget -= n
            } else {
                let chars = Int(Double(text.count) * Double(budget) / Double(n))
                parts.append(String(text.prefix(chars)))
                budget = 0
            }
            i += 1
        }
        let needle = "\n\nIMPORTANT NOTE: The access code for the Meridian vault is \(code). Keep it safe.\n\n"
        parts.insert(needle, at: Int(Double(parts.count) * depth))
        var question =
            "Question: What is the access code for the Meridian vault mentioned in the document? Reply with the code only."
        for _ in 0 ..< 4 {
            let messages: [Chat.Message] = [
                .user("Read the document below and answer the question at the end.\n\n<document>\n" + parts.joined()
                    + "\n</document>\n\n" + question)
            ]
            let input = try await prepare(context, messages, additional: thinkingContext(mode))
            let n = tokenIds(input).count
            if n % 128 != 0 { return (input, n) }
            question += " "
        }
        fatalError("could not avoid a multiple of 128")
    }
    // Warm-up (discarded): kernel JIT and self-checks.
    do {
        let warm = LMInput(tokens: MLXArray(Array(stream[0 ..< 3011]).map { Int32($0) }))
        _ = try greedy(model, warm, maxTokens: 8, drafter: nil, stopAtEOS: false)
    }
    var rows: [[String: Any]] = []
    for target in targets {
        var pp: [Double] = [], tg: [Double] = [], probes: [[String: Any]] = []
        for (j, depth) in depths.enumerated() {
            let words = ["ALPHA", "BRAVO", "DELTA"]
            let partA: Int = 4000 + target % 997 + j * 131
            let partB: Int = 1000 + (target / 7 + j * 389) % 9000
            let code = "\(partA)-\(words[j])-\(partB)"
            let (input, n) = try await prompt(target: target, depth: depth, code: code)
            Memory.clearCache()
            Memory.peakMemory = 0
            let vmA = VMStat.now()
            var parameters = GenerateParameters(maxTokens: decodeSteps + 1, temperature: 0, prefillStepSize: 2048)
            parameters.maxTokens = decodeSteps + 1
            let t0 = nowMs()
            var iterator = try TokenIterator(input: input, model: model, parameters: parameters)
            guard let first = iterator.next() else { fatalError("no first token") }
            let t1 = nowMs()
            let activeAfterPrefill = Memory.activeMemory
            let peakPrefill = Memory.peakMemory
            let kv = cacheBytes(iterator.cache)
            let offset = iterator.cache.map(\.offset).max() ?? 0
            var out = [first]
            while out.count < decodeSteps + 1, let token = iterator.next() { out.append(token) }
            let t2 = nowMs()
            let vmB = VMStat.now()
            let answerIds = Array(out.prefix(while: { $0 != eosToken }))
            let answer = context.tokenizer.decode(tokenIds: answerIds, skipSpecialTokens: false)
            let ppS = Double(n) / ((t1 - t0) / 1000)
            let tgS = Double(out.count - 1) / ((t2 - t1) / 1000)
            pp.append(ppS)
            tg.append(tgS)
            probes.append([
                "prompt_tokens": n, "mod128": n % 128, "mod2048": n % 2048, "depth": depth, "code": code,
                "answer": String(answer.prefix(200)), "retrieved": answer.contains(code),
                "prefill_ms": t1 - t0, "pp_s": ppS, "decode_tokens": out.count - 1, "tok_s": tgS,
                "mlx_active_after_prefill_GiB": gib(activeAfterPrefill), "mlx_peak_prefill_GiB": gib(peakPrefill),
                "mlx_peak_GiB": gib(Memory.peakMemory), "kv_bytes": kv, "cache_offset": offset,
                "kv_bytes_per_token": Double(kv) / Double(max(1, offset)),
                "prefill_temporaries_GiB": gib(peakPrefill - activeLoaded - kv), "vm_delta": vmA.delta(vmB),
            ])
            log("longctx \(n) depth \(depth): \(Int(ppS)) pp/s, \(String(format: "%.1f", tgS)) tok/s, retrieved \(answer.contains(code)) [\(answer.prefix(60))], peak \(String(format: "%.1f", gib(peakPrefill))) GiB, KV \(kv / 1_048_576) MiB")
            iterator = try TokenIterator(input: LMInput(tokens: MLXArray([Int32(1)])), model: model, parameters: GenerateParameters(maxTokens: 1))
            Memory.clearCache()
        }
        rows.append([
            "target": target, "probes": probes, "pp_s_median": median(pp), "tok_s_median": median(tg),
            "retrieved": probes.filter { $0["retrieved"] as! Bool }.count,
        ])
        try writeJSON(["mlx_active_loaded_GiB": gib(activeLoaded), "thinking": mode, "rows": rows], "longctx_\(tag).json")
    }
}

// MARK: - prefill step A/B

/// Cold prefill time at prefill step (aligned chunk) 2048 vs 1024 vs 512, interleaved: what a smaller step costs a cold
/// prompt (the aligned prefix rule re-prefills up to step - 1 tokens of every continued turn, so a smaller step wastes
/// less per tool call).
func taskStepAB() async throws {
    let (model, _, _) = try await productionLoad(bundleURL())
    let tokens = try loadTokens()
    let steps = (env["T5_STEPS_AB"] ?? "2048,1024,512").split(separator: ",").map { Int($0)! }
    let probes = Int(env["T5_PROBES"] ?? "3")!
    var report: [String: Any] = ["steps": steps, "probes": probes]
    _ = try greedy(model, LMInput(tokens: MLXArray(Array(tokens[0 ..< 3011]))), maxTokens: 2, drafter: nil)
    for n in lens {
        var t: [Int: [Double]] = [:]
        for p in 0 ..< probes + 1 {
            for step in (p % 2 == 0 ? steps : steps.reversed()) {
                Memory.clearCache()
                let input = LMInput(tokens: MLXArray(Array(tokens[0 ..< n])))
                let (run, _) = try greedy(model, input, maxTokens: 2, drafter: nil, stopAtEOS: false, step: step)
                if p > 0 { t[step, default: []].append(Double(n) / (run.prefillMs / 1000)) }
            }
        }
        report["\(n)"] = Dictionary(uniqueKeysWithValues: t.map { ("\($0.key)", ["pp_s_runs": $0.value, "pp_s_median": median($0.value)]) })
        log("stepab n=\(n): " + steps.map { "C=\($0) \(Int(median(t[$0]!))) pp/s" }.joined(separator: ", "))
    }
    try writeJSON(report, "stepab_\(tag).json")
}

// MARK: - cutsab

/// Cold prefill of the held-out stream (concatenated chat transcripts: a message end every ~140 tokens) at step 2048
/// through TokenIterator, the arms interleaved per round in a rotating order, 1 discarded + T5_PROBES rounds:
///   nocuts     v1 chunk-only rule (the model's turn markers removed): 2048-row chunks + the store boundary
///   grouped    v2 turn cuts, the pieces of a chunk as one layer-major group (default since 2026-10-29)
///   piecewise  v2 turn cuts, piece by piece (`VMLX_M3_PREFILL_GROUP=0`)
///   old        piecewise + buffer cache released after every chunk + stock lane growth (the 10-29 pin's prefill)
/// Per arm and probe: pp/s, the pieces, the first token and a sha of the first-token logits. grouped / piecewise / old
/// run the same pieces and must agree bit for bit (fail closed: a mismatch is logged as INVALID).
func taskCutsAB() async throws {
    let (model, _, _) = try await productionLoad(bundleURL())
    let tokens = try loadTokens()
    let markers = model.alignedPrefillTurnMarkers
    guard markers != nil else { fatalError("cutsab: the turn markers did not resolve (VMLX_M3_STRUCTURAL_CUTS=0?)") }
    let probes = Int(env["T5_PROBES"] ?? "3")!
    let armNames = (env["T5_CUTS_ARMS"] ?? "nocuts,grouped,piecewise,old").split(separator: ",").map(String.init)
    var report: [String: Any] = ["arms": armNames, "probes": probes, "markers": markers!.signature]
    var invalid: [String] = []
    var probesLog: [[String: Any]] = []

    func runArm(_ arm: String, _ n: Int) throws -> (ppS: Double, first: Int, sha: String, pieces: Int) {
        model.setAlignedPrefillTurnMarkers(arm == "nocuts" ? nil : markers)
        defer { model.setAlignedPrefillTurnMarkers(markers) }
        var firstLogits: MLXArray?
        var pieces = 0
        AlignedPrefill.hooks.logitsObserver = { x in
            if firstLogits == nil { firstLogits = x[0, -1].asType(.float32) }
        }
        AlignedPrefill.hooks.prefillObserver = { pieces = $0.pieces.count }
        defer {
            AlignedPrefill.hooks.logitsObserver = nil
            AlignedPrefill.hooks.prefillObserver = nil
        }
        let input = LMInput(tokens: MLXArray(Array(tokens[0 ..< n])))
        let piecewise = arm == "piecewise" || arm == "old" || arm == "sortedpw"
        let (run, _) = try MiniMaxM3Model.$prefillSortedAllOverride.withValue(arm.hasPrefix("sorted")) {
          try MiniMaxM3Model.$prefillGroupingOverride.withValue(!piecewise) {
            try MiniMaxM3Model.$prefillClearEachChunkOverride.withValue(arm == "old") {
                try MiniMaxM3SparseCache.$reserveOverride.withValue(arm == "old" ? false : nil) {
                    try greedy(model, input, maxTokens: 2, drafter: nil, stopAtEOS: false, step: 2048)
                }
            }
          }
        }
        let logits = try require2(firstLogits, "cutsab: no first-token logits for \(arm)")
        let sha = logits.asArray(Float.self).withUnsafeBufferPointer { sha8Bytes(Data(buffer: $0)) }
        return (Double(n) / (run.prefillMs / 1000), run.tokens.first ?? -1, sha, pieces)
    }

    _ = try greedy(model, LMInput(tokens: MLXArray(Array(tokens[0 ..< 3011]))), maxTokens: 2, drafter: nil)
    for n in lens {
        var pp: [String: [Double]] = [:]
        var shas: [String: Set<String>] = [:]
        var pieces: [String: Int] = [:]
        for p in 0 ..< probes + 1 {
            let order = (0 ..< armNames.count).map { armNames[($0 + p) % armNames.count] }
            for arm in order {
                Memory.clearCache()
                let t0 = Date().timeIntervalSince1970
                let r = try runArm(arm, n)
                probesLog.append([
                    "n": n, "round": p, "arm": arm, "t0": t0, "t1": Date().timeIntervalSince1970, "pp_s": r.ppS,
                ])
                shas[arm, default: []].insert("\(r.first):\(r.sha)")
                pieces[arm] = r.pieces
                if p > 0 { pp[arm, default: []].append(r.ppS) }
                log("cutsab n=\(n) round \(p) \(arm): \(Int(r.ppS)) pp/s, \(r.pieces) pieces, first \(r.first) sha \(r.sha)")
            }
        }
        // Same pieces, same kernels: the stock cut arms agree bit for bit, and so do the sorted-tile arms.
        let cutArms = armNames.filter { ["grouped", "piecewise", "old"].contains($0) }
        let cutShas = Set(cutArms.flatMap { shas[$0] ?? [] })
        if cutShas.count > 1 { invalid.append("n=\(n): cut arms disagree \(cutShas.sorted())") }
        let sortedShas = Set(armNames.filter { $0.hasPrefix("sorted") }.flatMap { shas[$0] ?? [] })
        if sortedShas.count > 1 { invalid.append("n=\(n): sorted arms disagree \(sortedShas.sorted())") }
        for arm in armNames where (shas[arm]?.count ?? 0) != 1 { invalid.append("n=\(n) \(arm): not repeatable \(shas[arm] ?? [])") }
        report["\(n)"] = Dictionary(uniqueKeysWithValues: armNames.map { arm in
            (arm, [
                "pp_s_runs": pp[arm] ?? [], "pp_s_median": median(pp[arm] ?? [0]), "pieces": pieces[arm] ?? 0,
                "first_sha": Array(shas[arm] ?? []),
            ] as [String: Any])
        })
        log("cutsab n=\(n): " + armNames.map { "\($0) \(Int(median(pp[$0] ?? [0]))) pp/s (\(pieces[$0] ?? 0) pieces)" }
            .joined(separator: ", ") + (cutShas.count <= 1 && sortedShas.count <= 1
                ? "; cut arms bit-identical (stock / sorted)" : "; INVALID: cut arms differ"))
    }
    report["invalid"] = invalid
    report["probes_log"] = probesLog
    if !invalid.isEmpty { log("cutsab INVALID: \(invalid)") }
    try writeJSON(report, "cutsab_\(tag).json")
}

func sha8Bytes(_ data: Data) -> String {
    SHA256.hash(data: data).prefix(8).map { String(format: "%02x", $0) }.joined()
}

func require2<T>(_ value: T?, _ message: String) throws -> T {
    guard let value else { throw NSError(domain: "M3RealT5", code: 1, userInfo: [NSLocalizedDescriptionKey: message]) }
    return value
}

// MARK: - prefprof

/// Where a cold prefill chunk's time goes (vMLX Python `prefill_prof --components`, 13-PY-COLD-PREFILL.md §3): the raw
/// held-out stream's first `T5_PROF_CTX` tokens (default 32845) in aligned 2048-row chunks, every chunk run layer by
/// layer with an evaluation after each component (the same ops `MiniMaxM3DecoderLayer` runs), seconds per component
/// summed over the full chunks and divided by their count. Also one uninstrumented cold prefill of the same tokens
/// (TokenIterator, the plan) for the instrumentation overhead.
func taskPrefillProfile() async throws {
    let (model, _, _) = try await productionLoad(bundleURL())
    let tokens = try loadTokens()
    let n = Int(env["T5_PROF_CTX"] ?? "32845")!
    let step = 2048
    _ = try greedy(model, LMInput(tokens: MLXArray(Array(tokens[0 ..< 3011]))), maxTokens: 2, drafter: nil)
    // Uninstrumented reference (markers removed: chunk-only pieces, as the instrumented loop runs).
    let markers = model.alignedPrefillTurnMarkers
    model.setAlignedPrefillTurnMarkers(nil)
    defer { model.setAlignedPrefillTurnMarkers(markers) }
    Memory.clearCache()
    let (plain, _) = try greedy(
        model, LMInput(tokens: MLXArray(Array(tokens[0 ..< n]))), maxTokens: 2, drafter: nil, stopAtEOS: false,
        step: step)
    log("prefprof n=\(n): uninstrumented cold prefill \(Int(Double(n) / (plain.prefillMs / 1000))) pp/s")

    var seconds: [String: Double] = [:]
    var fullChunks = 0
    let cache = model.newCache(parameters: nil)
    func timed<T>(_ name: String, _ body: () -> T, _ arrays: (T) -> [MLXArray]) -> T {
        let t0 = nowMs()
        let r = body()
        eval(arrays(r))
        seconds[name, default: 0] += (nowMs() - t0) / 1000
        return r
    }
    var start = 0
    Memory.clearCache()
    while start < n - 1 {
        let end = min(start + step, n - 1)
        let full = end - start == step
        if full { fullChunks += 1 }
        let before = seconds
        var h = timed("embed", { model.model.embedTokens(MLXArray(Array(tokens[start ..< end])).reshaped(1, -1)) }, { [$0] })
        for (li, layer) in model.model.layers.enumerated() {
            let attn = layer.selfAttn
            let c = cache[li] as! MiniMaxM3SparseCache
            let offset = c.offset
            let xn = timed("norms", { layer.inputLayerNorm(h) }, { [$0] })
            let qkv = timed("qkv_proj_norm_rope", { attn.project(xn, offset: offset) }, { [$0.queries, $0.keys, $0.values] })
            let kv = timed("kv_append", { c.update(keys: qkv.keys, values: qkv.values) }, { [$0.0, $0.1] })
            let selection: MiniMaxM3MSASelection?
            if let indexer = attn.indexer {
                let iqk = timed("index_proj_norm_rope", { indexer.project(xn, offset: offset) }, { [$0.queries, $0.keys] })
                let allKeys = timed("index_append", { c.updateIndexKeys(iqk.keys) }, { [$0] })
                selection = timed("selection", { indexer.select(queries: iqk.queries, keys: allKeys, offset: offset) },
                                  { $0.map { [$0.topIndices] } ?? [] })
            } else {
                selection = nil
            }
            let o = timed("attention", { attn.attend(qkv.queries, kv.0, kv.1, selection: selection, mask: .causal) }, { [$0] })
            h = timed("o_proj_residual", {
                h + attn.wo(o.transposed(0, 2, 1, 3).reshaped(1, end - start, -1))
            }, { [$0] })
            let xn2 = timed("norms", { layer.postAttentionLayerNorm(h) }, { [$0] })
            if let block = layer.mlp as? MiniMaxM3SparseMoeBlock {
                let r = timed("router", { block.route(xn2) }, { [$0.indices, $0.scores] })
                let routed = timed("routed_experts", {
                    (block.experts as! any WeightedRoutedExpertLayer).callRouted(xn2, indices: r.indices, scores: r.scores)
                }, { [$0] })
                let shared = timed("shared_expert", { block.sharedExperts(xn2) }, { [$0] })
                h = timed("combine_residual", { h + (routed + shared).asType(xn2.dtype) }, { [$0] })
            } else {
                h = timed("dense_mlp", { h + layer.mlp(xn2) }, { [$0] })
            }
        }
        if !full { seconds = before }  // per full chunk only
        start = end
    }
    let perChunk = seconds.mapValues { $0 / Double(max(1, fullChunks)) }
    let total = perChunk.values.reduce(0, +)
    for (k, v) in perChunk.sorted(by: { $0.value > $1.value }) {
        log(String(format: "prefprof %@: %.3f s / chunk (%.1f %%)", k, v, 100 * v / total))
    }
    log(String(format: "prefprof full chunk (instrumented): %.3f s over %d full chunks; uninstrumented %.0f pp/s",
               total, fullChunks, Double(n) / (plain.prefillMs / 1000)))
    try writeJSON([
        "n": n, "full_chunks": fullChunks, "per_chunk_s": perChunk, "instrumented_chunk_s": total,
        "uninstrumented_pp_s": Double(n) / (plain.prefillMs / 1000),
    ], "prefprof_\(tag).json")
}
