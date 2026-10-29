// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Test harness (not shipped): the agentic prefix-cache loop and the long-context runs on the real MiniMax-M3.1 bundle.

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
    let step = 2048
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
    var report: [String: Any] = ["lane": lane, "max_tokens": maxTokens, "step": step, "gen_suffix": suffix]
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
        let expectedMax = turn == 0 ? 0 : (min(common, previousK) / step) * step
        let ok = tokensSame && logitsSame && cold.cached == 0 && !cold.tokens.isEmpty
        allSame = allSame && ok
        turns.append([
            "turn": turn + 1, "prompt": ids.count, "K": k, "lcp_with_previous_prompt": common,
            "previous_K": previousK, "reuse": warm.cached, "rule_max_reuse": expectedMax,
            "reuse_is_chunk_or_K": warm.cached % step == 0 || warm.cached == k,
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
    let rule = (min(common, previousK) / step) * step
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
