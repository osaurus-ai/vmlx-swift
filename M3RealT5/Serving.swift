// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Test harness (not shipped): the serving path Osaurus drives (2026-10-29 GUI proof findings).
//
// serving   load with Osaurus' memory-safety plan (`resolvedMemorySafetyPlan`, Safe Auto, host status) and the plan's
//           cache settings, a ModelContainer with the CacheCoordinator, BatchEngine (maxBatchSize 1 = the solo
//           TokenIterator path), long document prompts. Arms switch MLX's scheduling memory limit at run time (it is
//           read on every eval): `plan` = what the plan applied at load, `native` = MLX's own ceiling. The allocator
//           reuse pool is held at the same value in both arms (Osaurus' admitted scratch floor, 2 GiB), so the arms
//           differ in one thing. Per probe: prefill pp/s, decode tok/s, MLX active / peak, vm deltas, output text.
//
//   T5_SERVE_LENS=4517,35665 T5_SERVE_ARMS=plan,native T5_SERVE_PROBES=3 T5_LANE=plain|dspark T5_STEP=512
//   T5_SERVE_DECODE=96 T5_SERVE_PLAN_LIMIT=<bytes>: the `plan` arm's limit instead of what this host's plan applied (the
//   Studio's 256 GB resolve Safe Auto to 0.7 x RAM = 179 GiB; a 128 GB Mac to weights + 4 GiB = 104,518,001,232 B
//   for M3.1 + drafter before the resident-pool fix, which is what the GUI proof ran under on erics-m5-max)

import Foundation
import MLX
import MLXHuggingFace
import MLXNN
import VMLXTokenizers

@testable import MLXLLM
@testable import MLXLMCommon

/// The load Osaurus performs (`ModelRuntime.resolveMemorySafetyLoadPlan`): base load configuration, then the memory
/// safety plan with the bundle facts and the host snapshot.
func osaurusPlanLoad(_ directory: URL) async throws -> (ModelContext, VMLXResolvedMemorySafetyPlan, Double) {
    let settings = VMLXServerRuntimeSettings()
    let configData = try? Data(contentsOf: directory.appendingPathComponent("config.json"))
    let base = settings.resolvedLoadConfiguration(
        base: .osaurusProduction, configData: configData, jangConfig: nil, status: nil, bundleDirectory: directory)
    let plan = settings.resolvedMemorySafetyPlan(
        baseLoadConfiguration: base, bundleFacts: LoadBundleFacts.inspect(bundleURL: directory),
        host: MemoryStatus.snapshot())
    let t0 = nowMs()
    let context = try await MLXLMCommon.loadModel(
        from: directory, using: #huggingFaceTokenizerLoader(), loadConfiguration: plan.loadConfiguration)
    return (context, plan, nowMs() - t0)
}

struct ServeResult {
    var text = ""
    var reasoning = ""
    var info: GenerateCompletionInfo?
    var firstEventMs: Double = 0
    var totalMs: Double = 0
}

func serve(_ engine: BatchEngine, _ input: LMInput, _ parameters: GenerateParameters) async -> ServeResult {
    var r = ServeResult()
    let t0 = nowMs()
    // The harness owns the input and never touches it again after this call.
    nonisolated(unsafe) let owned = input
    let stream = await engine.generate(input: owned, parameters: parameters)
    for await event in stream {
        switch event {
        case .chunk(let s):
            if r.firstEventMs == 0 { r.firstEventMs = nowMs() - t0 }
            r.text += s
        case .reasoning(let s):
            if r.firstEventMs == 0 { r.firstEventMs = nowMs() - t0 }
            r.reasoning += s
        case .info(let i): r.info = i
        default: break
        }
    }
    r.totalMs = nowMs() - t0
    return r
}

func taskServing() async throws {
    let dir = bundleURL()
    let lane = env["T5_LANE"] ?? "plain"
    let step = Int(env["T5_STEP"] ?? "512")!
    let decode = Int(env["T5_SERVE_DECODE"] ?? "96")!
    let probes = Int(env["T5_SERVE_PROBES"] ?? "3")!
    let targets = (env["T5_SERVE_LENS"] ?? "4517,35665").split(separator: ",").map { Int($0)! }
    let arms = (env["T5_SERVE_ARMS"] ?? "plan,native").split(separator: ",").map(String.init)
    let pool = Int(env["T5_SERVE_POOL_BYTES"] ?? "\(2 << 30)")!

    let vm0 = VMStat.now()
    let (context, plan, loadMs) = try await osaurusPlanLoad(dir)
    guard let model = context.model as? MiniMaxM3Model else { fatalError("not MiniMaxM3Model") }
    let planLimit = Int(env["T5_SERVE_PLAN_LIMIT"] ?? "") ?? Memory.memoryLimit
    let nativeLimit: Int = {
        let physical = Int(ProcessInfo.processInfo.physicalMemory)
        let recommended = MLX.GPU.maxRecommendedWorkingSetBytes() ?? Int.max
        return max(1 << 30, min(physical / 20 * 19, recommended + recommended / 2))
    }()
    var report: [String: Any] = [
        "lane": lane, "step": step, "decode": decode, "load_ms": loadMs,
        "plan_summary": plan.displaySummary, "plan_memory_limit": "\(plan.loadConfiguration.memoryLimit)",
        "mlx_memory_limit_after_load": Memory.memoryLimit, "plan_arm_limit": planLimit, "mlx_cache_limit_after_load": Memory.cacheLimit,
        "native_memory_limit": nativeLimit, "pool_bytes_both_arms": pool,
        "mlx_active_after_load_GiB": gib(Memory.activeMemory), "vm_load": vm0.delta(VMStat.now()),
    ]
    log("serving: plan limit \(planLimit) (\(String(format: "%.2f", gib(planLimit))) GiB), native \(nativeLimit), active \(String(format: "%.2f", gib(Memory.activeMemory))) GiB, plan: \(plan.displaySummary)")
    let drafter = lane == "dspark" ? try attachDrafter(model, dir) : nil
    if lane == "dspark" && drafter == nil { fatalError("no drafter resolved") }

    // Osaurus' container: the plan's cache settings, a scratch disk tier.
    let diskDir = outDir.appendingPathComponent("serving-disk-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: diskDir) }
    var settings = VMLXServerRuntimeSettings()
    settings.cache = plan.cache
    let processor = context.processor
    let tokenizer = context.tokenizer
    let container = ModelContainer(context: context)
    await container.enableCachingAsync(
        config: settings.cacheCoordinatorConfig(modelKey: "m31-serving-\(lane)", diskCacheDirectory: diskDir))
    let engine = await container.makeBatchEngine(maxBatchSize: 1)

    let stream = try loadTokens().map(Int.init)
    var pieces: [(String, Int)] = []
    var s = 0
    while s < stream.count {
        let e = min(s + 2000, stream.count)
        let text = tokenizer.decode(tokenIds: Array(stream[s ..< e]), skipSpecialTokens: true)
        pieces.append((text, tokenizer.encode(text: text, addSpecialTokens: false).count))
        s = e
    }
    // A distinct opening line per probe keeps the probes from reusing each other's cache entries.
    func prompt(target: Int, salt: Int) async throws -> (LMInput, Int) {
        var budget = target - 200
        var parts: [String] = ["Session \(salt * 7919 % 100_000): archive excerpt follows.\n"]
        var i = salt % 7
        while budget > 0 {
            let (text, n) = pieces[i % pieces.count]
            if n <= budget {
                parts.append(text)
                budget -= n
            } else {
                parts.append(String(text.prefix(Int(Double(text.count) * Double(budget) / Double(n)))))
                budget = 0
            }
            i += 1
        }
        let code = "\(4000 + salt * 37 % 900)-KESTREL-\(1000 + salt * 911 % 9000)"
        parts.insert("\n\nIMPORTANT NOTE: The access code for the Meridian vault is \(code).\n\n", at: parts.count / 2)
        var question = "Question: What is the access code for the Meridian vault? Reply with the code only."
        for _ in 0 ..< 8 {
            let messages: [Chat.Message] = [
                .user("Read the document and answer.\n\n<document>\n" + parts.joined() + "\n</document>\n\n" + question)
            ]
            let input = try await processor.prepare(
                input: UserInput(chat: messages, additionalContext: thinkingContext("disabled")))
            let n = tokenIds(input).count
            if n % 512 != 0 && n % 256 != 0 { return (input, n) }
            question += " "
        }
        fatalError("prompt length stuck on a multiple of 256")
    }

    // Warm-up (discarded): kernels, self-checks.
    do {
        let (input, _) = try await prompt(target: 3000, salt: 999)
        var p = GenerateParameters(maxTokens: 8, temperature: 0, prefillStepSize: step)
        if let drafter { p.draftStrategy = .dspark(drafterPath: drafter) }
        _ = await serve(engine, input, p)
    }
    var rows: [[String: Any]] = []
    var salt = 0
    for target in targets {
        var byArm: [String: (pp: [Double], tg: [Double])] = [:]
        var texts: [String: [String]] = [:]
        for probe in 0 ..< probes {
            for arm in (probe % 2 == 0 ? arms : arms.reversed()) {
                salt += 1
                let (input, n) = try await prompt(target: target, salt: salt)
                Memory.memoryLimit = arm == "native" ? nativeLimit : planLimit
                Memory.cacheLimit = pool
                Memory.clearCache()
                Memory.peakMemory = 0
                let vmA = VMStat.now()
                var p = GenerateParameters(maxTokens: decode, temperature: 0, prefillStepSize: step)
                if let drafter { p.draftStrategy = .dspark(drafterPath: drafter) }
                let r = await serve(engine, input, p)
                let vmB = VMStat.now()
                let info = r.info
                let pp = info.map { Double($0.promptTokenCount) / max(1e-9, $0.promptTime) } ?? 0
                let tg = info.map { Double(max(0, $0.generationTokenCount - 1)) / max(1e-9, $0.generateTime) } ?? 0
                byArm[arm, default: ([], [])].pp.append(pp)
                byArm[arm, default: ([], [])].tg.append(tg)
                texts[arm, default: []].append(r.text)
                rows.append([
                    "target": target, "prompt_tokens": n, "mod512": n % 512, "mod2048": n % 2048, "arm": arm,
                    "probe": probe, "memory_limit": Memory.memoryLimit,
                    "prompt_tokens_prefilled": info?.promptTokenCount ?? -1, "prompt_s": info?.promptTime ?? -1,
                    "generated": info?.generationTokenCount ?? -1, "generate_s": info?.generateTime ?? -1,
                    "pp_s": pp, "tok_s": tg, "first_event_ms": r.firstEventMs, "total_ms": r.totalMs,
                    "mlx_peak_GiB": gib(Memory.peakMemory), "mlx_active_end_GiB": gib(Memory.activeMemory),
                    "text": String(r.text.prefix(120)), "vm": vmA.delta(vmB),
                ])
                log("serving \(lane) \(n) \(arm) probe \(probe): \(Int(pp)) pp/s, \(String(format: "%.1f", tg)) tok/s, peak \(String(format: "%.1f", gib(Memory.peakMemory))) GiB, limit \(String(format: "%.1f", gib(Memory.memoryLimit))) GiB [\(r.text.prefix(40))]")
                report["probes"] = rows
                try writeJSON(report, "serving_\(tag)_\(lane).json")
            }
        }
        var summary: [String: Any] = ["target": target]
        for (arm, v) in byArm {
            summary[arm] = ["pp_s_median": median(v.pp), "tok_s_median": median(v.tg), "pp_s": v.pp, "tok_s": v.tg]
        }
        log("serving \(lane) \(target) medians: " + byArm.map { "\($0.key) \(Int(median($0.value.pp))) pp/s \(String(format: "%.1f", median($0.value.tg))) tok/s" }.joined(separator: " | "))
        var medians = report["medians"] as? [[String: Any]] ?? []
        medians.append(summary)
        report["medians"] = medians
    }
    Memory.memoryLimit = planLimit
    await engine.shutdown()
    try writeJSON(report, "serving_\(tag)_\(lane).json")
}

// MARK: - servingloop: the per-tool-call prefix cache through BatchEngine (the path Osaurus drives)

/// A 6-turn terminal-agent loop (as `cacheloop`) through Osaurus' load + ModelContainer + CacheCoordinator +
/// BatchEngine.generate, warm (the container's engine) vs cold (an engine without a coordinator on the same model).
/// Per turn: what the restore kept (AlignedPrefill hooks), the rule's expected reuse, first-event latency, output
/// equality warm == cold (greedy). `T5_LANE=dspark` attaches the bundle's drafter.
func taskServingLoop() async throws {
    let dir = bundleURL()
    let lane = env["T5_LANE"] ?? "plain"
    let step = Int(env["T5_STEP"] ?? "2048")!
    let maxTokens = Int(env["T5_MAX_TOKENS"] ?? "48")!
    let (context, plan, _) = try await osaurusPlanLoad(dir)
    guard let model = context.model as? MiniMaxM3Model else { fatalError("not MiniMaxM3Model") }
    let drafter = lane == "dspark" ? try attachDrafter(model, dir) : nil
    if lane == "dspark" && drafter == nil { fatalError("no drafter resolved") }
    let markers = model.alignedPrefillTurnMarkers
    log("servingloop \(lane): step \(step), markers \(markers.map(String.init(describing:)) ?? "none"), schema \(model.alignedPrefillSchema ?? "off"), MLX limit \(Memory.memoryLimit)")
    let processor = context.processor
    let tokenizer = context.tokenizer
    let diskDir = outDir.appendingPathComponent("servingloop-disk-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: diskDir) }
    var settings = VMLXServerRuntimeSettings()
    settings.cache = plan.cache
    let container = ModelContainer(context: context)
    await container.enableCachingAsync(
        config: settings.cacheCoordinatorConfig(modelKey: "m31-servingloop-\(lane)", diskCacheDirectory: diskDir))
    let warm = await container.makeBatchEngine(maxBatchSize: 1)
    let cold = await container.perform { ctx in BatchEngine(context: ctx, maxBatchSize: 1, cacheCoordinator: nil) }
    let suffix = generationPromptSuffix(tokenizer)
    let recorder = LoopRecorder()
    recorder.install()

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
    var previous: [Int] = []
    var previousK = 0
    var turns: [[String: Any]] = []
    var allEqual = true
    func parameters() -> GenerateParameters {
        var p = GenerateParameters(maxTokens: maxTokens, temperature: 0, prefillStepSize: step)
        if let drafter { p.draftStrategy = .dspark(drafterPath: drafter) }
        return p
    }
    for turn in 0 ..< 6 {
        let input = try await processor.prepare(input: UserInput(chat: messages, tools: agentTools))
        let ids = tokenIds(input)
        let k = AlignedPrefillGeometry.storeBoundary(
            promptTokenIds: ids, cachePrefixTokenCounts: input.cachePrefixTokenCounts ?? [], generationPromptSuffix: suffix) ?? 0
        let common = lcp(ids, previous)
        let cuts = AlignedPrefillGeometry.turnCuts(ids, markers: markers, end: k)
        let rule = turn == 0 ? 0 : AlignedPrefillGeometry.reuseLimit(
            matched: min(common, previousK), boundary: k, step: step, exactGeometry: false, turnCuts: cuts)
        let r0 = recorder.prefills.count
        let w = await serve(warm, input, parameters())
        let warmPrefill = recorder.prefills.count > r0 ? recorder.prefills[r0].start : -1
        let coldInput = try await processor.prepare(input: UserInput(chat: messages, tools: agentTools))
        let c = await serve(cold, coldInput, parameters())
        // Fail closed: an empty output on both arms is not equality.
        let same = w.text == c.text && w.reasoning == c.reasoning && !(c.text.isEmpty && c.reasoning.isEmpty)
        allEqual = allEqual && same
        turns.append([
            "turn": turn + 1, "prompt": ids.count, "K": k, "lcp_previous": common, "previous_K": previousK,
            "reused": warmPrefill, "rule": rule, "reuse_equals_rule": warmPrefill == rule,
            "warm_first_event_ms": w.firstEventMs, "cold_first_event_ms": c.firstEventMs,
            "warm_prompt_s": w.info?.promptTime ?? -1, "cold_prompt_s": c.info?.promptTime ?? -1,
            "warm_prefilled_tokens": w.info?.promptTokenCount ?? -1, "output_equal": same,
            "text": String(w.text.prefix(80)),
        ])
        log("servingloop \(lane) turn \(turn + 1): prompt \(ids.count) K \(k) lcp \(common) reused \(warmPrefill) (rule \(rule)); first event warm \(Int(w.firstEventMs)) / cold \(Int(c.firstEventMs)) ms; equal \(same)")
        previous = ids
        previousK = k
        if turn < 5 {
            messages.append(
                Chat.Message(
                    role: .assistant, content: w.text,
                    toolCalls: [ToolCall(function: .init(name: "run_shell", arguments: ["cmd": commands[turn]]))]))
            messages.append(.tool(terminalOutput(lines: toolLines[turn], seed: turn)))
        }
        try writeJSON(["lane": lane, "step": step, "turns": turns, "all_equal": allEqual], "servingloop_\(tag)_\(lane).json")
    }
    await warm.shutdown()
    await cold.shutdown()
}
