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
//   T5_SERVE_DECODE=96

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
    let planLimit = Memory.memoryLimit
    let nativeLimit: Int = {
        let physical = Int(ProcessInfo.processInfo.physicalMemory)
        let recommended = MLX.GPU.maxRecommendedWorkingSetBytes() ?? Int.max
        return max(1 << 30, min(physical / 20 * 19, recommended + recommended / 2))
    }()
    var report: [String: Any] = [
        "lane": lane, "step": step, "decode": decode, "load_ms": loadMs,
        "plan_summary": plan.displaySummary, "plan_memory_limit": "\(plan.loadConfiguration.memoryLimit)",
        "mlx_memory_limit_after_load": planLimit, "mlx_cache_limit_after_load": Memory.cacheLimit,
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
