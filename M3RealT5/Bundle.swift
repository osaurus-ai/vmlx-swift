// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Test harness (not shipped): MiniMax-M3.1 Swift runtime on the REAL final bundle (60 layers, JANGT routed experts on
// L3-59, dspark/ drafter), 2026-10-28, Mac Studio.
//
//   T5_BUNDLE=<bundle or template overlay> T5_OUT=<dir> .build/release/M3RealT5 <command>
//
// bload     production load (loadModel + VMLXServerRuntimeSettings load configuration, the drafter Osaurus resolves),
//           bytes vs MLX active memory, bindings, JANGT layers, wired limit read back, vm_stat across load and a decode
// evalrows  held-out rows (T5_EVAL_IN, written by prep_eval.py): per position the log-probs at the reference's top-128
//           ids, the rank of the reference argmax, next-token log-prob, P(think) / P(skip); scored by score_eval.py
// dspark    greedy DSpark == plain on chat prompts, acceptance / tokens per round / verify widths, decode tok/s
//           plain vs DSpark interleaved
// think     adaptive / enabled / disabled thinking_mode through the processor + generate stream (parser output)
// cacheloop agentic tool loop through TokenIterator + the serving CacheCoordinator: reuse per turn, cached == cold
// longctx   cold prefill pp/s, decode tok/s, needle retrieval, memory and KV bytes per token at long contexts
// profile   one decode step: graph node counts by primitive and per-component time

import Cmlx
import CryptoKit
import Foundation
import MLX
import MLXHuggingFace
import MLXNN
import VMLXTokenizers

@testable import MLXLLM
@testable import MLXLMCommon

let thinkToken = 200059
let skipToken = 200060
let eosToken = 200020

// MARK: - system memory

struct VMStat {
    var pageins: UInt64 = 0, pageouts: UInt64 = 0, compressions: UInt64 = 0, decompressions: UInt64 = 0
    var swapins: UInt64 = 0, swapouts: UInt64 = 0, wired: UInt64 = 0, free: UInt64 = 0, compressor: UInt64 = 0
    var swapUsedMiB: Double = 0

    static func now() -> VMStat {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        _ = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        var swap = xsw_usage()
        var size = MemoryLayout<xsw_usage>.size
        sysctlbyname("vm.swapusage", &swap, &size, nil, 0)
        return VMStat(
            pageins: UInt64(stats.pageins), pageouts: UInt64(stats.pageouts), compressions: UInt64(stats.compressions),
            decompressions: UInt64(stats.decompressions), swapins: UInt64(stats.swapins),
            swapouts: UInt64(stats.swapouts), wired: UInt64(stats.wire_count), free: UInt64(stats.free_count),
            compressor: UInt64(stats.compressor_page_count), swapUsedMiB: Double(swap.xsu_used) / 1_048_576)
    }

    static let pageBytes = Double(getpagesize())

    func delta(_ after: VMStat) -> [String: Any] {
        [
            "pageins": after.pageins &- pageins, "pageouts": after.pageouts &- pageouts,
            "compressions": after.compressions &- compressions, "decompressions": after.decompressions &- decompressions,
            "swapins": after.swapins &- swapins, "swapouts": after.swapouts &- swapouts,
            "wired_GiB_delta": (Double(after.wired) - Double(wired)) * VMStat.pageBytes / 1_073_741_824,
            "swap_used_MiB_delta": after.swapUsedMiB - swapUsedMiB,
        ]
    }

    var summary: [String: Any] {
        [
            "wired_GiB": Double(wired) * VMStat.pageBytes / 1_073_741_824,
            "free_GiB": Double(free) * VMStat.pageBytes / 1_073_741_824,
            "compressor_GiB": Double(compressor) * VMStat.pageBytes / 1_073_741_824, "swap_used_MiB": swapUsedMiB,
        ]
    }
}

func sysctlInt(_ name: String) -> Int? {
    var value: Int64 = 0
    var size = MemoryLayout<Int64>.size
    if sysctlbyname(name, &value, &size, nil, 0) == 0 { return Int(value) }
    var v32: Int32 = 0
    size = MemoryLayout<Int32>.size
    return sysctlbyname(name, &v32, &size, nil, 0) == 0 ? Int(v32) : nil
}

/// The process's MLX wired limit, read back by swapping it out and in (Cmlx has no getter).
func readWiredLimit() -> Int {
    var previous: size_t = 0
    var ignored: size_t = 0
    _ = mlx_set_wired_limit(&previous, 0)
    _ = mlx_set_wired_limit(&ignored, previous)
    return Int(previous)
}

func gib(_ bytes: Int) -> Double { Double(bytes) / 1_073_741_824 }

// MARK: - loading

/// The bundle as Osaurus loads it: `loadModel` with the server settings' load configuration (residency included).
func productionLoad(_ directory: URL) async throws -> (MiniMaxM3Model, ModelContext, Double) {
    let settings = VMLXServerRuntimeSettings()
    let configData = try? Data(contentsOf: directory.appendingPathComponent("config.json"))
    let loadConfiguration = settings.resolvedLoadConfiguration(
        base: .osaurusProduction, configData: configData, jangConfig: nil, status: nil, bundleDirectory: directory)
    let t0 = nowMs()
    let context = try await MLXLMCommon.loadModel(
        from: directory, using: #huggingFaceTokenizerLoader(), loadConfiguration: loadConfiguration)
    let ms = nowMs() - t0
    guard let model = context.model as? MiniMaxM3Model else {
        fatalError("loaded \(type(of: context.model)), not MiniMaxM3Model")
    }
    return (model, context, ms)
}

/// The drafter Osaurus resolves for this bundle (`bundledDSparkDrafter`), attached the way TokenIterator does.
func attachDrafter(_ model: MiniMaxM3Model, _ directory: URL) throws -> URL? {
    let settings = VMLXServerRuntimeSettings()
    let configData = try? Data(contentsOf: directory.appendingPathComponent("config.json"))
    guard let drafter = settings.bundledDSparkDrafter(configData: configData, bundleDirectory: directory) else {
        return nil
    }
    guard try model.attachDSparkDraft(directory: drafter) else { return nil }
    return drafter
}

/// Text tensor bytes of a safetensors bundle by class (from the shard headers: no tensor is read).
func safetensorsBytes(_ directory: URL) throws -> [String: Int] {
    var out: [String: Int] = [:]
    let files = try FileManager.default.contentsOfDirectory(atPath: directory.path).filter {
        $0.hasSuffix(".safetensors")
    }
    for file in files {
        let handle = try FileHandle(forReadingFrom: directory.appendingPathComponent(file))
        defer { try? handle.close() }
        let n = handle.readData(ofLength: 8).withUnsafeBytes { $0.load(as: UInt64.self) }
        let header = try JSONSerialization.jsonObject(with: handle.readData(ofLength: Int(n))) as! [String: Any]
        for (name, entry) in header where name != "__metadata__" {
            guard let entry = entry as? [String: Any], let offsets = entry["data_offsets"] as? [Int] else { continue }
            let bytes = offsets[1] - offsets[0]
            let kind: String
            if name.hasPrefix("vision_tower.") || name.hasPrefix("multi_modal_projector.")
                || name.hasPrefix("patch_merge_mlp.")
            {
                kind = "vision"
            } else if name.contains(".jt_") {
                kind = "jangt_banks"
            } else if name.contains("experts") || name.contains("switch_mlp") {
                kind = "routed_other"
            } else {
                kind = "trunk"
            }
            out[kind, default: 0] += bytes
            out["total", default: 0] += bytes
        }
    }
    return out
}

func cacheBytes(_ cache: [KVCache]) -> Int {
    cache.flatMap(\.state).reduce(0) { $0 + $1.nbytes }
}

// MARK: - chat rendering

let agentTools: [ToolSpec] = [
    [
        "type": "function",
        "function": [
            "name": "run_shell",
            "description": "Run a shell command in the project's Linux terminal and return stdout and stderr.",
            "parameters": [
                "type": "object",
                "properties": ["cmd": ["type": "string", "description": "The command line to run."]],
                "required": ["cmd"],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ],
    [
        "type": "function",
        "function": [
            "name": "read_file",
            "description": "Read a text file from the workspace.",
            "parameters": [
                "type": "object",
                "properties": ["path": ["type": "string", "description": "Path relative to the repo root."]],
                "required": ["path"],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ],
]

func thinkingContext(_ mode: String?, effort: String? = nil) -> [String: any Sendable]? {
    var context: [String: any Sendable] = [:]
    switch mode {
    case "on": context["enable_thinking"] = true
    case "off": context["enable_thinking"] = false
    case let m?: context["thinking_mode"] = m
    case nil: break
    }
    if let effort { context["reasoning_effort"] = effort }
    return context.isEmpty ? nil : context
}

func prepare(
    _ context: ModelContext, _ messages: [Chat.Message], tools: [ToolSpec]? = nil,
    additional: [String: any Sendable]? = nil
) async throws -> LMInput {
    try await context.processor.prepare(
        input: UserInput(chat: messages, tools: tools, additionalContext: additional))
}

func tokenIds(_ input: LMInput) -> [Int] {
    input.text.tokenIds ?? input.text.tokens.reshaped(-1).asArray(Int32.self).map(Int.init)
}

/// Generation-prompt suffix (as ModelContainer derives it for the cache coordinator).
func generationPromptSuffix(_ tokenizer: MLXLMCommon.Tokenizer) -> [Int] {
    guard let gp = tokenizer as? GenerationPromptControllableTokenizer else { return [] }
    let dummy: [[String: any Sendable]] = [["role": "user", "content": "x"]]
    guard
        let with = try? gp.applyChatTemplate(
            messages: dummy, tools: nil, additionalContext: nil, addGenerationPrompt: true),
        let without = try? gp.applyChatTemplate(
            messages: dummy, tools: nil, additionalContext: nil, addGenerationPrompt: false)
    else { return [] }
    var common = 0
    while common < min(with.count, without.count), with[common] == without[common] { common += 1 }
    return Array(with[common...])
}

// MARK: - greedy runs

struct GreedyRun {
    var tokens: [Int]
    var prefillMs: Double
    var decodeMs: Double
    var stats: DSparkGenerationStats?
    var decodeTokS: Double { Double(max(0, tokens.count - 1)) / (decodeMs / 1000) }
}

/// Greedy tokens through TokenIterator (stops after `maxTokens` or at EOS when `stopAtEOS`).
func greedy(
    _ model: MiniMaxM3Model, _ input: LMInput, maxTokens: Int, drafter: URL?, stopAtEOS: Bool = true,
    coordinator: CacheCoordinator? = nil, step: Int = 2048, temperature: Float = 0, topP: Float = 1
) throws -> (GreedyRun, TokenIterator) {
    var parameters = GenerateParameters(maxTokens: maxTokens, temperature: temperature, prefillStepSize: step)
    parameters.topP = topP
    if let drafter { parameters.draftStrategy = .dspark(drafterPath: drafter) }
    let t0 = nowMs()
    var iterator = try TokenIterator(
        input: input, model: model, parameters: parameters, cacheCoordinator: coordinator)
    var out: [Int] = []
    var t1 = t0
    while let token = iterator.next() {
        if out.isEmpty { t1 = nowMs() }
        out.append(token)
        if stopAtEOS && token == eosToken { break }
    }
    let t2 = nowMs()
    return (GreedyRun(tokens: out, prefillMs: t1 - t0, decodeMs: t2 - t1, stats: iterator.dsparkGenerationStats), iterator)
}

func firstDiff(_ a: [Int], _ b: [Int]) -> Int? {
    for i in 0 ..< min(a.count, b.count) where a[i] != b[i] { return i }
    return a.count == b.count ? nil : min(a.count, b.count)
}

func statsJSON(_ s: DSparkGenerationStats?) -> [String: Any] {
    guard let s else { return ["dspark": "not used"] }
    return [
        "steps": s.steps, "spec_steps": s.specSteps, "ar_steps": s.arSteps, "drafted": s.drafted,
        "accepted": s.accepted, "acceptance": s.acceptanceRate, "spec_emitted": s.specEmitted,
        "tokens_per_spec_round": s.specSteps > 0 ? Double(s.specEmitted) / Double(s.specSteps) : 0,
        "full_accepts": s.fullAccepts, "prefix_commits": s.prefixCommits, "commit_forwards": s.commitForwards,
        "verify_widths": Dictionary(uniqueKeysWithValues: s.verifyWidths.map { ("\($0.key)", $0.value) }),
        "draft_s": s.draftSeconds, "verify_s": s.verifySeconds, "commit_s": s.commitSeconds,
    ]
}

// MARK: - bload

func taskBundleLoad() async throws {
    let dir = bundleURL()
    var report: [String: Any] = ["bundle": dir.path]
    report["hw_memsize_GiB"] = gib(sysctlInt("hw.memsize") ?? 0)
    report["iogpu_wired_limit_mb"] = sysctlInt("iogpu.wired_limit_mb") ?? -1
    report["max_recommended_working_set_GiB"] = gib(GPU.maxRecommendedWorkingSetBytes() ?? 0)
    report["bytes_on_disk"] = try safetensorsBytes(dir).mapValues(gib)
    let vm0 = VMStat.now()
    let wire0 = SystemWireBudget.snapshot()
    report["vm_before_load"] = vm0.summary
    report["mlx_wired_limit_before_GiB"] = gib(readWiredLimit())
    Memory.peakMemory = 0
    let (model, context, loadMs) = try await productionLoad(dir)
    let vm1 = VMStat.now()
    report["load_ms"] = loadMs
    report["vm_load_delta"] = vm0.delta(vm1)
    report["mlx_active_after_load_GiB"] = gib(Memory.activeMemory)
    report["mlx_peak_during_load_GiB"] = gib(Memory.peakMemory)
    report["phys_footprint_after_load_GiB"] = physFootprintMiB() / 1024
    report["wire_report"] = ModelWireBudgetReport.lastLoad?.logLine ?? "none"
    report["residency_target_GiB"] = ModelWireBudgetReport.lastLoad?.residencyTargetBytes.map(gib) ?? -1
    report["mlx_wired_limit_after_load_GiB"] = gib(readWiredLimit())
    report["system_wired_GiB_before_after"] = [wire0.wiredBytes.map(gib) ?? -1, SystemWireBudget.snapshot().wiredBytes.map(gib) ?? -1]

    // Layers: dense MLP on the first layers, JANGT routed experts everywhere else.
    var kinds: [String: [Int]] = [:]
    for (index, layer) in model.model.layers.enumerated() {
        let kind: String
        if let block = layer.mlp as? MiniMaxM3SparseMoeBlock {
            kind = "\(type(of: block.experts))"
        } else {
            kind = "\(type(of: layer.mlp))"
        }
        kinds[kind, default: []].append(index)
    }
    report["layer_kinds"] = kinds.mapValues { "\($0.count) layers: \($0.first!)...\($0.last!)" }
    report["layers"] = model.model.layers.count
    report["sparse_layers"] = model.model.layers.filter { $0.selfAttn.indexer != nil }.count
    report["tool_call_format"] = String(describing: context.configuration.toolCallFormat)
    report["reasoning_parser"] = context.configuration.reasoningParserName ?? "nil"

    // Bindings: every text tensor of the index bound, nothing from outside; values compared shard by shard.
    let index = try JSONSerialization.jsonObject(
        with: Data(contentsOf: dir.appendingPathComponent("model.safetensors.index.json"))) as! [String: Any]
    let weightMap = index["weight_map"] as! [String: String]
    let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
    let isVision: (String) -> Bool = {
        $0.hasPrefix("vision_tower.") || $0.hasPrefix("multi_modal_projector.") || $0.hasPrefix("patch_merge_mlp.")
    }
    let banked = Set(weightMap.keys.filter { $0.contains(".jt_") })
    let text = Set(weightMap.keys.filter { !$0.contains(".jt_") && !isVision($0) })
    report["index_tensors"] = weightMap.count
    report["vision_tensors_dropped"] = weightMap.keys.filter(isVision).count
    report["banked_tensors"] = banked.count
    report["banked_not_in_exclusion_set"] = banked.subtracting(model.customTensorNames).sorted()
    report["unbound_text_tensors"] = text.subtracting(parameters.keys).sorted()
    report["parameters_not_from_bundle"] = Set(parameters.keys).subtracting(text).sorted()
    report["bound_parameter_GiB"] = gib(parameters.values.reduce(0) { $0 + $1.nbytes })
    if env["T5_COMPARE"] != "0" {
        var mismatch: [String] = []
        var compared = 0
        let t0 = nowMs()
        for shard in Set(weightMap.values).sorted() {
            try autoreleasepool {
                let stored = try loadArrays(url: dir.appendingPathComponent(shard))
                for (key, value) in stored where text.contains(key) {
                    guard let bound = parameters[key] else { continue }
                    if bound.dtype != value.dtype || bound.shape != value.shape
                        || !arrayEqual(bound, value).item(Bool.self)
                    {
                        mismatch.append(key)
                    }
                    compared += 1
                }
            }
            Memory.clearCache()
        }
        report["values_compared"] = compared
        report["value_mismatch"] = mismatch
        report["compare_ms"] = nowMs() - t0
    }

    // The drafter Osaurus resolves.
    let activeBeforeDrafter = Memory.activeMemory
    let t1 = nowMs()
    let drafter = try attachDrafter(model, dir)
    report["drafter"] = drafter?.path ?? "NOT RESOLVED"
    report["drafter_attach_ms"] = nowMs() - t1
    report["drafter_active_GiB"] = gib(Memory.activeMemory - activeBeforeDrafter)
    report["drafter_block"] = model.dsparkBlockSize ?? -1
    report["drafter_ladder_top"] = model.dsparkLadderTop ?? -1
    report["dspark_on_disk_GiB"] = gib(
        (try? safetensorsBytes(dir.appendingPathComponent("dspark")))?["total"] ?? 0)
    report["mlx_active_after_drafter_GiB"] = gib(Memory.activeMemory)

    // Decode run: a 3011-token chat-free prompt, 256 greedy tokens through TokenIterator with the drafter, vm_stat
    // around the prefill and around the decode separately.
    let tokens = try loadTokens()
    let prompt = Array(tokens[0 ..< 3011]).map(Int.init)
    let input = LMInput(tokens: MLXArray(prompt.map { Int32($0) }))
    for (name, useDrafter) in [("plain", false), ("dspark", true)] {
        let vmA = VMStat.now()
        var parameters = GenerateParameters(maxTokens: 257, temperature: 0, prefillStepSize: 2048)
        if useDrafter, let drafter { parameters.draftStrategy = .dspark(drafterPath: drafter) }
        let t2 = nowMs()
        var iterator = try TokenIterator(input: input, model: model, parameters: parameters)
        guard iterator.next() != nil else { fatalError("no first token") }
        let t3 = nowMs()
        let vmB = VMStat.now()
        var n = 0
        while iterator.next() != nil { n += 1 }
        let t4 = nowMs()
        let vmC = VMStat.now()
        report["decode_\(name)"] = [
            "prefill_ms": t3 - t2, "decode_tokens": n, "decode_tok_s": Double(n) / ((t4 - t3) / 1000),
            "vm_prefill_delta": vmA.delta(vmB), "vm_decode_delta": vmB.delta(vmC),
            "dspark": statsJSON(iterator.dsparkGenerationStats),
        ]
    }
    report["mlx_active_end_GiB"] = gib(Memory.activeMemory)
    report["mlx_peak_GiB"] = gib(Memory.peakMemory)
    report["vm_end"] = VMStat.now().summary
    report["mlx_wired_limit_end_GiB"] = gib(readWiredLimit())
    try writeJSON(report, "bload_\(tag).json")
    log("bload: \(Int(loadMs)) ms, unbound \((report["unbound_text_tensors"] as! [String]).count), drafter \(report["drafter"]!)")
}

// MARK: - evalrows

func taskEvalRows() async throws {
    guard let inPath = env["T5_EVAL_IN"] else { fatalError("T5_EVAL_IN not set") }
    let manifest = try JSONSerialization.jsonObject(
        with: Data(contentsOf: URL(fileURLWithPath: inPath + ".json"))) as! [String: Any]
    let rows = manifest["rows"] as! [[String: Any]]
    let arrays = try loadArrays(url: URL(fileURLWithPath: inPath + ".safetensors"))
    let (model, _, loadMs) = try await productionLoad(bundleURL())
    log("evalrows: loaded in \(Int(loadMs)) ms, \(rows.count) rows")
    let perShard = Int(env["T5_EVAL_SHARD"] ?? "40")!
    var out: [String: MLXArray] = [:]
    var shard = 0
    let t0 = nowMs()
    var done = 0
    for (i, row) in rows.enumerated() {
        let ids = arrays["ids_\(i)"]!.asType(.int32)
        let ti = arrays["ti_\(i)"]!.asType(.int32)
        let am = arrays["am_\(i)"]!.asType(.int32)
        let n = ids.dim(0)
        let idList = ids.asArray(Int32.self)
        let cache = model.newCache(parameters: nil)
        var qs: [MLXArray] = [], ranks: [MLXArray] = [], tls: [MLXArray] = [], ths: [MLXArray] = [], sks: [MLXArray] = []
        var s = 0
        while s < n {
            let e = min(s + chunk, n)
            let logits = model(MLXArray(Array(idList[s ..< e])).reshaped(1, e - s), cache: cache)[0]
            let lp = logProbs(logits)
            let q = takeAlong(lp, ti[s ..< e], axis: -1)
            let ref = takeAlong(lp, am[s ..< e].reshaped(-1, 1), axis: -1)
            let rank = (lp .> ref).asType(.int32).sum(axis: -1)
            let nxt = min(e, n - 1) - s
            let tl: MLXArray =
                nxt > 0
                ? takeAlong(lp[..<nxt], MLXArray(Array(idList[(s + 1) ..< (s + 1 + nxt)])).reshaped(-1, 1), axis: -1)
                    .reshaped(-1) : MLXArray.zeros([0], dtype: .float32)
            let th = lp[0..., thinkToken]
            let sk = lp[0..., skipToken]
            eval(q, rank, tl, th, sk)
            qs.append(q); ranks.append(rank); tls.append(tl); ths.append(th); sks.append(sk)
            s = e
        }
        out["q_\(i)"] = concatenated(qs, axis: 0)
        out["rank_\(i)"] = concatenated(ranks, axis: 0)
        out["tl_\(i)"] = concatenated(tls, axis: 0)
        out["th_\(i)"] = concatenated(ths, axis: 0)
        out["sk_\(i)"] = concatenated(sks, axis: 0)
        done += n
        Memory.clearCache()
        if (i + 1) % perShard == 0 || i == rows.count - 1 {
            try save(arrays: out, url: outDir.appendingPathComponent("evalrows_\(tag)_\(shard).safetensors"))
            out = [:]
            shard += 1
        }
        if i % 10 == 0 {
            log("evalrows [\(i + 1)/\(rows.count)] \(row["domain"] ?? "") \(n) tok, \(done) total, \(Int((nowMs() - t0) / 1000)) s")
        }
    }
    try writeJSON(
        ["rows": rows.count, "shards": shard, "tokens": done, "seconds": (nowMs() - t0) / 1000, "load_ms": loadMs],
        "evalrows_\(tag).json")
}

// MARK: - prompts

struct ChatCase: @unchecked Sendable {
    let name: String
    let messages: [Chat.Message]
    let tools: [ToolSpec]?
}

let buildLog = """
    $ swift build -c release
    Building for production...
    [412/419] Compiling CacheKit DiskStore.swift
    /work/cachekit/Sources/CacheKit/DiskStore.swift:88:31: error: value of optional type 'Data?' must be unwrapped to a value of type 'Data'
            let header = try decoder.decode(Header.self, from: handle.readData(upToCount: 64))
                                                           ^
    /work/cachekit/Sources/CacheKit/DiskStore.swift:88:31: note: coalesce using '??' to provide a default when the optional value contains 'nil'
    /work/cachekit/Sources/CacheKit/DiskStore.swift:88:31: note: force-unwrap using '!' to abort execution if the optional value contains 'nil'
    /work/cachekit/Sources/CacheKit/DiskStore.swift:131:17: warning: variable 'offset' was never mutated; consider changing to 'let' constant
    error: fatalError
    """

nonisolated(unsafe) let chatCases: [ChatCase] = [
    ChatCase(
        name: "coding",
        messages: [
            .user(
                "Write a Python function `parse_duration(s: str) -> int` that converts ISO-8601 durations like "
                    + "'PT1H30M', 'P2DT3H' and 'PT45S' into seconds. Raise ValueError on bad input. Include pytest tests.")
        ], tools: nil),
    ChatCase(
        name: "agentic_tool_call",
        messages: [
            .system("You are an autonomous coding agent. Use the tools to inspect and change the repository."),
            .user(
                "The CI job says port 8080 is already in use when the integration tests start. Find what is "
                    + "listening on 8080 on this machine and stop it."),
        ], tools: agentTools),
    ChatCase(
        name: "reasoning",
        messages: [
            .user(
                "A tank is filled by pipe A in 6 hours and by pipe B in 4 hours, and drained by pipe C in 12 hours. "
                    + "All three are opened at 9:00 with the tank one quarter full. At what time is the tank full? "
                    + "Show the steps.")
        ], tools: nil),
    ChatCase(
        name: "terminal",
        messages: [
            .system("You are a terminal assistant. Answer with the exact commands or code changes needed."),
            .user("My build fails like this, how do I fix it?\n\n```\n\(buildLog)\n```"),
        ], tools: nil),
    ChatCase(
        name: "multilingual_zh",
        messages: [.user("请用中文解释什么是哈希表，它的平均查找复杂度是多少，并给出一个 Python 的简单例子。")], tools: nil),
    ChatCase(
        name: "multilingual_ja",
        messages: [.user("TCPの3ウェイハンドシェイクを日本語で簡潔に説明してください。各ステップで送られるフラグも書いてください。")],
        tools: nil),
    ChatCase(
        name: "agentic_tool_result",
        messages: [
            .system("You are an autonomous coding agent. Use the tools to inspect and change the repository."),
            .user("Run the unit tests and fix whatever fails."),
            Chat.Message(
                role: .assistant, content: "",
                toolCalls: [ToolCall(function: .init(name: "run_shell", arguments: ["cmd": "swift test 2>&1 | tail -20"]))]),
            .tool(
                "Test Suite 'All tests' started.\nTest Case '-[CacheKitTests.DiskStoreTests testRoundTrip]' started.\n"
                    + "/work/cachekit/Tests/CacheKitTests/DiskStoreTests.swift:41: error: -[CacheKitTests.DiskStoreTests "
                    + "testRoundTrip] : XCTAssertEqual failed: (\"17\") is not equal to (\"16\") - header length\n"
                    + "Test Case '-[CacheKitTests.DiskStoreTests testRoundTrip]' failed (0.012 seconds).\n"
                    + "Executed 38 tests, with 1 failure (0 unexpected) in 0.412 (0.418) seconds"),
        ], tools: agentTools),
]

// MARK: - dspark

func taskDSpark() async throws {
    let dir = bundleURL()
    let (model, context, _) = try await productionLoad(dir)
    let maxTokens = Int(env["T5_MAX_TOKENS"] ?? "256")!
    let rounds = Int(env["T5_ROUNDS"] ?? "3")!
    let mode = env["T5_THINKING"]  // nil = adaptive (the template default)
    var inputs: [(String, LMInput)] = []
    for c in chatCases {
        inputs.append((c.name, try await prepare(context, c.messages, tools: c.tools, additional: thinkingContext(mode))))
    }
    var report: [String: Any] = ["max_tokens": maxTokens, "rounds": rounds, "thinking": mode ?? "adaptive (unset)"]
    // Control: plain decode before the drafter is attached (no tap recording), first two prompts, 3 probes.
    var preAttach: [String: [Double]] = [:]
    _ = try greedy(model, inputs[0].1, maxTokens: 16, drafter: nil)  // warm-up
    for (name, input) in inputs.prefix(2) {
        for _ in 0 ..< 3 {
            let (run, _) = try greedy(model, input, maxTokens: maxTokens, drafter: nil)
            preAttach[name, default: []].append(run.decodeTokS)
        }
    }
    report["plain_before_drafter_attached"] = preAttach.mapValues { ["runs": $0, "median": median($0)] }
    guard let drafter = try attachDrafter(model, dir) else { fatalError("no drafter resolved") }
    report["drafter"] = drafter.path
    report["ladder_top"] = model.dsparkLadderTop ?? -1
    report["block"] = model.dsparkBlockSize ?? -1
    _ = try greedy(model, inputs[0].1, maxTokens: 16, drafter: drafter)  // warm-up of the drafter path
    var rows: [[String: Any]] = []
    for (name, input) in inputs {
        var plainRuns: [GreedyRun] = [], sparkRuns: [GreedyRun] = []
        for r in 0 ..< rounds {
            let order = r % 2 == 0 ? [false, true] : [true, false]
            for useDrafter in order {
                let (run, _) = try greedy(model, input, maxTokens: maxTokens, drafter: useDrafter ? drafter : nil)
                if useDrafter { sparkRuns.append(run) } else { plainRuns.append(run) }
            }
        }
        let reference = plainRuns[0].tokens
        let diffs = sparkRuns.map { firstDiff(reference, $0.tokens) }
        let plainDiffs = plainRuns.map { firstDiff(reference, $0.tokens) }
        let text = context.tokenizer.decode(tokenIds: reference, skipSpecialTokens: false)
        let row: [String: Any] = [
            "name": name, "prompt_tokens": tokenIds(input).count, "generated": reference.count,
            "first_token": reference.first ?? -1,
            "first_diff_dspark_vs_plain": diffs.map { $0.map { $0 as Any } ?? NSNull() },
            "first_diff_plain_vs_plain": plainDiffs.map { $0.map { $0 as Any } ?? NSNull() },
            "plain_tok_s": ["runs": plainRuns.map(\.decodeTokS), "median": median(plainRuns.map(\.decodeTokS))],
            "dspark_tok_s": ["runs": sparkRuns.map(\.decodeTokS), "median": median(sparkRuns.map(\.decodeTokS))],
            "speedup_median": median(sparkRuns.map(\.decodeTokS)) / median(plainRuns.map(\.decodeTokS)),
            "dspark_stats": sparkRuns.map { statsJSON($0.stats) },
            "text_head": String(text.prefix(400)),
        ]
        rows.append(row)
        // Determinism control (fail closed): every plain run == the first, and the output is not degenerate (a NUL /
        // single-token run is what the M5 Ultra's broken fused attention produced).
        let degenerate = Set(reference).count <= 2 || reference.isEmpty
        let deterministic = plainDiffs.allSatisfy { $0 == nil } && !degenerate
        log("dspark \(name): first diff \(diffs), plain vs plain \(plainDiffs) \(deterministic ? "DETERMINISTIC" : "NONDETERMINISTIC_OR_DEGENERATE"), plain \(median(plainRuns.map(\.decodeTokS))) dspark \(median(sparkRuns.map(\.decodeTokS))) tok/s, accept \(sparkRuns.last?.stats?.acceptanceRate ?? -1), sha \(sha8(reference))")
    }
    report["prompts"] = rows
    try writeJSON(report, "dspark_\(tag).json")
}

// MARK: - think

/// Thinking modes on the real bundle through the production processor and the generate stream (reasoning parser).
func taskThink() async throws {
    let dir = bundleURL()
    let (model, context, _) = try await productionLoad(dir)
    let drafter = env["T5_THINK_DSPARK"] == "1" ? try attachDrafter(model, dir) : nil
    let maxTokens = Int(env["T5_MAX_TOKENS"] ?? "384")!
    var rows: [[String: Any]] = []
    let cases = chatCases.filter { ["coding", "agentic_tool_call", "reasoning", "multilingual_zh", "agentic_tool_result"].contains($0.name) }
    for c in cases {
        for (mode, effort) in [(nil, nil), ("on", nil), ("off", nil), ("enabled", nil), ("disabled", nil), (nil, "low")] as [(String?, String?)] {
            let input = try await prepare(context, c.messages, tools: c.tools, additional: thinkingContext(mode, effort: effort))
            let ids = tokenIds(input)
            let tail = context.tokenizer.decode(tokenIds: Array(ids.suffix(12)), skipSpecialTokens: false)
            var parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0, prefillStepSize: 2048)
            if let drafter { parameters.draftStrategy = .dspark(drafterPath: drafter) }
            var reasoning = "", content = "", calls: [String] = []
            var info: GenerateCompletionInfo?
            var firstKind: String?
            for await event in try MLXLMCommon.generate(input: input, parameters: parameters, context: context) {
                switch event {
                case .chunk(let s):
                    if firstKind == nil, !s.isEmpty { firstKind = "content" }
                    content += s
                case .reasoning(let s):
                    if firstKind == nil, !s.isEmpty { firstKind = "reasoning" }
                    reasoning += s
                case .toolCall(let call):
                    if firstKind == nil { firstKind = "tool_call" }
                    calls.append("\(call.function.name) \(call.function.arguments)")
                case .info(let i): info = i
                default: break
                }
            }
            // The raw first token (greedy, same prompt) decides think vs skip.
            let (raw, _) = try greedy(model, input, maxTokens: 1, drafter: nil)
            let modeName = mode ?? (effort != nil ? "adaptive+effort=\(effort!)" : "adaptive")
            let row: [String: Any] = [
                "case": c.name, "mode": modeName, "prompt_tokens": ids.count,
                "prompt_ends_with": ids.last ?? -1, "prompt_tail": tail,
                "system_has_enabled_line": context.tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
                    .contains("Current thinking mode: enabled."),
                "system_has_disabled_line": context.tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
                    .contains("Current thinking mode: disabled."),
                "system_has_adaptive_line": context.tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
                    .contains("Current thinking mode: adaptive."),
                "has_effort_tag": context.tokenizer.decode(tokenIds: ids, skipSpecialTokens: false).contains("<effort>"),
                "first_generated_token": raw.tokens.first ?? -1,
                "first_stream_event": firstKind ?? "none",
                "reasoning_chars": reasoning.count, "content_chars": content.count, "tool_calls": calls,
                "markers_leaked_in_content": content.contains("mm:think"),
                "markers_leaked_in_reasoning": reasoning.contains("mm:think"),
                "reasoning_head": String(reasoning.prefix(160)), "content_head": String(content.prefix(240)),
                "generated_tokens": info?.generationTokenCount ?? -1,
            ]
            rows.append(row)
            log("think \(c.name) \(modeName): ends \(ids.last ?? -1), first tok \(raw.tokens.first ?? -1), first event \(firstKind ?? "none"), reasoning \(reasoning.count) content \(content.count) calls \(calls.count)")
        }
    }
    try writeJSON(["rows": rows, "dspark": drafter != nil], "think_\(tag).json")
}


/// First 8 hex digits of SHA-256 over the token ids (little-endian Int32): a run's identity for logs.
func sha8(_ tokens: [Int]) -> String {
    var data = Data()
    for t in tokens { withUnsafeBytes(of: Int32(truncatingIfNeeded: t).littleEndian) { data.append(contentsOf: $0) } }
    return SHA256.hash(data: data).prefix(4).map { String(format: "%02x", $0) }.joined()
}
