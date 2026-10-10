import Foundation
import MLX
import MLXLMCommon
import MLXHuggingFace
import MLXLLM
@preconcurrency import VMLXTokenizers

/// BENCH_K2_MULTITURN=1 — sustained multi-turn chat on the production route (diagnostic; in-app numbers are the truth).
///
/// Loads like Osaurus (`.osaurusProduction`), enables caching through `ModelContainer.enableCachingAsync` with the config
/// `VMLXServerRuntimeSettings.cacheCoordinatorConfig` builds from default settings (RAM prefix + block-disk SSD tier,
/// paged off, KV cap 65,536 = safe_auto), and generates through `makeBatchEngine(maxBatchSize: 1)`. History keeps answers,
/// drops reasoning (what the chat template replays). One JSON line per turn.
///
/// Env: K2MT_ARM=app|ram|none · K2MT_TURNS (10) · K2MT_MAX_TOKENS (400) · K2MT_EFFORT (high) · K2MT_OUT (dir) ·
/// K2MT_GREEDY=1
func runK2MultiturnBench(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let arm = env["K2MT_ARM"] ?? "app"
    let turns = Int(env["K2MT_TURNS"] ?? "") ?? 10
    let maxTokens = Int(env["K2MT_MAX_TOKENS"] ?? "") ?? 400
    let effort = env["K2MT_EFFORT"] ?? "high"
    let out = URL(fileURLWithPath: env["K2MT_OUT"] ?? NSTemporaryDirectory())
    let diskDir = out.appendingPathComponent("disk-cache-\(arm)", isDirectory: true)
    try? FileManager.default.removeItem(at: diskDir)
    try FileManager.default.createDirectory(at: diskDir, withIntermediateDirectories: true)

    let modelDir = URL(fileURLWithPath: modelPath)
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader(), loadConfiguration: .osaurusProduction
    ).0
    if env["K2MT_SDPA"] == "1" {
        // Attention alone, 36 layers x (32q/8kv x 128, bf16), decode L=1. Contiguous KV vs a slice of a larger buffer (KVCacheSimple).
        do {
            let x = MLXRandom.normal([36, 2, 8, 14336, 128]).asType(.bfloat16); eval(x)
            var t: [Double] = []
            for _ in 0 ..< 25 { let t0 = Date(); let s = x.sum(axes: [3]); eval(s); t.append(Date().timeIntervalSince(t0)) }
            t.sort(); let gb = Double(x.nbytes) / 1e9
            print("K2MT_ROOF sum_over_seq GB=\(String(format: "%.2f", gb)) ms=\(String(format: "%.2f", t[12] * 1000)) GB/s=\(String(format: "%.0f", gb / t[12]))")
            var t2: [Double] = []
            for _ in 0 ..< 25 { let t0 = Date(); let s = x.sum(); eval(s); t2.append(Date().timeIntervalSince(t0)) }
            t2.sort(); print("K2MT_ROOF sum_all ms=\(String(format: "%.2f", t2[12] * 1000)) GB/s=\(String(format: "%.0f", gb / t2[12]))")
        }
        for ctx in [512, 4096, 8192, 14336, 16384, 32768] {
            let q = MLXRandom.normal([1, 32, 1, 128]).asType(.bfloat16)
            let cap = (ctx / 256 + 1) * 256
            let kb = (0 ..< 36).map { _ in MLXRandom.normal([1, 8, cap, 128]).asType(.bfloat16) }
            let vb = (0 ..< 36).map { _ in MLXRandom.normal([1, 8, cap, 128]).asType(.bfloat16) }
            let kc = kb.map { $0[0..., 0..., ..<ctx, 0...].contiguous() }, vc = vb.map { $0[0..., 0..., ..<ctx, 0...].contiguous() }
            eval(q); eval(kb); eval(vb); eval(kc); eval(vc)
            func run(_ ks: [MLXArray], _ vs: [MLXArray], sliced: Bool) -> Double {
                var t: [Double] = []
                for _ in 0 ..< 25 {
                    let t0 = Date()
                    var outs: [MLXArray] = []
                    for l in 0 ..< 36 {
                        let k = sliced ? ks[l][0..., 0..., ..<ctx, 0...] : ks[l]
                        let v = sliced ? vs[l][0..., 0..., ..<ctx, 0...] : vs[l]
                        outs.append(MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 0.088, mask: .none))
                    }
                    eval(outs); t.append(Date().timeIntervalSince(t0))
                }
                t.sort(); return t[t.count / 2]
            }
            let a = run(kc, vc, sliced: false), b = run(kb, vb, sliced: true)
            let gb = Double(ctx * 36 * 8 * 128 * 2 * 2) / 1e9
            print("K2MT_SDPA ctx=\(ctx) kvGB=\(String(format: "%.2f", gb)) contig_ms=\(String(format: "%.2f", a * 1000)) (\(String(format: "%.0f", gb / a)) GB/s) sliced_ms=\(String(format: "%.2f", b * 1000)) (\(String(format: "%.0f", gb / b)) GB/s)")
        }
        return
    }
    if env["K2MT_MULTIROW"] == "1" { runMultiRowAttentionProbe(); return }
    if env["K2MT_ROWS"] != nil {
        // Forward cost vs query rows (speculative verify/draft shape) at fixed context. Trim back after each timing.
        let model = context.model
        let rows = env["K2MT_ROWS"]!.split(separator: ",").compactMap { Int($0) }
        for ctx in (env["K2MT_CTXS"] ?? "2048,14336,32768").split(separator: ",").compactMap({ Int($0) }) {
            let cache = model.newCache(parameters: nil)
            var done = 0
            while done < ctx {
                let n = min(2048, ctx - done)
                let toks = MLXArray((0 ..< n).map { Int32(1000 + (($0 + done) * 7919) % 50000) }).reshaped(1, n)
                eval(model(LMInput.Text(tokens: toks), cache: cache, state: nil).logits); done += n
                Memory.clearCache()
            }
            var line = "K2MT_ROWS ctx=\(ctx)"
            for r in rows {
                let toks = MLXArray((0 ..< r).map { Int32(2000 + $0 * 13) }).reshaped(1, r)
                var t: [Double] = []
                for i in 0 ..< 14 {
                    let t0 = Date()
                    let lg = model(LMInput.Text(tokens: toks), cache: cache, state: nil).logits
                    eval(lg)
                    if i >= 4 { t.append(Date().timeIntervalSince(t0)) }
                    for c in cache { c.trim(r) }
                }
                t.sort()
                line += " r\(r)=\(String(format: "%.1f", t[t.count / 2] * 1000))"
            }
            print(line)
        }
        return
    }
    if env["K2MT_CURVE"] == "1" {
        // Decode cost vs context on a bare KVCacheSimple (no coordinator, no engine): the inherent attention growth.
        let model = context.model
        for ctx in (env["K2MT_CTXS"] ?? "512,2048,4096,8192,16384,32768").split(separator: ",").compactMap({ Int($0) }) {
            var cache = model.newCache(parameters: nil)
            var done = 0
            while done < ctx {
                let n = min(2048, ctx - done)
                let toks = MLXArray((0 ..< n).map { Int32(1000 + (($0 + done) * 7919) % 50000) }).reshaped(1, n)
                let lg = model(LMInput.Text(tokens: toks), cache: cache, state: nil).logits
                eval(lg); eval(cache.flatMap { $0.innerState() }); done += n
                Memory.clearCache()
            }
            if let bits = Int(env["K2MT_KVBITS"] ?? "") {
                cache = cache.map { ($0 as? KVCacheSimple)?.toQuantized(groupSize: 64, bits: bits) ?? $0 }
                eval(cache.flatMap { $0.innerState() })
            }
            var tok = MLXArray([Int32(4242)]).reshaped(1, 1)
            var times: [Double] = []
            for i in 0 ..< 72 {
                let t0 = Date()
                let lg = model(LMInput.Text(tokens: tok), cache: cache, state: nil).logits
                tok = argMax(lg[0..., -1, 0...], axis: -1).reshaped(1, 1).asType(.int32)
                eval(tok)
                if i >= 8 { times.append(Date().timeIntervalSince(t0)) }
            }
            times.sort()
            let med = times[times.count / 2]
            let k = cache.first?.innerState().first
            let iso = ISO8601DateFormatter(); iso.formatOptions = [.withFullTime, .withColonSeparatorInTime]
            print("K2MT_CURVE ctx=\(ctx) ms_per_tok=\(String(format: "%.2f", med * 1000)) tok_s=\(String(format: "%.1f", 1 / med)) kv_dtype=\(k.map { "\($0.dtype)" } ?? "?") kv_shape=\(k?.shape ?? []) t=\(iso.string(from: Date()))")
        }
        return
    }
    let container = ModelContainer(context: context)
    if arm != "none" {
        var config = VMLXServerRuntimeSettings().cacheCoordinatorConfig(
            modelKey: "k2mt-\(modelDir.lastPathComponent)",
            diskCacheDirectory: arm == "app" ? diskDir : nil)
        config.defaultMaxKVSize = 65_536
        if arm == "ram" { config.enableDiskCache = false; config.diskCacheDir = nil }
        await container.enableCachingAsync(config: config)
    }
    let engine = await container.makeBatchEngine(maxBatchSize: 1)
    let coordinator = container.cacheCoordinator
    print("K2MT_START arm=\(arm) disk=\(coordinator?.config.enableDiskCache ?? false) turns=\(turns) max_tokens=\(maxTokens)")

    let asks = [
        "Write a detailed explanation of how a hash map handles collisions, with a short Python example.",
        "Now compare open addressing and separate chaining in a table, then explain when each is better.",
        "Write a Python class implementing a hash map with linear probing and resizing.",
        "Add a delete method that uses tombstones, and explain why tombstones are needed.",
        "Explain the time complexity of every method, including amortized resizing, in detail.",
        "Write five unit tests for the class using unittest.",
        "Rewrite the class to support iteration over keys in insertion order. Show the full code.",
        "Explain how Python's built-in dict achieves insertion order and compact storage.",
        "Summarize everything we discussed so far as a structured study guide with headings.",
        "Write a short quiz with five questions and answers based on this conversation.",
        "Explain how consistent hashing differs from the hash maps above, with an example.",
        "Give a final checklist for choosing a hash table design in production systems.",
    ]
    var messages: [[String: any Sendable]] = [
        ["role": "system", "content": "You are a precise programming tutor. Answer completely."],
    ]
    let monitorURL = diskDir
    func diskBytes() -> Int {
        guard let e = FileManager.default.enumerator(at: monitorURL, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total = 0
        for case let u as URL in e { total += (try? u.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0 }
        return total
    }
    func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
    }
    let iso = ISO8601DateFormatter()
    iso.formatOptions = [.withFullTime, .withColonSeparatorInTime]
    let padTokens = Int(env["K2MT_PAD_TOKENS"] ?? "") ?? 0
    var pad = ""
    if padTokens > 0 {
        // Reference module (~4 chars/token), distinct lines so it tokenizes like real code.
        var i = 0
        while pad.count < padTokens * 4 {
            pad += "def handler_\(i)(request, store):\n    key = request.get('k\(i % 97)')\n    return store.lookup(key, default=\(i * 31 % 1009))\n"
            i += 1
        }
    }
    for turn in 0 ..< min(turns, asks.count) {
        let ask = turn == 0 && !pad.isEmpty ? "Reference module:\n```python\n\(pad)```\n\n" + asks[turn] : asks[turn]
        messages.append(["role": "user", "content": ask])
        nonisolated(unsafe) let input = try await context.processor.prepare(input: UserInput(
            messages: messages, additionalContext: ["reasoning_effort": effort]))
        let promptTokens = input.text.tokens.size
        var p = GenerateParameters(
            generationConfig: context.configuration.generationDefaults,
            fallback: GenerateParameters(maxTokens: maxTokens))
        p.maxTokens = maxTokens
        if env["K2MT_GREEDY"] == "1" { p.temperature = 0; p.topP = 1; p.topK = 0 }
        let diskBefore = diskBytes()
        let start = Date()
        var first: Date?
        var stamps: [Date] = []
        var text = "", reasoning = ""
        var tps = 0.0, genTokens = 0, promptTime = 0.0, genTime = 0.0
        for await item in await engine.generate(input: input, parameters: p) {
            switch item {
            case .chunk(let c): if first == nil { first = Date() }; text += c; stamps.append(Date())
            case .reasoning(let r): if first == nil { first = Date() }; reasoning += r; stamps.append(Date())
            case .info(let info):
                tps = info.tokensPerSecond; genTokens = info.generationTokenCount
                promptTime = info.promptTime; genTime = info.generateTime
            default: break
            }
        }
        let streamEnd = Date()
        // Store settle: SSD bytes stable for 1 s (max 20 s) after the stream ends.
        var last = diskBytes(), stableSince = Date(), settleEnd = Date()
        while Date().timeIntervalSince(streamEnd) < 20 {
            try await Task.sleep(nanoseconds: 250_000_000)
            let now = diskBytes()
            if now != last { last = now; stableSince = Date() }
            if Date().timeIntervalSince(stableSince) >= 1 { settleEnd = stableSince; break }
        }
        let stats = coordinator?.snapshotStats()
        let row: [String: Any] = [
            "arm": arm, "turn": turn, "prompt_tokens": promptTokens, "gen_tokens": genTokens,
            "ttft_s": first.map { $0.timeIntervalSince(start) } ?? -1,
            "prompt_time_s": promptTime, "decode_tps": tps, "gen_time_s": genTime,
            "turn_wall_s": streamEnd.timeIntervalSince(start),
            "post_decode_s": streamEnd.timeIntervalSince(start) - promptTime - genTime,
            "store_settle_s": max(0, settleEnd.timeIntervalSince(streamEnd)),
            "disk_bytes": last, "disk_written": last - diskBefore,
            "disk_hits": stats?.diskStats?.hits ?? -1, "disk_misses": stats?.diskStats?.misses ?? -1,
            "disk_stores": stats?.diskStats?.stores ?? -1,
            "footprint_mb": footprintMB(), "mlx_active_mb": Double(Memory.activeMemory) / 1_048_576,
            "mlx_cache_mb": Double(Memory.cacheMemory) / 1_048_576,
            "t_start": iso.string(from: start), "t_end": iso.string(from: streamEnd),
            "reasoning_chars": reasoning.count, "text_chars": text.count,
        ]
        if stamps.count > 600 {
            // Event rate per 512-event window (one event ~ one token at streamInterval 1).
            var w: [String] = []
            var i = 0
            while i + 512 < stamps.count { w.append(String(format: "%.1f", 512 / stamps[i + 512].timeIntervalSince(stamps[i]))); i += 512 }
            print("K2MT_WINDOWS turn=\(turn) events=\(stamps.count) rates=\(w.joined(separator: ","))")
        }
        let data = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
        print("K2MT_TURN " + String(decoding: data, as: UTF8.self))
        messages.append(["role": "assistant", "content": text])
    }
}
