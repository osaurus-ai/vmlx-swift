// BENCH_RIPPLE_PARITY=1 — Ripple-65B (ricursive) port check against the Python MLX reference (`ripple_mlx.py`,
// golden-checked against the release's BF16 logits).
//
// Reads BENCH_RIPPLE_DIR/ids.json (token ids + positions written by the Python side) and writes, next to it:
//   swift_logits_<pos>.f32   no-cache forward rows at each position (compare with py_logits_<pos>.f32)
//   swift_report.json        per-position top-1, cached-vs-no-cache deltas, rollback deltas, greedy tokens
//
// Cache checks, all against the same model's no-cache forward (rounding cannot hide a window or offset defect there):
//   chunked   prefill in BENCH_RIPPLE_CHUNK (default 333, deliberately not a divisor of the 1024 window) chunks
//   decode    the last 64 prompt tokens fed one at a time across the window edge
//   rollback  feed a 16-row block, trim 11 rows (a rejected speculative suffix), re-feed the true tokens one by one
// Not a speed measurement.

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import VMLXTokenizers

func runRippleParityProbe(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    guard let dir = env["BENCH_RIPPLE_DIR"] else {
        print("[RIPPLE] BENCH_RIPPLE_DIR is required")
        exit(2)
    }
    let outDir = URL(fileURLWithPath: dir)
    struct IDs: Decodable { let ids: [Int]; let positions: [Int] }
    let spec = try JSONDecoder().decode(
        IDs.self, from: Data(contentsOf: outDir.appendingPathComponent("ids.json")))
    guard !spec.ids.isEmpty, !spec.positions.isEmpty else {
        print("[RIPPLE] INVALID: empty ids/positions")
        exit(3)
    }
    let chunk = Int(env["BENCH_RIPPLE_CHUNK"] ?? "333") ?? 333

    let t0 = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: URL(fileURLWithPath: modelPath), using: #huggingFaceTokenizerLoader(),
        loadConfiguration: .default)
    let model = context.model
    print(
        String(
            format: "[RIPPLE] load=%.1fs type=%@ active=%.2fGB", CFAbsoluteTimeGetCurrent() - t0,
            String(describing: type(of: model)), Double(MLX.Memory.activeMemory) / 1e9))

    let ids = MLXArray(spec.ids.map { Int32($0) })[.newAxis]
    let n = spec.ids.count
    var report: [String: Any] = ["prompt_tokens": n, "chunk": chunk]

    // 1. No-cache forward (whole sequence, explicit window masks).
    let full = model(ids, cache: nil).asType(.float32)
    eval(full)
    var rows: [String: Any] = [:]
    for p in spec.positions where p < n {
        let row = full[0, p]
        let data = row.asArray(Float.self).withUnsafeBufferPointer { Data(buffer: $0) }
        try data.write(to: outDir.appendingPathComponent("swift_logits_\(p).f32"))
        rows[String(p)] = row.argMax().item(Int.self)
    }
    report["nocache_top1"] = rows

    func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }
    func top1(_ a: MLXArray) -> Int { a.argMax().item(Int.self) }
    /// KL(reference || candidate) over the full vocabulary, in float32.
    func kl(_ ref: MLXArray, _ cand: MLXArray) -> Float {
        let r = ref.asType(.float32), c = cand.asType(.float32)
        let lr = r - logSumExp(r), lc = c - logSumExp(c)
        return (exp(lr) * (lr - lc)).sum().item(Float.self)
    }

    // 2. Chunked cached prefill vs no-cache, every chunk's last row.
    var cache = model.newCache(parameters: nil)
    var start = 0
    var chunkRows: [[String: Any]] = []
    let prefillEnd = n - 64
    while start < prefillEnd {
        let end = min(start + chunk, prefillEnd)
        let out = model(ids[0..., start ..< end], cache: cache).asType(.float32)
        eval(out)
        let ref = full[0, end - 1]
        chunkRows.append([
            "end": end, "max_abs": maxAbs(out[0, -1], ref), "kl": kl(ref, out[0, -1]),
            "top1_match": top1(out[0, -1]) == top1(ref),
        ])
        start = end
    }
    report["chunked"] = chunkRows

    // 3. Token-by-token decode over the rest (crosses position 1024 when the prompt does).
    var decodeWorst: Float = 0
    var decodeKL: Float = 0
    var decodeTop1Mismatch = 0
    for i in prefillEnd ..< n {
        let out = model(ids[0..., i ..< (i + 1)], cache: cache).asType(.float32)
        eval(out)
        decodeWorst = max(decodeWorst, maxAbs(out[0, 0], full[0, i]))
        decodeKL = max(decodeKL, kl(full[0, i], out[0, 0]))
        if top1(out[0, 0]) != top1(full[0, i]) { decodeTop1Mismatch += 1 }
    }
    report["decode"] = ["rows": n - prefillEnd, "max_abs": decodeWorst, "max_kl": decodeKL, "top1_mismatch": decodeTop1Mismatch]

    // 4. Rollback: rebuild to (n - 40), feed a 16-row block whose last 11 rows are wrong, trim them, continue.
    cache = model.newCache(parameters: nil)
    let base = n - 40
    var s = 0
    while s < base {
        let e = min(s + chunk, base)
        eval(model(ids[0..., s ..< e], cache: cache))
        s = e
    }
    var block = Array(spec.ids[base ..< (base + 16)].map { Int32($0) })
    for j in 5 ..< 16 { block[j] = Int32((Int(block[j]) + 7919 * j) % 100_000) }
    eval(model(MLXArray(block)[.newAxis], cache: cache))
    let trimmed = cache.map { $0.trim(11) }
    var rollbackWorst: Float = 0
    var rollbackKL: Float = 0
    var rollbackMismatch = 0
    for i in (base + 5) ..< n {
        let out = model(ids[0..., i ..< (i + 1)], cache: cache).asType(.float32)
        eval(out)
        rollbackWorst = max(rollbackWorst, maxAbs(out[0, 0], full[0, i]))
        rollbackKL = max(rollbackKL, kl(full[0, i], out[0, 0]))
        if top1(out[0, 0]) != top1(full[0, i]) { rollbackMismatch += 1 }
    }
    report["rollback"] = [
        "base": base, "trimmed": Set(trimmed).sorted(), "rows": n - base - 5, "max_abs": rollbackWorst, "max_kl": rollbackKL,
        "top1_mismatch": rollbackMismatch,
    ]

    // 5. Greedy 48 tokens from the full prompt (the Python side writes its own).
    cache = model.newCache(parameters: nil)
    s = 0
    var last = MLXArray(0)
    while s < n {
        let e = min(s + chunk, n)
        last = model(ids[0..., s ..< e], cache: cache)[0, -1]
        eval(last)
        s = e
    }
    var greedy: [Int] = []
    var next = top1(last)
    for _ in 0 ..< 48 {
        greedy.append(next)
        if next == 100_265 || next == 100_257 { break }
        let out = model(MLXArray([Int32(next)])[.newAxis], cache: cache)
        next = top1(out[0, -1])
    }
    report["greedy"] = greedy
    report["greedy_text"] = context.tokenizer.decode(tokenIds: greedy)
    report["peak_gb"] = Double(MLX.Memory.peakMemory) / 1e9
    let json = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    try json.write(to: outDir.appendingPathComponent("swift_report.json"))
    print("[RIPPLE] " + String(decoding: json, as: UTF8.self))
}

// BENCH_RIPPLE_TIMING=1 — forward cost per row count (median of BENCH_RIPPLE_REPS, default 9) at a fixed context
// (BENCH_RIPPLE_CTX, default 1500 tokens), before and after `LaneQMM.installForDFlash2Target`. Each timed forward is
// trimmed back so the context stays fixed. Development measurement only; the app is the proof.
func runRippleTimingProbe(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let reps = Int(env["BENCH_RIPPLE_REPS"] ?? "9") ?? 9
    let ctx = Int(env["BENCH_RIPPLE_CTX"] ?? "1500") ?? 1500
    MLX.Memory.cacheLimit = 8 << 30
    let context = try await MLXLMCommon.loadModel(
        from: URL(fileURLWithPath: modelPath), using: #huggingFaceTokenizerLoader(),
        loadConfiguration: .default)
    let model = context.model
    let cache = model.newCache(parameters: nil)
    var s = 0
    while s < ctx {
        let e = min(s + 512, ctx)
        eval(model(MLXArray((s ..< e).map { Int32(1000 + $0 % 5000) })[.newAxis], cache: cache))
        s = e
    }
    func time(rows: Int) -> Double {
        var samples: [Double] = []
        for r in 0 ..< (reps + 2) {
            let x = MLXArray((0 ..< rows).map { Int32(2000 + ($0 * 31 + r) % 4000) })[.newAxis]
            let t = CFAbsoluteTimeGetCurrent()
            eval(model(x, cache: cache))
            let dt = (CFAbsoluteTimeGetCurrent() - t) * 1000
            for c in cache { c.trim(rows) }
            if r >= 2 { samples.append(dt) }
        }
        return samples.sorted()[samples.count / 2]
    }
    // BENCH_RIPPLE_DUMP=<dir>: write fixed 8- and 16-row verify-shape logits (kernel A/B bit-identity checks).
    if let dump = env["BENCH_RIPPLE_DUMP"] {
        for rows in [8, 16] {
            let x = MLXArray((0 ..< rows).map { Int32(3000 + $0 * 97) })[.newAxis]
            let y = model(x, cache: cache).asType(.float32)
            eval(y)
            for c in cache { c.trim(rows) }
            let data = y.asArray(Float.self).withUnsafeBufferPointer { Data(buffer: $0) }
            try data.write(to: URL(fileURLWithPath: dump).appendingPathComponent("rows\(rows).f32"))
        }
    }
    // CPU graph construction vs GPU execution for one decode step (median of reps).
    do {
        var build: [Double] = [], run: [Double] = []
        for r in 0 ..< (reps + 2) {
            let x = MLXArray([Int32(2000 + r)])[.newAxis]
            let t0 = CFAbsoluteTimeGetCurrent()
            let y = model(x, cache: cache)
            let t1 = CFAbsoluteTimeGetCurrent()
            eval(y)
            let t2 = CFAbsoluteTimeGetCurrent()
            for c in cache { c.trim(1) }
            if r >= 2 { build.append((t1 - t0) * 1000); run.append((t2 - t1) * 1000) }
        }
        print(String(format: "[RIPPLE_TIMING] graph_build=%.1fms eval=%.1fms", build.sorted()[build.count / 2],
                     run.sorted()[run.count / 2]))
    }
    let widths = (env["BENCH_RIPPLE_WIDTHS"] ?? "1,2,4,8,16").split(separator: ",").compactMap { Int($0) }
    func components(_ tag: String) {
        guard let r = model as? RicursiveModel, env["BENCH_RIPPLE_COMPONENTS"] != "0" else { return }
        for w in widths {
            let t = r.diagnosticComponentTimings(rows: w)
            print(String(format: "[RIPPLE_COMPONENTS] %@ rows=%d mlp=%.1fms attn_proj=%.1fms head=%.1fms", tag, w,
                         t["mlp"] ?? -1, t["attn_proj"] ?? -1, t["head"] ?? -1))
        }
    }
    components("mlx")
    // Same process, same thermal state: the production token iterator (eager, greedy) on this model.
    if env["BENCH_RIPPLE_ITER"] != "0" {
        for round in 0 ..< 3 {
            var params = GenerateParameters(maxTokens: 160, temperature: 0)
            params.enableCompiledDecode = false
            let input = try await context.processor.prepare(
                input: UserInput(prompt: "Write a long story about a lighthouse keeper and a storm."))
            let t = CFAbsoluteTimeGetCurrent()
            var n = 0
            var first = 0.0
            for await item in try MLXLMCommon.generate(input: input, parameters: params, context: context) {
                if case .chunk = item { n += 1; if n == 1 { first = CFAbsoluteTimeGetCurrent() } }
                if case .info(let info) = item {
                    print(String(format: "[RIPPLE_ITER] round=%d tokens=%d decode=%.2f tok/s (%.1f ms/token) wall_after_first=%.1f ms/chunk",
                                 round, info.generationTokenCount, info.tokensPerSecond, 1000 / max(info.tokensPerSecond, 1e-9),
                                 n > 1 ? (CFAbsoluteTimeGetCurrent() - first) * 1000 / Double(n - 1) : 0))
                }
            }
            _ = t
        }
        print(String(format: "[RIPPLE_TIMING] after-iter w1=%.1fms", time(rows: 1)))
        // The host chat window's route: BatchEngine.generate (maxBatchSize 1, as the speed probe and app use).
        let engine = BatchEngine(context: context, maxBatchSize: 1)
        for round in 0 ..< 3 {
            let prepared = try await context.processor.prepare(
                input: UserInput(prompt: "Write a long story about a lighthouse keeper and a storm."))
            nonisolated(unsafe) let send = prepared
            var p = GenerateParameters(maxTokens: 160, temperature: 0, topP: 1, topK: 0, minP: 0, repetitionPenalty: nil)
            p.draftStrategy = nil
            let stream = await engine.generate(input: send, parameters: p)
            for await ev in stream {
                if case .info(let info) = ev {
                    print(String(format: "[RIPPLE_BATCH] round=%d tokens=%d decode=%.2f tok/s", round,
                                 info.generationTokenCount, info.tokensPerSecond))
                }
            }
        }
    }
    var line = "[RIPPLE_TIMING] mlx ctx=\(ctx)"
    for w in widths { line += String(format: " w%d=%.1fms", w, time(rows: w)) }
    print(line)
    if env["BENCH_RIPPLE_LANE"] != "0" {
        LaneQMM.installForDFlash2Target(model)
        let r = LaneQMM.lastInstallReport
        line = "[RIPPLE_TIMING] lane(\(r?.lane ?? -1) tiled=\(r?.tiled ?? -1) skipped=\(r?.skipped ?? -1)) flat=\(LaneQMM.isLaneFlat(model))"
        for w in widths { line += String(format: " w%d=%.1fms", w, time(rows: w)) }
        print(line)
        components("lane")
    }
}

// BENCH_RIPPLE_TTFT=1 — multi-turn restore shape the app runs: [system ~2.5k tokens, user] then append
// (assistant, user) twice, through BatchEngine with a disk-backed prefix cache (BENCH_RIPPLE_CACHE_DIR, fresh per arm).
// Arm via BENCH_RIPPLE_ARM=ar|dflash2. Reports per turn: prompt tokens, engine promptTime, TTFT, decode tok/s.
// Restored-token counts come from the cache traces (VMLX_CACHE_FETCH_TRACE=1 / VMLX_DFLASH2_TRACE=1).
func runRippleTTFTProbe(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let dir = URL(fileURLWithPath: modelPath)
    let arm = env["BENCH_RIPPLE_ARM"] ?? "ar"
    MLX.Memory.cacheLimit = 8 << 30
    let context = try await MLXLMCommon.loadModel(
        from: dir, using: #huggingFaceTokenizerLoader(), loadConfiguration: .default)
    // BENCH_RIPPLE_SERVING=1: the app's serving configuration (write-behind SSD stores, default paged settings).
    var cfg: CacheCoordinatorConfig
    let cacheDir = URL(fileURLWithPath: env["BENCH_RIPPLE_CACHE_DIR"] ?? "/tmp/ripple-ttft-cache")
    if env["BENCH_RIPPLE_SERVING"] == "1" {
        cfg = VMLXServerRuntimeSettings().cacheCoordinatorConfig(
            modelKey: dir.lastPathComponent, diskCacheDirectory: cacheDir)
    } else {
        cfg = CacheCoordinatorConfig()
        cfg.usePagedCache = env["BENCH_RIPPLE_PAGED"] == "1"
        cfg.enableDiskCache = true
        cfg.diskCacheDir = cacheDir
        cfg.modelKey = dir.lastPathComponent
    }
    print("[RIPPLE_TTFT] config paged=\(cfg.usePagedCache) disk=\(cfg.enableDiskCache) writeBehind=\(cfg.diskWriteBehind) resident=\(cfg.residentEntryBytes)")
    let coordinator = CacheCoordinator(config: cfg)
    if !cfg.usePagedCache { coordinator.setPagedIncompatible(true) }
    let engine = BatchEngine(context: context, maxBatchSize: 1, cacheCoordinator: coordinator)
    let sentence = "Policy note: archive shipping manifests, tide tables and lighthouse logs by quarter, and cite the ledger page. "
    let system = "You are a careful assistant for a harbor office.\n" + String(repeating: sentence, count: Int(env["BENCH_RIPPLE_SYS_REPEAT"] ?? "") ?? 110)
    let users = [
        "Write a Python function that totals crates per section from a list of dicts. Code only.",
        "Now add a unittest for it with two cases. Code only.",
        "Now make it ignore rows whose section is missing. Code only.",
    ]
    var history: [Chat.Message] = [.system(system)]
    for (turn, u) in users.enumerated() {
        history.append(.user(u))
        var input = UserInput(chat: history)
        input.additionalContext = [:]
        let prepared = try await context.processor.prepare(input: input)
        nonisolated(unsafe) let send = prepared
        // BENCH_RIPPLE_SAMPLED=1: the app's bundle sampling (T 1.0, top-p 0.95, no top-k), fixed seed.
        let sampled = env["BENCH_RIPPLE_SAMPLED"] == "1"
        var p = GenerateParameters(
            maxTokens: Int(env["BENCH_RIPPLE_MAXTOK"] ?? "") ?? 120, temperature: sampled ? 1 : 0, topP: sampled ? 0.95 : 1,
            topK: 0, minP: 0, repetitionPenalty: nil)
        if sampled { p.randomSeed = 1234 + UInt64(turn) }
        if arm == "dflash2" { p.draftStrategy = .dflash2(drafterPath: dir.appendingPathComponent("dflash2"), blockSize: nil) }
        let t0 = CFAbsoluteTimeGetCurrent()
        var first: Double?
        var text = ""
        var line = ""
        var lastOutput = t0
        for await ev in await engine.generate(input: send, parameters: p) {
            if case .chunk = ev { lastOutput = CFAbsoluteTimeGetCurrent() }
            if case .reasoning = ev { lastOutput = CFAbsoluteTimeGetCurrent() }
            switch ev {
            case .chunk(let c):
                if first == nil { first = CFAbsoluteTimeGetCurrent() - t0 }
                text += c
            case .reasoning:
                if first == nil { first = CFAbsoluteTimeGetCurrent() - t0 }
            case .info(let info):
                line = String(
                    format: "[RIPPLE_TTFT] arm=%@ turn=%d prompt=%d promptTime=%.2fs ttft=%.2fs gen=%d decode=%.2f tok/s",
                    arm, turn, info.promptTokenCount, info.promptTime, first ?? -1, info.generationTokenCount,
                    info.tokensPerSecond)
            default: break
            }
        }
        let tail = CFAbsoluteTimeGetCurrent() - lastOutput
        let diskDir = cfg.diskCacheDir?.path ?? ""
        let files = (try? FileManager.default.subpathsOfDirectory(atPath: diskDir)) ?? []
        let bytes = files.reduce(0) { sum, name in
            sum + (((try? FileManager.default.attributesOfItem(atPath: diskDir + "/" + name))?[.size] as? Int) ?? 0)
        }
        print(line + String(format: " tail=%.3fs disk=%.2fGB active=%.2fGB", tail, Double(bytes) / 1e9, Double(MLX.Memory.activeMemory) / 1e9))
        if let out = env["BENCH_RIPPLE_TEXT_OUT"] {
            let url = URL(fileURLWithPath: out + "-turn\(turn).txt")
            try? text.write(to: url, atomically: true, encoding: .utf8)
        }
        history.append(.assistant(text))
    }
}

// BENCH_RIPPLE_CHOOSER=1 — paired chooser arms on one loaded model at the app's context length: every arm runs every prompt
// (prose and code, BENCH_RIPPLE_SAMPLED=1 for T 1.0 / top-p 0.95, fixed seed per prompt and repeat) in ABBA order, through
// BatchEngine with a disk-backed prefix cache. An arm is "name:ENV=VAL,ENV=VAL" (BENCH_RIPPLE_ARMS, ';'-separated); "ar" is
// the AR control. Prints one row per generation; the decode-end diagnostic carries widths and acceptance.
func runRippleChooserProbe(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let dir = URL(fileURLWithPath: modelPath)
    MLX.Memory.cacheLimit = 8 << 30
    let context = try await MLXLMCommon.loadModel(
        from: dir, using: #huggingFaceTokenizerLoader(), loadConfiguration: .default)
    let cacheDir = URL(fileURLWithPath: env["BENCH_RIPPLE_CACHE_DIR"] ?? "/tmp/ripple-chooser-cache")
    let cfg = VMLXServerRuntimeSettings().cacheCoordinatorConfig(
        modelKey: dir.lastPathComponent, diskCacheDirectory: cacheDir)
    let coordinator = CacheCoordinator(config: cfg)
    let engine = BatchEngine(context: context, maxBatchSize: 1, cacheCoordinator: coordinator)
    let sentence = "Policy note: archive shipping manifests, tide tables and lighthouse logs by quarter, and cite the ledger page. "
    let system = "You are a careful assistant for a harbor office.\n" + String(repeating: sentence, count: Int(env["BENCH_RIPPLE_SYS_REPEAT"] ?? "") ?? 80)
    let prompts: [(String, String)] = [
        ("tides", "Explain in about 300 words how ocean tides work, written for a curious 12-year-old."),
        ("bread", "Write about 300 words on why sourdough bread rises, in a warm conversational tone."),
        ("story", "Write a short story of about 300 words about a lighthouse keeper who finds a message in a bottle."),
        ("code", "Write a Python function that parses a CSV file of orders (id, customer, amount, date) and returns total revenue per customer per month, with type hints and a short docstring. Code only."),
    ]
    let arms: [(name: String, vars: [(String, String)])] = (env["BENCH_RIPPLE_ARMS"] ?? "ar;cur:").split(separator: ";").map { raw in
        let parts = raw.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
        let vars = parts.count > 1 ? parts[1].split(separator: ",").compactMap { kv -> (String, String)? in
            let f = kv.split(separator: "=", maxSplits: 1)
            return f.count == 2 ? (String(f[0]), String(f[1])) : nil
        } : []
        return (String(parts[0]), vars)
    }
    let sampled = env["BENCH_RIPPLE_SAMPLED"] == "1"
    let maxTokens = Int(env["BENCH_RIPPLE_MAXTOK"] ?? "") ?? 400
    let reps = Int(env["BENCH_RIPPLE_REPS"] ?? "") ?? 2
    var order: [Int] = []
    for rep in 0 ..< reps { order += rep % 2 == 0 ? Array(arms.indices) : Array(arms.indices.reversed()) }
    var seedByRep = 0
    for (position, armIndex) in order.enumerated() {
        let arm = arms[armIndex]
        for a in arms { for (k, _) in a.vars { unsetenv(k) } }
        for (k, v) in arm.vars { setenv(k, v, 1) }
        seedByRep = position / arms.count
        for (name, user) in prompts {
            let history: [Chat.Message] = [.system(system), .user(user)]
            var input = UserInput(chat: history)
            input.additionalContext = [:]
            let prepared = try await context.processor.prepare(input: input)
            nonisolated(unsafe) let send = prepared
            var p = GenerateParameters(
                maxTokens: maxTokens, temperature: sampled ? 1 : 0, topP: sampled ? 0.95 : 1, topK: 0, minP: 0,
                repetitionPenalty: nil)
            if sampled { p.randomSeed = 7000 + UInt64(seedByRep) }
            if arm.name != "ar" {
                p.draftStrategy = .dflash2(drafterPath: dir.appendingPathComponent("dflash2"), blockSize: nil)
            }
            var row = ""
            for await ev in await engine.generate(input: send, parameters: p) {
                if case .info(let info) = ev {
                    row = String(
                        format: "[RIPPLE_CHOOSER] arm=%@ prompt=%@ rep=%d ctx=%d gen=%d tps=%.2f", arm.name, name, seedByRep,
                        info.promptTokenCount, info.generationTokenCount, info.tokensPerSecond)
                }
            }
            print(row)
            fflush(stdout)
            try await Task.sleep(for: .seconds(2))
        }
    }
}
