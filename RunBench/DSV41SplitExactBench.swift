import CryptoKit
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
@preconcurrency import VMLXTokenizers

/// BENCH_DSV41_SPLIT_EXACT=1 — the exactness gate of a partial prefix restore on DeepSeek-V4.1.
///
/// For each `n:b` in `SX_CASES` (prompt length : restore boundary, fixed token ids from `SX_IDS`, a little-endian int32
/// file), with the served lane (dSpark head resident, caller step `SX_STEP`):
///
///   split    cold prefill split at b: the head [0, b) prefilled as its own prompt (`prepare` + remainder forward, the
///            TokenIterator boundary split), a deep copy of the cache taken there, then [b, n-1) the same way and the
///            last token forwarded alone (the prompt-minus-one split every request takes);
///   memory   the copy at b stored through a CacheCoordinator (write-behind, resident entries on), fetched back for the
///            n-token prompt (`tier` must be memory), restored with `restoreFromDiskArrays`, then [b, n) as in `split`;
///   disk     the same entry read by a second coordinator on the same directory with no resident entries (`tier`
///            must be disk: the safetensors file), then [b, n) as in `split`;
///   unsplit  cold prefill of [0, n-1) then the last token (no split at b).
///
/// Per arm: sha256 of the fp32 last-row logits, of every layer's lanes (offset, every state array, meta state), the
/// dSpark hidden tail, and `SX_GREEDY` (16) greedy tokens with their logits. The gate: memory == split and disk == split
/// on all of them; unsplit vs split is reported (top-1, greedy), not gated. One `SX_CASE` JSON line per case.
func runDSV41SplitExactBench(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let modelDir = URL(fileURLWithPath: modelPath)
    let step = Int(env["SX_STEP"] ?? "512") ?? 512
    let greedyCount = Int(env["SX_GREEDY"] ?? "16") ?? 16
    let cases: [(n: Int, b: Int)] = (env["SX_CASES"] ?? "9371:7333,8536:4099,5123:2479,3001:2861")
        .split(separator: ",").compactMap { pair in
            let parts = pair.split(separator: ":").compactMap { Int($0) }
            return parts.count == 2 && parts[1] > 0 && parts[1] < parts[0] - 1 ? (parts[0], parts[1]) : nil
        }
    guard let idsPath = env["SX_IDS"], let raw = FileManager.default.contents(atPath: idsPath), raw.count >= 4
    else {
        throw NSError(domain: "BENCH_DSV41_SPLIT_EXACT", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "set SX_IDS to a little-endian int32 token id file"])
    }
    let ids: [Int32] = raw.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
    let out = env["SX_OUT"]
    func emit(_ tag: String, _ record: [String: Any]) {
        let line = String(
            decoding: (try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])) ?? Data(),
            as: UTF8.self)
        print("[\(tag)] \(line)")
        fflush(stdout)
        if let out, let data = "[\(tag)] \(line)\n".data(using: .utf8) {
            if let h = FileHandle(forWritingAtPath: out) {
                h.seekToEndOfFile(); h.write(data); try? h.close()
            } else {
                FileManager.default.createFile(atPath: out, contents: data)
            }
        }
    }

    let configData = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
    let jang = try? JangLoader.loadConfig(at: modelDir)
    let status = try? MTPBundleInspector.inspect(modelDirectory: modelDir, jangConfig: jang)
    let settings = VMLXServerRuntimeSettings()
    let loadConfiguration = settings.resolvedLoadConfiguration(
        configData: configData, jangConfig: jang, status: status, bundleDirectory: modelDir)
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader(), loadConfiguration: loadConfiguration)
    let model = context.model
    var headAttached = false
    if env["SX_HEAD"] != "0", let target = model as? any DSparkSpeculativeModel {
        headAttached = try target.attachDSparkDraft(directory: modelDir.appendingPathComponent("dspark"))
    }
    emit("SX_LOAD", ["model": modelDir.lastPathComponent, "head": headAttached, "step": step,
        "cases": cases.map { "\($0.n):\($0.b)" }, "type": String(describing: type(of: model))])

    func sha(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    func bytes(_ a: MLXArray) -> Data { MLX.eval(a); return a.asData(access: .copy).data }
    func tokens(_ slice: ArraySlice<Int32>) -> MLXArray { MLXArray(Array(slice)).reshaped(1, -1) }

    /// One segment prefilled as its own prompt: `prepare` (pruned chunks aimed at the segment's end), then the
    /// remainder forward. Returns the last row's fp32 logits.
    func segment(_ slice: ArraySlice<Int32>, _ cache: [KVCache]) throws -> MLXArray {
        var logits: MLXArray
        switch try model.prepare(LMInput(text: LMInput.Text(tokens: tokens(slice))), cache: cache, windowSize: step) {
        case .tokens(let rest):
            let rows = rest.tokens.ndim == 1 ? rest.tokens.reshaped(1, -1) : rest.tokens
            logits = model(LMInput.Text(tokens: rows, mask: rest.mask), cache: cache, state: nil).logits
        case .logits(let output):
            logits = output.logits
        }
        logits = logits[0..., -1, 0...].asType(.float32)
        MLX.eval(logits)
        MLX.eval(cache)
        return logits
    }
    /// [start, n-1) as one segment, then the last token alone (the prompt-minus-one split).
    func finish(_ n: Int, from start: Int, _ cache: [KVCache]) throws -> MLXArray {
        if n - 1 > start { _ = try segment(ids[start ..< (n - 1)], cache) }
        return try segment(ids[(n - 1) ..< n], cache)
    }
    func fingerprint(_ first: MLXArray, _ cache: [KVCache]) -> [String: Any] {
        var lanes: [String] = []
        for (i, c) in cache.enumerated() {
            var h = SHA256()
            h.update(data: Data("\(c.offset)".utf8))
            for a in c.state { h.update(data: Data("\(a.shape)\(a.dtype)".utf8)); h.update(data: bytes(a)) }
            h.update(data: Data(c.metaState.joined(separator: "|").utf8))
            lanes.append("\(i):\(c.offset):" + h.finalize().map { String(format: "%02x", $0) }.joined())
        }
        var tail = "none"
        for c in cache {
            if let t = (c as? HybridPoolCache)?.auxiliaryHiddenTail, let j = t.joined() {
                tail = "\(j.start)-\(t.endPosition):" + sha(bytes(j.rows))
            }
        }
        var logits = first
        var greedy: [Int] = []
        var gh = SHA256()
        for _ in 0 ..< greedyCount {
            let t = logits.argMax(axis: -1).item(Int.self)
            greedy.append(t)
            gh.update(data: bytes(logits))
            logits = model(LMInput.Text(tokens: MLXArray([Int32(t)]).reshaped(1, 1)), cache: cache, state: nil)
                .logits[0..., -1, 0...].asType(.float32)
        }
        let flat = first.reshaped(-1)
        let top2 = MLX.argSort(flat, axis: -1)[(-2)...].asArray(Int.self)
        let values = flat.asArray(Float.self)
        return [
            "logits_sha": sha(bytes(first)), "lanes_sha": sha(Data(lanes.joined(separator: "\n").utf8)),
            "lanes": lanes, "tail": tail, "greedy": greedy,
            "greedy_logits_sha": gh.finalize().map { String(format: "%02x", $0) }.joined(),
            "top1": top2.last ?? -1, "top2_margin": top2.count == 2 ? values[top2[1]] - values[top2[0]] : 0,
        ]
    }

    let root = URL(fileURLWithPath: env["SX_CACHE_DIR"] ?? NSTemporaryDirectory())
        .appendingPathComponent("dsv41-split-exact-\(ProcessInfo.processInfo.processIdentifier)")
    var allPass = true
    for (index, c) in cases.enumerated() where c.n <= ids.count {
        let dir = root.appendingPathComponent("case\(index)")
        try? FileManager.default.removeItem(at: dir)
        MLX.Memory.clearCache()
        // split: the head as its own prompt, a copy at b, then the rest.
        let splitCache = model.newCache(parameters: nil)
        let t0 = CFAbsoluteTimeGetCurrent()
        _ = try segment(ids[0 ..< c.b], splitCache)
        let headSeconds = CFAbsoluteTimeGetCurrent() - t0
        let atBoundary = splitCache.map { $0.copy() }
        MLX.eval(atBoundary)
        let t1 = CFAbsoluteTimeGetCurrent()
        let splitFirst = try finish(c.n, from: c.b, splitCache)
        let tailSeconds = CFAbsoluteTimeGetCurrent() - t1
        let split = fingerprint(splitFirst, splitCache)

        // memory: store the copy, fetch the n-token prompt back from the same coordinator.
        var config = CacheCoordinatorConfig(usePagedCache: false, enableDiskCache: true, diskCacheDir: dir,
            modelKey: "dsv41-split-exact")
        config.diskWriteBehind = true
        config.residentEntryBytes = 64 << 30
        let writer = CacheCoordinator(config: config)
        writer.setPagedIncompatible(true)
        let boundaryTokens = ids[0 ..< c.b].map(Int.init)
        let prompt = ids[0 ..< c.n].map(Int.init)
        writer.storeAfterGeneration(promptTokens: boundaryTokens, perLayerData: [], ssmStates: nil,
            cache: atBoundary)
        func restore(_ coordinator: CacheCoordinator) throws -> (fp: [String: Any], tier: String, matched: Int) {
            guard case .hit(let matched, _, let detail, _, _, let arrays?) = coordinator.fetch(
                tokens: prompt, skipExactDiskBoundary: true)
            else { return (["miss": true], "miss", 0) }
            var cache = model.newCache(parameters: nil)
            let restored = restoreFromDiskArrays(arrays, into: &cache, requirePromptBoundary: true)
            MLX.eval(cache)
            guard restored == matched, matched == c.b else {
                return (["restored": restored], detail.rawValue, matched)
            }
            let first = try finish(c.n, from: matched, cache)
            return (fingerprint(first, cache), detail.rawValue, matched)
        }
        let memory = try restore(writer)
        _ = writer.waitForPendingDiskWrites()
        // disk: a second coordinator on the same directory, nothing resident: the file.
        var readConfig = config
        readConfig.residentEntryBytes = 0
        let reader = CacheCoordinator(config: readConfig)
        reader.setPagedIncompatible(true)
        let disk = try restore(reader)

        // unsplit: no cut at b.
        let unsplitCache = model.newCache(parameters: nil)
        let t2 = CFAbsoluteTimeGetCurrent()
        let unsplitFirst = try finish(c.n, from: 0, unsplitCache)
        let unsplitSeconds = CFAbsoluteTimeGetCurrent() - t2
        let unsplit = fingerprint(unsplitFirst, unsplitCache)

        func same(_ a: [String: Any], _ b: [String: Any]) -> Bool {
            ["logits_sha", "lanes_sha", "tail", "greedy_logits_sha"].allSatisfy {
                (a[$0] as? String) != nil && (a[$0] as? String) == (b[$0] as? String)
            }
        }
        let memoryPass = memory.tier == "memory" && same(memory.fp, split)
        let diskPass = disk.tier == "disk" && same(disk.fp, split)
        allPass = allPass && memoryPass && diskPass
        emit("SX_CASE", [
            "n": c.n, "b": c.b, "memory_tier": memory.tier, "disk_tier": disk.tier,
            "split_head_s": headSeconds, "split_tail_s": tailSeconds, "unsplit_s": unsplitSeconds,
            "memory_matched": memory.matched, "disk_matched": disk.matched,
            "memory_equals_split": memoryPass, "disk_equals_split": diskPass,
            "unsplit_top1_equals_split": (unsplit["top1"] as? Int) == (split["top1"] as? Int),
            "unsplit_greedy_equals_split": (unsplit["greedy"] as? [Int]) == (split["greedy"] as? [Int]),
            "unsplit_logits_equal_split": (unsplit["logits_sha"] as? String) == (split["logits_sha"] as? String),
            "split": split.filter { $0.key != "lanes" }, "memory": memory.fp.filter { $0.key != "lanes" },
            "disk": disk.fp.filter { $0.key != "lanes" }, "unsplit": unsplit.filter { $0.key != "lanes" },
            "lane_diffs_memory": zip((split["lanes"] as? [String]) ?? [], (memory.fp["lanes"] as? [String]) ?? [])
                .filter { $0.0 != $0.1 }.count,
            "lane_diffs_disk": zip((split["lanes"] as? [String]) ?? [], (disk.fp["lanes"] as? [String]) ?? [])
                .filter { $0.0 != $0.1 }.count,
            "resident": ["entries": writer.diskCache?.residentStats.entries ?? -1,
                "bytes": writer.diskCache?.residentStats.bytes ?? -1],
        ])
        try? FileManager.default.removeItem(at: dir)
    }
    emit("SX_DONE", ["all_pass": allPass])
}
