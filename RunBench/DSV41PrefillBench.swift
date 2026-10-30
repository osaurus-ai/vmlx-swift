// BENCH_DSV41_PREFILL=1 — DeepSeek-V4.1 cold prefill, prompt-end identity and prefix-cache reuse on fixed token ids
// (diagnostic; the DSV4.1 prefill work of 2026-10-30).
//
// The prompt is raw token ids (BENCH_PF_IDS: little-endian int32), so every arm and every runtime prefills the same
// tokens. The bundle loads with its bundle-aware default configuration and the dSpark head attached (the served lane:
// 1024-row chunks with the head resident). Phases (BENCH_PF_PHASES, default speed,identity,cache):
//   speed     cold prefills of the first n ids for n in BENCH_PF_LENS, a fresh cache each: `prepare` (the chunk loop)
//             then the remainder forward, logits and cache evaluated. BENCH_PF_ROUNDS rounds after one discarded
//             round, lengths rotated per round. One JSON line per probe ("[PF_SPEED]"): seconds, pp/s, wall window
//             (unix), MLX peak / active / cache GiB.
//   identity  for n in BENCH_PF_ID_LENS: the same prefill, then sha256 of the fp32 logits, every layer's cache lanes
//             (offset, state arrays, meta state) and the dSpark hidden tail at the prompt end, then BENCH_PF_GREEDY
//             (default 16) greedy tokens and the sha256 of their logits ("[PF_ID]").
//   cache     BatchEngine with a disk prefix cache (fresh directory): turn A (cold), turn B (same first P ids as A,
//             another suffix: a partial hit), turn C (A plus new ids: an appending hit), each greedy, then B and C cold
//             on an engine without a cache. Restored tokens, TTFT and texts per turn ("[PF_CACHE]").
//   profile   for n in BENCH_PF_PROFILE_LENS (default 32845): uninstrumented and component-instrumented prefills
//             interleaved (`DeepseekV41PrefillProfile`), per-component seconds and the walls ("[PF_PROFILE]").
// BENCH_PF_STEP (default 512) is the caller's prefill step; BENCH_PF_OUT appends every JSON line to a file.

import CryptoKit
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import VMLXTokenizers

func runDSV41PrefillBench(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let modelDir = URL(fileURLWithPath: modelPath)
    func ints(_ key: String, _ fallback: String) -> [Int] {
        (env[key] ?? fallback).split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    }
    let phases = Set((env["BENCH_PF_PHASES"] ?? "speed,identity,cache").split(separator: ",").map(String.init))
    let lens = ints("BENCH_PF_LENS", "1382,6563,8269,26513,32845")
    let idLens = ints("BENCH_PF_ID_LENS", "8269,32845")
    let rounds = Int(env["BENCH_PF_ROUNDS"] ?? "3") ?? 3
    let step = Int(env["BENCH_PF_STEP"] ?? "512") ?? 512
    let greedyCount = Int(env["BENCH_PF_GREEDY"] ?? "16") ?? 16
    let label = env["BENCH_PF_LABEL"] ?? "run"
    guard let idsPath = env["BENCH_PF_IDS"], let raw = FileManager.default.contents(atPath: idsPath),
        raw.count >= 4
    else {
        throw NSError(domain: "BENCH_DSV41_PREFILL", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "set BENCH_PF_IDS to a little-endian int32 token id file"])
    }
    let ids: [Int32] = raw.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
    let outPath = env["BENCH_PF_OUT"]

    func emit(_ tag: String, _ record: [String: Any]) {
        var r = record
        r["label"] = label
        let line = String(
            decoding: (try? JSONSerialization.data(withJSONObject: r, options: [.sortedKeys])) ?? Data(),
            as: UTF8.self)
        print("[\(tag)] \(line)")
        fflush(stdout)
        if let outPath, let data = "[\(tag)] \(line)\n".data(using: .utf8) {
            if let h = FileHandle(forWritingAtPath: outPath) {
                h.seekToEndOfFile()
                h.write(data)
                try? h.close()
            } else {
                FileManager.default.createFile(atPath: outPath, contents: data)
            }
        }
    }
    let gib = 1_073_741_824.0
    func mem() -> [String: Double] {
        [
            "peak_gib": Double(MLX.Memory.peakMemory) / gib, "active_gib": Double(MLX.Memory.activeMemory) / gib,
            "cache_gib": Double(MLX.Memory.cacheMemory) / gib,
        ]
    }

    // Load as the app does for this bundle (SpeedProbe's "default" arm), dSpark resolved from the bundle.
    let configData = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
    let jang = try? JangLoader.loadConfig(at: modelDir)
    let status = try? MTPBundleInspector.inspect(modelDirectory: modelDir, jangConfig: jang)
    let settings = VMLXServerRuntimeSettings()
    let strategy = settings.resolvedMTPDraftStrategy(
        configData: configData, jangConfig: jang, status: status, bundleDirectory: modelDir)
    let loadConfiguration = settings.resolvedLoadConfiguration(
        configData: configData, jangConfig: jang, status: status, bundleDirectory: modelDir)
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader(), loadConfiguration: loadConfiguration)
    let loadSeconds = CFAbsoluteTimeGetCurrent() - loadStart
    if let gb = env["BENCH_PF_CACHE_LIMIT_GB"].flatMap(Double.init) {
        MLX.Memory.cacheLimit = Int(gb * gib)
    }
    let model = context.model
    var headAttached = false
    if env["BENCH_PF_HEAD"] != "0", let target = model as? any DSparkSpeculativeModel {
        headAttached = try target.attachDSparkDraft(directory: modelDir.appendingPathComponent("dspark"))
    }
    emit("PF_LOAD", [
        "model": modelDir.lastPathComponent, "load_s": loadSeconds, "head": headAttached,
        "strategy": String(describing: strategy), "memory_limit": MLX.Memory.memoryLimit,
        "cache_limit": MLX.Memory.cacheLimit, "ids": ids.count, "step": step,
        "env_trim": env["VMLX_DSV41_KV_SOURCE_TRIM"] ?? "", "type": String(describing: type(of: model)),
    ])

    func tokens(_ slice: ArraySlice<Int32>) -> MLXArray { MLXArray(Array(slice)).reshaped(1, -1) }

    /// The chunked prefill of `prompt` into a fresh cache, as the generation path runs it: `prepare` (the chunk loop),
    /// then the remainder forward. Returns the last row's fp32 logits and the cache.
    func prefill(_ prompt: MLXArray) throws -> (logits: MLXArray, cache: [KVCache]) {
        let cache = model.newCache(parameters: nil)
        var logits: MLXArray
        switch try model.prepare(LMInput(text: LMInput.Text(tokens: prompt)), cache: cache, windowSize: step) {
        case .tokens(let rest):
            // `prepare` hands back the remainder as a flat id list; the forward takes [1, n] (as TokenIterator does).
            let ids = rest.tokens.ndim == 1 ? rest.tokens.reshaped(1, -1) : rest.tokens
            logits = model(LMInput.Text(tokens: ids, mask: rest.mask), cache: cache, state: nil).logits
        case .logits(let out):
            logits = out.logits
        }
        logits = logits[0..., -1, 0...].asType(.float32)
        MLX.eval(logits)
        MLX.eval(cache)
        return (logits, cache)
    }

    func sha(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    func bytes(_ a: MLXArray) -> Data {
        MLX.eval(a)
        return a.asData(access: .copy).data
    }

    if phases.contains("speed") {
        for round in 0 ... rounds {
            let order = lens.isEmpty ? [] : (0 ..< lens.count).map { lens[($0 + round) % lens.count] }
            for n in order where n <= ids.count {
                MLX.Memory.clearCache()
                MLX.Memory.peakMemory = 0
                let w0 = Date().timeIntervalSince1970
                let t0 = CFAbsoluteTimeGetCurrent()
                let (logits, _) = try prefill(tokens(ids[0 ..< n]))
                let seconds = CFAbsoluteTimeGetCurrent() - t0
                let w1 = Date().timeIntervalSince1970
                let first = logits.argMax(axis: -1).item(Int.self)
                var r: [String: Any] = [
                    "n": n, "round": round, "discard": round == 0, "s": seconds, "pp_s": Double(n) / seconds,
                    "wall": [w0, w1], "first_token": first,
                ]
                for (k, v) in mem() { r[k] = v }
                emit("PF_SPEED", r)
            }
        }
    }

    // profile: per-component seconds of an instrumented prefill (`DeepseekV41PrefillProfile`: evaluate and wait at
    // every component boundary) next to uninstrumented walls of the same prompt, interleaved, first probe discarded.
    if phases.contains("profile") {
        for n in ints("BENCH_PF_PROFILE_LENS", "32845") where n <= ids.count {
            for (k, instrumented) in [false, false, true, false, true].enumerated() {
                MLX.Memory.clearCache()
                MLX.Memory.peakMemory = 0
                DeepseekV41PrefillProfile.reset()
                DeepseekV41PrefillProfile.active = instrumented
                let w0 = Date().timeIntervalSince1970
                let t0 = CFAbsoluteTimeGetCurrent()
                let (logits, _) = try prefill(tokens(ids[0 ..< n]))
                let seconds = CFAbsoluteTimeGetCurrent() - t0
                DeepseekV41PrefillProfile.active = false
                var r: [String: Any] = [
                    "n": n, "probe": k, "discard": k == 0, "instrumented": instrumented, "s": seconds,
                    "pp_s": Double(n) / seconds, "wall": [w0, Date().timeIntervalSince1970],
                    "first_token": logits.argMax(axis: -1).item(Int.self),
                ]
                if instrumented {
                    r["components"] = DeepseekV41PrefillProfile.seconds
                    r["counts"] = DeepseekV41PrefillProfile.counts
                    r["sum_s"] = DeepseekV41PrefillProfile.seconds.values.reduce(0, +)
                }
                for (key, v) in mem() { r[key] = v }
                emit("PF_PROFILE", r)
            }
        }
    }

    if phases.contains("identity") {
        for n in idLens where n <= ids.count {
            MLX.Memory.clearCache()
            let (first, cache) = try prefill(tokens(ids[0 ..< n]))
            var logits = first
            let logitsSha = sha(bytes(logits))
            let argmax = logits.argMax(axis: -1).item(Int.self)
            // Every lane at the prompt end, per layer.
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
            var greedy: [Int] = []
            var gh = SHA256()
            for _ in 0 ..< greedyCount {
                let t = logits.argMax(axis: -1).item(Int.self)
                greedy.append(t)
                gh.update(data: bytes(logits))
                logits = model(
                    LMInput.Text(tokens: MLXArray([Int32(t)]).reshaped(1, 1)), cache: cache, state: nil
                ).logits[0..., -1, 0...].asType(.float32)
            }
            emit("PF_ID", [
                "n": n, "logits_sha": logitsSha, "argmax": argmax, "lanes": lanes,
                "lanes_sha": sha(Data(lanes.joined(separator: "\n").utf8)), "tail": tail, "greedy": greedy,
                "greedy_logits_sha": gh.finalize().map { String(format: "%02x", $0) }.joined(),
            ])
        }
    }

    if phases.contains("cache") {
        let aLen = Int(env["BENCH_PF_CACHE_A"] ?? "9371") ?? 9371
        let shared = Int(env["BENCH_PF_CACHE_SHARED"] ?? "7333") ?? 7333
        let suffixFrom = Int(env["BENCH_PF_CACHE_SUFFIX_FROM"] ?? "130000") ?? 130_000
        let bSuffix = Int(env["BENCH_PF_CACHE_B_SUFFIX"] ?? "1203") ?? 1203
        let cAppend = Int(env["BENCH_PF_CACHE_C_APPEND"] ?? "517") ?? 517
        let maxNew = Int(env["BENCH_PF_CACHE_TOKENS"] ?? "24") ?? 24
        let promptA = Array(ids[0 ..< aLen])
        let promptB = Array(ids[0 ..< shared]) + Array(ids[suffixFrom ..< (suffixFrom + bSuffix)])
        let promptC = promptA + Array(ids[(suffixFrom + 10_000) ..< (suffixFrom + 10_000 + cAppend)])
        let dir = URL(fileURLWithPath: env["BENCH_PF_CACHE_DIR"] ?? NSTemporaryDirectory())
            .appendingPathComponent("dsv41-pf-cache-\(ProcessInfo.processInfo.processIdentifier)")
        try? FileManager.default.removeItem(at: dir)
        var cfg = CacheCoordinatorConfig()
        cfg.usePagedCache = false
        cfg.enableDiskCache = true
        cfg.diskCacheDir = dir
        cfg.modelKey = modelDir.lastPathComponent
        let coordinator = CacheCoordinator(config: cfg)
        coordinator.setPagedIncompatible(true)
        nonisolated(unsafe) let ctx = context
        let warm = BatchEngine(context: ctx, maxBatchSize: 1, cacheCoordinator: coordinator)
        let cold = BatchEngine(context: ctx, maxBatchSize: 1, cacheCoordinator: nil)
        func turn(_ engine: BatchEngine, _ prompt: [Int32], _ name: String) async throws -> String {
            var p = GenerateParameters(
                maxTokens: maxNew, temperature: 0, topP: 1, topK: 0, minP: 0, repetitionPenalty: nil)
            p.prefillStepSize = step
            p.draftStrategy = strategy
            let input = LMInput(text: LMInput.Text(tokens: MLXArray(prompt).reshaped(1, -1)))
            nonisolated(unsafe) let send = input
            let t0 = CFAbsoluteTimeGetCurrent()
            let w0 = Date().timeIntervalSince1970
            let stream = await engine.generate(input: send, parameters: p)
            var first: Double?
            var text = ""
            var restored = 0
            var restoreDetail = ""
            var generated = 0
            var promptSeconds = 0.0
            for await ev in stream {
                switch ev {
                case .chunk(let c):
                    if first == nil { first = CFAbsoluteTimeGetCurrent() }
                    text += c
                case .reasoning(let r):
                    if first == nil { first = CFAbsoluteTimeGetCurrent() }
                    text += r
                case .prefillProgress(let pp):
                    if pp.stage == .cacheRestore {
                        restored = max(restored, pp.completedUnitCount)
                        restoreDetail = pp.detail ?? ""
                    }
                case .info(let info):
                    generated = info.generationTokenCount
                    promptSeconds = info.promptTime
                default:
                    break
                }
            }
            emit("PF_CACHE", [
                "turn": name, "prompt": prompt.count, "restored": restored, "restore_detail": restoreDetail,
                "ttft": (first ?? t0) - t0, "prompt_s": promptSeconds, "generated": generated,
                "text_sha": sha(Data(text.utf8)), "preview": String(text.prefix(80)),
                "wall": [w0, Date().timeIntervalSince1970],
            ])
            return text
        }
        let a = try await turn(warm, promptA, "A_cold_store")
        let b = try await turn(warm, promptB, "B_partial")
        let c = try await turn(warm, promptC, "C_append")
        let bCold = try await turn(cold, promptB, "B_cold")
        let cCold = try await turn(cold, promptC, "C_cold")
        let aCold = try await turn(cold, promptA, "A_cold")
        emit("PF_CACHE_CHECK", [
            "A_equal": a == aCold, "B_partial_equal_cold": b == bCold, "C_append_equal_cold": c == cCold,
            "A_nonempty": !a.isEmpty, "B_nonempty": !b.isEmpty, "C_nonempty": !c.isEmpty,
        ])
        try? FileManager.default.removeItem(at: dir)
    }
}
