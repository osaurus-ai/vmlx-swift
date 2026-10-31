import CryptoKit
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
@preconcurrency import VMLXTokenizers

/// BENCH_DSV41_MISS_PARITY=1 — why a request that restores nothing can answer differently on an engine with a prefix
/// cache than on one without (DeepSeek-V4.1; the JANG_2K "B turn" of `BENCH_DSV41_PREFILL`).
///
/// Prompts from fixed ids (`MP_IDS`, little-endian int32): A = ids[0, 9371), B = ids[0, 7333) + ids[130000, 131203)
/// (B shares only 7,333 ids with A, so with A stored B finds no prefix and restores 0). Greedy, `MP_TOKENS` (24) tokens.
///
/// Engine rows (`MP_ENGINE`), each a sequence of requests on ONE BatchEngine, dSpark as served (`MP_AR=1`: AR):
///   cold_BAB      no cache: B, A, B (does a request depend on the request before it?)
///   warm_AB       disk prefix cache (paged off, the bench's coordinator): A, B (the reported divergence)
///   warm_B        a fresh disk prefix cache: B first (a miss with nothing stored at all)
/// Model rows (no engine, AR greedy on the logits): B prefilled unsplit (`prepare` + remainder forward, what a cacheless
/// engine runs) and B split before its last token ([0, n-1) then [n-1, n): the prompt-minus-one seed split every
/// request on a cache engine takes for this cache topology).
func runDSV41MissParityBench(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let modelDir = URL(fileURLWithPath: modelPath)
    let step = Int(env["MP_STEP"] ?? "512") ?? 512
    let maxNew = Int(env["MP_TOKENS"] ?? "24") ?? 24
    guard let idsPath = env["MP_IDS"], let raw = FileManager.default.contents(atPath: idsPath), raw.count >= 4
    else {
        throw NSError(domain: "BENCH_DSV41_MISS_PARITY", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "set MP_IDS to a little-endian int32 token id file"])
    }
    let ids: [Int32] = raw.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
    func emit(_ tag: String, _ record: [String: Any]) {
        let line = String(
            decoding: (try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])) ?? Data(),
            as: UTF8.self)
        print("[\(tag)] \(line)")
        fflush(stdout)
    }
    let configData = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
    let jang = try? JangLoader.loadConfig(at: modelDir)
    let status = try? MTPBundleInspector.inspect(modelDirectory: modelDir, jangConfig: jang)
    let settings = VMLXServerRuntimeSettings()
    let strategy = env["MP_AR"] == "1" ? nil : settings.resolvedMTPDraftStrategy(
        configData: configData, jangConfig: jang, status: status, bundleDirectory: modelDir)
    let loadConfiguration = settings.resolvedLoadConfiguration(
        configData: configData, jangConfig: jang, status: status, bundleDirectory: modelDir)
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader(), loadConfiguration: loadConfiguration)
    let model = context.model
    let promptA = Array(ids[0 ..< 9371])
    let promptB = Array(ids[0 ..< 7333]) + Array(ids[130_000 ..< 131_203])
    emit("MP_LOAD", ["model": modelDir.lastPathComponent, "strategy": String(describing: strategy),
        "a": promptA.count, "b": promptB.count, "step": step])

    func sha(_ s: String) -> String { SHA256.hash(data: Data(s.utf8)).map { String(format: "%02x", $0) }.joined() }
    let root = URL(fileURLWithPath: env["MP_CACHE_DIR"] ?? NSTemporaryDirectory())
        .appendingPathComponent("dsv41-miss-parity-\(ProcessInfo.processInfo.processIdentifier)")
    func coordinator(_ name: String) -> CacheCoordinator {
        let dir = root.appendingPathComponent(name)
        try? FileManager.default.removeItem(at: dir)
        var cfg = CacheCoordinatorConfig()
        cfg.usePagedCache = false
        cfg.enableDiskCache = true
        cfg.diskCacheDir = dir
        cfg.modelKey = modelDir.lastPathComponent
        let c = CacheCoordinator(config: cfg)
        c.setPagedIncompatible(true)
        return c
    }
    nonisolated(unsafe) let ctx = context
    func turn(_ engine: BatchEngine, _ prompt: [Int32], row: String, name: String) async -> String {
        var p = GenerateParameters(
            maxTokens: maxNew, temperature: 0, topP: 1, topK: 0, minP: 0, repetitionPenalty: nil)
        p.prefillStepSize = step
        p.draftStrategy = strategy
        let input = LMInput(text: LMInput.Text(tokens: MLXArray(prompt).reshaped(1, -1)))
        nonisolated(unsafe) let send = input
        var text = ""
        var restored = 0
        for await ev in await engine.generate(input: send, parameters: p) {
            switch ev {
            case .chunk(let c): text += c
            case .reasoning(let r): text += r
            case .prefillProgress(let pp) where pp.stage == .cacheRestore: restored = pp.completedUnitCount
            default: break
            }
        }
        emit("MP_TURN", ["row": row, "name": name, "prompt": prompt.count, "restored": restored,
            "text_sha": sha(text), "text": text])
        return text
    }
    var texts: [String: String] = [:]
    let rows = Set((env["MP_ROWS"] ?? "cold_BAB,warm_AB,warm_B,model").split(separator: ",").map(String.init))
    if rows.contains("cold_BAB") {
        let cold = BatchEngine(context: ctx, maxBatchSize: 1, cacheCoordinator: nil)
        texts["cold_B1"] = await turn(cold, promptB, row: "cold_BAB", name: "B1")
        texts["cold_A"] = await turn(cold, promptA, row: "cold_BAB", name: "A")
        texts["cold_B2"] = await turn(cold, promptB, row: "cold_BAB", name: "B2")
    }
    if rows.contains("warm_AB") {
        let warm = BatchEngine(context: ctx, maxBatchSize: 1, cacheCoordinator: coordinator("ab"))
        texts["warm_A"] = await turn(warm, promptA, row: "warm_AB", name: "A")
        texts["warm_B"] = await turn(warm, promptB, row: "warm_AB", name: "B")
    }
    if rows.contains("warm_B") {
        let fresh = BatchEngine(context: ctx, maxBatchSize: 1, cacheCoordinator: coordinator("b"))
        texts["fresh_B"] = await turn(fresh, promptB, row: "warm_B", name: "B")
    }
    if rows.contains("model") {
        // AR greedy on the logits, as text, for the two prefill partitions.
        func greedy(_ splitBeforeLast: Bool) throws -> String {
            let cache = model.newCache(parameters: nil)
            func forward(_ slice: ArraySlice<Int32>) throws -> MLXArray {
                let tokens = MLXArray(Array(slice)).reshaped(1, -1)
                switch try model.prepare(LMInput(text: LMInput.Text(tokens: tokens)), cache: cache, windowSize: step) {
                case .tokens(let rest):
                    let rows = rest.tokens.ndim == 1 ? rest.tokens.reshaped(1, -1) : rest.tokens
                    return model(LMInput.Text(tokens: rows, mask: rest.mask), cache: cache, state: nil)
                        .logits[0..., -1, 0...]
                case .logits(let out):
                    return out.logits[0..., -1, 0...]
                }
            }
            let n = promptB.count
            var logits = splitBeforeLast
                ? try { _ = try forward(promptB[0 ..< (n - 1)]); return try forward(promptB[(n - 1) ..< n]) }()
                : try forward(promptB[0 ..< n])
            var out: [Int] = []
            for _ in 0 ..< maxNew {
                let t = logits.argMax(axis: -1).item(Int.self)
                out.append(t)
                logits = model(LMInput.Text(tokens: MLXArray([Int32(t)]).reshaped(1, 1)), cache: cache, state: nil)
                    .logits[0..., -1, 0...]
            }
            return context.tokenizer.decode(tokenIds: out)
        }
        texts["model_unsplit"] = try greedy(false)
        texts["model_split_n1"] = try greedy(true)
        emit("MP_MODEL", ["unsplit": texts["model_unsplit"]!, "split_n1": texts["model_split_n1"]!])
    }
    var pairs: [String: Bool] = [:]
    let keys = texts.keys.sorted()
    for (i, a) in keys.enumerated() { for b in keys[(i + 1)...] { pairs["\(a)==\(b)"] = texts[a] == texts[b] } }
    emit("MP_EQUAL", pairs)
    try? FileManager.default.removeItem(at: root)
}
