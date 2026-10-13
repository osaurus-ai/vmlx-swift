import CoreFoundation
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import VMLXTokenizers

private func diplodocusSendable(_ value: Any) -> any Sendable {
    if let object = value as? [String: Any] { return object.mapValues(diplodocusSendable) }
    if let array = value as? [Any] { return array.map(diplodocusSendable) }
    if let string = value as? String { return string }
    if let number = value as? NSNumber {
        if CFGetTypeID(number) == CFBooleanGetTypeID() { return number.boolValue }
        return number.doubleValue
    }
    return Optional<String>.none
}

/// Production BatchEngine route, native Swift template/parser, optional disk
/// cache. Defaults are read from the bundle; greedy is an explicit A/B option.
func runDiplodocusQualification(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    guard let probesPath = env["DIPLODOC_PROBES"], let output = env["DIPLODOC_OUTPUT"] else {
        throw NSError(domain: "DiplodocusBench", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing probe/output paths"])
    }
    let modelDir = URL(fileURLWithPath: modelPath)
    let config = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
    let settings = VMLXServerRuntimeSettings()
    let jang = try? JangLoader.loadConfig(at: modelDir)
    let status = try? MTPBundleInspector.inspect(modelDirectory: modelDir, jangConfig: jang)
    let strategy = settings.resolvedMTPDraftStrategy(
        configData: config, jangConfig: jang, status: status, bundleDirectory: modelDir)
    let context = try await MLXLMCommon.loadModel(from: modelDir, using: #huggingFaceTokenizerLoader())
    nonisolated(unsafe) let ctx = context
    let arm = env["DIPLODOC_ARM"] ?? "default"
    print("DIPLODOC_RESOLUTION arm=\(arm) strategy=\(String(describing: strategy)) model=\(type(of: ctx.model))")
    let coordinator: CacheCoordinator? = env["DIPLODOC_CACHE"].map { path in
        var c = CacheCoordinatorConfig()
        c.usePagedCache = false
        c.enableDiskCache = true
        c.diskCacheDir = URL(fileURLWithPath: path)
        c.modelKey = modelDir.lastPathComponent
        let coordinator = CacheCoordinator(config: c)
        coordinator.setPagedIncompatible(true)
        return coordinator
    }
    let engine = BatchEngine(context: ctx, maxBatchSize: 1, cacheCoordinator: coordinator)
    let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: probesPath))) as! [[String: Any]]
    let wanted = env["DIPLODOC_CASES"].map { Set($0.split(separator: ",").map(String.init)) }
    let directory = URL(fileURLWithPath: output)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    for probe in raw {
        let name = probe["name"] as! String
        if let wanted, !wanted.contains(name) { continue }
        let messages = (probe["messages"] as! [[String: Any]]).map { $0.mapValues(diplodocusSendable) }
        let tools = (probe["tools"] as? [[String: Any]])?.map { $0.mapValues(diplodocusSendable) }
        let effort = probe["effort"] as! String
        let input = try await ctx.processor.prepare(input: UserInput(
            messages: messages, tools: tools,
            additionalContext: ["reasoning_effort": effort, "tool_call_format": "xml"]))
        let actual = input.text.tokens.asArray(Int.self)
        // Probes without "ids" (other models' speed probes) skip the template-ID oracle.
        let expected = probe["ids"] as? [Int] ?? actual
        guard actual == expected else {
            throw NSError(domain: "DiplodocusBench", code: 2, userInfo: [NSLocalizedDescriptionKey: "Swift/Python template ID mismatch: \(name)"])
        }
        let cap = Int(env["DIPLODOC_MAX_TOKENS"] ?? "") ?? probe["max_tokens"] as! Int
        var p = GenerateParameters(
            generationConfig: ctx.configuration.generationDefaults,
            fallback: GenerateParameters(maxTokens: cap, prefillStepSize: 1024))
        p.maxTokens = cap
        p.prefillStepSize = 1024
        if env["DIPLODOC_GREEDY"] == "1" {
            p.temperature = 0; p.topP = 1; p.topK = 0; p.minP = 0
        }
        p.draftStrategy = arm == "ar" ? DraftStrategy.none : strategy
        if arm != "ar", let unoPath = env["DIPLODOC_UNO"] {
            p.draftStrategy = .uno(
                adapterPath: URL(fileURLWithPath: unoPath),
                blockSize: env["DIPLODOC_BLOCK"].flatMap(Int.init))
        } else if let pinned = env["DIPLODOC_BLOCK"].flatMap(Int.init),
            case .dflash2(let drafterPath, _)? = p.draftStrategy
        {
            p.draftStrategy = .dflash2(drafterPath: drafterPath, blockSize: pinned)
        }
        if arm != "ar", let drafter = env["DIPLODOC_DFLASH2"] {
            p.draftStrategy = .dflash2(
                drafterPath: URL(fileURLWithPath: drafter), blockSize: env["DIPLODOC_BLOCK"].flatMap(Int.init))
        }
        let passes = Int(env["DIPLODOC_PASSES"] ?? "1") ?? 1
        for pass in 0 ..< passes {
            nonisolated(unsafe) let prepared = input
            let start = Date()
            var text = "", reasoning = "", calls: [[String: Any]] = []
            var row: [String: Any] = ["case": name, "effort": effort, "arm": arm, "pass": pass,
                "target": modelPath, "template_ids_match": true, "temperature": p.temperature]
            for await item in await engine.generate(input: prepared, parameters: p) {
                switch item {
                case .chunk(let chunk): text += chunk
                case .reasoning(let chunk): reasoning += chunk
                case .toolCall(let call):
                    let arguments = try JSONSerialization.jsonObject(
                        with: JSONEncoder().encode(call.function.arguments))
                    calls.append([
                        "name": call.function.name, "arguments": arguments,
                        "argument_order": call.function.argumentOrder ?? [],
                    ])
                case .info(let info):
                    row["tokens"] = info.generationTokenCount
                    row["tokens_per_second"] = info.tokensPerSecond
                    row["prefill_seconds"] = info.promptTime
                    row["decode_seconds"] = info.generateTime
                    row["finish"] = String(describing: info.stopReason)
                    row["unclosed_reasoning"] = info.unclosedReasoning
                    row["protocol_failure"] = String(describing: info.toolCallProtocolFailure)
                    row["generation_failure"] = String(describing: info.generationFailure)
                default: break
                }
            }
            row["seconds"] = Date().timeIntervalSince(start)
            row["text"] = text; row["reasoning"] = reasoning; row["tool_calls"] = calls
            row["cache"] = coordinator.map { String(describing: $0.snapshotStats()) } ?? "disabled"
            let data = try JSONSerialization.data(withJSONObject: row, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: directory.appendingPathComponent("\(name)-\(arm)-\(pass).json"), options: .atomic)
            print("DIPLODOC_SERVED case=\(name) arm=\(arm) pass=\(pass) finish=\(row["finish"] ?? "missing") tok_s=\(row["tokens_per_second"] ?? "missing") calls=\(calls.count)")
        }
    }
}

/// Release-built component costs for one DFlash2 cycle (diagnostic). Every
/// timing drains the GPU; median of DIPLODOC_ITERS after three warm-ups.
func runDiplodocusComponentCosts(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    guard let drafterPath = env["DIPLODOC_DRAFTER"] else {
        throw NSError(domain: "DiplodocusBench", code: 3, userInfo: [NSLocalizedDescriptionKey: "Set DIPLODOC_DRAFTER"])
    }
    let context = try await MLXLMCommon.loadModel(
        from: URL(fileURLWithPath: modelPath), using: #huggingFaceTokenizerLoader())
    nonisolated(unsafe) let ctx = context
    guard let target = ctx.model as? any DFlash2Target else { return }
    let drafter = try DFlash2Loader.load(from: URL(fileURLWithPath: drafterPath))
    let iters = Int(env["DIPLODOC_ITERS"] ?? "") ?? 40
    func median(_ body: () -> MLXArray) -> Double {
        for _ in 0 ..< 3 { eval(body()) }
        var times: [Double] = []
        for _ in 0 ..< iters {
            let t0 = Date()
            eval(body())
            times.append(Date().timeIntervalSince(t0) * 1000)
        }
        return times.sorted()[times.count / 2]
    }
    func f(_ v: Double) -> String { String(format: "%.2f", v) }
    let layerIDs = drafter.config.dflash.targetLayerIds
    let contextRows = Int(env["DIPLODOC_CONTEXT"] ?? "") ?? 1024
    let ids = (0 ..< contextRows).map { 1000 + ($0 * 37) % 20000 }
    let cache = target.newCache(parameters: nil)
    var captured: [Int: MLXArray] = [:]
    for start in stride(from: 0, to: ids.count, by: 512) {
        let chunk = Array(ids[start ..< min(ids.count, start + 512)])
        let (_, c) = target.callAsFunction(
            MLXArray(chunk).expandedDimensions(axis: 0), cache: cache,
            captureLayerIDs: Set(layerIDs), recordPrefixCommitStates: false)
        captured = c
    }
    let context0 = extractContextFeature(captured: captured, targetLayerIDs: layerIDs)
    eval(context0)
    let draftCache = drafter.makeCache()
    let block = MLXArray([Int32(ids.last!)] + Array(repeating: Int32(250062), count: 7)).reshaped(1, 8)
    eval(drafter.hiddenStates(
        inputs: block, targetHidden: context0, cache: draftCache, embedder: target, logitsStart: 1))
    let primed = draftCache.map(\.offset)
    let steady = context0[0..., (-2)..., 0...]
    func resetDraft() {
        for (c, o) in zip(draftCache, primed) where c.isTrimmable && c.offset > o { _ = c.trim(c.offset - o) }
    }
    let hidden7 = MLXRandom.normal([1, 7, 4096]).asType(.bfloat16)
    let head = median { target.projectToLogits(hidden7) }
    let logits = target.projectToLogits(hidden7)
    eval(logits)
    let partition = median { argPartition(logits, kth: logits.dim(-1) - 16, axis: -1) }
    let backbone = median {
        resetDraft()
        return drafter.hiddenStates(
            inputs: block, targetHidden: steady, cache: draftCache, embedder: target, logitsStart: 1)
    }
    let propose = median {
        resetDraft()
        return drafter.propose(
            inputs: block, targetHidden: steady, cache: draftCache, embedder: target,
            temperature: 1.0, logitsStart: 1).tokens
    }
    var rows: [String] = []
    for (pass, width) in [("mlx", 1), ("mlx", 2), ("mlx", 4), ("mlx", 8), ("mlx", 16),
        ("lane", 1), ("lane", 2), ("lane", 4), ("lane", 8), ("lane", 16)]
    {
        if pass == "lane", width == 1 { LaneQMM.installForDFlash2Target(target) }
        let input = MLXArray(Array(ids.suffix(width))).reshaped(1, width)
        let ms = median {
            let out = target.callAsFunction(
                input, cache: cache, captureLayerIDs: Set(layerIDs), recordPrefixCommitStates: false).0
            for c in cache { _ = c.trim(width) }
            return out
        }
        rows.append("\(pass)-w\(width)=\(f(ms))")
    }
    print("DIPLODOC_COMPONENTS context=\(contextRows) head7=\(f(head)) partition7=\(f(partition)) drafter_backbone=\(f(backbone)) drafter_propose=\(f(propose)) target_rows \(rows.joined(separator: " ")) lane=\(LaneQMM.lastInstallReport.map { "\($0.lane)/\($0.skipped)" } ?? "none")")
}

/// Target forward cost by row count and context length (BENCH_DIPLODOC_FWD_SWEEP=1). Diagnostic.
func runK2ForwardSweep(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let context = try await MLXLMCommon.loadModel(
        from: URL(fileURLWithPath: modelPath), using: #huggingFaceTokenizerLoader())
    nonisolated(unsafe) let model = context.model
    if env["DIPLODOC_SWEEP_LANE"] == "1", let target = model as? any DFlash2Target {
        LaneQMM.installForDFlash2Target(target)
    }
    let iters = Int(env["DIPLODOC_ITERS"] ?? "") ?? 30
    let contexts = (env["DIPLODOC_SWEEP_CONTEXTS"] ?? "256,2304").split(separator: ",").compactMap { Int($0) }
    let rowsList = (env["DIPLODOC_SWEEP_ROWS"] ?? "1,4,8,9,12,16").split(separator: ",").compactMap { Int($0) }
    for contextRows in contexts {
        let ids = (0 ..< contextRows).map { 1000 + ($0 * 37) % 20000 }
        let cache = model.newCache(parameters: nil)
        for start in stride(from: 0, to: ids.count, by: 512) {
            let chunk = Array(ids[start ..< min(ids.count, start + 512)])
            eval(model(MLXArray(chunk).expandedDimensions(axis: 0), cache: cache))
        }
        for rows in rowsList {
            let input = MLXArray((0 ..< rows).map { Int32(2000 + $0) }).reshaped(1, rows)
            var times: [Double] = []
            for i in 0 ..< (iters + 3) {
                let t0 = Date()
                eval(model(input, cache: cache))
                if i >= 3 { times.append(Date().timeIntervalSince(t0) * 1000) }
                for c in cache { _ = c.trim(rows) }
            }
            let med = times.sorted()[times.count / 2]
            print(String(format: "FWD_SWEEP ctx=%d rows=%d median_ms=%.2f", contextRows, rows, med))
        }
    }
}

/// Teacher-forced row-exactness of K2 multi-row forwards (BENCH_DIPLODOC_ROWEXACT=1). Diagnostic.
/// Greedy AR path of DIPLODOC_ROWEXACT_TOKENS tokens after a DIPLODOC_SWEEP_CONTEXTS-long prompt; then, per
/// width W, a fresh prefill and the same tokens fed W rows at a time. Reports rows whose logits are
/// bitwise equal to the 1-row step and rows whose argmax differs.
func runK2RowExactness(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let context = try await MLXLMCommon.loadModel(
        from: URL(fileURLWithPath: modelPath), using: #huggingFaceTokenizerLoader())
    nonisolated(unsafe) let model = context.model
    if env["DIPLODOC_SWEEP_LANE"] == "1", let target = model as? any DFlash2Target {
        LaneQMM.installForDFlash2Target(target)
    }
    let steps = Int(env["DIPLODOC_ROWEXACT_TOKENS"] ?? "") ?? 96
    let widths = (env["DIPLODOC_ROWEXACT_WIDTHS"] ?? "5,8,12").split(separator: ",").compactMap { Int($0) }
    let contexts = (env["DIPLODOC_SWEEP_CONTEXTS"] ?? "300,2300").split(separator: ",").compactMap { Int($0) }
    for contextRows in contexts {
        let prompt = (0 ..< contextRows).map { 1000 + ($0 * 37) % 20000 }
        func prefill() -> ([KVCache], MLXArray) {
            let cache = model.newCache(parameters: nil)
            var last = MLXArray(0)
            for start in stride(from: 0, to: prompt.count, by: 512) {
                let chunk = Array(prompt[start ..< min(prompt.count, start + 512)])
                last = model(MLXArray(chunk).expandedDimensions(axis: 0), cache: cache)[0, -1]
            }
            eval(last)
            return (cache, last)
        }
        var (cache, logits) = prefill()
        var tokens: [Int] = []
        var reference: [MLXArray] = []
        for _ in 0 ..< steps {
            let t = argMax(logits).item(Int.self)
            tokens.append(t)
            logits = model(MLXArray([Int32(t)]).reshaped(1, 1), cache: cache)[0, -1]
            eval(logits)
            reference.append(logits)
        }
        for w in widths {
            (cache, _) = prefill()
            var bitwise = 0, argmaxDiff = 0, rows = 0
            var maxDiff: Float = 0
            var start = 0
            while start + w <= steps {
                let block = tokens[start ..< start + w].map { Int32($0) }
                let out = model(MLXArray(block).reshaped(1, w), cache: cache)[0]
                eval(out)
                for i in 0 ..< w {
                    let r = reference[start + i]
                    let d = abs(out[i].asType(.float32) - r.asType(.float32)).max().item(Float.self)
                    maxDiff = max(maxDiff, d)
                    if d == 0 { bitwise += 1 }
                    if argMax(out[i]).item(Int.self) != argMax(r).item(Int.self) { argmaxDiff += 1 }
                    rows += 1
                }
                start += w
            }
            print("ROWEXACT ctx=\(contextRows) W=\(w) rows=\(rows) bitwise=\(bitwise) argmax_diff=\(argmaxDiff) max_logit_diff=\(maxDiff)")
        }
    }
}
