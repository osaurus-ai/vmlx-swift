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
