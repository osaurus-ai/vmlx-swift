//
//  Ling26GLAProbe.swift
//  vmlx-swift — DIAGNOSTIC PROBE, not a regression test (skips unless env is set).
//
//  osaurus#2652: Ling 2.6 flash JANGTQ answered "!!!!…" (token 0) from 0.24.5
//  (vmlx 97676e19) and was refused outright from 0.24.7 (#424). Loads the bundle
//  through the factory, renders a chat prompt with the bundle's own template
//  (a long system prompt like the app sends, then a short user turn), reports
//  whether the last-position logits are finite and what greedy decode produces.
//
//  Env: VMLX_LING26_MODEL_DIR=<bundle dir>
//       VMLX_LING26_SYSTEM_TOKENS=<approx system prompt length, default 400>
//       VMLX_LING26_MAXTOK=<greedy tokens, default 24>
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import VMLXTokenizers
import VMLXHub
import Testing

@Suite(.serialized)
struct Ling26GLAProbe {

    @Test("Ling 2.6 flash JANGTQ: finite logits and a real answer through the GLA runtime")
    func gla() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelDir = env["VMLX_LING26_MODEL_DIR"] else {
            print("PROBE SKIPPED: set VMLX_LING26_MODEL_DIR")
            return
        }
        let systemWords = Int(env["VMLX_LING26_SYSTEM_TOKENS"] ?? "") ?? 400
        let maxTok = Int(env["VMLX_LING26_MAXTOK"] ?? "") ?? 24
        let userText = env["VMLX_LING26_USER"] ?? "Hello"

        let context = try await LLMModelFactory.shared.load(
            from: URL(fileURLWithPath: modelDir), using: #huggingFaceTokenizerLoader())
        let tokenizer = context.tokenizer
        let model = context.model
        print("PROBE model=\(type(of: model)) dir=\(modelDir)")
        // Effective parameter dtypes after load: the embedding scales and one
        // attention projection tell whether the loader materialised bf16.
        let flat = model.parameters().flattened()
        let samples = ["model.word_embeddings.scales", "model.word_embeddings.weight", "model.layers.0.attention.query_key_value.scales", "model.layers.7.attention.q_a_proj.scales", "model.layers.0.attention.dense.scales", "model.layers.0.input_layernorm.weight", "model.norm.weight", "lm_head.scales"]
        for name in samples {
            if let arr = flat.first(where: { $0.0 == name })?.1 { print("PROBE dtype \(name) = \(arr.dtype)") }
        }
        let dtypeCounts = Dictionary(grouping: flat.map { String(describing: $0.1.dtype) }, by: { $0 }).mapValues(\.count)
        print("PROBE dtype histogram: \(dtypeCounts)")

        // A system prompt in the app's shape (long, prose) so the GLA prefill
        // runs well past the ~80-token horizon where the fp16 overflow lived.
        let sentence = "You are Osaurus, a careful assistant running locally on this Mac; answer briefly, cite files when you read them, and never invent tool results. "
        var system = ""
        while system.split(separator: " ").count < systemWords { system += sentence }
        let messages: [[String: String]] = [
            ["role": "system", "content": system],
            ["role": "user", "content": userText],
        ]
        let prefillStep = Int(env["VMLX_LING26_PREFILL_STEP"] ?? "") ?? 512
        let prompt: [Int]
        if let rawPath = env["VMLX_LING26_RAW_PROMPT_FILE"] {
            // The app's own rendered prompt (a VMLX_REASONING_PROMPT_DUMP_DIR
            // file: header lines, then the prompt text after the first blank line).
            let raw = try String(contentsOfFile: rawPath, encoding: .utf8)
            let body = raw.range(of: "\n\n").map { String(raw[$0.upperBound...]) } ?? raw
            prompt = tokenizer.encode(text: body, addSpecialTokens: false)
            print("PROBE raw prompt from \(rawPath): \(prompt.count) tokens")
        } else {
            prompt = try tokenizer.applyChatTemplate(messages: messages)
        }
        print("PROBE prompt tokens=\(prompt.count) prefillStep=\(prefillStep)")

        let cache = model.newCache(parameters: nil)
        let array = MLXArray(prompt.map { Int32($0) }).expandedDimensions(axis: 0)
        let prepared = try model.prepare(LMInput(text: .init(tokens: array)), cache: cache, windowSize: prefillStep)
        var logits: MLXArray
        switch prepared {
        case .tokens(let tail):
            let t = tail.tokens.ndim == 1 ? tail.tokens[.newAxis] : tail.tokens
            logits = model(t, cache: cache)[0, -1].asType(.float32)
        case .logits(let r):
            logits = r.logits[0, -1].asType(.float32)
        }
        MLX.eval(logits)
        let values = logits.asArray(Float.self)
        let nonFinite = values.filter { !$0.isFinite }.count
        let top = argSort(logits, axis: -1)[(-5)...].asArray(Int32.self).reversed()
        print("PROBE prefill logits: nonFinite=\(nonFinite)/\(values.count) top5=\(Array(top)) vals=\(top.map { values[Int($0)] })")

        // Greedy decode from the prefilled cache.
        var out: [Int] = []
        var next = Int(argMax(logits).item(Int32.self))
        for _ in 0..<maxTok {
            out.append(next)
            if let eos = tokenizer.eosTokenId, next == eos { break }
            let step = MLXArray([Int32(next)]).expandedDimensions(axis: 0)
            let l = model(step, cache: cache)[0, -1].asType(.float32)
            MLX.eval(l)
            next = Int(argMax(l).item(Int32.self))
        }
        let text = tokenizer.decode(tokenIds: out, skipSpecialTokens: false)
        let bangs = out.filter { $0 == 0 }.count
        print("PROBE greedy tokens=\(out) token0Count=\(bangs)/\(out.count) text=\(text.replacingOccurrences(of: "\n", with: "\\n"))")
        print("PROBE VERDICT greedy nonFiniteLogits=\(nonFinite) token0Fraction=\(Double(bangs) / Double(max(out.count, 1)))")

        // The app's shape: TokenIterator with the bundle sampler defaults
        // (temperature/top_p/penalty as osaurus sends them), windowed prefill,
        // fresh cache; optional compiled decode (VMLX_LING26_COMPILED=1).
        var params = GenerateParameters(
            maxTokens: maxTok, temperature: 0.7, topP: 0.95, topK: 20, minP: 0,
            randomSeed: 11, repetitionPenalty: 1.05, repetitionContextSize: 64,
            prefillStepSize: prefillStep)
        if env["VMLX_LING26_COMPILED"] == "1" { params.enableCompiledDecode = true }
        var it = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map { Int32($0) }).expandedDimensions(axis: 0), tokenIds: prompt),
            model: model, parameters: params)
        var sampled: [Int] = []
        while sampled.count < maxTok, let t = it.next() {
            sampled.append(t)
            if let eos = tokenizer.eosTokenId, t == eos { break }
        }
        let sampledText = tokenizer.decode(tokenIds: sampled, skipSpecialTokens: false)
        let sampledBangs = sampled.filter { $0 == 0 }.count
        print("PROBE sampled tokens=\(sampled.count) token0Count=\(sampledBangs) compiled=\(params.enableCompiledDecode) text=\(sampledText.replacingOccurrences(of: "\n", with: "\\n"))")
        print("PROBE VERDICT sampled token0Fraction=\(Double(sampledBangs) / Double(max(sampled.count, 1)))")

        // ---------- BATCH ENGINE (the app's generation path: BatchEngine solo/batch slots) ----------
        if let bs = Int(env["VMLX_LING26_BATCH"] ?? "") {
            // App-shaped options: compiled BATCH decode (the native-MTP fallback the app
            // enables), a disk-backed CacheCoordinator, and the request's prefix boundaries.
            let cbd = env["VMLX_LING26_CBD"] == "1"
            var coordinator: CacheCoordinator? = nil
            if env["VMLX_LING26_COORD"] == "1" {
                let diskDir = FileManager.default.temporaryDirectory.appendingPathComponent("ling26-batch-\(UUID().uuidString)")
                let c = CacheCoordinator(config: CacheCoordinatorConfig(
                    usePagedCache: false, enableDiskCache: true, diskCacheDir: diskDir, modelKey: "ling26-batch"))
                let topology = ModelCacheTopologySnapshot(cache: model.newCache(parameters: nil))
                c.setHybrid(topology.requiresSSMCompanionState,
                    requiresRecurrentSSMCompanion: topology.requiresRecurrentSSMCompanionState,
                    requiresSeparateRecurrentPayload: topology.requiresSeparateRecurrentPayloadState)
                coordinator = c
            }
            let boundaries = (env["VMLX_LING26_BOUNDARIES"] ?? "").split(separator: ",").compactMap { Int($0) }
            let stable = (env["VMLX_LING26_STABLE"] ?? "").split(separator: ",").compactMap { Int($0) }
            let engine = coordinator.map { BatchEngine(context: context, maxBatchSize: bs, cacheCoordinator: $0) }
                ?? BatchEngine(context: context, maxBatchSize: bs)
            var bparams = GenerateParameters(maxTokens: maxTok, enableCompiledBatchDecode: cbd, temperature: 0)
            bparams.prefillStepSize = prefillStep
            print("PROBE batch shape cbd=\(cbd) coordinator=\(coordinator != nil) boundaries=\(boundaries) stable=\(stable)")
            if env["VMLX_LING26_COMPILED"] == "1" { bparams.enableCompiledDecode = true }
            if let kv = Int(env["VMLX_LING26_KVBITS"] ?? "") { bparams.kvBits = kv }
            print("PROBE batch params compiled=\(bparams.enableCompiledDecode) kvBits=\(String(describing: bparams.kvBits)) kvGroupSize=\(bparams.kvGroupSize) quantizedKVStart=\(bparams.quantizedKVStart)")
            let stream = await engine.generate(
                input: LMInput(tokens: MLXArray(prompt.map { Int32($0) }).expandedDimensions(axis: 0), tokenIds: prompt,
                    cachePrefixTokenCounts: boundaries, cacheStablePrefixTokenCounts: stable),
                parameters: bparams)
            var btext = ""
            var stop = "?"
            for await ev in stream {
                switch ev {
                case .chunk(let s): btext += s
                case .info(let info): stop = String(describing: info.stopReason)
                default: break
                }
            }
            await engine.shutdown()
            let bangs = btext.filter { $0 == "!" }.count
            print("PROBE batch(maxBatchSize=\(bs)) stop=\(stop) chars=\(btext.count) bangChars=\(bangs) text=\(btext.prefix(160).replacingOccurrences(of: "\n", with: "\\n"))")
            print("PROBE VERDICT batch bangFraction=\(Double(bangs) / Double(max(btext.count, 1)))")
        }

        // ---------- RESTORE (the app's shape) ----------
        // osaurus warms the system prompt up, stores that prefix through the
        // cache coordinator (disk tier), and the request restores it and
        // re-feeds the user turn. The boundary is a system-prompt render, not
        // a multiple of the prefill step. Compare the last-position logits with
        // the fresh one-shot forward above and decode greedily from the
        // restored cache.
        if env["VMLX_LING26_RESTORE"] == "1" {
            let systemOnly = try tokenizer.applyChatTemplate(messages: [messages[0]])
            var boundary = 0
            while boundary < min(systemOnly.count, prompt.count), systemOnly[boundary] == prompt[boundary] {
                boundary += 1
            }
            print("PROBE restore boundary=\(boundary) (system-only render \(systemOnly.count) tokens, full prompt \(prompt.count))")
            let diskDir = FileManager.default.temporaryDirectory
                .appendingPathComponent("ling26-restore-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: diskDir) }
            let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
                usePagedCache: false, enableDiskCache: true, diskCacheDir: diskDir,
                modelKey: "ling26-restore"))
            let topology = ModelCacheTopologySnapshot(cache: model.newCache(parameters: nil))
            coordinator.setHybrid(
                topology.requiresSSMCompanionState,
                requiresRecurrentSSMCompanion: topology.requiresRecurrentSSMCompanionState,
                requiresSeparateRecurrentPayload: topology.requiresSeparateRecurrentPayloadState)
            func makeInput(_ tokens: [Int], prefix: [Int]) -> LMInput {
                LMInput(
                    tokens: MLXArray(tokens.map { Int32($0) }).expandedDimensions(axis: 0),
                    tokenIds: tokens,
                    cachePrefixTokenCounts: prefix,
                    cacheStablePrefixTokenCounts: prefix)
            }
            var seedParams = GenerateParameters(maxTokens: 1, temperature: 0)
            seedParams.prefillStepSize = prefillStep
            let seedPrompt = Array(prompt.prefix(boundary + 8))
            var seedIt = try TokenIterator(
                input: makeInput(seedPrompt, prefix: [boundary]), model: model,
                parameters: seedParams, cacheCoordinator: coordinator)
            _ = seedIt.next()
            seedIt.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
            let salt = computeCacheSalt(for: makeInput(prompt, prefix: [boundary]), parameters: seedParams)
            let fetched = coordinator.fetch(
                tokens: prompt, mediaSalt: salt, skipExactDiskBoundary: true,
                preferredDiskBoundaries: [boundary])
            guard case .hit(let matched, let remaining, let detail, _, _, let arrays) = fetched, let record = arrays else {
                print("PROBE restore: fetch MISSED (\(fetched)) — restore path not exercised")
                return
            }
            print("PROBE restore hit matched=\(matched) remaining=\(remaining.count) detail=\(detail)")
            var restoredCache = model.newCache(parameters: nil)
            let restoredTokens = restoreFromDiskArrays(record, into: &restoredCache)
            let offsets = restoredCache.map { $0.offset }
            let perLayer = restoredCache.enumerated().map { "\($0.offset):\(String(describing: type(of: $0.element)).prefix(12)):\($0.element.offset)" }
            print("PROBE restored tokens=\(restoredTokens) layer offsets distinct=\(Set(offsets).sorted()) perLayer=\(perLayer.prefix(10).joined(separator: " ")) …")
            let remArray = MLXArray(remaining.map { Int32($0) }).expandedDimensions(axis: 0)
            let preparedR = try model.prepare(LMInput(text: .init(tokens: remArray)), cache: restoredCache, windowSize: prefillStep)
            var rl: MLXArray
            switch preparedR {
            case .tokens(let tail):
                let t = tail.tokens.ndim == 1 ? tail.tokens[.newAxis] : tail.tokens
                rl = model(t, cache: restoredCache)[0, -1].asType(.float32)
            case .logits(let r):
                rl = r.logits[0, -1].asType(.float32)
            }
            MLX.eval(rl)
            let rv = rl.asArray(Float.self)
            let rNonFinite = rv.filter { !$0.isFinite }.count
            let maxDelta = zip(rv, values).map { abs($0 - $1) }.max() ?? .nan
            let rTop = argSort(rl, axis: -1)[(-5)...].asArray(Int32.self).reversed()
            print("PROBE restore logits: nonFinite=\(rNonFinite)/\(rv.count) top5=\(Array(rTop)) maxAbsDeltaVsFresh=\(maxDelta) freshTop1=\(top.first ?? -1)")
            var rout: [Int] = []
            var rnext = Int(argMax(rl).item(Int32.self))
            for _ in 0..<maxTok {
                rout.append(rnext)
                if let eos = tokenizer.eosTokenId, rnext == eos { break }
                let l = model(MLXArray([Int32(rnext)]).expandedDimensions(axis: 0), cache: restoredCache)[0, -1].asType(.float32)
                MLX.eval(l)
                rnext = Int(argMax(l).item(Int32.self))
            }
            let rtext = tokenizer.decode(tokenIds: rout, skipSpecialTokens: false)
            let rBangs = rout.filter { $0 == 0 }.count
            print("PROBE restore greedy token0Count=\(rBangs)/\(rout.count) text=\(rtext.replacingOccurrences(of: "\n", with: "\\n"))")
            print("PROBE VERDICT restore nonFinite=\(rNonFinite) token0Fraction=\(Double(rBangs) / Double(max(rout.count, 1))) maxAbsDeltaVsFresh=\(maxDelta)")

            // The app path proper: a second request through TokenIterator with
            // the coordinator (fetch + validated restore + re-feed inside the
            // iterator), greedy. Any '[vmlx][cache/restore] REFUSED' line in
            // the log means the engine fell back to a full prefill.
            var reqParams = GenerateParameters(maxTokens: maxTok, temperature: 0)
            reqParams.prefillStepSize = prefillStep
            var reqIt = try TokenIterator(
                input: makeInput(prompt, prefix: [boundary]), model: model,
                parameters: reqParams, cacheCoordinator: coordinator)
            var iout: [Int] = []
            while iout.count < maxTok, let t = reqIt.next() {
                iout.append(t)
                if let eos = tokenizer.eosTokenId, t == eos { break }
            }
            let itext = tokenizer.decode(tokenIds: iout, skipSpecialTokens: false)
            let iBangs = iout.filter { $0 == 0 }.count
            print("PROBE iterator-restore greedy token0Count=\(iBangs)/\(iout.count) text=\(itext.replacingOccurrences(of: "\n", with: "\\n"))")
            print("PROBE VERDICT iterator-restore token0Fraction=\(Double(iBangs) / Double(max(iout.count, 1)))")
        }
    }
}
