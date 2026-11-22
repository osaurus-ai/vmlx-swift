// BENCH_INKLING2_PARITY=1 — Inkling-2-Mini (inkling2_loop) port check against the jang_tools MLX port
// (parity-tested vs transformers PR #51050). Development iteration only, not app proof.
//
// Reads BENCH_INKLING2_REF (py_reference.py output JSON + *_last.npy written as raw .f32 next to it) and reports
// per prompt: chunked-prefill last-row logits vs Python (max |d|, KL, top-1, top-10 overlap), no-cache vs cached
// consistency, greedy continuation agreement, and decode timing.

import CmlxGraphShim
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import VMLXTokenizers

func runInkling2ParityProbe(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    if let visionRef = env["BENCH_INKLING2_VISION_REF"] {
        try await runInkling2VisionParity(modelPath: modelPath, refPath: visionRef)
        return
    }
    guard let refPath = env["BENCH_INKLING2_REF"] else {
        print("[INK2] BENCH_INKLING2_REF is required")
        exit(2)
    }
    let chunk = Int(env["BENCH_INKLING2_CHUNK"] ?? "333") ?? 333
    let decodeN = Int(env["BENCH_INKLING2_DECODE"] ?? "128") ?? 128
    let ref =
        try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: refPath)))
        as! [String: [String: Any]]

    let t0 = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: URL(fileURLWithPath: modelPath), using: #huggingFaceTokenizerLoader(),
        loadConfiguration: .default)
    let model = context.model
    print(
        String(
            format: "[INK2] load=%.1fs type=%@ active=%.2fGB", CFAbsoluteTimeGetCurrent() - t0,
            String(describing: type(of: model)), Double(MLX.Memory.activeMemory) / 1e9))

    func kl(_ r: MLXArray, _ c: MLXArray) -> Float {
        let lr = r - logSumExp(r)
        let lc = c - logSumExp(c)
        return (exp(lr) * (lr - lc)).sum().item(Float.self)
    }
    for name in ["short", "long"] {
        guard let r = ref[name], let ids0 = r["ids"] as? [Int],
            let pyGreedy = r["greedy"] as? [Int],
            let pyTop = r["top10"] as? [Int]
        else { continue }
        let n = ids0.count
        let ids = MLXArray(ids0.map { Int32($0) })[.newAxis]
        let npy = refPath.replacingOccurrences(of: ".json", with: "_\(name)_last.f32")
        let pyLast: MLXArray? = (try? Data(contentsOf: URL(fileURLWithPath: npy))).map { d in
            d.withUnsafeBytes { MLXArray(Array($0.bindMemory(to: Float.self))) }
        }
        // cached chunked prefill
        var cache = model.newCache(parameters: nil)
        var s = 0
        var last = MLXArray(0)
        let tp = CFAbsoluteTimeGetCurrent()
        while s < n {
            let e = min(s + chunk, n)
            last = model(ids[0..., s ..< e], cache: cache)[0, -1].asType(.float32)
            eval(last)
            s = e
        }
        let prefillS = CFAbsoluteTimeGetCurrent() - tp
        // no-cache whole-sequence forward (explicit window masks): must agree with the cached one
        let nc = model(ids, cache: nil)[0, -1].asType(.float32)
        eval(nc)
        let swTop = Array(argSort(-last)[0 ..< 10].asArray(Int32.self).map(Int.init))
        var line =
            "[INK2] \(name) n=\(n) prefill=\(String(format: "%.2f", prefillS))s top1 swift=\(swTop[0]) py=\(pyTop[0])"
        line += " top10_overlap=\(Set(swTop).intersection(pyTop).count)"
        line += String(
            format: " nocache_vs_cached maxabs=%.4f kl=%.2e", abs(nc - last).max().item(Float.self),
            kl(nc, last))
        if let pyLast {
            line += String(
                format: " vs_py maxabs=%.4f kl=%.2e", abs(pyLast - last).max().item(Float.self),
                kl(pyLast, last))
        }
        print(line)
        // greedy continuation from the cached state
        var gen = [last.argMax().item(Int.self)]
        let td = CFAbsoluteTimeGetCurrent()
        var y = MLXArray([Int32(gen[0])])[.newAxis]
        for _ in 1 ..< max(decodeN, pyGreedy.count) {
            let lg = model(y, cache: cache)
            let t = lg[0, -1].argMax()
            asyncEval(t)
            let v = t.item(Int.self)
            gen.append(v)
            y = MLXArray([Int32(v)])[.newAxis]
        }
        let decodeS = CFAbsoluteTimeGetCurrent() - td
        var agree = 0
        for (a, b) in zip(gen, pyGreedy) { if a == b { agree += 1 } else { break } }
        let text = context.tokenizer.decode(
            tokenIds: Array(gen.prefix(pyGreedy.count)), skipSpecialTokens: false)
        // Teacher-forced vs Python: feed Python's greedy tokens, compare each step's logits.
        let stepsPath = refPath.replacingOccurrences(of: ".json", with: "_\(name)_steps.f32")
        if let d = try? Data(contentsOf: URL(fileURLWithPath: stepsPath)) {
            let all = d.withUnsafeBytes { MLXArray(Array($0.bindMemory(to: Float.self))) }
            let V = all.size / (pyGreedy.count - 1)
            let pySteps = all.reshaped([pyGreedy.count - 1, V])
            var c = model.newCache(parameters: nil)
            var s3 = 0
            while s3 < n {
                let e = min(s3 + chunk, n)
                eval(model(ids[0..., s3 ..< e], cache: c))
                s3 = e
            }
            var kls: [Float] = []
            var top1 = 0
            for t in 0 ..< (pyGreedy.count - 1) {
                let lg = model(MLXArray([Int32(pyGreedy[t])])[.newAxis], cache: c)[0, -1].asType(
                    .float32)
                eval(lg)
                kls.append(kl(pySteps[t], lg))
                if lg.argMax().item(Int.self) == pySteps[t].argMax().item(Int.self) { top1 += 1 }
            }
            c = []
            let srt = kls.sorted()
            print(
                String(
                    format:
                        "[INK2] %@ teacher-forced vs py: %d steps top1 %d/%d KL median %.2e max %.2e",
                    name, kls.count, top1, kls.count, srt[srt.count / 2], srt.last!))
        }
        // Self-consistency control: the same tokens decoded after two different prefill chunkings.
        if name == "long" {
            func tf(_ ck: Int) -> [MLXArray] {
                let c = model.newCache(parameters: nil)
                var s4 = 0
                while s4 < n {
                    let e = min(s4 + ck, n)
                    eval(model(ids[0..., s4 ..< e], cache: c))
                    s4 = e
                }
                var out: [MLXArray] = []
                for t in 0 ..< (pyGreedy.count - 1) {
                    let lg = model(MLXArray([Int32(pyGreedy[t])])[.newAxis], cache: c)[0, -1]
                        .asType(.float32)
                    eval(lg)
                    out.append(lg)
                }
                return out
            }
            let a1 = tf(512)
            let a2 = tf(333)
            let kls = zip(a1, a2).map { kl($0.0, $0.1) }.sorted()
            print(
                String(
                    format: "[INK2] control swift(512) vs swift(333): KL median %.2e max %.2e",
                    kls[kls.count / 2], kls.last!))
        }
        // Teacher-forced fast-decode vs general-path decode from identical cache states.
        if name == "long" {
            let gold = pyGreedy
            var cA = model.newCache(parameters: nil)
            var s2 = 0
            while s2 < n {
                let e = min(s2 + chunk, n)
                eval(model(ids[0..., s2 ..< e], cache: cA))
                s2 = e
            }
            let cB = cA.map { $0.copy() }
            var worstKL: Float = 0
            var mism = 0
            for t in 0 ..< min(24, gold.count) {
                let y = MLXArray([Int32(gold[t])])[.newAxis]
                Inkling2Runtime.forceGeneralDecode = false
                let la = model(y, cache: cA)[0, -1].asType(.float32)
                Inkling2Runtime.forceGeneralDecode = true
                let lb = model(y, cache: cB)[0, -1].asType(.float32)
                Inkling2Runtime.forceGeneralDecode = false
                eval(la, lb)
                worstKL = max(worstKL, kl(lb, la))
                if la.argMax().item(Int.self) != lb.argMax().item(Int.self) { mism += 1 }
            }
            _ = cA
            print(
                String(
                    format:
                        "[INK2] fast-vs-general teacher-forced 24 steps: max KL %.3e, argmax mismatches %d",
                    worstKL, mism))
        }
        print(
            "[INK2] \(name) greedy prefix agree=\(agree)/\(pyGreedy.count) decode=\(String(format: "%.1f", Double(gen.count - 1) / decodeS)) tok/s (naive loop, \(gen.count - 1) steps)"
        )
        print(
            "[INK2] \(name) text: \(text.replacingOccurrences(of: "\n", with: "\\n").prefix(300))")
    }
}

/// BENCH_INKLING2_TIMING=1: decode-step anatomy on a short prompt. Reports graph node count per
/// step, CPU graph-build time, synchronous eval time, and pipelined decode tok/s (median of 3 runs).
func runInkling2TimingProbe(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let steps = Int(env["BENCH_INKLING2_STEPS"] ?? "128") ?? 128
    let ctxTokens = Int(env["BENCH_INKLING2_CTX"] ?? "64") ?? 64
    let context = try await MLXLMCommon.loadModel(
        from: URL(fileURLWithPath: modelPath), using: #huggingFaceTokenizerLoader(),
        loadConfiguration: .default)
    let model = context.model
    let prompt = MLXArray((0 ..< ctxTokens).map { Int32(1000 + ($0 * 37) % 5000) })[.newAxis]
    func fresh() -> [KVCache] {
        let c = model.newCache(parameters: nil)
        eval(model(prompt, cache: c)[0, -1].argMax())
        return c
    }
    // anatomy
    var cache = fresh()
    var y = MLXArray([Int32(1234)])[.newAxis]
    var builds: [Double] = []
    var evals: [Double] = []
    var nodes: Int32 = -1
    for i in 0 ..< 24 {
        let t0 = CFAbsoluteTimeGetCurrent()
        let tok = model(y, cache: cache)[0, -1].argMax()
        let t1 = CFAbsoluteTimeGetCurrent()
        if i == 0 {
            var asType: Int32 = -1
            let ptr = unsafeBitCast(tok.ctx, to: UnsafeMutableRawPointer?.self)
            _ = vmlx_graph_stats(ptr, &nodes, &asType)
        }
        eval(tok, cache)
        let t2 = CFAbsoluteTimeGetCurrent()
        if i >= 4 {
            builds.append((t1 - t0) * 1000)
            evals.append((t2 - t1) * 1000)
        }
        y = tok.reshaped([1, 1]).asType(.int32)
    }
    func med(_ a: [Double]) -> Double { a.sorted()[a.count / 2] }
    print(
        String(
            format: "[INK2T] nodes/step=%d build=%.2fms eval(sync)=%.2fms", nodes, med(builds),
            med(evals)))
    // pipelined decode
    var rates: [Double] = []
    for _ in 0 ..< 3 {
        cache = fresh()
        var prev = model(MLXArray([Int32(1234)])[.newAxis], cache: cache)[0, -1].argMax()
        asyncEval(prev)
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0 ..< steps {
            let next = model(prev.reshaped([1, 1]), cache: cache)[0, -1].argMax()
            asyncEval(next)
            _ = prev.item(Int.self)
            prev = next
        }
        _ = prev.item(Int.self)
        rates.append(Double(steps) / (CFAbsoluteTimeGetCurrent() - t0))
    }
    print(
        String(
            format: "[INK2T] pipelined decode tok/s runs=%@ median=%.1f",
            rates.map { String(format: "%.1f", $0) }.joined(separator: ","), med(rates)))
}

/// BENCH_INKLING2_TEMPLATE=1: the app's prompt path (UserInput(chat:tools:additionalContext:) ->
/// processor -> chat template -> tokenizer) against transformers' apply_chat_template renders
/// (BENCH_INKLING2_RENDERS json from make_renders.py). Token ids must match exactly.
func runInkling2TemplateProbe(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let refs =
        try JSONSerialization.jsonObject(
            with: Data(contentsOf: URL(fileURLWithPath: env["BENCH_INKLING2_RENDERS"]!)))
        as! [String: [String: Any]]
    let context = try await MLXLMCommon.loadModel(
        from: URL(fileURLWithPath: modelPath), using: #huggingFaceTokenizerLoader(),
        loadConfiguration: .default)
    let weather: ToolSpec = [
        "type": "function",
        "function": [
            "name": "get_weather", "description": "Current weather for a city",
            "parameters": [
                "type": "object",
                "properties": [
                    "city": ["type": "string", "description": "City name"]
                        as [String: any Sendable],
                    "unit": ["type": "string", "enum": ["c", "f"]] as [String: any Sendable],
                ] as [String: any Sendable],
                "required": ["city"],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]
    let shell: ToolSpec = [
        "type": "function",
        "function": [
            "name": "run_shell", "description": "Run a shell command / script",
            "parameters": [
                "type": "object",
                "properties": [
                    "cmd": ["type": "string"] as [String: any Sendable]
                ] as [String: any Sendable],
                "required": ["cmd"],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]
    let call = ToolCall(
        function: .init(
            name: "get_weather",
            arguments: ["unit": "c", "city": "Paris"] as [String: any Sendable]), id: "call_1")
    var assistantTool = Chat.Message.assistant("", toolCalls: [call])
    assistantTool.reasoningContent = "Need tool."
    var assistantReason = Chat.Message.assistant("4")
    assistantReason.reasoningContent = "Simple sum."
    let cases: [String: ([Chat.Message], [ToolSpec]?, [String: any Sendable]?)] = [
        "plain": ([.user("Hello there")], nil, nil),
        "system": ([.system("You are terse."), .user("Hi")], nil, nil),
        "effort_low": ([.user("Hi")], nil, ["reasoning_effort": "low"]),
        "effort_none": ([.user("Hi")], nil, ["reasoning_effort": "none"]),
        "multiturn_reasoning": ([.user("2+2?"), assistantReason, .user("and 3+3?")], nil, nil),
        "tools": ([.user("Weather in Paris? Use unit c.")], [weather, shell], nil),
        "tool_roundtrip": (
            [
                .user("Weather in Paris?"), assistantTool,
                .tool("{\"temp\": 18, \"sky\": \"clear\"}", toolCallId: "call_1"),
                .user("Thanks. And unicode: café ☕ / path a/b"),
            ], [weather, shell], nil
        ),
    ]
    var fails = 0
    for (name, (msgs, tools, extra)) in cases.sorted(by: { $0.key < $1.key }) {
        guard let ref = refs[name], let refIds = ref["ids"] as? [Int] else { continue }
        do {
            let input = try await context.processor.prepare(
                input: UserInput(chat: msgs, tools: tools, additionalContext: extra))
            let ids = input.text.tokens.asArray(Int32.self).map(Int.init)
            if ids == refIds {
                print("[INK2TPL] \(name): MATCH (\(ids.count) tokens)")
            } else {
                fails += 1
                let k =
                    (0 ..< min(ids.count, refIds.count)).first(where: { ids[$0] != refIds[$0] })
                    ?? min(ids.count, refIds.count)
                let swiftText = context.tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
                print(
                    "[INK2TPL] \(name): MISMATCH swift=\(ids.count) py=\(refIds.count) first diff at \(k)"
                )
                print("  swift: \(swiftText.replacingOccurrences(of: "\n", with: "⏎"))")
                print(
                    "  py   : \((ref["text"] as? String ?? "").replacingOccurrences(of: "\n", with: "⏎"))"
                )
            }
        } catch {
            fails += 1
            print("[INK2TPL] \(name): ERROR \(error)")
        }
    }
    print("[INK2TPL] \(fails == 0 ? "ALL MATCH" : "\(fails) FAILED")")
}

/// BENCH_INKLING2_DSPARK_ITER=1: the production TokenIterator route with `draftStrategy = .dspark(bundle)`
/// (the generic dSpark runtime + Inkling draft session) vs AR, greedy and sampled. Text equality at T=0,
/// decode tok/s from the completion info.
func runInkling2DSparkIteratorProbe(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let ref =
        (env["BENCH_INKLING2_REF"].flatMap {
            try? JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: $0)))
        } as? [String: [String: Any]]) ?? [:]
    let maxNew = Int(env["BENCH_INKLING2_NEW"] ?? "256") ?? 256
    if let gb = env["BENCH_CACHE_LIMIT_GB"].flatMap(Double.init) {
        MLX.Memory.cacheLimit = Int(gb * 1_073_741_824)
        print("[INK2I] MLX cache limit \(gb) GiB")
    }
    let bundle = URL(fileURLWithPath: modelPath)
    let context = try await MLXLMCommon.loadModel(
        from: bundle, using: #huggingFaceTokenizerLoader(), loadConfiguration: .default)
    // BENCH_INKLING2_CHAT="<user message>": the message rendered through the app's chat path
    // replaces the reference prompts.
    var prompts: [(String, LMInput)] = []
    if let message = env["BENCH_INKLING2_CHAT"] {
        let input = try await context.processor.prepare(input: UserInput(chat: [.user(message)]))
        prompts.append(("chat", input))
    } else {
        for name in ["long", "short"] {
            guard let ids = ref[name]?["ids"] as? [Int] else { continue }
            prompts.append((name, LMInput(tokens: MLXArray(ids.map { Int32($0) }))))
        }
    }
    for (name, input) in prompts {
        for temp: Float in [0, 1] {
            var texts: [String] = []
            for useDSpark in [false, true] {
                var p = GenerateParameters(maxTokens: maxNew, temperature: temp)
                p.randomSeed = 7
                if useDSpark { p.draftStrategy = .dspark(drafterPath: bundle) }
                var text = ""
                var info: GenerateCompletionInfo?
                for await g in try MLXLMCommon.generate(
                    input: input, parameters: p, context: context)
                {
                    switch g {
                    case .chunk(let c): text += c
                    case .reasoning(let r): text += r
                    case .info(let i): info = i
                    default: break
                    }
                }
                texts.append(text)
                let tps =
                    info.map {
                        Double(max(0, $0.generationTokenCount - 1)) / max($0.generateTime, 1e-9)
                    } ?? 0
                print(
                    String(
                        format: "[INK2I] %@ T=%.0f %@ gen=%d decode tok/s=%.1f", name, temp,
                        useDSpark ? "dspark" : "ar    ", info?.generationTokenCount ?? -1, tps))
            }
            if temp == 0 {
                let a = texts[0]
                let b = texts[1]
                let common = zip(a, b).prefix { $0 == $1 }.count
                print(
                    "[INK2I] \(name) T=0 text equal=\(a == b) common prefix chars=\(common)/\(a.count)"
                )
            }
        }
    }
}

/// BENCH_INKLING2_VT=1: verify-shaped forward timing (T = 1 / 4 / 8 rows, median of 6) after
/// prefilling the reference long prompt (`VT_N` caps its length, to measure context dependence).
func runInkling2VerifyTimeProbe(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let ref =
        try JSONSerialization.jsonObject(
            with: Data(contentsOf: URL(fileURLWithPath: env["BENCH_INKLING2_REF"]!)))
        as! [String: [String: Any]]
    let ids0 = Array((ref["long"]!["ids"] as! [Int]).prefix(Int(env["VT_N"] ?? "100000")!))
    let context = try await MLXLMCommon.loadModel(
        from: URL(fileURLWithPath: modelPath), using: #huggingFaceTokenizerLoader(),
        loadConfiguration: .default)
    let model = context.model
    let ids = MLXArray(ids0.map { Int32($0) })[.newAxis]
    let cache = model.newCache(parameters: nil)
    var s = 0
    while s < ids0.count {
        let e = min(s + 512, ids0.count)
        eval(model(ids[0..., s ..< e], cache: cache))
        s = e
    }
    for T in [1, 4, 8, 1, 4, 8] {
        var ts: [Double] = []
        for r in 0 ..< 7 {
            let x = ids[0..., (100 + r * T) ..< (100 + r * T + T)]
            let t0 = CFAbsoluteTimeGetCurrent()
            eval(model(x, cache: cache))
            ts.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        let m = ts.dropFirst().sorted()[3]
        print(String(format: "[VT] T=%d S=%d median=%.2f ms", T, cache[0].offset, m))
    }
}

/// BENCH_INKLING2_CTXSWEEP="64,1024,2048,4096": AR decode tok/s (greedy, TokenIterator route) after a
/// prompt of each length (the reference long prompt tiled), to separate context-dependent decode cost
/// from fixed per-token cost. BENCH_INKLING2_NEW decode tokens per row.
func runInkling2ContextSweepProbe(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let ref =
        try JSONSerialization.jsonObject(
            with: Data(contentsOf: URL(fileURLWithPath: env["BENCH_INKLING2_REF"]!)))
        as! [String: [String: Any]]
    let base = ref["long"]!["ids"] as! [Int]
    let lengths = (env["BENCH_INKLING2_CTXSWEEP"] ?? "64,2048").split(separator: ",").compactMap {
        Int($0)
    }
    let maxNew = Int(env["BENCH_INKLING2_NEW"] ?? "160") ?? 160
    let context = try await MLXLMCommon.loadModel(
        from: URL(fileURLWithPath: modelPath), using: #huggingFaceTokenizerLoader(),
        loadConfiguration: .default)
    for (i, n) in (lengths + lengths).enumerated() {
        var ids: [Int] = []
        while ids.count < n { ids += base }
        ids = Array(ids.prefix(n))
        var p = GenerateParameters(maxTokens: maxNew, temperature: 0)
        if env["BENCH_INKLING2_SWEEP_DSPARK"] == "1" {
            p.draftStrategy = .dspark(drafterPath: URL(fileURLWithPath: modelPath))
        }
        var info: GenerateCompletionInfo?
        for await g in try MLXLMCommon.generate(
            input: LMInput(tokens: MLXArray(ids.map { Int32($0) })), parameters: p, context: context
        ) {
            if case .info(let x) = g { info = x }
        }
        guard let info else { continue }
        let tps = Double(max(0, info.generationTokenCount - 1)) / max(info.generateTime, 1e-9)
        print(
            String(
                format:
                    "[CTX] pass=%d n=%d gen=%d decode tok/s=%.1f ms/tok=%.2f prefill tok/s=%.0f",
                i < lengths.count ? 1 : 2, n, info.generationTokenCount, tps, 1000 / tps,
                info.promptTokensPerSecond))
    }
}

/// BENCH_INKLING2_CPUPROF=1: per decode step after a `VT_N`-token prompt (default 2048), the wall and
/// calling-thread CPU time of building the graph and of evaluating it (synchronous), median of 64
/// steps. eval CPU close to eval wall = the step is bound by encoding, not by the GPU.
func runInkling2CPUProfileProbe(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let ref =
        try JSONSerialization.jsonObject(
            with: Data(contentsOf: URL(fileURLWithPath: env["BENCH_INKLING2_REF"]!)))
        as! [String: [String: Any]]
    let base = ref["long"]!["ids"] as! [Int]
    let n = Int(env["VT_N"] ?? "2048") ?? 2048
    var ids0: [Int] = []
    while ids0.count < n { ids0 += base }
    ids0 = Array(ids0.prefix(n))
    let context = try await MLXLMCommon.loadModel(
        from: URL(fileURLWithPath: modelPath), using: #huggingFaceTokenizerLoader(),
        loadConfiguration: .default)
    let model = context.model
    let ids = MLXArray(ids0.map { Int32($0) })[.newAxis]
    let cache = model.newCache(parameters: nil)
    var s = 0
    while s < n {
        let e = min(s + 512, n)
        eval(model(ids[0..., s ..< e], cache: cache))
        s = e
    }
    func cpu() -> Double {
        var ts = timespec()
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts)
        return Double(ts.tv_sec) * 1000 + Double(ts.tv_nsec) / 1e6
    }
    func now() -> Double { CFAbsoluteTimeGetCurrent() * 1000 }
    var tok = MLXArray([Int32(ids0.last!)]).reshaped([1, 1])
    var bw: [Double] = []
    var bc: [Double] = []
    var ew: [Double] = []
    var ec: [Double] = []
    var pipe: [Double] = []
    for step in 0 ..< 80 {
        let w0 = now()
        let c0 = cpu()
        let logits = model(tok, cache: cache)
        let next = argMax(logits[0..., -1, 0...], axis: -1).reshaped([1, 1])
        let w1 = now()
        let c1 = cpu()
        eval(next)
        let w2 = now()
        let c2 = cpu()
        tok = next
        if step >= 16 {
            bw.append(w1 - w0)
            bc.append(c1 - c0)
            ew.append(w2 - w1)
            ec.append(c2 - c1)
        }
    }
    // pipelined (as TokenIterator): build + asyncEval step t+1 before waiting on step t
    var prev = tok
    var t0 = now()
    for step in 0 ..< 80 {
        let logits = model(prev, cache: cache)
        let next = argMax(logits[0..., -1, 0...], axis: -1).reshaped([1, 1])
        asyncEval(next)
        _ = prev.item(Int32.self)
        prev = next
        let t = now()
        if step >= 16 { pipe.append(t - t0) }
        t0 = t
    }
    func med(_ a: [Double]) -> Double { a.sorted()[a.count / 2] }
    print(
        String(
            format:
                "[CPUPROF] n=%d build wall=%.2f cpu=%.2f | eval wall=%.2f cpu=%.2f | sync total=%.2f ms | pipelined=%.2f ms/tok",
            n, med(bw), med(bc), med(ew), med(ec), med(bw) + med(ew), med(pipe)))
}

/// BENCH_INKLING2_APPPATH=1: the app's engine route — `ModelContainer` with the server cache
/// configuration (memory + disk), `makeBatchEngine(maxBatchSize: 1)`, a ~2k-token system prompt —
/// sending the same chat `BENCH_INKLING2_RUNS` (3) times: run 1 prefills cold, later runs restore
/// the prefix from the cache. Decode tok/s per run, AR or dSpark (`BENCH_INKLING2_SWEEP_DSPARK=1`).
func runInkling2AppPathProbe(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let modelDir = URL(fileURLWithPath: modelPath)
    let runs = Int(env["BENCH_INKLING2_RUNS"] ?? "3") ?? 3
    let maxNew = Int(env["BENCH_INKLING2_NEW"] ?? "384") ?? 384
    let diskDir = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("ink2-apppath-\(ProcessInfo.processInfo.processIdentifier)")
    try? FileManager.default.removeItem(at: diskDir)
    try FileManager.default.createDirectory(at: diskDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: diskDir) }
    let settings = VMLXServerRuntimeSettings()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader(), loadConfiguration: .osaurusProduction)
    let container = ModelContainer(context: context)
    let cacheOn = env["BENCH_INKLING2_NOCACHE"] != "1"
    if cacheOn {
        await container.enableCachingAsync(
            config: settings.cacheCoordinatorConfig(
                modelKey: "ink2-apppath", diskCacheDirectory: diskDir))
    }
    let engine = await container.makeBatchEngine(maxBatchSize: 1)
    var system = "You are a helpful assistant. Answer carefully and completely.\n"
    var i = 1
    while system.count < 8000 {
        system +=
            "Guideline \(i): keep answers accurate, structured and free of filler (item \(i)).\n"
        i += 1
    }
    let message =
        env["BENCH_INKLING2_CHAT"]
        ?? "Write a detailed 700-word explanation of how a refrigerator works, covering the compressor, condenser, expansion valve and evaporator."
    for r in 0 ..< runs {
        let input = try await context.processor.prepare(
            input: UserInput(chat: [.system(system), .user(message)]))
        var p = GenerateParameters(
            maxTokens: maxNew, temperature: Float(env["BENCH_INKLING2_TEMP"] ?? "0") ?? 0)
        if env["BENCH_INKLING2_SWEEP_DSPARK"] == "1" {
            p.draftStrategy = .dspark(drafterPath: modelDir)
        }
        nonisolated(unsafe) let send = input
        var restored = 0
        var info: GenerateCompletionInfo?
        for await item in await engine.generate(input: send, parameters: p) {
            switch item {
            case .prefillProgress(let progress):
                if progress.stage == .cacheRestore {
                    restored = max(restored, progress.completedUnitCount)
                }
            case .info(let x): info = x
            default: break
            }
        }
        guard let info else { continue }
        print(
            String(
                format: "[APPPATH] run=%d cache=%@ prompt=%d restored=%d gen=%d decode tok/s=%.1f",
                r + 1, cacheOn ? "on" : "off", info.promptTokenCount, restored,
                info.generationTokenCount, info.tokensPerSecond))
    }
}


/// BENCH_INKLING2_VISION_REF=<safetensors>: the vision tower's features for stored tiles against a
/// reference (`<name>.tiles` [N, 40, 40, 3], `<name>.grid` [rows, cols], `<name>.features`
/// [N, text_hidden] fp32). Reports max |d|, relative RMS error and the minimum per-tile cosine.
func runInkling2VisionParity(modelPath: String, refPath: String) async throws {
    let context = try await MLXLMCommon.loadModel(
        from: URL(fileURLWithPath: modelPath), using: #huggingFaceTokenizerLoader(),
        loadConfiguration: .default)
    guard let model = context.model as? Inkling2LoopModel, let vision = model.vision else {
        print("[INK2V] INVALID: no vision tower built")
        exit(2)
    }
    if ProcessInfo.processInfo.environment["BENCH_INKLING2_VISION_F32"] == "1" {
        // Precision isolation: the whole tower in fp32 (pair with MLX_ENABLE_TF32=0 and an fp32 reference).
        try vision.update(
            parameters: vision.parameters().mapValues { $0.asType(.float32) }, verify: .none)
    }
    let ref = try MLX.loadArrays(url: URL(fileURLWithPath: refPath))
    let names = Set(ref.keys.compactMap { $0.split(separator: ".").first.map(String.init) }).sorted()
    for name in names {
        guard let tiles = ref["\(name).tiles"], let grid = ref["\(name).grid"],
            let want = ref["\(name).features"]
        else { continue }
        let g = grid.asArray(Int32.self)
        let got = vision.features(tiles: tiles, rows: Int(g[0]), cols: Int(g[1])).asType(.float32)
        let d = got - want
        let maxAbs = MLX.abs(d).max().item(Float.self)
        let rel = (MLX.sqrt(MLX.mean(d * d)) / MLX.sqrt(MLX.mean(want * want))).item(Float.self)
        let cos = (MLX.sum(got * want, axis: -1)
            / (MLX.sqrt(MLX.sum(got * got, axis: -1)) * MLX.sqrt(MLX.sum(want * want, axis: -1)) + 1e-12))
            .min().item(Float.self)
        let mag = MLX.abs(want).mean().item(Float.self)
        print(String(
            format: "[INK2V] %@ grid=%dx%d tiles=%d ref_mean_abs=%.4f max_abs_diff=%.4f rel_rms=%.2e min_cos=%.6f",
            name, Int(g[0]), Int(g[1]), want.dim(0), mag, maxAbs, rel, cos))
    }
}
