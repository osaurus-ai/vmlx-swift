// BENCH_INKLING2_PARITY=1 — Inkling-2-Mini (inkling2_loop) port check against the jang_tools MLX port
// (parity-tested vs transformers PR #51050). Development iteration only, not app proof.
//
// Reads BENCH_INKLING2_REF (py_reference.py output JSON + *_last.npy written as raw .f32 next to it) and reports
// per prompt: chunked-prefill last-row logits vs Python (max |d|, KL, top-1, top-10 overlap), no-cache vs cached
// consistency, greedy continuation agreement, and decode timing.

import Foundation
import CmlxGraphShim
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import VMLXTokenizers

func runInkling2ParityProbe(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    guard let refPath = env["BENCH_INKLING2_REF"] else {
        print("[INK2] BENCH_INKLING2_REF is required"); exit(2)
    }
    let chunk = Int(env["BENCH_INKLING2_CHUNK"] ?? "333") ?? 333
    let decodeN = Int(env["BENCH_INKLING2_DECODE"] ?? "128") ?? 128
    let ref = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: refPath)))
        as! [String: [String: Any]]

    let t0 = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: URL(fileURLWithPath: modelPath), using: #huggingFaceTokenizerLoader(),
        loadConfiguration: .default)
    let model = context.model
    print(String(format: "[INK2] load=%.1fs type=%@ active=%.2fGB", CFAbsoluteTimeGetCurrent() - t0,
                 String(describing: type(of: model)), Double(MLX.Memory.activeMemory) / 1e9))

    func kl(_ r: MLXArray, _ c: MLXArray) -> Float {
        let lr = r - logSumExp(r), lc = c - logSumExp(c)
        return (exp(lr) * (lr - lc)).sum().item(Float.self)
    }
    for name in ["short", "long"] {
        guard let r = ref[name], let ids0 = r["ids"] as? [Int], let pyGreedy = r["greedy"] as? [Int],
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
        var line = "[INK2] \(name) n=\(n) prefill=\(String(format: "%.2f", prefillS))s top1 swift=\(swTop[0]) py=\(pyTop[0])"
        line += " top10_overlap=\(Set(swTop).intersection(pyTop).count)"
        line += String(format: " nocache_vs_cached maxabs=%.4f kl=%.2e", abs(nc - last).max().item(Float.self), kl(nc, last))
        if let pyLast {
            line += String(format: " vs_py maxabs=%.4f kl=%.2e", abs(pyLast - last).max().item(Float.self), kl(pyLast, last))
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
        let text = context.tokenizer.decode(tokenIds: Array(gen.prefix(pyGreedy.count)), skipSpecialTokens: false)
        // Teacher-forced vs Python: feed Python's greedy tokens, compare each step's logits.
        let stepsPath = refPath.replacingOccurrences(of: ".json", with: "_\(name)_steps.f32")
        if let d = try? Data(contentsOf: URL(fileURLWithPath: stepsPath)) {
            let all = d.withUnsafeBytes { MLXArray(Array($0.bindMemory(to: Float.self))) }
            let V = all.size / (pyGreedy.count - 1)
            let pySteps = all.reshaped([pyGreedy.count - 1, V])
            var c = model.newCache(parameters: nil)
            var s3 = 0
            while s3 < n { let e = min(s3 + chunk, n); eval(model(ids[0..., s3 ..< e], cache: c)); s3 = e }
            var kls: [Float] = []
            var top1 = 0
            for t in 0 ..< (pyGreedy.count - 1) {
                let lg = model(MLXArray([Int32(pyGreedy[t])])[.newAxis], cache: c)[0, -1].asType(.float32)
                eval(lg)
                kls.append(kl(pySteps[t], lg))
                if lg.argMax().item(Int.self) == pySteps[t].argMax().item(Int.self) { top1 += 1 }
            }
            c = []
            let srt = kls.sorted()
            print(String(format: "[INK2] %@ teacher-forced vs py: %d steps top1 %d/%d KL median %.2e max %.2e",
                         name, kls.count, top1, kls.count, srt[srt.count / 2], srt.last!))
        }
        // Self-consistency control: the same tokens decoded after two different prefill chunkings.
        if name == "long" {
            func tf(_ ck: Int) -> [MLXArray] {
                let c = model.newCache(parameters: nil)
                var s4 = 0
                while s4 < n { let e = min(s4 + ck, n); eval(model(ids[0..., s4 ..< e], cache: c)); s4 = e }
                var out: [MLXArray] = []
                for t in 0 ..< (pyGreedy.count - 1) {
                    let lg = model(MLXArray([Int32(pyGreedy[t])])[.newAxis], cache: c)[0, -1].asType(.float32)
                    eval(lg)
                    out.append(lg)
                }
                return out
            }
            let a1 = tf(512), a2 = tf(333)
            let kls = zip(a1, a2).map { kl($0.0, $0.1) }.sorted()
            print(String(format: "[INK2] control swift(512) vs swift(333): KL median %.2e max %.2e", kls[kls.count / 2], kls.last!))
        }
        // Teacher-forced fast-decode vs general-path decode from identical cache states.
        if name == "long" {
            let gold = pyGreedy
            var cA = model.newCache(parameters: nil)
            var s2 = 0
            while s2 < n { let e = min(s2 + chunk, n); eval(model(ids[0..., s2 ..< e], cache: cA)); s2 = e }
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
            if env["BENCH_INKLING2_LAYERDIFF"] == "1", let im = model as? Inkling2LoopModel {
                for line in im.inklingDebugCompareDecode(token: gold[0], cache: cB.map { $0.copy() }) {
                    print("[INK2L] " + line)
                }
            }
            _ = cA
            print(String(format: "[INK2] fast-vs-general teacher-forced 24 steps: max KL %.3e, argmax mismatches %d", worstKL, mism))
        }
        print("[INK2] \(name) greedy prefix agree=\(agree)/\(pyGreedy.count) decode=\(String(format: "%.1f", Double(gen.count - 1) / decodeS)) tok/s (naive loop, \(gen.count - 1) steps)")
        print("[INK2] \(name) text: \(text.replacingOccurrences(of: "\n", with: "\\n").prefix(300))")
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
        if i >= 4 { builds.append((t1 - t0) * 1000); evals.append((t2 - t1) * 1000) }
        y = tok.reshaped([1, 1]).asType(.int32)
    }
    func med(_ a: [Double]) -> Double { a.sorted()[a.count / 2] }
    print(String(format: "[INK2T] nodes/step=%d build=%.2fms eval(sync)=%.2fms", nodes, med(builds), med(evals)))
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
    print(String(format: "[INK2T] pipelined decode tok/s runs=%@ median=%.1f", rates.map { String(format: "%.1f", $0) }.joined(separator: ","), med(rates)))
}

func runInkling2MicroProbe(modelPath: String) async throws {
    let context = try await MLXLMCommon.loadModel(
        from: URL(fileURLWithPath: modelPath), using: #huggingFaceTokenizerLoader(),
        loadConfiguration: .default)
    guard let m = context.model as? Inkling2LoopModel else { print("[INK2M] not inkling"); return }
    for line in m.inklingMicrobench() { print("[INK2M] " + line) }
}

/// BENCH_INKLING2_TEMPLATE=1: the app's prompt path (UserInput(chat:tools:additionalContext:) ->
/// processor -> chat template -> tokenizer) against transformers' apply_chat_template renders
/// (BENCH_INKLING2_RENDERS json from make_renders.py). Token ids must match exactly.
func runInkling2TemplateProbe(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let refs = try JSONSerialization.jsonObject(
        with: Data(contentsOf: URL(fileURLWithPath: env["BENCH_INKLING2_RENDERS"]!))) as! [String: [String: Any]]
    let context = try await MLXLMCommon.loadModel(
        from: URL(fileURLWithPath: modelPath), using: #huggingFaceTokenizerLoader(),
        loadConfiguration: .default)
    let weather: ToolSpec = ["type": "function", "function": [
        "name": "get_weather", "description": "Current weather for a city",
        "parameters": ["type": "object", "properties": [
            "city": ["type": "string", "description": "City name"] as [String: any Sendable],
            "unit": ["type": "string", "enum": ["c", "f"]] as [String: any Sendable]] as [String: any Sendable],
            "required": ["city"]] as [String: any Sendable]] as [String: any Sendable]]
    let shell: ToolSpec = ["type": "function", "function": [
        "name": "run_shell", "description": "Run a shell command / script",
        "parameters": ["type": "object", "properties": [
            "cmd": ["type": "string"] as [String: any Sendable]] as [String: any Sendable],
            "required": ["cmd"]] as [String: any Sendable]] as [String: any Sendable]]
    let call = ToolCall(function: .init(name: "get_weather", arguments: ["unit": "c", "city": "Paris"] as [String: any Sendable]), id: "call_1")
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
        "tool_roundtrip": ([.user("Weather in Paris?"), assistantTool,
                            .tool("{\"temp\": 18, \"sky\": \"clear\"}", toolCallId: "call_1"),
                            .user("Thanks. And unicode: café ☕ / path a/b")], [weather, shell], nil),
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
                let k = (0 ..< min(ids.count, refIds.count)).first(where: { ids[$0] != refIds[$0] }) ?? min(ids.count, refIds.count)
                let swiftText = context.tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
                print("[INK2TPL] \(name): MISMATCH swift=\(ids.count) py=\(refIds.count) first diff at \(k)")
                print("  swift: \(swiftText.replacingOccurrences(of: "\n", with: "⏎"))")
                print("  py   : \((ref["text"] as? String ?? "").replacingOccurrences(of: "\n", with: "⏎"))")
            }
        } catch {
            fails += 1
            print("[INK2TPL] \(name): ERROR \(error)")
        }
    }
    print("[INK2TPL] \(fails == 0 ? "ALL MATCH" : "\(fails) FAILED")")
}

/// BENCH_INKLING2_DSPARK=1: greedy DSpark (native drafter) vs plain greedy decode on the parity
/// prompts: acceptance per round, emitted-token agreement, and wall tok/s for both.
func runInkling2DSparkProbe(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let ref = try JSONSerialization.jsonObject(
        with: Data(contentsOf: URL(fileURLWithPath: env["BENCH_INKLING2_REF"]!))) as! [String: [String: Any]]
    let maxNew = Int(env["BENCH_INKLING2_NEW"] ?? "192") ?? 192
    let widths = (env["BENCH_INKLING2_WIDTHS"] ?? "7").split(separator: ",").compactMap { Int($0) }
    let context = try await MLXLMCommon.loadModel(
        from: URL(fileURLWithPath: modelPath), using: #huggingFaceTokenizerLoader(),
        loadConfiguration: .default)
    guard let m = context.model as? Inkling2LoopModel, m.dsparkAvailable else {
        print("[INK2D] drafter not available"); return
    }
    for name in ["short", "long"] {
        guard let ids = ref[name]?["ids"] as? [Int] else { continue }
        // plain greedy (fast decode path)
        var cache = m.newCache(parameters: nil)
        var s = 0
        var last = MLXArray(0)
        let arr = MLXArray(ids.map { Int32($0) }).reshaped([1, -1])
        while s < ids.count { let e = min(s + 512, ids.count); last = m(arr[0..., s ..< e], cache: cache)[0, -1]; eval(last); s = e }
        var plain = [last.argMax().item(Int.self)]
        let t0 = CFAbsoluteTimeGetCurrent()
        while plain.count < maxNew {
            let lg = m(MLXArray([Int32(plain.last!)]).reshaped([1, 1]), cache: cache)[0, -1]
            plain.append(lg.argMax().item(Int.self))
        }
        let plainTps = Double(maxNew - 1) / (CFAbsoluteTimeGetCurrent() - t0)
        for w in widths {
            cache = m.newCache(parameters: nil)
            let (sess, _) = m.dsparkPrefill(ids, cache: cache)
            var out = [sess.anchor]
            let t1 = CFAbsoluteTimeGetCurrent()
            while out.count < maxNew { out += m.dsparkRound(sess, cache: cache, width: w) }
            let dt = CFAbsoluteTimeGetCurrent() - t1
            var agree = 0
            for (a, b) in zip(out, plain) { if a == b { agree += 1 } else { break } }
            print(String(format: "[INK2D] %@ w=%d rounds=%d accepted/round=%.2f emitted/round=%.2f tok/s=%.1f (plain %.1f) agree=%d/%d",
                         name, w, sess.rounds, Double(sess.accepted) / Double(max(sess.rounds, 1)),
                         Double(out.count - 1) / Double(max(sess.rounds, 1)), Double(out.count - 1) / dt, plainTps, agree, maxNew))
            let r = Double(max(sess.rounds, 1)) / 1000
            print(String(format: "[INK2D] %@ w=%d ms/round: block %.1f walk %.1f verify %.1f commit %.1f",
                         name, w, sess.timing[0] / r, sess.timing[1] / r, sess.timing[2] / r, sess.timing[3] / r))
            print("[INK2D] \(name) text: " + context.tokenizer.decode(tokenIds: Array(out.prefix(80)), skipSpecialTokens: false).replacingOccurrences(of: "\n", with: "⏎"))
        }
    }
}
