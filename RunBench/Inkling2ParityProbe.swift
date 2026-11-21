// BENCH_INKLING2_PARITY=1 — Inkling-2-Mini (inkling2_loop) port check against the jang_tools MLX port
// (parity-tested vs transformers PR #51050). Development iteration only, not app proof.
//
// Reads BENCH_INKLING2_REF (py_reference.py output JSON + *_last.npy written as raw .f32 next to it) and reports
// per prompt: chunked-prefill last-row logits vs Python (max |d|, KL, top-1, top-10 overlap), no-cache vs cached
// consistency, greedy continuation agreement, and decode timing.

import Foundation
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
        print("[INK2] \(name) greedy prefix agree=\(agree)/\(pyGreedy.count) decode=\(String(format: "%.1f", Double(gen.count - 1) / decodeS)) tok/s (naive loop, \(gen.count - 1) steps)")
        print("[INK2] \(name) text: \(text.replacingOccurrences(of: "\n", with: "\\n").prefix(300))")
    }
}
