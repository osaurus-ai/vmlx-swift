// BENCH_LILICORR_ACCEPT=1 BENCH_MODEL=<Nemotron-3.5-Bolt bundle> [BENCH_LILICORR_N=96]
//
// Offline acceptance of NVIDIA's LiLiCorr drafter (`<bundle>/lilicorr`) on OUR Bolt target — the
// Swift twin of the Python harness `lilicorr-ref/lilicorr_accept.py`. Greedy AR on three fixed
// prompts (enable_thinking=false) while recording the drafter's target taps (OUTPUTS of the
// backbone blocks dflash_config.target_layer_ids, before norm_f); then for every generated
// position p: context = taps[0 ..< p], block = [token p, mask x (B-1)] -> drafts for p+1 ..< p+B;
// accepted = longest prefix equal to the AR tokens. Reports tau = 1 + mean accepted for the
// LiLiCorr path and for plain per-slot top-1, for shift 0 (taps of block i) and the off-by-one
// control shift 1 (taps of block i-1), plus per-block drafter timing (draft_hidden + propose).
//
// Diagnostic only: offline acceptance on a fixed greedy stream is not a speed or app measurement.
// Fails closed: a prompt that yields no draft positions is INVALID, never a pass.

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import VMLXTokenizers

enum LiLiCorrAcceptBench {
    static let prompts: [(String, String)] = [
        ("code", "Write a complete Python LRU cache class with get, put, delete and resize, with type hints and docstrings."),
        ("prose", "Write a vivid 400-word story about a lighthouse keeper who finds a message in a bottle."),
        ("explain", "Explain step by step how TCP congestion control works, covering slow start, AIMD and fast recovery."),
    ]

    struct Stats {
        var accepted: [Int] = []
        var tau: Double { 1 + mean }
        var mean: Double { accepted.isEmpty ? 0 : Double(accepted.reduce(0, +)) / Double(accepted.count) }
        func hist(_ block: Int) -> [Int] {
            var h = [Int](repeating: 0, count: block)
            for a in accepted where a < block { h[a] += 1 }
            return h
        }
    }

    struct Timing {
        var draftHiddenMs: [Double] = []
        var proposeMs: [Double] = []
        static func summary(_ v: [Double]) -> String {
            guard !v.isEmpty else { return "n/a" }
            let s = v.sorted()
            let med = s[s.count / 2]
            let mean = v.reduce(0, +) / Double(v.count)
            return String(format: "median %.2f ms, mean %.2f ms, max %.2f ms", med, mean, s.last!)
        }
    }

    static func run(modelDir: URL) async throws {
        let env = ProcessInfo.processInfo.environment
        let n = Int(env["BENCH_LILICORR_N"] ?? "96") ?? 96

        let loadStart = CFAbsoluteTimeGetCurrent()
        let context = try await MLXLMCommon.loadModel(
            from: modelDir, using: #huggingFaceTokenizerLoader(),
            loadConfiguration: .osaurusProduction
        )
        print(String(
            format: "[LILICORR] target load: %.2fs model=%@", CFAbsoluteTimeGetCurrent() - loadStart,
            String(describing: type(of: context.model))))
        guard let target = context.model as? any NativeMTPModel,
            let embedder = context.model as? any TokenEmbedderModel,
            let tappable = context.model as? any NemotronHBlockOutputTappable
        else {
            throw NSError(domain: "LiLiCorrAccept", code: 2, userInfo: [
                NSLocalizedDescriptionKey:
                    "model \(type(of: context.model)) is not a tappable Nemotron-H target"])
        }

        let drafterStart = CFAbsoluteTimeGetCurrent()
        let drafter = try NemotronLiLiCorrDrafter.load(
            directory: modelDir.appendingPathComponent("lilicorr"))
        print(String(
            format: "[LILICORR] drafter load: %.2fs ids=%@ block=%d window=%d tap_dim=%d topk=%d",
            CFAbsoluteTimeGetCurrent() - drafterStart, drafter.targetLayerIds.description,
            drafter.blockSize, drafter.window, drafter.tapDimension,
            drafter.lilicorr?.topK ?? 0))
        let block = drafter.blockSize

        struct Row {
            let name: String
            let shift: Int
            let lili: Stats
            let top1: Stats
            let timing: Timing
            let preview: String
        }
        var rows: [Row] = []
        var invalid: [String] = []

        for (name, prompt) in prompts {
            var input = UserInput(prompt: prompt)
            input.additionalContext = ["enable_thinking": false]
            let promptTokens = try await context.processor.prepare(input: input).text.tokens
                .reshaped(1, -1)
            let promptIds = promptTokens.asArray(Int32.self).map { Int($0) }
            let P = promptIds.count

            for shift in [0, 1] {
                let tapIds = drafter.targetLayerIds.map { $0 - shift }
                guard tapIds.allSatisfy({ $0 >= 0 }) else {
                    print("[LILICORR] \(name)/shift\(shift): SKIP (tap index below 0)")
                    continue
                }
                // 1. greedy AR with tap recording: store[layer] = [(L, D)] per forward
                var store: [Int: [MLXArray]] = [:]
                tappable.blockOutputTap = NemotronHBlockOutputTap(
                    layerIndices: Set(tapIds),
                    record: { idx, y in store[idx, default: []].append(y[0]) })
                let cache = target.newCache(parameters: GenerateParameters())
                var logits = target.nativeBackboneForward(promptTokens, cache: cache).logits
                var toks: [Int] = []
                var next = argMax(logits[0, -1], axis: -1).item(Int.self)
                let arStart = CFAbsoluteTimeGetCurrent()
                for _ in 0 ..< n {
                    toks.append(next)
                    logits = target.nativeBackboneForward(
                        MLXArray([Int32(next)]).reshaped(1, 1), cache: cache
                    ).logits
                    next = argMax(logits[0, -1], axis: -1).item(Int.self)
                }
                tappable.blockOutputTap = nil
                let arSeconds = CFAbsoluteTimeGetCurrent() - arStart
                for id in tapIds {
                    guard let chunks = store[id], !chunks.isEmpty else {
                        throw NSError(domain: "LiLiCorrAccept", code: 3, userInfo: [
                            NSLocalizedDescriptionKey: "no taps recorded for layer \(id)"])
                    }
                    _ = chunks
                }
                let taps = concatenated(
                    tapIds.map { concatenated(store[$0]!, axis: 0) }, axis: -1)  // (P+N, n*D)
                eval(taps)
                store.removeAll()
                let seq = promptIds + toks
                guard taps.dim(0) == seq.count, taps.dim(1) == drafter.tapDimension else {
                    throw NSError(domain: "LiLiCorrAccept", code: 4, userInfo: [
                        NSLocalizedDescriptionKey:
                            "taps \(taps.shape) do not match sequence \(seq.count) x \(drafter.tapDimension)"])
                }
                print(String(
                    format: "[LILICORR] %@/shift%d AR: prompt=%d new=%d decode %.1f tok/s",
                    name, shift, P, toks.count, Double(toks.count) / max(arSeconds, 1e-9)))

                // 2. draft at every generated position
                var lili = Stats()
                var top1 = Stats()
                var timing = Timing()
                drafter.reset()
                try drafter.addContext(taps: taps[0 ..< P], position: 0)
                let maskRow = [Int32](repeating: Int32(drafter.maskTokenId), count: block - 1)
                var p = P
                while p < P + n - block {
                    let blockIds = MLXArray([Int32(seq[p])] + maskRow).reshaped(1, block)
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let h = try drafter.draftHidden(blockEmbeddings: embedder.embed(blockIds), position: p)
                    eval(h)
                    let t1 = CFAbsoluteTimeGetCurrent()
                    // one device sync per proposal (Python 3e93c98a): the lattice walk stays on the
                    // graph; `proposeTokens` evaluates path + top-1 together and reads back once.
                    let (path, plain) = try drafter.proposeTokens(
                        hidden: h, lmHead: { embedder.projectToLogits($0) }, embed: { embedder.embed($0) })
                    let t2 = CFAbsoluteTimeGetCurrent()
                    timing.draftHiddenMs.append((t1 - t0) * 1000)
                    timing.proposeMs.append((t2 - t1) * 1000)
                    let targetTokens = Array(seq[(p + 1) ..< (p + block)])
                    func accepted(_ prop: [Int32]) -> Int {
                        var acc = 0
                        for (a, b) in zip(prop, targetTokens) {
                            if Int(a) != b { break }
                            acc += 1
                        }
                        return acc
                    }
                    if let path { lili.accepted.append(accepted(path)) }
                    top1.accepted.append(accepted(plain))
                    try drafter.addContext(taps: taps[p ..< (p + 1)], position: p)
                    p += 1
                }
                let preview = context.tokenizer.decode(tokenIds: toks, skipSpecialTokens: false)
                let row = Row(
                    name: name, shift: shift, lili: lili, top1: top1, timing: timing,
                    preview: String(preview.prefix(80)))
                rows.append(row)
                if top1.accepted.isEmpty { invalid.append("\(name)/shift\(shift)") }
                print(
                    "[LILICORR] \(name)/shift\(shift) lilicorr tau=\(String(format: "%.3f", lili.tau)) hist=\(lili.hist(block)) | top1 tau=\(String(format: "%.3f", top1.tau)) hist=\(top1.hist(block)) | \(row.preview.debugDescription)")
                print("[LILICORR]   draft_hidden: \(Timing.summary(timing.draftHiddenMs)); propose: \(Timing.summary(timing.proposeMs))")
            }
        }

        print("LILICORR SUMMARY bundle=\(modelDir.lastPathComponent) N=\(n) block=\(block)")
        print("| prompt | shift | n | LiLiCorr tau | mean acc | hist | top-1 tau | mean acc | hist | draft_hidden ms (median) | propose ms (median) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows {
            let dh = r.timing.draftHiddenMs.sorted()
            let pr = r.timing.proposeMs.sorted()
            print(String(
                format: "| %@ | %d | %d | %.3f | %.3f | %@ | %.3f | %.3f | %@ | %.2f | %.2f |",
                r.name, r.shift, r.top1.accepted.count, r.lili.tau, r.lili.mean,
                r.lili.hist(block).description, r.top1.tau, r.top1.mean,
                r.top1.hist(block).description,
                dh.isEmpty ? 0 : dh[dh.count / 2], pr.isEmpty ? 0 : pr[pr.count / 2]))
        }
        if !invalid.isEmpty {
            print("LILICORR INVALID: no draft positions for \(invalid.joined(separator: ", "))")
            exit(3)
        }
        print("LILICORR DONE")
    }
}
