// BENCH_FIDELITY=1 — teacher-forced logit fidelity against a stored reference (top-k ids + log-probs per
// position), the Swift twin of the vMLX Python probe that scores a bundle against its bf16 reference:
// prefill each reference sequence in 2,048-token chunks through the runtime cache, then per position
// KL(ref || model) over the reference's top-k ids and top-1 agreement (model argmax == reference argmax).
// Equal numbers from both runtimes on the same sequences establish Swift/Python logit parity without running
// the other runtime. Env: BENCH_FIDELITY_REF (safetensors: p<j>.top_ids, p<j>.top_logprobs),
// BENCH_FIDELITY_SEQS (jsonl rows with "ids" and "set"), BENCH_FIDELITY_SET (default klref),
// BENCH_FIDELITY_N (default 8), BENCH_FIDELITY_CHUNK (default 2048; 1 = teacher-forced 1-row decode steps),
// BENCH_FIDELITY_MAXTOK (score each sequence's first N positions).

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import VMLXTokenizers

func runFidelityBench(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    guard let refPath = env["BENCH_FIDELITY_REF"], let seqPath = env["BENCH_FIDELITY_SEQS"] else {
        print("[FIDELITY] INVALID: set BENCH_FIDELITY_REF and BENCH_FIDELITY_SEQS")
        return
    }
    let set = env["BENCH_FIDELITY_SET"] ?? "klref"
    let limit = Int(env["BENCH_FIDELITY_N"] ?? "8") ?? 8
    let chunk = Int(env["BENCH_FIDELITY_CHUNK"] ?? "2048") ?? 2048
    // BENCH_FIDELITY_MAXTOK: score only each sequence's first N positions (1-row decode runs, BENCH_FIDELITY_CHUNK=1).
    let maxTokens = Int(env["BENCH_FIDELITY_MAXTOK"] ?? "") ?? Int.max

    let reference = try loadArrays(url: URL(fileURLWithPath: refPath))
    var selected: [(Int, [Int32])] = []
    let lines = try String(contentsOfFile: seqPath, encoding: .utf8).split(separator: "\n")
    for (j, line) in lines.enumerated() where selected.count < limit {
        guard let row = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
            row["set"] as? String == set, reference["p\(j).top_ids"] != nil,
            let ids = row["ids"] as? [Int]
        else { continue }
        selected.append((j, ids.prefix(maxTokens).map { Int32($0) }))
    }
    guard !selected.isEmpty else {
        print("[FIDELITY] INVALID: no reference rows for set=\(set)")
        return
    }

    let context = try await MLXLMCommon.loadModel(
        from: URL(fileURLWithPath: modelPath), using: #huggingFaceTokenizerLoader(),
        loadConfiguration: LoadConfiguration(nativeMTP: true)
    )
    guard let model = context.model as? any NativeMTPAutoregressiveBackboneModel else {
        print("[FIDELITY] INVALID: model lacks a native backbone forward")
        return
    }

    var kls: [Float] = []
    var agree = 0, positions = 0
    let started = Date()
    for (j, ids) in selected {
        let tokens = MLXArray(ids).reshaped(1, -1)
        let cache = model.newCache(parameters: GenerateParameters())
        let refIDs = reference["p\(j).top_ids"]!.asType(.int32)
        let refLP = reference["p\(j).top_logprobs"]!.asType(.float32)
        var rowAgree = 0
        var start = 0
        while start < ids.count {
            let end = min(start + chunk, ids.count)
            let logits = model.nativeBackboneForward(tokens[0..., start ..< end], cache: cache).logits[0]
                .asType(.float32)
            let rt = refIDs[start ..< end], rl = refLP[start ..< end]
            let lp = logits - logSumExp(logits, axis: -1, keepDims: true)
            let qv = takeAlong(lp, rt, axis: -1)
            let rp = softmax(rl, axis: -1), qp = softmax(qv, axis: -1)
            let kl = (rp * (log(rp + 1e-12) - log(qp + 1e-12))).sum(axis: -1)
            let hits = (argMax(lp, axis: -1).asType(.int32) .== rt[0..., 0]).sum()
            MLX.eval(kl, hits)
            kls.append(contentsOf: kl.asArray(Float.self))
            let h = hits.item(Int.self)
            agree += h
            rowAgree += h
            positions += end - start
            start = end
        }
        print(String(format: "[FIDELITY] row=p%d tokens=%d top1=%.2f%%", j, ids.count,
                     100 * Double(rowAgree) / Double(ids.count)))
        MLX.Memory.clearCache()
    }
    let sorted = kls.sorted()
    let median = sorted.count % 2 == 1
        ? sorted[sorted.count / 2] : (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
    let mean = kls.reduce(0, +) / Float(kls.count)
    print(String(
        format: "[FIDELITY] RESULT set=%@ sequences=%d positions=%d median_kl=%.5f mean_kl=%.5f top1=%.3f%% wall=%.1fs",
        set, selected.count, positions, median, mean, 100 * Double(agree) / Double(positions),
        Date().timeIntervalSince(started)))
}
