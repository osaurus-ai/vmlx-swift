// BENCH_MTP_PROXY=1 — teacher-forced native-MTP draft quality on stored sequences: the trunk's pre-mixer hidden at
// position t plus token t+1 through the MTP head predicts token t+2. Reports acceptance (argmax == actual t+2) and
// top-1 agreement with a stored reference head (m<j>.top_ids). Env: BENCH_MTP_PROXY_SEQS (jsonl "ids"),
// BENCH_MTP_PROXY_REF (safetensors), BENCH_MTP_PROXY_N (default 16), BENCH_MTP_PROXY_MAXTOK (default 1024).

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import VMLXTokenizers

func runMTPProxyBench(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    guard let seqPath = env["BENCH_MTP_PROXY_SEQS"] else {
        print("[MTPPROXY] INVALID: set BENCH_MTP_PROXY_SEQS")
        return
    }
    let limit = Int(env["BENCH_MTP_PROXY_N"] ?? "16") ?? 16
    let maxTokens = Int(env["BENCH_MTP_PROXY_MAXTOK"] ?? "1024") ?? 1024
    let reference = try env["BENCH_MTP_PROXY_REF"].map { try loadArrays(url: URL(fileURLWithPath: $0)) }
    var lines = try String(contentsOfFile: seqPath, encoding: .utf8).split(separator: "\n").map(String.init)
    let context = try await MLXLMCommon.loadModel(
        from: URL(fileURLWithPath: modelPath), using: #huggingFaceTokenizerLoader(),
        loadConfiguration: LoadConfiguration(nativeMTP: true)
    ).0
    guard let model = context.model as? any NativeMTPAutoregressiveBackboneModel,
        let mtp = context.model as? any NativeMTPModel, mtp.nativeMTPAvailable
    else {
        print("[MTPPROXY] INVALID: model has no native MTP head")
        return
    }
    // BENCH_MTP_PROXY_PROMPT: score the chat prompt plus its greedy AR continuation (BENCH_MTP_PROXY_GEN tokens)
    // instead of the stored sequences — the live-drafting control for the same text.
    var generatedFrom: Int?
    if let promptText = env["BENCH_MTP_PROXY_PROMPT"] {
        let promptIds = try context.tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": promptText]], tools: nil, additionalContext: nil)
        let gen = Int(env["BENCH_MTP_PROXY_GEN"] ?? "256") ?? 256
        let cache = model.newCache(parameters: GenerateParameters())
        var ids = promptIds
        var out = model.nativeBackboneForward(MLXArray(ids.map { Int32($0) }).reshaped(1, -1), cache: cache)
        for _ in 0 ..< gen {
            let next = argMax(out.logits[0, -1], axis: -1).item(Int.self)
            ids.append(next)
            out = model.nativeAutoregressiveBackboneForward(MLXArray([Int32(next)]).reshaped(1, 1), cache: cache)
        }
        generatedFrom = promptIds.count
        let data = try JSONSerialization.data(withJSONObject: ["ids": ids])
        lines = [String(decoding: data, as: UTF8.self)]
    }
    var accepted = 0, positions = 0, refAgree = 0, refPositions = 0
    for (j, line) in lines.prefix(limit).enumerated() {
        guard let row = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
            let all = row["ids"] as? [Int], all.count >= 4
        else { continue }
        let ids = Array(all.prefix(maxTokens)).map { Int32($0) }
        let t = ids.count
        let tokens = MLXArray(ids).reshaped(1, t)
        let cache = model.newCache(parameters: GenerateParameters())
        var hidden: [MLXArray] = []
        var start = 0
        while start < t {
            let end = min(start + 2048, t)
            let out = model.nativeBackboneForward(tokens[0..., start ..< end], cache: cache)
            hidden.append(out.hiddenStates)
            MLX.eval(hidden.last!)
            start = end
        }
        let h = concatenated(hidden, axis: 1)
        let draft = mtp.nativeMTPForward(
            hiddenStates: h[0..., 0 ..< (t - 1)], nextTokenIds: tokens[0..., 1 ..< t],
            cache: mtp.makeNativeMTPCache())
        let predicted = argMax(draft.logits[0], axis: -1).asType(.int32)   // position i predicts token i+2
        let actual = tokens[0, 2 ..< t]
        let hits = (predicted[0 ..< (t - 2)] .== actual).sum()
        MLX.eval(hits)
        accepted += hits.item(Int.self)
        positions += t - 2
        var line = String(format: "[MTPPROXY] seq=%d tokens=%d accept=%.2f%%", j, t,
                          100 * Double(hits.item(Int.self)) / Double(t - 2))
        if let from = generatedFrom, from < t - 2 {
            // Draft rows inside the generated continuation: row i (i >= from - 1) predicts token i + 2.
            let lo = from - 1
            let genHits = (predicted[lo ..< (t - 2)] .== actual[lo ..< (t - 2)]).sum().item(Int.self)
            line += String(format: " generated_accept=%.2f%% (rows=%d)", 100 * Double(genHits) / Double(t - 2 - lo),
                           t - 2 - lo)
        }
        if let ref = reference?["m\(j).top_ids"] {
            let n = min(ref.dim(0), t - 1)
            let agree = (predicted[0 ..< n] .== ref[0 ..< n, 0].asType(.int32)).sum().item(Int.self)
            refAgree += agree
            refPositions += n
            line += String(format: " ref_top1=%.2f%%", 100 * Double(agree) / Double(n))
        }
        print(line)
        MLX.Memory.clearCache()
    }
    guard positions > 0 else {
        print("[MTPPROXY] INVALID: no positions scored")
        return
    }
    print(String(format: "[MTPPROXY] RESULT seqs=%d positions=%d accept=%.3f%% ref_top1=%@",
                 min(limit, lines.count), positions, 100 * Double(accepted) / Double(positions),
                 refPositions > 0 ? String(format: "%.3f%%", 100 * Double(refAgree) / Double(refPositions)) : "n/a"))
}
