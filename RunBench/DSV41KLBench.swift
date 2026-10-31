// BENCH_DSV41_KL=1 — teacher-forced A/B of a DeepSeek-V4.1 prefill switch on held-out chat sequences, in one process.
//
// Every sequence is run through two fresh caches in BENCH_KL_CHUNK-row forwards (default 1024, the served chunk
// width), chunk by chunk: arm A with the environment variable BENCH_KL_ENV (default VMLX_DSV41_ATTN_NAX) unset,
// arm B with it set to "1" (the runtime reads the switch per call). Scored positions are the ASSISTANT positions:
// position i is scored when token i+1 lies in an assistant span (after <｜Assistant｜>, through the next
// <｜end▁of▁sentence｜>, or up to the next <｜User｜>). Per position: KL(A || B) over the full vocabulary (fp32 log
// softmax), top-1 agreement A vs B, and the next-token NLL under each arm. With BENCH_KL_REF (safetensors from the
// Python runtime: p<j>.pos int32 scored positions, p<j>.top_ids / p<j>.top_logprobs (n, k)), also KL(ref || A) and
// KL(ref || B) over the reference's top-k (renormalized on both sides) and top-1 agreement with the reference.
// BENCH_KL_SEQS: jsonl rows {"id", "domain", "ids"}; BENCH_KL_N caps the rows; BENCH_KL_OUT appends the JSON lines.

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import VMLXTokenizers

func runDSV41KLBench(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let modelDir = URL(fileURLWithPath: modelPath)
    let chunk = Int(env["BENCH_KL_CHUNK"] ?? "1024") ?? 1024
    let limit = Int(env["BENCH_KL_N"] ?? "") ?? Int.max
    let switchName = env["BENCH_KL_ENV"] ?? "VMLX_DSV41_ATTN_NAX"
    let label = env["BENCH_KL_LABEL"] ?? "kl"
    guard let seqPath = env["BENCH_KL_SEQS"] else {
        print("[KL] INVALID: set BENCH_KL_SEQS")
        return
    }
    let outPath = env["BENCH_KL_OUT"]
    func emit(_ tag: String, _ record: [String: Any]) {
        var r = record
        r["label"] = label
        let line = String(
            decoding: (try? JSONSerialization.data(withJSONObject: r, options: [.sortedKeys])) ?? Data(),
            as: UTF8.self)
        print("[\(tag)] \(line)")
        fflush(stdout)
        if let outPath, let data = "[\(tag)] \(line)\n".data(using: .utf8) {
            if let h = FileHandle(forWritingAtPath: outPath) {
                h.seekToEndOfFile()
                h.write(data)
                try? h.close()
            } else {
                FileManager.default.createFile(atPath: outPath, contents: data)
            }
        }
    }

    var rows: [(id: String, domain: String, ids: [Int32])] = []
    for line in try String(contentsOfFile: seqPath, encoding: .utf8).split(separator: "\n") where rows.count < limit {
        guard let r = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
            let ids = r["ids"] as? [Int]
        else { continue }
        rows.append(((r["id"] as? String) ?? "\(rows.count)", (r["domain"] as? String) ?? "?", ids.map { Int32($0) }))
    }
    let reference = env["BENCH_KL_REF"].map { try? loadArrays(url: URL(fileURLWithPath: $0)) } ?? nil

    /// Positions i (0-based) whose next token i+1 is inside an assistant span.
    func scored(_ ids: [Int32]) -> [Int] {
        var out: [Int] = []
        var inAssistant = false
        for (k, t) in ids.enumerated() {
            if t == 128_804 { inAssistant = true; continue }      // <｜Assistant｜>
            if inAssistant && t == 128_803 { inAssistant = false; continue }  // <｜User｜>
            if inAssistant, k > 0 { out.append(k - 1) }
            if t == 1 { inAssistant = false }                     // <｜end▁of▁sentence｜>
        }
        return out
    }

    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader(), loadConfiguration: .osaurusProduction)
    let model = context.model
    emit("KL_LOAD", [
        "model": modelDir.lastPathComponent, "load_s": CFAbsoluteTimeGetCurrent() - loadStart, "rows": rows.count,
        "switch": switchName, "chunk": chunk, "reference": reference != nil,
    ])

    func setArm(_ b: Bool) { if b { setenv(switchName, "1", 1) } else { unsetenv(switchName) } }
    var domainKL: [String: [Float]] = [:], domainAgree: [String: (Int, Int)] = [:]
    var domainNLL: [String: (Double, Double, Int)] = [:]
    var domainRef: [String: (Double, Double, Int, Int, Int)] = [:]
    let started = Date()
    for (j, row) in rows.enumerated() {
        let pos = scored(row.ids)
        guard !pos.isEmpty, row.ids.count > 1 else { continue }
        let posSet = Set(pos)
        let refPos = reference?["p\(j).pos"]?.asArray(Int32.self).map(Int.init)
        let refIDs = reference?["p\(j).top_ids"]?.asType(.int32)
        let refLP = reference?["p\(j).top_logprobs"]?.asType(.float32)
        var refRow: [Int: Int] = [:]
        if let refPos { for (r, p) in refPos.enumerated() { refRow[p] = r } }
        let cacheA = model.newCache(parameters: nil), cacheB = model.newCache(parameters: nil)
        var kls: [Float] = []
        var agree = 0, nllA = 0.0, nllB = 0.0
        var refKA = 0.0, refKB = 0.0, refAgreeA = 0, refAgreeB = 0, refN = 0
        let n = row.ids.count - 1  // the last token predicts nothing scored
        var start = 0
        while start < n {
            let end = min(start + chunk, n)
            let input = MLXArray(Array(row.ids[start ..< end])).reshaped(1, -1)
            setArm(false)
            let la = model(LMInput.Text(tokens: input), cache: cacheA, state: nil).logits[0].asType(.float32)
            MLX.eval(la)
            setArm(true)
            let lb = model(LMInput.Text(tokens: input), cache: cacheB, state: nil).logits[0].asType(.float32)
            MLX.eval(lb)
            setArm(false)
            let local = (start ..< end).filter { posSet.contains($0) }
            if !local.isEmpty {
                let rowsIdx = MLXArray(local.map { Int32($0 - start) })
                let a = take(la, rowsIdx, axis: 0), b = take(lb, rowsIdx, axis: 0)
                let lpa = a - logSumExp(a, axis: -1, keepDims: true)
                let lpb = b - logSumExp(b, axis: -1, keepDims: true)
                let kl = (exp(lpa) * (lpa - lpb)).sum(axis: -1)
                let next = MLXArray(local.map { row.ids[$0 + 1] }).reshaped(-1, 1)
                let na = -takeAlong(lpa, next, axis: -1), nb = -takeAlong(lpb, next, axis: -1)
                let same = (argMax(lpa, axis: -1) .== argMax(lpb, axis: -1)).sum()
                MLX.eval(kl, na, nb, same)
                kls.append(contentsOf: kl.asArray(Float.self))
                agree += same.item(Int.self)
                nllA += Double(na.sum().item(Float.self))
                nllB += Double(nb.sum().item(Float.self))
                if let refIDs, let refLP {
                    let have = local.enumerated().compactMap { (k, p) in refRow[p].map { (k, $0) } }
                    if !have.isEmpty {
                        let mine = MLXArray(have.map { Int32($0.0) }), theirs = MLXArray(have.map { Int32($0.1) })
                        let rid = take(refIDs, theirs, axis: 0), rlp = take(refLP, theirs, axis: 0)
                        let rp = softmax(rlp, axis: -1)
                        func klRef(_ lp: MLXArray) -> MLXArray {
                            let q = softmax(takeAlong(take(lp, mine, axis: 0), rid, axis: -1), axis: -1)
                            return (rp * (log(rp + 1e-12) - log(q + 1e-12))).sum(axis: -1).sum()
                        }
                        let ka = klRef(lpa), kb = klRef(lpb)
                        let top = rid[0..., 0]
                        let ha = (argMax(take(lpa, mine, axis: 0), axis: -1).asType(.int32) .== top).sum()
                        let hb = (argMax(take(lpb, mine, axis: 0), axis: -1).asType(.int32) .== top).sum()
                        MLX.eval(ka, kb, ha, hb)
                        refKA += Double(ka.item(Float.self))
                        refKB += Double(kb.item(Float.self))
                        refAgreeA += ha.item(Int.self)
                        refAgreeB += hb.item(Int.self)
                        refN += have.count
                    }
                }
            }
            start = end
        }
        MLX.Memory.clearCache()
        let m = kls.isEmpty ? 0 : kls.reduce(0, +) / Float(kls.count)
        var rec: [String: Any] = [
            "row": j, "id": row.id, "domain": row.domain, "tokens": row.ids.count, "scored": kls.count,
            "mean_kl_ab": m, "max_kl_ab": kls.max() ?? 0, "top1_ab": Double(agree) / Double(max(1, kls.count)),
            "nll_a": nllA / Double(max(1, kls.count)), "nll_b": nllB / Double(max(1, kls.count)),
        ]
        if refN > 0 {
            rec["ref_n"] = refN
            rec["kl_ref_a"] = refKA / Double(refN)
            rec["kl_ref_b"] = refKB / Double(refN)
            rec["top1_ref_a"] = Double(refAgreeA) / Double(refN)
            rec["top1_ref_b"] = Double(refAgreeB) / Double(refN)
        }
        emit("KL_SEQ", rec)
        domainKL[row.domain, default: []] += kls
        let (ag, tot) = domainAgree[row.domain] ?? (0, 0)
        domainAgree[row.domain] = (ag + agree, tot + kls.count)
        let (a0, b0, c0) = domainNLL[row.domain] ?? (0, 0, 0)
        domainNLL[row.domain] = (a0 + nllA, b0 + nllB, c0 + kls.count)
        let (ra, rb, raa, rab, rn) = domainRef[row.domain] ?? (0, 0, 0, 0, 0)
        domainRef[row.domain] = (ra + refKA, rb + refKB, raa + refAgreeA, rab + refAgreeB, rn + refN)
    }
    var all: [Float] = []
    for (d, v) in domainKL.sorted(by: { $0.key < $1.key }) {
        all += v
        let s = v.sorted()
        let (ag, tot) = domainAgree[d]!
        let (na, nb, c) = domainNLL[d]!
        var rec: [String: Any] = [
            "domain": d, "positions": v.count, "mean_kl_ab": v.reduce(0, +) / Float(max(1, v.count)),
            "median_kl_ab": s.isEmpty ? 0 : s[s.count / 2], "p99_kl_ab": s.isEmpty ? 0 : s[min(s.count - 1, s.count * 99 / 100)],
            "top1_ab": Double(ag) / Double(max(1, tot)), "nll_a": na / Double(max(1, c)), "nll_b": nb / Double(max(1, c)),
        ]
        if let (ra, rb, raa, rab, rn) = domainRef[d], rn > 0 {
            rec["ref_n"] = rn
            rec["kl_ref_a"] = ra / Double(rn)
            rec["kl_ref_b"] = rb / Double(rn)
            rec["top1_ref_a"] = Double(raa) / Double(rn)
            rec["top1_ref_b"] = Double(rab) / Double(rn)
        }
        emit("KL_DOMAIN", rec)
    }
    emit("KL_RESULT", [
        "positions": all.count, "mean_kl_ab": all.reduce(0, +) / Float(max(1, all.count)),
        "wall_s": Date().timeIntervalSince(started),
    ])
}
