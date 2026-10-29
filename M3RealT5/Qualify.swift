// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Test harness (not shipped), 2026-10-29: qualification of the MiniMax-M3 prefill defaults on the real M3.1 bundle.
//
// evalturns  the held-out rows of `evalrows` (T5_EVAL_IN), scored the same way (score_eval.py reads the output), but
//            every assistant turn is scored after its context was prefilled the way serving prefills a prompt: a
//            fresh cache, the request's aligned plan (turn cuts, store boundary from the generation-prompt suffix,
//            step 2048), `MiniMaxM3Model.prepare` (the non-final pieces, where the routed-expert rule under test
//            applies), the returned final piece, then the turn's tokens teacher-forced. `evalrows` calls the model on
//            fixed chunks and never reaches the pieces' rule, so it cannot qualify that rule. The positions before the
//            first assistant turn are scored by a plain chunked forward (identical in every arm). Per row: the
//            segments, their non-final pieces and how many of them are 18-127 rows (the pieces the sorted-tile
//            default moves off the decode kernels), and a sha of the row's scored log-probs (arm-to-arm identity).
// taps       drafter inputs on a fixed schedule (no AR / speculative governor, no timing): per prompt, the production
//            prefill (plan + prepare + final piece, as TokenIterator without a coordinator), the tap tail the drafter
//            reads (sha of every recorded row), then greedy decode one token per step and, at every step, the
//            drafter's proposal of the ladder top's width for the same seed and cache. Per prompt: tap shas, the
//            proposals' sha, accepted length per step against the arm's own greedy continuation, the greedy tokens'
//            sha8 (comparable with `dspark`'s). Arms are separate processes / binaries.
//
// Prompt sets (T5_PROMPT_SET, also used by `dspark`): chat (the 7 `chatCases`, default), served (vMLX Python
// `served_dspark.py`'s 8 prompts as one user message each, incl. easy_*), all.

import CryptoKit
import Foundation
import MLX

@testable import MLXLLM
@testable import MLXLMCommon

nonisolated(unsafe) let servedPrompts: [(String, String)] = [
    ("code", "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H12M' into total seconds. "
        + "Include type hints, a docstring and handle invalid input with ValueError. Then show three example calls."),
    ("prose", "Write a short story (about 250 words) about a lighthouse keeper who finds a message in a bottle "
        + "written in their own handwriting."),
    ("agentic", "You are operating a Linux terminal. The disk on /var is 97% full and the web server stopped writing "
        + "logs. List, step by step, the exact shell commands you would run to diagnose and fix this, and explain "
        + "each one briefly."),
    ("reasoning", "A train leaves city A at 09:10 at 84 km/h; another leaves city B, 312 km away, at 09:40 toward A at "
        + "96 km/h. At what time do they meet and how far from A? Show the calculation."),
    ("easy_code", "Write a Python dataclass `ServerConfig` with these 12 fields and type hints: name: str, version: str, "
        + "host: str, port: int, user: str, password: str, timeout: float, retries: int, debug: bool, "
        + "log_level: str, cache_dir: str, workers: int. Then add a `to_dict` method and a `from_dict` "
        + "classmethod that each list every field explicitly, one per line. Code only."),
    ("easy_html", "Write a complete HTML page with a table of the 12 months of the year. Each row: month number, month "
        + "name, number of days in a non-leap year. Use one <tr> per month. Code only."),
    ("easy_rewrite", "Rewrite the following text exactly, word for word, but replace every 'cat' with 'dog' and every "
        + "'cats' with 'dogs'. Output only the rewritten text.\n\nThe cat sat by the window every morning, "
        + "watching the street below. Other cats in the neighbourhood came and went, but this cat never "
        + "left its spot. When the cat finally stood up, the cats outside scattered. Everyone in the house "
        + "agreed the cat was the true owner of the window, and no other cat would ever take its place. "
        + "Years later, the cat still sat there, and the cats outside still scattered."),
    ("easy_list", "List the integers from 1 to 40, one per line, in the format 'n squared is n*n' (for example "
        + "'3 squared is 9'). Output only the list."),

]

nonisolated(unsafe) let servedCases: [ChatCase] = servedPrompts.map {
    ChatCase(name: "served_" + $0.0, messages: [.user($0.1)], tools: nil)
}

/// `]~b]ai\n`: an assistant turn's head (its first token is the think / skip decision).
let turnHead = [200019, 1361, 10]

func promptCases(default fallback: String = "chat") -> [ChatCase] {
    switch env["T5_PROMPT_SET"] ?? fallback {
    case "served": return servedCases
    case "all": return chatCases + servedCases
    default: return chatCases
    }
}

/// The routed-expert rule and the other prefill levers this process runs with (for every report).
func prefillLevers() -> [String: Any] {
    [
        "prefill_sorted_all": MiniMaxM3Model.prefillSortedAll,
        "prefill_sorted_min_pairs": MiniMaxM3Model.prefillSortedMinPairs,
        "prefill_batch_rows": MiniMaxM3Model.prefillBatchRows,
        "env": Dictionary(uniqueKeysWithValues: env.filter { $0.key.hasPrefix("VMLX_") || $0.key.hasPrefix("MLX_") }
            .map { ($0.key, $0.value as Any) }),
    ]
}

func sha16(_ array: MLXArray) -> String {
    let values = array.asType(.float32).asArray(Float.self)
    return values.withUnsafeBufferPointer { SHA256.hash(data: Data(buffer: $0)) }.prefix(8)
        .map { String(format: "%02x", $0) }.joined()
}

// MARK: - evalturns

func taskEvalTurns() async throws {
    guard let inPath = env["T5_EVAL_IN"] else { fatalError("T5_EVAL_IN not set") }
    let manifest = try JSONSerialization.jsonObject(
        with: Data(contentsOf: URL(fileURLWithPath: inPath + ".json"))) as! [String: Any]
    let rows = manifest["rows"] as! [[String: Any]]
    let arrays = try loadArrays(url: URL(fileURLWithPath: inPath + ".safetensors"))
    let (model, context, loadMs) = try await productionLoad(bundleURL())
    guard model.alignedPrefillTurnMarkers != nil else {
        fatalError("evalturns: the turn markers did not resolve (VMLX_M3_STRUCTURAL_CUTS=0?)")
    }
    let step = Int(env["T5_STEP"] ?? "2048")!
    // Serving's store boundary: the coordinator's generation-prompt suffix (ModelContainer); T5_SUFFIX=none = TokenIterator
    // without a coordinator (boundary = prompt - 1).
    let suffix = env["T5_SUFFIX"] == "none" ? [] : generationPromptSuffix(context.tokenizer)
    let limit = Int(env["T5_EVAL_LIMIT"] ?? "0")!
    let perShard = Int(env["T5_EVAL_SHARD"] ?? "40")!
    log("evalturns: loaded in \(Int(loadMs)) ms, \(rows.count) rows, step \(step), suffix \(suffix), schema \(model.alignedPrefillSchema ?? "off"), levers \(prefillLevers())")

    var out: [String: MLXArray] = [:]
    var shard = 0
    var rowInfo: [[String: Any]] = []
    let t0 = nowMs()
    var done = 0
    var forwardTokens = 0
    let count = limit > 0 ? min(limit, rows.count) : rows.count
    for i in 0 ..< count {
        let row = rows[i]
        let ids = arrays["ids_\(i)"]!.asType(.int32)
        let ti = arrays["ti_\(i)"]!.asType(.int32)
        let am = arrays["am_\(i)"]!.asType(.int32)
        let n = ids.dim(0)
        let idList = ids.asArray(Int32.self).map(Int.init)
        var starts: [Int] = []
        if n > 3 {
            for p in 3 ..< n where Array(idList[(p - 3) ..< p]) == turnHead { starts.append(p) }
        }
        var qs: [MLXArray] = [], ranks: [MLXArray] = [], tls: [MLXArray] = [], ths: [MLXArray] = [], sks: [MLXArray] = []
        /// Score logits rows for absolute positions `a ..< b`.
        func score(_ logits: MLXArray, _ a: Int, _ b: Int) {
            precondition(logits.dim(0) == b - a, "evalturns: \(logits.dim(0)) logits rows for positions \(a)..<\(b)")
            let lp = logProbs(logits)
            let q = takeAlong(lp, ti[a ..< b], axis: -1)
            let ref = takeAlong(lp, am[a ..< b].reshaped(-1, 1), axis: -1)
            let rank = (lp .> ref).asType(.int32).sum(axis: -1)
            let nxt = min(b, n - 1) - a
            let tl: MLXArray =
                nxt > 0
                ? takeAlong(
                    lp[..<nxt], MLXArray(idList[(a + 1) ..< (a + 1 + nxt)].map(Int32.init)).reshaped(-1, 1), axis: -1
                ).reshaped(-1) : MLXArray.zeros([0], dtype: .float32)
            let th = lp[0..., thinkToken]
            let sk = lp[0..., skipToken]
            eval(q, rank, tl, th, sk)
            qs.append(q); ranks.append(rank); tls.append(tl); ths.append(th); sks.append(sk)
        }
        /// Teacher-forced chunks of ids[a ..< b] on `cache`, scoring every position.
        func forward(_ cache: [KVCache], _ a: Int, _ b: Int) {
            var s = a
            while s < b {
                let e = min(s + chunk, b)
                let logits = model(MLXArray(idList[s ..< e].map(Int32.init)).reshaped(1, e - s), cache: cache)[0]
                score(logits, s, e)
                forwardTokens += e - s
                s = e
            }
        }
        // Segment 0: positions before the first assistant turn's decision position (plain chunks).
        let e0 = starts.first.map { $0 - 1 } ?? n
        forward(model.newCache(parameters: nil), 0, e0)
        var segments: [[String: Any]] = []
        for (k, s) in starts.enumerated() {
            let e = k + 1 < starts.count ? starts[k + 1] - 1 : n
            let cache = model.newCache(parameters: nil)
            let prompt = Array(idList[0 ..< s])
            let plan = AlignedPrefill.plan(
                model: model, cache: cache, promptTokenIds: prompt, cachePrefixTokenCounts: [],
                generationPromptSuffix: suffix, prefillStepSize: step)
            AlignedPrefill.install(plan, into: cache)
            var record: AlignedPrefillPrefillRecord?
            AlignedPrefill.hooks.prefillObserver = { record = $0 }
            let prepared = try model.prepare(
                LMInput(tokens: MLXArray(prompt.map(Int32.init))), cache: cache, windowSize: step)
            AlignedPrefill.hooks.prefillObserver = nil
            guard case .tokens(let text) = prepared, let record, record.aligned else {
                fatalError("evalturns: row \(i) turn \(k): prepare did not follow the plan \(String(describing: plan))")
            }
            let tail = text.tokens.reshaped(-1).asType(.int32)
            let tailCount = tail.size
            precondition(record.pieces.last.map { $0.count == tailCount } ?? false, "evalturns: tail != last piece")
            let logits = model(tail.reshaped(1, tailCount), cache: cache)[0]
            score(logits[(tailCount - 1)...], s - 1, s)
            forwardTokens += s
            forward(cache, s, e)
            let nonFinal = record.pieces.dropLast().map(\.count)
            segments.append([
                "start": s, "end": e, "pieces": record.pieces.count, "non_final": nonFinal,
                "affected": nonFinal.filter { $0 * 4 >= 72 && $0 * 4 < 512 }.count,
                "affected_rows": nonFinal.filter { $0 * 4 >= 72 && $0 * 4 < 512 }.reduce(0, +),
                "boundary": plan?.boundary ?? -1, "turn_cuts": plan?.turnCuts?.count ?? -1,
            ])
            Memory.clearCache()
        }
        let q = concatenated(qs, axis: 0)
        precondition(q.dim(0) == n, "evalturns: row \(i) scored \(q.dim(0)) of \(n) positions")
        out["q_\(i)"] = q
        out["rank_\(i)"] = concatenated(ranks, axis: 0)
        out["tl_\(i)"] = concatenated(tls, axis: 0)
        out["th_\(i)"] = concatenated(ths, axis: 0)
        out["sk_\(i)"] = concatenated(sks, axis: 0)
        rowInfo.append([
            "i": i, "id": row["id"] ?? "", "domain": row["domain"] ?? "", "n": n, "turns": starts.count,
            "segments": segments, "q_sha": sha16(q),
        ])
        done += n
        Memory.clearCache()
        if (i + 1) % perShard == 0 || i == count - 1 {
            try save(arrays: out, url: outDir.appendingPathComponent("evalturns_\(tag)_\(shard).safetensors"))
            out = [:]
            shard += 1
        }
        if i % 10 == 0 {
            log("evalturns [\(i + 1)/\(count)] \(row["domain"] ?? "") \(n) tok, \(starts.count) turns, \(done) total, \(forwardTokens) forwarded, \(Int((nowMs() - t0) / 1000)) s")
        }
    }
    try writeJSON(
        [
            "rows": count, "shards": shard, "tokens": done, "forward_tokens": forwardTokens,
            "seconds": (nowMs() - t0) / 1000, "load_ms": loadMs, "step": step, "suffix": suffix,
            "schema": model.alignedPrefillSchema ?? "off", "levers": prefillLevers(), "row_info": rowInfo,
        ], "evalturns_\(tag).json")
    log("evalturns: \(count) rows, \(done) tokens, \(forwardTokens) forwarded, \(Int((nowMs() - t0) / 1000)) s")
}

// MARK: - taps

func taskTaps() async throws {
    let dir = bundleURL()
    let (model, context, _) = try await productionLoad(dir)
    guard let drafter = try attachDrafter(model, dir), let head = model.dsparkHead else {
        fatalError("taps: no drafter resolved")
    }
    let steps = Int(env["T5_TAPS_STEPS"] ?? "256")!
    let width = model.dsparkLadderTop ?? 14
    let step = Int(env["T5_STEP"] ?? "2048")!
    var report: [String: Any] = [
        "drafter": drafter.path, "width": width, "steps": steps, "levers": prefillLevers(),
        "schema": model.alignedPrefillSchema ?? "off",
    ]
    var rows: [[String: Any]] = []
    for c in promptCases() {
        let input = try await prepare(context, c.messages, tools: c.tools, additional: thinkingContext(env["T5_THINKING"]))
        let ids = tokenIds(input)
        let cache = model.newCache(parameters: nil)
        // As TokenIterator without a coordinator (the `dspark` command): suffix [] -> boundary = prompt - 1.
        let plan = AlignedPrefill.plan(
            model: model, cache: cache, promptTokenIds: ids, cachePrefixTokenCounts: [], generationPromptSuffix: [],
            prefillStepSize: step)
        AlignedPrefill.install(plan, into: cache)
        var record: AlignedPrefillPrefillRecord?
        AlignedPrefill.hooks.prefillObserver = { record = $0 }
        let prepared = try model.prepare(LMInput(tokens: MLXArray(ids.map(Int32.init))), cache: cache, windowSize: step)
        AlignedPrefill.hooks.prefillObserver = nil
        guard case .tokens(let text) = prepared else { fatalError("taps: prepare returned logits") }
        let tail = text.tokens.reshaped(-1).asType(.int32)
        let first = model(tail.reshaped(1, tail.size), cache: cache)[0, -1]
        var token = argMax(first, axis: -1).item(Int.self)
        let firstSha = sha16(first)
        guard let holder = MiniMaxM3DSpark.tapHolder(cache), let prefillTail = holder.dsparkTail,
            let prefillRows = prefillTail.rows(from: prefillTail.startPosition, to: prefillTail.endPosition)
        else { fatalError("taps: no tap tail after the prefill of \(c.name)") }
        let tapsPrefill = sha16(prefillRows)
        let tapsPrefillRange = [prefillTail.startPosition, prefillTail.endPosition]
        let session = MiniMaxM3DSparkSession(head: head)
        var generated: [Int] = []
        var proposals: [[Int]] = []
        for _ in 0 ..< steps {
            proposals.append(session.draft(seed: token, cache: cache, count: width))
            generated.append(token)
            if token == eosToken { break }
            let logits = model(MLXArray([Int32(token)]).reshaped(1, 1), cache: cache)
            token = argMax(logits[0, -1], axis: -1).item(Int.self)
        }
        // Accepted length of the proposal at step j: its longest prefix equal to the greedy tokens after the seed
        // (only steps whose whole window of continuation exists).
        var accepted: [Int] = []
        for (j, p) in proposals.enumerated() where j + 1 + width <= generated.count {
            var a = 0
            while a < p.count && p[a] == generated[j + 1 + a] { a += 1 }
            accepted.append(a)
        }
        var propData = Data()
        for p in proposals { for t in p { withUnsafeBytes(of: Int32(t).littleEndian) { propData.append(contentsOf: $0) } } }
        let propSha = SHA256.hash(data: propData).prefix(8).map { String(format: "%02x", $0) }.joined()
        let endTail = holder.dsparkTail
        let tapsEnd = endTail.flatMap { $0.rows(from: $0.startPosition, to: $0.endPosition) }.map(sha16) ?? "none"
        let meanAccepted = accepted.isEmpty ? 0 : Double(accepted.reduce(0, +)) / Double(accepted.count)
        let nonFinal = (record?.pieces.dropLast().map(\.count)) ?? []
        rows.append([
            "name": c.name, "prompt_tokens": ids.count, "pieces": record?.pieces.map { [$0.lowerBound, $0.upperBound] } ?? [],
            "affected_pieces": nonFinal.filter { $0 * 4 >= 72 && $0 * 4 < 512 }.count,
            "first_logits_sha": firstSha, "taps_prefill_sha": tapsPrefill, "taps_prefill_range": tapsPrefillRange,
            "taps_end_sha": tapsEnd, "proposals_sha": propSha, "generated": generated.count,
            "greedy_sha8": sha8(generated), "accepted_per_step": accepted, "mean_accepted": meanAccepted,
            "proposals_head": Array(proposals.prefix(8)),
            "text": context.tokenizer.decode(tokenIds: generated, skipSpecialTokens: false),
        ])
        log("taps \(c.name): prompt \(ids.count), pieces \(record?.pieces.count ?? -1) (affected \(nonFinal.filter { $0 * 4 >= 72 && $0 * 4 < 512 }.count)), first \(firstSha), taps prefill \(tapsPrefill) \(tapsPrefillRange), end \(tapsEnd), proposals \(propSha), greedy \(sha8(generated)) (\(generated.count)), mean accepted \(String(format: "%.3f", meanAccepted)) over \(accepted.count) steps")
        Memory.clearCache()
    }
    report["prompts"] = rows
    try writeJSON(report, "taps_\(tag).json")
}
