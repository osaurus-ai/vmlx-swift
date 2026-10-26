// BENCH_DSPARK_DRAFTS=1 — teacher-forced dSpark draft dump, Swift twin of the Python
// `draft_compare.py` (DSV4.1 JANGHT campaign, 2026-10-26).
//
// Reads BENCH_DSPARK_DRAFTS_IN ({prompt, ar, drafts} written by the Python side), prefills the prompt
// the way TokenIterator does (`prepare` with suffix pruning, then the final forward), attaches the
// bundle's dSpark head, and for every teacher token: drafts a block from it (the session syncs its
// windows from the cache's hidden tail), then forwards that token. Both runtimes therefore draft from
// the same contexts. Writes {swift_argmax, drafts} to BENCH_DSPARK_DRAFTS_OUT and prints the offline
// acceptance of each side against the teacher stream and their per-step draft agreement.

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import VMLXTokenizers

func runDSparkDraftDump(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    guard let inPath = env["BENCH_DSPARK_DRAFTS_IN"], let outPath = env["BENCH_DSPARK_DRAFTS_OUT"] else {
        throw NSError(domain: "BENCH_DSPARK_DRAFTS", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "set BENCH_DSPARK_DRAFTS_IN and BENCH_DSPARK_DRAFTS_OUT"])
    }
    struct Teacher: Decodable { let prompt: [Int]; let ar: [Int]; let drafts: [[Int]] }
    let teacher = try JSONDecoder().decode(Teacher.self, from: Data(contentsOf: URL(fileURLWithPath: inPath)))
    let modelDir = URL(fileURLWithPath: modelPath)
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader(), loadConfiguration: .default)
    guard let target = context.model as? any DSparkSpeculativeModel,
        try target.attachDSparkDraft(directory: modelDir.appendingPathComponent("dspark")),
        let session = target.makeDSparkDraftSession()
    else {
        throw NSError(domain: "BENCH_DSPARK_DRAFTS", code: 2, userInfo: [
            NSLocalizedDescriptionKey: "model has no dSpark head"])
    }
    let model = context.model
    let cache = model.newCache(parameters: GenerateParameters())
    let promptTokens = MLXArray(teacher.prompt.map { Int32($0) }).reshaped(1, -1)
    var logits: MLXArray
    switch try model.prepare(
        LMInput(text: LMInput.Text(tokens: promptTokens)), cache: cache,
        windowSize: GenerateParameters().prefillStepSize)
    {
    case .tokens(let rest):
        // `prepare` returns the remainder flat; the model takes [1, L] (the iterator adds the batch axis).
        logits = model(LMInput.Text(tokens: rest.tokens.reshaped(1, -1)), cache: cache, state: nil).logits
    case .logits(let out):
        logits = out.logits
    }
    eval(logits)

    var swiftArgmax: [Int] = []
    var drafts: [[Int]] = []
    for seed in teacher.ar {
        swiftArgmax.append(argMax(logits[0, -1], axis: -1).item(Int.self))
        drafts.append(session.draft(seed: seed, cache: cache))
        logits = model(
            LMInput.Text(tokens: MLXArray([Int32(seed)]).reshaped(1, 1)), cache: cache, state: nil
        ).logits
        eval(logits)
    }

    func acceptance(_ d: [[Int]]) -> Double {
        let bs = d.first?.count ?? 0
        let steps = teacher.ar.count - bs
        guard steps > 0 else { return 0 }
        var total = 0
        for i in 0 ..< steps {
            var k = 0
            while k < bs, d[i][k] == teacher.ar[i + 1 + k] { k += 1 }
            total += k
        }
        return Double(total) / Double(steps)
    }
    let argmaxAgree = zip(swiftArgmax, teacher.ar).filter { $0 == $1 }.count
    var sameBlock = 0
    var sameFirst = 0
    var firstDivergence: Int?
    for (i, (s, p)) in zip(drafts, teacher.drafts).enumerated() {
        if s == p { sameBlock += 1 } else if firstDivergence == nil { firstDivergence = i }
        if s.first == p.first { sameFirst += 1 }
    }
    let summary: [String: Any] = [
        "steps": teacher.ar.count,
        "swift_argmax_agree": argmaxAgree,
        "swift_mean_accept": acceptance(drafts),
        "python_mean_accept": acceptance(teacher.drafts),
        "same_block": sameBlock,
        "same_first_draft": sameFirst,
        "first_divergent_step": firstDivergence ?? -1,
    ]
    print("[BENCH_DSPARK_DRAFTS] " + String(
        data: try JSONSerialization.data(withJSONObject: summary, options: [.sortedKeys]), encoding: .utf8)!)
    try JSONSerialization.data(withJSONObject: ["swift_argmax": swiftArgmax, "drafts": drafts])
        .write(to: URL(fileURLWithPath: outPath))
}
