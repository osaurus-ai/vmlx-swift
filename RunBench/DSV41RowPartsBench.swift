// BENCH_DSV41_ROWPARTS=1 — DeepSeek-V4.1 verify cost by row count, whole forward and per part (DSV4.1 JANGHT
// campaign, 2026-10-26; Swift side of the Python row-cost numbers: 1 row 36 ms, ~7-8 ms per extra row on m5-max).
//
// Loads the bundle as served (dSpark head attached unless BENCH_ROWPARTS_HEAD=0), prefills a ~1k-token prompt, then
// for each R in BENCH_ROWPARTS_ROWS (default 1,2,4,6):
//   forward  one R-token forward from an in-place checkpoint of the prefilled cache, with the dSpark verify capture
//            on (what a verify runs), restored after each repetition;
//   parts    the same projections over every layer on random rows, one eval per part: the mxfp8 attention
//            projections (wq_a, wq_b, wkv, wo_b), the mxfp8 shared expert (gate, up, down), and the routed experts
//            (R rows, top-k distinct random experts per row) through the layer's own routed module.
// Median of BENCH_ROWPARTS_REPS (default 7) after 2 warm-ups. Prints "[ROWPARTS] part=... rows=R ms=...".

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
import VMLXTokenizers

func runDSV41RowParts(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let modelDir = URL(fileURLWithPath: modelPath)
    let reps = Int(env["BENCH_ROWPARTS_REPS"] ?? "7") ?? 7
    let rowsList = (env["BENCH_ROWPARTS_ROWS"] ?? "1,2,4,6").split(separator: ",").compactMap { Int($0) }
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader(), loadConfiguration: .default)
    let model = context.model
    var session: (any DSparkDraftSession)?
    if env["BENCH_ROWPARTS_HEAD"] != "0", let target = model as? any DSparkSpeculativeModel,
        try target.attachDSparkDraft(directory: modelDir.appendingPathComponent("dspark"))
    {
        session = target.makeDSparkDraftSession()
    }
    guard let root = model as? Module else { return }
    var modules: [String: Module] = [:]
    for (name, module) in root.namedModules() { modules[name] = module }
    var layers = 0
    while modules["model.layers.\(layers).self_attn.wq_b"] != nil { layers += 1 }
    let topK = 6

    func median(_ body: () -> Void) -> Double {
        for _ in 0 ..< 2 { body() }
        var times: [Double] = []
        for _ in 0 ..< reps {
            let t0 = CFAbsoluteTimeGetCurrent()
            body()
            times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        return times.sorted()[times.count / 2]
    }
    func report(_ part: String, _ rows: Int, _ ms: Double) {
        print(String(format: "[ROWPARTS] part=%@ rows=%d ms=%.2f", part, rows, ms))
    }
    func inputs(of module: Module) -> Int? {
        guard let q = module as? QuantizedLinear else { return (module as? Linear)?.weight.dim(1) }
        return q.scales.dim(-1) * q.groupSize
    }

    // Prefill (~1k tokens of plain text) the way the iterator does.
    let text = String(repeating: "The quick brown fox jumps over the lazy dog while the compiler checks every branch. ", count: 60)
    let promptTokens = context.tokenizer.encode(text: text)
    let cache = model.newCache(parameters: GenerateParameters())
    switch try model.prepare(
        LMInput(tokens: MLXArray(promptTokens.map { Int32($0) }).reshaped(1, -1)), cache: cache,
        windowSize: GenerateParameters().prefillStepSize)
    {
    case .tokens(let rest): eval(model(rest, cache: cache, state: nil).logits)
    case .logits(let out): eval(out.logits)
    }
    print("[ROWPARTS] layers=\(layers) prompt=\(promptTokens.count) head=\(session != nil)")

    for rows in rowsList {
        let ids = MLXArray((0 ..< rows).map { Int32(100 + 37 * $0) }).reshaped(1, rows)
        guard let checkpoint = SpeculativeCacheListCheckpoint(cache) else { break }
        report("forward", rows, median {
            let capturing = session?.beginVerifyCapture(cache: cache) ?? false
            eval(model(LMInput.Text(tokens: ids), cache: cache, state: nil).logits)
            if capturing { session?.endVerifyCapture(cache: cache) }
            checkpoint.restore()
        })
        for name in ["wq_a", "wq_b", "wkv", "wo_b"] {
            let mods = (0 ..< layers).compactMap { modules["model.layers.\($0).self_attn.\(name)"] as? UnaryLayer }
            guard let first = mods.first as? Module, let k = inputs(of: first) else { continue }
            let x = MLXRandom.normal([1, rows, k]).asType(.bfloat16)
            report(name, rows, median { eval(mods.map { $0(x) }) })
        }
        let shared = (0 ..< layers).compactMap { modules["model.layers.\($0).mlp.shared_experts"] as? UnaryLayer }
        if let first = modules["model.layers.0.mlp.shared_experts.gate_proj"], let k = inputs(of: first) {
            let x = MLXRandom.normal([1, rows, k]).asType(.bfloat16)
            report("shared", rows, median { eval(shared.map { $0(x) }) })
        }
        let routed = (0 ..< layers).compactMap {
            modules["model.layers.\($0).mlp.switch_mlp"] as? any WeightedRoutedExpertLayer
        }
        if !routed.isEmpty, let k = modules["model.layers.0.mlp.shared_experts.gate_proj"].flatMap(inputs(of:)) {
            let x = MLXRandom.normal([1, rows, k]).asType(.bfloat16) * 0.5
            // Distinct experts across rows (a verify block rarely shares): row r takes 6r ..< 6r + 6, spread.
            let idx = MLXArray((0 ..< rows * topK).map { Int32(($0 * 41) % 256) }).reshaped(1, rows, topK)
            let scores = MLXArray.ones([1, rows, topK], dtype: .float32) / Float(topK)
            report("routed", rows, median { eval(routed.map { $0.callRouted(x, indices: idx, scores: scores) }) })
        }
    }
}
