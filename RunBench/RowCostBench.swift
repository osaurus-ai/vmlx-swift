// BENCH_ROWCOST=1 — target forward cost by row count (Swift twin of vMLX Python `step_ms_rows.py`).
// Loads with native MTP, prefills a ~500-token prompt, then for each row count R times one forward of R new
// tokens from a COPY of the prefilled cache (so every repetition starts from the same state), median of
// BENCH_ROWCOST_REPS (default 9) after 3 warm-ups.
//   exact   : R=1 nativeBackboneForward, R>1 nativeBackboneMTPVerifyForward (the iterator's verify entry,
//             FlashVerificationScope row-exact routes active)
//   plain   : nativeBackboneForward for every R (ordinary multi-row forward, no verification scope)
// Prints one line per R: "[ROWCOST] mode=exact rows=R ms=…".

import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import VMLXTokenizers

func runRowCostBench(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let modelDir = URL(fileURLWithPath: modelPath)
    let reps = Int(env["BENCH_ROWCOST_REPS"] ?? "9") ?? 9
    let rowsList = (env["BENCH_ROWCOST_ROWS"] ?? "1,2,3,4,5,6,8").split(separator: ",").compactMap {
        Int($0)
    }
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader(),
        loadConfiguration: LoadConfiguration(nativeMTP: env["BENCH_ROWCOST_NOMTP"] != "1")
    ).0
    guard let model = context.model as? any NativeMTPModel else {
        print("[ROWCOST] model is not a NativeMTPModel")
        return
    }
    if env["BENCH_ROWCOST_LANE"] == "1" {
        // The DFlash2 target lane install (VMLX_LANE_QMM_TILED=1 selects the tiled layout).
        LaneQMM.installForDFlash2Target(model)
    }
    if env["BENCH_PARTS"] == "1" {
        for line in Qwen35DFlash2PartsProbe.run(
            model: model, rowsList: rowsList, reps: Int(env["BENCH_PARTS_REPS"] ?? "15") ?? 15)
        {
            print(line)
        }
        return
    }
    let text = String(
        repeating: "The history of printing presses in Europe changed literacy and trade. ",
        count: 40)
    var input = UserInput(prompt: text)
    input.additionalContext = ["enable_thinking": false]
    let prepared = try await context.processor.prepare(input: input)
    let prompt = prepared.text.tokens.reshaped(1, -1)
    let base = model.newCache(parameters: GenerateParameters())
    let pre = model.nativeBackboneForward(prompt, cache: base)
    MLX.eval(pre.logits, pre.hiddenStates)
    MLX.eval(base)
    print("[ROWCOST] model=\(modelDir.lastPathComponent) prompt=\(prompt.dim(1)) tokens")

    if env["BENCH_ROWCOST_COMPARE_TREE"] == "1",
        let treeModel = model as? DFlash2TreeVerifyModel,
        let target = model as? any DFlash2Target
    {
        // Numeric diagnosis only: compare each tree row with a single-row forward over its
        // exact ancestor path. No generation, sampler override or width-policy change.
        for rows in rowsList {
            for branching in [false, true] {
                let tokens = (0 ..< rows).map { 1000 + $0 * 7 }
                let parents = (0 ..< rows).map { $0 == 0 ? -1 : (branching ? ($0 - 1) / 2 : $0 - 1) }
                let scope = DFlash2TreeScope(plan: DFlash2TreePlan(tokens: tokens, parents: parents))
                let treeCache = base.map { $0.copy() }
                let capture = env["BENCH_ROWCOST_TREE_LAYERS"] == "1"
                Qwen4ExpRowExactCapture.partLayer = Int(env["BENCH_ROWCOST_TREE_LAYER"] ?? "0") ?? 0
                Qwen4ExpRowExactCapture.layers = []; Qwen4ExpRowExactCapture.parts = []
                Qwen4ExpRowExactCapture.enabled = capture
                let (treeLogits, _) = treeModel.dflash2TreeForward(
                    MLXArray(tokens.map(Int32.init)).reshaped(1, rows), cache: treeCache,
                    captureLayerIDs: [1, 7], scope: scope)
                MLX.eval(treeLogits)
                let treeLayers = Qwen4ExpRowExactCapture.layers
                let treeParts = Qwen4ExpRowExactCapture.parts
                MLX.eval(treeLayers)
                Qwen4ExpRowExactCapture.enabled = false
                var exact = 0, matching = 0
                var worst: Float = 0
                var leafReferenceCache: [KVCache] = []
                for row in 0 ..< rows {
                    var path: [Int] = [], at = row
                    while at >= 0 { path.append(at); at = parents[at] }
                    let referenceCache = base.map { $0.copy() }
                    var reference: MLXArray?
                    Qwen4ExpRowExactCapture.layers = []; Qwen4ExpRowExactCapture.parts = []
                    Qwen4ExpRowExactCapture.enabled = capture && row == 0
                    for node in path.reversed() {
                        let (logits, _) = target.callAsFunction(
                            MLXArray([Int32(tokens[node])]).reshaped(1, 1), cache: referenceCache,
                            captureLayerIDs: [1, 7], recordPrefixCommitStates: false)
                        MLX.eval(logits)
                        reference = logits[0, -1].asType(.float32)
                    }
                    if capture && row == 0 {
                        for (layer, pair) in zip(treeLayers, Qwen4ExpRowExactCapture.layers).enumerated() {
                            let difference = abs(pair.0[0, 0].asType(.float32) - pair.1[0, 0].asType(.float32)).max().item(Float.self)
                            print("[TREE-LAYER] layer=\(layer) row0diff=\(difference)")
                        }
                        for (name, expected) in Qwen4ExpRowExactCapture.parts {
                            if let got = treeParts.first(where: { $0.0 == name })?.1 {
                                let difference = abs(got[0, 0].asType(.float32) - expected[0, 0].asType(.float32)).max().item(Float.self)
                                print("[TREE-PART] name=\(name) row0diff=\(difference)")
                            }
                        }
                    }
                    Qwen4ExpRowExactCapture.enabled = false
                    if row == rows - 1 { leafReferenceCache = referenceCache }
                    let got = treeLogits[0, row].asType(.float32)
                    let difference = abs(got - reference!).max().item(Float.self)
                    if difference == 0 { exact += 1 }
                    if argMax(got).item(Int.self) == argMax(reference!).item(Int.self) { matching += 1 }
                    worst = max(worst, difference)
                }
                print("[TREE-PARITY] rows=\(rows) branching=\(branching) exact=\(exact) argmax=\(matching) maxdiff=\(worst)")
                let committed = treeModel.dflash2CommitTree(cache: treeCache, scope: scope, keptRows: scope.plan.paths[rows - 1])
                let next = MLXArray([Int32(2048)]).reshaped(1, 1)
                let (committedLogits, _) = target.callAsFunction(next, cache: treeCache, captureLayerIDs: [], recordPrefixCommitStates: false)
                let (referenceLogits, _) = target.callAsFunction(next, cache: leafReferenceCache, captureLayerIDs: [], recordPrefixCommitStates: false)
                let commitDiff = abs(committedLogits.asType(.float32) - referenceLogits.asType(.float32)).max().item(Float.self)
                print("[TREE-COMMIT] rows=\(rows) branching=\(branching) committed=\(committed) nextLogitDiff=\(commitDiff)")
            }
        }
        return
    }

    for mode in (env["BENCH_ROWCOST_MODES"] ?? "exact,plain").split(separator: ",").map(String.init)
    {
        for rows in rowsList {
            var samples: [Double] = []
            var builds: [Double] = []
            NativeMTPPhaseDiagnostics.reset()
            for rep in 0 ..< (reps + 3) {
                if rep == 3 { NativeMTPPhaseDiagnostics.reset() }
                let cache = base.map { $0.copy() }
                MLX.eval(cache)
                let ids = MLXArray(
                    (0 ..< rows).map { Int32(1000 + (($0 * 7919 + rep * 31) % 20000)) }
                )
                .reshaped(1, rows)
                let t0 = CFAbsoluteTimeGetCurrent()
                let out: NativeMTPForwardResult
                if mode == "tree", let treeModel = model as? DFlash2TreeVerifyModel {
                    let plan = DFlash2TreePlan(
                        tokens: (0 ..< rows).map { 1000 + $0 }, parents: Array(-1 ..< (rows - 1)))
                    let scope = DFlash2TreeScope(plan: plan)
                    let (logits, captured) = treeModel.dflash2TreeForward(
                        ids, cache: cache, captureLayerIDs: [5, 19, 33, 47, 61], scope: scope)
                    out = NativeMTPForwardResult(
                        logits: logits, hiddenStates: captured.values.first ?? logits)
                } else if mode == "exact" && rows > 1 {
                    // The iterator's actual verify mode (NativeMTPTokenIterator.verifyCycle, staged path).
                    out = NativeMTPVerifierStatePolicy.withVerifierMode("input_capture_staged") {
                        model.nativeBackboneMTPVerifyForward(ids, cache: cache)
                    }
                } else {
                    out = model.nativeBackboneForward(ids, cache: cache)
                }
                let built = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                MLX.eval(out.logits, out.hiddenStates)
                let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                if rep >= 3 {
                    samples.append(ms)
                    builds.append(built)
                }
            }
            samples.sort()
            builds.sort()
            if NativeMTPPhaseDiagnostics.enabled {
                // Per-phase ms per forward (phases eval in isolation: serialized, no overlap).
                let snap = NativeMTPPhaseDiagnostics.snapshot(reset: true)
                let perForward = snap.seconds.sorted { $0.key < $1.key }.map { key, sec in
                    String(format: "%@=%.2f", key, sec * 1000 / Double(reps))
                }.joined(separator: " ")
                print("[ROWCOST_PHASES] mode=\(mode) rows=\(rows) \(perForward)")
            }
            print(
                String(
                    format: "[ROWCOST] mode=%@ rows=%d ms=%.2f min=%.2f build=%.2f", mode, rows,
                    samples[samples.count / 2], samples[0], builds[builds.count / 2]))
        }
    }
}
