// BENCH_ROWEXACT=1 — teacher-forced row-exactness of the native-MTP verify forward (Swift twin of the
// vMLX Python tests/test_row_exact_verify.py idea). For one prompt:
//   1. AR reference: prefill, then N greedy single-token steps through nativeAutoregressiveBackboneForward
//      (the S1 path MTP's AR steps use); keep every step's logits.
//   2. For each verify width W (BENCH_ROWEXACT_WIDTHS, default 2,4,8): prefill afresh, then feed the SAME
//      token stream in windows of W through nativeBackboneMTPVerifyForward in the staged verifier mode the
//      iterator uses, commit each full window (commitStagedVerifiedBlock), and compare every row's logits
//      with the AR logits of the same position: bitwise equality, max |diff|, argmax agreement.
// A row-exact verifier must report bitwise=all for every W. Env: BENCH_ROWEXACT_TOKENS (default 96),
// BENCH_ROWEXACT_PROMPT (default: the speed probe's first prose prompt).

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import VMLXTokenizers

func runRowExactBench(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let modelDir = URL(fileURLWithPath: modelPath)
    let steps = Int(env["BENCH_ROWEXACT_TOKENS"] ?? "96") ?? 96
    let widths = (env["BENCH_ROWEXACT_WIDTHS"] ?? "2,4,8").split(separator: ",").compactMap {
        Int($0)
    }
    let promptText =
        env["BENCH_ROWEXACT_PROMPT"]
        ?? "Write a detailed essay of about 400 words on the history of the printing press and its effect on literacy in Europe."
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader(),
        loadConfiguration: LoadConfiguration(nativeMTP: true)
    ).0
    guard let model = context.model as? any NativeMTPModel,
        let staged = context.model as? any DFlash2StagedVerifyRollbackModel
    else {
        print("[ROWEXACT] model lacks native MTP / staged verify")
        return
    }
    // Match NativeMTPTokenIterator's AR dispatch: Qwen3.5 uses the ordinary
    // backbone, while families with a dedicated one-row path opt into it.
    func autoregressiveForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        if let ar = model as? any NativeMTPAutoregressiveBackboneModel {
            return ar.nativeAutoregressiveBackboneForward(inputs, cache: cache)
        }
        return model.nativeBackboneForward(inputs, cache: cache)
    }
    var input = UserInput(prompt: promptText)
    input.additionalContext = ["enable_thinking": false]
    let prompt = try await context.processor.prepare(input: input).text.tokens.reshaped(1, -1)

    func prefill() -> ([KVCache], Int) {
        let cache = model.newCache(parameters: GenerateParameters())
        let out = model.nativeBackboneForward(prompt, cache: cache)
        let first = argMax(out.logits[0, -1], axis: -1).item(Int.self)
        MLX.eval(cache)
        return (cache, first)
    }

    if env["BENCH_ROWEXACT_TIME"] == "1" {
        // Verify cost: median wall time of one AR step vs one staged verify forward of W rows, each
        // from a fresh copy of the same prefilled cache (copy evaluated before the clock starts).
        let reps = Int(env["BENCH_ROWEXACT_TIME_REPS"] ?? "15") ?? 15
        let (base, first) = prefill()
        var buildSamples = [Int: [Double]]()
        func once(_ ids: [Int32], verify: Bool) -> Double {
            let c = base.map { $0.copy() }
            MLX.eval(c)
            let input = MLXArray(ids).reshaped(1, ids.count)
            let t0 = CFAbsoluteTimeGetCurrent()
            let out =
                verify
                ? NativeMTPVerifierStatePolicy.withVerifierMode("input_capture_staged") {
                    model.nativeBackboneMTPVerifyForward(input, cache: c)
                }
                : autoregressiveForward(input, cache: c)
            // Graph construction ends here (early-submit layers may already be running on the GPU).
            buildSamples[verify ? ids.count : 0, default: []].append(CFAbsoluteTimeGetCurrent() - t0)
            MLX.eval(out.logits)
            return CFAbsoluteTimeGetCurrent() - t0
        }
        let ids = (0 ..< 8).map { Int32(first + $0 * 7) }
        // Warm every shape (kernel JIT + bitwise admission), then interleave AR and every width
        // within each repetition so clock drift and GPU contention hit all arms alike.
        _ = once([ids[0]], verify: false)
        for w in widths { _ = once(Array(ids.prefix(w)), verify: true) }
        var arSamples: [Double] = []
        var verifySamples = [Int: [Double]]()
        for _ in 0 ..< reps {
            arSamples.append(once([ids[0]], verify: false))
            for w in widths { verifySamples[w, default: []].append(once(Array(ids.prefix(w)), verify: true)) }
        }
        func median(_ v: [Double]) -> Double { let s = v.sorted(); return s[s.count / 2] * 1000 }
        let ar = median(arSamples)
        print(String(format: "[ROWEXACT-TIME] AR step %.2f ms (graph build %.2f ms)", ar, median(buildSamples[0] ?? [0])))
        for w in widths {
            let v = median(verifySamples[w] ?? [0])
            print(String(format: "[ROWEXACT-TIME] verify W=%d %.2f ms (%.2fx AR) (graph build %.2f ms)", w, v, v / ar,
                         median(buildSamples[w] ?? [0])))
        }
        if env["BENCH_ROWEXACT_TIME_ONLY"] == "1" { return }
    }

    // 1. AR reference.
    var (cache, token) = prefill()
    var stream: [Int] = [token]
    var arLogits: [MLXArray] = []
    for _ in 0 ..< steps {
        let tokenInput = MLXArray([Int32(token)]).reshaped(1, 1)
        // BENCH_ROWEXACT_AR=plain: nativeBackboneForward (S=1, no autoregressive flag) as the reference.
        let out =
            env["BENCH_ROWEXACT_AR"] == "plain"
            ? model.nativeBackboneForward(tokenInput, cache: cache)
            : autoregressiveForward(tokenInput, cache: cache)
        let row = out.logits[0, -1].asType(.float32)
        MLX.eval(row)
        arLogits.append(row)
        token = argMax(row, axis: -1).item(Int.self)
        stream.append(token)
    }
    if env["BENCH_ROWEXACT_NEMO_LAYERS"] == "1", let tappable = context.model as? NemotronHBlockOutputTappable {
        // Nemotron-H layer bisect: every block output of the AR step on t0 vs row 0 of the staged
        // verify [t0, t1] (both from the same prefilled state).
        func capture(_ run: () -> Void) -> [Int: MLXArray] {
            var outputs: [Int: MLXArray] = [:]
            let previous = tappable.blockOutputTap
            tappable.blockOutputTap = NemotronHBlockOutputTap(
                layerIndices: Set(0 ..< 512), record: { outputs[$0] = $1 })
            run()
            tappable.blockOutputTap = previous
            MLX.eval(Array(outputs.values))
            return outputs
        }
        var (c1, t0) = prefill()
        let arLayers = capture {
            let a = autoregressiveForward(MLXArray([Int32(t0)]).reshaped(1, 1), cache: c1)
            MLX.eval(a.logits)
        }
        (c1, _) = prefill()
        let vLayers = capture {
            let v = NativeMTPVerifierStatePolicy.withVerifierMode("input_capture_staged") {
                model.nativeBackboneMTPVerifyForward(
                    MLXArray([Int32(t0), Int32(stream[1])]).reshaped(1, 2), cache: c1)
            }
            MLX.eval(v.logits)
        }
        for layer in arLayers.keys.sorted() {
            guard let x = arLayers[layer], let y = vLayers[layer] else { continue }
            let d = abs(x[0, 0].asType(.float32) - y[0, 0].asType(.float32))
            print(String(format: "[ROWEXACT] nemo layer=%d maxdiff=%.4g nonzero=%d", layer,
                d.max().item(Float.self), (d .> 0).sum().item(Int.self)))
        }
    }
    if env["BENCH_ROWEXACT_LAYERS"] == "1" {
        // Layer bisect: AR step on t0 vs verify window [t0, t1], row 0, after every layer.
        var (c1, t0) = prefill()
        Qwen4ExpRowExactCapture.layers = []
        Qwen4ExpRowExactCapture.parts = []
        Qwen4ExpRowExactCapture.enabled = true
        let a = autoregressiveForward(
            MLXArray([Int32(t0)]).reshaped(1, 1), cache: c1)
        MLX.eval(a.logits)
        let arLayers = Qwen4ExpRowExactCapture.layers
        let arParts = Qwen4ExpRowExactCapture.parts
        MLX.eval(arLayers)
        Qwen4ExpRowExactCapture.enabled = false
        (c1, _) = prefill()
        Qwen4ExpRowExactCapture.layers = []
        Qwen4ExpRowExactCapture.parts = []
        Qwen4ExpRowExactCapture.enabled = true
        let v = NativeMTPVerifierStatePolicy.withVerifierMode("input_capture_staged") {
            model.nativeBackboneMTPVerifyForward(
                MLXArray([Int32(t0), Int32(stream[1])]).reshaped(1, 2), cache: c1)
        }
        MLX.eval(v.logits)
        let vLayers = Qwen4ExpRowExactCapture.layers
        let vParts = Qwen4ExpRowExactCapture.parts
        for ((name, x), (_, y)) in zip(arParts, vParts) {
            let xr = x.reshaped(x.dim(0), x.dim(1), -1)[0, 0].asType(.float32)
            let yr = y.reshaped(y.dim(0), y.dim(1), -1)[0, 0].asType(.float32)
            let d = abs(xr - yr)
            print(
                String(
                    format: "[ROWEXACT] layer0.%@ shape=%@/%@ dtype=%@/%@ maxdiff=%.4g nonzero=%d",
                    name,
                    x.shape.description, y.shape.description, String(describing: x.dtype),
                    String(describing: y.dtype), d.max().item(Float.self),
                    (d .> 0).sum().item(Int.self)))
        }
        MLX.eval(vLayers)
        Qwen4ExpRowExactCapture.enabled = false
        for (i, (x, y)) in zip(arLayers, vLayers).enumerated() {
            let d = abs(x[0, 0].asType(.float32) - y[0, 0].asType(.float32))
            print(
                String(
                    format: "[ROWEXACT] layer=%d maxdiff=%.4g nonzero=%d", i,
                    d.max().item(Float.self),
                    (d .> 0).sum().item(Int.self)))
        }
    }
    if let atRaw = env["BENCH_ROWEXACT_LAYERS_AT"], let at = Int(atRaw), at >= 0,
        at + 1 < stream.count
    {
        // Layer bisect at position `at`: AR-step 0..<at, copy the cache, then AR step `at` vs verify [t_at, t_at+1].
        var (c0, _) = prefill()
        for i in 0 ..< at {
            let o = autoregressiveForward(
                MLXArray([Int32(stream[i])]).reshaped(1, 1), cache: c0)
            MLX.eval(o.logits)
        }
        MLX.eval(c0)
        let cAR = c0.map { $0.copy() }
        let cV = c0.map { $0.copy() }
        MLX.eval(cAR)
        MLX.eval(cV)
        Qwen4ExpRowExactCapture.layers = []
        Qwen4ExpRowExactCapture.parts = []
        Qwen4ExpRowExactCapture.enabled = true
        let a = autoregressiveForward(
            MLXArray([Int32(stream[at])]).reshaped(1, 1), cache: cAR)
        MLX.eval(a.logits)
        let arLayers = Qwen4ExpRowExactCapture.layers
        Qwen4ExpRowExactCapture.layers = []
        let v = NativeMTPVerifierStatePolicy.withVerifierMode("input_capture_staged") {
            model.nativeBackboneMTPVerifyForward(
                MLXArray([Int32(stream[at]), Int32(stream[at + 1])]).reshaped(1, 2), cache: cV)
        }
        MLX.eval(v.logits)
        let vLayers = Qwen4ExpRowExactCapture.layers
        Qwen4ExpRowExactCapture.enabled = false
        let row = abs(a.logits[0, -1].asType(.float32) - v.logits[0, 0].asType(.float32)).max()
            .item(Float.self)
        print(String(format: "[ROWEXACT] at=%d logits_row0_maxdiff=%.4g", at, row))
        for (i, (x, y)) in zip(arLayers, vLayers).enumerated() {
            let d = abs(x[0, 0].asType(.float32) - y[0, 0].asType(.float32))
            let m = d.max().item(Float.self)
            if m > 0 {
                print(
                    String(
                        format: "[ROWEXACT] at=%d first_diff_layer=%d maxdiff=%.4g nonzero=%d", at,
                        i, m,
                        (d .> 0).sum().item(Int.self)))
                // Component bisect inside that layer, from the same pre-step state.
                Qwen4ExpRowExactCapture.partLayer = i
                let c1 = c0.map { $0.copy() }
                let c2 = c0.map { $0.copy() }
                MLX.eval(c1)
                MLX.eval(c2)
                Qwen4ExpRowExactCapture.parts = []
                Qwen4ExpRowExactCapture.enabled = true
                MLX.eval(
                    autoregressiveForward(
                        MLXArray([Int32(stream[at])]).reshaped(1, 1), cache: c1
                    ).logits)
                let ap = Qwen4ExpRowExactCapture.parts
                Qwen4ExpRowExactCapture.parts = []
                MLX.eval(
                    NativeMTPVerifierStatePolicy.withVerifierMode("input_capture_staged") {
                        model.nativeBackboneMTPVerifyForward(
                            MLXArray([Int32(stream[at]), Int32(stream[at + 1])]).reshaped(1, 2),
                            cache: c2)
                    }.logits)
                let vp = Qwen4ExpRowExactCapture.parts
                Qwen4ExpRowExactCapture.enabled = false
                Qwen4ExpRowExactCapture.partLayer = 0
                for ((name, x), (_, y)) in zip(ap, vp) {
                    let xr = x.reshaped(x.dim(0), x.dim(1), -1)[0, 0].asType(.float32)
                    let yr = y.reshaped(y.dim(0), y.dim(1), -1)[0, 0].asType(.float32)
                    let dd = abs(xr - yr)
                    print(
                        String(
                            format:
                                "[ROWEXACT] at=%d layer=%d %@ dtype=%@/%@ maxdiff=%.4g nonzero=%d",
                            at, i, name,
                            String(describing: x.dtype), String(describing: y.dtype),
                            dd.max().item(Float.self), (dd .> 0).sum().item(Int.self)))
                }
                break
            }
        }
    }
    if let atRaw = env["BENCH_ROWEXACT_ROW1_AT"], let at = Int(atRaw), at >= 0, at + 1 < stream.count {
        // Row-1 bisect: AR steps t_at then t_{at+1} vs verify row 1 of [t_at, t_{at+1}], every layer, then the
        // parts of the first differing layer.
        var (c0, _) = prefill()
        for i in 0 ..< at {
            MLX.eval(autoregressiveForward(
                MLXArray([Int32(stream[i])]).reshaped(1, 1), cache: c0).logits)
        }
        MLX.eval(c0)
        func arSecond(part: Int?) -> ([MLXArray], [(String, MLXArray)]) {
            let c = c0.map { $0.copy() }
            MLX.eval(c)
            MLX.eval(autoregressiveForward(
                MLXArray([Int32(stream[at])]).reshaped(1, 1), cache: c).logits)
            Qwen4ExpRowExactCapture.layers = []
            Qwen4ExpRowExactCapture.parts = []
            Qwen4ExpRowExactCapture.partLayer = part ?? -1
            Qwen4ExpRowExactCapture.enabled = true
            MLX.eval(autoregressiveForward(
                MLXArray([Int32(stream[at + 1])]).reshaped(1, 1), cache: c).logits)
            Qwen4ExpRowExactCapture.enabled = false
            return (Qwen4ExpRowExactCapture.layers, Qwen4ExpRowExactCapture.parts)
        }
        func verify(part: Int?) -> ([MLXArray], [(String, MLXArray)]) {
            let c = c0.map { $0.copy() }
            MLX.eval(c)
            Qwen4ExpRowExactCapture.layers = []
            Qwen4ExpRowExactCapture.parts = []
            Qwen4ExpRowExactCapture.partLayer = part ?? -1
            Qwen4ExpRowExactCapture.enabled = true
            MLX.eval(NativeMTPVerifierStatePolicy.withVerifierMode("input_capture_staged") {
                model.nativeBackboneMTPVerifyForward(
                    MLXArray([Int32(stream[at]), Int32(stream[at + 1])]).reshaped(1, 2), cache: c)
            }.logits)
            Qwen4ExpRowExactCapture.enabled = false
            return (Qwen4ExpRowExactCapture.layers, Qwen4ExpRowExactCapture.parts)
        }
        func last(_ x: MLXArray, row: Int) -> MLXArray {
            let flat = x.reshaped(x.dim(0), x.dim(1), -1)
            return flat[0, min(row, flat.dim(1) - 1)].asType(.float32)
        }
        let (aL, _) = arSecond(part: nil)
        let (vL, _) = verify(part: nil)
        MLX.eval(aL)
        MLX.eval(vL)
        var first: Int?
        for (i, (x, y)) in zip(aL, vL).enumerated() {
            let d = abs(last(x, row: 0) - last(y, row: 1))
            let m = d.max().item(Float.self)
            if m > 0, first == nil { first = i }
            if m > 0 || i == aL.count - 1 {
                print(String(format: "[ROWEXACT] row1 at=%d layer=%d maxdiff=%.4g nonzero=%d", at, i, m,
                             (d .> 0).sum().item(Int.self)))
            }
            if first != nil, i > first! + 1 { break }
        }
        if let first {
            let (_, aP) = arSecond(part: first)
            let (_, vP) = verify(part: first)
            for ((name, x), (_, y)) in zip(aP, vP) {
                let d = abs(last(x, row: 0) - last(y, row: 1))
                print(String(format: "[ROWEXACT] row1 at=%d layer=%d %@ maxdiff=%.4g nonzero=%d", at, first,
                             name, d.max().item(Float.self), (d .> 0).sum().item(Int.self)))
            }
        }
        Qwen4ExpRowExactCapture.partLayer = 0
    }
    if let dump = env["BENCH_ROWEXACT_DUMP"] {
        // The reference AR stream as text (compare with BENCH_SPEED_TEXT_OUT dumps of the real iterators).
        let text = context.tokenizer.decode(tokenIds: Array(stream.dropLast()))
        try text.write(toFile: dump, atomically: true, encoding: .utf8)
    }
    print(
        "[ROWEXACT] model=\(modelDir.lastPathComponent) prompt=\(prompt.dim(1)) steps=\(steps) ar=\(env["BENCH_ROWEXACT_AR"] ?? "native-autoregressive")"
    )

    // 2. Verify windows over the same stream.
    for w in widths {
        (cache, _) = prefill()
        var p = 0
        var bitwise = 0
        var compared = 0
        var argmaxMiss = 0
        var firstMismatch: Int?
        var maxDiff: Float = 0
        while p + w <= steps {
            let ids = MLXArray(stream[p ..< p + w].map { Int32($0) }).reshaped(1, w)
            let out = NativeMTPVerifierStatePolicy.withVerifierMode("input_capture_staged") {
                model.nativeBackboneMTPVerifyForward(ids, cache: cache)
            }
            let rows = out.logits[0].asType(.float32)
            MLX.eval(rows)
            guard staged.commitStagedVerifiedBlock(cache: cache, acceptedInputs: w, blockLength: w)
            else {
                print("[ROWEXACT] W=\(w) commit failed at \(p)")
                break
            }
            if p == 0 {
                print(
                    "[ROWEXACT] debug W=\(w) verify_logits=\(out.logits.dtype) \(out.logits.shape) ar_row=\(arLogits[0].shape)"
                )
                for j in 0 ..< w {
                    let d = abs(rows[j] - arLogits[j])
                    print(
                        String(
                            format:
                                "[ROWEXACT] debug row=%d maxdiff=%.4g meandiff=%.4g nonzero=%d argmax=%d ar_argmax=%d",
                            j, d.max().item(Float.self), d.mean().item(Float.self),
                            (d .> 0).sum().item(Int.self), argMax(rows[j], axis: -1).item(Int.self),
                            argMax(arLogits[j], axis: -1).item(Int.self)))
                }
            }
            for j in 0 ..< w {
                let ref = arLogits[p + j]
                let got = rows[j]
                let diff = abs(got - ref).max().item(Float.self)
                compared += 1
                if diff == 0 { bitwise += 1 } else if firstMismatch == nil { firstMismatch = p + j }
                maxDiff = max(maxDiff, diff)
                if argMax(got, axis: -1).item(Int.self) != stream[p + j + 1] { argmaxMiss += 1 }
            }
            p += w
        }
        print(
            String(
                format:
                    "[ROWEXACT] W=%d rows=%d bitwise=%d argmax_miss=%d max_abs_diff=%.4g first_mismatch=%@",
                w, compared, bitwise, argmaxMiss, maxDiff, firstMismatch.map(String.init) ?? "none")
        )
    }
}
