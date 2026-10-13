// BENCH_GLMCOST=1 — target forward cost by row count for models outside NativeMTPModel (GLM-5.3, Naive-N0.5), the
// twin of BENCH_ROWCOST. Loads like the app, prefills a BENCH_GLMCOST_CTX-token prompt (default 600) in 512-token
// chunks, then for each row count R times one forward of R new tokens from a COPY of the prefilled cache, median of
// BENCH_GLMCOST_REPS (default 7) after 2 warm-ups. GLM-5.3 runs the DFlash 2 verify entry (capture of the drafter's
// tap layers, input-capture verifier mode for R > 1; BENCH_GLMCOST_MODE=plain drops the mode).
// Prints one line per R: "[GLMCOST] rows=R ms=… min=…".

import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import VMLXTokenizers

func runGlm5VerifyCostBench(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let modelDir = URL(fileURLWithPath: modelPath)
    let reps = Int(env["BENCH_GLMCOST_REPS"] ?? "7") ?? 7
    let rowsList = (env["BENCH_GLMCOST_ROWS"] ?? "1,2,4,5,8,16").split(separator: ",").compactMap { Int($0) }
    let promptLen = Int(env["BENCH_GLMCOST_CTX"] ?? "600") ?? 600
    let capture: Set<Int> = [5, 14, 24, 33, 42]
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader(), loadConfiguration: .default
    ).0
    let anyModel = context.model
    let glm = anyModel as? Glm5Next
    func forward(_ ids: MLXArray, _ cache: [KVCache]) -> MLXArray {
        guard let glm else { return anyModel(ids, cache: cache) }
        let run = { () -> MLXArray in
            let out = glm(ids, cache: cache, captureLayerIDs: capture)
            MLX.eval(Array(out.capturedHiddenStates.values))
            return out.logits
        }
        if ids.dim(1) > 1, env["BENCH_GLMCOST_MODE"] != "plain" {
            return NativeMTPVerifierStatePolicy.withVerifierMode("input_capture", operation: run)
        }
        return run()
    }
    let prompt = MLXArray((0 ..< promptLen).map { Int32(1000 + (($0 * 7919) % 30000)) }).reshaped(1, promptLen)
    let base = anyModel.newCache(parameters: GenerateParameters())
    var start = 0
    while start < promptLen {
        let stop = min(start + 512, promptLen)
        MLX.eval(forward(prompt[0..., start ..< stop], base))
        MLX.eval(base)
        start = stop
    }
    print("[GLMCOST] model=\(modelDir.lastPathComponent) ctx=\(promptLen)")
    for rows in rowsList {
        var samples: [Double] = []
        for rep in 0 ..< (reps + 2) {
            let cache = base.map { $0.copy() }
            MLX.eval(cache)
            let ids = MLXArray((0 ..< rows).map { Int32(1000 + (($0 * 7919 + rep * 31) % 20000)) })
                .reshaped(1, rows)
            let t0 = CFAbsoluteTimeGetCurrent()
            MLX.eval(forward(ids, cache))
            if rep >= 2 { samples.append((CFAbsoluteTimeGetCurrent() - t0) * 1000) }
        }
        samples.sort()
        print(String(format: "[GLMCOST] rows=%d ms=%.2f min=%.2f", rows, samples[samples.count / 2], samples[0]))
    }
}
