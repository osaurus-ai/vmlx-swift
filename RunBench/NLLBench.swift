import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import VMLXTokenizers

/// BENCH_NLL_FILE=<text file>: mean next-token negative log-likelihood of the
/// first BENCH_NLL_TOKENS tokens (default 2048), prefilled through the model's
/// own cache in BENCH_PREFILL_STEP chunks (default 512). Teacher-forced, so it
/// scores the model's numerics directly: two builds (or two kernel paths) that
/// should compute the same function must give the same NLL, and a routing or
/// kernel error shows up as a higher NLL on ordinary text.
enum NLLBench {
    static func run(modelDir: URL, file: String, maxTokens: Int, chunk: Int) async throws {
        let loadStart = CFAbsoluteTimeGetCurrent()
        let loaded = try await MLXLMCommon.loadModel(
            from: modelDir,
            using: #huggingFaceTokenizerLoader(),
            loadConfiguration: .osaurusProduction)
        let context = loaded.0
        print(String(format: "NLL load: %.2fs model=%@", CFAbsoluteTimeGetCurrent() - loadStart,
            String(describing: type(of: context.model))))

        let text = try String(contentsOfFile: file, encoding: .utf8)
        let tokens = Array(context.tokenizer.encode(text: text).prefix(maxTokens))
        guard tokens.count > 1 else {
            print("NLL INVALID: fewer than 2 tokens")
            return
        }
        let cache = context.model.newCache(parameters: nil)
        var total: Double = 0
        var counted = 0
        let step = max(1, chunk)
        let t0 = CFAbsoluteTimeGetCurrent()
        var start = 0
        while start < tokens.count - 1 {
            let end = min(start + step, tokens.count - 1)
            let input = MLXArray(tokens[start ..< end].map { Int32($0) })[.newAxis, .ellipsis]
            let logits = context.model(LMInput.Text(tokens: input), cache: cache, state: nil)
                .logits.asType(.float32)
            let targets = MLXArray(tokens[(start + 1) ... end].map { Int32($0) })
                .reshaped(1, end - start, 1)
            let picked = takeAlong(logits, targets, axis: -1).squeezed(axis: -1)
            let nll = (logSumExp(logits, axis: -1) - picked).sum()
            eval(nll)
            total += Double(nll.item(Float.self))
            counted += end - start
            start = end
        }
        let mean = total / Double(counted)
        print(String(format: "NLL tokens=%d mean=%.6f ppl=%.4f chunk=%d ms=%.0f",
            counted, mean, exp(mean), step, (CFAbsoluteTimeGetCurrent() - t0) * 1000))
    }
}
