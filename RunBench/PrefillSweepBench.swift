// BENCH_PREFILL_SWEEP=1 — cold-prefill throughput by chunk size on one prompt (diagnostic only).
// Each repetition prefills the same prompt into a fresh cache once per step size (interleaved, so clock
// drift hits every step alike) through the generation path: `model.prepare(windowSize:)` for the chunked
// head, then the remainder forward that produces the first logits. Median per step.
// Env: BENCH_PREFILL_FILE (prompt text), BENCH_PREFILL_STEPS (default 512,1024,2048,4096),
// BENCH_PREFILL_REPS (default 3).

import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import VMLXTokenizers

func runPrefillSweepBench(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let steps = (env["BENCH_PREFILL_STEPS"] ?? "512,1024,2048,4096").split(separator: ",")
        .compactMap { Int($0) }
    let reps = Int(env["BENCH_PREFILL_REPS"] ?? "3") ?? 3
    let text = try String(contentsOfFile: env["BENCH_PREFILL_FILE"] ?? "", encoding: .utf8)
    let context = try await MLXLMCommon.loadModel(
        from: URL(fileURLWithPath: modelPath), using: #huggingFaceTokenizerLoader(),
        loadConfiguration: .osaurusProduction
    ).0
    var input = UserInput(prompt: text)
    input.additionalContext = ["enable_thinking": false]
    let prepared = try await context.processor.prepare(input: input)
    let tokens = prepared.text.tokens.reshaped(1, -1)
    let count = tokens.dim(1)
    func prefill(_ step: Int) throws -> Double {
        var params = GenerateParameters()
        params.prefillStepSize = step
        let cache = context.model.newCache(parameters: params)
        MLX.eval(cache)
        let t0 = CFAbsoluteTimeGetCurrent()
        var logits: MLXArray
        switch try context.model.prepare(
            LMInput(text: LMInput.Text(tokens: tokens)), cache: cache, windowSize: step)
        {
        case .tokens(let rest):
            logits = context.model(rest, cache: cache, state: nil).logits
        case .logits(let out):
            logits = out.logits
        }
        MLX.eval(logits)
        let seconds = CFAbsoluteTimeGetCurrent() - t0
        print(String(format: "[PREFILL] run step=%d %.2f s", step, seconds))
        fflush(stdout)
        return seconds
    }
    _ = try prefill(steps.first ?? 512)
    var samples = [Int: [Double]]()
    for _ in 0 ..< reps {
        for step in steps { samples[step, default: []].append(try prefill(step)) }
    }
    for step in steps {
        let s = (samples[step] ?? []).sorted()
        let med = s[s.count / 2]
        print(String(format: "[PREFILL] tokens=%d step=%d median=%.2f s (%.0f tok/s) runs=%@", count, step,
            med, Double(count) / med, s.map { String(format: "%.2f", $0) }.joined(separator: ",")))
    }
}
