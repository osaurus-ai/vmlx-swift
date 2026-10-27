// BENCH_DSV41_LONG=1 — DeepSeek-V4.1 long-context needle retrieval and retained KV bytes on the production
// generation path (Swift twin of the Python campaign `kv_long.py`, DSV4.1 JANGHT campaign 2026-10-26).
//
// One chat prompt: a needle sentence, then BENCH_LONG_TOKENS (default 1048576) tokens of BENCH_LONG_FILE text
// (repeated to length), then the question; thinking off, greedy. TokenIterator runs the prefill (chunked, suffix
// pruning) and decodes up to 24 tokens. Run with VMLX_DSV41_KVBYTES=1 for the retained KV bytes after the prefill.
// Prints "[LONG] ..." lines: prompt tokens, time to first token (prefill rate), the answer, whether it holds the
// needle code, and MLX peak memory.

import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import VMLXTokenizers

func runDSV41LongContext(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let target = Int(env["BENCH_LONG_TOKENS"] ?? "1048576") ?? 1_048_576
    guard let file = env["BENCH_LONG_FILE"], let corpus = try? String(contentsOfFile: file, encoding: .utf8) else {
        throw NSError(domain: "BENCH_DSV41_LONG", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "set BENCH_LONG_FILE to a text corpus"])
    }
    let context = try await MLXLMCommon.loadModel(
        from: URL(fileURLWithPath: modelPath), using: #huggingFaceTokenizerLoader(),
        loadConfiguration: .osaurusProduction)
    // Osaurus caps MLX's buffer cache after every load (ModelRuntime.mlxCacheLimit: min(weights / 4, RAM / 8,
    // 8 GiB)). Uncapped, a 1M-token run reached a 114 GB peak footprint. BENCH_CACHE_LIMIT_GB overrides.
    let cacheLimitGiB = Double(env["BENCH_CACHE_LIMIT_GB"] ?? "8") ?? 8
    MLX.Memory.cacheLimit = Int(cacheLimitGiB * 1_073_741_824)
    print("[LONG] MLX cache limit \(cacheLimitGiB) GiB")
    let needle = "Remember this: the vault access code is 7341-ALPHA."
    var bodyIds = context.tokenizer.encode(text: corpus)
    while bodyIds.count < target { bodyIds += bodyIds }
    let body = context.tokenizer.decode(tokenIds: Array(bodyIds.prefix(target)))
    let text = needle + "\n\n" + body
        + "\n\nWhat is the vault access code from the start of this message? Reply with the code only."
    var userInput = UserInput(prompt: text)
    userInput.additionalContext = ["enable_thinking": false]
    let input = try await context.processor.prepare(input: userInput)
    print("[LONG] prompt tokens \(input.text.tokens.size)")
    MLX.Memory.peakMemory = 0

    var params = GenerateParameters(maxTokens: 24, temperature: 0)
    params.prefillStepSize = 512
    let t0 = CFAbsoluteTimeGetCurrent()
    var iterator = try TokenIterator(input: input, model: context.model, parameters: params)
    var tokens: [Int] = []
    var ttft = 0.0
    while let token = iterator.next() {
        if tokens.isEmpty { ttft = CFAbsoluteTimeGetCurrent() - t0 }
        tokens.append(token)
        // The bundle's EOS ids (jang_config chat.eos_token_ids) plus the configuration's.
        if context.configuration.eosTokenIds.contains(token) || [1, 128_803, 128_804, 128_828].contains(token) { break }
    }
    let answer = context.tokenizer.decode(tokenIds: tokens)
    let promptTokens = input.text.tokens.size
    print(String(
        format: "[LONG] ttft %.1f s, prefill %.1f tok/s, peak MLX memory %.2f GiB", ttft,
        Double(promptTokens) / max(ttft, 1e-9), Double(MLX.Memory.peakMemory) / 1_073_741_824))
    print("[LONG] answer \(answer.debugDescription) correct=\(answer.contains("7341"))")
}
