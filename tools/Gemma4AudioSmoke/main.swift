// Gemma4 E-series audio tower runtime smoke.
//
// Loads a Gemma 4 E2B/E4B bundle (conformer `audio_tower` + mel
// `Gemma4AudioFeatureExtractor`), feeds a wav file plus a text prompt
// through the full chat-template → mel → tower → embed_audio →
// decode pipeline, and prints the generated text.
//
// Usage:
//   GEMMA4_SMOKE_MODEL=/path/to/bundle \
//   GEMMA4_SMOKE_AUDIO=/path/to/audio.wav \
//   GEMMA4_SMOKE_PROMPT="What do you hear?" \
//   GEMMA4_SMOKE_MAX_TOKENS=64 \
//   swift run Gemma4AudioSmoke
//
// Requires the MLX metallib next to the executable —
// run scripts/prepare-mlx-metal.sh first.

import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
@preconcurrency import VMLXTokenizers

@main
struct Gemma4AudioSmoke {
    static func main() async throws {
        setvbuf(stdout, nil, _IONBF, 0)
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["GEMMA4_SMOKE_MODEL"], !modelPath.isEmpty else {
            fputs("Set GEMMA4_SMOKE_MODEL to a Gemma 4 E-series bundle path\n", stderr)
            exit(1)
        }
        guard let audioPath = env["GEMMA4_SMOKE_AUDIO"], !audioPath.isEmpty else {
            fputs("Set GEMMA4_SMOKE_AUDIO to a wav/audio file path\n", stderr)
            exit(1)
        }
        let prompt = env["GEMMA4_SMOKE_PROMPT"] ?? "What do you hear in this audio clip?"
        let maxTokens = max(1, Int(env["GEMMA4_SMOKE_MAX_TOKENS"] ?? "64") ?? 64)

        let modelDir = URL(fileURLWithPath: modelPath)
        let audioURL = URL(fileURLWithPath: audioPath)
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            fputs("audio file not found: \(audioURL.path)\n", stderr)
            exit(1)
        }

        print("[smoke] loading \(modelDir.lastPathComponent) ...")
        let loadStart = CFAbsoluteTimeGetCurrent()
        let context = try await MLXLMCommon.loadModel(
            from: modelDir, using: #huggingFaceTokenizerLoader())
        print(String(
            format: "[smoke] loaded %@ in %.1fs (model type: %@)",
            modelDir.lastPathComponent,
            CFAbsoluteTimeGetCurrent() - loadStart,
            String(describing: type(of: context.model))))

        let chat: [Chat.Message] = [
            .user(prompt, audios: [.url(audioURL)])
        ]
        var userInput = UserInput(chat: chat)
        userInput.additionalContext = ["enable_thinking": false]

        let prepareStart = CFAbsoluteTimeGetCurrent()
        let lmInput = try await context.processor.prepare(input: userInput)
        let promptTokens = lmInput.text.tokens.dim(-1)
        print(String(
            format: "[smoke] processor.prepare: %.0f ms, prompt tokens: %d, audio: %@",
            (CFAbsoluteTimeGetCurrent() - prepareStart) * 1000,
            promptTokens,
            lmInput.audio.map { "waveform shape \($0.waveform.shape), preEncoded: \($0.preEncodedEmbedding != nil)" }
                ?? "nil"))

        var parameters = GenerateParameters(
            generationConfig: context.configuration.generationDefaults)
        parameters.maxTokens = maxTokens
        parameters.temperature = 0.0

        let genStart = CFAbsoluteTimeGetCurrent()
        let iterator = try TokenIterator(
            input: lmInput, model: context.model, parameters: parameters)
        var tokenIds: [Int] = []
        let eosIds = Set(context.configuration.extraEOSTokens.compactMap {
            context.tokenizer.convertTokenToId($0)
        })
        let eosTokenId = context.tokenizer.eosTokenId
        for token in iterator {
            if token == eosTokenId || eosIds.contains(token) { break }
            tokenIds.append(token)
            if tokenIds.count >= maxTokens { break }
        }
        let genSeconds = CFAbsoluteTimeGetCurrent() - genStart
        let text = context.tokenizer.decode(tokenIds: tokenIds)
        print(String(
            format: "[smoke] generated %d tokens in %.1fs (%.1f tok/s)",
            tokenIds.count, genSeconds,
            Double(tokenIds.count) / max(genSeconds, 0.001)))
        print("[smoke] output: \(text)")
        if tokenIds.isEmpty {
            fputs("[smoke] FAIL: no tokens generated\n", stderr)
            exit(2)
        }
        print("[smoke] PASS")
    }
}
