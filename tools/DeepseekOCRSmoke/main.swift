// DeepSeek-OCR / Unlimited-OCR engine correctness smoke.
//
// Loads the DeepseekOCR VLM via VLMModelFactory, runs the processor + model
// prepare + a greedy generate on a test image, and prints the decoded OCR
// text so it can be diffed against the PyTorch ground truth.
//
// Usage:
//   DSOCR_MODEL=/tmp/ocr_models/unlimited-ocr-8bit-mlx \
//   DSOCR_IMAGE=/tmp/ocr_test.png \
//   DSOCR_PROMPT="Free OCR." \
//   DSOCR_MAX_TOKENS=128 \
//   DSOCR_DUMP=1 \
//   swift run DeepseekOCRSmoke
//
// Requires the MLX metallib next to the executable.

import CoreImage
import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
@preconcurrency import VMLXTokenizers

@main
struct DeepseekOCRSmoke {
    static func main() async throws {
        setvbuf(stdout, nil, _IONBF, 0)
        let env = ProcessInfo.processInfo.environment

        let modelPath = env["DSOCR_MODEL"] ?? "/tmp/ocr_models/unlimited-ocr-8bit-mlx"
        let imagePath = env["DSOCR_IMAGE"] ?? "/tmp/ocr_test.png"
        let prompt = env["DSOCR_PROMPT"] ?? "Free OCR."
        let maxTokens = max(1, Int(env["DSOCR_MAX_TOKENS"] ?? "128") ?? 128)

        let modelDir = URL(fileURLWithPath: modelPath)
        let imageURL = URL(fileURLWithPath: imagePath)
        guard FileManager.default.fileExists(atPath: imageURL.path) else {
            fputs("image not found: \(imageURL.path)\n", stderr)
            exit(1)
        }

        print("[dsocr] loading \(modelDir.lastPathComponent) ...")
        let loadStart = CFAbsoluteTimeGetCurrent()
        let context = try await MLXLMCommon.loadModel(
            from: modelDir, using: #huggingFaceTokenizerLoader())
        print(String(
            format: "[dsocr] loaded in %.1fs (model type: %@)",
            CFAbsoluteTimeGetCurrent() - loadStart,
            String(describing: type(of: context.model))))

        // Build a UserInput with one image + the OCR prompt.
        var userInput = UserInput(
            prompt: .text(prompt),
            images: [.url(imageURL)])
        userInput.additionalContext = ["enable_thinking": false]

        let prepareStart = CFAbsoluteTimeGetCurrent()
        let lmInput = try await context.processor.prepare(input: userInput)
        let promptTokens = lmInput.text.tokens.dim(-1)
        print(String(
            format: "[dsocr] processor.prepare: %.0f ms, prompt tokens: %d, image: %@",
            (CFAbsoluteTimeGetCurrent() - prepareStart) * 1000,
            promptTokens,
            lmInput.image.map { "pixels \($0.pixels.shape), frames \($0.frames?.count ?? 0)" }
                ?? "nil"))

        if let img = lmInput.image {
            let p = img.pixels.asType(.float32)
            print(String(
                format: "[dsocr] pixels stat: shape=%@ sum=%.3f mean=%.5f min=%.4f max=%.4f",
                "\(img.pixels.shape)",
                p.sum().item(Float.self), p.mean().item(Float.self),
                p.min().item(Float.self), p.max().item(Float.self)))
        }

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
        // DSOCR_NO_STOP=1 keeps decoding past EOS (useful because, on the bare
        // "Free OCR." prompt + crop path, the model can emit a single spurious
        // leading EOS before the real grounding output — the official mitigates
        // this with no_repeat_ngram_size; the recognized text after it matches
        // the ground truth either way).
        let stopOnEos = env["DSOCR_NO_STOP"] != "1"
        for token in iterator {
            if stopOnEos, token == eosTokenId || eosIds.contains(token) { break }
            tokenIds.append(token)
            if tokenIds.count >= maxTokens { break }
        }
        let genSeconds = CFAbsoluteTimeGetCurrent() - genStart
        let text = context.tokenizer.decode(tokenIds: tokenIds)
        print(String(
            format: "[dsocr] generated %d tokens in %.1fs (%.1f tok/s)",
            tokenIds.count, genSeconds,
            Double(tokenIds.count) / max(genSeconds, 0.001)))
        print("===== OCR OUTPUT =====")
        print(text)
        print("===== END OUTPUT =====")
        if tokenIds.isEmpty {
            fputs("[dsocr] FAIL: no tokens generated\n", stderr)
            exit(2)
        }
    }
}
