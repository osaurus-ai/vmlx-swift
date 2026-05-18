// Ling/Bailing real-bundle smoke tests. These are gated on a local bundle so
// normal CI can skip them, but when the bundle is present they exercise the
// production processor path rather than raw tokenizer rendering.

import Foundation
import BenchmarkHelpers
@testable import MLXHuggingFace
@preconcurrency import VMLXTokenizers
import MLXLMCommon
import MLXLLM
import Testing

@Suite("Ling JANGTQ2 smoke", .serialized)
struct LingSmokeJANGTQ2Tests {
    static let bundlePath: String = {
        if let override = ProcessInfo.processInfo.environment["VMLINUX_LING_JANGTQ2_BUNDLE"],
           !override.isEmpty {
            return override
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("models/dealign.ai/Ling-2.6-flash-JANGTQ2-CRACK")
            .path
    }()

    static let bundlePresent = FileManager.default.fileExists(
        atPath: "\(bundlePath)/config.json")

    @Test("production processor maps enable_thinking to Bailing directives",
          .enabled(if: LingSmokeJANGTQ2Tests.bundlePresent))
    func processorThinkingToggleUsesBailingDirectives() async throws {
        let url = URL(fileURLWithPath: Self.bundlePath)
        let context = try await MLXLMCommon.loadModel(
            from: url, using: #huggingFaceTokenizerLoader())

        func renderedPrompt(_ additionalContext: [String: any Sendable]?) async throws -> String {
            let input = try await context.processor.prepare(input: UserInput(
                chat: [
                    .system("You are concise."),
                    .user("Say OK."),
                ],
                additionalContext: additionalContext))
            let tokenIds = input.text.tokens.reshaped(-1).asArray(Int.self)
            return context.tokenizer.decode(tokenIds: tokenIds, skipSpecialTokens: false)
        }

        let defaultPrompt = try await renderedPrompt(nil)
        #expect(defaultPrompt.contains("detailed thinking off"))
        #expect(!defaultPrompt.contains("detailed thinking on"))

        let thinkingOffPrompt = try await renderedPrompt(["enable_thinking": false])
        #expect(thinkingOffPrompt.contains("detailed thinking off"))
        #expect(!thinkingOffPrompt.contains("detailed thinking on"))

        let thinkingOnPrompt = try await renderedPrompt(["enable_thinking": true])
        #expect(thinkingOnPrompt.contains("detailed thinking on"))
        #expect(!thinkingOnPrompt.contains("detailed thinking off"))
    }
}
