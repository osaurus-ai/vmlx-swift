// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// A swallowed MLX C++ error inside a decode step (the host installs a
/// non-fatal MLX error handler, as osaurus does) must finish the request
/// with a `GenerationFailure` instead of emitting the unrealized token and
/// decoding on. Sentry APPLE-MACOS-1ZE / 2PW / 1X5 crashed on exactly the
/// host read this iterator performs; with the MLX-layer fix that read returns
/// zeros, and this loop-level check turns it into a surfaced error.
@Suite("generate loop surfaces swallowed MLX errors", .serialized)
struct GenerateLoopMLXErrorTests {

    struct PieceTokenizer: MLXLMCommon.Tokenizer {
        let pieces: [String]
        var vocabularySize: Int { pieces.count }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map { pieces.indices.contains($0) ? pieces[$0] : "" }.joined()
        }
        func convertTokenToId(_ token: String) -> Int? { pieces.firstIndex(of: token) }
        func convertIdToToken(_ id: Int) -> String? {
            pieces.indices.contains(id) ? pieces[id] : nil
        }
        var bosToken: String? = nil
        var eosToken: String? = nil
        var eosTokenId: Int? { nil }
        var unknownToken: String? = nil
        var unknownTokenId: Int? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    /// Yields `tokens` in order. On step `failAt` it performs the same host
    /// read a model forward does (`asArray` of a sampled token) on an array
    /// whose graph fails to evaluate, then returns whatever it read.
    struct FailingStepIterator: TokenIteratorProtocol {
        let tokens: [Int]
        let failAt: Int
        var index = 0

        var maxTokens: Int? { nil }
        var tokenCount: Int { index }
        var promptPrefillTime: TimeInterval { 0 }

        mutating func next() -> Int? {
            guard index < tokens.count else { return nil }
            defer { index += 1 }
            if index == failAt {
                let kernel = MLXFast.metalKernel(
                    name: "vmlxGenerateLoopBrokenKernel", inputNames: ["a"], outputNames: ["out"],
                    source: "this is not valid metal source;")
                let failing = kernel(
                    [MLXArray.zeros([1], type: Float.self)], grid: (1, 1, 1), threadGroup: (1, 1, 1),
                    outputShapes: [[1]], outputDTypes: [.float32])[0]
                let sampled = (failing + MLXArray(Float(tokens[index]))).asType(.int32)
                return Int(sampled.asArray(Int32.self)[0])
            }
            return tokens[index]
        }
    }

    final class MessageBox: @unchecked Sendable {
        private let lock = NSLock()
        private var _messages: [String] = []
        var messages: [String] { lock.withLock { _messages } }
        func append(_ m: String) { lock.withLock { _messages.append(m) } }
    }

    @Test("an MLX error during a decode step ends the stream with a decoding GenerationFailure")
    func decodeErrorFinishesRequest() async {
        let box = MessageBox()
        let tokenizer = PieceTokenizer(pieces: ["one ", "two ", "three ", "four "])
        let iterator = FailingStepIterator(tokens: [0, 1, 2, 3], failAt: 2)

        // The generation Task is created inside this scope and inherits the
        // task-local handler, exactly like a host that scopes MLX errors.
        let (stream, task) = withErrorHandler({ box.append($0) }) {
            generateTask(
                promptTokenCount: 1,
                modelConfiguration: ModelConfiguration(id: "vmlx/generate-loop-mlx-error-stub"),
                tokenizer: tokenizer,
                iterator: iterator)
        }

        var text = ""
        var info: GenerateCompletionInfo?
        for await event in stream {
            switch event {
            case .chunk(let chunk): text += chunk
            case .info(let completion): info = completion
            default: break
            }
        }
        await task.value

        let failure = info?.generationFailure
        #expect(failure?.stage == .decoding)
        #expect(failure?.cause.contains("MLX error during decode") == true)
        #expect(info?.stopReason == .cancelled)
        // Tokens before the failing step were emitted; the failing step's
        // token and everything after it were not.
        #expect(text == "one two ")
        #expect(info?.generationTokenCount == 2)
        #expect(!box.messages.isEmpty)
    }

    @Test("a stream with no MLX error completes normally (control)")
    func healthyStreamHasNoFailure() async {
        let tokenizer = PieceTokenizer(pieces: ["one ", "two ", "three ", "four "])
        let iterator = FailingStepIterator(tokens: [0, 1, 2, 3], failAt: -1)
        let (stream, task) = generateTask(
            promptTokenCount: 1,
            modelConfiguration: ModelConfiguration(id: "vmlx/generate-loop-healthy-stub"),
            tokenizer: tokenizer,
            iterator: iterator)

        var text = ""
        var info: GenerateCompletionInfo?
        for await event in stream {
            switch event {
            case .chunk(let chunk): text += chunk
            case .info(let completion): info = completion
            default: break
            }
        }
        await task.value

        #expect(info?.generationFailure == nil)
        #expect(text == "one two three four ")
        #expect(info?.generationTokenCount == 4)
    }
}
