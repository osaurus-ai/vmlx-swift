// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// A chat template writes control markers into the prompt as literal text
/// (`<｜Assistant｜>`, `<think>`, …). If the bundle's vocabulary never shipped one of them
/// there is no id to return, and DeepSeek-class byte-level BPE declares no unknown token
/// to stand in either — so `convertTokenToId` yields nil.
///
/// `encode` force-unwrapped that nil. It killed the app in production (Sentry
/// APPLE-MACOS-10S, 9 events on 0.22.0) from inside a prefix-cache probe that was merely
/// measuring the generation-prompt suffix against a dummy message. The user had typed
/// nothing unusual, and the caller already had a `try?` fallback it could never reach —
/// a Swift trap is not an error you can catch.
///
/// Exercising this against a real tokenizer needs a model bundle whose vocab is missing a
/// marker, which a unit test has no business shipping. So these pin the two properties
/// that make the crash impossible, at the source level — the same convention this repo
/// uses for the compiled-decode guards. They prove the shape of the fix, not the runtime
/// behaviour; the runtime behaviour is proven live, in the app.
@Suite("Tokenizer: unencodable tokens are reported, not fatal")
struct TokenizerUnencodableTokenTests {

    private static func source(_ path: String) throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXLMTests/
            .deletingLastPathComponent()  // Tests/
            .deletingLastPathComponent()  // repo root
        return try String(contentsOf: root.appendingPathComponent(path), encoding: .utf8)
    }

    private static let tokenizerPath =
        "Vendors/swift-transformers/Sources/Tokenizers/Tokenizer.swift"

    @Test("A failure to encode is expressible as an error at all")
    func unencodableTokenIsAnError() throws {
        let src = try Self.source(Self.tokenizerPath)

        // Without a case for it, the only way to signal "this cannot be encoded" is to
        // trap — which is exactly how the app died.
        #expect(src.contains("case unencodableToken(String)"))

        // And a throwing entry point that reports it rather than force-unwrapping.
        #expect(src.contains("public func encodeThrowing(text: String"))
        #expect(src.contains("throw TokenizerError.unencodableToken(token)"))
    }

    @Test("The throwing path does not quietly drop the token instead")
    func unencodableTokenIsNotSilentlyDropped() throws {
        let src = try Self.source(Self.tokenizerPath)

        guard let start = src.range(of: "public func encodeThrowing(text: String") else {
            Issue.record("encodeThrowing not found")
            return
        }
        let body = String(src[start.lowerBound...].prefix(600))

        // `compactMap` here would silently shorten the prompt and shift every position
        // after the missing token — a model that reads as working while attending to the
        // wrong context. Failing loudly is the only honest option.
        #expect(
            !body.contains("compactMap"),
            "dropping an unencodable token corrupts the prompt silently — throw instead"
        )
        #expect(body.contains("guard let id = model.convertTokenToId(token)"))
    }
}
