// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@Suite("Tokenizer added-token regex focused guards")
struct TokenizerAddedTokenRegexFocusedTests {
    @Test("indexed placeholder added tokens are excluded from regex construction")
    func unusedPlaceholderAddedTokensAreExcludedFromRegexConstruction() throws {
        let source = try String(
            contentsOfFile: "Vendors/swift-transformers/Sources/Tokenizers/Tokenizer.swift",
            encoding: .utf8)

        #expect(source.contains("private static func isUnusedPlaceholderAddedToken"))
        #expect(source.contains("if Self.isUnusedPlaceholderAddedToken(content)"))
        #expect(source.contains("content.hasPrefix(\"<SPECIAL_\")"))
        #expect(source.contains("content.hasPrefix(\"<speechcodec_\")"))
        #expect(source.contains("content.hasPrefix(\"<audiocodec_\")"))
        #expect(source.contains("return nil"))
        #expect(source.contains("content[numberStart..<numberEnd].allSatisfy { $0.isNumber }"))
    }
}
