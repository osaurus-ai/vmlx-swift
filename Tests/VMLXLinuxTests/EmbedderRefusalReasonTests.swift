// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@testable import MLXEmbedders

/// An embedder that refuses a configuration on purpose says why. ModernBERT refuses an activation it
/// does not implement. Before this fix the loader showed only "Invalid data at 'hidden_activation'",
/// which reads like a corrupt file, not like an unsupported one.
struct EmbedderRefusalReasonTests {
    @Test("A configuration that ModernBERT refuses reports the reason through the real loader")
    func refusalReasonReachesTheLoaderError() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "embedder-refusal-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = #"{"model_type": "modernbert", "hidden_activation": "gelu_pytorch_tanh"}"#
        try Data(config.utf8).write(to: directory.appending(component: "config.json"))

        do {
            _ = try loadSynchronous(modelDirectory: directory, modelName: "refusal-test")
            Issue.record("the loader accepted an activation that ModernBERT does not implement")
        } catch let error as EmbedderError {
            let text = error.errorDescription ?? ""
            #expect(text.contains("hidden_activation"), "\(text)")
            #expect(text.contains("not \"gelu_pytorch_tanh\""), "the reason is missing: \(text)")
        }
    }

    @Test("A decoding error with no reason keeps the short message")
    func emptyReasonKeepsTheShortMessage() {
        struct Key: CodingKey {
            var stringValue: String
            var intValue: Int? { nil }
            init(stringValue: String) { self.stringValue = stringValue }
            init?(intValue: Int) { nil }
        }
        let error = EmbedderError.configurationDecodingError(
            "config.json", "refusal-test",
            .dataCorrupted(
                .init(codingPath: [Key(stringValue: "hidden_size")], debugDescription: "")))
        #expect(
            error.errorDescription
                == "Failed to parse config.json for model 'refusal-test': Invalid data at 'hidden_size'"
        )
    }
}
