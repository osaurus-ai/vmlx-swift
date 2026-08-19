// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Cross-version output probe.
//
// Greedy-decodes fixed prompts on a model that never touches DFlash 2 and
// prints a stable hash of the exact emitted text. Run the SAME file in a
// pre-change tree and a post-change tree: the hashes must match, or
// something outside the feature changed decode behaviour.
//
// This is the check that a filtered test suite cannot give you — passing
// assertions only prove the model still answers, not that it answers
// IDENTICALLY.
//
//   VMLX_PROBE_TARGET=$HOME/models/JANGQ-AI/Ornith-1.5-9B-JANG_4D \
//   swift test --filter CrossVersionOutputProbeTests

import CoreImage
import CryptoKit
import Foundation
import MLX
import XCTest
@preconcurrency import VMLXTokenizers
@testable import MLXHuggingFace
@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

final class CrossVersionOutputProbeTests: XCTestCase {

    private static var targetPath: String? {
        if let p = ProcessInfo.processInfo.environment["VMLX_PROBE_TARGET"] { return p }
        let p = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("models/JANGQ-AI/Ornith-1.5-9B-JANG_4D").path
        return FileManager.default.fileExists(atPath: p) ? p : nil
    }

    private func hash(_ s: String) -> String {
        SHA256.hash(data: Data(s.utf8)).map { String(format: "%02x", $0) }.joined()
            .prefix(16).description
    }

    private static func solidImage(_ color: CIColor, size: Int = 112) -> CIImage {
        let f = CIFilter(name: "CIConstantColorGenerator")!
        f.setValue(color, forKey: "inputColor")
        return f.outputImage!.cropped(to: CGRect(x: 0, y: 0, width: size, height: size))
    }

    func testDeterministicOutputHashes() async throws {
        guard let targetPath = Self.targetPath else {
            throw XCTSkip("Set VMLX_PROBE_TARGET")
        }
        let (ctx, _) = try await MLXLMCommon.loadModel(
            from: URL(fileURLWithPath: targetPath),
            using: #huggingFaceTokenizerLoader(),
            loadConfiguration: LoadConfiguration.default)

        func run(
            _ chat: [Chat.Message], thinking: Bool, images: [UserInput.Image] = [],
            maxTokens: Int
        ) async throws -> String {
            var ui = UserInput(chat: chat)
            if !images.isEmpty {
                ui = UserInput(chat: chat.map { m in
                    m.role == .user ? .user(m.content, images: images) : m
                })
            }
            ui.additionalContext = ["enable_thinking": thinking]
            let input = try await ctx.processor.prepare(input: ui)
            var p = GenerateParameters(maxTokens: maxTokens, temperature: 0)
            p.prefillStepSize = 1024
            p.topP = 1
            p.topK = 0
            p.minP = 0
            var out = ""
            for await item in try MLXLMCommon.generate(
                input: input, parameters: p, context: ctx)
            {
                switch item {
                case .chunk(let c): out += c
                case .reasoning(let c): out += c
                default: break
                }
            }
            return out
        }

        // 1. Plain decode, no reasoning — exercises the ordinary hybrid path.
        let plain = try await run(
            [.user("List the first five prime numbers and nothing else.")],
            thinking: false, maxTokens: 64)
        // 2. Reasoning path.
        let think = try await run(
            [.user("Why is 9 not a prime number? Answer in one sentence.")],
            thinking: true, maxTokens: 256)
        // 3. Vision path — the VLM forward this branch edited.
        let vision = try await run(
            [.user("What colour is this image? One word.")], thinking: false,
            images: [.ciImage(Self.solidImage(.red))], maxTokens: 24)
        // 4. Multiturn — exercises cache reuse and the snapshot/copy path.
        let history: [Chat.Message] = [
            .user("Remember the code word is Kestrel. Reply OK."),
            .assistant("OK"),
        ]
        let multi = try await run(
            history + [.user("What is the code word?")], thinking: false, maxTokens: 32)

        print("[probe] plain   \(hash(plain))  len=\(plain.count)")
        print("[probe] think   \(hash(think))  len=\(think.count)")
        print("[probe] vision  \(hash(vision)) len=\(vision.count)")
        print("[probe] multi   \(hash(multi))  len=\(multi.count)")
        print("[probe] COMBINED \(hash(plain + think + vision + multi))")

        XCTAssertFalse(plain.isEmpty)
        XCTAssertFalse(think.isEmpty)
        XCTAssertFalse(vision.isEmpty)
        XCTAssertFalse(multi.isEmpty)
    }
}
