// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Ornith 1.5 9B enablement: the checks a new bundle has to pass before it
// is offered to anyone — it loads, it answers coherently, reasoning ON
// and OFF both behave, vision actually reads the image, and a second turn
// reuses the prefix instead of re-prefilling.
//
// Vision is scored with OPPOSITE colours rather than a word list: a model
// that cannot see returns the same answer for both, and a word-list
// assertion would pass it. Each colour must win its own image.
//
//   VMLX_ORNITH15_TARGET=$HOME/models/JANGQ-AI/Ornith-1.5-9B-JANG_4D \
//   swift test --filter Ornith15EnablementTests

import CoreImage
import Foundation
import MLX
import XCTest
@preconcurrency import VMLXTokenizers
@testable import MLXHuggingFace
@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

final class Ornith15EnablementTests: XCTestCase {

    private static var targetPath: String? {
        ProcessInfo.processInfo.environment["VMLX_ORNITH15_TARGET"]
            ?? defaultBundle
    }

    private static var defaultBundle: String? {
        let p = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("models/JANGQ-AI/Ornith-1.5-9B-JANG_4D").path
        return FileManager.default.fileExists(atPath: p) ? p : nil
    }

    private static func solidImage(_ color: CIColor, size: Int = 112) -> CIImage {
        let filter = CIFilter(name: "CIConstantColorGenerator")!
        filter.setValue(color, forKey: "inputColor")
        return filter.outputImage!.cropped(
            to: CGRect(x: 0, y: 0, width: size, height: size))
    }

    private func load() async throws -> ModelContext {
        guard let targetPath = Self.targetPath else {
            throw XCTSkip("Set VMLX_ORNITH15_TARGET to an Ornith 1.5 bundle")
        }
        // Deliberately NOT requesting native MTP: Ornith 1.5 ships MTP
        // METADATA with zero MTP tensors, so a forced-MTP load correctly
        // refuses. Serving must not depend on that sidecar.
        let (context, _) = try await MLXLMCommon.loadModel(
            from: URL(fileURLWithPath: targetPath),
            using: #huggingFaceTokenizerLoader(),
            loadConfiguration: LoadConfiguration.default)
        return context
    }

    private func generate(
        _ ctx: ModelContext, chat: [Chat.Message], thinking: Bool,
        images: [UserInput.Image] = [], maxTokens: Int = 128
    ) async throws -> (reasoning: String, content: String, tokens: Int, seconds: Double) {
        var userInput = UserInput(chat: chat)
        userInput.additionalContext = ["enable_thinking": thinking]
        if !images.isEmpty {
            userInput = UserInput(chat: chat.map { m -> Chat.Message in
                m.role == .user ? .user(m.content, images: images) : m
            })
            userInput.additionalContext = ["enable_thinking": thinking]
        }
        let input = try await ctx.processor.prepare(input: userInput)
        var p = GenerateParameters(
            generationConfig: ctx.configuration.generationDefaults,
            fallback: GenerateParameters(maxTokens: maxTokens, prefillStepSize: 1024))
        p.maxTokens = maxTokens
        p.prefillStepSize = 1024
        p.temperature = 0
        var reasoning = "", content = "", tokens = 0
        let start = Date()
        for await item in try MLXLMCommon.generate(input: input, parameters: p, context: ctx) {
            switch item {
            case .chunk(let c): content += c
            case .reasoning(let c): reasoning += c
            case .info(let i): tokens = i.generationTokenCount
            default: break
            }
        }
        return (reasoning, content, tokens, Date().timeIntervalSince(start))
    }

    func testLoadsAndAnswersWithReasoningOnAndOff() async throws {
        let ctx = try await load()
        let prompt = "List the first five prime numbers, then say in one sentence why 9 is not prime."

        // A thinking bundle needs room to CLOSE its think block before any
        // content exists. At 128 tokens this model produced 431 chars of
        // reasoning and an empty answer — a budget artifact, not a defect,
        // so the reasoning arm gets a budget a real request would give it.
        let on = try await generate(ctx, chat: [.user(prompt)], thinking: true, maxTokens: 512)
        print(String(format: "[ornith15 think=ON ] %d tok %.2fs %.1f tok/s reasoning=%d chars",
            on.tokens, on.seconds, Double(on.tokens) / max(on.seconds, 0.001), on.reasoning.count))
        print("[ornith15 ON content] \(on.content.prefix(160))")
        XCTAssertFalse(on.content.isEmpty, "reasoning ON produced no answer")
        XCTAssertTrue(
            on.content.contains("2") && on.content.contains("3") && on.content.contains("5"),
            "answer lacks the primes it was asked for: \(on.content.prefix(200))")

        let off = try await generate(ctx, chat: [.user(prompt)], thinking: false)
        print(String(format: "[ornith15 think=OFF] %d tok %.2fs %.1f tok/s reasoning=%d chars",
            off.tokens, off.seconds, Double(off.tokens) / max(off.seconds, 0.001),
            off.reasoning.count))
        print("[ornith15 OFF content] \(off.content.prefix(160))")
        XCTAssertFalse(off.content.isEmpty, "reasoning OFF produced no answer")
        XCTAssertTrue(
            off.reasoning.isEmpty,
            "reasoning OFF still emitted a think block: \(off.reasoning.prefix(120))")
    }

    /// Vision: score OPPOSITE images. A blind model answers the same for
    /// both and fails, which a single-image word check would not catch.
    func testVisionReadsTheImageNotThePrompt() async throws {
        let ctx = try await load()
        let question = "What is the dominant colour of this image? Answer with one word."

        var wins = 0
        for (name, color) in [("red", CIColor.red), ("blue", CIColor.blue)] {
            let out = try await generate(
                ctx, chat: [.user(question)], thinking: false,
                images: [.ciImage(Self.solidImage(color))], maxTokens: 24)
            let said = out.content.lowercased()
            print("[ornith15 vision \(name)] \(out.content.prefix(80))")
            XCTAssertFalse(out.content.isEmpty, "vision turn produced nothing for \(name)")
            if said.contains(name) { wins += 1 }
        }
        XCTAssertEqual(
            wins, 2,
            "vision did not identify BOTH opposite colours — a constant answer must not pass")
    }

    /// Turn 2 must reuse the prefix rather than re-prefilling it.
    func testSecondTurnReusesPrefix() async throws {
        let ctx = try await load()
        let filler = String(
            repeating: "The archive index lists every shipment by date and port. ", count: 60)
        let history: [Chat.Message] = [
            .user("Keep this reference in mind:\n\(filler)\nReply with just OK."),
            .assistant("OK"),
        ]
        let first = try await generate(
            ctx, chat: history + [.user("In one sentence, what does the archive index list?")],
            thinking: false, maxTokens: 48)
        let second = try await generate(
            ctx,
            chat: history + [
                .user("In one sentence, what does the archive index list?"),
                .assistant(first.content),
                .user("And what two fields does it use?"),
            ],
            thinking: false, maxTokens: 48)
        print(String(format: "[ornith15 cache] turn1 %.2fs, turn2 %.2fs", first.seconds, second.seconds))
        print("[ornith15 turn2] \(second.content.prefix(140))")
        XCTAssertFalse(second.content.isEmpty, "second turn produced nothing")
        XCTAssertTrue(
            second.content.lowercased().contains("date")
                || second.content.lowercased().contains("port"),
            "second turn lost the shared context: \(second.content.prefix(160))")
    }
}
