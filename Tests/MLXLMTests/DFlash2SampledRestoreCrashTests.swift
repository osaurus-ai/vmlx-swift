// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Live-gate crash repro on the REAL bundle: a SAMPLED dFlash-2 turn
// (the bundle's own generation defaults — temp 1.0, top_p 0.95,
// top_k 20) generates, stores its cache through a coordinator, and a
// follow-on request builds a second iterator on the same coordinator.
// In the osaurus proof app the second `DFlash2TokenIterator.init` died
// on an `MLXArray.item` precondition during restore (2026-08-20).
//
//   VMLX_DFLASH2_MATRIX_TARGET=$HOME/models/JANGQ-AI/Qwen3.8-27B-JANG_4D \
//   swift test --filter DFlash2SampledRestoreCrashTests

import Foundation
import MLX
import XCTest
@preconcurrency import VMLXTokenizers
@testable import MLXHuggingFace
@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

final class DFlash2SampledRestoreCrashTests: XCTestCase {

    private static var targetPath: String? {
        ProcessInfo.processInfo.environment["VMLX_DFLASH2_MATRIX_TARGET"]
    }

    private static var drafterURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("models/Qwen3.8-27B-DFlash2")
    }

    func testSampledTurnStoreThenFollowOnInitSurvives() async throws {
        guard let targetPath = Self.targetPath else {
            throw XCTSkip("Set VMLX_DFLASH2_MATRIX_TARGET to a Qwen3.8-27B bundle")
        }
        guard DFlash2Loader.looksLikeDFlash2Drafter(at: Self.drafterURL) else {
            throw XCTSkip("No DFlash 2 drafter at \(Self.drafterURL.path)")
        }

        let (context, _) = try await MLXLMCommon.loadModel(
            from: URL(fileURLWithPath: targetPath),
            using: #huggingFaceTokenizerLoader(),
            loadConfiguration: LoadConfiguration.default)
        nonisolated(unsafe) let ctx = context

        let diskDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dflash2-sampled-crash-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: diskDir) }
        // The app's exact tier config: paged RAM cache OFF, block-disk ON
        // (osaurus ships SSD-L2-only by default), so the follow-on restore
        // goes through the DISK-ARRAY path — the tier the crash lives in.
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false,
            enableDiskCache: true,
            diskCacheDir: diskDir,
            modelKey: "qwen38-sampled-crash|reasoning=off"))

        // The bundle's own generation defaults — what osaurus actually runs.
        var params = GenerateParameters(maxTokens: 96, temperature: 1.0)
        params.topP = 0.95
        params.topK = 20
        params.prefillStepSize = 1024
        params.draftStrategy = .dflash2(drafterPath: Self.drafterURL, blockSize: nil)

        func makeIterator(_ chat: [Chat.Message]) async throws -> DFlash2TokenIterator {
            var ui = UserInput(chat: chat)
            ui.additionalContext = ["enable_thinking": false]
            let input = try await ctx.processor.prepare(input: ui)
            guard let target = ctx.model as? any DFlash2Target else {
                throw XCTSkip("model is not a DFlash2 target")
            }
            let drafter = try DFlash2Loader.load(from: Self.drafterURL)
            return try DFlash2TokenIterator(
                input: input,
                target: target,
                drafter: drafter,
                blockSize: nil,
                parameters: params,
                cacheCoordinator: coordinator)
        }

        // Turn 1: sampled generation + store. The live crash needed a LARGE
        // restored prefix (2986 rows in the app) — the drafter's
        // sliding-window/offset arithmetic only degenerates at scale, so a
        // short history passes and proves nothing.
        let filler = String(
            repeating: "The quick brown fox jumps over the lazy dog near the riverbank at dawn. ",
            count: 170)
        let turn1Question =
            "Here is a document to keep in mind:\n\(filler)\n"
                + "List the first eight prime numbers, then explain in two sentences why 1 is not prime."
        var first = try await makeIterator([.user(turn1Question)])
        var generated: [Int] = []
        while let token = first.next(), generated.count < 96 {
            generated.append(token)
        }
        XCTAssertFalse(generated.isEmpty, "turn 1 must generate")
        first.storeCacheAfterGeneration(
            generatedTokenIds: generated, includeGeneratedBoundary: false)

        // Turn 2: REAL multiturn — history plus a new user message, the
        // shape every follow-on osaurus request has. This restores the
        // stored stripped prefix as a PARTIAL hit and prefills only the
        // suffix; the live crash fired in the drafter's
        // GroupedDynamicCausalConv on the first cycle after exactly this
        // restore. A full-hit (identical prompt) never trips it.
        let reply = ctx.tokenizer.decode(tokenIds: generated)
        var second = try await makeIterator([
            .user(turn1Question),
            .assistant(reply),
            .user("Now list the first five composite numbers and explain what makes a number composite."),
        ])
        var followOn: [Int] = []
        while let token = second.next(), followOn.count < 48 {
            followOn.append(token)
        }
        XCTAssertFalse(followOn.isEmpty, "follow-on partial-hit turn must generate")
        second.storeCacheAfterGeneration(
            generatedTokenIds: followOn, includeGeneratedBoundary: false)
    }

    /// Same recipe through the APP's actual construction path:
    /// BatchEngine.generate → startSoloFastPath → DFlash2TokenIterator,
    /// engine warm cache, prompt tail, store, then a second generate.
    /// The osaurus crash stack goes through exactly this path; the
    /// direct-construction test above passes, so the trigger lives in
    /// what the engine adds.
    func testSampledTwoTurnsThroughBatchEngineSurvive() async throws {
        guard let targetPath = Self.targetPath else {
            throw XCTSkip("Set VMLX_DFLASH2_MATRIX_TARGET to a Qwen3.8-27B bundle")
        }
        guard DFlash2Loader.looksLikeDFlash2Drafter(at: Self.drafterURL) else {
            throw XCTSkip("No DFlash 2 drafter at \(Self.drafterURL.path)")
        }

        let (context, _) = try await MLXLMCommon.loadModel(
            from: URL(fileURLWithPath: targetPath),
            using: #huggingFaceTokenizerLoader(),
            loadConfiguration: LoadConfiguration.default)
        nonisolated(unsafe) let ctx = context

        let diskDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dflash2-engine-crash-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: diskDir) }
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false,
            enableDiskCache: true,
            diskCacheDir: diskDir,
            modelKey: "qwen38-engine-crash|reasoning=off"))
        let engine = BatchEngine(
            context: ctx, maxBatchSize: 1, cacheCoordinator: coordinator)

        var params = GenerateParameters(maxTokens: 96, temperature: 1.0)
        params.topP = 0.95
        params.topK = 20
        params.prefillStepSize = 1024
        params.draftStrategy = .dflash2(drafterPath: Self.drafterURL, blockSize: nil)

        func run(_ chat: [Chat.Message]) async throws -> String {
            var ui = UserInput(chat: chat)
            ui.additionalContext = ["enable_thinking": false]
            nonisolated(unsafe) let input = try await ctx.processor.prepare(input: ui)
            var text = ""
            let stream = await engine.generate(input: input, parameters: params)
            for await event in stream {
                if case .chunk(let c) = event { text += c }
            }
            return text
        }

        let question =
            "List the first eight prime numbers, then explain in two sentences why 1 is not prime."
        let reply = try await run([.user(question)])
        XCTAssertFalse(reply.isEmpty, "turn 1 through the engine must generate")

        let reply2 = try await run([
            .user(question),
            .assistant(reply),
            .user("Now list the first five composite numbers and explain what makes a number composite."),
        ])
        XCTAssertFalse(reply2.isEmpty, "turn 2 through the engine must generate")
    }
}
