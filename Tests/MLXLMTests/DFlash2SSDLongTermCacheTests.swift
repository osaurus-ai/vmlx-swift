// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Long-term SSD prefix caching, across PROCESSES, on the real model.
//
// Within-process reuse can be served by the paged RAM tier, which proves
// nothing about the SSD. The only honest test of "long term" is: store
// in one process, kill it, and restore in a fresh one — where RAM is
// empty and any hit MUST have come from disk.
//
// Two phases, driven by env, run as two separate `swift test`
// invocations against the same on-disk cache directory:
//
//   VMLX_SSD_PHASE=store   — turn 1 through DFlash2TokenIterator with a
//                            disk-backed coordinator; store; verify the
//                            artifact landed on disk.
//   VMLX_SSD_PHASE=restore — FRESH process. Same directory + modelKey.
//                            coordinator.fetch for turn 2 must hit with
//                            detail == .disk at the stripped boundary,
//                            and building the turn-2 iterator must
//                            prefill dramatically faster than the cold
//                            control (fresh empty cache dir).
//
//   VMLX_DFLASH2_MATRIX_TARGET=<bundle> VMLX_SSD_PHASE=store   swift test --filter DFlash2SSDLongTermCacheTests
//   VMLX_DFLASH2_MATRIX_TARGET=<bundle> VMLX_SSD_PHASE=restore swift test --filter DFlash2SSDLongTermCacheTests

import Foundation
import MLX
import XCTest
@preconcurrency import VMLXTokenizers
@testable import MLXHuggingFace
@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

final class DFlash2SSDLongTermCacheTests: XCTestCase {

    private static let cacheDir = URL(fileURLWithPath: "/private/tmp/vmlx-dflash2-ssd-longterm")
    private static let modelKey = "qwen38-ssd-proof|reasoning=on"

    private static var targetPath: String? {
        ProcessInfo.processInfo.environment["VMLX_DFLASH2_MATRIX_TARGET"]
    }

    private static var drafterURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("models/Qwen3.8-27B-DFlash2")
    }

    // A long shared history is what makes the reuse visible: ~1.7k tokens
    // of prefix that turn 2 must NOT re-prefill.
    private static func makeSharedHistory() -> [Chat.Message] {
        let filler = String(
            repeating: "The quick brown fox jumps over the lazy dog near the riverbank at dawn. ",
            count: 80)
        return [
            .user("Here is a document to keep in mind:\n\(filler)\nAcknowledge briefly."),
            .assistant("Acknowledged. I have the document in mind."),
        ]
    }

    private func makeCoordinator(dir: URL, context: ModelContext) -> CacheCoordinator {
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: true,
            enableDiskCache: true,
            diskCacheDir: dir,
            modelKey: Self.modelKey))
        // Same derivation the host uses (ModelContainer / Bench): the
        // template's generation-prompt suffix marks the strip boundary.
        if let gp = context.tokenizer as? GenerationPromptControllableTokenizer {
            let dummy: [[String: any Sendable]] = [["role": "user", "content": "x"]]
            if let withGen = try? gp.applyChatTemplate(
                messages: dummy, tools: nil, additionalContext: nil,
                addGenerationPrompt: true),
                let withoutGen = try? gp.applyChatTemplate(
                    messages: dummy, tools: nil, additionalContext: nil,
                    addGenerationPrompt: false)
            {
                var common = 0
                while common < min(withGen.count, withoutGen.count),
                    withGen[common] == withoutGen[common]
                { common += 1 }
                let suffix = Array(withGen[common...])
                if (1 ... 64).contains(suffix.count) {
                    coordinator.setGenPromptSuffixTokens(suffix)
                }
            }
        }
        return coordinator
    }

    private func prepare(_ chat: [Chat.Message], context: ModelContext) async throws -> LMInput {
        var userInput = UserInput(chat: chat)
        userInput.additionalContext = ["enable_thinking": true]
        return try await context.processor.prepare(input: userInput)
    }

    func testLongTermSSDCache() async throws {
        guard let targetPath = Self.targetPath else {
            throw XCTSkip("Set VMLX_DFLASH2_MATRIX_TARGET to a Qwen3.8-27B bundle")
        }
        guard let phase = ProcessInfo.processInfo.environment["VMLX_SSD_PHASE"] else {
            throw XCTSkip("Set VMLX_SSD_PHASE=store or =restore")
        }
        guard DFlash2Loader.looksLikeDFlash2Drafter(at: Self.drafterURL) else {
            throw XCTSkip("No DFlash 2 drafter at \(Self.drafterURL.path)")
        }

        let context = try await MLXLMCommon.loadModel(
            from: URL(fileURLWithPath: targetPath),
            using: #huggingFaceTokenizerLoader())
        nonisolated(unsafe) let ctx = context
        let drafter = try DFlash2Loader.load(from: Self.drafterURL)

        var params = GenerateParameters(maxTokens: 8, temperature: 0)
        params.prefillStepSize = 1024

        if phase == "store" {
            try? FileManager.default.removeItem(at: Self.cacheDir)
            let coordinator = makeCoordinator(dir: Self.cacheDir, context: ctx)

            let turn1 = try await prepare(
                Self.makeSharedHistory() + [.user("Summarize the document in one sentence.")],
                context: ctx)
            print("[ssd store] turn1 prompt tokens=\(turn1.text.tokens.size)")
            var iterator = try DFlash2TokenIterator(
                input: turn1, target: ctx.model as! any DFlash2Target, drafter: drafter,
                blockSize: 4, parameters: params, cacheCoordinator: coordinator)
            for _ in 0 ..< 8 { _ = iterator.next() }
            iterator.storeCacheAfterGeneration(
                generatedTokenIds: [], includeGeneratedBoundary: false)

            // The artifact must be ON DISK — that is the whole claim.
            let files = (FileManager.default.enumerator(
                at: Self.cacheDir, includingPropertiesForKeys: [.fileSizeKey])?
                .compactMap { $0 as? URL }
                .filter { (try? $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true }) ?? []
            let totalBytes = files.reduce(Int64(0)) {
                $0 + Int64((try? $1.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
            }
            print("[ssd store] wrote \(files.count) file(s), \(totalBytes / 1_048_576) MB at \(Self.cacheDir.path)")
            XCTAssertGreaterThan(files.count, 0, "nothing was persisted to the SSD cache dir")
            XCTAssertGreaterThan(totalBytes, 1_048_576, "persisted cache is implausibly small")
            return
        }

        // ---- restore phase: FRESH process, RAM tiers empty ----
        XCTAssertEqual(phase, "restore")
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: Self.cacheDir.path),
            "run the store phase first")

        let coordinator = makeCoordinator(dir: Self.cacheDir, context: ctx)
        let turn2Chat = Self.makeSharedHistory() + [
            .user("Summarize the document in one sentence."),
            .assistant("A repeated pangram about a fox and a dog."),
            .user("Now count how many sentences it contained."),
        ]
        let turn2 = try await prepare(turn2Chat, context: ctx)
        let turn2Tokens = turn2.text.tokens.reshaped(-1).asArray(Int.self)
        print("[ssd restore] turn2 prompt tokens=\(turn2Tokens.count)")

        // 1. Fetch-level proof: the hit must come from DISK (RAM is empty
        //    in this process) and must match a real prefix.
        let mediaSalt = computeCacheSalt(for: turn2, parameters: params)
        let probe = coordinator.fetch(
            tokens: turn2Tokens, mediaSalt: mediaSalt, skipExactDiskBoundary: true)
        guard case .hit(let matched, let remaining, let detail, _, _, _) = probe else {
            return XCTFail("no cache hit in a fresh process — SSD persistence is not working")
        }
        print("[ssd restore] hit detail=\(detail) matched=\(matched) remaining=\(remaining.count)")
        XCTAssertEqual(detail, .disk, "fresh-process hit must be served from the SSD tier")
        XCTAssertGreaterThan(matched, 1000, "matched prefix implausibly short for a ~1.7k-token history")

        // 2. End-to-end effect: warm turn-2 prefill vs cold control.
        let warmStart = Date()
        var warm = try DFlash2TokenIterator(
            input: turn2, target: ctx.model as! any DFlash2Target, drafter: drafter,
            blockSize: 4, parameters: params, cacheCoordinator: coordinator)
        _ = warm.next()
        let warmSeconds = Date().timeIntervalSince(warmStart)

        let coldDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dflash2-ssd-cold-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: coldDir) }
        let coldCoordinator = makeCoordinator(dir: coldDir, context: ctx)
        let turn2Cold = try await prepare(turn2Chat, context: ctx)
        let coldStart = Date()
        var cold = try DFlash2TokenIterator(
            input: turn2Cold, target: ctx.model as! any DFlash2Target, drafter: drafter,
            blockSize: 4, parameters: params, cacheCoordinator: coldCoordinator)
        _ = cold.next()
        let coldSeconds = Date().timeIntervalSince(coldStart)

        print(String(
            format: "[ssd restore] warm prefill+first-token %.2fs vs cold %.2fs (%.1fx)",
            warmSeconds, coldSeconds, coldSeconds / Swift.max(warmSeconds, 0.001)))
        XCTAssertLessThan(
            warmSeconds, coldSeconds * 0.7,
            "SSD restore produced no measurable prefill saving — the cache is not taking effect")
    }
}
