// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
@preconcurrency import VMLXTokenizers
import XCTest

@testable import MLXHuggingFace
@testable import MLXLMCommon
@testable import MLXVLM

final class DFlash2SeededRequestTests: XCTestCase {
    /// A bounded token-stream diagnostic, not a completed-answer quality test.
    /// Fixed width avoids timing-dependent adaptive-plan changes between runs.
    func testChainRequestSeedIsIndependentOfGlobalRandomness() async throws {
        guard let path = ProcessInfo.processInfo.environment["VMLX_DFLASH2_SEEDED_TARGET"] else {
            throw XCTSkip(
                "Set VMLX_DFLASH2_SEEDED_TARGET to an existing target with bundled dflash2")
        }
        try await MLXMetalTestLock.withLock {
            let directory = URL(fileURLWithPath: path)
            let draftDirectory = directory.appendingPathComponent("dflash2")
            guard DFlash2Loader.looksLikeDFlash2Drafter(at: draftDirectory) else {
                throw XCTSkip("The selected local bundle has no dflash2 drafter")
            }
            let previousTreeMode = ProcessInfo.processInfo.environment["VMLX_DFLASH2_TREE"]
            setenv("VMLX_DFLASH2_TREE", "0", 1)
            defer {
                if let previousTreeMode {
                    setenv("VMLX_DFLASH2_TREE", previousTreeMode, 1)
                } else {
                    unsetenv("VMLX_DFLASH2_TREE")
                }
            }

            let container = try await VLMModelFactory.shared.loadContainer(
                from: directory, using: #huggingFaceTokenizerLoader())
            let drafter = try DFlash2Loader.load(from: draftDirectory)
            try await container.perform(nonSendable: drafter) { context, drafter in
                guard let target = context.model as? any DFlash2Target else {
                    return XCTFail("Selected target does not support DFlash2")
                }
                let input = try await context.processor.prepare(
                    input: UserInput(chat: [
                        .user("Describe how a radix trie searches for an exact key.")
                    ]))
                var parameters = GenerateParameters(
                    generationConfig: context.configuration.generationDefaults)
                parameters.maxTokens = 96
                parameters.randomSeed = 42
                XCTAssertGreaterThan(parameters.temperature, 0)

                func run(globalSeed: UInt64) async throws -> [Int] {
                    MLXRandom.seed(globalSeed)
                    let iterator = try DFlash2TokenIterator(
                        input: input, target: target, drafter: drafter,
                        blockSize: 8, parameters: parameters)
                    let (stream, task) = generateTokenTask(
                        promptTokenCount: input.text.tokens.size,
                        modelConfiguration: context.configuration,
                        tokenizer: context.tokenizer, iterator: iterator)
                    var tokens: [Int] = []
                    var terminal: GenerateCompletionInfo?
                    for await event in stream {
                        switch event {
                        case .token(let token): tokens.append(token)
                        case .info(let info): terminal = info
                        default: break
                        }
                    }
                    await task.value
                    let info = try XCTUnwrap(terminal)
                    let stats = try XCTUnwrap(info.dflash2Stats)
                    XCTAssertGreaterThan(stats.verifyCalls, 0)
                    XCTAssertEqual(
                        stats.treeCycles, 0, "The regression must exercise chain sampling")
                    XCTAssertEqual(info.generationTokenCount, tokens.count)
                    XCTAssertNil(info.generationFailure)
                    XCTAssertTrue(info.tokensPerSecond.isFinite)
                    XCTAssertGreaterThan(info.tokensPerSecond, 0)
                    print(
                        "DFlash2 seeded chain: \(tokens.count) tokens, \(info.tokensPerSecond) tok/s"
                    )
                    return tokens
                }

                let first = try await run(globalSeed: 8)
                let perturbed = try await run(globalSeed: 555)
                XCTAssertFalse(first.isEmpty)
                XCTAssertEqual(
                    first, perturbed, "The request seed must own every chain sampling draw")
            }
        }
    }
}
