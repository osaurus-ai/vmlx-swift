// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

private final class TinyHybridSSMModel: Module, LanguageModel, @unchecked Sendable {
    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [MambaCache()]
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        let step = max(1, windowSize ?? 512)
        var flatTokens = input.text.tokens.reshaped([-1])

        while flatTokens.size > step {
            let chunk = flatTokens[..<step][.newAxis, 0...]
            _ = callAsFunction(chunk, cache: cache)
            MLX.eval(cache)
            flatTokens = flatTokens[step...]
            Memory.clearCache()
        }

        return .tokens(LMInput.Text(tokens: flatTokens))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        guard let mamba = cache?.first as? MambaCache else {
            return MLXArray.zeros([1, max(inputs.size, 1), 1])
        }

        let tokenIds = inputs.reshaped([-1]).asArray(Int32.self)
        let previous = mamba.state.first?.asArray(Float.self) ?? [0, 0]
        let previousSum = previous.indices.contains(0) ? previous[0] : 0
        let previousCount = previous.indices.contains(1) ? previous[1] : 0
        let chunkSum = tokenIds.reduce(Float(0)) { $0 + Float($1) }
        let totalCount = previousCount + Float(tokenIds.count)
        let lastToken = tokenIds.last.map(Float.init) ?? -1

        mamba.state = [
            MLXArray([previousSum + chunkSum, totalCount] as [Float])
                .reshaped([1, 1, 2]),
            MLXArray([lastToken, Float(tokenIds.count)] as [Float])
                .reshaped([1, 1, 2]),
        ]
        mamba.offset += tokenIds.count
        return MLXArray.zeros([1, max(tokenIds.count, 1), 1])
    }
}

final class SSMReDeriveParityTests: XCTestCase {

    func testReDerivedPromptStateMatchesWarmPassState() throws {
        let model = TinyHybridSSMModel()
        let tokens = [11, 22, 33, 44, 55, 66, 77]

        let warm = try warmPassStates(model: model, tokens: tokens, prefillStepSize: 3)
        let rederived = try XCTUnwrap(
            reDeriveSSMStates(model: model, tokens: tokens, prefillStepSize: 3))

        assertStatesEqual(rederived, warm)
    }

    func testInlinePromptBoundaryCaptureMatchesReDerivedState() throws {
        let model = TinyHybridSSMModel()
        let promptOnly = [101, 102, 103, 104, 105]
        let generationPrompt = [201, 202]
        let fullPrompt = promptOnly + generationPrompt

        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false,
            enableDiskCache: false,
            modelKey: "tiny-hybrid|reasoning=off"))
        coordinator.setHybrid(true)

        let liveCache = model.newCache(parameters: nil)
        try runWarmPass(model: model, cache: liveCache, tokens: promptOnly, prefillStepSize: 2)
        captureCleanSSMStateInline(
            coordinator: coordinator,
            liveCache: liveCache,
            promptTokenIds: fullPrompt,
            genPromptLen: generationPrompt.count,
            enableSSMReDerive: true)

        let captured = try XCTUnwrap(
            coordinator.ssmStateCache.fetch(tokens: promptOnly, boundary: promptOnly.count))
        let rederived = try XCTUnwrap(
            reDeriveSSMStates(model: model, tokens: promptOnly, prefillStepSize: 2))

        assertStatesEqual(captured, rederived)
        XCTAssertEqual(coordinator.ssmStateCache.reDerives, 1)
    }

    func testPromptBoundaryStoreMatchesWarmPassAtExactBoundary() throws {
        let model = TinyHybridSSMModel()
        let tokens = [7, 8, 9, 10, 11]

        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false,
            enableDiskCache: false,
            modelKey: "tiny-hybrid|reasoning=on"))
        coordinator.setHybrid(true)

        let stored = try XCTUnwrap(
            reDeriveAndStoreSSMStatesForPromptBoundaries(
                coordinator: coordinator,
                model: model,
                promptTokenIds: tokens,
                prefillStepSize: 2))
        let warm = try warmPassStates(model: model, tokens: tokens, prefillStepSize: 2)
        let fetched = try XCTUnwrap(
            coordinator.ssmStateCache.fetch(tokens: tokens, boundary: tokens.count))

        assertStatesEqual(stored, warm)
        assertStatesEqual(fetched, warm)
        XCTAssertEqual(coordinator.ssmStateCache.reDerives, 1)
    }

    func testLegacyMaybeRederiveWrapperStoresPagedBlockAndExactBoundaries() throws {
        let model = TinyHybridSSMModel()
        let promptOnly = [3, 5, 7, 11, 13]
        let generationPrompt = [101, 102]
        let fullPrompt = promptOnly + generationPrompt

        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: true,
            enableDiskCache: false,
            pagedBlockSize: 2,
            modelKey: "tiny-hybrid|legacy-wrapper"))
        coordinator.setHybrid(true)

        maybeReDeriveSSMState(
            coordinator: coordinator,
            model: model,
            promptTokenIds: fullPrompt,
            genPromptLen: generationPrompt.count,
            enableSSMReDerive: true)

        let blockBoundary = 4
        let blockStates = try XCTUnwrap(
            coordinator.ssmStateCache.fetch(
                tokens: promptOnly,
                boundary: blockBoundary))
        let exactStates = try XCTUnwrap(
            coordinator.ssmStateCache.fetch(
                tokens: promptOnly,
                boundary: promptOnly.count))

        let warmBlock = try warmPassStates(
            model: model,
            tokens: Array(promptOnly.prefix(blockBoundary)),
            prefillStepSize: 2)
        let warmExact = try warmPassStates(
            model: model,
            tokens: promptOnly,
            prefillStepSize: 2)

        assertStatesEqual(blockStates, warmBlock)
        assertStatesEqual(exactStates, warmExact)
        XCTAssertEqual(coordinator.ssmStateCache.reDerives, 1)
    }

    func testExactSnapshotSSMStatesAvoidReDeriveWhenNoIntermediateBoundaryIsNeeded() throws {
        let model = TinyHybridSSMModel()
        let tokens = [3, 4, 5, 6, 7]
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: true,
            enableDiskCache: false,
            pagedBlockSize: 8,
            modelKey: "tiny-hybrid|exact-snapshot"))
        coordinator.setHybrid(true)

        let cache = model.newCache(parameters: nil)
        try runWarmPass(model: model, cache: cache, tokens: tokens, prefillStepSize: 2)
        let snapshot = cache.map { $0.copy() }

        let states = try XCTUnwrap(
            exactBoundarySSMStatesFromSnapshotIfSufficient(
                coordinator: coordinator,
                snapshot: snapshot,
                tokenCount: tokens.count))
        let warm = extractSSMStates(from: cache)

        assertStatesEqual(states, warm)
        XCTAssertEqual(
            coordinator.ssmStateCache.reDerives,
            0,
            "Exact snapshot extraction must not count as a fresh SSM re-derive")
    }

    func testExactSnapshotSSMStatesDeclineWhenPagedIntermediateBoundaryIsNeeded() throws {
        let model = TinyHybridSSMModel()
        let tokens = [3, 4, 5, 6, 7]
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: true,
            enableDiskCache: false,
            pagedBlockSize: 4,
            modelKey: "tiny-hybrid|needs-block-boundary"))
        coordinator.setHybrid(true)

        let cache = model.newCache(parameters: nil)
        try runWarmPass(model: model, cache: cache, tokens: tokens, prefillStepSize: 2)

        XCTAssertNil(
            exactBoundarySSMStatesFromSnapshotIfSufficient(
                coordinator: coordinator,
                snapshot: cache.map { $0.copy() },
                tokenCount: tokens.count),
            "Paged hybrid prompts longer than one block still need re-derive to capture intermediate SSM boundaries")
    }

    private func warmPassStates(
        model: TinyHybridSSMModel,
        tokens: [Int],
        prefillStepSize: Int
    ) throws -> [MLXArray] {
        let cache = model.newCache(parameters: nil)
        try runWarmPass(model: model, cache: cache, tokens: tokens, prefillStepSize: prefillStepSize)
        return extractSSMStates(from: cache)
    }

    private func runWarmPass(
        model: TinyHybridSSMModel,
        cache: [KVCache],
        tokens: [Int],
        prefillStepSize: Int
    ) throws {
        let tokenArray = MLXArray(tokens.map(Int32.init)).reshaped([1, tokens.count])
        let input = LMInput(text: .init(tokens: tokenArray))
        let result = try model.prepare(input, cache: cache, windowSize: prefillStepSize)
        if case .tokens(let tail) = result, tail.tokens.size > 0 {
            let tailInput = tail.tokens.reshaped([1, tail.tokens.size])
            _ = model.callAsFunction(tailInput, cache: cache)
        }
        MLX.eval(cache)
    }

    private func assertStatesEqual(
        _ actual: [MLXArray],
        _ expected: [MLXArray],
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        for (lhs, rhs) in zip(actual, expected) {
            MLX.eval(lhs, rhs)
            XCTAssertEqual(lhs.shape, rhs.shape, file: file, line: line)
            XCTAssertEqual(lhs.asArray(Float.self), rhs.asArray(Float.self), file: file, line: line)
        }
    }
}
