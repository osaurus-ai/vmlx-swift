// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

private final class CacheRestoreProgressModel: Module, LanguageModel, @unchecked Sendable {
    var vocabularySize: Int { 64 }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [KVCacheSimple()]
    }

    func prepare(
        _ input: LMInput,
        cache: [KVCache],
        windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(LMInput.Text(tokens: input.text.tokens.reshaped([-1])))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let tokenCount = inputs.reshaped([-1]).size
        if let cache = cache?.first as? KVCacheSimple {
            let keys = MLXArray.zeros([1, 1, tokenCount, 4])
            let values = MLXArray.zeros([1, 1, tokenCount, 4])
            _ = cache.update(keys: keys, values: values)
        }
        return MLXArray.zeros([1, max(tokenCount, 1), vocabularySize])
    }
}

private final class CacheRestoreProgressRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [PrefillProgress] = []

    func append(_ value: PrefillProgress) {
        lock.lock()
        values.append(value)
        lock.unlock()
    }

    func snapshot() -> [PrefillProgress] {
        lock.lock()
        defer { lock.unlock() }
        return values
    }
}

/// A real iterator pipeline with distinguishable visible, stop and lookahead
/// tokens. Cache values record the tokens actually forwarded through the model.
private final class PostAnswerBoundaryModel: Module, LanguageModel, @unchecked Sendable {
    var vocabularySize: Int { 32 }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [KVCacheSimple(), RotatingKVCache(maxSize: 16, keep: 0)]
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(LMInput.Text(tokens: input.text.tokens.reshaped([-1])))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let tokens = inputs.reshaped([-1])
        let values = tokens.asType(.float32).reshaped(1, 1, -1, 1)
        for layer in cache ?? [] {
            _ = layer.update(keys: values, values: values)
        }
        let last = tokens[-1].item(Int.self)
        let next = last == 1 ? 10 : last == 10 ? 11 : 12
        var logits = [Float](repeating: -1000, count: vocabularySize)
        logits[next] = 0
        return MLXArray(logits).reshaped(1, 1, vocabularySize)
    }
}

final class TokenIteratorCacheRestoreProgressTests: XCTestCase {
    func testLegacyLookaheadBoundaryCannotRestoreUnderCurrentPolicy() throws {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }
        let diskDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("token-iterator-legacy-boundary-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: diskDir) }
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false, enableDiskCache: true, diskCacheDir: diskDir,
            modelKey: "token-iterator-legacy-boundary"))
        let model = PostAnswerBoundaryModel()
        let parameters = GenerateParameters(maxTokens: 8, temperature: 0)
        let cache = model.newCache(parameters: parameters)
        eval(model(MLXArray([Int32(1), 10, 11]), cache: cache))
        // Captured v4 policy SHA256 for text-only/default KV. This old key has
        // the right offset but the wrong final token, and may already have
        // been promoted from post-answer to resume by an earlier cache hit.
        let legacySalt = "f256f8c426fbc2b1061eafd33f84fe2dd0f7327a1b35eda38e70fc4777427f81"
        coordinator.storeAfterGeneration(
            promptTokens: [1, 10, 12], perLayerData: [], ssmStates: nil,
            cache: cache, mediaSalt: legacySalt, isResumeBoundary: true)
        guard case .hit(let matched, _, _, _, _, _) = coordinator.fetch(
            tokens: [1, 10, 12, 13], mediaSalt: legacySalt)
        else { return XCTFail("Legacy fixture must exist on disk") }
        XCTAssertEqual(matched, 3)
        let currentSalt = computeCacheSalt(
            for: LMInput(tokens: MLXArray([Int32(1)])), parameters: parameters)
        guard case .miss = coordinator.fetch(tokens: [1, 10, 12, 13], mediaSalt: currentSalt)
        else { return XCTFail("Current requests must isolate v4 lookahead-keyed snapshots") }
    }

    func testLengthStopBoundaryDoesNotAppendPrediction() throws {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }
        let diskDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("token-iterator-length-boundary-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: diskDir) }
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false, enableDiskCache: true, diskCacheDir: diskDir,
            modelKey: "token-iterator-length-boundary"))
        let input = LMInput(tokens: MLXArray([Int32(1)]))
        let parameters = GenerateParameters(maxTokens: 1, temperature: 0)
        var iterator = try TokenIterator(
            input: input, model: PostAnswerBoundaryModel(), parameters: parameters,
            cacheCoordinator: coordinator)
        XCTAssertEqual(iterator.next(), 10)
        XCTAssertNil(iterator.next())
        iterator.storeCacheAfterGeneration(generatedTokenIds: [10], includeGeneratedBoundary: true)
        guard case .hit(let matched, _, _, _, _, _) = coordinator.fetch(
            tokens: [1, 10, 11, 13], mediaSalt: computeCacheSalt(for: input, parameters: parameters))
        else { return XCTFail("The completed length boundary must be reusable") }
        XCTAssertEqual(matched, 2, "The unprocessed next prediction must not extend a length stop")
    }

    func testPostAnswerDiskBoundaryUsesForwardedStopNotLookahead() throws {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }
        let diskDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("token-iterator-stop-boundary-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: diskDir) }
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false, enableDiskCache: true, diskCacheDir: diskDir,
            modelKey: "token-iterator-stop-boundary"))
        let input = LMInput(tokens: MLXArray([Int32(1)]))
        let parameters = GenerateParameters(maxTokens: 8, temperature: 0)
        let salt = computeCacheSalt(for: input, parameters: parameters)
        var iterator = try TokenIterator(
            input: input,
            model: PostAnswerBoundaryModel(),
            parameters: parameters,
            cacheCoordinator: coordinator)
        XCTAssertEqual(iterator.next(), 10)
        // Returning stop=11 has already forwarded 11 and computed lookahead=12.
        XCTAssertEqual(iterator.next(), 11)
        iterator.storeCacheAfterGeneration(generatedTokenIds: [10], includeGeneratedBoundary: true)
        if case .hit(let matched, _, _, _, _, _) = coordinator.fetch(
            tokens: [1, 10, 11, 13], mediaSalt: salt)
        {
            XCTAssertEqual(matched, 3, "Reuse must include the forwarded stop, not fall back to the prompt")
        } else {
            XCTFail("The real consumed stop boundary must be reusable from disk")
        }
        if case .hit(let wrongMatched, _, _, _, _, _) = coordinator.fetch(
            tokens: [1, 10, 12, 13], mediaSalt: salt)
        {
            XCTAssertLessThan(wrongMatched, 3, "The unforwarded lookahead must never label this cache state")
        }
    }

    func testAcceptedDiskPrefixRestoreEmitsStructuredProgressBeforePrefill() throws {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }

        let diskDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("token-iterator-cache-progress-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: diskDir) }

        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false,
            enableDiskCache: true,
            diskCacheDir: diskDir,
            modelKey: "token-iterator-cache-progress"))
        let parameters = GenerateParameters(maxTokens: 1, temperature: 0)
        let cachedPrompt = [1, 2, 3, 4, 5, 6]

        var cold = try TokenIterator(
            input: LMInput(
                tokens: MLXArray(cachedPrompt.map(Int32.init))
                    .expandedDimensions(axis: 0)),
            model: CacheRestoreProgressModel(),
            parameters: parameters,
            cacheCoordinator: coordinator)
        cold.storeCacheAfterGeneration(
            generatedTokenIds: [],
            includeGeneratedBoundary: false)

        let recorder = CacheRestoreProgressRecorder()
        let warmPrompt = cachedPrompt + [7, 8]
        _ = try TokenIterator(
            input: LMInput(
                tokens: MLXArray(warmPrompt.map(Int32.init))
                    .expandedDimensions(axis: 0)),
            model: CacheRestoreProgressModel(),
            parameters: parameters,
            cacheCoordinator: coordinator,
            prefillProgressHandler: { recorder.append($0) })

        let progress = recorder.snapshot()
        guard let restoreIndex = progress.firstIndex(where: {
            $0.stage == .cacheRestore
        }) else {
            return XCTFail("accepted disk prefix restore must emit cacheRestore progress")
        }
        guard let prefillIndex = progress.firstIndex(where: {
            $0.stage == .prefill
        }) else {
            return XCTFail("warm suffix must still emit prefill progress")
        }

        let restore = progress[restoreIndex]
        XCTAssertEqual(restore.detail, CacheDetail.disk.rawValue)
        XCTAssertEqual(restore.completedUnitCount, cachedPrompt.count)
        XCTAssertEqual(restore.totalUnitCount, warmPrompt.count)
        XCTAssertLessThan(restoreIndex, prefillIndex)
        XCTAssertEqual(
            progress[prefillIndex].completedUnitCount,
            cachedPrompt.count)
    }
}
