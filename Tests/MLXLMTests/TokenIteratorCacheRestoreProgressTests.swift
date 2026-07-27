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

final class TokenIteratorCacheRestoreProgressTests: XCTestCase {
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
