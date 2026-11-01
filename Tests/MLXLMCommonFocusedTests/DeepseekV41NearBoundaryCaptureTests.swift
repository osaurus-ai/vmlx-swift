// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 stable cache boundaries captured during prefill only near the end of the span being prefilled.
//
// A capture cut prefills the head as its own prompt; with prompt-suffix pruning the head then runs the pruned upper
// layers over the rows its own end depends on, which the uncut prefill never computes. The TokenIterator therefore
// cuts only at stable boundaries within the model's unpruned tail of the end (`PrunedPromptPrefillModel`), leaves the
// others to a later prompt that ends near them, and never replays them after the answer. CPU, tiny random weights
// (the 8-layer model of `DeepseekV41SuffixPruneFocusedTests`: layers 5-7 pruned), a private SSD tier.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("DeepSeek-V4.1 stable boundaries cut in prefill only near the end (CPU)", .serialized)
struct DeepseekV41NearBoundaryCaptureTests {
    typealias Prune = DeepseekV41SuffixPruneFocusedTests

    /// The caller's prefill step: the tiny model's unpruned tail.
    static let step = 32

    static func coordinator(_ dir: URL, model: DeepseekV4Model) -> CacheCoordinator {
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false, enableDiskCache: true, diskCacheDir: dir, modelKey: "dsv41-near-cut"))
        let topology = ModelCacheTopologySnapshot(cache: model.newCache(parameters: nil))
        coordinator.setHybrid(topology.requiresSSMCompanionState,
            requiresRecurrentSSMCompanion: topology.requiresRecurrentSSMCompanionState,
            requiresSeparateRecurrentPayload: topology.requiresSeparateRecurrentPayloadState)
        return coordinator
    }

    static func input(_ ids: [Int32], stable: [Int]) -> LMInput {
        LMInput(tokens: MLXArray(ids).expandedDimensions(axis: 0), tokenIds: ids.map(Int.init),
            cachePrefixTokenCounts: stable, cacheStablePrefixTokenCounts: stable)
    }

    /// `ids[from..<to]` prefilled as its own prompt (`prepare` + the remainder forward), as a capture cut does.
    static func segment(_ model: DeepseekV4Model, _ ids: ArraySlice<Int32>, _ cache: [KVCache]) throws -> MLXArray {
        let tokens = MLXArray(Array(ids)).reshaped(1, -1)
        var logits: MLXArray
        switch try model.prepare(LMInput(text: LMInput.Text(tokens: tokens)), cache: cache, windowSize: step) {
        case .tokens(let rest):
            logits = model(LMInput.Text(tokens: rest.tokens.reshaped(1, -1)), cache: cache, state: nil).logits
        case .logits(let output):
            logits = output.logits
        }
        MLX.eval(logits)
        MLX.eval(cache)
        return logits
    }

    static func identical(_ a: [KVCache], _ b: [KVCache]) -> Bool {
        a.count == b.count && zip(a, b).allSatisfy { x, y in
            x.offset == y.offset && x.state.count == y.state.count
                && zip(x.state, y.state).allSatisfy { DeepseekV41DSparkFocusedTests.equal($0, $1) }
                && x.metaState == y.metaState
        }
    }

    @Test("the unpruned tail is the prepare step for V4.1 with pruning, nil without")
    func unprunedTail() throws {
        try Device.withDefaultDevice(.cpu) {
            let bundle = try Prune.writeEngramBundle(layers: [1, 4], vocab: 128)
            defer { try? FileManager.default.removeItem(at: bundle) }
            let model = try Prune.model(bundle: bundle)
            #expect(model.unprunedPromptTailRows(windowSize: Self.step) == Self.step)
            #expect(model.unprunedPromptTailRows(windowSize: 512) == 512)
            model.suffixPruneEnabled = false
            defer { model.suffixPruneEnabled = DeepseekV4Math.suffixPruneEnabled }
            #expect(model.unprunedPromptTailRows(windowSize: Self.step) == nil)
            // DSV4-0731 (no staggered mHC / Engram): nothing is pruned.
            let v40 = DeepseekV4Model(DeepseekV41DSparkFocusedTests.tinyConfig(v41: false))
            #expect(v40.unprunedPromptTailRows(windowSize: Self.step) == nil)
        }
    }

    @Test("a far stable boundary is not cut nor replayed; the near one is cut, stored, and exact")
    func farLeftNearCut() throws {
        try Device.withDefaultDevice(.cpu) {
            let bundle = try Prune.writeEngramBundle(layers: [1, 4], vocab: 128)
            defer { try? FileManager.default.removeItem(at: bundle) }
            let model = try Prune.model(bundle: bundle)
            let disk = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: disk) }
            let coordinator = Self.coordinator(disk, model: model)
            // n = 203 (prompt-minus-one seed at 202); stable 61 (store 60: 142 rows before 202, far) and
            // 187 (store 186: 16 rows before 202, within the 32-row tail).
            let ids = Prune.tokens(203, seed: 21)
            let parameters = GenerateParameters(maxTokens: 1, temperature: 0, prefillStepSize: Self.step)
            var iterator = try TokenIterator(
                input: Self.input(ids, stable: [61, 187]), model: model, parameters: parameters,
                cacheCoordinator: coordinator)
            #expect(iterator.poolStableBoundariesLeftUncut == [60])
            #expect(iterator.stableBoundarySnapshots[60] == nil)
            let near = try #require(iterator.stableBoundarySnapshots[186])

            // The near snapshot and the request's state are the cold prefill cut at 186 and 202 exactly.
            let reference = model.newCache(parameters: nil)
            _ = try Self.segment(model, ids[0 ..< 186], reference)
            let atNear = reference.map { $0.copy() }
            MLX.eval(atNear)
            _ = try Self.segment(model, ids[186 ..< 202], reference)
            let logits = try Self.segment(model, ids[202 ..< 203], reference)
            #expect(Self.identical(near, atNear))
            #expect(Self.identical(iterator.cache, reference))
            #expect(iterator.y.tokens.item(Int.self) == argMax(logits[0, -1]).item(Int.self))

            // The store stores the seed and the near boundary, and replays nothing for the far one.
            iterator.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
            #expect(coordinator.waitForPendingDiskWrites())
            let all = ids.map(Int.init)
            #expect(coordinator.hasDurableDiskEntry(tokens: Array(all.prefix(202)), mediaSalt: iterator.mediaSalt))
            #expect(coordinator.hasDurableDiskEntry(tokens: Array(all.prefix(186)), mediaSalt: iterator.mediaSalt))
            #expect(!coordinator.hasDurableDiskEntry(tokens: Array(all.prefix(60)), mediaSalt: iterator.mediaSalt))

            // The next turn of the same chat restores the seed (202 rows: both stable boundaries lie inside the
            // restored prefix, the far one not durable). Its store must not replay the far boundary either.
            let turn2 = ids + Prune.tokens(20, seed: 31)
            var warm = try TokenIterator(
                input: Self.input(turn2, stable: [61, 187]), model: model, parameters: parameters,
                cacheCoordinator: coordinator)
            #expect(coordinator.snapshotStats().diskStats!.hits > 0)
            #expect(warm.stableBoundarySnapshots[60] == nil)
            warm.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
            #expect(coordinator.waitForPendingDiskWrites())
            #expect(!coordinator.hasDurableDiskEntry(tokens: Array(all.prefix(60)), mediaSalt: warm.mediaSalt))
            #expect(coordinator.hasDurableDiskEntry(
                tokens: Array(turn2.prefix(turn2.count - 1)).map(Int.init), mediaSalt: warm.mediaSalt))

            // A later prompt that ends near the far boundary (same first 61 tokens + a short message) cuts it.
            let next = Array(ids[0 ..< 61]) + Prune.tokens(12, seed: 77)
            var second = try TokenIterator(
                input: Self.input(next, stable: [61]), model: model, parameters: parameters,
                cacheCoordinator: coordinator)
            #expect(second.poolStableBoundariesLeftUncut.isEmpty)
            #expect(second.stableBoundarySnapshots[60] != nil)
            second.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
            #expect(coordinator.waitForPendingDiskWrites())
            #expect(coordinator.hasDurableDiskEntry(tokens: Array(all.prefix(60)), mediaSalt: second.mediaSalt))
        }
    }

    @Test("without pruning every stable boundary is cut, as before")
    func noPruningCutsEverything() throws {
        try Device.withDefaultDevice(.cpu) {
            let bundle = try Prune.writeEngramBundle(layers: [1, 4], vocab: 128)
            defer { try? FileManager.default.removeItem(at: bundle) }
            let model = try Prune.model(bundle: bundle)
            model.suffixPruneEnabled = false
            defer { model.suffixPruneEnabled = DeepseekV4Math.suffixPruneEnabled }
            let disk = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: disk) }
            let coordinator = Self.coordinator(disk, model: model)
            let ids = Prune.tokens(203, seed: 21)
            let parameters = GenerateParameters(maxTokens: 1, temperature: 0, prefillStepSize: Self.step)
            let iterator = try TokenIterator(
                input: Self.input(ids, stable: [61, 187]), model: model, parameters: parameters,
                cacheCoordinator: coordinator)
            #expect(iterator.poolStableBoundariesLeftUncut.isEmpty)
            #expect(iterator.stableBoundarySnapshots[60] != nil)
            #expect(iterator.stableBoundarySnapshots[186] != nil)
        }
    }
}
