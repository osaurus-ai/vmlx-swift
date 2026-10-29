// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Cmlx
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Long-context serving fixes from the MiniMax-M3.1 Osaurus GUI proof (2026-10-29): what made a 35k-token agent turn
/// lose its prefix cache entry and decode at a quarter speed.
///
/// - Prompt snapshot: the solo path kept a full copy of the prompt's KV through the whole decode (4.5 GiB at 35k tokens)
///   and copied it again for each boundary store; the store budget then refused both stores ("the copies it requires do
///   not fit in memory"), so the next turn re-prefilled the document. M3's lanes only append, so the end-of-turn stores
///   now take views of the finished cache: nothing held through decode, nothing copied, the budget admits them.
/// - Post-answer entry: under aligned prefill a restore never reuses rows past the producing request's store boundary,
///   so the post-answer entry was a redundant 4.6 GiB copy + write per turn; skipped.
/// - Scheduling floor: MLX's scheduling limit below the live working set made every eval wait (9.7 vs 36.9 tok/s at
///   35.7k on the Studio); the generation loops keep it at the live set + 4 GiB.
@Suite("MiniMax-M3 prompt snapshot views, post-answer skip, scheduling floor", .serialized)
struct MiniMaxM3PromptSnapshotViewTests {
    typealias A = MiniMaxM3AlignedPrefixCacheTests
    typealias S = MiniMaxM3StructuralPrefixCacheTests

    static func address(_ a: MLXArray) -> UInt { UInt(bitPattern: mlx_array_data_uint8(a.ctx)) }

    static func filledCache(rows: Int, seed: UInt64) -> MiniMaxM3SparseCache {
        let cache = MiniMaxM3SparseCache()
        let key = MLXRandom.key(seed)
        let k = MLXRandom.normal([1, 2, rows, 16], key: key).asType(.bfloat16)
        let v = (MLXRandom.normal([1, 2, rows, 16], key: MLXRandom.split(key: key).0) * 2).asType(.bfloat16)
        let ix = MLXRandom.normal([1, 1, rows, 8], key: MLXRandom.split(key: key).1).asType(.bfloat16)
        // Two appends so the lanes carry step-padded capacity past `offset`, as in serving.
        let split = rows / 3
        _ = cache.update(keys: k[.ellipsis, ..<split, 0...], values: v[.ellipsis, ..<split, 0...])
        _ = cache.updateIndexKeys(ix[.ellipsis, ..<split, 0...])
        _ = cache.update(keys: k[.ellipsis, split..., 0...], values: v[.ellipsis, split..., 0...])
        _ = cache.updateIndexKeys(ix[.ellipsis, split..., 0...])
        MLX.eval(cache.innerState())
        return cache
    }

    @Test("prefixView: the first rows of every lane, bit for bit, in the source's own buffers (no copy)")
    func prefixViewSharesTheSourceBuffers() throws {
        try A.withRecorder { _ in
            let source = Self.filledCache(rows: 300, seed: 7)
            let rows = 257
            let view = try #require(source.prefixView(rows: rows) as? MiniMaxM3SparseCache)
            #expect(view.isPrefixView && !source.isPrefixView)
            #expect(view.offset == rows && view.indexRows == rows)
            let (a, b) = (view.state, source.state)
            MLX.eval(a + b)
            for (x, y) in zip(a, b) {
                #expect(A.equal(x, y[.ellipsis, ..<rows, 0...]))
            }
            // Same buffers: the view's arrays start where the source's capacity buffers start.
            #expect(Self.address(a[0]) == Self.address(source.kv.keys!))
            #expect(Self.address(a[1]) == Self.address(source.kv.values!))
            #expect(Self.address(a[2]) == Self.address(source.indexStorage!))
            // Control: copy() owns new buffers and is not a view.
            let copy = try #require(source.copy() as? MiniMaxM3SparseCache)
            MLX.eval(copy.state)
            #expect(!copy.isPrefixView && Self.address(copy.state[0]) != Self.address(source.kv.keys!))
            // Trimming the view leaves the source alone.
            view.trim(57)
            #expect(view.offset == 200 && source.offset == 300)
            // A view of a view is a view of the same rows.
            let inner = try #require(prefixViews(of: [view], rows: 100)?.first as? MiniMaxM3SparseCache)
            #expect(inner.isPrefixView && A.equal(inner.state[0], b[0][.ellipsis, ..<100, 0...]))
        }
    }

    @Test("Store budget: a view snapshot is admitted at zero headroom; the same rows as a copy are refused (control)")
    func storeBudgetAdmitsViews() throws {
        try A.withRecorder { _ in
            let source = Self.filledCache(rows: 300, seed: 9)
            let views = try #require(prefixViews(of: [source], rows: 300))
            let copies: [KVCache] = [source.copy()]
            MLX.eval(copies.flatMap(\.state))
            let prior = CacheStoreBudget.policy
            CacheStoreBudget.policy = CacheStorePolicy(headroomFraction: 0)
            defer { CacheStoreBudget.policy = prior }
            #expect(cacheIsPrefixViews(views) && !cacheIsPrefixViews(copies))
            #expect(CacheStoreBudget.canStore(views, ownsSnapshot: true))
            #expect(!CacheStoreBudget.canStore(copies, ownsSnapshot: true))
            // A live (not owned) view-shaped cache is still priced: only an owned view snapshot is free.
            #expect(!CacheStoreBudget.canStore(views, ownsSnapshot: false))
        }
    }

    /// Prefill one prompt through `TokenIterator` with the coordinator; return the iterator before decode.
    static func prefilled(_ model: MiniMaxM3Model, _ prompt: [Int], coordinator: CacheCoordinator, step: Int) throws
        -> TokenIterator
    {
        var parameters = GenerateParameters(maxTokens: A.maxTokens, temperature: 0)
        parameters.prefillStepSize = step
        return try TokenIterator(
            input: A.input(prompt, boundaries: [prompt.count - S.gen.count], stable: []), model: model,
            parameters: parameters, cacheCoordinator: coordinator)
    }

    @Test("Decode holds no prompt copy with views (deferred to the end of the turn); the retained copy when off (control)")
    func noPromptCopyThroughDecode() throws {
        try A.withRecorder { _ in
            let step = 320
            let dir = A.temporaryDirectory("snapshot-views")
            defer { try? FileManager.default.removeItem(at: dir) }
            let model = try S.model()
            let prompt = S.opening() + S.gen
            for viewsOn in [true, false] {
                try TokenIterator.$promptSnapshotViewsOverride.withValue(viewsOn) {
                    let coordinator = S.coordinator(dir.appendingPathComponent("\(viewsOn)"), step: step)
                    var iterator = try Self.prefilled(model, prompt, coordinator: coordinator, step: step)
                    MLX.eval(iterator.cache.flatMap(\.state))
                    if viewsOn {
                        #expect(iterator.promptCacheSnapshot == nil && iterator.deferredPromptSnapshotRows == prompt.count)
                    } else {
                        #expect(iterator.promptCacheSnapshot != nil && iterator.deferredPromptSnapshotRows == nil)
                    }
                    var out: [Int] = []
                    while let token = iterator.next() { out.append(token) }
                    iterator.storeCacheAfterGeneration(generatedTokenIds: out, includeGeneratedBoundary: false)
                    #expect(iterator.deferredPromptSnapshotRows == nil)
                    if viewsOn {
                        // The stores took views of the finished cache: the snapshot shares the live cache's buffers.
                        let snapshot = try #require(iterator.promptCacheSnapshot)
                        #expect(cacheIsPrefixViews(snapshot) && snapshot.allSatisfy { $0.offset == prompt.count })
                        MLX.eval(snapshot.flatMap(\.state))
                        for (view, live) in zip(snapshot, iterator.cache) {
                            let liveKeys = (live as? MiniMaxM3SparseCache)?.kv.keys ?? (live as? KVCacheSimple)?.keys
                            #expect(liveKeys.map { Self.address(view.state[0]) == Self.address($0) } == true)
                        }
                        #expect(iterator.cache.contains { $0 is MiniMaxM3SparseCache })
                    }
                }
            }
        }
    }

    @Test("Tool loop with view snapshots == with copies: same tokens and logits, same reuse (K_prev), cached == cold",
          arguments: [320, 2048])
    func toolLoopViewsEqualCopies(step: Int) throws {
        try A.withRecorder { recorder in
            let views = try TokenIterator.$promptSnapshotViewsOverride.withValue(true) {
                try S.toolLoop(step: step, recorder: recorder)
            }
            let copies = try TokenIterator.$promptSnapshotViewsOverride.withValue(false) {
                try S.toolLoop(step: step, recorder: recorder)
            }
            #expect(views.identical.allSatisfy { $0 } && copies.identical.allSatisfy { $0 })
            #expect(views.turns.map(\.cached) == copies.turns.map(\.cached))
            for t in 1 ..< views.turns.count {
                #expect(views.turns[t].cached == views.turns[t - 1].k, "turn \(t + 1)")
            }
            #expect(views.lastStore == copies.lastStore)
        }
    }

    /// Entries the disk tier holds once its writer is idle.
    static func diskEntries(_ coordinator: CacheCoordinator) -> Int {
        _ = coordinator.diskCache?.waitForPendingWrites()
        return coordinator.diskCache?.stores ?? -1
    }

    @Test("Aligned prefill: no post-answer entry (it can never serve more rows than the prompt boundary); written when the lever restores it (control); the next turn reuses K_prev either way")
    func postAnswerEntrySkipped() throws {
        try A.withRecorder { recorder in
            let step = 320
            var counts: [Bool: (stores: Int, reused: Int)] = [:]
            for skip in [true, false] {
                try TokenIterator.$alignedPostAnswerStoreSkippedOverride.withValue(skip) {
                    let dir = A.temporaryDirectory("post-answer-\(skip)")
                    defer { try? FileManager.default.removeItem(at: dir) }
                    let model = try S.model()
                    let coordinator = S.coordinator(dir, step: step)
                    let prompt = S.opening() + S.gen
                    var iterator = try Self.prefilled(model, prompt, coordinator: coordinator, step: step)
                    var out: [Int] = []
                    while let token = iterator.next() { out.append(token) }
                    iterator.storeCacheAfterGeneration(generatedTokenIds: out, includeGeneratedBoundary: true)
                    let stores = Self.diskEntries(coordinator)
                    // The next turn: the assistant reply + a tool result.
                    let next = S.opening() + S.message(S.ai, out.map { $0 % 990 })
                        + S.message(S.tool, S.content(211, seed: 41)) + S.gen
                    let warm = try S.run(model, next, coordinator: coordinator, step: step, recorder: recorder)
                    let cold = try S.run(model, next, coordinator: nil, step: step, recorder: recorder)
                    #expect(A.same(warm, cold), "skip \(skip): cached != cold")
                    counts[skip] = (stores, warm.cached)
                }
            }
            let on = try #require(counts[true]), off = try #require(counts[false])
            #expect(off.stores == on.stores + 1, "stores with the post-answer entry \(off.stores), without \(on.stores)")
            #expect(on.reused == S.k(S.opening() + S.gen) && on.reused == off.reused)
        }
    }

    // MARK: scheduling floor

    @Test("Floor arithmetic: raise to active + headroom when below, never lower, capped at MLX's ceiling")
    func floorArithmetic() {
        let g = 1 << 30
        #expect(SchedulingMemoryFloor.raisedLimit(active: 98 * g, limit: 97 * g, ceiling: 170 * g) == 102 * g)
        #expect(SchedulingMemoryFloor.raisedLimit(active: 95 * g, limit: 97 * g, ceiling: 170 * g) == 99 * g)
        #expect(SchedulingMemoryFloor.raisedLimit(active: 90 * g, limit: 97 * g, ceiling: 170 * g) == nil)
        #expect(SchedulingMemoryFloor.raisedLimit(active: 90 * g, limit: 120 * g, ceiling: 170 * g) == nil)
        #expect(SchedulingMemoryFloor.raisedLimit(active: 168 * g, limit: 97 * g, ceiling: 170 * g) == 170 * g)
        #expect(SchedulingMemoryFloor.raisedLimit(active: 175 * g, limit: 170 * g, ceiling: 170 * g) == nil)
        #expect(SchedulingMemoryFloor.raisedLimit(active: Int.max - 1, limit: 97 * g, ceiling: 170 * g) == 170 * g)
        #expect(SchedulingMemoryFloor.raisedLimit(active: 0, limit: 97 * g, ceiling: 170 * g) == nil)
    }

    @Test("observe(): a limit below the live set is raised above it; a higher limit is untouched; off = untouched (control)")
    func floorObserve() throws {
        try A.withRecorder { _ in
            let prior = MLX.Memory.memoryLimit
            defer { MLX.Memory.memoryLimit = prior }
            let live = MLXArray.zeros([64 << 20]).asType(.uint8) + 1
            MLX.eval(live)
            MLX.Memory.memoryLimit = 1 << 20
            let raised = SchedulingMemoryFloor.$enabledOverride.withValue(true) { SchedulingMemoryFloor.observe() }
            #expect(raised)
            #expect(MLX.Memory.memoryLimit >= min(MLX.Memory.activeMemory + SchedulingMemoryFloor.headroomBytes,
                                                  SchedulingMemoryFloor.nativeCeiling()))
            let high = MLX.Memory.memoryLimit
            #expect(!SchedulingMemoryFloor.$enabledOverride.withValue(true) { SchedulingMemoryFloor.observe() })
            #expect(MLX.Memory.memoryLimit == high)
            MLX.Memory.memoryLimit = 1 << 20
            #expect(!SchedulingMemoryFloor.$enabledOverride.withValue(false) { SchedulingMemoryFloor.observe() })
            #expect(MLX.Memory.memoryLimit == 1 << 20)
            _ = live
        }
    }

    @Test("A prefill chunk boundary (PrefillProgressReporter) raises a limit below the live set, with or without a handler")
    func floorAtPrefillChunks() throws {
        try A.withRecorder { _ in
            let prior = MLX.Memory.memoryLimit
            defer { MLX.Memory.memoryLimit = prior }
            let live = MLXArray.zeros([32 << 20]).asType(.uint8) + 1
            MLX.eval(live)
            try SchedulingMemoryFloor.$enabledOverride.withValue(true) {
                MLX.Memory.memoryLimit = 1 << 20
                PrefillProgressReporter.reportCompletedUnits(2048)
                #expect(MLX.Memory.memoryLimit > 1 << 20)
                MLX.Memory.memoryLimit = 1 << 20
                final class Seen: @unchecked Sendable { var units = 0 }
                let seen = Seen()
                PrefillProgressReporter.withHandler({ seen.units += $0 }) {
                    PrefillProgressReporter.reportCompletedUnits(512)
                }
                #expect(seen.units == 512)
                #expect(MLX.Memory.memoryLimit > 1 << 20)
            }
            _ = live
        }
    }

    @Test("BatchEngine solo generation under a limit below the live set ends with the limit above it (the loop's floor)")
    func floorInGenerationLoop() async throws {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }
        let prior = MLX.Memory.memoryLimit
        defer { MLX.Memory.memoryLimit = prior }
        let model = try S.model()
        nonisolated(unsafe) let shared = model
        let engine = A.engine(shared, coordinator: nil)
        let prompt = S.opening() + S.gen
        var parameters = GenerateParameters(maxTokens: A.maxTokens, temperature: 0)
        parameters.prefillStepSize = 320
        let before = SchedulingMemoryFloor.raises
        MLX.Memory.memoryLimit = 1 << 20
        let stream = await engine.generate(
            input: A.input(prompt, boundaries: [prompt.count - S.gen.count], stable: []), parameters: parameters)
        var events = 0
        for await _ in stream { events += 1 }
        #expect(events > 0)
        if SchedulingMemoryFloor.isEnabled {
            #expect(SchedulingMemoryFloor.raises > before)
            #expect(MLX.Memory.memoryLimit > 1 << 20)
        }
    }
}
