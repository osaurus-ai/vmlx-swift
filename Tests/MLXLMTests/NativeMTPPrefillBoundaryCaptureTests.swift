import Foundation
import MLX
import MLXNN
@testable import MLXLMCommon
import XCTest

/// Target state depends on token order, so an incorrect recurrent restore
/// changes the logits instead of passing through an all-zero-logit fixture.
private final class NativeBoundaryRecordingModel: Module, NativeMTPModel, NativeMTPMediaCapable, @unchecked Sendable {
    var nativeMTPAvailable: Bool { true }
    private(set) var forwarded: [Int] = []
    private(set) var forwardLengths: [Int] = []
    private(set) var prepareMedia: [Bool] = []
    private(set) var prepareMaskShapes: [[Int]?] = []
    private(set) var prepareMaskValues: [[Float]?] = []

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [MambaCache()] }
    func makeNativeMTPCache() -> [KVCache] { [MambaCache()] }

    static func state(after ids: [Int]) -> Int {
        ids.reduce(0) { ($0 * 7 + $1) % 997 }
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        prepareMedia.append(input.hasMediaContent)
        prepareMaskShapes.append(input.text.mask?.shape)
        prepareMaskValues.append(input.text.mask?.reshaped([-1]).asArray(Float.self))
        var tokens = input.text.tokens.reshaped([-1])
        let step = max(1, windowSize ?? 512)
        while tokens.size > step {
            _ = nativeBackboneForward(tokens[..<step].reshaped(1, step), cache: cache)
            MLX.eval(cache)
            tokens = tokens[step...]
            PrefillProgressReporter.reportCompletedUnits(input.text.tokens.size - tokens.size)
        }
        return .tokens(.init(tokens: tokens))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        nativeBackboneForward(inputs, cache: cache).logits
    }

    func nativeBackboneForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        let ids = inputs.reshaped([-1]).asArray(Int32.self).map(Int.init)
        forwarded += ids
        forwardLengths.append(ids.count)
        let recurrent = cache?.first as? MambaCache
        var value = Int(recurrent?.state.first?.asArray(Float.self).first ?? 0)
        var logits: [Float] = []
        var hidden: [Float] = []
        for token in ids {
            value = (value * 7 + token) % 997
            logits += (0..<32).map { $0 == value % 32 ? Float(10) : Float(-10) }
            hidden += [Float(value), 0, 0, 0]
        }
        if let recurrent {
            recurrent.offset += ids.count
            recurrent.state = [MLXArray([Float(value), Float(recurrent.offset)]).reshaped(1, 1, 2)]
        }
        return .init(logits: MLXArray(logits).reshaped(1, ids.count, 32),
                     hiddenStates: MLXArray(hidden).reshaped(1, ids.count, 4))
    }

    func nativeMTPForward(hiddenStates: MLXArray, nextTokenIds: MLXArray,
                          cache: [KVCache]?) -> NativeMTPForwardResult {
        // Draft work is deliberately distinct from the backbone recorder.
        let value = Int(hiddenStates.reshaped([-1]).asArray(Float.self).first ?? 0)
        let token = nextTokenIds.reshaped([-1]).item(Int.self)
        let next = (value * 7 + token) % 997
        return .init(
            logits: MLXArray((0..<32).map { $0 == next % 32 ? Float(10) : Float(-10) }).reshaped(1, 1, 32),
            hiddenStates: MLXArray([Float(next), 0, 0, 0]).reshaped(1, 1, 4))
    }
}

final class NativeMTPPrefillBoundaryCaptureTests: XCTestCase {
    private let stable = [1, 2, 3, 4, 5]
    private let user = [11, 12, 13, 14, 15, 16, 17]
    private let suffix = [201, 202, 203]
    private var roots: [URL] = []

    override func tearDown() {
        for root in roots { try? FileManager.default.removeItem(at: root) }
        roots.removeAll()
        super.tearDown()
    }

    private func coordinator(root: URL? = nil) -> CacheCoordinator {
        let directory = root ?? FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        if root == nil { roots.append(directory) }
        let result = CacheCoordinator(config: .init(usePagedCache: false, enableDiskCache: true,
            diskCacheDir: directory, modelKey: "native-boundary-recorder"))
        result.setHybrid(true)
        result.setGenPromptSuffixTokens(suffix)
        return result
    }

    private func input(_ ids: [Int]) -> LMInput {
        .init(tokens: MLXArray(ids.map(Int32.init)).reshaped(1, ids.count), tokenIds: ids,
              cachePrefixTokenCounts: [stable.count, ids.count - suffix.count],
              cacheStablePrefixTokenCounts: [stable.count])
    }

    private func parameters(step: Int = 3) -> GenerateParameters {
        // Four outputs cover initialization plus AR calibration. This suite
        // tests prefill/restore, not verification throughput or acceptance.
        var result = GenerateParameters(maxTokens: 4, temperature: 0, prefillStepSize: step)
        result.draftStrategy = .nativeMTP(depth: 1)
        return result
    }

    private func assertSnapshot(_ cache: [KVCache]?, prefix: [Int], file: StaticString = #filePath, line: UInt = #line) {
        guard let recurrent = cache?.first as? MambaCache else {
            return XCTFail("Missing recurrent boundary snapshot", file: file, line: line)
        }
        XCTAssertEqual(recurrent.offset, prefix.count, file: file, line: line)
        XCTAssertEqual(recurrent.state.first?.asArray(Float.self),
                       [Float(NativeBoundaryRecordingModel.state(after: prefix)), Float(prefix.count)], file: file, line: line)
    }

    private func assertSpilled(_ iterator: NativeMTPTokenIterator, disk: CacheCoordinator,
                               prefix: [Int], input prepared: LMInput,
                               step: Int = 512, file: StaticString = #filePath, line: UInt = #line) throws {
        XCTAssertNil(iterator.promptCacheSnapshot, file: file, line: line)
        XCTAssertNil(iterator.prefillBoundarySnapshots[prefix.count], file: file, line: line)
        XCTAssertEqual(iterator.spilledHybridStripBoundary, prefix.count, file: file, line: line)
        let salt = computeCacheSalt(for: prepared, parameters: parameters(step: step))
        let arrays = try XCTUnwrap(disk.diskCache?.fetch(tokens: prefix, mediaSalt: salt), file: file, line: line)
        var restored: [KVCache] = [MambaCache()]
        XCTAssertEqual(restoreFromDiskArrays(arrays, into: &restored), prefix.count, file: file, line: line)
        assertSnapshot(restored, prefix: prefix, file: file, line: line)
    }

    func testStableSeedAndCanonicalBoundaryAreCapturedInsideThePrefillHeadWithoutReplayAtStore() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        for step in [3, 512] {
            let model = NativeBoundaryRecordingModel()
            let prompt = stable + user + suffix
            let disk = coordinator()
            var iterator = try NativeMTPTokenIterator(input: input(prompt), model: model,
                parameters: parameters(step: step), depth: 1, cacheCoordinator: disk)
            XCTAssertEqual(Array(model.forwarded.prefix(prompt.count)), prompt)
            XCTAssertEqual(model.forwarded.count, prompt.count + 1)
            // Same segmentation as the AR iterator's stable-head capture: the
            // head splits once at the stable N-1 seed (4) and once at the
            // canonical boundary (12). No single-row forward is introduced —
            // N itself (5) is not captured — and every token is still
            // forwarded exactly once, in order.
            XCTAssertEqual(model.forwardLengths, step == 3 ? [3, 1, 3, 3, 2, 3, 1] : [4, 8, 3, 1])
            XCTAssertEqual(Set(iterator.prefillBoundarySnapshots.keys), Set([stable.count - 1]))
            assertSnapshot(iterator.prefillBoundarySnapshots[stable.count - 1], prefix: Array(stable.dropLast()))
            try assertSpilled(iterator, disk: disk, prefix: stable + user, input: input(prompt), step: step)
            let before = model.forwarded.count
            iterator.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
            // The terminal store consumes both captured snapshots; nothing is
            // replayed through the model after generation.
            XCTAssertEqual(Array(model.forwarded.dropFirst(before)), [])
            let salt = computeCacheSalt(for: input(prompt), parameters: parameters(step: step))
            XCTAssertTrue(disk.hasDurableDiskEntry(tokens: Array(stable.dropLast()), mediaSalt: salt),
                          "the stable N-1 seed must reach disk from the captured snapshot")
            XCTAssertTrue(disk.hasDurableDiskEntry(tokens: stable + user, mediaSalt: salt),
                          "the canonical boundary must reach disk from the captured snapshot")
            // Exactly one disk store per boundary: the seed from the history
            // loop and the canonical boundary from the stripped store. The
            // history loop must not store the canonical boundary as well.
            XCTAssertEqual(disk.snapshotStats().diskStats?.stores, 2)
            XCTAssertEqual(disk.snapshotStats().diskStats?.storeSkips, 0)
        }
    }

    func testAlreadyDurableStableSeedIsNotCapturedAndNotReplayed() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let prompt = stable + user + suffix
        let disk = coordinator()
        disk.setHybrid(true, requiresRecurrentSSMCompanion: true, requiresSeparateRecurrentPayload: false)
        let salt = computeCacheSalt(for: input(prompt), parameters: parameters(step: 512))
        let seedTokens = Array(stable.dropLast())
        let seedModel = NativeBoundaryRecordingModel()
        let seedCache = seedModel.newCache(parameters: parameters())
        _ = seedModel.nativeBackboneForward(
            MLXArray(seedTokens.map(Int32.init)).reshaped(1, seedTokens.count), cache: seedCache)
        MLX.eval(seedCache)
        disk.storeAfterGeneration(promptTokens: seedTokens, perLayerData: [nil],
            ssmStates: extractSSMStates(from: seedCache), cache: seedCache, mediaSalt: salt, isStableRoot: true)
        XCTAssertTrue(disk.hasDurableDiskEntry(tokens: seedTokens, mediaSalt: salt))

        // The forced-fresh policy keeps the durable seed from being restored,
        // so the cold head still crosses it and the capture decision is the
        // only thing under test.
        let fresh = LMInput(tokens: MLXArray(prompt.map(Int32.init)).reshaped(1, prompt.count),
            tokenIds: prompt, cachePrefixTokenCounts: [stable.count, prompt.count - suffix.count],
            cacheStablePrefixTokenCounts: [stable.count], cacheRestorePolicy: .freshRequiredToolSelection)
        let model = NativeBoundaryRecordingModel()
        var iterator = try NativeMTPTokenIterator(input: fresh, model: model,
            parameters: parameters(step: 512), depth: 1, cacheCoordinator: disk)
        // A durable seed is not worth a retained copy: the head stays one
        // segment up to the canonical boundary, exactly as before.
        XCTAssertEqual(model.forwardLengths, [12, 3, 1])
        XCTAssertNil(iterator.prefillBoundarySnapshots[stable.count - 1])
        try assertSpilled(iterator, disk: disk, prefix: stable + user, input: fresh)
        let before = model.forwarded.count
        iterator.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
        XCTAssertEqual(Array(model.forwarded.dropFirst(before)), [],
                       "a validated durable seed is skipped by the store loop, never replayed")
    }

    func testStoreSeedsInsidePrefillHeadRule() {
        // Recurrent topology: the seed is N-1; seeds at/below the restored
        // prefix and at/above the head end are left to their own paths
        // (5770 is the whole prompt; 5000's seed 4999 is still inside the
        // head that ends at 5000).
        let seeds = NativeMTPTokenIterator.storeSeedsInsidePrefillHead(
            stablePrefixTokenCounts: [1, 5, 2229, 4316, 5000, 5770],
            promptTokenCount: 5770, restored: 0, headEnd: 5000,
            requiresRecurrentSSMCompanion: true, isDurable: { _ in false })
        XCTAssertEqual(seeds, Set([1, 4, 2228, 4315, 4999]))
        // Durable seeds are excluded individually.
        let partiallyDurable = NativeMTPTokenIterator.storeSeedsInsidePrefillHead(
            stablePrefixTokenCounts: [2229, 4316], promptTokenCount: 5770, restored: 0,
            headEnd: 5769, requiresRecurrentSSMCompanion: true, isDurable: { $0 == 2228 })
        XCTAssertEqual(partiallyDurable, Set([4315]))
        // A warm turn restored past the seeds captures nothing for them.
        let warm = NativeMTPTokenIterator.storeSeedsInsidePrefillHead(
            stablePrefixTokenCounts: [2229, 4316], promptTokenCount: 5770, restored: 4792,
            headEnd: 5769, requiresRecurrentSSMCompanion: true, isDurable: { _ in false })
        XCTAssertEqual(warm, [])
        // Non-recurrent topologies store the exact boundary N.
        let exact = NativeMTPTokenIterator.storeSeedsInsidePrefillHead(
            stablePrefixTokenCounts: [2229, 4316], promptTokenCount: 5770, restored: 0,
            headEnd: 5769, requiresRecurrentSSMCompanion: false, isDurable: { _ in false })
        XCTAssertEqual(exact, Set([2229, 4316]))
    }

    func testLaterStableBoundariesCaptureNMinusOneAndNRegardlessOfValidatedDiskEntry() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        // With no processor canonical prefix, the suffix heuristic marks 6.
        // A later stable boundary at 10 exercises AR's [N-1, N] tail ordering.
        let ids = [1, 2, 3, 4, 5, 6, 201, 7, 8, 9, 10, 11, 12]
        let prepared = LMInput(tokens: MLXArray(ids.map(Int32.init)).reshaped(1, ids.count),
            tokenIds: ids, cacheStablePrefixTokenCounts: [10],
            cacheRestorePolicy: .freshRequiredToolSelection)
        for prevalidated in [false, true] {
            let disk = coordinator()
            if prevalidated {
                let seedModel = NativeBoundaryRecordingModel()
                let seedCache = seedModel.newCache(parameters: parameters())
                _ = seedModel.nativeBackboneForward(
                    MLXArray(ids.prefix(9).map(Int32.init)).reshaped(1, 9), cache: seedCache)
                MLX.eval(seedCache)
                let salt = computeCacheSalt(for: prepared, parameters: parameters(step: 512))
                disk.storeAfterGeneration(promptTokens: Array(ids.prefix(9)),
                    perLayerData: [nil], ssmStates: extractSSMStates(from: seedCache), cache: seedCache,
                    mediaSalt: salt, isStableRoot: true)
                XCTAssertTrue(disk.hasValidatedDiskEntry(tokens: Array(ids.prefix(9)), mediaSalt: salt))
            }
            let model = NativeBoundaryRecordingModel()
            let iterator = try NativeMTPTokenIterator(input: prepared, model: model,
                parameters: parameters(step: 512), depth: 1, cacheCoordinator: disk)
            // The forced-fresh policy deliberately prevents a restore from
            // hiding whether validation changes cold-prefill segmentation.
            XCTAssertEqual(model.forwardLengths, [6, 3, 1, 3, 1])
            XCTAssertEqual(Array(model.forwarded.prefix(ids.count)), ids)
            XCTAssertEqual(Set(iterator.prefillBoundarySnapshots.keys), Set([9, 10]))
            try assertSpilled(iterator, disk: disk, prefix: Array(ids.prefix(6)), input: prepared)
            for count in [9, 10] {
                assertSnapshot(iterator.prefillBoundarySnapshots[count], prefix: Array(ids.prefix(count)))
            }
        }
    }

    func testCapturedRecurrentSnapshotsRemainIndependentOfDecode() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let disk = coordinator()
        var iterator = try NativeMTPTokenIterator(input: input(stable + user + suffix),
            model: NativeBoundaryRecordingModel(), parameters: parameters(), depth: 1,
            cacheCoordinator: disk)
        while iterator.next() != nil {}
        assertSnapshot(iterator.prefillBoundarySnapshots[stable.count - 1], prefix: Array(stable.dropLast()))
        try assertSpilled(iterator, disk: disk, prefix: stable + user, input: input(stable + user + suffix))
    }

    func testFreshDiskCoordinatorContinuationMatchesColdStateAndTokens() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let firstCoordinator = coordinator()
        let root = try XCTUnwrap(roots.last)
        let first = stable + user + suffix
        let firstModel = NativeBoundaryRecordingModel()
        var initial = try NativeMTPTokenIterator(input: input(first), model: firstModel,
            parameters: parameters(), depth: 1, cacheCoordinator: firstCoordinator)
        initial.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
        let restoredCoordinator = coordinator(root: root)
        let next = stable + user + [23, 24, 25] + suffix
        let salt = computeCacheSalt(for: input(next), parameters: parameters())
        guard case .hit(let matched, _, let detail, _, _, _) = restoredCoordinator.fetch(
            tokens: next, mediaSalt: salt, skipExactDiskBoundary: true)
        else { return XCTFail("Fresh coordinator must restore the persisted stripped boundary") }
        XCTAssertEqual(detail, .disk)
        XCTAssertEqual(matched, stable.count + user.count)
        let warmModel = NativeBoundaryRecordingModel()
        let progressRows = ProgressRows()
        var warm = try NativeMTPTokenIterator(input: input(next), model: warmModel,
            parameters: parameters(), depth: 1, cacheCoordinator: restoredCoordinator,
            prefillProgressHandler: { progressRows.append($0) })
        let progress = progressRows.snapshot
        XCTAssertEqual(progress.first { $0.stage == .cacheRestore }?.completedUnitCount, matched)
        let prefill = progress.filter { $0.stage == .prefill }
        XCTAssertEqual(prefill.first?.completedUnitCount, matched)
        XCTAssertEqual(prefill.last?.completedUnitCount, next.count)
        XCTAssertEqual(prefill.map(\.completedUnitCount), prefill.map(\.completedUnitCount).sorted())
        var cold = try NativeMTPTokenIterator(input: input(next), model: NativeBoundaryRecordingModel(),
            parameters: parameters(), depth: 1)
        XCTAssertEqual(Array(warmModel.forwarded.prefix(next.count - matched)), Array(next.dropFirst(matched)))
        XCTAssertEqual(warmModel.forwarded.count, next.count - matched + 1)
        var warmIDs: [Int] = []; var coldIDs: [Int] = []
        while let token = warm.next() { warmIDs.append(token) }
        while let token = cold.next() { coldIDs.append(token) }
        XCTAssertEqual(warmIDs, coldIDs)
        XCTAssertEqual(warmIDs.count, 4)
        XCTAssertEqual(warm.cache[0].offset, cold.cache[0].offset)
        XCTAssertEqual(warm.cache[0].state[0].asArray(Float.self), cold.cache[0].state[0].asArray(Float.self))
    }

    func testReloadValidatesExactStableCheckpointWithoutBackboneReplay() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        for scenario in 0..<3 {
            let rejected = scenario != 0
            let writer = coordinator()
            writer.setHybrid(true, requiresRecurrentSSMCompanion: true,
                requiresSeparateRecurrentPayload: false)
            let root = try XCTUnwrap(roots.last)
            var first = try NativeMTPTokenIterator(input: input(stable + user + suffix),
                model: NativeBoundaryRecordingModel(), parameters: parameters(), depth: 1,
                cacheCoordinator: writer)
            first.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)

            let reader = coordinator(root: root)
            reader.setHybrid(true, requiresRecurrentSSMCompanion: true,
                requiresSeparateRecurrentPayload: false)
            let next = stable + user + [23, 24, 25] + suffix
            let salt = computeCacheSalt(for: input(next), parameters: parameters())
            let stableTokens = Array(stable.dropLast())
            XCTAssertFalse(reader.hasValidatedDiskEntry(tokens: stableTokens, mediaSalt: salt))
            let model = NativeBoundaryRecordingModel()
            let priorRecency = try XCTUnwrap(reader.diskCache!.quotaEntries()
                .first { $0.tokenCount == stableTokens.count }?.createdAt)
            var warm = try NativeMTPTokenIterator(input: input(next), model: model,
                parameters: parameters(), depth: 1, cacheCoordinator: reader)
            let retainedRecency = try XCTUnwrap(reader.diskCache!.quotaEntries()
                .first { $0.tokenCount == stableTokens.count }?.createdAt)
            XCTAssertGreaterThan(retainedRecency, priorRecency,
                "retaining the longer checkpoint must touch the stable N-1 seed")
            if scenario == 1 {
                guard case .arrays = reader.diskCache!.fetchCandidate(tokens: stableTokens, mediaSalt: salt)
                else { return XCTFail("stable checkpoint missing") }
                _ = reader.diskCache!.markRestoreRejected(tokens: stableTokens, mediaSalt: salt,
                    countedHit: false)
            } else if scenario == 2 {
                let row = try XCTUnwrap(reader.diskCache!.quotaEntries()
                    .first { $0.tokenCount == stableTokens.count })
                let badModel = NativeBoundaryRecordingModel()
                let badCache = badModel.newCache(parameters: parameters())
                _ = badModel.nativeBackboneForward(
                    MLXArray(stableTokens.map(Int32.init)).reshaped(1, stableTokens.count), cache: badCache)
                let badRecurrent = try XCTUnwrap(badCache.first as? MambaCache)
                badRecurrent.offset = stableTokens.count - 1
                MLX.eval(badCache)
                try MLX.save(arrays: TQDiskSerializer.serialize(cache: badCache),
                    url: reader.diskCache!.cacheDir.appendingPathComponent(row.hash + ".safetensors"))
            }
            let before = model.forwarded.count
            let hitsBefore = reader.diskCache!.snapshotStats().hits
            let offsetBefore = warm.cache[0].offset
            let stateBefore = warm.cache[0].state[0].asArray(Float.self)
            warm.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
            XCTAssertEqual(Array(model.forwarded.dropFirst(before)), rejected ? stableTokens : [])
            XCTAssertEqual(warm.cache[0].offset, offsetBefore)
            XCTAssertEqual(warm.cache[0].state[0].asArray(Float.self), stateBefore)
            XCTAssertEqual(reader.diskCache!.snapshotStats().hits, hitsBefore,
                "validation-only fetch/rejection must not change accepted-hit telemetry")
            XCTAssertTrue(reader.hasValidatedDiskEntry(tokens: stableTokens, mediaSalt: salt))
        }
    }

    func testSingleTokenStableCheckpointRecencyMatchesStorageOffset() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let disk = coordinator()
        disk.setHybrid(true, requiresRecurrentSSMCompanion: true,
            requiresSeparateRecurrentPayload: false)
        let model = NativeBoundaryRecordingModel()
        let cache = model.newCache(parameters: parameters())
        _ = model.nativeBackboneForward(MLXArray([Int32(1)]).reshaped(1, 1), cache: cache)
        disk.storeAfterGeneration(promptTokens: [1], perLayerData: [nil],
            ssmStates: nil, cache: cache, isStableRoot: true)
        let before = try XCTUnwrap(disk.diskCache!.quotaEntries().first?.createdAt)
        disk.touchStableDiskCheckpointsAfterRetainedRestore(requestTokens: [1, 2],
            matchedTokenCount: 1, preferredDiskBoundaries: [1],
            skipExactDiskBoundary: true, mediaSalt: nil)
        let after = try XCTUnwrap(disk.diskCache!.quotaEntries().first?.createdAt)
        XCTAssertGreaterThan(after, before)
    }

    func testMediaDoesNotAcquireTextOnlyBoundarySplits() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let model = NativeBoundaryRecordingModel()
        let ids = stable + user + suffix
        let mediaInput = LMInput(text: .init(tokens: MLXArray(ids.map(Int32.init)).reshaped(1, ids.count), tokenIds: ids),
                                image: .init(pixels: MLXArray.zeros([1, 3, 2, 2])), mediaTokenIds: [1],
                                cachePrefixTokenCounts: [stable.count, ids.count - suffix.count],
                                cacheStablePrefixTokenCounts: [stable.count])
        let iterator = try NativeMTPTokenIterator(input: mediaInput, model: model,
            parameters: parameters(step: 512), depth: 1, cacheCoordinator: coordinator())
        XCTAssertEqual(model.prepareMedia, [true])
        XCTAssertTrue(iterator.prefillBoundarySnapshots.isEmpty)
        XCTAssertEqual(Array(model.forwarded.prefix(ids.count)), ids)
    }

    private final class ProgressRows: @unchecked Sendable {
        private let lock = NSLock()
        private var rows: [PrefillProgress] = []
        func append(_ row: PrefillProgress) { lock.lock(); defer { lock.unlock() }; rows.append(row) }
        var snapshot: [PrefillProgress] { lock.lock(); defer { lock.unlock() }; return rows }
    }

    func testPrefillProgressPreservesTokensAndSegmentBoundaries() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let ids = stable + user + suffix
        for withCoordinator in [false, true] {
            let reference = NativeBoundaryRecordingModel()
            var plain = try NativeMTPTokenIterator(input: input(ids), model: reference,
                parameters: parameters(), depth: 1,
                cacheCoordinator: withCoordinator ? coordinator() : nil)
            let observed = NativeBoundaryRecordingModel()
            let rows = ProgressRows()
            var reporting = try NativeMTPTokenIterator(input: input(ids), model: observed,
                parameters: parameters(), depth: 1,
                cacheCoordinator: withCoordinator ? coordinator() : nil,
                prefillProgressHandler: { rows.append($0) })
            XCTAssertEqual(observed.forwardLengths, reference.forwardLengths)
            XCTAssertEqual(observed.forwarded, reference.forwarded)
            let progress = rows.snapshot
            XCTAssertTrue(progress.contains { $0.stage == .cacheLookup })
            let prefill = progress.filter { $0.stage == .prefill }
            XCTAssertTrue(prefill.contains { $0.completedUnitCount > 0 && $0.completedUnitCount < ids.count })
            XCTAssertEqual(prefill.last?.completedUnitCount, ids.count)
            XCTAssertTrue(prefill.allSatisfy { $0.totalUnitCount == ids.count })
            XCTAssertEqual(prefill.map(\.completedUnitCount), prefill.map(\.completedUnitCount).sorted())
            for _ in 0..<4 {
                XCTAssertEqual(reporting.next(), plain.next())
            }
        }
    }

    func testNoCoordinatorDoesNotSplitPreparation() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let model = NativeBoundaryRecordingModel()
        let ids = stable + user + suffix
        let iterator = try NativeMTPTokenIterator(input: input(ids), model: model,
            parameters: parameters(step: 512), depth: 1)
        XCTAssertEqual(model.prepareMedia, [false])
        XCTAssertTrue(iterator.prefillBoundarySnapshots.isEmpty)
        XCTAssertEqual(Array(model.forwarded.prefix(ids.count)), ids)
    }

    func testAuxiliaryPromptDoesNotCaptureOrPersistBoundaries() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let model = NativeBoundaryRecordingModel()
        let ids = stable + user + suffix
        let auxiliary = LMInput(tokens: MLXArray(ids.map(Int32.init)).reshaped(1, ids.count),
            tokenIds: ids, cachePrefixTokenCounts: [stable.count, ids.count - suffix.count],
            cacheStablePrefixTokenCounts: [stable.count], cachePromptIntent: .auxiliary)
        let disk = coordinator()
        var iterator = try NativeMTPTokenIterator(input: auxiliary, model: model,
            parameters: parameters(step: 512), depth: 1, cacheCoordinator: disk)
        XCTAssertEqual(model.prepareMedia, [false])
        XCTAssertTrue(iterator.prefillBoundarySnapshots.isEmpty)
        let before = model.forwarded
        iterator.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
        XCTAssertEqual(model.forwarded, before)
        XCTAssertEqual(disk.snapshotStats().diskStats?.stores, 0)
    }

    func testUnsupportedAttentionMaskPassesToPrepareUnchanged() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let model = NativeBoundaryRecordingModel()
        let ids = stable + user + suffix
        // A full attention mask is not a token-aligned 1D/2D padding mask.
        // The recorder verifies routing, not attention-mask mathematics.
        let values = (0..<(ids.count * ids.count)).map(Float.init)
        let masked = LMInput(tokens: MLXArray(ids.map(Int32.init)).reshaped(1, ids.count),
            mask: MLXArray(values).reshaped(1, ids.count, ids.count), tokenIds: ids,
            cachePrefixTokenCounts: [stable.count, ids.count - suffix.count],
            cacheStablePrefixTokenCounts: [stable.count])
        let iterator = try NativeMTPTokenIterator(input: masked, model: model,
            parameters: parameters(step: 512), depth: 1, cacheCoordinator: coordinator())
        XCTAssertEqual(model.prepareMedia, [false])
        XCTAssertEqual(model.prepareMaskShapes, [[1, ids.count, ids.count]])
        XCTAssertEqual(model.prepareMaskValues, [values])
        XCTAssertTrue(iterator.prefillBoundarySnapshots.isEmpty)
        XCTAssertEqual(Array(model.forwarded.prefix(ids.count)), ids)
    }
}
