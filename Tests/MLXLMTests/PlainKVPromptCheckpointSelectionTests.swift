import Foundation
import Dispatch
import MLX
import MLXLLM
import MLXNN
import XCTest

@testable import MLXLMCommon

/// Exercises TokenIterator -> coordinator -> real safetensors writer and
/// restart restore with a real small Llama, not fabricated write counters.
final class PlainKVPromptCheckpointSelectionTests: XCTestCase {
    private final class Writes: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0
        func record() { lock.lock(); count += 1; lock.unlock() }
        var value: Int { lock.lock(); defer { lock.unlock() }; return count }
    }
    private let history = [11, 12, 13, 14, 15, 16]
    private let rail = [61, 62, 63]
    private var parameters: GenerateParameters {
        GenerateParameters(maxTokens: 3, temperature: 0, prefillStepSize: 3)
    }
    private func model() -> LlamaModel {
        MLXRandom.seed(187)
        let model = LlamaModel(LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 2, intermediateSize: 128,
            attentionHeads: 8, rmsNormEps: 1e-5, vocabularySize: 128, kvHeads: 4))
        MLX.eval(model)
        return model
    }
    private func directory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("plain-kv-checkpoints-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
    private func coordinator(_ root: URL, suffix: [Int], behind: Bool = false, enabled: Bool = true) -> CacheCoordinator {
        var config = CacheCoordinatorConfig(
            usePagedCache: false, enableDiskCache: enabled, diskCacheDir: root, modelKey: "plain-kv-checkpoint")
        config.diskWriteBehind = behind
        let coordinator = CacheCoordinator(config: config)
        coordinator.setGenPromptSuffixTokens(suffix)
        return coordinator
    }
    private func input(_ tokens: [Int]? = nil, boundaries: [Int]? = nil) -> LMInput {
        let ids = tokens ?? history + rail
        return LMInput(tokens: MLXArray(ids.map(Int32.init)).reshaped(1, ids.count), tokenIds: ids,
            cachePrefixTokenCounts: boundaries ?? [history.count])
    }
    private func payloads(_ root: URL) throws -> [URL] {
        try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
            .filter { DiskCache.isPublishedPayloadName($0.lastPathComponent) }
    }
    private func bytes(_ files: [URL]) throws -> Int {
        try files.reduce(0) { result, file in
            result + ((try FileManager.default.attributesOfItem(atPath: file.path)[.size] as? NSNumber)?.intValue ?? 0)
        }
    }
    private func assertSameState(_ lhs: [KVCache], _ rhs: [KVCache], file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(lhs.count, rhs.count, file: file, line: line)
        for (a, b) in zip(lhs, rhs) {
            XCTAssertEqual(a.offset, b.offset, file: file, line: line)
            XCTAssertEqual(a.state.count, b.state.count, file: file, line: line)
            for (x, y) in zip(a.state, b.state) {
                XCTAssertEqual(x.dtype, y.dtype, file: file, line: line)
                XCTAssertEqual(x.shape, y.shape, file: file, line: line)
                XCTAssertTrue(arrayEqual(x, y).item(Bool.self), file: file, line: line)
            }
        }
    }

    func testRealWriterElidesOnlyTheExactPromptAndRestartReplaysItsActualRail() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let selectedRoot = try directory(), controlRoot = try directory()
        defer { try? FileManager.default.removeItem(at: selectedRoot); try? FileManager.default.removeItem(at: controlRoot) }
        let model = model(), prepared = input()
        let chosen = coordinator(selectedRoot, suffix: rail, behind: true)
        let control = coordinator(controlRoot, suffix: [], behind: true) // Unknown rail keeps the original policy.
        let selectedWrites = Writes(), controlWrites = Writes()
        chosen.diskCache?.publishFaultForTesting = { selectedWrites.record() }
        control.diskCache?.publishFaultForTesting = { controlWrites.record() }
        var first = try TokenIterator(input: prepared, model: model, parameters: parameters, cacheCoordinator: chosen)
        var original = try TokenIterator(input: prepared, model: model, parameters: parameters, cacheCoordinator: control)
        first.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
        original.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
        XCTAssertTrue(chosen.waitForPendingDiskWrites())
        XCTAssertTrue(control.waitForPendingDiskWrites())
        XCTAssertEqual(selectedWrites.value, 1)
        XCTAssertEqual(controlWrites.value, 2)
        let selectedFiles = try payloads(selectedRoot), controlFiles = try payloads(controlRoot)
        XCTAssertEqual(selectedFiles.count, 1)
        XCTAssertEqual(controlFiles.count, 2)
        XCTAssertLessThan(try bytes(selectedFiles), try bytes(controlFiles))
        let saved = try Data(contentsOf: XCTUnwrap(selectedFiles.first))
        XCTAssertTrue(try controlFiles.contains { try Data(contentsOf: $0) == saved }, "The common checkpoint's format and bytes are unchanged")
        let salt = computeCacheSalt(for: prepared, parameters: parameters)
        XCTAssertTrue(chosen.hasDurableDiskEntry(tokens: history, mediaSalt: salt))
        XCTAssertFalse(chosen.hasDurableDiskEntry(tokens: history + rail, mediaSalt: salt))

        // New coordinator means no in-memory hit or process validation shortcut.
        let restarted = coordinator(selectedRoot, suffix: rail)
        var restored = try TokenIterator(input: prepared, model: model, parameters: parameters, cacheCoordinator: restarted)
        XCTAssertEqual(restarted.snapshotStats().diskStats?.hits, 1)
        assertSameState(restored.cache, first.cache)
        let token = LMInput.Text(tokens: MLXArray([Int32(27)]).reshaped(1, 1))
        let actual = model(token, cache: restored.cache.map { $0.copy() }, state: nil).logits
        let expected = model(token, cache: first.cache.map { $0.copy() }, state: nil).logits
        XCTAssertTrue(arrayEqual(actual, expected).item(Bool.self), "Identical restored state must yield identical next logits")
        XCTAssertEqual(restored.next(), first.next())
    }

    func testEarlierPrefixesGrowingAndToolResultForksRestoreEquivalentState() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let model = model(), prepared = input(boundaries: [3, history.count])
        let disk = coordinator(root, suffix: rail)
        var initial = try TokenIterator(input: prepared, model: model, parameters: parameters, cacheCoordinator: disk)
        initial.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
        let salt = computeCacheSalt(for: prepared, parameters: parameters)
        XCTAssertTrue(disk.hasDurableDiskEntry(tokens: Array(history.prefix(3)), mediaSalt: salt))
        XCTAssertTrue(disk.hasDurableDiskEntry(tokens: history, mediaSalt: salt))
        // Includes a changed suffix, an earlier-turn fork, and additional tool/result-like turns.
        for tokens in [history + rail + [8, 9], history + [71, 72, 73], Array(history.prefix(3)) + [24, 25, 26]] {
            let request = input(tokens, boundaries: [])
            var warm = try TokenIterator(input: request, model: model, parameters: parameters,
                cacheCoordinator: coordinator(root, suffix: rail))
            var cold = try TokenIterator(input: request, model: model, parameters: parameters)
            assertSameState(warm.cache, cold.cache)
            XCTAssertEqual(warm.next(), cold.next())
        }
    }

    func testFinalAnswerCheckpointStillPersists() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let disk = coordinator(root, suffix: rail), model = model(), prepared = input()
        var iterator = try TokenIterator(input: prepared, model: model, parameters: parameters, cacheCoordinator: disk)
        let generated = [try XCTUnwrap(iterator.next()), try XCTUnwrap(iterator.next())]
        iterator.storeCacheAfterGeneration(generatedTokenIds: generated, includeGeneratedBoundary: true)
        let salt = computeCacheSalt(for: prepared, parameters: parameters)
        XCTAssertTrue(disk.hasDurableDiskEntry(tokens: history, mediaSalt: salt))
        XCTAssertTrue(disk.hasDurableDiskEntry(tokens: history + rail + generated, mediaSalt: salt))
        XCTAssertEqual(try payloads(root).count, 2)
    }

    func testPublicationFailureLeavesAnHonestMissWithoutASecondFullWrite() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let disk = coordinator(root, suffix: rail, behind: true), writes = Writes(), prepared = input()
        disk.diskCache?.publishFaultForTesting = { writes.record(); throw CocoaError(.fileWriteUnknown) }
        var iterator = try TokenIterator(input: prepared, model: model(), parameters: parameters, cacheCoordinator: disk)
        iterator.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
        XCTAssertTrue(disk.waitForPendingDiskWrites())
        XCTAssertEqual(writes.value, 1)
        XCTAssertEqual(try payloads(root).count, 0)
        XCTAssertEqual(disk.snapshotStats().diskStats?.currentEntryCount, 0)
        if case .hit = disk.fetch(tokens: history + rail, mediaSalt: computeCacheSalt(for: prepared, parameters: parameters)) {
            XCTFail("Failed publication cannot become a cache hit")
        }
    }

    func testPublicationFailurePreservesAnEarlierDurableFallbackAcrossRestart() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let disk = coordinator(root, suffix: rail, behind: true), model = model()
        let earlier = Array(history.prefix(3))
        var seed = try TokenIterator(input: input(earlier + rail, boundaries: [earlier.count]),
            model: model, parameters: parameters, cacheCoordinator: disk)
        seed.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
        XCTAssertTrue(disk.waitForPendingDiskWrites())
        let writes = Writes()
        disk.diskCache?.publishFaultForTesting = { writes.record(); throw CocoaError(.fileWriteUnknown) }
        var later = try TokenIterator(input: input(), model: model, parameters: parameters, cacheCoordinator: disk)
        later.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
        XCTAssertTrue(disk.waitForPendingDiskWrites())
        XCTAssertEqual(writes.value, 1)
        XCTAssertEqual(try payloads(root).count, 1)
        let restarted = coordinator(root, suffix: rail)
        let salt = computeCacheSalt(for: input(), parameters: parameters)
        XCTAssertTrue(restarted.hasDurableDiskEntry(tokens: earlier, mediaSalt: salt))
        XCTAssertFalse(restarted.hasDurableDiskEntry(tokens: history, mediaSalt: salt))
        var restored = try TokenIterator(input: input(), model: model, parameters: parameters, cacheCoordinator: restarted)
        var cold = try TokenIterator(input: input(), model: model, parameters: parameters)
        XCTAssertEqual(restarted.snapshotStats().diskStats?.hits, 1)
        assertSameState(restored.cache, cold.cache)
        XCTAssertEqual(restored.next(), cold.next())
    }

    func testWriteBehindAndClearKeepTheirExistingOwnershipContract() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let disk = coordinator(root, suffix: rail, behind: true), prepared = input()
        let entered = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        let publicationTimeouts = Writes()
        defer { release.signal() }
        disk.diskCache?.publishFaultForTesting = {
            entered.signal()
            if release.wait(timeout: .now() + 10) == .timedOut { publicationTimeouts.record() }
        }
        var iterator = try TokenIterator(input: prepared, model: model(), parameters: parameters, cacheCoordinator: disk)
        iterator.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
        XCTAssertEqual(entered.wait(timeout: .now() + 10), .success)
        XCTAssertEqual(publicationTimeouts.value, 0, "Store returned while publication was held; it must not await durability")
        release.signal()
        disk.clear()
        XCTAssertEqual(disk.diskCache?.pendingWriteCount, 0)
        XCTAssertEqual(try payloads(root).count, 0)
        if case .hit = disk.fetch(tokens: history + rail, mediaSalt: computeCacheSalt(for: prepared, parameters: parameters)) {
            XCTFail("Clear cannot leave the selected checkpoint available")
        }
    }

    func testDisabledDiskWarmupAndUnprovenRailKeepTheirOriginalPolicy() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        for variant in ["disabled", "warmup", "unknown", "mismatch", "not-declared"] {
            let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
            let suffix = variant == "unknown" ? [] : variant == "mismatch" ? [1, 2, 3] : rail
            let disk = coordinator(root, suffix: suffix, enabled: variant != "disabled")
            let ids = history + rail
            let request = LMInput(tokens: MLXArray(ids.map(Int32.init)).reshaped(1, ids.count), tokenIds: ids,
                cachePrefixTokenCounts: variant == "not-declared" ? [] : [history.count],
                cachePromptIntent: variant == "warmup" ? .reusablePrefixWarmup : .generation)
            var iterator = try TokenIterator(input: request, model: model(), parameters: parameters, cacheCoordinator: disk)
            iterator.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
            if variant == "disabled" { XCTAssertTrue(try payloads(root).isEmpty) }
            else {
                XCTAssertTrue(disk.hasDurableDiskEntry(tokens: ids, mediaSalt: computeCacheSalt(for: request, parameters: parameters)), variant)
            }
        }
    }

    func testTopologyMediaDraftAndMismatchedStateCannotSelectTheShorterCheckpoint() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let disk = coordinator(root, suffix: rail), prepared = input(), ids = history + rail
        let model = model()
        let iterator = try TokenIterator(input: prepared, model: model, parameters: parameters)
        let snapshot = iterator.cache.map { $0.copy() }
        XCTAssertNotNil(TokenIterator.shorterPlainKVPromptCheckpoint(
            coordinator: disk, promptTokenIds: ids, input: prepared, parameters: parameters,
            hasAuxiliaryState: false, snapshot: snapshot))
        for variant in ["rotating", "mamba", "offset", "media", "draft", "auxiliary", "hybrid", "companion", "affine", "quantized", "compiled"] {
            let candidate = coordinator(root, suffix: rail)
            var params = parameters
            var state = snapshot.map { $0.copy() }
            var request = prepared
            if variant == "rotating" { state = [RotatingKVCache(maxSize: 32)] }
            if variant == "mamba" { state = [MambaCache()] }
            if variant == "offset" { _ = state[0].trim(1) }
            if variant == "media" {
                request = LMInput(text: prepared.text, image: .init(pixels: MLXArray.zeros([1, 2, 2, 3])),
                    cachePrefixTokenCounts: [history.count])
            }
            if variant == "draft" { params.draftStrategy = .nativeMTP(depth: 1, verifierMode: nil) }
            if variant == "affine" { params.kvMode = .affine }
            if variant == "quantized" { params.kvBits = 4 }
            if variant == "hybrid" { candidate.setHybrid(true, requiresRecurrentSSMCompanion: false) }
            if variant == "companion" { candidate.setPagedBoundaryCompanionRequired(true) }
            XCTAssertNil(TokenIterator.shorterPlainKVPromptCheckpoint(
                coordinator: candidate, promptTokenIds: ids, input: request, parameters: params,
                hasAuxiliaryState: variant == "auxiliary", hasCompiledForward: variant == "compiled",
                snapshot: state), variant)
        }
    }

    func testSelectionUsesTheWholeRealRailWithoutAnArbitraryLengthCutoff() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        for count in [1, 3, 33] {
            let suffix = Array(40 ..< (40 + count)), ids = history + suffix
            let disk = coordinator(root, suffix: suffix)
            let state = KVCacheSimple()
            let values = MLXArray(0 ..< ids.count * 4).reshaped(1, 1, ids.count, 4).asType(.bfloat16)
            state.state = [values, values]
            let selected = try XCTUnwrap(TokenIterator.shorterPlainKVPromptCheckpoint(
                coordinator: disk, promptTokenIds: ids, input: input(ids), parameters: parameters,
                hasAuxiliaryState: false, snapshot: [state]))
            XCTAssertEqual(selected.boundary, history.count)
            XCTAssertEqual(selected.cache[0].offset, history.count)
            XCTAssertTrue(arrayEqual(selected.cache[0].state[0], values[.ellipsis, ..<history.count, 0...]).item(Bool.self))
            XCTAssertEqual(state.offset, ids.count, "Selection cannot mutate the source checkpoint")
        }
    }

    func testCancellationAfterPreparationKeepsOnlyValidPromptState() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let history = history, rail = rail, parameters = parameters
        let task = Task {
            let lock = lockSerializedMLXTest(); defer { lock.unlock() }
            let model = LlamaModel(LlamaConfiguration(
                hiddenSize: 64, hiddenLayers: 2, intermediateSize: 128,
                attentionHeads: 8, rmsNormEps: 1e-5, vocabularySize: 128, kvHeads: 4))
            MLX.eval(model)
            let disk = CacheCoordinator(config: CacheCoordinatorConfig(
                usePagedCache: false, enableDiskCache: true, diskCacheDir: root, modelKey: "cancelled-plain-checkpoint"))
            disk.setGenPromptSuffixTokens(rail)
            let ids = history + rail
            let prepared = LMInput(tokens: MLXArray(ids.map(Int32.init)).reshaped(1, ids.count),
                tokenIds: ids, cachePrefixTokenCounts: [history.count])
            var iterator = try TokenIterator(input: prepared, model: model, parameters: parameters, cacheCoordinator: disk)
            withUnsafeCurrentTask { $0?.cancel() }
            iterator.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
            let salt = computeCacheSalt(for: prepared, parameters: parameters)
            return (disk.hasDurableDiskEntry(tokens: history, mediaSalt: salt),
                disk.hasDurableDiskEntry(tokens: ids, mediaSalt: salt))
        }
        let (prefixStored, exactStored) = try await task.value
        XCTAssertTrue(prefixStored)
        XCTAssertFalse(exactStored)
    }
}
