import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

/// Both engines, end to end, on a disk entry they cannot restore.
///
/// The models here have no weights: `attentionLayers` plain KV layers that
/// take every token they are shown. A ONE-layer model stores the longest
/// boundary, so its payload describes one layer; the TWO-layer model that
/// then runs the same prompt is served that entry and cannot restore it. A
/// two-layer entry for a shorter prefix sits behind it.
///
/// Without the engine's report the long entry is served on every turn and
/// every turn prefills the whole prompt. With it: the turn that finds out
/// still prefills everything, the next one restores the shorter entry, and
/// a turn that ends ON the refused boundary writes it again, after which it
/// restores.
///
/// No token length is a multiple of 64.
@Suite(.serialized)
struct DiskRestoreRejectionEngineTests {

    private final class PlainAttentionModel: Module, LanguageModel, @unchecked Sendable {
        let attentionLayers: Int
        var vocabularySize: Int { 61 }

        init(attentionLayers: Int) {
            self.attentionLayers = attentionLayers
            super.init()
        }

        func newCache(parameters: GenerateParameters?) -> [KVCache] {
            (0 ..< attentionLayers).map { _ in KVCacheSimple() }
        }

        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            .tokens(LMInput.Text(tokens: input.text.tokens.reshaped([-1])))
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            let tokenCount = inputs.reshaped([-1]).size
            for layer in cache ?? [] {
                let keys = MLXArray.ones([1, 1, tokenCount, 4])
                _ = layer.update(keys: keys, values: keys * 2)
            }
            return MLXArray.zeros([1, max(tokenCount, 1), vocabularySize])
        }
    }

    private final class ProgressRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [PrefillProgress] = []

        func append(_ value: PrefillProgress) {
            lock.lock()
            values.append(value)
            lock.unlock()
        }

        /// Tokens restored from the cache before prefill; nil when the turn
        /// restored nothing.
        var restoredTokens: Int? {
            lock.lock()
            defer { lock.unlock() }
            return values.first { $0.stage == .cacheRestore }?.completedUnitCount
        }
    }

    private static let prompt: [Int] = (0 ..< 37).map { 3 + $0 % 53 }
    private static let parameters = GenerateParameters(maxTokens: 1, temperature: 0)

    private static func makeCoordinator(_ label: String) -> (CacheCoordinator, URL) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-restore-rejection-\(label)-\(UUID().uuidString)")
        let coordinator = CacheCoordinator(
            config: CacheCoordinatorConfig(
                usePagedCache: false,
                enableDiskCache: true,
                diskCacheMaxGB: 1,
                diskCacheDir: root,
                modelKey: "restore-rejection-\(label)"))
        return (coordinator, root)
    }

    private static func input(_ tokens: [Int]) -> LMInput {
        LMInput(tokens: MLXArray(tokens.map(Int32.init)).expandedDimensions(axis: 0))
    }

    private static func stats(_ coordinator: CacheCoordinator) throws -> DiskCacheStats {
        try #require(coordinator.snapshotStats().diskStats)
    }

    /// A two-layer entry for the 11-token boundary whose layers hold
    /// `payloadTokens` tokens, written straight through `DiskCache.store`
    /// under the key the engines look up. With 11 it is the entry an engine
    /// would have written; with 9 it restores — nine tokens, into both
    /// layers — and its offsets then disagree with the boundary it was
    /// served for: the second structural refusal.
    private static func plantLongEntry(
        payloadTokens: Int, in coordinator: CacheCoordinator
    ) throws {
        let long = Array(prompt.prefix(11))
        let cache: [any KVCache] = (0 ..< 2).map { _ in KVCacheSimple() }
        for layer in cache {
            let keys = MLXArray.ones([1, 1, payloadTokens, 4])
            _ = layer.update(keys: keys, values: keys * 2)
        }
        MLX.eval(cache)
        try #require(cache.allSatisfy { $0.offset == payloadTokens })
        let disk = try #require(coordinator.diskCache)
        disk.store(
            tokens: long, arrays: TQDiskSerializer.serialize(cache: cache),
            mediaSalt: computeCacheSalt(for: input(long), parameters: parameters),
            enforceQuota: false)
        try #require(disk.candidateTokenCounts(maxTokens: 11).first == 11)
    }

    // MARK: - TokenIterator

    /// One turn of the solo path. Returns what it restored before prefill.
    private static func soloTurn(
        _ tokens: [Int], layers: Int, coordinator: CacheCoordinator, store: Bool
    ) throws -> Int? {
        let recorder = ProgressRecorder()
        var iterator = try TokenIterator(
            input: input(tokens),
            model: PlainAttentionModel(attentionLayers: layers),
            parameters: parameters,
            cacheCoordinator: coordinator,
            prefillProgressHandler: { recorder.append($0) })
        if store {
            iterator.storeCacheAfterGeneration(
                generatedTokenIds: [], includeGeneratedBoundary: false)
        }
        return recorder.restoredTokens
    }

    @Test func tokenIteratorReportsARestoreItCouldNotUse() throws {
        try MLXMetalTestLock.withLock {
            let (coordinator, root) = Self.makeCoordinator("solo")
            defer { try? FileManager.default.removeItem(at: root) }
            let prompt = Self.prompt
            let disk = try #require(coordinator.diskCache)

            _ = try Self.soloTurn(
                Array(prompt.prefix(5)), layers: 2, coordinator: coordinator, store: true)
            _ = try Self.soloTurn(
                Array(prompt.prefix(11)), layers: 1, coordinator: coordinator, store: true)
            try #require(disk.candidateTokenCounts(maxTokens: prompt.count) == [11, 5])
            let planted = try Self.stats(coordinator)

            // Control: a one-layer model restores the long entry, so the
            // entry is whole and the refusal below is about the fit.
            try #require(
                try Self.soloTurn(prompt, layers: 1, coordinator: coordinator, store: false) == 11)
            let control = try Self.stats(coordinator)
            try #require(control.hits - planted.hits == 1)
            try #require(control.rejectedDiskRestores == 0)

            // The turn that finds out: served 11, restores nothing.
            #expect(
                try Self.soloTurn(prompt, layers: 2, coordinator: coordinator, store: false) == nil)
            let found = try Self.stats(coordinator)
            #expect(found.rejectedDiskRestores == 1)
            #expect(found.hits == control.hits, "the refused hit was counted")

            // The next turn gets the entry that fits.
            #expect(
                try Self.soloTurn(prompt, layers: 2, coordinator: coordinator, store: false) == 5)
            let next = try Self.stats(coordinator)
            #expect(next.rejectedDiskRestores == 1)
            #expect(next.hits - found.hits == 1)

            // A turn that ends on the refused boundary writes it again.
            #expect(
                try Self.soloTurn(
                    Array(prompt.prefix(11)), layers: 2, coordinator: coordinator, store: true) == 5
            )
            #expect(
                try Self.soloTurn(prompt, layers: 2, coordinator: coordinator, store: false) == 11)
            #expect(try Self.stats(coordinator).rejectedDiskRestores == 1)
        }
    }

    /// The other refusal: the payload fits the cache, restores more than
    /// nothing, and the offsets it leaves disagree with the boundary. The
    /// control differs in the payload's token count alone, and restores.
    @Test func tokenIteratorReportsRestoredOffsetsThatMissTheBoundary() throws {
        try MLXMetalTestLock.withLock {
            for payloadTokens in [11, 9] {
                let (coordinator, root) = Self.makeCoordinator("solo-offsets-\(payloadTokens)")
                defer { try? FileManager.default.removeItem(at: root) }
                let prompt = Self.prompt
                _ = try Self.soloTurn(
                    Array(prompt.prefix(5)), layers: 2, coordinator: coordinator, store: true)
                try Self.plantLongEntry(payloadTokens: payloadTokens, in: coordinator)
                let planted = try Self.stats(coordinator)
                try #require(planted.rejectedDiskRestores == 0)

                let first = try Self.soloTurn(
                    prompt, layers: 2, coordinator: coordinator, store: false)
                let found = try Self.stats(coordinator)
                if payloadTokens == 11 {
                    try #require(first == 11, "INVALID: the control entry does not restore")
                    try #require(found.rejectedDiskRestores == 0)
                    continue
                }
                #expect(first == nil, "a restore that was refused was reported as one")
                #expect(found.rejectedDiskRestores == 1)
                #expect(found.hits == planted.hits, "the refused hit was counted")
                #expect(
                    try Self.soloTurn(prompt, layers: 2, coordinator: coordinator, store: false)
                        == 5)
                #expect(try Self.stats(coordinator).rejectedDiskRestores == 1)
            }
        }
    }

    // MARK: - BatchEngine

    private static func engine(layers: Int, coordinator: CacheCoordinator) -> BatchEngine {
        let model = PlainAttentionModel(attentionLayers: layers)
        let processor = TestInputProcessor(
            tokenizer: TestTokenizer(vocabularySize: model.vocabularySize),
            configuration: ModelConfiguration(id: "restore-rejection-\(layers)"),
            messageGenerator: DefaultMessageGenerator())
        nonisolated(unsafe) let context = ModelContext(
            configuration: processor.configuration,
            model: model,
            processor: processor,
            tokenizer: processor.tokenizer)
        return BatchEngine(context: context, maxBatchSize: 2, cacheCoordinator: coordinator)
    }

    /// One request through the scheduler (`submit`, not the solo fast path).
    /// The engine stores the prompt boundary when the request finishes.
    private static func batchTurn(_ tokens: [Int], engine: BatchEngine) async -> Int? {
        let (_, stream) = await engine.submit(input: input(tokens), parameters: parameters)
        var restored: Int?
        for await event in stream {
            if case .prefillProgress(let progress) = event, progress.stage == .cacheRestore,
                restored == nil
            {
                restored = progress.completedUnitCount
            }
        }
        return restored
    }

    /// The long boundary followed by another conversation. The engine stores
    /// every finished turn's prompt boundary, so each turn here gets a tail
    /// of its own: what a later turn is served is then never something an
    /// earlier turn of this test wrote by the way.
    private static func divergingPrompt(tail seed: Int, count: Int) -> [Int] {
        let tail = (0 ..< count - 11).map { 3 + (seed * 7 + $0 * 5) % 53 }
        precondition(tail[0] != prompt[11])
        return Array(prompt.prefix(11)) + tail
    }

    /// The store runs when the request finishes, which the stream's end does
    /// not promise to wait for.
    private static func waitForBoundary(_ length: Int, in disk: DiskCache) async throws {
        for _ in 0 ..< 200 {
            if disk.candidateTokenCounts(maxTokens: length).first == length { return }
            try await Task.sleep(nanoseconds: 25_000_000)
        }
        Issue.record("INVALID: the engine never stored the \(length)-token boundary")
    }

    @Test func batchEngineReportsARestoreItCouldNotUse() async throws {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        let (coordinator, root) = Self.makeCoordinator("batch")
        defer { try? FileManager.default.removeItem(at: root) }
        let disk = try #require(coordinator.diskCache)
        let oneLayer = Self.engine(layers: 1, coordinator: coordinator)
        let twoLayers = Self.engine(layers: 2, coordinator: coordinator)
        let long = Array(Self.prompt.prefix(11))

        _ = await Self.batchTurn(Array(Self.prompt.prefix(5)), engine: twoLayers)
        try await Self.waitForBoundary(5, in: disk)
        _ = await Self.batchTurn(long, engine: oneLayer)
        try await Self.waitForBoundary(11, in: disk)
        try #require(disk.candidateTokenCounts(maxTokens: 11) == [11, 5])
        try #require(try Self.stats(coordinator).rejectedDiskRestores == 0)

        // Control: the one-layer model restores the long entry, so the entry
        // is whole and the refusal below is about the fit.
        try #require(
            await Self.batchTurn(Self.divergingPrompt(tail: 6, count: 17), engine: oneLayer) == 11)
        try await Self.waitForBoundary(17, in: disk)
        try #require(try Self.stats(coordinator).rejectedDiskRestores == 0)

        // The turn that finds out: served 11, restores nothing.
        #expect(
            await Self.batchTurn(Self.divergingPrompt(tail: 1, count: 23), engine: twoLayers)
                == nil)
        try await Self.waitForBoundary(23, in: disk)
        #expect(try Self.stats(coordinator).rejectedDiskRestores == 1)

        // The next turn gets the entry that fits.
        #expect(
            await Self.batchTurn(Self.divergingPrompt(tail: 2, count: 29), engine: twoLayers) == 5)
        try await Self.waitForBoundary(29, in: disk)
        #expect(try Self.stats(coordinator).rejectedDiskRestores == 1)

        // A turn that ends on the refused boundary writes it again.
        #expect(await Self.batchTurn(long, engine: twoLayers) == 5)
        // Already indexed, so wait for the payload to change hands instead.
        for _ in 0 ..< 200
        where !coordinator.hasValidatedDiskEntry(
            tokens: long,
            mediaSalt: computeCacheSalt(for: Self.input(long), parameters: Self.parameters))
        {
            try await Task.sleep(nanoseconds: 25_000_000)
        }
        #expect(
            await Self.batchTurn(Self.divergingPrompt(tail: 4, count: 31), engine: twoLayers)
                == 11)
        #expect(try Self.stats(coordinator).rejectedDiskRestores == 1)

        await oneLayer.shutdown()
        await twoLayers.shutdown()
    }

    /// The offsets refusal through the scheduler; the control differs in the
    /// payload's token count alone. (The engine announces a cache restore as
    /// soon as the payload is in, before it checks the offsets, so what the
    /// refusing turn reports as restored is not asserted here.)
    @Test func batchEngineReportsRestoredOffsetsThatMissTheBoundary() async throws {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        for payloadTokens in [11, 9] {
            let (coordinator, root) = Self.makeCoordinator("batch-offsets-\(payloadTokens)")
            defer { try? FileManager.default.removeItem(at: root) }
            let disk = try #require(coordinator.diskCache)
            let twoLayers = Self.engine(layers: 2, coordinator: coordinator)

            _ = await Self.batchTurn(Array(Self.prompt.prefix(5)), engine: twoLayers)
            try await Self.waitForBoundary(5, in: disk)
            try Self.plantLongEntry(payloadTokens: payloadTokens, in: coordinator)
            let planted = try Self.stats(coordinator)
            try #require(planted.rejectedDiskRestores == 0)

            let first = await Self.batchTurn(
                Self.divergingPrompt(tail: 1, count: 23), engine: twoLayers)
            try await Self.waitForBoundary(23, in: disk)
            let found = try Self.stats(coordinator)
            if payloadTokens == 11 {
                try #require(first == 11, "INVALID: the control entry does not restore")
                try #require(found.rejectedDiskRestores == 0)
            } else {
                #expect(found.rejectedDiskRestores == 1)
                #expect(found.hits == planted.hits, "the refused hit was counted")
                #expect(
                    await Self.batchTurn(
                        Self.divergingPrompt(tail: 2, count: 29), engine: twoLayers) == 5)
                #expect(try Self.stats(coordinator).rejectedDiskRestores == 1)
            }
            await twoLayers.shutdown()
        }
    }
}
