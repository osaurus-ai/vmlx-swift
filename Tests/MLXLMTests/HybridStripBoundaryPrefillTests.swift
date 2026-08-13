// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

/// Records every token it is asked to forward, so a test can tell the
/// difference between prefilling a prompt once and prefilling it twice.
private class RecordingHybridModel: Module, LanguageModel, @unchecked Sendable {
    private(set) var forwarded: [Int] = []
    private(set) var prepareCalls = 0
    private(set) var prepareMediaFlags: [Bool] = []

    var vocabularySize: Int { 64 }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [MambaCache()]
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        prepareCalls += 1
        prepareMediaFlags.append(input.hasMediaContent)
        let step = max(1, windowSize ?? 512)
        var flatTokens = input.text.tokens.reshaped([-1])
        while flatTokens.size > step {
            _ = callAsFunction(flatTokens[..<step][.newAxis, 0...], cache: cache)
            MLX.eval(cache)
            flatTokens = flatTokens[step...]
        }
        return .tokens(LMInput.Text(tokens: flatTokens))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let ids = inputs.reshaped([-1]).asArray(Int32.self).map(Int.init)
        forwarded.append(contentsOf: ids)
        if let mamba = cache?.first as? MambaCache {
            let runningSum = (mamba.state.first?.asArray(Float.self).first ?? 0)
                + ids.reduce(Float(0)) { $0 + Float($1) }
            mamba.state = [MLXArray([runningSum, Float(mamba.offset + ids.count)]).reshaped([1, 1, 2])]
            mamba.offset += ids.count
        }
        // Attention topologies have to advance too, or their `offset` stays 0 while the
        // boundary key claims the full prompt and #208's offset guard refuses every
        // store — which silently made the dense/rotating fixtures store nothing at all.
        if let rotating = cache?.first as? RotatingKVCache {
            _ = rotating.update(
                keys: MLXArray.zeros([1, 1, ids.count, 4]),
                values: MLXArray.zeros([1, 1, ids.count, 4]))
        } else if let simple = cache?.first as? KVCacheSimple {
            _ = simple.update(
                keys: MLXArray.zeros([1, 1, ids.count, 4]),
                values: MLXArray.zeros([1, 1, ids.count, 4]))
        }
        return MLXArray.zeros([1, max(ids.count, 1), vocabularySize])
    }
}

/// Same recorder, dense cache topology.
private final class RecordingDenseModel: RecordingHybridModel, @unchecked Sendable {
    override func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [KVCacheSimple()]
    }
}

/// Same recorder, rotating/SWA topology — Gemma 4's shape, and the family the
/// live `BENCH_GROWING_CHAT_CACHE` regression appeared on. Unlike `KVCacheSimple`
/// a rotating cache requires disk-backed restore, so its boundaries are actually
/// persisted at fixture-sized prompts instead of waiting for a full paged block.
private final class RecordingRotatingModel: RecordingHybridModel, @unchecked Sendable {
    override func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [RotatingKVCache(maxSize: 1024, keep: 0)]
    }
}

/// The gen-suffix-stripped boundary is the only prefix a hybrid model's next
/// chat turn can reuse, and hybrid cache state is path-dependent, so it cannot
/// be trimmed out of the finished prompt cache. It used to be reconstructed by
/// replaying the whole stripped prefix through the model after generation —
/// a second full prefill, which ran before the completion event reached the
/// client and held the response stream open for seconds at long context.
///
/// It is now captured from the live prefill as it passes the boundary. These
/// tests pin that: the prompt is forwarded exactly once, and the boundary the
/// store sees is the state a warm pass over the stripped prefix produces.
final class HybridStripBoundaryPrefillTests: XCTestCase {

    private var tempDirs: [URL] = []

    override func tearDown() {
        for dir in tempDirs {
            try? FileManager.default.removeItem(at: dir)
        }
        tempDirs.removeAll()
        super.tearDown()
    }

    private func makeCoordinator(hybrid: Bool = true) -> CacheCoordinator {
        let diskDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("hybrid-strip-\(UUID().uuidString)")
        tempDirs.append(diskDir)
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: true,
            enableDiskCache: true,
            diskCacheDir: diskDir,
            modelKey: "recording-hybrid|reasoning=off"))
        if hybrid { coordinator.setHybrid(true) }
        return coordinator
    }

    /// `<turn-start> assistant \n` — the generation prompt the next turn replaces.
    private let genPromptSuffix = [201, 202, 203]
    private let userTurn = [11, 12, 13, 14, 15, 16, 17]

    func testPrefillForwardsEachPromptTokenExactlyOnce() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }

        let model = RecordingHybridModel()
        let coordinator = makeCoordinator()
        coordinator.setGenPromptSuffixTokens(genPromptSuffix)

        let prompt = userTurn + genPromptSuffix
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map { Int32($0) }).expandedDimensions(axis: 0)),
            model: model,
            parameters: GenerateParameters(maxTokens: 1, temperature: 0, prefillStepSize: 3),
            cacheCoordinator: coordinator)

        let forwardedDuringPrefill = model.forwarded
        XCTAssertEqual(
            forwardedDuringPrefill, prompt,
            "prefill must forward the prompt once, in order, with no replay")

        // The boundary is captured, so storing it must not run the model again.
        let prepareCallsAfterPrefill = model.prepareCalls
        iterator.storeCacheAfterGeneration(
            generatedTokenIds: [], includeGeneratedBoundary: false)

        XCTAssertEqual(
            model.forwarded, forwardedDuringPrefill,
            "storing the stripped boundary must not re-derive it through the model")
        XCTAssertEqual(model.prepareCalls, prepareCallsAfterPrefill)
    }

    func testHybridStorePublishesOnlyCanonicalStrippedBoundary() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }

        let model = RecordingHybridModel()
        let coordinator = makeCoordinator()
        coordinator.setGenPromptSuffixTokens(genPromptSuffix)

        let prompt = userTurn + genPromptSuffix
        let input = LMInput(
            tokens: MLXArray(prompt.map { Int32($0) }).expandedDimensions(axis: 0))
        let parameters = GenerateParameters(maxTokens: 1, temperature: 0)
        let mediaSalt = computeCacheSalt(for: input, parameters: parameters)
        var iterator = try TokenIterator(
            input: input,
            model: model,
            parameters: parameters,
            cacheCoordinator: coordinator)

        iterator.storeCacheAfterGeneration(
            generatedTokenIds: [42], includeGeneratedBoundary: true)

        let stats = coordinator.snapshotStats().diskStats
        XCTAssertEqual(
            stats?.stores, 1,
            "hybrid chat must not serialize prompt + stripped + post-answer full snapshots")
        let restored = coordinator.fetch(
            tokens: prompt + [77],
            mediaSalt: mediaSalt,
            skipExactDiskBoundary: true)
        guard case .hit(let matched, let remaining, let detail, _, _, _) = restored else {
            return XCTFail("canonical stripped boundary should be restorable")
        }
        XCTAssertEqual(matched, userTurn.count)
        XCTAssertEqual(remaining, genPromptSuffix + [77])
        XCTAssertEqual(detail, .disk)
    }

    func testHybridStableBoundaryPublishesWarmupSafeSeed() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }

        let model = RecordingHybridModel()
        let coordinator = makeCoordinator()
        coordinator.setGenPromptSuffixTokens(genPromptSuffix)

        let stable = [1, 2, 3, 4, 5]
        let prompt = stable + userTurn + genPromptSuffix
        let input = LMInput(
            tokens: MLXArray(prompt.map { Int32($0) }).expandedDimensions(axis: 0),
            tokenIds: prompt,
            cachePrefixTokenCounts: [stable.count],
            cacheStablePrefixTokenCounts: [stable.count])
        let parameters = GenerateParameters(maxTokens: 1, temperature: 0)
        let mediaSalt = computeCacheSalt(for: input, parameters: parameters)
        var iterator = try TokenIterator(
            input: input,
            model: model,
            parameters: parameters,
            cacheCoordinator: coordinator)

        iterator.storeCacheAfterGeneration(
            generatedTokenIds: [42], includeGeneratedBoundary: true)

        let restored = coordinator.fetch(
            tokens: stable,
            mediaSalt: mediaSalt,
            skipExactDiskBoundary: true,
            preferredDiskBoundaries: [stable.count])
        guard case .hit(let matched, let remaining, let detail, _, _, _) = restored else {
            return XCTFail("first new-chat warmup should restore the stable N-1 seed")
        }
        XCTAssertEqual(matched, stable.count - 1)
        XCTAssertEqual(remaining, [stable.last!])
        XCTAssertEqual(detail, .disk)
    }

    func testPrefillSplitsAtTheTurnStartToken() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }

        let model = RecordingHybridModel()
        let coordinator = makeCoordinator()
        coordinator.setGenPromptSuffixTokens(genPromptSuffix)

        let prompt = userTurn + genPromptSuffix
        _ = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map { Int32($0) }).expandedDimensions(axis: 0)),
            model: model,
            parameters: GenerateParameters(maxTokens: 1, temperature: 0, prefillStepSize: 512),
            cacheCoordinator: coordinator)

        // Head (stripped prefix) and tail (generation prompt) are prepared
        // separately, so a window large enough to swallow the whole prompt in one
        // go still yields two `prepare` calls.
        XCTAssertEqual(model.prepareCalls, 2)
        XCTAssertEqual(model.forwarded, prompt)
    }

    /// This test used to assert the opposite — one `prepare` call, on the
    /// rationale that "dense models reuse via the post-answer boundary and must
    /// not pay for a split". That rationale does not survive a reasoning model:
    /// hosts strip think blocks when they re-render history, so the post-answer
    /// snapshot can never be a prefix of the next turn's prompt, and a dense or
    /// rotating model re-prefills the whole previous reply on every send.
    /// Measured on DSV4 (`DSV4-FINDINGS-AND-CROSSFAMILY-BACKLOG-2026-08-09.md`
    /// §3): `HIT disk boundary=3464 remaining=1367` then `MISS all tiers
    /// tokens=2644` — ~1367 tokens ≈ 3.4 s at 400 pp/s, growing with the
    /// conversation.
    ///
    /// The split's price is one extra `prepare` call. It is NOT extra token
    /// work — `forwarded` below is byte-identical to the unsplit prompt — so on
    /// a real prompt already chunked at `prefillStepSize` it costs at most one
    /// short partial chunk. The trade is that against a whole re-prefilled
    /// reply per turn.
    ///
    /// The case that stays free is a prompt with no turn-start token at all
    /// (raw completion, non-chat): see
    /// ``testPromptWithoutTurnStartTokenIsNeverSplit``.
    func testDenseModelPromptSplitsAtTheTurnStartToken() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }

        // A plain KV cache carries no path-dependent state, so the coordinator
        // stays dense — which no longer exempts it from the boundary.
        let model = RecordingDenseModel()
        let coordinator = makeCoordinator(hybrid: false)
        coordinator.setGenPromptSuffixTokens(genPromptSuffix)

        let prompt = userTurn + genPromptSuffix
        _ = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map { Int32($0) }).expandedDimensions(axis: 0)),
            model: model,
            parameters: GenerateParameters(maxTokens: 1, temperature: 0, prefillStepSize: 512),
            cacheCoordinator: coordinator)

        XCTAssertEqual(
            model.prepareCalls, 2,
            "a dense chat prompt lost its cross-turn checkpoint, so the next turn "
                + "re-prefills the entire previous assistant reply")
        // The split must not change what the model sees: same tokens, same order,
        // each exactly once.
        XCTAssertEqual(model.forwarded, prompt)
    }

    /// The counterpart to ``testHybridStorePublishesOnlyCanonicalStrippedBoundary``,
    /// and the regression that unit tests missed until a live bench row caught it.
    ///
    /// For an SSM hybrid the stripped boundary is the *only* one worth storing, so
    /// `usesCanonicalHybridBoundary` suppresses the exact-prompt store, the N-1 disk
    /// seed and the non-stable prefix boundaries. Widening THAT flag along with the
    /// capture looked harmless and passed every test in this file — but on
    /// `gemma-4-E2B-it-8bit` (rotating SWA) it dropped disk stores from 4 to 2 and
    /// moved the pre-turn-2 match from 991/1015 down to 982/1015, exactly the
    /// generation-prompt suffix length: the stripped boundary had *replaced* the
    /// better post-answer one instead of joining it, and `BENCH_GROWING_CHAT_CACHE`
    /// failed with "probe hit did not reach the safe prompt/cache boundary".
    ///
    /// So a non-hybrid must store strictly MORE than a hybrid does, not the same one.
    func testDenseStoreAddsStrippedBoundaryWithoutReplacingTheOthers() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }

        let model = RecordingRotatingModel()
        let coordinator = makeCoordinator(hybrid: false)
        coordinator.setGenPromptSuffixTokens(genPromptSuffix)

        let prompt = userTurn + genPromptSuffix
        let input = LMInput(
            tokens: MLXArray(prompt.map { Int32($0) }).expandedDimensions(axis: 0))
        let parameters = GenerateParameters(maxTokens: 1, temperature: 0)
        let mediaSalt = computeCacheSalt(for: input, parameters: parameters)
        var iterator = try TokenIterator(
            input: input, model: model, parameters: parameters, cacheCoordinator: coordinator)

        iterator.storeCacheAfterGeneration(
            generatedTokenIds: [42], includeGeneratedBoundary: true)

        // The live assertion, restated: the next turn must still be able to resume at
        // or past the END of this prompt. Suppressing the other stores drops the best
        // match back to the stripped boundary — `userTurn.count`, i.e. short by exactly
        // the generation-prompt suffix, which is the 991 -> 982 the bench reported.
        // Two boundaries, not one: the full prompt AND the turn-start strip.
        XCTAssertGreaterThan(
            coordinator.snapshotStats().diskStats?.stores ?? 0, 1,
            "a non-hybrid must publish more than the single canonical boundary")
        let restored = coordinator.fetch(
            tokens: prompt + [42, 77], mediaSalt: mediaSalt, skipExactDiskBoundary: true)
        guard case .hit(let matched, _, _, _, _, _) = restored else {
            return XCTFail(
                "a dense chat prompt published no reusable boundary at all")
        }
        XCTAssertGreaterThanOrEqual(
            matched, prompt.count,
            "a dense/rotating model must ADD the turn-start boundary to its existing "
                + "prompt and post-answer stores, not replace them — collapsing to the "
                + "single canonical boundary is an SSM-hybrid-only policy and costs "
                + "these models reuse they already had (matched \(matched), "
                + "stripped boundary is \(userTurn.count), prompt is \(prompt.count))")
    }

    /// A prompt with no turn-start token has no cross-turn checkpoint to store,
    /// so it must still be prefilled in a single pass. This is the raw /
    /// non-chat path, and widening the boundary past hybrids must not reach it.
    func testPromptWithoutTurnStartTokenIsNeverSplit() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }

        for hybrid in [true, false] {
            let model = hybrid ? RecordingHybridModel() : RecordingDenseModel()
            let coordinator = makeCoordinator(hybrid: hybrid)
            coordinator.setGenPromptSuffixTokens(genPromptSuffix)

            // `userTurn` alone — the generation-prompt suffix is what carries the
            // turn-start token, and it is deliberately absent here.
            let prompt = userTurn.filter { !genPromptSuffix.contains($0) }
            XCTAssertFalse(prompt.isEmpty)
            _ = try TokenIterator(
                input: LMInput(
                    tokens: MLXArray(prompt.map { Int32($0) }).expandedDimensions(axis: 0)),
                model: model,
                parameters: GenerateParameters(maxTokens: 1, temperature: 0, prefillStepSize: 512),
                cacheCoordinator: coordinator)

            XCTAssertEqual(
                model.prepareCalls, 1,
                "a non-chat prompt (hybrid=\(hybrid)) paid for a split it can never reuse")
            XCTAssertEqual(model.forwarded, prompt)
        }
    }

    func testMediaBeforeBoundaryStaysOnHeadAndStillSplits() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }

        let model = RecordingHybridModel()
        let coordinator = makeCoordinator()
        coordinator.setGenPromptSuffixTokens(genPromptSuffix)

        let imageToken = 99
        let prompt = [imageToken] + userTurn + genPromptSuffix
        let input = LMInput(
            text: .init(
                tokens: MLXArray(prompt.map { Int32($0) }).expandedDimensions(axis: 0),
                tokenIds: prompt),
            image: .init(pixels: MLXArray.zeros([1, 3, 2, 2])),
            mediaTokenIds: [imageToken])

        _ = try TokenIterator(
            input: input,
            model: model,
            parameters: GenerateParameters(maxTokens: 1, temperature: 0, prefillStepSize: 512),
            cacheCoordinator: coordinator)

        XCTAssertEqual(model.prepareCalls, 2)
        XCTAssertEqual(model.prepareMediaFlags, [true, false])
        XCTAssertEqual(model.forwarded, prompt)
    }

    func testMediaPlaceholderAfterBoundaryDoesNotSplit() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }

        let model = RecordingHybridModel()
        let coordinator = makeCoordinator()
        coordinator.setGenPromptSuffixTokens(genPromptSuffix)

        let imageToken = 99
        let prompt = userTurn + [genPromptSuffix[0], imageToken]
            + Array(genPromptSuffix.dropFirst())
        let input = LMInput(
            text: .init(
                tokens: MLXArray(prompt.map { Int32($0) }).expandedDimensions(axis: 0),
                tokenIds: prompt),
            image: .init(pixels: MLXArray.zeros([1, 3, 2, 2])),
            mediaTokenIds: [imageToken])

        _ = try TokenIterator(
            input: input,
            model: model,
            parameters: GenerateParameters(maxTokens: 1, temperature: 0, prefillStepSize: 512),
            cacheCoordinator: coordinator)

        XCTAssertEqual(model.prepareCalls, 1)
        XCTAssertEqual(model.prepareMediaFlags, [true])
        XCTAssertEqual(model.forwarded, prompt)
    }

    func testBoundaryIsNotSoughtWhenNoCacheTierCanHoldIt() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }

        let model = RecordingHybridModel()
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false,
            enableDiskCache: false,
            modelKey: "recording-hybrid|reasoning=off"))
        coordinator.setHybrid(true)
        coordinator.setGenPromptSuffixTokens(genPromptSuffix)
        XCTAssertFalse(coordinator.canPersistBoundaries)

        let prompt = userTurn + genPromptSuffix
        _ = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map { Int32($0) }).expandedDimensions(axis: 0)),
            model: model,
            parameters: GenerateParameters(maxTokens: 1, temperature: 0, prefillStepSize: 512),
            cacheCoordinator: coordinator)

        XCTAssertEqual(model.prepareCalls, 1)
    }
}
