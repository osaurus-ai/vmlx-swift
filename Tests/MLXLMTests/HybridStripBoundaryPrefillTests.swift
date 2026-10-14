// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
@testable import MLXLMCommon
import MLXNN
import XCTest

/// Records every token it is asked to forward, so a test can tell the
/// difference between prefilling a prompt once and prefilling it twice.
private class RecordingHybridModel: Module, LanguageModel, @unchecked Sendable {
    private(set) var forwarded: [Int] = []
    private(set) var prepareCalls = 0
    private(set) var prepareMediaFlags: [Bool] = []
    private(set) var prepareTokenSegments: [[Int]] = []
    private(set) var prepareMaskSegments: [[Int]] = []

    var vocabularySize: Int { 64 }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [MambaCache()]
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        prepareCalls += 1
        prepareMediaFlags.append(input.hasMediaContent)
        prepareTokenSegments.append(input.text.tokens.reshaped(-1).asArray(Int32.self).map(Int.init))
        prepareMaskSegments.append(input.text.mask?.reshaped(-1).asArray(Int32.self).map(Int.init) ?? [])
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
        return MLXArray.zeros([1, max(ids.count, 1), vocabularySize])
    }
}

private final class CancelledHybridModel: RecordingHybridModel, @unchecked Sendable {
    override func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        if prepareCalls == 1 { throw CancellationError() }
        return try super.prepare(input, cache: cache, windowSize: windowSize)
    }
}

private final class CanonicalHybridModel: RecordingHybridModel, CanonicalRequiredToolCacheModel,
    @unchecked Sendable {
    var canonicalRequiredToolCacheIdentity: String { "test-owned-cold-schedule" }
    func canonicalRequiredToolChunkSize(parameters: GenerateParameters) -> Int? { 3 }
    func validateCanonicalRequiredToolCache(_ cache: [KVCache], boundary: Int) -> Bool { true }
}

private final class RecordingPromptProcessor: LogitProcessor {
    var prompts: [[Int]] = []
    func prompt(_ prompt: MLXArray) { prompts.append(prompt.reshaped(-1).asArray(Int32.self).map(Int.init)) }
    func process(logits: MLXArray) -> MLXArray { logits }
    func didSample(token: MLXArray) {}
}

/// Same recorder, dense cache topology.
private final class RecordingDenseModel: RecordingHybridModel, @unchecked Sendable {
    override func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [KVCacheSimple()]
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

    func testStableBoundariesBeforeStripAreCapturedWithoutTerminalReplay() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let model = RecordingHybridModel()
        let coordinator = makeCoordinator()
        coordinator.setGenPromptSuffixTokens(genPromptSuffix)
        let prompt = [1, 2, 3, 4, 5] + userTurn + genPromptSuffix
        let input = LMInput(
            tokens: MLXArray(prompt.map(Int32.init)).expandedDimensions(axis: 0),
            tokenIds: prompt,
            cachePrefixTokenCounts: [5, 9],
            cacheStablePrefixTokenCounts: [5, 9])
        var iterator = try TokenIterator(
            input: input, model: model,
            parameters: GenerateParameters(maxTokens: 1, temperature: 0, prefillStepSize: 3),
            cacheCoordinator: coordinator)
        XCTAssertEqual(model.forwarded, prompt)
        let callsBeforeStore = model.prepareCalls
        iterator.storeCacheAfterGeneration(
            generatedTokenIds: [], includeGeneratedBoundary: false)
        XCTAssertEqual(model.forwarded, prompt,
            "stable seeds before the later strip must be captured, not replayed after output")
        XCTAssertEqual(model.prepareCalls, callsBeforeStore,
            "terminal cache publication must not prepare the earlier stable prefixes again")
        XCTAssertTrue(coordinator.diskCache!.waitForPendingWrites(timeout: 5))
    }

    func testWarmStableHeadUsesAbsoluteRestoredOffsetWithoutReplay() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let coordinator = makeCoordinator()
        coordinator.setGenPromptSuffixTokens(genPromptSuffix)
        let parameters = GenerateParameters(maxTokens: 1, temperature: 0, prefillStepSize: 3)
        let cold = [1, 2, 3, 4, 5] + userTurn + genPromptSuffix
        var first = try TokenIterator(
            input: LMInput(tokens: MLXArray(cold.map(Int32.init)).expandedDimensions(axis: 0),
                tokenIds: cold, cachePrefixTokenCounts: [5, 12], cacheStablePrefixTokenCounts: [5]),
            model: RecordingHybridModel(), parameters: parameters, cacheCoordinator: coordinator)
        first.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
        XCTAssertTrue(coordinator.diskCache!.waitForPendingWrites(timeout: 5))
        let prompt = [1, 2, 3, 4, 5, 21, 22, 23, 24, 25, 26, 27] + genPromptSuffix
        let model = RecordingHybridModel()
        var warm = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map(Int32.init)).expandedDimensions(axis: 0),
                tokenIds: prompt, cachePrefixTokenCounts: [5, 9, 12], cacheStablePrefixTokenCounts: [5, 9]),
            model: model, parameters: parameters, cacheCoordinator: coordinator)
        XCTAssertEqual(model.forwarded, Array(prompt.dropFirst(4)), "warm restore must resume at4")
        XCTAssertNil(warm.stableBoundarySnapshots[4], "already-restored seed must not be duplicated")
        XCTAssertNotNil(warm.stableBoundarySnapshots[8])
        let calls = model.prepareCalls
        warm.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
        XCTAssertEqual(model.forwarded, Array(prompt.dropFirst(4)))
        XCTAssertEqual(model.prepareCalls, calls)
        XCTAssertTrue(coordinator.diskCache!.waitForPendingWrites(timeout: 5))
    }

    func testStableHeadPreservesTokenMaskSegmentsAndOneProcessorPriming() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let coordinator = makeCoordinator()
        coordinator.setGenPromptSuffixTokens(genPromptSuffix)
        let prompt = [1, 2, 3, 4, 5] + userTurn + genPromptSuffix
        let input = LMInput(text: .init(
                tokens: MLXArray(prompt.map(Int32.init)).expandedDimensions(axis: 0),
                mask: MLXArray((0..<prompt.count).map(Int32.init)).expandedDimensions(axis: 0),
                tokenIds: prompt),
            cachePrefixTokenCounts: [5, 9, 12], cacheStablePrefixTokenCounts: [5, 9])
        let model = RecordingHybridModel()
        var iterator = try TokenIterator(input: input, model: model,
            parameters: GenerateParameters(maxTokens: 1, temperature: 0, prefillStepSize: 3),
            cacheCoordinator: coordinator)
        XCTAssertEqual(model.prepareTokenSegments.flatMap { $0 }, prompt)
        XCTAssertEqual(model.prepareMaskSegments.flatMap { $0 }, Array(0..<prompt.count))
        let spy = RecordingPromptProcessor()
        // Exercise the same internal preparation path with a caller processor,
        // resetting live cache/snapshots so the second pass is cold.
        iterator.cache = model.newCache(parameters: nil)
        iterator.hybridStripSnapshot = nil
        iterator.stableBoundarySnapshots = [:]
        iterator.processor = spy
        try iterator.prepare(input: input, windowSize: 3)
        XCTAssertEqual(spy.prompts, [prompt], "head snapshots must never prime/reset the processor")
    }

    func testStableHeadKeepsOpaqueMediaSpanOnOneSegment() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let coordinator = makeCoordinator()
        coordinator.setGenPromptSuffixTokens(genPromptSuffix)
        let prompt = [1, 99, 3, 4, 5, 99, 12, 13, 14, 15, 16, 17] + genPromptSuffix
        let input = LMInput(text: .init(
                tokens: MLXArray(prompt.map(Int32.init)).expandedDimensions(axis: 0), tokenIds: prompt),
            image: .init(pixels: MLXArray.zeros([1, 3, 2, 2])), mediaTokenIds: [99],
            cachePrefixTokenCounts: [5, 9, 12], cacheStablePrefixTokenCounts: [5, 9])
        let model = RecordingHybridModel()
        let iterator = try TokenIterator(input: input, model: model,
            parameters: GenerateParameters(maxTokens: 1, temperature: 0, prefillStepSize: 3),
            cacheCoordinator: coordinator)
        XCTAssertEqual(model.forwarded, prompt)
        XCTAssertEqual(model.prepareMediaFlags.filter { $0 }.count, 1)
        XCTAssertNil(iterator.stableBoundarySnapshots[4], "cannot split within a media span")
        XCTAssertNotNil(iterator.stableBoundarySnapshots[8], "later complete-media boundary remains eligible")
    }

    func testStableHeadPropagatesCancellationWithoutPublishingPartialSeeds() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let coordinator = makeCoordinator()
        coordinator.setGenPromptSuffixTokens(genPromptSuffix)
        let prompt = [1, 2, 3, 4, 5] + userTurn + genPromptSuffix
        let model = CancelledHybridModel()
        XCTAssertThrowsError(try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map(Int32.init)).expandedDimensions(axis: 0),
                tokenIds: prompt, cachePrefixTokenCounts: [5, 9, 12], cacheStablePrefixTokenCounts: [5, 9]),
            model: model, parameters: GenerateParameters(maxTokens: 1, temperature: 0, prefillStepSize: 3),
            cacheCoordinator: coordinator)) { error in
                XCTAssertTrue(error is CancellationError)
            }
        XCTAssertEqual(coordinator.snapshotStats().diskStats?.stores, 0)
    }

    func testCanonicalRequiredToolModelKeepsItsOriginalHeadSchedule() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let coordinator = makeCoordinator()
        let prompt = [1, 2, 3, 4, 5] + userTurn + genPromptSuffix
        let tools: [ToolSpec] = [["type": "function", "function": ["name": "calculate"] as [String: any Sendable]]]
        let scope = try XCTUnwrap(CanonicalRequiredToolContext(
            additionalContext: ["tool_choice": "required"], tools: tools))
        let model = CanonicalHybridModel()
        _ = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map(Int32.init)).expandedDimensions(axis: 0),
                tokenIds: prompt, cachePrefixTokenCounts: [5, 9, 12], cacheStablePrefixTokenCounts: [5, 9],
                cacheRestorePolicy: .freshRequiredToolSelection,
                toolSchemas: tools, canonicalRequiredToolContext: scope),
            model: model, parameters: GenerateParameters(maxTokens: 1, temperature: 0, prefillStepSize: 3),
            cacheCoordinator: coordinator)
        XCTAssertEqual(model.prepareTokenSegments, [Array(prompt[..<12]), Array(prompt[12...])],
            "typed canonical contract must keep the cold head layout unchanged")
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

    func testDenseModelPromptIsNotSplit() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }

        // A plain KV cache carries no path-dependent state, so the coordinator
        // stays dense and the boundary never applies.
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
            model.prepareCalls, 1,
            "dense models reuse via the post-answer boundary and must not pay for a split")
        XCTAssertEqual(model.forwarded, prompt)
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
