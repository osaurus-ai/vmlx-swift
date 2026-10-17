// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
@testable import MLXLMCommon
import MLXNN
import XCTest

/// Records every token it forwards; its recurrent state depends on token
/// order, so a wrong restore changes the state instead of passing silently.
/// Media is ignored by the forward on purpose: the tests are about keying.
private class RecordingHybridModel: Module, LanguageModel, @unchecked Sendable {
    private(set) var forwarded: [Int] = []
    var vocabularySize: Int { 64 }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [MambaCache()] }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        let step = max(1, windowSize ?? 512)
        var flat = input.text.tokens.reshaped([-1])
        while flat.size > step {
            _ = callAsFunction(flat[..<step][.newAxis, 0...], cache: cache)
            MLX.eval(cache)
            flat = flat[step...]
        }
        return .tokens(LMInput.Text(tokens: flat))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let ids = inputs.reshaped([-1]).asArray(Int32.self).map(Int.init)
        forwarded.append(contentsOf: ids)
        if let mamba = cache?.first as? MambaCache {
            var value = Int(mamba.state.first?.asArray(Float.self).first ?? 0)
            for id in ids { value = (value * 7 + id) % 997 }
            mamba.state = [MLXArray([Float(value), Float(mamba.offset + ids.count)]).reshaped([1, 1, 2])]
            mamba.offset += ids.count
        }
        return MLXArray.zeros([1, max(ids.count, 1), vocabularySize])
    }
}

/// Declares both invariants, as NemotronHOmni does.
private final class InvariantHybridModel: RecordingHybridModel, PrefixCacheInvariantModel, @unchecked Sendable {
    var statesBeforeFirstMediaTokenAreMediaIndependent: Bool { true }
}

final class MediaIndependentPrefixCacheTests: XCTestCase {
    // Step 7: none of the boundaries below (4, 12, 15, 25) is a multiple of it.
    private let step = 7
    private let system = [1, 2, 3, 4, 5]
    private let user1 = [11, 12, 13, 14, 15, 16, 17]
    private let suffix = [201, 202, 203]
    private let assistant1 = [31, 32]
    private let imageStart = 98
    private let imageToken = 99
    private let imageEnd = 97
    private let user2 = [41, 42, 43]
    private var roots: [URL] = []

    override func tearDown() {
        for root in roots { try? FileManager.default.removeItem(at: root) }
        roots.removeAll()
        super.tearDown()
    }

    private func coordinator() -> CacheCoordinator {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        roots.append(directory)
        let result = CacheCoordinator(config: .init(usePagedCache: false, enableDiskCache: true,
            diskCacheDir: directory, modelKey: "media-independent-prefix"))
        result.setHybrid(true, requiresRecurrentSSMCompanion: true, requiresSeparateRecurrentPayload: false)
        result.setGenPromptSuffixTokens(suffix)
        return result
    }

    private func parameters() -> GenerateParameters {
        GenerateParameters(maxTokens: 2, temperature: 0, prefillStepSize: step)
    }

    /// Turn 1: text only. Boundaries: stable system seed (5 -> stored at 4), strip at 12.
    private var textPrompt: [Int] { system + user1 + suffix }

    private func textInput(policy: LMInput.CacheRestorePolicy = .standard) -> LMInput {
        let ids = textPrompt
        return LMInput(tokens: MLXArray(ids.map(Int32.init)).reshaped(1, ids.count), tokenIds: ids,
            cachePrefixTokenCounts: [system.count, ids.count - suffix.count],
            cacheStablePrefixTokenCounts: [system.count], cacheRestorePolicy: policy)
    }

    /// Turn 2: the chat continues with an image. The first `<image>` token sits
    /// at index 15 (system 5 + user 7 + assistant 2 + `<img>`), so boundaries
    /// 4 and 12 are media-independent and 25 (the strip boundary) is not.
    private var mediaPrompt: [Int] {
        system + user1 + assistant1 + [imageStart] + Array(repeating: imageToken, count: 7) + [imageEnd]
            + user2 + suffix
    }
    private var firstMediaIndex: Int { system.count + user1.count + assistant1.count + 1 }

    private func mediaInput(pixelSeed: Float, policy: LMInput.CacheRestorePolicy = .standard) -> LMInput {
        let ids = mediaPrompt
        let strip = ids.count - suffix.count
        return LMInput(
            text: .init(tokens: MLXArray(ids.map(Int32.init)).reshaped(1, ids.count), tokenIds: ids),
            image: .init(pixels: MLXArray([pixelSeed, 0, 0, 0]).reshaped(1, 1, 2, 2)),
            mediaTokenIds: [imageToken],
            cachePrefixTokenCounts: [system.count, system.count + user1.count, strip],
            cacheStablePrefixTokenCounts: [system.count],
            cacheRestorePolicy: policy)
    }

    private func textSalt() -> String? { computeCacheSalt(for: textInput(), parameters: parameters()) }

    func testPlanKeysPreMediaBoundariesUnderTheTextTurnsExactSalt() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let input = mediaInput(pixelSeed: 1)
        let requestSalt = computeCacheSalt(for: input, parameters: parameters())
        let plan = CacheBoundarySaltPlan.make(requestSalt: requestSalt, input: input,
            lookupTokenIds: mediaPrompt, model: InvariantHybridModel(), parameters: parameters())
        XCTAssertEqual(plan.mediaIndependentTokenCount, firstMediaIndex)
        XCTAssertNotEqual(plan.mediaIndependentSalt, requestSalt)
        // The text-only salt of the media request IS the salt the text turn used.
        XCTAssertEqual(plan.mediaIndependentSalt, textSalt())
        XCTAssertEqual(plan.salt(forBoundary: 4), textSalt())
        XCTAssertEqual(plan.salt(forBoundary: firstMediaIndex), textSalt())
        XCTAssertEqual(plan.salt(forBoundary: firstMediaIndex + 1), requestSalt)
        XCTAssertEqual(plan.salt(forBoundary: 25), requestSalt)
        // No capability: every boundary keeps the request salt.
        let plain = CacheBoundarySaltPlan.make(requestSalt: requestSalt, input: input,
            lookupTokenIds: mediaPrompt, model: RecordingHybridModel(), parameters: parameters())
        XCTAssertEqual(plain.mediaIndependentTokenCount, 0)
        XCTAssertNil(plain.mediaIndependentPrefix)
        XCTAssertEqual(plain.salt(forBoundary: 4), requestSalt)
        // Text-only requests never get a media-independent prefix.
        let text = CacheBoundarySaltPlan.make(requestSalt: textSalt(), input: textInput(),
            lookupTokenIds: textPrompt, model: InvariantHybridModel(), parameters: parameters())
        XCTAssertNil(text.mediaIndependentPrefix)
    }

    func testImageTurnReusesTheTextTurnsPreMediaRowsAndOtherImagesNeverSharePostMediaRows() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let disk = coordinator()
        // Turn 1 (text): stores the seed at 4 and the strip boundary at 12.
        var first = try TokenIterator(input: textInput(), model: InvariantHybridModel(),
            parameters: parameters(), cacheCoordinator: disk)
        first.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
        XCTAssertTrue(disk.hasDurableDiskEntry(tokens: Array(textPrompt.prefix(12)), mediaSalt: textSalt()))

        // Turn 2 (image, partial-hit shape: same 12-token prefix, different
        // suffix): restores the text turn's row at 12 and prefills the rest.
        let warmModel = InvariantHybridModel()
        var warm = try TokenIterator(input: mediaInput(pixelSeed: 1), model: warmModel,
            parameters: parameters(), cacheCoordinator: disk)
        XCTAssertEqual(warm.cacheSaltPlan.mediaIndependentTokenCount, firstMediaIndex)
        XCTAssertEqual(warmModel.forwarded, Array(mediaPrompt.dropFirst(12)),
                       "the image turn must resume from the text turn's 12-token row")
        // The restored state equals the cold state after the same tokens.
        let cold = RecordingHybridModel()
        let coldCache = cold.newCache(parameters: parameters())
        _ = cold(MLXArray(mediaPrompt.map(Int32.init)).reshaped(1, mediaPrompt.count), cache: coldCache)
        MLX.eval(coldCache)
        XCTAssertEqual(warm.cache[0].state[0].asArray(Float.self), coldCache[0].state[0].asArray(Float.self))
        warm.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
        let imageSalt = computeCacheSalt(for: mediaInput(pixelSeed: 1), parameters: parameters())
        let strip = Array(mediaPrompt.prefix(mediaPrompt.count - suffix.count))
        XCTAssertTrue(disk.hasDurableDiskEntry(tokens: strip, mediaSalt: imageSalt),
                      "the post-media strip boundary is keyed under the image salt")
        XCTAssertFalse(disk.hasDurableDiskEntry(tokens: strip, mediaSalt: textSalt()),
                       "a post-media boundary must never be keyed under the text salt")

        // Turn 3: identical tokens, a DIFFERENT image. The post-media row of
        // turn 2 must not be served; the longest pre-media row (12) is.
        let otherModel = InvariantHybridModel()
        _ = try TokenIterator(input: mediaInput(pixelSeed: 2), model: otherModel,
            parameters: parameters(), cacheCoordinator: disk)
        XCTAssertEqual(otherModel.forwarded, Array(mediaPrompt.dropFirst(12)))
        let otherSalt = computeCacheSalt(for: mediaInput(pixelSeed: 2), parameters: parameters())
        if case .hit(let matched, _, _, _, _, _) = disk.fetch(
            tokens: mediaPrompt, mediaSalt: otherSalt, skipExactDiskBoundary: true,
            preferredDiskBoundaries: [system.count],
            mediaIndependentPrefix: .init(salt: textSalt(), tokenCount: firstMediaIndex))
        {
            XCTAssertEqual(matched, 12)
            XCTAssertLessThanOrEqual(matched, firstMediaIndex)
        } else {
            XCTFail("the pre-media row must be reachable under another image's salt")
        }

        // Turn 4: the SAME image again resumes from its own strip row (25).
        let sameModel = InvariantHybridModel()
        _ = try TokenIterator(input: mediaInput(pixelSeed: 1), model: sameModel,
            parameters: parameters(), cacheCoordinator: disk)
        XCTAssertEqual(sameModel.forwarded, Array(mediaPrompt.dropFirst(strip.count)))
    }

    func testTextTurnReusesTheSeedAnImageTurnStoredUnderTheTextSalt() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let disk = coordinator()
        var image = try TokenIterator(input: mediaInput(pixelSeed: 3), model: InvariantHybridModel(),
            parameters: parameters(), cacheCoordinator: disk)
        image.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
        let seed = Array(system.dropLast())
        XCTAssertTrue(disk.hasDurableDiskEntry(tokens: seed, mediaSalt: textSalt()),
                      "the stable seed before the image is stored under the text salt")
        let textModel = InvariantHybridModel()
        _ = try TokenIterator(input: textInput(), model: textModel, parameters: parameters(),
            cacheCoordinator: disk)
        XCTAssertEqual(textModel.forwarded, Array(textPrompt.dropFirst(seed.count)),
                       "a later text turn restores the seed the image turn wrote")
    }

    func testModelWithoutTheInvariantKeepsMediaSaltedKeysOnly() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let disk = coordinator()
        var first = try TokenIterator(input: textInput(), model: RecordingHybridModel(),
            parameters: parameters(), cacheCoordinator: disk)
        first.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
        XCTAssertTrue(disk.hasDurableDiskEntry(tokens: Array(textPrompt.prefix(12)), mediaSalt: textSalt()))
        let warmModel = RecordingHybridModel()
        var warm = try TokenIterator(input: mediaInput(pixelSeed: 1), model: warmModel,
            parameters: parameters(), cacheCoordinator: disk)
        XCTAssertNil(warm.cacheSaltPlan.mediaIndependentPrefix)
        XCTAssertEqual(warmModel.forwarded, mediaPrompt, "today's behaviour: no text-row reuse, full prefill")
        warm.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
        let imageSalt = computeCacheSalt(for: mediaInput(pixelSeed: 1), parameters: parameters())
        XCTAssertTrue(disk.hasDurableDiskEntry(tokens: Array(system.dropLast()), mediaSalt: imageSalt))
        // A media-only chat on such a model keys even the pre-media seed
        // under the image salt, so a later text turn cannot reuse it.
        let mediaOnly = coordinator()
        var only = try TokenIterator(input: mediaInput(pixelSeed: 1), model: RecordingHybridModel(),
            parameters: parameters(), cacheCoordinator: mediaOnly)
        only.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
        XCTAssertTrue(mediaOnly.hasDurableDiskEntry(tokens: Array(system.dropLast()), mediaSalt: imageSalt))
        XCTAssertFalse(mediaOnly.hasDurableDiskEntry(tokens: Array(system.dropLast()), mediaSalt: textSalt()),
                       "without the invariant the seed is only ever keyed under the image salt")
    }

    /// The batched lane routes every boundary through the slot's plan and
    /// hands the media-independent prefix to the coordinator's fetch.
    func testBatchEngineRoutesSaltsThroughTheSlotPlan() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let source = try String(contentsOf: root.appendingPathComponent(
            "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift"), encoding: .utf8)
        XCTAssertTrue(source.contains("mediaIndependentPrefix: saltPlan.mediaIndependentPrefix)"))
        XCTAssertTrue(source.contains("slot.cacheSaltPlan = saltPlan"))
        // Every disk/SSM store of a prompt boundary keys by the boundary's salt.
        for pattern in [
            "mediaSalt: slot.cacheSalt(forBoundary: tokens.count),",
            "mediaSalt: slot.cacheSalt(forBoundary: promptTokens.count),",
            "mediaSalt: slot.cacheSalt(forBoundary: boundaryTokens.count))",
            "mediaSalt: slot.cacheSalt(forBoundary: strippedTokens.count))",
        ] {
            XCTAssertTrue(source.contains(pattern), pattern)
        }
    }
}
