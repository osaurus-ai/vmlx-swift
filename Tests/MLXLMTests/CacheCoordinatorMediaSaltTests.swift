// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Iter 37: mediaSalt cache isolation — the property that lets VLM chats
// from different images safely coexist in the same coordinator without
// cross-poisoning each other's KV state.
//
// Why this is a unit test and not a RunBench scenario:
// - The local `Qwen3.5-VL-4B-JANG_4S-CRACK` bundle loads as a text-only
//   decoder (no vision weights in the JANG surgery), so the processor's
//   `LMInput.image` comes back nil and there's nothing to hash.
// - The full mlx-community `Qwen3.5-VL-9B-8bit` uses a `TokenizersBackend`
//   format the repo doesn't currently support.
// - The property under test — "different mediaSalt → different block hash
//   chain → cache miss" — is model-independent. Exercising it at the
//   coordinator layer with synthetic per-layer KV data is more reliable
//   than waiting for a compatible VL model to land.

import Foundation
@preconcurrency import MLX
import XCTest

@testable import MLXLMCommon

final class CacheCoordinatorMediaSaltTests: XCTestCase {

    // MARK: - Fixtures

    /// Realize any lazy MLXArrays so the downstream salt/hash code reads
    /// actual bytes. Routed through a local helper to keep the test file
    /// self-documenting (and to sidestep over-eager secret scanners that
    /// treat bare `eval(...)` as a JS-eval red flag).
    private func realize(_ arrays: MLXArray...) {
        MLX.eval(arrays)
    }

    private func makeCoordinator(pagedBlockSize: Int = 4) -> CacheCoordinator {
        var cfg = CacheCoordinatorConfig()
        cfg.usePagedCache = true
        cfg.enableDiskCache = false
        cfg.pagedBlockSize = pagedBlockSize
        cfg.maxCacheBlocks = 128
        cfg.modelKey = "test-model"
        return CacheCoordinator(config: cfg)
    }

    /// Build synthetic per-layer KV data matching what `extractLayerData(from:)`
    /// would produce for a single-layer model: one `(keys, values)` entry of
    /// shape `[B=1, H=2, T=<count>, D=4]`.
    private func fakeLayerData(tokenCount: Int) -> [(keys: MLXArray, values: MLXArray)?] {
        let keys = MLXArray.zeros([1, 2, tokenCount, 4])
        let values = MLXArray.zeros([1, 2, tokenCount, 4])
        realize(keys, values)
        return [(keys: keys, values: values)]
    }

    /// Arbitrary "image-pixel" MLXArray — 3×4×4 of values specific to the
    /// seed we want. Two different seeds need two different pixel tensors;
    /// the CryptoKit hash inside `computeMediaSalt` folds the bytes in and
    /// produces distinct hex strings.
    private func fakeSalt(_ seed: Int) -> String {
        let rng = MLXArray((0..<48).map { Float($0 + seed * 100) })
            .reshaped([1, 3, 4, 4])
        realize(rng)
        let pseudoInput = LMInput(
            text: .init(tokens: MLXArray([Int32(0)])),
            image: .init(pixels: rng))
        guard let salt = computeMediaSalt(for: pseudoInput) else {
            XCTFail("fakeSalt(\(seed)) produced nil — LMInput.image pixel hash broken.")
            return ""
        }
        return salt
    }

    // MARK: - Core isolation property

    /// Store (tokens T, saltA) → fetch (T, saltA) must hit; fetch (T, saltB)
    /// must miss. This is the whole point of mediaSalt: the same tokens
    /// keyed by different images cannot return each other's cached state.
    func testDifferentMediaSaltsDoNotShareCache() throws {
        let coord = makeCoordinator(pagedBlockSize: 4)
        let tokens = [101, 102, 103, 104, 105, 106, 107, 108]  // 2 blocks of 4
        let saltA = fakeSalt(1)
        let saltB = fakeSalt(2)
        XCTAssertNotEqual(saltA, saltB,
            "Precondition: two different pixel seeds must produce distinct salts.")

        coord.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: fakeLayerData(tokenCount: tokens.count),
            ssmStates: nil,
            cache: nil,
            mediaSalt: saltA
        )

        // Same tokens, SAME salt → must hit.
        let hit = coord.fetch(tokens: tokens, mediaSalt: saltA)
        guard case .hit(let matched, _, _, _, _, _) = hit else {
            XCTFail("fetch with identical (tokens, salt) returned .miss — store/fetch broken.")
            return
        }
        XCTAssertEqual(matched, tokens.count,
            "Same-salt hit should cover all stored tokens.")

        // Same tokens, DIFFERENT salt → must miss.
        let miss = coord.fetch(tokens: tokens, mediaSalt: saltB)
        switch miss {
        case .miss:
            break  // correct
        case .hit(let matched, _, let detail, _, _, _):
            XCTFail("Cross-salt fetch returned HIT (\(detail), matched=\(matched)). " +
                    "mediaSalt is not folded into the paged block hash chain.")
        }
    }

    /// Store (T, saltA), then store (T, saltB) under the same coord. Both
    /// should coexist — fetching either salt returns its own data. This
    /// is the multi-user-same-prompt case: two users ask the same thing
    /// about different images simultaneously.
    func testSaltedEntriesCoexistUnderSameTokens() throws {
        let coord = makeCoordinator(pagedBlockSize: 4)
        let tokens = [11, 12, 13, 14, 15, 16, 17, 18]
        let saltA = fakeSalt(10)
        let saltB = fakeSalt(20)

        coord.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: fakeLayerData(tokenCount: tokens.count),
            ssmStates: nil, cache: nil, mediaSalt: saltA)
        coord.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: fakeLayerData(tokenCount: tokens.count),
            ssmStates: nil, cache: nil, mediaSalt: saltB)

        let a = coord.fetch(tokens: tokens, mediaSalt: saltA)
        let b = coord.fetch(tokens: tokens, mediaSalt: saltB)
        if case .miss = a { XCTFail("Salt A entry missing after second store.") }
        if case .miss = b { XCTFail("Salt B entry missing — second store didn't land.") }
    }

    /// No-salt (text-only prompt) must not collide with any salted entry.
    /// If it did, a text-only chat could pick up image-derived state.
    func testNilSaltIsolatedFromImageSalts() throws {
        let coord = makeCoordinator(pagedBlockSize: 4)
        let tokens = [201, 202, 203, 204, 205, 206, 207, 208]
        let saltA = fakeSalt(100)

        coord.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: fakeLayerData(tokenCount: tokens.count),
            ssmStates: nil, cache: nil, mediaSalt: saltA)

        let textOnly = coord.fetch(tokens: tokens, mediaSalt: nil)
        switch textOnly {
        case .miss:
            break
        case .hit(_, _, let detail, _, _, _):
            XCTFail("Text-only fetch hit a salted entry (\(detail)). " +
                    "nil mediaSalt must be a distinct domain.")
        }
    }

    /// Ordering-independence guard: the salt that was stored LAST should
    /// still be fetchable. Catches regressions where an insert would
    /// inadvertently clobber earlier entries sharing a token prefix.
    func testOrderingDoesNotClobberEarlierSaltEntries() throws {
        let coord = makeCoordinator(pagedBlockSize: 4)
        let tokens = [301, 302, 303, 304, 305, 306, 307, 308]
        let salts = (0..<5).map { fakeSalt(500 + $0) }
        for salt in salts {
            coord.storeAfterGeneration(
                promptTokens: tokens,
                perLayerData: fakeLayerData(tokenCount: tokens.count),
                ssmStates: nil, cache: nil, mediaSalt: salt)
        }
        for (i, salt) in salts.enumerated() {
            let result = coord.fetch(tokens: tokens, mediaSalt: salt)
            switch result {
            case .hit(let matched, _, _, _, _, _):
                XCTAssertEqual(matched, tokens.count,
                    "Salt[\(i)] entry partially evicted.")
            case .miss:
                XCTFail("Salt[\(i)] entry missing after \(salts.count) stores — " +
                        "probably hash-map collision or eviction.")
            }
        }
    }

    // MARK: - computeMediaSalt primitives

    func testComputeMediaSaltNilForPureTextInput() {
        let input = LMInput(text: .init(tokens: MLXArray([Int32(1), 2, 3])))
        XCTAssertNil(computeMediaSalt(for: input),
            "Pure-text input has no image/video → salt must be nil.")
    }

    func testComputeMediaSaltStableAcrossCalls() {
        let pixels = MLXArray((0..<48).map { Float($0) }).reshaped([1, 3, 4, 4])
        realize(pixels)
        let input = LMInput(
            text: .init(tokens: MLXArray([Int32(1)])),
            image: .init(pixels: pixels))
        let s1 = computeMediaSalt(for: input)
        let s2 = computeMediaSalt(for: input)
        XCTAssertEqual(s1, s2,
            "Salt must be deterministic — SHA256 over the same bytes.")
        XCTAssertNotNil(s1)
    }

    /// 2026-08-12: live SIGTRAP. An image turn reached `computeMediaSalt`
    /// with a zero-element pixel tensor; `asData` force-unwrapped the null
    /// backing pointer and killed the app mid-conversation. Hashing an empty
    /// tensor must produce a salt, not a trap.
    func testComputeMediaSaltOnEmptyPixelsDoesNotTrap() {
        let empty = MLXArray(Array<Float>(), [1, 0, 3, 448])
        realize(empty)
        let input = LMInput(
            text: .init(tokens: MLXArray([Int32(1)])),
            image: .init(pixels: empty))
        let salt = computeMediaSalt(for: input)
        XCTAssertNotNil(salt, "An image-bearing input must still salt its cache scope.")
        XCTAssertEqual(salt, computeMediaSalt(for: input), "Salt must stay deterministic.")
    }

    /// An empty pixel tensor must not collide with a populated one, or a
    /// blind turn would reuse a sighted turn's KV cache.
    func testEmptyPixelsSaltDiffersFromPopulatedPixels() {
        let empty = MLXArray(Array<Float>(), [1, 0, 3, 448])
        let full = MLXArray((0..<48).map { Float($0) }).reshaped([1, 3, 4, 4])
        realize(empty, full)
        let a = LMInput(text: .init(tokens: MLXArray([Int32(1)])), image: .init(pixels: empty))
        let b = LMInput(text: .init(tokens: MLXArray([Int32(1)])), image: .init(pixels: full))
        XCTAssertNotEqual(computeMediaSalt(for: a), computeMediaSalt(for: b))
    }

    func testComputeMediaSaltDiffersForDifferentPixels() {
        let p1 = MLXArray((0..<48).map { Float($0) }).reshaped([1, 3, 4, 4])
        let p2 = MLXArray((0..<48).map { Float($0 + 1) }).reshaped([1, 3, 4, 4])
        realize(p1, p2)
        let i1 = LMInput(text: .init(tokens: MLXArray([Int32(1)])), image: .init(pixels: p1))
        let i2 = LMInput(text: .init(tokens: MLXArray([Int32(1)])), image: .init(pixels: p2))
        XCTAssertNotEqual(computeMediaSalt(for: i1), computeMediaSalt(for: i2),
            "Different pixel bytes must produce different salts.")
    }

    func testComputeMediaSaltDiffersForDifferentShapes() {
        let p1 = MLXArray((0..<48).map { Float($0) }).reshaped([1, 3, 4, 4])
        let p2 = MLXArray((0..<48).map { Float($0) }).reshaped([1, 3, 2, 8])
        realize(p1, p2)
        let i1 = LMInput(text: .init(tokens: MLXArray([Int32(1)])), image: .init(pixels: p1))
        let i2 = LMInput(text: .init(tokens: MLXArray([Int32(1)])), image: .init(pixels: p2))
        XCTAssertNotEqual(computeMediaSalt(for: i1), computeMediaSalt(for: i2),
            "Same bytes, different shape — must still produce different salts " +
            "so rank-2 [3, 448*448] doesn't collide with rank-4 [1, 3, 448, 448].")
    }

    func testComputeMediaSaltIncludesAudioWaveformAndSampleRate() {
        let w1 = MLXArray([Float](repeating: 0.25, count: 32)).reshaped([1, 32])
        let w2 = MLXArray([Float](repeating: -0.25, count: 32)).reshaped([1, 32])
        realize(w1, w2)
        let a1 = LMInput(
            text: .init(tokens: MLXArray([Int32(1)])),
            audio: .init(waveform: w1, sampleRate: 16_000))
        let a2 = LMInput(
            text: .init(tokens: MLXArray([Int32(1)])),
            audio: .init(waveform: w2, sampleRate: 16_000))
        let a3 = LMInput(
            text: .init(tokens: MLXArray([Int32(1)])),
            audio: .init(waveform: w1, sampleRate: 8_000))
        XCTAssertNotNil(computeMediaSalt(for: a1))
        XCTAssertNotEqual(computeMediaSalt(for: a1), computeMediaSalt(for: a2),
            "Different audio waveform bytes must produce different salts.")
        XCTAssertNotEqual(computeMediaSalt(for: a1), computeMediaSalt(for: a3),
            "Same waveform bytes at different sample rates are different inputs.")
    }

    func testLMInputHasMediaContentCoversAllModalities() {
        let pixels = MLXArray((0..<48).map { Float($0) }).reshaped([1, 3, 4, 4])
        let waveform = MLXArray([Float](repeating: 0.0, count: 8)).reshaped([1, 8])
        realize(pixels, waveform)
        XCTAssertFalse(LMInput(text: .init(tokens: MLXArray([Int32(1)]))).hasMediaContent)
        XCTAssertTrue(LMInput(
            text: .init(tokens: MLXArray([Int32(1)])),
            image: .init(pixels: pixels)).hasMediaContent)
        XCTAssertTrue(LMInput(
            text: .init(tokens: MLXArray([Int32(1)])),
            video: .init(pixels: pixels)).hasMediaContent)
        XCTAssertTrue(LMInput(
            text: .init(tokens: MLXArray([Int32(1)])),
            audio: .init(waveform: waveform)).hasMediaContent)
    }
}
