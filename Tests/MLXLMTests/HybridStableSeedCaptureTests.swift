// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Stable N-1 seed capture -> disk store -> warm restore -> re-feed, through
/// the REAL solo `TokenIterator` path, on the tiny synthetic Ling 3.0 model
/// (BailingMoeV3: MLA on `KVCacheSimple` + KDA on `MambaCache`).
///
/// Live shape this reproduces (osaurus chat, vmlx 577d9c63, Raptor JANG_6M):
/// `[vmlx][cache/fetch] HIT disk boundary=3136 remaining=115` — the stable
/// system+tools prefix (3137 tokens) is persisted one token short, a new chat
/// restores it and re-feeds the last prefix token plus the user turn. With the
/// prefix cache on, temperature 0 produced a degenerate stream; with the cache
/// off it produced "Paris.". Every restored offset was 3136 and no logit was
/// non-finite, so the defect is in CONTENT, not in offsets. A second probe
/// with `remaining=116` was clean, so the re-fed segment length is a suspect.
///
/// Three questions, each a separate test so a failure names its side:
///
/// 1. Capture side — is the seed the iterator persists equal to a fresh
///    prefill of the same 36 tokens? Both production seed paths are covered:
///    the rederive path (`cacheSnapshotForBoundary`, taken whenever a hybrid
///    strip boundary exists above the stable boundary — the osaurus chat
///    shape) and the inline path (`prepareCapturingStableBoundaries`, taken
///    when no strip boundary exists).
/// 2. Restore + re-feed, iterator level — after a warm restore of the seed,
///    do the first greedy token and every recurrent state match a cache-off
///    `TokenIterator` for every re-fed length T in the sweep, for both ways
///    the iterator can split the re-fed segment?
/// 3. Restore + re-feed, model level — the same sweep driven directly on the
///    model from the SAME disk record, comparing last-position logits, to
///    separate iterator plumbing from kernel behaviour.
///
/// 37 / 36 are deliberately not multiples of the 16-token prefill step, of the
/// 4-token conv kernel, or of the 256-token KV step: a padded or aligned
/// buffer cannot masquerade as the logical boundary (dots3 `% 256` lesson).
@Suite("Hybrid stable-seed capture (Ling 3 tiny)", .serialized)
struct HybridStableSeedCaptureTests {

    /// Processor-declared stable boundary (system + tools end).
    static let stableBoundary = 37
    /// The seed the store persists for a path-dependent hybrid: N-1.
    static var seedBoundary: Int { stableBoundary - 1 }  // 36
    /// Prompt A = stable prefix (37) + user turn (15) + generation suffix (8).
    /// Raptor's `<role>ASSISTANT</role>` generation suffix is 8 tokens (live:
    /// strip boundary 3243 on a 3251-token prompt).
    static let seedPromptLength = 60
    static let generationSuffixLength = 8
    static let prefillStep = 16

    /// Re-fed suffix lengths to sweep. The prompt for suffix T is
    /// `stable(37) + T`; the restore matches 36, so the iterator re-feeds
    /// T + 1 tokens. T = 114 / 115 reproduce the live `remaining=115`
    /// (degenerate, re-fed as 107 + 8) and `remaining=116` (clean, 108 + 8)
    /// shapes exactly; 105...118 bracket them.
    static let suffixSweep: [Int] =
        Array(1 ... 20) + [31, 32, 33, 63, 64, 65, 100] + Array(105 ... 118)
        + [127, 128, 129]

    /// State tolerance, relative to the reference magnitude; same as
    /// `HybridRestoreBoundaryRoundTripTests` / `BailingMoeV3Tests`.
    static let stateTolerance: Float = 2e-2
    /// Last-position logits: bf16/chunking noise on this fixture sits at
    /// 0.01-0.03, so 5e-2 is the hard bound; anything above `realDivergence`
    /// is far outside noise and is reported as such.
    static let logitTolerance: Float = 5e-2
    static let realDivergence: Float = 1e-1
    /// Greedy-token equality is only meaningful when the reference's top-2
    /// logit margin exceeds the numerical noise of a chunk-split forward.
    static let tokenMarginFloor: Float = 5e-2

    // MARK: - Deterministic token streams

    /// Fixed LCG (not the MLX RNG the model init draws from).
    private static func lcgTokens(count: Int, seed: UInt64) -> [Int] {
        var s = seed
        return (0 ..< count).map { _ in
            s = s &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int((s >> 33) % 128)
        }
    }

    static let seedPrompt: [Int] = lcgTokens(count: seedPromptLength, seed: 0x5EED_2026)
    static var stablePrefix: [Int] { Array(seedPrompt.prefix(stableBoundary)) }

    /// A different user turn on the same stable prefix (partial hit shape).
    /// When the suffix is long enough it ends in prompt A's generation
    /// suffix, like every real chat turn does.
    static func probePrompt(suffixLength: Int) -> [Int] {
        let generationSuffix = Array(seedPrompt.suffix(generationSuffixLength))
        guard suffixLength > generationSuffixLength else {
            return stablePrefix
                + lcgTokens(count: suffixLength, seed: 0xA11CE_0000 + UInt64(suffixLength))
        }
        let userTurn = lcgTokens(
            count: suffixLength - generationSuffixLength,
            seed: 0xA11CE_0000 + UInt64(suffixLength))
        return stablePrefix + userTurn + generationSuffix
    }

    // MARK: - How the iterator splits the re-fed segment

    /// `TokenIterator.hybridStripBoundaryIndex` puts the strip boundary at
    /// `max(cachePrefixTokenCounts)`, and `prepare` prefills up to it first.
    /// After restoring 36 tokens that gives one of two shapes, depending on
    /// which boundaries the host declared for the prompt.
    private enum SplitShape: String, CaseIterable {
        /// Only the stable boundary is declared: strip at 37, so the re-fed
        /// segment is a 1-token head followed by the whole T-token user turn.
        case headOne
        /// Stable boundary + user-turn end (prompt minus the 8-token
        /// generation suffix): strip at N-8, so the head is T-7 tokens and
        /// the tail is the 8-token suffix. This is the live osaurus shape
        /// (`cache/boundaries ... all=[3137, 3243]` on the 3251 prompt).
        case genSuffix

        func prefixCounts(promptCount: Int) -> [Int]? {
            switch self {
            case .headOne:
                return [stableBoundary]
            case .genSuffix:
                let userEnd = promptCount - generationSuffixLength
                guard userEnd > stableBoundary else { return nil }
                return [stableBoundary, userEnd]
            }
        }

        /// Tokens fed before the tail once 36 tokens are restored.
        func headLength(promptCount: Int) -> Int {
            switch self {
            case .headOne: return 1
            case .genSuffix: return promptCount - generationSuffixLength - seedBoundary
            }
        }
    }

    // MARK: - Fixture

    private struct Fixture {
        let model: BailingMoeV3Model
        let mambaIndices: [Int]
        let attentionIndices: [Int]
        var parameters: GenerateParameters {
            GenerateParameters(
                maxTokens: 1, temperature: 0,
                prefillStepSize: HybridStableSeedCaptureTests.prefillStep)
        }
    }

    /// Must be called under `MLXMetalTestLock`.
    private static func makeFixture() throws -> Fixture {
        MLXRandom.seed(11)
        let config = try JSONDecoder().decode(
            BailingMoeV3Configuration.self, from: BailingMoeV3Tests.v3ConfigJSON)
        let model = BailingMoeV3Model(config)
        MLX.eval(model.parameters())

        let probe = model.newCache(parameters: nil)
        let mamba = probe.indices.filter { probe[$0] is MambaCache }
        let attention = probe.indices.filter { probe[$0] is KVCacheSimple }
        try #require(!mamba.isEmpty, "fixture has no MambaCache layers")
        try #require(!attention.isEmpty, "fixture has no KVCacheSimple layers")
        try #require(mamba.count + attention.count == probe.count)
        return Fixture(model: model, mambaIndices: mamba, attentionIndices: attention)
    }

    private final class TempDirs {
        var dirs: [URL] = []
        func make(_ tag: String) -> URL {
            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("hybrid-stable-seed-\(tag)-\(UUID().uuidString)")
            dirs.append(dir)
            return dir
        }
        deinit { dirs.forEach { try? FileManager.default.removeItem(at: $0) } }
    }

    /// Mirrors `ModelContainer.makeCacheCoordinator`: disk tier on, paged tier
    /// off (a MambaCache topology cannot restore from paged blocks anyway, and
    /// the live hit was `HIT disk`), hybrid flags from the real topology
    /// snapshot so `requiresRecurrentSSMCompanion` / separate-payload policy
    /// are the production values, not a test guess.
    private static func makeCoordinator(_ fx: Fixture, diskDir: URL) -> CacheCoordinator {
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false,
            enableDiskCache: true,
            diskCacheDir: diskDir,
            modelKey: "ling3-tiny|stable-seed"))
        let topology = ModelCacheTopologySnapshot(cache: fx.model.newCache(parameters: nil))
        coordinator.setHybrid(
            topology.requiresSSMCompanionState,
            requiresRecurrentSSMCompanion: topology.requiresRecurrentSSMCompanionState,
            requiresSeparateRecurrentPayload: topology.requiresSeparateRecurrentPayloadState)
        return coordinator
    }

    private static func makeInput(_ tokens: [Int], prefixCounts: [Int]) -> LMInput {
        LMInput(
            tokens: MLXArray(tokens.map { Int32($0) }).expandedDimensions(axis: 0),
            tokenIds: tokens,
            cachePrefixTokenCounts: prefixCounts,
            cacheStablePrefixTokenCounts: prefixCounts.contains(stableBoundary) ? [stableBoundary] : [])
    }

    private static func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    private static func magnitude(_ a: MLXArray) -> Float {
        abs(a.asType(.float32)).max().item(Float.self)
    }

    private static func scale(_ a: MLXArray) -> Float {
        max(1, magnitude(a))
    }

    /// Feed tokens exactly the way the iterator does: `model.prepare` (chunks
    /// of `prefillStep`) then the returned tail through the model. Returns the
    /// logits of the tail forward (its last position is the segment end).
    @discardableResult
    private static func feed(_ tokens: [Int], model: BailingMoeV3Model, cache: [KVCache]) throws -> MLXArray {
        precondition(!tokens.isEmpty)
        let array = MLXArray(tokens.map { Int32($0) }).expandedDimensions(axis: 0)
        let prepared = try model.prepare(
            LMInput(text: .init(tokens: array)), cache: cache, windowSize: prefillStep)
        let logits: MLXArray
        switch prepared {
        case .tokens(let tail):
            let tailTokens = tail.tokens.ndim == 1 ? tail.tokens[.newAxis] : tail.tokens
            logits = model(tailTokens, cache: cache)
        case .logits(let result):
            logits = result.logits
        }
        MLX.eval(logits)
        MLX.eval(cache.flatMap(\.state))
        return logits
    }

    private static func mambaStates(_ cache: [KVCache], _ fx: Fixture) -> [Int: [MLXArray]] {
        var out: [Int: [MLXArray]] = [:]
        for m in fx.mambaIndices { out[m] = cache[m].state }
        return out
    }

    /// Compare every recurrent slot of `candidate` against `reference`.
    /// Returns a description of the first divergence, or nil when they match.
    private static func statesDivergence(
        _ candidate: [Int: [MLXArray]], _ reference: [Int: [MLXArray]]
    ) -> String? {
        for (m, ref) in reference.sorted(by: { $0.key < $1.key }) {
            guard let cand = candidate[m] else { return "layer \(m): missing" }
            guard cand.count == ref.count else {
                return "layer \(m): \(cand.count) slots vs \(ref.count)"
            }
            for (slot, (c, r)) in zip(cand, ref).enumerated() {
                guard c.shape == r.shape else {
                    return "layer \(m) slot \(slot): shape \(c.shape) vs \(r.shape)"
                }
                let diff = maxAbsDiff(c, r)
                let s = scale(r)
                if diff > stateTolerance * s {
                    return "layer \(m) slot \(slot): maxAbsDiff \(diff) (scale \(s))"
                }
            }
        }
        return nil
    }

    private static func greedy(_ lastLogits: MLXArray) -> (token: Int, margin: Float) {
        let row = lastLogits.asType(.float32)
        let sorted = MLX.sorted(row)
        let n = row.dim(0)
        let margin = (sorted[n - 1] - sorted[n - 2]).item(Float.self)
        let token = Int(argMax(row).asType(.int32).item(Int32.self))
        return (token, margin)
    }

    private static func trace(_ line: String) {
        FileHandle.standardError.write(Data(("[vmlx][stable-seed-test] " + line + "\n").utf8))
    }

    private enum SeedTestError: Error, CustomStringConvertible {
        case invalid(String)
        var description: String {
            switch self { case .invalid(let s): return "INVALID harness: \(s)" }
        }
    }

    // MARK: - Seed production through the real iterator

    private enum SeedPath: String {
        /// `cachePrefixTokenCounts = [37, 57]` (stable end + user-turn end):
        /// the hybrid strip boundary is 57, so the 36-token seed is not
        /// captured inline and the store rebuilds it through
        /// `cacheSnapshotForBoundary` (trim refused, rederive). This is the
        /// osaurus chat shape (`store-boundary ... allowRederive=true snapshot=ok`).
        case rederive
        /// `VMLX_HYBRID_STRIPPED_STORE=0`: no strip boundary, so
        /// `prepareCapturingStableBoundaries` splits prefill at 36 and the
        /// store copies that snapshot.
        case inlineCapture
    }

    /// Run prompt A through `TokenIterator` + the post-generation store and
    /// return the disk record of the 36-token seed (fail-closed: a miss is an
    /// INVALID harness, not a pass).
    private static func produceSeed(
        _ fx: Fixture, coordinator: CacheCoordinator, path: SeedPath
    ) throws -> [String: MLXArray] {
        if path == .inlineCapture {
            setenv("VMLX_HYBRID_STRIPPED_STORE", "0", 1)
        }
        defer {
            if path == .inlineCapture { unsetenv("VMLX_HYBRID_STRIPPED_STORE") }
        }
        let userEnd = seedPromptLength - generationSuffixLength  // 57
        let promptInput = makeInput(seedPrompt, prefixCounts: [stableBoundary, userEnd])
        var iterator = try TokenIterator(
            input: promptInput,
            model: fx.model,
            parameters: fx.parameters,
            cacheCoordinator: coordinator)
        for (i, layer) in iterator.cache.enumerated() {
            try #require(layer.offset == seedPromptLength, "prefill: layer \(i) offset \(layer.offset)")
        }
        iterator.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)

        let salt = computeCacheSalt(for: promptInput, parameters: fx.parameters)
        let probe = probePrompt(suffixLength: 5)
        let result = coordinator.fetch(
            tokens: probe,
            mediaSalt: salt,
            skipExactDiskBoundary: true,
            preferredDiskBoundaries: [stableBoundary])
        guard case .hit(let matched, let remaining, let detail, _, _, let diskArrays) = result else {
            throw SeedTestError.invalid("\(path.rawValue): the stable N-1 seed was not stored (fetch missed)")
        }
        try #require(matched == seedBoundary, "\(path.rawValue): matched \(matched), expected \(seedBoundary)")
        try #require(remaining == Array(probe.dropFirst(seedBoundary)))
        try #require(detail == .disk)
        let arrays = try #require(diskArrays, "\(path.rawValue): disk hit carried no arrays")
        try #require(TQDiskSerializer.formatVersion(of: arrays) >= 2)
        for m in fx.mambaIndices {
            let off = try #require(arrays["__mamba_\(m)_offset__"], "missing __mamba_\(m)_offset__")
            #expect(off[0].item(Int32.self) == Int32(seedBoundary), "\(path.rawValue): mamba \(m) offset")
            #expect(arrays["mamba_\(m)_state0"] != nil && arrays["mamba_\(m)_state1"] != nil)
        }
        for a in fx.attentionIndices {
            let keys = try #require(arrays["kv_\(a)_keys"], "missing kv_\(a)_keys")
            #expect(keys.dim(2) == seedBoundary, "\(path.rawValue): kv_\(a)_keys seq \(keys.dim(2))")
        }
        return arrays
    }

    // MARK: - 1. Capture side

    @Test("the persisted 36-token seed equals a fresh prefill of those 36 tokens (rederive + inline paths)")
    func stableSeedRecordMatchesFreshPrefill() throws {
        try MLXMetalTestLock.withLock {
            let fx = try Self.makeFixture()
            let temp = TempDirs()

            // Reference: fresh cache, the 36 seed tokens, same chunking.
            let reference = fx.model.newCache(parameters: nil)
            try Self.feed(Array(Self.seedPrompt.prefix(Self.seedBoundary)), model: fx.model, cache: reference)
            let referenceStates = Self.mambaStates(reference, fx)
            for m in fx.mambaIndices {
                try #require(referenceStates[m]?.count == 2, "reference mamba \(m) slot count")
                // Non-empty baseline: comparing zero against zero proves nothing.
                try #require(Self.magnitude(referenceStates[m]![1]) > 0, "reference mamba \(m) recurrent state is all zero")
            }

            for path in [SeedPath.rederive, .inlineCapture] {
                let coordinator = Self.makeCoordinator(fx, diskDir: temp.make(path.rawValue))
                let record = try Self.produceSeed(fx, coordinator: coordinator, path: path)

                var restored = fx.model.newCache(parameters: nil)
                let restoredTokens = restoreFromDiskArrays(record, into: &restored)
                MLX.eval(restored.flatMap(\.state))
                #expect(restoredTokens == Self.seedBoundary, "\(path.rawValue): restored \(restoredTokens)")
                #expect(validateRestoredCacheBoundary(
                    restored, matchedTokens: Self.seedBoundary,
                    restoredTokens: restoredTokens, detail: path.rawValue))

                let divergence = Self.statesDivergence(Self.mambaStates(restored, fx), referenceStates)
                Self.trace("capture path=\(path.rawValue) states=\(divergence ?? "match")")
                #expect(divergence == nil, "\(path.rawValue): seed recurrent state != fresh 36-token prefill: \(divergence ?? "")")

                for a in fx.attentionIndices {
                    let r = reference[a].state, c = restored[a].state
                    try #require(r.count == 2 && c.count == 2)
                    let kDiff = Self.maxAbsDiff(c[0], r[0]), vDiff = Self.maxAbsDiff(c[1], r[1])
                    #expect(kDiff < Self.stateTolerance * Self.scale(r[0]), "\(path.rawValue): kv \(a) keys diverged \(kDiff)")
                    #expect(vDiff < Self.stateTolerance * Self.scale(r[1]), "\(path.rawValue): kv \(a) values diverged \(vDiff)")
                }
            }
        }
    }

    // MARK: - 2. Restore + re-feed, iterator level

    private struct IteratorRun {
        let firstToken: Int
        let offsets: [Int]
        let states: [Int: [MLXArray]]
    }

    private static func runIterator(
        _ fx: Fixture, prompt: [Int], prefixCounts: [Int], coordinator: CacheCoordinator?
    ) throws -> IteratorRun {
        let iterator = try TokenIterator(
            input: makeInput(prompt, prefixCounts: prefixCounts),
            model: fx.model,
            parameters: fx.parameters,
            cacheCoordinator: coordinator)
        // `y` is the greedy token sampled at the prompt's last position; the
        // cache sits at the full prompt (nothing decoded yet).
        let first = iterator.y.tokens.item(Int.self)
        MLX.eval(iterator.cache.flatMap(\.state))
        return IteratorRun(
            firstToken: first,
            offsets: iterator.cache.map(\.offset),
            states: mambaStates(iterator.cache, fx))
    }

    /// One-shot forward of the whole prompt on a fresh cache. `allLogits` is
    /// `[1, N, V]` so callers can compare positions other than the last one.
    private static func oneShot(
        _ fx: Fixture, prompt: [Int]
    ) throws -> (token: Int, margin: Float, logits: MLXArray, allLogits: MLXArray, states: [Int: [MLXArray]]) {
        let cache = fx.model.newCache(parameters: nil)
        let array = MLXArray(prompt.map { Int32($0) }).expandedDimensions(axis: 0)
        let logits = fx.model(array, cache: cache)
        MLX.eval(logits)
        MLX.eval(cache.flatMap(\.state))
        let last = logits[0, prompt.count - 1].asType(.float32)
        let g = greedy(last)
        return (g.token, g.margin, last, logits, mambaStates(cache, fx))
    }

    /// A restore seats the record's own `MLXArray` objects into the cache
    /// (`restoreMambaLayer` -> `ArraysCache.state`), and the next KDA forward
    /// then updates those objects IN PLACE (`ArraysCache.subscript set` ->
    /// `_updateInternal`). Production never sees this because every fetch
    /// loads fresh arrays from disk; a test that restores the same dictionary
    /// twice must hand each restore its own buffers, or the second restore
    /// seats the state the first forward already advanced.
    private static func freshCopy(_ record: [String: MLXArray]) -> [String: MLXArray] {
        let copy = record.mapValues { $0 * 1 }
        MLX.eval(Array(copy.values))
        return copy
    }

    @Test("after a warm restore of the seed, every re-fed length T and split shape gives the cache-off first token and recurrent states")
    func restoredSeedRefeedSweepMatchesCacheOff() throws {
        try MLXMetalTestLock.withLock {
            let fx = try Self.makeFixture()
            let temp = TempDirs()
            let coordinator = Self.makeCoordinator(fx, diskDir: temp.make("sweep"))
            _ = try Self.produceSeed(fx, coordinator: coordinator, path: .rederive)

            var divergent: [String] = []
            var tokenSkipped: [String] = []
            var assertedTokens = 0
            for t in Self.suffixSweep {
                let prompt = Self.probePrompt(suffixLength: t)
                let refeed = prompt.count - Self.seedBoundary
                let shot = try Self.oneShot(fx, prompt: prompt)
                let marginOK = shot.margin >= Self.tokenMarginFloor

                for shape in SplitShape.allCases {
                    guard let prefixCounts = shape.prefixCounts(promptCount: prompt.count) else { continue }
                    let label = "T=\(t) refeed=\(refeed) split=\(shape.rawValue)(head \(shape.headLength(promptCount: prompt.count)))"

                    // Cache OFF: the same iterator code with no coordinator.
                    let off = try Self.runIterator(fx, prompt: prompt, prefixCounts: prefixCounts, coordinator: nil)
                    // Cache ON: must hit the 36-token seed and re-feed `refeed` tokens.
                    let on = try Self.runIterator(fx, prompt: prompt, prefixCounts: prefixCounts, coordinator: coordinator)

                    #expect(on.offsets.allSatisfy { $0 == prompt.count }, "\(label): cache-on offsets \(Set(on.offsets).sorted())")
                    #expect(off.offsets.allSatisfy { $0 == prompt.count }, "\(label): cache-off offsets \(Set(off.offsets).sorted())")

                    // Harness control: the chunked cache-off prefill must agree
                    // with the one-shot forward whenever the greedy choice is not
                    // a near-tie, and its states must match one-shot outright.
                    if marginOK {
                        #expect(off.firstToken == shot.token, "\(label): CONTROL cache-off token \(off.firstToken) != one-shot \(shot.token) (margin \(shot.margin))")
                        assertedTokens += 1
                    } else {
                        tokenSkipped.append(label)
                    }
                    if let d = Self.statesDivergence(off.states, shot.states) {
                        Issue.record("\(label): CONTROL cache-off iterator states != one-shot: \(d)")
                    }

                    let stateDivergence = Self.statesDivergence(on.states, off.states)
                    let tokenDiverged = marginOK && on.firstToken != off.firstToken
                    Self.trace(
                        "\(label) token on=\(on.firstToken) off=\(off.firstToken) oneShot=\(shot.token) "
                            + "margin=\(shot.margin) states=\(stateDivergence ?? "match")")
                    if tokenDiverged {
                        divergent.append("\(label): token on=\(on.firstToken) off=\(off.firstToken)")
                    }
                    if let d = stateDivergence {
                        divergent.append("\(label): \(d)")
                    }
                    #expect(!tokenDiverged, "\(label): first greedy token on=\(on.firstToken) off=\(off.firstToken)")
                    #expect(stateDivergence == nil, "\(label): \(stateDivergence ?? "")")
                }
            }
            Self.trace("sweep divergent=\(divergent) tokenSkippedNearTie=\(tokenSkipped.count)")
            #expect(divergent.isEmpty, "re-fed shapes that diverge from cache-off: \(divergent)")
            // The token assertion must have fired for a majority of shapes, or
            // this suite silently degraded to a state-only check.
            #expect(assertedTokens > tokenSkipped.count, "too many near-tie prompts: \(tokenSkipped)")
        }
    }

    @Test("a warm restore + re-feed is deterministic at the live shapes (remaining 115 / 116)")
    func restoredSeedRefeedIsDeterministic() throws {
        try MLXMetalTestLock.withLock {
            let fx = try Self.makeFixture()
            let temp = TempDirs()
            let coordinator = Self.makeCoordinator(fx, diskDir: temp.make("determinism"))
            _ = try Self.produceSeed(fx, coordinator: coordinator, path: .rederive)

            for t in [114, 115] {
                let prompt = Self.probePrompt(suffixLength: t)
                for shape in SplitShape.allCases {
                    guard let prefixCounts = shape.prefixCounts(promptCount: prompt.count) else { continue }
                    let first = try Self.runIterator(fx, prompt: prompt, prefixCounts: prefixCounts, coordinator: coordinator)
                    let second = try Self.runIterator(fx, prompt: prompt, prefixCounts: prefixCounts, coordinator: coordinator)
                    #expect(first.firstToken == second.firstToken, "T=\(t) \(shape.rawValue): token \(first.firstToken) vs \(second.firstToken) across identical runs")
                    for m in fx.mambaIndices {
                        for (slot, (a, b)) in zip(first.states[m]!, second.states[m]!).enumerated() {
                            let diff = Self.maxAbsDiff(a, b)
                            #expect(diff <= 1e-6, "T=\(t) \(shape.rawValue): layer \(m) slot \(slot) differs across identical runs by \(diff)")
                        }
                    }
                }
            }
        }
    }

    // MARK: - 3. Restore + re-feed, model level (same record, no iterator)

    @Test("restoring the seed record directly and re-feeding head + tail matches one-shot logits for every T and split")
    func restoredSeedRefeedSweepDirectModel() throws {
        try MLXMetalTestLock.withLock {
            let fx = try Self.makeFixture()
            let temp = TempDirs()
            let coordinator = Self.makeCoordinator(fx, diskDir: temp.make("direct"))
            let record = try Self.produceSeed(fx, coordinator: coordinator, path: .rederive)

            var divergent: [String] = []
            for t in Self.suffixSweep {
                let prompt = Self.probePrompt(suffixLength: t)
                let shot = try Self.oneShot(fx, prompt: prompt)
                let remaining = Array(prompt.dropFirst(Self.seedBoundary))

                for shape in SplitShape.allCases {
                    guard shape.prefixCounts(promptCount: prompt.count) != nil else { continue }
                    let head = shape.headLength(promptCount: prompt.count)
                    let label = "direct T=\(t) refeed=\(remaining.count) split=\(shape.rawValue)(head \(head))"

                    var cache = fx.model.newCache(parameters: nil)
                    let restoredTokens = restoreFromDiskArrays(Self.freshCopy(record), into: &cache)
                    MLX.eval(cache.flatMap(\.state))
                    try #require(restoredTokens == Self.seedBoundary)
                    try #require(validateRestoredCacheBoundary(
                        cache, matchedTokens: Self.seedBoundary, restoredTokens: restoredTokens, detail: "direct"))

                    // The head's last position is the earliest point at which a
                    // wrong seed shows: nothing has decayed yet. Compare it to
                    // the same position of the one-shot forward.
                    let headLogits = try Self.feed(Array(remaining.prefix(head)), model: fx.model, cache: cache)
                    let headPosition = Self.seedBoundary + head - 1
                    let headDiff = Self.maxAbsDiff(
                        headLogits[0, headLogits.dim(1) - 1].asType(.float32),
                        shot.allLogits[0, headPosition].asType(.float32))
                    let logits = try Self.feed(Array(remaining.dropFirst(head)), model: fx.model, cache: cache)
                    let last = logits[0, logits.dim(1) - 1].asType(.float32)

                    let logitDiff = Self.maxAbsDiff(last, shot.logits)
                    let stateDivergence = Self.statesDivergence(Self.mambaStates(cache, fx), shot.states)
                    Self.trace("\(label) headDiff@\(headPosition)=\(headDiff) logitDiff=\(logitDiff) states=\(stateDivergence ?? "match")")
                    let severity = { (d: Float) in d >= Self.realDivergence ? "REAL divergence" : "above tolerance" }
                    if headDiff >= Self.logitTolerance {
                        divergent.append("\(label): first re-fed position \(headPosition) logits \(severity(headDiff)) \(headDiff)")
                    }
                    if logitDiff >= Self.logitTolerance {
                        divergent.append("\(label): last-position logits \(severity(logitDiff)) \(logitDiff)")
                    }
                    if let d = stateDivergence { divergent.append("\(label): \(d)") }
                    #expect(headDiff < Self.logitTolerance, "\(label): position \(headPosition) logits \(severity(headDiff)) \(headDiff)")
                    #expect(logitDiff < Self.logitTolerance, "\(label): last-position logits \(severity(logitDiff)) \(logitDiff)")
                    #expect(stateDivergence == nil, "\(label): \(stateDivergence ?? "")")
                    #expect(cache.allSatisfy { $0.offset == prompt.count }, "\(label): offsets \(Set(cache.map(\.offset)).sorted())")
                }
            }
            Self.trace("direct sweep divergent=\(divergent)")
            #expect(divergent.isEmpty, "re-fed shapes that diverge from one-shot: \(divergent)")
        }
    }
}
