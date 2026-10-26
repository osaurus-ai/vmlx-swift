// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// dSpark speculative decode (MLXLMCommon/SpecDec/DSpark.swift), CPU only, no model weights:
//   - greedy identity: the spec loop returns exactly the AR tokens on a small synthetic attention
//     target (a wrapping rotating window plus a simple cache), with a draft that is sometimes right,
//     always wrong, or always right, and leaves the cache exactly where AR leaves it;
//   - in-place checkpoints: restore puts the same cache objects back, including after the rotating
//     window wrapped (no longer trimmable) and after in-place single-token writes;
//   - stop tokens and the token budget cut a block, so the cache never runs ahead of the output;
//   - the wall-clock governor under a synthetic clock (port of the Python
//     `dsv41_dspark_governor_test.py`): zero acceptance stays within a few percent of AR, a slow
//     partially-accepting draft falls back, high acceptance beats AR, acceptance returning
//     mid-stream recovers, and neither mode is terminal;
//   - the draft-window hidden tail: record / replace / truncate / prune / read-back by position;
//   - sampled speculation: with any drafts (uncoupled or coupled through the noise), the output
//     is token for token the position-keyed AR sampler's (each position's token is the AR
//     sampler's draw with that position's Gumbel noise), with filters, and the cache ends where
//     that AR run's does.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

// MARK: - Synthetic attention target

/// Two attention layers over a rotating window (wraps at 6 positions, so it stops being trimmable
/// early, like DSV4's local window) and a simple cache. The next token depends on the cached keys,
/// so a restore that left rejected rows behind changes the output.
final class DSparkToyTarget {
    let vocab = 23
    let dim = 8
    let embed: MLXArray
    let out: MLXArray
    var forwards = 0
    /// Verify capture (the Swift analogue of `capture_verify`): each layer's kv rows of the next
    /// forward, replayed by `commitPrefix`.
    var capturing = false
    var captured: [MLXArray] = []

    init(seed: UInt64 = 5) {
        var rng = SplitMix64(seed: seed)
        func values(_ n: Int) -> [Float] {
            (0 ..< n).map { _ in Float(Double(rng.next() % 20_001) / 10_000.0 - 1.0) }
        }
        embed = MLXArray(values(vocab * dim), [vocab, dim])
        out = MLXArray(values(dim * vocab), [dim, vocab])
    }

    func newCache() -> [KVCache] { [RotatingKVCache(maxSize: 6, keep: 0), KVCacheSimple()] }

    /// Logits `[1, L, vocab]`.
    func forward(_ ids: [Int], cache: [KVCache]) -> MLXArray {
        forwards += 1
        let L = ids.count
        var x = take(embed, MLXArray(ids.map { Int32($0) }), axis: 0).reshaped(1, L, dim)
        if capturing { captured = [] }
        for layer in cache {
            let mask = layer.makeMask(n: L, windowSize: nil, returnArray: false)
            let kv = x.reshaped(1, 1, L, dim)
            if capturing { captured.append(kv) }
            let (k, v) = layer.update(keys: kv, values: kv)
            let a = MLXFast.scaledDotProductAttention(
                queries: kv * 1.7, keys: k, values: v, scale: 1 / Float(dim).squareRoot(), mask: mask)
            x = x + a.reshaped(1, L, dim)
        }
        return matmul(x, out)
    }

    /// Keep only the first `keep` tokens of the captured forward: restore, re-append their rows.
    func commitPrefix(cache: [KVCache], checkpoint: SpeculativeCacheListCheckpoint, keep: Int) -> Bool {
        guard captured.count == cache.count else { return false }
        checkpoint.restore()
        for (layer, kv) in zip(cache, captured) {
            let rows = kv[0..., 0..., 0 ..< keep, 0...]
            _ = layer.update(keys: rows, values: rows)
        }
        captured = []
        return true
    }

    func argmaxLast(_ logits: MLXArray) -> Int {
        argMax(logits[0, -1], axis: -1).item(Int.self)
    }

    /// The sampled token for the position the cache is about to take (its offset), from logits
    /// `[1, L, vocab]`: the AR sampler's draw with that position's noise.
    func sampleLast(_ logits: MLXArray, cache: [KVCache], sampling: DSparkSampling) -> Int {
        let position = cache.map(\.offset).max() ?? 0
        return sampling.pick(
            logits[0, -1].reshaped(1, -1), noise: sampling.noise(first: position, count: 1, vocab: vocab)
        ).item(Int.self)
    }

    /// Position-keyed sampled AR: the prompt prefill, then one forward per returned token.
    func sampled(prompt: [Int], count: Int, sampling: DSparkSampling) -> (tokens: [Int], cache: [KVCache]) {
        let cache = newCache()
        var logits = forward(prompt, cache: cache)
        var out: [Int] = []
        while out.count < count {
            let t = sampleLast(logits, cache: cache, sampling: sampling)
            out.append(t)
            logits = forward([t], cache: cache)
        }
        return (out, cache)
    }

    /// Plain greedy AR: the prompt prefill, then one forward per returned token.
    func greedy(prompt: [Int], count: Int) -> (tokens: [Int], cache: [KVCache]) {
        let cache = newCache()
        var y = argmaxLast(forward(prompt, cache: cache))
        var out: [Int] = []
        while out.count < count {
            out.append(y)
            y = argmaxLast(forward([y], cache: cache))
        }
        return (out, cache)
    }
}

/// Drafts from the known AR continuation: the first `good(position)` tokens of each block right,
/// the rest wrong.
final class DSparkScriptedDraft: DSparkDraftSession {
    let reference: [Int: Int]  // absolute position -> AR token
    let blockSize: Int
    let vocab: Int
    let good: (Int) -> Int
    let onDraft: () -> Void
    var calls = 0
    /// Report a vocabulary, so the decoder drafts sampled steps through the coupled entry point
    /// (and reuses its noise rows for the verify).
    var coupled = false
    var coupledCalls = 0
    var vocabularySize: Int? { coupled ? vocab : nil }
    /// Set to let the decoder commit partial accepts in place through the toy target.
    var target: DSparkToyTarget?

    init(
        reference: [Int: Int], blockSize: Int = 5, vocab: Int, good: @escaping (Int) -> Int,
        onDraft: @escaping () -> Void = {}
    ) {
        self.reference = reference
        self.blockSize = blockSize
        self.vocab = vocab
        self.good = good
        self.onDraft = onDraft
    }

    func draft(seed: Int, cache: [KVCache]) -> [Int] {
        calls += 1
        onDraft()
        let end = cache[0].offset  // seed sits at `end`; drafts cover end+1 ... end+blockSize
        let g = good(end)
        return (0 ..< blockSize).map { i in
            let truth = reference[end + 1 + i] ?? 0
            return i < g ? truth : (truth + 1) % vocab
        }
    }

    func draft(seed: Int, cache: [KVCache], noise: MLXArray, inverseTemperature: Float) -> [Int] {
        precondition(noise.shape == [blockSize, vocab])
        coupledCalls += 1
        return draft(seed: seed, cache: cache)
    }

    func beginVerifyCapture(cache: [KVCache]) -> Bool {
        guard let target else { return false }
        target.capturing = true
        return true
    }

    func commitVerifiedPrefix(
        cache: [KVCache], checkpoint: SpeculativeCacheListCheckpoint, keep: Int
    ) -> Bool {
        target?.commitPrefix(cache: cache, checkpoint: checkpoint, keep: keep) ?? false
    }

    func endVerifyCapture(cache: [KVCache]) {
        target?.capturing = false
        target?.captured = []
    }
}

/// Mimics TokenIterator: queue committed tokens, take an ordinary step on `.autoregressive`.
func dsparkRun(
    decoder: inout DSparkDecoder, cache: [KVCache], firstToken: Int, count: Int,
    forward: ([Int]) -> MLXArray, argmaxLast: (MLXArray) -> Int,
    stopAtStopToken: Bool = false
) -> [Int] {
    var y = firstToken
    var queue: [Int] = []
    var out: [Int] = []
    func emit(_ t: Int) -> Bool {
        out.append(t)
        return stopAtStopToken && decoder.stopTokenIDs.contains(t)
    }
    while out.count < count {
        if !queue.isEmpty {
            if emit(queue.removeFirst()) { break }
            continue
        }
        let seed = y
        switch decoder.step(seed: seed, cache: cache, remaining: count - out.count, forward: forward) {
        case .autoregressive:
            y = argmaxLast(forward([seed]))
            if emit(seed) { break }
        case .committed(let tokens, let next):
            y = next.item(Int.self)
            queue = Array(tokens.dropFirst())
            if emit(tokens[0]) { break }
        }
        if stopAtStopToken, let last = out.last, decoder.stopTokenIDs.contains(last) { break }
    }
    decoder.finish()
    return out
}

func dsparkCachesMatch(_ a: [KVCache], _ b: [KVCache]) -> Bool {
    guard a.count == b.count else { return false }
    for (x, y) in zip(a, b) {
        guard x.offset == y.offset else { return false }
        let kx: MLXArray?
        let ky: MLXArray?
        if let rx = x as? RotatingKVCache, let ry = y as? RotatingKVCache {
            // A multi-row write may keep up to maxSize + S - 1 rows; only the last maxSize are live.
            func live(_ c: RotatingKVCache) -> MLXArray? {
                guard let k = c.temporallyOrderedKV()?.keys else { return nil }
                let n = Swift.min(k.dim(2), c.maxSize ?? k.dim(2))
                return k[0..., 0..., (k.dim(2) - n)..., 0...]
            }
            kx = live(rx)
            ky = live(ry)
        } else if let sx = x as? KVCacheSimple, let sy = y as? KVCacheSimple {
            kx = sx.readKV()?.keys
            ky = sy.readKV()?.keys
        } else {
            return false
        }
        guard let kx, let ky, kx.shape == ky.shape,
            allClose(kx, ky, rtol: 1e-4, atol: 1e-4).item(Bool.self)
        else { return false }
    }
    return true
}

/// The synthetic target as a `LanguageModel` with a dSpark head, to drive `TokenIterator`.
final class DSparkToyLanguageModel: Module, LanguageModel, DSparkSpeculativeModel {
    let target = DSparkToyTarget()
    var reference: [Int: Int] = [:]
    var attached = false
    let good: (Int) -> Int

    init(good: @escaping (Int) -> Int) { self.good = good }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { target.newCache() }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        target.forward(inputs.reshaped(-1).asArray(Int32.self).map(Int.init), cache: cache ?? [])
    }

    var dsparkBlockSize: Int? { attached ? 5 : nil }
    func attachDSparkDraft(directory: URL) throws -> Bool {
        attached = true
        return true
    }
    func makeDSparkDraftSession() -> (any DSparkDraftSession)? {
        attached ? DSparkScriptedDraft(reference: reference, vocab: target.vocab, good: good) : nil
    }
    func dsparkCacheUnsupportedReason(_ cache: [KVCache]) -> String? {
        SpeculativeCacheListCheckpoint.supports(cache) ? nil : "unsupported"
    }
}

@Suite("dSpark speculative decode", .serialized)
struct DSparkSpeculationFocusedTests {

    // MARK: Greedy identity

    static func identityRun(
        good: @escaping (Int) -> Int, count: Int = 60, commitInPlace: Bool = false
    ) -> (spec: [Int], ar: [Int], cachesMatch: Bool, stats: DSparkGenerationStats) {
        Device.withDefaultDevice(.cpu) {
            let target = DSparkToyTarget()
            let prompt = [3, 1, 4, 1, 5, 9, 2]
            let ar = target.greedy(prompt: prompt, count: count + 8)
            var reference: [Int: Int] = [:]
            for (i, t) in ar.tokens.enumerated() { reference[prompt.count + i] = t }
            let draft = DSparkScriptedDraft(reference: reference, vocab: target.vocab, good: good)
            if commitInPlace { draft.target = target }
            let cache = target.newCache()
            let first = target.argmaxLast(target.forward(prompt, cache: cache))
            // Governor that always speculates (AR re-measures are disabled) so every step verifies.
            var decoder = DSparkDecoder(
                session: draft, blockSize: 5, stopTokenIDs: [],
                governor: DSparkGovernor(probeEvery: 1, arProbeEvery: 1 << 30, warmupSpec: 1 << 30))
            let spec = dsparkRun(
                decoder: &decoder, cache: cache, firstToken: first, count: count,
                forward: { target.forward($0, cache: cache) }, argmaxLast: target.argmaxLast)
            let arCache = target.greedy(prompt: prompt, count: count).cache
            return (spec, Array(ar.tokens.prefix(count)), dsparkCachesMatch(cache, arCache), decoder.stats)
        }
    }

    @Test("partially right drafts: output and cache equal greedy AR (restore + commit path)")
    func identityPartial() {
        let r = Self.identityRun(good: { [0, 1, 2, 3, 5, 1, 4][$0 % 7] })
        #expect(r.spec == r.ar)
        #expect(r.cachesMatch)
        #expect(r.stats.commitForwards > 0 && r.stats.fullAccepts > 0)
        #expect(r.stats.accepted > 0 && r.stats.accepted < r.stats.drafted)
    }

    @Test("partial accepts committed in place (no second forward): output and cache equal greedy AR")
    func identityPartialInPlace() {
        let r = Self.identityRun(good: { [0, 1, 2, 3, 5, 1, 4][$0 % 7] }, commitInPlace: true)
        #expect(r.spec == r.ar)
        #expect(r.cachesMatch)
        #expect(r.stats.prefixCommits > 0)
        // Only k == 0 steps (and none of the partial ones) still refeed.
        #expect(r.stats.commitForwards < r.stats.specSteps - r.stats.fullAccepts - r.stats.prefixCommits + 1)
        let refeed = Self.identityRun(good: { [0, 1, 2, 3, 5, 1, 4][$0 % 7] })
        #expect(r.stats.commitForwards < refeed.stats.commitForwards)
    }

    @Test("always-wrong drafts: still exactly greedy AR")
    func identityZeroAccept() {
        let r = Self.identityRun(good: { _ in 0 })
        #expect(r.spec == r.ar)
        #expect(r.cachesMatch)
        #expect(r.stats.accepted == 0)
    }

    @Test("always-right drafts: no-refeed fast path, one forward per step, exactly greedy AR")
    func identityFullAccept() {
        let r = Self.identityRun(good: { _ in 5 })
        #expect(r.spec == r.ar)
        #expect(r.cachesMatch)
        // Only the last, budget-cut block may need a commit forward.
        #expect(r.stats.commitForwards <= 1)
        #expect(Double(r.stats.specEmitted) / Double(r.stats.specSteps) > 5)
    }

    // MARK: Sampled identity

    static func sampledRun(
        good: @escaping (Int) -> Int, sampling: DSparkSampling, count: Int = 60,
        commitInPlace: Bool = false, coupled: Bool = false
    ) -> (spec: [Int], ar: [Int], cachesMatch: Bool, stats: DSparkGenerationStats, coupledCalls: Int) {
        Device.withDefaultDevice(.cpu) {
            let target = DSparkToyTarget()
            let prompt = [3, 1, 4, 1, 5, 9, 2]
            let ar = target.sampled(prompt: prompt, count: count + 8, sampling: sampling)
            var reference: [Int: Int] = [:]
            for (i, t) in ar.tokens.enumerated() { reference[prompt.count + i] = t }
            let draft = DSparkScriptedDraft(reference: reference, vocab: target.vocab, good: good)
            draft.coupled = coupled
            if commitInPlace { draft.target = target }
            let cache = target.newCache()
            let first = target.sampleLast(target.forward(prompt, cache: cache), cache: cache, sampling: sampling)
            var decoder = DSparkDecoder(
                session: draft, blockSize: 5, stopTokenIDs: [], sampling: sampling,
                governor: DSparkGovernor(probeEvery: 1, arProbeEvery: 1 << 30, warmupSpec: 1 << 30))
            let spec = dsparkRun(
                decoder: &decoder, cache: cache, firstToken: first, count: count,
                forward: { target.forward($0, cache: cache) },
                argmaxLast: { target.sampleLast($0, cache: cache, sampling: sampling) })
            let arCache = target.sampled(prompt: prompt, count: count, sampling: sampling).cache
            return (
                spec, Array(ar.tokens.prefix(count)), dsparkCachesMatch(cache, arCache), decoder.stats,
                draft.coupledCalls
            )
        }
    }

    @Test("sampled: partially right drafts give the position-keyed AR samples (restore + commit path)")
    func sampledIdentityPartial() throws {
        let sampling = try #require(DSparkSampling(temperature: 0.9, seed: 11))
        let r = Self.sampledRun(good: { [0, 1, 2, 3, 5, 1, 4][$0 % 7] }, sampling: sampling)
        #expect(r.spec == r.ar)
        #expect(r.cachesMatch)
        #expect(r.stats.sampled && r.stats.commitForwards > 0 && r.stats.fullAccepts > 0)
        #expect(r.stats.accepted > 0 && r.stats.accepted < r.stats.drafted)
    }

    @Test("sampled with filters and in-place commits, coupled drafting: still the position-keyed AR samples")
    func sampledIdentityFilteredInPlaceCoupled() throws {
        let sampling = try #require(DSparkSampling(temperature: 1.3, topP: 0.9, topK: 7, minP: 0.02, seed: 4))
        let r = Self.sampledRun(
            good: { [0, 1, 2, 3, 5, 1, 4][$0 % 7] }, sampling: sampling, commitInPlace: true, coupled: true)
        #expect(r.spec == r.ar)
        #expect(r.cachesMatch)
        #expect(r.stats.prefixCommits > 0 && r.coupledCalls == r.stats.specSteps)
    }

    @Test("sampled: always-wrong and always-right drafts")
    func sampledIdentityExtremes() throws {
        let sampling = try #require(DSparkSampling(temperature: 0.7, seed: 99))
        let wrong = Self.sampledRun(good: { _ in 0 }, sampling: sampling)
        #expect(wrong.spec == wrong.ar && wrong.cachesMatch && wrong.stats.accepted == 0)
        let right = Self.sampledRun(good: { _ in 5 }, sampling: sampling, coupled: true)
        #expect(right.spec == right.ar && right.cachesMatch)
        #expect(Double(right.stats.specEmitted) / Double(right.stats.specSteps) > 5)
    }

    @Test("sampling: greedy requests have none; noise is keyed by seed and position; the sampled-off switch")
    func samplingContract() throws {
        #expect(DSparkSampling(temperature: 0, seed: 1) == nil)
        #expect(DSparkSampling(parameters: GenerateParameters(temperature: 0)) == nil)
        try Device.withDefaultDevice(.cpu) {
            let a = try #require(DSparkSampling(temperature: 1, seed: 7))
            let b = try #require(DSparkSampling(temperature: 1, seed: 8))
            let rows = a.noise(first: 10, count: 3, vocab: 50)
            #expect(rows.shape == [3, 50])
            #expect(allClose(rows[1 ..< 3], a.noise(first: 11, count: 2, vocab: 50)).item(Bool.self))
            #expect(!allClose(rows, b.noise(first: 10, count: 3, vocab: 50)).item(Bool.self))
        }
        #expect(DSparkEnvironment.sampledDisabled(["VMLX_DSV41_DSPARK_SAMPLED": "0"]))
        #expect(DSparkEnvironment.sampledDisabled(["VMLX_DSV41_DSPARK_SAMPLED": "off"]))
        #expect(!DSparkEnvironment.sampledDisabled([:]))
        #expect(!DSparkEnvironment.sampledDisabled(["VMLX_DSV41_DSPARK_SAMPLED": "1"]))
    }

    // MARK: Checkpoints

    @Test("in-place restore after the rotating window wrapped puts the exact pre-verify state back")
    func rotatingRestoreAfterWrap() {
        Device.withDefaultDevice(.cpu) {
            let target = DSparkToyTarget()
            let cache = target.newCache()
            _ = target.forward(Array(0 ..< 9), cache: cache)
            for t in [4, 7, 2] { _ = target.forward([t], cache: cache) }  // in-place ring writes
            #expect(!cache[0].isTrimmable)
            let before = cache.map { $0.copy() }
            let objects = cache.map { ObjectIdentifier($0 as AnyObject) }
            let ck = SpeculativeCacheListCheckpoint(cache)!
            _ = target.forward([1, 2, 3, 4, 5], cache: cache)  // a rejected verify block
            _ = target.forward([6], cache: cache)  // and an in-place write on top of it
            ck.restore()
            #expect(cache.map { ObjectIdentifier($0 as AnyObject) } == objects)
            #expect(dsparkCachesMatch(cache, before))
            // Continuing from the restored cache equals continuing from the untouched copy.
            let a = target.forward([8, 9], cache: cache)
            let b = target.forward([8, 9], cache: before)
            #expect(allClose(a, b, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            #expect(dsparkCachesMatch(cache, before))
        }
    }

    @Test("a checkpoint is refused for a cache layer it cannot snapshot exactly")
    func checkpointRefusesUnknownLayers() {
        #expect(SpeculativeCacheListCheckpoint([KVCacheSimple(), RotatingKVCache(maxSize: 4)]) != nil)
        #expect(SpeculativeCacheListCheckpoint([KVCacheSimple(), QuantizedKVCache()]) == nil)
        #expect(!SpeculativeCacheListCheckpoint.supports([]))
    }

    // MARK: Stop tokens and budget

    @Test("a block is cut at a stop token and at the budget; the cache ends where AR's does")
    func stopAndBudgetCut() {
        Device.withDefaultDevice(.cpu) {
            let target = DSparkToyTarget()
            let prompt = [3, 1, 4, 1, 5, 9, 2]
            let ar = target.greedy(prompt: prompt, count: 40).tokens
            var reference: [Int: Int] = [:]
            for (i, t) in ar.enumerated() { reference[prompt.count + i] = t }
            // A stop token that occurs inside the run (not as its first token).
            let stopIndex = (3 ..< ar.count).first { i in !ar[..<i].contains(ar[i]) } ?? 10
            let stop = ar[stopIndex]
            let draft = DSparkScriptedDraft(reference: reference, vocab: target.vocab, good: { _ in 5 })
            let cache = target.newCache()
            let first = target.argmaxLast(target.forward(prompt, cache: cache))
            var decoder = DSparkDecoder(
                session: draft, blockSize: 5, stopTokenIDs: [stop],
                governor: DSparkGovernor(probeEvery: 1, arProbeEvery: 1 << 30, warmupSpec: 1 << 30))
            let out = dsparkRun(
                decoder: &decoder, cache: cache, firstToken: first, count: 40,
                forward: { target.forward($0, cache: cache) }, argmaxLast: target.argmaxLast,
                stopAtStopToken: true)
            #expect(out == Array(ar.prefix(stopIndex + 1)))
            #expect(cache.allSatisfy { $0.offset == prompt.count + out.count })

            // Budget: 7 tokens with 5-token blocks never commits an 8th.
            let cache2 = target.newCache()
            let first2 = target.argmaxLast(target.forward(prompt, cache: cache2))
            var decoder2 = DSparkDecoder(
                session: draft, blockSize: 5, stopTokenIDs: [],
                governor: DSparkGovernor(probeEvery: 1, arProbeEvery: 1 << 30, warmupSpec: 1 << 30))
            let out2 = dsparkRun(
                decoder: &decoder2, cache: cache2, firstToken: first2, count: 7,
                forward: { target.forward($0, cache: cache2) }, argmaxLast: target.argmaxLast)
            #expect(out2 == Array(ar.prefix(7)))
            #expect(cache2.allSatisfy { $0.offset == prompt.count + 7 })
        }
    }

    // MARK: Governor under a synthetic clock

    /// Port of the Python fake backbone: next token = f(token, position); a forward of L tokens
    /// costs base + perToken * L on a synthetic clock, a draft costs `draftCost`.
    static func governorRun(
        accuracy: @escaping (Int) -> Double, count: Int = 400, draftCost: Double = 0.006
    ) -> (ratio: Double, stats: DSparkGenerationStats) {
        Device.withDefaultDevice(.cpu) {
            let V = 101
            let base = 0.020
            let perToken = 0.001
            func nxt(_ t: Int, _ p: Int) -> Int { (t * 13 + p * 5 + 7) % V }
            var now = 0.0
            let cache: [KVCache] = [KVCacheSimple()]
            func forward(_ ids: [Int]) -> MLXArray {
                now += base + perToken * Double(ids.count)
                let start = cache[0].offset
                let zeros = MLXArray.zeros([1, 1, ids.count, 1])
                _ = cache[0].update(keys: zeros, values: zeros)
                var rows = [Float](repeating: 0, count: ids.count * V)
                for (i, t) in ids.enumerated() { rows[i * V + nxt(t, start + i)] = 1 }
                return MLXArray(rows, [1, ids.count, V])
            }
            func argmaxLast(_ l: MLXArray) -> Int { argMax(l[0, -1], axis: -1).item(Int.self) }
            let prompt = [3, 1, 4, 1, 5]
            var reference: [Int: Int] = [:]
            var t = nxt(prompt.last!, prompt.count - 1)
            for p in prompt.count ..< prompt.count + count + 10 {
                reference[p] = t
                t = nxt(t, p)
            }
            let draft = DSparkScriptedDraft(
                reference: reference, vocab: V,
                good: { Int((accuracy($0) * 5).rounded()) }, onDraft: { now += draftCost })
            let first = argmaxLast(forward(prompt))
            let t0 = now
            var decoder = DSparkDecoder(
                session: draft, blockSize: 5, stopTokenIDs: [], clock: { now })
            let out = dsparkRun(
                decoder: &decoder, cache: cache, firstToken: first, count: count,
                forward: forward, argmaxLast: argmaxLast)
            let expected = (0 ..< count).map { reference[prompt.count + $0]! }
            #expect(out == expected)
            return ((now - t0) / (Double(count) * (base + perToken)), decoder.stats)
        }
    }

    @Test("governor: zero acceptance stays within a few percent of AR and keeps probing")
    func governorLowAcceptance() {
        let (ratio, stats) = Self.governorRun(accuracy: { _ in 0 })
        #expect(ratio < 1.06, "ratio \(ratio)")
        #expect(Double(stats.arSteps) > 0.85 * Double(stats.emitted))
        #expect(stats.specSteps >= 400 / 17 - 2)  // probes keep happening: no permanent latch
    }

    @Test("governor: a slow draft accepting one token per block falls back to AR")
    func governorPartialAcceptanceLoses() {
        let (ratio, stats) = Self.governorRun(accuracy: { _ in 0.2 }, draftCost: 0.04)
        #expect(ratio < 1.08, "ratio \(ratio)")
        #expect(Double(stats.arSteps) > 0.8 * Double(stats.emitted))
    }

    @Test("governor: high acceptance speculates and beats AR, AR still re-measured")
    func governorHighAcceptance() {
        let (ratio, stats) = Self.governorRun(accuracy: { _ in 1 })
        #expect(ratio < 0.45, "ratio \(ratio)")
        #expect(stats.specSteps > 3 * stats.arSteps)
        #expect(stats.arSteps >= 1)
    }

    @Test("governor: acceptance returning mid-stream recovers within a probe period")
    func governorRecovers() {
        let (ratio, stats) = Self.governorRun(accuracy: { $0 < 200 ? 0 : 1 }, count: 600)
        #expect(ratio < 0.7, "ratio \(ratio)")
        #expect(stats.specSteps > 60)
    }

    @Test("governor has no terminal state")
    func governorNoTerminalState() {
        var g = DSparkGovernor(probeEvery: 4, arProbeEvery: 8)
        var seq: [DSparkGovernor.Mode] = []
        for _ in 0 ..< 200 {
            let mode = g.choose()
            seq.append(mode)
            g.record(mode, seconds: mode == .spec ? 0.05 : 0.02, emitted: 1)  // spec always slower
        }
        #expect(seq.suffix(20).contains(.spec))  // still probing at the end
        var g2 = DSparkGovernor(probeEvery: 4, arProbeEvery: 8)
        var seq2: [DSparkGovernor.Mode] = []
        for _ in 0 ..< 200 {
            let mode = g2.choose()
            seq2.append(mode)
            g2.record(mode, seconds: mode == .spec ? 0.01 : 0.02, emitted: 1)  // spec always faster
        }
        #expect(seq2.suffix(20).contains(.ar))  // AR still re-measured at the end
        #expect(seq2.suffix(20).filter { $0 == .spec }.count > 15)
    }

    // MARK: TokenIterator integration

    @Test("TokenIterator with .dspark returns the AR tokens and leaves the cache at prompt + output")
    func tokenIteratorIntegration() throws {
        try Device.withDefaultDevice(.cpu) {
            let prompt = [3, 1, 4, 1, 5, 9, 2]
            let count = 45
            let input = LMInput(tokens: MLXArray(prompt.map { Int32($0) }))
            let model = DSparkToyLanguageModel(good: { [0, 2, 5, 1, 5, 3][$0 % 6] })
            var arIt = try TokenIterator(
                input: input, model: model, parameters: GenerateParameters(maxTokens: count, temperature: 0))
            var ar: [Int] = []
            while let t = arIt.next() { ar.append(t) }
            #expect(ar.count == count && arIt.dsparkGenerationStats == nil)
            let longer = model.target.greedy(prompt: prompt, count: count + 8).tokens
            #expect(Array(longer.prefix(count)) == ar)
            for (i, t) in longer.enumerated() { model.reference[prompt.count + i] = t }

            var params = GenerateParameters(maxTokens: count, temperature: 0)
            params.draftStrategy = .dspark(drafterPath: URL(fileURLWithPath: "/nonexistent/dspark"))
            var it = try TokenIterator(input: input, model: model, parameters: params)
            var spec: [Int] = []
            while let t = it.next() { spec.append(t) }
            #expect(spec == ar)
            let stats = try #require(it.dsparkGenerationStats)
            #expect(stats.specSteps > 0 && stats.accepted > 0)
            #expect(it.cache.allSatisfy { $0.offset == prompt.count + count })

            // A sampled request runs sampled dSpark (exactness: the decoder-level sampled tests).
            let sampledModel = DSparkToyLanguageModel(good: { _ in 5 })
            var sampled = GenerateParameters(maxTokens: count, temperature: 0.7)
            sampled.randomSeed = 3
            sampled.draftStrategy = params.draftStrategy
            var sIt = try TokenIterator(input: input, model: sampledModel, parameters: sampled)
            var sOut: [Int] = []
            while let t = sIt.next() { sOut.append(t) }
            let sStats = try #require(sIt.dsparkGenerationStats)
            #expect(sampledModel.attached && sStats.sampled && sStats.specSteps > 0)
            #expect(sOut.count == count && sIt.cache.allSatisfy { $0.offset == prompt.count + count })
        }
    }

    // MARK: Hidden tail

    @Test("hidden tail: record replaces from the start position, truncates, prunes, reads by position")
    func hiddenTail() {
        Device.withDefaultDevice(.cpu) {
            func rows(_ start: Int, _ n: Int) -> MLXArray {
                MLXArray((start ..< start + n).flatMap { [Float($0), Float(-$0)] }, [n, 2])
            }
            func firsts(_ a: MLXArray?) -> [Float] {
                guard let a else { return [] }
                return a[0..., 0].asArray(Float.self)
            }
            var tail = HybridPoolHiddenTail(capacity: 8)
            tail.record(rows: rows(0, 20), startPosition: 0)  // a long prefill keeps the last 8
            #expect(tail.startPosition == 12 && tail.endPosition == 20)
            tail.record(rows: rows(20, 1), startPosition: 20)
            tail.record(rows: rows(21, 5), startPosition: 21)  // a verify block
            #expect(tail.endPosition == 26)
            tail.record(rows: rows(21, 2), startPosition: 21)  // commit after restore
            #expect(tail.endPosition == 23)
            #expect(firsts(tail.rows(from: 18, to: 23)) == [18, 19, 20, 21, 22])
            #expect(tail.rows(from: 10, to: 23) == nil)  // older than the tail
            tail.truncate(toEnd: 21)
            #expect(tail.endPosition == 21)
            #expect(firsts(tail.rows(from: 13, to: 21)) == (13 ..< 21).map(Float.init))
            #expect(tail.count <= 8 + 8)  // bounded: capacity plus at most one partial chunk
            tail.record(rows: rows(30, 2), startPosition: 30)  // a gap starts over
            #expect(tail.startPosition == 30 && tail.count == 2)
            for p in 32 ..< 132 { tail.record(rows: rows(p, 1), startPosition: p) }
            #expect(tail.count >= 8 && tail.chunks.count <= HybridPoolHiddenTail.maxChunks)
            #expect(firsts(tail.rows(from: 124, to: 132)) == (124 ..< 132).map(Float.init))
        }
    }
}
