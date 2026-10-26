// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 prompt-suffix pruning (port of jang_tools/dsv41 `Model.__call__` `prompt_end`,
// test_suffix_prune.py). CPU only, tiny random weights, synthetic Engram tables, no bundle.
//
// The text model has 8 layers, a 16-row window, kv sources 1 and 3 (ratios 2 and 1), index sources
// 1/3/5/7, candidate source 3 and Engram at 1 and 4, so layers 5-7 are pruned (keep 46 / 31 / 16
// rows); 5 and 7 are index sources, 7 selects among layer 3's candidates, 6 reuses 5's selection.
// For several prompt lengths and chunk plans the body is prefilled with and without pruning, then
// the last prompt token and continuation tokens are fed one at a time. Checks:
//   1. layers before the first pruned layer: every cache array bit-identical;
//   2. pruned layers: same offset, same window rows within cos 0.9999 (fewer rows per matmul);
//   3. last-token and continuation logits within cos 0.999, same argmax;
//   4. the rows each layer ran (pruning engaged, sub-chunk straddles, skipped chunks);
//   5. pruning off (flag or no prompt end) leaves the forward unchanged.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("DeepSeek-V4.1 prompt-suffix pruning (CPU)", .serialized)
struct DeepseekV41SuffixPruneFocusedTests {

    static func config() -> DeepseekV4Configuration {
        var c = DeepseekV41DSparkFocusedTests.tinyConfig()
        c.numHiddenLayers = 8
        c.slidingWindow = 16
        c.compressRatios = [0, 2, 2, 1, 1, 1, 1, 1]
        c.kvSourceLayerIds = [1, 3]
        c.indexSourceLayerIds = [1, 3, 5, 7]
        c.candidateSourceLayerId = 3
        c.candidateTopkBlocks = 3
        c.candidateBlockSize = 4
        c.indexNHeads = 2
        c.indexHeadDim = 16
        c.indexTopk = 8
        c.numHashLayers = 0
        c.engramLayerIds = [1, 4]
        c.engramNHeads = 2
        c.engramHeadDim = 8
        c.engramMaxNgramSize = 3
        return c
    }

    /// Synthetic Engram aux + one FP8 table per layer (as `writeEngramBundle`, for `layers`).
    static func writeEngramBundle(layers: [Int], vocab: Int) throws -> URL {
        typealias V = DeepseekV41VisionFocusedTests
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dsv41-prune-bundle-\(UUID().uuidString)", isDirectory: true)
        let engram = dir.appendingPathComponent("engram", isDirectory: true)
        try FileManager.default.createDirectory(at: engram, withIntermediateDirectories: true)
        let nHeads = 2, maxNgram = 3, headDim = 8, cols = (maxNgram - 1) * nHeads
        let n = layers.count
        var rng = SplitMix64(seed: 43)
        let tokenMap = (0 ..< vocab).map { _ in Int32(rng.next() % 50) }
        let multipliers = (0 ..< n * maxNgram).map { _ in Int64(rng.next() % 100_003) * 2 + 1 }
        let primes: [Int64] = (0 ..< n * (maxNgram - 1) * nHeads).map { [11, 13, 17, 19][$0 % 4] }
        let offsets = (0 ..< n * cols).map { Int64(($0 % cols) * 20) }
        try V.writeSafetensors(dir.appendingPathComponent("engram_aux.safetensors"), [
            ("token_map", "I32", [vocab], V.bytes(tokenMap)),
            ("multipliers", "I64", [n, maxNgram], V.bytes(multipliers)),
            ("primes", "I64", [n, maxNgram - 1, nHeads], V.bytes(primes)),
            ("offsets", "I64", [n, cols], V.bytes(offsets)),
            ("pad_id", "I64", [1], V.bytes([Int64(49)])),
        ], metadata: [
            "head_dim": "\(headDim)", "layer_ids": layers.map(String.init).joined(separator: ","),
            "compressed_vocab_size": "50",
        ])
        let rows = cols * 20
        for l in layers {
            let weight = (0 ..< rows * headDim).map { _ in
                UInt8(0x28 + rng.next() % 16) | UInt8((rng.next() % 2) << 7)
            }
            let scale = [UInt8](repeating: 127, count: rows)
            try V.writeSafetensors(engram.appendingPathComponent("engram-\(l).safetensors"), [
                ("layers.\(l).engram.embed.weight", "F8_E4M3", [rows, headDim], weight),
                ("layers.\(l).engram.embed.scale", "F8_E8M0", [rows, 1], scale),
            ])
        }
        return dir
    }

    static func model(bundle: URL) throws -> DeepseekV4Model {
        let model = DeepseekV4Model(config())
        try model.configureBundle(modelDirectory: bundle)
        DeepseekV41VisionFocusedTests.randomize(model, seed: 4101)
        return model
    }

    static func tokens(_ n: Int, seed: UInt64 = 5) -> [Int32] {
        var rng = SplitMix64(seed: seed)
        return (0 ..< n).map { _ in Int32(rng.next() % 120) }
    }

    /// Body prefill in chunks of `step` through the inner forward (as `prepare` does), pruned
    /// toward the body end when `prune`. Returns the cache at the body end and the rows each
    /// pruned forward ran.
    static func prefillBody(
        _ model: DeepseekV4Model, _ body: [Int32], step: Int, prune: Bool
    ) -> (cache: [KVCache], rows: [(start: Int, rows: [Int])]) {
        let cache = model.newCache(parameters: nil)
        var seen: [(start: Int, rows: [Int])] = []
        model.model.suffixPruneObserver = { seen.append(($0, $1)) }
        defer { model.model.suffixPruneObserver = nil }
        var a = 0
        while a < body.count {
            let b = min(body.count, a + step)
            let chunk = MLXArray(Array(body[a ..< b])).reshaped(1, -1)
            _ = model.model(chunk, cache: cache, vision: nil, promptEnd: prune ? body.count : nil)
            DeepseekV4Model.compactSlidingWindows(cache)
            MLX.eval(cache)
            a = b
        }
        return (cache, seen)
    }

    static func cos(_ a: MLXArray, _ b: MLXArray) -> Float {
        let x = a.asType(.float32).reshaped(-1)
        let y = b.asType(.float32).reshaped(-1)
        let d = (x * y).sum() / (sqrt((x * x).sum()) * sqrt((y * y).sum()) + 1e-12)
        return d.item(Float.self)
    }

    static func window(_ cache: KVCache) -> MLXArray? {
        let rotating = (cache as? DeepseekV4Cache)?.local ?? (cache as? RotatingKVCache)
        return rotating?.temporallyOrderedKV()?.keys
    }

    /// Per-layer differences between a full and a pruned cache at the same position.
    static func compareCaches(
        _ full: [KVCache], _ pruned: [KVCache], pruneFrom: Int
    ) -> [String] {
        var why: [String] = []
        for (i, (c, r)) in zip(full, pruned).enumerated() {
            if c.offset != r.offset { why.append("L\(i) offset \(c.offset) vs \(r.offset)") }
            if i < pruneFrom {
                let a = c.state, b = r.state
                if a.count != b.count || !zip(a, b).allSatisfy({ DeepseekV41DSparkFocusedTests.equal($0, $1) }) {
                    why.append("L\(i) state differs")
                }
            } else {
                guard let w0 = window(c), let w1 = window(r) else {
                    why.append("L\(i) missing window"); continue
                }
                let rows = c.maxSize ?? 0
                if w0.dim(2) < min(rows, c.offset) || w0.shape != w1.shape {
                    why.append("L\(i) window \(w0.shape) vs \(w1.shape)")
                } else if cos(w0, w1) < 0.9999 {
                    why.append("L\(i) window cos \(cos(w0, w1))")
                }
            }
        }
        return why
    }

    /// The last prompt token and `continuation` fed one by one on both caches: worst logits cosine
    /// and argmax flips.
    static func continuation(
        _ model: DeepseekV4Model, _ tail: [Int32], _ full: [KVCache], _ pruned: [KVCache]
    ) -> (worst: Float, flips: Int) {
        var worst: Float = 1
        var flips = 0
        for t in tail {
            let x = MLXArray([t]).reshaped(1, 1)
            let l0 = model(x, cache: full)[0, 0]
            let l1 = model(x, cache: pruned)[0, 0]
            worst = min(worst, cos(l0, l1))
            if argMax(l0).item(Int32.self) != argMax(l1).item(Int32.self) { flips += 1 }
        }
        return (worst, flips)
    }

    // MARK: Pruned == full prefill

    @Test("pruned prefill == full prefill (windows at the prompt end, last-token and continuation logits)")
    func prunedEqualsFull() throws {
        try Device.withDefaultDevice(.cpu) {
            let bundle = try Self.writeEngramBundle(layers: [1, 4], vocab: 128)
            defer { try? FileManager.default.removeItem(at: bundle) }
            let model = try Self.model(bundle: bundle)
            let pruneFrom = try #require(model.model.suffixPruneFrom)
            #expect(pruneFrom == 5)
            // keep per pruned layer: 1 + (8 - i) * 15 -> layer 5: 46, 6: 31, 7: 16 rows
            #expect((5 ..< 8).map {
                DeepseekV4ModelInner.suffixPruneKeep(layer: $0, numLayers: 8, window: 16)
            } == [46, 31, 16])
            let all = Self.tokens(400)
            for (n, step) in [(40, 64), (47, 64), (100, 64), (161, 64), (161, 32), (161, 17), (250, 256)] {
                let ids = Array(all[..<n])
                let body = Array(ids.dropLast())
                let (full, fullRows) = Self.prefillBody(model, body, step: step, prune: false)
                let (pruned, rows) = Self.prefillBody(model, body, step: step, prune: true)
                #expect(fullRows.isEmpty)
                var why = Self.compareCaches(full, pruned, pruneFrom: pruneFrom)
                // Pruning engaged: the last chunk's pruned layers ran exactly their kept rows.
                let P = body.count
                if let last = rows.last {
                    let L = P - last.start
                    for i in 5 ..< 8 {
                        let keep = DeepseekV4ModelInner.suffixPruneKeep(layer: i, numLayers: 8, window: 16)
                        if last.rows[i] != min(L, keep) { why.append("L\(i) ran \(last.rows[i]) rows") }
                    }
                    if last.rows[..<5].contains(where: { $0 != L }) { why.append("unpruned layer ran fewer rows") }
                } else {
                    why.append("no pruned forward")
                }
                let tail = Array(all[(n - 1) ..< (n + 8)])
                let (worst, flips) = Self.continuation(model, tail, full, pruned)
                if worst < 0.999 || flips > 0 { why.append("logits cos \(worst), argmax flips \(flips)") }
                #expect(why.isEmpty, "n=\(n) step=\(step): \(why.joined(separator: "; ")) (logits cos \(worst))")
                print("[suffix-prune] n=\(n) step=\(step): rows \(rows.map(\.rows)) worst logits cos \(worst), "
                    + "pruned-window cos \((5 ..< 8).map { Self.cos(Self.window(full[$0])!, Self.window(pruned[$0])!) })")
            }
        }
    }

    @Test("prepare: one sub-chunked forward (kept rows straddle a sub-chunk) and a skipped chunk")
    func prepareSubchunkedAndSkipped() throws {
        let sub = DeepseekV4Math.attnSubchunkTokens
        guard sub == 512, ProcessInfo.processInfo.environment["DSV4_PREFILL_OUTER_CHUNK"] == nil
        else { return }
        try Device.withDefaultDevice(.cpu) {
            let bundle = try Self.writeEngramBundle(layers: [1, 4], vocab: 128)
            defer { try? FileManager.default.removeItem(at: bundle) }
            let model = try Self.model(bundle: bundle)
            model.model.dsparkTapLayers = [5, 6, 7]
            model.model.dsparkTailCapacity = 16
            let all = Self.tokens(2700, seed: 9)
            // 1035 tokens: one 1034-row forward in sub-chunks 512 / 512 / 10; layer 5 keeps rows
            // 988..<1034 (pieces 36 + 10), layer 7 1018..<1034 (6 + 10).
            // 2600 tokens: forwards of 2048 (every pruned layer skipped) and 551 rows.
            for (n, expected) in [
                (1035, [[1034, 1034, 1034, 1034, 1034, 46, 31, 16]]),
                (2600, [[2048, 2048, 2048, 2048, 2048, 0, 0, 0], [551, 551, 551, 551, 551, 46, 31, 16]]),
            ] {
                let input = LMInput(tokens: MLXArray(Array(all[..<n])))
                func run(_ prune: Bool) throws -> (cache: [KVCache], rest: Int, rows: [[Int]]) {
                    model.suffixPruneEnabled = prune
                    defer { model.suffixPruneEnabled = DeepseekV4Math.suffixPruneEnabled }
                    var seen: [[Int]] = []
                    model.model.suffixPruneObserver = { seen.append($1) }
                    defer { model.model.suffixPruneObserver = nil }
                    let cache = model.newCache(parameters: nil)
                    guard case .tokens(let rest) = try model.prepare(input, cache: cache, windowSize: nil)
                    else { fatalError("expected tokens") }
                    return (cache, rest.tokens.size, seen)
                }
                let full = try run(false)
                let pruned = try run(true)
                #expect(full.rows.isEmpty)
                #expect(pruned.rows == expected, "n=\(n): rows \(pruned.rows)")
                #expect(full.rest == 1 && pruned.rest == 1)
                let why = Self.compareCaches(full.cache, pruned.cache, pruneFrom: 5)
                #expect(why.isEmpty, "n=\(n): \(why.joined(separator: "; "))")
                // dSpark tail at the prompt end: the target layers' last 16 rows.
                let P = n - 1
                let holder = { (c: [KVCache]) in
                    c.first { ($0 as? DeepseekV4Cache)?.requiresAuxiliaryTokenHistory == true }
                        as? DeepseekV4Cache
                }
                let t0 = holder(full.cache)?.auxiliaryHiddenTail?.rows(from: P - 16, to: P)
                let t1 = holder(pruned.cache)?.auxiliaryHiddenTail?.rows(from: P - 16, to: P)
                if let t0, let t1 {
                    #expect(t0.shape == t1.shape && Self.cos(t0, t1) >= 0.9999, "n=\(n): tail cos \(Self.cos(t0, t1))")
                } else {
                    Issue.record("n=\(n): dSpark tail missing (full \(t0 != nil), pruned \(t1 != nil))")
                }
                let tail = Array(all[(n - 1) ..< (n + 6)])
                let (worst, flips) = Self.continuation(model, tail, full.cache, pruned.cache)
                #expect(worst >= 0.999 && flips == 0, "n=\(n): logits cos \(worst), flips \(flips)")
                print("[suffix-prune] prepare n=\(n): rows \(pruned.rows) worst logits cos \(worst)")
            }
        }
    }

    // MARK: Off == unchanged

    @Test("pruning off (flag, no prompt end, forward past the prompt end, 0731) leaves the forward unchanged")
    func offIsUnchanged() throws {
        #expect(DeepseekV4Math.suffixPruneFlag(nil))
        #expect(DeepseekV4Math.suffixPruneFlag("1") && DeepseekV4Math.suffixPruneFlag("on"))
        for off in ["0", "false", "OFF", " no "] { #expect(!DeepseekV4Math.suffixPruneFlag(off)) }
        #expect(DeepseekV4Model(DeepseekV41DSparkFocusedTests.tinyConfig(v41: false)).model.suffixPruneFrom == nil)

        try Device.withDefaultDevice(.cpu) {
            let bundle = try Self.writeEngramBundle(layers: [1, 4], vocab: 128)
            defer { try? FileManager.default.removeItem(at: bundle) }
            let model = try Self.model(bundle: bundle)
            let body = Self.tokens(161)
            var fired = 0
            model.model.suffixPruneObserver = { _, _ in fired += 1 }
            defer { model.model.suffixPruneObserver = nil }
            func logits(_ promptEnd: (Int) -> Int?) -> (MLXArray, [KVCache]) {
                let cache = model.newCache(parameters: nil)
                var a = 0
                while a < body.count {
                    let b = min(body.count, a + 40)
                    _ = model.model(
                        MLXArray(Array(body[a ..< b])).reshaped(1, -1), cache: cache, vision: nil,
                        promptEnd: promptEnd(b))
                    DeepseekV4Model.compactSlidingWindows(cache)
                    MLX.eval(cache)
                    a = b
                }
                return (model(MLXArray([Int32(7)]).reshaped(1, 1), cache: cache)[0, 0], cache)
            }
            let (base, baseCache) = logits { _ in nil }
            // A prompt end before the forward's end never prunes (the reference's
            // `prompt_end >= start + N`).
            let (early, earlyCache) = logits { $0 - 1 }
            #expect(fired == 0)
            #expect(DeepseekV41DSparkFocusedTests.equal(base, early))
            for (a, b) in zip(baseCache, earlyCache) {
                #expect(zip(a.state, b.state).allSatisfy { DeepseekV41DSparkFocusedTests.equal($0, $1) })
            }

            // prepare with the flag off: the unpruned chunk forward, no pruned forward.
            let input = LMInput(tokens: MLXArray(body))
            model.suffixPruneEnabled = false
            defer { model.suffixPruneEnabled = DeepseekV4Math.suffixPruneEnabled }
            let cache = model.newCache(parameters: nil)
            guard case .tokens(let rest) = try model.prepare(input, cache: cache, windowSize: 32)
            else { fatalError("expected tokens") }
            #expect(fired == 0 && rest.tokens.size <= 32)
            // Same state as the manual unpruned forward of the same tokens.
            let manual = model.newCache(parameters: nil)
            let consumed = body.count - rest.tokens.size
            _ = model.model(
                MLXArray(Array(body[..<consumed])).reshaped(1, -1), cache: manual, vision: nil)
            DeepseekV4Model.compactSlidingWindows(manual)
            for (a, b) in zip(cache, manual) {
                #expect(a.offset == b.offset)
                #expect(zip(a.state, b.state).allSatisfy { DeepseekV41DSparkFocusedTests.equal($0, $1) })
            }
        }
    }

    // MARK: VL path and sliding-only pruned layers

    @Test("image prefill: pruned sliding-only layers give the full prefill's logits for every chunk plan")
    func visionPrefillPruned() throws {
        try Device.withDefaultDevice(.cpu) {
            typealias V = DeepseekV41VisionFocusedTests
            let bundle = try V.writeEngramBundle(vocab: 128)
            defer { try? FileManager.default.removeItem(at: bundle) }
            let model = try V.tinyModel(vision: true, bundle: bundle)
            // Engram at layer 1 only, no kv sources: layers 2 and 3 (plain rotating caches) prune.
            #expect(model.model.suffixPruneFrom == 2)
            let input = V.imagePrompt()
            var rows: [[Int]] = []
            model.model.suffixPruneObserver = { rows.append($1) }
            defer { model.model.suffixPruneObserver = nil }
            for outer in [4, 5, 6, 9, 13] {
                model.suffixPruneEnabled = false
                let (full, r0) = try V.prefillLogits(model, input, outer: outer)
                #expect(rows.isEmpty)
                model.suffixPruneEnabled = true
                let (pruned, r1) = try V.prefillLogits(model, input, outer: outer)
                model.suffixPruneEnabled = DeepseekV4Math.suffixPruneEnabled
                #expect(r0 == r1)
                let err = zip(full, pruned).map { abs($0 - $1) }.max() ?? .infinity
                #expect(err < 1e-4, "outer \(outer): max |delta| \(err)")
                // Some chunk dropped rows on a pruned layer (or skipped it).
                #expect(rows.contains { r in r[2...].contains { $0 < r[0] } }, "outer \(outer): \(rows)")
                rows.removeAll()
            }
        }
    }
}
