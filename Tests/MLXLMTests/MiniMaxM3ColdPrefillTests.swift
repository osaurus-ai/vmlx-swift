// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// MiniMax-M3 cold prefill (`MiniMaxM3Model.prepare`), the Swift port of vMLX Python's `perf/minimax-m3-prefill`
/// (13-PY-COLD-PREFILL.md): the pieces of one prefill chunk as one layer-major group with the routed experts once per
/// group (71aee382), the K / V / index lanes reserved once per prefill (9d71d160 + 7cbde44c), MLX's buffer cache kept
/// through the prefill (0967ec2f) and no DSpark taps for pieces before the drafter's window (0f3eb8c0).
///
/// Every lever must be bit-identical to the piece-by-piece, stock-growth, per-chunk-release, all-taps prefill it
/// replaces: the logits of the final prefill piece and of every greedy decode step after it, the greedy tokens, and
/// every lane of every layer. Prompts of 2,701 (synthetic) / 1,301 (JANGHT) tokens with the turn cuts of Python's tests
/// (5 / 17 / 18-row pieces, pieces across the 128-row sorted-kernel threshold); steps 320 / 512 (not multiples of 128 /
/// 2048 boundaries in the prompt). Controls: the same prompt without the cuts gives different logits (the comparison
/// sees a change), the grouped routed call really ran (counted) and did not run with the lever off.
@Suite("MiniMax-M3 cold prefill: piece groups, lane reservation, buffer cache, DSpark taps", .serialized)
struct MiniMaxM3ColdPrefillTests {
    typealias A = MiniMaxM3AlignedPrefixCacheTests
    typealias DS = MiniMaxM3DSparkTests

    struct Arm {
        let tokens: [Int]
        let logits: [MLXArray]
        let lanes: [MLXArray]
        let tail: MLXArray?
        let tailStart: Int?
        let pieces: [Range<Int>]
    }

    /// Prefill `prompt` through `prepare` with an aligned plan (store boundary n - 3, `cuts` as turn cuts), then the final
    /// piece and `decode` greedy steps, as an engine runs them.
    static func run(
        _ model: MiniMaxM3Model, _ prompt: [Int], step: Int, cuts: [Int], decode: Int = 4
    ) throws -> Arm {
        let cache = model.newCache(parameters: nil)
        let n = prompt.count
        let plan = AlignedPrefillPlan(
            step: step, boundary: n - 3, promptTokenCount: n, turnCuts: cuts.filter { $0 > 0 && $0 < n - 3 })
        AlignedPrefill.install(plan, into: cache)
        var pieces: [Range<Int>] = []
        AlignedPrefill.hooks.prefillObserver = { pieces = $0.pieces }
        defer { AlignedPrefill.hooks.prefillObserver = nil }
        let result = try model.prepare(
            LMInput(tokens: MLXArray(prompt.map { Int32($0) })), cache: cache, windowSize: step)
        guard case .tokens(let last) = result else {
            Issue.record("prepare returned logits")
            return Arm(tokens: [], logits: [], lanes: [], tail: nil, tailStart: nil, pieces: [])
        }
        AlignedPrefill.install(nil, into: cache)
        var logits = model(last.tokens.reshaped(1, -1), cache: cache)[0, -1].asType(.float32)
        var outs: [MLXArray] = [logits]
        var tokens: [Int] = []
        for _ in 0 ..< decode {
            let token = argMax(logits).item(Int.self)
            tokens.append(token)
            logits = model(MLXArray([Int32(token)]).reshaped(1, 1), cache: cache)[0, -1].asType(.float32)
            outs.append(logits)
        }
        let holder = MiniMaxM3DSpark.tapHolder(cache)
        let tail = holder?.dsparkTail.flatMap { t in t.rows(from: t.startPosition, to: t.endPosition) }
        let lanes = cache.flatMap(\.state)
        eval(outs + lanes + [tail].compactMap { $0 })
        return Arm(
            tokens: tokens, logits: outs, lanes: lanes, tail: tail, tailStart: holder?.dsparkTail?.startPosition,
            pieces: pieces)
    }

    static func equal(_ a: MLXArray, _ b: MLXArray) -> Bool { a.shape == b.shape && arrayEqual(a, b).item(Bool.self) }

    /// Bit-identical arms: tokens, every logit row, every lane, the DSpark tail.
    static func problems(_ a: Arm, _ b: Arm) -> [String] {
        var out: [String] = []
        if a.tokens != b.tokens { out.append("tokens \(a.tokens) vs \(b.tokens)") }
        if a.logits.count != b.logits.count || a.logits.isEmpty { out.append("logit rows \(a.logits.count) vs \(b.logits.count)") }
        for (i, (x, y)) in zip(a.logits, b.logits).enumerated() where !equal(x, y) {
            out.append("logits \(i): max |d| \(abs(x - y).max().item(Float.self))")
        }
        if a.lanes.count != b.lanes.count || a.lanes.isEmpty { out.append("lanes \(a.lanes.count) vs \(b.lanes.count)") }
        for (i, (x, y)) in zip(a.lanes, b.lanes).enumerated() where !equal(x, y) { out.append("lane \(i) \(x.shape)") }
        switch (a.tail, b.tail) {
        case (nil, nil): break
        case (let x?, let y?): if !equal(x, y) || a.tailStart != b.tailStart { out.append("dspark tail") }
        default: out.append("dspark tail present in one arm only")
        }
        return out
    }

    /// Python's cut sets (tests/test_minimax_m3_cold_prefill.py `CUT_SETS`): 5 / 17 / 18-row pieces, a cut on a step
    /// multiple's neighbours, none.
    static let cutSets: [[Int]] = [[5, 17, 18, 40, 100, 333, 700, 961, 1000, 1290, 1937, 2400], [64, 1100, 1700], []]

    final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var _calls: [(Int, Int)] = []
        func add(_ layer: Int, _ pieces: Int) { lock.withLock { _calls.append((layer, pieces)) } }
        var calls: [(Int, Int)] { lock.withLock { _calls } }
    }

    static func withGPU<R>(_ body: () throws -> R) throws -> R {
        try MLXMetalTestLock.withLock { try Device.withDefaultDevice(.gpu) { try body() } }
    }

    // MARK: - Piece groups

    @Test("Grouped pieces == piece by piece on the synthetic M3 (logits, tokens, every lane), three cut sets",
          arguments: [0, 1, 2])
    func groupedEqualsPieceByPiece(cutSet: Int) throws {
        try Self.withGPU {
            let model = try A.model()
            let prompt = A.ids(2701, seed: 404)
            let cuts = Self.cutSets[cutSet]
            let piecewise = try MiniMaxM3Model.$prefillGroupingOverride.withValue(false) {
                try Self.run(model, prompt, step: 320, cuts: cuts)
            }
            let grouped = try MiniMaxM3Model.$prefillGroupingOverride.withValue(true) {
                try Self.run(model, prompt, step: 320, cuts: cuts)
            }
            #expect(piecewise.pieces == grouped.pieces && !grouped.pieces.isEmpty)
            let groups = MiniMaxM3Model.pieceGroups(Array(grouped.pieces.dropLast()), step: 320, grouping: true)
            if !cuts.isEmpty { #expect(groups.contains { $0.count > 1 }, "no multi-piece group: the test is vacuous") }
            let diff = Self.problems(piecewise, grouped)
            #expect(diff.isEmpty, "grouped != piece by piece: \(diff.prefix(4))")
        }
    }

    @Test("Control: the same prompt without the turn cuts gives other logits (the comparison sees geometry changes)")
    func controlWithoutCutsDiffers() throws {
        try Self.withGPU {
            let model = try A.model()
            let prompt = A.ids(2701, seed: 404)
            let cut = try Self.run(model, prompt, step: 320, cuts: Self.cutSets[0])
            let none = try Self.run(model, prompt, step: 320, cuts: [])
            #expect(!Self.problems(cut, none).isEmpty, "cut and uncut prefills agree bit for bit: the oracle is blind")
        }
    }

    @Test("JANGHT M3 (JANGT experts, sorted NAX prefill): one routed call per group, bit-identical to piece by piece")
    func jangtGroupedRoutedIsBitIdentical() async throws {
        guard JANGTPrefillKernel.available else {
            print("[MiniMaxM3ColdPrefillTests] sorted JANGT prefill kernel unavailable (needs NAX): skipped")
            return
        }
        let directory = try MiniMaxM3JANGHTTests.bundleCopy()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try await LLMModelFactory.shared.load(from: directory, using: ByteTokenizerLoader())
        let model = try #require(context.model as? MiniMaxM3Model)
        // >= 128 rows (512 (row, slot) pairs at top-4) take the sorted kernel: [47, 200), [200, 333), [333, 512) and
        // [552, 700), [700, 1024) group; 7 / 23 / 17 / 40-row pieces stay on the decode kernels alone.
        let cuts = [7, 30, 47, 200, 333, 552, 700, 1100, 1290]
        let prompt = A.ids(1301, seed: 9).map { $0 % 250 }
        try Self.withGPU {
            let counter = Counter()
            let piecewise = try MiniMaxM3Model.$prefillGroupingOverride.withValue(false) {
                try MiniMaxM3DecoderLayer.$groupedRoutedObserver.withValue({ counter.add($0, $1) }) {
                    try Self.run(model, prompt, step: 512, cuts: cuts)
                }
            }
            #expect(counter.calls.isEmpty, "piece-by-piece arm made grouped routed calls")
            let grouped = try MiniMaxM3Model.$prefillGroupingOverride.withValue(true) {
                try MiniMaxM3DecoderLayer.$groupedRoutedObserver.withValue({ counter.add($0, $1) }) {
                    try Self.run(model, prompt, step: 512, cuts: cuts)
                }
            }
            // Two MoE layers x two chunks with >= 2 sorted-kernel pieces (3 and 2 pieces).
            let calls = counter.calls
            #expect(calls.count == 4 && calls.map(\.1).sorted() == [2, 2, 3, 3], "grouped routed calls \(calls)")
            let layerMajor = try MiniMaxM3Model.$prefillGroupingOverride.withValue(true) {
                try MiniMaxM3Model.$prefillGroupRoutedOverride.withValue(false) {
                    try Self.run(model, prompt, step: 512, cuts: cuts)
                }
            }
            let diff = Self.problems(piecewise, grouped)
            #expect(diff.isEmpty, "grouped routed != piece by piece: \(diff.prefix(4))")
            let diff2 = Self.problems(piecewise, layerMajor)
            #expect(diff2.isEmpty, "layer-major (one routed call per piece) != piece by piece: \(diff2.prefix(4))")
        }
    }

    @Test("JANGHT M3, VMLX_M3_PREFILL_SORTED: every piece's routed experts on the sorted tiles, one call per group == one call per piece; differs from the stock rule (control)")
    func jangtSortedAllGroupsBitIdentically() async throws {
        guard JANGTPrefillKernel.available else {
            print("[MiniMaxM3ColdPrefillTests] sorted JANGT prefill kernel unavailable (needs NAX): skipped")
            return
        }
        let directory = try MiniMaxM3JANGHTTests.bundleCopy()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try await LLMModelFactory.shared.load(from: directory, using: ByteTokenizerLoader())
        let model = try #require(context.model as? MiniMaxM3Model)
        // 7 / 23 / 17 / 40 / 76-row pieces are below the stock rule's 128 rows: on the decode kernels there, on the
        // sorted tiles here.
        let cuts = [7, 30, 47, 200, 333, 552, 700, 1100, 1290]
        let prompt = A.ids(1301, seed: 9).map { $0 % 250 }
        try Self.withGPU {
            let counter = Counter()
            func arm(grouping: Bool, sorted: Bool) throws -> Arm {
                try MiniMaxM3Model.$prefillSortedAllOverride.withValue(sorted) {
                    try MiniMaxM3Model.$prefillGroupingOverride.withValue(grouping) {
                        try MiniMaxM3DecoderLayer.$groupedRoutedObserver.withValue({ counter.add($0, $1) }) {
                            try Self.run(model, prompt, step: 512, cuts: cuts)
                        }
                    }
                }
            }
            let piecewise = try arm(grouping: false, sorted: true)
            #expect(counter.calls.isEmpty)
            let grouped = try arm(grouping: true, sorted: true)
            // 2 MoE layers x 3 chunks, every piece in its chunk's call: 6, 3 and 3 pieces.
            #expect(counter.calls.map(\.1).sorted() == [3, 3, 3, 3, 6, 6], "grouped routed calls \(counter.calls)")
            let diff = Self.problems(piecewise, grouped)
            #expect(diff.isEmpty, "sorted-all grouped != piece by piece: \(diff.prefix(4))")
            let stock = try arm(grouping: true, sorted: false)
            #expect(!Self.problems(stock, grouped).isEmpty, "the sorted rule changed nothing: the control is blind")
            // vMLX Python's threshold (72 pairs = 18 rows): the 7 / 17-row pieces stay on the decode kernels.
            let py = try MiniMaxM3Model.$prefillSortedMinPairsOverride.withValue(72) {
                (try arm(grouping: false, sorted: true), try arm(grouping: true, sorted: true))
            }
            let diff72 = Self.problems(py.0, py.1)
            #expect(diff72.isEmpty, "sorted-72 grouped != piece by piece: \(diff72.prefix(4))")
            #expect(!Self.problems(py.1, grouped).isEmpty, "the 72-pair threshold changed nothing vs every piece sorted")
        }
    }

    // MARK: - Lane reservation

    @Test("Lanes reserved once per prefill == stock growth (logits, lanes); capacity grows once, to the prompt")
    func reservationIsBitIdentical() throws {
        try Self.withGPU {
            let model = try A.model()
            let prompt = A.ids(2701, seed: 31)
            let stock = try MiniMaxM3SparseCache.$reserveOverride.withValue(false) {
                try Self.run(model, prompt, step: 320, cuts: Self.cutSets[1])
            }
            let reserved = try MiniMaxM3SparseCache.$reserveOverride.withValue(true) {
                try MiniMaxM3SparseCache.$headroomOverride.withValue(1 << 40) {
                    try Self.run(model, prompt, step: 320, cuts: Self.cutSets[1])
                }
            }
            let diff = Self.problems(stock, reserved)
            #expect(diff.isEmpty, "reserved != stock growth: \(diff.prefix(4))")

            // The capacity after one piece + a reservation covers the whole prompt: no growth in the later pieces.
            let cache = model.newCache(parameters: nil)
            #expect(MiniMaxM3SparseCache.reservePrefill(cache, rows: 2701) == nil, "reserved before the lanes exist")
            _ = model.model(MLXArray(prompt[0 ..< 320].map { Int32($0) }).reshaped(1, -1), cache: cache)
            eval(cache)
            let decided = MiniMaxM3SparseCache.$reserveOverride.withValue(true) {
                MiniMaxM3SparseCache.$headroomOverride.withValue(1 << 40) {
                    MiniMaxM3SparseCache.reservePrefill(cache, rows: 2701 - 320)
                }
            }
            #expect(decided == true)
            _ = model.model(MLXArray(prompt[320 ..< 640].map { Int32($0) }).reshaped(1, -1), cache: cache)
            eval(cache)
            for layer in cache.compactMap({ $0 as? MiniMaxM3SparseCache }) {
                let capacities = layer.innerState().map { $0.dim(2) }
                #expect(capacities.allSatisfy { $0 >= 2701 && $0 < 2701 + 256 }, "capacities \(capacities)")
            }
        }
    }

    @Test("Reservation declines without allocator headroom and when disabled; declined == stock growth")
    func reservationDeclines() throws {
        try Self.withGPU {
            let model = try A.model()
            let cache = model.newCache(parameters: nil)
            _ = model.model(MLXArray(A.ids(300, seed: 2).map { Int32($0) }).reshaped(1, -1), cache: cache)
            eval(cache)
            let noRoom = MiniMaxM3SparseCache.$reserveOverride.withValue(true) {
                MiniMaxM3SparseCache.$headroomOverride.withValue(0) {
                    MiniMaxM3SparseCache.reservePrefill(cache, rows: 2000)
                }
            }
            #expect(noRoom == false)
            let disabled = MiniMaxM3SparseCache.$reserveOverride.withValue(false) {
                MiniMaxM3SparseCache.reservePrefill(cache, rows: 2000)
            }
            #expect(disabled == false)
            for layer in cache.compactMap({ $0 as? MiniMaxM3SparseCache }) {
                #expect(layer.kv.reservedRowsUntil == nil, "a declined reservation left a capacity hint")
            }
            let prompt = A.ids(2701, seed: 31)
            let stock = try MiniMaxM3SparseCache.$reserveOverride.withValue(false) {
                try Self.run(model, prompt, step: 320, cuts: Self.cutSets[1])
            }
            let declined = try MiniMaxM3SparseCache.$headroomOverride.withValue(0) {
                try Self.run(model, prompt, step: 320, cuts: Self.cutSets[1])
            }
            let diff = Self.problems(stock, declined)
            #expect(diff.isEmpty, "declined != stock growth: \(diff.prefix(4))")
        }
    }

    // MARK: - Buffer cache

    @Test("Buffer cache kept through the prefill == released after every chunk (logits, lanes)")
    func bufferCacheOnceIsBitIdentical() throws {
        try Self.withGPU {
            let model = try A.model()
            let prompt = A.ids(2701, seed: 77)
            let each = try MiniMaxM3Model.$prefillClearEachChunkOverride.withValue(true) {
                try Self.run(model, prompt, step: 320, cuts: Self.cutSets[0])
            }
            let once = try MiniMaxM3Model.$prefillClearEachChunkOverride.withValue(false) {
                try Self.run(model, prompt, step: 320, cuts: Self.cutSets[0])
            }
            let diff = Self.problems(each, once)
            #expect(diff.isEmpty, "kept buffer cache != per-chunk release: \(diff.prefix(4))")
        }
    }

    @Test("Row-wise ops batched across a group's pieces (VMLX_M3_PREFILL_BATCH_ROWS) == piece by piece, three cut sets",
          arguments: [0, 1, 2])
    func batchedRowsEqualPieceByPiece(cutSet: Int) throws {
        try Self.withGPU {
            let model = try A.model()
            let prompt = A.ids(2701, seed: 404)
            let cuts = Self.cutSets[cutSet]
            let piecewise = try MiniMaxM3Model.$prefillGroupingOverride.withValue(false) {
                try Self.run(model, prompt, step: 320, cuts: cuts)
            }
            let batched = try MiniMaxM3Model.$prefillBatchRowsOverride.withValue(true) {
                try Self.run(model, prompt, step: 320, cuts: cuts)
            }
            let diff = Self.problems(piecewise, batched)
            #expect(diff.isEmpty, "batched rows != piece by piece: \(diff.prefix(4))")
            // The probe admitted some projections (the batching of matmuls really ran) ...
            let admitted = MiniMaxM3RowInvariance.results.filter { $0.value != nil }
            if !cuts.isEmpty { #expect(!admitted.isEmpty, "no projection admitted: \(MiniMaxM3RowInvariance.results)") }
            print("[MiniMaxM3ColdPrefillTests] row invariance: \(MiniMaxM3RowInvariance.results)")
            // ... and admitting every projection from one row unprobed (control) is NOT bit-identical.
            if cutSet == 0 {
                let forced = try MiniMaxM3RowInvariance.$controlForcedThreshold.withValue(1) {
                    try MiniMaxM3Model.$prefillBatchRowsOverride.withValue(true) {
                        try Self.run(model, prompt, step: 320, cuts: cuts)
                    }
                }
                #expect(!Self.problems(piecewise, forced).isEmpty, "unprobed batching agreed: the oracle is blind")
            }
        }
    }

    @Test("JANGHT M3: batched rows + sorted-all routed == piece by piece (logits, lanes)")
    func jangtBatchedRows() async throws {
        guard JANGTPrefillKernel.available else { return }
        let directory = try MiniMaxM3JANGHTTests.bundleCopy()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try await LLMModelFactory.shared.load(from: directory, using: ByteTokenizerLoader())
        let model = try #require(context.model as? MiniMaxM3Model)
        let cuts = [7, 30, 47, 200, 333, 552, 700, 1100, 1290]
        let prompt = A.ids(1301, seed: 9).map { $0 % 250 }
        try Self.withGPU {
            for sorted in [false, true] {
                let piecewise = try MiniMaxM3Model.$prefillSortedAllOverride.withValue(sorted) {
                    try MiniMaxM3Model.$prefillGroupingOverride.withValue(false) {
                        try Self.run(model, prompt, step: 512, cuts: cuts)
                    }
                }
                let batched = try MiniMaxM3Model.$prefillSortedAllOverride.withValue(sorted) {
                    try MiniMaxM3Model.$prefillBatchRowsOverride.withValue(true) {
                        try Self.run(model, prompt, step: 512, cuts: cuts)
                    }
                }
                let diff = Self.problems(piecewise, batched)
                #expect(diff.isEmpty, "sorted \(sorted): batched rows != piece by piece: \(diff.prefix(4))")
            }
        }
    }

    @Test("Prefill groups evaluated asynchronously (VMLX_M3_PREFILL_ASYNC) == synchronously (logits, lanes)")
    func asyncGroupsAreBitIdentical() throws {
        try Self.withGPU {
            let model = try A.model()
            let prompt = A.ids(2701, seed: 78)
            let sync = try MiniMaxM3Model.$prefillAsyncOverride.withValue(false) {
                try Self.run(model, prompt, step: 320, cuts: Self.cutSets[0])
            }
            let async = try MiniMaxM3Model.$prefillAsyncOverride.withValue(true) {
                try Self.run(model, prompt, step: 320, cuts: Self.cutSets[0])
            }
            let diff = Self.problems(sync, async)
            #expect(diff.isEmpty, "async != sync: \(diff.prefix(4))")
        }
    }

    // MARK: - DSpark taps

    @Test("DSpark: no taps before the drafter window == every piece tapped (tail rows, logits, lanes); grouped too")
    func dsparkTapsOnlyInTheWindow() throws {
        let drafter = A.temporaryDirectory("cold-prefill-drafter")
        defer { try? FileManager.default.removeItem(at: drafter) }
        try DS.writeReleaseDrafter(drafter)
        try Self.withGPU {
            let model = try A.model()
            #expect(try model.attachDSparkDraft(directory: drafter))
            let window = model.model.dsparkTailCapacity
            #expect(window == DS.draftWindow)
            let prompt = A.ids(2701, seed: 55)
            var arms: [String: Arm] = [:]
            for (name, allTaps, grouping, batched) in [
                ("all", true, false, false), ("window", false, false, false), ("grouped", false, true, false),
                ("batched", false, true, true),
            ] {
                arms[name] = try MiniMaxM3Model.$prefillAllTapsOverride.withValue(allTaps) {
                    try MiniMaxM3Model.$prefillGroupingOverride.withValue(grouping) {
                        try MiniMaxM3Model.$prefillBatchRowsOverride.withValue(batched) {
                            try Self.run(model, prompt, step: 320, cuts: Self.cutSets[0], decode: 0)
                        }
                    }
                }
            }
            let all = try #require(arms["all"]), inWindow = try #require(arms["window"]), grouped = try #require(arms["grouped"])
            let batched = try #require(arms["batched"])
            // The tail covers at least the last `window` prompt rows (whole recorded pieces: it may start a little
            // earlier), the same rows in every arm (`problems` compares rows and start).
            #expect(all.tail != nil && (all.tailStart ?? .max) <= prompt.count - window,
                    "tail start \(String(describing: all.tailStart))")
            for (name, arm) in [("window", inWindow), ("grouped", grouped), ("batched", batched)] {
                let diff = Self.problems(all, arm)
                #expect(diff.isEmpty, "\(name) != every piece tapped: \(diff.prefix(4))")
            }
        }
    }
}

/// Bytes as tokens (the JANGHT fixture's loader needs a tokenizer; nothing here encodes text).
private struct ByteTokenizerLoader: TokenizerLoader {
    struct ByteTokenizer: Tokenizer {
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map { Int($0) } }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }
    func load(from directory: URL) async throws -> any Tokenizer { ByteTokenizer() }
}
