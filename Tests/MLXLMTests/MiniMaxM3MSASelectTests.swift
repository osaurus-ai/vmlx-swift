// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import CryptoKit
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// MiniMax-M3.1 MSA selection kernels (`MiniMaxM3MSA.kernelBlockScores`) against the score-sheet ops they replace
/// (`referenceBlockScores`) and against the vMLX Python runtime (msa.py at 5fc31aca).
///
/// The kernels must give the reference's block scores BIT FOR BIT, so the indexer's top-k (the same argPartition)
/// returns the same block ids in the same order. Every Swift-side comparison goes through `identity`, which fails
/// closed (an empty or degenerate reference is an error) and which the controls (block ids + 1, local block not forced,
/// mutated kernels, the other kernel arithmetic) must fail.
///
/// Parity with Python: `Fixtures/MiniMaxM3/msa-select.json` from
/// `scripts/minimax_m3_fixtures/generate_msa_select_fixtures.py` (Python 630929ed): SHA-256 digests of the inputs,
/// the reference and kernel block scores and the top-k ids, per TF32 mode; also the SHA-256 of every msa.py Metal
/// source. Inputs are regenerated here from the same MLX PRNG keys (U[-2, 2) on the CPU) and checked first.
///
/// Shapes: M3.1 index geometry (index heads = KV groups 4, plus 8; head dim 128; blocks of 128; top-16; local 1).
/// Lengths 2049 / 2853 / 3011 / 8268 / 32845 are not multiples of the block (128), the K / V step (256) or the chunk
/// (2048); rows 1 (decode), 2 / 5 / 16 (verify-sized), 512 / 1000 / 2048 (chunks).
@Suite("MiniMax-M3.1 MSA selection kernels", .serialized)
struct MiniMaxM3MSASelectTests {
    typealias Base = MiniMaxM3MSABlockSparseTests
    static let indexDim = 128, block = 128, topK = 16
    static let strictFloat32 = ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0"

    // MARK: fixture

    struct Fixture: Decodable {
        let vmlxCommit: String
        let kernelSourcesSha256: [String: String]
        let modes: [String: Mode]
    }

    struct Mode: Decodable {
        let arithmetic: String
        let arithmeticByDtype: [String: String?]
        let matrixUnits: Bool
        let rows: [Case]
        let chunk: [Case]
        let dtypes: [Case]
        let special: [Case]
    }

    struct Case: Decodable {
        let label: String
        let offset: Int
        let path: String
        let iq: String
        let ik: String
        let reference: String
        let kernel: String
        let top: String
        let blocks: Int
        let sk: Int?
        let groups: Int?
        let rows: Int?
        let seed: Int?
        let dtype: String?
        let kind: String?
        let oneRow: [String]?
    }

    static func fixture() throws -> Fixture {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/MiniMaxM3/msa-select.json")
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(Fixture.self, from: Data(contentsOf: url))
    }

    static func mode() throws -> Mode {
        try #require(try fixture().modes[strictFloat32 ? "strict" : "tf32"])
    }

    // MARK: inputs (the generator's PRNG keys)

    static func uniform(_ n: Int, _ shape: [Int]) -> MLXArray {
        MLXRandom.uniform(Float(-2) ..< Float(2), shape, key: MLXRandom.key(UInt64(n)), stream: .cpu)
    }

    /// Index queries `[batch, G, rows, 128]` and keys `[batch, 1, sk, 128]` (a slow drift: neighbouring blocks score
    /// alike, so the top-16 boundary is contested).
    static func index(
        seed: Int, sk: Int, groups: Int, rows: Int, dtype: DType = .bfloat16, batch: Int = 1
    ) -> (iq: MLXArray, ik: MLXArray) {
        let base = seed * 100
        return Device.withDefaultDevice(.cpu) {
            let drift = cumsum(uniform(base + 2, [batch, 1, sk, indexDim]) * Float(0.05), axis: 2)
            let ik = (uniform(base + 1, [batch, 1, sk, indexDim]) + drift).asType(dtype)
            let iq = uniform(base + 3, [batch, groups, rows, indexDim]).asType(dtype)
            eval(ik, iq)
            return (iq, ik)
        }
    }

    static func dtype(_ name: String?) -> DType {
        switch name {
        case "float16": return .float16
        case "float32": return .float32
        default: return .bfloat16
        }
    }

    // MARK: helpers

    /// The indexer's top-k over block scores (the op `select` runs).
    static func top(_ scores: MLXArray) -> MLXArray {
        argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
    }

    static func reference(_ iq: MLXArray, _ ik: MLXArray, offset: Int, local: Int = 1) -> (MLXArray, MLXArray) {
        let s = MiniMaxM3MSA.referenceBlockScores(
            queries: iq, keys: ik, offset: offset, blockSize: block, localBlocks: local)
        let t = top(s)
        eval(s, t)
        return (s, t)
    }

    static func routed(_ iq: MLXArray, _ ik: MLXArray, offset: Int, local: Int = 1) -> (MLXArray, MLXArray) {
        let s = MiniMaxM3MSA.blockScores(queries: iq, keys: ik, offset: offset, blockSize: block, localBlocks: local)
        let t = top(s)
        eval(s, t)
        return (s, t)
    }

    static func kernel(
        _ iq: MLXArray, _ ik: MLXArray, offset: Int, local: Int = 1, arithmetic: MiniMaxM3MSA.SelectPath,
        sources: MiniMaxM3MSA.SelectSources? = nil
    ) -> (MLXArray, MLXArray) {
        let s = MiniMaxM3MSA.kernelBlockScores(
            queries: iq, keys: ik, offset: offset, blockSize: block, localBlocks: local, arithmetic: arithmetic,
            sources: sources)
        let t = top(s)
        eval(s, t)
        return (s, t)
    }

    struct DegenerateReference: Error, CustomStringConvertible {
        let description: String
    }

    /// THE identity check: block scores equal bit for bit (NaN where the reference has NaN) and the top-k equal in
    /// value and order. Fails closed (throws): the reference must be a real selection (rows present, every row's local
    /// block forced to +inf, finite scores present, and the set changes when the block ids shift by one).
    static func identity(
        _ reference: (MLXArray, MLXArray), _ got: (MLXArray, MLXArray), offset: Int
    ) throws -> Bool {
        let (r, rt) = reference
        let (g, gt) = got
        guard r.size > 0, r.ndim == 4, r.shape == g.shape, rt.size > 0, rt.shape == gt.shape else {
            throw DegenerateReference(description: "shapes \(r.shape) / \(g.shape), top \(rt.shape) / \(gt.shape)")
        }
        let (B, G, L, nb) = (r.dim(0), r.dim(1), r.dim(2), r.dim(3))
        guard nb > topK else { throw DegenerateReference(description: "no selection at \(nb) blocks") }
        let local = broadcast(
            floorDivide((MLXArray(Int32(offset)) + arange(0, L)).reshaped(1, 1, L, 1), MLXArray(Int32(block))),
            to: [B, G, L, 1])
        let forced = takeAlong(r, local.asType(.int32), axis: -1)
        guard (forced .== MLXArray(Float.infinity)).all().item(Bool.self) else {
            throw DegenerateReference(description: "reference lost the forced local block")
        }
        guard isFinite(r).any().item(Bool.self) else {
            throw DegenerateReference(description: "no finite reference score")
        }
        let rti = rt.asType(.int32)
        let shifted = sorted((rti + 1) % Int32(nb), axis: -1)
        guard !arrayEqual(sorted(rti, axis: -1), shifted).item(Bool.self) else {
            throw DegenerateReference(description: "degenerate reference selection")
        }
        return arrayEqual(r, g, equalNAN: true).item(Bool.self) && arrayEqual(rti, gt.asType(.int32)).item(Bool.self)
    }

    /// Fraction of (batch, group, row) whose selected block SET differs.
    static func flipRate(_ a: MLXArray, _ b: MLXArray) -> Double {
        let sa = sorted(a.asType(.int32), axis: -1)
        let sb = sorted(b.asType(.int32), axis: -1)
        return (sa .!= sb).any(axis: -1).asType(.float32).mean().item(Double.self)
    }

    static func admitted() throws -> MiniMaxM3MSA.SelectPath {
        let arithmetic = try #require(
            MiniMaxM3MSA.selectArithmetic(.bfloat16, headDim: indexDim),
            "no selection kernel reproduces the reference on this Metal GPU")
        #expect(arithmetic == .nax || arithmetic == .fma)
        return arithmetic
    }

    /// Records the selection router's picks while `body` runs.
    final class SelectSpy: @unchecked Sendable {
        var picks: [MiniMaxM3MSA.SelectPath] = []
    }

    static func spying<R>(_ body: (SelectSpy) throws -> R) rethrows -> R {
        let spy = SelectSpy()
        let previous = MiniMaxM3MSA.hooks.selectObserver
        MiniMaxM3MSA.hooks.selectObserver = { spy.picks.append($0) }
        defer { MiniMaxM3MSA.hooks.selectObserver = previous }
        return try body(spy)
    }

    static func withSelect<R>(_ request: MiniMaxM3MSA.SelectRequest?, _ body: () throws -> R) rethrows -> R {
        let previous = MiniMaxM3MSA.hooks.selectRequest
        MiniMaxM3MSA.hooks.selectRequest = request
        defer { MiniMaxM3MSA.hooks.selectRequest = previous }
        return try body()
    }

    static func digest(_ x: MLXArray) -> String { Base.digest(x) }

    // MARK: - Sources and admission

    @Test("The Metal sources are msa.py's, byte for byte (SHA-256 from the Python commit)")
    func sourcesMatchPython() throws {
        let fixture = try Self.fixture()
        func sha(_ s: String) -> String { SHA256.hash(data: Data(s.utf8)).map { String(format: "%02x", $0) }.joined() }
        let pairs: [(String, String)] = [
            ("_SELECT_NAX_SOURCE", MiniMaxM3MSA.selectNaxSource),
            ("_SELECT_NAX_SPLIT_SOURCE", MiniMaxM3MSA.selectNaxSplitSource),
            ("_SELECT_FMA_SOURCE", MiniMaxM3MSA.selectFmaSource),
            ("_SELECT_SIMD_SOURCE", MiniMaxM3MSA.selectSimdSource),
            ("_NAX_HEADER", MiniMaxM3MSA.matrixUnitHeader),
            ("_NAX_SOURCE", MiniMaxM3MSA.matrixUnitSource),
            ("_PREFILL_SOURCE", MiniMaxM3MSA.prefillSource),
            ("_DECODE_PARTIAL_SOURCE", MiniMaxM3MSA.decodePartialSource),
            ("_DECODE_COMBINE_SOURCE", MiniMaxM3MSA.decodeCombineSource),
        ]
        for (python, swift) in pairs {
            let want = try #require(fixture.kernelSourcesSha256[python], "\(python) missing from the fixture")
            #expect(sha(swift) == want, "\(python) differs from msa.py at \(fixture.vmlxCommit)")
        }
        // Control: one character changed changes the digest the check compares.
        #expect(sha(MiniMaxM3MSA.selectFmaSource + " ") != fixture.kernelSourcesSha256["_SELECT_FMA_SOURCE"])
    }

    @Test("Admission: the kernel family MLX's float32 matmul matches (nax with TF32 on M5, fma with TF32 off)")
    func admittedArithmeticIsTheOneMLXRuns() throws {
        let mode = try Self.mode()
        try Device.withDefaultDevice(.gpu) {
            let arithmetic = try Self.admitted()
            #expect(arithmetic.rawValue == mode.arithmetic, "Python admitted \(mode.arithmetic)")
            if DeepseekV41MPPIndexScores.hasMatrixUnits(architecture: GPU.deviceInfo().architecture) {
                #expect(arithmetic == (Self.strictFloat32 ? .fma : .nax))
            }
            for (name, dtype) in [("bfloat16", DType.bfloat16), ("float16", .float16), ("float32", .float32)] {
                let swift = MiniMaxM3MSA.selectArithmetic(dtype, headDim: Self.indexDim)
                #expect(swift?.rawValue == mode.arithmeticByDtype[name] ?? nil, "\(name)")
            }
        }
    }

    // MARK: - Parity with the Python runtime (and identity with the reference)

    final class Tally {
        var cases = 0, inputs = 0, reference = 0, kernel = 0, top = 0, path = 0, identity = 0
        func report(_ title: String) {
            print(
                "[MSA select parity] \(title) (\(MiniMaxM3MSASelectTests.strictFloat32 ? "strict f32" : "TF32")): "
                    + "inputs \(inputs)/\(cases), reference scores \(reference)/\(cases), kernel scores "
                    + "\(kernel)/\(cases), top-k \(top)/\(cases), path \(path)/\(cases), Swift kernel == reference "
                    + "\(identity)/\(cases)")
        }
        var complete: Bool {
            cases > 0 && [inputs, reference, kernel, top, path, identity].allSatisfy { $0 == cases }
        }
    }

    /// One fixture case: inputs, reference scores, routed (kernel) scores, top-k and path vs Python; kernel ==
    /// reference here. `run` gives the views the routed call reads (default: the inputs themselves).
    static func compare(
        _ c: Case, iq: MLXArray, ik: MLXArray, tally: Tally, run: (MLXArray, MLXArray)? = nil
    ) throws {
        tally.cases += 1
        guard digest(iq) == c.iq, digest(ik) == c.ik else {
            Issue.record("\(c.label): generated inputs differ from Python's")
            return
        }
        tally.inputs += 1
        let ref = reference(iq, ik, offset: c.offset)
        let (got, picks) = spying { spy in
            (routed(run?.0 ?? iq, run?.1 ?? ik, offset: c.offset), spy.picks)
        }
        tally.reference += digest(ref.0) == c.reference ? 1 : 0
        tally.kernel += digest(got.0) == c.kernel ? 1 : 0
        tally.top += digest(got.1) == c.top && digest(ref.1) == c.top ? 1 : 0
        tally.path += picks.map(\.rawValue) == [c.path] ? 1 : 0
        tally.identity += try identity(ref, got, offset: c.offset) ? 1 : 0
        #expect(digest(ref.0) == c.reference, "\(c.label): reference scores differ from Python")
        #expect(digest(got.0) == c.kernel, "\(c.label): kernel scores differ from Python")
        #expect(digest(got.1) == c.top, "\(c.label): top-k differs from Python")
        #expect(picks.map(\.rawValue) == [c.path], "\(c.label): router took \(picks), Python \(c.path)")
    }

    @Test("Decode and verify-sized calls (1 / 2 / 5 / 16 rows, Sk 2049 ... 32845, G 4 / 8) == reference == Python")
    func rowsMatchPython() throws {
        let mode = try Self.mode()
        let tally = Tally()
        try Device.withDefaultDevice(.gpu) {
            for c in mode.rows {
                let (iq, ik) = Self.index(
                    seed: try #require(c.seed), sk: try #require(c.sk), groups: try #require(c.groups),
                    rows: try #require(c.rows))
                try Self.compare(c, iq: iq, ik: ik, tally: tally)
            }
        }
        tally.report("rows")
        #expect(mode.rows.count == 40 && tally.complete)
    }

    @Test("Chunks (512 / 1000 / 2048 rows, Sk 2853 ... 32845, G 4 / 8) == reference == Python")
    func chunksMatchPython() throws {
        let mode = try Self.mode()
        let tally = Tally()
        try Device.withDefaultDevice(.gpu) {
            for c in mode.chunk {
                let (iq, ik) = Self.index(
                    seed: try #require(c.seed), sk: try #require(c.sk), groups: try #require(c.groups),
                    rows: try #require(c.rows))
                try Self.compare(c, iq: iq, ik: ik, tally: tally)
                Memory.clearCache()
            }
        }
        tally.report("chunk")
        #expect(mode.chunk.count == 12 && tally.complete)
    }

    @Test("fp16 / fp32, batch 2, strided views over a NaN tail, per-row verify calls, exact ties, NaN == Python")
    func dtypesAndSpecialCasesMatchPython() throws {
        let mode = try Self.mode()
        let tally = Tally()
        var specials = Set<String>()
        try Device.withDefaultDevice(.gpu) {
            for c in mode.dtypes {
                let (iq, ik) = Self.index(
                    seed: try #require(c.seed), sk: try #require(c.sk), groups: try #require(c.groups),
                    rows: try #require(c.rows), dtype: Self.dtype(c.dtype))
                try Self.compare(c, iq: iq, ik: ik, tally: tally)
            }
            for c in mode.special {
                let kind = try #require(c.kind)
                specials.insert(kind)
                switch kind {
                case "batch2":
                    let (iq, ik) = Self.index(seed: 4001, sk: 3011, groups: 4, rows: 6, batch: 2)
                    try Self.compare(c, iq: iq, ik: ik, tally: tally)
                case "strided":
                    let (sk, rows) = (8268, 9)
                    let (iq, ik) = Self.index(seed: 4002, sk: sk, groups: 4, rows: rows)
                    let iqT = contiguous(iq.transposed(0, 2, 1, 3)).transposed(0, 2, 1, 3)
                    let buffer = concatenated(
                        [ik, MLXArray.full([1, 1, 300, Self.indexDim], values: MLXArray(Float.nan), dtype: ik.dtype)],
                        axis: 2)
                    let view = buffer[0..., 0..., ..<sk]
                    try Self.compare(c, iq: iq, ik: ik, tally: tally, run: (iqT, view))
                    let got = Self.routed(iqT, view, offset: sk - rows)
                    #expect(!isNaN(got.0).any().item(Bool.self), "the NaN tail past Sk was read")
                    // DSpark verify: each row alone, a strided one-row slice over its own prefix.
                    let oneRow = try #require(c.oneRow)
                    #expect(oneRow.count == rows)
                    for r in 0 ..< rows {
                        let end = sk - rows + r + 1
                        let ref = Self.reference(iq[0..., 0..., r ..< (r + 1)], ik[0..., 0..., ..<end], offset: end - 1)
                        let one = Self.routed(iqT[0..., 0..., r ..< (r + 1)], view[0..., 0..., ..<end], offset: end - 1)
                        let identical1 = try Self.identity(ref, one, offset: end - 1)
                        #expect(identical1, "row \(r)")
                        #expect(Self.digest(one.1) == oneRow[r], "row \(r) top-k vs Python")
                    }
                case "ties":
                    let (sk, rows, seed) = (try #require(c.sk), try #require(c.rows), try #require(c.seed))
                    let (iq, ik) = Device.withDefaultDevice(.cpu) { () -> (MLXArray, MLXArray) in
                        let one = Self.uniform(seed * 100 + 1, [Self.block, Self.indexDim])
                        let reps = (sk + Self.block - 1) / Self.block
                        let ik = tiled(one, repetitions: [reps, 1])[..<sk].reshaped(1, 1, sk, Self.indexDim)
                            .asType(.bfloat16)
                        let iq = Self.uniform(seed * 100 + 3, [1, 4, rows, Self.indexDim]).asType(.bfloat16)
                        eval(ik, iq)
                        return (iq, ik)
                    }
                    // The tie at the top-16 boundary is real.
                    let s = Self.reference(iq, ik, offset: sk - rows).0
                    let ordered = sorted(-s, axis: -1)
                    let full = (arange(0, rows) + Int32(sk - rows)) .>= Int32((Self.topK + 1) * Self.block)
                    #expect(full.any().item(Bool.self))
                    let tie = (ordered[.ellipsis, Self.topK - 1] .== ordered[.ellipsis, Self.topK])
                    #expect(MLX.where(full.reshaped(1, 1, rows), tie, true).all().item(Bool.self))
                    try Self.compare(c, iq: iq, ik: ik, tally: tally)
                case "nan":
                    let (sk, rows) = (3011, 5)
                    let (iq, ik) = Self.index(seed: 4200, sk: sk, groups: 4, rows: rows)
                    let pos = arange(0, sk).reshaped(1, 1, sk, 1)
                    let dim = arange(0, Self.indexDim).reshaped(1, 1, 1, Self.indexDim)
                    let poison = ((pos .== 700) .&& (dim .== 3)) .|| ((pos .== Int32(sk - 2)) .&& (dim .== 0))
                    let ikn = MLX.where(poison, MLXArray(Float.nan).asType(ik.dtype), ik)
                    try Self.compare(c, iq: iq, ik: ikn, tally: tally)
                    let r = Self.reference(iq, ikn, offset: sk - rows).0
                    #expect(isNaN(r[0..., 0..., 0..., 5]).all().item(Bool.self), "block 5 must be NaN for every row")
                default:
                    Issue.record("unknown special case \(kind)")
                }
            }
        }
        tally.report("dtypes + special")
        #expect(specials == ["batch2", "strided", "ties", "nan"])
        #expect(mode.dtypes.count == 6 && mode.special.count == 6 && tally.complete)
    }

    // MARK: - Swift invariants

    @Test("Through the indexer: VMLX_MINIMAX_MSA_SELECT=reference vs auto give the same selection")
    func indexerSelectKernelEqualsReferencePath() throws {
        try Device.withDefaultDevice(.gpu) {
            for (sk, rows) in [(2853, 1), (8268, 7), (8268, 600)] {
                let (iq, ik) = Self.index(seed: 5000 + rows, sk: sk, groups: 4, rows: rows)
                let a = try #require(
                    Self.withSelect(.reference) { Base.select(iq, ik, offset: sk - rows) })
                let (b, picks) = try Self.spying { spy in
                    (try #require(Base.select(iq, ik, offset: sk - rows)), spy.picks)
                }
                #expect(a.offset == b.offset && a.keyLength == b.keyLength && b.keyLength == sk)
                #expect(arrayEqual(a.topIndices, b.topIndices).item(Bool.self), "Sk \(sk) rows \(rows)")
                #expect(picks == [try Self.admitted()])
            }
            // The env lever itself (not only the test hook).
            let (iq, ik) = Self.index(seed: 5100, sk: 3011, groups: 4, rows: 3)
            setenv(MiniMaxM3MSA.selectEnvironmentVariable, "reference", 1)
            let viaEnv = Self.spying { spy in
                _ = Base.select(iq, ik, offset: 3008)
                return spy.picks
            }
            unsetenv(MiniMaxM3MSA.selectEnvironmentVariable)
            #expect(viaEnv == [.reference])
            let (iq2, ik2) = Self.index(seed: 5101, sk: 2048, groups: 4, rows: 1)
            #expect(Base.select(iq2, ik2, offset: 2047) == nil)  // every block visible: no selection
        }
    }

    @Test("A row of a multi-row call scores exactly like the one-row call a decode step makes at its position")
    func multiRowSelectionEqualsOneRowSelections() throws {
        try Device.withDefaultDevice(.gpu) {
            let (sk, rows) = (4095 + 7, 7)
            let (iq, ik) = Self.index(seed: 5200, sk: sk, groups: 8, rows: rows)
            let offset = sk - rows
            let multi = Self.routed(iq, ik, offset: offset)
            for r in 0 ..< rows {
                let one = Self.routed(iq[0..., 0..., r ..< (r + 1)], ik[0..., 0..., ..<(offset + r + 1)], offset: offset + r)
                let nb = one.0.dim(-1)
                #expect(arrayEqual(multi.0[0..., 0..., r ..< (r + 1), ..<nb], one.0).item(Bool.self), "row \(r)")
                // keys past the row's position: -inf in the multi-row call, absent in the one-row call
                if nb < multi.0.dim(-1) {
                    #expect((multi.0[0..., 0..., r, nb...] .== MLXArray(-Float.infinity)).all().item(Bool.self))
                }
            }
        }
    }

    // MARK: - Caches

    @Test("Restored / cloned caches at 2853 / 3011 decode bit-identically with kernel and with reference selection")
    func restoredAndClonedCachesSelectLikeTheReference() throws {
        try Device.withDefaultDevice(.gpu) {
            let arithmetic = try Self.admitted()
            let layer = try Base.attention(groups: 4, seed: 1, dtype: .bfloat16)
            for prefix in [2853, 3011] {
                let live = MiniMaxM3SparseCache()
                Base.prefill(layer, live, Base.hidden(prefix, seed: 1))
                let storage = try #require(live.indexStorage)
                #expect(prefix % 128 != 0 && storage.dim(2) > live.offset)
                let steps = Base.hidden(5, seed: 2)
                var outputs: [String: [MLXArray]] = [:]
                for request in [MiniMaxM3MSA.SelectRequest.reference, .auto] {
                    let caches: [(String, KVCache)] = [
                        ("live", live.copy()),
                        ("restored", {
                            let c = MiniMaxM3SparseCache()
                            _ = c.restoreDiskCacheState(
                                live.state.map { $0 * 1 }, metadata: live.metaState, offset: live.offset)
                            return c
                        }()),
                        ("cloned", live.copy()),
                    ]
                    let picks = Self.withSelect(request) {
                        Self.spying { spy in
                            for (name, cache) in caches {
                                outputs["\(request) \(name)"] = Base.decode(layer, cache, steps)
                            }
                            return spy.picks
                        }
                    }
                    #expect(!picks.isEmpty && Set(picks) == [request == .reference ? .reference : arithmetic])
                }
                let base = try #require(outputs["reference live"])
                #expect(outputs.count == 6)
                for (name, seq) in outputs {
                    for r in 0 ..< 5 {
                        #expect(arrayEqual(seq[r], base[r]).item(Bool.self), "\(prefix) \(name) step \(r)")
                    }
                }
            }
        }
    }

    @Test("Partial hit with a NaN-poisoned stale index tail selects exactly like a fresh cache (kernel selection)")
    func partialHitWithPoisonedIndexTail() throws {
        try Device.withDefaultDevice(.gpu) {
            let arithmetic = try Self.admitted()
            let layer = try Base.attention(groups: 4, seed: 3, dtype: .bfloat16)
            let prefix = 2853
            let x = Base.hidden(prefix, seed: 3)
            let hit = MiniMaxM3SparseCache()
            Base.prefill(layer, hit, x)
            eval(layer(Base.hidden(37, seed: 4), mask: .causal, cache: hit))
            #expect(hit.trim(37) == 37 && hit.offset == prefix)
            let index = try #require(hit.indexStorage)
            let keys = try #require(hit.kv.keys)
            let values = try #require(hit.kv.values)
            #expect(index.dim(2) >= prefix + 37)
            index[0..., 0..., prefix..., 0...] = MLXArray(Float.nan).asType(index.dtype)
            keys[0..., 0..., prefix..., 0...] = MLXArray(Float.nan).asType(keys.dtype)
            values[0..., 0..., prefix..., 0...] = MLXArray(Float.nan).asType(values.dtype)
            let fresh = MiniMaxM3SparseCache()
            Base.prefill(layer, fresh, x)
            let b = Base.hidden(41, seed: 5)
            let steps = Base.hidden(4, seed: 6)
            let (results, picks) = Self.spying { spy in
                let oh = layer(b, mask: .causal, cache: hit)
                let of = layer(b, mask: .causal, cache: fresh)
                eval(oh, of)
                return ((oh, of, Base.decode(layer, hit, steps), Base.decode(layer, fresh, steps)), spy.picks)
            }
            let (oh, of, dh, df) = results
            #expect(arrayEqual(oh, of).item(Bool.self) && isFinite(oh).all().item(Bool.self))
            for r in 0 ..< 4 {
                #expect(arrayEqual(dh[r], df[r]).item(Bool.self) && isFinite(dh[r]).all().item(Bool.self), "step \(r)")
            }
            #expect(!picks.isEmpty && Set(picks) == [arithmetic])
        }
    }

    // MARK: - Controls

    @Test("Control: shifted block ids fail the identity check (every set flips)", arguments: [1, 600])
    func controlShiftedBlockIdsFail(rows: Int) throws {
        try Device.withDefaultDevice(.gpu) {
            let sk = 8268
            let (iq, ik) = Self.index(seed: 5300, sk: sk, groups: 4, rows: rows)
            let off = sk - rows
            let ref = Self.reference(iq, ik, offset: sk - rows)
            let got = Self.routed(iq, ik, offset: sk - rows)
            let identical2 = try Self.identity(ref, got, offset: off)
            #expect(identical2)
            let nb = ref.0.dim(-1)
            let shifted = (got.1.asType(.int32) + 1) % Int32(nb)
            let identical3 = try Self.identity(ref, (got.0, shifted), offset: off)
            #expect(!identical3)
            #expect(Self.flipRate(ref.1, shifted) > 0.99)
        }
    }

    @Test("Control: the kernel without the forced local block fails (no +inf anywhere)", arguments: [1, 600])
    func controlLocalBlockNotForcedFails(rows: Int) throws {
        try Device.withDefaultDevice(.gpu) {
            let arithmetic = try Self.admitted()
            let sk = 8268
            let (iq, ik) = Self.index(seed: 5400, sk: sk, groups: 4, rows: rows)
            let off = sk - rows
            let ref = Self.reference(iq, ik, offset: sk - rows)
            let got = Self.kernel(iq, ik, offset: sk - rows, local: 0, arithmetic: arithmetic)
            let identical4 = try Self.identity(ref, got, offset: off)
            #expect(!identical4)
            #expect(!(got.0 .== MLXArray(Float.infinity)).any().item(Bool.self))
        }
    }

    /// (mutation, decode-sized or chunk) per kernel of the admitted family: a lane / tile dropped from the block max,
    /// a biased max, the causal limit one key late.
    static func mutations(_ arithmetic: MiniMaxM3MSA.SelectPath) -> [(MiniMaxM3MSA.SelectSources, String, Bool)] {
        func mutate(_ edit: (inout MiniMaxM3MSA.SelectSources) -> Void) -> MiniMaxM3MSA.SelectSources {
            var s = MiniMaxM3MSA.SelectSources()
            edit(&s)
            return s
        }
        func swap(_ s: String, _ old: String, _ new: String) -> String {
            precondition(s.contains(old), "mutation target missing: \(old)")
            return s.replacingOccurrences(of: old, with: new)
        }
        if arithmetic == .nax {
            return [
                (mutate { $0.naxSplit = swap($0.naxSplit, "mm = max(mm, simd_shuffle_xor(mm, ushort(8)));", "") },
                 "nax_split lane dropped", true),
                (mutate {
                    $0.naxSplit = swap(
                        $0.naxSplit, "for (int tt = 0; tt < NT; ++tt) {", "for (int tt = 0; tt < NT - 1; ++tt) {")
                }, "nax_split tile dropped", true),
                (mutate { $0.nax = swap($0.nax, "mm = max(mm, simd_shuffle_xor(mm, ushort(8)));", "") },
                 "nax lane dropped", false),
                (mutate {
                    $0.nax = swap($0.nax, "if (kp <= qp[u][i] && kp < sk) {", "if (kp <= qp[u][i] + 1 && kp < sk) {")
                }, "nax causal limit one key late", false),
            ]
        }
        return [
            (mutate {
                $0.fma = swap($0.fma, "for (int i = 0; i < NSGB; ++i) {", "for (int i = 0; i < NSGB - 1; ++i) {")
            }, "fma simdgroup dropped", true),
            (mutate {
                $0.fma = swap($0.fma, "const float bm = simd_max(x);", "const float bm = simd_max(x) - 1e-3f;")
            }, "fma biased max", true),
            (mutate {
                $0.simd = swap(
                    $0.simd, "const float m = max(red[tid], red[TP + tid]);", "const float m = red[tid];")
            }, "simd key half dropped", false),
            (mutate { $0.simd = swap($0.simd, "if (kp <= qpos && kp < sk) {", "if (kp <= qpos + 1 && kp < sk) {") },
             "simd causal limit one key late", false),
        ]
    }

    @Test("Control: each kernel of the admitted family, mutated, fails; the unmutated kernel passes", arguments: 0 ..< 4)
    func controlMutatedKernelFails(index: Int) throws {
        try Device.withDefaultDevice(.gpu) {
            let arithmetic = try Self.admitted()
            let (sources, label, decode) = Self.mutations(arithmetic)[index]
            let (sk, rows) = decode ? (8268, 1) : (8268, 600)
            let (iq, ik) = Self.index(seed: 5500 + index, sk: sk, groups: 4, rows: rows)
            let off = sk - rows
            let ref = Self.reference(iq, ik, offset: sk - rows)
            let mutated = Self.kernel(iq, ik, offset: sk - rows, arithmetic: arithmetic, sources: sources)
            let identical5 = try Self.identity(ref, mutated, offset: off)
            #expect(!identical5, "\(label) passed the identity check")
            let good = Self.routed(iq, ik, offset: sk - rows)
            let identical6 = try Self.identity(ref, good, offset: off)
            #expect(identical6, "\(label): the unmutated kernel failed")
        }
    }

    @Test("Control: the non-admitted arithmetic differs from the reference (never admitted)",
          arguments: [(8268, 1), (32845, 16), (8268, 2048)])
    func controlOtherArithmeticFails(sk: Int, rows: Int) throws {
        try Device.withDefaultDevice(.gpu) {
            let arithmetic = try Self.admitted()
            let other: MiniMaxM3MSA.SelectPath = arithmetic == .nax ? .fma : .nax
            if other == .nax && !MiniMaxM3MSA.matrixUnitsAvailable() { return }
            let (iq, ik) = Self.index(seed: 5600, sk: sk, groups: 4, rows: rows)
            let off = sk - rows
            let ref = Self.reference(iq, ik, offset: sk - rows)
            let got = Self.kernel(iq, ik, offset: sk - rows, arithmetic: other)
            let identical7 = try Self.identity(ref, got, offset: off)
            #expect(!identical7)
            let differing = (ref.0 .!= got.0).asType(.float32).mean().item(Float.self)
            print("[MSA select] \(other) at Sk \(sk) rows \(rows): \(differing * 100)% of block scores differ, "
                + "set flips \(Self.flipRate(ref.1, got.1) * 100)%")
        }
    }

    // MARK: - fma family on its own terms

    /// Block scores with one float32 multiply-add per head-dim element in ascending order, on the CPU. For bf16 inputs
    /// every product is exact in float32, so a separate multiply + add rounds exactly like a fused multiply-add.
    static func fmaChainScores(_ iq: MLXArray, _ ik: MLXArray, offset: Int) -> MLXArray {
        Device.withDefaultDevice(.cpu) {
            let (G, L, D) = (iq.dim(1), iq.dim(2), iq.dim(3))
            let sk = ik.dim(2)
            let q2 = iq[0].reshaped(G * L, D).asType(.float32)
            let k2 = ik[0, 0].asType(.float32)
            var s = MLXArray.zeros([G * L, sk], dtype: .float32)
            for d in 0 ..< D {
                s = s + q2[0..., d ..< (d + 1)] * k2[0..., d].reshaped(1, sk)
                if d % 16 == 15 { eval(s) }
            }
            var scores = s.reshaped(G, L, sk)
            let qpos = arange(offset, offset + L).reshaped(1, L, 1)
            scores = MLX.where(arange(0, sk).reshaped(1, 1, sk) .> qpos, MLXArray(-Float.infinity), scores)
            let nb = (sk + block - 1) / block
            scores = concatenated(
                [scores, MLXArray.full([G, L, nb * block - sk], values: MLXArray(-Float.infinity))], axis: -1)
            var out = scores.reshaped(G, L, nb, block).max(axis: -1)
            let local = broadcast(floorDivide(qpos.reshaped(1, L, 1), MLXArray(Int32(block))), to: [G, L, 1])
                .asType(.int32)
            out = putAlong(out, local, values: MLXArray(Float.infinity), axis: -1)
            eval(out)
            return out.reshaped(1, G, L, nb)
        }
    }

    @Test("The fma family (scalar and simdgroup-matrix kernels) == a sequential fma chain, whatever the TF32 mode",
          arguments: [(2853, 1), (3011, 5), (2853, 40), (3011, 300)])
    func fmaKernelsEqualASequentialChain(sk: Int, rows: Int) throws {
        try Device.withDefaultDevice(.gpu) {
            let (iq, ik) = Self.index(seed: 5700, sk: sk, groups: 4, rows: rows)
            let want = Self.fmaChainScores(iq, ik, offset: sk - rows)
            let got = Self.kernel(iq, ik, offset: sk - rows, arithmetic: .fma).0
            #expect(want.shape == got.shape && isFinite(want).any().item(Bool.self))
            #expect(arrayEqual(got, want).item(Bool.self), "Sk \(sk) rows \(rows)")
        }
    }

    // MARK: - Routing and memory

    @Test("Selection routing: admitted kernel when supported, the reference ops otherwise; levers only pick")
    func selectionRouting() throws {
        let (iq, ik) = Self.index(seed: 5800, sk: 3011, groups: 4, rows: 3)
        func pick(_ q: MLXArray, _ k: MLXArray) -> MiniMaxM3MSA.SelectPath {
            MiniMaxM3MSA.chooseSelectPath(queries: q, keys: k, blockSize: Self.block)
        }
        try Device.withDefaultDevice(.gpu) {
            let arithmetic = try Self.admitted()
            #expect(pick(iq, ik) == arithmetic)
            #expect(pick(iq[0..., 0..., ..<1], ik) == arithmetic)  // one row, 4 (group, row) pairs
            let (iq2, ik2) = Self.index(seed: 5801, sk: 3011, groups: 4, rows: 1, batch: 2)
            #expect(pick(iq2, ik2) == .reference)  // MLX: matrix-vector kernel
            let (iq3, ik3) = Self.index(seed: 5802, sk: 3011, groups: 4, rows: 3, batch: 2)
            #expect(pick(iq3, ik3) == arithmetic)
            #expect(pick(iq, ik.asType(.float32)) == .reference)  // mixed dtypes
            #expect(pick(iq[0..., 0..., 0..., ..<120], ik[0..., 0..., 0..., ..<120]) == .reference)  // head dim % 16
            Base.withRequest(.dense) {
                #expect(pick(iq, ik) == .reference)  // the pre-kernel runtime end to end
                Self.withSelect(.kernel) { #expect(pick(iq, ik) == arithmetic) }
            }
            Self.withSelect(.fma) { #expect(pick(iq, ik) == .fma) }
            Self.withSelect(.reference) { #expect(pick(iq, ik) == .reference) }
            Self.withSelect(.nax) {
                #expect(pick(iq, ik) == (MiniMaxM3MSA.matrixUnitsAvailable() ? .nax : .reference))
            }
        }
        Device.withDefaultDevice(.cpu) {
            #expect(pick(iq, ik) == .reference)  // CPU: the reference ops
        }
    }

    @Test("Kernel selection never builds the score sheet (2048 rows x 32845 keys: reference > 1 GiB, kernel < 32 MiB)")
    func kernelSelectionNeverBuildsTheScoreSheet() throws {
        try Device.withDefaultDevice(.gpu) {
            _ = try Self.admitted()
            let (sk, rows) = (32845, 2048)
            let (iq, ik) = Self.index(seed: 5900, sk: sk, groups: 4, rows: rows)
            var peaks: [String: Double] = [:]
            for (name, run) in [
                ("reference", { Self.reference(iq, ik, offset: sk - rows) }),
                ("kernel", { Self.routed(iq, ik, offset: sk - rows) }),
            ] as [(String, () -> (MLXArray, MLXArray))] {
                Memory.clearCache()
                let base = Memory.activeMemory
                GPU.resetPeakMemory()
                _ = run()
                peaks[name] = Double(Memory.peakMemory - base) / Double(1 << 20)
            }
            print("[MSA select] peak above inputs, MiB: \(peaks)")
            #expect(try #require(peaks["reference"]) > 1024)  // the sheet is real (the harness sees it)
            #expect(try #require(peaks["kernel"]) < 32)
        }
    }
}
