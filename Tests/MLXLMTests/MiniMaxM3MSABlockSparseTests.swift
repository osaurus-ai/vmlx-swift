// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import CryptoKit
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// MiniMax-M3.1 block-sparse MSA (`MiniMaxM3MSA.swift`) against the vMLX Python runtime and against itself.
///
/// Parity: `Fixtures/MiniMaxM3/msa-block-sparse.json`, written by
/// `scripts/minimax_m3_fixtures/generate_msa_block_sparse_fixtures.py` from the Python runtime's `msa.py`. Inputs are
/// regenerated here from the same MLX PRNG keys (their SHA-256 is checked first); the fixture holds SHA-256 digests of
/// Python's selection and of every path's output, so "bit for bit" is a digest comparison. Geometry: 64 query heads,
/// 4 or 8 KV groups, head dim 128, index dim 128, blocks of 128, top-16, local 1; Sk 2049 / 2853 / 3011 / 8268.
///
/// float32 matmuls run as TF32 on M5 unless the process starts with MLX_ENABLE_TF32=0; TF32 moves the indexer scores
/// enough to flip some selections (34 of 166 in the fixture). The fixture holds both modes and the test reads the one
/// this process runs in.
///
/// Self-checks (no Python): each path against a float64 CPU reference written from the MSA rule, wrong selections
/// that must fail that check, kernel mutations that must fail it, verify rows == one-token decode steps bit for bit,
/// restored / cloned / partially hit caches == live / fresh caches bit for bit, K / V rows at or past the cache offset
/// never reaching the output, routing.
@Suite("MiniMax-M3.1 block-sparse MSA", .serialized)
struct MiniMaxM3MSABlockSparseTests {
    static let heads = 64, headDim = 128, indexDim = 128, block = 128, topK = 16
    /// Per-row relative error vs float64 (the Python tests' bounds; float32 under TF32 as the fixture generator).
    static let float32Tolerance = 2e-5, bfloat16Tolerance = 6e-3, tf32Tolerance = 6e-3
    static let strictFloat32 = ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0"
    static let scale = pow(Float(headDim), -0.5)

    static func tolerance(_ dtype: DType) -> Double {
        dtype == .bfloat16 ? bfloat16Tolerance : strictFloat32 ? float32Tolerance : tf32Tolerance
    }

    // MARK: fixture

    struct Fixture: Decodable {
        let vmlxCommit: String
        let gpu: String
        let modes: [String: Mode]
    }

    struct Mode: Decodable {
        let matrixUnits: Bool
        let decode: [DecodeCase]
        let verify: [RowsCase]
        let chunk: [RowsCase]
    }

    struct PathResult: Decodable {
        let took: String
        let output: String
        let relMedian: Double
        let relMax: Double
    }

    struct DecodePosition: Decodable {
        let position: Int
        let q: String
        let iq: String
        let selection: String
        let auto: String
        let paths: [String: PathResult]
    }

    struct DecodeCase: Decodable {
        let dtype: String
        let sk: Int
        let groups: Int
        let seed: Int
        let inputs: [String: String]
        let positions: [DecodePosition]
    }

    struct RowsCase: Decodable {
        let dtype: String
        let sk: Int
        let groups: Int
        let seed: Int
        let width: Int?
        let rows: Int?
        let inputs: [String: String]
        let selection: String
        let auto: String
        let paths: [String: PathResult]
    }

    static func mode() throws -> Mode {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/MiniMaxM3/msa-block-sparse.json")
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let fixture = try decoder.decode(Fixture.self, from: Data(contentsOf: url))
        return try #require(fixture.modes[strictFloat32 ? "strict" : "tf32"])
    }

    static func dtype(_ name: String) -> DType { name == "bfloat16" ? .bfloat16 : .float32 }

    // MARK: inputs (the generator's PRNG keys)

    static func key(_ n: Int) -> MLXArray { MLXRandom.key(UInt64(n)) }

    /// U[-2, 2) from PRNG key `n`, on the CPU: integer bits and correctly rounded IEEE ops only. (MLX's normal sampler
    /// runs erfinv, whose last bits differ between MLX builds; the GPU's divide is not correctly rounded.)
    static func uniform(_ n: Int, _ shape: [Int]) -> MLXArray {
        MLXRandom.uniform(Float(-2) ..< Float(2), shape, key: key(n), stream: .cpu)
    }

    /// K / V `[1, G, sk, D]` and index keys `[1, 1, sk, d]` (a slow drift, so neighbouring blocks score alike).
    static func context(seed: Int, sk: Int, groups: Int, dtype: DType) -> (k: MLXArray, v: MLXArray, ik: MLXArray) {
        let base = seed * 100_000
        return Device.withDefaultDevice(.cpu) {
            let k = uniform(base + 1, [1, groups, sk, headDim]).asType(dtype)
            let v = uniform(base + 2, [1, groups, sk, headDim]).asType(dtype)
            let drift = cumsum(uniform(base + 4, [1, 1, sk, indexDim]) * Float(0.05), axis: 2)
            let ik = (uniform(base + 3, [1, 1, sk, indexDim]) + drift).asType(dtype)
            eval(k, v, ik)
            return (k, v, ik)
        }
    }

    /// q `[1, H, rows, D]` and index queries `[1, G, rows, d]` from key stream `n`.
    static func queries(seed: Int, stream n: Int, rows: Int, groups: Int, dtype: DType) -> (q: MLXArray, iq: MLXArray)
    {
        let base = seed * 100_000 + 10 + 2 * n
        return Device.withDefaultDevice(.cpu) {
            let q = (uniform(base, [1, heads, rows, headDim]) * Float(0.5)).asType(dtype)
            let iq = uniform(base + 1, [1, groups, rows, indexDim]).asType(dtype)
            eval(q, iq)
            return (q, iq)
        }
    }

    static func digest(_ x: MLXArray) -> String {
        let c = contiguous(x)
        eval(c)
        return SHA256.hash(data: c.asData(access: .copy).data).map { String(format: "%02x", $0) }.joined()
    }

    static func select(_ iq: MLXArray, _ ik: MLXArray, offset: Int) -> MiniMaxM3MSASelection? {
        MiniMaxM3MSA.select(queries: iq, keys: ik, offset: offset, blockSize: block, topK: topK, localBlocks: 1)
    }

    // MARK: module

    static func configuration(groups: Int, hidden: Int = 256) throws -> MiniMaxM3Configuration {
        let json: [String: Any] = [
            "model_type": "minimax_m3", "hidden_size": hidden, "num_hidden_layers": 1,
            "num_attention_heads": heads, "num_key_value_heads": groups, "head_dim": headDim, "rotary_dim": 64,
            "vocab_size": 16, "moe_layer_freq": [0],
            "sparse_attention_config": [
                "sparse_num_index_heads": groups, "sparse_index_dim": indexDim, "sparse_block_size": block,
                "sparse_topk_blocks": topK, "sparse_local_block": 1, "sparse_attention_freq": [1],
            ],
        ]
        return try JSONDecoder().decode(
            MiniMaxM3Configuration.self, from: JSONSerialization.data(withJSONObject: json))
    }

    /// One sparse attention layer with seeded random weights, every parameter in `dtype`.
    static func attention(groups: Int, seed: UInt64 = 0, dtype: DType = .float32) throws -> MiniMaxM3Attention {
        MLXRandom.seed(seed)
        let layer = MiniMaxM3Attention(try configuration(groups: groups), layer: 0)
        let cast = layer.parameters().flattened().map { ($0.0, $0.1.asType(dtype)) }
        try layer.update(parameters: ModuleParameters.unflattened(cast), verify: .all)
        eval(layer)
        return layer
    }

    /// Runs `body` with the MSA request forced (nil = the environment / auto).
    static func withRequest<R>(_ request: MiniMaxM3MSA.Request?, _ body: () throws -> R) rethrows -> R {
        let previous = MiniMaxM3MSA.hooks.request
        MiniMaxM3MSA.hooks.request = request
        defer { MiniMaxM3MSA.hooks.request = previous }
        return try body()
    }

    /// Records the router's picks while `body` runs.
    final class PathSpy: @unchecked Sendable {
        var picks: [MiniMaxM3MSA.Path] = []
    }

    static func spying<R>(_ body: (PathSpy) throws -> R) rethrows -> R {
        let spy = PathSpy()
        let previous = MiniMaxM3MSA.hooks.pathObserver
        MiniMaxM3MSA.hooks.pathObserver = { spy.picks.append($0) }
        defer { MiniMaxM3MSA.hooks.pathObserver = previous }
        return try body(spy)
    }

    /// (output, the path the router took) for a fixture path name; `kernel_sg` is the simdgroup chunk kernel forced.
    static func run(
        _ layer: MiniMaxM3Attention, _ path: String, _ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
        _ selection: MiniMaxM3MSASelection, mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) throws -> (MLXArray, String) {
        if path == "kernel_sg" {
            let out = MiniMaxM3MSA.kernelAttention(
                queries: q, keys: k, values: v, selection: selection, scale: layer.scale, matrixUnits: false)
            eval(out)
            return (out, path)
        }
        let request = try #require(MiniMaxM3MSA.Request(rawValue: path))
        return withRequest(request) {
            let took = MiniMaxM3MSA.choosePath(selection, queries: q, keys: k, mask: mask, kvHeads: layer.kvHeads)
            let out = layer.attend(q, k, v, selection: selection, mask: mask)
            eval(out)
            return (out, took.rawValue)
        }
    }

    // MARK: float64 reference

    /// MSA attention `[n, H * D]` in float64 on the CPU from the rule alone, for the query rows `rows` (all when nil)
    /// of `q` `[1, H, R, D]`: key j is allowed for row r of group g when j's block is one of the row's block ids and
    /// `j <= offset + r`.
    static func reference(
        _ q: MLXArray, _ k: MLXArray, _ v: MLXArray, top: MLXArray, offset: Int, rows subset: [Int]? = nil
    ) -> MLXArray {
        let (G, sk) = (k.dim(1), k.dim(2))
        let hg = heads / G
        let rows = subset ?? Array(0 ..< q.dim(2))
        return Device.withDefaultDevice(.cpu) {
            let index = MLXArray(rows.map { Int32($0) })
            let Q = take(q[0], index, axis: 1).asType(.float32).asType(.float64)
                .reshaped(G, hg, rows.count, headDim)
            let K = expandedDimensions(k[0].asType(.float32).asType(.float64), axis: 1)  // [G, 1, sk, D]
            let V = expandedDimensions(v[0].asType(.float32).asType(.float64), axis: 1)
            let keyPositions = MLXArray(Int32(0) ..< Int32(sk))
            let keyBlocks = floorDivide(keyPositions, MLXArray(Int32(block))).reshaped(1, 1, 1, sk)
            let ids = take(top[0].asType(.int32), index, axis: 1)  // [G, n, keep]
            let queryPositions = MLXArray(rows.map { Int32(offset + $0) })
            let s = MLXArray(float64: 1.0 / Double(headDim).squareRoot())
            let negInf = MLXArray(-Float.infinity).asType(.float64)
            let step = max(1, (1 << 28) / (heads * sk * 8))
            var outs: [MLXArray] = []
            for r0 in stride(from: 0, to: rows.count, by: step) {
                let r1 = min(rows.count, r0 + step)
                let selected = (expandedDimensions(ids[0..., r0 ..< r1, 0...], axis: -1) .== keyBlocks)
                    .any(axis: 2)  // [G, n, sk]
                let causal = keyPositions.reshaped(1, 1, sk) .<= queryPositions[r0 ..< r1].reshaped(1, r1 - r0, 1)
                let allowed = selected .&& causal
                let scores = matmul(Q[0..., 0..., r0 ..< r1, 0...], K.transposed(0, 1, 3, 2)) * s
                let p = softmax(MLX.where(expandedDimensions(allowed, axis: 1), scores, negInf), axis: -1)
                let o = matmul(p, V)  // [G, hg, n, D]
                outs.append(o.transposed(2, 0, 1, 3).reshaped(r1 - r0, heads * headDim))
                eval(outs.last!)
            }
            return concatenated(outs, axis: 0)
        }
    }

    /// The rows a long chunk's float64 check covers: all of a short one; every 4th row plus the last 8 of a long one
    /// (bit identity with Python covers every row; Python's float64 errors over all rows are in the fixture).
    static func referenceRows(_ n: Int) -> [Int]? {
        n <= 128 ? nil : Array(Set(stride(from: 0, to: n, by: 4)).union(n - 8 ..< n)).sorted()
    }

    /// Per-row relative L2 error of `out` `[1, H, rows, D]` against a reference `[rows, H * D]`.
    static func rowErrors(_ out: MLXArray, _ reference: MLXArray, rows subset: [Int]? = nil) -> [Double] {
        Device.withDefaultDevice(.cpu) {
            var a = out[0].asType(.float32).asType(.float64).transposed(1, 0, 2).reshaped(out.dim(2), -1)
            if let subset { a = take(a, MLXArray(subset.map { Int32($0) }), axis: 0) }
            let d = a - reference
            let errors = sqrt((d * d).sum(axis: -1)) / sqrt((reference * reference).sum(axis: -1))
            return errors.asArray(Double.self)
        }
    }

    /// THE equivalence check: every row finite and within `tolerance` of the reference.
    static func equivalent(_ out: MLXArray, _ reference: MLXArray, _ tolerance: Double) -> Bool {
        let errors = rowErrors(out, reference)
        return !errors.isEmpty && errors.allSatisfy { $0.isFinite && $0 <= tolerance }
    }

    // MARK: parity bookkeeping

    final class Tally {
        struct Row {
            var count = 0, identical = 0
            var swiftMax = 0.0, pythonMax = 0.0
        }
        var rows: [String: Row] = [:]

        func add(_ key: String, identical: Bool, swiftErrors: [Double], python: PathResult) {
            var row = rows[key, default: Row()]
            row.count += 1
            row.identical += identical ? 1 : 0
            row.swiftMax = max(row.swiftMax, swiftErrors.max() ?? .infinity)
            row.pythonMax = max(row.pythonMax, python.relMax)
            rows[key] = row
        }

        func report(_ title: String) {
            for (key, row) in rows.sorted(by: { $0.key < $1.key }) {
                print(
                    "[MSA parity] \(title) \(key): bit-identical \(row.identical)/\(row.count), max rel vs f64: "
                        + String(format: "swift %.2e python %.2e", row.swiftMax, row.pythonMax))
            }
        }
    }

    /// Compares one call (selection + every path) with Python's digests. Returns false when the inputs differ (the
    /// rest of the case is then meaningless).
    static func compare(
        _ layer: MiniMaxM3Attention, label: String, dtype: DType, q: MLXArray, k: MLXArray, v: MLXArray,
        iq: MLXArray, ik: MLXArray, offset: Int, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        selectionDigest: String, auto: String, paths: [String: PathResult], tally: Tally
    ) throws {
        let selection = try #require(select(iq, ik, offset: offset), "\(label): no selection")
        #expect(digest(selection.topIndices) == selectionDigest, "\(label): selection differs from Python")
        let took = withRequest(.auto) {
            MiniMaxM3MSA.choosePath(selection, queries: q, keys: k, mask: mask, kvHeads: layer.kvHeads)
        }
        #expect(took.rawValue == auto, "\(label): auto routing \(took) vs Python \(auto)")
        let checked = referenceRows(q.dim(2))
        let ref = reference(q, k, v, top: selection.topIndices, offset: offset, rows: checked)
        for (path, python) in paths.sorted(by: { $0.key < $1.key }) {
            let (out, tookPath) = try run(layer, path, q, k, v, selection, mask: mask)
            #expect(tookPath == python.took, "\(label) \(path): took \(tookPath), Python \(python.took)")
            let identical = digest(out) == python.output
            #expect(identical, "\(label) \(path): output differs from Python")
            let errors = rowErrors(out, ref, rows: checked)
            #expect(
                errors.allSatisfy { $0.isFinite && $0 <= tolerance(dtype) },
                "\(label) \(path): max rel vs float64 \(errors.max() ?? .nan)")
            tally.add("\(dtype) \(path)", identical: identical, swiftErrors: errors, python: python)
        }
    }

    // MARK: - Parity with the Python runtime

    @Test("Decode rows: selection and every path match Python bit for bit")
    func decodeRowsMatchPython() throws {
        let mode = try Self.mode()
        let tally = Tally()
        var positions = 0
        try Device.withDefaultDevice(.gpu) {
            for c in mode.decode {
                let dtype = Self.dtype(c.dtype)
                let layer = try Self.attention(groups: c.groups)
                let (k, v, ik) = Self.context(seed: c.seed, sk: c.sk, groups: c.groups, dtype: dtype)
                let label = "decode \(c.dtype) G\(c.groups) Sk\(c.sk)"
                guard
                    Self.digest(k) == c.inputs["k"], Self.digest(v) == c.inputs["v"], Self.digest(ik) == c.inputs["ik"]
                else {
                    Issue.record("\(label): generated K / V / index keys differ from Python's")
                    continue
                }
                for p in c.positions {
                    let (q, iq) = Self.queries(
                        seed: c.seed, stream: p.position, rows: 1, groups: c.groups, dtype: dtype)
                    guard Self.digest(q) == p.q, Self.digest(iq) == p.iq else {
                        Issue.record("\(label) p\(p.position): generated queries differ from Python's")
                        continue
                    }
                    let end = p.position + 1
                    try Self.compare(
                        layer, label: "\(label) p\(p.position)", dtype: dtype, q: q, k: k[0..., 0..., ..<end],
                        v: v[0..., 0..., ..<end], iq: iq, ik: ik[0..., 0..., ..<end], offset: p.position, mask: .none,
                        selectionDigest: p.selection, auto: p.auto, paths: p.paths, tally: tally)
                    positions += 1
                }
            }
        }
        tally.report("decode (\(Self.strictFloat32 ? "strict f32" : "TF32"))")
        // Fail closed: every fixture position ran, each with its three paths.
        let expected = mode.decode.map(\.positions.count).reduce(0, +)
        let compared = tally.rows.values.map(\.count).reduce(0, +)
        #expect(positions == expected && positions >= 100)
        #expect(compared == 3 * positions)
    }

    @Test("Verify rows (2-16 in one call): selection and every path match Python bit for bit")
    func verifyRowsMatchPython() throws {
        let mode = try Self.mode()
        let tally = Tally()
        try Self.rowsCases(mode.verify, kind: "verify", mask: .none, tally: tally)
        tally.report("verify (\(Self.strictFloat32 ? "strict f32" : "TF32"))")
        let compared = tally.rows.values.map(\.count).reduce(0, +)
        #expect(mode.verify.count == 40 && compared == 3 * 40)
    }

    @Test("Prefill chunks (101 / 512 rows): selection and every path match Python bit for bit")
    func chunksMatchPython() throws {
        let mode = try Self.mode()
        let tally = Tally()
        try Self.rowsCases(mode.chunk, kind: "chunk", mask: .causal, tally: tally)
        tally.report("chunk (\(Self.strictFloat32 ? "strict f32" : "TF32"))")
        let compared = tally.rows.values.map(\.count).reduce(0, +)
        #expect(mode.chunk.count == 14 && compared == 7 * 2 + 7 * 3)
        // The bf16 chunk kernel ran on the matrix units in Python; it must here too.
        #expect(mode.matrixUnits == MiniMaxM3MSA.matrixUnitsAvailable())
    }

    static func rowsCases(
        _ cases: [RowsCase], kind: String, mask: MLXFast.ScaledDotProductAttentionMaskMode, tally: Tally
    ) throws {
        try Device.withDefaultDevice(.gpu) {
            for c in cases {
                let dtype = Self.dtype(c.dtype)
                let rows = try #require(c.width ?? c.rows)
                let layer = try Self.attention(groups: c.groups)
                let (k, v, ik) = Self.context(seed: c.seed, sk: c.sk, groups: c.groups, dtype: dtype)
                let (q, iq) = Self.queries(
                    seed: c.seed, stream: kind == "verify" ? rows : 0, rows: rows, groups: c.groups, dtype: dtype)
                let label = "\(kind) \(c.dtype) G\(c.groups) Sk\(c.sk) rows \(rows)"
                let generated = ["k": k, "v": v, "ik": ik, "q": q, "iq": iq].mapValues { Self.digest($0) }
                guard generated == c.inputs else {
                    Issue.record("\(label): generated inputs differ from Python's")
                    continue
                }
                try Self.compare(
                    layer, label: label, dtype: dtype, q: q, k: k, v: v, iq: iq, ik: ik, offset: c.sk - rows,
                    mask: mask, selectionDigest: c.selection, auto: c.auto, paths: c.paths, tally: tally)
            }
        }
    }

    // MARK: - Controls

    enum Control: String, CaseIterable {
        case shared, offByOne, noLocal
    }

    /// A wrong selection: every group uses group 0's blocks; every block id + 1 (mod blocks); the query's own block
    /// swapped for the first unselected block.
    static func control(_ s: MiniMaxM3MSASelection, _ kind: Control) -> MiniMaxM3MSASelection {
        let shape = s.topIndices.shape
        let (G, rows, keep) = (shape[1], shape[2], shape[3])
        let blocks = Int32(s.blockCount)
        var top = s.topIndices.asType(.int32).asArray(Int32.self)
        func index(_ g: Int, _ r: Int, _ i: Int) -> Int { (g * rows + r) * keep + i }
        for g in 0 ..< G {
            for r in 0 ..< rows {
                switch kind {
                case .shared:
                    for i in 0 ..< keep { top[index(g, r, i)] = top[index(0, r, i)] }
                case .offByOne:
                    for i in 0 ..< keep { top[index(g, r, i)] = (top[index(g, r, i)] + 1) % blocks }
                case .noLocal:
                    let row = Set((0 ..< keep).map { top[index(g, r, $0)] })
                    let local = Int32((s.offset + r) / s.blockSize)
                    let spare = (0 ..< blocks).first { !row.contains($0) }!
                    for i in 0 ..< keep where top[index(g, r, i)] == local { top[index(g, r, i)] = spare }
                }
            }
        }
        return MiniMaxM3MSASelection(
            topIndices: MLXArray(top.map { UInt32($0) }, shape), offset: s.offset, keyLength: s.keyLength,
            blockSize: s.blockSize)
    }

    @Test("Controls: shared / shifted / local-dropped selections fail the check the real one passes, every path")
    func wrongSelectionsFail() throws {
        try Device.withDefaultDevice(.gpu) {
            // Decode rows (fp32, Sk 3011, G 4): every position must fail, on the decode kernel and the gather path.
            let layer = try Self.attention(groups: 4)
            let seed = 900_001
            let (k, v, ik) = Self.context(seed: seed, sk: 3011, groups: 4, dtype: .float32)
            var checked = 0
            for p in [2048, 2300, 2689, 3008, 3009, 3010] {
                let (q, iq) = Self.queries(seed: seed, stream: p, rows: 1, groups: 4, dtype: .float32)
                let (kp, vp) = (k[0..., 0..., ..<(p + 1)], v[0..., 0..., ..<(p + 1)])
                let selection = try #require(Self.select(iq, ik[0..., 0..., ..<(p + 1)], offset: p))
                let ref = Self.reference(q, kp, vp, top: selection.topIndices, offset: p)
                for path in ["decode", "gather"] {
                    let (good, took) = try Self.run(layer, path, q, kp, vp, selection, mask: .none)
                    #expect(took == path)
                    #expect(Self.equivalent(good, ref, Self.float32Tolerance), "p\(p) \(path) real selection")
                    for kind in Control.allCases {
                        let (bad, _) = try Self.run(layer, path, q, kp, vp, Self.control(selection, kind), mask: .none)
                        #expect(!Self.equivalent(bad, ref, Self.float32Tolerance), "p\(p) \(path) control \(kind)")
                        checked += 1
                    }
                }
            }
            #expect(checked == 6 * 2 * 3)

            // Chunks: fp32 simdgroup kernel (Sk 2853, 300 rows) and bf16 matrix-unit kernel (Sk 8268, 160 rows).
            for (sk, rows, dtype) in [(2853, 300, DType.float32), (8268, 160, .bfloat16)] {
                let seed = 900_002 + sk
                let (k, v, ik) = Self.context(seed: seed, sk: sk, groups: 4, dtype: dtype)
                let (q, iq) = Self.queries(seed: seed, stream: 0, rows: rows, groups: 4, dtype: dtype)
                let selection = try #require(Self.select(iq, ik, offset: sk - rows))
                let ref = Self.reference(q, k, v, top: selection.topIndices, offset: sk - rows)
                let matrixUnits = dtype == .bfloat16
                if matrixUnits && !MiniMaxM3MSA.matrixUnitsAvailable() { continue }
                func kernel(_ s: MiniMaxM3MSASelection) -> MLXArray {
                    let out = MiniMaxM3MSA.kernelAttention(
                        queries: q, keys: k, values: v, selection: s, scale: Self.scale, matrixUnits: matrixUnits)
                    eval(out)
                    return out
                }
                let tolerance = dtype == .bfloat16 ? Self.bfloat16Tolerance : Self.float32Tolerance
                #expect(Self.equivalent(kernel(selection), ref, tolerance), "chunk \(dtype) real selection")
                for kind in Control.allCases {
                    #expect(
                        !Self.equivalent(kernel(Self.control(selection, kind)), ref, tolerance),
                        "chunk \(dtype) control \(kind)")
                }
            }
        }
    }

    // MARK: - Kernel mutations

    /// Decisive-key check for the chunk kernels, independent of precision: one key gets a huge score and a distinct
    /// value. The output must follow it when the key is allowed (the row's own key, a partial last block) and ignore it
    /// when it is masked (one past the row inside its own block, an unselected block). Returns the failures.
    static func spikeFailures(dtype: DType, launch: (MLXArray, MLXArray, MLXArray, MiniMaxM3MSASelection) -> MLXArray)
        -> [String]
    {
        let (G, sk, rows) = (4, 2853, 4)
        var failures: [String] = []
        for (n, (spike, row, allowed)) in [(2852, 3, true), (2850, 1, true), (2851, 1, false), (5, 3, false)]
            .enumerated()
        {
            let keys = MLXRandom.split(key: MLXRandom.key(UInt64(700 + n)), into: 3)
            let q = MLXRandom.normal([1, heads, rows, headDim], key: keys[0]) * Float(0.1)
            q[.ellipsis, 0] = MLXArray(Float(1))
            let k = MLXRandom.normal([1, G, sk, headDim], key: keys[1]) * Float(0.1)
            k[0..., 0..., spike, 0...] = MLXArray(Float(0))
            k[0..., 0..., spike, 0] = MLXArray(Float(2000))  // score ~177 after scaling: decisive
            let v = MLXRandom.normal([1, G, sk, headDim], key: keys[2])
            v[0..., 0..., spike, 0...] = MLXArray(Float(5))
            // Local block first, then blocks 2 ... 16: blocks 0 and 1 unselected.
            var top = [UInt32]()
            for _ in 0 ..< G {
                for r in 0 ..< rows {
                    top.append(UInt32((sk - rows + r) / block))
                    top += (2 ... 16).map { UInt32($0) }
                }
            }
            let selection = MiniMaxM3MSASelection(
                topIndices: MLXArray(top, [1, G, rows, topK]), offset: sk - rows, keyLength: sk, blockSize: block)
            let out = launch(q.asType(dtype), k.asType(dtype), v.asType(dtype), selection)
                .asType(.float32)[0, 0..., row]
            let ok =
                allowed
                ? abs(out - 5).max().item(Float.self) < 0.05 : abs(out).max().item(Float.self) < 1
            if !ok { failures.append("spike \(spike) row \(row) allowed \(allowed)") }
        }
        return failures
    }

    @Test("The chunk kernels follow / ignore a decisive key exactly at the causal and selection boundaries")
    func chunkKernelsSpikeOnBoundaryKeys() throws {
        Device.withDefaultDevice(.gpu) {
            #expect(
                Self.spikeFailures(dtype: .float32) { q, k, v, s in
                    MiniMaxM3MSA.kernelAttention(
                        queries: q, keys: k, values: v, selection: s, scale: Self.scale, matrixUnits: false)
                }.isEmpty)
            if MiniMaxM3MSA.matrixUnitsAvailable() {
                #expect(
                    Self.spikeFailures(dtype: .bfloat16) { q, k, v, s in
                        MiniMaxM3MSA.kernelAttention(
                            queries: q, keys: k, values: v, selection: s, scale: Self.scale, matrixUnits: true)
                    }.isEmpty)
            }
        }
    }

    @Test("Kernel mutations: the causal mask one key off in any kernel fails a check")
    func kernelMutationsFail() throws {
        try Device.withDefaultDevice(.gpu) {
            // Decode kernel, fp32: the row's own key dropped. Every decode position must fail the reference check.
            let mutated = MiniMaxM3MSA.decodePartialSource.replacingOccurrences(
                of: "const bool valid = kp <= q_pos;", with: "const bool valid = kp < q_pos;")
            #expect(mutated != MiniMaxM3MSA.decodePartialSource)
            let kernels = MiniMaxM3MSA.decodeKernels(
                element: "float", hg: 16, d: Self.headDim, keep: Self.topK, block: Self.block, partialSource: mutated)
            let seed = 900_101
            let (k, v, ik) = Self.context(seed: seed, sk: 3011, groups: 4, dtype: .float32)
            for p in [2048, 2700, 3010] {
                let (q, iq) = Self.queries(seed: seed, stream: p, rows: 1, groups: 4, dtype: .float32)
                let (kp, vp) = (k[0..., 0..., ..<(p + 1)], v[0..., 0..., ..<(p + 1)])
                let selection = try #require(Self.select(iq, ik[0..., 0..., ..<(p + 1)], offset: p))
                let ref = Self.reference(q, kp, vp, top: selection.topIndices, offset: p)
                let real = MiniMaxM3MSA.decodeKernelAttention(
                    queries: q, keys: kp, values: vp, selection: selection, scale: Self.scale)
                let bad = MiniMaxM3MSA.decodeKernelAttention(
                    queries: q, keys: kp, values: vp, selection: selection, scale: Self.scale, kernels: kernels)
                #expect(Self.equivalent(real, ref, Self.float32Tolerance), "decode p\(p) real kernel")
                #expect(!Self.equivalent(bad, ref, Self.float32Tolerance), "decode p\(p) mutated kernel passed")
            }

            // Chunk kernels: own key dropped / one future key admitted, each must fail the spike check.
            let simdgroup = MiniMaxM3MSA.prefillSource
            let matrix = MiniMaxM3MSA.matrixUnitSource
            let mutations: [(String, String, String)] = [
                ("simdgroup own key dropped", "s = kp <= q_pos ? s : NEG;", "s = kp < q_pos ? s : NEG;"),
                ("simdgroup future key", "s = kp <= q_pos ? s : NEG;", "s = kp <= q_pos + 1 ? s : NEG;"),
                (
                    "matrix own key dropped", "const float x = kp <= q_pos ? s[e] * qscale : NEG;",
                    "const float x = kp < q_pos ? s[e] * qscale : NEG;"
                ),
                (
                    "matrix future key", "const float x = kp <= q_pos ? s[e] * qscale : NEG;",
                    "const float x = kp <= q_pos + 1 ? s[e] * qscale : NEG;"
                ),
            ]
            var ran = 0
            for (name, from, to) in mutations {
                let isMatrix = name.hasPrefix("matrix")
                if isMatrix && !MiniMaxM3MSA.matrixUnitsAvailable() { continue }
                let source = (isMatrix ? matrix : simdgroup).replacingOccurrences(of: from, with: to)
                #expect(source != (isMatrix ? matrix : simdgroup), "\(name): mutation did not apply")
                let dtype: DType = isMatrix ? .bfloat16 : .float32
                let failures = Self.spikeFailures(dtype: dtype) { q, k, v, s in
                    let element = MiniMaxM3MSA.elementType(dtype)!
                    if isMatrix {
                        let kernel = MiniMaxM3MSA.matrixUnitKernel(
                            element: element, hg: 16, d: Self.headDim, keep: Self.topK, block: Self.block, source: source)
                        return MiniMaxM3MSA.matrixUnitLaunch(
                            queries: q, keys: k, values: v, selection: s, scale: Self.scale, kernel: kernel)
                    }
                    let kernel = MiniMaxM3MSA.prefillKernel(
                        element: element, hg: 16, d: Self.headDim, keep: Self.topK, block: Self.block, source: source)
                    return MiniMaxM3MSA.simdgroupLaunch(
                        queries: q, keys: k, values: v, selection: s, scale: Self.scale, kernel: kernel)
                }
                #expect(!failures.isEmpty, "\(name): mutated kernel passed the spike check")
                print("[MSA mutation] \(name): caught by \(failures)")
                ran += 1
            }
            #expect(ran >= 2)
        }
    }

    // MARK: - DSpark invariant: verify rows == one-token decode

    @Test("One decode-kernel launch of several rows == one launch per row, bit for bit")
    func multiRowDecodeLaunchEqualsSingleRows() throws {
        try Device.withDefaultDevice(.gpu) {
            for (dtype, groups) in [(DType.bfloat16, 4), (.float32, 8)] {
                let (sk, rows) = (3011, 9)
                let seed = 900_201 + groups
                let (k, v, ik) = Self.context(seed: seed, sk: sk, groups: groups, dtype: dtype)
                let (q, iq) = Self.queries(seed: seed, stream: 0, rows: rows, groups: groups, dtype: dtype)
                let selection = try #require(Self.select(iq, ik, offset: sk - rows))
                let many = MiniMaxM3MSA.decodeKernelAttention(
                    queries: q, keys: k, values: v, selection: selection, scale: Self.scale)
                for r in 0 ..< rows {
                    let end = sk - rows + r + 1
                    let one = MiniMaxM3MSASelection(
                        topIndices: contiguous(selection.topIndices[0..., 0..., r ..< (r + 1)]), offset: end - 1,
                        keyLength: end, blockSize: Self.block)
                    let single = MiniMaxM3MSA.decodeKernelAttention(
                        queries: q[0..., 0..., r ..< (r + 1)], keys: k[0..., 0..., ..<end],
                        values: v[0..., 0..., ..<end], selection: one, scale: Self.scale)
                    #expect(
                        arrayEqual(many[0..., 0..., r ..< (r + 1)], single).item(Bool.self),
                        "\(dtype) G\(groups) row \(r)")
                }
            }
        }
    }

    static func hidden(_ n: Int, seed: UInt64, dtype: DType = .bfloat16) -> MLXArray {
        MLXRandom.normal([1, n, 256], key: MLXRandom.key(seed)).asType(dtype)
    }

    static func prefill(_ layer: MiniMaxM3Attention, _ cache: MiniMaxM3SparseCache, _ x: MLXArray, step: Int = 2048) {
        for start in stride(from: 0, to: x.dim(1), by: step) {
            eval(layer(x[0..., start ..< min(start + step, x.dim(1))], mask: .causal, cache: cache))
        }
    }

    static func decode(_ layer: MiniMaxM3Attention, _ cache: KVCache, _ xs: MLXArray) -> [MLXArray] {
        (0 ..< xs.dim(1)).map { r in
            let out = layer(xs[0..., r ..< (r + 1)], mask: .none, cache: cache)
            eval(out)
            return out
        }
    }

    /// DSpark-style verify of `x`'s rows: one-row projections (a decode step's arithmetic per row), every row appended
    /// to the cache at once, then for each row its own selection over its own prefix and one attend call.
    static func verify(_ layer: MiniMaxM3Attention, _ x: MLXArray, cache: MiniMaxM3SparseCache, offset: Int)
        -> MLXArray
    {
        let (B, W) = (x.dim(0), x.dim(1))
        func rows(_ linear: Linear, _ t: MLXArray) -> MLXArray {
            concatenated((0 ..< W).map { linear(t[0..., $0 ..< ($0 + 1)]) }, axis: 1)
        }
        let indexer = layer.indexer!
        let q = layer.rope(
            layer.qNorm(rows(layer.wq, x).reshaped(B, W, layer.heads, layer.headDim)).transposed(0, 2, 1, 3),
            offset: offset)
        let k = layer.rope(
            layer.kNorm(rows(layer.wk, x).reshaped(B, W, layer.kvHeads, layer.headDim)).transposed(0, 2, 1, 3),
            offset: offset)
        let v = rows(layer.wv, x).reshaped(B, W, layer.kvHeads, layer.headDim).transposed(0, 2, 1, 3)
        let (keys, values) = cache.update(keys: k, values: v)
        let iq = indexer.rope(
            indexer.queryNorm(rows(indexer.queryProjection, x).reshaped(B, W, indexer.heads, indexer.headDim))
                .transposed(0, 2, 1, 3), offset: offset)
        let ik = indexer.rope(
            indexer.keyNorm(rows(indexer.keyProjection, x).reshaped(B, W, 1, indexer.headDim))
                .transposed(0, 2, 1, 3), offset: offset)
        let indexKeys = cache.updateIndexKeys(ik)
        let outs = (0 ..< W).map { r in
            let end = offset + r + 1
            let selection = indexer.select(
                queries: iq[0..., 0..., r ..< (r + 1)], keys: indexKeys[0..., 0..., ..<end], offset: offset + r)
            return layer.attend(
                contiguous(q[0..., 0..., r ..< (r + 1)]), keys[0..., 0..., ..<end], values[0..., 0..., ..<end],
                selection: selection, mask: .none)
        }
        let o = concatenated(outs, axis: 2)
        return rows(layer.wo, o.transposed(0, 2, 1, 3).reshaped(B, W, -1))
    }

    @Test("Verify rows == one-token decode steps at the same positions, bit for bit, on the decode kernel")
    func verifyRowsEqualDecodeSteps() throws {
        try Device.withDefaultDevice(.gpu) {
            try Self.withRequest(.auto) {
                for groups in [4, 8] {
                    let layer = try Self.attention(groups: groups, seed: UInt64(groups), dtype: .bfloat16)
                    for (prefix, width) in [(2853, 7), (3069, 6), (4095, 4), (2047, 5)] {
                        let x = Self.hidden(prefix, seed: 7)
                        let rows = Self.hidden(width, seed: 8)
                        let (dec, ver) = (MiniMaxM3SparseCache(), MiniMaxM3SparseCache())
                        Self.prefill(layer, dec, x)
                        Self.prefill(layer, ver, x)
                        let (verified, verifyPicks) = Self.spying { spy in
                            let out = Self.verify(layer, rows, cache: ver, offset: prefix)
                            eval(out)
                            return (out, spy.picks)
                        }
                        let (decoded, decodePicks) = Self.spying { spy in (Self.decode(layer, dec, rows), spy.picks) }
                        let label = "G\(groups) prefix \(prefix) width \(width)"
                        #expect(verifyPicks == decodePicks, "\(label): \(verifyPicks) vs \(decodePicks)")
                        if prefix + 1 > Self.topK * Self.block {
                            #expect(!decodePicks.isEmpty && Set(decodePicks) == [.decode], "\(label)")
                        } else {
                            #expect(decodePicks.contains(.decode), "\(label): never crossed into the sparse regime")
                        }
                        for r in 0 ..< width {
                            #expect(
                                arrayEqual(verified[0..., r ..< (r + 1)], decoded[r]).item(Bool.self),
                                "\(label) row \(r)")
                        }
                        #expect(ver.offset == prefix + width && dec.offset == prefix + width)
                    }
                }
            }
        }
    }

    // MARK: - Caches and the logical extent

    @Test("A restored or cloned cache (offsets 2853, 3011) decodes exactly like the live, step-padded cache")
    func restoredCacheDecodesLikeLive() throws {
        try Device.withDefaultDevice(.gpu) {
            try Self.withRequest(.auto) {
                let layer = try Self.attention(groups: 4, seed: 1, dtype: .bfloat16)
                for prefix in [2853, 3011] {
                    let live = MiniMaxM3SparseCache()
                    Self.prefill(layer, live, Self.hidden(prefix, seed: 1))
                    #expect(prefix % 128 != 0 && prefix % 256 != 0)
                    #expect(try #require(live.kv.keys).dim(2) > live.offset && live.offset == prefix)
                    let restored = MiniMaxM3SparseCache()
                    #expect(
                        restored.restoreDiskCacheState(
                            live.state.map { $0 * 1 }, metadata: live.metaState, offset: live.offset))
                    #expect(try #require(restored.kv.keys).dim(2) == prefix)
                    let cloned = try #require(live.copy() as? MiniMaxM3SparseCache)
                    let steps = Self.hidden(5, seed: 2)
                    let (a, picks) = Self.spying { spy in (Self.decode(layer, live, steps), spy.picks) }
                    #expect(!picks.isEmpty && Set(picks) == [.decode], "the sparse decode path must run")
                    let b = Self.decode(layer, restored, steps)
                    let c = Self.decode(layer, cloned, steps)
                    for r in 0 ..< 5 {
                        #expect(arrayEqual(a[r], b[r]).item(Bool.self), "restored \(prefix) step \(r)")
                        #expect(arrayEqual(a[r], c[r]).item(Bool.self), "cloned \(prefix) step \(r)")
                    }
                }
            }
        }
    }

    @Test("Partial hit: prefix + suffix A, trimmed, stale tail NaN-poisoned, then suffix B == a cache that never saw A")
    func partialHitIgnoresTheStaleTail() throws {
        try Device.withDefaultDevice(.gpu) {
            let layer = try Self.attention(groups: 4, seed: 3, dtype: .bfloat16)
            let (prefix, aLength, bLength) = (2853, 37, 41)
            let x = Self.hidden(prefix, seed: 3)
            let b = Self.hidden(bLength, seed: 5)
            let steps = Self.hidden(4, seed: 6)
            for request in [MiniMaxM3MSA.Request.sparse, .auto] {
                try Self.withRequest(request) {
                    let hit = MiniMaxM3SparseCache()
                    Self.prefill(layer, hit, x)
                    eval(layer(Self.hidden(aLength, seed: 4), mask: .causal, cache: hit))
                    #expect(hit.trim(aLength) == aLength && hit.offset == prefix)
                    let keys = try #require(hit.kv.keys)
                    let values = try #require(hit.kv.values)
                    #expect(keys.dim(2) > prefix + aLength - 1)
                    keys[0..., 0..., prefix..., 0...] = MLXArray(Float.nan).asType(keys.dtype)
                    values[0..., 0..., prefix..., 0...] = MLXArray(Float.nan).asType(values.dtype)
                    let fresh = MiniMaxM3SparseCache()
                    Self.prefill(layer, fresh, x)
                    let (outputs, picks) = Self.spying { spy in
                        let oh = layer(b, mask: .causal, cache: hit)
                        let of = layer(b, mask: .causal, cache: fresh)
                        eval(oh, of)
                        return ((oh, of, Self.decode(layer, hit, steps), Self.decode(layer, fresh, steps)), spy.picks)
                    }
                    let (oh, of, dh, df) = outputs
                    #expect(arrayEqual(oh, of).item(Bool.self) && isFinite(oh).all().item(Bool.self), "\(request)")
                    for r in 0 ..< 4 {
                        #expect(
                            arrayEqual(dh[r], df[r]).item(Bool.self) && isFinite(dh[r]).all().item(Bool.self),
                            "\(request) step \(r)")
                    }
                    #expect(picks.contains(.decode))
                    if request == .sparse { #expect(picks.contains(.kernel)) }
                }
            }
        }
    }

    @Test("K / V rows at or past the logical extent (NaN-poisoned) never reach any sparse path's output")
    func keysPastTheExtentAreNeverRead() throws {
        try Device.withDefaultDevice(.gpu) {
            let layer = try Self.attention(groups: 4)
            let (sk, capacity, rows) = (3011, 3328, 40)
            let seed = 900_301
            let (k, v, ik) = Self.context(seed: seed, sk: sk, groups: 4, dtype: .bfloat16)
            let (q, iq) = Self.queries(seed: seed, stream: 0, rows: rows, groups: 4, dtype: .bfloat16)
            let tail = MLXArray.full([1, 4, capacity - sk, Self.headDim], values: MLXArray(Float.nan), dtype: .bfloat16)
            let kb = concatenated([k, tail], axis: 2)
            let vb = concatenated([v, tail], axis: 2)
            for L in [1, rows] {
                let qq = q[0..., 0..., (rows - L)...]
                let selection = try #require(
                    Self.select(iq[0..., 0..., (rows - L)...], ik, offset: sk - L))
                let mask: MLXFast.ScaledDotProductAttentionMaskMode = L > 1 ? .causal : .none
                for path in ["decode", "gather", "kernel"] {
                    let (clean, took) = try Self.run(layer, path, qq, k, v, selection, mask: mask)
                    #expect(took == path)
                    let (padded, _) = try Self.run(
                        layer, path, qq, kb[0..., 0..., ..<sk], vb[0..., 0..., ..<sk], selection, mask: mask)
                    let (raw, _) = try Self.run(layer, path, qq, kb, vb, selection, mask: mask)
                    #expect(isFinite(raw).all().item(Bool.self), "\(path) L\(L): raw buffer output not finite")
                    #expect(arrayEqual(raw, clean).item(Bool.self), "\(path) L\(L): raw")
                    #expect(arrayEqual(padded, clean).item(Bool.self), "\(path) L\(L): padded view")
                }
            }
        }
    }

    @Test("The gather path's source rows stay below the extent; allowed == selected and causal")
    func gatherIndicesStayBelowTheExtent() throws {
        let (sk, rows) = (3011, 70)
        let seed = 900_401
        let (_, _, ik) = Self.context(seed: seed, sk: sk, groups: 4, dtype: .float32)
        let (_, iq) = Self.queries(seed: seed, stream: 0, rows: rows, groups: 4, dtype: .float32)
        let selection = try #require(Self.select(iq, ik, offset: sk - rows))
        let (source, allowed) = MiniMaxM3MSA.keyPositions(selection)
        #expect(source.max().item(Int32.self) == Int32(sk - 1) && source.min().item(Int32.self) >= 0)
        let blocks = sorted(selection.topIndices.asType(.int32), axis: -1)
        let positions = (expandedDimensions(blocks, axis: -1) * MLXArray(Int32(Self.block))
            + MLXArray(Int32(0) ..< Int32(Self.block)))
            .reshaped(source.shape)
        let queryPositions = MLXArray(Int32(sk - rows) ..< Int32(sk)).reshaped(1, 1, rows, 1)
        #expect(arrayEqual(allowed, positions .<= queryPositions).item(Bool.self))
        let zero = MLXArray(Int32(0))
        #expect(arrayEqual(MLX.where(allowed, source, zero), MLX.where(allowed, positions, zero)).item(Bool.self))
        // The partial last block is exercised, and none of its out-of-extent keys is allowed.
        let extent = MLXArray(Int32(sk))
        #expect((positions .>= extent).any().item(Bool.self))
        #expect(!(allowed .&& (positions .>= extent)).any().item(Bool.self))
    }

    // MARK: - Selection

    @Test("The dense path's mask is exactly 'block selected and key <= query position'; <= 2048 keys select nothing")
    func denseBiasAndTheVisibleBoundary() throws {
        let (sk, rows) = (2853, 40)
        let seed = 900_501
        let (_, _, ik) = Self.context(seed: seed, sk: sk, groups: 4, dtype: .float32)
        let (_, iq) = Self.queries(seed: seed, stream: 0, rows: rows, groups: 4, dtype: .float32)
        let selection = try #require(Self.select(iq, ik, offset: sk - rows))
        let bias = selection.denseBias()[0]  // [G, rows, sk]
        let top = selection.topIndices[0].asType(.int32)
        let keyBlocks = floorDivide(MLXArray(Int32(0) ..< Int32(sk)), MLXArray(Int32(Self.block))).reshaped(1, 1, 1, sk)
        let selected = (expandedDimensions(top, axis: -1) .== keyBlocks).any(axis: 2)
        let causal = MLXArray(Int32(0) ..< Int32(sk)).reshaped(1, 1, sk)
            .<= MLXArray(Int32(sk - rows) ..< Int32(sk)).reshaped(1, rows, 1)
        #expect(arrayEqual(bias .== 0, selected .&& causal).item(Bool.self))
        #expect(MLX.where(bias .== 0, Float(-Float.infinity), bias).max().item(Float.self) == -Float.infinity)
        // Every group keeps its own blocks, and the local block is always in.
        #expect((selected[0] .!= selected[1]).any().item(Bool.self))
        let local = floorDivide(MLXArray(Int32(sk - rows) ..< Int32(sk)), MLXArray(Int32(Self.block))).reshaped(1, rows, 1)
        #expect((top .== local).any(axis: -1).all().item(Bool.self))

        // keep >= blocks (<= 2048 keys): no selection; one key more selects 16 of 17 blocks.
        let (_, _, ik2) = Self.context(seed: seed + 1, sk: 2049, groups: 4, dtype: .float32)
        let (_, iq2) = Self.queries(seed: seed + 1, stream: 0, rows: 1, groups: 4, dtype: .float32)
        for keys in [2047, 2048] {
            #expect(Self.select(iq2, ik2[0..., 0..., ..<keys], offset: keys - 1) == nil)
        }
        let s = try #require(Self.select(iq2, ik2, offset: 2048))
        #expect(s.blockCount == 17 && s.keep == 16)
    }

    // MARK: - Routing

    @Test("Routing picks by rows, keys, dtype and mask; it never refuses")
    func routing() throws {
        let layer = try Self.attention(groups: 4)
        var contexts: [String: (MLXArray, MLXArray)] = [:]
        func pick(
            rows: Int, sk: Int, dtype: DType, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
            request: MiniMaxM3MSA.Request = .auto
        ) -> MiniMaxM3MSA.Path? {
            let tag = "\(sk)-\(dtype)"
            if contexts[tag] == nil {
                let (k, _, ik) = Self.context(seed: 900_601, sk: sk, groups: 4, dtype: dtype)
                contexts[tag] = (k, ik)
            }
            let (k, ik) = contexts[tag]!
            let (q, iq) = Self.queries(seed: 900_601, stream: rows, rows: rows, groups: 4, dtype: dtype)
            guard let selection = Self.select(iq, ik, offset: sk - rows) else { return nil }
            return Self.withRequest(request) {
                MiniMaxM3MSA.choosePath(selection, queries: q, keys: k, mask: mask, kvHeads: layer.kvHeads)
            }
        }
        Device.withDefaultDevice(.gpu) {
            for dtype in [DType.bfloat16, .float32] {
                #expect(pick(rows: 1, sk: 2049, dtype: dtype) == .decode)
                #expect(pick(rows: 1, sk: 8268, dtype: dtype) == .decode)
                #expect(pick(rows: 8, sk: 2853, dtype: dtype) == .decode)
                #expect(pick(rows: 9, sk: 2853, dtype: dtype) == .dense)  // 9-16 rows below 4096 keys
                #expect(pick(rows: 16, sk: 4096, dtype: dtype) == .decode)
                #expect(pick(rows: 9, sk: 2853, dtype: dtype, request: .sparse) == .decode)
                #expect(
                    pick(rows: 1, sk: 8268, dtype: dtype, mask: .array(MLXArray.zeros([1, 1, 1, 8268])))
                        == .dense)
                #expect(pick(rows: 1, sk: 8268, dtype: dtype, request: .dense) == .dense)
                #expect(pick(rows: 1, sk: 8268, dtype: dtype, request: .gather) == .gather)
                #expect(pick(rows: 600, sk: 8268, dtype: dtype, request: .decode) == .decode)
                #expect(pick(rows: 1, sk: 8268, dtype: dtype, request: .kernel) == .kernel)
            }
            let units: MiniMaxM3MSA.Path = MiniMaxM3MSA.matrixUnitsAvailable() ? .kernel : .dense
            #expect(pick(rows: 600, sk: 8268, dtype: .bfloat16, mask: .causal) == units)
            #expect(pick(rows: 600, sk: 5120, dtype: .bfloat16, mask: .causal) == units)
            #expect(pick(rows: 600, sk: 5119, dtype: .bfloat16, mask: .causal) == .dense)
            #expect(pick(rows: 600, sk: 8268, dtype: .float32, mask: .causal) == .dense)  // fp32 chunks dense
            #expect(pick(rows: 600, sk: 3011, dtype: .bfloat16, mask: .causal) == .dense)
            #expect(pick(rows: 600, sk: 3011, dtype: .bfloat16, mask: .causal, request: .sparse) == .kernel)
        }
        // No Metal default device: the dense path (the kernels need the GPU).
        Device.withDefaultDevice(.cpu) {
            #expect(pick(rows: 1, sk: 8268, dtype: .float32) == .dense)
        }
        // Indexer heads that do not line up with the KV groups: dense.
        let (k, _, ik) = Self.context(seed: 900_602, sk: 3011, groups: 4, dtype: .bfloat16)
        let (q, iq) = Self.queries(seed: 900_602, stream: 0, rows: 1, groups: 2, dtype: .bfloat16)
        let selection = try #require(Self.select(iq, ik, offset: 3010))
        #expect(
            MiniMaxM3MSA.choosePath(selection, queries: q, keys: k, mask: .none, kvHeads: 4) == .dense)
    }

    @Test("Matrix units: admitted on M5-class GPUs (the self-check must pass there, not silently fall back)")
    func matrixUnitsAdmittedOnM5() throws {
        let architecture = GPU.deviceInfo().architecture
        let error = MiniMaxM3MSA.matrixUnitSelfCheck()
        print("[MSA] \(architecture): matrix-unit self-check error \(error)")
        if DeepseekV41MPPIndexScores.hasMatrixUnits(architecture: architecture) {
            #expect(MiniMaxM3MSA.matrixUnitsAvailable(), "\(architecture)")
            #expect(error < 2e-2)
        }
    }
}
