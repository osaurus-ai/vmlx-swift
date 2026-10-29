// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

/// vMLX's MiniMax-M3 fused attention half (K1-K5, `vmlx_engine/models/minimax_m3/fused_attn.py` at
/// `MiniMaxM3FusedKernelSources.vmlxCommit`), ported host-for-host onto the identical Metal sources: one decoder layer's
/// `x + o_proj(MSA(input_layernorm(x)))` for one sequence of 1...16 rows in five launches.
///
/// - K1 `attn_qkv`: input GemmaRMSNorm (fp32 `1 + w`) fused into one projection launch for q | k | v (affine) and
///   index q | k (MXFP8 or affine), fp32 out.
/// - K2 `attn_heads`: head norms + partial RoPE + casts; the new K / V / index-key rows.
/// - K3 `attn_scores`: indexer block scores of every (row, KV group), only when the last row is past the dense
///   threshold (top-k x block keys).
/// - K4 `attn_msa`: per (row, group) top-k selection + MSA attention + merge.
/// - K5 `attn_oproj`: o_proj + residual add, one rounding.
///
/// Row invariance: every row's arithmetic is independent of the other rows and of the row count, so a decode step and
/// the same row in a DSpark verify are bit-identical (the property the seam requires). NOT bit-identical to the stock
/// path (fp32 throughout, fp32 `1 + w`, one rounding per output).
///
/// Host differences from Python, on purpose: K2 always runs with INPLACE = 0 here: the new rows come back as outputs and
/// are appended with the cache's own MLX slice updates (`MiniMaxM3SparseCache.update` / `updateIndexKeys`, copy on
/// write), so no aliasing or encode-ordering hazard exists (Python's in-place lane writes need its lane-ownership
/// tracking and `mx.depends`; that is a speed lever left for later). A verify wider than 16 rows runs in consecutive
/// chunks of at most 16 (each chunk sees the previous chunks' rows in the cache), as Python does.
final class MiniMaxM3FusedAttention: @unchecked Sendable {
    static let shared = MiniMaxM3FusedAttention()

    static let maxRows = 16
    // K1 / K5 threadgroup shapes (fused_attn.py): one row (NSG 2, RPS 4, RT 1), several rows (4, 8, 2).
    static func projShape(_ rows: Int) -> (nsg: Int, rps: Int, rt: Int) { rows == 1 ? (2, 4, 1) : (4, 8, 2) }

    /// Diagnostics (fused_attn.STATS): calls, rows, calls that ran K3.
    final class Stats: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [String: Int] = [:]
        func add(_ key: String, _ n: Int = 1) { lock.withLock { values[key, default: 0] += n } }
        func reset() { lock.withLock { values = [:] } }
        var snapshot: [String: Int] { lock.withLock { values } }
    }
    static let stats = Stats()

    // MARK: plan

    /// Static facts of one layer's attention the kernels are built for (Python `Plan`).
    struct Plan {
        var hidden = 0, H = 0, KVH = 0, GI = 0, D = 0, ROT = 0
        var log2base = 0.0
        var epsIn: Float = 0, epsHead: Float = 0
        var block = 0, keep = 0, local = 0
        var bitsA = 0, gsA = 0, fmtB = 0, bitsB = 0, gsB = 0, bitsO = 0, gsO = 0
        var scale: Float = 0
        var HG = 0, HS = 0
    }

    private let lock = NSLock()
    /// Plans per layer object; the weak reference guards against a recycled address of a freed layer.
    private final class WeakLayer { weak var layer: MiniMaxM3DecoderLayer?; init(_ l: MiniMaxM3DecoderLayer) { layer = l } }
    private var plans: [ObjectIdentifier: (WeakLayer, Result<Plan, PlanError>)] = [:]

    struct PlanError: Error, CustomStringConvertible { let description: String }

    private static func affine(_ m: Linear) -> (bits: Int, gs: Int)? {
        guard let q = m as? QuantizedLinear, q.mode == .affine, q.bias == nil, let b = q.biases,
            [4, 8].contains(q.bits), [32, 64, 128].contains(q.groupSize), q.scales.dtype == b.dtype
        else { return nil }
        return (q.bits, q.groupSize)
    }

    private static func mxfp8(_ m: Linear) -> Bool {
        guard let q = m as? QuantizedLinear else { return false }
        return q.mode == .mxfp8 && q.bias == nil && q.bits == 8 && q.groupSize == 32
    }

    /// (output rows, input dims) of a quantized linear.
    private static func outIn(_ m: Linear) -> (Int, Int) {
        let q = m as! QuantizedLinear
        return (q.weight.dim(0), q.weight.dim(1) * 32 / q.bits)
    }

    static func buildPlan(_ layer: MiniMaxM3DecoderLayer) -> Result<Plan, PlanError> {
        let attn = layer.selfAttn
        guard let ix = attn.indexer else { return .failure(PlanError(description: "no sparse attention (indexer)")) }
        var p = Plan()
        p.H = attn.heads
        p.KVH = attn.kvHeads
        p.D = attn.headDim
        p.hidden = layer.inputLayerNorm.weight.dim(0)
        p.epsIn = layer.inputLayerNorm.eps
        p.GI = ix.heads
        p.block = ix.blockSize
        p.keep = ix.topKBlocks
        p.local = ix.localBlocks
        p.scale = attn.scale
        let qkv = [attn.wq, attn.wk, attn.wv].map(affine)
        guard let a = qkv[0], qkv.allSatisfy({ $0 != nil && $0! == a }) else {
            return .failure(PlanError(description: "q / k / v not one affine (4 / 8-bit) format"))
        }
        (p.bitsA, p.gsA) = a
        if mxfp8(ix.queryProjection) && mxfp8(ix.keyProjection) {
            (p.fmtB, p.bitsB, p.gsB) = (1, 8, 32)
        } else if let b = affine(ix.queryProjection), let b2 = affine(ix.keyProjection), b == b2 {
            (p.fmtB, p.bitsB, p.gsB) = (2, b.bits, b.gs)
        } else {
            return .failure(PlanError(description: "index q / k neither MXFP8 nor one affine format"))
        }
        guard let o = affine(attn.wo) else { return .failure(PlanError(description: "o_proj not affine 4 / 8-bit")) }
        (p.bitsO, p.gsO) = o
        let shapes: [(String, Linear, (Int, Int))] = [
            ("q", attn.wq, (p.H * p.D, p.hidden)), ("k", attn.wk, (p.KVH * p.D, p.hidden)),
            ("v", attn.wv, (p.KVH * p.D, p.hidden)), ("iq", ix.queryProjection, (p.GI * p.D, p.hidden)),
            ("ik", ix.keyProjection, (p.D, p.hidden)), ("o", attn.wo, (p.hidden, p.H * p.D)),
        ]
        for (name, m, want) in shapes where outIn(m) != want {
            return .failure(PlanError(description: "\(name) shape \(outIn(m)) != \(want)"))
        }
        let blkA = p.bitsA == 4 ? 512 : 256
        let blkB = (p.fmtB == 1 || p.bitsB == 4) ? 512 : 256
        let blkO = p.bitsO == 4 ? 512 : 256
        let one = projShape(1), many = projShape(2)
        let rowsTG = max(one.nsg * one.rps, many.nsg * many.rps)
        if p.hidden % blkA != 0 || p.hidden % blkB != 0 || (p.H * p.D) % blkO != 0 || p.hidden % rowsTG != 0
            || [p.H * p.D, p.KVH * p.D, p.GI * p.D, p.D].contains(where: { $0 % rowsTG != 0 })
        {
            return .failure(PlanError(description: "projection widths not multiples of the kernel blocks"))
        }
        if p.D % 32 != 0 || p.D > 256 || p.KVH < 1 || p.H % p.KVH != 0 || p.GI != p.KVH {
            return .failure(PlanError(description: "head geometry"))
        }
        p.HG = p.H / p.KVH
        p.HS = p.HG % 4 == 0 ? 4 : p.HG
        if p.HS > 8 || p.HG * p.D * 4 > 16384 { return .failure(PlanError(description: "query heads per KV group")) }
        let cpb = p.block / 32
        if p.block % 32 != 0 || cpb & (cpb - 1) != 0 || p.keep < 1 || p.local < 0 || (p.HG / p.HS) * cpb > 32
            || 4 * (p.HG * p.D + max(1, cpb / 2) * p.HG * (p.D + 2)) + 1400 + 4 * p.keep > 32768
        {
            return .failure(PlanError(description: "index block geometry"))
        }
        // Both RoPE modules are built non-traditional and unscaled from the same config (MiniMaxM3Attention / Indexer).
        p.ROT = attn.rotaryDim
        if p.ROT % 2 != 0 || p.ROT > p.D || !((p.ROT / 2) % 32 == 0 || p.ROT <= 32) {
            return .failure(PlanError(description: "rotary dims"))
        }
        p.log2base = log2(Double(attn.ropeTheta))
        let eps = Set([attn.qNorm.eps, attn.kNorm.eps, ix.queryNorm.eps, ix.keyNorm.eps])
        guard eps.count == 1 else { return .failure(PlanError(description: "head norms with different eps")) }
        p.epsHead = eps.first!
        return .success(p)
    }

    func plan(_ layer: MiniMaxM3DecoderLayer) -> Result<Plan, PlanError> {
        let key = ObjectIdentifier(layer)
        if let (ref, cached) = lock.withLock({ plans[key] }), ref.layer === layer { return cached }
        let built = Self.buildPlan(layer)
        lock.withLock {
            plans = plans.filter { $0.value.0.layer != nil }
            plans[key] = (WeakLayer(layer), built)
        }
        return built
    }

    // MARK: kernels

    static let specs: [String: (inputs: [String], outputs: [String], contiguous: Bool)] = [
        "qkv": (
            ["x", "nw", "q_w", "q_s", "q_b", "k_w", "k_s", "k_b", "v_w", "v_s", "v_b", "iq_w", "iq_s", "iq_b", "ik_w",
             "ik_s", "ik_b"], ["y"], true
        ),
        "heads": (["y", "qn", "kn", "iqn", "ikn", "meta", "kc", "vc", "ic"], ["q", "iq", "cnt", "kn_out", "vn_out", "ikn_out"], false),
        "scores": (["iq", "ic", "meta"], ["sc"], false),
        "msa": (["q", "kc", "vc", "sc", "cnt", "meta", "scale"], ["out", "po", "pm", "pl"], false),
        "oproj": (["o", "xr", "w", "s", "b"], ["h", "ssp"], true),
    ]

    static func metalType(_ dtype: DType) -> String {
        switch dtype {
        case .bfloat16: return "bfloat16_t"
        case .float16: return "float16_t"
        case .float32: return "float"
        case .uint8: return "uint8_t"
        case .uint32: return "uint32_t"
        default: fatalError("MiniMax-M3 fused attention: no Metal type for \(dtype)")
        }
    }

    /// Python repr of a float (what `kernels.prefix()` writes): the shortest text that round-trips.
    static func floatText(_ v: Double) -> String {
        var s = "\(v)"
        if s.hasSuffix(".0") && !s.contains("e") { s = String(s.dropLast(2)) + ".0" }
        return s
    }

    /// A Float config value as the double Python holds for it (the JSON decimal): `1e-06`, not 9.999999974752427e-07,
    /// so the source text matches vMLX's (the compiled constant is the same float either way).
    static func decimal(_ v: Float) -> Double { Double("\(v)")! }

    /// The constant lines a host prepends to a body (Python `kernels.prefix`).
    static func prefix(types: [(String, String)], ints: [(String, Int)], floats: [(String, Double)] = []) -> String {
        var out = ""
        for (n, t) in types { out += "  using \(n) = \(t);\n" }
        for (n, v) in ints { out += "  constexpr int \(n) = \(v);\n" }
        for (n, v) in floats { out += "  constexpr float \(n) = \(floatText(v))f;\n" }
        return out
    }

    static func body(_ name: String) -> String { MiniMaxM3FusedKernelSources.byName["attn_\(name)"]! }

    /// FNV-1a 64 of the full source: MLX caches compiled kernels by name, so the name carries the source's identity.
    static func digest(_ text: String) -> String {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for b in text.utf8 {
            h ^= UInt64(b)
            h = h &* 0x0000_0100_0000_01b3
        }
        return String(h, radix: 16)
    }

    private var kernels: [String: MLXFast.MLXFastKernel] = [:]

    func kernel(
        _ name: String, types: [(String, String)], ints: [(String, Int)], floats: [(String, Double)] = []
    ) -> MLXFast.MLXFastKernel {
        let source = Self.prefix(types: types, ints: ints, floats: floats) + Self.body(name)
        let header = MiniMaxM3FusedKernelSources.attnCommon
        let key = name + "|" + source
        if let k = lock.withLock({ kernels[key] }) { return k }
        let spec = Self.specs[name]!
        let k = MLXFast.metalKernel(
            name: "vmlx_m3_fused_attn_\(name)_\(Self.digest(header + source))", inputNames: spec.inputs,
            outputNames: spec.outputs, source: source, header: header, ensureRowContiguous: spec.contiguous)
        lock.withLock { kernels[key] = k }
        return k
    }

    private let dummies = NSCache<NSString, MLXArrayBox>()
    final class MLXArrayBox { let a: MLXArray; init(_ a: MLXArray) { self.a = a } }

    func dummy(_ dtype: DType = .float32) -> MLXArray {
        let key = "\(dtype)" as NSString
        if let d = dummies.object(forKey: key) { return d.a }
        let d = MLXArray.zeros([8], dtype: dtype)
        eval(d)
        dummies.setObject(MLXArrayBox(d), forKey: key)
        return d
    }

    // MARK: launches (fused_attn.py)

    /// K1: [R, hidden] residual rows -> [R, (H + 2 KVH + GI + 1) * D] float32 (q | k | v | index q | index k).
    func qkv(_ layer: MiniMaxM3DecoderLayer, _ p: Plan, _ x2: MLXArray) -> MLXArray {
        let attn = layer.selfAttn
        let ix = attn.indexer!
        let R = x2.dim(0)
        let q = attn.wq as! QuantizedLinear, k = attn.wk as! QuantizedLinear, v = attn.wv as! QuantizedLinear
        let iq = ix.queryProjection as! QuantizedLinear, ik = ix.keyProjection as! QuantizedLinear
        let (nq, nk, nv, niq, nik) = (p.H * p.D, p.KVH * p.D, p.KVH * p.D, p.GI * p.D, p.D)
        let (nsg, rps, rt) = Self.projShape(R)
        let sbt = p.fmtB == 2 ? Self.metalType(iq.scales.dtype) : "float"
        let kern = kernel(
            "qkv",
            types: [
                ("xt", Self.metalType(x2.dtype)), ("nwt", Self.metalType(layer.inputLayerNorm.weight.dtype)),
                ("sat", Self.metalType(q.scales.dtype)), ("sbt", sbt),
            ],
            ints: [
                ("KIN", p.hidden), ("NQ", nq), ("NK", nk), ("NV", nv), ("NIQ", niq), ("NIK", nik), ("BITS_A", p.bitsA),
                ("GS_A", p.gsA), ("FMT_B", p.fmtB), ("BITS_B", p.bitsB), ("GS_B", p.gsB), ("NSG", nsg), ("RPS", rps),
                ("RT", rt),
            ],
            floats: [("EPS", Self.decimal(p.epsIn))])
        let (iqb, ikb) = p.fmtB == 1 ? (dummy(), dummy()) : (iq.biases!, ik.biases!)
        let ntot = nq + nk + nv + niq + nik
        return kern(
            [
                x2, layer.inputLayerNorm.weight, q.weight, q.scales, q.biases!, k.weight, k.scales, k.biases!, v.weight,
                v.scales, v.biases!, iq.weight, iq.scales, iqb, ik.weight, ik.scales, ikb,
            ],
            grid: (32 * nsg * (ntot / (nsg * rps)) * ((R + rt - 1) / rt), 1, 1), threadGroup: (32 * nsg, 1, 1),
            outputShapes: [[R, ntot]], outputDTypes: [.float32])[0]
    }

    /// K2 with INPLACE = 0: (q [R, H, D] f32, index q [R, GI, D] f32, counters, new K / V / index rows).
    func heads(
        _ layer: MiniMaxM3DecoderLayer, _ p: Plan, _ y: MLXArray, _ meta: MLXArray, laneDTypes: (DType, DType, DType)
    ) -> (q: MLXArray, iq: MLXArray, cnt: MLXArray, k: MLXArray, v: MLXArray, ik: MLXArray) {
        let attn = layer.selfAttn
        let ix = attn.indexer!
        let R = y.dim(0)
        let (kdt, vdt, idt) = laneDTypes
        let cnt = max(8, R * p.KVH)
        let kern = kernel(
            "heads",
            types: [
                ("kt", Self.metalType(kdt)), ("vt", Self.metalType(vdt)), ("it", Self.metalType(idt)),
                ("qnt", Self.metalType(attn.qNorm.weight.dtype)), ("knt", Self.metalType(attn.kNorm.weight.dtype)),
                ("iqnt", Self.metalType(ix.queryNorm.weight.dtype)), ("iknt", Self.metalType(ix.keyNorm.weight.dtype)),
            ],
            ints: [
                ("H", p.H), ("KVH", p.KVH), ("GI", p.GI), ("D", p.D), ("ROT", p.ROT), ("INPLACE", 0), ("CNT", cnt),
            ],
            floats: [("EPS", Self.decimal(p.epsHead)), ("LOG2BASE", p.log2base)])
        let ht = p.H + 2 * p.KVH + p.GI + 1
        let threads = 32 * R * ht
        let d = dummy()
        let out = kern(
            [y, attn.qNorm.weight, attn.kNorm.weight, ix.queryNorm.weight, ix.keyNorm.weight, meta, d, d, d],
            grid: ((threads + 127) / 128 * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[R, p.H, p.D], [R, p.GI, p.D], [cnt], [R, p.KVH, p.D], [R, p.KVH, p.D], [R, 1, p.D]],
            outputDTypes: [.float32, .float32, .uint32, kdt, vdt, idt])
        return (out[0], out[1], out[2], out[3], out[4], out[5])
    }

    /// K3: block scores [R, GI, nb] float32.
    func scores(_ p: Plan, _ iq: MLXArray, _ indexLane: MLXArray, _ meta: MLXArray, nb: Int) -> MLXArray {
        let R = iq.dim(0)
        let pairs = R * p.GI
        let pt = pairs <= 4 ? 4 : (pairs <= 8 ? 8 : 16)
        let kern = kernel(
            "scores", types: [("it", Self.metalType(indexLane.dtype))],
            ints: [("GI", p.GI), ("D", p.D), ("BLOCK", p.block), ("LOCAL", p.local), ("PT", pt)])
        return kern(
            [iq, indexLane, meta], grid: (p.block * nb, 1, 1), threadGroup: (p.block, 1, 1),
            outputShapes: [[R, p.GI, nb]], outputDTypes: [.float32])[0]
    }

    /// K4: attention output [R, H, D] float32 over each row's selected blocks.
    func msa(
        _ p: Plan, _ q: MLXArray, _ kLane: MLXArray, _ vLane: MLXArray, _ sc: MLXArray, _ cnt: MLXArray,
        _ meta: MLXArray
    ) -> MLXArray {
        let R = q.dim(0)
        let nsg = p.HG / p.HS
        let threads = 32 * nsg * (p.block / 32)
        let kern = kernel(
            "msa", types: [("kt", Self.metalType(kLane.dtype)), ("vt", Self.metalType(vLane.dtype))],
            ints: [
                ("G", p.KVH), ("HG", p.HG), ("D", p.D), ("BLOCK", p.block), ("KEEP", p.keep), ("HS", p.HS), ("NSG", nsg),
            ])
        let rg = R * p.KVH
        return kern(
            [q, kLane, vLane, sc, cnt, meta, MLXArray([p.scale])],
            grid: (threads * p.keep, rg, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[R, p.H, p.D], [rg * p.keep * p.HG * p.D], [rg * p.keep * p.HG], [rg * p.keep * p.HG]],
            outputDTypes: [.float32, .float32, .float32, .float32])[0]
    }

    /// K5: x2 + o_proj(o) [R, hidden] in x2's dtype.
    func oproj(_ layer: MiniMaxM3DecoderLayer, _ p: Plan, _ o: MLXArray, _ x2: MLXArray) -> MLXArray {
        let m = layer.selfAttn.wo as! QuantizedLinear
        let R = o.dim(0)
        let (nsg, rps, rt) = Self.projShape(R)
        let kern = kernel(
            "oproj", types: [("xt", Self.metalType(x2.dtype)), ("st", Self.metalType(m.scales.dtype))],
            ints: [
                ("KIN", p.H * p.D), ("NOUT", p.hidden), ("BITS", p.bitsO), ("GS", p.gsO), ("NSG", nsg), ("RPS", rps),
                ("RT", rt), ("SSP", 0),
            ])
        return kern(
            [o, x2, m.weight, m.scales, m.biases!],
            grid: (32 * nsg * (p.hidden / (nsg * rps)) * ((R + rt - 1) / rt), 1, 1), threadGroup: (32 * nsg, 1, 1),
            outputShapes: [[R, p.hidden], [1]], outputDTypes: [x2.dtype, .float32])[0]
    }

    /// (K, V, index-key) lane dtypes: the lanes' own when they exist, else what the stock path would append.
    static func laneDTypes(_ layer: MiniMaxM3DecoderLayer, _ cache: MiniMaxM3SparseCache, _ x: DType)
        -> (DType, DType, DType)
    {
        let state = cache.state
        if state.count == 3 { return (state[0].dtype, state[1].dtype, state[2].dtype) }
        let attn = layer.selfAttn
        let promote = { (a: DType, b: DType) -> DType in
            (MLXArray.zeros([1], dtype: a) + MLXArray.zeros([1], dtype: b)).dtype
        }
        return (promote(x, attn.kNorm.weight.dtype), x, promote(x, attn.indexer!.keyNorm.weight.dtype))
    }

    // MARK: forward

    /// Why `layer` cannot take the fused path (nil: it can).
    func unsupportedReason(_ layer: MiniMaxM3DecoderLayer) -> String? {
        switch plan(layer) {
        case .success: return nil
        case .failure(let e): return e.description
        }
    }

    /// True when a call of `x` on `layer` with `cache` takes the fused path (Python `use_fused`, without the switch).
    func covers(_ layer: MiniMaxM3DecoderLayer, x: MLXArray, cache: KVCache?) -> Bool {
        guard x.ndim == 3, x.dim(0) == 1, x.dim(1) >= 1, cache is MiniMaxM3SparseCache,
            [DType.bfloat16, .float16, .float32].contains(x.dtype),
            Device.defaultDevice().deviceType == .gpu
        else { return false }
        if case .success = plan(layer) { return true }
        return false
    }

    /// `x + self_attn(input_layernorm(x))` for one sequence `x` `[1, R, hidden]` at the cache's offset; appends the
    /// rows' K / V / index keys to `cache`. Rows past 16 run in consecutive chunks of at most 16.
    func attentionBlock(_ layer: MiniMaxM3DecoderLayer, x: MLXArray, cache: MiniMaxM3SparseCache) -> MLXArray {
        let R = x.dim(1)
        if R > Self.maxRows {
            var outs: [MLXArray] = []
            var start = 0
            while start < R {
                let n = min(Self.maxRows, R - start)
                outs.append(attentionBlock(layer, x: x[0..., start ..< (start + n)], cache: cache))
                start += n
            }
            return concatenated(outs, axis: 1)
        }
        guard case .success(let p) = plan(layer) else { preconditionFailure("attentionBlock on an unsupported layer") }
        let hidden = x.dim(2)
        let x2 = x.reshaped(R, hidden)
        let offset = cache.offset
        let y = qkv(layer, p, x2)
        let dts = Self.laneDTypes(layer, cache, x.dtype)
        let nb = (offset + R + p.block - 1) / p.block
        let meta = MLXArray([Int32(offset), Int32(R), Int32(nb), 0, 0, 0, 0, 0])
        let h2 = heads(layer, p, y, meta, laneDTypes: dts)
        let (kLane, vLane) = cache.update(
            keys: h2.k.transposed(1, 0, 2).expandedDimensions(axis: 0),
            values: h2.v.transposed(1, 0, 2).expandedDimensions(axis: 0))
        let iLane = cache.updateIndexKeys(h2.ik.transposed(1, 0, 2).expandedDimensions(axis: 0))
        let sc: MLXArray
        if nb > p.keep {
            sc = scores(p, h2.iq, iLane, meta, nb: nb)
            Self.stats.add("sparse_calls")
        } else {
            sc = dummy()
        }
        let o = msa(p, h2.q, kLane, vLane, sc, h2.cnt, meta)
        let h = oproj(layer, p, o.reshaped(R, p.H * p.D), x2)
        Self.stats.add("calls")
        Self.stats.add("rows", R)
        return h.reshaped(1, R, hidden)
    }
}

/// The fused path registered with `MiniMaxM3FusedDecode`: K1-K5 attention (this file); the MoE half (K6-K8) is
/// `mlpBlock` (nil = stock until ported).
final class MiniMaxM3FusedKernelPath: MiniMaxM3FusedPath, @unchecked Sendable {
    static let shared = MiniMaxM3FusedKernelPath()

    func attentionBlock(
        layer: Int, module: MiniMaxM3DecoderLayer, x: MLXArray, cache: KVCache?, verify: Bool
    ) -> MLXArray? {
        let fa = MiniMaxM3FusedAttention.shared
        guard let sparse = cache as? MiniMaxM3SparseCache, fa.covers(module, x: x, cache: cache) else { return nil }
        return fa.attentionBlock(module, x: x, cache: sparse)
    }

    func mlpBlock(layer: Int, module: MiniMaxM3DecoderLayer, h: MLXArray, verify: Bool) -> MLXArray? {
        nil
    }
}
