// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

/// vMLX's MiniMax-M3 fused MoE block (K6 / K7 / R / K8, `vmlx_engine/models/minimax_m3/fused_moe.py` at
/// `MiniMaxM3FusedKernelSources.vmlxCommit`), ported onto the identical Metal sources: `resid + block(x_normed)` of
/// one sparse MoE layer for 1...16 rows in four launches.
///
/// - K6 `m3_router`: fp32 router logits (MLX's one-row gemv, transcribed) + sigmoid of every row; the Hadamard-128
///   rotation of the routed input.
/// - K7 `m3_moe_gate_up`: routing (top-k of sigmoid + correction bias, normalized, scaled); routed JANGT gate / up +
///   SwiGLU-OAI per (row, slot); the shared expert's (affine 4-bit g64) gate / up + SwiGLU-OAI per row (1-2 rows).
/// - R: vMLX's `rotate_rows` of the routed down input with each slot's expert column scales.
/// - K8 `m3_moe_down`: routed JANGT down + routing-weighted combine, shared expert down, residual add.
///
/// Row invariant (a row's arithmetic never depends on the other rows of the launch). vMLX's kernels follow vMLX
/// Python's unfused ops op for op, so they are bit-identical to THAT path; the Swift unfused path (mlx-swift's ops,
/// this repo's JANGT kernels) is compared and reported, not assumed equal.
///
/// Shared expert: K7 / K8 compute it in-kernel (one-row qmv_fast per output row, MLX's bf16 rounding, the COMPILED
/// SwiGLU op by op: JIT fast-exp sigmoid) up to `sharedInKernelRows` rows. Past that the host computes it on row-exact
/// projections (`MiniMaxM3RowExact.linear`) with a compiled SwiGLU of vMLX's op order (`swiglu`, `mx.compile`), and K8
/// reads it (`SHARED = 0`). The default here is 16 (always in-kernel): Swift's stock SwiGLU is NOT compiled (precise
/// sigmoid), so only the in-kernel path is known to give a row the same arithmetic at every row count; vMLX's 2 is
/// available through `VMLX_M3_FUSED_SHARED_ROWS` once its row invariance is shown (tests check both).
final class MiniMaxM3FusedMoE: @unchecked Sendable {
    static let shared = MiniMaxM3FusedMoE()
    static let maxRows = 16
    private static let sharedRowsLock = NSLock()
    nonisolated(unsafe) private static var _sharedInKernelRows: Int =
        Int(ProcessInfo.processInfo.environment["VMLX_M3_FUSED_SHARED_ROWS"] ?? "") ?? 16
    /// Rows up to which the shared expert runs inside K7 / K8 (`VMLX_M3_FUSED_SHARED_ROWS`; tests may set it).
    static var sharedInKernelRows: Int {
        get { sharedRowsLock.withLock { _sharedInKernelRows } }
        set { sharedRowsLock.withLock { _sharedInKernelRows = newValue } }
    }

    /// vMLX's `swiglu` (`@mx.compile(shapeless=True)`): clip the gate above, clip the up both ways,
    /// `(g * sigmoid(alpha * g)) * (u + 1)`, compiled so the sigmoid is the JIT one.
    private var compiledSwiGLU: [String: @Sendable ([MLXArray]) -> [MLXArray]] = [:]

    func swiglu(up: MLXArray, gate: MLXArray, alpha: Float, limit: Float) -> MLXArray {
        let key = "\(alpha.bitPattern)/\(limit.bitPattern)"
        let f: @Sendable ([MLXArray]) -> [MLXArray]
        if let cached = lock.withLock({ compiledSwiGLU[key] }) {
            f = cached
        } else {
            f = compile(shapeless: true) { a in
                let glu = minimum(a[1], limit)
                let lin = clip(a[0], min: -limit, max: limit)
                return [(glu * sigmoid(alpha * glu)) * (lin + 1)]
            }
            lock.withLock { compiledSwiGLU[key] = f }
        }
        return f([up, gate])[0]
    }

    /// The shared expert of `x2` `[R, D]` off-kernel: every projection on the one-row arithmetic per row.
    func sharedRows(_ L: Layer, _ x2: MLXArray) -> MLXArray {
        let x = x2.expandedDimensions(axis: 0)
        let lin = MiniMaxM3RowExact.linear
        let h = swiglu(up: lin(L.sharedUp, x), gate: lin(L.sharedGate, x), alpha: L.alpha, limit: L.limit)
        return lin(L.sharedDown, h).reshaped(x2.dim(0), L.D)
    }

    final class Stats: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [String: Int] = [:]
        func add(_ key: String, _ n: Int = 1) { lock.withLock { values[key, default: 0] += n } }
        func reset() { lock.withLock { values = [:] } }
        var snapshot: [String: Int] { lock.withLock { values } }
    }
    static let stats = Stats()

    static let inputs: [String: [String]] = [
        "m3_router": ["x", "gw", "su"],
        "m3_moe_gate_up": [
            "fc", "xr", "x", "sig", "bias", "wg", "sg", "wu", "su_", "ocol", "woutg", "woutu", "sgw", "sgs", "sgb", "suw",
            "sus", "sub",
        ],
        "m3_moe_down": [
            "fc", "hr", "h", "inds", "scores", "wd", "sd", "ocol", "wout", "hs", "sdw", "sds", "sdb", "resid", "sdo",
            "ry",
        ],
    ]
    static let outputs: [String: [String]] = [
        "m3_router": ["sig", "xr"], "m3_moe_gate_up": ["h", "hs", "inds", "scores"], "m3_moe_down": ["out"],
    ]

    private let lock = NSLock()
    private var kernels: [String: MLXFast.MLXFastKernel] = [:]

    /// The compiled kernel (vMLX `_kernel`: the body as the source, `m3_moe_common` as the header; MLX appends the
    /// template arguments to the name).
    func kernel(_ name: String) -> MLXFast.MLXFastKernel {
        if let k = lock.withLock({ kernels[name] }) { return k }
        let header = MiniMaxM3FusedKernelSources.m3MoeCommon
        let body = MiniMaxM3FusedKernelSources.byName[name]!
        let k = MLXFast.metalKernel(
            name: "vmlx_\(name)_\(MiniMaxM3FusedAttention.digest(header + body))", inputNames: Self.inputs[name]!,
            outputNames: Self.outputs[name]!, source: body, header: header)
        lock.withLock { kernels[name] = k }
        return k
    }

    // MARK: the layer's arrays

    /// Everything the kernels read for one MoE layer.
    struct Layer {
        let D: Int, I: Int, E: Int, KS: Int
        let gw: MLXArray  // [E, D] f32 router
        let bias: MLXArray  // [E] f32 correction bias
        let rsf: Float, alpha: Float, limit: Float
        let trellis: JANGTRoutedExpertLayer.TrellisTensors
        let sharedGate: QuantizedLinear, sharedUp: QuantizedLinear, sharedDown: QuantizedLinear
        let sharedMLP: MiniMaxM3MLP
        /// Outlier arrays as the kernels index them ([E, max(NOUT, 1)] / [E, N, max(NOUT, 1)]).
        let ocolGU: MLXArray, woutG: MLXArray, woutU: MLXArray, ocolDn: MLXArray, woutDn: MLXArray
    }

    /// The per-block cache (`MiniMaxM3SparseMoeBlock._fusedMoELayer`): nil `layer` = the block is not covered.
    final class LayerBox: @unchecked Sendable {
        let layer: Layer?
        init(_ layer: Layer?) { self.layer = layer }
    }

    private static func affine4(_ m: Linear) -> QuantizedLinear? {
        guard let q = m as? QuantizedLinear, q.mode == .affine, q.bits == 4, q.groupSize == 64, q.biases != nil,
            q.bias == nil
        else { return nil }
        return q
    }

    /// The kernels' contract (vMLX `supported`), or nil: JANGT trellis gate/up and down with SwiGLU-OAI and a limit, an
    /// affine-4 / group-64 shared expert of the routed width with the same alpha / limit, an fp32 router, top-k > 1
    /// normalized, hidden a multiple of 1024 and >= 16 E (MLX's one-row gemv configuration K6 transcribes).
    static func build(_ block: MiniMaxM3SparseMoeBlock) -> Layer? {
        guard let routed = block.experts as? JANGTRoutedExpertLayer, let t = routed.trellisTensors,
            case .oai(let alpha) = t.activation, t.limit > 0
        else { return nil }
        let sh = block.sharedExperts
        guard let sg = affine4(sh.gateProj), let su = affine4(sh.upProj), let sd = affine4(sh.downProj),
            sh.alpha == alpha, sh.limit == t.limit
        else { return nil }
        let (D, I, E) = (t.hiddenSize, t.intermediateSize, t.gate.experts)
        guard sg.weight.dim(0) == I, sd.weight.dim(0) == D, I % 512 == 0, D % 512 == 0, D % 1024 == 0, D >= 16 * E,
            I % 128 == 0, D % 128 == 0
        else { return nil }
        guard !(block.gate is QuantizedLinear), block.gate.bias == nil, block.gate.weight.dtype == .float32,
            block.eScoreCorrectionBias.dtype == .float32, block.normTopKProb, block.topK >= 2
        else { return nil }
        for k in [t.gate.k, t.down.k] where ![1.0, 1.5, 2.0, 2.5, 3.0].contains(k) { return nil }
        func padded(_ cols: MLXArray, _ wout: MLXArray, rows: Int) -> (MLXArray, MLXArray) {
            if cols.dim(1) > 0 { return (cols.asType(.int32), wout.asType(.float16)) }
            return (
                MLXArray.full([E, 1], values: MLXArray(Int32(-1))), MLXArray.zeros([E, rows, 1], dtype: .float16)
            )
        }
        let (ocolGU, woutG) = padded(t.gate.outlierColumnIndices, t.gate.outlierWeights, rows: I)
        let (_, woutU) = padded(t.up.outlierColumnIndices, t.up.outlierWeights, rows: I)
        let (ocolDn, woutDn) = padded(t.down.outlierColumnIndices, t.down.outlierWeights, rows: D)
        return Layer(
            D: D, I: I, E: E, KS: block.topK, gw: block.gate.weight, bias: block.eScoreCorrectionBias,
            rsf: block.routedScalingFactor, alpha: alpha, limit: t.limit, trellis: t, sharedGate: sg, sharedUp: su,
            sharedDown: sd, sharedMLP: sh, ocolGU: ocolGU, woutG: woutG, woutU: woutU, ocolDn: ocolDn, woutDn: woutDn)
    }

    func layer(_ block: MiniMaxM3SparseMoeBlock) -> Layer? {
        if let box = lock.withLock({ block._fusedMoELayer }) { return box.layer }
        let box = LayerBox(Self.build(block))
        lock.withLock { block._fusedMoELayer = box }
        return box.layer
    }

    // MARK: launches (fused_moe.py)

    /// A positive float32 constant as the int bit pattern the kernels read back with `as_type<float>`.
    static func fbits(_ v: Float) -> Int {
        let b = Int32(bitPattern: v.bitPattern)
        precondition(b >= 0, "fused M3 MoE: template float \(v) must be positive")
        return Int(b)
    }

    /// The code statistics of a JANGT width, zero-padded to 16 floats (vMLX `_fc`).
    static func fc(_ k: Double) -> MLXArray {
        let (mu0, mu1, isd0, isd1) = JANGTCode.codeStats(k)!
        return MLXArray([mu0, mu1, isd0, isd1] + [Float](repeating: 0, count: 12))
    }

    /// K6: (sig [rows, E] f32, xr [rows, D] f32 -- not written with `rot` false: the fast routed kernels pack their
    /// own input).
    func router(_ L: Layer, _ x2: MLXArray, rot: Bool = true) -> (MLXArray, MLXArray) {
        let rows = x2.dim(0)
        let ntg = L.E + (rot ? rows * ((L.D + 1023) / 1024) : 0)
        let out = kernel("m3_router")(
            [x2, L.gw, L.trellis.gateUpSigns],
            template: [("T", x2.dtype), ("D", L.D), ("E", L.E), ("ROWS", rows), ("ROT", rot ? 1 : 0)],
            grid: (256 * ntg, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, L.E], [rows, L.D]], outputDTypes: [.float32, .float32])
        return (out[0], out[1])
    }

    func common(_ L: Layer, rows: Int, dtype: DType) -> [(String, any KernelTemplateArg)] {
        [("T", dtype), ("D", L.D), ("I", L.I), ("E", L.E), ("ROWS", rows), ("KS", L.KS), ("L", 12)]
    }

    /// K7: (h [rows * KS, I] f32 -- not written with `routed` false, hs [rows, I], inds [rows, KS] u32, scores
    /// [rows, KS] f32). `xr` nil (only with `routed` false): K6 ran without the rotation; `sig` stands in (vMLX).
    func gateUp(_ L: Layer, _ x2: MLXArray, sig: MLXArray, xr: MLXArray?, shared: Bool, routed: Bool = true)
        -> (h: MLXArray, hs: MLXArray, inds: MLXArray, scores: MLXArray)
    {
        precondition(xr != nil || !routed, "fused M3 MoE: K7 needs the rotated input for the routed experts")
        let rows = x2.dim(0)
        let (g, u) = (L.trellis.gate, L.trellis.up)
        let template =
            common(L, rows: rows, dtype: x2.dtype) + [
                ("KB2", Int(g.k * 2)), ("W", g.packed.dim(-1)), ("NOUT", g.outlierColumns), ("ROUTED", routed ? 1 : 0),
                ("SHARED", shared ? 1 : 0), ("ALPHA", Self.fbits(L.alpha)), ("LIMIT", Self.fbits(L.limit)),
                ("RSF", Self.fbits(L.rsf)),
            ]
        let out = kernel("m3_moe_gate_up")(
            [
                Self.fc(g.k), xr ?? sig, x2, sig, L.bias, g.packed, g.scale, u.packed, u.scale, L.ocolGU, L.woutG,
                L.woutU, L.sharedGate.weight, L.sharedGate.scales, L.sharedGate.biases!, L.sharedUp.weight,
                L.sharedUp.scales, L.sharedUp.biases!,
            ],
            template: template,
            grid: (64, rows * L.KS + (shared ? rows : 0), L.I / 8), threadGroup: (64, 1, 1),
            outputShapes: [[rows * L.KS, L.I], [rows, L.I], [rows, L.KS], [rows, L.KS]],
            outputDTypes: [.float32, x2.dtype, .uint32, .float32])
        return (out[0], out[1], out[2], out[3])
    }

    /// R: the routed down input rotated with each slot's expert column scales.
    func downInput(_ L: Layer, _ h: MLXArray, inds: MLXArray) -> MLXArray {
        JANGTRoutedExpertLayer.rotateRows(h, signs: L.trellis.downSigns, rowExperts: inds.reshaped(-1))
    }

    /// K8: the new residual rows [rows, D]; `sdo` = the shared expert's output computed by the caller (else K8
    /// computes it from `hs`); `ry` = the routed experts' router-weighted sum [rows, D] f32 computed by the caller (the
    /// fast JANGT kernels; else K8 runs the routed down from `hr`).
    func down(
        _ L: Layer, hr: MLXArray?, h: MLXArray, inds: MLXArray, scores: MLXArray, hs: MLXArray?, r2: MLXArray,
        sdo: MLXArray?, ry: MLXArray? = nil
    ) -> MLXArray {
        precondition(hr != nil || ry != nil, "fused M3 MoE: K8 needs the rotated down input or the routed sum")
        let rows = r2.dim(0)
        let d = L.trellis.down
        let shared = sdo == nil
        let template =
            common(L, rows: rows, dtype: r2.dtype) + [
                ("KB2", Int(d.k * 2)), ("W", d.packed.dim(-1)), ("NOUT", d.outlierColumns), ("SHARED", shared ? 1 : 0),
                ("ROUTED", ry == nil ? 1 : 0),
            ]
        return kernel("m3_moe_down")(
            [
                Self.fc(d.k), hr ?? h, h, inds, scores, d.packed, d.scale, L.ocolDn, L.woutDn, hs ?? r2,
                L.sharedDown.weight, L.sharedDown.scales, L.sharedDown.biases!, r2, sdo ?? r2, ry ?? Self.zeroFC,
            ],
            template: template, grid: (64, rows, L.D / 8), threadGroup: (64, 1, 1),
            outputShapes: [[rows, L.D]], outputDTypes: [r2.dtype])[0]
    }

    /// vMLX `_fc((0.0,))`: the placeholder `ry` when K8 runs the routed experts itself.
    static var zeroFC: MLXArray { MLXArray([Float](repeating: 0, count: 16)) }

    // MARK: forward

    func covers(_ block: MiniMaxM3SparseMoeBlock, x: MLXArray) -> Bool {
        guard x.ndim == 3, x.dim(0) == 1, x.dim(1) >= 1, [DType.bfloat16, .float16].contains(x.dtype),
            Device.defaultDevice().deviceType == .gpu, let L = layer(block), x.dim(2) == L.D,
            L.sharedGate.scales.dtype == x.dtype
        else { return false }
        return true
    }

    /// `resid + block(xNormed)` for one sequence `[1, R, D]`; rows past 16 run in consecutive chunks of at most 16.
    func residual(_ block: MiniMaxM3SparseMoeBlock, xNormed: MLXArray, resid: MLXArray) -> MLXArray {
        let R = xNormed.dim(1)
        if R > Self.maxRows {
            var outs: [MLXArray] = []
            var start = 0
            while start < R {
                let n = min(Self.maxRows, R - start)
                outs.append(
                    residual(
                        block, xNormed: xNormed[0..., start ..< (start + n)], resid: resid[0..., start ..< (start + n)]))
                start += n
            }
            return concatenated(outs, axis: 1)
        }
        let L = layer(block)!
        let x2 = contiguous(xNormed.reshaped(R, L.D))
        let r2 = contiguous(resid.reshaped(R, L.D).asType(x2.dtype))
        let shared = R <= Self.sharedInKernelRows
        let sdo = shared ? nil : sharedRows(L, x2)
        let out: MLXArray
        if JANGTFastDecode.isEnabled, JANGTFastDecode.supported(L.trellis) {
            // VMLX_JANGT_FAST=1 (vMLX `fused_moe_residual`): routing + shared expert in K6 / K7, the routed experts on
            // the fast JANGT kernels (what the JANGT routed layer runs with the lever), K8 adds shared + residual.
            let (sig, _) = router(L, x2, rot: false)
            let gu = gateUp(L, x2, sig: sig, xr: nil, shared: shared, routed: false)
            let ry = JANGTFastDecode.routed(x2, ids: gu.inds, weights: gu.scores, L.trellis)
            out = down(
                L, hr: nil, h: gu.h, inds: gu.inds, scores: gu.scores, hs: shared ? gu.hs : nil, r2: r2, sdo: sdo,
                ry: ry)
            Self.stats.add("fast_routed")
        } else {
            let (sig, xr) = router(L, x2)
            let gu = gateUp(L, x2, sig: sig, xr: xr, shared: shared)
            out = down(
                L, hr: downInput(L, gu.h, inds: gu.inds), h: gu.h, inds: gu.inds, scores: gu.scores,
                hs: shared ? gu.hs : nil, r2: r2, sdo: sdo)
        }
        Self.stats.add("calls")
        Self.stats.add("rows", R)
        return out.reshaped(1, R, L.D)
    }
}
