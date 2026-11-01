// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast

/// vMLX's fast JANGT trellis decode kernels (`vmlx_engine/jangt/fast_decode.py`, the production path `routed_fast`),
/// behind `VMLX_JANGT_FAST=1` (default off, read per call like Python's `enabled()`).
///
/// Same math as the v3 decode kernels (`JANGTKernels`): 16-weight lane chunks of 12-bit trellis states, half2 codes,
/// half2 fma chain per lane chunk, fp32 fold, code statistics and row scales in the epilogue, outlier columns; with
/// - the activation side packed once per row in ONE launch (`pack1`: Hadamard-128 rotation, fp16 range guard, half2
///   codes, even / odd sums);
/// - the code statistics folded into the per-chunk fold (one fp32 accumulator per output row and projection);
/// - every output row's window words loaded up front per block, states extracted with `extract_bits`;
/// - down: the slot combine inside the kernel for one row, per-slot + combine kernel for 2+ rows (each row computed
///   identically, so a decode step == the same row inside a DSpark verify).
///
/// NOT bit-identical to v3 (decode and verify switch together). The Metal is GENERATED per shape in Python; this is the
/// same generator, string for string (`JANGTFastDecodeTests` compares every production configuration's source SHA-256
/// with vMLX Python's, at the test geometries and M3.1's real shapes), with Python's kernel names, input names, grids
/// and threadgroups, so Swift and vMLX Python compute the same bits on the same inputs.
public enum JANGTFastDecode {
    /// `VMLX_JANGT_FAST=1` (true / on / yes), read per call.
    public static var isEnabled: Bool {
        if let override { return override }
        guard let raw = getenv("VMLX_JANGT_FAST") else { return false }
        return ["1", "true", "on", "yes"].contains(String(cString: raw).trimmingCharacters(in: .whitespaces).lowercased())
    }

    /// Tests: the lever for the current task (nil = the environment's).
    @TaskLocal public static var override: Bool?

    /// Python `supported`: both groups trellis, input widths in whole 512-value blocks (at most 32 per row: one
    /// threadgroup packs a row), output widths a multiple of the 8 rows a threadgroup owns.
    public static func supported(_ t: JANGTRoutedExpertLayer.TrellisTensors) -> Bool {
        let D = t.hiddenSize, I = t.intermediateSize
        return D % 512 == 0 && I % 512 == 0 && D / 512 <= 32 && I / 512 <= 32 && D % 8 == 0 && I % 8 == 0
            && t.gate.outputs % 8 == 0 && t.down.outputs % 8 == 0
    }

    // MARK: generated sources (fast_decode.py, string for string)

    static let unroll = "_Pragma(\"clang loop unroll(full)\")"

    /// Python `_lit`: a float32 constant by its bit pattern.
    static func lit(_ bits: UInt32) -> String { "as_type<float>(\(bits)u)" }

    static func stats(_ k: Double) -> (mu0: UInt32, mu1: UInt32, isd0: UInt32, isd1: UInt32) {
        let s = JANGTCode.stats[k]!
        return (s.0, s.1, s.2, s.3)
    }

    /// Python `_nwin`: words the lane window loads.
    static func nwin(_ k: Double, _ l: Int) -> Int {
        let (_, maxShift, words) = JANGTCode.window(k, l: l)
        let span = JANGTCode.stepOffsets(k, 9)[7] + l
        return maxShift != 0 ? words : (span > 32 ? 2 : 1)
    }

    /// Python `_realign`: a0 / a1 from the window words `w`.
    static func realign(_ k: Double, _ l: Int, _ w: [String]) -> String {
        let (_, maxShift, _) = JANGTCode.window(k, l: l)
        let span = JANGTCode.stepOffsets(k, 9)[7] + l
        if maxShift == 0 {
            return "uint a0 = \(w[0]);" + (span > 32 ? " uint a1 = \(w[1]);" : " uint a1 = 0u;")
        }
        if span <= 32 {
            return "uint a0 = sh == 0u ? \(w[0]) : ((\(w[0]) >> sh) | (\(w[1]) << (32u - sh))); uint a1 = 0u;"
        }
        return "uint a0 = sh == 0u ? \(w[0]) : ((\(w[0]) >> sh) | (\(w[1]) << (32u - sh)));"
            + " uint a1 = sh == 0u ? \(w[1]) : ((\(w[1]) >> sh) | (\(w[2]) << (32u - sh)));"
    }

    /// Python `_extract(o, L, bfe=True)`.
    static func extract(_ o: Int, _ l: Int) -> String {
        if o + l <= 32 { return "extract_bits(a0, \(o)u, \(l)u)" }
        if o >= 32 { return "extract_bits(a1, \(o - 32)u, \(l)u)" }
        return "(((a0 >> \(o)u) | (a1 << \(32 - o)u)) & \((1 << l) - 1)u)"
    }

    /// Python `_tdot(K, L, acc, xh, "mul", True)`.
    static func tdot(_ k: Double, _ l: Int, acc: String, xh: String) -> String {
        let so = JANGTCode.stepOffsets(k, 9)
        let mu = JANGTCode.mult[k]!
        return (0 ..< 8).map { j in
            let p = "(((\(extract(so[j], l))) * \(mu)u) & \(JANGTCode.mask2)u) | \(JANGTCode.orb2)u"
            return "\(acc) = fma(\(xh)[\(j)], as_type<half2>(\(p)), \(acc));"
        }.joined(separator: " ")
    }

    /// Python `_xload4`.
    static let xload4 =
        "const uint4 q0 = xq4[0], q1 = xq4[1];\n"
        + "        half2 xh[8] = {as_type<half2>(q0.x), as_type<half2>(q0.y), as_type<half2>(q0.z), as_type<half2>(q0.w),\n"
        + "                       as_type<half2>(q1.x), as_type<half2>(q1.y), as_type<half2>(q1.z), as_type<half2>(q1.w)};\n"
        + "        const float xi = xinv_p[0]; const float x0 = xi * ISD0, x1 = xi * ISD1;"

    /// Python `_chunk_loads(K, L, "scalar", ...)`.
    static func chunkLoads(_ k: Double, _ l: Int, ptr: String, dst: String, r: Int, w: Int, off: String) -> String {
        (0 ..< nwin(k, l)).map { "\(dst)[\($0)] = \(ptr)[\(r * w) + \(off) + \($0)];" }.joined(separator: " ")
    }

    /// Python `_main_loop` (wload "scalar"): the K-part loop over `nbp` blocks; accumulators `<acc>[r]`.
    static func mainLoop(_ k: Double, _ l: Int, projs: [(String, String)], ptrs: [String: String], o: Int, w: Int,
                         adv: Int) -> String {
        let nw = nwin(k, l)
        func name(_ pn: String, _ r: Int) -> String { "w\(pn)\(r)" }
        func fold(_ acc: String, _ r: Int) -> String {
            "\(acc)[\(r)] = fma(float(a_.x), x0, \(acc)[\(r)]); \(acc)[\(r)] = fma(float(a_.y), x1, \(acc)[\(r)]);"
        }
        let decl = projs.flatMap { pn, _ in (0 ..< o).map { "uint \(name(pn, $0))[\(nw)];" } }.joined(separator: " ")
        let loads = projs.flatMap { pn, _ in
            (0 ..< o).map { chunkLoads(k, l, ptr: ptrs[pn]!, dst: name(pn, $0), r: $0, w: w, off: "0") }
        }.joined(separator: " ")
        var rows: [String] = []
        for r in 0 ..< o {
            for (pn, acc) in projs {
                let words = (0 ..< nw).map { "\(name(pn, r))[\($0)]" }
                rows.append(
                    "{ half2 a_ = half2(0.0h); { \(realign(k, l, words)) \(tdot(k, l, acc: "a_", xh: "xh")) } "
                        + "\(fold(acc, r)) }")
            }
        }
        let body = rows.joined(separator: "\n        ")
        let advAll = projs.map { "\(ptrs[$0.0]!) += \(adv)u;" }.joined(separator: " ")
        return decl + "\n    for (uint b = 0; b < nbp; b++) {\n      " + loads + "\n      { " + xload4 + " " + body
            + " }\n      xq4 += 64u; xinv_p += 32u; " + advAll + "\n    }"
    }

    /// Python `act_clamp_metal(limit, "gg", "uu")`.
    static func clamp(_ limit: Float) -> String {
        guard limit > 0 else { return "" }
        return " gg = metal::min(gg, \(limit)f); uu = metal::clamp(uu, \(-limit)f, \(limit)f);"
    }

    /// Python `act_tag`.
    static func actTag(_ activation: JANGTActivation, limit: Float) -> String {
        let lim = limit > 0 ? "_lim" + "\(limit)".replacingOccurrences(of: ".", with: "p") : ""
        switch activation {
        case .silu: return lim
        case .oai(let alpha):
            return "_oai" + "\(alpha)".replacingOccurrences(of: ".", with: "p").replacingOccurrences(of: "-", with: "m")
                + lim
        }
    }

    static func kname(_ k: Double) -> String {
        k == k.rounded() ? "\(Int(k))" : "\(k)".replacingOccurrences(of: ".", with: "p")
    }

    /// Python `_pack1_kernel(kin, xdt, per_row_signs, guard=True)`.
    static func pack1Source(kin: Int, perRowSigns: Bool) -> String {
        let nb = kin / 512
        let sidx = perRowSigns ? "sidx[r]" : "0u"
        let guardSrc = "float xm = 0.0f; " + "\(unroll) for (uint i = 0; i < 16u; i++) xm = metal::max(xm, metal::abs(xt[i]));"
            + " float xs = xm > 1024.0f ? metal::exp2(-metal::ceil(metal::log2(xm * (1.0f / 1024.0f)))) : 1.0f;"
        return "\n" + [
            "    uint lane = thread_index_in_simdgroup; uint blk = simdgroup_index_in_threadgroup; uint r = threadgroup_position_in_grid.y;",
            "    threadgroup float tv[\(nb) * 512];",
            "    threadgroup float red[\(nb) * 2];",
            "    const device float* sv = s + (size_t)(\(sidx)) * \(kin)u;",
            "    \(unroll) for (uint hb = 0; hb < 4u; hb++) {",
            "      const uint base = blk * 512u + hb * 128u + lane * 4u;",
            "      float w4[4];",
            "      for (uint i = 0; i < 4u; i++) w4[i] = float(x[(size_t)r * \(kin)u + base + i]) * sv[base + i];",
            "      { float a = w4[0], b = w4[1]; w4[0] = a + b; w4[1] = a - b; a = w4[2]; b = w4[3]; w4[2] = a + b; w4[3] = a - b; }",
            "      { float a = w4[0], b = w4[2]; w4[0] = a + b; w4[2] = a - b; a = w4[1]; b = w4[3]; w4[1] = a + b; w4[3] = a - b; }",
            "      for (uint d = 1; d < 32u; d <<= 1) {",
            "        bool upper = (lane & d) != 0u;",
            "        for (uint i = 0; i < 4u; i++) { float o = simd_shuffle_xor(w4[i], d); w4[i] = upper ? (o - w4[i]) : (w4[i] + o); }",
            "      }",
            "      for (uint i = 0; i < 4u; i++) tv[base + i] = w4[i] * 0.08838834764831845f;",
            "    }",
            "    simdgroup_barrier(mem_flags::mem_threadgroup);",
            "    float xt[16];",
            "    \(unroll) for (uint i = 0; i < 16u; i++) xt[i] = tv[blk * 512u + lane * 16u + i];",
            "    \(guardSrc)",
            "    float se = 0.0f, so = 0.0f;",
            "    uint q[8];",
            "    \(unroll) for (uint i = 0; i < 8u; i++) {",
            "      half2 h = half2(half(xt[2u * i] * xs), half(xt[2u * i + 1u] * xs));",
            "      q[i] = as_type<uint>(h); se += xt[2u * i]; so += xt[2u * i + 1u];",
            "    }",
            "    const size_t c = (size_t)r * \(kin / 16)u + blk * 32u + lane;",
            "    \(unroll) for (uint i = 0; i < 8u; i++) xq[c * 8u + i] = q[i];",
            "    xinv[c] = 1.0f / xs;",
            "    se = simd_sum(se); so = simd_sum(so);",
            "    if (lane == 0u) { red[blk * 2u] = se; red[blk * 2u + 1u] = so; }",
            "    threadgroup_barrier(mem_flags::mem_threadgroup);",
            "    if (blk == 0u && lane == 0u) {",
            "      float a = 0.0f, b = 0.0f;",
            "      for (uint k = 0; k < \(nb)u; k++) { a += red[k * 2u]; b += red[k * 2u + 1u]; }",
            "      st[r * 2u] = a; st[r * 2u + 1u] = b;",
            "    }",
            "    ",
        ].joined(separator: "\n")
    }

    /// Outlier-column epilogue of the gate / up kernel (Python `gen_gu4` `outl`).
    static func gateUpOutliers(kin: Int, n: Int, o: Int, outliers: Int) -> String {
        guard outliers > 0 else { return "" }
        return "\n    if (kp == 0u && lane < \(outliers)u) { int c = ocol[(size_t)e * \(outliers)u + lane];"
            + "\n      if (c >= 0) { float xo = float(x[(size_t)t * \(kin)u + uint(c)]);"
            + "\n        \(unroll) for (uint r = 0; r < \(o)u; r++) {"
            + "\n          g[r] += xo * float(woutg[((size_t)e * \(n)u + row0 + r) * \(outliers)u + lane]) / sg[(size_t)e * \(n)u + row0 + r];"
            + "\n          u[r] += xo * float(woutu[((size_t)e * \(n)u + row0 + r) * \(outliers)u + lane]) / su_[(size_t)e * \(n)u + row0 + r]; } } }"
    }

    /// Outlier-column epilogue of the down kernels (Python `gen_dn4` / `gen_dn4c` `outl`; `kpGuard` = dn4's).
    static func downOutliers(kin: Int, n: Int, o: Int, outliers: Int, kpGuard: Bool) -> String {
        guard outliers > 0 else { return "" }
        return "\n    if (\(kpGuard ? "kp == 0u && " : "")lane < \(outliers)u) { int c = ocol[(size_t)e * \(outliers)u + lane];"
            + "\n      if (c >= 0) { float ho = float(h[(size_t)ts * \(kin)u + uint(c)]);"
            + "\n        \(unroll) for (uint r = 0; r < \(o)u; r++) d[r] += ho * float(wout[((size_t)e * \(n)u + row0 + r) * \(outliers)u + lane]) / sd[(size_t)e * \(n)u + row0 + r]; } }"
    }

    /// Python `gen_gu4(..., O=4, NOG=2, KSPLIT=1, wload="scalar")`; template `KS`.
    static func gateUpSource(kin: Int, n: Int, k: Double, l: Int, outliers: Int, words: Int, odt: String, limit: Float,
                             activation: JANGTActivation) -> String {
        let (o, nog) = (4, 2)
        let (mu0, mu1, isd0, isd1) = stats(k)
        let pl = JANGTCode.laneBits(k)
        let nb = kin / 512
        let loop = mainLoop(k, l, projs: [("g", "g"), ("u", "u")], ptrs: ["g": "wp", "u": "up"], o: o, w: words, adv: pl)
        let outl = gateUpOutliers(kin: kin, n: n, o: o, outliers: outliers)
        let reduce = "float G[\(o)], U[\(o)]; \(unroll) for (uint r = 0; r < \(o)u; r++) { G[r] = simd_sum(g[r]); U[r] = simd_sum(u[r]); }"
        let value = activation.metalValue(g: "gg", u: "uu")
        return "\n" + [
            "    const float ISD0 = \(lit(isd0)), ISD1 = \(lit(isd1));",
            "    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;",
            "    uint og = sgi % \(nog)u, kp = sgi / \(nog)u;",
            "    uint row0 = threadgroup_position_in_grid.z * \(nog * o)u + og * \(o)u; uint ts = threadgroup_position_in_grid.y;",
            "    uint t = ts / KS; uint e = idx[ts];",
            "    const uint nbp = \(nb)u + (kp < 0u ? 1u : 0u);",
            "    const uint b0 = kp * \(nb)u + metal::min(kp, 0u);",
            "    const uint lb = lane * \(pl)u; const uint sh = lb & 31u; (void)sh;",
            "    const device uint32_t* wp = wg + ((size_t)e * \(n)u + row0) * \(words)u + (lb >> 5) + b0 * \(pl)u;",
            "    const device uint32_t* up = wu + ((size_t)e * \(n)u + row0) * \(words)u + (lb >> 5) + b0 * \(pl)u;",
            "    const device uint4* xq4 = (const device uint4*)(xq + (size_t)t * \(kin / 2)u) + lane * 2u + b0 * 64u;",
            "    const device float* xinv_p = xinv + (size_t)t * \(kin / 16)u + lane + b0 * 32u;",
            "    float g[\(o)], u[\(o)];",
            "    \(unroll) for (uint r = 0; r < \(o)u; r++) { g[r] = 0.0f; u[r] = 0.0f; }",
            "    \(loop)",
            "    \(outl)",
            "    \(reduce)",
            "    const float SE = st[t * 2u], SO = st[t * 2u + 1u];",
            "    const float cst = \(lit(mu0)) * ISD0 * SE + \(lit(mu1)) * ISD1 * SO;",
            "    \(unroll) for (uint r = 0; r < \(o)u; r++) {",
            "      if (lane == 0u) {",
            "        float gg = (G[r] - cst) * sg[(size_t)e * \(n)u + row0 + r];",
            "        float uu = (U[r] - cst) * su_[(size_t)e * \(n)u + row0 + r];",
            "        \(clamp(limit))",
            "        out[(size_t)ts * \(n)u + row0 + r] = \(odt)(\(value));",
            "      }",
            "    }",
            "    ",
        ].joined(separator: "\n")
    }

    /// Python `gen_dn4(..., O=4, NOG=2, KSPLIT=1, wload="scalar")`: per-slot down y (before the router weight).
    static func downSlotsSource(kin: Int, n: Int, k: Double, l: Int, outliers: Int, words: Int) -> String {
        let (o, nog) = (4, 2)
        let (mu0, mu1, isd0, isd1) = stats(k)
        let pl = JANGTCode.laneBits(k)
        let nb = kin / 512
        let loop = mainLoop(k, l, projs: [("d", "d")], ptrs: ["d": "wp"], o: o, w: words, adv: pl)
        let outl = downOutliers(kin: kin, n: n, o: o, outliers: outliers, kpGuard: true)
        let reduce = "float Dv[\(o)]; \(unroll) for (uint r = 0; r < \(o)u; r++) Dv[r] = simd_sum(d[r]);"
        return "\n" + [
            "    const float ISD0 = \(lit(isd0)), ISD1 = \(lit(isd1));",
            "    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;",
            "    uint og = sgi % \(nog)u, kp = sgi / \(nog)u;",
            "    uint row0 = threadgroup_position_in_grid.z * \(nog * o)u + og * \(o)u; uint ts = threadgroup_position_in_grid.y;",
            "    uint e = idx[ts];",
            "    const uint nbp = \(nb)u + (kp < 0u ? 1u : 0u);",
            "    const uint b0 = kp * \(nb)u + metal::min(kp, 0u);",
            "    const uint lb = lane * \(pl)u; const uint sh = lb & 31u; (void)sh;",
            "    const device uint32_t* wp = wd + ((size_t)e * \(n)u + row0) * \(words)u + (lb >> 5) + b0 * \(pl)u;",
            "    const device uint4* xq4 = (const device uint4*)(xq + (size_t)ts * \(kin / 2)u) + lane * 2u + b0 * 64u;",
            "    const device float* xinv_p = xinv + (size_t)ts * \(kin / 16)u + lane + b0 * 32u;",
            "    float d[\(o)];",
            "    \(unroll) for (uint r = 0; r < \(o)u; r++) d[r] = 0.0f;",
            "    \(loop)",
            "    \(outl)",
            "    \(reduce)",
            "    const float cst = \(lit(mu0)) * ISD0 * st[ts * 2u] + \(lit(mu1)) * ISD1 * st[ts * 2u + 1u];",
            "    \(unroll) for (uint r = 0; r < \(o)u; r++) if (lane == 0u) out[(size_t)ts * \(n)u + row0 + r] = (Dv[r] - cst) * sd[(size_t)e * \(n)u + row0 + r];",
            "    ",
        ].joined(separator: "\n")
    }

    /// Python `gen_dn4c(..., O=4, NOG=2, KS=slots, wload="scalar")`: down + router-weighted slot combine, one launch.
    static func downCombinedSource(kin: Int, n: Int, k: Double, l: Int, outliers: Int, words: Int, slots: Int) -> String {
        let (o, nog) = (4, 2)
        let (mu0, mu1, isd0, isd1) = stats(k)
        let pl = JANGTCode.laneBits(k)
        let nb = kin / 512
        let loop = mainLoop(k, l, projs: [("d", "d")], ptrs: ["d": "wp"], o: o, w: words, adv: pl)
        let outl = downOutliers(kin: kin, n: n, o: o, outliers: outliers, kpGuard: false)
        return "\n" + [
            "    const float ISD0 = \(lit(isd0)), ISD1 = \(lit(isd1));",
            "    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;",
            "    uint og = sgi % \(nog)u, sl = sgi / \(nog)u;",
            "    uint row0 = threadgroup_position_in_grid.z * \(nog * o)u + og * \(o)u; uint t = threadgroup_position_in_grid.y;",
            "    uint ts = t * \(slots)u + sl; uint e = idx[ts];",
            "    const uint nbp = \(nb)u;",
            "    threadgroup float ysl[\(slots) * \(nog) * \(o)];",
            "    const uint lb = lane * \(pl)u; const uint sh = lb & 31u; (void)sh;",
            "    const device uint32_t* wp = wd + ((size_t)e * \(n)u + row0) * \(words)u + (lb >> 5);",
            "    const device uint4* xq4 = (const device uint4*)(xq + (size_t)ts * \(kin / 2)u) + lane * 2u;",
            "    const device float* xinv_p = xinv + (size_t)ts * \(kin / 16)u + lane;",
            "    float d[\(o)];",
            "    \(unroll) for (uint r = 0; r < \(o)u; r++) d[r] = 0.0f;",
            "    \(loop)",
            "    \(outl)",
            "    const float cst = \(lit(mu0)) * ISD0 * st[ts * 2u] + \(lit(mu1)) * ISD1 * st[ts * 2u + 1u];",
            "    \(unroll) for (uint r = 0; r < \(o)u; r++) {",
            "      float Dv = simd_sum(d[r]);",
            "      if (lane == 0u) ysl[(sl * \(nog)u + og) * \(o)u + r] = (Dv - cst) * sd[(size_t)e * \(n)u + row0 + r];",
            "    }",
            "    threadgroup_barrier(mem_flags::mem_threadgroup);",
            "    if (sl == 0u && lane < \(o)u) {",
            "      float acc = 0.0f;",
            "      for (uint q = 0; q < \(slots)u; q++) acc = fma(float(rw[t * \(slots)u + q]), ysl[(q * \(nog)u + og) * \(o)u + lane], acc);",
            "      out[(size_t)t * \(n)u + row0 + lane] = acc;",
            "    }",
            "    ",
        ].joined(separator: "\n")
    }

    /// Python `_combine4(N, kslots)`: the slot-ordered router-weighted sum of per-slot rows.
    static func combineSource(n: Int, slots: Int) -> String {
        "\n" + [
            "    uint n = thread_position_in_grid.x, t = thread_position_in_grid.y;",
            "    if (n >= \(n)u) return;",
            "    float acc = 0.0f;",
            "    for (uint sl = 0; sl < \(slots)u; sl++) { uint ts = t * \(slots)u + sl; acc = fma(float(rw[ts]), ysl[(size_t)ts * \(n)u + n], acc); }",
            "    out[(size_t)t * \(n)u + n] = acc;",
            "    ",
        ].joined(separator: "\n")
    }

    // MARK: kernels (cached by Python's names)

    struct Kernel {
        let name: String
        let inputs: [String]
        let outputs: [String]
        let source: String
    }

    static func pack1Kernel(kin: Int, xdt: String, perRowSigns: Bool) -> Kernel {
        Kernel(
            name: "jtf_pack1_k\(kin)_\(xdt)_r\(perRowSigns ? 1 : 0)_g1",
            inputs: ["x", "s"] + (perRowSigns ? ["sidx"] : []), outputs: ["xq", "xinv", "st"],
            source: pack1Source(kin: kin, perRowSigns: perRowSigns))
    }

    static func gateUpKernel(kin: Int, n: Int, k: Double, outliers: Int, words: Int, xdt: String, odt: String,
                             limit: Float, activation: JANGTActivation) -> Kernel {
        Kernel(
            name: "jtf4_gu_k\(kin)_n\(n)_K\(kname(k))_o\(outliers)_w\(words)_\(xdt)_\(odt)"
                + "\(actTag(activation, limit: limit))_O4_G2_S1_scalar",
            inputs: ["xq", "xinv", "st", "x", "idx", "wg", "sg", "wu", "su_"]
                + (outliers > 0 ? ["ocol", "woutg", "woutu"] : []),
            outputs: ["out"],
            source: gateUpSource(
                kin: kin, n: n, k: k, l: JANGTCode.stateBits, outliers: outliers, words: words, odt: odt,
                limit: limit, activation: activation))
    }

    static func downSlotsKernel(kin: Int, n: Int, k: Double, outliers: Int, words: Int, hdt: String) -> Kernel {
        Kernel(
            name: "jtf4_dn_k\(kin)_n\(n)_K\(kname(k))_o\(outliers)_w\(words)_\(hdt)_O4_G2_S1_scalar",
            inputs: ["xq", "xinv", "st", "h", "idx", "wd", "sd"] + (outliers > 0 ? ["ocol", "wout"] : []),
            outputs: ["out"],
            source: downSlotsSource(kin: kin, n: n, k: k, l: JANGTCode.stateBits, outliers: outliers, words: words))
    }

    static func downCombinedKernel(kin: Int, n: Int, k: Double, outliers: Int, words: Int, hdt: String, slots: Int)
        -> Kernel
    {
        Kernel(
            name: "jtf4c_dn_k\(kin)_n\(n)_K\(kname(k))_o\(outliers)_w\(words)_\(hdt)_O4_G2_s\(slots)_scalar",
            inputs: ["xq", "xinv", "st", "h", "idx", "rw", "wd", "sd"] + (outliers > 0 ? ["ocol", "wout"] : []),
            outputs: ["out"],
            source: downCombinedSource(
                kin: kin, n: n, k: k, l: JANGTCode.stateBits, outliers: outliers, words: words, slots: slots))
    }

    static func combineKernel(n: Int, slots: Int) -> Kernel {
        Kernel(
            name: "jtf4_comb_n\(n)_s\(slots)", inputs: ["ysl", "rw"], outputs: ["out"],
            source: combineSource(n: n, slots: slots))
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var compiled: [String: MLXFast.MLXFastKernel] = [:]

    static func compile(_ k: @autoclosure () -> Kernel, name: String) -> MLXFast.MLXFastKernel {
        if let c = lock.withLock({ compiled[name] }) { return c }
        let kernel = k()
        let c = MLXFast.metalKernel(
            name: kernel.name, inputNames: kernel.inputs, outputNames: kernel.outputs, source: kernel.source)
        lock.withLock { compiled[name] = c }
        return c
    }

    // MARK: launches (fast_decode.py)

    /// Python `pack_rows1`: x (R, kin) -> (xq (R, kin / 2) u32, xinv (R, kin / 16) f32, st (R, 2) f32), x * s rotated
    /// (s (kin,) shared, or (E, kin) selected per row by `rowExperts`).
    public static func packRows1(_ x: MLXArray, signs s: MLXArray, rowExperts: MLXArray? = nil)
        -> (MLXArray, MLXArray, MLXArray)
    {
        let rows = x.dim(0), kin = x.dim(1)
        precondition(kin % 512 == 0 && kin / 512 <= 32, "JANGT fast pack: width \(kin)")
        let nb = kin / 512
        let xdt = JANGTKernels.metalType(x.dtype)
        let perRow = rowExperts != nil
        let name = "jtf_pack1_k\(kin)_\(xdt)_r\(perRow ? 1 : 0)_g1"
        let kernel = compile(pack1Kernel(kin: kin, xdt: xdt, perRowSigns: perRow), name: name)
        var inputs = [x, s]
        if let rowExperts { inputs.append(rowExperts.asType(.uint32)) }
        let out = kernel(
            inputs, grid: (32 * nb, rows, 1), threadGroup: (32 * nb, 1, 1),
            outputShapes: [[rows, kin / 2], [rows, kin / 16], [rows, 2]], outputDTypes: [.uint32, .float32, .float32])
        return (out[0], out[1], out[2])
    }

    /// Python `routed_gate_up`: (T * k, I) f32 SwiGLU activations of every (row, slot).
    public static func gateUp(_ x2: MLXArray, ids: MLXArray, _ t: JANGTRoutedExpertLayer.TrellisTensors) -> MLXArray {
        let (xq, xinv, st) = packRows1(x2, signs: t.gateUpSigns)
        let rows = x2.dim(0), kin = x2.dim(1), slots = ids.dim(1)
        let n = t.gate.outputs, words = t.gate.words, outliers = t.gate.outlierColumns
        let xdt = JANGTKernels.metalType(x2.dtype)
        let name = "jtf4_gu_k\(kin)_n\(n)_K\(kname(t.gate.k))_o\(outliers)_w\(words)_\(xdt)_float"
            + "\(actTag(t.activation, limit: t.limit))_O4_G2_S1_scalar"
        let kernel = compile(
            gateUpKernel(
                kin: kin, n: n, k: t.gate.k, outliers: outliers, words: words, xdt: xdt, odt: "float", limit: t.limit,
                activation: t.activation), name: name)
        var inputs: [MLXArray] = [
            xq, xinv, st, x2, ids.flattened().asType(.uint32), t.gate.packed, t.gate.scale, t.up.packed, t.up.scale,
        ]
        if outliers > 0 { inputs += [t.gate.outlierColumnIndices, t.gate.outlierWeights, t.up.outlierWeights] }
        return kernel(
            inputs, template: [("KS", slots)], grid: (64, rows * slots, n / 8), threadGroup: (64, 1, 1),
            outputShapes: [[rows * slots, n]], outputDTypes: [.float32])[0]
    }

    /// Python `routed_down`: (T, D) f32 router-weighted down projection of every row (one row: the in-kernel slot
    /// combine; 2+ rows: per-slot kernel + combine kernel; a row's arithmetic is the same in both).
    public static func down(_ h: MLXArray, ids: MLXArray, weights: MLXArray,
                            _ t: JANGTRoutedExpertLayer.TrellisTensors) -> MLXArray {
        let (xq, xinv, st) = packRows1(h, signs: t.downSigns, rowExperts: ids.flattened())
        let rows = ids.dim(0), slots = ids.dim(1), kin = h.dim(1)
        let d = t.down
        let n = d.outputs, words = d.words, outliers = d.outlierColumns
        let hdt = JANGTKernels.metalType(h.dtype)
        let idx = ids.flattened().asType(.uint32)
        let rw = weights.flattened().asType(.float32)
        if rows == 1 {
            let name = "jtf4c_dn_k\(kin)_n\(n)_K\(kname(d.k))_o\(outliers)_w\(words)_\(hdt)_O4_G2_s\(slots)_scalar"
            let kernel = compile(
                downCombinedKernel(kin: kin, n: n, k: d.k, outliers: outliers, words: words, hdt: hdt, slots: slots),
                name: name)
            var inputs: [MLXArray] = [xq, xinv, st, h, idx, rw, d.packed, d.scale]
            if outliers > 0 { inputs += [d.outlierColumnIndices, d.outlierWeights] }
            let tpg = 32 * 2 * slots
            return kernel(
                inputs, grid: (tpg, rows, n / 8), threadGroup: (tpg, 1, 1), outputShapes: [[rows, n]],
                outputDTypes: [.float32])[0]
        }
        let name = "jtf4_dn_k\(kin)_n\(n)_K\(kname(d.k))_o\(outliers)_w\(words)_\(hdt)_O4_G2_S1_scalar"
        let kernel = compile(
            downSlotsKernel(kin: kin, n: n, k: d.k, outliers: outliers, words: words, hdt: hdt), name: name)
        var inputs: [MLXArray] = [xq, xinv, st, h, idx, d.packed, d.scale]
        if outliers > 0 { inputs += [d.outlierColumnIndices, d.outlierWeights] }
        let ysl = kernel(
            inputs, grid: (64, rows * slots, n / 8), threadGroup: (64, 1, 1), outputShapes: [[rows * slots, n]],
            outputDTypes: [.float32])[0]
        let tg = min(256, n)
        let combine = compile(combineKernel(n: n, slots: slots), name: "jtf4_comb_n\(n)_s\(slots)")
        return combine(
            [ysl, rw], grid: ((n + tg - 1) / tg * tg, rows, 1), threadGroup: (tg, 1, 1), outputShapes: [[rows, n]],
            outputDTypes: [.float32])[0]
    }

    /// Python `routed_fast`: the routed block for decode / verify rows, (T, D) f32.
    public static func routed(_ x2: MLXArray, ids: MLXArray, weights: MLXArray,
                              _ t: JANGTRoutedExpertLayer.TrellisTensors) -> MLXArray {
        down(gateUp(x2, ids: ids, t), ids: ids, weights: weights, t)
    }
}
