// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast

/// JANGT v1 trellis-coded routed experts ("JANGHT" bundles: GLM-5.3-Flash JANGHT2.4, Naive-N0.5-Flash JANGHT2.x).
///
/// Port of vMLX Python `vmlx_engine/jangt/{v2core,v2l,moe_kernels}.py` (decode set, "v3" kernels), kept
/// line-for-line in the generated Metal so both engines compute the same thing:
///
/// - Code (`v2_halfbits`, state length L = 12): each STATE s (12 bits) yields TWO weights. h = s * MULT (mod 2^32),
///   p = (h & MASK2) | ORB2, the two fp16 halves of p are the raw values; value = (raw - mu_i) * isd_i per half
///   (mu/isd = mean and 1/std of the raw halves over all 2^L states, per bit-width K).
/// - Stream (per output row, LSB first): state_0 is the first L bits; each later step shifts in kpat(K)[t & 1] new
///   bits at the top (K=1: 2, K=1.5: alternating 4/2, K=2: 4, K=2.5: alternating 6/4, K=3: 6),
///   state_t = (S >> offset_t) & (2^L - 1).
/// - A lane owns 16 consecutive weights = 8 steps; one 512-value block = 32 lanes. Offsets (mu terms) leave the inner
///   loop: y = ((Σ x·raw_even - mu0·Σx_even)·isd0 + (Σ x·raw_odd - mu1·Σx_odd)·isd1) · row_scale.
/// - Input widths that are not a multiple of 512 (Qwen3.8-Flash-Next down: I = 640) run the full blocks unmasked and
///   then ONE lane-masked tail block (Python `_blocks`, 2026-10-16). Before that, the k0 < kin loop read a whole
///   512-value block past the row at kin % 512 != 0, for every width.
/// - Input basis: x · su (per-LAYER signs, 0 on outlier columns) then blockwise Hadamard-128. Outlier columns are
///   dense fp16 weights added in the epilogue from the RAW input.
/// - down_proj adds per-EXPERT column scales `cs`, folded into its input signs: sv = cs · su · mask.
public enum JANGTCode {
    public static let stateBits = 12
    /// K=1 / K=1.5 (Qwen3.8-Flash-Next JANGHT, 2026-10-16): the encoder campaign's code search
    /// (`qwen38fn-janght-spark-2026-10-15/scripts/lowbit/jt_patch.py`, vMLX Python `jangt/v2l.py` MULTS). Must stay
    /// identical to the encoder registration; bundles also declare them in `jangt.code_multipliers`.
    static let mult: [Double: UInt32] = [
        1: 590_428_815, 1.5: 2_124_823_417, 2: 636_231_159, 2.5: 636_231_159, 3: 668_265_263,
    ]
    static let mask: UInt32 = 36863
    static let orb: UInt32 = 12288
    static var mask2: UInt32 { mask | (mask << 16) }
    static var orb2: UInt32 { orb | (orb << 16) }

    /// Exact float32 code statistics (mu0, mu1, isd0, isd1) at L = 12, from numpy's float32 mean/std over the 4096
    /// states (the values Python bakes into its kernels). `JANGTCodeTests` recomputes them.
    static let stats: [Double: (UInt32, UInt32, UInt32, UInt32)] = [
        1: (0x3a2c_8c00, 0x3a08_5a00, 0x3f91_77ed, 0x3f91_0cbe),
        1.5: (0xb90b_4000, 0x39f2_bc00, 0x3f91_77e4, 0x3f90_89d8),
        2: (0xbb03_e600, 0xbc3b_eac0, 0x3f91_7802, 0x3f91_98ae),
        2.5: (0xbb03_e600, 0xbc3b_eac0, 0x3f91_7802, 0x3f91_98ae),
        3: (0xb824_0000, 0x3a99_9600, 0x3f91_77e4, 0x3f91_553a),
    ]

    public static func isSupported(_ k: Double) -> Bool { mult[k] != nil }

    /// (mu0, mu1, isd0, isd1) of width `k` as float32 (the values `stats` holds as bit patterns), nil when unsupported.
    public static func codeStats(_ k: Double) -> (Float, Float, Float, Float)? {
        guard let s = stats[k] else { return nil }
        return (Float(bitPattern: s.0), Float(bitPattern: s.1), Float(bitPattern: s.2), Float(bitPattern: s.3))
    }

    /// New bits entering at transition t >= 1: (even t, odd t). Integer K: (2K, 2K); K=2.5: (6, 4) = 10 bits per 4
    /// weights; K=1.5: (4, 2) = 6 bits per 4 weights. A lane's first step is always even (k0/2 + 8 * lane), so every
    /// 8-step lane segment starts on the same phase.
    static func kpat(_ k: Double) -> (Int, Int) {
        if k == 2.5 { return (6, 4) }
        if k == 1.5 { return (4, 2) }
        precondition(k == k.rounded(), "JANGT: unsupported code width \(k)")
        return (2 * Int(k), 2 * Int(k))
    }

    /// The bundle's declared `jangt.code_multipliers` / `jangt.step_bits` (string width keys) must agree with this
    /// runtime's code for every width they name: decoding with another multiplier or step pattern is garbage, not a
    /// degraded model. Absent tables are accepted (GLM-5.3 / N0.5 bundles predate them).
    static func validateDeclaredCode(_ header: [String: Any]) -> String? {
        if let declared = header["code_multipliers"] {
            guard let table = declared as? [String: Any] else { return "invalid jangt.code_multipliers" }
            for (key, value) in table {
                guard let k = Double(key), let m = mult[k], let v = (value as? NSNumber)?.int64Value,
                      v == Int64(m)
                else { return "jangt.code_multipliers[\(key)] does not match the runtime code" }
            }
        }
        if let declared = header["step_bits"] {
            guard let table = declared as? [String: Any] else { return "invalid jangt.step_bits" }
            for (key, value) in table {
                guard let k = Double(key), isSupported(k), let pair = value as? [NSNumber], pair.count == 2,
                      pair[0].intValue == kpat(k).0, pair[1].intValue == kpat(k).1
                else { return "jangt.step_bits[\(key)] does not match the runtime code" }
            }
        }
        return nil
    }

    /// Bit offset of state t in the row stream (state 0 at 0), t in 0..<count.
    static func stepOffsets(_ k: Double, _ count: Int) -> [Int] {
        let (p0, p1) = kpat(k)
        var out = [0]
        var acc = 0
        if count > 1 {
            for t in 1 ..< count {
                acc += (t & 1) == 1 ? p1 : p0
                out.append(acc)
            }
        }
        return out
    }

    public static func wordsPerRow(inputs: Int, k: Double, l: Int = stateBits) -> Int {
        if k == k.rounded() {
            let kk = Int(k)
            return (kk * inputs + (l - 2 * kk) + 31) / 32 + 1
        }
        let bits = stepOffsets(k, inputs / 2).last! + l
        return (bits + 31) / 32 + 1
    }

    /// Bits per lane per 512-value block (= words a lane's pointer advances per block): 16 (K=1), 24 (K=1.5),
    /// 32 (K=2), 40 (K=2.5), 48 (K=3).
    static func laneBits(_ k: Double) -> Int { stepOffsets(k, 17)[16] / 2 }

    /// (bits per lane PL, largest sub-word start shift over the 32 lanes, words the lane window must load). Python
    /// `moe_kernels._window`. Lane j of a 512-value block starts at bit j * PL of the block (blocks start on whole
    /// words: 32 * PL bits); its 8 states end so[7] + L bits past the lane start:
    ///   K=2   PL=32  sh=0          -> 2 words, static shifts (a0/a1)
    ///   K=3   PL=48  sh in {0,16}  -> 3 words realigned into a0/a1 (so[7] + L = 54 > 32)
    ///   K=2.5 PL=40  sh in 0..24   -> 3 words realigned into a0/a1 (46 > 32)
    ///   K=1.5 PL=24  sh in 0..24   -> 2 words realigned into a0 only (so[7] + L = 20 + 12 = 32)
    ///   K=1   PL=16  sh in {0,16}  -> 2 words realigned into a0 only (26 bits)
    /// Loading only the words the window needs keeps the last lane of a row inside words_per_row (one spare word).
    static func window(_ k: Double, l: Int = stateBits) -> (pl: Int, maxShift: Int, words: Int) {
        let pl = laneBits(k)
        let span = stepOffsets(k, 9)[7] + l
        let maxShift = (0 ..< 32).map { ($0 * pl) & 31 }.max()!
        return (pl, maxShift, (maxShift + span + 31) / 32)
    }

    static func floatLiteral(_ bits: UInt32) -> String {
        String(format: "%.9e", Float(bitPattern: bits)) + "f"
    }

    /// Decoded (normalized, unscaled) value pair of one state, the exact float math of the kernels.
    public static func values(state s: UInt32, k: Double) -> (Float, Float) {
        let p = ((s &* mult[k]!) & mask2) | orb2
        let lo = Float(Float16(bitPattern: UInt16(p & 0xFFFF)))
        let hi = Float(Float16(bitPattern: UInt16(p >> 16)))
        let st = stats[k]!
        return ((lo - Float(bitPattern: st.0)) * Float(bitPattern: st.2),
                (hi - Float(bitPattern: st.1)) * Float(bitPattern: st.3))
    }
}

/// The SwiGLU variant a routed stack applies to its gate pre-activation g and up pre-activation u. It belongs to the
/// model, never to the bundle (vMLX Python `moe_kernels.ACTS`). With a limit L > 0 both variants first clamp
/// g' = min(g, L) (from above only) and u' = clip(u, -L, L); with L = 0, g' = g and u' = u.
public enum JANGTActivation: Sendable, Equatable {
    /// h = silu(g') * u'. GLM-5.3 and DeepSeek-V4.1 (with a limit), Naive-N0.5 and MiMo (without).
    case silu
    /// SwiGLU-OAI: h = g' * sigmoid(alpha * g') * (u' + 1). MiniMax-M3 / M3.1 (alpha 1.702, limit 7).
    case oai(alpha: Float)
    /// Inkling-2 (`inkling2` / `inkling2_loop`): h = softcap(g, gateCap) * sigmoid(g) * softcap(u, upCap) with
    /// softcap(x, c) = tanh(x / c) * c; the sigmoid sees the RAW gate. Inkling-2-Mini: caps 4 and 25, no limit.
    case inkling(gateCap: Float, upCap: Float)
    /// ERNIE W4A8 rounds each expert projection to BF16 before SiTU.
    case ernie55(gateCap: Float, upCap: Float)

    /// Metal float expression of h from the (already clamped) `g` / `u`. `.silu` is the exact expression the kernels
    /// carried before the variants existed, so their generated source is unchanged; `.oai` writes alpha by its float32
    /// bit pattern (the value MLX multiplies a float32 array by).
    func metalValue(g: String = "g", u: String = "u") -> String {
        switch self {
        case .silu:
            return "(\(g) / (1.0f + metal::fast::exp(-\(g)))) * \(u)"
        case .oai(let alpha):
            return "(\(g) / (1.0f + metal::fast::exp(-(as_type<float>(\(alpha.bitPattern)u) * \(g))))) * (\(u) + 1.0f)"
        case .ernie55(let gc, let uc):
            return JANGTActivation.inkling(gateCap: gc, upCap: uc).metalValue(
                g: "float(bfloat16_t(\(g)))", u: "float(bfloat16_t(\(u)))")
        case .inkling(let gc, let uc):
            let gcs = "as_type<float>(\(gc.bitPattern)u)", ucs = "as_type<float>(\(uc.bitPattern)u)"
            return "(metal::precise::tanh(\(g) / \(gcs)) * \(gcs)) * (1.0f / (1.0f + metal::precise::exp(-\(g))))"
                + " * (metal::precise::tanh(\(u) / \(ucs)) * \(ucs))"
        }
    }

    /// Kernel-name suffix: empty for `.silu` (existing names unchanged), else the variant and alpha's bit pattern.
    /// MLX caches compiled kernels by name, so every generated source needs its own.
    var nameTag: String {
        switch self {
        case .silu: return ""
        case .oai(let alpha): return "_oai\(String(alpha.bitPattern, radix: 16))"
        case .ernie55(let gc, let uc):
            return "_ernie55\(String(gc.bitPattern, radix: 16))_\(String(uc.bitPattern, radix: 16))"
        case .inkling(let gc, let uc):
            return "_ink\(String(gc.bitPattern, radix: 16))_\(String(uc.bitPattern, radix: 16))"
        }
    }

    /// The same activation in MLX ops (unfused fallbacks), in the inputs' dtype (Python `moe_kernels.act_mlx`).
    public func callAsFunction(gate: MLXArray, up: MLXArray, limit: Float) -> MLXArray {
        var g = gate, u = up
        if limit > 0 {
            g = minimum(g, limit)
            u = clip(u, min: -limit, max: limit)
        }
        switch self {
        case .silu: return g * sigmoid(g) * u
        case .oai(let alpha): return g * sigmoid(alpha * g) * (u + 1)
        case .ernie55(let gc, let uc):
            return JANGTActivation.inkling(gateCap: gc, upCap: uc)(
                gate: g.asType(.bfloat16).asType(.float32), up: u.asType(.bfloat16).asType(.float32), limit: 0)
        case .inkling(let gc, let uc): return tanh(g / gc) * gc * sigmoid(g) * (tanh(u / uc) * uc)
        }
    }
}

/// Generated Metal for the JANGT decode set (Python `moe_kernels.py` v3 + `rotate_rows`).
enum JANGTKernelSource {
    static let unroll = "_Pragma(\"clang loop unroll(full)\")"

    /// Metal statements loading the lane window words `word(i)` and realigning them by the lane's runtime sub-word
    /// shift `sh` (must be in scope) into a0 (and a1 when a state crosses the first realigned word). Python
    /// `_window_load`: one rule for every width, derived from the lane geometry.
    static func windowLoad(k: Double, l: Int, word: (Int) -> String) -> String {
        let (_, maxShift, words) = JANGTCode.window(k, l: l)
        let span = JANGTCode.stepOffsets(k, 9)[7] + l
        if maxShift == 0 {   // K=2: lane segments are whole words, shifts are static
            return "uint a0 = \(word(0));" + (span > 32 ? " uint a1 = \(word(1));" : "")
        }
        if span <= 32 {      // K=1, K=1.5: one realigned word holds the 8 states
            precondition(words == 2, "JANGT window K=\(k)")
            return "uint w0 = \(word(0)), w1 = \(word(1)); "
                + "uint a0 = sh == 0u ? w0 : ((w0 >> sh) | (w1 << (32u - sh)));"
        }
        precondition(span <= 64 && words <= 3, "JANGT window K=\(k)")
        return "uint w0 = \(word(0)), w1 = \(word(1)), w2 = \(word(2)); "
            + "uint a0 = sh == 0u ? w0 : ((w0 >> sh) | (w1 << (32u - sh))); "
            + "uint a1 = sh == 0u ? w1 : ((w1 >> sh) | (w2 << (32u - sh)));"
    }

    /// Expression for the state at static offset `o` of the realigned window a0 (a1). Python `_step_extract`.
    static func stepExtract(_ o: Int, l: Int) -> String {
        precondition(o + l <= 64)
        if o + l <= 32 { return o == 0 ? "a0" : "(a0 >> \(o)u)" }
        if o >= 32 { return o == 32 ? "a1" : "(a1 >> \(o - 32)u)" }
        return "((a0 >> \(o)u) | (a1 << \(32 - o)u))"
    }

    /// Trellis dot of one row's lane chunk into half2 `acc` (x in `xh[8]`). Python `_tdot3`.
    static func tdot(k: Double, l: Int, row: String, acc: String) -> String {
        let so = JANGTCode.stepOffsets(k, 9)
        let mk = (1 << l) - 1
        let mu = JANGTCode.mult[k]!
        let load = windowLoad(k: k, l: l) { "\(row)[\($0)]" }
        var steps: [String] = []
        for j in 0 ..< 8 {
            steps.append(
                "{ uint s = (\(stepExtract(so[j], l: l))) & \(mk)u; uint p = ((s * \(mu)u) & \(JANGTCode.mask2)u) | \(JANGTCode.orb2)u; "
                    + "\(acc) = fma(xh[\(j)], as_type<half2>(p), \(acc)); }")
        }
        return "{ " + load + " " + steps.joined(separator: " ") + " }"
    }

    /// The 512-value block loop of a row (Python `_blocks`): full blocks unmasked, then — when kin % 512 != 0, e.g.
    /// Qwen3.8-Flash-Next down with I = 640 — ONE tail block in which only lanes j < (kin % 512) / 16 decode. A lane
    /// never straddles the tail (16 weights per lane, kin % 16 == 0) and the stream is continuous, so tail lane j sits
    /// at the same bit offset j * PL of its block as in a full block. Masked lanes add nothing to se/so or the
    /// accumulators; the simd_sum epilogue stays outside the divergent branch.
    static func blocks(kin: Int, body: String, advance: String) -> String {
        precondition(kin % 16 == 0, "JANGT decode: input width \(kin) is not a multiple of 16")
        let full = kin / 512 * 512, tail = kin % 512
        var src = """
            for (uint k0 = 0; k0 < \(full)u; k0 += 512u) {
                  \(body)
                  \(advance)
                }
            """
        if tail > 0 {
            src += """

                    if (lane < \(tail / 16)u) {
                      \(body)
                    }
                """
        }
        return src
    }

    /// 16 inputs of this lane -> half2 xh[8] with the fp16 range guard (Python `_xload3(guard=True)`).
    static let xload = """
        float xt[16];
              \(unroll) for (uint i = 0; i < 16u; i++) xt[i] = xp[i];
              float xm = 0.0f;
              \(unroll) for (uint i = 0; i < 16u; i++) xm = metal::max(xm, metal::abs(xt[i]));
              float xs = xm > 1024.0f ? metal::exp2(-metal::ceil(metal::log2(xm * (1.0f / 1024.0f)))) : 1.0f; float xi = 1.0f / xs;
              half2 xh[8];
              \(unroll) for (uint i = 0; i < 8u; i++) { xh[i] = half2(half(xt[2u*i] * xs), half(xt[2u*i+1u] * xs)); se += xt[2u*i]; so += xt[2u*i+1u]; }
        """

    static func statsLiterals(_ k: Double) -> (String, String, String, String) {
        let s = JANGTCode.stats[k]!
        return (JANGTCode.floatLiteral(s.0), JANGTCode.floatLiteral(s.1),
                JANGTCode.floatLiteral(s.2), JANGTCode.floatLiteral(s.3))
    }

    /// Split-simdgroup gathered gate+up+SwiGLU decode (Python `_gu_v3` Phase 2, rot_out = false).
    /// sg0 decodes gate (4 rows) and sg1 decodes up (same 4 rows), passing gate via 16-byte
    /// threadgroup memory. Halves live accumulators and outlier registers while remaining bit-exact.
    /// Grid (64, N/4, T*k).
    static func gateUp(kin: Int, n: Int, k: Double, l: Int, outliers: Int, words: Int, xdt: String,
                       odt: String, limit: Float, activation: JANGTActivation = .silu) -> String {
        let (mu0, mu1, isd0, isd1) = statsLiterals(k)
        let pl = JANGTCode.laneBits(k)
        let outlPre = outliers == 0 ? "" : """

            float oo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (lane < \(outliers)u) { int c = ocol[(size_t)e * \(outliers)u + lane];
              if (c >= 0) { float xo = float(x[(size_t)t * \(kin)u + uint(c)]);
                \(unroll) for (uint r = 0; r < 4u; r++) {
                  float wov = (sgi == 0u) ? float(woutg[((size_t)e * \(n)u + row0 + r) * \(outliers)u + lane])
                                          : float(woutu[((size_t)e * \(n)u + row0 + r) * \(outliers)u + lane]);
                  oo[r] = xo * wov; } } }
            \(unroll) for (uint r = 0; r < 4u; r++) oo[r] = simd_sum(oo[r]);
            """
        let outl = outliers == 0 ? "" : " v += oo[r];"
        let gateUpBody = """
            \(xload)
                  \(unroll) for (uint r = 0; r < 4u; r++) {
                    half2 aa = half2(0.0h);
                    const device uint32_t* wr = wp + r * \(words)u; \(tdot(k: k, l: l, row: "wr", acc: "aa"))
                    a0[r] += float(aa.x) * xi; a1[r] += float(aa.y) * xi;
                  }
            """
        let clamp = limit > 0
            ? " g = metal::min(g, \(String(format: "%.9e", limit))f); u = metal::clamp(u, \(String(format: "%.9e", -limit))f, \(String(format: "%.9e", limit))f);"
            : ""
        return """
            uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
            uint row0 = threadgroup_position_in_grid.y * 4u; uint ts = threadgroup_position_in_grid.z;
            uint t = ts / kslots[0]; uint e = idx[ts];\(outlPre)
            const uint lb = lane * \(pl)u; const uint sh = lb & 31u; (void)sh;
            const device uint32_t* wp = (sgi == 0u ? wg : wu) + ((size_t)e * \(n)u + row0) * \(words)u + (lb >> 5);
            const device float* xp = xr + (size_t)t * \(kin)u + lane * 16u;
            float a0[4], a1[4];
            \(unroll) for (uint r = 0; r < 4u; r++) { a0[r] = 0.0f; a1[r] = 0.0f; }
            float se = 0.0f, so = 0.0f;
            \(blocks(kin: kin, body: gateUpBody, advance: "wp += \(pl)u; xp += 512u;"))
            float SE = simd_sum(se), SO = simd_sum(so);
            threadgroup float g_sh[4];
            float val[4];
            \(unroll) for (uint r = 0; r < 4u; r++) {
              float A0 = simd_sum(a0[r]), A1 = simd_sum(a1[r]);
              if (lane == 0u) {
                float sc = (sgi == 0u) ? float(sg[(size_t)e * \(n)u + row0 + r]) : float(su_[(size_t)e * \(n)u + row0 + r]);
                float v = ((A0 - \(mu0) * SE) * \(isd0) + (A1 - \(mu1) * SO) * \(isd1)) * sc;\(outl)
                val[r] = v;
              }
            }
            if (sgi == 0u && lane == 0u) {
              \(unroll) for (uint r = 0; r < 4u; r++) g_sh[r] = val[r];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgi == 1u && lane == 0u) {
              \(unroll) for (uint r = 0; r < 4u; r++) {
                float g = g_sh[r], u = val[r];\(clamp)
                out[(size_t)ts * \(n)u + row0 + r] = \(odt)(\(activation.metalValue()));
              }
            }
            """
    }

    /// Parallel-slot weighted down decode (T <= 8, slots >= 8, N % 4 == 0, slots % 5 == 0 || slots % 4 == 0):
    /// `pSG` simdgroups per TG split `slots` across 4 output rows and reduce via threadgroup memory.
    /// Grid (pSG * 32, N/4, T) or swapped (pSG * 32, T, N/4) when `slotFastest`.
    static func downParallel(kin: Int, n: Int, k: Double, l: Int, outliers: Int, slots: Int, pSG: Int,
                             words: Int, roundRouteToBFloat16: Bool = false) -> String {
        let (mu0, mu1, isd0, isd1) = statsLiterals(k)
        let pl = JANGTCode.laneBits(k)
        let outlPre = outliers == 0 ? "" : """

              float od[4] = {0.0f, 0.0f, 0.0f, 0.0f};
              if (lane < \(outliers)u) { int c = ocol[(size_t)e * \(outliers)u + lane];
                if (c >= 0) { float ho = float(h[(size_t)ts * \(kin)u + uint(c)]);
                  \(unroll) for (uint r = 0; r < 4u; r++) od[r] = ho * float(wout[((size_t)e * \(n)u + row0 + r) * \(outliers)u + lane]); } }
              \(unroll) for (uint r = 0; r < 4u; r++) od[r] = simd_sum(od[r]);
            """
        let outl = outliers == 0 ? "" : " y += od[r];"
        let roundBF16 = roundRouteToBFloat16 ? " y = float(bfloat16_t(y));" : ""
        let downBody = """
            \(xload)
                    \(unroll) for (uint r = 0; r < 4u; r++) {
                      half2 ad = half2(0.0h);
                      const device uint32_t* wr = wp + r * \(words)u; \(tdot(k: k, l: l, row: "wr", acc: "ad"))
                      d0[r] += float(ad.x) * xi; d1[r] += float(ad.y) * xi;
                    }
            """
        let perSG = slots / pSG
        return """
            uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
            uint row0 = threadgroup_position_in_grid.y * 4u; uint t = threadgroup_position_in_grid.z;
            const uint lb = lane * \(pl)u; const uint sh = lb & 31u; (void)sh;
            threadgroup float sg_buf[\(pSG)][4];
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            uint k_start = sgi * \(perSG)u;
            for (uint j = 0; j < \(perSG)u; j++) {
              uint sl = k_start + j;
              uint ts = t * \(slots)u + sl; uint e = idx[ts]; float wgt = float(rw[ts]);\(outlPre)
              const device uint32_t* wp = wd + ((size_t)e * \(n)u + row0) * \(words)u + (lb >> 5);
              const device float* xp = hr + (size_t)ts * \(kin)u + lane * 16u;
              float d0[4], d1[4];
              \(unroll) for (uint r = 0; r < 4u; r++) { d0[r] = 0.0f; d1[r] = 0.0f; }
              float se = 0.0f, so = 0.0f;
              \(blocks(kin: kin, body: downBody, advance: "wp += \(pl)u; xp += 512u;"))
              float SE = simd_sum(se), SO = simd_sum(so);
              \(unroll) for (uint r = 0; r < 4u; r++) {
                float D0 = simd_sum(d0[r]), D1 = simd_sum(d1[r]);
                if (lane == 0u) {
                  float y = ((D0 - \(mu0) * SE) * \(isd0) + (D1 - \(mu1) * SO) * \(isd1)) * sd[(size_t)e * \(n)u + row0 + r];\(outl)\(roundBF16)
                  acc[r] = fma(wgt, y, acc[r]);
                }
              }
            }
            if (lane == 0u) { \(unroll) for (uint r = 0; r < 4u; r++) sg_buf[sgi][r] = acc[r]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgi == 0u && lane == 0u) {
              \(unroll) for (uint r = 0; r < 4u; r++) {
                float tot = sg_buf[0][r];
                \(unroll) for (uint s = 1u; s < \(pSG)u; s++) tot += sg_buf[s][r];
                out[(size_t)t * \(n)u + row0 + r] = tot;
              }
            }
            """
    }

    /// Weighted down decode accumulating the token's k slots (Python `_down_v3`). Grid (64, N/8, T).
    static func down(kin: Int, n: Int, k: Double, l: Int, outliers: Int, slots: Int, words: Int,
                     roundRouteToBFloat16: Bool = false) -> String {
        let (mu0, mu1, isd0, isd1) = statsLiterals(k)
        let pl = JANGTCode.laneBits(k)
        let outlPre = outliers == 0 ? "" : """

              float od[4] = {0.0f, 0.0f, 0.0f, 0.0f};
              if (lane < \(outliers)u) { int c = ocol[(size_t)e * \(outliers)u + lane];
                if (c >= 0) { float ho = float(h[(size_t)ts * \(kin)u + uint(c)]);
                  \(unroll) for (uint r = 0; r < 4u; r++) od[r] = ho * float(wout[((size_t)e * \(n)u + row0 + r) * \(outliers)u + lane]); } }
              \(unroll) for (uint r = 0; r < 4u; r++) od[r] = simd_sum(od[r]);
            """
        let outl = outliers == 0 ? "" : " y += od[r];"
        let roundBF16 = roundRouteToBFloat16 ? " y = float(bfloat16_t(y));" : ""
        let downBody = """
            \(xload)
                    \(unroll) for (uint r = 0; r < 4u; r++) {
                      half2 ad = half2(0.0h);
                      const device uint32_t* wr = wp + r * \(words)u; \(tdot(k: k, l: l, row: "wr", acc: "ad"))
                      d0[r] += float(ad.x) * xi; d1[r] += float(ad.y) * xi;
                    }
            """
        return """
            uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
            uint row0 = threadgroup_position_in_grid.y * 8u + sgi * 4u; uint t = threadgroup_position_in_grid.z;
            const uint lb = lane * \(pl)u; const uint sh = lb & 31u; (void)sh;
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint sl = 0; sl < \(slots)u; sl++) {
              uint ts = t * \(slots)u + sl; uint e = idx[ts]; float wgt = float(rw[ts]);\(outlPre)
              const device uint32_t* wp = wd + ((size_t)e * \(n)u + row0) * \(words)u + (lb >> 5);
              const device float* xp = hr + (size_t)ts * \(kin)u + lane * 16u;
              float d0[4], d1[4];
              \(unroll) for (uint r = 0; r < 4u; r++) { d0[r] = 0.0f; d1[r] = 0.0f; }
              float se = 0.0f, so = 0.0f;
              \(blocks(kin: kin, body: downBody, advance: "wp += \(pl)u; xp += 512u;"))
              float SE = simd_sum(se), SO = simd_sum(so);
              \(unroll) for (uint r = 0; r < 4u; r++) {
                float D0 = simd_sum(d0[r]), D1 = simd_sum(d1[r]);
                float y = ((D0 - \(mu0) * SE) * \(isd0) + (D1 - \(mu1) * SO) * \(isd1)) * sd[(size_t)e * \(n)u + row0 + r];\(outl)\(roundBF16)
                acc[r] = fma(wgt, y, acc[r]);
              }
            }
            if (lane == 0u) { \(unroll) for (uint r = 0; r < 4u; r++) out[(size_t)t * \(n)u + row0 + r] = acc[r]; }
            """
    }

    static func swapGridYZ(_ source: String) -> String {
        let placeholder = "__VMLX_GRID_YZ_SWAP__"
        return source
            .replacingOccurrences(of: "threadgroup_position_in_grid.y", with: placeholder)
            .replacingOccurrences(of: "threadgroup_position_in_grid.z", with: "threadgroup_position_in_grid.y")
            .replacingOccurrences(of: placeholder, with: "threadgroup_position_in_grid.z")
    }

    private static let e4m3InlineExpr = """
          float absmax = simd_max(local_max);
          float ratio = absmax * (1.0f / 448.0f);
          float exp_f = metal::ceil(metal::log2(ratio));
          float scale = metal::exp2(exp_f);
          float inv_scale = metal::exp2(-exp_f);
          #pragma unroll
          for (uint i = 0; i < 4u; i++) {
            float q = metal::clamp(v[i] * inv_scale, -448.0f, 448.0f);
            uint bits = as_type<uint>(q);
            uint sign = bits & 0x80000000u;
            uint mag = bits & 0x7FFFFFFFu;
            float rr;
            if (mag == 0u) {
              rr = 0.0f;
            } else {
              int f32_exp = int((mag >> 23) & 0xFFu) - 127;
              if (f32_exp >= -6) {
                uint keep = (mag >> 20) & 0x7u;
                uint rem = mag & 0x000FFFFFu;
                uint round_up = (rem > 0x00080000u) || (rem == 0x00080000u && (keep & 1u) != 0u);
                uint rounded = (mag & 0xFFF00000u) + (round_up ? 0x00100000u : 0u);
                if (rounded > 0x43E00000u) rounded = 0x43E00000u;
                rr = as_type<float>(sign | rounded);
              } else {
                float scaled_sub = as_type<float>(mag) * 512.0f;
                float rounded_int = metal::rint(scaled_sub);
                rr = (sign ? -1.0f : 1.0f) * (rounded_int * (1.0f / 512.0f));
              }
            }
            v[i] = rr * scale;
          }
        """

    /// Fused per-128 power-of-two E4M3FN rounding + sign multiply + H128 butterfly in 1 dispatch.
    /// Writes rounded `xq` (input dtype, for outlier columns) and rotated `xr` (float32).
    static func roundE4M3AndRot128(width: Int, xdt: String, perRowSigns: Bool) -> String {
        let sidx = perRowSigns ? "sidx[r]" : "0u"
        return """
            const uint lane = thread_index_in_simdgroup;
            const uint blk  = threadgroup_position_in_grid.x;
            const uint r    = threadgroup_position_in_grid.y;
            const size_t base = (size_t)r * \(width)u + blk * 128u + lane * 4u;
            float v[4];
            float local_max = 1e-10f;
            #pragma unroll
            for (uint i = 0; i < 4u; i++) {
              float val = float(x[base + i]);
              v[i] = val;
              local_max = metal::max(local_max, metal::abs(val));
            }
            \(e4m3InlineExpr)
            const device float* sv = s + (size_t)(\(sidx)) * \(width)u + blk * 128u + lane * 4u;
            #pragma unroll
            for (uint i = 0; i < 4u; i++) {
              xq[base + i] = \(xdt)(v[i]);
              v[i] *= sv[i];
            }
            { float a = v[0], b = v[1]; v[0] = a + b; v[1] = a - b; a = v[2]; b = v[3]; v[2] = a + b; v[3] = a - b; }
            { float a = v[0], b = v[2]; v[0] = a + b; v[2] = a - b; a = v[1]; b = v[3]; v[1] = a + b; v[3] = a - b; }
            for (uint d = 1u; d < 32u; d <<= 1) {
              bool upper = (lane & d) != 0u;
              #pragma unroll
              for (uint i = 0; i < 4u; i++) {
                float o = simd_shuffle_xor(v[i], d);
                v[i] = upper ? (o - v[i]) : (v[i] + o);
              }
            }
            #pragma unroll
            for (uint i = 0; i < 4u; i++) xr[base + i] = v[i] * 0.08838834764831845f;
            """
    }

    /// Fused per-128 power-of-two E4M3FN rounding + blockwise H32 butterfly in 1 dispatch.
    static func roundE4M3AndH32(width: Int, odt: String) -> String {
        return """
            const uint lane = thread_index_in_simdgroup;
            const uint blk  = threadgroup_position_in_grid.x;
            const uint r    = threadgroup_position_in_grid.y;
            const size_t base = (size_t)r * \(width)u + blk * 128u + lane * 4u;
            float v[4];
            float local_max = 1e-10f;
            #pragma unroll
            for (uint i = 0; i < 4u; i++) {
              float val = float(x[base + i]);
              v[i] = val;
              local_max = metal::max(local_max, metal::abs(val));
            }
            \(e4m3InlineExpr)
            { float a = v[0], b = v[1]; v[0] = a + b; v[1] = a - b; a = v[2]; b = v[3]; v[2] = a + b; v[3] = a - b; }
            { float a = v[0], b = v[2]; v[0] = a + b; v[2] = a - b; a = v[1]; b = v[3]; v[1] = a + b; v[3] = a - b; }
            for (uint d = 1u; d < 8u; d <<= 1) {
              bool upper = (lane & d) != 0u;
              #pragma unroll
              for (uint i = 0; i < 4u; i++) {
                float o = simd_shuffle_xor(v[i], d);
                v[i] = upper ? (o - v[i]) : (v[i] + o);
              }
            }
            #pragma unroll
            for (uint i = 0; i < 4u; i++) out[base + i] = \(odt)(v[i] * 0.17677669529663687f);
            """
    }

    /// out[r] = H128_blockwise(x[r] * s) in f32; s shared (K) or per-row expert signs s[sidx[r]] (E, K).
    static func rotateRows(width: Int, perRowSigns: Bool) -> String {
        let sidx = perRowSigns ? "sidx[r]" : "0u"
        return """
            uint lane = thread_index_in_simdgroup; uint blk = threadgroup_position_in_grid.x; uint r = threadgroup_position_in_grid.y;
            size_t base = (size_t)r * \(width)u + blk * 128u + lane * 4u;
            const device float* sv = s + (size_t)(\(sidx)) * \(width)u;
            float v[4];
            for (uint i = 0; i < 4u; i++) v[i] = float(x[base + i]) * sv[blk * 128u + lane * 4u + i];
            { float a = v[0], b = v[1]; v[0] = a + b; v[1] = a - b; a = v[2]; b = v[3]; v[2] = a + b; v[3] = a - b; }
            { float a = v[0], b = v[2]; v[0] = a + b; v[2] = a - b; a = v[1]; b = v[3]; v[1] = a + b; v[3] = a - b; }
            for (uint d = 1u; d < 32u; d <<= 1) {
              bool upper = (lane & d) != 0u;
              for (uint i = 0; i < 4u; i++) { float o = simd_shuffle_xor(v[i], d); v[i] = upper ? (o - v[i]) : (v[i] + o); }
            }
            for (uint i = 0; i < 4u; i++) out[base + i] = v[i] * 0.08838834764831845f;
            """
    }
}

/// Compiled JANGT decode kernels, cached by their generated name (one per shape/width/dtype combination).
enum JANGTKernels {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var cache: [String: MLXFast.MLXFastKernel] = [:]

    private static func kernel(_ name: String, inputs: [String], outputs: [String], source: () -> String)
        -> MLXFast.MLXFastKernel
    {
        lock.lock()
        defer { lock.unlock() }
        if let k = cache[name] { return k }
        let k = MLXFast.metalKernel(name: name, inputNames: inputs, outputNames: outputs, source: source())
        cache[name] = k
        return k
    }

    static func metalType(_ dtype: DType) -> String {
        switch dtype {
        case .bfloat16: return "bfloat16_t"
        case .float16: return "half"
        default: return "float"
        }
    }

    static func kname(_ k: Double) -> String { String(k).replacingOccurrences(of: ".", with: "p") }

    /// x (R, K) -> f32 H128(x * s); s (K,) or (E, K) with `rowExperts` (R,) selecting the signs row.
    static func rotateRows(_ x: MLXArray, signs: MLXArray, rowExperts: MLXArray? = nil) -> MLXArray {
        let rows = x.dim(0), width = x.dim(1)
        let perRow = rowExperts != nil
        let name = "jt_rotrows_k\(width)_\(metalType(x.dtype))_\(perRow ? 1 : 0)"
        let k = kernel(name, inputs: perRow ? ["x", "s", "sidx"] : ["x", "s"], outputs: ["out"]) {
            JANGTKernelSource.rotateRows(width: width, perRowSigns: perRow)
        }
        var inputs = [contiguous(x), contiguous(signs)]
        if let rowExperts { inputs.append(contiguous(rowExperts.asType(.uint32))) }
        return k(inputs, grid: (32 * (width / 128), rows, 1), threadGroup: (32, 1, 1),
                 outputShapes: [[rows, width]], outputDTypes: [.float32])[0]
    }

    /// Fused per-128 E4M3FN rounding + JANGT sign multiply + H128 rotation -> `(xq, xr_f32)`.
    static func roundE4M3AndRot128(_ x: MLXArray, signs: MLXArray, rowExperts: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let rows = x.dim(0), width = x.dim(1)
        let perRow = rowExperts != nil
        let xdt = metalType(x.dtype)
        let name = "jt_e4m3_rot128_k\(width)_\(xdt)_\(perRow ? 1 : 0)"
        let k = kernel(name, inputs: perRow ? ["x", "s", "sidx"] : ["x", "s"], outputs: ["xq", "xr"]) {
            JANGTKernelSource.roundE4M3AndRot128(width: width, xdt: xdt, perRowSigns: perRow)
        }
        var inputs = [contiguous(x), contiguous(signs)]
        if let rowExperts { inputs.append(contiguous(rowExperts.asType(.uint32))) }
        let out = k(inputs, grid: (32 * (width / 128), rows, 1), threadGroup: (32, 1, 1),
                    outputShapes: [[rows, width], [rows, width]], outputDTypes: [x.dtype, .float32])
        return (out[0], out[1])
    }

    /// Fused per-128 E4M3FN rounding + blockwise H32 rotation -> `outputDType` (`float32` for decode).
    static func roundE4M3AndH32(_ x: MLXArray, outputDType: DType = .float32) -> MLXArray {
        let rows = x.dim(0), width = x.dim(1)
        let odt = metalType(outputDType)
        let name = "jt_e4m3_h32_k\(width)_\(metalType(x.dtype))_\(odt)"
        let k = kernel(name, inputs: ["x"], outputs: ["out"]) {
            JANGTKernelSource.roundE4M3AndH32(width: width, odt: odt)
        }
        return k([contiguous(x)], grid: (32 * (width / 128), rows, 1), threadGroup: (32, 1, 1),
                 outputShapes: [[rows, width]], outputDTypes: [outputDType])[0]
    }

    /// Gathered gate+up+SwiGLU: xr (T, kin) f32 rotated, x (T, kin) raw, idx (T, k) -> (T*k, N) `outputDType`.
    static func gateUp(xr: MLXArray, x: MLXArray, indices: MLXArray, gate: JANGTProjection, up: JANGTProjection,
                       limit: Float, activation: JANGTActivation = .silu, outputDType: DType,
                       slotFastest: Bool = false) -> MLXArray {
        let tokens = x.dim(0), kin = x.dim(1), slots = indices.dim(1)
        let n = gate.outputs, words = gate.words, outliers = gate.outlierColumns
        let yz = slotFastest && tokens > 1
        let lim = limit > 0 ? "_lim\(String(limit).replacingOccurrences(of: ".", with: "p"))" : ""
        let name = "jt4_gusg_k\(kin)_n\(n)_K\(kname(gate.k))_o\(outliers)_w\(words)_\(metalType(x.dtype))_\(metalType(outputDType))\(lim)\(activation.nameTag)\(yz ? "_yz" : "")"
        var inputNames = ["xr", "x", "idx", "kslots", "wg", "sg", "wu", "su_"]
        if outliers > 0 { inputNames += ["ocol", "woutg", "woutu"] }
        let k = kernel(name, inputs: inputNames, outputs: ["out"]) {
            let src = JANGTKernelSource.gateUp(
                kin: kin, n: n, k: gate.k, l: JANGTCode.stateBits, outliers: outliers,
                words: words, xdt: metalType(x.dtype), odt: metalType(outputDType), limit: limit,
                activation: activation)
            return yz ? JANGTKernelSource.swapGridYZ(src) : src
        }
        var inputs: [MLXArray] = [
            contiguous(xr), contiguous(x), contiguous(indices.flattened().asType(.uint32)),
            MLXArray([UInt32(slots)]), gate.packed, gate.scale, up.packed, up.scale,
        ]
        if outliers > 0 { inputs += [gate.outlierColumnIndices, gate.outlierWeights, up.outlierWeights] }
        let grid = yz ? (64, tokens * slots, n / 4) : (64, n / 4, tokens * slots)
        return k(inputs, grid: grid, threadGroup: (64, 1, 1),
                 outputShapes: [[tokens * slots, n]], outputDTypes: [outputDType])[0]
    }

    /// Weighted down: hr (T*k, kin) f32 rotated, h (T*k, kin) raw, idx/rw (T, k) -> (T, N) f32.
    static func down(hr: MLXArray, h: MLXArray, indices: MLXArray, weights: MLXArray, projection: JANGTProjection,
                     slotFastest: Bool = false, roundRouteToBFloat16: Bool = false)
        -> MLXArray
    {
        let tokens = indices.dim(0), slots = indices.dim(1), kin = h.dim(1)
        let n = projection.outputs, words = projection.words, outliers = projection.outlierColumns
        let yz = slotFastest && tokens > 1
        let rtag = roundRouteToBFloat16 ? "_rbf16" : ""
        let yztag = yz ? "_yz" : ""
        var inputNames = ["hr", "h", "idx", "rw", "wd", "sd"]
        if outliers > 0 { inputNames += ["ocol", "wout"] }
        var inputs: [MLXArray] = [
            contiguous(hr), contiguous(h), contiguous(indices.flattened().asType(.uint32)),
            contiguous(weights.flattened().asType(.float32)), projection.packed, projection.scale,
        ]
        if outliers > 0 { inputs += [projection.outlierColumnIndices, projection.outlierWeights] }
        if tokens <= 8 && slots >= 8 && n % 4 == 0 && (slots % 5 == 0 || slots % 4 == 0) {
            let pSG = slots % 5 == 0 ? 5 : 4
            let name = "jt4_dnp_k\(kin)_n\(n)_K\(kname(projection.k))_o\(outliers)_s\(slots)_p\(pSG)_w\(words)_\(metalType(h.dtype))\(rtag)\(yztag)"
            let k = kernel(name, inputs: inputNames, outputs: ["out"]) {
                let src = JANGTKernelSource.downParallel(
                    kin: kin, n: n, k: projection.k, l: JANGTCode.stateBits, outliers: outliers,
                    slots: slots, pSG: pSG, words: words, roundRouteToBFloat16: roundRouteToBFloat16)
                return yz ? JANGTKernelSource.swapGridYZ(src) : src
            }
            let tg = pSG * 32
            let grid = yz ? (tg, tokens, n / 4) : (tg, n / 4, tokens)
            return k(inputs, grid: grid, threadGroup: (tg, 1, 1),
                     outputShapes: [[tokens, n]], outputDTypes: [.float32])[0]
        }
        let name = "jt4_dn_k\(kin)_n\(n)_K\(kname(projection.k))_o\(outliers)_s\(slots)_w\(words)_\(metalType(h.dtype))\(rtag)\(yztag)"
        let k = kernel(name, inputs: inputNames, outputs: ["out"]) {
            let src = JANGTKernelSource.down(
                kin: kin, n: n, k: projection.k, l: JANGTCode.stateBits, outliers: outliers,
                slots: slots, words: words, roundRouteToBFloat16: roundRouteToBFloat16)
            return yz ? JANGTKernelSource.swapGridYZ(src) : src
        }
        let grid = yz ? (64, tokens, n / 8) : (64, n / 8, tokens)
        return k(inputs, grid: grid, threadGroup: (64, 1, 1),
                 outputShapes: [[tokens, n]], outputDTypes: [.float32])[0]
    }
}

/// One JANGT projection of a routed layer: (E, N, W) trellis rows plus its scales, signs and outliers.
public struct JANGTProjection {
    public let k: Double
    public let inputs: Int
    public let outputs: Int
    public let experts: Int
    public let words: Int
    /// (E, N, W) uint32
    public let packed: MLXArray
    /// (E, N) float32 row scales
    public let scale: MLXArray
    /// (K_in,) float32 per-layer input signs
    public let signs: MLXArray
    /// (E, n_out) int32 outlier input columns (-1 = padding); n_out = 0 when the bundle has none.
    public let outlierColumnIndices: MLXArray
    /// (E, N, n_out) float16 outlier column weights
    public let outlierWeights: MLXArray
    /// (E, K_in) float32 per-expert column scales (down only)
    public let columnScales: MLXArray?
    public let outlierColumns: Int

    public init(k: Double, packed: MLXArray, scale: MLXArray, signs: MLXArray, outlierColumnIndices: MLXArray,
                outlierWeights: MLXArray, columnScales: MLXArray?) throws {
        guard JANGTCode.isSupported(k), packed.ndim == 3, packed.dtype == .uint32, scale.ndim == 2,
              scale.dim(0) == packed.dim(0), scale.dim(1) == packed.dim(1), signs.ndim == 1,
              outlierColumnIndices.ndim == 2, outlierColumnIndices.dim(0) == packed.dim(0),
              outlierWeights.ndim == 3
        else { throw JANGHFormatContract.ValidationError.invalid("invalid JANGT projection tensors") }
        let inputs = signs.dim(0)
        // Format geometry, not a size limit: the input basis is blockwise Hadamard-128 (so kin % 128 == 0; a
        // kin % 512 remainder runs the decode kernels' masked tail block), the decode kernels own 8 output rows per
        // threadgroup, and they give each outlier column one SIMD lane, so a bundle with more than 32 outlier columns
        // per row would have columns silently dropped in decode (the encoder default is 8).
        guard inputs % 128 == 0, packed.dim(1) % 8 == 0,
              packed.dim(2) == JANGTCode.wordsPerRow(inputs: inputs, k: k),
              outlierColumnIndices.dim(1) <= 32,
              outlierWeights.shape == [packed.dim(0), packed.dim(1), outlierColumnIndices.dim(1)],
              columnScales.map({ $0.shape == [packed.dim(0), inputs] }) ?? true
        else { throw JANGHFormatContract.ValidationError.invalid("JANGT projection geometry mismatch") }
        // The kernels read x at each outlier column: every index is padding (-1) or a real input column.
        let hasIndices = outlierColumnIndices.size > 0
        let low = hasIndices ? outlierColumnIndices.min().item(Int32.self) : -1
        let high = hasIndices ? outlierColumnIndices.max().item(Int32.self) : -1
        guard low >= -1, high < Int32(inputs) else {
            throw JANGHFormatContract.ValidationError.invalid("JANGT outlier column index out of range")
        }
        self.k = k
        self.inputs = inputs
        self.outputs = packed.dim(1)
        self.experts = packed.dim(0)
        self.words = packed.dim(2)
        self.packed = packed
        self.scale = scale.asType(.float32)
        self.signs = signs.asType(.float32)
        self.outlierColumnIndices = outlierColumnIndices.asType(.int32)
        self.outlierWeights = outlierWeights.asType(.float16)
        self.columnScales = columnScales?.asType(.float32)
        // Python `unit()`: n_out counts only when any column index is real (>= 0).
        self.outlierColumns = high >= 0 ? outlierColumnIndices.dim(1) : 0
    }
}
