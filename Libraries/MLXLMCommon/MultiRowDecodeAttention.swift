import Foundation
import MLX

/// Multi-row decode attention for speculative verify/draft blocks (opt-in: `VMLX_MULTIROW_ATTENTION=1`, or
/// `MultiRowDecodeAttention.enabled = true` from a speculative runtime).
///
/// Why: MLX serves 2–16 query rows over a long KV by re-reading the whole KV once per row (vector kernel, rows·gqa ≤ 32)
/// or with a tiled kernel that parallelises only over query tiles and walks every key serially (above that). Full K2
/// Horizon 7B forward (36 layers, GQA 32/8 × 128) at 32k context: 1 row 22 ms, 8 rows 115 ms, 16 rows 200 ms. That is
/// why Uno/DFlash speculation gets SLOWER as a chat grows — the verify block costs 5–9 AR steps.
///
/// What: flash-decoding with L rows. Threadgroup = (kv head, key block), 8 simdgroups; 32-key tiles of K (transposed) and
/// V are staged once in threadgroup memory and every (query head, row) pair of the GQA group consumes them. QK and PV are
/// 8×8 fp32 `simdgroup_matrix` MMAs; online softmax per tile; a second pass merges key blocks. Bottom-right causal.
/// Attention-only, 36 chained layers: 14k 8/12/16 rows 21.1/51.2/51.2 → 7.7/10.2/12.5 ms; 32k 48.9/116.8/116.8 →
/// 16.2/20.7/25.8 ms. Relative RMS error vs fp32 1.7e-3 (MLX bf16 SDPA: 2.3e-3).
///
/// DO NOT drop the `_Pragma("clang loop unroll(full)")` lines: without full unrolling the fragment arrays (Qf/Of/Sf)
/// are indexed dynamically, spill to stack memory, and the kernel runs ~5× SLOWER than MLX.
///
/// Not default-on: it changes rounding of every 2–16-row forward, including short prefill tails after a prefix-cache
/// hit, so ordinary decoding would no longer be byte-identical to stock.
public enum MultiRowDecodeAttention {
    nonisolated(unsafe) public static var enabled: Bool =
        ProcessInfo.processInfo.environment["VMLX_MULTIROW_ATTENTION"] == "1"

    /// Caches with an offset hand multi-row forwards an explicit `.array` causal mask (BaseKVCache.makeMask), which
    /// cannot be told apart from a tree/custom mask without reading it back. A caller that knows its rows form a plain
    /// linear chain (Uno, DFlash2 chain verify, MTP chain verify) sets this for the duration of that forward.
    nonisolated(unsafe) public static var arrayMaskIsCausal: Bool =
        ProcessInfo.processInfo.environment["VMLX_MULTIROW_ATTENTION_ARRAY_MASK_CAUSAL"] == "1"

    /// Run `body` with linear-chain array masks routed to the multi-row kernel.
    public static func withLinearChainMask<R>(_ body: () throws -> R) rethrows -> R {
        let previous = arrayMaskIsCausal
        arrayMaskIsCausal = true
        defer { arrayMaskIsCausal = previous }
        return try body()
    }

    public static func maskAllowed(_ mask: MLXFast.ScaledDotProductAttentionMaskMode, queryLength: Int, keyLength: Int) -> Bool {
        switch mask {
        case .causal: return true
        case .array(let m):
            // [L, N] or [1, 1, L, N]: one causal mask shared by every head (leading dims of 1 only).
            guard arrayMaskIsCausal, m.ndim >= 2, m.ndim <= 4,
                m.dim(-2) == queryLength, m.dim(-1) == keyLength
            else { return false }
            return m.shape.dropLast(2).allSatisfy { $0 == 1 }
        default: return false
        }
    }

    public static func eligible(queries: MLXArray, keys: MLXArray, values: MLXArray) -> Bool {
        guard queries.ndim == 4, keys.ndim == 4, values.ndim == 4, queries.dim(0) == 1,
            queries.dim(3) == 128 || queries.dim(3) == 256, keys.dim(3) == queries.dim(3),
            values.dim(3) == queries.dim(3),
            queries.dtype == .bfloat16, keys.dtype == .bfloat16, values.dtype == .bfloat16
        else { return false }
        let qL = queries.dim(2), hq = queries.dim(1), hkv = keys.dim(1)
        guard qL >= 2, qL <= 16, hkv > 0, hq % hkv == 0, keys.dim(2) >= qL else { return false }
        return (hq / hkv) * qL <= 256
    }

    static let pass1Source = """
        // MMA flash-decoding over L query rows. Threadgroup = (kv head, key block, pair group), 8 simdgroups; simdgroup sg
        // owns pairs [PB + 8sg, PB + 8sg + 8) of the P = gqa*L (query head, row) pairs, PB = 64 * pair group. Per TK-key
        // tile: S = Q·Kᵀ (8x8 fp32 MMA), row softmax in threadgroup memory, O += P·V (fp32 MMA). K staged transposed,
        // V row-major, both bf16. D and TK are template constants so every fragment loop fully unrolls.
        const int KS = TK + 2;      // ushort stride of transposed K rows (dims)
        const int VS = D + 8;       // ushort stride of V rows (keys)
        const int DC = D / 8;       // 8-wide dim chunks
        const int KB = TK / 8;      // 8-wide key chunks
        const int KPL = TK / 4;     // keys per lane in the softmax
        int h = threadgroup_position_in_grid.x;
        int blk = threadgroup_position_in_grid.y;
        int nblk = threadgroups_per_grid.y;
        int PB = threadgroup_position_in_grid.z * 64;
        uint tid = thread_position_in_threadgroup.x;
        int sg = tid / 32;
        ushort lane = tid % 32;
        int N = meta[0];
        int qL = meta[1];
        int gqa = meta[2];
        size_t kh = k_strides[1], kr = k_strides[2];
        size_t vh = v_strides[1], vr = v_strides[2];
        int P = gqa * qL;
        float scale = sc[0];
        int per = (N + nblk - 1) / nblk;
        per = ((per + TK - 1) / TK) * TK;
        int t0 = blk * per;
        int t1 = min(N, t0 + per);
        bool active = PB + sg * 8 < P;

        threadgroup ushort Kt[D * KS];
        threadgroup ushort Vt[TK * VS];
        threadgroup float St[8 * 8 * TK];
        threadgroup float Corr[8 * 8];

        const short qid = lane / 4;
        const short fm = (qid & 4) + ((lane / 2) % 4);
        const short fn = (qid & 2) * 2 + (lane % 2) * 2;

        simdgroup_matrix<float, 8, 8> Qf[DC];
        simdgroup_matrix<float, 8, 8> Of[DC];
        int prow = PB + sg * 8 + fm;
        _Pragma("clang loop unroll(full)") for (int c = 0; c < DC; c++) {
            float2 e = float2(0);
            if (active && prow < P) {
                size_t qb = ((size_t)h * P + prow) * D + c * 8 + fn;
                e = float2(float(q[qb]), float(q[qb + 1])) * scale;
            }
            Qf[c].thread_elements()[0] = e.x; Qf[c].thread_elements()[1] = e.y;
            Of[c] = simdgroup_matrix<float, 8, 8>(0);
        }
        int srow = lane / 4;
        int spair = PB + sg * 8 + srow;
        // Visibility: every prefix key (key < N - qL); inside the L-row window, the keys whose bit is set in
        // win[row] (bit a = window row a). A linear chain is win[row] = (1 << (row + 1)) - 1; a draft tree sets the
        // rows on the row's own root path.
        int wstart = N - qL;
        uint wbits = spair < P ? win[spair % qL] : 0u;
        float m = -INFINITY, l = 0;

        const device ushort* kw = (const device ushort*)k;
        const device ushort* vw = (const device ushort*)v;
        for (int tb = t0; tb < t1; tb += TK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int w = tid; w < TK * DC; w += 256) {        // DC x uint4 (8 bf16) per key row
                int key = w / DC, c8 = (w % DC) * 8;
                int t = min(tb + key, N - 1);
                uint4 kv4 = *((const device uint4*)(kw + h * kh + t * kr + c8));
                uint4 vv4 = *((const device uint4*)(vw + h * vh + t * vr + c8));
                *((threadgroup uint4*)(Vt + key * VS + c8)) = vv4;
                uint kwv[4] = {kv4.x, kv4.y, kv4.z, kv4.w};
                for (int i = 0; i < 4; i++) {
                    Kt[(c8 + 2 * i) * KS + key] = ushort(kwv[i] & 0xffff);
                    Kt[(c8 + 2 * i + 1) * KS + key] = ushort(kwv[i] >> 16);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (!active) continue;
            simdgroup_matrix<float, 8, 8> Sf[KB];
            _Pragma("clang loop unroll(full)") for (int kb = 0; kb < KB; kb++) Sf[kb] = simdgroup_matrix<float, 8, 8>(0);
            _Pragma("clang loop unroll(full)") for (int c = 0; c < DC; c++) {
                _Pragma("clang loop unroll(full)") for (int kb = 0; kb < KB; kb++) {
                    simdgroup_matrix<float, 8, 8> kf;
                    const threadgroup ushort* kp = Kt + (c * 8 + fm) * KS + kb * 8 + fn;
                    kf.thread_elements()[0] = float(as_type<bfloat>(kp[0]));
                    kf.thread_elements()[1] = float(as_type<bfloat>(kp[1]));
                    simdgroup_multiply_accumulate(Sf[kb], Qf[c], kf, Sf[kb]);
                }
            }
            threadgroup float* Sp = St + sg * 8 * TK;
            _Pragma("clang loop unroll(full)") for (int kb = 0; kb < KB; kb++) simdgroup_store(Sf[kb], Sp + kb * 8, TK);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            float sv[KPL];
            float tmax = -INFINITY;
            _Pragma("clang loop unroll(full)") for (int i = 0; i < KPL; i++) {
                int kk = (lane % 4) * KPL + i;
                int key = tb + kk;
                bool ok = spair < P && key < t1
                    && (key < wstart || ((wbits >> uint(key - wstart)) & 1u) != 0u);
                sv[i] = ok ? Sp[srow * TK + kk] : -INFINITY;
                tmax = max(tmax, sv[i]);
            }
            tmax = max(tmax, simd_shuffle_xor(tmax, 1));
            tmax = max(tmax, simd_shuffle_xor(tmax, 2));
            float mn = max(m, tmax);
            float corr = (mn == -INFINITY) ? 1.0f : ((m == -INFINITY) ? 0.0f : fast::exp(m - mn));
            float ps = 0;
            _Pragma("clang loop unroll(full)") for (int i = 0; i < KPL; i++) {
                float pv = (sv[i] == -INFINITY) ? 0.0f : fast::exp(sv[i] - mn);
                ps += pv;
                Sp[srow * TK + (lane % 4) * KPL + i] = pv;
            }
            ps += simd_shuffle_xor(ps, 1);
            ps += simd_shuffle_xor(ps, 2);
            l = l * corr + ps;
            m = mn;
            if (lane % 4 == 0) Corr[sg * 8 + srow] = corr;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            float cr = Corr[sg * 8 + fm];
            _Pragma("clang loop unroll(full)") for (int dc = 0; dc < DC; dc++) {
                Of[dc].thread_elements()[0] *= cr; Of[dc].thread_elements()[1] *= cr;
            }
            _Pragma("clang loop unroll(full)") for (int kb = 0; kb < KB; kb++) {
                simdgroup_matrix<float, 8, 8> pf;
                simdgroup_load(pf, Sp + kb * 8, TK);
                _Pragma("clang loop unroll(full)") for (int dc = 0; dc < DC; dc++) {
                    simdgroup_matrix<float, 8, 8> vf;
                    const threadgroup ushort* vp = Vt + (kb * 8 + fm) * VS + dc * 8 + fn;
                    vf.thread_elements()[0] = float(as_type<bfloat>(vp[0]));
                    vf.thread_elements()[1] = float(as_type<bfloat>(vp[1]));
                    simdgroup_multiply_accumulate(Of[dc], pf, vf, Of[dc]);
                }
            }
        }
        if (!active) return;
        if (lane % 4 == 0 && spair < P) {
            size_t o = ((size_t)h * P + spair) * nblk + blk;
            pm[o] = m; pl[o] = l;
        }
        if (prow < P) {
            size_t o = ((size_t)h * P + prow) * nblk + blk;
            _Pragma("clang loop unroll(full)") for (int dc = 0; dc < DC; dc++) {
                pacc[o * D + dc * 8 + fn] = Of[dc].thread_elements()[0];
                pacc[o * D + dc * 8 + fn + 1] = Of[dc].thread_elements()[1];
            }
        }
    """

    static let pass2Source = """
        int pr = threadgroup_position_in_grid.x;     // (query head, row) flattened
        uint d = thread_position_in_threadgroup.x;
        int nblk = meta[0];
        float M = -INFINITY;
        for (int b = 0; b < nblk; b++) M = max(M, pm[pr * nblk + b]);
        float L = 0, A = 0;
        for (int b = 0; b < nblk; b++) {
            float mb = pm[pr * nblk + b];
            float w = (mb == -INFINITY) ? 0.0f : fast::exp(mb - M);
            L += pl[pr * nblk + b] * w;
            A += pacc[((size_t)pr * nblk + b) * D + d] * w;
        }
        out[pr * D + d] = static_cast<OutT>(A / L);
    """

    static let pass1 = MLXFast.metalKernel(
        name: "vmlx_multirow_attn_pass1", inputNames: ["q", "k", "v", "meta", "sc", "win"],
        outputNames: ["pm", "pl", "pacc"], source: pass1Source, ensureRowContiguous: false)
    static let pass2 = MLXFast.metalKernel(
        name: "vmlx_multirow_attn_pass2", inputNames: ["pm", "pl", "pacc", "meta"], outputNames: ["out"],
        source: pass2Source, ensureRowContiguous: false)

    /// Lower-triangular window bits for an L-row linear chain.
    public static func causalWindowBits(_ rows: Int) -> MLXArray {
        MLXArray((0 ..< rows).map { UInt32((UInt64(1) << UInt64($0 + 1)) - 1) })
    }

    /// Window bits from row-to-window-row visibility (`allowed[r][a]`), e.g. a draft tree's root paths.
    public static func windowBits(_ allowed: [[Bool]]) -> MLXArray {
        MLXArray(allowed.map { row in
            row.enumerated().reduce(UInt32(0)) { $0 | ($1.element ? UInt32(1) << UInt32($1.offset) : 0) }
        })
    }

    /// Forwards served by the kernel in this process (diagnostic; read by RunBench).
    nonisolated(unsafe) public static var servedCalls = 0

    /// queries [1, Hq, L, D] (D = 128 or 256); keys/values [1, Hkv, N, D] (last dim contiguous). Returns [1, Hq, L, D].
    ///
    /// `windowBits` (uint32 [L]): bit a of row r set = row r sees window row a (the last L keys). nil = causal chain.
    public static func attend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, windowBits: MLXArray? = nil,
        blocks: Int? = nil
    ) -> MLXArray {
        servedCalls += 1
        let q = contiguous(queries)
        let hq = q.dim(1), qL = q.dim(2), d = q.dim(3), hkv = keys.dim(1), n = keys.dim(2)
        let pairs = (hq / hkv) * qL
        let pairGroups = (pairs + 63) / 64
        // Threadgroup memory: K tile D*(TK+2) + V tile TK*(D+8) ushorts + 8*8*TK floats — TK=16 keeps D=256 under 32 KB.
        let tk = d == 128 ? 32 : 16
        let nb = blocks ?? (n < 4096 ? 32 : 64)
        let p = pass1(
            [q, keys, values, MLXArray([Int32(n), Int32(qL), Int32(hq / hkv)]), MLXArray([scale]),
             windowBits ?? causalWindowBits(qL)],
            template: [("D", d), ("TK", tk)],
            grid: (hkv * 256, nb, pairGroups), threadGroup: (256, 1, 1),
            outputShapes: [[hq * qL, nb], [hq * qL, nb], [hq * qL, nb, d]],
            outputDTypes: [.float32, .float32, .float32])
        return pass2(
            [p[0], p[1], p[2], MLXArray([Int32(nb)])], template: [("OutT", q.dtype), ("D", d)],
            grid: (hq * qL * d, 1, 1), threadGroup: (d, 1, 1), outputShapes: [[1, hq, qL, d]],
            outputDTypes: [q.dtype])[0]
    }
}
