// JANGH codebook tile loader and fused Hadamard prefill.
// Ported from the JANG reference; MLX native headers are generated separately.
enum JANGHPrefillSource {
    static let loader = #"""

// v2 tile loader: identical byte walk to MLX QuantizedBlockLoader (bitstream packing); decode = scale[row]*level(q)
template <typename T, short BROWS, short BCOLS, short dst_ld, short tgp_size, short bits>
struct TQBlockLoader {
  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads = (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short NCB = 1 << bits;
  const int src_ld; const int tile_stride; const short thread_idx; const short bi; const short bj;
  threadgroup T* dst; const device uint8_t* src; float s_row;
  TQBlockLoader(const device uint8_t* src_, const device half* scales_, const int src_ld_,
                threadgroup T* dst_, ushort simd_group_id, ushort simd_lane_id, short valid_rows)
      : src_ld(src_ld_), tile_stride(BCOLS_PACKED * bytes_per_pack),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED), bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor + bj * bytes_per_pack) {
    s_row = (bi < valid_rows) ? float(scales_[bi]) : 0.0f;
  }
  METAL_FUNC void decode_(short i) const {
    uint v = src[i * bytes_per_pack];
    if (bytes_per_pack > 1) v |= uint(src[i * bytes_per_pack + 1]) << 8;
    if (bytes_per_pack > 2) v |= uint(src[i * bytes_per_pack + 2]) << 16;
    for (short j = 0; j < pack_factor; j++)
      dst[i * pack_factor + j] = T(s_row * tq_level<bits>((v >> (j * bits)) & (NCB - 1)));
  }
  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) return;
    for (short i = 0; i < n_reads; i++) decode_(i);
  }
  void load_safe(short2 src_tile_dim) const {   // K % BK == 0 is enforced host-side; only rows can be ragged
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) return;
    if (bi >= src_tile_dim.y) { for (short i = 0; i < n_reads * pack_factor; i++) dst[i] = T(0); return; }
    for (short i = 0; i < n_reads; i++) decode_(i);
  }
  void next() { src += tile_stride; }
};

"""#
    static let nax = #"""

        using namespace mlx::steel;
        METAL_FUNC float tq_act(float g, float u, float lim) {
          if (lim > 0.0f) { g = metal::min(g, lim); u = metal::clamp(u, -lim, lim); }
          return (g / (1.0f + metal::fast::exp(-g))) * u;
        }
        // FUSED=false: y = x W^T for one weight.  FUSED=true: y = act(x Wg^T, x Wu^T).
        template <typename T, int bits, int bits_u, bool FUSED, bool ROT_OUT, int BM_ = 64>
        METAL_FUNC void tq_gather_qmm_nax(
            const device T* x, const device uint32_t* wg, const device half* sg, const device uint32_t* wu, const device half* su,
            const device uint32_t* indices, device T* y, const int M, const int N, const int K, const int EXPERTS, const float lim,
            threadgroup T* Wg, threadgroup T* Wu, uint3 tid, uint simd_group_id, uint simd_lane_id) {
          // BM_ = 64: prefill tile (2x2 simdgroups of 32x32, 128 threads). BM_ = 16: verify-window tile (1x2 simdgroups of
          // 16x32, 64 threads) so an 8-row DFlash2/MTP verify does not pay MMA for 24 padded rows. Each simdgroup keeps
          // SN = 32: tile_matmad_nax only issues MMAs when TN or TM is even (a 16x16 simdgroup tile computes NOTHING), and
          // ROT_OUT needs the 32-wide output block.
          constexpr int BM = BM_, BK = 64, BN = 64, WM = BM_ == 64 ? 2 : 1, WN = 2;
          constexpr int pack_factor = get_pack_factor<bits, 8>();
          constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
          constexpr int BK_padded = (BK + 16 / sizeof(T));
          using loader_w_t = TQBlockLoader<T, BN, BK, BK_padded, WM * WN * SIMD_SIZE, bits>;
          using loader_u_t = TQBlockLoader<T, BN, BK, BK_padded, WM * WN * SIMD_SIZE, bits_u>;
          const int K_w = K * bytes_per_pack / pack_factor; const int K_it = K / BK;
          const size_t stride_w = size_t(N) * K_w;
          const int K_wu = K * get_bytes_per_pack<bits_u>() / get_pack_factor<bits_u, 8>();
          const size_t stride_wu = size_t(N) * K_wu;
          const int y_row = tid.y * BM; const int y_col = tid.x * BN;
          const short tgp_bm = short(min(BM, M - y_row));
          const short tgp_bn = short(min(BN, N - y_col));
          auto wgl = (const device uint8_t*)wg; auto wul = (const device uint8_t*)wu;
          x += size_t(y_row) * K; y += size_t(y_row) * N + y_col;
          wgl += size_t(y_col) * K_w; if (FUSED) wul += size_t(y_col) * K_wu;
          constexpr short SM = BM / WM, SN = BN / WN, SK = 32;
          constexpr short TM = SM / 16, TN = SN / 16, TK = SK / 16;
          const short tm = SM * (simd_group_id / WN); const short tn = SN * (simd_group_id % WN);
          const short sgp_sm = short(min(int(SM), max(0, M - (y_row + tm))));  // clamp in int: M - row >= 32768 wraps a short
          const short sgp_sn = short(min(int(SN), max(0, N - (y_col + tn))));
          uint32_t index; short offset; uint32_t index_next = indices[y_row]; short offset_next = 0; int n = 0;
          while (n < tgp_bm) {
            n++; offset = offset_next; index = index_next; offset_next = tgp_bm;
            for (; n < tgp_bm; n++) { if (indices[y_row + n] != index) { offset_next = n; index_next = indices[y_row + n]; break; } }
            // Invalid routes propagate NaN without reading outside a mapped bank.
            // Segment identity is uniform across the complete threadgroup.
            if (index >= uint(EXPERTS)) {
              for (int q = simd_group_id * 32 + simd_lane_id;
                   q < (offset_next - offset) * tgp_bn; q += WM * WN * 32)
                y[(offset + q / tgp_bn) * N + q % tgp_bn] = T(as_type<float>(0x7fc00000u));
              continue;
            }
            threadgroup_barrier(mem_flags::mem_none);
            NAXTile<float, TM, TN> Gt; Gt.clear();
            NAXTile<float, TM, TN> Ut; if (FUSED) Ut.clear();
            const device T* xn = x + tm * K;
            thread loader_w_t lg(wgl + index * stride_w, sg + size_t(index) * N + y_col, K, Wg, simd_group_id, simd_lane_id, tgp_bn);
            thread loader_u_t lu(FUSED ? wul + index * stride_wu : wul, FUSED ? su + size_t(index) * N + y_col : sg, K,
                                 FUSED ? Wu : Wg, simd_group_id, simd_lane_id, tgp_bn);
            const bool full_n = (tgp_bn == BN);
            // MLX 0.32 optimization: a simdgroup whose rows lie outside this expert segment skips the MMA
            // (it still helps load the shared weight tile and joins every barrier).
            const short m_lo_lim = min(int(sgp_sm), max(0, offset - tm));
            const short m_hi_lim = min(int(sgp_sm), max(0, offset_next - tm));
            const bool sg_active = (m_hi_lim > m_lo_lim) && (sgp_sn > 0);
            dispatch_bool(sgp_sm == SM, [&](auto kAlignedM) {
              for (int k = 0; k < K_it; k++) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (full_n) { lg.load_unsafe(); if (FUSED) lu.load_unsafe(); }
                else { lg.load_safe(short2(BK, tgp_bn)); if (FUSED) lu.load_safe(short2(BK, tgp_bn)); }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg_active) {
                  STEEL_PRAGMA_NO_UNROLL
                  for (int kk1 = 0; kk1 < BK; kk1 += SK) {
                    NAXTile<T, TM, TK> Atile; NAXTile<T, TN, TK> Bg;
                    volatile int compiler_barrier;
                    if constexpr (kAlignedM.value) Atile.load(xn + kk1, K); else Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));
                    Bg.template load<T, BK_padded, 1>(Wg + tn * BK_padded + kk1);
                    tile_matmad_nax(Gt, Atile, metal::bool_constant<false>{}, Bg, metal::bool_constant<true>{});
                    if (FUSED) {
                      NAXTile<T, TN, TK> Bu;
                      Bu.template load<T, BK_padded, 1>(Wu + tn * BK_padded + kk1);
                      tile_matmad_nax(Ut, Atile, metal::bool_constant<false>{}, Bu, metal::bool_constant<true>{});
                    }
                    (void)compiler_barrier;
                  }
                }
                xn += BK; lg.next(); if (FUSED) lu.next();
              }
            });
            if (FUSED) {
              for (short i = 0; i < decltype(Gt)::kNumFrags; i++)
                for (short e = 0; e < decltype(Gt)::kElemsPerFrag; e++)
                  Gt.val_frags[i][e] = tq_act(Gt.val_frags[i][e], Ut.val_frags[i][e], lim);
            }
            if (ROT_OUT) {
              // JANGH rotation "hadamard32" of the NEXT projection's input, fused here: this simdgroup's output tile spans
              // exactly one 32-wide block of the hidden dimension (y_col + tn is a multiple of 32). Column index bits:
              // 0-1 = element within the lane, 2 = lane bit 0, 3 = lane bit 3, 4 = fragment column (NAX frag layout).
              const ushort ln = ushort(simd_lane_id);
              for (short fi = 0; fi < TM; fi++) {
                for (short hf = 0; hf < 2; hf++) {
                  float a[8];
                  for (short t = 0; t < 4; t++) { a[t] = Gt.val_frags[fi * TN][hf * 4 + t]; a[4 + t] = Gt.val_frags[fi * TN + 1][hf * 4 + t]; }
                  for (short j = 0; j < 8; j += 4) {
                    float p0 = a[j] + a[j + 1], p1 = a[j] - a[j + 1], p2 = a[j + 2] + a[j + 3], p3 = a[j + 2] - a[j + 3];
                    a[j] = p0 + p2; a[j + 1] = p1 + p3; a[j + 2] = p0 - p2; a[j + 3] = p1 - p3;
                  }
                  for (short j = 0; j < 8; j++) { float o = simd_shuffle_xor(a[j], ushort(1)); a[j] = (ln & 1) ? (o - a[j]) : (a[j] + o); }
                  for (short j = 0; j < 8; j++) { float o = simd_shuffle_xor(a[j], ushort(8)); a[j] = (ln & 8) ? (o - a[j]) : (a[j] + o); }
                  for (short t = 0; t < 4; t++) {
                    float lo = a[t], hi = a[4 + t];
                    Gt.val_frags[fi * TN][hf * 4 + t] = (lo + hi) * 0.17677669529663687f;
                    Gt.val_frags[fi * TN + 1][hf * 4 + t] = (lo - hi) * 0.17677669529663687f;
                  }
                }
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg_active) {
              if (m_lo_lim == 0 && m_hi_lim == SM && sgp_sn == SN) Gt.store(y + tm * N + tn, N);
              else Gt.store_slice(y + tm * N + tn, N, short2(0, m_lo_lim), short2(sgp_sn, m_hi_lim));
            }
          }
        }

        """#
    /// Dense (single-bank) verify-window split-K: the 16-row tile launched N/64 threadgroups of 64 threads, each
    /// walking ALL of K serially (27B down_proj: 80 threadgroups x 272 barrier-separated steps — the GPU mostly idle;
    /// a 4..16-row JANGH2 MLP cost 3x its 1-row decode). Here grid z slices K: each threadgroup covers a 64-column
    /// tile x a K-slice with the same codebook tile loader and MMA sequence, and writes FP32 partials [S, M, N] that
    /// the host sums. Dense 27B DFlash 2 is speed-first (summation order changes; admitted by agreement).
    static let naxSplitK = #"""

        template <typename T, int bits, int BM_ = 16>
        METAL_FUNC void tq_dense_qmm_nax_splitk(
            const device T* x, const device uint32_t* wg, const device half* sg, device float* yp,
            const int M, const int N, const int K, const int SPLITK,
            threadgroup T* Wg, uint3 tid, uint simd_group_id, uint simd_lane_id) {
          constexpr int BM = BM_, BK = 64, BN = 64, WM = BM_ == 64 ? 2 : 1, WN = 2;
          constexpr int pack_factor = get_pack_factor<bits, 8>();
          constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
          constexpr int BK_padded = (BK + 16 / sizeof(T));
          using loader_w_t = TQBlockLoader<T, BN, BK, BK_padded, WM * WN * SIMD_SIZE, bits>;
          const int K_w = K * bytes_per_pack / pack_factor;
          const int K_it = K / BK;
          const int per = (K_it + SPLITK - 1) / SPLITK;
          const int it0 = min(K_it, int(tid.z) * per);
          const int it1 = min(K_it, it0 + per);
          const int y_row = tid.y * BM; const int y_col = tid.x * BN;
          const short tgp_bn = short(min(BN, N - y_col));
          auto wgl = (const device uint8_t*)wg;
          wgl += size_t(y_col) * K_w + size_t(it0) * (BK * bytes_per_pack / pack_factor);
          device float* y = yp + size_t(tid.z) * size_t(M) * size_t(N) + size_t(y_row) * N + y_col;
          constexpr short SM = BM / WM, SN = BN / WN, SK = 32;
          constexpr short TM = SM / 16, TN = SN / 16, TK = SK / 16;
          const short tm = SM * (simd_group_id / WN); const short tn = SN * (simd_group_id % WN);
          const short sgp_sm = short(min(int(SM), max(0, M - (y_row + tm))));  // clamp in int: M - row >= 32768 wraps a short
          const short sgp_sn = short(min(int(SN), max(0, N - (y_col + tn))));
          NAXTile<float, TM, TN> Gt; Gt.clear();
          const device T* xn = x + size_t(y_row + tm) * K + size_t(it0) * BK;
          thread loader_w_t lg(wgl, sg + y_col, K, Wg, simd_group_id, simd_lane_id, tgp_bn);
          const bool full_n = (tgp_bn == BN);
          const bool sg_active = (sgp_sm > 0) && (sgp_sn > 0);
          dispatch_bool(sgp_sm == SM, [&](auto kAlignedM) {
            for (int k = it0; k < it1; k++) {
              threadgroup_barrier(mem_flags::mem_threadgroup);
              if (full_n) { lg.load_unsafe(); } else { lg.load_safe(short2(BK, tgp_bn)); }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              if (sg_active) {
                STEEL_PRAGMA_NO_UNROLL
                for (int kk1 = 0; kk1 < BK; kk1 += SK) {
                  NAXTile<T, TM, TK> Atile; NAXTile<T, TN, TK> Bg;
                  volatile int compiler_barrier;
                  if constexpr (kAlignedM.value) Atile.load(xn + kk1, K); else Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));
                  Bg.template load<T, BK_padded, 1>(Wg + tn * BK_padded + kk1);
                  tile_matmad_nax(Gt, Atile, metal::bool_constant<false>{}, Bg, metal::bool_constant<true>{});
                  (void)compiler_barrier;
                }
              }
              xn += BK; lg.next();
            }
          });
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (sg_active) {
            if (sgp_sm == SM && sgp_sn == SN) Gt.store(y + tm * N + tn, N);
            else Gt.store_slice(y + tm * N + tn, N, short2(0, 0), short2(sgp_sn, sgp_sm));
          }
        }

        """#
    static let steelSplitK = #"""

template <typename T, int bits>
METAL_FUNC void tq_dense_qmm_steel_splitk(
    const device T* x, const device uint32_t* wg, const device half* sg, device float* y,
    const int M, const int N, const int K, const int SPLITK, threadgroup T* Xs, threadgroup T* Ws,
    uint3 tid, uint simd_group_id, uint simd_lane_id) {
  constexpr int BM = 16, BN = 32, BK = 32, WM = 1, WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int PAD = BK + 16 / sizeof(T);
  using mma_t = mlx::steel::BlockMMA<T, float, BM, BN, BK, WM, WN, false, true, PAD, PAD>;
  using loader_x_t = mlx::steel::BlockLoader<T, BM, BK, PAD, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = TQBlockLoader<T, BN, BK, PAD, WM * WN * SIMD_SIZE, bits>;
  const int steps = K / BK, part = int(tid.z);
  const int start = part * (steps / SPLITK) + min(part, steps % SPLITK);
  const int count = steps / SPLITK + int(part < steps % SPLITK);
  const int col = int(tid.x) * BN;
  const short rows = short(M), cols = short(min(BN, N - col));
  x += size_t(start) * BK;
  auto w = (const device uint8_t*)wg + size_t(col) * (K * bytes_per_pack / pack_factor)
      + size_t(start) * (BK * bytes_per_pack / pack_factor);
  y += size_t(part) * M * N + col;
  thread mma_t mma_op(simd_group_id, simd_lane_id);
  thread loader_x_t lx(x, K, Xs, simd_group_id, simd_lane_id);
  thread loader_w_t lw(w, sg + col, K, Ws, simd_group_id, simd_lane_id, cols);
  if (rows == BM && cols == BN) gemm_loop_aligned(Xs, Ws, mma_op, lx, lw, count);
  else if (cols == BN) gemm_loop_unaligned<false, true, true>(Xs, Ws, mma_op, lx, lw, count, rows, cols, (short)BK);
  else if (rows == BM) gemm_loop_unaligned<true, false, true>(Xs, Ws, mma_op, lx, lw, count, rows, cols, (short)BK);
  else gemm_loop_unaligned<false, false, true>(Xs, Ws, mma_op, lx, lw, count, rows, cols, (short)BK);
  if (rows == BM && cols == BN) mma_op.store_result(y, N);
  else mma_op.store_result_slice(y, N, short2(0, 0), short2(cols, rows));
}

"""#
    static let steel = #"""

template <typename T, int bits>
METAL_FUNC void tq_gather_qmm_steel(
    const device T* x, const device uint32_t* wg, const device half* sg, const device uint32_t* indices, device T* y,
    const int M, const int N, const int K, const int EXPERTS, threadgroup T* Xs, threadgroup T* Ws,
    uint3 tid, uint simd_group_id, uint simd_lane_id) {
  // MLX affine_gather_qmm_rhs (non-NAX, transpose=true) with the TQ tile loader. Tiles: 16x32x32, 1x2 simdgroups.
  constexpr int BM = 16, BN = 32, BK = 32, WM = 1, WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t = mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = TQBlockLoader<T, BN, BK, BK_padded, WM * WN * SIMD_SIZE, bits>;
  const int K_w = K * bytes_per_pack / pack_factor; const int K_it = K / BK;
  const size_t stride_w = size_t(N) * K_w;
  const int y_row = tid.y * BM; const int y_col = tid.x * BN;
  const short tgp_bm = short(min(BM, M - y_row));
  const short tgp_bn = short(min(BN, N - y_col));
  auto wl = (const device uint8_t*)wg;
  x += size_t(y_row) * K; y += size_t(y_row) * N + y_col; wl += size_t(y_col) * K_w;
  uint32_t index; short offset; uint32_t index_next = indices[y_row]; short offset_next = 0; int n = 0;
  while (n < tgp_bm) {
    n++; offset = offset_next; index = index_next; offset_next = tgp_bm;
    for (; n < tgp_bm; n++) { if (indices[y_row + n] != index) { offset_next = n; index_next = indices[y_row + n]; break; } }
    // Invalid routes propagate NaN without reading outside a mapped bank.
    // Segment identity is uniform across the complete threadgroup.
    if (index >= uint(EXPERTS)) {
      for (int q = simd_group_id * 32 + simd_lane_id;
           q < (offset_next - offset) * tgp_bn; q += WM * WN * 32)
        y[(offset + q / tgp_bn) * N + q % tgp_bn] = T(as_type<float>(0x7fc00000u));
      continue;
    }
    threadgroup_barrier(mem_flags::mem_none);
    thread mma_t mma_op(simd_group_id, simd_lane_id);
    thread loader_x_t loader_x(x, K, Xs, simd_group_id, simd_lane_id);
    thread loader_w_t loader_w(wl + index * stride_w, sg + size_t(index) * N + y_col, K, Ws, simd_group_id, simd_lane_id, tgp_bn);
    if (tgp_bm == BM && tgp_bn == BN) gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
    else if (tgp_bn == BN) gemm_loop_unaligned<false, true, true>(Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, (short)BK);
    else if (tgp_bm == BM) gemm_loop_unaligned<true, false, true>(Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, (short)BK);
    else gemm_loop_unaligned<false, false, true>(Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, (short)BK);
    if (offset_next - offset == BM && tgp_bn == BN) mma_op.store_result(y, N);
    else mma_op.store_result_slice(y, N, short2(0, offset), short2(tgp_bn, offset_next));
  }
}

"""#
}
