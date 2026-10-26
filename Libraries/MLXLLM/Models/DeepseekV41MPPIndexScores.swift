// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 DSA index scores on the M5 matrix units: out[s, p] = sum_h w[s, h] * relu(q[s, h, :] . k[p, :]).
//
// Port of vMLX `vmlx_engine/metal/dsa_index_scores_mpp.py` (3b9e0e6d; wired into `dsa_pool_index_scores` by
// 391ea34f): `_SOURCE`, `_HEADER`, the 16 x 256 x 4 tile, `_launch`, `available()` and the admission and tail
// padding of `dsa_pool_index_scores_mpp`.
//
// One threadgroup of 4 simdgroups owns a 16-query x 256-key output tile. Per head it runs one cooperative
// [16 x D] @ [D x 256] Metal 4 `mpp::tensor_ops::matmul2d` (bf16 operands, fp32 accumulation), then applies
// relu and the head weight in registers and adds into the tile's running head sum; the fp32 [S, P] tile is
// written once. The last row / column tile is clamped to end at S / P (it overlaps its neighbour and
// recomputes those cells with the same arithmetic). Python measured 14-22 ms vs 62 ms for the fp32
// simdgroup-MMA kernel (`DeepseekV41FusedIndexScores`) at S=2048, P=100k, H=32, D=128 on the M5 Ultra.
//
// The tile shape is fixed: wider per-thread tiles (16x512x2, 16x1024x4) compute wrong results on the current
// Metal compiler, so `available()` runs the kernel once per process against an fp32 reference, tails included.
//
// Arithmetic: not bit-identical to the fp32 kernel. bf16 x bf16 products are exact in fp32, so for bf16-valued
// operands (DeepSeek-V4.1's) the only difference is the op's own dot-product order; heads sum 0..H-1 in fp32.

import Foundation
import MLX
import MLXFast

enum DeepseekV41MPPIndexScores {
    /// `VMLX_DSV41_INDEX_MPP=0` (or false/off/no) keeps the fp32 fused kernel for prefill.
    static func requested(_ environment: [String: String]) -> Bool {
        let raw = environment["VMLX_DSV41_INDEX_MPP"] ?? "1"
        return !["0", "false", "off", "no"].contains(
            raw.trimmingCharacters(in: .whitespaces).lowercased())
    }

    static let enabled: Bool = requested(ProcessInfo.processInfo.environment)

    /// Python `TQ, TK, NSG`: queries x keys per threadgroup tile, simdgroups per threadgroup.
    static let tq = 16
    static let tk = 256
    static let nsg = 4

    static let header = """

        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        using namespace mpp::tensor_ops;

        """

    /// Verbatim `_SOURCE` of `vmlx_engine/metal/dsa_index_scores_mpp.py`.
    static let source = #"""
          const int S = int(q_shape[1]);
          const int T = int(k_shape[1]);
          const int c0 = min(int(threadgroup_position_in_grid.x) * TK, T - TK);
          const int r0 = min(int(threadgroup_position_in_grid.y) * TQ, S - TQ);
          const uint tid = thread_position_in_threadgroup.x;
          const float sc = scale[0];

          threadgroup float ws[TQ * H];                        // this tile's head weights, [TQ][H]
          for (uint i = tid; i < TQ * H; i += NSG * 32) ws[i] = float(w[(size_t)r0 * H + i]);
          threadgroup_barrier(mem_flags::mem_threadgroup);

          constexpr auto desc = matmul2d_descriptor(TQ, TK, D, false, true, false, matmul2d_descriptor::mode::multiply);
          matmul2d<desc, execution_simdgroups<NSG>> op;
          // Q rows r0.. as a [TQ][H*D] matrix (head h is columns h*D..h*D+D), K rows c0.. as [TK][D] (transposed operand).
          tensor<device elem_t, dextents<int32_t, 2>, tensor_inline> tQ(
              (device elem_t*)q + (size_t)r0 * (H * D), dextents<int32_t, 2>(H * D, TQ));
          tensor<device elem_t, dextents<int32_t, 2>, tensor_inline> tK(
              (device elem_t*)k + (size_t)c0 * D, dextents<int32_t, 2>(D, TK));
          auto b = tK.slice<D, TK>(0, 0);
          auto a0 = tQ.slice<D, TQ>(0, 0);
          auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b), float>();
          #pragma unroll
          for (uint16_t i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = 0.0f;

          for (int h = 0; h < H; ++h) {
            auto a = tQ.slice<D, TQ>(h * D, 0);
            auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
            op.run(a, b, P);
            #pragma unroll
            for (uint16_t i = 0; i < P.get_capacity(); ++i) {
              if (P.is_valid_element(i)) {
                const auto idx = P.get_multidimensional_index(i);      // (key column, query row) within the tile
                const float s = HAS_SCALE ? P[i] * sc : P[i];
                acc[i] += metal::max(s, 0.0f) * ws[idx[1] * H + h];
              }
            }
          }

          // The tile's [TQ][TK] block of the [S][T] output (row stride T).
          tensor<device float, dextents<int32_t, 2>, tensor_inline> tO(
              out + (size_t)r0 * T + c0, dextents<int32_t, 2>(TK, TQ), array<int32_t, 2>{1, T});
          acc.store(tO);
        """#

    // MARK: kernels

    private static let lock = NSLock()
    nonisolated(unsafe) private static var compiled: [String: MLXFast.MLXFastKernel] = [:]

    /// Python `_kernel`: one kernel per (element type, H, D, has-scale); the constants are baked into the
    /// source as `constexpr` lines so `matmul2d_descriptor` sees compile-time values.
    private static func kernel(elem: String, heads H: Int, dim D: Int, hasScale: Bool)
        -> MLXFast.MLXFastKernel
    {
        let key = "\(elem)_h\(H)_d\(D)_s\(hasScale ? 1 : 0)"
        lock.lock()
        defer { lock.unlock() }
        if let k = compiled[key] { return k }
        let consts =
            "  using elem_t = \(elem);\n"
            + [("H", H), ("D", D), ("TQ", tq), ("TK", tk), ("NSG", nsg), ("HAS_SCALE", hasScale ? 1 : 0)]
            .map { "  constexpr int \($0.0) = \($0.1);\n" }.joined()
        // METAL-ONLY: case 2. Metal 4 tensor-op kernel; `poolIndexScores` returns nil unless `available()`
        // (M5-class GPU, self-check passed), and the indexer then takes the fp32 fused kernel or the
        // MLX-op path.
        let k = MLXFast.metalKernel(
            name: "vmlx_dsv41_mpp_index_scores_v1_\(key)",
            inputNames: ["q", "k", "w", "scale"],
            outputNames: ["out"],
            source: consts + source,
            header: header,
            ensureRowContiguous: true)
        compiled[key] = k
        return k
    }

    static func elementType(_ dtype: DType) -> String? {
        switch dtype {
        case .bfloat16: return "bfloat"
        case .float16: return "half"
        default: return nil
        }
    }

    /// Python `_launch`: q (1, S, H, D) and k (1, T, D) of one 16/32-bit float type with S >= 16, T >= 256.
    static func launch(q: MLXArray, k: MLXArray, w: MLXArray, scale: Float) -> MLXArray {
        let S = q.dim(1), H = q.dim(2), D = q.dim(3), T = k.dim(1)
        let run = kernel(elem: elementType(q.dtype)!, heads: H, dim: D, hasScale: scale != 1.0)
        return run(
            [q, k, w, MLXArray([scale])],
            grid: (((T + tk - 1) / tk) * 32 * nsg, (S + tq - 1) / tq, 1),
            threadGroup: (32 * nsg, 1, 1),
            outputShapes: [[1, S, T]],
            outputDTypes: [.float32])[0]
    }

    // MARK: availability

    /// Metal 4 tensor ops on the matrix units: an M5-class GPU (`applegpu_g17*` and later; `p` parts from
    /// g18), the same rule as MLX's NAX gate (`JANGHPrefillKernel.backend(forMetalArchitecture:)`).
    static func hasMatrixUnits(architecture: String) -> Bool {
        guard let kind = architecture.last, "gspd".contains(kind) else { return false }
        let stem = architecture.dropLast()
        guard let marker = stem.lastIndex(of: "g"),
            let generation = Int(stem[stem.index(after: marker)...])
        else { return false }
        return generation >= (kind == "p" ? 18 : 17)
    }

    nonisolated(unsafe) private static var availability: Bool?

    /// Metal 4 tensor ops are present (macOS 26.2+, M5-class GPU) and this kernel matches an fp32 reference,
    /// clamped row / column tails included (Python `available()`: S=40, T=600, H=4, D=128, scale 0.5,
    /// max error < 1e-4 of the largest score). The self-check runs once per process; a compile or runtime
    /// failure counts as unavailable.
    static func available() -> Bool {
        lock.lock()
        if let a = availability {
            lock.unlock()
            return a
        }
        lock.unlock()
        let ok = hardwareSupported() && selfCheck()
        lock.lock()
        availability = ok
        lock.unlock()
        return ok
    }

    static func hardwareSupported() -> Bool {
        #if os(macOS)
            guard #available(macOS 26.2, *) else { return false }
            return hasMatrixUnits(architecture: GPU.deviceInfo().architecture)
        #else
            return false
        #endif
    }

    /// Python `available()` body: the kernel on bf16 inputs against an fp32 CPU reference of the same values.
    /// MLX errors (a kernel that does not compile here) are caught by `withError` and count as a failure.
    static func selfCheck() -> Bool {
        do {
            return try withError { error in
                try Device.withDefaultDevice(.gpu) {
                    // Deterministic signed values of unit scale (the Python check uses mx.random key 7).
                    func wave(_ shape: [Int], _ step: Float) -> MLXArray {
                        let n = shape.reduce(1, *)
                        return (sin(MLXArray(0 ..< Int32(n)).asType(.float32) * step) * 1.7).reshaped(shape)
                    }
                    let S = 40, T = 600, H = 4, D = 128
                    let q = wave([1, S, H, D], 0.7310).asType(.bfloat16)
                    let kk = wave([1, T, D], 0.3779).asType(.bfloat16)
                    let w = wave([1, S, H], 1.1213)
                    let got = launch(q: q, k: kk, w: w, scale: 0.5)
                    eval(got)
                    try error.check()
                    // [S, H, D] x [D, T] -> [S, H, T]; relu(scale * s) * w summed over heads, on the CPU.
                    let s = matmul(
                        q[0].asType(.float32), kk[0].asType(.float32).transposed(), stream: .cpu)
                    let ref = (maximum(s * 0.5, Float(0), stream: .cpu)
                        * w[0].expandedDimensions(axis: -1)).sum(axis: 1, stream: .cpu)
                    let peak = abs(ref).max().item(Float.self)
                    let err = abs(got[0] - ref).max().item(Float.self) / max(peak, 1e-6)
                    try error.check()
                    return err < 1e-4
                }
            }
        } catch {
            return false
        }
    }

    // MARK: entry point

    /// Python `dsa_pool_index_scores_mpp`. q (1, S, H, D) and poolKeys (1, P, D), both bf16 or both fp16, and
    /// headW (1, S, H) fp32/bf16/fp16 -> (1, S, P) fp32 = sum_h headW * relu(scale * q . poolKey) on the matrix
    /// units; nil when not admitted, turned off (`VMLX_DSV41_INDEX_MPP=0` or `enabled: false`), off the GPU,
    /// or when the matrix units are unavailable (callers fall back to `DeepseekV41FusedIndexScores`, then the
    /// stock path).
    ///
    /// Admitted: batch 1, D a multiple of 32 up to 512, 1 <= H <= 256, S >= 1, P >= 1, scale > 0. S below one 16-row tile or P below one 256-key tile is zero-padded to
    /// one tile.
    static func poolIndexScores(
        q: MLXArray, poolKeys: MLXArray, headW: MLXArray, scale: Float, enabled on: Bool = enabled
    ) -> MLXArray? {
        guard on else { return nil }
        // METAL-ONLY: case 1. Guards the Metal kernel above; off the GPU the caller takes the MLX-op path.
        guard Device.defaultDevice().deviceType == .gpu else { return nil }
        guard admit(q: q, poolKeys: poolKeys, headW: headW, scale: scale) else { return nil }
        guard available() else { return nil }
        let S = q.dim(1), P = poolKeys.dim(1)
        let padS = max(0, tq - S), padP = max(0, tk - P)
        if padS > 0 || padP > 0 {
            let qq = padS > 0 ? padded(q, widths: [0, .init((0, padS)), 0, 0]) : q
            let ww = padS > 0 ? padded(headW, widths: [0, .init((0, padS)), 0]) : headW
            let kk = padP > 0 ? padded(poolKeys, widths: [0, .init((0, padP)), 0]) : poolKeys
            return launch(q: qq, k: kk, w: ww, scale: scale)[0..., 0 ..< S, 0 ..< P]
        }
        return launch(q: q, k: poolKeys, w: headW, scale: scale)
    }

    /// The shape / dtype half of the admission (no device or availability check).
    static func admit(q: MLXArray, poolKeys: MLXArray, headW: MLXArray, scale: Float) -> Bool {
        guard q.ndim == 4, poolKeys.ndim == 3, headW.ndim == 3 else { return false }
        let B = q.dim(0), S = q.dim(1), H = q.dim(2), D = q.dim(3)
        guard B == 1, poolKeys.dim(0) == 1, poolKeys.dim(2) == D, headW.shape == [1, S, H] else {
            return false
        }
        let P = poolKeys.dim(1)
        guard elementType(q.dtype) != nil, poolKeys.dtype == q.dtype,
            [DType.float32, .bfloat16, .float16].contains(headW.dtype)
        else { return false }
        return admits(heads: H, dim: D) && S >= 1 && P >= 1 && scale > 0
    }

    /// The head-geometry half of the admission: the indexer checks it once per model; every per-chunk call it
    /// then makes (bf16 operands, S >= 1, P >= 1, scale 1) is admitted.
    static func admits(heads H: Int, dim D: Int) -> Bool {
        D % 32 == 0 && D <= 512 && (1 ... 256).contains(H)
    }
}

extension DeepseekV41MPPIndexScores {
    /// How `DeepseekV41Indexer.select` scores a call.
    enum PrefillPath: Equatable {
        /// bf16 q / keys / weights on the matrix units (this kernel).
        case matrixUnits
        /// fp32 simdgroup-MMA fused kernel (`DeepseekV41FusedIndexScores`) on the bf16 values upcast to fp32.
        case fp32Fused
        /// Stock bf16 MLX ops (decode, dSpark verify, `VMLX_DSV41_FUSED_INDEX=0`, off the GPU).
        case stock
    }

    /// Prefill gate: the fused kernels serve exactly the calls `DeepseekV41FusedIndexScores.eligible` admits
    /// (more than 8 query rows, at least 32 keys, on the GPU); among those the matrix units are preferred when
    /// turned on, the head geometry is admitted and `available()`. Decode and dSpark verify (<= 8 rows) stay
    /// on the stock path.
    static func prefillPath(
        queries: Int, keys: Int, heads: Int, dim: Int,
        mppEnabled: Bool = enabled, mppAvailable: () -> Bool = available
    ) -> PrefillPath {
        guard DeepseekV41FusedIndexScores.eligible(queries: queries, keys: keys) else { return .stock }
        guard mppEnabled, admits(heads: heads, dim: dim), mppAvailable() else { return .fp32Fused }
        return .matrixUnits
    }
}
