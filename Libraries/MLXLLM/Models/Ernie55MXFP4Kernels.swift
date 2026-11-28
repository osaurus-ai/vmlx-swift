// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast
import MLXLMCommon

/// Routed MoE expert interface shared by JANGT/JANGH banks (`ERNIE-5.5-Flash-Preview-JANGHT2.4`)
/// and native OCP MXFP4 expert banks (`ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4`).
protocol Ernie55RoutedExperts: AnyObject {
    func callRouted(
        _ latent: MLXArray, indices: MLXArray, scores: MLXArray, caps: (gate: Float, up: Float)
    ) -> MLXArray
}

extension JANGTRoutedExpertLayer: Ernie55RoutedExperts {
    func callRouted(
        _ latent: MLXArray, indices: MLXArray, scores: MLXArray, caps: (gate: Float, up: Float)
    ) -> MLXArray {
        if let prefill = routedW4A8Prefill(
            latent, indices: indices, scores: scores,
            roundActivations: Ernie55Math.roundExpertActivations,
            applyActivation: { Ernie55Math.situ($0, $1, gateCap: caps.gate, upCap: caps.up) })
        {
            return prefill
        }
        return routedW4A8Decode(
            latent, indices: indices, scores: scores,
            roundActivations: Ernie55Math.roundExpertActivations)
    }
}

/// Native OCP MXFP4 routed expert bank (`groupSize = 32`, `bits = 4`, E2M1 nibbles packed in
/// `[E, N, K/8]` `uint32` + E8M0 exponent scales in `[E, N, K/32]` `uint8`).
///
/// - Decode & DSpark verify (`T = 1...16`): runs two custom megakernels:
///   1. Split-simdgroup fused `gate(w1) + up(w3) + SiTU` (`sg0=gate`, `sg1=up` via 16 B TG
///      `g_sh[4]`, 16-byte coalesced `uint4` weight loads, zero intermediate gate/up tensors).
///   2. Parallel-slot weighted `down(w2)` (`pSG = 5` for 10 routes, accumulates router-weighted
///      outputs directly into `[T, H]` without materializing `[T, slots, H]`).
/// - Prefill (`T > 16`): sorts `(token, slot)` pairs by expert ID and runs MLX's native NAX
///   MXFP4 `gatherQuantizedMM(..., sortedIndices: true)` with separate `gate` and `up` passes
///   for maximum cold-DRAM occupancy, followed by fused `Ernie55Math.situ` and sorted `down`.
final class Ernie55MXFP4RoutedExpertLayer: Ernie55RoutedExperts, @unchecked Sendable {
    let gateWeight: MLXArray   // [E, N, K/8] uint32
    let gateScales: MLXArray   // [E, N, K/32] uint8
    let upWeight: MLXArray     // [E, N, K/8] uint32
    let upScales: MLXArray     // [E, N, K/32] uint8
    let downWeight: MLXArray   // [E, H, N/8] uint32
    let downScales: MLXArray   // [E, H, N/32] uint8
    let hiddenSize: Int
    let intermediateSize: Int

    init(
        gateWeight: MLXArray, gateScales: MLXArray,
        upWeight: MLXArray, upScales: MLXArray,
        downWeight: MLXArray, downScales: MLXArray,
        hiddenSize: Int, intermediateSize: Int
    ) {
        self.gateWeight = gateWeight
        self.gateScales = gateScales
        self.upWeight = upWeight
        self.upScales = upScales
        self.downWeight = downWeight
        self.downScales = downScales
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
    }

    func callRouted(
        _ latent: MLXArray, indices: MLXArray, scores: MLXArray, caps: (gate: Float, up: Float)
    ) -> MLXArray {
        let lead = Array(latent.shape.dropLast())
        let tokens = latent.size / hiddenSize
        let slots = indices.dim(-1)
        let x2 = contiguous(latent.reshaped(tokens, hiddenSize))
        let idx2 = contiguous(indices.reshaped(tokens, slots).asType(.uint32))
        let sc2 = contiguous(scores.reshaped(tokens, slots).asType(.float32))

        if tokens <= 16,
            let fused = Ernie55MXFP4Kernels.fusedDecodeVerify(
                x2, indices: idx2, scores: sc2,
                gateW: gateWeight, gateS: gateScales,
                upW: upWeight, upS: upScales,
                downW: downWeight, downS: downScales,
                gateCap: caps.gate, upCap: caps.up)
        {
            return fused.reshaped(lead + [hiddenSize])
        }

        // Prefill (or CPU / fallback): sort pairs by expert for native NAX MXFP4 gatherQuantizedMM.
        let totalPairs = tokens * slots
        let flatIds = idx2.reshaped(totalPairs)
        let order = argSort(flatIds, axis: 0).asType(.uint32)
        let sortedIds = flatIds[order]
        let tokenIdx = floorDivide(order, MLXArray(UInt32(slots))).asType(.uint32)
        let xSorted = x2[tokenIdx].reshaped(totalPairs, 1, hiddenSize)

        let g = gatherQuantizedMM(
            xSorted, gateWeight, scales: gateScales, biases: nil,
            rhsIndices: sortedIds, transpose: true, groupSize: 32, bits: 4, mode: .mxfp4,
            sortedIndices: true)
        let u = gatherQuantizedMM(
            xSorted, upWeight, scales: upScales, biases: nil,
            rhsIndices: sortedIds, transpose: true, groupSize: 32, bits: 4, mode: .mxfp4,
            sortedIndices: true)
        let act = Ernie55Math.situ(g, u, gateCap: caps.gate, upCap: caps.up)
        let downSorted = gatherQuantizedMM(
            act, downWeight, scales: downScales, biases: nil,
            rhsIndices: sortedIds, transpose: true, groupSize: 32, bits: 4, mode: .mxfp4,
            sortedIndices: true).reshaped(totalPairs, hiddenSize)
        let invOrder = argSort(order, axis: 0).asType(.uint32)
        let down = downSorted[invOrder].reshaped(tokens, slots, hiddenSize)
        let weighted = (down.asType(.float32) * sc2[.ellipsis, .newAxis]).sum(axis: 1)
        return weighted.asType(latent.dtype).reshaped(lead + [hiddenSize])
    }
}

enum Ernie55MXFP4Kernels {
    static let enabled = ProcessInfo.processInfo.environment["VMLX_ERNIE_MXFP4_FUSED"] != "0"

    #if canImport(Metal)
    private static let fp4Header = """
        // OCP FP4 E2M1 16-entry lookup table (exact).
        constant float FP4_LUT[16] = {
             0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
        };
        static inline float e8m0_scale(uint s) {
            return as_type<float>(s == 0u ? 0x00400000u : (s << 23u));
        }
        template <typename T>
        static inline float fp4_dot32_regs(const uint4 p, const thread float4* xv) {
            float acc = 0.0f;
            #pragma clang loop unroll(full)
            for (uint w = 0u; w < 4u; ++w) {
                const uint u = (w == 0u) ? p.x : (w == 1u) ? p.y : (w == 2u) ? p.z : p.w;
                float4 f0 = float4(
                    FP4_LUT[u & 15u],
                    FP4_LUT[(u >> 4u) & 15u],
                    FP4_LUT[(u >> 8u) & 15u],
                    FP4_LUT[(u >> 12u) & 15u]);
                float4 f1 = float4(
                    FP4_LUT[(u >> 16u) & 15u],
                    FP4_LUT[(u >> 20u) & 15u],
                    FP4_LUT[(u >> 24u) & 15u],
                    FP4_LUT[(u >> 28u) & 15u]);
                acc += dot(xv[2u * w], f0) + dot(xv[2u * w + 1u], f1);
            }
            return acc;
        }
        template <typename T>
        static inline void load_x32(const device T* xp, thread float4* xv) {
            #pragma clang loop unroll(full)
            for (uint w = 0u; w < 4u; ++w) {
                const device T* xw = xp + w * 8u;
                xv[2u * w] = float4(float(xw[0]), float(xw[1]), float(xw[2]), float(xw[3]));
                xv[2u * w + 1u] = float4(float(xw[4]), float(xw[5]), float(xw[6]), float(xw[7]));
            }
        }
        """

    /// Split-simdgroup fused MXFP4 gate(w1) + up(w3) + Ernie55 SiTU.
    /// Grid: (64, (N + 3) / 4, T * SLOTS), TG: (64, 1, 1).
    /// sg0 computes gate (4 rows), sg1 computes up (4 rows); 16 B threadgroup handoff.
    private static let gateUpSiTUKernel = MLXFast.metalKernel(
        name: "ernie55_mxfp4_gate_up_situ_sg",
        inputNames: ["x", "indices", "gw", "gs", "uw", "us", "caps"],
        outputNames: ["out"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup; // 0 = gate, 1 = up
            const uint row0 = (YZ ? threadgroup_position_in_grid.z : threadgroup_position_in_grid.y) * 4u;
            const uint pair = YZ ? threadgroup_position_in_grid.y : threadgroup_position_in_grid.z;
            const uint tok = pair / uint(SLOTS);
            const uint expert = indices[pair];
            const uint groups = uint(K) / 32u;
            const uint k_u32 = uint(K) / 8u;

            const device uint* wbank = (sg == 0u ? gw : uw) + size_t(expert) * uint(N) * k_u32;
            const device uint8_t* sbank = (sg == 0u ? gs : us) + size_t(expert) * uint(N) * groups;
            const device T* xrow = x + size_t(tok) * uint(K);

            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint g = lane; g < groups; g += 32u) {
                float4 xv[8];
                load_x32<T>(xrow + g * 32u, xv);
                #pragma clang loop unroll(full)
                for (uint r = 0u; r < 4u; ++r) {
                    const uint row = row0 + r;
                    if (row < uint(N)) {
                        const uint4 p = *reinterpret_cast<const device uint4*>(wbank + size_t(row) * k_u32 + g * 4u);
                        const float sf = e8m0_scale(sbank[size_t(row) * groups + g]);
                        acc[r] += fp4_dot32_regs<T>(p, xv) * sf;
                    }
                }
            }

            threadgroup float g_sh[4];
            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 4u; ++r) {
                acc[r] = simd_sum(acc[r]);
                if (sg == 0u && lane == 0u) g_sh[r] = acc[r];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sg == 1u && lane == 0u) {
                const float gc = caps[0];
                const float uc = caps[1];
                device T* orow = out + size_t(pair) * uint(N);
                #pragma clang loop unroll(full)
                for (uint r = 0u; r < 4u; ++r) {
                    const uint row = row0 + r;
                    if (row < uint(N)) {
                        const float gv = g_sh[r];
                        const float uv = acc[r];
                        const float gAct = gc * metal::precise::tanh(gv / gc)
                            * (1.0f / (1.0f + metal::precise::exp(-gv)));
                        const float uAct = uc * metal::precise::tanh(uv / uc);
                        orow[row] = T(gAct * uAct);
                    }
                }
            }
            """,
        header: fp4Header)

    /// Parallel-slot weighted MXFP4 down projection.
    /// Grid: (P_SG * 32, (H + 3) / 4, T), TG: (P_SG * 32, 1, 1).
    /// Each simdgroup handles slots `sg, sg + P_SG, ...` and combines into `out[t, row]`.
    private static let downWeightedKernel = MLXFast.metalKernel(
        name: "ernie55_mxfp4_down_weighted_p",
        inputNames: ["act", "indices", "scores", "dw", "ds"],
        outputNames: ["out"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint row0 = (YZ ? threadgroup_position_in_grid.z : threadgroup_position_in_grid.y) * 4u;
            const uint tok = YZ ? threadgroup_position_in_grid.y : threadgroup_position_in_grid.z;
            const uint groups = uint(K) / 32u;
            const uint k_u32 = uint(K) / 8u;

            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint s = sg; s < uint(SLOTS); s += uint(P_SG)) {
                const uint pair = tok * uint(SLOTS) + s;
                const uint expert = indices[pair];
                const float wt = scores[pair];
                const device T* xrow = act + size_t(pair) * uint(K);
                const device uint* wbank = dw + size_t(expert) * uint(H) * k_u32;
                const device uint8_t* sbank = ds + size_t(expert) * uint(H) * groups;

                float slot_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (uint g = lane; g < groups; g += 32u) {
                    float4 xv[8];
                    load_x32<T>(xrow + g * 32u, xv);
                    #pragma clang loop unroll(full)
                    for (uint r = 0u; r < 4u; ++r) {
                        const uint row = row0 + r;
                        if (row < uint(H)) {
                            const uint4 p = *reinterpret_cast<const device uint4*>(wbank + size_t(row) * k_u32 + g * 4u);
                            const float sf = e8m0_scale(sbank[size_t(row) * groups + g]);
                            slot_acc[r] += fp4_dot32_regs<T>(p, xv) * sf;
                        }
                    }
                }
                #pragma clang loop unroll(full)
                for (uint r = 0u; r < 4u; ++r) {
                    acc[r] += simd_sum(slot_acc[r]) * wt;
                }
            }

            threadgroup float sg_sh[8][4];
            if (lane == 0u) {
                #pragma clang loop unroll(full)
                for (uint r = 0u; r < 4u; ++r) sg_sh[sg][r] = acc[r];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sg == 0u && lane == 0u) {
                device T* orow = out + size_t(tok) * uint(H);
                #pragma clang loop unroll(full)
                for (uint r = 0u; r < 4u; ++r) {
                    const uint row = row0 + r;
                    if (row < uint(H)) {
                        float total = 0.0f;
                        #pragma clang loop unroll(full)
                        for (uint p = 0u; p < uint(P_SG); ++p) total += sg_sh[p][r];
                        orow[row] = T(total);
                    }
                }
            }
            """,
        header: fp4Header)
    #endif

    static func fusedDecodeVerify(
        _ x: MLXArray, indices: MLXArray, scores: MLXArray,
        gateW: MLXArray, gateS: MLXArray,
        upW: MLXArray, upS: MLXArray,
        downW: MLXArray, downS: MLXArray,
        gateCap: Float, upCap: Float
    ) -> MLXArray? {
        #if canImport(Metal)
        guard enabled, Device.defaultDevice().deviceType == .gpu,
            x.ndim == 2, indices.ndim == 2, scores.ndim == 2,
            x.dtype == .bfloat16 || x.dtype == .float16
        else { return nil }
        let tokens = x.dim(0)
        let k = x.dim(1)
        let slots = indices.dim(1)
        let n = gateW.dim(1)
        let h = downW.dim(1)
        guard tokens >= 1, tokens <= 16, slots >= 1,
            k % 128 == 0, n % 128 == 0, h % 4 == 0,
            gateW.dtype == .uint32, gateS.dtype == .uint8,
            upW.dtype == .uint32, upS.dtype == .uint8,
            downW.dtype == .uint32, downS.dtype == .uint8
        else { return nil }

        let capsArr = MLXArray([gateCap, upCap], [2])
        let flatIdx = indices.reshaped(tokens * slots)
        let flatScores = scores.reshaped(tokens * slots)
        let yz = tokens > 1
        let guTiles = (n + 3) / 4
        let guGrid = yz ? (64, tokens * slots, guTiles) : (64, guTiles, tokens * slots)
        let act = gateUpSiTUKernel(
            [x, flatIdx, gateW, gateS, upW, upS, capsArr],
            template: [("T", x.dtype), ("K", k), ("N", n), ("SLOTS", slots), ("YZ", yz)],
            grid: guGrid,
            threadGroup: (64, 1, 1),
            outputShapes: [[tokens * slots, n]],
            outputDTypes: [x.dtype])[0]

        let pSG: Int
        if slots % 5 == 0 {
            pSG = 5
        } else if slots % 4 == 0 {
            pSG = 4
        } else {
            pSG = min(slots, 4)
        }
        let dnTiles = (h + 3) / 4
        let dnGrid = yz ? (pSG * 32, tokens, dnTiles) : (pSG * 32, dnTiles, tokens)
        return downWeightedKernel(
            [act, flatIdx, flatScores, downW, downS],
            template: [("T", x.dtype), ("K", n), ("H", h), ("SLOTS", slots), ("P_SG", pSG), ("YZ", yz)],
            grid: dnGrid,
            threadGroup: (pSG * 32, 1, 1),
            outputShapes: [[tokens, h]],
            outputDTypes: [x.dtype])[0]
        #else
        return nil
        #endif
    }
}
