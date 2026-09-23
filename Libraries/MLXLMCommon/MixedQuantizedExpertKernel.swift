// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

final class MixedQuantizedExpertKernel {
    let affine: MLXFast.MLXFastKernel
    let mxfp4: MLXFast.MLXFastKernel
    let pairedAffine: MLXFast.MLXFastKernel
    let pairedMXFP4: MLXFast.MLXFastKernel
    let fusedAffine: MLXFast.MLXFastKernel
    let fusedMXFP4: MLXFast.MLXFastKernel
    let downReduce: MLXFast.MLXFastKernel
    init() {
        func build(_ fpMode: Bool, header: String) -> MLXFast.MLXFastKernel {
            var names = ["x"]
            for i in 0..<8 { names += ["w\(i)","s\(i)"]; if !fpMode { names.append("b\(i)") } }
            let selectW = (0..<7).map { "r == \($0) ? w\($0) : " }.joined()+"w7"
            let selectS = (0..<7).map { "r == \($0) ? s\($0) : " }.joined()+"s7"
            let selectB = (0..<7).map { "r == \($0) ? b\($0) : " }.joined()+"b7"
            let helper = fpMode ? "fp_qmv_fast_impl" : "qmv_fast_impl"
            let source = """
                uint r=threadgroup_position_in_grid.z;
                const device uint* w=\(selectW);
                const device \(fpMode ? "uchar" : "T")* s=\(selectS);
                \(fpMode ? "" : "const device T* b="+selectB+";")
                \(helper)<T,GROUP_SIZE,BITS>(w,s,\(fpMode ? "nullptr" : "b"),
                    x + (PER_EXPERT_INPUT ? r*IN_DIM : 0),out+r*OUT_DIM,
                    IN_DIM,OUT_DIM,uint3(0,threadgroup_position_in_grid.y,0),
                    simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
                """
            return MLXFast.metalKernel(name:fpMode ? "mimo_region_mxfp4_e8" : "mimo_region_affine_e8",
                inputNames:names,outputNames:["out"],source:source,
                header:header.replacingOccurrences(of:"const constant int&",with:"const int"),ensureRowContiguous:false)
        }
        affine = build(false,header:MixedQuantizedExpertKernelSource.affine)
        mxfp4 = build(true,header:MixedQuantizedExpertKernelSource.mxfp4)
        func paired(_ mxGate: Bool) -> MLXFast.MLXFastKernel {
            var names = ["x", "indices", "gw", "gs"]
            if !mxGate { names.append("gb") }
            names += ["uw", "us", "ub"]
            let source = """
                uint r = threadgroup_position_in_grid.z;
                uint e = indices[r];
                uint3 row(0, threadgroup_position_in_grid.y, 0);
                if (threadgroup_position_in_grid.x == 0) {
                    \(mxGate ? "mx::fp_qmv_fast_impl" : "aff::qmv_fast_impl")<T,G_GROUP,G_BITS>(
                        gw + e * OUT_DIM * (IN_DIM / (32 / G_BITS)),
                        gs + e * OUT_DIM * (IN_DIM / G_GROUP),
                        \(mxGate ? "nullptr" : "gb + e * OUT_DIM * (IN_DIM / G_GROUP)"),
                        x, gate + r * OUT_DIM, IN_DIM, OUT_DIM, row,
                        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
                } else {
                    aff::qmv_fast_impl<T,U_GROUP,U_BITS>(
                        uw + e * OUT_DIM * (IN_DIM / (32 / U_BITS)),
                        us + e * OUT_DIM * (IN_DIM / U_GROUP),
                        ub + e * OUT_DIM * (IN_DIM / U_GROUP),
                        x, up + r * OUT_DIM, IN_DIM, OUT_DIM, row,
                        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
                }
                """
            var header = "namespace aff {\n" + MixedQuantizedExpertKernelSource.affine + "\n}\n"
            if mxGate { header += "namespace mx {\n" + MixedQuantizedExpertKernelSource.mxfp4 + "\n}\n" }
            return MLXFast.metalKernel(name: mxGate ? "mimo_resident_gate_up_mxfp4" : "mimo_resident_gate_up_affine",
                inputNames: names, outputNames: ["gate", "up"], source: source,
                header: header.replacingOccurrences(of: "const constant int&", with: "const int"),
                ensureRowContiguous: false)
        }
        pairedAffine = paired(false)
        pairedMXFP4 = paired(true)
        func fused(_ mxGate: Bool) -> MLXFast.MLXFastKernel {
            var names = ["x", "indices", "gw", "gs"]
            if !mxGate { names.append("gb") }
            names += ["uw", "us", "ub"]
            // Preserve each native QMV's reduction order and BF16 boundaries.
            // Each SIMD group computes four matching gate/up rows, sharing x.
            let source = """
                uint r = threadgroup_position_in_grid.z;
                uint e = indices[r];
                uint lane = thread_index_in_simdgroup;
                uint row = threadgroup_position_in_grid.y * 8 + simdgroup_index_in_threadgroup * 4;
                const device uchar* g = (const device uchar*)gw + (e * OUT_DIM + row) * (IN_DIM * G_BITS / 8) + lane * (16 * G_BITS / 8);
                const device uchar* u = (const device uchar*)uw + (e * OUT_DIM + row) * (IN_DIM / 4) + lane * 4;
                uint gi = (e * OUT_DIM + row) * (IN_DIM / G_GROUP) + lane / (G_GROUP / 16);
                uint ui = (e * OUT_DIM + row) * (IN_DIM / U_GROUP) + lane / (U_GROUP / 16);
                float gr[4] = {0}, ur[4] = {0};
                for (int k = 0; k < IN_DIM; k += 512) {
                    float xa[16];
                    float sum = aff::load_vector<T, float, 16, 2>(x + k + lane * 16, xa);
                    \(mxGate ? "float xm[16]; mx::load_vector<T, float, 16>(x + k + lane * 16, xm);" : "")
                    for (int j = 0; j < 4; ++j) {
                        uint gsi = gi + j * (IN_DIM / G_GROUP);
                        uint usi = ui + j * (IN_DIM / U_GROUP);
                        \(mxGate
                          ? "gr[j] += mx::qdot<float, 16, 4>(g + j * (IN_DIM / 2), xm, mx::dequantize_scale<float, G_GROUP>(gs[gsi]));"
                          : "gr[j] += aff::qdot<float, 16, 2>(g + j * (IN_DIM / 4), xa, float(gs[gsi]), float(gb[gsi]), sum);")
                        ur[j] += aff::qdot<float, 16, 2>(u + j * (IN_DIM / 4), xa, float(us[usi]), float(ub[usi]), sum);
                    }
                    g += 512 * G_BITS / 8; u += 128;
                    gi += 512 / G_GROUP; ui += 512 / U_GROUP;
                }
                for (int j = 0; j < 4; ++j) {
                    float gateSum = simd_sum(gr[j]);
                    float upSum = simd_sum(ur[j]);
                    if (lane == 0) {
                        T gate = T(gateSum), up = T(upSum);
                        auto y = 1 / (1 + metal::exp(metal::abs(gate)));
                        T sigmoid = (gate < 0) ? y : 1 - y;
                        T activated = gate * sigmoid;
                        out[r * OUT_DIM + row + j] = T(activated * up);
                    }
                }
                """
            var header = "namespace aff {\n" + MixedQuantizedExpertKernelSource.affine + "\n}\n"
            if mxGate { header += "namespace mx {\n" + MixedQuantizedExpertKernelSource.mxfp4 + "\n}\n" }
            return MLXFast.metalKernel(name: mxGate ? "mimo_fused_swiglu_mxfp4" : "mimo_fused_swiglu_affine",
                inputNames: names, outputNames: ["out"], source: source,
                header: header.replacingOccurrences(of: "const constant int&", with: "const int"),
                ensureRowContiguous: false)
        }
        fusedAffine = fused(false)
        fusedMXFP4 = fused(true)
        downReduce = {
        let source = """
            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;
            uint r = sg / 2;
            uint e = indices[r];
            uint localRow = (sg % 2) * 4;
            uint row = threadgroup_position_in_grid.y * 8 + localRow;
            const device uchar* w = (const device uchar*)weight
                + (e * OUT_DIM + row) * (IN_DIM / 4) + lane * 4;
            uint si = (e * OUT_DIM + row) * (IN_DIM / 64) + lane / 4;
            float result[4] = {0};
            for (int k = 0; k < IN_DIM; k += 512) {
                float xv[16];
                float sx = aff::load_vector<T, float, 16, 2>(x + r * IN_DIM + k + lane * 16, xv);
                for (int j = 0; j < 4; ++j) {
                    uint scaleIndex = si + j * (IN_DIM / 64);
                    result[j] += aff::qdot<float, 16, 2>(w + j * (IN_DIM / 4), xv,
                        float(scales[scaleIndex]), float(biases[scaleIndex]), sx);
                }
                w += 128;
                si += 8;
            }
            threadgroup float weighted[64];
            for (int j = 0; j < 4; ++j) {
                float sum = simd_sum(result[j]);
                if (lane == 0) {
                    // Native gathered QMV stores BF16 before FP32 weighting.
                    weighted[r * 8 + localRow + j] = float(T(sum)) * scores[r];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0 && lane < 8) {
                // Match the eight-row col_reduce_small sequential reduction.
                float sum = weighted[lane] + 0.0f;
                for (int route = 1; route < 8; ++route) {
                    sum = (weighted[route * 8 + lane] + 0.0f) + sum;
                }
                out[threadgroup_position_in_grid.y * 8 + lane] = T(sum);
            }
            """
        let header = ("namespace aff {\n" + MixedQuantizedExpertKernelSource.affine + "\n}")
            .replacingOccurrences(of: "const constant int&", with: "const int")
        return MLXFast.metalKernel(name: "mimo_down_weight_reduce_affine2",
            inputNames: ["x", "indices", "weight", "scales", "biases", "scores"],
            outputNames: ["out"], source: source, header: header, ensureRowContiguous: false)
        }()
    }

    /// Native BF16 down projection, FP32 route weighting/reduction, BF16 output.
    /// This narrow single-token path preserves the native packed two-bit bank.
    func fusedDownReduce(_ x: MLXArray, indices: MLXArray, scores: MLXArray,
                         down: MixedQuantizedExpertCatalog.Projection) -> MLXArray? {
        guard down.weight.ndim == 3, down.weight.dtype == .uint32,
            down.bits == 2, down.groupSize == 64, down.mode == .affine,
            down.scales.dtype == .bfloat16, let biases = down.biases,
            biases.dtype == .bfloat16, x.dtype == .bfloat16,
            indices.dtype == .int32 || indices.dtype == .uint32,
            indices.size == 8, scores.shape == indices.shape,
            scores.dtype == .float32 else { return nil }
        let input = down.weight.dim(-1) * 16
        let output = down.weight.dim(-2)
        guard input > 0, input.isMultiple(of: 512), output > 0,
            output.isMultiple(of: 8), x.size == 8 * input,
            x.dim(-1) == input,
            down.scales.shape == [down.weight.dim(0), output, input / 64],
            biases.shape == down.scales.shape else { return nil }
        return downReduce([contiguous(x), contiguous(indices), down.weight,
                           down.scales, biases, contiguous(scores)],
            template: [("T", DType.bfloat16), ("IN_DIM", input), ("OUT_DIM", output)],
            grid: (32, 16 * (output / 8), 1), threadGroup: (32, 16, 1),
            outputShapes: [Array(indices.shape.dropLast()) + [output]],
            outputDTypes: [.bfloat16])[0]
    }

    /// Fuses the two packed projections and SwiGLU without materializing gate/up.
    /// Narrowly restricted to the bundle's native two-bit affine / MXFP4 modes.
    func fusedGateUp(_ x: MLXArray, indices: MLXArray,
                     gate: MixedQuantizedExpertCatalog.Projection,
                     up: MixedQuantizedExpertCatalog.Projection) -> MLXArray? {
        let input = gate.weight.dim(-1) * 32 / gate.bits
        let output = gate.weight.dim(-2)
        let mx = gate.mode == .mxfp4
        guard x.size == input, x.dtype == .bfloat16, indices.size == 8,
            input.isMultiple(of: 512), output.isMultiple(of: 8),
            up.weight.dim(-1) * 16 == input, up.weight.dim(-2) == output,
            up.bits == 2, up.mode == .affine, [64, 128].contains(up.groupSize),
            up.scales.dtype == .bfloat16, let bias = up.biases, bias.dtype == .bfloat16,
            (mx && gate.bits == 4 && gate.groupSize == 32 && gate.scales.dtype == .uint8 && gate.biases == nil)
                || (gate.mode == .affine && gate.bits == 2 && gate.groupSize == 64
                    && gate.scales.dtype == .bfloat16 && gate.biases?.dtype == .bfloat16)
        else { return nil }
        var arrays = [contiguous(x), contiguous(indices), gate.weight, gate.scales]
        if let bias = gate.biases { arrays.append(bias) }
        arrays += [up.weight, up.scales, bias]
        return (mx ? fusedMXFP4 : fusedAffine)(arrays,
            template: [("T", DType.bfloat16), ("IN_DIM", input), ("OUT_DIM", output),
                ("G_BITS", gate.bits), ("G_GROUP", gate.groupSize), ("U_GROUP", up.groupSize)],
            grid: (32, output / 8 * 2, 8), threadGroup: (32, 2, 1),
            outputShapes: [Array(x.shape.dropLast()) + [8, 1, output]], outputDTypes: [.bfloat16])[0]
    }

    /// One dispatch for two native projections, with GPU-resident route indices.
    /// Keeps each projection's packed representation and BF16 rounding intact.
    func pairedGateUp(_ x: MLXArray, indices: MLXArray,
                      gate: MixedQuantizedExpertCatalog.Projection,
                      up: MixedQuantizedExpertCatalog.Projection) -> [MLXArray]? {
        let input = gate.weight.dim(-1) * 32 / gate.bits
        let output = gate.weight.dim(-2)
        guard x.size == input, x.dtype == .bfloat16, indices.size == 8,
            input.isMultiple(of: 512), output.isMultiple(of: 8),
            up.weight.dim(-1) * 32 / up.bits == input, up.weight.dim(-2) == output,
            [2, 4, 8].contains(gate.bits), [2, 4, 8].contains(up.bits),
            up.mode == .affine, up.scales.dtype == .bfloat16,
            let upBias = up.biases, upBias.dtype == .bfloat16 else { return nil }
        let mx = gate.mode == .mxfp4
        guard (mx && gate.bits == 4 && gate.scales.dtype == .uint8 && gate.biases == nil)
            || (gate.mode == .affine && gate.scales.dtype == .bfloat16 && gate.biases?.dtype == .bfloat16)
        else { return nil }
        var arrays = [contiguous(x), contiguous(indices), gate.weight, gate.scales]
        if let bias = gate.biases { arrays.append(bias) }
        arrays += [up.weight, up.scales, upBias]
        let shape = Array(x.shape.dropLast()) + [8, 1, output]
        return (mx ? pairedMXFP4 : pairedAffine)(arrays,
            template: [("T", DType.bfloat16), ("IN_DIM", input), ("OUT_DIM", output),
                ("G_BITS", gate.bits), ("G_GROUP", gate.groupSize),
                ("U_BITS", up.bits), ("U_GROUP", up.groupSize)],
            grid: (64, output / 8 * 2, 8), threadGroup: (32, 2, 1),
            outputShapes: [shape, shape], outputDTypes: [.bfloat16, .bfloat16])
    }
    func call(_ x: MLXArray, weights: [(MLXArray,MLXArray,MLXArray?)], bits: Int, group: Int,
              mode: QuantizationMode, perExpertInput: Bool) -> MLXArray {
        precondition(weights.count == 8 && x.dtype == .bfloat16)
        let input = weights[0].0.dim(1)*32/bits, output = weights[0].0.dim(0)
        var arrays = [x]
        for (w,s,b) in weights { arrays += [w,s]; if let b { arrays.append(b) } }
        return (mode == .mxfp4 ? mxfp4 : affine)(arrays,
            template:[("T",DType.bfloat16),("GROUP_SIZE",group),("BITS",bits),
                ("IN_DIM",input),("OUT_DIM",output),("PER_EXPERT_INPUT",perExpertInput)],
            grid:(32,output/8*2,8),threadGroup:(32,2,1),outputShapes:[[8,1,output]],outputDTypes:[.bfloat16])[0]
    }
}
