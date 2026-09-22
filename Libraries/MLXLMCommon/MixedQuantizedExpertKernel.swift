// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

final class MixedQuantizedExpertKernel {
    let affine: MLXFast.MLXFastKernel
    let mxfp4: MLXFast.MLXFastKernel
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

