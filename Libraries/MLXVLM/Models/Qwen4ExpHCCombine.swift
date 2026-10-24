// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// Single-row HC residual combine. The product must round to the activation
/// dtype BEFORE adding the residual, as it does in the two-operation graph.
/// This is deliberately not a fused multiply-add. Quantization is irrelevant:
/// the dimensions and dtype come from this layer's actual activation tensors.
enum Qwen4ExpHCCombine {
    // Preserve an explicit opt-out (including malformed diagnostic values).
    // Admission below still comes from the actual activation shape and dtype.
    static func parse(_ value: String?) -> Bool { value == nil || value == "1" }

    private static let enabled = parse(
        RuntimeEnvironment.value("VMLX_QWEN4_EXACT_HC_COMBINE"))

    #if canImport(Metal)
    private static let kernel = MLXFast.metalKernel(
        name: "vmlx_qwen4_exact_hc_combine",
        inputNames: ["residual", "block", "inject"], outputNames: ["output"],
        source: """
            {
            #pragma clang fp contract(off)
            #pragma clang fp reassociate(off)
            uint i = thread_position_in_grid.x;
            if (i >= STREAMS * HIDDEN) return;
            uint stream = i / HIDDEN;
            uint feature = i % HIDDEN;
            T product = T(float(block[feature]) * float(inject[stream]));
            output[i] = T(float(residual[i]) + float(product));
            }
            """)

    private static let report: Void = {
        RuntimeDiagnostics.write(Data(
            "[Qwen4Exp] exact_hc_combine=active product_rounding=preserved fma=off\n".utf8))
    }()
    #endif

    static func call(
        residual: MLXArray, block: MLXArray, injection: MLXArray,
        enabled override: Bool? = nil
    ) -> MLXArray? {
        #if canImport(Metal)
        guard override ?? enabled, !CompiledDecodeTrace.isActive,
            residual.ndim == 3, block.ndim == 3, injection.ndim == 3,
            residual.dim(0) == 1, residual.dim(1) == 1,
            block.dim(0) == 1, block.dim(1) == 1,
            injection.dim(0) == 1, injection.dim(1) == 1,
            residual.dtype == .float16 || residual.dtype == .bfloat16
                || residual.dtype == .float32,
            block.dtype == residual.dtype, injection.dtype == residual.dtype,
            Device.defaultDevice().deviceType == .gpu
        else { return nil }
        let streams = injection.dim(2), hidden = block.dim(2)
        guard streams > 0, hidden > 0, residual.dim(2) == streams * hidden else { return nil }
        let result = kernel(
            [residual, block, injection],
            template: [("T", residual.dtype), ("STREAMS", streams), ("HIDDEN", hidden)],
            grid: (streams * hidden, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [residual.shape], outputDTypes: [residual.dtype])[0]
        _ = report
        return result
        #else
        return nil
        #endif
    }
}
