import MLX

/// HC up projection only. Keep the original single-row GEMV and prefill path;
/// small verify slabs use independent GEMVs instead of row-count-dependent GEMM.
/// The bounded saved-operand admission covers dense BF16 [10240, 320] weights.
enum Qwen4ExpHCUpProjection {
    static func project(_ input: MLXArray, weight: MLXArray) -> MLXArray? {
        guard input.ndim == 3, input.dim(0) == 1,
            (2...8).contains(input.dim(1)), input.dim(2) == 320,
            input.dtype == .bfloat16, weight.dtype == .bfloat16,
            weight.shape == [10240, 320]
        else { return nil }
        let rows = input.dim(1)
        return matmul(
            broadcast(weight, to: [rows, 10240, 320]),
            input.reshaped(rows, 320, 1)
        ).reshaped(1, rows, 10240)
    }
}
