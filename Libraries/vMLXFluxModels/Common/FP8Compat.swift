@preconcurrency import MLX

func mfluxFromFP8(
    _ x: MLXArray,
    dtype: DType = .bfloat16,
    stream: StreamOrDevice = .default
) -> MLXArray {
    fromFP8(x, dtype: dtype, stream: stream)
}
