// Naive-N0.5 reference math. Vendor revision 0235b3b5; no model registration.
import Foundation
import MLX

/// Vendor reference operations retained beside narrowly admitted optimized paths.
/// Asymmetric prefill remains explicit until a supported fused kernel is proven.
enum NaiveN05FlashMath {
    static func allowedMaskGPUArangeRequested(environment: [String: String]) -> Bool {
        // Unset uses the model default. Exact 1 enables; 0 and unrecognized
        // values keep the reference path rather than accepting truthy spellings.
        guard let value = environment["VMLX_NAIVE_ALLOWED_MASK_GPU_ARANGE"] else { return true }
        return value == "1"
    }

    /// One forward's positions only. Reusing these lazy arrays across layers
    /// avoids rebuilding identical trigonometric graphs; nothing survives into
    /// the next decode step or into a restored KV cache.
    final class RotaryTables {
        private struct Key: Hashable {
            let dimensions: Int
            let theta: Double
            let dtype: DType
        }
        private let positions: MLXArray
        let fusedApply: Bool
        private var values: [Key: (MLXArray, MLXArray)] = [:]
        var count: Int { values.count }
        init(positions: MLXArray, fusedApply: Bool = false) {
            self.positions = positions
            self.fusedApply = fusedApply
        }
        func phases(dimensions: Int, theta: Double, dtype: DType) -> (MLXArray, MLXArray) {
            let key = Key(dimensions: dimensions, theta: theta, dtype: dtype)
            if let found = values[key] { return found }
            let result = NaiveN05FlashMath.rotaryPhases(
                positions: positions, dimensions: dimensions, theta: theta, dtype: dtype)
            values[key] = result
            return result
        }
    }

    static func roundIndexerFP8(_ input: MLXArray) -> MLXArray {
        let x = input.asType(.float32)
        let scale = maximum(abs(x).max(axis: -1, keepDims: true), 1e-4) / 448
        let normalized = clip(x / scale, min: -448, max: 448)
        // E4M3FN subnormal spacing is 2^-9; normals have three mantissa bits.
        let exponent = maximum(floor(log2(maximum(abs(normalized), 1.0 / 512))), -6)
        let spacing = pow(MLXArray(Float(2)), exponent - 3)
        return round(normalized / spacing) * spacing * scale
    }

    /// GPTNeoX half-rotation on only the prefix; positions may differ per batch.
    private static func rotaryPhases(positions: MLXArray, dimensions: Int, theta: Double, dtype: DType) -> (MLXArray, MLXArray) {
        let inverse = exp(-MLXArray(0 ..< dimensions / 2).asType(.float32)
            * (Float(2 * log(theta)) / Float(dimensions)))
        let angle = positions.asType(.float32).expandedDimensions(axis: -1) * inverse
        return (cos(angle).expandedDimensions(axis: 1).asType(dtype),
                sin(angle).expandedDimensions(axis: 1).asType(dtype))
    }

    static func rotary(_ x: MLXArray, positions: MLXArray, dimensions: Int, theta: Double,
                       tables: RotaryTables? = nil) -> MLXArray {
        precondition(dimensions > 0 && dimensions.isMultiple(of: 2) && dimensions <= x.dim(-1))
        let (c, s) = tables?.phases(dimensions: dimensions, theta: theta, dtype: x.dtype)
            ?? rotaryPhases(positions: positions, dimensions: dimensions, theta: theta, dtype: x.dtype)
        if tables?.fusedApply == true,
           let fused = NaiveN05FusedRotaryApply.apply(x, cosine: c, sine: s, dimensions: dimensions) {
            return fused
        }
        let a = x[.ellipsis, ..<(dimensions / 2)]
        let b = x[.ellipsis, (dimensions / 2)..<dimensions]
        let rotated = concatenated([a * c - b * s, b * c + a * s], axis: -1)
        return dimensions == x.dim(-1) ? rotated : concatenated([rotated, x[.ellipsis, dimensions...]], axis: -1)
    }

    /// Preserve the Sequence<Int> constructor's Int32 values without its two
    /// host arrays. Callers retain an explicit reference-path comparison policy.
    static func maskPositionRange(start: Int, count: Int, gpuArange: Bool) -> MLXArray {
        precondition(count >= 0)
        let stop = start + count
        // The Metal encoder casts start and start + step before subtraction.
        // Int32.max singletons and unsupported bounds retain the exact baseline
        // constructor, including its existing out-of-Int32 precondition.
        if gpuArange,
           count == 0 || (start >= Int(Int32.min) && start < Int(Int32.max)
                          && stop - 1 <= Int(Int32.max)) {
            return arange(start, stop, dtype: .int32)
        }
        return MLXArray(start ..< stop)
    }

    static func allowedMask(padding: MLXArray, queryOffset: Int, length: Int, keyOffset: Int, keyLength: Int, window: Int?, gpuArange: Bool = false) -> MLXArray {
        let q = maskPositionRange(start: queryOffset, count: length, gpuArange: gpuArange)
            .expandedDimensions(axis: -1)
        let k = maskPositionRange(start: keyOffset, count: keyLength, gpuArange: gpuArange)
        var mask = q .>= k
        if let window { mask = mask .&& (q - k .< window) }
        return mask.expandedDimensions(axis: 0) .&& padding[0..., keyOffset ..< keyOffset + keyLength].expandedDimensions(axis: 1)
    }

    /// MLX merge-sort preserves left input on equality. Descending order is
    /// obtained by negating scores rather than reversing sorted equal keys.
    static func sparseMask(scores: MLXArray, allowed: MLXArray, topK: Int) -> MLXArray {
        // Selecting every key cannot change the causal/padding mask. Avoid
        // evaluating indexer scores and sorting the full history in this case.
        // Indexer key insertion is owned by the caller and still advances.
        if scores.dim(-1) <= topK { return allowed }
        let masked = which(allowed, scores, MLXArray(-Float.infinity))
        let selected = argSort(-masked, axis: -1)[.ellipsis, ..<min(topK, scores.dim(-1))]
        let picked = putAlong(MLXArray.zeros(scores.shape, dtype: .bool), selected,
            values: MLXArray(true), axis: -1)
        return allowed .&& picked
    }

    /// The pinned Metal vector kernel supports asymmetric 192/128 heads only
    /// for short queries. Larger prefill retains the independent reference.
    static func usesAsymmetricDecodeSDPA(query: MLXArray, key: MLXArray, value: MLXArray) -> Bool {
        guard query.ndim == 4, key.ndim == 4, value.ndim == 4,
              query.dtype == key.dtype, query.dtype == value.dtype,
              [.float32, .float16, .bfloat16].contains(query.dtype),
              query.dim(-1) == 192, key.dim(-1) == 192, value.dim(-1) == 128,
              query.dim(0) == key.dim(0), key.dim(0) == value.dim(0),
              query.dim(1) > 0, key.dim(1) > 0, query.dim(1).isMultiple(of: key.dim(1)),
              key.dim(1) == value.dim(1), key.dim(2) == value.dim(2),
              query.dim(2) > 0, query.dim(2) <= 8, query.dim(2) <= key.dim(2)
        else { return false }
        return query.dim(2) <= 32 / (query.dim(1) / key.dim(1))
    }

    static func attention(query: MLXArray, key: MLXArray, value: MLXArray, allowed: MLXArray, sink: MLXArray?, valueScale: Float?) -> MLXArray {
        guard usesAsymmetricDecodeSDPA(query: query, key: key, value: value) else {
            return referenceAttention(query: query, key: key, value: value,
                                      allowed: allowed, sink: sink, valueScale: valueScale)
        }
        var v = value
        if let valueScale { v = v * MLXArray(valueScale, dtype: v.dtype) }
        let result = MLXFast.scaledDotProductAttention(
            queries: query, keys: key, values: v,
            scale: Float(1 / sqrt(Double(query.dim(-1)))),
            mask: allowed.expandedDimensions(axis: 1), sinks: sink?.asType(query.dtype))
        // The generic SDPA fallback substitutes a finite minimum for a false
        // bool mask, which would make an all-masked row average V without a
        // sink. Explicitly preserve the vendor's zero-output padding contract.
        let hasKey = allowed.any(axis: -1).expandedDimensions(axis: 1).expandedDimensions(axis: -1)
        return which(hasKey, result, MLXArray.zeros(result.shape, dtype: result.dtype))
    }

    static func referenceAttention(query: MLXArray, key: MLXArray, value: MLXArray, allowed: MLXArray, sink: MLXArray?, valueScale: Float?) -> MLXArray {
        let repeats = query.dim(1) / key.dim(1)
        let k = repeated(key, count: repeats, axis: 1)
        var v = repeated(value, count: repeats, axis: 1)
        if let valueScale { v = v * MLXArray(valueScale, dtype: v.dtype) }
        var logits = matmul(query, k.swappedAxes(-1, -2)) * Float(1 / sqrt(Double(query.dim(-1))))
        logits = which(allowed.expandedDimensions(axis: 1), logits, MLXArray(-Float.infinity, dtype: logits.dtype))
        if let sink {
            let column = broadcast(sink.asType(logits.dtype).reshaped(1, -1, 1, 1), to: [query.dim(0), query.dim(1), query.dim(2), 1])
            logits = concatenated([logits, column], axis: -1)
        }
        let probabilities = nanToNum(softmax(logits.asType(.float32), axis: -1), nan: 0)[.ellipsis, ..<key.dim(2)].asType(v.dtype)
        return matmul(probabilities, v)
    }
}
