import Foundation
import MLX
import MLXNN
import os

/// Bonsai2 keeps its checkpoint weights and Hadamard/normalization arithmetic
/// unchanged. Only the attention boundary uses half-precision Q/K/V. Recurrent
/// GDN state is deliberately not part of this policy.
public enum JangHadamardAttention {
    /// Diagnostic control for matched runs, not a sampler or template override.
    public static let useFloat16 = !RuntimeEnvironment.flag("VMLX_BONSAI_ATTENTION_FP32")
    private static let reported = OSAllocatedUnfairLock(initialState: false)

    public static func applies(query: Linear, key: Linear, value: Linear) -> Bool {
        query is HadamardQuantizedLinear && key is HadamardQuantizedLinear
            && value is HadamardQuantizedLinear
    }

    /// Detect installed modules, not model names. This is computed once when a
    /// container takes ownership of a loaded model, for both caching entrypoints.
    public static func cacheKeyComponent(model: any LanguageModel) -> String? {
        guard model is any JangHadamardRuntimeModel,
            model.namedModules().contains(where: { $0.1 is HadamardQuantizedLinear })
        else { return nil }
        return useFloat16 ? "bonsai-attention-fp16-v1" : "bonsai-attention-fp32-v1"
    }

    public static func attention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, cache: KVCache?,
        scale: Float, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        enabled: Bool
    ) -> MLXArray {
        guard enabled, useFloat16, prepareStorage(cache) else {
            return attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values, cache: cache,
                scale: scale, mask: mask)
        }
        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
        if case .array(let array) = mask, array.dtype != .bool {
            attentionMask = .array(array.asType(.float16))
        } else {
            attentionMask = mask
        }
        let output = attentionWithCacheUpdate(
            queries: queries.asType(.float16), keys: keys.asType(.float16),
            values: values.asType(.float16), cache: cache, scale: scale, mask: attentionMask)
        let shouldReport = reported.withLock { reported in
            if reported { return false }
            reported = true
            return true
        }
        if shouldReport {
            let stored =
                cache?.innerState().filter { $0.ndim == 4 }
                .map { String(describing: $0.dtype) }.joined(separator: ",") ?? "none"
            FileHandle.standardError.write(
                Data(
                    ("[BonsaiAttention] source_qkv=\(queries.dtype)/\(keys.dtype)/\(values.dtype) "
                        + "attention_output=\(output.dtype) stored_kv=\(stored) "
                        + "recurrent_state_policy=unchanged\n").utf8))
        }
        // Keep the original gate/output-projection arithmetic. In particular,
        // do not let the narrower SDPA output change Hadamard projection math.
        return output.asType(queries.dtype)
    }

    /// Conversion occurs only when an existing floating buffer is wider. The
    /// normal decode path does not slice/copy a whole cache or read a GPU offset.
    /// Keeping the buffers' capacity and ring/graph counters avoids changing
    /// restore, mask or compiled-decode semantics. Batch views recurse to owners.
    static func prepareStorage(_ cache: KVCache?) -> Bool {
        func half(_ array: MLXArray?) -> MLXArray? {
            guard let array else { return nil }
            return array.dtype == .float16 ? array : array.asType(.float16)
        }
        switch cache {
        case nil:
            return true
        case let cache as KVCacheSimple:
            cache.keys = half(cache.keys)
            cache.values = half(cache.values)
        case let cache as RotatingKVCache:
            cache.keys = half(cache.keys)
            cache.values = half(cache.values)
        case let cache as CompilableKVCache:
            cache.keys = half(cache.keys)
            cache.values = half(cache.values)
        case let cache as BatchKVCache:
            return cache.prepareHadamardAttentionStorage()
        default:
            // Respect explicitly selected quantized/custom cache layouts.
            return false
        }
        return true
    }
}
