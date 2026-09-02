import Foundation
import MLX

/// Attention utilities that match Python mlx-lm's interface
///
/// This provides a single function that automatically routes to quantized or regular
/// attention based on cache type, matching Python's `scaled_dot_product_attention`

/// Automatic attention with cache update
///
/// This function matches Python's `scaled_dot_product_attention` in base.py:
/// - Detects if cache is `QuantizedKVCache` using `isinstance` pattern
/// - Routes to `quantizedScaledDotProductAttention` or `MLXFast.scaledDotProductAttention`
/// - Handles cache updating automatically
/// - Transparent to models - they just call this function
///
/// **Usage in models:**
/// ```swift
/// let output = attentionWithCacheUpdate(
///     queries: queries,
///     keys: keys,
///     values: values,
///     cache: cache,
///     scale: scale,
///     mask: mask
/// )
/// ```
///
/// - Parameters:
///   - queries: Query tensor [B, nHeads, L, D]
///   - keys: Raw key tensor to be cached [B, nKVHeads, L, D]
///   - values: Raw value tensor to be cached [B, nKVHeads, L, D]
///   - cache: Cache instance (any type)
///   - scale: Attention scale factor
///   - mask: Attention mask
/// - Returns: Attention output [B, nHeads, L, D]
/// MLA-specific SDPA helper for DeepSeek-style latent attention families.
///
/// Unlike `attentionWithCacheUpdate`, this helper assumes the caller already
/// called `cache.update(...)` and is passing the full post-update keys/values.
/// Calling the generic helper after a manual MLA update appends the same keys
/// twice and corrupts cache length/positioning.
/// Decode-step fp32 cast gate. The historical decode path cast Q/K/V to
/// float32 for every S==1 step, which materializes a full-context fp32 copy
/// of the KV cache per token per MLA layer — ~5x the KV traffic of a bf16
/// SDPA and the dominant cost of long-context decode for every MLA family
/// (measured: Raptor 8B-A1B fell 173 -> 35 tok/s from 0.4k -> 26k context,
/// ~0.95us per context token per step, all of it attention-side copies).
/// `VMLX_MLA_SDPA_FP32=1` restores the old cast for A/B and fallback.
enum MLAAttentionRuntime {
    /// Read at every decode step; written only by the env initializer and by
    /// the focused dtype tests (which run serialized under the shared MLX
    /// test lock). Plain Bool loads/stores are atomic on arm64; a torn or
    /// stale read here could at worst run one step down the other (equally
    /// correct) SDPA path.
    nonisolated(unsafe) static var decodeFP32Cast: Bool = {
        ProcessInfo.processInfo.environment["VMLX_MLA_SDPA_FP32"] == "1"
    }()
}

public func mlaScaledDotProductAttention(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
    let originalDtype = queries.dtype
    let queryLength = queries.dim(2)
    if MLAAttentionRuntime.decodeFP32Cast, queryLength == 1, originalDtype != .float32 {
        let maskFp32: MLXFast.ScaledDotProductAttentionMaskMode = {
            switch mask {
            case .array(let m): return .array(m.asType(.float32))
            default: return mask
            }
        }()
        let out = MLXFast.scaledDotProductAttention(
            queries: queries.asType(.float32),
            keys: keys.asType(.float32),
            values: values.asType(.float32),
            scale: scale,
            mask: maskFp32
        )
        return out.asType(originalDtype)
    }
    return MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: keys,
        values: values,
        scale: scale,
        mask: mask
    )
}

public func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
    guard let cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
    }
    if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode
        )
    } else {
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)

        // Stage 2 (iter 9): when the caller passes `.none`, give the cache
        // a chance to contribute a mask. Most caches (KVCacheSimple,
        // RotatingKVCache, TurboQuantKVCache) return `.none` for n=1 decode
        // and rely on the K/V shape itself to bound attention. But the
        // compile-path variants (CompilableKVCache,
        // CompilableTurboQuantKVCache) return the FULL fixed-size buffer
        // from update() and need a real `.array` mask to exclude
        // uninitialised tail positions. Without this hook, tail zeros
        // would dilute softmax weights — negligible when scores are large
        // (KVCacheSimple-valued prefill) but material when scores are
        // small (TQ-decoded prefill, which is why Stage 2 rolled back
        // originally before the real fix landed).
        //
        // Caches whose makeMask returns `.none` for n=1 (the base class
        // default) get `effectiveMask == .none` here, preserving today's
        // behaviour exactly.
        var effectiveMask = mask
        if case .none = mask {
            let cacheMask = cache.makeMask(
                n: queries.dim(2), windowSize: nil, returnArray: true)
            if case .array = cacheMask {
                effectiveMask = cacheMask
            }
        }

        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: effectiveMask
        )
    }
}
