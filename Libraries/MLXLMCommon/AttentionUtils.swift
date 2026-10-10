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

        // Opt-in multi-row decode attention for 2–16-row speculative blocks (see MultiRowDecodeAttention). Causal,
        // or an array mask the caller declared linear-chain causal; plain KVCacheSimple only (compile-path caches return full fixed-size buffers that need their mask).
        if MultiRowDecodeAttention.enabled, type(of: cache) == KVCacheSimple.self,
            MultiRowDecodeAttention.eligible(queries: queries, keys: cachedKeys, values: cachedValues),
            MultiRowDecodeAttention.maskAllowed(
                mask, queryLength: queries.dim(2), keyLength: cachedKeys.dim(2))
        {
            return MultiRowDecodeAttention.attend(
                queries: queries, keys: cachedKeys, values: cachedValues, scale: scale)
        }

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

/// Causal attention for a short multi-row query (a speculative draft or verify block) over a long
/// cached context, run as row chunks the vector SDPA kernel accepts.
///
/// MLX's fused SDPA uses its vector kernel only while `queryLength * gqaFactor <= 32` (8 rows at
/// GQA 4). Past that it takes the tiled full-attention kernel, which parallelises over query tiles:
/// a 12-row block is one tile per head walking every key serially, so its cost grows with the
/// context far faster than the vector kernel's. K2 Horizon 7B, 12-row forward, M5 Max:
///
///     cached tokens      256     2,304    8,192
///     one fused call    20.4 ms  30.4     59.3
///     8 + 4 row chunks  21.0     28.1     44.3
///
/// Chunk `[a, b)` attends to keys `[0, kL - qL + b)`. The causal mask is bottom-right aligned, so
/// query `a + i` sees exactly the keys up to `kL - qL + a + i`, as in one call over all rows; an
/// explicit `(…, qL, kL)` mask is sliced to the same rows and keys. Every chunk runs the vector
/// kernel that 1-row decoding runs, and carries its precision (relative RMS error against a
/// float32 reference within 3% of per-position decode calls, bf16 and f16).
/// Returns nil for shapes that do not need or cannot take chunking.
public func chunkedVectorAttention(
    queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> MLXArray? {
    let (H, qL, kL) = (queries.dim(1), queries.dim(2), keys.dim(2))
    let kvH = keys.dim(1)
    guard kvH > 0, H % kvH == 0, kL >= qL else { return nil }
    let rowsPerChunk = min(8, 32 / (H / kvH))
    guard rowsPerChunk >= 1, qL > rowsPerChunk else { return nil }
    let maskArray: MLXArray?
    switch mask {
    case .causal: maskArray = nil
    case .array(let array):
        guard array.ndim >= 2, array.dim(-2) == qL, array.dim(-1) == kL else { return nil }
        maskArray = array
    default: return nil
    }
    var outputs: [MLXArray] = []
    var start = 0
    while start < qL {
        let end = min(qL, start + rowsPerChunk)
        let visible = kL - qL + end
        outputs.append(
            MLXFast.scaledDotProductAttention(
                queries: queries[0..., 0..., start ..< end, 0...],
                keys: keys[0..., 0..., ..<visible, 0...],
                values: values[0..., 0..., ..<visible, 0...],
                scale: scale,
                mask: maskArray.map { .array($0[.ellipsis, start ..< end, ..<visible]) } ?? .causal))
        start = end
    }
    return concatenated(outputs, axis: 2)
}
