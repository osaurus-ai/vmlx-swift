// ZayaCCACache — first-class hybrid cache for ZAYA1 CCA-attention layers.
//
// CCA mixes ordinary K/V tensors with two path-dependent state arrays
// (`conv_state` and `prev_hs`) that the conv_qk module produces during
// attention. Restoring KV without restoring the CCA state is a false hit
// (per the Zyphra runtime contract: ZAYA1-8B-RUNTIME-PREP-2026-05-06.md),
// so this cache holds them together and the disk serializer round-trips
// them as one unit.
//
// State layout (always 4 arrays, sentinels used when KV slot is empty):
//   [0] keys        [B, kv_heads, T, head_dim]   bf16/fp16  (empty: shape [1,1,0,1])
//   [1] values      [B, kv_heads, T, head_dim]   bf16/fp16  (empty: shape [1,1,0,1])
//   [2] conv_state  [B, conv_channels, 2]        fp32       (always populated, init zeros)
//   [3] prev_hs     [B, hidden_size]             fp32       (always populated, init zeros)

import Foundation
import MLX
import MLXNN

public final class ZayaCCACache: KVCache {
    /// Inner standard rolling KV. We delegate `update`, `makeMask`, and
    /// `offset` to it. The CCA-specific state lives outside it.
    private var kv: any KVCache

    /// Path-dependent CCA state. Float32 per the runtime contract; do not
    /// downcast or compress these (the conv_qk module reads/writes them
    /// at every prefill chunk + every decode token).
    private var convState: MLXArray
    private var prevHS: MLXArray

    public let convChannels: Int   // q_dim + kv_dim  (e.g. 1024 + 256 = 1280)
    public let hiddenSize: Int     // model hidden_size (e.g. 2048)
    public let batchSize: Int      // 1; BatchZayaCCACache wraps for B>1

    public init(batchSize B: Int = 1, convChannels: Int = 1280, hiddenSize: Int = 2048) {
        precondition(B > 0, "ZayaCCACache batchSize must be positive")
        precondition(convChannels > 0, "ZayaCCACache convChannels must be positive")
        precondition(hiddenSize > 0, "ZayaCCACache hiddenSize must be positive")
        self.kv = KVCacheSimple()
        self.convState = MLXArray.zeros([B, convChannels, 2], dtype: .float32)
        self.prevHS = MLXArray.zeros([B, hiddenSize], dtype: .float32)
        self.convChannels = convChannels
        self.hiddenSize = hiddenSize
        self.batchSize = B
    }

    // MARK: - KVCache protocol

    public var offset: Int { kv.offset }
    public var maxSize: Int? { kv.maxSize }
    public var isTrimmable: Bool { false }   // v1: prefix-cache off, no trim path needed

    public func innerState() -> [MLXArray] {
        // Compile-traceable inner state: same as the disk-stable `state` view.
        state
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        kv.update(keys: keys, values: values)
    }

    /// True only when the *real CCA attention KV* is in TurboQuant's
    /// compressed phase. ZAYA's odd decoder entries are MoE-only placeholders
    /// and must never be counted as compressed attention layers.
    public var usesTurboQuantKV: Bool {
        (kv as? TurboQuantKVCache)?.phase == .compressed
    }

    /// The encoded attention KV owned by this CCA layer, when present.
    /// CCA companion state is deliberately not part of this object and stays
    /// native fp32 in ``convState`` and ``prevHS``.
    public var turboQuantKVCache: TurboQuantKVCache? {
        guard let tq = kv as? TurboQuantKVCache, tq.phase == .compressed else {
            return nil
        }
        return tq
    }

    /// Promote only this CCA layer's ordinary attention KV to TurboQuant.
    /// Returns true only after an actual encoded transition; a short/empty
    /// cache remains native and can be retried after more prompt tokens exist.
    @discardableResult
    public func promoteAttentionKVToTurboQuant(
        keyBits: Int,
        valueBits: Int,
        sinkTokens: Int = 4,
        residualTokens: Int = TurboQuantKVCache.defaultResidualTokens
    ) -> Bool {
        guard let simple = kv as? KVCacheSimple,
              TurboQuantKVCache.hasCompressibleMiddle(
                  tokenCount: simple.offset,
                  sinkTokens: sinkTokens,
                  residualTokens: residualTokens)
        else { return false }

        let promoted = TurboQuantKVCache.fromSimpleCache(
            simple,
            keyBits: keyBits,
            valueBits: valueBits,
            sinkTokens: sinkTokens,
            residualTokens: residualTokens)
        guard promoted.phase == .compressed else { return false }
        kv = promoted
        return true
    }

    /// Restore TQ-native attention KV while leaving the native CCA companion
    /// arrays untouched. The caller restores `conv_state` / `prev_hs` from
    /// the same typed disk payload immediately before or after this call.
    @discardableResult
    public func restoreTurboQuantAttentionKV(
        encodedKeys: EncodedKeys,
        encodedValues: EncodedValues,
        sourceOffset: Int
    ) -> Bool {
        let keyBits = encodedKeys.indexBits + 1
        let valueBits = encodedValues.indexBits
        let sinkTokens = max(encodedKeys.sinkCount, encodedValues.sinkCount, 4)
        let residualTokens = max(
            encodedKeys.tailCount,
            encodedValues.tailCount,
            TurboQuantKVCache.defaultResidualTokens)
        let restored = TurboQuantKVCache(
            keyBits: keyBits,
            valueBits: valueBits,
            sinkTokens: sinkTokens,
            residualTokens: residualTokens)
        restored.restoreCompressed(
            encodedKeys: encodedKeys,
            encodedValues: encodedValues,
            sourceOffset: sourceOffset)
        guard restored.phase == .compressed, restored.offset == sourceOffset else {
            return false
        }
        kv = restored
        return true
    }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        kv.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    @discardableResult
    public func trim(_ n: Int) -> Int { 0 }

    /// Round-trip layout (kept stable for `TQDiskSerializer`):
    ///   [0] keys (or zero-seq sentinel)
    ///   [1] values (or zero-seq sentinel)
    ///   [2] conv_state
    ///   [3] prev_hs
    public var state: [MLXArray] {
        get {
            let kvState = kv.state
            let kPart: MLXArray
            let vPart: MLXArray
            if kvState.count == 2 {
                kPart = kvState[0]
                vPart = kvState[1]
            } else {
                // Empty KV — emit sentinels with axis-2 (sequence) dim of 0.
                kPart = MLXArray.zeros([batchSize, 1, 0, 1], dtype: .bfloat16)
                vPart = MLXArray.zeros([batchSize, 1, 0, 1], dtype: .bfloat16)
            }
            return [kPart, vPart, convState, prevHS]
        }
        set {
            precondition(newValue.count == 4,
                "ZayaCCACache.state requires exactly 4 arrays [keys, values, conv_state, prev_hs]")
            let k = newValue[0]
            let v = newValue[1]
            // Sentinel detection: zero-length sequence dim ⇒ empty KV.
            if k.ndim >= 3 && k.dim(2) == 0 {
                // Leave kv empty — kv.offset stays 0.
            } else {
                kv.state = [k, v]
            }
            convState = newValue[2]
            prevHS = newValue[3]
        }
    }

    /// metaState layout: kv's metaState (single empty string) + ZAYA trailer.
    public var metaState: [String] {
        get {
            kv.metaState + [
                "zaya_cca_v1",
                String(convChannels),
                String(hiddenSize),
                String(batchSize),
            ]
        }
        set {
            let trailer = 4
            if newValue.count >= trailer,
               newValue[newValue.count - trailer] == "zaya_cca_v1"
            {
                kv.metaState = Array(newValue[0..<(newValue.count - trailer)])
            } else {
                // Legacy path: pass everything through to kv.
                kv.metaState = newValue
            }
        }
    }

    public func copy() -> any KVCache {
        let dup = ZayaCCACache(
            batchSize: batchSize,
            convChannels: convChannels,
            hiddenSize: hiddenSize)
        dup.kv = kv.copy()
        dup.convState = convState[.ellipsis]
        dup.prevHS = prevHS[.ellipsis]
        return dup
    }

    // MARK: - CCA-state accessors (used only by ZayaCCAAttention.forward)

    /// Read the current CCA state. Caller must reshape/move-to-dtype as needed.
    public func readCCA() -> (conv: MLXArray, prev: MLXArray) {
        (convState, prevHS)
    }

    /// Write the CCA state at the end of an attention forward.
    public func writeCCA(conv: MLXArray, prev: MLXArray) {
        precondition(conv.shape == [batchSize, convChannels, 2],
            "ZayaCCACache: conv_state must be [B=\(batchSize), \(convChannels), 2], got \(conv.shape)")
        precondition(prev.shape == [batchSize, hiddenSize],
            "ZayaCCACache: prev_hs must be [B=\(batchSize), \(hiddenSize)], got \(prev.shape)")
        convState = conv
        prevHS = prev
    }
}

/// Explicit cache placeholder for ZAYA's odd MoE-only decoder entries.
///
/// Those layers have no attention operation and never read or write KV. A
/// `KVCacheSimple` placeholder was structurally convenient but made generic
/// cache policy and telemetry misidentify 40 non-attention slots as eligible
/// KV layers. This type is intentionally state-free and non-quantizable.
public final class ZayaMoEPlaceholderCache: BaseKVCache {
    public override init() {
        super.init()
    }

    public override func copy() -> any KVCache {
        ZayaMoEPlaceholderCache()
    }
}
