// Copyright © 2026 Osaurus.
//
// OpenPangu-v2 per-layer cache. Each layer is EITHER a DSA full-attention layer
// (KVCacheSimple + lightning-indexer pool) OR a SWA sliding-window layer
// (RotatingKVCache), and BOTH additionally carry the 3 causal-conv states
// (qa / compresskv / o). The conv-state is path-dependent (SSM-style): it MUST
// round-trip with the KV or a KV-only prefix-cache hit is a silent false hit
// (garbled turn-2). See OPENPANGU-V2-PORT-STATUS.md and the cache-tier map.
//
// Design (decision B): one cache type that is BOTH a `HybridPoolCache` (so the
// SWA rotating window + DSA pool serialize through the SSD/L2 disk tier) AND
// carries ArraysCache-style path-dependent conv-state. The ~6 SSM/topology
// switch-sites (extractSSMStates / restoreSSMStates / cacheContainsPathDependentState
// / ModelCacheTopologySnapshot.record / CacheFamily.perLayerFamily / TQDiskSerializer)
// are taught to recognize this type in the cache-integration pass.

import Foundation
import MLX
import MLXLMCommon

/// Which of the 3 causal convs a state slot belongs to.
public enum OpenPanguConvKind: Int, CaseIterable, Sendable {
    case qa = 0
    case compressKv = 1
    case o = 2
}

public final class OpenPanguV2Cache: HybridPoolCache {
    /// The underlying KV cache: `RotatingKVCache` on SWA layers, `KVCacheSimple`
    /// (unbounded, full attention) on DSA layers. `var` (not `let`) so the
    /// `state`/`metaState` setters can mutate through the existential — KVCache
    /// is not class-constrained, so member-setter access requires a `var`.
    public var kv: KVCache
    /// True for DSA (full-attention + indexer) layers.
    public let isDSA: Bool
    /// Per-layer sliding window (SWA) or 0 (DSA full).
    public let slidingWindow: Int

    /// The 3 conv-states, each `(B, kernel-1, channels)` or nil at sequence start.
    private var conv: [MLXArray?] = [nil, nil, nil]

    /// DSA lightning-indexer pooled state (added with the indexer pass).
    fileprivate var idxPooled: MLXArray?

    public init(kv: KVCache, isDSA: Bool, slidingWindow: Int) {
        self.kv = kv
        self.isDSA = isDSA
        self.slidingWindow = slidingWindow
    }

    // MARK: conv-state accessors used by OpenPanguV2Attention
    public func convState(_ which: OpenPanguConvKind) -> MLXArray? { conv[which.rawValue] }
    public func setConvState(_ which: OpenPanguConvKind, _ value: MLXArray?) {
        conv[which.rawValue] = value
    }

    // MARK: HybridPoolCache / RotatingKVCacheWrapper
    /// Only meaningful for SWA layers; DSA layers have no rotating window, but the
    /// protocol requires it — expose the rotating cache when present.
    public var rotating: RotatingKVCache {
        (kv as? RotatingKVCache) ?? RotatingKVCache(maxSize: slidingWindow, keep: 0)
    }
    /// DSA layers keep the full KV (indexer selects top-k, not a pooled summary),
    /// so there is no compressor pool: compressRatio reports 0 for DSA and the
    /// serializer treats the pool slots as empty.
    public var compressRatio: Int { isDSA ? 0 : 0 }

    public func hybridPool(branch: HybridPoolBranch) -> MLXArray? {
        branch == .indexer ? idxPooled : nil
    }
    public func setHybridPool(branch: HybridPoolBranch, value: MLXArray?) {
        if branch == .indexer { idxPooled = value }
    }
    public func hybridBuffers(branch: HybridPoolBranch) -> (kv: MLXArray?, gate: MLXArray?) {
        (nil, nil)
    }
    public func setHybridBuffers(branch: HybridPoolBranch, kv: MLXArray?, gate: MLXArray?) {}

    // MARK: KVCache — delegate to the underlying kv
    public var offset: Int { kv.offset }
    public var maxSize: Int? { kv.maxSize }
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        kv.update(keys: keys, values: values)
    }
    public var isTrimmable: Bool { kv.isTrimmable }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        // Conv-state is start_pos-keyed and invalidated by any trim → drop it
        // (rederived from prompt tokens on the next prefill). Same contract as
        // ZayaCCACache / the DSV4 incomplete-window buffers.
        conv = [nil, nil, nil]
        idxPooled = nil
        return kv.trim(n)
    }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        kv.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    /// Round-trip layout: `kv.state` first, then the 3 conv slots, then idxPooled.
    /// A nil slot is encoded as a `(1,0,1)` zero-row sentinel (matches DSV4).
    public var state: [MLXArray] {
        get {
            kv.state
                + conv.map { OpenPanguV2Cache.serializable($0) }
                + [OpenPanguV2Cache.serializable(idxPooled)]
        }
        set {
            // Last 4 slots are 3 conv states + idxPooled; the rest is kv.state.
            precondition(newValue.count >= 4,
                "OpenPanguV2Cache.state setter expects >= 4 trailing slots")
            let split = newValue.count - 4
            kv.state = Array(newValue[0..<split])
            conv = [
                OpenPanguV2Cache.nullable(newValue[split + 0]),
                OpenPanguV2Cache.nullable(newValue[split + 1]),
                OpenPanguV2Cache.nullable(newValue[split + 2]),
            ]
            idxPooled = OpenPanguV2Cache.nullable(newValue[split + 3])
        }
    }

    public var metaState: [String] {
        get { kv.metaState + ["openpangu_v2_cache_v1", isDSA ? "dsa" : "swa", String(slidingWindow)] }
        set {
            if newValue.count >= 3, newValue[newValue.count - 3] == "openpangu_v2_cache_v1" {
                kv.metaState = Array(newValue[0..<(newValue.count - 3)])
            } else {
                kv.metaState = newValue
            }
        }
    }

    public func copy() -> any KVCache {
        let c = OpenPanguV2Cache(kv: kv.copy(), isDSA: isDSA, slidingWindow: slidingWindow)
        c.conv = conv.map { $0.map { $0 * 1 } }
        c.idxPooled = idxPooled.map { $0 * 1 }
        return c
    }

    public func innerState() -> [MLXArray] { state }

    // MARK: nil <-> zero-row sentinel (matches DeepseekV4Cache)
    static func serializable(_ a: MLXArray?) -> MLXArray {
        a ?? MLXArray.zeros([1, 0, 1])
    }
    static func nullable(_ a: MLXArray) -> MLXArray? {
        (a.ndim == 3 && a.dim(1) == 0) ? nil : a
    }
}
