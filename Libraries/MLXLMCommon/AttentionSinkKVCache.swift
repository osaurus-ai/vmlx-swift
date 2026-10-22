// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import MLX

/// Cache contracts that cannot be replaced by generic lossy KV promotion.
/// Learned attention sinks require the floating SDPA path.
public protocol NonQuantizableKVCache: KVCache {}

/// Full-attention cache for architectures with learned sinks. Uses ordinary KV
/// buffers, but preserves its contract across copies and disk restore.
public final class AttentionSinkKVCache: KVCacheSimple, NonQuantizableKVCache, DiskCacheStateProviding {
    public let headCount: Int
    public let headDimension: Int
    public init(headCount: Int, headDimension: Int) {
        self.headCount = headCount
        self.headDimension = headDimension
        super.init()
    }

    public var diskCacheStateIdentifier: String { "attention-sink-kv-v1" }

    public func restoreDiskCacheState(_ state: [MLXArray], metadata: [String], offset: Int) -> Bool {
        guard metadata == metaState, offset > 0, state.count == 2,
            state.allSatisfy({ $0.ndim == 4 && $0.dim(0) == 1
                && $0.dim(1) == headCount && $0.dim(2) == offset
                && $0.dim(3) == headDimension && $0.dtype == .bfloat16 })
        else { return false }
        self.state = state
        return self.offset == offset
    }

    public override func copy() -> any KVCache {
        let copy = AttentionSinkKVCache(headCount: headCount, headDimension: headDimension)
        copy.step = step
        if !state.isEmpty { copy.state = state.map(ownedStateCopy) }
        return copy
    }
}
