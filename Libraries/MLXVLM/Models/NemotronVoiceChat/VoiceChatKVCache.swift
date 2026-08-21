// NemotronLabs VoiceChat — the KV cache the speech decoder runs on.
//
// 🚨 Why this exists rather than reusing the shared caches.
//
// The speech decoder's whole-sequence prefill matched the Python reference
// exactly (cosine 1.0000), while its SINGLE-TOKEN decode step did not
// (layer 0 cosine 0.9992, layer 1 0.8863, compounding to ~0.94 at the output).
// That is the signature of a cache whose step read differs from the
// reference's, and it renders as fluent-sounding babble: audio with healthy
// RMS, dynamic range and zero-crossing rate that contains no words.
//
// The reference (`mlx_vlm.models.cache`) keeps its own `_offset`,
// `left_padding` and `rotated` state and returns `keys[..., :_offset, :]` from
// an in-place buffer. The shared Swift `RotatingKVCache` has different
// bookkeeping, and `KVCacheSimple` grows in 256-row steps — so neither
// reproduces the reference's step read here.
//
// For THIS model the distinction is moot in the only regime that occurs: a
// turn is a few hundred frames against a 7500-frame window, so the reference's
// rotating cache never rotates and is exactly a concatenating cache. This one
// therefore concatenates, reports `offset` as the number of rows stored BEFORE
// the current update (which is what the RoPE position must be), and returns
// precisely the rows it holds — no padding, no reordering.

import Foundation
import MLX
import MLXFast
import MLXLMCommon

/// Plain concatenating KV cache with reference-matching semantics.
///
/// Conforms to `KVCache` directly rather than subclassing `BaseKVCache`,
/// whose initialiser is internal to MLXLMCommon.
public class VoiceChatConcatKVCache: KVCache {
    private var keys: MLXArray?
    private var values: MLXArray?

    public private(set) var offset: Int = 0
    public var maxSize: Int? { nil }
    public var isTrimmable: Bool { true }

    public init() {}

    public func innerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        if let existingKeys = keys, let existingValues = values {
            keys = MLX.concatenated([existingKeys, newKeys], axis: 2)
            values = MLX.concatenated([existingValues, newValues], axis: 2)
        } else {
            keys = newKeys
            values = newValues
        }
        // `offset` is read BEFORE the next update to position RoPE, so it must
        // equal the number of rows already stored — exactly what the reference
        // reports (`self.offset = self.keys.shape[-2]`).
        offset = keys!.dim(2)
        return (keys!, values!)
    }

    public var state: [MLXArray] {
        get { [keys, values].compactMap { $0 } }
        set {
            guard newValue.count == 2 else {
                keys = nil
                values = nil
                offset = 0
                return
            }
            keys = newValue[0]
            values = newValue[1]
            offset = newValue[0].dim(2)
        }
    }

    public var metaState: [String] {
        get { [""] }
        set { _ = newValue }
    }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        guard let currentKeys = keys, let currentValues = values else { return 0 }
        let trimmed = Swift.min(n, currentKeys.dim(2))
        guard trimmed > 0 else { return 0 }
        keys = currentKeys[.ellipsis, ..<(currentKeys.dim(2) - trimmed), 0...]
        values = currentValues[.ellipsis, ..<(currentValues.dim(2) - trimmed), 0...]
        offset -= trimmed
        return trimmed
    }

    /// A single new token attends to everything held; a multi-token prefill is
    /// causal. Matches what the reference builds for this backbone.
    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        n <= 1 ? .none : .causal
    }

    public func copy() -> any KVCache {
        let clone = VoiceChatConcatKVCache()
        clone.keys = keys
        clone.values = values
        clone.offset = offset
        return clone
    }
}
