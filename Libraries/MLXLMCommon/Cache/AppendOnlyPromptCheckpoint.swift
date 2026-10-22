import MLX

/// A prompt boundary retains only state that decode can overwrite. Ordinary
/// full-attention KV is append-only: remember its layer index, not its arrays.
/// Materialize prefix views after decode, when no later forward mutates them.
/// This preserves unique live-buffer ownership throughout token generation.
struct AppendOnlyPromptCheckpoint {
    let tokenCount: Int
    private let layers: [KVCache?]

    static func capture(cache: [KVCache], tokenCount: Int) -> Self? {
        guard tokenCount > 0, !cache.isEmpty,
              cache.allSatisfy({ isAppendOnly($0) || type(of: $0) == RotatingKVCache.self }),
              cacheBoundaryLeafOffsets(cache).allSatisfy({ $0 == tokenCount }) else { return nil }
        let layers: [KVCache?] = cache.map { isAppendOnly($0) ? nil : $0.copy() }
        MLX.eval(layers.compactMap { $0 })
        return Self(tokenCount: tokenCount, layers: layers)
    }

    static func isAppendOnly(_ cache: KVCache) -> Bool {
        type(of: cache) == KVCacheSimple.self || cache is AttentionSinkKVCache
    }

    /// Source is immutable for the remainder of finalization. No view from this
    /// method may survive into a subsequent update of the same cache objects.
    func materialize(from live: [KVCache]) -> [KVCache]? {
        guard live.count == layers.count else { return nil }
        var result: [KVCache] = []
        for (index, sealed) in layers.enumerated() {
            if let sealed { result.append(sealed); continue }
            guard Self.isAppendOnly(live[index]), live[index].offset >= tokenCount,
                  let prefix = appendOnlyPrefix(live[index], count: tokenCount, owned: false) else { return nil }
            result.append(prefix)
        }
        return result
    }
}

/// Slice first, then optionally seal only the requested rows. Generic copy()
/// followed by trim retains full-prompt allocations for much shorter files.
func appendOnlyPrefix(_ layer: KVCache, count: Int, owned: Bool) -> KVCache? {
    guard AppendOnlyPromptCheckpoint.isAppendOnly(layer), let simple = layer as? KVCacheSimple,
          count > 0, count <= simple.offset, let keys = simple.keys, let values = simple.values else { return nil }
    let result: KVCacheSimple
    if let sink = simple as? AttentionSinkKVCache {
        result = AttentionSinkKVCache(headCount: sink.headCount, headDimension: sink.headDimension)
    } else { result = KVCacheSimple() }
    result.step = simple.step
    let state = [keys[.ellipsis, ..<count, 0...], values[.ellipsis, ..<count, 0...]]
    result.state = owned ? state.map(ownedStateCopy) : state
    return result
}

func compactPromptPrefix(_ cache: [KVCache], count: Int) -> [KVCache]? {
    guard count > 0 else { return nil }
    var result: [KVCache] = []
    for layer in cache {
        if let prefix = appendOnlyPrefix(layer, count: count, owned: true) {
            result.append(prefix)
        } else {
            // Recurrent and custom topologies keep their established restore
            // policy; this helper is only a trim path, never a state rewind.
            guard layer.isTrimmable, layer.offset >= count else { return nil }
            let copy = layer.copy()
            let trim = layer.offset - count
            guard copy.trim(trim) == trim else { return nil }
            result.append(copy)
        }
    }
    MLX.eval(result)
    return result
}
