// Copyright © 2024 Apple Inc.

import Foundation
import MLX

// MARK: - KV Cache Extraction

/// Deep-copy a cache list at the prompt boundary for later cache-tier storage.
///
/// Cache coordinator keys are built from prompt tokens only. Disk arrays must
/// therefore represent exactly that prompt boundary, not the live cache after
/// decode has appended generated tokens. DSV4's `HybridPoolCache` is especially
/// sensitive here because post-decode CSA/HSA pool rows can look structurally
/// valid while carrying generated-token state under a prompt-only key.
func makePromptBoundaryCacheSnapshot(from cache: [any KVCache]) -> [any KVCache] {
    let snapshot = cache.map { $0.copy() }
    // Materialize the whole list unconditionally, for two reasons:
    //
    // 1. ArraysCache/MambaCache update their recurrent tensors in place;
    //    the snapshot must be sealed before decode mutates the live cache so
    //    the prompt boundary cannot inherit later tool/output state.
    // 2. Every copy() now builds owned `* 1` nodes (`ownedStateCopy`), but an
    //    UNEVALUATED node still holds a graph reference to the live cache's
    //    buffers. A snapshot retained lazily through decode (the slot lane's
    //    `promptCacheSnapshot`) keeps those buffers multiply-referenced,
    //    which blocks buffer donation on the per-token in-place KV update and
    //    turns each decode step into a full-capacity copy — measured 32 -> 6.1
    //    tok/s on a 13.8k prompt. Evaluating here pays one boundary-sized
    //    copy per turn and releases the live buffers back to unique ownership
    //    before the first decode step.
    //
    // One batched eval keeps this substantially cheaper than synchronizing
    // every layer separately.
    MLX.eval(snapshot)
    return snapshot
}

/// Retain the exact prompt snapshot unless the cache uses DSV4's typed pool
/// topology, whose coordinator boundary must always be prompt-minus-one.
///
/// DeepSeek V4 deliberately restores N-1 and re-feeds the final prompt token.
/// Its exact prompt snapshot is not a usable coordinator boundary. Suppress it
/// even on a warm N-1 restore where no new seed is captured; otherwise an exact
/// cache copy is retained through decode only to be discarded at store time.
/// Standalone rotating/SWA caches retain their existing exact snapshot because
/// their prompt/stable-boundary storage policy still consumes it.
func makeRetainedExactPromptSnapshot(
    from cache: [any KVCache]
) -> [any KVCache]? {
    guard !cacheRequiresPrefillCapturedDiskSeed(cache) else { return nil }
    return makePromptBoundaryCacheSnapshot(from: cache)
}

/// Build the cache object list passed to ``TQDiskSerializer`` from a clean
/// prompt-boundary snapshot.
///
/// This preserves cache-specific disk paths without using the live decode
/// cache: ordinary `KVCacheSimple` layers can be compressed here when the
/// caller requested TurboQuant, while DSV4 `HybridPoolCache` layers remain
/// in their SWA+CSA+HSA shape for `LayerKind.deepseekV4` serialization.
func makeDiskStoreCache(
    fromPromptBoundary snapshot: [any KVCache],
    kvBits: Int?,
    kvGroupSize: Int,
    quantizedKVStart: Int,
    kvMode: KVQuantizationMode
) -> [any KVCache] {
    // The copy exists only to protect `snapshot` from `maybeQuantizeKVCache`'s
    // in-place mutation. When nothing will be quantized that call is a no-op
    // (`kvMode == .none` falls through to a `guard let kvBits` that returns), so
    // the copy protects against nothing — and it is not a cheap nothing: it is a
    // second full duplicate of the KV cache, allocated at the moment memory is
    // already at its high-water mark. The prompt-boundary store takes exactly this
    // path (`kvBits: nil, kvMode: .none`), and on a 64K-token context that copy is
    // tens of GiB. Share the snapshot instead; the caller drops it right after.
    guard kvBits != nil || kvMode != .none else { return snapshot }

    var diskCache = snapshot.map { $0.copy() }
    maybeQuantizeKVCache(
        cache: &diskCache,
        kvBits: kvBits,
        kvGroupSize: kvGroupSize,
        quantizedKVStart: quantizedKVStart,
        kvMode: kvMode)
    return diskCache
}

/// Resolve the lossy codec used for a solo-generation prompt-boundary disk
/// store.
///
/// The generic solo path keeps prompt boundaries raw so an exact warm hit
/// samples its first token from the same KV values as a cold request. ZAYA is
/// a narrower case: CCA makes the topology path-dependent, so coordinator
/// lookup deliberately skips the exact prompt boundary and only uses a stored
/// boundary as a prefix for additional prefill. Its typed serializer can keep
/// CCA companion state native beside encoded attention KV, but exact JANG_6M
/// A/B proof found that 3-bit disk KV drifted while 4-bit and raw disk did not.
/// Keep sub-4-bit ZAYA prompt records lossless; the live attention KV can still
/// transition to the user-selected TurboQuant mode after first-token sampling.
///
/// Do not generalize this to other hybrid families. Each needs its own typed
/// companion-state proof before a lossy prompt-boundary record is eligible.
func selectivePromptBoundaryDiskKVMode(
    cache: [any KVCache],
    requested: KVQuantizationMode
) -> KVQuantizationMode {
    guard case .turboQuant(let keyBits, let valueBits) = requested,
          keyBits >= 4,
          valueBits >= 4,
          cache.contains(where: { $0 is ZayaCCACache })
    else { return .none }
    return requested
}

/// A sub-4-bit encoded ZAYA cache is valid for the live decode path, but is not
/// a proven cross-turn disk boundary. Such a boundary must not be persisted as
/// TQ-native; the raw prefill/stripped boundary remains available for reuse.
func containsUnprovenZayaTurboQuantDiskState(_ cache: [any KVCache]) -> Bool {
    cache.contains { layer in
        guard let tq = (layer as? ZayaCCACache)?.turboQuantKVCache else {
            return false
        }
        return tq.keyBits < 4 || tq.valueBits < 4
    }
}

func makeDiskStoreCache(
    fromPromptBoundary snapshot: [any KVCache],
    parameters: GenerateParameters
) -> [any KVCache] {
    makeDiskStoreCache(
        fromPromptBoundary: snapshot,
        kvBits: parameters.kvBits,
        kvGroupSize: parameters.kvGroupSize,
        quantizedKVStart: parameters.quantizedKVStart,
        kvMode: parameters.kvMode)
}

/// True when any cache layer carries path-dependent non-KV recurrent state.
///
/// Paged KV blocks store attention KV tensors only. Mamba, linear-attention
/// arrays, and ZAYA CCA layers also carry state that depends on the exact
/// prompt path, so a paged-only hit would be a false positive unless a
/// companion disk restore or re-derive path supplies that state too.
public func cacheContainsPathDependentState(_ cache: [any KVCache]) -> Bool {
    cache.contains { layer in
        if layer is MambaCache || layer is ArraysCache || layer is ZayaCCACache {
            return true
        }
        if let cacheList = layer as? CacheList {
            for i in 0..<cacheList.count {
                if cacheContainsPathDependentState([cacheList[i]]) {
                    return true
                }
            }
        }
        return false
    }
}

/// True when cache restore must bypass paged KV blocks and use disk-backed
/// layer-kind serialization.
///
/// This includes path-dependent recurrent caches plus sliding-window /
/// hybrid-pool layers whose ring or pool metadata is not represented by
/// plain paged KV blocks.
public func cacheRequiresDiskBackedCoordinatorRestore(_ cache: [any KVCache]) -> Bool {
    if cacheContainsPathDependentState(cache) {
        return true
    }
    return cache.contains { layer in
        layer is HybridPoolCache ||
            layer is RotatingKVCache ||
            layer is RotatingKVCacheWrapper ||
            layer is TurboQuantKVCache ||
            layer is QuantizedKVCache
    }
}

/// True when a disk-backed prompt must publish an exact prompt-minus-one
/// checkpoint captured while prefill crosses that boundary.
///
/// DeepSeek V4's typed cache is fully serialized (local SWA plus compressor
/// and indexer pools), but its 128-token rotating window cannot be trimmed
/// after it wraps. Exact prompt checkpoints are deliberately not consumed by
/// the coordinator because generation needs logits from the final prompt
/// token. Capturing N-1 before that token is evaluated gives a later request a
/// lossless disk seed that can restore the longest valid prefix and re-feed
/// only the final token. Recurrent Mamba/Arrays/CCA topologies keep their
/// separate companion-state policy.
func cacheRequiresPrefillCapturedDiskSeed(_ cache: [any KVCache]) -> Bool {
    cache.contains { layer in
        if layer is HybridPoolCache { return true }
        if let cacheList = layer as? CacheList {
            for index in 0..<cacheList.count {
                if cacheRequiresPrefillCapturedDiskSeed([cacheList[index]]) {
                    return true
                }
            }
        }
        return false
    } && !cacheContainsPathDependentState(cache)
}

/// Whether the complete prompt boundary from a reusable-prefix warmup is safe
/// to publish for later coordinator restores.
///
/// Recurrent hybrid caches remain path-dependent even when their token key is
/// an exact prefix. A live Ornith reasoning Off -> On warmup restored such a
/// checkpoint and replayed the prior tool command. The processor-proven stable
/// seed remains reusable, but the complete warmup boundary must not become the
/// longest candidate until that topology has byte-parity proof.
func shouldPersistExactPromptBoundary(
    cachePromptIntent: LMInput.CachePromptIntent,
    requiresRecurrentSSMCompanion: Bool
) -> Bool {
    cachePromptIntent != .reusablePrefixWarmup
        || !requiresRecurrentSSMCompanion
}

/// Whether a processor-declared stable boundary may synchronously replay the
/// model after a cache-trim miss.
///
/// A reusable-prefix warmup on a non-recurrent disk-only topology (notably
/// DeepSeek V4's SWA + compressor/indexer pools) already publishes its exact
/// prompt-minus-one checkpoint captured during prefill. Replaying every older
/// stable boundary before reporting warmup completion duplicates prefill and
/// can hold the chat UI in "Warming up" for minutes. Recurrent hybrids are
/// different: their exact warmup boundary is intentionally rejected, so they
/// still need the proven N-1 stable seed re-derive.
func shouldForceStableBoundaryRederive(
    isStableBoundary: Bool,
    isReusablePrefixWarmup: Bool,
    requiresRecurrentSSMCompanion: Bool
) -> Bool {
    isStableBoundary
        && (!isReusablePrefixWarmup || requiresRecurrentSSMCompanion)
}

/// Whether paged KV blocks can safely serve this mixed rotating topology when
/// the exact leaf also carries a typed rotating-boundary companion.
///
/// This is intentionally narrow. It accepts only the Gemma-style composition
/// of direct `RotatingKVCache` layers plus ordinary or TurboQuant attention
/// KV. All-rotating caches have no pageable payload, while CacheList wrappers,
/// affine quantized KV, CCA, SSM, and hybrid-pool caches retain their existing
/// disk-only contracts until they have independent companion proof.
public func cacheCanUsePagedWithRotatingCompanion(_ cache: [any KVCache]) -> Bool {
    var hasRotating = false
    var hasPagedKV = false

    for layer in cache {
        if layer is RotatingKVCache {
            hasRotating = true
        } else if layer is KVCacheSimple || layer is TurboQuantKVCache {
            hasPagedKV = true
        } else {
            return false
        }
    }

    return hasRotating && hasPagedKV
}

/// True when the paged coordinator tier cannot represent enough of the cache
/// topology to restore it safely.
///
/// This is deliberately narrower than
/// ``cacheRequiresDiskBackedCoordinatorRestore(_:)``. Mamba/Arrays companion
/// state is path-dependent and therefore still needs typed disk persistence,
/// but the paged tier can restore its attention KV blocks when
/// ``SSMStateCache`` supplies a complete native companion snapshot at the same
/// boundary. ``CacheCoordinator.fetch`` enforces that boundary match and falls
/// through to disk when the companion is absent or incomplete.
///
/// TurboQuant attention slots are also paged-compatible: paged blocks hold the
/// already-decoded prefix and ``restoreLayerData`` seats it directly in the
/// cache's compressed phase through `restoreFromDecodedKV`, avoiding a second
/// lossy encode. The recurrent companion state itself is never TurboQuant
/// encoded.
///
/// Rotating/SWA rings without an explicitly admitted exact-boundary companion,
/// legacy affine quantized caches, ZAYA CCA, and DSV4 hybrid pools remain
/// disk-only because ordinary paged blocks do not carry their ring, quantized
/// tuple, CCA, or compressor/indexer metadata.
public func cacheCannotUsePagedCoordinatorRestore(_ cache: [any KVCache]) -> Bool {
    func incompatible(_ layer: any KVCache) -> Bool {
        if let cacheList = layer as? CacheList {
            for index in 0..<cacheList.count where incompatible(cacheList[index]) {
                return true
            }
            return false
        }

        // KVCacheSimple and TurboQuantKVCache are carried by paged KV blocks.
        // MambaCache and ArraysCache are carried by exact-boundary companion
        // snapshots, whose presence is mandatory at fetch time.
        if layer is KVCacheSimple
            || layer is TurboQuantKVCache
            || layer is MambaCache
            || layer is ArraysCache
        {
            return false
        }

        // Fail closed for rotating/SWA, affine quantized, CCA, DSV4 pools,
        // compiled/batched caches, placeholders, and every future cache type
        // until its block payload + restore semantics are explicitly proven.
        return true
    }

    return cache.contains(where: incompatible)
}

/// True for Gemma/Mistral-style standalone sliding-window attention caches.
///
/// These caches are disk-backed because paged KV blocks cannot represent the
/// rotating ring metadata. For active tool-call prompts, a stale or partial
/// rotating restore can corrupt the model's next structured-argument emission.
/// DSV4 hybrid-pool and recurrent SSM/CCA models are intentionally excluded:
/// they have separate architecture-specific restore/companion-state contracts.
public func cacheHasStandaloneRotatingWindowState(_ cache: [any KVCache]) -> Bool {
    var hasRotating = false
    var hasHybridPool = false

    func visit(_ layer: any KVCache) {
        if layer is HybridPoolCache {
            hasHybridPool = true
        }
        if layer is RotatingKVCache || layer is RotatingKVCacheWrapper {
            hasRotating = true
        }
        if let cacheList = layer as? CacheList {
            for i in 0..<cacheList.count {
                visit(cacheList[i])
            }
        }
    }

    for layer in cache {
        visit(layer)
    }

    return hasRotating && !hasHybridPool && !cacheContainsPathDependentState(cache)
}

/// True when a failed trim should not fall back to a model re-derive while
/// writing optional history-boundary cache entries.
///
/// Disk-backed topologies carry state that is more than ordinary append-only
/// KV: rotating ring metadata, DSV4 hybrid-pool compressor/indexer state, or
/// lossy/compressed cache payloads. If a prompt-boundary snapshot cannot be
/// trimmed to the requested history boundary, re-entering model prepare/eval
/// from the generation-completion path can race the just-finished decode and
/// abort the host process. The history-boundary entry is only an optimization;
/// the prompt-boundary and post-answer stores remain the authoritative cache
/// entries for these families.
func shouldSkipHistoryBoundaryRederiveAfterTrimMiss(_ cache: [any KVCache]) -> Bool {
    cacheRequiresDiskBackedCoordinatorRestore(cache)
}

/// Extract per-layer KV tensors from a model's cache array.
///
/// Returns per-layer `(keys, values)` tuples. SSM/MambaCache layers return `nil`.
/// Used to populate ``CacheBlock/cacheData`` for paged cache storage.
///
/// - Parameter cache: The model's per-layer cache array.
/// - Returns: An array of optional `(keys, values)` tuples, one per layer.
public func extractLayerData(from cache: [any KVCache]) -> [(keys: MLXArray, values: MLXArray)?] {
    cache.map { layer in
        if let simple = layer as? KVCacheSimple {
            let state = simple.state
            guard state.count == 2 else { return nil }
            return (keys: state[0], values: state[1])
        }
        if let quantized = layer as? QuantizedKVCache {
            // QuantizedKVCache stores quantized tuples, not raw KV.
            // Dequantize back to float keys/values for cache block storage.
            let unquantized = quantized.toUnquantized()
            let state = unquantized.state
            guard state.count == 2 else { return nil }
            return (keys: state[0], values: state[1])
        }
        if let tq = layer as? TurboQuantKVCache {
            // TurboQuantKVCache.state returns float KV in both fill and compressed phases.
            // In fill phase: returns raw float keys/values.
            // In compressed phase: returns unified (decompressed prefix + float window).
            let state = tq.state
            guard state.count == 2 else { return nil }
            return (keys: state[0], values: state[1])
        }
        if let cacheList = layer as? CacheList {
            // CacheList: check sub-caches for KV data.
            // Sub-cache[0] is typically MambaCache, Sub-cache[1] is KVCacheSimple.
            // We only extract the KV part; SSM state is handled separately.
            for i in 0..<cacheList.count {
                if let simple = cacheList[i] as? KVCacheSimple {
                    let state = simple.state
                    if state.count == 2 { return (keys: state[0], values: state[1]) }
                }
                if let quantized = cacheList[i] as? QuantizedKVCache {
                    let unquantized = quantized.toUnquantized()
                    let state = unquantized.state
                    if state.count == 2 { return (keys: state[0], values: state[1]) }
                }
                if let tq = cacheList[i] as? TurboQuantKVCache {
                    let state = tq.state
                    if state.count == 2 { return (keys: state[0], values: state[1]) }
                }
            }
            return nil
        }
        // MambaCache, ArraysCache, RotatingKVCache — no KV extraction
        return nil
    }
}

// MARK: - KV Cache Restoration

/// Restore per-layer KV tensors from cached blocks into a model's cache array.
///
/// Blocks only contain KV-bearing layers (SSM/RotatingKVCache layers are filtered
/// during storage). This function maps block layer indices to the KV-bearing
/// cache layers, skipping non-KV layers.
///
/// - Parameters:
///   - blocks: The cache blocks to restore from, ordered by sequence position.
///   - cache: The model's per-layer cache array to restore into.
/// - Returns: The total number of tokens restored across all blocks.
@discardableResult
public func restoreLayerData(from blocks: [CacheBlock], into cache: [any KVCache]) -> Int {
    guard let firstBlock = blocks.first, let firstData = firstBlock.cacheData else { return 0 }
    let numBlockLayers = firstData.count

    // Build mapping: block layer index → cache layer index
    // Only KVCacheSimple, QuantizedKVCache, TurboQuantKVCache, and CacheList-with-KV layers are KV-bearing
    var kvCacheIndices: [Int] = []
    for (i, layer) in cache.enumerated() {
        if layer is KVCacheSimple {
            kvCacheIndices.append(i)
        } else if layer is QuantizedKVCache {
            kvCacheIndices.append(i)
        } else if layer is TurboQuantKVCache {
            kvCacheIndices.append(i)
        } else if let cacheList = layer as? CacheList {
            // Check if any sub-cache is KV-bearing
            for j in 0..<cacheList.count {
                if cacheList[j] is KVCacheSimple || cacheList[j] is QuantizedKVCache
                    || cacheList[j] is TurboQuantKVCache
                {
                    kvCacheIndices.append(i)
                    break
                }
            }
        }
    }

    // Block layers should match KV-bearing cache layers
    guard numBlockLayers == kvCacheIndices.count else { return 0 }

    for (blockLayerIdx, cacheLayerIdx) in kvCacheIndices.enumerated() {
        var keySlices: [MLXArray] = []
        var valueSlices: [MLXArray] = []

        for block in blocks {
            guard let data = block.cacheData, blockLayerIdx < data.count,
                  let kv = data[blockLayerIdx] else { continue }
            keySlices.append(kv.keys)
            valueSlices.append(kv.values)
        }

        guard !keySlices.isEmpty else { continue }

        var restoredKeys = keySlices.count == 1 ? keySlices[0] : concatenated(keySlices, axis: 2)
        var restoredValues = valueSlices.count == 1 ? valueSlices[0] : concatenated(valueSlices, axis: 2)

        // Ensure restored KV matches bfloat16 (prevents dtype mismatch from stale
        // disk cache entries created before the universal bfloat16 conversion)
        if restoredKeys.dtype == .float16 {
            restoredKeys = restoredKeys.asType(.bfloat16)
            restoredValues = restoredValues.asType(.bfloat16)
        }

        if let simple = cache[cacheLayerIdx] as? KVCacheSimple {
            simple.state = [restoredKeys, restoredValues]
        } else if let quantizedCache = cache[cacheLayerIdx] as? QuantizedKVCache {
            let qKeys = quantized(restoredKeys, groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
            let qValues = quantized(restoredValues, groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
            var stateArrays: [MLXArray] = [qKeys.wq, qKeys.scales]
            if let biases = qKeys.biases { stateArrays.append(biases) }
            stateArrays.append(contentsOf: [qValues.wq, qValues.scales])
            if let biases = qValues.biases { stateArrays.append(biases) }
            quantizedCache.state = stateArrays
            quantizedCache.offset = restoredKeys.dim(2)
        } else if let tq = cache[cacheLayerIdx] as? TurboQuantKVCache {
            // 2026-05-01: route through restoreFromDecodedKV (NOT `state =`).
            // The previous `state =` path transitioned TQ to fill phase with
            // the lossy decoded float as the new prefill, which then re-
            // compressed at the next threshold cross — compounding the lossy
            // round each turn and producing visible token degeneracy on
            // multi-turn JANGTQ/MXFP4 conversations. The new method seats
            // the decoded float DIRECTLY as the compressed-phase prefix and
            // stays in `.compressed` so maybeQuantize skips it.
            tq.restoreFromDecodedKV(
                keys: restoredKeys, values: restoredValues,
                sourceOffset: restoredKeys.dim(2))
        } else if let cacheList = cache[cacheLayerIdx] as? CacheList {
            for i in 0..<cacheList.count {
                if let simple = cacheList[i] as? KVCacheSimple {
                    simple.state = [restoredKeys, restoredValues]
                    break
                }
                if let quantizedCache = cacheList[i] as? QuantizedKVCache {
                    let qKeys = quantized(restoredKeys, groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
                    let qValues = quantized(restoredValues, groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
                    var stateArrays: [MLXArray] = [qKeys.wq, qKeys.scales]
                    if let biases = qKeys.biases { stateArrays.append(biases) }
                    stateArrays.append(contentsOf: [qValues.wq, qValues.scales])
                    if let biases = qValues.biases { stateArrays.append(biases) }
                    quantizedCache.state = stateArrays
                    quantizedCache.offset = restoredKeys.dim(2)
                    break
                }
                if let tq = cacheList[i] as? TurboQuantKVCache {
                    // Same compounding-quantization fix as the top-level
                    // TurboQuantKVCache branch above.
                    tq.restoreFromDecodedKV(
                        keys: restoredKeys, values: restoredValues,
                        sourceOffset: restoredKeys.dim(2))
                    break
                }
            }
        }
    }

    let totalTokens = blocks.reduce(0) { $0 + $1.tokenCount }
    if let companion = blocks.last?.boundaryCompanionData {
        var mutableCache = cache
        let companionTokens = restoreFromDiskArrays(companion, into: &mutableCache)
        guard companionTokens == totalTokens else { return 0 }
    }
    return totalTokens
}

// MARK: - SSM State Extraction

/// Extract SSM (MambaCache/ArraysCache) states from a model's cache array.
///
/// Returns the state arrays from each SSM layer. Non-SSM layers are skipped.
/// Used to populate ``SSMStateCache`` for hybrid model companion storage.
///
/// - Parameter cache: The model's per-layer cache array.
/// - Returns: All SSM state arrays, flattened across layers.
public func extractSSMStates(from cache: [any KVCache]) -> [MLXArray] {
    var states: [MLXArray] = []
    for layer in cache {
        if let mamba = layer as? MambaCache {
            // MambaCache.state returns [conv_state, hidden_state]
            states.append(contentsOf: mamba.state)
        } else if let arrays = layer as? ArraysCache {
            states.append(contentsOf: arrays.state)
        } else if layer is ZayaCCACache {
            // ZAYA CCA state is path-dependent, but it is not SSM state.
            // TQDiskSerializer's LayerKind.zayaCCA stores keys, values,
            // conv_state, and prev_hs together as one disk-backed layer.
            // Sending those arrays through SSMStateCache as a second
            // companion path duplicates ownership and can start another
            // Metal eval while the ZAYA disk payload is still draining.
            continue
        } else if let cacheList = layer as? CacheList {
            // Extract SSM sub-cache from composite layers
            for i in 0..<cacheList.count {
                if let mamba = cacheList[i] as? MambaCache {
                    states.append(contentsOf: mamba.state)
                } else if let arrays = cacheList[i] as? ArraysCache {
                    states.append(contentsOf: arrays.state)
                }
            }
        }
    }
    return states
}

// MARK: - SSM State Restoration

/// Restore SSM states into a model's cache array.
///
/// The `states` array should match the output order of ``extractSSMStates(from:)``.
/// Each MambaCache consumes 2 state arrays (conv state + hidden state).
///
/// - Parameters:
///   - states: The SSM state arrays to restore.
///   - cache: The model's per-layer cache array to restore into.
///   - boundary: Exact prompt boundary represented by `states`. Recurrent
///     caches use this logical offset when constructing their next-step mask;
///     restoring tensors without it is an invalid partial restore.
public func restoreSSMStates(
    _ states: [MLXArray],
    into cache: [any KVCache],
    boundary: Int? = nil
) {
    // `extractSSMStates` emits each MambaCache's OCCUPIED slots: full Mamba
    // layers (Nemotron-H, Jamba) contribute 2 arrays, LFM2/LFM2.5 short-conv
    // layers contribute exactly 1 (slot 1 is never written). A fresh restore
    // target has empty state, so the per-layer arity must be recovered from
    // the list itself: every mamba layer in one companion list shares one
    // arity, so the total length disambiguates. A list matching neither
    // layout is refused outright — consuming a best-effort prefix cross-wired
    // layer N with layer N+1's state and left the tail layers empty.
    var totalWithTwoSlotMamba = 0
    var totalWithOneSlotMamba = 0
    for layer in cache {
        if layer is MambaCache {
            totalWithTwoSlotMamba += 2
            totalWithOneSlotMamba += 1
        } else if let arrays = layer as? ArraysCache {
            totalWithTwoSlotMamba += arrays.slotCount
            totalWithOneSlotMamba += arrays.slotCount
        } else if let cacheList = layer as? CacheList {
            for i in 0..<cacheList.count {
                if cacheList[i] is MambaCache {
                    totalWithTwoSlotMamba += 2
                    totalWithOneSlotMamba += 1
                } else if let arrays = cacheList[i] as? ArraysCache {
                    totalWithTwoSlotMamba += arrays.slotCount
                    totalWithOneSlotMamba += arrays.slotCount
                }
            }
        }
    }
    let mambaArity: Int
    if states.count == totalWithTwoSlotMamba {
        mambaArity = 2
    } else if states.count == totalWithOneSlotMamba {
        mambaArity = 1
    } else {
        // Unknown companion layout (e.g. a stale entry from a run whose
        // model layout has drifted). Restoring nothing is safe — the caller
        // re-prefills; restoring a misaligned prefix is silent corruption.
        return
    }

    var stateIdx = 0
    for layer in cache {
        if let mamba = layer as? MambaCache {
            if stateIdx + mambaArity <= states.count {
                if mambaArity == 2 {
                    mamba.state = Array(states[stateIdx..<(stateIdx + 2)])
                        .map { $0[.ellipsis] }
                } else {
                    // Single-slot short-conv layout: keep the 2-slot
                    // container geometry, occupy only slot 0.
                    mamba[0] = states[stateIdx][.ellipsis]
                }
                if let boundary { mamba.offset = boundary }
                stateIdx += mambaArity
            }
        } else if let arrays = layer as? ArraysCache {
            // ArraysCache (GatedDeltaNet / linear-attention recurrence, e.g.
            // qwen3.5/ornith). Use `slotCount` (the fixed number of state
            // slots), NOT `state.count`: a FRESH cache (the disk-restore target
            // is `model.newCache`) has all-nil slots, so `state` — which is
            // `cache.compactMap { $0 }` — returns [] and `state.count == 0`.
            // Gating on `state.count > 0` (the old code) therefore SKIPPED
            // restore into every fresh cache, leaving the GatedDeltaNet state
            // empty so the next turn's prefill built its recurrence from only
            // the post-boundary tokens → wrong output. A prefilled cache stores
            // exactly `slotCount` companion arrays, so this consumes the right
            // span. (Mamba's branch already special-cases the empty cache with
            // its fixed 2-slot layout; this is the ArraysCache analogue.)
            let slotCount = arrays.slotCount
            if slotCount > 0, stateIdx + slotCount <= states.count {
                arrays.state = Array(states[stateIdx..<(stateIdx + slotCount)])
                    .map { $0[.ellipsis] }
                if let boundary { arrays.offset = boundary }
                stateIdx += slotCount
            }
        } else if layer is ZayaCCACache {
            // ZAYA CCA restore is owned by the LayerKind.zayaCCA disk payload,
            // not by the generic SSM companion list.
            continue
        } else if let cacheList = layer as? CacheList {
            for i in 0..<cacheList.count {
                if let mamba = cacheList[i] as? MambaCache {
                    if stateIdx + mambaArity <= states.count {
                        if mambaArity == 2 {
                            mamba.state = Array(states[stateIdx..<(stateIdx + 2)])
                                .map { $0[.ellipsis] }
                        } else {
                            mamba[0] = states[stateIdx][.ellipsis]
                        }
                        if let boundary { mamba.offset = boundary }
                        stateIdx += mambaArity
                    }
                } else if let arrays = cacheList[i] as? ArraysCache {
                    let slotCount = arrays.slotCount
                    if slotCount > 0, stateIdx + slotCount <= states.count {
                        arrays.state = Array(states[stateIdx..<(stateIdx + slotCount)])
                            .map { $0[.ellipsis] }
                        if let boundary { arrays.offset = boundary }
                        stateIdx += slotCount
                    }
                }
            }
        }
    }
}

// MARK: - Disk Cache KV Restoration

/// Restore KV state (and, for hybrid models, Mamba SSM state) from a disk
/// cache arrays dictionary back into the model's per-layer cache array.
///
/// ## Format handling
///
/// Two on-disk formats are supported:
///
/// - **Version 2 (current)** — produced by `TQDiskSerializer.serialize`.
///   Each layer has an explicit `__layer_kind_{i}__` tag so mixed-kind
///   caches (hybrid attention + Mamba) round-trip correctly. Restored into
///   `cache[i]` by real cache position.
///
/// - **Version 1 (legacy block format)** — old entries using
///   `b{block}_l{layer}_keys/values`. Handled via the KV-only restoration
///   path. Hybrid models can't restore from v1 entries (the Mamba layers
///   were effectively corrupt), so they get a silent 0-token miss and
///   re-prefill. Acceptable — v1 entries expire naturally as new v2 entries
///   overwrite them.
///
/// - Parameters:
///   - arrays: The disk cache dictionary loaded via `DiskCache.fetch()`.
///   - cache: The model's per-layer KV cache array to restore into.
/// - Returns: The total number of tokens restored, measured from the first
///   attention layer's key tensor sequence dim, or `0` if nothing matched.
@discardableResult
public func restoreFromDiskArrays(_ arrays: [String: MLXArray], into cache: inout [any KVCache]) -> Int {
    let version = TQDiskSerializer.formatVersion(of: arrays)
    if version >= 2 {
        return restoreFromV2Arrays(arrays, into: &cache)
    }
    return restoreFromLegacyArrays(arrays, into: cache)
}

/// Restore a format v2 disk dictionary into the model cache.
///
/// The serializer tags each layer with an authoritative `LayerKind`, so this
/// path can restore attention layers, Mamba SSM layers, and skipped layers
/// independently and by real cache index.
@discardableResult
private func restoreFromV2Arrays(
    _ arrays: [String: MLXArray],
    into cache: inout [any KVCache]
) -> Int {
    let indexed = TQDiskSerializer.deserializeIndexed(arrays)
    guard !indexed.isEmpty else { return 0 }

    // A record that describes FEWER layers than the runtime cache has is a
    // truncated record, and must miss.
    //
    // `deserializeIndexed` detects a GAP in the layer-kind tags, but a gap
    // requires a tag on both sides of it. Losing the TRAILING tags — which is
    // what a partially-flushed write actually leaves behind — creates no gap:
    // the contiguous walk simply ends early, and the damaged payload becomes a
    // smaller well-formed one. The surviving layers then restore, seed
    // `totalTokens`, and the caller is told the entry hit while the tail
    // layers sit at offset 0. Attention runs against empty layers and the
    // model produces confident wrong text instead of re-prefilling.
    //
    // The serializer writes a tag for EVERY layer, `.skip` included, so a
    // well-formed record always describes exactly `cache.count` layers. Only
    // the restore path knows that count, which is why this check cannot live
    // in the deserializer. Refusing costs one re-prefill.
    guard indexed.count >= cache.count else { return 0 }

    // ZAYA disk records are an all-or-nothing prompt-boundary snapshot. Restore
    // every typed layer into a private copy first, including encoded TQ state,
    // so a corrupt later CCA layer cannot leave the caller's cache half-seated.
    var validatedZayaIndices = Set<Int>()
    var zayaBoundary: Int?
    var validatedDeepseekV4Indices = Set<Int>()
    var deepseekV4Boundary: Int?
    for entry in indexed {
        guard entry.index < cache.count else {
            if case .deepseekV4 = entry.data { return 0 }
            if case .zayaCCA = entry.data { return 0 }
            if case .zayaCCATQ = entry.data { return 0 }
            if case .qkv = entry.data { return 0 }
            continue
        }
        switch entry.data {
        case .qkv(let comp):
            // Quantized-KV layers share the atomic contract: a record whose
            // group size / bit width no longer matches the runtime cache (or
            // whose layer class drifted) must refuse BEFORE any live layer is
            // seated. The apply loop below seeds `totalTokens` from sibling
            // layers, so skipping just this layer would report a "hit" that
            // leaves it empty.
            guard canRestoreQKVLayer(comp, into: cache[entry.index]) else { return 0 }
        case .deepseekV4(let comp):
            let staged = cache[entry.index].copy()
            guard canRestoreDeepseekV4Layer(comp, into: staged),
                  deepseekV4Boundary == nil || deepseekV4Boundary == comp.offset,
                  restoreDeepseekV4Layer(comp, into: staged)
            else {
                return 0
            }
            deepseekV4Boundary = comp.offset
            validatedDeepseekV4Indices.insert(entry.index)
        case .zayaCCA(let comp):
            let staged = cache[entry.index].copy()
            guard canRestoreZayaCCALayer(comp, into: staged),
                  zayaBoundary == nil || zayaBoundary == comp.offset,
                  restoreZayaCCALayer(comp, into: staged)
            else {
                return 0
            }
            zayaBoundary = comp.offset
            validatedZayaIndices.insert(entry.index)
        case .zayaCCATQ(let comp):
            let staged = cache[entry.index].copy()
            guard canRestoreZayaCCATQLayer(comp, into: staged),
                  zayaBoundary == nil || zayaBoundary == comp.tq.offset,
                  restoreZayaCCATQLayer(comp, into: staged)
            else {
                return 0
            }
            zayaBoundary = comp.tq.offset
            validatedZayaIndices.insert(entry.index)
        case .requiredMiss:
            return 0
        default:
            break
        }
    }

    var totalTokens = 0

    for entry in indexed {
        let i = entry.index
        guard i < cache.count else { continue }

        switch entry.data {
        case .standard(let kv):
            var keys = kv.keys
            var values = kv.values
            if keys.dtype == .float16 {
                keys = keys.asType(.bfloat16)
                values = values.asType(.bfloat16)
            }
            // Defensive shape guard — vmlx #68 / SmallVector out of range.
            // A 2D disk-stored layer (shape.count < 3) would crash on
            // `.dim(2)` below. Skip the whole restore as a clean miss so
            // the caller re-prefills instead of fatal-trapping. Belt &
            // suspenders on top of v2 layer-kind tagging.
            guard keys.shape.count >= 3, values.shape.count >= 3 else {
                FileHandle.standardError.write(Data(
                    "[disk cache] restore SKIPPED: layer \(i) has incompatible shape (k=\(keys.shape) v=\(values.shape)). Need >= 3D. Falling back to fresh prefill.\n".utf8))
                return 0
            }
            if totalTokens == 0 {
                totalTokens = keys.dim(2)
            }
            restoreKVLayer(keys: keys, values: values, into: cache[i])

        case .mamba(let comp):
            // Mamba state arrays are cumulative — no sequence dim to
            // measure, so they don't contribute to `totalTokens`. The
            // attention side already provides that number.
            restoreMambaLayer(comp, into: cache[i])

        case .tq(let comp):
            // Restore the compressed prefix into the existing
            // TurboQuantKVCache instance (or a TQ inside a CacheList).
            // A fresh request starts from KVCacheSimple even when the
            // coordinator resolved a TurboQuant KV policy. Materialize the
            // TQ layer from the disk payload before reporting a cache hit;
            // otherwise the caller would seed decode from an empty cache.
            // A failed seat must refuse the WHOLE record, exactly as the
            // .qkv / .deepseekV4 / .zayaCCA paths below already do. `continue`
            // left this layer empty while sibling layers seeded `totalTokens`,
            // so the caller was told the entry hit and then decoded against an
            // empty TQ layer -- the silent attention corruption the .qkv case
            // documents. A miss costs a re-prefill; this cost a wrong answer.
            guard restoreTQLayer(comp, into: &cache[i]) else { return 0 }
            // TQ layers don't expose a sequence-dim tensor in the same way
            // as KV layers, so they can't drive `totalTokens`. The
            // attention sequence length is sourced from the .standard
            // entries; if a model is purely TQ-compressed we fall back to
            // the offset stored in the components.
            if totalTokens == 0 {
                totalTokens = comp.offset
            }

        case .qkv(let comp):
            // Restore quantized KV state. The runtime's QuantizedKVCache
            // must agree on group size and bit width (otherwise the qweight
            // shapes won't line up). A fresh request starts from
            // KVCacheSimple, so materialize the quantized layer from the
            // disk payload before reporting a hit.
            //
            // A failed seat (group-size/bits mismatch or drifted layer
            // class) must refuse the WHOLE record: `totalTokens` is seeded
            // from sibling layers, so skipping just this layer reported a
            // "hit" that left it empty — silent attention corruption for
            // the restored prefix. Same atomic contract as .deepseekV4 /
            // .zayaCCA below.
            guard restoreQKVLayer(comp, into: &cache[i]) else { return 0 }
            if totalTokens == 0 {
                totalTokens = comp.offset
            }

        case .rotating(let comp):
            // Restore RotatingKVCache (sliding-window attention). Reseats
            // the ring buffer + 5-tuple metaState (keep, maxSize, step,
            // offset, idx) so the wrap position survives the restart.
            // SLIDING-1 (2026-04-15): closes the central skip in
            // CacheCoordinator.swift that previously dropped Gemma4 SWA,
            // Mistral4-with-maxKVSize, MiMoV2Flash, BaichuanM1.
            restoreRotatingLayer(comp, into: cache[i])
            if totalTokens == 0 {
                totalTokens = comp.offset
            }

        case .deepseekV4(let comp):
            // 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
            // Restore the rotating window AND the compressor + indexer
            // pool tensors + per-branch incomplete-window buffer state
            // so multi-turn /v1/chat/completions prefix-cache reuse
            // doesn't have to re-derive the long-context summary from
            // prompt tokens every turn.
            guard validatedDeepseekV4Indices.contains(i),
                  restoreDeepseekV4Layer(comp, into: cache[i])
            else { return 0 }
            if totalTokens == 0 {
                totalTokens = comp.offset
            }

        case .zayaCCA(let comp):
            // 2026-05-06 (ZAYA1 CCA-attention port):
            // Restore the four-array state (keys, values, conv_state,
            // prev_hs) as one unit. KV-only restore would be a false hit
            // because conv_state and prev_hs are path-dependent.
            guard validatedZayaIndices.contains(i),
                  restoreZayaCCALayer(comp, into: cache[i])
            else { return 0 }
            if totalTokens == 0 {
                totalTokens = comp.offset
            }

        case .zayaCCATQ(let comp):
            // Atomic restore: encoded attention KV and native fp32 CCA state
            // must both seat successfully at the same prompt boundary.
            guard validatedZayaIndices.contains(i),
                  restoreZayaCCATQLayer(comp, into: cache[i])
            else { return 0 }
            if totalTokens == 0 {
                totalTokens = comp.tq.offset
            }

        case .requiredMiss:
            return 0

        case .cacheList(let subLayers):
            // CacheList composite (BaichuanM1, FalconH1, MiMoV2Flash
            // hybrid stacks). Dispatch each sub-LayerData via the
            // existing per-type helpers using the FULL CacheList
            // (`cache[i]`) as `into:` — restore* helpers already
            // introspect CacheList sub-caches by type and find the
            // matching sub-slot.
            //
            // Sub-layer ORDER on disk does NOT have to match the
            // runtime CacheList's internal order — type matching does
            // the dispatch. This is the same convention SSM state
            // restoration in extractSSMStates uses.
            for subData in subLayers {
                switch subData {
                case .standard(let kv):
                    var keys = kv.keys
                    var values = kv.values
                    if keys.dtype == .float16 {
                        keys = keys.asType(.bfloat16)
                        values = values.asType(.bfloat16)
                    }
                    guard keys.shape.count >= 3, values.shape.count >= 3 else {
                        continue
                    }
                    if totalTokens == 0 {
                        totalTokens = keys.dim(2)
                    }
                    restoreKVLayer(keys: keys, values: values, into: cache[i])

                case .mamba(let comp):
                    restoreMambaLayer(comp, into: cache[i])

                case .rotating(let comp):
                    restoreRotatingLayer(comp, into: cache[i])
                    if totalTokens == 0 {
                        totalTokens = comp.offset
                    }

                case .tq, .qkv, .deepseekV4, .zayaCCA, .zayaCCATQ, .cacheList,
                     .requiredMiss, .skip:
                    // .skip is a per-sub no-op (sub-cache had no
                    // persistable state). The other cases are not
                    // currently emitted as sub-cache kinds — see
                    // TQDiskSerializer.deserializeCacheListLayer.
                    continue
                }
            }

        case .skip:
            // Cache type we don't know how to persist. No-op.
            continue
        }
    }

    return totalTokens
}

/// Legacy v1 block-format restore. KV-only. Hybrid-unsafe; if any Mamba
/// layer is in `cache`, the counts won't line up and this function returns
/// 0, forcing a re-prefill (which is correct, because old v1 entries never
/// captured Mamba state properly anyway).
@discardableResult
private func restoreFromLegacyArrays(
    _ arrays: [String: MLXArray],
    into cache: [any KVCache]
) -> Int {
    var kvByLayer: [Int: (keys: MLXArray, values: MLXArray)] = [:]

    if TQDiskSerializer.isTQNative(arrays) {
        // Legacy v1 "TQ-native" dict — flat deserialize, ignore tq entries.
        let layers = TQDiskSerializer.deserialize(arrays)
        for (i, layerData) in layers.enumerated() {
            if case .standard(let kv) = layerData {
                kvByLayer[i] = (keys: kv.keys, values: kv.values)
            }
        }
    } else {
        // Old block-layer format: b{block}_l{layer}_keys / b{block}_l{layer}_values.
        var layerBlocks: [Int: [(blockIdx: Int, keys: MLXArray, values: MLXArray)]] = [:]

        for (key, array) in arrays {
            guard key.hasSuffix("_keys") else { continue }
            let base = String(key.dropLast(5))
            let parts = base.split(separator: "_")
            guard parts.count == 2,
                  parts[0].hasPrefix("b"), parts[1].hasPrefix("l"),
                  let blockIdx = Int(parts[0].dropFirst()),
                  let layerIdx = Int(parts[1].dropFirst())
            else { continue }

            let valuesKey = "b\(blockIdx)_l\(layerIdx)_values"
            guard let valuesArray = arrays[valuesKey] else { continue }

            layerBlocks[layerIdx, default: []].append(
                (blockIdx: blockIdx, keys: array, values: valuesArray))
        }

        for (layerIdx, blocks) in layerBlocks {
            let sorted = blocks.sorted { $0.blockIdx < $1.blockIdx }
            let keySlices = sorted.map(\.keys)
            let valueSlices = sorted.map(\.values)

            let concatKeys = keySlices.count == 1
                ? keySlices[0] : concatenated(keySlices, axis: 2)
            let concatValues = valueSlices.count == 1
                ? valueSlices[0] : concatenated(valueSlices, axis: 2)

            kvByLayer[layerIdx] = (keys: concatKeys, values: concatValues)
        }
    }

    guard !kvByLayer.isEmpty else { return 0 }

    // Build mapping of KV-bearing cache layer indices (same logic as restoreLayerData).
    var kvCacheIndices: [Int] = []
    for (i, layer) in cache.enumerated() {
        if layer is KVCacheSimple {
            kvCacheIndices.append(i)
        } else if layer is QuantizedKVCache {
            kvCacheIndices.append(i)
        } else if layer is TurboQuantKVCache {
            kvCacheIndices.append(i)
        } else if let cacheList = layer as? CacheList {
            for j in 0..<cacheList.count {
                if cacheList[j] is KVCacheSimple || cacheList[j] is QuantizedKVCache
                    || cacheList[j] is TurboQuantKVCache
                {
                    kvCacheIndices.append(i)
                    break
                }
            }
        }
    }

    // Legacy v1 entries have no layer-kind metadata, so hybrid models can't
    // be restored safely. Counts mismatch → abort and force re-prefill.
    let sortedLayers = kvByLayer.keys.sorted()
    guard sortedLayers.count == kvCacheIndices.count else { return 0 }

    var totalTokens = 0

    for (diskLayerIdx, cacheLayerIdx) in zip(sortedLayers, kvCacheIndices) {
        guard var (restoredKeys, restoredValues) = kvByLayer[diskLayerIdx] else { continue }

        if restoredKeys.dtype == .float16 {
            restoredKeys = restoredKeys.asType(.bfloat16)
            restoredValues = restoredValues.asType(.bfloat16)
        }

        if totalTokens == 0 {
            totalTokens = restoredKeys.dim(2)
        }

        restoreKVLayer(keys: restoredKeys, values: restoredValues, into: cache[cacheLayerIdx])
    }

    return totalTokens
}

/// Helper: restore a pair of KV tensors into whatever KV-bearing cache
/// class `layer` happens to be. Mirrors the behavior of the original
/// legacy path but factored out so both v1 and v2 restores share it.
private func restoreKVLayer(
    keys restoredKeys: MLXArray,
    values restoredValues: MLXArray,
    into layer: any KVCache
) {
    if let simple = layer as? KVCacheSimple {
        simple.state = [restoredKeys, restoredValues]
    } else if let quantizedCache = layer as? QuantizedKVCache {
        let qKeys = quantized(
            restoredKeys, groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
        let qValues = quantized(
            restoredValues, groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
        var stateArrays: [MLXArray] = [qKeys.wq, qKeys.scales]
        if let biases = qKeys.biases { stateArrays.append(biases) }
        stateArrays.append(contentsOf: [qValues.wq, qValues.scales])
        if let biases = qValues.biases { stateArrays.append(biases) }
        quantizedCache.state = stateArrays
        quantizedCache.offset = restoredKeys.dim(2)
    } else if let tq = layer as? TurboQuantKVCache {
        // Same restoreFromDecodedKV path as paged-cache restore — avoid
        // compounding quantization on cross-turn round trips.
        tq.restoreFromDecodedKV(
            keys: restoredKeys, values: restoredValues,
            sourceOffset: restoredKeys.dim(2))
    } else if let cacheList = layer as? CacheList {
        for i in 0..<cacheList.count {
            if let simple = cacheList[i] as? KVCacheSimple {
                simple.state = [restoredKeys, restoredValues]
                return
            }
            if let quantizedCache = cacheList[i] as? QuantizedKVCache {
                let qKeys = quantized(
                    restoredKeys, groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
                let qValues = quantized(
                    restoredValues,
                    groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
                var stateArrays: [MLXArray] = [qKeys.wq, qKeys.scales]
                if let biases = qKeys.biases { stateArrays.append(biases) }
                stateArrays.append(contentsOf: [qValues.wq, qValues.scales])
                if let biases = qValues.biases { stateArrays.append(biases) }
                quantizedCache.state = stateArrays
                quantizedCache.offset = restoredKeys.dim(2)
                return
            }
            if let tq = cacheList[i] as? TurboQuantKVCache {
                tq.restoreFromDecodedKV(
                    keys: restoredKeys, values: restoredValues,
                    sourceOffset: restoredKeys.dim(2))
                return
            }
        }
    }
}

/// Helper: restore quantized-KV state into a `QuantizedKVCache` layer.
/// Verifies group size + bit width match the runtime cache before
/// touching state — a mismatch means the model was reconfigured since the
/// disk entry was written and the qweight shapes will not align, so the
/// only safe behavior is to no-op and let the caller re-prefill.
/// Non-mutating twin of `restoreQKVLayer` used by the atomic pre-validation
/// pass: mirrors exactly the conditions under which the restore would fail,
/// without seating any state.
private func canRestoreQKVLayer(
    _ comp: TQDiskSerializer.QKVLayerComponents,
    into layer: any KVCache
) -> Bool {
    if let qkv = layer as? QuantizedKVCache {
        return qkv.groupSize == comp.groupSize && qkv.bits == comp.bits
    }
    if layer is KVCacheSimple { return true }
    if let cacheList = layer as? CacheList {
        for i in 0..<cacheList.count {
            if let qkv = cacheList[i] as? QuantizedKVCache {
                return qkv.groupSize == comp.groupSize && qkv.bits == comp.bits
            }
            if cacheList[i] is KVCacheSimple { return true }
        }
    }
    return false
}

private func restoreQKVLayer(
    _ comp: TQDiskSerializer.QKVLayerComponents,
    into layer: inout any KVCache
) -> Bool {
    if let qkv = layer as? QuantizedKVCache {
        guard qkv.groupSize == comp.groupSize, qkv.bits == comp.bits else { return false }
        qkv.state = comp.stateArrays
        qkv.offset = comp.offset
        return true
    }
    if layer is KVCacheSimple {
        let qkv = QuantizedKVCache(groupSize: comp.groupSize, bits: comp.bits)
        qkv.state = comp.stateArrays
        qkv.offset = comp.offset
        layer = qkv
        return true
    }
    if let cacheList = layer as? CacheList {
        for i in 0..<cacheList.count {
            if let qkv = cacheList[i] as? QuantizedKVCache {
                guard qkv.groupSize == comp.groupSize, qkv.bits == comp.bits else {
                    return false
                }
                qkv.state = comp.stateArrays
                qkv.offset = comp.offset
                return true
            }
            if cacheList[i] is KVCacheSimple {
                let qkv = QuantizedKVCache(groupSize: comp.groupSize, bits: comp.bits)
                qkv.state = comp.stateArrays
                qkv.offset = comp.offset
                cacheList.caches[i] = qkv
                return true
            }
        }
    }
    return false
}

/// Helper: restore TQ-compressed state into a `TurboQuantKVCache` layer
/// (or a `CacheList` containing one). Fresh request caches start as
/// `KVCacheSimple`, so this materializes the TQ layer before reporting a hit.
private func restoreTQLayer(
    _ comp: TQDiskSerializer.TQLayerComponents,
    into layer: inout any KVCache
) -> Bool {
    if let tq = layer as? TurboQuantKVCache {
        tq.restoreCompressed(
            encodedKeys: comp.encodedKeys,
            encodedValues: comp.encodedValues,
            sourceOffset: comp.offset,
            windowKeys: comp.windowKeys,
            windowValues: comp.windowValues
        )
        return tq.offset > 0
    }
    if layer is KVCacheSimple {
        let tq = makeTurboQuantLayer(from: comp)
        tq.restoreCompressed(
            encodedKeys: comp.encodedKeys,
            encodedValues: comp.encodedValues,
            sourceOffset: comp.offset,
            windowKeys: comp.windowKeys,
            windowValues: comp.windowValues
        )
        guard tq.offset > 0 else { return false }
        layer = tq
        return true
    }
    if let cacheList = layer as? CacheList {
        for i in 0..<cacheList.count {
            if let tq = cacheList[i] as? TurboQuantKVCache {
                tq.restoreCompressed(
                    encodedKeys: comp.encodedKeys,
                    encodedValues: comp.encodedValues,
                    sourceOffset: comp.offset,
                    windowKeys: comp.windowKeys,
                    windowValues: comp.windowValues
                )
                return tq.offset > 0
            }
            if cacheList[i] is KVCacheSimple {
                let tq = makeTurboQuantLayer(from: comp)
                tq.restoreCompressed(
                    encodedKeys: comp.encodedKeys,
                    encodedValues: comp.encodedValues,
                    sourceOffset: comp.offset,
                    windowKeys: comp.windowKeys,
                    windowValues: comp.windowValues
                )
                guard tq.offset > 0 else { return false }
                cacheList.caches[i] = tq
                return true
            }
        }
    }
    return false
}

private func makeTurboQuantLayer(
    from comp: TQDiskSerializer.TQLayerComponents
) -> TurboQuantKVCache {
    let keyBits = comp.encodedKeys.indexBits + 1
    let valueBits = comp.encodedValues.indexBits
    let sinkTokens = max(comp.encodedKeys.sinkCount, comp.encodedValues.sinkCount, 4)
    let residualTokens = max(
        comp.encodedKeys.tailCount,
        comp.encodedValues.tailCount,
        TurboQuantKVCache.defaultResidualTokens)
    return TurboQuantKVCache(
        keyBits: keyBits,
        valueBits: valueBits,
        sinkTokens: sinkTokens,
        residualTokens: residualTokens)
}

/// Helper: restore RotatingKVCache state (ring buffer + metaState) into a
/// `RotatingKVCache` layer (or a `CacheList` containing one). The ring
/// buffer keys/values are reseated via `state =` and the wrap position
/// via `metaState = [keep, maxSize, step, offset, idx]` so generation
/// continues at exactly the same idx pointer. Silently no-ops on type
/// mismatch — caller falls back to re-prefill on this layer.
private func restoreRotatingLayer(
    _ comp: TQDiskSerializer.RotatingLayerComponents,
    into layer: any KVCache
) {
    func apply(_ rot: RotatingKVCache) {
        rot.state = [comp.keys, comp.values]
        rot.metaState = [
            String(comp.keep),
            String(comp.maxSize),
            String(comp.step),
            String(comp.offset),
            String(comp.idx),
        ]
    }
    if let rot = layer as? RotatingKVCache {
        apply(rot)
        return
    }
    // Composite cache that wraps a RotatingKVCache (e.g. DeepseekV4Cache).
    // Restores the inner rotating state; the wrapper's ephemeral buffer
    // state is cleared and will be repopulated on the next prefill.
    if let wrapper = layer as? RotatingKVCacheWrapper {
        apply(wrapper.rotating)
        return
    }
    if let cacheList = layer as? CacheList {
        for i in 0..<cacheList.count {
            if let rot = cacheList[i] as? RotatingKVCache {
                apply(rot)
                return
            }
            if let wrapper = cacheList[i] as? RotatingKVCacheWrapper {
                apply(wrapper.rotating)
                return
            }
        }
    }
}

/// 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
/// Restore a full hybrid `DeepseekV4Cache` layer — rotating window
/// state PLUS compressor + indexer pool tensors + per-branch
/// incomplete-window buffer state. Quantized pools stay encoded; they are not
/// expanded into a duplicate BF16 payload during restore.
private func canRestoreDeepseekV4Layer(
    _ comp: TQDiskSerializer.DeepseekV4LayerComponents,
    into layer: any KVCache
) -> Bool {
    guard let hybrid = layer as? HybridPoolCache,
          hybrid.compressRatio == comp.compressRatio,
          hybrid.slidingWindow == comp.slidingWindow,
          comp.keys.ndim >= 3,
          comp.values.ndim >= 3,
          comp.keys.shape == comp.values.shape,
          comp.offset >= 0,
          comp.maxSize == hybrid.rotating.maxSize
    else { return false }

    func validPool(_ pool: MLXArray?) -> Bool {
        guard let pool else { return true }
        return pool.ndim == 3 && pool.dim(0) > 0 && pool.dim(1) > 0 && pool.dim(2) > 0
    }
    guard validPool(comp.poolComp), validPool(comp.poolIdx) else { return false }

    if comp.quantizedPoolComp != nil || comp.quantizedPoolIdx != nil {
        guard layer is QuantizedHybridPoolCache else { return false }
    }
    return true
}

private func restoreDeepseekV4Layer(
    _ comp: TQDiskSerializer.DeepseekV4LayerComponents,
    into layer: any KVCache
) -> Bool {
    guard canRestoreDeepseekV4Layer(comp, into: layer),
          let hybrid = layer as? HybridPoolCache
    else { return false }

    func apply(_ hybrid: HybridPoolCache) -> Bool {
        hybrid.rotating.state = [comp.keys, comp.values]
        hybrid.rotating.metaState = [
            String(comp.keep),
            String(comp.maxSize),
            String(comp.step),
            String(comp.offset),
            String(comp.idx),
        ]
        if let segments = comp.quantizedPoolComp {
            guard let quantized = hybrid as? QuantizedHybridPoolCache else { return false }
            quantized.setHybridPoolQuantizedSegments(branch: .compressor, segments: segments)
        } else {
            hybrid.setHybridPool(branch: .compressor, value: comp.poolComp)
        }
        if let segments = comp.quantizedPoolIdx {
            guard let quantized = hybrid as? QuantizedHybridPoolCache else { return false }
            quantized.setHybridPoolQuantizedSegments(branch: .indexer, segments: segments)
        } else {
            hybrid.setHybridPool(branch: .indexer, value: comp.poolIdx)
        }
        hybrid.setHybridBuffers(
            branch: .compressor,
            kv: comp.bufCompKV, gate: comp.bufCompGate)
        hybrid.setHybridBuffers(
            branch: .indexer,
            kv: comp.bufIdxKV, gate: comp.bufIdxGate)
        return true
    }
    return apply(hybrid)
}

/// 2026-05-06 (ZAYA1 CCA-attention port):
/// Restore a full `ZayaCCACache` layer — keys, values, conv_state, prev_hs
/// — as one unit. The KV slots may be zero-seq sentinels (empty source
/// cache); the cache's setter handles that case without instantiating an
/// empty KVCacheSimple buffer. Silently no-ops on type mismatch so the
/// caller falls back to fresh prefill.
private func canRestoreZayaCCALayer(
    _ comp: TQDiskSerializer.ZayaCCALayerComponents,
    into layer: any KVCache
) -> Bool {
    guard let zaya = layer as? ZayaCCACache,
          zaya.convChannels == comp.convChannels,
          zaya.hiddenSize == comp.hiddenSize,
          zaya.batchSize == comp.batchSize,
          comp.keys.ndim >= 3,
          comp.values.ndim >= 3,
          comp.keys.dim(2) == comp.values.dim(2),
          comp.keys.dim(2) == comp.offset,
          comp.convState.shape == [comp.batchSize, comp.convChannels, 2],
          comp.prevHS.shape == [comp.batchSize, comp.hiddenSize]
    else { return false }
    return true
}

private func restoreZayaCCALayer(
    _ comp: TQDiskSerializer.ZayaCCALayerComponents,
    into layer: any KVCache
) -> Bool {
    guard let zaya = layer as? ZayaCCACache,
          canRestoreZayaCCALayer(comp, into: layer)
    else { return false }
    zaya.state = [comp.keys, comp.values, comp.convState, comp.prevHS]
    return zaya.offset == comp.offset
}

private func canRestoreZayaCCATQLayer(
    _ comp: TQDiskSerializer.ZayaCCATQLayerComponents,
    into layer: any KVCache
) -> Bool {
    guard let zaya = layer as? ZayaCCACache,
          zaya.convChannels == comp.convChannels,
          zaya.hiddenSize == comp.hiddenSize,
          zaya.batchSize == comp.batchSize,
          comp.tq.offset > 0,
          comp.convState.shape == [comp.batchSize, comp.convChannels, 2],
          comp.prevHS.shape == [comp.batchSize, comp.hiddenSize]
    else { return false }
    return true
}

private func restoreZayaCCATQLayer(
    _ comp: TQDiskSerializer.ZayaCCATQLayerComponents,
    into layer: any KVCache
) -> Bool {
    guard let zaya = layer as? ZayaCCACache,
          canRestoreZayaCCATQLayer(comp, into: layer),
          zaya.restoreTurboQuantAttentionKV(
              encodedKeys: comp.tq.encodedKeys,
              encodedValues: comp.tq.encodedValues,
              sourceOffset: comp.tq.offset,
              windowKeys: comp.tq.windowKeys,
              windowValues: comp.tq.windowValues)
    else { return false }
    zaya.writeCCA(conv: comp.convState, prev: comp.prevHS)
    return zaya.offset == comp.tq.offset && zaya.usesTurboQuantKV
}

/// Helper: restore Mamba SSM state into a `MambaCache` layer (or a
/// `CacheList` containing one). Silently no-ops for other cache classes,
/// which can happen if the serialized model layout has drifted from the
/// current runtime.
private func restoreMambaLayer(
    _ comp: TQDiskSerializer.MambaLayerComponents,
    into layer: any KVCache
) {
    func apply(_ mamba: MambaCache) {
        if let state1 = comp.state1 {
            mamba.state = [comp.state0, state1]
        } else {
            // Single-slot layout (LFM2/LFM2.5 short-conv): restore into
            // slot 0 via the subscript so the 2-slot container geometry is
            // preserved and slot 1 stays genuinely empty.
            mamba[0] = comp.state0
        }
        mamba.offset = comp.offset
    }
    if let mamba = layer as? MambaCache {
        apply(mamba)
        return
    }
    if let cacheList = layer as? CacheList {
        for i in 0..<cacheList.count {
            if let mamba = cacheList[i] as? MambaCache {
                apply(mamba)
                return
            }
        }
    }
}
