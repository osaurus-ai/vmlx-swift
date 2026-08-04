// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

/// LFM2 / LFM2.5 short-conv layers use `MambaCache` (a fixed 2-slot
/// `ArraysCache`) but only ever write slot 0 — `state.count == 1` after
/// prefill (`jang_runtime.cache.conv: "arrays_state_size_1"`). Both cache
/// persistence rails must round-trip that single-slot state exactly:
///
/// - the v2 disk payload (`TQDiskSerializer`) must persist and restore the
///   one conv slot instead of silently skipping the layer, and an entry
///   whose mamba layer has no state must be an atomic required miss rather
///   than a KV-only "hit" that hands the model 22 empty conv windows;
/// - the paged/L1 companion rail (`restoreSSMStates`) must consume ONE
///   array per short-conv layer for a 1-per-layer companion list instead
///   of assuming the full-Mamba 2-slot layout, which cross-wired layer N
///   with layer N+1's state and left the tail layers empty.
@Suite("LFM2 short-conv cache restore", .serialized)
struct LFM2ShortConvCacheRestoreFocusedTests {

    /// Build an LFM2-style short-conv cache: 2-slot MambaCache with only
    /// slot 0 occupied (the conv window), never slot 1.
    private func shortConvCache(fill: Float, offset: Int) -> MambaCache {
        let mamba = MambaCache()
        mamba[0] = MLXArray.full([1, 2, 8], values: MLXArray(fill))
        mamba.offset = offset
        return mamba
    }

    private func prefilledKV(tokens: Int) -> KVCacheSimple {
        let kv = KVCacheSimple()
        let keys = MLXArray.full([1, 2, tokens, 4], values: MLXArray(Float(0.5)))
        let values = MLXArray.full([1, 2, tokens, 4], values: MLXArray(Float(0.25)))
        _ = kv.update(keys: keys, values: values)
        return kv
    }

    @Test("v2 disk payload round-trips a single-slot short-conv layer")
    func diskRoundTripSingleSlotConv() {
        FocusedMLXTestSupport.withLock {
            let source: [any KVCache] = [
                prefilledKV(tokens: 6),
                shortConvCache(fill: 3.5, offset: 6),
            ]
            let arrays = TQDiskSerializer.serialize(cache: source)

            var target: [any KVCache] = [KVCacheSimple(), MambaCache()]
            let restored = restoreFromDiskArrays(arrays, into: &target)

            #expect(restored == 6)
            guard let mamba = target[1] as? MambaCache else {
                Issue.record("restore target lost its MambaCache layer")
                return
            }
            #expect(mamba.offset == 6)
            #expect(mamba.state.count == 1)
            if let conv = mamba[0] {
                #expect(conv.shape == [1, 2, 8])
                #expect(abs(conv.mean().item(Float.self) - 3.5) < 1e-6)
            } else {
                Issue.record("conv slot 0 was not restored from the v2 payload")
            }
            // Slot 1 is genuinely absent for short-conv layers and must
            // stay empty rather than aliasing another layer's tensor.
            #expect(mamba[1] == nil)
        }
    }

    @Test("v2 disk payload still round-trips the full two-slot Mamba layout")
    func diskRoundTripTwoSlotMamba() {
        FocusedMLXTestSupport.withLock {
            let full = MambaCache()
            full[0] = MLXArray.full([1, 2, 8], values: MLXArray(Float(1.5)))
            full[1] = MLXArray.full([1, 4, 4], values: MLXArray(Float(2.5)))
            full.offset = 9
            let source: [any KVCache] = [prefilledKV(tokens: 9), full]
            let arrays = TQDiskSerializer.serialize(cache: source)

            var target: [any KVCache] = [KVCacheSimple(), MambaCache()]
            let restored = restoreFromDiskArrays(arrays, into: &target)

            #expect(restored == 9)
            guard let mamba = target[1] as? MambaCache else {
                Issue.record("restore target lost its MambaCache layer")
                return
            }
            #expect(mamba.offset == 9)
            #expect(mamba.state.count == 2)
            if mamba.state.count == 2 {
                #expect(abs(mamba.state[0].mean().item(Float.self) - 1.5) < 1e-6)
                #expect(abs(mamba.state[1].mean().item(Float.self) - 2.5) < 1e-6)
            }
        }
    }

    @Test("mamba layer with no persisted state is an atomic required miss")
    func statelessMambaLayerIsRequiredMiss() {
        FocusedMLXTestSupport.withLock {
            // Reproduce a pre-fix stored entry: the serializer stamped the
            // .mamba kind tag but persisted no state for the layer. Restoring
            // such an entry as a KV-only hit hands the model empty conv
            // windows for the restored prefix — it must be a full miss.
            let source: [any KVCache] = [
                prefilledKV(tokens: 6),
                shortConvCache(fill: 3.5, offset: 6),
            ]
            var arrays = TQDiskSerializer.serialize(cache: source)
            for key in Array(arrays.keys)
            where key.contains("mamba_1_state") || key.contains("__mamba_1_offset__") {
                arrays.removeValue(forKey: key)
            }

            var target: [any KVCache] = [KVCacheSimple(), MambaCache()]
            let restored = restoreFromDiskArrays(arrays, into: &target)

            #expect(restored == 0)
            if let mamba = target[1] as? MambaCache {
                #expect(mamba.state.isEmpty)
            }
        }
    }

    @Test("companion restore consumes one array per short-conv layer")
    func companionRestoreSingleSlotAlignment() {
        FocusedMLXTestSupport.withLock {
            // Two LFM2-style layers, each contributing exactly one conv
            // state to the companion list (extractSSMStates order).
            let sourceA = shortConvCache(fill: 10, offset: 5)
            let sourceB = shortConvCache(fill: 20, offset: 5)
            let states = extractSSMStates(from: [sourceA, sourceB])
            #expect(states.count == 2)

            let freshA = MambaCache()
            let freshB = MambaCache()
            restoreSSMStates(states, into: [freshA, freshB], boundary: 5)

            #expect(freshA.offset == 5)
            #expect(freshB.offset == 5)
            if let a = freshA[0], let b = freshB[0] {
                #expect(abs(a.mean().item(Float.self) - 10) < 1e-6)
                #expect(abs(b.mean().item(Float.self) - 20) < 1e-6)
            } else {
                Issue.record("companion restore left short-conv slot 0 empty")
            }
            #expect(freshA[1] == nil)
            #expect(freshB[1] == nil)
        }
    }

    @Test("companion restore keeps the full-Mamba two-slot layout")
    func companionRestoreTwoSlotAlignment() {
        FocusedMLXTestSupport.withLock {
            let full = MambaCache()
            full[0] = MLXArray.full([1, 2, 8], values: MLXArray(Float(1)))
            full[1] = MLXArray.full([1, 4, 4], values: MLXArray(Float(2)))
            full.offset = 7
            let second = MambaCache()
            second[0] = MLXArray.full([1, 2, 8], values: MLXArray(Float(3)))
            second[1] = MLXArray.full([1, 4, 4], values: MLXArray(Float(4)))
            second.offset = 7
            let states = extractSSMStates(from: [full, second])
            #expect(states.count == 4)

            let freshA = MambaCache()
            let freshB = MambaCache()
            restoreSSMStates(states, into: [freshA, freshB], boundary: 7)

            #expect(freshA.state.count == 2)
            #expect(freshB.state.count == 2)
            if freshA.state.count == 2, freshB.state.count == 2 {
                #expect(abs(freshA.state[0].mean().item(Float.self) - 1) < 1e-6)
                #expect(abs(freshA.state[1].mean().item(Float.self) - 2) < 1e-6)
                #expect(abs(freshB.state[0].mean().item(Float.self) - 3) < 1e-6)
                #expect(abs(freshB.state[1].mean().item(Float.self) - 4) < 1e-6)
            }
        }
    }

    @Test("companion restore refuses a companion list with unknown arity")
    func companionRestoreRefusesMismatchedTotals() {
        FocusedMLXTestSupport.withLock {
            // 3 arrays for 2 mamba layers matches neither 1-per-layer nor
            // 2-per-layer. Restoring a best-effort prefix would cross-wire
            // layers; the whole list must be refused (caller re-prefills).
            let states = [
                MLXArray.full([1, 2, 8], values: MLXArray(Float(1))),
                MLXArray.full([1, 2, 8], values: MLXArray(Float(2))),
                MLXArray.full([1, 2, 8], values: MLXArray(Float(3))),
            ]
            let freshA = MambaCache()
            let freshB = MambaCache()
            restoreSSMStates(states, into: [freshA, freshB], boundary: 4)

            #expect(freshA.state.isEmpty)
            #expect(freshB.state.isEmpty)
            #expect(freshA.offset == 0)
            #expect(freshB.offset == 0)
        }
    }

    @Test("mixed KV + GatedDeltaNet companion restore is unchanged")
    func companionRestoreArraysCacheUnchanged() {
        FocusedMLXTestSupport.withLock {
            // qwen3.5/ornith-style layer list: attention + ArraysCache with
            // every slot occupied. The slotCount-driven consumption must not
            // regress while the mamba arity logic changes around it.
            let gdn = ArraysCache(size: 2)
            gdn[0] = MLXArray.full([1, 2, 4], values: MLXArray(Float(6)))
            gdn[1] = MLXArray.full([1, 3, 3], values: MLXArray(Float(7)))
            gdn.offset = 11
            let states = extractSSMStates(from: [KVCacheSimple(), gdn])
            #expect(states.count == 2)

            let fresh = ArraysCache(size: 2)
            restoreSSMStates(states, into: [KVCacheSimple(), fresh], boundary: 11)

            #expect(fresh.state.count == 2)
            if fresh.state.count == 2 {
                #expect(abs(fresh.state[0].mean().item(Float.self) - 6) < 1e-6)
                #expect(abs(fresh.state[1].mean().item(Float.self) - 7) < 1e-6)
            }
            #expect(fresh.offset == 11)
        }
    }

    @Test("LFM2 configuration honors intermediate_size without block_ff_dim")
    func lfm2ConfigIntermediateSizeFallback() {
        // Stock LiquidAI / transformers-5 LFM2.5 configs carry only
        // `intermediate_size`; the JANG converter injects `block_ff_dim`.
        // Both spellings must produce the same FFN width.
        let json = """
            {
              "hidden_size": 2048,
              "num_hidden_layers": 2,
              "num_attention_heads": 32,
              "num_key_value_heads": 8,
              "intermediate_size": 10752,
              "vocab_size": 128000,
              "layer_types": ["conv", "full_attention"],
              "conv_L_cache": 3,
              "block_auto_adjust_ff_dim": false,
              "norm_eps": 1e-5,
              "rope_theta": 10000000.0
            }
            """.data(using: .utf8)!
        do {
            let config = try JSONDecoder().decode(LFM2Configuration.self, from: json)
            #expect(config.blockFFDim == 10752)
        } catch {
            Issue.record("LFM2Configuration failed to decode: \(error)")
        }
    }
}
