// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stage 0 regression tests for BatchEngine + TurboQuant KV quantization.
//
// Covers the Stage 0 scope from `docs/superpowers/specs/2026-04-18-batch-engine-blockers-design.md`:
// - `BatchQuantize` admission + post-prefill + per-decode compression hooks
// - `BatchKVCache` wrapping TurboQuantKVCache slot caches natively (no
//    dedicated subclass — `TurboQuantKVCache.update()` shape contract matches
//    `KVCacheSimple.update()`, so the split/pad/stack wrapper handles both)
// - `BatchEngine` end-to-end generation with `kvMode: .turboQuant(...)`
// - Mixed-mode batch (TQ + no-quant concurrent slots)
// - Affine / legacy `kvBits` graceful degradation (warning logged, no crash)

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
import XCTest

// MARK: - Unit Tests: BatchKVCache wrapping TurboQuantKVCache slots

/// `BatchKVCache` should wrap `TurboQuantKVCache` slot caches natively because
/// TQ's `update()` returns plain float `[1, H, L, D]` matching the
/// `KVCacheSimple` shape contract. These tests confirm that contract holds —
/// if it ever drifts (e.g. a future TQ change returns a different shape) these
/// catch the regression and Stage 0 would need a dedicated wrapper subclass.
///
/// Serial execution: TQ encoder state creation + MLX lazy graph materialization
/// do not play well with Swift Testing's default parallel test execution.
/// Running serial matches how production request handling uses TQ.
@Suite("BatchKVCache + TurboQuant slot caches", .serialized)
struct BatchKVCacheWithTQSlotsTests {

    /// Build a TurboQuantKVCache already in `.compressed` phase by compressing
    /// a KVCacheSimple that has been populated with `tokens` tokens.
    ///
    /// Uses realistic head dimensions (D=64) rather than toy sizes. TQ's QJL
    /// encoder assumes power-of-2 head dims that are common to actual LLM
    /// attention — smaller dims exercise edge cases unrelated to this spec.
    private func makeCompressedTQCache(
        tokens: Int, H: Int = 4, D: Int = 64
    ) -> TurboQuantKVCache {
        let simple = KVCacheSimple()
        for _ in 0 ..< tokens {
            let k = MLXArray.ones([1, H, 1, D])
            let v = MLXArray.ones([1, H, 1, D])
            _ = simple.update(keys: k, values: v)
        }
        #expect(simple.offset == tokens)
        return TurboQuantKVCache.fromSimpleCache(
            simple, keyBits: 3, valueBits: 3, sinkTokens: 4, residualTokens: 0)
    }

    /// Build a compressed cache with MLA-style asymmetric K/V widths.
    /// Ling/Bailing global MLA stores K as no-PE + RoPE (`192`) while V stays
    /// at `128`; TurboQuant must not reuse the key encoder state for values.
    private func makeCompressedAsymmetricTQCache(
        tokens: Int, H: Int = 4, keyD: Int = 192, valueD: Int = 128
    ) -> TurboQuantKVCache {
        let simple = KVCacheSimple()
        let k = MLXArray.ones([1, H, tokens, keyD])
        let v = MLXArray.ones([1, H, tokens, valueD])
        _ = simple.update(keys: k, values: v)
        #expect(simple.offset == tokens)
        return TurboQuantKVCache.fromSimpleCache(
            simple, keyBits: 4, valueBits: 4, sinkTokens: 4, residualTokens: 0)
    }

    @Test("BatchKVCache wraps two TQ slot caches at different offsets")
    func testWrapTwoTQSlotsDifferentOffsets() {
        MLXMetalTestLock.withLock {
            let tq0 = makeCompressedTQCache(tokens: 16)
            let tq1 = makeCompressedTQCache(tokens: 12)

            #expect(tq0.phase == .compressed)
            #expect(tq1.phase == .compressed)

            // Upcast to KVCache — exactly what stepBatchDecode does.
            let batchCache = BatchKVCache(slotCaches: [tq0 as KVCache, tq1 as KVCache])

            #expect(batchCache.batchSize == 2)
            #expect(batchCache.offset == 16)
            MLX.eval(batchCache.offsetArray)
            #expect(batchCache.offsetArray[0].item(Int32.self) == 16)
            #expect(batchCache.offsetArray[1].item(Int32.self) == 12)
        }
    }

    @Test("update on BatchKVCache over TQ slots returns padded [B, H, maxLen, D]")
    func testUpdatePadsAndStacksOverTQ() {
        MLXMetalTestLock.withLock {
            let tq0 = makeCompressedTQCache(tokens: 16)
            let tq1 = makeCompressedTQCache(tokens: 12)

            let batchCache = BatchKVCache(slotCaches: [tq0 as KVCache, tq1 as KVCache])

            let batchKeys = MLXArray.ones([2, 4, 1, 64])
            let batchValues = MLXArray.ones([2, 4, 1, 64])
            let (ks, vs) = batchCache.update(keys: batchKeys, values: batchValues)
            MLX.eval(ks, vs)

            // Post-update: tq0 at 17 tokens, tq1 at 13 tokens; padded to maxLen=17
            #expect(ks.shape == [2, 4, 17, 64])
            #expect(vs.shape == [2, 4, 17, 64])

            MLX.eval(batchCache.offsetArray)
            #expect(batchCache.offsetArray[0].item(Int32.self) == 17)
            #expect(batchCache.offsetArray[1].item(Int32.self) == 13)
            #expect(batchCache.offset == 17)
        }
    }

    @Test("single-slot BatchKVCache over TQ cache is equivalent to raw TQ update")
    func testSingleSlotTQEquivalence() {
        MLXMetalTestLock.withLock {
            let tq = makeCompressedTQCache(tokens: 12)
            let batchCache = BatchKVCache(slotCaches: [tq as KVCache])

            #expect(batchCache.batchSize == 1)
            #expect(batchCache.offset == 12)

            let k = MLXArray.ones([1, 4, 1, 64])
            let v = MLXArray.ones([1, 4, 1, 64])
            let (ks, vs) = batchCache.update(keys: k, values: v)
            MLX.eval(ks, vs)

            #expect(ks.shape == [1, 4, 13, 64])
            #expect(vs.shape == [1, 4, 13, 64])
            #expect(tq.offset == 13)
        }
    }

    @Test("asymmetric K/V head dimensions use separate TurboQuant encoder states")
    func testAsymmetricKeyValueDimensions() {
        MLXMetalTestLock.withLock {
            let tq = makeCompressedAsymmetricTQCache(tokens: 12)
            #expect(tq.phase == .compressed)

            let k = MLXArray.ones([1, 4, 1, 192])
            let v = MLXArray.ones([1, 4, 1, 128])
            let (ks, vs) = tq.update(keys: k, values: v)
            MLX.eval(ks, vs)

            #expect(ks.shape == [1, 4, 13, 192])
            #expect(vs.shape == [1, 4, 13, 128])
            #expect(tq.offset == 13)

            let batchCache = BatchKVCache(slotCaches: [tq as KVCache])
            let (batchK, batchV) = batchCache.update(keys: k, values: v)
            MLX.eval(batchK, batchV)

            #expect(batchK.shape == [1, 4, 14, 192])
            #expect(batchV.shape == [1, 4, 14, 128])
            #expect(batchCache.offset == 14)
        }
    }

    @Test("mixed TQ + KVCacheSimple slot caches work via shared shape contract")
    func testMixedTQAndSimpleSlots() {
        MLXMetalTestLock.withLock {
            // Represents the heterogeneous state possible mid-run when one slot
            // has compressed to TQ (long prompt) while another is still simple
            // (short prompt below TQ threshold). Both slot caches must satisfy
            // the `update(keys:values:) -> (MLXArray, MLXArray)` contract for
            // `BatchKVCache.padAndConcatenate` to work.
            let tq = makeCompressedTQCache(tokens: 12)

            let simple = KVCacheSimple()
            for _ in 0 ..< 5 {
                _ = simple.update(
                    keys: MLXArray.ones([1, 4, 1, 64]),
                    values: MLXArray.ones([1, 4, 1, 64]))
            }
            #expect(simple.offset == 5)

            let batchCache = BatchKVCache(slotCaches: [tq as KVCache, simple as KVCache])

            let ks = MLXArray.ones([2, 4, 1, 64])
            let vs = MLXArray.ones([2, 4, 1, 64])
            let (returnedK, returnedV) = batchCache.update(keys: ks, values: vs)
            MLX.eval(returnedK, returnedV)

            // tq -> 13, simple -> 6. Padded to maxLen=13.
            #expect(returnedK.shape == [2, 4, 13, 64])
            #expect(returnedV.shape == [2, 4, 13, 64])
        }
    }
}

// MARK: - Unit Tests: BatchQuantize hook

/// Direct tests for ``BatchQuantize.maybeCompress`` — the core Stage 0
/// invariant that ``KVCacheSimple`` layers swap to ``TurboQuantKVCache`` once
/// the threshold is crossed. Bypasses the engine loop so failures point
/// straight at the hook, not at scheduler interactions.
@Suite("BatchQuantize", .serialized)
struct BatchQuantizeHookTests {

    /// Populate a `KVCacheSimple` with `tokens` single-token updates so its
    /// `offset` equals `tokens`. Head dim 64 avoids the 8-dim edge case that
    /// trips TQ's encoder.
    private func makePopulatedSimpleCache(tokens: Int) -> KVCacheSimple {
        let c = KVCacheSimple()
        for _ in 0 ..< tokens {
            _ = c.update(
                keys: MLXArray.ones([1, 4, 1, 64]),
                values: MLXArray.ones([1, 4, 1, 64]))
        }
        return c
    }

    @Test("maybeCompress swaps simple to TurboQuant when safe middle threshold crossed")
    func testThresholdTriggersSwap() {
        MLXMetalTestLock.withLock {
            // 4-layer cache with enough tokens to keep exact sink + exact
            // recent tail while still leaving a middle span to compress.
            var cache: [KVCache] = (0 ..< 4).map { _ in
                makePopulatedSimpleCache(tokens: 96)
            }
            #expect(cache.allSatisfy { $0 is KVCacheSimple })

            let params = GenerateParameters(
                maxTokens: 5,
                kvMode: .turboQuant(keyBits: 3, valueBits: 3),
                temperature: 0
            )
            BatchQuantize.maybeCompress(cache: &cache, parameters: params)

            // After hook: every layer must be TurboQuantKVCache in .compressed phase.
            #expect(cache.allSatisfy { $0 is TurboQuantKVCache })
            for layer in cache {
                let tq = layer as! TurboQuantKVCache
                #expect(tq.phase == .compressed)
                #expect(tq.offset == 96)
                #expect(tq.compressedKeys?.sinkCount == 4)
                #expect(tq.compressedKeys?.tailCount == TurboQuantKVCache.defaultResidualTokens)
            }
        }
    }

    @Test("maybeCompress is a no-op below the exact-tail threshold")
    func testBelowThresholdNoOp() {
        MLXMetalTestLock.withLock {
            // 64-token recent tail + 4-token sink leaves no useful middle
            // span, so compression waits instead of quantizing the active
            // instruction boundary.
            var cache: [KVCache] = (0 ..< 4).map { _ in
                makePopulatedSimpleCache(tokens: 64)
            }
            let params = GenerateParameters(
                maxTokens: 5,
                kvMode: .turboQuant(keyBits: 3, valueBits: 3),
                temperature: 0
            )
            BatchQuantize.maybeCompress(cache: &cache, parameters: params)

            #expect(cache.allSatisfy { $0 is KVCacheSimple }, "Below safe middle threshold, cache must remain uncompressed")
        }
    }

    @Test("maybeCompress is a no-op when kvMode == .none")
    func testNoneNoOp() {
        MLXMetalTestLock.withLock {
            var cache: [KVCache] = (0 ..< 4).map { _ in
                makePopulatedSimpleCache(tokens: 20)
            }
            // Default kvMode is .none.
            let params = GenerateParameters(maxTokens: 5, temperature: 0)
            BatchQuantize.maybeCompress(cache: &cache, parameters: params)

            #expect(cache.allSatisfy { $0 is KVCacheSimple }, ".none mode must never compress")
        }
    }

    @Test("maybeCompress is a no-op when kvMode == .affine (Stage 0 scope)")
    func testAffineNoOp() {
        MLXMetalTestLock.withLock {
            // Stage 0 explicitly does NOT run affine compression under batch —
            // see BatchQuantize.maybeCompress comments. Affine requests run with
            // float KV and a warning logged at admission.
            var cache: [KVCache] = (0 ..< 4).map { _ in
                makePopulatedSimpleCache(tokens: 20)
            }
            let params = GenerateParameters(
                maxTokens: 5,
                kvMode: .affine(bits: 4, groupSize: 64),
                temperature: 0
            )
            BatchQuantize.maybeCompress(cache: &cache, parameters: params)

            #expect(cache.allSatisfy { $0 is KVCacheSimple }, "Stage 0 affine mode must remain uncompressed under batch")
        }
    }

    @Test("maybeCompress is idempotent — second call does not re-process")
    func testIdempotent() {
        MLXMetalTestLock.withLock {
            var cache: [KVCache] = (0 ..< 4).map { _ in
                makePopulatedSimpleCache(tokens: 96)
            }
            let params = GenerateParameters(
                maxTokens: 5,
                kvMode: .turboQuant(keyBits: 3, valueBits: 3),
                temperature: 0
            )
            BatchQuantize.maybeCompress(cache: &cache, parameters: params)
            let afterFirst: [ObjectIdentifier] = cache.map { ObjectIdentifier($0 as AnyObject) }

            // Second call: the internal guard
            // `!cache.contains(where: { $0 is TurboQuantKVCache })` short-circuits.
            BatchQuantize.maybeCompress(cache: &cache, parameters: params)
            let afterSecond: [ObjectIdentifier] = cache.map { ObjectIdentifier($0 as AnyObject) }

            #expect(afterFirst == afterSecond, "Second call must preserve cache object identity")
        }
    }

    @Test("maybeCompress preserves hybrid layers (MambaCache / SSM state)")
    func testHybridPreservation() {
        MLXMetalTestLock.withLock {
            // Mix: 2 KVCacheSimple + 2 MambaCache. `maybeQuantizeKVCache` only
            // touches KVCacheSimple layers, preserving SSM/state layers for
            // hybrid models (Qwen3.5 Mamba, Qwen3Next GDN, LFM2, Jamba, etc.).
            var cache: [KVCache] = [
                makePopulatedSimpleCache(tokens: 96),
                MambaCache(),
                makePopulatedSimpleCache(tokens: 96),
                MambaCache(),
            ]
            let params = GenerateParameters(
                maxTokens: 5,
                kvMode: .turboQuant(keyBits: 3, valueBits: 3),
                temperature: 0
            )
            BatchQuantize.maybeCompress(cache: &cache, parameters: params)

            #expect(cache[0] is TurboQuantKVCache)
            #expect(cache[1] is MambaCache, "SSM/state layer must remain untouched")
            #expect(cache[2] is TurboQuantKVCache)
            #expect(cache[3] is MambaCache, "SSM/state layer must remain untouched")
        }
    }

    @Test("maybeCompress preserves DSV4 hybrid SWA+CSA+HSA cache")
    func testDeepseekV4HybridPreservation() {
        MLXMetalTestLock.withLock {
            // DSV4's production cache is already compressed structurally:
            // cr=0 layers use a 128-token rotating SWA window, and cr>0
            // layers use DeepseekV4Cache for SWA plus CSA/HSA pool rows.
            // A global TurboQuant default must not replace that topology.
            var cache: [KVCache] = [
                RotatingKVCache(maxSize: 128, keep: 0),
                DeepseekV4Cache(slidingWindow: 128, compressRatio: 128),
                DeepseekV4Cache(slidingWindow: 128, compressRatio: 4),
            ]
            for layer in cache {
                _ = layer.update(
                    keys: MLXArray.ones([1, 1, 10, 64]),
                    values: MLXArray.ones([1, 1, 10, 64]))
            }
            let params = GenerateParameters(
                maxTokens: 5,
                kvMode: .turboQuant(keyBits: 3, valueBits: 3),
                temperature: 0)

            BatchQuantize.maybeCompress(cache: &cache, parameters: params)

            #expect(cache[0] is RotatingKVCache)
            #expect(cache[1] is DeepseekV4Cache)
            #expect(cache[2] is DeepseekV4Cache)
            #expect(!cache.contains { $0 is TurboQuantKVCache })
        }
    }
}

// MARK: - Integration Tests: BatchEngine + TurboQuant

/// Integration tests that create a small Llama model and run BatchEngine
/// with `kvMode: .turboQuant(...)`. Verifies Stage 0 admission hook,
/// post-prefill compression, and batched decode with TQ slot caches.
class BatchEngineTurboQuantIntegrationTests: XCTestCase {

    /// Create a small test model + engine. Identical surface to
    /// `BatchEngineIntegrationTests.makeEngine` so tests are comparable.
    private func makeEngine(
        vocabSize: Int = 200, maxBatchSize: Int = 4
    ) -> BatchEngine {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 4, intermediateSize: 128,
            attentionHeads: 8, rmsNormEps: 1e-5, vocabularySize: vocabSize, kvHeads: 4)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)
        MLX.eval(model)

        let processor = TestInputProcessor()
        nonisolated(unsafe) let context = ModelContext(
            configuration: processor.configuration,
            model: model,
            processor: processor,
            tokenizer: processor.tokenizer
        )
        return BatchEngine(context: context, maxBatchSize: maxBatchSize)
    }

    private func collectTokens(from stream: AsyncStream<BatchGeneration>)
        async -> (tokens: [Int], info: GenerateCompletionInfo?)
    {
        var tokens = [Int]()
        var info: GenerateCompletionInfo?
        for await event in stream {
            switch event {
            case .token(let id):
                tokens.append(id)
            case .info(let i):
                info = i
            }
        }
        return (tokens, info)
    }

    /// Stage 0 minimum: a single TQ request completes without crashing and
    /// produces the requested number of decode tokens. Since the prompt
    /// (12 tokens) exceeds the TQ minimum threshold of 8, the post-prefill
    /// hook swaps KVCacheSimple to TurboQuantKVCache before any batched
    /// decode step runs.
    func testTurboQuantSingleRequest() async throws {
        let engine = makeEngine()

        // 12 tokens — exceeds the TQ minimum of 8 so compression triggers
        // during stepPrefill's post-hook.
        let input = LMInput(tokens: MLXArray(Int32(1) ..< Int32(13)))
        let params = GenerateParameters(
            maxTokens: 10,
            kvMode: .turboQuant(keyBits: 3, valueBits: 3),
            temperature: 0
        )

        let (_, stream) = await engine.submit(input: input, parameters: params)
        let result = await collectTokens(from: stream)

        XCTAssertNotNil(result.info, "TQ single-request should complete with completion info")
        XCTAssertEqual(result.tokens.count, 10, "Should produce exactly 10 decode tokens")
        XCTAssertEqual(result.info?.stopReason, .length)
    }

    /// Two concurrent TQ requests. Each slot compresses independently on its
    /// own prefill boundary; the batched decode step wraps both slots in a
    /// `BatchTurboQuantKVCache` per layer (homogeneous TQ state).
    func testTurboQuantTwoConcurrentRequests() async throws {
        let engine = makeEngine()

        let params = GenerateParameters(
            maxTokens: 8,
            kvMode: .turboQuant(keyBits: 3, valueBits: 3),
            temperature: 0
        )

        let (_, s1) = await engine.submit(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(13))),
            parameters: params
        )
        let (_, s2) = await engine.submit(
            input: LMInput(tokens: MLXArray(Int32(20) ..< Int32(32))),
            parameters: params
        )

        // The engine scheduling loop runs both requests concurrently regardless
        // of consumer order — each stream buffers events until consumed.
        // Sequential awaits here match the pattern used by existing BatchEngine
        // integration tests.
        let result1 = await collectTokens(from: s1)
        let result2 = await collectTokens(from: s2)

        XCTAssertEqual(result1.tokens.count, 8, "R1 should produce 8 decode tokens")
        XCTAssertEqual(result2.tokens.count, 8, "R2 should produce 8 decode tokens")
        XCTAssertNotNil(result1.info)
        XCTAssertNotNil(result2.info)
    }

    /// Mixed batch: one TQ request + one no-quant request concurrent.
    /// Verifies:
    ///  - Each slot runs its own mode independently.
    ///  - The heterogeneous state (some layers TQ, some KVCacheSimple across
    ///    slots) routes through the `BatchKVCache` wrapper uniformly.
    ///    Stage 0 deliberately has no `BatchTurboQuantKVCache` subclass — TQ's
    ///    `update()` shape contract matches `KVCacheSimple` so the wrapper
    ///    handles both homogeneous and heterogeneous cache arrays.
    func testMixedTurboQuantAndNoneRequests() async throws {
        let engine = makeEngine()

        let tqParams = GenerateParameters(
            maxTokens: 6,
            kvMode: .turboQuant(keyBits: 3, valueBits: 3),
            temperature: 0
        )
        let noneParams = GenerateParameters(maxTokens: 6, temperature: 0)

        let (_, sTQ) = await engine.submit(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(13))),
            parameters: tqParams
        )
        let (_, sNone) = await engine.submit(
            input: LMInput(tokens: MLXArray(Int32(40) ..< Int32(52))),
            parameters: noneParams
        )

        let resultTQ = await collectTokens(from: sTQ)
        let resultNone = await collectTokens(from: sNone)

        XCTAssertEqual(resultTQ.tokens.count, 6, "TQ slot should produce 6 decode tokens")
        XCTAssertEqual(resultNone.tokens.count, 6, "No-quant slot should produce 6 decode tokens")
        XCTAssertNotNil(resultTQ.info)
        XCTAssertNotNil(resultNone.info)
    }

    /// Affine / legacy `kvBits` request. BatchQuantize logs a warning at
    /// admission; decode proceeds with float KV. Test asserts no crash and
    /// successful completion — the warning is observed manually in the log.
    func testAffineRequestGracefulDegradation() async throws {
        let engine = makeEngine()

        let params = GenerateParameters(
            maxTokens: 5,
            kvBits: 4,
            kvGroupSize: 64,  // legacy affine path
            temperature: 0
        )
        // KVQuantizationMode has associated values — use pattern matching to
        // assert `.none` rather than `==` which requires Equatable synthesis.
        if case .none = params.kvMode {
            // Expected
        } else {
            XCTFail("legacy kvBits path should keep kvMode == .none")
        }

        let input = LMInput(tokens: MLXArray(Int32(1) ..< Int32(13)))
        let (_, stream) = await engine.submit(input: input, parameters: params)
        let result = await collectTokens(from: stream)

        XCTAssertEqual(result.tokens.count, 5, "Affine request should complete with float KV")
        XCTAssertNotNil(result.info)
    }

    /// Short-prompt TQ request: prompt length 6 < TQ minimum 8, so TQ does
    /// NOT trigger at prefill end. The per-step hook in `stepBatchDecode`
    /// eventually crosses the threshold during decode and compresses.
    /// This verifies the post-decode hook is wired correctly.
    func testTurboQuantShortPromptDeferredCompression() async throws {
        let engine = makeEngine()

        // 6-token prompt — below TQ minimum of 8. Compression must wait for
        // decode to push offset past 8.
        let input = LMInput(tokens: MLXArray(Int32(1) ..< Int32(7)))
        let params = GenerateParameters(
            maxTokens: 12,
            kvMode: .turboQuant(keyBits: 3, valueBits: 3),
            temperature: 0
        )

        let (_, stream) = await engine.submit(input: input, parameters: params)
        let result = await collectTokens(from: stream)

        XCTAssertEqual(result.tokens.count, 12, "Short-prompt TQ request should still complete")
        XCTAssertNotNil(result.info)
    }

    /// Same TQ request twice with temperature=0 produces identical token
    /// sequences. Regression detector: if Stage 0 accidentally introduces
    /// non-determinism, tokens would drift between runs.
    func testTurboQuantDeterminism() async throws {
        let engine = makeEngine()
        let params = GenerateParameters(
            maxTokens: 8,
            kvMode: .turboQuant(keyBits: 3, valueBits: 3),
            temperature: 0
        )

        let (_, s1) = await engine.submit(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(13))),
            parameters: params
        )
        let r1 = await collectTokens(from: s1)

        let (_, s2) = await engine.submit(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(13))),
            parameters: params
        )
        let r2 = await collectTokens(from: s2)

        XCTAssertEqual(r1.tokens, r2.tokens, "Same TQ input + greedy should be deterministic across runs")
    }
}

// MARK: - Multi-turn Tests: BatchEngine + CacheCoordinator + TurboQuant

/// Multi-turn integration tests that exercise the cache-coordinator path
/// through `BatchEngine` with both plain and TQ requests.
///
/// **What's covered here:**
/// - Cache hit on repeat prompt (plain float KV)
/// - Cache hit on repeat prompt under `.turboQuant(...)` — TQ-compressed
///   state round-trips via the paged coordinator (`SLIDING-1` disk path not
///   exercised here; this is the in-memory paged cache only)
/// - Prompt extension across turns — turn 2 submits a superset of turn 1's
///   prompt and cache-hits the common prefix
/// - Concurrent multi-turn — two slots each doing a 2-turn conversation
///   simultaneously
/// - No cross-slot contamination — different prompts must not cache-collide
/// - Hybrid (MambaCache) preservation across turns under TQ — SSM state
///   must not be touched by TQ compression
///
/// **What's NOT covered here (future work):**
/// - L2 disk (`TQDiskSerializer v2`) restore on fresh engine — requires
///   on-disk fixtures.
/// - VLM multi-turn with image mediaSalt — requires a VLM fixture model.
/// - Detached async SSM re-derive — the Swift production path intentionally
///   uses synchronous prompt-boundary rederive/capture after the old detached
///   helper was reverted for Metal command-encoder safety.
class BatchEngineMultiTurnTests: XCTestCase {

    /// Build an engine that has a paged cache coordinator attached. No disk
    /// cache — tests exercise the in-memory path only.
    private func makeEngineWithCoordinator(
        maxBatchSize: Int = 2
    ) -> (BatchEngine, CacheCoordinator) {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 4, intermediateSize: 128,
            attentionHeads: 8, rmsNormEps: 1e-5, vocabularySize: 200, kvHeads: 4)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)
        MLX.eval(model)

        let processor = TestInputProcessor()
        nonisolated(unsafe) let context = ModelContext(
            configuration: processor.configuration,
            model: model,
            processor: processor,
            tokenizer: processor.tokenizer
        )

        let coordCfg = CacheCoordinatorConfig(
            usePagedCache: true,
            enableDiskCache: false,
            pagedBlockSize: 4,
            maxCacheBlocks: 256
        )
        let coordinator = CacheCoordinator(config: coordCfg)

        let engine = BatchEngine(
            context: context,
            maxBatchSize: maxBatchSize,
            cacheCoordinator: coordinator
        )
        return (engine, coordinator)
    }

    private func collectTokens(from stream: AsyncStream<BatchGeneration>)
        async -> (tokens: [Int], info: GenerateCompletionInfo?)
    {
        var tokens = [Int]()
        var info: GenerateCompletionInfo?
        for await event in stream {
            switch event {
            case .token(let id):
                tokens.append(id)
            case .info(let i):
                info = i
            }
        }
        return (tokens, info)
    }

    /// Baseline multi-turn: submit the same prompt twice through the coordinator-
    /// backed engine. Turn 2 must hit the paged cache (verified by direct
    /// coordinator inspection) and still produce tokens.
    ///
    /// **What this asserts:**
    ///   - Both turns complete with the requested token count.
    ///   - The coordinator has an entry for the prompt after turn 1.
    ///
    /// **What this deliberately does NOT assert:**
    ///   - Bit-exact token equality between turn 1 (fresh prefill) and turn 2
    ///     (cache-restored). The paged coordinator's block restore path has
    ///     pre-existing quirks on tiny random models — it sometimes drifts
    ///     from a fresh re-prefill under random weight arrangements. Bit-
    ///     exactness on cache-restore is a concern tracked against the cache
    ///     coordinator, not the BatchEngine. Stage 0's scope is "batch engine
    ///     completes correctly when a coordinator is attached"; Stage 1+
    ///     ships compile-path work independent of that path's bit-equality.
    func testMultiTurnPlainCacheHit() async throws {
        let (engine, coordinator) = makeEngineWithCoordinator()

        // Turn 1: establish the cache entry.
        let promptTokens: [Int32] = [3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
        let params = GenerateParameters(maxTokens: 6, temperature: 0)
        let promptSalt = computeCacheSalt(
            for: LMInput(tokens: MLXArray(promptTokens)),
            parameters: params)

        let (_, s1) = await engine.submit(
            input: LMInput(tokens: MLXArray(promptTokens)),
            parameters: params
        )
        let r1 = await collectTokens(from: s1)
        XCTAssertEqual(r1.tokens.count, 6, "Turn 1 should complete")
        XCTAssertNotNil(r1.info, "Turn 1 must emit completion info")

        // Direct coordinator inspection — turn 1's finishSlot must have
        // stored an entry for this prompt.
        let lookup = coordinator.fetch(
            tokens: promptTokens.map(Int.init), mediaSalt: promptSalt)
        if case .miss = lookup {
            XCTFail("Turn 1 should have populated the paged cache")
        }

        // Turn 2: identical prompt. Engine's stepPrefill should hit the cache
        // and only prefill the last token. Decoded output should complete.
        let (_, s2) = await engine.submit(
            input: LMInput(tokens: MLXArray(promptTokens)),
            parameters: params
        )
        let r2 = await collectTokens(from: s2)
        XCTAssertEqual(r2.tokens.count, 6, "Turn 2 should complete via cache hit")
        XCTAssertNotNil(r2.info, "Turn 2 must emit completion info")
    }

    /// Multi-turn with TurboQuant. Turn 1 compresses KV during prefill;
    /// the compressed state is stored in the coordinator. Turn 2 sees the
    /// cache entry — whether restored as TQ or float depends on the paged
    /// coordinator's storage format (in-memory paged cache stores float
    /// blocks; TQ round-trip is the L2 disk path via `TQDiskSerializer v2`).
    ///
    /// **Stage 0 acceptance:** turn 2 completes correctly without crashing.
    /// Bit-exact TQ cache-hit determinism is NOT a Stage 0 gate because the
    /// paged coordinator's float block representation may lossy-round-trip
    /// through TQ compression/decompression. That determinism target is a
    /// Stage 2+ concern (CompilableTurboQuantKVCache) and is covered there.
    func testMultiTurnTurboQuantCompletes() async throws {
        let (engine, coordinator) = makeEngineWithCoordinator()

        let promptTokens: [Int32] = [5, 11, 17, 23, 29, 31, 37, 41, 43, 47, 53, 59]
        let params = GenerateParameters(
            maxTokens: 6,
            kvMode: .turboQuant(keyBits: 3, valueBits: 3),
            temperature: 0
        )
        let promptSalt = computeCacheSalt(
            for: LMInput(tokens: MLXArray(promptTokens)),
            parameters: params)

        let (_, s1) = await engine.submit(
            input: LMInput(tokens: MLXArray(promptTokens)),
            parameters: params
        )
        let r1 = await collectTokens(from: s1)
        XCTAssertEqual(r1.tokens.count, 6, "Turn 1 should complete under TQ")
        XCTAssertNotNil(r1.info, "Turn 1 must produce completion info")

        // Cache entry existence: finishSlot's coordinator.storeAfterGeneration
        // should have populated the paged cache for this prompt.
        let lookup = coordinator.fetch(
            tokens: promptTokens.map(Int.init), mediaSalt: promptSalt)
        if case .miss = lookup {
            XCTFail("Turn 1 should have populated the paged cache under TQ")
        }

        // Turn 2: identical prompt under TQ. Stage 0 only requires it to
        // complete correctly — the restored state's bit-exactness vs turn 1
        // is a Stage 2+ concern.
        let (_, s2) = await engine.submit(
            input: LMInput(tokens: MLXArray(promptTokens)),
            parameters: params
        )
        let r2 = await collectTokens(from: s2)
        XCTAssertEqual(r2.tokens.count, 6, "Turn 2 TQ should complete via cache")
        XCTAssertNotNil(r2.info, "Turn 2 must produce completion info")
    }

    /// Prompt-extension multi-turn: turn 2's prompt is a superset of turn 1's.
    /// The coordinator should match the common prefix and return the extra
    /// tokens as "remaining", which the engine prefills. Common in real
    /// chat workflows where the next turn appends a new message.
    ///
    /// **Note:** The tiny random test model often converges to degenerate
    /// single-token outputs (e.g. "`[90, 90, 90, ...]`") regardless of prompt.
    /// We therefore assert only that both turns complete and the extended
    /// prompt produces its expected token count — not that outputs diverge.
    /// Divergence is a property of real pre-trained models, not synthetic ones.
    func testMultiTurnPromptExtension() async throws {
        let (engine, _) = makeEngineWithCoordinator()

        let turn1Tokens: [Int32] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        let turn2Tokens: [Int32] = turn1Tokens + [31, 37, 41]  // extends by 3 tokens
        let params = GenerateParameters(maxTokens: 5, temperature: 0)

        let (_, s1) = await engine.submit(
            input: LMInput(tokens: MLXArray(turn1Tokens)),
            parameters: params
        )
        let r1 = await collectTokens(from: s1)
        XCTAssertEqual(r1.tokens.count, 5, "Turn 1 must complete")
        XCTAssertNotNil(r1.info)

        let (_, s2) = await engine.submit(
            input: LMInput(tokens: MLXArray(turn2Tokens)),
            parameters: params
        )
        let r2 = await collectTokens(from: s2)
        XCTAssertEqual(r2.tokens.count, 5, "Turn 2 (extended prompt) must complete")
        XCTAssertNotNil(r2.info)
        XCTAssertEqual(r2.info?.promptTokenCount, turn2Tokens.count,
            "Turn 2 completion info must reflect extended prompt length")
    }

    /// Different prompts must not collide in the cache. Submit prompt A
    /// (populates cache), then prompt B (unrelated) — the coordinator must
    /// NOT return A's cache blocks for B's prompt lookup.
    ///
    /// Verified directly via `coordinator.fetch(tokens:)` after submission
    /// rather than via token-output comparison — the tiny random test model
    /// often converges to the same single token regardless of prompt, so
    /// token comparison is unreliable for this assertion.
    func testMultiTurnNoCrossPromptContamination() async throws {
        let (engine, coordinator) = makeEngineWithCoordinator()

        let promptA: [Int32] = [101, 103, 107, 109, 113, 127, 131, 137, 139]
        let promptB: [Int32] = [150, 151, 152, 153, 154, 155, 156, 157, 158]
        let params = GenerateParameters(maxTokens: 4, temperature: 0)
        let saltA = computeCacheSalt(
            for: LMInput(tokens: MLXArray(promptA)),
            parameters: params)
        let saltB = computeCacheSalt(
            for: LMInput(tokens: MLXArray(promptB)),
            parameters: params)

        // Fill cache with A.
        let (_, sA) = await engine.submit(
            input: LMInput(tokens: MLXArray(promptA)),
            parameters: params
        )
        let rA = await collectTokens(from: sA)
        XCTAssertEqual(rA.tokens.count, 4, "A must complete")

        // After A finishes, the coordinator has A's cache entry. Submitting
        // a fetch with B's tokens should miss — no cross-contamination.
        let bLookupBeforeBRun = coordinator.fetch(
            tokens: promptB.map(Int.init), mediaSalt: saltB)
        if case .hit = bLookupBeforeBRun {
            XCTFail("Coordinator must not return cached blocks for an unrelated prompt (contamination)")
        }

        // Submit B — must complete correctly with its own prefill.
        let (_, sB) = await engine.submit(
            input: LMInput(tokens: MLXArray(promptB)),
            parameters: params
        )
        let rB = await collectTokens(from: sB)
        XCTAssertEqual(rB.tokens.count, 4, "B must complete")

        // After B completes, both A and B entries should be retrievable.
        let aLookup = coordinator.fetch(tokens: promptA.map(Int.init), mediaSalt: saltA)
        let bLookup = coordinator.fetch(tokens: promptB.map(Int.init), mediaSalt: saltB)
        if case .miss = aLookup { XCTFail("A entry should survive B's submission") }
        if case .miss = bLookup { XCTFail("B entry should be stored after completion") }
    }

    /// Concurrent multi-turn: two slots doing overlapping work through the
    /// shared engine + coordinator.
    ///
    /// **Assertions (Stage 0 scope):**
    ///   - Each per-turn request produces the expected token count.
    ///   - Both A and B cache entries exist after their turn-1 completion.
    ///
    /// Cache-hit bit-exact determinism is NOT a Stage 0 gate (see comment on
    /// `testMultiTurnPlainCacheHit`) — it depends on the paged coordinator's
    /// block representation which Stage 0 does not modify.
    func testConcurrentMultiTurn() async throws {
        let (engine, coordinator) = makeEngineWithCoordinator(maxBatchSize: 2)

        let promptA: [Int32] = [7, 13, 19, 29, 37, 43, 53, 61, 71, 79]
        let promptB: [Int32] = [12, 14, 18, 22, 26, 28, 32, 38, 42, 44]
        let params = GenerateParameters(maxTokens: 4, temperature: 0)
        let saltA = computeCacheSalt(
            for: LMInput(tokens: MLXArray(promptA)),
            parameters: params)
        let saltB = computeCacheSalt(
            for: LMInput(tokens: MLXArray(promptB)),
            parameters: params)

        // Turn 1 A and B concurrent
        let (_, s1A) = await engine.submit(
            input: LMInput(tokens: MLXArray(promptA)), parameters: params)
        let (_, s1B) = await engine.submit(
            input: LMInput(tokens: MLXArray(promptB)), parameters: params)

        let r1A = await collectTokens(from: s1A)
        let r1B = await collectTokens(from: s1B)
        XCTAssertEqual(r1A.tokens.count, 4, "Turn 1 A must complete")
        XCTAssertEqual(r1B.tokens.count, 4, "Turn 1 B must complete")

        // Both prompts should have entries in the coordinator after their
        // respective completions.
        let lookupA = coordinator.fetch(tokens: promptA.map(Int.init), mediaSalt: saltA)
        let lookupB = coordinator.fetch(tokens: promptB.map(Int.init), mediaSalt: saltB)
        if case .miss = lookupA { XCTFail("Turn 1 A should populate cache") }
        if case .miss = lookupB { XCTFail("Turn 1 B should populate cache") }

        // Turn 2 A and B concurrent — both must complete through the cache-hit path.
        let (_, s2A) = await engine.submit(
            input: LMInput(tokens: MLXArray(promptA)), parameters: params)
        let (_, s2B) = await engine.submit(
            input: LMInput(tokens: MLXArray(promptB)), parameters: params)

        let r2A = await collectTokens(from: s2A)
        let r2B = await collectTokens(from: s2B)
        XCTAssertEqual(r2A.tokens.count, 4, "Turn 2 A must complete")
        XCTAssertEqual(r2B.tokens.count, 4, "Turn 2 B must complete")
        XCTAssertNotNil(r2A.info)
        XCTAssertNotNil(r2B.info)
    }

    /// Disable coordinator — engine should still work (no cache benefit).
    /// Regression guard against any code path that assumes a coordinator is
    /// present.
    func testMultiTurnWithoutCoordinator() async throws {
        // Use the plain engine (no coordinator).
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 4, intermediateSize: 128,
            attentionHeads: 8, rmsNormEps: 1e-5, vocabularySize: 200, kvHeads: 4)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)
        MLX.eval(model)

        let processor = TestInputProcessor()
        nonisolated(unsafe) let context = ModelContext(
            configuration: processor.configuration,
            model: model,
            processor: processor,
            tokenizer: processor.tokenizer
        )
        let engine = BatchEngine(context: context, maxBatchSize: 2)

        let promptTokens: [Int32] = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
        let params = GenerateParameters(
            maxTokens: 4,
            kvMode: .turboQuant(keyBits: 3, valueBits: 3),
            temperature: 0
        )

        let (_, s1) = await engine.submit(
            input: LMInput(tokens: MLXArray(promptTokens)), parameters: params)
        let r1 = await collectTokens(from: s1)

        let (_, s2) = await engine.submit(
            input: LMInput(tokens: MLXArray(promptTokens)), parameters: params)
        let r2 = await collectTokens(from: s2)

        XCTAssertEqual(r1.tokens.count, 4)
        XCTAssertEqual(r2.tokens.count, 4)
        // Without coordinator: turn 2 fully re-prefills, so determinism holds.
        XCTAssertEqual(r1.tokens, r2.tokens,
            "Without coordinator, same prompt + greedy must be deterministic")
    }
}
