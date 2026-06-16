import Foundation
import MLX
import MLXLLM
@testable import MLXLMCommon
import MLXNN
import Testing
import XCTest

// MARK: - BatchKVCache Unit Tests

@Suite("BatchKVCache", .serialized)
struct BatchKVCacheTests {

    @Test("update splits and pads correctly for 2 sequences at different offsets")
    func testUpdateSplitPadStack() {
        // Create two KVCacheSimple instances and populate them to different offsets
        let cache0 = KVCacheSimple()
        let cache1 = KVCacheSimple()

        // Populate cache0 with 5 tokens
        for _ in 0 ..< 5 {
            let k = MLXArray.ones([1, 4, 1, 8]) // [B=1, H=4, L=1, D=8]
            let v = MLXArray.ones([1, 4, 1, 8])
            _ = cache0.update(keys: k, values: v)
        }
        // Populate cache1 with 3 tokens
        for _ in 0 ..< 3 {
            let k = MLXArray.ones([1, 4, 1, 8])
            let v = MLXArray.ones([1, 4, 1, 8])
            _ = cache1.update(keys: k, values: v)
        }

        #expect(cache0.offset == 5)
        #expect(cache1.offset == 3)

        // Create BatchKVCache wrapping both
        let batchCache = BatchKVCache(slotCaches: [cache0, cache1])
        #expect(batchCache.batchSize == 2)
        #expect(batchCache.offset == 5) // max(5, 3)

        // Check offsetArray
        let offsets = batchCache.offsetArray
        MLX.eval(offsets)
        #expect(offsets.shape == [2])
        #expect(offsets[0].item(Int32.self) == 5)
        #expect(offsets[1].item(Int32.self) == 3)

        // Now update with batched keys [B=2, H=4, L=1, D=8]
        let batchKeys = MLXArray.ones([2, 4, 1, 8])
        let batchValues = MLXArray.ones([2, 4, 1, 8])
        let (returnedKeys, returnedValues) = batchCache.update(keys: batchKeys, values: batchValues)

        MLX.eval(returnedKeys, returnedValues)

        // After update: cache0 at offset 6, cache1 at offset 4
        // Padded to max = 6
        #expect(returnedKeys.shape == [2, 4, 6, 8])
        #expect(returnedValues.shape == [2, 4, 6, 8])

        // Verify offsets updated
        let newOffsets = batchCache.offsetArray
        MLX.eval(newOffsets)
        #expect(newOffsets[0].item(Int32.self) == 6)
        #expect(newOffsets[1].item(Int32.self) == 4)
        #expect(batchCache.offset == 6)
    }

    @Test("makeMask returns correct per-sequence causal mask")
    func testMakeMask() {
        let cache0 = KVCacheSimple()
        let cache1 = KVCacheSimple()

        // Populate to different offsets
        _ = cache0.update(
            keys: MLXArray.ones([1, 2, 3, 4]),
            values: MLXArray.ones([1, 2, 3, 4]))
        _ = cache1.update(
            keys: MLXArray.ones([1, 2, 1, 4]),
            values: MLXArray.ones([1, 2, 1, 4]))

        #expect(cache0.offset == 3)
        #expect(cache1.offset == 1)

        let batchCache = BatchKVCache(slotCaches: [cache0, cache1])

        // Decode step: n=1
        let mask = batchCache.makeMask(n: 1, windowSize: nil, returnArray: false)
        if case .array(let maskArray) = mask {
            MLX.eval(maskArray)
            // Shape: [B=2, 1, 1, maxTotal=4] where maxTotal = max(3+1, 1+1) = 4
            #expect(maskArray.shape == [2, 1, 1, 4])

            // Seq 0 at offset 3, query at position 3: attends to [0,1,2,3]
            // [T, T, T, T]
            #expect(maskArray[0, 0, 0, 0].item(Bool.self) == true)
            #expect(maskArray[0, 0, 0, 3].item(Bool.self) == true)

            // Seq 1 at offset 1, query at position 1: attends to [0,1], masks [2,3]
            // [T, T, F, F]
            #expect(maskArray[1, 0, 0, 0].item(Bool.self) == true)
            #expect(maskArray[1, 0, 0, 1].item(Bool.self) == true)
            #expect(maskArray[1, 0, 0, 2].item(Bool.self) == false)
            #expect(maskArray[1, 0, 0, 3].item(Bool.self) == false)
        } else {
            Issue.record("Expected .array mask, got \(mask)")
        }
    }

    @Test("single-sequence BatchKVCache is equivalent to direct cache")
    func testSingleSequence() {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 2, 5, 4]),
            values: MLXArray.ones([1, 2, 5, 4]))

        let batchCache = BatchKVCache(slotCaches: [cache])
        #expect(batchCache.batchSize == 1)
        #expect(batchCache.offset == 5)

        let newK = MLXArray.ones([1, 2, 1, 4])
        let newV = MLXArray.ones([1, 2, 1, 4])
        let (rk, rv) = batchCache.update(keys: newK, values: newV)
        MLX.eval(rk, rv)

        // Single sequence: no padding needed
        #expect(rk.shape == [1, 2, 6, 4])
        #expect(rv.shape == [1, 2, 6, 4])
    }
}

@Suite("BatchArraysCache", .serialized)
struct BatchArraysCacheTests {
    @Test("preserves per-slot offsets for one-slot ArraysCache")
    func preservesOneSlotOffsets() {
        let cache0 = ArraysCache(size: 1)
        let cache1 = ArraysCache(size: 1)
        cache0.offset = 5
        cache1.offset = 3
        cache0[0] = MLXArray.ones([1, 2, 4, 4], dtype: .float32)
        cache1[0] = MLXArray.ones([1, 2, 4, 4], dtype: .float32) * 7

        let batch = BatchArraysCache(slotCaches: [cache0, cache1])
        #expect(batch.offset == 5)
        #expect(batch.offsetArray.asArray(Int32.self) == [5, 3])
        #expect(batch[0]?.shape == [2, 2, 4, 4])

        batch[0] = MLXArray.ones([2, 2, 4, 4], dtype: .float32) * 11
        batch.advance(by: 1)
        batch.splitBack()

        #expect(batch.offset == 6)
        #expect(batch.offsetArray.asArray(Int32.self) == [6, 4])
        #expect(cache0.offset == 6)
        #expect(cache1.offset == 4)
        #expect(cache0[0]?.shape == [1, 2, 4, 4])
        #expect(cache1[0]?.shape == [1, 2, 4, 4])
    }
}

// MARK: - Batch Causal Mask Tests

@Suite("BatchCausalMask", .serialized)
struct BatchCausalMaskTests {

    @Test("two sequences at different offsets, decode step")
    func testBasicMask() {
        let mask = createBatchCausalMask(queryLen: 1, offsets: [5, 3])
        MLX.eval(mask)

        // Shape: [2, 1, 1, 6] — maxTotal = max(5+1, 3+1) = 6
        #expect(mask.shape == [2, 1, 1, 6])

        // Seq 0 at offset 5: attends to all 6 positions
        for j in 0 ..< 6 {
            #expect(mask[0, 0, 0, j].item(Bool.self) == true)
        }

        // Seq 1 at offset 3: attends to 0-3, masks 4-5
        for j in 0 ..< 4 {
            #expect(mask[1, 0, 0, j].item(Bool.self) == true)
        }
        #expect(mask[1, 0, 0, 4].item(Bool.self) == false)
        #expect(mask[1, 0, 0, 5].item(Bool.self) == false)
    }

    @Test("sliding window mask")
    func testSlidingWindow() {
        let mask = createBatchCausalMask(queryLen: 1, offsets: [5, 3], windowSize: 3)
        MLX.eval(mask)

        // Seq 0 at offset 5, window 3: attends to positions 3,4,5 only
        #expect(mask[0, 0, 0, 2].item(Bool.self) == false) // outside window
        #expect(mask[0, 0, 0, 3].item(Bool.self) == true)
        #expect(mask[0, 0, 0, 4].item(Bool.self) == true)
        #expect(mask[0, 0, 0, 5].item(Bool.self) == true)

        // Seq 1 at offset 3, window 3: attends to positions 1,2,3
        #expect(mask[1, 0, 0, 0].item(Bool.self) == false) // outside window
        #expect(mask[1, 0, 0, 1].item(Bool.self) == true)
        #expect(mask[1, 0, 0, 2].item(Bool.self) == true)
        #expect(mask[1, 0, 0, 3].item(Bool.self) == true)
    }

    @Test("same offset sequences produce standard causal mask")
    func testSameOffset() {
        let mask = createBatchCausalMask(queryLen: 1, offsets: [4, 4])
        MLX.eval(mask)

        // Both at offset 4: both attend to [0,1,2,3,4]
        #expect(mask.shape == [2, 1, 1, 5])
        for b in 0 ..< 2 {
            for j in 0 ..< 5 {
                #expect(mask[b, 0, 0, j].item(Bool.self) == true)
            }
        }
    }

    @Test("prefill mask with multiple query tokens")
    func testPrefillMask() {
        // Prefill 3 tokens with offset 2 (already have 2 cached)
        let mask = createBatchCausalMask(queryLen: 3, offsets: [2])
        MLX.eval(mask)

        // Shape: [1, 1, 3, 5] — maxTotal = 2 + 3 = 5
        #expect(mask.shape == [1, 1, 3, 5])

        // Query 0 at position 2: attends to [0,1,2]
        #expect(mask[0, 0, 0, 2].item(Bool.self) == true)
        #expect(mask[0, 0, 0, 3].item(Bool.self) == false)

        // Query 2 at position 4: attends to [0,1,2,3,4]
        for j in 0 ..< 5 {
            #expect(mask[0, 0, 2, j].item(Bool.self) == true)
        }
    }
}

// MARK: - BatchKVCache with Rotating Slots (Gemma-4 SWA regression)

/// Regression suite for the broadcast_shapes crash on Gemma-4 / Mistral-4 /
/// MiMoV2Flash / BaichuanM1 and any other sliding-window model under the
/// batch engine.
///
/// Without the effective-key-length fix, `BatchKVCache.makeMask` produced a
/// mask whose last axis was `offset + n` while `RotatingKVCache.update` only
/// returned `maxCacheSize` keys after wrap, so MLX trapped in
/// `broadcast_shapes` on the very first decode step.
///
/// See `Libraries/MLXLMCommon/BatchEngine/GEMMA4-SLIDING-WINDOW-CRASH.md`.
@Suite("BatchKVCache rotating-slot (Gemma-4 SWA regression)", .serialized)
struct BatchKVCacheRotatingSlotTests {

    @Test("makeMask last axis matches update key count after ring wrap")
    func testMaskMatchesUpdatedKeyShape() {
        // Tiny window so we can wrap cheaply in-test. Real Gemma-4 has
        // maxSize 1024 and prompt 2152 — same topology, bigger numbers.
        let maxSize = 16
        let prompt = 40
        let H = 4
        let D = 8
        let rotating = RotatingKVCache(maxSize: maxSize, keep: 0)

        // Prefill: single multi-token update past the ring (matches how
        // non-chunked VLM prepare populates Gemma-4's sliding layers).
        _ = rotating.update(
            keys: MLXArray.ones([1, H, prompt, D]),
            values: MLXArray.ones([1, H, prompt, D]))
        #expect(rotating.offset == prompt)

        // One decode step with n = 1 — this is the step that used to crash.
        let batchCache = BatchKVCache(slotCaches: [rotating])
        let mask = batchCache.makeMask(n: 1, windowSize: maxSize, returnArray: false)

        let newK = MLXArray.ones([1, H, 1, D])
        let newV = MLXArray.ones([1, H, 1, D])
        let (rk, _) = batchCache.update(keys: newK, values: newV)

        // After update, rotating slot has wrapped: keys shape last axis = maxSize.
        #expect(rk.shape == [1, H, maxSize, D])

        // The fix: mask's last axis equals update's key length. Without the
        // fix this was `offset + n == prompt + 1 == 41` and MLX crashed.
        if case .array(let maskArray) = mask {
            #expect(maskArray.shape.last == rk.shape[2])
            #expect(maskArray.shape == [1, 1, 1, maxSize])

            // Post-wrap ring → every stored key is a valid attention target.
            for j in 0 ..< maxSize {
                #expect(maskArray[0, 0, 0, j].item(Bool.self) == true)
            }
        } else {
            Issue.record("Expected .array mask, got \(mask)")
        }
    }

    @Test("pre-wrap rotating slot still gets standard causal mask")
    func testPreWrapMaskUnchanged() {
        // Before wrap: rotating cache behaves like a standard growing cache.
        let maxSize = 32
        let prompt = 8 // well under maxSize
        let H = 2
        let D = 4
        let rotating = RotatingKVCache(maxSize: maxSize, keep: 0)
        _ = rotating.update(
            keys: MLXArray.ones([1, H, prompt, D]),
            values: MLXArray.ones([1, H, prompt, D]))

        let batchCache = BatchKVCache(slotCaches: [rotating])
        let mask = batchCache.makeMask(n: 1, windowSize: maxSize, returnArray: false)

        if case .array(let maskArray) = mask {
            // Pre-wrap: mask last axis = offset + n = 9
            #expect(maskArray.shape == [1, 1, 1, prompt + 1])
            // Query at logical pos 8 attends to [0..8]
            for j in 0 ..< prompt + 1 {
                #expect(maskArray[0, 0, 0, j].item(Bool.self) == true)
            }
        } else {
            Issue.record("Expected .array mask, got \(mask)")
        }
    }

    @Test("mixed batch: wrapped rotating + unbounded slot produces compatible mask")
    func testMixedBatchWrappedAndUnbounded() {
        // NOTE: in the real engine, slots for a given layer always share cache
        // type (one BatchKVCache per layer). This test is a belt-and-braces
        // check that createBatchCausalMask handles the mixed case gracefully
        // — e.g., if a future model puts different layer topologies behind
        // the same BatchKVCache, the mask shouldn't corrupt either slot.
        let maxSize = 16
        let rotating = RotatingKVCache(maxSize: maxSize, keep: 0)
        _ = rotating.update(
            keys: MLXArray.ones([1, 2, 40, 4]),
            values: MLXArray.ones([1, 2, 40, 4])) // wrapped: offset=40, keys=maxSize

        let simple = KVCacheSimple()
        _ = simple.update(
            keys: MLXArray.ones([1, 2, 5, 4]),
            values: MLXArray.ones([1, 2, 5, 4])) // offset=5, keys=5

        let batch = BatchKVCache(slotCaches: [rotating, simple])
        let mask = batch.makeMask(n: 1, windowSize: maxSize, returnArray: false)

        if case .array(let maskArray) = mask {
            // maxTotal = max(16 (capped rotating), 6 (simple)) = 16
            #expect(maskArray.shape == [2, 1, 1, maxSize])

            // Slot 0 (wrapped rotating): all 16 positions valid (ring full).
            for j in 0 ..< maxSize {
                #expect(maskArray[0, 0, 0, j].item(Bool.self) == true)
            }

            // Slot 1 (unbounded): valid through position 5, padded [6..16) = false.
            for j in 0 ..< 6 {
                #expect(maskArray[1, 0, 0, j].item(Bool.self) == true)
            }
            for j in 6 ..< maxSize {
                #expect(maskArray[1, 0, 0, j].item(Bool.self) == false)
            }
        } else {
            Issue.record("Expected .array mask, got \(mask)")
        }
    }

    @Test("explicit effectiveKeyLens parameter caps maxTotal")
    func testCreateBatchCausalMaskWithEffectiveKeyLens() {
        // Call the low-level helper directly — slot A wrapped, slot B unbounded.
        let mask = createBatchCausalMask(
            queryLen: 1,
            offsets: [100, 5],
            effectiveKeyLens: [16, 6], // slot A ring cap = 16, slot B = offset + n
            windowSize: nil)

        // maxTotal = max(16, 6) = 16
        #expect(mask.shape == [2, 1, 1, 16])

        // Slot A (wrapped): all 16 positions valid.
        for j in 0 ..< 16 {
            #expect(mask[0, 0, 0, j].item(Bool.self) == true)
        }

        // Slot B (unbounded, offset=5): valid through position 5, rest padding.
        for j in 0 ..< 6 {
            #expect(mask[1, 0, 0, j].item(Bool.self) == true)
        }
        for j in 6 ..< 16 {
            #expect(mask[1, 0, 0, j].item(Bool.self) == false)
        }
    }
}

// MARK: - BatchEngine Integration Tests (uses small Llama model)

/// Integration tests that create a small Llama model and run BatchEngine
/// with actual generation. Tests correctness, multi-request batching,
/// per-request parameters, and throughput.
class BatchEngineIntegrationTests: XCTestCase {

    /// Create a small test model and batch engine for testing
    private func makeEngine(vocabSize: Int = 100, maxBatchSize: Int = 4) -> BatchEngine {
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

    private func makeSlowPrefillEngine(
        prefillDelayMicroseconds: UInt32 = 800_000,
        maxBatchSize: Int = 2
    ) -> BatchEngine {
        let model = SlowPrefillLanguageModel(prefillDelayMicroseconds: prefillDelayMicroseconds)
        let tokenizer = TestTokenizer(vocabularySize: model.vocabularySize)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "slow-prefill-test"),
            messageGenerator: DefaultMessageGenerator()
        )
        nonisolated(unsafe) let context = ModelContext(
            configuration: processor.configuration,
            model: model,
            processor: processor,
            tokenizer: processor.tokenizer
        )
        return BatchEngine(context: context, maxBatchSize: maxBatchSize)
    }

    private func makeEngineWithCoordinator(
        vocabSize: Int = 200,
        maxBatchSize: Int = 1
    ) -> (BatchEngine, CacheCoordinator) {
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
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: true,
            enableDiskCache: false,
            pagedBlockSize: 4,
            maxCacheBlocks: 256
        ))
        return (
            BatchEngine(
                context: context,
                maxBatchSize: maxBatchSize,
                cacheCoordinator: coordinator),
            coordinator
        )
    }

    /// Regression for long-prefill admission starvation. A large model can spend
    /// seconds in prefill after the first submit. The engine must still leave a
    /// control-plane turn for the immediately following submit to enqueue a
    /// second request; otherwise B=2 rows falsely run as serial B=1.
    func testSequentialSubmitDoesNotWaitForLongPrefill() async throws {
        let engine = makeSlowPrefillEngine()
        let params = GenerateParameters(maxTokens: 2, temperature: 0)

        let (_, firstStream) = await engine.submit(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(8))),
            parameters: params)

        // Give the background scheduler a chance to enter prefill before the
        // next request arrives. Without an admission-coalescing yield, the
        // actor is then monopolized by the slow prefill and this submit call
        // cannot return until the first request has nearly finished.
        await Task.yield()

        let secondSubmitReturned = await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                _ = await engine.submit(
                    input: LMInput(tokens: MLXArray(Int32(10) ..< Int32(17))),
                    parameters: params)
                return true
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: 150_000_000)
                return false
            }

            let first = await group.next() ?? false
            group.cancelAll()
            return first
        }

        _ = await collectTokens(from: firstStream)
        await engine.shutdown()

        XCTAssertTrue(
            secondSubmitReturned,
            "Second submit should enqueue promptly instead of waiting for first long prefill")
    }

    /// Test: single request through BatchEngine produces tokens
    func testSingleRequest() async throws {
        let engine = makeEngine()

        let input = LMInput(tokens: MLXArray(Int32(1) ..< Int32(6)))
        let params = GenerateParameters(maxTokens: 10, temperature: 0)

        var tokenCount = 0
        var gotInfo = false
        let stream = await engine.generate(input: input, parameters: params)
        for await generation in stream {
            switch generation {
            case .chunk(let text):
                XCTAssertFalse(text.isEmpty, "Chunk should not be empty")
                tokenCount += 1
            case .info(let info):
                gotInfo = true
                XCTAssertEqual(info.promptTokenCount, 5)
                XCTAssertGreaterThan(info.generationTokenCount, 0)
                XCTAssertEqual(info.stopReason, .length)
            case .reasoning, .toolCall:
                break
            @unknown default:
                break
            }
        }
        XCTAssert(gotInfo, "Should receive completion info")
        XCTAssertGreaterThan(tokenCount, 0, "Should receive at least one text chunk")
    }

    /// Test: two concurrent requests both complete correctly
    func testTwoConcurrentRequests() async throws {
        let engine = makeEngine()

        let input1 = LMInput(tokens: MLXArray(Int32(1) ..< Int32(4)))
        let input2 = LMInput(tokens: MLXArray(Int32(10) ..< Int32(15)))
        let params = GenerateParameters(maxTokens: 5, temperature: 0)

        // Submit both
        let (id1, stream1) = await engine.submit(input: input1, parameters: params)
        let (id2, stream2) = await engine.submit(input: input2, parameters: params)

        XCTAssertNotEqual(id1, id2, "Request IDs should be unique")

        // Collect tokens — engine runs both concurrently internally
        let result1 = await collectTokens(from: stream1)
        let result2 = await collectTokens(from: stream2)

        XCTAssertGreaterThan(result1.tokens.count, 0, "Request 1 should produce tokens")
        XCTAssertGreaterThan(result2.tokens.count, 0, "Request 2 should produce tokens")
        XCTAssertNotNil(result1.info, "Request 1 should have completion info")
        XCTAssertNotNil(result2.info, "Request 2 should have completion info")
    }

    /// Test: different parameters per request (greedy vs sampled)
    func testDifferentParametersPerRequest() async throws {
        let engine = makeEngine()

        let input = LMInput(tokens: MLXArray(Int32(1) ..< Int32(4)))

        // Request 1: greedy, 3 tokens
        let params1 = GenerateParameters(maxTokens: 3, temperature: 0)
        // Request 2: sampled, 7 tokens
        let params2 = GenerateParameters(maxTokens: 7, temperature: 0.8)

        let (_, stream1) = await engine.submit(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(4))), parameters: params1)
        let (_, stream2) = await engine.submit(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(4))), parameters: params2)

        let r1 = await collectTokens(from: stream1)
        let r2 = await collectTokens(from: stream2)

        // Request 1 should have fewer tokens (maxTokens=3)
        XCTAssertLessThanOrEqual(r1.tokens.count, 3)
        // Request 2 should have more tokens (maxTokens=7)
        XCTAssertGreaterThan(r2.tokens.count, r1.tokens.count)
    }

    /// Test: request cancellation mid-generation
    func testCancellation() async throws {
        let engine = makeEngine()

        let input = LMInput(tokens: MLXArray(Int32(1) ..< Int32(4)))
        let params = GenerateParameters(maxTokens: 100, temperature: 0)

        let (requestID, stream) = await engine.submit(input: input, parameters: params)

        // Cancel after receiving a couple tokens
        var tokenCount = 0
        for await event in stream {
            if case .token = event {
                tokenCount += 1
                if tokenCount >= 2 {
                    await engine.cancel(requestID)
                    break
                }
            }
        }
        XCTAssertGreaterThanOrEqual(tokenCount, 2)
    }

    /// Test: more requests than maxBatchSize — queuing works
    func testQueueOverflow() async throws {
        let engine = makeEngine(maxBatchSize: 2)

        let tokens = MLXArray(Int32(1) ..< Int32(4))
        let params = GenerateParameters(maxTokens: 3, temperature: 0)

        // Submit 4 requests with maxBatchSize=2 — 2 active, 2 queued
        var streams = [AsyncStream<BatchGeneration>]()
        for _ in 0 ..< 4 {
            let (_, stream) = await engine.submit(
                input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(4))), parameters: params)
            streams.append(stream)
        }

        // All 4 should complete
        var completedCount = 0
        for stream in streams {
            let result = await collectTokens(from: stream)
            if result.info != nil {
                completedCount += 1
            }
        }
        XCTAssertEqual(completedCount, 4, "All 4 requests should complete")
    }

    /// Test: runtime maxBatchSize updates immediately admit queued work.
    func testUpdateMaxBatchSizeAdmitsQueuedRequests() async throws {
        let engine = makeEngine(maxBatchSize: 1)
        let initialMaxBatchSize = await engine.maxBatchSize
        XCTAssertEqual(initialMaxBatchSize, 1)

        let params = GenerateParameters(maxTokens: 25, temperature: 0)
        var streams = [AsyncStream<BatchGeneration>]()
        for _ in 0 ..< 3 {
            let (_, stream) = await engine.submit(
                input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(4))),
                parameters: params)
            streams.append(stream)
        }

        try await engine.updateMaxBatchSize(3)

        let resizedMaxBatchSize = await engine.maxBatchSize
        let pendingCount = await engine.pendingCount
        let activeCount = await engine.activeCount
        XCTAssertEqual(resizedMaxBatchSize, 3)
        XCTAssertEqual(pendingCount, 0)
        XCTAssertLessThanOrEqual(activeCount, 3)

        for stream in streams {
            let result = await collectTokens(from: stream)
            XCTAssertNotNil(result.info, "Resized engine should finish every stream")
        }
    }

    /// Test: invalid runtime maxBatchSize updates fail without mutating state.
    func testUpdateMaxBatchSizeRejectsInvalidValue() async throws {
        let engine = makeEngine(maxBatchSize: 1)

        do {
            try await engine.updateMaxBatchSize(0)
            XCTFail("Expected invalidMaxBatchSize")
        } catch BatchEngineConfigurationError.invalidMaxBatchSize(let value) {
            XCTAssertEqual(value, 0)
        } catch {
            XCTFail("Unexpected error: \(error)")
        }

        let currentMaxBatchSize = await engine.maxBatchSize
        XCTAssertEqual(currentMaxBatchSize, 1)
    }

    /// Test: invalid construction maxBatchSize cannot abort the process.
    func testConstructionNormalizesInvalidMaxBatchSize() async throws {
        let engine = makeEngine(maxBatchSize: 0)

        let currentMaxBatchSize = await engine.maxBatchSize
        XCTAssertEqual(currentMaxBatchSize, 1)
    }

    /// Test: runtime maxBatchSize updates fail once shutdown owns the engine.
    func testUpdateMaxBatchSizeAfterShutdownFailsClosed() async throws {
        let engine = makeEngine(maxBatchSize: 1)

        await engine.shutdown()

        do {
            try await engine.updateMaxBatchSize(2)
            XCTFail("Expected engineShutdown")
        } catch BatchEngineConfigurationError.engineShutdown {
            // Expected: a stale engine handle cannot be made configurable again.
        } catch {
            XCTFail("Unexpected error: \(error)")
        }

        let currentMaxBatchSize = await engine.maxBatchSize
        let accepting = await engine.isAcceptingRequests
        let shutdown = await engine.isShutdown
        XCTAssertEqual(currentMaxBatchSize, 1)
        XCTAssertFalse(accepting)
        XCTAssertTrue(shutdown)
    }

    /// Test: batch throughput vs serial — measures actual tok/s
    func testBatchThroughput() async throws {
        let maxTokens = 20
        let numRequests = 4

        let tokens = MLXArray(Int32(1) ..< Int32(6))
        let params = GenerateParameters(maxTokens: maxTokens, temperature: 0)

        // Measure B=1: 4 requests one at a time through engine with maxBatchSize=1
        let serialEngine = makeEngine(vocabSize: 200, maxBatchSize: 1)
        let serialStart = Date()
        for _ in 0 ..< numRequests {
            let (_, stream) = await serialEngine.submit(
                input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(6))), parameters: params)
            _ = await collectTokens(from: stream)
        }
        let serialTime = Date().timeIntervalSince(serialStart)

        // Measure B=4: 4 requests simultaneously through engine with maxBatchSize=4
        let batchEngine = makeEngine(vocabSize: 200, maxBatchSize: numRequests)
        let batchStart = Date()
        var batchStreams = [AsyncStream<BatchGeneration>]()
        for _ in 0 ..< numRequests {
            let (_, stream) = await batchEngine.submit(
                input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(6))), parameters: params)
            batchStreams.append(stream)
        }
        // Wait for all to complete
        for stream in batchStreams {
            _ = await collectTokens(from: stream)
        }
        let batchTime = Date().timeIntervalSince(batchStart)

        let serialTokPerSec = Double(numRequests * maxTokens) / serialTime
        let batchTokPerSec = Double(numRequests * maxTokens) / batchTime

        print("""
        === Throughput Benchmark ===
        Serial: \(String(format: "%.1f", serialTokPerSec)) total tok/s (\(String(format: "%.2f", serialTime))s)
        Batch:  \(String(format: "%.1f", batchTokPerSec)) total tok/s (\(String(format: "%.2f", batchTime))s)
        Speedup: \(String(format: "%.2f", batchTokPerSec / serialTokPerSec))x
        """)

        // Batch should be at least as fast as serial (on a tiny model the overhead may dominate,
        // but on real models batch should be faster)
        // We don't assert speedup > 1 because the tiny test model may not show benefit
    }

    /// Test: shutdown cleans up all pending and active requests
    func testShutdown() async throws {
        let engine = makeEngine(maxBatchSize: 2)

        let tokens = MLXArray(Int32(1) ..< Int32(4))
        let params = GenerateParameters(maxTokens: 1000, temperature: 0)

        // Submit requests
        let (_, stream1) = await engine.submit(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(4))), parameters: params)
        let (_, stream2) = await engine.submit(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(4))), parameters: params)

        // Let them start
        try await Task.sleep(nanoseconds: 100_000_000)  // 100ms

        // Shutdown
        await engine.shutdown()

        // Both streams should finish (with .cancelled or naturally)
        let r1 = await collectTokens(from: stream1)
        let r2 = await collectTokens(from: stream2)

        // At least one should have info (cancelled or completed)
        XCTAssert(r1.info != nil || r2.info != nil, "Shutdown should finish streams")
    }

    /// Test: shutdown closes the high-level generate() stream without hanging.
    func testShutdownDuringGenerateStreamFinishesTextPath() async throws {
        let engine = makeEngine(maxBatchSize: 1)
        let stream = await engine.generate(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(5))),
            parameters: GenerateParameters(maxTokens: 1000, temperature: 0)
        )

        try await Task.sleep(nanoseconds: 50_000_000)
        await engine.shutdown()

        let result = await collectGenerations(from: stream)
        XCTAssertNotNil(result.info, "generate() should emit completion info on shutdown")
        XCTAssertEqual(result.info?.stopReason, .cancelled)

        let pendingCount = await engine.pendingCount
        let activeCount = await engine.activeCount
        let isRunning = await engine.isRunning
        XCTAssertEqual(pendingCount, 0)
        XCTAssertEqual(activeCount, 0)
        XCTAssertFalse(isRunning)
    }

    /// Regression for the Osaurus MiniMax speed path: the high-level
    /// `generate(...)` API should use the direct TokenIterator loop when the
    /// engine is configured as B=1 and no other work is queued. That keeps the
    /// app's default single-request path aligned with the proven CLI path while
    /// leaving `submit(...)` and maxBatchSize > 1 on the batching scheduler.
    func testGenerateUsesSoloFastPathWhenEngineIsIdleAndBOne() async throws {
        let engine = makeSlowPrefillEngine(
            prefillDelayMicroseconds: 20_000,
            maxBatchSize: 1)

        let stream = await engine.generate(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(5))),
            parameters: GenerateParameters(maxTokens: 20, temperature: 0)
        )

        let soloActive = await engine.isSoloFastPathActiveForTesting
        let activeCount = await engine.activeCount
        XCTAssertTrue(soloActive, "Idle B=1 generate() should use the direct solo fast path")
        XCTAssertEqual(activeCount, 1, "solo fast path should count as active work")

        await engine.shutdown()
        let result = await collectGenerations(from: stream)
        XCTAssertEqual(result.info?.stopReason, .cancelled)
    }

    func testGenerateSoloFastPathCompletesWithoutQueuedWork() async throws {
        let engine = makeSlowPrefillEngine(
            prefillDelayMicroseconds: 20_000,
            maxBatchSize: 1)

        let stream = await engine.generate(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(5))),
            parameters: GenerateParameters(maxTokens: 2, temperature: 0)
        )

        let soloActive = await engine.isSoloFastPathActiveForTesting
        XCTAssertTrue(soloActive)

        let result = await collectGenerations(from: stream)
        XCTAssertEqual(
            result.info?.stopReason, .length,
            "info=\(String(describing: result.info)), chunks=\(result.chunks.count)")
        let finalSoloActive = await engine.isSoloFastPathActiveForTesting
        XCTAssertFalse(finalSoloActive)
    }

    /// If a raw `submit(...)` arrives while the direct B=1 `generate(...)`
    /// path owns the model, it must queue without starting the actor decode
    /// loop. Starting both would create two concurrent MLX paths over the same
    /// model/cache objects.
    func testSubmitQueuesBehindActiveSoloFastPath() async throws {
        let engine = makeSlowPrefillEngine(
            prefillDelayMicroseconds: 20_000,
            maxBatchSize: 1)

        let stream = await engine.generate(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(5))),
            parameters: GenerateParameters(maxTokens: 20, temperature: 0)
        )
        let soloActiveBeforeSubmit = await engine.isSoloFastPathActiveForTesting
        XCTAssertTrue(soloActiveBeforeSubmit)

        let generatedTask = Task { @Sendable [stream] in
            var chunks = [String]()
            var info: GenerateCompletionInfo?
            for await event in stream {
                switch event {
                case .chunk(let text):
                    chunks.append(text)
                case .info(let i):
                    info = i
                case .reasoning, .toolCall:
                    break
                @unknown default:
                    break
                }
            }
            return (chunks: chunks, info: info)
        }

        let (_, queuedStream) = await engine.submit(
            input: LMInput(tokens: MLXArray(Int32(8) ..< Int32(12))),
            parameters: GenerateParameters(maxTokens: 2, temperature: 0)
        )

        let pendingCount = await engine.pendingCount
        XCTAssertEqual(pendingCount, 1)
        let running = await engine.isRunning
        let soloStillActive = await engine.isSoloFastPathActiveForTesting
        let schedulerOnlyRunning = running && !soloStillActive
        XCTAssertFalse(schedulerOnlyRunning,
            "batch scheduler should not run concurrently with the solo fast path")

        let generated = await generatedTask.value
        let queued = await collectTokens(from: queuedStream)
        XCTAssertEqual(
            generated.info?.stopReason, .length,
            "info=\(String(describing: generated.info)), chunks=\(generated.chunks.count)")
        XCTAssertEqual(queued.info?.stopReason, .length)
        let finalPendingCount = await engine.pendingCount
        let finalSoloActive = await engine.isSoloFastPathActiveForTesting
        XCTAssertEqual(finalPendingCount, 0)
        XCTAssertFalse(finalSoloActive)
    }

    /// Regression coverage for the app Stop-button symptom: if the client
    /// stops reading the high-level B=1 stream, the direct TokenIterator task
    /// must cancel, release solo ownership, and let queued engine work run.
    func testGenerateSoloFastPathConsumerCancellationReleasesQueuedSubmit() async throws {
        let engine = makeSlowPrefillEngine(
            prefillDelayMicroseconds: 20_000,
            maxBatchSize: 1)

        let stream = await engine.generate(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(5))),
            parameters: GenerateParameters(maxTokens: 1_000, temperature: 0)
        )
        let soloActiveBeforeSubmit = await engine.isSoloFastPathActiveForTesting
        XCTAssertTrue(soloActiveBeforeSubmit)

        let (_, queuedStream) = await engine.submit(
            input: LMInput(tokens: MLXArray(Int32(9) ..< Int32(13))),
            parameters: GenerateParameters(maxTokens: 2, temperature: 0)
        )
        let pendingAfterSubmit = await engine.pendingCount
        XCTAssertEqual(pendingAfterSubmit, 1)

        let sawEvent = await Task { @Sendable [stream] in
            for await _ in stream {
                return true
            }
            return false
        }.value
        XCTAssertTrue(sawEvent)

        let queued = await collectTokens(from: queuedStream)
        XCTAssertEqual(queued.info?.stopReason, .length)
        let finalPendingCount = await engine.pendingCount
        let finalSoloActive = await engine.isSoloFastPathActiveForTesting
        let finalRunning = await engine.isRunning
        XCTAssertEqual(finalPendingCount, 0)
        XCTAssertFalse(finalSoloActive)
        XCTAssertFalse(finalRunning)
    }

    /// Runtime resizing must not wake the batch scheduler while a B=1 direct
    /// generate owns the model. The queued request can start only after the
    /// solo task drains or cancels.
    func testUpdateMaxBatchSizeDoesNotWakeSchedulerDuringSoloFastPath() async throws {
        let engine = makeSlowPrefillEngine(
            prefillDelayMicroseconds: 20_000,
            maxBatchSize: 1)

        let stream = await engine.generate(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(5))),
            parameters: GenerateParameters(maxTokens: 1_000, temperature: 0)
        )
        let soloActiveBeforeSubmit = await engine.isSoloFastPathActiveForTesting
        XCTAssertTrue(soloActiveBeforeSubmit)

        let (_, queuedStream) = await engine.submit(
            input: LMInput(tokens: MLXArray(Int32(9) ..< Int32(13))),
            parameters: GenerateParameters(maxTokens: 2, temperature: 0)
        )
        let pendingAfterSubmit = await engine.pendingCount
        XCTAssertEqual(pendingAfterSubmit, 1)

        try await engine.updateMaxBatchSize(2)

        let soloActiveAfterResize = await engine.isSoloFastPathActiveForTesting
        let pendingAfterResize = await engine.pendingCount
        let activeAfterResize = await engine.activeCount
        XCTAssertTrue(soloActiveAfterResize)
        XCTAssertEqual(pendingAfterResize, 1)
        XCTAssertEqual(activeAfterResize, 1)

        let generated = await collectGenerations(from: stream)
        XCTAssertEqual(generated.info?.stopReason, .length)

        let queued = await collectTokens(from: queuedStream)
        XCTAssertEqual(queued.info?.stopReason, .length)
        let finalSoloActive = await engine.isSoloFastPathActiveForTesting
        let finalPendingCount = await engine.pendingCount
        XCTAssertFalse(finalSoloActive)
        XCTAssertEqual(finalPendingCount, 0)
    }

    /// The single-request fast path must still participate in prompt-cache
    /// storage. This covers the app default B=1 path, not the lower-level
    /// `submit(...)` scheduler path covered by compile/coordinator tests.
    func testGenerateSoloFastPathStoresIntoCacheCoordinator() async throws {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        let (engine, coordinator) = makeEngineWithCoordinator(maxBatchSize: 1)
        let promptTokens: [Int32] = [3, 7, 11, 13, 17, 19, 23, 29]
        let params = GenerateParameters(maxTokens: 3, temperature: 0)
        let promptSalt = computeCacheSalt(
            for: LMInput(tokens: MLXArray(promptTokens)),
            parameters: params)
        let stream = await engine.generate(
            input: LMInput(tokens: MLXArray(promptTokens)),
            parameters: params
        )

        let result = await collectGenerations(from: stream)
        XCTAssertEqual(result.info?.stopReason, .length)

        let expectedTokens = promptTokens.map { Int($0) }
        switch coordinator.fetch(tokens: expectedTokens, mediaSalt: promptSalt) {
        case .hit(let matchedTokens, let remainingTokens, let detail, _, _, _):
            XCTAssertEqual(matchedTokens, expectedTokens.count)
            XCTAssertTrue(remainingTokens.isEmpty)
            XCTAssertEqual(detail.rawValue, CacheDetail.paged.rawValue)
        case .miss:
            XCTFail("B=1 generate fast path should populate the cache coordinator")
        }
    }

    /// Test: shutdown is terminal, so stale engine handles cannot restart GPU work.
    func testSubmitAfterShutdownRejectsWithoutRestartingEngine() async throws {
        let engine = makeEngine(maxBatchSize: 2)

        await engine.shutdown()

        let accepting = await engine.isAcceptingRequests
        let shutdown = await engine.isShutdown
        XCTAssertFalse(accepting)
        XCTAssertTrue(shutdown)

        let (_, stream) = await engine.submit(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(4))),
            parameters: GenerateParameters(maxTokens: 10, temperature: 0)
        )
        let result = await collectTokens(from: stream)

        XCTAssertTrue(result.tokens.isEmpty)
        XCTAssertEqual(result.info?.promptTokenCount, 3)
        XCTAssertEqual(result.info?.generationTokenCount, 0)
        XCTAssertEqual(result.info?.stopReason, .cancelled)
        let pendingCount = await engine.pendingCount
        let activeCount = await engine.activeCount
        let isRunning = await engine.isRunning
        XCTAssertEqual(pendingCount, 0)
        XCTAssertEqual(activeCount, 0)
        XCTAssertFalse(isRunning)
    }

    /// Test: high-level text generate() also rejects stale post-shutdown handles.
    func testGenerateAfterShutdownRejectsWithoutRestartingEngine() async throws {
        let engine = makeEngine(maxBatchSize: 2)

        await engine.shutdown()

        let stream = await engine.generate(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(5))),
            parameters: GenerateParameters(maxTokens: 10, temperature: 0)
        )
        let result = await collectGenerations(from: stream)

        XCTAssertTrue(result.chunks.isEmpty)
        XCTAssertEqual(result.info?.promptTokenCount, 4)
        XCTAssertEqual(result.info?.generationTokenCount, 0)
        XCTAssertEqual(result.info?.stopReason, .cancelled)
        let pendingCount = await engine.pendingCount
        let activeCount = await engine.activeCount
        let isRunning = await engine.isRunning
        XCTAssertEqual(pendingCount, 0)
        XCTAssertEqual(activeCount, 0)
        XCTAssertFalse(isRunning)
    }

    // MARK: - Helpers

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

    private func collectGenerations(from stream: AsyncStream<Generation>)
        async -> (chunks: [String], info: GenerateCompletionInfo?)
    {
        var chunks = [String]()
        var info: GenerateCompletionInfo?
        for await event in stream {
            switch event {
            case .chunk(let text):
                chunks.append(text)
            case .info(let i):
                info = i
            case .reasoning, .toolCall:
                break
            @unknown default:
                break
            }
        }
        return (chunks, info)
    }
}

private final class SlowPrefillLanguageModel: Module, LanguageModel,
    KVCacheDimensionProvider, @unchecked Sendable
{
    let prefillDelayMicroseconds: UInt32
    let vocabularySize = 32
    var kvHeads: [Int] { [1] }

    init(prefillDelayMicroseconds: UInt32) {
        self.prefillDelayMicroseconds = prefillDelayMicroseconds
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        usleep(prefillDelayMicroseconds)
        return .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let batch = inputs.shape.first ?? 1
        let length = inputs.shape.count > 1 ? inputs.shape[1] : inputs.size
        return MLXArray.zeros([batch, length, vocabularySize], dtype: .float32)
    }
}
