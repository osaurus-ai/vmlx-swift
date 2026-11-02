// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Metal caps the NUMBER of live MLX buffers, not just their bytes: DSV4's
// cumulative compressor/indexer pools retain ~one buffer per layer per decode
// token, and long generations died at the 499000-resource allocator wall as a
// mid-stream failure (Python vMLX R21-13). These tests pin both halves of the
// Swift port: the projection maths, and the requirement that a cache topology
// whose retention has not been measured is never capped on a limit it provably
// cannot reach.

import Foundation
import MLX
import MLXLLM
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite("MetalLiveBufferGuard projection maths")
struct MetalLiveBufferGuardProjectionTests {

    @Test("DSV4 cap lands below the measured ceiling")
    func dsv4CapLandsBelowMeasuredCeiling() {
        // Python measurement: 12000 generated tokens succeeded; 14336 died.
        // The cap must sit under the failure point without being punitive.
        let cap = MetalLiveBufferGuard.projectedOutputTokenCap(
            resourceLimit: 499_000,
            liveBaselineBuffers: 16384,
            buffersPerToken: 43.0)
        #expect(cap != nil)
        #expect(cap! >= 8000 && cap! < 12000)
    }

    @Test("zero retention returns nil so the guard stays inert")
    func zeroRetentionIsInert() {
        #expect(
            MetalLiveBufferGuard.projectedOutputTokenCap(
                resourceLimit: 499_000,
                liveBaselineBuffers: 16384,
                buffersPerToken: 0.0) == nil)
    }

    @Test("unknown resource limit returns nil")
    func unknownResourceLimitIsInert() {
        #expect(
            MetalLiveBufferGuard.projectedOutputTokenCap(
                resourceLimit: 0,
                liveBaselineBuffers: 0,
                buffersPerToken: 43.0) == nil)
    }

    @Test("cap scales inversely with retention rate")
    func capScalesInverselyWithRetention() {
        let few = MetalLiveBufferGuard.projectedOutputTokenCap(
            resourceLimit: 499_000, liveBaselineBuffers: 0, buffersPerToken: 16.0)!
        let many = MetalLiveBufferGuard.projectedOutputTokenCap(
            resourceLimit: 499_000, liveBaselineBuffers: 0, buffersPerToken: 64.0)!
        #expect(few > many)
    }

    @Test("baseline is deducted from the budget")
    func baselineIsDeducted() {
        let noBaseline = MetalLiveBufferGuard.projectedOutputTokenCap(
            resourceLimit: 499_000, liveBaselineBuffers: 0, buffersPerToken: 43.0)!
        let withBaseline = MetalLiveBufferGuard.projectedOutputTokenCap(
            resourceLimit: 499_000, liveBaselineBuffers: 100_000, buffersPerToken: 43.0)!
        #expect(withBaseline < noBaseline)
    }

    @Test("baseline larger than budget yields zero, not negative")
    func oversizedBaselineYieldsZero() {
        #expect(
            MetalLiveBufferGuard.projectedOutputTokenCap(
                resourceLimit: 499_000,
                liveBaselineBuffers: 10_000_000,
                buffersPerToken: 43.0) == 0)
    }

    @Test("out-of-range safety fraction falls back to the default")
    func outOfRangeFractionFallsBack() {
        let expected = MetalLiveBufferGuard.projectedOutputTokenCap(
            resourceLimit: 499_000, liveBaselineBuffers: 0, buffersPerToken: 43.0)
        for bad in [0.0, -1.0, 1.5] {
            #expect(
                MetalLiveBufferGuard.projectedOutputTokenCap(
                    resourceLimit: 499_000, liveBaselineBuffers: 0,
                    buffersPerToken: 43.0, safetyFraction: bad) == expected)
        }
    }

    @Test("safety fraction reduces the cap")
    func safetyFractionReducesCap() {
        let strict = MetalLiveBufferGuard.projectedOutputTokenCap(
            resourceLimit: 499_000, liveBaselineBuffers: 0,
            buffersPerToken: 43.0, safetyFraction: 0.5)!
        let loose = MetalLiveBufferGuard.projectedOutputTokenCap(
            resourceLimit: 499_000, liveBaselineBuffers: 0,
            buffersPerToken: 43.0, safetyFraction: 1.0)!
        #expect(strict < loose)
    }
}

@Suite("MetalLiveBufferGuard topology detection and advisory compatibility")
struct MetalLiveBufferGuardClampTests {

    @Test("conventional KV topologies retain nothing and pass through untouched")
    func conventionalTopologyIsUntouched() {
        // Measured: Qwen3.6-27B held its live-buffer count flat across 600
        // decode steps. A conventional cache must never be capped.
        let cache: [KVCache] = (0 ..< 64).map { _ in KVCacheSimple() }
        #expect(MetalLiveBufferGuard.retainedBuffersPerDecodeToken(cache: cache) == 0.0)
        #expect(
            MetalLiveBufferGuard.clampedMaxTokens(
                requested: 1_000_000, cache: cache, environment: [:]) == 1_000_000)
        #expect(
            MetalLiveBufferGuard.clampedMaxTokens(
                requested: nil, cache: cache, environment: [:]) == nil)
    }

    @Test("DeepseekV4Cache reports one retained buffer per layer per token")
    func dsv4CacheReportsRetention() {
        let layer: any KVCache = DeepseekV4Cache(slidingWindow: 128, compressRatio: 4)
        #expect(
            (layer as? MetalBufferRetainingCache)?.retainedBuffersPerDecodeToken == 1.0)
    }

    @Test("a projected DSV4 buffer budget never replaces requested output limits")
    func dsv4TopologyPreservesOutputLimits() {
        let cache: [KVCache] = (0 ..< 43).map { _ in
            DeepseekV4Cache(slidingWindow: 128, compressRatio: 4)
        }
        let unlimited = MetalLiveBufferGuard.clampedMaxTokens(
            requested: nil, cache: cache, environment: [:])
        #expect(unlimited == nil)

        let oversized = MetalLiveBufferGuard.clampedMaxTokens(
            requested: 1_000_000, cache: cache, environment: [:])
        #expect(oversized == 1_000_000)

        let modest = MetalLiveBufferGuard.clampedMaxTokens(
            requested: 256, cache: cache, environment: [:])
        #expect(modest == 256)
    }

    @Test("disabling buffer diagnostics preserves an omitted output limit")
    func envKillSwitchDisables() {
        let cache: [KVCache] = (0 ..< 43).map { _ in
            DeepseekV4Cache(slidingWindow: 128, compressRatio: 4)
        }
        #expect(
            MetalLiveBufferGuard.clampedMaxTokens(
                requested: nil, cache: cache,
                environment: ["VMLX_METAL_BUFFER_COUNT_GUARD": "0"]) == nil)
    }
}

@Suite("Output limits survive both generation admission paths", .serialized)
struct MetalLiveBufferOutputAdmissionTests {
    private func retainingCache(contiguous: Bool = false) -> [KVCache] {
        (0 ..< 38).map { _ in
            DeepseekV4Cache(
                slidingWindow: 16, compressRatio: 1,
                poolQuantizationEnabled: true, poolBF16HotByteLimit: 0,
                poolQuantizationBits: contiguous ? 4 : 8,
                contiguousPools: contiguous)
        }
    }

    @Test("38-layer projection reproduces the Studio's hidden 11387-token stop")
    func historicalProjectionMatchesFailure() {
        #expect(
            MetalLiveBufferGuard.projectedOutputTokenCap(
                resourceLimit: 499_000, liveBaselineBuffers: 16_384,
                buffersPerToken: 38) == 11_387)
    }

    @Test("V4.1 contiguous slabs do not declare per-token buffer retention")
    func contiguousTopologyIsNotHistoricalRetention() {
        Device.withDefaultDevice(.cpu) {
            #expect(
                MetalLiveBufferGuard.retainedBuffersPerDecodeToken(
                    cache: retainingCache(contiguous: true)) == 0)
        }
    }

    @Test(
        "BatchEngine slots retain explicit, omitted, and short output budgets",
        arguments: [Int?.none, 1_000_000, 8])
    func batchSlotPreservesRequestedLimit(requested: Int?) {
        Device.withDefaultDevice(.cpu) {
            let stream = AsyncStream<BatchGeneration>.makeStream()
            defer { stream.continuation.finish() }
            let request = BatchPendingRequest(
                input: LMInput(tokens: MLXArray([Int32(0)])),
                parameters: GenerateParameters(maxTokens: requested, temperature: 0),
                continuation: stream.continuation)
            let slot = BatchSlot(from: request, cache: retainingCache(), stopTokenIDs: [])
            #expect(slot.maxTokens == requested)
        }
    }

    @Test("solo decode crosses 11387 while still respecting an explicit short budget")
    func soloDecodeCrossesHistoricalLimit() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = OutputLimitAdmissionModel()
            var iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray([Int32(0)])), model: model,
                cache: retainingCache(),
                parameters: GenerateParameters(maxTokens: 1_000_000, temperature: 0))
            #expect(iterator.maxTokens == 1_000_000)
            let start = Date()
            var count = 0
            for _ in 0 ..< 12_001 {
                guard iterator.next() != nil else { break }
                count += 1
            }
            #expect(count == 12_001)
            #expect(iterator.generationFailure == nil)
            print(
                "OUTPUT_LIMIT_SYNTHETIC tokens=\(count) tokens_per_second=\(Double(count) / Date().timeIntervalSince(start)) model=constant-logits CPU=true"
            )
            let short = try TokenIterator(
                input: LMInput(tokens: MLXArray([Int32(0)])), model: model,
                cache: retainingCache(),
                parameters: GenerateParameters(maxTokens: 8, temperature: 0))
            #expect(Array(short).count == 8)
            let omitted = try TokenIterator(
                input: LMInput(tokens: MLXArray([Int32(0)])), model: model,
                cache: retainingCache(),
                parameters: GenerateParameters(maxTokens: nil, temperature: 0))
            #expect(omitted.maxTokens == nil)
        }
    }
}

private final class OutputLimitAdmissionModel: Module, LanguageModel, KVCacheDimensionProvider {
    var kvHeads: [Int] { [1] }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([inputs.dim(0), inputs.dim(1), 2], dtype: .float32)
    }
}
