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
import MLXLLM
import MLXLMCommon
import Testing

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

@Suite("MetalLiveBufferGuard topology detection and clamping")
struct MetalLiveBufferGuardClampTests {

    @Test("conventional KV topologies retain nothing and pass through untouched")
    func conventionalTopologyIsUntouched() {
        // Measured: Qwen3.6-27B held its live-buffer count flat across 600
        // decode steps. A conventional cache must never be capped.
        let cache: [KVCache] = (0..<64).map { _ in KVCacheSimple() }
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

    @Test("a DSV4-shaped topology clamps unlimited and oversized requests")
    func dsv4TopologyClamps() {
        let cache: [KVCache] = (0..<43).map { _ in
            DeepseekV4Cache(slidingWindow: 128, compressRatio: 4)
        }
        let unlimited = MetalLiveBufferGuard.clampedMaxTokens(
            requested: nil, cache: cache, environment: [:])
        #expect(unlimited != nil)
        #expect(unlimited! >= 8000 && unlimited! < 12000)

        let oversized = MetalLiveBufferGuard.clampedMaxTokens(
            requested: 1_000_000, cache: cache, environment: [:])
        #expect(oversized == unlimited)

        let modest = MetalLiveBufferGuard.clampedMaxTokens(
            requested: 256, cache: cache, environment: [:])
        #expect(modest == 256)
    }

    @Test("VMLX_METAL_BUFFER_COUNT_GUARD=0 disables the clamp")
    func envKillSwitchDisables() {
        let cache: [KVCache] = (0..<43).map { _ in
            DeepseekV4Cache(slidingWindow: 128, compressRatio: 4)
        }
        #expect(
            MetalLiveBufferGuard.clampedMaxTokens(
                requested: nil, cache: cache,
                environment: ["VMLX_METAL_BUFFER_COUNT_GUARD": "0"]) == nil)
    }
}
