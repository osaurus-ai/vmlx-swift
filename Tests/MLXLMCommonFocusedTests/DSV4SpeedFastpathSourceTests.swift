// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Repin tripwires for the DSV4 decode fastpaths ported from Python vmlx
// (v1.6.19–v1.6.24). Each pin exists because the speedup is a DEFAULT: a
// refactor that quietly flips the lm_head back to whole-head dequant, drops
// the shared RoPE tables, or unwires the Metal live-buffer clamp would ship
// silently — decode would still be correct, just slow (or, for the guard,
// long generations would die at the allocator wall again). Guards must be
// REACHED, not just correct, so the clamp pins assert the serving call
// sites, not just the helper.

import Foundation
import Testing

@Suite("DSV4 speed fastpath source contracts")
struct DSV4SpeedFastpathSourceTests {

    private func source(_ path: String) throws -> String {
        try String(contentsOfFile: path, encoding: .utf8)
    }

    @Test("lm_head defaults to fused quantized matmul with an exact opt-out")
    func lmHeadDefaultsToFusedQMM() throws {
        let helpers = try source("Libraries/MLXLLM/Models/DeepseekV4MathHelpers.swift")

        // Default mode is the fused kernel; "exact" is the opt-out, never
        // the default. Python vmlx 23d2c294 measured greedy-identical
        // outputs; the Swift A/B reproduced byte-identical T1/T2/T3.
        #expect(helpers.contains(#"environment["VMLX_DSV4_LM_HEAD_MODE"] ?? "qmm""#))
        #expect(helpers.contains("MLX.quantizedMM("))
        // fp32 accumulation is the correctness half of the contract: the
        // fused path must feed fp32 activations, not bf16.
        #expect(helpers.contains("let hF32 = h.asType(.float32)"))
        // The opt-in cache and its shadow rail must retain an explicit exact
        // path instead of treating qmm's expected rounding delta as a cache
        // mismatch.
        #expect(helpers.contains("lmHeadExactFp32"))
        #expect(
            helpers.contains(
                "let baseline = DeepseekV4Math.lmHeadExactFp32(hidden, lmHead: lmHead)"))
    }

    @Test("RoPE cos/sin tables are shared across equal-frequency instances")
    func ropeTablesAreShared() throws {
        let model = try source("Libraries/MLXLLM/Models/DeepseekV4.swift")

        // Every layer's cosSin must route through the memoized shared table
        // (Python vmlx 4472fe876) — ~61 identical subgraphs per decode token
        // otherwise.
        #expect(model.contains("return DeepseekV4RoPE.sharedCosSin("))
        #expect(model.contains("nonisolated(unsafe) private static var tables:"))
    }

    @Test("Metal live-buffer clamp is wired at both serving admission points")
    func bufferClampIsReached() throws {
        let scheduler = try source(
            "Libraries/MLXLMCommon/BatchEngine/BatchScheduler.swift")
        let evaluate = try source("Libraries/MLXLMCommon/Evaluate.swift")

        // BatchEngine slots (osaurus serving path) and the solo
        // TokenIterator both admit generations; miss either and a
        // DSV4 long generation dies at the 499000 allocator wall.
        #expect(scheduler.contains("MetalLiveBufferGuard.clampedMaxTokens("))
        #expect(evaluate.contains("MetalLiveBufferGuard.clampedMaxTokens("))

        // DSV4's cache layer is the measured retaining topology.
        let compressor = try source(
            "Libraries/MLXLLM/Models/DeepseekV4Compressor.swift")
        #expect(
            compressor.contains(
                "extension DeepseekV4Cache: MetalBufferRetainingCache"))
    }

    @Test("memory-safety plan advertises the limits the loader applies")
    func planMirrorsLoaderResolution() throws {
        let settings = try source("Libraries/MLXLMCommon/ServerRuntimeSettings.swift")

        // resolvedMemorySafetyPlan must route through the bundle-facts
        // resolvers exactly like ModelFactory.loadModel, or the plan
        // surface advertises a decode-throttling cap the engine ignores.
        #expect(settings.contains("facts.resolveMLXMemoryLimit("))
        #expect(settings.contains("facts.resolveMmapSafetensors("))
        #expect(settings.contains("bundleFacts?.resolveMLXAllocatorCacheLimit("))
    }
}
