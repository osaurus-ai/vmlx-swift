// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Focused coverage for the Qwen 3.5 text-only compile-decode policy (issue #453):
//
// 1. The shared `Qwen35CompiledDecodePolicy` predicate is architecture-scoped:
//    only the validated `qwen3_5_moe_text` shape enables, neighbors do not, and
//    the explicit `VMLX_QWEN35_COMPILE_DECODE_REGIONS` override (current and
//    legacy spelling) opts out or forces on.
// 2. The text-only construction actually threads that policy into the
//    `SwitchGLU(compileSeparatedDecode:)` decision: eligible topologies reach
//    the trusted routed-MoE decode region, neighbors fall back to eager, and
//    the MTP layers keep the eager default (mirroring the VLM path). The
//    region's own shape/dtype/quantization guards and the eager fallback are
//    unchanged; they live in `SwitchLayers.swift`.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXLLM

@Suite("Qwen3.5 text-only compile-decode policy")
struct Qwen35CompiledDecodePolicyTests {

    private static let variable = Qwen35CompiledDecodePolicy.environmentVariable

    private static func shouldCompile(
        modelType: String = "qwen3_5_moe_text",
        hiddenSize: Int = 2048,
        hiddenLayers: Int = 40,
        fullAttentionInterval: Int = 4,
        numExperts: Int = 256,
        numExpertsPerTok: Int = 8,
        moeIntermediateSize: Int = 512,
        linearNumKeyHeads: Int = 16,
        linearNumValueHeads: Int = 32,
        linearKeyHeadDim: Int = 128,
        linearValueHeadDim: Int = 128,
        environment: [String: String] = [:]
    ) -> Bool {
        Qwen35CompiledDecodePolicy.shouldCompileDecodeRegions(
            modelType: modelType,
            hiddenSize: hiddenSize,
            hiddenLayers: hiddenLayers,
            fullAttentionInterval: fullAttentionInterval,
            numExperts: numExperts,
            numExpertsPerTok: numExpertsPerTok,
            moeIntermediateSize: moeIntermediateSize,
            linearNumKeyHeads: linearNumKeyHeads,
            linearNumValueHeads: linearNumValueHeads,
            linearKeyHeadDim: linearKeyHeadDim,
            linearValueHeadDim: linearValueHeadDim,
            environment: environment)
    }

    /// The validated `qwen3_5_moe_text` topology enables the compiled region.
    @Test("eligible Qwen3.5 MoE text topology enables the compiled region")
    func eligibleTopologyEnables() {
        #expect(Self.shouldCompile())
    }

    /// Neighboring shapes must not enable compilation: the region was not
    /// validated for them, so they keep the eager routed path.
    @Test("neighboring shapes stay eager")
    func neighboringShapesStayEager() {
        #expect(!Self.shouldCompile(modelType: "qwen3_5_moe"))
        #expect(!Self.shouldCompile(hiddenSize: 2560))
        #expect(!Self.shouldCompile(hiddenLayers: 48))
        #expect(!Self.shouldCompile(fullAttentionInterval: 8))
        #expect(!Self.shouldCompile(numExperts: 128))
        #expect(!Self.shouldCompile(numExpertsPerTok: 4))
        #expect(!Self.shouldCompile(moeIntermediateSize: 640))
        #expect(!Self.shouldCompile(linearNumKeyHeads: 8))
        #expect(!Self.shouldCompile(linearNumValueHeads: 16))
        #expect(!Self.shouldCompile(linearKeyHeadDim: 192))
        #expect(!Self.shouldCompile(linearValueHeadDim: 64))
    }

    /// The explicit override wins over the architecture match in both
    /// directions, and the legacy `VMLINUX_` spelling still works.
    @Test("explicit override opts out or forces on, legacy spelling honored")
    func explicitOverrideOptsOutOrForcesOn() {
        #expect(!Self.shouldCompile(environment: [Self.variable: "0"]))
        #expect(!Self.shouldCompile(environment: [Self.variable: "false"]))
        #expect(
            Self.shouldCompile(
                hiddenSize: 2560, environment: [Self.variable: "1"]))
        #expect(
            !Self.shouldCompile(
                environment: ["VMLINUX_QWEN35_COMPILE_DECODE_REGIONS": "0"]))
        #expect(
            Self.shouldCompile(
                hiddenSize: 2560,
                environment: ["VMLINUX_QWEN35_COMPILE_DECODE_REGIONS": "true"]))
    }

    /// The text-only decoder layer feeds the shared policy into the MoE block.
    /// Only the eligible topology may attempt the compiled region; a neighbor
    /// falls back to the eager `SwitchGLU` path.
    @Test("text decoder threads the policy into the routed-MoE block")
    func textDecoderThreadsPolicy() throws {
        var eligible = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data("{}".utf8))
        eligible.modelType = "qwen3_5_moe_text"
        eligible.hiddenSize = 2048
        eligible.hiddenLayers = 40
        eligible.fullAttentionInterval = 4
        eligible.numExperts = 256
        eligible.numExpertsPerTok = 8
        eligible.moeIntermediateSize = 512
        eligible.linearNumKeyHeads = 16
        eligible.linearNumValueHeads = 32
        eligible.linearKeyHeadDim = 128
        eligible.linearValueHeadDim = 128
        eligible.sharedExpertIntermediateSize = 64

        let eligibleBlock = try #require(
            Qwen35DecoderLayer(eligible, layerIdx: 0).mlp
                as? Qwen35SparseMoeBlock)
        #expect(eligibleBlock.compileDecodeRegions)

        var neighbor = eligible
        neighbor.hiddenSize = 2560
        let neighborBlock = try #require(
            Qwen35DecoderLayer(neighbor, layerIdx: 0).mlp
                as? Qwen35SparseMoeBlock)
        #expect(!neighborBlock.compileDecodeRegions)

        // MTP layers keep the eager default, exactly like the VLM's MTP path.
        let mtpBlock = try #require(
            Qwen35MTPDecoderLayer(eligible).mlp as? Qwen35SparseMoeBlock)
        #expect(!mtpBlock.compileDecodeRegions)
    }
}
