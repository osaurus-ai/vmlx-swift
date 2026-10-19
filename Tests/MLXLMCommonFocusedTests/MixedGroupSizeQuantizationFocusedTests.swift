// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("Mixed group-size quantization")
struct MixedGroupSizeQuantizationFocusedTests {
    private let bareExpertPath = "layers.1.mlp.switch_mlp.gate_proj"

    @Test("Qwen4 MTP split banks retain checkpoint quantization")
    func qwen4MTPSplitBanksRetainQuantization() {
        let expert = BaseConfiguration.Quantization(groupSize: 64, bits: 2)
        let runtimeGate = "mtp.layers.0.mlp.switch_mlp.gate_proj"
        var declarations: [String: BaseConfiguration.QuantizationOption] = [
            "mtp.layers.0.mlp.experts.gate_up_proj": .quantize(expert),
            "mtp.layers.0.mlp.experts.down_proj": .quantize(expert),
        ]
        func settings() -> BaseConfiguration.PerLayerQuantization {
            .init(quantization: .init(groupSize: 64, bits: 8),
                  perLayerQuantization: declarations)
        }
        for prefix in ["", "model.", "language_model.", "language_model.model."] {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                #expect(settings().quantization(
                    layer: "\(prefix)mtp.layers.0.mlp.switch_mlp.\(projection)") == expert)
            }
        }
        // Exact runtime metadata keeps precedence. The alias is MTP-only,
        // and an explicit skip must survive the split as well.
        declarations[runtimeGate] = .quantize(.init(groupSize: 64, bits: 4))
        #expect(settings().quantization(layer: runtimeGate)?.bits == 4)
        #expect(settings().quantization(layer: "layers.0.mlp.switch_mlp.gate_proj")?.bits == 8)
        declarations.removeValue(forKey: runtimeGate)
        declarations["mtp.layers.0.mlp.experts.gate_up_proj"] = .skip
        #expect(settings().quantization(layer: runtimeGate) == nil)
        #expect(settings().declaresSkipForAnyAlias(of: runtimeGate))
    }

    @Test("model-prefixed override resolves for a bare module path")
    func modelPrefixedOverrideResolvesForBareModulePath() throws {
        let configJSON = """
            {
              "model_type": "laguna",
              "quantization": {
                "bits": 8,
                "group_size": 64,
                "model.layers.1.mlp.switch_mlp.gate_proj": {
                  "bits": 4,
                  "group_size": 128,
                  "mode": "affine"
                }
              }
            }
            """
        let config = try JSONDecoder().decode(
            BaseConfiguration.self,
            from: Data(configJSON.utf8))

        let bare = config.perLayerQuantization?.quantization(layer: bareExpertPath)
        let prefixed = config.perLayerQuantization?.quantization(
            layer: "model.\(bareExpertPath)")

        #expect(bare?.bits == 4)
        #expect(bare?.groupSize == 128)
        #expect(prefixed == bare)
    }

    @Test("Raptor expert shapes use the module group size during inference")
    func raptorExpertShapesUseModuleGroupSizeDuringInference() {
        let checkpointPath = "model.\(bareExpertPath)"
        let declared = BaseConfiguration.PerLayerQuantization(
            quantization: BaseConfiguration.Quantization(
                groupSize: 64, bits: 8, mode: .affine),
            perLayerQuantization: [
                checkpointPath: .quantize(
                    BaseConfiguration.Quantization(
                        groupSize: 128, bits: 4, mode: .affine))
            ])

        guard let expert = declared.quantization(layer: bareExpertPath) else {
            Issue.record("Expected the Raptor routed-expert quantization override")
            return
        }

        // The real expert tensor is [128, 512, 256] and its scales are
        // [128, 512, 16]. The resolved module group size must establish
        // in_features = scales_last * group_size = 16 * 128 = 2048 and
        // therefore bits = packed_last * 32 / in_features = 4.
        let inferred = JangLoader.inferBitWidthAndGroupSize(
            packedDim: 256,
            numGroups: 16,
            knownGroupSize: expert.groupSize,
            bitWidthsUsed: [4, 8])

        #expect(inferred.bits == 4)
        #expect(inferred.groupSize == 128)
        #expect(16 * inferred.groupSize == 2048)
    }

    @Test("model-prefixed skip is not replaced by the global default")
    func modelPrefixedSkipRemainsSkipped() {
        let settings = BaseConfiguration.PerLayerQuantization(
            quantization: BaseConfiguration.Quantization(groupSize: 64, bits: 8),
            perLayerQuantization: ["model.\(bareExpertPath)": .skip])

        #expect(settings.quantization(layer: bareExpertPath) == nil)
    }

    @Test("DSV4 sanitized self-attention path resolves checkpoint attention override")
    func dsv4SanitizedAttentionPathResolvesCheckpointOverride() throws {
        let configJSON = """
            {
              "model_type": "deepseek_v4",
              "quantization": {
                "bits": 2,
                "group_size": 64,
                "layers.0.attn.wq_a": {
                  "bits": 4,
                  "group_size": 64,
                  "mode": "affine"
                }
              }
            }
            """
        let config = try JSONDecoder().decode(
            BaseConfiguration.self,
            from: Data(configJSON.utf8))

        let sanitized = config.perLayerQuantization?.quantization(
            layer: "model.layers.0.self_attn.wq_a")

        #expect(sanitized?.bits == 4)
        #expect(sanitized?.groupSize == 64)
    }

    @Test("DSV4 sanitized shared-expert paths resolve checkpoint w1 w2 w3 overrides")
    func dsv4SanitizedSharedExpertPathsResolveCheckpointOverrides() throws {
        let configJSON = """
            {
              "model_type": "deepseek_v4",
              "quantization": {
                "bits": 2,
                "group_size": 64,
                "layers.0.ffn.shared_experts.w1": {
                  "bits": 4,
                  "group_size": 64
                },
                "layers.0.ffn.shared_experts.w2": {
                  "bits": 4,
                  "group_size": 64
                },
                "layers.0.ffn.shared_experts.w3": {
                  "bits": 4,
                  "group_size": 64
                }
              }
            }
            """
        let config = try JSONDecoder().decode(
            BaseConfiguration.self,
            from: Data(configJSON.utf8))

        for projection in ["gate_proj", "down_proj", "up_proj"] {
            let resolved = config.perLayerQuantization?.quantization(
                layer: "model.layers.0.mlp.shared_experts.\(projection)")
            #expect(resolved?.bits == 4, "\(projection) must stay 4-bit")
            #expect(resolved?.groupSize == 64, "\(projection) must stay group-64")
        }
    }
}
