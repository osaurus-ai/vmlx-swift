// Copyright © 2025 Apple Inc.

import Foundation
import MLXLMCommon
import XCTest

public class BaseConfigurationTests: XCTestCase {

    func testQuantization() throws {
        let json =
            """
            {
                "model_type": "Test",
                "quantization": {
                    "group_size": 128,
                    "bits": 4
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "x"), .init(groupSize: 128, bits: 4))
    }

    func testHeterogenousQuantization() throws {
        // from https://huggingface.co/mlx-community/Qwen3-1.7B-4bit-AWQ/blob/main/config.json#L20
        let json =
            """
            {
                "model_type": "Test",
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "model.embed_tokens": {
                        "group_size": 32,
                        "bits": 4
                    },
                    "model.layers.0.self_attn.q_norm": false,
                    "true_layer": true
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        // a random layer -- no specific configuration gets default
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "x"),
            .init(groupSize: 64, bits: 4))

        // layer with an override
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "model.embed_tokens"),
            .init(groupSize: 32, bits: 4))

        // layer with an override -- not quant
        XCTAssertNil(
            config.perLayerQuantization?.quantization(layer: "model.layers.0.self_attn.q_norm"))

        // layer with an override -- true, use the default
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "true_layer"),
            .init(groupSize: 64, bits: 4))
    }

    func testDSV4RoutedExpertBitPlanIsQuantizationMetadata() throws {
        let json =
            """
            {
                "model_type": "deepseek_v4",
                "weight_format": "mxtq",
                "quantization": {
                    "bits": 8,
                    "group_size": 32,
                    "mode": "affine",
                    "routed_expert_bits": 2,
                    "routed_expert_bit_plan": {
                        "default_bits": 2,
                        "codec": "mxtq",
                        "routed_layer_bits": {
                            "23": 4,
                            "25": 4,
                            "28": 4,
                            "34": 4,
                            "36": 4
                        }
                    },
                    "mxtq_bits": {
                        "routed_expert": 2,
                        "attention": 8,
                        "shared_expert": 8
                    }
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        let fallback = config.perLayerQuantization?.quantization(layer: "model.layers.23.mlp.experts")
        XCTAssertEqual(fallback?.groupSize, 32)
        XCTAssertEqual(fallback?.bits, 8)
        XCTAssertEqual(fallback?.mode, .affine)
        XCTAssertTrue(
            config.perLayerQuantization?.perLayerQuantization.isEmpty ?? false,
            "DSV4 routed_expert_bit_plan is consumed by DeepseekV4Configuration/JANGTQ, not by BaseConfiguration per-layer affine overrides")
    }

    func testMXFPQuantizationMetadataDoesNotDecodeAsLayerOverride() throws {
        let json =
            """
            {
                "model_type": "qwen3_5",
                "quantization": {
                    "bits": 8,
                    "group_size": 32,
                    "mode": "mxfp8",
                    "quantization_backend": "mx.quantize",
                    "vision": "fp16_passthrough",
                    "mtp": "preserved",
                    "mtp_policy": "native_mxfp8_linears_fp16_norms",
                    "passthrough_bit_widths_used": [16],
                    "norm_convention": "qwen3_5_language_mlx_plus_one",
                    "model.embed_tokens": {
                        "bits": 8,
                        "group_size": 32,
                        "mode": "mxfp8"
                    },
                    "model.layers.0.self_attn.q_norm": false
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        let fallback = config.perLayerQuantization?.quantization(layer: "model.layers.0.mlp.down_proj")
        XCTAssertEqual(fallback?.groupSize, 32)
        XCTAssertEqual(fallback?.bits, 8)
        XCTAssertEqual(fallback?.mode, .mxfp8)
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "model.embed_tokens")?.mode,
            .mxfp8)
        XCTAssertNil(
            config.perLayerQuantization?.quantization(layer: "model.layers.0.self_attn.q_norm"))
        XCTAssertEqual(
            config.perLayerQuantization?.perLayerQuantization.count,
            2,
            "Only real per-layer dictionary overrides and false skips should enter the override map.")
    }

}
