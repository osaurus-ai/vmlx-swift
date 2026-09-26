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

    func testQuantizationConfigAlias() throws {
        // Hugging Face configs spell the plan `quantization_config` instead of
        // the historical `quantization` key; the alias must decode to the same
        // plan. From https://github.com/osaurus-ai/vmlx-swift/issues/515
        let json =
            """
            {
                "model_type": "qwen3_5",
                "quantization_config": {
                    "group_size": 64,
                    "bits": 4
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "x"),
            .init(groupSize: 64, bits: 4))
    }

    func testQuantizationConfigAliasWithPerLayerOverrides() throws {
        let json =
            """
            {
                "model_type": "Test",
                "quantization_config": {
                    "group_size": 64,
                    "bits": 4,
                    "model.embed_tokens": {
                        "group_size": 32,
                        "bits": 4
                    },
                    "model.layers.0.self_attn.q_norm": false
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "x"),
            .init(groupSize: 64, bits: 4))
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "model.embed_tokens"),
            .init(groupSize: 32, bits: 4))
        XCTAssertNil(
            config.perLayerQuantization?.quantization(layer: "model.layers.0.self_attn.q_norm"))
    }

    func testEquivalentDualQuantizationKeys() throws {
        let json =
            """
            {
                "model_type": "Test",
                "quantization": {
                    "group_size": 64,
                    "bits": 4
                },
                "quantization_config": {
                    "group_size": 64,
                    "bits": 4
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "x"),
            .init(groupSize: 64, bits: 4))
    }

    func testEquivalentDualQuantizationKeysWithPerLayerOverrides() throws {
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
                    }
                },
                "quantization_config": {
                    "group_size": 64,
                    "bits": 4,
                    "model.embed_tokens": {
                        "group_size": 32,
                        "bits": 4
                    }
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "model.embed_tokens"),
            .init(groupSize: 32, bits: 4))
    }

    func testConflictingDualQuantizationKeys() throws {
        let json =
            """
            {
                "model_type": "Test",
                "quantization": {
                    "group_size": 64,
                    "bits": 4
                },
                "quantization_config": {
                    "group_size": 64,
                    "bits": 8
                }
            }
            """

        XCTAssertThrowsError(
            try JSONDecoder().decode(BaseConfiguration.self, from: json.data(using: .utf8)!)
        ) { error in
            guard case DecodingError.dataCorrupted(let context) = error else {
                return XCTFail("Expected dataCorrupted DecodingError, got \(error)")
            }
            XCTAssertTrue(
                context.debugDescription.contains("quantization_config"),
                "Error should name the conflicting alias key: \(context.debugDescription)")
        }
    }

    func testConflictingDualQuantizationKeysWithPerLayerOverrides() throws {
        let json =
            """
            {
                "model_type": "Test",
                "quantization": {
                    "group_size": 64,
                    "bits": 4
                },
                "quantization_config": {
                    "group_size": 64,
                    "bits": 4,
                    "model.embed_tokens": {
                        "group_size": 32,
                        "bits": 4
                    }
                }
            }
            """

        XCTAssertThrowsError(
            try JSONDecoder().decode(BaseConfiguration.self, from: json.data(using: .utf8)!)
        ) { error in
            guard case DecodingError.dataCorrupted(let context) = error else {
                return XCTFail("Expected dataCorrupted DecodingError, got \(error)")
            }
            XCTAssertTrue(
                context.debugDescription.contains("quantization_config"),
                "Error should explain the conflicting plans: \(context.debugDescription)")
        }
    }

    func testQuantizationEncodingUsesHistoricalKey() throws {
        // Encoding must keep writing `quantization` (never `quantization_config`)
        // so existing consumers and caches see the same wire format.
        let json =
            """
            {
                "model_type": "Test",
                "quantization_config": {
                    "group_size": 64,
                    "bits": 4
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)
        let encoded = try JSONEncoder().encode(config)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])

        XCTAssertNotNil(object["quantization"])
        XCTAssertNil(object["quantization_config"])
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "x"),
            .init(groupSize: 64, bits: 4))
    }

    func testEquivalentDualKeysWithImplicitAndExplicitAffineMode() throws {
        let json = #"{"model_type":"test","quantization":{"group_size":64,"bits":4},"quantization_config":{"group_size":64,"bits":4,"mode":"affine"}}"#
        let config = try JSONDecoder().decode(BaseConfiguration.self, from: Data(json.utf8))
        XCTAssertEqual(config.perLayerQuantization?.quantization(layer: "x")?.mode, .affine)
    }

    func testConvertedBundleRetainsForeignQuantizationMetadata() throws {
        let json = #"{"model_type":"step3p5","quantization":{"bits":2,"group_size":128},"quantization_config":{"quant_method":"modelopt","config_groups":{},"quant_algo":"FP8"}}"#
        let config = try JSONDecoder().decode(BaseConfiguration.self, from: Data(json.utf8))
        XCTAssertEqual(config.perLayerQuantization?.quantization(layer: "x")?.bits, 2)
    }

    func testForeignMetadataIsNotAnMLXQuantizationPlan() throws {
        let json = #"{"model_type":"test","quantization_config":{"quant_method":"fp8","weight_block_size":[128,128]}}"#
        let config = try JSONDecoder().decode(BaseConfiguration.self, from: Data(json.utf8))
        XCTAssertNil(config.perLayerQuantization)
    }

    func testMalformedMLXAliasStillThrows() throws {
        for alias in [#"{"quant_method":"mlx"}"#, #"{"bits":4}"#, #"{"group_size":64}"#] {
            let json = "{\"model_type\":\"test\",\"quantization_config\":\(alias)}"
            XCTAssertThrowsError(try JSONDecoder().decode(BaseConfiguration.self, from: Data(json.utf8)))
        }
    }

}
