import Foundation
import XCTest

@testable import MLXLMCommon

final class JANGHConfigurationPartitionTests: XCTestCase {
    private let prefix = "model.layers.3.mlp.switch_mlp."

    private func configuration() -> [String: Any] {
        let book: [String: Any] = ["alpha": 1.0, "beta": 0.0, "levels": [-1.5, -0.5, 0.5, 1.5]]
        let header: [String: Any] = [
            "version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
            "codebook_family": "odd-cubic", "rotation": "hadamard32", "codebooks": ["2": book],
        ]
        var plan: [String: Any] = [
            "mode": "affine", "bits": 8, "group_size": 64,
            "model.layers.3.self_attn.q_proj": ["mode": "mxfp8", "bits": 8, "group_size": 32],
        ]
        for role in ["gate_proj", "up_proj", "down_proj"] {
            plan[prefix + role] = ["mode": "jangtq2", "bits": 2, "rotation": "hadamard32"]
        }
        return ["model_type": "glm5_next", "jangtq": header,
                "quantization": plan, "quantization_config": plan]
    }

    private func data(_ value: [String: Any]) throws -> Data {
        try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
    }

    private func sidecar(_ config: [String: Any]) -> [String: Any] {
        ["format": "jangtq2", "format_version": 2, "jangtq": config["jangtq"]!]
    }

    func testFlintCalibrationSidecarPreservesAuthoritativePlan() throws {
        var config = configuration()
        config["model_type"] = "flint"
        let summary: [String: Any] = [
            "format": "jangtq2", "format_version": 2,
            "quantization": [
                "routed_experts": "Hadamard-32 codebook experts",
                "non_experts": "affine 8-bit g64",
                "gptq_vs_rtn": ["non_expert_gptq": 344],
                "awq_moe_in": [prefix + "gate_proj": ["alpha_gu": 0.0, "pow2": true]],
                "awq_attn": ["L00": ["attn_in": ["alpha": 0.3]]],
                "expert_units_jangh": 3,
            ],
        ]
        let plain = try JANGHConfigurationPartition(configuration: data(config))
        let prepared = try JANGHConfigurationPartition(configuration: data(config), sidecar: data(summary))
        XCTAssertEqual(plain.customConfiguration, prepared.customConfiguration)
        XCTAssertEqual(plain.ordinaryConfiguration, prepared.ordinaryConfiguration)
        XCTAssertNoThrow(try JSONDecoder().decode(BaseConfiguration.self, from: prepared.ordinaryConfiguration))

        var invalid = summary
        invalid["quantization_config"] = config["quantization"]
        XCTAssertThrowsError(try JANGHConfigurationPartition(configuration: data(config), sidecar: data(invalid)))
        invalid = summary
        invalid["quantization"] = [prefix + "gate_proj": ["mode": "affine", "bits": 2]]
        XCTAssertThrowsError(try JANGHConfigurationPartition(configuration: data(config), sidecar: data(invalid)))
        invalid = summary
        invalid["quantization"] = ["awq_attn": ["L00": ["mode": "affine"]]]
        XCTAssertThrowsError(try JANGHConfigurationPartition(configuration: data(config), sidecar: data(invalid)))
        invalid = summary
        var conflicting = config["jangtq"] as! [String: Any]
        conflicting["rotation"] = "none"
        invalid["jangtq"] = conflicting
        XCTAssertThrowsError(try JANGHConfigurationPartition(configuration: data(config), sidecar: data(invalid)))
    }

    private func denseQwenConfiguration() -> [String: Any] {
        var config = configuration()
        config["model_type"] = "qwen3_5"
        var plan = config["quantization"] as! [String: Any]
        for role in ["gate_proj", "up_proj", "down_proj"] {
            plan["language_model.model.layers.3.mlp." + role] = plan.removeValue(
                forKey: prefix + role)
        }
        config["quantization"] = plan
        config["quantization_config"] = plan
        return config
    }

    private func denseQwenSummary() -> [String: Any] {
        // Actual local 27B JANGH2 sidecar shape: format_version is a string,
        // quantization is a scalar summary, and the authoritative header is in config.
        [
            "format": "jangtq2", "format_version": "2.0",
            "quantization": ["bits": 4, "group_size": 64, "bit_widths_used": [2, 8]],
            "chat": ["sampling_defaults": ["temperature": 1.0]],
            "runtime": ["bundle_has_mtp": false],
        ]
    }

    func testDenseQwenSummaryPreservesAuthoritativePlanAndChecksHeaders() throws {
        let config = denseQwenConfiguration()
        var summary = denseQwenSummary()
        let plain = try JANGHConfigurationPartition(configuration: data(config))
        let withSummary = try JANGHConfigurationPartition(
            configuration: data(config), sidecar: data(summary))
        XCTAssertEqual(plain.customConfiguration, withSummary.customConfiguration)
        XCTAssertEqual(plain.ordinaryConfiguration, withSummary.ordinaryConfiguration)
        summary["jangtq"] = config["jangtq"]
        XCTAssertNoThrow(
            try JANGHConfigurationPartition(configuration: data(config), sidecar: data(summary)))
        var conflicting = config["jangtq"] as! [String: Any]
        conflicting["rotation"] = "none"
        summary["jangtq"] = conflicting
        XCTAssertThrowsError(
            try JANGHConfigurationPartition(configuration: data(config), sidecar: data(summary)))
        summary["jangtq"] = NSNull()
        XCTAssertThrowsError(
            try JANGHConfigurationPartition(configuration: data(config), sidecar: data(summary)))
    }

    func testDenseQwenSummaryCannotIntroduceASecondPlan() throws {
        let config = denseQwenConfiguration()
        for replacement: [String: Any] in [
            ["quantization_config": config["quantization"]!],
            [
                "quantization": [
                    "bits": 4, "group_size": 64, "bit_widths_used": [2, 8],
                    "language_model.model.layers.3.mlp.gate_proj": ["bits": 4],
                ]
            ],
            ["quantization": ["mode": "affine", "bits": 4, "group_size": 64]],
            ["format_version": "3.0"], ["format": "affine"],
        ] {
            var summary = denseQwenSummary()
            for (key, value) in replacement { summary[key] = value }
            XCTAssertThrowsError(
                try JANGHConfigurationPartition(configuration: data(config), sidecar: data(summary))
            )
        }
        var missingHeader = config
        missingHeader.removeValue(forKey: "jangtq")
        XCTAssertThrowsError(
            try JANGHConfigurationPartition(
                configuration: data(missingHeader), sidecar: data(denseQwenSummary())))
    }

    func testQwenMoERoutedPlanKeepsAffineMTPAndCalibrationMetadataSeparate() throws {
        var config = configuration()
        config["model_type"] = "qwen3_5_moe"
        var plan = config["quantization"] as! [String: Any]
        for role in ["gate_proj", "up_proj", "down_proj"] {
            plan["language_model." + prefix + role] = plan.removeValue(forKey: prefix + role)
        }
        let mtp = "mtp.layers.0.mlp.experts.0.gate_proj"
        plan[mtp] = ["mode": "affine", "bits": 4, "group_size": 64]
        config["quantization"] = plan
        config["quantization_config"] = plan
        var summary: [String: Any] = [
            "format": "jangtq2", "format_version": "2.0",
            "quantization": ["method": "jang-gptq", "block_size": 64,
                "bit_widths_used": [2, 3, 4, 8], "profile": "JANGH2",
                "awq": ["experts": ["folds": "calibration provenance"]]],
        ]
        let partition = try JANGHConfigurationPartition(
            configuration: data(config), sidecar: data(summary))
        XCTAssertEqual(partition.customModules.count, 3)
        let ordinary = try XCTUnwrap(JSONSerialization.jsonObject(with: partition.ordinaryConfiguration)
            as? [String: Any])
        let result = try XCTUnwrap(ordinary["quantization"] as? [String: Any])
        XCTAssertEqual((result[mtp] as? [String: Any])?["bits"] as? Int, 4)
        for role in ["gate_proj", "up_proj", "down_proj"] {
            XCTAssertEqual(result["language_model." + prefix + role] as? Bool, false)
        }
        var badPlan = summary["quantization"] as! [String: Any]
        badPlan[mtp] = ["bits": 8]
        summary["quantization"] = badPlan
        XCTAssertThrowsError(try JANGHConfigurationPartition(configuration: data(config), sidecar: data(summary)))
        summary.removeValue(forKey: "quantization")
        var wrongHeader = config["jangtq"] as! [String: Any]
        wrongHeader["rotation"] = "none"
        summary["jangtq"] = wrongHeader
        XCTAssertThrowsError(try JANGHConfigurationPartition(configuration: data(config), sidecar: data(summary)))
    }

    func testDenseK2RetainsStrictSidecarOwnership() throws {
        var config = configuration()
        config["model_type"] = "k2_horizon"
        config["mlp_layout"] = "dense_jangh_down"
        let projection = (config["quantization"] as! [String: Any])[prefix + "down_proj"]!
        let plan: [String: Any] = [
            "mode": "affine", "bits": 8, "group_size": 64,
            "model.layers.3.mlp.down_proj": projection,
        ]
        config["quantization"] = plan
        config["quantization_config"] = plan
        XCTAssertNoThrow(
            try JANGHConfigurationPartition(
                configuration: data(config), sidecar: data(sidecar(config))))
        XCTAssertThrowsError(
            try JANGHConfigurationPartition(
                configuration: data(config), sidecar: data(denseQwenSummary())))
        var conflicting = sidecar(config)
        var header = config["jangtq"] as! [String: Any]
        header["rotation"] = "none"
        conflicting["jangtq"] = header
        XCTAssertThrowsError(
            try JANGHConfigurationPartition(configuration: data(config), sidecar: data(conflicting))
        )
    }

    func testActualDenseQwenMetadataWhenProvided() throws {
        guard
            let path = ProcessInfo.processInfo.environment[
                "VMLX_DENSE_QWEN_JANGH_METADATA_DIRECTORY"]
        else {
            throw XCTSkip("Optional read-only local metadata fixture")
        }
        let directory = URL(fileURLWithPath: path)
        let partition = try JANGHConfigurationPartition(
            configuration: Data(contentsOf: directory.appendingPathComponent("config.json")),
            sidecar: Data(contentsOf: directory.appendingPathComponent("jang_config.json")))
        XCTAssertEqual(partition.modelType, "qwen3_5")
        XCTAssertFalse(partition.customModules.isEmpty)
    }

    func testEquivalentAliasesKeepCustomSkipsAndOrdinaryModes() throws {
        for modelType in ["glm5_next", "naive_n05_flash"] {
            var config = configuration()
            config["model_type"] = modelType
            let partition = try JANGHConfigurationPartition(
                configuration: data(config), sidecar: data(sidecar(config)))
            XCTAssertEqual(partition.modelType, modelType)
            XCTAssertEqual(partition.customModules.count, 3)
            let ordinary = try XCTUnwrap(
                JSONSerialization.jsonObject(with: partition.ordinaryConfiguration) as? [String: Any])
            let plan = try XCTUnwrap(ordinary["quantization"] as? [String: Any])
            XCTAssertEqual(try data(plan), try data(ordinary["quantization_config"] as! [String: Any]))
            for name in partition.customModules { XCTAssertEqual(plan[name] as? Bool, false) }
            XCTAssertEqual(plan["mode"] as? String, "affine")
            let attention = plan["model.layers.3.self_attn.q_proj"] as! [String: Any]
            XCTAssertEqual(attention["mode"] as? String, "mxfp8")
            XCTAssertEqual(try JANGHFormatContract(configuration: partition.customConfiguration).projections.count, 3)
        }
    }

    func testSidecarOnlyHeaderAliasOnlyPlanAndPerTensorPlan() throws {
        let original = configuration()
        var config = original
        config.removeValue(forKey: "jangtq")
        config.removeValue(forKey: "quantization")
        var plan = original["quantization"] as! [String: Any]
        var nested: [String: Any] = [:]
        for name in plan.keys.sorted() where name.contains(".") {
            nested[name] = plan.removeValue(forKey: name)
        }
        plan["per_tensor"] = nested
        config["quantization_config"] = plan
        let result = try JANGHConfigurationPartition(
            configuration: data(config), sidecar: data(sidecar(original)))
        XCTAssertEqual(result.customModules.count, 3)
        // Equivalent flat and nested definitions have one unambiguous owner.
        plan[prefix + "gate_proj"] = nested[prefix + "gate_proj"]
        config["quantization_config"] = plan
        config["quantization"] = original["quantization"]
        XCTAssertNoThrow(try JANGHConfigurationPartition(
            configuration: data(config), sidecar: data(sidecar(original))))
    }

    func testContradictionsAndUnsupportedOwnersFailClosed() throws {
        let original = configuration()
        var badAlias = original
        var plan = original["quantization"] as! [String: Any]
        plan["bits"] = 4
        badAlias["quantization_config"] = plan
        XCTAssertThrowsError(try JANGHConfigurationPartition(configuration: data(badAlias)))
        var badHeader = sidecar(original)
        var header = original["jangtq"] as! [String: Any]
        header["rotation"] = "none"
        badHeader["jangtq"] = header
        XCTAssertThrowsError(try JANGHConfigurationPartition(
            configuration: data(original), sidecar: data(badHeader)))
        for value in ["legacy", "jangtq"] {
            var other = sidecar(original)
            other["format"] = value
            XCTAssertThrowsError(try JANGHConfigurationPartition(
                configuration: data(original), sidecar: data(other)))
        }
        for version in [1 as Any, true] {
            var other = sidecar(original)
            other["format_version"] = version
            XCTAssertThrowsError(try JANGHConfigurationPartition(
                configuration: data(original), sidecar: data(other)))
        }
        var nested = original
        nested["text_config"] = ["quantization": original["quantization"]!]
        XCTAssertThrowsError(try JANGHConfigurationPartition(configuration: data(nested)))
        var collision = original
        var collided = original["quantization"] as! [String: Any]
        collided["per_tensor"] = [prefix + "gate_proj": ["mode": "affine", "bits": 2]]
        collision["quantization"] = collided
        collision.removeValue(forKey: "quantization_config")
        XCTAssertThrowsError(try JANGHConfigurationPartition(configuration: data(collision)))
        for alias in ["layers.3.mlp.switch_mlp.gate_proj",
                      "language_model.model.layers.3.mlp.switch_mlp.gate_proj"] {
            var aliased = original
            var aliasedPlan = original["quantization"] as! [String: Any]
            aliasedPlan[alias] = ["mode": "affine", "bits": 4, "group_size": 64]
            aliased["quantization"] = aliasedPlan
            aliased["quantization_config"] = aliasedPlan
            XCTAssertThrowsError(try JANGHConfigurationPartition(configuration: data(aliased)))
        }
    }

    func testSharedExpertCustomTriplesCannotBypassGenericAliasExclusions() throws {
        var config = configuration()
        var plan = config["quantization"] as! [String: Any]
        for role in ["gate_proj", "up_proj", "down_proj"] {
            plan["model.layers.3.mlp.shared_experts." + role] = plan.removeValue(forKey: prefix + role)
        }
        // Generic BaseConfiguration maps this spelling to shared_experts.gate_proj.
        // Such a custom triple is unsupported by this routed-bank adapter even
        // before the ordinary alias is added; it must not be admitted by name alone.
        for addOrdinaryAlias in [false, true] {
            if addOrdinaryAlias {
                plan["model.layers.3.ffn.shared_experts.w1"] = ["mode": "affine", "bits": 4, "group_size": 64]
            }
            config["quantization"] = plan
            config["quantization_config"] = plan
            XCTAssertThrowsError(try JANGHConfigurationPartition(configuration: data(config)))
        }
    }

    func testUnknownOrdinaryModeIsPreservedAndCustomUnknownIsRejected() throws {
        var config = configuration()
        var plan = config["quantization"] as! [String: Any]
        let name = "model.layers.3.self_attn.q_proj"
        plan[name] = ["mode": "future-unknown", "bits": 8] // inherited group_size path
        config["quantization"] = plan
        config["quantization_config"] = plan
        let result = try JANGHConfigurationPartition(configuration: data(config))
        let ordinary = try JSONSerialization.jsonObject(with: result.ordinaryConfiguration) as! [String: Any]
        let ordinaryPlan = ordinary["quantization"] as! [String: Any]
        XCTAssertEqual((ordinaryPlan[name] as! [String: Any])["mode"] as? String, "future-unknown")
        XCTAssertNil((ordinaryPlan[name] as! [String: Any])["group_size"])
        plan[prefix + "gate_proj"] = ["mode": "future-unknown", "bits": 2]
        config["quantization"] = plan
        config["quantization_config"] = plan
        XCTAssertThrowsError(try JANGHConfigurationPartition(configuration: data(config)))
    }

    func testDuplicateJSONKeysAndLegacyLabelAloneCannotAdmitCustomBanks() throws {
        let config = configuration()
        var noHeader = config
        noHeader.removeValue(forKey: "jangtq")
        noHeader["weight_format"] = "jangtq2"
        XCTAssertThrowsError(try JANGHConfigurationPartition(configuration: data(noHeader)))
        let encoded = String(decoding: try data(config), as: UTF8.self)
        let duplicate = "{\"model_type\":\"wrong\"," + encoded.dropFirst()
        XCTAssertThrowsError(try JANGHConfigurationPartition(configuration: Data(duplicate.utf8)))
        var other = sidecar(config)
        other["quantization"] = config["quantization"]
        XCTAssertThrowsError(try JANGHConfigurationPartition(
            configuration: data(config), sidecar: data(other)))
    }
}
