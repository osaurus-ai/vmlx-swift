import Foundation
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Nemotron 3.5 Bolt JANGH2: gate-less relu2 routed stacks (`fc1` / `fc2`) with
/// per-block geometry from `block_configs`.
final class NemotronHRelu2JANGHTests: XCTestCase {
    private func header() -> [String: Any] {
        var books: [String: [String: Any]] = [:]
        for bits in [2, 3, 4, 6, 8] {
            let count = 1 << bits
            let center = Double(count - 1) / 2
            books[String(bits)] = [
                "alpha": 0.25, "beta": 0.0,
                "levels": (0 ..< count).map { (Double($0) - center) * 0.25 },
            ]
        }
        return [
            "version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
            "codebook_family": "odd-cubic", "rotation": "hadamard32", "codebooks": books,
        ]
    }

    private func relu2Config(_ modules: [String: Any], modelType: String = "nemotron_h") -> [String: Any] {
        var quantization: [String: Any] = ["mode": "affine", "bits": 8, "group_size": 64]
        quantization.merge(modules) { $1 }
        return ["model_type": modelType, "jangtq": header(), "quantization": quantization]
    }

    private func entry(_ bits: Int, _ rotation: String = "hadamard32") -> [String: Any] {
        ["mode": "jangtq2", "bits": bits, "rotation": rotation]
    }

    func testContractAcceptsRelu2PairWithIndependentWidthsAndRotations() throws {
        let parent = "backbone.layers.1.mixer.switch_mlp"
        let config = relu2Config([parent + ".fc1": entry(2), parent + ".fc2": entry(3, "none")])
        let contract = try JANGHFormatContract(
            configuration: JSONSerialization.data(withJSONObject: config))
        XCTAssertTrue(contract.isRelu2Stack(parent))
        XCTAssertEqual(contract.projections[parent + ".fc1"]?.bits, 2)
        XCTAssertEqual(contract.projections[parent + ".fc2"]?.bits, 3)
        XCTAssertEqual(contract.projections[parent + ".fc2"]?.rotation, JANGHFormatContract.Rotation.none)
    }

    func testContractRejectsIncompleteOrMixedRelu2Stacks() {
        let parent = "backbone.layers.1.mixer.switch_mlp"
        for modules: [String: Any] in [
            [parent + ".fc1": entry(2)],
            [parent + ".fc2": entry(2)],
            [parent + ".fc1": entry(2), parent + ".fc2": entry(2), parent + ".down_proj": entry(2)],
            [parent + ".fc1": entry(2), parent + ".fc2": entry(2), parent + ".gate_proj": entry(2)],
            [parent + ".fc3": entry(2)],
        ] {
            XCTAssertThrowsError(
                try JANGHFormatContract(
                    configuration: JSONSerialization.data(withJSONObject: relu2Config(modules))))
        }
    }

    func testPartitionAdmitsNemotronBackbonePathsOnlyForNemotronH() throws {
        let parent = "backbone.layers.3.mixer.switch_mlp"
        let modules: [String: Any] = [parent + ".fc1": entry(2), parent + ".fc2": entry(3)]
        let partition = try JANGHConfigurationPartition(
            configuration: JSONSerialization.data(withJSONObject: relu2Config(modules)))
        XCTAssertEqual(partition.customModules, [parent + ".fc1", parent + ".fc2"])
        let ordinary = try XCTUnwrap(
            JSONSerialization.jsonObject(with: partition.ordinaryConfiguration) as? [String: Any])
        let plan = try XCTUnwrap(ordinary["quantization"] as? [String: Any])
        XCTAssertEqual(plan[parent + ".fc1"] as? Bool, false)
        XCTAssertEqual(plan[parent + ".fc2"] as? Bool, false)

        XCTAssertThrowsError(
            try JANGHConfigurationPartition(
                configuration: JSONSerialization.data(
                    withJSONObject: relu2Config(modules, modelType: "glm5_next"))))
        for bad in [
            "backbone.layers.03.mixer.switch_mlp", "language_model.backbone.layers.3.mixer.switch_mlp",
            "backbone.layers.3.mlp.switch_mlp", "backbone.layers.3.mixer.experts",
        ] {
            XCTAssertThrowsError(
                try JANGHConfigurationPartition(
                    configuration: JSONSerialization.data(
                        withJSONObject: relu2Config([bad + ".fc1": entry(2), bad + ".fc2": entry(2)]))),
                bad)
        }
    }

    private func puzzleConfig(blocks: [[String: Any]], extra: [String: Any] = [:]) -> Data {
        var config: [String: Any] = [
            "model_type": "nemotron_h", "vocab_size": 64, "hidden_size": 64,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "mamba_num_heads": 4, "mamba_head_dim": 16, "ssm_state_size": 16,
            "conv_kernel": 4, "n_groups": 2, "intermediate_size": 96,
            "moe_shared_expert_intermediate_size": 32, "moe_latent_size": 32,
            "n_shared_experts": 1, "num_hidden_layers": blocks.count,
            "layers_block_type": blocks.map { $0["block_type"] as! String },
            "block_configs": blocks, "num_experts": 512, "time_step_limit": [0.0, 1e30],
        ]
        config.merge(extra) { $1 }
        return try! JSONSerialization.data(withJSONObject: config)
    }

    func testPuzzleBlockConfigsDriveMoEGeometry() throws {
        let blocks: [[String: Any]] = [
            ["block_type": "mamba"],
            ["block_type": "moe", "n_routed_experts": 6, "moe_intermediate_size": 64, "num_experts_per_tok": 3],
            ["block_type": "attention"],
            ["block_type": "moe", "n_routed_experts": 10, "moe_intermediate_size": 128, "num_experts_per_tok": 3],
        ]
        let config = try JSONDecoder().decode(NemotronHConfiguration.self, from: puzzleConfig(blocks: blocks))
        XCTAssertEqual(config.hybridOverridePattern, "ME*E")
        // Globals are filled from the first MoE block, never from the stale `num_experts`.
        XCTAssertEqual(config.nRoutedExperts, 6)
        XCTAssertEqual(config.moeIntermediateSize, 64)
        XCTAssertEqual(config.numExpertsPerTok, 3)
        XCTAssertEqual(config.moeGeometry(layer: 1).experts, 6)
        XCTAssertEqual(config.moeGeometry(layer: 3).experts, 10)
        XCTAssertEqual(config.moeGeometry(layer: 3).intermediate, 128)
        XCTAssertEqual(config.timeStepLimitMin, 0, "the configured 0.0 floor is honoured")
    }

    func testPuzzleBlockConfigsMustAgreeWithThePattern() {
        let mismatched: [[String: Any]] = [
            ["block_type": "mamba"],
            ["block_type": "moe", "n_routed_experts": 6, "moe_intermediate_size": 64, "num_experts_per_tok": 3],
        ]
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                NemotronHConfiguration.self,
                from: puzzleConfig(blocks: mismatched, extra: ["layers_block_type": ["mamba", "attention"]])))
        let missingGeometry: [[String: Any]] = [["block_type": "moe"]]
        XCTAssertThrowsError(
            try JSONDecoder().decode(NemotronHConfiguration.self, from: puzzleConfig(blocks: missingGeometry)))
    }

    /// Real-bundle header admission (no tensor reads). Opt-in:
    /// VMLX_BOLT_BUNDLE=<Nemotron-3.5-Bolt-74B-A9B-24GB-JANGH2 directory>.
    func testRealBoltBundleAdmitsEveryRoutedBlock() throws {
        guard let path = ProcessInfo.processInfo.environment["VMLX_BOLT_BUNDLE"] else {
            throw XCTSkip("VMLX_BOLT_BUNDLE not set")
        }
        let directory = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let prepared = try XCTUnwrap(
            try NemotronHJANGHPreparation.loadIfDeclared(directory: directory, configurationData: data))
        let moeBlocks = prepared.configuration.hybridOverridePattern.filter { $0 == "E" }.count
        XCTAssertEqual(moeBlocks, 40)
        XCTAssertEqual(prepared.banks.moduleByLayer.count, 40)
        XCTAssertEqual(prepared.banks.excludedTensorNames.count, 160)
    }
}
