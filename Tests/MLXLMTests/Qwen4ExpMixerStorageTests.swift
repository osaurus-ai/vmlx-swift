import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXVLM

final class Qwen4ExpMixerStorageTests: XCTestCase {
    func testDenseFusionReplacesMemberAllocationsWithoutChangingWeightsOrRepeatedOutput() throws {
        try checkStorage(metadataType: nil)
    }

    func testAffineBF16FusionReplacesMemberAllocations() throws {
        try checkStorage(metadataType: .bfloat16)
    }

    func testAffineF16FusionReplacesMemberAllocations() throws {
        try checkStorage(metadataType: .float16)
    }

    private func checkStorage(metadataType: DType?) throws {
        try MLXMetalTestLock.withLock {
            let configuration: [String: Any] = [
                "model_type": "qwen4_exp",
                "jang_config": ["norm_convention": "runtime_plus1_applied"],
                "text_config": [
                    "model_type": "qwen4_exp_text", "dtype": "bfloat16",
                    "hidden_size": 2560, "num_hidden_layers": 1, "intermediate_size": 640,
                    "num_attention_heads": 20, "num_key_value_heads": 2, "head_dim": 128,
                    "vocab_size": 128, "num_experts": 4, "num_experts_per_tok": 2,
                    "moe_intermediate_size": 640, "shared_expert_intermediate_size": 640,
                    "hc_count": 4, "hc_lowrank": 320, "ple_layer_ids": [],
                ],
            ]
            let config = try JSONDecoder().decode(
                Qwen4ExpConfiguration.self,
                from: JSONSerialization.data(withJSONObject: configuration))
            let width = 10240
            func bank(_ rows: Int, _ columns: Int, phase: Int) -> MLXArray {
                MLXArray(
                    (0 ..< rows * columns).map { Float(($0 + phase) % 17 - 8) / 1024 },
                    [rows, columns]
                ).asType(.bfloat16)
            }
            var parameters = [
                "hc_norm.weight": MLXArray.ones([width], dtype: .bfloat16),
                "input_mix_weight_down.weight": bank(320, width, phase: 1),
                "input_mix_weight_up.weight": bank(width, 320, phase: 2),
                "block_inject_weight.weight": bank(4, width, phase: 3),
            ]
            MLX.eval(Array(parameters.values))
            let bridge = try Qwen4ExpMHCSubstageBridge(config: config, parameters: parameters)
            parameters.removeAll()  // The caller must not keep the original banks alive.
            // Reuse the existing diagnostic bridge without adding a shipping
            // API to expose its private mixer solely for this storage test.
            let module = try XCTUnwrap(
                Mirror(reflecting: bridge).children.first { $0.label == "residual" }?.value
                    as? Module)
            if let metadataType {
                quantize(model: module, groupSize: 64, bits: 8)
                for (_, child) in module.leafModules().flattened() {
                    guard let linear = child as? QuantizedLinear, let biases = linear.biases else {
                        continue
                    }
                    _ = linear.update(
                        parameters: ModuleParameters.unflattened([
                            "scales": linear.scales.asType(metadataType),
                            "biases": biases.asType(metadataType),
                        ]))
                }
            }
            func snapshot() -> [String: Data] {
                Dictionary(
                    uniqueKeysWithValues: module.parameters().flattened().map {
                        ($0.0, $0.1.asData(access: .copy).data)
                    })
            }
            MLX.eval(module.parameters())
            let expected = snapshot()
            let input = bank(1, width, phase: 4).reshaped(1, 1, width)
            MLX.eval(input)
            let before = Memory.activeMemory
            let first = try bridge.mix(input)
            MLX.eval(first.mixed, first.injection)
            let retained = Memory.activeMemory - before
            // The retained fused bank used to add 6.6 MB for ONE mixer. Allow
            // small outputs/compiled intermediates, but never another weight bank.
            XCTAssertLessThan(retained, 1 << 20, "permanent extra bytes after fusion: \(retained)")
            XCTAssertEqual(snapshot(), expected)
            let second = try bridge.mix(input)
            MLX.eval(second.mixed, second.injection)
            XCTAssertEqual(
                first.mixed.asData(access: .copy).data, second.mixed.asData(access: .copy).data)
            XCTAssertEqual(
                first.injection.asData(access: .copy).data,
                second.injection.asData(access: .copy).data)
        }
    }
}
