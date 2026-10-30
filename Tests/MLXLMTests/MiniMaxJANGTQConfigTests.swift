// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class MiniMaxJANGTQConfigTests: XCTestCase {
    func testUniformMxtqBitsDecode() throws {
        let config = try decodeMiniMaxConfig(extra: #"""
            "mxtq_bits": 2
            """#)

        XCTAssertEqual(config.mxtqBits, 2)
        XCTAssertNil(config.mxtqGateUpBits)
        XCTAssertNil(config.mxtqDownBits)
    }

    func testJANGTQKPerProjectionBitsDecode() throws {
        let config = try decodeMiniMaxConfig(extra: #"""
            "mxtq_bits": {
              "routed_expert": {
                "gate_proj": 2,
                "up_proj": 2,
                "down_proj": 4
              }
            }
            """#)

        XCTAssertEqual(config.mxtqBits, 2)
        XCTAssertEqual(config.mxtqGateUpBits, 2)
        XCTAssertEqual(config.mxtqDownBits, 4)
    }

    func testQuantizationRoutedExpertBitsFallback() throws {
        let config = try decodeMiniMaxConfig(extra: #"""
            "quantization": {
              "routed_expert_bits": 4
            }
            """#)

        XCTAssertEqual(config.mxtqBits, 4)
        XCTAssertNil(config.mxtqGateUpBits)
        XCTAssertNil(config.mxtqDownBits)
    }

    func testQuantizationNestedMxtqBitsFallback() throws {
        let config = try decodeMiniMaxConfig(extra: #"""
            "quantization": {
              "mxtq_bits": {
                "routed_expert": {
                  "gate_proj": 2,
                  "up_proj": 2,
                  "down_proj": 4
                }
              }
            }
            """#)

        XCTAssertEqual(config.mxtqBits, 2)
        XCTAssertEqual(config.mxtqGateUpBits, 2)
        XCTAssertEqual(config.mxtqDownBits, 4)
    }

    func testExplicitProjectionBitFieldsWinOverNestedBits() throws {
        let config = try decodeMiniMaxConfig(extra: #"""
            "mxtq_gate_up_bits": 4,
            "mxtq_down_bits": 4,
            "mxtq_bits": {
              "routed_expert": {
                "gate_proj": 2,
                "up_proj": 2,
                "down_proj": 2
              }
            }
            """#)

        XCTAssertEqual(config.mxtqBits, 2)
        XCTAssertEqual(config.mxtqGateUpBits, 4)
        XCTAssertEqual(config.mxtqDownBits, 4)
    }

    /// Resident experts are retired as a production mechanism:
    /// `JANGTQStreamingExperts.residentExpertsEnabled` is `false` whatever
    /// `MLXPRESS_RESIDENT_EXPERTS` says, so sanitize stacks the per-expert
    /// tensors into `switch_mlp` banks, as it does without the flag.
    func testLegacyResidentExpertsFlagStillStacksSwitchMLPBanks() throws {
        // Building the model and stacking the banks is MLX work, so it holds
        // the lock, as the target's other MLX tests do.
        try MLXMetalTestLock.withLock {
            let config = try decodeMiniMaxConfig(
                extra: #"""
                    "mxtq_bits": {
                      "routed_expert": {
                        "gate_proj": 2,
                        "up_proj": 2,
                        "down_proj": 4
                      }
                    }
                    """#,
                hiddenLayers: 1,
                numLocalExperts: 2)
            let model = MiniMaxJANGTQModel(config)
            var weights: [String: MLXArray] = [:]
            let prefix = "model.layers.0.block_sparse_moe"
            for expert in 0..<2 {
                for projection in ["w1", "w2", "w3"] {
                    weights["\(prefix).experts.\(expert).\(projection).tq_packed"] =
                        MLXArray.zeros([1, 1], dtype: .uint32)
                    weights["\(prefix).experts.\(expert).\(projection).tq_norms"] =
                        MLXArray.zeros([1], dtype: .float16)
                }
            }

            try withEnvironment("MLXPRESS_RESIDENT_EXPERTS", value: "1") {
                let sanitized = model.sanitize(weights: weights)

                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    XCTAssertEqual(
                        sanitized["\(prefix).switch_mlp.\(projection).tq_packed"]?.shape,
                        [2, 1, 1],
                        "sanitize must stack the experts' \(projection) into a switch_mlp bank")
                    XCTAssertEqual(
                        sanitized["\(prefix).switch_mlp.\(projection).tq_norms"]?.shape,
                        [2, 1],
                        "sanitize must stack the experts' \(projection) into a switch_mlp bank")
                }
                XCTAssertNil(
                    sanitized["\(prefix).experts.0.w1.tq_packed"],
                    "per-expert tensors should be moved out of the staging dictionary")
                XCTAssertTrue(
                    sanitized.keys.allSatisfy { !$0.contains(".block_sparse_moe.experts.") },
                    "sanitize should leave no per-expert staging keys for model.update")
            }
        }
    }

    private func decodeMiniMaxConfig(
        extra: String,
        hiddenLayers: Int = 62,
        numLocalExperts: Int = 256
    ) throws -> MiniMaxJANGTQConfiguration {
        let suffix = extra.trimmingCharacters(in: .whitespacesAndNewlines)
        let optionalExtra = suffix.isEmpty ? "" : ",\n\(suffix)"
        let json = """
            {
              "model_type": "minimax_m2",
              "hidden_size": 3072,
              "intermediate_size": 1536,
              "num_attention_heads": 32,
              "num_key_value_heads": 8,
              "max_position_embeddings": 262144,
              "num_experts_per_tok": 8,
              "num_local_experts": \(numLocalExperts),
              "shared_intermediate_size": 0,
              "num_hidden_layers": \(hiddenLayers),
              "rms_norm_eps": 0.000001,
              "rope_theta": 1000000.0,
              "rotary_dim": 96,
              "vocab_size": 200064
              \(optionalExtra)
            }
            """
        let data = Data(json.utf8)
        return try JSONDecoder().decode(MiniMaxJANGTQConfiguration.self, from: data)
    }
}

private func withEnvironment<T>(_ key: String, value: String, body: () throws -> T) rethrows -> T {
    let oldValue = getenv(key).map { String(cString: $0) }
    setenv(key, value, 1)
    defer {
        if let oldValue {
            setenv(key, oldValue, 1)
        } else {
            unsetenv(key)
        }
    }
    return try body()
}
