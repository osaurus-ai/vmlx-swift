// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
@testable import MLXLMCommon
@testable import MLXLLM
import XCTest

final class MiniMaxJANGTQResidentExpertTests: XCTestCase {
    /// Streaming experts are retired as a production mechanism, and with them
    /// the `JANGPRESS_STREAMING_EXPERTS` alias of the flag:
    /// `JANGTQStreamingExperts.explicitStreamingEnabled` is `false` whatever
    /// the environment says, so the legacy alias must leave streaming off.
    func testLegacyJangPressStreamingAliasLeavesStreamingDisabled() {
        withFocusedEnvironment("MLXPRESS_STREAMING_EXPERTS", value: nil) {
            withFocusedEnvironment("JANGPRESS_STREAMING_EXPERTS", value: "1") {
                XCTAssertFalse(JANGTQStreamingExperts.isEnabled)
            }
        }
    }

    func testConfiguredModelDirectoryDoesNotDependOnProcessEnvironment() {
        let envDirectory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("vmlx-env-\(UUID().uuidString)")
        let configuredDirectory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("vmlx-configured-\(UUID().uuidString)")

        withFocusedEnvironment("MLXPRESS_MODEL_DIR", value: envDirectory.path) {
            JANGTQStreamingExperts.configureModelDirectory(configuredDirectory)
            defer { JANGTQStreamingExperts.clearConfiguredModelDirectory() }

            XCTAssertEqual(
                JANGTQStreamingExperts.configuredModelDirectoryForDiagnostics(),
                configuredDirectory.resolvingSymlinksInPath())
        }
    }

    /// Resident experts are retired as a production mechanism, like streaming
    /// experts: `JANGTQStreamingExperts.residentExpertsEnabled` is `false`
    /// whatever `MLXPRESS_RESIDENT_EXPERTS` says, so sanitize stacks the
    /// per-expert tensors into `switch_mlp` banks, as it does without the flag.
    func testLegacyResidentExpertsFlagStillStacksSwitchMLPBanks() throws {
        // Building the model and stacking the banks is MLX work, so it holds
        // the lock, as the bundle's other MLX tests do.
        try FocusedMLXTestSupport.withLock {
            let config = try decodeMiniMaxConfig(
                hiddenLayers: 1,
                numLocalExperts: 2)
            let model = MiniMaxJANGTQModel(config)
            let prefix = "model.layers.0.block_sparse_moe"
            var weights: [String: MLXArray] = [:]
            for expert in 0..<2 {
                for projection in ["w1", "w2", "w3"] {
                    weights["\(prefix).experts.\(expert).\(projection).tq_packed"] =
                        MLXArray.zeros([1, 1], dtype: .uint32)
                    weights["\(prefix).experts.\(expert).\(projection).tq_norms"] =
                        MLXArray.zeros([1], dtype: .float16)
                }
            }

            withFocusedEnvironment("MLXPRESS_RESIDENT_EXPERTS", value: "1") {
                let sanitized = model.sanitize(weights: weights)

                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    XCTAssertEqual(
                        sanitized["\(prefix).switch_mlp.\(projection).tq_packed"]?.shape,
                        [2, 1, 1])
                    XCTAssertEqual(
                        sanitized["\(prefix).switch_mlp.\(projection).tq_norms"]?.shape,
                        [2, 1])
                }
                XCTAssertTrue(
                    sanitized.keys.allSatisfy { !$0.contains(".block_sparse_moe.experts.") },
                    "sanitize must move the per-expert tensors into the stacked banks")
            }
        }
    }

    private func decodeMiniMaxConfig(
        hiddenLayers: Int,
        numLocalExperts: Int
    ) throws -> MiniMaxJANGTQConfiguration {
        let json = """
            {
              "model_type": "minimax_m2",
              "hidden_size": 3072,
              "intermediate_size": 1536,
              "num_attention_heads": 32,
              "num_key_value_heads": 8,
              "max_position_embeddings": 262144,
              "num_experts_per_tok": 2,
              "num_local_experts": \(numLocalExperts),
              "shared_intermediate_size": 0,
              "num_hidden_layers": \(hiddenLayers),
              "rms_norm_eps": 0.000001,
              "rope_theta": 1000000.0,
              "rotary_dim": 96,
              "vocab_size": 200064,
              "mxtq_bits": {
                "routed_expert": {
                  "gate_proj": 2,
                  "up_proj": 2,
                  "down_proj": 4
                }
              }
            }
            """
        return try JSONDecoder().decode(MiniMaxJANGTQConfiguration.self, from: Data(json.utf8))
    }
}

private func withFocusedEnvironment<T>(
    _ key: String,
    value: String?,
    body: () throws -> T
) rethrows -> T {
    let oldValue = getenv(key).map { String(cString: $0) }
    if let value {
        setenv(key, value, 1)
    } else {
        unsetenv(key)
    }
    defer {
        if let oldValue {
            setenv(key, oldValue, 1)
        } else {
            unsetenv(key)
        }
    }
    return try body()
}
