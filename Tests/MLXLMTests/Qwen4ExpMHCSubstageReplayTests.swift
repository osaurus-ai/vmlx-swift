#if DEBUG
import CryptoKit
import Foundation
import MLX
import Metal
import XCTest
@testable import MLXLMCommon
@testable import MLXVLM

/// Private, opt-in saved-operand diagnosis. No full model or generation.
final class Qwen4ExpMHCSubstageReplayTests: XCTestCase {
    private static func require(_ condition: Bool, _ message: String) throws {
        guard condition else { XCTFail(message); throw NSError(domain: message, code: 1) }
    }
    private static func hash(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    private static func bytes(_ array: MLXArray) -> Data { array.asData(access: .copy).data }
    private static func exact(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.dtype == b.dtype && a.shape == b.shape && bytes(a) == bytes(b)
    }
    private static func difference(_ a: MLXArray, _ b: MLXArray) throws -> [String: Any] {
        try require(a.dtype == .bfloat16 && b.dtype == .bfloat16 && a.shape == b.shape,
            "Compare actual matching BF16 storage")
        let lhs = bytes(a), rhs = bytes(b)
        try require(lhs.count == a.size * 2 && rhs.count == lhs.count, "BF16 byte count")
        let x = a.asArray(Float.self), y = b.asArray(Float.self)
        var indices: [Int] = [], maximum: Float = 0
        for index in x.indices {
            if lhs[2 * index] != rhs[2 * index] || lhs[2 * index + 1] != rhs[2 * index + 1] {
                indices.append(index)
            }
            maximum = max(maximum, abs(x[index] - y[index]))
        }
        return ["shape": a.shape, "dtype": "BF16", "exact_storage": lhs == rhs,
            "unequal_indices": indices, "unequal_values": indices.count, "max_abs": maximum,
            "finite": x.allSatisfy(\.isFinite) && y.allSatisfy(\.isFinite),
            "a_sha256": hash(lhs), "b_sha256": hash(rhs)]
    }
    func testSavedLayerZeroEagerSubstageDiscriminator() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["VMLX_MHC_SUBSTAGE_REPLAY"] == "1" else {
            throw XCTSkip("Private saved-operand replay requires explicit opt-in")
        }
        try Self.require(env["VMLX_QWEN4_EXP_COMPILE_MHC"] == "0"
            && env["VMLX_QWEN4_EXP_FUSE_DECODE_INPUTS"] == "0"
            && env["VMLX_QWEN4_EXP_COMPILE_MHC_S2"] == "0"
            && env["VMLX_QWEN4_HC_COMBINE_NORM"] == "0",
            "Fresh process with immutable eager-separate policy")
        let fixture = URL(fileURLWithPath: try XCTUnwrap(env["VMLX_MHC_REPLAY_FIXTURE"]), isDirectory: true)
        let output = URL(fileURLWithPath: try XCTUnwrap(env["VMLX_MHC_REPLAY_OUTPUT"]), isDirectory: true)
        try Self.require(!FileManager.default.fileExists(atPath: output.path), "Fresh private output")
        let manifestData = try Data(contentsOf: fixture.appendingPathComponent("FIXTURE-MANIFEST.json"))
        try Self.require(Self.hash(manifestData) == env["VMLX_MHC_REPLAY_FIXTURE_SHA256"],
            "Root-sealed fixture manifest")
        let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with: manifestData) as? [String: Any])
        let files = try XCTUnwrap(manifest["file_sha256"] as? [String: String])
        try Self.require(Set(files.keys) == ["config.json", "weights.safetensors", "inputs.safetensors"],
            "Only exact bounded fixture files")
        for (name, expected) in files {
            try Self.require(Self.hash(try Data(contentsOf: fixture.appendingPathComponent(name))) == expected,
                "Fixture bytes changed: \(name)")
        }
        let configData = try Data(contentsOf: fixture.appendingPathComponent("config.json"))
        try Self.require(Self.hash(configData) == "e8637fd9b09fbb590a787ea16d7dcdbbe5f90256bb82d0a097a416a4470a61db",
            "Actual captured bundle config")
        // Same metadata-only normalizer as JANGHModelPreparation.init and the real
        // VLMModelFactory/Qwen4ExpJANGHPreparation path. Do not map routed banks or
        // rewrite the sealed raw bundle config for this dense-HC-only replay.
        var rawCustomRejected = false
        do {
            _ = try JSONDecoder.json5().decode(Qwen4ExpConfiguration.self, from: configData)
        } catch DecodingError.dataCorrupted(let context) {
            rawCustomRejected = context.debugDescription.contains("Unsupported quantization mode 'jangtq2'")
        }
        try Self.require(rawCustomRejected, "Raw custom modes still fail strict ordinary decoding")
        let partition = try JANGHConfigurationPartition(configuration: configData, sidecar: nil)
        let expectedCustom = Set((0..<48).flatMap { layer in
            ["gate_proj", "up_proj", "down_proj"].map {
                "model.layers.\(layer).mlp.switch_mlp." + $0
            }
        })
        try Self.require(partition.modelType == "qwen4_exp" && partition.customModules == expectedCustom,
            "Actual production partition covers exactly the original 144 routed projections")
        let config = try JSONDecoder.json5().decode(
            Qwen4ExpConfiguration.self, from: partition.ordinaryConfiguration)
        try Self.require(config.base.textConfiguration.hiddenLayers == 48
            && config.jangMetadata?.normConvention == "runtime_plus1_applied",
            "Captured architecture and already-applied norm convention retained")
        // The root must own the process-wide model/Metal lock before invoking this test.
        // No full model or installed multi-GB shard is loaded by MLX here.
        let weights = try loadArrays(url: fixture.appendingPathComponent("weights.safetensors"))
        let inputs = try loadArrays(url: fixture.appendingPathComponent("inputs.safetensors"))
        let input = try XCTUnwrap(inputs["attentionCombined"])
        let gold1 = try XCTUnwrap(inputs["moeInputS1"]), gold4 = try XCTUnwrap(inputs["moeInputS4"])
        try Self.require(input.shape == [1, 1, 10240] && input.dtype == .bfloat16
            && gold1.shape == [1, 1, 2560] && gold4.shape == gold1.shape
            && gold1.dtype == .bfloat16 && gold4.dtype == .bfloat16, "Actual saved boundary geometry")
        try Self.require(Self.hash(Self.bytes(input)) == "bde6316a50176894f7e63b8667b97951c3d0e4cbf4a685c68842ed5ae85d797e"
            && Self.hash(Self.bytes(gold1)) == "896c3eb17bce661494bb0361750a635ec426d16369a493be0473d2fd037f186f"
            && Self.hash(Self.bytes(gold4)) == "20736f0a77607ffc84147502843236ad0520ad7822a8691d67947689c5245f9c",
            "Original unchanged endpoints")
        let bridge = try Qwen4ExpMHCSubstageBridge(config: config, parameters: weights)
        try Self.require(Set(bridge.parameters.keys) == Set(weights.keys), "Actual module keys")
        for (name, value) in bridge.parameters {
            try Self.require(Self.exact(value, try XCTUnwrap(weights[name])), "Actual installed weight: \(name)")
        }
        let beforeInput = Self.bytes(input), beforeWeights = weights.mapValues(Self.bytes)
        let block = tiled(input, repetitions: [1, 4, 1])
        eval(input, block)
        for row in 0..<4 {
            try Self.require(Self.exact(block[0..., row..<(row + 1), 0...], input), "Identical S4 row")
        }
        let s1 = try bridge.mix(input)
        let s4 = try bridge.mix(block)
        eval(s1.mixed, s1.injection, s4.mixed, s4.injection)
        try Self.require(Self.exact(s1.mixed, gold1)
            && Self.exact(s4.mixed[0..., 0..<1, 0...], gold4),
            "Both original 123b endpoints must reproduce before decomposition")
        let stages1 = bridge.stages(input), stages4 = bridge.stages(block)
        var reportStages: [[String: Any]] = []
        var saved: [String: MLXArray] = [:]
        var firstDifference: String?
        for ((name, a), (otherName, b)) in zip(stages1, stages4) {
            try Self.require(name == otherName, "Stable stage order")
            eval(a, b)
            let row = b[0..., 0..<1, 0...]
            let comparison = try Self.difference(a, row)
            if firstDifference == nil && !name.hasPrefix("injection") && !Self.exact(a, row) {
                firstDifference = name
            }
            try Self.require(comparison["finite"] as? Bool == true, "Finite stage: \(name)")
            for i in 1..<4 {
                try Self.require(Self.exact(row, b[0..., i..<(i + 1), 0...]),
                    "Identical repeated row at stage \(name)")
            }
            reportStages.append(["stage": name, "s1_vs_s4_row0": comparison])
            saved["s1_" + name] = a
            saved["s4_" + name] = b
        }
        for (prefix, result) in [("s1_", s1), ("s4_", s4)] {
            try Self.require(Self.exact(try XCTUnwrap(saved[prefix + "mixed"]), result.mixed)
                && Self.exact(try XCTUnwrap(saved[prefix + "injection"]), result.injection),
                "Decomposition must match BOTH actual outputs: \(prefix)")
            try Self.require(Self.exact(try XCTUnwrap(saved[prefix + "norm_product"]),
                try XCTUnwrap(saved[prefix + "normalized"])), "RMS substage calibration")
        }
        let repeated = try bridge.mix(input)
        eval(repeated.mixed, repeated.injection)
        try Self.require(Self.exact(repeated.mixed, s1.mixed)
            && Self.exact(repeated.injection, s1.injection), "Warm S1 unchanged")
        try Self.require(Self.bytes(input) == beforeInput && weights.mapValues(Self.bytes) == beforeWeights,
            "Input and parameters preserved")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
        try save(arrays: saved, url: output.appendingPathComponent("substage-operands.safetensors"))
        let report: [String: Any] = [
            "complete": true, "mode": "eager_separate", "strict_endpoint_calibration": true,
            "fixture_manifest_sha256": Self.hash(manifestData),
            "first_mixed_path_difference": firstDifference ?? "none",
            "stages": reportStages, "environment": env.filter {
                ["VMLX_MHC_SUBSTAGE_REPLAY", "VMLX_QWEN4_EXP_COMPILE_MHC",
                 "VMLX_QWEN4_EXP_FUSE_DECODE_INPUTS", "VMLX_QWEN4_EXP_COMPILE_MHC_S2",
                 "VMLX_QWEN4_HC_COMBINE_NORM"].contains($0.key)
            },
            "metal_device_name": MTLCreateSystemDefaultDevice()?.name ?? "unavailable",
            "actual_runtime_private_mix_called": true, "full_model_loaded": false,
            "routed_banks_constructed": false, "input_preserved": true, "parameters_preserved": true,
            "tokens_per_second": NSNull(), "no_generation": true, "speed_claim": false,
            "whole_model_equivalence_claim": false, "accepted_cache_correctness_claim": false,
        ]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("substage-replay.json"), options: .withoutOverwriting)
        print("[MHCSubstageReplay] first_difference=\(firstDifference ?? "none") endpoint_calibration=PASS tokens_per_second=NA no_generation=1")
    }
}
#endif
