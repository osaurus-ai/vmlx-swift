#if DEBUG
import CryptoKit
import Foundation
import MLX
import Metal
import XCTest
@testable import MLXLMCommon
@testable import MLXVLM

/// Corrected actual private mix versus immutable original S1, with the old
/// S4 graph retained as a negative control. No model or generation.
final class Qwen4ExpMHCCorrectedMixTests: XCTestCase {
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
    func testCorrectedActualMixPreservesOriginalS1AndSerialRows() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["VMLX_MHC_CORRECTED_REPLAY"] == "1" else {
            throw XCTSkip("Private corrected actual-mix replay requires opt-in")
        }
        let mode = try XCTUnwrap(env["VMLX_MHC_CORRECTED_MODE"])
        try Self.require(mode == "eager" || mode == "native", "Named fresh-process arm")
        let value = mode == "native" ? "1" : "0"
        try Self.require(env["VMLX_QWEN4_EXP_COMPILE_MHC"] == value
            && env["VMLX_QWEN4_EXP_FUSE_DECODE_INPUTS"] == value
            && env["VMLX_QWEN4_EXP_COMPILE_MHC_S2"] == "0"
            && env["VMLX_QWEN4_HC_COMBINE_NORM"] == "0",
            "Frozen native/eager arm; no opt-in combine norm or S2 compile")
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
        let ordinaryBefore = try bridge.mix(block)
        eval(ordinaryBefore.mixed, ordinaryBefore.injection)
        let s4 = try FlashVerificationScope.withVerification(inputShape: [1, 4]) {
            try bridge.mix(block)
        }
        eval(s1.mixed, s1.injection, s4.mixed, s4.injection)
        let ordinaryAfter = try bridge.mix(block)
        eval(ordinaryAfter.mixed, ordinaryAfter.injection)
        try Self.require(Self.exact(s1.mixed, gold1), "Current actual S1 equals immutable original S1")
        let oldStages1 = Dictionary(uniqueKeysWithValues: bridge.stages(input))
        let oldStages4 = Dictionary(uniqueKeysWithValues: bridge.stages(block))
        let old1 = try XCTUnwrap(oldStages1["mixed"])
        let old4 = try XCTUnwrap(oldStages4["mixed"])
        let oldInjection = try XCTUnwrap(oldStages4["injection"])
        eval(old1, old4, oldInjection)
        try Self.require(Self.exact(ordinaryBefore.mixed, old4)
            && Self.exact(ordinaryBefore.injection, oldInjection),
            "Unscoped short prefill preserves original eager arithmetic")
        try Self.require(Self.exact(ordinaryAfter.mixed, old4)
            && Self.exact(ordinaryAfter.injection, oldInjection),
            "Verifier scope cannot leak into subsequent short prefill")
        try Self.require(!Self.exact(ordinaryAfter.mixed, s4.mixed),
            "Saved negative control distinguishes prefill and exact verifier dispatch")
        // stages() deliberately retains the ORIGINAL actual module up Linear,
        // including old matmul dispatch, sigmoid/product/mean. It is a control,
        // never the corrected production mix implementation.
        try Self.require(Self.exact(old1, gold1)
            && Self.exact(old4[0..., 0..<1, 0...], gold4),
            "Old eager graph reconstructs BOTH immutable endpoints")
        try Self.require(!Self.exact(gold1, gold4), "Original S4 negative control remains distinct")
        let oldDifference = try Self.difference(gold1, gold4)
        try Self.require(oldDifference["unequal_indices"] as? [Int] == [178],
                         "Original single-word negative control")
        var serialMixed: [MLXArray] = []
        var serialInjection: [MLXArray] = []
        for row in 0..<4 {
            let serial = try bridge.mix(block[0..., row..<(row + 1), 0...])
            eval(serial.mixed, serial.injection)
            try Self.require(Self.exact(serial.mixed, gold1), "Serial actual S1 row unchanged")
            try Self.require(Self.exact(s4.mixed[0..., row..<(row + 1), 0...], serial.mixed),
                             "Corrected actual S4 mixed row equals serial actual S1")
            try Self.require(Self.exact(s4.injection[0..., row..<(row + 1), 0...], serial.injection),
                             "Corrected actual S4 injection row equals serial actual S1")
            serialMixed.append(serial.mixed)
            serialInjection.append(serial.injection)
        }
        try Self.require(Self.exact(s4.injection, oldInjection), "Injection unchanged from original graph")
        let repeatS1 = try bridge.mix(input)
        eval(repeatS1.mixed, repeatS1.injection)
        try Self.require(Self.exact(repeatS1.mixed, s1.mixed)
            && Self.exact(repeatS1.injection, s1.injection), "Warm actual S1 stable")
        try Self.require(Self.bytes(input) == beforeInput && weights.mapValues(Self.bytes) == beforeWeights,
                         "Input and actual parameter bytes unchanged")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
        try save(arrays: ["currentS1Mixed": s1.mixed, "currentS4Mixed": s4.mixed,
            "currentS1Injection": s1.injection, "currentS4Injection": s4.injection,
            "serialS1Mixed": concatenated(serialMixed, axis: 1),
            "serialS1Injection": concatenated(serialInjection, axis: 1),
            "originalGraphS4Mixed": old4, "immutableOriginalS4": gold4],
            url: output.appendingPathComponent("corrected-actual-mix.safetensors"))
        let report: [String: Any] = [
            "complete": true, "mode": mode, "actual_runtime_private_mix_called": true,
            "fixture_manifest_sha256": Self.hash(manifestData),
            "s1_equals_original_s1": true, "s4_equals_serial_actual_s1": true,
            "old_s4_negative_control_reconstructed": true,
            "original_s1_vs_original_s4": oldDifference,
            "injection_unchanged": true, "parameters_preserved": true, "input_preserved": true,
            "metal_device_name": MTLCreateSystemDefaultDevice()?.name ?? "unavailable",
            "full_model_loaded": false, "whole_model_equivalence_claim": false,
            "accepted_cache_correctness_claim": false, "speed_claim": false,
            "tokens_per_second": NSNull(), "no_generation": true,
        ]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("corrected-mix.json"), options: .withoutOverwriting)
    }
}
#endif
