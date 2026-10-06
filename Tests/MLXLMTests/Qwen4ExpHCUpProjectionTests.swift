import CryptoKit
import Foundation
import MLX
import XCTest
@testable import MLXVLM

final class Qwen4ExpHCUpProjectionTests: XCTestCase {
    private func hash(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    private func exact(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.shape == b.shape && a.dtype == b.dtype
            && a.asData(access: .copy).data == b.asData(access: .copy).data
    }

    func testUnsupportedShapesAndS1KeepOriginalPath() {
        let weight = MLXArray.zeros([10240, 320], dtype: .bfloat16)
        for shape in [[1, 1, 320], [1, 9, 320], [2, 2, 320], [4, 320]] {
            XCTAssertNil(Qwen4ExpHCUpProjection.project(
                MLXArray.zeros(shape, dtype: .bfloat16), weight: weight))
        }
        XCTAssertNil(Qwen4ExpHCUpProjection.project(
            MLXArray.zeros([1, 4, 320], dtype: .float16), weight: weight.asType(.float16)))
        XCTAssertNil(Qwen4ExpHCUpProjection.project(
            MLXArray.zeros([1, 4, 320], dtype: .bfloat16),
            weight: MLXArray.zeros([2560, 320], dtype: .bfloat16)))
    }

    /// Root supplies bounded actual banks from several HC layers. The saved
    /// activation is real; perturbations are explicitly synthetic controls,
    /// not claimed to be natural per-layer activations or full-target proof.
    func testActualHCBanksAgainstOriginalS1WithVariedInputs() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["VMLX_HC_UP_BANK_REPLAY"] == "1" else {
            throw XCTSkip("Requires sealed private multi-bank fixture")
        }
        let dir = URL(fileURLWithPath: try XCTUnwrap(env["VMLX_HC_UP_BANK_FIXTURE"]))
        let manifestBytes = try Data(contentsOf: dir.appendingPathComponent("BANK-MANIFEST.json"))
        XCTAssertEqual(hash(manifestBytes), env["VMLX_HC_UP_BANK_MANIFEST_SHA256"])
        guard hash(manifestBytes) == env["VMLX_HC_UP_BANK_MANIFEST_SHA256"] else { return }
        let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with: manifestBytes) as? [String: Any])
        let bankURL = dir.appendingPathComponent("banks.safetensors")
        let bankBytes = try Data(contentsOf: bankURL)
        XCTAssertEqual(hash(bankBytes), manifest["banks_sha256"] as? String)
        guard hash(bankBytes) == manifest["banks_sha256"] as? String else { return }
        let inputURL = URL(fileURLWithPath: try XCTUnwrap(env["VMLX_HC_UP_OPERANDS"]))
        let inputBytes = try Data(contentsOf: inputURL)
        XCTAssertEqual(hash(inputBytes), env["VMLX_HC_UP_OPERANDS_SHA256"])
        guard hash(inputBytes) == env["VMLX_HC_UP_OPERANDS_SHA256"] else { return }
        let banks = try loadArrays(url: bankURL)
        let inputs = try loadArrays(url: inputURL)
        let actual = try XCTUnwrap(inputs["s1_silu"])
        XCTAssertEqual(actual.shape, [1, 1, 320]); XCTAssertEqual(actual.dtype, .bfloat16)
        XCTAssertGreaterThanOrEqual(banks.count, 8, "Four actual layers, both HC roles")
        guard banks.count >= 8, actual.shape == [1, 1, 320], actual.dtype == .bfloat16 else { return }
        let values = actual.asArray(Float.self)
        var variants: [(String, MLXArray)] = [("saved_actual_layer0_silu", actual)]
        for shift in [1, 17, 113] {
            let rows = (0..<8).flatMap { row in
                (0..<320).map { column -> Float in
                    let scale: Float = row % 2 == 0 ? 0.5 : -1.25
                    return values[(column + shift * (row + 1)) % 320] * scale
                }
            }
            variants.append(("synthetic_shift_\(shift)", MLXArray(rows).reshaped(1, 8, 320).asType(.bfloat16)))
        }
        var results: [[String: Any]] = []
        var failures: [String: MLXArray] = [:]
        for name in banks.keys.sorted() {
            let weight = try XCTUnwrap(banks[name])
            XCTAssertEqual(weight.shape, [10240, 320]); XCTAssertEqual(weight.dtype, .bfloat16)
            for (kind, slab) in variants {
                for count in [1, 2, 3, 4, 8] {
                    let x = slab.dim(1) == 1 ? tiled(slab, repetitions: [1, count, 1])
                        : slab[0..., 0..<count, 0...]
                    let oracle = concatenated((0..<count).map { row in
                        matmul(x[0..., row..<(row + 1), 0...], weight.transposed())
                    }, axis: 1)
                    let candidate = Qwen4ExpHCUpProjection.project(x, weight: weight)
                        ?? matmul(x, weight.transposed())
                    eval(oracle, candidate)
                    let matches = exact(candidate, oracle)
                    let key = "\(name)_\(kind)_s\(count)"
                    results.append(["case": key, "exact_to_original_s1": matches])
                    if !matches {
                        failures[key + "_oracle"] = oracle
                        failures[key + "_candidate"] = candidate
                    }
                }
            }
        }
        let output = URL(fileURLWithPath: try XCTUnwrap(env["VMLX_HC_UP_BANK_OUTPUT"]), isDirectory: true)
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
        guard !FileManager.default.fileExists(atPath: output.path) else { return }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
        if !failures.isEmpty {
            try save(arrays: failures, url: output.appendingPathComponent("mismatches.safetensors"))
        }
        let report: [String: Any] = ["cases": results, "all_exact": failures.isEmpty,
            "actual_bank_count": banks.count, "input_manifest_sha256": hash(manifestBytes),
            "actual_input_capture": "layer0_only", "varied_controls": "synthetic_labeled",
            "full_model_loaded": false, "whole_target_parity": false,
            "cache_qualification": false, "speed_claim": false, "tokens_per_second": NSNull()]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("multi-bank-up.json"), options: .withoutOverwriting)
        XCTAssertTrue(failures.isEmpty, "Exact original S1 oracle required for every admitted case")
    }
}
