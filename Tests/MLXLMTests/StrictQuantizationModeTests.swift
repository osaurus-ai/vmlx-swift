import Foundation
import MLXLMCommon
import XCTest

final class StrictQuantizationModeTests: XCTestCase {
    private func decode(mode: String?, group: Int?, nested: Bool) throws -> BaseConfiguration {
        var entry: [String: Any] = ["bits": 8]
        if let mode { entry["mode"] = mode }
        if let group { entry["group_size"] = group }
        var plan: [String: Any] = ["group_size": 64, "bits": 4, "mode": "affine"]
        if nested { plan["per_tensor"] = ["model.projection": entry] }
        else { plan["model.projection"] = entry }
        return try JSONDecoder().decode(
            BaseConfiguration.self,
            from: JSONSerialization.data(withJSONObject: ["model_type": "test", "quantization": plan]))
    }

    func testUnknownExplicitModesNeverBecomeAffine() throws {
        for nested in [false, true] {
            for group: Int? in [nil, 32] {
                for mode in ["jangtq2", "unsupported_codec"] {
                    XCTAssertThrowsError(try decode(mode: mode, group: group, nested: nested)) { error in
                        guard case DecodingError.dataCorrupted(let context) = error else {
                            return XCTFail("Expected explicit mode rejection, got \(error)")
                        }
                        XCTAssertEqual(context.codingPath.last?.stringValue, "mode")
                        XCTAssertTrue(context.codingPath.contains { $0.stringValue == "model.projection" })
                        XCTAssertTrue(context.debugDescription.contains(mode))
                    }
                }
            }
        }
    }

    func testSupportedModesPreserveInheritedAndExplicitGeometry() throws {
        for nested in [false, true] {
            for group: Int? in [nil, 32] {
                for (mode, expected) in [("affine", "affine"), ("MXFP8", "mxfp8"),
                                         ("affine+mxtq", "affine"), ("affine_mxtq", "affine")] {
                    let config = try decode(mode: mode, group: group, nested: nested)
                    let q = try XCTUnwrap(config.perLayerQuantization?.quantization(layer: "model.projection"))
                    XCTAssertEqual(q.mode.rawValue, expected)
                    XCTAssertEqual(q.groupSize, group ?? 64)
                    XCTAssertEqual(q.bits, 8)
                }
            }
        }
    }

    func testAbsentModeRetainsAffineDefault() throws {
        for nested in [false, true] {
            let config = try decode(mode: nil, group: nil, nested: nested)
            let q = try XCTUnwrap(config.perLayerQuantization?.quantization(layer: "model.projection"))
            XCTAssertEqual(q.mode.rawValue, "affine")
            XCTAssertEqual(q.groupSize, 64)
        }
    }
}
