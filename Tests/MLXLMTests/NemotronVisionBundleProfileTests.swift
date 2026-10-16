import Foundation
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class NemotronVisionBundleProfileTests: XCTestCase {
    private let weight = "vision_projector.vision_final_layernorm.weight"
    private let bias = "vision_projector.vision_final_layernorm.bias"

    private func withBundle(_ body: (URL) throws -> Void) throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("nemotron-vision-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try body(directory)
    }

    private func writeShard(_ names: [String], at url: URL, truncated: Bool = false) throws {
        var header: [String: Any] = [:]
        for (index, name) in names.enumerated() {
            header[name] = ["dtype": "F32", "shape": [1], "data_offsets": [index * 4, (index + 1) * 4]]
        }
        let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var count = UInt64(json.count).littleEndian
        var data = withUnsafeBytes(of: &count) { Data($0) }
        data.append(json)
        if !truncated { data.append(Data(repeating: 0, count: names.count * 4)) }
        try data.write(to: url)
    }

    func testUnindexedSingleShardFindsPhysicalFinalNorm() throws {
        try withBundle { directory in
            try writeShard([weight, bias], at: directory.appendingPathComponent("model.safetensors"))
            let names = Set(try MTPBundleInspector.loadableTensorNames(from: directory))
            XCTAssertTrue(try nemotronVisionFinalLayerNormPresent(tensorNames: names, nextTokenLayers: 1))
        }
    }

    func testIncompleteIndexStillFindsNormInItsLoadedShard() throws {
        try withBundle { directory in
            let other = "backbone.norm_f.weight"
            try writeShard([other, weight, bias], at: directory.appendingPathComponent("model.safetensors"))
            let index = try JSONSerialization.data(withJSONObject: ["weight_map": [other: "model.safetensors"]])
            try index.write(to: directory.appendingPathComponent("model.safetensors.index.json"))
            let names = Set(try MTPBundleInspector.loadableTensorNames(from: directory))
            XCTAssertTrue(try nemotronVisionFinalLayerNormPresent(tensorNames: names, nextTokenLayers: 1))
        }
    }

    func testIndexCannotInventMissingOrTruncatedNorm() throws {
        for truncated in [false, true] {
            try withBundle { directory in
                if truncated {
                    try writeShard([weight, bias], at: directory.appendingPathComponent("model.safetensors"), truncated: true)
                }
                let index = try JSONSerialization.data(withJSONObject: [
                    "weight_map": [weight: "model.safetensors", bias: "model.safetensors"]])
                try index.write(to: directory.appendingPathComponent("model.safetensors.index.json"))
                let names = Set(try MTPBundleInspector.loadableTensorNames(from: directory))
                XCTAssertThrowsError(try nemotronVisionFinalLayerNormPresent(tensorNames: names, nextTokenLayers: 1))
            }
        }
    }

    func testArchitectureAndBothNormTensorsMustAgree() throws {
        XCTAssertFalse(try nemotronVisionFinalLayerNormPresent(tensorNames: [], nextTokenLayers: 0))
        let incomplete: [Set<String>] = [[], [weight], [bias]]
        for names in incomplete {
            XCTAssertThrowsError(try nemotronVisionFinalLayerNormPresent(tensorNames: names, nextTokenLayers: 1))
        }
        XCTAssertThrowsError(try nemotronVisionFinalLayerNormPresent(tensorNames: [weight, bias], nextTokenLayers: 0))
    }
}
