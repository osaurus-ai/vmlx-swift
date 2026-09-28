import Foundation
import XCTest

@testable import MLXLMCommon

final class JANGHFormatContractTests: XCTestCase {
    private func configuration() -> [String: Any] {
        var books: [String: [String: Any]] = [:]
        for bits in [2, 3, 4, 6, 8] {
            let count = 1 << bits
            let center = Double(count - 1) / 2
            let levels: [Double] = (0 ..< count).map { (Double($0) - center) * 0.25 }
            books[String(bits)] = ["alpha": 0.25, "beta": 0.0, "levels": levels]
        }
        return [
            "jangtq": [
                "version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
                "codebook_family": "odd-cubic", "rotation": "hadamard32", "codebooks": books,
            ],
            "quantization": [
                "mode": "affine", "bits": 8, "group_size": 64,
                "model.layers.0.mlp.switch_mlp.gate_proj": [
                    "mode": "jangtq2", "bits": 2, "rotation": "hadamard32",
                ],
                "model.layers.0.mlp.switch_mlp.up_proj": [
                    "mode": "jangtq2", "bits": 3, "rotation": "hadamard32",
                ],
                "model.layers.0.mlp.switch_mlp.down_proj": [
                    "mode": "jangtq2", "bits": 4, "rotation": "none",
                ],
                "model.layers.0.self_attn.q_proj": ["mode": "mxfp8", "bits": 8, "group_size": 32],
            ],
        ]
    }

    private func parse(_ value: [String: Any]) throws -> JANGHFormatContract {
        try JANGHFormatContract(configuration: JSONSerialization.data(withJSONObject: value))
    }

    func testMixedRolesAndTensorGeometry() throws {
        let contract = try parse(configuration())
        XCTAssertEqual(contract.projections.count, 3)
        XCTAssertEqual(contract.projections["model.layers.0.mlp.switch_mlp.up_proj"]?.bits, 3)
        for (role, bits) in [("gate_proj", 2), ("up_proj", 3), ("down_proj", 4)] {
            let module = "model.layers.0.mlp.switch_mlp." + role
            try contract.validateTensorHeaders(
                module: module, experts: 8, inputDimensions: 96,
                outputDimensions: 64, packedShape: [8, 64, 3 * bits], packedDType: "U32",
                scalesShape: [8, 64], scalesDType: "F16")
            XCTAssertThrowsError(
                try contract.validateTensorHeaders(
                    module: module, experts: 8,
                    inputDimensions: 96, outputDimensions: 64, packedShape: [8, 64, 3 * bits + 1],
                    packedDType: "U32", scalesShape: [8, 64], scalesDType: "F16"))
            XCTAssertThrowsError(
                try contract.validateTensorHeaders(
                    module: module, experts: 8,
                    inputDimensions: 96, outputDimensions: 64, packedShape: [8, 64, 3 * bits],
                    packedDType: "U32", scalesShape: [8, 64, 1], scalesDType: "F16"))
        }
    }

    func testUnknownFormatAndIncompleteExpertsFailClosed() throws {
        for (key, value) in [
            ("version", 3 as Any), ("packing", "msb-bitstream"),
            ("scale_dtype", "bfloat16"), ("codebook_family", "beta"), ("rotation", "seeded"),
        ] {
            var config = configuration()
            var header = config["jangtq"] as! [String: Any]
            header[key] = value
            config["jangtq"] = header
            XCTAssertThrowsError(try parse(config))
        }
        for (key, value) in [("rotation", "seeded" as Any), ("bits", 5), ("mode", "affine")] {
            var config = configuration()
            var modules = config["quantization"] as! [String: Any]
            let name = "model.layers.0.mlp.switch_mlp.up_proj"
            var projection = modules[name] as! [String: Any]
            projection[key] = value
            modules[name] = projection
            config["quantization"] = modules
            XCTAssertThrowsError(try parse(config))
        }
    }

    func testBundleCodebookCoefficientsAndSixBitPacking() throws {
        var config = configuration()
        var header = config["jangtq"] as! [String: Any]
        var books = header["codebooks"] as! [String: [String: Any]]
        // Odd-cubic coefficients from the actual JANGH metadata contract.
        for (bits, alpha, beta) in [(2, 0.893, 0.05065), (3, 0.47125, 0.0116), (4, 0.2405, 0.0021)]
        {
            books[String(bits)] = [
                "alpha": alpha, "beta": beta,
                "levels": (0 ..< (1 << bits)).map { index -> Double in
                    let u = Double(index) - Double((1 << bits) - 1) / 2
                    return Double(Float(u * (alpha + beta * u * u)))
                },
            ]
        }
        header["codebooks"] = books
        config["jangtq"] = header
        var modules = config["quantization"] as! [String: Any]
        let name = "model.layers.0.mlp.switch_mlp.down_proj"
        modules[name] = ["mode": "jangtq2", "bits": 6, "rotation": "hadamard32"]
        config["quantization"] = modules
        let contract = try parse(config)
        try contract.validateTensorHeaders(
            module: name, experts: 8, inputDimensions: 96,
            outputDimensions: 64, packedShape: [8, 64, 18], packedDType: "U32",
            scalesShape: [8, 64], scalesDType: "F16")
        // Padding a non-block-aligned K is not part of this format contract.
        XCTAssertThrowsError(
            try contract.validateTensorHeaders(
                module: name, experts: 8,
                inputDimensions: 95, outputDimensions: 64, packedShape: [8, 64, 18],
                packedDType: "U32", scalesShape: [8, 64], scalesDType: "F16"))
        XCTAssertThrowsError(
            try contract.validateTensorHeaders(
                module: name, experts: 8,
                inputDimensions: Int.max - 31, outputDimensions: 64, packedShape: [8, 64, 18],
                packedDType: "U32", scalesShape: [8, 64], scalesDType: "F16"))
    }

    func testFiniteDoubleCoefficientMustFitMetalFloat() throws {
        var config = configuration()
        var header = config["jangtq"] as! [String: Any]
        var books = header["codebooks"] as! [String: [String: Any]]
        books["2"] = [
            "alpha": 1e100, "beta": 0.0,
            "levels": [-1.5e100, -0.5e100, 0.5e100, 1.5e100],
        ]
        header["codebooks"] = books
        config["jangtq"] = header
        XCTAssertThrowsError(try parse(config))
    }

    func testCodebookDisagreementFailsClosed() throws {
        var config = configuration()
        var header = config["jangtq"] as! [String: Any]
        var books = header["codebooks"] as! [String: [String: Any]]
        books["2"]?["alpha"] = 0.5
        header["codebooks"] = books
        config["jangtq"] = header
        XCTAssertThrowsError(try parse(config))
    }
}
