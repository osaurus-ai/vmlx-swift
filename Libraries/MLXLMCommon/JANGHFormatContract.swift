import Foundation

/// Validates JANGH metadata without enabling an executable model path.
/// Keep this separate from affine quantization: identical bit packing does not
/// imply identical weight decoding or activation rotation.
struct JANGHFormatContract: Sendable {
    enum ValidationError: Error, Equatable {
        case invalid(String)
    }

    enum Rotation: String, Decodable, Sendable {
        case none
        case hadamard32
    }

    struct Codebook: Decodable, Sendable {
        let alpha: Double
        let beta: Double
        let levels: [Double]
    }

    struct Projection: Decodable, Sendable {
        let mode: String
        let bits: Int
        let rotation: Rotation
    }

    private struct Header: Decodable {
        let version: Int
        let packing: String
        let scale_dtype: String
        let codebook_family: String
        let rotation: Rotation
        let codebooks: [String: Codebook]
    }

    let codebooks: [Int: Codebook]
    let projections: [String: Projection]
    private static let supportedBits: Set<Int> = [2, 3, 4, 6, 8]

    init(configuration data: Data) throws {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let rawHeader = root["jangtq"] as? [String: Any],
            let quantization = root["quantization"] as? [String: Any]
        else { throw ValidationError.invalid("missing JANGH metadata") }
        let decoder = JSONDecoder()
        let header = try decoder.decode(
            Header.self, from: JSONSerialization.data(withJSONObject: rawHeader))
        guard header.version == 2, header.packing == "lsb-bitstream",
            header.scale_dtype == "float16", header.codebook_family == "odd-cubic"
        else { throw ValidationError.invalid("unsupported JANGH format contract") }

        var books: [Int: Codebook] = [:]
        for (key, book) in header.codebooks {
            guard let bits = Int(key), String(bits) == key, Self.supportedBits.contains(bits),
                book.alpha.isFinite, book.beta.isFinite, Float(book.alpha).isFinite,
                Float(book.beta).isFinite, book.levels.count == 1 << bits
            else { throw ValidationError.invalid("invalid codebook \(key)") }
            for (index, level) in book.levels.enumerated() {
                let u = Double(index) - Double((1 << bits) - 1) / 2
                let expected = u * (book.alpha + book.beta * u * u)
                // The manifest stores F32-rounded levels and decimal coefficients.
                guard level.isFinite, expected.isFinite, Float(level).isFinite,
                    Float(expected).isFinite,
                    abs(level - expected) <= max(1, abs(expected)) * 2e-6,
                    index == 0 || level > book.levels[index - 1]
                else { throw ValidationError.invalid("inconsistent codebook \(key)") }
            }
            books[bits] = book
        }

        var modules: [String: Projection] = [:]
        for (name, raw) in quantization {
            guard let entry = raw as? [String: Any], let mode = entry["mode"] as? String,
                mode == "jangtq2"
            else { continue }
            let projection = try decoder.decode(
                Projection.self, from: JSONSerialization.data(withJSONObject: entry))
            guard name.contains("."), Self.supportedBits.contains(projection.bits),
                books[projection.bits] != nil,
                ["gate_proj", "up_proj", "down_proj"].contains(
                    String(name.split(separator: ".").last ?? ""))
            else { throw ValidationError.invalid("unsupported JANGH projection \(name)") }
            modules[name] = projection
        }
        guard !modules.isEmpty else { throw ValidationError.invalid("no JANGH projections") }
        for name in modules.keys {
            let parent = String(name[..<name.lastIndex(of: ".")!])
            guard let gate = modules[parent + ".gate_proj"],
                let up = modules[parent + ".up_proj"], modules[parent + ".down_proj"] != nil,
                gate.rotation == up.rotation
            else {
                throw ValidationError.invalid("incomplete or incompatible JANGH expert \(parent)")
            }
        }
        codebooks = books
        projections = modules
    }

    /// Cross-check the bundle's safetensor headers against architecture dimensions.
    /// This does not read weights or allocate an expanded expert bank.
    func validateTensorHeaders(
        module: String, experts: Int, inputDimensions: Int, outputDimensions: Int,
        packedShape: [Int], packedDType: String, scalesShape: [Int], scalesDType: String
    ) throws {
        guard let projection = projections[module], experts > 0,
            inputDimensions > 0, outputDimensions > 0, inputDimensions.isMultiple(of: 32)
        else { throw ValidationError.invalid("invalid JANGH dimensions or module") }
        let width = inputDimensions.multipliedReportingOverflow(by: projection.bits)
        guard !width.overflow, packedDType == "U32", scalesDType == "F16",
            packedShape == [experts, outputDimensions, width.partialValue / 32],
            scalesShape == [experts, outputDimensions]
        else { throw ValidationError.invalid("JANGH tensor header mismatch for \(module)") }
    }
}
