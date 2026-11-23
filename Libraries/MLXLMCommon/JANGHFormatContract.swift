import Foundation

/// Validates JANGH metadata without enabling an executable model path.
/// Keep this separate from affine quantization: identical bit packing does not
/// imply identical weight decoding or activation rotation.
struct JANGHFormatContract: Sendable {
    enum ValidationError: Error, Equatable, LocalizedError {
        case invalid(String)

        /// The message itself: apps show `localizedDescription`, which otherwise reads "The operation couldn't be
        /// completed. (MLXLMCommon.JANGHFormatContract.ValidationError error 0.)".
        var errorDescription: String? {
            switch self {
            case .invalid(let message): return "JANGH bundle: \(message)"
            }
        }
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
    /// JANGT trellis projections ("mode": "jangt") and their code width K, keyed like `projections`. A routed layer
    /// may mix the two formats per group (gate/up share one format and width; down may differ).
    let trellis: [String: Double]
    /// Routed projections kept in the source checkpoint's own NVFP4 / MXFP8 bytes inside a stack that has custom
    /// roles (Inkling-2-Mini JANGHT), keyed like `projections`. Empty unless the caller names the architecture's
    /// routed projections that may carry them (`nativeRoutedCandidates`); the host model runs them, not the custom-bank
    /// kernels.
    let native: [String: JANGHNativeRoutedFormat]
    private static let supportedBits: Set<Int> = [2, 3, 4, 6, 8]
    static let swigluRoles = ["gate_proj", "up_proj", "down_proj"]
    static let relu2Roles = ["fc1", "fc2"]

    /// Whether `parent` (a routed expert stack) is a gate-less relu2 pair.
    func isRelu2Stack(_ parent: String) -> Bool {
        projections[parent + ".fc1"] != nil && projections[parent + ".fc2"] != nil
    }

    init(configuration data: Data, nativeRoutedCandidates: Set<String> = []) throws {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let quantization = root["quantization"] as? [String: Any]
        else { throw ValidationError.invalid("missing JANGH metadata") }
        let decoder = JSONDecoder()
        // The codebook header (`jangtq`) is required by every `jangtq2` projection. A pure-JANGT bundle (Qwen3.8-Flash-
        // Next JANGHT, 2026-10-16: every routed projection `mode: jangt`) needs no codebooks and may omit it; a
        // `jangtq2` projection without the header is still refused below.
        let header: Header?
        if let rawHeader = root["jangtq"] {
            guard let object = rawHeader as? [String: Any] else {
                throw ValidationError.invalid("invalid JANGH header")
            }
            let decoded = try decoder.decode(Header.self, from: JSONSerialization.data(withJSONObject: object))
            guard decoded.version == 2, decoded.packing == "lsb-bitstream",
                decoded.scale_dtype == "float16", decoded.codebook_family == "odd-cubic"
            else { throw ValidationError.invalid("unsupported JANGH format contract") }
            header = decoded
        } else {
            guard root["jangt"] is [String: Any] else { throw ValidationError.invalid("missing JANGH metadata") }
            header = nil
        }

        var books: [Int: Codebook] = [:]
        for (key, book) in header?.codebooks ?? [:] {
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
                (Self.swigluRoles + Self.relu2Roles).contains(
                    String(name.split(separator: ".").last ?? ""))
            else { throw ValidationError.invalid("unsupported JANGH projection \(name)") }
            modules[name] = projection
        }
        var trellisModules: [String: Double] = [:]
        for (name, raw) in quantization {
            guard let entry = raw as? [String: Any], entry["mode"] as? String == "jangt" else { continue }
            guard let k = (entry["kbits"] as? NSNumber)?.doubleValue, JANGTCode.isSupported(k), name.contains("."),
                ["gate_proj", "up_proj", "down_proj"].contains(String(name.split(separator: ".").last ?? ""))
            else { throw ValidationError.invalid("unsupported JANGT projection \(name)") }
            trellisModules[name] = k
        }
        if !trellisModules.isEmpty {
            guard let jt = root["jangt"] as? [String: Any], (jt["version"] as? NSNumber)?.intValue == 1,
                jt["code"] as? String == "v2_halfbits",
                (jt["state_bits"] as? NSNumber)?.intValue == JANGTCode.stateBits,
                jt["rotation"] as? String == "hadamard128-signs"
            else { throw ValidationError.invalid("unsupported JANGT header") }
            if let mismatch = JANGTCode.validateDeclaredCode(jt) { throw ValidationError.invalid(mismatch) }
        }
        guard !modules.isEmpty || !trellisModules.isEmpty else {
            throw ValidationError.invalid("no JANGH projections")
        }
        // Native routed roles: only at the routed projections the architecture names, only with the exact entry of
        // the format. A caller that names none keeps the previous contract (a mixed stack is refused as incomplete).
        var nativeModules: [String: JANGHNativeRoutedFormat] = [:]
        if !nativeRoutedCandidates.isEmpty {
            for (name, raw) in quantization {
                guard nativeRoutedCandidates.contains(name), let entry = raw as? [String: Any],
                    let mode = entry["mode"] as? String, let format = JANGHNativeRoutedFormat(rawValue: mode),
                    Self.swigluRoles.contains(String(name.split(separator: ".").last ?? ""))
                else { continue }
                guard Set(entry.keys) == ["mode", "group_size", "bits"],
                    let group = entry["group_size"] as? NSNumber, let bits = entry["bits"] as? NSNumber,
                    group.doubleValue == Double(format.groupSize), bits.doubleValue == Double(format.bits)
                else { throw ValidationError.invalid("unsupported native routed projection \(name)") }
                nativeModules[name] = format
            }
        }
        let denseK2 = root["model_type"] as? String == "k2_horizon"
            && root["mlp_layout"] as? String == "dense_jangh_down"
        guard !denseK2 || trellisModules.isEmpty else {
            throw ValidationError.invalid("JANGT projections are not supported in dense K2 bundles")
        }
        for name in trellisModules.keys {
            let parent = String(name[..<name.lastIndex(of: ".")!])
            let roles = ["gate_proj", "up_proj", "down_proj"].map { parent + "." + $0 }
            guard roles.allSatisfy({ modules[$0] != nil || trellisModules[$0] != nil || nativeModules[$0] != nil }),
                (trellisModules[roles[0]] != nil) == (trellisModules[roles[1]] != nil),
                trellisModules[roles[0]] == trellisModules[roles[1]],
                nativeModules[roles[0]] == nativeModules[roles[1]]
            else { throw ValidationError.invalid("incomplete or incompatible JANGT expert \(parent)") }
        }
        for name in modules.keys {
            if denseK2 {
                let parts = name.split(separator: ".", omittingEmptySubsequences: false)
                guard parts.count == 5, parts[0] == "model", parts[1] == "layers",
                      let layer = Int(parts[2]), layer >= 0, String(layer) == parts[2],
                      parts[3] == "mlp", parts[4] == "down_proj" else {
                    throw ValidationError.invalid("unsupported K2 dense JANGH projection \(name)")
                }
                continue
            }
            let parent = String(name[..<name.lastIndex(of: ".")!])
            let role = String(name[name.index(after: name.lastIndex(of: ".")!)...])
            if Self.relu2Roles.contains(role) {
                // Gate-less relu2 stack (Nemotron-H latent MoE): fc1 and fc2 are
                // both required, carry independent widths and rotations, and a
                // parent never mixes them with SwiGLU roles.
                guard modules[parent + ".fc1"] != nil, modules[parent + ".fc2"] != nil,
                    Self.swigluRoles.allSatisfy({
                        modules[parent + "." + $0] == nil && trellisModules[parent + "." + $0] == nil
                    })
                else {
                    throw ValidationError.invalid("incomplete or mixed JANGH relu2 expert \(parent)")
                }
                continue
            }
            if trellisModules[parent + ".gate_proj"] != nil {
                continue // gate/up are trellis; the JANGT loop above checked the whole triple.
            }
            if let gate = nativeModules[parent + ".gate_proj"] {
                // Native gate/up feeding this codebook down: one native format for both.
                guard nativeModules[parent + ".up_proj"] == gate, role == "down_proj" else {
                    throw ValidationError.invalid("incomplete or incompatible JANGH expert \(parent)")
                }
                continue
            }
            guard let gate = modules[parent + ".gate_proj"],
                let up = modules[parent + ".up_proj"],
                modules[parent + ".down_proj"] != nil || trellisModules[parent + ".down_proj"] != nil
                    || nativeModules[parent + ".down_proj"] != nil,
                gate.rotation == up.rotation
            else {
                throw ValidationError.invalid("incomplete or incompatible JANGH expert \(parent)")
            }
        }
        codebooks = books
        projections = modules
        trellis = trellisModules
        native = nativeModules
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

/// A routed projection stored in the source checkpoint's own block-scaled float format (Inkling-2-Mini JANGHT native
/// units): `weight` uint32 codes `(E, N, kin * bits / 32)`, `scales` uint8 `(E, N, kin / group)` (e4m3 for NVFP4,
/// e8m0 for MXFP8), and for NVFP4 a per-expert `global_scale` float32 `(E,)` (MLX semantics: the dequantized value
/// is `e2m1 * e4m3(s) * global_scale / 2688`).
public enum JANGHNativeRoutedFormat: String, Sendable, Equatable {
    case nvfp4
    case mxfp8

    public var groupSize: Int { self == .nvfp4 ? 16 : 32 }
    public var bits: Int { self == .nvfp4 ? 4 : 8 }
    /// Tensor suffixes of one native projection, exactly.
    var suffixes: [String] { self == .nvfp4 ? [".weight", ".scales", ".global_scale"] : [".weight", ".scales"] }
}
