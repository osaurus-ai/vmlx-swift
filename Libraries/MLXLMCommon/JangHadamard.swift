// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import CoreFoundation
import Foundation
import MLX
import MLXNN

/// Opt-in for model implementations whose projection paths preserve Hadamard
/// wrappers (including decode fusions). An unported route must refuse the
/// bundle, not load rotated weights as ordinary affine matrices.
public protocol JangHadamardRuntimeModel: LanguageModel {
    func validateJangHadamardRuntime() throws
}

/// Validated Prism activation-basis contract. This is unrelated to JANGTQ's
/// weight-side rotation or historical converter-generated random signs.
public struct JangHadamardRuntimeContract: Sendable, Equatable {
    public let blockSize: Int
    public let forward: Set<String>
    public let inverse: Set<String>
    public let computeDType: DType
    let sidecarSigns: [Int: [Float]]
    let supportsPrismV2MTP: Bool

    public var modulePaths: Set<String> { forward.union(inverse) }

    init(
        blockSize: Int, forward: Set<String>, inverse: Set<String>,
        sidecarSigns: [Int: [Float]] = [:], supportsPrismV2MTP: Bool = false
    ) {
        self.blockSize = blockSize
        self.forward = forward
        self.inverse = inverse
        self.computeDType = .float32
        self.sidecarSigns = sidecarSigns
        self.supportsPrismV2MTP = supportsPrismV2MTP
    }
}

public struct JangTernaryPackedRuntimeContract: Sendable, Equatable {
    public let modulePaths: Set<String>
}

extension JangLoader {
    public static let prismHadamardModelType = "prism_hadamard_qwen35"

    private static let hadamardTransformName = "normalized-sylvester-walsh-hadamard"

    private static func runtimeContractJSON(at url: URL) throws -> [String: Any] {
        guard FileManager.default.fileExists(atPath: url.path) else { return [:] }
        guard
            let value = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
                as? [String: Any]
        else {
            throw JangLoaderError.invalidConfig(
                "runtime contract is not an object: \(url.lastPathComponent)")
        }
        return value
    }

    private static func runtimeContractOwners(at directory: URL) throws
        -> (config: [String: Any], jang: [String: Any])
    {
        let config = try runtimeContractJSON(at: directory.appendingPathComponent("config.json"))
        let jang: [String: Any]
        if let url = findConfigPath(at: directory) {
            jang = try runtimeContractJSON(at: url)
        } else {
            jang = config["jang_config"] as? [String: Any] ?? [:]
        }
        return (config, jang)
    }

    private static func runtimeContractMarkers(_ jang: [String: Any]) throws -> [String: Any] {
        guard let raw = jang["runtime"] else { return [:] }
        guard let runtime = raw as? [String: Any] else {
            throw JangLoaderError.invalidConfig("JANG runtime markers must be an object")
        }
        return runtime
    }

    /// JSON numbers and strings must not masquerade as Boolean contract flags.
    /// A malformed explicit marker cannot make a rotated bundle look ordinary.
    private static func runtimeContractFlag(_ runtime: [String: Any], key: String) throws -> Bool {
        guard let raw = runtime[key] else { return false }
        guard let value = raw as? NSNumber, CFGetTypeID(value) == CFBooleanGetTypeID() else {
            throw JangLoaderError.invalidConfig("JANG runtime marker must be Boolean: \(key)")
        }
        return value.boolValue
    }

    private static func contractPaths(_ raw: Any?, key: String) throws -> Set<String> {
        guard let paths = raw as? [String], Set(paths).count == paths.count,
            paths.allSatisfy({ path in
                !path.isEmpty && !path.hasSuffix(".weight")
                    && path.split(separator: ".", omittingEmptySubsequences: false)
                        .allSatisfy {
                            !$0.isEmpty && $0.allSatisfy { $0.isLetter || $0.isNumber || $0 == "_" }
                        }
            })
        else { throw JangLoaderError.invalidConfig("invalid or duplicate Hadamard \(key)") }
        return Set(paths)
    }

    private static func parseHadamard(_ raw: [String: Any]) throws
        -> JangHadamardRuntimeContract
    {
        guard raw["contract"] as? String == "prism.hadamard.v1",
            raw["transform"] as? String == hadamardTransformName,
            raw["axis"] as? String == "input-last-dimension",
            raw["sign_mode"] as? String == "explicit",
            raw["signs_dtype"] as? String == "float32",
            raw["signs_tensor_suffix"] as? String == "signs",
            raw["compute_dtype"] as? String == "float32",
            raw["gdn_v_grouped"] as? Bool == true,
            let block = raw["block_size"] as? Int, [512, 1024, 2048, 4096].contains(block)
        else {
            throw JangLoaderError.invalidConfig(
                "unsupported Hadamard contract; expected prism.hadamard.v1, normalized input-axis transform, explicit float32 signs/compute and grouped GDN"
            )
        }
        let forward = try contractPaths(raw["forward_modules"], key: "forward_modules")
        let inverse = try contractPaths(raw["inverse_modules"], key: "inverse_modules")
        guard !forward.union(inverse).isEmpty, forward.isDisjoint(with: inverse) else {
            throw JangLoaderError.invalidConfig("Hadamard module manifests are empty or overlap")
        }
        return JangHadamardRuntimeContract(blockSize: block, forward: forward, inverse: inverse)
    }

    private static func checkedRuntimeManifest(_ jang: [String: Any]) throws -> [String: Any] {
        guard let quantization = jang["quantization"] as? [String: Any],
            quantization["tensor_quantization_manifest_schema"] as? Int == 2,
            let manifest = quantization["tensor_quantization_manifest"] as? [String: Any],
            !manifest.isEmpty,
            quantization["tensor_quantization_manifest_count"] as? Int == manifest.count
        else {
            throw JangLoaderError.invalidConfig(
                "Hadamard/ternary storage requires a complete schema-2 tensor manifest with matching count"
            )
        }
        return manifest
    }

    /// Parse both copies and the declared Prism sidecar. Marker-only, malformed,
    /// contradictory or incompletely declared bundles fail before shard reads.
    public static func loadHadamardRuntimeContract(at directory: URL) throws
        -> JangHadamardRuntimeContract?
    {
        let (config, jang) = try runtimeContractOwners(at: directory)
        if config["model_type"] as? String == prismHadamardModelType {
            let declared = try parsePrismHadamard(config)
            return try loadHadamardSidecar(
                declared, from: directory.appendingPathComponent("hadamard.json"), supportsPrismV2MTP: true)
        }
        let runtime = try runtimeContractMarkers(jang)
        let requiresTransform = try runtimeContractFlag(
            runtime, key: "requires_hadamard_activation_transform")
        let usesPrismMTP = try runtimeContractFlag(runtime, key: "prism_mtp_v2_sidecar")
        if usesPrismMTP {
            let source = jang["source_model"] as? [String: Any]
            guard requiresTransform, config["model_type"] as? String == "qwen3_5",
                source?["model_type"] as? String == prismHadamardModelType
            else {
                throw JangLoaderError.invalidConfig("Prism MTP requires a Hadamard Qwen3.5 repack")
            }
        }
        let manifest =
            (jang["quantization"] as? [String: Any])?["tensor_quantization_manifest"]
            as? [String: Any] ?? [:]
        let hasManifestMarker = manifest.values.contains {
            ($0 as? [String: Any])?["hadamard"] != nil
        }
        let declared =
            config["hadamard"] != nil || jang["hadamard"] != nil
            || requiresTransform
            || hasManifestMarker
        guard declared else { return nil }

        var parsed: JangHadamardRuntimeContract?
        var sidecar: String?
        for owner in [config, jang] where owner["hadamard"] != nil {
            guard let raw = owner["hadamard"] as? [String: Any],
                let name = raw["sidecar"] as? String, name == "hadamard.json"
            else {
                throw JangLoaderError.invalidConfig(
                    "Hadamard declaration requires hadamard.json sidecar")
            }
            let candidate = try parseHadamard(raw)
            guard parsed == nil || parsed == candidate else {
                throw JangLoaderError.invalidConfig(
                    "config.json and JANG Hadamard declarations disagree")
            }
            parsed = candidate
            sidecar = name
        }
        guard let parsed, let sidecar else {
            throw JangLoaderError.invalidConfig(
                "Hadamard runtime marker is present but its contract is missing")
        }

        let checked = try checkedRuntimeManifest(jang)
        var covered = Set<String>()
        for (path, entryValue) in checked {
            guard let entry = entryValue as? [String: Any] else {
                throw JangLoaderError.invalidConfig("invalid tensor manifest entry: \(path)")
            }
            guard let raw = entry["hadamard"] else { continue }
            guard let spec = raw as? [String: Any], parsed.modulePaths.contains(path),
                spec["block_size"] as? Int == parsed.blockSize,
                spec["signs_tensor"] as? String == "\(path).signs",
                spec["direction"] as? String
                    == (parsed.forward.contains(path) ? "forward" : "inverse"),
                entry["bits"] as? Int == 2, entry["group_size"] as? Int == 128,
                entry["mode"] as? String == "affine",
                entry["storage"] == nil || entry["storage"] as? String == "ternary_packed_26b"
            else {
                throw JangLoaderError.invalidConfig(
                    "inconsistent Hadamard tensor manifest: \(path)")
            }
            covered.insert(path)
        }
        guard covered == parsed.modulePaths else {
            throw JangLoaderError.invalidConfig(
                "Hadamard declaration and tensor manifest coverage disagree")
        }

        return try loadHadamardSidecar(
            parsed, from: directory.appendingPathComponent(sidecar), supportsPrismV2MTP: usesPrismMTP)
    }

    private static func parsePrismHadamard(_ config: [String: Any]) throws
        -> JangHadamardRuntimeContract
    {
        guard config["schema_version"] as? Int == 2,
            config["base_model_type"] as? String == "qwen3_5",
            config["tensor_namespace"] as? String == "mlx-vlm-qwen3_5",
            config["gdn_activation_layout"] as? String == "grouped",
            config["hadamard_config"] as? String == "hadamard.json",
            let components = config["components"] as? [String: Any],
            try runtimeContractFlag(components, key: "text"),
            try runtimeContractFlag(components, key: "vision"),
            let quantization = config["quantization"] as? [String: Any],
            quantization["bits"] as? Int == 2,
            quantization["group_size"] as? Int == 128,
            quantization["mode"] as? String == "affine",
            let modules = config["modules"] as? [[String: Any]], !modules.isEmpty,
            let block = modules.first?["block"] as? Int, [512, 1024, 2048, 4096].contains(block)
        else {
            throw JangLoaderError.invalidConfig("unsupported Prism Hadamard Qwen3.5 bundle layout")
        }

        var forward: [String] = []
        var inverse: [String] = []
        for entry in modules {
            guard let path = entry["path"] as? String,
                entry["block"] as? Int == block,
                entry["dtype"] as? String == "float16",
                entry["embedding"] != nil
            else { throw JangLoaderError.invalidConfig("invalid Prism Hadamard module declaration") }
            let isEmbedding = try runtimeContractFlag(entry, key: "embedding")
            if isEmbedding {
                inverse.append("language_model.\(path)")
            } else {
                forward.append("language_model.\(path)")
            }
        }
        let forwardPaths = try contractPaths(forward, key: "forward_modules")
        let inversePaths = try contractPaths(inverse, key: "inverse_modules")
        guard !forwardPaths.isEmpty,
            inversePaths == ["language_model.model.embed_tokens"],
            forwardPaths.isDisjoint(with: inversePaths)
        else {
            throw JangLoaderError.invalidConfig("inconsistent Prism Hadamard module coverage")
        }
        return JangHadamardRuntimeContract(
            blockSize: block, forward: forwardPaths, inverse: inversePaths)
    }

    private static func loadHadamardSidecar(
        _ parsed: JangHadamardRuntimeContract, from url: URL, supportsPrismV2MTP: Bool
    ) throws -> JangHadamardRuntimeContract
    {
        let prism = try runtimeContractJSON(at: url)
        guard prism["prism.hadamard.version"] as? Int == 1,
            prism["prism.hadamard.block_size"] as? Int == parsed.blockSize,
            prism["prism.hadamard.transform"] as? String == hadamardTransformName,
            prism["prism.hadamard.axis"] as? String == "input-last-dimension",
            prism["prism.hadamard.sign_mode"] as? String == "explicit",
            prism["prism.hadamard.gdn_v_grouped"] as? Bool == true,
            let forward = prism["prism.hadamard.weight_names"] as? [String],
            let inverse = prism["prism.hadamard.inverse_weight_names"] as? [String],
            forward.count == parsed.forward.count, inverse.count == parsed.inverse.count,
            Set(forward) == Set(parsed.forward.map { "\($0).weight" }),
            Set(inverse) == Set(parsed.inverse.map { "\($0).weight" }),
            let widths = prism["prism.hadamard.sign_widths"] as? [Int], !widths.isEmpty,
            Set(widths).count == widths.count,
            widths.allSatisfy({ $0 > 0 && $0 % parsed.blockSize == 0 }),
            let values = prism["prism.hadamard.sign_values"] as? [NSNumber],
            values.allSatisfy({ $0.doubleValue == 1 || $0.doubleValue == -1 })
        else {
            throw JangLoaderError.invalidConfig("missing or inconsistent Prism Hadamard sidecar")
        }
        var signCount = 0
        for width in widths {
            let (next, overflow) = signCount.addingReportingOverflow(width)
            guard !overflow, next <= values.count else {
                throw JangLoaderError.invalidConfig(
                    "Prism Hadamard sign widths exceed stored values")
            }
            signCount = next
        }
        guard signCount == values.count else {
            throw JangLoaderError.invalidConfig("Prism Hadamard sign count disagrees with widths")
        }
        var signs: [Int: [Float]] = [:]
        var offset = 0
        for width in widths {
            signs[width] = values[offset ..< offset + width].map(\.floatValue)
            offset += width
        }
        return JangHadamardRuntimeContract(
            blockSize: parsed.blockSize, forward: parsed.forward, inverse: parsed.inverse,
            sidecarSigns: signs, supportsPrismV2MTP: supportsPrismV2MTP)
    }

    public static func loadTernaryPackedRuntimeContract(at directory: URL) throws
        -> JangTernaryPackedRuntimeContract?
    {
        let (_, jang) = try runtimeContractOwners(at: directory)
        let runtime = try runtimeContractMarkers(jang)
        let requiresExpansion = try runtimeContractFlag(
            runtime, key: "requires_jang_ternary_packed_expansion")
        let quantization = jang["quantization"] as? [String: Any] ?? [:]
        let manifest = quantization["tensor_quantization_manifest"] as? [String: Any] ?? [:]
        let expansionValue = quantization["ternary_packed_runtime_expansion"]
        let hasExpansion = expansionValue != nil && !(expansionValue is NSNull)
        let hasStorage = manifest.values.contains {
            ($0 as? [String: Any])?["storage"] as? String == "ternary_packed_26b"
        }
        guard
            requiresExpansion || hasExpansion || hasStorage
        else { return nil }
        let requiresAffine1Expansion = try runtimeContractFlag(
            runtime, key: "requires_jang_affine1_expansion")
        guard requiresExpansion, !requiresAffine1Expansion,
            let expansion = expansionValue as? [String: Any],
            expansion["storage"] as? String == "ternary_packed_26b",
            expansion["runtime_bits"] as? Int == 2,
            expansion["group_size"] as? Int == 128,
            expansion["lossless"] as? Bool == true,
            expansion["scales_unchanged"] as? Bool == true,
            expansion["biases"] as? String == "materialized as -scales at load"
        else {
            throw JangLoaderError.invalidConfig(
                "unsupported or incomplete ternary_packed_26b runtime contract")
        }
        var paths = Set<String>()
        for (path, value) in try checkedRuntimeManifest(jang) {
            guard let entry = value as? [String: Any] else {
                throw JangLoaderError.invalidConfig("invalid tensor manifest entry: \(path)")
            }
            guard let storage = entry["storage"] else { continue }
            guard storage as? String == "ternary_packed_26b",
                entry["bits"] as? Int == 2, entry["runtime_bits"] as? Int == 2,
                entry["group_size"] as? Int == 128, entry["mode"] as? String == "affine"
            else {
                throw JangLoaderError.invalidConfig("unsupported packed tensor storage: \(path)")
            }
            paths.insert(path)
        }
        guard !paths.isEmpty else {
            throw JangLoaderError.invalidConfig("ternary storage declares no packed modules")
        }
        return JangTernaryPackedRuntimeContract(modulePaths: paths)
    }
}

/// Normalized block FWHT in float32; sign multiplication moves to the other
/// side for inverse embedding lookup. The activation dtype is not changed.
public func jangHadamardActivation(
    _ input: MLXArray, signs: MLXArray, blockSize: Int, inverse: Bool = false
) -> MLXArray {
    precondition([512, 1024, 2048, 4096].contains(blockSize))
    precondition(input.ndim > 0 && input.dim(-1) % blockSize == 0)
    let source = input.asType(.float32)
    let signed = inverse ? source : source * signs
    let transformed = MLX.hadamardTransform(
        signed.reshaped(-1, blockSize), scale: 1 / sqrt(Float(blockSize))
    )
    .reshaped(input.shape)
    return (inverse ? transformed * signs : transformed).asType(input.dtype)
}

public final class HadamardQuantizedLinear: QuantizedLinear {
    public let blockSize: Int
    @ParameterInfo(key: "signs") public var signs: MLXArray

    public init(_ source: QuantizedLinear, blockSize: Int) {
        self.blockSize = blockSize
        self._signs.wrappedValue = .zeros([source.shape.1], dtype: .float32)
        super.init(
            weight: source.weight, bias: source.bias, scales: source.scales, biases: source.biases,
            groupSize: source.groupSize, bits: source.bits, mode: source.mode)
        freeze()
    }

    public override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let rotated = jangHadamardActivation(input, signs: signs, blockSize: blockSize)
        guard mode == .affine, bits == 2, groupSize == 128,
            rotated.ndim == 3, rotated.dim(0) == 1,
            FlashVerificationScope.usesRowExactVerification(inputShape: rotated.shape)
        else { return super.callAsFunction(rotated) }
        let rows = rotated.dim(1)
        if bias == nil, let output = RowExactAffineQMV.project(
            rotated.reshaped(rows, -1), weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            reference: { projectRows($0.reshaped(1, rows, -1)).reshaped(rows, -1) })
        {
            return output.reshaped(1, rows, -1)
        }
        return projectRows(rotated)
    }

    private func projectRows(_ rotated: MLXArray) -> MLXArray {
        // Apply the original packed projection using its decode shape for
        // each independent verification row.
        var outputs: [MLXArray] = []
        for row in 0..<rotated.dim(1) {
            outputs.append(super.callAsFunction(rotated[0..., row..<(row + 1), 0...]))
        }
        return concatenated(outputs, axis: 1)
    }
}

public final class HadamardQuantizedEmbedding: QuantizedEmbedding {
    public let blockSize: Int
    @ParameterInfo(key: "signs") public var signs: MLXArray

    public init(_ source: QuantizedEmbedding, blockSize: Int) {
        self.blockSize = blockSize
        self._signs.wrappedValue = .zeros([source.shape.1], dtype: .float32)
        super.init(
            weight: source.weight, scales: source.scales, biases: source.biases,
            groupSize: source.groupSize, bits: source.bits, mode: source.mode)
        outputDType = source.outputDType
        freeze()
    }

    public override func callAsFunction(_ input: MLXArray) -> MLXArray {
        jangHadamardActivation(
            super.callAsFunction(input), signs: signs, blockSize: blockSize, inverse: true)
    }

    public override func asLinear(_ input: MLXArray) -> MLXArray {
        preconditionFailure("Hadamard-rotated embeddings cannot serve as a tied output head")
    }
}

/// Raw-array projection fusions cannot silently bypass a module's activation
/// transform. Ordinary QuantizedLinear fast paths retain their old eligibility.
public func jangAllowsRawQuantizedProjection(_ module: Module) -> Bool {
    !(module is HadamardQuantizedLinear)
}

extension JangHadamardRuntimeContract {
    func validateRoute(model: LanguageModel) throws {
        guard let supported = model as? JangHadamardRuntimeModel else {
            throw JangLoaderError.loadFailed(
                "Hadamard activation transforms are not implemented for runtime route \(type(of: model)); use the Qwen3.5 VLM or wrapped text route"
            )
        }
        try supported.validateJangHadamardRuntime()
        let modules = Dictionary(uniqueKeysWithValues: model.namedModules())
        for path in modulePaths {
            let module = modules[path]
            guard
                (forward.contains(path) && module is Linear)
                    || (inverse.contains(path) && module is Embedding)
            else {
                throw JangLoaderError.loadFailed(
                    "Hadamard module has no compatible destination: \(path)")
            }
        }
    }

    func install(model: Module) throws {
        let modules = Dictionary(uniqueKeysWithValues: model.namedModules())
        var updates: [(String, Module)] = []
        for path in modulePaths.sorted() {
            if forward.contains(path), let source = modules[path] as? QuantizedLinear {
                try validateLayout(
                    path: path, shape: source.shape, bits: source.bits, groupSize: source.groupSize,
                    mode: source.mode)
                updates.append((path, HadamardQuantizedLinear(source, blockSize: blockSize)))
            } else if inverse.contains(path), let source = modules[path] as? QuantizedEmbedding {
                try validateLayout(
                    path: path, shape: source.shape, bits: source.bits, groupSize: source.groupSize,
                    mode: source.mode)
                updates.append((path, HadamardQuantizedEmbedding(source, blockSize: blockSize)))
            } else {
                throw JangLoaderError.loadFailed(
                    "Hadamard module was not constructed with its declared quantization: \(path)")
            }
        }
        try model.update(modules: ModuleChildren.unflattened(updates), verify: .none)
    }

    /// Compare checkpoint dimensions to the still-unquantized architecture,
    /// not merely to another self-consistent (but possibly wrong) tensor.
    func validateWeights(_ weights: [String: MLXArray], model: Module) throws {
        let modules = Dictionary(uniqueKeysWithValues: model.namedModules())
        for path in modulePaths {
            let shape: (Int, Int)
            if forward.contains(path), let linear = modules[path] as? Linear {
                shape = linear.shape
            } else if inverse.contains(path), let embedding = modules[path] as? Embedding {
                shape = embedding.shape
            } else {
                throw JangLoaderError.loadFailed(
                    "missing Hadamard architecture destination: \(path)")
            }
            try validateLayout(path: path, shape: shape, bits: 2, groupSize: 128, mode: .affine)
            guard let weight = weights["\(path).weight"], weight.dtype == .uint32,
                weight.shape == [shape.0, shape.1 / 16],
                let scales = weights["\(path).scales"], scales.dtype == .float16,
                scales.shape == [shape.0, shape.1 / 128],
                let biases = weights["\(path).biases"], biases.dtype == .float16,
                biases.shape == scales.shape,
                let signs = weights["\(path).signs"], signs.dtype == .float32,
                signs.shape == [shape.1]
            else {
                throw JangLoaderError.loadFailed(
                    "Hadamard weight/scale/bias/sign shape or dtype mismatch: \(path)")
            }
        }
    }

    private func validateLayout(
        path: String, shape: (Int, Int), bits: Int, groupSize: Int, mode: QuantizationMode
    ) throws {
        guard bits == 2, groupSize == 128, mode == .affine,
            shape.0 > 0, shape.1 > 0, shape.1 % blockSize == 0,
            sidecarSigns.isEmpty || sidecarSigns[shape.1] != nil
        else {
            throw JangLoaderError.loadFailed("incompatible Hadamard shape/quantization: \(path)")
        }
    }

    func verifyLoaded(model: Module) throws {
        let modules = Dictionary(uniqueKeysWithValues: model.namedModules())
        let paths = modulePaths.sorted()
        let expectedSigns = sidecarSigns.mapValues { MLXArray($0) }
        var checks: [MLXArray] = []
        checks.reserveCapacity(paths.count)
        for path in paths {
            let signs: MLXArray
            let width: Int
            if forward.contains(path), let module = modules[path] as? HadamardQuantizedLinear {
                signs = module.signs
                width = module.shape.1
            } else if inverse.contains(path),
                let module = modules[path] as? HadamardQuantizedEmbedding
            {
                signs = module.signs
                width = module.shape.1
            } else {
                throw JangLoaderError.loadFailed("Hadamard wrapper missing after load: \(path)")
            }
            guard signs.dtype == .float32, signs.shape == [width] else {
                throw JangLoaderError.loadFailed("Malformed Hadamard signs: \(path)")
            }
            if let expected = expectedSigns[width] {
                checks.append(MLX.all(signs .== expected))
            } else {
                checks.append(MLX.all((signs .== 1) .|| (signs .== -1)))
            }
        }
        // Check all settled arrays with one host readback. The sidecar parser
        // already requires every reference sign to be +1 or -1.
        let matches = MLX.stacked(checks).asArray(Bool.self)
        if let index = matches.firstIndex(of: false) {
            throw JangLoaderError.loadFailed(
                "Hadamard signs are invalid or disagree with the sidecar: \(paths[index])")
        }
    }
}
