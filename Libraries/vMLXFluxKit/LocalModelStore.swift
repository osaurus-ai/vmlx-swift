import Foundation

public enum LocalFluxComponent: String, CaseIterable, Codable, Sendable {
    case root
    case tokenizer
    case transformer
    case scheduler
    case textEncoder
    case vae
    case assets
}

public enum LocalFluxReadiness: String, Codable, Sendable {
    case loadableScaffold
    case incomplete
    case unknown
}

public struct LocalFluxModel: Sendable {
    public let directory: URL
    public let directoryName: String
    public let canonicalName: String?
    public let displayName: String
    public let kind: ModelKind?
    public let quantizationBits: Int?
    public let components: Set<LocalFluxComponent>
    public let safetensorCount: Int
    public let totalBytes: UInt64
    public let hasModelIndex: Bool
    public let readiness: LocalFluxReadiness
    public let blockedReasons: [String]

    public var canEnterNativeLoadPath: Bool {
        readiness == .loadableScaffold
    }
}

public struct MLXStudioModelStore: Sendable {
    public let root: URL

    public init(root: URL = MLXStudioModelStore.defaultImageRoot) {
        self.root = root
    }

    public static var defaultImageRoot: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".mlxstudio/models/image", isDirectory: true)
    }

    public func scan() throws -> [LocalFluxModel] {
        let fm = FileManager.default
        guard fm.fileExists(atPath: root.path) else { return [] }
        let entries = try fm.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )
        return try entries.flatMap { url -> [LocalFluxModel] in
            let values = try url.resourceValues(forKeys: [.isDirectoryKey])
            guard values.isDirectory == true else { return [] }
            let parent = try Self.inspect(directory: url)
            let variants = try Self.nestedQuantVariants(in: url, parent: parent)
            if !variants.isEmpty && parent.readiness != .loadableScaffold {
                return variants
            }
            return variants.isEmpty ? [parent] : [parent] + variants
        }
        .sorted { $0.directoryName.localizedStandardCompare($1.directoryName) == .orderedAscending }
    }

    public func resolve(name: String) throws -> LocalFluxModel? {
        let requestedDirectory = Self.normalizedName(name)
        let models = try scan()
        // 1. Literal, case-insensitive directory-name match FIRST — this preserves the
        //    `-4bit`/`-8bit` quant suffix so requesting an exact bundle (e.g.
        //    "FLUX.1-schnell-mflux-8bit") never collapses onto a different-quant sibling
        //    ("...-4bit"), which `normalizedName` would do (it strips the bit suffix).
        if let literal = models.first(where: {
            $0.directoryName.compare(name, options: .caseInsensitive) == .orderedSame
        }) {
            return literal
        }
        // 2. Normalized exact match (quant-insensitive, separator-insensitive).
        if let exact = models.first(where: {
            Self.normalizedName($0.directoryName) == requestedDirectory
        }) {
            return exact
        }
        let requested = Self.canonicalName(for: name)
            ?? ModelRegistry.lookupFuzzy(name: name)?.name
            ?? requestedDirectory
        return models.first { model in
            guard let canonicalName = model.canonicalName else {
                return Self.normalizedName(model.directoryName) == requested
            }
            return canonicalName == requested
                || Self.normalizedName(model.directoryName) == requested
        }
    }

    public static func inspect(directory: URL) throws -> LocalFluxModel {
        try inspect(directory: directory, directoryName: directory.lastPathComponent)
    }

    private static func inspect(
        directory: URL,
        directoryName: String,
        canonicalName canonicalOverride: String? = nil,
        quantizationBits quantizationOverride: Int? = nil
    ) throws -> LocalFluxModel {
        let name = directoryName
        let canonical = canonicalOverride
            ?? canonicalName(for: name)
            ?? ModelRegistry.lookupFuzzy(name: name)?.name
        let entry = canonical.flatMap { ModelRegistry.lookup(name: $0) }
        let components = try detectComponents(in: directory)
        let safetensors = try safetensorSummary(in: directory)
        let hasModelIndex = FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("model_index.json").path
        )
        let reasons = blockedReasons(
            canonicalName: canonical,
            components: components,
            safetensorCount: safetensors.count
        )
        let readiness: LocalFluxReadiness
        if canonical == nil {
            readiness = .unknown
        } else if reasons.isEmpty {
            readiness = .loadableScaffold
        } else {
            readiness = .incomplete
        }
        return LocalFluxModel(
            directory: directory,
            directoryName: name,
            canonicalName: canonical,
            displayName: entry?.displayName ?? displayName(forCanonicalName: canonical) ?? name,
            kind: entry?.kind ?? defaultKind(forCanonicalName: canonical),
            quantizationBits: quantizationOverride ?? quantizationBits(in: name),
            components: components,
            safetensorCount: safetensors.count,
            totalBytes: safetensors.bytes,
            hasModelIndex: hasModelIndex,
            readiness: readiness,
            blockedReasons: reasons
        )
    }

    public static func canonicalName(for name: String) -> String? {
        let key = normalizedName(name)
        if key.contains("z-image") || key.contains("zimage") {
            return "z-image-turbo"
        }
        if key.contains("qwen-image") || key.contains("qwenimage") {
            return key.contains("edit") ? "qwen-image-edit" : "qwen-image"
        }
        if key.contains("ideogram") {
            return "ideogram"
        }
        if key.contains("flux2") || key.contains("flux-2") {
            return key.contains("edit") ? "flux2-klein-edit" : "flux2-klein"
        }
        if key.contains("flux1") || key.contains("flux-1") {
            if key.contains("kontext") { return "flux1-kontext" }
            if key.contains("fill") { return "flux1-fill" }
            if key.contains("dev") { return "flux1-dev" }
            if key.contains("schnell") { return "flux1-schnell" }
        }
        if key.contains("fibo") { return "fibo" }
        if key.contains("seedvr2") || key.contains("seed-vr2") { return "seedvr2" }
        if key.contains("wan-2-1") || key.contains("wan21") { return "wan-2.1" }
        if key.contains("wan-2-2") || key.contains("wan22") { return "wan-2.2" }
        return nil
    }

    public static func normalizedName(_ name: String) -> String {
        var key = name.lowercased()
        if let slash = key.lastIndex(of: "/") {
            key = String(key[key.index(after: slash)...])
        }
        for replacement in [".", "_", " "] {
            key = key.replacingOccurrences(of: replacement, with: "-")
        }
        while key.contains("--") {
            key = key.replacingOccurrences(of: "--", with: "-")
        }
        for suffix in ["-mflux", "-mlx"] {
            if key.hasSuffix(suffix) {
                key.removeLast(suffix.count)
            }
        }
        if let bits = quantizationBits(in: key) {
            let suffix = "-\(bits)bit"
            if key.hasSuffix(suffix) {
                key.removeLast(suffix.count)
            }
        }
        return key
    }

    public static func quantizationBits(in name: String) -> Int? {
        let parts = name.lowercased()
            .replacingOccurrences(of: "_", with: "-")
            .split(separator: "-")
        for part in parts where part.hasSuffix("bit") {
            let digits = part.dropLast(3)
            if let bits = Int(digits) {
                return bits
            }
        }
        for part in parts where part.hasPrefix("q") {
            let digits = part.dropFirst()
            if let bits = Int(digits) {
                return bits
            }
        }
        return nil
    }

    private static func nestedQuantVariants(
        in directory: URL,
        parent: LocalFluxModel
    ) throws -> [LocalFluxModel] {
        let fm = FileManager.default
        let children = try fm.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )
        return try children.compactMap { child in
            let values = try child.resourceValues(forKeys: [.isDirectoryKey])
            guard values.isDirectory == true else { return nil }
            guard let bits = quantizationBits(in: child.lastPathComponent) else { return nil }
            let variantName = "\(parent.directoryName)-\(child.lastPathComponent)"
            return try inspect(
                directory: child,
                directoryName: variantName,
                canonicalName: parent.canonicalName,
                quantizationBits: bits)
        }
    }

    private static func detectComponents(in directory: URL) throws -> Set<LocalFluxComponent> {
        let fm = FileManager.default
        var components: Set<LocalFluxComponent> = []
        if try hasSafetensors(in: directory, recursive: false) {
            components.insert(.root)
        }
        let componentDirs: [(LocalFluxComponent, String)] = [
            (.tokenizer, "tokenizer"),
            (.transformer, "transformer"),
            (.scheduler, "scheduler"),
            (.textEncoder, "text_encoder"),
            (.vae, "vae"),
            (.assets, "assets"),
        ]
        for (component, child) in componentDirs {
            let url = directory.appendingPathComponent(child, isDirectory: true)
            var isDirectory: ObjCBool = false
            guard fm.fileExists(atPath: url.path, isDirectory: &isDirectory),
                  isDirectory.boolValue
            else { continue }
            if component == .tokenizer || component == .scheduler || component == .assets {
                components.insert(component)
            } else if try hasSafetensors(in: url, recursive: true) {
                components.insert(component)
            }
        }
        return components
    }

    private static func hasSafetensors(in directory: URL, recursive: Bool) throws -> Bool {
        let fm = FileManager.default
        if recursive {
            guard let enumerator = fm.enumerator(
                at: directory,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            ) else { return false }
            for case let url as URL in enumerator where url.pathExtension == "safetensors" {
                return true
            }
            return false
        }
        let entries = try fm.contentsOfDirectory(atPath: directory.path)
        return entries.contains { $0.hasSuffix(".safetensors") }
    }

    private static func safetensorSummary(in directory: URL) throws -> (count: Int, bytes: UInt64) {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return (0, 0) }
        var count = 0
        var bytes: UInt64 = 0
        for case let url as URL in enumerator where url.pathExtension == "safetensors" {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            guard values.isRegularFile == true else { continue }
            count += 1
            bytes += UInt64(values.fileSize ?? 0)
        }
        return (count, bytes)
    }

    private static func blockedReasons(
        canonicalName: String?,
        components: Set<LocalFluxComponent>,
        safetensorCount: Int
    ) -> [String] {
        guard canonicalName != nil else {
            return safetensorCount == 0 ? ["no safetensors found"] : ["unknown flux-family model"]
        }
        var reasons: [String] = []
        if safetensorCount == 0 {
            reasons.append("no safetensors found")
        }
        for component in [LocalFluxComponent.transformer, .textEncoder, .vae, .tokenizer] {
            if !components.contains(component) {
                reasons.append("missing \(component.rawValue)")
            }
        }
        return reasons
    }

    private static func displayName(forCanonicalName canonicalName: String?) -> String? {
        switch canonicalName {
        case "flux1-schnell": return "FLUX.1 Schnell"
        case "flux1-dev": return "FLUX.1 Dev"
        case "flux1-kontext": return "FLUX.1 Kontext"
        case "flux1-fill": return "FLUX.1 Fill"
        case "flux2-klein": return "FLUX.2 Klein"
        case "flux2-klein-edit": return "FLUX.2 Klein Edit"
        case "z-image-turbo": return "Z-Image Turbo"
        case "qwen-image": return "Qwen-Image"
        case "qwen-image-edit": return "Qwen-Image-Edit"
        case "fibo": return "FIBO"
        case "seedvr2": return "SeedVR2"
        case "wan-2.1": return "Wan 2.1"
        case "wan-2.2": return "Wan 2.2"
        default: return nil
        }
    }

    private static func defaultKind(forCanonicalName canonicalName: String?) -> ModelKind? {
        switch canonicalName {
        case "flux1-kontext", "flux1-fill", "flux2-klein-edit", "qwen-image-edit":
            return .imageEdit
        case "seedvr2":
            return .imageUpscale
        case "wan-2.1", "wan-2.2":
            return .videoGen
        case .some:
            return .imageGen
        case .none:
            return nil
        }
    }
}
