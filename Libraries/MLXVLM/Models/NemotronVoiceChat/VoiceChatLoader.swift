// NemotronLabs VoiceChat — load a shipped bundle (bf16 or quantized).
//
// 🚨 A quantized bundle must be `quantize`d BEFORE the weights land, using the
// PER-MODULE map in `config.json["quantization"]`. That map is not uniform:
// the published quants list 566 explicit module entries alongside the global
// `group_size`/`bits`/`mode`, and any module ABSENT from it is a floating-point
// passthrough. Quantizing everything uniformly destroys exactly the tensors
// this model cannot afford to lose — the RVQ codebook, the speaker latent,
// `mog_head.proj_mus` (read raw), and the character embedding table (its dtype
// is read to allocate a buffer).

import Foundation
import MLX
import MLXNN

public enum VoiceChatLoadError: Error, LocalizedError {
    case missingConfig(URL)
    case noWeights(URL)
    case weightUpdateFailed(Error)

    public var errorDescription: String? {
        switch self {
        case .missingConfig(let url):
            return "VoiceChat bundle has no config.json at \(url.path)"
        case .noWeights(let url):
            return "VoiceChat bundle has no .safetensors files in \(url.path)"
        case .weightUpdateFailed(let error):
            return "VoiceChat weight update failed: \(error)"
        }
    }
}

/// Per-module quantization recipe parsed from `config.json["quantization"]`.
public struct VoiceChatQuantizationMap {
    public let defaultGroupSize: Int
    public let defaultBits: Int
    public let defaultMode: QuantizationMode
    /// Module path → recipe. A path ABSENT here is an fp passthrough.
    public let perModule: [String: (groupSize: Int, bits: Int, mode: QuantizationMode)]

    public init?(_ json: [String: Any]?) {
        guard let json else { return nil }
        let groupSize = json["group_size"] as? Int ?? 64
        let bits = json["bits"] as? Int ?? 4
        let mode =
            (json["mode"] as? String).flatMap(QuantizationMode.init(rawValue:)) ?? .affine
        var modules = [String: (groupSize: Int, bits: Int, mode: QuantizationMode)]()
        for (key, value) in json {
            guard let entry = value as? [String: Any] else { continue }
            modules[key] = (
                groupSize: entry["group_size"] as? Int ?? groupSize,
                bits: entry["bits"] as? Int ?? bits,
                mode: (entry["mode"] as? String).flatMap(QuantizationMode.init(rawValue:))
                    ?? mode
            )
        }
        guard !modules.isEmpty else { return nil }
        self.defaultGroupSize = groupSize
        self.defaultBits = bits
        self.defaultMode = mode
        self.perModule = modules
    }

    /// Recipe for `path`, or nil to leave the module in floating point.
    public func recipe(for path: String)
        -> (groupSize: Int, bits: Int, mode: QuantizationMode)?
    {
        perModule[path]
    }

    /// The quantization map is keyed by BUNDLE tensor paths, but the Swift
    /// module tree renames several of them (see the sanitizers). Look-ups must
    /// therefore go through the same remapping, or every entry misses and the
    /// module is quantized with the global default instead of its own recipe —
    /// which is a hard shape error at the first matmul on a mixed-bit bundle
    /// like JANG, and silent quality loss on a uniform one.
    ///
    /// `VoiceChatQuantizationMapPathTests` pins this against the real bundles.
    public static func modulePath(forBundlePath path: String) -> String {
        var path = path
        if path.hasPrefix("stt_model.embed_tokens") {
            path = path.replacingOccurrences(
                of: "stt_model.embed_tokens", with: "stt_model.llm.backbone.embeddings")
        } else if path.hasPrefix("stt_model.lm_head") {
            path = path.replacingOccurrences(
                of: "stt_model.lm_head", with: "stt_model.llm.lm_head")
        } else if path.hasPrefix("stt_model.llm.") {
            path = path.replacingOccurrences(
                of: "stt_model.llm.", with: "stt_model.llm.backbone.")
        }
        path = path.replacingOccurrences(of: ".joint_net.2", with: ".joint_net.out")
        path = path.replacingOccurrences(
            of: "prvq._variance_list.", with: "prvq.variance_list.")
        return path
    }

    /// Recipe for a Swift module path, resolved through the bundle-path map.
    public func recipe(forModulePath modulePath: String)
        -> (groupSize: Int, bits: Int, mode: QuantizationMode)?
    {
        if let direct = perModule[modulePath] { return direct }
        for (bundlePath, recipe) in perModule
        where Self.modulePath(forBundlePath: bundlePath) == modulePath {
            return recipe
        }
        return nil
    }

    /// Swift module path → recipe, precomputed once (the linear scan above is
    /// fine for a lookup but not for 566 of them during a load).
    public func remappedByModulePath()
        -> [String: (groupSize: Int, bits: Int, mode: QuantizationMode)]
    {
        var out = [String: (groupSize: Int, bits: Int, mode: QuantizationMode)]()
        for (bundlePath, recipe) in perModule {
            out[Self.modulePath(forBundlePath: bundlePath)] = recipe
        }
        return out
    }
}

public enum VoiceChatLoader {

    /// Load a VoiceChat bundle from `directory`.
    ///
    /// Handles both the bf16 MLX layout and the published quants: the module
    /// tree is built, quantized per the bundle's own map when one is present,
    /// and only then updated with the sanitized weights.
    public static func load(from directory: URL) throws -> (
        model: NemotronVoiceChatModel, config: NemotronVoiceChatConfiguration
    ) {
        let dir = directory.resolvingSymlinksInPath()
        let configURL = dir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw VoiceChatLoadError.missingConfig(configURL)
        }
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(
            NemotronVoiceChatConfiguration.self, from: configData)
        let root = try JSONSerialization.jsonObject(with: configData) as? [String: Any]

        var weights = [String: MLXArray]()
        if let enumerator = FileManager.default.enumerator(
            at: dir, includingPropertiesForKeys: nil)
        {
            for case let url as URL in enumerator where url.pathExtension == "safetensors" {
                for (key, value) in try loadArrays(url: url) {
                    weights[key] = value
                }
            }
        }
        guard !weights.isEmpty else { throw VoiceChatLoadError.noWeights(dir) }

        let model = NemotronVoiceChatModel(config)
        let sanitized = NemotronVoiceChatModel.sanitized(weights, config: config)

        if let map = VoiceChatQuantizationMap(root?["quantization"] as? [String: Any]) {
            // Quantize ONLY what the bundle says it quantized, with THAT
            // module's own recipe. JANG bundles are mixed-bit per module, so
            // falling back to the global default is not a mild approximation —
            // it produces a packed-weight/scales shape mismatch on the first
            // matmul. A module the map omits stays fp.
            let recipes = map.remappedByModulePath()
            quantize(
                model: model,
                filter: { path, _ in
                    guard sanitized["\(path).scales"] != nil else { return nil }
                    return recipes[path]
                        ?? (
                            groupSize: map.defaultGroupSize, bits: map.defaultBits,
                            mode: map.defaultMode
                        )
                })
        }

        do {
            try model.update(
                parameters: ModuleParameters.unflattened(sanitized), verify: [.noUnusedKeys])
        } catch {
            throw VoiceChatLoadError.weightUpdateFailed(error)
        }
        MLX.eval(model)
        return (model, config)
    }

    /// Cheap probe: does this directory hold a VoiceChat bundle?
    public static func looksLikeVoiceChatBundle(at directory: URL) -> Bool {
        let cfg = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: cfg.path),
            let data = try? Data(contentsOf: cfg),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return false }
        return (json["model_type"] as? String) == "nemotron_voicechat"
    }
}
