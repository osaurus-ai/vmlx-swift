import Foundation
@preconcurrency import MLX
#if canImport(CryptoKit)
import CryptoKit
#else
import Crypto
#endif

public struct FluxAdapter: Sendable, Hashable, Codable {
    public enum Target: String, Sendable, Codable { case transformer, textEncoder, both }
    public var url: URL
    public var scale: Float
    public var target: Target
    public init(url: URL, scale: Float = 1, target: Target = .transformer) {
        self.url = url; self.scale = scale; self.target = target
    }
}

/// Installed only by a serialized engine operation. Linear layers capture this
/// model-owned context when constructed; base packed weights are never changed.
public enum FluxAdapterScope {
    @TaskLocal public static var current: FluxAdapterState?
}

public final class FluxAdapterState: @unchecked Sendable {
    private struct Pair {
        let down: MLXArray
        let up: MLXArray
        let scale: Float
    }
    private struct Shape { let input: Int; let output: Int }
    private var shapes: [String: Shape] = [:]
    private var active: [String: [Pair]] = [:]
    private var cached: [String: [Pair]] = [:]
    private var cachedIdentity: [String] = []
    public private(set) var loadCount = 0
    public var identity: [String] { active.isEmpty ? [] : cachedIdentity }

    public init() {}
    public func register(name: String, input: Int, output: Int) {
        shapes[name] = Shape(input: input, output: output)
    }

    public func clear() { active = [:] }

    public func install(_ adapters: [FluxAdapter], loading: (String) -> Void) throws {
        // Zero scale deliberately takes the untouched base path, including no
        // adapter matmuls, dtype casts or file loads.
        guard adapters.allSatisfy({ $0.scale.isFinite }) else {
            throw FluxError.invalidField(field: "adapters.scale", value: "non-finite", allowed: "finite scales")
        }
        let effective = adapters.filter { $0.scale != 0 }
        guard !effective.isEmpty else { active = [:]; return }
        let identities = try effective.map { try Self.digest($0.url) + ":\($0.scale.bitPattern):\($0.target.rawValue)" }
        if identities == cachedIdentity { active = cached; return }
        var next: [String: [Pair]] = [:]
        for (adapter, identity) in zip(effective, identities) {
            loading(adapter.url.lastPathComponent)
            let arrays = try MLX.loadArrays(url: adapter.url)
            guard identity.hasPrefix(try Self.digest(adapter.url) + ":") else {
                throw FluxError.invalidRequest("adapter changed while loading: \(adapter.url.lastPathComponent)")
            }
            var consumed: Set<String> = []
            var pairs = 0
            for key in arrays.keys.sorted() {
                let suffix: String
                let upSuffix: String
                if key.hasSuffix(".lora_A.weight") { suffix = ".lora_A.weight"; upSuffix = ".lora_B.weight" }
                else if key.hasSuffix(".lora_down.weight") { suffix = ".lora_down.weight"; upSuffix = ".lora_up.weight" }
                else { continue }
                let prefix = String(key.dropLast(suffix.count))
                let upKey = prefix + upSuffix
                guard let down = arrays[key], let up = arrays[upKey], down.ndim == 2, up.ndim == 2,
                    down.dim(0) > 0, up.dim(1) == down.dim(0),
                    down.dtype.isFloatingPoint, up.dtype.isFloatingPoint else {
                    throw FluxError.invalidRequest("invalid adapter pair \(prefix)")
                }
                let candidates = Self.names(prefix, target: adapter.target, available: shapes.keys)
                guard candidates.count == 1, let name = candidates.first, let shape = shapes[name] else {
                    throw FluxError.invalidField(field: "adapters.target", value: prefix,
                        allowed: "one registered linear layer in the selected target")
                }
                guard down.dim(1) == shape.input, up.dim(0) == shape.output else {
                    throw FluxError.invalidRequest("adapter dimensions do not match \(name)")
                }
                var scale = adapter.scale
                if let alpha = arrays[prefix + ".alpha"] {
                    guard alpha.size == 1 else { throw FluxError.invalidRequest("adapter alpha must be scalar") }
                    let value = try alpha.asType(.float32).asArrayChecked(Float.self)[0]
                    guard value.isFinite else { throw FluxError.invalidRequest("adapter alpha must be finite") }
                    scale *= value / Float(down.dim(0))
                    consumed.insert(prefix + ".alpha")
                }
                guard scale.isFinite else { throw FluxError.invalidRequest("adapter scale overflows") }
                next[name, default: []].append(Pair(down: down, up: up, scale: scale))
                consumed.insert(key); consumed.insert(upKey); pairs += 1
            }
            guard pairs > 0, consumed.count == arrays.count else {
                throw FluxError.invalidRequest("adapter contains unsupported or unpaired tensors: \(adapter.url.lastPathComponent)")
            }
            loadCount += 1
        }
        cachedIdentity = identities; cached = next; active = next
    }

    public func apply(name: String, input: MLXArray, output: MLXArray) -> MLXArray {
        guard let pairs = active[name] else { return output }
        var result = output
        for pair in pairs {
            let delta = matmul(matmul(input.asType(.float32), pair.down.asType(.float32).T), pair.up.asType(.float32).T)
            result = result + (delta * pair.scale).asType(result.dtype)
        }
        return result
    }

    private static func names(_ raw: String, target: FluxAdapter.Target, available: Dictionary<String, Shape>.Keys) -> [String] {
        let prefix = raw.replacingOccurrences(of: "base_model.model.", with: "")
        return available.filter { name in
            let text = name.hasPrefix("text_encoder.") || name.hasPrefix("text_encoder_2.")
            let transformer = name.hasPrefix("transformer.") || name.hasPrefix("unconditional_transformer.")
            guard (target == .both && (text || transformer)) || (target == .textEncoder && text)
                || (target == .transformer && transformer) else { return false }
            if name == prefix { return true }
            if transformer && name == "transformer." + prefix { return true }
            if text && name == "text_encoder." + prefix { return true }
            let kohya = name.hasPrefix("transformer.")
                ? "lora_unet_" + name.dropFirst("transformer.".count).replacingOccurrences(of: ".", with: "_")
                : "lora_te_" + name.dropFirst("text_encoder.".count).replacingOccurrences(of: ".", with: "_")
            return kohya == prefix
        }
    }

    private static func digest(_ url: URL) throws -> String {
        guard url.isFileURL else { throw FluxError.invalidRequest("adapters require local files") }
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var hash = SHA256()
        while let data = try file.read(upToCount: 1024 * 1024), !data.isEmpty { hash.update(data: data) }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
