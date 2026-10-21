// Copyright © 2026 Jinho Jang. All rights reserved.
import Foundation

/// Read-only compatibility for previously created stacked routed tensors.
/// This recognizes existing bundle files; it never writes or prepares overlays.
enum LegacyRoutedTensorLayout {
    private static let outputName = "jangpress-prestacked.safetensors"
    static func prestackedRoutedReplacementKeys(in directory: URL) throws -> Set<String> {
        let outputURL = directory.resolvingSymlinksInPath().appendingPathComponent(outputName)
        guard FileManager.default.fileExists(atPath: outputURL.path) else {
            return []
        }
        return Set(try readSafetensorsHeader(outputURL).tensors.keys)
    }

    static func prestackedReplacementKey(forPerExpertKey key: String) -> String? {
        matchPerExpertKey(key)?.outputKey
    }

    private struct HeaderRead {
        var dataBase: UInt64
        var metadata: [String: Any]?
        var tensors: [String: [String: Any]]
    }

    private static func readSafetensorsHeader(_ url: URL) throws -> HeaderRead {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let prefix = try handle.read(upToCount: 8) ?? Data()
        guard prefix.count == 8 else {
            throw NSError(
                domain: "LegacyRoutedTensorLayout", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "short safetensors header: \(url.path)"])
        }
        let headerLength = prefix.withUnsafeBytes {
            UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
        }
        let headerData = try handle.read(upToCount: Int(headerLength)) ?? Data()
        guard headerData.count == Int(headerLength) else {
            throw NSError(
                domain: "LegacyRoutedTensorLayout", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "truncated safetensors header: \(url.path)"])
        }
        let json = try JSONSerialization.jsonObject(with: headerData)
        guard let dict = json as? [String: Any] else {
            throw NSError(
                domain: "LegacyRoutedTensorLayout", code: 3,
                userInfo: [NSLocalizedDescriptionKey: "invalid safetensors JSON: \(url.path)"])
        }
        let metadata = dict["__metadata__"] as? [String: Any]
        var tensors: [String: [String: Any]] = [:]
        for (key, value) in dict where key != "__metadata__" {
            guard let item = value as? [String: Any] else { continue }
            var normalized = item
            if let shape = item["shape"] as? [NSNumber] {
                normalized["shape"] = shape.map { $0.intValue }
            }
            if let offsets = item["data_offsets"] as? [NSNumber] {
                normalized["data_offsets"] = offsets.map { $0.uint64Value }
            }
            tensors[key] = normalized
        }
        return HeaderRead(dataBase: 8 + headerLength, metadata: metadata, tensors: tensors)
    }

    private struct KeyMatch {
        var expert: Int
        var outputKey: String
    }

    private static func matchPerExpertKey(_ key: String) -> KeyMatch? {
        if let m = match(
            key,
            #"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.(tq_packed|tq_norms)$"#
        ) {
            return KeyMatch(
                expert: Int(m[2])!,
                outputKey:
                    "model.layers.\(m[1]).block_sparse_moe.switch_mlp.\(projName(m[3])).\(m[4])")
        }
        if let m = match(
            key,
            #"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(tq_packed|tq_norms)$"#
        ) {
            return KeyMatch(
                expert: Int(m[2])!,
                outputKey: "layers.\(m[1]).ffn.switch_mlp.\(projName(m[3])).\(m[4])")
        }
        if let m = match(
            key,
            #"^(language_model\.)?model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(w[123])\.(tq_packed|tq_norms)$"#
        ) {
            let root = m[1]
            return KeyMatch(
                expert: Int(m[3])!,
                outputKey: "\(root)model.layers.\(m[2]).mlp.switch_mlp.\(projName(m[4])).\(m[5])")
        }
        if let m = match(
            key,
            #"^(language_model\.)?model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(tq_packed|tq_norms)$"#
        ) {
            let root = m[1]
            return KeyMatch(
                expert: Int(m[3])!,
                outputKey: "\(root)model.layers.\(m[2]).mlp.switch_mlp.\(m[4]).\(m[5])")
        }
        if let m = match(
            key,
            #"^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up_proj|down_proj)\.(tq_packed|tq_norms)$"#
        ) {
            let fc = m[3] == "up_proj" ? "fc1" : "fc2"
            return KeyMatch(
                expert: Int(m[2])!,
                outputKey: "backbone.layers.\(m[1]).mixer.switch_mlp.\(fc).\(m[4])")
        }
        return nil
    }

    private static func isStackedRoutedKey(_ key: String) -> Bool {
        key.contains(".switch_mlp.") && (key.hasSuffix(".tq_packed") || key.hasSuffix(".tq_norms"))
    }

    private static func projName(_ raw: String) -> String {
        switch raw {
        case "w1": return "gate_proj"
        case "w2": return "down_proj"
        case "w3": return "up_proj"
        default: return raw
        }
    }

    private static func match(_ value: String, _ pattern: String) -> [String]? {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return nil }
        let ns = value as NSString
        let range = NSRange(location: 0, length: ns.length)
        guard let m = re.firstMatch(in: value, range: range) else { return nil }
        var captures: [String] = []
        for i in 0 ..< m.numberOfRanges {
            let r = m.range(at: i)
            captures.append(r.location == NSNotFound ? "" : ns.substring(with: r))
        }
        return captures
    }

}
