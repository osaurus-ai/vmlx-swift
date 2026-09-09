// Foundation-only proof of the production storage implementation. No MLX or model load.
import Foundation

@main
struct AlignmentRepairStandaloneProof {
    static func require(_ value: Bool, _ label: String) throws {
        if !value { throw NSError(domain: label, code: 1) }
    }

    static func fixture(_ shard: URL) throws {
        let header: [String: Any] = [
            "__metadata__": ["format": "pt", "mxtq_bits": "8"],
            "norm.weight": ["dtype": "BF16", "shape": [1], "data_offsets": [0, 2]],
            "experts.0.weight": ["dtype": "U32", "shape": [1], "data_offsets": [2, 6]],
            "dense.weight": ["dtype": "F32", "shape": [1], "data_offsets": [6, 10]],
        ]
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while (8 + json.count) % 8 != 0 { json.append(32) }
        var length = UInt64(json.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }
        data.append(json)
        data.append(contentsOf: Array(0..<10))
        try data.write(to: shard)
    }

    static func payloads(_ shard: URL) throws -> [String: Data] {
        let data = try Data(contentsOf: shard)
        let length = data.prefix(8).withUnsafeBytes { Int($0.loadUnaligned(as: UInt64.self)) }
        let base = 8 + length
        let header = try JSONSerialization.jsonObject(with: data[8..<base]) as! [String: Any]
        var result: [String: Data] = [:]
        for (name, value) in header where name != "__metadata__" {
            let tensor = value as! [String: Any]
            let offsets = tensor["data_offsets"] as! [Int]
            result[name] = data[(base + offsets[0])..<(base + offsets[1])]
        }
        result["__metadata__"] = try JSONSerialization.data(
            withJSONObject: header["__metadata__"]!, options: [.sortedKeys])
        return result
    }

    static func main() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("alignment-proof-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let shard = root.appendingPathComponent("model.safetensors")
        var config = SafetensorsStorageHealer.Configuration(
            environment: ["MLXPRESS_HEAL_SAFETENSORS": "1"],
            availableBytes: { _ in UInt64.max }, logger: { print($0) })
        try fixture(shard)
        let original = try Data(contentsOf: shard)
        let expected = try payloads(shard)
        let saved = ProcessInfo.processInfo.environment["MLXPRESS_HEAL_SAFETENSORS"]
        setenv("MLXPRESS_HEAL_SAFETENSORS", "1", 1)
        defer {
            if let saved { setenv("MLXPRESS_HEAL_SAFETENSORS", saved, 1) }
            else { unsetenv("MLXPRESS_HEAL_SAFETENSORS") }
        }
        SafetensorsStorageHealer.healBundleIfEligible(at: root)
        try require(try Data(contentsOf: shard) == original, "default denied with env opt-in")
        print("PASS default-denied production entry")
        SafetensorsStorageHealer.healBundleIfEligible(at: root, authorization: .directUserSend)
        try require(try payloads(shard) == expected && Data(contentsOf: shard) != original,
                    "direct-send production entry")
        print("PASS explicitly authorized production entry")
        try fixture(shard)

        var events: [AlignmentRepairProgress] = []
        config.progress = { events.append($0) }
        let result = SafetensorsStorageHealer.healBundle(at: root, configuration: config)
        try require(result.healedShards == 1, "repair")
        try require(try payloads(shard) == expected, "named payload and metadata preservation")
        try require(events.map(\.stage).contains(.verifying) && events.last?.stage == .installed, "progress stages")
        try require(events.last?.copiedBytes == 10 && events.last?.totalBytes == 10, "measured bytes")
        print("PASS repair + exact payload/metadata + real progress")
        let repaired = try Data(contentsOf: shard)
        events.removeAll()
        let repeatResult = SafetensorsStorageHealer.healBundle(at: root, configuration: config)
        try require(repeatResult.healedShards == 0 && events.isEmpty, "repeat silent no-op")
        try require(try Data(contentsOf: shard) == repaired, "repeat unchanged")
        print("PASS repeat/aligned no-op")

        for mode in ["disk", "interrupt", "cancel", "optout"] {
            try fixture(shard)
            var failure = config
            if mode == "disk" { failure.availableBytes = { _ in 0 } }
            if mode == "interrupt" { failure.failAfterCopiedBytes = 3 }
            if mode == "optout" { failure.environment = [:] }
            var cancelled = false
            if mode == "cancel" {
                failure.progress = { if $0.stage == .verifying { cancelled = true } }
                failure.checkCancellation = { if cancelled { throw CancellationError() } }
            }
            let outcome = SafetensorsStorageHealer.healBundle(at: root, configuration: failure)
            try require(outcome.healedShards == 0, mode)
            try require(try Data(contentsOf: shard) == original, "original preserved: \(mode)")
            let files = try FileManager.default.contentsOfDirectory(atPath: root.path)
            try require(!files.contains(where: { $0.contains(".vmlx-heal-") }), "temp cleanup")
            print("PASS \(mode) original preserved + temp cleanup")
        }
        try Data([0, 1, 2]).write(to: shard)
        let malformed = SafetensorsStorageHealer.healBundle(at: root, configuration: config)
        try require(malformed.fallbackShards == 1, "malformed refusal")
        try require(try Data(contentsOf: shard) == Data([0, 1, 2]), "malformed unchanged")
        print("PASS malformed refusal; storage-only proof complete")
        try fixture(shard)
        var changed = original
        changed.append(99)
        var sourceChange = config
        sourceChange.progress = {
            if $0.stage == .verifying { try! changed.write(to: shard) }
        }
        let changedResult = SafetensorsStorageHealer.healBundle(at: root, configuration: sourceChange)
        try require(changedResult.healedShards == 0, "source-change refused")
        try require(try Data(contentsOf: shard) == changed, "external writer not overwritten")
        print("PASS changed source is not replaced")
        let dtypes: [(String, Int)] = [
            ("C128", 16), ("F64", 8), ("I64", 8), ("U64", 8), ("C64", 8),
            ("F32", 4), ("I32", 4), ("U32", 4), ("F16", 2), ("BF16", 2),
            ("I16", 2), ("U16", 2), ("BOOL", 1), ("U8", 1), ("I8", 1),
            ("F8_E5M2", 1), ("F8_E4M3", 1), ("F8_E4M3FN", 1),
            ("F8_E4M3FNUZ", 1), ("F8_E5M2FNUZ", 1), ("F8_E8M0", 1),
        ]
        for (dtype, bytes) in dtypes {
            let header: [String: Any] = [
                "__metadata__": ["format": "pt"],
                "prefix": ["dtype": "U8", "shape": [1], "data_offsets": [0, 1]],
                "weight": ["dtype": dtype, "shape": [1], "data_offsets": [1, 1 + bytes]],
            ]
            var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
            while (8 + json.count) % 16 != 0 { json.append(32) }
            var length = UInt64(json.count).littleEndian
            var data = withUnsafeBytes(of: &length) { Data($0) }
            data.append(json)
            data.append(contentsOf: Array(repeating: UInt8(37), count: 1 + bytes))
            try data.write(to: shard)
            let before = try payloads(shard)
            let outcome = SafetensorsStorageHealer.healBundle(at: root, configuration: config)
            try require(outcome.healedShards == (bytes > 1 ? 1 : 0), "dtype \(dtype)")
            try require(try payloads(shard) == before, "dtype payload \(dtype)")
        }
        print("PASS 21 supported storage dtypes, including byte-aligned no-ops")
    }
}
