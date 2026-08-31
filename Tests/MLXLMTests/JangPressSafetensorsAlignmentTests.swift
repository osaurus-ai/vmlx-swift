// Copyright © 2026 Jinho Jang. All rights reserved.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("Safetensors storage heal-on-load", .serialized)
struct JangPressSafetensorsAlignmentTests {
    @Test("dense and routed tensors heal in place with identical named payloads")
    func healsAllModelFamiliesInPlace() throws {
        let bundle = try Self.makeDirectory()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let shard = bundle.appendingPathComponent("model.safetensors")
        try Self.writeUnalignedFixture(shard)
        let before = try Self.readFixture(shard)

        let result = SafetensorsStorageHealer.healBundle(
            at: bundle, configuration: Self.configuration())

        #expect(result.scannedShards == 1)
        #expect(result.healedShards == 1)
        #expect(result.fallbackShards == 0)
        let after = try Self.readFixture(shard)
        #expect(after.metadata == before.metadata)
        #expect(after.payloads == before.payloads)
        #expect(after.unalignedCount == 0)
        #expect(try Self.temporaryFiles(in: bundle).isEmpty)
    }

    @Test("second load is a no-op after the one-time heal")
    func secondLoadDoesNotRewrite() throws {
        let bundle = try Self.makeDirectory()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let shard = bundle.appendingPathComponent("model.safetensors")
        try Self.writeUnalignedFixture(shard)
        _ = SafetensorsStorageHealer.healBundle(
            at: bundle, configuration: Self.configuration())
        let first = try Data(contentsOf: shard)

        let result = SafetensorsStorageHealer.healBundle(
            at: bundle, configuration: Self.configuration())

        #expect(result.scannedShards == 1)
        #expect(result.healedShards == 0)
        #expect(result.fallbackShards == 0)
        #expect(try Data(contentsOf: shard) == first)
    }

    @Test("insufficient transient disk advises and leaves the original intact")
    func diskShortageFallsBackWithoutMutation() throws {
        let bundle = try Self.makeDirectory()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let shard = bundle.appendingPathComponent("model.safetensors")
        try Self.writeUnalignedFixture(shard)
        let before = try Data(contentsOf: shard)
        var configuration = Self.configuration()
        configuration.availableBytes = { _ in 0 }

        let result = SafetensorsStorageHealer.healBundle(
            at: bundle, configuration: configuration)

        #expect(result.healedShards == 0)
        #expect(result.fallbackShards == 1)
        #expect(try Data(contentsOf: shard) == before)
        #expect(try Self.temporaryFiles(in: bundle).isEmpty)
    }

    @Test("an interrupted stream never replaces the original shard")
    func interruptedWriteLeavesOriginalIntact() throws {
        let bundle = try Self.makeDirectory()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let shard = bundle.appendingPathComponent("model.safetensors")
        try Self.writeUnalignedFixture(shard)
        let before = try Data(contentsOf: shard)
        var configuration = Self.configuration()
        configuration.failAfterCopiedBytes = 1

        let result = SafetensorsStorageHealer.healBundle(
            at: bundle, configuration: configuration)

        #expect(result.healedShards == 0)
        #expect(result.fallbackShards == 1)
        #expect(try Data(contentsOf: shard) == before)
        #expect(try Self.temporaryFiles(in: bundle).isEmpty)
    }

    @Test("Hugging Face hash-addressed snapshots are never rewritten")
    func hashAddressedStorageIsExempt() throws {
        let root = try Self.makeDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let snapshot = root.appendingPathComponent(
            ".cache/huggingface/hub/models--owner--model/snapshots/revision")
        try FileManager.default.createDirectory(
            at: snapshot, withIntermediateDirectories: true)
        let shard = snapshot.appendingPathComponent("model.safetensors")
        try Self.writeUnalignedFixture(shard)
        let before = try Data(contentsOf: shard)

        let result = SafetensorsStorageHealer.healBundle(
            at: snapshot, configuration: Self.configuration())

        #expect(result.scannedShards == 0)
        #expect(result.healedShards == 0)
        #expect(try Data(contentsOf: shard) == before)
    }

    @Test("symlinked blob shards are skipped even in a plain directory")
    func symlinkShardIsExempt() throws {
        let root = try Self.makeDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let bundle = root.appendingPathComponent("bundle")
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        let blob = root.appendingPathComponent("blob.safetensors")
        try Self.writeUnalignedFixture(blob)
        let before = try Data(contentsOf: blob)
        try FileManager.default.createSymbolicLink(
            at: bundle.appendingPathComponent("model.safetensors"),
            withDestinationURL: blob)

        let result = SafetensorsStorageHealer.healBundle(
            at: bundle, configuration: Self.configuration())

        #expect(result.scannedShards == 1)
        #expect(result.skippedShards == 1)
        #expect(result.healedShards == 0)
        #expect(try Data(contentsOf: blob) == before)
    }

    @Test("the production JangPress hook heals before returning the mmap directory")
    func productionLoadHookHealsByDefault() throws {
        let bundle = try Self.makeDirectory()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let shard = bundle.appendingPathComponent("model.safetensors")
        try Self.writeUnalignedFixture(shard)
        let saved = Self.saveAndUnset([
            "MLXPRESS_HEAL_SAFETENSORS", "JANGPRESS_HEAL_SAFETENSORS",
            "MLXPRESS_PRESTACK", "JANGPRESS_PRESTACK",
            "MLXPRESS_ALIGN_SAFETENSORS", "JANGPRESS_ALIGN_SAFETENSORS",
        ])
        defer { Self.restore(saved) }

        let prepared = try JangPressPrestacker.prepareBundleIfNeeded(
            originalURL: bundle, enabled: true)

        #expect(prepared.standardizedFileURL == bundle.standardizedFileURL)
        #expect(try Self.readFixture(shard).unalignedCount == 0)
    }

    @Test("resident/non-mmap loads do not mutate model storage")
    func disabledMmapDoesNotHeal() throws {
        let bundle = try Self.makeDirectory()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let shard = bundle.appendingPathComponent("model.safetensors")
        try Self.writeUnalignedFixture(shard)
        let before = try Data(contentsOf: shard)

        _ = try JangPressPrestacker.prepareBundleIfNeeded(
            originalURL: bundle, enabled: false)

        #expect(try Data(contentsOf: shard) == before)
    }

    private static func configuration() -> SafetensorsStorageHealer.Configuration {
        SafetensorsStorageHealer.Configuration(
            environment: [:],
            availableBytes: { _ in UInt64.max },
            failAfterCopiedBytes: nil,
            logger: { _ in })
    }

    private static func makeDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-heal-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        return directory
    }

    private static func writeUnalignedFixture(_ url: URL) throws {
        let header: [String: Any] = [
            "__metadata__": ["format": "pt", "proof": "payloads-unchanged"],
            "model.layers.0.input_layernorm.weight": [
                "dtype": "BF16", "shape": [1], "data_offsets": [0, 2],
            ],
            "model.layers.0.mlp.experts.0.gate_proj.weight": [
                "dtype": "U32", "shape": [1], "data_offsets": [2, 6],
            ],
        ]
        var encoded = try JSONSerialization.data(
            withJSONObject: header, options: [.sortedKeys])
        while (8 + encoded.count) % 2 != 0 || (8 + encoded.count + 2) % 4 == 0 {
            encoded.append(0x20)
        }

        var output = Data()
        var length = UInt64(encoded.count).littleEndian
        output.append(contentsOf: withUnsafeBytes(of: &length) { Array($0) })
        output.append(encoded)
        output.append(contentsOf: [0xA1, 0xA2, 0xB1, 0xB2, 0xB3, 0xB4])
        try output.write(to: url)
    }

    private struct FixtureRead {
        var metadata: [String: String]
        var payloads: [String: Data]
        var unalignedCount: Int
    }

    private static func readFixture(_ url: URL) throws -> FixtureRead {
        let data = try Data(contentsOf: url)
        let headerLength = data.prefix(8).withUnsafeBytes {
            UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
        }
        let dataBase = 8 + Int(headerLength)
        let headerData = data.subdata(in: 8 ..< dataBase)
        let object = try JSONSerialization.jsonObject(with: headerData) as! [String: Any]
        let metadata = object["__metadata__"] as! [String: String]
        var payloads: [String: Data] = [:]
        var unaligned = 0
        for (name, raw) in object where name != "__metadata__" {
            let descriptor = raw as! [String: Any]
            let offsets = (descriptor["data_offsets"] as! [NSNumber]).map(\.intValue)
            let dtype = descriptor["dtype"] as! String
            let alignment = dtype == "U32" ? 4 : 2
            if (dataBase + offsets[0]) % alignment != 0 { unaligned += 1 }
            payloads[name] = data.subdata(
                in: (dataBase + offsets[0]) ..< (dataBase + offsets[1]))
        }
        return FixtureRead(
            metadata: metadata, payloads: payloads, unalignedCount: unaligned)
    }

    private static func temporaryFiles(in directory: URL) throws -> [URL] {
        try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        )
        .filter { $0.lastPathComponent.contains(".vmlx-heal-") }
    }

    private static func saveAndUnset(_ keys: [String]) -> [String: String?] {
        var result: [String: String?] = [:]
        for key in keys {
            result[key] = getenv(key).map { String(cString: $0) }
            unsetenv(key)
        }
        return result
    }

    private static func restore(_ values: [String: String?]) {
        for (key, value) in values {
            if let value { setenv(key, value, 1) } else { unsetenv(key) }
        }
    }
}
