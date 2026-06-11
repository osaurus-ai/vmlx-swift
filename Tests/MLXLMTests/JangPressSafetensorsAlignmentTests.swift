// Copyright © 2026 Jinho Jang. All rights reserved.

import Foundation
import Testing
@testable import MLXLMCommon

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

@Suite("JangPress safetensors alignment overlay")
struct JangPressSafetensorsAlignmentTests {

    @Test("routed safetensors with unaligned data base is rewritten into aligned overlay")
    func routedUnalignedBundleGetsAlignedOverlay() throws {
        let bundle = try Self.makeBundleDir()
        let cache = try Self.makeBundleDir(prefix: "jpress-align-cache")
        defer {
            try? FileManager.default.removeItem(at: bundle)
            try? FileManager.default.removeItem(at: cache)
        }
        try Self.writeSafetensors(
            at: bundle.appendingPathComponent("model.safetensors"),
            tensorName: "model.layers.0.mlp.experts.0.gate_proj.weight")
        try Data("{}".utf8).write(to: bundle.appendingPathComponent("config.json"))

        let priorCache = getenv("JANGPRESS_ALIGN_CACHE_DIR").map { String(cString: $0) }
        let priorAlign = getenv("JANGPRESS_ALIGN_SAFETENSORS").map { String(cString: $0) }
        setenv("JANGPRESS_ALIGN_CACHE_DIR", cache.path, 1)
        setenv("JANGPRESS_ALIGN_SAFETENSORS", "1", 1)
        defer { Self.restoreEnv("JANGPRESS_ALIGN_CACHE_DIR", priorCache) }
        defer { Self.restoreEnv("JANGPRESS_ALIGN_SAFETENSORS", priorAlign) }

        let prepared = try JangPressPrestacker.prepareBundleIfNeeded(
            originalURL: bundle,
            enabled: true)

        #expect(prepared.path != bundle.path)
        let alignedShard = prepared.appendingPathComponent("model.safetensors")
        #expect(FileManager.default.fileExists(atPath: alignedShard.path))
        #expect(FileManager.default.fileExists(
            atPath: prepared.appendingPathComponent("config.json").path))

        let header = try Self.readHeader(alignedShard)
        #expect(header.dataBase % 4096 == 0)
        for (_, item) in header.tensors {
            let dtype = item["dtype"] as! String
            let offsets = item["data_offsets"] as! [UInt64]
            let absolute = header.dataBase + offsets[0]
            #expect(absolute % UInt64(Self.dtypeAlignment(dtype)) == 0)
        }

        let payload = try Data(contentsOf: alignedShard)
        #expect(payload.suffix(6) == Data([1, 2, 3, 4, 5, 6]))
    }

    @Test("dense unaligned safetensors are not copied by JangPress aligner")
    func denseUnalignedBundleIsLeftAlone() throws {
        let bundle = try Self.makeBundleDir()
        let cache = try Self.makeBundleDir(prefix: "jpress-align-cache")
        defer {
            try? FileManager.default.removeItem(at: bundle)
            try? FileManager.default.removeItem(at: cache)
        }
        try Self.writeSafetensors(
            at: bundle.appendingPathComponent("model.safetensors"),
            tensorName: "model.layers.0.input_layernorm.weight")

        let priorCache = getenv("JANGPRESS_ALIGN_CACHE_DIR").map { String(cString: $0) }
        setenv("JANGPRESS_ALIGN_CACHE_DIR", cache.path, 1)
        defer { Self.restoreEnv("JANGPRESS_ALIGN_CACHE_DIR", priorCache) }

        let prepared = try JangPressPrestacker.prepareBundleIfNeeded(
            originalURL: bundle,
            enabled: true)
        #expect(prepared.path == bundle.resolvingSymlinksInPath().path)
    }

    @Test("alignment overlay is opt-in when env is unset")
    func alignmentOverlayIsOffByDefault() throws {
        let bundle = try Self.makeBundleDir()
        let cache = try Self.makeBundleDir(prefix: "jpress-align-cache")
        defer {
            try? FileManager.default.removeItem(at: bundle)
            try? FileManager.default.removeItem(at: cache)
        }
        try Self.writeSafetensors(
            at: bundle.appendingPathComponent("model.safetensors"),
            tensorName: "model.layers.0.mlp.experts.0.gate_proj.weight")

        let priorCache = getenv("JANGPRESS_ALIGN_CACHE_DIR").map { String(cString: $0) }
        let priorAlign = getenv("JANGPRESS_ALIGN_SAFETENSORS").map { String(cString: $0) }
        setenv("JANGPRESS_ALIGN_CACHE_DIR", cache.path, 1)
        unsetenv("JANGPRESS_ALIGN_SAFETENSORS")
        defer { Self.restoreEnv("JANGPRESS_ALIGN_CACHE_DIR", priorCache) }
        defer { Self.restoreEnv("JANGPRESS_ALIGN_SAFETENSORS", priorAlign) }

        let prepared = try JangPressPrestacker.prepareBundleIfNeeded(
            originalURL: bundle,
            enabled: true)
        #expect(prepared.path == bundle.resolvingSymlinksInPath().path)
    }

    private static func makeBundleDir(prefix: String = "jpress-align") throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(prefix)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private static func writeSafetensors(at url: URL, tensorName: String) throws {
        let header: [String: Any] = [
            tensorName: [
                "dtype": "U32",
                "shape": [1],
                "data_offsets": [0, 4],
            ],
            "model.layers.0.post_attention_layernorm.weight": [
                "dtype": "BF16",
                "shape": [1],
                "data_offsets": [4, 6],
            ],
        ]
        var headerData = try JSONSerialization.data(
            withJSONObject: header,
            options: [.sortedKeys])
        while (8 + headerData.count) % 4 == 0 {
            headerData.append(0x20)
        }

        var data = Data()
        var headerLength = UInt64(headerData.count).littleEndian
        data.append(contentsOf: withUnsafeBytes(of: &headerLength) { Array($0) })
        data.append(headerData)
        data.append(contentsOf: [1, 2, 3, 4, 5, 6])
        try data.write(to: url)
    }

    private struct Header {
        var dataBase: UInt64
        var tensors: [String: [String: Any]]
    }

    private static func readHeader(_ url: URL) throws -> Header {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let prefix = try handle.read(upToCount: 8) ?? Data()
        let headerLength = prefix.withUnsafeBytes {
            UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
        }
        let headerData = try handle.read(upToCount: Int(headerLength)) ?? Data()
        let json = try JSONSerialization.jsonObject(with: headerData)
        let object = json as! [String: Any]
        var tensors: [String: [String: Any]] = [:]
        for (key, value) in object where key != "__metadata__" {
            var item = value as! [String: Any]
            if let offsets = item["data_offsets"] as? [NSNumber] {
                item["data_offsets"] = offsets.map { $0.uint64Value }
            }
            tensors[key] = item
        }
        return Header(dataBase: 8 + headerLength, tensors: tensors)
    }

    private static func dtypeAlignment(_ dtype: String) -> Int {
        switch dtype {
        case "F64", "I64", "U64", "C64": return 8
        case "F32", "I32", "U32": return 4
        case "F16", "BF16", "I16", "U16": return 2
        default: return 1
        }
    }

    private static func restoreEnv(_ key: String, _ value: String?) {
        if let value {
            setenv(key, value, 1)
        } else {
            unsetenv(key)
        }
    }
}
