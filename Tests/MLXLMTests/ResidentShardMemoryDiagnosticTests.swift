import Foundation
import Darwin
import MLX
import Testing
@testable import MLXLMCommon

/// Opt-in component diagnostic, not a cold full-model load or speed benchmark.
@Suite(.serialized)
struct ResidentShardMemoryDiagnosticTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["VMLX_RESIDENT_REAL_SHARD"] != nil))
    func realShardSubsetLifecycle() throws {
        try MLXMetalTestLock.withLock {
            let path = try #require(ProcessInfo.processInfo.environment["VMLX_RESIDENT_REAL_SHARD"])
            let url = URL(fileURLWithPath: path)
            let input = try FileHandle(forReadingFrom: url)
            defer { try? input.close() }
            let prefix = try #require(try input.read(upToCount: 8))
            try #require(prefix.count == 8)
            let length = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (8 * $1.offset) }
            try #require(length < 16 * 1024 * 1024)
            let data = try #require(try input.read(upToCount: Int(length)))
            let header = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
            var selected = Set<String>()
            var bytes = 0
            var bytesByType: [String: Int] = [:]
            let types: [String: DType] = ["U32": .uint32, "F16": .float16, "BF16": .bfloat16]
            let orderedKeys = header.keys.sorted {
                let a = (header[$0] as? [String: Any])?["dtype"] as? String ?? ""
                let b = (header[$1] as? [String: Any])?["dtype"] as? String ?? ""
                return a == b ? $0 < $1 : a < b
            }
            for key in orderedKeys {
                guard !key.contains("ngram"), !key.contains("layer_multipliers"), !key.contains("mtp"),
                    let entry = header[key] as? [String: Any],
                    let dtype = entry["dtype"] as? String, types[dtype] != nil,
                    let offsets = entry["data_offsets"] as? [Int], offsets.count == 2 else { continue }
                let size = offsets[1] - offsets[0]
                guard size > 0, size <= 256 * 1024 * 1024 - bytes,
                    size <= 80 * 1024 * 1024 - bytesByType[dtype, default: 0] else { continue }
                selected.insert(key)
                bytes += size
                bytesByType[dtype, default: 0] += size
            }
            try #require(!selected.isEmpty && bytes <= 256 * 1024 * 1024)
            let selectedTypes = Set(selected.compactMap { (header[$0] as? [String: Any])?["dtype"] as? String })
            try #require(selectedTypes == Set(types.keys))
            print("RESIDENT_REAL dtypes=\(selectedTypes.sorted())")
            print("RESIDENT_REAL file=\(path) bytes=\(bytes) keys=\(selected.sorted()) cache_temperature=unknown")
            if let arm = ProcessInfo.processInfo.environment["VMLX_RESIDENT_READ_ARM"] {
                try #require(arm == "mapped" || arm == "direct" || arm == "resident")
                func fileResidency(_ stage: String) throws {
                    var status = stat()
                    try #require(fstat(input.fileDescriptor, &status) == 0)
                    let size = Int(status.st_size)
                    let mapping = mmap(nil, size, PROT_NONE, MAP_SHARED, input.fileDescriptor, 0)
                    try #require(mapping != MAP_FAILED)
                    defer { munmap(mapping, size) }
                    let pageSize = Int(getpagesize())
                    var states = [CChar](repeating: 0, count: (size + pageSize - 1) / pageSize)
                    try #require(mincore(mapping, size, &states) == 0)
                    print("FILE_RESIDENCY stage=\(stage) pages=\(states.filter { ($0 & 1) != 0 }.count) pageSize=\(pageSize) mlxActive=\(MLX.Memory.activeMemory) mlxCache=\(MLX.Memory.cacheMemory)")
                }
                MLX.Memory.clearCache()
                try sample("\(arm)-start")
                try fileResidency("before")
                var values: [String: MLXArray] = [:]
                if arm == "resident" {
                    let excluded = Set(header.keys).subtracting(selected).subtracting(["__metadata__"])
                    (values, _) = try ResidentSafetensorsReader.load(url: url, excludingKeys: excluded)
                } else if arm == "mapped" {
                    try autoreleasepool {
                        let excluded = Set(header.keys).subtracting(selected).subtracting(["__metadata__"])
                        let (mapped, _) = try loadArraysAndMetadata(url: url, excludingKeys: excluded, exactTensorBuffers: true)
                        values = mapped.mapValues { $0 * 1 }
                        MLX.eval(Array(values.values))
                        try sample("mapped-input-and-owned-live")
                    }
                } else {
                    let file = try FileHandle(forReadingFrom: url)
                    defer { try? file.close() }
                    try #require(fcntl(file.fileDescriptor, F_NOCACHE, 1) == 0)
                    for key in selected.sorted() {
                        try autoreleasepool {
                            let entry = try #require(header[key] as? [String: Any])
                            let offsets = try #require(entry["data_offsets"] as? [Int])
                            let shape = try #require(entry["shape"] as? [Int])
                            let name = try #require(entry["dtype"] as? String)
                            let dtype = try #require(types[name])
                            try file.seek(toOffset: 8 + length + UInt64(offsets[0]))
                            let data = try #require(try file.read(upToCount: offsets[1] - offsets[0]))
                            try #require(data.count == offsets[1] - offsets[0])
                            values[key] = MLXArray(data, shape, dtype: dtype)
                            MLX.eval(values[key]!)
                        }
                    }
                }
                #expect(Set(values.keys) == selected)
                #expect(values.values.reduce(0) { $0 + $1.nbytes } == bytes)
                try sample("\(arm)-owned-live")
                try fileResidency("owned-live")
                MLX.Memory.clearCache()
                try sample("\(arm)-owned-live-cache-cleared")
                values.removeAll()
                MLX.Memory.clearCache()
                try sample("\(arm)-released")
                try fileResidency("released")
                return
            }
            let excluded = Set(header.keys).subtracting(selected).subtracting(["__metadata__"])
            MLX.Memory.clearCache()
            try sample("real-before-map")
            var owned: [String: MLXArray] = [:]
            try autoreleasepool {
                let (mapped, _) = try loadArraysAndMetadata(url: url, excludingKeys: excluded, exactTensorBuffers: true)
                #expect(Set(mapped.keys) == selected)
                try sample("real-mapped-before-eval")
                owned = mapped.mapValues { $0 * 1 }
                MLX.eval(Array(owned.values))
                try sample("real-owned-inputs-live")
                for key in selected {
                    #expect(MLX.all(owned[key]! .== mapped[key]!).item(Bool.self))
                }
            }
            try sample("real-mapped-released")
            MLX.Memory.clearCache()
            try sample("real-cache-cleared-owned-live")
            // Diagnostic alternative: independently open the same file for
            // uncached reads. This does not purge pre-existing cached pages.
            let start = ProcessInfo.processInfo.systemUptime
            try autoreleasepool {
                let (direct, _) = try ResidentSafetensorsReader.load(url: url, excludingKeys: excluded)
                #expect(Set(direct.keys) == selected)
                for key in selected.sorted() {
                    let array = try #require(direct[key])
                    #expect(MLX.all(array.reshaped([-1]).view(dtype: .uint8) .== owned[key]!.reshaped([-1]).view(dtype: .uint8)).item(Bool.self))
                }
            }
            print("RESIDENT_REAL uncached_read_and_parity_seconds=\(ProcessInfo.processInfo.systemUptime - start)")
            MLX.Memory.clearCache()
            try sample("real-uncached-read-parity-complete")
            owned.removeAll()
            MLX.Memory.clearCache()
            try sample("real-all-released")
        }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["VMLX_RESIDENT_SHARD_DIAGNOSTIC"] == "1"))
    func mappedToOwnedLifecycle() throws {
        try MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("resident-shard-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
            defer { try? FileManager.default.removeItem(at: directory) }
            let url = directory.appendingPathComponent("fixture.safetensors")
            // Four 64 MiB tensors, hard bounded before construction. The file is
            // newly written, so its filesystem cache starts warm, not cold.
            let count = 16 * 1024 * 1024
            try autoreleasepool {
                let arrays = Dictionary(uniqueKeysWithValues: (0..<4).map {
                    ("w\($0)", MLXArray.full([count], values: MLXArray(UInt32($0 + 1)), dtype: .uint32))
                })
                let rawURL = directory.appendingPathComponent("raw.safetensors")
                try MLX.save(arrays: arrays, url: rawURL)
                try alignFixture(from: rawURL, to: url)
                try FileManager.default.removeItem(at: rawURL)
            }
            MLX.Memory.clearCache()
            try sample("fixture-written-warm")
            var owned: [String: MLXArray] = [:]
            try autoreleasepool {
                let (mapped, _) = try loadArraysAndMetadata(url: url, excludingKeys: [], exactTensorBuffers: true)
                #expect(mapped.count == 4)
                try sample("mapped-before-eval")
                owned = mapped.mapValues { $0 * 1 }
                MLX.eval(Array(owned.values))
                try sample("owned-evaluated-inputs-in-scope")
            }
            try sample("mapped-scope-released")
            MLX.Memory.clearCache()
            try sample("allocator-cache-cleared-owned-live")
            for index in 0..<4 {
                let array = try #require(owned["w\(index)"])
                #expect(array.dtype == .uint32)
                #expect(array.size == count)
                #expect(MLX.all(array .== UInt32(index + 1)).item(Bool.self))
            }
            owned.removeAll()
            MLX.Memory.clearCache()
            try sample("all-arrays-released")
        }
    }

    private func alignFixture(from source: URL, to target: URL) throws {
        let input = try FileHandle(forReadingFrom: source)
        defer { try? input.close() }
        let prefix = try #require(try input.read(upToCount: 8))
        #expect(prefix.count == 8)
        let length = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (8 * $1.offset) }
        #expect(length < 1024 * 1024)
        let header = try #require(try input.read(upToCount: Int(length)))
        let padding = (8 - header.count % 8) % 8
        var paddedLength = UInt64(header.count + padding).littleEndian
        #expect(FileManager.default.createFile(atPath: target.path, contents: nil))
        let output = try FileHandle(forWritingTo: target)
        defer { try? output.close() }
        try withUnsafeBytes(of: &paddedLength) { try output.write(contentsOf: Data($0)) }
        try output.write(contentsOf: header)
        try output.write(contentsOf: Data(repeating: 0x20, count: padding))
        while let chunk = try input.read(upToCount: 1024 * 1024), !chunk.isEmpty {
            try output.write(contentsOf: chunk)
        }
    }

    private func sample(_ phase: String) throws {
        let memory = MLX.Memory.snapshot()
        print("RESIDENT_SHARD phase=\(phase) active=\(memory.activeMemory) cache=\(memory.cacheMemory)")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/vm_stat")
        let pipe = Pipe()
        process.standardOutput = pipe
        try process.run()
        let output = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        #expect(process.terminationStatus == 0)
        print(String(decoding: output, as: UTF8.self))
        let footprint = Process()
        footprint.executableURL = URL(fileURLWithPath: "/usr/bin/footprint")
        footprint.arguments = ["-p", String(ProcessInfo.processInfo.processIdentifier)]
        let footprintPipe = Pipe()
        footprint.standardOutput = footprintPipe
        try footprint.run()
        let footprintData = footprintPipe.fileHandleForReading.readDataToEndOfFile()
        footprint.waitUntilExit()
        try #require(footprint.terminationStatus == 0)
        print(String(decoding: footprintData, as: UTF8.self))
    }
}
