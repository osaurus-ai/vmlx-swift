import Foundation
import MLX
import Cmlx
import CmlxGraphShim
#if canImport(Darwin)
import Darwin
#endif

/// Direct uncached reads into owned packed compute weights for every model family.
/// Does not modify the source file or manufacture model/dtype defaults.
enum ResidentSafetensorsReader {
    struct InvalidFile: LocalizedError {
        let detail: String
        var errorDescription: String? { "Resident safetensors reader: \(detail)" }
    }

    private struct Entry: Decodable {
        let dtype: String
        let shape: [Int]
        let data_offsets: [Int]
    }

    private struct Key: CodingKey {
        var stringValue: String
        var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
    }

    private struct Header: Decodable {
        var entries: [String: Entry] = [:]
        var metadata: [String: String] = [:]
        init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: Key.self)
            for key in values.allKeys {
                if key.stringValue == "__metadata__" {
                    // Existing safetensors readers treat null metadata as absent.
                    // Some Flash Next shards use null while others omit the key.
                    metadata = try values.decodeIfPresent([String: String].self, forKey: key) ?? [:]
                } else {
                    entries[key.stringValue] = try values.decode(Entry.self, forKey: key)
                }
            }
        }
    }

    private static let dtypes: [String: DType] = [
        "BOOL": .bool, "U8": .uint8, "U16": .uint16, "U32": .uint32, "U64": .uint64,
        "I8": .int8, "I16": .int16, "I32": .int32, "I64": .int64,
        "F16": .float16, "BF16": .bfloat16, "F32": .float32, "F64": .float64,
        "C64": .complex64, "F8_E4M3": .uint8, "F8_E8M0": .uint8,
    ]

    /// One owned MLX allocation, filled directly from the verified descriptor.
    static func loadRegion(file: FileHandle, offset: UInt64, length: Int,
                           shape: [Int], dtype: DType) throws -> MLXArray {
        try Task.checkCancellation()
        guard length > 0, shape.count <= 32,
              shape.allSatisfy({ $0 >= 0 && $0 <= Int(Int32.max) }) else {
            throw InvalidFile(detail: "invalid tensor span or shape")
        }
        var raw = mlx_array_new()
        var dimensions = shape.map(Int32.init)
        do {
            let status = try withError {
                vmlx_resident_file_array(&raw.ctx, file.fileDescriptor, offset, length,
                    &dimensions, Int32(dimensions.count), Int32(dtype.cmlxDtype.rawValue))
            }
            guard status == 0 else { throw InvalidFile(detail: "owned tensor read failed") }
        } catch {
            mlx_array_free(raw)
            throw error
        }
        let result = MLXArray(raw)
        try Task.checkCancellation()
        return result
    }

    static func loadRegion(url: URL, offset: UInt64, length: Int,
                           shape: [Int], dtype: DType) throws -> MLXArray {
        guard url.isFileURL else { throw InvalidFile(detail: "not a file URL") }
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        return try loadRegion(file: file, offset: offset, length: length, shape: shape, dtype: dtype)
    }

    /// Shared by base models and speculative companion models. Linux keeps the
    /// stock allocator-owned loader; Darwin also avoids duplicate filesystem cache.
    static func loadWeights(url: URL, excludingKeys: Set<String> = []) throws -> ([String: MLXArray], [String: String]) {
        #if canImport(Darwin)
        return try load(url: url, excludingKeys: excludingKeys)
        #else
        return try MLX.loadArraysAndMetadata(url: url, excludingKeys: excludingKeys)
        #endif
    }

    static func load(url: URL, excludingKeys: Set<String>) throws -> ([String: MLXArray], [String: String]) {
        #if canImport(Darwin)
        try Task.checkCancellation()
        guard url.isFileURL else { throw InvalidFile(detail: "not a file URL") }
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        guard fcntl(file.fileDescriptor, F_NOCACHE, 1) == 0 else {
            throw InvalidFile(detail: "uncached I/O unavailable (errno \(errno))")
        }
        var status = stat()
        guard fstat(file.fileDescriptor, &status) == 0, status.st_size >= 8 else {
            throw InvalidFile(detail: "missing or short file")
        }
        let prefix = try readExactly(file, count: 8, fileSize: Int(status.st_size))
        let length = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (8 * $1.offset) }
        guard length > 0, length <= 64 * 1024 * 1024,
            length <= UInt64(status.st_size - 8) else {
            throw InvalidFile(detail: "header length out of bounds")
        }
        let header = try JSONDecoder().decode(Header.self, from: readExactly(file, count: Int(length), fileSize: Int(status.st_size)))
        let dataStart = 8 + Int(length)
        let dataLength = Int(status.st_size) - dataStart
        // Validate every entry before allocating any tensors, including excluded
        // entries, so malformed shapes never reach MLX's nonthrowing constructors.
        for (name, entry) in header.entries {
            guard let dtype = dtypes[entry.dtype], entry.data_offsets.count == 2 else {
                throw InvalidFile(detail: "unsupported dtype or offsets for \(name)")
            }
            let start = entry.data_offsets[0], end = entry.data_offsets[1]
            guard start >= 0, end >= start, end <= dataLength else {
                throw InvalidFile(detail: "out-of-bounds tensor \(name)")
            }
            var elements = 1
            for dimension in entry.shape {
                guard dimension >= 0, dimension <= Int(Int32.max) else {
                    throw InvalidFile(detail: "shape outside MLX dimension range for \(name)")
                }
                let product = elements.multipliedReportingOverflow(by: dimension)
                guard !product.overflow else { throw InvalidFile(detail: "shape overflow for \(name)") }
                elements = product.partialValue
            }
            let size = elements.multipliedReportingOverflow(by: dtype.size)
            guard !size.overflow, size.partialValue == end - start else {
                throw InvalidFile(detail: "shape/byte mismatch for \(name)")
            }
        }
        let ordered = header.entries.sorted {
            if $0.value.data_offsets[0] == $1.value.data_offsets[0] {
                return $0.value.data_offsets[1] < $1.value.data_offsets[1]
            }
            return $0.value.data_offsets[0] < $1.value.data_offsets[0]
        }
        var previousEnd = 0
        for (_, entry) in ordered {
            guard entry.data_offsets[0] == previousEnd else {
                throw InvalidFile(detail: "overlapping or noncontiguous payload")
            }
            previousEnd = entry.data_offsets[1]
        }
        guard previousEnd == dataLength else { throw InvalidFile(detail: "unindexed payload bytes") }
        var arrays: [String: MLXArray] = [:]
        for (name, entry) in ordered where !excludingKeys.contains(name) {
            try Task.checkCancellation()
            try autoreleasepool {
                let count = entry.data_offsets[1] - entry.data_offsets[0]
                let array = count == 0
                    ? MLXArray.zeros(entry.shape, dtype: dtypes[entry.dtype]!)
                    : try loadRegion(file: file, offset: UInt64(dataStart + entry.data_offsets[0]),
                                     length: count, shape: entry.shape, dtype: dtypes[entry.dtype]!)
                arrays[name] = array
            }
        }
        var finalStatus = stat()
        guard fstat(file.fileDescriptor, &finalStatus) == 0,
              finalStatus.st_dev == status.st_dev, finalStatus.st_ino == status.st_ino,
              finalStatus.st_size == status.st_size,
              finalStatus.st_mtimespec.tv_sec == status.st_mtimespec.tv_sec,
              finalStatus.st_mtimespec.tv_nsec == status.st_mtimespec.tv_nsec,
              finalStatus.st_ctimespec.tv_sec == status.st_ctimespec.tv_sec,
              finalStatus.st_ctimespec.tv_nsec == status.st_ctimespec.tv_nsec else {
            throw InvalidFile(detail: "source changed while loading shard")
        }
        return (arrays, header.metadata)
        #else
        throw InvalidFile(detail: "uncached owned reader requires Darwin")
        #endif
    }

    private static func readExactly(_ file: FileHandle, count: Int, fileSize: Int) throws -> Data {
        #if canImport(Darwin)
        let offset = try file.offset()
        guard offset <= UInt64(fileSize), count >= 0, count <= fileSize - Int(offset) else {
            throw InvalidFile(detail: "read range out of bounds")
        }
        if count == 0 { return Data() }
        let start = Int(offset)
        let end = start + count
        let page = Int(getpagesize())
        // Unaligned large reads can populate the filesystem cache even with
        // F_NOCACHE. Read page-aligned ranges and retain only the requested bytes.
        // This is I/O alignment, not a rewrite of the safetensors bundle.
        var position = start - start % page
        let padding = end % page == 0 ? 0 : min(page - end % page, fileSize - end)
        let physicalEnd = end + padding
        try file.seek(toOffset: UInt64(position))
        var result = Data()
        result.reserveCapacity(count)
        while position < physicalEnd {
            try Task.checkCancellation()
            let remaining = physicalEnd - position
            // Keep a partial EOF page separate from the aligned bulk read.
            let amount = remaining < page ? remaining : min(remaining / page * page, 4 * 1024 * 1024)
            guard let chunk = try file.read(upToCount: amount), chunk.count == amount else {
                throw InvalidFile(detail: "short aligned read")
            }
            let lower = max(start - position, 0)
            let upper = min(end - position, amount)
            if lower < upper { result.append(chunk[lower..<upper]) }
            position += amount
        }
        try file.seek(toOffset: UInt64(end))
        return result
        #else
        let offset = try file.offset()
        guard offset <= UInt64(fileSize), count >= 0, count <= fileSize - Int(offset) else {
            throw InvalidFile(detail: "read range out of bounds")
        }
        var result = Data()
        result.reserveCapacity(count)
        while result.count < count {
            try Task.checkCancellation()
            guard let chunk = try file.read(upToCount: min(count - result.count, 4 * 1024 * 1024)),
                !chunk.isEmpty else { throw InvalidFile(detail: "short owned read") }
            result.append(chunk)
        }
        return result
        #endif
    }
}
