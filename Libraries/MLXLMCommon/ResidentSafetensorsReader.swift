import Foundation
import MLX
#if canImport(Darwin)
import Darwin
#endif

/// Uncached reads for models that already require owned compute weights.
/// Does not modify the source file or manufacture model/dtype defaults.
enum ResidentSafetensorsReader {
    static func shouldUse(requiresOwnedCompute: Bool, readerOverride: String?) -> Bool {
        #if canImport(Darwin)
        return requiresOwnedCompute && readerOverride != "mmap"
        #else
        return false
        #endif
    }
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
    ]

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
                try file.seek(toOffset: UInt64(dataStart + entry.data_offsets[0]))
                let data = try readExactly(file, count: entry.data_offsets[1] - entry.data_offsets[0], fileSize: Int(status.st_size))
                // Empty tensors are legal safetensors entries. Avoid the Data
                // initializer's forced baseAddress unwrap for a zero-byte buffer.
                let array = data.isEmpty
                    ? MLXArray.zeros(entry.shape, dtype: dtypes[entry.dtype]!)
                    : MLXArray(data, entry.shape, dtype: dtypes[entry.dtype]!)
                MLX.eval(array)
                arrays[name] = array
            }
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
        throw InvalidFile(detail: "uncached owned reader requires Darwin")
        #endif
    }
}
