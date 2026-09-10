import Foundation
import MLX
#if canImport(Darwin)
import Darwin
#endif

/// Diagnostic opt-in for models that already require owned compute weights.
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
                    metadata = try values.decode([String: String].self, forKey: key)
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
        let prefix = try readExactly(file, count: 8)
        let length = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (8 * $1.offset) }
        guard length > 0, length <= 64 * 1024 * 1024,
            length <= UInt64(status.st_size - 8) else {
            throw InvalidFile(detail: "header length out of bounds")
        }
        let header = try JSONDecoder().decode(Header.self, from: readExactly(file, count: Int(length)))
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
                guard dimension >= 0 else { throw InvalidFile(detail: "negative shape for \(name)") }
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
            try autoreleasepool {
                try file.seek(toOffset: UInt64(dataStart + entry.data_offsets[0]))
                let data = try readExactly(file, count: entry.data_offsets[1] - entry.data_offsets[0])
                let array = MLXArray(data, entry.shape, dtype: dtypes[entry.dtype]!)
                MLX.eval(array)
                arrays[name] = array
            }
        }
        return (arrays, header.metadata)
        #else
        throw InvalidFile(detail: "uncached owned reader requires Darwin")
        #endif
    }

    private static func readExactly(_ file: FileHandle, count: Int) throws -> Data {
        var result = Data()
        result.reserveCapacity(count)
        while result.count < count {
            guard let chunk = try file.read(upToCount: min(count - result.count, 4 * 1024 * 1024)),
                !chunk.isEmpty else { throw InvalidFile(detail: "unexpected end of file") }
            result.append(chunk)
        }
        return result
    }
}
