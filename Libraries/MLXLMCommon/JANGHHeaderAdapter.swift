import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Metadata-only adapter. Never reads tensor payloads or constructs MLX arrays.
/// A later mapper must reopen/revalidate these identities, or retain the same
/// descriptors through mapping; this snapshot does not eliminate filesystem races.
enum JANGHHeaderAdapter {
    struct FileIdentity: Equatable, Sendable {
        let device: UInt64
        let inode: UInt64
        let bytes: Int
        let modifiedSeconds: Int64
        let modifiedNanoseconds: Int64
        let changedSeconds: Int64
        let changedNanoseconds: Int64
    }

    struct Snapshot: Sendable {
        let indexName: String
        let weightMap: [String: String]
        let shards: [String: JANGHTensorIndexPlan.ShardHeader]
        let identities: [String: FileIdentity]
        let indexIdentity: FileIdentity
        /// Exact metadata bytes allow evidence hashing without reading weights.
        let indexData: Data
        let headerData: [String: Data]
    }

    private struct Index: Decodable { let weight_map: [String: String] }
    private struct Tensor: Decodable {
        let dtype: String
        let shape: [Int]
        let data_offsets: [Int]
    }

    private typealias Failure = JANGHFormatContract.ValidationError

    static func read(
        directory: URL, indexName: String, maximumMetadataBytes: Int = 16 * 1024 * 1024,
        maximumTotalMetadataBytes: Int = 64 * 1024 * 1024
    ) throws -> Snapshot {
        guard directory.isFileURL, maximumMetadataBytes >= 8, maximumTotalMetadataBytes >= maximumMetadataBytes else {
            throw Failure.invalid("invalid JANGH metadata read options")
        }
        // Resolve the bundle root once (user model directories may be symlinks).
        // Only direct, non-symlink regular files inside this root are admitted.
        let root = directory.resolvingSymlinksInPath().standardizedFileURL
        let rootDescriptor = open(root.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard rootDescriptor >= 0 else { throw Failure.invalid("cannot open JANGH bundle directory") }
        defer { close(rootDescriptor) }
        let indexFile = try openRegular(rootDescriptor: rootDescriptor, name: indexName)
        defer { try? indexFile.close() }
        let indexIdentity = try identity(indexFile)
        guard indexIdentity.bytes > 0, indexIdentity.bytes <= maximumMetadataBytes else {
            throw Failure.invalid("JANGH index exceeds metadata bound")
        }
        let indexData = try readExactly(indexFile, count: indexIdentity.bytes)
        try validateJSON(indexData)
        let map = try JSONDecoder().decode(Index.self, from: indexData).weight_map
        guard !map.isEmpty else { throw Failure.invalid("empty safetensors index") }
        guard try identity(indexFile) == indexIdentity else {
            throw Failure.invalid("JANGH index changed during read")
        }
        var remainingBytes = maximumTotalMetadataBytes - indexData.count
        var summaries: [String: JANGHTensorIndexPlan.ShardHeader] = [:]
        var identities: [String: FileIdentity] = [:]
        var headers: [String: Data] = [:]
        for name in Set(map.values).sorted() {
            guard name.hasSuffix(".safetensors") else {
                throw Failure.invalid("invalid safetensors shard name")
            }
            let file = try openRegular(rootDescriptor: rootDescriptor, name: name)
            defer { try? file.close() }
            let before = try identity(file)
            guard before.bytes >= 8 else { throw Failure.invalid("short safetensors file") }
            guard remainingBytes >= 8 else { throw Failure.invalid("aggregate metadata bound exceeded") }
            remainingBytes -= 8
            let prefix = try readExactly(file, count: 8)
            let length = prefix.enumerated().reduce(UInt64(0)) {
                $0 | UInt64($1.element) << (8 * $1.offset)
            }
            guard length > 0, length <= UInt64(maximumMetadataBytes),
                length <= UInt64(before.bytes - 8), length <= UInt64(remainingBytes)
            else { throw Failure.invalid("safetensors header length out of bounds") }
            remainingBytes -= Int(length)
            let data = try readExactly(file, count: Int(length))
            let raw = try validateJSON(data)
            guard let object = raw as? [String: Any] else {
                throw Failure.invalid("safetensors header must be an object")
            }
            var tensors: [String: JANGHTensorIndexPlan.TensorHeader] = [:]
            for (tensorName, rawTensor) in object where tensorName != "__metadata__" {
                guard rawTensor is [String: Any] else {
                    throw Failure.invalid("invalid safetensors tensor descriptor")
                }
                let tensor = try JSONDecoder().decode(
                    Tensor.self, from: JSONSerialization.data(withJSONObject: rawTensor))
                guard tensor.data_offsets.count == 2 else {
                    throw Failure.invalid("invalid safetensors offset count")
                }
                tensors[tensorName] = .init(
                    dtype: tensor.dtype, shape: tensor.shape,
                    start: tensor.data_offsets[0], end: tensor.data_offsets[1])
            }
            guard try identity(file) == before else {
                throw Failure.invalid("safetensors file changed during header read")
            }
            summaries[name] = .init(fileBytes: before.bytes, dataStart: 8 + Int(length), tensors: tensors)
            identities[name] = before
            headers[name] = data
        }
        return Snapshot(indexName: indexName, weightMap: map, shards: summaries, identities: identities,
                        indexIdentity: indexIdentity, indexData: indexData, headerData: headers)
    }

    static func openRegular(rootDescriptor: Int32, name: String) throws -> FileHandle {
        guard !name.isEmpty, name != ".", name != "..", !name.contains("/"),
            !name.contains("\\"), !name.contains("\0")
        else { throw Failure.invalid("unsafe JANGH metadata filename") }
        let descriptor = openat(rootDescriptor, name, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard descriptor >= 0 else {
            // Name the file and the reason: a symlinked shard (refused by design, the bundle's files are read and
            // identity-checked in place) used to surface only as the next factory's unrelated error.
            let code = errno
            let reason = String(cString: strerror(code))
            let hint =
                code == ELOOP
                ? " (a symbolic link: JANGH bundles read their regular files in place; link the bundle directory, "
                    + "not its files)" : ""
            throw Failure.invalid("cannot open JANGH metadata file '\(name)': \(reason)\(hint)")
        }
        let file = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        do {
            _ = try identity(file)
            return file
        } catch {
            try? file.close()
            throw error
        }
    }

    static func identity(_ file: FileHandle) throws -> FileIdentity {
        var value = stat()
        guard fstat(file.fileDescriptor, &value) == 0,
            (value.st_mode & S_IFMT) == S_IFREG, value.st_size >= 0,
            let bytes = Int(exactly: value.st_size)
        else { throw Failure.invalid("JANGH metadata requires a regular file") }
        #if canImport(Darwin)
        let modified = value.st_mtimespec
        let changed = value.st_ctimespec
        #else
        let modified = value.st_mtim
        let changed = value.st_ctim
        #endif
        return FileIdentity(device: UInt64(truncatingIfNeeded: value.st_dev), inode: UInt64(value.st_ino), bytes: bytes,
                            modifiedSeconds: Int64(modified.tv_sec), modifiedNanoseconds: Int64(modified.tv_nsec),
                            changedSeconds: Int64(changed.tv_sec), changedNanoseconds: Int64(changed.tv_nsec))
    }

    private static func readExactly(_ file: FileHandle, count: Int) throws -> Data {
        var result = Data()
        while result.count < count {
            guard let part = try file.read(upToCount: count - result.count), !part.isEmpty else {
                throw Failure.invalid("truncated JANGH metadata")
            }
            result.append(part)
        }
        return result
    }

    /// Foundation parses syntax and scalar types. This additional scan rejects
    /// duplicate keys (including escaped spellings), which dictionary decoding
    /// would silently collapse, and bounds nesting before walking descriptors.
    @discardableResult
    static func validateJSON(_ data: Data) throws -> Any {
        let result = try JSONSerialization.jsonObject(with: data)
        struct Frame { var keys: Set<String>?; var expectsKey: Bool }
        var stack: [Frame] = []
        let bytes = Array(data)
        var index = 0
        while index < bytes.count {
            switch bytes[index] {
            case 123, 91: // object / array
                guard stack.count < 128 else { throw Failure.invalid("metadata nesting exceeds limit") }
                stack.append(Frame(keys: bytes[index] == 123 ? [] : nil, expectsKey: bytes[index] == 123))
            case 125, 93:
                _ = stack.popLast()
            case 44:
                if !stack.isEmpty, stack[stack.count - 1].keys != nil { stack[stack.count - 1].expectsKey = true }
            case 58:
                if !stack.isEmpty { stack[stack.count - 1].expectsKey = false }
            case 34:
                let start = index
                index += 1
                while index < bytes.count {
                    if bytes[index] == 92 { index += 2; continue }
                    if bytes[index] == 34 { break }
                    index += 1
                }
                if !stack.isEmpty, stack[stack.count - 1].expectsKey {
                    let key = try JSONDecoder().decode(String.self, from: Data(bytes[start...index]))
                    guard stack[stack.count - 1].keys!.insert(key).inserted else {
                        throw Failure.invalid("duplicate JSON metadata key")
                    }
                }
            default: break
            }
            index += 1
        }
        return result
    }
}
