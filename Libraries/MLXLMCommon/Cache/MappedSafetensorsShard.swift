// Copyright © 2026 Jinho Jang. All rights reserved.
// Read-only safetensors mapping and exact-range reads for routed tensor loading.

import Foundation
#if canImport(Darwin)
    import Darwin
#elseif canImport(Glibc)
    import Glibc
#endif

public enum MappedSafetensorsShardError: Error, CustomStringConvertible {
    case openFailed(URL, errno: Int32)
    case statFailed(URL, errno: Int32)
    case mmapFailed(URL, errno: Int32)
    case noCacheFailed(URL, errno: Int32)
    case readFailed(URL, errno: Int32)
    case truncatedHeader(URL)
    case malformedHeaderJSON(URL, String)
    case headerSizeOutOfBounds(URL, declared: UInt64, fileSize: UInt64)
    case unknownTensor(name: String)

    public var description: String {
        switch self {
        case .openFailed(let url, let e): return "open(\(url.lastPathComponent)) failed errno=\(e)"
        case .statFailed(let url, let e): return "fstat(\(url.lastPathComponent)) failed errno=\(e)"
        case .mmapFailed(let url, let e): return "mmap(\(url.lastPathComponent)) failed errno=\(e)"
        case .noCacheFailed(let url, let e): return "fcntl(F_NOCACHE, \(url.lastPathComponent)) failed errno=\(e)"
        case .readFailed(let url, let e): return "pread(\(url.lastPathComponent)) failed errno=\(e)"
        case .truncatedHeader(let url): return "safetensors header truncated in \(url.lastPathComponent)"
        case .malformedHeaderJSON(let url, let m): return "safetensors header JSON malformed in \(url.lastPathComponent): \(m)"
        case .headerSizeOutOfBounds(let url, let h, let f): return "header size \(h) > file size \(f) in \(url.lastPathComponent)"
        case .unknownTensor(let n): return "tensor not found: \(n)"
        }
    }
}

public struct TensorDescriptor: Sendable {
    public let name: String
    public let dtype: String
    public let shape: [Int]
    public let dataOffset: UInt64        // absolute byte offset in shard file
    public let dataLength: UInt64        // tensor byte length
}

public final class MappedSafetensorsShard: @unchecked Sendable {

    public let url: URL
    public let fileSize: UInt64
    public let baseAddress: UnsafeRawPointer
    public let noCacheIOEnabled: Bool
    public private(set) var tensors: [String: TensorDescriptor] = [:]

    private let fd: Int32

    public init(path: URL, noCacheIO: Bool = false) throws {
        self.url = path

        // 1. open + fstat
        let fdLocal = open(path.path, O_RDONLY)
        guard fdLocal >= 0 else { throw MappedSafetensorsShardError.openFailed(path, errno: errno) }
        self.fd = fdLocal

        #if canImport(Darwin)
        if noCacheIO {
            guard fcntl(fdLocal, F_NOCACHE, 1) == 0 else {
                let savedErrno = errno
                close(fdLocal)
                throw MappedSafetensorsShardError.noCacheFailed(path, errno: savedErrno)
            }
        }
        self.noCacheIOEnabled = noCacheIO
        #else
        // F_NOCACHE is Darwin-only: on Linux the shard's reads go through the page cache.
        self.noCacheIOEnabled = false
        #endif

        var st = stat()
        guard fstat(fdLocal, &st) == 0 else {
            close(fdLocal)
            throw MappedSafetensorsShardError.statFailed(path, errno: errno)
        }
        self.fileSize = UInt64(st.st_size)

        // 2. mmap whole file PROT_READ
        guard let raw = mmap(nil, Int(self.fileSize), PROT_READ, MAP_SHARED, fdLocal, 0),
              raw != UnsafeMutableRawPointer(bitPattern: -1) else {
            close(fdLocal)
            throw MappedSafetensorsShardError.mmapFailed(path, errno: errno)
        }
        self.baseAddress = UnsafeRawPointer(raw)

        // 3. parse safetensors header (first 8 bytes = header size)
        guard fileSize >= 8 else {
            munmap(raw, Int(fileSize))
            close(fdLocal)
            throw MappedSafetensorsShardError.truncatedHeader(path)
        }
        let headerSize = baseAddress.load(as: UInt64.self).littleEndian
        guard 8 + headerSize <= fileSize else {
            munmap(raw, Int(fileSize))
            close(fdLocal)
            throw MappedSafetensorsShardError.headerSizeOutOfBounds(path, declared: headerSize, fileSize: fileSize)
        }

        let headerJSONData = Data(bytes: baseAddress.advanced(by: 8), count: Int(headerSize))
        let dataAreaStart: UInt64 = 8 + headerSize
        let parsed: [String: Any]
        do {
            guard let obj = try JSONSerialization.jsonObject(with: headerJSONData)
                as? [String: Any] else {
                throw MappedSafetensorsShardError.malformedHeaderJSON(path, "not a JSON object")
            }
            parsed = obj
        } catch {
            munmap(raw, Int(fileSize))
            close(fdLocal)
            throw MappedSafetensorsShardError.malformedHeaderJSON(path, "\(error)")
        }

        // 4. extract tensor descriptors. The `__metadata__` key (if
        //    present) is non-tensor metadata.
        var byName: [String: TensorDescriptor] = [:]
        byName.reserveCapacity(parsed.count)
        for (name, value) in parsed {
            if name == "__metadata__" { continue }
            guard
                let entry = value as? [String: Any],
                let dtype = entry["dtype"] as? String,
                let shape = Self.parseShape(entry["shape"]),
                let offsets = entry["data_offsets"] as? [Any],
                offsets.count == 2,
                let startNum = (offsets[0] as? NSNumber)?.uint64Value,
                let endNum = (offsets[1] as? NSNumber)?.uint64Value
            else {
                continue
            }
            byName[name] = TensorDescriptor(
                name: name,
                dtype: dtype,
                shape: shape,
                dataOffset: dataAreaStart + startNum,
                dataLength: endNum - startNum
            )
        }
        self.tensors = byName
    }

    private static func parseShape(_ raw: Any?) -> [Int]? {
        if let shape = raw as? [Int] {
            return shape
        }
        guard let values = raw as? [Any] else {
            return nil
        }
        var shape: [Int] = []
        shape.reserveCapacity(values.count)
        for value in values {
            if let intValue = value as? Int {
                guard intValue >= 0 else { return nil }
                shape.append(intValue)
            } else if let number = value as? NSNumber {
                let intValue = number.intValue
                guard intValue >= 0 else { return nil }
                shape.append(intValue)
            } else {
                return nil
            }
        }
        return shape
    }

    deinit {
        munmap(UnsafeMutableRawPointer(mutating: baseAddress), Int(fileSize))
        close(fd)
    }

    // MARK: - Header inspection

    /// Returns just the tensor names in the shard's safetensors header,
    /// without mmap'ing the data area. Returns nil if header parse fails
    /// (caller should fall back to full open via init).
    public static func sniffTensorNames(at path: URL) -> [String]? {
        let fd = open(path.path, O_RDONLY)
        guard fd >= 0 else { return nil }
        defer { close(fd) }

        // Read first 8 bytes — header size.
        var sizeBytes = [UInt8](repeating: 0, count: 8)
        let n = sizeBytes.withUnsafeMutableBufferPointer { buf -> Int in
            read(fd, buf.baseAddress, 8)
        }
        guard n == 8 else { return nil }
        let headerSize: UInt64 = sizeBytes.withUnsafeBytes { ptr in
            ptr.load(as: UInt64.self).littleEndian
        }
        // Cap header size at 32 MB — anything larger is suspicious and
        // not worth pre-reading.
        guard headerSize > 0 && headerSize < 32 * 1024 * 1024 else { return nil }

        var headerBytes = [UInt8](repeating: 0, count: Int(headerSize))
        let h = headerBytes.withUnsafeMutableBufferPointer { buf -> Int in
            read(fd, buf.baseAddress, Int(headerSize))
        }
        guard h == Int(headerSize) else { return nil }

        let headerData = Data(headerBytes)
        guard let obj = try? JSONSerialization.jsonObject(with: headerData)
                as? [String: Any] else { return nil }
        // Filter out __metadata__ + any non-tensor keys.
        return obj.keys.filter { $0 != "__metadata__" }
    }

    // MARK: - Tensor lookup

    /// Returns the absolute byte range in the shard file for the given
    /// tensor name. nil if the tensor isn't in this shard.
    public func byteRange(for tensorName: String) -> Range<UInt64>? {
        guard let d = tensors[tensorName] else { return nil }
        return d.dataOffset..<(d.dataOffset + d.dataLength)
    }

    public func descriptor(for tensorName: String) -> TensorDescriptor? {
        tensors[tensorName]
    }

    public func bytes(in range: Range<UInt64>) -> UnsafeRawBufferPointer {
        let start = baseAddress.advanced(by: Int(range.lowerBound))
        let count = Int(range.upperBound - range.lowerBound)
        return UnsafeRawBufferPointer(start: start, count: count)
    }

    /// Read an exact file range through `pread`. When the shard was opened
    /// with `noCacheIO: true`, Darwin services these reads with `F_NOCACHE`,
    /// so one-shot row gathers do not populate the unified file cache with the
    /// full backing table. The mmap remains only for the safetensors header and
    /// descriptor index; payload callers can stay off the mapped data pages.
    public func readBytes(range: Range<UInt64>) throws -> Data {
        guard range.lowerBound <= range.upperBound,
              range.upperBound <= fileSize,
              range.count <= UInt64(Int.max)
        else {
            throw MappedSafetensorsShardError.readFailed(url, errno: EINVAL)
        }
        let byteCount = Int(range.count)
        var data = Data(count: byteCount)
        var completed = 0
        while completed < byteCount {
            let readNow = data.withUnsafeMutableBytes { raw -> Int in
                guard let base = raw.baseAddress else { return 0 }
                return pread(
                    fd,
                    base.advanced(by: completed),
                    byteCount - completed,
                    off_t(range.lowerBound) + off_t(completed))
            }
            guard readNow > 0 else {
                throw MappedSafetensorsShardError.readFailed(
                    url, errno: readNow == 0 ? EIO : errno)
            }
            completed += readNow
        }
        return data
    }

}
