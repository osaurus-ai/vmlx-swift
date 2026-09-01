// Copyright © 2026 Jinho Jang. All rights reserved.

import Foundation

#if canImport(Darwin)
    import Darwin
#elseif canImport(Glibc)
    import Glibc
#endif

/// Repairs dtype-misaligned safetensors containers before MLX mmaps them.
///
/// The repair changes only container order and offsets. Tensor names, shapes,
/// dtypes, metadata, and payload bytes are preserved. Every replacement is
/// written beside its source, flushed, byte-verified, and atomically renamed.
/// Any failure is advisory: the original shard remains available to MLX's
/// aligned-copy fallback.
enum SafetensorsStorageHealer {
    struct Configuration {
        var environment: [String: String] = ProcessInfo.processInfo.environment
        var availableBytes: ((URL) throws -> UInt64)?
        var failAfterCopiedBytes: UInt64?
        var logger: (String) -> Void = { message in
            fputs("[SafetensorsStorageHealer] \(message)\n", stderr)
        }
    }

    struct Result: Equatable {
        var scannedShards = 0
        var healedShards = 0
        var skippedShards = 0
        var fallbackShards = 0
    }

    private struct Tensor {
        var name: String
        var descriptor: [String: Any]
        var dtype: String
        var alignment: UInt64
        var sourceOffset: UInt64
        var byteLength: UInt64
        var outputOffset: UInt64 = 0
    }

    private struct Header {
        var dataBase: UInt64
        var metadata: Any?
        var tensors: [Tensor]
    }

    private static let processLock = NSLock()
    private static let bufferBytes = 4 * 1024 * 1024
    private static let freeSpaceReserve = UInt64(512 * 1024 * 1024)
    private static let maximumHeaderBytes = UInt64(100_000_000)

    static func healBundleIfEligible(at originalURL: URL) {
        _ = healBundle(at: originalURL, configuration: Configuration())
    }

    @discardableResult
    static func healBundle(
        at originalURL: URL,
        configuration: Configuration
    ) -> Result {
        let env = configuration.environment
        let rawEnabled =
            env["MLXPRESS_HEAL_SAFETENSORS"]
            ?? env["JANGPRESS_HEAL_SAFETENSORS"]
        if isDisabledFlag(rawEnabled) {
            configuration.logger("event=disabled path=\(quoted(originalURL.path))")
            return Result()
        }

        let directory = originalURL.standardizedFileURL
        guard !isHashAddressedStorage(directory) else {
            configuration.logger(
                "event=skip reason=hash_addressed_storage path=\(quoted(directory.path))")
            return Result()
        }
        guard FileManager.default.isWritableFile(atPath: directory.path) else {
            configuration.logger(
                "event=skip reason=directory_not_writable path=\(quoted(directory.path))")
            return Result()
        }

        return processLock.withLock {
            healLocked(at: directory, configuration: configuration)
        }
    }

    private static func healLocked(
        at directory: URL,
        configuration: Configuration
    ) -> Result {
        var result = Result()
        let directoryFD = systemOpen(directory.path, O_RDONLY, 0)
        guard directoryFD >= 0 else {
            configuration.logger(
                "event=fallback reason=directory_open_failed path=\(quoted(directory.path)) error=\(errno)"
            )
            result.fallbackShards = 1
            return result
        }
        defer { _ = systemClose(directoryFD) }

        let entries: [URL]
        do {
            // RESOLVED: `contentsOfDirectory(at:)` throws on a URL naming a symlink to a
            // directory, so a bundle reached through one would enumerate to nothing and take the
            // fallback below — the healer silently declining to do its job on exactly the bundles
            // an operator has deduplicated by symlinking between namespaces.
            entries = try FileManager.default.contentsOfDirectory(
                at: directory.resolvingSymlinksInPath(),
                includingPropertiesForKeys: [.isSymbolicLinkKey, .fileSizeKey],
                options: [.skipsHiddenFiles]
            )
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        } catch {
            configuration.logger(
                "event=fallback reason=enumeration_failed path=\(quoted(directory.path)) detail=\(quoted(String(describing: error)))"
            )
            result.fallbackShards = 1
            return result
        }

        for shard in entries {
            result.scannedShards += 1
            do {
                let values = try shard.resourceValues(forKeys: [.isSymbolicLinkKey])
                guard values.isSymbolicLink != true,
                    !isHashAddressedStorage(shard)
                else {
                    result.skippedShards += 1
                    configuration.logger(
                        "event=skip reason=linked_or_hash_addressed_shard shard=\(quoted(shard.path))"
                    )
                    continue
                }

                let header = try readHeader(shard)
                guard
                    header.tensors.contains(where: {
                        $0.sourceOffset % $0.alignment != 0
                    })
                else {
                    continue
                }

                let attributes = try FileManager.default.attributesOfItem(atPath: shard.path)
                let sourceSize = (attributes[.size] as? NSNumber)?.uint64Value ?? 0
                let available =
                    try configuration.availableBytes?(directory)
                    ?? availableBytes(at: directory)
                let required = sourceSize.addingReportingOverflow(freeSpaceReserve)
                guard !required.overflow, available >= required.partialValue else {
                    result.fallbackShards += 1
                    configuration.logger(
                        String(
                            format:
                                "event=fallback reason=insufficient_disk shard=%@ required_bytes=%llu available_bytes=%llu",
                            quoted(shard.path), required.partialValue, available))
                    continue
                }

                try rewriteAndInstall(
                    shard: shard,
                    sourceHeader: header,
                    sourceAttributes: attributes,
                    directoryFD: directoryFD,
                    failAfterCopiedBytes: configuration.failAfterCopiedBytes)
                result.healedShards += 1
                configuration.logger(
                    "event=healed shard=\(quoted(shard.path)) tensors=\(header.tensors.count) bytes=\(sourceSize)"
                )
            } catch {
                result.fallbackShards += 1
                configuration.logger(
                    "event=fallback reason=heal_failed shard=\(quoted(shard.path)) detail=\(quoted(String(describing: error)))"
                )
            }
        }

        return result
    }

    private static func rewriteAndInstall(
        shard: URL,
        sourceHeader: Header,
        sourceAttributes: [FileAttributeKey: Any],
        directoryFD: Int32,
        failAfterCopiedBytes: UInt64?
    ) throws {
        let temporary = shard.deletingLastPathComponent().appendingPathComponent(
            ".\(shard.lastPathComponent).vmlx-heal-\(UUID().uuidString).tmp")
        var installed = false
        defer {
            if !installed {
                try? FileManager.default.removeItem(at: temporary)
            }
        }

        let arranged = sourceHeader.tensors.sorted { lhs, rhs in
            if lhs.alignment != rhs.alignment { return lhs.alignment > rhs.alignment }
            return lhs.name < rhs.name
        }
        var rewritten = arranged
        var cursor: UInt64 = 0
        for index in rewritten.indices {
            rewritten[index].outputOffset = cursor
            let addition = cursor.addingReportingOverflow(rewritten[index].byteLength)
            guard !addition.overflow else { throw healerError("payload size overflow") }
            cursor = addition.partialValue
        }
        let headerData = try makeHeader(metadata: sourceHeader.metadata, tensors: rewritten)

        let outputFD = systemOpen(
            temporary.path,
            O_WRONLY | O_CREAT | O_EXCL | O_TRUNC,
            S_IRUSR | S_IWUSR)
        guard outputFD >= 0 else { throw posixError("open", temporary.path) }
        var outputOpen = true
        defer {
            if outputOpen { _ = systemClose(outputFD) }
        }
        _ = systemNoCache(outputFD)

        var encodedLength = UInt64(headerData.count).littleEndian
        try withUnsafeBytes(of: &encodedLength) {
            try writeAll(fd: outputFD, bytes: $0)
        }
        try headerData.withUnsafeBytes {
            try writeAll(fd: outputFD, bytes: $0)
        }

        let sourceFD = systemOpen(shard.path, O_RDONLY, 0)
        guard sourceFD >= 0 else { throw posixError("open", shard.path) }
        defer { _ = systemClose(sourceFD) }
        _ = systemNoCache(sourceFD)

        var copied: UInt64 = 0
        for tensor in rewritten {
            try copyRange(
                sourceFD: sourceFD,
                sourcePath: shard.path,
                sourceOffset: tensor.sourceOffset,
                byteLength: tensor.byteLength,
                outputFD: outputFD,
                copied: &copied,
                failAfterCopiedBytes: failAfterCopiedBytes)
        }
        guard systemFsync(outputFD) == 0 else {
            throw posixError("fsync", temporary.path)
        }
        guard systemClose(outputFD) == 0 else {
            outputOpen = false
            throw posixError("close", temporary.path)
        }
        outputOpen = false

        if let permissions = (sourceAttributes[.posixPermissions] as? NSNumber)?.uint16Value {
            try FileManager.default.setAttributes(
                [.posixPermissions: NSNumber(value: permissions)],
                ofItemAtPath: temporary.path)
        }

        let outputHeader = try readHeader(temporary)
        try verifyDescriptorsAndPayloads(
            sourceURL: shard,
            sourceHeader: sourceHeader,
            outputURL: temporary,
            outputHeader: outputHeader)

        let currentAttributes = try FileManager.default.attributesOfItem(atPath: shard.path)
        guard sameSourceIdentity(sourceAttributes, currentAttributes) else {
            throw healerError("source changed during heal")
        }

        guard systemRename(temporary.path, shard.path) == 0 else {
            throw posixError("rename", shard.path)
        }
        installed = true
        // The shard itself is already durable. Directory fsync is best effort
        // because some filesystems reject fsync on directory descriptors.
        _ = systemFsync(directoryFD)
    }

    private static func readHeader(_ url: URL) throws -> Header {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let prefix = try handle.read(upToCount: 8) ?? Data()
        guard prefix.count == 8 else { throw healerError("truncated length") }
        let headerLength = prefix.withUnsafeBytes {
            UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
        }
        guard headerLength > 0, headerLength <= maximumHeaderBytes else {
            throw healerError("invalid header length \(headerLength)")
        }
        let rawHeader = try handle.read(upToCount: Int(headerLength)) ?? Data()
        guard rawHeader.count == Int(headerLength) else {
            throw healerError("truncated header")
        }
        let object = try JSONSerialization.jsonObject(with: rawHeader)
        guard let dictionary = object as? [String: Any] else {
            throw healerError("header is not an object")
        }

        let fileSize = try FileManager.default.attributesOfItem(atPath: url.path)[.size]
        let size = (fileSize as? NSNumber)?.uint64Value ?? 0
        let dataBase = 8 + headerLength
        var tensors: [Tensor] = []
        for (name, rawDescriptor) in dictionary where name != "__metadata__" {
            guard let descriptor = rawDescriptor as? [String: Any],
                let dtype = descriptor["dtype"] as? String,
                let alignment = dtypeAlignment(dtype),
                descriptor["shape"] is [NSNumber],
                let offsets = descriptor["data_offsets"] as? [NSNumber],
                offsets.count == 2
            else {
                throw healerError("invalid descriptor for \(name)")
            }
            let start = offsets[0].uint64Value
            let end = offsets[1].uint64Value
            guard end >= start,
                (end - start) % alignment == 0,
                dataBase <= size,
                end <= size - dataBase
            else {
                throw healerError("invalid payload range for \(name)")
            }
            tensors.append(
                Tensor(
                    name: name,
                    descriptor: descriptor,
                    dtype: dtype,
                    alignment: alignment,
                    sourceOffset: dataBase + start,
                    byteLength: end - start))
        }

        let bySource = tensors.sorted { $0.sourceOffset < $1.sourceOffset }
        var expectedOffset = dataBase
        for tensor in bySource {
            guard tensor.sourceOffset == expectedOffset else {
                throw healerError("non-contiguous payload at \(tensor.name)")
            }
            expectedOffset += tensor.byteLength
        }
        guard expectedOffset == size else {
            throw healerError("payload does not cover file")
        }
        return Header(
            dataBase: dataBase,
            metadata: dictionary["__metadata__"],
            tensors: tensors)
    }

    private static func makeHeader(metadata: Any?, tensors: [Tensor]) throws -> Data {
        var object: [String: Any] = [:]
        for tensor in tensors {
            var descriptor = tensor.descriptor
            descriptor["data_offsets"] = [
                tensor.outputOffset,
                tensor.outputOffset + tensor.byteLength,
            ]
            object[tensor.name] = descriptor
        }
        if let metadata { object["__metadata__"] = metadata }
        var encoded = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        let maximumAlignment = max(UInt64(8), tensors.map(\.alignment).max() ?? 8)
        let remainder = UInt64(8 + encoded.count) % maximumAlignment
        if remainder != 0 {
            encoded.append(Data(repeating: 0x20, count: Int(maximumAlignment - remainder)))
        }
        return encoded
    }

    private static func verifyDescriptorsAndPayloads(
        sourceURL: URL,
        sourceHeader: Header,
        outputURL: URL,
        outputHeader: Header
    ) throws {
        guard sourceHeader.tensors.count == outputHeader.tensors.count,
            jsonEqual(sourceHeader.metadata, outputHeader.metadata)
        else { throw healerError("header verification failed") }

        let outputByName = Dictionary(
            uniqueKeysWithValues:
                outputHeader.tensors.map { ($0.name, $0) })
        let sourceFD = systemOpen(sourceURL.path, O_RDONLY, 0)
        guard sourceFD >= 0 else { throw posixError("open", sourceURL.path) }
        defer { _ = systemClose(sourceFD) }
        let outputFD = systemOpen(outputURL.path, O_RDONLY, 0)
        guard outputFD >= 0 else { throw posixError("open", outputURL.path) }
        defer { _ = systemClose(outputFD) }
        _ = systemNoCache(sourceFD)
        _ = systemNoCache(outputFD)

        for source in sourceHeader.tensors {
            guard let output = outputByName[source.name],
                output.dtype == source.dtype,
                output.byteLength == source.byteLength,
                output.sourceOffset % output.alignment == 0,
                descriptorsEqual(source.descriptor, output.descriptor)
            else { throw healerError("descriptor verification failed for \(source.name)") }
            try compareRanges(
                lhsFD: sourceFD,
                lhsOffset: source.sourceOffset,
                rhsFD: outputFD,
                rhsOffset: output.sourceOffset,
                byteLength: source.byteLength)
        }
    }

    private static func copyRange(
        sourceFD: Int32,
        sourcePath: String,
        sourceOffset: UInt64,
        byteLength: UInt64,
        outputFD: Int32,
        copied: inout UInt64,
        failAfterCopiedBytes: UInt64?
    ) throws {
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: bufferBytes, alignment: 4096)
        defer { buffer.deallocate() }
        var remaining = byteLength
        var offset = sourceOffset
        while remaining > 0 {
            if let limit = failAfterCopiedBytes, copied >= limit {
                throw healerError("injected interrupted write")
            }
            var count = min(UInt64(bufferBytes), remaining)
            if let limit = failAfterCopiedBytes {
                count = min(count, limit - copied)
                if count == 0 { throw healerError("injected interrupted write") }
            }
            let read = systemPread(sourceFD, buffer, Int(count), Int64(offset))
            guard read > 0 else { throw posixError("pread", sourcePath) }
            try writeAll(fd: outputFD, pointer: buffer, count: read)
            remaining -= UInt64(read)
            offset += UInt64(read)
            copied += UInt64(read)
        }
    }

    private static func compareRanges(
        lhsFD: Int32,
        lhsOffset: UInt64,
        rhsFD: Int32,
        rhsOffset: UInt64,
        byteLength: UInt64
    ) throws {
        let lhs = UnsafeMutableRawPointer.allocate(byteCount: bufferBytes, alignment: 4096)
        let rhs = UnsafeMutableRawPointer.allocate(byteCount: bufferBytes, alignment: 4096)
        defer {
            lhs.deallocate()
            rhs.deallocate()
        }
        var remaining = byteLength
        var lhsCursor = lhsOffset
        var rhsCursor = rhsOffset
        while remaining > 0 {
            let count = Int(min(UInt64(bufferBytes), remaining))
            let lhsRead = systemPread(lhsFD, lhs, count, Int64(lhsCursor))
            let rhsRead = systemPread(rhsFD, rhs, count, Int64(rhsCursor))
            guard lhsRead == count, rhsRead == count,
                memcmp(lhs, rhs, count) == 0
            else { throw healerError("payload byte verification failed") }
            remaining -= UInt64(count)
            lhsCursor += UInt64(count)
            rhsCursor += UInt64(count)
        }
    }

    private static func writeAll(fd: Int32, bytes: UnsafeRawBufferPointer) throws {
        guard let base = bytes.baseAddress else { return }
        try writeAll(fd: fd, pointer: base, count: bytes.count)
    }

    private static func writeAll(fd: Int32, pointer: UnsafeRawPointer, count: Int) throws {
        var written = 0
        while written < count {
            let result = systemWrite(fd, pointer.advanced(by: written), count - written)
            guard result > 0 else { throw posixError("write", "fd=\(fd)") }
            written += result
        }
    }

    private static func availableBytes(at directory: URL) throws -> UInt64 {
        let attributes = try FileManager.default.attributesOfFileSystem(forPath: directory.path)
        guard let number = attributes[.systemFreeSize] as? NSNumber else {
            throw healerError("free disk capacity unavailable")
        }
        return number.uint64Value
    }

    private static func sameSourceIdentity(
        _ lhs: [FileAttributeKey: Any],
        _ rhs: [FileAttributeKey: Any]
    ) -> Bool {
        let lhsSize = (lhs[.size] as? NSNumber)?.uint64Value
        let rhsSize = (rhs[.size] as? NSNumber)?.uint64Value
        let lhsDate = lhs[.modificationDate] as? Date
        let rhsDate = rhs[.modificationDate] as? Date
        return lhsSize == rhsSize && lhsDate == rhsDate
    }

    private static func descriptorsEqual(_ lhs: [String: Any], _ rhs: [String: Any]) -> Bool {
        var lhs = lhs
        var rhs = rhs
        lhs.removeValue(forKey: "data_offsets")
        rhs.removeValue(forKey: "data_offsets")
        return jsonEqual(lhs, rhs)
    }

    private static func jsonEqual(_ lhs: Any?, _ rhs: Any?) -> Bool {
        switch (lhs, rhs) {
        case (nil, nil): return true
        case (let lhs?, let rhs?):
            guard JSONSerialization.isValidJSONObject(["value": lhs]),
                JSONSerialization.isValidJSONObject(["value": rhs]),
                let lhsData = try? JSONSerialization.data(
                    withJSONObject: ["value": lhs], options: [.sortedKeys]),
                let rhsData = try? JSONSerialization.data(
                    withJSONObject: ["value": rhs], options: [.sortedKeys])
            else { return false }
            return lhsData == rhsData
        default: return false
        }
    }

    private static func dtypeAlignment(_ dtype: String) -> UInt64? {
        switch dtype {
        case "C128": return 16
        case "F64", "I64", "U64", "C64": return 8
        case "F32", "I32", "U32": return 4
        case "F16", "BF16", "I16", "U16": return 2
        case "BOOL", "U8", "I8", "F8_E5M2", "F8_E4M3", "F8_E4M3FN",
            "F8_E4M3FNUZ", "F8_E5M2FNUZ", "F8_E8M0":
            return 1
        default: return nil
        }
    }

    private static func isHashAddressedStorage(_ url: URL) -> Bool {
        let path = url.standardizedFileURL.path.lowercased()
        if path.contains("/.cache/huggingface/hub/")
            || path.contains("/huggingface/hub/models--")
        {
            return true
        }
        let components = url.standardizedFileURL.pathComponents.map { $0.lowercased() }
        guard components.contains(where: { $0.hasPrefix("models--") }) else { return false }
        return components.contains("blobs") || components.contains("snapshots")
    }

    private static func isDisabledFlag(_ raw: String?) -> Bool {
        guard let raw = raw?.lowercased() else { return false }
        return raw == "0" || raw == "false" || raw == "no" || raw == "off"
    }

    private static func quoted(_ value: String) -> String {
        let escaped =
            value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        return "\"" + escaped + "\""
    }

    private static func healerError(_ message: String) -> Error {
        NSError(
            domain: "SafetensorsStorageHealer",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: message])
    }

    private static func posixError(_ operation: String, _ path: String) -> Error {
        NSError(
            domain: NSPOSIXErrorDomain,
            code: Int(errno),
            userInfo: [
                NSLocalizedDescriptionKey:
                    "\(operation) failed for \(path): \(String(cString: strerror(errno)))"
            ])
    }

    #if canImport(Darwin)
        private static func systemOpen(_ path: String, _ flags: Int32, _ mode: mode_t) -> Int32 {
            Darwin.open(path, flags, mode)
        }
        private static func systemClose(_ fd: Int32) -> Int32 { Darwin.close(fd) }
        private static func systemWrite(_ fd: Int32, _ pointer: UnsafeRawPointer, _ count: Int)
            -> Int
        {
            Darwin.write(fd, pointer, count)
        }
        private static func systemPread(
            _ fd: Int32, _ pointer: UnsafeMutableRawPointer, _ count: Int, _ offset: Int64
        ) -> Int { Darwin.pread(fd, pointer, count, offset) }
        private static func systemFsync(_ fd: Int32) -> Int32 { Darwin.fsync(fd) }
        private static func systemRename(_ source: String, _ destination: String) -> Int32 {
            Darwin.rename(source, destination)
        }
        private static func systemNoCache(_ fd: Int32) -> Int32 { Darwin.fcntl(fd, F_NOCACHE, 1) }
    #elseif canImport(Glibc)
        private static func systemOpen(_ path: String, _ flags: Int32, _ mode: mode_t) -> Int32 {
            Glibc.open(path, flags, mode)
        }
        private static func systemClose(_ fd: Int32) -> Int32 { Glibc.close(fd) }
        private static func systemWrite(_ fd: Int32, _ pointer: UnsafeRawPointer, _ count: Int)
            -> Int
        {
            Glibc.write(fd, pointer, count)
        }
        private static func systemPread(
            _ fd: Int32, _ pointer: UnsafeMutableRawPointer, _ count: Int, _ offset: Int64
        ) -> Int { Glibc.pread(fd, pointer, count, offset) }
        private static func systemFsync(_ fd: Int32) -> Int32 { Glibc.fsync(fd) }
        private static func systemRename(_ source: String, _ destination: String) -> Int32 {
            Glibc.rename(source, destination)
        }
        private static func systemNoCache(_ fd: Int32) -> Int32 { 0 }
    #endif
}
