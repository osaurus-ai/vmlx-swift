// Copyright © 2026 Jinho Jang. All rights reserved.
//
// JangPressPrestacker
//
// Some JANGTQ MoE bundles store routed experts as thousands of separate
// `experts.N.w{1,2,3}.tq_*` tensors. The Swift model sanitizers used to
// combine those tensors with MLX.stacked(...), which materializes the whole
// routed expert bank into resident Metal buffers and defeats mmap-backed
// JangPress. This helper builds a low-RAM safetensors overlay where those
// per-expert tensors are stream-copied into the pre-stacked
// `switch_mlp.*.tq_*` layout that TurboQuantSwitchGLU already consumes.

import Foundation

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

private func jangPressPrestackerCleanupAtExit() {
    JangPressPrestacker.cleanupEphemeralPrestackDirectories()
}

public enum JangPressPrestacker {
    private static let version = "v2"
    private static let alignmentVersion = "align-v1"
    private static let outputName = "jangpress-prestacked.safetensors"
    private static let alignmentManifestName = "jangpress-align-manifest.json"
    private static let cleanupLock = NSLock()
    private nonisolated(unsafe) static var cleanupURLs = Set<String>()
    private nonisolated(unsafe) static var cleanupInstalled = false

    static func prestackedRoutedReplacementKeys(in directory: URL) throws -> Set<String> {
        let outputURL = directory.resolvingSymlinksInPath().appendingPathComponent(outputName)
        guard FileManager.default.fileExists(atPath: outputURL.path) else {
            return []
        }
        return Set(try readSafetensorsHeader(outputURL).tensors.keys)
    }

    static func prestackedReplacementKey(forPerExpertKey key: String) -> String? {
        matchPerExpertKey(key)?.outputKey
    }

    public static func prepareBundleIfNeeded(
        originalURL: URL,
        enabled: Bool
    ) throws -> URL {
        guard enabled else { return originalURL }

        let env = ProcessInfo.processInfo.environment
        let prestackRaw = env["MLXPRESS_PRESTACK"] ?? env["JANGPRESS_PRESTACK"]
        let prestackExplicitlyEnabled = isEnabledFlag(prestackRaw)
        let prestackDisabled = !prestackExplicitlyEnabled
        let alignSafetensors = env["MLXPRESS_ALIGN_SAFETENSORS"]
            ?? env["JANGPRESS_ALIGN_SAFETENSORS"]
        let alignExplicitlyEnabled = isEnabledFlag(alignSafetensors)
        let prestackStrict = env["MLXPRESS_PRESTACK_STRICT"]
            ?? env["JANGPRESS_PRESTACK_STRICT"]
        if prestackDisabled && !alignExplicitlyEnabled {
            return originalURL
        }

        let originalURL = originalURL.resolvingSymlinksInPath()
        let sidecar = originalURL.appendingPathComponent("jangtq_runtime.safetensors")
        let hasJANGTQSidecar = FileManager.default.fileExists(atPath: sidecar.path)
        var preparedURL = originalURL

        if hasJANGTQSidecar && prestackExplicitlyEnabled {
            do {
                let scan = try scanBundle(originalURL)
                guard !scan.groups.isEmpty else {
                    return try prepareAlignedBundleIfNeeded(
                        originalURL,
                        originalIsJANGTQ: true,
                        env: env)
                }

                let cacheURL = try cacheDirectory(for: originalURL, files: scan.files)
                try ensureOverlayLinks(from: originalURL, to: cacheURL)
                let outputURL = cacheURL.appendingPathComponent(outputName)
                let manifestURL = cacheURL.appendingPathComponent("jangpress-prestack-manifest.json")

                if FileManager.default.fileExists(atPath: outputURL.path),
                   FileManager.default.fileExists(atPath: manifestURL.path)
                {
                    log("using existing prestacked overlay \(cacheURL.path)")
                    preparedURL = cacheURL
                } else {
                    try FileManager.default.createDirectory(
                        at: cacheURL, withIntermediateDirectories: true)
                    let plan = try buildWritePlan(from: scan.groups)
                    guard !plan.isEmpty else { return originalURL }
                    try writeSafetensors(plan: plan, to: outputURL)
                    try writeManifest(plan: plan, source: originalURL, to: manifestURL)
                    let totalBytes = plan.reduce(UInt64(0)) { $0 + $1.totalBytes }
                    log(String(format:
                        "wrote %d prestacked routed tensors (%.1f GB) into %@",
                        plan.count, Double(totalBytes) / 1_073_741_824.0,
                        outputURL.path))
                    preparedURL = cacheURL
                }
            } catch {
                if prestackStrict == "1" {
                    throw error
                }
                log("prestack failed, falling back to original bundle: \(error)")
                preparedURL = originalURL
            }
        }

        do {
            return try prepareAlignedBundleIfNeeded(
                preparedURL,
                originalIsJANGTQ: hasJANGTQSidecar,
                env: env)
        } catch {
            if prestackStrict == "1" {
                throw error
            }
            log("safetensors alignment failed, falling back to \(preparedURL.path): \(error)")
            return preparedURL
        }
    }

    private struct SourceFile {
        var url: URL
        var size: UInt64
        var modified: TimeInterval
    }

    private static func isEnabledFlag(_ raw: String?) -> Bool {
        guard let raw = raw?.lowercased() else { return false }
        return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
    }

    private static func ephemeralPrestackEnabled(_ env: [String: String]) -> Bool {
        isEnabledFlag(env["MLXPRESS_PRESTACK_EPHEMERAL"] ?? env["JANGPRESS_PRESTACK_EPHEMERAL"])
    }

    private struct TensorSource {
        var key: String
        var expert: Int
        var fileURL: URL
        var byteOffset: UInt64
        var byteLength: UInt64
        var dtype: String
        var shape: [Int]
    }

    private struct TensorGroup {
        var outputKey: String
        var byExpert: [Int: TensorSource] = [:]
    }

    private struct BundleScan {
        var files: [SourceFile]
        var groups: [String: TensorGroup]
    }

    private struct WriteTensor {
        var key: String
        var dtype: String
        var shape: [Int]
        var sources: [TensorSource]
        var dataOffset: UInt64 = 0

        var totalBytes: UInt64 {
            sources.reduce(UInt64(0)) { $0 + $1.byteLength }
        }
    }

    private static func scanBundle(_ directory: URL) throws -> BundleScan {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: directory, includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey])
        else {
            return BundleScan(files: [], groups: [:])
        }

        var files: [SourceFile] = []
        var groups: [String: TensorGroup] = [:]
        var existingStackedKeys = Set<String>()
        var pending: [(String, TensorSource)] = []

        for case let url as URL in enumerator {
            guard url.pathExtension == "safetensors",
                  url.lastPathComponent != "jangtq_runtime.safetensors",
                  url.lastPathComponent != outputName
            else { continue }

            let values = try url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
            files.append(SourceFile(
                url: url,
                size: UInt64(values.fileSize ?? 0),
                modified: values.contentModificationDate?.timeIntervalSince1970 ?? 0))

            let header = try readSafetensorsHeader(url)
            for (key, value) in header.tensors {
                if isStackedRoutedKey(key) {
                    existingStackedKeys.insert(key)
                }
                guard let match = matchPerExpertKey(key),
                      let dtype = value["dtype"] as? String,
                      let shape = value["shape"] as? [Int],
                      let offsets = value["data_offsets"] as? [UInt64],
                      offsets.count == 2,
                      offsets[1] >= offsets[0]
                else { continue }

                let source = TensorSource(
                    key: key,
                    expert: match.expert,
                    fileURL: url,
                    byteOffset: header.dataBase + offsets[0],
                    byteLength: offsets[1] - offsets[0],
                    dtype: dtype,
                    shape: shape)
                pending.append((match.outputKey, source))
            }
        }

        for (outputKey, source) in pending where !existingStackedKeys.contains(outputKey) {
            let expert = source.expert
            var group = groups[outputKey] ?? TensorGroup(outputKey: outputKey)
            group.byExpert[expert] = source
            groups[outputKey] = group
        }

        groups = groups.filter { _, group in
            guard let maxExpert = group.byExpert.keys.max(), group.byExpert[0] != nil else {
                return false
            }
            return (0...maxExpert).allSatisfy { group.byExpert[$0] != nil }
        }

        return BundleScan(files: files.sorted { $0.url.path < $1.url.path }, groups: groups)
    }

    private struct HeaderRead {
        var dataBase: UInt64
        var metadata: [String: Any]?
        var tensors: [String: [String: Any]]
    }

    private static func readSafetensorsHeader(_ url: URL) throws -> HeaderRead {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let prefix = try handle.read(upToCount: 8) ?? Data()
        guard prefix.count == 8 else {
            throw NSError(domain: "JangPressPrestacker", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "short safetensors header: \(url.path)"])
        }
        let headerLength = prefix.withUnsafeBytes {
            UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
        }
        let headerData = try handle.read(upToCount: Int(headerLength)) ?? Data()
        guard headerData.count == Int(headerLength) else {
            throw NSError(domain: "JangPressPrestacker", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "truncated safetensors header: \(url.path)"])
        }
        let json = try JSONSerialization.jsonObject(with: headerData)
        guard let dict = json as? [String: Any] else {
            throw NSError(domain: "JangPressPrestacker", code: 3,
                userInfo: [NSLocalizedDescriptionKey: "invalid safetensors JSON: \(url.path)"])
        }
        let metadata = dict["__metadata__"] as? [String: Any]
        var tensors: [String: [String: Any]] = [:]
        for (key, value) in dict where key != "__metadata__" {
            guard let item = value as? [String: Any] else { continue }
            var normalized = item
            if let shape = item["shape"] as? [NSNumber] {
                normalized["shape"] = shape.map { $0.intValue }
            }
            if let offsets = item["data_offsets"] as? [NSNumber] {
                normalized["data_offsets"] = offsets.map { $0.uint64Value }
            }
            tensors[key] = normalized
        }
        return HeaderRead(dataBase: 8 + headerLength, metadata: metadata, tensors: tensors)
    }

    private static func buildWritePlan(from groups: [String: TensorGroup]) throws -> [WriteTensor] {
        var result: [WriteTensor] = []
        for group in groups.values.sorted(by: { $0.outputKey < $1.outputKey }) {
            let experts = group.byExpert.keys.sorted()
            guard experts.first == 0, experts.last == experts.count - 1 else { continue }
            let sources = experts.compactMap { group.byExpert[$0] }
            guard let first = sources.first else { continue }
            guard sources.allSatisfy({ $0.dtype == first.dtype && $0.shape == first.shape }) else {
                continue
            }
            let perTensorBytes = sources.reduce(UInt64(0)) { $0 + $1.byteLength }
            let expectedBytes = UInt64(sources.count) * first.byteLength
            guard perTensorBytes == expectedBytes else { continue }
            result.append(WriteTensor(
                key: group.outputKey,
                dtype: first.dtype,
                shape: [sources.count] + first.shape,
                sources: sources))
        }
        var offset: UInt64 = 0
        for i in result.indices {
            result[i].dataOffset = offset
            offset += result[i].totalBytes
        }
        return result
    }

    private static func writeSafetensors(plan: [WriteTensor], to outputURL: URL) throws {
        let tmpURL = outputURL.deletingLastPathComponent()
            .appendingPathComponent(".\(outputURL.lastPathComponent).tmp-\(UUID().uuidString)")
        FileManager.default.createFile(atPath: tmpURL.path, contents: nil)

        let outFD = systemOpen(tmpURL.path, O_WRONLY | O_TRUNC)
        guard outFD >= 0 else {
            throw posixError("open", path: tmpURL.path)
        }
        defer { _ = systemClose(outFD) }

        var header: [String: Any] = [
            "__metadata__": [
                "format": "pt",
                "jangpress_prestacked": "1",
                "jangpress_prestack_version": version,
            ]
        ]
        for item in plan {
            header[item.key] = [
                "dtype": item.dtype,
                "shape": item.shape,
                "data_offsets": [item.dataOffset, item.dataOffset + item.totalBytes],
            ]
        }
        var headerData = try JSONSerialization.data(
            withJSONObject: header, options: [.sortedKeys])
        let dataBaseAlignment = 4096
        let misalignment = (8 + headerData.count) % dataBaseAlignment
        if misalignment != 0 {
            headerData.append(Data(
                repeating: 0x20,
                count: dataBaseAlignment - misalignment))
        }
        var headerLength = UInt64(headerData.count).littleEndian
        try withUnsafeBytes(of: &headerLength) { raw in
            try writeAll(fd: outFD, raw.baseAddress!, raw.count)
        }
        try headerData.withUnsafeBytes { raw in
            try writeAll(fd: outFD, raw.baseAddress!, raw.count)
        }

        try copyTensorPayloads(plan: plan, outFD: outFD)
        try FileManager.default.moveItem(at: tmpURL, to: outputURL)
    }

    private static func copyTensorPayloads(plan: [WriteTensor], outFD: Int32) throws {
        let bufferSize = 4 * 1024 * 1024
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: bufferSize, alignment: 4096)
        defer { buffer.deallocate() }
        var fdCache: [String: Int32] = [:]
        defer {
            for fd in fdCache.values {
                _ = systemClose(fd)
            }
        }

        for item in plan {
            for source in item.sources {
                let path = source.fileURL.path
                let fd: Int32
                if let cached = fdCache[path] {
                    fd = cached
                } else {
                    let opened = systemOpen(path, O_RDONLY)
                    guard opened >= 0 else { throw posixError("open", path: path) }
                    fdCache[path] = opened
                    fd = opened
                }

                var remaining = source.byteLength
                var offset = source.byteOffset
                while remaining > 0 {
                    let toRead = min(UInt64(bufferSize), remaining)
                    let n = systemPread(fd, buffer, Int(toRead), Int64(offset))
                    guard n > 0 else {
                        throw posixError("pread", path: path)
                    }
                    try writeAll(fd: outFD, buffer, n)
                    remaining -= UInt64(n)
                    offset += UInt64(n)
                }
            }
        }
    }

    private static func writeAll(fd: Int32, _ pointer: UnsafeRawPointer, _ count: Int) throws {
        var written = 0
        while written < count {
            let n = systemWrite(fd, pointer.advanced(by: written), count - written)
            guard n > 0 else { throw posixError("write", path: "fd:\(fd)") }
            written += n
        }
    }

    private static func cacheDirectory(for originalURL: URL, files: [SourceFile]) throws -> URL {
        let env = ProcessInfo.processInfo.environment
        if ephemeralPrestackEnabled(env) {
            let root: URL
            if let override = env["MLXPRESS_PRESTACK_CACHE_DIR"]
                ?? env["JANGPRESS_PRESTACK_CACHE_DIR"],
                !override.isEmpty
            {
                root = URL(fileURLWithPath: override)
            } else {
                root = FileManager.default.temporaryDirectory
                    .appendingPathComponent("vmlx-swift-lm", isDirectory: true)
                    .appendingPathComponent("jangpress-prestack-ephemeral", isDirectory: true)
            }
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            let url = root.appendingPathComponent(
                originalURL.lastPathComponent + "-" + UUID().uuidString,
                isDirectory: true)
            registerEphemeralPrestackDirectory(url)
            log("using ephemeral prestack overlay \(url.path)")
            return url
        }

        let root: URL
        if let override = env["MLXPRESS_PRESTACK_CACHE_DIR"]
            ?? env["JANGPRESS_PRESTACK_CACHE_DIR"],
            !override.isEmpty
        {
            root = URL(fileURLWithPath: override)
        } else {
            root = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("vmlx-swift-lm", isDirectory: true)
                .appendingPathComponent("jangpress-prestack", isDirectory: true)
        }
        let hash = stableHash(([version, originalURL.path] + files.map {
            "\($0.url.lastPathComponent):\($0.size):\($0.modified)"
        }).joined(separator: "|"))
        return root
            .appendingPathComponent(originalURL.lastPathComponent + "-" + hash, isDirectory: true)
    }

    static func registerEphemeralPrestackDirectory(_ url: URL) {
        let path = cleanupPath(for: url)
        cleanupLock.lock()
        cleanupURLs.insert(path)
        if !cleanupInstalled {
            atexit(jangPressPrestackerCleanupAtExit)
            cleanupInstalled = true
        }
        cleanupLock.unlock()
    }

    @discardableResult
    public static func cleanupEphemeralPrestackDirectory(_ url: URL) -> Bool {
        let path = cleanupPath(for: url)
        cleanupLock.lock()
        let wasRegistered = cleanupURLs.remove(path) != nil
        cleanupLock.unlock()

        guard wasRegistered else { return false }
        try? FileManager.default.removeItem(atPath: path)
        log("removed ephemeral prestack overlay \(path)")
        return true
    }

    public static func cleanupEphemeralPrestackDirectories() {
        cleanupLock.lock()
        let paths = cleanupURLs
        cleanupURLs.removeAll(keepingCapacity: false)
        cleanupLock.unlock()

        for path in paths {
            try? FileManager.default.removeItem(atPath: path)
        }
    }

    private static func cleanupPath(for url: URL) -> String {
        url.standardizedFileURL.path
    }

    private static func ensureOverlayLinks(from source: URL, to cache: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: cache, withIntermediateDirectories: true)
        let entries = try fm.contentsOfDirectory(
            at: source, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        for entry in entries where entry.lastPathComponent != outputName {
            let dst = cache.appendingPathComponent(entry.lastPathComponent)
            if fm.fileExists(atPath: dst.path) { continue }
            try fm.createSymbolicLink(at: dst, withDestinationURL: entry)
        }
    }

    private static func writeManifest(plan: [WriteTensor], source: URL, to url: URL) throws {
        let total = plan.reduce(UInt64(0)) { $0 + $1.totalBytes }
        let object: [String: Any] = [
            "version": version,
            "source": source.path,
            "tensor_count": plan.count,
            "bytes": total,
            "created": Date().timeIntervalSince1970,
        ]
        let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: [.atomic])
    }

    // MARK: - Generic safetensors alignment overlay

    private struct AlignmentTensor {
        var key: String
        var dtype: String
        var shape: [Int]
        var sourceOffset: UInt64
        var byteLength: UInt64
        var dataOffset: UInt64 = 0
    }

    private struct AlignmentFilePlan {
        var source: SourceFile
        var metadata: [String: Any]?
        var tensors: [AlignmentTensor]
        var needsAlignment: Bool
        var containsRoutedTensor: Bool

        var payloadBytes: UInt64 {
            tensors.reduce(UInt64(0)) { $0 + $1.byteLength }
        }
    }

    private struct AlignmentScan {
        var files: [SourceFile]
        var plans: [AlignmentFilePlan]

        var needsAlignment: Bool {
            plans.contains { $0.needsAlignment }
        }

        var containsRoutedTensor: Bool {
            plans.contains { $0.containsRoutedTensor }
        }

        var rewriteBytes: UInt64 {
            plans.filter(\.needsAlignment).reduce(UInt64(0)) {
                $0 + $1.payloadBytes
            }
        }
    }

    private static func prepareAlignedBundleIfNeeded(
        _ directory: URL,
        originalIsJANGTQ: Bool,
        env: [String: String]
    ) throws -> URL {
        let alignSafetensors = env["MLXPRESS_ALIGN_SAFETENSORS"]
            ?? env["JANGPRESS_ALIGN_SAFETENSORS"]
        guard isEnabledFlag(alignSafetensors) else {
            return directory
        }
        // JANGTQ bundles already use the prestack overlay path. Rewriting
        // every source shard for those models can double very large cache
        // footprints, so keep it opt-in unless a focused investigation
        // requests it.
        let alignJANGTQ = env["MLXPRESS_ALIGN_JANGTQ"]
            ?? env["JANGPRESS_ALIGN_JANGTQ"]
        if originalIsJANGTQ && alignJANGTQ != "1" {
            return directory
        }

        let scan = try scanAlignmentBundle(directory)
        guard scan.containsRoutedTensor, scan.needsAlignment else {
            return directory
        }

        let cacheURL = try alignmentCacheDirectory(for: directory, files: scan.files)
        let manifestURL = cacheURL.appendingPathComponent(alignmentManifestName)
        if FileManager.default.fileExists(atPath: manifestURL.path) {
            log("using existing aligned safetensors overlay \(cacheURL.path)")
            return cacheURL
        }

        try FileManager.default.createDirectory(
            at: cacheURL, withIntermediateDirectories: true)
        let rewriteNames = Set(scan.plans.filter(\.needsAlignment).map {
            $0.source.url.lastPathComponent
        })
        try ensureAlignedOverlayLinks(from: directory, to: cacheURL, rewriting: rewriteNames)

        var rewritten = 0
        for plan in scan.plans where plan.needsAlignment {
            let outputURL = cacheURL.appendingPathComponent(plan.source.url.lastPathComponent)
            try writeAlignedSafetensors(plan: plan, to: outputURL)
            rewritten += 1
        }
        try writeAlignmentManifest(
            scan: scan,
            rewrittenCount: rewritten,
            source: directory,
            to: manifestURL)
        log(String(format:
            "wrote aligned safetensors overlay: files=%d payload=%.1f GB into %@",
            rewritten,
            Double(scan.rewriteBytes) / 1_073_741_824.0,
            cacheURL.path))
        return cacheURL
    }

    private static func scanAlignmentBundle(_ directory: URL) throws -> AlignmentScan {
        let fm = FileManager.default
        let entries = try fm.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey],
            options: [.skipsHiddenFiles])

        var files: [SourceFile] = []
        var plans: [AlignmentFilePlan] = []

        for url in entries where url.pathExtension == "safetensors" {
            let values = try url.resourceValues(
                forKeys: [.fileSizeKey, .contentModificationDateKey])
            let source = SourceFile(
                url: url,
                size: UInt64(values.fileSize ?? 0),
                modified: values.contentModificationDate?.timeIntervalSince1970 ?? 0)
            files.append(source)

            let header = try readSafetensorsHeader(url)
            var tensors: [AlignmentTensor] = []
            var needsAlignment = false
            var containsRouted = false
            for (key, value) in header.tensors {
                guard let dtype = value["dtype"] as? String,
                      let shape = value["shape"] as? [Int],
                      let offsets = value["data_offsets"] as? [UInt64],
                      offsets.count == 2,
                      offsets[1] >= offsets[0]
                else { continue }

                let sourceOffset = header.dataBase + offsets[0]
                let byteLength = offsets[1] - offsets[0]
                let alignment = UInt64(dtypeAlignment(dtype))
                if alignment > 1, sourceOffset % alignment != 0 {
                    needsAlignment = true
                }
                if isRoutedTensorKey(key) {
                    containsRouted = true
                }
                tensors.append(AlignmentTensor(
                    key: key,
                    dtype: dtype,
                    shape: shape,
                    sourceOffset: sourceOffset,
                    byteLength: byteLength))
            }

            plans.append(AlignmentFilePlan(
                source: source,
                metadata: header.metadata,
                tensors: tensors.sorted { lhs, rhs in
                    if lhs.sourceOffset == rhs.sourceOffset {
                        return lhs.key < rhs.key
                    }
                    return lhs.sourceOffset < rhs.sourceOffset
                },
                needsAlignment: needsAlignment,
                containsRoutedTensor: containsRouted))
        }

        return AlignmentScan(
            files: files.sorted { $0.url.path < $1.url.path },
            plans: plans.sorted { $0.source.url.path < $1.source.url.path })
    }

    private static func writeAlignedSafetensors(
        plan: AlignmentFilePlan,
        to outputURL: URL
    ) throws {
        let tmpURL = outputURL.deletingLastPathComponent()
            .appendingPathComponent(".\(outputURL.lastPathComponent).tmp-\(UUID().uuidString)")
        FileManager.default.createFile(atPath: tmpURL.path, contents: nil)

        let outFD = systemOpen(tmpURL.path, O_WRONLY | O_TRUNC)
        guard outFD >= 0 else {
            throw posixError("open", path: tmpURL.path)
        }
        defer { _ = systemClose(outFD) }

        let (headerData, arranged) = try buildAlignedHeader(plan: plan)
        var headerLength = UInt64(headerData.count).littleEndian
        try withUnsafeBytes(of: &headerLength) { raw in
            try writeAll(fd: outFD, raw.baseAddress!, raw.count)
        }
        try headerData.withUnsafeBytes { raw in
            try writeAll(fd: outFD, raw.baseAddress!, raw.count)
        }

        let inFD = systemOpen(plan.source.url.path, O_RDONLY)
        guard inFD >= 0 else {
            throw posixError("open", path: plan.source.url.path)
        }
        defer { _ = systemClose(inFD) }

        var currentOffset: UInt64 = 0
        for tensor in arranged {
            if tensor.dataOffset > currentOffset {
                try writeZeroPadding(
                    fd: outFD,
                    count: tensor.dataOffset - currentOffset)
                currentOffset = tensor.dataOffset
            }
            try copyByteRange(
                inFD: inFD,
                sourcePath: plan.source.url.path,
                sourceOffset: tensor.sourceOffset,
                byteLength: tensor.byteLength,
                outFD: outFD)
            currentOffset += tensor.byteLength
        }

        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }
        try FileManager.default.moveItem(at: tmpURL, to: outputURL)
    }

    private static func buildAlignedHeader(
        plan: AlignmentFilePlan
    ) throws -> (Data, [AlignmentTensor]) {
        var dataBase: UInt64 = 4096
        var arranged = plan.tensors
        var headerData = Data()

        for _ in 0..<8 {
            var dataOffset: UInt64 = 0
            for i in arranged.indices {
                let alignment = UInt64(dtypeAlignment(arranged[i].dtype))
                if alignment > 1 {
                    let absoluteMisalignment = (dataBase + dataOffset) % alignment
                    if absoluteMisalignment != 0 {
                        dataOffset += alignment - absoluteMisalignment
                    }
                }
                arranged[i].dataOffset = dataOffset
                dataOffset += arranged[i].byteLength
            }

            var header: [String: Any] = [:]
            if let metadata = plan.metadata {
                header["__metadata__"] = metadata
            }
            for tensor in arranged {
                header[tensor.key] = [
                    "dtype": tensor.dtype,
                    "shape": tensor.shape,
                    "data_offsets": [
                        tensor.dataOffset,
                        tensor.dataOffset + tensor.byteLength,
                    ],
                ]
            }

            headerData = try JSONSerialization.data(
                withJSONObject: header,
                options: [.sortedKeys])
            let dataBaseAlignment = 4096
            let misalignment = (8 + headerData.count) % dataBaseAlignment
            if misalignment != 0 {
                headerData.append(Data(
                    repeating: 0x20,
                    count: dataBaseAlignment - misalignment))
            }
            let newDataBase = UInt64(8 + headerData.count)
            if newDataBase == dataBase {
                return (headerData, arranged)
            }
            dataBase = newDataBase
        }

        return (headerData, arranged)
    }

    private static func copyByteRange(
        inFD: Int32,
        sourcePath: String,
        sourceOffset: UInt64,
        byteLength: UInt64,
        outFD: Int32
    ) throws {
        let bufferSize = 4 * 1024 * 1024
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: bufferSize, alignment: 4096)
        defer { buffer.deallocate() }

        var remaining = byteLength
        var offset = sourceOffset
        while remaining > 0 {
            let toRead = min(UInt64(bufferSize), remaining)
            let n = systemPread(inFD, buffer, Int(toRead), Int64(offset))
            guard n > 0 else {
                throw posixError("pread", path: sourcePath)
            }
            try writeAll(fd: outFD, buffer, n)
            remaining -= UInt64(n)
            offset += UInt64(n)
        }
    }

    private static func writeZeroPadding(fd: Int32, count: UInt64) throws {
        guard count > 0 else { return }
        let bufferSize = 4096
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: bufferSize, alignment: 4096)
        defer { buffer.deallocate() }
        buffer.initializeMemory(as: UInt8.self, repeating: 0, count: bufferSize)
        var remaining = count
        while remaining > 0 {
            let n = min(UInt64(bufferSize), remaining)
            try writeAll(fd: fd, buffer, Int(n))
            remaining -= n
        }
    }

    private static func ensureAlignedOverlayLinks(
        from source: URL,
        to cache: URL,
        rewriting: Set<String>
    ) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: cache, withIntermediateDirectories: true)
        let entries = try fm.contentsOfDirectory(
            at: source, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        for entry in entries {
            let name = entry.lastPathComponent
            if rewriting.contains(name) || name == alignmentManifestName {
                continue
            }
            let dst = cache.appendingPathComponent(name)
            if fm.fileExists(atPath: dst.path) { continue }
            try fm.createSymbolicLink(at: dst, withDestinationURL: entry)
        }
    }

    private static func writeAlignmentManifest(
        scan: AlignmentScan,
        rewrittenCount: Int,
        source: URL,
        to url: URL
    ) throws {
        let object: [String: Any] = [
            "version": alignmentVersion,
            "source": source.path,
            "file_count": scan.files.count,
            "rewritten_file_count": rewrittenCount,
            "payload_bytes": scan.rewriteBytes,
            "created": Date().timeIntervalSince1970,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: [.atomic])
    }

    private static func alignmentCacheDirectory(
        for originalURL: URL,
        files: [SourceFile]
    ) throws -> URL {
        let env = ProcessInfo.processInfo.environment
        let root: URL
        if let override = env["MLXPRESS_ALIGN_CACHE_DIR"]
            ?? env["JANGPRESS_ALIGN_CACHE_DIR"]
            ?? env["MLXPRESS_PRESTACK_CACHE_DIR"]
            ?? env["JANGPRESS_PRESTACK_CACHE_DIR"],
           !override.isEmpty
        {
            root = URL(fileURLWithPath: override)
        } else {
            root = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("vmlx-swift-lm", isDirectory: true)
                .appendingPathComponent("jangpress-align", isDirectory: true)
        }
        let hash = stableHash(([alignmentVersion, originalURL.path] + files.map {
            "\($0.url.lastPathComponent):\($0.size):\($0.modified)"
        }).joined(separator: "|"))
        return root
            .appendingPathComponent(originalURL.lastPathComponent + "-" + hash, isDirectory: true)
    }

    private static func dtypeAlignment(_ dtype: String) -> Int {
        switch dtype {
        case "F64", "I64", "U64", "C64": return 8
        case "F32", "I32", "U32": return 4
        case "F16", "BF16", "I16", "U16": return 2
        default: return 1
        }
    }

    private static func isRoutedTensorKey(_ key: String) -> Bool {
        if key.hasSuffix(".tq_bits") {
            return false
        }
        let payloadSuffixes = [
            ".weight", ".scales", ".biases", ".tq_packed", ".tq_norms",
        ]
        guard payloadSuffixes.contains(where: { key.contains($0) }) else {
            return false
        }
        let routedMarkers = [
            ".mlp.switch_mlp.",
            ".mlp.experts.",
            ".block_sparse_moe.switch_mlp.",
            ".block_sparse_moe.experts.",
            ".ffn.switch_mlp.",
            ".ffn.experts.",
            ".mixer.switch_mlp.",
            ".mixer.experts.",
        ]
        return routedMarkers.contains { key.contains($0) }
    }

    private struct KeyMatch {
        var expert: Int
        var outputKey: String
    }

    private static func matchPerExpertKey(_ key: String) -> KeyMatch? {
        if let m = match(
            key,
            #"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.(tq_packed|tq_norms)$"#
        ) {
            return KeyMatch(
                expert: Int(m[2])!,
                outputKey: "model.layers.\(m[1]).block_sparse_moe.switch_mlp.\(projName(m[3])).\(m[4])")
        }
        if let m = match(
            key,
            #"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(tq_packed|tq_norms)$"#
        ) {
            return KeyMatch(
                expert: Int(m[2])!,
                outputKey: "layers.\(m[1]).ffn.switch_mlp.\(projName(m[3])).\(m[4])")
        }
        if let m = match(
            key,
            #"^(language_model\.)?model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(w[123])\.(tq_packed|tq_norms)$"#
        ) {
            let root = m[1]
            return KeyMatch(
                expert: Int(m[3])!,
                outputKey: "\(root)model.layers.\(m[2]).mlp.switch_mlp.\(projName(m[4])).\(m[5])")
        }
        if let m = match(
            key,
            #"^(language_model\.)?model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(tq_packed|tq_norms)$"#
        ) {
            let root = m[1]
            return KeyMatch(
                expert: Int(m[3])!,
                outputKey: "\(root)model.layers.\(m[2]).mlp.switch_mlp.\(m[4]).\(m[5])")
        }
        if let m = match(
            key,
            #"^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up_proj|down_proj)\.(tq_packed|tq_norms)$"#
        ) {
            let fc = m[3] == "up_proj" ? "fc1" : "fc2"
            return KeyMatch(
                expert: Int(m[2])!,
                outputKey: "backbone.layers.\(m[1]).mixer.switch_mlp.\(fc).\(m[4])")
        }
        return nil
    }

    private static func isStackedRoutedKey(_ key: String) -> Bool {
        key.contains(".switch_mlp.") &&
            (key.hasSuffix(".tq_packed") || key.hasSuffix(".tq_norms"))
    }

    private static func projName(_ raw: String) -> String {
        switch raw {
        case "w1": return "gate_proj"
        case "w2": return "down_proj"
        case "w3": return "up_proj"
        default: return raw
        }
    }

    private static func match(_ value: String, _ pattern: String) -> [String]? {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return nil }
        let ns = value as NSString
        let range = NSRange(location: 0, length: ns.length)
        guard let m = re.firstMatch(in: value, range: range) else { return nil }
        var captures: [String] = []
        for i in 0..<m.numberOfRanges {
            let r = m.range(at: i)
            captures.append(r.location == NSNotFound ? "" : ns.substring(with: r))
        }
        return captures
    }

    private static func stableHash(_ input: String) -> String {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in input.utf8 {
            hash ^= UInt64(byte)
            hash &*= 0x100000001b3
        }
        return String(format: "%016llx", hash)
    }

    private static func log(_ message: String) {
        FileHandle.standardError.write(Data("[MLXPressPrestacker] \(message)\n".utf8))
    }

    private static func posixError(_ op: String, path: String) -> NSError {
        NSError(domain: "JangPressPrestacker", code: Int(errno),
            userInfo: [NSLocalizedDescriptionKey: "\(op)(\(path)) failed: \(String(cString: strerror(errno)))"])
    }

    #if canImport(Darwin)
    private static func systemOpen(_ path: String, _ flags: Int32) -> Int32 {
        Darwin.open(path, flags, 0)
    }
    private static func systemClose(_ fd: Int32) -> Int32 { Darwin.close(fd) }
    private static func systemPread(
        _ fd: Int32, _ buffer: UnsafeMutableRawPointer, _ count: Int, _ offset: Int64
    ) -> Int {
        Darwin.pread(fd, buffer, count, off_t(offset))
    }
    private static func systemWrite(
        _ fd: Int32, _ buffer: UnsafeRawPointer, _ count: Int
    ) -> Int {
        Darwin.write(fd, buffer, count)
    }
    #elseif canImport(Glibc)
    private static func systemOpen(_ path: String, _ flags: Int32) -> Int32 {
        Glibc.open(path, flags, 0)
    }
    private static func systemClose(_ fd: Int32) -> Int32 { Glibc.close(fd) }
    private static func systemPread(
        _ fd: Int32, _ buffer: UnsafeMutableRawPointer, _ count: Int, _ offset: Int64
    ) -> Int {
        Glibc.pread(fd, buffer, count, off_t(offset))
    }
    private static func systemWrite(
        _ fd: Int32, _ buffer: UnsafeRawPointer, _ count: Int
    ) -> Int {
        Glibc.write(fd, buffer, count)
    }
    #endif
}
