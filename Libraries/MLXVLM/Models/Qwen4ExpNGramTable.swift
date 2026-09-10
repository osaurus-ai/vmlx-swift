// Copyright © 2026 Apple Inc.

import Darwin
import Foundation
import MLXLMCommon

enum Qwen4ExpNGramTableError: Error, CustomStringConvertible {
    case missingIndex(URL)
    case missingTensor(String)
    case missingBitSpec(Int)
    case invalidTensor(String, String)
    case rowOutOfRange(Int64)
    case openFailed(URL, Int32)
    case statFailed(URL, Int32)
    case noCacheFailed(URL, Int32)
    case readFailed(URL, Int32)
    case truncatedHeader(URL)
    case malformedHeader(URL, String)

    var description: String {
        switch self {
        case .missingIndex(let url):
            return "qwen4_exp PLE index is missing at \(url.path)"
        case .missingTensor(let name):
            return "qwen4_exp PLE tensor is missing: \(name)"
        case .missingBitSpec(let shard):
            return "qwen4_exp PLE bit_map has no affine spec for shard \(shard)"
        case .invalidTensor(let name, let reason):
            return "qwen4_exp PLE tensor \(name) is invalid: \(reason)"
        case .rowOutOfRange(let row):
            return "qwen4_exp PLE row \(row) is outside the table"
        case .openFailed(let url, let code):
            return "qwen4_exp PLE open failed for \(url.lastPathComponent), errno=\(code)"
        case .statFailed(let url, let code):
            return "qwen4_exp PLE fstat failed for \(url.lastPathComponent), errno=\(code)"
        case .noCacheFailed(let url, let code):
            return "qwen4_exp PLE F_NOCACHE failed for \(url.lastPathComponent), errno=\(code)"
        case .readFailed(let url, let code):
            return "qwen4_exp PLE pread failed for \(url.lastPathComponent), errno=\(code)"
        case .truncatedHeader(let url):
            return "qwen4_exp PLE safetensors header is truncated in \(url.lastPathComponent)"
        case .malformedHeader(let url, let reason):
            return "qwen4_exp PLE safetensors header is malformed in \(url.lastPathComponent): \(reason)"
        }
    }
}

private struct Qwen4ExpTensorDescriptor: Sendable {
    let dtype: String
    let shape: [Int]
    let dataOffset: UInt64
}

/// Minimal safetensors descriptor reader for Qwen PLE.
///
/// This is deliberately independent of model residency tiers and never mmaps payloads.
/// The header and requested rows are read with `pread` from an `F_NOCACHE`
/// descriptor so the multi-gigabyte n-gram table remains an SSD lookup table.
private final class Qwen4ExpSafetensorsReader: @unchecked Sendable {
    let url: URL
    let fileSize: UInt64
    private let fd: Int32
    private let tensors: [String: Qwen4ExpTensorDescriptor]

    init(url: URL) throws {
        let opened = Darwin.open(url.path, O_RDONLY)
        guard opened >= 0 else {
            throw Qwen4ExpNGramTableError.openFailed(url, errno)
        }
        do {
            var status = stat()
            guard fstat(opened, &status) == 0 else {
                throw Qwen4ExpNGramTableError.statFailed(url, errno)
            }
            let size = UInt64(status.st_size)
            guard fcntl(opened, F_NOCACHE, 1) == 0 else {
                throw Qwen4ExpNGramTableError.noCacheFailed(url, errno)
            }
            guard size >= 8 else {
                throw Qwen4ExpNGramTableError.truncatedHeader(url)
            }
            let lengthData = try Self.readExactly(
                fd: opened, url: url, fileSize: size, range: 0 ..< 8)
            let headerSize = lengthData.withUnsafeBytes {
                UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
            }
            guard headerSize <= size - 8, headerSize <= UInt64(Int.max) else {
                throw Qwen4ExpNGramTableError.truncatedHeader(url)
            }
            let header = try Self.readExactly(
                fd: opened, url: url, fileSize: size, range: 8 ..< (8 + headerSize))
            guard let object = try JSONSerialization.jsonObject(with: header) as? [String: Any]
            else {
                throw Qwen4ExpNGramTableError.malformedHeader(url, "not a JSON object")
            }
            let dataStart = 8 + headerSize
            var parsed: [String: Qwen4ExpTensorDescriptor] = [:]
            parsed.reserveCapacity(object.count)
            for (name, raw) in object where name != "__metadata__" {
                guard let entry = raw as? [String: Any],
                    let dtype = entry["dtype"] as? String,
                    let shapeValues = entry["shape"] as? [Any],
                    let offsets = entry["data_offsets"] as? [Any], offsets.count == 2,
                    let start = (offsets[0] as? NSNumber)?.uint64Value,
                    let end = (offsets[1] as? NSNumber)?.uint64Value,
                    start <= end, dataStart + end <= size
                else { continue }
                let shape = shapeValues.compactMap { ($0 as? NSNumber)?.intValue }
                guard shape.count == shapeValues.count else { continue }
                parsed[name] = Qwen4ExpTensorDescriptor(
                    dtype: dtype, shape: shape, dataOffset: dataStart + start)
            }
            self.url = url
            self.fileSize = size
            self.fd = opened
            self.tensors = parsed
        } catch {
            Darwin.close(opened)
            throw error
        }
    }

    deinit { Darwin.close(fd) }

    func descriptor(for name: String) -> Qwen4ExpTensorDescriptor? {
        tensors[name]
    }

    func readBytes(range: Range<UInt64>) throws -> Data {
        try Self.readExactly(fd: fd, url: url, fileSize: fileSize, range: range)
    }

    private static func readExactly(
        fd: Int32, url: URL, fileSize: UInt64, range: Range<UInt64>
    ) throws -> Data {
        guard range.lowerBound <= range.upperBound,
            range.upperBound <= fileSize,
            range.count <= UInt64(Int.max)
        else {
            throw Qwen4ExpNGramTableError.readFailed(url, EINVAL)
        }
        let count = Int(range.count)
        var data = Data(count: count)
        var completed = 0
        while completed < count {
            let amount = data.withUnsafeMutableBytes { bytes -> Int in
                guard let base = bytes.baseAddress else { return 0 }
                return Darwin.pread(
                    fd, base.advanced(by: completed), count - completed,
                    off_t(range.lowerBound) + off_t(completed))
            }
            guard amount > 0 else {
                throw Qwen4ExpNGramTableError.readFailed(
                    url, amount == 0 ? EIO : errno)
            }
            completed += amount
        }
        return data
    }
}

/// Page-granular reader for the Qwen 3.8 Flash Next PLE table.
///
/// The table is roughly 95 GiB before quantization. Loading its shard tensors
/// as ordinary MLX arrays makes a tiny gather fault hundreds of MiB per shard.
/// This reader keeps each safetensors file open with `F_NOCACHE` and reads only
/// requested packed rows plus their FP16 affine scale/bias rows. It supports
/// the real mixed-width bundle ABI (including 3-bit rows), rather than assuming
/// one table width from the profile name.
final class Qwen4ExpNGramTable: @unchecked Sendable {
    private final class ParallelGatherBuffers: @unchecked Sendable {
        let values: UnsafeMutablePointer<Float>
        let bytes: UnsafeMutablePointer<UInt64>
        let valueCount: Int
        let rowCount: Int
        let errorLock = NSLock()
        var firstError: Error?

        init(valueCount: Int, rowCount: Int) {
            self.valueCount = valueCount
            self.rowCount = rowCount
            values = .allocate(capacity: valueCount)
            values.initialize(repeating: 0, count: valueCount)
            bytes = .allocate(capacity: rowCount)
            bytes.initialize(repeating: 0, count: rowCount)
        }

        deinit {
            values.deinitialize(count: valueCount)
            values.deallocate()
            bytes.deinitialize(count: rowCount)
            bytes.deallocate()
        }

        func record(_ error: Error) {
            errorLock.lock()
            if firstError == nil { firstError = error }
            errorLock.unlock()
        }
    }

    private static let parallelRows: Bool = {
        let raw = ProcessInfo.processInfo.environment[
            "VMLX_QWEN4_PLE_PARALLEL_ROWS"] ?? "1"
        return raw != "0" && raw.lowercased() != "false"
    }()
    struct AffineSpec: Equatable, Sendable {
        let bits: Int
        let groupSize: Int
    }

    struct IOStats: Equatable, Sendable {
        let gatherCalls: UInt64
        let rowsRead: UInt64
        let payloadBytesRead: UInt64
        let backingFileCount: Int
        let backingFileBytes: UInt64
        let noCacheFileCount: Int
    }

    private struct IndexFile: Decodable {
        let weightMap: [String: String]

        enum CodingKeys: String, CodingKey {
            case weightMap = "weight_map"
        }
    }

    private struct RowShard: @unchecked Sendable {
        let weightFile: Qwen4ExpSafetensorsReader
        let scaleFile: Qwen4ExpSafetensorsReader?
        let biasFile: Qwen4ExpSafetensorsReader?
        let weight: Qwen4ExpTensorDescriptor
        let scales: Qwen4ExpTensorDescriptor?
        let biases: Qwen4ExpTensorDescriptor?
        let spec: AffineSpec?
        let rows: Int
        let dimensions: Int
    }

    private let shards: [RowShard]
    private let rowStarts: [Int64]
    private let sourceDirectory: URL
    private let backingFileCount: Int
    private let backingFileBytes: UInt64
    private let noCacheFileCount: Int
    private let telemetryLock = NSLock()
    private var gatherCalls: UInt64 = 0
    private var rowsRead: UInt64 = 0
    private var payloadBytesRead: UInt64 = 0
    let dimensions: Int
    let rowCount: Int64

    init(
        modelDirectory: URL,
        layerIndex: Int,
        shardCount: Int
    ) throws {
        let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
        guard let indexData = try? Data(contentsOf: indexURL) else {
            throw Qwen4ExpNGramTableError.missingIndex(indexURL)
        }
        let index = try JSONDecoder().decode(IndexFile.self, from: indexData)
        let specs = try Self.readAffineSpecs(
            modelDirectory: modelDirectory, layerIndex: layerIndex)

        var openedFiles: [URL: Qwen4ExpSafetensorsReader] = [:]
        var built: [RowShard] = []
        built.reserveCapacity(shardCount)
        var starts: [Int64] = []
        starts.reserveCapacity(shardCount)
        var nextStart: Int64 = 0
        var commonDimensions: Int?

        for shardIndex in 0 ..< shardCount {
            let base = "language_model.layers.\(layerIndex).ple.ngram_embedding.shards.\(shardIndex)"
            let weightName = base + ".weight"
            guard let filename = index.weightMap[weightName] else {
                throw Qwen4ExpNGramTableError.missingTensor(weightName)
            }
            let fileURL = modelDirectory.appendingPathComponent(filename)
            func opened(_ url: URL) throws -> Qwen4ExpSafetensorsReader {
                if let existing = openedFiles[url] { return existing }
                let opened = try Qwen4ExpSafetensorsReader(url: url)
                openedFiles[url] = opened
                return opened
            }
            let file = try opened(fileURL)
            guard let weight = file.descriptor(for: weightName), weight.shape.count == 2 else {
                throw Qwen4ExpNGramTableError.missingTensor(weightName)
            }

            let rows = weight.shape[0]
            let spec: AffineSpec?
            let scales: Qwen4ExpTensorDescriptor?
            let biases: Qwen4ExpTensorDescriptor?
            let scaleFile: Qwen4ExpSafetensorsReader?
            let biasFile: Qwen4ExpSafetensorsReader?
            let dimensions: Int
            if weight.dtype == "U32" {
                guard let affine = specs[shardIndex] else {
                    throw Qwen4ExpNGramTableError.missingBitSpec(shardIndex)
                }
                let scaleName = base + ".scales"
                let biasName = base + ".biases"
                guard let scaleFileName = index.weightMap[scaleName],
                    let biasFileName = index.weightMap[biasName]
                else {
                    throw Qwen4ExpNGramTableError.missingTensor(scaleName + " / " + biasName)
                }
                let scaleOwner = try opened(modelDirectory.appendingPathComponent(scaleFileName))
                let biasOwner = try opened(modelDirectory.appendingPathComponent(biasFileName))
                guard let scale = scaleOwner.descriptor(for: scaleName),
                    let bias = biasOwner.descriptor(for: biasName)
                else { throw Qwen4ExpNGramTableError.missingTensor(scaleName + " / " + biasName) }
                dimensions = weight.shape[1] * 32 / affine.bits
                guard dimensions * affine.bits == weight.shape[1] * 32,
                    dimensions % affine.groupSize == 0,
                    scale.dtype == "F16", bias.dtype == "F16",
                    scale.shape == [rows, dimensions / affine.groupSize],
                    bias.shape == scale.shape
                else {
                    throw Qwen4ExpNGramTableError.invalidTensor(
                        weightName, "packed/scales/biases shapes disagree with bit_map")
                }
                spec = affine
                scales = scale
                biases = bias
                scaleFile = scaleOwner
                biasFile = biasOwner
            } else if weight.dtype == "F16" || weight.dtype == "BF16" || weight.dtype == "F32" {
                dimensions = weight.shape[1]
                spec = nil
                scales = nil
                biases = nil
                scaleFile = nil
                biasFile = nil
            } else {
                throw Qwen4ExpNGramTableError.invalidTensor(
                    weightName, "unsupported dtype \(weight.dtype)")
            }

            if let commonDimensions, commonDimensions != dimensions {
                throw Qwen4ExpNGramTableError.invalidTensor(
                    weightName, "dimension \(dimensions) differs from \(commonDimensions)")
            }
            commonDimensions = dimensions
            starts.append(nextStart)
            nextStart += Int64(rows)
            built.append(RowShard(
                weightFile: file, scaleFile: scaleFile, biasFile: biasFile,
                weight: weight, scales: scales, biases: biases,
                spec: spec, rows: rows, dimensions: dimensions))
        }

        self.shards = built
        self.rowStarts = starts
        self.sourceDirectory = modelDirectory
        self.backingFileCount = openedFiles.count
        self.backingFileBytes = openedFiles.values.reduce(0) { $0 + $1.fileSize }
        self.noCacheFileCount = openedFiles.count
        self.dimensions = commonDimensions ?? 0
        self.rowCount = nextStart
        Self.log(String(format:
            "ssd-row-reader ready backend=pread-fnocache cache=F_NOCACHE row_schedule=%@ files=%d backing_file_bytes=%.3f_GiB rows=%lld dimensions=%d source=%@",
            Self.parallelRows ? "parallel" : "sequential-opt-out",
            backingFileCount,
            Double(backingFileBytes) / 1_073_741_824.0,
            rowCount,
            dimensions,
            modelDirectory.path))
    }

    /// Returns row-major Float32 values. Only bytes belonging to selected rows
    /// are touched. Duplicate output rows are preserved, but each distinct row
    /// is read and dequantized once per gather. No data is retained across calls.
    func gather(_ rows: [Int64], parallelRows: Bool? = nil) throws -> [Float] {
        // Avoid dictionary/scatter allocation for single rows or an already
        // strictly increasing set. A decode token has multiple n-gram heads;
        // disjoint ordered head ranges need no duplicate detection dictionary.
        if rows.count <= 1 || zip(rows, rows.dropFirst()).allSatisfy({ $0.0 < $0.1 }) {
            let (values, bytesRead) = try gatherDistinct(rows, parallelRows: parallelRows)
            recordGather(rows: rows.count, bytesRead: bytesRead)
            return values
        }
        var uniqueRows: [Int64] = []
        var positions: [Int64: Int] = [:]
        var outputPositions: [Int] = []
        uniqueRows.reserveCapacity(rows.count)
        outputPositions.reserveCapacity(rows.count)
        for row in rows {
            guard row >= 0, row < rowCount else {
                throw Qwen4ExpNGramTableError.rowOutOfRange(row)
            }
            if let position = positions[row] {
                outputPositions.append(position)
            } else {
                positions[row] = uniqueRows.count
                outputPositions.append(uniqueRows.count)
                uniqueRows.append(row)
            }
        }
        let (values, bytesRead) = try gatherDistinct(
            uniqueRows, parallelRows: parallelRows)
        // rowsRead remains the logical requested-row count; payloadBytesRead
        // measures actual selected-row I/O, excluding duplicate reads avoided.
        recordGather(rows: rows.count, bytesRead: bytesRead)
        guard uniqueRows.count != rows.count else { return values }
        var output = [Float](repeating: 0, count: rows.count * dimensions)
        for (outputRow, position) in outputPositions.enumerated() {
            output.replaceSubrange(
                outputRow * dimensions ..< (outputRow + 1) * dimensions,
                with: values[position * dimensions ..< (position + 1) * dimensions])
        }
        return output
    }

    private func gatherDistinct(
        _ rows: [Int64], parallelRows: Bool?
    ) throws -> ([Float], UInt64) {
        if parallelRows ?? Self.parallelRows, rows.count > 1 {
            return try gatherParallel(rows)
        }
        var output = [Float](repeating: 0, count: rows.count * dimensions)
        var bytesRead: UInt64 = 0
        for (outputRow, row) in rows.enumerated() {
            guard row >= 0, row < rowCount else {
                throw Qwen4ExpNGramTableError.rowOutOfRange(row)
            }
            let shardIndex = Self.upperBound(rowStarts, row) - 1
            let localRow = Int(row - rowStarts[shardIndex])
            let (values, rowBytes) = try Self.readRow(
                shards[shardIndex], row: localRow)
            bytesRead += rowBytes
            output.replaceSubrange(
                outputRow * dimensions ..< (outputRow + 1) * dimensions,
                with: values)
        }
        return (output, bytesRead)
    }

    private func gatherParallel(_ rows: [Int64]) throws -> ([Float], UInt64) {
        let buffers = ParallelGatherBuffers(
            valueCount: rows.count * dimensions, rowCount: rows.count)
        DispatchQueue.concurrentPerform(iterations: rows.count) { outputRow in
            do {
                let row = rows[outputRow]
                guard row >= 0, row < rowCount else {
                    throw Qwen4ExpNGramTableError.rowOutOfRange(row)
                }
                let shardIndex = Self.upperBound(rowStarts, row) - 1
                let localRow = Int(row - rowStarts[shardIndex])
                let (values, rowBytes) = try Self.readRow(
                    shards[shardIndex], row: localRow)
                values.withUnsafeBufferPointer { source in
                    buffers.values.advanced(by: outputRow * dimensions)
                        .update(from: source.baseAddress!, count: dimensions)
                }
                buffers.bytes[outputRow] = rowBytes
            } catch {
                buffers.record(error)
            }
        }
        if let error = buffers.firstError { throw error }
        let output = Array(UnsafeBufferPointer(
            start: buffers.values, count: rows.count * dimensions))
        let bytesRead = (0 ..< rows.count).reduce(UInt64(0)) {
            $0 + buffers.bytes[$1]
        }
        return (output, bytesRead)
    }

    func ioStats() -> IOStats {
        telemetryLock.lock()
        defer { telemetryLock.unlock() }
        return IOStats(
            gatherCalls: gatherCalls,
            rowsRead: rowsRead,
            payloadBytesRead: payloadBytesRead,
            backingFileCount: backingFileCount,
            backingFileBytes: backingFileBytes,
            noCacheFileCount: noCacheFileCount)
    }

    private func recordGather(rows: Int, bytesRead: UInt64) {
        telemetryLock.lock()
        gatherCalls += 1
        rowsRead += UInt64(rows)
        payloadBytesRead += bytesRead
        let calls = gatherCalls
        let totalRows = rowsRead
        let totalBytes = payloadBytesRead
        telemetryLock.unlock()
        if calls == 1 || calls % 128 == 0 {
            Self.log(
                "ssd-row-read backend=pread-fnocache cache=F_NOCACHE "
                    + "call=\(calls) rows=\(rows) bytes=\(bytesRead) "
                    + "cumulative_rows=\(totalRows) cumulative_bytes=\(totalBytes) "
                    + "source=\(sourceDirectory.path)")
        }
    }

    private static func readRow(
        _ shard: RowShard, row: Int
    ) throws -> ([Float], UInt64) {
        if let spec = shard.spec, let scales = shard.scales, let biases = shard.biases {
            let packedWords = shard.weight.shape[1]
            let packedByteCount = packedWords * 4
            let packedStart = shard.weight.dataOffset + UInt64(row * packedByteCount)
            let groupCount = shard.dimensions / spec.groupSize
            let affineByteCount = groupCount * 2
            let scaleStart = scales.dataOffset + UInt64(row * affineByteCount)
            let biasStart = biases.dataOffset + UInt64(row * affineByteCount)
            let packed = try readPayload(
                shard.weightFile,
                range: packedStart..<(packedStart + UInt64(packedByteCount)))
            let scaleData = try readPayload(
                shard.scaleFile!,
                range: scaleStart..<(scaleStart + UInt64(affineByteCount)))
            let biasData = try readPayload(
                shard.biasFile!,
                range: biasStart..<(biasStart + UInt64(affineByteCount)))
            let result = packed.withUnsafeBytes { packedBytes in
                scaleData.withUnsafeBytes { scaleBytes in
                    biasData.withUnsafeBytes { biasBytes in
                        dequantizeRow(
                            packedPointer: packedBytes.baseAddress!, packedWordCount: packedWords,
                            scalePointer: scaleBytes.baseAddress!, biasPointer: biasBytes.baseAddress!,
                            dimensions: shard.dimensions, spec: spec)
                    }
                }
            }
            return (result, UInt64(packedByteCount + 2 * affineByteCount))
        }

        let bytesPerValue = shard.weight.dtype == "F32" ? 4 : 2
        let rowByteCount = shard.dimensions * bytesPerValue
        let rowStart = shard.weight.dataOffset + UInt64(row * rowByteCount)
        let rowData = try readPayload(
            shard.weightFile,
            range: rowStart..<(rowStart + UInt64(rowByteCount)))
        let values = rowData.withUnsafeBytes { rowBytes in
            (0 ..< shard.dimensions).map { column in
                switch shard.weight.dtype {
                case "F32":
                    return rowBytes.loadUnaligned(fromByteOffset: column * 4, as: Float.self)
                case "F16":
                    let bits = rowBytes.loadUnaligned(fromByteOffset: column * 2, as: UInt16.self)
                    return Float(Float16(bitPattern: UInt16(littleEndian: bits)))
                default: // BF16
                    let bits = rowBytes.loadUnaligned(fromByteOffset: column * 2, as: UInt16.self)
                    return Float(bitPattern: UInt32(UInt16(littleEndian: bits)) << 16)
                }
            }
        }
        return (values, UInt64(rowByteCount))
    }

    private static func readPayload(
        _ file: Qwen4ExpSafetensorsReader,
        range: Range<UInt64>
    ) throws -> Data {
        try file.readBytes(range: range)
    }

    static func dequantizeRow(
        packedWords: [UInt32], scales: [Float], biases: [Float],
        dimensions: Int, spec: AffineSpec
    ) -> [Float] {
        packedWords.withUnsafeBytes { packed in
            scales.withUnsafeBytes { scaleBytes in
                biases.withUnsafeBytes { biasBytes in
                    // Test helper uses Float32 metadata; keep its bit-unpack path
                    // identical while applying the supplied values directly.
                    var result = [Float](repeating: 0, count: dimensions)
                    let mask = UInt64((1 << spec.bits) - 1)
                    for column in 0 ..< dimensions {
                        let bitOffset = column * spec.bits
                        let word = bitOffset / 32
                        let shift = bitOffset % 32
                        let lo = UInt64(UInt32(littleEndian:
                            packed.loadUnaligned(fromByteOffset: word * 4, as: UInt32.self)))
                        let hi: UInt64 = word + 1 < packedWords.count
                            ? UInt64(UInt32(littleEndian:
                                packed.loadUnaligned(fromByteOffset: (word + 1) * 4, as: UInt32.self)))
                            : 0
                        let code = Float((lo | (hi << 32)) >> shift & mask)
                        let group = column / spec.groupSize
                        result[column] = code * scales[group] + biases[group]
                    }
                    _ = scaleBytes
                    _ = biasBytes
                    return result
                }
            }
        }
    }

    private static func dequantizeRow(
        packedPointer: UnsafeRawPointer, packedWordCount: Int,
        scalePointer: UnsafeRawPointer, biasPointer: UnsafeRawPointer,
        dimensions: Int, spec: AffineSpec
    ) -> [Float] {
        let mask = UInt64((1 << spec.bits) - 1)
        var result = [Float](repeating: 0, count: dimensions)
        for column in 0 ..< dimensions {
            let bitOffset = column * spec.bits
            let word = bitOffset / 32
            let shift = bitOffset % 32
            let lo = UInt64(UInt32(littleEndian:
                packedPointer.loadUnaligned(fromByteOffset: word * 4, as: UInt32.self)))
            let hi: UInt64 = word + 1 < packedWordCount
                ? UInt64(UInt32(littleEndian:
                    packedPointer.loadUnaligned(fromByteOffset: (word + 1) * 4, as: UInt32.self)))
                : 0
            let code = Float((lo | (hi << 32)) >> shift & mask)
            let group = column / spec.groupSize
            let scaleBits = scalePointer.loadUnaligned(
                fromByteOffset: group * 2, as: UInt16.self)
            let biasBits = biasPointer.loadUnaligned(
                fromByteOffset: group * 2, as: UInt16.self)
            let scale = Float(Float16(bitPattern: UInt16(littleEndian: scaleBits)))
            let bias = Float(Float16(bitPattern: UInt16(littleEndian: biasBits)))
            result[column] = code * scale + bias
        }
        return result
    }

    private static func upperBound(_ values: [Int64], _ needle: Int64) -> Int {
        var low = 0
        var high = values.count
        while low < high {
            let mid = (low + high) / 2
            if values[mid] <= needle { low = mid + 1 } else { high = mid }
        }
        return low
    }

    private static func readAffineSpecs(
        modelDirectory: URL, layerIndex: Int
    ) throws -> [Int: AffineSpec] {
        let configURL = modelDirectory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let jang = root["jang_config"] as? [String: Any],
            let bitMap = jang["bit_map"] as? [String: Any]
        else { return [:] }

        var result: [Int: AffineSpec] = [:]
        for (key, value) in bitMap {
            guard key.contains("ple.ngram_embedding.shards."), key.hasSuffix(".weight"),
                let dict = value as? [String: Any],
                let bits = (dict["bits"] as? NSNumber)?.intValue,
                let groupSize = (dict["group_size"] as? NSNumber)?.intValue
            else { continue }
            let components = key.split(separator: ".")
            guard let marker = components.firstIndex(of: "shards"), marker + 1 < components.count,
                let shard = Int(components[marker + 1])
            else { continue }
            if key.contains("layers.*") || key.contains("layers.\(layerIndex)") {
                result[shard] = AffineSpec(bits: bits, groupSize: groupSize)
            }
        }
        return result
    }

    private static func log(_ message: String) {
        FileHandle.standardError.write(Data("[Qwen4ExpPLE] \(message)\n".utf8))
    }
}
