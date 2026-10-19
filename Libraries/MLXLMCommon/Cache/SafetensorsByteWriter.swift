// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Cmlx
import Foundation
import MLX

/// Writes a safetensors file from tensor bytes that are already in CPU-visible memory, without calling
/// into MLX.
///
/// `DiskCache` stores through MLX's own `save`, which submits Metal work and therefore has to run on
/// the thread that owns MLX state, under the process-wide cache IO lock. The write-behind tier takes
/// the bytes of evaluated arrays on that thread (no copy, contiguous or strided, see ``payload(of:)``),
/// and this writer puts them on disk from a background queue: plain file I/O only.
///
/// Format (safetensors): little-endian u64 header length, a JSON header naming each tensor's dtype,
/// shape and `data_offsets` into the byte buffer that follows (plus `__metadata__`), padded with
/// spaces to an 8-byte boundary, then the tensor bytes in offset order.
enum SafetensorsByteWriter {
    struct Tensor {
        let name: String
        let dtype: String
        let shape: [Int]
        let payload: Payload

        /// Bytes this tensor occupies in the file.
        var byteCount: Int {
            switch payload {
            case .contiguous(let data): return data.count
            case .strided(_, let chunkBytes, let offsets): return chunkBytes * offsets.count
            }
        }
    }

    /// Where a tensor's bytes are. `.strided` is a non-contiguous view (e.g. a `..<offset` slice of a
    /// step-padded KV buffer `[1, H, capacity, D]`): `offsets.count` runs of `chunkBytes` contiguous
    /// bytes each, at byte offsets from `base`, in row-major order. Written run by run, so the file gets
    /// exactly the bytes a contiguous copy would hold without building that copy.
    enum Payload {
        case contiguous(Data)
        case strided(base: UnsafeRawPointer, chunkBytes: Int, offsets: [Int])
    }

    enum WriteError: Error {
        case unsupportedDType(String, DType)
        case byteCountMismatch(String)
        case cannotCreate(String)
    }

    static func dtypeName(_ dtype: DType) -> String? {
        switch dtype {
        case .bool: return "BOOL"
        case .uint8: return "U8"
        case .uint16: return "U16"
        case .uint32: return "U32"
        case .uint64: return "U64"
        case .int8: return "I8"
        case .int16: return "I16"
        case .int32: return "I32"
        case .int64: return "I64"
        case .float16: return "F16"
        case .float32: return "F32"
        case .bfloat16: return "BF16"
        case .float64: return "F64"
        case .complex64: return nil
        }
    }

    /// Bytes of each evaluated array, taken on the MLX-owning thread, WITHOUT copying: contiguous
    /// arrays as one run, non-contiguous views as runs over their backing buffer (``Payload``). The
    /// payload points into the arrays' buffers, so the caller keeps `arrays` alive until the write has
    /// finished (`DiskCache.InflightWrite` holds them; MLX never updates a buffer in place while
    /// another array still references it). Nil when a dtype has no safetensors name.
    ///
    /// Why the strided form: `asData(.noCopyIfContiguous)` silently falls back to a single-threaded
    /// CPU copy into a fresh host allocation for a non-contiguous array. A KV snapshot whose state is
    /// a `..<offset` slice of a step-padded buffer (post-answer stores of the live cache, trimmed
    /// boundaries) would pay that copy, the full context size, on the synchronous end-of-turn path.
    static func tensors(from arrays: [String: MLXArray]) -> [Tensor]? {
        var out: [Tensor] = []
        out.reserveCapacity(arrays.count)
        for (name, array) in arrays.sorted(by: { $0.key < $1.key }) {
            guard let dtype = dtypeName(array.dtype) else { return nil }
            guard let payload = payload(of: array) else { return nil }
            let tensor = Tensor(name: name, dtype: dtype, shape: array.shape, payload: payload)
            guard tensor.byteCount == array.nbytes else { return nil }
            out.append(tensor)
        }
        return out
    }

    /// Smallest run worth a separate write; below it (or with more runs than this) a view is copied
    /// like before rather than written in many tiny pieces.
    static let minimumRunBytes = 64 << 10
    static let maximumRuns = 65_536

    static func payload(of array: MLXArray) -> Payload? {
        guard array.size > 0 else { return .contiguous(Data()) }
        let raw = array.asData(access: .noCopy)
        let shape = array.shape
        let strides = raw.strides
        let item = array.itemSize
        // Split point: dims split..<ndim are row-major contiguous (size-1 dims never break it).
        var expected = 1
        var split = shape.count
        for d in stride(from: shape.count - 1, through: 0, by: -1) {
            if shape[d] == 1 {
                split = d
                continue
            }
            guard strides[d] == expected else { break }
            expected *= shape[d]
            split = d
        }
        if split == 0 {
            // `raw.data` itself (zero-copy for real buffers; Foundation keeps a tiny one inline).
            return .contiguous(raw.data)
        }
        // The buffer address straight from MLX: a pointer taken out of `Data.withUnsafeBytes` is only
        // valid inside the closure (inline storage for small Data).
        guard let base = mlx_array_data_uint8(array.ctx).map({ UnsafeRawPointer($0) }) else { return nil }
        let chunkBytes = shape[split...].reduce(1, *) * item
        let runs = shape[..<split].reduce(1, *)
        guard chunkBytes >= minimumRunBytes, runs <= maximumRuns,
            !strides[..<split].contains(where: { $0 < 0 })
        else {
            let copy = array.asData(access: .copy)
            return .contiguous(copy.data)
        }
        var offsets: [Int] = []
        offsets.reserveCapacity(runs)
        var index = [Int](repeating: 0, count: split)
        for _ in 0 ..< runs {
            var offset = 0
            for d in 0 ..< split { offset += index[d] * strides[d] }
            offsets.append(offset * item)
            var d = split - 1
            while d >= 0 {
                index[d] += 1
                if index[d] < shape[d] { break }
                index[d] = 0
                d -= 1
            }
        }
        return .strided(base: base, chunkBytes: chunkBytes, offsets: offsets)
    }

    /// Names (with counts) of floating-point tensors holding a NaN or ±Inf, read from the host
    /// bytes ``payload(of:)`` produced — no MLX call, so any thread may run it while the payload's
    /// arrays are alive. A value is non-finite exactly when its exponent bits are all ones, which
    /// is what this tests per element, so unlike a sum screen it has no overflow false alarms.
    /// Work is split into slices across cores; a clean 3.36 GB payload reads in well under the
    /// time one GPU reduction took on the synchronous store path.
    static func nonFiniteTensorNames(_ tensors: [Tensor], limit: Int = 4) -> [String] {
        struct Slice { let tensor: Int; let base: UnsafeRawPointer; let bytes: Int }
        func width(_ dtype: String) -> Int? {
            switch dtype {
            case "BF16", "F16": return 2
            case "F32": return 4
            case "F64": return 8
            default: return nil
            }
        }
        let sliceBytes = 8 << 20
        var slices: [Slice] = []
        // `Data` storage for contiguous payloads must stay pinned while slices point into it.
        var pinned: [Data] = []
        var smallFlagged: [Int] = []
        for (index, tensor) in tensors.enumerated() {
            guard let item = width(tensor.dtype) else { continue }
            func add(_ base: UnsafeRawPointer, _ bytes: Int) {
                var offset = 0
                while offset < bytes {
                    let n = min(sliceBytes, bytes - offset)
                    slices.append(Slice(tensor: index, base: base + offset, bytes: n - n % item))
                    offset += n
                }
            }
            switch tensor.payload {
            case .contiguous(let data):
                guard !data.isEmpty else { continue }
                if data.count < sliceBytes {
                    // Small `Data` may be stored inline, where a pointer is only valid inside
                    // `withUnsafeBytes`: screen it right here.
                    let hit = data.withUnsafeBytes { raw in
                        raw.baseAddress.map {
                            containsNonFinite($0, bytes: raw.count - raw.count % item, dtype: tensor.dtype)
                        } ?? false
                    }
                    if hit { smallFlagged.append(index) }
                    continue
                }
                // Large payloads are heap buffers (the MLX array's own, no copy); `pinned`
                // keeps each `Data` alive while its slices are read.
                pinned.append(data)
                data.withUnsafeBytes { raw in
                    if let base = raw.baseAddress { add(base, raw.count) }
                }
            case .strided(let base, let chunkBytes, let offsets):
                for offset in offsets { add(base + offset, chunkBytes) }
            }
        }
        let dtypes = tensors.map(\.dtype)
        var flags = [Bool](repeating: false, count: slices.count)
        flags.withUnsafeMutableBufferPointer { out in
            DispatchQueue.concurrentPerform(iterations: slices.count) { i in
                let slice = slices[i]
                out[i] = containsNonFinite(slice.base, bytes: slice.bytes, dtype: dtypes[slice.tensor])
            }
        }
        var flagged = Set(smallFlagged)
        for (slice, flag) in zip(slices, flags) where flag { flagged.insert(slice.tensor) }
        var names: [String] = []
        for index in flagged.sorted() {
            let count: Int
            if case .contiguous(let data) = tensors[index].payload, data.count < sliceBytes {
                count = data.withUnsafeBytes { raw in
                    raw.baseAddress.map { countNonFinite($0, bytes: raw.count, dtype: dtypes[index]) } ?? 0
                }
            } else {
                count = slices.filter { $0.tensor == index }.reduce(0) {
                    $0 + countNonFinite($1.base, bytes: $1.bytes, dtype: dtypes[index])
                }
            }
            names.append("\(tensors[index].name)(\(count))")
            if names.count >= limit { break }
        }
        withExtendedLifetime(pinned) {}
        return names
    }

    private static func containsNonFinite(_ base: UnsafeRawPointer, bytes: Int, dtype: String) -> Bool {
        switch dtype {
        case "BF16", "F16":
            let mask: UInt16 = dtype == "BF16" ? 0x7F80 : 0x7C00
            let vmask = SIMD32<UInt16>(repeating: mask)
            let count = bytes / 2
            var hit = SIMDMask<SIMD32<UInt16>.MaskStorage>(repeating: false)
            var i = 0
            while i + 32 <= count {
                let v = base.loadUnaligned(fromByteOffset: i * 2, as: SIMD32<UInt16>.self)
                hit .|= (v & vmask) .== vmask
                i += 32
            }
            if any(hit) { return true }
            while i < count {
                if base.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self) & mask == mask { return true }
                i += 1
            }
            return false
        case "F32":
            let mask: UInt32 = 0x7F80_0000
            let vmask = SIMD16<UInt32>(repeating: mask)
            let count = bytes / 4
            var hit = SIMDMask<SIMD16<UInt32>.MaskStorage>(repeating: false)
            var i = 0
            while i + 16 <= count {
                let v = base.loadUnaligned(fromByteOffset: i * 4, as: SIMD16<UInt32>.self)
                hit .|= (v & vmask) .== vmask
                i += 16
            }
            if any(hit) { return true }
            while i < count {
                if base.loadUnaligned(fromByteOffset: i * 4, as: UInt32.self) & mask == mask { return true }
                i += 1
            }
            return false
        default:
            return countNonFinite(base, bytes: bytes, dtype: dtype) > 0
        }
    }

    private static func countNonFinite(_ base: UnsafeRawPointer, bytes: Int, dtype: String) -> Int {
        var n = 0
        switch dtype {
        case "BF16", "F16":
            let mask: UInt16 = dtype == "BF16" ? 0x7F80 : 0x7C00
            for i in 0 ..< bytes / 2
            where base.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self) & mask == mask { n += 1 }
        case "F32":
            for i in 0 ..< bytes / 4
            where base.loadUnaligned(fromByteOffset: i * 4, as: UInt32.self) & 0x7F80_0000 == 0x7F80_0000 { n += 1 }
        case "F64":
            let mask: UInt64 = 0x7FF0_0000_0000_0000
            for i in 0 ..< bytes / 8
            where base.loadUnaligned(fromByteOffset: i * 8, as: UInt64.self) & mask == mask { n += 1 }
        default:
            break
        }
        return n
    }

    /// The header bytes (length prefix included) for `tensors` in the given order.
    static func header(for tensors: [Tensor], metadata: [String: String]) throws -> Data {
        var entries: [String: Any] = [:]
        var offset = 0
        for tensor in tensors {
            let end = offset + tensor.byteCount
            entries[tensor.name] = [
                "dtype": tensor.dtype, "shape": tensor.shape, "data_offsets": [offset, end],
            ] as [String: Any]
            offset = end
        }
        if !metadata.isEmpty { entries["__metadata__"] = metadata }
        var json = try JSONSerialization.data(withJSONObject: entries, options: [.sortedKeys])
        let padding = (8 - json.count % 8) % 8
        json.append(contentsOf: [UInt8](repeating: 0x20, count: padding))
        var length = UInt64(json.count).littleEndian
        var header = Data(bytes: &length, count: 8)
        header.append(json)
        return header
    }

    /// `VMLX_CACHE_WRITEV=0`: one `write` per tensor/run (pre-2026-10-11 behavior, A/B control).
    static let gatheredWrites = ProcessInfo.processInfo.environment["VMLX_CACHE_WRITEV"] != "0"

    /// Write `tensors` to `url` (created; must not exist). Returns the file size.
    ///
    /// Default: the header and every tensor run go out as gathered `writev` calls (up to `IOV_MAX`
    /// segments and 1 GiB each, resumed on short writes) instead of one `FileHandle.write` per tensor.
    /// A boundary file carries a few hundred tensors (hybrids: ~400, many of them tiny I32 layer tags);
    /// this is the Python W1 idea (`os.writev` per payload file). Bytes on disk are identical.
    /// Measured 2026-10-11 (RunBench `K2MT_IOBENCH=1`, alternating arms, page-cache-bound writes):
    /// K2-sized 4.1 GB / 145 tensors 397 -> 334 ms median (5 rounds); 0.79 GB / 257 tensors
    /// 103 -> 57 ms (5 rounds). The writer thread finishes each queued store sooner, so the write-behind
    /// queue (and the payload memory it holds) drains sooner.
    @discardableResult
    static func write(_ tensors: [Tensor], metadata: [String: String], to url: URL) throws -> Int {
        if gatheredWrites {
            return try writeGathered(tensors, metadata: metadata, to: url)
        }
        return try writePerTensor(tensors, metadata: metadata, to: url)
    }

    /// One `write` per tensor (per strided run) through `FileHandle`.
    static func writePerTensor(_ tensors: [Tensor], metadata: [String: String], to url: URL) throws -> Int {
        let header = try header(for: tensors, metadata: metadata)
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
            throw WriteError.cannotCreate(url.lastPathComponent)
        }
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.write(contentsOf: header)
        var size = header.count
        for tensor in tensors where tensor.byteCount > 0 {
            switch tensor.payload {
            case .contiguous(let data):
                try handle.write(contentsOf: data)
            case .strided(let base, let chunkBytes, let offsets):
                for offset in offsets {
                    try handle.write(contentsOf: Data(
                        bytesNoCopy: UnsafeMutableRawPointer(mutating: base + offset), count: chunkBytes,
                        deallocator: .none))
                }
            }
            size += tensor.byteCount
        }
        return size
    }

    enum GatheredWriteError: Error { case openFailed(Int32), writeFailed(Int32), noProgress }

    /// Bytes per `writev` call (Darwin rejects a call past a 32-bit total). Tests lower it to exercise
    /// split segments and multi-call files.
    nonisolated(unsafe) static var maxGatheredCallBytes = 1 << 30

    static func writeGathered(_ tensors: [Tensor], metadata: [String: String], to url: URL) throws -> Int {
        let header = try header(for: tensors, metadata: metadata)
        let fd = open(url.path, O_WRONLY | O_CREAT | O_EXCL, 0o644)
        guard fd >= 0 else { throw GatheredWriteError.openFailed(errno) }
        defer { close(fd) }
        // Every segment points into memory that outlives this call: the header `Data` (kept below),
        // contiguous tensor `Data`, or a strided run of an MLX buffer the caller keeps alive.
        var segments: [(UnsafeRawPointer, Int)] = []
        var keep: [Data] = [header]
        var owned: [UnsafeMutableRawPointer] = []
        defer { owned.forEach { $0.deallocate() } }
        var total = 0
        func add(_ data: Data) {
            guard !data.isEmpty else { return }
            // Foundation stores small `Data` inline: a pointer from `withUnsafeBytes` is only valid
            // inside the closure. Copy anything small into memory owned by this call; larger `Data`
            // (heap or no-copy MLX buffer, kept alive by `keep`) is referenced in place.
            if data.count <= 4096 {
                let copy = UnsafeMutableRawPointer.allocate(byteCount: data.count, alignment: 16)
                data.copyBytes(to: copy.assumingMemoryBound(to: UInt8.self), count: data.count)
                owned.append(copy)
                segments.append((UnsafeRawPointer(copy), data.count))
                return
            }
            keep.append(data)
            data.withUnsafeBytes { raw in
                if let base = raw.baseAddress { segments.append((base, data.count)) }
            }
        }
        add(header)
        for tensor in tensors where tensor.byteCount > 0 {
            switch tensor.payload {
            case .contiguous(let data):
                add(data)
            case .strided(let base, let chunkBytes, let offsets):
                for offset in offsets { segments.append((base + offset, chunkBytes)) }
            }
        }
        return try withExtendedLifetime(keep) {
            var index = 0
            var consumed = 0  // bytes of segments[index] already written
            // Darwin `writev` rejects (EINVAL) a call whose lengths sum past a 32-bit int, so each call
            // carries at most `maxCallBytes` (a larger segment is written in pieces).
            let maxCallBytes = maxGatheredCallBytes
            #if canImport(Darwin)
                let maxVectors = Int(IOV_MAX)
            #else
                let configuredMax = sysconf(Int32(_SC_IOV_MAX))
                let maxVectors = configuredMax > 0 ? Int(configuredMax) : 16
            #endif
            while index < segments.count {
                var iov: [iovec] = []
                var callBytes = 0
                var k = 0
                while index + k < segments.count, iov.count < maxVectors, callBytes < maxCallBytes {
                    let (base, length) = segments[index + k]
                    let skip = k == 0 ? consumed : 0
                    let take = min(length - skip, maxCallBytes - callBytes)
                    iov.append(iovec(iov_base: UnsafeMutableRawPointer(mutating: base + skip), iov_len: take))
                    callBytes += take
                    if take < length - skip { break }
                    k += 1
                }
                let written = iov.withUnsafeBufferPointer { writev(fd, $0.baseAddress, Int32($0.count)) }
                if written < 0 {
                    if errno == EINTR { continue }
                    throw GatheredWriteError.writeFailed(errno)
                }
                if written == 0 { throw GatheredWriteError.noProgress }
                total += written
                // Advance past fully written segments; keep the offset into a partial one.
                var left = written + consumed
                consumed = 0
                while index < segments.count, left >= segments[index].1 {
                    left -= segments[index].1
                    index += 1
                }
                consumed = left
            }
            return total
        }
    }
}
