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

    /// Write `tensors` to `url` (created; must not exist). Returns the file size.
    @discardableResult
    static func write(_ tensors: [Tensor], metadata: [String: String], to url: URL) throws -> Int {
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
}
