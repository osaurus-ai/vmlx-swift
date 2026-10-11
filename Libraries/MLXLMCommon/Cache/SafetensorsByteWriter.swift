// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

/// Writes a safetensors file from tensor bytes that are already in CPU-visible memory, without calling
/// into MLX.
///
/// `DiskCache` stores through MLX's own `save`, which submits Metal work and therefore has to run on
/// the thread that owns MLX state, under the process-wide cache IO lock. The write-behind tier takes
/// the bytes of evaluated arrays on that thread (`MLXArray.asData(access: .noCopyIfContiguous)`, no
/// copy), and this writer puts them on disk from a background queue: plain file I/O only.
///
/// Format (safetensors): little-endian u64 header length, a JSON header naming each tensor's dtype,
/// shape and `data_offsets` into the byte buffer that follows (plus `__metadata__`), padded with
/// spaces to an 8-byte boundary, then the tensor bytes in offset order.
enum SafetensorsByteWriter {
    struct Tensor {
        let name: String
        let dtype: String
        let shape: [Int]
        let data: Data
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

    /// Bytes of each evaluated array, taken on the MLX-owning thread. Zero-copy for contiguous
    /// arrays: the returned `Data` points into the arrays' buffers, so the caller keeps `arrays`
    /// alive until the write has finished. Nil when a dtype has no safetensors name.
    static func tensors(from arrays: [String: MLXArray]) -> [Tensor]? {
        var out: [Tensor] = []
        out.reserveCapacity(arrays.count)
        for (name, array) in arrays.sorted(by: { $0.key < $1.key }) {
            guard let dtype = dtypeName(array.dtype) else { return nil }
            let bytes = array.asData(access: .noCopyIfContiguous)
            guard bytes.data.count == array.nbytes else { return nil }
            out.append(Tensor(name: name, dtype: dtype, shape: array.shape, data: bytes.data))
        }
        return out
    }

    /// The header bytes (length prefix included) for `tensors` in the given order.
    static func header(for tensors: [Tensor], metadata: [String: String]) throws -> Data {
        var entries: [String: Any] = [:]
        var offset = 0
        for tensor in tensors {
            let end = offset + tensor.data.count
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
        for tensor in tensors where !tensor.data.isEmpty {
            try handle.write(contentsOf: tensor.data)
            size += tensor.data.count
        }
        return size
    }
}
