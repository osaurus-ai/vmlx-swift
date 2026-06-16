// Copyright © 2026 Jinho Jang. All rights reserved.
//
// MmapSafetensorsLoader — header-aware mmap reader for safetensors
// shards. This Swift-only loader is retained for diagnostics and
// header/CPU-side tests. Production mmap-backed weight loading now
// belongs in the osaurus MLX C++ whole-shard loader enabled by
// `MLX_SAFETENSORS_MMAP=1`.
//
// This Swift direct-pointer prototype is CPU-only: the resulting `MLXArray`s read correctly via
// `.asArray(...)` but GPU operations (matmul, allClose, ...) see zeros
// instead of the file contents.
//
// **EXPERIMENTAL — NOT WIRED INTO THE PRODUCTION LOAD PATH.**
//
// Why this hits a wall
// ====================
// MLX's stock safetensors loader (`mlx/backend/common/load.cpp:30`) does:
//
//     out.set_data(allocator::malloc(out.nbytes()));
//     reader->read(out_ptr, size * itemsize, offset);
//
// → tensors live in MLX's MetalAllocator pool (Metal-backed shared
// storage that both CPU and GPU can read). The page-eviction story
// JangPress's docs claim doesn't work in this state: heap-allocated
// shared-storage MTLBuffers aren't file-backed, so the kernel cannot
// trivially evict them under pressure.
//
// The natural fix is to point the array's backing storage at the file
// mapping directly. The public Swift API for this is
// `MLXArray(rawPointer:shape:dtype:finalizer:)`, which underneath
// calls `MetalAllocator::make_buffer(ptr, size)` →
// `MTLDevice.newBuffer(bytesNoCopy:length:options:)`. That call:
//
//   - silently copies into a fresh allocator-malloc'd buffer when the
//     pointer isn't page-aligned (mmap base + safetensors offset
//     usually isn't), and
//   - even when it succeeds (page-aligned region), produces an
//     MTLBuffer whose host pointer alias works for CPU but whose
//     GPU side reads as zero on Apple Silicon. Diagnostic in
//     docs/WIRED-LIMIT-INVESTIGATION-2026-05-03.md "GPU sees zeros"
//     section.
//
// The real fix lives below Swift. The current osaurus MLX fork maps the
// whole shard at a page-aligned base and creates tensor views into one
// shared Metal buffer. `SaveTests/testMmapSafetensorsLoadCanFeedGPUComputation`
// proves that aligned safetensors loaded through `MLX_SAFETENSORS_MMAP=1`
// can feed a Metal reduction with nonzero, correct values. This Swift
// loader remains useful only for CPU-side header inspection (e.g. byte
// counting, dtype/shape probing).
//
// Use cases that DO work today:
//   * Reading the JSON header to compute total tensor bytes / detect
//     routed-MoE structure (`LoadBundleFacts`).
//   * Touching tensor bytes from CPU (`array.asArray(Float.self)`).
//   * Anything that doesn't trigger Metal compute on the array.
//
// See docs/WIRED-LIMIT-INVESTIGATION-2026-05-03.md for the full
// architecture analysis + the upstream-MLX work plan.
//
// Safetensors format (per spec):
//   bytes [0..8)         : little-endian uint64 N = JSON header length
//   bytes [8..8+N)       : UTF-8 JSON header
//                          {
//                            "__metadata__": { "key": "value", ... },
//                            "tensor_name": {
//                              "dtype": "F32" | "F16" | "BF16" | ... ,
//                              "shape": [d1, d2, ...],
//                              "data_offsets": [start, end]   // bytes within data segment
//                            },
//                            ...
//                          }
//   bytes [8+N..end)     : tensor data segment (offsets are relative)

import Foundation
import MLX

#if canImport(Darwin)
import Darwin

public enum MmapSafetensorsError: LocalizedError {
    case nonFileURL(URL)
    case openFailed(String, errno: Int32)
    case statFailed(String, errno: Int32)
    case mmapFailed(String, errno: Int32)
    case headerTooShort(String, size: Int)
    case headerLengthOutOfBounds(String, headerLen: UInt64, fileSize: Int)
    case headerJSONInvalid(String, underlying: Error)
    case headerJSONShape(String)
    case unsupportedDtype(String, dtype: String)
    case tensorOffsetsOutOfBounds(String, tensor: String, range: ClosedRange<UInt64>, dataLen: UInt64)

    public var errorDescription: String? {
        switch self {
        case .nonFileURL(let url):
            return "MmapSafetensorsLoader: expected a file URL, got \(url.absoluteString)"
        case .openFailed(let path, let e):
            return "MmapSafetensorsLoader: open(\(path)) failed: errno=\(e) (\(String(cString: strerror(e))))"
        case .statFailed(let path, let e):
            return "MmapSafetensorsLoader: stat(\(path)) failed: errno=\(e)"
        case .mmapFailed(let path, let e):
            return "MmapSafetensorsLoader: mmap(\(path)) failed: errno=\(e) (\(String(cString: strerror(e))))"
        case .headerTooShort(let path, let size):
            return "MmapSafetensorsLoader: \(path) is too short (\(size) bytes) to be a safetensors file"
        case .headerLengthOutOfBounds(let path, let h, let s):
            return "MmapSafetensorsLoader: \(path) declares header length \(h) but file size is \(s)"
        case .headerJSONInvalid(let path, let e):
            return "MmapSafetensorsLoader: \(path) JSON header parse failed: \(e)"
        case .headerJSONShape(let path):
            return "MmapSafetensorsLoader: \(path) header is not a top-level JSON object"
        case .unsupportedDtype(let path, let dt):
            return "MmapSafetensorsLoader: \(path) tensor uses unsupported dtype \(dt)"
        case .tensorOffsetsOutOfBounds(let path, let name, let r, let d):
            return "MmapSafetensorsLoader: \(path) tensor \(name) data_offsets \(r) exceed data segment length \(d)"
        }
    }
}

/// Reference-counted owner of one shard's mmap region. The finalizer
/// closures captured by every MLXArray built from this shard hold a
/// strong reference; when the last array drops, this object's deinit
/// fires and calls `munmap`. The fd is closed eagerly after mmap
/// because the mapping itself keeps the underlying file alive.
private final class MmapHandle: @unchecked Sendable {
    let basePointer: UnsafeMutableRawPointer
    let length: Int
    let path: String

    init(basePointer: UnsafeMutableRawPointer, length: Int, path: String) {
        self.basePointer = basePointer
        self.length = length
        self.path = path
    }

    deinit {
        munmap(basePointer, length)
    }
}

public enum MmapSafetensorsLoader {

    /// Load every tensor in `url` as mmap-backed `MLXArray`s. Returns
    /// `(arrays, metadata)` matching the shape of MLX's stock
    /// `loadArraysAndMetadata(url:)`.
    ///
    /// Each returned MLXArray's data pointer aliases the file mapping.
    /// When all returned arrays are dropped, the mapping releases.
    /// Holding even one array keeps the entire shard mapped (acceptable
    /// — model weights stay loaded for the model's lifetime anyway).
    public static func loadArraysAndMetadata(url: URL) throws -> (
        [String: MLXArray], [String: String]
    ) {
        guard url.isFileURL else {
            throw MmapSafetensorsError.nonFileURL(url)
        }
        let path = url.path(percentEncoded: false)

        // 1. open + stat
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            throw MmapSafetensorsError.openFailed(path, errno: errno)
        }
        var st = stat()
        guard fstat(fd, &st) == 0 else {
            let e = errno
            close(fd)
            throw MmapSafetensorsError.statFailed(path, errno: e)
        }
        let fileSize = Int(st.st_size)

        // 2. mmap PROT_READ MAP_PRIVATE — the kernel may collapse
        //    multiple mappings of the same file into shared pages.
        guard let raw = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fd, 0),
            raw != MAP_FAILED
        else {
            let e = errno
            close(fd)
            throw MmapSafetensorsError.mmapFailed(path, errno: e)
        }
        // The mapping itself holds a reference to the underlying file;
        // we can close the fd immediately. munmap (via MmapHandle.deinit)
        // releases the mapping.
        close(fd)

        let basePointer = UnsafeMutableRawPointer(mutating: raw)
        let handle = MmapHandle(
            basePointer: basePointer, length: fileSize, path: path)

        // 3. Parse safetensors header.
        guard fileSize >= 8 else {
            throw MmapSafetensorsError.headerTooShort(path, size: fileSize)
        }
        let headerLen = basePointer.load(as: UInt64.self).littleEndian
        guard 8 + headerLen <= UInt64(fileSize) else {
            throw MmapSafetensorsError.headerLengthOutOfBounds(
                path, headerLen: headerLen, fileSize: fileSize)
        }
        let headerData = Data(
            bytes: basePointer.advanced(by: 8),
            count: Int(headerLen))
        let parsedHeader: [String: Any]
        do {
            parsedHeader = try JSONSerialization.jsonObject(
                with: headerData) as? [String: Any] ?? [:]
        } catch {
            throw MmapSafetensorsError.headerJSONInvalid(
                path, underlying: error)
        }
        if parsedHeader.isEmpty {
            throw MmapSafetensorsError.headerJSONShape(path)
        }

        // 4. Build mmap-backed MLXArrays + metadata dictionary.
        let dataSegmentBase = basePointer.advanced(by: 8 + Int(headerLen))
        let dataSegmentLen = UInt64(fileSize) - 8 - headerLen

        var arrays: [String: MLXArray] = [:]
        var metadata: [String: String] = [:]

        for (key, raw) in parsedHeader {
            // Metadata block.
            if key == "__metadata__" {
                if let m = raw as? [String: Any] {
                    for (mk, mv) in m {
                        if let s = mv as? String { metadata[mk] = s }
                    }
                }
                continue
            }

            guard let entry = raw as? [String: Any],
                let dtypeStr = entry["dtype"] as? String,
                let shape = entry["shape"] as? [Int],
                let offsets = entry["data_offsets"] as? [NSNumber],
                offsets.count == 2
            else {
                // Malformed individual entry — skip but keep loading.
                FileHandle.standardError.write(Data(
                    "[MmapSafetensorsLoader] skipping malformed entry \(key) in \(path)\n".utf8))
                continue
            }

            let begin = offsets[0].uint64Value
            let end = offsets[1].uint64Value
            guard begin <= end, end <= dataSegmentLen else {
                throw MmapSafetensorsError.tensorOffsetsOutOfBounds(
                    path, tensor: key, range: begin...end, dataLen: dataSegmentLen)
            }

            guard let dtype = mlxDType(safetensorsDtype: dtypeStr) else {
                throw MmapSafetensorsError.unsupportedDtype(path, dtype: dtypeStr)
            }

            // Build a no-copy MLXArray pointing into the mapping. The
            // finalizer keeps the MmapHandle alive (and therefore the
            // mapping) until this array's last reference drops. Other
            // arrays from the same shard each capture their own strong
            // reference to the same handle — last-one-out unmaps.
            let dataPtr = dataSegmentBase.advanced(by: Int(begin))
            // 0-element tensors (shape contains 0) are legal in
            // safetensors; MLXArray's no-copy ctor accepts them.
            let array = MLXArray(
                rawPointer: dataPtr, shape, dtype: dtype
            ) {
                // Strong capture of the handle; runs on last drop.
                _ = handle
            }
            arrays[key] = array
        }

        return (arrays, metadata)
    }

    /// Map a safetensors dtype string to MLX's DType. Returns nil for
    /// unsupported dtypes — the caller surfaces a clear error.
    private static func mlxDType(safetensorsDtype: String) -> DType? {
        switch safetensorsDtype {
        case "F32": return .float32
        case "F16": return .float16
        case "BF16": return .bfloat16
        case "F64": return .float32   // promote — MLX has no float64
        case "I8":  return .int8
        case "I16": return .int16
        case "I32": return .int32
        case "I64": return .int64
        case "U8":  return .uint8
        case "U16": return .uint16
        case "U32": return .uint32
        case "U64": return .uint64
        case "BOOL": return .bool
        default: return nil
        }
    }
}

#else  // !canImport(Darwin)

public enum MmapSafetensorsError: Error {
    case nonFileURL(URL)
    case unsupportedPlatform
}

public enum MmapSafetensorsLoader {
    public static func loadArraysAndMetadata(url: URL) throws -> (
        [String: MLXArray], [String: String]
    ) {
        guard url.isFileURL else {
            throw MmapSafetensorsError.nonFileURL(url)
        }
        throw MmapSafetensorsError.unsupportedPlatform
    }
}

#endif
