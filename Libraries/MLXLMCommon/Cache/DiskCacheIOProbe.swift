// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

/// No-model measurement of the SSD cache file I/O mechanisms (RunBench `K2MT_IOBENCH=1`):
/// gathered vs per-tensor writes of a boundary payload, and a cold restore with and without
/// `DiskCache.adviseReadAhead`. Prints `IOPROBE ...` lines; never used by serving code.
public enum DiskCacheIOProbe {
    /// A boundary payload shaped like the app's: per layer, bf16 keys and values `[1, heads, tokens, dim]`
    /// plus the small I32 layer tags the serializer adds.
    public static func payload(layers: Int, heads: Int, tokens: Int, dim: Int) -> [String: MLXArray] {
        var arrays: [String: MLXArray] = [:]
        for layer in 0 ..< layers {
            arrays["kv_\(layer)_keys"] = MLXRandom.normal([1, heads, tokens, dim]).asType(.bfloat16)
            arrays["kv_\(layer)_values"] = MLXRandom.normal([1, heads, tokens, dim]).asType(.bfloat16)
            arrays["__layer_kind_\(layer)__"] = MLXArray([Int32(1)])
            arrays["__layer_offset_\(layer)__"] = MLXArray([Int32(tokens)])
        }
        arrays["__jang_cache_format_version__"] = MLXArray([Int32(2)])
        eval(Array(arrays.values))
        return arrays
    }

    /// Fraction of `url`'s pages resident in the page cache (`mincore` over a read-only mapping).
    public static func residentFraction(_ url: URL) -> Double {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { return -1 }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_size > 0 else { return -1 }
        let size = Int(info.st_size)
        guard let map = mmap(nil, size, PROT_READ, MAP_SHARED, fd, 0), map != MAP_FAILED else { return -1 }
        defer { munmap(map, size) }
        let page = Int(getpagesize())
        let pages = (size + page - 1) / page
        var vec = [CChar](repeating: 0, count: pages)
        guard mincore(map, size, &vec) == 0 else { return -1 }
        return Double(vec.reduce(0) { $0 + (($1 & 1) != 0 ? 1 : 0) }) / Double(pages)
    }

    /// Drop `url` from the page cache without root: rewrite nothing, just ask the kernel to invalidate
    /// the cached pages of a mapping (`msync(MS_INVALIDATE)`) after a full sync. Returns the resident
    /// fraction afterwards (callers report it; a run that is not cold is not a cold measurement).
    public static func makeCold(_ url: URL) -> Double {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { return -1 }
        #if canImport(Darwin)
            _ = fcntl(fd, F_FULLFSYNC)
        #else
            _ = fsync(fd)
        #endif
        var info = stat()
        if fstat(fd, &info) == 0, info.st_size > 0,
            let map = mmap(nil, Int(info.st_size), PROT_READ, MAP_SHARED, fd, 0), map != MAP_FAILED
        {
            _ = msync(map, Int(info.st_size), MS_INVALIDATE)
            munmap(map, Int(info.st_size))
        }
        close(fd)
        return residentFraction(url)
    }

    public static func run(directory: URL, layers: Int, heads: Int, tokens: Int, dim: Int, rounds: Int) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let arrays = payload(layers: layers, heads: heads, tokens: tokens, dim: dim)
        guard let tensors = SafetensorsByteWriter.tensors(from: arrays) else {
            print("IOPROBE INVALID: payload not writable")
            return
        }
        let bytes = tensors.reduce(0) { $0 + $1.byteCount }
        print("IOPROBE payload tensors=\(tensors.count) bytes=\(bytes)")
        func fresh(_ name: String) -> URL {
            let url = directory.appendingPathComponent(name)
            try? FileManager.default.removeItem(at: url)
            return url
        }
        // Writes: alternate the order every round; same tensors, same bytes.
        var gathered: [Double] = []
        var perTensor: [Double] = []
        for round in 1 ... rounds {
            for mode in round % 2 == 1 ? ["gathered", "per-tensor"] : ["per-tensor", "gathered"] {
                let url = fresh("w-\(mode)-\(round).safetensors")
                let t0 = Date()
                if mode == "gathered" {
                    try SafetensorsByteWriter.writeGathered(tensors, metadata: ["format": "mlx"], to: url)
                } else {
                    try SafetensorsByteWriter.writePerTensor(tensors, metadata: ["format": "mlx"], to: url)
                }
                let ms = Date().timeIntervalSince(t0) * 1000
                (mode == "gathered" ? { gathered.append(ms) } : { perTensor.append(ms) })()
                print(String(format: "IOPROBE write round=%d mode=%@ ms=%.1f", round, mode, ms))
            }
        }
        let a = try Data(contentsOf: directory.appendingPathComponent("w-gathered-1.safetensors"), options: .alwaysMapped)
        let b = try Data(contentsOf: directory.appendingPathComponent("w-per-tensor-1.safetensors"), options: .alwaysMapped)
        print("IOPROBE write identical=\(a == b) gathered_median_ms=\(median(gathered)) per_tensor_median_ms=\(median(perTensor))")
        // Cold restore: a fresh copy per arm, made cold and checked; load + evaluate every tensor.
        var advised: [Double] = []
        var plain: [Double] = []
        for round in 1 ... rounds {
            for mode in round % 2 == 1 ? ["readahead", "plain"] : ["plain", "readahead"] {
                let url = fresh("r-\(mode)-\(round).safetensors")
                try SafetensorsByteWriter.writeGathered(tensors, metadata: ["format": "mlx"], to: url)
                let resident = makeCold(url)
                let t0 = Date()
                if mode == "readahead" { DiskCache.adviseReadAhead(url: url, force: true) }
                let (loaded, _) = try loadArraysAndMetadata(url: url)
                for (_, array) in loaded.sorted(by: { $0.key < $1.key }) { eval(array) }
                let ms = Date().timeIntervalSince(t0) * 1000
                (mode == "readahead" ? { advised.append(ms) } : { plain.append(ms) })()
                print(String(format: "IOPROBE restore round=%d mode=%@ resident_before=%.3f ms=%.1f", round, mode, resident, ms))
                try? FileManager.default.removeItem(at: url)
            }
        }
        print("IOPROBE restore readahead_median_ms=\(median(advised)) plain_median_ms=\(median(plain))")
        for round in 1 ... rounds {
            for mode in ["gathered", "per-tensor"] {
                try? FileManager.default.removeItem(at: directory.appendingPathComponent("w-\(mode)-\(round).safetensors"))
            }
        }
    }

    static func median(_ values: [Double]) -> Double {
        let s = values.sorted()
        return s.isEmpty ? 0 : (s[s.count / 2] * 10).rounded() / 10
    }
}
