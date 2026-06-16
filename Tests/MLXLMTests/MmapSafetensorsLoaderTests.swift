// Copyright © 2026 Jinho Jang. All rights reserved.
//
// Tests for `MmapSafetensorsLoader` — the experimental Swift-only mmap-backed
// safetensors loader that only supports CPU-side reads (`.asArray(...)`).
// GPU operations on these Swift direct-pointer arrays return zeros. The
// production C++ whole-shard mmap path is covered separately by
// `SaveTests/testMmapSafetensorsLoadCanFeedGPUComputation`.
//
// Coverage scoped to what the loader can correctly do:
//   1. Header parsing + shape/dtype matches stock loader.
//   2. CPU-side element-wise data matches stock loader (via .asArray).
//   3. Metadata (__metadata__ block) is parsed.
//   4. Mapping lifetime (handle survives loader scope).
//   5. Failure modes (corrupt header, missing file).
//
// Explicitly NOT tested here: GPU op parity (allClose, sum, matmul, ...).
// Those fail for this Swift prototype and document why it is diagnostics-only.

import Foundation
import MLX
import Testing
@testable import MLXLMCommon

@Suite("MmapSafetensorsLoader")
struct MmapSafetensorsLoaderTests {

    // MARK: - Helpers

    /// Make a unique temp safetensors file with the given arrays + metadata.
    private static func makeTempSafetensors(
        arrays: [String: MLXArray],
        metadata: [String: String] = [:]
    ) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("MmapLoaderTest-\(UUID().uuidString).safetensors")
        try MLX.save(arrays: arrays, metadata: metadata, url: url)
        return url
    }

    // MARK: - Round-trip

    @Test("CPU-side parity: mmap loader matches stock loader on shape + dtype")
    func cpuSideShapeDtypeParity() throws {
        let arrays: [String: MLXArray] = [
            "alpha": MLXArray(0..<60).reshaped([3, 4, 5]).asType(.float32),
            "beta":  MLXArray(0..<24).reshaped([2, 12]).asType(.float16),
            "gamma": MLXArray([1, 2, 3, 4, 5, 6, 7, 8] as [Int32]),
        ]
        let url = try Self.makeTempSafetensors(
            arrays: arrays,
            metadata: ["model": "test", "format": "pt"])
        defer { try? FileManager.default.removeItem(at: url) }

        let (stock, stockMeta) = try loadArraysAndMetadata(url: url)
        let (mmap, mmapMeta) = try MmapSafetensorsLoader
            .loadArraysAndMetadata(url: url)

        #expect(stock.keys.sorted() == mmap.keys.sorted())
        #expect(stockMeta == mmapMeta)

        for key in stock.keys {
            #expect(stock[key]!.shape == mmap[key]!.shape, "shape diff on \(key)")
            #expect(stock[key]!.dtype == mmap[key]!.dtype, "dtype diff on \(key)")
        }
    }

    /// Same-dtype CPU read works for float32 — `.asArray(Float.self)`
    /// pulls bytes through the host pointer without going through any
    /// GPU op. Other dtypes (int32, int8, ...) and any cast via
    /// `asType(...)` route through GPU and return garbage today.
    @Test("CPU-side float32 values match stock loader exactly")
    func cpuSideFloat32Values() throws {
        let original = MLXArray(0..<60).reshaped([3, 4, 5]).asType(.float32)
        let url = try Self.makeTempSafetensors(arrays: ["t": original])
        defer { try? FileManager.default.removeItem(at: url) }

        let (stock, _) = try loadArraysAndMetadata(url: url)
        let (mmap, _) = try MmapSafetensorsLoader.loadArraysAndMetadata(url: url)

        #expect(stock["t"]!.asArray(Float.self) == mmap["t"]!.asArray(Float.self))
    }

    // MARK: - Mapping lifetime

    @Test("mmap region survives loader returning (CPU read after scope exit)")
    func mappingSurvivesLoaderScope() throws {
        let original = MLXArray(0..<100).reshaped([10, 10]).asType(.float32)
        let url = try Self.makeTempSafetensors(arrays: ["x": original])
        defer { try? FileManager.default.removeItem(at: url) }

        // Force loader scope to end before we touch the array.
        let loadedArray = try {
            let (loaded, _) = try MmapSafetensorsLoader
                .loadArraysAndMetadata(url: url)
            return loaded["x"]!
        }()

        // The mmap handle must still be alive (captured in the array's
        // finalizer). CPU read confirms the mapping hasn't been
        // munmap'd. (GPU op `.sum()` would return 0 today — see header.)
        let bytes = loadedArray.asArray(Float.self)
        let sum = bytes.reduce(0, +)
        #expect(sum == 4950.0)  // 0+1+...+99 = 4950
    }

    // MARK: - Metadata

    @Test("__metadata__ block is parsed into metadata dict")
    func metadataParsed() throws {
        let url = try Self.makeTempSafetensors(
            arrays: ["x": MLXArray([1.0, 2.0, 3.0] as [Float])],
            metadata: [
                "format": "safetensors",
                "framework": "mlx",
                "custom_key": "custom_value",
            ])
        defer { try? FileManager.default.removeItem(at: url) }

        let (_, metadata) = try MmapSafetensorsLoader
            .loadArraysAndMetadata(url: url)
        #expect(metadata["format"] == "safetensors")
        #expect(metadata["framework"] == "mlx")
        #expect(metadata["custom_key"] == "custom_value")
    }

    // MARK: - Failure modes

    @Test("missing file throws openFailed")
    func missingFileThrows() {
        let url = URL(fileURLWithPath: "/nonexistent/\(UUID().uuidString).safetensors")
        #expect(throws: MmapSafetensorsError.self) {
            try MmapSafetensorsLoader.loadArraysAndMetadata(url: url)
        }
    }

    @Test("non-file URL throws instead of trapping")
    func nonFileURLThrows() {
        let url = URL(string: "https://example.invalid/model.safetensors")!
        #expect(throws: MmapSafetensorsError.self) {
            try MmapSafetensorsLoader.loadArraysAndMetadata(url: url)
        }
    }

    @Test("file too short throws headerTooShort")
    func tooShortThrows() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("MmapLoaderTest-short-\(UUID().uuidString).safetensors")
        try Data([0x00, 0x01]).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(throws: MmapSafetensorsError.self) {
            try MmapSafetensorsLoader.loadArraysAndMetadata(url: url)
        }
    }

    @Test("garbage header length throws headerLengthOutOfBounds")
    func headerLenOutOfBoundsThrows() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("MmapLoaderTest-bad-\(UUID().uuidString).safetensors")
        // 8 bytes claiming 1 TB header; file is only 100 bytes.
        var bytes = Data(count: 100)
        let big: UInt64 = 1024 * 1024 * 1024 * 1024  // 1 TB
        bytes.withUnsafeMutableBytes { (ptr: UnsafeMutableRawBufferPointer) in
            ptr.baseAddress!.storeBytes(of: big.littleEndian, as: UInt64.self)
        }
        try bytes.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(throws: MmapSafetensorsError.self) {
            try MmapSafetensorsLoader.loadArraysAndMetadata(url: url)
        }
    }
}
