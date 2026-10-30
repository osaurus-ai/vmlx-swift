//
//  SaveTests.swift
//
//
//  Created by Rounak Jain on 4/2/24.
//

import MLX
import XCTest
import Cmlx
import Darwin

final class SaveTests: XCTestCase {

    let temporaryPath = FileManager.default.temporaryDirectory.appending(
        path: UUID().uuidString,
        directoryHint: .isDirectory
    )

    override func setUpWithError() throws {
        prepareMLXMetallibForTests()
        setDefaultDevice()
        try FileManager.default.createDirectory(
            at: temporaryPath,
            withIntermediateDirectories: false
        )
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: temporaryPath)
    }

    public func testSaveArrays() throws {
        let safetensorsPath = temporaryPath.appending(
            path: "arrays.safetensors",
            directoryHint: .notDirectory
        )

        let arrays: [String: MLXArray] = [
            "foo": MLX.ones([1, 2]),
            "bar": MLX.zeros([2, 1]),
        ]

        try MLX.save(arrays: arrays, url: safetensorsPath)

        let loadedArrays = try MLX.loadArrays(url: safetensorsPath)
        XCTAssertEqual(loadedArrays.keys.sorted(), arrays.keys.sorted())

        assertEqual(try XCTUnwrap(loadedArrays["foo"]), try XCTUnwrap(arrays["foo"]))
        assertEqual(try XCTUnwrap(loadedArrays["bar"]), try XCTUnwrap(arrays["bar"]))
    }

    /// Excluded keys never load. Environment-selected mmap loading is retired in
    /// the MLX dependency, so with the legacy `MLX_SAFETENSORS_MMAP` switch set
    /// the kept tensor is still read into an owned allocation: no layer region
    /// and no mapped Metal buffer.
    public func testLoadSafetensorsExcludingKeysBeforeMaterialization() throws {
        let safetensorsPath = temporaryPath.appending(
            path: "filtered-arrays.safetensors",
            directoryHint: .notDirectory
        )
        // Keys must match the loader's layer pattern, or the zero checks go blind.
        let keepKey = "layers.44.input_layernorm.weight"
        let dropKey = "layers.44.mlp.switch_mlp.gate_proj.weight"
        // A dtype-aligned layout (`MLX.save` does not pad its header), so the
        // retired mapped loader would have mapped the kept tensor.
        try writeSparseFloat32Safetensors(
            at: safetensorsPath,
            firstKey: keepKey,
            firstValues: [1, 2, 3, 4],
            secondKey: dropKey,
            secondValues: [5, 6, 7, 8],
            gapBytes: 0,
            metadataJSON: #"{"owner":"selected-key-test"}"#)

        try withEnvironment("MLX_SAFETENSORS_MMAP", value: "1") {
            let trackedBefore = mlx_safetensors_mmap_tracked_buffer_bytes()
            let (loaded, metadata) = try MLX.loadArraysAndMetadata(
                url: safetensorsPath,
                excludingKeys: [dropKey],
                stream: .cpu)
            XCTAssertEqual(
                mlx_safetensors_mmap_advise_layer(1, 44),
                0,
                "a filtered load must not register layer regions")
            XCTAssertEqual(
                mlx_safetensors_mmap_tracked_buffer_bytes() - trackedBefore,
                0,
                "a filtered load must not keep a mapped Metal buffer")
            XCTAssertEqual(loaded.keys.sorted(), [keepKey])
            XCTAssertNil(loaded[dropKey])
            XCTAssertEqual(metadata["owner"], "selected-key-test")
            assertEqual(
                try XCTUnwrap(loaded[keepKey]),
                MLXArray([Float(1), 2, 3, 4]))
        }
    }

    /// Environment-selected mmap loading is retired in the MLX dependency:
    /// `MLX_SAFETENSORS_MMAP` no longer routes a load to the mapped loader, and
    /// `exactTensorBuffers` no longer selects per-tensor mappings. A filtered
    /// load reads the kept tensor into an owned allocation, still never loads
    /// the excluded one, and leaves no mapped Metal buffer behind.
    public func testFilteredExactTensorBuffersLoadMapsNothing() throws {
        try withMLXMetallibForTests {
            let safetensorsPath = temporaryPath.appending(
                path: "filtered-exact-tensor-buffers.safetensors",
                directoryHint: .notDirectory
            )
            // Keys must match the loader's layer pattern, or the zero checks go blind.
            let keepKey = "layers.0.input_layernorm.weight"
            let dropKey = "layers.0.mlp.switch_mlp.gate_proj.weight"
            try writeSparseFloat32Safetensors(
                at: safetensorsPath,
                firstKey: keepKey,
                firstValues: [1, 2, 3, 4],
                secondKey: dropKey,
                secondValues: [5, 6, 7, 8],
                gapBytes: 1 << 20)

            // The legacy switches stay set: they must not bring a mapping back.
            try withEnvironment("MLX_SAFETENSORS_MMAP", value: "1") {
                try withEnvironment("MLX_SAFETENSORS_MMAP_TENSOR_BUFFERS", value: "0") {
                    let trackedBefore = mlx_safetensors_mmap_tracked_buffer_bytes()
                    let (loaded, _) = try MLX.loadArraysAndMetadata(
                        url: safetensorsPath,
                        excludingKeys: [dropKey],
                        exactTensorBuffers: true,
                        stream: .cpu)
                    let keep = try XCTUnwrap(loaded[keepKey])
                    XCTAssertNil(loaded[dropKey])

                    XCTAssertEqual(
                        mlx_safetensors_mmap_tracked_buffer_bytes() - trackedBefore,
                        0,
                        "a filtered load must not keep any part of the shard in a mapped Metal buffer")

                    let result = MLX.sum(keep, stream: .gpu)
                    XCTAssertEqual(result.item(Float.self), 10.0, accuracy: 0.001)
                }
            }
        }
    }

    /// The mapped bank that registers layer regions is now a named mmap file
    /// region; environment-selected safetensors mmap loading, which used to
    /// register them, is retired in the MLX dependency. A named bank registers
    /// its bytes under its layer, takes cold advice, and refaults into GPU
    /// computation.
    public func testNamedMmapBankRegistersLayerAndFeedsGPUAfterColdAdvice() throws {
        try withMLXMetallibForTests {
            let path = temporaryPath.appending(
                path: "named-mmap-bank.bin",
                directoryHint: .notDirectory
            )
            let offset = 4096
            try writeUInt32Bank(at: path, offset: offset, values: Array(1 ... 8))

            let trackedBefore = mlx_safetensors_mmap_tracked_buffer_bytes()
            let bank = try mapNamedUInt32Bank(
                at: path,
                offset: offset,
                shape: [2, 2, 2],
                name: "model.layers.42.mlp.switch_mlp.gate_proj.tq2_packed")
            withExtendedLifetime(bank) {
                XCTAssertEqual(
                    mlx_safetensors_mmap_advise_layer(1, 42),
                    32,
                    "a named bank must register its 32 bytes as a layer 42 region")
                XCTAssertGreaterThan(
                    mlx_safetensors_mmap_tracked_buffer_bytes() - trackedBefore,
                    0,
                    "a named bank must keep its mapped Metal buffer tracked")
                XCTAssertEqual(
                    mlx_safetensors_mmap_advise_layer(0, 42),
                    32,
                    "cold advice must reach the registered region")
                XCTAssertEqual(
                    mlx_safetensors_mmap_advise_routed(0, 50),
                    16,
                    "half-cold routed advice must reach one of the bank's two experts")

                let result = MLX.sum(
                    (bank.asType(.float32) * MLXArray(3.0)) + MLXArray(1.0), stream: .gpu)
                XCTAssertEqual(result.item(Float.self), 116.0, accuracy: 0.001)
            }
        }
    }

    /// The retired environment-selected mmap loader realigned a dtype-unaligned
    /// tensor and kept its aligned neighbour mapped. The owned loader reads both
    /// from wherever they sit in the shard, and maps neither. A named mmap bank
    /// still cannot start at a dtype-unaligned offset.
    public func testSafetensorsLoadReadsDtypeUnalignedTensorBesideAlignedOne() throws {
        try withMLXMetallibForTests {
            let safetensorsPath = temporaryPath.appending(
                path: "unaligned-u32.safetensors",
                directoryHint: .notDirectory
            )
            // Keys must match the loader's layer pattern, or the zero checks go blind.
            let unalignedKey = "layers.97.mlp.gate_proj.weight"
            let alignedKey = "layers.98.mlp.gate_proj.weight"
            try writeMixedAlignmentUInt32Safetensors(
                at: safetensorsPath,
                unalignedKey: unalignedKey,
                unalignedValues: [1, 2, 3, 4],
                alignedKey: alignedKey,
                alignedValues: [5, 6, 7, 8],
                shape: [2, 2]
            )

            let trackedBefore = mlx_safetensors_mmap_tracked_buffer_bytes()
            let loadedArrays = try MLX.loadArrays(url: safetensorsPath, stream: .cpu)
            let unaligned = try XCTUnwrap(loadedArrays[unalignedKey])
            let aligned = try XCTUnwrap(loadedArrays[alignedKey])
            XCTAssertEqual(
                mlx_safetensors_mmap_advise_layer(1, 98),
                0,
                "the owned loader must not register the aligned tensor's layer")
            XCTAssertEqual(
                mlx_safetensors_mmap_tracked_buffer_bytes() - trackedBefore,
                0,
                "the owned loader must not keep a mapped Metal buffer")
            XCTAssertEqual(unaligned.asArray(UInt32.self), [1, 2, 3, 4])
            XCTAssertEqual(aligned.asArray(UInt32.self), [5, 6, 7, 8])

            let unalignedResult = MLX.sum(unaligned.asType(.float32), stream: .gpu)
            let alignedResult = MLX.sum(aligned.asType(.float32), stream: .gpu)
            XCTAssertEqual(unalignedResult.item(Float.self), 10.0, accuracy: 0.001)
            XCTAssertEqual(alignedResult.item(Float.self), 26.0, accuracy: 0.001)

            // mmap_file_region rejects a dtype-unaligned offset before it registers
            // anything. The named-bank tests map this layout at offset 4096.
            let bankPath = temporaryPath.appending(
                path: "unaligned-named-bank.bin",
                directoryHint: .notDirectory
            )
            try writeUInt32Bank(at: bankPath, offset: 4096, values: Array(1 ... 8))
            let bankTrackedBefore = mlx_safetensors_mmap_tracked_buffer_bytes()
            XCTAssertThrowsError(
                try mapNamedUInt32Bank(
                    at: bankPath,
                    offset: 4098,
                    shape: [1, 2, 2],
                    name: "model.layers.45.mlp.switch_mlp.gate_proj.tq2_packed")
            ) { error in
                XCTAssertTrue("\(error)".contains("dtype-unaligned offset"), "\(error)")
            }
            XCTAssertEqual(mlx_safetensors_mmap_advise_layer(1, 45), 0)
            XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes() - bankTrackedBefore, 0)
        }
    }

    /// Environment-selected mmap loading is retired in the MLX dependency:
    /// `MLX_SAFETENSORS_MMAP` and `MLX_SAFETENSORS_MMAP_TENSOR_BUFFERS` no longer
    /// select the mapped loader. A load reads every tensor into an owned
    /// allocation, registers no layer region, and keeps no mapped Metal buffer.
    public func testLegacyMmapEnvironmentNoLongerMapsSafetensors() throws {
        try withMLXMetallibForTests {
            let safetensorsPath = temporaryPath.appending(
                path: "legacy-mmap-environment.safetensors",
                directoryHint: .notDirectory
            )
            // Keys must match the loader's layer pattern, or the zero checks go blind.
            let firstKey = "layers.41.mlp.switch_mlp.gate_proj.tq_packed"
            let secondKey = "layers.41.mlp.switch_mlp.up_proj.tq_packed"
            try writeSparseFloat32Safetensors(
                at: safetensorsPath,
                firstKey: firstKey,
                firstValues: [1, 2, 3, 4],
                secondKey: secondKey,
                secondValues: [5, 6, 7, 8],
                gapBytes: 1 << 20)

            try withEnvironment("MLX_SAFETENSORS_MMAP", value: "1") {
                try withEnvironment("MLX_SAFETENSORS_MMAP_TENSOR_BUFFERS", value: "1") {
                    let trackedBefore = mlx_safetensors_mmap_tracked_buffer_bytes()
                    let loadedArrays = try MLX.loadArrays(url: safetensorsPath, stream: .cpu)
                    let first = try XCTUnwrap(loadedArrays[firstKey])
                    let second = try XCTUnwrap(loadedArrays[secondKey])

                    XCTAssertEqual(
                        mlx_safetensors_mmap_advise_layer(1, 41),
                        0,
                        "the legacy environment must not register layer regions")
                    XCTAssertEqual(
                        mlx_safetensors_mmap_tracked_buffer_bytes() - trackedBefore,
                        0,
                        "the legacy environment must not keep a mapped Metal buffer")

                    let result = MLX.sum(first + second, stream: .gpu)
                    XCTAssertEqual(result.item(Float.self), 36.0, accuracy: 0.001)
                }
            }

            // Positive control: the probes above can see layer 41. A named bank
            // there registers its 16 bytes and a tracked buffer.
            let bankPath = temporaryPath.appending(
                path: "legacy-mmap-environment-control.bin",
                directoryHint: .notDirectory
            )
            try writeUInt32Bank(at: bankPath, offset: 4096, values: [1, 2, 3, 4])
            let controlTrackedBefore = mlx_safetensors_mmap_tracked_buffer_bytes()
            let bank = try mapNamedUInt32Bank(
                at: bankPath,
                offset: 4096,
                shape: [1, 2, 2],
                name: "model.layers.41.mlp.switch_mlp.gate_proj.tq2_packed")
            withExtendedLifetime(bank) {
                XCTAssertEqual(mlx_safetensors_mmap_advise_layer(1, 41), 16)
                XCTAssertGreaterThan(
                    mlx_safetensors_mmap_tracked_buffer_bytes() - controlTrackedBefore, 0)
            }
        }
    }

    /// Environment-selected mmap loading is retired in the MLX dependency, and
    /// `exactTensorBuffers` no longer selects per-tensor mappings. With an empty
    /// exclusion list the option still takes the option-carrying loader, which
    /// returns every tensor in an owned allocation.
    public func testExactTensorBuffersWithNoExcludedKeysMapsNothing() throws {
        try withMLXMetallibForTests {
            let url = temporaryPath.appending(path: "exact-no-exclusions.safetensors")
            // Keys must match the loader's layer pattern, or the zero checks go blind.
            let firstKey = "layers.0.mlp.switch_mlp.gate_proj.tq_packed"
            let secondKey = "layers.0.mlp.switch_mlp.up_proj.tq_packed"
            try writeSparseFloat32Safetensors(
                at: url, firstKey: firstKey, firstValues: [1, 2, 3, 4],
                secondKey: secondKey, secondValues: [5, 6, 7, 8], gapBytes: 1 << 20)
            // The legacy switches stay set: they must not bring a mapping back.
            try withEnvironment("MLX_SAFETENSORS_MMAP", value: "1") {
                try withEnvironment("MLX_SAFETENSORS_MMAP_TENSOR_BUFFERS", value: "0") {
                    let before = mlx_safetensors_mmap_tracked_buffer_bytes()
                    let (arrays, _) = try MLX.loadArraysAndMetadata(
                        url: url, excludingKeys: [], exactTensorBuffers: true)
                    let tracked = mlx_safetensors_mmap_tracked_buffer_bytes() - before
                    XCTAssertEqual(tracked, 0,
                        "an empty exclusion list must not map the shard either")
                    XCTAssertEqual(arrays.keys.sorted(), [firstKey, secondKey].sorted())
                    let first = try XCTUnwrap(arrays[firstKey])
                    let second = try XCTUnwrap(arrays[secondKey])
                    XCTAssertEqual(MLX.sum(first + second, stream: .gpu).item(Float.self), 36)
                }
            }
        }
    }

    /// With `MLX_SAFETENSORS_MMAP_COLD_ADVICE=force`, cold advice on a registered
    /// region takes the `msync` invalidate path. Registered regions now come from
    /// named mmap banks; environment-selected safetensors mmap loading is retired
    /// in the MLX dependency. The test shows that the call succeeds and that the
    /// bank refaults into GPU computation. It cannot show that `force` was
    /// parsed: the default `MADV_DONTNEED` path returns the same length.
    public func testNamedMmapBankForceInvalidateCanRefaultGPUComputation() throws {
        try withMLXMetallibForTests {
            let path = temporaryPath.appending(
                path: "named-mmap-bank-force-invalidate.bin",
                directoryHint: .notDirectory
            )
            let offset = 4096
            try writeUInt32Bank(at: path, offset: offset, values: [1, 2, 3, 4])
            let bank = try mapNamedUInt32Bank(
                at: path,
                offset: offset,
                shape: [1, 2, 2],
                name: "model.layers.43.mlp.switch_mlp.gate_proj.tq2_packed")

            withEnvironment("MLX_SAFETENSORS_MMAP_COLD_ADVICE", value: "force") {
                withExtendedLifetime(bank) {
                    XCTAssertEqual(
                        mlx_safetensors_mmap_advise_layer(0, 43),
                        16,
                        "force invalidate should advise the registered layer region")

                    let result = MLX.sum(
                        (bank.asType(.float32) * MLXArray(2.0)) + MLXArray(1.0), stream: .gpu)
                    XCTAssertEqual(result.item(Float.self), 24.0, accuracy: 0.001)
                }
            }
        }
    }

    public func testMmapFileRegionCanFeedGPUComputation() throws {
        try withMLXMetallibForTests {
            let path = temporaryPath.appending(
                path: "mmap-region.bin",
                directoryHint: .notDirectory
            )
            let offset = 4096
            var data = Data(repeating: 0, count: offset)
            for value in [Float(1), Float(2), Float(3), Float(4)] {
                var bits = value.bitPattern.littleEndian
                withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
            }
            try data.write(to: path)

            var array = mlx_array_new()
            var shape = [Int32(4)]
            let rc = path.withUnsafeFileSystemRepresentation { pathPtr -> Int32 in
                guard let pathPtr else { return 1 }
                return mlx_array_new_mmap_file_region(
                    &array,
                    pathPtr,
                    UInt64(offset),
                    4 * MemoryLayout<Float>.stride,
                    &shape,
                    Int32(shape.count),
                    DType.float32.cmlxDtype)
            }
            XCTAssertEqual(rc, 0)

            let loaded = MLXArray(array)
            let result = MLX.sum((loaded * MLXArray(4.0)) + MLXArray(1.0), stream: .gpu)
            XCTAssertEqual(result.item(Float.self), 44.0, accuracy: 0.001)
        }
    }

    public func testSaveArray() throws {
        // single array npy file
        let path = temporaryPath.appending(
            path: "array.npy",
            directoryHint: .notDirectory
        )

        let array = MLX.ones([2, 4])

        try MLX.save(array: array, url: path)

        let loaded = try MLX.loadArray(url: path)

        assertEqual(array, loaded)
    }

    public func testSaveArraysData() throws {
        let arrays: [String: MLXArray] = [
            "foo": MLX.ones([1, 2]),
            "bar": MLX.zeros([2, 1]),
        ]

        let data = try saveToData(arrays: arrays)
        let loadedArrays = try loadArrays(data: data)
        XCTAssertEqual(loadedArrays.keys.sorted(), arrays.keys.sorted())

        assertEqual(try XCTUnwrap(loadedArrays["foo"]), try XCTUnwrap(arrays["foo"]))
        assertEqual(try XCTUnwrap(loadedArrays["bar"]), try XCTUnwrap(arrays["bar"]))
    }

    public func testSaveArraysMetadataData() throws {
        let arrays: [String: MLXArray] = [
            "foo": MLX.ones([1, 2]),
            "bar": MLX.zeros([2, 1]),
        ]
        let metadata = [
            "key": "value",
            "key2": "value2",
        ]

        let data = try saveToData(arrays: arrays, metadata: metadata)
        let (loadedArrays, loadedMetadata) = try loadArraysAndMetadata(data: data)
        XCTAssertEqual(loadedArrays.keys.sorted(), arrays.keys.sorted())

        assertEqual(try XCTUnwrap(loadedArrays["foo"]), try XCTUnwrap(arrays["foo"]))
        assertEqual(try XCTUnwrap(loadedArrays["bar"]), try XCTUnwrap(arrays["bar"]))
        XCTAssertEqual(loadedMetadata, metadata)
    }

}

private func writeMixedAlignmentUInt32Safetensors(
    at url: URL,
    unalignedKey: String,
    unalignedValues: [UInt32],
    alignedKey: String,
    alignedValues: [UInt32],
    shape: [Int]
) throws {
    let unalignedByteCount = unalignedValues.count * MemoryLayout<UInt32>.stride
    let alignmentPadding = 2
    let alignedStart = unalignedByteCount + alignmentPadding
    let alignedEnd = alignedStart + alignedValues.count * MemoryLayout<UInt32>.stride
    var header = """
    {"__metadata__":{},"\(unalignedKey)":{"dtype":"U32","shape":\(shape),"data_offsets":[0,\(unalignedByteCount)]},"\(alignedKey)":{"dtype":"U32","shape":\(shape),"data_offsets":[\(alignedStart),\(alignedEnd)]}}
    """
    while (8 + header.utf8.count) % MemoryLayout<UInt32>.stride != 2 {
        header.append(" ")
    }

    var data = Data()
    var headerLength = UInt64(header.utf8.count).littleEndian
    withUnsafeBytes(of: &headerLength) { data.append(contentsOf: $0) }
    data.append(contentsOf: header.utf8)
    for value in unalignedValues {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }
    data.append(contentsOf: repeatElement(UInt8(0), count: alignmentPadding))
    for value in alignedValues {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }
    try data.write(to: url)
}

private func writeSparseFloat32Safetensors(
    at url: URL,
    firstKey: String,
    firstValues: [Float],
    secondKey: String,
    secondValues: [Float],
    gapBytes: Int,
    metadataJSON: String = "{}"
) throws {
    let firstByteCount = firstValues.count * MemoryLayout<Float>.stride
    let gapStart = ((firstByteCount + 15) / 16) * 16
    let secondStart = gapStart + max(0, gapBytes)
    let secondEnd = secondStart + secondValues.count * MemoryLayout<Float>.stride
    var header = """
    {"__metadata__":\(metadataJSON),"\(firstKey)":{"dtype":"F32","shape":[\(firstValues.count)],"data_offsets":[0,\(firstByteCount)]},"\(
        secondKey)":{"dtype":"F32","shape":[\(secondValues.count)],"data_offsets":[\(secondStart),\(secondEnd)]}}
    """
    while (8 + header.utf8.count) % MemoryLayout<Float>.stride != 0 {
        header.append(" ")
    }

    var data = Data()
    var headerLength = UInt64(header.utf8.count).littleEndian
    withUnsafeBytes(of: &headerLength) { data.append(contentsOf: $0) }
    data.append(contentsOf: header.utf8)
    for value in firstValues {
        var bits = value.bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
    }
    data.append(Data(repeating: 0, count: secondStart - firstByteCount))
    for value in secondValues {
        var bits = value.bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
    }
    try data.write(to: url)
}

private func withEnvironment<T>(_ key: String, value: String, body: () throws -> T) rethrows -> T {
    let oldValue = getenv(key).map { String(cString: $0) }
    setenv(key, value, 1)
    defer {
        if let oldValue {
            setenv(key, oldValue, 1)
        } else {
            unsetenv(key)
        }
    }
    return try body()
}

private func writeUInt32Bank(at url: URL, offset: Int, values: [UInt32]) throws {
    var data = Data(repeating: 0, count: offset)
    for value in values {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }
    try data.write(to: url)
}

private struct NamedBankMappingFailed: Error {
    let code: Int32
}

/// Maps the UInt32 bank of `shape` at `offset` in `url` as the named JANGH bank
/// `name`, which registers its bytes under the layer in the name.
private func mapNamedUInt32Bank(
    at url: URL,
    offset: Int,
    shape: [Int32],
    name: String
) throws -> MLXArray {
    var raw = mlx_array_new()
    var dimensions = shape
    let length = shape.reduce(1) { $0 * Int($1) } * MemoryLayout<UInt32>.stride
    let rc: Int32
    do {
        rc = try withError {
            url.withUnsafeFileSystemRepresentation { file -> Int32 in
                guard let file else { return 1 }
                return name.withCString { tensor in
                    mlx_array_new_mmap_file_region_named(
                        &raw, file, UInt64(offset), length, &dimensions,
                        Int32(dimensions.count), DType.uint32.cmlxDtype, tensor)
                }
            }
        }
    } catch {
        mlx_array_free(raw)
        throw error
    }
    guard rc == 0 else {
        mlx_array_free(raw)
        throw NamedBankMappingFailed(code: rc)
    }
    return MLXArray(raw)
}
