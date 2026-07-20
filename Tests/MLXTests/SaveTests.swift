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

    public func testMmapSafetensorsLoadCanFeedGPUComputation() throws {
        try withMLXMetallibForTests {
            let safetensorsPath = temporaryPath.appending(
                path: "mmap-gpu-arrays.safetensors",
                directoryHint: .notDirectory
            )
            let key = "layers.0.mlp.switch_mlp.gate_proj.tq_packed"
            let values: [Float] = (1 ... 8).map { Float($0) }
            try writeAlignedFloat32Safetensors(
                at: safetensorsPath,
                key: key,
                values: values,
                shape: [2, 4])

            try withEnvironment("MLX_SAFETENSORS_MMAP", value: "1") {
                try withEnvironment("MLX_SAFETENSORS_MMAP_START_COLD", value: "1") {
                    try withEnvironment("MLX_SAFETENSORS_MMAP_COLD_PCT", value: "100") {
                        let loadedArrays = try MLX.loadArrays(url: safetensorsPath, stream: .cpu)
                        let loaded = try XCTUnwrap(loadedArrays[key])
                        let advisedBytes = mlx_safetensors_mmap_advise_layer(1, 0)
                        guard advisedBytes > 0 else {
                            XCTFail("mmap loader did not register layer regions")
                            return
                        }

                        let result = MLX.sum((loaded * MLXArray(3.0)) + MLXArray(1.0), stream: .gpu)
                        XCTAssertEqual(result.item(Float.self), 116.0, accuracy: 0.001)
                    }
                }
            }
        }
    }

    public func testMmapSafetensorsKeepsUnalignedTensorInMappedShard() throws {
        try withMLXMetallibForTests {
            let safetensorsPath = temporaryPath.appending(
                path: "mmap-unaligned-u32.safetensors",
                directoryHint: .notDirectory
            )
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

            try withEnvironment("MLX_SAFETENSORS_MMAP", value: "1") {
                let loadedArrays = try MLX.loadArrays(url: safetensorsPath, stream: .cpu)
                let unaligned = try XCTUnwrap(loadedArrays[unalignedKey])
                let aligned = try XCTUnwrap(loadedArrays[alignedKey])
                let unalignedAdvisedBytes = mlx_safetensors_mmap_advise_layer(1, 97)
                let alignedAdvisedBytes = mlx_safetensors_mmap_advise_layer(1, 98)
                XCTAssertEqual(
                    unalignedAdvisedBytes,
                    0,
                    "the realigned tensor must not be registered as mmap-backed"
                )
                XCTAssertGreaterThan(
                    alignedAdvisedBytes,
                    0,
                    "an unaligned tensor must not make aligned tensors in the same shard fall back"
                )
                XCTAssertEqual(unaligned.asArray(UInt32.self), [1, 2, 3, 4])
                XCTAssertEqual(aligned.asArray(UInt32.self), [5, 6, 7, 8])

                let unalignedResult = MLX.sum(unaligned.asType(.float32), stream: .gpu)
                let alignedResult = MLX.sum(aligned.asType(.float32), stream: .gpu)
                XCTAssertEqual(unalignedResult.item(Float.self), 10.0, accuracy: 0.001)
                XCTAssertEqual(alignedResult.item(Float.self), 26.0, accuracy: 0.001)
            }
        }
    }

    public func testMmapSafetensorsTensorBufferModeDoesNotTrackWholeShard() throws {
        try withMLXMetallibForTests {
            let safetensorsPath = temporaryPath.appending(
                path: "mmap-tensor-buffer-arrays.safetensors",
                directoryHint: .notDirectory
            )
            let firstKey = "layers.0.mlp.switch_mlp.gate_proj.tq_packed"
            let secondKey = "layers.0.mlp.switch_mlp.up_proj.tq_packed"
            try writeSparseFloat32Safetensors(
                at: safetensorsPath,
                firstKey: firstKey,
                firstValues: [1, 2, 3, 4],
                secondKey: secondKey,
                secondValues: [5, 6, 7, 8],
                gapBytes: 1 << 20)

            let fileBytes = try FileManager.default
                .attributesOfItem(atPath: safetensorsPath.path)[.size] as? Int
                ?? 0

            try withEnvironment("MLX_SAFETENSORS_MMAP", value: "1") {
                try withEnvironment("MLX_SAFETENSORS_MMAP_TENSOR_BUFFERS", value: "1") {
                    let loadedArrays = try MLX.loadArrays(url: safetensorsPath, stream: .cpu)
                    let first = try XCTUnwrap(loadedArrays[firstKey])
                    let second = try XCTUnwrap(loadedArrays[secondKey])
                    let advisedBytes = mlx_safetensors_mmap_advise_layer(1, 0)
                    let trackedBytes = mlx_safetensors_mmap_tracked_buffer_bytes()

                    XCTAssertGreaterThan(
                        advisedBytes,
                        0,
                        "tensor-buffer mmap mode should register layer regions")
                    XCTAssertGreaterThan(
                        trackedBytes,
                        0,
                        "tensor-buffer mmap mode should register live mmap-backed Metal buffers")
                    XCTAssertLessThan(
                        trackedBytes,
                        Int64(fileBytes / 4),
                        "tensor-buffer mmap mode should not keep a Metal buffer over the whole shard")

                    let result = MLX.sum(first + second, stream: .gpu)
                    XCTAssertEqual(result.item(Float.self), 36.0, accuracy: 0.001)
                }
            }
        }
    }

    public func testMmapSafetensorsForceInvalidateCanRefaultGPUComputation() throws {
        try withMLXMetallibForTests {
            let safetensorsPath = temporaryPath.appending(
                path: "mmap-force-invalidate-arrays.safetensors",
                directoryHint: .notDirectory
            )
            let key = "layers.0.mlp.switch_mlp.gate_proj.tq_packed"
            try writeAlignedFloat32Safetensors(
                at: safetensorsPath,
                key: key,
                values: [1, 2, 3, 4],
                shape: [4])

            try withEnvironment("MLX_SAFETENSORS_MMAP", value: "1") {
                try withEnvironment("MLX_SAFETENSORS_MMAP_TENSOR_BUFFERS", value: "1") {
                    try withEnvironment("MLX_SAFETENSORS_MMAP_COLD_ADVICE", value: "force") {
                        let loadedArrays = try MLX.loadArrays(url: safetensorsPath, stream: .cpu)
                        let loaded = try XCTUnwrap(loadedArrays[key])
                        let advisedBytes = mlx_safetensors_mmap_advise_layer(0, 0)
                        XCTAssertGreaterThan(
                            advisedBytes,
                            0,
                            "force invalidate should advise registered layer regions")

                        let result = MLX.sum((loaded * MLXArray(2.0)) + MLXArray(1.0), stream: .gpu)
                        XCTAssertEqual(result.item(Float.self), 24.0, accuracy: 0.001)
                    }
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

private func writeAlignedFloat32Safetensors(
    at url: URL,
    key: String,
    values: [Float],
    shape: [Int]
) throws {
    let byteCount = values.count * MemoryLayout<Float>.stride
    var header = """
    {"__metadata__":{},"\(key)":{"dtype":"F32","shape":\(shape),"data_offsets":[0,\(byteCount)]}}
    """
    while (8 + header.utf8.count) % MemoryLayout<Float>.stride != 0 {
        header.append(" ")
    }

    var data = Data()
    var headerLength = UInt64(header.utf8.count).littleEndian
    withUnsafeBytes(of: &headerLength) { data.append(contentsOf: $0) }
    data.append(contentsOf: header.utf8)
    for value in values {
        var bits = value.bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
    }
    try data.write(to: url)
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
    gapBytes: Int
) throws {
    let firstByteCount = firstValues.count * MemoryLayout<Float>.stride
    let gapStart = ((firstByteCount + 15) / 16) * 16
    let secondStart = gapStart + max(0, gapBytes)
    let secondEnd = secondStart + secondValues.count * MemoryLayout<Float>.stride
    var header = """
    {"__metadata__":{},"\(firstKey)":{"dtype":"F32","shape":[\(firstValues.count)],"data_offsets":[0,\(firstByteCount)]},"\(
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
