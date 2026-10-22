import Foundation
import MLX
@testable import MLXLMCommon
import XCTest
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

final class DiskPromptSnapshotTests: XCTestCase {
    func testNativeSnapshotOutlivesReleasedSourceAndCleansItsFile() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var source: KVCacheSimple? = KVCacheSimple()
        weak var weakSource = source
        let rows = MLXArray((0..<48).map(Float.init)).reshaped(1, 2, 3, 8).asType(.bfloat16)
        _ = source!.update(keys: rows, values: rows + 10)
        var checkpoint: DiskPromptSnapshot? = try .capture(cache: [source!], tokenCount: 3, directory: directory)
        let descriptor = checkpoint!.fileDescriptor
        source = nil
        XCTAssertNil(weakSource, "The checkpoint must not retain the live cache object")
        let restored = try checkpoint!.restore(into: [KVCacheSimple()])
        XCTAssertEqual(restored[0].offset, 3)
        XCTAssertEqual(restored[0].state[0].dtype, .bfloat16)
        XCTAssertEqual(restored[0].state[0].asType(.float32).asArray(Float.self), rows.asType(.float32).asArray(Float.self))
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: directory.path).isEmpty)
        checkpoint = nil
        XCTAssertEqual(fcntl(descriptor, F_GETFD), -1)
        XCTAssertEqual(errno, EBADF)
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: directory.path).isEmpty)
    }

    func testWrappedSWAAndLearnedSinkSnapshotSurvivesDecodeMutation() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let sink = AttentionSinkKVCache(headCount: 2, headDimension: 8)
        let ring = RotatingKVCache(maxSize: 5, keep: 0)
        let cache: [KVCache] = [sink, ring]
        for step in 0..<13 {
            let rows = MLXArray.full([1, 2, 1, 8], values: MLXArray(Float(step))).asType(.bfloat16)
            for layer in cache { _ = layer.update(keys: rows, values: rows + 1) }
        }
        let expectedState = cache.map { $0.state.map { $0.asType(.float32).asArray(Float.self) } }
        let checkpoint = try DiskPromptSnapshot.capture(cache: cache, tokenCount: 13, directory: directory)
        for step in 13..<19 {
            let rows = MLXArray.full([1, 2, 1, 8], values: MLXArray(Float(step))).asType(.bfloat16)
            for layer in cache { _ = layer.update(keys: rows, values: rows + 1) }
        }
        let restored = try checkpoint.restore(into: [
            AttentionSinkKVCache(headCount: 2, headDimension: 8), RotatingKVCache(maxSize: 5, keep: 0),
        ])
        XCTAssertTrue(restored[0] is AttentionSinkKVCache)
        XCTAssertEqual(restored.map(\.offset), [13, 13])
        XCTAssertEqual(restored.map { $0.state.map { $0.asType(.float32).asArray(Float.self) } }, expectedState)
    }

    func testCorruptCheckpointRefusesRestoreAndStillCleansUp() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let cache = KVCacheSimple()
        let rows = MLXArray.ones([1, 1, 2, 8])
        _ = cache.update(keys: rows, values: rows)
        var checkpoint: DiskPromptSnapshot? = try .capture(cache: [cache], tokenCount: 2, directory: directory)
        try FileHandle(fileDescriptor: checkpoint!.fileDescriptor, closeOnDealloc: false).truncate(atOffset: 2)
        XCTAssertThrowsError(try checkpoint!.restore(into: [KVCacheSimple()]))
        checkpoint = nil
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: directory.path).isEmpty)
    }
}
