import Foundation
import MLX
import XCTest
#if canImport(Cmlx)
import Cmlx
#endif

@testable import MLXLMCommon

final class DFlash2PrefixContextTests: XCTestCase {
    private let layers = [5, 19]

    override func setUpWithError() throws {
        try super.setUpWithError()
        #if canImport(Cmlx)
        if let path = ProcessInfo.processInfo.environment["VMLX_TEST_METALLIB_PATH"] {
            guard FileManager.default.isReadableFile(atPath: path),
                path.withCString({ mlx_metal_set_metallib_path($0) }) == 0
            else { throw NSError(domain: "DFlash2PrefixContextTests", code: 1) }
        }
        #endif
    }

    private func context(dtype: DType = .bfloat16) throws -> DFlash2PrefixContext {
        try XCTUnwrap(DFlash2PrefixContext(
            hidden: MLXArray(Array(0..<128)).reshaped(1, 16, 8).asType(dtype),
            end: 553, layerIDs: layers))
    }

    func testDiskRoundTripPreservesValuesDtypeAndAbsoluteWindow() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        for dtype in [DType.float16, .bfloat16, .float32] {
            let original = try context(dtype: dtype)
            let arrays = try XCTUnwrap(original.diskArrays(at: 553, byteBudget: 4096))
            let file = directory.appendingPathComponent("\(dtype).safetensors")
            try MLX.save(arrays: arrays, url: file)
            let decoded = try XCTUnwrap(DFlash2PrefixContext.restore(
                from: MLX.loadArrays(url: file), boundary: 553,
                layerIDs: layers, featureWidth: 8, limit: nil))
            XCTAssertEqual(decoded.start, 537)
            XCTAssertEqual(decoded.end, 553)
            XCTAssertEqual(decoded.hidden.dtype, dtype)
            XCTAssertTrue(all(decoded.hidden .== original.hidden).item(Bool.self))
        }
    }

    func testRestoreUsesDeclaredWindowWithoutChangingBoundary() throws {
        let arrays = try XCTUnwrap(context().diskArrays(at: 553, byteBudget: 4096))
        let decoded = try XCTUnwrap(DFlash2PrefixContext.restore(
            from: arrays, boundary: 553, layerIDs: layers, featureWidth: 8, limit: 7))
        XCTAssertEqual(decoded.start, 546)
        XCTAssertEqual(decoded.end, 553)
        XCTAssertEqual(decoded.hidden.shape, [1, 7, 8])
        XCTAssertEqual(decoded.hidden[0, 0, 0].item(Int.self), 72)
    }

    func testMissingFeaturesAndMismatchedCaptureContractMiss() throws {
        let arrays = try XCTUnwrap(context().diskArrays(at: 553, byteBudget: 4096))
        XCTAssertNil(DFlash2PrefixContext.restore(
            from: [:], boundary: 553, layerIDs: layers, featureWidth: 8, limit: nil))
        for (boundary, ids, width) in [(554, layers, 8), (553, [19, 5], 8), (553, layers, 9)] {
            XCTAssertNil(DFlash2PrefixContext.restore(
                from: arrays, boundary: boundary, layerIDs: ids, featureWidth: width, limit: nil))
        }
        var missing = arrays
        missing.removeValue(forKey: DFlash2PrefixContext.hiddenKey)
        XCTAssertNil(DFlash2PrefixContext.restore(
            from: missing, boundary: 553, layerIDs: layers, featureWidth: 8, limit: nil))
    }

    func testMalformedDescriptorShapeVersionAndPositionsMiss() throws {
        let original = try XCTUnwrap(context().diskArrays(at: 553, byteBudget: 4096))
        let descriptors = [
            MLXArray([Int64(2), 537, 553, 8, 5, 19]),
            MLXArray([Int64(1), -1, 553, 8, 5, 19]),
            MLXArray([Int64(1), 538, 553, 8, 5, 19]),
            MLXArray([Int64(1), 537, 553, 8, 5, 19]).reshaped(2, 3),
            MLXArray([Int32(1), 537, 553, 8, 5, 19]),
        ]
        for descriptor in descriptors {
            var arrays = original
            arrays[DFlash2PrefixContext.metadataKey] = descriptor
            XCTAssertNil(DFlash2PrefixContext.restore(
                from: arrays, boundary: 553, layerIDs: layers, featureWidth: 8, limit: nil))
        }
    }

    func testWrongHiddenShapeDtypeAndNonfiniteValuesMiss() throws {
        let original = try XCTUnwrap(context().diskArrays(at: 553, byteBudget: 4096))
        for hidden in [
            MLXArray.zeros([2, 16, 8]), MLXArray.zeros([16, 8]),
            MLXArray.zeros([1, 16, 8], type: Int32.self),
            MLXArray.full([1, 16, 8], values: MLXArray(Float.infinity)),
            MLXArray.full([1, 16, 8], values: MLXArray(Float.nan)),
        ] {
            var arrays = original
            arrays[DFlash2PrefixContext.hiddenKey] = hidden
            XCTAssertNil(DFlash2PrefixContext.restore(
                from: arrays, boundary: 553, layerIDs: layers, featureWidth: 8, limit: nil))
        }
    }

    func testOptionalFeaturesRespectByteBudget() throws {
        let original = try context()
        let bytes = Int64(original.hidden.nbytes + (4 + layers.count) * 8)
        XCTAssertNil(original.diskArrays(at: 553, byteBudget: bytes - 1))
        XCTAssertNotNil(original.diskArrays(at: 553, byteBudget: bytes))
        XCTAssertNil(original.diskArrays(at: 554, byteBudget: 4096))
    }

    func testContiguousAppendKeepsActualTrailingPositions() throws {
        let suffix = MLXArray(Array(128..<184)).reshaped(1, 7, 8).asType(.bfloat16)
        let joined = try XCTUnwrap(DFlash2PrefixContext.appending(
            suffix, endingAt: 560, layerIDs: layers, to: context(), limit: 16))
        XCTAssertEqual(joined.start, 544)
        XCTAssertEqual(joined.end, 560)
        XCTAssertEqual(joined.hidden[0, 0, 0].item(Int.self), 56)
        XCTAssertEqual(joined.hidden[0, 15, 7].item(Int.self), 183)
        let suffixOnly = try XCTUnwrap(DFlash2PrefixContext.appending(
            suffix, endingAt: 560, layerIDs: layers, to: nil, limit: 16))
        XCTAssertEqual(suffixOnly.start, 553)
        XCTAssertEqual(suffixOnly.hidden.dim(1), 7)
    }

    func testAppendRejectsGapsAndDifferentCaptureOrder() throws {
        let suffix = MLXArray.zeros([1, 7, 8], dtype: .bfloat16)
        let previous = try context()
        XCTAssertNil(DFlash2PrefixContext.appending(
            suffix, endingAt: 561, layerIDs: layers, to: previous, limit: 16))
        XCTAssertNil(DFlash2PrefixContext.appending(
            suffix, endingAt: 560, layerIDs: [19, 5], to: previous, limit: 16))
    }

    func testAuxiliaryPayloadDoesNotChangeTargetKVRestoration() throws {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 553, 4]), values: MLXArray.ones([1, 1, 553, 4]) * 2)
        var arrays = TQDiskSerializer.serialize(cache: [cache], preserveStandardKVStorageDType: true)
        arrays.merge(try XCTUnwrap(context().diskArrays(at: 553, byteBudget: 4096))) { _, new in new }
        var restored: [KVCache] = [KVCacheSimple()]
        XCTAssertEqual(restoreFromDiskArrays(arrays, into: &restored, requirePromptBoundary: true), 553)
        XCTAssertTrue(all(restored[0].state[0] .== cache.state[0]).item(Bool.self))
        arrays[DFlash2PrefixContext.hiddenKey] = MLXArray.zeros([1, 1, 1])
        var withoutContext: [KVCache] = [KVCacheSimple()]
        XCTAssertNil(DFlash2PrefixContext.restore(
            from: arrays, boundary: 553, layerIDs: layers, featureWidth: 8, limit: nil))
        XCTAssertEqual(restoreFromDiskArrays(arrays, into: &withoutContext, requirePromptBoundary: true), 553)
    }

    private func coordinator(directory: URL, capGB: Float = 1) -> CacheCoordinator {
        CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false, enableDiskCache: true, pagedBlockSize: 4,
            maxCacheBlocks: 16, diskCacheMaxGB: capGB, diskCacheDir: directory,
            modelKey: "dflash-conditioning-test"))
    }

    private func targetCache() -> KVCacheSimple {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 553, 4]), values: MLXArray.ones([1, 1, 553, 4]) * 2)
        return cache
    }

    func testCoordinatorPersistsConditioningWithTargetAndReopensIt() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let tokens = Array(0..<553)
        let original = coordinator(directory: directory)
        original.storeAfterGeneration(
            promptTokens: tokens, perLayerData: [], ssmStates: nil,
            cache: [targetCache()], dflash2Context: try context())
        XCTAssertTrue(try XCTUnwrap(original.diskCache).waitForPendingWrites(timeout: 5))
        let reopened = coordinator(directory: directory)
        guard case .hit(let matched, _, _, _, _, let arrays) = reopened.fetch(tokens: tokens + [900])
        else { return XCTFail("stored target and conditioning did not reopen") }
        XCTAssertEqual(matched, 553)
        let restored = try XCTUnwrap(DFlash2PrefixContext.restore(
            from: XCTUnwrap(arrays), boundary: matched, layerIDs: layers, featureWidth: 8, limit: nil))
        XCTAssertEqual(restored.start, 537)
        let hash = DiskCache.hashTokens(tokens, modelKey: "dflash-conditioning-test")
        let onDisk = try MLX.loadArrays(url: directory.appendingPathComponent(hash + ".safetensors"))
        XCTAssertNotNil(onDisk[DFlash2PrefixContext.hiddenKey])
    }

    func testOversizedOptionalFeaturesStillStoreValidBaseKV() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let cache = coordinator(directory: directory, capGB: 0.001)
        let large = try XCTUnwrap(DFlash2PrefixContext(
            hidden: MLXArray.zeros([1, 16, 65536], dtype: .bfloat16), end: 553, layerIDs: layers))
        XCTAssertGreaterThan(large.hidden.nbytes, try XCTUnwrap(cache.diskCache).maxSizeBytes)
        let tokens = Array(0..<553)
        cache.storeAfterGeneration(
            promptTokens: tokens, perLayerData: [], ssmStates: nil,
            cache: [targetCache()], dflash2Context: large)
        XCTAssertTrue(try XCTUnwrap(cache.diskCache).waitForPendingWrites(timeout: 5))
        guard case .hit(let matched, _, _, _, _, let arrays) = cache.fetch(tokens: tokens + [900])
        else { return XCTFail("optional context incorrectly prevented a valid base store") }
        XCTAssertEqual(matched, 553)
        let payload = try XCTUnwrap(arrays)
        XCTAssertNil(payload[DFlash2PrefixContext.hiddenKey])
        var target: [KVCache] = [KVCacheSimple()]
        XCTAssertEqual(restoreFromDiskArrays(payload, into: &target, requirePromptBoundary: true), 553)
    }
}
