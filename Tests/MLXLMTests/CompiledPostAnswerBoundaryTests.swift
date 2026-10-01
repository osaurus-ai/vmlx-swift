import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class CompiledPostAnswerBoundaryTests: XCTestCase {
    private final class TraceCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var value = 0
        func increment() { lock.lock(); defer { lock.unlock() }; value += 1 }
        func read() -> Int { lock.lock(); defer { lock.unlock() }; return value }
    }

    private func row(_ value: Float, count: Int = 1) -> MLXArray {
        MLXArray((0..<(2 * count * 4)).map { Float($0) + value }).reshaped([1, 2, count, 4])
    }

    private func configuration(_ directory: URL) -> CacheCoordinatorConfig {
        CacheCoordinatorConfig(
            usePagedCache: false, enableDiskCache: true, diskCacheMaxGB: 1,
            diskCacheDir: directory, modelKey: "compiled-post-answer")
    }

    /// Exercises actual traced updates, production key admission, copy, coordinator
    /// storage, a separate disk reader, restore, and the next native update.
    func testCompiledPostAnswerPersistsAndContinuesAcrossCacheTopologies() throws {
        guard HardwareInfo.isCompiledDecodeSupported else {
            throw XCTSkip("Requires supported compiled decode; a skip is not runtime proof")
        }
        try MLXMetalTestLock.withLock {
            // Keep the native compile-cache context fixed. Core keys traces
            // by peek_default_stream(), whereas Swift passes Stream.gpu to
            // operations explicitly. The first safetensors write invokes
            // native contiguous() without a stream and lazily creates a core
            // default, legitimately changing that key. Bind the same stream
            // before tracing so this assertion measures storage/state changes,
            // rather than one-time initialization of a different context.
            Stream.restoreDefault()
            // The wrapped case begins after a native wrap and stays away from
            // idx==capacity. It isolates metadata export from the separate ring-write fix.
            for (capacity, promptLength) in [(16, 3), (8, 10)] {
                // Independent control: three identical-shape native calls with
                // no snapshot, metadata synchronization or disk operations.
                let controlRaw = RotatingKVCache(maxSize: capacity)
                for token in 0..<promptLength {
                    let next = row(Float(token * 10))
                    _ = controlRaw.update(keys: next, values: next * 2)
                }
                let control = CompilableRotatingKVCache(from: controlRaw)
                eval(control)
                let controlCounter = TraceCounter()
                let controlForward: @Sendable ([MLXArray]) -> [MLXArray] = compile(
                    inputs: [control], outputs: [control]
                ) { args in
                    controlCounter.increment()
                    let pair = control.update(keys: args[0], values: args[0] * 2)
                    return [pair.0 + 0, pair.1 + 0]
                }
                for value in [Float(100), Float(110), Float(120)] {
                    eval(controlForward([row(value)]))
                    XCTAssertEqual(controlCounter.read(), 1, "Unstored control must keep its trace")
                    print("postanswer no-store control capacity=\(capacity) value=\(value) traces=\(controlCounter.read())")
                }
                for layout in 0..<3 {
                    print("postanswer variant capacity=\(capacity) prompt=\(promptLength) layout=\(layout)")
                    let boundary = promptLength + 2
                    let prompt = Array(1...promptLength)
                    let generated = [promptLength + 1, boundary]
                    let directory = FileManager.default.temporaryDirectory
                        .appendingPathComponent("compiled-postanswer-\(UUID().uuidString)")
                    defer { try? FileManager.default.removeItem(at: directory) }
                    let raw = RotatingKVCache(maxSize: capacity)
                    let simpleRaw = KVCacheSimple()
                    for token in 0..<promptLength {
                        let first = row(Float(token * 10))
                        _ = raw.update(keys: first, values: first * 2)
                        _ = simpleRaw.update(keys: first, values: first * 2)
                    }
                    let promptSnapshot = makePromptBoundaryCacheSnapshot(from: [raw])
                    let promptBytes = promptSnapshot[0].state[0].asArray(Float.self)
                    let reference = raw.copy() as! RotatingKVCache
                    let simpleReference = simpleRaw.copy() as! KVCacheSimple
                    let rotating = CompilableRotatingKVCache(from: raw)
                    let simple = CompilableKVCache(from: simpleRaw, maxLength: 32)
                    let includeSimple = layout != 0
                    let caches: [any KVCache] = layout == 0 ? [rotating]
                        : layout == 1 ? [simple, rotating]
                        : [CompilableCacheList(compilableSubCaches: [simple, rotating])]
                    eval(caches)
                    let traceCounter = TraceCounter()
                    let forward: @Sendable ([MLXArray]) -> [MLXArray] = compile(
                        inputs: caches, outputs: caches
                    ) { args in
                        traceCounter.increment()
                        let kv = rotating.update(keys: args[0], values: args[0] * 2)
                        if includeSimple {
                            let full = simple.update(keys: args[0], values: args[0] * 2)
                            return [kv.0 + 0, kv.1 + 0, full.0 + 0, full.1 + 0]
                        }
                        return [kv.0 + 0, kv.1 + 0]
                    }
                    for value in [Float(100), Float(110)] {
                        let next = row(value)
                        eval(forward([next]))
                        _ = reference.update(keys: next, values: next * 2)
                        _ = simpleReference.update(keys: next, values: next * 2)
                    }
                    let tracedCount = traceCounter.read()
                    XCTAssertGreaterThan(tracedCount, 0)
                    if includeSimple { XCTAssertEqual(simple.offset, boundary) }
                    XCTAssertEqual(rotating.offsetArray.item(Int.self), boundary)
                    XCTAssertEqual(rotating.offset, promptLength, "Original host metadata remains at promotion")
                    let key = prompt + generated
                    let writer = CacheCoordinator(config: configuration(directory))
                    // Negative control: the real coordinator must refuse the stale
                    // original cache; do not weaken its exact-all-leaves guard.
                    writer.storeAfterGeneration(
                        promptTokens: key, perLayerData: [], ssmStates: nil,
                        cache: caches, isPostAnswer: true)
                    XCTAssertNil(writer.diskCache?.fetch(tokens: key))
                    let indexIdentity = rotating.idxArray
                    let offsetIdentity = rotating.offsetArray
                    synchronizeCompiledRotatingCacheMetadataForStorage(caches)
                    synchronizeCompiledRotatingCacheMetadataForStorage(caches)
                    XCTAssertTrue(rotating.idxArray === indexIdentity)
                    XCTAssertTrue(rotating.offsetArray === offsetIdentity)
                    XCTAssertTrue(cacheCoversTokenCount(key.count, cache: caches))
                    XCTAssertEqual(
                        TokenIterator.generatedBoundaryTokensAligned(
                            promptTokenIds: prompt, generatedTokenIds: generated,
                            cacheOffsets: cacheBoundaryLeafOffsets(caches), pendingDrainedTokenId: nil), key)
                    let snapshot = makePromptBoundaryCacheSnapshot(from: caches)
                    writer.storeAfterGeneration(
                        promptTokens: key, perLayerData: [], ssmStates: nil,
                        cache: snapshot, isPostAnswer: true)
                    let reader = CacheCoordinator(config: configuration(directory))
                    let arrays = try XCTUnwrap(reader.diskCache?.fetch(tokens: key))
                    XCTAssertTrue(try FileManager.default.contentsOfDirectory(
                        at: directory, includingPropertiesForKeys: nil).contains { $0.pathExtension == "safetensors" })
                    for requireBoundary in [false, true] {
                        let restoredRotating = RotatingKVCache(maxSize: capacity)
                        let restoredSimple = KVCacheSimple()
                        var restored: [any KVCache] = layout == 0 ? [restoredRotating]
                            : layout == 1 ? [restoredSimple, restoredRotating]
                            : [CacheList(restoredSimple, restoredRotating)]
                        XCTAssertEqual(restoreFromDiskArrays(
                            arrays, into: &restored, requirePromptBoundary: requireBoundary), boundary)
                        XCTAssertTrue(validateRestoredCacheBoundary(restored, matchedTokens: boundary, restoredTokens: boundary))
                        let expected = reference.copy() as! RotatingKVCache
                        let next = row(120)
                        let a = restoredRotating.update(keys: next, values: next * 2)
                        let b = expected.update(keys: next, values: next * 2)
                        XCTAssertEqual(a.0.asArray(Float.self), b.0.asArray(Float.self))
                        XCTAssertEqual(a.1.asArray(Float.self), b.1.asArray(Float.self))
                        if includeSimple {
                            let expectedSimple = simpleReference.copy() as! KVCacheSimple
                            let fullActual = restoredSimple.update(keys: next, values: next * 2)
                            let fullExpected = expectedSimple.update(keys: next, values: next * 2)
                            XCTAssertEqual(fullActual.0.asArray(Float.self), fullExpected.0.asArray(Float.self))
                            XCTAssertEqual(fullActual.1.asArray(Float.self), fullExpected.1.asArray(Float.self))
                        }
                    }
                    // Host materialization must not break the existing trace, and
                    // must never rewrite the retained prompt boundary.
                    eval(forward([row(120)]))
                    print("postanswer continued traces=\(traceCounter.read())")
                    XCTAssertEqual(traceCounter.read(), tracedCount, "Storage must not retrace the forward")
                    if includeSimple { XCTAssertEqual(simple.offset, boundary + 1) }
                    synchronizeCompiledRotatingCacheMetadataForStorage(caches)
                    XCTAssertEqual(rotating.offset, boundary + 1)
                    XCTAssertEqual(
                        TokenIterator.generatedBoundaryTokensAligned(
                            promptTokenIds: prompt, generatedTokenIds: generated,
                            cacheOffsets: cacheBoundaryLeafOffsets(caches), pendingDrainedTokenId: boundary + 1),
                        key + [boundary + 1], "Preserve the forwarded-stop key extension")
                    XCTAssertEqual(snapshot.flatMap { cacheBoundaryLeafOffsets([$0]) }, Array(repeating: boundary, count: includeSimple ? 2 : 1))
                    XCTAssertEqual(promptSnapshot[0].offset, promptLength)
                    XCTAssertEqual(promptSnapshot[0].state[0].asArray(Float.self), promptBytes)
                }
            }
        }
    }

    func testNestedTraversalDoesNotAdmitGenuineLeafMismatch() throws {
        try MLXMetalTestLock.withLock {
            let raw = RotatingKVCache(maxSize: 16)
            let initial = row(1, count: 3)
            _ = raw.update(keys: initial, values: initial)
            let rotating = CompilableRotatingKVCache(from: raw)
            _ = rotating.update(keys: row(4), values: row(4))
            let other = KVCacheSimple()
            _ = other.update(keys: row(1, count: 5), values: row(1, count: 5))
            let nested = CacheList(CacheList(rotating), other)
            synchronizeCompiledRotatingCacheMetadataForStorage([nested])
            XCTAssertEqual(cacheBoundaryLeafOffsets([nested]), [4, 5])
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("compiled-postanswer-mismatch-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: directory) }
            let coordinator = CacheCoordinator(config: configuration(directory))
            coordinator.storeAfterGeneration(
                promptTokens: [1, 2, 3, 4, 5], perLayerData: [], ssmStates: nil,
                cache: [nested], isPostAnswer: true)
            XCTAssertNil(coordinator.diskCache?.fetch(tokens: [1, 2, 3, 4, 5]))
        }
    }
}
