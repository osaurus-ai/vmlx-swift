import MLX
import XCTest

@testable import MLXLMCommon

final class CompilableRotatingPromotionWindowTests: XCTestCase {
    private static func rows(_ positions: Range<Int>, heads: Int = 2, dimensions: Int = 8)
        -> MLXArray
    {
        MLXArray((0 ..< heads).flatMap { head in
            positions.flatMap { position in
                Array(repeating: Float(position + head * 100), count: dimensions)
            }
        }, [1, heads, positions.count, dimensions])
    }

    private static func append(_ positions: Range<Int>, to cache: RotatingKVCache) {
        _ = cache.update(
            keys: rows(positions), values: rows(positions, dimensions: 4))
        eval(cache)
    }

    func testOversizePrefillPromotionRetainsPrefixAndLatestTailWithoutChangingSource() throws {
        try MLXMetalTestLock.withLock {
            for keep in [0, 2] {
                let source = RotatingKVCache(maxSize: 8, keep: keep, step: 8)
                Self.append(0 ..< 8, to: source)
                Self.append(8 ..< 11, to: source)
                XCTAssertEqual(source.state[0].dim(2), 10)
                let sourceMetadata = source.metaState
                let sourceState = source.state.map { $0.asArray(Float.self) }
                let promoted = CompilableRotatingKVCache(from: source)
                eval(promoted)
                XCTAssertEqual(promoted.innerState()[0].shape, [1, 2, 8, 8])
                XCTAssertEqual(promoted.innerState()[1].shape, [1, 2, 8, 4])
                XCTAssertEqual(promoted.offsetArray.item(Int.self), 11)
                XCTAssertEqual(promoted.idxArray.item(Int.self), keep)
                let expectedPositions = Array(0 ..< keep) + Array((3 + keep) ..< 11)
                for head in 0 ..< 2 {
                    let actual = promoted.innerState()[0][0, head, 0..., 0].asArray(Float.self)
                    XCTAssertEqual(actual, expectedPositions.map { Float($0 + head * 100) })
                }
                _ = promoted.update(keys: Self.rows(11 ..< 12), values: Self.rows(11 ..< 12, dimensions: 4))
                eval(promoted)
                XCTAssertEqual(source.metaState, sourceMetadata)
                XCTAssertEqual(source.state.map { $0.asArray(Float.self) }, sourceState)
            }
        }
    }

    func testExactCapacityPromotionNormalizesWriteAndOwnsState() throws {
        try MLXMetalTestLock.withLock {
            for keep in [0, 2] {
                let source = RotatingKVCache(maxSize: 8, keep: keep, step: 8)
                Self.append(0 ..< 8, to: source)
                let original = source.state.map { $0.asArray(Float.self) }
                let promoted = CompilableRotatingKVCache(from: source)
                XCTAssertEqual(promoted.idxArray.item(Int.self), keep)
                _ = promoted.update(keys: Self.rows(8 ..< 9), values: Self.rows(8 ..< 9, dimensions: 4))
                eval(promoted)
                var expected = Array(0 ..< 8).map(Float.init)
                expected[keep] = 8
                XCTAssertEqual(promoted.innerState()[0][0, 0, 0..., 0].asArray(Float.self), expected)
                XCTAssertEqual(source.state.map { $0.asArray(Float.self) }, original)
                XCTAssertEqual(source.offset, 8)
            }
        }
    }

    func testProductionWindow1024WithThreeTokenPrefillTail() throws {
        try MLXMetalTestLock.withLock {
            let source = RotatingKVCache(maxSize: 1024, step: 1024)
            Self.append(0 ..< 1024, to: source)
            Self.append(1024 ..< 1027, to: source)
            XCTAssertEqual(source.state[0].dim(2), 1026)
            let promoted = CompilableRotatingKVCache(from: source)
            guard case .array(let mask) = promoted.makeMask(n: 1, windowSize: 1024, returnArray: true) else {
                return XCTFail("Expected an explicit fixed-capacity mask")
            }
            XCTAssertEqual(promoted.innerState()[0].dim(2), 1024)
            XCTAssertEqual(mask.shape, [1, 1024])
            XCTAssertEqual(mask.asArray(Bool.self).filter { $0 }.count, 1024)
        }
    }

    private static func windowMatrix(compiled: Bool) throws {
        for (keep, window) in [(0, 8), (0, 4), (2, 4), (2, 8)] {
            for seed in [3, 8, 11, 19] {
                let source = RotatingKVCache(maxSize: 8, keep: keep, step: 8)
                for token in 0 ..< seed { append(token ..< token + 1, to: source) }
                let original = source.state.map { $0.asArray(Float.self) }
                let cache = CompilableRotatingKVCache(from: source)
                eval(cache)
                let step: @Sendable ([MLXArray]) -> [MLXArray] = { args in
                    // Gemma builds its type-shared mask before the KV owner updates.
                    let mask = cache.makeMask(n: 1, windowSize: window, returnArray: false)
                    let pair = cache.update(keys: args[0], values: args[1])
                    guard case .array(let array) = mask else {
                        preconditionFailure("Expected an explicit fixed-capacity mask")
                    }
                    let output = MLXFast.scaledDotProductAttention(
                        queries: args[2], keys: pair.0,
                        values: pair.1, scale: 1, mask: mask)
                    // update returns the mutable cache-state object. compile
                    // restores that object's original context after recording
                    // the body, so a direct result aliases an uncaptured input.
                    // Observe an independent graph value without changing V.
                    let observedValues = pair.1 + 0
                    return [output, array, observedValues]
                }
                let forward = compiled ? compile(inputs: [cache], outputs: [cache], step) : step
                let query = MLXArray.zeros([1, 2, 1, 8])
                for token in seed ..< seed + 20 {
                    let result = forward([
                        rows(token ..< token + 1), rows(token ..< token + 1, dimensions: 4), query,
                    ])
                    eval(result)
                    let allowed = result[1].asArray(Bool.self)
                    let positions = result[2][0, 0, 0..., 0].asArray(Float.self)
                    let expectedMask = positions.map { $0 >= Float(max(0, token - window + 1)) && $0 <= Float(token) }
                    // Before the ring fills, padded zeros are invalid independently of their value.
                    let validMask = expectedMask.enumerated().map { column, valid in
                        valid && (token >= 7 || column <= token)
                    }
                    XCTAssertEqual(allowed, validMask, "seed=\(seed), token=\(token), keep=\(keep), window=\(window)")
                    let visible = zip(positions, validMask).compactMap { $1 ? $0 : nil }
                    XCTAssertFalse(visible.isEmpty)
                    let mean = visible.reduce(0, +) / Float(visible.count)
                    let output = result[0].asArray(Float.self)
                    for head in 0 ..< 2 {
                        for dimension in 0 ..< 4 {
                            XCTAssertEqual(output[head * 4 + dimension], mean + Float(head * 100), accuracy: 0.0001)
                        }
                    }
                    XCTAssertEqual(cache.offsetArray.item(Int.self), token + 1)
                }
                XCTAssertEqual(source.state.map { $0.asArray(Float.self) }, original)
                XCTAssertEqual(source.offset, seed)
            }
        }
    }

    func testLogicalWindowMasksAcrossWrapsAndPinnedColumnsEager() throws {
        try MLXMetalTestLock.withLock { try Self.windowMatrix(compiled: false) }
    }

    func testLogicalWindowMasksAcrossWrapsAndPinnedColumnsCompiled() throws {
        try MLXMetalTestLock.withLock {
            guard HardwareInfo.isCompiledDecodeSupported else { throw XCTSkip("Compiled decode unsupported") }
            try Self.windowMatrix(compiled: true)
        }
    }
}
