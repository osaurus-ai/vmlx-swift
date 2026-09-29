import MLX
import MLXLMCommon
import XCTest

/// Fixed-capacity masks must compare logical token positions after the ring wraps.
final class CompilableRotatingExplicitWindowTests: XCTestCase {
    private static func runExplicitWindow(
        compiled: Bool, restored: Bool, seed: Int, width: Int, window: Int, keep: Int
    ) throws {
        let original = RotatingKVCache(maxSize: width, keep: keep, step: width)
        for token in 0 ..< seed {
            let value = MLXArray.full([1, 1, 1, 8], values: MLXArray(Float(token)))
            let pair = original.update(keys: MLXArray.zeros([1, 1, 1, 8]), values: value)
            eval(pair.0, pair.1)
        }
        let source: RotatingKVCache
        if restored && seed > 0 {
            source = RotatingKVCache(maxSize: width, keep: keep, step: width)
            source.state = original.state
            source.metaState = original.metaState
        } else {
            source = original
        }
        let cache = CompilableRotatingKVCache(from: source)
        // Ensure fixed buffers exist before compile captures mutable state.
        if seed == 0 {
            _ = cache.update(
                keys: MLXArray.zeros([1, 1, 1, 8]), values: MLXArray.zeros([1, 1, 1, 8]))
            eval(cache)
        }
        let firstToken = max(1, seed)
        let step: @Sendable ([MLXArray]) -> [MLXArray] = { args in
            // Production Gemma builds this explicit-window mask before update.
            let mask = cache.makeMask(n: 1, windowSize: window, returnArray: false)
            let pair = cache.update(keys: MLXArray.zeros([1, 1, 1, 8]), values: args[0])
            let output = MLXFast.scaledDotProductAttention(
                queries: MLXArray.zeros([1, 1, 1, 8]), keys: pair.0,
                values: pair.1, scale: 1, mask: mask)
            guard case .array(let array) = mask else {
                preconditionFailure("Expected fixed-cache mask")
            }
            return [output, array]
        }
        let forward = compiled ? compile(inputs: [cache], outputs: [cache], step) : step
        for token in firstToken ..< 25 {
            let result = forward([MLXArray.full([1, 1, 1, 8], values: MLXArray(Float(token)))])
            eval(result)
            let expectedCount = min(token + 1, window)
            let allowed = result[1].asArray(Bool.self).filter { $0 }.count
            XCTAssertEqual(
                allowed, expectedCount,
                "token=\(token), seed=\(seed), restored=\(restored), compiled=\(compiled)")
            let earliest = max(0, token - window + 1)
            let expectedMean = Float(earliest + token) / 2
            for value in result[0].asArray(Float.self) {
                XCTAssertEqual(
                    value, expectedMean, accuracy: 0.0001,
                    "chronological rolling-window average; token=\(token)")
            }
        }
    }

    private static func runMatrix(compiled: Bool) throws {
        for (width, window, keep) in [(4, 4, 0), (8, 4, 0), (8, 4, 2)] {
            for seed in [0, width - 1, width, 2 * width - 1, 2 * width + 1] {
                for restored in [false, true] {
                    try runExplicitWindow(
                        compiled: compiled, restored: restored, seed: seed,
                        width: width, window: window, keep: keep)
                }
            }
        }
    }

    func testExplicitWindowEagerAcrossWrapAndRestore() throws {
        try MLXMetalTestLock.withLock { try Self.runMatrix(compiled: false) }
    }

    func testExplicitWindowCompiledAcrossWrapAndRestore() throws {
        try MLXMetalTestLock.withLock {
            guard HardwareInfo.isCompiledDecodeSupported else {
                throw XCTSkip("Compiled decode unsupported")
            }
            try Self.runMatrix(compiled: true)
        }
    }
}
