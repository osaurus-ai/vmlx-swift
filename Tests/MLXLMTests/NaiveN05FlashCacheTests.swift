import Foundation
import MLX
import XCTest

@testable import MLXLLM

/// The full-attention Naive cache writes into capacity buffers in place. Against a concatenating reference: every
/// returned K / V / indexer view equals the concatenation, appends cross the 256-row step at sizes that do not divide
/// it, a copy keeps its rows while the original keeps appending, trim then append, and a state round trip.
final class NaiveN05FlashCacheTests: XCTestCase {
    private func chunk(_ n: Int, seed: UInt64) -> (MLXArray, MLXArray, MLXArray) {
        let key = MLXRandom.key(seed)
        let k = MLXRandom.normal([1, 4, n, 8], key: key).asType(.bfloat16)
        let v = MLXRandom.normal([1, 4, n, 6], key: MLXRandom.key(seed + 1000)).asType(.bfloat16)
        let ix = MLXRandom.normal([1, 1, n, 5], key: MLXRandom.key(seed + 2000))
        return (k, v, ix)
    }

    private func same(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.shape == b.shape && abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self) == 0
    }

    func testInPlaceAppendMatchesConcatenation() throws {
        let cache = NaiveN05FlashCache(window: nil, requiresIndexer: true)
        var ref: [MLXArray] = []
        var copySnapshot: (NaiveN05FlashCache, [MLXArray])?
        for (i, n) in [300, 1, 1, 255, 3, 513, 1, 1, 1, 777].enumerated() {
            let (k, v, ix) = chunk(n, seed: UInt64(i * 10))
            let out = try cache.append(keys: k, values: v, indexer: ix)
            ref = ref.isEmpty ? [k, v, ix] : zip(ref, [k, v, ix]).map { concatenated([$0, $1], axis: 2) }
            XCTAssertTrue(same(out.0, ref[0]) && same(out.1, ref[1]) && same(out.2!, ref[2]), "append \(i)")
            XCTAssertEqual(cache.offset, ref[0].dim(2))
            XCTAssertEqual(cache.keyOffset, 0)
            if i == 4 {
                copySnapshot = (cache.copy() as! NaiveN05FlashCache, ref)
            }
        }
        // The copy taken at append 4 kept its rows through the later in-place writes.
        let (copy, refAtCopy) = try XCTUnwrap(copySnapshot)
        XCTAssertEqual(copy.offset, refAtCopy[0].dim(2))
        XCTAssertTrue(zip(copy.state, refAtCopy).allSatisfy { same($0, $1) })
        // Trim then append.
        XCTAssertEqual(cache.trim(5), 5)
        ref = ref.map { $0[0..., 0..., ..<($0.dim(2) - 5), 0...] }
        XCTAssertTrue(zip(cache.state, ref).allSatisfy { same($0, $1) })
        let (k, v, ix) = chunk(9, seed: 999)
        let out = try cache.append(keys: k, values: v, indexer: ix)
        ref = zip(ref, [k, v, ix]).map { concatenated([$0, $1], axis: 2) }
        XCTAssertTrue(same(out.0, ref[0]) && same(out.2!, ref[2]))
        // State round trip into a fresh cache, then one more append.
        let restored = NaiveN05FlashCache(window: nil, requiresIndexer: true)
        XCTAssertTrue(restored.restoreDiskCacheState(cache.state, metadata: cache.metaState, offset: cache.offset))
        let (k2, v2, ix2) = chunk(2, seed: 4242)
        let a = try restored.append(keys: k2, values: v2, indexer: ix2)
        let b = try cache.append(keys: k2, values: v2, indexer: ix2)
        XCTAssertTrue(same(a.0, b.0) && same(a.1, b.1) && same(a.2!, b.2!))
    }
}
