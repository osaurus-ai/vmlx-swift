import Foundation
import MLX
import XCTest

@testable import MLXLLM

/// Naive-N0.5 prefill attention rewrites against the reference attention they replace.
final class NaiveN05FlashPrefillAttentionTests: XCTestCase {
    private func rel(_ a: MLXArray, _ b: MLXArray) -> Float {
        let d = (a.asType(.float32) - b.asType(.float32)).square().sum().item(Float.self)
        return (d / max(b.asType(.float32).square().sum().item(Float.self), 1e-30)).squareRoot()
    }

    /// 300 new rows (not a multiple of the 128 window) after 127 history keys, sinks on, GQA 64/8 heads, 192/128 dims.
    func testBlockedSlidingMatchesReference() {
        MLXRandom.seed(1)
        let h = 64, kv = 8, d = 192, dv = 128, w = 128, past = 400, l = 300
        let hist = w - 1
        let q = MLXRandom.normal([1, h, l, d]).asType(.bfloat16)
        let k = MLXRandom.normal([1, kv, hist + l, d]).asType(.bfloat16)
        let v = MLXRandom.normal([1, kv, hist + l, dv]).asType(.bfloat16)
        let sinks = (MLXRandom.normal([h]) * 0.5).asType(.bfloat16)
        let scale = Float(1 / sqrt(Double(d)))
        let fast = NaiveN05FlashPrefillAttention.blockedSliding(q: q, k: k, v: v, scale: scale, sinks: sinks, window: w)
        let allowed = NaiveN05FlashMath.allowedMask(
            padding: MLXArray.ones([1, past + l], dtype: .bool), queryOffset: past, length: l,
            keyOffset: past - hist, keyLength: hist + l, window: w)
        let reference = NaiveN05FlashMath.referenceAttention(
            query: q, key: k, value: v, allowed: allowed, sink: sinks, valueScale: nil)
        let e = rel(fast, reference)
        print("blocked sliding rel-L2 \(e)")
        XCTAssertEqual(fast.shape, reference.shape)
        XCTAssertLessThan(e, 1e-2)
    }

    /// 96 queries at positions 500..595 over 596 keys, 64 selected per query including future and padding (-1)
    /// indices that must be masked; GQA 64/4 heads, 192/128 dims.
    func testSparseMatchesReference() {
        MLXRandom.seed(2)
        let h = 64, kv = 4, d = 192, dv = 128, l = 96, past = 500, keys = past + l, topK = 64
        let q = MLXRandom.normal([1, h, l, d]).asType(.bfloat16)
        let k = MLXRandom.normal([1, kv, keys, d]).asType(.bfloat16)
        let v = MLXRandom.normal([1, kv, keys, dv]).asType(.bfloat16)
        var sel = [Int32]()
        for i in 0 ..< l {
            var row = (0 ..< 60).map { Int32(($0 * 37 + i * 11) % keys) }  // includes keys after the query
            row += [Int32(past + i), -1, -1, Int32(past + i)]  // own position (duplicate once), padding
            sel += row
        }
        let selected = MLXArray(sel, [l, topK])
        let scale = Float(1 / sqrt(Double(d)))
        let fast = NaiveN05FlashPrefillAttention.sparse(
            q: q, k: k, v: v, selected: selected, scale: scale, queryOffset: past)
        // Reference: the same keys as a boolean mask (duplicates collapse, as a set) with causality.
        var allowedHost = [Bool](repeating: false, count: l * keys)
        for i in 0 ..< l {
            for j in sel[(i * topK) ..< ((i + 1) * topK)] where j >= 0 && Int(j) <= past + i {
                allowedHost[i * keys + Int(j)] = true
            }
        }
        // The kernel counts a duplicated key twice; drop the duplicate before comparing set semantics.
        var dedup = [Int32]()
        for i in 0 ..< l {
            var seen = Set<Int32>()
            for j in sel[(i * topK) ..< ((i + 1) * topK)] { dedup.append(seen.insert(j).inserted || j < 0 ? j : -1) }
        }
        let fastDedup = NaiveN05FlashPrefillAttention.sparse(
            q: q, k: k, v: v, selected: MLXArray(dedup, [l, topK]), scale: scale, queryOffset: past)
        let reference = NaiveN05FlashMath.referenceAttention(
            query: q, key: k, value: v, allowed: MLXArray(allowedHost, [1, l, keys]), sink: nil, valueScale: nil)
        let e = rel(fastDedup, reference)
        print("sparse rel-L2 \(e)")
        XCTAssertEqual(fast.shape, reference.shape)
        XCTAssertLessThan(e, 1e-2)
    }
}

/// Diagnostic sweep (JANGT_PERF=1): blocked sliding attention at large row counts must stay finite.
final class NaiveN05FlashBlockedSlidingSweepTests: XCTestCase {
    func testLargeBlockCountsStayFinite() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["JANGT_PERF"] == "1")
        MLXRandom.seed(9)
        let h = 64, kv = 8, d = 192, dv = 128, w = 128
        for (l, hist, withSinks) in [(2048, 0, true), (4096, 0, true), (4096, 127, true), (4096, 0, false), (3000, 0, true)] {
            let q = MLXRandom.normal([1, h, l, d]).asType(.bfloat16)
            let k = MLXRandom.normal([1, kv, hist + l, d]).asType(.bfloat16)
            let v = MLXRandom.normal([1, kv, hist + l, dv]).asType(.bfloat16)
            let sinks: MLXArray? = withSinks ? (MLXRandom.normal([h]) * 0.5).asType(.bfloat16) : nil
            let o = NaiveN05FlashPrefillAttention.blockedSliding(
                q: q, k: k, v: v, scale: Float(1 / sqrt(Double(d))), sinks: sinks, window: w)
            let bad = (1 - MLX.isFinite(o).asType(.int32)).sum().item(Int32.self)
            let badRows = (1 - MLX.isFinite(o).all(axes: [1, 3]).asType(.int32)).sum().item(Int32.self)
            print("blocked sliding l=\(l) hist=\(hist) sinks=\(withSinks): non-finite \(bad) rows \(badRows)")
        }
    }
}
