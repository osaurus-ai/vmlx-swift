import Foundation
import MLX
import XCTest
@testable import MLXLMCommon

final class DFlash2PrefillHiddenCaptureTests: XCTestCase {
    func testExactRowsAcrossUnevenBoundariesAndDTypes() {
        for dtype in [DType.float32, .float16, .bfloat16] {
            for limit in [nil, 0, 5, 64] as [Int?] {
                var capture = DFlash2PrefillHiddenCapture(limit: limit)
                var baseline: MLXArray?
                var start = 0
                for rows in [3, 1, 17, 2, 1] {
                    let chunk = MLXArray(Array(start..<(start + rows * 8)))
                        .reshaped(1, rows, 8).asType(dtype)
                    start += rows * 8
                    let combined = baseline.map { concatenated([$0, chunk], axis: 1) } ?? chunk
                    if let limit, combined.dim(1) > limit {
                        baseline = combined[0..., (combined.dim(1) - limit)..., 0...]
                    } else { baseline = combined }
                    eval(capture.append(chunk), baseline!)
                }
                let actual = capture.finish()!
                eval(actual)
                XCTAssertEqual(actual.dtype, dtype)
                XCTAssertEqual(actual.shape, baseline!.shape)
                XCTAssertTrue(all(actual .== baseline!).item(Bool.self))
            }
        }
    }

    func testEmptyAndSingleChunk() {
        var empty = DFlash2PrefillHiddenCapture(limit: nil)
        XCTAssertNil(empty.finish())
        let chunk = MLXArray([Float(1), -2, 3, 4]).reshaped(1, 1, 4)
        eval(empty.append(chunk))
        let actual = empty.finish()!
        XCTAssertEqual(actual.shape, chunk.shape)
        XCTAssertTrue(all(actual .== chunk).item(Bool.self))
    }

    func testBoundedCopyTimingProbe() throws {
        guard ProcessInfo.processInfo.environment["VMLX_PREFILL_CAPTURE_PROBE"] == "1" else {
            throw XCTSkip("Opt-in component timing; not model/app throughput proof")
        }
        let rows = 27219
        let width = 10240
        let chunks = stride(from: 0, to: rows, by: 512).map { start in
            let count = min(512, rows - start)
            return sin(MLXArray(0..<(count * width)).asType(.float32) + Float(start))
                .asType(.bfloat16).reshaped(1, count, width)
        }
        for chunk in chunks {
            eval(chunk)
            XCTAssertEqual(chunk.strides.last, 1)
            XCTAssertEqual(chunk.strides[1], width)
        }
        print("CAPTURE_INPUT materialized_sine active_bytes=\(Memory.activeMemory) chunks=\(chunks.count)")
        for iteration in 0..<3 {
            let oldActive = Memory.activeMemory
            Memory.peakMemory = 0
            let oldStart = Date.timeIntervalSinceReferenceDate
            var old: MLXArray?
            for chunk in chunks {
                old = old.map { concatenated([$0, chunk], axis: 1) } ?? chunk
                eval(old!)
            }
            let oldSeconds = Date.timeIntervalSinceReferenceDate - oldStart
            let oldPeakDelta = Memory.peakMemory - oldActive
            let newActive = Memory.activeMemory
            Memory.peakMemory = 0
            let newStart = Date.timeIntervalSinceReferenceDate
            var capture = DFlash2PrefillHiddenCapture(limit: nil)
            for chunk in chunks { eval(capture.append(chunk)) }
            let actual = capture.finish()!
            eval(actual)
            let newSeconds = Date.timeIntervalSinceReferenceDate - newStart
            let newPeakDelta = Memory.peakMemory - newActive
            XCTAssertTrue(all(actual .== old!).item(Bool.self))
            print("CAPTURE_PROBE iteration=\(iteration) rows=\(rows) width=\(width) old_ms=\(oldSeconds * 1000) new_ms=\(newSeconds * 1000) old_peak_delta=\(oldPeakDelta) new_peak_delta=\(newPeakDelta) exact=true")
        }
    }
}
