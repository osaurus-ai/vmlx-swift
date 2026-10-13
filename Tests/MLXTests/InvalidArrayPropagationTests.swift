import Foundation
import MLX
import XCTest

/// A failed `mlx-c` operation returns an empty `mlx_array` (ctx == nil) once a
/// recovering error handler is installed (`withError`, or the process-wide
/// `setErrorHandler` the osaurus app installs). That value reads as ndim 0 /
/// shape [], so Swift-side shape logic used to treat the next subscript as
/// over-rank and trap the process (Sentry APPLE-MACOS-14W, 18H, 2X8, 36B, 1BE,
/// 1ZB, 1EK, 1N7, 2R1, 230). Each test below runs the exact production
/// expression from one of those stacks on such an array and requires it to
/// propagate instead, with the originating error still thrown by `withError`.
class InvalidArrayPropagationTests: XCTestCase {

    /// `[2, 3] @ [4, 2]` is a shape error; under `withError` the result is the
    /// empty array every failed operation hands back.
    private func invalidArray() -> MLXArray {
        matmul(MLXArray.zeros([2, 3]), MLXArray.zeros([4, 2]))
    }

    private func assertOriginatingError(_ error: Error, file: StaticString = #filePath, line: UInt = #line)
    {
        guard case .caught(let message)? = error as? MLXError else {
            XCTFail("expected MLXError.caught, got \(error)", file: file, line: line)
            return
        }
        XCTAssertTrue(
            message.lowercased().contains("matmul"),
            "first error must be the originating matmul failure, got: \(message)", file: file,
            line: line)
    }

    func testInvalidArrayHostConversionPropagatesWithoutRecursing() {
        XCTAssertThrowsError(try withError {
            let bad = invalidArray()
            _ = bad.asArray(Float.self)
        }) { assertOriginatingError($0) }
    }

    func testInvalidScalarHostReadPropagatesWithoutSizeTrap() {
        XCTAssertThrowsError(try withError {
            let bad = invalidArray()
            _ = bad.item(Int.self)
        }) { assertOriginatingError($0) }
    }

    func testInvalidArrayReadsAsRankZero() {
        XCTAssertThrowsError(
            try withError { error in
                let bad = invalidArray()
                XCTAssertFalse(bad.isValid)
                XCTAssertEqual(bad.ndim, 0)
                XCTAssertEqual(bad.shape, [])
                try error.check()
            }
        ) { assertOriginatingError($0) }
    }

    // APPLE-MACOS-14W: TokenIterator.convertToToken `logits[0..., -1, 0...]`
    // (MLXArray+Indexing.swift:696 getItemND over-rank precondition).
    func testSampleRowSubscriptOnInvalidLogitsPropagates() {
        XCTAssertThrowsError(
            try withError { error in
                let logits = invalidArray()
                let row = logits[0..., -1, 0...]
                XCTAssertFalse(row.isValid)
                try error.check()
            }
        ) { assertOriginatingError($0) }
    }

    // APPLE-MACOS-18H / 2R1: `q[.ellipsis, ..<rotaryDim]` and `q[.ellipsis, rotaryDim...]`
    // (MLXArray+Indexing.swift:517 expandEllipsisOperations precondition).
    func testEllipsisSubscriptOnInvalidQueriesPropagates() {
        XCTAssertThrowsError(
            try withError { error in
                let q = invalidArray()
                let rotaryDim = 64
                let qRot = q[.ellipsis, ..<rotaryDim]
                let qPass = q[.ellipsis, rotaryDim...]
                XCTAssertFalse(qRot.isValid)
                XCTAssertFalse(qPass.isValid)
                try error.check()
            }
        ) { assertOriginatingError($0) }
    }

    // APPLE-MACOS-2X8 / 1ZB: GatedDeltaNet `convInput[0..., start ..< end, 0...]`.
    func testConvTailSubscriptOnInvalidInputPropagates() {
        XCTAssertThrowsError(
            try withError { error in
                let convInput = invalidArray()
                let end = convInput.dim(1)
                let start = max(0, end - 3)
                let tail = convInput[0..., start ..< end, 0...]
                XCTAssertFalse(tail.isValid)
                try error.check()
            }
        ) { assertOriginatingError($0) }
    }

    // APPLE-MACOS-36B / 1N7: Qwen3VL vision Attention `split(qkv, parts: 3)` then
    // `parts[0][0, 0..., 0..., 0...]` (Array._checkSubscript on the empty split).
    func testSplitOfInvalidArrayKeepsPositionalPieces() {
        XCTAssertThrowsError(
            try withError { error in
                let qkvStates = invalidArray().reshaped(4, 3, 2, 8).transposed(1, 0, 2, 3)
                let parts = split(qkvStates, parts: 3, axis: 0)
                XCTAssertEqual(parts.count, 3)
                let queries = parts[0][0, 0..., 0..., 0...]
                let keys = parts[1][0, 0..., 0..., 0...]
                let values = parts[2][0, 0..., 0..., 0...]
                XCTAssertFalse(queries.isValid)
                XCTAssertFalse(keys.isValid)
                XCTAssertFalse(values.isValid)

                let sections = split(invalidArray(), indices: [8, 16], axis: -1)
                XCTAssertEqual(sections.count, 3)
                XCTAssertFalse(sections[2].isValid)

                let (first, second) = split(invalidArray(), axis: 0)
                XCTAssertFalse(first.isValid)
                XCTAssertFalse(second.isValid)

                // APPLE-MACOS-2AA: Qwen35 Attention `qProj.reshaped(...).split(parts: 2, axis: -1)[0]`
                let qSplit = invalidArray().reshaped(1, 1, 4, -1).split(parts: 2, axis: -1)
                XCTAssertEqual(qSplit.count, 2)
                XCTAssertFalse(qSplit[0].isValid)
                XCTAssertEqual(invalidArray().split(indices: [2], axis: 0).count, 2)
                let (a, b) = invalidArray().split(axis: 0)
                XCTAssertFalse(a.isValid)
                XCTAssertFalse(b.isValid)
                try error.check()
            }
        ) { assertOriginatingError($0) }
    }

    /// A genuine (valid) split must be untouched by the padding path.
    func testValidSplitUnchanged() {
        let x = MLXArray(0 ..< 24, [3, 8])
        let parts = split(x, parts: 3, axis: 0)
        XCTAssertEqual(parts.count, 3)
        XCTAssertTrue(parts.allSatisfy { $0.isValid && $0.shape == [1, 8] })
        let sections = split(x, indices: [2, 5], axis: -1)
        XCTAssertEqual(sections.map { $0.shape }, [[3, 2], [3, 3], [3, 3]])
    }

    // APPLE-MACOS-1BE: RotatingKVCache.updateInPlace
    // `keys[.ellipsis, offset ..< offset + n, 0...] = newKeys` with an invalid update
    // (MLXArray+Indexing.swift:994 scatterArguments -> :517).
    func testSubscriptSetWithInvalidUpdateInvalidatesDestination() {
        XCTAssertThrowsError(
            try withError { error in
                let keys = MLXArray.zeros([1, 2, 8, 4])
                let newKeys = invalidArray()
                keys[.ellipsis, 2 ..< 4, 0...] = newKeys
                XCTAssertFalse(keys.isValid)

                let dest = MLXArray.zeros([4, 4])
                dest[1 ..< 3, axis: 0] = invalidArray()
                XCTAssertFalse(dest.isValid)
                try error.check()
            }
        ) { assertOriginatingError($0) }
    }

    func testSubscriptSetOnInvalidDestinationStaysInvalid() {
        XCTAssertThrowsError(
            try withError { error in
                let dest = invalidArray()
                dest[0..., 0 ..< 1] = MLXArray.ones([1, 1])
                XCTAssertFalse(dest.isValid)
                try error.check()
            }
        ) { assertOriginatingError($0) }
    }

    // APPLE-MACOS-1EK: single-operation `getItem` slice (`ends[0]` on an empty shape).
    func testSingleSliceAndAxisSubscriptsOnInvalidArrayPropagate() {
        XCTAssertThrowsError(
            try withError { error in
                let x = invalidArray()
                XCTAssertFalse(x[1 ..< 2].isValid)
                XCTAssertFalse(x[1].isValid)
                XCTAssertFalse(x[1 ..< 2, axis: 1].isValid)
                XCTAssertFalse(x[.ellipsis].isValid)
                XCTAssertFalse(x[.newAxis].isValid)
                try error.check()
            }
        ) { assertOriginatingError($0) }
    }

    /// The propagation path must not change results for valid arrays: the
    /// existing over-rank preconditions and normal indexing stay as they are.
    func testValidIndexingUnchanged() {
        let logits = MLXArray(0 ..< 24).asType(.float32).reshaped(1, 2, 12)
        let row = logits[0..., -1, 0...]
        XCTAssertEqual(row.shape, [1, 12])
        XCTAssertEqual(row[0, 0].item(Float.self), 12)

        let q = MLXArray(0 ..< 32).reshaped(1, 2, 2, 8)
        XCTAssertEqual(q[.ellipsis, ..<4].shape, [1, 2, 2, 4])
        XCTAssertEqual(q[.ellipsis, 4...].shape, [1, 2, 2, 4])

        let keys = MLXArray.zeros([1, 2, 8, 4])
        keys[.ellipsis, 2 ..< 4, 0...] = MLXArray.ones([1, 2, 2, 4])
        XCTAssertEqual(keys.sum().item(Float.self), 16)
    }
}
