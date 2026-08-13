import Foundation
import MLX
import XCTest

/// Reproduces the SIGTRAP seen live in `computeMediaSalt` -> `hashMLXArray`
/// -> `MLXArray.asData(access:)`: the backing data pointer is null for an
/// array with no elements, and `asData` force-unwraps it.
class EmptyArrayDataTests: XCTestCase {
    func testAsDataOnZeroElementArray() {
        let empty = MLXArray(Array<Float>(), [0])
        XCTAssertEqual(empty.size, 0)
        let d = empty.asData(access: .noCopyIfContiguous)
        XCTAssertEqual(d.data.count, 0)
    }

    func testAsDataNoCopyOnZeroElementArray() {
        let empty = MLXArray(Array<Float>(), [0])
        let d = empty.asData(access: .noCopy)
        XCTAssertEqual(d.data.count, 0)
    }

    func testAsDataStillReturnsBytesForANonEmptyArray() {
        let a = MLXArray([Float(1), 2, 3], [3])
        let d = a.asData(access: .noCopyIfContiguous)
        XCTAssertEqual(d.data.count, 3 * MemoryLayout<Float>.size)
        XCTAssertEqual(a.asData(access: .copy).data.count, 3 * MemoryLayout<Float>.size)
    }

    func testAsDataOnZeroElementArrayWithNonZeroRank() {
        let empty = MLXArray(Array<Float>(), [1, 0, 3, 448])
        XCTAssertEqual(empty.size, 0)
        let d = empty.asData(access: .noCopyIfContiguous)
        XCTAssertEqual(d.data.count, 0)
        let c = empty.asData(access: .copy)
        XCTAssertEqual(c.data.count, 0)
    }
}
