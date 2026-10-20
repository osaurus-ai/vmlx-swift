import XCTest
@testable import MLXLMCommon

final class RoutedExpertVerificationScopeTests: XCTestCase {
    func testOnlyActualTargetRowsUseDecodeAndScopeRestores() throws {
        func admitted(_ shape: [Int], _ routes: Int = 8) -> Bool {
            RoutedExpertVerificationScope.usesMappedDecode(inputShape: shape, routes: routes)
        }
        XCTAssertFalse(admitted([1, 8, 3072]))
        RoutedExpertVerificationScope.withVerification(inputShape: [1, 8]) {
            XCTAssertTrue(admitted([1, 8, 3072]))
            XCTAssertFalse(admitted([1, 4, 3072]))
            XCTAssertFalse(admitted([2, 8, 3072]))
            XCTAssertFalse(admitted([1, 8, 3072], 64))
            RoutedExpertVerificationScope.withVerification(inputShape: [1, 1]) {
                XCTAssertFalse(admitted([1, 8, 3072]))
            }
            XCTAssertTrue(admitted([1, 8, 3072]))
        }
        XCTAssertFalse(admitted([1, 8, 3072]))
        enum Failure: Error { case expected }
        XCTAssertThrowsError(try RoutedExpertVerificationScope.withVerification(inputShape: [1, 8]) {
            throw Failure.expected
        })
        XCTAssertFalse(admitted([1, 8, 3072]))
        for shape in [[1, 1], [1, 9], [2, 8], [8], [1, 8, 3072]] {
            RoutedExpertVerificationScope.withVerification(inputShape: shape) {
                XCTAssertFalse(admitted([1, 8, 3072]))
            }
        }
    }
}
