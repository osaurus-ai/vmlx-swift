import Dispatch
import XCTest

@testable import MLX

/// Run with ThreadSanitizer to check concurrent publication/read synchronization.
final class ErrorBoxConcurrencyTests: XCTestCase {
    private enum ProbeError: Error, Equatable { case first, later }

    func testConcurrentCheckAndErrorPublication() {
        let boxes = (0 ..< 10_000).map { _ in ErrorBox() }
        DispatchQueue.concurrentPerform(iterations: 2) { worker in
            if worker == 0 {
                for box in boxes { box.firstError = ProbeError.first }
            } else {
                for box in boxes {
                    do {
                        try box.check()
                    } catch {
                        XCTAssertEqual(error as? ProbeError, .first)
                    }
                }
            }
        }
        // After both workers join, every publication must be observable.
        for box in boxes {
            XCTAssertThrowsError(try box.check()) { error in
                XCTAssertEqual(error as? ProbeError, .first)
            }
        }
    }

    func testCheckPreservesFirstErrorAndDoesNotResetIt() throws {
        let box = ErrorBox()
        try box.check()
        box.firstError = ProbeError.first
        box.firstError = ProbeError.later
        box.firstError = nil
        for _ in 0 ..< 2 {
            XCTAssertThrowsError(try box.check()) { error in
                XCTAssertEqual(error as? ProbeError, .first)
            }
        }
        XCTAssertEqual(box.firstError as? ProbeError, .first)
    }
}
