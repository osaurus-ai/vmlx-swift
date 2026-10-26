import XCTest
@testable import MLX

/// Controlled dispatch through the production MLX error chain. These tests
/// neither submit a failing Metal command buffer nor reproduce GPU error0e.
final class MetalFailureErrorEpochTests: XCTestCase {
    private let signature = "[METAL] Command buffer execution failed: Impacting Interactivity (0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)."

    func testFirstInteractivityErrorSurvivesSecondaryDiagnosticAndRepeatedChecks() {
        let before = MLXErrorMonitor.sequence()
        XCTAssertThrowsError(try withError { box in
            reportMLXError(signature)
            reportMLXError("controlled secondary invalid-array diagnostic")
            for _ in 0 ..< 2 {
                XCTAssertThrowsError(try box.check()) { error in
                    XCTAssertEqual(error as? MLXError, .caught(self.signature))
                }
            }
            // A caller cannot accidentally erase the originating failure.
            box.firstError = nil
        }) { error in
            XCTAssertEqual(error as? MLXError, .caught(self.signature))
        }
        XCTAssertGreaterThanOrEqual(MLXErrorMonitor.sequence(), before + 2)
        XCTAssertNotNil(MLXErrorMonitor.message(since: before))
    }

    func testFreshScopeDoesNotConsumeHistoricalProcessErrorAsItsOwnFailure() throws {
        let before = MLXErrorMonitor.sequence()
        XCTAssertThrowsError(try withError { reportMLXError(signature) })
        let afterFailure = MLXErrorMonitor.sequence()
        XCTAssertGreaterThan(afterFailure, before)
        XCTAssertNotNil(MLXErrorMonitor.message(since: before))

        // The monitor is deliberately process-wide: a historical string is
        // not authority to fail an operation with a new scoped collector.
        let value = try withError { box in
            try box.check()
            return 42
        }
        XCTAssertEqual(value, 42)

        let freshEpoch = MLXErrorMonitor.sequence()
        let freshMessage = "controlled independent next-operation diagnostic"
        XCTAssertThrowsError(try withError { reportMLXError(freshMessage) }) { error in
            XCTAssertEqual(error as? MLXError, .caught(freshMessage))
            XCTAssertNotEqual(error as? MLXError, .caught(self.signature))
        }
        XCTAssertGreaterThan(MLXErrorMonitor.sequence(), freshEpoch)
    }

    func testRecoveredNestedErrorDoesNotPoisonOuterCollector() throws {
        let result = try withError { outer in
            XCTAssertThrowsError(try withError { reportMLXError(signature) }) { error in
                XCTAssertEqual(error as? MLXError, .caught(self.signature))
            }
            try outer.check()
            return "healthy outer operation"
        }
        XCTAssertEqual(result, "healthy outer operation")
    }
}
