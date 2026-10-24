import Foundation
import XCTest
#if canImport(Darwin)
    import Darwin
#endif

@testable import MLXLMCommon

final class ModelResidencyTests: XCTestCase {
    private final class Backend: @unchecked Sendable {
        var limits: [Int] = []
        var succeed = true
        func apply(_ limit: Int) -> Bool {
            limits.append(limit)
            return succeed
        }
    }

    private func request(_ gib: Int, enabled: Bool = true, ram: Int = 128) -> ModelResidency.Request {
        .init(weightBytes: gib << 30, enabled: enabled, physicalMemory: ram << 30,
              workingSet: (ram * 9 / 10) << 30, systemWireLimit: (ram * 85 / 100) << 30)
    }

    func testSecondModelAndUnloadRestoreSurvivorWithoutZeroingIt() {
        let backend = Backend()
        let registry = ModelResidency(apply: backend.apply)
        var a: ModelResidency.Lease? = registry.acquire(request(28)).lease
        let original = backend.limits.last!
        var b: ModelResidency.Lease? = registry.acquire(request(2)).lease
        XCTAssertGreaterThan(backend.limits.last!, original)
        XCTAssertFalse(backend.limits.contains(0))
        withExtendedLifetime(b) {}
        b = nil
        XCTAssertEqual(backend.limits.last, original)
        withExtendedLifetime(a) {}
        a = nil
        XCTAssertEqual(backend.limits.last, 0)
    }

    func testOffSecondModelDoesNotUnpinLiveEnabledModel() {
        let backend = Backend()
        let registry = ModelResidency(apply: backend.apply)
        var a: ModelResidency.Lease? = registry.acquire(request(28)).lease
        let original = backend.limits.last!
        let b = registry.acquire(request(2, enabled: false)).lease
        XCTAssertGreaterThanOrEqual(backend.limits.last!, original)
        withExtendedLifetime(a) {}
        a = nil
        XCTAssertEqual(backend.limits.last, 0)
        withExtendedLifetime(b) {}
    }

    func testDuplicateContextsShareLeaseAndIndependentLoadsCountTwice() {
        let backend = Backend()
        let registry = ModelResidency(apply: backend.apply)
        var a: ModelResidency.Lease? = registry.acquire(request(10)).lease
        var copy = a
        let original = backend.limits.last!
        a = nil
        XCTAssertEqual(backend.limits.last, original)
        var independent: ModelResidency.Lease? = registry.acquire(request(10)).lease
        XCTAssertEqual(backend.limits.last, 36 << 30)
        withExtendedLifetime(independent) {}
        independent = nil
        XCTAssertEqual(backend.limits.last, original)
        withExtendedLifetime(copy) {}
        copy = nil
        XCTAssertEqual(backend.limits.last, 0)
    }

    func testFailedLoadScopeReleasesReservation() {
        enum Failure: Error { case load }
        let backend = Backend()
        let registry = ModelResidency(apply: backend.apply)
        let a = registry.acquire(request(28)).lease
        let original = backend.limits.last!
        func fail() throws {
            let pending = registry.acquire(request(2)).lease
            defer { withExtendedLifetime(pending) {} }
            throw Failure.load
        }
        XCTAssertThrowsError(try fail())
        XCTAssertEqual(backend.limits.last, original)
        withExtendedLifetime(a) {}
    }

    func testNonFittingSecondModelDoesNotClearSmallMacSurvivor() {
        let a = request(2, ram: 32)
        let b = request(30, ram: 32)
        XCTAssertEqual(ModelResidency.target(for: [a, b]), ModelResidency.target(for: [a]))
        XCTAssertGreaterThan(ModelResidency.target(for: [a]), 0)
        XCTAssertEqual(ModelResidency.target(for: [request(28, enabled: false)]), 0)
    }

    func testFailedBackendApplicationIsRetriedOnNextTransition() {
        let backend = Backend()
        backend.succeed = false
        let registry = ModelResidency(apply: backend.apply)
        let a = registry.acquire(request(28))
        XCTAssertFalse(a.applied)
        backend.succeed = true
        let b = registry.acquire(request(2))
        XCTAssertTrue(b.applied)
        XCTAssertEqual(backend.limits.count, 2)
        withExtendedLifetime((a, b)) {}
    }

    #if canImport(Darwin)
        /// The legacy FileHandle write raises an Objective-C exception (or SIGPIPE),
        /// so exercise the production registry in an isolated XCTest subprocess.
        func testResidencySurvivesDisconnectedDiagnosticSinks() throws {
            for mode in ["broken-default", "broken-ignored", "closed"] {
                let child = Process()
                child.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
                child.arguments = [
                    "xctest", "-XCTest",
                    "MLXLMCommonFocusedTests.ModelResidencyTests/testDiagnosticSinkChild",
                    Bundle(for: Self.self).bundleURL.path,
                ]
                var environment = ProcessInfo.processInfo.environment
                environment["VMLX_RESIDENCY_DIAGNOSTIC_TEST"] = mode
                child.environment = environment
                child.standardOutput = FileHandle.nullDevice
                child.standardError = FileHandle.nullDevice
                try child.run()
                child.waitUntilExit()
                XCTAssertEqual(child.terminationReason, .exit, mode)
                XCTAssertEqual(child.terminationStatus, 0, mode)
            }
        }

        func testDiagnosticSinkChild() throws {
            guard let mode = ProcessInfo.processInfo.environment["VMLX_RESIDENCY_DIAGNOSTIC_TEST"]
            else { return }
            let saved = dup(STDERR_FILENO)
            XCTAssertGreaterThanOrEqual(saved, 0)
            defer { _ = dup2(saved, STDERR_FILENO); close(saved) }
            if mode == "closed" {
                close(STDERR_FILENO)
            } else {
                var descriptors: [Int32] = [0, 0]
                XCTAssertEqual(pipe(&descriptors), 0)
                close(descriptors[0])
                XCTAssertEqual(dup2(descriptors[1], STDERR_FILENO), STDERR_FILENO)
                close(descriptors[1])
                signal(SIGPIPE, mode == "broken-default" ? SIG_DFL : SIG_IGN)
            }
            let backend = Backend()
            let registry = ModelResidency(apply: backend.apply)
            var lease: ModelResidency.Lease? = registry.acquire(request(2)).lease
            withExtendedLifetime(lease) {}
            lease = nil
            RuntimeDiagnostics.write(Data("[Runtime] disconnected diagnostic sink\n".utf8))
            // The healer's default diagnostic used C stdio rather than FileHandle;
            // that path terminated with SIGPIPE even though no repair was enabled.
            let repair = SafetensorsStorageHealer.healBundle(
                at: URL(fileURLWithPath: "/unused-healer-diagnostic-probe"),
                configuration: .init(environment: [:]))
            XCTAssertEqual(repair.scannedShards, 0)
            // Both acquire and deinit must finish even when diagnostic output cannot.
            _ = dup2(saved, STDERR_FILENO)
            XCTAssertEqual(backend.limits.count, 2)
            XCTAssertEqual(backend.limits.last, 0)
        }
    #endif
}
