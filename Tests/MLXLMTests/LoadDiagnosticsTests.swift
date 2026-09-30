// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import XCTest

@testable import MLXLMCommon

#if canImport(Darwin)
    import Darwin

    final class LoadDiagnosticsTests: XCTestCase {
        func testDiagnosticsDoNotChangeDescriptorFlagsOrSignalMask() {
            let flags = fcntl(STDERR_FILENO, F_GETNOSIGPIPE)
            var beforeMask = sigset_t()
            var afterMask = sigset_t()
            XCTAssertEqual(pthread_sigmask(SIG_BLOCK, nil, &beforeMask), 0)
            LoadDiagnostics.write(Data("[Load] diagnostic sink regression\n".utf8))
            XCTAssertEqual(pthread_sigmask(SIG_BLOCK, nil, &afterMask), 0)
            XCTAssertEqual(beforeMask, afterMask)
            XCTAssertEqual(fcntl(STDERR_FILENO, F_GETNOSIGPIPE), flags)
        }

        func testConcurrentDiagnosticsLeaveThreadMasksUnchanged() {
            DispatchQueue.concurrentPerform(iterations: 16) { index in
                var beforeMask = sigset_t()
                var afterMask = sigset_t()
                XCTAssertEqual(pthread_sigmask(SIG_BLOCK, nil, &beforeMask), 0)
                LoadDiagnostics.write(Data("[Load] concurrent sink regression \(index)\n".utf8))
                XCTAssertEqual(pthread_sigmask(SIG_BLOCK, nil, &afterMask), 0)
                XCTAssertEqual(beforeMask, afterMask)
            }
        }
    }
#endif
