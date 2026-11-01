// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

#if canImport(os)
    import Darwin
    import os
#endif

/// Best-effort diagnostic output shared by runtime libraries and their host.
/// A launcher's disconnected stderr must not turn a successful model operation
/// into an NSFileHandleOperationException or SIGPIPE. Errors still propagate
/// through their original typed paths; this only changes diagnostic output.
public enum RuntimeDiagnostics {
    /// Scoped observation for in-process diagnostics tests. This does not
    /// replace the platform logger or change its privacy. Task-local scope
    /// keeps simultaneous/nested captures independent without redirecting a
    /// process-wide descriptor. Unstructured GCD work is not inherited.
    @TaskLocal internal static var observerForTesting: (@Sendable (Data) -> Void)?

    #if canImport(os)
        private static let logger = Logger(subsystem: "vmlx", category: "RuntimeDiagnostics")
    #endif

    /// `VMLX_DIAGNOSTICS_STDERR=1`: also write every line to stderr with a plain
    /// `write(2)` (benchmarks and terminal runs). Opt-in only: a host whose
    /// stderr may be a disconnected pipe leaves it unset.
    static let mirrorsToStandardError =
        ProcessInfo.processInfo.environment["VMLX_DIAGNOSTICS_STDERR"] == "1"

    public static func write(_ data: Data) {
        guard !data.isEmpty else { return }
        #if canImport(os)
            let savedErrno = errno
            defer { errno = savedErrno }
            observerForTesting?(data)
            // Some optional diagnostic payloads include paths or decoded
            // tokens. Do not make these public unified-log fields.
            logger.info("\(String(decoding: data, as: UTF8.self), privacy: .private)")
            // Benchmarks and terminal runs read the same lines from stderr
            // (`VMLX_DIAGNOSTICS_STDERR=1`); the private unified log stays.
            if mirrorsToStandardError {
                _ = data.withUnsafeBytes { Darwin.write(STDERR_FILENO, $0.baseAddress, $0.count) }
            }
        #else
            observerForTesting?(data)
            FileHandle.standardError.write(data)
        #endif
    }
}
