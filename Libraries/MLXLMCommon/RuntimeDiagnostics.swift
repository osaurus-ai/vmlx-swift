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

    public static func write(_ data: Data) {
        guard !data.isEmpty else { return }
        observerForTesting?(data)
        #if canImport(os)
            let savedErrno = errno
            defer { errno = savedErrno }
            // Some optional diagnostic payloads include paths or decoded
            // tokens. Do not make these public unified-log fields.
            logger.info("\(String(decoding: data, as: UTF8.self), privacy: .private)")
        #else
            FileHandle.standardError.write(data)
        #endif
    }
}
