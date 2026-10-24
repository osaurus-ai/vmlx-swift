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
    #if canImport(os)
        private static let logger = Logger(subsystem: "vmlx", category: "RuntimeDiagnostics")
    #endif

    public static func write(_ data: Data) {
        guard !data.isEmpty else { return }
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
