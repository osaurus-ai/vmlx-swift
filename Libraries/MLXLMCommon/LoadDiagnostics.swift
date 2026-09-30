// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

#if canImport(os)
    import os
    import Darwin
#endif

/// Weight-loader diagnostics must not abort a load when a launcher's stderr
/// pipe closes. This helper handles the diagnostic sink only, never model errors.
enum LoadDiagnostics {
    #if canImport(os)
        private static let logger = Logger(subsystem: "vmlx", category: "LoadWeights")
    #endif

    static func write(_ data: Data) {
        guard !data.isEmpty else { return }
        #if canImport(os)
            let savedErrno = errno
            defer { errno = savedErrno }
            // Match the existing vmlx unified-log convention. These messages
            // contain loader diagnostics, not prompts or generated content.
            // Do not touch stderr, its descriptor flags, or signal handling.
            logger.info("\(String(decoding: data, as: UTF8.self), privacy: .public)")
        #else
            // Preserve the existing non-Darwin diagnostic behavior.
            FileHandle.standardError.write(data)
        #endif
    }
}
