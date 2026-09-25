// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

// Linux stand-ins for Glibc declarations that Swift 6 rejects where Apple code uses them. Compiled
// only where Glibc exists, so Apple builds never see these declarations.

#if canImport(Glibc)
    import Glibc

    /// Glibc declares `stderr` as a mutable C global, which Swift 6 rejects as shared mutable
    /// state. This constant shadows it inside MLXLMCommon, so Apple's `fputs(…, stderr)` call sites
    /// compile unchanged on Linux. Reading Glibc's `stderr` to initialize it is rejected the same
    /// way, so it opens its own stream on descriptor 2, unbuffered like `stderr`, and the lines
    /// reach the same place in the same order. With descriptor 2 closed, `fdopen` fails, and the
    /// lines go to /dev/null: dropped, as a closed `stderr` drops them.
    nonisolated(unsafe) let stderr: UnsafeMutablePointer<FILE> = {
        guard let stream = fdopen(STDERR_FILENO, "w") ?? fopen("/dev/null", "w") else {
            fatalError("GlibcCompat: cannot open a stream for standard error")
        }
        setvbuf(stream, nil, _IONBF, 0)
        return stream
    }()
#endif
