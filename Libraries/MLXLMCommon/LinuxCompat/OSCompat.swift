// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

// Linux stand-ins for the parts of Apple's `os` module that the LM libraries use. Compiled only
// where `os` does not exist, so Apple builds never see these declarations.

#if !canImport(os)
    import Foundation

    /// Stand-in for `os.OSAllocatedUnfairLock`, backed by `NSLock`.
    package final class OSAllocatedUnfairLock<State>: @unchecked Sendable {
        private let mutex = NSLock()
        private var state: State

        package init(initialState: State) {
            self.state = initialState
        }

        package func withLock<R>(_ body: (inout State) throws -> R) rethrows -> R {
            mutex.lock()
            defer { mutex.unlock() }
            return try body(&state)
        }

        package func withLockUnchecked<R>(_ body: (inout State) throws -> R) rethrows -> R {
            try withLock(body)
        }

        package func lock() { mutex.lock() }
        package func unlock() { mutex.unlock() }
    }

    extension OSAllocatedUnfairLock where State == Void {
        package convenience init() {
            self.init(initialState: ())
        }

        package func withLock<R>(_ body: () throws -> R) rethrows -> R {
            lock()
            defer { unlock() }
            return try body()
        }

        package func withLockUnchecked<R>(_ body: () throws -> R) rethrows -> R {
            try withLock(body)
        }
    }

    package enum OSLogPrivacy: Sendable {
        case auto, `public`, `private`, sensitive
    }

    /// Accepts the `\(value, privacy: .public)` interpolations the call sites use.
    package struct OSLogMessage: ExpressibleByStringInterpolation, Sendable {
        package let text: String

        package init(stringLiteral value: String) {
            text = value
        }

        package init(stringInterpolation: StringInterpolation) {
            text = stringInterpolation.buffer
        }

        package struct StringInterpolation: StringInterpolationProtocol {
            var buffer = ""

            package init(literalCapacity: Int, interpolationCount: Int) {
                buffer.reserveCapacity(literalCapacity)
            }

            package mutating func appendLiteral(_ literal: String) {
                buffer += literal
            }

            package mutating func appendInterpolation<T>(_ value: T, privacy: OSLogPrivacy = .auto)
            {
                buffer += String(describing: value)
            }
        }
    }

    /// Stand-in for `os.Logger`: warnings and errors go to stderr; the lower levels only when the
    /// `VMLX_LINUX_LOG` flag is on (`1`, `true`, `yes` or `on`; `0`, `false` and `off` mean off).
    ///
    /// `privacy:` is ignored: Linux prints values Apple would redact. No call site compiled on
    /// Linux marks a value `.private` or `.sensitive` today, but some leave strings at the default
    /// `.auto`, which Apple redacts.
    package struct Logger: Sendable {
        private let label: String
        private static let verbose = RuntimeEnvironment.flag("VMLX_LINUX_LOG")

        package init(subsystem: String, category: String) {
            label = "\(subsystem).\(category)"
        }

        package init() {
            label = "default"
        }

        private func emit(_ level: String, _ message: OSLogMessage, always: Bool) {
            guard always || Self.verbose else { return }
            FileHandle.standardError.write(Data("[\(level)] \(label): \(message.text)\n".utf8))
        }

        package func trace(_ message: OSLogMessage) { emit("trace", message, always: false) }
        package func debug(_ message: OSLogMessage) { emit("debug", message, always: false) }
        package func info(_ message: OSLogMessage) { emit("info", message, always: false) }
        package func notice(_ message: OSLogMessage) { emit("notice", message, always: false) }
        package func log(_ message: OSLogMessage) { emit("log", message, always: false) }
        package func warning(_ message: OSLogMessage) { emit("warning", message, always: true) }
        package func error(_ message: OSLogMessage) { emit("error", message, always: true) }
        package func fault(_ message: OSLogMessage) { emit("fault", message, always: true) }
        package func critical(_ message: OSLogMessage) { emit("critical", message, always: true) }
    }
#endif
