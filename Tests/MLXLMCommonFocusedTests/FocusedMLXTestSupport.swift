// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Darwin
import os

enum FocusedMLXTestSupport {
    private static let queue = DispatchQueue(label: "ai.osaurus.vmlx.focused-mlx-tests")
    private static let lock = ProcessWideMLXTestLock()

    static func withLock<T>(_ body: () throws -> T) rethrows -> T {
        _ = metallibPrepared
        lock.wait()
        defer { lock.signal() }
        return try body()
    }

    /// Acquires the process-wide MLX lock without blocking a cooperative
    /// thread.
    ///
    /// The previous shape parked a thread on `done.wait()` while a nested
    /// `Task` ran `body()`. That task needs a cooperative-pool thread, and
    /// under pool starvation it never gets one, so the wait never returns and
    /// the lock is never released. Every later `withLock` — including
    /// the SYNCHRONOUS one used by the Hy3 sanitizer tests — then blocks
    /// forever at 0% CPU, hanging the whole test target and holding the
    /// SwiftPM `.build` lock with it.
    ///
    /// Now only the private serial queue's own thread blocks while waiting for
    /// the lock; `body()` is awaited normally by the caller, so no task
    /// is ever waited on from inside a blocked thread.
    static func withLock<T: Sendable>(
        _ body: @Sendable @escaping () async throws -> T
    ) async throws -> T {
        _ = metallibPrepared
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            queue.async {
                lock.wait()
                continuation.resume()
            }
        }
        defer { lock.signal() }
        return try await body()
    }

    private static let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .standardizedFileURL

    private final class BundleProbe {}

    private static let metallibPrepared: Void = {
        let sourceDirectories = [
            repoRoot.appendingPathComponent(".build/arm64-apple-macosx/debug"),
            repoRoot.appendingPathComponent(".build/debug"),
        ]
        guard let sourceDirectory = sourceDirectories.first(where: {
            FileManager.default.fileExists(atPath: $0.appendingPathComponent("default.metallib").path)
        }) else { return }

        let source = sourceDirectory.appendingPathComponent("default.metallib")
        let fileManager = FileManager.default
        var targetDirectories: [URL] = []

        if let executableURL = Bundle.main.executableURL {
            targetDirectories.append(executableURL.deletingLastPathComponent())
        }
        if let resourceURL = Bundle.main.resourceURL {
            targetDirectories.append(resourceURL)
        }
        let testBundle = Bundle(for: BundleProbe.self)
        if let executableURL = testBundle.executableURL {
            targetDirectories.append(executableURL.deletingLastPathComponent())
        }
        if let resourceURL = testBundle.resourceURL {
            targetDirectories.append(resourceURL)
        }
        if let firstArgument = CommandLine.arguments.first, !firstArgument.isEmpty {
            targetDirectories.append(URL(fileURLWithPath: firstArgument).deletingLastPathComponent())
        }
        targetDirectories.append(sourceDirectory)

        var scanned = Set<String>()
        for candidate in targetDirectories {
            var directory = candidate.standardizedFileURL
            for _ in 0..<4 {
                if scanned.insert(directory.path).inserted {
                    try? fileManager.copyFocusedMLXMetallibsIfMissing(from: source, into: directory)
                }
                directory.deleteLastPathComponent()
            }
        }
    }()
}

/// A lock that the kernel releases when its holder dies: `flock(2)` on a file
/// in the user's temporary directory. Every test process of the user opens the
/// same file, so the lock serializes MLX work across SwiftPM test targets too.
///
/// It replaces a named POSIX semaphore, whose count outlived a holder killed
/// inside `withLock`. Every later waiter then blocked, and the stale-lock
/// recovery that reset it raced: two waiters reset it at once, and the loser
/// trapped. A dead process holds no `flock` lock, so there is nothing to
/// recover.
///
/// Waiting has no time limit: a holder that hangs blocks every waiter until
/// it is killed. The first time a waiter finds the lock taken, it says so on
/// standard error, with the path; `lsof <path>` names the holder.
///
/// Each acquisition opens the file anew. A `flock` lock belongs to the open
/// file description, and a second `flock` on a description that holds it
/// returns at once, so the threads of one process sharing a descriptor would
/// all get the lock. The descriptor is close-on-exec, so that a child process
/// never inherits a held lock. A waiter whose file was deleted or replaced
/// while it waited does not keep that file's lock, which no longer excludes
/// a process that opens the path; it opens the path again.
final class ProcessWideMLXTestLock: Sendable {
    /// The lock every test uses. `confstr(_CS_DARWIN_USER_TEMP_DIR)` is the
    /// same for every process of the user. `NSTemporaryDirectory()` is no
    /// fallback: it follows `TMPDIR`, which a test runner may set, and a lock
    /// in another directory would exclude no one.
    static let defaultPath: String = {
        var buffer = [CChar](repeating: 0, count: Int(PATH_MAX))
        let length = confstr(_CS_DARWIN_USER_TEMP_DIR, &buffer, buffer.count)
        let code = errno
        guard length > 0, length <= buffer.count else {
            let reason =
                length == 0 ? ProcessWideMLXTestLock.describe(code) : "it needs \(length) bytes"
            fatalError("No user temporary directory for the MLX Metal test lock: \(reason)")
        }
        let directory = buffer.withUnsafeBufferPointer { String(cString: $0.baseAddress!) }
        return (directory as NSString).appendingPathComponent("vmlx_mlx_lock")
    }()

    let path: String

    /// The holder's descriptor, or -1.
    private let held = OSAllocatedUnfairLock(initialState: Int32(-1))

    /// Whether a waiter has said that the lock was taken.
    private let saidItWaits = OSAllocatedUnfairLock(initialState: false)

    init(path: String = ProcessWideMLXTestLock.defaultPath) {
        self.path = path
    }

    func wait() {
        while true {
            let descriptor = open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0o600)
            guard descriptor >= 0 else {
                let code = errno
                fatalError("Unable to open the MLX Metal test lock \(path): \(Self.describe(code))")
            }
            if !take(descriptor, LOCK_EX | LOCK_NB) {
                sayItWaits()
                _ = take(descriptor, LOCK_EX)
            }
            if isTheFileAtPath(descriptor) {
                held.withLock { $0 = descriptor }
                return
            }
            close(descriptor)
        }
    }

    func signal() {
        let descriptor = held.withLock { held in
            defer { held = -1 }
            return held
        }
        guard descriptor >= 0 else {
            fatalError("The MLX Metal test lock \(path) was released without being held")
        }
        // Closing the only descriptor releases the lock as well; unlocking
        // first says so.
        flock(descriptor, LOCK_UN)
        close(descriptor)
    }

    /// Locks `descriptor` with `operation`; false if `LOCK_NB` finds the lock
    /// taken.
    private func take(_ descriptor: Int32, _ operation: Int32) -> Bool {
        while flock(descriptor, operation) != 0 {
            let code = errno
            if code == EWOULDBLOCK { return false }
            guard code == EINTR else {
                fatalError("Unable to take the MLX Metal test lock \(path): \(Self.describe(code))")
            }
        }
        return true
    }

    /// Whether `descriptor` is still the file at `path`. A file deleted or
    /// replaced after it was opened does not exclude a process that opens the
    /// path now.
    private func isTheFileAtPath(_ descriptor: Int32) -> Bool {
        var locked = stat()
        guard fstat(descriptor, &locked) == 0 else {
            let code = errno
            fatalError("Unable to inspect the MLX Metal test lock \(path): \(Self.describe(code))")
        }
        var current = stat()
        guard stat(path, &current) == 0 else {
            let code = errno
            guard code == ENOENT else {
                fatalError(
                    "Unable to inspect the MLX Metal test lock \(path): \(Self.describe(code))")
            }
            return false
        }
        return locked.st_dev == current.st_dev && locked.st_ino == current.st_ino
    }

    private func sayItWaits() {
        let first = saidItWaits.withLock { said in
            defer { said = true }
            return !said
        }
        guard first else { return }
        FileHandle.standardError.write(
            Data(
                """
                [vmlx-tests] The MLX Metal test lock \(path) is taken; waiting for it, with no \
                time limit. `lsof \(path)` names the process that holds it.

                """.utf8))
    }

    private static func describe(_ code: Int32) -> String {
        String(cString: strerror(code))
    }
}

private extension FileManager {
    func copyFocusedMLXMetallibsIfMissing(from source: URL, into directory: URL) throws {
        try createDirectory(at: directory, withIntermediateDirectories: true)
        for name in ["default.metallib", "mlx.metallib"] {
            let destination = directory.appendingPathComponent(name)
            if !fileExists(atPath: destination.path) {
                try copyItem(at: source, to: destination)
            }
        }
    }
}
