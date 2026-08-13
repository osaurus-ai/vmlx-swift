// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Darwin

enum FocusedMLXTestSupport {
    private static let queue = DispatchQueue(label: "ai.osaurus.vmlx.focused-mlx-tests")
    private static let semaphore = ProcessWideMLXTestSemaphore()

    static func withLock<T>(_ body: () throws -> T) rethrows -> T {
        _ = metallibPrepared
        semaphore.wait()
        defer { semaphore.signal() }
        return try body()
    }

    /// Acquires the process-wide MLX lock without blocking a cooperative
    /// thread.
    ///
    /// The previous shape parked a thread on `done.wait()` while a nested
    /// `Task` ran `body()`. That task needs a cooperative-pool thread, and
    /// under pool starvation it never gets one, so the wait never returns and
    /// the semaphore is never signalled. Every later `withLock` — including
    /// the SYNCHRONOUS one used by the Hy3 sanitizer tests — then blocks
    /// forever at 0% CPU, hanging the whole test target and holding the
    /// SwiftPM `.build` lock with it.
    ///
    /// Now only the private serial queue's own thread blocks while waiting for
    /// the semaphore; `body()` is awaited normally by the caller, so no task
    /// is ever waited on from inside a blocked thread.
    static func withLock<T: Sendable>(
        _ body: @Sendable @escaping () async throws -> T
    ) async throws -> T {
        _ = metallibPrepared
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            queue.async {
                semaphore.wait()
                continuation.resume()
            }
        }
        defer { semaphore.signal() }
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

private final class ProcessWideMLXTestSemaphore: @unchecked Sendable {
    private static let name = "/vmlx_mlx_lock"

    /// How long to wait for the current holder before treating the semaphore
    /// as abandoned. Real MLX sections here are well under a second; minutes
    /// mean the holder is gone.
    private static let staleAfter: TimeInterval = 90

    private var pointer: UnsafeMutablePointer<sem_t>

    init() {
        pointer = Self.open()
    }

    private static func open() -> UnsafeMutablePointer<sem_t> {
        guard let sem = sem_open(name, O_CREAT, 0o600, 1), sem != SEM_FAILED else {
            fatalError("Unable to create MLX Metal test semaphore")
        }
        return sem
    }

    deinit {
        sem_close(pointer)
    }

    /// Acquire, recovering from a holder that died without signalling.
    ///
    /// This is a NAMED POSIX semaphore, so its count outlives the process. A
    /// test run killed mid-section (Ctrl-C, a CI timeout, a crash) leaves the
    /// count at zero permanently, and every later run on that machine then
    /// blocks forever at 0% CPU — including runs that only touch unrelated
    /// tests, because the whole target shares this lock. Recovering that state
    /// previously required knowing to call `sem_unlink` by hand.
    ///
    /// Poll instead of blocking outright: if nobody releases within
    /// ``staleAfter``, unlink the abandoned semaphore and reopen a fresh one.
    /// Unlinking only detaches the name — a live holder keeps its own handle
    /// and its `signal()` stays valid — so the worst case for a false positive
    /// is two runs briefly sharing MLX, not a corrupted lock.
    func wait() {
        let deadline = Date().addingTimeInterval(Self.staleAfter)
        while true {
            if sem_trywait(pointer) == 0 { return }
            if errno != EAGAIN && errno != EINTR {
                fatalError("Unable to wait on MLX Metal test semaphore")
            }
            if Date() >= deadline {
                FileHandle.standardError.write(
                    Data(
                        """
                        [vmlx-tests] MLX lock \(Self.name) held for >\(Int(Self.staleAfter))s; \
                        assuming a killed run abandoned it and resetting.

                        """.utf8))
                sem_unlink(Self.name)
                sem_close(pointer)
                pointer = Self.open()
                if sem_trywait(pointer) == 0 { return }
                fatalError("Unable to recover abandoned MLX Metal test semaphore")
            }
            usleep(20_000)
        }
    }

    func signal() {
        if sem_post(pointer) == -1 {
            fatalError("Unable to signal MLX Metal test semaphore")
        }
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
