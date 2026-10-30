import Foundation
import Darwin
import os

/// Process-wide gate to serialize MLX Metal work across test suites.
///
/// Swift Testing's default parallelism runs tests across suites concurrently.
/// MLX kernels share a single Metal command buffer, so two tests that issue
/// Metal work simultaneously trigger
/// `AGXG17XFamilyCommandBuffer tryCoalescingPreviousComputeCommandEncoder…`
/// assertions or signal-11 segfaults. Wrapping each MLX-touching test body
/// in `MLXMetalTestLock.withLock { … }` (or `try await … withLock { … }`
/// for async tests) serializes Metal work across the entire test process,
/// independent of `@Suite(.serialized)` (which only protects within a
/// single suite).
///
/// Implementation: a lock file that every test process locks with `flock(2)`
/// (``ProcessWideMLXTestLock``) enforces single-tenant access even across
/// SwiftPM test targets, and the kernel releases it when its holder dies.
/// The async overload keeps a serial queue occupied until the async body
/// finishes; a plain actor method would be reentrant at `await` points and
/// would not protect Metal submissions that happen after suspension.
enum MLXMetalTestLock {
    private static let lock = ProcessWideMLXTestLock()
    private static let repoRoot: URL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()

    static func withLock<T>(_ body: () throws -> T) rethrows -> T {
        _ = metallibAliasPrepared
        lock.wait()
        defer { lock.signal() }
        return try body()
    }

    static func withLock<T: Sendable>(
        _ body: @Sendable @escaping () async throws -> T
    ) async throws -> T {
        _ = metallibAliasPrepared
        return try await withCheckedThrowingContinuation {
            (continuation: CheckedContinuation<T, Error>) in
            mlxTestSerializationQueue.async {
                lock.wait()
                defer { lock.signal() }
                let done = DispatchSemaphore(value: 0)
                // `nonisolated(unsafe)` is required because Swift 6 strict
                // concurrency cannot prove the cross-isolation transfer is
                // safe through the lock + queue boundary. The transfer
                // IS safe here: the Task signals `done` only after writing
                // `output`, and `done.wait()` happens-before the read.
                nonisolated(unsafe) var output: Result<T, Error>?
                Task { @Sendable in
                    do {
                        let value = try await body()
                        output = .success(value)
                    } catch {
                        output = .failure(error)
                    }
                    done.signal()
                }
                done.wait()
                switch output {
                case .success(let result):
                    continuation.resume(returning: result)
                case .failure(let error):
                    continuation.resume(throwing: error)
                case .none:
                    continuation.resume(throwing: CocoaError(.userCancelled))
                }
            }
        }
    }

    /// mlx-swift's Metal loader first looks for `mlx.metallib` colocated with
    /// the test binary, while SwiftPM currently emits `default.metallib` in
    /// this package's test layout. Create a local build-artifact alias before
    /// the first MLX-backed assertion runs so tests exercise kernels instead
    /// of failing on a runner packaging detail.
    private static let metallibAliasPrepared: Void = {
        prepareMLXTestMetalLibrary()
        if ProcessInfo.processInfo.environment["VMLX_TEST_METALLIB_PATH"] != nil {
            return
        }
        let sourceDirectories = [
            repoRoot.appendingPathComponent(".build/arm64-apple-macosx/debug"),
            repoRoot.appendingPathComponent(".build/debug"),
        ]
        let sourceDirectory = sourceDirectories.first {
            FileManager.default.fileExists(atPath: $0.appendingPathComponent("default.metallib").path)
        }
        let fileManager = FileManager.default
        var candidates: [URL] = []

        if let executableURL = Bundle.main.executableURL {
            candidates.append(executableURL.deletingLastPathComponent())
        }
        if let resourceURL = Bundle.main.resourceURL {
            candidates.append(resourceURL)
        }
        if let firstArgument = CommandLine.arguments.first, !firstArgument.isEmpty {
            candidates.append(URL(fileURLWithPath: firstArgument).deletingLastPathComponent())
        }
        candidates.append(repoRoot.appendingPathComponent(".build/arm64-apple-macosx/debug"))
        candidates.append(
            repoRoot.appendingPathComponent(
                ".build/arm64-apple-macosx/debug/mlx-swiftPackageTests.xctest/Contents/MacOS"))
        candidates.append(
            repoRoot.appendingPathComponent(
                ".build/arm64-apple-macosx/debug/vmlx-swift-lmPackageTests.xctest/Contents/MacOS"))
        candidates.append(repoRoot.appendingPathComponent(".build/debug"))
        candidates.append(
            repoRoot.appendingPathComponent(
                ".build/debug/mlx-swiftPackageTests.xctest/Contents/MacOS"))
        candidates.append(
            repoRoot.appendingPathComponent(
                ".build/debug/vmlx-swift-lmPackageTests.xctest/Contents/MacOS"))

        var scanned = Set<String>()
        for candidate in candidates {
            var directory = candidate.standardizedFileURL
            for _ in 0 ..< 4 {
                let path = directory.path
                if scanned.insert(path).inserted {
                    if let sourceDirectory {
                        let source = sourceDirectory.appendingPathComponent("default.metallib")
                        try? fileManager.copyMLXTestMetallibsIfMissing(from: source, into: directory)
                    } else {
                        let defaultURL = directory.appendingPathComponent("default.metallib")
                        let aliasURL = directory.appendingPathComponent("mlx.metallib")
                        if fileManager.fileExists(atPath: defaultURL.path),
                           !fileManager.fileExists(atPath: aliasURL.path)
                        {
                            try? fileManager.copyItem(at: defaultURL, to: aliasURL)
                        }
                    }
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
    func copyMLXTestMetallibsIfMissing(from source: URL, into directory: URL) throws {
        try createDirectory(at: directory, withIntermediateDirectories: true)
        for name in ["default.metallib", "mlx.metallib"] {
            let destination = directory.appendingPathComponent(name)
            if !fileExists(atPath: destination.path) {
                try copyItem(at: source, to: destination)
            }
        }
    }
}
