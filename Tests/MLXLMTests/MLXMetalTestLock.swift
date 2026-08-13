import Foundation
import Darwin

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
/// Implementation: a named POSIX semaphore enforces single-tenant access even
/// across SwiftPM test targets. The async overload keeps a serial queue
/// occupied until the async body finishes; a plain actor method would be
/// reentrant at `await` points and would not protect Metal submissions that
/// happen after suspension.
enum MLXMetalTestLock {
    private static let semaphore = ProcessWideMLXTestSemaphore()
    private static let repoRoot: URL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()

    static func withLock<T>(_ body: () throws -> T) rethrows -> T {
        _ = metallibAliasPrepared
        semaphore.wait()
        defer { semaphore.signal() }
        return try body()
    }

    static func withLock<T: Sendable>(
        _ body: @Sendable @escaping () async throws -> T
    ) async throws -> T {
        _ = metallibAliasPrepared
        return try await withCheckedThrowingContinuation {
            (continuation: CheckedContinuation<T, Error>) in
            mlxTestSerializationQueue.async {
                semaphore.wait()
                defer { semaphore.signal() }
                let done = DispatchSemaphore(value: 0)
                // `nonisolated(unsafe)` is required because Swift 6 strict
                // concurrency cannot prove the cross-isolation transfer is
                // safe through the semaphore + queue boundary. The transfer
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
