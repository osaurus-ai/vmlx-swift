import Foundation

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
/// Implementation: a serial `DispatchQueue` enforces single-tenant access.
/// The async overload keeps that queue occupied until the async body finishes;
/// a plain actor method would be reentrant at `await` points and would not
/// protect Metal submissions that happen after suspension.
enum MLXMetalTestLock {
    private static let repoRoot: URL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()

    static func withLock<T>(_ body: () throws -> T) rethrows -> T {
        _ = metallibAliasPrepared
        return try mlxTestSerializationQueue.sync(execute: body)
    }

    static func withLock<T: Sendable>(
        _ body: @Sendable @escaping () async throws -> T
    ) async throws -> T {
        _ = metallibAliasPrepared
        return try await withCheckedThrowingContinuation {
            (continuation: CheckedContinuation<T, Error>) in
            mlxTestSerializationQueue.async {
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
