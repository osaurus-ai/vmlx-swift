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

    static func withLock<T: Sendable>(
        _ body: @Sendable @escaping () async throws -> T
    ) async throws -> T {
        _ = metallibPrepared
        return try await withCheckedThrowingContinuation {
            (continuation: CheckedContinuation<T, Error>) in
            queue.async {
                semaphore.wait()
                defer { semaphore.signal() }
                let done = DispatchSemaphore(value: 0)
                nonisolated(unsafe) var output: Result<T, Error>?
                Task { @Sendable in
                    do {
                        output = .success(try await body())
                    } catch {
                        output = .failure(error)
                    }
                    done.signal()
                }
                done.wait()
                switch output {
                case .success(let value):
                    continuation.resume(returning: value)
                case .failure(let error):
                    continuation.resume(throwing: error)
                case .none:
                    continuation.resume(throwing: CocoaError(.userCancelled))
                }
            }
        }
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
    private let pointer: UnsafeMutablePointer<sem_t>

    init() {
        let name = "/vmlx_mlx_lock"
        guard let sem = sem_open(name, O_CREAT, 0o600, 1), sem != SEM_FAILED else {
            fatalError("Unable to create MLX Metal test semaphore")
        }
        pointer = sem
    }

    deinit {
        sem_close(pointer)
    }

    func wait() {
        while sem_wait(pointer) == -1 {
            if errno != EINTR {
                fatalError("Unable to wait on MLX Metal test semaphore")
            }
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
