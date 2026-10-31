import Darwin
import Foundation
import XCTest
import os

/// `ProcessWideMLXTestLock` serializes MLX work across every test process.
/// Its holder can die inside `withLock` (a crash, a signal, a killed run);
/// the lock must then be free for the next waiter at once.
final class MLXMetalTestLockTests: XCTestCase {
    private static let childLockPathKey = "VMLX_TEST_LOCK_CHILD_PATH"
    private static let childReadyPathKey = "VMLX_TEST_LOCK_CHILD_READY_PATH"
    private static let childStartsAProcessKey = "VMLX_TEST_LOCK_CHILD_STARTS_PROCESS"

    func testAHolderKilledInsideTheLockLeavesItFreeForTwoWaitersInTurn() throws {
        try killTheHolderAndCheckTheWaiters(name: "vmlx_t\(getpid())", childStartsAProcess: false)
    }

    /// A process that the holder started, and that outlives it, does not
    /// keep the lock: it inherits no descriptor of the lock file.
    func testAProcessTheHolderStartedDoesNotKeepTheLockAfterTheHolderDies() throws {
        try killTheHolderAndCheckTheWaiters(name: "vmlx_t\(getpid())p", childStartsAProcess: true)
    }

    /// Two threads of one process: B asks while A holds the lock, and gets it
    /// only after A has released it.
    func testASecondThreadWaitsUntilTheFirstReleases() throws {
        let lockPath = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx_t\(getpid()).threads").path
        defer { try? FileManager.default.removeItem(atPath: lockPath) }
        let lock = ProcessWideMLXTestLock(path: lockPath)
        let events = Events()
        let aHolds = DispatchSemaphore(value: 0)
        let releaseA = DispatchSemaphore(value: 0)
        let bHolds = DispatchSemaphore(value: 0)
        let bothDone = DispatchGroup()

        bothDone.enter()
        DispatchQueue.global().async {
            lock.wait()
            aHolds.signal()
            releaseA.wait()
            events.append("A releases")
            lock.signal()
            bothDone.leave()
        }
        XCTAssertEqual(aHolds.wait(timeout: .now() + 10), .success, "A did not get the free lock")

        bothDone.enter()
        DispatchQueue.global().async {
            lock.wait()
            events.append("B holds")
            bHolds.signal()
            lock.signal()
            bothDone.leave()
        }
        XCTAssertEqual(
            bHolds.wait(timeout: .now() + 1), .timedOut, "B got the lock while A held it")

        releaseA.signal()
        XCTAssertEqual(
            bothDone.wait(timeout: .now() + 10), .success,
            "B did not get the lock within 10 s of A's release")
        XCTAssertEqual(events.all, ["A releases", "B holds"])
    }

    /// The lock file is deleted while A holds it and B waits for it. A thread
    /// that asks after that opens a new file, so when A releases, B must not
    /// keep the deleted file's lock: C would get the new file's lock while B
    /// holds the old one. Each thread has an instance of its own, as separate
    /// processes would.
    func testAWaiterDoesNotKeepTheLockOfAFileDeletedWhileItWaited() throws {
        let lockPath = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx_t\(getpid()).deleted").path
        defer { try? FileManager.default.removeItem(atPath: lockPath) }
        let holders = Holders()
        let allDone = DispatchGroup()
        // Takes the lock, signals `holds`, and keeps the lock until `release`.
        func hold(_ holds: DispatchSemaphore, until release: DispatchSemaphore?) {
            allDone.enter()
            DispatchQueue.global().async {
                let lock = ProcessWideMLXTestLock(path: lockPath)
                lock.wait()
                holders.enter()
                holds.signal()
                release?.wait()
                holders.leave()
                lock.signal()
                allDone.leave()
            }
        }

        let aHolds = DispatchSemaphore(value: 0)
        let releaseA = DispatchSemaphore(value: 0)
        hold(aHolds, until: releaseA)
        XCTAssertEqual(aHolds.wait(timeout: .now() + 10), .success, "A did not get the free lock")
        let file = try identity(ofFileAtPath: lockPath)

        let bHolds = DispatchSemaphore(value: 0)
        let releaseB = DispatchSemaphore(value: 0)
        hold(bHolds, until: releaseB)
        // B has opened the file once this process has two descriptors on it.
        XCTAssertTrue(waitForDescriptors(2, on: file, timeout: 10), "B did not open the lock file")
        XCTAssertEqual(
            bHolds.wait(timeout: .now() + 0.2), .timedOut, "B got the lock while A held it")

        try FileManager.default.removeItem(atPath: lockPath)
        releaseA.signal()
        XCTAssertEqual(
            bHolds.wait(timeout: .now() + 10), .success,
            "B did not get the lock after A released it")

        let cHolds = DispatchSemaphore(value: 0)
        hold(cHolds, until: nil)
        XCTAssertEqual(
            cHolds.wait(timeout: .now() + 1), .timedOut, "C got the lock while B held it")
        releaseB.signal()
        XCTAssertEqual(
            allDone.wait(timeout: .now() + 10), .success,
            "C did not get the lock after B released it")
        XCTAssertEqual(holders.mostAtOnce, 1, "two threads held the lock at the same time")
    }

    /// The child process that the tests above start: takes the lock, starts
    /// a process of its own if asked to, says so, and holds the lock until it
    /// is killed.
    func testChildHoldsTheLockUntilKilled() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let lockPath = environment[Self.childLockPathKey],
            let readyPath = environment[Self.childReadyPathKey]
        else {
            throw XCTSkip("runs only in the child process of the tests above")
        }
        let lock = ProcessWideMLXTestLock(path: lockPath)
        lock.wait()
        var started: pid_t = 0
        if environment[Self.childStartsAProcessKey] != nil {
            started = try startSleeper()
        }
        // Written whole, then renamed, so the parent never reads it half done.
        try Data("\(started)".utf8).write(to: URL(fileURLWithPath: readyPath), options: .atomic)
        Thread.sleep(forTimeInterval: 120)
        lock.signal()
    }

    /// Starts a child that holds the lock, checks that two waiters in this
    /// process do not get it, kills the child, and checks that both waiters
    /// then get it in turn. With `childStartsAProcess`, the child first starts
    /// a process that is still running when the waiters are checked.
    private func killTheHolderAndCheckTheWaiters(name: String, childStartsAProcess: Bool) throws {
        let directory = FileManager.default.temporaryDirectory
        let lockPath = directory.appendingPathComponent(name).path
        let readyPath = directory.appendingPathComponent(name + ".ready").path
        let childLog = directory.appendingPathComponent(name + ".log")
        defer {
            for path in [lockPath, readyPath, childLog.path] {
                try? FileManager.default.removeItem(atPath: path)
            }
        }

        // A child process takes the lock.
        let child = try startChild(
            lockPath: lockPath, readyPath: readyPath, log: childLog,
            startsAProcess: childStartsAProcess)
        defer { if child.isRunning { kill(child.processIdentifier, SIGKILL) } }
        guard waitForFile(atPath: readyPath, timeout: 60) else {
            XCTFail("the child did not take the lock within 60 s; its output:\n\(childOutput(childLog))")
            return
        }
        var started = pid_t(
            (try? String(contentsOfFile: readyPath, encoding: .utf8)).flatMap { Int32($0) } ?? -1)
        // Killed once only: once reaped, its pid may name another process.
        defer { if started > 0 { kill(started, SIGKILL) } }
        if childStartsAProcess {
            guard started > 0, kill(started, 0) == 0 else {
                XCTFail("the child did not start a process; its output:\n\(childOutput(childLog))")
                return
            }
        }

        // Two waiters share one instance, as the threads of a test process do.
        let lock = ProcessWideMLXTestLock(path: lockPath)
        let holders = Holders()
        let bothDone = DispatchGroup()
        for _ in 0 ..< 2 {
            bothDone.enter()
            DispatchQueue.global().async {
                lock.wait()
                holders.enter()
                Thread.sleep(forTimeInterval: 0.2)
                holders.leave()
                lock.signal()
                bothDone.leave()
            }
        }
        Thread.sleep(forTimeInterval: 1)
        XCTAssertEqual(holders.acquisitions, 0, "a waiter got the lock while the child held it")

        // The holder dies without releasing it.
        kill(child.processIdentifier, SIGKILL)
        child.waitUntilExit()
        XCTAssertEqual(child.terminationReason, .uncaughtSignal)
        XCTAssertEqual(child.terminationStatus, SIGKILL)

        let inTime = bothDone.wait(timeout: .now() + 10) == .success
        XCTAssertTrue(inTime, "the two waiters did not both get the lock within 10 s of its holder's death")
        if childStartsAProcess {
            XCTAssertEqual(kill(started, 0), 0, "the process the child started is no longer running")
        }
        if !inTime {
            // A process that kept the lock: end it. A lock that recovers on a
            // timer: let it. Either way the outcome belongs to this test.
            if started > 0 {
                kill(started, SIGKILL)
                started = -1
            }
            _ = bothDone.wait(timeout: .now() + 110)
        }
        XCTAssertEqual(holders.acquisitions, 2)
        XCTAssertEqual(holders.mostAtOnce, 1, "two waiters held the lock at the same time")
    }

    /// This test bundle, run again by its own `xctest` host, filtered to the
    /// child test.
    private func startChild(
        lockPath: String, readyPath: String, log: URL, startsAProcess: Bool
    ) throws -> Process {
        guard let host = Bundle.main.executableURL, host.lastPathComponent == "xctest" else {
            throw XCTSkip("needs the xctest host to start the child; the host is \(Bundle.main.executablePath ?? "unknown")")
        }
        let child = Process()
        child.executableURL = host
        child.arguments = [
            "-XCTest", "\(String(reflecting: Self.self))/testChildHoldsTheLockUntilKilled",
            Bundle(for: Self.self).bundlePath,
        ]
        var environment = ProcessInfo.processInfo.environment
        environment[Self.childLockPathKey] = lockPath
        environment[Self.childReadyPathKey] = readyPath
        environment[Self.childStartsAProcessKey] = startsAProcess ? "1" : nil
        child.environment = environment
        FileManager.default.createFile(atPath: log.path, contents: nil)
        let output = try FileHandle(forWritingTo: log)
        child.standardOutput = output
        child.standardError = output
        try child.run()
        return child
    }

    /// `/bin/sleep 120`, started with `posix_spawn`'s defaults, so that it
    /// inherits every descriptor of this process that is not close-on-exec.
    private func startSleeper() throws -> pid_t {
        let arguments: [UnsafeMutablePointer<CChar>?] = [strdup("/bin/sleep"), strdup("120"), nil]
        defer { for argument in arguments { free(argument) } }
        let environment: [UnsafeMutablePointer<CChar>?] = [nil]
        var pid: pid_t = 0
        let status = posix_spawn(&pid, "/bin/sleep", nil, nil, arguments, environment)
        guard status == 0 else {
            XCTFail("posix_spawn /bin/sleep: \(String(cString: strerror(status)))")
            return -1
        }
        return pid
    }

    private func waitForFile(atPath path: String, timeout: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if FileManager.default.fileExists(atPath: path) { return true }
            Thread.sleep(forTimeInterval: 0.05)
        }
        return false
    }

    private typealias FileIdentity = (device: dev_t, inode: ino_t)

    private func identity(ofFileAtPath path: String) throws -> FileIdentity {
        var status = stat()
        guard stat(path, &status) == 0 else {
            let code = errno
            throw POSIXError(POSIXErrorCode(rawValue: code) ?? .EIO)
        }
        return (status.st_dev, status.st_ino)
    }

    /// Waits until this process has `count` descriptors open on `file`.
    private func waitForDescriptors(
        _ count: Int, on file: FileIdentity, timeout: TimeInterval
    ) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            var found = 0
            for descriptor in 0 ..< min(getdtablesize(), 1 << 16) {
                var status = stat()
                if fstat(descriptor, &status) == 0, status.st_dev == file.device,
                    status.st_ino == file.inode
                {
                    found += 1
                }
            }
            if found >= count { return true }
            Thread.sleep(forTimeInterval: 0.01)
        }
        return false
    }

    private func childOutput(_ log: URL) -> String {
        (try? String(contentsOf: log, encoding: .utf8)) ?? "(none)"
    }
}

/// How many waiters got the lock, and the most that held it at once.
private final class Holders: Sendable {
    private struct State {
        var inside = 0
        var acquisitions = 0
        var mostAtOnce = 0
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    var acquisitions: Int { state.withLock { $0.acquisitions } }
    var mostAtOnce: Int { state.withLock { $0.mostAtOnce } }

    func enter() {
        state.withLock {
            $0.inside += 1
            $0.acquisitions += 1
            $0.mostAtOnce = max($0.mostAtOnce, $0.inside)
        }
    }

    func leave() {
        state.withLock { $0.inside -= 1 }
    }
}

/// What the threads of a test did, in order.
private final class Events: Sendable {
    private let list = OSAllocatedUnfairLock(initialState: [String]())

    var all: [String] { list.withLock { $0 } }

    func append(_ event: String) {
        list.withLock { $0.append(event) }
    }
}
