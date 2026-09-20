import Foundation
import MLX
import SQLite3
import Testing

@testable import MLXLMCommon

/// The cache root is a user setting, and what is in it is data. A file that
/// carries one of this cache's names but was not written by it must never
/// take the process down, block it, or be read as anything it is not:
///
/// - a safetensors header whose offsets overflow is malformed, not a trap —
///   the open-time sweep reads every payload-named file at every launch;
/// - a FIFO under a payload's name is a miss, not an `open` that never
///   returns while the process-wide IO lock is held;
/// - under an index a newer build has claimed, the companion store removes
///   nothing from a listing either.
extension DiskCacheCompanionAccountingTests {

    @Suite(.serialized)
    struct HostilePayloads {

        private typealias Support = DiskCacheAccountingTestSupport
        private typealias RawDB = Support.RawDB

        private static func makeRoot(_ label: String) -> URL {
            FileManager.default.temporaryDirectory
                .appendingPathComponent("vmlx-hostile-payload-\(label)-\(UUID().uuidString)")
        }

        private static func tokens(_ count: Int, seed: Int) -> [Int] {
            (0 ..< count).map { seed * 100_000 + $0 }
        }

        private static func kv(_ elements: Int = 1_024) -> [String: MLXArray] {
            ["data": MLXArray.ones([elements], dtype: .float32)]
        }

        private static func recurrent(_ elements: Int = 1_024) -> [MLXArray] {
            [MLXArray.ones([elements], dtype: .float32)]
        }

        /// 8-byte little-endian header length, the header, then `payload`
        /// bytes. `declaredLength` overrides the length field.
        private static func safetensors(
            header: String, payload: Int = 64, declaredLength: UInt64? = nil
        ) -> Data {
            var data = Data()
            var length = (declaredLength ?? UInt64(header.utf8.count)).littleEndian
            data.append(Data(bytes: &length, count: 8))
            data.append(Data(header.utf8))
            data.append(Data(repeating: 0xA7, count: payload))
            return data
        }

        private static func header(offsets: String) -> String {
            #"{"kv_0_keys":{"dtype":"F32","shape":[4],"data_offsets":\#(offsets)}}"#
        }

        private static let foreignName = "model-00001-of-00002.safetensors"

        // MARK: - R2

        /// `"data_offsets":[0,9223372036854775807]`: `8 + headerLength + end`
        /// overflowed and TRAPPED — in the sweep every `DiskCache.init` runs,
        /// so at every launch for as long as the file existed, and in `fetch`
        /// when a row indexes it. Kept on its own so that a regression takes
        /// down one named test.
        @Test func overflowingHeaderOffsetsNeverTrap() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("overflow")
                defer { try? FileManager.default.removeItem(at: root) }
                try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
                let hostile = Self.safetensors(
                    header: Self.header(offsets: "[0,9223372036854775807]"))
                let ours = root.appendingPathComponent(
                    "00112233445566778899aabbccddeeff.safetensors")
                let foreign = root.appendingPathComponent(Self.foreignName)
                try hostile.write(to: ours)
                try hostile.write(to: foreign)

                // The pure inspection first: a trap here names the function.
                #expect(DiskCache.declaredPayloadEnd(url: ours) == nil)
                #expect(DiskCache.inspectSafetensors(url: ours) == .shortOrMalformed)

                let modelKey = "hostile-payload-overflow"
                let disk = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                #expect(
                    !FileManager.default.fileExists(atPath: ours.path),
                    "a malformed payload under OUR name is removed at open")
                #expect(
                    (try? Data(contentsOf: foreign)) == hostile,
                    "VICTIM: the same bytes under a foreign name were touched")

                // The same header under a row: `fetch` and `hasDurableEntry`.
                let tokens = Self.tokens(517, seed: 7_001)
                disk.store(tokens: tokens, arrays: Self.kv(), enforceQuota: false)
                let hash = DiskCache.hashTokens(tokens, modelKey: modelKey)
                let url = Support.payloadURL(root, hash)
                try #require(Support.fileBytes(url) > 0)
                try hostile.write(to: url)
                try RawDB(root: root).require(
                    "UPDATE cache_entries SET file_size = \(hostile.count) WHERE hash = '\(hash)'")
                // Header-only, and it must come back; what it answers for a
                // payload that lies is `fetch`'s to settle.
                _ = disk.hasDurableEntry(tokens: tokens, requireNativeRecurrent: true)
                #expect(disk.fetch(tokens: tokens) == nil)
                #expect(
                    Support.fileBytes(url) == 0, "a malformed payload of ours is removed by fetch")
                #expect(try Support.indexedRows(root).isEmpty)
                #expect((try? Data(contentsOf: foreign)) == hostile)
            }
        }

        /// Every other way a header can lie about its offsets or its own
        /// length. None of these trapped; each was, or could be, read as
        /// something it is not.
        @Test func lyingHeadersAreMalformedAndOnlyOursAreRemoved() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("lying")
                defer { try? FileManager.default.removeItem(at: root) }
                try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
                let plain = Self.header(offsets: "[0,16]")
                let cases: [(label: String, data: Data)] = [
                    ("negative end", Self.safetensors(header: Self.header(offsets: "[0,-16]"))),
                    ("negative begin", Self.safetensors(header: Self.header(offsets: "[-16,0]"))),
                    ("end before begin", Self.safetensors(header: Self.header(offsets: "[32,16]"))),
                    ("fractional end", Self.safetensors(header: Self.header(offsets: "[0,15.5]"))),
                    (
                        "end beyond Int64",
                        Self.safetensors(header: Self.header(offsets: "[0,18446744073709551615]"))
                    ),
                    (
                        "header longer than the file",
                        Self.safetensors(header: plain, declaredLength: 200 * 1024 * 1024)
                    ),
                    (
                        "header length 2^40",
                        Self.safetensors(header: plain, declaredLength: 1 << 40)
                    ),
                ]
                var planted: [(label: String, ours: URL, foreign: URL, data: Data)] = []
                for (index, entry) in cases.enumerated() {
                    let hash = String(format: "%032x", 0xA000 + index)
                    try #require(DiskCache.isPayloadHash(hash))
                    let sub = root.appendingPathComponent("foreign-\(index)")
                    try FileManager.default.createDirectory(
                        at: sub, withIntermediateDirectories: true)
                    let ours = root.appendingPathComponent("\(hash).safetensors")
                    let foreign = sub.appendingPathComponent(Self.foreignName)
                    try entry.data.write(to: ours)
                    try entry.data.write(to: foreign)
                    planted.append((entry.label, ours, foreign, entry.data))
                    #expect(
                        DiskCache.inspectSafetensors(url: ours) == .shortOrMalformed,
                        "\(entry.label) was read as \(DiskCache.inspectSafetensors(url: ours))")
                    #expect(!DiskCache.isCompleteSafetensors(url: ours), "\(entry.label)")
                }
                // The control: the same shape with honest offsets is complete.
                let honest = root.appendingPathComponent(
                    "\(String(format: "%032x", 0xAFFF)).safetensors")
                try Self.safetensors(header: plain).write(to: honest)
                try #require(
                    DiskCache.inspectSafetensors(url: honest) == .complete,
                    "INVALID: the honest control is not complete")
                let foreignInRoot = root.appendingPathComponent(Self.foreignName)
                try planted[0].data.write(to: foreignInRoot)

                _ = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: "hostile-lying")

                for entry in planted {
                    #expect(
                        !FileManager.default.fileExists(atPath: entry.ours.path),
                        "\(entry.label): a malformed payload under our name survived the open")
                    #expect(
                        (try? Data(contentsOf: entry.foreign)) == entry.data,
                        "VICTIM: \(entry.label) under a foreign name was touched")
                }
                #expect((try? Data(contentsOf: foreignInRoot)) == planted[0].data)
                #expect(FileManager.default.fileExists(atPath: honest.path))
            }
        }

        // MARK: - O1

        private final class Detached<T>: @unchecked Sendable {
            let done = DispatchSemaphore(value: 0)
            var value: T?
        }

        /// `body` on a detached thread, given ten seconds: a call that blocks
        /// on a FIFO FAILS its test instead of hanging the suite. nil when it
        /// did not come back in time — after opening `fifo` for writing, which
        /// lets a blocked `open` return (and release whatever lock it holds)
        /// so that the rest of the suite can run.
        private static func withTimeout<T>(
            _ what: String, fifo: URL, _ body: @escaping @Sendable () -> T
        ) -> T? {
            let outcome = Detached<T>()
            Thread.detachNewThread {
                outcome.value = body()
                outcome.done.signal()
            }
            if outcome.done.wait(timeout: .now() + 10) == .success {
                return outcome.value
            }
            Issue.record("\(what) blocked on a FIFO")
            let writer = open(fifo.path, O_WRONLY | O_NONBLOCK)
            if writer >= 0 { close(writer) }
            _ = outcome.done.wait(timeout: .now() + 10)
            return nil
        }

        /// `open(O_RDONLY)` on a FIFO blocks until somebody opens it for
        /// writing. `fetch` holds the process-wide IO lock, so that was every
        /// disk cache in the process, for good.
        @Test func aFIFOUnderAPayloadNameIsAMissNotAHang() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("fifo")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "hostile-payload-fifo"
                let tokens = Self.tokens(1_003, seed: 7_101)
                let disk = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                disk.store(tokens: tokens, arrays: Self.kv(), enforceQuota: false)
                let hash = DiskCache.hashTokens(tokens, modelKey: modelKey)
                let url = Support.payloadURL(root, hash)
                try FileManager.default.removeItem(at: url)
                try #require(mkfifo(url.path, 0o644) == 0, "INVALID: mkfifo failed")

                func fetchWithTimeout(_ what: String) -> Bool? {
                    Self.withTimeout(what, fifo: url) { disk.fetch(tokens: tokens) == nil }
                }

                DiskCache.resetRateLimitedReportsForTesting()
                let (results, log) = try Support.capturingStandardError {
                    [fetchWithTimeout("fetch"), fetchWithTimeout("a second fetch")]
                }
                #expect(results == [true, true], "\(results)")
                var info = stat()
                #expect(
                    lstat(url.path, &info) == 0 && (info.st_mode & S_IFMT) == S_IFIFO,
                    "the FIFO is not this cache's file and stays")
                #expect(
                    try RawDB(root: root).rows("SELECT 1 FROM cache_entries WHERE hash = '\(hash)'")
                        .count == 1, "the row is kept")
                #expect(disk.snapshotStats().unreadablePayloadFetches == 2)
                #expect(
                    log.split(separator: "\n").filter {
                        $0.hasPrefix("[vmlx][cache/disk] fetch could not read ")
                    }.count == 1, "reported once: \(log)")
                #expect(!disk.hasDurableEntry(tokens: tokens))
                #expect(!disk.touchRecency(tokens: tokens, at: Date()))
            }
        }

        /// Every FIFO path above returns BEFORE it reaches the header reader
        /// (`fetch` and the sweep at their `lstat`, `hasDurableEntry` at the
        /// size), so without this test `O_NONBLOCK` and the regular-file
        /// check in the reader could both be deleted with the suite green.
        /// They are what stands between a FIFO and a caller that has no
        /// `lstat` of its own — or one that loses the race with a rename.
        @Test func theHeaderReaderItselfNeverBlocksOnAFIFO() throws {
            let root = Self.makeRoot("fifo-reader")
            defer { try? FileManager.default.removeItem(at: root) }
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            let url = root.appendingPathComponent("00112233445566778899aabbccddeeff.safetensors")
            try #require(mkfifo(url.path, 0o644) == 0, "INVALID: mkfifo failed")
            var info = stat()
            try #require(
                lstat(url.path, &info) == 0 && (info.st_mode & S_IFMT) == S_IFIFO,
                "INVALID: not a FIFO")

            let result = Self.withTimeout("inspectSafetensors", fifo: url) {
                DiskCache.inspectSafetensors(url: url)
            }
            let inspection = try #require(result, "the header reader blocked on a FIFO")
            guard case .unreadable(let code) = inspection else {
                // `.shortOrMalformed` is what permits a caller to DELETE.
                Issue.record("a FIFO was read, and judged \(inspection)")
                return
            }
            print("FIFO_INSPECTION errno=\(code) (\(String(cString: strerror(code))))")
            #expect(code != 0)
            #expect(
                Self.withTimeout("declaredPayloadEnd", fifo: url) {
                    DiskCache.declaredPayloadEnd(url: url) == nil
                } == true)
            #expect(
                Self.withTimeout("isCompleteSafetensors", fifo: url) {
                    !DiskCache.isCompleteSafetensors(url: url)
                } == true)
        }

        // MARK: - O2

        /// `CacheCoordinator.clear()` under a newer build's index: the
        /// companion store removes what the index names and nothing from a
        /// listing, as `DiskCache.clear()` does in the root.
        @Test func companionClearRemovesNothingFromAListingUnderANewerSchema() throws {
            try MLXMetalTestLock.withLock {
                for newer in [true, false] {
                    let root = Self.makeRoot("clear-\(newer ? "newer" : "current")")
                    defer { try? FileManager.default.removeItem(at: root) }
                    let modelKey = "hostile-payload-clear-\(newer)"
                    // Only the writer imports. The subject opens a root this
                    // process has already imported, so the planted pair stays
                    // what it is: on disk, and named by nothing.
                    func coordinator(importing: Bool) -> CacheCoordinator {
                        if importing { CacheCoordinator.resetImportedRootsForTesting() }
                        let made = CacheCoordinator(
                            config: CacheCoordinatorConfig(
                                usePagedCache: false, enableDiskCache: true,
                                diskCacheMaxGB: 1, diskCacheDir: root, modelKey: modelKey))
                        made.setHybrid(true, requiresRecurrentSSMCompanion: true)
                        return made
                    }
                    let tokens = Self.tokens(523, seed: 7_201)
                    coordinator(importing: true).storePersistentBoundary(
                        tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent())
                    let named = SSMCompanionDiskStore.keyFor(
                        tokens: tokens, boundary: tokens.count, modelKey: modelKey)
                    try #require(Support.companionBytes(root, named) > 0)

                    // Carry this store's names, and the index names none of them.
                    let dir = Support.companionDir(root)
                    let unlisted = String(repeating: "5e", count: 32)
                    var listingOnly: [URL: Data] = [:]
                    for url in Support.companionURLs(root, unlisted) {
                        listingOnly[url] = Data(repeating: 0x5E, count: 4_099)
                    }
                    let partial = DiskCache.temporaryURL(
                        for: dir.appendingPathComponent("ssm-\(unlisted).safetensors"))
                    listingOnly[partial] = Data(repeating: 0x5F, count: 4_111)
                    for (url, data) in listingOnly {
                        try data.write(to: url)
                        try Support.age(url, by: 11 * 60)
                    }
                    if newer { try RawDB(root: root).require("PRAGMA user_version = 7") }

                    let subject = coordinator(importing: false)
                    try #require(subject.diskCache?.indexIsFromANewerBuild == newer)
                    try #require(
                        try Support.legacyRows(root).isEmpty,
                        "INVALID: the index names the planted pair")
                    let (_, log) = try Support.capturingStandardError { subject.clear() }

                    #expect(
                        Support.companionBytes(root, named) == 0,
                        "the companion the index names goes (newer=\(newer))")
                    for (url, data) in listingOnly {
                        if newer {
                            #expect(
                                (try? Data(contentsOf: url)) == data,
                                "VICTIM: \(url.lastPathComponent) was removed from a listing under a newer schema"
                            )
                        } else {
                            #expect(
                                !FileManager.default.fileExists(atPath: url.path),
                                "INVALID: the control — the current schema — left \(url.lastPathComponent)"
                            )
                        }
                    }
                    #expect(
                        log.contains("[vmlx][cache/ssm-store] clear skipped: ") == newer, "\(log)")
                }
            }
        }

        // MARK: - O5

        /// `fetch` tells "the loader could not OPEN the file" from "the loader
        /// read it and rejected it" by a substring of the vendored loader's
        /// message, and deletes on the second. If an MLX bump rewords that
        /// message this fails, instead of users' caches being deleted on a
        /// transient EACCES.
        @Test func theLoaderStillSaysItCouldNotOpenAFile() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("loader-text")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "hostile-payload-loader"
                let tokens = Self.tokens(517, seed: 7_301)
                let disk = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                disk.store(tokens: tokens, arrays: Self.kv(), enforceQuota: false)
                let url = Support.payloadURL(root, DiskCache.hashTokens(tokens, modelKey: modelKey))
                try #require(Support.fileBytes(url) > 0)
                // The control: the real loader reads the real payload.
                _ = try loadArraysAndMetadata(url: url)

                try #require(chmod(url.path, 0) == 0)
                defer { chmod(url.path, 0o644) }
                let probe = open(url.path, O_RDONLY)
                if probe >= 0 { close(probe) }
                try #require(
                    probe < 0, "INVALID: chmod 000 had no effect (root / filesystem)")

                var message: String?
                do {
                    _ = try loadArraysAndMetadata(url: url)
                } catch {
                    message = "\(error)"
                }
                let thrown = try #require(
                    message, "the loader did not throw for an unreadable file")
                #expect(
                    thrown.contains(DiskCache.loaderCouldNotOpenMarker),
                    "the loader now says \(thrown.debugDescription); DiskCache.isPositiveCorruption matches \(DiskCache.loaderCouldNotOpenMarker.debugDescription) and would DELETE this payload"
                )
            }
        }
    }
}
