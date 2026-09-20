import Foundation
import MLX
import SQLite3
import Testing

@testable import MLXLMCommon

/// `cache_index.db` is a plain SQLite file. An older build, another tool or
/// corruption can put anything in it, and the cache root is a user setting
/// that may be somebody's model folder. So a VALUE read from the index — a
/// row's `hash`, a `companion_key`, a `legacy_companions.key` — is data, never
/// a path component: one that is not exactly what this cache computes names
/// no file. It is never turned into a path, never stat'ed and never deleted
/// through; the record that carries it is dropped, so it stops counting.
///
/// The same rule for reads: a payload that could not be READ (EACCES, EMFILE,
/// EIO) has not been shown to be corrupt, and is never deleted for it.
///
/// Every test plants real victim files exactly where the hostile value would
/// point, and asserts them byte for byte afterwards. Token counts are
/// deliberately not multiples of 64 or 256.
extension DiskCacheCompanionAccountingTests {

    @Suite(.serialized)
    struct IndexValueSafety {

        private typealias Support = DiskCacheAccountingTestSupport
        private typealias RawDB = Support.RawDB

        // MARK: - Fixtures

        private static func makeRoot(_ label: String) -> URL {
            FileManager.default.temporaryDirectory
                .appendingPathComponent("vmlx-index-value-\(label)-\(UUID().uuidString)")
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

        private static let cap: Int64 = 1 << 20

        private static func coordinator(
            root: URL, capBytes: Int64 = cap, modelKey: String
        ) -> CacheCoordinator {
            CacheCoordinator(
                config: CacheCoordinatorConfig(
                    usePagedCache: false,
                    enableDiskCache: true,
                    diskCacheMaxGB: Float(capBytes) / 1_073_741_824,
                    diskCacheDir: root,
                    modelKey: modelKey))
        }

        private static func listing(_ dir: URL) throws -> Set<String> {
            Set(try FileManager.default.contentsOfDirectory(atPath: dir.path))
        }

        /// Files that are somebody else's, each planted exactly where a
        /// hostile index value would point.
        private struct Victims {
            private(set) var files: [URL: Data] = [:]

            mutating func plant(_ url: URL, fill: UInt8) throws {
                try FileManager.default.createDirectory(
                    at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                let data = Data(repeating: fill, count: 4_099 + Int(fill))
                try data.write(to: url)
                files[url] = data
            }

            func requirePlanted() throws {
                try #require(!files.isEmpty, "INVALID: no victim was planted")
                for (url, data) in files {
                    try #require(
                        (try? Data(contentsOf: url)) == data,
                        "INVALID: victim \(url.path) is not on disk")
                }
            }

            func expectIntact(after what: String, sourceLocation: SourceLocation = #_sourceLocation)
            {
                for (url, data) in files.sorted(by: { $0.key.path < $1.key.path }) {
                    #expect(
                        (try? Data(contentsOf: url)) == data,
                        "VICTIM \(url.path) did not survive \(what) byte for byte",
                        sourceLocation: sourceLocation)
                }
            }
        }

        // MARK: - R1 fixtures: hostile `cache_entries.hash`

        private struct HostileHashes {
            var hashes: [String] = []
            var victims = Victims()
            /// Victim names in the root, for the completeness walk.
            var namesInRoot: Set<String> = []
        }

        /// Rows planted by raw SQL, the oldest in the index and each larger
        /// than the whole cap, next to the files their `hash` would address
        /// if it were pasted into a path.
        private static func plantHostileHashRows(
            root: URL, outside: URL
        ) throws -> HostileHashes {
            var hostile = HostileHashes()
            try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)

            // Traversal out of the root (root and `outside` are siblings).
            hostile.hashes.append("../\(outside.lastPathComponent)/victim")
            try hostile.victims.plant(
                outside.appendingPathComponent("victim.safetensors"), fill: 0x11)

            // No traversal at all: the root is a model folder.
            hostile.hashes.append("model-00001-of-00008")
            try hostile.victims.plant(
                root.appendingPathComponent("model-00001-of-00008.safetensors"), fill: 0x12)

            hostile.hashes.append("")
            try hostile.victims.plant(root.appendingPathComponent(".safetensors"), fill: 0x13)

            // An absolute-path fragment. Appended to the root it addresses a
            // file under the root; the absolute file itself is planted too.
            hostile.hashes.append("\(outside.path)/absolute-victim")
            try hostile.victims.plant(
                outside.appendingPathComponent("absolute-victim.safetensors"), fill: 0x14)
            try hostile.victims.plant(
                URL(fileURLWithPath: root.path + outside.path + "/absolute-victim.safetensors"),
                fill: 0x15)

            let upper = "ABCDEF0123456789ABCDEF0123456789"
            hostile.hashes.append(upper)
            try hostile.victims.plant(
                root.appendingPathComponent("\(upper).safetensors"), fill: 0x16)

            let long = String(repeating: "0123456789abcdef", count: 4)
            hostile.hashes.append(long)
            try hostile.victims.plant(
                root.appendingPathComponent("\(long).safetensors"), fill: 0x17)

            hostile.namesInRoot = [
                "model-00001-of-00008.safetensors", ".safetensors", "\(upper).safetensors",
                "\(long).safetensors",
            ]

            let raw = try RawDB(root: root)
            for hash in hostile.hashes {
                try #require(!hash.contains("'"), "INVALID: fixture value needs SQL quoting")
                try raw.require(
                    """
                    INSERT INTO cache_entries (hash, token_count, file_size, created_at)
                    VALUES ('\(hash)', 307, \(2 * cap), 2440587.5)
                    """)
            }
            try hostile.victims.requirePlanted()
            let planted = Set(
                try RawDB(root: root).rows("SELECT hash FROM cache_entries").map { $0[0] ?? "" })
            try #require(
                planted.isSuperset(of: hostile.hashes), "INVALID: a hostile row was not planted")
            return hostile
        }

        private static func invalidHashLines(_ log: String) -> [Substring] {
            log.split(separator: "\n").filter {
                $0.hasPrefix("[vmlx][cache/disk-index] dropped row with an invalid hash")
            }
        }

        private static func invalidKeyLines(_ log: String) -> [Substring] {
            log.split(separator: "\n").filter {
                $0.hasPrefix("[vmlx][cache/disk-index] forgot companion with an invalid key")
            }
        }

        /// After any index-driven deleter: every victim is intact, only the
        /// control rows are left and counted, and a VALID hash still works.
        private static func expectOnlyControlsSurvive(
            disk: DiskCache, root: URL, hostile: HostileHashes, controls: [[Int]],
            modelKey: String, after what: String,
            sourceLocation: SourceLocation = #_sourceLocation
        ) throws {
            hostile.victims.expectIntact(after: what, sourceLocation: sourceLocation)
            let rows = try Support.indexedRows(root)
            let controlHashes = controls.map { DiskCache.hashTokens($0, modelKey: modelKey) }
                .sorted()
            #expect(
                rows.map(\.hash) == controlHashes,
                "rows after \(what): \(rows.map(\.hash))", sourceLocation: sourceLocation)
            let controlBytes = controlHashes.reduce(Int64(0)) {
                $0 + Support.fileBytes(Support.payloadURL(root, $1))
            }
            #expect(
                controlBytes > 0, "INVALID: no control payload is on disk",
                sourceLocation: sourceLocation)
            #expect(
                disk.usageBytes() == controlBytes, "usage still counts a hostile row after \(what)",
                sourceLocation: sourceLocation)
            try Support.expectUsageMatchesDisk(
                disk, root: root, foreign: hostile.namesInRoot, sourceLocation: sourceLocation)
            for tokens in controls {
                #expect(
                    disk.touchRecency(tokens: tokens, at: Date()),
                    "touchRecency of a valid hash after \(what)", sourceLocation: sourceLocation)
                #expect(
                    disk.fetch(tokens: tokens) != nil, "fetch of a valid hash after \(what)",
                    sourceLocation: sourceLocation)
            }
        }

        // MARK: - R1

        /// The coordinator's index quota pass: `quotaEntries()` → planner →
        /// `removeQuotaEntries(hashes:)`.
        @Test func hostileRowHashIsNeverAPathInTheCoordinatorQuotaPass() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("r1-quota")
                let outside = Self.makeRoot("r1-quota-outside")
                defer {
                    try? FileManager.default.removeItem(at: root)
                    try? FileManager.default.removeItem(at: outside)
                }
                let modelKey = "index-value-r1-quota"
                let coordinator = Self.coordinator(root: root, modelKey: modelKey)
                let disk = try #require(coordinator.diskCache)
                try #require(disk.indexHasV2Columns)
                let controls = [Self.tokens(301, seed: 9_001), Self.tokens(517, seed: 9_002)]
                for tokens in controls {
                    coordinator.storePersistentBoundary(
                        tokens: tokens, diskArrays: Self.kv(), ssmStates: nil)
                }
                let hostile = try Self.plantHostileHashRows(root: root, outside: outside)
                try #require(
                    disk.usageBytes() > Self.cap, "INVALID: the fixture is not over the cap")

                DiskCache.resetRateLimitedReportsForTesting()
                let (_, log) = try Support.capturingStandardError {
                    coordinator.enforceCombinedDiskQuota()
                }

                try Self.expectOnlyControlsSurvive(
                    disk: disk, root: root, hostile: hostile, controls: controls,
                    modelKey: modelKey, after: "the coordinator's index quota pass")
                #expect(Self.invalidHashLines(log).count == hostile.hashes.count, "\(log)")
                #expect(
                    disk.snapshotStats().evictions == 0, "a dropped hostile row is not an eviction")
                try Self.expectThePassStillEvicts(
                    coordinator: coordinator, disk: disk, oldest: controls[0])
            }
        }

        /// The positive control of the quota-pass tests: the pass that left
        /// every victim alone is one that CAN delete. Real entries that do
        /// not fit are evicted, oldest first, and counted.
        private static func expectThePassStillEvicts(
            coordinator: CacheCoordinator, disk: DiskCache, oldest: [Int],
            sourceLocation: SourceLocation = #_sourceLocation
        ) throws {
            let before = disk.snapshotStats().evictions
            // Three of these do not fit under the cap together.
            for seed in 0 ..< 3 {
                Thread.sleep(forTimeInterval: 0.02)
                coordinator.storePersistentBoundary(
                    tokens: Self.tokens(1_291 + seed, seed: 9_990 + seed),
                    diskArrays: Self.kv(100_003), ssmStates: nil)
            }
            #expect(
                disk.snapshotStats().evictions > before,
                "INVALID: the quota pass evicted nothing when real entries did not fit",
                sourceLocation: sourceLocation)
            #expect(
                disk.fetch(tokens: oldest) == nil, "INVALID: the oldest real entry was not evicted",
                sourceLocation: sourceLocation)
            #expect(disk.usageBytes() <= cap, sourceLocation: sourceLocation)
        }

        /// `DiskCache`'s own quota on a direct store: `_evictIfNeededLocked`.
        @Test func hostileRowHashIsNeverAPathInTheStandaloneEviction() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("r1-standalone")
                let outside = Self.makeRoot("r1-standalone-outside")
                defer {
                    try? FileManager.default.removeItem(at: root)
                    try? FileManager.default.removeItem(at: outside)
                }
                let modelKey = "index-value-r1-standalone"
                let disk = DiskCache(
                    cacheDir: root, maxSizeBytes: Int(Self.cap), modelKey: modelKey)
                let controls = [Self.tokens(301, seed: 9_011), Self.tokens(517, seed: 9_012)]
                disk.store(tokens: controls[0], arrays: Self.kv())
                let hostile = try Self.plantHostileHashRows(root: root, outside: outside)

                // The store whose standalone pass finds the index over its cap.
                disk.store(tokens: controls[1], arrays: Self.kv())

                try Self.expectOnlyControlsSurvive(
                    disk: disk, root: root, hostile: hostile, controls: controls,
                    modelKey: modelKey, after: "DiskCache's standalone eviction")
                #expect(
                    disk.snapshotStats().evictions == 0, "a dropped hostile row is not an eviction")
            }
        }

        /// `clear()` already validated the row's hash; pinned with the others.
        @Test func hostileRowHashIsNeverAPathInClear() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("r1-clear")
                let outside = Self.makeRoot("r1-clear-outside")
                defer {
                    try? FileManager.default.removeItem(at: root)
                    try? FileManager.default.removeItem(at: outside)
                }
                let modelKey = "index-value-r1-clear"
                let disk = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                let control = Self.tokens(517, seed: 9_021)
                disk.store(tokens: control, arrays: Self.kv(), enforceQuota: false)
                let controlURL = Support.payloadURL(
                    root, DiskCache.hashTokens(control, modelKey: modelKey))
                try #require(Support.fileBytes(controlURL) > 0)
                let hostile = try Self.plantHostileHashRows(root: root, outside: outside)

                disk.clear()

                hostile.victims.expectIntact(after: "clear()")
                #expect(Support.fileBytes(controlURL) == 0, "INVALID: clear() left its own payload")
                #expect(try Support.indexedRows(root).isEmpty)
                #expect(disk.usageBytes() == 0)
                try Support.expectIndexNamesEveryPublishedFile(root, foreign: hostile.namesInRoot)
            }
        }

        /// The import (`reconcileCompanionAccounting`) only ever lstat'ed a
        /// row's path, but it kept the row: the bytes it claims stay counted
        /// for good and every later quota pass is offered it again.
        @Test func hostileRowHashIsDroppedByTheImport() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("r1-import")
                let outside = Self.makeRoot("r1-import-outside")
                defer {
                    try? FileManager.default.removeItem(at: root)
                    try? FileManager.default.removeItem(at: outside)
                }
                let modelKey = "index-value-r1-import"
                // Large enough that no quota pass runs: the import alone acts.
                let coordinator = Self.coordinator(
                    root: root, capBytes: 1 << 30, modelKey: modelKey)
                let disk = try #require(coordinator.diskCache)
                let controls = [Self.tokens(301, seed: 9_031), Self.tokens(1_003, seed: 9_032)]
                for tokens in controls {
                    coordinator.storePersistentBoundary(
                        tokens: tokens, diskArrays: Self.kv(), ssmStates: nil)
                }
                let hostile = try Self.plantHostileHashRows(root: root, outside: outside)

                DiskCache.resetRateLimitedReportsForTesting()
                let (committed, log) = try Support.capturingStandardError {
                    coordinator.reconcileDiskAccounting()
                }
                #expect(committed, "the import did not commit")

                try Self.expectOnlyControlsSurvive(
                    disk: disk, root: root, hostile: hostile, controls: controls,
                    modelKey: modelKey, after: "the import")
                #expect(Self.invalidHashLines(log).count == hostile.hashes.count, "\(log)")
                // Idempotent: nothing left to drop.
                let again = try #require(disk.reconcileCompanionAccounting(companions: []))
                #expect(!again.changedAnything, "\(again)")
            }
        }

        /// One line per distinct value, at most eight distinct values per
        /// process: a corrupt index with a million bad rows is not a million
        /// lines, and the same bad row met again says nothing new.
        @Test func invalidIndexValueReportIsRateLimited() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("r1-rate")
                defer { try? FileManager.default.removeItem(at: root) }
                let disk = DiskCache(
                    cacheDir: root, maxSizeBytes: 1 << 30, modelKey: "index-value-rate")
                try #require(disk.indexHasV2Columns)
                func plant() throws {
                    let raw = try RawDB(root: root)
                    for index in 0 ..< 12 {
                        try raw.require(
                            "INSERT INTO cache_entries (hash, token_count, file_size) VALUES ('not-a-hash-\(index)', 7, 11)"
                        )
                    }
                }

                DiskCache.resetRateLimitedReportsForTesting()
                try plant()
                let (_, first) = try Support.capturingStandardError {
                    disk.reconcileCompanionAccounting(companions: [])
                }
                #expect(try Support.indexedRows(root).isEmpty)
                #expect(Self.invalidHashLines(first).count == 8, "\(first)")

                try plant()
                let (_, second) = try Support.capturingStandardError {
                    disk.reconcileCompanionAccounting(companions: [])
                }
                #expect(try Support.indexedRows(root).isEmpty)
                #expect(Self.invalidHashLines(second).isEmpty, "\(second)")
            }
        }

        // MARK: - R2 fixtures: hostile `companion_key` / `legacy_companions.key`

        enum Placement: String, CaseIterable, Sendable {
            /// `cache_entries.companion_key` of a real row.
            case linked
            /// `legacy_companions.key`.
            case legacy
        }

        private struct HostileKeys {
            var keys: [String] = []
            var victims = Victims()
            var foreignNames: Set<String> = []
        }

        /// The files `ssm-<key>.safetensors` / `ssm-<key>.json` would address
        /// for each hostile key, then the records that name those keys.
        private static func plantHostileCompanionKeys(
            root: URL, placement: Placement, controls: [[Int]], modelKey: String
        ) throws -> HostileKeys {
            var hostile = HostileKeys()
            let dir = Support.companionDir(root)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

            // The very names `companionClearAndSweepNeverTouchForeignEntries`
            // says are untouchable.
            hostile.keys.append("notes")
            try hostile.victims.plant(
                dir.appendingPathComponent("ssm-notes.safetensors"), fill: 0x21)
            try hostile.victims.plant(dir.appendingPathComponent("ssm-notes.json"), fill: 0x22)

            // `ssm-../x.safetensors`: a directory named `ssm-..` is enough.
            hostile.keys.append("../x")
            try hostile.victims.plant(
                dir.appendingPathComponent("ssm-../x.safetensors"), fill: 0x23)
            try hostile.victims.plant(dir.appendingPathComponent("ssm-../x.json"), fill: 0x24)

            // `ssm-x/../../y.safetensors`: with a planted `ssm-x/` directory
            // this leaves the companion directory and lands in the root.
            hostile.keys.append("x/../../y")
            try FileManager.default.createDirectory(
                at: dir.appendingPathComponent("ssm-x"), withIntermediateDirectories: true)
            try hostile.victims.plant(root.appendingPathComponent("y.safetensors"), fill: 0x25)
            try hostile.victims.plant(root.appendingPathComponent("y.json"), fill: 0x26)

            let short = "0123456789abcdef0123456789abcdef"
            hostile.keys.append(short)
            try hostile.victims.plant(
                dir.appendingPathComponent("ssm-\(short).safetensors"), fill: 0x27)
            try hostile.victims.plant(dir.appendingPathComponent("ssm-\(short).json"), fill: 0x28)

            let upper = String(repeating: "ABCDEF0123456789", count: 4)
            hostile.keys.append(upper)
            try hostile.victims.plant(
                dir.appendingPathComponent("ssm-\(upper).safetensors"), fill: 0x29)
            try hostile.victims.plant(dir.appendingPathComponent("ssm-\(upper).json"), fill: 0x2A)

            hostile.foreignNames = [
                "ssm-notes.safetensors", "ssm-notes.json", "y.safetensors",
                "ssm-\(short).safetensors", "ssm-\(short).json",
                "ssm-\(upper).safetensors", "ssm-\(upper).json",
            ]

            try #require(controls.count >= hostile.keys.count, "INVALID: one control row per key")
            let raw = try RawDB(root: root)
            for (key, tokens) in zip(hostile.keys, controls) {
                switch placement {
                case .linked:
                    let hash = DiskCache.hashTokens(tokens, modelKey: modelKey)
                    try raw.require(
                        """
                        UPDATE cache_entries SET companion_key = '\(key)', companion_bytes = \(2 * cap)
                        WHERE hash = '\(hash)'
                        """)
                case .legacy:
                    try raw.require(
                        """
                        INSERT INTO legacy_companions (key, bytes, modified)
                        VALUES ('\(key)', \(2 * cap), 0)
                        """)
                }
            }
            try hostile.victims.requirePlanted()
            let named = try Self.namedCompanionKeys(root)
            try #require(
                named.isSuperset(of: hostile.keys), "INVALID: a hostile key was not planted")
            return hostile
        }

        private static func namedCompanionKeys(_ root: URL) throws -> Set<String> {
            Set(try Support.indexedRows(root).compactMap(\.companionKey))
                .union(try Support.legacyRows(root).keys)
        }

        private static func controlTokens(_ base: Int) -> [[Int]] {
            [301, 517, 1_003, 1_291, 307].enumerated().map { Self.tokens($1, seed: base + $0) }
        }

        /// After any index-driven deleter: every victim is intact, no record
        /// names a hostile key or counts its bytes, every control row is
        /// still there and still served.
        private static func expectHostileKeysForgotten(
            disk: DiskCache, root: URL, hostile: HostileKeys, controls: [[Int]], modelKey: String,
            after what: String, sourceLocation: SourceLocation = #_sourceLocation
        ) throws {
            hostile.victims.expectIntact(after: what, sourceLocation: sourceLocation)
            let named = try Self.namedCompanionKeys(root)
            #expect(
                named.isDisjoint(with: hostile.keys),
                "still named after \(what): \(named.intersection(hostile.keys).sorted())",
                sourceLocation: sourceLocation)
            let rows = try Support.indexedRows(root)
            #expect(
                rows.map(\.hash)
                    == controls.map { DiskCache.hashTokens($0, modelKey: modelKey) }.sorted(),
                "a control row was lost to \(what)", sourceLocation: sourceLocation)
            #expect(
                disk.usageBytes() < cap, "usage still counts a hostile key after \(what)",
                sourceLocation: sourceLocation)
            try Support.expectUsageMatchesDisk(
                disk, root: root, foreign: hostile.foreignNames, sourceLocation: sourceLocation)
            for tokens in controls {
                #expect(
                    disk.fetch(tokens: tokens) != nil, "a control payload was lost to \(what)",
                    sourceLocation: sourceLocation)
            }
        }

        // MARK: - R2

        /// The coordinator's index quota pass:
        /// `SSMCompanionDiskStore.removeQuotaEntries(hashes:)`.
        @Test(arguments: Placement.allCases)
        func hostileCompanionKeyIsNeverAPathInTheCoordinatorQuotaPass(_ placement: Placement) throws
        {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("r2-quota-\(placement.rawValue)")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-value-r2-quota-\(placement.rawValue)"
                let coordinator = Self.coordinator(root: root, modelKey: modelKey)
                let disk = try #require(coordinator.diskCache)
                try #require(disk.indexHasV2Columns)
                let controls = Self.controlTokens(9_100)
                for tokens in controls {
                    coordinator.storePersistentBoundary(
                        tokens: tokens, diskArrays: Self.kv(), ssmStates: nil)
                }
                let hostile = try Self.plantHostileCompanionKeys(
                    root: root, placement: placement, controls: controls, modelKey: modelKey)
                try #require(
                    disk.usageBytes() > Self.cap, "INVALID: the fixture is not over the cap")

                DiskCache.resetRateLimitedReportsForTesting()
                let (_, log) = try Support.capturingStandardError {
                    coordinator.enforceCombinedDiskQuota()
                }

                try Self.expectHostileKeysForgotten(
                    disk: disk, root: root, hostile: hostile, controls: controls,
                    modelKey: modelKey,
                    after: "the coordinator's index quota pass (\(placement.rawValue))")
                #expect(Self.invalidKeyLines(log).count == hostile.keys.count, "\(log)")
                try Self.expectThePassStillEvicts(
                    coordinator: coordinator, disk: disk, oldest: controls[0])
            }
        }

        /// The companion store's own cap on a direct write: `evictOverCap`.
        @Test(arguments: Placement.allCases)
        func hostileCompanionKeyIsNeverAPathInEvictOverCap(_ placement: Placement) throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("r2-overcap-\(placement.rawValue)")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-value-r2-overcap-\(placement.rawValue)"
                let coordinator = Self.coordinator(root: root, modelKey: modelKey)
                let disk = try #require(coordinator.diskCache)
                let companion = try #require(coordinator.ssmStateCache.diskStore)
                let controls = Self.controlTokens(9_200)
                for tokens in controls {
                    coordinator.storePersistentBoundary(
                        tokens: tokens, diskArrays: Self.kv(), ssmStates: nil)
                }
                let hostile = try Self.plantHostileCompanionKeys(
                    root: root, placement: placement, controls: controls, modelKey: modelKey)
                try #require(
                    disk.companionUsageBytes() > Self.cap, "INVALID: not over the companion cap")

                // A direct companion write, which applies the store's own cap.
                let direct = Self.tokens(523, seed: 9_290)
                try companion.store(
                    ssmStates: Self.recurrent(), tokens: direct, boundary: direct.count)

                let directKey = SSMCompanionDiskStore.keyFor(
                    tokens: direct, boundary: direct.count, modelKey: modelKey)
                #expect(
                    Support.companionBytes(root, directKey) > 0,
                    "the real companion was evicted to make up for bytes that name nothing")
                try Self.expectHostileKeysForgotten(
                    disk: disk, root: root, hostile: hostile, controls: controls,
                    modelKey: modelKey,
                    after: "evictOverCap (\(placement.rawValue))")
            }
        }

        /// The import lstat's `ssm-<key>.…` for every companion the index
        /// names and the walk did not list.
        @Test(arguments: Placement.allCases)
        func hostileCompanionKeyIsForgottenByTheImport(_ placement: Placement) throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("r2-import-\(placement.rawValue)")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-value-r2-import-\(placement.rawValue)"
                let coordinator = Self.coordinator(
                    root: root, capBytes: 1 << 30, modelKey: modelKey)
                let disk = try #require(coordinator.diskCache)
                let controls = Self.controlTokens(9_300)
                for tokens in controls {
                    coordinator.storePersistentBoundary(
                        tokens: tokens, diskArrays: Self.kv(), ssmStates: nil)
                }
                let hostile = try Self.plantHostileCompanionKeys(
                    root: root, placement: placement, controls: controls, modelKey: modelKey)

                DiskCache.resetRateLimitedReportsForTesting()
                let (committed, log) = try Support.capturingStandardError {
                    coordinator.reconcileDiskAccounting()
                }
                #expect(committed, "the import did not commit")

                try Self.expectHostileKeysForgotten(
                    disk: disk, root: root, hostile: hostile, controls: controls,
                    modelKey: modelKey,
                    after: "the import (\(placement.rawValue))")
                #expect(Self.invalidKeyLines(log).count == hostile.keys.count, "\(log)")
            }
        }

        /// `clear()` removes from a listing, by the name predicate; pinned.
        @Test(arguments: Placement.allCases)
        func hostileCompanionKeyIsNeverAPathInClear(_ placement: Placement) throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("r2-clear-\(placement.rawValue)")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-value-r2-clear-\(placement.rawValue)"
                let coordinator = Self.coordinator(
                    root: root, capBytes: 1 << 30, modelKey: modelKey)
                let disk = try #require(coordinator.diskCache)
                let controls = Self.controlTokens(9_400)
                for tokens in controls {
                    coordinator.storePersistentBoundary(
                        tokens: tokens, diskArrays: Self.kv(), ssmStates: nil)
                }
                let hostile = try Self.plantHostileCompanionKeys(
                    root: root, placement: placement, controls: controls, modelKey: modelKey)

                coordinator.clear()

                hostile.victims.expectIntact(after: "clear() (\(placement.rawValue))")
                #expect(try Support.indexedRows(root).isEmpty)
                #expect(try Support.legacyRows(root).isEmpty)
                #expect(disk.usageBytes() == 0)
            }
        }

        /// `publishedEntryState(key:in:)` is what the import and the failed-
        /// write report lstat through.
        @Test func invalidCompanionKeyNamesNothing() throws {
            let root = Self.makeRoot("r2-state")
            defer { try? FileManager.default.removeItem(at: root) }
            var victims = Victims()
            try victims.plant(root.appendingPathComponent("ssm-notes.safetensors"), fill: 0x31)
            try victims.plant(root.appendingPathComponent("ssm-notes.json"), fill: 0x32)
            let valid = SSMCompanionDiskStore.keyFor(
                tokens: Self.tokens(307, seed: 9_500), boundary: 307, modelKey: "index-value-state")
            try #require(DiskCache.isLowercaseHex(valid, count: SSMCompanionDiskStore.keyLength))
            try victims.plant(root.appendingPathComponent("ssm-\(valid).safetensors"), fill: 0x33)

            guard case .absent = SSMCompanionDiskStore.publishedEntryState(key: "notes", in: root)
            else {
                Issue.record("an invalid key was looked up on disk")
                return
            }
            #expect(SSMCompanionDiskStore.publishedEntry(key: "notes", in: root) == nil)
            // The control: a real key is found.
            let found = SSMCompanionDiskStore.publishedEntry(key: valid, in: root)
            #expect(found?.bytes == Int64(4_150), "\(String(describing: found))")
            victims.expectIntact(after: "publishedEntryState")
        }

        // MARK: - R3: "could not read" is not "corrupt"

        private static func makeUnreadable(_ url: URL) throws {
            try #require(chmod(url.path, 0) == 0, "INVALID: chmod failed")
            let fd = open(url.path, O_RDONLY)
            if fd >= 0 { close(fd) }
            try #require(fd < 0, "INVALID: chmod 000 had no effect (root / filesystem)")
        }

        /// By `lstat`, so it answers for a file that cannot be opened, and
        /// nil for a link.
        private static func regularFileSize(_ url: URL) -> Int64? {
            if case .regularFile(let size, _) = DiskCache.pathState(at: url) { return size }
            return nil
        }

        private static func rowExists(_ root: URL, _ hash: String) throws -> Bool {
            try Support.indexedRows(root).contains { $0.hash == hash }
        }

        /// (a) The sweep at open read "could not open" as "incomplete".
        @Test func unreadablePayloadSurvivesTheOpenSweep() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("r3-open")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-value-r3-open"
                let tokens = Self.tokens(517, seed: 9_601)
                let hash = DiskCache.hashTokens(tokens, modelKey: modelKey)
                let url = Support.payloadURL(root, hash)
                do {
                    let writer = DiskCache(
                        cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                    writer.store(tokens: tokens, arrays: Self.kv(), enforceQuota: false)
                }
                let bytes = try Data(contentsOf: url)
                try Self.makeUnreadable(url)
                defer { chmod(url.path, 0o644) }

                let reopened = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)

                #expect(
                    Self.regularFileSize(url) == Int64(bytes.count),
                    "VICTIM: a valid payload that could not be read was removed at open")
                #expect(try Self.rowExists(root, hash))

                chmod(url.path, 0o644)
                #expect((try? Data(contentsOf: url)) == bytes)
                #expect(reopened.fetch(tokens: tokens) != nil, "nothing was lost")
            }
        }

        /// (b) + (c) `fetch` deleted the payload it could not read.
        @Test func unreadablePayloadIsAMissThatLeavesFileAndRowAlone() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("r3-fetch")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-value-r3-fetch"
                let tokens = Self.tokens(1_003, seed: 9_611)
                let hash = DiskCache.hashTokens(tokens, modelKey: modelKey)
                let url = Support.payloadURL(root, hash)
                let disk = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                disk.store(tokens: tokens, arrays: Self.kv(), enforceQuota: false)
                let bytes = try Data(contentsOf: url)
                try Self.makeUnreadable(url)
                defer { chmod(url.path, 0o644) }

                DiskCache.resetRateLimitedReportsForTesting()
                let (first, log) = try Support.capturingStandardError { disk.fetch(tokens: tokens) }
                #expect(first == nil)
                #expect(disk.fetch(tokens: tokens) == nil)
                guard case .regularFile(let size, _) = DiskCache.pathState(at: url) else {
                    Issue.record(
                        "VICTIM: a valid payload that could not be read was removed by fetch")
                    return
                }
                #expect(size == Int64(bytes.count))
                #expect(try Self.rowExists(root, hash))
                let stats = disk.snapshotStats()
                #expect(stats.unreadablePayloadFetches == 2)
                #expect(stats.misses == 2)
                #expect(disk.usageBytes() == Int64(bytes.count), "the row still counts its payload")
                #expect(
                    log.split(separator: "\n").filter {
                        $0.hasPrefix("[vmlx][cache/disk] fetch could not read ")
                    }.count == 1, "\(log)")

                // (c) Permissions restored: nothing was lost.
                chmod(url.path, 0o644)
                let restored = try #require(disk.fetch(tokens: tokens))
                #expect(restored["data"]?.shape == [1_024])
                #expect(restored["data"]?.sum().item(Float.self) == 1_024)
                #expect((try? Data(contentsOf: url)) == bytes)
                #expect(disk.snapshotStats().unreadablePayloadFetches == 2)
            }
        }

        /// A loader that could not OPEN the file has not shown it corrupt; a
        /// loader that opened it and could not decode it has.
        @Test func onlyADecodeErrorRemovesAPayloadTheLoaderRejected() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("r3-loader")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-value-r3-loader"
                let transient = Self.tokens(517, seed: 9_621)
                let corrupt = Self.tokens(1_291, seed: 9_622)
                let disk = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                for tokens in [transient, corrupt] {
                    disk.store(tokens: tokens, arrays: Self.kv(), enforceQuota: false)
                }
                let transientHash = DiskCache.hashTokens(transient, modelKey: modelKey)
                let corruptHash = DiskCache.hashTokens(corrupt, modelKey: modelKey)

                disk.loadFaultForTesting = { url in
                    throw MLXError.caught("[load_safetensors] Failed to open \(url.path)")
                }
                #expect(disk.fetch(tokens: transient) == nil)
                disk.loadFaultForTesting = nil
                #expect(
                    Support.fileBytes(Support.payloadURL(root, transientHash)) > 0,
                    "VICTIM: a payload the loader could not open was removed")
                let transientRowKept = try Self.rowExists(root, transientHash)
                #expect(transientRowKept)
                #expect(disk.snapshotStats().unreadablePayloadFetches == 1)
                #expect(disk.fetch(tokens: transient) != nil, "nothing was lost")

                // The control: a decode error on a file that opens and whose
                // header parses is still removed, row and all.
                disk.loadFaultForTesting = { _ in
                    throw MLXError.caught("[load_safetensors] Invalid json metadata")
                }
                #expect(disk.fetch(tokens: corrupt) == nil)
                disk.loadFaultForTesting = nil
                #expect(Support.fileBytes(Support.payloadURL(root, corruptHash)) == 0)
                let corruptRowKept = try Self.rowExists(root, corruptHash)
                #expect(!corruptRowKept)
                #expect(disk.snapshotStats().unreadablePayloadFetches == 1)
            }
        }

        /// The positive control for both paths: a payload of OURS that really
        /// is short of the bytes its header declares still goes.
        @Test func truncatedPayloadIsStillRemovedByFetchAndByTheOpenSweep() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("r3-truncated")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-value-r3-truncated"
                let byFetch = Self.tokens(517, seed: 9_631)
                let byOpen = Self.tokens(1_003, seed: 9_632)
                let disk = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                for tokens in [byFetch, byOpen] {
                    disk.store(tokens: tokens, arrays: Self.kv(), enforceQuota: false)
                }
                func truncate(_ tokens: [Int]) throws -> URL {
                    let url = Support.payloadURL(
                        root, DiskCache.hashTokens(tokens, modelKey: modelKey))
                    let size = Support.fileBytes(url)
                    try #require(size > 64)
                    let handle = try FileHandle(forWritingTo: url)
                    try handle.truncate(atOffset: UInt64(size - 13))
                    try handle.close()
                    try #require(
                        DiskCache.inspectSafetensors(url: url) == .shortOrMalformed,
                        "INVALID: the fixture is not positively short")
                    return url
                }

                let fetched = try truncate(byFetch)
                #expect(disk.fetch(tokens: byFetch) == nil)
                #expect(DiskCache.pathState(at: fetched) == .missing)
                #expect(
                    try !Self.rowExists(root, DiskCache.hashTokens(byFetch, modelKey: modelKey)))
                #expect(disk.snapshotStats().unreadablePayloadFetches == 0)

                let opened = try truncate(byOpen)
                _ = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                #expect(DiskCache.pathState(at: opened) == .missing)
            }
        }

        @Test func headerInspectionTellsUnreadableFromMalformed() throws {
            let root = Self.makeRoot("r3-inspect")
            defer { try? FileManager.default.removeItem(at: root) }
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

            let header = #"{"kv_0_keys":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#
            var complete = Data()
            var length = UInt64(header.utf8.count).littleEndian
            complete.append(Data(bytes: &length, count: 8))
            complete.append(Data(header.utf8))
            let headerOnly = complete
            complete.append(Data(repeating: 0, count: 16))

            let good = root.appendingPathComponent("good.safetensors")
            try complete.write(to: good)
            #expect(DiskCache.inspectSafetensors(url: good) == .complete)
            #expect(DiskCache.isCompleteSafetensors(url: good))

            for (name, data) in [
                ("short", headerOnly), ("empty", Data()), ("seven", Data(repeating: 1, count: 7)),
                ("junk", Data(repeating: 0x7B, count: 4_099)),
            ] {
                let url = root.appendingPathComponent("\(name).safetensors")
                try data.write(to: url)
                #expect(DiskCache.inspectSafetensors(url: url) == .shortOrMalformed, "\(name)")
                #expect(!DiskCache.isCompleteSafetensors(url: url), "\(name)")
            }

            try Self.makeUnreadable(good)
            defer { chmod(good.path, 0o644) }
            #expect(DiskCache.inspectSafetensors(url: good) == .unreadable(errno: EACCES))
            #expect(
                DiskCache.inspectSafetensors(url: root.appendingPathComponent("absent.safetensors"))
                    == .unreadable(errno: ENOENT))
        }

        // MARK: - O1: an occupied `.partial-` name is not written through

        @Test func storeRefusesToWriteThroughALinkAtItsTemporaryName() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("o1-kv")
                let outside = Self.makeRoot("o1-kv-outside")
                defer {
                    try? FileManager.default.removeItem(at: root)
                    try? FileManager.default.removeItem(at: outside)
                }
                let modelKey = "index-value-o1-kv"
                let disk = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                let tokens = Self.tokens(517, seed: 9_701)
                let control = Self.tokens(1_003, seed: 9_702)
                let hash = DiskCache.hashTokens(tokens, modelKey: modelKey)
                let partial = root.appendingPathComponent("\(hash).partial-1A2B3C4D.safetensors")
                try #require(DiskCache.isUnpublishedPayloadName(partial.lastPathComponent))

                var victims = Victims()
                let target = outside.appendingPathComponent("somebody-elses.safetensors")
                try victims.plant(target, fill: 0x41)
                try FileManager.default.createSymbolicLink(at: partial, withDestinationURL: target)

                disk.temporaryURLForTesting = { _ in partial }
                disk.store(tokens: tokens, arrays: Self.kv(), enforceQuota: false)
                disk.temporaryURLForTesting = nil

                victims.expectIntact(after: "a store whose temporary name is a link")
                #expect(
                    (try? FileManager.default.destinationOfSymbolicLink(atPath: partial.path))
                        == target.path,
                    "the link itself is not ours to remove")
                #expect(try Support.indexedRows(root).isEmpty, "a refused store left a row")
                #expect(Support.fileBytes(Support.payloadURL(root, hash)) == 0)
                #expect(disk.fetch(tokens: tokens) == nil)

                // The control: the next store, under an ordinary name, works.
                disk.store(tokens: control, arrays: Self.kv(), enforceQuota: false)
                #expect(disk.fetch(tokens: control) != nil)
            }
        }

        @Test func companionStoreRefusesToWriteThroughALinkAtItsTemporaryName() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("o1-companion")
                let outside = Self.makeRoot("o1-companion-outside")
                defer {
                    try? FileManager.default.removeItem(at: root)
                    try? FileManager.default.removeItem(at: outside)
                }
                let modelKey = "index-value-o1-companion"
                let coordinator = Self.coordinator(
                    root: root, capBytes: 1 << 30, modelKey: modelKey)
                let companion = try #require(coordinator.ssmStateCache.diskStore)
                let tokens = Self.tokens(517, seed: 9_711)
                let key = SSMCompanionDiskStore.keyFor(
                    tokens: tokens, boundary: tokens.count, modelKey: modelKey)
                let dir = Support.companionDir(root)
                let partial = dir.appendingPathComponent("ssm-\(key).partial-1A2B3C4D.safetensors")
                try #require(
                    SSMCompanionDiskStore.isUnpublishedTensorName(partial.lastPathComponent))

                var victims = Victims()
                let target = outside.appendingPathComponent("somebody-elses.safetensors")
                try victims.plant(target, fill: 0x42)
                try FileManager.default.createSymbolicLink(at: partial, withDestinationURL: target)

                companion.temporaryURLForTesting = { _ in partial }
                #expect(throws: (any Error).self) {
                    _ = try companion.store(
                        ssmStates: Self.recurrent(), tokens: tokens, boundary: tokens.count,
                        enforceQuota: false)
                }
                companion.temporaryURLForTesting = nil

                victims.expectIntact(after: "a companion store whose temporary name is a link")
                let left = try Self.listing(dir)
                #expect(left == [partial.lastPathComponent], "\(left.sorted())")
                #expect(try Support.legacyRows(root).isEmpty)

                // The control.
                _ = try companion.store(
                    ssmStates: Self.recurrent(), tokens: tokens, boundary: tokens.count,
                    enforceQuota: false)
                #expect(Support.companionBytes(root, key) > 0)
            }
        }

        // MARK: - O2: publishing is one rename

        /// The old valid payload used to be unlinked BEFORE the move, so a
        /// move that failed had already cost the entry.
        @Test func failedPublishKeepsTheOldValidPayload() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("o2-kv")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-value-o2-kv"
                let disk = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                let tokens = Self.tokens(517, seed: 9_801)
                let hash = DiskCache.hashTokens(tokens, modelKey: modelKey)
                let url = Support.payloadURL(root, hash)
                disk.store(tokens: tokens, arrays: Self.kv(), enforceQuota: false)
                let bytes = try Data(contentsOf: url)
                let rowBefore = try #require(try Support.indexedRows(root).first)

                struct InjectedFault: Error {}
                disk.forgetValidatedFiles()
                disk.publishFaultForTesting = { throw InjectedFault() }
                disk.store(tokens: tokens, arrays: Self.kv(2_048), enforceQuota: false)
                disk.publishFaultForTesting = nil

                #expect(
                    (try? Data(contentsOf: url)) == bytes,
                    "VICTIM: the old valid payload was gone when the publish failed")
                #expect(try Support.indexedRows(root) == [rowBefore])
                #expect(
                    try Self.listing(root).allSatisfy { !$0.contains(".partial-") },
                    "a failed publish left its partial behind")
                #expect(disk.fetch(tokens: tokens)?["data"]?.shape == [1_024])
                try Support.expectUsageMatchesDisk(disk, root: root)

                // The control: an ordinary re-store replaces the payload.
                disk.forgetValidatedFiles()
                disk.store(tokens: tokens, arrays: Self.kv(2_048), enforceQuota: false)
                #expect(disk.fetch(tokens: tokens)?["data"]?.shape == [2_048])
                try Support.expectUsageMatchesDisk(disk, root: root)
            }
        }

        /// `rename(2)` itself: it replaces a regular file, replaces a symlink
        /// and not its target, and refuses a directory.
        @Test func renameReplacesALinkItselfAndRefusesADirectory() throws {
            let root = Self.makeRoot("o2-rename")
            let outside = Self.makeRoot("o2-rename-outside")
            defer {
                try? FileManager.default.removeItem(at: root)
                try? FileManager.default.removeItem(at: outside)
            }
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            func source(_ fill: UInt8) throws -> URL {
                let url = root.appendingPathComponent("source-\(fill)")
                try Data(repeating: fill, count: 1_031).write(to: url)
                return url
            }

            let file = root.appendingPathComponent("file")
            try Data(repeating: 0x01, count: 2_053).write(to: file)
            #expect(DiskCache.renameFile(from: try source(0x51), to: file) == 0)
            #expect((try? Data(contentsOf: file)) == Data(repeating: 0x51, count: 1_031))

            var victims = Victims()
            let target = outside.appendingPathComponent("target")
            try victims.plant(target, fill: 0x52)
            let link = root.appendingPathComponent("link")
            try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
            #expect(DiskCache.renameFile(from: try source(0x53), to: link) == 0)
            victims.expectIntact(after: "a rename over a link to it")
            #expect(Self.regularFileSize(link) == 1_031, "the link itself was not replaced")

            let directory = root.appendingPathComponent("directory")
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            try victims.plant(directory.appendingPathComponent("inner.bin"), fill: 0x54)
            let refused = try source(0x55)
            #expect(DiskCache.renameFile(from: refused, to: directory) == EISDIR)
            victims.expectIntact(after: "a rename over a directory")
            #expect(Support.fileBytes(refused) == 1_031, "the source of a refused rename stays")
        }

        // MARK: - O3: a touch does not follow a link

        @Test func companionTouchNeverFollowsALink() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("o3-touch")
                let outside = Self.makeRoot("o3-touch-outside")
                defer {
                    try? FileManager.default.removeItem(at: root)
                    try? FileManager.default.removeItem(at: outside)
                }
                let modelKey = "index-value-o3"
                let coordinator = Self.coordinator(
                    root: root, capBytes: 1 << 30, modelKey: modelKey)
                let companion = try #require(coordinator.ssmStateCache.diskStore)
                let tokens = Self.tokens(517, seed: 9_901)
                let key = SSMCompanionDiskStore.keyFor(
                    tokens: tokens, boundary: tokens.count, modelKey: modelKey)
                _ = try companion.store(
                    ssmStates: Self.recurrent(), tokens: tokens, boundary: tokens.count,
                    enforceQuota: false)
                let tensor = Support.companionURLs(root, key)[0]
                try #require(Support.fileBytes(tensor) > 0)
                // The control: an ordinary pair is touched.
                #expect(companion.touchRecency(tokens: tokens, boundary: tokens.count, at: Date()))

                // The tensor's name now holds a link to somebody else's file.
                var victims = Victims()
                let target = outside.appendingPathComponent("somebody-elses.safetensors")
                try victims.plant(target, fill: 0x61)
                try Support.age(target, by: 3_600)
                guard case .regularFile(_, let before) = DiskCache.pathState(at: target) else {
                    Issue.record("INVALID: the target is not on disk")
                    return
                }
                try FileManager.default.removeItem(at: tensor)
                try FileManager.default.createSymbolicLink(at: tensor, withDestinationURL: target)

                let touched = companion.touchRecency(
                    tokens: tokens, boundary: tokens.count, at: Date())

                #expect(!touched, "a link under the tensor's name is not an entry to refresh")
                guard case .regularFile(_, let after) = DiskCache.pathState(at: target) else {
                    Issue.record("VICTIM: the link's target is gone")
                    return
                }
                #expect(
                    after == before, "VICTIM: the touch followed the link and re-dated its target")
                victims.expectIntact(after: "a touch through a link")
            }
        }

        // MARK: - O4: a young partial may be another connection's store in flight

        @Test func youngPartialsSurviveAnOpenAndOldOnesDoNot() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("o4-young")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-value-o4"
                let dir = Support.companionDir(root)
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                let key = SSMCompanionDiskStore.keyFor(
                    tokens: Self.tokens(307, seed: 9_951), boundary: 307, modelKey: modelKey)
                let stems = [
                    root.appendingPathComponent("00112233445566778899aabbccddeeff.safetensors"),
                    dir.appendingPathComponent("ssm-\(key).safetensors"),
                ]
                var young: [URL: Data] = [:]
                var old: [URL] = []
                for stem in stems {
                    let youngURL = DiskCache.temporaryURL(for: stem)
                    let data = Data(repeating: 0xEE, count: 40_003)
                    try data.write(to: youngURL)
                    young[youngURL] = data
                    let oldURL = DiskCache.temporaryURL(for: stem)
                    try Data(repeating: 0xED, count: 40_009).write(to: oldURL)
                    try Support.age(oldURL, by: 11 * 60)
                    old.append(oldURL)
                }
                // In the future: says nothing about how long it has been there.
                let future = DiskCache.temporaryURL(for: stems[0])
                try Data(repeating: 0xEC, count: 1_031).write(to: future)
                try Support.age(future, by: -3_600)

                CacheCoordinator.resetImportedRootsForTesting()
                _ = Self.coordinator(root: root, capBytes: 1 << 30, modelKey: modelKey)

                for (url, data) in young {
                    #expect(
                        (try? Data(contentsOf: url)) == data,
                        "VICTIM: the in-flight partial \(url.lastPathComponent) was removed at open"
                    )
                }
                #expect(FileManager.default.fileExists(atPath: future.path))
                for url in old {
                    #expect(
                        !FileManager.default.fileExists(atPath: url.path),
                        "INVALID: the dead partial \(url.lastPathComponent) was not swept")
                }

                // Injectable: with no guard age the young ones are dead writes.
                DiskCache.sweepUnpublishedAndIncompleteFiles(in: root, unpublishedGuardAge: 0)
                SSMCompanionDiskStore.sweepUnpublishedFiles(in: dir, unpublishedGuardAge: 0)
                for url in young.keys {
                    #expect(!FileManager.default.fileExists(atPath: url.path))
                }
            }
        }

        // MARK: - O5: nothing is removed from a listing under a newer schema

        @Test func openSweepAndClearListingAreSkippedOnANewerSchema() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("o5-newer")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-value-o5"
                let indexed = Self.tokens(517, seed: 9_961)
                let rowless = Self.tokens(1_003, seed: 9_962)
                do {
                    let writer = DiskCache(
                        cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                    for tokens in [indexed, rowless] {
                        writer.store(tokens: tokens, arrays: Self.kv(), enforceQuota: false)
                    }
                }
                let indexedURL = Support.payloadURL(
                    root, DiskCache.hashTokens(indexed, modelKey: modelKey))
                let rowlessURL = Support.payloadURL(
                    root, DiskCache.hashTokens(rowless, modelKey: modelKey))
                try RawDB(root: root).require(
                    "DELETE FROM cache_entries WHERE hash = '\(DiskCache.hashTokens(rowless, modelKey: modelKey))'"
                )

                // What this build's sweeps WOULD remove: an old dead partial
                // and a payload-named file short of its declared bytes.
                var kept: [URL: Data] = [:]
                let partial = DiskCache.temporaryURL(for: rowlessURL)
                kept[partial] = Data(repeating: 0xEE, count: 40_003)
                let header = #"{"kv_0_keys":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#
                var short = Data()
                var length = UInt64(header.utf8.count).littleEndian
                short.append(Data(bytes: &length, count: 8))
                short.append(Data(header.utf8))
                let shortURL = root.appendingPathComponent(
                    "00112233445566778899aabbccddeeff.safetensors")
                kept[shortURL] = short
                for (url, data) in kept {
                    try data.write(to: url)
                    try Support.age(url, by: 11 * 60)
                }
                kept[rowlessURL] = try Data(contentsOf: rowlessURL)

                try RawDB(root: root).require("PRAGMA user_version = 7")
                let (newer, log) = try Support.capturingStandardError {
                    DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                }
                try #require(
                    newer.indexSchemaVersion == 7, "INVALID: the newer version was not read")
                for (url, data) in kept {
                    #expect(
                        (try? Data(contentsOf: url)) == data,
                        "VICTIM: \(url.lastPathComponent) was swept at open under a schema this build does not know"
                    )
                }
                #expect(log.contains("[vmlx][cache/disk] integrity sweep skipped: "), "\(log)")

                // The companion directory of the same root, as the
                // coordinator opens it.
                let dir = Support.companionDir(root)
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                let key = SSMCompanionDiskStore.keyFor(
                    tokens: indexed, boundary: indexed.count, modelKey: modelKey)
                let companionPartial = DiskCache.temporaryURL(
                    for: dir.appendingPathComponent("ssm-\(key).safetensors"))
                let companionPartialData = Data(repeating: 0xEB, count: 40_009)
                try companionPartialData.write(to: companionPartial)
                try Support.age(companionPartial, by: 11 * 60)
                CacheCoordinator.resetImportedRootsForTesting()
                _ = Self.coordinator(root: root, capBytes: 1 << 30, modelKey: modelKey)
                #expect(
                    (try? Data(contentsOf: companionPartial)) == companionPartialData,
                    "VICTIM: a companion partial was swept at open under a schema this build does not know"
                )
                for (url, data) in kept {
                    #expect((try? Data(contentsOf: url)) == data, "\(url.lastPathComponent)")
                }

                let (_, clearLog) = try Support.capturingStandardError { newer.clear() }
                for (url, data) in kept {
                    #expect(
                        (try? Data(contentsOf: url)) == data,
                        "VICTIM: \(url.lastPathComponent) was removed from a listing by clear() under a newer schema"
                    )
                }
                #expect(
                    clearLog.contains("[vmlx][cache/disk-index] clear skipped: "), "\(clearLog)")
                #expect(
                    Support.fileBytes(indexedURL) == 0, "the payload the index names still goes")
                #expect(try Support.indexedRows(root).isEmpty)

                // The control: under the current schema the same files go.
                try RawDB(root: root).require(
                    "PRAGMA user_version = \(DiskCacheIndexSchema.currentVersion)")
                CacheCoordinator.resetImportedRootsForTesting()
                _ = Self.coordinator(root: root, capBytes: 1 << 30, modelKey: modelKey)
                #expect(!FileManager.default.fileExists(atPath: partial.path))
                #expect(!FileManager.default.fileExists(atPath: shortURL.path))
                #expect(!FileManager.default.fileExists(atPath: companionPartial.path))
            }
        }
    }
}
