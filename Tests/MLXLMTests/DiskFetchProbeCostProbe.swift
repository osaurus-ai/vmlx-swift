import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Measurement probe, not a regression test: what one `CacheCoordinator.fetch`
/// costs when the index is full of rows that cannot match it.
///
/// Gated on `VMLX_QUOTA_PROBE=1`, like ``DiskQuotaScanCostProbe``. Every
/// candidate length is hashed and probed, so the cost is the number of probes
/// times (prefix copy + SHA-256 of the prefix + the index lookups).
///
/// Three arms, the same 20 011-token prompt and the same single matching row
/// (5 003 tokens) in each; only who owns the 5 003 decoy rows varies:
///
/// - `foreign`: another model wrote them, and the index says so.
/// - `foreign-unkeyed`: the SAME rows with `model_key` set to NULL, which is
///   what an older build writes. Nothing can tell these from this model's own
///   legacy rows, so every one of them stays a candidate.
/// - `same-model`: this model wrote them, in another conversation. No filter
///   on the model can remove these; this arm is the cost of the query itself.
///
/// Each line reports `probes`, so a fast fetch that skipped the work and a
/// slow one that did it cannot be confused.
@Suite(.serialized)
struct DiskFetchProbeCostProbe {

    private static let decoyCount = 5_003
    private static let storedLength = 5_003
    private static let promptLength = 20_011

    /// Odd lengths above the stored one: distinct, never a multiple of 64.
    private static var decoyLengths: [Int] {
        (0 ..< decoyCount).map { 5_005 + 2 * $0 }
    }

    private static var buildConfiguration: String {
        #if DEBUG
            return "debug"
        #else
            return "release"
        #endif
    }

    private static func coordinator(root: URL, modelKey: String) -> CacheCoordinator {
        CacheCoordinator(
            config: CacheCoordinatorConfig(
                usePagedCache: false,
                enableDiskCache: true,
                diskCacheMaxGB: 64,
                diskCacheDir: root,
                modelKey: modelKey))
    }

    /// Seven fetches, the first discarded; the median of the rest, and the
    /// probes (misses) of the last one.
    private static func measure(
        _ coordinator: CacheCoordinator, prompt: [Int]
    ) throws -> (medianMs: Double, minMs: Double, maxMs: Double, probes: Int) {
        var millis: [Double] = []
        var probes = -1
        for _ in 0 ..< 7 {
            let before = try #require(coordinator.snapshotStats().diskStats)
            let start = DispatchTime.now().uptimeNanoseconds
            let result = coordinator.fetch(tokens: prompt)
            millis.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000)
            let after = try #require(coordinator.snapshotStats().diskStats)
            // Fail closed: a fetch that did not find the row timed a miss.
            guard case .hit(let matched, _, let detail, _, _, _) = result else {
                throw ProbeError.noHit
            }
            try #require(matched == storedLength && detail == .disk)
            try #require(after.hits - before.hits == 1)
            probes = after.misses - before.misses
        }
        let steady = millis.dropFirst().sorted()
        return (steady[steady.count / 2], steady[0], steady[steady.count - 1], probes)
    }

    private enum ProbeError: Error { case noHit }

    private static func report(
        arm: String, rows: Int, foreign: Int,
        _ sample: (medianMs: Double, minMs: Double, maxMs: Double, probes: Int)
    ) {
        print(
            "FETCH_PROBE arm=\(arm) rows=\(rows) foreign=\(foreign) "
                + "ms=\(String(format: "%.3f", sample.medianMs)) "
                + "min_ms=\(String(format: "%.3f", sample.minMs)) "
                + "max_ms=\(String(format: "%.3f", sample.maxMs)) "
                + "probes=\(sample.probes) prompt_tokens=\(promptLength) "
                + "build=\(buildConfiguration) samples=6")
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["VMLX_QUOTA_PROBE"] == "1"))
    func fetchCostAmongRowsThatCannotMatch() throws {
        try MLXMetalTestLock.withLock {
            let prompt = (0 ..< Self.promptLength).map { 700_000 + $0 }
            let otherConversation = (0 ..< Self.promptLength).map { 900_000 + $0 }
            let payload = ["data": MLXArray.ones([3], dtype: .float32)]
            let lengths = Self.decoyLengths
            try #require(Set(lengths).count == Self.decoyCount)
            try #require(lengths.allSatisfy { $0 % 64 != 0 && $0 < Self.promptLength - 1 })

            // Arms 1 and 2: the decoys belong to another model.
            do {
                let root = FileManager.default.temporaryDirectory
                    .appendingPathComponent("vmlx-fetch-probe-foreign-\(UUID().uuidString)")
                defer { try? FileManager.default.removeItem(at: root) }
                let ours = Self.coordinator(root: root, modelKey: "fetch-probe-A")
                let theirs = Self.coordinator(root: root, modelKey: "fetch-probe-B")
                let theirDisk = try #require(theirs.diskCache)
                try #require(ours.diskCache).store(
                    tokens: Array(prompt.prefix(Self.storedLength)), arrays: payload,
                    enforceQuota: false)
                for length in lengths {
                    theirDisk.store(
                        tokens: Array(prompt.prefix(length)), arrays: payload,
                        enforceQuota: false)
                }
                let raw = try DiskCacheAccountingTestSupport.RawDB(root: root)
                let keyed = try raw.rows(
                    "SELECT COUNT(*) FROM cache_entries WHERE model_key = 'fetch-probe-B'")
                try #require(keyed.first?.first == "\(Self.decoyCount)")
                let rows = Self.decoyCount + 1

                Self.report(
                    arm: "foreign", rows: rows, foreign: Self.decoyCount,
                    try Self.measure(ours, prompt: prompt))

                // The same rows, as an older build would have written them.
                try raw.require(
                    "UPDATE cache_entries SET model_key = NULL WHERE model_key = 'fetch-probe-B'")
                let unkeyed = try raw.rows(
                    "SELECT COUNT(*) FROM cache_entries WHERE model_key IS NULL")
                try #require(unkeyed.first?.first == "\(Self.decoyCount)")
                Self.report(
                    arm: "foreign-unkeyed", rows: rows, foreign: Self.decoyCount,
                    try Self.measure(ours, prompt: prompt))
            }

            // Arm 3: the decoys are this model's own, from another conversation.
            do {
                let root = FileManager.default.temporaryDirectory
                    .appendingPathComponent("vmlx-fetch-probe-same-\(UUID().uuidString)")
                defer { try? FileManager.default.removeItem(at: root) }
                let ours = Self.coordinator(root: root, modelKey: "fetch-probe-A")
                let disk = try #require(ours.diskCache)
                disk.store(
                    tokens: Array(prompt.prefix(Self.storedLength)), arrays: payload,
                    enforceQuota: false)
                for length in lengths {
                    disk.store(
                        tokens: Array(otherConversation.prefix(length)), arrays: payload,
                        enforceQuota: false)
                }
                try #require(disk.quotaEntries().count == Self.decoyCount + 1)
                Self.report(
                    arm: "same-model", rows: Self.decoyCount + 1, foreign: 0,
                    try Self.measure(ours, prompt: prompt))
            }
        }
    }
}
