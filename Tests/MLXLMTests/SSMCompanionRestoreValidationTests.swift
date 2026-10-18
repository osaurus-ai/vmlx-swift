import Foundation
import MLX
@testable import MLXLMCommon
import XCTest

/// A warm hybrid turn publishes the same prompt boundary it restored. The
/// companion store's skip-validated path exists so that re-store touches the
/// pair instead of rewriting it, and the coordinator's durable/validated
/// predicates exist so the stable seed is not produced again. Both depend on
/// the recorded fingerprint matching the file after the touch.
final class SSMCompanionRestoreValidationTests: XCTestCase {
    private var roots: [URL] = []

    override func tearDown() {
        for root in roots { try? FileManager.default.removeItem(at: root) }
        roots.removeAll()
        super.tearDown()
    }

    private func coordinator() -> CacheCoordinator {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        roots.append(directory)
        let result = CacheCoordinator(config: .init(usePagedCache: false, enableDiskCache: true,
            diskCacheDir: directory, modelKey: "companion-restore-validation"))
        // ArraysCache-style topology: the recurrent state travels as a
        // separate companion payload beside the KV row.
        result.setHybrid(true, requiresRecurrentSSMCompanion: true, requiresSeparateRecurrentPayload: true)
        return result
    }

    private func snapshot(tokens: [Int]) -> [KVCache] {
        let recurrent = MambaCache()
        recurrent.offset = tokens.count
        recurrent.state = [MLXArray([Float(tokens.reduce(0, +)), Float(tokens.count)]).reshaped(1, 1, 2)]
        MLX.eval(recurrent.state)
        return [recurrent]
    }

    func testRepublishingTheSameBoundaryKeepsTheCompanionValidatedAndDurable() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let disk = coordinator()
        let tokens = Array(1...12)
        let salt = "salt"
        for round in 1...3 {
            let cache = snapshot(tokens: tokens)
            disk.storeAfterGeneration(promptTokens: tokens, perLayerData: [nil],
                ssmStates: extractSSMStates(from: cache), cache: cache, mediaSalt: salt, isStableRoot: true)
            XCTAssertTrue(disk.ssmStateCache.hasValidatedCompleteDiskEntry(
                tokens: tokens, boundary: tokens.count, mediaSalt: salt),
                "round \(round): the companion must stay validated across a re-store")
            XCTAssertTrue(disk.hasValidatedDiskEntry(tokens: tokens, mediaSalt: salt), "round \(round)")
            XCTAssertTrue(disk.hasDurableDiskEntry(tokens: tokens, mediaSalt: salt), "round \(round)")
            // Re-stores touch the pair instead of rewriting it.
            XCTAssertEqual(disk.ssmStateCache.diskStore?.snapshotStoreSkips(), round - 1, "round \(round)")
            XCTAssertEqual(disk.snapshotStats().diskStats?.storeSkips, round - 1, "round \(round)")
        }
    }

    func testTouchedCompanionRecencyIsReadBackFromDisk() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let disk = coordinator()
        let tokens = Array(1...12)
        let cache = snapshot(tokens: tokens)
        disk.storeAfterGeneration(promptTokens: tokens, perLayerData: [nil],
            ssmStates: extractSSMStates(from: cache), cache: cache, mediaSalt: nil, isStableRoot: true)
        let store = try XCTUnwrap(disk.ssmStateCache.diskStore)
        // A recency touch re-dates both files; what the store remembers must
        // be what a fresh stat of the pair reports afterwards.
        XCTAssertTrue(store.touchRecency(tokens: tokens, boundary: tokens.count, mediaSalt: nil,
                                         at: Date().addingTimeInterval(5)))
        XCTAssertTrue(disk.ssmStateCache.hasValidatedCompleteDiskEntry(
            tokens: tokens, boundary: tokens.count, mediaSalt: nil))
        XCTAssertTrue(disk.hasDurableDiskEntry(tokens: tokens, mediaSalt: nil))
    }
}
