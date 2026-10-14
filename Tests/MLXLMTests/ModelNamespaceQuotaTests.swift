import Cmlx
import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct ModelNamespaceQuotaTests {
    @Test func currentModelResumeSurvivesOlderLongerSameChatUnderQuota() throws {
        try MLXMetalTestLock.withLock {
            if let path = ProcessInfo.processInfo.environment["VMLX_DIAGNOSTIC_METALLIB"] {
                _ = path.withCString { mlx_metal_set_metallib_path($0) }
            }
            try Device.withDefaultDevice(.cpu) {
                let evidenceDir = ProcessInfo.processInfo.environment["VMLX_MODEL_CHAIN_REPRO_DIR"]
                let root = evidenceDir.map { URL(fileURLWithPath: $0) }
                    ?? FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
                defer {
                    if evidenceDir == nil { try? FileManager.default.removeItem(at: root) }
                }
                func coordinator(_ key: String) -> CacheCoordinator {
                    CacheCoordinator(config: CacheCoordinatorConfig(
                        usePagedCache: false, enableDiskCache: true,
                        diskCacheMaxGB: Float(20_000) / 1_073_741_824,
                        diskCacheDir: root, modelKey: key))
                }
                let old = coordinator("old-model-8M")
                let current = coordinator("current-model-4M")
                let oldTokens = Array(0 ..< 128_000)
                let currentTokens = Array(0 ..< 16_000)
                old.storePersistentBoundary(
                    tokens: oldTokens,
                    diskArrays: ["keys": MLXArray(Array(repeating: Int32(1), count: 4_096))],
                    ssmStates: nil, chainId: "same-chat", isResumeBoundary: true)
                let before = try DiskCacheAccountingTestSupport.RawDB(root: root)
                    .rows("SELECT model_key, chain_id, kind, token_count, file_size FROM cache_entries")
                current.storePersistentBoundary(
                    tokens: currentTokens,
                    diskArrays: ["keys": MLXArray(Array(repeating: Int32(2), count: 2_048))],
                    ssmStates: nil, chainId: "same-chat", isResumeBoundary: true)
                #expect(old.waitForPendingDiskWrites())
                #expect(current.waitForPendingDiskWrites())
                let rows = try DiskCacheAccountingTestSupport.RawDB(root: root)
                    .rows("SELECT model_key, chain_id, kind, token_count, file_size FROM cache_entries")
                let disk = try #require(current.diskCache)
                let oldDurable = old.hasDurableDiskEntry(tokens: oldTokens)
                let currentDurable = current.hasDurableDiskEntry(tokens: currentTokens)
                let receipt: [String: Any] = [
                    "before": before.map { $0.map { $0 ?? "NULL" } },
                    "after": rows.map { $0.map { $0 ?? "NULL" } },
                    "cap_bytes": disk.maxSizeBytes,
                    "used_bytes": disk.usageBytes(),
                    "old_durable": oldDurable,
                    "current_durable": currentDurable,
                    "evictions": disk.snapshotStats().evictions,
                    "device": "CPU",
                ]
                try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys])
                    .write(to: root.appendingPathComponent("diagnostic-receipt.json"))
                print("MIXED_MODEL_CHAIN rows=\(rows) oldDurable=\(oldDurable) currentDurable=\(currentDurable) bytes=\(disk.usageBytes()) cap=\(disk.maxSizeBytes)")
                #expect(currentDurable, "current model's fitting resume row should remain available")
                #expect(!oldDurable, "the older different-model row should yield to the current model")
            }
        }
    }

    private func row(
        _ id: String, model: String?, tokens: Int, bytes: Int64 = 100,
        recency: Double = 1, stable: Bool = false
    ) -> QuotaRow {
        QuotaRow(
            id: id, tokenCount: tokens, bytes: bytes, recency: recency,
            isStableRoot: stable, isResumeBoundary: !stable, chainId: "same-chat",
            modelKey: model, isLegacyCompanion: false)
    }

    @Test func nilAndEmptyModelKeysStillShareTheLegacyChain() {
        let rows = [
            row("long", model: nil, tokens: 200),
            row("short", model: "", tokens: 100, recency: 2),
        ]
        for key: String? in [nil, ""] {
            let plan = DiskQuotaPlanner.plan(
                rows: rows, capBytes: 100, activeChain: "same-chat", activeModelKey: key)
            #expect(plan.evict == ["short"])
            #expect(plan.totalAfter == 100)
            #expect(plan.event?.chainId == "same-chat")
        }
    }

    @Test func currentNamespacePressureDoesNotConfirmAnotherModelsLoss() {
        let rows = [
            row("old", model: "old", tokens: 128_000),
            row("oversized-current", model: "current", tokens: 16_000, bytes: 201),
        ]
        let plan = DiskQuotaPlanner.plan(
            rows: rows, capBytes: 100, activeChain: "same-chat", activeModelKey: "current")
        #expect(plan.event?.kind == .activeTipDropped)
        #expect(plan.event?.chainId == "same-chat")
        #expect(plan.event?.tipBytes == 201)
        #expect(plan.confirmedEvent(rows: rows, lostRows: ["old"]) == nil)
        #expect(plan.confirmedEvent(rows: rows, lostRows: ["oversized-current"]) == plan.event)
    }

    @Test func modelSwitchKeepsStableRootAndSharedQuotaPolicy() {
        let rows = [
            row("old", model: "old", tokens: 128_000, recency: 0),
            row("root", model: "old", tokens: 500, stable: true),
            row("current", model: "current", tokens: 16_000, recency: 2),
        ]
        let plan = DiskQuotaPlanner.plan(
            rows: rows, capBytes: 200, activeChain: "same-chat", activeModelKey: "current")
        #expect(plan.totalBefore == 300)
        #expect(plan.evict == ["old"])
        #expect(plan.totalAfter == 200)
        #expect(plan.event == nil)
    }

    @Test func otherModelBoundaryCannotHealCurrentModelsPressure() throws {
        try MLXMetalTestLock.withLock {
            if let path = ProcessInfo.processInfo.environment["VMLX_DIAGNOSTIC_METALLIB"] {
                _ = path.withCString { mlx_metal_set_metallib_path($0) }
            }
            try Device.withDefaultDevice(.cpu) {
                let root = FileManager.default.temporaryDirectory
                    .appendingPathComponent("model-pressure-\(UUID().uuidString)")
                defer { try? FileManager.default.removeItem(at: root) }
                func coordinator(_ key: String) -> CacheCoordinator {
                    CacheCoordinator(config: CacheCoordinatorConfig(
                        usePagedCache: false, enableDiskCache: true,
                        diskCacheMaxGB: Float(20_000) / 1_073_741_824,
                        diskCacheDir: root, modelKey: key))
                }
                let old = coordinator("old")
                let current = coordinator("current")
                old.storePersistentBoundary(
                    tokens: Array(0 ..< 128_000),
                    diskArrays: ["keys": MLXArray(Array(repeating: Int32(1), count: 4_096))],
                    ssmStates: nil, chainId: "same-chat", isResumeBoundary: true)
                current.storePersistentBoundary(
                    tokens: Array(0 ..< 16_000),
                    diskArrays: ["keys": MLXArray(Array(repeating: Int32(2), count: 8_192))],
                    ssmStates: nil, chainId: "same-chat", isResumeBoundary: true)
                let disk = try #require(current.diskCache)
                let pending = try #require(disk.snapshotStats().capacityPressureByChain["same-chat"])
                #expect(pending.event.kind == .activeTipDropped)
                #expect(pending.tipTokenCount == 16_000)
                disk.reconcileCapacityPressure(chainId: "same-chat", requiresCompanion: false)
                #expect(disk.snapshotStats().capacityPressureByChain["same-chat"] == pending)
                current.storePersistentBoundary(
                    tokens: Array(0 ..< 16_000),
                    diskArrays: ["keys": MLXArray(Array(repeating: Int32(2), count: 2_048))],
                    ssmStates: nil, chainId: "same-chat", isResumeBoundary: true)
                #expect(current.waitForPendingDiskWrites())
                #expect(disk.snapshotStats().capacityPressureByChain["same-chat"] == nil)
                #expect(current.hasDurableDiskEntry(tokens: Array(0 ..< 16_000)))
            }
        }
    }

}
