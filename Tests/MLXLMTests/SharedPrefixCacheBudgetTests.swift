import Foundation
import Metal
import MLX
@testable import MLXLMCommon
import XCTest

/// Real-Metal, model-free checks of shared paged/SSM retention.
final class SharedPrefixCacheBudgetTests: XCTestCase {
    private var mlxLock: MLXTestSerializationToken?

    override func tearDown() {
        mlxLock?.unlock()
        mlxLock = nil
        super.tearDown()
    }

    private func coordinator(_ cap: Int?, disk: URL? = nil, paged: Bool = true) -> CacheCoordinator {
        CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: paged, enableDiskCache: disk != nil,
            pagedBlockSize: 4, maxCacheBlocks: 32, maxCacheBytes: cap,
            diskCacheMaxGB: 0.1, diskCacheDir: disk,
            ssmMaxEntries: 50, modelKey: "shared-retained-budget"))
    }

    private func state(_ value: Float) -> MLXArray {
        MLXArray(Array(repeating: value, count: 16_384)) // 64 KiB
    }

    private func storePaged(_ coordinator: CacheCoordinator, token: Int) {
        let keys = MLXArray(Array(repeating: Float(token), count: 256)).reshaped(1, 1, 4, 64)
        coordinator.pagedCache!.storeTokenSequence(
            tokens: [token, 1, 2, 3], layerData: [[(keys: keys, values: keys + 1)]],
            boundaryCompanionData: ["ring": MLXArray.zeros([2048])])
    }

    private func assertCombinedCharge(_ coordinator: CacheCoordinator, cap: Int,
                                      file: StaticString = #filePath, line: UInt = #line) {
        let paged = coordinator.pagedCache?.snapshotStats().residentBytes ?? 0
        let ssm = coordinator.ssmStateCache.snapshotStats().residentBytes
        XCTAssertEqual(coordinator.retainedCacheBytes, paged + ssm, file: file, line: line)
        XCTAssertLessThanOrEqual(coordinator.retainedCacheBytes, cap, file: file, line: line)
    }

    override func setUpWithError() throws {
        try super.setUpWithError()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("A real Metal allocator is required")
        }
        mlxLock = lockSerializedMLXTest()
    }

    func testDistinctSSMPrefixesRespectCoordinatorRetainedBudget() {
        let cap = 128 * 1024
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: true, enableDiskCache: false,
            pagedBlockSize: 4, maxCacheBlocks: 32, maxCacheBytes: cap,
            ssmMaxEntries: 50, modelKey: "shared-retained-budget"))
        for token in 0..<20 {
            let state = MLXArray(Array(repeating: Float(token), count: 16_384))
            coordinator.ssmStateCache.store(ssmStates: [state], tokens: [token], boundary: 1)
        }
        var retainedLogicalBytes = 0
        for token in 0..<20 {
            if let states = coordinator.ssmStateCache.fetch(tokens: [token], boundary: 1) {
                retainedLogicalBytes += states.reduce(0) { $0 + $1.nbytes }
            }
        }
        XCTAssertLessThanOrEqual(retainedLogicalBytes, cap,
            "The coordinator cap must include separate SSM snapshots, not just paged KV")
        coordinator.releaseVolatile()
    }

    func testMixedKVRotatingAndSSMShareOneAllowance() {
        let cap = 256 * 1024
        let c = coordinator(cap)
        for token in 0..<30 {
            storePaged(c, token: token)
            c.ssmStateCache.store(ssmStates: [state(Float(token))], tokens: [token], boundary: 1)
            assertCombinedCharge(c, cap: cap)
            XCTAssertGreaterThan(c.pagedCache!.snapshotStats().residentBytes, 0)
            XCTAssertGreaterThan(c.pagedCache!.snapshotStats().companionBytes, 0)
            XCTAssertGreaterThan(c.ssmStateCache.snapshotStats().residentBytes, 0)
        }
        c.releaseVolatile()
        assertCombinedCharge(c, cap: cap)
        XCTAssertEqual(c.retainedCacheBytes, 0)
    }

    func testSSMLRUReplacementCompletenessAndClearReleaseCharge() throws {
        let charge = 64 * 1024 + 16_384
        let c = coordinator(2 * charge, paged: false)
        for token in [1, 2] {
            c.ssmStateCache.store(ssmStates: [state(Float(token))], tokens: [token], boundary: 1)
        }
        XCTAssertNotNil(c.ssmStateCache.fetch(tokens: [1], boundary: 1)) // Touch A, evict B next.
        c.ssmStateCache.store(ssmStates: [state(3)], tokens: [3], boundary: 1)
        XCTAssertFalse(c.ssmStateCache.contains(tokens: [2], boundary: 1))
        XCTAssertTrue(c.ssmStateCache.contains(tokens: [1], boundary: 1))
        for value in 4..<24 {
            c.ssmStateCache.store(ssmStates: [state(Float(value))], tokens: [1], boundary: 1,
                                  isComplete: false)
            XCTAssertEqual(c.retainedCacheBytes, 2 * charge)
        }
        XCTAssertNil(c.ssmStateCache.fetchEntry(tokens: [1], boundary: 1, requireComplete: true))
        let partial = try XCTUnwrap(c.ssmStateCache.fetchEntry(tokens: [1], boundary: 1))
        XCTAssertFalse(partial.isComplete)
        XCTAssertEqual(partial.states[0].asArray(Float.self).first, 23)
        let oversized = MLXArray.zeros([131_072])
        c.ssmStateCache.store(ssmStates: [oversized], tokens: [99], boundary: 1)
        XCTAssertEqual(c.retainedCacheBytes, 2 * charge, "Oversized arrivals must preserve useful entries")
        c.ssmStateCache.store(ssmStates: [oversized], tokens: [1], boundary: 1)
        XCTAssertNil(c.ssmStateCache.fetch(tokens: [1], boundary: 1), "Rejected replacement must not expose stale state")
        XCTAssertEqual(c.retainedCacheBytes, charge)
        c.releaseVolatile()
        XCTAssertEqual(c.retainedCacheBytes, 0)
        XCTAssertEqual(c.ssmStateCache.snapshotStats().residentBytes, 0)
        c.ssmStateCache.store(ssmStates: [state(5)], tokens: [5], boundary: 1)
        XCTAssertEqual(c.retainedCacheBytes, charge)
    }

    func testNilBudgetRetainsLegacySSMEntryLimitAndZeroRetainsNothing() {
        let unlimited = coordinator(nil, paged: false)
        for token in 0..<4 {
            unlimited.ssmStateCache.store(ssmStates: [state(Float(token))], tokens: [token], boundary: 1)
        }
        XCTAssertGreaterThan(unlimited.retainedCacheBytes, 128 * 1024)
        for token in 0..<4 { XCTAssertTrue(unlimited.ssmStateCache.contains(tokens: [token], boundary: 1)) }
        let legacy = SSMStateCache(maxEntries: 2)
        for token in 0..<3 { legacy.store(ssmStates: [state(Float(token))], tokens: [token], boundary: 1) }
        XCTAssertFalse(legacy.contains(tokens: [0], boundary: 1))
        XCTAssertTrue(legacy.contains(tokens: [1], boundary: 1))
        XCTAssertTrue(legacy.contains(tokens: [2], boundary: 1))
        let zero = coordinator(0)
        zero.ssmStateCache.store(ssmStates: [state(1)], tokens: [1], boundary: 1)
        storePaged(zero, token: 1)
        XCTAssertNil(zero.ssmStateCache.fetch(tokens: [1], boundary: 1))
        XCTAssertNil(zero.pagedCache!.fetchPrefix(tokens: [1, 1, 2, 3]))
        XCTAssertEqual(zero.retainedCacheBytes, 0)
    }

    func testOversizedAndZeroRAMStoresPersistAndFetchWithoutRetention() throws {
        for cap in [0, 4096] {
            let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: dir) }
            let c = coordinator(cap, disk: dir)
            c.ssmStateCache.store(ssmStates: [state(7)], tokens: [7], boundary: 1, isComplete: false)
            XCTAssertFalse(c.ssmStateCache.contains(tokens: [7], boundary: 1))
            XCTAssertEqual(c.retainedCacheBytes, 0)
            XCTAssertNotNil(c.ssmStateCache.diskStore!.fetch(tokens: [7], boundary: 1))
            XCTAssertNil(c.ssmStateCache.fetchEntry(tokens: [7], boundary: 1, requireComplete: true))
            let fetched = try XCTUnwrap(c.ssmStateCache.fetchEntry(tokens: [7], boundary: 1))
            XCTAssertFalse(fetched.isComplete)
            XCTAssertEqual(fetched.states[0].asArray(Float.self).first, 7)
            XCTAssertFalse(c.ssmStateCache.contains(tokens: [7], boundary: 1))
            XCTAssertEqual(c.retainedCacheBytes, 0)
            c.releaseVolatile()
            XCTAssertNotNil(c.ssmStateCache.fetch(tokens: [7], boundary: 1), "Volatile release must preserve disk")
            c.clear()
            XCTAssertNil(c.ssmStateCache.fetch(tokens: [7], boundary: 1))
        }
    }

    func testDiskHydrationChargesOnlyWhenSharedAllowanceFits() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: dir) }
        let cap = 128 * 1024
        let c = coordinator(cap, disk: dir)
        c.ssmStateCache.store(ssmStates: [state(7)], tokens: [7], boundary: 1)
        c.ssmStateCache.clear()
        storePaged(c, token: 1)
        storePaged(c, token: 2)
        let pagedBytes = c.pagedCache!.snapshotStats().residentBytes
        XCTAssertGreaterThan(pagedBytes, cap - (64 * 1024 + 16_384))
        XCTAssertNotNil(c.ssmStateCache.fetch(tokens: [7], boundary: 1))
        XCTAssertFalse(c.ssmStateCache.contains(tokens: [7], boundary: 1), "Other-tier entries are not evicted by callback")
        XCTAssertEqual(c.retainedCacheBytes, pagedBytes)
        c.pagedCache!.clear()
        XCTAssertNotNil(c.ssmStateCache.fetch(tokens: [7], boundary: 1))
        XCTAssertTrue(c.ssmStateCache.contains(tokens: [7], boundary: 1))
        XCTAssertEqual(c.retainedCacheBytes, 64 * 1024 + 16_384)
        assertCombinedCharge(c, cap: cap)
    }

    func testZeroRAMCoordinatorStillRestoresLinkedKVDiskAndSSMPair() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: dir) }
        let c = coordinator(0, disk: dir)
        c.setHybrid(true)
        let data = MLXArray.zeros([1, 1, 4, 64])
        let tokens = [1, 2, 3, 4]
        c.storeAfterGeneration(promptTokens: tokens, perLayerData: [(keys: data, values: data)],
                               ssmStates: [state(7)])
        c.releaseVolatile()
        guard case .hit(let matched, _, let detail, let blocks, let states, _) = c.fetch(tokens: tokens) else {
            return XCTFail("Zero retained-RAM allowance must not disable linked disk restore")
        }
        XCTAssertEqual(matched, tokens.count)
        XCTAssertEqual(detail, .disk)
        XCTAssertTrue(blocks.isEmpty)
        XCTAssertEqual(try XCTUnwrap(states).first?.asArray(Float.self).first, 7)
        XCTAssertFalse(c.ssmStateCache.contains(tokens: tokens, boundary: tokens.count))
        XCTAssertEqual(c.retainedCacheBytes, 0)
    }

    func testStoredAndFetchedSnapshotsHaveIndependentBackingBytes() throws {
        let c = coordinator(128 * 1024)
        let original = state(1)
        c.ssmStateCache.store(ssmStates: [original], tokens: [1], boundary: 1)
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let sourceBuffer = try XCTUnwrap(original.asMTLBuffer(device: device, noCopy: true))
        Stream.gpu.synchronize()
        sourceBuffer.contents().assumingMemoryBound(to: Float.self).pointee = 99
        let fetched = try XCTUnwrap(c.ssmStateCache.fetch(tokens: [1], boundary: 1))
        XCTAssertEqual(fetched[0].asArray(Float.self).first, 1)
        let fetchedBuffer = try XCTUnwrap(fetched[0].asMTLBuffer(device: device, noCopy: true))
        Stream.gpu.synchronize()
        fetchedBuffer.contents().assumingMemoryBound(to: Float.self).pointee = 88
        let again = try XCTUnwrap(c.ssmStateCache.fetch(tokens: [1], boundary: 1))
        XCTAssertEqual(again[0].asArray(Float.self).first, 1)
    }

    func testTierDeinitializationAndFailedPagedAdmissionReleaseReservations() {
        let budget = RetainedCacheBudget(limit: 128 * 1024)
        var ssm: SSMStateCache? = SSMStateCache(maxEntries: 50, modelKey: nil, retainedBudget: budget)
        ssm!.store(ssmStates: [state(1)], tokens: [1], boundary: 1)
        XCTAssertGreaterThan(budget.residentBytes, 0)
        ssm = nil
        XCTAssertEqual(budget.residentBytes, 0)
        var paged: PagedCacheManager? = PagedCacheManager(
            blockSize: 4, maxBlocks: 2, modelKey: nil, retainedBudget: budget)
        let held = paged!.allocateBlock()! // No free block, although the byte allowance fits.
        let data = MLXArray.zeros([1, 1, 4, 64])
        paged!.storeTokenSequence(tokens: [1, 2, 3, 4], layerData: [[(keys: data, values: data)]])
        XCTAssertEqual(budget.residentBytes, 0)
        paged!.freeBlock(held)
        paged!.storeTokenSequence(tokens: [1, 2, 3, 4], layerData: [[(keys: data, values: data)]])
        XCTAssertGreaterThan(budget.residentBytes, 0)
        paged = nil
        XCTAssertEqual(budget.residentBytes, 0)
    }

    func testSharedAccountArithmeticRefusesOverflowAndReturnsDeinitReservations() {
        let budget = RetainedCacheBudget(limit: nil)
        var first: RetainedCacheBudget.Account? = budget.makeAccount()
        let second = budget.makeAccount()
        XCTAssertTrue(first!.reserve(Int.max))
        XCTAssertFalse(second.reserve(1))
        XCTAssertEqual(budget.residentBytes, Int.max)
        first = nil
        XCTAssertEqual(budget.residentBytes, 0)
        XCTAssertTrue(second.reserve(1))
        second.release(1)
        XCTAssertEqual(budget.residentBytes, 0)
    }

    func testInFlightSSMReservationSurvivesClearAndBlocksPagedOverAdmission() {
        let cap = 64 * 1024 + 16_384
        let c = coordinator(cap)
        let input = SSMStateCache.FetchResult(states: [state(7)], isComplete: true)
        let keys = MLXArray.zeros([1, 1, 4, 64])
        eval(input.states + [keys])
        Stream.gpu.synchronize()

        let worker = DispatchGroup()
        var joined = false
        // The public bridge releases its lock before this cleanup runs, even
        // when a state assertion fails. Never leave a gated worker behind.
        defer {
            if !joined {
                joined = worker.wait(timeout: .now() + 10) == .success
                XCTAssertTrue(joined, "SSM worker did not finish after releasing the I/O gate")
            }
            if joined { c.clear() }
        }

        MLXCacheIOLock.withSerializedMLXCacheIO {
            worker.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                defer { worker.leave() }
                c.ssmStateCache.store(ssmStates: input.states, tokens: [7], boundary: 1)
            }
            // Observe the actual reservation rather than assuming the worker
            // reached it after a sleep. Materialization cannot pass our gate.
            let deadline = ProcessInfo.processInfo.systemUptime + 10
            while c.retainedCacheBytes != cap,
                  ProcessInfo.processInfo.systemUptime < deadline {
                Thread.sleep(forTimeInterval: 0.001)
            }
            guard c.retainedCacheBytes == cap else {
                return XCTFail("Worker never reserved its SSM copy before the deadline")
            }
            XCTAssertEqual(c.ssmStateCache.snapshotStats().residentBytes, 0)
            c.releaseVolatile()
            XCTAssertEqual(c.retainedCacheBytes, cap, "Volatile release must preserve the pending copy's reservation")
            c.clear()
            XCTAssertEqual(c.retainedCacheBytes, cap, "Clear must release committed entries only")
            XCTAssertEqual(c.ssmStateCache.snapshotStats().residentBytes, 0)

            // This payload fits an empty allowance. The pending SSM copy owns
            // all of it, so paged admission must fail before evaluating KV.
            c.pagedCache!.storeTokenSequence(
                tokens: [1, 2, 3, 4], layerData: [[(keys: keys, values: keys)]])
            XCTAssertEqual(c.pagedCache!.snapshotStats().residentBytes, 0)
            XCTAssertNil(c.pagedCache!.fetchPrefix(tokens: [1, 2, 3, 4]))
            XCTAssertEqual(c.retainedCacheBytes, cap)
        }

        joined = worker.wait(timeout: .now() + 10) == .success
        XCTAssertTrue(joined, "SSM store did not finish after the gate opened")
        guard joined else { return }
        XCTAssertTrue(c.ssmStateCache.contains(tokens: [7], boundary: 1))
        XCTAssertEqual(c.ssmStateCache.snapshotStats().residentBytes, cap)
        assertCombinedCharge(c, cap: cap)
        c.clear()
        XCTAssertEqual(c.retainedCacheBytes, 0)
    }

    func testReplacingShrinkingAndRemovingRotatingCompanionReturnsSharedBytes() throws {
        let cap = 128 * 1024
        let c = coordinator(cap)
        defer { c.clear() }
        let tokens = [1, 2, 3, 4]
        let keys = MLXArray.zeros([1, 1, 4, 64])
        func storeCompanion(_ ring: MLXArray?) {
            c.pagedCache!.storeTokenSequence(
                tokens: tokens, layerData: [[(keys: keys, values: keys)]],
                boundaryCompanionData: ring.map { ["ring": $0] })
        }

        storeCompanion(MLXArray.full([1, 2, 128, 64], values: MLXArray(Float(1))))
        let large = c.pagedCache!.snapshotStats()
        XCTAssertGreaterThan(large.companionBytes, 0)
        storeCompanion(MLXArray.full([1, 2, 128, 64], values: MLXArray(Float(2))))
        XCTAssertEqual(c.retainedCacheBytes, large.residentBytes)
        let replacement = try XCTUnwrap(c.pagedCache!.fetchPrefix(tokens: tokens))
        XCTAssertEqual(replacement.blocks.last!.boundaryCompanionData!["ring"]!.asArray(Float.self).first, 2)
        c.release(blocks: replacement.blocks)

        storeCompanion(MLXArray.zeros([1, 2, 16, 64]))
        let small = c.pagedCache!.snapshotStats()
        XCTAssertLessThan(small.companionBytes, large.companionBytes)
        XCTAssertEqual(large.residentBytes - small.residentBytes,
                       large.companionBytes - small.companionBytes)
        XCTAssertEqual(c.retainedCacheBytes, small.residentBytes)
        let ssm = state(7)
        c.ssmStateCache.store(ssmStates: [ssm], tokens: [7], boundary: 1)
        XCTAssertFalse(c.ssmStateCache.contains(tokens: [7], boundary: 1),
                       "The small companion still leaves too little allowance for this SSM copy")

        storeCompanion(nil)
        let removed = c.pagedCache!.snapshotStats()
        XCTAssertEqual(removed.companionBytes, 0)
        XCTAssertEqual(small.residentBytes - removed.residentBytes, small.companionBytes)
        XCTAssertEqual(c.retainedCacheBytes, removed.residentBytes)
        let withoutCompanion = try XCTUnwrap(c.pagedCache!.fetchPrefix(tokens: tokens))
        XCTAssertNil(withoutCompanion.blocks.last!.boundaryCompanionData)
        c.release(blocks: withoutCompanion.blocks)
        c.ssmStateCache.store(ssmStates: [ssm], tokens: [7], boundary: 1)
        XCTAssertTrue(c.ssmStateCache.contains(tokens: [7], boundary: 1),
                      "Removing the ring must make its shared bytes reusable by SSM")
        XCTAssertEqual(c.pagedCache!.snapshotStats().evictions, 0)
        assertCombinedCharge(c, cap: cap)
        c.clear()
        XCTAssertEqual(c.retainedCacheBytes, 0)
    }
}
