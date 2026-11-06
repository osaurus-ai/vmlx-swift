import MLX
import MLXLMCommon
import XCTest

/// Replaying the last prompt token requires every cache layer at N-1.
/// A full rotating ring cannot be rolled back by the ordinary trim path.
final class RotatingPrefixReplayTests: XCTestCase {
  private var releaseMLX: (() -> Void)?
  override func setUp() {
    super.setUp()
    let lock = lockSerializedMLXTest()
    releaseMLX = { lock.unlock() }
  }
  override func tearDown() {
    releaseMLX?()
    releaseMLX = nil
    super.tearDown()
  }

  private func coordinator(rotating: Bool = true) -> CacheCoordinator {
    let c = CacheCoordinator(
      config: CacheCoordinatorConfig(
        usePagedCache: true, enableDiskCache: false,
        pagedBlockSize: 4, maxCacheBlocks: 128, modelKey: "rotating-replay-regression"))
    c.setPagedBoundaryCompanionRequired(rotating)
    return c
  }

  private func store(_ tokens: [Int], in c: CacheCoordinator, rotating: Bool = true) {
    let keys = MLXArray(tokens.map(Float.init)).reshaped(1, 1, tokens.count, 1)
    let values = keys + 100
    let full = KVCacheSimple()
    _ = full.update(keys: keys, values: values)
    var caches: [KVCache] = [full]
    if rotating {
      let ring = RotatingKVCache(maxSize: 5, keep: 0)
      for index in tokens.indices {
        _ = ring.update(
          keys: keys[.ellipsis, index..<(index + 1), 0...],
          values: values[.ellipsis, index..<(index + 1), 0...])
        eval(ring)
      }
      caches.append(ring)
    }
    eval(caches)
    c.storeAfterGeneration(
      promptTokens: tokens, perLayerData: extractLayerData(from: caches),
      ssmStates: nil, cache: caches)
  }

  private func expectHit(
    _ tokens: [Int], count: Int, in c: CacheCoordinator,
    rotating: Bool = true, file: StaticString = #filePath, line: UInt = #line
  ) {
    guard
      case .hit(let matched, let remaining, let detail, let blocks, _, _) =
        c.fetch(tokens: tokens, skipExactDiskBoundary: true)
    else {
      return XCTFail("expected a safe reusable prefix", file: file, line: line)
    }
    defer { c.release(blocks: blocks) }
    XCTAssertEqual(matched, count, file: file, line: line)
    XCTAssertEqual(remaining, Array(tokens.dropFirst(count)), file: file, line: line)
    XCTAssertEqual(detail, .paged, file: file, line: line)
    var restored: [KVCache] = [KVCacheSimple()]
    if rotating { restored.append(RotatingKVCache(maxSize: 5, keep: 0)) }
    let actual = restoreLayerData(from: blocks, into: restored)
    XCTAssertEqual(actual, count, file: file, line: line)
    XCTAssertTrue(restored.allSatisfy { $0.offset == count }, file: file, line: line)
    let prefix = Array(tokens.prefix(count)).map(Float.init)
    XCTAssertEqual(restored[0].state[0].asArray(Float.self), prefix, file: file, line: line)
    if let ring = restored.last as? RotatingKVCache {
      XCTAssertEqual(
        ring.temporallyOrderedKV()!.keys.asArray(Float.self), Array(prefix.suffix(5)), file: file,
        line: line)
      XCTAssertEqual(
        ring.temporallyOrderedKV()!.values.asArray(Float.self),
        Array(prefix.suffix(5)).map { $0 + 100 }, file: file, line: line)
    }
  }

  func testRepeatedPromptUsesCapturedSeedAcrossBlockAndWindowBoundaries() {
    for length in [6, 7, 8, 9, 10, 16, 17] {
      for seedFirst in [true, false] {
        let c = coordinator()
        let tokens = Array(0..<length)
        let seed = Array(tokens.dropLast())
        for entry in seedFirst ? [seed, tokens] : [tokens, seed] { store(entry, in: c) }
        expectHit(tokens, count: length - 1, in: c)
      }
    }
  }

  func testGrowingAndDivergingPromptsKeepTheirCompleteSuffix() {
    let c = coordinator()
    let a = Array(0..<10)
    store(Array(a.dropLast()), in: c)
    store(a, in: c)
    expectHit(a + [10, 11], count: a.count, in: c)
    expectHit(Array(a.dropLast()) + [99], count: a.count - 1, in: c)
    expectHit(a, count: a.count - 1, in: c)
  }

  func testPartialCompanionRemainsReachableBehindLongerFullBlocks() {
    for length in [14, 256] {
      let c = coordinator()
      let seed = length - (length == 14 ? 3 : 5)
      store(Array(0..<seed), in: c)
      store(Array(0..<length), in: c)
      // Longer matching full blocks bypass a reusable partial leaf. Select
      // that off-chain leaf and release every unused block after each lookup.
      let before = c.pagedCache!.snapshotStats()
      for _ in 0..<32 { expectHit(Array(0..<(length - 1)) + [999], count: seed, in: c) }
      let after = c.pagedCache!.snapshotStats()
      XCTAssertEqual(after.allocatedBlocks, before.allocatedBlocks)
      XCTAssertEqual(after.freeBlocks, before.freeBlocks)
    }
  }

  func testMissingSafeSeedMissesRatherThanRewindingOnlySomeLayers() {
    let c = coordinator()
    let tokens = Array(0..<10)
    store(tokens, in: c)
    let result = c.fetch(tokens: tokens, skipExactDiskBoundary: true)
    if case .hit(_, _, _, let blocks, _, _) = result {
      c.release(blocks: blocks)
      XCTFail("an exact rotating hit without a captured earlier boundary must miss")
    }
  }

  func testDenseCacheStillUsesExactPagedHit() {
    let c = coordinator(rotating: false)
    let tokens = Array(0..<10)
    store(tokens, in: c, rotating: false)
    expectHit(tokens, count: tokens.count, in: c, rotating: false)
  }
}
