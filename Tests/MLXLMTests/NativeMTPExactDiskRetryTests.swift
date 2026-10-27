import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

/// Integration coverage of NativeMTP's real fetch/restore/prefill path.
/// The seeded, untrained tiny model is a state/token-fidelity oracle, not a
/// language-quality, performance or physical Metal-fault qualification.
final class NativeMTPExactDiskRetryTests: XCTestCase {
  func testExactHybridDiskRetryUsesNMinusOneWithoutReadingAndDiscardingN() throws {
    try MLXMetalTestLock.withLock {
      MLXRandom.seed(7104)
      let config = try JSONDecoder().decode(
        Qwen4ExpConfiguration.self, from: Data(Self.config.utf8))
      let real = Qwen4Exp(config)
      let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("native-mtp-exact-retry-\(UUID().uuidString)")
      defer { try? FileManager.default.removeItem(at: root) }
      let ids = [3, 7, 11, 5, 19, 2, 23, 13, 29]
      let n = ids.count
      var parameters = GenerateParameters(
        maxTokens: 4, temperature: 0, prefillStepSize: 1)
      parameters.draftStrategy = .nativeMTP(depth: 1)
      parameters.nativeMTPDepthPolicy = .fixed
      let input = LMInput(
        tokens: MLXArray(ids.map(Int32.init)).reshaped(1, n), tokenIds: ids)
      let salt = computeCacheSalt(for: input, parameters: parameters)

      func coordinator(_ directory: URL) -> CacheCoordinator {
        let value = CacheCoordinator(
          config: .init(
            usePagedCache: false, enableDiskCache: true, diskCacheDir: directory,
            modelKey: "tiny-qwen-native-mtp-exact-retry"))
        value.setHybrid(
          true, requiresRecurrentSSMCompanion: true,
          requiresSeparateRecurrentPayload: false)
        return value
      }

      // Build both persisted rows through the actual Qwen backbone and
      // native hybrid serializers. One-token calls match the cold
      // oracle's prepare schedule, avoiding a different numerical path.
      do {
        let writer = coordinator(root)
        for boundary in [n - 1, n] {
          let prefix = Array(ids.prefix(boundary))
          let cache = real.newCache(parameters: parameters)
          XCTAssertEqual(cache.count, 2)
          XCTAssertTrue(cacheRequiresDiskBackedCoordinatorRestore(cache))
          for token in prefix {
            _ = real.nativeBackboneForward(
              MLXArray([Int32(token)]).reshaped(1, 1), cache: cache)
          }
          MLX.eval(cache)
          XCTAssertEqual(Set(cacheBoundaryLeafOffsets(cache)), Set([boundary]))
          writer.storeAfterGeneration(
            promptTokens: prefix, perLayerData: extractLayerData(from: cache),
            ssmStates: extractSSMStates(from: cache), cache: cache,
            mediaSalt: salt, isStableRoot: true)
        }
        let disk = try XCTUnwrap(writer.diskCache)
        XCTAssertTrue(disk.waitForPendingPayloadRelease(), "Await actual writer ownership/drain")
        XCTAssertEqual(disk.pendingWriteCount, 0)
        XCTAssertTrue(writer.hasDurableDiskEntry(tokens: ids, mediaSalt: salt))
        XCTAssertTrue(writer.hasDurableDiskEntry(tokens: Array(ids.dropLast()), mediaSalt: salt))
      }

      // A new coordinator owns no paged or recurrent RAM entries. Do not
      // probe fetch before constructing the iterator: that would hide
      // which boundary the production NativeMTP caller actually asked for.
      let reader = coordinator(root)
      let before = reader.snapshotStats()
      XCTAssertFalse(before.pagedEnabled)
      XCTAssertNil(before.pagedStats)
      XCTAssertEqual(before.ssmStats.hits, 0)
      XCTAssertEqual(before.diskStats?.hits, 0)
      XCTAssertEqual(before.diskStats?.currentEntryCount, 2)
      let progress = ProgressRows()
      var warm = try NativeMTPTokenIterator(
        input: input, model: real, parameters: parameters, depth: 1,
        cacheCoordinator: reader, prefillProgressHandler: { progress.append($0) })

      let rows = progress.snapshot()
      let restored = rows.filter { $0.stage == .cacheRestore }
      XCTAssertEqual(restored.count, 1)
      XCTAssertEqual(restored.first?.completedUnitCount, n - 1)
      XCTAssertEqual(restored.first?.totalUnitCount, n)
      let prefill = rows.filter { $0.stage == .prefill }
      XCTAssertEqual(prefill.first?.completedUnitCount, n - 1)
      XCTAssertEqual(prefill.last?.completedUnitCount, n)
      XCTAssertTrue(prefill.allSatisfy { $0.totalUnitCount == n })
      XCTAssertEqual(
        prefill.map { $0.completedUnitCount },
        prefill.map { $0.completedUnitCount }.sorted())
      XCTAssertEqual(
        warm.promptPrimedRows, 1,
        "Only the actual one-token suffix is prepared/primed")
      let after = reader.snapshotStats()
      XCTAssertEqual(
        after.diskStats?.hits, 1,
        "Reading full N and then retrying N-1 would be two payload hits")
      XCTAssertEqual(after.diskStats?.rejectedDiskRestores, 0)
      XCTAssertEqual(after.ssmStats.hits, 0, "This must remain a reopened-disk row")
      XCTAssertFalse(after.pagedEnabled)

      // Real Qwen4Exp.prepare forwards its final tail and returns .logits.
      // The iterator snapshots AFTER prepare, so this prompt snapshot is N,
      // while cacheRestore above must still report the actual N-1 disk seed.
      // A missing coordinator intentionally retains no prompt snapshot.
      // Use an empty independent disk root so the cold reference exercises
      // the same persistence contract without finding either saved row.
      let coldReader = coordinator(root.appendingPathComponent("cold-reference"))
      var cold = try NativeMTPTokenIterator(
        input: input, model: real, parameters: parameters, depth: 1,
        cacheCoordinator: coldReader)
      let snapshot = try XCTUnwrap(warm.promptCacheSnapshot)
      let coldSnapshot = try XCTUnwrap(cold.promptCacheSnapshot)
      XCTAssertEqual(Set(cacheBoundaryLeafOffsets(snapshot)), Set([n]))
      XCTAssertEqual(Set(cacheBoundaryLeafOffsets(coldSnapshot)), Set([n]))
      assertState(snapshot, equals: coldSnapshot)

      // Both constructors then perform one initial target bridge row.
      // Compare their real N+1 state and greedy outputs; no fixture-specific
      // token is injected or forced.
      XCTAssertEqual(Set(cacheBoundaryLeafOffsets(warm.cache)), Set([n + 1]))
      XCTAssertEqual(Set(cacheBoundaryLeafOffsets(cold.cache)), Set([n + 1]))
      assertState(warm.cache, equals: cold.cache)
      var warmTokens: [Int] = []
      var coldTokens: [Int] = []
      let warmStart = ProcessInfo.processInfo.systemUptime
      while let token = warm.next() { warmTokens.append(token) }
      let warmSeconds = ProcessInfo.processInfo.systemUptime - warmStart
      let coldStart = ProcessInfo.processInfo.systemUptime
      while let token = cold.next() { coldTokens.append(token) }
      let coldSeconds = ProcessInfo.processInfo.systemUptime - coldStart
      print("[TinyNativeMTPRetry] warm_tokens=\(warmTokens.count) warm_decode_seconds=\(warmSeconds) warm_tokens_per_second=\(Double(warmTokens.count) / warmSeconds) cold_tokens=\(coldTokens.count) cold_decode_seconds=\(coldSeconds) cold_tokens_per_second=\(Double(coldTokens.count) / coldSeconds) scope=untrained_fixture_not_model_speed")
      XCTAssertEqual(warmTokens.count, 4)
      XCTAssertEqual(coldTokens.count, 4)
      XCTAssertEqual(warmTokens, coldTokens)
      XCTAssertNil(warm.generationFailure)
      XCTAssertNil(cold.generationFailure)
    }
  }

  private func assertState(
    _ actual: [KVCache], equals expected: [KVCache],
    file: StaticString = #filePath, line: UInt = #line
  ) {
    XCTAssertEqual(actual.count, expected.count, file: file, line: line)
    guard actual.count == expected.count else { return }
    for (left, right) in zip(actual, expected) {
      XCTAssertEqual(left.offset, right.offset, file: file, line: line)
      XCTAssertEqual(
        String(describing: type(of: left)), String(describing: type(of: right)),
        file: file, line: line)
      let lhs = left.state
      let rhs = right.state
      XCTAssertEqual(lhs.count, rhs.count, file: file, line: line)
      guard lhs.count == rhs.count else { continue }
      for (a, b) in zip(lhs, rhs) {
        XCTAssertEqual(a.shape, b.shape, file: file, line: line)
        XCTAssertEqual(a.dtype, b.dtype, file: file, line: line)
        guard a.shape == b.shape, a.size > 0 else { continue }
        let error = abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
        XCTAssertTrue(error.isFinite, file: file, line: line)
        XCTAssertLessThanOrEqual(
          error, 1e-5,
          "Actual F32 hybrid state must match the captured seed/cold reference",
          file: file, line: line)
      }
    }
  }

  private final class ProgressRows: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [PrefillProgress] = []
    func append(_ value: PrefillProgress) {
      lock.lock()
      defer { lock.unlock() }
      values.append(value)
    }
    func snapshot() -> [PrefillProgress] {
      lock.lock()
      defer { lock.unlock() }
      return values
    }
  }

  // Same real two-layer hybrid topology as NativeMTPPromptPrimingTests:
  // one GDN/recurrent layer, one attention layer and one recursive MTP head.
  private static let config = #"""
    {"model_type":"qwen4_exp","text_config":{
      "model_type":"qwen4_exp_text","dtype":"float32","mamba_ssm_dtype":"float32",
      "mtp_num_hidden_layers":1,"hidden_size":64,"num_hidden_layers":2,
      "intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":1,"head_dim":16,
      "linear_num_value_heads":4,"linear_num_key_heads":1,"linear_key_head_dim":16,
      "linear_value_head_dim":16,"linear_conv_kernel_dim":4,"vocab_size":128,
      "num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16,
      "shared_expert_intermediate_size":16,"layer_types":["linear_attention","full_attention"],
      "hc_count":4,"hc_lowrank":8,"ple_layer_ids":[],"ple_embed_dim":64,
      "ple_conv_kernel_size":4,"ngram_size":3,"heads_per_ngram":2,"ngram_vocab_size_base":101,
      "make_ngram_vocab_size_divisible_by":128,"seed":1234,"split_ngram_parts":4,
      "indexer_n_heads":2,"indexer_kv_heads":1,"indexer_head_dim":8,
      "indexer_budget":32,"indexer_compress_ratio":4}}
    """#
}
