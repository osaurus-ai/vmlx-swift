import Foundation
import MLX
import MLXNN
import XCTest
@testable import MLXLMCommon

final class NativeMTPBoundaryCaptureTests: XCTestCase {
    func testMaskSplitAndAuxiliaryExclusion() throws {
        try FocusedMLXTestSupport.withLock {
            for mode in 0..<3 {
                let directory = FileManager.default.temporaryDirectory
                    .appendingPathComponent("native-mask-\(UUID().uuidString)")
                defer { try? FileManager.default.removeItem(at: directory) }
                let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
                    usePagedCache: false, enableDiskCache: true,
                    diskCacheDir: directory, modelKey: "native-mask-fixture"))
                coordinator.setHybrid(true)
                coordinator.setGenPromptSuffixTokens([21])
                let ids = [11, 12, 21, 22]
                let mask = mode == 1
                    ? MLXArray.ones([1, 1, 4, 4])
                    : MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4])
                let model = BoundaryRecordingTarget(returnsLogits: true)
                let input = LMInput(
                    text: .init(tokens: MLXArray(ids.map(Int32.init))[.newAxis],
                                mask: mask, tokenIds: ids),
                    cacheScopeSalt: "reasoning=off",
                    cachePromptIntent: mode == 2 ? .auxiliary : .generation)
                let iterator = try NativeMTPTokenIterator(
                    input: input, model: model,
                    parameters: GenerateParameters(maxTokens: 16, temperature: 0),
                    depth: 3, cacheCoordinator: coordinator)
                XCTAssertEqual(model.forwarded, ids + [1])
                XCTAssertEqual(model.preparedInputs.count, mode == 0 ? 2 : 1)
                if mode == 0 {
                    XCTAssertEqual(model.preparedInputs.map { $0.text.tokenIds! }, [[11, 12], [21, 22]])
                    XCTAssertEqual(model.preparedInputs.map { $0.text.mask!.asArray(Int32.self) },
                                   [[1, 2], [3, 4]])
                    XCTAssertTrue(model.preparedInputs.allSatisfy { $0.cacheScopeSalt == "reasoning=off" })
                    XCTAssertNotNil(iterator.strippedPromptSnapshot)
                } else {
                    XCTAssertNil(iterator.strippedPromptSnapshot)
                }
            }
        }
    }

    func testCancellationAfterHeadPublishesNothing() async throws {
        try await FocusedMLXTestSupport.withLock {
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("native-cancel-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: directory) }
            let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
                usePagedCache: false, enableDiskCache: true,
                diskCacheDir: directory, modelKey: "native-cancel-fixture"))
            coordinator.setHybrid(true)
            coordinator.setGenPromptSuffixTokens([21])
            let model = BoundaryRecordingTarget(returnsLogits: true)
            model.afterPrepare = { withUnsafeCurrentTask { $0?.cancel() } }
            do {
                _ = try NativeMTPTokenIterator(
                    input: LMInput(tokens: MLXArray([Int32(11), 12, 21, 22])[.newAxis]),
                    model: model, parameters: GenerateParameters(maxTokens: 16, temperature: 0),
                    depth: 3, cacheCoordinator: coordinator)
                XCTFail("Expected cancellation before suffix/bridge")
            } catch is CancellationError {
                XCTAssertEqual(model.forwarded, [11, 12])
                XCTAssertEqual(coordinator.snapshotStats().diskStats?.stores ?? 0, 0)
            }
        }
    }

    func testStrippedBoundaryDoesNotReplayPromptAtStore() throws {
        try FocusedMLXTestSupport.withLock {
            for returnsLogits in [false, true] {
                for depth in 1...3 {
                    let directory = FileManager.default.temporaryDirectory
                        .appendingPathComponent("native-boundary-\(UUID().uuidString)")
                    defer { try? FileManager.default.removeItem(at: directory) }
                    let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
                        usePagedCache: false, enableDiskCache: true,
                        diskCacheDir: directory, modelKey: "native-boundary-fixture"))
                    coordinator.setHybrid(true)
                    coordinator.setGenPromptSuffixTokens([21, 22, 23])
                    let model = BoundaryRecordingTarget(returnsLogits: returnsLogits)
                    var parameters = GenerateParameters(
                        maxTokens: 16, temperature: 0, prefillStepSize: 3)
                    parameters.draftStrategy = .nativeMTP(depth: depth)
                    parameters.nativeMTPDepthPolicy = .fixed
                    let prompt = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23]
                    let input = LMInput(tokens: MLXArray(prompt.map(Int32.init))[.newAxis])
                    var iterator = try NativeMTPTokenIterator(
                        input: input, model: model, parameters: parameters,
                        depth: depth, cacheCoordinator: coordinator)
                    // Construction also forwards the first sampled token to prime the bridge.
                    XCTAssertEqual(model.forwarded, prompt + [1])
                    XCTAssertEqual(iterator.promptCacheSnapshot?.first?.offset, prompt.count,
                                   "The exact snapshot must include the final prepare remainder")
                    let captured = try XCTUnwrap(iterator.strippedPromptSnapshot)
                    XCTAssertEqual(captured.boundary, 7)
                    XCTAssertEqual(captured.cache.first?.offset, 7)
                    XCTAssertEqual(captured.cache.first?.state.first?.asArray(Float.self), [98],
                                   "Later suffix/bridge forwards must not mutate the owned boundary")
                    let beforeStore = model.forwarded
                    iterator.storeCacheAfterGeneration(
                        generatedTokenIds: [], includeGeneratedBoundary: false)
                    XCTAssertEqual(model.forwarded, beforeStore,
                                   "The reusable boundary must come from live prefill, not replay")
                    let hit = coordinator.fetch(
                        tokens: prompt + [9],
                        mediaSalt: computeCacheSalt(for: input, parameters: parameters),
                        skipExactDiskBoundary: true)
                    guard case .hit(let count, _, let tier, _, _, _) = hit else {
                        XCTFail("No disk-restorable boundary"); continue
                    }
                    XCTAssertEqual(count, 7)
                    XCTAssertEqual(tier, .disk)
                    // Restore the published recurrent state, not merely its key.
                    model.forwarded = []
                    var warm = try NativeMTPTokenIterator(
                        input: input, model: model, parameters: parameters,
                        depth: depth, cacheCoordinator: coordinator)
                    XCTAssertEqual(model.forwarded, [21, 22, 23, 1])
                    XCTAssertEqual(warm.strippedPromptSnapshot?.cache.first?.state.first?
                        .asArray(Float.self), [98])
                    XCTAssertEqual(warm.promptCacheSnapshot?.first?.state.first?
                        .asArray(Float.self), [164])
                    let warmForwards = model.forwarded
                    warm.storeCacheAfterGeneration(
                        generatedTokenIds: [], includeGeneratedBoundary: false)
                    XCTAssertEqual(model.forwarded, warmForwards)
                    model.forwarded = []
                    let grownPrompt = Array(prompt.prefix(7)) + [24, 25, 21, 22, 23]
                    var grown = try NativeMTPTokenIterator(
                        input: LMInput(tokens: MLXArray(grownPrompt.map(Int32.init))[.newAxis]),
                        model: model, parameters: parameters,
                        depth: depth, cacheCoordinator: coordinator)
                    XCTAssertEqual(model.forwarded, [24, 25, 21, 22, 23, 1])
                    XCTAssertEqual(grown.strippedPromptSnapshot?.boundary, 9)
                    XCTAssertEqual(grown.strippedPromptSnapshot?.cache.first?.state.first?
                        .asArray(Float.self), [147])
                    let grownForwards = model.forwarded
                    grown.storeCacheAfterGeneration(
                        generatedTokenIds: [], includeGeneratedBoundary: false)
                    XCTAssertEqual(model.forwarded, grownForwards)
                }
            }
        }
    }
}

private final class BoundaryRecordingTarget: Module, NativeMTPModel, @unchecked Sendable {
    let returnsLogits: Bool
    var forwarded: [Int] = []
    var preparedInputs: [LMInput] = []
    var afterPrepare: (() -> Void)?
    var vocabularySize: Int { 32 }
    var nativeMTPAvailable: Bool { true }
    init(returnsLogits: Bool) { self.returnsLogits = returnsLogits; super.init() }
    // Hybrid restore derives the matched token count from the attention lane.
    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [MambaCache(), KVCacheSimple()]
    }
    func makeNativeMTPCache() -> [KVCache] { [] }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        preparedInputs.append(input)
        defer { afterPrepare?() }
        if returnsLogits {
            return .logits(LMOutput(logits: callAsFunction(input.text.tokens, cache: cache)))
        }
        return .tokens(input.text)
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        nativeBackboneForward(inputs, cache: cache).logits
    }
    func nativeBackboneForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        let ids = inputs.reshaped([-1]).asArray(Int32.self).map(Int.init)
        forwarded += ids
        if let cache = cache?.first as? MambaCache {
            let previous = cache.state.first?.asArray(Float.self).first ?? 0
            cache[0] = MLXArray([previous + Float(ids.reduce(0, +))]).reshaped([1, 1, 1])
            cache.offset += ids.count
        }
        if let attention = cache?.last as? KVCacheSimple {
            let rows = MLXArray(ids.map(Float.init)).reshaped([1, 1, ids.count, 1])
            _ = attention.update(keys: rows, values: rows)
        }
        return result(count: inputs.size)
    }
    func nativeMTPForward(
        hiddenStates: MLXArray, nextTokenIds: MLXArray, cache: [KVCache]?
    ) -> NativeMTPForwardResult { result(count: nextTokenIds.size) }
    private func result(count: Int) -> NativeMTPForwardResult {
        var values = Array(repeating: Float(-100), count: 32)
        values[1] = 100
        return NativeMTPForwardResult(
            logits: broadcast(MLXArray(values), to: [1, count, 32]),
            hiddenStates: MLXArray.zeros([1, count, 1]))
    }
}
