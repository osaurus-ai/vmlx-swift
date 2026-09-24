import Foundation
import MLX
import MLXNN
import MLXFast
import Testing
import os
import MLXLMCommon

@Suite("Auxiliary hybrid prefill disk isolation", .serialized)
struct AuxiliaryPrefillSeedTests {
    @Test(arguments: [LMInput.CachePromptIntent.generation, .reusablePrefixWarmup, .auxiliary])
    func prefillSeedRespectsIntent(_ intent: LMInput.CachePromptIntent) async throws {
        try await FocusedMLXTestSupport.withLock {
            let root = FileManager.default.temporaryDirectory
                .appendingPathComponent("auxiliary-prefill-seed-\(UUID())")
            defer { try? FileManager.default.removeItem(at: root) }
            let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
                usePagedCache: false, enableDiskCache: true,
                diskCacheMaxGB: 0.01, diskCacheDir: root, modelKey: "tiny-hybrid"))
            let calls = OSAllocatedUnfairLock(initialState: [Int]())
            let model = SeedModel(calls: calls)
            var configuration = ModelConfiguration(id: "tiny-hybrid")
            configuration.eosTokenIds = [0, 1]
            let context = ModelContext(configuration: configuration, model: model,
                processor: SeedProcessor(), tokenizer: SeedTokenizer())
            // maxBatchSize=2 plus submit exercises batched prefill, not the solo iterator.
            let engine = BatchEngine(context: context, maxBatchSize: 2,
                cacheCoordinator: coordinator)
            let (_, stream) = await engine.submit(
                input: LMInput(tokens: MLXArray(Array(Int32(2)...Int32(9))),
                    cachePromptIntent: intent),
                parameters: GenerateParameters(maxTokens: 2, temperature: 0))
            var stopped = false
            for await event in stream {
                if case .info(let info) = event { stopped = info.stopReason == .stop }
            }
            #expect(stopped)
            let prepared = calls.withLock { $0 }
            let files = FileManager.default.enumerator(at: root,
                includingPropertiesForKeys: nil)?.allObjects as? [URL] ?? []
            let payloads = files.filter { $0.pathExtension == "safetensors" }
            if intent == .auxiliary {
                #expect(prepared == [8], "Utility requests must not create an N-1 snapshot")
                #expect(payloads.isEmpty, "Utility requests must not publish disk boundaries")
            } else {
                #expect(prepared == [7, 1], "Ordinary requests and warmups still capture N-1")
                #expect(!payloads.isEmpty, "Positive control must actually persist a boundary")
                let before = try Dictionary(uniqueKeysWithValues: payloads.map {
                    ($0, try Data(contentsOf: $0))
                })
                let (_, restored) = await engine.submit(
                    input: LMInput(tokens: MLXArray(Array(Int32(2)...Int32(9))),
                        cachePromptIntent: .auxiliary),
                    parameters: GenerateParameters(maxTokens: 2, temperature: 0))
                var restoredStop = false
                for await event in restored {
                    if case .info(let info) = event { restoredStop = info.stopReason == .stop }
                }
                #expect(restoredStop)
                #expect(calls.withLock { $0 } == [7, 1, 1],
                    "Auxiliary requests must restore the N-1 boundary and prefill only the tail")
                let afterFiles = FileManager.default.enumerator(at: root,
                    includingPropertiesForKeys: nil)?.allObjects as? [URL] ?? []
                let after = try Dictionary(uniqueKeysWithValues: afterFiles
                    .filter { $0.pathExtension == "safetensors" }
                    .map { ($0, try Data(contentsOf: $0)) })
                #expect(before == after, "Restore must neither add nor replace payloads")
            }
            await engine.shutdown()
        }
    }
}

private final class SeedModel: Module, LanguageModel {
    let calls: OSAllocatedUnfairLock<[Int]>
    init(calls: OSAllocatedUnfairLock<[Int]>) { self.calls = calls }
    var kvHeads: [Int] { [1] }
    var vocabularySize: Int { 2 }
    func newCache(parameters: GenerateParameters?) -> [KVCache] { [SeedPoolCache()] }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        let count = input.text.tokens.size
        calls.withLock { $0.append(count) }
        return .tokens(input.text)
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let count = inputs.dim(-1)
        let kv = MLXArray.ones([1, 1, count, 4])
        for layer in cache ?? [] { _ = layer.update(keys: kv, values: kv) }
        return MLXArray.zeros([1, count, 2])
    }
}

private final class SeedPoolCache: HybridPoolCache {
    var rotating = RotatingKVCache(maxSize: 16)
    let compressRatio = 4
    let slidingWindow = 16
    var offset: Int { rotating.offset }
    var maxSize: Int? { rotating.maxSize }
    var isTrimmable: Bool { false }
    var metaState: [String] {
        get { rotating.metaState }
        set { rotating.metaState = newValue }
    }
    func innerState() -> [MLXArray] { rotating.state }
    func trim(_ n: Int) -> Int { 0 }
    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        rotating.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }
    var state: [MLXArray] {
        get { rotating.state }
        set { rotating.state = newValue }
    }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let result = rotating.update(keys: keys, values: values)
        return result
    }
    func copy() -> KVCache {
        let result = SeedPoolCache()
        result.rotating = rotating.copy() as! RotatingKVCache
        return result
    }
    func hybridPool(branch: HybridPoolBranch) -> MLXArray? { nil }
    func setHybridPool(branch: HybridPoolBranch, value: MLXArray?) {}
    func hybridBuffers(branch: HybridPoolBranch) -> (kv: MLXArray?, gate: MLXArray?) { (nil, nil) }
    func setHybridBuffers(branch: HybridPoolBranch, kv: MLXArray?, gate: MLXArray?) {}
}

private struct SeedProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput { LMInput(tokens: MLXArray([Int32(2)])) }
}

private struct SeedTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [2] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "fixture" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int] { [2] }
}
