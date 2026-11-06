import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

#if canImport(Cmlx)
    import Cmlx
#endif

private struct BoundaryTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var eosTokenId: Int? { 63 }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [1] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { "t\($0)" }.joined(separator: " ")
    }
    func convertTokenToId(_ token: String) -> Int? { Int(token.dropFirst()) }
    func convertIdToToken(_ id: Int) -> String? { "t\(id)" }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [1] }
}

private struct BoundaryProcessor: UserInputProcessor {
    let tokenizer: any Tokenizer = BoundaryTokenizer()
    let configuration = ModelConfiguration(id: "batch-warm-media-boundary-test")
    func prepare(input: UserInput) throws -> LMInput { LMInput(tokens: MLXArray([Int32(1)])) }
}

/// A deterministic cache recorder, not a language-model quality fixture.
private final class BoundaryHybridModel: Module, LanguageModel, @unchecked Sendable {
    var vocabularySize: Int { 64 }
    private(set) var forwarded: [Int] = []
    private(set) var prepareMediaFlags: [Bool] = []
    private(set) var prepareSegments: [[Int]] = []

    func clearObservations() {
        forwarded.removeAll()
        prepareMediaFlags.removeAll()
        prepareSegments.removeAll()
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [KVCacheSimple(), MambaCache()]
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        prepareMediaFlags.append(input.hasMediaContent)
        prepareSegments.append(input.text.tokens.reshaped(-1).asArray(Int32.self).map(Int.init))
        return .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let ids = inputs.reshaped(-1).asArray(Int32.self).map(Int.init)
        forwarded += ids
        for layer in cache ?? [] {
            if let recurrent = layer as? MambaCache {
                let previous = recurrent.state.first?.asArray(Float.self).first ?? 0
                let sum = previous + ids.reduce(Float(0)) { $0 + Float($1) }
                recurrent.offset += ids.count
                recurrent.state = [MLXArray([sum, Float(recurrent.offset)]).reshaped(1, 1, 2)]
            } else if let attention = layer as? KVCacheSimple {
                let values = inputs.asType(.float32).reshaped(1, 1, ids.count, 1)
                _ = attention.update(keys: values, values: values + 1)
            }
        }
        let vocab = MLXArray(Int32(0) ..< Int32(64))
        return broadcast(
            (vocab .== MLXArray(Int32(4))).asType(.float32) * 100,
            to: [1, ids.count, 64])
    }
}

final class BatchWarmMediaBoundaryTests: XCTestCase {
    private let ids = [3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 47, 53]

    override func setUpWithError() throws {
        try super.setUpWithError()
        #if canImport(Cmlx)
            if let path = ProcessInfo.processInfo.environment["VMLX_TEST_METALLIB_PATH"] {
                guard path.withCString({ mlx_metal_set_metallib_path($0) }) == 0
                else { throw NSError(domain: "BatchWarmMediaBoundaryTests", code: 1) }
            }
        #endif
    }

    private func input(pixel: Float = 1, canonicalBoundaries: Bool = true) -> sending LMInput {
        LMInput(
            text: .init(
                tokens: MLXArray(ids.map(Int32.init)).reshaped(1, ids.count), tokenIds: ids),
            image: .init(pixels: MLXArray.full([1, 3, 2, 2], values: MLXArray(pixel))),
            mediaTokenIds: [3], cachePrefixTokenCounts: canonicalBoundaries ? [11] : [])
    }

    private func exercise(
        prime: Bool, changedImage: Bool = false, canonicalBoundaries: Bool = true,
        maxBatchSize: Int = 1
    ) async throws
        -> (model: BoundaryHybridModel, requiredSuffix: [Int])
    {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let coordinator = CacheCoordinator(
            config: .init(
                usePagedCache: false, enableDiskCache: true, diskCacheDir: root,
                modelKey: "batch-warm-media-boundary-test"))
        coordinator.setHybrid(
            true, requiresRecurrentSSMCompanion: true,
            requiresSeparateRecurrentPayload: false)
        coordinator.setGenPromptSuffixTokens([47, 53])
        let model = BoundaryHybridModel()
        // One generated token bounds this recorder test; length is expected.
        // Real-model natural-stop/coherence proof is a separate gate.
        var parameters = GenerateParameters()
        parameters.maxTokens = 1
        parameters.temperature = 0
        parameters.prefillStepSize = 4
        if prime {
            let seed = model.newCache(parameters: parameters)
            _ = model(MLXArray(ids.prefix(6).map(Int32.init)).reshaped(1, 6), cache: seed)
            coordinator.storeAfterGeneration(
                promptTokens: Array(ids.prefix(6)), perLayerData: [],
                ssmStates: extractSSMStates(from: seed), cache: seed,
                mediaSalt: computeCacheSalt(for: input(), parameters: parameters, model: model),
                isResumeBoundary: true)
            XCTAssertTrue(try XCTUnwrap(coordinator.diskCache).waitForPendingWrites(timeout: 5))
        }
        model.clearObservations()
        let processor = BoundaryProcessor()
        let context = ModelContext(
            configuration: processor.configuration, model: model,
            processor: processor, tokenizer: processor.tokenizer)
        let engine = BatchEngine(
            context: context, maxBatchSize: maxBatchSize, cacheCoordinator: coordinator)
        let task = await engine.submit(
            input: input(pixel: changedImage ? 2 : 1, canonicalBoundaries: canonicalBoundaries),
            parameters: parameters)
        var generated = 0
        for await event in task.stream {
            if case .token = event { generated += 1 }
        }
        XCTAssertEqual(generated, 1, "the recorder must perform a nonempty generation")
        await engine.shutdown()
        XCTAssertTrue(try XCTUnwrap(coordinator.diskCache).waitForPendingWrites(timeout: 5))
        let arrays = try XCTUnwrap(
            coordinator.diskCache?.fetch(
                tokens: Array(ids.prefix(11)),
                mediaSalt: computeCacheSalt(
                    for: input(pixel: changedImage ? 2 : 1), parameters: parameters, model: model)))
        var restored: [KVCache] = [KVCacheSimple(), MambaCache()]
        XCTAssertEqual(
            restoreFromDiskArrays(arrays, into: &restored, requirePromptBoundary: true), 11)
        XCTAssertEqual(restored.map(\.offset), [11, 11])
        XCTAssertEqual(
            restored[1].state[0].asArray(Float.self),
            [ids.prefix(11).reduce(Float(0)) { $0 + Float($1) }, 11])
        return (model, Array(ids.dropFirst(6)))
    }

    func testWarmMediaTextSuffixDoesNotReplayCompletedPrefixAtFinalization() async throws {
        let result = try await exercise(prime: true)
        XCTAssertEqual(
            result.model.forwarded, result.requiredSuffix,
            "only the uncached suffix should be forwarded, never a full post-answer replay")
        XCTAssertTrue(result.model.prepareMediaFlags.allSatisfy { !$0 })
    }

    func testWarmMediaTemplateBoundaryDoesNotReplayWhenCanonicalCountsAreAbsent() async throws {
        for maxBatchSize in [1, 2] {
            let result = try await exercise(
                prime: true, canonicalBoundaries: false, maxBatchSize: maxBatchSize)
            XCTAssertEqual(
                result.model.forwarded, result.requiredSuffix,
                "batch \(maxBatchSize): the existing template strip boundary must be captured "
                    + "during the text-only restored suffix, without replaying the old image")
            XCTAssertTrue(result.model.prepareMediaFlags.allSatisfy { !$0 })
        }
    }

    func testColdMediaKeepsPayloadDuringPreparation() async throws {
        let result = try await exercise(prime: false)
        XCTAssertEqual(result.model.prepareSegments.first, ids)
        XCTAssertEqual(result.model.prepareMediaFlags.first, true)
    }

    func testColdMediaWithoutCanonicalCountsKeepsPayload() async throws {
        let result = try await exercise(prime: false, canonicalBoundaries: false)
        XCTAssertEqual(result.model.prepareSegments.first, ids)
        XCTAssertEqual(result.model.prepareMediaFlags.first, true)
    }

    func testChangedImageWithoutCanonicalCountsCannotReuseTheOldImage() async throws {
        let result = try await exercise(
            prime: true, changedImage: true, canonicalBoundaries: false)
        XCTAssertEqual(result.model.prepareSegments.first, ids)
        XCTAssertEqual(result.model.prepareMediaFlags.first, true)
    }

    func testChangedImageCannotReuseTheOtherImagesPrefix() async throws {
        let result = try await exercise(prime: true, changedImage: true)
        XCTAssertEqual(result.model.prepareSegments.first, ids)
        XCTAssertEqual(result.model.prepareMediaFlags.first, true)
    }
}
