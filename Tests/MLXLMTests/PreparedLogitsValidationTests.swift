import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

private final class PreparedLogitsFixture: Module, LanguageModel, @unchecked Sendable {
    var outputShape: [Int]
    var preparedOffsets: [Int] = []
    var preparedTokenCounts: [Int] = []
    let failProjection: Bool
    let mutateCache: Bool
    var returnsTokens = false
    var cancelPreparation = false
    var vocabularySize: Int { 4 }

    init(shape: [Int], failProjection: Bool = false, mutateCache: Bool = false) {
        self.outputShape = shape
        self.failProjection = failProjection
        self.mutateCache = mutateCache
        super.init()
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        mutateCache ? [KVCacheSimple()] : []
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        if cancelPreparation { throw CancellationError() }
        preparedOffsets.append(cache.first?.offset ?? 0)
        preparedTokenCounts.append(input.text.tokens.size)
        if mutateCache {
            let row = MLXArray.ones([1, 1, input.text.tokens.size, 4])
            for layer in cache { _ = layer.update(keys: row, values: row) }
        }
        if returnsTokens { return .tokens(input.text) }
        if failProjection {
            let invalid = matmul(MLXArray.zeros([2, 3]), MLXArray.zeros([4, 2]))
            return .logits(LMOutput(logits: invalid))
        }
        let count = outputShape.reduce(1, *)
        let logits = MLXArray(0 ..< count).asType(.float32).reshaped(outputShape)
        return .logits(LMOutput(logits: logits))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        if returnsTokens {
            if failProjection {
                return matmul(MLXArray.zeros([2, 3]), MLXArray.zeros([4, 2]))
            }
            return MLXArray(0 ..< outputShape.reduce(1, *)).asType(.float32).reshaped(outputShape)
        }
        return MLXArray([Float(0), 1, 2, 3]).reshaped(1, 1, 4)
    }
}

private final class FailingHistoryProcessor: UserInputProcessor, @unchecked Sendable {
    let underlying: TestInputProcessor
    var failNext = true

    init() {
        underlying = TestInputProcessor(
            tokenizer: PreparationHistoryTokenizer(),
            configuration: ModelConfiguration(id: "fixture/history-retry"),
            messageGenerator: DefaultMessageGenerator())
    }

    func prepare(input: UserInput) throws -> LMInput {
        if failNext {
            failNext = false
            throw NSError(domain: "history-template", code: 1)
        }
        return try underlying.prepare(input: input)
    }
}

final class PreparedLogitsValidationTests: XCTestCase {
    func testBatchSubmitValidatesPreparedOutputAndPreservesFailure() async throws {
        try await MLXMetalTestLock.withLock {
            // submit always uses real scheduler slots, bypassing generate's solo iterator.
            for shape in [[1, 4], [1, 0, 4], [1, 1, 4]] {
                let fixture = PreparedLogitsFixture(shape: shape, mutateCache: true)
                let processor = TestInputProcessor()
                let context = ModelContext(
                    configuration: processor.configuration, model: fixture,
                    processor: processor, tokenizer: processor.tokenizer)
                let engine = BatchEngine(context: context, maxBatchSize: 2)
                let parameters = GenerateParameters(maxTokens: 1, temperature: 0)
                let (_, first) = await engine.submit(
                    input: LMInput(tokens: MLXArray([Int32(1), 2])), parameters: parameters)
                let (_, second) = await engine.submit(
                    input: LMInput(tokens: MLXArray([Int32(2), 1])), parameters: parameters)
                for stream in [first, second] {
                    var infos: [GenerateCompletionInfo] = []
                    var tokens: [Int] = []
                    for await event in stream {
                        if case .info(let info) = event { infos.append(info) }
                        if case .token(let token) = event { tokens.append(token) }
                    }
                    XCTAssertEqual(infos.count, 1)
                    if shape == [1, 1, 4] {
                        XCTAssertNil(infos.first?.generationFailure)
                        XCTAssertEqual(tokens, [3])
                    } else {
                        XCTAssertTrue(tokens.isEmpty)
                        XCTAssertEqual(infos.first?.stopReason, .cancelled)
                        XCTAssertEqual(infos.first?.generationFailure?.stage, .preparation)
                        XCTAssertEqual(
                            infos.first?.generationFailure?.cause,
                            PreparedLogitsValidationError.invalidShape(shape).localizedDescription)
                    }
                }
                XCTAssertEqual(fixture.preparedOffsets, [0, 0])
                let highWatermark = await engine.activeCountHighWatermarkForDiagnostics
                XCTAssertEqual(
                    highWatermark, 2, "Both requests must be admitted to scheduler slots")
            }
        }
    }

    func testBatchTokenTailFailureDoesNotPublishCache() async throws {
        try await MLXMetalTestLock.withLock {
            for projectionFailure in [false, true] {
                let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
                    UUID().uuidString)
                defer { try? FileManager.default.removeItem(at: directory) }
                let coordinator = CacheCoordinator(
                    config: CacheCoordinatorConfig(
                        usePagedCache: false, enableDiskCache: true, diskCacheDir: directory,
                        modelKey: "batch-token-tail-failure"))
                let fixture = PreparedLogitsFixture(
                    shape: [1, 4], failProjection: projectionFailure, mutateCache: true)
                fixture.returnsTokens = true
                let processor = TestInputProcessor()
                let context = ModelContext(
                    configuration: processor.configuration, model: fixture,
                    processor: processor, tokenizer: processor.tokenizer)
                let engine = BatchEngine(
                    context: context, maxBatchSize: 2, cacheCoordinator: coordinator)
                let (_, stream) = await engine.submit(
                    input: LMInput(tokens: MLXArray([Int32(1), 2])),
                    parameters: GenerateParameters(maxTokens: 1, temperature: 0))
                var infos: [GenerateCompletionInfo] = []
                for await event in stream {
                    if case .token = event { XCTFail("Failed token-tail projection must not emit") }
                    if case .info(let info) = event { infos.append(info) }
                }
                XCTAssertEqual(infos.count, 1)
                XCTAssertEqual(infos.first?.generationFailure?.stage, .preparation)
                if projectionFailure {
                    XCTAssertTrue(infos.first?.generationFailure?.cause.contains("matmul") == true)
                } else {
                    XCTAssertEqual(
                        infos.first?.generationFailure?.cause,
                        PreparedLogitsValidationError.invalidShape([1, 4]).localizedDescription)
                }
                XCTAssertEqual(coordinator.diskCache?.stores, 0)
                if case .hit = coordinator.fetch(tokens: [1, 2]) {
                    XCTFail("Failed token-tail cache must not be reusable")
                }
            }
        }
    }

    func testBatchSubmitPreservesMLXErrorAndExplicitCancellation() async throws {
        try await MLXMetalTestLock.withLock {
            for cancel in [false, true] {
                let fixture = PreparedLogitsFixture(shape: [1, 1, 4], failProjection: !cancel)
                fixture.cancelPreparation = cancel
                let processor = TestInputProcessor()
                let context = ModelContext(
                    configuration: processor.configuration, model: fixture,
                    processor: processor, tokenizer: processor.tokenizer)
                let engine = BatchEngine(context: context, maxBatchSize: 2)
                let (_, stream) = await engine.submit(
                    input: LMInput(tokens: MLXArray([Int32(1)])),
                    parameters: GenerateParameters(maxTokens: 1, temperature: 0))
                var infos: [GenerateCompletionInfo] = []
                for await event in stream {
                    if case .token = event { XCTFail("Failed prepare must not emit tokens") }
                    if case .info(let info) = event { infos.append(info) }
                }
                XCTAssertEqual(infos.count, 1)
                XCTAssertEqual(infos.first?.stopReason, .cancelled)
                if cancel {
                    XCTAssertNil(infos.first?.generationFailure)
                } else {
                    XCTAssertTrue(infos.first?.generationFailure?.cause.contains("matmul") == true)
                }
            }
        }
    }

    func testProcessorFailurePreservesPendingSessionHistoryOnRetry() async throws {
        try await MLXMetalTestLock.withLock {
            let fixture = PreparedLogitsFixture(shape: [1, 1, 4], mutateCache: true)
            let processor = FailingHistoryProcessor()
            let context = ModelContext(
                configuration: processor.underlying.configuration, model: fixture,
                processor: processor, tokenizer: processor.underlying.tokenizer)
            let session = ChatSession(
                context, history: [.user("retained fact"), .assistant("remembered")],
                generateParameters: GenerateParameters(maxTokens: 1, temperature: 0))
            do {
                _ = try await session.respond(to: "template fails")
                XCTFail("Expected processor failure")
            } catch {
                XCTAssertEqual((error as NSError).domain, "history-template")
            }
            XCTAssertTrue(fixture.preparedOffsets.isEmpty)
            _ = try await session.respond(to: "retry with history")
            XCTAssertEqual(fixture.preparedTokenCounts, [3])
            XCTAssertEqual(fixture.preparedOffsets, [0])
        }
    }

    func testSessionInvalidatesFailedPreparationUntilExplicitReset() async throws {
        try await MLXMetalTestLock.withLock {
            let fixture = PreparedLogitsFixture(shape: [1, 1, 4], mutateCache: true)
            let tokenizer = PreparationHistoryTokenizer()
            let processor = TestInputProcessor(
                tokenizer: tokenizer,
                configuration: ModelConfiguration(id: "fixture/session"),
                messageGenerator: DefaultMessageGenerator())
            let context = ModelContext(
                configuration: processor.configuration, model: fixture,
                processor: processor, tokenizer: tokenizer)
            let parameters = GenerateParameters(maxTokens: 1, temperature: 0)
            let session = ChatSession(context, generateParameters: parameters)
            _ = try await session.respond(to: "first fact")
            fixture.outputShape = [1, 4]
            do {
                _ = try await session.respond(to: "failing continuation")
                XCTFail("Expected malformed prepared logits")
            } catch is PreparedLogitsValidationError {}
            XCTAssertGreaterThan(
                fixture.preparedOffsets.last ?? 0, 0,
                "Failure fixture must begin with retained conversation state")
            let callsAfterFailure = fixture.preparedOffsets.count
            await session.withCache { XCTAssertNil($0) }
            fixture.outputShape = [1, 1, 4]
            do {
                _ = try await session.respond(to: "must not silently lose history")
                XCTFail("Invalid session must require explicit reset/history replay")
            } catch ChatSessionError.cacheInvalidatedAfterPreparation {}
            XCTAssertEqual(fixture.preparedOffsets.count, callsAfterFailure)
            let url = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: url) }
            do {
                try await session.saveCache(to: url)
                XCTFail("Failed session cache must not be saved")
            } catch ChatSessionError.cacheInvalidatedAfterPreparation {}
            XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
            await session.clear()
            _ = try await session.respond(to: "explicitly new conversation")
            XCTAssertEqual(fixture.preparedOffsets.last, 0)
            let restored = ChatSession(
                context,
                history: [.user("first fact"), .assistant("remembered fact")],
                generateParameters: parameters)
            _ = try await restored.respond(to: "continuation after replay")
            XCTAssertEqual(fixture.preparedOffsets.last, 0)
            XCTAssertEqual(
                fixture.preparedTokenCounts.last, 3,
                "A replacement session must prepare complete supplied history plus next turn")
        }
    }

    func testSoloTokenTailFailureTransportAndCancellationDrain() async throws {
        try await MLXMetalTestLock.withLock {
            for cancel in [false, true] {
                let fixture = PreparedLogitsFixture(
                    shape: [1, 1, 4], failProjection: true, mutateCache: true)
                fixture.returnsTokens = true
                fixture.cancelPreparation = cancel
                let processor = TestInputProcessor()
                let context = ModelContext(
                    configuration: processor.configuration, model: fixture,
                    processor: processor, tokenizer: processor.tokenizer)
                let engine = BatchEngine(context: context, maxBatchSize: 1)
                let stream = await engine.generate(
                    input: LMInput(tokens: MLXArray([Int32(1), 2])),
                    parameters: GenerateParameters(maxTokens: 1, temperature: 0))
                var infos: [GenerateCompletionInfo] = []
                for await event in stream {
                    switch event {
                    case .info(let info): infos.append(info)
                    case .prefillProgress: break
                    default: XCTFail("Failed solo preparation must not emit model output")
                    }
                }
                XCTAssertEqual(infos.count, 1)
                XCTAssertEqual(infos.first?.generationTokenCount, 0)
                XCTAssertEqual(infos.first?.stopReason, .cancelled)
                if cancel {
                    XCTAssertNil(infos.first?.generationFailure)
                } else {
                    XCTAssertEqual(infos.first?.generationFailure?.stage, .preparation)
                    XCTAssertTrue(infos.first?.generationFailure?.cause.contains("matmul") == true)
                }
                let active = await engine.activeCount
                let soloActive = await engine.isSoloFastPathActiveForTesting
                XCTAssertEqual(active, 0, "The producer must drain before stream termination")
                XCTAssertFalse(soloActive)
                await engine.shutdown()
            }
        }
    }

    func testTokenTailMalformedShapesThrowBeforeSampling() throws {
        try MLXMetalTestLock.withLock {
            for shape in [[], [4], [1, 4], [1, 1, 1, 4], [0, 1, 4], [1, 0, 4], [1, 1, 0]] {
                let fixture = PreparedLogitsFixture(shape: shape)
                fixture.returnsTokens = true
                XCTAssertThrowsError(
                    try TokenIterator(
                        input: LMInput(tokens: MLXArray([Int32(1)])), model: fixture,
                        parameters: GenerateParameters(maxTokens: 1, temperature: 0))
                ) { error in
                    XCTAssertEqual(error as? PreparedLogitsValidationError, .invalidShape(shape))
                }
            }
        }
    }

    func testTokenTailProjectionPreservesOriginatingErrorAndFreshRequest() throws {
        try MLXMetalTestLock.withLock {
            let failing = PreparedLogitsFixture(shape: [1, 1, 4], failProjection: true)
            failing.returnsTokens = true
            XCTAssertThrowsError(
                try TokenIterator(
                    input: LMInput(tokens: MLXArray([Int32(1), 2])), model: failing,
                    parameters: GenerateParameters(maxTokens: 1, temperature: 0))
            ) { error in
                guard let mlxError = error as? MLXError, case .caught(let message) = mlxError else {
                    return XCTFail("Expected originating MLX projection error, got \(error)")
                }
                XCTAssertTrue(message.contains("matmul"), message)
                XCTAssertFalse(error is PreparedLogitsValidationError)
            }
            // Both single-token and remaining multi-token prepare paths retain
            // the same last-position greedy result after a failed request.
            for length in [1, 3] {
                let valid = PreparedLogitsFixture(shape: [1, length, 4])
                valid.returnsTokens = true
                var iterator = try TokenIterator(
                    input: LMInput(tokens: MLXArray(Array(repeating: Int32(1), count: length))),
                    model: valid, parameters: GenerateParameters(maxTokens: 1, temperature: 0))
                XCTAssertEqual(iterator.next(), 3)
            }
        }
    }

    func testTokenTailFailureDoesNotPublishMutatedCache() throws {
        try MLXMetalTestLock.withLock {
            for projectionFailure in [false, true] {
                let directory = FileManager.default.temporaryDirectory
                    .appendingPathComponent("token-tail-failure-\(UUID().uuidString)")
                defer { try? FileManager.default.removeItem(at: directory) }
                let coordinator = CacheCoordinator(
                    config: CacheCoordinatorConfig(
                        usePagedCache: false, enableDiskCache: true, diskCacheDir: directory,
                        modelKey: "token-tail-failure"))
                let fixture = PreparedLogitsFixture(
                    shape: [1, 4], failProjection: projectionFailure, mutateCache: true)
                fixture.returnsTokens = true
                let input = LMInput(tokens: MLXArray([Int32(1)]))
                let parameters = GenerateParameters(maxTokens: 1, temperature: 0)
                let callerCache = KVCacheSimple()
                XCTAssertThrowsError(
                    try TokenIterator(
                        input: input, model: fixture, cache: [callerCache], parameters: parameters,
                        cacheCoordinator: coordinator))
                // Ownership remains with the caller on failure; no implicit rollback.
                XCTAssertEqual(callerCache.offset, 1)
                XCTAssertEqual(coordinator.diskCache?.stores, 0)
                let salt = computeCacheSalt(for: input, parameters: parameters)
                guard case .miss = coordinator.fetch(tokens: [1], mediaSalt: salt) else {
                    return XCTFail("Failed token-tail projection must not publish cache")
                }
            }
        }
    }

    func testSessionInvalidatesTokenTailFailureUntilReset() async throws {
        try await MLXMetalTestLock.withLock {
            let fixture = PreparedLogitsFixture(shape: [1, 1, 4], mutateCache: true)
            fixture.returnsTokens = true
            let processor = TestInputProcessor()
            let context = ModelContext(
                configuration: processor.configuration, model: fixture,
                processor: processor, tokenizer: processor.tokenizer)
            let session = ChatSession(
                context, generateParameters: GenerateParameters(maxTokens: 1, temperature: 0))
            _ = try await session.respond(to: "first fact")
            fixture.outputShape = [1, 4]
            do {
                _ = try await session.respond(to: "failing continuation")
                XCTFail("Expected malformed token-tail projection")
            } catch is PreparedLogitsValidationError {}
            XCTAssertGreaterThan(fixture.preparedOffsets.last ?? 0, 0)
            await session.withCache { XCTAssertNil($0) }
            let callsAfterFailure = fixture.preparedOffsets.count
            fixture.outputShape = [1, 1, 4]
            do {
                _ = try await session.respond(to: "must require history replay")
                XCTFail("Invalid session must require explicit reset/history replay")
            } catch ChatSessionError.cacheInvalidatedAfterPreparation {}
            XCTAssertEqual(fixture.preparedOffsets.count, callsAfterFailure)
            await session.clear()
            _ = try await session.respond(to: "new conversation")
            XCTAssertEqual(fixture.preparedOffsets.last, 0)
        }
    }

    func testMalformedPreparedShapesThrowBeforeSampling() throws {
        try MLXMetalTestLock.withLock {
            for shape in [[], [4], [1, 4], [1, 1, 1, 4], [0, 1, 4], [1, 0, 4], [1, 1, 0]] {
                XCTAssertThrowsError(
                    try TokenIterator(
                        input: LMInput(tokens: MLXArray([Int32(1)])),
                        model: PreparedLogitsFixture(shape: shape),
                        parameters: GenerateParameters(maxTokens: 1, temperature: 0))
                ) { error in
                    XCTAssertEqual(error as? PreparedLogitsValidationError, .invalidShape(shape))
                }
            }
        }
    }

    func testOriginalProjectionErrorIsNotReplacedByShapeError() throws {
        try MLXMetalTestLock.withLock {
            XCTAssertThrowsError(
                try TokenIterator(
                    input: LMInput(tokens: MLXArray([Int32(1)])),
                    model: PreparedLogitsFixture(shape: [1, 1, 4], failProjection: true),
                    parameters: GenerateParameters(maxTokens: 1, temperature: 0))
            ) { error in
                guard let mlxError = error as? MLXError, case .caught(let message) = mlxError else {
                    return XCTFail("Expected originating MLX projection error, got \(error)")
                }
                XCTAssertTrue(message.contains("matmul"), message)
                XCTAssertFalse(error is PreparedLogitsValidationError)
            }
            // A fresh valid request must not inherit the failed request's scope.
            var iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray([Int32(1)])),
                model: PreparedLogitsFixture(shape: [1, 1, 4]),
                parameters: GenerateParameters(maxTokens: 1, temperature: 0))
            XCTAssertEqual(iterator.next(), 3)
        }
    }

    func testFailedPrepareDoesNotPublishMutatedCache() throws {
        try MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("prepared-logits-failure-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: directory) }
            let coordinator = CacheCoordinator(
                config: CacheCoordinatorConfig(
                    usePagedCache: false, enableDiskCache: true, diskCacheDir: directory,
                    modelKey: "prepared-logits-failure"))
            let input = LMInput(tokens: MLXArray([Int32(1)]))
            let parameters = GenerateParameters(maxTokens: 1, temperature: 0)
            let callerCache = KVCacheSimple()
            XCTAssertThrowsError(
                try TokenIterator(
                    input: input, model: PreparedLogitsFixture(shape: [1, 4], mutateCache: true),
                    cache: [callerCache], parameters: parameters, cacheCoordinator: coordinator))
            // A throwing prepare is not transactional for caller-owned objects.
            // Establish that this fixture really mutated one before the failure.
            XCTAssertEqual(callerCache.offset, 1)
            XCTAssertEqual(coordinator.diskCache?.stores, 0)
            let salt = computeCacheSalt(for: input, parameters: parameters)
            guard case .miss = coordinator.fetch(tokens: [1], mediaSalt: salt) else {
                return XCTFail("Failed prepared output must not publish a prefix checkpoint")
            }
            // Discard the caller cache after failure; test the supported fresh
            // request path without implicitly claiming rollback or B=2 support.
            var next = try TokenIterator(
                input: input, model: PreparedLogitsFixture(shape: [1, 1, 4]),
                parameters: parameters, cacheCoordinator: coordinator)
            XCTAssertEqual(next.next(), 3)
            XCTAssertEqual(coordinator.diskCache?.stores, 0)
        }
    }

    func testValidSingleAndBatchedGeometryPreservesLastRows() throws {
        try MLXMetalTestLock.withLock {
            for shape in [[1, 1, 4], [1, 3, 4], [2, 1, 4], [2, 3, 4]] {
                let logits = MLXArray(0 ..< shape.reduce(1, *)).asType(.float32).reshaped(shape)
                try validatePreparedLogitsForSampling(logits)
                let row = logits[0..., -1, 0...]
                let expected = (0 ..< shape[0]).flatMap { batch in
                    (0 ..< shape[2]).map {
                        Float((batch * shape[1] + shape[1] - 1) * shape[2] + $0)
                    }
                }
                XCTAssertEqual(row.asArray(Float.self), expected)
            }
            // TokenIterator is single-request; batched geometry above tests only
            // the rank contract, not scalar iteration over multiple sequences.
            for length in [1, 3] {
                var iterator = try TokenIterator(
                    input: LMInput(tokens: MLXArray([Int32(1)])),
                    model: PreparedLogitsFixture(shape: [1, length, 4]),
                    parameters: GenerateParameters(maxTokens: 1, temperature: 0))
                XCTAssertEqual(iterator.next(), 3)
            }
        }
    }
}

private struct PreparationHistoryTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [1] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "answer" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { "answer" }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        Array(repeating: 1, count: messages.count)
    }
}
