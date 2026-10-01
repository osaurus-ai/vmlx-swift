import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

/// Tiny deterministic next-token fixture; all cache mutation goes through real
/// cache objects and the real iterator/scheduler. No installed weights are used.
private final class PostAnswerEntryFixture: Module, LanguageModel, @unchecked Sendable {
    let wrapped: Bool
    private let observationLock = NSLock()
    private var observedCompiled: CompilableRotatingKVCache?
    var vocabularySize: Int { 8 }

    init(wrapped: Bool = false) {
        self.wrapped = wrapped
        super.init()
    }

    func compiledCache() -> CompilableRotatingKVCache? {
        observationLock.lock()
        defer { observationLock.unlock() }
        return observedCompiled
    }

    func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        let rotating = RotatingKVCache(maxSize: 16)
        return wrapped ? [CacheList(rotating)] : [rotating]
    }

    func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        let tokens = input.text.tokens.reshaped(1, -1)
        return .logits(LMOutput(logits: callAsFunction(tokens, cache: cache)))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let tokens = inputs.reshaped(1, -1)
        let row = broadcast(
            tokens.asType(.float32).reshaped(1, 1, tokens.dim(1), 1),
            to: [1, 2, tokens.dim(1), 4])
        let layer = cache![0]
        let leaf = (layer as? CacheList)?[0] ?? layer
        if let compiled = leaf as? CompilableRotatingKVCache {
            observationLock.lock()
            observedCompiled = compiled
            observationLock.unlock()
        }
        let pair = leaf.update(keys: row, values: row * 2)
        // A derived output keeps cache results connected without returning the
        // mutable state wrappers themselves from a compiled function.
        let vocabulary = MLXArray(Int32(0)..<Int32(8)).reshaped(1, 1, 8)
        let target = (tokens + 1).expandedDimensions(axis: -1)
        return (vocabulary .== target).asType(.float32) * 100 + sum(pair.0) * 0
    }
}

final class CompiledPostAnswerEntrypointTests: XCTestCase {
    private static func config(_ directory: URL) -> CacheCoordinatorConfig {
        CacheCoordinatorConfig(
            usePagedCache: false, enableDiskCache: true, diskCacheMaxGB: 1,
            diskCacheDir: directory, modelKey: "post-answer-entrypoint")
    }

    private static func assertRestoredStateMatchesCold(
        _ restored: [any KVCache], key: [Int], model: PostAnswerEntryFixture,
        parameters: GenerateParameters
    ) {
        let cold = model.newCache(parameters: parameters)
        let output = model(MLXArray(key.map(Int32.init)).reshaped(1, -1), cache: cold)
        eval(output, cold)
        XCTAssertEqual(restored.count, cold.count)
        for (actual, expected) in zip(restored, cold) {
            XCTAssertEqual(actual.state.count, expected.state.count)
            for (a, b) in zip(actual.state, expected.state) {
                XCTAssertEqual(a.shape, b.shape)
                XCTAssertEqual(a.asArray(Float.self), b.asArray(Float.self))
            }
        }
    }

    private func requireCompile() throws {
        guard HardwareInfo.isCompiledDecodeSupported else {
            throw XCTSkip("Requires supported compiled decode; skip is not proof")
        }
    }

    func testTokenIteratorFinalizationPublishesCompiledPostAnswer() throws {
        try requireCompile()
        try MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("solo-postanswer-entry-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: directory) }
            let coordinator = CacheCoordinator(config: Self.config(directory))
            let model = PostAnswerEntryFixture()
            let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))
            let parameters = GenerateParameters(
                maxTokens: 8, enableCompiledDecode: true,
                compiledMaxCacheLength: 16, temperature: 0)
            var iterator = try TokenIterator(
                input: input, model: model, parameters: parameters, cacheCoordinator: coordinator)
            let started = Date()
            var emitted: [Int] = []
            var sawStop = false
            for _ in 0..<4 {
                guard let token = iterator.next() else { break }
                if token == 6 { sawStop = true; break }
                emitted.append(token)
            }
            XCTAssertTrue(sawStop)
            XCTAssertEqual(emitted, [4, 5])
            let compiled = try XCTUnwrap(model.compiledCache(), "Must execute the real compiled path")
            XCTAssertEqual(compiled.offset, 3, "Storage has not published device counters yet")
            XCTAssertEqual(compiled.offsetArray.item(Int.self), 6)
            // This is the production entry point, intentionally not the helper.
            iterator.storeCacheAfterGeneration(generatedTokenIds: emitted, includeGeneratedBoundary: true)
            XCTAssertEqual(compiled.offset, 6)
            let reader = CacheCoordinator(config: Self.config(directory))
            let key = [1, 2, 3, 4, 5, 6] // includes the actually forwarded stop token
            let arrays = try XCTUnwrap(reader.diskCache?.fetch(
                tokens: key, mediaSalt: computeCacheSalt(for: input, parameters: parameters)))
            var restored = model.newCache(parameters: parameters)
            XCTAssertEqual(restoreFromDiskArrays(arrays, into: &restored), key.count)
            XCTAssertTrue(validateRestoredCacheBoundary(restored, matchedTokens: key.count, restoredTokens: key.count))
            Self.assertRestoredStateMatchesCold(restored, key: key, model: model, parameters: parameters)
            print("solo cache fixture tokens_per_second=\(Double(emitted.count) / Date().timeIntervalSince(started))")
        }
    }

    func testBatchSubmitFinalizationPublishesCompiledPostAnswerIncludingCacheList() async throws {
        try requireCompile()
        try await MLXMetalTestLock.withLock {
            for wrapped in [false, true] {
                let directory = FileManager.default.temporaryDirectory
                    .appendingPathComponent("batch-postanswer-entry-\(UUID().uuidString)")
                defer { try? FileManager.default.removeItem(at: directory) }
                let coordinator = CacheCoordinator(config: Self.config(directory))
                let model = PostAnswerEntryFixture(wrapped: wrapped)
                let processor = TestInputProcessor()
                var modelConfiguration = processor.configuration
                modelConfiguration.eosTokenIds = [6]
                let context = ModelContext(
                    configuration: modelConfiguration, model: model,
                    processor: processor, tokenizer: processor.tokenizer)
                let engine = BatchEngine(context: context, maxBatchSize: 1, cacheCoordinator: coordinator)
                let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))
                let parameters = GenerateParameters(
                    maxTokens: 8, compiledMaxCacheLength: 16,
                    enableCompiledBatchDecode: true, temperature: 0)
                let mediaSalt = computeCacheSalt(for: input, parameters: parameters)
                let started = Date()
                let (_, stream) = await engine.submit(input: input, parameters: parameters)
                var emitted: [Int] = []
                var completion: GenerateCompletionInfo?
                for await event in stream {
                    if case .token(let token) = event { emitted.append(token) }
                    if case .info(let info) = event { completion = info }
                }
                // Drain the real scheduler before any assertion can throw.
                await engine.shutdown()
                XCTAssertEqual(emitted, [4, 5])
                XCTAssertEqual(completion?.stopReason, .stop)
                XCTAssertNil(completion?.generationFailure)
                let compiled = try XCTUnwrap(model.compiledCache(), "Must execute actual batch compiled forward")
                XCTAssertEqual(compiled.offsetArray.item(Int.self), 5)
                XCTAssertEqual(compiled.offset, 5, "finishSlot must publish counters before admission/copy")
                let reader = CacheCoordinator(config: Self.config(directory))
                let key = [1, 2, 3, 4, 5]
                let arrays = try XCTUnwrap(reader.diskCache?.fetch(
                    tokens: key, mediaSalt: mediaSalt))
                var restored = model.newCache(parameters: parameters)
                XCTAssertEqual(restoreFromDiskArrays(arrays, into: &restored), key.count)
                XCTAssertTrue(validateRestoredCacheBoundary(restored, matchedTokens: key.count, restoredTokens: key.count))
                Self.assertRestoredStateMatchesCold(restored, key: key, model: model, parameters: parameters)
                print("batch cache fixture wrapped=\(wrapped) tokens_per_second=\(Double(emitted.count) / Date().timeIntervalSince(started))")
            }
        }
    }
}
