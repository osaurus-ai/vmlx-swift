import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class LFMCacheComputationIdentityTests: XCTestCase {
    private let identity = "lfm2-moe:sigmoid-f32-bias-positive-topk-v1"

    // This regression uses only APIs present before the fix: identical token
    // IDs/weights/topology must not retain the old arithmetic's disk namespace.
    func testActualLFMModelSeparatesOldArithmeticInBothCoordinatorEntryPoints() async throws {
        let config = try JSONDecoder().decode(
            LFM2MoEConfiguration.self,
            from: Data(
                """
                {"model_type":"lfm2_moe","vocab_size":16,"hidden_size":8,"intermediate_size":16,
                 "moe_intermediate_size":8,"num_hidden_layers":2,"num_experts":2,"num_experts_per_tok":1,
                 "norm_topk_prob":true,"num_attention_heads":2,"num_key_value_heads":1,
                 "max_position_embeddings":256,"use_expert_bias":true,"num_dense_layers":1,
                 "norm_eps":0.00001,"conv_bias":false,"conv_L_cache":3,"full_attn_idxs":[0]}
                """.utf8))
        // The basename deliberately says another family. Actual model owns identity.
        let container = Device.withDefaultDevice(.cpu) {
            makeContainer(LFM2MoEModel(config), name: "org/not-an-lfm-name")
        }
        let snapshot = await container.cacheTopologySnapshot()
        XCTAssertTrue(snapshot.topologyTags.contains("compute=\(identity)"))
        let originalKey = "unchanged-weights-and-token-ids|topology=kv-plus-mamba"
        let coordinatorConfig = CacheCoordinatorConfig(modelKey: originalKey)
        container.enableCaching(config: coordinatorConfig)
        let syncKey = try XCTUnwrap(container.cacheCoordinator?.config.modelKey)
        XCTAssertNotEqual(syncKey, originalKey)
        XCTAssertTrue(syncKey.contains("compute=\(identity)"))
        XCTAssertNotEqual(
            DiskCache.hashTokens([1, 2, 3, 4], modelKey: syncKey),
            DiskCache.hashTokens([1, 2, 3, 4], modelKey: originalKey))
        await container.enableCachingAsync(config: coordinatorConfig)
        XCTAssertEqual(container.cacheCoordinator?.config.modelKey, syncKey)
        // Host topologyTags already carry compute=. Both native entry points
        // must be idempotent, not allocate a second namespace by duplicating it.
        let hostConfig = CacheCoordinatorConfig(modelKey: syncKey)
        container.enableCaching(config: hostConfig)
        XCTAssertEqual(container.cacheCoordinator?.config.modelKey, syncKey)
        await container.enableCachingAsync(config: hostConfig)
        let hostKey = try XCTUnwrap(container.cacheCoordinator?.config.modelKey)
        XCTAssertEqual(
            hostKey.split(separator: "|").filter { $0 == "compute=\(identity)" }.count, 1)
        XCTAssertEqual(
            DiskCache.hashTokens([1, 2, 3, 4], modelKey: hostKey),
            DiskCache.hashTokens([1, 2, 3, 4], modelKey: syncKey))
        let legacyTag = "compute=lfm2-moe:legacy-v0"
        container.enableCaching(
            config: CacheCoordinatorConfig(modelKey: originalKey + "|" + legacyTag))
        let conflictingKey = try XCTUnwrap(container.cacheCoordinator?.config.modelKey)
        XCTAssertTrue(conflictingKey.split(separator: "|").contains(Substring(legacyTag)))
        XCTAssertTrue(
            conflictingKey.split(separator: "|").contains(Substring("compute=\(identity)")))
        // A semantics tag must not invent a new disk-state serialization contract.
        XCTAssertNil(snapshot.diskCacheStateIdentifiers)
        XCTAssertFalse(container.cacheCoordinator!.config.preserveStandardKVStorageDType)
        container.disableCaching()
    }

    func testLegacyTopologyAndUnchangedModelRetainNamespace() async throws {
        let container = makeContainer(PlainModel(), name: "lfm2-moe-looking-name")
        let originalKey = "weights=unchanged|kv=none"
        container.enableCaching(config: CacheCoordinatorConfig(modelKey: originalKey))
        XCTAssertEqual(container.cacheCoordinator?.config.modelKey, originalKey)
        await container.enableCachingAsync(config: CacheCoordinatorConfig(modelKey: originalKey))
        XCTAssertEqual(container.cacheCoordinator?.config.modelKey, originalKey)
        let topology = await container.cacheTopologySnapshot()
        XCTAssertNil(topology.cacheComputationIdentity)
        XCTAssertFalse(topology.topologyTags.contains { $0.hasPrefix("compute=") })
        // Older telemetry lacks the optional property; synthesis must decode it.
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(topology)) as? [String: Any])
        object.removeValue(forKey: "cacheComputationIdentity")
        let legacy = try JSONDecoder().decode(
            ModelCacheTopologySnapshot.self,
            from: JSONSerialization.data(withJSONObject: object))
        XCTAssertEqual(legacy, topology)
        container.disableCaching()
    }

    func testSemanticsRoundTripSeparatesOnlyStateIdentityNotTopologyPolicy() throws {
        let before = ModelCacheTopologySnapshot(cache: [KVCacheSimple(), MambaCache()])
        let corrected = ModelCacheTopologySnapshot(
            cache: [KVCacheSimple(), MambaCache()], cacheComputationIdentity: identity)
        XCTAssertEqual(
            before.requiresDiskBackedCoordinatorRestore,
            corrected.requiresDiskBackedCoordinatorRestore)
        XCTAssertEqual(
            before.requiresSeparateRecurrentPayloadState,
            corrected.requiresSeparateRecurrentPayloadState)
        XCTAssertEqual(before.diskCacheStateIdentifiers, corrected.diskCacheStateIdentifiers)
        XCTAssertNotEqual(before.topologyTags, corrected.topologyTags)
        let roundTrip = try JSONDecoder().decode(
            ModelCacheTopologySnapshot.self, from: JSONEncoder().encode(corrected))
        XCTAssertEqual(roundTrip, corrected)
        let transition = TurboQuantCacheTransitionSnapshot(before: corrected, after: corrected)
        let decoded = try JSONDecoder().decode(
            TurboQuantCacheTransitionSnapshot.self, from: JSONEncoder().encode(transition))
        XCTAssertEqual(decoded.before.cacheComputationIdentity, identity)
        XCTAssertEqual(decoded.after.cacheComputationIdentity, identity)
        XCTAssertEqual(decoded.convertedTurboQuantKVLayerCount, 0)
    }

    func testOldDiskRecordRemainsAvailableOnlyInItsOwnNamespace() throws {
        try Device.withDefaultDevice(.cpu) {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: directory) }
            let tokens = [1, 2, 3, 4]
            let oldKey = "unchanged-weights-and-token-ids|topology=kv-plus-mamba"
            let correctedKey = oldKey + "|compute=\(identity)"
            let old = DiskCache(cacheDir: directory, maxSizeBytes: 1 << 20, modelKey: oldKey)
            old.store(tokens: tokens, arrays: ["sentinel": MLXArray([Float(11), 12])])
            XCTAssertNotNil(old.fetch(tokens: tokens))
            let corrected = DiskCache(
                cacheDir: directory, maxSizeBytes: 1 << 20, modelKey: correctedKey)
            XCTAssertNil(corrected.fetch(tokens: tokens))
            // The fix never clears or deletes old records. Each namespace reads
            // its own state after a new corrected-state publication.
            corrected.store(tokens: tokens, arrays: ["sentinel": MLXArray([Float(21), 22])])
            let restart = DiskCache(
                cacheDir: directory, maxSizeBytes: 1 << 20, modelKey: correctedKey)
            XCTAssertEqual(
                try XCTUnwrap(restart.fetch(tokens: tokens)?["sentinel"]).asArray(Float.self),
                [21, 22])
            XCTAssertEqual(
                try XCTUnwrap(old.fetch(tokens: tokens)?["sentinel"]).asArray(Float.self), [11, 12])
        }
    }

    private func makeContainer(_ model: any LanguageModel, name: String) -> ModelContainer {
        ModelContainer(
            context: ModelContext(
                configuration: ModelConfiguration(id: name),
                model: model, processor: Processor(), tokenizer: StubTokenizer()))
    }
}

private final class PlainModel: Module, LanguageModel {
    var vocabularySize: Int { 16 }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray { inputs }
    func newCache(parameters: GenerateParameters?) -> [KVCache] { [KVCacheSimple(), MambaCache()] }
}
private struct Processor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput { LMInput(tokens: MLXArray([Int32(0)])) }
}
private struct StubTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
