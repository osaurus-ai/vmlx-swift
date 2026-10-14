import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("LFM stable snapshots before the strip boundary", .serialized)
struct LFMStableHeadSnapshotTests {
    private let tokens = (0..<60).map { ($0 * 7 + 3) % 64 }
    private let window = 16

    private func model() throws -> LFM2MoEModel {
        let json = """
        {"model_type":"lfm2_moe","vocab_size":64,"hidden_size":128,
         "intermediate_size":256,"moe_intermediate_size":64,"num_hidden_layers":8,
         "num_experts":4,"num_experts_per_tok":2,"norm_topk_prob":true,
         "num_attention_heads":2,"num_key_value_heads":1,"max_position_embeddings":512,
         "use_expert_bias":true,"num_dense_layers":1,"norm_eps":0.00001,
         "conv_bias":false,"conv_L_cache":3,"routed_scaling_factor":1,
         "layer_types":["conv","conv","conv","full_attention","conv","conv","conv","full_attention"]}
        """
        let model = LFM2MoEModel(try JSONDecoder().decode(
            LFM2MoEConfiguration.self, from: Data(json.utf8)))
        let replacements = Dictionary(uniqueKeysWithValues:
            model.parameters().flattened().enumerated().map { index, item in
                let (name, array) = item
                let value: MLXArray
                if name.hasSuffix("norm.weight") {
                    value = MLXArray.ones(array.shape).asType(.bfloat16)
                } else {
                    let values = sin(MLXArray(0..<array.size).asType(.float32) * 0.17
                        + Float(index) * 0.11) * 0.08
                    value = values.reshaped(array.shape).asType(
                        name.hasSuffix("expert_bias") ? .float32 : .bfloat16)
                }
                return (name, value)
            })
        try model.update(parameters: ModuleParameters.unflattened(replacements), verify: [.all])
        MLX.eval(model.parameters())
        return model
    }

    private func input(_ tokens: [Int], prefixes: [Int], stable: [Int]) -> LMInput {
        LMInput(tokens: MLXArray(tokens.map(Int32.init)).expandedDimensions(axis: 0),
            tokenIds: tokens, cachePrefixTokenCounts: prefixes, cacheStablePrefixTokenCounts: stable)
    }

    @discardableResult
    private func feed(_ tokens: [Int], model: LFM2MoEModel, cache: [KVCache]) throws -> MLXArray {
        let result = try model.prepare(input(tokens, prefixes: [], stable: []),
            cache: cache, windowSize: window)
        let logits: MLXArray
        switch result {
        case .tokens(let tail): logits = model(tail.tokens.reshaped(1, -1), cache: cache)
        case .logits(let output): logits = output.logits
        }
        MLX.eval(logits)
        MLX.eval(cache)
        return logits
    }

    private func diff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    private func compare(_ actual: [KVCache], _ reference: [KVCache]) -> Float {
        var maximum: Float = 0
        #expect(actual.count == reference.count)
        for (a, b) in zip(actual, reference) {
            #expect(a.offset == b.offset)
            #expect(a.state.count == b.state.count)
            for (x, y) in zip(a.state, b.state) {
                #expect(x.shape == y.shape)
                #expect(x.dtype == y.dtype)
                maximum = max(maximum, diff(x, y))
            }
        }
        return maximum
    }

    @Test("disk-restored BF16 LFM state matches explicit warm segmentation")
    func warmExplicitOrderAndOldOrder() throws {
        try MLXMetalTestLock.withLock {
            let model = try model()
            let disk = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: disk) }
            let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
                usePagedCache: false, enableDiskCache: true, diskCacheDir: disk,
                modelKey: "lfm-tiny-warm-head"))
            let topology = ModelCacheTopologySnapshot(cache: model.newCache(parameters: nil))
            coordinator.setHybrid(true,
                requiresRecurrentSSMCompanion: topology.requiresRecurrentSSMCompanionState,
                requiresSeparateRecurrentPayload: topology.requiresSeparateRecurrentPayloadState)
            let parameters = GenerateParameters(maxTokens: 1, temperature: 0, prefillStepSize: window)
            var cold = try TokenIterator(
                input: input(tokens, prefixes: [17, 37, 52], stable: [17, 37]), model: model,
                parameters: parameters, cacheCoordinator: coordinator)
            cold.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
            #expect(coordinator.diskCache!.waitForPendingWrites(timeout: 5))
            let warmTokens = Array(tokens[..<17]) + (0..<48).map { ($0 * 11 + 9) % 64 }
            let warm = try TokenIterator(
                input: input(warmTokens, prefixes: [17, 39, 56], stable: [17, 39]), model: model,
                parameters: parameters, cacheCoordinator: coordinator)
            #expect(coordinator.snapshotStats().diskStats!.hits > 0)
            #expect(warm.stableBoundarySnapshots[16] == nil)
            #expect(warm.stableBoundarySnapshots[38] != nil)
            let reference = model.newCache(parameters: nil)
            var at = 0
            var newLogits: MLXArray!
            for boundary in [16, 38, 56, 65] {
                newLogits = try feed(Array(warmTokens[at..<boundary]), model: model, cache: reference)
                at = boundary
            }
            let stateDiff = compare(warm.cache, reference)
            #expect(stateDiff == 0, "same stored16-token seed and segment order must have exact state")
            #expect(warm.y.tokens.item(Int.self) == argMax(newLogits[0, -1]).item(Int.self))
            let oldOrder = model.newCache(parameters: nil)
            _ = try feed(Array(warmTokens[..<16]), model: model, cache: oldOrder)
            _ = try feed(Array(warmTokens[16..<56]), model: model, cache: oldOrder)
            let oldLogits = try feed(Array(warmTokens[56...]), model: model, cache: oldOrder)
            let oldStateDiff = compare(warm.cache, oldOrder)
            let oldLogitDiff = diff(newLogits[0, -1], oldLogits[0, -1])
            let bytes = warm.stableBoundarySnapshots[38]!.flatMap(\.state).reduce(0) { $0 + $1.nbytes }
            print("LFM_HEAD_WARM restored=16 new_state_diff=\(stateDiff) old_state_diff=\(oldStateDiff) old_logit_diff=\(oldLogitDiff) old_token=\(argMax(oldLogits[0, -1]).item(Int.self)) new_token=\(warm.y.tokens.item(Int.self)) new_head_seed_logical_bytes=\(bytes)")
            #expect(oldStateDiff < 0.05)
            #expect(oldLogitDiff < 0.05)
            #expect(warm.cache.flatMap(\.state).allSatisfy { $0.dtype == .bfloat16 })
        }
    }

    @Test("cold BF16 cache matches explicit capture order and measures old-order drift")
    func coldExplicitOrderAndOldOrder() throws {
        try MLXMetalTestLock.withLock {
            let model = try model()
            // A private SSD tier enables the actual structural-capture path.
            let disk = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: disk) }
            let liveCoordinator = CacheCoordinator(config: CacheCoordinatorConfig(
                usePagedCache: false, enableDiskCache: true, diskCacheDir: disk,
                modelKey: "lfm-tiny-head"))
            let topology = ModelCacheTopologySnapshot(cache: model.newCache(parameters: nil))
            liveCoordinator.setHybrid(true,
                requiresRecurrentSSMCompanion: topology.requiresRecurrentSSMCompanionState,
                requiresSeparateRecurrentPayload: topology.requiresSeparateRecurrentPayloadState)
            let iterator = try TokenIterator(
                input: input(tokens, prefixes: [17, 37, 52], stable: [17, 37]), model: model,
                parameters: GenerateParameters(maxTokens: 1, temperature: 0, prefillStepSize: window),
                cacheCoordinator: liveCoordinator)
            let newOrder = model.newCache(parameters: nil)
            var at = 0
            var snapshots: [Int: [KVCache]] = [:]
            var newLogits: MLXArray!
            for boundary in [16, 36, 52, 60] {
                newLogits = try feed(Array(tokens[at..<boundary]), model: model, cache: newOrder)
                if boundary < 60 { snapshots[boundary] = makePromptBoundaryCacheSnapshot(from: newOrder) }
                at = boundary
            }
            let newStateDiff = compare(iterator.cache, newOrder)
            #expect(newStateDiff == 0, "explicit identical segmentation must have exact cache state")
            #expect(iterator.y.tokens.item(Int.self) == argMax(newLogits[0, -1]).item(Int.self))
            #expect(iterator.stableBoundarySnapshots[16] != nil)
            #expect(iterator.stableBoundarySnapshots[36] != nil)
            #expect(iterator.stableBoundarySnapshots[17] == nil)
            #expect(iterator.stableBoundarySnapshots[37] == nil)
            for boundary in [16, 36] {
                #expect(compare(iterator.stableBoundarySnapshots[boundary]!, snapshots[boundary]!) == 0)
            }
            let oldOrder = model.newCache(parameters: nil)
            _ = try feed(Array(tokens[..<52]), model: model, cache: oldOrder)
            let oldLogits = try feed(Array(tokens[52...]), model: model, cache: oldOrder)
            let oldStateDiff = compare(iterator.cache, oldOrder)
            let oldLogitDiff = diff(newLogits[0, -1], oldLogits[0, -1])
            let bytes = iterator.stableBoundarySnapshots.filter { [16, 36].contains($0.key) }.reduce(0) { total, item in
                total + item.value.flatMap(\.state).reduce(0) { $0 + $1.nbytes }
            }
            print("LFM_HEAD_COLD new_state_diff=\(newStateDiff) old_state_diff=\(oldStateDiff) old_logit_diff=\(oldLogitDiff) old_token=\(argMax(oldLogits[0, -1]).item(Int.self)) new_token=\(iterator.y.tokens.item(Int.self)) new_head_seed_logical_bytes=\(bytes)")
            #expect(oldStateDiff < 0.05)
            #expect(oldLogitDiff < 0.05)
        }
    }
}
