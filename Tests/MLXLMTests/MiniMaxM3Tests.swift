// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// MiniMax-M3 / M3.1 text model against golden tensors from the vMLX Python runtime.
///
/// The fixtures are tiny synthetic models written by `scripts/minimax_m3_fixtures/generate_fixtures.py` (provenance,
/// spec commit and measured cross-checks in `Fixtures/MiniMaxM3/reference.json`). Every model is loaded through
/// the factory registry and the generic `loadWeights`, from checkpoint key names.
///
/// MSA: 23 tokens, block 4, top-2 blocks, so every query past token 7 sees only part of the context, and 23 is not a
/// multiple of the block. The goldens are strict float32. MLX on M5 runs float32 matmuls as TF32 unless
/// MLX_ENABLE_TF32=0 is set before MLX initializes, and TF32 moves the indexer's block scores by up to 0.043 on these
/// models (measured in Python), more than the closest top-k margins. So float32 comparisons run on the CPU (no TF32)
/// and, when the process was started with MLX_ENABLE_TF32=0, also on the GPU. The source-format model's BF16 run
/// is on the GPU in either mode.
@Suite("MiniMax-M3 text model vs the vMLX Python runtime", .serialized)
struct MiniMaxM3Tests {
    static let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("Fixtures/MiniMaxM3")

    /// Max |error| relative to max |reference|, strict float32. Measured agreement is printed by each test.
    static let float32Tolerance: Float = 1e-4
    /// The same for BF16 activations (one BF16 ulp is 2^-8 relative).
    static let bfloat16Tolerance: Float = 1e-2

    static var float32Devices: [Device] {
        ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0" ? [.cpu, .gpu] : [.cpu]
    }

    static func reference() throws -> [String: MLXArray] {
        try MLX.loadArrays(url: fixtures.appendingPathComponent("reference.safetensors"))
    }

    static func relativeError(_ actual: MLXArray, _ expected: MLXArray) -> Float {
        let expected = expected.asType(.float32)
        return (abs(actual.asType(.float32) - expected).max() / abs(expected).max()).item(Float.self)
    }

    /// Builds the model from `config.json` through the LLM registry and loads its safetensors with `loadWeights`,
    /// from a temporary copy of the bundle.
    static func load(_ bundle: String, modelType: String = "minimax_m3_vl") async throws -> MiniMaxM3Model {
        let source = fixtures.appendingPathComponent(bundle)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("minimax-m3-\(bundle)-\(UUID().uuidString)")
        try FileManager.default.copyItem(at: source, to: directory)
        defer { try? FileManager.default.removeItem(at: directory) }
        let configuration = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let created = try await LLMTypeRegistry.shared.createModel(
            configuration: configuration, modelType: modelType)
        let model = try #require(created as? MiniMaxM3Model)
        try loadWeights(modelDirectory: directory, model: model)
        return model
    }

    /// Logits for full prefill, decode through the cache after `prefix` tokens, and prefill in chunks of `chunk`.
    static func paths(
        _ model: MiniMaxM3Model, tokens: MLXArray, prefix: Int, chunk: Int
    ) -> (full: MLXArray, decode: MLXArray, chunked: MLXArray, cache: [KVCache]) {
        let count = tokens.dim(0)
        let batch = tokens[.newAxis]
        let full = model(batch, cache: nil)[0]
        var cache = model.newCache(parameters: nil)
        _ = model(batch[0..., ..<prefix], cache: cache)
        let decode = concatenated(
            (prefix ..< count).map { model(batch[0..., $0 ..< ($0 + 1)], cache: cache)[0] }, axis: 0)
        cache = model.newCache(parameters: nil)
        let chunked = concatenated(
            stride(from: 0, to: count, by: chunk).map {
                model(batch[0..., $0 ..< min($0 + chunk, count)], cache: cache)[0]
            }, axis: 0)
        eval(full, decode, chunked)
        return (full, decode, chunked, cache)
    }

    /// Each sparse layer's attended keys `[G, S, S]` for an uncached prefill, from the layer's own indexer.
    static func keySelections(_ model: MiniMaxM3Model, tokens: MLXArray) -> [MLXArray] {
        var h = model.model.embedTokens(tokens[.newAxis])
        var selections: [MLXArray] = []
        for layer in model.model.layers {
            if let indexer = layer.selfAttn.indexer,
                let bias = indexer.keyBias(layer.inputLayerNorm(h), cache: nil, offset: 0)
            {
                selections.append(bias[0] .== 0)
            }
            h = layer(h, mask: .causal, cache: nil)
        }
        return selections
    }

    // MARK: - Configuration, registration, cache layout

    @Test("config.json decodes from text_config, the layer layout follows sparse_attention_freq")
    func configurationAndLayout() async throws {
        for (bundle, sparse) in [("m31-float", [1, 1, 1, 1, 1]), ("m3-float", [0, 0, 0, 1, 1])] {
            for modelType in ["minimax_m3_vl", "minimax_m3"] {
                let data = try Data(
                    contentsOf: Self.fixtures.appendingPathComponent("\(bundle)/config.json"))
                let model = try #require(
                    try await LLMTypeRegistry.shared.createModel(configuration: data, modelType: modelType)
                        as? MiniMaxM3Model)
                let config = model.configuration
                #expect(config.hiddenSize == 32 && config.hiddenLayers == 5 && config.numLocalExperts == 4)
                #expect(config.sparseAttention.blockSize == 4 && config.sparseAttention.topKBlocks == 2)
                #expect(model.model.layers.map { $0.selfAttn.indexer != nil } == sparse.map { $0 == 1 })
                #expect(model.model.layers.map { $0.mlp is MiniMaxM3SparseMoeBlock } == [false, false, false, true, true])
                // M3.1 is sparse on every layer, so layers 0-2 keep indexer keys too.
                #expect(
                    model.newCache(parameters: nil).map { $0 is MiniMaxM3SparseCache } == sparse.map { $0 == 1 })
                #expect(
                    model.newCache(parameters: nil).map { type(of: $0) == KVCacheSimple.self }
                        == sparse.map { $0 == 0 })

                // The same fields at the top level (a text-only config) decode identically.
                var root = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
                let text = try #require(root.removeValue(forKey: "text_config") as? [String: Any])
                root.merge(text) { outer, _ in outer }
                let flat = try JSONDecoder().decode(
                    MiniMaxM3Configuration.self, from: JSONSerialization.data(withJSONObject: root))
                #expect(flat.sparseAttention.layerFrequency == config.sparseAttention.layerFrequency)
                #expect(flat.moeLayerFrequency == config.moeLayerFrequency && flat.headDim == config.headDim)
            }
        }
        // Only the text model is built; asking for a vision lane is refused, not served text-only.
        let data = try Data(contentsOf: Self.fixtures.appendingPathComponent("m31-float/config.json"))
        await #expect(throws: UnconstructibleModalities.self) {
            _ = try await LLMTypeRegistry.shared.createModel(
                configuration: data, modelType: "minimax_m3_vl", requesting: [.text, .vision])
        }
    }

    // MARK: - MSA per KV group

    @Test("One sparse attention layer matches the NumPy reference; the KV groups select different blocks")
    func attentionMatchesIndependentReference() throws {
        let reference = try Self.reference()
        let config = try JSONDecoder().decode(
            MiniMaxM3Configuration.self,
            from: Data(contentsOf: Self.fixtures.appendingPathComponent("attention/config.json")))
        let attention = MiniMaxM3Attention(config, layer: 0)
        let weights = reference.compactMap { key, value in
            key.hasPrefix("attention.weight.") ? (String(key.dropFirst("attention.weight.".count)), value) : nil
        }
        try attention.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let x = try #require(reference["attention.x"])
        let expected = try #require(reference["attention.expected"])
        let keyMask = try #require(reference["attention.key_mask"]).asType(.bool)
        let length = x.dim(1)

        // Fail closed: the groups' selections must differ, or a selection shared by all groups would pass too.
        #expect((keyMask[0] .!= keyMask[1]).any().item(Bool.self))

        for device in Self.float32Devices {
            try Device.withDefaultDevice(device) {
                let indexer = try #require(attention.indexer)
                let selection = try #require(indexer.keyBias(x, cache: nil, offset: 0))[0] .== 0
                #expect(arrayEqual(selection, keyMask).item(Bool.self), "selection on \(device)")

                let prefill = attention(x, mask: .causal, cache: nil)[0]
                let prefillError = Self.relativeError(prefill, expected)
                #expect(prefillError < Self.float32Tolerance, "prefill on \(device): \(prefillError)")

                var cache = MiniMaxM3SparseCache()
                _ = attention(x[0..., ..<(length - 1)], mask: .causal, cache: cache)
                let last = attention(x[0..., (length - 1)...], mask: .none, cache: cache)[0]
                let decodeError = Self.relativeError(last, expected[(length - 1)...])
                #expect(decodeError < Self.float32Tolerance, "decode on \(device): \(decodeError)")

                cache = MiniMaxM3SparseCache()
                let rows = stride(from: 0, to: length, by: 7).map { start in
                    let chunk = x[0..., start ..< min(start + 7, length)]
                    return attention(
                        chunk, mask: createAttentionMask(h: chunk, cache: cache), cache: cache)[0]
                }
                let chunkedError = Self.relativeError(concatenated(rows, axis: 0), expected)
                #expect(chunkedError < Self.float32Tolerance, "chunked on \(device): \(chunkedError)")
                print("[MiniMaxM3Tests] attention \(device): prefill \(prefillError) decode \(decodeError) "
                    + "chunked \(chunkedError)")
            }
        }
    }

    @Test("M3.1 and M3 float models match the Python logits for prefill, decode and chunked prefill")
    func floatModelsMatchPython() async throws {
        let reference = try Self.reference()
        for bundle in ["m31-float", "m3-float"] {
            let model = try await Self.load(bundle)
            let tokens = try #require(reference["\(bundle).tokens"])
            let expectedMasks = try #require(reference["\(bundle).key_masks"]).asType(.bool)
            #expect(
                (0 ..< expectedMasks.dim(0)).contains {
                    (expectedMasks[$0][0] .!= expectedMasks[$0][1]).any().item(Bool.self)
                }, "\(bundle): fixture groups never disagree")

            for device in Self.float32Devices {
                try Device.withDefaultDevice(device) {
                    let selections = Self.keySelections(model, tokens: tokens)
                    #expect(selections.count == expectedMasks.dim(0))
                    for (layer, selection) in selections.enumerated() {
                        #expect(
                            arrayEqual(selection, expectedMasks[layer]).item(Bool.self),
                            "\(bundle) sparse layer \(layer) selection on \(device)")
                    }

                    let result = Self.paths(model, tokens: tokens, prefix: 15, chunk: 7)
                    let errors = [
                        ("prefill", Self.relativeError(result.full, try #require(reference["\(bundle).logits"]))),
                        (
                            "decode",
                            Self.relativeError(result.decode, try #require(reference["\(bundle).logits_decode"]))
                        ),
                        (
                            "chunked",
                            Self.relativeError(
                                result.chunked, try #require(reference["\(bundle).logits_chunked"]))
                        ),
                    ]
                    for (path, error) in errors {
                        #expect(error < Self.float32Tolerance, "\(bundle) \(path) on \(device): \(error)")
                    }
                    print("[MiniMaxM3Tests] \(bundle) \(device): \(errors)")

                    // Every sparse layer kept K/V and indexer keys for all 23 positions.
                    for (layer, cache) in result.cache.enumerated() {
                        #expect(cache.offset == tokens.dim(0))
                        if let sparse = cache as? MiniMaxM3SparseCache {
                            #expect(sparse.state.count == 3 && sparse.state[2].dim(2) == tokens.dim(0), "layer \(layer)")
                        }
                    }
                }
            }
        }
    }

    @Test("A restored or trimmed sparse cache decodes like the live one")
    func sparseCacheRestoreAndTrim() async throws {
        let reference = try Self.reference()
        let model = try await Self.load("m31-float")
        let tokens = try #require(reference["m31-float.tokens"])[.newAxis]
        let expected = try #require(reference["m31-float.logits_decode"])  // rows for positions 15..<23

        try Device.withDefaultDevice(.cpu) {
            // Prefill 17, trim back to 15, decode position 15.
            let trimmed = model.newCache(parameters: nil)
            _ = model(tokens[0..., ..<17], cache: trimmed)
            for cache in trimmed {
                #expect(cache.trim(2) == 2)
            }
            let afterTrim = model(tokens[0..., 15 ..< 16], cache: trimmed)[0]
            #expect(Self.relativeError(afterTrim, expected[0 ..< 1]) < Self.float32Tolerance)

            // Persist the 15-token state of every layer and restore it into fresh caches.
            let live = model.newCache(parameters: nil)
            _ = model(tokens[0..., ..<15], cache: live)
            let restored = model.newCache(parameters: nil)
            for (source, target) in zip(live, restored) {
                let custom = try #require(target as? MiniMaxM3SparseCache)
                #expect(
                    custom.restoreDiskCacheState(
                        source.state.map { $0 * 1 }, metadata: source.metaState, offset: source.offset))
            }
            let afterRestore = model(tokens[0..., 15 ..< 16], cache: restored)[0]
            #expect(Self.relativeError(afterRestore, expected[0 ..< 1]) < Self.float32Tolerance)

            // A record without the indexer lane is refused, not half-restored.
            let probe = MiniMaxM3SparseCache()
            #expect(!probe.restoreDiskCacheState(Array(live[0].state.prefix(2)), metadata: [""], offset: 15))
            #expect(probe.offset == 0)
        }
    }

    // MARK: - Official checkpoint format

    @Test("The official M3.1 checkpoint (MXFP8 + NVFP4) loads as stored and matches Python and the spec dequant")
    func sourceCheckpointLoadsAsStored() async throws {
        let reference = try Self.reference()
        let model = try await Self.load("m31-source")
        let tokens = try #require(reference["m31-source.tokens"])

        // Modules: MXFP8 linears, NVFP4 experts with one inverse global scale per expert; vision/MTP dropped.
        let attention = model.model.layers[0].selfAttn
        let qProj = try #require(attention.wq as? QuantizedLinear)
        #expect(qProj.mode == .mxfp8 && qProj.groupSize == 32 && qProj.bits == 8)
        #expect((attention.indexer?.queryProjection as? QuantizedLinear)?.mode == .mxfp8)
        let moe = try #require(model.model.layers[1].mlp as? MiniMaxM3SparseMoeBlock)
        #expect((moe.sharedExperts.downProj as? QuantizedLinear)?.mode == .mxfp8)
        let experts = [moe.switchMLP.gateProj, moe.switchMLP.upProj, moe.switchMLP.downProj]
        for projection in experts {
            let nvfp4 = try #require(projection as? MiniMaxM3NVFP4SwitchLinear)
            #expect(nvfp4.inverseGlobalScale.dtype == .float32)
            #expect(Set(nvfp4.inverseGlobalScale.asArray(Float.self)).count == 4)
        }
        #expect(model.model.embedTokens.weight.dtype == .bfloat16)
        #expect(moe.gate.weight.dtype == .float32)
        let parameterNames = model.parameters().flattened().map(\.0)
        #expect(!parameterNames.contains { $0.contains("vision") || $0.hasPrefix("mtp") })

        // As stored: BF16 activations, on the GPU.
        let bf16 = Self.paths(model, tokens: tokens, prefix: 9, chunk: 13)
        let bf16Error = Self.relativeError(bf16.full, try #require(reference["m31-source.logits_bf16"]))
        let bf16DecodeError = Self.relativeError(
            bf16.decode, try #require(reference["m31-source.logits_bf16_decode"]))
        #expect(bf16Error < Self.bfloat16Tolerance, "bf16 prefill: \(bf16Error)")
        #expect(bf16DecodeError < Self.bfloat16Tolerance, "bf16 decode: \(bf16DecodeError)")
        print("[MiniMaxM3Tests] m31-source bf16 gpu: prefill \(bf16Error) decode \(bf16DecodeError)")

        // The same stored bytes with the BF16 tensors widened to float32: against Python and against a float model
        // built from a spec dequant of the stored bytes.
        let widened = model.parameters().flattened().compactMap { key, value in
            value.dtype == .bfloat16 ? (key, value.asType(.float32)) : nil
        }
        try model.update(parameters: ModuleParameters.unflattened(widened), verify: [.noUnusedKeys])
        let specDequant = try #require(reference["m31-source.logits_spec_dequant"])
        for device in Self.float32Devices {
            try Device.withDefaultDevice(device) {
                let logits = model(tokens[.newAxis], cache: nil)[0]
                let pythonError = Self.relativeError(logits, try #require(reference["m31-source.logits_f32"]))
                let specError = Self.relativeError(logits, specDequant)
                #expect(pythonError < Self.float32Tolerance, "f32 vs Python on \(device): \(pythonError)")
                #expect(specError < Self.float32Tolerance, "f32 vs spec dequant on \(device): \(specError)")
                print("[MiniMaxM3Tests] m31-source f32 \(device): python \(pythonError) spec \(specError)")
            }
        }

        // Fail closed: the routed experts must move the output, or an expert-layout error could pass unseen.
        let zeroed = experts.map { projection in
            let nvfp4 = projection as! MiniMaxM3NVFP4SwitchLinear
            return ("inv_global_scale", nvfp4.inverseGlobalScale * 0)
        }
        for (projection, update) in zip(experts, zeroed) {
            try projection.update(parameters: ModuleParameters.unflattened([update]), verify: .none)
        }
        try Device.withDefaultDevice(.cpu) {
            let withoutExperts = Self.relativeError(model(tokens[.newAxis], cache: nil)[0], specDequant)
            #expect(withoutExperts > 100 * Self.float32Tolerance, "experts zeroed: \(withoutExperts)")
            print("[MiniMaxM3Tests] m31-source experts zeroed: \(withoutExperts)")
        }
    }
}
