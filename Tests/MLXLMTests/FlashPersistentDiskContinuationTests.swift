import Foundation
import MLX
import MLXNN
import MLXRandom
@testable import MLXLMCommon
@testable import MLXVLM
import Testing

// Bounded generated fixture: no installed model or experimental sampled verifier.
@Suite(.serialized)
struct FlashPersistentDiskContinuationTests {
    private func withFixture(routedBits: [Int] = [], mtpEnabled: Bool = false,
                             inputProjectionBits: Int? = nil, _ body: (Qwen4Exp) throws -> Void) throws {
        // Entire geometry is bounded before construction; no installed model is read.
        let data = Data(
            """
            {
              "model_type":"qwen4_exp","eos_token_id":1,
              "text_config":{
                "model_type":"qwen4_exp_text","dtype":"float32",
                "hidden_size":64,"num_hidden_layers":2,"intermediate_size":64,
                "num_attention_heads":4,"num_key_value_heads":1,"head_dim":16,
                "linear_num_value_heads":4,"linear_num_key_heads":1,
                "linear_key_head_dim":16,"linear_value_head_dim":16,
                "linear_conv_kernel_dim":4,"vocab_size":128,
                "num_experts":8,"num_experts_per_tok":2,
                "moe_intermediate_size":16,"shared_expert_intermediate_size":16,
                "layer_types":["linear_attention","full_attention"],
                "hc_count":4,"hc_lowrank":8,"ple_layer_ids":[1],
                "ple_embed_dim":64,"ple_conv_kernel_size":4,
                "ngram_size":3,"heads_per_ngram":2,"ngram_vocab_size_base":101,
                "make_ngram_vocab_size_divisible_by":128,"seed":1234,
                "split_ngram_parts":4,"indexer_n_heads":2,"indexer_kv_heads":1,
                "indexer_head_dim":8,"indexer_budget":32,"indexer_compress_ratio":4,
                "mrope_section":[1,1,1],"mtp_num_hidden_layers":0
              }
            }
            """.utf8)
        var fixtureData = data
        if mtpEnabled {
            var root = try #require(JSONSerialization.jsonObject(with: fixtureData) as? [String: Any])
            var text = try #require(root["text_config"] as? [String: Any])
            text["mtp_num_hidden_layers"] = 1
            root["text_config"] = text
            fixtureData = try JSONSerialization.data(withJSONObject: root)
        }
        var routedSpecs: [String: Int] = [:]
        if !routedBits.isEmpty {
            try #require(routedBits.count == 3 && (routedBits == [0, 0, 0] || routedBits.allSatisfy { [2, 3, 4, 6].contains($0) }))
            var root = try #require(JSONSerialization.jsonObject(with: fixtureData) as? [String: Any])
            var text = try #require(root["text_config"] as? [String: Any])
            text["dtype"] = "bfloat16"
            text["moe_intermediate_size"] = 64
            text["shared_expert_intermediate_size"] = 64
            root["text_config"] = text
            var quantization: [String: Any] = ["bits": 8, "group_size": 64]
            for layer in 0..<2 {
                for (projection, bits) in zip(["gate_proj", "up_proj", "down_proj"], routedBits) {
                    if bits == 0 { continue } // BF16 dense control, same geometry.
                    let path = "language_model.layers.\(layer).mlp.switch_mlp.\(projection)"
                    routedSpecs[path] = bits
                    quantization[path] = ["bits": bits, "group_size": 64]
                }
            }
            if let bits = inputProjectionBits {
                try #require([2, 3, 4, 6, 8].contains(bits))
                for name in ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"] {
                    let path = "language_model.layers.0.linear_attn.\(name)"
                    routedSpecs[path] = bits
                    quantization[path] = ["bits": bits, "group_size": 64]
                }
            }
            if !routedSpecs.isEmpty { root["quantization"] = quantization }
            fixtureData = try JSONSerialization.data(withJSONObject: root)
        }
        let config = try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: fixtureData)
        let text = config.base.textConfiguration
        try #require(text.hiddenSize == 64 && text.hiddenLayers == 2)
        try #require(text.vocabularySize == 128 && text.numExperts == 8)
        MLXRandom.seed(0)
        let model = Qwen4Exp(config)
        let count = model.parameters().flattened().reduce(0) { $0 + $1.1.size }
        try #require(count < 2_000_000, "refuse fixture drift before MLX evaluation")
        if !routedBits.isEmpty {
            // Match the retained live parameter-domain contract: ordinary
            // parameters BF16, router weights FP32. This is still a tiny
            // generated fixture, not a shipped-weight numerical reference.
            model.update(parameters: ModuleParameters.unflattened(
                model.parameters().flattened().map { path, value in
                    (path, path.hasSuffix(".mlp.gate.weight") ? value : value.asType(.bfloat16))
                }))
        }
        if !routedSpecs.isEmpty {
            quantize(model: model, filter: { path, _ in
                guard let bits = routedSpecs[path] else { return nil }
                return (groupSize: 64, bits: bits, mode: QuantizationMode.affine)
            })
            let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
            for (path, bits) in routedSpecs {
                let projection = try #require(leaves[path] as? any Quantized)
                try #require(projection.bits == bits && projection.groupSize == 64)
            }
            model.update(parameters: ModuleParameters.unflattened(
                model.parameters().flattened().compactMap { path, value in
                    guard path.hasSuffix(".scales") || path.hasSuffix(".biases") else { return nil }
                    return (path, value.asType(.float16))
                }))
        }

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("flash-prefill-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var arrays: [String: MLXArray] = [:]
        var map: [String: String] = [:]
        var specs: [String: [String: Int]] = [:]
        for shard in 0 ..< 4 {
            let base = "language_model.layers.0.ple.ngram_embedding.shards.\(shard)"
            arrays[base + ".weight"] = MLXArray(
                (0 ..< (128 * 2)).map { UInt32(truncatingIfNeeded: ($0 + shard) * 0x12345) }
            ).reshaped(128, 2)
            arrays[base + ".scales"] = MLXArray.full([128, 4], values: MLXArray(Float(0.01)))
                .asType(.float16)
            arrays[base + ".biases"] = MLXArray.full([128, 4], values: MLXArray(Float(-0.05)))
                .asType(.float16)
            specs[base + ".weight"] = ["bits": 4, "group_size": 4]
            for suffix in [".weight", ".scales", ".biases"] {
                map[base + suffix] = "model.safetensors"
            }
        }
        try MLX.save(arrays: arrays, url: directory.appendingPathComponent("model.safetensors"))
        try JSONSerialization.data(withJSONObject: ["weight_map": map])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        try JSONSerialization.data(withJSONObject: ["jang_config": ["bit_map": specs]])
            .write(to: directory.appendingPathComponent("config.json"))
        try model.configure(modelDirectory: directory)
        try body(model)
    }

    @Test("native Flash token cap never publishes pending tokens under an emitted-prefix key",
          .enabled(if: ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0",
                   "Requires strict quantized-fixture oracle"), arguments: [2, 6])
    func nativeFinalBoundaryDiskPublication(inputBits: Int) throws {
        try MLXMetalTestLock.withLock {
            try withFixture(routedBits: [2, 3, 4], mtpEnabled: true,
                            inputProjectionBits: inputBits) { model in
                for cap in [2, 3, 4, 9, 32] {
                    let root = FileManager.default.temporaryDirectory
                        .appendingPathComponent("flash-final-boundary-\(UUID().uuidString)")
                    defer { try? FileManager.default.removeItem(at: root) }
                    let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
                        usePagedCache: false, enableDiskCache: true, diskCacheMaxGB: 1,
                        diskCacheDir: root, modelKey: "flash-final-\(inputBits)-\(cap)"))
                    // Let the production iterator derive topology from the
                    // actual PLE/GDN/QSA caches, exactly as an ordinary load does.
                    let promptIDs = [2, 3, 4]
                    let prompt = MLXArray(promptIDs.map(Int32.init)).reshaped(1, 3)
                    var parameters = GenerateParameters(maxTokens: cap, temperature: 1, topP: 0.95, topK: 20)
                    parameters.randomSeed = 829
                    parameters.draftStrategy = .nativeMTP(depth: 3)
                    parameters.nativeMTPDepthPolicy = .fixed
                    var iterator = try NativeMTPTokenIterator(
                        input: LMInput(tokens: prompt), model: model, parameters: parameters,
                        depth: 3, cacheCoordinator: coordinator)
                    let started = Date()
                    var emitted: [Int] = []
                    while let token = iterator.next() { emitted.append(token) }
                    #expect(emitted.count == cap)
                    #expect(iterator.mtpForwardCount > 0, "native MTP must actually execute")
                    iterator.storeCacheAfterGeneration(generatedTokenIds: emitted, includeGeneratedBoundary: true)
                    let disk = try #require(coordinator.diskCache)
                    // Exercise the production restore caller too; raw serializer
                    // comparisons alone could omit a legitimate companion restore.
                    let continuation = LMInput(tokens: MLXArray([Int32(2), 3, 4, 73]).reshaped(1, 4))
                    var cachedIterator = try NativeMTPTokenIterator(
                        input: continuation, model: model, parameters: parameters, depth: 3,
                        cacheCoordinator: coordinator)
                    var freshIterator = try NativeMTPTokenIterator(
                        input: continuation, model: model, parameters: parameters, depth: 3)
                    for (layer, pair) in zip(cachedIterator.cache, freshIterator.cache).enumerated() {
                        #expect(pair.0.offset == pair.1.offset)
                        #expect(pair.0.state.count == pair.1.state.count)
                        for (slot, values) in zip(pair.0.state, pair.1.state).enumerated() {
                            expectEqual(values.0, values.1,
                                        "native restored inputBits\(inputBits) cap\(cap) layer\(layer) slot\(slot)")
                        }
                    }
                    cachedIterator.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
                    freshIterator.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
                    // A missing post-answer snapshot is allowed for non-trimmable
                    // recurrent state. A snapshot published under this key must
                    // represent exactly the emitted prefix, never queued tokens.
                    var restoredCount = 0
                    for ids in [promptIDs, promptIDs + emitted] {
                        guard let arrays = disk.fetch(
                            tokens: ids,
                            mediaSalt: computeCacheSalt(for: LMInput(tokens: prompt), parameters: parameters))
                        else { continue }
                        var restored = model.newCache(parameters: parameters)
                        #expect(restoreFromDiskArrays(arrays, into: &restored) == ids.count)
                        let reference = model.newCache(parameters: parameters)
                        MLX.eval(model.nativeBackboneForward(prompt, cache: reference).logits)
                        for id in ids.dropFirst(promptIDs.count) {
                            MLX.eval(model.nativeBackboneForward(
                                MLXArray([Int32(id)]).reshaped(1, 1), cache: reference).logits)
                        }
                        for (layer, pair) in zip(restored, reference).enumerated() {
                            #expect(pair.0.offset == pair.1.offset)
                            #expect(pair.0.state.count == pair.1.state.count)
                            for (slot, values) in zip(pair.0.state, pair.1.state).enumerated() {
                                expectEqual(values.0, values.1,
                                            "disk inputBits\(inputBits) cap\(cap) boundary\(ids.count) layer\(layer) slot\(slot)")
                            }
                        }
                        let followup = MLXArray([Int32(73)]).reshaped(1, 1)
                        expectEqual(model(followup, cache: restored), model(followup, cache: reference),
                                    "disk continuation inputBits\(inputBits) cap\(cap) boundary\(ids.count)")
                        restoredCount += 1
                    }
                    #expect(restoredCount > 0, "positive disk publication control must execute")
                    print("FLASH-FINAL-DISK inputBits=\(inputBits) cap=\(cap) restored=\(restoredCount) emitted=\(emitted.count) fixtureTokS=\(Double(emitted.count) / max(Date().timeIntervalSince(started), 1e-9)) realModelSpeedProof=false")
                }
            }
        }
    }

    private func expectEqual(_ actual: MLXArray, _ expected: MLXArray, _ label: String) {
        #expect(actual.shape == expected.shape, "\(label) shape")
        guard actual.shape == expected.shape else { return }
        let a = actual.asType(.float32)
        let b = expected.asType(.float32)
        let error = MLX.max(abs(a - b)).item(Float.self)
        if !allClose(a, b, rtol: 1e-5, atol: 1e-5).item(Bool.self) {
            let actualValues = a.reshaped(-1).asArray(Float.self)
            let expectedValues = b.reshaped(-1).asArray(Float.self)
            if let index = actualValues.indices.first(where: {
                abs(actualValues[$0] - expectedValues[$0]) > 1e-5 + 1e-5 * abs(expectedValues[$0])
            }) {
                print("FLASH-PARITY-DIFF \(label) shape=\(a.shape) dtype=\(actual.dtype) index=\(index) actual=\(actualValues[index]) expected=\(expectedValues[index]) maxAbs=\(error)")
            }
        }
        #expect(
            allClose(a, b, rtol: 1e-5, atol: 1e-5).item(Bool.self),
            "\(label) maxAbs=\(error)")
    }

}
