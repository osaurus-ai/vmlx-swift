// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import MLXVLM
import Testing
import os

@testable import MLXLLM
@testable import MLXLMCommon

/// A real four-layer Qwen graph, not hand-authored cache arrays. The fixture
/// has three GDN layers and one attention layer. It exercises storage/checkpoint
/// mechanics only: the token IDs stand in for an already-rendered canonical
/// prompt, and media tensors below exercise hashing, not a vision encoder.
@Suite("Bonsai2 Qwen canonical disk checkpoint", .serialized)
struct Qwen35HadamardCacheTests {
    private static let modelJSON = """
        {
          "model_type": "qwen3_5", "text_config": {
            "model_type": "qwen3_5_text", "hidden_size": 512, "num_hidden_layers": 4,
            "intermediate_size": 512, "num_attention_heads": 8, "num_key_value_heads": 2,
            "linear_num_value_heads": 4, "linear_num_key_heads": 2,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
            "head_dim": 64, "vocab_size": 128, "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false
          }
        }
        """

    private static let canonical = Array(11 ..< 28)
    private static let generationRail = [101, 102]
    private static let continuation = [31, 32, 33]

    private static func input(
        _ tokens: [Int], thinking: Bool = false, effort: String? = nil,
        pixels: MLXArray? = nil, policy: LMInput.CacheRestorePolicy = .standard
    ) -> LMInput {
        var context: [String: any Sendable] = ["enable_thinking": thinking]
        if let effort { context["reasoning_effort"] = effort }
        return LMInput(
            text: .init(
                tokens: MLXArray(tokens.map(Int32.init), [1, tokens.count]), tokenIds: tokens),
            image: pixels.map { .init(pixels: $0) },
            cacheScopeSalt: cacheScopeSalt(from: context),
            cachePrefixTokenCounts: [canonical.count],
            cacheRestorePolicy: policy)
    }

    /// Both on-disk representations are independently encoded from the same
    /// scalar trits. No production expansion result constructs the reference.
    private static func ternaryWeight(rows: Int, width: Int, seed: Int, packed: Bool) -> MLXArray {
        let trits = (0 ..< rows * width).map { UInt8(($0 * 17 + $0 / 13 + seed) % 3) }
        if !packed {
            let words = stride(from: 0, to: trits.count, by: 16).map { base in
                (0 ..< 16).reduce(UInt32(0)) { word, lane in
                    word | (UInt32(trits[base + lane]) << (lane * 2))
                }
            }
            return MLXArray(words, [rows, width / 16])
        }
        var bytes: [UInt8] = []
        bytes.reserveCapacity(rows * width / 128 * 26)
        for group in stride(from: 0, to: trits.count, by: 128) {
            for offset in stride(from: 0, to: 128, by: 5) {
                var byte = 0
                var power = 1
                for lane in 0 ..< min(5, 128 - offset) {
                    byte += Int(trits[group + offset + lane]) * power
                    power *= 3
                }
                bytes.append(UInt8(byte))
            }
        }
        return MLXArray(bytes, [rows, width / 128 * 26])
    }

    static func writeFixture(
        at directory: URL, packed: Bool, vision: Bool = false
    ) throws -> any LanguageModel {
        var config = try #require(
            JSONSerialization.jsonObject(with: Data(modelJSON.utf8)) as? [String: Any])
        if vision {
            config["vision_config"] =
                [
                    "model_type": "qwen3_vl", "depth": 1, "hidden_size": 16,
                    "intermediate_size": 32, "out_hidden_size": 512, "num_heads": 4,
                    "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 1,
                    "num_position_embeddings": 16,
                ] as [String: Any]
            config["vocab_size"] = 128
            config["image_token_id"] = 98
            config["video_token_id"] = 97
            config["vision_start_token_id"] = 96
            config["vision_end_token_id"] = 95
            var text = config["text_config"] as! [String: Any]
            text["rope_parameters"] =
                [
                    "rope_type": "default", "rope_theta": 10000.0,
                    "partial_rotary_factor": 0.25, "mrope_section": [2, 3, 3],
                ] as [String: Any]
            config["text_config"] = text
        }
        let data = try JSONSerialization.data(withJSONObject: config)
        let model: any LanguageModel
        if vision {
            model = MLXVLM.Qwen35(
                try JSONDecoder().decode(MLXVLM.Qwen35Configuration.self, from: data))
        } else {
            model = Qwen35Model(
                try JSONDecoder().decode(MLXLLM.Qwen35Configuration.self, from: data))
        }
        let modules = model.namedModules()
        let forward = modules.compactMap { path, module -> String? in
            guard path.hasPrefix("language_model."), module is Linear,
                !path.hasSuffix(".in_proj_a"), !path.hasSuffix(".in_proj_b")
            else { return nil }
            return path
        }.sorted()
        let inverse = ["language_model.model.embed_tokens"]
        #expect(forward.count == 26)

        var fixture = JangHadamardFixture(packed: packed)
        var hadamard = fixture.config["hadamard"] as! [String: Any]
        hadamard["forward_modules"] = forward
        hadamard["inverse_modules"] = inverse
        let baseConfig = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        fixture.config.merge(baseConfig) { _, value in value }
        fixture.config["hadamard"] = hadamard
        fixture.jang["hadamard"] = hadamard
        var quantization = fixture.jang["quantization"] as! [String: Any]
        let templateManifest = quantization["tensor_quantization_manifest"] as! [String: Any]
        let templateEntry = templateManifest[JangHadamardFixture.forward] as! [String: Any]
        var manifest: [String: Any] = [:]
        for path in forward + inverse {
            var entry = templateEntry
            entry["hadamard"] = [
                "block_size": 512, "direction": inverse.contains(path) ? "inverse" : "forward",
                "signs_tensor": "\(path).signs",
            ]
            manifest[path] = entry
        }
        quantization["tensor_quantization_manifest"] = manifest
        quantization["tensor_quantization_manifest_count"] = manifest.count
        fixture.jang["quantization"] = quantization
        fixture.sidecar["prism.hadamard.weight_names"] = forward.map { "\($0).weight" }
        fixture.sidecar["prism.hadamard.inverse_weight_names"] = inverse.map { "\($0).weight" }
        try fixture.write(to: directory)

        // Replace every randomly initialized parameter with a deterministic
        // checkpoint value. Equality never depends on replaying a random seed.
        var tensors: [String: MLXArray] = [:]
        for (path, value) in model.parameters().flattened() {
            let values: [Float]
            if path.hasSuffix(".linear_attn.norm.weight")
                || (path.hasPrefix("vision_tower.") && path.contains("norm.weight"))
            {
                values = Array(repeating: 1, count: value.size)
            } else if path.contains("norm.weight") || path.hasSuffix(".A_log")
                || path.hasSuffix(".dt_bias")
            {
                // Qwen's RMSNorm checkpoint is gamma-1, unlike GDN's norm.
                values = Array(repeating: 0, count: value.size)
            } else {
                values = (0 ..< value.size).map { Float($0 % 11 - 5) * 0.0031 }
            }
            tensors[path] = MLXArray(values, value.shape)
        }
        let byPath = Dictionary(uniqueKeysWithValues: modules)
        for path in forward + inverse {
            let shape: (Int, Int)
            if let linear = byPath[path] as? Linear {
                shape = linear.shape
            } else {
                shape = try #require(byPath[path] as? Embedding).shape
            }
            try #require(shape.1 == 512)
            let seed = path.utf8.reduce(0) { $0 + Int($1) }
            let scaleValues = (0 ..< shape.0 * (shape.1 / 128)).map {
                Float($0 % 13 + 1) * 0.0007
            }
            let scales = MLXArray(scaleValues, [shape.0, shape.1 / 128]).asType(.float16)
            tensors["\(path).weight"] = ternaryWeight(
                rows: shape.0, width: shape.1, seed: seed, packed: packed)
            tensors["\(path).scales"] = scales
            tensors["\(path).signs"] = MLXArray(JangHadamardFixture.signs)
            if !packed { tensors["\(path).biases"] = -scales }
        }
        try MLX.save(arrays: tensors, url: directory.appendingPathComponent("model.safetensors"))
        try loadWeights(
            modelDirectory: directory, model: model,
            quantization: .init(groupSize: 128, bits: 2),
            jangConfig: try JangLoader.loadConfig(at: directory))
        return model
    }

    private static func coordinator(at directory: URL, packed: Bool) -> CacheCoordinator {
        let coordinator = CacheCoordinator(
            config: .init(
                usePagedCache: false, enableDiskCache: true, diskCacheMaxGB: 0.1,
                diskCacheDir: directory, modelKey: "bonsai2-four-layer|packed=\(packed)"))
        coordinator.setHybrid(
            true, requiresRecurrentSSMCompanion: true, requiresSeparateRecurrentPayload: false)
        return coordinator
    }

    private static func assertCacheEqual(_ actual: [KVCache], _ expected: [KVCache]) throws {
        try #require(actual.count == expected.count)
        for (lhs, rhs) in zip(actual, expected) {
            #expect(String(reflecting: type(of: lhs)) == String(reflecting: type(of: rhs)))
            #expect(lhs.offset == rhs.offset)
            #expect(lhs.metaState == rhs.metaState)
            try #require(lhs.state.count == rhs.state.count)
            for (left, right) in zip(lhs.state, rhs.state) {
                try #require(left.shape == right.shape)
                #expect(left.dtype == right.dtype)
                #expect(MLX.all(left .== right).item(Bool.self))
            }
        }
    }

    private static func assertMiss(_ result: CacheFetchResult) {
        if case .hit = result {
            Issue.record("incompatible token/salt identity returned a cache hit")
        }
    }

    @Test(
        "both weight storages preserve canonical GDN/KV state through disk reopen",
        arguments: [false, true])
    func checkpointReopenAndContinue(rotating: Bool) throws {
        try MLXMetalTestLock.withLock {
            var storageLogits: [[Float]] = []
            for packed in [false, true] {
                let root = FileManager.default.temporaryDirectory
                    .appendingPathComponent("bonsai2-qwen-checkpoint-\(UUID().uuidString)")
                try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
                defer { try? FileManager.default.removeItem(at: root) }
                let model = try Self.writeFixture(at: root, packed: packed)
                let parameters = GenerateParameters(
                    maxTokens: 1, maxKVSize: rotating ? 8 : nil, temperature: 0, prefillStepSize: 3)
                let coldCache = model.newCache(parameters: parameters)
                #expect(coldCache.count == 4)
                #expect(coldCache.prefix(3).allSatisfy { $0 is MambaCache })
                #expect(
                    rotating ? coldCache.last is RotatingKVCache : coldCache.last is KVCacheSimple)
                let boundaryInput = Self.input(Self.canonical)
                switch try model.prepare(
                    boundaryInput, cache: coldCache, windowSize: parameters.prefillStepSize)
                {
                case .tokens(let remaining):
                    _ = model(remaining.tokens.reshaped(1, -1), cache: coldCache)
                case .logits:
                    break
                }
                MLX.eval(coldCache)
                #expect(coldCache.last?.state.allSatisfy { $0.dtype == .float16 } == true)
                #expect(coldCache.allSatisfy { $0.offset == Self.canonical.count })
                for cache in coldCache.prefix(3) {
                    try #require(cache.state.count == 2)
                    #expect(cache.state[1].dtype == .float32)
                    for state in cache.state {
                        #expect(
                            MLX.any(state .!= 0).item(Bool.self),
                            "fixture GDN state must not be trivial")
                    }
                }
                let prompt = Self.canonical + Self.generationRail
                let input = Self.input(prompt)
                let salt = try #require(computeCacheSalt(for: input, parameters: parameters))
                let pixelsA = MLXArray((0 ..< 12).map { Float($0) / 12 }, [1, 3, 2, 2])
                let pixelsB = pixelsA + Float(0.125)
                let mediaSaltA = try #require(
                    computeCacheSalt(
                        for: Self.input(prompt, pixels: pixelsA), parameters: parameters))
                let mediaSaltB = try #require(
                    computeCacheSalt(
                        for: Self.input(prompt, pixels: pixelsB), parameters: parameters))
                #expect(salt != mediaSaltA && mediaSaltA != mediaSaltB)
                let diskRoot = root.appendingPathComponent("cache")
                do {
                    let writer = Self.coordinator(at: diskRoot, packed: packed)
                    var iterator = try TokenIterator(
                        input: input, model: model, parameters: parameters, cacheCoordinator: writer
                    )
                    // Match a parsed-tool turn: do not invent a generated
                    // boundary to make the cache test produce a larger hit.
                    iterator.storeCacheAfterGeneration(
                        generatedTokenIds: [41, 42], includeGeneratedBoundary: false)
                    #expect(writer.hasDurableDiskEntry(tokens: Self.canonical, mediaSalt: salt))
                    #expect(!writer.hasDurableDiskEntry(tokens: prompt, mediaSalt: salt))
                    #expect(
                        !writer.hasDurableDiskEntry(tokens: prompt + [41, 42], mediaSalt: salt))
                    // Separate salt-only fixture: the real text graph's state
                    // is reused solely to exercise media key isolation. This
                    // is deliberately not a VLM forward or image-cache claim.
                    writer.storeAfterGeneration(
                        promptTokens: Self.canonical, perLayerData: [], ssmStates: nil,
                        cache: coldCache, mediaSalt: mediaSaltA)
                }

                // New coordinator: no writer's in-memory recurrent state or
                // validated-file records can satisfy this restore.
                let reader = Self.coordinator(at: diskRoot, packed: packed)
                let nextPrompt = Self.canonical + Self.continuation
                let fetched = reader.fetch(
                    tokens: nextPrompt, mediaSalt: salt, skipExactDiskBoundary: true)
                guard
                    case .hit(
                        let matched, let remaining, let detail, let blocks, let ssm, let disk) =
                        fetched
                else {
                    Issue.record("reopened canonical Qwen checkpoint missed")
                    return
                }
                #expect(matched == Self.canonical.count)
                #expect(remaining == Self.continuation)
                #expect(detail == .disk && blocks.isEmpty)
                #expect(ssm == nil, "disk-only Mamba state belongs in the typed payload")
                let arrays = try #require(disk)
                for index in 0 ..< 3 {
                    #expect(arrays["mamba_\(index)_state0"] != nil)
                    #expect(arrays["mamba_\(index)_state1"] != nil)
                    #expect(arrays["__mamba_\(index)_offset__"]?.item(Int32.self) == Int32(matched))
                }
                #expect(arrays["__ssm_count__"] == nil)
                var restored = model.newCache(parameters: parameters)
                #expect(
                    restoreFromDiskArrays(arrays, into: &restored, requirePromptBoundary: true)
                        == matched)
                try Self.assertCacheEqual(restored, coldCache)

                if case .miss = reader.fetch(
                    tokens: nextPrompt, mediaSalt: mediaSaltA, skipExactDiskBoundary: true)
                {
                    Issue.record("same-media namespace did not survive reopen")
                }
                Self.assertMiss(reader.fetch(tokens: nextPrompt, mediaSalt: mediaSaltB))
                var changedTokens = nextPrompt
                changedTokens[0] += 1
                Self.assertMiss(reader.fetch(tokens: changedTokens, mediaSalt: salt))
                for effort in ["low", "medium", "xhigh"] {
                    let changedSalt = try #require(
                        computeCacheSalt(
                            for: Self.input(nextPrompt, thinking: true, effort: effort),
                            parameters: parameters))
                    #expect(changedSalt != salt)
                    Self.assertMiss(reader.fetch(tokens: nextPrompt, mediaSalt: changedSalt))
                }

                // Also use the public iterator restore/prefill entry point.
                // A low-level disk fetch alone is not proof that generation
                // accepts the prefix rather than discarding it for re-prefill.
                let warmProgress = OSAllocatedUnfairLock(initialState: [PrefillProgress]())
                let warmCache = model.newCache(parameters: parameters)
                let warm = try TokenIterator(
                    input: Self.input(nextPrompt), model: model,
                    cache: warmCache, parameters: parameters, cacheCoordinator: reader,
                    prefillProgressHandler: { event in warmProgress.withLock { $0.append(event) } })
                withExtendedLifetime(warm) { MLX.eval(warmCache) }
                let accepted = warmProgress.withLock { $0 }.first { $0.stage == .cacheRestore }
                #expect(accepted?.completedUnitCount == Self.canonical.count)
                #expect(accepted?.totalUnitCount == nextPrompt.count)
                #expect(accepted?.detail == "disk")
                #expect(warmCache.allSatisfy { $0.offset == nextPrompt.count })

                // An existing eligible record must NOT override the explicit
                // required-tool fresh-selection gate. Observe actual iterator
                // prefill stages and fetch telemetry, not a source keyword.
                let progress = OSAllocatedUnfairLock(initialState: [PrefillProgress]())
                let hitsBefore = reader.snapshotStats().diskStats?.hits
                let requiredCache = model.newCache(parameters: parameters)
                let required = try TokenIterator(
                    input: Self.input(nextPrompt, policy: .freshRequiredToolSelection),
                    model: model,
                    cache: requiredCache, parameters: parameters, cacheCoordinator: reader,
                    prefillProgressHandler: { event in progress.withLock { $0.append(event) } })
                withExtendedLifetime(required) { MLX.eval(requiredCache) }
                let observed = progress.withLock { $0 }
                #expect(!observed.contains { $0.stage == .cacheRestore })
                #expect(observed.first(where: { $0.stage == .prefill })?.completedUnitCount == 0)
                #expect(requiredCache.allSatisfy { $0.offset == nextPrompt.count })
                #expect(reader.snapshotStats().diskStats?.hits == hitsBefore)

                var logits: [Float] = []
                for token in Self.continuation {
                    let ids = MLXArray([Int32(token)], [1, 1])
                    let expected = model(ids, cache: coldCache)
                    let actual = model(ids, cache: restored)
                    MLX.eval(expected, actual)
                    #expect(MLX.all(actual .== expected).item(Bool.self))
                    let values = actual.asType(.float32).asArray(Float.self)
                    let allFinite = values.allSatisfy(\.isFinite)
                    #expect(allFinite)
                    logits.append(contentsOf: values)
                    try Self.assertCacheEqual(restored, coldCache)
                }
                storageLogits.append(logits)
            }
            try #require(storageLogits.count == 2)
            #expect(storageLogits[0] == storageLogits[1])
        }
    }
}
