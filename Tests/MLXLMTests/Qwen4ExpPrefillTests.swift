import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom
import Testing

@testable import MLXVLM

@Suite("Flash text prefill window", .serialized)
struct Qwen4ExpPrefillTests {
    private final class Progress: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [Int] = []
        func append(_ value: Int) { lock.withLock { values.append(value) } }
        func read() -> [Int] { lock.withLock { values } }
    }

    private func withFixture(_ body: (Qwen4Exp) throws -> Void) throws {
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
        let config = try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: data)
        let text = config.base.textConfiguration
        try #require(text.hiddenSize == 64 && text.hiddenLayers == 2)
        try #require(text.vocabularySize == 128 && text.numExperts == 8)
        MLXRandom.seed(0)
        let model = Qwen4Exp(config)
        let count = model.parameters().flattened().reduce(0) { $0 + $1.1.size }
        try #require(count < 2_000_000, "refuse fixture drift before MLX evaluation")

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

    @Test("caller window bounds Flash prefill and reports completed chunks")
    func flashPrepareHonorsWindowAndReportsProgress() throws {
        try MLXMetalTestLock.withLock {
            try withFixture { model in
                let cache = model.newCache(parameters: nil)
                let tokens = MLXArray([Int32(2), 3, 4, 5, 6, 7, 8]).reshaped(1, 7)
                let progress = Progress()
                let result = try PrefillProgressReporter.withHandler({ progress.append($0) }) {
                    try model.prepare(
                        LMInput(text: .init(tokens: tokens)),
                        cache: cache, windowSize: 3)
                }
                guard case .logits(let output) = result else {
                    Issue.record("Flash prepare must return last-chunk logits")
                    return
                }
                MLX.eval(output.logits)
                MLX.eval(cache)
                #expect(progress.read() == [3, 6])
                #expect(output.logits.shape == [1, 1, 128])
                #expect(cache.allSatisfy { $0.offset == 7 })
                #expect(cache.first is MambaCache)
                #expect(cache.last is QSAKVCache)
            }
        }
    }

    private func expectEqual(_ actual: MLXArray, _ expected: MLXArray, _ label: String) {
        #expect(actual.shape == expected.shape, "\(label) shape")
        guard actual.shape == expected.shape else { return }
        let a = actual.asType(.float32)
        let b = expected.asType(.float32)
        let error = MLX.max(abs(a - b)).item(Float.self)
        #expect(
            allClose(a, b, rtol: 1e-5, atol: 1e-5).item(Bool.self),
            "\(label) maxAbs=\(error)")
    }

    @Test("chunked suffix matches explicit forwards including restored PLE and QSA")
    func chunkedSuffixMatchesExplicitForwards() throws {
        try MLXMetalTestLock.withLock {
            try withFixture { model in
                for prefixLength in [0, 35] {
                    let prefix = model.newCache(parameters: nil)
                    if prefixLength > 0 {
                        let ids = MLXArray((0 ..< prefixLength).map { Int32(2 + $0 % 100) })
                            .reshaped(1, prefixLength)
                        MLX.eval(model(ids, cache: prefix))
                        MLX.eval(prefix)
                    }
                    // copy() discards derived QSA pools, as a restore does;
                    // raw index keys and PLE token/conv history must suffice.
                    let actualCache = prefix.map { $0.copy() }
                    let referenceCache = prefix.map { $0.copy() }
                    let tokens = MLXArray((0 ..< 17).map { Int32(3 + $0 % 100) }).reshaped(1, 17)
                    let result = try model.prepare(
                        LMInput(text: .init(tokens: tokens)),
                        cache: actualCache, windowSize: 8)
                    guard case .logits(let output) = result else {
                        Issue.record("missing logits")
                        return
                    }
                    for range in [0 ..< 8, 8 ..< 16] {
                        MLX.eval(model(tokens[0..., range], cache: referenceCache))
                        MLX.eval(referenceCache)
                    }
                    let expected = model(tokens[0..., 16...], cache: referenceCache)
                    expectEqual(output.logits, expected, "prefix\(prefixLength) last logits")
                    for (layer, pair) in zip(actualCache, referenceCache).enumerated() {
                        #expect(pair.0.offset == prefixLength + 17)
                        #expect(pair.0.offset == pair.1.offset)
                        #expect(pair.0.state.count == pair.1.state.count)
                        for (slot, arrays) in zip(pair.0.state, pair.1.state).enumerated() {
                            expectEqual(arrays.0, arrays.1, "layer\(layer) slot\(slot)")
                        }
                    }
                    let next = MLXArray([Int32(31)]).reshaped(1, 1)
                    expectEqual(
                        model(next, cache: actualCache),
                        model(next, cache: referenceCache), "next decode")
                }
            }
        }
    }

    // MLX's default TF32 path uses reduced precision for float32 matrix ops.
    // Keep the 1e-5 FP32 oracle strict and explicitly run this row with
    // MLX_ENABLE_TF32=0; a default run must report it skipped, not proven.
    @Test(
        "one-shot and chunked prefill agree across QSA budget",
        .enabled(
            if: ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0",
            "Strict FP32 oracle requires MLX_ENABLE_TF32=0; default TF32 drift is recorded separately"
        ))
    func chunkedMatchesOneShot() throws {
        try MLXMetalTestLock.withLock {
            try withFixture { model in
                let tokens = MLXArray((0 ..< 44).map { Int32(2 + $0 % 100) }).reshaped(1, 44)
                let whole = model.newCache(parameters: nil)
                let chunked = model.newCache(parameters: nil)
                let expected = model(tokens, cache: whole)
                MLX.eval(expected)
                let result = try model.prepare(
                    LMInput(text: .init(tokens: tokens)),
                    cache: chunked, windowSize: 8)
                guard case .logits(let output) = result else {
                    Issue.record("missing logits")
                    return
                }
                expectEqual(output.logits, expected[0..., 40..., 0...], "one-shot last chunk")
                let next = MLXArray([Int32(47)]).reshaped(1, 1)
                expectEqual(
                    model(next, cache: chunked), model(next, cache: whole),
                    "one-shot next decode")
            }
        }
    }

    @Test("cancelled text prefill does no cache work")
    func cancelledPrefillDoesNotAdvanceCache() async throws {
        try await MLXMetalTestLock.withLock {
            try withFixture { model in
                let cache = model.newCache(parameters: nil)
                let progress = Progress()
                let tokens = MLXArray([Int32(2), 3, 4, 5]).reshaped(1, 4)
                withUnsafeCurrentTask { $0?.cancel() }
                do {
                    _ = try PrefillProgressReporter.withHandler({ progress.append($0) }) {
                        try model.prepare(
                            LMInput(text: .init(tokens: tokens)),
                            cache: cache, windowSize: 2)
                    }
                    Issue.record("expected cancellation")
                } catch is CancellationError {
                    #expect(progress.read().isEmpty)
                    #expect(cache.allSatisfy { $0.offset == 0 })
                }
            }
        }
    }

    @Test("cancellation after one completed chunk prevents subsequent chunks")
    func cancellationAtChunkBoundary() async throws {
        try await MLXMetalTestLock.withLock {
            try withFixture { model in
                let cache = model.newCache(parameters: nil)
                let progress = Progress()
                let tokens = MLXArray([Int32(2), 3, 4, 5, 6]).reshaped(1, 5)
                do {
                    _ = try PrefillProgressReporter.withHandler({ value in
                        progress.append(value)
                        withUnsafeCurrentTask { $0?.cancel() }
                    }) {
                        try model.prepare(
                            LMInput(text: .init(tokens: tokens)),
                            cache: cache, windowSize: 2)
                    }
                    Issue.record("expected cancellation at second chunk")
                } catch is CancellationError {
                    #expect(progress.read() == [2])
                    #expect(cache.allSatisfy { $0.offset == 2 })
                }
            }
        }
    }

    @Test("short prompts and disabled windows retain single-shot behavior")
    func shortAndDisabledWindows() throws {
        try MLXMetalTestLock.withLock {
            try withFixture { model in
                for window: Int? in [nil, 0, -1, 4, 8] {
                    let cache = model.newCache(parameters: nil)
                    let progress = Progress()
                    let tokens = MLXArray([Int32(2), 3, 4, 5]).reshaped(1, 4)
                    let result = try PrefillProgressReporter.withHandler({ progress.append($0) }) {
                        try model.prepare(
                            LMInput(text: .init(tokens: tokens)),
                            cache: cache, windowSize: window)
                    }
                    guard case .logits(let output) = result else {
                        Issue.record("missing logits")
                        return
                    }
                    MLX.eval(output.logits)
                    #expect(progress.read().isEmpty)
                    #expect(output.logits.shape == [1, 4, 128])
                    #expect(cache.allSatisfy { $0.offset == 4 })
                }
            }
        }
    }
}
