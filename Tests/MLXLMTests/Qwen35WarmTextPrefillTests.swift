import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

@Suite("Qwen3.5 warm text prefill positions", .serialized)
struct Qwen35WarmTextPrefillTests {
    private func model() throws -> Qwen35 {
        let config = try JSONDecoder().decode(
            Qwen35Configuration.self,
            from: Data(
                """
                {
                  "model_type": "qwen3_5", "text_config": {
                    "model_type": "qwen3_5_text", "hidden_size": 32,
                    "num_hidden_layers": 4, "intermediate_size": 64,
                    "num_attention_heads": 4, "num_key_value_heads": 2,
                    "linear_num_value_heads": 4, "linear_num_key_heads": 2,
                    "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                    "linear_conv_kernel_dim": 4, "head_dim": 8,
                    "full_attention_interval": 4, "vocab_size": 100,
                    "tie_word_embeddings": false, "rms_norm_eps": 1e-6,
                    "rope_parameters": {
                      "rope_type": "default", "rope_theta": 10000.0,
                      "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]
                    }
                  },
                  "vision_config": {
                    "model_type": "qwen3_vl", "depth": 1,
                    "hidden_size": 16, "intermediate_size": 32,
                    "out_hidden_size": 32, "num_heads": 4, "patch_size": 2,
                    "spatial_merge_size": 2, "temporal_patch_size": 1,
                    "num_position_embeddings": 16
                  },
                  "vocab_size": 100, "image_token_id": 98, "video_token_id": 97,
                  "vision_start_token_id": 96, "vision_end_token_id": 95
                }
                """.utf8))
        let model = Qwen35(config)
        // Explicit parameters make the row repeatable without changing a
        // process-global random seed or relying on random initialization order.
        var values: [String: MLXArray] = [:]
        for (name, parameter) in model.parameters().flattened() {
            let seed = name.utf8.reduce(0) { $0 + Int($1) }
            let data: [Float]
            if name.hasSuffix(".A_log") || name.hasSuffix(".dt_bias") {
                data = Array(repeating: 0, count: parameter.size)
            } else if name.contains("norm"), name.hasSuffix(".weight") {
                let gamma: Float = 1
                data = Array(repeating: gamma, count: parameter.size)
            } else {
                data = (0 ..< parameter.size).map { Float(($0 * 17 + seed) % 41 - 20) * 0.015 }
            }
            values[name] = MLXArray(data, parameter.shape).asType(parameter.dtype)
        }
        model.update(parameters: ModuleParameters.unflattened(values))
        MLX.eval(model)
        return model
    }

    private func text(_ ids: [Int32]) -> LMInput {
        LMInput(tokens: MLXArray(ids, [1, ids.count]))
    }

    private func logits(_ prepared: PrepareResult) throws -> MLXArray {
        guard case .logits(let result) = prepared else {
            Issue.record("Expected native VLM logits")
            throw CocoaError(.coderInvalidValue)
        }
        MLX.eval(result.logits)
        return result.logits
    }

    private func exact(_ actual: MLXArray, _ expected: MLXArray) {
        #expect(actual.shape == expected.shape)
        guard actual.shape == expected.shape else { return }
        #expect(MLX.all(MLX.isFinite(actual)).item(Bool.self))
        #expect(MLX.arrayEqual(actual, expected).item(Bool.self))
    }

    @Test("warm native prepare matches cold native chunking with nonzero norms",
        arguments: [4, 12], [1, 4, 5, 25])
    func warmText(prefixCount: Int, suffixCount: Int) throws {
        try MLXMetalTestLock.withLock {
            let reference = try model()
            let candidate = try model()
            let expectedCache = reference.newCache(parameters: nil)
            let actualCache = candidate.newCache(parameters: nil)
            let ids = (0..<(prefixCount + suffixCount)).map { Int32(1 + $0 % 30) }
            let expected = try logits(reference.prepare(text(ids), cache: expectedCache, windowSize: 4))
            _ = try logits(candidate.prepare(text(Array(ids.prefix(prefixCount))), cache: actualCache, windowSize: 4))
            let actual = try logits(candidate.prepare(text(Array(ids.dropFirst(prefixCount))), cache: actualCache, windowSize: 4))
            MLX.eval(actualCache, expectedCache)
            // Zero-gamma fixtures cannot detect position errors. Check the
            // signal itself before asserting cold/warm equality.
            #expect(MLX.max(abs(expected)).item(Float.self) > 0.01)
            #expect(MLX.max(abs(expectedCache[3].state[0])).item(Float.self) > 0.01)
            exact(actual, expected)
            for (lhs, rhs) in zip(actualCache, expectedCache) {
                #expect(lhs.offset == ids.count)
                #expect(lhs.offset == rhs.offset)
                #expect(lhs.state.count == rhs.state.count)
                for (a, b) in zip(lhs.state, rhs.state) { exact(a, b) }
            }
            for token: Int32 in [41, 42] {
                exact(candidate(text([token]).text, cache: actualCache, state: nil).logits,
                    reference(text([token]).text, cache: expectedCache, state: nil).logits)
            }
        }
    }

    @Test("a previous media request cannot lend its delta to a new text cache")
    func unrelatedMediaStateIsReset() throws {
        try MLXMetalTestLock.withLock {
            let reference = try model()
            let candidate = try model()
            let mediaIds: [Int32] = [1, 2, 96, 98, 98, 98, 98, 95, 3, 4]
            let media = LMInput(
                text: .init(tokens: MLXArray(mediaIds, [1, mediaIds.count])),
                image: .init(
                    pixels: MLXArray((0..<16*12).map { Float($0 % 19) / 19 }, [16, 12]),
                    frames: [THW(1, 4, 4)]))
            _ = try logits(candidate.prepare(media, cache: candidate.newCache(parameters: nil), windowSize: 4))
            let ids = (0..<17).map { Int32(1 + $0) }
            let expectedCache = reference.newCache(parameters: nil)
            let actualCache = candidate.newCache(parameters: nil)
            let expected = try logits(reference.prepare(text(ids), cache: expectedCache, windowSize: 4))
            _ = try logits(candidate.prepare(text(Array(ids.prefix(12))), cache: actualCache, windowSize: 4))
            let actual = try logits(candidate.prepare(text(Array(ids.dropFirst(12))), cache: actualCache, windowSize: 4))
            #expect(MLX.max(abs(expected)).item(Float.self) > 0.01)
            exact(actual, expected)
            for (lhs, rhs) in zip(actualCache, expectedCache) {
                for (a, b) in zip(lhs.state, rhs.state) { exact(a, b) }
            }
        }
    }

    @Test("cache-free text prefill retains the complete prompt")
    func cacheFreeText() throws {
        try MLXMetalTestLock.withLock {
            let reference = try model()
            let candidate = try model()
            let ids = (0..<9).map { Int32(1 + $0) }
            let expected = try logits(reference.prepare(text(ids), cache: [], windowSize: 0))
            let actual = try logits(candidate.prepare(text(ids), cache: [], windowSize: 4))
            #expect(MLX.max(abs(expected)).item(Float.self) > 0.01)
            exact(actual, expected)
        }
    }
}
