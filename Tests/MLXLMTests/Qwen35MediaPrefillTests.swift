import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
import os

@testable import MLXVLM

@Suite("Qwen3.5 media prefill windows", .serialized)
struct Qwen35MediaPrefillTests {
    private func model() throws -> Qwen35 {
        let config = try JSONDecoder().decode(
            Qwen35Configuration.self,
            from: Data("""
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
                let gamma: Float = name.hasPrefix("vision_tower.")
                    || name.hasSuffix(".linear_attn.norm.weight") ? 1 : 0
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

    private func input(videoFirst: Bool, mask: Bool = false) -> LMInput {
        let imageTokens = [Int32(96), 98, 98, 98, 98, 95]
        let tokens: [Int32] = [1, 2]
            + (videoFirst ? [96, 97, 97, 95, 3] : [])
            + imageTokens + [4, 5, 6, 7, 8, 9]
        let pixels = MLXArray((0 ..< 16 * 12).map { Float($0 % 19) / 19 }, [16, 12])
        let videoPixels = MLXArray((0 ..< 8 * 12).map { Float($0 % 13) / 13 }, [8, 12])
        return LMInput(
            text: .init(
                tokens: MLXArray(tokens, [1, tokens.count]),
                mask: mask ? MLXArray.ones([1, tokens.count], dtype: .int32) : nil),
            image: .init(pixels: pixels, frames: [THW(1, 4, 4)]),
            video: videoFirst ? .init(pixels: videoPixels, frames: [THW(2, 2, 2)]) : nil)
    }

    private func logits(_ prepared: PrepareResult) throws -> MLXArray {
        guard case .logits(let result) = prepared else {
            Issue.record("VLM prepare must return logits")
            throw CocoaError(.coderInvalidValue)
        }
        return result.logits
    }

    private func equal(_ actual: MLXArray, _ expected: MLXArray) {
        #expect(actual.shape == expected.shape)
        guard actual.shape == expected.shape else { return }
        let finite = MLX.all(isFinite(actual)).item(Bool.self)
        let error = MLX.max(abs(actual.asType(.float32) - expected.asType(.float32))).item(Float.self)
        let scale = MLX.max(abs(expected.asType(.float32))).item(Float.self)
        #expect(finite)
        #expect(error <= 0.0001 + 0.0001 * scale)
    }

    @Test("actual vision/hybrid forward preserves positions, KV, GDN and next decode",
          arguments: [false, true], [3, 4, 8])
    func chunkParity(videoFirst: Bool, window: Int) throws {
        try MLXMetalTestLock.withLock {
            for initialOffset in [0, 3] {
                let reference = try model()
                let chunked = try model()
                let expectedCache = reference.newCache(parameters: nil)
                let actualCache = chunked.newCache(parameters: nil)
                if initialOffset > 0 {
                    let prefix = MLXArray([Int32(11), 12, 13], [1, 3])
                    MLX.eval(reference(prefix, cache: expectedCache))
                    MLX.eval(chunked(prefix, cache: actualCache))
                }
                let input = input(videoFirst: videoFirst, mask: initialOffset > 0)
                let expected = try logits(reference.prepare(input, cache: expectedCache, windowSize: 0))
                MLX.eval(expected, expectedCache)
                let progress = OSAllocatedUnfairLock(initialState: [Int]())
                let actual = try PrefillProgressReporter.withHandler({ completed in
                    progress.withLock { $0.append(completed) }
                }) {
                    try logits(chunked.prepare(input, cache: actualCache, windowSize: window))
                }
                MLX.eval(actual, actualCache)
                equal(actual[0..., -1, 0...], expected[0..., -1, 0...])
                #expect(actual.dim(1) <= window)
                let expectedProgress = Array(stride(from: window, to: input.text.tokens.dim(1), by: window))
                #expect(progress.withLock { $0 } == expectedProgress)
                #expect(actualCache.count == expectedCache.count)
                for (lhs, rhs) in zip(actualCache, expectedCache) {
                    #expect(lhs.offset == input.text.tokens.dim(1) + initialOffset)
                    #expect(lhs.offset == rhs.offset)
                    #expect(lhs.state.count == rhs.state.count)
                    for (a, b) in zip(lhs.state, rhs.state) { equal(a, b) }
                }
                // Decode uses the delta retained from the entire media prompt,
                // not from its last chunk. A second token catches advancing it.
                for token: Int32 in [31, 32] {
                    let next = MLXArray([token], [1, 1])
                    equal(chunked(next, cache: actualCache), reference(next, cache: expectedCache))
                }
            }
        }
    }

    @Test("fitting and disabled windows keep full-prompt logits without progress", arguments: [0, -1, 19, 64])
    func unchunkedWindows(window: Int) throws {
        try MLXMetalTestLock.withLock {
            let model = try model()
            let input = input(videoFirst: true)
            let progress = OSAllocatedUnfairLock(initialState: [Int]())
            let result = try PrefillProgressReporter.withHandler({ completed in
                progress.withLock { $0.append(completed) }
            }) {
                try logits(model.prepare(input, cache: model.newCache(parameters: nil), windowSize: window))
            }
            MLX.eval(result)
            #expect(result.dim(1) == input.text.tokens.dim(1))
            #expect(progress.withLock { $0.isEmpty })
        }
    }

    @Test("cache-free media forward stays one-shot even with a small window")
    func cacheFreeForward() throws {
        try MLXMetalTestLock.withLock {
            let reference = try model()
            let candidate = try model()
            let input = input(videoFirst: true)
            let expected = try logits(reference.prepare(input, cache: [], windowSize: 0))
            let progress = OSAllocatedUnfairLock(initialState: [Int]())
            let actual = try PrefillProgressReporter.withHandler({ completed in
                progress.withLock { $0.append(completed) }
            }) {
                try logits(candidate.prepare(input, cache: [], windowSize: 3))
            }
            equal(actual, expected)
            #expect(actual.dim(1) == input.text.tokens.dim(1))
            #expect(progress.withLock { $0.isEmpty })
        }
    }

    @Test("Stop before work or after a completed chunk never processes the tail", arguments: [false, true])
    func cancellation(afterChunk: Bool) async throws {
        try await Task {
            try MLXMetalTestLock.withLock {
                let model = try model()
                let cache = model.newCache(parameters: nil)
                let input = input(videoFirst: false)
                if !afterChunk { withUnsafeCurrentTask { $0?.cancel() } }
                do {
                    _ = try PrefillProgressReporter.withHandler({ _ in
                        withUnsafeCurrentTask { $0?.cancel() }
                    }) {
                        try model.prepare(input, cache: cache, windowSize: 8)
                    }
                    Issue.record("cancelled media prefill returned normally")
                } catch is CancellationError {
                    #expect(cache.allSatisfy { $0.offset == (afterChunk ? 8 : 0) })
                }
            }
        }.value
    }
}
