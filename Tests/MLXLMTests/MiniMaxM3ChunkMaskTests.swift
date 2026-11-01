// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// The mask a MiniMax-M3 forward hands its layers, and what it does to MSA routing on chunks past the first.
///
/// The model passes the Python runtime's mask (`.causal` for every multi-token call, any cache offset). The cache's own
/// mask is an explicit array once the offset is nonzero, and the MSA router sends any array mask to the dense mask
/// path, so with it every prefill chunk after the first skipped the block-sparse kernels (found on the real 5-layer
/// M3.1 bundles, 2026-10-28: Swift auto never left the dense path in chunked prefill; Python R8 auto took the chunk
/// kernel from 5120 keys; with the fix the two are bit-identical in strict float32).
///
/// Geometry: 16 query heads over 2 KV groups (8 per group: the chunk kernel's minimum), head dim 64, 32-key blocks,
/// top-2, so a 100-token first chunk already selects (more than 64 keys) and the 37-row second chunk sits at offset
/// 100 (neither a block nor a step multiple).
@Suite("MiniMax-M3 chunk mask and MSA routing", .serialized)
struct MiniMaxM3ChunkMaskTests {
    static let first = 100, second = 37

    static func model(sparse: Bool, seed: UInt64) throws -> MiniMaxM3Model {
        let json: [String: Any] = [
            "model_type": "minimax_m3", "hidden_size": 256, "num_hidden_layers": 1,
            "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 64, "rotary_dim": 32,
            "vocab_size": 64, "moe_layer_freq": [0], "dense_intermediate_size": 128,
            "sparse_attention_config": [
                "sparse_num_index_heads": 2, "sparse_index_dim": 64, "sparse_block_size": 32,
                "sparse_topk_blocks": 2, "sparse_local_block": 1, "sparse_attention_freq": [sparse ? 1 : 0],
            ],
        ]
        let configuration = try JSONDecoder().decode(
            MiniMaxM3Configuration.self, from: JSONSerialization.data(withJSONObject: json))
        MLXRandom.seed(seed)
        let model = MiniMaxM3Model(configuration)
        eval(model)
        return model
    }

    static func tokens(_ n: Int, seed: UInt64) -> MLXArray {
        MLXRandom.randInt(Int32(0) ..< Int32(64), [1, n], key: MLXRandom.key(seed))
    }

    static func withRequest<R>(_ request: MiniMaxM3MSA.Request?, _ body: () throws -> R) rethrows -> R {
        let previous = MiniMaxM3MSA.hooks.request
        MiniMaxM3MSA.hooks.request = request
        defer { MiniMaxM3MSA.hooks.request = previous }
        return try body()
    }

    final class Spy: @unchecked Sendable { var picks: [MiniMaxM3MSA.Path] = [] }

    static func spying<R>(_ body: (Spy) throws -> R) rethrows -> R {
        let spy = Spy()
        let previous = MiniMaxM3MSA.hooks.pathObserver
        MiniMaxM3MSA.hooks.pathObserver = { spy.picks.append($0) }
        defer { MiniMaxM3MSA.hooks.pathObserver = previous }
        return try body(spy)
    }

    /// Logits of the second chunk (offset `first`) after the first, plus one decode step.
    static func chunked(_ model: MiniMaxM3Model, _ tokens: MLXArray) -> (chunk: MLXArray, decode: MLXArray) {
        let cache = model.newCache(parameters: nil)
        eval(model(tokens[0..., ..<first], cache: cache))
        let chunk = model(tokens[0..., first ..< (first + second)], cache: cache)
        let decode = model(tokens[0..., (first + second) ..< (first + second + 1)], cache: cache)
        eval(chunk, decode)
        #expect(cache.allSatisfy { $0.offset == first + second + 1 })
        return (chunk, decode)
    }

    static func relativeError(_ a: MLXArray, _ b: MLXArray) -> Float {
        (abs(a.asType(.float32) - b.asType(.float32)).max() / abs(b.asType(.float32)).max()).item(Float.self)
    }

    @Test("A multi-token call at a nonzero cache offset gets the symbolic causal mask, as in the Python runtime")
    func maskIsSymbolicCausal() throws {
        let model = try Self.model(sparse: true, seed: 1)
        let cache = model.newCache(parameters: nil)
        let tokens = Self.tokens(Self.first + Self.second + 1, seed: 2)
        eval(model(tokens[0..., ..<Self.first], cache: cache))
        // The cache would hand out an explicit array here; the model must not.
        if case .array = cache[0].makeMask(n: Self.second, windowSize: nil, returnArray: false) {
        } else {
            Issue.record("precondition: the cache's own mask at offset \(Self.first) is expected to be an array")
        }
        let observed = Self.spying { spy -> [MiniMaxM3MSA.Path] in
            Self.withRequest(.sparse) {
                eval(model(tokens[0..., Self.first ..< (Self.first + Self.second)], cache: cache))
            }
            return spy.picks
        }
        // Sparse requested, 37 rows, 137 keys > top-2 x 32: the chunk kernel, not the dense mask path.
        #expect(observed == [.kernel], "second chunk took \(observed)")
    }

    @Test("Chunk past the first: block-sparse kernel == dense mask path; dense layers: causal == explicit array mask")
    func chunkPathsAgree() throws {
        let tokens = Self.tokens(Self.first + Self.second + 1, seed: 3)
        // Sparse layer: the chunk kernel (now reachable at a nonzero offset) against the dense mask reference.
        let sparse = try Self.model(sparse: true, seed: 4)
        try Device.withDefaultDevice(.gpu) {
            let (kernelChunk, kernelDecode) = Self.spying { spy -> (MLXArray, MLXArray) in
                let r = Self.withRequest(.sparse) { Self.chunked(sparse, tokens) }
                // Both chunks (offset 0 and offset 100) on the chunk kernel; the decode step routes by its own rule.
                #expect(Array(spy.picks.prefix(2)) == [.kernel, .kernel], "picks \(spy.picks)")
                return r
            }
            let (denseChunk, denseDecode) = Self.withRequest(.dense) { Self.chunked(sparse, tokens) }
            let tolerance: Float = ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0" ? 2e-5 : 6e-3
            let (e1, e2) = (Self.relativeError(kernelChunk, denseChunk), Self.relativeError(kernelDecode, denseDecode))
            print("[MiniMaxM3ChunkMaskTests] sparse layer, kernel vs dense: chunk \(e1), decode \(e2)")
            #expect(e1 < tolerance && e2 < tolerance)
        }
        // Dense attention layer: the model's causal mask vs the explicit offset-aware array the cache builds.
        let dense = try Self.model(sparse: false, seed: 5)
        let cache = dense.newCache(parameters: nil)
        eval(dense(tokens[0..., ..<Self.first], cache: cache))
        let x = tokens[0..., Self.first ..< (Self.first + Self.second)]
        let copy = cache.map { $0.copy() }
        let viaModel = dense(x, cache: cache)
        let inner = dense.model
        var h = inner.embedTokens(x)
        let explicit = MLXFast.ScaledDotProductAttentionMaskMode.array(
            createCausalMask(n: Self.second, offset: Self.first))
        h = inner.layers[0](h, mask: explicit, cache: copy[0])
        let viaArray = dense.lmHead!(inner.norm(h))
        eval(viaModel, viaArray)
        let e = Self.relativeError(viaModel, viaArray)
        print("[MiniMaxM3ChunkMaskTests] dense layer, causal vs explicit array mask at offset \(Self.first): \(e)")
        #expect(e < 1e-5)
    }
}
