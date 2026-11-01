// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Cmlx
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The Swift M3 decode loops audited for vMLX Python's 2026-10-29 bug class (a per-step copy of the cache lanes; appends
/// turned copy-on-write; per-step syncs): synthetic REAL-WIDTH MiniMax-M3 (hidden 6144, 64 x 128 heads, 4 KV, index
/// 4 x 128, MSA blocks 128 / top 16, every layer sparse, 8 affine-4 experts), contexts that are not multiples of the
/// 256-row cache step or the 2048 prefill step. Opt-in: `VMLX_M3_LOOP_BENCH=1` (release build for timings).
///
/// - Plain lane (`TokenIterator`, the BatchEngine solo path's iterator): per-token time and, in a separate pass, every
///   layer's lane buffers compared token to token (in place / copied / grown).
/// - DSpark verify rounds (checkpoint, an 8-row verify, a rollback to 3 kept rows): per-round time and lane copies with
///   the row checkpoint vs the reference checkpoint, interleaved.
@Suite("MiniMax-M3 decode loop audit (opt-in, real width)", .serialized)
struct MiniMaxM3DecodeLoopBenchTests {
    typealias S = MiniMaxM3ServingFixesTests
    typealias DS = MiniMaxM3DSparkTests

    static let vocabulary = 32_000

    static func configuration(layers: Int) throws -> MiniMaxM3Configuration {
        let text: [String: Any] = [
            "model_type": "minimax_m3", "hidden_size": 6144, "num_hidden_layers": layers, "intermediate_size": 3072,
            "dense_intermediate_size": 12288, "shared_intermediate_size": 3072, "num_attention_heads": 64,
            "num_key_value_heads": 4, "head_dim": 128, "rotary_dim": 64, "rope_theta": 10_000_000.0, "rms_norm_eps": 1e-6,
            "vocab_size": vocabulary, "num_local_experts": 8, "num_experts_per_tok": 4, "routed_scaling_factor": 2.0,
            "norm_topk_prob": true, "swiglu_alpha": 1.702, "swiglu_limit": 7.0,
            "moe_layer_freq": [Int](repeating: 1, count: layers), "tie_word_embeddings": false,
            "sparse_attention_config": [
                "sparse_num_index_heads": 4, "sparse_index_dim": 128, "sparse_block_size": 128,
                "sparse_topk_blocks": 16, "sparse_local_block": 1,
                "sparse_attention_freq": [Int](repeating: 1, count: layers),
            ],
        ]
        let json: [String: Any] = ["model_type": "minimax_m3_vl", "text_config": text]
        return try JSONDecoder.json5().decode(
            MiniMaxM3Configuration.self, from: JSONSerialization.data(withJSONObject: json))
    }

    static func model(layers: Int = 8) throws -> MiniMaxM3Model {
        let model = MiniMaxM3Model(try configuration(layers: layers))
        var key = MLXRandom.key(17)
        var update: [(String, MLXArray)] = []
        for (name, value) in model.parameters().flattened() {
            let split = MLXRandom.split(key: key)
            key = split.0
            let r = MLXRandom.normal(value.shape, key: split.1)
            if name.hasSuffix("norm.weight") {
                update.append((name, (r * Float(0.1)).asType(.bfloat16)))
            } else if name.hasSuffix("e_score_correction_bias") {
                update.append((name, (r * Float(0.05)).asType(.float32)))
            } else if name.hasSuffix(".mlp.gate.weight") {
                update.append((name, (r * Float(0.3)).asType(.float32)))
            } else if name.contains("embed_tokens") {
                update.append((name, r.asType(.bfloat16)))
            } else {
                update.append((name, (r * Float(1.5 / Double(value.dim(-1)).squareRoot())).asType(.bfloat16)))
            }
            if update.count % 16 == 0 { eval(update.suffix(16).map(\.1)) }
        }
        try model.update(parameters: ModuleParameters.unflattened(update), verify: .all)
        quantize(model: model, filter: DS.plan)
        eval(model)
        return model
    }

    static func prompt(_ n: Int, seed: UInt64) -> [Int] {
        MLXRandom.randInt(0 ..< Int32(vocabulary - 1), [n], key: MLXRandom.key(seed)).asArray(Int32.self).map(Int.init)
    }

    static func median(_ v: [Double]) -> Double {
        let s = v.sorted()
        return s.isEmpty ? .nan : s[s.count / 2]
    }

    static let contexts = [3011, 9001, 33013]

    @Test("Plain lane (TokenIterator): per-token cost vs context and every lane append in place")
    func plainLane() throws {
        guard ProcessInfo.processInfo.environment["VMLX_M3_LOOP_BENCH"] == "1" else { return }
        try Device.withDefaultDevice(.gpu) {
            let model = try Self.model()
            let steps = 48
            for ctx in Self.contexts {
                #expect(ctx % 256 != 0 && ctx % 2048 != 0)
                var perToken: [Double] = []
                var tokens: [[Int]] = []
                for probe in 0 ..< 4 {
                    var parameters = GenerateParameters(maxTokens: steps, temperature: 0, prefillStepSize: 2048)
                    parameters.maxTokens = steps
                    var it = try TokenIterator(
                        input: LMInput(tokens: MLXArray(Self.prompt(ctx, seed: 5).map { Int32($0) })), model: model,
                        parameters: parameters)
                    var out: [Int] = []
                    var t4 = 0.0
                    while let t = it.next() {
                        out.append(t)
                        if out.count == 4 { t4 = Date().timeIntervalSince1970 }
                    }
                    let ms = (Date().timeIntervalSince1970 - t4) * 1000 / Double(out.count - 4)
                    if probe > 0 { perToken.append(ms) }
                    tokens.append(out)
                }
                #expect(Set(tokens.map { $0 }).count == 1, "ctx \(ctx): greedy tokens differ between probes")
                // Counting passes: lane buffers after every token, (a) as the loop runs (no sync: next() returns with the
                // next step encoded, so its lane arrays are allocated) and (b) synchronized after each token.
                func countPass(sync: Bool) throws -> S.CopyCount {
                    var parameters = GenerateParameters(maxTokens: 40, temperature: 0, prefillStepSize: 2048)
                    parameters.maxTokens = 40
                    var it = try TokenIterator(
                        input: LMInput(tokens: MLXArray(Self.prompt(ctx, seed: 5).map { Int32($0) })), model: model,
                        parameters: parameters)
                    var sparse = S.CopyCount(), dense = S.CopyCount()
                    var previous: [[(UInt, Int)]]?
                    while it.next() != nil {
                        if sync { Stream.gpu.synchronize() }
                        let now = S.lanes(it.cache)
                        if let previous { S.count(previous, now, cache: it.cache, sparse: &sparse, dense: &dense) }
                        previous = now
                    }
                    return sparse
                }
                let sparse = try countPass(sync: false)
                let synced = try countPass(sync: true)
                print("[m3 loop] plain ctx \(ctx): \(String(format: "%.2f", Self.median(perToken))) ms/token "
                    + "(probes \(perToken.map { String(format: "%.2f", $0) })); lanes per token, pipelined \(sparse), "
                    + "synchronized \(synced)")
                #expect(sparse.copied == 0, "ctx \(ctx): \(sparse.copied) lane appends copied")
                #expect(sparse.inPlace > 0)
            }
        }
    }

    @Test("DSpark verify rounds: per-round cost and lane copies, row checkpoint vs reference checkpoint (interleaved)")
    func dsparkRounds() throws {
        guard ProcessInfo.processInfo.environment["VMLX_M3_LOOP_BENCH"] == "1" else { return }
        try Device.withDefaultDevice(.gpu) {
            let model = try Self.model()
            model.model.dsparkTapLayers = [1, 5]
            model.model.dsparkTailCapacity = 512
            for ctx in Self.contexts {
                let cache = model.newCache(parameters: nil)
                DS.prefill(model, Self.prompt(ctx, seed: 9), cache: cache, step: 2048)
                var ms: [Bool: [Double]] = [true: [], false: []]
                var copies: [Bool: S.CopyCount] = [true: .init(), false: .init()]
                for round in 0 ..< 14 {
                    let reference = round % 2 == 1
                    try MiniMaxM3SparseCache.$referenceCheckpointOverride.withValue(reference) {
                        let before = S.lanes(cache)
                        let t0 = Date()
                        let checkpoint = try #require(SpeculativeCacheListCheckpoint(cache))
                        let rows = Self.prompt(8, seed: UInt64(100 + round))
                        _ = try DS.verify(model, rows, cache: cache)
                        eval(cache.flatMap { $0.innerState() })
                        // Rollback to 3 kept rows (as a partial accept), then the next pending token is row 3's.
                        for layer in cache { layer.trim(5) }
                        withExtendedLifetime(checkpoint) {}
                        let dt = Date().timeIntervalSince(t0) * 1000
                        var dense = S.CopyCount()
                        var c = copies[reference]!
                        S.count(before, S.lanes(cache), cache: cache, sparse: &c, dense: &dense)
                        copies[reference] = c
                        if round >= 2 { ms[reference]!.append(dt) }
                    }
                }
                let row = Self.median(ms[false]!), ref = Self.median(ms[true]!)
                print("[m3 loop] dspark ctx \(ctx): round (8-row verify + rollback) row checkpoint "
                    + "\(String(format: "%.2f", row)) ms, reference \(String(format: "%.2f", ref)) ms "
                    + "(\(String(format: "%.2fx", ref / row))); lanes row \(copies[false]!) reference \(copies[true]!)")
                #expect(copies[false]!.copied == 0 && copies[true]!.copied > 0)
            }
        }
    }
}
