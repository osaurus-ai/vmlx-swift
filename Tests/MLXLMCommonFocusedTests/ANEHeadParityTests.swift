// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The ANE head program against the real MLX `Qwen35MTPModule` it is emitted
// from: same random weights, same (hidden, token) inputs, chained over a few
// positions and through the tile commit path. Skips without an ANE.

import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("ANE head parity", .serialized)
struct ANEHeadParityTests {

    private static func makeModel() throws -> Qwen35TextModel {
        let json = """
            {
              "hidden_size": 128,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "head_dim": 32,
              "intermediate_size": 256,
              "vocab_size": 512,
              "num_hidden_layers": 2,
              "mtp_num_hidden_layers": 1,
              "num_experts": 0,
              "partial_rotary_factor": 0.25,
              "rms_norm_eps": 1e-6
            }
            """
        let config = try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(json.utf8))
        let model = Qwen35TextModel(config)
        // Random gains on every norm so the emitter's norm folding is exercised.
        let params = model.parameters().flattened()
        var updates: [String: MLXArray] = [:]
        for (key, value) in params where key.hasSuffix("norm.weight") || key.contains("norm_") {
            updates[key] = MLXRandom.uniform(low: 0.5, high: 1.5, value.shape)
        }
        model.update(parameters: ModuleParameters.unflattened(updates))
        MLX.eval(model.parameters())
        return model
    }

    private static func cosine(_ a: [Float16], _ b: [Float]) -> Double {
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0 ..< a.count {
            let x = Double(Float(a[i])), y = Double(b[i])
            dot += x * y; na += x * x; nb += y * y
        }
        return dot / ((na * nb).squareRoot() + 1e-30)
    }

    @Test("chained draft steps match the MLX head")
    func chainedDraftSteps() throws {
        guard ANEProgram.isAvailable else { print("ANE unavailable; skipping"); return }
        try FocusedMLXTestSupport.withLock {
            MLXRandom.seed(41)
            let model = try Self.makeModel()
            let source = try #require(model.aneHeadWeightSource(draftVocab: 512, window: 64))
            let runner = try ANEHeadRunner(source: source, cacheDirectory: nil)
            print("ANE head compiled in \(runner.program.compileSeconds)s")

            let cache = model.makeNativeMTPCache()
            var matches = 0
            var minCos = 1.0
            let steps = 8
            for _ in 0 ..< steps {
                let hidden = MLXRandom.normal([1, 1, 128])
                let token = Int.random(in: 0 ..< 512)
                MLX.eval(hidden)
                let ref = model.nativeMTPForward(
                    hiddenStates: hidden, nextTokenIds: MLXArray([Int32(token)]).reshaped(1, 1), cache: cache)
                let refToken = ref.logits.reshaped(-1).argMax().item(Int.self)
                let refHidden = ref.hiddenStates.reshaped(-1).asArray(Float.self)

                let h16 = hidden.reshaped(-1).asArray(Float.self).map { Float16($0) }
                let out = try runner.draftStep(hidden: h16, token: token)
                let cos = Self.cosine(out.hidden, refHidden)
                minCos = min(minCos, cos)
                if out.token == refToken { matches += 1 }
                print("step \(runner.length): cos \(cos) ane \(out.token) ref \(refToken)")
            }
            #expect(minCos > 0.995)
            #expect(matches >= steps - 1)
        }
    }

    @Test("tile commit then a draft step matches the MLX head fed the same pairs")
    func commitThenDraft() throws {
        guard ANEProgram.isAvailable else { print("ANE unavailable; skipping"); return }
        try FocusedMLXTestSupport.withLock {
            MLXRandom.seed(43)
            let model = try Self.makeModel()
            let source = try #require(model.aneHeadWeightSource(draftVocab: 512, window: 64))
            let runner = try ANEHeadRunner(source: source, cacheDirectory: nil)

            let n = 5
            let hiddens = MLXRandom.normal([1, n, 128])
            let tokens = (0 ..< n).map { _ in Int32.random(in: 0 ..< 512) }
            MLX.eval(hiddens)
            let cache = model.makeNativeMTPCache()
            _ = model.nativeMTPForward(hiddenStates: hiddens, nextTokenIds: MLXArray(tokens).reshaped(1, n), cache: cache)

            var pairs: [(hidden: [Float16], token: Int)] = []
            for i in 0 ..< n {
                let h = hiddens[0, i].asArray(Float.self).map { Float16($0) }
                pairs.append((h, Int(tokens[i])))
            }
            try runner.commit(pairs: pairs)
            #expect(runner.length == n)

            let hidden = MLXRandom.normal([1, 1, 128])
            let token = 77
            MLX.eval(hidden)
            let ref = model.nativeMTPForward(
                hiddenStates: hidden, nextTokenIds: MLXArray([Int32(token)]).reshaped(1, 1), cache: cache)
            let refToken = ref.logits.reshaped(-1).argMax().item(Int.self)
            let refHidden = ref.hiddenStates.reshaped(-1).asArray(Float.self)
            let out = try runner.draftStep(hidden: hidden.reshaped(-1).asArray(Float.self).map { Float16($0) }, token: token)
            let cos = Self.cosine(out.hidden, refHidden)
            print("after commit: cos \(cos) ane \(out.token) ref \(refToken)")
            #expect(cos > 0.995)
            #expect(out.token == refToken)
        }
    }
}
