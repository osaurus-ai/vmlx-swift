// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Regression guard: the native MTP (multi-token-prediction) head runs in the
// model's activation dtype, not float32. An fp32 promotion anywhere in the
// head (norms, the fusion `fc`, the decoder layers, or the final norm) would
// feed fp32 hidden states back into the draft-logit projection and the MTP
// KV cache — the same promotion-cascade tax fixed for the base model in #407,
// but on the speculative-decode hot path. This constructs the real
// `Qwen35MTPModule` at bf16 and pins that its output stays bf16.

import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct MTPHeadDtypeTests {

    private static func makeConfig() throws -> Qwen35TextConfiguration {
        // Minimal Qwen3.5 text config with a single MTP decoder layer and a
        // dense (non-MoE) MLP so the head is small but structurally real.
        let json = """
            {
              "hidden_size": 64,
              "num_attention_heads": 8,
              "num_key_value_heads": 8,
              "head_dim": 8,
              "intermediate_size": 128,
              "vocab_size": 256,
              "mtp_num_hidden_layers": 1,
              "num_experts": 0,
              "rms_norm_eps": 1e-6
            }
            """
        return try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
    }

    @Test("native MTP head keeps bf16 activations end to end")
    func mtpHeadStaysBf16() throws {
        try FocusedMLXTestSupport.withLock {
            MLXRandom.seed(19)
            let config = try Self.makeConfig()
            let head = Qwen35MTPModule(config)
            let embed = Embedding(embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
            // The real loader converts the MTP head weights to bf16 (same as
            // the base model); mirror that so this pins the activation path,
            // not MLXNN's fp32 default weight init.
            head.update(parameters: head.parameters().mapValues { $0.asType(.bfloat16) })
            embed.update(parameters: embed.parameters().mapValues { $0.asType(.bfloat16) })
            MLX.eval(head.parameters())
            MLX.eval(embed.parameters())

            // bf16 hidden state + the next-token id (int) — the shapes a
            // draft step feeds the head.
            let hidden = MLXRandom.normal([1, 1, 64]).asType(.bfloat16)
            let nextIds = MLXArray([Int32(7)]).reshaped(1, 1)
            MLX.eval(hidden, nextIds)

            let out = head(
                hiddenStates: hidden,
                nextTokenIds: nextIds,
                embedTokens: embed,
                cache: nil)
            MLX.eval(out)

            #expect(
                out.dtype == .bfloat16,
                Comment(rawValue: "MTP head output dtype was \(out.dtype), expected bfloat16"))
        }
    }

    @Test("fp16 activations stay fp16 through the MTP head (no fp32 promotion)")
    func mtpHeadStaysFp16() throws {
        try FocusedMLXTestSupport.withLock {
            MLXRandom.seed(23)
            let config = try Self.makeConfig()
            let head = Qwen35MTPModule(config)
            let embed = Embedding(embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
            // Cast the head + embedding weights to fp16 so any fp32 op inside
            // the head would surface as an fp32 output (fp16 + fp32 -> fp32).
            head.update(parameters: head.parameters().mapValues { $0.asType(.float16) })
            embed.update(parameters: embed.parameters().mapValues { $0.asType(.float16) })
            MLX.eval(head.parameters())
            MLX.eval(embed.parameters())

            let hidden = MLXRandom.normal([1, 1, 64]).asType(.float16)
            let nextIds = MLXArray([Int32(11)]).reshaped(1, 1)
            MLX.eval(hidden, nextIds)

            let out = head(
                hiddenStates: hidden,
                nextTokenIds: nextIds,
                embedTokens: embed,
                cache: nil)
            MLX.eval(out)

            #expect(
                out.dtype == .float16,
                Comment(rawValue: "MTP head promoted fp16 -> \(out.dtype) (fp32 leak on the head)"))
        }
    }
}
