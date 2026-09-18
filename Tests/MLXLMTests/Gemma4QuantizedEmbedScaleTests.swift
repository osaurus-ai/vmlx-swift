// Copyright © 2026 osaurus-eval contributors
// SPDX-License-Identifier: MIT
//
// A quantized Gemma-4 text embedding has to scale decoded tokens like the prompt.
//
// Gemma-4 multiplies its token embeddings by sqrt(hidden_size). `prepare` does that in the dtype of
// the rows it looked up. A decode step reaches the text model with token ids, and that path built the
// scale in `embed_tokens.weight.dtype`. Once the table is quantized that is the packed uint32 array,
// so the scale was truncated to floor(sqrt(hidden_size)) for every generated token, while the prompt
// kept the exact one. Scoring that runs the whole sequence through `prepare` never takes the decode
// path, so it could not see this.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

@Suite("Gemma-4 quantized embedding scale")
struct Gemma4QuantizedEmbedScaleTests {

    /// hidden_size 96 is deliberately not a square: sqrt(96) = 9.80, and the defect truncated the
    /// scale to 9.
    static let gemma4JSON = """
        {
            "model_type": "gemma4",
            "text_config": {
                "hidden_size": 96,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 48,
                "global_head_dim": 48,
                "vocab_size": 64,
                "sliding_window": 16,
                "layer_types": ["full_attention"]
            },
            "vision_config": {
                "hidden_size": 8,
                "output_proj_dims": 8,
                "default_output_length": 2
            }
        }
        """

    /// The comparison's premise: with a float table the two entry points already agree, so any gap
    /// with a quantized one comes from how the table is stored.
    @Test("a float Gemma-4 embedding: the prompt and decode paths agree")
    func gemma4FloatPathsAgree() throws {
        try MLXMetalTestLock.withLock {
            let config = try JSONDecoder().decode(
                Gemma4Configuration.self, from: Data(Self.gemma4JSON.utf8))
            let model = Gemma4(config)
            let tokens = MLXArray((1 ... 6).map { Int32($0) }).reshaped(1, 6)

            guard
                case .logits(let prefill) = try model.prepare(
                    LMInput(tokens: tokens), cache: model.newCache(parameters: nil),
                    windowSize: nil)
            else {
                Issue.record("prepare returned no logits")
                return
            }
            let decode = model(tokens, cache: model.newCache(parameters: nil))

            let worst = abs(prefill.logits.asType(.float32) - decode.asType(.float32)).max()
                .item(Float.self)
            #expect(worst <= 1e-4, "the decode path's logits differ by up to \(worst)")
        }
    }

    /// `prepare` embeds the prompt and scales it by sqrt(hidden_size) in the rows' dtype. A decode
    /// step reaches the text model with token ids, which scaled in `embed_tokens.weight.dtype`, i.e.
    /// by floor(sqrt(hidden_size)) once the table is quantized. Two entry points of one model, fed
    /// the same tokens, must agree.
    @Test("a quantized Gemma-4 embedding scales decoded tokens like the prompt")
    func gemma4DecodeScalesLikePrefill() throws {
        try MLXMetalTestLock.withLock {
            let config = try JSONDecoder().decode(
                Gemma4Configuration.self, from: Data(Self.gemma4JSON.utf8))
            let model = Gemma4(config)
            quantize(
                model: model,
                filter: { path, _ in path.hasSuffix(".embed_tokens") ? (32, 4, .affine) : nil })
            let tokens = MLXArray((1 ... 6).map { Int32($0) }).reshaped(1, 6)

            guard
                case .logits(let prefill) = try model.prepare(
                    LMInput(tokens: tokens), cache: model.newCache(parameters: nil),
                    windowSize: nil)
            else {
                Issue.record("prepare returned no logits")
                return
            }
            let decode = model(tokens, cache: model.newCache(parameters: nil))

            let worst = abs(prefill.logits.asType(.float32) - decode.asType(.float32)).max()
                .item(Float.self)
            #expect(worst <= 1e-4, "the decode path's logits differ by up to \(worst)")
        }
    }
}
