// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepseekV3 MoE dtype-preservation gate. The router computes its scores
// in float32 for numerical stability; applying them to the expert
// outputs without casting back promoted the entire residual stream to
// float32 from the first MoE layer on. Every downstream matmul and every
// cached KV then ran at twice the bf16 cost — observed live on a
// deepseek_v3 MoE bundle whose disk cache stored 46/48 layers' KV as F32
// at exactly 2x the bf16 entry size (1880 KB/token), with only the
// first_k_dense_replace layers still bf16. The expert mix must return
// the input dtype.

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import MLXLLM

final class DeepseekV3MoEDtypeTests: XCTestCase {

    private static let minimalConfigJSON = """
        {
          "model_type": "deepseek_v3",
          "hidden_size": 64,
          "num_hidden_layers": 2,
          "intermediate_size": 128,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "rms_norm_eps": 1e-6,
          "vocab_size": 100,
          "max_position_embeddings": 2048,
          "rope_theta": 10000.0,
          "q_lora_rank": 16,
          "kv_lora_rank": 16,
          "qk_nope_head_dim": 8,
          "qk_rope_head_dim": 8,
          "v_head_dim": 8,
          "moe_intermediate_size": 32,
          "first_k_dense_replace": 1,
          "moe_layer_freq": 1,
          "n_routed_experts": 4,
          "num_experts_per_tok": 2,
          "norm_topk_prob": true,
          "topk_group": 1,
          "n_group": 1,
          "routed_scaling_factor": 1.0
        }
        """

    func testMoEOutputPreservesBF16InputDtype() throws {
        let config = try JSONDecoder().decode(
            DeepseekV3Configuration.self,
            from: Data(Self.minimalConfigJSON.utf8))
        let moe = DeepseekV3MoE(config: config, layerIdx: 1)

        // Freshly initialized modules carry float32 weights; cast them to
        // bf16 so the module matches a real loaded bundle, where the leak
        // was the f32 router scores, not the weights.
        let bf16Params = ModuleParameters.unflattened(
            moe.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }
        )
        try moe.update(parameters: bf16Params, verify: [.noUnusedKeys])

        let x = MLXRandom.normal([1, 3, 64]).asType(.bfloat16)
        let y = moe(x)
        eval(y)

        XCTAssertEqual(
            y.dtype, .bfloat16,
            "MoE expert mix must preserve the residual-stream dtype; an f32 "
                + "result means the router scores leaked into the residual "
                + "and every downstream layer (and its cached KV) runs f32")
    }
}
