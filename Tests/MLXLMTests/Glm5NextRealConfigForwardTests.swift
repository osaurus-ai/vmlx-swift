// Copyright © 2026 osaurus-eval contributors
// SPDX-License-Identifier: MIT
//
// The shipped CONFIG, with random weights — the cheap half of "does the real model run".
//
// Every GLM-5.3 test so far used a tiny fixture whose dimensions I chose, or loaded a single layer
// from the bundle. Neither exercises the shipped geometry end to end: 64 heads of 256, a 1536 query
// LoRA, a 512 KV LoRA, hc_mult 4. A forward built from the real `config.json` costs seconds and no
// disk, and catches every mismatch that is a property of the DIMENSIONS rather than of the weights.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Testing

@Suite("GLM-5.3 forward at the shipped geometry")
struct Glm5NextRealConfigForwardTests {

    static let bundle = URL(fileURLWithPath: NSHomeDirectory())
        .appending(path: "Library/MLModels/JANGQ-AI/GLM-5.3-Flash-JANG-MTP")

    @Test("a forward at the shipped dimensions produces finite logits")
    func realGeometryForward() throws {
        let configURL = Self.bundle.appending(path: "config.json")
        try #require(
            FileManager.default.fileExists(atPath: configURL.path),
            "the GLM-5.3 bundle is not on this machine")

        var raw = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL))
            as! [String: Any]
        var text = raw["text_config"] as! [String: Any]
        // Shrink DEPTH only. Every width stays exactly as shipped, which is what this test is for;
        // 45 layers of 4096 with random weights would need the same memory the real model does.
        text["num_hidden_layers"] = 4
        text["layer_types"] = ["linear_attention", "deepseek_sparse_attention",
                               "linear_attention", "deepseek_sparse_attention"]
        text["mlp_layer_types"] = ["dense", "dense", "sparse", "sparse"]
        text["num_nextn_predict_layers"] = 0
        text["n_routed_experts"] = 4
        text["num_experts_per_tok"] = 2
        text["first_k_dense_replace"] = 2
        text["indexer_types"] = ["full", "full", "full", "full"]
        // The bundle states its linear-attention schedule twice; the model's own guard refuses a
        // config where the two disagree, so the reduced depth has to be applied to both.
        if var linear = text["linear_attn_config"] as? [String: Any] {
            linear["kda_layers"] = [0, 2]
            linear["full_attn_layers"] = [1, 3]
            text["linear_attn_config"] = linear
        }
        text["vocab_size"] = 1024
        raw["text_config"] = text
        raw.removeValue(forKey: "vision_config")
        raw.removeValue(forKey: "quantization")

        let data = try JSONSerialization.data(withJSONObject: raw)
        let config = try JSONDecoder().decode(Glm5NextConfiguration.self, from: data)

        try MLXMetalTestLock.withLock {
            let model = try Glm5Next(config, requesting: [.text])
            let tokens = MLXArray((0 ..< 8).map { Int32($0 + 1) }).reshaped(1, 8)
            let logits = try model(tokens)
            eval(logits)
            let values = logits.asType(.float32).asArray(Float.self)
            #expect(values.allSatisfy { $0.isFinite }, "shipped geometry produced non-finite logits")
            #expect(Set(values.prefix(64)).count > 1, "logits are constant — a dead path")
        }
    }
    /// A no-MTP bundle would bind cleanly against the SHIPPED key set.
    ///
    /// GLM-5.3 is published in two variants and only the MTP one is on this machine, so "the other
    /// variant loads" cannot be tested by loading it. What CAN be tested is the thing that would
    /// break: with `num_nextn_predict_layers = 0` at the real widths, the model's parameter set must
    /// be exactly the shipped tensors MINUS the MTP layer's — no parameter left unfed, and nothing
    /// declared that such a checkpoint would not carry.
    ///
    /// The shipped MTP layer is index `num_hidden_layers` (45), so its tensors are identifiable by
    /// prefix without hardcoding a list of module names.
    @Test("a no-MTP bundle would bind against the real key set, with nothing left over")
    func noMTPBindsAgainstRealKeys() throws {
        let indexURL = Self.bundle.appending(path: "model.safetensors.index.json")
        try #require(
            FileManager.default.fileExists(atPath: indexURL.path),
            "the GLM-5.3 bundle is not on this machine")

        var raw = try JSONSerialization.jsonObject(with: Data(contentsOf: Self.bundle
            .appending(path: "config.json"))) as! [String: Any]
        var text = raw["text_config"] as! [String: Any]
        let depth = text["num_hidden_layers"] as! Int
        text["num_nextn_predict_layers"] = 0
        raw["text_config"] = text
        raw.removeValue(forKey: "quantization")
        let config = try JSONDecoder().decode(
            Glm5NextConfiguration.self,
            from: try JSONSerialization.data(withJSONObject: raw))

        // Every shipped tensor that is NOT the MTP layer's.
        let index = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL))
            as! [String: Any]
        let shipped = Set((index["weight_map"] as! [String: Any]).keys)
        let mtpPrefix = "model.layers.\(depth)."
        let withoutMTP = shipped.filter { !$0.hasPrefix(mtpPrefix) }
        #expect(
            shipped.count > withoutMTP.count,
            "the shipped bundle has no layer \(depth); this test is checking nothing")

        try MLXMetalTestLock.withLock {
            let model = try Glm5Next(config, requesting: [.vision])
            #expect(model.languageModel.numMTPLayers == 0)
            #expect(model.languageModel.multiTokenPredictionLayer == nil)

            // Every declared parameter must be suppliable from the no-MTP key set, resolved
            // through the family's own key policy — the same two-step the whole-model test uses.
            let declared = Set(model.parameters().flattened().map(\.0))
            #expect(declared.count > 900, "model declared only \(declared.count) parameters")

            var unsuppliable: [String] = []
            for key in declared.sorted() {
                let base = key.hasSuffix(".weight") ? String(key.dropLast(".weight".count)) : key
                let viaPolicy = withoutMTP.contains {
                    Glm5NextCheckpointKeys.bareTensorWeightKey($0) == key
                        || Glm5NextCheckpointKeys.stripHyperConnectionPrefix($0) == key
                }
                if !withoutMTP.contains(key), !withoutMTP.contains("\(base).scales"), !viaPolicy {
                    unsuppliable.append(key)
                }
            }
            #expect(
                unsuppliable.isEmpty,
                "a no-MTP checkpoint could not supply \(unsuppliable.count): \(unsuppliable.prefix(8))")

            // And nothing MTP-shaped is declared, so such a bundle leaves nothing unconsumed.
            let mtpish = declared.filter {
                $0.contains(".enorm") || $0.contains(".hnorm") || $0.contains(".eh_proj")
                    || $0.contains(".shared_head")
            }
            #expect(mtpish.isEmpty, "MTP modules declared with MTP off: \(mtpish.sorted().prefix(4))")
        }
    }

}
