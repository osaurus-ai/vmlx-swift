//
//  Qwen35MoE.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/25.
//
//  Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen3_5_moe
//

import MLX
import MLXLMCommon

public final class Qwen35MoE: Qwen35 {
    public override func sanitize(
        weights: [String: MLXArray],
        metadata: [String: String]
    ) -> [String: MLXArray] {
        super.sanitize(weights: sanitizeMoEWeights(weights), metadata: metadata)
    }

    public override func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        super.sanitize(weights: sanitizeMoEWeights(weights))
    }

    private func sanitizeMoEWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var remapped = [String: MLXArray]()
        remapped.reserveCapacity(weights.count)
        for (key, value) in weights {
            if !key.hasSuffix(".tq_bits") {
                remapped[key] = value
            }
        }

        for layer in 0 ..< config.textConfiguration.hiddenLayers {
            let prefixes = [
                "model.language_model.layers.\(layer).mlp",
                "language_model.model.layers.\(layer).mlp",
            ]

            for prefix in prefixes {
                stackTurboQuantExpertsIfNeeded(prefix: prefix, weights: &remapped)

                let gateUpKey = "\(prefix).experts.gate_up_proj"
                if let gateUp = remapped.removeValue(forKey: gateUpKey) {
                    let mid = gateUp.dim(-2) / 2
                    remapped["\(prefix).switch_mlp.gate_proj.weight"] =
                        gateUp[
                            .ellipsis, ..<mid, 0...]
                    remapped["\(prefix).switch_mlp.up_proj.weight"] =
                        gateUp[
                            .ellipsis, mid..., 0...]

                    let downProjKey = "\(prefix).experts.down_proj"
                    if let downProj = remapped.removeValue(forKey: downProjKey) {
                        remapped["\(prefix).switch_mlp.down_proj.weight"] = downProj
                    }
                }
            }
        }

        for layer in 0 ..< config.textConfiguration.mtpNumHiddenLayers {
            for prefix in [
                "mtp.layers.\(layer).mlp",
                "model.mtp_layers.\(layer).mlp",
            ] {
                stackAffineExpertsIfNeeded(prefix: prefix, weights: &remapped)
            }
        }

        return remapped
    }

    /// Some Qwen3.5-MoE checkpoints keep the base routed experts pre-stacked
    /// but preserve the MTP head in Hugging Face's per-expert affine layout:
    /// `mtp.layers.L.mlp.experts.E.{gate,up,down}_proj.{weight,scales,biases}`.
    /// The runtime module is a `SwitchGLU`, so those tensors must become one
    /// materialized expert bank per projection. Materializing at load avoids a
    /// lazy stack retaining every source mmap through the first decode.
    private func stackAffineExpertsIfNeeded(
        prefix: String,
        weights: inout [String: MLXArray]
    ) {
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            for suffix in ["weight", "scales", "biases"] {
                let target = "\(prefix).switch_mlp.\(projection).\(suffix)"
                if weights[target] != nil { continue }

                let sourceKeys = (0 ..< config.textConfiguration.numExperts).map {
                    "\(prefix).experts.\($0).\(projection).\(suffix)"
                }
                guard sourceKeys.first.flatMap({ weights[$0] }) != nil else { continue }
                let tensors = sourceKeys.compactMap { weights[$0] }
                guard tensors.count == config.textConfiguration.numExperts else {
                    // Leave the incomplete source layout intact so strict model
                    // update fails closed and reports the unsupported payload.
                    continue
                }
                for key in sourceKeys {
                    weights.removeValue(forKey: key)
                }
                weights[target] = loadTimeMaterializedStacked(tensors)
            }
        }
    }

    private func stackTurboQuantExpertsIfNeeded(
        prefix: String,
        weights: inout [String: MLXArray]
    ) {
        let renames: [(String, String)] = [
            ("w1", "gate_proj"),
            ("w2", "down_proj"),
            ("w3", "up_proj"),
        ]

        for (sourceName, targetName) in renames {
            for tensorKind in ["tq_packed", "tq_norms"] {
                let targetKey = "\(prefix).switch_mlp.\(targetName).\(tensorKind)"
                if weights[targetKey] != nil { continue }

                let firstKey = "\(prefix).experts.0.\(sourceName).\(tensorKind)"
                guard weights[firstKey] != nil else { continue }

                if JANGTQStreamingExperts.isEnabled {
                    for expert in 0 ..< config.textConfiguration.numExperts {
                        weights.removeValue(
                            forKey: "\(prefix).experts.\(expert).\(sourceName).\(tensorKind)")
                    }
                    continue
                }

                let sourceKeys = (0 ..< config.textConfiguration.numExperts).map {
                    "\(prefix).experts.\($0).\(sourceName).\(tensorKind)"
                }
                let tensors = sourceKeys.compactMap { weights[$0] }
                if tensors.count == config.textConfiguration.numExperts {
                    for key in sourceKeys {
                        weights.removeValue(forKey: key)
                    }
                    weights[targetKey] = loadTimeMaterializedStacked(tensors)
                }
            }
        }
    }
}
