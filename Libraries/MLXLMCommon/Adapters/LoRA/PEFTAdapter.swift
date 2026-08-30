// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Reading HuggingFace PEFT LoRA adapters, which is what essentially every published adapter is.
//
// This is the adapter-side equivalent of a model's `sanitize(weights:)`. Base models load from HF
// unchanged only because 85 model files each carry a hand-written translator from checkpoint
// naming to module naming; the adapter path had none, so PEFT adapters failed at the first
// missing key. The gap is not specific to this port — `mlx-lm` has no mention of PEFT either, and
// its loader validates leaf names against exactly `{lora_a, lora_b, m}`.

import Foundation
import MLX
import MLXNN

/// A PEFT `adapter_config.json`, in the fields that matter for loading one.
struct PEFTConfiguration: Decodable {
    let peftType: String
    let r: Int
    let loraAlpha: Float
    let targetModules: [String]?
    let useDora: Bool?

    enum CodingKeys: String, CodingKey {
        case peftType = "peft_type"
        case r
        case loraAlpha = "lora_alpha"
        case targetModules = "target_modules"
        case useDora = "use_dora"
    }
}

enum PEFTAdapter {
    /// PEFT names its weights file differently from MLX, so both spellings are accepted.
    static let weightsFilenames = ["adapter_model.safetensors", "adapters.safetensors"]

    /// Is this directory a PEFT adapter rather than an MLX one?
    ///
    /// Keyed on `peft_type` being present while `fine_tune_type` is absent, so an MLX adapter is
    /// never mistaken for a PEFT one even if a converter later writes both.
    static func detect(configuration: Data) -> PEFTConfiguration? {
        guard let obj = try? JSONSerialization.jsonObject(with: configuration) as? [String: Any],
            obj["peft_type"] != nil, obj["fine_tune_type"] == nil
        else { return nil }
        return try? JSONDecoder().decode(PEFTConfiguration.self, from: configuration)
    }

    /// Translate PEFT weights into the naming and ORIENTATION MLX expects.
    ///
    /// Three things change, and the last is the one that fails silently if it is wrong:
    ///
    ///   1. the `base_model.model.` wrapper PEFT adds is stripped;
    ///   2. `…lora_A.weight` / `…lora_B.weight` become `…lora_a` / `…lora_b`;
    ///   3. BOTH factors are TRANSPOSED.
    ///
    /// On (3): `LoRALinear` stores `loraA` as `[in, rank]` and `loraB` as `[rank, out]`, and its
    /// forward is `y + scale * (x @ loraA @ loraB)`. PEFT stores the transpose of each — verified
    /// against this adapter rather than assumed, where `out_proj` gives A=(32, 6144) and
    /// B=(5120, 32) against a 5120-wide model. Loading those un-transposed would either throw on
    /// a shape mismatch or, where dimensions happen to agree, quietly compute a different function
    /// — which is why the round trip is checked numerically in the tests rather than by loading
    /// successfully.
    static func translate(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (key, value) in weights {
            var k = key
            if k.hasPrefix("base_model.model.") {
                k = String(k.dropFirst("base_model.model.".count))
            }
            if k.hasSuffix(".lora_A.weight") {
                k = String(k.dropLast(".lora_A.weight".count)) + ".lora_a"
            } else if k.hasSuffix(".lora_B.weight") {
                k = String(k.dropLast(".lora_B.weight".count)) + ".lora_b"
            } else {
                continue                       // biases, magnitudes, anything else: not ours
            }
            out[k] = value.ndim == 2 ? value.transposed(1, 0) : value
        }
        return out
    }

    /// The MLX configuration a PEFT one implies.
    ///
    /// `scale` is `lora_alpha / r`, which is PEFT's own convention and NOT interchangeable with
    /// MLX's default of 10–20. For this adapter alpha and r are both 32, so the scale is 1.0 —
    /// taking the MLX default instead would apply the update at ten to twenty times its intended
    /// strength, and the model would still load and still generate.
    ///
    /// `numLayers` is counted from the weights rather than declared, because PEFT does not record
    /// it: it is the number of distinct `layers.N` indices the adapter actually touches.
    static func configuration(
        from peft: PEFTConfiguration, weights: [String: MLXArray]
    ) -> LoRAConfiguration {
        var layerIndices = Set<Int>()
        for key in weights.keys {
            let parts = key.split(separator: ".")
            for (i, p) in parts.enumerated() where p == "layers" && i + 1 < parts.count {
                if let n = Int(parts[i + 1]) { layerIndices.insert(n) }
            }
        }
        // `keys` is carried through verbatim; `expandTargetKeys` is what reconciles PEFT's two
        // spellings of target_modules with this model's actual module paths.
        return LoRAConfiguration(
            numLayers: max(layerIndices.count, 1),
            fineTuneType: (peft.useDora ?? false) ? .dora : .lora,
            loraParameters: .init(
                rank: peft.r,
                scale: peft.loraAlpha / Float(peft.r),
                keys: peft.targetModules))
    }

    /// Expand PEFT's LEAF target names into the child keys this model actually uses.
    ///
    /// `target_modules: ["o_proj", "out_proj", "down_proj"]` names leaves. `replaceLayers` matches
    /// `namedModules()` keys EXACTLY, and in a model where the projection sits under a submodule
    /// that key is `linear_attn.out_proj`, not `out_proj` — so nothing matches and no LoRA slots
    /// are created. The failure is silent in the worst way: the adapter then reports every one of
    /// its parameters as unmatched, which reads like a bad adapter rather than an unexpanded key.
    ///
    /// Resolving against the live layers keeps this family-agnostic, for the same reason `rekey`
    /// does.
    static func expandTargetKeys(_ targets: [String], in layers: ArraySlice<Module>) -> [String] {
        // PEFT writes target_modules EITHER as leaf names ("o_proj") or as full checkpoint paths
        // ("model.language_model.layers.16.mlp.down_proj") — both are valid and both appear in
        // the wild, in two adapters from the same author. Reduce to leaves so one rule covers both;
        // the layer selection is carried by `numLayers` and by which weights the adapter ships,
        // not by these strings.
        let wanted = Set(targets.map { $0.split(separator: ".").last.map(String.init) ?? $0 })
        var expanded = Set<String>()
        // EVERY layer, not just the first. Qwen 3.8 is hybrid: early layers carry
        // `linear_attn.out_proj` and later ones `self_attn.o_proj`, and this adapter targets both.
        // Sampling layer 0 alone found 112 of 144 slots and left 32 parameters unmatched — a
        // failure that reads like a bad adapter rather than an unrepresentative sample.
        for layer in layers {
            for (key, _) in layer.namedModules() {
                let leaf = key.split(separator: ".").last.map(String.init) ?? key
                if wanted.contains(leaf) { expanded.insert(key) }
            }
        }
        return expanded.isEmpty ? targets : expanded.sorted()
    }

    /// Re-key adapter parameters onto the paths the LIVE MODULE TREE actually uses.
    ///
    /// PEFT records checkpoint paths, which are not module paths. Qwen 3.5 is the plain case: the
    /// adapter says `model.language_model.layers.0.…` while the module tree says
    /// `language_model.model.layers.0.…` — the same swap a base model's `sanitize` performs, and
    /// the reason `update(parameters:)` reports `Unhandled keys ["model"]`.
    ///
    /// Matching by SUFFIX against the model rather than hardcoding per family is deliberate. The
    /// 85 `sanitize` implementations exist because every family spells its checkpoint differently;
    /// requiring an 86th for adapters would repeat that cost and leave the next family broken
    /// again. A LoRA target is identified by its tail — `layers.N.<module>.<proj>.lora_a` — and
    /// that tail is stable across the prefixes families disagree about.
    ///
    /// An unmatched or ambiguous key is an ERROR, never a silent drop: an adapter that loads with
    /// half its factors missing still generates fluent text, and nothing downstream would say so.
    static func rekey(
        _ parameters: [String: MLXArray], toModelKeys modelKeys: [String]
    ) throws -> [String: MLXArray] {
        func tail(_ key: String) -> String {
            let parts = key.split(separator: ".")
            if let i = parts.lastIndex(where: { $0 == "layers" }), i + 1 < parts.count {
                return parts[i...].joined(separator: ".")
            }
            return key
        }
        var byTail: [String: [String]] = [:]
        for k in modelKeys { byTail[tail(k), default: []].append(k) }

        var out: [String: MLXArray] = [:]
        var unmatched: [String] = []
        for (key, value) in parameters {
            if modelKeys.contains(key) {            // already correct — MLX-native adapter
                out[key] = value
                continue
            }
            let candidates = byTail[tail(key)] ?? []
            guard candidates.count == 1 else {
                unmatched.append(candidates.isEmpty ? key : "\(key) [ambiguous: \(candidates.count)]")
                continue
            }
            out[candidates[0]] = value
        }
        guard unmatched.isEmpty else {
            // Report BOTH sides. "does not correspond" alone cannot distinguish a wrong prefix
            // from a target module the model never LoRA-ised, and those need opposite fixes.
            let offered = modelKeys.filter { $0.hasSuffix(".lora_a") }.sorted()
            throw ModelAdapterError.unsupportedAdapterType(
                "\(unmatched.count) adapter parameter(s) match no module. "
                + "adapter wants: \(unmatched.sorted().prefix(2).joined(separator: ", ")). "
                + "model offers \(offered.count) lora slot(s)"
                + (offered.isEmpty
                    ? " — none, so the target modules were never replaced (check target_modules "
                      + "against this model's layer names)"
                    : ": \(offered.prefix(2).joined(separator: ", "))"))
        }
        return out
    }
}

/// Test seam.
///
/// `PEFTAdapter` is internal because nothing outside the loader should call it, but the
/// orientation of the translated factors is precisely the thing that must be checked numerically
/// rather than by a successful load. This exposes the three pure functions and nothing else.
public enum PEFTAdapterTestBridge {
    public static func translate(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        PEFTAdapter.translate(weights: weights)
    }

    public static func detect(_ configuration: Data) -> PEFTConfigurationBox? {
        PEFTAdapter.detect(configuration: configuration).map(PEFTConfigurationBox.init)
    }

    public static func configuration(
        _ box: PEFTConfigurationBox, _ weights: [String: MLXArray]
    ) -> LoRAConfiguration {
        PEFTAdapter.configuration(from: box.value, weights: weights)
    }

    public static func expandTargetKeys(_ targets: [String], in layers: [Module]) -> [String] {
        PEFTAdapter.expandTargetKeys(targets, in: layers[...])
    }
}

public struct PEFTConfigurationBox {
    let value: PEFTConfiguration
    init(_ value: PEFTConfiguration) { self.value = value }
}
