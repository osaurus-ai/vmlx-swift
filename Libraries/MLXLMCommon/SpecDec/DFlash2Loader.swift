// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Loads a DFlash 2 drafter checkpoint (`z-lab/<model>-DFlash2`) from a
// local directory. Port of `load_draft` in z-lab/dflash `model_mlx.py`.

import Foundation
import MLX
import MLXNN

public enum DFlash2Loader {

    /// Load the drafter at `directory`.
    ///
    /// The one non-obvious step is the codebook rename: the checkpoint
    /// stores `candidate_selector.predecessor_codebook` as a bare tensor,
    /// but it is consumed as an embedding table, so it has to land at
    /// `…predecessor_codebook.weight`. The reference does the same
    /// `weights.pop` dance for the same reason.
    public static func load(from directory: URL) throws -> DFlash2DraftModel {
        let dir = directory.resolvingSymlinksInPath()

        let configURL = dir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw DFlash2LoadError.missingConfig(configURL)
        }
        let configData = try Data(contentsOf: configURL)
        guard let root = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw DFlash2LoadError.malformedConfig(configURL)
        }
        let config = try DFlash2Configuration(json: root)
        guard config.isDFlash2 else {
            throw DFlash2LoadError.notDFlash2(dir)
        }

        var weights = [String: MLXArray]()
        if let enumerator = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: nil)
        {
            for case let url as URL in enumerator where url.pathExtension == "safetensors" {
                let (w, _) = try loadArraysAndMetadata(url: url)
                for (k, v) in w { weights[k] = v }
            }
        }
        guard !weights.isEmpty else {
            throw DFlash2LoadError.noWeights(dir)
        }

        for name in ["predecessor_codebook", "successor_codebook"] {
            let key = "candidate_selector.\(name)"
            if let tensor = weights.removeValue(forKey: key) {
                weights["\(key).weight"] = tensor
            }
        }

        let model = DFlash2DraftModel(config)

        // Community MLX re-quantisations of the bf16 release carry
        // `.scales`/`.biases` beside each packed `.weight`; an unquantized
        // `Linear` has no slot for those and the strict verify below would
        // reject the whole bundle. Quantize exactly the modules the file
        // actually packed rather than assuming it is uniform.
        if let q = root["quantization"] as? [String: Any],
            let groupSize = q["group_size"] as? Int,
            let bits = q["bits"] as? Int
        {
            let mode = (q["mode"] as? String).flatMap(QuantizationMode.init(rawValue:)) ?? .affine
            quantize(
                model: model, groupSize: groupSize, bits: bits, mode: mode,
                filter: { path, _ in weights["\(path).scales"] != nil })
        }

        do {
            try model.update(
                parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
        } catch {
            throw DFlash2LoadError.weightUpdateFailed(error)
        }

        // The bf16 release runs quantized in the reference runtime: the
        // oMLX recipe's "Draft quantization enabled" is `nn.quantize(draft,
        // group_size=64, bits=4)`, and the 60+ t/s contract was measured
        // with it (acceptance held ~88%). Quantize AFTER the weights land
        // so the bf16 tensors load into the still-bf16 modules first.
        // `VMLX_DFLASH2_NO_DRAFT_QUANT=1` keeps a bf16 A/B arm for benching.
        let alreadyQuantized = root["quantization"] != nil
        let quantOptOut =
            ProcessInfo.processInfo.environment["VMLX_DFLASH2_NO_DRAFT_QUANT"] == "1"
        if !alreadyQuantized && !quantOptOut {
            // Same predicate as Python's nn.quantize default: only modules
            // whose weight's last dim divides the group size.
            quantize(
                model: model, groupSize: 64, bits: 4,
                filter: { _, module in
                    if let linear = module as? Linear {
                        return linear.weight.dim(-1) % 64 == 0
                    }
                    if let embedding = module as? Embedding {
                        return embedding.weight.dim(-1) % 64 == 0
                    }
                    return true
                })
        }

        MLX.eval(model)
        return model
    }

    /// Cheap probe: does this directory hold a DFlash 2 drafter?
    ///
    /// Keyed on the selector and conv fields rather than the
    /// `architectures` string. DFlash 1 checkpoints carry `dflash_config`
    /// too, so the key alone is not discriminating — and the architecture
    /// name is just a label, which makes it exactly the kind of evidence
    /// that is right until it isn't. What this runtime needs is the
    /// selector and the two-tap conv; ask for those.
    public static func looksLikeDFlash2Drafter(at directory: URL) -> Bool {
        let cfg = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: cfg.path),
            let data = try? Data(contentsOf: cfg),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let dflash = json["dflash_config"] as? [String: Any]
        else { return false }
        let hasSelector = ((dflash["selector_top_k"] as? Int) ?? 0) > 0
            && ((dflash["selector_rank"] as? Int) ?? 0) > 0
        let hasConv = ((dflash["conv_kernel_size"] as? Int) ?? 0) > 0
            && ((dflash["conv_group_size"] as? Int) ?? 0) > 0
        return hasSelector && hasConv
    }

    /// Human-readable reason a directory cannot be used, for the picker
    /// in the host app. `nil` means it is usable.
    public static func rejectionReason(at directory: URL) -> String? {
        let cfg = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: cfg.path) else {
            return "No config.json in \(directory.lastPathComponent)"
        }
        guard let data = try? Data(contentsOf: cfg),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return "config.json is not readable JSON"
        }
        guard json["dflash_config"] != nil else {
            return "Not a DFlash drafter — config.json has no dflash_config"
        }
        guard looksLikeDFlash2Drafter(at: directory) else {
            return "This is a DFlash 1 drafter; DFlash 2 is required"
        }
        let hasWeights =
            (try? FileManager.default.contentsOfDirectory(atPath: directory.path))?
            .contains { $0.hasSuffix(".safetensors") } ?? false
        guard hasWeights else {
            return "No .safetensors weights in \(directory.lastPathComponent)"
        }
        return nil
    }
}

/// Errors produced while loading or validating a DFlash 2 drafter.
public enum DFlash2LoadError: Error, LocalizedError {
    case missingConfig(URL)
    case malformedConfig(URL)
    case missingConfigKey(String)
    case layerTypeCountMismatch(Int, Int)
    case unsupportedLayerTypes([String])
    case notDFlash2(URL)
    case noWeights(URL)
    case weightUpdateFailed(Error)
    case targetMismatch(String)

    public var errorDescription: String? {
        switch self {
        case .missingConfig(let url):
            return "DFlash 2 drafter missing config.json at \(url.path)"
        case .malformedConfig(let url):
            return "DFlash 2 drafter config.json malformed at \(url.path)"
        case .missingConfigKey(let key):
            return "DFlash 2 drafter config.json missing required key '\(key)'"
        case .layerTypeCountMismatch(let got, let expected):
            return
                "DFlash 2 drafter layer_types has \(got) entries but num_hidden_layers is \(expected)"
        case .unsupportedLayerTypes(let types):
            return "DFlash 2 drafter has unsupported layer_types: \(types.joined(separator: ", "))"
        case .notDFlash2(let url):
            return
                "\(url.lastPathComponent) is not a DFlash 2 drafter (no selector/conv config) — DFlash 1 checkpoints are not supported on this path"
        case .noWeights(let url):
            return "DFlash 2 drafter has no .safetensors files in \(url.path)"
        case .weightUpdateFailed(let err):
            return "DFlash 2 drafter weight update failed: \(err)"
        case .targetMismatch(let detail):
            return "DFlash 2 drafter does not match the loaded model: \(detail)"
        }
    }
}
