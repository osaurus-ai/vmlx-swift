// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Loads a `z-lab/<model>-DFlash` drafter checkpoint from a local
// directory into a ``DFlashDraftModel`` instance. The directory layout
// matches what `hf download z-lab/<model>-DFlash` produces:
//
//     <dir>/
//         config.json                  ← DFlashDrafterConfiguration + dflash_config
//         model.safetensors            ← weights
//         dflash.py                    ← Python reference (ignored by Swift)
//         utils.py                     ← Python helpers (ignored)
//         assets/, README.md, …        ← ignored

import Foundation
import MLX
import MLXNN

/// Load + configure a ``DFlashDraftModel`` from a local drafter snapshot.
public enum DFlashDrafterLoader {

    /// Load the drafter at `directory`.
    ///
    /// Steps:
    /// 1. Decode `config.json` → ``DFlashDrafterConfiguration``.
    /// 2. Enumerate `*.safetensors` files in the directory, load arrays.
    /// 3. Map Python keys (`layers.0.self_attn.q_proj.weight`) to the
    ///    Swift `Module` tree via `ModuleParameters.unflattened`. Since
    ///    our Swift type names mirror the Python source 1:1 (see
    ///    ``DFlashDraftModel``), the keys land without remapping.
    /// 4. `model.update(parameters:, verify: [.noUnusedKeys])`.
    ///
    /// - Parameter directory: resolved local directory containing
    ///   `config.json` + `model.safetensors`. Supports symlinked paths.
    /// - Returns: configured ``DFlashDraftModel`` with weights loaded.
    public static func load(from directory: URL) throws -> DFlashDraftModel {
        let dir = directory.resolvingSymlinksInPath()

        // 1. Decode config.json
        let configURL = dir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw DFlashDrafterLoadError.missingConfig(configURL)
        }
        let configData = try Data(contentsOf: configURL)
        let config: DFlashDrafterConfiguration
        do {
            config = try JSONDecoder().decode(
                DFlashDrafterConfiguration.self, from: configData)
        } catch {
            throw DFlashDrafterLoadError.malformedConfig(configURL, error)
        }

        // 2. Load all safetensors
        var weights = [String: MLXArray]()
        let enumerator = FileManager.default.enumerator(
            at: dir, includingPropertiesForKeys: nil)
        if let enumerator {
            for case let url as URL in enumerator
            where url.pathExtension == "safetensors" {
                let (w, _) = try loadArraysAndMetadata(url: url)
                for (k, v) in w { weights[k] = v }
            }
        }
        guard !weights.isEmpty else {
            throw DFlashDrafterLoadError.noWeights(dir)
        }

        // 3. Instantiate + apply weights
        let model = DFlashDraftModel(config)

        // Published DFlash drafters are frequently QUANTIZED — the community MLX conversions of
        // z-lab's bf16 release ship 4-bit and 8-bit variants. Those safetensors carry `.scales` and
        // `.biases` beside each packed `.weight`, and an unquantized `Linear` has no parameter slots
        // for them, so the strict `.noUnusedKeys` verify below rejects the whole bundle with
        // `unhandledKeys(keys: ["biases", "scales"])`. Converting the modules to their quantized
        // form first is what makes those bundles loadable at all.
        //
        // The filter is driven by which paths ACTUALLY carry scales in the file rather than by
        // assuming every `Linear` is quantized: real bundles leave selected layers in full
        // precision, and blanket-quantizing those would fail the same verify from the other side
        // (a quantized module whose weights arrive unpacked).
        if let root = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
            let q = root["quantization"] as? [String: Any],
            let groupSize = q["group_size"] as? Int,
            let bits = q["bits"] as? Int
        {
            let mode = (q["mode"] as? String)
                .flatMap(QuantizationMode.init(rawValue:)) ?? .affine
            quantize(
                model: model, groupSize: groupSize, bits: bits, mode: mode,
                filter: { path, _ in weights["\(path).scales"] != nil })
        }

        do {
            let parameters = ModuleParameters.unflattened(weights)
            try model.update(parameters: parameters, verify: [.noUnusedKeys])
        } catch {
            throw DFlashDrafterLoadError.weightUpdateFailed(error)
        }

        // Force materialization of the uploaded parameters. `MLX.eval` is the
        // MLX-Swift framework's graph-materialize call, NOT the dynamic code
        // evaluator. Matches what Load.swift does for target models.
        MLX.eval(model)
        return model
    }

    /// Probe a directory and report whether it looks like a drafter
    /// checkpoint — used by tests to skip when no drafter is on disk.
    public static func looksLikeDrafter(at directory: URL) -> Bool {
        let fm = FileManager.default
        let cfg = directory.appendingPathComponent("config.json")
        guard fm.fileExists(atPath: cfg.path) else { return false }
        guard let data = try? Data(contentsOf: cfg),
            let json = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any]
        else { return false }
        // Both `block_size` and `dflash_config` are DFlash-specific.
        return json["block_size"] != nil && json["dflash_config"] != nil
    }

    /// Resolve a drafter path from environment variable or a default
    /// location. Tests call this to locate an optional drafter checkpoint
    /// without hardcoding paths.
    ///
    /// Looks at:
    /// 1. `DDTREE_DRAFTER_PATH` env var (absolute path).
    /// 2. `$HOME/models/<defaultName>` (common local mirror).
    /// 3. `/tmp/ddtree-downloads/<defaultName>` (where the phase-0
    ///    background download lands).
    ///
    /// Returns `nil` if none of these resolve to a drafter.
    public static func resolvedDrafterPath(
        defaultName: String
    ) -> URL? {
        let fm = FileManager.default
        if let env = ProcessInfo.processInfo.environment["DDTREE_DRAFTER_PATH"] {
            let url = URL(fileURLWithPath: env)
            if looksLikeDrafter(at: url) { return url }
        }
        let home = fm.homeDirectoryForCurrentUser
        let candidates: [URL] = [
            home.appendingPathComponent("models").appendingPathComponent(defaultName),
            URL(fileURLWithPath: "/tmp/ddtree-downloads").appendingPathComponent(defaultName),
        ]
        for c in candidates where looksLikeDrafter(at: c) {
            return c
        }
        return nil
    }
}

/// Errors produced by ``DFlashDrafterLoader``.
public enum DFlashDrafterLoadError: Error, LocalizedError {
    case missingConfig(URL)
    case malformedConfig(URL, Error)
    case noWeights(URL)
    case weightUpdateFailed(Error)

    public var errorDescription: String? {
        switch self {
        case .missingConfig(let url):
            return "DFlash drafter missing config.json at \(url.path)"
        case .malformedConfig(let url, let err):
            return "DFlash drafter config.json malformed at \(url.path): \(err)"
        case .noWeights(let url):
            return "DFlash drafter has no .safetensors files in \(url.path)"
        case .weightUpdateFailed(let err):
            return "DFlash drafter weight update failed: \(err)"
        }
    }
}
