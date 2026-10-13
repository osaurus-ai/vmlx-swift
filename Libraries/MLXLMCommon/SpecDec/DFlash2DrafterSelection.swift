// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Host-facing resolution of a user-selected DFlash 2 drafter folder.
//
// A drafter is trained against ONE target. Pointing the runtime at
// `Qwen3.8-27B-DFlash2` and then loading Gemma must not engage
// speculation — the drafter borrows the target's embedding and LM head,
// so a mismatch would not error, it would emit fluent tokens from a
// different vocabulary that the target then rejects at ~0% acceptance.
//
// This file gives the host everything it needs to decide BEFORE a
// request: read the drafter's config once, compare it against the loaded
// bundle's config, and report a reason the UI can show. The runtime's own
// checks stay as a backstop, but a user should learn their drafter does
// not fit this model from the settings pane, not from a failed request.

import Foundation

/// What the host knows about a selected drafter folder without loading
/// 3.8 GB of weights.
public struct VMLXDFlash2DrafterInfo: Codable, Sendable, Equatable {
    /// Folder the user picked.
    public let path: String
    /// `dflash_config.block_size` — the trained block, one position of
    /// which is the anchor, so `8` drafts seven tokens per step.
    public let blockSize: Int
    /// Vocabulary the drafter was trained against. Must equal the
    /// target's.
    public let vocabularySize: Int
    /// Number of layers the drafter expects the target to have.
    public let targetLayerCount: Int
    /// Which target layers it reads hidden states from.
    public let targetLayerIDs: [Int]
    /// Bytes on disk, for the settings pane.
    public let weightBytes: Int64

    public init(
        path: String, blockSize: Int, vocabularySize: Int, targetLayerCount: Int,
        targetLayerIDs: [Int], weightBytes: Int64
    ) {
        self.path = path
        self.blockSize = blockSize
        self.vocabularySize = vocabularySize
        self.targetLayerCount = targetLayerCount
        self.targetLayerIDs = targetLayerIDs
        self.weightBytes = weightBytes
    }

    /// Read a drafter folder's metadata. `nil` when the folder is not a
    /// DFlash 2 drafter.
    public static func read(at directory: URL) -> VMLXDFlash2DrafterInfo? {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let dflash = root["dflash_config"] as? [String: Any],
            // Same discriminator as `DFlash2Loader.looksLikeDFlash2Drafter`:
            // the selector and the two-tap conv are what this runtime
            // needs, and a DFlash 1 checkpoint has neither despite
            // carrying `dflash_config`.
            // ... or a DSpark drafter: the DFlash 1 backbone plus the Markov head (`markov_rank`, at the root or nested).
            ((dflash["selector_top_k"] as? Int ?? 0) > 0 && (dflash["selector_rank"] as? Int ?? 0) > 0
                && (dflash["conv_kernel_size"] as? Int ?? 0) > 0)
                || ((dflash["markov_rank"] as? Int ?? root["markov_rank"] as? Int ?? 0) > 0),
            let vocabularySize = root["vocab_size"] as? Int,
            let targetLayerIDs = dflash["target_layer_ids"] as? [Int],
            !targetLayerIDs.isEmpty, targetLayerIDs.allSatisfy({ $0 >= 0 }),
            Set(targetLayerIDs).count == targetLayerIDs.count
        else { return nil }

        // Match the loader's recursive layout, but read headers only. A config-only,
        // missing-shard, malformed-header or truncated download is not a usable drafter.
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: directory, includingPropertiesForKeys: nil) else {
            return nil
        }
        var tensorFiles: [String: Set<String>] = [:]
        var bytes: Int64 = 0
        for case let file as URL in enumerator where file.pathExtension == "safetensors" {
            guard let keys = safetensorsTensorKeys(file), !keys.isEmpty,
                safetensorsMissingByteCount(file) == nil,
                let attrs = try? fm.attributesOfItem(atPath: file.resolvingSymlinksInPath().path),
                let size = attrs[.size] as? NSNumber, size.int64Value > 8
            else { return nil }
            let relative = String(file.path.dropFirst(directory.path.count + 1))
            tensorFiles[relative] = Set(keys)
            bytes += size.int64Value
        }
        guard !tensorFiles.isEmpty else { return nil }
        let index = directory.appendingPathComponent("model.safetensors.index.json")
        if fm.fileExists(atPath: index.path) {
            guard let data = try? Data(contentsOf: index),
                let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let weights = object["weight_map"] as? [String: String], !weights.isEmpty,
                weights.allSatisfy({ tensorFiles[$0.value]?.contains($0.key) == true })
            else { return nil }
        }

        guard DFlash2ArtifactMetadata.rejectionReason(at: directory) == nil else { return nil }
        return VMLXDFlash2DrafterInfo(
            path: directory.path,
            // Verify width: a DSpark `block_size` counts drafts (the anchor row drafts too), so +1.
            blockSize: ((dflash["block_size"] as? Int) ?? (root["block_size"] as? Int) ?? 8)
                + (((dflash["markov_rank"] as? Int ?? root["markov_rank"] as? Int ?? 0) > 0) ? 1 : 0),
            vocabularySize: vocabularySize,
            targetLayerCount: (root["num_target_layers"] as? Int)
                ?? ((targetLayerIDs.max() ?? 0) + 1),
            targetLayerIDs: targetLayerIDs,
            weightBytes: bytes)
    }

    /// Why this drafter cannot serve the bundle described by
    /// `configData`, or `nil` when it can.
    ///
    /// Deliberately advisory in shape: it returns a sentence, and the
    /// caller decides what to do with it. Nothing here refuses to load a
    /// model or blocks a request — the worst case is that speculation
    /// stays off and decoding runs exactly as it does today.
    public func mismatchReason(configData: Data?) -> String? {
        guard let configData,
            let root = try? JSONSerialization.jsonObject(with: configData) as? [String: Any]
        else {
            return "Target configuration is unavailable; drafter compatibility cannot be verified."
        }
        let text = (root["text_config"] as? [String: Any]) ?? root
        guard ((text["vocab_size"] as? Int) ?? (root["vocab_size"] as? Int)) != nil,
            ((text["num_hidden_layers"] as? Int) ?? (root["num_hidden_layers"] as? Int)) != nil,
            ((text["hidden_size"] as? Int) ?? (root["hidden_size"] as? Int)) != nil,
            drafterHiddenSize != nil
        else {
            return "Target or drafter dimensions are missing; compatibility cannot be verified."
        }
        if let vocabulary = (text["vocab_size"] as? Int) ?? (root["vocab_size"] as? Int),
            vocabulary != vocabularySize
        {
            return
                "Drafter was trained for a \(vocabularySize)-token vocabulary; this model has \(vocabulary)."
        }
        if let layers = (text["num_hidden_layers"] as? Int) ?? (root["num_hidden_layers"] as? Int),
            let deepest = targetLayerIDs.max(), deepest >= layers
        {
            return
                "Drafter reads layer \(deepest) of its target; this model has \(layers) layers."
        }
        if let layers = (text["num_hidden_layers"] as? Int) ?? (root["num_hidden_layers"] as? Int),
            layers != targetLayerCount
        {
            return "Drafter expects \(targetLayerCount) target layers; this model has \(layers)."
        }
        if let hidden = (text["hidden_size"] as? Int) ?? (root["hidden_size"] as? Int),
            let drafterHidden = drafterHiddenSize, hidden != drafterHidden
        {
            return
                "Drafter expects a hidden size of \(drafterHidden); this model uses \(hidden)."
        }
        return nil
    }

    private var drafterHiddenSize: Int? {
        guard
            let data = try? Data(
                contentsOf: URL(fileURLWithPath: path).appendingPathComponent("config.json")),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return root["hidden_size"] as? Int
    }

    /// Short line for the settings pane.
    public var summary: String {
        let gigabytes = Double(weightBytes) / 1_073_741_824
        return String(
            format: "block %d · %d tokens drafted per step · %.1f GB",
            blockSize, Swift.max(blockSize - 1, 0), gigabytes)
    }
}
