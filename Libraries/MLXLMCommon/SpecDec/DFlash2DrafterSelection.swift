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
            let topK = dflash["selector_top_k"] as? Int, topK > 0,
            let rank = dflash["selector_rank"] as? Int, rank > 0,
            let convKernel = dflash["conv_kernel_size"] as? Int, convKernel > 0,
            let vocabularySize = root["vocab_size"] as? Int,
            let targetLayerIDs = dflash["target_layer_ids"] as? [Int]
        else { return nil }

        let fm = FileManager.default
        var bytes: Int64 = 0
        if let names = try? fm.contentsOfDirectory(atPath: directory.path) {
            for name in names where name.hasSuffix(".safetensors") {
                let attrs = try? fm.attributesOfItem(
                    atPath: directory.appendingPathComponent(name).path)
                bytes += (attrs?[.size] as? NSNumber)?.int64Value ?? 0
            }
        }

        return VMLXDFlash2DrafterInfo(
            path: directory.path,
            blockSize: (dflash["block_size"] as? Int) ?? (root["block_size"] as? Int) ?? 8,
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
            // No config to check against. Let the runtime's own vocabulary
            // check be the gate rather than guessing here.
            return nil
        }
        let text = (root["text_config"] as? [String: Any]) ?? root
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
