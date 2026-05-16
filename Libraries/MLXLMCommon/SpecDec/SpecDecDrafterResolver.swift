// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Resolves a `DraftStrategy` to a loaded drafter + the per-strategy
// parameters needed by `SpecDecRuntimeLinear.run` / `SpecDecRuntimeDDTree.run`.
// Caches loaded drafters per-path so the same drafter checkpoint is
// only loaded once across multiple generation calls.

import Foundation
import MLX

/// Actor-backed drafter cache. One instance per process is typical;
/// osaurus can hold one on the runtime container.
public actor SpecDecDrafterResolver {

    /// Package-wide shared resolver used by the `Evaluate.generate`
    /// dispatch when a caller sets
    /// ``GenerateParameters/draftStrategy``. Applications that want a
    /// private cache lifetime (e.g. osaurus per-container) can
    /// instantiate their own ``SpecDecDrafterResolver`` with `init()`.
    public static let shared = SpecDecDrafterResolver()

    /// Thread-safe by-path drafter cache. Keys are `resolvingSymlinksInPath`
    /// of each drafter directory so aliased paths hit.
    private var cache: [String: DFlashDraftModel] = [:]

    public init() {}

    /// Load a drafter from a local HF snapshot directory, caching the
    /// result so repeated calls with the same path skip safetensors load.
    public func loadDrafter(at path: URL) throws -> DFlashDraftModel {
        let key = path.resolvingSymlinksInPath().path
        if let hit = cache[key] {
            return hit
        }
        let model = try DFlashDrafterLoader.load(from: path)
        cache[key] = model
        return model
    }

    /// Drop a cached drafter — useful when the caller knows a model has
    /// been unloaded / replaced on disk.
    public func evict(path: URL) {
        cache.removeValue(
            forKey: path.resolvingSymlinksInPath().path)
    }

    /// Drop every cached drafter.
    public func evictAll() {
        cache.removeAll()
    }

    /// Convenience: given a ``DraftStrategy``, return the configured
    /// drafter + the target-layer IDs + mask-token-ID the runtime needs.
    /// Throws for strategies that don't use block-diffusion drafters
    /// (`.none` / `.autoregressive`).
    public func resolve(
        strategy: DraftStrategy
    ) throws -> ResolvedDrafter {
        switch strategy {
        case .none, .autoregressive, .nativeMTP:
            throw SpecDecError.notImplemented(
                "resolve: \(strategy.kindName) is not a block-diffusion strategy")
        case .dflash(let path, _):
            let model = try loadDrafter(at: path)
            return ResolvedDrafter(
                model: model,
                targetBlockIDs: zeroBasedTargetLayerIDs(model.config),
                maskTokenID: Int32(model.config.dflashConfig.maskTokenId))
        case .ddtree(let path, _, _):
            let model = try loadDrafter(at: path)
            return ResolvedDrafter(
                model: model,
                targetBlockIDs: zeroBasedTargetLayerIDs(model.config),
                maskTokenID: Int32(model.config.dflashConfig.maskTokenId))
        }
    }

    /// Convert HF's 1-based `target_layer_ids` (which count the
    /// HF `target_layer_ids` ARE 0-based indices into
    /// `target.model.layers` per z-lab/dflash `_patch_model`. Use as-is.
    private func zeroBasedTargetLayerIDs(
        _ config: DFlashDrafterConfiguration
    ) -> [Int] {
        config.dflashConfig.targetLayerIds
    }
}

/// Bundle of values a ``DraftStrategy`` resolves to for the SpecDec runtime.
public struct ResolvedDrafter: @unchecked Sendable {
    public let model: DFlashDraftModel
    public let targetBlockIDs: [Int]
    public let maskTokenID: Int32

    public init(
        model: DFlashDraftModel,
        targetBlockIDs: [Int],
        maskTokenID: Int32
    ) {
        self.model = model
        self.targetBlockIDs = targetBlockIDs
        self.maskTokenID = maskTokenID
    }
}
