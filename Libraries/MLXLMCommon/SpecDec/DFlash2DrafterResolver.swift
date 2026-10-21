// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Process-wide cache of loaded DFlash 2 drafters, keyed by directory.
// A drafter is ~3.8 GB for the 27B checkpoint; reloading it per request
// would cost more than the speculation saves.
//
// Lock-backed rather than an actor because `Evaluate.generate` is a
// synchronous throwing function — the drafter has to be in hand before
// the iterator is constructed, and hopping to an actor there would mean
// restructuring every dispatch site for one cache lookup.

import Foundation
import MLX

#if canImport(os)
    import os
#endif

public final class DFlash2DrafterResolver: @unchecked Sendable {

    public static let shared = DFlash2DrafterResolver()

    private final class Load {
        var result: Result<DFlash2DraftModel, Error>?
    }

    // The condition protects ownership only; model I/O for different paths remains parallel.
    private let condition = NSCondition()
    private var cache: [String: DFlash2DraftModel] = [:]
    private var loads: [String: Load] = [:]
    private let loader: (URL) throws -> DFlash2DraftModel

    public init() {
        loader = DFlash2Loader.load(from:)
    }

    init(loader: @escaping (URL) throws -> DFlash2DraftModel) {
        self.loader = loader
    }

    /// Concurrent callers for one directory share the same allocation, including load failures.
    public func drafter(at path: URL) throws -> DFlash2DraftModel {
        let key = path.resolvingSymlinksInPath().path
        condition.lock()
        if let hit = cache[key] {
            condition.unlock()
            return hit
        }
        if let load = loads[key] {
            while load.result == nil { condition.wait() }
            let result = load.result!
            condition.unlock()
            return try result.get()
        }
        let load = Load()
        loads[key] = load
        condition.unlock()

        let result = Result { try loader(path) }
        condition.lock()
        load.result = result
        // Eviction invalidates publication, but callers already waiting still receive their result.
        // A new load after eviction must not be removed or overwritten by this older completion.
        if loads[key] === load {
            loads.removeValue(forKey: key)
            if case .success(let model) = result { cache[key] = model }
        }
        condition.broadcast()
        condition.unlock()
        return try result.get()
    }

    /// An optional acceleration artifact must not prevent ordinary generation.
    /// Only classified artifact failures fall back, and only before target prefill.
    public func optionalDrafter(at path: URL) throws -> DFlash2DraftModel? {
        try Task.checkCancellation()
        do {
            guard DFlash2ArtifactMetadata.rejectionReason(at: path) == nil else {
                try Task.checkCancellation()
                Logger(subsystem: "vmlx", category: "DFlash2").warning(
                    "Optional drafter metadata invalid; using AR")
                return nil
            }
            let model = try drafter(at: path)
            try Task.checkCancellation()
            return model
        } catch let error as DFlash2LoadError {
            try Task.checkCancellation()
            Logger(subsystem: "vmlx", category: "DFlash2").warning(
                "Optional drafter unavailable; using AR: \(error.localizedDescription, privacy: .public)"
            )
            return nil
        }
    }

    /// Load a drafter and check it against the model that will verify its
    /// drafts.
    ///
    /// Both checks are cheap and both failures are quiet if unchecked: a
    /// vocabulary mismatch produces plausible-looking garbage through the
    /// wrong LM head, and an out-of-range `target_layer_ids` would only
    /// surface as a missing key deep inside the capture path.
    public func drafter(
        at path: URL, vocabularySize: Int?, targetLayerCount: Int?
    ) throws -> DFlash2DraftModel {
        let model = try drafter(at: path)
        if let count = targetLayerCount,
            let maxLayer = model.config.targetLayerIds.max(), maxLayer >= count
        {
            throw DFlash2LoadError.targetMismatch(
                "drafter reads target layer \(maxLayer) but the model has \(count) layers")
        }
        if let vocabularySize, vocabularySize != model.config.vocabSize {
            throw DFlash2LoadError.targetMismatch(
                "drafter vocab_size \(model.config.vocabSize) != model vocabulary \(vocabularySize)"
            )
        }
        return model
    }

    public func evict(path: URL) {
        let key = path.resolvingSymlinksInPath().path
        condition.lock()
        defer { condition.unlock() }
        cache.removeValue(forKey: key)
        loads.removeValue(forKey: key)
    }

    public func evictAll() {
        condition.lock()
        defer { condition.unlock() }
        cache.removeAll()
        loads.removeAll()
    }
}
