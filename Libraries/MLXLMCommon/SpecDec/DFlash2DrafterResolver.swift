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

public final class DFlash2DrafterResolver: @unchecked Sendable {

    public static let shared = DFlash2DrafterResolver()

    private let lock = NSLock()
    private var cache: [String: DFlash2DraftModel] = [:]

    public init() {}

    /// Load (or return the cached) drafter at `path`.
    public func drafter(at path: URL) throws -> DFlash2DraftModel {
        let key = path.resolvingSymlinksInPath().path
        lock.lock()
        let hit = cache[key]
        lock.unlock()
        if let hit { return hit }

        // Loading happens outside the lock: it is seconds of disk I/O, and
        // holding the lock across it would serialise unrelated requests
        // behind the first one. A concurrent duplicate load is possible and
        // harmless — the second one just replaces an identical entry.
        let model = try DFlash2Loader.load(from: path)
        lock.lock()
        cache[key] = model
        lock.unlock()
        return model
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
        lock.lock()
        defer { lock.unlock() }
        cache.removeValue(forKey: path.resolvingSymlinksInPath().path)
    }

    public func evictAll() {
        lock.lock()
        defer { lock.unlock() }
        cache.removeAll()
    }
}
