import Foundation
import MLX

/// Model-owned checkpoint signal. Unlike UI progress, this reports a materialized
/// complete chunk from an unchanged, cold model-owned text preparation.
public enum CanonicalTextPrefillCheckpointReporter {
    private final class Scope {
        let capture: CanonicalStablePrefillCapture
        init(_ capture: CanonicalStablePrefillCapture) { self.capture = capture }
    }
    private static let key = "ai.osaurus.vmlx.canonicalTextPrefillCheckpoint"

    public static var isActive: Bool { Thread.current.threadDictionary[key] is Scope }

    static func withCapture<T>(_ capture: CanonicalStablePrefillCapture?,
                               operation: () throws -> T) rethrows -> T {
        guard let capture else { return try operation() }
        let dictionary = Thread.current.threadDictionary
        let previous = dictionary[key]
        dictionary[key] = Scope(capture)
        defer {
            if let previous { dictionary[key] = previous }
            else { dictionary.removeObject(forKey: key) }
        }
        return try operation()
    }

    public static func reportQwen4ExpColdTextChunk(
        input: LMInput, cache: [KVCache], chunkSize: Int, completed: Int,
        beganWithEmptyCache: Bool
    ) {
        guard let scope = Thread.current.threadDictionary[key] as? Scope else { return }
        guard scope.capture.modelIdentity == nil else { return }
        scope.capture.receive(input: input, cache: cache, chunkSize: chunkSize,
                              completed: completed, beganWithEmptyCache: beganWithEmptyCache)
    }

    public static func reportModelColdTextChunk(
        modelIdentity: String, input: LMInput, cache: [KVCache], chunkSize: Int,
        completed: Int, beganWithEmptyCache: Bool
    ) {
        guard let scope = Thread.current.threadDictionary[key] as? Scope,
              scope.capture.modelIdentity == modelIdentity else { return }
        scope.capture.receive(input: input, cache: cache, chunkSize: chunkSize,
                              completed: completed, beganWithEmptyCache: beganWithEmptyCache)
    }
}

/// Request-local, one-snapshot retention. Never interprets aligned warm offsets
/// as canonical provenance and never changes the live prepare partition.
final class CanonicalStablePrefillCapture {
    let chunkSize: Int
    let seedCount: Int
    let modelIdentity: String?
    private let promptTokens: [Int]
    private let salt: String?
    private let owners: [ObjectIdentifier]
    private let schema: [String]
    private let exactReplayTarget: Int?
    private let ordinaryModel: (any CanonicalRequiredToolCacheModel)?
    private(set) var snapshot: [KVCache]?
    var isOrdinaryStableRederive: Bool { exactReplayTarget != nil }

    init?(input: LMInput, promptTokens: [Int], cache: [KVCache],
          chunkSize: Int, targets: [Int], salt: String?,
          canonicalModel: (any CanonicalRequiredToolCacheModel)? = nil,
          exactReplayTarget: Int? = nil) {
        guard chunkSize > 0, !input.hasMediaContent, !input.requiresPostPrepareCacheKey,
              input.cachePromptIntent != .auxiliary,
              input.text.mask == nil || input.text.mask?.size == promptTokens.count,
              !cache.isEmpty, cache.allSatisfy({ $0.offset == 0 && $0.state.isEmpty }),
              canonicalModel.map({ $0.validateCanonicalRequiredToolCache(cache, boundary: 0) })
                ?? cache.allSatisfy({ type(of: $0) == MambaCache.self || type(of: $0) == QSAKVCache.self }),
              let first = targets.filter({ $0 >= chunkSize && $0 < promptTokens.count }).min()
        else { return nil }
        let seed = (first / chunkSize) * chunkSize
        guard seed > 0, exactReplayTarget == nil
            || (exactReplayTarget == first && input.cacheStablePrefixTokenCounts.contains(first + 1)
                && input.text.mask == nil
                && input.cacheRestorePolicy == .standard && input.cachePromptIntent == .generation
                && input.canonicalRequiredToolContext == nil
                && canonicalModel?.supportsOrdinaryStablePrefixRederive == true)
        else { return nil }
        self.chunkSize = chunkSize; seedCount = seed
        self.exactReplayTarget = exactReplayTarget
        ordinaryModel = exactReplayTarget == nil ? nil : canonicalModel
        modelIdentity = canonicalModel?.canonicalRequiredToolCacheIdentity
        self.promptTokens = promptTokens; self.salt = salt
        owners = cache.map { ObjectIdentifier($0 as AnyObject) }
        schema = cache.map { String(reflecting: type(of: $0)) }
    }

    func receive(input: LMInput, cache: [KVCache], chunkSize: Int, completed: Int,
                 beganWithEmptyCache: Bool) {
        guard snapshot == nil, !Task.isCancelled, beganWithEmptyCache,
              chunkSize == self.chunkSize, completed == seedCount,
              !input.hasMediaContent,
              let ids = input.text.tokenIds, completed < ids.count,
              input.text.mask == nil || input.text.mask?.size == ids.count,
              ids.count <= promptTokens.count, promptTokens.starts(with: ids),
              cache.map({ ObjectIdentifier($0 as AnyObject) }) == owners,
              cache.map({ String(reflecting: type(of: $0)) }) == schema,
              cache.allSatisfy({ $0.offset == completed }) else { return }
        if let ordinaryModel {
            guard input.text.mask == nil,
                  ordinaryModel.validateCanonicalRequiredToolCache(cache, boundary: completed),
                  CacheStoreBudget.canStore(cache) else { return }
        }
        let start = Date.timeIntervalSinceReferenceDate
        let owned = makePromptBoundaryCacheSnapshot(from: cache)
        guard !Task.isCancelled,
              ordinaryModel.map({ $0.validateCanonicalRequiredToolCache(owned, boundary: completed) }) ?? true
        else { return }
        let elapsed = Date.timeIntervalSinceReferenceDate - start
        snapshot = owned
        if ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1" {
            // Metadata-only logical payload count, not allocator/physical footprint.
            let bytes = owned.reduce(0) { total, layer in
                total + layer.state.reduce(0) { $0 + $1.nbytes }
            }
            FileHandle.standardError.write(Data(
                ("[vmlx][cache/canonical-capture] seed=\(seedCount) chunk=\(chunkSize)"
                    + " snapshot_state_bytes=\(bytes) copy_eval_seconds=\(elapsed)\n").utf8))
        }
    }

    func copySeed(for tokens: [Int], salt: String?, chunkSize: Int) -> [KVCache]? {
        guard !Task.isCancelled, salt == self.salt, chunkSize == self.chunkSize,
              exactReplayTarget.map({ tokens.count == $0 }) ?? true,
              tokens.count >= seedCount, tokens.count < promptTokens.count,
              promptTokens.starts(with: tokens), let snapshot,
              snapshot.allSatisfy({ $0.offset == seedCount }),
              snapshot.map({ String(reflecting: type(of: $0)) }) == schema else { return nil }
        if let ordinaryModel {
            guard ordinaryModel.validateCanonicalRequiredToolCache(snapshot, boundary: seedCount) else { return nil }
        }
        let owned = makePromptBoundaryCacheSnapshot(from: snapshot)
        if let ordinaryModel {
            guard !Task.isCancelled, ordinaryModel.validateCanonicalRequiredToolCache(owned, boundary: seedCount) else { return nil }
        }
        return owned
    }
}
