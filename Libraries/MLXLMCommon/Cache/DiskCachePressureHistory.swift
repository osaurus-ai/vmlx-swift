import Foundation
#if canImport(os)
    import os
#endif

/// Process-lifetime capacity-loss metadata. It owns no model, tensors, cache
/// coordinator or SQLite connection, so idle unload cannot erase a notice
/// before a host polls it. Model fingerprints keep different topologies apart.
public enum DiskCachePressureHistory {
    private struct Key: Hashable {
        let root: String
        let model: String?
    }
    private static let storage = OSAllocatedUnfairLock(
        initialState: [Key: [String: DiskCachePressureRecord]]())

    /// The canonical form of a root. It walks symlinks, so a cache resolves
    /// it once at open and passes the result to the `rootKey:` overloads
    /// instead of paying for it on every poll and store.
    static func rootKey(for directory: URL) -> String {
        directory.standardizedFileURL.resolvingSymlinksInPath().path
    }

    private static func key(directory: URL, modelKey: String?) -> Key {
        Key(root: rootKey(for: directory), model: modelKey)
    }

    public static func records(
        directory: URL, modelKey: String?, maxSizeBytes: Int
    ) -> [String: DiskCachePressureRecord] {
        records(rootKey: rootKey(for: directory), modelKey: modelKey, maxSizeBytes: maxSizeBytes)
    }

    static func records(
        rootKey: String, modelKey: String?, maxSizeBytes: Int
    ) -> [String: DiskCachePressureRecord] {
        let key = Key(root: rootKey, model: modelKey)
        return storage.withLock { entries in
            let retained = (entries[key] ?? [:]).filter {
                $0.value.event.tipBytes > Int64(maxSizeBytes)
            }
            entries[key] = retained.isEmpty ? nil : retained
            return retained
        }
    }

    static func record(
        rootKey: String, modelKey: String?, event: DiskCachePressureEvent,
        tick: UInt64, tipTokenCount: Int
    ) {
        guard event.kind == .activeTipDropped, let chain = event.chainId else { return }
        let key = Key(root: rootKey, model: modelKey)
        storage.withLock { entries in
            // Uptime remains ordered across coordinator destruction/recreation.
            entries[key, default: [:]][chain] = DiskCachePressureRecord(
                event: event, sequence: tick, tick: tick, tipTokenCount: tipTokenCount)
        }
    }

    static func resolve(rootKey: String, modelKey: String?, chain: String, sequence: UInt64) {
        let key = Key(root: rootKey, model: modelKey)
        storage.withLock { entries in
            guard entries[key]?[chain]?.sequence == sequence else { return }
            entries[key]?.removeValue(forKey: chain)
            if entries[key]?.isEmpty == true { entries.removeValue(forKey: key) }
        }
    }

    /// Called only after an explicit purge of this root. Ordinary model unload
    /// and volatile-cache release must never call this.
    public static func clear(directory: URL) {
        let root = key(directory: directory, modelKey: nil).root
        storage.withLock { entries in entries = entries.filter { $0.key.root != root } }
    }
}
