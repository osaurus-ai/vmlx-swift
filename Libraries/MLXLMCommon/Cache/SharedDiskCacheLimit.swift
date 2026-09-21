import Foundation
import os

/// All coordinators writing one index must enforce the same live quota.
/// Weak registry entries let a root start with a freshly measured policy once
/// its last model unloads, without retaining caches or model state globally.
final class SharedDiskCacheLimit: @unchecked Sendable {
    private final class WeakLimit: @unchecked Sendable {
        weak var value: SharedDiskCacheLimit?
        init(_ value: SharedDiskCacheLimit) { self.value = value }
    }

    private static let roots = OSAllocatedUnfairLock(initialState: [String: WeakLimit]())
    private let storage: OSAllocatedUnfairLock<Int>

    private init(bytes: Int) {
        storage = OSAllocatedUnfairLock(initialState: max(0, bytes))
    }

    static func forRoot(_ root: URL, initialBytes: Int) -> SharedDiskCacheLimit {
        let key = root.standardizedFileURL.resolvingSymlinksInPath().path
        return roots.withLock { roots in
            if let existing = roots[key]?.value {
                existing.update(bytes: initialBytes)
                return existing
            }
            if roots.count >= 128 { roots = roots.filter { $0.value.value != nil } }
            let limit = SharedDiskCacheLimit(bytes: initialBytes)
            roots[key] = WeakLimit(limit)
            return limit
        }
    }

    var bytes: Int { storage.withLock { $0 } }

    func update(bytes: Int) {
        guard bytes > 0 else { return }
        storage.withLock { $0 = bytes }
    }
}
