import Foundation

/// One process-wide reservation for immutable arrays held by asynchronous SSD
/// writers. Admission prices the incoming payload atomically; no producer waits
/// while holding a cache/quota/MLX lock. A full budget uses the synchronous path.
final class DiskWriterPayloadBudget: @unchecked Sendable {
    static let shared = DiskWriterPayloadBudget()
    private let lock = NSLock()
    private var reservedBytes = 0

    final class Reservation {
        private let owner: DiskWriterPayloadBudget
        let bytes: Int
        fileprivate init(owner: DiskWriterPayloadBudget, bytes: Int) {
            self.owner = owner; self.bytes = bytes
        }
        deinit { owner.release(bytes) }
    }

    func reserve(bytes: Int, limit: Int) -> Reservation? {
        guard bytes >= 0, limit >= 0, bytes <= limit else { return nil }
        lock.lock()
        defer { lock.unlock() }
        guard reservedBytes <= limit - bytes else { return nil }
        reservedBytes += bytes
        return Reservation(owner: self, bytes: bytes)
    }

    var currentBytes: Int {
        lock.lock(); defer { lock.unlock() }
        return reservedBytes
    }

    private func release(_ bytes: Int) {
        lock.lock(); defer { lock.unlock() }
        reservedBytes -= bytes
        assert(reservedBytes >= 0)
    }
}
