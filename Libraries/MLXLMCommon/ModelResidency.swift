// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

/// The Metal residency set belongs to the process, not to the last model loaded.
/// A context owns one lease; copies of that context share the lease. Failed loads
/// release their provisional lease, and the last context release recomputes the
/// budget for the surviving models. This does not admit models or bypass RAM safety.
final class ModelResidency: @unchecked Sendable {
    struct Request: Sendable {
        let weightBytes: Int
        let enabled: Bool
        let physicalMemory: Int
        let workingSet: Int
        let systemWireLimit: Int?
    }

    final class Lease: @unchecked Sendable {
        private let owner: ModelResidency
        private let id: UUID

        fileprivate init(owner: ModelResidency, id: UUID) {
            self.owner = owner
            self.id = id
        }

        deinit { owner.remove(id) }
    }

    static let shared = ModelResidency(apply: setModelResidencyWiredLimit)

    private let lock = NSLock()
    private var requests: [UUID: Request] = [:]
    private var appliedTarget = 0
    private let apply: @Sendable (Int) -> Bool

    init(apply: @escaping @Sendable (Int) -> Bool) {
        self.apply = apply
    }

    /// All live allocations count, including a model loaded with pinning Off:
    /// Metal has one shared residency set, so it cannot exclude that model's
    /// buffers while keeping another model pinned. If every lease opts out,
    /// explicit residency is disabled. Scratch headroom is charged once.
    static func target(for requests: [Request]) -> Int {
        guard requests.contains(where: { $0.enabled }) else { return 0 }
        let bytes = requests.reduce(0) { total, request in
            let sum = total.addingReportingOverflow(max(0, request.weightBytes))
            return sum.overflow ? Int.max : sum.partialValue
        }
        let physical = requests.map(\.physicalMemory).min() ?? 0
        let workingSet = requests.map(\.workingSet).min() ?? 0
        let wireLimit = requests.compactMap(\.systemWireLimit).min()
        let aggregate = modelResidencyWiredTarget(
            modelBytes: bytes, workingSet: workingSet,
            physicalMemory: physical, systemWireLimit: wireLimit)
        // On small Macs the established policy declines oversized requests.
        // A new non-fitting model must not remove a survivor's valid budget.
        let survivor = requests.filter(\.enabled).compactMap {
            modelResidencyWiredTarget(
                modelBytes: $0.weightBytes, workingSet: workingSet,
                physicalMemory: physical, systemWireLimit: wireLimit)
        }.max() ?? 0
        return max(aggregate ?? 0, survivor)
    }

    func acquire(_ request: Request) -> (lease: Lease, target: Int, applied: Bool) {
        lock.lock()
        defer { lock.unlock() }
        let id = UUID()
        requests[id] = request
        let result = recompute()
        return (Lease(owner: self, id: id), result.target, result.applied)
    }

    private func remove(_ id: UUID) {
        lock.lock()
        defer { lock.unlock() }
        requests.removeValue(forKey: id)
        _ = recompute()
    }

    private func recompute() -> (target: Int, applied: Bool) {
        let requested = Self.target(for: Array(requests.values))
        // Do not zero/re-pin the survivor during every load. Apart from being
        // incorrect for multi-model hosts, that forces needless residency work.
        let applied = requested == appliedTarget || apply(requested)
        if applied { appliedTarget = requested }
        FileHandle.standardError.write(Data(
            "[Load] residency_models=\(requests.count) target_bytes=\(requested) applied=\(applied)\n".utf8))
        return (requested, applied)
    }
}
