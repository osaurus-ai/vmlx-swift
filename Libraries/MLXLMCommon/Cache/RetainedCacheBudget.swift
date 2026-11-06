// Copyright © 2026 Tijs Teulings

import Foundation
import MLX

/// One retained-byte allowance shared by a coordinator's volatile tiers.
/// Owners evict only their own entries; the budget never calls into a cache.
/// Thus cache locks may acquire this lock, but no reverse lock order exists.
/// A store may miss a retention opportunity while a different tier owns cold
/// entries; accepting that miss avoids cross-tier callbacks and a global LRU.
final class RetainedCacheBudget: @unchecked Sendable {
    let limit: Int?
    private let lock = NSLock()
    private var usedBytes = 0

    init(limit: Int?) {
        precondition(limit == nil || limit! >= 0)
        self.limit = limit
    }

    var residentBytes: Int { lock.withLock { usedBytes } }

    func makeAccount() -> Account { Account(budget: self) }

    /// Includes the pinned Metal backend's compact-buffer slack, just as the
    /// paged tier did before the allowance was shared with separate SSM state.
    static func storageCharge(_ arrays: [MLXArray]) -> Int {
        arrays.reduce(0) { total, array in
            let (charge, slackOverflow) = array.nbytes.addingReportingOverflow(16_384)
            let (sum, totalOverflow) = total.addingReportingOverflow(charge)
            return slackOverflow || totalOverflow ? Int.max : sum
        }
    }

    /// Each cache owns one account. In-flight optional copies are reserved
    /// before materialization and remain charged until committed or abandoned.
    final class Account: @unchecked Sendable {
        private let budget: RetainedCacheBudget
        private var bytes = 0 // Protected by budget.lock, including deinit.

        fileprivate init(budget: RetainedCacheBudget) { self.budget = budget }

        var limit: Int? { budget.limit }

        func reserve(_ amount: Int) -> Bool {
            precondition(amount >= 0)
            return budget.lock.withLock {
                if let limit = budget.limit,
                   amount > limit || budget.usedBytes > limit - amount {
                    return false
                }
                let (next, overflow) = budget.usedBytes.addingReportingOverflow(amount)
                guard !overflow else { return false }
                budget.usedBytes = next
                bytes += amount
                return true
            }
        }

        func release(_ amount: Int) {
            budget.lock.withLock {
                precondition(amount >= 0 && amount <= bytes)
                bytes -= amount
                budget.usedBytes -= amount
            }
        }

        func releaseAll() {
            budget.lock.withLock {
                budget.usedBytes -= bytes
                bytes = 0
            }
        }

        deinit { releaseAll() }
    }
}
