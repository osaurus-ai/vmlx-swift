// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

/// Diagnostic probe for non-finite (NaN / ±inf) logits rows at the AR decode
/// sampling sites. **Observation only — never a guard.** It does not touch
/// the logits, the sampler, or the sampled token; it only reports.
///
/// Enabled by `VMLX_LOGITS_NAN_TRACE=1`. When the flag is off every call site
/// short-circuits on a cached `Bool` and no MLX work is scheduled.
///
/// Background: `TopPSampler` returns token id 0 (`!` on Ling 3 / KDA
/// vocabularies) for a NaN / −inf row, and users see `!!!!` streams on
/// Raptor. This trace answers "did the last-position row go non-finite
/// mid-decode, at which step, and what did the sampler return?".
///
/// Output (stderr), first occurrence per generation:
///
///     [vmlx][nan-logits] site=<solo|solo-compiled|batch|mtp> step=<n> slot=<id or -> model=<key> nan=<count> inf=<count> sampled=<token id>
///
/// and once at the end of a generation that saw at least one such row:
///
///     [vmlx][nan-logits] summary generations-with-nonfinite=<process total> steps=<rows this generation> ...
public final class NaNLogitsTrace: @unchecked Sendable {

    /// Cached once per process. Call sites gate on this before doing any work.
    public static let isEnabled: Bool =
        ProcessInfo.processInfo.environment["VMLX_LOGITS_NAN_TRACE"] == "1"

    private static let globalLock = NSLock()
    nonisolated(unsafe) private static var generationsWithNonFinite = 0  // guarded by globalLock

    /// Process-wide count of generations that saw at least one non-finite
    /// row. Exposed for tests.
    public static var generationsWithNonFiniteCount: Int {
        globalLock.lock()
        defer { globalLock.unlock() }
        return generationsWithNonFinite
    }

    public let slot: String
    public let model: String

    private let lock = NSLock()
    private var nonFiniteSteps = 0
    private var firstStep: Int?
    private var firstSite: String?
    private var totalNaN = 0
    private var totalInf = 0
    private var finished = false

    /// Returns `nil` when the trace flag is off so call sites can hold an
    /// optional and skip the probe entirely.
    public init?(slot: String = "-", model: String) {
        guard Self.isEnabled else { return nil }
        self.slot = slot
        self.model = model
    }

    /// Count NaN and ±inf entries in `rows` (any shape; typically the
    /// `[1, V]` last-position row, or the `[1, K, V]` verify slab). One
    /// `eval` for both reductions. Returns `nil` when everything is finite.
    public static func nonFiniteCounts(_ rows: MLXArray) -> (nan: Int, inf: Int)? {
        let nanCount = sum(isNaN(rows)).asType(.int32)
        let infCount = sum(isInf(rows)).asType(.int32)
        eval(nanCount, infCount)
        let nan = nanCount.item(Int.self)
        let inf = infCount.item(Int.self)
        if nan == 0 && inf == 0 { return nil }
        return (nan, inf)
    }

    /// Probe `rows`; when non-finite, record it (printing on the first
    /// occurrence for this generation). `sampled` is only evaluated when the
    /// row is non-finite, so a finite row never forces a token readback.
    public func observe(
        _ rows: MLXArray,
        site: String,
        step: Int,
        role: String? = nil,
        sampled: () -> Int
    ) {
        guard let counts = Self.nonFiniteCounts(rows) else { return }
        record(site: site, step: step, nan: counts.nan, inf: counts.inf, role: role,
               sampled: sampled())
    }

    /// Record an already-counted non-finite row.
    public func record(
        site: String, step: Int, nan: Int, inf: Int, role: String? = nil, sampled: Int
    ) {
        lock.lock()
        nonFiniteSteps += 1
        totalNaN += nan
        totalInf += inf
        let isFirst = firstStep == nil
        if isFirst {
            firstStep = step
            firstSite = site
        }
        lock.unlock()

        guard isFirst else { return }
        Self.globalLock.lock()
        Self.generationsWithNonFinite += 1
        Self.globalLock.unlock()

        var line =
            "[vmlx][nan-logits] site=\(site) step=\(step) slot=\(slot) model=\(model) nan=\(nan) inf=\(inf) sampled=\(sampled)"
        if let role { line += " role=\(role)" }
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }

    /// Emit the per-generation summary once. Silent when the generation
    /// never saw a non-finite row.
    public func finish(totalSteps: Int) {
        lock.lock()
        guard !finished else { lock.unlock(); return }
        finished = true
        let steps = nonFiniteSteps
        let first = firstStep
        let site = firstSite ?? "-"
        let nan = totalNaN
        let inf = totalInf
        lock.unlock()
        guard steps > 0 else { return }

        let line =
            "[vmlx][nan-logits] summary generations-with-nonfinite=\(Self.generationsWithNonFiniteCount) steps=\(steps) site=\(site) slot=\(slot) model=\(model) first-step=\(first ?? -1) nan=\(nan) inf=\(inf) total-steps=\(totalSteps)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }
}
