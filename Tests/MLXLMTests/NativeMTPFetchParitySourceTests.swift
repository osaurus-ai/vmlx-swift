// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

/// The native-MTP lane's prefix fetch must match the AR lanes (`TokenIterator`
/// in Evaluate.swift, the `BatchEngine` slot path) in two places that only
/// show up on a disk-backed hybrid topology (Allosaurus / Qwen 3.5 family:
/// GatedDeltaNet or Mamba state next to attention KV):
///
/// 1. `skipExactDiskBoundary` — a path-dependent cache cannot consume its
///    EXACT prompt boundary from disk, so the coordinator must serve the N-1
///    seed instead. Without the flag the MTP lane restored the exact row in
///    full (a whole-conversation SSD read evaluated on the GPU), then threw
///    it away in `unsafeFullHit` and prefilled the entire prompt cold. The
///    AR lanes never did this; it was the one MTP-only asymmetry in
///    preparation and hit every exact-prompt retry after a tool call.
/// 2. `MLXCacheIOLock` — the AR lanes serialize the restore's Metal evals
///    against other cache-adjacent GPU submitters; the MTP lane did not.
///
/// The lane builds its iterator inline inside `init`, with no seam to drive
/// without a real model and coordinator, so the two invariants are pinned
/// in the source the way `BatchStopTailBeforeToolCallSourceTests` pins the
/// stop-matcher drain.
@Suite("NativeMTP prefix fetch parity with the AR lanes")
struct NativeMTPFetchParitySourceTests {

    private static func source(_ relativePath: String) throws -> String {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(
            contentsOf: repoRoot.appending(path: relativePath),
            encoding: .utf8)
    }

    private static let iteratorPath = "Libraries/MLXLMCommon/SpecDec/NativeMTPTokenIterator.swift"

    /// The `coordinator.fetch(` call inside `init` (the one assigned to
    /// `result`), from its opening parenthesis to the matching close.
    private static func initFetchCall(in source: String) -> String? {
        guard let start = source.range(of: "result = coordinator.fetch(") else { return nil }
        // `start.upperBound` is just past the opening parenthesis.
        var depth = 1
        var index = start.upperBound
        while index < source.endIndex, depth > 0 {
            let ch = source[index]
            if ch == "(" { depth += 1 } else if ch == ")" { depth -= 1 }
            index = source.index(after: index)
        }
        return String(source[start.lowerBound ..< index])
    }

    @Test("init's coordinator.fetch passes skipExactDiskBoundary for disk-backed topologies")
    func fetchSkipsTheExactDiskBoundaryLikeTheARLane() throws {
        let source = try Self.source(Self.iteratorPath)
        guard let call = Self.initFetchCall(in: source) else {
            Issue.record(
                "`result = coordinator.fetch(` is gone from NativeMTPTokenIterator.init — rewrite this guard"
            )
            return
        }
        #expect(
            call.contains("skipExactDiskBoundary: requiresDiskBackedRestore"),
            """
            NativeMTPTokenIterator.init must fetch with \
            `skipExactDiskBoundary: requiresDiskBackedRestore` exactly as \
            Evaluate.swift's TokenIterator and BatchEngine.stepPrefill do. \
            Without it an exact disk hit on a hybrid cache is restored and \
            discarded, and the whole prompt is prefilled cold. Call was:
            \(call)
            """)
    }

    @Test("the AR lanes still pass the same flag (parity has a reference)")
    func arLanesStillPassTheFlag() throws {
        let evaluate = try Self.source("Libraries/MLXLMCommon/Evaluate.swift")
        let batch = try Self.source("Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift")
        #expect(evaluate.contains("skipExactDiskBoundary: requiresDiskBackedRestore,"))
        #expect(batch.contains("skipExactDiskBoundary: requiresDiskBackedRestore,"))
    }

    @Test("the disk restore in init runs under MLXCacheIOLock")
    func diskRestoreIsSerializedUnderTheCacheIOLock() throws {
        let source = try Self.source(Self.iteratorPath)
        guard let initStart = source.range(of: "result = coordinator.fetch(") else {
            Issue.record("init's fetch is gone — rewrite this guard")
            return
        }
        let afterFetch = source[initStart.upperBound...]
        guard
            let restore = afterFetch.range(
                of: "diskArrays, into: &self.cache, requirePromptBoundary: true")
        else {
            Issue.record(
                "init's `restoreFromDiskArrays(diskArrays, into: &self.cache, requirePromptBoundary: true)` is gone — rewrite this guard"
            )
            return
        }
        // The lock call must open between the fetch and the restore, and
        // the restore must be the first disk restore after it.
        let between = String(afterFetch[afterFetch.startIndex ..< restore.lowerBound])
        guard
            let lockAt = between.range(
                of: "MLXCacheIOLock.withSerializedMLXCacheIO", options: .backwards)
        else {
            Issue.record(
                """
                NativeMTPTokenIterator.init restores disk arrays outside \
                `MLXCacheIOLock.withSerializedMLXCacheIO`; the AR lanes \
                (Evaluate.swift, BatchEngine.swift) serialize the restore's \
                Metal evals under it.
                """)
            return
        }
        // The anchor is the restore's argument list, so the call's own name
        // sits in this span: exactly one call means no other restore between.
        let lockToRestore = between[lockAt.upperBound...]
        #expect(
            lockToRestore.components(separatedBy: "restoreFromDiskArrays(").count - 1 == 1,
            "another disk restore sits between the lock and init's restore; the lock must wrap this restore"
        )
        // The materializing eval is the submission that matters; it must be
        // inside the same closure (before the closure returns its count).
        let afterRestore = afterFetch[restore.upperBound...]
        guard let evalAt = afterRestore.range(of: "MLX.eval(self.cache)"),
            let returnAt = afterRestore.range(of: "return count")
        else {
            Issue.record(
                "the restore's `MLX.eval(self.cache)` / `return count` pair is gone — rewrite this guard"
            )
            return
        }
        #expect(
            evalAt.lowerBound < returnAt.lowerBound,
            "the restore's materializing eval must run inside the MLXCacheIOLock closure, not after it"
        )
    }
}
