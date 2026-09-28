// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The drafter's context across prompt-cache hits.
//
// The drafter conditions on the target's hidden states for the tokens before
// the block it drafts. A prompt-cache hit restores the target's KV for the
// prefix without running it, so those hidden states are never produced: the
// drafter saw only the prefilled suffix — 1 row on a full hit — and its
// acceptance fell from 2.31 to 1.33 tokens per verify on Spark2.5 4B. In a
// chat almost every request after the first is a hit.
//
// Prefill already computes the rows the drafter can use (its sliding window
// bounds them). This keeps the last few prompts' rows, keyed by their
// tokens, so a hit splices back the rows of the restored prefix — no extra
// forward and no added time to first token. Held by the drafter, so it is
// released with it.

import Foundation
import MLX

final class DFlash2ContextStore: @unchecked Sendable {
    private struct Entry {
        let tokens: [Int]
        let salt: String?
        /// Hidden rows for positions `tokens.count - rows.dim(1) ..< tokens.count`.
        let rows: MLXArray
    }

    /// Entries kept, most recent first. One row set is up to window × 5 ×
    /// hidden in bf16 (~52 MB for Spark2.5), so only the latest few prompts.
    static let capacity = 4

    private let lock = NSLock()
    private var entries: [Entry] = []

    /// Remember the context rows of a prefilled prompt.
    func store(
        tokens: [Int], salt: String?, rows: MLXArray,
        intent: LMInput.CachePromptIntent = .generation, isExact: Bool = true
    ) {
        // Auxiliary prompts must not evict conversational features. Features
        // reconstructed from a truncated full-attention prefix are not exact
        // and must not become a persistent source for subsequent requests.
        guard intent != .auxiliary, isExact else { return }
        guard rows.ndim == 3, rows.dim(1) > 0, rows.dim(1) <= tokens.count else { return }
        lock.lock()
        defer { lock.unlock() }
        entries.removeAll { $0.tokens == tokens && $0.salt == salt }
        entries.insert(Entry(tokens: tokens, salt: salt, rows: rows), at: 0)
        if entries.count > Self.capacity { entries.removeLast(entries.count - Self.capacity) }
    }

    func removeAll() {
        lock.lock()
        defer { lock.unlock() }
        entries.removeAll()
    }

    /// Rows for the last positions of `prompt[0 ..< restored]`, from a stored
    /// prompt sharing that prefix, or `nil` when none covers at least
    /// `minimumRows` positions ending at the requested boundary.
    func rows(
        endingAt restored: Int, of prompt: [Int], salt: String?, minimumRows: Int = 1
    ) -> MLXArray? {
        guard restored > 0, restored <= prompt.count,
            minimumRows > 0, minimumRows <= restored
        else { return nil }
        lock.lock()
        defer { lock.unlock() }
        for entry in entries where entry.salt == salt && entry.tokens.count >= restored {
            let rowsStart = entry.tokens.count - entry.rows.dim(1)
            guard restored - rowsStart >= minimumRows,
                entry.tokens[0 ..< restored].elementsEqual(prompt[0 ..< restored])
            else { continue }
            return entry.rows[0..., 0 ..< (restored - rowsStart), 0...]
        }
        return nil
    }

    /// Raw target features travel in the same file as the target KV boundary.
    /// They precede the drafter projection, so compatibility depends on the
    /// target's cache identity and ordered capture layers, not drafter weights.
    private static func diskContractKey(_ layers: [Int]) -> String {
        "dflash2_context_v1_layers_" + layers.map(String.init).joined(separator: "_")
    }

    func diskPayload(
        endingAt boundary: Int, of prompt: [Int], salt: String?, layers: [Int]
    ) -> [String: MLXArray]? {
        guard let rows = rows(endingAt: boundary, of: prompt, salt: salt),
            let end = Int32(exactly: boundary),
            let count = Int32(exactly: rows.dim(1)),
            let width = Int32(exactly: rows.dim(2)),
            !layers.isEmpty, layers.allSatisfy({ Int32(exactly: $0) != nil })
        else { return nil }
        return [
            "dflash2_context_meta": MLXArray([Int32(1), end, count, width]),
            "dflash2_context_layers": MLXArray(layers.map(Int32.init)),
            "dflash2_context_rows": rows,
            Self.diskContractKey(layers): MLXArray(Int32(1)),
        ]
    }

    static func validatedDiskPayload(
        _ arrays: [String: MLXArray], boundary: Int
    ) -> [String: MLXArray]? {
        guard let meta = arrays["dflash2_context_meta"], meta.shape == [4],
            meta.dtype == .int32,
            let layers = arrays["dflash2_context_layers"], layers.ndim == 1,
            layers.size > 0, layers.dtype == .int32,
            let rows = arrays["dflash2_context_rows"], rows.ndim == 3,
            rows.dim(0) == 1, rows.dim(1) > 0, rows.dim(1) <= boundary,
            rows.dim(2) > 0, [.bfloat16, .float16, .float32].contains(rows.dtype)
        else { return nil }
        let fields = meta.asArray(Int32.self)
        guard fields[0] == 1, Int(fields[1]) == boundary,
            Int(fields[2]) == rows.dim(1), Int(fields[3]) == rows.dim(2)
        else { return nil }
        // The contract is part of the key set as well as tensor metadata:
        // DiskCache's validated-store shortcut compares payload layouts.
        // Different tap IDs with the same shape must force a replacement.
        let contractKey = diskContractKey(layers.asArray(Int32.self).map(Int.init))
        guard let contract = arrays[contractKey], contract.ndim == 0,
            contract.dtype == .int32, contract.item(Int32.self) == 1
        else { return nil }
        return [
            "dflash2_context_meta": meta, "dflash2_context_layers": layers,
            "dflash2_context_rows": rows,
            contractKey: contract,
        ]
    }

    static func diskRows(
        _ arrays: [String: MLXArray], cacheBoundary: Int, endingAt end: Int,
        minimumRows: Int, layers: [Int], width: Int
    ) -> MLXArray? {
        guard end > 0, end <= cacheBoundary, minimumRows > 0,
            let payload = validatedDiskPayload(arrays, boundary: cacheBoundary),
            let rows = payload["dflash2_context_rows"], rows.dim(2) == width,
            let savedLayers = payload["dflash2_context_layers"],
            savedLayers.asArray(Int32.self).map(Int.init) == layers
        else { return nil }
        let start = cacheBoundary - rows.dim(1)
        guard end - start >= minimumRows else { return nil }
        return rows[0..., 0 ..< (end - start), 0...]
    }
}
