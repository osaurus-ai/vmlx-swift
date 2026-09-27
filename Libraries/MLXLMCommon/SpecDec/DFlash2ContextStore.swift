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
    func store(tokens: [Int], salt: String?, rows: MLXArray) {
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
    /// prompt sharing that prefix, or `nil` when none covers it.
    func rows(endingAt restored: Int, of prompt: [Int], salt: String?) -> MLXArray? {
        guard restored > 0, restored <= prompt.count else { return nil }
        lock.lock()
        defer { lock.unlock() }
        for entry in entries where entry.salt == salt && entry.tokens.count >= restored {
            let rowsStart = entry.tokens.count - entry.rows.dim(1)
            guard restored > rowsStart,
                entry.tokens[0 ..< restored].elementsEqual(prompt[0 ..< restored])
            else { continue }
            return entry.rows[0..., 0 ..< (restored - rowsStart), 0...]
        }
        return nil
    }
}
