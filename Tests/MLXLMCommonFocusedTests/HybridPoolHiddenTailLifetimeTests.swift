// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Cmlx
import MLX
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct HybridPoolHiddenTailLifetimeTests {
    @Test("Compaction schedules hidden rows instead of retaining a lazy history graph")
    func compactionMaterializesRows() {
        Device.withDefaultDevice(.cpu) {
            var tail = HybridPoolHiddenTail(capacity: 128)
            for position in 0 ... HybridPoolHiddenTail.maxChunks {
                let rows = MLXArray([Float(position), Float(-position)], [1, 2]) + 1
                tail.record(rows: rows, startPosition: position)
            }
            #expect(tail.chunks.count == 1)
            let compacted = tail.chunks[0].rows
            Stream.defaultStream(.cpu).synchronize()
            var available = false
            #expect(_mlx_array_is_available(&available, compacted.ctx) == 0)
            #expect(available, "A lazy compacted slice still retains every previous compaction and its forward events")
            #expect(compacted[0..., 0].asArray(Float.self) == (0 ... HybridPoolHiddenTail.maxChunks).map { Float($0 + 1) })
        }
    }

    @Test("Long plain decoding preserves the exact bounded tail across compaction and rollback")
    func longHistoryAndRollback() {
        Device.withDefaultDevice(.cpu) {
            var tail = HybridPoolHiddenTail(capacity: 128)
            for position in 0 ..< 4096 {
                let rows = MLXArray([Float(position), Float(-position)], [1, 2]) + 1
                tail.record(rows: rows, startPosition: position)
            }
            #expect(tail.endPosition == 4096)
            #expect(tail.chunks.count <= HybridPoolHiddenTail.maxChunks)
            let last = tail.rows(from: 3968, to: 4096)
            #expect(last?[0..., 0].asArray(Float.self) == (3968 ..< 4096).map { Float($0 + 1) })
            var restored = tail
            restored.truncate(toEnd: 4090)
            restored.record(rows: MLXArray([9000.0, -9000.0], [1, 2]), startPosition: 4090)
            #expect(restored.endPosition == 4091)
            #expect(restored.rows(from: 4090, to: 4091)?.asArray(Float.self) == [9000, -9000])
            #expect(tail.endPosition == 4096)
        }
    }

    #if canImport(Metal)
    @Test("Long GPU decoding keeps unread tap rows bounded without consuming the draft tail")
    func longGPUHistory() throws {
        try Device.withDefaultDevice(.gpu) {
            var tail = HybridPoolHiddenTail(capacity: 128)
            try withError { error in
                for position in 0 ..< 4096 {
                    let a = MLXArray([Float(position)], [1, 1]) + 1
                    asyncEval(a)
                    let b = a + 1
                    asyncEval(b)
                    let c = b + 1
                    asyncEval(c)
                    // Like a resident drafter's taps during an AR request:
                    // the token path reads c, but never consumes these rows.
                    tail.record(rows: concatenated([a, b, c], axis: -1), startPosition: position)
                    try error.check()
                    let token = c.item(Float.self)
                    try error.check()
                    if position % 512 == 0 { #expect(token == Float(position + 3)) }
                }
                let rows = try #require(tail.rows(from: 3968, to: 4096))
                #expect(rows[0..., 0].asArray(Float.self) == (3968 ..< 4096).map { Float($0 + 1) })
                #expect(rows[0..., 2].asArray(Float.self) == (3968 ..< 4096).map { Float($0 + 3) })
                try error.check()
            }
        }
    }
    #endif
}
