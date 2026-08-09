// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

/// The shared RoPE cos/sin memos are process-wide statics. Two concurrent
/// requests (Osaurus dispatches chat warmup alongside the visible turn) reach
/// them at the same time, so whatever they hand out is used from two threads
/// against two different MLX graphs.
@Suite("DSV4 shared RoPE tables are concurrency-safe", .serialized)
struct DeepseekV4RoPETableConcurrencyTests {

    private static func makeRoPE() -> DeepseekV4RoPE {
        DeepseekV4RoPE(dim: 64, base: 10000, factor: 1.0, origMaxPos: 65536)
    }

    private static func checksum(_ cos: MLXArray, _ sin: MLXArray) -> Float {
        let value = cos.asType(.float32).sum() * 3.0 + sin.asType(.float32).sum() * 7.0
        value.eval()
        return value.item(Float.self)
    }

    private static let lanes = 8
    private static let iterations = 40

    private static func countMismatches(
        reference: [Float], _ sample: @escaping (Int, Int) -> Float
    ) -> Int {
        let tally = NSLock()
        nonisolated(unsafe) var mismatches = 0
        DispatchQueue.concurrentPerform(iterations: lanes) { lane in
            var bad = 0
            for iteration in 0 ..< iterations {
                let slot = (lane + iteration) % reference.count
                if sample(lane, iteration) != reference[slot] { bad += 1 }
            }
            tally.lock()
            mismatches += bad
            tally.unlock()
        }
        return mismatches
    }

    /// Each entry is stored while still an unevaluated graph node. Handing the
    /// same node to another thread lets both threads drive `eval` on one
    /// `array_desc`, so the table can be read half-written.
    @Test("contended cos/sin windows stay bit-exact against a serial reference")
    func concurrentCosSinMatchesSerialReference() {
        nonisolated(unsafe) let rope = Self.makeRoPE()
        let windows: [(offset: Int, length: Int)] = [
            (0, 512), (512, 512), (2048, 724), (0, 1), (2772, 1),
        ]

        var reference: [Float] = []
        for w in windows {
            let (c, s) = rope.cosSin(offset: w.offset, length: w.length)
            reference.append(Self.checksum(c, s))
        }

        let mismatches = Self.countMismatches(reference: reference) { lane, iteration in
            let w = windows[(lane + iteration) % windows.count]
            let (c, s) = rope.cosSin(offset: w.offset, length: w.length)
            return Self.checksum(c, s)
        }
        #expect(mismatches == 0)
    }

    @Test("contended strided pool windows stay bit-exact against a serial reference")
    func concurrentStridedCosSinMatchesSerialReference() {
        nonisolated(unsafe) let rope = Self.makeRoPE()
        let windows: [(base: Int, count: Int, stride: Int)] = [
            (0, 512, 4), (0, 16, 128), (2048, 181, 4), (2048, 5, 128), (512, 128, 4),
        ]

        var reference: [Float] = []
        for w in windows {
            let (c, s) = rope.cosSin(base: w.base, count: w.count, stride: w.stride)
            reference.append(Self.checksum(c, s))
        }

        let mismatches = Self.countMismatches(reference: reference) { lane, iteration in
            let w = windows[(lane + iteration) % windows.count]
            let (c, s) = rope.cosSin(base: w.base, count: w.count, stride: w.stride)
            return Self.checksum(c, s)
        }
        #expect(mismatches == 0)
    }

    /// The structural invariant behind the fix: nothing may leave the memo
    /// while it is still a pending graph node, because a pending node is what
    /// two threads can race. MLX Swift exposes no `isAvailable`, so this pins
    /// the source instead — drop either `eval` and it goes red deterministically
    /// rather than waiting for the scheduler to lose the race.
    @Test("memoized tables are materialized before they are shared")
    func memoizedTablesAreEvaluatedBeforeSharing() throws {
        let source = try String(
            contentsOf: URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()  // MLXLMTests
                .deletingLastPathComponent()  // Tests
                .deletingLastPathComponent()  // package root
                .appendingPathComponent("Libraries/MLXLLM/Models/DeepseekV4.swift"),
            encoding: .utf8)

        let memos = [
            ("sharedCosSin", "tables[key] = (offset, length, c, s)"),
            ("sharedStridedCosSin", "stridedTables[sKey] = (base, count, c, s)"),
        ]
        for (memo, storeStatement) in memos {
            let start = try #require(source.range(of: "private static func \(memo)("))
            let body = String(source[start.lowerBound...].prefix(1400))
            let store = try #require(body.range(of: storeStatement)?.lowerBound)
            let materialize = try #require(body.range(of: "MLX.eval(c, s)")?.lowerBound)
            #expect(
                materialize < store,
                "\(memo) must eval the table before publishing it to other threads")
        }
    }
}
