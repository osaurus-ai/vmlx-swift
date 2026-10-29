// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The fused-decode seam (`MiniMaxM3FusedDecode`) with a spy path: off by default, offered only 1...16-row
/// single-sequence decode calls (any verify width), consulted by both the decode forward and the DSpark verify forward with the same rule, a
/// decline runs the stock path unchanged, and a returned block output is what the layer uses.
@Suite("MiniMax-M3 fused-decode seam", .serialized)
struct MiniMaxM3FusedDecodeSeamTests {
    /// Records every offer; returns `transform(stock)` or declines.
    final class SpyPath: MiniMaxM3FusedPath, @unchecked Sendable {
        let lock = NSLock()
        var offers: [(kind: String, layer: Int, rows: Int, verify: Bool)] = []
        let mode: Mode
        enum Mode { case decline, stock, perturb }
        init(_ mode: Mode) { self.mode = mode }

        func attentionBlock(
            layer: Int, module: MiniMaxM3DecoderLayer, x: MLXArray, cache: KVCache?, verify: Bool
        ) -> MLXArray? {
            lock.withLock { offers.append(("attn", layer, x.dim(1), verify)) }
            switch mode {
            case .decline: return nil
            case .stock, .perturb:
                let out =
                    verify
                    ? x + module.selfAttn.verifyRows(
                        module.inputLayerNorm(x), cache: cache!, offset: cache!.offset)
                    : x + module.selfAttn(module.inputLayerNorm(x), mask: x.dim(1) > 1 ? .causal : .none, cache: cache)
                return mode == .perturb ? out * 1.5 : out
            }
        }

        func mlpBlock(layer: Int, module: MiniMaxM3DecoderLayer, h: MLXArray, verify: Bool) -> MLXArray? {
            lock.withLock { offers.append(("mlp", layer, h.dim(1), verify)) }
            guard mode != .decline else { return nil }
            let y = module.postAttentionLayerNorm(h)
            if verify, let moe = module.mlp as? MiniMaxM3SparseMoeBlock { return h + moe.verifyRows(y) }
            if verify, let dense = module.mlp as? MiniMaxM3MLP { return h + dense.verifyRows(y) }
            return h + module.mlp(y)
        }
    }

    static func run(_ model: MiniMaxM3Model, prompt: [Int], next: [Int]) throws -> (MLXArray, MLXArray, MLXArray) {
        let cache = model.newCache(parameters: nil)
        let last = MiniMaxM3DSparkTests.prefill(model, prompt, cache: cache)
        let step = MiniMaxM3DSparkTests.decode(model, next[0], cache: cache)
        let verify = try MiniMaxM3DSparkTests.verify(model, Array(next[1...]), cache: cache)
        return (last, step, verify)
    }

    @Test("Off by default, decline == stock, offers only 1...16-row calls; decode and verify both consult it")
    func seamWiring() throws {
        let directory = MiniMaxM3DSparkTests.temporaryDirectory("seam")
        defer { try? FileManager.default.removeItem(at: directory) }
        try MiniMaxM3DSparkTests.writeReleaseDrafter(directory)
        try Device.withDefaultDevice(.gpu) {
            defer {
                MiniMaxM3FusedDecode.path = nil
                MiniMaxM3FusedDecode.forceEnabled = nil
            }
            let model = try MiniMaxM3DSparkTests.attachedModel(directory)
            let layers = model.model.layers.count
            let prompt = MiniMaxM3DSparkTests.ids(301, seed: 9)
            let next = MiniMaxM3DSparkTests.ids(8, seed: 10)
            let (refLast, refStep, refVerify) = try Self.run(model, prompt: prompt, next: next)

            // Registered but not enabled: never offered.
            let idle = SpyPath(.perturb)
            MiniMaxM3FusedDecode.path = idle
            MiniMaxM3FusedDecode.forceEnabled = false
            let (l0, s0, v0) = try Self.run(model, prompt: prompt, next: next)
            #expect(idle.offers.isEmpty)
            #expect(MiniMaxM3DSparkTests.equal(l0, refLast) && MiniMaxM3DSparkTests.equal(s0, refStep)
                && MiniMaxM3DSparkTests.equal(v0, refVerify))

            // Enabled, declining: offered for the decode step (1 row) and the 7-row verify, not for prefill chunks;
            // outputs bit-identical to stock.
            let spy = SpyPath(.decline)
            MiniMaxM3FusedDecode.path = spy
            MiniMaxM3FusedDecode.forceEnabled = true
            let (l1, s1, v1) = try Self.run(model, prompt: prompt, next: next)
            #expect(MiniMaxM3DSparkTests.equal(l1, refLast) && MiniMaxM3DSparkTests.equal(s1, refStep)
                && MiniMaxM3DSparkTests.equal(v1, refVerify))
            let rows = Set(spy.offers.map(\.rows))
            #expect(rows == [1, 7], "offered row counts \(rows)")
            #expect(spy.offers.filter { $0.rows == 1 }.allSatisfy { !$0.verify })
            #expect(spy.offers.filter { $0.rows == 7 }.allSatisfy { $0.verify })
            #expect(spy.offers.count == 2 * 2 * layers)  // attention + MLP, decode + verify, every layer

            // Enabled, returning the stock math: identical (the seam's output is used, the cache appends happen once).
            MiniMaxM3FusedDecode.path = SpyPath(.stock)
            let (l2, s2, v2) = try Self.run(model, prompt: prompt, next: next)
            #expect(MiniMaxM3DSparkTests.equal(l2, refLast) && MiniMaxM3DSparkTests.equal(s2, refStep)
                && MiniMaxM3DSparkTests.equal(v2, refVerify))

            // Control: a path whose attention output is off changes the decode step and the verify rows.
            MiniMaxM3FusedDecode.path = SpyPath(.perturb)
            let (l3, s3, v3) = try Self.run(model, prompt: prompt, next: next)
            #expect(MiniMaxM3DSparkTests.equal(l3, refLast))  // prefill chunks are never offered
            #expect(!MiniMaxM3DSparkTests.equal(s3, refStep) && !MiniMaxM3DSparkTests.equal(v3, refVerify))

            // More than 16 decode-forward rows or a batch: not offered; a verify of any width is (chunked by the path).
            #expect(MiniMaxM3FusedDecode.active(rows: 17, batch: 1) == nil)
            #expect(MiniMaxM3FusedDecode.active(rows: 17, batch: 1, verify: true) != nil)
            #expect(MiniMaxM3FusedDecode.active(rows: 17, batch: 2, verify: true) == nil)
            #expect(MiniMaxM3FusedDecode.active(rows: 4, batch: 2) == nil)
            #expect(MiniMaxM3FusedDecode.active(rows: 16, batch: 1) != nil)
        }
    }
}
