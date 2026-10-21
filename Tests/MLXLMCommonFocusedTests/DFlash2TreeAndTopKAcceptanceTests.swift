// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Draft trees (DFlash2Tree.swift), the tree GatedDelta kernel, and the batched top-K speculative
// acceptance (SpeculativeTopKAcceptance.swift).

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

final class DFlash2TreeAndTopKAcceptanceTests: XCTestCase {

    // MARK: cost table target lifetime

    func testWidthCostsPersistForTheSameLivingTarget() {
        let target = NSObject()
        let table = DFlash2WidthCostTable.shared(for: target, widths: [1, 5, 8, 16])
        table.observe(width: 5, seconds: 0.1)
        table.observe(width: 5, seconds: 0.2)
        let reused = DFlash2WidthCostTable.shared(for: target, widths: [1, 5, 8, 16])
        XCTAssertTrue(table === reused)
        XCTAssertEqual(reused.cost(of: 5), 0.2)
        withExtendedLifetime(target) {}
    }

    func testWidthCostsDoNotRetainOrLeakAcrossUnloadedTargets() {
        final class Target {}
        weak var releasedTarget: Target?
        // Repeated lifetimes exercise allocator address reuse without requiring a particular
        // allocation address. Every newly loaded target must start with a cold cost table.
        for generation in 0 ..< 1_000 {
            autoreleasepool {
                let target = Target()
                releasedTarget = target
                let table = DFlash2WidthCostTable.shared(for: target, widths: [1, 5, 8, 16])
                XCTAssertNil(table.cost(of: 5), "inherited costs at generation \(generation)")
                table.observe(width: 5, seconds: 1)
                table.observe(width: 5, seconds: 1)
                XCTAssertEqual(table.cost(of: 5), 1)
                withExtendedLifetime(target) {}
            }
            XCTAssertNil(releasedTarget, "cost table retained model at generation \(generation)")
        }
    }

    // MARK: top-K distribution == the controller's full-vocabulary filter

    /// Host reference of `SpeculativeSamplingController.probabilities` (full vocab: temperature,
    /// top-p by ascending cumulative mass, min-p, top-k, renormalize).
    private func referenceDistribution(
        logits: [Float], temperature: Float, topP: Float, topK: Int, minP: Float
    ) -> [Int: Double] {
        let x = logits.map { Double($0) / Double(temperature) }
        let m = x.max()!
        let lse = m + log(x.reduce(0) { $0 + exp($1 - m) })
        let lp = x.map { $0 - lse }
        let ascending = lp.indices.sorted { lp[$0] < lp[$1] }
        var keep = Set(lp.indices)
        if topP > 0 && topP < 1 {
            var cumulative = 0.0
            for i in ascending {
                cumulative += exp(lp[i])
                if !(cumulative > 1 - Double(topP)) { keep.remove(i) }
            }
        }
        if minP > 0 {
            let threshold = lp.max()! + log(Double(minP))
            keep = keep.filter { lp[$0] >= threshold }
        }
        let topSet = Set(lp.indices.sorted { lp[$0] > lp[$1] }.prefix(topK))
        keep = keep.intersection(topSet)
        let total = keep.reduce(0) { $0 + exp(lp[$1]) }
        return Dictionary(uniqueKeysWithValues: keep.map { ($0, exp(lp[$0]) / total) })
    }

    func testTopKDistributionMatchesFullVocabularyFilter() throws {
        try FocusedMLXTestSupport.withLock {
            MLXRandom.seed(7)
            for (temperature, topP, topK, minP) in [
                (Float(1.0), Float(0.95), 20, Float(0)), (0.7, 0.8, 16, 0), (1.3, 1.0, 40, 0.05),
            ] {
                let logits = MLXRandom.normal([3, 2_000]) * 3
                let filter = try XCTUnwrap(
                    SpeculativeTopKAcceptance.Filter(
                        temperature: temperature, topP: topP, topK: topK, minP: minP))
                let (ids, lps) = SpeculativeTopKAcceptance.topK(logits: logits, filter: filter)
                let idRows = ids.asArray(Int32.self)
                let lpRows = lps.asArray(Float.self)
                let host = logits.asArray(Float.self)
                let k = ids.dim(1)
                for r in 0 ..< 3 {
                    let ours = SpeculativeTopKAcceptance.distribution(
                        ids: idRows[(r * k) ..< (r * k + k)], logprobs: lpRows[(r * k) ..< (r * k + k)],
                        filter: filter)
                    let reference = referenceDistribution(
                        logits: Array(host[(r * 2_000) ..< (r * 2_000 + 2_000)]),
                        temperature: temperature, topP: topP, topK: topK, minP: minP)
                    XCTAssertEqual(Set(ours.map(\.id)), Set(reference.keys), "kept set row \(r)")
                    for entry in ours {
                        XCTAssertEqual(entry.p, reference[entry.id]!, accuracy: 1e-4)
                    }
                    // The controller's (top-K fast path) vector agrees too.
                    let controller = SpeculativeSamplingController(
                        parameters: GenerateParameters(
                            temperature: temperature, topP: topP, topK: topK, minP: minP))
                    let vector = controller.probabilities(logits: logits[r ..< (r + 1)])
                        .asArray(Float.self)
                    for (id, p) in reference {
                        XCTAssertEqual(Double(vector[id]), p, accuracy: 1e-4)
                    }
                    XCTAssertEqual(
                        vector.enumerated().filter { $0.element > 0 }.count, reference.count)
                }
            }
        }
    }

    // MARK: acceptance is lossless (empirical)

    func testChainAcceptanceEmitsTargetDistribution() {
        let p: [(id: Int, p: Double)] = [(1, 0.5), (2, 0.3), (3, 0.2)]
        let q: [Int: Double] = [1: 0.2, 2: 0.6, 3: 0.2]
        var rng = SpeculativeHostRNG(seed: 11)
        var counts: [Int: Int] = [:]
        let trials = 60_000
        for _ in 0 ..< trials {
            // Draft sampled from q.
            var u = rng.uniform()
            var draft = 3
            for id in [1, 2, 3] {
                u -= q[id]!
                if u < 0 { draft = id; break }
            }
            let decision = SpeculativeTopKAcceptance.acceptChain(
                drafts: [draft], p: [p, [(9, 1.0)]], qAtKept: [p.map { q[$0.id]! }],
                qAtDraft: [q[draft]!], rng: &rng)
            counts[decision.tokenIds[0], default: 0] += 1
        }
        for entry in p {
            XCTAssertEqual(Double(counts[entry.id] ?? 0) / Double(trials), entry.p, accuracy: 0.01)
        }
    }

    func testTreeSampledAcceptanceEmitsTargetDistribution() {
        // Root (row 0) with three deterministic children: tokens 1, 2, 3 (draft order).
        let plan = DFlash2TreePlan(tokens: [0, 1, 2, 3], parents: [-1, 0, 0, 0])
        let p: [(id: Int, p: Double)] = [(1, 0.1), (2, 0.25), (3, 0.15), (4, 0.5)]
        let leaf: [(id: Int, p: Double)] = [(7, 1.0)]
        var rng = SpeculativeHostRNG(seed: 5)
        var counts: [Int: Int] = [:]
        let trials = 60_000
        for _ in 0 ..< trials {
            let path = DFlash2TreeAcceptance.sampledPath(
                plan: plan, distributions: [p, leaf, leaf, leaf], rng: &rng)
            let first = path.rows.count > 1 ? plan.tokens[path.rows[1]] : path.bonus
            counts[first, default: 0] += 1
        }
        for entry in p {
            XCTAssertEqual(Double(counts[entry.id] ?? 0) / Double(trials), entry.p, accuracy: 0.01)
        }
    }

    // MARK: tree plan geometry

    func testPlanMaskPositionsAndConvWindows() {
        // row 0 anchor; 1,2 children of 0; 3 child of 1; 4 child of 2.
        let plan = DFlash2TreePlan(tokens: [10, 11, 12, 13, 14], parents: [-1, 0, 0, 1, 2])
        XCTAssertEqual(plan.depths, [0, 1, 1, 2, 2])
        XCTAssertEqual(plan.paths[4], [0, 2, 4])
        let mask = plan.attentionMask(prefix: 2, dtype: .float32).reshaped(5, 7).asArray(Float.self)
        func visible(_ r: Int, _ c: Int) -> Bool { mask[r * 7 + c] == 0 }
        XCTAssertTrue(visible(4, 0) && visible(4, 1))  // prefix
        XCTAssertTrue(visible(4, 2 + 0) && visible(4, 2 + 2) && visible(4, 2 + 4))
        XCTAssertFalse(visible(4, 2 + 1) || visible(4, 2 + 3))
        XCTAssertEqual(
            plan.positionIds(start: 100)[0].reshaped(-1).asArray(Int32.self), [100, 101, 101, 102, 102])
        // nKeep 3: row 4's window = [conv state rows 1, 2] + path rows (0, 2, 4) shifted by 3, last 4.
        XCTAssertEqual(
            plan.convWindows(nKeep: 3)[4].asArray(Int32.self), [2, 3, 5, 7])
        XCTAssertEqual(plan.children[0], [1, 2])
    }

    func testBestFirstOnConfidentLatticeIsAChain() {
        let K = 4
        let L = 5
        let lattice = DFlash2Lattice(
            candidates: (0 ..< L).map { d in (0 ..< K).map { 100 * d + $0 } },
            unary: (0 ..< L).map { _ in [10, 0, 0, 0] },
            rootEdges: [Float](repeating: 0, count: K),
            edges: (0 ..< (L - 1)).map { _ in (0 ..< K).map { _ in [Float](repeating: 0, count: K) } })
        let tree = DFlash2TreeSearch.bestFirst(lattice: lattice, maxNodes: L, temperature: 0)
        XCTAssertEqual(tree.tokens, [0, 100, 200, 300, 400])
        XCTAssertEqual(tree.parents, [-1, 0, 1, 2, 3])
    }

    func testGreedyPathFollowsArgmaxThroughBranches() {
        let plan = DFlash2TreePlan(tokens: [0, 5, 6, 7, 8], parents: [-1, 0, 0, 2, 2])
        // root argmax 6 → row 2; row 2 argmax 8 → row 4; row 4 argmax 9 (no child) → bonus 9.
        let result = DFlash2TreeAcceptance.greedyPath(plan: plan, argmax: [6, 0, 8, 0, 9])
        XCTAssertEqual(result.rows, [0, 2, 4])
        XCTAssertEqual(result.bonus, 9)
    }

    // MARK: tree GatedDelta kernel

    /// A chain-shaped tree reproduces the production chain kernel bit for bit, and a branching
    /// tree's row r equals the chain kernel run over r's root path (its last output).
    func testTreeKernelMatchesChainKernelAlongEveryPath() throws {
        try FocusedMLXTestSupport.withLock {
            MLXRandom.seed(3)
            // The adaptive ladder can select 32 rows, beyond the old 16-row workspace.
            for W in [7, 16, 32] {
                let Hk = 2, Hv = 4, Dk = 64, Dv = 32
                let q = (MLXRandom.normal([1, W, Hk, Dk]) * 0.1).asType(.bfloat16)
                let k = (MLXRandom.normal([1, W, Hk, Dk]) * 0.1).asType(.bfloat16)
                let v = MLXRandom.normal([1, W, Hv, Dv]).asType(.bfloat16)
                let g = MLXRandom.uniform(low: 0.5, high: 0.99, [1, W, Hv])
                let b = MLXRandom.normal([1, W, Hv]).asType(.bfloat16)
                let state = MLXRandom.normal([1, Hv, Dv, Dk]) * 0.1

                let chainParents = Array(-1 ..< (W - 1))
                let treeY = Qwen35DFlash2TreeKernelProbe.tree(
                    q: q, k: k, v: v, g: g, b: b, state: state, parents: chainParents)
                let (chainY, _) = Qwen35DFlash2TreeKernelProbe.chain(
                    q: q, k: k, v: v, g: g, b: b, state: state)
                XCTAssertEqual(
                    treeY.asType(.float32).asArray(Float.self), chainY.asType(.float32).asArray(Float.self))

                let parents = [-1] + (1 ..< W).map { ($0 - 1) / 2 }
                let plan = DFlash2TreePlan(tokens: Array(0 ..< W), parents: parents)
                let branching = Qwen35DFlash2TreeKernelProbe.tree(
                    q: q, k: k, v: v, g: g, b: b, state: state, parents: parents)
                for row in 0 ..< W {
                    let index = MLXArray(plan.paths[row].map(Int32.init))
                    func rows(_ x: MLXArray) -> MLXArray { take(x, index, axis: 1) }
                    let (pathY, _) = Qwen35DFlash2TreeKernelProbe.chain(
                        q: rows(q), k: rows(k), v: rows(v), g: rows(g), b: rows(b), state: state)
                    let expected = pathY[0..., (plan.paths[row].count - 1)..., 0..., 0...]
                    XCTAssertEqual(
                        branching[0..., row ..< (row + 1), 0..., 0...].asType(.float32).asArray(Float.self),
                        expected.asType(.float32).asArray(Float.self), "width \(W), row \(row)")
                }
            }
        }
    }

    // MARK: KV compaction

    func testCompactTreeWindowKeepsPathRowsContiguously() {
        let cache = KVCacheSimple()
        let k = MLXArray(0 ..< 8).asType(.float32).reshaped(1, 1, 8, 1)
        _ = cache.update(keys: k, values: k * 10)
        // Window = last 5 rows (positions 3…7); keep window rows [0, 2, 4] → positions 3, 5, 7.
        XCTAssertTrue(cache.compactTreeWindow(window: 5, keptRows: [0, 2, 4]))
        XCTAssertEqual(cache.offset, 6)
        let (keys, values) = cache.readKV()!
        XCTAssertEqual(keys.reshaped(-1).asArray(Float.self), [0, 1, 2, 3, 5, 7])
        XCTAssertEqual(values.reshaped(-1).asArray(Float.self), [0, 10, 20, 30, 50, 70])
    }

    // MARK: lane matmul — direct 8-bit kernel

    /// `main8` / `main8_tiled` read 8-bit codes straight from device memory; they must give the
    /// widening bytes kernel's bits exactly (same matmul2d descriptor, same epilogue).
    func testLaneDirect8MatchesBytesKernelBitwise() throws {
        try FocusedMLXTestSupport.withLock {
            guard LaneQMM.available() else { throw XCTSkip("no Metal-4 tensor ops") }
            MLXRandom.seed(21)
            let n = 256, k = 1024
            let w = (MLXRandom.normal([n, k]) * 0.05).asType(.bfloat16)
            let (q, s, b) = MLX.quantized(w, groupSize: 64, bits: 8)
            let sbt = LaneQMM.packScales(s.asType(.bfloat16), b!.asType(.bfloat16))
            let tiledQ = LaneQMM.tileWeight(q, bits: 8)
            defer { LaneQMM.direct8Enabled = true }
            for rows in [1, 5, 16, 33] {
                let x = MLXRandom.normal([rows, k]).asType(.bfloat16)
                for (weight, tiled) in [(q, false), (tiledQ, true)] {
                    LaneQMM.direct8Enabled = false
                    let reference = LaneQMM.laneMatmul(
                        x, weight: weight, sbt: sbt, bits: 8, tiled: tiled)
                    LaneQMM.direct8Enabled = true
                    let direct = LaneQMM.laneMatmul(
                        x, weight: weight, sbt: sbt, bits: 8, tiled: tiled)
                    XCTAssertEqual(
                        direct.asType(.float32).asArray(Float.self),
                        reference.asType(.float32).asArray(Float.self),
                        "rows \(rows) tiled \(tiled)")
                }
                let dense = matmul(
                    x.asType(.float32),
                    dequantized(q, scales: s, biases: b, groupSize: 64, bits: 8).asType(.float32).T)
                let direct = LaneQMM.laneMatmul(x, weight: q, sbt: sbt, bits: 8, tiled: false)
                XCTAssertTrue(
                    allClose(direct.asType(.float32), dense, rtol: 0.02, atol: 0.05).item(Bool.self))
            }
        }
    }
}
