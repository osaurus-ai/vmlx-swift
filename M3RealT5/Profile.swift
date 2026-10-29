// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Test harness (not shipped): one MiniMax-M3.1 decode step on the real bundle, decomposed. Each variant is a whole
// one-token forward with parts of every layer left out, evaluated once per step (no extra syncs); the variants'
// differences give each component's share of the step. The graph of each variant is exported as DOT (one node per
// primitive: the launch-count proxy). Caches are trimmed back after every step, so every probe runs at one context.

import Cmlx
import Foundation
import MLXNN

@testable import MLX
@testable import MLXLLM
@testable import MLXLMCommon

func exportDot(_ arrays: [MLXArray], _ name: String) -> String {
    let path = outDir.appendingPathComponent("\(name).dot").path
    guard let file = fopen(path, "w") else { return path }
    let vector = new_mlx_vector_array(arrays)
    let namer = mlx_node_namer_new()
    _ = mlx_export_to_dot(file, namer, vector)
    fclose(file)
    mlx_vector_array_free(vector)
    _ = mlx_node_namer_free(namer)
    return path
}

func taskProfile() async throws {
    let (model, _, _) = try await productionLoad(bundleURL())
    let inner = model.model
    let lens = (env["T5_PROF_LENS"] ?? "3011,32845").split(separator: ",").map { Int($0)! }
    let probes = Int(env["T5_PROBES"] ?? "9")!
    let tokens = try loadTokens()
    func head(_ h: MLXArray) -> MLXArray {
        let n = inner.norm(h)
        return model.lmHead.map { $0(n) } ?? inner.embedTokens.asLinear(n)
    }
    func router(_ block: MiniMaxM3SparseMoeBlock, _ x: MLXArray) -> MLXArray {
        let routerScores = sigmoid(block.gate(x).asType(.float32))
        let choice = routerScores + block.eScoreCorrectionBias
        let indices = argPartition(-choice, kth: block.topK - 1, axis: -1)[.ellipsis, ..<block.topK]
        var scores = takeAlong(routerScores, indices, axis: -1)
        scores = scores / scores.sum(axis: -1, keepDims: true) * block.routedScalingFactor
        return (scores.sum(axis: -1, keepDims: true) * 0).asType(x.dtype) + indices.asType(x.dtype)[.ellipsis, ..<1] * 0
    }
    typealias Variant = (String, Bool, (MLXArray, [KVCache]) -> MLXArray)
    let variants: [Variant] = [
        ("full", true, { x, cache in model(x, cache: cache) }),
        ("head_only", false, { x, _ in head(inner.embedTokens(x)) }),
        ("attention_only", true, { x, cache in
            var h = inner.embedTokens(x)
            for (i, layer) in inner.layers.enumerated() {
                h = h + layer.selfAttn(layer.inputLayerNorm(h), mask: .none, cache: cache[i])
            }
            return head(h)
        }),
        ("qkv_o_projections_only", false, { x, cache in
            var h = inner.embedTokens(x)
            for (i, layer) in inner.layers.enumerated() {
                let sa = layer.selfAttn
                let xn = layer.inputLayerNorm(h)
                let (q, k, v) = sa.project(xn, offset: cache[i].offset)
                var o = sa.wo(q.transposed(0, 2, 1, 3).reshaped(1, 1, -1))
                o = o + (k.sum() + v.sum()).asType(o.dtype) * 0
                if let indexer = sa.indexer {
                    let iq = indexer.queryProjection(xn)
                    let ik = indexer.keyProjection(xn)
                    o = o + (iq.sum() + ik.sum()).asType(o.dtype) * 0
                }
                h = h + o
            }
            return head(h)
        }),
        ("mlp_only", false, { x, _ in
            var h = inner.embedTokens(x)
            for layer in inner.layers { h = h + layer.mlp(layer.postAttentionLayerNorm(h)) }
            return head(h)
        }),
        ("mlp_shared_only", false, { x, _ in
            var h = inner.embedTokens(x)
            for layer in inner.layers {
                let xn = layer.postAttentionLayerNorm(h)
                if let block = layer.mlp as? MiniMaxM3SparseMoeBlock {
                    h = h + block.sharedExperts(xn)
                } else {
                    h = h + layer.mlp(xn)
                }
            }
            return head(h)
        }),
        ("mlp_router_and_shared", false, { x, _ in
            var h = inner.embedTokens(x)
            for layer in inner.layers {
                let xn = layer.postAttentionLayerNorm(h)
                if let block = layer.mlp as? MiniMaxM3SparseMoeBlock {
                    h = h + block.sharedExperts(xn) + router(block, xn)
                } else {
                    h = h + layer.mlp(xn)
                }
            }
            return head(h)
        }),
        ("norms_only", false, { x, _ in
            var h = inner.embedTokens(x)
            for layer in inner.layers { h = layer.postAttentionLayerNorm(layer.inputLayerNorm(h)) }
            return head(h)
        }),
    ]
    var report: [String: Any] = ["probes": probes, "msa": env["VMLX_MINIMAX_MSA"] ?? "auto"]
    for n in lens {
        let cache = model.newCache(parameters: nil)
        var s = 0
        while s < n {
            let e = min(s + chunk, n)
            eval(model(MLXArray(Array(tokens[s ..< e])).reshaped(1, e - s), cache: cache))
            s = e
        }
        Memory.clearCache()
        let x = MLXArray([tokens[n]]).reshaped(1, 1)
        var row: [String: Any] = [:]
        // Interleave: every probe runs every variant once (order rotates), 2 discarded probes.
        var times: [String: [Double]] = [:], build: [String: [Double]] = [:]
        for p in 0 ..< probes + 2 {
            let order = p % 2 == 0 ? variants : variants.reversed()
            for (name, mutates, body) in order {
                let t0 = nowMs()
                let out = body(x, cache)
                let t1 = nowMs()
                eval(out)
                let t2 = nowMs()
                if mutates { for c in cache where c.offset > n { _ = c.trim(c.offset - n) } }
                if p >= 2 {
                    times[name, default: []].append(t2 - t0)
                    build[name, default: []].append(t1 - t0)
                }
            }
        }
        precondition(Set(cache.map(\.offset)) == [n], "cache moved: \(Set(cache.map(\.offset)))")
        for (name, mutates, body) in variants {
            let out = body(x, cache)
            let dot = exportDot([out], "profile_\(tag)_\(n)_\(name)")
            eval(out)
            if mutates { for c in cache where c.offset > n { _ = c.trim(c.offset - n) } }
            row[name] = [
                "step_ms_median": median(times[name]!), "step_ms_runs": times[name]!,
                "graph_build_ms_median": median(build[name]!), "dot": dot,
            ]
        }
        report["\(n)"] = row
        log("profile n=\(n): " + variants.map { "\($0.0) \(String(format: "%.2f", median(times[$0.0]!)))" }.joined(separator: ", "))
    }
    try writeJSON(report, "profile_\(tag).json")
}
