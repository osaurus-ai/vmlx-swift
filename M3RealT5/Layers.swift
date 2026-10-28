// Branch-only (needs MiniMaxM3Attention.project): layer-by-layer dump for the Swift-vs-Python first-divergence search.
import Foundation
import MLX
import MLXLMCommon
import MLXNN
@testable import MLXLLM

// MARK: - layer dump

/// The decoder stack decomposed as MiniMaxM3ModelInner / MiniMaxM3DecoderLayer / MiniMaxM3SparseMoeBlock run it, with
/// every intermediate saved (keys as py_layers.py).
func taskLayers() async throws {
    let tokens = try loadTokens()
    let (model, _, _) = try await loadModel()
    let layerLens = (env["T5_LAYER_LENS"] ?? "100,600,3011").split(separator: ",").map { Int($0)! }
    var out: [String: MLXArray] = [:]
    let inner = model.model
    for n in layerLens {
        let cache = model.newCache(parameters: nil)
        let x = MLXArray(Array(tokens[0 ..< n])).reshaped(1, n)
        var h = inner.embedTokens(x)
        out["n\(n)_embed"] = h
        let mask = createAttentionMask(h: h, cache: cache.first)
        for (i, layer) in inner.layers.enumerated() {
            let p = "n\(n)_L\(i)_"
            let xn = layer.inputLayerNorm(h)
            if n <= 600 {
                // Attention internals without a cache (fresh sequence at offset 0).
                let sa = layer.selfAttn
                out[p + "a_qraw"] = sa.wq(xn)
                out[p + "a_kraw"] = sa.wk(xn)
                out[p + "a_vraw"] = sa.wv(xn)
                let (q, k, v) = sa.project(xn, offset: 0)
                out[p + "a_q"] = q
                out[p + "a_k"] = k
                out[p + "a_v"] = v
                let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: sa.scale, mask: .causal)
                out[p + "a_sdpa"] = o
                out[p + "a_out"] = sa.wo(o.transposed(0, 2, 1, 3).reshaped(1, n, -1))
            }
            let att = layer.selfAttn(xn, mask: mask, cache: cache[i])
            let h2 = h + att
            let pn = layer.postAttentionLayerNorm(h2)
            let m: MLXArray
            if let moe = layer.mlp as? MiniMaxM3SparseMoeBlock {
                let g = moe.gate(pn)
                let routerScores = sigmoid(g.asType(.float32))
                let choice = routerScores + moe.eScoreCorrectionBias
                let indices = argPartition(-choice, kth: moe.topK - 1, axis: -1)[.ellipsis, ..<moe.topK]
                var scores = takeAlong(routerScores, indices, axis: -1)
                if moe.topK > 1 && moe.normTopKProb {
                    scores = scores / scores.sum(axis: -1, keepDims: true)
                }
                scores = scores * moe.routedScalingFactor
                let routed = (moe.experts as! any WeightedRoutedExpertLayer).callRouted(
                    pn, indices: indices, scores: scores)
                let shared = moe.sharedExperts(pn)
                m = (routed + shared).asType(pn.dtype)
                out[p + "router_logits"] = g
                out[p + "inds"] = indices.asType(.int32)
                out[p + "scores"] = scores
                out[p + "routed"] = routed
                out[p + "shared"] = shared
                out[p + "mlp_module"] = moe(pn)
            } else {
                m = layer.mlp(pn)
            }
            let hn = h2 + m
            out[p + "in_norm"] = xn
            out[p + "attn"] = att
            out[p + "h_mid"] = h2
            out[p + "post_norm"] = pn
            out[p + "mlp"] = m
            out[p + "h_out"] = hn
            eval(Array(out.values))
            h = hn
        }
        let fn = inner.norm(h)
        out["n\(n)_final_norm"] = fn
        out["n\(n)_logits"] = model.lmHead!(fn)
        eval(Array(out.values))
        log("layers n=\(n) done")
    }
    try save(arrays: out, url: outDir.appendingPathComponent("swift_layers_\(tag).safetensors"))
}

