import Foundation
import MLX
import MLXLMCommon

/// BENCH_K2_MULTITURN=1 K2MT_MULTIROW=1 — prototype multi-row decode attention (speculative verify/draft shape).
///
/// Problem: MLX serves L query rows at long context by re-reading the whole KV per row (vector kernel, L·gqa ≤ 32) or
/// with a tiled kernel that walks every key serially (above). K2 at 32k: 1 row 22 ms, 8 rows 115 ms, 16 rows 200 ms.
/// Here a threadgroup owns (kv head, key block); 32-key tiles of K and V are staged ONCE in threadgroup memory and every
/// (query head, row) pair of the GQA group consumes them. QK: lane = key, full 128-dim dot (no shuffles). Softmax: one
/// simd_max + simd_sum per tile per pair. PV: lane = 4 dims, probabilities broadcast by shuffle. fp32 throughout.
/// Bottom-right causal: row i sees keys [0, N − L + i].
func runMultiRowAttentionProbe() {
    let env = ProcessInfo.processInfo.environment
    let blocksList = (env["K2MT_MRBLOCKS"] ?? "32,64,128").split(separator: ",").compactMap { Int($0) }
    // K2MT_MR_SHAPE = "kvHeads,gqa,headDim,layers" (default K2 Horizon 7B: 8,4,128,36; Qwen3.8 27B: 4,6,256,16).
    let shape = (env["K2MT_MR_SHAPE"] ?? "8,4,128,36").split(separator: ",").compactMap { Int($0) }
    let (hkv, gqa, dim, layers) = (shape[0], shape[1], shape[2], shape[3])
    let scale = Float(1 / Double(dim).squareRoot())
    let rowsList = (env["K2MT_MR_ROWS"] ?? "1,4,8,12,16").split(separator: ",").compactMap { Int($0) }
    for ctx in [2048, 14336, 32768] {
        let cap = (ctx / 256 + 1) * 256
        let kb = (0 ..< layers).map { _ in MLXRandom.normal([1, hkv, cap, dim]).asType(.bfloat16) }
        let vb = (0 ..< layers).map { _ in MLXRandom.normal([1, hkv, cap, dim]).asType(.bfloat16) }
        eval(kb); eval(vb)
        for qL in rowsList {
            let q = (MLXRandom.normal([1, hkv * gqa, qL, dim]) * 1.5).asType(.bfloat16)
            eval(q)
            let k0 = kb[0][0..., 0..., ..<ctx, 0...], v0 = vb[0][0..., 0..., ..<ctx, 0...]
            let mask: MLXFast.ScaledDotProductAttentionMaskMode = qL == 1 ? .none : .causal
            // fp32 reference (MLX SDPA in fp32), then MLX bf16, then ours.
            let ref = MLXFast.scaledDotProductAttention(
                queries: q.asType(.float32), keys: k0.asType(.float32), values: v0.asType(.float32), scale: scale,
                mask: mask)
            let mlx = MLXFast.scaledDotProductAttention(queries: q, keys: k0, values: v0, scale: scale, mask: mask)
            let mine = MultiRowDecodeAttention.attend(queries: q, keys: k0, values: v0, scale: scale, blocks: 64)
            func rel(_ a: MLXArray) -> Float {
                let d = (a.asType(.float32) - ref)
                return (sqrt((d * d).mean()) / sqrt((ref * ref).mean())).item(Float.self)
            }
            let errMLX = rel(mlx), errMine = rel(mine)
            func time(_ body: () -> MLXArray) -> Double {
                var t: [Double] = []
                for _ in 0 ..< 15 { let t0 = Date(); eval(body()); t.append(Date().timeIntervalSince(t0)) }
                t.sort(); return t[7]
            }
            // chained layers: next layer's queries depend on this layer's output.
            let tm = time {
                var x = q
                for l in 0 ..< layers {
                    x = MLXFast.scaledDotProductAttention(
                        queries: x, keys: kb[l][0..., 0..., ..<ctx, 0...], values: vb[l][0..., 0..., ..<ctx, 0...],
                        scale: scale, mask: mask)
                }
                return x
            }
            var line = "K2MT_MR ctx=\(ctx) L=\(qL) mlx_ms=\(String(format: "%.2f", tm * 1000))"
            for nb in blocksList {
                let t = time {
                    var x = q
                    for l in 0 ..< layers {
                        x = MultiRowDecodeAttention.attend(
                            queries: x, keys: kb[l][0..., 0..., ..<ctx, 0...], values: vb[l][0..., 0..., ..<ctx, 0...],
                            scale: scale, blocks: nb)
                    }
                    return x
                }
                line += " b\(nb)=\(String(format: "%.2f", t * 1000))"
            }
            line += " relerr_mlx=\(String(format: "%.1e", errMLX)) relerr_mine=\(String(format: "%.1e", errMine))"
            print(line)
        }
    }
}
