import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Development diagnostics for the Ripple port (RunBench `BENCH_RIPPLE_TIMING`). Not on any generation path.
extension RicursiveModel {
    /// Median milliseconds to run one component over every layer for `rows` rows (one eval per sample):
    /// `mlp` (gate/up/down), `attn_proj` (q/k/v/g/o projections only), `head` (lm_head once).
    public func diagnosticComponentTimings(rows: Int, reps: Int = 7) -> [String: Double] {
        let c = configuration
        let x = MLXRandom.normal([1, rows, c.hiddenSize]).asType(.bfloat16) * 0.05
        let a = MLXRandom.normal([1, rows, c.attentionHeads * c.headDim]).asType(.bfloat16) * 0.05
        eval(x, a)
        func median(_ body: () -> [MLXArray]) -> Double {
            var s: [Double] = []
            for r in 0 ..< (reps + 2) {
                let t = CFAbsoluteTimeGetCurrent()
                eval(body())
                if r >= 2 { s.append((CFAbsoluteTimeGetCurrent() - t) * 1000) }
            }
            return s.sorted()[s.count / 2]
        }
        var out: [String: Double] = [:]
        out["mlp"] = median { model.layers.map { $0.mlp(x) } }
        out["attn_proj"] = median {
            model.layers.flatMap { l -> [MLXArray] in
                let at = l.attention
                var o = [at.wq(x), at.wk(x), at.wv(x), at.wo(a)]
                if let g = at.wg { o.append(g(x)) }
                return o
            }
        }
        out["head"] = median { [projectToLogits(x)] }
        return out
    }
}
