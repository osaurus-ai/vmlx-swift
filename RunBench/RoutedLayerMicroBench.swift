// BENCH_ROUTED_LAYER_MICRO=1 — decode cost of a bundle's real routed-expert layers (JANGT / JANGH custom banks):
// one token, top-k distinct experts, every routed layer chained like a decode step's routed part. Reports the
// median wall time per full 48-layer pass over BENCH_ROUTED_LAYER_ITERS iterations (default 40) and per layer.

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
import VMLXTokenizers

func runRoutedLayerMicroBench(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let iterations = Int(env["BENCH_ROUTED_LAYER_ITERS"] ?? "40") ?? 40
    let routes = Int(env["BENCH_ROUTED_LAYER_ROUTES"] ?? "10") ?? 10
    let rows = Int(env["BENCH_ROUTED_LAYER_ROWS"] ?? "1") ?? 1
    let context = try await MLXLMCommon.loadModel(
        from: URL(fileURLWithPath: modelPath), using: #huggingFaceTokenizerLoader(),
        loadConfiguration: LoadConfiguration()
    ).0
    let layers = context.model.namedModules()
        .compactMap { name, module -> (String, any WeightedRoutedExpertLayer)? in
            guard let routed = module as? any WeightedRoutedExpertLayer, !name.hasPrefix("mtp") else { return nil }
            return (name, routed)
        }
        .sorted { $0.0.localizedStandardCompare($1.0) == .orderedAscending }
    guard let first = layers.first else {
        print("[ROUTEDMICRO] INVALID: no custom routed layers")
        return
    }
    print("[ROUTEDMICRO] layers=\(layers.count) type=\(type(of: first.1)) rows=\(rows) routes=\(routes)")
    let hidden = 2560
    let x = (MLXRandom.normal([1, rows, hidden]) * 0.5).asType(.bfloat16)
    var picks: [Int32] = []
    for r in 0 ..< rows { for s in 0 ..< routes { picks.append(Int32((r * 97 + s * 37) % 512)) } }
    let indices = MLXArray(picks).reshaped(1, rows, routes)
    let scores = MLXArray(Array(repeating: Float(1) / Float(routes), count: rows * routes)).reshaped(1, rows, routes)
    MLX.eval(x, indices, scores)
    func pass() -> MLXArray {
        var acc = x
        for (_, layer) in layers {
            acc = (layer.callRouted(acc, indices: indices, scores: scores) * 0.5 + acc * 0.5).asType(.bfloat16)
        }
        return acc
    }
    for _ in 0 ..< 5 { MLX.eval(pass()) }
    var times: [Double] = []
    for _ in 0 ..< iterations {
        let t0 = CFAbsoluteTimeGetCurrent()
        MLX.eval(pass())
        times.append(CFAbsoluteTimeGetCurrent() - t0)
    }
    times.sort()
    let median = times[times.count / 2] * 1000
    print(String(format: "[ROUTEDMICRO] RESULT pass_ms=%.3f per_layer_ms=%.4f min_ms=%.3f",
                 median, median / Double(layers.count), times[0] * 1000))
}
