import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
import BenchmarkHelpers

@testable import MLXLLM
@testable import MLXVLM

/// Decode on this family is bandwidth-bound in principle — 20GB of 4-bit
/// weights per token — but the live app measures 12-13 tok/s against a ceiling
/// near 30, so most of the budget is going somewhere other than streaming
/// weights. This separates the two: it times the text tower's per-token
/// forward directly, with no sampler, detokenizer, cache I/O or UI in the way,
/// so what remains is the forward pass itself.
@Suite("Muse Glimmer decode perf")
struct MuseGlimmerDecodePerfTests {

    static let bundle = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("models/JANGQ-AI/Muse-Glimmer-30B-JANG_4M")

    static var enabled: Bool {
        ProcessInfo.processInfo.environment["MUSE_PERF"] == "1"
            && FileManager.default.fileExists(
                atPath: bundle.appendingPathComponent("config.json").path)
    }

    @Test("per-token forward cost, isolated from the serving stack",
        .enabled(if: enabled))
    func decodeForwardCost() async throws {
        let context = try await MLXLLM.LLMModelFactory.shared.load(
            from: Self.bundle, using: NoOpTokenizerLoader())
        let model = context.model

        // Prime a cache with a short prefix, then time single-token steps.
        let cache = model.newCache(parameters: nil)
        let prefix = MLXArray((0 ..< 64).map { Int32(1000 + $0) })[.newAxis, .ellipsis]
        _ = model(prefix, cache: cache)
        eval(cache.map { $0.state }.flatMap { $0 })

        var token = MLXArray([Int32(1234)])[.newAxis, .ellipsis]
        // Warm the kernels so the first dispatch is not counted.
        for _ in 0 ..< 3 {
            let out = model(token, cache: cache)
            eval(out)
        }

        let steps = 40
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< steps {
            let out = model(token, cache: cache)
            eval(out)
        }
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        let perToken = elapsed / Double(steps)
        let rate = 1.0 / perToken

        // 4-bit 30B is ~20GB of weight reads per token.
        let gbPerToken = 20.0
        print("[perf] forward-only: \(String(format: "%.2f", rate)) tok/s  (\(String(format: "%.1f", perToken * 1000)) ms/token)")
        print("[perf] implied bandwidth: \(String(format: "%.0f", gbPerToken * rate)) GB/s")
        print("[perf] live app measured 12-13 tok/s end to end")
        _ = token

        #expect(rate > 0, "no forward throughput measured")
    }
}
