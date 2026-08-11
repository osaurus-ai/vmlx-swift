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

    /// Prefill is the other half of the latency the user feels: TTFT is
    /// dominated by it on a cold prompt. Unlike decode, prefill is compute-
    /// bound rather than bandwidth-bound — every weight read is amortized over
    /// the whole chunk — so it should reach a far higher effective rate, and
    /// the interesting number is tokens/second at realistic chunk sizes.
    @Test("prefill throughput at several prompt lengths", .enabled(if: enabled))
    func prefillThroughput() async throws {
        let context = try await MLXLLM.LLMModelFactory.shared.load(
            from: Self.bundle, using: NoOpTokenizerLoader())
        let model = context.model

        for length in [256, 1024, 4096] {
            let cache = model.newCache(parameters: nil)
            let ids = MLXArray((0 ..< length).map { Int32(1000 + ($0 % 5000)) })[
                .newAxis, .ellipsis]
            // Warm once at this shape so kernel selection is not timed.
            let warmCache = model.newCache(parameters: nil)
            let warm = model(ids, cache: warmCache)
            eval(warm)

            let start = DispatchTime.now().uptimeNanoseconds
            let out = model(ids, cache: cache)
            eval(out)
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            let pps = Double(length) / elapsed
            print("[prefill] \(length) tokens: \(String(format: "%.3f", elapsed))s -> \(String(format: "%.0f", pps)) tok/s")
        }
        print("[prefill] target in the backlog is ~400 pp/s")
    }
}
