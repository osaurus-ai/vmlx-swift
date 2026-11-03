import Dispatch
import Foundation
import MLX
import XCTest

@testable import MLXLLM

/// Diagnostic paired activation timing; no model generation or speed assertion.
final class Spark25MLPInputCastPerformanceTests: XCTestCase {
    private enum Failure: Error { case invalid(String) }
    private struct Sample {
        let nanoseconds: UInt64
        let outputs: [MLXArray]
    }
    private func measure(_ stream: MLX.Stream, _ body: () throws -> [MLXArray]) rethrows -> Sample {
        stream.synchronize()
        let start = DispatchTime.now().uptimeNanoseconds
        let outputs = try body()
        eval(outputs)
        stream.synchronize()
        return Sample(nanoseconds: DispatchTime.now().uptimeNanoseconds - start, outputs: outputs)
    }
    private func stats(_ samples: [Double]) -> [String: Any] {
        let values = samples.sorted()
        func q(_ p: Double) -> Double {
            let at = Double(values.count - 1) * p
            let lo = Int((Double(values.count - 1) * p).rounded(.down))
            let hi = min(lo + 1, values.count - 1)
            return values[lo] + (values[hi] - values[lo]) * (at - Double(lo))
        }
        return [
            "count": values.count, "median_ns": q(0.5), "q1_ns": q(0.25), "q3_ns": q(0.75),
            "iqr_ns": q(0.75) - q(0.25),
        ]
    }
    func testOptInPairedInputCastBenchmark() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["VMLX_SPARK_MLP_INPUT_CAST_BENCH"] == "1" else {
            throw XCTSkip("Set VMLX_SPARK_MLP_INPUT_CAST_BENCH=1 for diagnostic timing")
        }
        #if !canImport(Metal)
            throw XCTSkip("Metal unavailable for opt-in projection timing")
        #endif
        guard let path = env["VMLX_SPARK_MLP_INPUT_CAST_BENCH_OUTPUT"],
            let iterations = Int(env["VMLX_SPARK_MLP_INPUT_CAST_BENCH_ITERATIONS"] ?? "1"),
            iterations > 0,
            env["VMLX_SPARK_MLP_INPUT_CAST_REFERENCE"] != "1",
            env["VMLX_SPARK_DUAL_Q6_REFERENCE"] != "1"
        else {
            throw Failure.invalid(
                "Require new output path, positive iterations and both candidates enabled")
        }
        let output = URL(fileURLWithPath: path)
        guard !FileManager.default.fileExists(atPath: path) else {
            throw Failure.invalid("Output exists")
        }
        var receipt: [String: Any] = [
            "scope":
                "native norm unchanged; old explicitBF16cast+existingq6 vsFP32readcastq6 activation",
            "iterations": iterations, "warmups": 5, "rounds": 3, "samples_per_round": 30,
            "status": "running",
            "timing":
                "graph construction+eval+scoped GPU completion; input preparation and validation excluded",
            "first_use_scope": "process-first-use only; compiler/driver caches may be warm",
            "performance_claim": "none",
        ]
        var records = [[String: Any]]()
        defer {
            receipt["raw_pairs"] = records
            do {
                try FileManager.default.createDirectory(
                    at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
                try JSONSerialization.data(
                    withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys]
                ).write(to: output, options: .withoutOverwriting)
                print("SPARK_INPUT_CAST_BENCH_RECEIPT \(path)")
            } catch { XCTFail("Could not retain benchmark receipt: \(error)") }
        }
        do {
            try MLXMetalTestLock.withLock {
                try MLX.Stream.withNewDefaultStream(device: .gpu) {
                    let stream = MLX.Stream()
                    let (g, u) = Spark25InputCastFixture.layers()
                    let normWeight = MLXArray.ones([2560], dtype: .bfloat16)
                    eval(g.parameters(), u.parameters(), normWeight)
                    var oldTimes = [Double]()
                    var newTimes = [Double]()
                    for round in -2 ..< 3 {
                        let count = round == -2 ? 1 : (round == -1 ? 5 : 30)
                        for index in 0 ..< count {
                            let seed = UInt64(2000 + (round + 2) * 100 + index)
                            let base = MLXFast.rmsNorm(
                                MLXRandom.normal([iterations, 1, 2560], key: MLXRandom.key(seed)),
                                weight: normWeight, eps: 1e-6)
                            let inputs = (0 ..< iterations).map { base[$0 ..< $0 + 1, 0..., 0...] }
                            eval(inputs)
                            stream.synchronize()
                            func old() throws -> Sample {
                                try measure(stream) {
                                    try inputs.map {
                                        guard
                                            let value = Spark25DualProjection.apply(
                                                $0.asType(.bfloat16), gate: g, up: u)
                                        else {
                                            throw Failure.invalid(
                                                "Existing BF16 fused baseline declined")
                                        }
                                        return value
                                    }
                                }
                            }
                            func candidate() throws -> Sample {
                                try measure(stream) {
                                    try inputs.map {
                                        guard
                                            let value = Spark25DualProjection.applyNormalizedFP32(
                                                $0, inputDType: .bfloat16, gate: g, up: u)
                                        else {
                                            throw Failure.invalid("New FP32 input cast declined")
                                        }
                                        return value
                                    }
                                }
                            }
                            let first = (index + max(round, 0)).isMultiple(of: 2)
                            let a: Sample
                            let b: Sample
                            if first {
                                a = try candidate()
                                b = try old()
                            } else {
                                b = try old()
                                a = try candidate()
                            }
                            records.append([
                                "round": round, "sample": index, "seed": seed,
                                "phase": round == -2
                                    ? "first_use" : (round == -1 ? "warmup" : "measured"),
                                "order": first
                                    ? ["candidate", "old_cast_fused"]
                                    : ["old_cast_fused", "candidate"],
                                "candidate_ns": a.nanoseconds, "old_cast_fused_ns": b.nanoseconds,
                                "exact_bits": false,
                            ])
                            for (actual, expected) in zip(a.outputs, b.outputs) {
                                guard actual.dtype == .bfloat16, expected.dtype == .bfloat16,
                                    actual.shape == expected.shape,
                                    actual.view(dtype: .uint16).asArray(UInt16.self)
                                        == expected.view(dtype: .uint16).asArray(UInt16.self)
                                else {
                                    throw Failure.invalid(
                                        "Exact output mismatch round\(round) sample\(index)")
                                }
                            }
                            records[records.count - 1]["exact_bits"] = true
                            if round >= 0 {
                                newTimes.append(Double(a.nanoseconds) / Double(iterations))
                                oldTimes.append(Double(b.nanoseconds) / Double(iterations))
                            }
                        }
                    }
                    receipt["candidate_summary"] = stats(newTimes)
                    receipt["old_cast_fused_summary"] = stats(oldTimes)
                }
            }
            receipt["status"] = "completed_exact_bits; full-model proof still required"
        } catch {
            receipt["status"] = "failed"
            receipt["error"] = String(describing: error)
            throw error
        }
    }
}
