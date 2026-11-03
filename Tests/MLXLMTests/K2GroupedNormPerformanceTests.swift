import CmlxGraphShim
import Dispatch
import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM

/// Opt-in synthetic norm timing. No model, tokenizer, sampler or cache is loaded.
/// All samples include graph construction, eval and GPU completion; they are
/// not device-only kernel timings or end-to-end model measurements.
final class K2GroupedNormPerformanceTests: XCTestCase {
    private enum Failure: Error {
        case configuration(String)
        case admission(String)
        case mismatch(String)
    }

    private struct Measurement {
        let nanoseconds: UInt64
        let outputs: [MLXArray]
    }

    private func measure(
        _ inputs: [MLXArray], stream: MLX.Stream,
        operation: (MLXArray) throws -> MLXArray
    ) rethrows -> Measurement {
        stream.synchronize()
        let start = DispatchTime.now().uptimeNanoseconds
        let outputs = try inputs.map(operation)
        eval(outputs)
        stream.synchronize()
        return Measurement(
            nanoseconds: DispatchTime.now().uptimeNanoseconds - start, outputs: outputs)
    }

    private func inputs(iterations: Int, seed: UInt64, stream: MLX.Stream) -> [MLXArray] {
        // Distinct offsets contain independent random values. Build and eval
        // them before timing; each sample also receives a different seed.
        let base = MLXRandom.normal([iterations, 1, 4096], key: MLXRandom.key(seed))
            .asType(.bfloat16)
        let arrays = (0 ..< iterations).map { base[$0 ..< $0 + 1, 0..., 0...] }
        eval(arrays)
        stream.synchronize()
        return arrays
    }

    private func validate(_ a: Measurement, _ b: Measurement, label: String) throws {
        guard a.outputs.count == b.outputs.count else { throw Failure.mismatch(label + " count") }
        for (i, values) in zip(a.outputs, b.outputs).enumerated() {
            guard values.0.shape == values.1.shape, values.0.dtype == .bfloat16,
                values.1.dtype == .bfloat16
            else { throw Failure.mismatch("\(label) output \(i) shape/dtype") }
            let actual = values.0.view(dtype: .uint16).asArray(UInt16.self)
            let expected = values.1.view(dtype: .uint16).asArray(UInt16.self)
            // Inputs/weights are finite seeded values. Every BF16 output bit
            // must agree, including signed zero; do not hide small errors.
            guard actual == expected else { throw Failure.mismatch("\(label) output \(i) bits") }
        }
    }

    private func stats(_ values: [Double]) -> [String: Any] {
        let sorted = values.sorted()
        func percentile(_ p: Double) -> Double {
            let index = Double(sorted.count - 1) * p
            let lower = Int(index.rounded(.down))
            let upper = Int(index.rounded(.up))
            return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - Double(lower))
        }
        let q1 = percentile(0.25)
        let q3 = percentile(0.75)
        return [
            "count": sorted.count, "median_ns_per_norm": percentile(0.5),
            "q1_ns_per_norm": q1, "q3_ns_per_norm": q3,
            "iqr_ns_per_norm": q3 - q1, "min_ns_per_norm": sorted.first!,
            "max_ns_per_norm": sorted.last!,
        ]
    }

    func testOptInSingleRowNormBenchmark() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["VMLX_K2_NORM_BENCH"] == "1" else {
            throw XCTSkip("Set VMLX_K2_NORM_BENCH=1 for the diagnostic norm benchmark")
        }
        #if !canImport(Metal)
            throw XCTSkip("Metal required for the diagnostic norm benchmark")
        #else
            guard !K2GroupedNormFusion.referenceOverride else {
                throw Failure.configuration(
                    "Unset VMLX_K2_GROUPED_NORM_REFERENCE for paired timing")
            }
            let rawIterations = env["VMLX_K2_NORM_BENCH_ITERATIONS"] ?? "16"
            guard let iterations = Int(rawIterations), iterations > 0 else {
                throw Failure.configuration(
                    "VMLX_K2_NORM_BENCH_ITERATIONS must be a positive integer")
            }
            let output =
                env["VMLX_K2_NORM_BENCH_OUTPUT"].map { URL(fileURLWithPath: $0) }
                ?? FileManager.default.temporaryDirectory.appendingPathComponent(
                    "k2-grouped-norm-benchmark-\(UUID().uuidString).json")
            guard !FileManager.default.fileExists(atPath: output.path) else {
                throw Failure.configuration(
                    "Refusing to replace existing benchmark receipt: \(output.path)")
            }
            let warmups = 5
            let samplesPerRound = 30
            let rounds = 3
            var cold: [[String: Any]] = []
            var records: [[String: Any]] = []
            var receipt: [String: Any] = [
                "schema": 1, "started_utc": ISO8601DateFormatter().string(from: Date()),
                "host": ProcessInfo.processInfo.hostName, "status": "running",
                "scope": "synthetic single-row norm; no model inference",
                "shape": [1, 1, 4096], "dtype": "bfloat16", "groups": 4, "eps": 1e-6,
                "weight_seed": 313, "iterations_per_sample": iterations,
                "warmup_pairs_per_surface": warmups, "samples_per_round": samplesPerRound,
                "rounds": rounds,
                "timing_includes": ["graph construction", "eval", "stream completion"],
                "input_construction_timed": false,
                "cold_probe_scope":
                    "first use in this test process; persistent compiler/driver caches may already be warm",
                "physical_footprint":
                    "not measured by this harness; external process monitoring required",
                "runtime_source_identity": "bind externally to exact tested source and native core",
            ]
            defer {
                receipt["finished_utc"] = ISO8601DateFormatter().string(from: Date())
                receipt["first_use_probes"] = cold
                receipt["raw_samples"] = records
                do {
                    try FileManager.default.createDirectory(
                        at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
                    let data = try JSONSerialization.data(
                        withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys])
                    try data.write(to: output, options: .withoutOverwriting)
                    print("K2_NORM_BENCH_RECEIPT \(output.path)")
                } catch {
                    XCTFail("Could not preserve benchmark receipt: \(error)")
                }
            }
            do {
                try MLXMetalTestLock.withLock {
                    try MLX.Stream.withNewDefaultStream(device: .gpu) {
                        let stream = MLX.Stream()
                        let norm = K2GroupedRMSNorm(dimensions: 4096, groups: 4, eps: 1e-6)
                        let weight = (MLXRandom.normal([4096], key: MLXRandom.key(313)) * 0.1 + 1)
                            .asType(.bfloat16)
                        try norm.update(
                            parameters: ModuleParameters.unflattened(["weight": weight]),
                            verify: [.all])
                        let unitWeights = UnitWeightCache()
                        // Prepare the same cached unit weight used by the old
                        // expression. Its lookup remains inside native timing.
                        eval(norm.weight, unitWeights.ones(count: 1024, dtype: .bfloat16))
                        stream.synchronize()
                        func native(_ x: MLXArray) -> MLXArray {
                            var shape = x.shape
                            let dimensions = shape.removeLast()
                            return MLXFast.rmsNorm(
                                x.reshaped(shape + [4, dimensions / 4]),
                                weight: unitWeights.ones(count: dimensions / 4, dtype: x.dtype),
                                eps: 1e-6
                            ).reshaped(x.shape) * norm.weight
                        }
                        func helper(_ x: MLXArray) throws -> MLXArray {
                            guard
                                let result = K2GroupedNormFusion.apply(
                                    x, weight: norm.weight, groups: 4, eps: 1e-6)
                            else { throw Failure.admission("Eligible helper input declined") }
                            return result
                        }
                        let firstInputs = inputs(iterations: 1, seed: 317, stream: stream)
                        let firstNative = measure(firstInputs, stream: stream, operation: native)
                        cold.append([
                            "arm": "native_reference", "elapsed_ns": firstNative.nanoseconds,
                        ])
                        let firstHelper = try measure(
                            firstInputs, stream: stream, operation: helper)
                        cold.append(["arm": "fused_helper", "elapsed_ns": firstHelper.nanoseconds])
                        try validate(firstHelper, firstNative, label: "first helper")
                        let firstModule = measure(
                            firstInputs, stream: stream, operation: norm.callAsFunction)
                        cold.append([
                            "arm": "ordinary_module", "elapsed_ns": firstModule.nanoseconds,
                            "kernel_already_used_by_helper": true,
                        ])
                        try validate(firstModule, firstNative, label: "first module")
                        // Inspect a fresh, unevaluated output graph outside
                        // timed work; a numeric match alone cannot prove use.
                        func nodes(_ value: MLXArray) throws -> Int32 {
                            var count: Int32 = -1
                            var casts: Int32 = -1
                            guard vmlx_graph_stats(value.ctx.ctx, &count, &casts) == 0 else {
                                throw Failure.admission("Graph inspection failed")
                            }
                            return count
                        }
                        let directNodes = try nodes(helper(firstInputs[0]))
                        let moduleNodes = try nodes(norm(firstInputs[0]))
                        let nativeNodes = try nodes(native(firstInputs[0]))
                        receipt["dispatch_graph_nodes"] = [
                            "fused_helper": directNodes, "ordinary_module": moduleNodes,
                            "native_reference": nativeNodes,
                        ]
                        guard directNodes > 0, moduleNodes == directNodes, nativeNodes > directNodes
                        else {
                            throw Failure.admission(
                                "Ordinary module graph does not establish fused dispatch")
                        }
                        for (surfaceIndex, surface) in ["fused_helper", "ordinary_module"]
                            .enumerated()
                        {
                            var candidateTimes = [Double]()
                            var nativeTimes = [Double]()
                            var summariesByRound = [[String: Any]]()
                            for round in -1 ..< rounds {
                                var roundCandidate = [Double]()
                                var roundNative = [Double]()
                                let count = round < 0 ? warmups : samplesPerRound
                                for sample in 0 ..< count {
                                    let seed = UInt64(
                                        1000 + surfaceIndex * 10000 + (round + 1) * 100 + sample)
                                    let values = inputs(
                                        iterations: iterations, seed: seed, stream: stream)
                                    let candidateFirst = (sample + max(round, 0)).isMultiple(of: 2)
                                    func candidate() throws -> Measurement {
                                        if surface == "fused_helper" {
                                            return try measure(
                                                values, stream: stream, operation: helper)
                                        }
                                        return measure(
                                            values, stream: stream, operation: norm.callAsFunction)
                                    }
                                    let a: Measurement
                                    let b: Measurement
                                    if candidateFirst {
                                        a = try candidate()
                                        b = measure(values, stream: stream, operation: native)
                                    } else {
                                        b = measure(values, stream: stream, operation: native)
                                        a = try candidate()
                                    }
                                    var record: [String: Any] = [
                                        "surface": surface,
                                        "phase": round < 0 ? "warmup" : "measured",
                                        "round": round, "sample": sample, "input_seed": seed,
                                        "order": candidateFirst
                                            ? ["candidate", "native"] : ["native", "candidate"],
                                        "iterations": iterations,
                                        "candidate_elapsed_ns": a.nanoseconds,
                                        "native_elapsed_ns": b.nanoseconds, "bit_exact": false,
                                    ]
                                    // Store durations before checking outputs so
                                    // a failing pair survives in the receipt.
                                    records.append(record)
                                    try validate(
                                        a, b, label: "\(surface) round\(round) sample\(sample)")
                                    record["bit_exact"] = true
                                    records[records.count - 1] = record
                                    if round >= 0 {
                                        roundCandidate.append(
                                            Double(a.nanoseconds) / Double(iterations))
                                        roundNative.append(
                                            Double(b.nanoseconds) / Double(iterations))
                                    }
                                }
                                if round >= 0 {
                                    candidateTimes += roundCandidate
                                    nativeTimes += roundNative
                                    summariesByRound.append([
                                        "round": round, "candidate": stats(roundCandidate),
                                        "native": stats(roundNative),
                                    ])
                                }
                            }
                            receipt[surface + "_summary"] = [
                                "candidate": stats(candidateTimes), "native": stats(nativeTimes),
                                "by_round": summariesByRound,
                            ]
                        }
                    }
                }
                receipt["status"] = "completed_with_exact_outputs"
                receipt["performance_claim"] =
                    "none; inspect raw measurements and obtain full-model proof separately"
            } catch {
                receipt["status"] = "failed"
                receipt["failure"] = String(describing: error)
                throw error
            }
        #endif
    }
}
