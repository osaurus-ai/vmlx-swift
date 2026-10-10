import CryptoKit
import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// IFM Uno self-speculation (`Uno.swift`) on a tiny K2 Horizon target with a synthetic adapter.
/// Greedy must equal ordinary decoding token for token; sampled must match the target's own
/// distribution; the cache must end exactly where ordinary decoding leaves it.
final class UnoSelfSpeculationTests: XCTestCase {
    private let layers = 2

    private func configuration() throws -> K2HorizonConfiguration {
        let root: [String: Any] = [
            "model_type": "k2_horizon", "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": layers, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 8, "vocab_size": 64, "layernorm_num_groups": 4, "rms_norm_eps": 1e-6,
            "rope_parameters": ["rope_theta": 10_000_000, "rope_type": "default"],
        ]
        return try JSONDecoder().decode(
            K2HorizonConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
    }

    /// Random float32 weights (seeded): no exact logit ties to blur greedy comparisons.
    private func model() throws -> K2HorizonModel {
        let model = try K2HorizonModel(configuration())
        MLXRandom.seed(11)
        var weights: [String: MLXArray] = [:]
        for (name, parameter) in model.parameters().flattened() {
            let isNorm = name.contains("layernorm") || name == "model.norm.weight"
            weights[name] = isNorm
                ? 1 + 0.1 * MLXRandom.normal(parameter.shape)
                : 0.35 * MLXRandom.normal(parameter.shape)
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model.parameters())
        return model
    }

    /// A PEFT adapter directory for the tiny target (rank 4, alpha 8).
    private func adapterDirectory(dropping: String? = nil) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("uno-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let config: [String: Any] = [
            "r": 4, "lora_alpha": 8, "peft_type": "LORA",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ]
        try JSONSerialization.data(withJSONObject: config)
            .write(to: dir.appendingPathComponent("adapter_config.json"))
        let shapes: [(String, Int, Int)] = [
            ("self_attn.q_proj", 32, 32), ("self_attn.k_proj", 32, 16), ("self_attn.v_proj", 32, 16),
            ("self_attn.o_proj", 32, 32), ("mlp.gate_proj", 32, 64), ("mlp.up_proj", 32, 64),
            ("mlp.down_proj", 64, 32),
        ]
        MLXRandom.seed(5)
        var arrays: [String: MLXArray] = [:]
        for layer in 0 ..< layers {
            for (name, input, output) in shapes {
                let base = "base_model.model.model.layers.\(layer).\(name)"
                arrays[base + ".lora_A.weight"] = 0.3 * MLXRandom.normal([4, input])
                if base + ".lora_B.weight" != dropping {
                    arrays[base + ".lora_B.weight"] = 0.3 * MLXRandom.normal([output, 4])
                }
            }
        }
        try save(arrays: arrays, url: dir.appendingPathComponent("adapter_model.safetensors"))
        return dir
    }

    private let prompt = [3, 17, 42, 9, 27, 51]

    private func generate(
        _ model: K2HorizonModel, parameters: GenerateParameters, count: Int,
        stop: Set<Int> = []
    ) throws -> (tokens: [Int], iterator: TokenIterator) {
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), model: model, parameters: parameters,
            stopTokenIDs: stop)
        var tokens: [Int] = []
        while tokens.count < count, let token = iterator.next() {
            tokens.append(token)
            if stop.contains(token) { break }
        }
        return (tokens, iterator)
    }

    func testAdapterLoadsInstallsAndFailsClosedWhenIncomplete() throws {
        let model = try model()
        let adapter = try UnoAdapter.load(from: adapterDirectory())
        XCTAssertEqual(adapter.rank, 4)
        XCTAssertEqual(adapter.scale, 2)
        XCTAssertEqual(adapter.projections.count, 7 * layers)
        let mask = try model.installUnoAdapter(adapter)
        XCTAssertTrue(try model.installUnoAdapter(adapter) === mask, "install is idempotent")
        XCTAssertThrowsError(
            try UnoAdapter.load(
                from: adapterDirectory(dropping: "base_model.model.model.layers.1.mlp.down_proj.lora_B.weight")))
    }

    /// A converted (AWQ) target declares input scales in `uno_config.json`; they fold into LoRA A's
    /// input columns for q/k/v and gate/up only. A declared file with the wrong hash, or one missing
    /// a layer, fails the load.
    func testAWQInputScalesFoldIntoLoRAAOnly() throws {
        let dir = try adapterDirectory()
        let plain = try UnoAdapter.load(from: dir)
        MLXRandom.seed(9)
        var scales: [String: MLXArray] = [:]
        for layer in 0 ..< layers {
            scales["layers.\(layer).attn_in"] = 0.5 + MLXRandom.uniform(0 ..< 1, [32])
            scales["layers.\(layer).ffn_in"] = 0.5 + MLXRandom.uniform(0 ..< 1, [32])
        }
        let scalesURL = dir.appendingPathComponent("target_input_scales.safetensors")
        try save(arrays: scales, url: scalesURL)
        func writeConfig(sha: String) throws {
            let config: [String: Any] = [
                "input_scales": "target_input_scales.safetensors", "input_scales_sha256": sha,
            ]
            try JSONSerialization.data(withJSONObject: config)
                .write(to: dir.appendingPathComponent("uno_config.json"))
        }
        let digest = SHA256.hash(data: try Data(contentsOf: scalesURL))
            .map { String(format: "%02x", $0) }.joined()
        try writeConfig(sha: digest)
        let folded = try UnoAdapter.load(from: dir)
        for layer in 0 ..< layers {
            let base = "model.layers.\(layer)."
            for (name, key) in [
                ("self_attn.q_proj", "attn_in"), ("self_attn.k_proj", "attn_in"),
                ("self_attn.v_proj", "attn_in"), ("mlp.gate_proj", "ffn_in"), ("mlp.up_proj", "ffn_in"),
            ] {
                let expected = (plain.projections[base + name]!.aT.asType(.float32)
                    * scales["layers.\(layer).\(key)"]!.reshaped(-1, 1)).asType(.bfloat16)
                let actual = folded.projections[base + name]!.aT
                XCTAssertLessThanOrEqual(
                    abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self),
                    abs(expected.asType(.float32)).max().item(Float.self) / 128, base + name)
                XCTAssertGreaterThan(
                    abs(actual.asType(.float32) - plain.projections[base + name]!.aT.asType(.float32))
                        .max().item(Float.self), 0, base + name + " unchanged")
            }
            for name in ["self_attn.o_proj", "mlp.down_proj"] {
                XCTAssertEqual(
                    folded.projections[base + name]!.aT.asArray(Float.self),
                    plain.projections[base + name]!.aT.asArray(Float.self), base + name)
            }
        }
        try writeConfig(sha: String(repeating: "0", count: 64))
        XCTAssertThrowsError(try UnoAdapter.load(from: dir), "hash mismatch must fail")
        scales["layers.1.ffn_in"] = nil
        try save(arrays: scales, url: scalesURL)
        let shortDigest = SHA256.hash(data: try Data(contentsOf: scalesURL))
            .map { String(format: "%02x", $0) }.joined()
        try writeConfig(sha: shortDigest)
        XCTAssertThrowsError(try UnoAdapter.load(from: dir), "missing layer scale must fail")
    }

    /// With no pinned width the chooser runs: it times every candidate (AR stretches included)
    /// and the response still has the requested length.
    func testAdaptiveWidthTimesEveryCandidate() throws {
        let model = try model()
        var p = GenerateParameters(maxTokens: 400, temperature: 1, topP: 1)
        p.draftStrategy = .uno(adapterPath: try adapterDirectory(), blockSize: nil)
        p.randomSeed = 21
        let run = try generate(model, parameters: p, count: 400)
        XCTAssertEqual(run.tokens.count, 400)
        let stats = try XCTUnwrap(run.iterator.unoGenerationStats)
        for width in UnoWidthPlan.candidates where width > 1 {
            XCTAssertGreaterThan(stats.cyclesByWidth[width] ?? 0, 0, "width \(width) never ran: \(stats)")
        }
        XCTAssertGreaterThan(stats.arSteps, 0, "the AR option was never timed: \(stats)")
    }

    /// The conditional LoRA must leave every forward without a row mask untouched.
    func testInstalledAdapterDoesNotChangeOrdinaryForward() throws {
        let model = try model()
        let input = MLXArray(prompt).reshaped(1, prompt.count)
        let before = model(input, cache: nil)
        eval(before)
        _ = try model.installUnoAdapter(UnoAdapter.load(from: adapterDirectory()))
        let after = model(input, cache: nil)
        XCTAssertEqual(before.asArray(Float.self), after.asArray(Float.self))
    }

    /// Ordinary greedy decoding with the top-2 logit margin at every step.
    private func greedyWithMargins(_ model: K2HorizonModel, count: Int) -> (tokens: [Int], margins: [Float]) {
        let cache = model.newCache(parameters: nil)
        var logits = model(MLXArray(prompt).reshaped(1, prompt.count), cache: cache)[0, -1]
        var tokens: [Int] = []
        var margins: [Float] = []
        for _ in 0 ..< count {
            let top = MLX.sorted(logits)[(-2)...].asArray(Float.self)
            tokens.append(argMax(logits).item(Int.self))
            margins.append(top[1] - top[0])
            logits = model(MLXArray([tokens.last!]).reshaped(1, 1), cache: cache)[0, -1]
        }
        return (tokens, margins)
    }

    /// Greedy Uno cycles commit exactly the target's argmax tokens. Driven through `UnoDecoder`
    /// directly: the token iterator declines greedy Uno on K2 (next test), whose bundle requires
    /// greedy output to equal ordinary decoding bit for bit. The only tolerated difference is the
    /// device's row-count rounding: on NAX GPUs MLX runs float32 multi-row matmuls as TF32 and 1-row
    /// products as float32, so a multi-row forward and ordinary decoding can split a near-tie
    /// differently (this tiny float32 model: incremental vs full-forward logits differ by up to
    /// 5e-2; the one divergence seen was at a margin of 0.014). A divergence must sit on such a tie
    /// after a long exact stretch; a cache or offset fault breaks decisive positions and fails.
    func testGreedyUnoCyclesEqualOrdinaryDecoding() throws {
        let model = try model()
        let (reference, margins) = greedyWithMargins(model, count: 80)
        var ar = GenerateParameters(maxTokens: 80, temperature: 0)
        ar.draftStrategy = nil
        XCTAssertEqual(try generate(model, parameters: ar, count: 80).tokens, reference)
        let mask = try model.installUnoAdapter(UnoAdapter.load(from: adapterDirectory()))
        for block in [2, 5, 12] {
            let cache = model.newCache(parameters: nil)
            let first = model(MLXArray(prompt).reshaped(1, prompt.count), cache: cache)[0, -1]
            var decoder = UnoDecoder(
                model: model, mask: mask, blockSize: block, maskTokenBound: model.unoNoiseTokenBound,
                temperature: 0, topP: 1, topK: 0, stopTokenIDs: [])
            var tokens = [argMax(first).item(Int.self)]
            while tokens.count < reference.count {
                let committed = decoder.cycle(seed: tokens.last!, cache: cache, maxCommit: 1 << 20)
                XCTAssertGreaterThanOrEqual(committed.count, 2)
                // The cache holds the prompt, the seed and every committed token but the last.
                XCTAssertEqual(cache[0].offset, prompt.count + tokens.count + committed.count - 1)
                tokens += committed
            }
            tokens = Array(tokens.prefix(reference.count))
            XCTAssertGreaterThanOrEqual(decoder.stats.tokensPerCycle, 2, "block \(block)")
            guard let d = tokens.indices.first(where: { tokens[$0] != reference[$0] }) else { continue }
            XCTAssertGreaterThanOrEqual(d, 40, "block \(block): early divergence at \(d)")
            XCTAssertLessThan(margins[d], 0.1, "block \(block): diverged at \(d) on a decisive argmax")
        }
    }

    /// K2 requires greedy output identical to ordinary decoding, which multi-row forwards cannot
    /// promise, so a greedy request decodes autoregressively (and still returns the AR tokens).
    func testGreedyRequestOnK2DecodesAutoregressively() throws {
        let model = try model()
        var ar = GenerateParameters(maxTokens: 30, temperature: 0)
        ar.draftStrategy = nil
        let reference = try generate(model, parameters: ar, count: 30).tokens
        var p = GenerateParameters(maxTokens: 30, temperature: 0)
        p.draftStrategy = .uno(adapterPath: try adapterDirectory(), blockSize: 5)
        let run = try generate(model, parameters: p, count: 30)
        XCTAssertNil(run.iterator.unoGenerationStats)
        XCTAssertEqual(run.tokens, reference)
    }

    /// A cycle can commit past a stop token; the caller stops there. The cache must then hold
    /// exactly the prompt plus every returned token, as ordinary decoding leaves it. Sampled with a
    /// request seed (which Uno honours: the same seed reproduces the run), the stop token taken
    /// from an unrestricted run.
    func testStopTokenMidCycleLeavesOrdinaryCacheState() throws {
        let model = try model()
        let dir = try adapterDirectory()
        for (seed, block) in [(UInt64(3), 12), (UInt64(4), 5), (UInt64(5), 8)] {
            var p = GenerateParameters(maxTokens: 40, temperature: 1, topP: 1)
            p.draftStrategy = .uno(adapterPath: dir, blockSize: block)
            p.randomSeed = seed
            let free = try generate(model, parameters: p, count: 40).tokens
            let stop = free[9]
            var run = try generate(model, parameters: p, count: 40, stop: [stop])
            XCTAssertEqual(run.tokens, Array(free[...free.firstIndex(of: stop)!]), "seed \(seed)")
            XCTAssertNotNil(run.iterator.unoGenerationStats)
            run.iterator.storeCacheAfterGeneration(
                generatedTokenIds: run.tokens, includeGeneratedBoundary: true)
            XCTAssertEqual(
                run.iterator.cache.map(\.offset),
                Array(repeating: prompt.count + run.tokens.count, count: layers), "seed \(seed)")
        }
        // The same formula on ordinary decoding (the healthy reference for the invariant).
        var ar = GenerateParameters(maxTokens: 40, temperature: 0)
        ar.draftStrategy = nil
        let free = try generate(model, parameters: ar, count: 40).tokens
        var arRun = try generate(model, parameters: ar, count: 40, stop: [free[9]])
        arRun.iterator.storeCacheAfterGeneration(
            generatedTokenIds: arRun.tokens, includeGeneratedBoundary: true)
        XCTAssertEqual(
            arRun.iterator.cache.map(\.offset),
            Array(repeating: prompt.count + arRun.tokens.count, count: layers))
    }

    /// Exact marginal of the third generated token under the target (brute force over 64² histories),
    /// with the same temperature / top-p filtering the samplers apply.
    private func exactThirdTokenMarginal(_ model: K2HorizonModel, topP: Float) -> [Double] {
        func filtered(_ logits: MLXArray) -> MLXArray {
            DFlash2Sampling.probabilities(logits, temperature: 1, topP: topP)
        }
        let v = 64
        let p1 = filtered(model(MLXArray(prompt).reshaped(1, prompt.count), cache: nil)[0, -1])
        let firstRows = MLXArray((0 ..< v).flatMap { prompt + [$0] }).reshaped(v, prompt.count + 1)
        let p2 = filtered(model(firstRows, cache: nil)[0..., -1])  // (v, v): p(t2 | t1)
        let pairs = (0 ..< v).flatMap { a in (0 ..< v).map { b in prompt + [a, b] } }
        let secondRows = MLXArray(pairs.flatMap { $0 }).reshaped(v * v, prompt.count + 2)
        let p3 = filtered(model(secondRows, cache: nil)[0..., -1]).reshaped(v, v, v)
        let joint = p1.reshaped(v, 1, 1) * p2.reshaped(v, v, 1) * p3
        return joint.sum(axes: [0, 1]).asType(.float32).asArray(Float.self).map(Double.init)
    }

    private func totalVariation(_ counts: [Int], _ exact: [Double]) -> Double {
        let n = Double(counts.reduce(0, +))
        return 0.5 * zip(counts, exact).map { abs(Double($0) / n - $1) }.reduce(0, +)
    }

    /// Sampled Uno is lossless: the third token's empirical distribution matches the exact target
    /// marginal as closely as ordinary sampling does (same sample size).
    func testSampledUnoMatchesTargetDistribution() throws {
        let model = try model()
        let dir = try adapterDirectory()
        let samples = 4000
        for topP: Float in [1.0, 0.9] {
            let exact = exactThirdTokenMarginal(model, topP: topP)
            var arCounts = [Int](repeating: 0, count: 64)
            var unoCounts = [Int](repeating: 0, count: 64)
            var cycles = 0
            for i in 0 ..< samples {
                MLXRandom.seed(UInt64(10_000 + i))
                var ar = GenerateParameters(maxTokens: 8, temperature: 1, topP: topP)
                ar.draftStrategy = nil
                arCounts[try generate(model, parameters: ar, count: 3).tokens[2]] += 1
                MLXRandom.seed(UInt64(90_000 + i))
                var p = GenerateParameters(maxTokens: 8, temperature: 1, topP: topP)
                p.draftStrategy = .uno(adapterPath: dir, blockSize: 5)
                let run = try generate(model, parameters: p, count: 3)
                cycles += run.iterator.unoGenerationStats?.cycles ?? 0
                unoCounts[run.tokens[2]] += 1
            }
            XCTAssertGreaterThanOrEqual(cycles, samples, "Uno did not run (top-p \(topP))")
            let arTV = totalVariation(arCounts, exact)
            let unoTV = totalVariation(unoCounts, exact)
            print("UNO_DISTRIBUTION topP=\(topP) tv_ar=\(arTV) tv_uno=\(unoTV)")
            XCTAssertLessThan(unoTV, 0.08, "top-p \(topP)")
            XCTAssertLessThan(unoTV, arTV + 0.03, "top-p \(topP): Uno farther from the target than AR sampling")
        }
    }
}
