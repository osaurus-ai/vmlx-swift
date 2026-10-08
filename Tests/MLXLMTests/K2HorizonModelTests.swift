import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class K2HorizonModelTests: XCTestCase {
    private func configuration(_ changes: [String: Any] = [:]) throws -> K2HorizonConfiguration {
        var root: [String: Any] = [
            "model_type": "k2_horizon", "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 8, "vocab_size": 64, "layernorm_num_groups": 4, "rms_norm_eps": 1e-6,
            "rope_parameters": ["rope_theta": 10_000_000, "rope_type": "default"],
        ]
        root.merge(changes) { _, value in value }
        return try JSONDecoder().decode(
            K2HorizonConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
    }

    func testUnsupportedArchitectureFailsBeforeModelConstruction() throws {
        let changes: [[String: Any]] = [
            ["model_type": "llama"], ["mova_num_experts": 1], ["num_experts": 2],
            ["query_key_norm": true], ["attention_gate_func": "sigmoid"], ["attention_bias": true],
            ["rope_head_dim": 4], ["use_sliding_window": true], ["layernorm_num_groups": 3],
            ["layernorm_num_groups": 0], ["num_key_value_heads": 3], ["num_key_value_heads": 0],
            ["head_dim": 7], ["num_hidden_layers": 0], ["rms_norm_eps": 0],
            ["mlp_layout": "unknown"], ["rope_parameters": ["rope_type": "linear"]],
            ["rope_parameters": ["rope_type": 3]], ["rope_parameters": ["rope_theta": "bad"]],
            ["rope_parameters": ["rope_theta": -1]],
        ]
        for change in changes {
            XCTAssertThrowsError(try configuration(change), "accepted \(change)")
        }
    }

    func testNativeActivationAndFullAttentionConfigContract() throws {
        for layout in ["dense", "switch1", "dense_jangh_down"] {
            // Missing fields retain the native default; explicit null is valid
            // only for sliding_window, matching the vendor full-attention test.
            XCTAssertNoThrow(try configuration(["mlp_layout": layout]))
            XCTAssertNoThrow(
                try configuration([
                    "mlp_layout": layout, "hidden_act": "silu",
                    "sliding_window": NSNull(), "use_sliding_window": false,
                ]))
            let unsupportedActivations: [Any] = ["gelu", "relu", "", NSNull(), 7]
            for activation in unsupportedActivations {
                XCTAssertThrowsError(
                    try configuration([
                        "mlp_layout": layout, "hidden_act": activation,
                    ]), "accepted non-native activation \(activation) in \(layout)")
            }
            // The vendor chooses sliding attention from this field even when
            // use_sliding_window is absent or false. Neither may bypass rejection.
            for window in [0, 128] {
                XCTAssertThrowsError(
                    try configuration([
                        "mlp_layout": layout, "sliding_window": window,
                    ]))
                XCTAssertThrowsError(
                    try configuration([
                        "mlp_layout": layout, "sliding_window": window,
                        "use_sliding_window": false,
                    ]))
            }
        }
    }

    func testGroupedNormUsesIndependentContiguousGroups() throws {
        try MLXMetalTestLock.withLock {
            let values: [Float] = [1, 2, 100, 200, -3, 4, -30, 40]
            let weights: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
            let norm = K2GroupedRMSNorm(dimensions: 8, groups: 4, eps: 1e-6)
            try norm.update(
                parameters: ModuleParameters.unflattened(["weight": MLXArray(weights)]),
                verify: [.all])
            let actual = norm(MLXArray(values, [1, 1, 8])).asArray(Float.self)
            for i in values.indices {
                let start = i / 2 * 2
                let denominator = sqrt(
                    (values[start] * values[start] + values[start + 1] * values[start + 1]) / 2
                        + 1e-6)
                XCTAssertEqual(actual[i], values[i] / denominator * weights[i], accuracy: 2e-5)
            }
        }
    }

    /// The cached unit weight must not change a single bit versus building
    /// `ones` per call (the previous implementation).
    func testGroupedNormCachedUnitWeightIsBitIdentical() throws {
        try MLXMetalTestLock.withLock {
            for dtype in [DType.bfloat16, .float32] {
                let norm = K2GroupedRMSNorm(dimensions: 4096, groups: 4, eps: 1e-6)
                let weight = MLXRandom.normal([4096], key: MLXRandom.key(7)).asType(dtype)
                try norm.update(
                    parameters: ModuleParameters.unflattened(["weight": weight]), verify: [.all])
                for rows in [1, 3] {
                    let x = (MLXRandom.normal([1, rows, 4096], key: MLXRandom.key(UInt64(rows))) * 4)
                        .asType(dtype)
                    let reference = MLXFast.rmsNorm(
                        x.reshaped([1, rows, 4, 1024]),
                        weight: MLXArray.ones([1024], dtype: dtype), eps: 1e-6
                    ).reshaped(x.shape) * weight
                    for _ in 0 ..< 2 {  // first call fills the cache, second reuses it
                        let actual = norm(x)
                        XCTAssertEqual(actual.dtype, reference.dtype)
                        XCTAssertTrue(
                            MLX.all(actual .== reference).item(Bool.self),
                            "grouped norm differs for \(dtype) rows=\(rows)")
                    }
                }
            }
        }
    }

    /// Cold grouped normalization through the unscoped compile closure used by
    /// BatchCompile.compileForward. No checkpoint or model allocation is needed.
    func testColdGroupedNormBatchCompileTrace() throws {
        try MLXMetalTestLock.withLock {
            let norm = K2GroupedRMSNorm(dimensions: 8, groups: 4, eps: 1e-6)
            let x = MLXArray([Float(1), 2, 3, 4, 5, 6, 7, 8], [1, 1, 8])
            eval(x, norm.weight)
            // Do not run norm eagerly first: that would conceal a cold-cache
            // eval during tracing by warming the dtype/count entry.
            let forward: @Sendable ([MLXArray]) -> [MLXArray] = compile {
                (args: [MLXArray]) -> [MLXArray] in [norm(args[0])]
            }
            let actual = forward([x])[0]
            eval(actual)
            let expected = MLXFast.rmsNorm(
                x.reshaped([1, 1, 4, 2]), weight: MLXArray.ones([2]), eps: 1e-6
            ).reshaped(x.shape) * norm.weight
            XCTAssertEqual(actual.shape, x.shape)
            XCTAssertTrue(MLX.all(actual .== expected).item(Bool.self))
        }
    }

    private func deterministicModel(dtype: DType = .float32) throws -> K2HorizonModel {
        let model = try K2HorizonModel(configuration())
        var weights: [String: MLXArray] = [:]
        for (name, parameter) in model.parameters().flattened() {
            let seed: Int = name.utf8.reduce(0) { $0 + Int($1) }
            let isNorm = name.contains("layernorm") || name == "model.norm.weight"
            var values = [Float]()
            values.reserveCapacity(parameter.size)
            for index in 0 ..< parameter.size {
                let centered: Int = (index * 7 + seed) % 31 - 15
                let value = Float(centered) / 128
                values.append(isNorm ? 1 + value : value)
            }
            weights[name] = MLXArray(values, parameter.shape).asType(dtype)
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model.parameters())
        XCTAssertTrue(model.parameters().flattenedValues().allSatisfy { $0.dtype == dtype })
        return model
    }

    /// Independent Double scalar decoder for this tiny fixture. No MLX matmul,
    /// attention, rotary, normalization, cache or sampling primitive is reused.
    private func scalarReference(model: K2HorizonModel, ids: [Int]) -> [Float] {
        let tensors = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map {
                ($0.0, $0.1.asType(.float32).asArray(Float.self).map(Double.init))
            })
        func linear(_ x: [Double], _ path: String, outputs: Int) -> [Double] {
            let w = tensors[path + ".weight"]!
            return (0 ..< outputs).map { row in
                var sum = 0.0
                for k in x.indices { sum += x[k] * w[row * x.count + k] }
                return sum
            }
        }
        func norm(_ x: [Double], _ path: String) -> [Double] {
            let w = tensors[path + ".weight"]!
            var y = x
            for group in 0 ..< 4 {
                let start = group * 8
                var squareSum = 0.0
                for k in start ..< (start + 8) { squareSum += x[k] * x[k] }
                let denom = sqrt(squareSum / 8 + 1e-6)
                for k in start ..< (start + 8) { y[k] = x[k] / denom * w[k] }
            }
            return y
        }
        func rotary(_ x: [Double], position: Int) -> [Double] {
            var y = x
            for head in 0 ..< (x.count / 8) {
                for pair in 0 ..< 4 {
                    let angle = Double(position) * pow(10_000_000.0, -Double(pair) / 4)
                    let lo = head * 8 + pair
                    let hi = lo + 4
                    y[lo] = x[lo] * cos(angle) - x[hi] * sin(angle)
                    y[hi] = x[lo] * sin(angle) + x[hi] * cos(angle)
                }
            }
            return y
        }
        let embeddings = tensors["model.embed_tokens.weight"]!
        var hidden = ids.map { Array(embeddings[($0 * 32) ..< ($0 * 32 + 32)]) }
        for layer in 0 ..< 2 {
            let path = "model.layers.\(layer)"
            let normal = hidden.map { norm($0, path + ".input_layernorm") }
            let queries = normal.enumerated().map {
                rotary(
                    linear($0.element, path + ".self_attn.q_proj", outputs: 32), position: $0.offset
                )
            }
            let keys = normal.enumerated().map {
                rotary(
                    linear($0.element, path + ".self_attn.k_proj", outputs: 16), position: $0.offset
                )
            }
            let values = normal.map { linear($0, path + ".self_attn.v_proj", outputs: 16) }
            for token in ids.indices {
                var attention = [Double](repeating: 0, count: 32)
                for head in 0 ..< 4 {
                    let kvHead = head / 2
                    var scores = [Double]()
                    for past in 0 ... token {
                        var score = 0.0
                        for d in 0 ..< 8 {
                            score += queries[token][head * 8 + d] * keys[past][kvHead * 8 + d]
                        }
                        scores.append(score / sqrt(8))
                    }
                    let maximum = scores.max()!
                    let probabilities = scores.map { exp($0 - maximum) }
                    let denominator = probabilities.reduce(0, +)
                    for past in 0 ... token {
                        for d in 0 ..< 8 {
                            attention[head * 8 + d] +=
                                probabilities[past] / denominator * values[past][kvHead * 8 + d]
                        }
                    }
                }
                let projected = linear(attention, path + ".self_attn.o_proj", outputs: 32)
                for d in 0 ..< 32 { hidden[token][d] += projected[d] }
                let post = norm(hidden[token], path + ".post_attention_layernorm")
                let gate = linear(post, path + ".mlp.gate_proj", outputs: 64)
                let up = linear(post, path + ".mlp.up_proj", outputs: 64)
                let activation = (0 ..< 64).map { gate[$0] / (1 + exp(-gate[$0])) * up[$0] }
                let down = linear(activation, path + ".mlp.down_proj", outputs: 32)
                for d in 0 ..< 32 { hidden[token][d] += down[d] }
            }
        }
        return hidden.flatMap {
            linear(norm($0, "model.norm"), "lm_head", outputs: 64).map(Float.init)
        }
    }

    // The precise arithmetic oracle runs on CPU: default GPU F32 matmul may
    // intentionally use TF32. A matched fixed-weight control measured max error
    // 0.0009743571 with TF32 vs 2.3841858e-7 without it. This is a precision
    // selection for the test, not a cache or production arithmetic change.
    func testSplitPrefillAndDecodeMatchFullForward() throws {
        try MLXMetalTestLock.withLock {
            try Device.withDefaultDevice(.cpu) {
                try checkPreciseSplitReference()
            }
        }
    }

    /// Explicit replay in a fresh process, e.g. K2_GPU_F32_REFERENCE=1
    /// MLX_ENABLE_TF32=0. The test never changes global TF32 settings itself.
    func testOptInGPUFloat32ScalarReference() throws {
        guard ProcessInfo.processInfo.environment["K2_GPU_F32_REFERENCE"] == "1" else {
            throw XCTSkip("Set K2_GPU_F32_REFERENCE=1 for the explicit GPU precision diagnostic")
        }
        try MLXMetalTestLock.withLock {
            try Device.withDefaultDevice(.gpu) {
                try checkPreciseSplitReference()
            }
        }
    }

    private func checkPreciseSplitReference() throws {
        let model = try deterministicModel()
        let tokens = MLXArray([Int32(1), 5, 7, 3, 11, 6, 2], [1, 7])
        let full = model(tokens, cache: nil)
        let cache = model.newCache(parameters: nil)
        XCTAssertEqual(cache.count, 2)
        XCTAssertTrue(cache.allSatisfy { $0 is KVCacheSimple })
        let a = model(tokens[0..., 0 ..< 3], cache: cache)
        let b = model(tokens[0..., 3 ..< 6], cache: cache)
        eval(a, b)
        let cacheFile = FileManager.default.temporaryDirectory.appendingPathComponent(
            "k2-cache-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: cacheFile) }
        try savePromptCache(url: cacheFile, cache: cache, metadata: ["case": "k2-boundary-6"])
        let (restored, metadata) = try loadPromptCache(url: cacheFile)
        XCTAssertEqual(metadata["case"], "k2-boundary-6")
        XCTAssertEqual(restored.map(\.metaState), cache.map(\.metaState))
        let c = model(tokens[0..., 6 ..< 7], cache: cache)
        let fromDisk = model(tokens[0..., 6 ..< 7], cache: restored)
        XCTAssertLessThan(abs(c - fromDisk).max().item(Float.self), 1e-6)
        let joined = concatenated([a, b, c], axis: 1)
        let reference = MLXArray(
            scalarReference(model: model, ids: [1, 5, 7, 3, 11, 6, 2]), [1, 7, 64])
        let fullError = abs(full - reference).max().item(Float.self)
        let cachedError = abs(joined - reference).max().item(Float.self)
        print(
            "K2_SCALAR_REFERENCE full_max_abs=\(fullError) cached_max_abs=\(cachedError) dtype=\(full.dtype) tf32=\(ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] ?? "unset")"
        )
        XCTAssertLessThan(fullError, 2e-4)
        XCTAssertLessThan(cachedError, 2e-4)
        XCTAssertLessThan(abs(full - joined).max().item(Float.self), 2e-4)
        XCTAssertTrue(cache.allSatisfy { $0.offset == 7 })
    }

    func testBF16DiskRestoredContinuationIsExactAtNonalignedBoundary() throws {
        try MLXMetalTestLock.withLock {
            let model = try deterministicModel(dtype: .bfloat16)
            let ids = (0 ..< 72).map { Int32(($0 * 7 + 3) % 64) }
            let tokens = MLXArray(ids, [1, ids.count])
            let cache = model.newCache(parameters: nil)
            let first = model(tokens[0..., 0 ..< 63], cache: cache)
            eval(first)
            let second = model(tokens[0..., 63 ..< 67], cache: cache)
            eval(second)
            XCTAssertTrue(cache.allSatisfy { $0.offset == 67 })
            XCTAssertTrue(cache.flatMap(\.state).allSatisfy { $0.dtype == .bfloat16 })
            let file = FileManager.default.temporaryDirectory.appendingPathComponent(
                "k2-bf16-\(UUID().uuidString).safetensors")
            defer { try? FileManager.default.removeItem(at: file) }
            try savePromptCache(url: file, cache: cache, metadata: ["case": "bf16-k2-boundary-67"])
            let (restored, metadata) = try loadPromptCache(url: file)
            XCTAssertEqual(metadata["case"], "bf16-k2-boundary-67")
            XCTAssertEqual(restored.map(\.metaState), cache.map(\.metaState))
            XCTAssertTrue(restored.flatMap(\.state).allSatisfy { $0.dtype == .bfloat16 })
            var offset = 67
            for count in [4, 1] {
                let input = tokens[0..., offset ..< (offset + count)]
                let live = model(input, cache: cache)
                let disk = model(input, cache: restored)
                eval(live, disk)
                XCTAssertEqual(live.dtype, .bfloat16)
                XCTAssertTrue(all(live .== disk).item(Bool.self))
                XCTAssertTrue(live.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
                offset += count
                XCTAssertTrue(restored.allSatisfy { $0.offset == offset })
            }
        }
    }

    func testDenseJANGHCannotConstructWithoutAdmittedBanks() throws {
        let c = try configuration(["mlp_layout": "dense_jangh_down"])
        XCTAssertThrowsError(try K2HorizonModel(c))
    }
}
