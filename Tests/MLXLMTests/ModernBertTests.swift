// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXEmbedders

/// Tier 1 of the ModernBERT port's verification: a four-layer model with unit-gain random weights,
/// compared against Hugging Face's outputs for the same weights. Tier 2, the comparison against the
/// real checkpoints, lives outside this repository. Regenerate the fixture with
/// `scripts/modernbert-oracle.py tier1` (transformers 4.57.3); its JSON records the generator's
/// hashes and the conditions it ran under.
struct ModernBertTests {

    @Test(
        "absent fields take transformers 4.57.3's defaults; local_attention follows sliding_window"
    )
    func configurationDefaults() throws {
        // An explicit "gelu" and a null rope_scaling are what the port implements, so both load.
        let json =
            #"{"model_type": "modernbert", "hidden_activation": "gelu", "rope_scaling": null}"#
        let c = try JSONDecoder().decode(ModernBertConfiguration.self, from: Data(json.utf8))
        #expect(c.vocabSize == 50368)
        #expect(c.hiddenSize == 768)
        #expect(c.intermediateSize == 1152)
        #expect(c.numHiddenLayers == 22)
        #expect(c.numAttentionHeads == 12)
        #expect(c.maxPositionEmbeddings == 8192)
        #expect(c.normEps == 1e-5)
        #expect(c.localAttention == 128)
        #expect(c.globalAttnEveryNLayers == 3)
        #expect(c.globalRopeTheta == 160_000)
        #expect(c.localRopeTheta == 10_000)
        #expect(c.headDim == 64)
        #expect(c.windowHalfWidth == 64)

        let fromSlidingWindow = try JSONDecoder().decode(
            ModernBertConfiguration.self, from: Data(#"{"sliding_window": 32}"#.utf8))
        #expect(fromSlidingWindow.localAttention == 64)
        #expect(fromSlidingWindow.windowHalfWidth == 32)
    }

    @Test("an explicit null local_rope_theta falls back to the global base")
    func nullLocalThetaFallsBack() throws {
        let c = try JSONDecoder().decode(
            ModernBertConfiguration.self,
            from: Data(#"{"global_rope_theta": 5.0, "local_rope_theta": null}"#.utf8))
        #expect(c.globalRopeTheta == 5)
        #expect(c.localRopeTheta == 5)
    }

    @Test("layer i is global when i is a multiple of global_attn_every_n_layers")
    func layerKinds() throws {
        let c = try JSONDecoder().decode(ModernBertConfiguration.self, from: Data("{}".utf8))
        let kinds = (0 ..< 7).map { c.isGlobal(layer: $0) }
        #expect(kinds == [true, false, false, true, false, false, true])
    }

    @Test(
        "RoPE bases are read in both transformers formats, and rope_parameters wins over 4's keys"
    )
    func decodesRopeBasesInBothFormats() throws {
        let legacy = try JSONDecoder().decode(
            ModernBertConfiguration.self,
            from: Data(#"{"global_rope_theta": 150000.0, "local_rope_theta": 160000.0}"#.utf8))
        #expect(legacy.globalRopeTheta == 150_000)
        #expect(legacy.localRopeTheta == 160_000)

        let v5 = try JSONDecoder().decode(
            ModernBertConfiguration.self,
            from: Data(
                #"""
                {"num_hidden_layers": 4, "global_rope_theta": 1.0, "local_rope_theta": 2.0,
                 "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
                 "rope_parameters": {
                   "full_attention": {"rope_theta": 150000.0, "rope_type": "default"},
                   "sliding_attention": {"rope_theta": 160000.0, "rope_type": "default"}}}
                """#.utf8))
        #expect(v5.globalRopeTheta == 150_000)
        #expect(v5.localRopeTheta == 160_000)
    }

    @Test(
        "a config is refused if the port cannot run it as transformers would, or it is degenerate",
        arguments: [
            (json: #"{"hidden_activation": "gelu_python"}"#, refusedAt: nil),
            (json: #"{"rope_scaling": {}}"#, refusedAt: nil),
            (json: #"{"rope_scaling": {"rope_type": "default"}}"#, refusedAt: nil),
            (json: #"{"rope_scaling": {"type": "default"}}"#, refusedAt: nil),
            (json: #"{"local_attention": 128, "sliding_window": 64}"#, refusedAt: nil),
            (json: #"{"local_attention": 127, "sliding_window": 63}"#, refusedAt: nil),
            (json: #"{"hidden_activation": "gelu_pytorch_tanh"}"#, refusedAt: "hidden_activation"),
            (
                json: #"{"rope_scaling": {"rope_type": "linear", "factor": 2.0}}"#,
                refusedAt: "rope_scaling"
            ),
            (
                json:
                    #"{"rope_parameters": {"full_attention": {"rope_theta": 1.0, "rope_type": "yarn"}}}"#,
                refusedAt: "rope_parameters"
            ),
            (
                json:
                    #"{"rope_parameters": {"sliding_attention": {"rope_theta": 1.0, "type": "linear"}}}"#,
                refusedAt: "rope_parameters"
            ),
            (
                json:
                    #"{"num_hidden_layers": 3, "layer_types": ["sliding_attention", "full_attention", "full_attention"]}"#,
                refusedAt: "layer_types"
            ),
            (json: #"{"num_attention_heads": 0}"#, refusedAt: "num_attention_heads"),
            (
                json: #"{"hidden_size": 100, "num_attention_heads": 12}"#,
                refusedAt: "num_attention_heads"
            ),
            (json: #"{"global_attn_every_n_layers": 0}"#, refusedAt: "global_attn_every_n_layers"),
            (json: #"{"local_attention": 128, "sliding_window": 32}"#, refusedAt: "sliding_window"),
            (json: #"{"num_hidden_layers": 0}"#, refusedAt: "num_hidden_layers"),
            (json: #"{"local_attention": 0}"#, refusedAt: "local_attention"),
            (json: #"{"sliding_window": 0}"#, refusedAt: "sliding_window"),
            (
                json: #"{"rope_scaling": {"rope_type": "default", "rope_theta": 5.0}}"#,
                refusedAt: "rope_scaling"
            ),
            (json: #"{"vocab_size": 0}"#, refusedAt: "vocab_size"),
            (json: #"{"hidden_size": -768}"#, refusedAt: "hidden_size"),
            (json: #"{"intermediate_size": 0}"#, refusedAt: "intermediate_size"),
            (json: #"{"max_position_embeddings": 0}"#, refusedAt: "max_position_embeddings"),
        ] as [(json: String, refusedAt: String?)]
    )
    func configurationRefusals(json: String, refusedAt: String?) throws {
        let decode = {
            try JSONDecoder().decode(ModernBertConfiguration.self, from: Data(json.utf8))
        }
        guard let refusedAt else {
            _ = try decode()
            return
        }
        let error = #expect(throws: DecodingError.self) { try decode() }
        guard let error else { return }
        guard case .dataCorrupted(let context) = error else {
            Issue.record("expected .dataCorrupted, got \(error)")
            return
        }
        #expect(context.codingPath.last?.stringValue == refusedAt)
    }

    // MARK: - Fixture

    struct Fixture {
        let weights: [String: MLXArray]
        let references: [String: MLXArray]
        let config: ModernBertConfiguration
        let clampConfig: ModernBertConfiguration
        let modelBound: Float
        let mlpBound: Float

        func reference(_ name: String) throws -> MLXArray {
            try #require(references["ref.\(name)"], "fixture has no ref.\(name)")
        }
    }

    static func loadFixture() throws -> Fixture {
        let tensorsURL = try #require(
            Bundle.module.url(forResource: "modernbert-tiny", withExtension: "safetensors"))
        let metaURL = try #require(
            Bundle.module.url(forResource: "modernbert-tiny", withExtension: "json"))
        let arrays = try loadArrays(url: tensorsURL)
        let meta = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: metaURL)) as? [String: Any])
        func configuration(_ key: String) throws -> ModernBertConfiguration {
            let object = try #require(meta[key], "fixture JSON has no \(key)")
            return try JSONDecoder().decode(
                ModernBertConfiguration.self, from: JSONSerialization.data(withJSONObject: object))
        }
        let bounds = try #require(meta["bounds"] as? [String: Double])
        return Fixture(
            weights: arrays.filter { !$0.key.hasPrefix("ref.") },
            references: arrays.filter { $0.key.hasPrefix("ref.") },
            config: try configuration("config"),
            clampConfig: try configuration("clamp_config"),
            modelBound: Float(try #require(bounds["model"])),
            mlpBound: Float(try #require(bounds["mlp"])))
    }

    static func model(_ config: ModernBertConfiguration, _ weights: [String: MLXArray]) throws
        -> ModernBertModel
    {
        let model = ModernBertModel(config)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        return model
    }

    /// Largest absolute difference, optionally only where `keep` is true along the first two axes.
    /// A NaN or an infinity anywhere in either array, even where `keep` is false, makes the result
    /// NaN or infinite, which fails every bound: MLX's `max` propagates a NaN, and zero times
    /// either is NaN. Differing shapes throw, where a trap would kill the process while it holds
    /// the MLX test lock.
    static func maxAbsDiff(_ a: MLXArray, _ b: MLXArray, keeping keep: MLXArray? = nil) throws
        -> Float
    {
        try #require(a.shape == b.shape, "comparing shape \(a.shape) with \(b.shape)")
        var d = abs(a.asType(.float32) - b.asType(.float32))
        if let keep { d = d * keep.asType(.float32).expandedDimensions(axis: -1) }
        return d.max().item(Float.self)
    }

    static func isFinite(_ x: MLXArray) -> Bool {
        !logicalOr(isNaN(x), isInf(x)).any().item(Bool.self)
    }

    /// How a case hands the model its attention mask.
    enum MaskForm {
        /// The fixture's raw `Int32` 0/1 values.
        case int32
        /// Converted to `Bool` first.
        case bool
    }

    /// Every position finite; values compared only at unpadded positions and on pooled vectors,
    /// because a fully masked padded row takes a value that depends on which kernel ran (see
    /// `ModernBertModel.masks(_:length:)`). The pooled vector goes through `Pooling` with the
    /// model's own strategy, the route callers take. The model gets the mask in `form`; the
    /// comparisons always use its boolean form.
    static func checkCase(_ name: String, _ f: Fixture, _ m: ModernBertModel, mask form: MaskForm)
        throws
    {
        let ids = try f.reference("\(name).input_ids")
        let raw = try f.reference("\(name).attention_mask")
        let keep = raw .!= MLXArray(Int32(0))
        let mask = form == .int32 ? raw : keep
        let states = m.layerHiddenStates(ids, attentionMask: mask)
        let output = m(ids, attentionMask: mask)
        let final = try #require(output.hiddenStates)
        let pooled = Pooling(strategy: try #require(m.poolingStrategy))(output, mask: mask)
        try #require(
            states.count == f.config.numHiddenLayers + 1, "the embedding output plus one per layer")
        let finalReference = try f.reference("\(name).final")
        try #require(final.shape == finalReference.shape)
        for (i, s) in (states + [final]).enumerated() {
            #expect(isFinite(s), "\(name): state \(i) has non-finite values")
        }
        let layer0Diff = try maxAbsDiff(states[1], f.reference("\(name).layer0"), keeping: keep)
        #expect(layer0Diff <= f.modelBound, "\(name): state 1, layer 0's output")
        let layer1Diff = try maxAbsDiff(states[2], f.reference("\(name).layer1"), keeping: keep)
        #expect(layer1Diff <= f.modelBound, "\(name): state 2, layer 1's output")
        let layer3Diff = try maxAbsDiff(states[4], f.reference("\(name).layer3"), keeping: keep)
        #expect(layer3Diff <= f.modelBound, "\(name): state 4, layer 3's output")
        let finalDiff = try maxAbsDiff(final, finalReference, keeping: keep)
        #expect(finalDiff <= f.modelBound, "\(name): the final output")
        let pooledDiff = try maxAbsDiff(pooled, f.reference("\(name).pooled"))
        #expect(pooledDiff <= f.modelBound, "\(name): the pooled vector")
    }

    /// Whether float32 computes exactly here. MLX's `MLX_ENABLE_TF32` defaults to on, and on Apple
    /// GPU generation 17 and later with macOS 26.2 and later (M5 class) float32 matmuls and
    /// attention then run in TF32, about 1e-3 off, which is above every bound below. Follows MLX's
    /// own `is_nax_available()` reading of the architecture name (e.g. "applegpu_g16s"), erring
    /// toward skipping.
    static let float32IsExact: Bool = {
        let environment = ProcessInfo.processInfo.environment
        if environment["MLX_ENABLE_TF32"] == "0" { return true }
        // MLX_METAL_GPU_ARCH overrides the device's name inside MLX, so it must here too.
        let name =
            environment["MLX_METAL_GPU_ARCH"].flatMap { $0.isEmpty ? nil : $0 }
            ?? GPU.deviceInfo().architecture
        let arch = Array(name)
        guard arch.count >= 3, let tens = arch[arch.count - 3].wholeNumberValue,
            let ones = arch[arch.count - 2].wholeNumberValue
        else { return false }
        return tens * 10 + ones < 17
    }()

    /// The gate on every test that compares float32 values against the reference.
    static let exactFloat32: ConditionTrait = .enabled(
        if: ModernBertTests.float32IsExact,
        "float32 may run as TF32 on this GPU: set MLX_ENABLE_TF32=0")

    // MARK: - Numerical cases

    @Test(
        "the MLP applies exact GELU to the first half of Wi's output",
        ModernBertTests.exactFloat32)
    func mlpCase() throws {
        try MLXMetalTestLock.withLock {
            let f = try Self.loadFixture()
            let m = try Self.model(f.config, f.weights)
            let out = m.layers[1].mlp(try f.reference("mlp.input"))
            let d = try Self.maxAbsDiff(out, f.reference("mlp.output"))
            #expect(d <= f.mlpBound)
        }
    }

    @Test(
        "standard case: a padded batch matches the reference",
        ModernBertTests.exactFloat32)
    func standardCase() throws {
        try MLXMetalTestLock.withLock {
            let f = try Self.loadFixture()
            try Self.checkCase("standard", f, try Self.model(f.config, f.weights), mask: .int32)
        }
    }

    @Test(
        "aligned case: a full kernel tile of keys keeps padded rows finite",
        ModernBertTests.exactFloat32)
    func alignedCase() throws {
        try MLXMetalTestLock.withLock {
            let f = try Self.loadFixture()
            try Self.checkCase("aligned", f, try Self.model(f.config, f.weights), mask: .bool)
        }
    }

    @Test(
        "short case: a padded batch of at most eight tokens, the short-query kernel's path",
        ModernBertTests.exactFloat32)
    func shortCase() throws {
        try MLXMetalTestLock.withLock {
            let f = try Self.loadFixture()
            let m = try Self.model(f.config, f.weights)
            try Self.checkCase("short", f, m, mask: .bool)
            // The query-time path: one sequence of at most eight tokens, 1-D and unmasked, as a
            // single search query arrives. Row 0 has no padding, so every position is compared.
            let raw = try f.reference("short.attention_mask")
            try #require(
                (raw[0] .!= MLXArray(Int32(0))).all().item(Bool.self), "row 0 must have no padding")
            let output = m(try f.reference("short.input_ids")[0])
            let final = try #require(output.hiddenStates)
            let pooled = Pooling(strategy: try #require(m.poolingStrategy))(output)
            let finalDiff = try Self.maxAbsDiff(final, f.reference("short.final")[0 ..< 1])
            #expect(finalDiff <= f.modelBound, "short row 0 as a 1-D query: the final output")
            let pooledDiff = try Self.maxAbsDiff(pooled, f.reference("short.pooled")[0 ..< 1])
            #expect(pooledDiff <= f.modelBound, "short row 0 as a 1-D query: the pooled vector")
        }
    }

    @Test(
        "input beyond max_position_embeddings is clamped to the configured limit",
        ModernBertTests.exactFloat32)
    func clampCase() throws {
        // Twice against the same references, which were computed with an all-ones mask: with no
        // mask, where global layers take none and local layers the band alone, and with an all-ones
        // Int32 mask over the full, unclamped input, which the model must clamp with the ids.
        try MLXMetalTestLock.withLock {
            let f = try Self.loadFixture()
            let m = try Self.model(f.clampConfig, f.weights)
            let ids = try f.reference("clamp.input_ids")
            let shape = [ids.dim(0), f.clampConfig.maxPositionEmbeddings, f.clampConfig.hiddenSize]
            let runs: [(label: String, mask: MLXArray?)] = [
                ("no mask", nil),
                ("a full-length all-ones mask", MLXArray.ones([1, ids.dim(1)], dtype: .int32)),
            ]
            for run in runs {
                let output = m(ids, attentionMask: run.mask)
                let final = try #require(output.hiddenStates)
                try #require(
                    final.shape == shape, "\(run.label): got \(final.shape), expected \(shape)")
                let pooled = Pooling(strategy: try #require(m.poolingStrategy))(output)
                let finalDiff = try Self.maxAbsDiff(final, f.reference("clamp.final"))
                #expect(finalDiff <= f.modelBound, "\(run.label): the final output")
                let pooledDiff = try Self.maxAbsDiff(pooled, f.reference("clamp.pooled"))
                #expect(pooledDiff <= f.modelBound, "\(run.label): the pooled vector")
            }
        }
    }

    @Test("a bfloat16 model stays bfloat16 through every layer")
    func dtypeCase() throws {
        try MLXMetalTestLock.withLock {
            let f = try Self.loadFixture()
            let m = try Self.model(f.config, f.weights.mapValues { $0.asType(.bfloat16) })
            let ids = try f.reference("standard.input_ids")
            let mask = try f.reference("standard.attention_mask") .!= MLXArray(Int32(0))
            for (i, s) in m.layerHiddenStates(ids, attentionMask: mask).enumerated() {
                #expect(s.dtype == .bfloat16, "state \(i) is \(s.dtype)")
            }
            let pooled = Pooling(strategy: .cls)(m(ids, attentionMask: mask), mask: mask)
            #expect(pooled.dtype == .bfloat16)
        }
    }

    /// Ungated, unlike the value tests: where float32 runs as TF32 they skip, and this is then the
    /// only check that the kernels the GPU picks, such as M5's NAX ones, keep padded rows free of
    /// NaN.
    @Test("padded rows stay finite in float32 and bfloat16, whatever kernel runs")
    func paddedRowsStayFinite() throws {
        try MLXMetalTestLock.withLock {
            let f = try Self.loadFixture()
            let hidden = f.config.hiddenSize
            let ids = try f.reference("aligned.input_ids")
            let mask = try f.reference("aligned.attention_mask")
            for dtype in [DType.float32, .bfloat16] {
                let m = try Self.model(f.config, f.weights.mapValues { $0.asType(dtype) })
                let final = try #require(m(ids, attentionMask: mask).hiddenStates)
                #expect(final.shape == ids.shape + [hidden], "\(dtype): got \(final.shape)")
                let states = m.layerHiddenStates(ids, attentionMask: mask) + [final]
                for (i, s) in states.enumerated() {
                    #expect(Self.isFinite(s), "\(dtype): state \(i) has non-finite values")
                }
                // Row 1 alone, as 1-D ids with a 1-D mask: the promotion to a batch of one.
                let row = try #require(m(ids[1], attentionMask: mask[1]).hiddenStates)
                #expect(
                    row.shape == [1, ids.dim(1), hidden], "\(dtype), row 1 alone: got \(row.shape)")
                let rowStates = m.layerHiddenStates(ids[1], attentionMask: mask[1]) + [row]
                for (i, s) in rowStates.enumerated() {
                    #expect(
                        Self.isFinite(s), "\(dtype), row 1 alone: state \(i) has non-finite values")
                }
            }
        }
    }

    // MARK: - Integration with the library

    @Test("the registry builds ModernBERT with CLS pooling, whatever classifier_pooling says")
    func registry() throws {
        try MLXMetalTestLock.withLock {
            let data = Data(
                #"""
                {"model_type": "modernbert", "hidden_size": 128, "num_attention_heads": 2,
                 "intermediate_size": 192, "num_hidden_layers": 4, "vocab_size": 64,
                 "classifier_pooling": "mean"}
                """#.utf8)
            let model = try ModelType(rawValue: "modernbert").createModel(configuration: data)
            let modern = try #require(model as? ModernBertModel)
            // A unique, never-created directory: no 1_Pooling/config.json is found there, exactly
            // the 8-bit conversion's case, so the loader falls back to the model's own strategy.
            let missingDirectory = FileManager.default.temporaryDirectory
                .appending(component: UUID().uuidString)
            let pooling = loadPooling(modelDirectory: missingDirectory, model: modern)
            #expect(pooling.strategy == .cls)
            #expect(throws: DecodingError.self) {
                try ModelType(rawValue: "modernbert").createModel(
                    configuration: Data(#"{"hidden_activation": "gelu_pytorch_tanh"}"#.utf8))
            }
        }
    }

    @Test(
        "sanitize maps ModernBertFor* exports onto the model and leaves bare checkpoints untouched"
    )
    func sanitizeWeights() throws {
        try MLXMetalTestLock.withLock {
            let f = try Self.loadFixture()
            let (h, v) = (f.config.hiddenSize, f.config.vocabSize)
            let m = ModernBertModel(f.config)
            // A bare ModernBertModel checkpoint is untouched.
            #expect(Set(m.sanitize(weights: f.weights).keys) == Set(f.weights.keys))
            // transformers' ModernBertFor* layout: the encoder under `model.`, the task heads at
            // the top level; decoder.weight is tied to the embedding and never saved.
            var exported = Dictionary(
                uniqueKeysWithValues: f.weights.map { ("model." + $0.key, $0.value) })
            exported["head.dense.weight"] = MLXArray.zeros([h, h])
            exported["head.norm.weight"] = MLXArray.ones([h])
            exported["decoder.bias"] = MLXArray.zeros([v])
            exported["classifier.weight"] = MLXArray.zeros([2, h])
            exported["classifier.bias"] = MLXArray.zeros([2])
            let sanitized = m.sanitize(weights: exported)
            #expect(Set(sanitized.keys) == Set(f.weights.keys))
            try m.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
            // A name keeps `model.` when its bare form is present too, so a duplicate fails the
            // load by name; an unconditional strip would keep one of the two silently.
            let a = MLXArray.ones([h])
            let kept = m.sanitize(weights: ["final_norm.weight": a, "model.final_norm.weight": a])
            #expect(Set(kept.keys) == ["final_norm.weight", "model.final_norm.weight"])
        }
    }
}
