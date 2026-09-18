// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXEmbedders

/// The ModernBERT port's quantization cases, in every mode MLX offers, each through four tests:
/// which modules `quantize(model:)` replaces and the settings they carry; a float twin built from
/// the quantized weights' own dequantization, compared layer by layer on inputs that reach each
/// regime of the quantized matmul (the vector kernels, split-K and plain qmm), a differential check
/// of the wiring whatever precision the mode keeps; the loader's route from a saved checkpoint back
/// to the same model; and bfloat16 kept throughout.
struct ModernBertQuantizationTests {

    struct Case: Sendable, Codable, CustomStringConvertible {
        let mode: QuantizationMode
        let groupSize: Int
        let bits: Int
        var description: String { "\(mode) group \(groupSize) bits \(bits)" }
    }

    /// `QuantizationMode` is not `CaseIterable`, so an exhaustive switch stands in for it: a mode
    /// MLX adds fails to compile here until the switch handles it. The list the switch runs over is
    /// a literal, though, so a mode added to the switch alone never runs; add it to the list too.
    static let cases: [Case] = [QuantizationMode.affine, .mxfp4, .mxfp8, .nvfp4].flatMap {
        mode -> [Case] in
        switch mode {
        case .affine:
            [Case(mode: mode, groupSize: 64, bits: 8), Case(mode: mode, groupSize: 64, bits: 4)]
        case .mxfp4: [Case(mode: mode, groupSize: 32, bits: 4)]
        case .mxfp8: [Case(mode: mode, groupSize: 32, bits: 8)]
        case .nvfp4: [Case(mode: mode, groupSize: 16, bits: 4)]
        }
    }

    /// The fixture's model in `dtype`, quantized for `c` by `quantize(model:)`. The loader runs
    /// its filtered variant, which quantizes each module whose `.scales` the checkpoint holds.
    ///
    /// MLX takes the whole process down when `quantize` gets a group size or bit width the mode
    /// does not support, or a group size that does not divide the width it quantizes (`quantize`
    /// in mlx's `ops.cpp`): the op throws, and the Swift wrapper then indexes its empty result, so
    /// not even `withError` can catch it. `quantize(model:)` defaults to (64, 4) whatever the mode,
    /// so each case names its own pair, and the widths are checked here first.
    static func quantizedModel(
        _ f: ModernBertTests.Fixture, _ c: Case, dtype: DType = .float32
    ) throws -> ModernBertModel {
        for width in [f.config.hiddenSize, f.config.intermediateSize] {
            try #require(width % c.groupSize == 0, "\(c): the group size does not divide \(width)")
        }
        let model = try ModernBertTests.model(f.config, f.weights.mapValues { $0.asType(dtype) })
        quantize(model: model, groupSize: c.groupSize, bits: c.bits, mode: c.mode)
        return model
    }

    // MARK: - Structure

    /// Reads no float value, so it runs on every GPU.
    @Test(
        "quantize replaces only the embedding and every projection, with the case's own settings",
        arguments: ModernBertQuantizationTests.cases)
    func structure(_ c: Case) throws {
        try MLXMetalTestLock.withLock {
            let f = try ModernBertTests.loadFixture()
            let leaves = try Self.quantizedModel(f, c).leafModules().flattened()
            // The embedding and each layer's four projections: 17 for the fixture's four layers.
            let expected: Set<String> = Set(
                ["embeddings.tok_embeddings"]
                    + (0 ..< f.config.numHiddenLayers).flatMap { layer in
                        ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"].map { "layers.\(layer).\($0)" }
                    })
            let found = Set(leaves.filter { $0.1 is Quantized }.map { $0.0 })
            let missing = expected.subtracting(found).sorted()
            let unexpected = found.subtracting(expected).sorted()
            #expect(found == expected, "missing \(missing), unexpected \(unexpected)")
            let unquantized = leaves.filter {
                !($0.1 is Quantized) && ($0.1 is Linear || $0.1 is Embedding)
            }.map { $0.0 }
            #expect(unquantized.isEmpty, "still float: \(unquantized)")
            // The twin is built from each module's own settings, so it cannot see a quantize that
            // quietly fell back to affine; this check names each module that did.
            for (path, module) in leaves {
                guard let packed = module as? Quantized else { continue }
                #expect(
                    packed.mode == c.mode && packed.groupSize == c.groupSize
                        && packed.bits == c.bits,
                    "\(path) is \(packed.mode) group \(packed.groupSize) bits \(packed.bits)")
            }
        }
    }

    // MARK: - Twin

    /// An input for the twin comparison. Its row count, batch times length, is what each
    /// projection's matmul sees.
    struct Input {
        let name: String
        let ids: MLXArray
        let mask: MLXArray?
        var label: String { "\(name) (\(ids.size) rows)" }
    }

    /// The twin's inputs, chosen by row count, which is what picks the quantized matmul kernel
    /// (`QuantizedMatmul::eval_gpu` in mlx's `backend/metal/quantized.cpp`). The thresholds below
    /// are the pin's, MLX 0.32.2; the test cannot tell which kernel ran, so this is the record.
    ///
    /// Below `get_qmv_batch_limit`, which at these widths is 13 on M3 and M4, 14 on M1 and M2, 32
    /// on Ultras and 33 on M5, the vector kernels run: `qmv_quad` for the three projections whose
    /// input is 128 wide, and for `mlp.Wo`'s 192 `qmv_wide` from two rows (for affine only from M3
    /// on) and plain `qmv` for one. From the limit on `qmm_splitk` runs, falling back to plain
    /// `qmm` once its split count reaches 1: past 2048 rows for the two projections 128 wide in
    /// their output. So 12, 6 and 1 rows take the vector kernels everywhere, 45 and 64 rows
    /// split-K everywhere, 32 rows split-K except on an M5, and 2079 rows plain qmm everywhere.
    /// Out of reach here: `qmv_fast`, which needs K to be a multiple of 512 at 4 bits or 256 at 8,
    /// and M5's `qmm_nax`, which float32 takes only with TF32 on.
    static func twinInputs(_ f: ModernBertTests.Fixture) throws -> [Input] {
        func fixture(_ name: String) throws -> Input {
            Input(
                name: name, ids: try f.reference("\(name).input_ids"),
                mask: try f.reference("\(name).attention_mask") .!= MLXArray(Int32(0)))
        }
        let short = try fixture("short")
        let synthetic = try syntheticBatches(f)
        return [
            short,
            Input(name: "short row 0 as a 1-D query", ids: short.ids[0], mask: nil),
            Input(name: "short row 0's first token alone", ids: short.ids[0][..<1], mask: nil),
            try fixture("standard"),
            try fixture("aligned"),
            synthetic.splitK,
            synthetic.plainQmm,
        ]
    }

    /// Two synthetic batches that end on a partial 32-row tile, as real batches almost always do,
    /// which the `qmm` and `qmm_splitk` kernels load and store by separate code: 5 × 9 = 45 rows,
    /// 13 in the last tile and past every vector limit, so split-K on every GPU; and 33 × 63 =
    /// 2079 rows, 31 in the last tile, so plain qmm on every GPU. Position j of row b holds the id
    /// (37j + 11b + 5) mod the vocabulary, so no two rows are alike, and a few rows are padded.
    static func syntheticBatches(
        _ f: ModernBertTests.Fixture
    ) throws -> (splitK: Input, plainQmm: Input) {
        func batch(_ count: Int, _ length: Int, unpadded: [Int: Int]) throws -> Input {
            try #require(
                length <= f.config.maxPositionEmbeddings,
                "a synthetic batch must fit in \(f.config.maxPositionEmbeddings) positions")
            try #require(
                count * length % 32 != 0, "a synthetic batch must end on a partial 32-row tile")
            let rows = (0 ..< count).map { b in
                (0 ..< length).map { j in Int32((37 * j + 11 * b + 5) % f.config.vocabSize) }
            }
            try #require(Set(rows).count == count, "the synthetic rows must all differ")
            let lengths = (0 ..< count).map { unpadded[$0] ?? length }
            let keep = (0 ..< count * length).map { $0 % length < lengths[$0 / length] }
            return Input(
                name: "synthetic \(count) × \(length)",
                ids: MLXArray(rows.flatMap { $0 }, [count, length]),
                mask: MLXArray(keep, [count, length]))
        }
        let splitK = try batch(5, 9, unpadded: [1: 6, 3: 2])
        let plainQmm = try batch(33, 63, unpadded: [1: 50, 2: 27, 3: 9])
        try #require(
            splitK.ids.size >= 33 && plainQmm.ids.size > 2048,
            "the synthetic row counts no longer reach split-K and plain qmm on every GPU")
        return (splitK, plainQmm)
    }

    /// Every position is compared, padding included: both models run the same attention kernel on
    /// the same shapes, so padded rows are comparable, and a masked comparison would hide a fault
    /// confined to them. State 0 is the embedding output, state i the output of layer i - 1, and
    /// the last state the final output; each has a bound relative to its own largest value.
    @Test(
        "a quantized model matches its dequantized float twin at every layer in each kernel regime",
        ModernBertTests.exactFloat32,
        arguments: ModernBertQuantizationTests.cases)
    func matchesDequantizedTwin(_ c: Case) throws {
        try MLXMetalTestLock.withLock {
            let f = try ModernBertTests.loadFixture()
            let model = try Self.quantizedModel(f, c)
            var twinWeights = f.weights
            for (path, module) in model.leafModules().flattened() {
                guard let packed = module as? Quantized else { continue }
                let arrays = Dictionary(uniqueKeysWithValues: packed.parameters().flattened())
                let weight = try #require(arrays["weight"], "\(path) has no weight")
                let scales = try #require(arrays["scales"], "\(path) has no scales")
                twinWeights["\(path).weight"] = dequantized(
                    weight, scales: scales, biases: arrays["biases"], groupSize: packed.groupSize,
                    bits: packed.bits, mode: packed.mode, dtype: .float32)
            }
            let twin = try ModernBertTests.model(f.config, twinWeights)

            for input in try Self.twinInputs(f) {
                let (ids, mask) = (input.ids, input.mask)
                let modelFinal = try #require(model(ids, attentionMask: mask).hiddenStates)
                let twinFinal = try #require(twin(ids, attentionMask: mask).hiddenStates)
                let modelStates = model.layerHiddenStates(ids, attentionMask: mask) + [modelFinal]
                let twinStates = twin.layerHiddenStates(ids, attentionMask: mask) + [twinFinal]
                try #require(
                    modelStates.count == f.config.numHiddenLayers + 2
                        && twinStates.count == modelStates.count)
                for (i, (q, t)) in zip(modelStates, twinStates).enumerated() {
                    let place = "\(input.label), state \(i)"
                    #expect(
                        ModernBertTests.isFinite(q),
                        "\(place): the quantized model has non-finite values")
                    #expect(ModernBertTests.isFinite(t), "\(place): the twin has non-finite values")
                    let bound = 1e-3 * abs(t).max().item(Float.self)
                    let d = try ModernBertTests.maxAbsDiff(q, t)
                    #expect(d <= bound, "\(place): \(d / bound)× the bound, \(d) against \(bound)")
                }
            }
        }
    }

    // MARK: - Loader

    /// The fixture JSON's `config` object, which the round trip writes out as a `config.json`.
    static func fixtureConfigObject() throws -> [String: Any] {
        let url = try #require(
            Bundle.module.url(forResource: "modernbert-tiny", withExtension: "json"))
        let meta = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        return try #require(meta["config"] as? [String: Any], "fixture JSON has no config")
    }

    /// Ungated on purpose: it checks no value against a tolerance, only that the loaded model
    /// computes exactly what the in-process one does. Both run the same packed arrays through the
    /// same kernels, so TF32, where it applies, changes both alike.
    @Test(
        "the loader reads a saved quantized model back into one that computes exactly the same",
        arguments: ModernBertQuantizationTests.cases)
    func loaderRoundTrip(_ c: Case) throws {
        try MLXMetalTestLock.withLock {
            let f = try ModernBertTests.loadFixture()
            let model = try Self.quantizedModel(f, c)
            let directory = FileManager.default.temporaryDirectory.appending(
                component: "modernbert-quantized-\(UUID().uuidString)", directoryHint: .isDirectory)
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }

            // What a conversion writes: the packed arrays, and a `quantization` block that the
            // loader's `quantize(model:)` pass applies to every module with `.scales` saved.
            try MLX.save(
                arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
                url: directory.appending(component: "model.safetensors"))
            var config = try Self.fixtureConfigObject()
            if config["model_type"] == nil { config["model_type"] = "modernbert" }
            config["quantization"] = [
                "group_size": c.groupSize, "bits": c.bits, "mode": c.mode.rawValue,
            ]
            try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
                .write(to: directory.appending(component: "config.json"))

            let loaded = try loadSynchronous(modelDirectory: directory, modelName: "\(c)")
            let reloaded = try #require(
                loaded as? ModernBertModel, "the loader built \(type(of: loaded))")
            let ids = try f.reference("standard.input_ids")
            let mask = try f.reference("standard.attention_mask") .!= MLXArray(Int32(0))
            let expected = try #require(model(ids, attentionMask: mask).hiddenStates)
            let actual = try #require(reloaded(ids, attentionMask: mask).hiddenStates)
            let d = try ModernBertTests.maxAbsDiff(actual, expected)
            #expect(d == 0, "standard: the loaded model differs from the in-process one by \(d)")
        }
    }

    // MARK: - dtype

    /// A native bfloat16 model must stay bfloat16 end to end; this pins that on the quantized path.
    /// A dtype is fixed when the graph is built, so the finiteness check is what runs the bfloat16
    /// kernels: on the 2079-row batch, plain qmm, which on M5 is `qmm_nax`, reached by no other
    /// test here. It reads no value against a tolerance, so it runs on every GPU.
    @Test(
        "a bfloat16 model stays bfloat16 and finite through every quantized layer",
        arguments: ModernBertQuantizationTests.cases)
    func bfloat16Throughout(_ c: Case) throws {
        try MLXMetalTestLock.withLock {
            let f = try ModernBertTests.loadFixture()
            let model = try Self.quantizedModel(f, c, dtype: .bfloat16)
            let input = try Self.syntheticBatches(f).plainQmm
            let (ids, mask) = (input.ids, input.mask)
            let final = try #require(model(ids, attentionMask: mask).hiddenStates)
            let states = model.layerHiddenStates(ids, attentionMask: mask) + [final]
            for (i, s) in states.enumerated() {
                #expect(s.dtype == .bfloat16, "\(input.label), state \(i) is \(s.dtype)")
                #expect(
                    ModernBertTests.isFinite(s), "\(input.label), state \(i) has non-finite values")
            }
        }
    }
}
