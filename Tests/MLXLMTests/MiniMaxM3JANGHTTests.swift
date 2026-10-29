// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// A MiniMax-M3.1 JANGHT bundle (mxfp8 / affine trunk per module, quantized embed / head, JANGT trellis routed experts
/// with SwiGLU-OAI) loaded through `LLMModelFactory` and run against the vMLX Python runtime.
///
/// Fixtures: `scripts/minimax_m3_fixtures/generate_janght_fixtures.py` writes `Fixtures/MiniMaxM3/m31-janght/` the way
/// the M3.1 assembler writes a bundle and runs it through vMLX's own JANG entry point (`load_jang_model`); provenance,
/// shapes and the Python-side cross-checks are in `janght-reference.json`. M3.1's MSA geometry (128-key blocks, top-16):
/// the 2853-token prefill has 23 key blocks, so its last 805 queries attend 16 of them per KV group. 2853 is not a
/// multiple of 128, 64 or 8; the 101-token chunks are not either.
///
/// Routes: the goldens were generated with Python's sort threshold at Swift's 512 (token, slot) pairs, so each batch
/// runs the same kernel family in both engines: the 2853-token prefill (11412 pairs) on the sorted NAX tiles, the
/// 101-token chunks (404 pairs) and the decode steps on the decode kernels. MSA attention: vMLX 885355f0 (block-sparse
/// MSA, `MiniMaxM3MSA.swift`), so the decode steps attend on the MSA decode kernel in both engines; the prefill and the
/// chunks take the dense mask path (2 query heads per KV group, below the chunk kernel's 8).
///
/// Tolerance (per-position relative L2 error of the logits; measured on M5 Max, 2026-10-28):
/// - vs the Python runtime, strict float32 (process started with MLX_ENABLE_TF32=0, as the goldens were made): the
///   logits are bit-identical on all three paths (max error 0), so the bound is `strictTolerance`;
/// - vs the Python runtime with MLX's default TF32 float32 matmuls (indexer scores, outlier columns, routing sum):
///   median 0.58%, a few positions up to 42% where TF32 flips an MSA block or router choice. Bounded like the
///   dequantized comparison: median and 90% of positions under `noiseFloorTolerance`;
/// - vs the dequantized-expert float reference: median 0.9% (prefill) / 1.4% (decode), p90 1.2% / 1.9%, same bound.
///   This is the fixture's bf16 noise floor (the Python runtime's own NAX-tile vs decode-kernel chunked prefill
///   differ by 0.97% median, recorded in the json);
/// - controls, the same bundle with a SiLU routed activation: plain SiLU misses by 29-36% at its best position (46%
///   median), SiLU clamped at the same limit by 21-27% (31-34% median), on every path and either reference.
@Suite("MiniMax-M3.1 JANGHT bundle vs the vMLX Python runtime", .serialized)
struct MiniMaxM3JANGHTTests {
    static let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("Fixtures/MiniMaxM3")
    static let prefill = 2853, decodeSteps = 5, chunk = 101

    /// Swift vs the threshold-matched Python runtime in strict float32: max over positions, per path.
    static let strictTolerance: Float = 1e-3
    /// Median and 90th percentile over positions: vs the dequantized reference, and vs Python under TF32.
    static let noiseFloorTolerance: Float = 3e-2
    static let strictFloat32 = ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0"

    private struct FixtureTokenizer: Tokenizer {
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map { Int($0) } }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    private struct FixtureTokenizerLoader: TokenizerLoader {
        func load(from directory: URL) async throws -> any Tokenizer { FixtureTokenizer() }
    }

    /// A private copy of the bundle (the loader maps its files).
    static func bundleCopy() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("minimax-m31-janght-\(UUID().uuidString)")
        try FileManager.default.copyItem(at: fixtures.appendingPathComponent("m31-janght"), to: directory)
        return directory
    }

    static func reference() throws -> [String: MLXArray] {
        try MLX.loadArrays(url: fixtures.appendingPathComponent("janght-reference.safetensors"))
    }

    /// Per-row relative L2 error of `actual` vs `expected` (rows = positions). Fails closed on an empty, non-finite or
    /// all-zero reference.
    static func rowErrors(_ actual: MLXArray, _ expected: MLXArray) throws -> [Float] {
        let a = actual.asType(.float32), e = expected.asType(.float32)
        try #require(a.shape == e.shape && e.dim(0) > 0)
        try #require(isFinite(e).all().item(Bool.self) && isFinite(a).all().item(Bool.self))
        try #require(abs(e).max().item(Float.self) > 0)
        let num = sqrt(((a - e) * (a - e)).sum(axis: -1)), den = sqrt((e * e).sum(axis: -1))
        return (num / den).asArray(Float.self)
    }

    static func median(_ v: [Float]) -> Float { v.sorted()[v.count / 2] }
    static func quantile(_ v: [Float], _ q: Double) -> Float { v.sorted()[min(v.count - 1, Int(Double(v.count) * q))] }

    struct Paths {
        let prefill: MLXArray, decode: MLXArray, chunked: MLXArray
    }

    /// Teacher-forced logits: the prefill rows `rows`, decode steps on the prefill cache, chunked prefill rows.
    static func run(_ model: MiniMaxM3Model, tokens: MLXArray, rows: MLXArray) -> Paths {
        let batch = tokens[.newAxis]
        let full = model(batch[0..., ..<prefill], cache: nil)[0]
        let full_rows = take(full, rows, axis: 0)
        eval(full_rows)

        var cache = model.newCache(parameters: nil)
        eval(model(batch[0..., ..<prefill], cache: cache))
        let decode = concatenated(
            (prefill ..< prefill + decodeSteps).map { model(batch[0..., $0 ..< ($0 + 1)], cache: cache)[0] },
            axis: 0)
        eval(decode)

        cache = model.newCache(parameters: nil)
        let chunked = concatenated(
            stride(from: 0, to: prefill, by: chunk).map {
                model(batch[0..., $0 ..< min($0 + chunk, prefill)], cache: cache)[0]
            }, axis: 0)
        let chunkedRows = take(chunked, rows, axis: 0)
        eval(chunkedRows)
        return Paths(prefill: full_rows, decode: decode, chunked: chunkedRows)
    }

    /// The same bundle built with another routed activation (the controls): admission, banks and generic load exactly
    /// as the factory does them, only the activation differs.
    static func loadWithActivation(_ directory: URL, _ activation: JANGTActivation, limit: Float?) throws
        -> MiniMaxM3Model
    {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let preparation = try #require(
            try MiniMaxM3JANGHPreparation.loadIfDeclared(directory: directory, configurationData: data))
        let model = try MiniMaxM3Model(
            preparation.configuration,
            routedExperts: try preparation.banks.makeRoutedExperts(activationLimit: limit, activation: activation),
            excludedTensorNames: preparation.banks.excludedTensorNames)
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: preparation.ordinaryConfiguration)
        try loadWeights(
            modelDirectory: directory, model: model, quantization: base.quantizationContainer?.quantization,
            perLayerQuantization: base.perLayerQuantization)
        return model
    }

    // MARK: - Load

    @Test("The factory admits the bundle: every module carries exactly its plan entry, every tensor is bound")
    func factoryLoad() async throws {
        let directory = try Self.bundleCopy()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try await LLMModelFactory.shared.load(from: directory, using: FixtureTokenizerLoader())
        let model = try #require(context.model as? MiniMaxM3Model)
        // M3's own chat protocol: `<mm:think>` rail and the namespace-token tool envelope (the bundle stamps neither).
        #expect(context.configuration.toolCallFormat == .minimaxM3)
        #expect(context.configuration.reasoningParserName == "minimax_m3")

        // Routed experts: mapped JANGT banks on exactly the MoE layers.
        for (index, layer) in model.model.layers.enumerated() {
            if let moe = layer.mlp as? MiniMaxM3SparseMoeBlock {
                #expect(moe.experts is JANGTRoutedExpertLayer, "layer \(index)")
                #expect(moe.switchMLP == nil)
                #expect(moe.gate.weight.dtype == .float32 && !(moe.gate is QuantizedLinear), "router \(index)")
                #expect(moe.eScoreCorrectionBias.dtype == .float32)
            }
        }
        #expect(model.model.layers.map { $0.mlp is MiniMaxM3SparseMoeBlock } == [false, true, true])

        // Every quantized module carries its plan entry; nothing else is quantized.
        let config = try #require(
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appendingPathComponent("config.json"))) as? [String: Any])
        let plan = try #require(config["quantization"] as? [String: Any])
        var expected: [String: String] = [:]
        var routed = Set<String>()
        for (path, entry) in plan {
            guard let entry = entry as? [String: Any] else { continue }
            if entry["mode"] as? String == "jangt" {
                routed.insert(path)
            } else {
                expected[path] = "\(entry["mode"] as? String ?? "affine")/\(entry["bits"]!)/\(entry["group_size"]!)"
            }
        }
        #expect(routed.count == 6 && expected.count == 29)
        var actual: [String: String] = [:]
        for (path, module) in model.namedModules() {
            if let q = module as? Quantized {
                actual[path] = "\(q.mode.rawValue)/\(q.bits)/\(q.groupSize)"
            }
        }
        #expect(actual == expected)

        // Every text tensor of the bundle is bound as stored; jt_* banks are mapped by the routed layers; the vision
        // tensors are dropped.
        let stored = try MLX.loadArrays(url: directory.appendingPathComponent("model-00001-of-00001.safetensors"))
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let text = stored.filter { !$0.key.contains(".jt_") && !$0.key.hasPrefix("vision_tower.")
            && !$0.key.hasPrefix("multi_modal_projector.") && !$0.key.hasPrefix("patch_merge_mlp.") }
        #expect(
            Set(parameters.keys) == Set(text.keys),
            "unbound: \(Set(text.keys).subtracting(parameters.keys).sorted()) not from the bundle: \(Set(parameters.keys).subtracting(text.keys).sorted())")
        for (key, value) in text {
            let bound = try #require(parameters[key], "\(key)")
            #expect(bound.dtype == value.dtype && arrayEqual(bound, value).item(Bool.self), "\(key)")
        }
        // The exclusion set names each trellis projection's whole tensor set; gate/up have no column scales.
        let banked = Set(stored.keys.filter { $0.contains(".jt_") })
        #expect(banked.count == 6 * 5 + 2)
        #expect(banked.isSubset(of: model.customTensorNames))
        #expect(
            model.customTensorNames.subtracting(banked).allSatisfy {
                $0.hasSuffix("gate_proj.jt_cs") || $0.hasSuffix("up_proj.jt_cs")
            })
    }

    // MARK: - Numerics

    @Test("Prefill, decode and chunked prefill match the Python runtime; SiLU controls fail")
    func logitsMatchPython() async throws {
        let reference = try Self.reference()
        let tokens = try #require(reference["tokens"]).asType(.int32)
        let rows = try #require(reference["rows"]).asType(.int32)
        #expect(tokens.dim(0) == Self.prefill + Self.decodeSteps)
        // MSA must be selective at this length: more key blocks than the top-k keeps.
        #expect((Self.prefill + 127) / 128 > 16)
        #expect(rows.max().item(Int32.self) >= 2048)

        let directory = try Self.bundleCopy()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try await LLMModelFactory.shared.load(from: directory, using: FixtureTokenizerLoader())
        let model = try #require(context.model as? MiniMaxM3Model)

        try Device.withDefaultDevice(.gpu) {
            let paths = Self.run(model, tokens: tokens, rows: rows)
            let checks: [(String, MLXArray, String)] = [
                ("prefill", paths.prefill, "prefill"), ("decode", paths.decode, "decode"),
                ("chunked", paths.chunked, "chunked"),
            ]
            for (name, actual, key) in checks {
                let errors = try Self.rowErrors(actual, try #require(reference[key]))
                let (med, p90, worst) = (Self.median(errors), Self.quantile(errors, 0.9), errors.max()!)
                print("[MiniMaxM3JANGHTTests] \(name) vs Python runtime (strict f32 \(Self.strictFloat32)): "
                    + "median \(med) p90 \(p90) max \(worst)")
                if Self.strictFloat32 {
                    #expect(worst < Self.strictTolerance, "\(name) vs Python runtime: max \(worst)")
                } else {
                    #expect(med < Self.noiseFloorTolerance && p90 < Self.noiseFloorTolerance, "\(name) (TF32)")
                }
            }
            for (name, actual, key) in [
                ("prefill", paths.prefill, "deq_prefill"), ("decode", paths.decode, "deq_decode"),
            ] {
                let errors = try Self.rowErrors(actual, try #require(reference[key]))
                let (med, p90) = (Self.median(errors), Self.quantile(errors, 0.9))
                print("[MiniMaxM3JANGHTTests] \(name) vs dequantized reference: median \(med) p90 \(p90)")
                #expect(med < Self.noiseFloorTolerance && p90 < Self.noiseFloorTolerance, "\(name) vs dequant")
            }
        }

        // Controls: the same bundle with a SiLU routed activation must miss both references on every path.
        for (name, activation, limit) in [
            ("plain SiLU", JANGTActivation.silu, nil), ("SiLU clamped at 7", JANGTActivation.silu, Float(7)),
        ] as [(String, JANGTActivation, Float?)] {
            let control = try Self.loadWithActivation(directory, activation, limit: limit)
            try Device.withDefaultDevice(.gpu) {
                let paths = Self.run(control, tokens: tokens, rows: rows)
                for (path, actual, key) in [
                    ("prefill", paths.prefill, "prefill"), ("decode", paths.decode, "decode"),
                    ("chunked", paths.chunked, "chunked"), ("prefill", paths.prefill, "deq_prefill"),
                ] {
                    let errors = try Self.rowErrors(actual, try #require(reference[key]))
                    let (med, least) = (Self.median(errors), errors.min()!)
                    print("[MiniMaxM3JANGHTTests] control \(name) \(path) vs \(key): median \(med) min \(least)")
                    // Every position misses the strict bound; the median misses the noise-floor bound by 5x.
                    #expect(least > 10 * Self.strictTolerance, "control \(name) \(path) vs \(key)")
                    #expect(med > 5 * Self.noiseFloorTolerance, "control \(name) \(path) vs \(key)")
                }
            }
        }
    }
}
