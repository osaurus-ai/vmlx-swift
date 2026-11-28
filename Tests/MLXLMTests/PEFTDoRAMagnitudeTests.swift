// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

/// A PEFT DoRA adapter's trained magnitudes (`lora_magnitude_vector`) reach the DoRA layers' `m`.
@Suite("PEFT DoRA magnitudes", .serialized)
struct PEFTDoRAMagnitudeTests {

    private static let prefix = "base_model.model.model.layers.0.mlp.down_proj"

    @Test(
        "a magnitude becomes `m`, 1-D and unchanged, under PEFT's key and under the state-dict key",
        arguments: ["lora_magnitude_vector", "lora_magnitude_vector.weight"])
    func magnitudeTranslates(spelling: String) {
        let out = PEFTAdapterTestBridge.translate([
            "\(Self.prefix).lora_A.weight": MLXArray.zeros([8, 96]),
            "\(Self.prefix).lora_B.weight": MLXArray.zeros([32, 8]),
            "\(Self.prefix).\(spelling)": MLXArray((0 ..< 32).map(Float.init)),
        ])
        let m = out["model.layers.0.mlp.down_proj.m"]
        #expect(m?.shape == [32], "got \(out.keys.sorted())")
        #expect(m.map { $0[5].item(Float.self) } == 5)
        // The factors keep their transpose.
        #expect(out["model.layers.0.mlp.down_proj.lora_a"]?.shape == [96, 8])
        #expect(out["model.layers.0.mlp.down_proj.lora_b"]?.shape == [8, 32])
    }

    // MARK: - Through the loader

    static func makeModel() -> LlamaModel {
        LlamaModel(
            LlamaConfiguration(
                hiddenSize: 64, hiddenLayers: 2, intermediateSize: 128,
                attentionHeads: 4, rmsNormEps: 1e-5, vocabularySize: 64, kvHeads: 2))
    }

    /// q_proj is 64 → 64 and v_proj 64 → 32 in `makeModel()`.
    static let targets = [("q_proj", 64), ("v_proj", 32)]
    static let rank = 4

    /// Distinct values per layer and projection, so that a magnitude landing in the wrong layer
    /// shows.
    static func magnitude(layer: Int, projection: String, width: Int) -> MLXArray {
        let base = Float(100 * (layer + 1) + (projection == "q_proj" ? 0 : 50))
        return MLXArray((0 ..< width).map { base + Float($0) / 8 })
    }

    /// A PEFT adapter directory: `adapter_config.json` and `adapter_model.safetensors`.
    static func writeAdapter(to directory: URL, useDora: Bool) throws {
        let config: [String: Any] = [
            "peft_type": "LORA", "r": rank, "lora_alpha": 8,
            "target_modules": targets.map(\.0), "use_dora": useDora,
        ]
        try JSONSerialization.data(withJSONObject: config)
            .write(to: directory.appendingPathComponent("adapter_config.json"))
        var weights = [String: MLXArray]()
        for layer in 0 ..< 2 {
            for (projection, width) in targets {
                let key = "base_model.model.model.layers.\(layer).self_attn.\(projection)"
                weights["\(key).lora_A.weight"] = MLXRandom.normal([rank, 64]) * 0.01
                weights["\(key).lora_B.weight"] = MLXRandom.normal([width, rank]) * 0.01
                weights["\(key).lora_magnitude_vector"] = magnitude(
                    layer: layer, projection: projection, width: width)
            }
        }
        try MLX.save(arrays: weights, url: directory.appendingPathComponent("adapter_model.safetensors"))
    }

    static func withAdapterDirectory(_ body: (URL) throws -> Void) throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("peft-dora-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try body(directory)
    }

    @Test("a loaded DoRA adapter's layers hold the trained magnitudes")
    func loadedLayersHoldTheTrainedMagnitudes() throws {
        try MLXMetalTestLock.withLock {
            try Self.withAdapterDirectory { directory in
                try Self.writeAdapter(to: directory, useDora: true)
                let adapter = try LoRAContainer.from(directory: directory)
                #expect(adapter.configuration.fineTuneType == .dora)

                let model = Self.makeModel()
                try adapter.load(into: model)
                let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
                for layer in 0 ..< 2 {
                    for (projection, width) in Self.targets {
                        let key = "model.layers.\(layer).self_attn.\(projection).m"
                        let m = try #require(parameters[key], "no \(key)")
                        let trained = Self.magnitude(layer: layer, projection: projection, width: width)
                        #expect(
                            allClose(m, trained).item(Bool.self),
                            "\(key) is not the adapter's magnitude")
                    }
                }
            }
        }
    }

    @Test("a magnitude in an adapter that is not DoRA fails the load instead of being dropped")
    func magnitudeWithoutDoRAIsAnError() throws {
        try MLXMetalTestLock.withLock {
            try Self.withAdapterDirectory { directory in
                try Self.writeAdapter(to: directory, useDora: false)
                let adapter = try LoRAContainer.from(directory: directory)
                #expect(adapter.configuration.fineTuneType == .lora)
                #expect(throws: (any Error).self) { try adapter.load(into: Self.makeModel()) }
            }
        }
    }
}
