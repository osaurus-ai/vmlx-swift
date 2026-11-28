// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

/// A loaded model runs in eval mode, as the reference's loaders leave it (`model.eval()` in mlx_lm's
/// and mlx_vlm's `utils.py`), and adapter layers added later take the model's mode.
@Suite("Eval mode at load", .serialized)
struct EvalModeAtLoadTests {

    static func makeModel() -> LlamaModel {
        LlamaModel(
            LlamaConfiguration(
                hiddenSize: 64, hiddenLayers: 2, intermediateSize: 128,
                attentionHeads: 4, rmsNormEps: 1e-5, vocabularySize: 64, kvHeads: 2))
    }

    /// The paths of the modules that are in training mode.
    static func training(_ model: Module) -> [String] {
        model.namedModules().filter { $0.1.training }.map(\.0)
    }

    @Test("loadWeights leaves every module in eval mode")
    func loadedModelIsInEvalMode() throws {
        try MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("eval-mode-at-load-\(UUID().uuidString)")
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            try Data(#"{"model_type":"llama"}"#.utf8)
                .write(to: directory.appendingPathComponent("config.json"))
            try MLX.save(
                arrays: Dictionary(
                    uniqueKeysWithValues: Self.makeModel().parameters().flattened()),
                url: directory.appendingPathComponent("model.safetensors"))

            let model = Self.makeModel()
            // A new module starts in training mode; without that the check below proves nothing.
            #expect(Self.training(model).count == model.namedModules().count)

            try loadWeights(modelDirectory: directory, model: model)
            #expect(Self.training(model).isEmpty, "still in training mode: \(Self.training(model))")
        }
    }

    @Test(
        "an adapter's new layers take the model's mode",
        arguments: [LoRAConfiguration.FineTuneType.lora, .dora], [false, true])
    func adapterLayersTakeTheModelsMode(
        fineTuneType: LoRAConfiguration.FineTuneType, training: Bool
    ) throws {
        try MLXMetalTestLock.withLock {
            let configuration = LoRAConfiguration(
                numLayers: 2, fineTuneType: fineTuneType, loraParameters: .init(rank: 4))
            // Adapter parameters as a trainer makes them, from a model of the same shape.
            let adapter = try LoRAContainer.from(model: Self.makeModel(), configuration: configuration)

            let model = Self.makeModel()
            model.train(training)
            try adapter.load(into: model)

            let adapterLayers = model.namedModules().filter { $0.1 is LoRALayer }
            #expect(!adapterLayers.isEmpty, "the adapter replaced no layers")
            let wrong = model.namedModules().filter { $0.1.training != training }.map(\.0)
            #expect(wrong.isEmpty, "expected training == \(training), but not in: \(wrong)")
        }
    }
}
