//
//  LoRA+Container.swift
//  mlx-libraries
//
//  Created by Ivan Petrukha on 02.06.2025.
//

import Foundation
import MLX
import MLXNN

/// Configuration for how LoRA or DoRA should be applied.
///
/// Note: It's compatible with `adapter_config.json` file created during training using MLX libraries.
///
/// Example:
/// ```json
/// {
///   "fine_tune_type": "lora",
///   "num_layers": 28,
///   "lora_parameters": {
///     "rank": 16,
///     "scale": 20.0
///   }
/// }
/// ```
public struct LoRAConfiguration: Sendable, Codable {

    public enum FineTuneType: String, Sendable, Codable {
        case lora
        case dora
    }

    public struct LoRAParameters: Sendable, Codable {

        public let rank: Int
        public let scale: Float
        public let keys: [String]?

        public init(rank: Int = 8, scale: Float = 10.0, keys: [String]? = nil) {
            self.rank = rank
            self.scale = scale
            self.keys = keys
        }
    }

    public let numLayers: Int
    public let fineTuneType: FineTuneType
    public let loraParameters: LoRAParameters

    public init(
        numLayers: Int = 16,
        fineTuneType: FineTuneType = .lora,
        loraParameters: LoRAParameters = .init()
    ) {
        self.numLayers = numLayers
        self.fineTuneType = fineTuneType
        self.loraParameters = loraParameters
    }

    enum CodingKeys: String, CodingKey {
        case numLayers = "num_layers"
        case fineTuneType = "fine_tune_type"
        case loraParameters = "lora_parameters"
    }
}

/// A container for managing LoRA or DoRA adapters and applying them to a language model.
///
/// This struct conforms to `ModelAdapter` and can dynamically inject, remove, or fuse adapters into a model at runtime.
public struct LoRAContainer: ModelAdapter, @unchecked Sendable {

    /// The configuration used to construct this adapter container.
    public let configuration: LoRAConfiguration

    /// The parameter values for the adapter modules.
    ///
    /// * Important: this contains MLXArray which is not Sendable, but in this case
    /// we ensure that they are fully evaluated and never updated.  These static MLXArrays
    /// are usable across threads but we cannot prove so to the type checker so this
    /// is left as `@unchecked Sendable`.
    public let parameters: ModuleParameters

    public init(
        configuration: LoRAConfiguration,
        parameters: consuming ModuleParameters
    ) {
        self.configuration = configuration
        self.parameters = parameters
        // ensure that the parameters are fully evaluated before we promise
        // that they are Sendable
        eval(self.parameters)
    }

    /// Creates a `LoRAContainer` by applying the configuration to a compatible `LanguageModel`.
    ///
    /// Note:  This function freezes the model base weights and applies LoRA layers to it.
    public static func from(
        model: LanguageModel,
        configuration: LoRAConfiguration = .init()
    ) throws -> LoRAContainer {
        guard let lora = model as? LoRAModel else {
            throw ModelAdapterError.incompatibleModelType
        }

        model.freeze()
        let layers = lora.loraLayers.suffix(configuration.numLayers)
        let keys = configuration.loraParameters.keys ?? lora.loraDefaultKeys
        replaceLayers(layers: layers, keys: keys) { (layer: Module) in
            createReplacementLayer(target: layer, configuration: configuration)
        }

        return LoRAContainer(
            configuration: configuration,
            parameters: model.trainableParameters()
        )
    }

    /// Loads a `LoRAContainer` from a directory containing adapter weights and configuration.
    ///
    /// Accepts BOTH layouts: MLX's own (`fine_tune_type` + `adapters.safetensors`) and
    /// HuggingFace PEFT's (`peft_type` + `adapter_model.safetensors`), which is what essentially
    /// every published adapter is. PEFT weights are translated on load rather than by an offline
    /// converter, so an adapter downloaded from the Hub works directly — see `PEFTAdapter`.
    public static func from(directory: URL) throws -> LoRAContainer {
        let configurationURL = directory.appending(component: "adapter_config.json")
        let configurationData = try Data(contentsOf: configurationURL)

        func loadWeights() throws -> [String: MLXArray] {
            for name in PEFTAdapter.weightsFilenames {
                let url = directory.appending(component: name)
                if FileManager.default.fileExists(atPath: url.path) {
                    return try MLX.loadArrays(url: url)
                }
            }
            // Preserve the original error surface when neither spelling is present.
            return try MLX.loadArrays(url: directory.appending(component: "adapters.safetensors"))
        }

        let raw = try loadWeights()
        let configuration: LoRAConfiguration
        let weights: [String: MLXArray]
        if let peft = PEFTAdapter.detect(configuration: configurationData) {
            weights = PEFTAdapter.translate(weights: raw)
            configuration = PEFTAdapter.configuration(from: peft, weights: weights)
        } else {
            configuration = try JSONDecoder()
                .decode(LoRAConfiguration.self, from: configurationData)
            weights = raw
        }

        return LoRAContainer(
            configuration: configuration,
            parameters: ModuleParameters.unflattened(weights)
        )
    }

    /// Applies adapter modules (LoRA or DoRA) to the given model.
    ///
    /// This method replaces target layers in the model with corresponding
    /// adapter layers based on the configuration. It also loads adapter-specific
    /// weights into the model.
    public func load(into model: LanguageModel) throws {
        guard let lora = model as? LoRAModel else {
            throw ModelAdapterError.incompatibleModelType
        }

        let layers = lora.loraLayers.suffix(configuration.numLayers)
        // PEFT names LEAVES ("out_proj"); replaceLayers matches full child keys
        // ("linear_attn.out_proj"). Expand against the live layers, or nothing is replaced.
        let keys = PEFTAdapter.expandTargetKeys(
            configuration.loraParameters.keys ?? lora.loraDefaultKeys, in: layers)
        replaceLayers(layers: layers, keys: keys) { (layer: Module) in
            createReplacementLayer(target: layer, configuration: configuration)
        }

        // The adapter records CHECKPOINT paths; the tree uses MODULE paths. Re-key against the
        // live tree — after `replaceLayers`, so the lora_a/lora_b slots exist to match against.
        let modelKeys = model.parameters().flattened().map(\.0)
        let rekeyed = try PEFTAdapter.rekey(
            Dictionary(uniqueKeysWithValues: parameters.flattened()), toModelKeys: modelKeys)

        try model.update(
            parameters: ModuleParameters.unflattened(rekeyed),
            verify: .noUnusedKeys
        )
    }

    /// Permanently fuses the adapter weights into the model's base layers.
    ///
    /// After fusion, adapter weights become part of the model’s original parameters,
    /// and adapter layers are no longer needed.
    public func fuse(with model: LanguageModel) throws {
        guard let lora = model as? LoRAModel else {
            throw ModelAdapterError.incompatibleModelType
        }

        try load(into: model)
        let layers = lora.loraLayers.suffix(configuration.numLayers)
        let keys = configuration.loraParameters.keys ?? lora.loraDefaultKeys
        replaceLayers(layers: layers, keys: keys) { (lora: LoRALayer) in
            lora.fused()
        }
    }

    /// Removes adapter layers (LoRA or DoRA) and restores the model to its original form.
    ///
    /// This method reverts each adapted layer to its original linear layer, if possible.
    public func unload(from model: LanguageModel) {
        guard let lora = model as? LoRAModel else {
            return  // Don't throw an error because nothing was likely applied before
        }

        let layers = lora.loraLayers.suffix(configuration.numLayers)
        let keys = configuration.loraParameters.keys ?? lora.loraDefaultKeys
        replaceLayers(layers: layers, keys: keys) { (lora: LoRALayer) in
            lora.reverted()
        }
    }
}

/// Creates an adapter replacement layer for a given linear layer based on the configuration.
private func createReplacementLayer(
    target: Module,
    configuration: LoRAConfiguration
) -> LoRALayer? {
    switch (target, configuration.fineTuneType) {
    case (let linear as Linear, .lora):
        return LoRALinear.from(
            linear: linear,
            rank: configuration.loraParameters.rank,
            scale: configuration.loraParameters.scale
        )
    case (let linear as Linear, .dora):
        return DoRALinear.from(
            linear: linear,
            rank: configuration.loraParameters.rank,
            scale: configuration.loraParameters.scale
        )
    default:
        return nil
    }
}

/// Traverses the model and replaces its layers using a transformation closure.
private func replaceLayers<T>(
    layers: ArraySlice<Module>,
    keys: [String],
    transforming transform: (T) -> Module?
) {
    for layer in layers {
        var update: [(String, Module)] = []
        for (key, child) in layer.namedModules() where keys.contains(key) {
            if let child = child as? T, let transformed = transform(child) {
                update.append((key, transformed))
            }
        }

        if !update.isEmpty {
            layer.update(modules: .unflattened(update))
        }
    }
}
