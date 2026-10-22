import Foundation

/// Architecture-supplied dimensions, independent of the packed tensor shapes.
/// This is an admission contract; it does not register an architecture.
public struct JANGHRoutedModelLayout: Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let expertCount: Int
    public let sparseLayers: Set<Int>
    /// Routed experts per token (top-k). Only the selected-expert (SSD/LRU) path uses it.
    public let routesPerToken: Int
    /// Wrapper in front of `model.layers.L.mlp.switch_mlp` in the checkpoint (`language_model.` for MiniCPM-V 4.7).
    public let modulePrefix: String
    /// Checkpoint-local feed-forward child; the default preserves existing families.
    public let feedForwardModule: String

    public init(modelType: String, hiddenSize: Int, intermediateSize: Int,
                expertCount: Int, sparseLayers: Set<Int>, routesPerToken: Int = 8,
                modulePrefix: String = "", feedForwardModule: String = "mlp") {
        self.routesPerToken = routesPerToken
        self.modulePrefix = modulePrefix
        self.feedForwardModule = feedForwardModule
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.expertCount = expertCount
        self.sparseLayers = sparseLayers
    }
}

/// Gate-less relu2 routed stacks (Nemotron-H latent MoE). Unlike the SwiGLU
/// layout, every routed layer may have its own expert count, intermediate width
/// and top-k (Nemotron 3.5 Bolt's `block_configs` puzzle blocks).
public struct JANGHRelu2RoutedLayout: Sendable {
    public struct Geometry: Sendable, Equatable {
        public let experts: Int
        public let intermediate: Int
        public let routes: Int
        public init(experts: Int, intermediate: Int, routes: Int) {
            self.experts = experts
            self.intermediate = intermediate
            self.routes = routes
        }
    }

    public let modelType: String
    /// Width the experts read and write (the MoE latent size when present).
    public let expertInputWidth: Int
    public let layers: [Int: Geometry]
    /// Checkpoint path of a layer's routed stack, `{L}` replaced by the layer.
    public let parentTemplate: String

    public init(modelType: String, expertInputWidth: Int, layers: [Int: Geometry],
                parentTemplate: String) {
        self.modelType = modelType
        self.expertInputWidth = expertInputWidth
        self.layers = layers
        self.parentTemplate = parentTemplate
    }

    public func parent(_ layer: Int) -> String {
        parentTemplate.replacingOccurrences(of: "{L}", with: String(layer))
    }
}

/// Retains the admitted shard descriptors until explicit bank construction.
/// The ordinary configuration must only be consumed together with all admitted
/// custom banks; its false quantization overrides are not affine substitutes.
public final class JANGHModelPreparation {
    public let ordinaryConfiguration: Data
    public let excludedTensorNames: Set<String>
    public let moduleByLayer: [Int: String]
    let source: JANGHMappedBanks.SourceLease
    private let hiddenSize: Int
    private let intermediateSize: Int
    private let routesPerToken: Int
    private let relu2Layout: JANGHRelu2RoutedLayout?

    /// Shared declaration gate for architecture factories. This only selects
    /// strict admission; it never establishes support from a label alone.
    public static func declaresCustomFormat(configuration: Data, sidecar: Data?) -> Bool {
        guard let root = try? JSONSerialization.jsonObject(with: configuration) as? [String: Any]
        else { return false }
        let sidecarObject = sidecar.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        // `jangt` (trellis) projections are custom banks too: a pure-JANGT bundle (Qwen3.8-Flash-Next JANGHT,
        // 2026-10-16) has no `jangtq2` projection and no codebook header, and the strict ordinary decoder refuses
        // `mode: jangt` just as it refuses `jangtq2`.
        func customMode(_ value: Any?) -> Bool {
            guard let object = value as? [String: Any] else { return false }
            if let mode = object["mode"] as? String, mode == "jangtq2" || mode == "jangt" { return true }
            return object.values.contains { customMode($0) }
        }
        let text = root["text_config"] as? [String: Any]
        let header = root["jangtq"] as? [String: Any]
        return customMode(root["quantization"]) || customMode(root["quantization_config"])
            || customMode(text?["quantization"]) || customMode(text?["quantization_config"])
            || header?["codebook_family"] as? String == "odd-cubic"
            || sidecarObject?["format"] as? String == "jangtq2"
    }

    public init(directory: URL, configuration: Data, sidecar: Data?,
                layout: JANGHRoutedModelLayout) throws {
        // The generic loader treats these historical files as executable
        // codebook/overlay inputs. Never mix them with a custom-v2 bank owner.
        for legacyName in ["jangtq_runtime.safetensors", "jangtq_stacked.safetensors"] {
            guard !FileManager.default.fileExists(atPath: directory.appendingPathComponent(legacyName).path) else {
                throw JANGHFormatContract.ValidationError.invalid(
                    "JANGH cannot coexist with legacy JANGTQ runtime or overlay files")
            }
        }
        let partition = try JANGHConfigurationPartition(
            configuration: configuration, sidecar: sidecar)
        guard partition.modelType == layout.modelType,
              layout.hiddenSize > 0, layout.intermediateSize > 0,
              layout.expertCount > 0, !layout.sparseLayers.isEmpty,
              layout.sparseLayers.allSatisfy({ $0 >= 0 }),
              layout.feedForwardModule == (layout.modelType == "lfm2_moe" ? "feed_forward" : "mlp") else {
            throw JANGHFormatContract.ValidationError.invalid(
                "JANGH architecture layout does not match configuration")
        }
        var dimensions: [String: JANGHTensorIndexPlan.Dimensions] = [:]
        var modules: [Int: String] = [:]
        for layer in layout.sparseLayers.sorted() {
            let parent = layout.modulePrefix + "model.layers.\(layer).\(layout.feedForwardModule).switch_mlp"
            modules[layer] = parent
            for role in ["gate_proj", "up_proj", "down_proj"] {
                dimensions[parent + "." + role] = .init(
                    experts: layout.expertCount,
                    input: role == "down_proj" ? layout.intermediateSize : layout.hiddenSize,
                    output: role == "down_proj" ? layout.hiddenSize : layout.intermediateSize)
            }
        }
        guard Set(dimensions.keys) == partition.customModules else {
            throw JANGHFormatContract.ValidationError.invalid(
                "JANGH custom banks do not cover the architecture sparse layers")
        }
        let metadata = try JANGHHeaderAdapter.read(directory: directory, indexName: "model.safetensors.index.json")
        // Mixed JANGHT layers: each projection is mapped by its own format (jangtq2 codebook or jangt trellis).
        let trellis = partition.contract.trellis
        source = try JANGHMappedBanks.SourceLease(
            directory: directory, metadata: metadata, contract: partition.contract,
            dimensions: dimensions.filter { trellis[$0.key] == nil },
            trellisDimensions: dimensions.filter { trellis[$0.key] != nil })
        hiddenSize = layout.hiddenSize
        intermediateSize = layout.intermediateSize
        routesPerToken = layout.routesPerToken
        relu2Layout = nil
        ordinaryConfiguration = partition.ordinaryConfiguration
        excludedTensorNames = Set(partition.customModules.flatMap { module in
            trellis[module] != nil
                ? JANGHTensorIndexPlan.trellisSuffixes.map { module + $0 }
                : [module + ".tq2_packed", module + ".tq2_scales"]
        })
        moduleByLayer = modules
    }

    /// Relu2 admission: every listed layer must own exactly a codebook fc1/fc2
    /// pair with its own geometry, and the bundle may declare nothing else.
    public init(directory: URL, configuration: Data, sidecar: Data?,
                relu2Layout layout: JANGHRelu2RoutedLayout) throws {
        for legacyName in ["jangtq_runtime.safetensors", "jangtq_stacked.safetensors"] {
            guard !FileManager.default.fileExists(atPath: directory.appendingPathComponent(legacyName).path) else {
                throw JANGHFormatContract.ValidationError.invalid(
                    "JANGH cannot coexist with legacy JANGTQ runtime or overlay files")
            }
        }
        let partition = try JANGHConfigurationPartition(configuration: configuration, sidecar: sidecar)
        guard partition.modelType == layout.modelType, layout.expertInputWidth > 0,
            !layout.layers.isEmpty, layout.layers.keys.allSatisfy({ $0 >= 0 }),
            layout.layers.values.allSatisfy({ $0.experts > 0 && $0.intermediate > 0 && $0.routes > 0 }),
            partition.contract.trellis.isEmpty
        else {
            throw JANGHFormatContract.ValidationError.invalid(
                "JANGH relu2 layout does not match configuration")
        }
        var dimensions: [String: JANGHTensorIndexPlan.Dimensions] = [:]
        var modules: [Int: String] = [:]
        for (layer, geometry) in layout.layers {
            let parent = layout.parent(layer)
            modules[layer] = parent
            dimensions[parent + ".fc1"] = .init(
                experts: geometry.experts, input: layout.expertInputWidth, output: geometry.intermediate)
            dimensions[parent + ".fc2"] = .init(
                experts: geometry.experts, input: geometry.intermediate, output: layout.expertInputWidth)
        }
        guard Set(dimensions.keys) == partition.customModules,
            modules.values.allSatisfy({ partition.contract.isRelu2Stack($0) })
        else {
            throw JANGHFormatContract.ValidationError.invalid(
                "JANGH relu2 banks do not cover the architecture routed layers")
        }
        let metadata = try JANGHHeaderAdapter.read(directory: directory, indexName: "model.safetensors.index.json")
        source = try JANGHMappedBanks.SourceLease(
            directory: directory, metadata: metadata, contract: partition.contract,
            dimensions: dimensions, trellisDimensions: [:])
        hiddenSize = layout.expertInputWidth
        intermediateSize = layout.layers.values.map(\.intermediate).max() ?? 0
        routesPerToken = layout.layers.values.map(\.routes).max() ?? 0
        relu2Layout = layout
        ordinaryConfiguration = partition.ordinaryConfiguration
        excludedTensorNames = Set(partition.customModules.flatMap {
            [$0 + ".tq2_packed", $0 + ".tq2_scales"]
        })
        moduleByLayer = modules
    }

    /// Relu2 counterpart of ``makeRoutedExperts(activationLimit:)``.
    public func makeRelu2RoutedExperts() throws -> [Int: any WeightedRoutedExpertLayer] {
        guard let layout = relu2Layout else {
            throw JANGHFormatContract.ValidationError.invalid("not a relu2 JANGH preparation")
        }
        let banks = try mapBanks()
        return try moduleByLayer.reduce(into: [:]) { result, entry in
            let geometry = layout.layers[entry.key]!
            result[entry.key] = try JANGHRelu2RoutedExpertLayer(
                banks: banks, parentModule: entry.value, inputDimensions: layout.expertInputWidth,
                intermediateSize: geometry.intermediate, experts: geometry.experts)
        }
    }

    /// Architecture-side audit of the loader exclusions: every routed (layer, role) is excluded as exactly one whole
    /// custom format — the codebook pair (`tq2_packed`, `tq2_scales`) or the JANGT trellis set — and nothing else is.
    public static func isCanonicalRoutedExclusionSet(_ names: Set<String>, routedLayers: Set<Int>,
                                                   feedForwardModule: String = "mlp") -> Bool {
        let codebook = ["tq2_packed", "tq2_scales"]
        let trellis = JANGHTensorIndexPlan.trellisSuffixes.map { String($0.dropFirst()) }
        var covered = 0
        for layer in routedLayers {
            for role in ["gate_proj", "up_proj", "down_proj"] {
                let prefix = "model.layers.\(layer).\(feedForwardModule).switch_mlp.\(role)."
                let format = names.contains(prefix + "tq2_packed") ? codebook : trellis
                guard format.allSatisfy({ names.contains(prefix + $0) }) else { return false }
                covered += format.count
            }
        }
        return covered == names.count
    }

    /// Call only after strict ordinary quantization decoding succeeds. All banks
    /// own their packed storage; no dense expert placeholders are constructed.
    public func makeRoutedExperts(activationLimit: Float?) throws
        -> [Int: any WeightedRoutedExpertLayer]
    {
        let banks = try mapBanks()
        let trellis = source.contract.trellis
        return try moduleByLayer.mapValues { parent -> any WeightedRoutedExpertLayer in
            let anyTrellis = ["gate_proj", "up_proj", "down_proj"].contains { trellis[parent + "." + $0] != nil }
            if anyTrellis {
                return try JANGTRoutedExpertLayer(banks: banks, parentModule: parent, hiddenSize: hiddenSize,
                                                  intermediateSize: intermediateSize, activationLimit: activationLimit)
            }
            return try JANGHRoutedExpertLayer(banks: banks, parentModule: parent,
                                              inputDimensions: hiddenSize, activationLimit: activationLimit)
        }
    }

    static func selectedDiagnosticCacheBytes(environment: [String: String]) throws -> Int {
        guard let raw = environment["VMLX_JANGH_SELECTED_CACHE_MIB"] else { return 128 * 1024 * 1024 }
        guard let mib = Int(raw), mib >= 0, mib <= 16 * 1024 else {
            throw JANGHFormatContract.ValidationError.invalid("selected diagnostic cache must be 0...16384 MiB")
        }
        return mib * 1024 * 1024
    }

    func makeSelectedRoutedExperts(activationLimit: Float?, cacheByteLimit: Int = 128 * 1024 * 1024,
                                  storage: JANGHExpertMappedBanks.Storage = .independentMappings) throws
        -> [Int: any WeightedRoutedExpertLayer]
    {
        let owner = try JANGHExpertMappedBanks(source: source, cacheByteLimit: cacheByteLimit, storage: storage)
        return try moduleByLayer.mapValues { parent -> any WeightedRoutedExpertLayer in
            try JANGHSelectedRoutedExpertLayer(source: source, owner: owner, parentModule: parent,
                                               inputDimensions: hiddenSize, activationLimit: activationLimit,
                                               routes: routesPerToken)
        }
    }

    /// Mapping is deferred until the strict ordinary configuration has decoded.
    /// The retained descriptors prevent pathname replacement from redirecting it.
    func mapBanks() throws -> JANGHMappedBanks {
        try JANGHMappedBanks(source: source)
    }
}
