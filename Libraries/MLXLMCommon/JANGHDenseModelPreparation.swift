import Foundation
import MLX
import MLXNN

/// K2's mixed dense layout: only declared down projections use JANGH. Ordinary
/// affine gate/up and exceptional affine down projections retain their own plan.
public final class JANGHDenseModelPreparation {
    public let ordinaryConfiguration: Data
    public let excludedTensorNames: Set<String>
    private let source: JANGHMappedBanks.SourceLease
    private let modules: [Int: String]
    private let inputDimensions: Int

    public init(
        directory: URL, configuration: Data, sidecar: Data?,
        hiddenSize: Int, intermediateSize: Int, layerCount: Int
    ) throws {
        for name in ["jangtq_runtime.safetensors", "jangtq_stacked.safetensors"] {
            guard
                !FileManager.default.fileExists(atPath: directory.appendingPathComponent(name).path)
            else {
                throw JANGHFormatContract.ValidationError.invalid(
                    "dense JANGH cannot coexist with legacy overlays")
            }
        }
        let root = try JSONSerialization.jsonObject(with: configuration) as? [String: Any]
        let partition = try JANGHConfigurationPartition(
            configuration: configuration, sidecar: sidecar)
        guard partition.modelType == "k2_horizon",
            root?["mlp_layout"] as? String == "dense_jangh_down",
            hiddenSize > 0, intermediateSize > 0, layerCount > 0,
            root?["hidden_size"] as? Int == hiddenSize,
            root?["intermediate_size"] as? Int == intermediateSize,
            root?["num_hidden_layers"] as? Int == layerCount
        else {
            throw JANGHFormatContract.ValidationError.invalid(
                "unsupported dense JANGH architecture")
        }
        let allowed = Dictionary(
            uniqueKeysWithValues: (0 ..< layerCount).map {
                ("model.layers.\($0).mlp.down_proj", $0)
            })
        guard !partition.customModules.isEmpty,
            partition.customModules.allSatisfy({ allowed[$0] != nil })
        else {
            throw JANGHFormatContract.ValidationError.invalid(
                "dense JANGH must cover a nonempty subset of K2 down projections")
        }
        let ordinaryRoot =
            try JSONSerialization.jsonObject(with: partition.ordinaryConfiguration)
            as? [String: Any]
        let ordinaryPlan = ordinaryRoot?["quantization"] as? [String: Any]
        for name in allowed.keys where !partition.customModules.contains(name) {
            guard let entry = ordinaryPlan?[name] as? [String: Any],
                entry["mode"] as? String == "affine"
            else {
                throw JANGHFormatContract.ValidationError.invalid(
                    "missing explicit affine exception for K2 down projection")
            }
        }
        let dimensions = Dictionary(
            uniqueKeysWithValues: partition.customModules.map {
                (
                    $0,
                    JANGHTensorIndexPlan.Dimensions(
                        experts: 1, input: intermediateSize, output: hiddenSize)
                )
            })
        let metadata = try JANGHHeaderAdapter.read(
            directory: directory, indexName: "model.safetensors.index.json")
        source = try JANGHMappedBanks.SourceLease(
            directory: directory, metadata: metadata,
            contract: partition.contract, dimensions: dimensions)
        modules = Dictionary(
            uniqueKeysWithValues: partition.customModules.map { (allowed[$0]!, $0) })
        inputDimensions = intermediateSize
        ordinaryConfiguration = partition.ordinaryConfiguration
        excludedTensorNames = Set(
            partition.customModules.flatMap { [$0 + ".tq2_packed", $0 + ".tq2_scales"] })
    }

    public func makeProjections() throws -> [Int: JANGHDenseLinear] {
        let banks = try JANGHMappedBanks(source: source)
        return try modules.mapValues {
            try JANGHDenseLinear(banks: banks, module: $0, inputDimensions: inputDimensions)
        }
    }
}

/// Packed one-expert projection. Bank arrays stay with an opaque mmap owner,
/// outside generic module parameters; no expanded weight placeholders are made.
public final class JANGHDenseLinear: Module, UnaryLayer, SupplementalModelWeights {
    private final class Storage {
        let bank: JANGHMappedBanks.Projection
        init(_ bank: JANGHMappedBanks.Projection) { self.bank = bank }
    }
    private let storage: Storage
    private let decode: JANGHProjectionKernel
    private let prefill: JANGHPrefillKernel
    private let rotation: JANGHFormatContract.Rotation
    private let rowRotation = JANGHRowRotation()
    private let inputDimensions: Int
    public let supplementalWeightBytes: Int
    public let supplementalParameterCount: Int

    init(banks: JANGHMappedBanks, module: String, inputDimensions: Int) throws {
        let bank = try banks.projection(module)
        guard bank.packed.dim(0) == 1, bank.scales.dim(0) == 1 else {
            throw JANGHFormatContract.ValidationError.invalid(
                "dense JANGH requires exactly one expert")
        }
        storage = Storage(bank)
        decode = try JANGHProjectionKernel(contract: banks.contract, module: module)
        prefill = try JANGHPrefillKernel(contract: banks.contract, module: module)
        rotation = banks.contract.projections[module]!.rotation
        self.inputDimensions = inputDimensions
        supplementalWeightBytes = bank.packed.nbytes + bank.scales.nbytes
        supplementalParameterCount = inputDimensions * bank.scales.size
        super.init()
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        precondition(input.ndim >= 1 && input.dim(-1) == inputDimensions)
        let count = input.size / inputDimensions
        let outputs = storage.bank.scales.dim(1)
        let shape = Array(input.shape.dropLast()) + [outputs]
        if count == 0 { return MLXArray.zeros(shape, dtype: input.dtype) }
        let x = input.reshaped(count, inputDimensions)
        let indices = MLXArray.zeros([count], type: UInt32.self)
        do {
            let y: MLXArray
            if count >= 64 && inputDimensions.isMultiple(of: 64) {
                let rotated = rotation == .hadamard32 ? try rowRotation(x) : x
                y = try prefill.projectSorted(
                    rotated, packed: storage.bank.packed,
                    scales: storage.bank.scales, indices: indices)
            } else {
                // Rotate once per input row in FP32. Keeping the rotation in
                // FP32 preserves the dense decode contract while avoiding its
                // repetition in every output-row group of the fused kernel.
                let decodeInput = rotation == .hadamard32 ? x.asType(.float32) : x
                y = try decode.project(
                    decodeInput, packed: storage.bank.packed, scales: storage.bank.scales, indices: indices)
            }
            return y.asType(input.dtype).reshaped(shape)
        } catch {
            preconditionFailure("dense JANGH execution rejected admitted bank: \(error)")
        }
    }
}
