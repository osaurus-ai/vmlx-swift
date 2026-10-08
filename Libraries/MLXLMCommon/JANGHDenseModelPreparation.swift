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
    /// Path-keyed modules (input width per module) for architectures where several dense projections per
    /// layer are JANGH (Qwen3.5-family: gate/up/down). Empty for K2.
    private var pathInputs: [String: Int] = [:]

    /// Dense Qwen3.5-family JANGH (Qwen3.8-27B JANGH2): all 3 MLP projections of every decoder layer are
    /// one-expert codebook banks (gate/up hidden->intermediate, down intermediate->hidden). Mirrors vMLX
    /// Python `jangh/dense.py` (`TQLinear`). Fails closed: every custom module must be one of those paths
    /// and every layer's triple must be present (the format contract enforces complete triples).
    public init(
        qwen35Directory directory: URL, configuration: Data, sidecar: Data?,
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
        let partition = try JANGHConfigurationPartition(
            configuration: configuration, sidecar: sidecar)
        guard partition.modelType == "qwen3_5", hiddenSize > 0, intermediateSize > 0,
            layerCount > 0,
            hiddenSize.isMultiple(of: 32), intermediateSize.isMultiple(of: 32)
        else {
            throw JANGHFormatContract.ValidationError.invalid(
                "unsupported dense qwen3_5 JANGH architecture")
        }
        var dims: [String: JANGHTensorIndexPlan.Dimensions] = [:]
        var inputs: [String: Int] = [:]
        for name in partition.customModules {
            let parts = name.split(separator: ".")
            guard let layer = Int(parts[3]), layer < layerCount else {
                throw JANGHFormatContract.ValidationError.invalid(
                    "dense JANGH layer out of range \(name)")
            }
            let isDown = name.hasSuffix(".down_proj")
            let input = isDown ? intermediateSize : hiddenSize
            dims[name] = .init(
                experts: 1, input: input, output: isDown ? hiddenSize : intermediateSize)
            inputs[name] = input
        }
        let metadata = try JANGHHeaderAdapter.read(
            directory: directory, indexName: "model.safetensors.index.json")
        source = try JANGHMappedBanks.SourceLease(
            directory: directory, metadata: metadata, contract: partition.contract, dimensions: dims
        )
        modules = [:]
        inputDimensions = intermediateSize
        pathInputs = inputs
        ordinaryConfiguration = partition.ordinaryConfiguration
        excludedTensorNames = Set(
            partition.customModules.flatMap { [$0 + ".tq2_packed", $0 + ".tq2_scales"] })
    }

    /// Path-keyed projections (Qwen3.5-family). `sortedThreshold` = rows from which the one-pass sorted QMM
    /// (NAX, 16-row tile for verify windows) replaces the per-row QMV. Measured 27B JANGH2 target forward
    /// (2026-10-06, lane on): per-row QMV +12 ms per extra row; sorted NAX 54-59 ms flat for 4-16 rows.
    public func makeProjectionsByPath(sortedThreshold: Int = 2) throws -> [String: JANGHDenseLinear]
    {
        let banks = try JANGHMappedBanks(source: source)
        var out: [String: JANGHDenseLinear] = [:]
        for (path, input) in pathInputs {
            out[path] = try JANGHDenseLinear(
                banks: banks, module: path, inputDimensions: input,
                sortedThreshold: sortedThreshold,
                qwen35Optimizations: true)
        }
        return out
    }

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

    /// K2 dense-down projections. `fastDecode` admits the bit-exact fast QMV
    /// for single-row decode (each geometry/dtype is compared bitwise against
    /// the generic kernel before use, falling back on any mismatch). The
    /// prefill tile is unchanged: the dense split-K tile is speed-first, not
    /// bitwise, and K2 has not been qualified on it.
    public func makeProjections(fastDecode: Bool = false) throws -> [Int: JANGHDenseLinear] {
        let banks = try JANGHMappedBanks(source: source)
        return try modules.mapValues {
            try JANGHDenseLinear(
                banks: banks, module: $0, inputDimensions: inputDimensions,
                fastDecode: fastDecode)
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
    private let sortedThreshold: Int
    private let fast: JANGHDenseFastQMV?
    private let fastAdmission = JANGHDenseFastAdmission(enabled: JANGHDenseFastAdmission.enabled)
    private let fastKey: String
    public let supplementalWeightBytes: Int
    public let supplementalParameterCount: Int

    init(
        banks: JANGHMappedBanks, module: String, inputDimensions: Int,
        sortedThreshold: Int = 64, qwen35Optimizations: Bool = false,
        fastDecode: Bool? = nil
    ) throws {
        self.sortedThreshold = Swift.max(1, sortedThreshold)
        let bank = try banks.projection(module)
        guard bank.packed.dim(0) == 1, bank.scales.dim(0) == 1 else {
            throw JANGHFormatContract.ValidationError.invalid(
                "dense JANGH requires exactly one expert")
        }
        storage = Storage(bank)
        decode = try JANGHProjectionKernel(contract: banks.contract, module: module)
        prefill = try JANGHPrefillKernel(
            contract: banks.contract, module: module, enableDenseSmallTile: qwen35Optimizations)
        rotation = banks.contract.projections[module]!.rotation
        let bits = banks.contract.projections[module]!.bits
        let book = banks.contract.codebooks[bits]
        fast =
            ((fastDecode ?? qwen35Optimizations) && book != nil
                && JANGHDenseFastQMV.eligible(k: inputDimensions, n: bank.scales.dim(1), bits: bits))
            ? JANGHDenseFastQMV(bits: bits, alpha: book!.alpha, beta: book!.beta) : nil
        fastKey =
            "K=\(inputDimensions) N=\(bank.scales.dim(1)) bits=\(bits) rot=\(rotation.rawValue) book=\(book.map { "\($0.alpha.bitPattern)/\($0.beta.bitPattern)" } ?? "-")"
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
            if count >= sortedThreshold && inputDimensions.isMultiple(of: 64) {
                let rotated = rotation == .hadamard32 ? try rowRotation(x) : x
                if prefill.enableDenseSmallTile, JANGHPrefillKernel.denseSplitKEnabled,
                    let split = prefill.projectDenseSplitK(
                        rotated, packed: storage.bank.packed, scales: storage.bank.scales)
                {
                    y = split
                } else {
                    y = try prefill.projectSorted(
                        rotated, packed: storage.bank.packed,
                        scales: storage.bank.scales, indices: indices)
                }
            } else {
                // Rotate once per input row in FP32. Keeping the rotation in
                // FP32 preserves the dense decode contract while avoiding its
                // repetition in every output-row group of the fused kernel.
                let decodeInput = rotation == .hadamard32 ? x.asType(.float32) : x
                let packed = storage.bank.packed
                let scales = storage.bank.scales
                if let fast, !CompiledDecodeTrace.isActive,
                    fastAdmission.admits(
                        key: "\(fastKey) dtype=\(decodeInput.dtype) rows=\(count)",
                        compare: {
                            // Use actual operands; admission must never consume request RNG.
                            let rotated =
                                rotation == .hadamard32
                                ? try decode.hadamard32(decodeInput) : decodeInput
                            return (
                                fast.project(rotated, packed: packed, scales: scales),
                                try decode.project(
                                    decodeInput, packed: packed, scales: scales, indices: indices)
                            )
                        })
                {
                    let rotated =
                        rotation == .hadamard32 ? try decode.hadamard32(decodeInput) : decodeInput
                    y = fast.project(rotated, packed: packed, scales: scales)
                } else {
                    y = try decode.project(
                        decodeInput, packed: packed, scales: scales, indices: indices)
                }
            }
            return y.asType(input.dtype).reshaped(shape)
        } catch {
            preconditionFailure("dense JANGH execution rejected admitted bank: \(error)")
        }
    }
}
