import MLX

/// Experimental composition over owned mmap banks, not a SwitchGLU replacement.
/// Callers explicitly supply the vendor activation limit and result dtype.
/// Factory admission remains a separate loader gate. Prefill uses sorted packed
/// matrix tiles rather than repeating the decode matrix-vector kernel.
final class JANGHRoutedDecodeBlock {
    private let gate: JANGHMappedBanks.Projection
    private let up: JANGHMappedBanks.Projection
    private let down: JANGHMappedBanks.Projection
    private let gateUpKernel: JANGHFusedGateUpKernel
    private let downKernel: JANGHWeightedDownKernel
    private let limit: Float?
    private let prefillGateUp: JANGHPrefillKernel
    private let prefillDown: JANGHPrefillKernel
    private let inputRotation: JANGHFormatContract.Rotation
    private let rowRotation = JANGHRowRotation()

    init(banks: JANGHMappedBanks, parentModule: String, activationLimit: Float?) throws {
        if let activationLimit, !activationLimit.isFinite || activationLimit <= 0 {
            throw JANGHFormatContract.ValidationError.invalid("invalid JANGH activation limit")
        }
        gate = try banks.projection(parentModule + ".gate_proj")
        up = try banks.projection(parentModule + ".up_proj")
        down = try banks.projection(parentModule + ".down_proj")
        downKernel = try JANGHWeightedDownKernel(contract: banks.contract, module: parentModule + ".down_proj")
        gateUpKernel = try JANGHFusedGateUpKernel(
            contract: banks.contract, gateModule: parentModule + ".gate_proj",
            upModule: parentModule + ".up_proj", outputRotation: downKernel.inputRotation)
        limit = activationLimit
        prefillGateUp = try JANGHPrefillKernel(
            contract: banks.contract, module: parentModule + ".gate_proj",
            upModule: parentModule + ".up_proj")
        prefillDown = try JANGHPrefillKernel(
            contract: banks.contract, module: parentModule + ".down_proj")
        inputRotation = banks.contract.projections[parentModule + ".gate_proj"]!.rotation
    }

    /// Flattening and restoring route order occurs only on the GPU. The sorted
    /// row order never changes the original token/slot score association.
    func routed(
        _ input: MLXArray, indices: MLXArray, scores: MLXArray,
        outputDType: DType, backend: JANGHPrefillKernel.Backend? = nil,
        forceDecode: Bool = false
    ) throws -> MLXArray {
        guard input.ndim >= 2, indices.ndim == input.ndim,
            Array(indices.shape.dropLast()) == Array(input.shape.dropLast()),
            scores.shape == indices.shape, scores.dtype == .float32,
            indices.dtype == .uint32, indices.dim(-1) > 0
        else { throw JANGHFormatContract.ValidationError.invalid("invalid JANGH routed shape") }
        let width = input.dim(-1), routes = indices.dim(-1)
        let flat = input.reshaped(-1, width), ids = indices.reshaped(-1, routes)
        if indices.size < 64 || forceDecode {
            return try callAsFunction(flat, indices: ids, scores: scores.reshaped(ids.shape), outputDType: outputDType)
                .reshaped(input.shape)
        }
        let order = argSort(ids.flattened())
        let inverse = argSort(order)
        let sortedIDs = take(ids.flattened(), order)
        var prepared = flat
        if inputRotation == .hadamard32 {
            prepared = try rowRotation(flat)
        }
        let sortedInput = take(prepared, order.floorDivide(routes), axis: 0)
        let hidden = try prefillGateUp.projectSorted(
            sortedInput, packed: gate.packed, scales: gate.scales, indices: sortedIDs,
            upPacked: up.packed, upScales: up.scales, limit: limit,
            rotateOutput: downKernel.inputRotation == .hadamard32, backend: backend)
        let sortedOutput = try prefillDown.projectSorted(
            hidden, packed: down.packed, scales: down.scales, indices: sortedIDs, backend: backend)
        let restored = take(sortedOutput, inverse, axis: 0).reshaped(flat.dim(0), routes, width)
        // Match JANG's prefill contract: contributions are weighted and summed
        // in the projection dtype. Decode's F32 reduction is intentionally distinct.
        return (restored * scores.reshaped(flat.dim(0), routes, 1).asType(restored.dtype))
            .sum(axis: 1).asType(outputDType).reshaped(input.shape)
    }

    func callAsFunction(
        _ input: MLXArray, indices: MLXArray, scores: MLXArray, outputDType: DType
    ) throws -> MLXArray {
        guard input.ndim == 2, indices.ndim == 2, indices.dim(0) == input.dim(0),
            scores.shape == indices.shape, scores.dtype == .float32,
            [.float16, .bfloat16, .float32].contains(outputDType)
        else { throw JANGHFormatContract.ValidationError.invalid("invalid JANGH decode routing") }
        let prepared = try gateUpKernel.prepareInputForFusedDecode(input)
        let hidden = try gateUpKernel.activatePreparedInput(
            prepared, gatePacked: gate.packed, gateScales: gate.scales,
            upPacked: up.packed, upScales: up.scales, indices: indices, limit: limit)
        return try downKernel.projectPreparedHidden(
            hidden, preparedBasis: downKernel.inputRotation,
            packed: down.packed, scales: down.scales,
            indices: indices, scores: scores, outputDType: outputDType)
    }
}
