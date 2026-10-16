import Foundation
import MLX
import MLXNN

/// Gate-less relu2 routed experts on mapped JANGH banks (Nemotron-H latent MoE,
/// Nemotron 3.5 Bolt): `y = Σ_r score_r · fc2_e(relu2(fc1_e(x)))`.
///
/// Composed from the activation-agnostic JANGH primitives — no new Metal:
/// - decode (< 64 routed rows, or a forced mapped-decode verify window):
///   `JANGHProjectionKernel` QMV for fc1 (applies its own H32, F32 out) →
///   relu2 in F32 → H32 rows when fc2 is rotated → `JANGHWeightedDownKernel`
///   (F32 projection + routing sum, one cast);
/// - prefill: argsort by expert → sorted tile projection fc1 → relu2 → H32 →
///   sorted tile projection fc2 → unsort → router-weighted sum in F32.
///
/// fc1 and fc2 carry independent bit widths and rotations (the contract
/// validates the pair). Matches vMLX Python `TQSwitchMLP` (relu2) numerically.
public final class JANGHRelu2RoutedExpertLayer: Module, WeightedRoutedExpertLayer,
    SupplementalModelWeights
{
    private let fc1: JANGHMappedBanks.Projection
    private let fc2: JANGHMappedBanks.Projection
    private let fc1Decode: JANGHProjectionKernel
    private let fc2Decode: JANGHWeightedDownKernel
    private let fc1Prefill: JANGHPrefillKernel
    private let fc2Prefill: JANGHPrefillKernel
    private let fc1Rotation: JANGHFormatContract.Rotation
    private let fc2Rotation: JANGHFormatContract.Rotation
    private let rowRotation = JANGHRowRotation()
    private let inputDimensions: Int
    private let intermediateSize: Int
    public let supplementalWeightBytes: Int
    public let supplementalParameterCount: Int

    init(banks: JANGHMappedBanks, parentModule: String, inputDimensions: Int,
         intermediateSize: Int, experts: Int) throws {
        let fc1 = try banks.projection(parentModule + ".fc1")
        let fc2 = try banks.projection(parentModule + ".fc2")
        guard inputDimensions > 0, intermediateSize > 0, experts > 0,
            fc1.scales.shape == [experts, intermediateSize],
            fc2.scales.shape == [experts, inputDimensions],
            fc1.packed.dim(0) == experts, fc2.packed.dim(0) == experts,
            intermediateSize.isMultiple(of: 64), inputDimensions.isMultiple(of: 64),
            let fc1Projection = banks.contract.projections[parentModule + ".fc1"],
            let fc2Projection = banks.contract.projections[parentModule + ".fc2"]
        else {
            throw JANGHFormatContract.ValidationError.invalid(
                "JANGH relu2 routed geometry mismatch for \(parentModule)")
        }
        self.fc1 = fc1
        self.fc2 = fc2
        fc1Rotation = fc1Projection.rotation
        fc2Rotation = fc2Projection.rotation
        fc1Decode = try JANGHProjectionKernel(contract: banks.contract, module: parentModule + ".fc1")
        fc2Decode = try JANGHWeightedDownKernel(contract: banks.contract, module: parentModule + ".fc2")
        fc1Prefill = try JANGHPrefillKernel(contract: banks.contract, module: parentModule + ".fc1")
        fc2Prefill = try JANGHPrefillKernel(contract: banks.contract, module: parentModule + ".fc2")
        self.inputDimensions = inputDimensions
        self.intermediateSize = intermediateSize
        supplementalWeightBytes = [fc1, fc2].reduce(0) { $0 + $1.packed.nbytes + $1.scales.nbytes }
        supplementalParameterCount = 2 * experts * intermediateSize * inputDimensions
        super.init()
    }

    public func callRouted(_ input: MLXArray, indices: MLXArray, scores: MLXArray) -> MLXArray {
        precondition(input.ndim >= 2 && input.dim(-1) == inputDimensions,
                     "JANGH relu2 routed input does not match admitted architecture")
        precondition(indices.ndim == input.ndim && indices.shape.dropLast() == input.shape.dropLast()
                     && indices.dim(-1) > 0 && scores.shape == indices.shape,
                     "JANGH relu2 routed indices and scores must match token dimensions")
        let tokens = input.size / inputDimensions
        let routes = indices.dim(-1)
        do {
            let result = try routed(
                input.reshaped(tokens, inputDimensions),
                indices: indices.reshaped(tokens, routes).asType(.uint32),
                scores: scores.reshaped(tokens, routes).asType(.float32),
                outputDType: input.dtype,
                // SPEED-CRITICAL (same rule as JANGHRoutedExpertLayer): small MTP/DFlash
                // verify windows of mapped experts take the decode kernels.
                forceDecode: FlashVerificationScope.usesMappedDecode(
                    inputShape: input.shape, routes: routes))
            precondition(result.shape == [tokens, inputDimensions] && result.dtype == input.dtype,
                         "JANGH relu2 routed output violates admitted architecture")
            return result.reshaped(input.shape)
        } catch {
            preconditionFailure("JANGH relu2 routed execution rejected admitted bank: \(error)")
        }
    }

    /// `input` [T, K] (K = latent width), `indices` [T, R] u32, `scores` [T, R] f32 → [T, K].
    func routed(
        _ input: MLXArray, indices: MLXArray, scores: MLXArray, outputDType: DType,
        forceDecode: Bool = false, backend: JANGHPrefillKernel.Backend? = nil
    ) throws -> MLXArray {
        let routes = indices.dim(-1)
        if indices.size < 64 || forceDecode {
            var hidden = try fc1Decode.project(
                input, packed: fc1.packed, scales: fc1.scales, indices: indices)
            hidden = relu2F32(hidden)
            if fc2Rotation == .hadamard32 {
                hidden = try rowRotation(hidden, outputDType: .float32)
            }
            return try fc2Decode.projectPreparedHidden(
                hidden, preparedBasis: fc2Rotation, packed: fc2.packed, scales: fc2.scales,
                indices: indices, scores: scores, outputDType: outputDType)
        }
        let order = argSort(indices.flattened())
        let inverse = argSort(order)
        let sortedIDs = take(indices.flattened(), order)
        let prepared = fc1Rotation == .hadamard32 ? try rowRotation(input) : input
        let sortedInput = take(prepared, order.floorDivide(routes), axis: 0)
        var hidden = try fc1Prefill.projectSorted(
            sortedInput, packed: fc1.packed, scales: fc1.scales, indices: sortedIDs,
            backend: backend)
        hidden = relu2F32(hidden).asType(input.dtype)
        if fc2Rotation == .hadamard32 {
            hidden = try rowRotation(hidden)
        }
        let sortedOutput = try fc2Prefill.projectSorted(
            hidden, packed: fc2.packed, scales: fc2.scales, indices: sortedIDs, backend: backend)
        let restored = take(sortedOutput, inverse, axis: 0)
            .reshaped(input.dim(0), routes, inputDimensions)
        // Router-weighted sum in F32 (mlx-lm `(y * scores[..., None]).sum(-2)`
        // with F32 scores), one cast at the end.
        return (restored.asType(.float32) * scores.reshaped(input.dim(0), routes, 1))
            .sum(axis: 1).asType(outputDType)
    }

    private func relu2F32(_ x: MLXArray) -> MLXArray {
        let r = maximum(x.asType(.float32), MLXArray(Float(0)))
        return r * r
    }
}
