import Foundation
import MLX
import MLXNN

/// A pre-admitted mapped routed bank. Storage stays in an opaque owner rather
/// than generic Module parameters, so weight loading cannot evaluate or replace
/// the entire packed bank. Construction errors are thrown before model creation.
public final class JANGHRoutedExpertLayer: Module, WeightedRoutedExpertLayer, SupplementalModelWeights {
    private let block: JANGHRoutedDecodeBlock
    private let inputDimensions: Int
    private let synchronizeRoutesForDiagnostic: Bool
    public let supplementalWeightBytes: Int
    public let supplementalParameterCount: Int

    init(banks: JANGHMappedBanks, parentModule: String, inputDimensions: Int,
         activationLimit: Float?) throws {
        let gate = try banks.projection(parentModule + ".gate_proj")
        let up = try banks.projection(parentModule + ".up_proj")
        let down = try banks.projection(parentModule + ".down_proj")
        guard inputDimensions > 0, gate.scales.ndim == 2,
              up.scales.shape == gate.scales.shape, down.scales.ndim == 2,
              down.scales.dim(0) == gate.scales.dim(0),
              down.scales.dim(1) == inputDimensions else {
            throw JANGHFormatContract.ValidationError.invalid("JANGH routed architecture geometry mismatch")
        }
        let projections = [gate, up, down]
        supplementalWeightBytes = projections.reduce(0) { $0 + $1.packed.nbytes + $1.scales.nbytes }
        // The sidecar scales have one entry per expert/output row. Count the
        // represented scalar weights, not packed words or codebook levels.
        let hidden = gate.scales.dim(1)
        supplementalParameterCount = gate.scales.size * inputDimensions
            + up.scales.size * inputDimensions + down.scales.size * hidden
        self.block = try JANGHRoutedDecodeBlock(
            banks: banks, parentModule: parentModule, activationLimit: activationLimit)
        self.inputDimensions = inputDimensions
        // Explicit ablation: retain all original kernels/banks while introducing
        // only the host route synchronization required by selected-buffer decode.
        synchronizeRoutesForDiagnostic = ProcessInfo.processInfo.environment[
            "VMLX_JANGH_ROUTE_SYNC_DIAGNOSTIC"] == "1"
        super.init()
    }

    public func callRouted(_ input: MLXArray, indices: MLXArray, scores: MLXArray) -> MLXArray {
        precondition(input.ndim >= 2 && input.dim(-1) == inputDimensions,
                     "JANGH routed input does not match admitted architecture")
        precondition(indices.ndim == input.ndim && indices.shape.dropLast() == input.shape.dropLast()
                     && indices.dim(-1) > 0 && scores.shape == indices.shape,
                     "JANGH routed indices and scores must match token dimensions")
        let tokens = input.size / inputDimensions
        let routes = indices.dim(-1)
        // Both vendor routers select on F32 probabilities. GLM may already have
        // rounded its output weights to input dtype; this cast does not undo it.
        do {
            if synchronizeRoutesForDiagnostic { _ = indices.asArray(UInt32.self) }
            let result = try block.routed(
                input.reshaped(tokens, inputDimensions),
                indices: indices.reshaped(tokens, routes),
                scores: scores.reshaped(tokens, routes).asType(.float32),
                outputDType: input.dtype,
                forceDecode: FlashVerificationScope.usesMappedDecode(
                    inputShape: input.shape, routes: routes))
            precondition(result.shape == [tokens, inputDimensions] && result.dtype == input.dtype,
                         "JANGH routed output violates admitted architecture")
            return result.reshaped(input.shape)
        } catch {
            // Module forwards are nonthrowing throughout MLXNN. Refuse violated
            // invariants explicitly rather than substituting an affine route.
            preconditionFailure("JANGH routed execution rejected admitted bank: \(error)")
        }
    }
}
