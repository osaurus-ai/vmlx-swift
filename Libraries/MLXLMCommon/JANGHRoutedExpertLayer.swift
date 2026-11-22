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
    private let parentModule: String
    public let supplementalWeightBytes: Int
    public let supplementalParameterCount: Int

    /// One routed projection's mapped codebook bank, for model-specific fused kernels (read only).
    public struct CodebookBank: @unchecked Sendable {
        /// `[E, N, K * bits / 32]` UInt32, LSB-first bitstream per row.
        public let packed: MLXArray
        /// `[E, N]` Float16 row scales.
        public let scales: MLXArray
        public let bits: Int
        /// The `2^bits` codebook levels (F32, as the manifest stores them).
        public let levels: [Float]
        /// The projection reads its input in the H32 basis.
        public let hadamard32: Bool
    }
    public let gateBank: CodebookBank
    public let upBank: CodebookBank
    public let downBank: CodebookBank

    init(banks: JANGHMappedBanks, parentModule: String, inputDimensions: Int,
         activationLimit: Float?, activation: JANGTActivation = .silu) throws {
        let gate = try banks.projection(parentModule + ".gate_proj")
        let up = try banks.projection(parentModule + ".up_proj")
        let down = try banks.projection(parentModule + ".down_proj")
        guard inputDimensions > 0, gate.scales.ndim == 2,
              up.scales.shape == gate.scales.shape, down.scales.ndim == 2,
              down.scales.dim(0) == gate.scales.dim(0),
              down.scales.dim(1) == inputDimensions else {
            throw JANGHFormatContract.ValidationError.invalid("JANGH routed architecture geometry mismatch")
        }
        func bank(_ p: JANGHMappedBanks.Projection, _ role: String) throws -> CodebookBank {
            guard let spec = banks.contract.projections[parentModule + "." + role],
                let book = banks.contract.codebooks[spec.bits]
            else { throw JANGHFormatContract.ValidationError.invalid("JANGH routed bank without codebook") }
            return CodebookBank(packed: p.packed, scales: p.scales, bits: spec.bits,
                                levels: book.levels.map { Float($0) }, hadamard32: spec.rotation == .hadamard32)
        }
        gateBank = try bank(gate, "gate_proj")
        upBank = try bank(up, "up_proj")
        downBank = try bank(down, "down_proj")
        let projections = [gate, up, down]
        supplementalWeightBytes = projections.reduce(0) { $0 + $1.packed.nbytes + $1.scales.nbytes }
        // The sidecar scales have one entry per expert/output row. Count the
        // represented scalar weights, not packed words or codebook levels.
        let hidden = gate.scales.dim(1)
        supplementalParameterCount = gate.scales.size * inputDimensions
            + up.scales.size * inputDimensions + down.scales.size * hidden
        self.block = try JANGHRoutedDecodeBlock(
            banks: banks, parentModule: parentModule, activationLimit: activationLimit,
            activation: activation)
        self.inputDimensions = inputDimensions
        self.parentModule = parentModule
        // Explicit ablation: retain all original kernels/banks while introducing
        // only the host route synchronization required by selected-buffer decode.
        synchronizeRoutesForDiagnostic = ProcessInfo.processInfo.environment[
            "VMLX_JANGH_ROUTE_SYNC_DIAGNOSTIC"] == "1"
        super.init()
    }

    /// Batch-1 decode for a model's own fused decode step: `input` `[T, H]` (any float dtype), `indices` `[T, k]`
    /// UInt32, `scores` `[T, k]` F32 → the router-weighted expert sum `[T, H]` in F32 (H32 input rotation, fused
    /// gate/up + activation + H32 output rotation, weighted down; F32 accumulation, no cast).
    public func decodeWeightedSum(_ input: MLXArray, indices: MLXArray, scores: MLXArray) -> MLXArray {
        do {
            return try block(input, indices: indices, scores: scores, outputDType: .float32)
        } catch {
            preconditionFailure("JANGH routed decode rejected admitted bank: \(error)")
        }
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
            // Diagnostic only (off by default): per-layer expert-use histogram for sizing a hot-expert
            // residency tier on wire-limited Macs. Forces a host sync per MoE layer when enabled.
            if JANGHRouteHistogram.enabled {
                JANGHRouteHistogram.record(layer: parentModule, indices: indices.asArray(UInt32.self))
            }
            let result = try block.routed(
                input.reshaped(tokens, inputDimensions),
                indices: indices.reshaped(tokens, routes),
                scores: scores.reshaped(tokens, routes).asType(.float32),
                outputDType: input.dtype,
                // SPEED-CRITICAL: small MTP/DFlash verify batches (2-8 rows) of mapped JANGH experts
                // MUST take the decode kernel; the sorted prefill tiles re-read whole expert banks per
                // verify and MTP stops beating AR. Do not remove or narrow without an Allosaurus
                // adaptive RunBench + in-app code turn A/B.
                forceDecode: FlashVerificationScope.usesMappedDecode(
                    inputShape: input.shape, routes: routes)
                    || RoutedExpertVerificationScope.usesMappedDecode(
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


/// Diagnostic route histogram (`VMLX_JANGH_ROUTE_HISTOGRAM=<path>`): counts how often each routed expert is
/// selected per MoE layer and rewrites `<path>` as JSON every 256 recorded calls. Off by default; never
/// used for serving decisions.
enum JANGHRouteHistogram {
    static let path = ProcessInfo.processInfo.environment["VMLX_JANGH_ROUTE_HISTOGRAM"]
    static var enabled: Bool { path != nil }
    private static let lock = NSLock()
    nonisolated(unsafe) private static var counts: [String: [UInt32: Int]] = [:]
    nonisolated(unsafe) private static var calls = 0

    static func record(layer: String, indices: [UInt32]) {
        guard let path else { return }
        lock.lock(); defer { lock.unlock() }
        var layerCounts = counts[layer, default: [:]]
        for i in indices { layerCounts[i, default: 0] += 1 }
        counts[layer] = layerCounts
        calls += 1
        if calls % 256 == 0 {
            let json = counts.mapValues { $0.reduce(into: [String: Int]()) { $0[String($1.key)] = $1.value } }
            if let data = try? JSONSerialization.data(withJSONObject: json) {
                try? data.write(to: URL(fileURLWithPath: path))
            }
        }
    }
}
