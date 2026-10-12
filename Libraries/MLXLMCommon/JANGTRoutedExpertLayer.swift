// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN

/// A routed MoE layer of a JANGHT bundle: the gate/up group and the down group are each JANGT (trellis) or JANGH
/// (jangtq2 codebook), per the bundle's per-module contract. Port of vMLX Python `jangt/switch.py`
/// `JTMixedSwitchGLU.routed` decode route (4 launches when both groups are trellis):
///
///  1. xr = H128(x * su_gu)                         per-layer input signs, 0 on the layer's outlier columns
///  2. h  = gate/up trellis decode + (clamped) SwiGLU, F32, outlier columns from the raw x
///  3. hr = H128(h * sv_dn[expert])                 sv_dn = cs * su * mask, per expert
///  4. y  = weighted down over the token's k slots, F32 accumulation, cast to the input dtype
///
/// A JANGH group runs the existing JANGH decode kernels (H32 basis). Every batch size uses these kernels — the
/// sorted NAX prefill tiles of the Python runtime are not ported yet, so long prefills are correct but slower.
/// Banks are mapped views of the bundle files (no copies); the only derived arrays are the folded sign vectors.
public final class JANGTRoutedExpertLayer: Module, WeightedRoutedExpertLayer, SupplementalModelWeights {
    private enum GateUp {
        case trellis(gate: JANGTProjection, up: JANGTProjection, signs: MLXArray)
        case codebook(kernel: JANGHFusedGateUpKernel, prefill: JANGHPrefillKernel,
                      gate: JANGHMappedBanks.Projection, up: JANGHMappedBanks.Projection)
    }

    private enum Down {
        case trellis(JANGTProjection, signs: MLXArray)
        case codebook(kernel: JANGHWeightedDownKernel, prefill: JANGHPrefillKernel, bank: JANGHMappedBanks.Projection)
    }

    /// Routed (token, slot) pairs from which the expert-sorted prefill tiles run instead of the decode kernels
    /// (vMLX Python `SORT_THRESHOLD`): > 64 so an 8-token speculative verify (8 x top-8) stays on decode.
    static let sortThreshold = 72
    /// `VMLX_JANGT_PREFILL=0` keeps the decode kernels for every batch (A/B and diagnosis).
    static let prefillEnabled = ProcessInfo.processInfo.environment["VMLX_JANGT_PREFILL"] != "0"
    private let intermediateSize: Int
    /// The gate/up outlier columns when they are layer-wide (the same real columns for every expert, gate and up):
    /// the fused prefill epilogue needs one column list per tile.
    private var sharedOutlierColumns: MLXArray?

    private let gateUp: GateUp
    private let down: Down
    private let hiddenSize: Int
    private let limit: Float
    private let rowRotation = JANGHRowRotation()
    public let supplementalWeightBytes: Int
    public let supplementalParameterCount: Int

    init(banks: JANGHMappedBanks, parentModule: String, hiddenSize: Int, intermediateSize: Int,
         activationLimit: Float?) throws {
        if let activationLimit, !activationLimit.isFinite || activationLimit <= 0 {
            throw JANGHFormatContract.ValidationError.invalid("invalid JANGT activation limit")
        }
        let contract = banks.contract
        let gateName = parentModule + ".gate_proj", upName = parentModule + ".up_proj"
        let downName = parentModule + ".down_proj"
        var bytes = 0, params = 0
        func count(_ p: JANGTProjection) {
            bytes += p.packed.nbytes + p.scale.nbytes + p.signs.nbytes + p.outlierColumnIndices.nbytes
                + p.outlierWeights.nbytes + (p.columnScales?.nbytes ?? 0)
            params += p.experts * p.outputs * p.inputs
        }
        func count(_ b: JANGHMappedBanks.Projection, inputs: Int) {
            bytes += b.packed.nbytes + b.scales.nbytes
            params += b.scales.size * inputs
        }

        let downIsTrellis = contract.trellis[downName] != nil
        if contract.trellis[gateName] != nil {
            let gate = try banks.trellisProjection(gateName), up = try banks.trellisProjection(upName)
            guard gate.inputs == hiddenSize, gate.outputs == intermediateSize, up.inputs == hiddenSize,
                up.outputs == intermediateSize, up.experts == gate.experts, up.k == gate.k
            else { throw JANGHFormatContract.ValidationError.invalid("JANGT gate/up geometry mismatch") }
            // Python `_unit`: the gate/up outlier columns are layer-wide, taken from expert 0's gate row.
            let signs = gate.signs * Self.keepMask(width: hiddenSize, columns: gate.outlierColumnIndices[0 ..< 1])[0]
            eval(signs)
            gateUp = .trellis(gate: gate, up: up, signs: signs)
            if gate.outlierColumns > 0 {
                let g = gate.outlierColumnIndices, u = up.outlierColumnIndices
                let first = g[0 ..< 1]
                let shared = g.shape == u.shape && (g .== first).all().item(Bool.self)
                    && (g .== u).all().item(Bool.self) && (first .>= MLXArray(Int32(0))).all().item(Bool.self)
                if shared { sharedOutlierColumns = first.reshaped(-1).asType(.int32) }
            }
            count(gate)
            count(up)
        } else {
            let gate = try banks.projection(gateName), up = try banks.projection(upName)
            // This layer only exists when some group is trellis: a JANGH gate/up here always feeds a trellis down,
            // whose input is the raw (unrotated) activation.
            gateUp = .codebook(
                kernel: try JANGHFusedGateUpKernel(contract: contract, gateModule: gateName, upModule: upName,
                                                   outputRotation: .none),
                prefill: try JANGHPrefillKernel(contract: contract, module: gateName, upModule: upName),
                gate: gate, up: up)
            count(gate, inputs: hiddenSize)
            count(up, inputs: hiddenSize)
        }
        if downIsTrellis {
            let d = try banks.trellisProjection(downName)
            guard d.inputs == intermediateSize, d.outputs == hiddenSize else {
                throw JANGHFormatContract.ValidationError.invalid("JANGT down geometry mismatch")
            }
            let cs = d.columnScales ?? MLXArray.ones([d.experts, intermediateSize], dtype: .float32)
            let signs = cs * d.signs.expandedDimensions(axis: 0)
                * Self.keepMask(width: intermediateSize, columns: d.outlierColumnIndices)
            eval(signs)
            down = .trellis(d, signs: signs)
            count(d)
        } else {
            let bank = try banks.projection(downName)
            down = .codebook(kernel: try JANGHWeightedDownKernel(contract: contract, module: downName),
                             prefill: try JANGHPrefillKernel(contract: contract, module: downName), bank: bank)
            count(bank, inputs: intermediateSize)
        }
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.limit = activationLimit ?? 0
        supplementalWeightBytes = bytes
        supplementalParameterCount = params
        super.init()
    }

    /// (R, width) float32: 0 at each row's outlier columns (-1 entries are padding), else 1.
    static func keepMask(width: Int, columns: MLXArray) -> MLXArray {
        let positions = MLXArray(0 ..< Int32(width)).reshaped(1, width, 1)
        let hit = positions .== columns.asType(.int32).expandedDimensions(axis: 1)
        return MLX.where(hit.any(axis: -1), MLXArray(Float(0)), MLXArray(Float(1)))
    }

    public func callRouted(_ input: MLXArray, indices: MLXArray, scores: MLXArray) -> MLXArray {
        precondition(input.ndim >= 2 && input.dim(-1) == hiddenSize,
                     "JANGT routed input does not match admitted architecture")
        precondition(indices.ndim == input.ndim && indices.shape.dropLast() == input.shape.dropLast()
                     && indices.dim(-1) > 0 && scores.shape == indices.shape,
                     "JANGT routed indices and scores must match token dimensions")
        let routes = indices.dim(-1)
        let x = input.reshaped(-1, hiddenSize)
        let ids = indices.reshaped(-1, routes).asType(.uint32)
        let weights = scores.reshaped(-1, routes).asType(.float32)
        if Self.prefillEnabled, ids.size >= Self.sortThreshold, JANGTPrefillKernel.available,
            [.bfloat16, .float16].contains(input.dtype)
        {
            do {
                return try prefill(x, ids: ids, weights: weights).asType(input.dtype).reshaped(input.shape)
            } catch {
                preconditionFailure("JANGT routed prefill rejected admitted bank: \(error)")
            }
        }
        do {
            let hidden: MLXArray
            switch gateUp {
            case .trellis(let gate, let up, let signs):
                hidden = JANGTKernels.gateUp(
                    xr: JANGTKernels.rotateRows(x, signs: signs), x: x, indices: ids, gate: gate, up: up,
                    limit: limit, outputDType: .float32)
            case .codebook(let kernel, _, let gate, let up):
                hidden = try kernel.activatePreparedInput(
                    kernel.prepareInputForFusedDecode(x), gatePacked: gate.packed, gateScales: gate.scales,
                    upPacked: up.packed, upScales: up.scales, indices: ids, limit: limit > 0 ? limit : nil)
            }
            let y: MLXArray
            switch down {
            case .trellis(let projection, let signs):
                let rotated = JANGTKernels.rotateRows(hidden, signs: signs, rowExperts: ids.flattened())
                y = JANGTKernels.down(hr: rotated, h: hidden, indices: ids, weights: weights, projection: projection)
                    .asType(input.dtype)
            case .codebook(let kernel, _, let bank):
                let prepared = kernel.inputRotation == .hadamard32
                    ? try rowRotation(hidden, outputDType: .float32) : hidden
                y = try kernel.projectPreparedHidden(
                    prepared, preparedBasis: kernel.inputRotation, packed: bank.packed, scales: bank.scales,
                    indices: ids, scores: weights, outputDType: input.dtype)
            }
            return y.reshaped(input.shape)
        } catch {
            // Module forwards are nonthrowing throughout MLXNN; admission validated every invariant up front.
            preconditionFailure("JANGT routed execution rejected admitted bank: \(error)")
        }
    }

    /// Expert-sorted prefill (vMLX Python `JTMixedSwitchGLU._prefill`): rows sorted by expert, NAX tiles per group,
    /// outlier columns added in fp32, rows restored and combined with the routing weights in fp32.
    private func prefill(_ x: MLXArray, ids: MLXArray, weights: MLXArray) throws -> MLXArray {
        let tokens = x.dim(0), routes = ids.dim(1)
        let flat = ids.flattened()
        let order = argSort(flat)
        let sortedIDs = take(flat, order).asType(.uint32)
        let tok = order.floorDivide(routes)
        let dtype = x.dtype
        let downIsCodebook: Bool
        if case .codebook = down { downIsCodebook = true } else { downIsCodebook = false }

        var hidden: MLXArray
        var hiddenRotated = false
        switch gateUp {
        case .trellis(let gate, let up, let signs):
            let xr = take(JANGTKernels.rotateRows(x, signs: signs), tok, axis: 0).asType(dtype)
            if gate.outlierColumns == 0 {
                // One NAX pass over both weight tiles with the (clamped) SwiGLU epilogue; a JANGH down gets its
                // input already Hadamard-32 rotated (the H32 store epilogue).
                let rotate = downIsCodebook && intermediateSize % 32 == 0
                hidden = JANGTPrefillKernel.project(xr, gate, up: up, indices: sortedIDs, limit: limit,
                                                    rotateOutput: rotate)
                hiddenRotated = rotate
            } else if let columns = sharedOutlierColumns {
                hidden = JANGTPrefillKernel.projectGateUpWithOutliers(
                    xr, rawRows: take(x, tok, axis: 0), gate: gate, up: up, columns: columns,
                    indices: sortedIDs, limit: limit)
            } else {
                let raw = take(x, tok, axis: 0)
                var g = JANGTPrefillKernel.project(xr, gate, indices: sortedIDs).asType(.float32)
                var u = JANGTPrefillKernel.project(xr, up, indices: sortedIDs).asType(.float32)
                if let o = JANGTPrefillKernel.outliers(raw, gate, indices: sortedIDs) { g = g + o }
                if let o = JANGTPrefillKernel.outliers(raw, up, indices: sortedIDs) { u = u + o }
                if limit > 0 { g = minimum(g, limit); u = clip(u, min: -limit, max: limit) }
                hidden = (g * sigmoid(g) * u).asType(dtype)
            }
        case .codebook(_, let prefillKernel, let gate, let up):
            let xs = take(try rowRotation(x, outputDType: dtype), tok, axis: 0)
            hidden = try prefillKernel.projectSorted(
                xs, packed: gate.packed, scales: gate.scales, indices: sortedIDs,
                upPacked: up.packed, upScales: up.scales, limit: limit > 0 ? limit : nil)
        }

        let sortedOutput: MLXArray
        switch down {
        case .trellis(let projection, let signs):
            let hr = JANGTKernels.rotateRows(hidden, signs: signs, rowExperts: sortedIDs).asType(dtype)
            var y = JANGTPrefillKernel.project(hr, projection, indices: sortedIDs).asType(.float32)
            if let o = JANGTPrefillKernel.outliers(hidden, projection, indices: sortedIDs) { y = y + o }
            sortedOutput = y
        case .codebook(let kernel, let prefillKernel, let bank):
            let hr = hiddenRotated || kernel.inputRotation != .hadamard32
                ? hidden : try rowRotation(hidden, outputDType: dtype)
            sortedOutput = try prefillKernel.projectSorted(
                hr, packed: bank.packed, scales: bank.scales, indices: sortedIDs)
        }
        // Unsort, then the routing-weighted sum as one fp32 (1 x k) @ (k x D) batched matmul.
        let restored = take(sortedOutput, argSort(order), axis: 0).reshaped(tokens, routes, hiddenSize)
            .asType(.float32)
        return matmul(weights.reshaped(tokens, 1, routes), restored).reshaped(tokens, hiddenSize)
    }
}
