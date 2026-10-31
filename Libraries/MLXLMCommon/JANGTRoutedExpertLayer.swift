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
///  2. h  = gate/up trellis decode + the model's (clamped) SwiGLU variant, F32, outlier columns from the raw x
///  3. hr = H128(h * sv_dn[expert])                 sv_dn = cs * su * mask, per expert
///  4. y  = weighted down over the token's k slots, F32 accumulation, cast to the input dtype
///
/// A JANGH group runs the existing JANGH decode kernels (H32 basis). Banks are mapped views of the bundle files (no
/// copies); the only derived arrays are the folded sign vectors.
///
/// The activation is the model's (`JANGTActivation`): SiLU by default, SwiGLU-OAI for MiniMax-M3 / M3.1. Every trellis
/// gate/up path applies it in its own epilogue. The JANGH fused gate/up epilogues are SiLU only, so a JANGH gate/up
/// group of a non-SiLU model runs gate and up as two single-weight passes and applies the variant in F32 (Python
/// `JTMixedSwitchGLU._decode_h` / `_prefill`).
public final class JANGTRoutedExpertLayer: Module, WeightedRoutedExpertLayer, SupplementalModelWeights {
    private enum GateUp {
        case trellis(gate: JANGTProjection, up: JANGTProjection, signs: MLXArray)
        case codebook(kernel: JANGHFusedGateUpKernel, prefill: JANGHPrefillKernel,
                      gate: JANGHMappedBanks.Projection, up: JANGHMappedBanks.Projection,
                      separate: SeparateCodebookPasses?)
    }

    /// Single-weight JANGH passes for a non-SiLU activation: decode QMVs rotating the raw input to H32 in F32 (the
    /// basis the fused decode kernel reads) and the up projection's own prefill tile (the gate's is `prefill`).
    private struct SeparateCodebookPasses {
        let gateDecode: JANGHProjectionKernel
        let upDecode: JANGHProjectionKernel
        let upPrefill: JANGHPrefillKernel
    }

    private enum Down {
        case trellis(JANGTProjection, signs: MLXArray)
        case codebook(kernel: JANGHWeightedDownKernel, prefill: JANGHPrefillKernel, bank: JANGHMappedBanks.Projection)
    }

    /// Routed (token, slot) pairs from which the expert-sorted prefill tiles run instead of the decode kernels.
    /// Measured crossover, one process alternating both routes (whole-model forward, 600-token context): GLM-5.3
    /// JANGHT2.4 between 512 slots (tiles 497 ms, decode 434) and 768 (571 vs 618); Naive-N0.5 JANGHT2.5 between 384
    /// (408 vs 371) and 512 (459 vs 479). The row-tiled tiles carry a ~250 ms whole-model fixed cost on sparse
    /// batches, so the 72 vMLX Python uses (`SORT_THRESHOLD`) sent 9-16-row DFlash 2 verifies to them at 2.3-3.5x
    /// the decode-kernel cost (GLM 9 rows: 346 vs 98 ms).
    static let sortThreshold = 512
    /// `VMLX_JANGT_PREFILL=0` keeps the decode kernels for every batch (A/B and diagnosis).
    static let prefillEnabled = ProcessInfo.processInfo.environment["VMLX_JANGT_PREFILL"] != "0"

    /// True when `callRouted` runs every one of `pairs` routed (token, slot) pairs of a `dtype` input on the decode
    /// kernels, the kernels a one-token call uses (a DSpark verify batches its rows' routed experts only then).
    public static func usesDecodeKernels(pairs: Int, dtype: DType) -> Bool {
        !(prefillEnabled && pairs >= sortThreshold && JANGTPrefillKernel.available
            && [.bfloat16, .float16].contains(dtype))
    }
    private let intermediateSize: Int
    /// The gate/up outlier columns when they are layer-wide (the same real columns for every expert, gate and up):
    /// the fused prefill epilogue needs one column list per tile. Derived, not a bundle tensor: the leading
    /// underscore keeps it out of `parameters()` (MLXNN reflects every stored MLXArray as a parameter).
    private var _sharedOutlierColumns: MLXArray?

    private let gateUp: GateUp
    private let down: Down
    private let hiddenSize: Int
    private let limit: Float
    private let activation: JANGTActivation
    private let rowRotation = JANGHRowRotation()
    public let supplementalWeightBytes: Int
    public let supplementalParameterCount: Int

    init(banks: JANGHMappedBanks, parentModule: String, hiddenSize: Int, intermediateSize: Int,
         activationLimit: Float?, activation: JANGTActivation = .silu) throws {
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
                if shared { _sharedOutlierColumns = first.reshaped(-1).asType(.int32) }
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
                gate: gate, up: up,
                separate: activation == .silu ? nil : SeparateCodebookPasses(
                    gateDecode: try JANGHProjectionKernel(contract: contract, module: gateName,
                                                          fuseFloatRotation: true),
                    upDecode: try JANGHProjectionKernel(contract: contract, module: upName, fuseFloatRotation: true),
                    upPrefill: try JANGHPrefillKernel(contract: contract, module: upName)))
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
        self.activation = activation
        supplementalWeightBytes = bytes
        supplementalParameterCount = params
        super.init()
    }

    /// A pure-trellis layer from in-memory projections (tensors not served by the mapped banks, e.g. the DeepSeek-V4.1
    /// dSpark draft's `dspark/dspark.safetensors`): the same folded signs and kernels as the banked trellis branches.
    public init(gate: JANGTProjection, up: JANGTProjection, down d: JANGTProjection,
                hiddenSize: Int, intermediateSize: Int, activationLimit: Float?,
                activation: JANGTActivation = .silu) throws {
        if let activationLimit, !activationLimit.isFinite || activationLimit <= 0 {
            throw JANGHFormatContract.ValidationError.invalid("invalid JANGT activation limit")
        }
        guard gate.inputs == hiddenSize, gate.outputs == intermediateSize, up.inputs == hiddenSize,
            up.outputs == intermediateSize, up.experts == gate.experts, up.k == gate.k,
            d.inputs == intermediateSize, d.outputs == hiddenSize, d.experts == gate.experts
        else { throw JANGHFormatContract.ValidationError.invalid("JANGT projection geometry mismatch") }
        let signs = gate.signs * Self.keepMask(width: hiddenSize, columns: gate.outlierColumnIndices[0 ..< 1])[0]
        eval(signs)
        gateUp = .trellis(gate: gate, up: up, signs: signs)
        if gate.outlierColumns > 0 {
            let g = gate.outlierColumnIndices, u = up.outlierColumnIndices
            let first = g[0 ..< 1]
            let shared = g.shape == u.shape && (g .== first).all().item(Bool.self)
                && (g .== u).all().item(Bool.self) && (first .>= MLXArray(Int32(0))).all().item(Bool.self)
            if shared { _sharedOutlierColumns = first.reshaped(-1).asType(.int32) }
        }
        let cs = d.columnScales ?? MLXArray.ones([d.experts, intermediateSize], dtype: .float32)
        let dSigns = cs * d.signs.expandedDimensions(axis: 0)
            * Self.keepMask(width: intermediateSize, columns: d.outlierColumnIndices)
        eval(dSigns)
        down = .trellis(d, signs: dSigns)
        var bytes = 0, params = 0
        for p in [gate, up, d] {
            bytes += p.packed.nbytes + p.scale.nbytes + p.signs.nbytes + p.outlierColumnIndices.nbytes
                + p.outlierWeights.nbytes + (p.columnScales?.nbytes ?? 0)
            params += p.experts * p.outputs * p.inputs
        }
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.limit = activationLimit ?? 0
        self.activation = activation
        supplementalWeightBytes = bytes
        supplementalParameterCount = params
        super.init()
    }

    /// (R, width) float32: 0 at each row's outlier columns (-1 entries are padding), else 1.
    /// A pure-trellis layer's tensors as the decode kernels read them (fused-kernel hosts, e.g. vMLX's MiniMax-M3
    /// fused MoE block): the gate / up / down projections, the gate/up input signs with the layer's outlier columns
    /// zeroed (Python `_unit()["su_gu"]`), the down input's per-expert column scales x signs with each expert's outlier
    /// columns zeroed (`sv_dn`), and the activation. nil for codebook (JANGH) groups.
    public struct TrellisTensors {
        public let gate: JANGTProjection
        public let up: JANGTProjection
        public let gateUpSigns: MLXArray
        public let down: JANGTProjection
        public let downSigns: MLXArray
        public let hiddenSize: Int
        public let intermediateSize: Int
        /// The SwiGLU clamp (0: none).
        public let limit: Float
        public let activation: JANGTActivation
    }

    /// vMLX's `moe_kernels.rotate_rows` (same source and kernel name): x (R, K) -> float32 H128(x * s), s (K,) shared or
    /// (E, K) selected per row by `rowExperts` (R,).
    public static func rotateRows(_ x: MLXArray, signs: MLXArray, rowExperts: MLXArray? = nil) -> MLXArray {
        JANGTKernels.rotateRows(x, signs: signs, rowExperts: rowExperts)
    }

    public var trellisTensors: TrellisTensors? {
        guard case .trellis(let gate, let up, let signs) = gateUp, case .trellis(let d, let dSigns) = down else {
            return nil
        }
        return TrellisTensors(
            gate: gate, up: up, gateUpSigns: signs, down: d, downSigns: dSigns, hiddenSize: hiddenSize,
            intermediateSize: intermediateSize, limit: limit, activation: activation)
    }

    static func keepMask(width: Int, columns: MLXArray) -> MLXArray {
        let positions = MLXArray(0 ..< Int32(width)).reshaped(1, width, 1)
        let hit = positions .== columns.asType(.int32).expandedDimensions(axis: 1)
        return MLX.where(hit.any(axis: -1), MLXArray(Float(0)), MLXArray(Float(1)))
    }

    /// True when `callRoutedSorted` runs the expert-sorted prefill tiles for a `dtype` input (VMLX_JANGT_PREFILL on, the
    /// NAX kernel present, bf16 / fp16), at any row count.
    public static func sortedPrefillAvailable(dtype: DType) -> Bool {
        prefillEnabled && JANGTPrefillKernel.available && [.bfloat16, .float16].contains(dtype)
    }

    /// `callRouted` on the expert-sorted prefill tiles whatever the row count (`sortedPrefillAvailable`; otherwise the
    /// ordinary `callRouted`). A row's output there depends only on its own input and routing, so a prompt prefill
    /// that sends every piece here gets the same rows whether its pieces are called one by one or concatenated.
    public func callRoutedSorted(_ input: MLXArray, indices: MLXArray, scores: MLXArray) -> MLXArray {
        guard Self.sortedPrefillAvailable(dtype: input.dtype), input.ndim >= 2, input.dim(-1) == hiddenSize,
            indices.ndim == input.ndim, indices.shape.dropLast() == input.shape.dropLast(), indices.dim(-1) > 0,
            scores.shape == indices.shape
        else { return callRouted(input, indices: indices, scores: scores) }
        let routes = indices.dim(-1)
        do {
            return try prefill(
                input.reshaped(-1, hiddenSize), ids: indices.reshaped(-1, routes).asType(.uint32),
                weights: scores.reshaped(-1, routes).asType(.float32)
            ).asType(input.dtype).reshaped(input.shape)
        } catch {
            preconditionFailure("JANGT routed prefill rejected admitted bank: \(error)")
        }
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
        if JANGTFastDecode.isEnabled, let t = trellisTensors, JANGTFastDecode.supported(t) {
            // VMLX_JANGT_FAST=1: vMLX's fast decode kernels (packed rows, folded code statistics) for decode steps and
            // verify rows alike: row-invariant, not bit-identical to the v3 kernels below (Python
            // `JTMixedSwitchGLU.routed` with the lever).
            return JANGTFastDecode.routed(x, ids: ids, weights: weights, t).asType(input.dtype).reshaped(input.shape)
        }
        do {
            let hidden: MLXArray
            switch gateUp {
            case .trellis(let gate, let up, let signs):
                hidden = JANGTKernels.gateUp(
                    xr: JANGTKernels.rotateRows(x, signs: signs), x: x, indices: ids, gate: gate, up: up,
                    limit: limit, activation: activation, outputDType: .float32)
            case .codebook(_, _, let gate, let up, let separate?):
                let g = try separate.gateDecode.project(x, packed: gate.packed, scales: gate.scales, indices: ids)
                let u = try separate.upDecode.project(x, packed: up.packed, scales: up.scales, indices: ids)
                hidden = activation(gate: g, up: u, limit: limit)
            case .codebook(let kernel, _, let gate, let up, nil):
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

    /// The token of each sorted (token, slot) pair: `order / routes` for argSort's non-negative positions. A right
    /// shift when `routes` is a power of two (top-4 / top-8): exact, and one kernel. MLX's integer `floorDivide` is a
    /// composed op whose parts land on the C++ thread-local default stream, not Swift's: every call split the command
    /// stream (4 command buffers, ~0.3 ms of sync; 01-LOG 10-28 "Integer floorDivide"), once per layer per chunk here.
    static func sourceRows(_ order: MLXArray, routes: Int) -> MLXArray {
        if routes > 0 && routes & (routes - 1) == 0 {
            return rightShift(order, MLXArray(UInt32(routes.trailingZeroBitCount)).asType(order.dtype))
        }
        return order.floorDivide(routes)
    }

    /// Expert-sorted prefill (vMLX Python `JTMixedSwitchGLU._prefill`): rows sorted by expert, NAX tiles per group,
    /// outlier columns added in fp32, rows restored and combined with the routing weights in fp32.
    ///
    /// The trellis tiles run the expert-aligned schedule (one GPU-side `expertTilePlan` per layer, shared by gate/up
    /// and down; vMLX Python edf7e36e8) unless `VMLX_JANGT_EXPERT_TILES=0`: same rows, same arithmetic per row.
    private func prefill(_ x: MLXArray, ids: MLXArray, weights: MLXArray) throws -> MLXArray {
        let tokens = x.dim(0), routes = ids.dim(1)
        let flat = ids.flattened()
        let order = argSort(flat)
        let sortedIDs = take(flat, order).asType(.uint32)
        let tok = Self.sourceRows(order, routes: routes)
        let dtype = x.dtype
        let downIsCodebook: Bool
        if case .codebook = down { downIsCodebook = true } else { downIsCodebook = false }
        var plan: JANGTPrefillKernel.ExpertTilePlan?
        if JANGTPrefillKernel.expertTilesEnabled {
            var experts: Int?
            if case .trellis(let gate, _, _) = gateUp { experts = gate.experts }
            if case .trellis(let projection, _) = down { experts = projection.experts }
            plan = experts.map { JANGTPrefillKernel.expertTilePlan(sortedIndices: sortedIDs, experts: $0) }
        }

        var hidden: MLXArray
        var hiddenRotated = false
        switch gateUp {
        case .trellis(let gate, let up, let signs):
            // Narrow to the input dtype before gathering the k copies of each row (exact: an elementwise cast).
            let xr = take(JANGTKernels.rotateRows(x, signs: signs).asType(dtype), tok, axis: 0)
            if gate.outlierColumns == 0 {
                // One NAX pass over both weight tiles with the (clamped) SwiGLU epilogue; a JANGH down gets its
                // input already Hadamard-32 rotated (the H32 store epilogue).
                let rotate = downIsCodebook && intermediateSize % 32 == 0
                hidden = JANGTPrefillKernel.project(xr, gate, up: up, indices: sortedIDs, limit: limit,
                                                    activation: activation, rotateOutput: rotate, plan: plan)
                hiddenRotated = rotate
            } else if let columns = _sharedOutlierColumns {
                // The layer-wide outlier columns of each token, then its k copies (not k copies of every raw row).
                hidden = JANGTPrefillKernel.projectGateUpWithOutliers(
                    xr, outlierRows: take(take(x, columns, axis: 1).asType(.float32), tok, axis: 0),
                    gate: gate, up: up, columns: columns.size, indices: sortedIDs, limit: limit,
                    activation: activation, plan: plan)
            } else {
                let raw = take(x, tok, axis: 0)
                var g = JANGTPrefillKernel.project(xr, gate, indices: sortedIDs, plan: plan).asType(.float32)
                var u = JANGTPrefillKernel.project(xr, up, indices: sortedIDs, plan: plan).asType(.float32)
                if let o = JANGTPrefillKernel.outliers(raw, gate, indices: sortedIDs) { g = g + o }
                if let o = JANGTPrefillKernel.outliers(raw, up, indices: sortedIDs) { u = u + o }
                hidden = activation(gate: g, up: u, limit: limit).asType(dtype)
            }
        case .codebook(_, let prefillKernel, let gate, let up, let separate):
            let xs = take(try rowRotation(x, outputDType: dtype), tok, axis: 0)
            if let separate {
                let g = try prefillKernel.projectSorted(
                    xs, packed: gate.packed, scales: gate.scales, indices: sortedIDs).asType(.float32)
                let u = try separate.upPrefill.projectSorted(
                    xs, packed: up.packed, scales: up.scales, indices: sortedIDs).asType(.float32)
                hidden = activation(gate: g, up: u, limit: limit)
            } else {
                hidden = try prefillKernel.projectSorted(
                    xs, packed: gate.packed, scales: gate.scales, indices: sortedIDs,
                    upPacked: up.packed, upScales: up.scales, limit: limit > 0 ? limit : nil)
            }
        }

        let sortedOutput: MLXArray
        switch down {
        case .trellis(let projection, let signs):
            let hr = JANGTKernels.rotateRows(hidden, signs: signs, rowExperts: sortedIDs).asType(dtype)
            var y = JANGTPrefillKernel.project(hr, projection, indices: sortedIDs, plan: plan).asType(.float32)
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
