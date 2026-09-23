// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import MLX
import MLXNN

/// Native mixed-quantized projections. Resident banks keep routing on the GPU;
/// explicit mapped diagnostics retain exact file regions and host routing.
public final class MixedQuantizedSwitchGLU: Module, SwitchGLULayer {
    private let experts: [MixedQuantizedExpertCatalog.Expert]
    private let resident: MixedQuantizedExpertCatalog.Expert?
    public var usesResidentGPURouting: Bool { resident != nil }
    private let kernels = MixedQuantizedExpertKernel()
    private let inputDimensions: Int
    private let pairedGateUp = RuntimeEnvironment.value("VMLX_MIMO_PAIRED_GATE_UP") == "1"
    private let fusedGateUp = RuntimeEnvironment.value("VMLX_MIMO_FUSED_GATE_UP") == "1"
    private let fusedDownReduce = RuntimeEnvironment.value("VMLX_MIMO_FUSED_DOWN_REDUCE") == "1"
    // Match native SwitchGLU's grouped GEMM prefill; retain an explicit A/B opt-out.
    private let sortPrefill = RuntimeEnvironment.value("VMLX_MIMO_SORT_PREFILL") != "0"

    public init(catalog: MixedQuantizedExpertCatalog, layer: Int, inputDimensions: Int,
                storage: MixedQuantizedExpertCatalog.Storage = .mapped) throws {
        if storage == .resident && RuntimeEnvironment.value("VMLX_MIMO_RESIDENT_ROUTING") != "host" {
            self.resident = try catalog.loadResidentLayer(layer: layer)
            self.experts = []
        } else {
            self.resident = nil
            self.experts = try catalog.loadExperts(layer: layer, storage: storage)
        }
        self.inputDimensions = inputDimensions
        super.init()
    }

    /// Optional single-token weighted path; all other shapes use the ordinary
    /// expert outputs and caller reduction. No route indices cross to the CPU.
    public func fusedWeightedOutput(_ x: MLXArray, _ indices: MLXArray,
                                    scores: MLXArray) -> MLXArray? {
        guard fusedDownReduce, let resident, x.size == inputDimensions,
            x.dtype == .bfloat16, indices.size == 8,
            scores.shape == indices.shape, scores.dtype == .float32,
            resident.down.bits == 2, resident.down.groupSize == 64,
            resident.down.mode == .affine else { return nil }
        let activated: MLXArray
        if fusedGateUp, let fused = kernels.fusedGateUp(x, indices: indices,
            gate: resident.gate, up: resident.up) {
            activated = fused
        } else {
            let input = expandedDimensions(x, axes: [-2, -3])
            func project(_ p: MixedQuantizedExpertCatalog.Projection) -> MLXArray {
                gatherQuantizedMM(input, p.weight, scales: p.scales, biases: p.biases,
                    rhsIndices: indices, groupSize: p.groupSize, bits: p.bits, mode: p.mode)
            }
            let pair = pairedGateUp ? kernels.pairedGateUp(x, indices: indices,
                gate: resident.gate, up: resident.up) : nil
            activated = silu(pair?[0] ?? project(resident.gate)) * (pair?[1] ?? project(resident.up))
        }
        return kernels.fusedDownReduce(activated, indices: indices, scores: scores, down: resident.down)
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        if let resident {
            let doSort = sortPrefill && indices.size >= 64
            var input = expandedDimensions(x, axes: [-2, -3])
            var routes = indices
            var inverseOrder = MLXArray()
            if doSort {
                (input, routes, inverseOrder) = gatherSort(x: input, indices: indices)
            }
            func project(_ input: MLXArray, _ projection: MixedQuantizedExpertCatalog.Projection) -> MLXArray {
                gatherQuantizedMM(input, projection.weight, scales: projection.scales,
                    biases: projection.biases, rhsIndices: routes,
                    groupSize: projection.groupSize, bits: projection.bits,
                    mode: projection.mode, sortedIndices: doSort)
            }
            let activated: MLXArray
            if fusedGateUp, let fused = kernels.fusedGateUp(x, indices: indices,
                gate: resident.gate, up: resident.up) {
                activated = fused
            } else {
                let pair = pairedGateUp ? kernels.pairedGateUp(x, indices: indices,
                    gate: resident.gate, up: resident.up) : nil
                activated = silu(pair?[0] ?? project(input, resident.gate))
                    * (pair?[1] ?? project(input, resident.up))
            }
            var output = project(activated, resident.down)
            if doSort {
                output = scatterUnsort(x: output, invOrder: inverseOrder, shape: indices.shape)
            }
            return output.squeezed(axis: -2)
        }
        let routes = indices.asArray(Int32.self)
        let k = indices.dim(-1)
        // Router indices come from argPartition over the configured expert
        // dimension. These checks catch a model/programming contract violation.
        precondition(x.dim(-1) == inputDimensions && k > 0)
        precondition(routes.count == x.size / inputDimensions * k)
        precondition(routes.allSatisfy { $0 >= 0 && $0 < experts.count })
        let selected = routes.map { experts[Int($0)] }

        func fast(_ projection: MixedQuantizedExpertCatalog.Projection) -> Bool {
            let width = projection.weight.dim(1) * 32 / projection.bits
            return width >= 512 && width.isMultiple(of: 512)
                && projection.weight.dim(0).isMultiple(of: 8)
                && (projection.mode == .mxfp4 || projection.scales.dtype == .bfloat16)
        }
        if x.size == inputDimensions, k == 8, x.dtype == .bfloat16,
            selected.allSatisfy({ fast($0.gate) && fast($0.up) && fast($0.down) }) {
            func project(_ input: MLXArray,
                         _ key: KeyPath<MixedQuantizedExpertCatalog.Expert, MixedQuantizedExpertCatalog.Projection>,
                         individual: Bool) -> MLXArray {
                let projections = selected.map { $0[keyPath: key] }
                let first = projections[0]
                return kernels.call(contiguous(input),
                    weights: projections.map { ($0.weight, $0.scales, $0.biases) },
                    bits: first.bits, group: first.groupSize, mode: first.mode,
                    perExpertInput: individual)
            }
            let gate = project(x, \.gate, individual: false)
            let up = project(x, \.up, individual: false)
            let down = project(silu(gate) * up, \.down, individual: true)
            return down.reshaped(Array(x.shape.dropLast()) + [k, inputDimensions])
        }

        // Group prefill routes by expert without copying a weight bank, then
        // restore the router's original order before the caller reduces scores.
        var groups: [Int: [Int]] = [:]
        for (position, expert) in routes.enumerated() {
            groups[Int(expert), default: []].append(position)
        }
        let flat = x.reshaped(-1, inputDimensions)
        var values: [MLXArray] = [], order: [Int] = []
        func project(_ x: MLXArray, _ p: MixedQuantizedExpertCatalog.Projection) -> MLXArray {
            quantizedMM(x, p.weight, scales: p.scales, biases: p.biases,
                        groupSize: p.groupSize, bits: p.bits, mode: p.mode)
        }
        for index in groups.keys.sorted() {
            let positions = groups[index]!, expert = experts[index]
            let input = flat[MLXArray(positions.map { Int32($0 / k) })]
            values.append(project(silu(project(input, expert.gate)) * project(input, expert.up), expert.down))
            order.append(contentsOf: positions)
        }
        var inverse = Array(repeating: Int32(0), count: order.count)
        for (row, position) in order.enumerated() { inverse[position] = Int32(row) }
        return concatenated(values, axis: 0)[MLXArray(inverse)]
            .reshaped(Array(x.shape.dropLast()) + [k, inputDimensions])
    }
}
