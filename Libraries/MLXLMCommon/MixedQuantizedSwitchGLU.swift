// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import MLX
import MLXNN

/// Native expert projections backed by exact file regions. All mappings are
/// created during loading, so failures propagate through the throwing loader.
/// These are read-only mappings, not copied banks: macOS can reclaim cold pages.
public final class MixedQuantizedSwitchGLU: Module, SwitchGLULayer {
    private let experts: [MixedQuantizedExpertCatalog.Expert]
    private let kernels = MixedQuantizedExpertKernel()
    private let inputDimensions: Int

    public init(catalog: MixedQuantizedExpertCatalog, layer: Int, inputDimensions: Int) throws {
        let mapped = try (0..<catalog.expertCount).map {
            try Task.checkCancellation()
            return try catalog.loadExpert(layer: layer, index: $0)
        }
        self.experts = mapped
        self.inputDimensions = inputDimensions
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
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
