import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Tensor-parallel routed-expert projection. Each rank holds an output shard
/// for every expert, so the routed result is sharded on the feature axis.
open class AllToShardedSwitchLinear: SwitchLinear {
    public let group: Group

    public static func from(
        _ linear: SwitchLinear,
        group: Group? = nil,
        segments: Int = 1
    ) -> AllToShardedSwitchLinear {
        let g = group ?? Group(strict: false)
        let shape = linear.weight.shape
        precondition(shape.count == 3, "SwitchLinear weight must be [experts, out, in]")
        let experts = shape[0]
        let out = shape[1]
        let input = shape[2]
        precondition(out % segments == 0, "output dims must be divisible by segments")
        let perSegmentOut = out / segments
        precondition(
            perSegmentOut % g.size == 0,
            "(output dims / segments) must be divisible by group size")

        let perRankOut = perSegmentOut / g.size
        var chunks: [MLXArray] = []
        for segment in 0 ..< segments {
            let segmentStart = segment * perSegmentOut
            let rankStart = segmentStart + g.rank * perRankOut
            let rankEnd = rankStart + perRankOut
            chunks.append(linear.weight[0..., rankStart ..< rankEnd, 0...])
        }
        let shardedWeight = concatenated(chunks, axis: 1)

        var shardedBias: MLXArray? = nil
        if let bias = linear.bias {
            var biasChunks: [MLXArray] = []
            for segment in 0 ..< segments {
                let segmentStart = segment * perSegmentOut
                let rankStart = segmentStart + g.rank * perRankOut
                let rankEnd = rankStart + perRankOut
                biasChunks.append(bias[0..., rankStart ..< rankEnd])
            }
            shardedBias = concatenated(biasChunks, axis: 1)
        }

        return AllToShardedSwitchLinear(
            inputDims: input,
            outputDims: shardedWeight.dim(1),
            numExperts: experts,
            weight: shardedWeight,
            bias: shardedBias,
            group: g)
    }

    public init(
        inputDims: Int,
        outputDims: Int,
        numExperts: Int,
        weight: MLXArray,
        bias: MLXArray?,
        group: Group
    ) {
        self.group = group
        super.init(
            inputDims: inputDims,
            outputDims: outputDims,
            numExperts: numExperts,
            weight: weight,
            bias: bias)
    }
}

/// Quantized tensor-parallel routed-expert projection. Each rank holds an
/// output shard for every expert while preserving packed weights, scales, and
/// affine zero-points so `SwitchGLU` can keep using `gatherQuantizedMM`.
open class QuantizedAllToShardedSwitchLinear: QuantizedSwitchLinear {
    public let group: Group

    public static func from(
        _ linear: QuantizedSwitchLinear,
        group: Group? = nil,
        segments: Int = 1
    ) -> QuantizedAllToShardedSwitchLinear {
        let g = group ?? Group(strict: false)
        let shape = linear.weight.shape
        precondition(shape.count == 3, "QuantizedSwitchLinear weight must be [experts, out, in_packed]")
        let experts = shape[0]
        let out = shape[1]
        let inputPacked = shape[2]
        let input = (inputPacked * 32) / linear.bits
        precondition(out % segments == 0, "output dims must be divisible by segments")
        let perSegmentOut = out / segments
        precondition(
            perSegmentOut % g.size == 0,
            "(output dims / segments) must be divisible by group size")

        let perRankOut = perSegmentOut / g.size
        var weightChunks: [MLXArray] = []
        var scaleChunks: [MLXArray] = []
        var biasChunks: [MLXArray] = []
        var quantBiasChunks: [MLXArray] = []
        for segment in 0 ..< segments {
            let segmentStart = segment * perSegmentOut
            let rankStart = segmentStart + g.rank * perRankOut
            let rankEnd = rankStart + perRankOut
            weightChunks.append(linear.weight[0..., rankStart ..< rankEnd, 0...])
            scaleChunks.append(linear.scales[0..., rankStart ..< rankEnd, 0...])
            if let bias = linear.bias {
                biasChunks.append(bias[0..., rankStart ..< rankEnd])
            }
            if let biases = linear.biases {
                quantBiasChunks.append(biases[0..., rankStart ..< rankEnd, 0...])
            }
        }

        let shardedBias = biasChunks.isEmpty ? nil : concatenated(biasChunks, axis: 1)
        let shardedQuantBias = quantBiasChunks.isEmpty ? nil : concatenated(quantBiasChunks, axis: 1)

        return QuantizedAllToShardedSwitchLinear(
            inputDims: input,
            outputDims: perRankOut * segments,
            numExperts: experts,
            weight: concatenated(weightChunks, axis: 1),
            bias: shardedBias,
            scales: concatenated(scaleChunks, axis: 1),
            biases: shardedQuantBias,
            groupSize: linear.groupSize,
            bits: linear.bits,
            mode: linear.mode,
            group: g)
    }

    public init(
        inputDims: Int,
        outputDims: Int,
        numExperts: Int,
        weight: MLXArray,
        bias: MLXArray?,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode,
        group: Group
    ) {
        self.group = group
        super.init(
            inputDims: inputDims,
            outputDims: outputDims,
            numExperts: numExperts,
            weight: weight,
            bias: bias,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            mode: mode)
    }
}

/// Tensor-parallel routed-expert projection that consumes a sharded feature
/// axis and all-reduces partial expert outputs back to full hidden size.
open class ShardedToAllSwitchLinear: SwitchLinear {
    public let group: Group

    public static func from(
        _ linear: SwitchLinear,
        group: Group? = nil,
        segments: Int = 1
    ) -> ShardedToAllSwitchLinear {
        let g = group ?? Group(strict: false)
        let shape = linear.weight.shape
        precondition(shape.count == 3, "SwitchLinear weight must be [experts, out, in]")
        let experts = shape[0]
        let out = shape[1]
        let input = shape[2]
        precondition(input % segments == 0, "input dims must be divisible by segments")
        let perSegmentIn = input / segments
        precondition(
            perSegmentIn % g.size == 0,
            "(input dims / segments) must be divisible by group size")

        let perRankIn = perSegmentIn / g.size
        var chunks: [MLXArray] = []
        for segment in 0 ..< segments {
            let segmentStart = segment * perSegmentIn
            let rankStart = segmentStart + g.rank * perRankIn
            let rankEnd = rankStart + perRankIn
            chunks.append(linear.weight[0..., 0..., rankStart ..< rankEnd])
        }
        let shardedWeight = concatenated(chunks, axis: 2)

        return ShardedToAllSwitchLinear(
            inputDims: shardedWeight.dim(2),
            outputDims: out,
            numExperts: experts,
            weight: shardedWeight,
            bias: linear.bias,
            group: g)
    }

    public init(
        inputDims: Int,
        outputDims: Int,
        numExperts: Int,
        weight: MLXArray,
        bias: MLXArray?,
        group: Group
    ) {
        self.group = group
        super.init(
            inputDims: inputDims,
            outputDims: outputDims,
            numExperts: numExperts,
            weight: weight,
            bias: bias)
    }

    public override func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let weightT = self.weight.swappedAxes(-1, -2)
        let partial = MLX.gatherMM(x, weightT, rhsIndices: indices, sortedIndices: sortedIndices)
        var result = Collectives.allSum(partial, group: group)
        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }
        return result
    }
}

/// Quantized tensor-parallel routed-expert projection that consumes a sharded
/// feature axis and all-reduces partial quantized expert outputs back to the
/// full hidden size.
open class QuantizedShardedToAllSwitchLinear: QuantizedSwitchLinear {
    public let group: Group

    public static func from(
        _ linear: QuantizedSwitchLinear,
        group: Group? = nil,
        segments: Int = 1
    ) -> QuantizedShardedToAllSwitchLinear {
        let g = group ?? Group(strict: false)
        let shape = linear.weight.shape
        precondition(shape.count == 3, "QuantizedSwitchLinear weight must be [experts, out, in_packed]")
        let experts = shape[0]
        let out = shape[1]
        let inputPacked = shape[2]
        let input = (inputPacked * 32) / linear.bits
        precondition(input % segments == 0, "input dims must be divisible by segments")
        let perSegmentIn = input / segments
        precondition(
            perSegmentIn % g.size == 0,
            "(input dims / segments) must be divisible by group size")

        let packFactor = 32 / linear.bits
        let groupSize = linear.groupSize
        let perRankIn = perSegmentIn / g.size
        var weightChunks: [MLXArray] = []
        var scaleChunks: [MLXArray] = []
        var quantBiasChunks: [MLXArray] = []
        for segment in 0 ..< segments {
            let segmentStart = segment * perSegmentIn
            let rankStart = segmentStart + g.rank * perRankIn
            let rankEnd = rankStart + perRankIn
            precondition(rankStart % packFactor == 0 && rankEnd % packFactor == 0)
            precondition(rankStart % groupSize == 0 && rankEnd % groupSize == 0)
            let packedStart = rankStart / packFactor
            let packedEnd = rankEnd / packFactor
            let scaleStart = rankStart / groupSize
            let scaleEnd = rankEnd / groupSize
            weightChunks.append(linear.weight[0..., 0..., packedStart ..< packedEnd])
            scaleChunks.append(linear.scales[0..., 0..., scaleStart ..< scaleEnd])
            if let biases = linear.biases {
                quantBiasChunks.append(biases[0..., 0..., scaleStart ..< scaleEnd])
            }
        }

        let shardedQuantBias = quantBiasChunks.isEmpty ? nil : concatenated(quantBiasChunks, axis: 2)

        return QuantizedShardedToAllSwitchLinear(
            inputDims: perRankIn * segments,
            outputDims: out,
            numExperts: experts,
            weight: concatenated(weightChunks, axis: 2),
            bias: linear.bias,
            scales: concatenated(scaleChunks, axis: 2),
            biases: shardedQuantBias,
            groupSize: linear.groupSize,
            bits: linear.bits,
            mode: linear.mode,
            group: g)
    }

    public init(
        inputDims: Int,
        outputDims: Int,
        numExperts: Int,
        weight: MLXArray,
        bias: MLXArray?,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode,
        group: Group
    ) {
        self.group = group
        super.init(
            inputDims: inputDims,
            outputDims: outputDims,
            numExperts: numExperts,
            weight: weight,
            bias: bias,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            mode: mode)
    }

    public override func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let partial = MLX.gatherQuantizedMM(
            x,
            self.weight,
            scales: self.scales,
            biases: self.biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits,
            mode: self.mode,
            sortedIndices: sortedIndices)
        var result = Collectives.allSum(partial, group: group)
        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }
        return result
    }
}
