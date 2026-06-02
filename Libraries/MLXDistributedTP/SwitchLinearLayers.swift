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
