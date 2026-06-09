//
//  RoPEUtils.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2025/8/11.
//

import Foundation
import MLX
import MLXNN

public class Llama3RoPE: Module, OffsetLayer, ArrayOffsetLayer {
    let dims: Int
    let maxPositionEmbeddings: Int
    let traditional: Bool
    let _freqs: MLXArray

    init(
        dims: Int,
        maxPositionEmbeddings: Int = 2048,
        traditional: Bool = false,
        base: Float = 10000,
        scalingConfig: [String: StringOrNumber]? = nil
    ) {
        self.dims = dims
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.traditional = traditional

        guard let scalingConfig = scalingConfig else {
            fatalError("Llama3RoPE requires scaling_config")
        }

        let factor = scalingConfig["factor"]?.asFloat() ?? 1.0
        let lowFreqFactor = scalingConfig["low_freq_factor"]?.asFloat() ?? 1.0
        let highFreqFactor = scalingConfig["high_freq_factor"]?.asFloat() ?? 4.0
        let oldContextLen = scalingConfig["original_max_position_embeddings"]?.asFloat() ?? 8192.0

        let lowFreqWavelen = oldContextLen / lowFreqFactor
        let highFreqWavelen = oldContextLen / highFreqFactor

        let indices = MLXArray(stride(from: 0, to: dims, by: 2))
        var frequencies = MLX.pow(base, indices / Float(dims))
        let wavelens = 2 * Float.pi * frequencies

        frequencies = MLX.where(
            wavelens .> MLXArray(lowFreqWavelen),
            frequencies * factor,
            frequencies
        )

        let isMediumFreq = MLX.logicalAnd(
            wavelens .> MLXArray(highFreqWavelen),
            wavelens .< MLXArray(lowFreqWavelen)
        )

        let smoothFactors =
            (oldContextLen / wavelens - lowFreqFactor) / (highFreqFactor - lowFreqFactor)
        let smoothFreqs = frequencies / ((1 - smoothFactors) / factor + smoothFactors)

        self._freqs = MLX.where(isMediumFreq, smoothFreqs, frequencies)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x,
            dimensions: dims,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: _freqs
        )
    }

    public func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        MLXFast.RoPE(
            x,
            dimensions: dims,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: _freqs
        )
    }

}

public class YarnRoPE: Module, OffsetLayer, ArrayOffsetLayer {
    let dimensions: Int
    let traditional: Bool

    private let _mscale: Float
    private let _freqs: MLXArray

    public init(
        dimensions: Int,
        traditional: Bool = false,
        maxPositionEmbeddings: Int = 2048,
        base: Float = 10000,
        scalingFactor: Float = 1.0,
        originalMaxPositionEmbeddings: Int = 4096,
        betaFast: Float = 32,
        betaSlow: Float = 1,
        mscale: Float = 1,
        mscaleAllDim: Float = 0
    ) {
        precondition(dimensions % 2 == 0, "Dimensions must be even")

        self.dimensions = dimensions
        self.traditional = traditional

        func yarnFindCorrectionDim(numRotations: Float) -> Float {
            return Float(dimensions)
                * log(Float(originalMaxPositionEmbeddings) / (numRotations * 2 * Float.pi))
                / (2 * log(base))
        }

        func yarnFindCorrectionRange() -> (low: Int, high: Int) {
            let low = Int(floor(yarnFindCorrectionDim(numRotations: betaFast)))
            let high = Int(ceil(yarnFindCorrectionDim(numRotations: betaSlow)))
            return (max(low, 0), min(high, dimensions - 1))
        }

        func yarnGetMscale(scale: Float, mscale: Float) -> Float {
            if scale <= 1 {
                return 1.0
            }
            return 0.1 * mscale * log(scale) + 1.0
        }

        func yarnLinearRampMask(minVal: Float, maxVal: Float, dim: Int) -> MLXArray {
            var maxVal = maxVal
            if minVal == maxVal {
                maxVal += 0.001
            }

            let linearFunc = (MLXArray(0 ..< dim).asType(.float32) - minVal) / (maxVal - minVal)
            return clip(linearFunc, min: 0, max: 1)
        }

        self._mscale =
            yarnGetMscale(scale: scalingFactor, mscale: mscale)
            / yarnGetMscale(scale: scalingFactor, mscale: mscaleAllDim)

        let freqExtra = pow(
            base,
            MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
                / dimensions)
        let freqInter =
            scalingFactor
            * pow(
                base,
                MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
                    / dimensions)

        let (low, high) = yarnFindCorrectionRange()
        let freqMask =
            1.0 - yarnLinearRampMask(minVal: Float(low), maxVal: Float(high), dim: dimensions / 2)

        self._freqs = (freqInter * freqExtra) / (freqInter * freqMask + freqExtra * (1 - freqMask))
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        // "copy" of x as we are going to write through it and don't want to update
        // through the reference
        // https://github.com/ml-explore/mlx-swift/issues/364
        var x = x
        if _mscale != 1.0 {
            x = x[0..., .ellipsis]
            x[.ellipsis, 0 ..< dimensions] *= _mscale
        }

        return MLXFast.RoPE(
            x,
            dimensions: dimensions,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: self._freqs
        )
    }

    public func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        var x = x
        if _mscale != 1.0 {
            x = x[0..., .ellipsis]
            x[.ellipsis, 0 ..< dimensions] *= _mscale
        }

        return MLXFast.RoPE(
            x,
            dimensions: dimensions,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: self._freqs
        )
    }

}

public class ProportionalRoPE: Module, OffsetLayer, ArrayOffsetLayer {
    let dimensions: Int
    let rotatedDimensions: Int
    let traditional: Bool

    private let _freqs: MLXArray?

    public init(
        dimensions: Int,
        traditional: Bool = false,
        base: Float = 10000,
        scalingConfig: [String: StringOrNumber]? = nil
    ) {
        precondition(dimensions % 2 == 0, "Dimensions must be even")

        self.dimensions = dimensions
        self.traditional = traditional

        let factor = scalingConfig?["factor"]?.asFloat() ?? 1.0
        let partialRotaryFactor = scalingConfig?["partial_rotary_factor"]?.asFloat() ?? 1.0
        let ropeAngles = Int(Float(dimensions) * partialRotaryFactor / 2.0)
        self.rotatedDimensions = 2 * ropeAngles
        precondition(rotatedDimensions <= dimensions, "Rotated dimensions must not exceed head dimension")

        if rotatedDimensions > 0 {
            let exponents =
                MLXArray(stride(from: 0, to: rotatedDimensions, by: 2)).asType(.float32)
                / Float(dimensions)
            self._freqs = factor * MLX.pow(base, exponents)
        } else {
            self._freqs = nil
        }

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        guard rotatedDimensions > 0, let freqs = _freqs else {
            return x
        }
        return apply(x, offset: offset, freqs: freqs)
    }

    public func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        guard rotatedDimensions > 0, let freqs = _freqs else {
            return x
        }
        return apply(x, offset: offset, freqs: freqs)
    }

    private func splitApplyAndMerge(
        _ x: MLXArray,
        freqs: MLXArray,
        applyRoPE: (MLXArray, Int, MLXArray) -> MLXArray
    ) -> MLXArray {
        let actualDimensions = x.shape.last ?? dimensions
        let effectiveDimensions = min(dimensions, actualDimensions)
        let evenDimensions = effectiveDimensions - (effectiveDimensions % 2)
        let effectiveRotatedDimensions = min(rotatedDimensions, evenDimensions)
        let evenRotatedDimensions = effectiveRotatedDimensions - (effectiveRotatedDimensions % 2)

        guard evenDimensions > 0, evenRotatedDimensions > 0 else {
            return x
        }

        let head = x[.ellipsis, ..<evenDimensions]
        let half = evenDimensions / 2
        let rotatedHalf = evenRotatedDimensions / 2
        let effectiveFreqs = evenRotatedDimensions == rotatedDimensions ? freqs : freqs[..<rotatedHalf]

        let left = head[.ellipsis, ..<half]
        let right = head[.ellipsis, half...]
        let rotatedInput = concatenated(
            [left[.ellipsis, ..<rotatedHalf], right[.ellipsis, ..<rotatedHalf]], axis: -1)
        let rotated = applyRoPE(rotatedInput, evenRotatedDimensions, effectiveFreqs)

        let mergedLeft = concatenated(
            [rotated[.ellipsis, ..<rotatedHalf], left[.ellipsis, rotatedHalf...]], axis: -1)
        let mergedRight = concatenated(
            [rotated[.ellipsis, rotatedHalf...], right[.ellipsis, rotatedHalf...]], axis: -1)
        let mergedHead = concatenated([mergedLeft, mergedRight], axis: -1)

        guard actualDimensions > evenDimensions else {
            return mergedHead
        }
        return concatenated([mergedHead, x[.ellipsis, evenDimensions...]], axis: -1)
    }

    private func apply(_ x: MLXArray, offset: Int, freqs: MLXArray) -> MLXArray {
        splitApplyAndMerge(x, freqs: freqs) { rotatedInput, effectiveRotatedDimensions, effectiveFreqs in
            MLXFast.RoPE(
                rotatedInput,
                dimensions: effectiveRotatedDimensions,
                traditional: traditional,
                base: nil,
                scale: 1.0,
                offset: offset,
                freqs: effectiveFreqs
            )
        }
    }

    private func apply(_ x: MLXArray, offset: MLXArray, freqs: MLXArray) -> MLXArray {
        splitApplyAndMerge(x, freqs: freqs) { rotatedInput, effectiveRotatedDimensions, effectiveFreqs in
            MLXFast.RoPE(
                rotatedInput,
                dimensions: effectiveRotatedDimensions,
                traditional: traditional,
                base: nil,
                scale: 1.0,
                offset: offset,
                freqs: effectiveFreqs
            )
        }
    }
}

private let yarnTypes: Set = ["yarn", "deepseek_yarn", "telechat3-yarn"]

public typealias RoPELayer = OffsetLayer & ArrayOffsetLayer

public func initializeRope(
    dims: Int,
    base: Float,
    traditional: Bool,
    scalingConfig: [String: StringOrNumber]?,
    maxPositionEmbeddings: Int?
) -> RoPELayer {
    let ropeType: String = {
        if let config = scalingConfig,
            let typeValue = config["type"] ?? config["rope_type"],
            case .string(let s) = typeValue
        {
            return s
        }
        return "default"
    }()

    if ropeType == "default" || ropeType == "linear" {
        let scale: Float
        if ropeType == "linear", let factor = scalingConfig?["factor"]?.asFloat() {
            scale = 1 / factor
        } else {
            scale = 1.0
        }
        return RoPE(dimensions: dims, traditional: traditional, base: base, scale: scale)
    } else if ropeType == "proportional" {
        return ProportionalRoPE(
            dimensions: dims,
            traditional: traditional,
            base: base,
            scalingConfig: scalingConfig
        )
    } else if ropeType == "llama3" {
        return Llama3RoPE(
            dims: dims,
            maxPositionEmbeddings: maxPositionEmbeddings ?? 2048,
            traditional: traditional,
            base: base,
            scalingConfig: scalingConfig
        )
    } else if yarnTypes.contains(ropeType) {
        let factor = scalingConfig?["factor"]?.asFloat() ?? 32.0
        let origMax = scalingConfig?["original_max_position_embeddings"]?.asInt() ?? 4096
        let betaFast = scalingConfig?["beta_fast"]?.asFloat() ?? 32.0
        let betaSlow = scalingConfig?["beta_slow"]?.asFloat() ?? 1.0
        let mscale = scalingConfig?["mscale"]?.asFloat() ?? 1.0
        let mscaleAllDim = scalingConfig?["mscale_all_dim"]?.asFloat() ?? 0.0

        return YarnRoPE(
            dimensions: dims,
            traditional: traditional,
            maxPositionEmbeddings: maxPositionEmbeddings ?? 2048,
            base: base,
            scalingFactor: factor,
            originalMaxPositionEmbeddings: origMax,
            betaFast: betaFast,
            betaSlow: betaSlow,
            mscale: mscale,
            mscaleAllDim: mscaleAllDim
        )
    } else if ropeType == "longrope" {
        guard let config = scalingConfig else {
            fatalError("longrope requires scaling_config")
        }
        guard let origMax = config["original_max_position_embeddings"]?.asInt() else {
            fatalError("longrope requires original_max_position_embeddings")
        }
        guard let shortFactor = config["short_factor"]?.asFloats() else {
            fatalError("longrope requires short_factor")
        }
        guard let longFactor = config["long_factor"]?.asFloats() else {
            fatalError("longrope requires long_factor")
        }

        return SuScaledRoPE(
            dimensions: dims,
            base: base,
            maxPositionEmbeddings: maxPositionEmbeddings ?? 131072,
            originalMaxPositionEmbeddings: origMax,
            shortFactor: shortFactor,
            longFactor: longFactor
        )
    } else if ropeType == "mrope" {
        // MRoPE returns basic RoPE here. The actual multi-modal rotary embedding logic
        // (applying different embeddings per modality) is handled in the attention layer
        // of multimodal models like Qwen2VL, not in the RoPE module itself.
        if let config = scalingConfig,
            let mropeSection = config["mrope_section"]?.asInts()
        {
            precondition(
                mropeSection.count == 3,
                "MRoPE currently only supports 3 sections, got \(mropeSection.count)"
            )
        }
        return RoPE(dimensions: dims, traditional: traditional, base: base, scale: 1.0)
    } else {
        fatalError("Unsupported RoPE type: \(ropeType)")
    }
}
