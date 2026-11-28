//
//  DoRA+Layers.swift
//  mlx-libraries
//
//  Created by Ivan Petrukha on 02.06.2025.
//

import Foundation
import MLX
import MLXNN

/// DoRA's per-output-row rescaling, `magnitude / norm(W + scale * Bᵀ Aᵀ, axis: 1)`.
///
/// This is the expensive part of the forward: it expands the low-rank update into a dense
/// [out, in] matrix, adds the base weight and reduces every row, while the rest of the forward is
/// one pass over the weight plus two thin matmuls.
private func doraRescaling(
    weight: MLXArray, loraA: MLXArray, loraB: MLXArray, scale: Float, magnitude: MLXArray
) -> MLXArray {
    let adapted = weight + matmul(scale * loraB.T, loraA.T)
    return magnitude / norm(adapted, axis: 1)
}

/// Performs the forward pass for a DoRA linear layer.
private func forward(
    x: MLXArray, y: MLXArray, bias: MLXArray?,
    loraA: MLXArray, loraB: MLXArray,
    scale: Float, rescaling: MLXArray
) -> MLXArray {
    let z = matmul(matmul(x, loraA), loraB)
    var out = y + (scale * z).asType(x.dtype)

    out *= rescaling.asType(x.dtype)

    return if let bias {
        out + bias
    } else {
        out
    }
}

/// Fuses the base weights with the DoRA parameters.
private func fuse(
    weight: MLXArray,
    loraA: MLXArray, loraB: MLXArray,
    scale: Float, magnitude: MLXArray
) -> MLXArray {
    let loraA = loraA.T.asType(weight.dtype)
    let loraB = (scale * loraB.T).asType(weight.dtype)

    var adapted = weight + matmul(loraB, loraA)
    let denom = norm(adapted, axis: 1)
    adapted *= (magnitude / denom).reshaped([-1, 1])

    return adapted
}

/// The rescaling of a DoRA layer, computed once while the layer is in eval mode.
///
/// Its inputs are the layer's parameters, which change only through `update(parameters:)`: an
/// adapter load, `Module.apply`, an optimizer step. The layers override it to drop the stored
/// value, and also drop it on a change of mode. So in eval mode it is computed at the first
/// forward and kept, and no later token pays for the dense expansion. In training mode it is
/// computed on every forward, as before, because the gradient of the low-rank factors and of the
/// magnitude flows through it.
///
/// The stored value is one vector of the layer's output width. It is computed from the base weight
/// as the layer holds it and evaluated at once, so the dense intermediates of one layer are freed
/// before the next layer's are built. A quantized layer's dequantized weight is one of them, and
/// is not kept.
///
/// The value lives in this class rather than in an `MLXArray?` property of the layer: `Module`
/// reflection treats a non-nil `MLXArray` property as a parameter, which would put the value into
/// `parameters()` and into saved adapters. A property of any other class type is left alone.
private final class RescalingCache {
    var value: MLXArray?

    func value(training: Bool, compute: () -> MLXArray) -> MLXArray {
        if training { return compute() }
        if let value { return value }
        // While a compiled decode step is being traced, `eval` is not allowed and the result
        // would be a value of the trace. The trace then computes it inline, as before; a value
        // stored before tracing began is captured as a constant.
        if CompiledDecodeTrace.isActive { return compute() }
        let computed = compute()
        eval(computed)
        value = computed
        return computed
    }
}

/// Filters out DoRA-specific parameters from a list of module keys.
private func filterFreezeKeys(from module: Module, keys: [String]?) -> [String] {
    return
        (keys
        ?? module.filterMap(filter: type(of: module).filterLocalParameters)
        .flattened()
        .map { $0.0 })
        .filter { !["lora_a", "lora_b", "m"].contains($0) }
}

/// Implementation of DoRA `Linear` replacement layer.
///
/// This layer implements DoRA (Weight-Decomposed Low-Rank Adaptation) for `Linear` layers.
///
/// ``QDoRALinear`` is the equivalent class for `QuantizedLinear`.
public class DoRALinear: Linear, LoRALayer {

    let scale: Float

    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray
    @ParameterInfo(key: "m") var magnitude: MLXArray

    required public init(linear: Linear, rank: Int = 8, scale: Float = 20.0) {
        let (outputDimensions, inputDimensions) = linear.shape
        let loraScale = 1 / sqrt(Float(inputDimensions))

        self.scale = scale
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputDimensions])
        self._magnitude.wrappedValue = MLXLinalg.norm(linear.weight, axis: 1)

        super.init(weight: linear.weight, bias: linear.bias)

        freeze()
    }

    public static func from(linear: Linear, rank: Int = 8, scale: Float = 20.0) -> LoRALayer {
        if let linear = linear as? QuantizedLinear {
            QDoRALinear(linear: linear, rank: rank, scale: scale)
        } else {
            DoRALinear(linear: linear, rank: rank, scale: scale)
        }
    }

    public override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false)
        throws
    {
        let keys = filterFreezeKeys(from: self, keys: keys)
        try super.freeze(recursive: recursive, keys: keys, strict: strict)
    }

    private let rescalingCache = RescalingCache()

    public override func didSetTrain(_ mode: Bool) {
        super.didSetTrain(mode)
        rescalingCache.value = nil
    }

    @discardableResult
    public override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        rescalingCache.value = nil
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    public func fused() -> Module {
        Linear(
            weight: fuse(
                weight: weight, loraA: loraA, loraB: loraB, scale: scale, magnitude: magnitude),
            bias: bias
        )
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = matmul(x, weight.T)
        let rescaling = rescalingCache.value(training: training) {
            doraRescaling(
                weight: weight, loraA: loraA, loraB: loraB, scale: scale, magnitude: magnitude)
        }
        return forward(
            x: x, y: y, bias: bias,
            loraA: loraA, loraB: loraB,
            scale: scale, rescaling: rescaling
        )
    }
}

/// Implementation of DoRA `QuantizedLinear` replacement layer.
///
/// See ``DoRALinear`` (equivalent class for `Linear` layers) for more information.
///
/// ### See Also
/// - ``DoRALinear``
public class QDoRALinear: QuantizedLinear, LoRALayer {

    let scale: Float

    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray
    @ParameterInfo(key: "m") var magnitude: MLXArray

    required public init(linear: QuantizedLinear, rank: Int = 8, scale: Float = 20.0) {
        let (outputDimensions, inputDimensions) = linear.shape
        let loraScale = 1 / sqrt(Float(inputDimensions))

        self.scale = scale
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputDimensions])
        self._magnitude.wrappedValue = MLXLinalg.norm(linear.dequantizedWeight, axis: 1)

        super.init(
            weight: linear.weight, bias: linear.bias,
            scales: linear.scales, biases: linear.biases,
            groupSize: linear.groupSize, bits: linear.bits,
            mode: linear.mode
        )

        freeze()
    }

    public override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false)
        throws
    {
        let keys = filterFreezeKeys(from: self, keys: keys)
        try super.freeze(recursive: recursive, keys: keys, strict: strict)
    }

    private let rescalingCache = RescalingCache()

    public override func didSetTrain(_ mode: Bool) {
        super.didSetTrain(mode)
        rescalingCache.value = nil
    }

    @discardableResult
    public override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        rescalingCache.value = nil
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    public func fused() -> Module {
        QuantizedLinear(
            weight: fuse(
                weight: dequantizedWeight, loraA: loraA, loraB: loraB, scale: scale,
                magnitude: magnitude),
            bias: bias, groupSize: groupSize, bits: bits
        )
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = quantizedMM(
            x, weight, scales: scales, biases: biases, groupSize: groupSize, bits: bits,
            mode: mode)
        let rescaling = rescalingCache.value(training: training) {
            doraRescaling(
                weight: dequantizedWeight, loraA: loraA, loraB: loraB, scale: scale,
                magnitude: magnitude)
        }
        return forward(
            x: x, y: y, bias: bias,
            loraA: loraA, loraB: loraB,
            scale: scale, rescaling: rescaling
        )
    }
}
