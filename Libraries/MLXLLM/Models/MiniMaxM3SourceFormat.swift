// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// NVFP4 routed-expert projection with one global scale per expert matrix.
///
/// `weight` holds two e2m1 codes per byte (low nibble first) viewed as `uint32` `[E, out, in / 8]`, `scales` one
/// e4m3 scale per 16 inputs `[E, out, in / 16]`. The checkpoint's weight is divided by its `weight_global_scale`;
/// `gatherQuantizedMM` takes no global scale here, so each expert's output is multiplied by `1 / weight_global_scale`.
final class MiniMaxM3NVFP4SwitchLinear: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "scales") var scales: MLXArray
    @ParameterInfo(key: "inv_global_scale") var inverseGlobalScale: MLXArray

    init(weight: MLXArray, scales: MLXArray, inverseGlobalScale: MLXArray) {
        _weight.wrappedValue = weight
        _scales.wrappedValue = scales
        _inverseGlobalScale.wrappedValue = inverseGlobalScale
        super.init()
        freeze()
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool) -> MLXArray {
        let y = gatherQuantizedMM(
            x, weight, scales: scales, biases: nil, rhsIndices: indices, transpose: true,
            groupSize: 16, bits: 4, mode: .nvfp4, sortedIndices: sortedIndices)
        let scale = inverseGlobalScale[indices]
        return (y * scale.reshaped(scale.shape + [1, 1])).asType(y.dtype)
    }
}

/// Loads the official MiniMax-M3.1 checkpoint as stored.
///
/// - MXFP8 linears (attention, indexer, dense and shared-expert MLPs): `weight` is e4m3 bytes, `weight_scale_inv`
///   holds one e8m0 exponent per 32 inputs (despite the name: w = fp8 * 2^(s - 127)). The bytes are viewed as
///   `uint32` and the module becomes an mxfp8 `QuantizedLinear` (group 32, 8 bits).
/// - NVFP4 routed experts, stored per expert as `experts.E.{w1,w3,w2}.{weight_packed, weight_scale,
///   weight_global_scale, input_global_scale}`: stacked over the experts into `MiniMaxM3NVFP4SwitchLinear`.
///   `input_global_scale` (an activation scale for 4-bit-activation serving) is not used: activations stay in the
///   model dtype.
///
/// Nothing is dequantized. The converted tensors are consumed here: the generic loader would otherwise re-read
/// `weight` + `scales` pairs as affine quantization. A tensor that fails validation is left in place under its
/// checkpoint name, so the parameter update rejects the load and names it.
enum MiniMaxM3SourceFormat {
    private static let expertProjections = ["w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"]
    private static let expertFields = ["weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"]

    static func isSourceCheckpoint(_ weights: [String: MLXArray]) -> Bool {
        weights.keys.contains { $0.hasSuffix(".weight_scale_inv") || $0.hasSuffix(".weight_packed") }
    }

    /// Replaces the matching modules of `model` with quantized ones built from `weights` (already in the model's
    /// names) and returns the remaining weights.
    static func install(into model: MiniMaxM3Model, weights: [String: MLXArray]) -> [String: MLXArray] {
        var remaining = weights
        let modules = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        var replacements: [(String, Module)] = []

        for key in weights.keys.sorted() where key.hasSuffix(".weight_scale_inv") {
            let path = String(key.dropLast(".weight_scale_inv".count))
            guard let linear = modules[path] as? Linear, !(linear is QuantizedLinear),
                let bytes = weights[path + ".weight"], let exponents = weights[key]
            else {
                report("\(key): no float Linear at \(path) or no \(path).weight")
                continue
            }
            let (outputs, inputs) = linear.weight.shape2
            guard bytes.dtype == .uint8, bytes.shape == [outputs, inputs], inputs % 32 == 0,
                exponents.dtype == .uint8, exponents.shape == [outputs, inputs / 32]
            else {
                report(
                    "\(path): mxfp8 expects weight uint8 [\(outputs), \(inputs)] and weight_scale_inv uint8 "
                        + "[\(outputs), \(inputs / 32)]; found \(bytes.dtype) \(bytes.shape), "
                        + "\(exponents.dtype) \(exponents.shape)")
                continue
            }
            replacements.append(
                (
                    path,
                    QuantizedLinear(
                        weight: bytes.view(dtype: .uint32), bias: weights[path + ".bias"], scales: exponents,
                        biases: nil, groupSize: 32, bits: 8, mode: .mxfp8)
                ))
            for suffix in [".weight", ".weight_scale_inv", ".bias"] {
                remaining[path + suffix] = nil
            }
        }

        // `<prefix>.mlp.experts.<e>.<w1|w2|w3>.<field>`, grouped by (prefix, projection).
        var experts: [String: [Int: [String: MLXArray]]] = [:]
        for (key, value) in weights {
            guard let range = key.range(of: ".mlp.experts.") else { continue }
            let parts = key[range.upperBound...].split(separator: ".").map(String.init)
            guard parts.count == 3, let expert = Int(parts[0]), expertProjections[parts[1]] != nil else {
                continue
            }
            experts[String(key[..<range.lowerBound]) + "|" + parts[1], default: [:]][expert, default: [:]][
                parts[2]] = value
        }
        let config = model.configuration
        for (group, byExpert) in experts.sorted(by: { $0.key < $1.key }) {
            let separator = group.firstIndex(of: "|")!
            let prefix = String(group[..<separator])
            let projectionName = expertProjections[String(group[group.index(after: separator)...])]!
            let path = "\(prefix).mlp.switch_mlp.\(projectionName)"
            guard modules[path] is SwitchLinear else {
                report("\(path): no routed-expert projection for \(byExpert.count) source experts")
                continue
            }
            let count = config.numLocalExperts
            let (outputs, inputs) =
                projectionName == "down_proj"
                ? (config.hiddenSize, config.intermediateSize)
                : (config.intermediateSize, config.hiddenSize)
            let complete = byExpert.count == count && (0 ..< count).allSatisfy { expert in
                guard let fields = byExpert[expert], Set(fields.keys) == Set(expertFields),
                    let packed = fields["weight_packed"], let scale = fields["weight_scale"],
                    let global = fields["weight_global_scale"]
                else { return false }
                return packed.dtype == .uint8 && packed.shape == [outputs, inputs / 2]
                    && scale.dtype == .uint8 && scale.shape == [outputs, inputs / 16] && global.size == 1
                    && global.dtype.isFloatingPoint
            }
            guard inputs % 16 == 0, complete else {
                report(
                    "\(path): nvfp4 expects experts 0..<\(count), each with \(expertFields.sorted()) "
                        + "(weight_packed uint8 [\(outputs), \(inputs / 2)], weight_scale uint8 "
                        + "[\(outputs), \(inputs / 16)], one global scale); found experts "
                        + "\(byExpert.keys.sorted())")
                continue
            }
            let ordered = (0 ..< count).map { byExpert[$0]! }
            let globalScale = stacked(ordered.map { $0["weight_global_scale"]!.reshaped([]) })
                .asType(.float32)
            replacements.append(
                (
                    path,
                    MiniMaxM3NVFP4SwitchLinear(
                        weight: stacked(ordered.map { $0["weight_packed"]!.view(dtype: .uint32) }),
                        scales: stacked(ordered.map { $0["weight_scale"]! }),
                        inverseGlobalScale: 1 / globalScale)
                ))
            let source = prefix + ".mlp.experts."
            let projectionKey = group[group.index(after: separator)...]
            for expert in 0 ..< count {
                for field in expertFields {
                    remaining["\(source)\(expert).\(projectionKey).\(field)"] = nil
                }
            }
        }

        do {
            try model.update(modules: ModuleChildren.unflattened(replacements), verify: .none)
        } catch {
            // The replacements are Linear / Module subclasses built for exactly these slots.
            preconditionFailure("MiniMax-M3 source-format module install failed: \(error)")
        }
        return remaining
    }

    private static func report(_ message: String) {
        RuntimeDiagnostics.write(
            Data("[MiniMaxM3] source checkpoint tensor left unconverted: \(message)\n".utf8))
    }
}
