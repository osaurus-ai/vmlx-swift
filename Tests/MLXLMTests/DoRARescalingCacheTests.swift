// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

/// `DoRALinear` and `QDoRALinear` compute DoRA's rescaling, `m / norm(W + scale * Bᵀ Aᵀ)`, once
/// in eval mode, recompute it after any parameter update or change of mode, and never store it
/// in training mode.
@Suite("DoRA rescaling computed once in eval mode", .serialized)
struct DoRARescalingCacheTests {

    typealias DoRALayer = Module & LoRALayer & UnaryLayer

    static let inputs = 128
    static let outputs = 64

    /// A DoRA layer whose rescaling is not 1: a fresh layer has `lora_b == 0` and `m == norm(W)`,
    /// which makes every rescaling exactly 1.
    static func makeLayer(quantized: Bool) -> DoRALayer {
        let linear = Linear(
            weight: MLXRandom.normal([outputs, inputs], key: MLXRandom.key(1)) * 0.05)
        let layer: DoRALayer =
            quantized
            ? QDoRALinear(
                linear: QuantizedLinear(weight: linear.weight, bias: nil, groupSize: 64, bits: 4),
                rank: 8, scale: 2)
            : DoRALinear(linear: linear, rank: 8, scale: 2)
        layer.update(
            parameters: ModuleParameters.unflattened([
                "lora_b": MLXRandom.normal([8, outputs], key: MLXRandom.key(2)) * 0.05,
                "m": MLXRandom.uniform(low: 0.5, high: 2, [outputs], key: MLXRandom.key(3)),
            ]))
        return layer
    }

    /// Computed, not stored: a stored `MLXArray` is not `Sendable`, and this keeps every MLX call
    /// inside the test's Metal lock. The fixed key gives the same input on every call.
    static var x: MLXArray { MLXRandom.normal([2, 3, inputs], key: MLXRandom.key(4)) }

    static func forward(_ layer: DoRALayer) -> MLXArray {
        let out = layer(x)
        eval(out)
        return out
    }

    /// The magnitude written past `update(parameters:)`, which nothing in the library does: what
    /// the next forward makes of it shows whether the layer recomputed its rescaling.
    static func overwriteMagnitudeInPlace(_ layer: DoRALayer, by factor: Float) {
        let m = magnitude(layer)
        let scaled = m * factor
        eval(scaled)
        m._updateInternal(scaled)
    }

    static func magnitude(_ layer: DoRALayer) -> MLXArray {
        layer.parameters().flattened().first { $0.0 == "m" }!.1
    }

    static func same(_ a: MLXArray, _ b: MLXArray) -> Bool {
        allClose(a, b, rtol: 1e-5, atol: 1e-6).item(Bool.self)
    }

    @Test(
        "eval mode stores the rescaling and gives the training-mode result",
        arguments: [false, true])
    func evalModeStoresTheRescaling(quantized: Bool) throws {
        try MLXMetalTestLock.withLock {
            let layer = Self.makeLayer(quantized: quantized)
            layer.train(true)
            let reference = Self.forward(layer)

            layer.train(false)
            let first = Self.forward(layer)
            #expect(Self.same(first, reference))

            // Stored: a forward after a write the layer cannot see still uses the old rescaling.
            Self.overwriteMagnitudeInPlace(layer, by: 2)
            #expect(Self.same(Self.forward(layer), reference), "the rescaling was recomputed")
        }
    }

    @Test("training mode recomputes it on every forward", arguments: [false, true])
    func trainingModeRecomputes(quantized: Bool) throws {
        try MLXMetalTestLock.withLock {
            let layer = Self.makeLayer(quantized: quantized)
            layer.train(true)
            let before = Self.forward(layer)
            Self.overwriteMagnitudeInPlace(layer, by: 2)
            #expect(Self.same(Self.forward(layer), before * 2))
        }
    }

    @Test(
        "update(parameters:) drops it, also through a parent module", arguments: [false, true])
    func updateDropsIt(quantized: Bool) throws {
        final class Parent: Module {
            @ModuleInfo var proj: Module
            init(_ proj: Module) { self._proj.wrappedValue = proj }
        }
        try MLXMetalTestLock.withLock {
            let layer = Self.makeLayer(quantized: quantized)
            let parent = Parent(layer)
            parent.train(false)
            let before = Self.forward(layer)

            let m = Self.magnitude(layer) * 2
            eval(m)
            try parent.update(
                parameters: ModuleParameters.unflattened(["proj.m": m]), verify: .noUnusedKeys)
            #expect(Self.same(Self.forward(layer), before * 2), "a stale rescaling was used")
        }
    }

    @Test("a change of mode drops it", arguments: [false, true])
    func modeChangeDropsIt(quantized: Bool) throws {
        try MLXMetalTestLock.withLock {
            let layer = Self.makeLayer(quantized: quantized)
            layer.train(false)
            let before = Self.forward(layer)
            Self.overwriteMagnitudeInPlace(layer, by: 2)
            layer.train(true)
            layer.train(false)
            #expect(Self.same(Self.forward(layer), before * 2), "a stale rescaling was used")
        }
    }

    @Test("the stored rescaling is not a parameter", arguments: [false, true])
    func notAParameter(quantized: Bool) throws {
        try MLXMetalTestLock.withLock {
            let layer = Self.makeLayer(quantized: quantized)
            let expected = Set(layer.parameters().flattened().map(\.0))
            #expect(expected.isSuperset(of: ["weight", "lora_a", "lora_b", "m"]))
            layer.train(false)
            _ = Self.forward(layer)
            #expect(Set(layer.parameters().flattened().map(\.0)) == expected)
        }
    }

    @Test("a compiled-decode trace computes it inline and stores nothing", arguments: [false, true])
    func compiledTraceStoresNothing(quantized: Bool) throws {
        try MLXMetalTestLock.withLock {
            let layer = Self.makeLayer(quantized: quantized)
            layer.train(false)
            let traced = CompiledDecodeTrace.withActive { Self.forward(layer) }
            Self.overwriteMagnitudeInPlace(layer, by: 2)
            // Nothing was stored during the trace, so this forward computes the new rescaling.
            #expect(Self.same(Self.forward(layer), traced * 2))
        }
    }
}
