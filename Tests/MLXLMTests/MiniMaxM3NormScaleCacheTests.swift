// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

/// The M3 RMSNorm caches `1 + weight` once instead of building it on every call: outputs must equal the per-call form
/// bit for bit, and a module update (which replaces the weight's contents in place) must refresh the cache.
@Suite("MiniMax-M3 RMSNorm scale cache", .serialized)
struct MiniMaxM3NormScaleCacheTests {
    @Test("Cached scale == per-call 1 + weight, bit for bit; an update refreshes it")
    func cachedScaleMatchesAndRefreshes() throws {
        try Device.withDefaultDevice(.gpu) {
            let norm = MiniMaxM3RMSNorm(dimensions: 256, eps: 1e-6)
            let w0 = MLXRandom.uniform(low: -0.5, high: 0.5, [256], key: MLXRandom.key(1)).asType(.bfloat16)
            try norm.update(parameters: ModuleParameters.unflattened(["weight": w0]), verify: [.all])
            let x = MLXRandom.normal([3, 7, 256], key: MLXRandom.key(2)).asType(.bfloat16)
            let reference0 = MLXFast.rmsNorm(x, weight: 1 + w0, eps: 1e-6)
            let first = norm(x)
            let second = norm(x)  // served from the cache
            #expect(arrayEqual(first, reference0).item(Bool.self) && arrayEqual(second, reference0).item(Bool.self))
            // Replace the weight through a module update (in place, same object): the next call must use it.
            let w1 = MLXRandom.uniform(low: -0.5, high: 0.5, [256], key: MLXRandom.key(3)).asType(.bfloat16)
            try norm.update(parameters: ModuleParameters.unflattened(["weight": w1]), verify: [.all])
            let reference1 = MLXFast.rmsNorm(x, weight: 1 + w1, eps: 1e-6)
            #expect(arrayEqual(norm(x), reference1).item(Bool.self))
            // Control: the two references differ, so a stale cache would have been caught.
            #expect(!arrayEqual(reference0, reference1).item(Bool.self))
            // A dtype change through update (the f32 conversion tests use) refreshes too.
            try norm.update(parameters: ModuleParameters.unflattened(["weight": w1.asType(.float32)]), verify: [.all])
            let reference2 = MLXFast.rmsNorm(x.asType(.float32), weight: 1 + w1.asType(.float32), eps: 1e-6)
            #expect(arrayEqual(norm(x.asType(.float32)), reference2).item(Bool.self))
        }
    }
}
