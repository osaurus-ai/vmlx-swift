// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct RotatingSharedKVCacheLifetimeTests {
    @Test("Shared latent windows match paired updates through growth, rotation and restore")
    func sharedWindowParity() {
        Device.withDefaultDevice(.cpu) {
            let paired = RotatingKVCache(maxSize: 8, keep: 2, step: 4)
            var shared = RotatingKVCache(maxSize: 8, keep: 2, step: 4)
            for (index, count) in [3, 1, 5, 1, 1, 1, 1, 4, 1, 1, 1].enumerated() {
                let row = MLXArray(
                    (0 ..< count * 2).map { Float($0 + index * 20) }, [1, 1, count, 2])
                let expected = paired.update(keys: row, values: row)
                eval(expected.0, expected.1)
                let actual = shared.updateSharedKV(row)
                eval(actual)
                #expect(actual.asArray(Float.self) == expected.0.asArray(Float.self))
                #expect(shared.metaState == paired.metaState)
                for (a, b) in zip(shared.state, paired.state) {
                    #expect(a.asArray(Float.self) == b.asArray(Float.self))
                }
                if index == 4 {
                    let restored = RotatingKVCache(maxSize: 8, keep: 2, step: 4)
                    restored.state = shared.state.map { $0.reshaped($0.shape) }
                    restored.metaState = shared.metaState
                    shared = restored
                }
            }
        }
    }

    @Test("DeepSeek companion caches preserve local shared-KV state during prefill and decode")
    func companionWindowParity() {
        Device.withDefaultDevice(.cpu) {
            let paired = DeepseekV4Cache(
                slidingWindow: 8, compressRatio: 4, compactLocalWindow: true)
            let shared = DeepseekV4Cache(
                slidingWindow: 8, compressRatio: 4, compactLocalWindow: true)
            for (index, count) in [6, 1, 1, 1, 4, 1, 1].enumerated() {
                let row = MLXArray(
                    (0 ..< count * 2).map { Float($0 + index * 20) }, [1, 1, count, 2])
                let expected = paired.update(keys: row, values: row)
                eval(expected.0, expected.1)
                let actual = shared.updateSharedKV(row)
                eval(actual)
                #expect(actual.asArray(Float.self) == expected.0.asArray(Float.self))
                #expect(shared.metaState == paired.metaState)
                for (a, b) in zip(shared.local.state, paired.local.state) {
                    #expect(a.asArray(Float.self) == b.asArray(Float.self))
                }
            }
        }
    }

    #if canImport(Metal)
        @Test("Long shared-KV decode releases unused-value history across layer submissions")
        func longSharedKVDecode() throws {
            try Device.withDefaultDevice(.gpu) {
                let caches = (0 ..< 40).map { _ in RotatingKVCache(maxSize: 128, step: 128) }
                try withError { error in
                    for position in 0 ..< 4096 {
                        var last: MLXArray?
                        for (layer, cache) in caches.enumerated() {
                            let row = MLXArray([Float(position + layer)], [1, 1, 1, 1]) + 1
                            let k = cache.updateSharedKV(row)
                            asyncEval(k)
                            try error.check()
                            last = k
                        }
                        let value = sum(last!).item(Float.self)
                        try error.check()
                        if position % 256 == 0 {
                            #expect(value.isFinite)
                            print("shared-kv completed=\(position + 1)")
                        }
                    }
                    #expect(caches.allSatisfy { $0.offset == 4096 })
                }
            }
        }
    #endif
}
