// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 attention over the contiguous 4-bit pools against the segmented 4-bit
// layout (same encoded values, host-side gather): a kv-source + index-source layer run
// through a prefill and several decode steps with more visible rows than the top-k, so
// prefill takes the dense-mask path and decode the selected-row gather. Tiny geometry
// on the CPU device (the Metal-only heads-16 kernel declines these shapes).

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("DeepSeek-V4.1 attention on contiguous pools (CPU)")
struct DeepseekV41PoolLayoutAttentionTests {

    static func config() -> DeepseekV4Configuration {
        var c = DeepseekV4Configuration()
        c.hiddenSize = 32
        c.numAttentionHeads = 4
        c.headDim = 32
        c.qkRopeHeadDim = 8
        c.qLoraRank = 16
        c.oGroups = 2
        c.oLoraRank = 8
        c.slidingWindow = 8
        c.numHiddenLayers = 3
        c.compressRatios = [0, 0, 1]
        c.kvSourceLayerIds = [2]
        c.indexSourceLayerIds = [2]
        c.candidateSourceLayerId = 2
        c.candidateTopkBlocks = 3
        c.candidateBlockSize = 4
        c.indexNHeads = 2
        c.indexHeadDim = 16
        c.indexTopk = 6
        return c
    }

    static func values(_ n: Int, seed: UInt64, scale: Float = 0.25) -> [Float] {
        var rng = SplitMix64(seed: seed)
        return (0 ..< n).map { _ in Float(Int(rng.next() % 65) - 32) / 16 * scale }
    }

    @Test("prefill and decode match the segmented 4-bit layout")
    func contiguousMatchesSegmented() throws {
        try Device.withDefaultDevice(.cpu) {
            let cfg = Self.config()
            #expect(cfg.isDeepseekV41)
            let attn = DeepseekV4Attention(config: cfg, layerIdx: 2)
            var params: [String: MLXArray] = [:]
            for (i, (key, value)) in attn.parameters().flattened().enumerated() {
                params[key] = MLXArray(Self.values(value.size, seed: UInt64(1 + i)), value.shape)
            }
            try attn.update(parameters: ModuleParameters.unflattened(params), verify: .noUnusedKeys)

            func makeCache(contiguous: Bool) -> DeepseekV4Cache {
                DeepseekV4Cache(
                    slidingWindow: cfg.slidingWindow, compressRatio: 1,
                    poolQuantizationEnabled: true, poolBF16HotByteLimit: 0,
                    poolQuantizationBits: 4, contiguousPools: contiguous,
                    compactLocalWindow: contiguous)
            }
            let contiguous = makeCache(contiguous: true)
            let segmented = makeCache(contiguous: false)

            let prompt = 40
            let x = MLXArray(Self.values(prompt * cfg.hiddenSize, seed: 900, scale: 1),
                [1, prompt, cfg.hiddenSize])
            let outA = attn(x, mask: .none, cache: contiguous, compressShare: DSV41CompressShare())
            let outB = attn(x, mask: .none, cache: segmented, compressShare: DSV41CompressShare())
            MLX.eval(outA, outB)
            #expect(contiguous.getPooled(.compressor)!.dim(1) == prompt)
            #expect(
                MLX.abs(contiguous.getPooled(.compressor)! - segmented.getPooled(.compressor)!)
                    .max().item(Float.self) == 0, "both layouts keep the same 4-bit rows")
            let prefillDiff = MLX.abs(outA - outB).max().item(Float.self)
            #expect(prefillDiff < 1e-3, "prefill \(prefillDiff)")

            for step in 0 ..< 6 {
                let t = MLXArray(Self.values(cfg.hiddenSize, seed: 1000 + UInt64(step), scale: 1),
                    [1, 1, cfg.hiddenSize])
                let a = attn(t, mask: .none, cache: contiguous, compressShare: DSV41CompressShare())
                let b = attn(t, mask: .none, cache: segmented, compressShare: DSV41CompressShare())
                MLX.eval(a, b)
                let diff = MLX.abs(a - b).max().item(Float.self)
                #expect(diff == 0, "decode step \(step): \(diff)")
            }
            #expect(contiguous.offset == prompt + 6)
            #expect(contiguous.local.state[0].dim(2) == cfg.slidingWindow)
        }
    }
}
