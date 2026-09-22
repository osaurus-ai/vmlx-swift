// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct MiMoV26VisionConfiguration: Decodable, Sendable {
    let depth: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let outputSize: Int
    let patchSize: Int
    let temporalPatchSize: Int
    let mergeSize: Int
    let channels: Int
    let epsilon: Float
    let fullAttentionLayers: [Int]
    let windowTypes: [Int]
    let window: Int
    let useSink: Bool

    enum CodingKeys: String, CodingKey {
        case depth, hiddenSize = "hidden_size", intermediateSize = "intermediate_size"
        case heads = "num_heads", kvHeads = "num_key_value_heads", headDim = "qk_channels"
        case outputSize = "out_hidden_size", patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size", mergeSize = "spatial_merge_size"
        case channels = "in_channels", inChans = "in_chans", epsilon = "rms_norm_eps"
        case fullAttentionLayers = "fullatt_block_indexes", windowTypes = "vit_window_attn_types"
        case window = "visual_token_window_size", useSink = "use_sink"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        depth = try c.decodeIfPresent(Int.self, forKey: .depth) ?? 28
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        heads = try c.decode(Int.self, forKey: .heads)
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? heads
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        outputSize = try c.decode(Int.self, forKey: .outputSize)
        patchSize = try c.decode(Int.self, forKey: .patchSize)
        temporalPatchSize = try c.decode(Int.self, forKey: .temporalPatchSize)
        mergeSize = try c.decodeIfPresent(Int.self, forKey: .mergeSize) ?? 2
        channels = try c.decodeIfPresent(Int.self, forKey: .channels)
            ?? c.decodeIfPresent(Int.self, forKey: .inChans) ?? 3
        epsilon = try c.decodeIfPresent(Float.self, forKey: .epsilon) ?? 1e-6
        fullAttentionLayers = try c.decodeIfPresent([Int].self, forKey: .fullAttentionLayers) ?? []
        windowTypes = try c.decodeIfPresent([Int].self, forKey: .windowTypes) ?? Array(repeating: -1, count: depth)
        window = try c.decodeIfPresent(Int.self, forKey: .window) ?? -1
        useSink = try c.decodeIfPresent(Bool.self, forKey: .useSink) ?? false
        guard depth > 0, windowTypes.count == depth,
            windowTypes.allSatisfy({ [-1, 0, 1].contains($0) }),
            heads > 0, kvHeads > 0, heads.isMultiple(of: kvHeads),
            headDim > 0, headDim.isMultiple(of: 4), patchSize > 0,
            temporalPatchSize > 0, mergeSize > 0, channels > 0
        else { throw VLMError.processing("Invalid MiMo vision configuration") }
    }
}

/// Ordered merge-unit positions and per-frame attention boundaries.
struct MiMoV26VisionLayout {
    let columnIndices: MLXArray
    let inverseColumnIndices: MLXArray
    let cosine: MLXArray
    let sine: MLXArray
    let sequenceLengths: [Int]
    let patchCount: Int

    init(grid: [THW], headDim: Int, merge: Int) throws {
        guard !grid.isEmpty, grid.allSatisfy({
            $0.t > 0 && $0.h > 0 && $0.w > 0 && $0.h.isMultiple(of: merge) && $0.w.isMultiple(of: merge)
        }) else { throw VLMError.processing("Invalid MiMo vision grid") }
        var columns: [Int32] = [], positions: [(Int, Int)] = [], lengths: [Int] = []
        var base = 0
        for item in grid {
            let rows = item.h / merge, cols = item.w / merge
            for frame in 0..<item.t {
                lengths.append(item.h * item.w)
                for col in 0..<cols {
                    for row in 0..<rows {
                        columns.append(Int32(base + frame * rows * cols + row * cols + col))
                    }
                }
                for row in 0..<rows {
                    for col in 0..<cols {
                        for dy in 0..<merge {
                            for dx in 0..<merge { positions.append((row * merge + dy, col * merge + dx)) }
                        }
                    }
                }
            }
            base += item.t * rows * cols
        }
        var inverse = Array(repeating: Int32(0), count: columns.count)
        for (i, column) in columns.enumerated() { inverse[Int(column)] = Int32(i) }
        let frequencies = stride(from: 0, to: headDim / 2, by: 2).map {
            1 / pow(Float(10_000), Float($0) / Float(headDim / 2))
        }
        var phase: [Float] = []
        phase.reserveCapacity(positions.count * headDim)
        for (h, w) in positions {
            let axes = frequencies.map { Float(h) * $0 } + frequencies.map { Float(w) * $0 }
            phase.append(contentsOf: axes)
            phase.append(contentsOf: axes)
        }
        let angles = MLXArray(phase).reshaped(positions.count, headDim)
        cosine = cos(angles)
        sine = sin(angles)
        columnIndices = MLXArray(columns)
        inverseColumnIndices = MLXArray(inverse)
        sequenceLengths = lengths
        patchCount = positions.count
    }
}

final class MiMoV26VisionAttention: Module {
    let qkv: Linear
    let proj: Linear
    @ParameterInfo var sinks: MLXArray?
    let config: MiMoV26VisionConfiguration

    init(_ config: MiMoV26VisionConfiguration, useSink: Bool) {
        self.config = config
        qkv = Linear(config.hiddenSize, (config.heads + 2 * config.kvHeads) * config.headDim)
        proj = Linear(config.heads * config.headDim, config.hiddenSize)
        _sinks.wrappedValue = useSink ? MLXArray.zeros([config.heads]) : nil
    }

    private func rotate(_ x: MLXArray, cosine: MLXArray, sine: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        let half = x.dim(-1) / 2
        let rotated = concatenated([-f[.ellipsis, half...], f[.ellipsis, ..<half]], axis: -1)
        return (f * cosine[0..., .newAxis, 0...] + rotated * sine[0..., .newAxis, 0...]).asType(x.dtype)
    }

    private func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, full: Bool) -> MLXArray {
        let length = q.dim(2)
        let scale = pow(Float(config.headDim), -0.5)
        if full || (config.window <= 0 && sinks == nil) {
            return MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        }
        let windowed = config.window > 0 && length > config.window + 1
        let window = windowed ? config.window : length
        let step = windowed ? 512 : length
        var outputs: [MLXArray] = []
        for first in stride(from: 0, to: length, by: step) {
            let end = min(first + step, length)
            let keyFirst = max(0, first - window), keyEnd = min(length, end + window)
            let rows = MLXArray(Array(first..<end))[0..., .newAxis]
            let cols = MLXArray(Array(keyFirst..<keyEnd))[.newAxis, 0...]
            var bias = MLX.where(abs(rows - cols) .> window, -Float.infinity, Float(0))
                .reshaped(1, 1, end - first, keyEnd - keyFirst)
            // Vision sinks bias the existing key zero; text sinks are an extra null logit.
            if let sinks, keyFirst == 0 {
                let keyZero = (MLXArray(Array(keyFirst..<keyEnd)) .== 0).asType(.float32)
                    .reshaped(1, 1, 1, keyEnd - keyFirst)
                bias = bias + keyZero * sinks.asType(.float32).reshaped(1, -1, 1, 1)
            }
            outputs.append(MLXFast.scaledDotProductAttention(
                queries: q[0..., 0..., first..<end, 0...],
                keys: k[0..., 0..., keyFirst..<keyEnd, 0...],
                values: v[0..., 0..., keyFirst..<keyEnd, 0...], scale: scale,
                mask: .array(bias.asType(q.dtype))))
        }
        return concatenated(outputs, axis: 2)
    }

    func callAsFunction(_ x: MLXArray, layout: MiMoV26VisionLayout, cosine: MLXArray, sine: MLXArray, full: Bool) -> MLXArray {
        let qSize = config.heads * config.headDim, kvSize = config.kvHeads * config.headDim
        let parts = split(qkv(x), indices: [qSize, qSize + kvSize], axis: -1)
        let q = rotate(parts[0].reshaped(-1, config.heads, config.headDim), cosine: cosine, sine: sine)
        let k = rotate(parts[1].reshaped(-1, config.kvHeads, config.headDim), cosine: cosine, sine: sine)
        let v = parts[2].reshaped(-1, config.kvHeads, config.headDim)
        var offset = 0, outputs: [MLXArray] = []
        for count in layout.sequenceLengths {
            let range = offset..<(offset + count)
            let value = attend(
                q[range].transposed(1, 0, 2)[.newAxis],
                k[range].transposed(1, 0, 2)[.newAxis],
                v[range].transposed(1, 0, 2)[.newAxis], full: full)
            outputs.append(value[0].transposed(1, 0, 2).reshaped(count, -1))
            offset += count
        }
        return proj(concatenated(outputs, axis: 0))
    }
}

final class MiMoV26VisionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ config: MiMoV26VisionConfiguration) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class MiMoV26VisionBlock: Module {
    let norm1: RMSNorm
    let norm2: RMSNorm
    let attn: MiMoV26VisionAttention
    let mlp: MiMoV26VisionMLP
    init(_ config: MiMoV26VisionConfiguration, useSink: Bool) {
        norm1 = RMSNorm(dimensions: config.hiddenSize, eps: config.epsilon)
        norm2 = RMSNorm(dimensions: config.hiddenSize, eps: config.epsilon)
        attn = MiMoV26VisionAttention(config, useSink: useSink)
        mlp = MiMoV26VisionMLP(config)
    }
    func callAsFunction(_ x: MLXArray, layout: MiMoV26VisionLayout, cosine: MLXArray, sine: MLXArray, full: Bool) -> MLXArray {
        let h = x + attn(norm1(x), layout: layout, cosine: cosine, sine: sine, full: full)
        return h + mlp(norm2(h))
    }
}

final class MiMoV26PatchEmbed: Module {
    let proj: Linear
    init(_ config: MiMoV26VisionConfiguration) {
        proj = Linear(config.channels * config.temporalPatchSize * config.patchSize * config.patchSize,
                      config.hiddenSize, bias: false)
    }
}

final class MiMoV26PatchMerger: Module {
    @ModuleInfo(key: "ln_q") var norm: RMSNorm
    @ModuleInfo var mlp: (Linear, GELU, Linear)
    let width: Int
    init(_ config: MiMoV26VisionConfiguration) {
        width = config.hiddenSize * config.mergeSize * config.mergeSize
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
        mlp = (Linear(width, width, bias: false), GELU(), Linear(width, config.outputSize, bias: false))
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        mlp.2(mlp.1(mlp.0(norm(x).reshaped(-1, width))))
    }
}

final class MiMoV26VisionTower: Module {
    @ModuleInfo(key: "patch_embed") var patch: MiMoV26PatchEmbed
    let blocks: [MiMoV26VisionBlock]
    let merger: MiMoV26PatchMerger
    let configuration: MiMoV26VisionConfiguration
    init(_ config: MiMoV26VisionConfiguration) {
        configuration = config
        _patch.wrappedValue = MiMoV26PatchEmbed(config)
        blocks = (0..<config.depth).map {
            MiMoV26VisionBlock(config, useSink: config.useSink && !config.fullAttentionLayers.contains($0))
        }
        merger = MiMoV26PatchMerger(config)
    }

    func callAsFunction(_ pixels: MLXArray, grid: [THW]) throws -> MLXArray {
        let config = configuration
        let layout = try MiMoV26VisionLayout(grid: grid, headDim: config.headDim, merge: config.mergeSize)
        guard pixels.ndim == 2, pixels.dim(0) == layout.patchCount else {
            throw VLMError.processing("MiMo vision patch count does not match its grid")
        }
        let unit = config.mergeSize * config.mergeSize
        func reordered(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
            x.reshaped(-1, unit, x.dim(-1)).take(indices, axis: 0).reshaped(-1, x.dim(-1))
        }
        let columnCos = reordered(layout.cosine, layout.columnIndices)
        let columnSin = reordered(layout.sine, layout.columnIndices)
        var x = patch.proj(pixels.asType(patch.proj.weight.dtype))
        var columnOrder = false
        for (index, block) in blocks.enumerated() {
            let needsColumn = config.windowTypes[index] == 1
            if needsColumn != columnOrder {
                x = reordered(x, needsColumn ? layout.columnIndices : layout.inverseColumnIndices)
                columnOrder = needsColumn
            }
            x = block(x, layout: layout,
                      cosine: needsColumn ? columnCos : layout.cosine,
                      sine: needsColumn ? columnSin : layout.sine,
                      full: config.fullAttentionLayers.contains(index))
        }
        if columnOrder { x = reordered(x, layout.inverseColumnIndices) }
        return merger(x)
    }
}
