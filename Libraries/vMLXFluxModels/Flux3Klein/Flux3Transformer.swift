import Foundation
@preconcurrency import MLX
import MLXNN
import vMLXFluxKit

/// Native mflux d587aac Flux3Transformer. Weights already use the mflux tree;
/// there is no diffusers name conversion at runtime.
enum Flux3Rotary {
    static func apply(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let shape = x.shape
        let pairs = x.reshaped(Array(shape.dropLast()) + [shape.last! / 2, 2])
        let a = pairs[.ellipsis, 0], b = pairs[.ellipsis, 1]
        return stacked([cos * a - sin * b, sin * a + cos * b], axis: -1).reshaped(shape)
    }
}

private struct Flux3Modulation {
    let shift, scale, gate: MLXArray
    init(_ projected: MLXArray, hidden: Int) {
        shift = projected[0..., .newAxis, ..<hidden]
        scale = projected[0..., .newAxis, hidden ..< 2 * hidden]
        gate = projected[0..., .newAxis, (2 * hidden)...]
    }
    init(shift: MLXArray, scale: MLXArray, gate: MLXArray) {
        self.shift = shift; self.scale = scale; self.gate = gate
    }
}

private final class Flux3TimeEmbedding {
    let first, second: MFluxLinear
    let channels: Int
    init(store: MFluxStore, prefix: String, cfg: Flux3TransformerConfig) throws {
        channels = cfg.timestepChannels
        first = try store.linear("transformer", "\(prefix).linear_1", inputDimensions: channels, outputDimensions: cfg.hidden)
        second = try store.linear("transformer", "\(prefix).linear_2", inputDimensions: cfg.hidden, outputDimensions: cfg.hidden)
    }
    func callAsFunction(_ time: Float) -> MLXArray {
        // Round t before multiplication and round t*1000 to bf16, as mflux does.
        let t = MLXArray([time]).asType(.bfloat16) * 1000
        let half = channels / 2
        let frequencies = exp(-log(Float(10000)) * MLXArray((0..<half).map(Float.init)) / Float(half))
        let args = t.asType(.float32).reshaped([1, 1]) * frequencies.reshaped([1, half])
        let embedding = concatenated([MLX.cos(args), MLX.sin(args)], axis: -1).asType(.bfloat16)
        return second(silu(first(embedding)))
    }
}

private final class Flux3Block {
    let q, k, v, out, ffIn, ffOut: MFluxLinear
    let qWeight, kWeight: MLXArray
    let heads, dim, hidden: Int
    init(store: MFluxStore, prefix: String, cfg: Flux3TransformerConfig) throws {
        heads = cfg.numAttentionHeads; dim = cfg.attentionHeadDim; hidden = cfg.hidden
        func linear(_ name: String, _ input: Int, _ output: Int) throws -> MFluxLinear {
            try store.linear("transformer", "\(prefix).\(name)", inputDimensions: input, outputDimensions: output)
        }
        q = try linear("attn.to_q", hidden, hidden); k = try linear("attn.to_k", hidden, hidden)
        v = try linear("attn.to_v", hidden, hidden); out = try linear("attn.to_out", hidden, hidden)
        qWeight = try store.tensor("transformer", "\(prefix).attn.norm_q.weight")
        kWeight = try store.tensor("transformer", "\(prefix).attn.norm_k.weight")
        let intermediate = Int(Float(hidden) * cfg.mlpRatio)
        ffIn = try linear("ff.linear_in", hidden, intermediate * 2)
        ffOut = try linear("ff.linear_out", intermediate, hidden)
    }
    func callAsFunction(_ x: MLXArray, rotary: (MLXArray, MLXArray), mod: Flux3Modulation) -> MLXArray {
        let normed = MLXFast.layerNorm(x, weight: nil, bias: nil, eps: 1e-6)
        let h = (1 + mod.scale) * normed + mod.shift
        let b = h.dim(0), s = h.dim(1)
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([b, s, heads, dim]).transposed(0, 2, 1, 3) }
        func normRope(_ t: MLXArray, _ weight: MLXArray) -> MLXArray {
            let n = MLXFast.rmsNorm(t.asType(.float32), weight: weight.asType(.float32), eps: 1e-6)
            return Flux3Rotary.apply(n, cos: rotary.0, sin: rotary.1).asType(t.dtype)
        }
        let qs = normRope(split(q(h)), qWeight), ks = normRope(split(k(h)), kWeight)
        let vs = split(v(h))
        let attention = MLX.scaledDotProductAttention(queries: qs, keys: ks, values: vs,
            scale: 1 / Float(dim).squareRoot(), mask: nil)
        let attentionOut = out(attention.transposed(0, 2, 1, 3).reshaped([b, s, hidden]))
        let ff = ffIn(h), half = ff.dim(-1) / 2
        let feedForward = ffOut(silu(ff[.ellipsis, ..<half]) * ff[.ellipsis, half...])
        return x + mod.gate * (attentionOut + feedForward)
    }
}

final class Flux3Transformer {
    let cfg: Flux3TransformerConfig
    private let time, targetTime: Flux3TimeEmbedding
    private let early, single, embedders: [String: MFluxLinear]
    private let textEmbed, normOut, projOut: MFluxLinear
    private let modeBlocks: [String: [Flux3Block]]
    private let singleBlocks: [Flux3Block]
    private static let modes = ["image", "image_cond", "txt"]

    init(store: MFluxStore, cfg: Flux3TransformerConfig) throws {
        guard cfg.flowMap, cfg.axesDimsRope.count == 4, cfg.axesDimsRope.allSatisfy({ $0 > 0 && $0 % 2 == 0 }),
              cfg.axesDimsRope.reduce(0, +) == cfg.attentionHeadDim,
              cfg.numLayers > 0, cfg.numSingleLayers > 0, cfg.timestepChannels % 2 == 0 else {
            throw FluxError.invalidRequest("unsupported FLUX 3 transformer config")
        }
        self.cfg = cfg
        time = try Flux3TimeEmbedding(store: store, prefix: "timestep_embedder", cfg: cfg)
        targetTime = try Flux3TimeEmbedding(store: store, prefix: "target_timestep_embedder", cfg: cfg)
        func linear(_ name: String, _ input: Int, _ output: Int) throws -> MFluxLinear {
            try store.linear("transformer", name, inputDimensions: input, outputDimensions: output)
        }
        var e: [String: MFluxLinear] = [:], s: [String: MFluxLinear] = [:], inputs: [String: MFluxLinear] = [:]
        var blocks: [String: [Flux3Block]] = [:]
        for mode in Self.modes {
            e[mode] = try linear("early_stream_modulations.\(mode).linear", cfg.hidden, cfg.hidden * 3)
            s[mode] = try linear("single_stream_modulations.\(mode).linear", cfg.hidden, cfg.hidden * 3)
            let prefix = mode == "txt" ? "txt_mode_blocks" : "mode_blocks.\(mode)"
            blocks[mode] = try (0..<cfg.numLayers).map { try Flux3Block(store: store, prefix: "\(prefix).\($0)", cfg: cfg) }
        }
        inputs["image"] = try linear("x_embedders.image", cfg.inChannels, cfg.hidden)
        inputs["image_cond"] = try linear("x_embedders.image_cond", cfg.condInChannels, cfg.hidden)
        early = e; single = s; embedders = inputs; modeBlocks = blocks
        textEmbed = try linear("context_embedder", cfg.jointAttentionDim, cfg.hidden)
        normOut = try linear("norm_out.linear", cfg.hidden, cfg.hidden * 2)
        projOut = try linear("proj_out", cfg.hidden, cfg.inChannels)
        singleBlocks = try (0..<cfg.numSingleLayers).map { try Flux3Block(store: store, prefix: "single_transformer_blocks.\($0)", cfg: cfg) }
    }

    /// ids have rows (time, y*0.5, x*0.5, layer). References use layer -(i+1).
    func callAsFunction(_ latents: MLXArray, imageIDs: [[Float]], text: MLXArray, textIDs: [[Float]],
                        timestep: Float, targetTimestep: Float, references: MLXArray? = nil,
                        referenceIDs: [[Float]] = []) throws -> MLXArray {
        try FluxInterruptScope.check()
        let hasReferences = references.map { $0.dim(1) > 0 } ?? false
        let order = hasReferences ? Self.modes : ["image", "txt"]
        let target = targetTime(targetTimestep)
        let imageTime = time(timestep) + target, contextTime = time(0) + target
        let vectors = ["image": imageTime, "image_cond": contextTime, "txt": contextTime]
        var streams: [String: MLXArray] = ["image": embedders["image"]!(latents.asType(.bfloat16)), "txt": textEmbed(text.asType(.bfloat16))]
        if hasReferences { streams["image_cond"] = embedders["image_cond"]!(references!.asType(.bfloat16)) }
        let ids = ["image": imageIDs, "image_cond": referenceIDs, "txt": textIDs]
        func rotary(_ rows: [[Float]]) -> (MLXArray, MLXArray) {
            let table = Flux3RotaryTables.tables(ids: rows, axes: cfg.axesDimsRope, theta: Double(cfg.ropeTheta))
            return (MLXArray(table.cos, [rows.count, cfg.attentionHeadDim / 2]), MLXArray(table.sin, [rows.count, cfg.attentionHeadDim / 2]))
        }
        var singleMods: [String: Flux3Modulation] = [:]
        for mode in order {
            var x = streams[mode]!
            let vec = silu(vectors[mode]!)
            let mod = Flux3Modulation(early[mode]!(vec), hidden: cfg.hidden)
            singleMods[mode] = Flux3Modulation(single[mode]!(vec), hidden: cfg.hidden)
            let rope = rotary(ids[mode]!)
            for block in modeBlocks[mode]! {
                try FluxInterruptScope.check()
                x = block(x, rotary: rope, mod: mod)
                eval(x)
            }
            streams[mode] = x
        }
        var x = concatenated(order.map { streams[$0]! }, axis: 1)
        func combined(_ select: (Flux3Modulation) -> MLXArray) -> MLXArray {
            concatenated(order.map { mode in broadcast(select(singleMods[mode]!), to: streams[mode]!.shape) }, axis: 1)
        }
        let mod = Flux3Modulation(shift: combined { $0.shift }, scale: combined { $0.scale }, gate: combined { $0.gate })
        let rope = rotary(order.flatMap { ids[$0]! })
        for block in singleBlocks {
            try FluxInterruptScope.check()
            x = block(x, rotary: rope, mod: mod)
            eval(x)
        }
        x = x[0..., ..<latents.dim(1), 0...]
        let modulation = normOut(silu(imageTime))
        let scale = modulation[0..., .newAxis, ..<cfg.hidden], shift = modulation[0..., .newAxis, cfg.hidden...]
        return projOut(MLXFast.layerNorm(x, weight: nil, bias: nil, eps: 1e-6) * (1 + scale) + shift)
    }
}
