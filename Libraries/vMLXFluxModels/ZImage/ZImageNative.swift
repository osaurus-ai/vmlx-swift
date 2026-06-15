import Foundation
@preconcurrency import MLX
import MLXNN
import MLXRandom
import VMLXTokenizers
import vMLXFluxKit

private enum ZImageNative {
    static let dim = 3840
    static let textDim = 2560
    static let heads = 30
    static let headDim = 128
    static let layers = 30
    static let refinerLayers = 2
    static let patchSize = 2
    static let framePatchSize = 1
    static let channels = 16
    static let tScale: Float = 1000
    static let quantizationGroupSize = 64
    static let maxPromptTokens = 512
    static let vaeScale: Float = 0.3611
    static let vaeShift: Float = 0.1159
}

private final class ZImageWeightStore {
    private let loaded: LoadedWeights

    init(_ loaded: LoadedWeights) {
        self.loaded = loaded
    }

    func tensor(_ component: String, _ key: String) throws -> MLXArray {
        if let value = optionalTensor(component, key) {
            return value
        }
        throw FluxError.invalidRequest("missing \(component) weight \(key)")
    }

    func optionalTensor(_ component: String, _ key: String) -> MLXArray? {
        for candidate in candidateKeys(component: component, key: key) {
            if let value = loaded.componentWeights[component]?[candidate] {
                return value
            }
            if let value = loaded.weights["\(component).\(candidate)"] {
                return value
            }
            if let value = loaded.weights[candidate] {
                return value
            }
        }
        return nil
    }

    private func candidateKeys(component: String, key: String) -> [String] {
        var candidates = [key]
        if component == "text_encoder" {
            candidates.append("model.\(key)")
        }
        if key.hasPrefix("t_embedder.linear1.") {
            candidates.append(key.replacingOccurrences(of: "t_embedder.linear1.", with: "t_embedder.mlp.0."))
        }
        if key.hasPrefix("t_embedder.linear2.") {
            candidates.append(key.replacingOccurrences(of: "t_embedder.linear2.", with: "t_embedder.mlp.2."))
        }
        if key.hasPrefix("all_final_layer.2-1.adaLN_modulation.0.") {
            candidates.append(key.replacingOccurrences(
                of: "all_final_layer.2-1.adaLN_modulation.0.",
                with: "all_final_layer.2-1.adaLN_modulation.1."))
        }
        if component == "vae" {
            if key.hasPrefix("decoder.conv_in.conv.") {
                candidates.append(key.replacingOccurrences(of: "decoder.conv_in.conv.", with: "decoder.conv_in."))
            }
            if key.hasPrefix("decoder.conv_norm_out.norm.") {
                candidates.append(key.replacingOccurrences(
                    of: "decoder.conv_norm_out.norm.",
                    with: "decoder.conv_norm_out."))
            }
            if key.hasPrefix("decoder.conv_out.conv.") {
                candidates.append(key.replacingOccurrences(of: "decoder.conv_out.conv.", with: "decoder.conv_out."))
            }
        }
        return candidates
    }

    func linear(
        component: String,
        prefix: String,
        inputDimensions: Int,
        outputDimensions: Int,
        bias: Bool
    ) throws -> ZImageLinear {
        try ZImageLinear(
            weight: tensor(component, "\(prefix).weight"),
            scales: optionalTensor(component, "\(prefix).scales"),
            biases: optionalTensor(component, "\(prefix).biases"),
            bias: bias ? optionalTensor(component, "\(prefix).bias") : nil,
            inputDimensions: inputDimensions,
            outputDimensions: outputDimensions,
            name: "\(component).\(prefix)"
        )
    }

    func embedding(
        component: String,
        prefix: String,
        dimensions: Int
    ) throws -> ZImageEmbedding {
        try ZImageEmbedding(
            weight: tensor(component, "\(prefix).weight"),
            scales: optionalTensor(component, "\(prefix).scales"),
            biases: optionalTensor(component, "\(prefix).biases"),
            dimensions: dimensions,
            name: "\(component).\(prefix)"
        )
    }

    func rmsNorm(component: String, prefix: String, eps: Float) throws -> ZImageRMSNorm {
        try ZImageRMSNorm(weight: tensor(component, "\(prefix).weight"), eps: eps)
    }

    func groupNorm(component: String, prefix: String) throws -> ZImageGroupNorm {
        try ZImageGroupNorm(
            weight: tensor(component, "\(prefix).weight"),
            bias: tensor(component, "\(prefix).bias"))
    }

    func conv2d(
        component: String,
        prefix: String,
        stride: Int = 1,
        padding: Int = 0
    ) throws -> ZImageConv2D {
        try ZImageConv2D(
            weight: tensor(component, "\(prefix).weight"),
            bias: optionalTensor(component, "\(prefix).bias"),
            stride: stride,
            padding: padding)
    }
}

private final class ZImageLinear {
    private let weight: MLXArray
    private let scales: MLXArray?
    private let biases: MLXArray?
    private let bias: MLXArray?
    private let groupSize: Int
    private let bits: Int

    init(
        weight: MLXArray,
        scales: MLXArray?,
        biases: MLXArray?,
        bias: MLXArray?,
        inputDimensions: Int,
        outputDimensions: Int,
        name: String
    ) throws {
        guard weight.dim(0) == outputDimensions else {
            throw FluxError.invalidRequest(
                "\(name) output mismatch: weight=\(weight.shape), expected output \(outputDimensions)")
        }
        if weight.ndim == 4, weight.dim(2) == weight.dim(3), weight.dim(1) != weight.dim(2) {
            self.weight = weight.transposed(0, 2, 3, 1)
        } else {
            self.weight = weight
        }
        self.scales = scales
        self.biases = biases
        self.bias = bias

        if scales != nil {
            var inferredBits: Int?
            for candidate in [2, 3, 4, 5, 6, 8] where weight.dim(1) * 32 / candidate == inputDimensions {
                inferredBits = candidate
                break
            }
            guard let inferredBits else {
                throw FluxError.invalidRequest(
                    "\(name) quantized input mismatch: weight=\(weight.shape), expected input \(inputDimensions)")
            }
            let scaleColumns = scales?.dim(1) ?? 0
            guard scaleColumns > 0, inputDimensions % scaleColumns == 0 else {
                throw FluxError.invalidRequest("\(name) invalid quantization scales \(scales?.shape ?? [])")
            }
            self.bits = inferredBits
            self.groupSize = inputDimensions / scaleColumns
        } else {
            guard weight.dim(1) == inputDimensions else {
                throw FluxError.invalidRequest(
                    "\(name) input mismatch: weight=\(weight.shape), expected input \(inputDimensions)")
            }
            self.bits = 0
            self.groupSize = 0
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y: MLXArray
        if let scales {
            y = quantizedMM(
                x,
                weight,
                scales: scales,
                biases: biases,
                transpose: true,
                groupSize: groupSize,
                bits: bits,
                mode: .affine)
        } else {
            y = matmul(x, weight.T)
        }
        if let bias {
            y = y + bias
        }
        return y
    }
}

private final class ZImageEmbedding {
    private let weight: MLXArray
    private let scales: MLXArray?
    private let biases: MLXArray?
    private let groupSize: Int
    private let bits: Int

    init(
        weight: MLXArray,
        scales: MLXArray?,
        biases: MLXArray?,
        dimensions: Int,
        name: String
    ) throws {
        self.weight = weight
        self.scales = scales
        self.biases = biases
        if scales != nil {
            var inferredBits: Int?
            for candidate in [2, 3, 4, 5, 6, 8] where weight.dim(1) * 32 / candidate == dimensions {
                inferredBits = candidate
                break
            }
            guard let inferredBits else {
                throw FluxError.invalidRequest(
                    "\(name) quantized dimension mismatch: weight=\(weight.shape), expected \(dimensions)")
            }
            let scaleColumns = scales?.dim(1) ?? 0
            guard scaleColumns > 0, dimensions % scaleColumns == 0 else {
                throw FluxError.invalidRequest("\(name) invalid quantization scales \(scales?.shape ?? [])")
            }
            self.bits = inferredBits
            self.groupSize = dimensions / scaleColumns
        } else {
            guard weight.dim(1) == dimensions else {
                throw FluxError.invalidRequest(
                    "\(name) dimension mismatch: weight=\(weight.shape), expected \(dimensions)")
            }
            self.bits = 0
            self.groupSize = 0
        }
    }

    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        let selected = weight[ids]
        guard let scales else {
            return selected
        }
        let selectedScales = scales[ids]
        let selectedBiases = biases == nil ? nil : biases![ids]
        return dequantized(
            selected,
            scales: selectedScales,
            biases: selectedBiases,
            groupSize: groupSize,
            bits: bits,
            mode: .affine)
    }
}

private final class ZImageRMSNorm {
    private let weight: MLXArray
    private let eps: Float

    init(weight: MLXArray, eps: Float) {
        self.weight = weight
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

private final class ZImageLayerNorm {
    private let eps: Float

    init(eps: Float) {
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let meanValue = mean(x, axis: -1, keepDims: true)
        let centered = x - meanValue
        let varianceValue = mean(centered * centered, axis: -1, keepDims: true)
        return centered * rsqrt(varianceValue + MLXArray(eps))
    }
}

private final class ZImageGroupNorm {
    private let weight: MLXArray
    private let bias: MLXArray
    private let groups: Int = 32
    private let eps: Float = 1e-6

    init(weight: MLXArray, bias: MLXArray) {
        self.weight = weight
        self.bias = bias
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let dims = x.dim(-1)
        let rest = Array(x.shape.dropFirst().dropLast())
        let groupSize = dims / groups
        var y = x.reshaped([batch, -1, groups, groupSize])
        y = y.transposed(0, 2, 1, 3).reshaped([batch, groups, -1])
        let meanValue = mean(y, axis: -1, keepDims: true)
        let centered = y - meanValue
        let varianceValue = mean(centered * centered, axis: -1, keepDims: true)
        y = centered * rsqrt(varianceValue + MLXArray(eps))
        y = y.reshaped([batch, groups, -1, groupSize])
            .transposed(0, 2, 1, 3)
            .reshaped([batch] + rest + [dims])
        return y * weight + bias
    }
}

private final class ZImageConv2D {
    private let weight: MLXArray
    private let bias: MLXArray?
    private let stride: Int
    private let padding: Int

    init(weight: MLXArray, bias: MLXArray?, stride: Int, padding: Int) {
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv2d(x, weight, stride: IntOrPair(stride), padding: IntOrPair(padding))
        if let bias {
            y = y + bias
        }
        return y
    }
}

private final class ZImageTokenizer {
    private let tokenizer: any VMLXTokenizers.Tokenizer
    private let padTokenId: Int

    init(modelPath: URL) async throws {
        let tokenizerDirectory = modelPath.appendingPathComponent("tokenizer", isDirectory: true)
        self.tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDirectory, strict: false)
        self.padTokenId = tokenizer.convertTokenToId("<|endoftext|>") ?? tokenizer.eosTokenId ?? 0
    }

    func tokenizePrompt(_ prompt: String) -> [Int] {
        let formatted = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
        let encoded = tokenizer.encode(text: formatted, addSpecialTokens: true)
        if encoded.count > ZImageNative.maxPromptTokens {
            return Array(encoded.prefix(ZImageNative.maxPromptTokens))
        }
        return encoded
    }

    func inputArray(for prompt: String) -> MLXArray {
        let tokens = tokenizePrompt(prompt)
        if tokens.isEmpty {
            return MLXArray([padTokenId]).reshaped([1, 1]).asType(.int32)
        }
        return MLXArray(tokens.map(Int32.init)).reshaped([1, tokens.count])
    }
}

private final class ZImageTextAttention {
    private let numHeads = 32
    private let numKVHeads = 8
    private let headDim = 128
    private let numKVGroups = 4
    private let scale = Float(1.0 / sqrt(Float(128)))
    private let qProj: ZImageLinear
    private let kProj: ZImageLinear
    private let vProj: ZImageLinear
    private let oProj: ZImageLinear
    private let qNorm: ZImageRMSNorm
    private let kNorm: ZImageRMSNorm

    init(store: ZImageWeightStore, prefix: String) throws {
        self.qProj = try store.linear(
            component: "text_encoder", prefix: "\(prefix).q_proj",
            inputDimensions: 2560, outputDimensions: 4096, bias: false)
        self.kProj = try store.linear(
            component: "text_encoder", prefix: "\(prefix).k_proj",
            inputDimensions: 2560, outputDimensions: 1024, bias: false)
        self.vProj = try store.linear(
            component: "text_encoder", prefix: "\(prefix).v_proj",
            inputDimensions: 2560, outputDimensions: 1024, bias: false)
        self.oProj = try store.linear(
            component: "text_encoder", prefix: "\(prefix).o_proj",
            inputDimensions: 4096, outputDimensions: 2560, bias: false)
        self.qNorm = try store.rmsNorm(component: "text_encoder", prefix: "\(prefix).q_norm", eps: 1e-6)
        self.kNorm = try store.rmsNorm(component: "text_encoder", prefix: "\(prefix).k_norm", eps: 1e-6)
    }

    func callAsFunction(_ hiddenStates: MLXArray, mask: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let batch = hiddenStates.dim(0)
        let seqLen = hiddenStates.dim(1)
        var q = qProj(hiddenStates).reshaped([batch, seqLen, numHeads, headDim])
        var k = kProj(hiddenStates).reshaped([batch, seqLen, numKVHeads, headDim])
        var v = vProj(hiddenStates).reshaped([batch, seqLen, numKVHeads, headDim])
        q = qNorm(q)
        k = kNorm(k)
        let roped = applyTextRotary(q: q, k: k, cos: cos, sin: sin)
        q = roped.q
        k = roped.k
        if numKVGroups > 1 {
            k = repeated(k, count: numKVGroups, axis: 2)
            v = repeated(v, count: numKVGroups, axis: 2)
        }
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)
        let attn = MLX.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)
        let merged = attn.transposed(0, 2, 1, 3).reshaped([batch, seqLen, numHeads * headDim])
        return oProj(merged)
    }

    private func applyTextRotary(q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray)
        -> (q: MLXArray, k: MLXArray)
    {
        let cos = cos.reshaped([cos.dim(0), cos.dim(1), 1, cos.dim(2)])
        let sin = sin.reshaped([sin.dim(0), sin.dim(1), 1, sin.dim(2)])
        return (
            q: q * cos + rotateHalf(q) * sin,
            k: k * cos + rotateHalf(k) * sin
        )
    }

    private func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, half ..< x.dim(-1)], x[.ellipsis, 0 ..< half]], axis: -1)
    }
}

private final class ZImageTextMLP {
    private let gateProj: ZImageLinear
    private let upProj: ZImageLinear
    private let downProj: ZImageLinear

    init(store: ZImageWeightStore, prefix: String) throws {
        self.gateProj = try store.linear(
            component: "text_encoder", prefix: "\(prefix).gate_proj",
            inputDimensions: 2560, outputDimensions: 9728, bias: false)
        self.upProj = try store.linear(
            component: "text_encoder", prefix: "\(prefix).up_proj",
            inputDimensions: 2560, outputDimensions: 9728, bias: false)
        self.downProj = try store.linear(
            component: "text_encoder", prefix: "\(prefix).down_proj",
            inputDimensions: 9728, outputDimensions: 2560, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

private final class ZImageTextEncoderLayer {
    private let inputLayerNorm: ZImageRMSNorm
    private let postAttentionLayerNorm: ZImageRMSNorm
    private let selfAttention: ZImageTextAttention
    private let mlp: ZImageTextMLP

    init(store: ZImageWeightStore, index: Int) throws {
        let prefix = "layers.\(index)"
        self.inputLayerNorm = try store.rmsNorm(
            component: "text_encoder", prefix: "\(prefix).input_layernorm", eps: 1e-6)
        self.postAttentionLayerNorm = try store.rmsNorm(
            component: "text_encoder", prefix: "\(prefix).post_attention_layernorm", eps: 1e-6)
        self.selfAttention = try ZImageTextAttention(store: store, prefix: "\(prefix).self_attn")
        self.mlp = try ZImageTextMLP(store: store, prefix: "\(prefix).mlp")
    }

    func callAsFunction(_ hiddenStates: MLXArray, mask: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let residual = hiddenStates
        var hiddenStates = selfAttention(inputLayerNorm(hiddenStates), mask: mask, cos: cos, sin: sin)
        hiddenStates = residual + hiddenStates
        return hiddenStates + mlp(postAttentionLayerNorm(hiddenStates))
    }
}

private final class ZImageTextEncoder {
    private let embedTokens: ZImageEmbedding
    private let layers: [ZImageTextEncoderLayer]
    private let invFreq: MLXArray

    init(store: ZImageWeightStore) throws {
        self.embedTokens = try store.embedding(
            component: "text_encoder", prefix: "embed_tokens", dimensions: 2560)
        var builtLayers: [ZImageTextEncoderLayer] = []
        builtLayers.reserveCapacity(36)
        for index in 0..<36 {
            builtLayers.append(try ZImageTextEncoderLayer(store: store, index: index))
        }
        self.layers = builtLayers
        let values = stride(from: 0, to: 128, by: 2).map {
            Float(1.0 / pow(1_000_000.0, Float($0) / 128.0))
        }
        self.invFreq = MLXArray(values)
    }

    func encode(inputIds: MLXArray) -> MLXArray {
        let batch = inputIds.dim(0)
        let seqLen = inputIds.dim(1)
        var hiddenStates = embedTokens(inputIds).asType(.float32)
        let position = MLXArray((0..<seqLen).map(Float.init))
        let freqs = matmul(position.reshaped([seqLen, 1]), invFreq.reshaped([1, invFreq.dim(0)]))
        let emb = concatenated([freqs, freqs], axis: -1)
        let cos = MLX.cos(emb).reshaped([1, seqLen, 128]).asType(hiddenStates.dtype)
        let sin = MLX.sin(emb).reshaped([1, seqLen, 128]).asType(hiddenStates.dtype)
        let mask = causalMask(batchSize: batch, seqLen: seqLen, dtype: hiddenStates.dtype)
        var penultimate = hiddenStates
        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cos: cos, sin: sin)
            if index == layers.count - 2 {
                penultimate = hiddenStates
            }
        }
        return penultimate.asType(.bfloat16)
    }

    private func causalMask(batchSize: Int, seqLen: Int, dtype: DType) -> MLXArray {
        let idx = arange(seqLen, dtype: .int32)
        let allowed = idx.reshaped([seqLen, 1]) .>= idx.reshaped([1, seqLen])
        let zeros = MLXArray.zeros([seqLen, seqLen], dtype: dtype)
        let negInf = full([seqLen, seqLen], values: MLXArray(-Float.infinity), dtype: dtype)
        return MLX.where(allowed, zeros, negInf).reshaped([1, 1, seqLen, seqLen])
    }
}

private final class ZImageAttention {
    private let dim = ZImageNative.dim
    private let heads = ZImageNative.heads
    private let headDim = ZImageNative.headDim
    private let scale = Float(1.0 / sqrt(Float(ZImageNative.headDim)))
    private let toQ: ZImageLinear
    private let toK: ZImageLinear
    private let toV: ZImageLinear
    private let toOut: ZImageLinear
    private let normQ: ZImageRMSNorm
    private let normK: ZImageRMSNorm

    init(store: ZImageWeightStore, prefix: String) throws {
        self.toQ = try store.linear(
            component: "transformer", prefix: "\(prefix).to_q",
            inputDimensions: dim, outputDimensions: dim, bias: false)
        self.toK = try store.linear(
            component: "transformer", prefix: "\(prefix).to_k",
            inputDimensions: dim, outputDimensions: dim, bias: false)
        self.toV = try store.linear(
            component: "transformer", prefix: "\(prefix).to_v",
            inputDimensions: dim, outputDimensions: dim, bias: false)
        self.toOut = try store.linear(
            component: "transformer", prefix: "\(prefix).to_out.0",
            inputDimensions: dim, outputDimensions: dim, bias: false)
        self.normQ = try store.rmsNorm(component: "transformer", prefix: "\(prefix).norm_q", eps: 1e-5)
        self.normK = try store.rmsNorm(component: "transformer", prefix: "\(prefix).norm_k", eps: 1e-5)
    }

    func callAsFunction(_ hiddenStates: MLXArray, freqsCis: MLXArray) -> MLXArray {
        let batch = hiddenStates.dim(0)
        let seqLen = hiddenStates.dim(1)
        var query = toQ(hiddenStates).reshaped([batch, seqLen, heads, headDim])
        var key = toK(hiddenStates).reshaped([batch, seqLen, heads, headDim])
        var value = toV(hiddenStates).reshaped([batch, seqLen, heads, headDim])
        query = applyRotary(normQ(query), freqsCis: freqsCis)
        key = applyRotary(normK(key), freqsCis: freqsCis)
        query = query.transposed(0, 2, 1, 3)
        key = key.transposed(0, 2, 1, 3)
        value = value.transposed(0, 2, 1, 3)
        let attended = MLX.scaledDotProductAttention(
            queries: query, keys: key, values: value, scale: scale, mask: nil)
        let merged = attended.transposed(0, 2, 1, 3).reshaped([batch, seqLen, dim])
        return toOut(merged)
    }

    private func applyRotary(_ x: MLXArray, freqsCis: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let seqLen = x.dim(1)
        let reshaped = x.reshaped([batch, seqLen, heads, headDim / 2, 2])
        let freqs = freqsCis.reshaped([1, freqsCis.dim(0), 1, freqsCis.dim(1), 2])
        let xReal = reshaped[.ellipsis, 0]
        let xImag = reshaped[.ellipsis, 1]
        let cosFreqs = freqs[.ellipsis, 0]
        let sinFreqs = freqs[.ellipsis, 1]
        let outReal = xReal * cosFreqs - xImag * sinFreqs
        let outImag = xReal * sinFreqs + xImag * cosFreqs
        return stacked([outReal, outImag], axis: -1).reshaped([batch, seqLen, heads, headDim])
    }
}

private final class ZImageFeedForward {
    private let w1: ZImageLinear
    private let w2: ZImageLinear
    private let w3: ZImageLinear

    init(store: ZImageWeightStore, prefix: String) throws {
        self.w1 = try store.linear(
            component: "transformer", prefix: "\(prefix).w1",
            inputDimensions: ZImageNative.dim, outputDimensions: 10240, bias: false)
        self.w2 = try store.linear(
            component: "transformer", prefix: "\(prefix).w2",
            inputDimensions: 10240, outputDimensions: ZImageNative.dim, bias: false)
        self.w3 = try store.linear(
            component: "transformer", prefix: "\(prefix).w3",
            inputDimensions: ZImageNative.dim, outputDimensions: 10240, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

private final class ZImageTransformerBlock {
    private let attention: ZImageAttention
    private let feedForward: ZImageFeedForward
    private let attentionNorm1: ZImageRMSNorm
    private let attentionNorm2: ZImageRMSNorm
    private let ffnNorm1: ZImageRMSNorm
    private let ffnNorm2: ZImageRMSNorm
    private let modulation: ZImageLinear

    init(store: ZImageWeightStore, prefix: String) throws {
        self.attention = try ZImageAttention(store: store, prefix: "\(prefix).attention")
        self.feedForward = try ZImageFeedForward(store: store, prefix: "\(prefix).feed_forward")
        self.attentionNorm1 = try store.rmsNorm(
            component: "transformer", prefix: "\(prefix).attention_norm1", eps: 1e-5)
        self.attentionNorm2 = try store.rmsNorm(
            component: "transformer", prefix: "\(prefix).attention_norm2", eps: 1e-5)
        self.ffnNorm1 = try store.rmsNorm(
            component: "transformer", prefix: "\(prefix).ffn_norm1", eps: 1e-5)
        self.ffnNorm2 = try store.rmsNorm(
            component: "transformer", prefix: "\(prefix).ffn_norm2", eps: 1e-5)
        self.modulation = try store.linear(
            component: "transformer", prefix: "\(prefix).adaLN_modulation.0",
            inputDimensions: 256, outputDimensions: ZImageNative.dim * 4, bias: true)
    }

    func callAsFunction(_ x: MLXArray, freqsCis: MLXArray, tEmb: MLXArray) -> MLXArray {
        let mod = modulation(silu(tEmb)).reshaped([tEmb.dim(0), 1, ZImageNative.dim * 4])
        let scaleMSA = MLXArray(1) + mod[.ellipsis, 0 ..< ZImageNative.dim]
        let gateMSA = tanh(mod[.ellipsis, ZImageNative.dim ..< ZImageNative.dim * 2])
        let scaleMLP = MLXArray(1) + mod[.ellipsis, ZImageNative.dim * 2 ..< ZImageNative.dim * 3]
        let gateMLP = tanh(mod[.ellipsis, ZImageNative.dim * 3 ..< ZImageNative.dim * 4])
        var out = x
        let attnOut = attention(attentionNorm1(out) * scaleMSA, freqsCis: freqsCis)
        out = out + gateMSA * attentionNorm2(attnOut)
        let ffnOut = feedForward(ffnNorm1(out) * scaleMLP)
        out = out + gateMLP * ffnNorm2(ffnOut)
        return out
    }
}

private final class ZImageContextBlock {
    private let attention: ZImageAttention
    private let feedForward: ZImageFeedForward
    private let attentionNorm1: ZImageRMSNorm
    private let attentionNorm2: ZImageRMSNorm
    private let ffnNorm1: ZImageRMSNorm
    private let ffnNorm2: ZImageRMSNorm

    init(store: ZImageWeightStore, prefix: String) throws {
        self.attention = try ZImageAttention(store: store, prefix: "\(prefix).attention")
        self.feedForward = try ZImageFeedForward(store: store, prefix: "\(prefix).feed_forward")
        self.attentionNorm1 = try store.rmsNorm(
            component: "transformer", prefix: "\(prefix).attention_norm1", eps: 1e-5)
        self.attentionNorm2 = try store.rmsNorm(
            component: "transformer", prefix: "\(prefix).attention_norm2", eps: 1e-5)
        self.ffnNorm1 = try store.rmsNorm(
            component: "transformer", prefix: "\(prefix).ffn_norm1", eps: 1e-5)
        self.ffnNorm2 = try store.rmsNorm(
            component: "transformer", prefix: "\(prefix).ffn_norm2", eps: 1e-5)
    }

    func callAsFunction(_ x: MLXArray, freqsCis: MLXArray) -> MLXArray {
        var out = x
        let attnOut = attention(attentionNorm1(out), freqsCis: freqsCis)
        out = out + attentionNorm2(attnOut)
        let ffnOut = feedForward(ffnNorm1(out))
        out = out + ffnNorm2(ffnOut)
        return out
    }
}

private final class ZImageFinalLayer {
    private let norm = ZImageLayerNorm(eps: 1e-6)
    private let linear: ZImageLinear
    private let modulation: ZImageLinear

    init(store: ZImageWeightStore) throws {
        self.linear = try store.linear(
            component: "transformer", prefix: "all_final_layer.2-1.linear",
            inputDimensions: ZImageNative.dim, outputDimensions: 64, bias: true)
        self.modulation = try store.linear(
            component: "transformer", prefix: "all_final_layer.2-1.adaLN_modulation.0",
            inputDimensions: 256, outputDimensions: ZImageNative.dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray, tEmb: MLXArray) -> MLXArray {
        let scale = MLXArray(1) + modulation(silu(tEmb)).reshaped([tEmb.dim(0), 1, ZImageNative.dim])
        return linear(norm(x) * scale)
    }
}

private final class ZImageRopeEmbedder {
    private let freqsCis: [MLXArray]

    init(theta: Float = 256, axesDims: [Int] = [32, 48, 48], axesLens: [Int] = [1024, 512, 512]) {
        var caches: [MLXArray] = []
        for (dim, length) in zip(axesDims, axesLens) {
            let freqs = MLXArray(stride(from: 0, to: dim, by: 2).map {
                Float(1.0 / pow(theta, Float($0) / Float(dim)))
            })
            let positions = MLXArray((0..<length).map(Float.init))
            let outer = matmul(positions.reshaped([length, 1]), freqs.reshaped([1, freqs.dim(0)]))
            caches.append(stacked([cos(outer), sin(outer)], axis: -1))
        }
        self.freqsCis = caches
    }

    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        var parts: [MLXArray] = []
        for axis in 0..<freqsCis.count {
            let indices = ids[0 ..< ids.dim(0), axis].asType(.int32)
            parts.append(freqsCis[axis][indices])
        }
        return concatenated(parts, axis: 1)
    }
}

private final class ZImageTransformer {
    private let xEmbedder: ZImageLinear
    private let tLinear1: ZImageLinear
    private let tLinear2: ZImageLinear
    private let capNorm: ZImageRMSNorm
    private let capLinear: ZImageLinear
    private let xPadToken: MLXArray
    private let capPadToken: MLXArray
    private let noiseRefiner: [ZImageTransformerBlock]
    private let contextRefiner: [ZImageContextBlock]
    private let layers: [ZImageTransformerBlock]
    private let finalLayer: ZImageFinalLayer
    private let ropeEmbedder = ZImageRopeEmbedder()

    init(store: ZImageWeightStore) throws {
        self.xEmbedder = try store.linear(
            component: "transformer", prefix: "all_x_embedder.2-1",
            inputDimensions: 64, outputDimensions: ZImageNative.dim, bias: true)
        self.tLinear1 = try store.linear(
            component: "transformer", prefix: "t_embedder.linear1",
            inputDimensions: 256, outputDimensions: 1024, bias: true)
        self.tLinear2 = try store.linear(
            component: "transformer", prefix: "t_embedder.linear2",
            inputDimensions: 1024, outputDimensions: 256, bias: true)
        self.capNorm = try store.rmsNorm(component: "transformer", prefix: "cap_embedder.0", eps: 1e-5)
        self.capLinear = try store.linear(
            component: "transformer", prefix: "cap_embedder.1",
            inputDimensions: ZImageNative.textDim, outputDimensions: ZImageNative.dim, bias: true)
        self.xPadToken = try store.tensor("transformer", "x_pad_token")
        self.capPadToken = try store.tensor("transformer", "cap_pad_token")

        var noise: [ZImageTransformerBlock] = []
        var context: [ZImageContextBlock] = []
        var blocks: [ZImageTransformerBlock] = []
        for index in 0..<ZImageNative.refinerLayers {
            noise.append(try ZImageTransformerBlock(store: store, prefix: "noise_refiner.\(index)"))
            context.append(try ZImageContextBlock(store: store, prefix: "context_refiner.\(index)"))
        }
        for index in 0..<ZImageNative.layers {
            blocks.append(try ZImageTransformerBlock(store: store, prefix: "layers.\(index)"))
        }
        self.noiseRefiner = noise
        self.contextRefiner = context
        self.layers = blocks
        self.finalLayer = try ZImageFinalLayer(store: store)
    }

    func callAsFunction(x: MLXArray, timestep: MLXArray, capFeats: MLXArray) -> MLXArray {
        let tEmb = timestepEmbedding(timestep.asType(.float32) * MLXArray(ZImageNative.tScale))
        let packed = patchify(image: x, capFeats: capFeats)
        var xEmb = xEmbedder(packed.image)
        xEmb = MLX.where(packed.imagePadMask.reshaped([packed.imagePadMask.dim(0), 1]), xPadToken, xEmb)
        let xFreqs = ropeEmbedder(packed.imagePositionIds)
        xEmb = xEmb.reshaped([1, xEmb.dim(0), xEmb.dim(1)])
        for layer in noiseRefiner {
            xEmb = layer(xEmb, freqsCis: xFreqs, tEmb: tEmb)
        }

        var capEmb = capLinear(capNorm(packed.caption))
        capEmb = MLX.where(packed.captionPadMask.reshaped([packed.captionPadMask.dim(0), 1]), capPadToken, capEmb)
        let capFreqs = ropeEmbedder(packed.captionPositionIds)
        capEmb = capEmb.reshaped([1, capEmb.dim(0), capEmb.dim(1)])
        for layer in contextRefiner {
            capEmb = layer(capEmb, freqsCis: capFreqs)
        }

        let xLen = xEmb.dim(1)
        var unified = concatenated([xEmb, capEmb], axis: 1)
        let freqs = concatenated([xFreqs, capFreqs], axis: 0)
        for layer in layers {
            unified = layer(unified, freqsCis: freqs, tEmb: tEmb)
        }
        let final = finalLayer(unified, tEmb: tEmb)
        let imageTokens = final[0, 0 ..< xLen, 0 ..< final.dim(2)]
        return -unpatchify(
            imageTokens,
            size: packed.imageSize,
            outChannels: ZImageNative.channels)
    }

    private func timestepEmbedding(_ t: MLXArray) -> MLXArray {
        let half = 128
        let freqs = MLXArray((0..<half).map { -log(Float(10000)) * Float($0) / Float(half) })
        let args = t.reshaped([t.dim(0), 1]) * exp(freqs).reshaped([1, half])
        return tLinear2(silu(tLinear1(concatenated([cos(args), sin(args)], axis: -1))))
    }

    private struct Patchified {
        let image: MLXArray
        let caption: MLXArray
        let imageSize: (frames: Int, height: Int, width: Int)
        let imagePositionIds: MLXArray
        let captionPositionIds: MLXArray
        let imagePadMask: MLXArray
        let captionPadMask: MLXArray
    }

    private func patchify(image: MLXArray, capFeats: MLXArray) -> Patchified {
        let capLen = capFeats.dim(0)
        let capPadding = (32 - (capLen % 32)) % 32
        let capPositionIds = coordinateGrid(
            size: (capLen + capPadding, 1, 1),
            start: (1, 0, 0))
        let capPadMask = MLXArray(
            Array(repeating: false, count: capLen) + Array(repeating: true, count: capPadding))
        let capPadded: MLXArray
        if capPadding > 0 {
            capPadded = concatenated(
                [capFeats, repeated(capFeats[capLen - 1 ..< capLen], count: capPadding, axis: 0)],
                axis: 0)
        } else {
            capPadded = capFeats
        }

        let channels = image.dim(0)
        let frames = image.dim(1)
        let height = image.dim(2)
        let width = image.dim(3)
        let frameTokens = frames / ZImageNative.framePatchSize
        let heightTokens = height / ZImageNative.patchSize
        let widthTokens = width / ZImageNative.patchSize
        var packedImage = image.reshaped([
            channels,
            frameTokens,
            ZImageNative.framePatchSize,
            heightTokens,
            ZImageNative.patchSize,
            widthTokens,
            ZImageNative.patchSize,
        ])
        packedImage = packedImage.transposed(1, 3, 5, 2, 4, 6, 0)
        packedImage = packedImage.reshaped([
            frameTokens * heightTokens * widthTokens,
            ZImageNative.framePatchSize * ZImageNative.patchSize * ZImageNative.patchSize * channels,
        ])

        let imageLen = packedImage.dim(0)
        let imagePadding = (32 - (imageLen % 32)) % 32
        var imagePositionIds = coordinateGrid(
            size: (frameTokens, heightTokens, widthTokens),
            start: (capLen + capPadding + 1, 0, 0))
        if imagePadding > 0 {
            imagePositionIds = concatenated(
                [imagePositionIds, MLXArray.zeros([imagePadding, 3], dtype: .int32)],
                axis: 0)
            packedImage = concatenated(
                [packedImage, repeated(packedImage[imageLen - 1 ..< imageLen], count: imagePadding, axis: 0)],
                axis: 0)
        }
        let imagePadMask = MLXArray(
            Array(repeating: false, count: imageLen) + Array(repeating: true, count: imagePadding))
        return Patchified(
            image: packedImage,
            caption: capPadded,
            imageSize: (frames, height, width),
            imagePositionIds: imagePositionIds,
            captionPositionIds: capPositionIds,
            imagePadMask: imagePadMask,
            captionPadMask: capPadMask)
    }

    private func coordinateGrid(size: (Int, Int, Int), start: (Int, Int, Int)) -> MLXArray {
        var values: [Int32] = []
        values.reserveCapacity(size.0 * size.1 * size.2 * 3)
        for f in 0..<size.0 {
            for h in 0..<size.1 {
                for w in 0..<size.2 {
                    values.append(Int32(start.0 + f))
                    values.append(Int32(start.1 + h))
                    values.append(Int32(start.2 + w))
                }
            }
        }
        return MLXArray(values).reshaped([size.0 * size.1 * size.2, 3])
    }

    private func unpatchify(
        _ x: MLXArray,
        size: (frames: Int, height: Int, width: Int),
        outChannels: Int
    ) -> MLXArray {
        let frameTokens = size.frames / ZImageNative.framePatchSize
        let heightTokens = size.height / ZImageNative.patchSize
        let widthTokens = size.width / ZImageNative.patchSize
        let originalLength = frameTokens * heightTokens * widthTokens
        var out = x[0 ..< originalLength].reshaped([
            frameTokens,
            heightTokens,
            widthTokens,
            ZImageNative.framePatchSize,
            ZImageNative.patchSize,
            ZImageNative.patchSize,
            outChannels,
        ])
        out = out.transposed(6, 0, 3, 1, 4, 2, 5)
        return out.reshaped([outChannels, size.frames, size.height, size.width])
    }
}

private final class ZImageVAEAttention {
    private let groupNorm: ZImageGroupNorm
    private let toQ: ZImageLinear
    private let toK: ZImageLinear
    private let toV: ZImageLinear
    private let toOut: ZImageLinear
    private let channels: Int

    init(store: ZImageWeightStore, prefix: String, channels: Int = 512) throws {
        self.channels = channels
        self.groupNorm = try store.groupNorm(component: "vae", prefix: "\(prefix).group_norm")
        self.toQ = try store.linear(
            component: "vae", prefix: "\(prefix).to_q",
            inputDimensions: channels, outputDimensions: channels, bias: true)
        self.toK = try store.linear(
            component: "vae", prefix: "\(prefix).to_k",
            inputDimensions: channels, outputDimensions: channels, bias: true)
        self.toV = try store.linear(
            component: "vae", prefix: "\(prefix).to_v",
            inputDimensions: channels, outputDimensions: channels, bias: true)
        self.toOut = try store.linear(
            component: "vae", prefix: "\(prefix).to_out.0",
            inputDimensions: channels, outputDimensions: channels, bias: true)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let nhwc = input.transposed(0, 2, 3, 1)
        let batch = nhwc.dim(0)
        let height = nhwc.dim(1)
        let width = nhwc.dim(2)
        let normed = groupNorm(nhwc.asType(.float32)).asType(input.dtype)
        var q = toQ(normed).reshaped([batch, height * width, 1, channels]).transposed(0, 2, 1, 3)
        var k = toK(normed).reshaped([batch, height * width, 1, channels]).transposed(0, 2, 1, 3)
        var v = toV(normed).reshaped([batch, height * width, 1, channels]).transposed(0, 2, 1, 3)
        let scale = Float(1.0 / sqrt(Float(channels)))
        let attended = MLX.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        let out = attended.transposed(0, 2, 1, 3).reshaped([batch, height, width, channels])
        return (nhwc + toOut(out)).transposed(0, 3, 1, 2)
    }
}

private final class ZImageVAEResnetBlock {
    private let norm1: ZImageGroupNorm
    private let conv1: ZImageConv2D
    private let norm2: ZImageGroupNorm
    private let conv2: ZImageConv2D
    private let shortcut: ZImageConv2D?

    init(store: ZImageWeightStore, prefix: String) throws {
        self.norm1 = try store.groupNorm(component: "vae", prefix: "\(prefix).norm1")
        self.conv1 = try store.conv2d(component: "vae", prefix: "\(prefix).conv1", padding: 1)
        self.norm2 = try store.groupNorm(component: "vae", prefix: "\(prefix).norm2")
        self.conv2 = try store.conv2d(component: "vae", prefix: "\(prefix).conv2", padding: 1)
        if store.optionalTensor("vae", "\(prefix).conv_shortcut.weight") != nil {
            self.shortcut = try store.conv2d(component: "vae", prefix: "\(prefix).conv_shortcut")
        } else {
            self.shortcut = nil
        }
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let nhwc = input.transposed(0, 2, 3, 1)
        var hidden = norm1(nhwc)
        hidden = silu(hidden)
        hidden = conv1(hidden)
        hidden = norm2(hidden)
        hidden = silu(hidden)
        hidden = conv2(hidden)
        let residual = shortcut?(nhwc) ?? nhwc
        return (residual + hidden).transposed(0, 3, 1, 2)
    }
}

private final class ZImageVAEMidBlock {
    private let resnet0: ZImageVAEResnetBlock
    private let attention: ZImageVAEAttention
    private let resnet1: ZImageVAEResnetBlock

    init(store: ZImageWeightStore, prefix: String) throws {
        self.resnet0 = try ZImageVAEResnetBlock(store: store, prefix: "\(prefix).resnets.0")
        self.attention = try ZImageVAEAttention(store: store, prefix: "\(prefix).attentions.0")
        self.resnet1 = try ZImageVAEResnetBlock(store: store, prefix: "\(prefix).resnets.1")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        resnet1(attention(resnet0(x)))
    }
}

private final class ZImageVAEUpSampler {
    private let conv: ZImageConv2D

    init(store: ZImageWeightStore, prefix: String) throws {
        self.conv = try store.conv2d(component: "vae", prefix: "\(prefix).conv", padding: 1)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let up = repeated(repeated(input, count: 2, axis: 2), count: 2, axis: 3)
        return conv(up.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
    }
}

private final class ZImageVAEUpBlock {
    private let resnets: [ZImageVAEResnetBlock]
    private let upsampler: ZImageVAEUpSampler?

    init(store: ZImageWeightStore, index: Int, addUpsample: Bool) throws {
        var blocks: [ZImageVAEResnetBlock] = []
        for layer in 0..<3 {
            blocks.append(try ZImageVAEResnetBlock(
                store: store,
                prefix: "decoder.up_blocks.\(index).resnets.\(layer)"))
        }
        self.resnets = blocks
        self.upsampler = addUpsample
            ? try ZImageVAEUpSampler(store: store, prefix: "decoder.up_blocks.\(index).upsamplers.0")
            : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for resnet in resnets {
            out = resnet(out)
        }
        if let upsampler {
            out = upsampler(out)
        }
        return out
    }
}

private final class ZImageVAEDecoder {
    private let convIn: ZImageConv2D
    private let midBlock: ZImageVAEMidBlock
    private let upBlocks: [ZImageVAEUpBlock]
    private let normOut: ZImageGroupNorm
    private let convOut: ZImageConv2D

    init(store: ZImageWeightStore) throws {
        self.convIn = try store.conv2d(component: "vae", prefix: "decoder.conv_in.conv", padding: 1)
        self.midBlock = try ZImageVAEMidBlock(store: store, prefix: "decoder.mid_block")
        var blocks: [ZImageVAEUpBlock] = []
        for index in 0..<4 {
            blocks.append(try ZImageVAEUpBlock(store: store, index: index, addUpsample: index < 3))
        }
        self.upBlocks = blocks
        self.normOut = try store.groupNorm(component: "vae", prefix: "decoder.conv_norm_out.norm")
        self.convOut = try store.conv2d(component: "vae", prefix: "decoder.conv_out.conv", padding: 1)
    }

    func decode(_ latents: MLXArray) -> MLXArray {
        let scaled = latents / MLXArray(ZImageNative.vaeScale) + MLXArray(ZImageNative.vaeShift)
        var hidden = convIn(scaled.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
        hidden = midBlock(hidden)
        for block in upBlocks {
            hidden = block(hidden)
        }
        hidden = normOut(hidden.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
        hidden = silu(hidden)
        hidden = convOut(hidden.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
        return VAEDecoder.postprocess(hidden)
    }
}

final class ZImageNativePipeline {
    private let tokenizer: ZImageTokenizer
    private let textEncoder: ZImageTextEncoder
    private let transformer: ZImageTransformer
    private let vae: ZImageVAEDecoder

    init(modelPath: URL, loadedWeights: LoadedWeights) async throws {
        let store = ZImageWeightStore(loadedWeights)
        self.tokenizer = try await ZImageTokenizer(modelPath: modelPath)
        self.textEncoder = try ZImageTextEncoder(store: store)
        self.transformer = try ZImageTransformer(store: store)
        self.vae = try ZImageVAEDecoder(store: store)
    }

    func generate(
        prompt: String,
        negativePrompt: String?,
        guidance: Float,
        width: Int,
        height: Int,
        steps: Int,
        seed: UInt64?,
        progress: (Int, Int, Double?) -> Void
    ) async throws -> MLXArray {
        guard width % 16 == 0, height % 16 == 0 else {
            throw FluxError.invalidRequest("Z-Image width and height must be divisible by 16")
        }
        let start = Date()
        var latents = initialNoise(width: width, height: height, seed: seed)
        let textEncodings = encodePrompt(prompt)
        let negativeEncodings = (guidance > 0 && negativePrompt != nil) ? encodePrompt(negativePrompt ?? "") : nil
        let sigmas = linearSigmas(steps: steps)
        for step in 0..<steps {
            if Task.isCancelled {
                throw CancellationError()
            }
            let timestep = MLXArray([Float(1) - sigmas[step]])
            var noise = transformer(x: latents, timestep: timestep, capFeats: textEncodings)
            if let negativeEncodings {
                let negativeNoise = transformer(x: latents, timestep: timestep, capFeats: negativeEncodings)
                noise = noise + MLXArray(guidance) * (noise - negativeNoise)
            }
            latents = latents + noise * MLXArray(sigmas[step + 1] - sigmas[step])
            eval(latents)
            let elapsed = Date().timeIntervalSince(start)
            let perStep = elapsed / Double(step + 1)
            progress(step + 1, steps, perStep * Double(steps - step - 1))
        }
        let unpacked = latents.reshaped([
            1,
            ZImageNative.channels,
            height / 8,
            width / 8,
        ])
        return vae.decode(unpacked)
    }

    private func encodePrompt(_ prompt: String) -> MLXArray {
        let inputIds = tokenizer.inputArray(for: prompt)
        let hidden = textEncoder.encode(inputIds: inputIds)
        return hidden[0]
    }

    private func initialNoise(width: Int, height: Int, seed: UInt64?) -> MLXArray {
        if let seed {
            MLXRandom.seed(seed)
        }
        return MLXRandom.normal(
            [ZImageNative.channels, 1, height / 8, width / 8],
            dtype: .bfloat16)
    }

    private func linearSigmas(steps: Int) -> [Float] {
        guard steps > 1 else {
            return [1, 0]
        }
        var values: [Float] = []
        values.reserveCapacity(steps + 1)
        for index in 0..<steps {
            let t = Float(index) / Float(steps - 1)
            values.append(1 - t * (1 - 1 / Float(steps)))
        }
        values.append(0)
        return values
    }
}
