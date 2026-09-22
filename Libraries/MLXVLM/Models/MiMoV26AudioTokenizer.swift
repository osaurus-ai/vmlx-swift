// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN

struct MiMoV26AudioTokenizerConfiguration: Decodable, Sendable {
    let d_model: Int
    let encoder_layers: Int
    let encoder_attention_heads: Int
    let encoder_ffn_dim: Int
    let encoder_skip_layer_id: Int?
    let encoder_causal: Bool
    let encoder_attn_window_size: [Int]
    let hybrid_attention: Bool
    let swa_per_block: Int
    let kernel_size: Int
    let stride_size: Int
    let avg_pooler: Int
    let rope_theta: Float
    let n_mels: Int
    let num_quantizers: Int
    let codebook_size: [Int]
    let ln_type: String
    let activation_function: String

    func validate() throws {
        guard d_model > 0, encoder_layers > 0, encoder_attention_heads > 0,
            d_model.isMultiple(of: encoder_attention_heads),
            (d_model / encoder_attention_heads).isMultiple(of: 2),
            encoder_ffn_dim > 0, !encoder_attn_window_size.isEmpty,
            swa_per_block > 0, kernel_size > 0, stride_size > 0, avg_pooler > 0,
            rope_theta > 0, n_mels > 0, num_quantizers > 0,
            codebook_size.count == num_quantizers, codebook_size.allSatisfy({ $0 > 0 }),
            ln_type == "LayerNorm", activation_function == "gelu"
        else { throw VLMError.processing("Unsupported MiMo audio tokenizer configuration") }
    }
    func window(_ layer: Int) -> Int {
        hybrid_attention && layer % swa_per_block == swa_per_block - 1 ? -1 : encoder_attn_window_size[0]
    }
    func convLength(_ input: Int) -> Int { (input + 5 - 2 * kernel_size) / stride_size + 1 }
}

final class MiMoV26TokenizerAttention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    let heads: Int
    let headDim: Int
    init(_ c: MiMoV26AudioTokenizerConfiguration) {
        heads = c.encoder_attention_heads; headDim = c.d_model / heads
        _q.wrappedValue = Linear(c.d_model, c.d_model)
        _k.wrappedValue = Linear(c.d_model, c.d_model, bias: false)
        _v.wrappedValue = Linear(c.d_model, c.d_model)
        _output.wrappedValue = Linear(c.d_model, c.d_model)
    }
    func callAsFunction(_ x: MLXArray, cosine: MLXArray, sine: MLXArray,
                        mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        let shape = [x.dim(0), x.dim(1), heads, headDim]
        let query = MiMoV26AudioMath.rotate(q(x).reshaped(shape).transposed(0, 2, 1, 3), cosine: cosine, sine: sine)
        let key = MiMoV26AudioMath.rotate(k(x).reshaped(shape).transposed(0, 2, 1, 3), cosine: cosine, sine: sine)
        let value = v(x).reshaped(shape).transposed(0, 2, 1, 3)
        let result = MLXFast.scaledDotProductAttention(queries: query, keys: key, values: value,
            scale: pow(Float(headDim), -0.5), mask: mask)
        return output(result.transposed(0, 2, 1, 3).reshaped(x.shape))
    }
}

final class MiMoV26TokenizerLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: MiMoV26TokenizerAttention
    @ModuleInfo(key: "self_attn_layer_norm") var inputNorm: LayerNorm
    @ModuleInfo(key: "final_layer_norm") var postNorm: LayerNorm
    let fc1: Linear
    let fc2: Linear
    let window: Int
    init(_ c: MiMoV26AudioTokenizerConfiguration, layer: Int) {
        _attention.wrappedValue = MiMoV26TokenizerAttention(c)
        _inputNorm.wrappedValue = LayerNorm(dimensions: c.d_model, eps: 1e-5)
        _postNorm.wrappedValue = LayerNorm(dimensions: c.d_model, eps: 1e-5)
        fc1 = Linear(c.d_model, c.encoder_ffn_dim)
        fc2 = Linear(c.encoder_ffn_dim, c.d_model)
        window = c.window(layer)
    }
    func callAsFunction(_ x: MLXArray, cosine: MLXArray, sine: MLXArray,
                        mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        let h = x + attention(inputNorm(x), cosine: cosine, sine: sine, mask: mask)
        return h + fc2(MiMoV26AudioMath.gelu(fc1(postNorm(h))))
    }
}

final class MiMoV26AudioDownsample: Module, UnaryLayer {
    let conv: Conv1d
    init(_ c: MiMoV26AudioTokenizerConfiguration) {
        conv = Conv1d(inputChannels: c.d_model, outputChannels: c.d_model,
                      kernelSize: c.avg_pooler, stride: c.avg_pooler, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { MiMoV26AudioMath.gelu(conv(x)) }
}

final class MiMoV26Codebook: Module {
    @ParameterInfo var weight: MLXArray
    init(size: Int, dimensions: Int) { _weight.wrappedValue = MLXArray.zeros([size, dimensions]) }
}

/// Encoder half of MiMo's audio tokenizer. Decoder/vocoder weights are never
/// part of this module. Codebook distance and residual arithmetic stay FP32.
final class MiMoV26AudioTokenizer: Module {
    let conv1: Conv1d
    let conv2: Conv1d
    let layers: [MiMoV26TokenizerLayer]
    @ModuleInfo(key: "layer_norm") var norm: LayerNorm
    @ModuleInfo(key: "down_sample_layer") var downsample: MiMoV26AudioDownsample?
    @ModuleInfo(key: "down_sample_norm") var downsampleNorm: LayerNorm?
    let codebooks: [MiMoV26Codebook]
    let configuration: MiMoV26AudioTokenizerConfiguration

    init(_ c: MiMoV26AudioTokenizerConfiguration) throws {
        try c.validate()
        configuration = c
        conv1 = Conv1d(inputChannels: c.n_mels, outputChannels: c.d_model, kernelSize: c.kernel_size, padding: 1)
        conv2 = Conv1d(inputChannels: c.d_model, outputChannels: c.d_model,
                      kernelSize: c.kernel_size, stride: c.stride_size, padding: 1)
        layers = (0..<c.encoder_layers).map { MiMoV26TokenizerLayer(c, layer: $0) }
        _norm.wrappedValue = LayerNorm(dimensions: c.d_model, eps: 1e-5)
        _downsample.wrappedValue = c.avg_pooler == 1 ? nil : MiMoV26AudioDownsample(c)
        _downsampleNorm.wrappedValue = c.avg_pooler == 1 ? nil : LayerNorm(dimensions: c.d_model, eps: 1e-5)
        codebooks = c.codebook_size.map { MiMoV26Codebook(size: $0, dimensions: c.d_model) }
    }

    private func transformer(_ input: MLXArray) -> MLXArray {
        let c = configuration, length = input.dim(1)
        let (cosine, sine) = MiMoV26AudioMath.rotary(length: length, dimensions: c.d_model / c.encoder_attention_heads,
            theta: c.rope_theta, dtype: input.dtype, roundFrequencies: true)
        var h = input, skip: MLXArray?
        let positions = MLXArray(Array(0..<length))
        let rows = positions[0..., .newAxis], cols = positions[.newAxis, 0...]
        for (index, layer) in layers.enumerated() {
            let mask: MLXFast.ScaledDotProductAttentionMaskMode
            if layer.window <= 0 { mask = c.encoder_causal ? .causal : .none }
            else {
                var allowed = abs(rows - cols) .<= layer.window
                if c.encoder_causal { allowed = allowed .&& (cols .<= rows) }
                mask = .array(allowed)
            }
            h = layer(h, cosine: cosine, sine: sine, mask: mask)
            if let layer = c.encoder_skip_layer_id, index == layer - 1 { skip = h }
        }
        if let skip { h = h + skip }
        return norm(h)
    }

    func features(_ segments: [MLXArray]) throws -> [MLXArray] {
        let c = configuration
        guard !segments.isEmpty, segments.allSatisfy({ $0.ndim == 2 && $0.dim(0) > 0 && $0.dim(1) == c.n_mels }) else {
            throw VLMError.processing("MiMo audio tokenizer expects nonempty mel segments")
        }
        let maximum = segments.map { $0.dim(0) }.max()!
        let paddedSegments = segments.map {
            padded($0, widths: [.init((0, maximum - $0.dim(0))), .init((0, 0))])
        }
        var h = stacked(paddedSegments).asType(conv1.weight.dtype)
        h = MiMoV26AudioMath.gelu(conv1(h))
        h = MiMoV26AudioMath.gelu(conv2(h))
        let lengths = segments.map { c.convLength($0.dim(0)) }
        // Segment positions reset independently. Convolutions above retain the
        // reference's shared zero-padding at the right edge of shorter clips.
        let packed = lengths.enumerated().map { index, length in
            transformer(h[index, ..<length][.newAxis])[0]
        }
        guard let downsample, let downsampleNorm else { return packed }
        let frames = h.dim(1)
        let rows = packed.enumerated().map { index, value -> MLXArray in
            let length = lengths[index]
            guard length < frames else { return value }
            let last = value[(length - 1)..<length, 0...]
            return concatenated([value, repeated(last, count: frames - length, axis: 0)], axis: 0)
        }
        var batched = stacked(rows)
        let extra = (c.avg_pooler - frames % c.avg_pooler) % c.avg_pooler
        if extra > 0 {
            batched = padded(batched, widths: [.init((0, 0)), .init((0, extra)), .init((0, 0))])
        }
        let pooled = downsample(batched)
        return lengths.enumerated().map { index, length in
            downsampleNorm(pooled[index, ..<((length + c.avg_pooler - 1) / c.avg_pooler)])
        }
    }

    func quantize(_ hidden: MLXArray) -> MLXArray {
        var residual = hidden.asType(.float32)
        var codes: [MLXArray] = []
        for book in codebooks {
            let weights = book.weight.asType(.float32)
            let distance = -((residual * residual).sum(axis: 1, keepDims: true)
                - 2 * residual.matmul(weights.T) + (weights * weights).sum(axis: 1)[.newAxis, 0...])
            let indices = argMax(distance, axis: -1)
            residual = residual - weights[indices]
            codes.append(indices)
        }
        return stacked(codes, axis: 1).asType(.int32)
    }

    func tokenize(_ mels: [MLXArray], segmentSize: Int, maximumBatchFrames: Int = 256_000) throws -> [MLXArray] {
        // RVQ assignments require the native FP32 encoder/codebook contract.
        // On M5, the default Metal F32 GEMM uses TF32, which changes actual
        // audio codes. Scope precise CPU operations to this frontend only;
        // do not alter the application's global Metal precision policy.
        try Device.withDefaultDevice(.cpu) {
            try tokenizePrecise(mels, segmentSize: segmentSize, maximumBatchFrames: maximumBatchFrames)
        }
    }

    private func tokenizePrecise(_ mels: [MLXArray], segmentSize: Int, maximumBatchFrames: Int) throws -> [MLXArray] {
        guard segmentSize > 0, maximumBatchFrames >= segmentSize, !mels.isEmpty,
            mels.allSatisfy({ $0.ndim == 2 && $0.dim(0) > 0 }) else {
            throw VLMError.processing("Invalid MiMo audio segment batch")
        }
        var segments: [MLXArray] = [], clipCounts: [Int] = []
        for mel in mels {
            var count = 0
            for offset in stride(from: 0, to: mel.dim(0), by: segmentSize) {
                segments.append(mel[offset..<min(offset + segmentSize, mel.dim(0))])
                count += 1
            }
            clipCounts.append(count)
        }
        var allCodes: [MLXArray] = [], batch: [MLXArray] = [], frames = 0
        func finishBatch() throws {
            guard !batch.isEmpty else { return }
            let encoded = try features(batch)
            let codes = quantize(concatenated(encoded, axis: 0))
            eval(codes)
            var offset = 0
            for hidden in encoded {
                allCodes.append(codes[offset..<(offset + hidden.dim(0))])
                offset += hidden.dim(0)
            }
        }
        for segment in segments {
            if frames + segment.dim(0) > maximumBatchFrames && !batch.isEmpty {
                try finishBatch(); batch = []; frames = 0
            }
            batch.append(segment); frames += segment.dim(0)
        }
        try finishBatch()
        var offset = 0
        return clipCounts.map { count in
            defer { offset += count }
            return concatenated(Array(allCodes[offset..<(offset + count)]), axis: 0)
        }
    }
}
