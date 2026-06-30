// Rampart PII NER — encoder-only BERT (MiniLM-L6) token classifier.
//
// Module tree is named so its flattened parameter paths match the
// HuggingFace-style keys in `model.safetensors` exactly
// (e.g. `bert.encoder.layer.0.attention.self.query.weight`), so weights
// load with no remapping.

import Foundation
import MLX
import MLXNN

public struct RampartConfig: Codable, Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let intermediateSize: Int
    public let vocabSize: Int
    public let maxPositionEmbeddings: Int
    public let typeVocabSize: Int
    public let layerNormEps: Float
    public let id2label: [String: String]

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case vocabSize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case typeVocabSize = "type_vocab_size"
        case layerNormEps = "layer_norm_eps"
        case id2label
    }

    public var numLabels: Int { id2label.count }

    public func label(_ index: Int) -> String { id2label[String(index)] ?? "O" }

    public static func load(from url: URL) throws -> RampartConfig {
        try JSONDecoder().decode(RampartConfig.self, from: Data(contentsOf: url))
    }
}

// MARK: - bert.embeddings

private final class Embeddings: Module {
    @ModuleInfo(key: "word_embeddings") var word: Embedding
    @ModuleInfo(key: "position_embeddings") var position: Embedding
    @ModuleInfo(key: "token_type_embeddings") var tokenType: Embedding
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm

    init(_ c: RampartConfig) {
        _word.wrappedValue = Embedding(embeddingCount: c.vocabSize, dimensions: c.hiddenSize)
        _position.wrappedValue = Embedding(
            embeddingCount: c.maxPositionEmbeddings, dimensions: c.hiddenSize)
        _tokenType.wrappedValue = Embedding(
            embeddingCount: c.typeVocabSize, dimensions: c.hiddenSize)
        _layerNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
    }

    func callAsFunction(_ inputIds: MLXArray, _ tokenTypeIds: MLXArray) -> MLXArray {
        let seq = inputIds.dim(1)
        let posIds = MLXArray.arange(seq).reshaped(1, seq)
        let h = word(inputIds) + position(posIds) + tokenType(tokenTypeIds)
        return layerNorm(h)
    }
}

// MARK: - one encoder layer (bert.encoder.layer.N)

private final class SelfAttention: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear

    init(_ c: RampartConfig) {
        _query.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        _key.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        _value.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
    }
}

private final class SelfOutput: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm

    init(_ c: RampartConfig) {
        _dense.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        _layerNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
    }
}

private final class Attention: Module {
    @ModuleInfo(key: "self") var selfAttn: SelfAttention
    @ModuleInfo(key: "output") var output: SelfOutput

    init(_ c: RampartConfig) {
        _selfAttn.wrappedValue = SelfAttention(c)
        _output.wrappedValue = SelfOutput(c)
    }
}

private final class Intermediate: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    init(_ c: RampartConfig) { _dense.wrappedValue = Linear(c.hiddenSize, c.intermediateSize) }
}

private final class OutputFFN: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm

    init(_ c: RampartConfig) {
        _dense.wrappedValue = Linear(c.intermediateSize, c.hiddenSize)
        _layerNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
    }
}

private final class EncoderLayer: Module {
    @ModuleInfo(key: "attention") var attention: Attention
    @ModuleInfo(key: "intermediate") var intermediate: Intermediate
    @ModuleInfo(key: "output") var output: OutputFFN

    let numHeads: Int
    let headDim: Int

    init(_ c: RampartConfig) {
        _attention.wrappedValue = Attention(c)
        _intermediate.wrappedValue = Intermediate(c)
        _output.wrappedValue = OutputFFN(c)
        numHeads = c.numAttentionHeads
        headDim = c.hiddenSize / c.numAttentionHeads
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let (b, s) = (x.dim(0), x.dim(1))
        func split(_ t: MLXArray) -> MLXArray {
            t.reshaped(b, s, numHeads, headDim).transposed(0, 2, 1, 3)
        }
        let q = split(attention.selfAttn.query(x))
        let k = split(attention.selfAttn.key(x))
        let v = split(attention.selfAttn.value(x))

        var scores = matmul(q, k.transposed(0, 1, 3, 2)) / Float(Foundation.sqrt(Double(headDim)))
        scores = scores + mask
        let probs = softmax(scores, axis: -1)
        let ctx = matmul(probs, v).transposed(0, 2, 1, 3).reshaped(b, s, numHeads * headDim)

        let attnOut = attention.output.layerNorm(attention.output.dense(ctx) + x)
        let inter = gelu(intermediate.dense(attnOut))
        return output.layerNorm(output.dense(inter) + attnOut)
    }
}

// MARK: - bert.encoder

private final class Encoder: Module {
    @ModuleInfo(key: "layer") var layer: [EncoderLayer]
    init(_ c: RampartConfig) {
        _layer.wrappedValue = (0 ..< c.numHiddenLayers).map { _ in EncoderLayer(c) }
    }
}

// MARK: - bert

private final class Bert: Module {
    @ModuleInfo(key: "embeddings") var embeddings: Embeddings
    @ModuleInfo(key: "encoder") var encoder: Encoder
    init(_ c: RampartConfig) {
        _embeddings.wrappedValue = Embeddings(c)
        _encoder.wrappedValue = Encoder(c)
    }
}

// MARK: - top-level token classifier

public final class RampartModel: Module {
    @ModuleInfo(key: "bert") fileprivate var bert: Bert
    @ModuleInfo(key: "classifier") fileprivate var classifier: Linear

    public let config: RampartConfig

    public init(_ c: RampartConfig) {
        config = c
        _bert.wrappedValue = Bert(c)
        _classifier.wrappedValue = Linear(c.hiddenSize, c.numLabels)
        super.init()
    }

    /// - Parameters:
    ///   - inputIds: `[batch, seq]` Int32 token ids.
    ///   - attentionMask: `[batch, seq]` 1 for real tokens, 0 for padding.
    /// - Returns: logits `[batch, seq, numLabels]`.
    public func callAsFunction(_ inputIds: MLXArray, attentionMask: MLXArray) -> MLXArray {
        let tokenTypeIds = MLXArray.zeros(like: inputIds)
        var h = bert.embeddings(inputIds, tokenTypeIds)
        let additive =
            (1.0 - attentionMask.asType(.float32)).expandedDimensions(axes: [1, 2]) * -1e9
        for layer in bert.encoder.layer {
            h = layer(h, mask: additive)
        }
        return classifier(h)
    }
}
