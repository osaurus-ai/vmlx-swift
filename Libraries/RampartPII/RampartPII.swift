// Public entry point: load the Rampart PII model from a directory and detect
// PII entity spans in text.

import Foundation
import MLX
import MLXNN

public struct PIISpan: Sendable, Equatable {
    public let type: String          // e.g. "EMAIL", "GIVEN_NAME"
    public let text: String
    public let range: Range<Int>     // character-index range into the input
    public let score: Float          // mean softmax confidence over the span
}

public final class RampartPII {
    private let model: RampartModel
    private let tokenizer: RampartTokenizer
    private let config: RampartConfig

    /// Load from a directory containing `model.safetensors`, `config.json`,
    /// and `vocab.txt`.
    public init(directory: URL) throws {
        let config = try RampartConfig.load(from: directory.appendingPathComponent("config.json"))
        let model = RampartModel(config)
        let weights = try loadArrays(url: directory.appendingPathComponent("model.safetensors"))
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)

        self.config = config
        self.model = model
        self.tokenizer = try RampartTokenizer(
            vocabURL: directory.appendingPathComponent("vocab.txt"),
            maxLength: config.maxPositionEmbeddings)
    }

    /// Detect PII spans. Adjacent same-type tokens are merged (the model emits
    /// `B-` on most subwords), using BIO tags and the tokenizer's char offsets.
    public func detect(_ text: String) -> [PIISpan] {
        let tokens = tokenizer.encode(text)
        let ids = MLXArray(tokens.map { Int32($0.id) }).reshaped(1, tokens.count)
        let mask = MLXArray(Array(repeating: Int32(1), count: tokens.count)).reshaped(1, tokens.count)

        let logits = model(ids, attentionMask: mask)[0]   // [seq, numLabels]
        let probs = softmax(logits, axis: -1)
        let labelIds = logits.argMax(axis: -1)
        eval(labelIds, probs)

        let chars = Array(text)
        var spans: [PIISpan] = []
        var current: (type: String, lo: Int, hi: Int, scoreSum: Float, count: Int)?

        func flush() {
            guard let c = current else { return }
            spans.append(PIISpan(
                type: c.type,
                text: String(chars[c.lo..<c.hi]),
                range: c.lo..<c.hi,
                score: c.scoreSum / Float(c.count)))
            current = nil
        }

        for (i, tok) in tokens.enumerated() {
            guard let range = tok.range else { continue }  // skip [CLS]/[SEP]
            let labelIndex = labelIds[i].item(Int.self)
            let label = config.label(labelIndex)
            if label == "O" { flush(); continue }

            let entityType = String(label.dropFirst(2))   // strip "B-"/"I-"
            let score = probs[i, labelIndex].item(Float.self)

            if var c = current, c.type == entityType, range.lowerBound <= c.hi + 1 {
                c.hi = range.upperBound
                c.scoreSum += score
                c.count += 1
                current = c
            } else {
                flush()
                current = (entityType, range.lowerBound, range.upperBound, score, 1)
            }
        }
        flush()
        return spans
    }

    /// Replace each detected span with `[TYPE]` (simple redaction helper).
    public func redact(_ text: String) -> String {
        let chars = Array(text)
        var out = ""
        var cursor = 0
        for span in detect(text).sorted(by: { $0.range.lowerBound < $1.range.lowerBound }) {
            if span.range.lowerBound > cursor {
                out += String(chars[cursor..<span.range.lowerBound])
            }
            out += "[\(span.type)]"
            cursor = span.range.upperBound
        }
        if cursor < chars.count { out += String(chars[cursor...]) }
        return out
    }
}
