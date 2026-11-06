import Foundation
import MLX

/// Decode callers must inspect this immediately after processing and sampling.
/// A failed constraint never authorizes sampling the unmodified logits.
public protocol ConstraintFailureReporting {
    var constraintFailure: JSONSchemaGrammarError? { get }
    /// True only after the matcher accepted a grammar-authorized stop token.
    var constraintIsComplete: Bool { get }
}

/// Request-local schema masking, composed after the existing processor pipeline.
/// No prompt injection, sampler overrides, reasoning tags, or completion repair.
public struct JSONSchemaLogitProcessor: LogitProcessor, ConstraintFailureReporting {
    private final class State {
        var grammar: JSONSchemaGrammar?
        var failure: JSONSchemaGrammarError?
        var terminated = false
        init(grammar: JSONSchemaGrammar?, failure: JSONSchemaGrammarError? = nil) {
            self.grammar = grammar
            self.failure = failure
        }
    }

    private var base: (any LogitProcessor)?
    private let state: State
    private let vocabularySize: Int
    private let stopTokenIDs: Set<Int>
    private let excludedTokenIDs: Set<Int>

    public var constraintFailure: JSONSchemaGrammarError? {
        state.failure ?? logitConstraintFailure(base)
    }
    public var constraintIsComplete: Bool {
        state.terminated && constraintFailure == nil && logitConstraintCompletion(base) != false
    }
    public var isTerminated: Bool { state.terminated && constraintFailure == nil }

    public init(
        schema: String, tokenizer: any Tokenizer, stopTokenIDs: Set<Int>,
        base: (any LogitProcessor)? = nil
    ) throws {
        guard let metadata = tokenizer.grammarTokenVocabulary else {
            throw JSONSchemaGrammarError.invalidTokenizer(
                "Tokenizer does not expose exact grammar vocabulary metadata")
        }
        guard metadata.specialTokenIDs.allSatisfy({ metadata.vocabulary.indices.contains($0) })
        else {
            throw JSONSchemaGrammarError.invalidTokenizer(
                "Special token ID is outside the vocabulary")
        }
        let grammarTokenizer = try JSONSchemaGrammarTokenizer(
            vocabulary: metadata.vocabulary, vocabularyType: metadata.vocabularyType,
            stopTokenIDs: stopTokenIDs.sorted())
        self.base = base
        self.state = State(
            grammar: try JSONSchemaGrammar(tokenizer: grammarTokenizer, schema: schema))
        self.vocabularySize = grammarTokenizer.vocabularySize
        self.stopTokenIDs = stopTokenIDs
        self.excludedTokenIDs = metadata.specialTokenIDs.subtracting(stopTokenIDs)
    }

    private init(
        base: (any LogitProcessor)?, state: State, vocabularySize: Int,
        stopTokenIDs: Set<Int>, excludedTokenIDs: Set<Int>
    ) {
        self.base = base
        self.state = state
        self.vocabularySize = vocabularySize
        self.stopTokenIDs = stopTokenIDs
        self.excludedTokenIDs = excludedTokenIDs
    }

    public func independentCopy() -> Self {
        let copied: State
        do {
            copied = State(grammar: try state.grammar?.independentCopy(), failure: state.failure)
        } catch {
            copied = State(grammar: nil, failure: Self.constraintError(error))
        }
        copied.terminated = state.terminated
        return Self(
            base: base?.independentCopy(), state: copied,
            vocabularySize: vocabularySize, stopTokenIDs: stopTokenIDs,
            excludedTokenIDs: excludedTokenIDs)
    }

    public mutating func prompt(_ prompt: MLXArray) {
        // The grammar describes generated response bytes, never prompt/history.
        base?.prompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        guard state.failure == nil else { return logits }
        let processed = base?.process(logits: logits) ?? logits
        if let failure = logitConstraintFailure(base) {
            state.failure = failure
            return logits
        }
        do {
            guard let grammar = state.grammar else {
                throw JSONSchemaGrammarError.runtime("Missing request-local grammar matcher")
            }
            guard logits.ndim >= 1, let count = logits.shape.last, count >= vocabularySize,
                logits.size == count
            else {
                throw JSONSchemaGrammarError.runtime("Schema processor requires one vocabulary row")
            }
            var allowed = Array(repeating: false, count: count)
            if state.terminated {
                // The AR iterator may prepare one token ahead before observing
                // EOS. Only an already authorized stop is legal in that state.
                for id in stopTokenIDs { allowed[id] = true }
            } else {
                let mask = try grammar.nextTokenMask()
                guard mask.vocabularySize == vocabularySize,
                    !mask.needsApply || mask.words.count >= (vocabularySize + 31) / 32
                else {
                    throw JSONSchemaGrammarError.runtime(
                        "Grammar mask does not match tokenizer vocabulary")
                }
                for id in 0 ..< vocabularySize {
                    allowed[id] =
                        !mask.needsApply
                        || (mask.words[id / 32] & (UInt32(1) << UInt32(id % 32))) != 0
                }
                for id in excludedTokenIDs { allowed[id] = false }
            }
            guard allowed.contains(true) else {
                throw JSONSchemaGrammarError.runtime("JSON schema has no valid next token")
            }
            let masked = MLX.where(
                MLXArray(allowed), processed,
                MLXArray(-Float.infinity, dtype: processed.dtype))
            // Existing suppression/reasoning processors can eliminate every
            // grammar-allowed token. Report that conflict, never sample NaNs.
            guard MLX.any(MLX.isFinite(masked)).item(Bool.self) else {
                throw JSONSchemaGrammarError.runtime(
                    "No finite token remains after schema and generation constraints")
            }
            return masked
        } catch {
            state.failure = Self.constraintError(error)
            return logits  // Caller must check constraintFailure before sampling.
        }
    }

    public mutating func didSample(token: MLXArray) {
        guard state.failure == nil else { return }
        do {
            guard token.size == 1, let grammar = state.grammar else {
                throw JSONSchemaGrammarError.runtime("Schema processor requires one sampled token")
            }
            let id = token.item(Int.self)
            guard (0 ..< vocabularySize).contains(id), !excludedTokenIDs.contains(id) else {
                throw JSONSchemaGrammarError.runtime(
                    "Sampled token is outside the allowed grammar vocabulary")
            }
            if state.terminated {
                guard stopTokenIDs.contains(id) else {
                    throw JSONSchemaGrammarError.runtime(
                        "Non-stop token sampled after schema completion")
                }
                return
            }
            try grammar.accept(tokenID: id)
            state.terminated = try grammar.isTerminated()
            base?.didSample(token: token)
        } catch {
            state.failure = Self.constraintError(error)
        }
    }

    private static func constraintError(_ error: Error) -> JSONSchemaGrammarError {
        (error as? JSONSchemaGrammarError) ?? .runtime(String(describing: error))
    }
}
