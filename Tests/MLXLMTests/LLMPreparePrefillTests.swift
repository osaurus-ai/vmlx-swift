// Copyright © 2026 Osaurus. All rights reserved.

import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

/// Regression coverage for `LLMModel.prepare` default chunked-prefill extension.
///
/// The bug: prior to commit 5b26831 the chunked-prefill loop used the
/// `LMInput.Text` variadic subscripts `y[.newAxis, ..<step]` and `y[step...]`
/// on the tokens tensor. Those two calls silently sliced the WRONG axis when
/// the tokens came in as 2D `[1, T]` instead of 1D `[T]`:
///
///   - `y[.newAxis, ..<step]` left the chunk shape unchanged (the `..<step`
///     applied to the batch dim of size 1, not the time dim).
///   - `y[step...]` produced an empty `[0, T]` tensor because `step...` was
///     slicing the batch axis past its end.
///
/// The loop then exited with an empty remainder, and the next forward pass
/// inside `TokenIterator.step` crashed with `[reshape] Cannot infer the shape
/// of an empty array`.
///
/// The user-visible reproducer was Qwen3.5-35B-A3B-JANG_2S-TEXT at prompt
/// lengths just above `prefillStepSize`, but the same axis-slicing bug would
/// fire on any `LLMModel`-factory-loaded model because the default prepare
/// extension is shared. These tests pin the behaviour with a tiny random
/// Llama so regressions surface under `swift test` without any model files.
final class LLMPreparePrefillTests: XCTestCase {

    /// Small random-init Llama that exercises the default `LLMModel.prepare`.
    private func makeModel(vocab: Int = 64) -> LlamaModel {
        let config = LlamaConfiguration(
            hiddenSize: 32, hiddenLayers: 2, intermediateSize: 64,
            attentionHeads: 4, rmsNormEps: 1e-5,
            vocabularySize: vocab, kvHeads: 4)
        let model = LlamaModel(config)
        MLX.eval(model)
        return model
    }

    /// Prompt length chosen so the chunked-prefill loop runs more than once:
    /// 600 > step(256) forces two full chunks + an 88-token remainder.
    private let totalLen = 600
    private let step = 256

    // MARK: - 1D input (legacy shape)

    func testChunkedPrefill1DInput() throws {
        let model = makeModel()
        let cache = model.newCache(parameters: nil)

        let tokens = MLXArray((0..<totalLen).map { Int32($0 % 64) })
        XCTAssertEqual(tokens.ndim, 1, "Starting shape must be 1D for this case")

        let input = LMInput(text: .init(tokens: tokens))
        let result = try model.prepare(input, cache: cache, windowSize: step)

        guard case .tokens(let remainder) = result else {
            XCTFail("expected .tokens remainder, got \(result)")
            return
        }

        // 600 - (2 * 256) = 88 tokens should remain.
        XCTAssertEqual(remainder.tokens.size, totalLen - 2 * step)
        // Contract: prepare always returns a 1D remainder so
        // TokenIterator.step / BatchEngine.stepPrefill can safely
        // add a batch axis themselves.
        XCTAssertEqual(
            remainder.tokens.ndim, 1,
            "prepare must return a 1D tokens remainder")
    }

    // MARK: - 2D input (TokenIterator / Bench / Osaurus shape)

    /// This is the case that used to crash with
    /// `[reshape] Cannot infer the shape of an empty array`.
    func testChunkedPrefill2DInputDoesNotCrashAndReturns1D() throws {
        let model = makeModel()
        let cache = model.newCache(parameters: nil)

        let tokens1D = MLXArray((0..<totalLen).map { Int32($0 % 64) })
        let tokens2D = tokens1D[.newAxis, 0...]
        XCTAssertEqual(tokens2D.ndim, 2)
        XCTAssertEqual(tokens2D.shape, [1, totalLen])

        let input = LMInput(text: .init(tokens: tokens2D))
        let result = try model.prepare(input, cache: cache, windowSize: step)

        guard case .tokens(let remainder) = result else {
            XCTFail("expected .tokens remainder, got \(result)")
            return
        }

        XCTAssertEqual(remainder.tokens.size, totalLen - 2 * step)
        XCTAssertEqual(
            remainder.tokens.ndim, 1,
            "prepare must return a 1D tokens remainder even for 2D input")
    }

    // MARK: - Short input (no chunking)

    /// When the prompt fits in a single chunk the loop does not run at all.
    /// Both 1D and 2D input must still produce a non-empty 1D remainder.
    func testShortInputReturnsAllTokensAs1D() throws {
        let model = makeModel()
        let cache = model.newCache(parameters: nil)

        let short = 5
        let tokens = MLXArray((0..<short).map { Int32($0 + 1) })

        // 1D
        do {
            let input = LMInput(text: .init(tokens: tokens))
            let result = try model.prepare(input, cache: cache, windowSize: step)
            guard case .tokens(let r) = result else { return XCTFail() }
            XCTAssertEqual(r.tokens.size, short)
            XCTAssertEqual(r.tokens.ndim, 1)
        }

        // 2D
        do {
            let input = LMInput(text: .init(tokens: tokens[.newAxis, 0...]))
            let result = try model.prepare(input, cache: cache, windowSize: step)
            guard case .tokens(let r) = result else { return XCTFail() }
            XCTAssertEqual(r.tokens.size, short)
            XCTAssertEqual(r.tokens.ndim, 1)
        }
    }

    // MARK: - Exact chunk boundary

    /// totalLen == prefillStepSize is the off-by-one boundary case that the
    /// loop `while size > step` keeps as a single forward pass (no chunking);
    /// the full prompt is returned as the remainder.
    func testPromptEqualToStepReturnsAllTokens() throws {
        let model = makeModel()
        let cache = model.newCache(parameters: nil)

        let tokens = MLXArray((0..<step).map { Int32($0 % 64) })
        let input = LMInput(text: .init(tokens: tokens[.newAxis, 0...]))
        let result = try model.prepare(input, cache: cache, windowSize: step)
        guard case .tokens(let r) = result else { return XCTFail() }
        XCTAssertEqual(r.tokens.size, step)
        XCTAssertEqual(r.tokens.ndim, 1)
    }

    // MARK: - Mask rank follows tokens rank

    func testMaskFlattenedAlongsideTokens() throws {
        let model = makeModel()
        let cache = model.newCache(parameters: nil)

        let tokens = MLXArray((0..<totalLen).map { Int32($0 % 64) })[.newAxis, 0...]
        let mask = MLXArray.ones([1, totalLen]).asType(.int8)

        let input = LMInput(text: .init(tokens: tokens, mask: mask))
        let result = try model.prepare(input, cache: cache, windowSize: step)
        guard case .tokens(let r) = result else { return XCTFail() }
        XCTAssertEqual(r.tokens.size, totalLen - 2 * step)
        XCTAssertEqual(r.tokens.ndim, 1)
        // Mask follows tokens rank.
        if let remainderMask = r.mask {
            XCTAssertEqual(remainderMask.ndim, 1)
            XCTAssertEqual(remainderMask.size, totalLen - 2 * step)
        }
    }

    func testTrueBatchedInputThrowsUnsupportedBatchInput() throws {
        let model = makeModel()
        let cache = model.newCache(parameters: nil)
        let tokens = MLXArray((0..<12).map { Int32($0 % 64) }).reshaped(2, 6)
        let input = LMInput(text: .init(tokens: tokens))

        do {
            _ = try model.prepare(input, cache: cache, windowSize: step)
            XCTFail("Expected true batched input to throw")
        } catch LLMModelError.unsupportedBatchInput(let shape) {
            XCTAssertEqual(shape, [2, 6])
        }
    }
}
