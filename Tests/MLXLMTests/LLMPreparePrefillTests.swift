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
    private final class ProgressRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [Int] = []

        func append(_ value: Int) {
            lock.lock()
            values.append(value)
            lock.unlock()
        }

        func snapshot() -> [Int] {
            lock.lock()
            defer { lock.unlock() }
            return values
        }
    }

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

    func testChunkedPrefillReportsCompletedUnitsAfterEachChunk() throws {
        let model = makeModel()
        let cache = model.newCache(parameters: nil)
        let recorder = ProgressRecorder()

        let tokens = MLXArray((0..<totalLen).map { Int32($0 % 64) })[.newAxis, 0...]
        let input = LMInput(text: .init(tokens: tokens))
        _ = try PrefillProgressReporter.withHandler({ recorder.append($0) }) {
            try model.prepare(input, cache: cache, windowSize: step)
        }

        XCTAssertEqual(recorder.snapshot(), [step, step * 2])
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

    // MARK: - Cancellation between chunks

    /// A client that disconnects mid-prefill cancels the producer task; the
    /// chunk loop must bail at the next chunk boundary instead of encoding
    /// the rest of the prompt for a dead request (an orphan producer racing
    /// a follow-up request's prefill on the shared GPU command queue aborts
    /// the process — osaurus cold-load disconnect crash).
    func testPrepareThrowsCancellationErrorWhenTaskCancelled() async {
        let recorder = ProgressRecorder()
        let totalLen = self.totalLen
        let step = self.step

        let task = Task { () throws -> Void in
            // Build the (non-Sendable) model/input inside the task so the
            // closure sends nothing across the boundary.
            let model = makeTinyPrefillLlama()
            let cache = model.newCache(parameters: nil)
            let tokens = MLXArray((0..<totalLen).map { Int32($0 % 64) })[.newAxis, 0...]
            let input = LMInput(text: .init(tokens: tokens))
            // Deterministic: set the cancellation flag before prepare runs so
            // the first chunk-boundary check observes it.
            withUnsafeCurrentTask { $0?.cancel() }
            _ = try PrefillProgressReporter.withHandler({ recorder.append($0) }) {
                try model.prepare(input, cache: cache, windowSize: step)
            }
        }

        do {
            try await task.value
            XCTFail("prepare must throw CancellationError when the task is cancelled")
        } catch is CancellationError {
            // Expected — and no chunk may have completed.
            XCTAssertEqual(
                recorder.snapshot(), [],
                "cancelled prepare must not encode any prompt chunks")
        } catch {
            XCTFail("expected CancellationError, got \(error)")
        }
    }

    /// Uncancelled tasks are unaffected: the loop still consumes the whole
    /// prompt and returns the remainder (guards against an over-eager check).
    func testPrepareInsideLiveTaskStillCompletes() async throws {
        let totalLen = self.totalLen
        let step = self.step

        let task = Task { () throws -> Int in
            let model = makeTinyPrefillLlama()
            let cache = model.newCache(parameters: nil)
            let tokens = MLXArray((0..<totalLen).map { Int32($0 % 64) })[.newAxis, 0...]
            let input = LMInput(text: .init(tokens: tokens))
            let result = try model.prepare(input, cache: cache, windowSize: step)
            guard case .tokens(let r) = result else { return -1 }
            return r.tokens.size
        }
        let remainderSize = try await task.value
        XCTAssertEqual(remainderSize, totalLen - 2 * step)
    }

    /// The VLM chunked-embedding helper shares the same contract.
    func testChunkedPrefillEmbeddingThrowsCancellationErrorWhenTaskCancelled() async {
        let stepCalls = ProgressRecorder()
        let totalLen = self.totalLen
        let step = self.step

        let task = Task { () throws -> Void in
            let embedding = MLXArray.zeros([1, totalLen, 8])
            withUnsafeCurrentTask { $0?.cancel() }
            _ = try chunkedPrefillEmbedding(
                inputEmbedding: embedding,
                cache: [],
                prefillStepSize: step
            ) { chunk in
                stepCalls.append(chunk.dim(1))
                return chunk
            }
        }

        do {
            try await task.value
            XCTFail("chunkedPrefillEmbedding must throw when the task is cancelled")
        } catch is CancellationError {
            XCTAssertEqual(
                stepCalls.snapshot(), [],
                "cancelled chunked prefill must not invoke the model step")
        } catch {
            XCTFail("expected CancellationError, got \(error)")
        }
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
}

/// Free-function twin of `LLMPreparePrefillTests.makeModel` for use inside
/// `Task` closures, where capturing the (non-Sendable) test instance would
/// trip strict-concurrency sending checks.
private func makeTinyPrefillLlama(vocab: Int = 64) -> LlamaModel {
    let config = LlamaConfiguration(
        hiddenSize: 32, hiddenLayers: 2, intermediateSize: 64,
        attentionHeads: 4, rmsNormEps: 1e-5,
        vocabularySize: vocab, kvHeads: 4)
    let model = LlamaModel(config)
    MLX.eval(model)
    return model
}
