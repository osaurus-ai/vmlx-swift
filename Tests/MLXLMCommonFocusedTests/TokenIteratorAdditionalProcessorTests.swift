// Copyright © 2026 Tijs Teulings

import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

/// The smallest runtime seam token-level constrained decoding needs from the
/// fork: a request-supplied `LogitProcessor` riding the ordinary
/// parameters-based `TokenIterator` initializer, composed AFTER the built-in
/// penalty processors, before the native schema mask.
///
/// Contract under test:
/// 1. `TokenIterator(input:model:cache:parameters:...:additionalProcessor:)`
///    accepts a request-supplied processor and composes it with the built-in
///    effective processor.
/// 2. Composition order is penalties FIRST, additional constraint LAST.
/// 3. `prompt` / `process` / `didSample` are forwarded through the seam.
/// 4. `independentCopy` isolates the mutable state of both halves.
/// 5. Not supplying one changes nothing for ordinary callers.
@Suite("TokenIterator additional processor seam")
struct TokenIteratorAdditionalProcessorTests {

    // MARK: - Doubles

    /// Ordered record of everything the seam forwarded. Reference type: the
    /// processor itself is a value type, so observation needs shared storage.
    private final class Recorder {
        enum Event: Equatable {
            case prompt([Int])
            case process([Float])
            case didSample(Int)
        }

        var events: [Event] = []

        var processRows: [[Float]] {
            events.compactMap {
                if case .process(let row) = $0 { return row }
                return nil
            }
        }

        var didSampleIDs: [Int] {
            events.compactMap {
                if case .didSample(let id) = $0 { return id }
                return nil
            }
        }
    }

    /// Reference-typed state, so a plain value copy of the processor would
    /// SHARE it. `independentCopy` must replace it — same discipline as
    /// `PresencePenaltyContext`.
    private final class MutableBox {
        var samples: [Int] = []
    }

    private struct RecordingProcessor: LogitProcessor {
        let recorder: Recorder
        var box: MutableBox

        /// Offset added AFTER the row is recorded, so tests can both observe
        /// the incoming (already penalised) logits and prove the processor's
        /// return value reaches the sampler. +100 at `markerIndex`.
        let marker: MLXArray?

        init(recorder: Recorder, markerIndex: Int? = nil, vocab: Int = 8) {
            self.recorder = recorder
            self.box = MutableBox()
            if let markerIndex {
                var values = [Float](repeating: 0, count: vocab)
                values[markerIndex] = 100
                self.marker = MLXArray(values)
            } else {
                self.marker = nil
            }
        }

        mutating func prompt(_ prompt: MLXArray) {
            recorder.events.append(.prompt(prompt.reshaped(-1).asArray(Int.self)))
        }

        func process(logits: MLXArray) -> MLXArray {
            recorder.events.append(.process(logits.reshaped(-1).asArray(Float.self)))
            guard let marker else { return logits }
            return logits + marker
        }

        mutating func didSample(token: MLXArray) {
            let id = token.reshaped(-1)[0].item(Int.self)
            recorder.events.append(.didSample(id))
            box.samples.append(id)
        }

        func independentCopy() -> Self {
            var copy = self
            let fresh = MutableBox()
            fresh.samples = box.samples
            copy.box = fresh
            return copy
        }
    }

    /// The lightest model that satisfies the runtime path: prefill as plain
    /// tokens and a zero-logit forward over an 8-token vocabulary.
    private final class SeamTestModel: Module, LanguageModel, KVCacheDimensionProvider,
        @unchecked Sendable
    {
        var kvHeads: [Int] { [1] }

        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            .tokens(input.text)
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            let batch = inputs.shape.first ?? 1
            let length = inputs.shape.count > 1 ? inputs.shape[1] : inputs.size
            return MLXArray.zeros([batch, length, 8], dtype: .float32)
        }
    }

    private func makeInput(_ tokens: [Int]) -> LMInput {
        LMInput(tokens: MLXArray(tokens.map { Int32($0) }))
    }

    // MARK: - 1. The seam exists and composes

    @Test("a request-supplied processor is accepted and composed with the built-in one")
    func requestSuppliedProcessorIsAcceptedAndComposed() throws {
        try FocusedMLXTestSupport.withLock {
            let recorder = Recorder()
            var parameters = GenerateParameters(maxTokens: 4, temperature: 0)
            parameters.repetitionPenalty = 2.0
            parameters.repetitionContextSize = 8

            let iterator = try TokenIterator(
                input: makeInput([1, 2]),
                model: SeamTestModel(),
                parameters: parameters,
                additionalProcessor: RecordingProcessor(recorder: recorder, markerIndex: 5))

            let processor = try #require(iterator.processor)
            #expect(
                processor is CompositeLogitProcessor,
                "the request-supplied processor must be composed with the built-in effective processor")
            }
    }

    // MARK: - 2. Ordinary callers are untouched

    @Test("without an additional processor the built-in effective processor is unchanged")
    func defaultPathKeepsBuiltInProcessor() throws {
        try FocusedMLXTestSupport.withLock {
            var parameters = GenerateParameters(maxTokens: 4, temperature: 0)
            parameters.repetitionPenalty = 2.0
            parameters.repetitionContextSize = 8

            let iterator = try TokenIterator(
                input: makeInput([1, 2]),
                model: SeamTestModel(),
                parameters: parameters)

            let processor = try #require(iterator.processor)
            #expect(!(processor is CompositeLogitProcessor))
            #expect(processor is PenaltyProcessor)
            }
    }

    @Test("a request with no configured penalties and no additional processor keeps a nil processor")
    func noPenaltiesNoAdditionalProcessorStaysNil() throws {
        try FocusedMLXTestSupport.withLock {
            let parameters = GenerateParameters(maxTokens: 4, temperature: 0)
            let iterator = try TokenIterator(
                input: makeInput([1, 2]),
                model: SeamTestModel(),
                parameters: parameters)
            #expect(iterator.processor == nil)
            }
    }

    // MARK: - 3. Order: penalties first, additional constraint last

    @Test("the additional constraint sees already-penalised logits (penalties run first)")
    func penaltiesRunBeforeTheAdditionalConstraint() throws {
        try FocusedMLXTestSupport.withLock {
            let recorder = Recorder()
            var parameters = GenerateParameters(maxTokens: 4, temperature: 0)
            parameters.repetitionPenalty = 2.0
            parameters.repetitionContextSize = 8

            var iterator = try TokenIterator(
                input: makeInput([1, 2]),
                model: SeamTestModel(),
                parameters: parameters,
                additionalProcessor: RecordingProcessor(recorder: recorder, markerIndex: 5))

            // Ring after init: prompt {1, 2} plus the first sampled token {5}.
            // Probe row: in-ring positives must be halved (4 -> 2), the in-ring
            // negative doubled (-4 -> -8), the out-of-ring positive untouched.
            let probe = MLXArray([Float(0), 4, 4, 4, 0, -4, 0, 0]).reshaped(1, 1, 8)
            let sampled = iterator.convertToToken(logits: probe)

            let observed = try #require(recorder.processRows.last)
            #expect(observed.count == 8)
            #expect(
                observed[1] == 2.0,
                "positive in-ring logit must already be halved when the additional processor sees it")
            #expect(
                observed[5] == -8.0,
                "negative in-ring logit must already be doubled when the additional processor sees it")
            #expect(observed[3] == 4.0, "out-of-ring logit must be untouched by the penalty stage")

            // And the additional processor's own return value reaches the sampler:
            // its +100 marker at index 5 (92 after the penalty) is the argmax.
            #expect(sampled.item(Int.self) == 5)
            #expect(recorder.didSampleIDs.last == 5)
            }
    }

    // MARK: - 4. prompt / process / didSample forwarding

    @Test("prompt, process and didSample are forwarded through the seam in order")
    func promptProcessDidSampleAreForwarded() throws {
        try FocusedMLXTestSupport.withLock {
            let recorder = Recorder()
            var parameters = GenerateParameters(maxTokens: 4, temperature: 0)
            parameters.repetitionPenalty = 2.0
            parameters.repetitionContextSize = 8

            var iterator = try TokenIterator(
                input: makeInput([1, 2]),
                model: SeamTestModel(),
                parameters: parameters,
                additionalProcessor: RecordingProcessor(recorder: recorder, markerIndex: 5))

            // The initializer's own prefill must have primed the processor with
            // the full prompt, then produced exactly one sample.
            #expect(recorder.events.first == .prompt([1, 2]))

            let sampled = iterator.convertToToken(logits: MLXArray.zeros([1, 1, 8]))
            let sampledID = sampled.item(Int.self)

            let expected: [Recorder.Event] = [
                .prompt([1, 2]),
                .process([0, 0, 0, 0, 0, 0, 0, 0]),
                .didSample(5),
                .process([0, 0, 0, 0, 0, 0, 0, 0]),
                .didSample(sampledID),
            ]
            #expect(recorder.events == expected)
            #expect(sampledID == 5)
            }
    }

    // MARK: - 5. independentCopy isolates state

    @Test("independentCopy isolates both halves of the composite")
    func independentCopyIsolatesBothHalves() throws {
        try FocusedMLXTestSupport.withLock {
            let recorder = Recorder()
            var parameters = GenerateParameters(maxTokens: 4, temperature: 0)
            parameters.presencePenalty = 0.5  // unbounded mode: reference-typed counts

            let iterator = try TokenIterator(
                input: makeInput([1, 2]),
                model: SeamTestModel(),
                parameters: parameters,
                additionalProcessor: RecordingProcessor(recorder: recorder))

            // Init sampled one token: the ORIGINAL's built-in presence counts hold
            // it, and the ORIGINAL's additional box recorded it.
            let originalBox = try #require(iteratorAdditionalBox(of: iterator))
            let boxBefore = originalBox.samples
            let inherited = try #require(boxBefore.last, "init must have sampled through the seam")

            var copy = try #require(iterator.processor?.independentCopy())
            let eventsBeforeCopy = recorder.events.count
            _ = copy.process(logits: MLXArray.zeros([1, 8]))
            // The copy's process still forwards through its own additional half.
            #expect(recorder.events.count == eventsBeforeCopy + 1)
            for _ in 0 ..< 4 { copy.didSample(token: MLXArray([Int32(3)])) }

            // The additional half's reference state was replaced, not shared.
            #expect(originalBox.samples == boxBefore, "copy wrote into the original's box")

            // The copy's own built-in half advanced from its own samples ...
            let copyRow = row(of: copy.process(logits: MLXArray.zeros([1, 8])))
            #expect(copyRow[3] == -0.5, "the copy must accumulate its own presence penalty")
            // ... and inherited the pre-copy state rather than starting empty.
            #expect(copyRow[inherited] == -0.5, "the copy must inherit the original's counts")

            // The original's built-in half must NOT have seen the copy's samples.
            let original = try #require(iterator.processor)
            let originalRow = row(of: original.process(logits: MLXArray.zeros([1, 8])))
            #expect(originalRow[inherited] == -0.5, "the original keeps its own state")
            #expect(originalRow[3] == 0.0, "the copy's samples leaked into the original")
            }
    }

    private func row(of logits: MLXArray) -> [Float] {
        eval(logits)
        return logits.reshaped(-1).asArray(Float.self)
    }

    /// Reaches the test processor's reference box through the composite so
    /// the isolation assertion can look at the ORIGINAL additional half.
    private func iteratorAdditionalBox(of iterator: TokenIterator) -> MutableBox? {
        guard let composite = iterator.processor as? CompositeLogitProcessor,
            let recording = composite.additional as? RecordingProcessor
        else { return nil }
        return recording.box
    }
}
